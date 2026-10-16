//! Execute a prepared rank program eagerly or as a cached decode graph.

use std::cell::Cell;
use std::collections::BTreeMap;

use kernel_core::contract::{ArgumentList, LaunchPlan};
use runtime_bundle_abi::{
    BundleAbiError, BundleContextHandleV1, BundleCudaSymbolHandleV1, BundleEventHandleV1,
    BundleStatusV1, BundleStreamHandleV1, HostServicesV1,
};

use crate::dispatch::{self, BufferBindings, DispatchError};
use crate::geometry::BundleGeometry;
use crate::graphs::{GraphCache, GraphKey};
use crate::lanes::LaunchFacts;

pub struct PreparedLaunch {
    pub symbol: String,
    /// The token rows of its step this call processes, at least one: the
    /// rows its plan and arguments are sized for (a bucket's rows for a
    /// bucketed launch, the step's rows for one sized to them), or one for
    /// a launch per row. The bundles' journal tests check an exact verify
    /// step's launches against their entries' one-row claims at these rows.
    pub rows: u32,
    pub plan: LaunchPlan,
    pub arguments: ArgumentList,
    pub bindings: BufferBindings,
}

pub struct TracedLaunch {
    pub rank: u32,
    pub index: usize,
    pub symbol: String,
    pub event: BundleEventHandleV1,
}

/// Enqueue the regular program while retaining a completion event after each
/// launch, traced as `first` onwards. The caller waits only after every rank
/// has been queued, so a TP collective can make progress while the first
/// rank's events are inspected.
#[allow(clippy::too_many_arguments)]
pub fn launch_all_traced(
    services: &HostServicesV1,
    context: BundleContextHandleV1,
    stream: BundleStreamHandleV1,
    rank: u32,
    symbols: &BTreeMap<String, BundleCudaSymbolHandleV1>,
    launches: &[PreparedLaunch],
    first: usize,
    trace: &mut Vec<TracedLaunch>,
) -> Result<(), DispatchError> {
    for (index, launch) in (first..).zip(launches) {
        launch_all(services, stream, symbols, std::slice::from_ref(launch))?;
        let event = services.create_event(context)?;
        if let Err(error) = services.record_event(event, stream) {
            let _ = services.release_event(event);
            return Err(error.into());
        }
        trace.push(TracedLaunch {
            rank,
            index,
            symbol: launch.symbol.clone(),
            event,
        });
    }
    Ok(())
}

pub fn synchronize_trace(
    services: &HostServicesV1,
    trace: &mut Vec<TracedLaunch>,
) -> Result<(), BundleAbiError> {
    for launch in trace.iter() {
        services.log(
            format!(
                "trace rank {} launch {} {}",
                launch.rank, launch.index, launch.symbol
            )
            .as_bytes(),
        )?;
        services.synchronize_event(launch.event)?;
    }
    release_trace(services, trace)
}

pub fn release_trace(
    services: &HostServicesV1,
    trace: &mut Vec<TracedLaunch>,
) -> Result<(), BundleAbiError> {
    while let Some(launch) = trace.last() {
        services.release_event(launch.event)?;
        trace.pop();
    }
    Ok(())
}

/// Enqueues `launches` on `stream`. Every launch is resolved before the
/// first is submitted, so a host-side refusal queues nothing; only the device
/// can refuse a submission after an earlier one of the list.
pub fn launch_all(
    services: &HostServicesV1,
    stream: BundleStreamHandleV1,
    symbols: &BTreeMap<String, BundleCudaSymbolHandleV1>,
    launches: &[PreparedLaunch],
) -> Result<(), DispatchError> {
    if launches.is_empty() || launches.iter().any(|item| !item.plan.pdl) {
        return Err(DispatchError::InvalidPlan);
    }
    let failed = |launch: &PreparedLaunch, error: DispatchError| {
        let _ = services.log(format!("launch {} failed: {error:?}", launch.symbol).as_bytes());
        error
    };
    let resolved = launches
        .iter()
        .map(|launch| {
            dispatch::resolve(
                symbols,
                &launch.symbol,
                launch.plan,
                &launch.arguments,
                &launch.bindings,
            )
            .map_err(|error| failed(launch, error))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (launch, resolved) in launches.iter().zip(&resolved) {
        resolved
            .submit(services, stream)
            .map_err(|error| failed(launch, error))?;
    }
    Ok(())
}

/// Collective kernels pair every rank's calls and cannot be aborted. Once
/// device work reached one rank, a failure to submit its peers' share leaves
/// that rank waiting at a collective forever, and whatever is queued behind
/// it (a cancel's restore copies, the next step) never runs. Bundles submit
/// work to their ranks through `Unit`, which poisons the instance on such a
/// failure. A poisoned instance refuses every later call that would queue
/// device work or release resources, cancel and destroy included, and keeps
/// its resources for process-level recovery.
#[derive(Debug, Default)]
pub struct Collectives {
    poisoned: Cell<bool>,
}

impl Collectives {
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.get()
    }

    /// Poisons the instance when a bundle cannot establish that device work
    /// it submitted has finished with the buffers a later call reuses.
    pub fn poison(&self) {
        self.poisoned.set(true);
    }
}

/// Poisons `collectives` when a panic unwinds through a submission: the
/// unit's result then never reaches `Unit::record`, and ranks queued before
/// the panic may wait at a collective their peers never join.
pub struct PoisonOnUnwind<'a>(pub &'a Collectives);

impl Drop for PoisonOnUnwind<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.poisoned.set(true);
        }
    }
}

/// One unit of work (a stage, a program, a graph replay) submitted to the
/// ranks in turn.
#[derive(Debug, Default)]
pub struct Unit {
    started: bool,
}

impl Unit {
    /// Records one submission of this unit. A failure poisons `collectives`
    /// when an earlier submission of the unit succeeded, or when the device
    /// refused it (a list may fail after some of its launches). Host-side
    /// refusals of `launch_all` submit nothing.
    pub fn record<T>(
        &mut self,
        collectives: &Collectives,
        result: Result<T, DispatchError>,
    ) -> Result<T, DispatchError> {
        match result {
            Ok(value) => {
                self.started = true;
                Ok(value)
            }
            Err(error) => {
                if self.started || matches!(error, DispatchError::Host(_)) {
                    collectives.poisoned.set(true);
                }
                Err(error)
            }
        }
    }
}

/// A rank's lowered launches, with whether each is a collective that
/// waits for its peers (`collective`, one per launch, as its op states).
#[derive(Default)]
pub struct Program {
    pub launches: Vec<PreparedLaunch>,
    pub collective: Vec<bool>,
}

impl Program {
    /// One stage of a run that queues its stages one at a time, queued
    /// whole on each rank as staged runs queue every stage: whatever
    /// follows the stage's collectives is a bounded tail, never a program
    /// that could fill a stream. Its collectives are neither split nor
    /// checked.
    pub fn stage(launches: Vec<PreparedLaunch>) -> Self {
        Self {
            collective: vec![false; launches.len()],
            launches,
        }
    }

    /// The program of `lowered` launches, each with its facts.
    pub fn of(lowered: impl IntoIterator<Item = (PreparedLaunch, LaunchFacts)>) -> Self {
        let (launches, facts): (Vec<_>, Vec<LaunchFacts>) = lowered.into_iter().unzip();
        Self {
            launches,
            collective: facts.iter().map(|facts| facts.collective).collect(),
        }
    }
}

/// Runs one program on every rank as one `Unit`. Eagerly (`graph` None),
/// every rank is lowered before any is submitted, the ranks' collectives
/// are checked to match (`check_collectives`), and the programs are queued
/// collective by collective across the ranks (`launch_segments`). With a
/// graph key, every rank's missing graphs are captured before any rank
/// launches, then all ranks replay piece-major. `prepare(rank)` lowers a
/// rank's program; `submitted(rank)` runs once a rank's first segment
/// (eager) or first piece (graph) is queued. Empty eager programs are
/// skipped.
///
/// `streams` and `symbols` hold one entry per rank of `graphs`; any other
/// count, or a poisoned `collectives`, is refused before anything is lowered
/// or launched. A panic once submission began poisons `collectives`.
#[allow(clippy::too_many_arguments)]
pub fn run_ranks<G: BundleGeometry>(
    services: &HostServicesV1,
    streams: &[BundleStreamHandleV1],
    symbols: &[BTreeMap<String, BundleCudaSymbolHandleV1>],
    graph: Option<(u32, u32)>,
    graphs: &mut GraphCache<G>,
    collectives: &Collectives,
    mut prepare: impl FnMut(usize) -> Result<Program, DispatchError>,
    submitted: impl FnMut(usize),
) -> Result<(), DispatchError> {
    if collectives.is_poisoned() {
        return Err(BundleAbiError::Status(BundleStatusV1::InvalidState).into());
    }
    if streams.len() != graphs.ranks().count() || symbols.len() != streams.len() {
        return Err(DispatchError::InvalidPlan);
    }
    let mut unit = Unit::default();
    let Some((bucket, items)) = graph else {
        let (programs, waits): (Vec<_>, Vec<_>) = (0..streams.len())
            .map(|rank| prepare(rank).map(|program| (program.launches, program.collective)))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();
        if let Err(error) = check_collectives(&programs, &waits) {
            let _ = services.log(error.as_bytes());
            return Err(DispatchError::InvalidPlan);
        }
        let segments = programs
            .iter()
            .zip(&waits)
            .map(|(program, waits)| collective_segments(program, waits))
            .collect::<Vec<_>>();
        let _unwinding = PoisonOnUnwind(collectives);
        return launch_segments(
            collectives,
            &mut unit,
            &mut vec![false; streams.len()],
            &segments,
            |rank, launches| launch_all(services, streams[rank], &symbols[rank], launches),
            submitted,
            |_| Ok(()),
        );
    };
    for (rank, &stream) in streams.iter().enumerate() {
        let key = GraphKey {
            rank: rank as u32,
            bucket,
            items,
        };
        if graphs.contains(key) {
            continue;
        }
        let launches = prepare(rank)?.launches;
        graphs.capture(services, stream, key, launches.len(), |range| {
            launch_all(services, stream, &symbols[rank], &launches[range])
        })?;
    }
    let _unwinding = PoisonOnUnwind(collectives);
    let replayed = unit.record(
        collectives,
        graphs
            .replay_piece_major(services, streams, (bucket, items), submitted)
            .map_err(DispatchError::from),
    )?;
    if replayed {
        Ok(())
    } else {
        Err(DispatchError::InvalidPlan)
    }
}

/// A rank's program in segments that each end at a collective
/// (`collective`, one per launch, as the launch's op states), and its
/// launches after the last one.
pub fn collective_segments<'a>(
    program: &'a [PreparedLaunch],
    collective: &[bool],
) -> Vec<&'a [PreparedLaunch]> {
    let mut segments = Vec::new();
    let mut start = 0;
    for (end, _) in collective.iter().enumerate().filter(|(_, waits)| **waits) {
        segments.push(&program[start..=end]);
        start = end + 1;
    }
    if start < program.len() {
        segments.push(&program[start..]);
    }
    segments
}

/// One collective's rank-independent signature: its symbol, rows and
/// launch plan, then per argument its kind with the byte length of each
/// buffer it names (the group size for a peer array) and its 64-bit
/// scalars (element counts). 32-bit scalars, which carry rank IDs, and
/// the buffers' addresses may differ by rank.
#[derive(Debug, Eq, PartialEq)]
struct Signature {
    symbol: String,
    rows: u32,
    plan: LaunchPlan,
    arguments: Vec<SignedArgument>,
}

#[derive(Debug, Eq, PartialEq)]
enum SignedArgument {
    Buffer(u64),
    Peers(Vec<u64>),
    U64(u64),
    U32,
    I32,
    Null,
    TensorMap,
}

fn signature(launch: &PreparedLaunch) -> Result<Signature, String> {
    use kernel_core::contract::Argument;
    let length = |id: u64| {
        launch
            .bindings
            .get(id)
            .map(|span| span.length)
            .map_err(|_| format!("{} names an unbound buffer", launch.symbol))
    };
    let arguments = launch
        .arguments
        .0
        .iter()
        .map(|argument| {
            Ok(match argument {
                Argument::DeviceBuffer(id) => SignedArgument::Buffer(length(*id)?),
                Argument::DevicePointerArray(ids) => SignedArgument::Peers(
                    ids.iter()
                        .map(|&id| length(id))
                        .collect::<Result<_, String>>()?,
                ),
                Argument::U64(value) => SignedArgument::U64(*value),
                Argument::U32(_) => SignedArgument::U32,
                Argument::I32(_) => SignedArgument::I32,
                Argument::NullDevicePointer => SignedArgument::Null,
                Argument::TensorMap(_) => SignedArgument::TensorMap,
            })
        })
        .collect::<Result<_, String>>()?;
    Ok(Signature {
        symbol: launch.symbol.clone(),
        rows: launch.rows,
        plan: launch.plan,
        arguments,
    })
}

/// Checks that every rank's program issues the same collectives
/// (`collectives`, one per launch) in the same order, by their signatures
/// (`signature`), so that each one meets its peers' at the same protocol
/// and shape. Local launches between them may differ by rank. 32-bit
/// scalars are not compared, since they carry rank IDs, and so neither
/// are rank-independent ones such as an owner gather's target: the check
/// catches a different sequence or entry, not a different parameter.
pub fn check_collectives(
    programs: &[impl AsRef<[PreparedLaunch]>],
    collectives: &[impl AsRef<[bool]>],
) -> Result<(), String> {
    if programs.len() != collectives.len()
        || programs
            .iter()
            .zip(collectives)
            .any(|(program, waits)| program.as_ref().len() != waits.as_ref().len())
    {
        return Err("a program's collectives do not follow its launches".into());
    }
    let signatures = programs
        .iter()
        .zip(collectives)
        .map(|(program, waits)| {
            program
                .as_ref()
                .iter()
                .zip(waits.as_ref())
                .filter(|(_, waits)| **waits)
                .map(|(launch, _)| signature(launch))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let Some(first) = signatures.first() else {
        return Ok(());
    };
    for (rank, own) in signatures.iter().enumerate().skip(1) {
        if own.len() != first.len() {
            return Err(format!(
                "rank {rank}'s eager program issues {} collectives, rank 0's {}",
                own.len(),
                first.len()
            ));
        }
        if let Some(index) = (0..own.len()).find(|&index| own[index] != first[index]) {
            return Err(format!(
                "rank {rank}'s eager collective {index} differs from rank 0's: {:?} against {:?}",
                own[index], first[index]
            ));
        }
    }
    Ok(())
}

/// Queues every rank's `segments` (`collective_segments`) round by round
/// through `launch(rank, launches)`: each rank's next segment, rank after
/// rank, then the next round. A stream holds a bounded number of pending
/// launches, and a rank spins at a collective until every peer reaches it;
/// queueing one rank's whole program first could leave the host blocked in
/// a launch call on that rank's full stream while it spins at its first
/// collective, before any peer has work. Each submission is recorded in
/// `unit`; `submitted(rank)` runs once a rank's first segment is queued
/// (`started` marks the ranks that have one), and `round(segment)` after
/// every rank queued a round.
pub fn launch_segments(
    collectives: &Collectives,
    unit: &mut Unit,
    started: &mut [bool],
    segments: &[Vec<&[PreparedLaunch]>],
    mut launch: impl FnMut(usize, &[PreparedLaunch]) -> Result<(), DispatchError>,
    mut submitted: impl FnMut(usize),
    mut round: impl FnMut(usize) -> Result<(), DispatchError>,
) -> Result<(), DispatchError> {
    let depth = segments.iter().map(Vec::len).max().unwrap_or(0);
    for segment in 0..depth {
        for (rank, launches) in segments.iter().enumerate() {
            let Some(launches) = launches.get(segment) else {
                continue;
            };
            unit.record(collectives, launch(rank, launches))?;
            if !std::mem::replace(&mut started[rank], true) {
                submitted(rank);
            }
        }
        round(segment)?;
    }
    Ok(())
}

pub fn run_rank<G: BundleGeometry>(
    services: &HostServicesV1,
    stream: BundleStreamHandleV1,
    rank: u32,
    graph: Option<(u32, u32)>,
    symbols: &BTreeMap<String, BundleCudaSymbolHandleV1>,
    graphs: &mut GraphCache<G>,
    launches: &[PreparedLaunch],
) -> Result<(), DispatchError> {
    if let Some((bucket, items)) = graph {
        graphs.run(
            services,
            stream,
            GraphKey {
                rank,
                bucket,
                items,
            },
            launches.len(),
            |range| launch_all(services, stream, symbols, &launches[range]),
        )
    } else {
        launch_all(services, stream, symbols, launches)
    }
}

#[cfg(test)]
mod tests {
    use kernel_core::contract::{Argument, ArgumentList, LaunchPlan};
    use runtime_bundle_abi::{
        BundleAbiError, BundleDeviceBufferHandleV1, BundleStatusV1, DeviceBufferSpanV1,
    };

    use super::{Collectives, PreparedLaunch, Unit, check_collectives, collective_segments};
    use crate::dispatch::{BufferBindings, DispatchError};

    fn device_error() -> DispatchError {
        DispatchError::Host(BundleAbiError::Status(BundleStatusV1::HostError))
    }

    #[test]
    fn only_a_submission_that_may_leave_peers_waiting_poisons() {
        // A host-side refusal before any submission of the unit is balanced.
        let collectives = Collectives::default();
        let mut unit = Unit::default();
        assert!(
            unit.record(&collectives, Err::<(), _>(DispatchError::InvalidPlan))
                .is_err()
        );
        assert!(!collectives.is_poisoned());
        // So is a unit that reached every rank.
        let mut unit = Unit::default();
        for _ in 0..4 {
            unit.record(&collectives, Ok(())).unwrap();
        }
        assert!(!collectives.is_poisoned());
        // A refusal after an earlier rank received the unit is not.
        let mut unit = Unit::default();
        unit.record(&collectives, Ok(())).unwrap();
        assert!(
            unit.record(&collectives, Err::<(), _>(DispatchError::InvalidPlan))
                .is_err()
        );
        assert!(collectives.is_poisoned());
        // Nor is a device refusal: earlier launches of its list may be queued.
        let collectives = Collectives::default();
        assert!(
            Unit::default()
                .record(&collectives, Err::<(), _>(device_error()))
                .is_err()
        );
        assert!(collectives.is_poisoned());
    }

    /// A launch of `symbol` over `grid` CTAs on `rank`: its own buffer of
    /// `bytes` (in a rank-specific allocation), one per peer, its element
    /// count and its rank.
    fn launch(symbol: &str, grid: u32, bytes: u64, rank: u32) -> PreparedLaunch {
        let span = |slot: u32| {
            DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(1, slot), 0, bytes)
        };
        let mut bindings = BufferBindings::default();
        let own = bindings.bind(span(100 + rank));
        let peers = (0..4).map(|peer| bindings.bind(span(200 + peer))).collect();
        PreparedLaunch {
            symbol: symbol.into(),
            rows: 1,
            plan: LaunchPlan {
                grid: [grid, 1, 1],
                block: [128, 1, 1],
                dynamic_shared_bytes: 0,
                workspace_bytes: 0,
                cluster: [1, 1, 1],
                pdl: true,
            },
            arguments: ArgumentList(vec![
                Argument::DeviceBuffer(own),
                Argument::DevicePointerArray(peers),
                Argument::U64(bytes / 4),
                Argument::U32(rank),
                Argument::U32(4),
            ]),
            bindings,
        }
    }

    const ONE_SHOT: &str = "chop_comm_allreduce_oneshot_f32_v1";
    const SCATTER: &str = "chop_comm_reduce_scatter_f32_v1";

    /// Each rank's program: a local launch, `collective(rank)`, a local
    /// launch and a one-shot reduction.
    fn programs(collective: impl Fn(u32) -> PreparedLaunch) -> Vec<Vec<PreparedLaunch>> {
        (0..4)
            .map(|rank| {
                vec![
                    launch("chop_row_local", 4, 64, rank),
                    collective(rank),
                    launch("chop_row_local", 4, 64, rank),
                    launch(ONE_SHOT, 8, 1024, rank),
                ]
            })
            .collect()
    }

    /// Whether each launch of `programs` waits for its peers, as its op
    /// would state: the fixtures' all-reduce launches wait, their local
    /// launches do not.
    fn waits(programs: &[Vec<PreparedLaunch>]) -> Vec<Vec<bool>> {
        programs
            .iter()
            .map(|program| {
                program
                    .iter()
                    .map(|launch| [ONE_SHOT, SCATTER].contains(&launch.symbol.as_str()))
                    .collect()
            })
            .collect()
    }

    /// A program splits after each launch whose op states that it waits
    /// for its peers, whatever its symbol.
    #[test]
    fn collectives_follow_their_ops_facts() {
        let comm = launch(ONE_SHOT, 8, 1024, 0);
        let local = launch("chop_row_local", 4, 64, 0);
        let program = vec![
            launch("chop_row_local", 4, 64, 0),
            comm,
            local,
            launch("chop_row_local", 4, 64, 0),
        ];
        let segments = collective_segments(&program, &[false, true, false, true]);
        assert_eq!(
            segments
                .iter()
                .map(|segment| segment.len())
                .collect::<Vec<_>>(),
            [2, 2]
        );
        let segments = collective_segments(&program, &[false, true, false, false]);
        assert_eq!(
            segments
                .iter()
                .map(|segment| segment.len())
                .collect::<Vec<_>>(),
            [2, 2]
        );
        assert!(collective_segments(&[], &[]).is_empty());
        // Facts that do not follow the launches are refused.
        assert!(check_collectives(&[program], &[vec![true]]).is_err());
    }

    /// Rank-specific buffers, rank scalars and local work between the
    /// collectives pass; a different collective, launch geometry, buffer
    /// size or group size at the same place is refused, though the counts
    /// match.
    #[test]
    fn eager_programs_must_issue_the_same_ordered_collectives() {
        let same = programs(|rank| launch(ONE_SHOT, 8, 1024, rank));
        assert_eq!(check_collectives(&same, &waits(&same)), Ok(()));
        let mut local = same;
        local[1].insert(0, launch("chop_row_other", 2, 32, 1));
        local[2].push(launch("chop_row_tail", 1, 16, 2));
        assert_eq!(check_collectives(&local, &waits(&local)), Ok(()));
        let differ = |change: &dyn Fn(u32) -> PreparedLaunch| {
            programs(|rank| match rank {
                2 => change(rank),
                _ => launch(ONE_SHOT, 8, 1024, rank),
            })
        };
        for (name, programs) in [
            ("symbol", differ(&|rank| launch(SCATTER, 8, 1024, rank))),
            ("geometry", differ(&|rank| launch(ONE_SHOT, 16, 1024, rank))),
            (
                "buffer size",
                differ(&|rank| launch(ONE_SHOT, 8, 2048, rank)),
            ),
        ] {
            let refused = check_collectives(&programs, &waits(&programs)).expect_err(name);
            assert!(
                refused.contains("rank 2") && refused.contains("collective 0"),
                "{refused}"
            );
        }
        let mut group = programs(|rank| launch(ONE_SHOT, 8, 1024, rank));
        if let Argument::DevicePointerArray(peers) = &mut group[3][1].arguments.0[1] {
            peers.pop();
        }
        let refused = check_collectives(&group, &waits(&group)).expect_err("group");
        assert!(refused.contains("rank 3"), "{refused}");
    }
}
