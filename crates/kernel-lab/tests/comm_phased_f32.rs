//! The phased F32 all-reduce (`kernels-sm120/cuda/comm/allreduce_phased_f32.cuh`,
//! lock family `comm_phased_f32`), two-shot and one-shot entries, on two or
//! more GPUs (`CHOPINFER_KERNEL_GPUS`):
//!
//! - `phased_f32_matches_pair_and_host`: at 1 to 8192 rows of 4096 and
//!   several sender and worker counts (no workers included), every rank's
//!   F32 output equals the host's rank-ordered F32 sum from zero and the
//!   current collectives' (the one-shot `chop_comm_allreduce_oneshot_f32_v1`
//!   up to 32 rows, and the phased pair `chop_comm_reduce_scatter_f32_v1`
//!   then `chop_comm_allgather_scattered_f32_v1` at every count), bit for
//!   bit; the round entries' BF16 output equals the host sum rounded to
//!   nearest even at every live row count and nothing past the live rows is
//!   written. Every call runs on one state, so epochs and one-shot parities
//!   follow each other across row counts, geometries, protocols and entries.
//! - `phased_f32_senders_only_every_count`: the same at every row count
//!   from 1 to 32 in the senders-only geometry of 16 CTAs
//!   ([`SENDERS_ONLY`]), every entry, with and without live rows (every
//!   live count from 0 to the rows), on one state: the geometry a table of
//!   small row counts serves, where the timing found it fastest.
//! - `phased_f32_replays`: poisoned outputs, eager, relaunched, from a
//!   graph (including one graph that alternates protocols) and behind a PDL
//!   producer that zeroes the partials and the live count until after its
//!   tail, at the default geometries and at [`SENDERS_ONLY`], at full,
//!   partial and zero live counts.
//! - `phased_f32_mutants_are_caught`: each lab mutant (`MUTANTS`) changes
//!   some rank's output in an eager call and in a graph replay, two-shot at
//!   2048 rows and one-shot at 32 rows with workers, and in
//!   [`SENDERS_ONLY`] one-shot at 4 rows and two-shot at 32; other row
//!   counts and senders-only geometries are recorded. A mutant counts only
//!   on an observed output difference in one of up to five fresh runs per mode;
//!   an execution error fails the test.
//! - `phased_f32_generations`: eight (two-shot: four) back-to-back calls on
//!   one state, each with distinct partials, mixing one-shot and two-shot
//!   entries, a different peer delayed before each call, every result
//!   snapshotted on the device right after its call; eager and replayed.
//!   One [`SENDERS_ONLY`] case changes the row count and the live count
//!   (full, partial, zero) from call to call, as a server's buckets do.
//!   The served entries show no wrong word; the shared-parity mutant (both
//!   one-shot parities in one inbox region) must show observed wrong words.
//!   The reversed-order mutant runs only at three or more ranks: a two-rank
//!   sum rounds alike in either order.
//! - `phased_f32_tag_reuse`: flags tagged `0x80000000 | epoch` repeat once
//!   the epoch moved 2^31 calls past a call whose flags remain; with late
//!   peers the repeat sums old partials, and zeroing every rank's flags and
//!   epoch while every rank is idle (what a server does before tags can
//!   repeat) makes the call exact again. The reset arm must be exact, the
//!   unreset arm must show wrong words.
//! - `phased_f32_timing` (`CHOPINFER_KERNEL_BENCH=1`): microseconds per call
//!   of the current one-shot (up to 32 rows), the pair and all four entries
//!   per sender and worker count, in graphs of repeated calls on every rank,
//!   critical path over ranks. The low-latency entry of PF6 is not on this
//!   branch; compare with its own lab log.

use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::{Argument, ArgumentList, CommQuery, DType, LaunchPlan, Validate};
use kernel_lab::launch::{self, Bindings, CollectiveModules, RankCall};
use kernel_lab::poison::{self, Fill, POISON, WriteMap};
use kernel_lab::topology::RankReplay;
use kernel_lab::{LabResult, Session, gpu_test_multi, time_multi_graph};
use kernels_sm120::comm::{self as comm, phased_f32};
use runtime_cuda::{DeviceBuffer, KernelArgument, LaunchConfig};
use serde_json::json;
use std::collections::BTreeMap;

const WIDTH: usize = 4096;
const BITWISE_ROWS: [usize; 14] = [1, 2, 4, 8, 16, 24, 32, 33, 64, 256, 1024, 2048, 4096, 8192];
/// (senders, workers) of the bitwise sweep up to 64 rows, and above.
const SMALL_GEOMETRIES: [(u32, u32); 5] = [(1, 0), (4, 0), (1, 1), (4, 4), (8, 8)];
const LARGE_GEOMETRIES: [(u32, u32); 5] = [(1, 1), (8, 8), (16, 32), (64, 64), (32, 0)];
/// Sixteen sending CTAs and no workers: the timing's fastest geometry of
/// the one-shot entries at 2 and 4 rows and of the two-shot round entry
/// from 8 to 32 rows, which `phased_f32_senders_only_every_count` checks at
/// every count up to [`SENDERS_ONLY_MAX_ROWS`].
const SENDERS_ONLY: (u32, u32) = (16, 0);
const SENDERS_ONLY_MAX_ROWS: usize = 32;
/// Rows the lab runs the current one-shot at (its served range and PF6's
/// timed range).
const CURRENT_ONE_SHOT_MAX_ROWS: usize = 32;
/// Rows at which every mutant must be caught in both modes, per protocol.
const TWO_SHOT_MUTANT_ROWS: [usize; 2] = [32, 2048];
const ONE_SHOT_MUTANT_ROWS: [usize; 2] = [4, 32];
const REQUIRED_TWO_SHOT_ROWS: usize = 2048;
const REQUIRED_ONE_SHOT_ROWS: usize = 32;
/// Runs per mode before a mutant counts as missed.
const MUTANT_TRIALS: usize = 5;
/// The PDL producer's tail before it restores the partials and live count.
const PRODUCER_TAIL_NS: u64 = 50_000;

// Buffer handles: one per rank-local buffer, a range per peer array.
const INPUT: u64 = 1;
const OUTPUT: u64 = 2;
const ROUNDED: u64 = 3;
const EPOCH: u64 = 4;
const LIVE: u64 = 5;
const PAIR_OUTPUT: u64 = 6;
const PAIR_EPOCH: u64 = 7;
const CUR_EPOCH: u64 = 8;
const INBOXES: u64 = 100;
const FLAGS: u64 = 200;
const OUTPUTS: u64 = 300;
const ROUNDEDS: u64 = 400;
const PAIR_INBOXES: u64 = 500;
const PAIR_FLAGS: u64 = 600;
const PAIR_OUTPUTS: u64 = 700;
const CUR_INBOXES: u64 = 800;
const CUR_FLAGS: u64 = 900;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Partials whose exponents spread over 24 binades, so the order of the
/// rank sum changes bits; every 4099th element is -0 on every rank.
fn partial(seed: u64, rank: usize, elements: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            if index % 4099 == 0 {
                return -0.0;
            }
            let h = splitmix(seed << 48 ^ (rank as u64) << 40 ^ index as u64);
            let mantissa = (h >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            let exponent = ((h >> 8) % 24) as i32 - 12;
            mantissa * 2f32.powi(exponent)
        })
        .collect()
}

/// The rank-ordered F32 sum from zero of the first `count` elements.
fn host_sums(inputs: &[Vec<f32>], count: usize) -> Vec<f32> {
    (0..count)
        .map(|index| {
            let mut sum = 0.0_f32;
            for input in inputs {
                sum += input[index];
            }
            sum
        })
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn first_difference<T: PartialEq>(actual: &[T], expected: &[T]) -> Option<usize> {
    actual
        .iter()
        .zip(expected)
        .position(|(a, e)| a != e)
        .or((actual.len() != expected.len()).then_some(actual.len().min(expected.len())))
}

/// One rank's persistent state: the entries' and the pair's, sized for
/// `capacity` rows, and the live count.
struct State {
    inbox: DeviceBuffer,
    flags: DeviceBuffer,
    epoch: DeviceBuffer,
    live: DeviceBuffer,
    pair_inbox: DeviceBuffer,
    pair_flags: DeviceBuffer,
    pair_epoch: DeviceBuffer,
}

/// One rank's buffers of one row count, with the current one-shot's state
/// of that count (it indexes flags and epochs by tile) up to 32 rows.
struct Rows {
    input: DeviceBuffer,
    staged: DeviceBuffer,
    output: DeviceBuffer,
    rounded: DeviceBuffer,
    pair_output: DeviceBuffer,
    current: Option<[DeviceBuffer; 3]>,
}

fn states(sessions: &[Session], capacity: usize) -> LabResult<Vec<State>> {
    let ranks = sessions.len() as u32;
    let elements = (capacity * WIDTH) as u64;
    let own = phased_f32::buffers(ranks, elements).map_err(err)?;
    let pair = comm::phased_buffers(ranks, elements, DType::F32).map_err(err)?;
    let states = sessions
        .iter()
        .map(|session| {
            Ok(State {
                inbox: session.allocate(own.inbox_bytes as usize)?,
                flags: session.allocate(own.flag_bytes as usize)?,
                epoch: session.allocate(own.epoch_bytes as usize)?,
                live: session.allocate(4)?,
                pair_inbox: session.allocate(pair.inbox_bytes as usize)?,
                pair_flags: session.allocate(pair.flag_bytes as usize)?,
                pair_epoch: session.allocate(pair.epoch_bytes as usize)?,
            })
        })
        .collect::<LabResult<Vec<_>>>()?;
    reset(sessions, &states)?;
    Ok(states)
}

fn zero(session: &Session, buffer: &DeviceBuffer) -> LabResult<()> {
    buffer
        .copy_from_bytes(&vec![0; buffer.byte_len()], &session.stream)
        .map_err(err)
}

/// Zeroes every rank's flags and epochs once every rank is idle.
fn reset(sessions: &[Session], states: &[State]) -> LabResult<()> {
    poison::synchronize(sessions)?;
    for (session, state) in sessions.iter().zip(states) {
        for buffer in [
            &state.flags,
            &state.epoch,
            &state.pair_flags,
            &state.pair_epoch,
        ] {
            zero(session, buffer)?;
        }
    }
    Ok(())
}

fn set_live(sessions: &[Session], states: &[State], live: usize) -> LabResult<()> {
    poison::synchronize(sessions)?;
    for (session, state) in sessions.iter().zip(states) {
        state
            .live
            .copy_from_bytes(&(live as u32).to_ne_bytes(), &session.stream)
            .map_err(err)?;
    }
    Ok(())
}

fn rows_buffers(sessions: &[Session], inputs: &[Vec<f32>]) -> LabResult<Vec<Rows>> {
    let ranks = sessions.len();
    sessions
        .iter()
        .zip(inputs)
        .map(|(session, input)| {
            let elements = input.len();
            let current = if elements <= CURRENT_ONE_SHOT_MAX_ROWS * WIDTH {
                let tiles = elements.div_ceil(4096);
                let buffers = [
                    session.allocate(ranks * elements * 4)?,
                    session.allocate(2 * ranks * tiles * 4)?,
                    session.allocate(tiles * 4)?,
                ];
                zero(session, &buffers[1])?;
                zero(session, &buffers[2])?;
                Some(buffers)
            } else {
                None
            };
            Ok(Rows {
                input: session.upload_f32(input)?,
                staged: session.upload_f32(input)?,
                output: session.allocate(elements * 4)?,
                rounded: session.allocate(elements * 2)?,
                pair_output: session.allocate(elements * 4)?,
                current,
            })
        })
        .collect()
}

fn bindings<'a>(states: &'a [State], rows: &'a [Rows], rank: usize) -> Bindings<'a> {
    let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
    buffers.insert(INPUT, &rows[rank].input);
    buffers.insert(OUTPUT, &rows[rank].output);
    buffers.insert(ROUNDED, &rows[rank].rounded);
    buffers.insert(EPOCH, &states[rank].epoch);
    buffers.insert(LIVE, &states[rank].live);
    buffers.insert(PAIR_OUTPUT, &rows[rank].pair_output);
    buffers.insert(PAIR_EPOCH, &states[rank].pair_epoch);
    if let Some(current) = &rows[rank].current {
        buffers.insert(CUR_EPOCH, &current[2]);
    }
    for peer in 0..states.len() {
        let q = peer as u64;
        buffers.insert(INBOXES + q, &states[peer].inbox);
        buffers.insert(FLAGS + q, &states[peer].flags);
        buffers.insert(OUTPUTS + q, &rows[peer].output);
        buffers.insert(ROUNDEDS + q, &rows[peer].rounded);
        buffers.insert(PAIR_INBOXES + q, &states[peer].pair_inbox);
        buffers.insert(PAIR_FLAGS + q, &states[peer].pair_flags);
        buffers.insert(PAIR_OUTPUTS + q, &rows[peer].pair_output);
        if let Some(current) = &rows[peer].current {
            buffers.insert(CUR_INBOXES + q, &current[0]);
            buffers.insert(CUR_FLAGS + q, &current[1]);
        }
    }
    Bindings { buffers }
}

/// Which entry a call runs.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Entry {
    F32,
    Round,
    OneShot,
    OneShotRound,
}

const ENTRIES: [Entry; 4] = [
    Entry::F32,
    Entry::Round,
    Entry::OneShot,
    Entry::OneShotRound,
];

impl Entry {
    fn symbol(self) -> &'static str {
        match self {
            Entry::F32 => phased_f32::SYMBOL,
            Entry::Round => phased_f32::ROUND_SYMBOL,
            Entry::OneShot => phased_f32::ONE_SHOT_SYMBOL,
            Entry::OneShotRound => phased_f32::ONE_SHOT_ROUND_SYMBOL,
        }
    }

    fn round(self) -> bool {
        matches!(self, Entry::Round | Entry::OneShotRound)
    }

    fn one_shot(self) -> bool {
        matches!(self, Entry::OneShot | Entry::OneShotRound)
    }

    fn takes(self, rows: usize) -> bool {
        !self.one_shot() || (rows * WIDTH) as u64 <= phased_f32::ONE_SHOT_MAX_ELEMENTS
    }

    fn name(self) -> &'static str {
        match self {
            Entry::F32 => "f32",
            Entry::Round => "round",
            Entry::OneShot => "oneshot_f32",
            Entry::OneShotRound => "oneshot_round",
        }
    }
}

/// The sender and worker counts the replays and live sweeps use.
fn default_geometry(rows: usize) -> (u32, u32) {
    if rows <= 64 { (4, 0) } else { (16, 32) }
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    rows: usize,
    live: bool,
    senders: u32,
    workers: u32,
}

impl Shape {
    fn new(rows: usize, live: bool, (senders, workers): (u32, u32)) -> Self {
        Self {
            rows,
            live,
            senders,
            workers,
        }
    }
}

fn entry_args(rank: usize, ranks: usize, entry: Entry, shape: Shape) -> LabResult<ArgumentList> {
    let peers = |base: u64| (0..ranks as u64).map(|q| base + q).collect::<Vec<_>>();
    let (output, outputs) = if entry.round() {
        (ROUNDED, ROUNDEDS)
    } else {
        (OUTPUT, OUTPUTS)
    };
    phased_f32::arguments(
        entry.symbol(),
        phased_f32::Call {
            input: INPUT,
            output,
            inboxes: peers(INBOXES),
            outputs: peers(outputs),
            flags: peers(FLAGS),
            epoch: EPOCH,
            elements: (shape.rows * WIDTH) as u64,
            row_elements: WIDTH as u32,
            live_rows: shape.live.then_some(LIVE),
            rank: rank as u32,
            ranks: ranks as u32,
            senders: shape.senders,
        },
    )
    .map_err(err)
}

/// One call of `symbol` (`entry`'s symbol in `modules`, served or mutant)
/// on every rank, or on `only_rank`.
#[allow(clippy::too_many_arguments)]
fn launch_entry(
    sessions: &[Session],
    modules: &CollectiveModules,
    symbol: &str,
    states: &[State],
    rows: &[Rows],
    entry: Entry,
    shape: Shape,
    only_rank: Option<usize>,
) -> LabResult<()> {
    launch_bound(sessions, modules, symbol, entry, shape, only_rank, |rank| {
        bindings(states, rows, rank)
    })
}

/// [`launch_entry`] with the handles bound by `bind(rank)`.
fn launch_bound<'a>(
    sessions: &[Session],
    modules: &CollectiveModules,
    symbol: &str,
    entry: Entry,
    shape: Shape,
    only_rank: Option<usize>,
    bind: impl Fn(usize) -> Bindings<'a>,
) -> LabResult<()> {
    let plan = phased_f32::plan_with(shape.senders, shape.workers).map_err(err)?;
    launch::launch_collective(sessions, modules, symbol, plan, only_rank, |rank| {
        Ok(RankCall {
            args: entry_args(rank, sessions.len(), entry, shape)?,
            bindings: bind(rank),
        })
    })
}

fn query(ranks: usize, rows: usize) -> CommQuery {
    CommQuery::AllReduce {
        ranks: ranks as u32,
        elements: (rows * WIDTH) as u64,
        dtype: DType::F32,
        fused_residual: false,
        fused_rmsnorm: None,
    }
}

/// The comm planner's protocol for `rows` rows: the one the bundle serves.
fn served_variant(ranks: usize, rows: usize) -> LabResult<comm::Variant> {
    comm::choose(&query(ranks, rows)).map_err(err)
}

/// The pair's plan at any row count (`comm::plan`'s two-shot plan: one
/// sending cluster and up to seven worker clusters).
fn pair_plan(rows: usize) -> LabResult<LaunchPlan> {
    let cluster = 8;
    let tiles = (rows * WIDTH).div_ceil(4096) as u32;
    let plan = LaunchPlan {
        grid: [cluster * (1 + tiles.div_ceil(cluster).clamp(1, 7)), 1, 1],
        block: [256, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [cluster, 1, 1],
        pdl: true,
    };
    plan.validate().map_err(err)?;
    Ok(plan)
}

/// The phased pair into the pair output, on every rank or `only_rank`.
fn launch_pair(
    sessions: &[Session],
    modules: &CollectiveModules,
    states: &[State],
    rows: &[Rows],
    count: usize,
    only_rank: Option<usize>,
) -> LabResult<()> {
    let ranks = sessions.len();
    let elements = (count * WIDTH) as u64;
    let plan = pair_plan(count)?;
    let peers = |base: u64| (0..ranks as u64).map(|q| base + q).collect::<Vec<_>>();
    launch::launch_collective(
        sessions,
        modules,
        "chop_comm_reduce_scatter_f32_v1",
        plan,
        only_rank,
        |rank| {
            Ok(RankCall {
                args: comm::arguments(comm::OneShotCall {
                    input: INPUT,
                    output: PAIR_OUTPUT,
                    peer_inboxes: peers(PAIR_INBOXES),
                    peer_flags: peers(PAIR_FLAGS),
                    epoch: PAIR_EPOCH,
                    elements,
                    rank: rank as u32,
                    ranks: ranks as u32,
                })
                .map_err(err)?,
                bindings: bindings(states, rows, rank),
            })
        },
    )?;
    launch::launch_collective(
        sessions,
        modules,
        "chop_comm_allgather_scattered_f32_v1",
        plan,
        only_rank,
        |rank| {
            Ok(RankCall {
                args: ArgumentList(vec![
                    Argument::DeviceBuffer(PAIR_OUTPUT),
                    Argument::DevicePointerArray(peers(PAIR_OUTPUTS)),
                    Argument::DevicePointerArray(peers(PAIR_FLAGS)),
                    Argument::DeviceBuffer(PAIR_EPOCH),
                    Argument::U64(elements),
                    Argument::U32(rank as u32),
                    Argument::U32(ranks as u32),
                ]),
                bindings: bindings(states, rows, rank),
            })
        },
    )
}

/// The current one-shot (`chop_comm_allreduce_oneshot_f32_v1`, one CTA per
/// tile, every rank writing every peer) into the pair output, on its own
/// state of this row count.
fn launch_current_one_shot(
    sessions: &[Session],
    modules: &CollectiveModules,
    states: &[State],
    rows: &[Rows],
    count: usize,
    only_rank: Option<usize>,
) -> LabResult<()> {
    let ranks = sessions.len();
    let elements = (count * WIDTH) as u64;
    let plan = LaunchPlan {
        grid: [elements.div_ceil(4096) as u32, 1, 1],
        block: [256, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    };
    let peers = |base: u64| (0..ranks as u64).map(|q| base + q).collect::<Vec<_>>();
    launch::launch_collective(
        sessions,
        modules,
        "chop_comm_allreduce_oneshot_f32_v1",
        plan,
        only_rank,
        |rank| {
            Ok(RankCall {
                args: comm::arguments(comm::OneShotCall {
                    input: INPUT,
                    output: PAIR_OUTPUT,
                    peer_inboxes: peers(CUR_INBOXES),
                    peer_flags: peers(CUR_FLAGS),
                    epoch: CUR_EPOCH,
                    elements,
                    rank: rank as u32,
                    ranks: ranks as u32,
                })
                .map_err(err)?,
                bindings: bindings(states, rows, rank),
            })
        },
    )
}

fn synchronize(sessions: &[Session]) -> LabResult<()> {
    for session in sessions {
        session.stream.synchronize().map_err(err)?;
    }
    Ok(())
}

fn poison_outputs(sessions: &[Session], rows: &[Rows], entry: Entry) -> LabResult<()> {
    poison::synchronize(sessions)?;
    for (session, rows) in sessions.iter().zip(rows) {
        let buffer = if entry.round() {
            &rows.rounded
        } else {
            &rows.output
        };
        poison::poison(session, &[buffer])?;
    }
    Ok(())
}

/// Every rank's output of `entry` against the host: the first `live` rows
/// written with the expected bits, every other byte still poison.
fn check_outputs(
    sessions: &[Session],
    rows: &[Rows],
    entry: Entry,
    expected: &[f32],
    live: usize,
    label: &str,
) -> LabResult<()> {
    let count = live * WIDTH;
    for (rank, (session, rows)) in sessions.iter().zip(rows).enumerate() {
        let label = format!("{label}, rank {rank}");
        if entry.round() {
            let bytes = poison::read(session, &rows.rounded)?;
            let map = WriteMap::of(DType::Bf16).prefix(count)?;
            poison::check(&label, &bytes, &map, Fill::Pattern(&[POISON]))?;
            let actual = bytes[..count * 2]
                .chunks_exact(2)
                .map(|c| u16::from_ne_bytes([c[0], c[1]]))
                .collect::<Vec<_>>();
            let wanted = expected[..count]
                .iter()
                .map(|&v| f32_to_bf16(v))
                .collect::<Vec<_>>();
            if let Some(at) = first_difference(&actual, &wanted) {
                return Err(format!(
                    "{label}: rounded sum differs at {at}: {:#06x} != {:#06x}",
                    actual[at], wanted[at]
                ));
            }
        } else {
            let bytes = poison::read(session, &rows.output)?;
            let map = WriteMap::of(DType::F32).prefix(count)?;
            poison::check(&label, &bytes, &map, Fill::Pattern(&[POISON]))?;
            let actual = bytes[..count * 4]
                .chunks_exact(4)
                .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<_>>();
            if let Some(at) = first_difference(&actual, &bits(&expected[..count])) {
                return Err(format!(
                    "{label}: F32 sum differs at {at}: {:#010x} != {:#010x}",
                    actual[at],
                    expected[at].to_bits()
                ));
            }
        }
    }
    Ok(())
}

/// Every rank's pair output (where the pair and the current one-shot
/// write) against the host.
fn check_baseline(
    sessions: &[Session],
    rows: &[Rows],
    expected: &[f32],
    label: &str,
) -> LabResult<()> {
    synchronize(sessions)?;
    for (rank, session) in sessions.iter().enumerate() {
        let actual = session.download_f32(&rows[rank].pair_output, expected.len())?;
        if let Some(at) = first_difference(&bits(&actual), &bits(expected)) {
            return Err(format!(
                "{label}, rank {rank}: differs from the host at {at}"
            ));
        }
    }
    Ok(())
}

fn inputs_for(ranks: usize, seed: u64, rows: usize) -> Vec<Vec<f32>> {
    (0..ranks)
        .map(|rank| partial(seed, rank, rows * WIDTH))
        .collect()
}

fn live_counts(rows: usize) -> Vec<usize> {
    let mut counts = Vec::new();
    for live in [rows, 0, 1, rows / 2, rows - 1] {
        if !counts.contains(&live) {
            counts.push(live);
        }
    }
    counts
}

fn receipt(session: &Session, data: Vec<(&str, serde_json::Value)>) -> LabResult<()> {
    session.receipt(
        "comm_phased_f32",
        data.into_iter().map(|(k, v)| (k.to_owned(), v)).collect(),
    )?;
    Ok(())
}

fn run_bitwise(sessions: &[Session]) -> LabResult<()> {
    let ranks = sessions.len();
    let served = CollectiveModules::load(sessions, phased_f32::LAB_CUBIN)?;
    let current = CollectiveModules::load(sessions, "comm_sm120")?;
    let capacity = *BITWISE_ROWS.iter().max().unwrap();
    let states = states(sessions, capacity)?;
    for (seed, &count) in BITWISE_ROWS.iter().enumerate() {
        let inputs = inputs_for(ranks, seed as u64 + 1, count);
        let expected = host_sums(&inputs, count * WIDTH);
        let rows = rows_buffers(sessions, &inputs)?;
        let variant = served_variant(ranks, count)?;
        launch_pair(sessions, &current, &states, &rows, count, None)?;
        check_baseline(sessions, &rows, &expected, &format!("{count} rows, pair"))?;
        if rows[0].current.is_some() {
            launch_current_one_shot(sessions, &current, &states, &rows, count, None)?;
            check_baseline(
                sessions,
                &rows,
                &expected,
                &format!("{count} rows, current one-shot"),
            )?;
        }
        let geometries = if count <= 64 {
            &SMALL_GEOMETRIES
        } else {
            &LARGE_GEOMETRIES
        };
        for &geometry in geometries {
            for entry in ENTRIES.into_iter().filter(|e| e.takes(count)) {
                let shape = Shape::new(count, entry.round(), geometry);
                let label = format!("{count} rows, {geometry:?}, {}", entry.name());
                set_live(sessions, &states, count)?;
                poison_outputs(sessions, &rows, entry)?;
                launch_entry(
                    sessions,
                    &served,
                    entry.symbol(),
                    &states,
                    &rows,
                    entry,
                    shape,
                    None,
                )?;
                synchronize(sessions)?;
                check_outputs(sessions, &rows, entry, &expected, count, &label)?;
            }
        }
        // Live row counts below the rows, every entry, on the same state.
        for live in live_counts(count) {
            set_live(sessions, &states, live)?;
            for entry in ENTRIES.into_iter().filter(|e| e.takes(count)) {
                let shape = Shape::new(count, true, default_geometry(count));
                poison_outputs(sessions, &rows, entry)?;
                launch_entry(
                    sessions,
                    &served,
                    entry.symbol(),
                    &states,
                    &rows,
                    entry,
                    shape,
                    None,
                )?;
                synchronize(sessions)?;
                check_outputs(
                    sessions,
                    &rows,
                    entry,
                    &expected,
                    live,
                    &format!("{count} rows, {live} live, {}", entry.name()),
                )?;
            }
        }
        for session in sessions {
            receipt(
                session,
                vec![
                    ("case", json!("phased_f32_matches_pair_and_host")),
                    ("ranks", json!(ranks)),
                    ("rows", json!(count)),
                    ("served_variant", json!(format!("{variant:?}"))),
                    (
                        "current_one_shot_compared",
                        json!(rows[0].current.is_some()),
                    ),
                    ("pair_compared", json!(true)),
                    ("geometries", json!(geometries)),
                    ("live_counts", json!(live_counts(count))),
                    ("bitwise", json!(true)),
                ],
            )?;
        }
    }
    Ok(())
}

/// Every row count up to [`SENDERS_ONLY_MAX_ROWS`] in [`SENDERS_ONLY`],
/// on one state: every entry over all rows (no live count) and over each
/// live count below them, against the host, after the pair matched it.
fn run_senders_only(sessions: &[Session]) -> LabResult<()> {
    let ranks = sessions.len();
    let served = CollectiveModules::load(sessions, phased_f32::LAB_CUBIN)?;
    let current = CollectiveModules::load(sessions, "comm_sm120")?;
    let states = states(sessions, SENDERS_ONLY_MAX_ROWS)?;
    for count in 1..=SENDERS_ONLY_MAX_ROWS {
        let inputs = inputs_for(ranks, 100 + count as u64, count);
        let expected = host_sums(&inputs, count * WIDTH);
        let rows = rows_buffers(sessions, &inputs)?;
        launch_pair(sessions, &current, &states, &rows, count, None)?;
        check_baseline(sessions, &rows, &expected, &format!("{count} rows, pair"))?;
        let mut calls = 0;
        for entry in ENTRIES.into_iter().filter(|e| e.takes(count)) {
            let mut run = |live: Option<usize>| -> LabResult<()> {
                let shape = Shape::new(count, live.is_some(), SENDERS_ONLY);
                set_live(sessions, &states, live.unwrap_or(count))?;
                poison_outputs(sessions, &rows, entry)?;
                launch_entry(
                    sessions,
                    &served,
                    entry.symbol(),
                    &states,
                    &rows,
                    entry,
                    shape,
                    None,
                )?;
                synchronize(sessions)?;
                let label = match live {
                    Some(live) => format!("{count} rows, {live} live, {SENDERS_ONLY:?}"),
                    None => format!("{count} rows, {SENDERS_ONLY:?}"),
                };
                calls += 1;
                check_outputs(
                    sessions,
                    &rows,
                    entry,
                    &expected,
                    live.unwrap_or(count),
                    &format!("{label}, {}", entry.name()),
                )
            };
            run(None)?;
            for live in 0..=count {
                run(Some(live))?;
            }
        }
        for session in sessions {
            receipt(
                session,
                vec![
                    ("case", json!("phased_f32_senders_only_every_count")),
                    ("ranks", json!(ranks)),
                    ("rows", json!(count)),
                    ("senders", json!(SENDERS_ONLY.0)),
                    ("workers", json!(SENDERS_ONLY.1)),
                    ("pair_compared", json!(true)),
                    ("live_counts", json!((0..=count).collect::<Vec<_>>())),
                    ("calls", json!(calls)),
                    ("bitwise", json!(true)),
                ],
            )?;
        }
    }
    Ok(())
}

#[test]
fn phased_f32_senders_only_every_count() {
    gpu_test_multi(1 << 30, run_senders_only).expect("phased F32 all-reduce in 16 senders");
}

#[test]
fn phased_f32_matches_pair_and_host() {
    gpu_test_multi(2 << 30, run_bitwise).expect("phased F32 all-reduce bitwise sweep");
}

/// A PDL producer in front of a call (`probe_pdl_prefetch_sm120`): it
/// zeroes the rank's partials and live count, releases its dependent, and
/// restores both from the staged copy and `rows` after its tail. It reads
/// nothing from the host, so a graph can capture it.
struct Producer {
    function: runtime_cuda::Function,
    eviction: DeviceBuffer,
    checksum: DeviceBuffer,
    operands: DeviceBuffer,
}

impl Producer {
    fn new(session: &Session, rows: &Rows) -> LabResult<Self> {
        let module = session.module("probe_pdl_prefetch_sm120")?;
        let eviction = session.allocate(16)?;
        eviction
            .copy_from_bytes(&[0; 16], &session.stream)
            .map_err(err)?;
        // One PdlOperand: fresh (the staged copy), input, words, flags 0.
        let words = rows.input.byte_len() / 4;
        let mut operand = Vec::with_capacity(24);
        operand.extend(
            rows.staged
                .device_address_at(0, rows.staged.byte_len())
                .map_err(err)?
                .to_ne_bytes(),
        );
        operand.extend(
            rows.input
                .device_address_at(0, rows.input.byte_len())
                .map_err(err)?
                .to_ne_bytes(),
        );
        operand.extend((words as u32).to_ne_bytes());
        operand.extend(0u32.to_ne_bytes());
        let operands = session.allocate(24)?;
        operands
            .copy_from_bytes(&operand, &session.stream)
            .map_err(err)?;
        Ok(Self {
            function: module
                .function("chop_probe_pdl_prefetch_producer_v1")
                .map_err(err)?,
            eviction,
            checksum: session.allocate(128 * 4)?,
            operands,
        })
    }

    fn launch(&self, session: &Session, live: &DeviceBuffer, rows: usize) -> LabResult<()> {
        self.function
            .launch(
                &session.stream,
                launch::simple_config((1, 1, 1), (128, 1, 1), 0, LaunchConfig::NO_CLUSTER, true)?,
                &[
                    KernelArgument::Buffer(&self.eviction),
                    KernelArgument::Buffer(&self.checksum),
                    KernelArgument::U64(1),
                    KernelArgument::Buffer(&self.operands),
                    KernelArgument::U32(1),
                    KernelArgument::Buffer(live),
                    KernelArgument::U32(rows as u32),
                    KernelArgument::U64(PRODUCER_TAIL_NS),
                ],
            )
            .map_err(err)
    }
}

fn run_replays(sessions: &[Session]) -> LabResult<()> {
    let ranks = sessions.len();
    let served = CollectiveModules::load(sessions, phased_f32::LAB_CUBIN)?;
    let states = states(sessions, 2048)?;
    // (seed, rows, geometry, live rows): the rounded entries take the live
    // count, which the PDL producer restores after its tail; the F32
    // entries every row.
    let cases = [
        (40, 4, default_geometry(4), 4),
        (41, 32, default_geometry(32), 32),
        (42, 2048, default_geometry(2048), 2048),
        (43, 2, SENDERS_ONLY, 2),
        (44, 4, SENDERS_ONLY, 4),
        (45, 5, SENDERS_ONLY, 5),
        (46, 32, SENDERS_ONLY, 32),
        (47, 5, SENDERS_ONLY, 3),
        (48, 32, SENDERS_ONLY, 17),
        (49, 4, SENDERS_ONLY, 0),
        (50, 2, SENDERS_ONLY, 1),
        (51, 32, SENDERS_ONLY, 0),
    ];
    for (seed, count, geometry, live) in cases {
        let inputs = inputs_for(ranks, seed, count);
        let expected = host_sums(&inputs, count * WIDTH);
        let rows = rows_buffers(sessions, &inputs)?;
        set_live(sessions, &states, live)?;
        let check = |pair: [Entry; 2], phase: &str| -> LabResult<()> {
            synchronize(sessions)?;
            for entry in pair {
                let written = if entry.round() { live } else { count };
                check_outputs(
                    sessions,
                    &rows,
                    entry,
                    &expected,
                    written,
                    &format!("{count} rows, {live} live, {phase}, {}", entry.name()),
                )?;
            }
            Ok(())
        };
        let poison_both = || -> LabResult<()> {
            poison_outputs(sessions, &rows, Entry::F32)?;
            poison_outputs(sessions, &rows, Entry::Round)
        };
        let producers = sessions
            .iter()
            .zip(&rows)
            .map(|(session, rows)| Producer::new(session, rows))
            .collect::<LabResult<Vec<_>>>()?;
        let mut sequences = vec![[Entry::F32, Entry::Round]];
        if Entry::OneShot.takes(count) {
            sequences.push([Entry::OneShot, Entry::OneShotRound]);
            // Both protocols in one sequence, either order.
            sequences.push([Entry::OneShot, Entry::Round]);
            sequences.push([Entry::F32, Entry::OneShotRound]);
        }
        for pair in sequences {
            let label = format!("{}+{} {geometry:?}", pair[0].name(), pair[1].name());
            let both = |only_rank: Option<usize>| -> LabResult<()> {
                for entry in pair {
                    launch_entry(
                        sessions,
                        &served,
                        entry.symbol(),
                        &states,
                        &rows,
                        entry,
                        Shape::new(count, entry.round(), geometry),
                        only_rank,
                    )?;
                }
                Ok(())
            };
            poison_both()?;
            both(None)?;
            check(pair, &format!("{label} eager"))?;
            poison_both()?;
            both(None)?;
            check(pair, &format!("{label} relaunch"))?;
            let graphs = RankReplay::new(sessions, |rank, _| both(Some(rank)))?;
            for replay in 0..3 {
                poison_both()?;
                graphs.launch(sessions)?;
                check(pair, &format!("{label} graph replay {replay}"))?;
            }
            // Behind the PDL producer, eager and captured: each call
            // follows its own producer, which zeroes the partials and the
            // live count.
            let behind = |only_rank: Option<usize>| -> LabResult<()> {
                for entry in pair {
                    for rank in 0..ranks {
                        if only_rank.is_some_and(|only| only != rank) {
                            continue;
                        }
                        producers[rank].launch(&sessions[rank], &states[rank].live, live)?;
                        launch_entry(
                            sessions,
                            &served,
                            entry.symbol(),
                            &states,
                            &rows,
                            entry,
                            Shape::new(count, entry.round(), geometry),
                            Some(rank),
                        )?;
                    }
                }
                Ok(())
            };
            poison_both()?;
            behind(None)?;
            check(pair, &format!("{label} PDL producer"))?;
            let graphs = RankReplay::new(sessions, |rank, _| behind(Some(rank)))?;
            poison_both()?;
            graphs.launch(sessions)?;
            check(pair, &format!("{label} PDL producer graph"))?;
            for session in sessions {
                receipt(
                    session,
                    vec![
                        ("case", json!("phased_f32_replays")),
                        ("ranks", json!(ranks)),
                        ("rows", json!(count)),
                        ("live", json!(live)),
                        ("senders", json!(geometry.0)),
                        ("workers", json!(geometry.1)),
                        ("entries", json!(label)),
                        (
                            "phases",
                            json!(["eager", "relaunch", "graph", "pdl", "pdl_graph"]),
                        ),
                        ("bitwise", json!(true)),
                    ],
                )?;
            }
        }
    }
    Ok(())
}

#[test]
fn phased_f32_replays() {
    gpu_test_multi(1 << 30, run_replays).expect("phased F32 all-reduce replays");
}

/// Whether some rank's F32 output (or, for `no_gather_wait`, its snapshot
/// taken right after the call on its stream) differs from the host.
fn differs(
    sessions: &[Session],
    rows: &[Rows],
    snapshots: &[DeviceBuffer],
    expected: &[f32],
    snapshot: bool,
) -> LabResult<bool> {
    synchronize(sessions)?;
    for (rank, session) in sessions.iter().enumerate() {
        let buffer = if snapshot {
            &snapshots[rank]
        } else {
            &rows[rank].output
        };
        let actual = session.download_f32(buffer, expected.len())?;
        if first_difference(&bits(&actual), &bits(expected)).is_some() {
            return Ok(true);
        }
    }
    Ok(false)
}

fn run_mutants(sessions: &[Session]) -> LabResult<()> {
    let ranks = sessions.len();
    let served = CollectiveModules::load(sessions, phased_f32::LAB_CUBIN)?;
    let states = states(sessions, REQUIRED_TWO_SHOT_ROWS)?;
    let mut missed = Vec::new();
    // (entry, rows, geometry, required): the two-shot with workers, the
    // one-shot with workers, and the senders-only geometry recorded.
    let mut cases = Vec::new();
    for count in TWO_SHOT_MUTANT_ROWS {
        cases.push((Entry::F32, count, (16, 32), count == REQUIRED_TWO_SHOT_ROWS));
    }
    for count in ONE_SHOT_MUTANT_ROWS {
        cases.push((
            Entry::OneShot,
            count,
            (4, 4),
            count == REQUIRED_ONE_SHOT_ROWS,
        ));
        cases.push((Entry::OneShot, count, (4, 0), false));
    }
    cases.push((Entry::F32, 32, (4, 0), false));
    // The senders-only geometry a table of small row counts serves.
    cases.push((Entry::OneShot, 4, SENDERS_ONLY, true));
    cases.push((Entry::F32, 32, SENDERS_ONLY, true));
    for (seed, (entry, count, geometry, required)) in cases.into_iter().enumerate() {
        let inputs = inputs_for(ranks, 60 + seed as u64, count);
        let expected = host_sums(&inputs, count * WIDTH);
        let rows = rows_buffers(sessions, &inputs)?;
        let snapshots = sessions
            .iter()
            .map(|session| session.allocate(count * WIDTH * 4))
            .collect::<LabResult<Vec<_>>>()?;
        let producers = sessions
            .iter()
            .zip(&rows)
            .map(|(session, rows)| Producer::new(session, rows))
            .collect::<LabResult<Vec<_>>>()?;
        set_live(sessions, &states, count)?;
        let shape = Shape::new(count, false, geometry);
        for (tag, _) in phased_f32::MUTANTS {
            if tag == "no_gather_wait" && entry.one_shot() {
                // The one-shot has no gather.
                continue;
            }
            if tag == "shared_parity" {
                // A reuse defect across calls: phased_f32_generations.
                continue;
            }
            if tag == "reversed" && ranks < 3 {
                // (0 + x0) + x1 and (0 + x1) + x0 round alike: the order of
                // a two-rank sum cannot show.
                eprintln!("mutant reversed at {count} rows: inapplicable at {ranks} ranks");
                for session in sessions {
                    receipt(
                        session,
                        vec![
                            ("case", json!("phased_f32_mutants_are_caught")),
                            ("ranks", json!(ranks)),
                            ("rows", json!(count)),
                            ("entry", json!(entry.name())),
                            ("mutant", json!(tag)),
                            ("inapplicable", json!("two ranks")),
                        ],
                    )?;
                }
                continue;
            }
            let modules = CollectiveModules::load(sessions, &phased_f32::mutant_cubin(tag))?;
            let symbol = phased_f32::mutant_symbol(entry.symbol(), tag);
            let snapshot = tag == "no_gather_wait";
            // One call per rank (`only_rank`), behind the producer for the
            // late-wait mutant, then the snapshot copy on the same stream.
            let call = |only_rank: Option<usize>,
                        modules: &CollectiveModules,
                        symbol: &str|
             -> LabResult<()> {
                for rank in 0..ranks {
                    if only_rank.is_some_and(|only| only != rank) {
                        continue;
                    }
                    if tag == "late_wait" {
                        producers[rank].launch(&sessions[rank], &states[rank].live, count)?;
                    }
                    launch_entry(
                        sessions,
                        modules,
                        symbol,
                        &states,
                        &rows,
                        entry,
                        shape,
                        Some(rank),
                    )?;
                    if snapshot {
                        snapshots[rank]
                            .copy_range_from_async(
                                0,
                                &rows[rank].output,
                                0,
                                count * WIDTH * 4,
                                &sessions[rank].stream,
                            )
                            .map_err(err)?;
                    }
                }
                Ok(())
            };
            // A fresh state with poisoned inboxes and outputs per run: a
            // mutant may leave flags no later call can follow.
            let fresh = || -> LabResult<()> {
                reset(sessions, &states)?;
                for (session, (state, rows)) in sessions.iter().zip(states.iter().zip(&rows)) {
                    poison::poison(session, &[&state.inbox, &rows.output])?;
                }
                for (session, snapshot) in sessions.iter().zip(&snapshots) {
                    poison::poison(session, &[snapshot])?;
                }
                Ok(())
            };
            let label = format!("{} {tag} at {count} rows, {geometry:?}", entry.name());
            // The served entry passes the same harness.
            fresh()?;
            call(None, &served, entry.symbol())?;
            if differs(sessions, &rows, &snapshots, &expected, snapshot)? {
                return Err(format!("{label}: the served entry differs in this harness"));
            }
            // A race mutant shows only when the race is lost: up to
            // MUTANT_TRIALS runs each, caught at the first difference.
            let mut eager = 0;
            for trial in 1..=MUTANT_TRIALS {
                fresh()?;
                call(None, &modules, &symbol)?;
                if differs(sessions, &rows, &snapshots, &expected, snapshot)? {
                    eager = trial;
                    break;
                }
            }
            let graphs = RankReplay::new(sessions, |rank, _| call(Some(rank), &modules, &symbol))?;
            let mut replay = 0;
            for trial in 1..=MUTANT_TRIALS {
                fresh()?;
                graphs.launch(sessions)?;
                if differs(sessions, &rows, &snapshots, &expected, snapshot)? {
                    replay = trial;
                    break;
                }
            }
            let (eager_trial, replay_trial) = (eager, replay);
            let (eager, replay) = (eager > 0, replay > 0);
            eprintln!(
                "mutant {label}: eager caught {eager} (trial {eager_trial}), replay caught {replay} (trial {replay_trial})"
            );
            for session in sessions {
                receipt(
                    session,
                    vec![
                        ("case", json!("phased_f32_mutants_are_caught")),
                        ("ranks", json!(ranks)),
                        ("rows", json!(count)),
                        ("entry", json!(entry.name())),
                        ("senders", json!(geometry.0)),
                        ("workers", json!(geometry.1)),
                        ("mutant", json!(tag)),
                        ("required", json!(required)),
                        ("eager_caught", json!(eager)),
                        ("replay_caught", json!(replay)),
                        ("eager_trial", json!(eager_trial)),
                        ("replay_trial", json!(replay_trial)),
                    ],
                )?;
            }
            if required && !(eager && replay) {
                missed.push(format!("{label} (eager {eager}, replay {replay})"));
            }
        }
        // Leave a clean state for the next case.
        reset(sessions, &states)?;
    }
    if missed.is_empty() {
        Ok(())
    } else {
        Err(format!("mutants not caught: {}", missed.join(", ")))
    }
}

#[test]
fn phased_f32_mutants_are_caught() {
    gpu_test_multi(1 << 30, run_mutants).expect("phased F32 all-reduce mutants");
}

/// The calls of one generations case, back to back on one state: (entry,
/// generation) pairs, every generation with its own partials.
const ONE_SHOT_GENERATIONS: [Entry; 8] = [
    Entry::OneShot,
    Entry::OneShot,
    Entry::OneShotRound,
    Entry::F32,
    Entry::OneShot,
    Entry::Round,
    Entry::OneShot,
    Entry::OneShotRound,
];
const TWO_SHOT_GENERATIONS: [Entry; 4] = [Entry::F32, Entry::Round, Entry::F32, Entry::Round];
/// One call of a generations case: its entry, rows and live rows (the
/// rounded entries reduce only the live rows; the F32 entries every row).
type Call = (Entry, usize, usize);
/// Calls whose row count and live count change from one to the next, as a
/// server's decode and verify buckets do: in [`SENDERS_ONLY`], the one-shot
/// entries up to 4 rows and the two-shot round entry above, with full,
/// partial and zero live counts (a zero-live call reduces nothing and keeps
/// the epoch).
const MIXED_GENERATIONS: [Call; 12] = [
    (Entry::OneShotRound, 2, 2),
    (Entry::Round, 5, 3),
    (Entry::OneShot, 4, 4),
    (Entry::Round, 32, 0),
    (Entry::OneShotRound, 3, 1),
    (Entry::Round, 16, 16),
    (Entry::OneShotRound, 4, 0),
    (Entry::OneShot, 2, 2),
    (Entry::Round, 8, 5),
    (Entry::OneShotRound, 2, 1),
    (Entry::Round, 32, 31),
    (Entry::OneShotRound, 4, 4),
];
/// Sending and summing CTAs of a launch.
type Geometry = (u32, u32);
/// Clock cycles the delayed peer spins before its call (about 100 us).
const DELAY_CYCLES: u64 = 200_000;

/// One rank's buffers of a generations case: each generation's partials,
/// the one shared F32 and BF16 output every call writes (of the case's
/// most rows), and each generation's snapshot of its result, copied on the
/// rank's stream right after the call, before any later call or host
/// fence.
struct Generations {
    inputs: Vec<DeviceBuffer>,
    output: DeviceBuffer,
    rounded: DeviceBuffer,
    snapshots: Vec<DeviceBuffer>,
    /// Each generation's live count, copied into the state's live count on
    /// the rank's stream right before its call.
    lives: DeviceBuffer,
}

fn generation_bindings<'a>(
    states: &'a [State],
    ranks: &'a [Generations],
    generation: usize,
    rank: usize,
) -> Bindings<'a> {
    let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
    buffers.insert(INPUT, &ranks[rank].inputs[generation]);
    buffers.insert(OUTPUT, &ranks[rank].output);
    buffers.insert(ROUNDED, &ranks[rank].rounded);
    buffers.insert(EPOCH, &states[rank].epoch);
    buffers.insert(LIVE, &states[rank].live);
    for peer in 0..states.len() {
        let q = peer as u64;
        buffers.insert(INBOXES + q, &states[peer].inbox);
        buffers.insert(FLAGS + q, &states[peer].flags);
        buffers.insert(OUTPUTS + q, &ranks[peer].output);
        buffers.insert(ROUNDEDS + q, &ranks[peer].rounded);
    }
    Bindings { buffers }
}

/// Words of every generation's snapshot on every rank that differ from
/// that generation's own host result.
fn generation_wrong_words(
    sessions: &[Session],
    ranks: &[Generations],
    sequence: &[Call],
    expected: &[Vec<f32>],
) -> LabResult<usize> {
    synchronize(sessions)?;
    let mut wrong = 0;
    for (session, rank) in sessions.iter().zip(ranks) {
        for (generation, &(entry, _, live)) in sequence.iter().enumerate() {
            let bytes = poison::read(session, &rank.snapshots[generation])?;
            let sums = &expected[generation];
            if entry.round() {
                // Only the live rows are this generation's.
                for (chunk, &sum) in bytes.chunks_exact(2).zip(&sums[..live * WIDTH]) {
                    wrong +=
                        usize::from(u16::from_ne_bytes([chunk[0], chunk[1]]) != f32_to_bf16(sum));
                }
            } else {
                for (chunk, &sum) in bytes.chunks_exact(4).zip(sums) {
                    let bits = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    wrong += usize::from(bits != sum.to_bits());
                }
            }
        }
    }
    Ok(wrong)
}

/// Back-to-back generations on one state, each with distinct partials,
/// alternating one-shot and two-shot entries (the one-shot's parity
/// regions are reused two calls later), the rank `generation % ranks`
/// delayed by a spinning kernel before each call, every result snapshotted
/// on the device before the next call: eager, then captured and replayed.
/// The served entries must show no wrong word; the shared-parity mutant
/// (both one-shot parities in one inbox region) must show wrong words on
/// some case and mode, counted from observed outputs only.
fn run_generations(sessions: &[Session]) -> LabResult<()> {
    let ranks = sessions.len();
    let served = CollectiveModules::load(sessions, phased_f32::LAB_CUBIN)?;
    let mutant = CollectiveModules::load(sessions, &phased_f32::mutant_cubin("shared_parity"))?;
    let spins = sessions
        .iter()
        .map(|session| {
            session
                .module(&kernel_lab::latent_shapes::cubin_name())?
                .function(&kernel_lab::latent_shapes::pdl_copy_symbol())
                .map_err(err)
        })
        .collect::<LabResult<Vec<_>>>()?;
    let scratch = sessions
        .iter()
        .map(|session| session.allocate(16))
        .collect::<LabResult<Vec<_>>>()?;
    let states = states(sessions, 2048)?;
    let at = |sequence: &[Entry], count: usize| -> Vec<Call> {
        sequence
            .iter()
            .map(|&entry| (entry, count, count))
            .collect()
    };
    let cases: [(Geometry, Vec<Call>); 8] = [
        ((4, 0), at(&ONE_SHOT_GENERATIONS, 4)),
        ((4, 4), at(&ONE_SHOT_GENERATIONS, 32)),
        ((1, 0), at(&ONE_SHOT_GENERATIONS, 64)),
        ((4, 0), at(&ONE_SHOT_GENERATIONS, 64)),
        ((16, 32), at(&TWO_SHOT_GENERATIONS, 2048)),
        (SENDERS_ONLY, at(&ONE_SHOT_GENERATIONS, 4)),
        (SENDERS_ONLY, at(&ONE_SHOT_GENERATIONS, 32)),
        (SENDERS_ONLY, MIXED_GENERATIONS.to_vec()),
    ];
    let mut mutant_wrong = 0;
    let mut failures = Vec::new();
    for (case, (geometry, sequence)) in cases.iter().enumerate() {
        let (geometry, sequence) = (*geometry, sequence.as_slice());
        // The largest call's rows size the shared outputs.
        let count = sequence.iter().map(|&(_, rows, _)| rows).max().unwrap();
        let elements = count * WIDTH;
        let expected = sequence
            .iter()
            .enumerate()
            .map(|(generation, &(_, rows, _))| {
                host_sums(
                    &inputs_for(ranks, 200 + (case * 16 + generation) as u64, rows),
                    rows * WIDTH,
                )
            })
            .collect::<Vec<_>>();
        let buffers = sessions
            .iter()
            .enumerate()
            .map(|(rank, session)| {
                Ok(Generations {
                    inputs: sequence
                        .iter()
                        .enumerate()
                        .map(|(generation, &(_, rows, _))| {
                            session.upload_f32(&partial(
                                200 + (case * 16 + generation) as u64,
                                rank,
                                rows * WIDTH,
                            ))
                        })
                        .collect::<LabResult<Vec<_>>>()?,
                    output: session.allocate(elements * 4)?,
                    rounded: session.allocate(elements * 2)?,
                    snapshots: sequence
                        .iter()
                        .map(|&(entry, rows, _)| {
                            session.allocate(rows * WIDTH * if entry.round() { 2 } else { 4 })
                        })
                        .collect::<LabResult<Vec<_>>>()?,
                    lives: {
                        let lives = sequence
                            .iter()
                            .flat_map(|&(_, _, live)| (live as u32).to_ne_bytes())
                            .collect::<Vec<_>>();
                        let buffer = session.allocate(lives.len())?;
                        buffer
                            .copy_from_bytes(&lives, &session.stream)
                            .map_err(err)?;
                        buffer
                    },
                })
            })
            .collect::<LabResult<Vec<_>>>()?;
        for generation in 1..sequence.len() {
            if bits(&expected[generation]) == bits(&expected[generation - 1]) {
                return Err("consecutive generations must differ".into());
            }
        }
        set_live(sessions, &states, count)?;
        // Every generation on `only_rank` (or every rank), in order.
        let sequence_on = |modules: &CollectiveModules, mutant: bool, only_rank: Option<usize>| {
            for (generation, &(entry, rows, _)) in sequence.iter().enumerate() {
                let delayed = generation % ranks;
                if only_rank.is_none_or(|only| only == delayed) {
                    spins[delayed]
                        .launch(
                            &sessions[delayed].stream,
                            launch::simple_config(
                                (1, 1, 1),
                                (32, 1, 1),
                                0,
                                LaunchConfig::NO_CLUSTER,
                                false,
                            )?,
                            &[
                                KernelArgument::Buffer(&scratch[delayed]),
                                KernelArgument::Buffer(&scratch[delayed]),
                                KernelArgument::U64(0),
                                KernelArgument::U64(DELAY_CYCLES),
                            ],
                        )
                        .map_err(err)?;
                }
                let symbol = if mutant {
                    phased_f32::mutant_symbol(entry.symbol(), "shared_parity")
                } else {
                    entry.symbol().to_owned()
                };
                // The generation's live count, on each rank's stream.
                for (rank, session) in sessions.iter().enumerate() {
                    if only_rank.is_some_and(|only| only != rank) {
                        continue;
                    }
                    states[rank]
                        .live
                        .copy_range_from_async(
                            0,
                            &buffers[rank].lives,
                            generation * 4,
                            4,
                            &session.stream,
                        )
                        .map_err(err)?;
                }
                launch_bound(
                    sessions,
                    modules,
                    &symbol,
                    entry,
                    Shape::new(rows, entry.round(), geometry),
                    only_rank,
                    |rank| generation_bindings(&states, &buffers, generation, rank),
                )?;
                for (rank, session) in sessions.iter().enumerate() {
                    if only_rank.is_some_and(|only| only != rank) {
                        continue;
                    }
                    let (source, bytes) = if entry.round() {
                        (&buffers[rank].rounded, rows * WIDTH * 2)
                    } else {
                        (&buffers[rank].output, rows * WIDTH * 4)
                    };
                    buffers[rank].snapshots[generation]
                        .copy_range_from_async(0, source, 0, bytes, &session.stream)
                        .map_err(err)?;
                }
            }
            LabResult::Ok(())
        };
        let fresh = || -> LabResult<()> {
            reset(sessions, &states)?;
            for (session, (state, rank)) in sessions.iter().zip(states.iter().zip(&buffers)) {
                poison::poison(session, &[&state.inbox, &rank.output, &rank.rounded])?;
                for snapshot in &rank.snapshots {
                    poison::poison(session, &[snapshot])?;
                }
            }
            Ok(())
        };
        let label = match sequence
            .iter()
            .all(|&(_, rows, live)| rows == count && live == count)
        {
            true => format!("{count} rows, {geometry:?}"),
            false => format!("mixed rows up to {count}, {geometry:?}"),
        };
        // The served entries: eager, then three replays of one captured
        // sequence, each on a fresh state.
        fresh()?;
        sequence_on(&served, false, None)?;
        let mut served_wrong = vec![generation_wrong_words(
            sessions, &buffers, sequence, &expected,
        )?];
        let graphs = RankReplay::new(sessions, |rank, _| sequence_on(&served, false, Some(rank)))?;
        for _ in 0..3 {
            fresh()?;
            graphs.launch(sessions)?;
            served_wrong.push(generation_wrong_words(
                sessions, &buffers, sequence, &expected,
            )?);
        }
        drop(graphs);
        if served_wrong.iter().any(|&wrong| wrong > 0) {
            failures.push(format!("{label}: served wrong words {served_wrong:?}"));
        }
        // The mutant, where it applies (one-shot sequences): up to
        // MUTANT_TRIALS runs per mode, stopping at the first wrong word.
        let mut mutant_counts = Vec::new();
        if sequence.iter().any(|(entry, _, _)| entry.one_shot()) {
            for trial in 0..MUTANT_TRIALS {
                fresh()?;
                sequence_on(&mutant, true, None)?;
                let wrong = generation_wrong_words(sessions, &buffers, sequence, &expected)?;
                mutant_counts.push(("eager", trial + 1, wrong));
                if wrong > 0 {
                    break;
                }
            }
            let graphs =
                RankReplay::new(sessions, |rank, _| sequence_on(&mutant, true, Some(rank)))?;
            for trial in 0..MUTANT_TRIALS {
                fresh()?;
                graphs.launch(sessions)?;
                let wrong = generation_wrong_words(sessions, &buffers, sequence, &expected)?;
                mutant_counts.push(("replay", trial + 1, wrong));
                if wrong > 0 {
                    break;
                }
            }
            mutant_wrong += mutant_counts
                .iter()
                .map(|&(_, _, wrong)| wrong)
                .sum::<usize>();
        }
        eprintln!(
            "generations {label}: served wrong words {served_wrong:?}; shared_parity mutant (mode, trial, wrong words) {mutant_counts:?}"
        );
        for session in sessions {
            receipt(
                session,
                vec![
                    ("case", json!("phased_f32_generations")),
                    ("ranks", json!(ranks)),
                    ("rows", json!(count)),
                    ("senders", json!(geometry.0)),
                    ("workers", json!(geometry.1)),
                    (
                        "sequence",
                        json!(
                            sequence
                                .iter()
                                .map(|&(entry, rows, live)| {
                                    format!("{}@{rows}/{live}", entry.name())
                                })
                                .collect::<Vec<_>>()
                        ),
                    ),
                    ("served_wrong_words", json!(served_wrong)),
                    ("mutant_wrong_words", json!(mutant_counts)),
                ],
            )?;
        }
        reset(sessions, &states)?;
    }
    if mutant_wrong == 0 {
        failures.push("the shared-parity mutant showed no wrong word in any case".into());
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}

#[test]
fn phased_f32_generations() {
    gpu_test_multi(1 << 30, run_generations).expect("phased F32 all-reduce generations");
}

/// The epoch whose next call repeats a fresh state's first flag tag:
/// receivers compare flags for equality with `0x80000000 | epoch`, so the
/// call at epoch 2^31 + 1 tags like the call at epoch 1, and the one-shot's
/// parity (`epoch & 1`) is the same.
const TAG_REPEATS_AT: u32 = 0x8000_0000;

/// Words of every rank's rounded output that differ from the host's sums
/// rounded to BF16, over `count` rows.
fn rounded_wrong_words(
    sessions: &[Session],
    rows: &[Rows],
    expected: &[f32],
    count: usize,
) -> LabResult<usize> {
    synchronize(sessions)?;
    let mut wrong = 0;
    for (session, rows) in sessions.iter().zip(rows) {
        let bytes = poison::read(session, &rows.rounded)?;
        for (chunk, &sum) in bytes.chunks_exact(2).zip(&expected[..count * WIDTH]) {
            wrong += usize::from(u16::from_ne_bytes([chunk[0], chunk[1]]) != f32_to_bf16(sum));
        }
    }
    Ok(wrong)
}

/// A state whose flag tags repeat: a fresh state's first call (epoch 1),
/// then every rank's epoch set to [`TAG_REPEATS_AT`] (as after 2^31 - 1
/// calls of the other protocol, which leave this protocol's flags and
/// inbox as the first call left them), then a call with new partials whose
/// peers but rank 0 start about 100 us late. Without a reset rank 0 matches
/// the first call's flags and sums the late peers' old partials; with the
/// reset a server applies before tags can repeat (every rank idle, every
/// rank's flags and epoch zeroed, every rank idle again) the call is exact.
/// One-shot (4 rows) and two-shot (32 rows) rounded entries in
/// [`SENDERS_ONLY`]: the reset arm must show no wrong word in every trial,
/// and the unreset arm must show wrong words in some trial, so the case
/// sees the hazard the reset removes.
fn run_tag_reuse(sessions: &[Session]) -> LabResult<()> {
    let ranks = sessions.len();
    let served = CollectiveModules::load(sessions, phased_f32::LAB_CUBIN)?;
    let spins = sessions
        .iter()
        .map(|session| {
            session
                .module(&kernel_lab::latent_shapes::cubin_name())?
                .function(&kernel_lab::latent_shapes::pdl_copy_symbol())
                .map_err(err)
        })
        .collect::<LabResult<Vec<_>>>()?;
    let scratch = sessions
        .iter()
        .map(|session| session.allocate(16))
        .collect::<LabResult<Vec<_>>>()?;
    let states = states(sessions, SENDERS_ONLY_MAX_ROWS)?;
    let mut failures = Vec::new();
    for (case, (entry, count)) in [(Entry::OneShotRound, 4), (Entry::Round, 32)]
        .into_iter()
        .enumerate()
    {
        let shape = Shape::new(count, true, SENDERS_ONLY);
        let first_inputs = inputs_for(ranks, 300 + 2 * case as u64, count);
        let next_inputs = inputs_for(ranks, 301 + 2 * case as u64, count);
        let first_expected = host_sums(&first_inputs, count * WIDTH);
        let next_expected = host_sums(&next_inputs, count * WIDTH);
        let first = rows_buffers(sessions, &first_inputs)?;
        let next = rows_buffers(sessions, &next_inputs)?;
        set_live(sessions, &states, count)?;
        let run = |reset_before: bool| -> LabResult<usize> {
            reset(sessions, &states)?;
            for (session, state) in sessions.iter().zip(&states) {
                poison::poison(session, &[&state.inbox])?;
            }
            poison_outputs(sessions, &first, entry)?;
            launch_entry(
                sessions,
                &served,
                entry.symbol(),
                &states,
                &first,
                entry,
                shape,
                None,
            )?;
            check_outputs(
                sessions,
                &first,
                entry,
                &first_expected,
                count,
                &format!("tag reuse {count} rows, first call"),
            )?;
            for (session, state) in sessions.iter().zip(&states) {
                state
                    .epoch
                    .copy_from_bytes(&TAG_REPEATS_AT.to_ne_bytes(), &session.stream)
                    .map_err(err)?;
            }
            if reset_before {
                reset(sessions, &states)?;
            }
            poison_outputs(sessions, &next, entry)?;
            for (rank, session) in sessions.iter().enumerate().skip(1) {
                spins[rank]
                    .launch(
                        &session.stream,
                        launch::simple_config(
                            (1, 1, 1),
                            (32, 1, 1),
                            0,
                            LaunchConfig::NO_CLUSTER,
                            false,
                        )?,
                        &[
                            KernelArgument::Buffer(&scratch[rank]),
                            KernelArgument::Buffer(&scratch[rank]),
                            KernelArgument::U64(0),
                            KernelArgument::U64(DELAY_CYCLES),
                        ],
                    )
                    .map_err(err)?;
            }
            launch_entry(
                sessions,
                &served,
                entry.symbol(),
                &states,
                &next,
                entry,
                shape,
                None,
            )?;
            rounded_wrong_words(sessions, &next, &next_expected, count)
        };
        let label = format!("{} at {count} rows", entry.name());
        let mut reset_wrong = Vec::new();
        for _ in 0..3 {
            reset_wrong.push(run(true)?);
        }
        let mut unreset_wrong = Vec::new();
        for _ in 0..MUTANT_TRIALS {
            let wrong = run(false)?;
            unreset_wrong.push(wrong);
            if wrong > 0 {
                break;
            }
        }
        eprintln!(
            "tag reuse {label}: wrong words with the reset {reset_wrong:?}, without it {unreset_wrong:?}"
        );
        for session in sessions {
            receipt(
                session,
                vec![
                    ("case", json!("phased_f32_tag_reuse")),
                    ("ranks", json!(ranks)),
                    ("rows", json!(count)),
                    ("entry", json!(entry.name())),
                    ("senders", json!(SENDERS_ONLY.0)),
                    ("epoch_set", json!(TAG_REPEATS_AT)),
                    ("reset_wrong_words", json!(reset_wrong)),
                    ("unreset_wrong_words", json!(unreset_wrong)),
                ],
            )?;
        }
        if reset_wrong.iter().any(|&wrong| wrong > 0) {
            failures.push(format!(
                "{label}: wrong words after the reset {reset_wrong:?}"
            ));
        }
        if unreset_wrong.iter().all(|&wrong| wrong == 0) {
            failures.push(format!(
                "{label}: no wrong word without the reset in {} trials: the case does not see the stale tags",
                unreset_wrong.len()
            ));
        }
        // Leave a clean state for the next case.
        reset(sessions, &states)?;
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}

#[test]
fn phased_f32_tag_reuse() {
    gpu_test_multi(1 << 30, run_tag_reuse).expect("phased F32 all-reduce tag reuse");
}

fn env_list<T: std::str::FromStr + Clone>(name: &str, default: &[T]) -> Vec<T> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|item| {
                    item.trim()
                        .parse()
                        .map_err(|_| format!("{name}: {item:?} is not a number"))
                        .expect("a comma-separated number list")
                })
                .collect()
        })
        .unwrap_or_else(|| default.to_vec())
}

/// Microseconds per call, critical path over ranks, of `calls` repeated
/// calls in one graph per rank.
fn time_calls(
    sessions: &[Session],
    calls: usize,
    one: impl Fn(usize) -> LabResult<()>,
) -> LabResult<(f32, f32)> {
    let graphs = sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| {
            session.capture(|_| {
                for _ in 0..calls {
                    one(rank)?;
                }
                Ok(())
            })
        })
        .collect::<LabResult<Vec<_>>>()?;
    let timing = time_multi_graph(sessions, &graphs, calls, 2, 5)?;
    let per = |ms: f32| ms * 1000.0 / calls as f32;
    Ok((
        per(timing.critical_path.median_ms),
        per(timing.critical_path.min_ms),
    ))
}

/// The (senders, workers) the timing sweeps at `rows`:
/// `CHOPINFER_KERNEL_BENCH_SENDERS` x `CHOPINFER_KERNEL_BENCH_WORKERS` when
/// either is set, else a small-call or a large-call list.
fn timing_geometries(rows: usize) -> Vec<(u32, u32)> {
    let senders = std::env::var("CHOPINFER_KERNEL_BENCH_SENDERS").is_ok();
    let workers = std::env::var("CHOPINFER_KERNEL_BENCH_WORKERS").is_ok();
    if senders || workers {
        let senders = env_list("CHOPINFER_KERNEL_BENCH_SENDERS", &[16]);
        let workers = env_list("CHOPINFER_KERNEL_BENCH_WORKERS", &[32]);
        return workers
            .iter()
            .flat_map(|&w| senders.iter().map(move |&s| (s, w)))
            .collect();
    }
    if rows <= 64 {
        vec![
            (1, 0),
            (2, 0),
            (4, 0),
            (8, 0),
            (16, 0),
            (1, 1),
            (4, 4),
            (8, 8),
        ]
    } else {
        [8, 16, 24, 32, 48, 64]
            .into_iter()
            .map(|s| (s, 32))
            .chain([(32, 0), (64, 0)])
            .collect()
    }
}

fn run_timing(sessions: &[Session]) -> LabResult<()> {
    let ranks = sessions.len();
    let served = CollectiveModules::load(sessions, phased_f32::LAB_CUBIN)?;
    let current = CollectiveModules::load(sessions, "comm_sm120")?;
    let all_rows = env_list(
        "CHOPINFER_KERNEL_BENCH_ROWS",
        &[
            1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
    );
    let capacity = *all_rows.iter().max().ok_or("no benchmark rows")?;
    let states = states(sessions, capacity)?;
    for &count in &all_rows {
        let calls = (8192 / count).clamp(4, 64);
        let inputs = vec![vec![0.0_f32; count * WIDTH]; ranks];
        let rows = rows_buffers(sessions, &inputs)?;
        set_live(sessions, &states, count)?;
        let bytes = (count * WIDTH * 4) as f32;
        // Bytes each rank sends: (ranks - 1) / ranks of the F32 payload in
        // the scatter and of the F32 or BF16 payload in the gather; the
        // one-shots send ranks - 1 payloads.
        let share = (ranks - 1) as f32 / ranks as f32;
        let record = |variant: &str,
                      senders: u32,
                      workers: u32,
                      sent: f32,
                      (median, min): (f32, f32)|
         -> LabResult<()> {
            let gbps = sent / median / 1000.0;
            eprintln!(
                "phased_f32 timing: rows {count:5} {variant:14} senders {senders:2} workers {workers:2}: median {median:9.2} us, min {min:9.2} us, {gbps:6.2} GB/s sent per rank"
            );
            for session in sessions {
                receipt(
                    session,
                    vec![
                        ("case", json!("phased_f32_timing")),
                        ("ranks", json!(ranks)),
                        ("rows", json!(count)),
                        ("variant", json!(variant)),
                        ("senders", json!(senders)),
                        ("workers", json!(workers)),
                        ("calls_per_graph", json!(calls)),
                        ("median_us_per_call", json!(median)),
                        ("min_us_per_call", json!(min)),
                        ("sent_gb_per_s_per_rank", json!(gbps)),
                    ],
                )?;
            }
            Ok(())
        };
        if rows[0].current.is_some() {
            let timing = time_calls(sessions, calls, |rank| {
                launch_current_one_shot(sessions, &current, &states, &rows, count, Some(rank))
            })?;
            record("current_oneshot", 0, 0, (ranks - 1) as f32 * bytes, timing)?;
        }
        let timing = time_calls(sessions, calls, |rank| {
            launch_pair(sessions, &current, &states, &rows, count, Some(rank))
        })?;
        record("pair", 8, 0, 2.0 * share * bytes, timing)?;
        for (senders, workers) in timing_geometries(count) {
            for entry in ENTRIES.into_iter().filter(|e| e.takes(count)) {
                let shape = Shape::new(count, entry.round(), (senders, workers));
                let timing = time_calls(sessions, calls, |rank| {
                    launch_entry(
                        sessions,
                        &served,
                        entry.symbol(),
                        &states,
                        &rows,
                        entry,
                        shape,
                        Some(rank),
                    )
                })?;
                let sent = match entry {
                    Entry::F32 => 2.0 * share * bytes,
                    Entry::Round => 1.5 * share * bytes,
                    Entry::OneShot | Entry::OneShotRound => (ranks - 1) as f32 * bytes,
                };
                record(entry.name(), senders, workers, sent, timing)?;
            }
        }
    }
    Ok(())
}

#[test]
fn phased_f32_timing() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        eprintln!(
            "skipping phased F32 all-reduce timing: set CHOPINFER_KERNEL_BENCH=1 on idle GPUs"
        );
        return;
    }
    gpu_test_multi(2 << 30, run_timing).expect("phased F32 all-reduce timing");
}
