//! Short text prefills split into graph segments
//! (`bundle_runtime::graphs::programs`).
//!
//! An eligible prefill (`eligible`) runs the upper-skip plan of one text
//! item under 512 rows, without upper-skip DMA. Its body splits into graph
//! segments at the host work between them (`split`): the join of each
//! prepared Engram gather, before its layer's first consumer, and the index
//! selections of the full index sources, whose bounded batches follow the
//! uploaded positions (`ops::indexer::prefill`), so their launches change
//! with the context every few dozen tokens and run eagerly. Every other
//! launch is fixed by the rows, the input layout, the scored rows and a
//! DSpark target's retained feature range, or by a coarse context class
//! such as the block selections' early-exit route; the key
//! (`Key`) holds the fingerprint of every graph segment on every rank, so a
//! launch that does change only makes another key, never a stale replay.
//! The head, after the logits compaction copies, is one more program with
//! one segment.

use std::cell::RefCell;
use std::ops::Range;

use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::graphs::programs::{
    Admission, Eager, Fingerprint, Limits, Programs as Store, fingerprint,
};
use bundle_runtime::pipeline::{PoisonOnUnwind, PreparedLaunch};
use model_deepseek41_contract::indexer;
use model_deepseek41_contract::serving::DeepSeek;
use runtime_bundle_abi::{BundleAbiError, BundleStreamHandleV1};

use super::{PREFILL, Programs, Run, skip};
use crate::instance::Instance;
use crate::ops::indexer::REINDEX_BLOCK;
use crate::program::Stage;

/// The most rows of an eligible prefill.
pub(super) const MAX_ROWS: u32 = 511;

/// Free graph handles per rank a bounded ledger keeps after a prefill
/// capture, for the decode, verify and drafter graphs serving captures
/// lazily. The served DSpark profile's warmup captured 158 handles per rank
/// of its 512, and 12 hours of agent traffic raised that to 173.
pub(super) const HEADROOM: usize = 128;

/// Prefill keys that ran once remembered at once.
const SEEN: usize = 64;

/// Head programs captured at once: one per head input layout.
const HEADS: usize = 4;

/// The captured body and head programs.
pub(in crate::instance) struct Stores {
    body: Store<Key, DeepSeek>,
    head: Store<Key, DeepSeek>,
}

impl Stores {
    /// Stores holding at most `programs` body programs, or none at zero,
    /// which keeps every prefill eager.
    pub(in crate::instance) fn new(programs: usize) -> Option<RefCell<Self>> {
        let limits = |programs| Limits {
            programs,
            seen: SEEN,
            headroom: HEADROOM,
        };
        (programs > 0).then(|| {
            RefCell::new(Self {
                body: Store::new(limits(programs)),
                head: Store::new(limits(HEADS)),
            })
        })
    }
}

/// The host work between two graph segments.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum Gap {
    /// The join of the prepared Engram gather of this layer index.
    Join(usize),
    /// An eager section: these launch ranges of every rank.
    Eager(Vec<Range<usize>>),
}

/// A program split into graph segments and the gaps between them:
/// `graphs[s][rank]` is a launch range of rank `rank`, and `gaps[s]` runs
/// between segments `s` and `s + 1`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Split {
    pub graphs: Vec<Vec<Range<usize>>>,
    pub gaps: Vec<Gap>,
}

impl Split {
    /// Each segment's launches on every rank, for capture.
    pub fn segments<'a>(
        &self,
        programs: &'a [Vec<PreparedLaunch>],
    ) -> Vec<Vec<&'a [PreparedLaunch]>> {
        self.graphs
            .iter()
            .map(|ranks| {
                ranks
                    .iter()
                    .zip(programs)
                    .map(|(range, program)| &program[range.clone()])
                    .collect()
            })
            .collect()
    }

    /// Graph segments with launches on the ranks: the handles a capture
    /// takes per rank.
    pub fn handles(&self) -> usize {
        self.graphs
            .iter()
            .filter(|ranks| ranks.iter().any(|range| !range.is_empty()))
            .count()
    }
}

/// A program's key: its rows and the fingerprint of every graph segment on
/// every rank, segment-major.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Key {
    pub rows: u32,
    pub prints: Vec<Fingerprint>,
}

impl Key {
    /// The key of `programs` split by `split`.
    pub fn of(rows: u32, programs: &[Vec<PreparedLaunch>], split: &Split) -> Result<Self, String> {
        let prints = split
            .segments(programs)
            .iter()
            .flatten()
            .map(|launches| fingerprint(launches).map_err(|e| format!("{e:?}")))
            .collect::<Result<_, _>>()?;
        Ok(Self { rows, prints })
    }

    /// One digest of every fingerprint, for logs.
    pub fn digest(&self) -> String {
        use std::hash::{DefaultHasher, Hash, Hasher};
        let mut digest = DefaultHasher::new();
        self.rows.hash(&mut digest);
        self.prints.hash(&mut digest);
        format!("{:016x}", digest.finish())
    }
}

/// Whether `run`'s upper-skip plan may split into graph segments: one text
/// item under 512 rows, no upper-skip DMA, no position axes.
pub(super) fn eligible(run: &Run, plan: &skip::Plan) -> bool {
    let inputs = &run.inputs;
    run.arena == PREFILL
        && run.media.is_none()
        && !plan.uses_dma()
        && inputs.request_ids.len() == 1
        && inputs.position_axes.is_empty()
        && (1..=MAX_ROWS).contains(&inputs.live_tokens())
}

/// Whether an eligible prefill may replay as graphs: it continues a prefix.
/// Startup warmup runs about a hundred prefix-zero prompts of 1 to 32 rows,
/// repeating each length; captured, they would fill the store under the
/// decode and verify graphs warmup captures after them, which pins them
/// for good, and served traffic rarely repeats them (29 of 5,017 logged
/// steps under 512 rows started at position zero). Continuations are what
/// agent turns and prompt tails run.
pub(super) fn admits(run: &Run, plan: &skip::Plan) -> bool {
    eligible(run, plan)
        && run
            .inputs
            .prefix_len
            .first()
            .is_some_and(|&start| start > 0)
}

/// Whether `stage` runs eagerly between graph segments: an index selection
/// whose bounded prefill batches follow the uploaded positions.
fn eager(instance: &Instance, stage: Stage) -> bool {
    let Stage::IndexSelect(layer) = stage else {
        return false;
    };
    let bounded = instance
        .ops
        .select
        .as_ref()
        .is_some_and(|select| select.prefill.is_some());
    bounded
        && indexer::block_selection(instance.config.shape.geometry, layer, REINDEX_BLOCK).is_none()
}

/// Splits `programs` at `joins` (each rank's first consumer of each
/// prepared gather layer, in layer order; none without prepared gathers)
/// and around its eager index selections. Every rank must split alike, and
/// every graph segment and eager section must issue the same collectives
/// on every rank.
pub(super) fn split(
    instance: &Instance,
    programs: &Programs,
    joins: Option<&[Vec<usize>]>,
) -> Result<Split, String> {
    let (launches, collectives, stages) = programs;
    if launches.is_empty()
        || launches.len() != stages.len()
        || launches.len() != collectives.len()
        || joins.is_some_and(|joins| joins.len() != launches.len())
    {
        return Err("prefill graph split extent".into());
    }
    let mut per_rank = Vec::with_capacity(launches.len());
    for (rank, stages) in stages.iter().enumerate() {
        if stages.len() != launches[rank].len() {
            return Err("prefill graph stage extent".into());
        }
        // Each event: its first launch, its end and its join layer.
        let mut events: Vec<(usize, usize, Option<usize>)> = joins
            .map(|joins| {
                joins[rank]
                    .iter()
                    .enumerate()
                    .map(|(layer, &at)| (at, at, Some(layer)))
                    .collect()
            })
            .unwrap_or_default();
        let mut at = 0;
        while at < stages.len() {
            if !eager(instance, stages[at]) {
                at += 1;
                continue;
            }
            let first = at;
            while at < stages.len() && eager(instance, stages[at]) {
                at += 1;
            }
            events.push((first, at, None));
        }
        // A join precedes an eager section that starts at its consumer.
        events.sort_by_key(|&(first, _, join)| (first, join.is_none()));
        let mut graphs = Vec::new();
        let mut gaps = Vec::new();
        let mut start = 0;
        for (first, end, join) in events {
            if first < start {
                return Err("prefill graph gaps overlap".into());
            }
            graphs.push(start..first);
            gaps.push(match join {
                Some(layer) => Gap::Join(layer),
                None => Gap::Eager(std::iter::once(first..end).collect()),
            });
            start = end;
        }
        graphs.push(start..stages.len());
        per_rank.push((graphs, gaps));
    }
    let (first_graphs, first_gaps) = &per_rank[0];
    let mut split = Split {
        graphs: first_graphs.iter().map(|_| Vec::new()).collect(),
        gaps: first_gaps
            .iter()
            .map(|gap| match gap {
                Gap::Join(layer) => Gap::Join(*layer),
                Gap::Eager(_) => Gap::Eager(Vec::new()),
            })
            .collect(),
    };
    for (rank, (graphs, gaps)) in per_rank.into_iter().enumerate() {
        if graphs.len() != split.graphs.len() || gaps.len() != split.gaps.len() {
            return Err(format!("rank {rank} splits its prefill differently"));
        }
        for (into, range) in split.graphs.iter_mut().zip(graphs) {
            into.push(range);
        }
        for (into, gap) in split.gaps.iter_mut().zip(gaps) {
            match (into, gap) {
                (Gap::Join(a), Gap::Join(b)) if *a == b => {}
                (Gap::Eager(ranges), Gap::Eager(mut range)) => ranges.append(&mut range),
                _ => return Err(format!("rank {rank} splits its prefill differently")),
            }
        }
    }
    let check = |ranges: &[Range<usize>]| {
        let ops: Vec<_> = ranges
            .iter()
            .zip(launches)
            .map(|(range, program)| &program[range.clone()])
            .collect();
        let waits: Vec<_> = ranges
            .iter()
            .zip(collectives)
            .map(|(range, waits)| &waits[range.clone()])
            .collect();
        bundle_runtime::pipeline::check_collectives(&ops, &waits)
    };
    for ranges in &split.graphs {
        check(ranges).map_err(|e| format!("prefill graph segment collectives: {e}"))?;
    }
    for gap in &split.gaps {
        if let Gap::Eager(ranges) = gap {
            check(ranges).map_err(|e| format!("prefill eager section collectives: {e}"))?;
        }
    }
    Ok(split)
}

/// Splits a whole-program head: one segment.
pub(super) fn whole(programs: &[Vec<PreparedLaunch>]) -> Split {
    Split {
        graphs: vec![programs.iter().map(|program| 0..program.len()).collect()],
        gaps: Vec::new(),
    }
}

impl Instance {
    /// The program record of an upper-skip prefill, with diagnostic step
    /// logs: its rows and start, whether it may split into graph segments,
    /// and when it may, its segments, handles, eager and graph launches on
    /// rank 0 and its body and head keys' digests. Equal digests are equal
    /// programs, so a served log counts how often each would replay.
    pub(super) fn log_prefill_program(
        &self,
        run: &Run,
        plan: &skip::Plan,
        gathers: Option<&crate::instance::prefill_engram::Plan>,
    ) {
        let rows = run.inputs.live_tokens();
        let start = run.inputs.prefix_len.first().copied().unwrap_or(0);
        let mut record = format!(
            "{{\"schema\":\"chopinfer.deepseek41.prefill-program.v1\",\"rows\":{rows},\
             \"start\":{start},\"head_rows\":{},\"eligible\":{}",
            run.head_rows,
            eligible(run, plan)
        );
        if eligible(run, plan) {
            let joins = gathers.map(|gathers| gathers.joins());
            let keyed = split(self, plan.body(), joins.as_deref()).and_then(|split| {
                let body = Key::of(rows, &plan.body().0, &split)?;
                let head = Key::of(run.head_rows, &plan.head().0, &whole(&plan.head().0))?;
                Ok((split, body, head))
            });
            match keyed {
                Ok((split, body, head)) => {
                    let eager: usize = split
                        .gaps
                        .iter()
                        .map(|gap| match gap {
                            Gap::Eager(ranges) => ranges[0].len(),
                            Gap::Join(_) => 0,
                        })
                        .sum();
                    let graphs: usize = split.graphs.iter().map(|ranks| ranks[0].len()).sum();
                    record += &format!(
                        ",\"segments\":{},\"handles\":{},\"joins\":{},\"eager_launches\":{eager},\
                         \"graph_launches\":{graphs},\"head_launches\":{},\"body\":\"{}\",\
                         \"head\":\"{}\"",
                        split.graphs.len(),
                        split.handles(),
                        joins.map_or(0, |joins| joins[0].len()),
                        plan.head().0[0].len(),
                        body.digest(),
                        head.digest()
                    );
                }
                Err(error) => record += &format!(",\"error\":{error:?}"),
            }
        }
        // A diagnostic build also places the step: its slot and page table.
        #[cfg(feature = "diagnostic-taps")]
        if let (Some(slot), Some(pages)) =
            (run.inputs.slot_ids.first(), run.inputs.block_tables.first())
        {
            record += &format!(",\"slot\":{slot},\"pages\":{pages:?}");
        }
        record.push('}');
        let _ = self.services.log(record.as_bytes());
    }

    /// A diagnostic build created with `CHOPINFER_PREFILL_SCRATCH_POISON`
    /// set fills every rank's transient stage scratch (the members of
    /// `buffers::REGIONS`, which each stage writes before the next reads)
    /// with 0xff bytes, NaN in every float format, before an admitted short
    /// prefill of an instance without a drafter uploads anything: graphs
    /// and eager runs alike must not read what an earlier step left there.
    #[cfg(feature = "diagnostic-taps")]
    pub(super) fn poison_prefill_scratch(&self, run: &Run) -> Result<(), BundleAbiError> {
        use kernel_core::allocation::Traversal;
        const CHUNK: u64 = 16 << 20;
        let Some(poisoned) = &self.scratch_poison else {
            return Ok(());
        };
        let super::Execution::Skip { plan, .. } = &run.execution else {
            return Ok(());
        };
        if self.prefill_graphs.is_none() || self.dspark.is_some() || !admits(run, plan) {
            return Ok(());
        }
        let members = crate::buffers::REGIONS
            .iter()
            .flat_map(|region| region.members.iter().copied())
            .collect::<Vec<_>>();
        let pattern = vec![0xff_u8; CHUNK as usize];
        for (buffers, &(_, _, stream)) in self.buffers.iter().zip(&self.ranks) {
            let mut spans = Vec::new();
            buffers.scratch.visit_named(&mut |name, span| {
                if members.contains(&name) {
                    spans.push(*span);
                }
            });
            for span in spans {
                let mut at = 0;
                while at < span.length {
                    let bytes = (span.length - at).min(CHUNK);
                    let part = runtime_bundle_abi::DeviceBufferSpanV1::new(
                        span.buffer,
                        span.offset + at,
                        bytes,
                    );
                    bundle_runtime::resources::upload_bytes(
                        &self.services,
                        stream,
                        part,
                        &pattern[..bytes as usize],
                    )?;
                    at += bytes;
                }
            }
        }
        poisoned.set(poisoned.get() + 1);
        if poisoned.get().is_power_of_two() {
            self.log(&format!(
                "prefill scratch poisoned before {} admitted prefills",
                poisoned.get()
            ));
        }
        Ok(())
    }
}

impl Instance {
    /// Runs an eligible prefill's upper-skip plan (`eligible`) with its body
    /// and head as graphs: each replays when its key is captured, captures
    /// first when admitted, and otherwise runs as the eager plan does. The
    /// gaps between body segments join the prepared gathers and launch the
    /// eager selections in program order; the logits compaction copies run
    /// between the body and the head. A program that does not split runs
    /// eagerly.
    pub(super) fn launch_prefill_graphs(
        &self,
        run: &Run,
        plan: &skip::Plan,
        gathers: Option<&crate::instance::prefill_engram::Plan>,
    ) -> Result<(), BundleAbiError> {
        let Some(stores) = &self.prefill_graphs else {
            return plan.execute(self, gathers);
        };
        let streams: Vec<_> = self.ranks.iter().map(|rank| rank.2).collect();
        let rows = run.inputs.live_tokens();
        let joins = gathers.map(|gathers| gathers.joins());
        let (programs, collectives, _) = plan.body();
        let keyed = split(self, plan.body(), joins.as_deref()).and_then(|split| {
            let body = Key::of(rows, programs, &split)?;
            let head = Key::of(run.head_rows, &plan.head().0, &whole(&plan.head().0))?;
            Ok((split, body, head))
        });
        let (split, body, head) = match keyed {
            Ok(keyed) => keyed,
            Err(error) => {
                self.log(&format!(
                    "prefill graphs: {error}; the prefill runs eagerly"
                ));
                return plan.execute(self, gathers);
            }
        };
        let mut stores = stores.borrow_mut();
        let gap = |at: usize| match &split.gaps[at] {
            Gap::Join(layer) => self
                .prefill_engram
                .as_ref()
                .ok_or(DispatchError::InvalidPlan)?
                .join(&self.services, &streams, *layer)
                .map_err(DispatchError::from),
            Gap::Eager(ranges) => {
                let ops: Vec<_> = ranges
                    .iter()
                    .zip(programs)
                    .map(|(range, program)| &program[range.clone()])
                    .collect();
                let waits: Vec<_> = ranges
                    .iter()
                    .zip(collectives)
                    .map(|(range, waits)| &waits[range.clone()])
                    .collect();
                self.launch_eager(&streams, &ops, &waits)
            }
        };
        let segments = split.segments(programs);
        if !self.run_program(&mut stores.body, &streams, body, &segments, "prefill", gap)? {
            plan.execute_body(self, gathers)?;
        }
        plan.compact_rows(self)?;
        let head_split = whole(&plan.head().0);
        let segments = head_split.segments(&plan.head().0);
        let no_gap = |_| Err(DispatchError::InvalidPlan);
        if !self.run_program(
            &mut stores.head,
            &streams,
            head,
            &segments,
            "prefill-head",
            no_gap,
        )? {
            plan.execute_head(self)?;
        }
        Ok(())
    }

    /// Replays `key`'s program, capturing its `segments` first when the
    /// store admits it, with `gap` between segments. Returns false,
    /// launching nothing, when it must run eagerly instead.
    fn run_program(
        &self,
        store: &mut Store<Key, DeepSeek>,
        streams: &[BundleStreamHandleV1],
        key: Key,
        segments: &[Vec<&[PreparedLaunch]>],
        label: &'static str,
        gap: impl FnMut(usize) -> Result<(), DispatchError>,
    ) -> Result<bool, BundleAbiError> {
        let handles = segments
            .iter()
            .filter(|ranks| ranks.iter().any(|launches| !launches.is_empty()))
            .count();
        // A chunk that scores no row has no head to capture.
        if handles == 0 {
            return Ok(false);
        }
        let drain = || super::dma::drain_streams(self).map_err(DispatchError::from);
        let admitted = store
            .admit(&self.services, &self.graphs, &key, handles, drain)
            .map_err(|error| self.dispatch_failed(error))?;
        if self.config.execution.diagnostic_step_logs {
            let decision = match &admitted {
                Admission::Replay => "replay".to_owned(),
                Admission::Capture(_) => "capture".to_owned(),
                Admission::Eager(why) => format!("eager-{why:?}").to_lowercase(),
            };
            let record = format!(
                "{{\"schema\":\"chopinfer.deepseek41.prefill-graph.v1\",\"program\":\"{label}\",\
                 \"rows\":{},\"key\":\"{}\",\"decision\":\"{decision}\"}}",
                key.rows,
                key.digest()
            );
            let _ = self.services.log(record.as_bytes());
        }
        match admitted {
            Admission::Replay => {
                let stats = store.stats();
                if stats.replays.is_power_of_two() {
                    self.log(&format!(
                        "{label} graph replays: {}; {stats:?}",
                        stats.replays
                    ));
                }
            }
            Admission::Eager(why) => {
                let stats = store.stats();
                let denied = stats.full + stats.ledger;
                if matches!(why, Eager::Full | Eager::Ledger) && denied.is_power_of_two() {
                    self.log(&format!(
                        "{label} graphs: {} rows run eagerly ({why:?}); {stats:?}",
                        key.rows
                    ));
                }
                return Ok(false);
            }
            Admission::Capture(mut reservation) => {
                let started = std::time::Instant::now();
                let launches: usize = segments.iter().map(|ranks| ranks[0].len()).sum();
                let rows = key.rows;
                store
                    .capture(
                        &self.services,
                        &self.graphs,
                        key.clone(),
                        &mut reservation,
                        streams,
                        segments,
                        (label, Some(rows)),
                        |rank, launches| {
                            super::super::prefill_cache::launch(
                                &self.services,
                                streams[rank],
                                &self.symbols[rank],
                                launches,
                            )
                        },
                    )
                    .map_err(|error| self.dispatch_failed(error))?;
                self.log(&format!(
                    "{label} graph capture: {rows} rows, {handles} graphs and {launches} \
                     launches on rank 0, {:.3}s; {:?}",
                    started.elapsed().as_secs_f64(),
                    store.stats()
                ));
            }
        }
        let _unwinding = PoisonOnUnwind(&self.collectives);
        let replayed = store
            .replay(&self.services, &key, streams, &self.collectives, gap)
            .map_err(|error| self.dispatch_failed(error))?;
        if !replayed {
            return Err(self.dispatch_failed(DispatchError::InvalidPlan));
        }
        Ok(true)
    }
}
