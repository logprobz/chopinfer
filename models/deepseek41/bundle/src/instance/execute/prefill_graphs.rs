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

use std::ops::Range;

use bundle_runtime::graphs::programs::{Fingerprint, fingerprint};
use bundle_runtime::pipeline::PreparedLaunch;
use model_deepseek41_contract::indexer;

use super::{PREFILL, Programs, Run, skip};
use crate::instance::Instance;
use crate::ops::indexer::REINDEX_BLOCK;
use crate::program::Stage;

/// The most rows of an eligible prefill.
pub(super) const MAX_ROWS: u32 = 511;

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
        record.push('}');
        let _ = self.services.log(record.as_bytes());
    }
}
