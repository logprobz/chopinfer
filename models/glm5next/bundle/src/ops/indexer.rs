//! The pooled indexer's ops: each phase's key write at every group rank,
//! the selection of every group rank (the global selection without
//! context-parallel groups, each rank's pairs with them), and the draft
//! share compaction.
//!
//! Each op is the model's query at the capacity its served entry compiles
//! (`Served`). The backend runs a selection's one-split, tree or radix form
//! where its rows take one (`Sm120::with_indexer_*`), all of the same bits.

use kernel_core::backend::{Backend, Need, Prepared, Refused, View};
use kernel_core::contract::{RecurrentPhase, StaticQuery};
use kernel_core::pooled_indexer::{
    IndexerKeyWriteBuffers, IndexerSelectBuffers, IndexerSelectionMode, PooledIndexerQuery,
    SelectionCompactBuffers,
};

use super::{Asks, Plan, Served, exact};

/// The pooled indexer ops of an instance.
pub(crate) struct IndexerOps {
    /// By the served phase (prefill, decode, verify), each group rank's key
    /// write.
    key_writes: [Vec<Prepared<IndexerKeyWriteBuffers<View>>>; 3],
    /// Each group rank's selection.
    selections: Vec<Prepared<IndexerSelectBuffers<View>>>,
    /// The draft step's share of the extend pass's key lists.
    pub(crate) compact: Prepared<SelectionCompactBuffers<View>>,
}

/// The one served query `pick` keeps.
fn one(
    served: &Served,
    what: &str,
    pick: impl Fn(&PooledIndexerQuery) -> bool,
) -> Result<PooledIndexerQuery, Refused> {
    let found = served.find(|query| match query {
        StaticQuery::PooledIndexer(query) if pick(query) => Some(*query),
        _ => None,
    });
    match found[..] {
        [query] => Ok(query),
        _ => Err(Refused::Rule(format!(
            "GLM 5.3 Flash's indexer {what} has {} served queries",
            found.len()
        ))),
    }
}

/// A query's capacity.
fn capacity(query: &PooledIndexerQuery) -> u32 {
    match *query {
        PooledIndexerQuery::KeyWrite { capacity, .. }
        | PooledIndexerQuery::Select { capacity, .. }
        | PooledIndexerQuery::HeadSumSelect { capacity, .. }
        | PooledIndexerQuery::CompactSelection { capacity, .. } => capacity,
    }
}

/// The index of a key write of `phase`.
fn phase_index(phase: RecurrentPhase) -> usize {
    match phase {
        RecurrentPhase::Prefill => 0,
        RecurrentPhase::Decode => 1,
        RecurrentPhase::Verify { .. } => 2,
    }
}

impl IndexerOps {
    pub(crate) fn prepare(
        backend: &dyn Backend,
        served: &Served,
        plan: Plan,
        asks: &Asks,
    ) -> Result<IndexerOps, Refused> {
        let size = plan.context_parallel;
        let p = plan.shape.profile;
        let key_write = |phase: RecurrentPhase| {
            let query = one(served, "key write", |query| {
                matches!(*query, PooledIndexerQuery::KeyWrite { phase: at, pooling, .. }
                    if phase_index(at) == phase_index(phase)
                        && pooling.width == p.indexer_dim
                        && pooling.group == p.index_pool)
            })?;
            (0..size)
                .map(|rank| {
                    let what = format!("indexer.key_write {phase:?} rank {rank}");
                    let rows = capacity(&query);
                    let need = exact::need(rows, exact::with_state(exact::of_phase(phase, rows)));
                    asks.ask(what, need, |need| {
                        backend.indexer_key_write(&query, rank, size, need)
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        };
        let mode = if size > 1 {
            IndexerSelectionMode::PerRank { ranks: size }
        } else {
            IndexerSelectionMode::Global
        };
        let select = one(served, "selection", |query| {
            matches!(*query, PooledIndexerQuery::Select { heads, head_dim, mode: m, .. }
                if heads == p.indexer_heads && head_dim == p.indexer_dim && m == mode)
        })?;
        let compact = one(served, "compaction", |query| {
            matches!(query, PooledIndexerQuery::CompactSelection { .. })
        })?;
        Ok(IndexerOps {
            key_writes: [
                key_write(RecurrentPhase::Prefill)?,
                key_write(RecurrentPhase::Decode)?,
                key_write(RecurrentPhase::Verify { max_rows: 1 })?,
            ],
            selections: (0..size)
                .map(|rank| {
                    let what = format!("indexer.select rank {rank}");
                    let rows = capacity(&select);
                    let exact = exact::every_phase(exact::verify_rows().min(rows));
                    asks.ask(what, exact::need(rows, exact), |need| {
                        backend.indexer_select(&select, plan.index_topk, rank, need)
                    })
                })
                .collect::<Result<_, _>>()?,
            compact: asks.ask("indexer.compact", Need::rows(capacity(&compact)), |need| {
                backend.indexer_compact(&compact, need)
            })?,
        })
    }

    /// Group rank `rank`'s key write of the served phase `phase`.
    pub(crate) fn key_write(
        &self,
        phase: RecurrentPhase,
        rank: u32,
    ) -> Option<&Prepared<IndexerKeyWriteBuffers<View>>> {
        self.key_writes[phase_index(phase)].get(rank as usize)
    }

    /// Group rank `rank`'s selection.
    pub(crate) fn select(&self, rank: u32) -> Option<&Prepared<IndexerSelectBuffers<View>>> {
        self.selections.get(rank as usize)
    }
}
