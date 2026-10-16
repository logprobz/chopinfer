//! The context-parallel ops of an instance whose groups hold 2 or more
//! ranks (`crate::context_parallel`), each group rank's in group rank
//! order: the localization of a step's appended positions and of the key
//! lists to their owners, the exact merge of the group's selections, the
//! merge of the partial blocks each rank receives, and each bucket's group
//! transfers with the barrier that ends a piece.
//!
//! The per-rank selections and key writes are the indexer's
//! (`super::indexer`), the partial attention the latent family's in
//! partial mode (`super::latent`). The localizations, the merges and the
//! group transfers serve every phase and ask what every phase needs
//! (`exact`); the barrier, whose word no op reads, is asked for its rows
//! alone.

use kernel_core::backend::{Backend, Need, Prepared, Refused, View};
use kernel_core::context_parallel::{
    ContextParallelQuery, GroupTransfer, GroupTransferBuffers, GroupTransferOp,
    LocalizeKeysBuffers, PartialMergeBuffers,
};
use kernel_core::contract::StaticQuery;
use kernel_core::pooled_indexer::IndexerMergeBuffers;
use model_glm5next_contract::context_parallel::{BARRIER_WORDS, Dims, MAX_BUCKET, buckets};

use super::{Asks, Plan, Served, exact};

/// The one served query `pick` keeps.
fn one(
    served: &Served,
    what: &str,
    pick: impl Fn(&ContextParallelQuery) -> bool,
) -> Result<ContextParallelQuery, Refused> {
    let found = served.find(|query| match query {
        StaticQuery::ContextParallel(query) if pick(query) => Some(*query),
        _ => None,
    });
    match found[..] {
        [query] => Ok(query),
        _ => Err(Refused::Rule(format!(
            "GLM 5.3 Flash's context-parallel {what} has {} served queries",
            found.len()
        ))),
    }
}

/// A query's capacity.
fn capacity(query: &ContextParallelQuery) -> u32 {
    match *query {
        ContextParallelQuery::PartialMerge { capacity, .. }
        | ContextParallelQuery::TopKMerge { capacity, .. }
        | ContextParallelQuery::LocalizeKeys { capacity, .. }
        | ContextParallelQuery::PagedAttentionPartials { capacity, .. }
        | ContextParallelQuery::SelectedAttentionPartials { capacity, .. }
        | ContextParallelQuery::TopKMergeBlocks { capacity, .. } => capacity,
        ContextParallelQuery::OwnedQkNormRopeKvWrite { .. }
        | ContextParallelQuery::OwnedIndexerKeyWrite { .. } => 0,
    }
}

/// The group transfers of one bucket of rows (`context_parallel::bucket`),
/// each group rank's: the gathers of the selections' pairs and of the
/// absorbed queries, and the exchange of the partial blocks.
pub(crate) struct GroupTransfers {
    pub(crate) pairs: Vec<Prepared<GroupTransferBuffers<View>>>,
    pub(crate) queries: Vec<Prepared<GroupTransferBuffers<View>>>,
    pub(crate) partials: Vec<Prepared<GroupTransferBuffers<View>>>,
}

/// The context-parallel ops of an instance in groups of 2 or 4 ranks.
pub(crate) struct CpOps {
    /// Each group rank's localization of a step's appended positions.
    pub(crate) positions: Vec<Prepared<LocalizeKeysBuffers<View>>>,
    /// Each group rank's localization of the attention's key lists.
    pub(crate) lists: Vec<Prepared<LocalizeKeysBuffers<View>>>,
    /// The merge of the group's (score, pool) pairs into the key lists.
    pub(crate) merge: Prepared<IndexerMergeBuffers<View>>,
    /// The merge of the rank's own heads over the group's partial blocks.
    pub(crate) partial_merge: Prepared<PartialMergeBuffers<View>>,
    /// By bucket, as `context_parallel::buckets`.
    pub(crate) buckets: Vec<GroupTransfers>,
    /// Each group rank's barrier: a gather of one word no op reads.
    pub(crate) barrier: Vec<Prepared<GroupTransferBuffers<View>>>,
}

impl CpOps {
    /// The ops of `plan`'s groups, `None` at size 1.
    pub(crate) fn prepare(
        backend: &dyn Backend,
        served: &Served,
        plan: Plan,
        asks: &Asks,
    ) -> Result<Option<CpOps>, Refused> {
        let size = plan.context_parallel;
        if size < 2 {
            return Ok(None);
        }
        let every_phase =
            |rows: u32| exact::need(rows, exact::every_phase(exact::verify_rows().min(rows)));
        let localize = |what: &str, width: u32| {
            let query = one(served, "key localization", |query| {
                matches!(*query, ContextParallelQuery::LocalizeKeys { ranks, width: w, .. }
                    if ranks == size && w == width)
            })?;
            (0..size)
                .map(|rank| {
                    asks.ask(
                        format!("cp.localize {what} rank {rank}"),
                        every_phase(capacity(&query)),
                        |need| backend.localize_keys(&query, rank, need),
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        };
        let merge = one(
            served,
            "selection merge",
            |query| matches!(*query, ContextParallelQuery::TopKMerge { ranks, .. } if ranks == size),
        )?;
        let partial_merge = one(
            served,
            "partial merge",
            |query| matches!(*query, ContextParallelQuery::PartialMerge { ranks, .. } if ranks == size),
        )?;
        let dims = Dims::of(plan.shape).group();
        let transfers = |what: &str, transfer, words, rows| {
            let op = GroupTransferOp {
                transfer,
                size,
                words,
            };
            (0..size)
                .map(|rank| {
                    let need = match transfer {
                        GroupTransfer::Barrier => Need::rows(rows),
                        _ => every_phase(rows),
                    };
                    asks.ask(format!("cp.{what} rank {rank}"), need, |need| {
                        backend.group_transfer(&op, rank, need)
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(Some(CpOps {
            positions: localize("positions", 1)?,
            lists: localize("lists", plan.shape.latent_keys())?,
            merge: asks.ask("cp.merge", every_phase(capacity(&merge)), |need| {
                backend.indexer_merge(&merge, plan.index_topk, need)
            })?,
            partial_merge: asks.ask(
                "cp.partial_merge",
                every_phase(capacity(&partial_merge)),
                |need| backend.partial_merge(&partial_merge, need),
            )?,
            buckets: buckets()
                .map(|bucket| {
                    Ok(GroupTransfers {
                        pairs: transfers(
                            &format!("pairs {bucket}"),
                            GroupTransfer::Gather,
                            dims.pair_words(bucket),
                            bucket,
                        )?,
                        queries: transfers(
                            &format!("queries {bucket}"),
                            GroupTransfer::Gather,
                            dims.query_words(bucket),
                            bucket,
                        )?,
                        partials: transfers(
                            &format!("partials {bucket}"),
                            GroupTransfer::Exchange,
                            dims.block_floats(bucket),
                            bucket,
                        )?,
                    })
                })
                .collect::<Result<_, Refused>>()?,
            barrier: transfers("barrier", GroupTransfer::Barrier, BARRIER_WORDS, MAX_BUCKET)?,
        }))
    }
}

#[cfg(test)]
mod tests {
    use kernel_core::context_parallel::{LocalizeKeysBuffers, PartialMergeBuffers};
    use kernel_core::properties::Invariance;
    use model_glm5next_contract::contract::Shape;
    use model_glm5next_contract::quantization::FP8_BLOCK128;

    use super::*;

    /// A backend whose localizations and partial merges keep no claims,
    /// as it would without their declarations: each refuses exactness and
    /// otherwise prepares on the backend it wraps.
    struct Undeclared<'a>(&'a dyn Backend);

    fn inexact(need: Need) -> Result<(), Refused> {
        match need.exact == Invariance::NONE {
            true => Ok(()),
            false => Err(Refused::Inexact {
                needed: need.exact,
                guaranteed: Invariance::NONE,
            }),
        }
    }

    impl Backend for Undeclared<'_> {
        fn name(&self) -> &str {
            "undeclared"
        }

        fn localize_keys(
            &self,
            op: &ContextParallelQuery,
            group_rank: u32,
            need: Need,
        ) -> Result<Prepared<LocalizeKeysBuffers<View>>, Refused> {
            inexact(need)?;
            self.0.localize_keys(op, group_rank, need)
        }

        fn partial_merge(
            &self,
            op: &ContextParallelQuery,
            need: Need,
        ) -> Result<Prepared<PartialMergeBuffers<View>>, Refused> {
            inexact(need)?;
            self.0.partial_merge(op, need)
        }

        fn indexer_merge(
            &self,
            op: &ContextParallelQuery,
            top_k: u32,
            need: Need,
        ) -> Result<Prepared<IndexerMergeBuffers<View>>, Refused> {
            self.0.indexer_merge(op, top_k, need)
        }

        fn group_transfer(
            &self,
            op: &GroupTransferOp,
            group_rank: u32,
            need: Need,
        ) -> Result<Prepared<GroupTransferBuffers<View>>, Refused> {
            self.0.group_transfer(op, group_rank, need)
        }
    }

    /// The localizations and the partial merge ask what every phase needs
    /// at both group sizes, granted by their entries' declarations, and
    /// create refuses entries that do not keep it.
    #[test]
    fn localizations_and_partial_merges_ask_what_every_phase_needs() {
        let shape = Shape::TP4;
        let target = crate::sm120::own_ops::Sm120Own::new(shape);
        let (backend, served) = crate::target::Target::backend(&target, &FP8_BLOCK128).unwrap();
        for size in model_glm5next_contract::contract::CONTEXT_PARALLEL_SIZES {
            let plan = Plan {
                shape,
                layout: &FP8_BLOCK128,
                index_topk: shape.profile.index_topk_max,
                context_parallel: size,
            };
            if size < 2 {
                continue;
            }
            let asks = Asks::default();
            CpOps::prepare(backend.as_ref(), &served, plan, &asks).unwrap();
            let asked = asks.asked();
            let ours = asked
                .iter()
                .filter(|asked| {
                    asked.what.starts_with("cp.localize") || asked.what == "cp.partial_merge"
                })
                .collect::<Vec<_>>();
            assert_eq!(ours.len(), 2 * size as usize + 1, "size {size}");
            for asked in ours {
                assert_eq!(
                    asked.need.exact,
                    exact::every_phase(exact::verify_rows()),
                    "{}",
                    asked.what
                );
                assert!(asked.granted.provides(asked.need.exact), "{}", asked.what);
            }
            let undeclared = Undeclared(backend.as_ref());
            let refused = CpOps::prepare(&undeclared, &served, plan, &Asks::default());
            assert!(
                matches!(refused, Err(Refused::Inexact { .. })),
                "size {size}: {:?}",
                refused.err()
            );
        }
    }
}
