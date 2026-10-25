//! The embedding gather (token embedding, DFlash's embedding and prefill's
//! head-row gather) and the drafters' sharded argmax.

use kernel_core::backend::{Backend, Counts, Need, Prepared, Refused, View};
use kernel_core::contract::{ArgmaxLayout, DType, Refusal, RowQuery};
use kernel_core::rows::{ArgmaxFinalBuffers, ArgmaxPairs, ArgmaxPartialBuffers, EmbeddingBuffers};
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Bind, ByPhase, bucket_of, need, no_workspace, one};
use crate::program_lower::Lowered;
use crate::selection::{MimoPhase as _, Phase, RowBuckets};
use crate::switches::Switches;
use crate::topology::Tp;

type Partial = Prepared<ArgmaxPartialBuffers<View>>;

/// The views of an embedding gather: `output` row `r` is row `ids[r]` of
/// `table`.
pub struct EmbeddingViews {
    pub table: DeviceBufferSpanV1,
    pub ids: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
}

/// The gathers of whole rows of a `vocab`-row table of hidden-width rows,
/// one per phase: the token embedding, whose table is the vocabulary, and
/// prefill's head-row gather, whose table is the step's hidden rows.
#[derive(Default)]
pub struct Embeddings {
    ops: ByPhase<Prepared<EmbeddingBuffers<View>>>,
    vocab: u32,
    row_buckets: RowBuckets,
}

impl Embeddings {
    /// The gathers of `tp`'s embedding at every phase, with what `switches`
    /// need of the phase, gridding the target's `row_buckets` at prefill.
    pub fn prepare(
        backend: &dyn Backend,
        tp: Tp,
        switches: Switches,
        row_buckets: RowBuckets,
    ) -> Result<Self, Refused> {
        let g = tp.geometry;
        let query = RowQuery::EmbeddingGather {
            vocab: g.vocab(),
            width: g.hidden(),
        };
        Ok(Self {
            ops: ByPhase::prepare(|phase| {
                no_workspace(backend.embedding_gather(&query, 0, need(phase, switches)?)?).map(Some)
            })?,
            vocab: g.vocab(),
            row_buckets,
        })
    }

    /// The embedding of a `phase` step's `live_tokens` token ids, which its
    /// ids and output hold. It grids the phase's row bucket
    /// (`MimoPhase::row_grid`), which it records as its rows.
    pub fn tokens(
        &self,
        phase: Phase,
        switches: Switches,
        live_tokens: u32,
        views: &EmbeddingViews,
    ) -> Result<Lowered, Refusal> {
        let op = self.ops.get(phase, bucket_of(phase)?)?;
        let counts = Counts {
            padded_rows: phase.row_grid(live_tokens, switches, self.row_buckets)?,
            ..Counts::rows(live_tokens)
        };
        launch(op, counts, views, self.vocab)
    }

    /// Prefill's copy of `rows` of the step's `table_rows` hidden rows, by
    /// the rows' ids, into consecutive output rows. It grids the rows'
    /// bucket and records the rows.
    pub fn head_rows(
        &self,
        rows: u32,
        table_rows: u32,
        views: &EmbeddingViews,
    ) -> Result<Lowered, Refusal> {
        if rows == 0 || rows > table_rows {
            return Err(Refusal("MiMo row gather geometry"));
        }
        let op = self.ops.get(Phase::Prefill, bucket_of(Phase::Prefill)?)?;
        launch(op, Counts::rows(rows), views, table_rows)
    }
}

fn launch(
    op: &Prepared<EmbeddingBuffers<View>>,
    counts: Counts,
    views: &EmbeddingViews,
    table_rows: u32,
) -> Result<Lowered, Refusal> {
    let mut bind = Bind::default();
    let buffers = EmbeddingBuffers {
        table: bind.view(views.table),
        ids: bind.view(views.ids),
        output: bind.view(views.output),
        live_rows: bind.view(views.live_rows),
        table_rows,
    };
    one(bind.lower(op.as_ref(), &buffers, counts)?)
}

/// One rank's sharded argmax partials: the MTP drafter's, over up to
/// `DRAFT_ITEMS` draft rows, and a DFlash instance's, over its verify
/// buckets.
pub(super) struct Partials {
    mtp: Partial,
    dflash: ByPhase<Partial>,
}

/// The drafters' sharded argmax: each rank's partials and the final over
/// every rank's pairs.
#[derive(Default)]
pub(super) struct Argmax {
    partials: Vec<Partials>,
    last: Option<Prepared<ArgmaxFinalBuffers<View>>>,
}

impl Argmax {
    /// The argmax of `tp`'s vocabulary shards, with DFlash's partials when
    /// `dflash`. Drafter ops need their rows only.
    pub(super) fn prepare(backend: &dyn Backend, tp: Tp, dflash: bool) -> Result<Self, Refused> {
        let ranks = tp.ranks();
        let partial = RowQuery::ShardedArgmaxPartial {
            vocab_per_rank: crate::topology::vocab_per_rank(tp),
            ranks,
            logits: DType::Bf16,
        };
        let drafts = Need::rows(crate::contract::DRAFT_ITEMS);
        let partials = (0..ranks)
            .map(|rank| {
                Ok(Partials {
                    mtp: no_workspace(backend.sharded_argmax_partial(&partial, rank, drafts)?)?,
                    dflash: ByPhase::prepare(|phase| {
                        if !dflash || !matches!(phase, Phase::Verify(_)) {
                            return Ok(None);
                        }
                        let need = Need::rows(phase.capacity()?);
                        no_workspace(backend.sharded_argmax_partial(&partial, rank, need)?)
                            .map(Some)
                    })?,
                })
            })
            .collect::<Result<_, Refused>>()?;
        let last = RowQuery::ShardedArgmaxFinal {
            ranks,
            pairs: ArgmaxLayout::RowMajor,
        };
        let last = backend.sharded_argmax_final(&last, drafts)?;
        Ok(Self {
            partials,
            last: Some(no_workspace(last)?),
        })
    }
}

/// The views of a sharded argmax partial: the rank's BF16 logits rows,
/// their live width, and the pairs it writes.
pub struct PartialViews {
    pub logits: DeviceBufferSpanV1,
    pub live_width: DeviceBufferSpanV1,
    pub scores: DeviceBufferSpanV1,
    pub ids: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
}

fn partial(op: &Partial, rows: u32, views: &PartialViews, peer: bool) -> Result<Lowered, Refusal> {
    let mut bind = Bind::default();
    let buffers = ArgmaxPartialBuffers {
        logits: bind.view(views.logits),
        live_width: bind.view(views.live_width),
        pairs: ArgmaxPairs {
            scores: bind.view(views.scores),
            ids: bind.view(views.ids),
            peer,
        },
        live_rows: bind.view(views.live_rows),
    };
    one(bind.lower(op.as_ref(), &buffers, Counts::rows(rows))?)
}

impl super::Ops {
    /// The embedding of a `phase` step's `live_tokens` token ids from
    /// `views.table`, the vocabulary. It grids the phase's row bucket
    /// (`MimoPhase::row_grid`), which it records as its rows.
    pub fn embedding(
        &self,
        phase: Phase,
        switches: Switches,
        live_tokens: u32,
        views: &EmbeddingViews,
    ) -> Result<Lowered, Refusal> {
        self.embeddings.tokens(phase, switches, live_tokens, views)
    }

    /// Prefill's copy of `rows` of the step's `table_rows` hidden rows
    /// (`views.table`), by the rows' ids, into consecutive output rows. It
    /// grids the rows' bucket and records the rows.
    pub fn head_rows(
        &self,
        rows: u32,
        table_rows: u32,
        views: &EmbeddingViews,
    ) -> Result<Lowered, Refusal> {
        self.embeddings.head_rows(rows, table_rows, views)
    }

    /// Rank `rank`'s argmax partial of an MTP draft's `rows` rows, whose
    /// pairs are rank 0's memory that every rank writes through a peer
    /// lease.
    pub fn mtp_argmax_partial(
        &self,
        rank: u32,
        rows: u32,
        views: &PartialViews,
    ) -> Result<Lowered, Refusal> {
        let partials = self.partials(rank)?;
        partial(&partials.mtp, rows, views, true)
    }

    /// The argmax of an MTP draft's `rows` rows over every rank's pairs
    /// (`ranks` per row, in rank order, in rank 0's memory), into `output`.
    pub fn mtp_argmax_final(
        &self,
        rows: u32,
        scores: DeviceBufferSpanV1,
        ids: DeviceBufferSpanV1,
        output: DeviceBufferSpanV1,
        live_rows: DeviceBufferSpanV1,
    ) -> Result<Lowered, Refusal> {
        let op = self
            .argmax
            .last
            .as_ref()
            .ok_or(Refusal("MiMo has no prepared sharded argmax"))?;
        let mut bind = Bind::default();
        let buffers = ArgmaxFinalBuffers {
            pairs: ArgmaxPairs {
                scores: bind.view(scores),
                ids: bind.view(ids),
                peer: true,
            },
            output: Some(bind.view(output)),
            scatter: Vec::new(),
            live_rows: bind.view(live_rows),
        };
        one(bind.lower(op.as_ref(), &buffers, Counts::rows(rows))?)
    }

    /// Rank `rank`'s argmax partial of a DFlash `phase` step (bucket
    /// `bucket` of `COMM_BUCKETS`) into its own pairs. It grids and records
    /// the phase's capacity.
    pub fn dflash_argmax_partial(
        &self,
        rank: u32,
        phase: Phase,
        bucket: usize,
        views: &PartialViews,
    ) -> Result<Lowered, Refusal> {
        let op = self.partials(rank)?.dflash.get(phase, bucket)?;
        partial(op, phase.capacity()?, views, false)
    }

    fn partials(&self, rank: u32) -> Result<&Partials, Refusal> {
        self.argmax
            .partials
            .get(rank as usize)
            .ok_or(Refusal("MiMo has no sharded argmax for this rank"))
    }
}

#[cfg(test)]
mod tests {
    use kernel_core::contract::{Argument, RuntimeCounts};
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    use super::super::tests::{any_gathers, phases, resolved};
    use super::*;
    use crate::dispatch::BufferBindings;
    use crate::ops::{Ops, Plan};
    use crate::pipeline::PreparedLaunch;
    use crate::program_lower::launches;

    /// A span of its own buffer, large enough for any row op.
    fn span(buffer: u32) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(buffer, 1),
            256,
            1 << 34,
        )
    }

    /// A span of its own buffer of exactly `bytes`.
    fn exact(buffer: u32, bytes: u64) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(buffer, 1),
            256,
            bytes,
        )
    }

    fn prepared(tp: Tp, switches: Switches) -> Ops {
        Ops::prepare(
            &*crate::sm120::backend(tp, switches, true).unwrap(),
            &Plan {
                tp,
                switches,
                kv: kernel_core::contract::DType::Bf16,
                items: crate::contract::DECODE_ENTRY_ROWS,
                target: &crate::sm120::own_ops::Sm120Own::new(tp),
                state: &|_| [u64::MAX; 3],
                gathers: &any_gathers,
                dflash: Some([u64::MAX; 2]),
                own: crate::ops::OwnPlan::new(
                    &crate::sm120::own_ops::Sm120Own::new(tp),
                    crate::contract::DECODE_ENTRY_ROWS,
                ),
            },
        )
        .unwrap()
    }

    /// Every switch setting that moves an embedding launch: the prefill
    /// row grids.
    fn settings() -> [Switches; 2] {
        [false, true].map(|capacity_row_grids| Switches {
            capacity_row_grids,
            ..Switches::configured()
        })
    }

    /// The row counts compared up to `capacity` rows: every row bucket's
    /// least and most rows.
    fn counts(capacity: u32) -> Vec<u32> {
        let mut counts = vec![1, capacity];
        let mut bucket = 1;
        while bucket < capacity {
            counts.extend([bucket, bucket + 1]);
            bucket *= 2;
        }
        counts.sort_unstable();
        counts.dedup();
        counts
    }

    fn compare(lowered: Lowered, planned: PreparedLaunch, what: &str) {
        assert_eq!(
            resolved(&launches(vec![lowered])),
            resolved(&[planned]),
            "{what}"
        );
    }

    /// The prepared embeddings launch what MiMo's row builders planned: the
    /// token embedding (`row_plan::embedding`, which DFlash's embedding
    /// also built), through the instance's ops and through a lookup
    /// launcher's own backend, in both row grid settings at every phase,
    /// bucket and row grid, and prefill's head-row gather
    /// (`row_plan::gather_rows`) at every row bucket, at TP2 and TP4.
    #[test]
    fn embeddings_launch_what_the_row_builders_planned() {
        let mut compared = 0;
        for tp in Tp::SUPPORTED {
            let crate::sm120::selection::Selected::Row(entry) =
                crate::sm120::selection::row_for(tp, "embedding_full").unwrap()
            else {
                unreachable!()
            };
            let views = EmbeddingViews {
                table: span(1),
                ids: span(2),
                output: span(3),
                live_rows: span(4),
            };
            for switches in settings() {
                let ops = prepared(tp, switches);
                let lookup = Embeddings::prepare(
                    &*crate::sm120::lookup_backend(entry.clone()),
                    tp,
                    switches,
                    crate::sm120::ROW_BUCKETS,
                )
                .unwrap();
                for (phase, _) in phases() {
                    for live in counts(phase.capacity().unwrap()) {
                        // A step's ids and output hold only its rows.
                        let width = u64::from(tp.geometry.hidden()) * 2;
                        let held = EmbeddingViews {
                            ids: exact(2, u64::from(live) * 4),
                            output: exact(3, u64::from(live) * width),
                            ..views
                        };
                        let planned = || {
                            crate::row_plan::embedding(
                                tp,
                                phase,
                                switches,
                                live,
                                held.table,
                                held.ids,
                                held.output,
                                held.live_rows,
                            )
                            .unwrap()
                        };
                        let what = format!("{tp:?} {switches:?} {phase:?} {live}");
                        compare(
                            ops.embedding(phase, switches, live, &held).unwrap(),
                            planned(),
                            &what,
                        );
                        compare(
                            lookup.tokens(phase, switches, live, &held).unwrap(),
                            planned(),
                            &what,
                        );
                        compared += 2;
                    }
                }
            }
            // The head-row gather grids its rows' bucket in both settings.
            // Under the capacity row grids its builder gridded the capacity:
            // the one launch the labelled grid change moves, whose kernel
            // writes only the live rows, so its bits are equal.
            let capacity = crate::contract::PREFILL_TOKENS;
            for switches in settings() {
                let ops = prepared(tp, switches);
                for rows in counts(capacity) {
                    for table_rows in [rows, capacity] {
                        let planned = |switches| {
                            crate::row_plan::gather_rows(
                                tp,
                                switches,
                                rows,
                                span(1),
                                span(2),
                                span(3),
                                span(4),
                                table_rows,
                            )
                            .unwrap()
                        };
                        let bucket = Switches {
                            capacity_row_grids: false,
                            ..switches
                        };
                        let lowered = ops.head_rows(rows, table_rows, &views).unwrap();
                        if switches.capacity_row_grids {
                            let mut gridded = planned(switches);
                            assert_eq!(gridded.plan.grid[0], capacity);
                            gridded.plan.grid = planned(bucket).plan.grid;
                            compare(lowered, gridded, &format!("{tp:?} {rows} capacity"));
                        } else {
                            let what = format!("{tp:?} {rows} of {table_rows}");
                            compare(lowered, planned(bucket), &what);
                        }
                        compared += 1;
                    }
                }
            }
            let ops = prepared(tp, Switches::configured());
            assert!(ops.head_rows(0, 1, &views).is_err());
            assert!(ops.head_rows(2, 1, &views).is_err());
            assert!(ops.head_rows(capacity + 1, capacity + 1, &views).is_err());
        }
        assert!(compared > 1000, "{compared}");
    }

    /// The embedding checks its token ids and output at the step's rows,
    /// which they hold, while it grids the phase's row bucket, and refuses
    /// a table short of the vocabulary.
    #[test]
    fn embeddings_check_their_views_at_the_step_rows() {
        let tp = Tp::TP4;
        let switches = Switches {
            capacity_row_grids: false,
            ..Switches::configured()
        };
        let ops = prepared(tp, switches);
        let width = u64::from(tp.geometry.hidden()) * 2;
        let table = u64::from(tp.geometry.vocab()) * width;
        let phase = Phase::Prefill;
        // The ids and output hold the step's five rows; it grids eight.
        assert_eq!(
            phase.row_grid(5, switches, crate::sm120::ROW_BUCKETS),
            Ok(8)
        );
        let grid = 5;
        let views = |table, ids, output| EmbeddingViews {
            table: exact(1, table),
            ids: exact(2, ids),
            output: exact(3, output),
            live_rows: exact(4, 4),
        };
        assert!(
            ops.embedding(phase, switches, 5, &views(table, grid * 4, grid * width))
                .is_ok()
        );
        for short in [
            views(table - 1, grid * 4, grid * width),
            views(table, grid * 4 - 1, grid * width),
            views(table, grid * 4, grid * width - 1),
        ] {
            assert!(ops.embedding(phase, switches, 5, &short).is_err());
        }
    }

    /// DFlash's argmax partial as its program planned it: the rank's entry
    /// over the phase's capacity, its own pairs.
    fn planned_dflash_partial(
        tp: Tp,
        rank: u32,
        phase: Phase,
        live: u32,
        views: &PartialViews,
    ) -> PreparedLaunch {
        let symbol = crate::sm120::selection::drafter_row_symbol(tp, "dflash_argmax_partial");
        let entry = crate::sm120::launch::row::entries_for(tp)
            .find(|entry| entry.symbol == symbol)
            .unwrap();
        let mut bindings = BufferBindings::default();
        let arguments = vec![
            Argument::DeviceBuffer(bindings.bind(views.logits)),
            Argument::DeviceBuffer(bindings.bind(views.scores)),
            Argument::DeviceBuffer(bindings.bind(views.ids)),
            Argument::DeviceBuffer(bindings.bind(views.live_rows)),
            Argument::DeviceBuffer(bindings.bind(views.live_width)),
            Argument::U32(rank * crate::topology::vocab_per_rank(tp)),
        ];
        let capacity = phase.capacity().unwrap();
        let (plan, arguments) = crate::sm120::launch::row::plan(
            entry,
            capacity,
            RuntimeCounts {
                m: live,
                batch: 1,
                context: 1,
                routed_rows: 8,
                split_k: 1,
            },
            arguments,
        )
        .unwrap();
        PreparedLaunch {
            symbol,
            rows: capacity,
            plan,
            arguments,
            bindings,
        }
    }

    /// The prepared sharded argmax launches what the drafters planned: the
    /// MTP partial of every rank and the final at every draft row count
    /// (`mtp::argmax`), and DFlash's partial of every rank at every verify
    /// bucket, at TP2 and TP4.
    #[test]
    fn sharded_argmax_launches_what_the_drafters_planned() {
        let views = PartialViews {
            logits: span(1),
            live_width: span(2),
            scores: span(3),
            ids: span(4),
            live_rows: span(5),
        };
        let mut compared = 0;
        for tp in Tp::SUPPORTED {
            let ops = prepared(tp, Switches::configured());
            for rows in 1..=crate::contract::DRAFT_ITEMS {
                for rank in 0..tp.ranks() {
                    let planned = crate::mtp::argmax::partial_batch(
                        tp,
                        rank,
                        span(1),
                        span(3),
                        span(4),
                        span(5),
                        span(2),
                        rows,
                    )
                    .unwrap();
                    compare(
                        ops.mtp_argmax_partial(rank, rows, &views).unwrap(),
                        planned,
                        &format!("{tp:?} rank {rank} {rows}"),
                    );
                }
                let planned =
                    crate::mtp::argmax::finish_batch(tp, span(3), span(4), span(1), span(5), rows)
                        .unwrap();
                compare(
                    ops.mtp_argmax_final(rows, span(3), span(4), span(1), span(5))
                        .unwrap(),
                    planned,
                    &format!("{tp:?} final {rows}"),
                );
                compared += 1;
            }
            let rows = crate::contract::DRAFT_ITEMS + 1;
            assert!(ops.mtp_argmax_partial(0, rows, &views).is_err());
            assert!(ops.mtp_argmax_partial(tp.ranks(), 1, &views).is_err());
            assert!(
                ops.mtp_argmax_final(rows, span(3), span(4), span(1), span(5))
                    .is_err()
            );
            for (phase, bucket) in phases() {
                let Phase::Verify(_) = phase else {
                    assert!(ops.dflash_argmax_partial(0, phase, bucket, &views).is_err());
                    continue;
                };
                for rank in 0..tp.ranks() {
                    compare(
                        ops.dflash_argmax_partial(rank, phase, bucket, &views)
                            .unwrap(),
                        planned_dflash_partial(tp, rank, phase, 1, &views),
                        &format!("{tp:?} rank {rank} {phase:?}"),
                    );
                    compared += 1;
                }
            }
        }
        assert!(compared > 64, "{compared}");
    }

    /// Every symbol a prepared embedding or argmax launches is in a cubin a
    /// `tp` instance loads.
    #[test]
    fn row_ops_launch_only_loaded_entries() {
        for tp in Tp::SUPPORTED {
            let loaded = crate::CUBINS
                .iter()
                .filter(|cubin| crate::topology::cubin_serves(cubin.name, tp))
                .flat_map(|cubin| cubin.symbols.iter().copied())
                .collect::<std::collections::BTreeSet<_>>();
            let ops = prepared(tp, Switches::configured());
            let mut symbols = Vec::new();
            for (phase, bucket) in phases() {
                symbols.extend(ops.embeddings.ops.get(phase, bucket).map(|op| op.info()));
                for partials in &ops.argmax.partials {
                    symbols.extend(partials.dflash.get(phase, bucket).map(|op| op.info()));
                }
            }
            for partials in &ops.argmax.partials {
                symbols.push(partials.mtp.info());
            }
            symbols.extend(ops.argmax.last.as_ref().map(|op| op.info()));
            assert!(symbols.len() > 8);
            for info in symbols {
                for (symbol, _) in &info.symbols {
                    assert!(loaded.contains(symbol.as_str()), "{symbol} at {tp:?}");
                }
            }
        }
    }
}
