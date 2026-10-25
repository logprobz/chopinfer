//! The logits gathers onto rank 0, the logits row exchange, and the DFlash
//! candidate all-gather: collectives that move rows and compute nothing.

use kernel_core::backend::{Backend, Counts, Need, Prepared, Refused, View};
use kernel_core::collective::{GatherBuffers, RankViews, RowGatherBuffers, ShardedRows};
use kernel_core::contract::{CommQuery, DType, Refusal};
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Bind, ByPhase, need, no_workspace};
use crate::program_lower::Lowered;
use crate::selection::{MimoPhase as _, Phase};
use crate::switches::Switches;
use crate::topology::Tp;

type RowGather = Prepared<RowGatherBuffers<View>>;
type Gather = Prepared<GatherBuffers<View>>;

/// The logits dtypes a step gathers: BF16 rows, or F32 rows of scored
/// prompts and the F32 head.
const DTYPES: [DType; 2] = [DType::Bf16, DType::F32];

/// The views of a logits gather or exchange on one rank: its shard rows
/// and, per rank in rank order, the output, flags and epoch.
#[derive(Clone)]
pub struct LogitsBuffers {
    pub local: DeviceBufferSpanV1,
    /// Vocabulary rows of each rank's shard.
    pub shard: u32,
    pub outputs: Vec<DeviceBufferSpanV1>,
    pub flags: Vec<DeviceBufferSpanV1>,
    pub epochs: Vec<DeviceBufferSpanV1>,
}

/// The bytes of a rank's flag and epoch regions of a gather.
pub type State = [u64; 2];

/// One rank's gathers, by logits dtype (`DTYPES`) and phase.
#[derive(Default)]
pub(super) struct Gathers {
    /// The logits rows of every rank's shard onto rank 0.
    rows: [ByPhase<RowGather>; 2],
    /// The logits rows onto the ranks that own them, for the decode and
    /// verify buckets that hold the exchange's state.
    exchange: [ByPhase<Gather>; 2],
    /// A DFlash verify step's candidate pairs onto every rank.
    candidates: ByPhase<Gather>,
}

/// The comm state regions of one `COMM_BUCKETS` bucket that its gathers use.
pub struct Regions {
    /// The logits row gather's flags and epoch.
    pub rows: State,
    /// The logits row exchange's, in the buckets that hold one.
    pub exchange: Option<State>,
}

impl Gathers {
    /// Rank `rank`'s gathers of `tp`. `regions(bucket)` are the bucket's
    /// regions, and `candidates` the DFlash candidate all-gather's of a
    /// DFlash instance.
    pub(super) fn prepare(
        backend: &dyn Backend,
        tp: Tp,
        rank: u32,
        switches: Switches,
        regions: &dyn Fn(usize) -> Regions,
        candidates: Option<State>,
    ) -> Result<Self, Refused> {
        let logits = |dtype| ShardedRows {
            ranks: tp.ranks(),
            shard: u64::from(crate::topology::vocab_per_rank(tp)),
            dtype,
        };
        let rows = DTYPES.map(|dtype| {
            ByPhase::prepare(|phase| {
                let op = backend.gather_rows(&logits(dtype), rank, 0, need(phase, switches)?)?;
                fits(op, regions(super::bucket_of(phase)?).rows).map(Some)
            })
        });
        let exchange = DTYPES.map(|dtype| {
            ByPhase::prepare(|phase| {
                if matches!(phase, Phase::Prefill) {
                    return Ok(None);
                }
                let Some(state) = regions(super::bucket_of(phase)?).exchange else {
                    return Ok(None);
                };
                let op = backend.exchange_rows(&logits(dtype), rank, need(phase, switches)?)?;
                fits(op, state).map(Some)
            })
        });
        let candidates = ByPhase::prepare(|phase| {
            let (Some(state), Phase::Verify(_)) = (candidates, phase) else {
                return Ok(None);
            };
            let capacity = phase.capacity()?;
            let op = backend.all_gather(
                &CommQuery::AllGather {
                    ranks: tp.ranks(),
                    elements_per_rank: 2 * u64::from(capacity),
                    dtype: DType::F32,
                },
                rank,
                None,
                Need::rows(capacity),
            )?;
            fits(op, state).map(Some)
        })?;
        let [rows_bf16, rows_f32] = rows;
        let [exchange_bf16, exchange_f32] = exchange;
        Ok(Self {
            rows: [rows_bf16?, rows_f32?],
            exchange: [exchange_bf16?, exchange_f32?],
            candidates,
        })
    }
}

/// `op`, which must ask for no workspace and fit `state`'s regions.
fn fits<B>(op: Prepared<B>, state: State) -> Result<Prepared<B>, Refused> {
    let op = no_workspace(op)?;
    if op.info().state.len() != state.len()
        || op
            .info()
            .state
            .iter()
            .zip(state)
            .any(|(&op, held)| op > held)
    {
        return Err(Refused::Rule(
            "a MiMo gather needs more comm state than its region holds".into(),
        ));
    }
    Ok(op)
}

fn dtype_index(dtype: DType) -> Result<usize, Refusal> {
    DTYPES
        .iter()
        .position(|&gathered| gathered == dtype)
        .ok_or(Refusal("MiMo gathers BF16 and F32 logits only"))
}

/// Every rank's output and gather state, in rank order.
fn peers(
    bind: &mut Bind,
    outputs: &[DeviceBufferSpanV1],
    flags: &[DeviceBufferSpanV1],
    epochs: &[DeviceBufferSpanV1],
) -> Result<Vec<RankViews<View>>, Refusal> {
    if flags.len() != outputs.len() || epochs.len() != outputs.len() {
        return Err(Refusal("MiMo peer buffers do not cover the ranks"));
    }
    Ok((0..outputs.len())
        .map(|peer| RankViews {
            output: bind.view(outputs[peer]),
            state: vec![bind.view(flags[peer]), bind.view(epochs[peer])],
        })
        .collect())
}

impl super::Ops {
    /// Rank `rank`'s gather of the logits rows `rows` of a `phase` step
    /// (bucket `bucket` of `COMM_BUCKETS`) in `dtype`: onto rank 0, which
    /// moves rows `0..n` and so takes the rows in order, or onto the ranks
    /// that own them when `exchange`. Every rank's output holds them as
    /// whole vocabulary rows.
    #[allow(clippy::too_many_arguments)]
    pub fn gather_logits(
        &self,
        rank: u32,
        phase: Phase,
        bucket: usize,
        exchange: bool,
        dtype: DType,
        rows: &[u32],
        buffers: &LogitsBuffers,
    ) -> Result<Vec<Lowered>, Refusal> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        if !exchange
            && rows
                .iter()
                .enumerate()
                .any(|(index, &row)| row != index as u32)
        {
            return Err(Refusal("MiMo gathers logits rows in order"));
        }
        let rows = u32::try_from(rows.len()).map_err(|_| Refusal("MiMo logits rows overflow"))?;
        let gathers = self
            .gathers
            .get(rank as usize)
            .ok_or(Refusal("MiMo has no gathers for this rank"))?;
        let index = dtype_index(dtype)?;
        // Every rank receives whole rows, though the gather writes rank 0's.
        let element: u64 = if dtype == DType::F32 { 4 } else { 2 };
        let whole =
            u64::from(rows) * u64::from(buffers.shard) * element * buffers.outputs.len() as u64;
        if buffers.outputs.iter().any(|span| span.length < whole) {
            return Err(Refusal("MiMo logits row gather exceeds prepared buffers"));
        }
        let mut bind = Bind::default();
        let input = bind.view(buffers.local);
        let ranks = peers(&mut bind, &buffers.outputs, &buffers.flags, &buffers.epochs)?;
        let counts = Counts::rows(rows);
        let launches = if exchange {
            let op = gathers.exchange[index].get(phase, bucket)?;
            op.launches(&GatherBuffers { input, ranks }, View::EMPTY, counts)?
        } else {
            let op = gathers.rows[index].get(phase, bucket)?;
            let buffers = RowGatherBuffers {
                input,
                indices: None,
                ranks,
            };
            op.launches(&buffers, View::EMPTY, counts)?
        };
        Ok(bind.launches(launches))
    }

    /// Rank `rank`'s all-gather of a DFlash `phase` step's candidate pairs
    /// (bucket `bucket` of `COMM_BUCKETS`): its `pairs` onto every rank's
    /// `gathered`, with every rank's flags and epoch.
    #[allow(clippy::too_many_arguments)]
    pub fn gather_candidates(
        &self,
        rank: u32,
        phase: Phase,
        bucket: usize,
        pairs: DeviceBufferSpanV1,
        gathered: &[DeviceBufferSpanV1],
        flags: &[DeviceBufferSpanV1],
        epochs: &[DeviceBufferSpanV1],
    ) -> Result<Vec<Lowered>, Refusal> {
        let op = self
            .gathers
            .get(rank as usize)
            .ok_or(Refusal("MiMo has no gathers for this rank"))?
            .candidates
            .get(phase, bucket)?;
        let mut bind = Bind::default();
        let buffers = GatherBuffers {
            input: bind.view(pairs),
            ranks: peers(&mut bind, gathered, flags, epochs)?,
        };
        let counts = Counts::rows(phase.capacity()?);
        bind.lower(op.as_ref(), &buffers, counts)
    }
}

#[cfg(test)]
mod tests {
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    use super::super::tests::{any_gathers, phases, prepare_on, resolved};
    use super::*;
    use crate::ops::{GatherRegions, Ops};
    use crate::program_lower::launches;
    use crate::scratch::COMM_BUCKETS;

    /// A span of `buffer` large enough for any gather.
    fn span(buffer: u32, offset: u64) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(buffer, 1),
            offset,
            1 << 40,
        )
    }

    /// `tp`'s ops, with state regions that hold any gather and DFlash's
    /// candidate all-gathers.
    fn prepared(tp: Tp) -> Ops {
        let switches = Switches::configured();
        prepare_on(
            &*crate::sm120::backend(tp, switches, true).unwrap(),
            tp,
            switches,
            &|_| [u64::MAX; 3],
            &any_gathers,
            Some([u64::MAX; 2]),
        )
        .unwrap()
    }

    /// Every rank's views, each role in its own buffer.
    fn peers(ranks: u32, buffer: u32) -> Vec<DeviceBufferSpanV1> {
        (0..u64::from(ranks))
            .map(|rank| span(buffer, rank << 20))
            .collect()
    }

    /// The row counts compared at a phase of `capacity` rows: every one.
    fn counts(capacity: u32) -> Vec<u32> {
        (1..=capacity).collect()
    }

    /// The prepared logits gathers and exchanges launch what MiMo's comm
    /// builders planned (`comm_plan::gather_logits_rows`,
    /// `exchange_logits_rows`): symbol, rows, plan and resolved arguments, at
    /// every rank, phase, bucket, dtype and row count, at TP2 and TP4.
    #[test]
    fn logits_gathers_launch_what_mimo_planned() {
        for tp in Tp::SUPPORTED {
            let ops = prepared(tp);
            for rank in 0..tp.ranks() {
                let buffers = LogitsBuffers {
                    local: span(1, 0),
                    shard: crate::topology::vocab_per_rank(tp),
                    outputs: peers(tp.ranks(), 2),
                    flags: peers(tp.ranks(), 3),
                    epochs: peers(tp.ranks(), 4),
                };
                for (phase, bucket) in phases() {
                    for dtype in DTYPES {
                        for rows in counts(phase.capacity().unwrap()) {
                            let order = (0..rows).collect::<Vec<_>>();
                            let gather = |exchange| {
                                ops.gather_logits(
                                    rank, phase, bucket, exchange, dtype, &order, &buffers,
                                )
                                .map(|lowered| resolved(&launches(lowered)))
                            };
                            let planned = crate::sm120::comm_plan::gather_logits_rows(
                                &order,
                                rank,
                                dtype,
                                buffers.clone(),
                            )
                            .unwrap();
                            assert_eq!(
                                gather(false).unwrap(),
                                resolved(&planned),
                                "{tp:?} rank {rank} {phase:?} {dtype:?} {rows}"
                            );
                            if phase == Phase::Prefill {
                                assert!(gather(true).is_err());
                                continue;
                            }
                            let planned = crate::sm120::comm_plan::exchange_logits_rows(
                                rows,
                                rank,
                                dtype,
                                buffers.clone(),
                            )
                            .unwrap();
                            assert_eq!(
                                gather(true).unwrap(),
                                resolved(&planned),
                                "{tp:?} rank {rank} {phase:?} {dtype:?} {rows} exchange"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The row gather moves rows `0..n` only; an exchange moves any rows.
    #[test]
    fn logits_gathers_take_their_rows_in_order() {
        let tp = Tp::TP4;
        let ops = prepared(tp);
        let buffers = LogitsBuffers {
            local: span(1, 0),
            shard: crate::topology::vocab_per_rank(tp),
            outputs: peers(tp.ranks(), 2),
            flags: peers(tp.ranks(), 3),
            epochs: peers(tp.ranks(), 4),
        };
        let bucket = COMM_BUCKETS.iter().position(|&rows| rows == 8).unwrap();
        let gather = |exchange, rows: &[u32]| {
            ops.gather_logits(
                0,
                Phase::Decode(8),
                bucket,
                exchange,
                DType::Bf16,
                rows,
                &buffers,
            )
        };
        assert!(gather(false, &[0, 2]).is_err());
        assert!(gather(true, &[0, 2]).is_ok());
        assert!(gather(false, &[]).unwrap().is_empty());
        // Every rank's output holds whole rows.
        let short = LogitsBuffers {
            outputs: vec![
                DeviceBufferSpanV1::new(
                    BundleDeviceBufferHandleV1::from_parts(2, 1),
                    0,
                    8,
                );
                4
            ],
            ..buffers.clone()
        };
        assert!(
            ops.gather_logits(
                0,
                Phase::Decode(8),
                bucket,
                false,
                DType::Bf16,
                &[0],
                &short
            )
            .is_err()
        );
    }

    /// The prepared DFlash candidate all-gather launches what MiMo planned
    /// before (`sm120::comm::candidates`) at every rank and verify bucket.
    #[test]
    fn dflash_candidates_launch_what_mimo_planned() {
        for tp in Tp::SUPPORTED {
            let ops = prepared(tp);
            for rank in 0..tp.ranks() {
                for (phase, bucket) in phases() {
                    let gathered = peers(tp.ranks(), 2);
                    let flags = peers(tp.ranks(), 3);
                    let epochs = peers(tp.ranks(), 4);
                    let lowered = ops.gather_candidates(
                        rank,
                        phase,
                        bucket,
                        span(1, 0),
                        &gathered,
                        &flags,
                        &epochs,
                    );
                    let Phase::Verify(capacity) = phase else {
                        assert!(lowered.is_err());
                        continue;
                    };
                    let planned = crate::sm120::comm::candidates(
                        rank,
                        capacity,
                        span(1, 0),
                        &gathered,
                        &flags,
                        epochs[rank as usize],
                    )
                    .unwrap();
                    assert_eq!(
                        resolved(&launches(lowered.unwrap())),
                        resolved(&[planned]),
                        "{tp:?} rank {rank} {phase:?}"
                    );
                }
            }
        }
    }

    /// A gather region smaller than the gather's state refuses the op when
    /// the instance is created.
    #[test]
    fn undersized_gather_state_refuses_at_create() {
        let tp = Tp::TP4;
        let switches = Switches::configured();
        let backend = crate::sm120::backend(tp, switches, true).unwrap();
        let prepare = |rows: State, exchange: State, candidates: State| {
            prepare_on(
                &*backend,
                tp,
                switches,
                &|_| [u64::MAX; 3],
                &|_| GatherRegions {
                    rows,
                    exchange: Some(exchange),
                },
                Some(candidates),
            )
        };
        let any = [u64::MAX; 2];
        assert!(prepare(any, any, any).is_ok());
        for short in [[1, u64::MAX], [u64::MAX, 1]] {
            for result in [
                prepare(short, any, any),
                prepare(any, short, any),
                prepare(any, any, short),
            ] {
                assert!(
                    matches!(&result, Err(Refused::Rule(rule)) if rule.contains("comm state")),
                    "{:?}",
                    result.err()
                );
            }
        }
    }
}
