//! The launches MiMo planned for its collectives before `crate::ops`
//! prepared them, which tests compare the prepared ops with. The peer
//! arrays hold one span per rank; their length is the rank count.

use kernel_core::contract::{DType, Refusal};
use runtime_bundle_abi::DeviceBufferSpanV1;

use crate::dispatch::BufferBindings;
use crate::ops::LogitsBuffers;
use crate::pipeline::PreparedLaunch;
pub use crate::sm120::launch::comm::HiddenRoute;
use crate::topology::Tp;

#[derive(Clone)]
pub struct HiddenBuffers {
    pub input: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    pub inboxes: Vec<DeviceBufferSpanV1>,
    pub flags: Vec<DeviceBufferSpanV1>,
    pub epoch: DeviceBufferSpanV1,
    pub gathered_outputs: Vec<DeviceBufferSpanV1>,
    pub residual: DeviceBufferSpanV1,
    pub residual_output: DeviceBufferSpanV1,
    pub norm_weight: DeviceBufferSpanV1,
    /// When set, `input` holds preweighted routed rows `[rows][8][4096]`
    /// and the all-reduce sums them first, rows past this live count as
    /// zero: the decode and verify MoE combine folded into its all-reduce.
    pub combine_live_rows: Option<DeviceBufferSpanV1>,
}

/// A hidden all-reduce of `rows` rows. Decode and verify graphs take the
/// wide entries above 32 rows when their route allows it; prefill the phased
/// stages. Steps launch `crate::ops`' prepared all-reduces, which tests
/// compare with this builder.
pub fn fused_hidden_for(
    tp: Tp,
    rows: u32,
    rank: u32,
    route: HiddenRoute,
    buffers: HiddenBuffers,
) -> Result<Vec<PreparedLaunch>, Refusal> {
    let ranks = buffers.inboxes.len() as u32;
    let mut bindings = BufferBindings::default();
    let input = bindings.bind(buffers.input);
    let output = bindings.bind(buffers.output);
    let inboxes = buffers
        .inboxes
        .iter()
        .map(|&span| bindings.bind(span))
        .collect();
    let flags = buffers
        .flags
        .iter()
        .map(|&span| bindings.bind(span))
        .collect();
    let epoch = bindings.bind(buffers.epoch);
    let gathered = buffers
        .gathered_outputs
        .iter()
        .map(|&span| bindings.bind(span))
        .collect();
    let residual = bindings.bind(buffers.residual);
    let residual_output = bindings.bind(buffers.residual_output);
    let weights = bindings.bind(buffers.norm_weight);
    if let Some(live) = buffers.combine_live_rows {
        if route == HiddenRoute::Prefill {
            return Err(Refusal("MiMo folds the MoE combine in graphs only"));
        }
        let live = bindings.bind(live);
        let (symbol, plan, arguments) = crate::sm120::launch::comm::fused_combine_hidden_for(
            tp,
            rows,
            rank,
            route,
            crate::sm120::launch::comm::OneShotCall {
                input,
                output,
                peer_inboxes: inboxes,
                peer_flags: flags,
                epoch,
                elements: u64::from(rows) * tp.geometry.hidden() as u64,
                rank,
                ranks,
            },
            crate::sm120::launch::comm::ResidualBuffers {
                input: residual,
                output: residual_output,
            },
            weights,
            tp.geometry.eps() as f32,
            live,
        )?;
        return Ok(vec![PreparedLaunch {
            symbol,
            rows,
            plan,
            arguments,
            bindings,
        }]);
    }
    let stages = crate::sm120::launch::comm::fused_hidden_for(
        tp,
        rows,
        rank,
        route,
        crate::sm120::launch::comm::OneShotCall {
            input,
            output,
            peer_inboxes: inboxes,
            peer_flags: flags,
            epoch,
            elements: u64::from(rows) * tp.geometry.hidden() as u64,
            rank,
            ranks,
        },
        gathered,
        crate::sm120::launch::comm::ResidualBuffers {
            input: residual,
            output: residual_output,
        },
        weights,
        tp.geometry.eps() as f32,
    )?;
    Ok(stages
        .into_iter()
        .map(|(symbol, plan, arguments)| PreparedLaunch {
            symbol,
            rows,
            plan,
            arguments,
            bindings: bindings.clone(),
        })
        .collect())
}

impl LogitsBuffers {
    fn ranks(&self) -> u32 {
        self.outputs.len() as u32
    }
}

/// One launch gathers rank `rank`'s rows into full vocabulary rows on rank
/// 0, as MiMo planned it before `crate::ops` prepared its gathers, which
/// tests compare the prepared ops with.
pub fn gather_logits_rows(
    rows: &[u32],
    rank: u32,
    dtype: DType,
    buffers: LogitsBuffers,
) -> Result<Vec<PreparedLaunch>, Refusal> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let element_bytes = match dtype {
        DType::F32 => crate::contract::PREFILL_LOGIT_BYTES,
        DType::Bf16 => crate::contract::DECODE_LOGIT_BYTES,
        _ => return Err(Refusal("MiMo logits gather dtype is unsupported")),
    };
    let last = u64::from(*rows.iter().max().expect("nonempty rows")) + 1;
    let shard_bytes = u64::from(buffers.shard) * element_bytes;
    let full_bytes = shard_bytes * u64::from(buffers.ranks());
    if rows.len() > crate::contract::PREFILL_TOKENS as usize
        || last * shard_bytes > buffers.local.length
        || buffers
            .outputs
            .iter()
            .any(|span| last * full_bytes > span.length)
        || buffers
            .flags
            .iter()
            .any(|span| span.length < kernels_sm120::comm::GATHER_ROWS_FLAG_BYTES)
        || buffers.epochs[rank as usize].length < kernels_sm120::comm::GATHER_ROWS_EPOCH_BYTES
        || rows
            .iter()
            .enumerate()
            .any(|(index, &row)| row != index as u32)
    {
        return Err(Refusal("MiMo logits row gather exceeds prepared buffers"));
    }
    let mut bindings = BufferBindings::default();
    let input = bindings.bind(buffers.local);
    let outputs = buffers
        .outputs
        .iter()
        .map(|&span| bindings.bind(span))
        .collect();
    let flags = buffers
        .flags
        .iter()
        .map(|&span| bindings.bind(span))
        .collect();
    let epoch = bindings.bind(buffers.epochs[rank as usize]);
    let (symbol, plan, arguments) = crate::sm120::launch::comm::gather_logits_rows(
        dtype,
        kernels_sm120::comm::GatherRowsCall {
            input,
            outputs,
            flags,
            epoch,
            row_indices: None,
            rows: rows.len() as u32,
            shard: u64::from(buffers.shard),
            rank,
            ranks: buffers.ranks(),
            target: 0,
        },
    )?;
    Ok(vec![PreparedLaunch {
        symbol,
        rows: rows.len() as u32,
        plan,
        arguments,
        bindings,
    }])
}

/// One launch exchanges decode logits rows so each rank holds full rows for
/// its block (`kernels_sm120::comm::exchange_rows_owned`), at the same
/// offsets the gather uses on rank 0, as MiMo planned it before
/// `crate::ops` prepared its exchanges, which tests compare them with.
pub fn exchange_logits_rows(
    rows: u32,
    rank: u32,
    dtype: DType,
    buffers: LogitsBuffers,
) -> Result<Vec<PreparedLaunch>, Refusal> {
    bundle_runtime::logits::exchange_rows(
        rows,
        crate::contract::VERIFY_TOKENS,
        u64::from(buffers.shard),
        rank,
        dtype,
        bundle_runtime::logits::ExchangeRowsBuffers {
            local: buffers.local,
            outputs: &buffers.outputs,
            flags: &buffers.flags,
            epoch: buffers.epochs[rank as usize],
        },
    )
    .map(|launch| vec![launch])
}

pub fn fused_hidden(
    rows: u32,
    rank: u32,
    route: HiddenRoute,
    buffers: HiddenBuffers,
) -> Result<Vec<PreparedLaunch>, Refusal> {
    fused_hidden_for(
        Tp::new(buffers.inboxes.len() as u32).unwrap(),
        rows,
        rank,
        route,
        buffers,
    )
}

#[cfg(test)]
mod tests {
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, KernelArgumentV1};

    use super::*;
    use crate::contract::RANKS;

    #[test]
    fn hidden_and_vocab_collectives_bind_peer_arrays_in_rank_order() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let span = |offset| DeviceBufferSpanV1::new(buffer, offset, 8192 * 4096 * 2 * 4);
        let buffers = |combine_live_rows| HiddenBuffers {
            input: span(1),
            output: span(2),
            inboxes: vec![span(10), span(11), span(12), span(13)],
            flags: vec![span(20), span(21), span(22), span(23)],
            epoch: span(3),
            gathered_outputs: vec![span(30), span(31), span(32), span(33)],
            residual: span(4),
            residual_output: span(6),
            norm_weight: span(5),
            combine_live_rows,
        };
        for (rows, route) in [1, 2, 4, 8, 16, 32, 64, 128, 256, 8192]
            .into_iter()
            .flat_map(|rows| {
                [
                    (rows, HiddenRoute::Prefill),
                    (rows, HiddenRoute::Graph { wide: false }),
                    (rows, HiddenRoute::Graph { wide: true }),
                    (rows, HiddenRoute::ExactRows),
                ]
            })
        {
            let graph = route != HiddenRoute::Prefill;
            // Exact rows take the verify entries: one-shot rows, and two-shot
            // rows in the low-latency norm order up to 32.
            if route == HiddenRoute::ExactRows {
                if rows > kernels_sm120::comm::verify_entry_rows(4).unwrap() {
                    assert!(fused_hidden(rows, 1, route, buffers(None)).is_err());
                    assert!(fused_hidden(rows, 1, route, buffers(Some(span(7)))).is_err());
                    continue;
                }
                let [plain] = &fused_hidden(rows, 1, route, buffers(None)).unwrap()[..] else {
                    panic!("one exact-row stage");
                };
                let [combined] = &fused_hidden(rows, 1, route, buffers(Some(span(7)))).unwrap()[..]
                else {
                    panic!("one exact-row combine stage");
                };
                let (plain_symbol, combine_symbol) = if rows <= 2 {
                    crate::sm120::launch::comm::push_symbols(4, rows)
                } else {
                    (
                        kernels_sm120::comm::VERIFY_SYMBOL,
                        kernels_sm120::comm::VERIFY_COMBINE_SYMBOL,
                    )
                };
                assert_eq!(plain.symbol, plain_symbol);
                assert_eq!(combined.symbol, combine_symbol);
                let [graph] =
                    &fused_hidden(rows, 1, HiddenRoute::Graph { wide: false }, buffers(None))
                        .unwrap()[..]
                else {
                    panic!("one graph stage");
                };
                assert_eq!(plain.plan, graph.plan);
                assert_eq!(
                    crate::dispatch::arguments(&plain.arguments, &plain.bindings).unwrap(),
                    crate::dispatch::arguments(&graph.arguments, &graph.bindings).unwrap()
                );
                continue;
            }
            let stages = fused_hidden(rows, 1, route, buffers(None)).unwrap();
            // Graphs that may take the wide entry take it from 33 to 256
            // rows; prefill and the others the reduce-scatter and fused
            // gather stages above 32.
            let wide = route == (HiddenRoute::Graph { wide: true }) && rows > 32 && rows <= 256;
            assert_eq!(stages.len(), if rows <= 32 || wide { 1 } else { 2 });
            if wide {
                assert_eq!(stages[0].symbol, kernels_sm120::comm::WIDE_SYMBOL);
            }
            // The push and wide entries take the MoE combine (the bundle
            // folds it only up to 32 rows, `graph_folds_combine`).
            let combined = fused_hidden(rows, 1, route, buffers(Some(span(7))));
            if graph && (rows <= 32 || wide) {
                let combined = combined.unwrap();
                assert_eq!(combined.len(), 1);
                assert_eq!(
                    combined[0].symbol,
                    if wide {
                        kernels_sm120::comm::WIDE_COMBINE_SYMBOL
                    } else {
                        crate::sm120::launch::comm::push_symbols(4, rows).1
                    }
                );
                assert_eq!(combined[0].plan, stages[0].plan);
                let args =
                    crate::dispatch::arguments(&combined[0].arguments, &combined[0].bindings)
                        .unwrap();
                let plain =
                    crate::dispatch::arguments(&stages[0].arguments, &stages[0].bindings).unwrap();
                assert_eq!(args[..plain.len()], plain[..]);
                assert!(matches!(
                    args.last(),
                    Some(KernelArgumentV1::DeviceBuffer(span)) if span.offset == 7
                ));
            } else {
                assert!(combined.is_err());
            }
            for stage in stages {
                assert!(stage.plan.pdl);
                let args = crate::dispatch::arguments(&stage.arguments, &stage.bindings).unwrap();
                // The round-once scatter inserts the residual before the peers.
                let peers =
                    usize::from(stage.symbol == kernels_sm120::comm::PREFILL_ROUND_ONCE_SYMBOL) + 2;
                assert!(
                    matches!(&args[peers], KernelArgumentV1::DevicePointerArray(slots) if slots.len() == 4)
                );
            }
        }
    }

    #[test]
    fn vocabulary_rows_gather_into_row_major_device_views() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let shard =
            u64::from(crate::contract::VOCAB_PER_RANK) * crate::contract::PREFILL_LOGIT_BYTES;
        let full = shard * u64::from(RANKS);
        let span = |offset, length| DeviceBufferSpanV1::new(buffer, offset, length);
        let launches = gather_logits_rows(
            &[0, 1],
            0,
            DType::F32,
            LogitsBuffers {
                local: span(100, 3 * shard),
                shard: crate::contract::VOCAB_PER_RANK,
                outputs: vec![span(1_000_000, 3 * full); RANKS as usize],
                flags: vec![
                    span(3_000_000, kernels_sm120::comm::GATHER_ROWS_FLAG_BYTES);
                    RANKS as usize
                ],
                epochs: vec![
                    span(4_000_000, kernels_sm120::comm::GATHER_ROWS_EPOCH_BYTES);
                    RANKS as usize
                ],
            },
        )
        .unwrap();
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].symbol, "chop_comm_gather_rows_f32_v1");
        assert_eq!(launches[0].plan.grid, [8, 1, 1]);
        assert!(launches[0].plan.pdl);
        let arguments =
            crate::dispatch::arguments(&launches[0].arguments, &launches[0].bindings).unwrap();
        assert_eq!(arguments.len(), 10);
        assert_eq!(
            arguments[0],
            KernelArgumentV1::DeviceBuffer(span(100, 3 * shard))
        );
        assert!(
            matches!(&arguments[1], KernelArgumentV1::DevicePointerArray(slots)
            if slots[0] == Some(span(1_000_000, 3 * full)))
        );
        assert_eq!(arguments[5], KernelArgumentV1::ScalarU32(2));
    }

    #[test]
    fn decode_rows_exchange_into_every_rank_with_exchange_flags() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let shard =
            u64::from(crate::contract::VOCAB_PER_RANK) * crate::contract::PREFILL_LOGIT_BYTES;
        let full = shard * u64::from(RANKS);
        let span = |offset, length| DeviceBufferSpanV1::new(buffer, offset, length);
        let buffers = |flags| LogitsBuffers {
            local: span(100, 32 * shard),
            shard: crate::contract::VOCAB_PER_RANK,
            outputs: vec![span(1_000_000, 32 * full); RANKS as usize],
            flags: vec![span(30_000_000, flags); RANKS as usize],
            epochs: vec![
                span(40_000_000, kernels_sm120::comm::EXCHANGE_ROWS_EPOCH_BYTES);
                RANKS as usize
            ],
        };
        let launches = exchange_logits_rows(
            32,
            2,
            DType::F32,
            buffers(kernels_sm120::comm::EXCHANGE_ROWS_FLAG_BYTES),
        )
        .unwrap();
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].symbol, "chop_comm_exchange_rows_f32_v1");
        assert_eq!(launches[0].plan.cluster, [8, 1, 1]);
        assert!(launches[0].plan.pdl);
        let arguments =
            crate::dispatch::arguments(&launches[0].arguments, &launches[0].bindings).unwrap();
        assert_eq!(arguments.len(), 8);
        assert_eq!(arguments[4], KernelArgumentV1::ScalarU32(32));
        assert_eq!(arguments[6], KernelArgumentV1::ScalarU32(2));
        // The gather's smaller flags and prefill row counts are refused.
        for (rows, flags) in [
            (32, kernels_sm120::comm::GATHER_ROWS_FLAG_BYTES),
            (33, kernels_sm120::comm::EXCHANGE_ROWS_FLAG_BYTES),
        ] {
            assert!(exchange_logits_rows(rows, 0, DType::F32, buffers(flags)).is_err());
        }
        // Verify buckets exchange up to 256 rows.
        let wide = |rows: u64| LogitsBuffers {
            local: span(100, rows * shard),
            outputs: vec![span(1_000_000, rows * full); RANKS as usize],
            flags: vec![
                span(30_000_000, kernels_sm120::comm::EXCHANGE_ROWS_FLAG_BYTES);
                RANKS as usize
            ],
            epochs: vec![
                span(40_000_000, kernels_sm120::comm::EXCHANGE_ROWS_EPOCH_BYTES);
                RANKS as usize
            ],
            shard: crate::contract::VOCAB_PER_RANK,
        };
        assert!(exchange_logits_rows(256, 3, DType::F32, wide(256)).is_ok());
        assert!(exchange_logits_rows(257, 3, DType::F32, wide(257)).is_err());
    }
    #[test]
    fn exchange_uses_the_topology_shard_width_at_every_rank_count() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let span = |offset, length| DeviceBufferSpanV1::new(buffer, offset, length);
        for ranks in crate::topology::Tp::SUPPORTED.map(|tp| tp.ranks()) {
            let shard = crate::contract::VOCAB_PER_RANK * RANKS / ranks;
            for (dtype, bytes) in [(DType::Bf16, 2_u64), (DType::F32, 4)] {
                let rows = 32;
                let local_bytes = u64::from(rows) * u64::from(shard) * bytes;
                let buffers = LogitsBuffers {
                    local: span(0, local_bytes),
                    shard,
                    outputs: vec![span(1_000_000, local_bytes * u64::from(ranks)); ranks as usize],
                    flags: vec![
                        span(30_000_000, kernels_sm120::comm::EXCHANGE_ROWS_FLAG_BYTES);
                        ranks as usize
                    ],
                    epochs: vec![
                        span(40_000_000, kernels_sm120::comm::EXCHANGE_ROWS_EPOCH_BYTES);
                        ranks as usize
                    ],
                };
                let launch = exchange_logits_rows(rows, ranks - 1, dtype, buffers).unwrap();
                let args =
                    crate::dispatch::arguments(&launch[0].arguments, &launch[0].bindings).unwrap();
                for index in [1, 2] {
                    assert!(matches!(
                        &args[index], KernelArgumentV1::DevicePointerArray(slots)
                            if slots.iter().flatten().count() == ranks as usize
                    ));
                }
                assert_eq!(args[4], KernelArgumentV1::ScalarU32(rows));
                assert_eq!(args[5], KernelArgumentV1::ScalarU64(u64::from(shard)));
                assert_eq!(args[6], KernelArgumentV1::ScalarU32(ranks - 1));
                assert_eq!(args[7], KernelArgumentV1::ScalarU32(ranks));
            }
        }
    }

    #[test]
    fn peer_arrays_carry_the_rank_count() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let span = |offset| DeviceBufferSpanV1::new(buffer, offset, 8192 * 4096 * 2 * 8);
        for ranks in crate::topology::Tp::SUPPORTED.map(|tp| tp.ranks()) {
            let peers = |base| {
                (0..ranks)
                    .map(|rank| span(base + u64::from(rank)))
                    .collect()
            };
            let stages = fused_hidden(
                8192,
                ranks - 1,
                HiddenRoute::Prefill,
                HiddenBuffers {
                    input: span(1),
                    output: span(2),
                    inboxes: peers(10),
                    flags: peers(20),
                    epoch: span(3),
                    gathered_outputs: peers(30),
                    residual: span(4),
                    residual_output: span(6),
                    norm_weight: span(5),
                    combine_live_rows: None,
                },
            )
            .unwrap();
            for stage in stages {
                let args = crate::dispatch::arguments(&stage.arguments, &stage.bindings).unwrap();
                assert!(args.contains(&KernelArgumentV1::ScalarU32(ranks)));
            }
        }
    }
}
