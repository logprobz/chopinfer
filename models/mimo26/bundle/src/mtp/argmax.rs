//! The sharded argmax launches MTP built before `ops` prepared them, which
//! tests compare the prepared ops with. Each draft row gathers one scalar
//! pair per rank.

use kernel_core::contract::{Argument, Refusal, RuntimeCounts};
use runtime_bundle_abi::DeviceBufferSpanV1;

use crate::dispatch::BufferBindings;
use crate::pipeline::PreparedLaunch;
use crate::topology::Tp;

fn plan(
    tp: Tp,
    name: &str,
    spans: &[DeviceBufferSpanV1],
    offset: Option<u32>,
    rows: u32,
) -> Result<PreparedLaunch, Refusal> {
    let symbol = crate::sm120::selection::drafter_row_symbol(tp, &format!("dflash_argmax_{name}"));
    let entry = crate::sm120::launch::row::entries_for(tp)
        .find(|entry| entry.symbol == symbol)
        .ok_or(Refusal("MTP sharded argmax entry missing"))?;
    let mut bindings = BufferBindings::default();
    let mut args = spans
        .iter()
        .map(|&span| Argument::DeviceBuffer(bindings.bind(span)))
        .collect::<Vec<_>>();
    if let Some(offset) = offset {
        args.push(Argument::U32(offset));
    }
    let (plan, mut arguments) = crate::sm120::launch::row::plan(
        entry,
        rows.next_power_of_two(),
        RuntimeCounts {
            m: rows,
            batch: rows,
            context: 1,
            routed_rows: 1,
            split_k: 1,
        },
        args,
    )?;
    // A one-address by-value array has the same CUDA parameter bytes as a
    // pointer, while explicitly asking the host to validate a peer lease.
    // Ordinary DeviceBuffer arguments are restricted to the launch context.
    for index in if name == "partial" { [1, 2] } else { [0, 1] } {
        let Argument::DeviceBuffer(id) = arguments.0[index] else {
            return Err(Refusal("MTP argmax peer pointer encoding"));
        };
        arguments.0[index] = Argument::DevicePointerArray(vec![id]);
    }
    Ok(PreparedLaunch {
        symbol,
        rows,
        plan,
        arguments,
        bindings,
    })
}

/// `logits` starts at the single row whose next token is being drafted.
pub fn partial(
    tp: Tp,
    rank: u32,
    logits: DeviceBufferSpanV1,
    scores: DeviceBufferSpanV1,
    ids: DeviceBufferSpanV1,
    live: DeviceBufferSpanV1,
    width: DeviceBufferSpanV1,
) -> Result<PreparedLaunch, Refusal> {
    partial_batch(tp, rank, logits, scores, ids, live, width, 1)
}

#[allow(clippy::too_many_arguments)]
pub fn partial_batch(
    tp: Tp,
    rank: u32,
    logits: DeviceBufferSpanV1,
    scores: DeviceBufferSpanV1,
    ids: DeviceBufferSpanV1,
    live: DeviceBufferSpanV1,
    width: DeviceBufferSpanV1,
    rows: u32,
) -> Result<PreparedLaunch, Refusal> {
    let shard = crate::topology::vocab_per_rank(tp);
    if rank >= tp.ranks()
        || rows == 0
        || rows > crate::contract::DRAFT_ITEMS
        || logits.length < u64::from(rows) * u64::from(shard) * 2
        || scores.length < u64::from(rows) * 4
        || ids.length < u64::from(rows) * 4
        || [scores, ids, live, width]
            .iter()
            .any(|span| span.length < 4)
    {
        return Err(Refusal("MTP argmax partial buffers"));
    }
    plan(
        tp,
        "partial",
        &[logits, scores, ids, live, width],
        Some(rank * shard),
        rows,
    )
}

/// Every rank's score and ID, in rank order, after stream-ordered gathering.
pub fn finish(
    tp: Tp,
    scores: DeviceBufferSpanV1,
    ids: DeviceBufferSpanV1,
    token: DeviceBufferSpanV1,
    live: DeviceBufferSpanV1,
) -> Result<PreparedLaunch, Refusal> {
    finish_batch(tp, scores, ids, token, live, 1)
}

/// Row-major pairs: `rows` rows of one score (or ID) per rank.
pub fn finish_batch(
    tp: Tp,
    scores: DeviceBufferSpanV1,
    ids: DeviceBufferSpanV1,
    token: DeviceBufferSpanV1,
    live: DeviceBufferSpanV1,
    rows: u32,
) -> Result<PreparedLaunch, Refusal> {
    let row_bytes = u64::from(tp.ranks()) * 4;
    if rows == 0
        || rows > crate::contract::DRAFT_ITEMS
        || scores.length < u64::from(rows) * row_bytes
        || ids.length < u64::from(rows) * row_bytes
        || token.length < u64::from(rows) * 4
        || live.length < 4
    {
        return Err(Refusal("MTP argmax final buffers"));
    }
    plan(tp, "final", &[scores, ids, token, live], None, rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, KernelArgumentV1};

    #[test]
    fn batch_argmax_uses_prepared_buckets_and_checks_full_spans() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let span = |bytes| DeviceBufferSpanV1::new(buffer, 0, bytes);
        for rows in [1, 3, 16, 32] {
            let bytes = u64::from(rows);
            let partial = partial_batch(
                Tp::TP4,
                0,
                span(bytes * 38144 * 2),
                span(bytes * 4),
                span(bytes * 4),
                span(4),
                span(4),
                rows,
            )
            .unwrap();
            let finish = finish_batch(
                Tp::TP4,
                span(bytes * 16),
                span(bytes * 16),
                span(bytes * 4),
                span(4),
                rows,
            )
            .unwrap();
            assert_eq!(partial.plan.grid, [rows.next_power_of_two(), 1, 1]);
            assert_eq!(finish.plan.grid, [rows.next_power_of_two(), 1, 1]);
            assert!(
                finish_batch(
                    Tp::TP4,
                    span(bytes * 16 - 1),
                    span(bytes * 16),
                    span(bytes * 4),
                    span(4),
                    rows
                )
                .is_err()
            );
        }
    }

    #[test]
    fn partial_assigns_global_vocabulary_ids_and_final_requires_all_ranks() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let span = |bytes| DeviceBufferSpanV1::new(buffer, 0, bytes);
        for rank in 0..4 {
            let launch = partial(
                Tp::TP4,
                rank,
                span(38144 * 2),
                span(4),
                span(4),
                span(4),
                span(4),
            )
            .unwrap();
            let args = crate::dispatch::arguments(&launch.arguments, &launch.bindings).unwrap();
            assert_eq!(
                args.last(),
                Some(&KernelArgumentV1::ScalarU32(rank * 38144))
            );
            assert!(
                matches!(&args[1], KernelArgumentV1::DevicePointerArray(slots) if slots.len() == 1)
            );
            assert!(
                matches!(&args[2], KernelArgumentV1::DevicePointerArray(slots) if slots.len() == 1)
            );
            assert_eq!(launch.plan.grid, [1, 1, 1]);
        }
        assert!(
            partial(
                Tp::TP4,
                4,
                span(38144 * 2),
                span(4),
                span(4),
                span(4),
                span(4)
            )
            .is_err()
        );
        assert!(finish(Tp::TP4, span(12), span(16), span(4), span(4)).is_err());
        assert!(finish(Tp::TP4, span(16), span(16), span(4), span(4)).is_ok());
        // TP2 shards the vocabulary two ways and finishes over two pairs.
        let launch = partial(
            Tp::TP2,
            1,
            span(76288 * 2),
            span(4),
            span(4),
            span(4),
            span(4),
        )
        .unwrap();
        assert_eq!(launch.symbol, "chop_row_dflash_tp2_argmax_partial_v1");
        let args = crate::dispatch::arguments(&launch.arguments, &launch.bindings).unwrap();
        assert_eq!(args.last(), Some(&KernelArgumentV1::ScalarU32(76288)));
        assert!(finish(Tp::TP2, span(4), span(8), span(4), span(4)).is_err());
        let finish = finish(Tp::TP2, span(8), span(8), span(4), span(4)).unwrap();
        assert_eq!(finish.symbol, "chop_row_dflash_tp2_argmax_final_v1");
    }
}
