//! Decode-only ordered merging. Prefill keeps the faster single-CTA merge.
use crate::ops::Lowered;
use crate::program::Phase;
use crate::shape::Shape;
use bundle_runtime::pipeline::PreparedLaunch;
use kernel_core::contract::{Argument, Refusal};
use kernels_sm120::pooled_indexer::position_hierarchical as hierarchy;
use model_deepseek41_contract::{geometry::FLASH, state::KvStorage};
use runtime_bundle_abi::DeviceBufferSpanV1;

impl crate::Kernels {
    pub(crate) fn hierarchical_indexer_enabled(
        shape: &Shape,
        storage: KvStorage,
        phase: Phase,
    ) -> bool {
        enabled(shape, storage, phase)
    }

    pub(crate) fn hierarchical_indexer_select(
        calls: Vec<Lowered>,
        scores: DeviceBufferSpanV1,
        rows: u32,
        width: u32,
    ) -> Result<Vec<Lowered>, Refusal> {
        apply(calls, scores, rows, width)
    }
}

fn enabled(shape: &Shape, storage: KvStorage, phase: Phase) -> bool {
    shape.geometry == &FLASH
        && storage == KvStorage::Native
        && shape.runtime.indexer_mma
        && matches!(phase, Phase::Decode(1..=32))
}

/// The score allocation is 256 MiB. Decode uses at most 128 MiB (32 rows
/// at the largest dense stride). Reserve the intermediate lists at its end,
/// outside every row's score slots, including invisible and poisoned slots.
/// No allocation or backend workspace changes; these stages run serially on
/// the attention stream, before the candidate producer reuses score scratch.
fn scratch(
    scores: DeviceBufferSpanV1,
    rows: u32,
    width: u32,
) -> Result<DeviceBufferSpanV1, Refusal> {
    if !(1..=32).contains(&rows) || width == 0 {
        return Err(Refusal("hierarchical score scratch is not admitted"));
    }
    let bytes = hierarchy::workspace_bytes(rows, 512);
    let used = u64::from(rows) * u64::from(width) * 4;
    let offset = scores
        .length
        .checked_sub(bytes)
        .filter(|&n| n >= used)
        .ok_or(Refusal("hierarchical lists overlap score rows"))?;
    let offset = scores
        .offset
        .checked_add(offset)
        .ok_or(Refusal("hierarchical score offset overflows"))?;
    if !offset.is_multiple_of(8) || offset.checked_add(bytes).is_none() {
        return Err(Refusal("hierarchical score scratch is not admitted"));
    }
    Ok(DeviceBufferSpanV1::new(scores.buffer, offset, bytes))
}

fn apply(
    mut calls: Vec<Lowered>,
    scores: DeviceBufferSpanV1,
    rows: u32,
    width: u32,
) -> Result<Vec<Lowered>, Refusal> {
    let intermediate = scratch(scores, rows, width)?;
    let mut final_call = calls.pop().ok_or(Refusal("no position merge"))?;
    let call = &mut final_call.launch;
    if final_call.collective
        || call.rows != rows
        || call.plan.grid != [1, rows, 1]
        || !(call.symbol.ends_with("_live_merge_v1") || call.symbol.ends_with("_valid_merge_v1"))
    {
        return Err(Refusal(
            "hierarchical merge requires the qualified position merge",
        ));
    }
    call.arguments
        .0
        .push(Argument::DeviceBuffer(call.bindings.bind(intermediate)));
    let mut group = PreparedLaunch {
        symbol: hierarchy::symbol(&call.symbol, false),
        rows: call.rows,
        plan: call.plan,
        arguments: call.arguments.clone(),
        bindings: call.bindings.clone(),
    };
    group.plan.grid[0] = hierarchy::GROUPS;
    call.symbol = hierarchy::symbol(&call.symbol, true);
    calls.push(Lowered {
        launch: group,
        collective: false,
    });
    calls.push(final_call);
    Ok(calls)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    #[test]
    fn only_qualified_decode_phases_select_the_hierarchy() {
        use bundle_runtime::geometry::Ranks;
        let mut shape = Shape::new(&FLASH, Ranks::TP1).unwrap();
        for rows in [1, 2, 4, 6, 8, 16, 32, 33, 64, 8192] {
            for storage in [KvStorage::Native, KvStorage::Bf16] {
                for mma in [false, true] {
                    shape.runtime.indexer_mma = mma;
                    assert_eq!(
                        enabled(&shape, storage, Phase::Decode(rows)),
                        rows <= 32 && mma && storage == KvStorage::Native
                    );
                    assert!(!enabled(&shape, storage, Phase::Prefill { rows }));
                    assert!(!enabled(&shape, storage, Phase::Verify { rows, items: 1 }));
                }
            }
        }
    }

    #[test]
    fn score_tail_is_disjoint_at_every_decode_bucket_and_maximum_stride() {
        let handle = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let scores = DeviceBufferSpanV1::new(handle, 256, 256 << 20);
        for rows in [1, 2, 4, 6, 8, 16, 32] {
            for width in [16384, 1 << 19, 1 << 20] {
                let lists = scratch(scores, rows, width).unwrap();
                assert!(lists.offset >= scores.offset + u64::from(rows) * u64::from(width) * 4);
                assert_eq!(lists.offset + lists.length, scores.offset + scores.length);
                assert_eq!(lists.length, hierarchy::workspace_bytes(rows, 512));
            }
        }
        assert!(scratch(scores, 0, 1).is_err());
        assert!(scratch(scores, 33, 1).is_err());
        assert!(scratch(scores, 32, 1 << 21).is_err());
        assert!(scratch(DeviceBufferSpanV1::new(handle, 1, 256 << 20), 1, 1).is_err());
        assert!(
            scratch(
                DeviceBufferSpanV1::new(handle, u64::MAX - 7, 256 << 20),
                1,
                1
            )
            .is_err()
        );
    }
}
