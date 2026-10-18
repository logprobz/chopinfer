use super::*;
use crate::dspark::state::DraftBlock;
use model_deepseek41_contract::geometry::{FLASH, GEOMETRIES};

fn input(rows: u64, budget: u32) -> DraftInput {
    DraftInput {
        block: DraftBlock {
            request: 7,
            slot: 3,
            main_rows: rows,
            compute_rows: 5,
            emit_rows: budget,
            verify_rows: budget + 1,
        },
        bonus: 42,
    }
}

#[test]
fn every_budget_builds_the_same_full_noncausal_block() {
    for main_rows in [1, 127, 128, 129, 511] {
        let full = metadata(&FLASH, &[input(main_rows, 5)], DraftMode::Greedy).unwrap();
        for budget in 1..5 {
            assert_eq!(
                metadata(&FLASH, &[input(main_rows, budget)], DraftMode::Greedy).unwrap(),
                full
            );
        }
        let stride = FLASH.shape.attention.window as usize + 5;
        let expected = (0..main_rows.min(128) as i32)
            .chain(128..133)
            .collect::<Vec<_>>();
        for row in full.keys.chunks_exact(stride) {
            assert_eq!(
                row.iter()
                    .copied()
                    .filter(|&key| key >= 0)
                    .collect::<Vec<_>>(),
                expected
            );
        }
        // A causal first row would omit the last four block positions.
        assert!(full.keys[..stride].contains(&132));
        assert_eq!(full.tokens[0], 42);
        assert_eq!(full.slots, [3]);
        assert_eq!(full.staged_slots, [0]);
        assert_eq!(&full.tokens[1..], &[FLASH.shape.dspark.noise_token; 4]);
    }
}

#[test]
fn metadata_refuses_sampled_drafts_empty_history_and_duplicate_leases() {
    assert_eq!(
        metadata(&FLASH, &[input(1, 5)], DraftMode::Sampled).unwrap_err(),
        "DSpark currently supports greedy draft selection only"
    );
    assert!(metadata(&FLASH, &[input(0, 5)], DraftMode::Greedy).is_err());
    assert!(metadata(&FLASH, &[input(1, 0)], DraftMode::Greedy).is_err());
    assert!(metadata(&FLASH, &[input(1, 5), input(1, 5)], DraftMode::Greedy).is_err());
    assert!(metadata(&FLASH, &[input(i32::MAX as u64, 5)], DraftMode::Greedy).is_err());
}

#[test]
fn every_compiled_topology_prepares_the_drafter_primitives() {
    let mut tested = 0;
    for shape in crate::shape::topologies() {
        let kernels = crate::Kernels::new(&shape).unwrap();
        let prepared = Prepared::prepare(kernels.backend.as_ref(), shape, DraftMode::Greedy);
        assert!(
            prepared.is_ok(),
            "{}: {:?}",
            shape.topology(),
            prepared.err()
        );
        tested += 1;
    }
    assert!(tested >= 3);
}

#[test]
fn feature_packing_preserves_full_prefill_capacity_and_row_order() {
    use model_deepseek41_contract::serving::PREFILL_TOKENS;
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1 as Span};

    let shape = crate::shape::topologies()[0];
    let kernels = crate::Kernels::new(&shape).unwrap();
    let prepared = Prepared::prepare(kernels.backend.as_ref(), shape, DraftMode::Greedy).unwrap();
    let targets = shape.geometry.shape.dspark.targets.len() as u32;
    assert_eq!(targets, 3);
    assert_eq!(
        prepared
            .feature_pack_counts(PREFILL_TOKENS, PREFILL_TOKENS)
            .unwrap(),
        [PREFILL_TOKENS; 3]
    );
    assert_eq!(
        prepared.feature_pack_counts(PREFILL_TOKENS, 3000).unwrap(),
        [8192, 808, 0]
    );
    assert_eq!(
        prepared.feature_pack_counts(PREFILL_TOKENS, 0).unwrap(),
        [0; 3]
    );
    assert_eq!(prepared.feature_pack_counts(5, 3).unwrap(), [9]);
    assert!(prepared.feature_pack_counts(5, 6).is_err());
    let indices = prepared.feature_indices(PREFILL_TOKENS).unwrap();
    for (row, group) in indices.chunks_exact(targets as usize).enumerate() {
        assert_eq!(
            group,
            &[
                row as u32,
                PREFILL_TOKENS + row as u32,
                2 * PREFILL_TOKENS + row as u32
            ]
        );
    }
    let span = |id, bytes| Span::new(BundleDeviceBufferHandleV1::from_raw(id), 0, bytes);
    let bytes = u64::from(PREFILL_TOKENS * targets) * u64::from(shape.geometry.shape.hidden) * 2;
    let counts = [span(4, 4), span(5, 4), span(6, 4)];
    let launches = prepared
        .pack_features(
            PREFILL_TOKENS,
            span(1, bytes),
            span(2, indices.len() as u64 * 4),
            &counts,
            span(3, bytes),
        )
        .unwrap();
    assert_eq!(launches.len(), 3);
    assert!(
        launches
            .iter()
            .all(|launch| launch.launch.rows == PREFILL_TOKENS)
    );
    assert!(
        prepared
            .pack_features(
                PREFILL_TOKENS,
                span(1, bytes),
                span(2, indices.len() as u64 * 4),
                &counts[..2],
                span(3, bytes)
            )
            .is_err()
    );
    assert!(
        prepared
            .pack_features(
                PREFILL_TOKENS,
                span(1, bytes),
                span(2, indices.len() as u64 * 4),
                &counts,
                span(3, bytes - 1)
            )
            .is_err()
    );
}

#[test]
fn synthetic_blocks_use_the_complete_ratio_zero_attention_and_expert_stage() {
    for g in GEOMETRIES {
        let phase = Phase::Verify { rows: 10, items: 2 };
        if g.shape.moe.experts != g.shape.dspark.experts {
            assert!(
                block_stages(g, 0, phase)
                    .unwrap_err()
                    .contains("expert set")
            );
            continue;
        }
        for stage in 0..g.dspark_stages() {
            let layer = g.layers() + stage;
            let stages = block_stages(g, stage, phase).unwrap();
            assert!(stages.contains(&Stage::WindowKv(layer)));
            assert!(stages.contains(&Stage::LatentAttention(layer)));
            assert!(stages.contains(&Stage::ExpertsGateUp(layer)));
            assert!(stages.contains(&Stage::ExpertsDown(layer)));
            assert!(!stages.iter().any(|s| matches!(
                s,
                Stage::Compress(_) | Stage::IndexSelect(_) | Stage::WindowRingWrite(_)
            )));
            assert_eq!(stages.last(), Some(&Stage::HcExpand(layer, Sublayer::Mlp)));
        }
        assert!(block_stages(g, 0, Phase::Verify { rows: 8, items: 2 }).is_err());
    }
}
