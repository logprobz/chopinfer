use kernel_core::contract::{DType, NumericalContract, Validate};
use kernel_core::oracle::check_numeric;
use kernel_core::pooled_indexer::{CandidateBlocksQuery, IndexerCandidates, PositionIndexerQuery};
use kernel_core::position_indexer_oracle as oracle;

fn query() -> PositionIndexerQuery {
    PositionIndexerQuery {
        heads: 3,
        head_dim: 160,
        compression_ratio: 2,
        tail_rope_dims: 64,
        key_norm_eps_bits: 1e-20_f32.to_bits(),
        max_top_k: 3,
        page_size: 64,
        candidates: IndexerCandidates::All,
        capacity: 16,
    }
}

#[test]
fn position_geometry_and_visibility_are_explicit() {
    let q = query();
    q.validate().unwrap();
    assert!(q.numerical_contract_id().unwrap().as_str().ends_with(".v3"));
    assert!(
        q.selection()
            .numerical_contract_id()
            .unwrap()
            .as_str()
            .ends_with(".v3")
    );
    assert_eq!(q.page_bytes(), (5120, 320));
    assert_eq!(
        (0..6)
            .map(|t| q.visible_positions(t).unwrap())
            .collect::<Vec<_>>(),
        [0, 1, 1, 2, 2, 3]
    );
    assert_eq!(q.visible_positions(u64::MAX), None);
    assert_ne!(
        q.numerical_contract_id().unwrap(),
        PositionIndexerQuery {
            candidates: IndexerCandidates::Mask,
            ..q
        }
        .numerical_contract_id()
        .unwrap()
    );
    for bad in [
        PositionIndexerQuery {
            compression_ratio: 0,
            ..q
        },
        PositionIndexerQuery { head_dim: 129, ..q },
        PositionIndexerQuery {
            tail_rope_dims: 65,
            ..q
        },
        PositionIndexerQuery {
            tail_rope_dims: 162,
            ..q
        },
        PositionIndexerQuery {
            key_norm_eps_bits: f32::NAN.to_bits(),
            ..q
        },
        PositionIndexerQuery { max_top_k: 0, ..q },
    ] {
        assert!(bad.validate().is_err());
    }
    PositionIndexerQuery {
        compression_ratio: 1,
        ..q
    }
    .validate()
    .unwrap();
}

#[test]
fn operand_preparation_uses_the_shared_trained_mxfp4_quantizer() {
    use kernel_core::precision::quantizer::{BlockQuantization, QuantizeOutput};
    let q = query();
    let queries = q.query_quantization().unwrap();
    let keys = q.key_quantization().unwrap();
    queries.validate().unwrap();
    keys.validate().unwrap();
    assert_eq!(queries.width, q.heads * q.head_dim);
    assert_eq!(keys.width, q.head_dim);
    assert_eq!(queries.input, DType::F32);
    assert_eq!(keys.input, DType::F32);
    assert_eq!(queries.format, BlockQuantization::E2M1Ceil32);
    assert_eq!(queries.format, keys.format);
    assert_eq!(queries.output, QuantizeOutput::Bf16);
    assert_eq!(keys.output, QuantizeOutput::Codes);
    let input = vec![6.015625_f32; q.head_dim as usize];
    let packed = keys.format.apply(&input, 1, q.head_dim as usize).unwrap();
    let fake = queries
        .format
        .fake_quant(&input, 1, q.head_dim as usize)
        .unwrap();
    for (i, value) in fake.into_iter().enumerate() {
        assert_eq!(
            f64::from(kernel_core::codec::bf16_to_f32(value)),
            packed.decode(0, i)
        );
        assert_eq!(packed.decode(0, i), 6.0);
    }
    assert!(
        PositionIndexerQuery {
            heads: u32::MAX,
            ..q
        }
        .query_quantization()
        .is_err()
    );
}

#[test]
fn f32_preparation_rounds_only_at_the_trained_quantizer() {
    use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
    let q = PositionIndexerQuery {
        heads: 1,
        head_dim: 128,
        ..query()
    };
    let mut table = [[1.0, 0.0]; 32];
    table[0] = [0.8, 0.6];
    let bf16 = |v: f32| bf16_to_f32(f32_to_bf16(v));
    let quantized_rotation = |input: &[f32]| {
        let input = input.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
        let rotated = oracle::rotate_tail(q, &input, &table).unwrap();
        let rotated = rotated.values.iter().map(|&v| v as f32).collect::<Vec<_>>();
        q.query_quantization()
            .unwrap()
            .format
            .fake_quant(&rotated, 1, 128)
            .unwrap()
    };
    let mut projected = vec![6.0_f32; 128];
    projected[64] = 1.0038;
    projected[65] = 0.8164;
    let kept = quantized_rotation(&projected);
    let rounded = quantized_rotation(&projected.iter().copied().map(bf16).collect::<Vec<_>>());
    assert_eq!(bf16_to_f32(kept[65]), 1.5);
    assert_eq!(
        bf16_to_f32(rounded[65]),
        1.0,
        "premature projection rounding control"
    );

    let mut key = vec![1.0_f32; 128];
    key[64] = 1.004;
    key[65] = 0.81;
    let normalized = oracle::key_norm(
        q,
        &key.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
        &[1.0; 128],
    )
    .unwrap();
    let normalized = normalized
        .values
        .iter()
        .map(|&v| v as f32)
        .collect::<Vec<_>>();
    let kept = quantized_rotation(&normalized);
    let rounded = quantized_rotation(&normalized.iter().copied().map(bf16).collect::<Vec<_>>());
    assert_eq!(bf16_to_f32(kept[65]), 1.0);
    assert_eq!(
        bf16_to_f32(rounded[65]),
        1.5,
        "premature norm rounding control"
    );
}

#[test]
fn f32_head_weights_are_not_rounded_to_bf16() {
    use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
    let q = query().selection();
    let mut queries = vec![0.0_f32; 3 * 160];
    queries[0] = 1.0;
    let weights = [1.001_f32, 0.0, 0.0];
    let keys = [1.0_f32; 160];
    let widen = |v: &[f32]| v.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    let reference = oracle::scores(q, &widen(&queries), &widen(&weights), &widen(&keys)).unwrap();
    let actual = oracle::scores_f32(q, &queries, &weights, &keys).unwrap();
    check_numeric(&widen(&actual), &reference, DType::F32).unwrap();
    let rounded = weights.map(|v| bf16_to_f32(f32_to_bf16(v)));
    let wrong = oracle::scores_f32(q, &queries, &rounded, &keys).unwrap();
    assert!(check_numeric(&widen(&wrong), &reference, DType::F32).is_err());
}

#[test]
fn score_bound_covers_chains_relu_and_negative_head_weights() {
    let q = query().selection();
    // Exact decoded E2M1 values at power-of-two scales, with two 128-K
    // chains and cancellation in each head.
    let decoded = |n: usize, seed: usize| {
        (0..n)
            .map(|i| {
                let codes = [-6.0_f32, -3.0, -1.5, -0.5, 0.0, 1.0, 2.0, 4.0];
                codes[(i * 7 + seed) % codes.len()] * 2.0_f32.powi((i % 9) as i32 - 4)
            })
            .collect::<Vec<_>>()
    };
    let queries = decoded(3 * 160, 1);
    let keys = decoded(17 * 160, 4);
    let weights = [0.25, -1.5, 2.0];
    let widen = |xs: &[f32]| xs.iter().copied().map(f64::from).collect::<Vec<_>>();
    let actual = oracle::scores_f32(q, &queries, &weights, &keys).unwrap();
    let reference = oracle::scores(q, &widen(&queries), &widen(&weights), &widen(&keys)).unwrap();
    check_numeric(&widen(&actual), &reference, DType::F32).unwrap();
    assert_eq!(reference.reduction_lengths, vec![164; 17]);
    let all_positive = vec![1.0; 3 * 160];
    let negative = oracle::scores_f32(q, &all_positive, &[-1.0; 3], &[1.0; 160]).unwrap();
    assert!(negative[0] < 0.0, "ReLU must precede head weighting");
    let expected = oracle::weight_scale(q).unwrap();
    assert_eq!(
        expected,
        ((160.0_f64).sqrt().recip() * (3.0_f64).sqrt().recip()) as f32
    );
}

#[test]
fn score_scale_stays_on_weights_and_chains_restart_after_128_channels() {
    let q = query().selection();
    let mut queries = vec![0.0; 3 * 160];
    queries[0] = 3.0;
    let scale = oracle::weight_scale(q).unwrap();
    let actual = oracle::scores_f32(q, &queries, &[2.5, 0.0, 0.0], &[1.0; 160]).unwrap();
    assert_eq!(actual[0].to_bits(), (3.0_f32 * (2.5 * scale)).to_bits());
    assert_ne!(actual[0].to_bits(), ((3.0_f32 * scale) * 2.5).to_bits());

    let q = PositionIndexerQuery {
        heads: 1,
        ..query()
    }
    .selection();
    let mut queries = vec![1.0; 160];
    queries[0] = 2.0_f32.powi(24);
    let scale = oracle::weight_scale(q).unwrap();
    let actual = oracle::scores_f32(q, &queries, &[1.0], &[1.0; 160]).unwrap();
    let folded = (2.0_f32.powi(24) + 32.0) * scale;
    let single_chain = 2.0_f32.powi(24) * scale;
    assert_eq!(actual[0].to_bits(), folded.to_bits());
    assert_ne!(actual[0].to_bits(), single_chain.to_bits());
}

#[test]
fn selection_uses_own_f32_scores_masks_and_lower_index_ties() {
    let q = query();
    let f64_scores = [1.0_f64, 1.0 + 2.0_f64.powi(-25), 0.5, 100.0];
    let scores = f64_scores.map(|s| s as f32);
    // The f64 top-1 would be 1; the kernel's F32 tie keeps 0. Position 3
    // is future data and cannot win even though it is much larger.
    assert_eq!(oracle::select(q, 1, 5, &scores, None).unwrap(), [0, -1, -1]);
    assert_eq!(oracle::select(q, 3, 0, &scores, None).unwrap(), [-1; 3]);
    let masked = PositionIndexerQuery {
        candidates: IndexerCandidates::Mask,
        ..q
    };
    let mask = [false, true, true, true];
    let selected = oracle::select(masked, 1, 5, &scores, Some(&mask)).unwrap();
    assert_eq!(selected, [1, -1, -1]);
    assert_ne!(
        selected,
        oracle::select(q, 1, 5, &scores, None).unwrap(),
        "ignored-mask control"
    );
    assert_eq!(
        oracle::select(masked, 3, 5, &scores, Some(&[false; 4])).unwrap(),
        [-1; 3]
    );
    assert!(oracle::select(masked, 3, 5, &scores, None).is_err());
    assert!(oracle::select(q, 3, 5, &scores, Some(&mask)).is_err());
    assert!(oracle::select(masked, 3, 5, &scores, Some(&mask[..3])).is_err());
    assert_eq!(
        oracle::select(q, 2, 3, &[-0.0, 0.0, f32::NAN], None).unwrap(),
        [0, 1, -1]
    );
    assert!(oracle::select(q, 3, u64::MAX, &scores, None).is_err());
}

#[test]
fn candidates_pin_the_newest_visible_block_within_the_budget() {
    let filter = CandidateBlocksQuery {
        block_size: 8,
        top_k_blocks: 2,
        capacity: 1,
    };
    let mut scores = vec![100.0; 24];
    scores[8..16].fill(50.0);
    scores[16..].fill(-10.0);
    let partial = oracle::select_candidate_blocks(filter, &scores, 17).unwrap();
    assert!(partial[..8].iter().all(|&v| v));
    assert!(partial[8..16].iter().all(|&v| !v));
    assert!(partial[16..].iter().all(|&v| v));
    // A block score is a max, not a sum: one high position wins its block.
    scores[..8].fill(-100.0);
    scores[0] = 60.0;
    assert_eq!(
        oracle::select_candidate_blocks(filter, &scores, 17).unwrap(),
        partial
    );
    let empty = oracle::select_candidate_blocks(filter, &scores, 0).unwrap();
    assert!(empty.iter().all(|&v| !v));
    let only_newest = CandidateBlocksQuery {
        top_k_blocks: 1,
        ..filter
    };
    let full = oracle::select_candidate_blocks(only_newest, &scores, 16).unwrap();
    assert!(full[..8].iter().all(|&v| !v));
    assert!(full[8..16].iter().all(|&v| v));
    assert!(full[16..].iter().all(|&v| !v));
    // Wrongly pinning floor(visible / block_size) picks an invisible block
    // at this exact boundary; skipping the pin picks the older block.
    assert_ne!(full, partial);
    let negative_inf = vec![f32::NEG_INFINITY; 24];
    assert_eq!(
        oracle::select_candidate_blocks(filter, &negative_inf, 16).unwrap(),
        full
    );
}

#[test]
fn candidate_filter_crosses_16384_positions_and_reindex_stays_inside_it() {
    let filter = CandidateBlocksQuery {
        block_size: 8,
        top_k_blocks: 2048,
        capacity: 1,
    };
    let mut source = vec![1.0; 16392];
    source[16384..].fill(-100.0);
    let mask = oracle::select_candidate_blocks(filter, &source, 16385).unwrap();
    assert_eq!(mask.iter().filter(|&&b| b).count(), 16384);
    assert!(mask[..16376].iter().all(|&b| b));
    assert!(mask[16376..16384].iter().all(|&b| !b));
    assert!(mask[16384..].iter().all(|&b| b));
    let q = PositionIndexerQuery {
        candidates: IndexerCandidates::Mask,
        ..query()
    };
    let mut reindex = vec![0.0; source.len()];
    reindex[16376] = 1000.0; // outside source candidates
    reindex[16384] = 10.0; // newest visible position
    reindex[16385] = 10000.0; // in a pinned block but still invisible
    assert_eq!(
        oracle::select(q, 1, 32769, &reindex, Some(&mask)).unwrap(),
        [16384, -1, -1]
    );
    let unmasked = PositionIndexerQuery {
        candidates: IndexerCandidates::All,
        ..q
    };
    assert_eq!(
        oracle::select(unmasked, 1, 32769, &reindex, None).unwrap(),
        [16376, -1, -1]
    );
}

#[test]
fn candidate_filter_refuses_invalid_shapes_and_nonfinite_visible_scores() {
    let q = CandidateBlocksQuery {
        block_size: 8,
        top_k_blocks: 2,
        capacity: 1,
    };
    assert!(
        CandidateBlocksQuery { block_size: 0, ..q }
            .validate()
            .is_err()
    );
    assert!(oracle::select_candidate_blocks(q, &[1.0], 2).is_err());
    assert!(oracle::select_candidate_blocks(q, &[f32::NAN], 1).is_err());
    assert!(oracle::select_candidate_blocks(q, &[f32::INFINITY], 1).is_err());
    assert_eq!(
        oracle::select_candidate_blocks(q, &[1.0, f32::NAN], 1).unwrap(),
        [true, true]
    );
}

#[test]
fn nonfinite_position_arithmetic_refuses_before_relu_or_selection() {
    let q = PositionIndexerQuery {
        heads: 1,
        head_dim: 160,
        ..query()
    };
    let huge = 2.0_f32.powi(80);
    let mut input = vec![0.0; 160];
    let mut key = vec![0.0; 160];
    input[0] = huge;
    key[0] = huge;
    assert!(
        oracle::scores_f32(q.selection(), &input, &[1.0], &key).is_err(),
        "dot overflow"
    );
    input[128] = -huge;
    key[128] = huge;
    assert!(
        oracle::scores_f32(q.selection(), &input, &[1.0], &key).is_err(),
        "NaN dot must not disappear in ReLU"
    );
    input.fill(1.0);
    key.fill(1.0);
    assert!(
        oracle::scores_f32(q.selection(), &input, &[f32::NAN], &key).is_err(),
        "NaN weight"
    );
    let masked = PositionIndexerQuery {
        candidates: IndexerCandidates::Mask,
        ..q
    };
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            oracle::select(q, 1, 1, &[bad], None).is_err(),
            "non-finite admitted score {bad}"
        );
        assert!(
            oracle::select(masked, 1, 1, &[bad], Some(&[false])).is_ok(),
            "masked values are not read"
        );
        assert!(
            oracle::select(q, 1, 0, &[bad], None).is_ok(),
            "future data is not read"
        );
    }
}

#[test]
fn legacy_pooled_oracle_refuses_position_selection() {
    for score in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0] {
        assert!(
            kernel_core::pooled_indexer_oracle::select(query().selection(), 1, 0, &[score])
                .is_err(),
            "Position selection must use its visibility, mask and failure policy"
        );
    }
}

#[test]
fn query_positions_report_bounds_without_changing_valid_rows() {
    use kernel_core::pooled_indexer::{POSITION_QUERY_INVALID_POSITION, position_query_status};
    for position in [-1, i32::MIN, 4, i32::MAX] {
        assert_eq!(
            position_query_status(position, 4),
            POSITION_QUERY_INVALID_POSITION
        );
    }
    for position in [0, 1, 3] {
        assert_eq!(position_query_status(position, 4), 0);
    }
    assert_eq!(position_query_status(0, 0), POSITION_QUERY_INVALID_POSITION);
    assert_eq!(
        POSITION_QUERY_INVALID_POSITION
            & (kernel_core::precision::quantizer::NONFINITE_INPUT
                | kernel_core::precision::quantizer::SCALE_OVERFLOW
                | kernel_core::precision::quantizer::UNREPRESENTABLE_OUTPUT),
        0
    );
}
