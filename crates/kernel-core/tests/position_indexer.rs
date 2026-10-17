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
        oracle::select(q, 2, 5, &[-0.0, 0.0, f32::NAN], None).unwrap(),
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
