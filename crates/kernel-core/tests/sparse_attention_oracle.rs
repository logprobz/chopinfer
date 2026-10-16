use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{
    AttentionLayout, AttentionPhase, AttentionQuery, DType, NumericalContract,
    SparseAttentionQuery, Validate,
};
use kernel_core::oracle::{PagedAttention, attention};
use kernel_core::sparse_attention_oracle::{
    IndexerKeyCache, SparseKv, block_scores, block_topk, indexer_key_write, indexer_query,
    sparse_attention,
};

const MROPE: [u32; 3] = [11, 11, 10];

fn values(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn key_write(ring: u32) -> SparseAttentionQuery {
    SparseAttentionQuery::IndexerKeyWrite {
        head_dim: 128,
        rotary_dim: 64,
        theta_bits: 1e7_f32.to_bits(),
        mrope: MROPE,
        eps_bits: 1e-6_f32.to_bits(),
        block: 4,
        ring,
        capacity: 64,
    }
}

fn scores(heads: u32, max_blocks: u32) -> SparseAttentionQuery {
    SparseAttentionQuery::BlockScores {
        heads,
        head_dim: 128,
        max_blocks,
        capacity: 64,
    }
}

fn topk(top_blocks: u32) -> SparseAttentionQuery {
    SparseAttentionQuery::BlockTopK {
        block: 4,
        top_blocks,
        max_blocks: 65_536,
        capacity: 64,
    }
}

fn sparse(q_heads: u32, kv_heads: u32, head_dim: u32, top_blocks: u32) -> SparseAttentionQuery {
    SparseAttentionQuery::SparseAttention {
        q_heads,
        kv_heads,
        head_dim,
        block: 4,
        top_blocks,
        softmax_scale_bits: (1.0 / (head_dim as f32).sqrt()).to_bits(),
        page_size: 4,
        kv_dtype: DType::Bf16,
        phase: AttentionPhase::Prefill,
        capacity: 64,
    }
}

#[test]
fn qwen38_sparse_contracts_validate_with_stable_ids() {
    let query = SparseAttentionQuery::IndexerQueryGemmaNormMrope {
        heads: 4,
        head_dim: 128,
        rotary_dim: 64,
        theta_bits: 1e7_f32.to_bits(),
        mrope: MROPE,
        eps_bits: 1e-6_f32.to_bits(),
        capacity: 64,
    };
    let ids = [
        query,
        key_write(8),
        scores(4, 65_536),
        topk(512),
        sparse(6, 1, 256, 512),
    ]
    .map(|q| q.numerical_contract_id().unwrap().as_str().to_owned());
    assert_eq!(
        ids,
        [
            "sparse.indexer-query-gemma-norm-mrope.v1",
            "sparse.indexer-key-ring-block-mean-write.v1",
            "sparse.indexer-block-scores-relu-sum.v1",
            "sparse.indexer-block-topk-lower-index.v1",
            "sparse.attention-paged-kv-bf16.prefill.v1",
        ]
    );
    // Budget 2048 over blocks of 4, plus a tail of up to 3 tokens.
    assert_eq!(topk(512).dense_tokens(), Some(2051));
    assert!(key_write(6).validate().is_err());
    assert!(key_write(2).validate().is_err());
    assert!(sparse(6, 4, 256, 512).validate().is_err());
}

#[test]
fn block_keys_pool_normalize_and_rotate_at_the_first_token() {
    let d = 128;
    let raw = values(6 * d, 1)
        .into_iter()
        .map(|x| f64::from(bf16_to_f32(f32_to_bf16(x as f32))))
        .collect::<Vec<_>>();
    let weight = values(d, 2).iter().map(|x| x * 0.1).collect::<Vec<_>>();
    let rope = (0..6).map(|p| [p, p, p]).collect::<Vec<_>>();
    let mut ring = vec![0.0; 8 * d];
    let mut ring_positions = vec![[0; 3]; 8];
    let mut block_keys = vec![0.0; 2 * d];
    let writes = indexer_key_write(
        key_write(8),
        &raw,
        &[0, 1, 2, 3, 4, 5],
        &rope,
        &weight,
        IndexerKeyCache {
            ring_keys: &mut ring,
            ring_positions: &mut ring_positions,
            block_keys: &mut block_keys,
        },
    )
    .unwrap();
    assert_eq!(writes.blocks, [0]);
    assert_eq!(&ring[..6 * d], &raw[..]);
    // Block 0 key: BF16 mean, Gemma norm, then RoPE at position 0 (identity).
    let mean = (0..d)
        .map(|c| {
            let sum = (0..4).map(|t| raw[t * d + c]).sum::<f64>();
            f64::from(bf16_to_f32(f32_to_bf16((sum / 4.0) as f32)))
        })
        .collect::<Vec<_>>();
    let inv = (mean.iter().map(|x| x * x).sum::<f64>() / d as f64 + f64::from(1e-6_f32))
        .sqrt()
        .recip();
    for c in 0..d {
        let normalized = mean[c] * inv * (1.0 + weight[c]);
        let expected = if c < 64 {
            (normalized as f32) as f64
        } else {
            normalized
        };
        assert_eq!(block_keys[c], expected);
    }
    assert!(block_keys[d..].iter().all(|&x| x == 0.0));
    // Two more rows complete block 1 from ring rows written by the first call.
    let more = values(2 * d, 3);
    let writes = indexer_key_write(
        key_write(8),
        &more,
        &[6, 7],
        &[[6, 6, 6], [7, 7, 7]],
        &weight,
        IndexerKeyCache {
            ring_keys: &mut ring,
            ring_positions: &mut ring_positions,
            block_keys: &mut block_keys,
        },
    )
    .unwrap();
    assert_eq!(writes.blocks, [1]);
    let all = [raw.clone(), more.clone()].concat();
    let mut fresh_ring = vec![0.0; 8 * d];
    let mut fresh_positions = vec![[0; 3]; 8];
    let mut fresh_keys = vec![0.0; 2 * d];
    indexer_key_write(
        key_write(8),
        &all,
        &(0..8).collect::<Vec<_>>(),
        &(0..8).map(|p| [p, p, p]).collect::<Vec<_>>(),
        &weight,
        IndexerKeyCache {
            ring_keys: &mut fresh_ring,
            ring_positions: &mut fresh_positions,
            block_keys: &mut fresh_keys,
        },
    )
    .unwrap();
    assert_eq!(fresh_keys, block_keys);
    // Block 1 rotates at its first token, position 4, not at position 7.
    let pooled_at = |position: usize| {
        let mut ring = vec![0.0; 8 * d];
        let mut ring_positions = vec![[0; 3]; 8];
        let mut keys = vec![0.0; 2 * d];
        let mut rope = (0..8).map(|p| [p, p, p]).collect::<Vec<_>>();
        rope[4] = [position; 3];
        indexer_key_write(
            key_write(8),
            &all,
            &(0..8).collect::<Vec<_>>(),
            &rope,
            &weight,
            IndexerKeyCache {
                ring_keys: &mut ring,
                ring_positions: &mut ring_positions,
                block_keys: &mut keys,
            },
        )
        .unwrap();
        keys[d..].to_vec()
    };
    assert_eq!(pooled_at(4), block_keys[d..]);
    assert_ne!(pooled_at(7), block_keys[d..]);
}

#[test]
fn scores_sum_relu_dots_over_heads() {
    let d = 128;
    let mut q = vec![0.0; 2 * d];
    q[0] = 2.0;
    q[d] = -1.0;
    q[d + 1] = 3.0;
    let mut keys = vec![0.0; 3 * d];
    keys[0] = 1.0; // head 0 dot 2, head 1 dot -1
    keys[d + 1] = 1.0; // head 0 dot 0, head 1 dot 3
    keys[2 * d] = -1.0; // head 0 dot -2, head 1 dot 1
    let out = block_scores(scores(2, 4), &q, &keys).unwrap();
    let scale = 1.0 / (d as f64).sqrt();
    assert_eq!(out.values, [2.0 * scale, 3.0 * scale, 1.0 * scale]);
    assert!(block_scores(scores(2, 2), &q, &keys).is_err());
}

#[test]
fn wider_scores_use_late_heads_and_relu_before_head_sum() {
    let d = 128;
    for heads in [8, 16] {
        let mut q = vec![0.0; heads * d];
        q[0] = 2.0;
        q[(heads - 1) * d + 1] = 3.0;
        for head in 4..heads {
            q[head * d] = -16.0;
        }
        let mut keys = vec![0.0; 3 * d];
        keys[0] = 1.0;
        keys[d + 1] = 1.0;
        keys[2 * d] = 0.5;
        let out = block_scores(scores(heads as u32, 3), &q, &keys).unwrap();
        let scale = (d as f64).sqrt().recip();
        assert_eq!(out.values, [2.0 * scale, 3.0 * scale, scale]);
        // The last head makes block 1 win; head 0 keeps block 0 ahead of 2
        // even though the negative late heads would cancel it before ReLU.
        assert_eq!(block_topk(topk(1), &out.values, 12).unwrap(), [4, 5, 6, 7]);
        assert_eq!(
            block_topk(topk(2), &out.values, 12).unwrap(),
            (0..8).collect::<Vec<_>>()
        );
    }
}

#[test]
fn wider_scores_keep_lower_blocks_for_zero_equal_and_mixed_head_ties() {
    let d = 128;
    for heads in [8, 16] {
        let mut keys = vec![0.0; 3 * d];
        for block in 0..3 {
            keys[block * d + block] = 1.0;
        }
        for dot in [0.0, -2.0, 2.0] {
            let q = vec![dot; heads * d];
            let out = block_scores(scores(heads as u32, 3), &q, &keys).unwrap();
            let expected = (dot * heads as f64).max(0.0) / (d as f64).sqrt();
            assert_eq!(out.values, [expected; 3]);
            assert_eq!(
                block_topk(topk(2), &out.values, 14).unwrap(),
                [0, 1, 2, 3, 4, 5, 6, 7, 12, 13]
            );
        }
        let mut q = vec![0.0; heads * d];
        q[0] = 3.0;
        q[d + 1] = 1.0;
        q[(heads - 1) * d + 1] = 2.0;
        q[(heads - 1) * d + 2] = 3.0;
        let out = block_scores(scores(heads as u32, 3), &q, &keys).unwrap();
        assert_eq!(out.values, [3.0 / (d as f64).sqrt(); 3]);
        assert_eq!(
            block_topk(topk(2), &out.values, 12).unwrap(),
            (0..8).collect::<Vec<_>>()
        );
    }
}

#[test]
fn indexer_key_rotary_32_uses_half_split_f32_fma() {
    let d = 128;
    let raw = values(4 * d, 91);
    let weight = values(d, 92);
    let positions = [262143, 513, 37];
    let mut ring = vec![0.0; 4 * d];
    let mut ring_positions = vec![[0; 3]; 4];
    let mut keys = vec![0.0; d];
    let out = indexer_key_write(
        SparseAttentionQuery::IndexerKeyWrite {
            head_dim: 128,
            rotary_dim: 32,
            theta_bits: 1e7_f32.to_bits(),
            mrope: [6, 5, 5],
            eps_bits: 1e-6_f32.to_bits(),
            block: 4,
            ring: 4,
            capacity: 4,
        },
        &raw,
        &[0, 1, 2, 3],
        &[positions, [1; 3], [2; 3], [3; 3]],
        &weight,
        IndexerKeyCache {
            ring_keys: &mut ring,
            ring_positions: &mut ring_positions,
            block_keys: &mut keys,
        },
    )
    .unwrap();
    let pooled: Vec<f64> = (0..d)
        .map(|col| {
            let mean = (0..4).map(|row| raw[row * d + col]).sum::<f64>() / 4.0;
            f64::from(bf16_to_f32(f32_to_bf16(mean as f32)))
        })
        .collect();
    let inverse = (pooled.iter().map(|x| x * x).sum::<f64>() / d as f64 + f64::from(1e-6_f32))
        .sqrt()
        .recip();
    let normed: Vec<f64> = pooled
        .iter()
        .zip(&weight)
        .map(|(x, w)| x * inverse * (1.0 + w))
        .collect();
    let mut expected = normed.clone();
    for pair in 0..16 {
        let axis = if pair % 3 == 1 && pair < 15 {
            1
        } else if pair % 3 == 2 && pair < 15 {
            2
        } else {
            0
        };
        let cs = kernel_core::hf_rope::cos_sin(positions[axis], 2 * pair, 1e7);
        let (x, y) = (normed[pair] as f32, normed[pair + 16] as f32);
        let (c, s) = (cs.0 as f32, cs.1 as f32);
        expected[pair] = f64::from(x.mul_add(c, -(y * s)));
        expected[pair + 16] = f64::from(y.mul_add(c, x * s));
    }
    assert_eq!(out.blocks, [0]);
    assert_eq!(out.keys.values, expected);
    assert_eq!(keys, expected);
}

#[test]
fn topk_prefers_higher_scores_then_lower_blocks_and_appends_the_tail() {
    let scores = [0.5, 2.0, 0.5, 2.0, 1.0];
    let tokens = block_topk(topk(3), &scores, 22).unwrap();
    // Blocks 1 and 3 (score 2), then block 4; the tail is tokens 20 and 21.
    assert_eq!(tokens, [4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]);
    let tokens = block_topk(topk(2), &[1.0, 1.0, 1.0], 12).unwrap();
    assert_eq!(tokens, [0, 1, 2, 3, 4, 5, 6, 7]);
    assert!(block_topk(topk(2), &[1.0; 2], 12).is_err());
    assert!(block_topk(topk(2), &[f64::NAN; 3], 12).is_err());
}

#[test]
fn rows_up_to_2051_tokens_attend_densely() {
    for visible in [1, 3, 4, 2048, 2051] {
        let blocks = visible / 4;
        let tokens = block_topk(topk(512), &vec![0.0; blocks], visible).unwrap();
        assert_eq!(tokens, (0..visible).collect::<Vec<_>>());
    }
    // 513 blocks: equal (all-zero ReLU) scores drop the highest block.
    let tokens = block_topk(topk(512), &[0.0; 513], 2052).unwrap();
    assert_eq!(tokens, (0..2048).collect::<Vec<_>>());
    let tokens = block_topk(topk(512), &[0.0; 513], 2055).unwrap();
    assert_eq!(tokens.len(), 2051);
    assert_eq!(&tokens[2048..], &[2052, 2053, 2054]);
}

#[test]
fn sparse_attention_with_every_token_equals_dense_paged_attention() {
    check_sparse_equals_dense(64);
}

#[test]
fn head_128_sparse_equals_dense_paged_attention() {
    check_sparse_equals_dense(128);
}

fn check_sparse_equals_dense(d: usize) {
    let (tokens, qh, kh) = (7, 12, 2);
    let q = values(tokens * qh * d, 5);
    let keys = values(2 * 4 * kh * d, 6);
    let vals = values(2 * 4 * kh * d, 7);
    let table = [1, 0];
    let key_sets = (0..tokens).map(|p| (0..=p).collect()).collect::<Vec<_>>();
    let sparse_out = sparse_attention(
        sparse(12, 2, d as u32, 512),
        &q,
        &key_sets,
        SparseKv {
            keys: &keys,
            values: &vals,
            block_table: &table,
        },
    )
    .unwrap();
    let dense = attention(
        &AttentionQuery {
            q_heads: 12,
            kv_heads: 2,
            qk_dim: d as u32,
            v_dim: d as u32,
            page_size: 4,
            kv_dtype: DType::Bf16,
            layout: AttentionLayout::Paged,
            causal: true,
            window_left: None,
            sink: false,
            softcap: None,
            softmax_scale: 1.0 / (d as f32).sqrt(),
            phase: AttentionPhase::Prefill,
            max_context_capacity: 8,
            max_batch_capacity: 1,
        },
        &PagedAttention {
            q: &q,
            keys: &keys,
            values: &vals,
            page_tables: &[table.to_vec()],
            query_lengths: &[tokens],
            query_starts: &[0],
            key_lengths: &[tokens],
            sink_logits: None,
        },
    )
    .unwrap();
    let difference = sparse_out
        .values
        .iter()
        .zip(&dense.values)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(difference < 1e-14);
    assert_eq!(sparse_out.reduction_lengths, dense.reduction_lengths);
}

#[test]
fn sparse_attention_reads_only_the_selected_tokens_for_all_heads() {
    let (qh, d) = (6, 64);
    let q = values(qh * d, 8);
    let mut keys = values(4 * 4 * d, 9);
    let mut vals = values(4 * 4 * d, 10);
    let set = vec![vec![0, 1, 2, 3, 12, 13]];
    let run = |keys: &[f64], vals: &[f64]| {
        sparse_attention(
            sparse(6, 1, 64, 2),
            &q,
            &set,
            SparseKv {
                keys,
                values: vals,
                block_table: &[0, 1, 2, 3],
            },
        )
        .unwrap()
        .values
    };
    let before = run(&keys, &vals);
    // Tokens 4..=11 are unselected: changing them changes nothing.
    for x in &mut keys[4 * d..12 * d] {
        *x += 1.0;
    }
    for x in &mut vals[4 * d..12 * d] {
        *x -= 1.0;
    }
    assert_eq!(run(&keys, &vals), before);
    // One shared value for the selected tokens makes every head return it.
    let mut flat = vals.clone();
    for &t in &set[0] {
        flat[t * d..(t + 1) * d].fill(0.75);
    }
    assert!(run(&keys, &flat).iter().all(|&x| (x - 0.75).abs() < 1e-15));
    // A key set longer than the budget plus tail is refused.
    let long = vec![(0..12).collect::<Vec<_>>()];
    assert!(
        sparse_attention(
            sparse(6, 1, 64, 2),
            &q,
            &long,
            SparseKv {
                keys: &keys,
                values: &vals,
                block_table: &[0, 1, 2, 3],
            },
        )
        .is_err()
    );
}

#[test]
fn indexer_pipeline_selects_per_row_blocks_from_the_block_cache() {
    check_indexer_pipeline(4, 64, MROPE);
}

#[test]
fn wider_indexer_heads_and_rotary_32_select_their_own_scores() {
    for heads in [4, 8, 16] {
        for (rotary, sections) in [(32, [6, 5, 5]), (64, MROPE)] {
            check_indexer_pipeline(heads, rotary, sections);
        }
    }
}

fn check_indexer_pipeline(heads: u32, rotary: u32, sections: [u32; 3]) {
    let (d, tokens) = (128, 23);
    let query = SparseAttentionQuery::IndexerQueryGemmaNormMrope {
        heads,
        head_dim: 128,
        rotary_dim: rotary,
        theta_bits: 1e7_f32.to_bits(),
        mrope: sections,
        eps_bits: 1e-6_f32.to_bits(),
        capacity: 64,
    };
    let rope = (0..tokens).map(|p| [p, p, p]).collect::<Vec<_>>();
    let q = indexer_query(
        query,
        &values(tokens * heads as usize * d, 11),
        &vec![0.0; d],
        &rope,
        None,
    )
    .unwrap();
    let mut ring = vec![0.0; 4 * d];
    let mut ring_positions = vec![[0; 3]; 4];
    let mut block_keys = vec![0.0; 5 * d];
    let positions = (0..tokens).collect::<Vec<_>>();
    indexer_key_write(
        SparseAttentionQuery::IndexerKeyWrite {
            head_dim: 128,
            rotary_dim: rotary,
            theta_bits: 1e7_f32.to_bits(),
            mrope: sections,
            eps_bits: 1e-6_f32.to_bits(),
            block: 4,
            ring: 4,
            capacity: 64,
        },
        &values(tokens * d, 12),
        &positions,
        &rope,
        &vec![0.0; d],
        IndexerKeyCache {
            ring_keys: &mut ring,
            ring_positions: &mut ring_positions,
            block_keys: &mut block_keys,
        },
    )
    .unwrap();
    // Each row scores its complete blocks of the block cache, then selects.
    let per_head = heads as usize * d;
    for position in positions {
        let visible = position + 1;
        let row_scores = block_scores(
            scores(heads, 8),
            &q.values[position * per_head..(position + 1) * per_head],
            &block_keys[..visible / 4 * d],
        )
        .unwrap();
        let set = block_topk(topk(2), &row_scores.values, visible).unwrap();
        if visible <= 11 {
            assert_eq!(set, (0..visible).collect::<Vec<_>>());
        } else {
            assert_eq!(set.len(), 8 + visible % 4);
        }
        assert!(set.contains(&position) || position % 4 == 3);
        assert!(set.windows(2).all(|w| w[0] < w[1]));
    }
}
