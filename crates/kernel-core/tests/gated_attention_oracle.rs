use kernel_core::contract::{
    AttentionLayout, AttentionPhase, AttentionQuery, DType, GatedAttentionQuery, NumericalContract,
    Validate,
};
use kernel_core::gated_attention_oracle::{
    GatedAttentionInputs, PagedKvPair, attention_output_gate, mrope_axis, mrope_cos_sin,
    qk_norm_rope_kv_write, rope_normalized_f32,
};
use kernel_core::oracle::{PagedAttention, attention};
use kernel_core::recurrent_oracle::sigmoid;

const QWEN38_MROPE: [u32; 3] = [11, 11, 10];

fn prep(q_heads: u32, kv_heads: u32, head_dim: u32, rotary_dim: u32) -> GatedAttentionQuery {
    GatedAttentionQuery::QkNormRopeKvWrite {
        q_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        theta_bits: 1e7_f32.to_bits(),
        mrope: QWEN38_MROPE,
        eps_bits: 1e-6_f32.to_bits(),
        page_size: 4,
        kv_dtype: DType::Bf16,
        capacity: 64,
    }
}

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

#[test]
fn qwen38_gated_attention_contracts_validate_with_stable_ids() {
    // TP4: 6 query heads and one replicated KV head per rank.
    let rank = prep(6, 1, 256, 64);
    assert_eq!(
        rank.numerical_contract_id().unwrap().as_str(),
        "row.qk-gemma-norm-mrope-kv-write-bf16.v1"
    );
    let gate = GatedAttentionQuery::OutputGate {
        heads: 6,
        head_dim: 256,
        capacity: 64,
    };
    assert_eq!(
        gate.numerical_contract_id().unwrap().as_str(),
        "row.attention-output-sigmoid-gate.v1"
    );
    let GatedAttentionQuery::QkNormRopeKvWrite {
        q_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        theta_bits,
        eps_bits,
        page_size,
        kv_dtype,
        capacity,
        ..
    } = rank
    else {
        unreachable!()
    };
    let with_sections = |mrope| GatedAttentionQuery::QkNormRopeKvWrite {
        q_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        theta_bits,
        mrope,
        eps_bits,
        page_size,
        kv_dtype,
        capacity,
    };
    assert!(with_sections([0; 3]).validate().is_ok());
    assert!(with_sections([11, 11, 11]).validate().is_err());
    assert!(prep(6, 4, 256, 64).validate().is_err());
    assert!(prep(6, 1, 256, 320).validate().is_err());
    for (q_heads, kv_heads) in [(6, 1), (24, 2)] {
        let query = AttentionQuery {
            q_heads,
            kv_heads,
            qk_dim: 256,
            v_dim: 256,
            page_size: 16,
            kv_dtype: DType::Bf16,
            layout: AttentionLayout::Paged,
            causal: true,
            window_left: None,
            sink: false,
            softcap: None,
            softmax_scale: 1.0 / 16.0,
            phase: AttentionPhase::Decode,
            max_context_capacity: 262_144,
            max_batch_capacity: 64,
        };
        query.validate().unwrap();
    }
}

#[test]
fn interleaved_mrope_slots_follow_the_reference_slices() {
    // HF: H overwrites slice(1, 33, 3) and W overwrites slice(2, 30, 3).
    let axes = (0..32)
        .map(|p| mrope_axis(p, QWEN38_MROPE))
        .collect::<Vec<_>>();
    let h = (0..32).filter(|&p| axes[p] == 1).collect::<Vec<_>>();
    let w = (0..32).filter(|&p| axes[p] == 2).collect::<Vec<_>>();
    let t = (0..32).filter(|&p| axes[p] == 0).collect::<Vec<_>>();
    assert_eq!(h, (1..33).step_by(3).collect::<Vec<_>>());
    assert_eq!(w, (2..30).step_by(3).collect::<Vec<_>>());
    assert_eq!(t.len(), 11);
    assert_eq!(*t.last().unwrap(), 30);
    let table = mrope_cos_sin([5, 7, 9], 64, 1e7, QWEN38_MROPE);
    for (pair, &(cos, sin)) in table.iter().enumerate() {
        let position = [5, 7, 9][axes[pair]];
        assert_eq!(
            (cos, sin),
            kernel_core::hf_rope::cos_sin(position, pair, 1e7)
        );
    }
}

#[test]
fn text_positions_reduce_to_partial_neox_rope() {
    check_text_positions(64, [11, 11, 10]);
}

#[test]
fn head_128_rotary_32_norm_rope_and_paged_write() {
    check_text_positions(32, [6, 5, 5]);
}

fn check_text_positions(rotary: usize, sections: [u32; 3]) {
    let (heads, head_dim) = (2, 128);
    let input = values(3 * heads * head_dim, 3);
    let positions = [[0, 0, 0], [5, 5, 5], [13, 13, 13]];
    let mut query = prep(2, 2, 128, rotary as u32);
    if let GatedAttentionQuery::QkNormRopeKvWrite { mrope, .. } = &mut query {
        *mrope = sections;
    }
    let mut keys = vec![0.0; 4 * 4 * 2 * 128];
    let mut vals = keys.clone();
    let q_gate = input
        .chunks_exact(head_dim)
        .flat_map(|head| head.iter().chain(head.iter()).copied())
        .collect::<Vec<_>>();
    let ones = vec![0.0; head_dim];
    let rows = qk_norm_rope_kv_write(
        query,
        GatedAttentionInputs {
            q_gate: &q_gate,
            k: &input,
            v: &input,
            q_norm: &ones,
            k_norm: &ones,
            rope_positions: &positions,
            token_positions: &[0, 5, 13],
            tables: None,
        },
        PagedKvPair {
            keys: &mut keys,
            values: &mut vals,
            block_table: &[3, 1, 0, 2],
        },
    )
    .unwrap();
    // Gemma with zero weight is a plain RMSNorm; then partial NeoX RoPE.
    let normed = input
        .chunks_exact(head_dim)
        .flat_map(|head| {
            let inv = (head.iter().map(|x| x * x).sum::<f64>() / head_dim as f64
                + f64::from(1e-6_f32))
            .sqrt()
            .recip();
            head.iter().map(move |x| x * inv).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let expanded = positions
        .iter()
        .flat_map(|p| std::iter::repeat_n(p[0], heads))
        .collect::<Vec<_>>();
    let expected = rope_normalized_f32(&normed, &expanded, head_dim, rotary, None).unwrap();
    let difference = rows
        .q
        .values
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(difference < 1e-12);
    assert_eq!(rows.k.values, rows.q.values);
    assert_eq!(rows.gate, input);
    // Position 13 is slot 1 of logical page 3, which maps to physical page 2.
    let slot = (2 * 4 + 1) * 2 * 128;
    assert_eq!(&keys[slot..slot + 256], &rows.k.values[512..]);
    assert_eq!(&vals[slot..slot + 256], &input[512..]);
}

#[test]
fn gemma_norm_and_gate_split_match_hand_computation() {
    let d = 64;
    let mut q_gate = vec![0.0; 2 * d];
    q_gate[0] = 2.0;
    q_gate[d] = -3.0;
    let mut k = vec![0.0; d];
    k[1] = 4.0;
    let v = vec![1.5; d];
    let mut weight = vec![0.0; d];
    weight[0] = 0.5;
    weight[1] = -0.25;
    let mut keys = vec![0.0; 4 * d];
    let mut vals = keys.clone();
    let rows = qk_norm_rope_kv_write(
        prep(1, 1, 64, 64),
        GatedAttentionInputs {
            q_gate: &q_gate,
            k: &k,
            v: &v,
            q_norm: &weight,
            k_norm: &weight,
            rope_positions: &[[9, 4, 4]],
            token_positions: &[2],
            tables: Some(&[vec![(1.0, 0.0); 32]]),
        },
        PagedKvPair {
            keys: &mut keys,
            values: &mut vals,
            block_table: &[0],
        },
    )
    .unwrap();
    let eps = f64::from(1e-6_f32);
    assert_eq!(
        rows.q.values[0],
        ((2.0 / (4.0 / 64.0 + eps).sqrt() * 1.5) as f32) as f64
    );
    assert_eq!(
        rows.k.values[1],
        ((4.0 / (16.0 / 64.0 + eps).sqrt() * 0.75) as f32) as f64
    );
    assert_eq!(rows.gate[0], -3.0);
    assert_eq!(&keys[2 * d..3 * d], &rows.k.values[..]);
    assert_eq!(&vals[2 * d..3 * d], &v[..]);
    let gated = attention_output_gate(
        GatedAttentionQuery::OutputGate {
            heads: 1,
            head_dim: 2,
            capacity: 1,
        },
        &[2.0, -4.0],
        &[0.0, 1.0],
    )
    .unwrap();
    assert_eq!(gated.values, [1.0, -4.0 * sigmoid(1.0)]);
}

#[test]
fn group_six_rank_attention_equals_a_slice_of_the_full_layer() {
    let (tokens, d) = (5, 64);
    let full = AttentionQuery {
        q_heads: 24,
        kv_heads: 2,
        qk_dim: 64,
        v_dim: 64,
        page_size: 8,
        kv_dtype: DType::Bf16,
        layout: AttentionLayout::Paged,
        causal: true,
        window_left: None,
        sink: false,
        softcap: None,
        softmax_scale: 0.125,
        phase: AttentionPhase::Prefill,
        max_context_capacity: 8,
        max_batch_capacity: 1,
    };
    let q = values(tokens * 24 * d, 5);
    let keys = values(8 * 2 * d, 6);
    let vals = values(8 * 2 * d, 7);
    let table = vec![vec![0]];
    let run = |query: &AttentionQuery, q: &[f64], keys: &[f64], vals: &[f64]| {
        attention(
            query,
            &PagedAttention {
                q,
                keys,
                values: vals,
                page_tables: &table,
                query_lengths: &[tokens],
                query_starts: &[0],
                key_lengths: &[tokens],
                sink_logits: None,
            },
        )
        .unwrap()
        .values
    };
    let reference = run(&full, &q, &keys, &vals);
    let rank_query = AttentionQuery {
        q_heads: 6,
        kv_heads: 1,
        ..full
    };
    for rank in 0..4 {
        let kv = rank / 2;
        let q_rank = (0..tokens)
            .flat_map(|t| q[(t * 24 + 6 * rank) * d..(t * 24 + 6 * rank + 6) * d].to_vec())
            .collect::<Vec<_>>();
        let slice = |cache: &[f64]| {
            (0..8)
                .flat_map(|t| cache[(t * 2 + kv) * d..(t * 2 + kv + 1) * d].to_vec())
                .collect::<Vec<_>>()
        };
        let out = run(&rank_query, &q_rank, &slice(&keys), &slice(&vals));
        for t in 0..tokens {
            assert_eq!(
                out[t * 6 * d..(t + 1) * 6 * d],
                reference[(t * 24 + 6 * rank) * d..(t * 24 + 6 * rank + 6) * d]
            );
        }
    }
}

#[test]
fn qwen_reference_matches_independent_torch_fixture_bits() {
    use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
    let fixture = include_bytes!("../../../fixtures/qwen_rope_cpu.bin");
    for (pair, word) in fixture[..128].chunks_exact(4).enumerate() {
        assert_eq!(
            u32::from_le_bytes(word.try_into().unwrap()),
            kernel_core::hf_rope::INVERSE_BITS[1][pair]
        );
    }
    for row in fixture[128..].chunks_exact(17) {
        let pair = row[0] as usize;
        let position = u32::from_le_bytes(row[1..5].try_into().unwrap()) as usize;
        let words: Vec<_> = row[5..]
            .chunks_exact(2)
            .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
            .collect();
        let table = mrope_cos_sin([position; 3], 64, 1e7, QWEN38_MROPE);
        assert_eq!(f32_to_bf16(table[pair].0 as f32), words[2]);
        assert_eq!(f32_to_bf16(table[pair].1 as f32), words[3]);
        let mut input = vec![0.0; 128];
        input[pair] = bf16_to_f32(words[0]) as f64;
        input[pair + 32] = bf16_to_f32(words[1]) as f64;
        input[127] = 0.123456789;
        let output = rope_normalized_f32(&input, &[0], 128, 64, Some(&table)).unwrap();
        let (x, y) = (input[pair], input[pair + 32]);
        let (c, s) = table[pair];
        assert_eq!(output[pair], ((x * c - y * s) as f32) as f64);
        assert_eq!(output[pair + 32], ((y * c + x * s) as f32) as f64);
        assert_eq!(
            f32_to_bf16(kernel_core::hf_rope::rotate_pure_hf(x, y, (c, s), false) as f32),
            words[4]
        );
        assert_eq!(
            f32_to_bf16(kernel_core::hf_rope::rotate_pure_hf(y, x, (c, s), true) as f32),
            words[5]
        );
        assert_eq!(output[127], input[127]);
    }
}

#[test]
fn qwen_rotation_preserves_f32_normalized_operands() {
    let input = vec![1.001_f32 as f64; 64];
    let table = vec![(1.0, 0.0); 32];
    let output = rope_normalized_f32(&input, &[0], 64, 64, Some(&table)).unwrap();
    assert_eq!(output, input);
    assert_ne!(
        output[0],
        kernel_core::hf_rope::round_bf16(input[0] as f32) as f64
    );
}

#[test]
fn served_rotation_obeys_pure_hf_distance_bound() {
    use kernel_core::codec::bf16_to_f32;
    use kernel_core::hf_rope::{pure_hf_distance_bound, rotate_pure_hf, round_bf16};
    use kernel_core::rotary::RotaryArithmetic;
    let served = |x: f32, y: f32, (cos, sin): (f64, f64), second| {
        RotaryArithmetic::F32Fma.element(f64::from(x), f64::from(y), cos, sin, second) as f32
    };
    let fixture = include_bytes!("../../../fixtures/qwen_rope_cpu.bin");
    let mut differences = 0;
    for row in fixture[128..].chunks_exact(17) {
        let words: Vec<_> = row[5..]
            .chunks_exact(2)
            .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
            .collect();
        let x = bf16_to_f32(words[0]);
        let y = bf16_to_f32(words[1]);
        let cs = (bf16_to_f32(words[2]) as f64, bf16_to_f32(words[3]) as f64);
        for (x, y, second) in [(x, y, false), (y, x, true)] {
            let actual = round_bf16(served(x, y, cs, second)) as f64;
            let hf = rotate_pure_hf(x as f64, y as f64, cs, second);
            differences += usize::from(actual != hf);
            assert!((actual - hf).abs() <= pure_hf_distance_bound(x, y, cs, second));
            let exact = x as f64 * cs.0
                + if second {
                    y as f64 * cs.1
                } else {
                    -(y as f64 * cs.1)
                };
            assert_eq!(actual, round_bf16(exact as f32) as f64);
        }
    }
    assert!(
        differences > 0,
        "fixture must distinguish the rejected BF16 products"
    );
}

#[test]
fn rotary_32_uses_hf_tables_and_f32_normalized_rotation() {
    let positions = [262143, 513, 37];
    let sections = [6, 5, 5];
    let table = mrope_cos_sin(positions, 32, 1e7, sections);
    let input = values(128, 91);
    let output = rope_normalized_f32(&input, &[0], 128, 32, Some(&table)).unwrap();
    for pair in 0..16 {
        let cs =
            kernel_core::hf_rope::cos_sin(positions[mrope_axis(pair, sections)], 2 * pair, 1e7);
        assert_eq!(table[pair], cs);
        let (x, y) = (input[pair] as f32, input[pair + 16] as f32);
        let (c, s) = (cs.0 as f32, cs.1 as f32);
        assert_eq!(
            output[pair].to_bits(),
            f64::from(x.mul_add(c, -(y * s))).to_bits()
        );
        assert_eq!(
            output[pair + 16].to_bits(),
            f64::from(y.mul_add(c, x * s)).to_bits()
        );
    }
    assert_eq!(&output[32..], &input[32..]);
}
