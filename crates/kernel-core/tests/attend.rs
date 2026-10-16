use kernel_core::attend::{
    Attend, AttendRounding, AttendValues, KeyBias, KvRows, Normalize, attend, paged_rows,
    selected_blocks, window,
};
use kernel_core::contract::DType;
use kernel_core::oracle::bf16_round;

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

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-13 * (1.0 + b.abs())
}

/// Softmax attention of one head over explicit (key, value) rows with an
/// optional sink logit, as written in DeepSeek's `sparse_attn`.
fn reference(q: &[f64], rows: &[(&[f64], &[f64])], scale: f64, sink: Option<f64>) -> Vec<f64> {
    let logits = rows
        .iter()
        .map(|(k, _)| q.iter().zip(*k).map(|(a, b)| a * b).sum::<f64>() * scale)
        .collect::<Vec<_>>();
    let max = logits
        .iter()
        .copied()
        .chain(sink)
        .fold(f64::NEG_INFINITY, f64::max);
    let denominator =
        logits.iter().map(|l| (l - max).exp()).sum::<f64>() + sink.map_or(0.0, |s| (s - max).exp());
    let dim = rows.first().map_or(0, |(_, v)| v.len());
    (0..dim)
        .map(|d| {
            rows.iter()
                .zip(&logits)
                .map(|((_, v), l)| (l - max).exp() / denominator * v[d])
                .sum()
        })
        .collect()
}

#[test]
fn shared_latent_with_sinks_over_index_lists_matches_direct_math() {
    // DeepSeek-style MQA: 4 query heads, one 32-wide latent that is both K
    // and V, a learned sink per head, and fixed-width top-k lists padded
    // with -1. Keys come from two segments (a window ring and compressed
    // entries).
    let (heads, dim) = (4, 32);
    let ring = values(5 * dim, 1);
    let compressed = values(3 * dim, 2);
    let segments = [ring.as_slice(), compressed.as_slice()];
    let latent = KvRows {
        segments: &segments,
        heads: 1,
        dim,
    };
    let lists = vec![
        vec![0, 6, -1, 2, -1],
        vec![-1, -1, -1, -1, -1],
        vec![7, 5, 4, 3, 1],
    ];
    let q = values(lists.len() * heads * dim, 3);
    let sinks = [0.5, -1.0, 2.0, 0.0];
    let scale = (dim as f64).powf(-0.5);
    let params = Attend {
        sink: Some(&sinks),
        ..Attend::f64(scale)
    };
    let out = attend(&q, latent, AttendValues::SharedKeys, &lists, &params).unwrap();
    let row_of = |index: i64| {
        let index = index as usize;
        if index < 5 {
            &ring[index * dim..(index + 1) * dim]
        } else {
            &compressed[(index - 5) * dim..(index - 4) * dim]
        }
    };
    for (row, list) in lists.iter().enumerate() {
        let keys = list
            .iter()
            .filter(|&&i| i >= 0)
            .map(|&i| (row_of(i), row_of(i)))
            .collect::<Vec<_>>();
        for (head, &sink) in sinks.iter().enumerate() {
            let at = (row * heads + head) * dim;
            let expected = reference(&q[at..at + dim], &keys, scale, Some(sink));
            if keys.is_empty() {
                // Only the sink: the output is zero.
                assert!(out.values[at..at + dim].iter().all(|&v| v == 0.0));
            }
            for (d, e) in expected.iter().enumerate() {
                assert!(close(out.values[at + d], *e));
            }
            assert_eq!(out.reduction_lengths[at], keys.len() + 1);
        }
    }
    // Without a sink, an all-empty list is refused.
    let err = attend(
        &q,
        latent,
        AttendValues::SharedKeys,
        &lists,
        &Attend::f64(scale),
    );
    assert_eq!(err.unwrap_err().0, "attention row has no visible key");
}

#[test]
fn gqa_softcap_bias_and_normalization_points_are_explicit() {
    let (q_heads, kv_heads, dim, v_dim) = (4, 2, 8, 6);
    let k = values(4 * kv_heads * dim, 4);
    let v = values(4 * kv_heads * v_dim, 5);
    let (ks, vs) = ([k.as_slice()], [v.as_slice()]);
    let keys = KvRows {
        segments: &ks,
        heads: kv_heads,
        dim,
    };
    let vals = KvRows {
        segments: &vs,
        heads: kv_heads,
        dim: v_dim,
    };
    let q = values(q_heads * dim, 6);
    let list = vec![vec![3, 0, 2]];
    let bias = [0.25, -0.5, 1.0, 0.0];
    let params = Attend {
        softcap: Some(0.7),
        key_bias: Some(KeyBias {
            rows: &[0, 2],
            per_head: &bias,
        }),
        normalize: Normalize::Weights,
        ..Attend::f64(0.5)
    };
    let out = attend(&q, keys, AttendValues::Rows(vals), &list, &params).unwrap();
    for head in 0..q_heads {
        let kv = head / 2;
        let q = &q[head * dim..(head + 1) * dim];
        let logits = [3, 0, 2].map(|row: usize| {
            let key = &k[(row * kv_heads + kv) * dim..][..dim];
            let dot = q.iter().zip(key).map(|(a, b)| a * b).sum::<f64>() * 0.5;
            0.7 * (dot / 0.7).tanh() + if row != 3 { bias[head] } else { 0.0 }
        });
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total = logits.iter().map(|l| (l - max).exp()).sum::<f64>();
        for d in 0..v_dim {
            let expected = [3, 0, 2]
                .iter()
                .zip(&logits)
                .map(|(&row, l)| (l - max).exp() / total * v[(row * kv_heads + kv) * v_dim + d])
                .sum::<f64>();
            assert!(close(out.values[head * v_dim + d], expected));
        }
    }
    // Normalizing after the sum gives the same value up to rounding.
    let after = attend(
        &q,
        keys,
        AttendValues::Rows(vals),
        &list,
        &Attend {
            normalize: Normalize::AfterSum,
            ..params
        },
    )
    .unwrap();
    for (a, b) in after.values.iter().zip(&out.values) {
        assert!(close(*a, *b));
    }
}

#[test]
fn rounding_points_round_where_declared() {
    let dim = 4;
    let k = [1.0, 2.0, 3.0, 4.0, 0.5, 0.25, 0.125, 1.0 / 3.0];
    let v = [1.0 / 3.0, 0.1, 0.7, 1.1, 0.2, 0.3, 0.9, 0.6];
    let (ks, vs) = ([k.as_slice()], [v.as_slice()]);
    let keys = KvRows {
        segments: &ks,
        heads: 1,
        dim,
    };
    let vals = KvRows {
        segments: &vs,
        heads: 1,
        dim,
    };
    let q = [0.3, 0.1, 0.2, 0.7];
    let rounding = AttendRounding {
        dot: Some(DType::Bf16),
        logit: Some(DType::Bf16),
        probability: Some(DType::Bf16),
        output: Some(DType::Bf16),
    };
    let params = Attend {
        normalize: Normalize::Weights,
        rounding,
        ..Attend::f64(0.5)
    };
    let out = attend(&q, keys, AttendValues::Rows(vals), &[vec![0, 1]], &params).unwrap();
    let logits = [0, 1].map(|row: usize| {
        let mut dot = 0.0;
        for d in 0..dim {
            dot += q[d] * k[row * dim + d];
        }
        bf16_round(bf16_round(dot) * 0.5)
    });
    let max = logits[0].max(logits[1]);
    let total = (logits[0] - max).exp() + (logits[1] - max).exp();
    for d in 0..dim {
        let mut sum = 0.0;
        for (row, l) in logits.iter().enumerate() {
            sum += bf16_round((l - max).exp() / total) * v[row * dim + d];
        }
        assert_eq!(out.values[d], bf16_round(sum));
    }
    // A probability rounding point needs per-weight normalization, and only
    // BF16 rounding is implemented.
    for bad in [
        Attend {
            normalize: Normalize::AfterSum,
            ..params
        },
        Attend {
            rounding: AttendRounding {
                output: Some(DType::Fp8E4M3),
                ..rounding
            },
            ..params
        },
    ] {
        assert!(attend(&q, keys, AttendValues::Rows(vals), &[vec![0]], &bad).is_err());
    }
}

#[test]
fn malformed_lists_and_parameters_are_refused() {
    let k = values(3 * 8, 7);
    let ks = [k.as_slice()];
    let keys = KvRows {
        segments: &ks,
        heads: 1,
        dim: 8,
    };
    let q = values(2 * 8, 8);
    let ok = Attend::f64(1.0);
    for (list, params) in [
        (vec![3], ok),
        (vec![-2], ok),
        (vec![0], Attend::f64(0.0)),
        (vec![0], Attend::f64(f64::NAN)),
        (
            vec![0],
            Attend {
                softcap: Some(-1.0),
                ..ok
            },
        ),
        (
            vec![0],
            Attend {
                sink: Some(&[0.0]),
                ..ok
            },
        ),
        (
            vec![0],
            Attend {
                key_bias: Some(KeyBias {
                    rows: &[2, 1],
                    per_head: &[0.0, 0.0],
                }),
                ..ok
            },
        ),
    ] {
        assert!(attend(&q, keys, AttendValues::SharedKeys, &[list], &params).is_err());
    }
    assert!(attend(&q, keys, AttendValues::SharedKeys, &[vec![0, 1, 2]], &ok).is_ok());
}

#[test]
fn index_list_generators_cover_masks_pages_and_blocks() {
    // Causal with a left window, bidirectional, and a right window.
    assert_eq!(window(5, 0..9, true, Some(2), None), 3..6);
    assert_eq!(window(5, 0..9, true, None, None), 0..6);
    assert_eq!(window(5, 4..9, false, None, None), 4..9);
    assert_eq!(window(5, 4..9, false, Some(0), Some(1)), 5..7);
    assert!(window(2, 3..9, true, None, None).is_empty());
    // A page table maps positions to cache rows and rejects missing pages.
    assert_eq!(
        paged_rows([0, 3, 4, 9], &[2, 0, 1], 4, 3),
        Some(vec![8, 11, 0, 5])
    );
    assert_eq!(paged_rows([12], &[2, 0, 1], 4, 3), None);
    assert_eq!(paged_rows([0], &[3], 4, 3), None);
    // Selected blocks in selection order, then the incomplete tail.
    assert_eq!(selected_blocks(&[2, 0], 2, 7), vec![4, 5, 0, 1, 6]);
}

#[test]
fn logit_reduction_bound_adds_the_documented_term() {
    let (heads, dim) = (2, 4);
    let k = values(3 * dim, 9);
    let v = values(3 * dim, 10);
    let (ks, vs) = ([k.as_slice()], [v.as_slice()]);
    let rows = |segments| KvRows {
        segments,
        heads: 1,
        dim,
    };
    let q = values(heads * dim, 11);
    let list = [vec![2, 0, 1]];
    let plain = Attend {
        normalize: Normalize::Weights,
        ..Attend::f64(0.5)
    };
    let bounded = Attend {
        logit_reduction_bound: true,
        ..plain
    };
    let base = attend(&q, rows(&ks), AttendValues::Rows(rows(&vs)), &list, &plain).unwrap();
    let out = attend(
        &q,
        rows(&ks),
        AttendValues::Rows(rows(&vs)),
        &list,
        &bounded,
    )
    .unwrap();
    assert_eq!(out.values, base.values);
    for head in 0..heads {
        let q = &q[head * dim..(head + 1) * dim];
        let dot = |row: usize| (0..dim).map(|i| q[i] * k[row * dim + i]).sum::<f64>();
        let magnitude = |row: usize| {
            (0..dim)
                .map(|i| (q[i] * k[row * dim + i]).abs())
                .sum::<f64>()
        };
        let logits = [2, 0, 1].map(|row| dot(row) * 0.5);
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total = logits.iter().map(|l| (l - max).exp()).sum::<f64>();
        for d in 0..dim {
            let at = head * dim + d;
            let o = out.values[at];
            let spread = [2, 0, 1]
                .iter()
                .zip(&logits)
                .map(|(&row, l)| {
                    (l - max).exp() / total * (v[row * dim + d] - o).abs() * magnitude(row) * 0.5
                })
                .sum::<f64>();
            let expected = base.sum_abs_products[at] + spread * dim as f64 / 3.0;
            assert!((out.sum_abs_products[at] - expected).abs() <= 1e-13 * expected);
        }
    }
}
