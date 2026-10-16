use kernel_core::contract::{NumericalContract, SigmoidMeanHyperConnectionQuery, Validate};
use kernel_core::hyper_connection_oracle::{
    HyperConnectionWeights, all_reduce_inject_norm, bottleneck, grouped_norm,
    hyper_connection_input, inject, mix,
};
use kernel_core::oracle::{GemmEpilogue, GemmShape, gemm};
use kernel_core::recurrent_oracle::sigmoid;

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

fn max_difference(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    x.iter()
        .zip(y)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

const EPS: u32 = 0x358637bd; // 1e-6 as F32

fn norm(streams: u32, width: u32) -> SigmoidMeanHyperConnectionQuery {
    SigmoidMeanHyperConnectionQuery::GroupedNorm {
        streams,
        width,
        eps_bits: EPS,
        capacity: 16,
    }
}

#[test]
fn qwen38_hyper_connection_contracts_validate_with_stable_ids() {
    assert_eq!(EPS, 1e-6_f32.to_bits());
    let queries = [
        norm(4, 2560),
        SigmoidMeanHyperConnectionQuery::Bottleneck {
            streams: 4,
            rank: 320,
            capacity: 16,
        },
        SigmoidMeanHyperConnectionQuery::Mix {
            streams: 4,
            width: 2560,
            capacity: 16,
        },
        SigmoidMeanHyperConnectionQuery::Inject {
            streams: 4,
            width: 2560,
            capacity: 16,
        },
        SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
            ranks: 4,
            streams: 4,
            width: 2560,
            eps_bits: EPS,
            capacity: 16,
        },
    ];
    let ids = queries.map(|q| q.numerical_contract_id().unwrap().as_str().to_owned());
    assert_eq!(
        ids,
        [
            "row.hc-grouped-gemma-rmsnorm.v1",
            "row.hc-bottleneck-silu.v1",
            "row.hc-mix-sigmoid-mean.v1",
            "row.hc-inject-residual.v1",
            "comm.all-reduce-hc-inject-grouped-rmsnorm.v1",
        ]
    );
    assert!(norm(1, 2560).validate().is_err());
    assert!(
        SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
            ranks: 0,
            streams: 4,
            width: 2560,
            eps_bits: EPS,
            capacity: 16,
        }
        .validate()
        .is_err()
    );
}

#[test]
fn wrapper_matches_hand_computation_with_zero_projections() {
    let x = [3.0, 4.0, 1.0, 1.0];
    let eps = f64::from(1e-6_f32);
    let first = (12.5 + eps).sqrt().recip();
    let second = (1.0 + eps).sqrt().recip();
    let normed = [3.0 * first, 4.0 * first, second, second];
    let input = hyper_connection_input(
        &x,
        HyperConnectionWeights {
            norm: &[0.0; 4],
            down: &[0.0; 4],
            up: &[0.0; 4],
            inject: Some(&[0.0; 8]),
            streams: 2,
            width: 2,
            rank: 1,
            eps,
        },
    )
    .unwrap();
    assert!(max_difference(&input.normed, &normed) < 1e-15);
    // Zero projections: every mix gate is sigmoid(0) = 1/2.
    let mixed = [
        0.25 * (normed[0] + normed[2]),
        0.25 * (normed[1] + normed[3]),
    ];
    assert!(max_difference(&input.mixed, &mixed) < 1e-15);
    assert_eq!(input.inject_logits, [0.0, 0.0]);
    // Inject weight 2 * sigmoid(0) = 1 adds y to both streams.
    let updated = inject(
        SigmoidMeanHyperConnectionQuery::Inject {
            streams: 2,
            width: 2,
            capacity: 1,
        },
        &x,
        &[0.5, -1.0],
        &input.inject_logits,
    )
    .unwrap();
    assert_eq!(updated.values, [3.5, 3.0, 1.5, 0.0]);
    let scaled = inject(
        SigmoidMeanHyperConnectionQuery::Inject {
            streams: 2,
            width: 2,
            capacity: 1,
        },
        &x,
        &[1.0, 1.0],
        &[2.0, -2.0],
    )
    .unwrap();
    let (a, b) = (2.0 * sigmoid(1.0), 2.0 * sigmoid(-1.0));
    assert_eq!(scaled.values, [3.0 + a, 4.0 + a, 1.0 + b, 1.0 + b]);
}

#[test]
fn wrapper_composes_from_the_row_ops_and_gemms() {
    let (tokens, streams, width, rank) = (3, 4, 8, 3);
    let wide = streams * width;
    let x = values(tokens * wide, 1);
    let norm_weight = values(wide, 2).iter().map(|v| v * 0.2).collect::<Vec<_>>();
    let down = values(rank * wide, 3);
    let up = values(wide * rank, 4);
    let inject_weight = values(streams * wide, 5);
    let composite = hyper_connection_input(
        &x,
        HyperConnectionWeights {
            norm: &norm_weight,
            down: &down,
            up: &up,
            inject: Some(&inject_weight),
            streams,
            width,
            rank,
            eps: f64::from(1e-6_f32),
        },
    )
    .unwrap();
    let plain = GemmEpilogue {
        residual: None,
        swiglu: None,
    };
    let shape = |n, k| GemmShape {
        m: tokens,
        n,
        k,
        experts: 1,
    };
    let normed = grouped_norm(norm(4, 8), &x, &norm_weight).unwrap().values;
    let d = gemm(&normed, &down, shape(rank, wide), &[], plain).unwrap();
    let hidden = bottleneck(
        SigmoidMeanHyperConnectionQuery::Bottleneck {
            streams: 4,
            rank: 3,
            capacity: 16,
        },
        &d.values,
    )
    .unwrap();
    let u = gemm(&hidden.values, &up, shape(wide, rank), &[], plain).unwrap();
    let mixed = mix(
        SigmoidMeanHyperConnectionQuery::Mix {
            streams: 4,
            width: 8,
            capacity: 16,
        },
        &u.values,
        &normed,
    )
    .unwrap();
    assert_eq!(composite.normed, normed);
    assert!(max_difference(&composite.mixed, &mixed.values) < 1e-15);
    let logits = gemm(&normed, &inject_weight, shape(streams, wide), &[], plain).unwrap();
    assert_eq!(composite.inject_logits, logits.values);
}

#[test]
fn fused_all_reduce_inject_norm_matches_the_unfused_steps() {
    let (tokens, streams, width) = (2, 4, 8);
    let wide = streams * width;
    let x = values(tokens * wide, 6);
    let logits = values(tokens * streams, 7);
    let partials = (0..4)
        .map(|r| values(tokens * width, 8 + r))
        .collect::<Vec<_>>();
    let next_norm = values(wide, 12);
    let fused = all_reduce_inject_norm(
        SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
            ranks: 4,
            streams: 4,
            width: 8,
            eps_bits: EPS,
            capacity: 16,
        },
        &partials,
        &x,
        &logits,
        &next_norm,
    )
    .unwrap();
    let y = (0..tokens * width)
        .map(|i| partials.iter().map(|p| p[i]).sum::<f64>())
        .collect::<Vec<_>>();
    let updated = inject(
        SigmoidMeanHyperConnectionQuery::Inject {
            streams: 4,
            width: 8,
            capacity: 16,
        },
        &x,
        &y,
        &logits,
    )
    .unwrap();
    assert!(max_difference(&fused.residual.values, &updated.values) < 1e-15);
    let normed = grouped_norm(norm(4, 8), &fused.residual.values, &next_norm).unwrap();
    assert_eq!(fused.normed.values, normed.values);
    assert!(
        all_reduce_inject_norm(
            SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
                ranks: 4,
                streams: 4,
                width: 8,
                eps_bits: EPS,
                capacity: 16,
            },
            &partials[..3],
            &x,
            &logits,
            &next_norm,
        )
        .is_err()
    );
}
