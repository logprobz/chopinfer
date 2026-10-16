use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::{
    ActivationFormat, DType, Epilogues, GemmQuery, GroupedGemm, MoeQuery, NumericalContract,
    RoutedCapacity, RouterKind, RowQuery, ScaleMode, SwigluColumns, Validate, WeightFormat,
};
use kernel_core::moe_oracle::{
    ExpertMlp, ExpertMlpRounding, ExpertWeights, RouteWeightAt, expert_mlp,
    fp8_block_checkpoint_weight, local_expert_routes, shared_expert_gate_combine,
};
use kernel_core::oracle::{RouterSelection, SiluGlu, router};
use kernel_core::quantize::Quantizer;
use kernel_core::recurrent_oracle::{sigmoid, silu};

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

const SOFTMAX_RENORMALIZED: RouterKind = RouterKind::SoftmaxTopK { renormalize: true };

#[test]
fn qwen38_moe_contracts_validate() {
    let route = RowQuery::Router {
        experts: 512,
        top_k: 10,
        kind: SOFTMAX_RENORMALIZED,
    };
    assert_eq!(
        route.numerical_contract_id().unwrap().as_str(),
        "row.router-softmax-topk.v1"
    );
    let local = MoeQuery::LocalRoutes {
        experts: 512,
        local_experts: 128,
        top_k: 10,
        capacity: 256,
    };
    assert_eq!(
        local.numerical_contract_id().unwrap().as_str(),
        "row.moe-local-expert-routes.v1"
    );
    let shared = MoeQuery::SharedExpertGateCombine {
        width: 2560,
        capacity: 256,
    };
    assert_eq!(
        shared.numerical_contract_id().unwrap().as_str(),
        "row.moe-shared-expert-sigmoid-combine.v1"
    );
    assert!(
        MoeQuery::LocalRoutes {
            experts: 512,
            local_experts: 96,
            top_k: 10,
            capacity: 256
        }
        .validate()
        .is_err()
    );
    // Routed experts on one rank: 128 local experts, every token replicated.
    let grouped = Some(GroupedGemm {
        experts: 128,
        top_k: 10,
        capacity: RoutedCapacity::TotalRows(2560),
    });
    let a8 = ActivationFormat::Fp8Group128(ScaleMode::ExactF32);
    let gate_up = GemmQuery {
        m_capacity: 2560,
        n: 1280,
        k: 2560,
        weight: WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
        activation: a8,
        output: DType::Bf16,
        epilogues: Epilogues {
            swiglu: Some(SwigluColumns::ConcatenatedHalves),
            ..Epilogues::default()
        },
        grouped,
    };
    let down = GemmQuery {
        n: 2560,
        k: 640,
        epilogues: Epilogues {
            weighted_moe_combine: true,
            ..Epilogues::default()
        },
        ..gate_up
    };
    assert_eq!(
        gate_up.numerical_contract_id().unwrap().as_str(),
        "gemm.w-fp8-block128-f32.a-fp8-group128-f32.acc-f32.v1"
    );
    down.validate().unwrap();
    let a16 = GemmQuery {
        activation: ActivationFormat::Bf16,
        ..down
    };
    assert_ne!(
        a16.numerical_contract_id().unwrap(),
        down.numerical_contract_id().unwrap()
    );
}

#[test]
fn softmax_top10_renormalizes_and_breaks_ties_to_lower_experts() {
    let mut scores = vec![0.0; 512];
    for (rank, expert) in [500, 7, 300, 42, 9, 100, 200, 400, 1, 2, 3]
        .iter()
        .enumerate()
    {
        scores[*expert] = 5.0 - rank as f64 * 0.25;
    }
    // Expert 3 ties expert 2 at the tenth place; the lower ID wins.
    scores[3] = scores[2];
    let selection = router(&scores, None, 10, SOFTMAX_RENORMALIZED).unwrap();
    assert_eq!(
        selection.expert_ids,
        [500, 7, 300, 42, 9, 100, 200, 400, 1, 2]
    );
    let exps = selection
        .expert_ids
        .iter()
        .map(|&e| scores[e].exp())
        .collect::<Vec<_>>();
    let total = exps.iter().sum::<f64>();
    for (weight, e) in selection.weights.iter().zip(&exps) {
        assert!((weight - e / total).abs() < 1e-15);
    }
    assert!((selection.weights.iter().sum::<f64>() - 1.0).abs() < 1e-15);
}

#[test]
fn checkpoint_fp8_blocks_multiply_the_bf16_scale() {
    // Header shapes: gate/up [640, 2560] with [5, 20] scales, down [2560, 640]
    // with [20, 5].
    assert!(fp8_block_checkpoint_weight(&vec![0; 640 * 2560], 640, 2560, &[0; 100]).is_ok());
    assert!(fp8_block_checkpoint_weight(&vec![0; 2560 * 640], 2560, 640, &[0; 100]).is_ok());
    assert!(fp8_block_checkpoint_weight(&vec![0; 2560 * 640], 2560, 640, &[0; 99]).is_err());
    let mut codes = vec![0x38_u8; 130 * 130];
    codes[1] = 0xc0; // -2.0
    let scales = [2.0_f32, 0.5, 4.0, 0.25].map(f32_to_bf16);
    let weight = fp8_block_checkpoint_weight(&codes, 130, 130, &scales).unwrap();
    assert_eq!(weight.decode(0, 0), 2.0);
    assert_eq!(weight.decode(0, 1), -4.0);
    assert_eq!(weight.decode(0, 128), 0.5);
    assert_eq!(weight.decode(129, 0), 4.0);
    assert_eq!(weight.decode(129, 129), 0.25);
}

fn selections(tokens: usize, experts: usize, top_k: usize, seed: u64) -> Vec<RouterSelection> {
    values(tokens * experts, seed)
        .chunks_exact(experts)
        .map(|scores| router(scores, None, top_k, SOFTMAX_RENORMALIZED).unwrap())
        .collect()
}

#[test]
fn expert_sharded_ranks_sum_to_the_full_layer() {
    let (tokens, experts, top_k, hidden, inter) = (5, 8, 3, 128, 128);
    let x = values(tokens * hidden, 1)
        .iter()
        .map(|v| v * 4.0)
        .collect::<Vec<_>>();
    let gate_up = values(experts * 2 * inter * hidden, 2)
        .iter()
        .map(|v| v * 0.1)
        .collect::<Vec<_>>();
    let down = values(experts * hidden * inter, 3)
        .iter()
        .map(|v| v * 0.1)
        .collect::<Vec<_>>();
    let chosen = selections(tokens, experts, top_k, 4);
    let flat_weights = chosen
        .iter()
        .flat_map(|s| s.weights.clone())
        .collect::<Vec<_>>();
    let fp8 = Some(Quantizer::fp8_blocks((1, 128), ScaleMode::ExactF32));
    for input in [None, fp8] {
        let mlp = ExpertMlp {
            activation: SiluGlu::UNCLAMPED,
            route_weight_at: RouteWeightAt::AfterDown,
            rounding: ExpertMlpRounding {
                input,
                down_input: input,
                ..ExpertMlpRounding::default()
            },
        };
        let run = |local_experts: usize, rank: usize| {
            let query = MoeQuery::LocalRoutes {
                experts: experts as u32,
                local_experts: local_experts as u32,
                top_k: top_k as u32,
                capacity: 16,
            };
            let routes = local_expert_routes(query, &chosen, rank).unwrap();
            let first = rank * local_experts;
            let slice =
                |w: &[f64], per: usize| w[first * per..(first + local_experts) * per].to_vec();
            let weights = routes
                .routes
                .iter()
                .map(|&r| flat_weights[r])
                .collect::<Vec<_>>();
            if routes.tokens.is_empty() {
                return vec![0.0; tokens * hidden];
            }
            expert_mlp(
                &x,
                tokens,
                &routes,
                &weights,
                ExpertWeights {
                    gate_up: &slice(&gate_up, 2 * inter * hidden),
                    down: &slice(&down, hidden * inter),
                    experts: local_experts,
                    hidden,
                    intermediate: inter,
                },
                mlp,
            )
            .unwrap()
            .combined
            .values
        };
        let full = run(experts, 0);
        let mut sum = vec![0.0; tokens * hidden];
        for rank in 0..4 {
            for (s, v) in sum.iter_mut().zip(run(2, rank)) {
                *s += v;
            }
        }
        assert!(max_difference(&sum, &full) < 1e-12);
    }
}

#[test]
fn routed_mlp_matches_a_hand_expanded_expert() {
    let (hidden, inter) = (128, 128);
    let x = values(2 * hidden, 5);
    let gate_up = values(2 * 2 * inter * hidden, 6);
    let down = values(2 * hidden * inter, 7);
    let query = MoeQuery::LocalRoutes {
        experts: 2,
        local_experts: 2,
        top_k: 1,
        capacity: 4,
    };
    let chosen = [
        RouterSelection {
            expert_ids: vec![1],
            weights: vec![0.75],
        },
        RouterSelection {
            expert_ids: vec![0],
            weights: vec![1.0],
        },
    ];
    let routes = local_expert_routes(query, &chosen, 0).unwrap();
    assert_eq!(routes.experts, [1, 0]);
    let out = expert_mlp(
        &x,
        2,
        &routes,
        &[0.75, 1.0],
        ExpertWeights {
            gate_up: &gate_up,
            down: &down,
            experts: 2,
            hidden,
            intermediate: inter,
        },
        ExpertMlp {
            activation: SiluGlu::UNCLAMPED,
            route_weight_at: RouteWeightAt::AfterDown,
            rounding: ExpertMlpRounding::default(),
        },
    )
    .unwrap();
    let expert = 1;
    let dot = |w: &[f64], v: &[f64]| w.iter().zip(v).map(|(a, b)| a * b).sum::<f64>();
    let h = (0..inter)
        .map(|i| {
            let row = |r: usize| &gate_up[(expert * 2 * inter + r) * hidden..][..hidden];
            silu(dot(row(i), &x[..hidden])) * dot(row(inter + i), &x[..hidden])
        })
        .collect::<Vec<_>>();
    let y = (0..hidden)
        .map(|o| 0.75 * dot(&down[(expert * hidden + o) * inter..][..inter], &h))
        .collect::<Vec<_>>();
    assert!(max_difference(&out.intermediate.values[..inter], &h) < 1e-12);
    assert!(max_difference(&out.combined.values[..hidden], &y) < 1e-12);
}

#[test]
fn shared_expert_gate_applies_to_each_rank_partial() {
    let query = MoeQuery::SharedExpertGateCombine {
        width: 2,
        capacity: 2,
    };
    let out = shared_expert_gate_combine(
        query,
        &[1.0, 2.0, 0.0, 0.0],
        &[4.0, -4.0, 2.0, 2.0],
        &[0.0, 1.0],
    )
    .unwrap();
    assert_eq!(
        out.values,
        [3.0, 0.0, 2.0 * sigmoid(1.0), 2.0 * sigmoid(1.0)]
    );
    let routed = values(4 * 4, 8);
    let shared = values(4 * 4, 9);
    let logits = [0.3, -1.2];
    let whole = |r: &[f64], s: &[f64]| {
        shared_expert_gate_combine(query, r, s, &logits)
            .unwrap()
            .values
    };
    let mut partials = vec![0.0; 4];
    for rank in 0..4 {
        let part = whole(
            &routed[rank * 4..(rank + 1) * 4],
            &shared[rank * 4..(rank + 1) * 4],
        );
        for (p, v) in partials.iter_mut().zip(part) {
            *p += v;
        }
    }
    let sum = |x: &[f64]| {
        (0..4)
            .map(|i| (0..4).map(|r| x[r * 4 + i]).sum())
            .collect::<Vec<f64>>()
    };
    assert!(max_difference(&partials, &whole(&sum(&routed), &sum(&shared))) < 1e-14);
}
