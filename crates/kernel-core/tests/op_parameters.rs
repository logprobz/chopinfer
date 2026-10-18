//! Explicit numerical parameters of the shared per-op references, including a
//! DeepSeek-style composition built from small synthetic tensors and checked
//! against straightforward f64 math.

use kernel_core::codec::{SmallFloat, bf16_to_f32, decode_ue8m0_f64, f32_to_bf16};
use kernel_core::contract::{
    DType, NumericalContract, RmsNormKind, RouterKind, ScaleMode, SigmoidMeanHyperConnectionQuery,
    Validate,
};
use kernel_core::hyper_connection_oracle::grouped_norm;
use kernel_core::latent_visibility::WindowBlockVisibility;
use kernel_core::moe_oracle::{
    ExpertMlp, ExpertMlpRounding, ExpertWeights, LocalRoutes, RouteWeightAt, expert_mlp,
};
use kernel_core::oracle::{
    GroupLimit, GroupSumOverflow, NormArithmetic, NormOffset, NormScale, RmsNorm, RouteKey,
    RouteTies, RouterScore, Routing, SiluGlu, rmsnorm, route, router,
};
use kernel_core::quantize::{
    AmaxRatio, QuantFormat, Quantized, Quantizer, ScaleFormat, ScaleProduct, ScaleRule, Scales,
    fp8_blocks, quantize,
};
use kernel_core::transform::ChunkedScaleGrid;

#[test]
fn position_mxfp4_indexer_selects_individual_compressed_positions() {
    use kernel_core::contract::Validate;
    use kernel_core::pooled_indexer::{
        IndexerGranularity, IndexerKeyFormat, IndexerSelectionMode, PooledIndexerQuery,
    };
    let query = PooledIndexerQuery::Select {
        heads: 32,
        head_dim: 128,
        group: 1,
        max_top_k: 2048,
        always_tail: false,
        granularity: IndexerGranularity::Position,
        key_format: IndexerKeyFormat::Fp4,
        mode: IndexerSelectionMode::Global,
        page_size: 64,
        capacity: 1,
    };
    query.validate().unwrap();
    assert_eq!(query.list_width(), Some(2048));
}

#[test]
fn f32_pool_without_position_keeps_f32_through_post_pool_rmsnorm() {
    use kernel_core::contract::{RecurrentPhase, Validate};
    use kernel_core::gated_pooling::GatedPoolingQuery;
    use kernel_core::gated_pooling_oracle::{norm_after_pool, pool_without_position};
    let query = GatedPoolingQuery {
        group: 2,
        width: 512,
        norm_eps_bits: 1e-20_f32.to_bits(),
        phase: RecurrentPhase::Verify { max_rows: 4 },
        capacity: 8,
    };
    query.validate().unwrap();
    assert_eq!(query.tail_state_bytes(), 2 * 512 * 4);
    assert_eq!(query.geometry().tail_slot(0), Some(0));
    assert_eq!(query.geometry().tail_slot(1), None);
    assert_eq!(query.geometry().tail_slot(2), Some(0));
    assert_eq!(query.geometry().tail_slot_offsets(0), Some((0, 512)));
    let keys = (0..1024)
        .map(|i| {
            if i < 512 {
                1.003 + (i % 7) as f32
            } else {
                -0.499
            }
        })
        .collect::<Vec<_>>();
    let gates = vec![0.0; 1024];
    let pooled = pool_without_position(query, &keys, &gates).unwrap();
    for (i, &value) in pooled.values.iter().enumerate() {
        assert_eq!(value, (f64::from(keys[i]) + f64::from(keys[i + 512])) / 2.0);
    }
    let pool_f32 = pooled.values.iter().map(|&v| v as f32).collect::<Vec<_>>();
    let weights = vec![1.5; 512];
    let norm = norm_after_pool(query, &pool_f32, &weights).unwrap();
    let rounded = pool_f32
        .iter()
        .map(|&v| bf16(f64::from(v)))
        .collect::<Vec<_>>();
    let unrounded: Vec<f64> = pool_f32.iter().copied().map(f64::from).collect();
    let inverse = (unrounded.iter().map(|v| v * v).sum::<f64>() / 512.0 + 1e-20_f32 as f64)
        .sqrt()
        .recip();
    for (&actual, &x) in norm.values.iter().zip(&unrounded) {
        assert!((actual - x * inverse * 1.5).abs() < 1e-14);
    }
    let prematurely_rounded = keys
        .iter()
        .map(|&v| bf16(f64::from(v)) as f32)
        .collect::<Vec<_>>();
    assert_ne!(
        pool_without_position(query, &prematurely_rounded, &gates)
            .unwrap()
            .values,
        pooled.values
    );
    let early_boundary = RmsNorm::f64(1e-20_f32 as f64, NormOffset::Plain)
        .apply(&rounded, &weights, 512)
        .unwrap();
    assert_ne!(early_boundary.values, norm.values);
}

#[test]
fn position_keys_use_rmsnorm_tail_rope_and_trained_mxfp4_quantization() {
    use kernel_core::pooled_indexer::{IndexerCandidates, PositionIndexerQuery};
    use kernel_core::position_indexer_oracle::{key_norm, quantizer, rotate_tail};
    let query = PositionIndexerQuery {
        heads: 32,
        head_dim: 128,
        compression_ratio: 2,
        tail_rope_dims: 64,
        key_norm_eps_bits: 1e-20_f32.to_bits(),
        max_top_k: 2048,
        page_size: 64,
        candidates: IndexerCandidates::All,
        capacity: 1,
    };
    // A constant nonzero key remains nonzero under RMSNorm, unlike LayerNorm.
    let normalized = key_norm(query, &[2.0; 128], &[1.5; 128]).unwrap();
    assert_eq!(normalized.values, vec![1.5; 128]);
    let rotated = rotate_tail(query, &normalized.values, &[[0.0, 1.0]; 32]).unwrap();
    assert_eq!(&rotated.values[..64], &[1.5; 64]);
    for pair in rotated.values[64..].chunks_exact(2) {
        assert_eq!(pair, &[-1.5, 1.5]);
    }
    let rounded = rotated
        .values
        .iter()
        .map(|&v| bf16(v) as f32)
        .collect::<Vec<_>>();
    let quantized = quantize(&rounded, 1, 128, quantizer()).unwrap();
    for (i, &value) in rounded.iter().enumerate() {
        assert_eq!(quantized.decode(0, i), f64::from(value));
    }
    // The amax sees BF16. This boundary takes scale 1 after BF16 rounding,
    // while an unrounded F32 amax takes scale 2.
    let raw = [6.015625_f32; 32];
    let trained = raw.map(|x| bf16(f64::from(x)) as f32);
    let rounded_quant = quantize(&trained, 1, 32, quantizer()).unwrap();
    assert_eq!(rounded_quant.scales, Scales::Ue8M0(vec![127]));
    assert_eq!(
        quantize(&raw, 1, 32, quantizer()).unwrap().scales,
        Scales::Ue8M0(vec![128])
    );
    let zeros = quantize(&[0.0; 32], 1, 32, quantizer()).unwrap();
    assert_eq!(zeros.scales, Scales::Ue8M0(vec![1]));
    assert!(zeros.codes.iter().all(|&code| code == 0));
}

fn values(len: usize, seed: u64, scale: f64) -> Vec<f64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0) * scale
        })
        .collect()
}

#[test]
fn row_bias_router_contract_matches_the_pinned_formula() {
    use kernel_core::router::RowBiasRouterQuery;
    use kernel_core::router_oracle::{row_bias_router, select_f32_scores, sqrt_softplus_scores};
    for (experts, top_k) in [(384, 6), (128, 3)] {
        let op = RowBiasRouterQuery {
            experts,
            top_k,
            scaling_factor: 1.5,
        };
        let row = (0..experts)
            .map(|i| (i % 17) as f32 / 8.0 - 1.0)
            .collect::<Vec<_>>();
        let first = (0..experts)
            .map(|i| if i < top_k { 8.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let second = (0..experts)
            .map(|i| if i >= experts - top_k { 8.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let logits = row.repeat(4);
        let select = [0, 1, 1, 0];
        let actual = row_bias_router(op, &logits, [&first, &second], &select).unwrap();
        let scores = row
            .iter()
            .map(|&v| (1.0 + f64::from(v).exp()).ln().sqrt())
            .collect::<Vec<_>>();
        for (r, result) in actual.iter().enumerate() {
            let bias = [&first, &second][select[r] as usize];
            let mut ids = (0..experts as usize).collect::<Vec<_>>();
            ids.sort_by(|&a, &b| {
                (scores[b] + f64::from(bias[b]))
                    .total_cmp(&(scores[a] + f64::from(bias[a])))
                    .then(a.cmp(&b))
            });
            ids.truncate(top_k as usize);
            assert_eq!(result.expert_ids, ids);
            let sum = ids.iter().map(|&i| scores[i]).sum::<f64>() + 1e-20;
            for (&i, &w) in ids.iter().zip(&result.weights) {
                assert!((w - scores[i] / sum * 1.5).abs() < 1e-14);
            }
        }
        assert_ne!(actual[0].expert_ids, actual[1].expert_ids);
        assert_eq!(actual[0], actual[3]);
        assert_eq!(actual[1], actual[2]);
        let own_scores = sqrt_softplus_scores(&logits)
            .unwrap()
            .into_iter()
            .map(|v| v as f32)
            .collect::<Vec<_>>();
        let exact = select_f32_scores(op, &own_scores, [&first, &second], &select).unwrap();
        assert!(
            exact
                .iter()
                .zip(actual)
                .all(|(a, b)| a.expert_ids == b.expert_ids)
        );
        assert_ne!(
            op.numerical_contract_id().unwrap(),
            op.query().numerical_contract_id().unwrap()
        );
    }
}

#[test]
fn row_bias_router_selection_uses_rounded_f32_keys_and_unbiased_weights() {
    use kernel_core::router::RowBiasRouterQuery;
    use kernel_core::router_oracle::select_f32_scores;
    let op = RowBiasRouterQuery {
        experts: 4,
        top_k: 2,
        scaling_factor: 1.5,
    };
    let scores = [1.0, 1.0 + f32::EPSILON, 0.5, 0.25];
    let bias = [33_554_432.0, 33_554_432.0, 0.0, 0.0];
    let exact = select_f32_scores(op, &scores, [&bias, &[0.0; 4]], &[0]).unwrap();
    assert_eq!(exact[0].expert_ids, [0, 1]);
    // F64 bias addition reverses these F32 ties.
    assert!(f64::from(scores[1]) + f64::from(bias[1]) > f64::from(scores[0]) + f64::from(bias[0]));
    let sum = scores[0] + scores[1] + 1e-20_f32;
    assert_eq!(
        exact[0].weights,
        [
            f64::from(scores[0] / sum * 1.5),
            f64::from(scores[1] / sum * 1.5)
        ]
    );
    let zero = select_f32_scores(op, &[0.0; 4], [&bias, &bias], &[0]).unwrap();
    assert_eq!(zero[0].weights, [0.0, 0.0]);
}

#[test]
fn row_bias_router_refuses_invalid_selectors_and_nonfinite_operands() {
    use kernel_core::router::RowBiasRouterQuery;
    use kernel_core::router_oracle::{row_bias_router, select_f32_scores, sqrt_softplus_scores};
    let op = RowBiasRouterQuery {
        experts: 4,
        top_k: 2,
        scaling_factor: 1.5,
    };
    for scaling_factor in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            RowBiasRouterQuery {
                scaling_factor,
                ..op
            }
            .validate()
            .is_err()
        );
    }
    assert!(RowBiasRouterQuery { top_k: 5, ..op }.validate().is_err());
    for selectors in [&[][..], &[2][..], &[0, 1][..]] {
        assert!(row_bias_router(op, &[1.0; 4], [&[0.0; 4]; 2], selectors).is_err());
    }
    assert!(row_bias_router(op, &[1.0; 4], [&[0.0; 4], &[f32::NAN; 4]], &[0]).is_err());
    assert!(row_bias_router(op, &[1.0; 4], [&[0.0; 3], &[0.0; 4]], &[0]).is_err());
    assert!(select_f32_scores(op, &[-1.0; 4], [&[0.0; 4]; 2], &[0]).is_err());
    assert!(select_f32_scores(op, &[f32::MAX; 4], [&[f32::MAX; 4]; 2], &[0]).is_err());
    assert!(select_f32_scores(op, &[f32::MAX; 4], [&[0.0; 4]; 2], &[0]).is_err());
    assert!(sqrt_softplus_scores(&[f32::NAN]).is_err());
    let edge = sqrt_softplus_scores(&[-1000.0, 0.0, 1000.0, f32::MAX]).unwrap();
    assert_eq!(edge[0], 0.0);
    assert_eq!(edge[1], 2_f64.ln().sqrt());
    assert!(edge.iter().all(|v| v.is_finite()));
}

fn quantized_chained_projection(
    gate_up: bool,
) -> kernel_core::moe_chained::ChainedMlpProjectionQuery {
    use kernel_core::contract::{
        ActivationFormat, Epilogues, GemmQuery, GroupedGemm, OutputQuantization, RoutedCapacity,
        SwigluColumns, WeightFormat,
    };
    use kernel_core::gated_mlp::{
        GatedMlpProjection, GatedMlpProjectionQuery, RouteWeightPoint, SwigluClamp,
    };
    use kernel_core::moe_chained::{ChainedK, ChainedMlpProjectionQuery};
    let k = ChainedK::new(if gate_up { 5120 } else { 576 }).unwrap();
    ChainedMlpProjectionQuery {
        projection: GatedMlpProjectionQuery {
            gemm: GemmQuery {
                m_capacity: 18,
                n: if gate_up { 1152 } else { 5120 },
                k: k.stored,
                weight: WeightFormat::Mxfp4,
                activation: ActivationFormat::Mxfp8,
                output: if gate_up { DType::Bf16 } else { DType::F32 },
                epilogues: Epilogues {
                    swiglu: gate_up.then_some(SwigluColumns::ConcatenatedHalves),
                    output_quantization: gate_up.then_some(OutputQuantization::Mxfp8),
                    ..Epilogues::default()
                },
                grouped: Some(GroupedGemm {
                    experts: 384,
                    top_k: 6,
                    capacity: RoutedCapacity::TotalRows(18),
                }),
            },
            projection: if gate_up {
                GatedMlpProjection::GateUp
            } else {
                GatedMlpProjection::Down
            },
            swiglu_clamp: gate_up.then_some(SwigluClamp::symmetric(10.0)),
            route_weight: Some(RouteWeightPoint::BeforeDown),
        },
        k,
    }
}

#[test]
fn quantized_chained_contract_pins_before_down_and_separate_stored_k() {
    use kernel_core::gated_mlp::RouteWeightPoint;
    use kernel_core::moe_chained::ChainedK;
    let gate = quantized_chained_projection(true);
    let down = quantized_chained_projection(false);
    gate.validate().unwrap();
    down.validate().unwrap();
    assert_eq!(
        down.k,
        ChainedK {
            stored: 576,
            loop_k: 640
        }
    );
    assert!(
        down.numerical_contract_id()
            .unwrap()
            .as_str()
            .contains("chain128.zero-tail.ceil-mxfp8-bf16-amax")
    );
    assert_ne!(
        gate.numerical_contract_id().unwrap(),
        down.numerical_contract_id().unwrap()
    );
    let mut bad = down;
    bad.projection.gemm.k = 640;
    assert!(bad.validate().is_err());
    bad = down;
    bad.projection.gemm.epilogues.weighted_moe_combine = true;
    assert!(bad.validate().is_err());
    bad = down;
    bad.projection.route_weight = Some(RouteWeightPoint::AfterDown);
    assert!(bad.validate().is_err());
    bad = gate;
    bad.projection.gemm.output = DType::F32;
    assert!(bad.validate().is_err());
    bad = gate;
    bad.projection.gemm.epilogues.output_quantization = None;
    assert!(bad.validate().is_err());
    for stored in [0, 32, 575, 577, u32::MAX, u32::MAX - 63] {
        assert!(ChainedK::new(stored).is_err());
    }
    assert!(
        ChainedK {
            stored: 576,
            loop_k: 768
        }
        .validate()
        .is_err()
    );
}

#[test]
fn chained_k_tail_has_64_real_products_then_exact_zeros() {
    use kernel_core::moe_chained::ChainedK;
    use kernel_core::moe_chained_oracle::{fold_f32_chains, partial_chains};
    let k = ChainedK::new(576).unwrap();
    let x = vec![1.0; 576];
    let mut weight = vec![1.0; 576];
    for (i, w) in weight[512..].iter_mut().enumerate() {
        *w = (i + 1) as f64;
    }
    let actual = partial_chains(&x, &weight, k).unwrap();
    assert_eq!(actual.values, [128.0, 128.0, 128.0, 128.0, 2080.0]);
    assert_eq!(actual.sum_abs_products, actual.values);
    assert_eq!(actual.reduction_lengths, [128; 5]);
    let mut poisoned = weight.clone();
    poisoned.extend([f64::NAN; 64]);
    assert!(partial_chains(&x, &poisoned, k).is_err());
    assert_eq!(partial_chains(&x, &poisoned[..576], k).unwrap(), actual);
    assert_eq!(
        fold_f32_chains(&[16_777_216.0, 1.0, -16_777_216.0, 1.0, 64.0]).unwrap(),
        65.0
    );
    assert!(fold_f32_chains(&[f32::MAX, f32::MAX]).is_err());
}

#[test]
fn before_down_epilogue_clamps_then_multiplies_in_vendor_order() {
    use kernel_core::gated_mlp::SwigluClamp;
    use kernel_core::moe_chained_oracle::{before_down_f64, weight_before_down};
    let clamp = Some(SwigluClamp::symmetric(10.0));
    for (g, u, w) in [
        (12.0_f64, -20.0_f64, 0.25_f64),
        (-12.0, 20.0, 0.75),
        (0.0, 2.0, 1.5),
    ] {
        let gate = g.min(10.0);
        let up = u.clamp(-10.0, 10.0);
        assert_eq!(
            before_down_f64(g, u, w, clamp).unwrap(),
            w * ((gate / (1.0 + (-gate).exp())) * up)
        );
    }
    // Reassociation loses a subnormal SiLU before the up branch rescales it.
    let (silu, up, weight) = (f32::from_bits(1 << 9), 2_f32.powi(100), 2_f32.powi(-10));
    assert_eq!(weight_before_down(silu, up, weight), 2_f32.powi(-50));
    assert_eq!((weight * silu) * up, 0.0);
    assert!(before_down_f64(f64::NAN, 1.0, 1.0, None).is_err());
}

#[test]
fn before_down_quantization_rounds_bf16_before_amax_and_uses_ceil_scales() {
    use kernel_core::moe_chained_oracle::quantize_before_down;
    let mut values = vec![0.0; 64];
    values[0] = 448.03125;
    values[1] = -0.0;
    values[32] = 500.0;
    let result = quantize_before_down(&values, 1, 64).unwrap();
    assert_eq!(result.scales, Scales::Ue8M0(vec![127, 128]));
    assert_eq!(result.codes[0], 0x7e);
    assert_eq!(result.codes[1], 0x80);
    let zero = quantize_before_down(&[0.0; 32], 1, 32).unwrap();
    assert_eq!(zero.scales, Scales::Ue8M0(vec![105]));
    let unrounded = quantize(
        &values,
        1,
        64,
        Quantizer {
            rule: ScaleRule::CeilPowerOfTwo(AmaxRatio::MultiplyReciprocal),
            amax_floor: Some(1e-4),
            ..Quantizer::mx(SmallFloat::E4M3)
        },
    )
    .unwrap();
    assert_ne!(result.scales, unrounded.scales);
    let ocp = quantize(&values, 1, 64, Quantizer::mx(SmallFloat::E4M3)).unwrap();
    assert_ne!(result.scales, ocp.scales);
    assert!(quantize_before_down(&[f32::NAN; 32], 1, 32).is_err());
    assert!(quantize_before_down(&[1.0; 31], 1, 31).is_err());
}

fn bf16(value: f64) -> f64 {
    f64::from(bf16_to_f32(f32_to_bf16(value as f32)))
}

fn bf16_exact(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|&v| f64::from(bf16_to_f32(f32_to_bf16(v as f32))))
        .collect()
}

#[test]
fn latent_sink_and_in_place_tail_contract_validate() {
    use kernel_core::contract::{AttentionPhase, Validate};
    use kernel_core::latent_attention::{
        LatentAttentionOutput, LatentAttentionQuery, LatentKvFormat,
    };
    let query = LatentAttentionQuery::Attention {
        heads: 16,
        latent: 512,
        max_keys: 128,
        softmax_scale_bits: (512_f32).powf(-0.5).to_bits(),
        page_size: 128,
        kv: LatentKvFormat::Bf16,
        phase: AttentionPhase::Decode,
        output: LatentAttentionOutput::Normalized,
        sink: true,
        tail_rope_dims: 64,
        second_fp4_keys: false,
        capacity: 1,
    };
    query.validate().unwrap();
}

/// DeepSeek's RMSNorm: F32 arithmetic, weight multiplied before the single
/// BF16 rounding, eps 1e-20.
const DEEPSEEK_NORM: RmsNorm = RmsNorm {
    eps: 1e-20,
    offset: NormOffset::Plain,
    arithmetic: NormArithmetic::F32,
    scale: NormScale::MultiplyReciprocal,
    before_weight: None,
    output: Some(DType::Bf16),
};

#[test]
fn weight_before_rounding_matches_f64_math_within_one_bf16_rounding() {
    let width = 64;
    let x = bf16_exact(&values(3 * width, 1, 3.0));
    let w = bf16_exact(&values(width, 2, 1.5));
    let out = DEEPSEEK_NORM.apply(&x, &w, width).unwrap();
    for (row, x) in x.chunks_exact(width).enumerate() {
        let mean = x.iter().map(|v| v * v).sum::<f64>() / width as f64;
        let inv = 1.0 / (mean + 1e-20).sqrt();
        for (col, (&x, &w)) in x.iter().zip(&w).enumerate() {
            let exact = x * inv * w;
            let actual = out.values[row * width + col];
            assert_eq!(actual, f64::from(bf16_to_f32(f32_to_bf16(actual as f32))));
            // Half a BF16 ulp is at most 2^-8 relative; F32 adds a few 2^-24.
            assert!((actual - exact).abs() <= exact.abs() * (2f64.powi(-8) + 2f64.powi(-20)));
        }
    }
    // Rounding before the weight is a different reference.
    let early = RmsNorm {
        before_weight: Some(DType::Bf16),
        ..DEEPSEEK_NORM
    };
    assert_ne!(early.apply(&x, &w, width).unwrap().values, out.values);
}

#[test]
fn plain_f32_output_keeps_weight_products_without_bf16_rounding() {
    use kernel_core::rmsnorm::RmsNormRequirement;
    let width = 1280;
    let x = values(3 * width, 1, 3.0 / 256.0)
        .into_iter()
        .map(|v| f64::from(v as f32))
        .collect::<Vec<_>>();
    let w = bf16_exact(&values(width, 2, 1.5));
    let requirement = RmsNormRequirement {
        input: DType::F32,
        output: DType::F32,
        ..RmsNormRequirement::bf16_backend_chosen(width as u32, 1e-20, RmsNormKind::Plain, false)
    };
    let reference = requirement.oracle(&x, &w, 3).unwrap();
    let arithmetic = RmsNorm {
        eps: 1e-20,
        offset: NormOffset::Plain,
        arithmetic: NormArithmetic::F32,
        scale: NormScale::MultiplyReciprocal,
        before_weight: None,
        output: None,
    };
    let actual = arithmetic.apply(&x, &w, width).unwrap();
    kernel_core::oracle::check_numeric(&actual.values, &reference, DType::F32).unwrap();
    let narrowed = arithmetic.apply(&bf16_exact(&x), &w, width).unwrap();
    assert!(kernel_core::oracle::check_numeric(&narrowed.values, &reference, DType::F32).is_err());
    for mutant in [
        RmsNorm {
            output: Some(DType::Bf16),
            ..arithmetic
        },
        RmsNorm {
            before_weight: Some(DType::Bf16),
            ..arithmetic
        },
    ] {
        let wrong = mutant.apply(&x, &w, width).unwrap();
        assert!(kernel_core::oracle::check_numeric(&wrong.values, &reference, DType::F32).is_err());
    }
}

#[test]
fn offset_and_groups_are_explicit() {
    let (width, head) = (32, 8);
    let x = values(2 * width, 3, 2.0);
    let w = values(width, 4, 0.5);
    for (offset, kind) in [
        (NormOffset::Plain, RmsNormKind::Plain),
        (NormOffset::OnePlusWeight, RmsNormKind::GemmaOnePlusWeight),
    ] {
        let norm = RmsNorm::f64(1e-6, offset).apply(&x, &w, width).unwrap();
        let contract = rmsnorm(&x, width, &w, 1e-6, kind, None).unwrap();
        assert_eq!(norm, contract);
    }
    // Groups normalize independently while the weight follows the row.
    let grouped = RmsNorm::f64(1e-6, NormOffset::Plain)
        .apply(&x, &w, head)
        .unwrap();
    for (g, chunk) in x.chunks_exact(head).enumerate() {
        let first = (g * head) % width;
        let alone = RmsNorm::f64(1e-6, NormOffset::Plain)
            .apply(chunk, &w[first..first + head], head)
            .unwrap();
        assert_eq!(&grouped.values[g * head..(g + 1) * head], &alone.values[..]);
    }
    // The F32 offset is added in F32: 1 + 2^-30 is 1.
    let tiny = vec![2f64.powi(-30); 4];
    let ones = vec![0.0; 4];
    let f32_norm = |offset| RmsNorm {
        eps: 1e-6,
        offset,
        arithmetic: NormArithmetic::F32,
        scale: NormScale::MultiplyReciprocal,
        before_weight: None,
        output: None,
    };
    let x = [1.0, -2.0, 0.5, 3.0];
    assert_eq!(
        f32_norm(NormOffset::OnePlusWeight)
            .apply(&x, &tiny, 4)
            .unwrap()
            .values,
        f32_norm(NormOffset::OnePlusWeight)
            .apply(&x, &ones, 4)
            .unwrap()
            .values
    );
}

#[test]
fn norm_adapters_keep_their_signed_zero_and_refusal_bits() {
    // The grouped norm adds +0.0 to its input first, as the row adapter does
    // without a residual, so -0.0 normalizes to +0.0.
    let query = SigmoidMeanHyperConnectionQuery::GroupedNorm {
        streams: 2,
        width: 1,
        eps_bits: 1e-6_f32.to_bits(),
        capacity: 1,
    };
    let grouped = grouped_norm(query, &[-0.0, -0.0], &[0.0, 0.0]).unwrap();
    let row = rmsnorm(
        &[-0.0],
        1,
        &[0.0],
        f64::from(1e-6_f32),
        RmsNormKind::GemmaOnePlusWeight,
        None,
    )
    .unwrap();
    assert_eq!(row.values[0].to_bits(), 0);
    assert!(grouped.values.iter().all(|v| v.to_bits() == 0));
    // The primitive itself keeps the sign.
    let plain = RmsNorm::f64(1e-6, NormOffset::OnePlusWeight)
        .apply(&[-0.0], &[0.0], 1)
        .unwrap();
    assert_eq!(plain.values[0].to_bits(), (-0.0_f64).to_bits());
    // An eps that vanishes in F32 keeps the HF kind's refusal.
    let refusal = rmsnorm(
        &[1.0],
        1,
        &[1.0],
        1e-50,
        RmsNormKind::HfBf16Intermediate,
        None,
    );
    assert_eq!(
        refusal.unwrap_err().0,
        "HF RMSNorm shape or epsilon mismatch"
    );
}

#[test]
fn unimplemented_norm_combinations_are_refused() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let w = [1.0; 4];
    let f64_rounded = RmsNorm {
        output: Some(DType::Bf16),
        ..RmsNorm::f64(1e-6, NormOffset::Plain)
    };
    assert!(f64_rounded.apply(&x, &w, 4).is_err());
    let f16 = RmsNorm {
        before_weight: Some(DType::F16),
        ..DEEPSEEK_NORM
    };
    assert!(f16.apply(&x, &w, 4).is_err());
    // An eps that vanishes in F32 is refused rather than dropped.
    let vanishing = RmsNorm {
        eps: 1e-50,
        ..DEEPSEEK_NORM
    };
    assert!(vanishing.apply(&x, &w, 4).is_err());
    assert!(DEEPSEEK_NORM.apply(&x, &w, 3).is_err());
    assert!(DEEPSEEK_NORM.apply(&x[..3], &w, 4).is_err());
}

/// DeepSeek's activation quantization: E4M3 per 1 x 32 with a UE8M0 scale
/// `2^ceil(log2(max(amax, 1e-4) * (1 / 448)))`.
const DEEPSEEK_FP8: Quantizer = Quantizer {
    format: QuantFormat {
        element: SmallFloat::E4M3,
        block: (1, 32),
        scale: ScaleFormat::Ue8M0,
        chunk_rows: None,
    },
    rule: ScaleRule::CeilPowerOfTwo(AmaxRatio::MultiplyReciprocal),
    amax_floor: Some(1e-4),
};

#[test]
fn ceil_power_of_two_ue8m0_activations_match_f64_math() {
    let (rows, k) = (3, 96);
    let mut x = values(rows * k, 5, 20.0)
        .iter()
        .map(|&v| v as f32)
        .collect::<Vec<_>>();
    // One all-zero block, raised to the amax floor.
    x[k..k + 32].fill(0.0);
    let q = quantize(&x, rows, k, DEEPSEEK_FP8).unwrap();
    let Scales::Ue8M0(scales) = &q.scales else {
        panic!("UE8M0 storage expected");
    };
    for (block, chunk) in x.chunks_exact(32).enumerate() {
        let amax = chunk
            .iter()
            .fold(0.0_f64, |m, &v| m.max(f64::from(v).abs()));
        let scale = 2f64.powf((amax.max(1e-4) / 448.0).log2().ceil());
        assert_eq!(decode_ue8m0_f64(scales[block]), scale);
        for (i, &v) in chunk.iter().enumerate() {
            let (row, col) = ((block * 32 + i) / k, (block * 32 + i) % k);
            let ratio = f64::from(v) / scale;
            assert!(ratio.abs() <= 448.0);
            let rounded = SmallFloat::E4M3.decode_f64(SmallFloat::E4M3.encode_f32(ratio as f32));
            assert_eq!(q.decode(row, col), rounded * scale);
        }
    }
    assert_eq!(
        scales[k / 32],
        (2f64.powf((1e-4_f64 / 448.0).log2().ceil()).log2() + 127.0) as u8
    );
    // The OCP floor rule under the same storage is a different reference.
    let ocp = Quantizer {
        rule: ScaleRule::OcpFloor,
        amax_floor: None,
        ..DEEPSEEK_FP8
    };
    assert_ne!(quantize(&x, rows, k, ocp).unwrap().scales, q.scales);
}

#[test]
fn amax_ratio_and_power_of_two_rules_are_explicit() {
    let mut x = vec![0.0_f32; 32];
    x[3] = 0.37;
    let exact = |ratio| Quantizer {
        rule: ScaleRule::Ratio(ratio),
        ..Quantizer::fp8_blocks((1, 32), ScaleMode::ExactF32)
    };
    let scale = |q: Quantized| match q.scales {
        Scales::F32(s) => s[0],
        _ => panic!("F32 storage expected"),
    };
    let divide = scale(quantize(&x, 1, 32, exact(AmaxRatio::Divide)).unwrap());
    let reciprocal = scale(quantize(&x, 1, 32, exact(AmaxRatio::MultiplyReciprocal)).unwrap());
    assert_eq!(divide, 0.37_f32 / 448.0);
    assert_eq!(reciprocal, 0.37_f32 * (1.0 / 448.0));
    assert_ne!(divide, reciprocal);
    // The helper and its explicit quantizer agree.
    assert_eq!(
        fp8_blocks(&x, 1, 32, 1, 32, ScaleMode::ExactF32).unwrap(),
        quantize(&x, 1, 32, exact(AmaxRatio::Divide)).unwrap()
    );
    // Exponent-bit ceiling: never below the ratio, never twice above it.
    let ceil = Quantizer {
        rule: ScaleRule::CeilPowerOfTwo(AmaxRatio::Divide),
        ..exact(AmaxRatio::Divide)
    };
    for (i, v) in values(200, 6, 1000.0).iter().enumerate() {
        x[i % 32] = *v as f32;
        let ratio = f64::from(x.iter().fold(0.0_f32, |m, v| m.max(v.abs())) / 448.0);
        let s = f64::from(scale(quantize(&x, 1, 32, ceil).unwrap()));
        assert!(s >= ratio && s / 2.0 < ratio && s.log2().fract() == 0.0);
    }
    x.fill(0.0);
    x[0] = 448.0 * 0.25;
    assert_eq!(scale(quantize(&x, 1, 32, ceil).unwrap()), 0.25);
}

#[test]
fn chunked_scale_grids_restart_at_each_chunk() {
    let format = QuantFormat {
        element: SmallFloat::E4M3,
        block: (2, 4),
        scale: ScaleFormat::F32,
        chunk_rows: Some(3),
    };
    let grid = ChunkedScaleGrid {
        chunks: 2,
        rows_per_chunk: 3,
        k_blocks: 2,
        block_rows: 2,
    };
    assert_eq!(format.scale_rows(6), grid.total_scale_rows());
    for row in 0..6 {
        assert_eq!(format.scale_row(row), grid.scale_row(row).unwrap());
    }
    let scales = (0..8).map(|i| 2f32.powi(i)).collect::<Vec<_>>();
    let codes = vec![SmallFloat::E4M3.encode_f32(1.0); 48];
    let q = Quantized::from_parts(6, 8, format, codes.clone(), Scales::F32(scales), 1.0).unwrap();
    // Row 2 is the partial block of chunk 0; row 3 starts chunk 1.
    assert_eq!((q.decode(2, 5), q.decode(3, 0)), (8.0, 16.0));
    let x = (0..48).map(|i| i as f32 - 20.0).collect::<Vec<_>>();
    let quantizer = Quantizer {
        format,
        rule: ScaleRule::Ratio(AmaxRatio::Divide),
        amax_floor: None,
    };
    let chunked = quantize(&x, 6, 8, quantizer).unwrap();
    let Scales::F32(scales) = &chunked.scales else {
        panic!("F32 storage expected");
    };
    // Chunk 0's partial block holds row 2 alone: values -4..-1 in block 0.
    assert_eq!(scales[2], 4.0 / 448.0);
    assert!(Quantized::from_parts(6, 8, format, codes, Scales::F32(vec![1.0; 6]), 1.0).is_err());
}

#[test]
fn unimplemented_quantizer_combinations_are_refused() {
    let x = vec![1.0_f32; 32];
    let with = |scale, rule| Quantizer {
        format: QuantFormat {
            scale,
            ..DEEPSEEK_FP8.format
        },
        rule,
        amax_floor: None,
    };
    let divide = AmaxRatio::Divide;
    for refused in [
        with(ScaleFormat::Bf16, ScaleRule::Ratio(divide)),
        with(ScaleFormat::F32, ScaleRule::OcpFloor),
        with(ScaleFormat::Ue8M0, ScaleRule::Ratio(divide)),
        with(ScaleFormat::Ue8M0, ScaleRule::PowfCeilLog2(divide)),
        with(
            ScaleFormat::Ue8M0,
            ScaleRule::LocalOverGlobal { global: 1.0 },
        ),
        with(ScaleFormat::E4M3, ScaleRule::CeilPowerOfTwo(divide)),
        with(
            ScaleFormat::E4M3,
            ScaleRule::LocalOverGlobal { global: 0.0 },
        ),
        Quantizer {
            amax_floor: Some(0.0),
            ..DEEPSEEK_FP8
        },
        Quantizer {
            format: QuantFormat {
                element: SmallFloat::Ue4M3,
                ..DEEPSEEK_FP8.format
            },
            ..DEEPSEEK_FP8
        },
    ] {
        assert!(quantize(&x, 1, 32, refused).is_err(), "{refused:?}");
    }
    // BF16 scale storage decodes; its values must be BF16.
    let bf16 = QuantFormat {
        scale: ScaleFormat::Bf16,
        ..DEEPSEEK_FP8.format
    };
    let codes = vec![0x38; 32];
    assert!(Quantized::from_parts(1, 32, bf16, codes.clone(), Scales::F32(vec![0.5]), 1.0).is_ok());
    assert!(Quantized::from_parts(1, 32, bf16, codes, Scales::F32(vec![0.1]), 1.0).is_err());
}

/// DeepSeek's router: sqrt(softplus(logit / temperature)), selection on
/// score plus the row's bias, weights normalized with eps 1e-20 and scaled.
const DEEPSEEK_ROUTING: Routing = Routing {
    score: RouterScore::SqrtSoftplus,
    temperature: 1.25,
    key: RouteKey::Scores,
    group_limit: None,
    normalize: Some(1e-20),
    scale: 1.5,
    ties: RouteTies::LowerIndex,
};

#[test]
fn sqrt_softplus_routing_matches_f64_math() {
    let experts = 16;
    let logits = values(experts, 7, 6.0);
    let text_bias = values(experts, 8, 0.2);
    let image_bias = values(experts, 9, 0.2);
    for bias in [&text_bias, &image_bias] {
        let selection = route(&logits, Some(bias), 4, DEEPSEEK_ROUTING).unwrap();
        let scores = logits
            .iter()
            .map(|&l| (1.0 + (l / 1.25).exp()).ln().sqrt())
            .collect::<Vec<_>>();
        let mut order = (0..experts).collect::<Vec<_>>();
        order.sort_by(|&a, &b| (scores[b] + bias[b]).total_cmp(&(scores[a] + bias[a])));
        order.truncate(4);
        assert_eq!(selection.expert_ids, order);
        let sum = order.iter().map(|&i| scores[i]).sum::<f64>() + 1e-20;
        for (&id, &weight) in order.iter().zip(&selection.weights) {
            let expected = scores[id] / sum * 1.5;
            assert!((weight - expected).abs() <= 1e-14 * expected);
        }
    }
    // The bias picks experts but never weights them.
    let flat = vec![0.0; experts];
    let unbiased = route(&logits, Some(&flat), 4, DEEPSEEK_ROUTING).unwrap();
    let mut boosted = flat.clone();
    for &id in &unbiased.expert_ids {
        boosted[id] = 0.5;
    }
    assert_eq!(
        route(&logits, Some(&boosted), 4, DEEPSEEK_ROUTING).unwrap(),
        unbiased
    );
    // Equal keys keep the lower index.
    let tied = route(&[1.0, 2.0, 2.0, 0.5], None, 1, DEEPSEEK_ROUTING).unwrap();
    assert_eq!(tied.expert_ids, vec![1]);
    // Temperature divides the logits before the score.
    let cooled = Routing {
        temperature: 1.0,
        ..DEEPSEEK_ROUTING
    };
    let tempered = logits.iter().map(|l| l / 1.25).collect::<Vec<_>>();
    assert_eq!(
        route(&logits, None, 4, DEEPSEEK_ROUTING).unwrap(),
        route(&tempered, None, 4, cooled).unwrap()
    );
}

#[test]
fn unimplemented_routing_combinations_are_refused() {
    let logits = values(8, 10, 1.0);
    for refused in [
        Routing {
            score: RouterScore::SelectedSoftmax,
            ..DEEPSEEK_ROUTING
        },
        Routing {
            temperature: 0.0,
            ..DEEPSEEK_ROUTING
        },
        Routing {
            normalize: Some(-1e-20),
            ..DEEPSEEK_ROUTING
        },
        Routing {
            scale: 0.0,
            ..DEEPSEEK_ROUTING
        },
        Routing {
            group_limit: Some(GroupLimit {
                groups: 3,
                keep: 1,
                per_group: 2,
                overflow: GroupSumOverflow::Refuse,
            }),
            ..DEEPSEEK_ROUTING
        },
        Routing {
            group_limit: Some(GroupLimit {
                groups: 4,
                keep: 1,
                per_group: 2,
                overflow: GroupSumOverflow::Refuse,
            }),
            ..DEEPSEEK_ROUTING
        },
    ] {
        assert!(route(&logits, None, 4, refused).is_err(), "{refused:?}");
    }
}

/// A small routed MoE: 2 experts, H = 64, I = 32, 3 tokens and 4 routes.
struct Moe {
    x: Vec<f64>,
    gate_up: Vec<f64>,
    down: Vec<f64>,
    routes: LocalRoutes,
    weights: Vec<f64>,
}

const H: usize = 64;
const I: usize = 32;

impl Moe {
    fn new() -> Self {
        Self {
            x: bf16_exact(&values(3 * H, 11, 2.0)),
            gate_up: bf16_exact(&values(2 * 2 * I * H, 12, 0.2)),
            down: bf16_exact(&values(2 * H * I, 13, 0.2)),
            routes: LocalRoutes {
                tokens: vec![0, 0, 1, 2],
                experts: vec![1, 0, 1, 0],
                routes: vec![0, 1, 2, 4],
            },
            weights: vec![0.75, 0.5, 1.25, 0.3],
        }
    }

    fn run(&self, mlp: ExpertMlp) -> kernel_core::moe_oracle::ExpertMlpOutput {
        let weights = ExpertWeights {
            gate_up: &self.gate_up,
            down: &self.down,
            experts: 2,
            hidden: H,
            intermediate: I,
        };
        expert_mlp(&self.x, 3, &self.routes, &self.weights, weights, mlp).unwrap()
    }

    /// Straightforward f64 composition; `round` is applied to each GEMM
    /// input row, `proj` to every projection output and `mid` to the
    /// weighted intermediate.
    fn expected(
        &self,
        limit: f64,
        round: &dyn Fn(&[f64]) -> Vec<f64>,
        proj: fn(f64) -> f64,
        mid: fn(f64) -> f64,
    ) -> (Vec<f64>, usize) {
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let mut out = vec![0.0; 3 * H];
        let mut clamped = 0;
        for (r, (&token, &expert)) in self
            .routes
            .tokens
            .iter()
            .zip(&self.routes.experts)
            .enumerate()
        {
            let x = round(&self.x[token * H..(token + 1) * H]);
            let gate_up = &self.gate_up[expert * 2 * I * H..(expert + 1) * 2 * I * H];
            let h = (0..I)
                .map(|j| {
                    let gate = proj(dot(&gate_up[j * H..(j + 1) * H], &x));
                    let up = proj(dot(&gate_up[(I + j) * H..(I + j + 1) * H], &x));
                    clamped += usize::from(gate > limit) + usize::from(up.abs() > limit);
                    let (gate, up) = (gate.min(limit), up.clamp(-limit, limit));
                    mid(self.weights[r] * (gate / (1.0 + (-gate).exp()) * up))
                })
                .collect::<Vec<_>>();
            let h = round(&h);
            let down = &self.down[expert * H * I..(expert + 1) * H * I];
            for col in 0..H {
                out[token * H + col] += proj(dot(&down[col * I..(col + 1) * I], &h));
            }
        }
        (out, clamped)
    }
}

/// DeepSeek's expert: up clamped to [-L, L], gate clamped above at L, and
/// the route weight applied to the intermediate before the down projection.
fn deepseek_mlp(limit: f64, rounding: ExpertMlpRounding) -> ExpertMlp {
    ExpertMlp {
        activation: SiluGlu {
            gate_limit: Some((f64::NEG_INFINITY, limit)),
            up_limit: Some((-limit, limit)),
            silu_output: None,
        },
        route_weight_at: RouteWeightAt::BeforeDown,
        rounding,
    }
}

#[test]
fn clamped_weight_before_down_experts_match_f64_math() {
    let moe = Moe::new();
    let limit = 0.6;
    let out = moe.run(deepseek_mlp(limit, ExpertMlpRounding::default()));
    let (expected, clamped) = moe.expected(limit, &|row| row.to_vec(), |v| v, |v| v);
    assert!(clamped > 0, "the clamps must be exercised");
    for (actual, expected) in out.combined.values.iter().zip(&expected) {
        assert!((actual - expected).abs() <= 1e-13 * (1.0 + expected.abs()));
    }
    // With FP8 inputs and BF16 projections and intermediate, each rounding
    // point sits where the reference applies it.
    let rounding = ExpertMlpRounding {
        input: Some(DEEPSEEK_FP8),
        gate_up_output: Some(DType::Bf16),
        intermediate: Some(DType::Bf16),
        down_input: Some(DEEPSEEK_FP8),
        down_output: Some(DType::Bf16),
        weighted_output: None,
    };
    let rounded = moe.run(deepseek_mlp(limit, rounding));
    let fp8 = |row: &[f64]| {
        let input = row.iter().map(|&v| v as f32).collect::<Vec<_>>();
        quantize(&input, 1, row.len(), DEEPSEEK_FP8)
            .unwrap()
            .decoded()
    };
    let (expected, _) = moe.expected(limit, &fp8, bf16, bf16);
    for (actual, expected) in rounded.combined.values.iter().zip(&expected) {
        assert!((actual - expected).abs() <= 1e-13 * (1.0 + expected.abs()));
    }
    assert_ne!(rounded.combined.values, out.combined.values);
}

#[test]
fn route_weight_placement_changes_routed_rows_not_math() {
    let moe = Moe::new();
    let plain = ExpertMlp {
        activation: SiluGlu::UNCLAMPED,
        route_weight_at: RouteWeightAt::AfterDown,
        rounding: ExpertMlpRounding::default(),
    };
    let routed = moe.run(plain);
    let before = moe.run(ExpertMlp {
        route_weight_at: RouteWeightAt::BeforeDown,
        ..plain
    });
    assert_eq!(before.intermediate, routed.intermediate);
    for (a, b) in before.combined.values.iter().zip(&routed.combined.values) {
        assert!((a - b).abs() <= 1e-13 * (1.0 + b.abs()));
    }
}

#[test]
fn unimplemented_expert_mlp_combinations_are_refused() {
    let moe = Moe::new();
    let weights = ExpertWeights {
        gate_up: &moe.gate_up,
        down: &moe.down,
        experts: 2,
        hidden: H,
        intermediate: I,
    };
    let two_row = Quantizer {
        format: QuantFormat {
            block: (2, 32),
            ..DEEPSEEK_FP8.format
        },
        ..DEEPSEEK_FP8
    };
    for refused in [
        deepseek_mlp(f64::NAN, ExpertMlpRounding::default()),
        ExpertMlp {
            activation: SiluGlu {
                gate_limit: Some((1.0, -1.0)),
                up_limit: None,
                silu_output: None,
            },
            ..deepseek_mlp(1.0, ExpertMlpRounding::default())
        },
        deepseek_mlp(
            1.0,
            ExpertMlpRounding {
                intermediate: Some(DType::F16),
                ..ExpertMlpRounding::default()
            },
        ),
        deepseek_mlp(
            1.0,
            ExpertMlpRounding {
                input: Some(two_row),
                ..ExpertMlpRounding::default()
            },
        ),
        deepseek_mlp(
            1.0,
            ExpertMlpRounding {
                gate_up_output: Some(DType::F16),
                ..ExpertMlpRounding::default()
            },
        ),
        deepseek_mlp(
            1.0,
            ExpertMlpRounding {
                down_output: Some(DType::F32),
                ..ExpertMlpRounding::default()
            },
        ),
        deepseek_mlp(
            1.0,
            ExpertMlpRounding {
                weighted_output: Some(DType::F16),
                ..ExpertMlpRounding::default()
            },
        ),
        ExpertMlp {
            activation: SiluGlu {
                silu_output: Some(DType::F16),
                ..SiluGlu::UNCLAMPED
            },
            ..deepseek_mlp(1.0, ExpertMlpRounding::default())
        },
    ] {
        let result = expert_mlp(&moe.x, 3, &moe.routes, &moe.weights, weights, refused);
        assert!(result.is_err(), "{refused:?}");
    }
}

#[test]
fn projection_outputs_round_where_declared() {
    // One expert with H = I = 1; every operand is exact BF16.
    let (x, gate_w) = (1.0078125, 1.078125);
    let routes = LocalRoutes {
        tokens: vec![0],
        experts: vec![0],
        routes: vec![0],
    };
    let run = |rounding, down: f64| {
        let weights = ExpertWeights {
            gate_up: &[gate_w, 1.0],
            down: &[down],
            experts: 1,
            hidden: 1,
            intermediate: 1,
        };
        let mlp = ExpertMlp {
            activation: SiluGlu::UNCLAMPED,
            route_weight_at: RouteWeightAt::BeforeDown,
            rounding,
        };
        expert_mlp(&[x], 1, &routes, &[1.0], weights, mlp)
            .unwrap()
            .combined
            .values[0]
    };
    let silu = |g: f64| g / (1.0 + (-g).exp());
    let intermediate = ExpertMlpRounding {
        intermediate: Some(DType::Bf16),
        ..ExpertMlpRounding::default()
    };
    // The gate projection 1.0865478515625 is not BF16; rounding it first
    // changes the result.
    assert_eq!(x * gate_w, 1.0865478515625);
    assert_eq!(run(intermediate, 1.0), bf16(silu(x * gate_w) * x));
    assert_eq!(run(intermediate, 1.0), 0.8203125);
    let projections = ExpertMlpRounding {
        gate_up_output: Some(DType::Bf16),
        ..intermediate
    };
    assert_eq!(run(projections, 1.0), bf16(silu(1.0859375) * x));
    assert_eq!(run(projections, 1.0), 0.81640625);
    // The down output rounds before the combine.
    let down = 1.0078125;
    assert_eq!(run(projections, down), 0.81640625 * down);
    let rounded_down = ExpertMlpRounding {
        down_output: Some(DType::Bf16),
        ..projections
    };
    assert_eq!(run(rounded_down, down), bf16(0.81640625 * down));
    assert_ne!(run(rounded_down, down), 0.81640625 * down);
}

#[test]
fn ceil_power_of_two_scales_stay_positive_below_the_normal_range() {
    let ceil = |scale| Quantizer {
        format: QuantFormat {
            element: SmallFloat::E4M3,
            block: (1, 2),
            scale,
            chunk_rows: None,
        },
        rule: ScaleRule::CeilPowerOfTwo(AmaxRatio::Divide),
        amax_floor: None,
    };
    // A positive amax whose ratio underflows to zero in F32.
    let tiny = quantize(&[f32::from_bits(1), 0.0], 1, 2, ceil(ScaleFormat::F32)).unwrap();
    assert_eq!(tiny.scales, Scales::F32(vec![f32::MIN_POSITIVE]));
    assert_eq!(tiny.decoded(), vec![0.0, 0.0]);
    let tiny = quantize(&[f32::from_bits(1), 0.0], 1, 2, ceil(ScaleFormat::Ue8M0)).unwrap();
    assert_eq!(tiny.scales, Scales::Ue8M0(vec![1]));
    // Around the subnormal and normal boundary of the ratio.
    let boundary = 448.0 * f32::MIN_POSITIVE;
    for amax in [
        boundary,
        f32::from_bits(boundary.to_bits() + 1),
        f32::from_bits(boundary.to_bits() - 1),
        boundary / 2.0,
        2.0 * boundary,
    ] {
        let ratio = amax / 448.0;
        let expected = if ratio < f32::MIN_POSITIVE {
            2f64.powi(-126)
        } else {
            2f64.powf(f64::from(ratio).log2().ceil())
        };
        let q = quantize(&[amax, -amax / 2.0], 1, 2, ceil(ScaleFormat::F32)).unwrap();
        assert_eq!(q.scales, Scales::F32(vec![expected as f32]), "{amax:e}");
        assert!(q.decoded().iter().all(|v| v.is_finite()));
        let q = quantize(&[amax, -amax / 2.0], 1, 2, ceil(ScaleFormat::Ue8M0)).unwrap();
        assert_eq!(
            q.scales,
            Scales::Ue8M0(vec![(expected.log2() + 127.0) as u8]),
            "{amax:e}"
        );
    }
}

#[test]
fn routing_outside_the_finite_range_is_refused() {
    let softmax = Routing {
        score: RouterScore::Softmax,
        temperature: 1.0,
        key: RouteKey::Logits,
        group_limit: None,
        normalize: Some(0.0),
        scale: 1.0,
        ties: RouteTies::LowerIndex,
    };
    // Tempered logits overflow.
    let hot = Routing {
        temperature: f64::MIN_POSITIVE,
        ..softmax
    };
    assert!(route(&[1e300, 2e300], None, 1, hot).is_err());
    // Every selected score underflows to zero and eps is zero.
    for score in [RouterScore::Sigmoid, RouterScore::SqrtSoftplus] {
        let zero = Routing {
            score,
            key: RouteKey::Scores,
            ..softmax
        };
        assert!(route(&[-1000.0, -2000.0], None, 2, zero).is_err());
        let floored = Routing {
            normalize: Some(1e-20),
            ..zero
        };
        let selection = route(&[-1000.0, -2000.0], None, 2, floored).unwrap();
        assert!(selection.weights.iter().all(|w| w.is_finite()));
    }
    // Large positive logits stay finite through the stable softplus.
    let sqrt = Routing {
        score: RouterScore::SqrtSoftplus,
        key: RouteKey::Scores,
        ..softmax
    };
    let selection = route(&[1000.0, 2000.0], None, 2, sqrt).unwrap();
    assert_eq!(selection.expert_ids, vec![1, 0]);
    assert!(selection.weights.iter().all(|w| w.is_finite()));
}

#[test]
fn group_limits_refuse_more_keys_than_a_group_holds() {
    let grouped = |per_group| Routing {
        score: RouterScore::SelectedSoftmax,
        temperature: 1.0,
        key: RouteKey::Logits,
        group_limit: Some(GroupLimit {
            groups: 2,
            keep: 1,
            per_group,
            overflow: GroupSumOverflow::Refuse,
        }),
        normalize: None,
        scale: 1.0,
        ties: RouteTies::LowerIndex,
    };
    let logits = [1.0, 2.0, 3.0, 4.0];
    assert!(route(&logits, None, 1, grouped(3)).is_err());
    assert_eq!(
        route(&logits, None, 1, grouped(2)).unwrap().expert_ids,
        vec![3]
    );
    // The legacy grouped kind scores single-expert groups by their one key.
    let kind = RouterKind::GroupLimitedSelectedSoftmax {
        n_group: 4,
        topk_group: 2,
    };
    assert_eq!(
        router(&logits, None, 2, kind).unwrap().expert_ids,
        vec![3, 2]
    );
}

#[test]
fn legacy_router_validates_an_unused_bias() {
    for kind in [
        RouterKind::SoftmaxTopK { renormalize: true },
        RouterKind::GroupLimitedSelectedSoftmax {
            n_group: 2,
            topk_group: 1,
        },
        RouterKind::SigmoidBiasSelectNormalized {
            scaling_factor: 1.0,
        },
    ] {
        let logits = [1.0, 2.0, 3.0, 4.0];
        assert!(router(&logits, Some(&[0.0; 4]), 1, kind).is_ok());
        assert!(router(&logits, Some(&[f64::NAN; 4]), 1, kind).is_err());
        assert!(router(&logits, Some(&[f64::INFINITY; 4]), 1, kind).is_err());
        assert!(router(&logits, Some(&[0.0; 3]), 1, kind).is_err());
    }
}

#[test]
fn overflowing_group_sums_are_refused_unless_legacy() {
    let grouped = |per_group, overflow| Routing {
        score: RouterScore::SelectedSoftmax,
        temperature: 1.0,
        key: RouteKey::Logits,
        group_limit: Some(GroupLimit {
            groups: 2,
            keep: 1,
            per_group,
            overflow,
        }),
        normalize: None,
        scale: 1.0,
        ties: RouteTies::LowerIndex,
    };
    // Both pair sums overflow although the second group is stronger;
    // scaled by an exact 1/4 they are representable and pick expert 2.
    let large = [1e308, 1e308, 1.5e308, 1.5e308];
    assert!(route(&large, None, 1, grouped(2, GroupSumOverflow::Refuse)).is_err());
    let scaled = large.map(|v| v / 4.0);
    let fits = route(&scaled, None, 1, grouped(2, GroupSumOverflow::Refuse)).unwrap();
    assert_eq!(fits.expert_ids, vec![2]);
    // The first sum overflows before cancelling to 0 while the second is 4.
    let cancelling = [1e308, 1e308, -1e308, -1e308, 1.0, 1.0, 1.0, 1.0];
    assert!(route(&cancelling, None, 1, grouped(4, GroupSumOverflow::Refuse)).is_err());
    // The legacy policy and the RouterKind adapter keep their old (wrong)
    // choice of expert 0.
    for (logits, per_group) in [(&large[..], 2), (&cancelling[..], 4)] {
        let legacy = route(
            logits,
            None,
            1,
            grouped(per_group, GroupSumOverflow::LegacyIeee),
        );
        assert_eq!(legacy.unwrap().expert_ids, vec![0]);
    }
    let kind = RouterKind::GroupLimitedSelectedSoftmax {
        n_group: 2,
        topk_group: 1,
    };
    assert_eq!(router(&large, None, 1, kind).unwrap().expert_ids, vec![0]);
}

#[test]
fn rms_norm_divides_or_multiplies_by_the_reciprocal() {
    let x = values(64, 31, 3.0);
    let w = values(16, 32, 0.5);
    for arithmetic in [NormArithmetic::F64, NormArithmetic::F32] {
        let norm = |scale| RmsNorm {
            eps: 1e-6,
            offset: NormOffset::OnePlusWeight,
            arithmetic,
            scale,
            before_weight: None,
            output: None,
        };
        let multiply = norm(NormScale::MultiplyReciprocal)
            .apply(&x, &w, 16)
            .unwrap();
        let divide = norm(NormScale::Divide).apply(&x, &w, 16).unwrap();
        assert_eq!(multiply.sum_abs_products, divide.sum_abs_products);
        assert_ne!(multiply.values, divide.values);
        for (row, x) in x.chunks_exact(16).enumerate() {
            let squares = x.iter().map(|v| v * v).sum::<f64>();
            let rms = (squares / 16.0 + 1e-6).sqrt();
            for (c, &x) in x.iter().enumerate() {
                let got = divide.values[row * 16 + c];
                if arithmetic == NormArithmetic::F64 {
                    assert_eq!(got, x / rms * (1.0 + w[c]));
                } else {
                    assert!((got - x / rms * (1.0 + w[c])).abs() <= 1e-5 * (1.0 + got.abs()));
                }
            }
        }
    }
}

#[test]
fn f32_scale_products_overflow_where_an_f32_kernel_does() {
    let mxfp8 = |scale: u8| {
        let format = QuantFormat {
            element: SmallFloat::E4M3,
            block: (1, 2),
            scale: ScaleFormat::Ue8M0,
            chunk_rows: None,
        };
        // E4M3 0x40 is 2.0 and 0xc0 is -2.0.
        Quantized::from_parts(
            1,
            2,
            format,
            vec![0x40, 0xc0],
            Scales::Ue8M0(vec![scale]),
            1.0,
        )
        .unwrap()
    };
    // 2 * 2^127 overflows F32 but not F64.
    let top = mxfp8(254);
    assert_eq!(top.decoded(), [2f64.powi(128), -2f64.powi(128)]);
    assert_eq!(
        top.decoded_with(ScaleProduct::F32),
        [f64::INFINITY, f64::NEG_INFINITY]
    );
    // One scale lower, both rules agree.
    let below = mxfp8(253);
    assert_eq!(below.decoded_with(ScaleProduct::F32), below.decoded());
    assert_eq!(below.decoded(), [2f64.powi(127), -2f64.powi(127)]);
    // 2 * 2^-127 rounds exactly into F32's subnormal range.
    let bottom = mxfp8(0);
    assert_eq!(bottom.decoded_with(ScaleProduct::F32), bottom.decoded());
    // NVFP4: the F32 product covers the local scale only; the global scale
    // then multiplies in F64.
    let format = QuantFormat {
        element: SmallFloat::E2M1,
        block: (1, 16),
        scale: ScaleFormat::E4M3,
        chunk_rows: None,
    };
    let nvfp4 = Quantized::from_parts(
        1,
        16,
        format,
        vec![0x77; 8],
        Scales::Ue4M3(vec![0x7e]),
        3e38,
    )
    .unwrap();
    let value = nvfp4.decode_with(0, 0, ScaleProduct::F32);
    assert_eq!(value, 6.0 * 448.0 * f64::from(3e38_f32));
    assert_eq!(value, nvfp4.decode(0, 0));
}

#[test]
fn silu_and_weighted_outputs_round_where_declared() {
    let bf16 = |v: f64| f64::from(bf16_to_f32(f32_to_bf16(v as f32)));
    // silu(1.5) is not a BF16 value; the rounded form multiplies its BF16.
    let (gate, up) = (1.5_f64, 1.0078125);
    let silu = gate / (1.0 + (-gate).exp());
    assert_ne!(silu, bf16(silu));
    let rounded = SiluGlu {
        silu_output: Some(DType::Bf16),
        ..SiluGlu::UNCLAMPED
    };
    assert_eq!(rounded.apply(&[gate], &[up]).unwrap(), [bf16(silu) * up]);
    assert_eq!(
        SiluGlu::UNCLAMPED.apply(&[gate], &[up]).unwrap(),
        [silu * up]
    );
    assert!(SiluGlu::UNCLAMPED.apply(&[gate], &[]).is_err());

    // One expert with H = I = 1: the weighted output rounds after the
    // route weight, and the combine sums the rounded rows.
    let routes = LocalRoutes {
        tokens: vec![0, 0],
        experts: vec![0, 0],
        routes: vec![0, 1],
    };
    let weights = ExpertWeights {
        gate_up: &[1.5, 1.0],
        down: &[1.0078125],
        experts: 1,
        hidden: 1,
        intermediate: 1,
    };
    let mlp = |rounding| ExpertMlp {
        activation: SiluGlu::UNCLAMPED,
        route_weight_at: RouteWeightAt::AfterDown,
        rounding,
    };
    let route_weights = [0.3, 0.7];
    let plain = expert_mlp(
        &[1.0],
        1,
        &routes,
        &route_weights,
        weights,
        mlp(ExpertMlpRounding::default()),
    )
    .unwrap();
    let rounding = ExpertMlpRounding {
        weighted_output: Some(DType::Bf16),
        ..ExpertMlpRounding::default()
    };
    let out = expert_mlp(&[1.0], 1, &routes, &route_weights, weights, mlp(rounding)).unwrap();
    let y = plain.routed.values[0];
    assert_eq!(out.routed, plain.routed);
    assert_eq!(plain.weighted.values, [y * 0.3, y * 0.7]);
    assert_eq!(out.weighted.values, [bf16(y * 0.3), bf16(y * 0.7)]);
    assert_eq!(out.combined.values, [bf16(y * 0.3) + bf16(y * 0.7)]);
    assert_eq!(plain.combined.values, [y * 0.3 + y * 0.7]);
}

#[test]
fn trained_ceil_quantization_rounds_to_bf16_before_selecting_scales() {
    use kernel_core::precision::quantizer::BlockQuantization;
    let recipe = BlockQuantization::E4M3_CEIL32;
    let input = [448.03125_f32; 32];
    let trained = recipe.apply(&input, 1, 32).unwrap();
    assert_eq!(trained.scales, Scales::Ue8M0(vec![127]));
    assert_eq!(trained.codes, vec![0x7e; 32]);
    assert_eq!(trained.decoded(), vec![448.0; 32]);
    let unrounded = quantize(&input, 1, 32, recipe.quantizer().unwrap()).unwrap();
    assert_eq!(unrounded.scales, Scales::Ue8M0(vec![128]));
}

#[test]
fn fp4_fake_quant_preserves_all_e2m1_values_and_signed_zeros() {
    use kernel_core::precision::quantizer::BlockQuantization;
    for format in [
        BlockQuantization::E2M1E4M3Scale16,
        BlockQuantization::E2M1Ceil32,
    ] {
        let width = format.group_size() as usize;
        let input: Vec<_> = (0..width)
            .map(|i| SmallFloat::E2M1.decode_f32((i % 16) as u8))
            .collect();
        let actual = format.fake_quant(&input, 1, width).unwrap();
        let expected: Vec<_> = input.iter().map(|&v| f32_to_bf16(v)).collect();
        assert_eq!(actual, expected);
    }
}

#[test]
fn clamped_silu_mul_rounds_only_its_product_before_ceil_quantization() {
    use kernel_core::gated_mlp::{SwigluClamp, quantized::SiluMulQuantizeQuery};
    use kernel_core::precision::quantizer::BlockQuantization;
    let (rows, width) = (32, 32);
    let query = SiluMulQuantizeQuery {
        width: width as u32,
        clamp: SwigluClamp::symmetric(10.0),
        amax_floor_bits: 1e-4_f32.to_bits(),
    };
    let mut gate: Vec<_> = values(rows * width, 45, 15.0)
        .into_iter()
        .map(|v| v as f32)
        .collect();
    let mut up: Vec<_> = values(rows * width, 61, 13.0)
        .into_iter()
        .map(|v| v as f32)
        .collect();
    gate[..4].copy_from_slice(&[-100.0, 12.0, -0.0, 3.0]);
    up[..4].copy_from_slice(&[4.0, 11.0, 1.0, -15.0]);
    let wide = query.oracle(&gate, &up, rows).unwrap();
    for (i, &value) in wide.iter().enumerate() {
        let g = f64::from(gate[i]).min(10.0);
        let u = f64::from(up[i]).clamp(-10.0, 10.0);
        assert_eq!(value.to_bits(), ((g / (1.0 + (-g).exp())) * u).to_bits());
    }
    let reference: Vec<_> = wide.iter().map(|&v| v as f32).collect();
    let expected = BlockQuantization::E4M3_CEIL32
        .apply(&reference, rows, width)
        .unwrap();
    let actual = query.apply(&gate, &up, rows).unwrap();
    assert_eq!(actual, expected);
    let early: Vec<_> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| {
            let g = g.min(10.0);
            let silu = bf16_to_f32(f32_to_bf16(g / (1.0 + (-g).exp())));
            silu * u.clamp(-10.0, 10.0)
        })
        .collect();
    assert_ne!(
        actual,
        BlockQuantization::E4M3_CEIL32
            .apply(&early, rows, width)
            .unwrap()
    );
    gate[0] = f32::NAN;
    assert_eq!(query.row_status(&gate, &up, &vec![0; rows]).unwrap()[0], 1);
    assert!(query.oracle(&gate, &up, rows).is_err());
}

#[test]
fn silu_mul_quantize_nonfinite_inputs_zero_the_entire_row() {
    use kernel_core::gated_mlp::{SwigluClamp, quantized::SiluMulQuantizeQuery};
    let query = SiluMulQuantizeQuery {
        width: 64,
        clamp: SwigluClamp::symmetric(10.0),
        amax_floor_bits: 1e-4_f32.to_bits(),
    };
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for in_gate in [true, false] {
            let mut gate = vec![2.0; 128];
            let mut up = vec![3.0; 128];
            if in_gate {
                gate[63] = bad;
            } else {
                up[63] = bad;
            }
            let result = query.apply(&gate, &up, 2).unwrap();
            assert_eq!(query.row_status(&gate, &up, &[2, 0]).unwrap(), [3, 0]);
            assert_eq!(result.codes[..64], [0; 64]);
            let Scales::Ue8M0(scales) = result.scales else {
                panic!("UE8M0 scales")
            };
            assert_eq!(scales[..2], [127; 2]);
            let passing = query.apply(&gate[64..], &up[64..], 1).unwrap();
            assert_eq!(result.codes[64..], passing.codes);
        }
    }
}

#[test]
fn silu_mul_quantize_retains_status_across_launches_and_reports_overflow() {
    use kernel_core::gated_mlp::{SwigluClamp, quantized::SiluMulQuantizeQuery};
    let query = SiluMulQuantizeQuery {
        width: 64,
        clamp: SwigluClamp::symmetric(10.0),
        amax_floor_bits: 1e-4_f32.to_bits(),
    };
    let mut gate = vec![2.0; 256];
    let up = vec![3.0; 256];
    gate[63] = f32::NEG_INFINITY;
    let (_, first) = query
        .apply_with_status(&gate, &up, &[0, 0, 2, 0x8000_0000])
        .unwrap();
    gate[63] = 2.0;
    gate[127] = f32::INFINITY;
    let (output, second) = query.apply_with_status(&gate, &up, &first).unwrap();
    assert_eq!(second, [1, 1, 2, 0x8000_0000]);
    assert_eq!(output.codes, vec![0; 256]);
    assert_eq!(output.scales, Scales::Ue8M0(vec![127; 8]));
    assert!(query.row_status(&gate, &up, &[0]).is_err());
    let overflow = SiluMulQuantizeQuery {
        clamp: SwigluClamp::symmetric(2e19),
        ..query
    };
    let (output, flags) = overflow
        .apply_with_status(&[2e19; 64], &[2e19; 64], &[2])
        .unwrap();
    assert_eq!(flags, [6]);
    assert_eq!(output.codes, vec![0; 64]);
}

mod latent_parameters {
    use super::*;
    use kernel_core::contract::{AttentionPhase, NumericalContract, Validate};
    use kernel_core::latent_attention::{
        LatentAttentionOutput as Output, LatentAttentionQuery as Query, LatentKvFormat as Format,
        LatentWindowStorage,
    };
    use kernel_core::latent_attention_oracle::{
        LatentAttentionResult as ResultRows, LatentCompressedInput, LatentKeyLists, LatentPaging,
        LatentSourcesInput, LatentWindowInput, decode_native_page, encode_latent_row,
        latent_attention_with_sources, latent_kv_write, latent_tail_rope, merge_partials_with_sink,
    };

    const D: usize = 512;
    const WINDOW: Format = Format::Fp8E4M3Ue8M0Block32;
    const COMPRESSED: Format = Format::Fp4E2M1E4M3Block16;

    fn query(output: Output) -> Query {
        Query::Attention {
            heads: 2,
            latent: D as u32,
            max_keys: 8,
            softmax_scale_bits: (D as f32).powf(-0.5).to_bits(),
            page_size: 2,
            kv: WINDOW,
            phase: AttentionPhase::Decode,
            output,
            sink: true,
            tail_rope_dims: 64,
            second_fp4_keys: true,
            capacity: 2,
        }
    }

    fn normalized(result: ResultRows) -> Vec<f64> {
        let ResultRows::Normalized(out) = result else {
            panic!("expected normalized output")
        };
        out.values
    }

    struct Sources {
        window: Vec<u8>,
        compressed: Vec<u8>,
        q: Vec<f64>,
        sinks: [f32; 2],
    }

    impl Sources {
        fn new() -> Self {
            // Ring slot 127 is 2, slot 0 is 1. Every 32-channel block has
            // its own power-of-two scale: 1, 2, 4, repeated.
            let mut window = vec![0; WINDOW.page_bytes(D as u32, 128) as usize];
            for slot in 0..128 {
                window[slot * D..(slot + 1) * D].fill(if slot % 2 == 0 { 0x38 } else { 0x40 });
                for block in 0..D / 32 {
                    window[128 * D + slot * (D / 32) + block] = 127 + (block % 3) as u8;
                }
            }
            // Two FP4 pages. The table reverses them. Every channel pair
            // is 1, 1.5 times a positive scale, 0.5 on page 0 and 1 on 1.
            let page = COMPRESSED.page_bytes(D as u32, 2) as usize;
            let mut compressed = vec![0; 2 * page];
            for p in 0..2 {
                compressed[p * page..p * page + D].fill(0x32);
                compressed[p * page + D..(p + 1) * page].fill(if p == 0 { 0x30 } else { 0x38 });
            }
            Self {
                window,
                compressed,
                q: vec![0.0; 2 * D],
                sinks: [0.0, 2.0],
            }
        }

        fn input<'a>(&'a self, window: &'a [i32], compressed: &'a [i32]) -> LatentSourcesInput<'a> {
            LatentSourcesInput {
                q: &self.q,
                row_batch: &[0],
                sinks: Some(&self.sinks),
                window: LatentWindowInput {
                    codes: &self.window[..128 * D],
                    scales: &self.window[128 * D..],
                    item_slots: &[0],
                    sequences: 1,
                    code_stride: self.window.len(),
                    scale_stride: self.window.len(),
                    storage: LatentWindowStorage::Ring128,
                    lists: LatentKeyLists {
                        indices: window,
                        stride: window.len(),
                        budget: window.len(),
                    },
                },
                compressed: Some(LatentCompressedInput {
                    pool: &self.compressed,
                    pages: 2,
                    paging: LatentPaging {
                        block_tables: &[1, 0],
                        max_pages: 2,
                        page_stride: COMPRESSED.page_bytes(D as u32, 2) as usize,
                    },
                    lists: LatentKeyLists {
                        indices: compressed,
                        stride: compressed.len(),
                        budget: compressed.len(),
                    },
                }),
            }
        }
    }

    #[test]
    fn latent_native_layout_and_quantizer_formulas() {
        assert_eq!(WINDOW.token_bytes(512), 528);
        assert_eq!(COMPRESSED.token_bytes(512), 288);
        for format in [WINDOW, COMPRESSED] {
            let zero = encode_latent_row(format, &vec![0; D]).unwrap();
            assert!(zero.values.iter().all(|&c| c == 0));
            assert!(zero.scales.is_empty());
            let expected = if format == WINDOW { 105 } else { 1 };
            assert!(zero.scale_bytes.iter().all(|&s| s == expected));
        }
        // Exact power of two and just above it exercise ceil, not RNE.
        for (amax, scale) in [(448_f32, 127), (450_f32, 128)] {
            let mut row = vec![f32_to_bf16(-1.0); D];
            row[0] = f32_to_bf16(amax);
            let encoded = encode_latent_row(WINDOW, &row).unwrap();
            assert_eq!(encoded.scale_bytes[0], scale);
            assert_eq!(encoded.values[0], if scale == 127 { 0x7e } else { 0x76 });
        }
        let mut row = vec![f32_to_bf16(6.0); D];
        row[0] = f32_to_bf16(1.25); // E2M1 tie rounds to even code 2.
        row[1] = f32_to_bf16(-1.75); // tie rounds to even code 4, negative.
        let encoded = encode_latent_row(COMPRESSED, &row).unwrap();
        assert_eq!(encoded.values[0], 0xc2);
        assert!(encoded.scale_bytes.iter().all(|&s| s == 0x38));
        // Exact decode for every E2M1 code and every positive E4M3 scale.
        for scale in 1..=126 {
            let mut page = vec![0; 72];
            for (i, code) in page[..64].iter_mut().enumerate() {
                *code = ((i % 16) * 17) as u8;
            }
            page[64..].fill(scale);
            let decoded = decode_native_page(COMPRESSED, 128, 1, &page, 0).unwrap();
            for (i, &x) in decoded.iter().enumerate() {
                assert_eq!(
                    x,
                    SmallFloat::E2M1.decode_f64(((i / 2) % 16) as u8)
                        * SmallFloat::E4M3.decode_f64(scale)
                );
                assert_eq!(x, bf16(x));
            }
        }
    }

    #[test]
    fn latent_window_planes_have_independent_strides_and_136_slots() {
        let source = Sources::new();
        let slots = 136;
        let code_stride = slots * D + 18;
        let scale_stride = slots * (D / 32) + 3;
        let mut codes = vec![0xa5; code_stride + slots * D];
        let scales = vec![128; scale_stride + slots * (D / 32)];
        for slot in 0..slots {
            codes[code_stride + slot * D..code_stride + (slot + 1) * D].fill(if slot == 135 {
                0x40
            } else {
                0x38
            });
        }
        let mut input = source.input(&[135, 0], &[]);
        input.row_batch = &[1];
        input.compressed = None;
        input.window = LatentWindowInput {
            codes: &codes,
            scales: &scales,
            item_slots: &[0, 1],
            sequences: 2,
            code_stride,
            scale_stride,
            storage: LatentWindowStorage::Staged {
                slots: slots as u32,
            },
            lists: input.window.lists,
        };
        for phase in [AttentionPhase::Decode, AttentionPhase::Verify] {
            let mut query = query(Output::Normalized);
            if let Query::Attention {
                second_fp4_keys,
                phase: p,
                ..
            } = &mut query
            {
                *second_fp4_keys = false;
                *p = phase;
            }
            let out = normalized(latent_attention_with_sources(query, input).unwrap());
            assert!(out[..D].iter().all(|&v| v == 2.0));
            let mut short = input;
            short.window.codes = &codes[..codes.len() - 1];
            assert!(latent_attention_with_sources(query, short).is_err());
            short = input;
            short.window.scales = &scales[..scales.len() - 1];
            assert!(latent_attention_with_sources(query, short).is_err());
            short = input;
            short.window.code_stride = slots * D - 1;
            assert!(latent_attention_with_sources(query, short).is_err());
            short = input;
            short.window.scale_stride = slots * (D / 32) - 1;
            assert!(latent_attention_with_sources(query, short).is_err());
        }
    }

    #[test]
    fn latent_two_sources_one_softmax_and_sink_once_across_splits() {
        let source = Sources::new();
        let whole = normalized(
            latent_attention_with_sources(
                query(Output::Normalized),
                source.input(&[127, -1, 0], &[0, -1, 3]),
            )
            .unwrap(),
        );
        for head in 0..2 {
            for col in 0..D {
                let ring_scale = 2_f64.powi(((col / 32) % 3) as i32);
                let compressed = if col % 2 == 0 { 1.0 } else { 1.5 };
                let expected = (3.0 * ring_scale + 1.5 * compressed)
                    / (4.0 + f64::from(source.sinks[head]).exp());
                assert!((whole[head * D + col] - expected).abs() < 1e-14);
            }
        }
        let mut parts = Vec::new();
        for (w, c) in [
            (&[127, -1][..], &[-1][..]),
            (&[-1, 0][..], &[0, 3][..]),
            (&[-1][..], &[-1][..]),
        ] {
            let ResultRows::Partial(partial) =
                latent_attention_with_sources(query(Output::Partial), source.input(w, c)).unwrap()
            else {
                panic!()
            };
            parts.push(partial);
        }
        let merged = merge_partials_with_sink(&parts, &source.sinks).unwrap();
        for (a, b) in merged.values.iter().zip(&whole) {
            assert!((a - b).abs() < 1e-14);
        }
        // Negative control: each split receives a sink, counting it three
        // times at the final merge. It must differ from the correct result.
        let mut wrong = parts.clone();
        for part in &mut wrong {
            for h in 0..2 {
                let old = part.maximum.values[h];
                let sink = f64::from(source.sinks[h]);
                let top = old.max(sink);
                let rescale = (old - top).exp();
                part.maximum.values[h] = top;
                part.denominator.values[h] =
                    part.denominator.values[h] * rescale + (sink - top).exp();
                for col in 0..D {
                    part.numerator.values[h * D + col] *= rescale;
                }
            }
        }
        let wrong = kernel_core::latent_attention_oracle::merge_partials(&wrong).unwrap();
        assert!((wrong.values[0] - whole[0]).abs() > 0.1);
        assert_eq!(parts[2].maximum.values, vec![f64::NEG_INFINITY; 2]);
        assert_eq!(parts[2].denominator.values, vec![0.0; 2]);
    }

    #[test]
    fn latent_scale_and_source_order_follow_the_vendor_formula() {
        let mut source = Sources::new();
        source.q[0] = 0.5;
        source.q[D - 1] = -0.25;
        source.q[D] = -0.5;
        source.q[2 * D - 1] = 0.25;
        let out = normalized(
            latent_attention_with_sources(
                query(Output::Normalized),
                source.input(&[127, 0], &[0, 3]),
            )
            .unwrap(),
        );
        let mut keys = vec![vec![0.0; D]; 4];
        for col in 0..D {
            let scale = 2_f64.powi(((col / 32) % 3) as i32);
            keys[0][col] = 2.0 * scale;
            keys[1][col] = scale;
            keys[2][col] = if col % 2 == 0 { 1.0 } else { 1.5 };
            keys[3][col] = keys[2][col] * 0.5;
        }
        for h in 0..2 {
            let scale = f64::from((D as f32).powf(-0.5));
            let weights = keys
                .iter()
                .map(|k| {
                    (scale * (source.q[h * D] * k[0] + source.q[(h + 1) * D - 1] * k[D - 1])).exp()
                })
                .collect::<Vec<_>>();
            let den = weights.iter().sum::<f64>() + f64::from(source.sinks[h]).exp();
            for col in 0..D {
                let expected = keys
                    .iter()
                    .zip(&weights)
                    .map(|(k, w)| k[col] * w)
                    .sum::<f64>()
                    / den;
                assert!((expected - out[h * D + col]).abs() < 1e-14);
            }
        }
        // The two sources are never separately normalized and added.
        assert_ne!(out[0], 1.5 + 0.75);
    }

    #[test]
    fn latent_window_items_map_to_recycled_state_slots() {
        let source = Sources::new();
        let slots = 136;
        let code_stride = slots * D;
        let scale_stride = slots * D / 32;
        let mut codes = vec![0x38; 2 * code_stride];
        codes[code_stride..].fill(0x48); // State 0 is 1; state 1 is 4.
        let scales = vec![127; 2 * scale_stride];
        let q = vec![0.0; 3 * 2 * D];
        let mut compressed = source.compressed.clone();
        let page = COMPRESSED.page_bytes(D as u32, 2) as usize;
        // Item 0 reads page 1 (2); item 1 reads page 0 (6); item 2 reads page 1.
        compressed[..D].fill(0x77);
        compressed[D..page].fill(0x38);
        compressed[page..page + D].fill(0x44);
        compressed[page + D..].fill(0x38);
        let mut input = source.input(&[0, 0, 0], &[0, 0, 0]);
        input.q = &q;
        input.row_batch = &[2, 0, 1];
        input.sinks = Some(&[0.0, 0.0]);
        input.window = LatentWindowInput {
            codes: &codes,
            scales: &scales,
            item_slots: &[1, -1, 0],
            sequences: 2,
            code_stride,
            scale_stride,
            storage: LatentWindowStorage::Staged {
                slots: slots as u32,
            },
            lists: LatentKeyLists {
                indices: &[0, 0, 0],
                stride: 1,
                budget: 1,
            },
        };
        let compressed_input = input.compressed.as_mut().unwrap();
        compressed_input.pool = &compressed;
        compressed_input.paging.block_tables = &[1, 0, 1];
        compressed_input.paging.max_pages = 1;
        compressed_input.lists = input.window.lists;
        fn check(input: LatentSourcesInput<'_>, expected: [f64; 3]) {
            let mut query = query(Output::Normalized);
            if let Query::Attention { capacity, .. } = &mut query {
                *capacity = 3;
            }
            let actual = normalized(latent_attention_with_sources(query, input).unwrap());
            for (row, value) in expected.into_iter().enumerate() {
                assert!(
                    actual[row * 2 * D..(row + 1) * 2 * D]
                        .iter()
                        .all(|&v| v == value),
                    "row {row}"
                );
            }
        }
        check(input, [1.0, 2.0, 3.0]);
        // Retire item 2, then assign its state slot to item 1. Compressed
        // tables keep the original item index through both map changes.
        input.window.item_slots = &[1, 0, -1];
        check(input, [1.0, 2.0, 7.0 / 3.0]);
        for invalid in [-1, i32::MIN, 2, i32::MAX] {
            let mapping = [invalid; 3];
            let mut empty = input;
            empty.window.item_slots = &mapping;
            empty.window.codes = &[];
            empty.window.scales = &[];
            check(empty, [1.0, 1.0, 3.0]);
        }
    }

    #[test]
    fn latent_compressed_pages_use_their_own_row_count_and_skip_hole_pages() {
        let source = Sources::new();
        for page_size in [32, 64] {
            let stride = COMPRESSED.page_bytes(D as u32, page_size as u32) as usize + 64;
            let mut pool = vec![0x38; 3 * stride];
            for (physical, code) in [(0, 0x22), (1, 0x44), (2, 0x66)] {
                pool[physical * stride..physical * stride + page_size * D / 2].fill(code);
            }
            let ids = [
                page_size as i32 - 1,
                page_size as i32,
                2 * page_size as i32,
                3 * page_size as i32,
            ];
            let mut input = source.input(&[], &ids);
            let compressed = input.compressed.as_mut().unwrap();
            compressed.pool = &pool;
            compressed.pages = 3;
            compressed.paging = LatentPaging {
                block_tables: &[2, 0, -1, 1],
                max_pages: 4,
                page_stride: stride,
            };
            let mut query = query(Output::Normalized);
            if let Query::Attention { page_size: p, .. } = &mut query {
                *p = page_size as u32;
            }
            let out = normalized(latent_attention_with_sources(query, input).unwrap());
            assert!(out[..D].iter().all(|&x| x == 7.0 / 4.0));
            // A hole page must not read the pool, even with a live position.
            let mut hole = input;
            let hole_id = [2 * page_size as i32];
            let compressed = hole.compressed.as_mut().unwrap();
            compressed.pool = &[];
            compressed.lists = LatentKeyLists {
                indices: &hole_id,
                stride: 1,
                budget: 1,
            };
            assert_eq!(
                normalized(latent_attention_with_sources(query, hole).unwrap()),
                vec![0.0; 2 * D]
            );
        }
    }

    #[test]
    fn latent_bf16_sources_read_the_same_quantized_operands() {
        let mut source = Sources::new();
        for (i, q) in source.q.iter_mut().enumerate() {
            *q = bf16(((i % 7) as f64 - 3.0) * 0.125);
        }
        let expand = |format: Format, page: usize, count: usize, bytes: &[u8]| {
            let stride = format.page_bytes(D as u32, page as u32) as usize;
            let mut out = Vec::new();
            for p in 0..count {
                for slot in 0..page {
                    for value in decode_native_page(
                        format,
                        D,
                        page,
                        &bytes[p * stride..(p + 1) * stride],
                        slot,
                    )
                    .unwrap()
                    {
                        out.extend_from_slice(&f32_to_bf16(value as f32).to_le_bytes());
                    }
                }
            }
            out
        };
        let window = expand(WINDOW, 128, 1, &source.window);
        let compressed = expand(COMPRESSED, 2, 2, &source.compressed);
        for output in [Output::Normalized, Output::Partial] {
            let input = source.input(&[127, -1, 0], &[0, 2, -1, 3]);
            let expected = latent_attention_with_sources(query(output), input).unwrap();
            let mut stored = input;
            stored.window.codes = &window;
            stored.window.code_stride = 128 * D * 2;
            stored.window.scales = &[];
            stored.window.scale_stride = 0;
            let c = stored.compressed.as_mut().unwrap();
            c.pool = &compressed;
            c.paging.page_stride = 2 * D * 2;
            let mut q = query(output);
            if let Query::Attention { kv, .. } = &mut q {
                *kv = Format::Bf16;
            }
            let actual = latent_attention_with_sources(q, stored).unwrap();
            match (actual, expected) {
                (ResultRows::Normalized(a), ResultRows::Normalized(b)) => {
                    assert_eq!(a.values, b.values)
                }
                (ResultRows::Partial(a), ResultRows::Partial(b)) => {
                    assert_eq!(a.maximum.values, b.maximum.values);
                    assert_eq!(a.denominator.values, b.denominator.values);
                    assert_eq!(a.numerator.values, b.numerator.values);
                }
                _ => panic!("output mode"),
            }
            let mut invalid = window.clone();
            invalid[..2].copy_from_slice(&0x7fc0_u16.to_le_bytes());
            stored.window.codes = &invalid;
            assert_eq!(
                latent_attention_with_sources(q, stored).unwrap_err().0,
                "latent BF16 stored operands must be finite"
            );
        }
    }

    #[test]
    fn latent_positive_page_ids_outside_admission_are_holes() {
        let source = Sources::new();
        for page in [32, 64] {
            let payload = COMPRESSED.page_bytes(D as u32, page) as usize;
            let pool = vec![0x38; 2 * payload];
            let mut query = query(Output::Normalized);
            if let Query::Attention { page_size, .. } = &mut query {
                *page_size = page;
            }
            for outside in [1, 2, i32::MAX] {
                let table = [outside];
                let mut input = source.input(&[], &[0]);
                let compressed = input.compressed.as_mut().unwrap();
                compressed.pages = 1;
                compressed.pool = &pool[..payload];
                compressed.paging = LatentPaging {
                    block_tables: &table,
                    max_pages: 1,
                    page_stride: payload,
                };
                assert_eq!(
                    normalized(latent_attention_with_sources(query, input).unwrap()),
                    vec![0.0; 2 * D]
                );
                // Unadmitted trailing pages in a larger view stay holes.
                input.compressed.as_mut().unwrap().pool = &pool;
                assert_eq!(
                    normalized(latent_attention_with_sources(query, input).unwrap()),
                    vec![0.0; 2 * D]
                );
                input.compressed.as_mut().unwrap().paging.page_stride = payload - 1;
                assert!(latent_attention_with_sources(query, input).is_err());
            }
            let mut input = source.input(&[], &[0]);
            let compressed = input.compressed.as_mut().unwrap();
            compressed.pages = 1;
            compressed.pool = &pool[..payload - 1];
            compressed.paging = LatentPaging {
                block_tables: &[0],
                max_pages: 1,
                page_stride: payload,
            };
            assert!(latent_attention_with_sources(query, input).is_err());
            input.compressed.as_mut().unwrap().pages = 0;
            assert!(latent_attention_with_sources(query, input).is_err());
        }
    }

    #[test]
    fn latent_out_of_range_window_slots_are_holes() {
        let source = Sources::new();
        for (storage, slots, outside) in [
            (LatentWindowStorage::Ring128, 128, vec![128, -2, i32::MIN]),
            (
                LatentWindowStorage::Staged { slots: 136 },
                136,
                vec![136, 137, i32::MAX, -2, i32::MIN],
            ),
        ] {
            let codes = vec![0x38; slots * D];
            let scales = vec![127; slots * (D / 32)];
            let mut query = query(Output::Normalized);
            if let Query::Attention {
                second_fp4_keys, ..
            } = &mut query
            {
                *second_fp4_keys = false;
            }
            let baseline = [-1, 0, slots as i32 - 1];
            let mut input = source.input(&baseline, &[]);
            input.compressed = None;
            input.window.codes = &codes;
            input.window.scales = &scales;
            input.window.code_stride = codes.len();
            input.window.scale_stride = scales.len();
            input.window.storage = storage;
            let expected = normalized(latent_attention_with_sources(query, input).unwrap());
            for invalid in outside {
                let indices = [invalid, 0, slots as i32 - 1];
                let mut actual = input;
                actual.window.lists.indices = &indices;
                assert_eq!(
                    normalized(latent_attention_with_sources(query, actual).unwrap()),
                    expected
                );
                // An out-of-range-only list must not touch either plane.
                let mut empty = input;
                let single = [invalid];
                empty.window.lists = LatentKeyLists {
                    indices: &single,
                    stride: 1,
                    budget: 1,
                };
                empty.window.codes = &[];
                empty.window.scales = &[];
                assert_eq!(
                    normalized(latent_attention_with_sources(query, empty).unwrap()),
                    vec![0.0; 2 * D]
                );
            }
        }
    }

    #[test]
    fn latent_holes_do_not_read_storage_and_large_sinks_are_stable() {
        let mut source = Sources::new();
        source.sinks = [1000.0, -1000.0];
        let mut input = source.input(&[-1, -1], &[-1]);
        input.window.codes = &[];
        input.window.scales = &[];
        input.compressed.as_mut().unwrap().pool = &[];
        input.compressed.as_mut().unwrap().paging.block_tables = &[];
        assert_eq!(
            normalized(latent_attention_with_sources(query(Output::Normalized), input).unwrap()),
            vec![0.0; 2 * D]
        );
        let out = normalized(
            latent_attention_with_sources(query(Output::Normalized), source.input(&[0], &[]))
                .unwrap(),
        );
        assert_eq!(out[0], 0.0);
        assert_eq!(out[D], 1.0);
    }

    #[test]
    fn latent_native_paged_write_and_staged_prefill_preserve_earlier_rows() {
        for format in [WINDOW, COMPRESSED] {
            let page = format.page_bytes(D as u32, 2) as usize;
            let mut pool = vec![0xa5; 2 * (page + 64)];
            let paging = LatentPaging {
                block_tables: &[1, 0],
                max_pages: 2,
                page_stride: page + 64,
            };
            let row = vec![f32_to_bf16(1.0); 2 * D];
            latent_kv_write(
                Query::KvWrite {
                    latent: D as u32,
                    page_size: 2,
                    kv: format,
                    capacity: 2,
                },
                &row,
                &[0, 0],
                &[3, -1],
                paging,
                &mut pool,
            )
            .unwrap();
            let decoded = decode_native_page(format, D, 2, &pool[..page], 1).unwrap();
            // E4M3(1/6) = 0.171875, then E2M1(1/scale) = 6.
            let expected = if format == WINDOW { 1.0 } else { 1.03125 };
            assert!(decoded.iter().all(|&x| x == expected));
            assert!(pool[page..].iter().all(|&x| x == 0xa5));
        }
        // A staged prefill can read row 128 as well as row 0, even though
        // both would occupy slot 0 in the final 128-slot ring.
        let source = Sources::new();
        let mut staged = vec![0x38; 129 * D];
        staged.extend(vec![127; 129 * (D / 32)]);
        staged[128 * D..129 * D].fill(0x40);
        let mut input = source.input(&[0, 128], &[]);
        input.window.codes = &staged[..129 * D];
        input.window.scales = &staged[129 * D..];
        input.window.code_stride = staged.len();
        input.window.scale_stride = staged.len();
        input.window.storage = LatentWindowStorage::Staged { slots: 129 };
        let out =
            normalized(latent_attention_with_sources(query(Output::Normalized), input).unwrap());
        assert_eq!(out[0], 1.0); // (1 + 2) / (2 + exp(0))
        input.window.storage = LatentWindowStorage::Ring128;
        let out =
            normalized(latent_attention_with_sources(query(Output::Normalized), input).unwrap());
        assert_eq!(out[0], 0.5); // Slot 128 is a hole; only slot 0 contributes.
    }

    #[test]
    fn latent_tail_is_in_place_and_inverse_follows_the_bf16_boundary() {
        let mut input = vec![0.0; 2 * D];
        input[0] = -0.0;
        input[D - 64] = 3.0;
        input[D - 63] = 4.0;
        input[2 * D - 64] = -2.0;
        input[2 * D - 63] = 1.0;
        let frequencies = [0.6_f32, 0.8_f32].repeat(32);
        let forward = latent_tail_rope(&input, D, 2, 64, &frequencies, false).unwrap();
        assert_eq!(forward.values.len(), 2 * D);
        assert_eq!(forward.values[0].to_bits(), (-0.0_f64).to_bits());
        assert!((forward.values[D - 64] + 1.4).abs() < 2e-7);
        let stored = bf16_exact(&forward.values);
        let inverse = latent_tail_rope(&stored, D, 2, 64, &frequencies, true).unwrap();
        let x = stored[D - 64];
        let y = stored[D - 63];
        assert_eq!(
            inverse.values[D - 64],
            x * f64::from(0.6_f32) + y * f64::from(0.8_f32)
        );
        assert_eq!(
            inverse.values[D - 63],
            -x * f64::from(0.8_f32) + y * f64::from(0.6_f32)
        );
        assert_ne!(inverse.values[D - 64], input[D - 64]);
        let unchanged = latent_tail_rope(&input, D, 2, 0, &[f32::NAN], true).unwrap();
        assert_eq!(
            unchanged
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            input.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn latent_online_softmax_rounds_unnormalized_probabilities_before_pv() {
        use kernel_core::latent_attention_oracle::latent_online_softmax;
        // A masked tile does not read its poisoned values. The next tile
        // raises the maximum, so its rescale must precede the final sink.
        let mut scores = vec![f64::NEG_INFINITY; 129];
        let mut values = vec![f64::NAN; 129];
        scores[0] = 0.0;
        values[0] = 2.0;
        scores[1] = -0.7;
        values[1] = -3.0;
        scores[128] = 0.3;
        values[128] = 4.0;
        let out = latent_online_softmax(&scores, &values, 1, 0.2).unwrap();
        let p = (-0.7_f64).exp();
        let rescale = (-0.3_f64).exp();
        let den = (1.0 + p) * rescale + 1.0 + (f64::from(0.2_f32) - 0.3).exp();
        let expected = ((2.0 - 3.0 * bf16(p)) * rescale + 4.0) / den;
        assert!((out[0] - expected).abs() < 1e-15);
        let wrong =
            2.0 * bf16(rescale / den) - 3.0 * bf16(p * rescale / den) + 4.0 * bf16(1.0 / den);
        assert!((wrong - expected).abs() > 1e-4);
        // Letting the sink join the maximum before PV changes its rounding.
        let sink = 5.0_f32;
        let correct = latent_online_softmax(&[0.0, -0.7], &[2.0, -3.0], 1, sink).unwrap()[0];
        let bad = (2.0 * bf16((-5.0_f64).exp()) - 3.0 * bf16((-5.7_f64).exp()))
            / ((-5.0_f64).exp() + (-5.7_f64).exp() + 1.0);
        assert!((correct - bad).abs() > 1e-7);
    }

    #[test]
    fn latent_partial_bound_propagates_the_maximum_score_error() {
        use kernel_core::oracle::{OracleOutput, check_numeric, numeric_bound};
        let mut source = Sources::new();
        source.q.fill(0.0);
        let a = 2_f32.powi(24);
        source.q[0] = f64::from(a);
        source.q[128] = 1.0;
        source.q[256] = -f64::from(a);
        source.window.fill(0);
        source.window[128 * D..].fill(127);
        for at in [0, 128, 256, D + 1] {
            source.window[at] = 0x38;
        }
        let mut query = query(Output::Partial);
        if let Query::Attention {
            second_fp4_keys, ..
        } = &mut query
        {
            *second_fp4_keys = false;
        }
        let mut input = source.input(&[0, 1], &[]);
        input.compressed = None;
        let ResultRows::Partial(partial) = latent_attention_with_sources(query, input).unwrap()
        else {
            panic!("partial result")
        };
        let scale = f64::from((D as f32).powf(-0.5));
        assert_eq!(partial.maximum.values[0], scale);
        assert_eq!(partial.numerator.values[1], (-scale).exp());
        // Each 128-channel block is exact. The ordered F32 block fold
        // loses the middle one, so both device scores and their max are 0.
        let device_score = ((a + 1.0) - a) * scale as f32;
        assert_eq!(device_score, 0.0);
        for (values, at, actual) in [
            (&partial.maximum, 0, f64::from(device_score)),
            (&partial.denominator, 0, 2.0),
            (&partial.numerator, 1, 1.0),
        ] {
            let reference = OracleOutput::new(
                vec![values.values[at]],
                vec![values.sum_abs_products[at]],
                values.reduction_lengths[at],
            )
            .unwrap();
            check_numeric(&[actual], &reference, DType::F32).unwrap();
            let limit = numeric_bound(&reference, 0, DType::F32);
            assert!(
                check_numeric(&[reference.values[0] + 2.0 * limit], &reference, DType::F32)
                    .is_err()
            );
        }
    }

    #[test]
    fn latent_probability_rounding_bound_covers_cancellation() {
        use kernel_core::latent_attention_oracle::latent_online_softmax;
        use kernel_core::oracle::{OracleOutput, check_numeric};
        let mut source = Sources::new();
        source.window.fill(0);
        source.sinks = [0.0; 2];
        source.q[0] = -22.625;
        let mut keys = vec![0.0; 2 * D];
        keys[1] = 1.0;
        keys[D] = 1.0;
        keys[D + 1] = -2.75;
        let bits = keys
            .iter()
            .map(|&x| f32_to_bf16(x as f32))
            .collect::<Vec<_>>();
        latent_kv_write(
            Query::KvWrite {
                latent: D as u32,
                page_size: 128,
                kv: WINDOW,
                capacity: 2,
            },
            &bits,
            &[0, 0],
            &[0, 1],
            LatentPaging {
                block_tables: &[0],
                max_pages: 1,
                page_stride: source.window.len(),
            },
            &mut source.window,
        )
        .unwrap();
        for row in 0..2 {
            assert_eq!(
                decode_native_page(WINDOW, D, 128, &source.window, row).unwrap(),
                keys[row * D..(row + 1) * D]
            );
        }
        let mut query = query(Output::Normalized);
        if let Query::Attention {
            second_fp4_keys,
            tail_rope_dims,
            ..
        } = &mut query
        {
            *second_fp4_keys = false;
            *tail_rope_dims = 0;
        }
        let mut input = source.input(&[0, 1], &[]);
        input.compressed = None;
        let ResultRows::Normalized(ideal) = latent_attention_with_sources(query, input).unwrap()
        else {
            panic!()
        };
        let scores = [0.0, -22.625 * f64::from((D as f32).powf(-0.5))];
        let staged = latent_online_softmax(&scores, &keys, D, 0.0).unwrap();
        let actual = bf16(staged[1]);
        assert!((actual - (-0.004119873046875)).abs() < 1e-12);
        assert!((ideal.values[1] - (-0.004973376)).abs() < 1e-9);
        let reference = OracleOutput::new(
            vec![ideal.values[1]],
            vec![ideal.sum_abs_products[1]],
            ideal.reduction_lengths[1],
        )
        .unwrap();
        check_numeric(&[actual], &reference, DType::Bf16).unwrap();
        let limit = kernel_core::oracle::numeric_bound(&reference, 0, DType::Bf16);
        assert!(
            check_numeric(
                &[reference.values[0] + 2.0 * limit],
                &reference,
                DType::Bf16
            )
            .is_err()
        );

        // Partials must carry the same PV rounding allowance into their
        // final sink-once merge. Their denominator still uses unrounded P.
        if let Query::Attention { output, .. } = &mut query {
            *output = Output::Partial;
        }
        let ResultRows::Partial(partial) = latent_attention_with_sources(query, input).unwrap()
        else {
            panic!()
        };
        let p = scores[1].exp();
        assert_eq!(partial.denominator.values[0], 1.0 + p);
        let numerator = 1.0 - 2.75 * bf16(p);
        let num_reference = OracleOutput::new(
            vec![partial.numerator.values[1]],
            vec![partial.numerator.sum_abs_products[1]],
            partial.numerator.reduction_lengths[1],
        )
        .unwrap();
        check_numeric(&[numerator], &num_reference, DType::F32).unwrap();
        let merged = merge_partials_with_sink(&[partial], &source.sinks).unwrap();
        let merged_reference = OracleOutput::new(
            vec![merged.values[1]],
            vec![merged.sum_abs_products[1]],
            merged.reduction_lengths[1],
        )
        .unwrap();
        check_numeric(&[actual], &merged_reference, DType::Bf16).unwrap();

        // Reversing keys across two tiles changes the rounded P path but
        // not the ideal reference or the bound it must satisfy.
        let mut tiled_scores = vec![f64::NEG_INFINITY; 65];
        let mut tiled_values = vec![f64::NAN; 65];
        tiled_scores[0] = scores[1];
        tiled_values[0] = -2.75;
        tiled_scores[64] = scores[0];
        tiled_values[64] = 1.0;
        let tiled = bf16(latent_online_softmax(&tiled_scores, &tiled_values, 1, 0.0).unwrap()[0]);
        assert_ne!(tiled, actual);
        check_numeric(&[tiled], &reference, DType::Bf16).unwrap();
    }

    #[test]
    fn latent_sources_refuse_malformed_storage_and_operands() {
        let source = Sources::new();
        for (w, c) in [
            (&[][..], &[-2][..]),
            (&[0, 0][..], &[][..]),
            (&[][..], &[4][..]),
            (&[][..], &[0, 0][..]),
        ] {
            assert!(
                latent_attention_with_sources(query(Output::Normalized), source.input(w, c))
                    .is_err()
            );
        }
        let mut input = source.input(&[0], &[0]);
        input.sinks = None;
        assert!(latent_attention_with_sources(query(Output::Normalized), input).is_err());
        input.sinks = Some(&[f32::NAN, 0.0]);
        assert!(latent_attention_with_sources(query(Output::Normalized), input).is_err());
        input.sinks = Some(&source.sinks);
        input.compressed = None;
        assert!(latent_attention_with_sources(query(Output::Normalized), input).is_err());
        for (format, code, scale) in [
            (WINDOW, 0x7f, 127),
            (WINDOW, 0x38, 255),
            (WINDOW, 0x7e, 254),
            (WINDOW, 1, 0),
            (COMPRESSED, 0x22, 0),
            (COMPRESSED, 0x22, 0xb8),
        ] {
            let bytes = if format == WINDOW { D } else { D / 2 };
            let mut page = vec![code; bytes];
            page.extend(vec![scale; if format == WINDOW { D / 32 } else { D / 16 }]);
            assert!(decode_native_page(format, D, 1, &page, 0).is_err());
        }
    }

    #[test]
    fn latent_contracts_cover_phases_and_preserve_existing_ids() {
        for phase in [
            AttentionPhase::Prefill,
            AttentionPhase::Decode,
            AttentionPhase::Verify,
        ] {
            for output in [Output::Normalized, Output::Partial] {
                let mut q = query(output);
                if let Query::Attention { phase: p, .. } = &mut q {
                    *p = phase;
                }
                q.validate().unwrap();
                let id = q.numerical_contract_id().unwrap();
                assert!(
                    id.as_str()
                        .contains("sink-once.tail-rope64-in-place.window-fp4-paged")
                );
            }
        }
        let mut q = query(Output::Normalized);
        if let Query::Attention { kv, .. } = &mut q {
            *kv = Format::Bf16;
        }
        q.validate().unwrap();
        if let Query::Attention { kv, .. } = &mut q {
            *kv = Format::Fp8E4M3Block128;
        }
        assert_eq!(
            q.validate().unwrap_err().0,
            "latent attention second source requires a native FP8 or BF16 window"
        );
    }
}

/// One lagged Sinkhorn mixer's learned parameters: `fn` `[(2 + N) * N,
/// N * H]` as F32 values, `base` and `scale`.
struct HcMixer {
    projection: Vec<f64>,
    base: Vec<f64>,
    scale: [f64; 3],
}

/// The pinned config's mHC parameters: 4 streams, 20 iterations, Sinkhorn
/// eps 1e-6 (`hc_eps`), norm eps 1e-20 (`norm_eps`, the mixer's rsqrt), F32
/// `fn`, the lagged placement.
fn lagged_hc_mix(width: u32) -> kernel_core::sinkhorn_hyper_connection::SinkhornMix {
    use kernel_core::sinkhorn_hyper_connection::{
        HyperConnectionPlacement, SinkhornMix, SinkhornOrder,
    };
    SinkhornMix {
        streams: 4,
        width,
        iterations: 20,
        eps_bits: 1e-6_f32.to_bits(),
        norm_eps_bits: 1e-20_f32.to_bits(),
        order: SinkhornOrder::RowSoftmaxThenColumnFirst,
        placement: HyperConnectionPlacement::LaggedSinglePass,
        fn_dtype: DType::F32,
    }
}

/// `Block.hc_mixes` and `hc_split_sinkhorn` of the vendor code
/// (`model.py:950-957`, `kernel.py:407-462`) in plain f64 for one token's
/// streams `[N * H]`: the RMS scale over the flat streams after the
/// projection, `pre = sigmoid(m * s0 + b) + eps`, `post = 2 sigmoid(m * s1 +
/// b)`, the comb's row softmax plus eps, one column normalization, then 19
/// (row, column) rounds. The values are unrounded; the vendor stores them
/// in F32 (`kernel.py:417-419`), as [`f32_stored`] does.
fn vendor_hc_mixes(x: &[f64], mixer: &HcMixer, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let eps = f64::from(1e-6_f32);
    let norm_eps = f64::from(1e-20_f32);
    let k = x.len();
    let rsqrt = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / k as f64 + norm_eps).sqrt();
    let mixes = mixer
        .projection
        .chunks_exact(k)
        .map(|w| w.iter().zip(x).map(|(w, x)| w * x).sum::<f64>() * rsqrt)
        .collect::<Vec<_>>();
    let sigmoid = |v: f64| 1.0 / (1.0 + (-v).exp());
    let pre = (0..n)
        .map(|j| sigmoid(mixes[j] * mixer.scale[0] + mixer.base[j]) + eps)
        .collect();
    let post = (0..n)
        .map(|j| 2.0 * sigmoid(mixes[n + j] * mixer.scale[1] + mixer.base[n + j]))
        .collect();
    let mut comb = (0..n * n)
        .map(|m| mixes[2 * n + m] * mixer.scale[2] + mixer.base[2 * n + m])
        .collect::<Vec<_>>();
    for row in comb.chunks_exact_mut(n) {
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        row.iter_mut().for_each(|v| *v = (*v - max).exp());
        let sum = row.iter().sum::<f64>();
        row.iter_mut().for_each(|v| *v = *v / sum + eps);
    }
    let columns = |comb: &mut Vec<f64>| {
        for j in 0..n {
            let sum = (0..n).map(|i| comb[i * n + j]).sum::<f64>();
            (0..n).for_each(|i| comb[i * n + j] /= sum + eps);
        }
    };
    columns(&mut comb);
    for _ in 1..20 {
        for row in comb.chunks_exact_mut(n) {
            let sum = row.iter().sum::<f64>();
            row.iter_mut().for_each(|v| *v /= sum + eps);
        }
        columns(&mut comb);
    }
    (pre, post, comb)
}

/// Mixer coefficients as the vendor stores them, in F32.
fn f32_stored(values: &[f64]) -> Vec<f32> {
    values.iter().map(|&v| v as f32).collect()
}

/// `Block.hc_pre` (`model.py:959-962`) in f64 over F32 weights: one token's
/// streams `[N * H]` collapsed by `weights` `[N]`, rounded to BF16.
fn vendor_hc_pre(x: &[f64], weights: &[f32]) -> Vec<f64> {
    let (n, h) = (weights.len(), x.len() / weights.len());
    (0..h)
        .map(|col| {
            bf16(
                (0..n)
                    .map(|i| f64::from(weights[i]) * x[i * h + col])
                    .sum::<f64>(),
            )
        })
        .collect()
}

/// `Block.hc_post` (`model.py:964-968`) in f64 over F32 `post` `[N]` and
/// `comb` `[N, N]`: one token's streams `[N * H]` and sublayer output `y`
/// `[H]`, each expanded stream rounded once to BF16.
fn vendor_hc_post(x: &[f64], y: &[f64], post: &[f32], comb: &[f32]) -> Vec<f64> {
    let (n, h) = (post.len(), y.len());
    (0..n * h)
        .map(|e| {
            let (j, col) = (e / h, e % h);
            let mixed = (0..n)
                .map(|i| f64::from(comb[i * n + j]) * x[i * h + col])
                .sum::<f64>();
            bf16(f64::from(post[j]) * y[col] + mixed)
        })
        .collect()
}

/// A sublayer stand-in: `bf16(collapsed_k * g_k)` with a fixed `g`.
fn hc_sublayer(collapsed: &[f64], g: &[f64]) -> Vec<f64> {
    collapsed
        .iter()
        .zip(g.iter().cycle())
        .map(|(&c, &g)| bf16(c * g))
        .collect()
}

/// The lagged stack (`Block.forward`, `model.py:972-997`;
/// `make_identity_pre_mix`, `:1162-1166`; the head collapse, `:1263-1271`)
/// through the contract's ops, each stage checked with matched inputs: the
/// vendor's stage of the same streams, weights and sublayer output, in f64
/// over its F32 coefficients, lies within that stage's f64 bound, and so do
/// the contract's exact forms. Every sublayer collapses with the previous
/// mixer's `pre` (one-hot first) and the head with the last FFN's.
#[test]
fn lagged_sinkhorn_hyper_connections_match_the_vendor_stages() {
    use kernel_core::oracle::{OracleOutput, check_numeric};
    use kernel_core::sinkhorn_hyper_connection::{SinkhornHyperConnectionQuery, StreamHeadMode};
    use kernel_core::sinkhorn_hyper_connection_oracle::{
        MixWeights, collapse, expand, expand_f32, expand_f32_reference, lagged_mix_collapse,
        one_hot_pre, weighted_head, weighted_head_reference,
    };
    // The pinned width, two layers (attention and FFN mixers each), two
    // token rows. Collapse-weight bases spread over [-3, 3] make each
    // sublayer's pre differ from the previous one's.
    let (n, h, rows, sublayers) = (4_usize, 5120_usize, 2_usize, 4_usize);
    let k = n * h;
    let mix = lagged_hc_mix(h as u32);
    let mixers = (0..sublayers as u64)
        .map(|s| {
            let mut base = values((2 + n) * n, 100 + s, 1.0);
            base[..n].copy_from_slice(&values(n, 200 + s, 3.0));
            HcMixer {
                projection: values((2 + n) * n * k, 300 + s, 0.05)
                    .into_iter()
                    .map(|v| f64::from(v as f32))
                    .collect(),
                base: base.into_iter().map(|v| f64::from(v as f32)).collect(),
                scale: [0.5, 0.25, 2.0],
            }
        })
        .collect::<Vec<_>>();
    let gates = (0..sublayers as u64)
        .map(|s| values(h, 400 + s, 1.5))
        .collect::<Vec<_>>();

    let collapse_query = SinkhornHyperConnectionQuery::MixCollapse {
        mix,
        capacity: rows as u32,
    };
    let expand_query = SinkhornHyperConnectionQuery::ExpandF32 {
        streams: n as u32,
        width: h as u32,
        capacity: rows as u32,
    };
    let bf16_expand_query = SinkhornHyperConnectionQuery::Expand {
        streams: n as u32,
        width: h as u32,
        capacity: rows as u32,
    };
    let head_query = SinkhornHyperConnectionQuery::Head {
        streams: n as u32,
        width: h as u32,
        mode: StreamHeadMode::SigmoidWeighted,
        capacity: rows as u32,
    };
    let to_bits = |v: &[f64]| v.iter().map(|&v| f32_to_bf16(v as f32)).collect::<Vec<_>>();
    let of_bits = |b: &[u16]| {
        b.iter()
            .map(|&b| f64::from(bf16_to_f32(b)))
            .collect::<Vec<_>>()
    };
    let widen = |v: &[f32]| v.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    let within = |actual: &[f64], bound: &OracleOutput, dtype: DType, what: &str| {
        if let Err(failure) = check_numeric(actual, bound, dtype) {
            panic!("{what}: {failure:?}");
        }
    };
    let outside = |actual: &[f64], bound: &OracleOutput, dtype: DType, what: &str| {
        assert!(
            check_numeric(actual, bound, dtype).is_err(),
            "{what} passed its bound"
        );
    };
    let close = |a: &[f64], b: &[f64], what: &str| {
        for (index, (a, b)) in a.iter().zip(b).enumerate() {
            assert!((a - b).abs() <= 1e-12, "{what} {index}: {a} vs {b}");
        }
    };

    let mut x = to_bits(&bf16_exact(&values(rows * k, 7, 2.0)));
    let mut pre_in = one_hot_pre(rows, n);
    let mut previous_pre = pre_in.clone();
    for (mixer, g) in mixers.iter().zip(&gates) {
        let streams = of_bits(&x);
        let weights = MixWeights {
            projection: &mixer.projection,
            base: &mixer.base,
            scale: &mixer.scale,
        };
        let out = lagged_mix_collapse(collapse_query, &streams, &pre_in, weights).unwrap();

        // The mixer: the vendor's coefficients of the same streams agree with
        // the oracle's in f64, and stored in F32 lie within its F32 bounds.
        let (mut pre, mut post, mut comb) = (Vec::new(), Vec::new(), Vec::new());
        for row in streams.chunks_exact(k) {
            let (p, q, c) = vendor_hc_mixes(row, mixer, n);
            pre.extend(p);
            post.extend(q);
            comb.extend(c);
        }
        close(&out.pre.values, &pre, "pre");
        close(&out.post.values, &post, "post");
        close(&out.comb.values, &comb, "comb");
        within(
            &widen(&f32_stored(&pre)),
            &out.pre,
            DType::F32,
            "vendor pre",
        );
        within(
            &widen(&f32_stored(&post)),
            &out.post,
            DType::F32,
            "vendor post",
        );
        within(
            &widen(&f32_stored(&comb)),
            &out.comb,
            DType::F32,
            "vendor comb",
        );

        // The collapse with the previous sublayer's stored pre.
        let vendor = streams
            .chunks_exact(k)
            .zip(pre_in.chunks_exact(n))
            .flat_map(|(row, w)| vendor_hc_pre(row, w))
            .collect::<Vec<_>>();
        within(&vendor, &out.collapsed, DType::Bf16, "vendor collapse");
        let collapsed = collapse(collapse_query, &x, &pre_in).unwrap();
        within(
            &of_bits(&collapsed),
            &out.collapsed,
            DType::Bf16,
            "collapse",
        );
        // Negative control: this mixer's own pre (the per-sublayer
        // placement) is outside the lagged collapse's bound.
        let own = collapse(collapse_query, &x, &f32_stored(&out.pre.values)).unwrap();
        outside(
            &of_bits(&own),
            &out.collapsed,
            DType::Bf16,
            "own-pre collapse",
        );

        // The expand of the same streams and sublayer output with the
        // stored F32 post and comb.
        let y = to_bits(&hc_sublayer(&of_bits(&collapsed), g));
        let (post, comb) = (f32_stored(&out.post.values), f32_stored(&out.comb.values));
        let bound = expand_f32_reference(
            expand_query,
            &streams,
            &of_bits(&y),
            &widen(&post),
            &widen(&comb),
        )
        .unwrap();
        let vendor = (0..rows)
            .flat_map(|r| {
                vendor_hc_post(
                    &streams[r * k..][..k],
                    &of_bits(&y[r * h..][..h]),
                    &post[r * n..][..n],
                    &comb[r * n * n..][..n * n],
                )
            })
            .collect::<Vec<_>>();
        within(&vendor, &bound, DType::Bf16, "vendor expand");
        let expanded = expand_f32(expand_query, &x, &y, &post, &comb).unwrap();
        within(&of_bits(&expanded), &bound, DType::Bf16, "expand");
        // Negative control: BF16 post and comb with the BF16 expand's three
        // roundings are outside the F32 expand's bound.
        let rounded = |v: &[f32]| v.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>();
        let bf16_mixed =
            expand(bf16_expand_query, &x, &y, &rounded(&post), &rounded(&comb)).unwrap();
        outside(&of_bits(&bf16_mixed), &bound, DType::Bf16, "BF16 expand");

        x = expanded;
        previous_pre = pre_in;
        pre_in = f32_stored(&out.pre.values);
    }

    // The head collapses with the last FFN's stored pre.
    let streams = of_bits(&x);
    let bound = weighted_head_reference(head_query, &streams, &pre_in).unwrap();
    let vendor = streams
        .chunks_exact(k)
        .zip(pre_in.chunks_exact(n))
        .flat_map(|(row, w)| vendor_hc_pre(row, w))
        .collect::<Vec<_>>();
    within(&vendor, &bound, DType::Bf16, "vendor head");
    let head = weighted_head(head_query, &x, &pre_in).unwrap();
    within(&of_bits(&head), &bound, DType::Bf16, "head");
    // Negative control: the attention mixer's pre, one sublayer early.
    let early = weighted_head(head_query, &x, &previous_pre).unwrap();
    outside(&of_bits(&early), &bound, DType::Bf16, "early-pre head");
}

#[test]
fn window_block_visibility_pins_full_key_lists() {
    let visibility = WindowBlockVisibility {
        window_size: 8,
        block_size: 5,
    };
    // The newest projected main row is included. Uninitialized ring slots
    // are skipped without moving the block's base from the full capacity.
    for (main_rows, expected) in [
        (1, vec![0, 8, 9, 10, 11, 12]),
        (2, vec![0, 1, 8, 9, 10, 11, 12]),
        (7, vec![0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]),
        (8, (0..13).collect()),
        (9, (0..13).collect()),
        (16, (0..13).collect()),
        (17, (0..13).collect()),
        (u64::MAX, (0..13).collect()),
    ] {
        for query in 0..5 {
            assert_eq!(
                visibility
                    .key_indices(main_rows, query)
                    .unwrap()
                    .collect::<Vec<_>>(),
                expected,
                "main_rows={main_rows}, query={query}"
            );
        }
    }
    // Pin the 128-row window and five-position block at both boundaries.
    let visibility = WindowBlockVisibility {
        window_size: 128,
        ..visibility
    };
    for (start_pos, main_end) in [(1, 2), (126, 127), (127, 128), (128, 128), (256, 128)] {
        let mut expected = (0..main_end).collect::<Vec<_>>();
        expected.extend([128, 129, 130, 131, 132]);
        for query in 0..5 {
            assert_eq!(
                visibility
                    .key_indices(start_pos + 1, query)
                    .unwrap()
                    .collect::<Vec<_>>(),
                expected
            );
        }
    }
}

#[test]
fn window_block_visibility_tracks_committed_rows_through_wrap() {
    // Simulate writes independently of the index generator. Logical token
    // sets stay the last window plus the block, while physical order stays
    // slot order. Different block widths keep the contract model-neutral.
    for window_size in [1, 3, 8, 128] {
        for block_size in [1, 2, 5, 7] {
            let visibility = WindowBlockVisibility {
                window_size,
                block_size,
            };
            let mut ring = vec![None; window_size as usize];
            for position in 0..3 * u64::from(window_size) + 6 {
                ring[(position % u64::from(window_size)) as usize] = Some(position);
                let main_rows = position + 1;
                for query in 0..block_size {
                    let indices = visibility
                        .key_indices(main_rows, query)
                        .unwrap()
                        .collect::<Vec<_>>();
                    let mut actual = indices
                        .iter()
                        .map(|&index| {
                            if index < window_size as i32 {
                                ring[index as usize].expect("uninitialized main row is invisible")
                            } else {
                                main_rows + (index as u64 - u64::from(window_size))
                            }
                        })
                        .collect::<Vec<_>>();
                    let mut physical_order = ring.iter().flatten().copied().collect::<Vec<_>>();
                    physical_order.extend(main_rows..main_rows + u64::from(block_size));
                    assert_eq!(actual, physical_order);
                    actual.sort_unstable();
                    let first = main_rows.saturating_sub(u64::from(window_size));
                    assert_eq!(
                        actual,
                        (first..main_rows + u64::from(block_size)).collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}

#[test]
fn window_block_visibility_rejects_causal_and_truncated_block_controls() {
    use kernel_core::attend::{Attend, AttendValues, KvRows, attend, window};

    let visibility = WindowBlockVisibility {
        window_size: 8,
        block_size: 5,
    };
    let lists = (0..5)
        .map(|query| {
            visibility
                .key_indices(3, query)
                .unwrap()
                .map(i64::from)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    // Zero queries give every visible key equal weight. Poisoned ring gaps
    // catch a block incorrectly placed after the initialized main prefix.
    let data = [
        0.0,
        0.0,
        0.0,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        1.0,
        2.0,
        3.0,
        4.0,
        16.0,
    ];
    let segments = [&data[..]];
    let keys = KvRows {
        segments: &segments,
        heads: 1,
        dim: 1,
    };
    let evaluate = |lists: &[Vec<i64>]| {
        attend(
            &vec![0.0; lists.len()],
            keys,
            AttendValues::SharedKeys,
            lists,
            &Attend::f64(1.0),
        )
        .unwrap()
        .values
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>()
    };
    let full = evaluate(&lists);
    assert_eq!(full, vec![3.25_f64.to_bits(); 5]);

    let causal = (0..5)
        .map(|query| {
            (0..3)
                .chain(window(query, 0..5, true, None, None).map(|key| 8 + key as i64))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let causal_output = evaluate(&causal);
    for query in 0..4 {
        assert_ne!(lists[query], causal[query]);
        assert_ne!(full[query], causal_output[query]);
    }
    assert_eq!(full[4], causal_output[4]);

    // A scheduler budget slices emitted outputs, never the computed block.
    // Evaluating only the first K queries with all five keys is bitwise
    // equal; shrinking the block's key set changes every retained output.
    for budget in 1..5 {
        assert_eq!(evaluate(&lists[..budget]), full[..budget]);
        let shortened = WindowBlockVisibility {
            block_size: budget as u32,
            ..visibility
        };
        let truncated = (0..budget as u32)
            .map(|query| {
                shortened
                    .key_indices(3, query)
                    .unwrap()
                    .map(i64::from)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert!(
            evaluate(&truncated)
                .iter()
                .zip(&full)
                .all(|(short, full)| short != full)
        );
    }
}

#[test]
fn retained_window_stage_contract_requires_complete_independent_planes() {
    use kernel_core::latent_window_stage::LatentWindowStage;
    let query = LatentWindowStage {
        latent: 512,
        window: 128,
        ring_slots: 136,
        staged_slots: 512,
    };
    query.validate().unwrap();
    let id = query.numerical_contract_id().unwrap();
    for invalid in [
        LatentWindowStage { latent: 0, ..query },
        LatentWindowStage {
            latent: 33,
            ..query
        },
        LatentWindowStage { window: 1, ..query },
        LatentWindowStage {
            ring_slots: 127,
            ..query
        },
        LatentWindowStage {
            staged_slots: 126,
            ..query
        },
    ] {
        assert!(invalid.validate().is_err());
        assert!(invalid.numerical_contract_id().is_err());
    }
    assert_ne!(
        id,
        LatentWindowStage {
            staged_slots: 256,
            ..query
        }
        .numerical_contract_id()
        .unwrap()
    );
}

#[test]
fn staged_window_lists_bind_the_retained_prefix_before_slot_mapping() {
    use kernel_core::latent_staged_window_keys::{LatentStagedWindowKeys, keys};
    let q = LatentStagedWindowKeys {
        window: 128,
        ring_slots: 136,
        staged_slots: 256,
    };
    let actual = keys(q, &[8], &[0], &[8], &[8], &[0], &[200]).unwrap();
    assert_eq!(
        actual
            .iter()
            .filter(|&&p| p >= 0)
            .copied()
            .collect::<Vec<_>>(),
        vec![8]
    );
    let old = kernel_core::latent_glue_oracle::window_keys(
        kernel_core::latent_glue::LatentWindowKeys {
            window: 128,
            addressing: kernel_core::latent_glue::WindowAddressing::Staged { slots: 256 },
        },
        8,
        0,
    )
    .unwrap();
    assert_ne!(
        actual, old,
        "storage bounds alone expose missing prefix cells"
    );
    q.validate().unwrap();
    assert_ne!(
        q.numerical_contract_id().unwrap(),
        LatentStagedWindowKeys {
            ring_slots: 160,
            ..q
        }
        .numerical_contract_id()
        .unwrap()
    );
}
