//! Explicit numerical parameters of the shared per-op references, including a
//! DeepSeek-style composition built from small synthetic tensors and checked
//! against straightforward f64 math.

use kernel_core::codec::{SmallFloat, bf16_to_f32, decode_ue8m0_f64, f32_to_bf16};
use kernel_core::contract::{
    DType, RmsNormKind, RouterKind, ScaleMode, SigmoidMeanHyperConnectionQuery,
};
use kernel_core::hyper_connection_oracle::grouped_norm;
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
fn f32_pool_without_position_rounds_before_post_pool_rmsnorm() {
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
    let inverse = (rounded.iter().map(|v| v * v).sum::<f64>() / 512.0 + 1e-20_f32 as f64)
        .sqrt()
        .recip();
    for (&actual, &x) in norm.values.iter().zip(&rounded) {
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
    let before_boundary = RmsNorm::f64(1e-20_f32 as f64, NormOffset::Plain)
        .apply(&pooled.values, &weights, 512)
        .unwrap();
    assert_ne!(before_boundary.values, norm.values);
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

fn bf16(value: f64) -> f64 {
    f64::from(bf16_to_f32(f32_to_bf16(value as f32)))
}

fn bf16_exact(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|&v| f64::from(bf16_to_f32(f32_to_bf16(v as f32))))
        .collect()
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
    assert!(query.apply(&gate, &up, rows).is_err());
    assert!(query.oracle(&gate, &up, rows).is_err());
}
