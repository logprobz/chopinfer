//! Explicit block-scale arithmetic and trained BF16 quantizer inputs.

use kernel_core::codec::{SmallFloat, bf16_to_f32, decode_ue8m0_f64, f32_to_bf16};
use kernel_core::contract::{DType, NumericalContract, WeightFormat};
use kernel_core::precision::quantizer::{BlockQuantization, BlockQuantizeQuery, QuantizeOutput};
use kernel_core::precision::{ActivationSpec, Calibration, DynamicScale};
use kernel_core::quantize::{Quantizer, Scales, quantize};

fn rounded(value: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(value))
}

// Independent scalar construction: F32 reciprocal and product, then a wide
// logarithm. Adjacent F32 ratios are far from the F64 logarithm's rounding.
fn ceil_scale(amax: f32, floor: f32, maximum: f32) -> f64 {
    let reciprocal = (1.0_f64 / f64::from(maximum)) as f32;
    let ratio = (f64::from(amax.max(floor)) * f64::from(reciprocal)) as f32;
    2_f64.powf(f64::from(ratio).log2().ceil())
}

fn scale(q: &kernel_core::quantize::Quantized) -> f64 {
    match &q.scales {
        Scales::Ue8M0(s) => decode_ue8m0_f64(s[0]),
        Scales::Ue4M3(s) => SmallFloat::Ue4M3.decode_f64(s[0]),
        Scales::F32(_) => panic!("expected scale bytes"),
    }
}

#[test]
fn adjacent_values_pin_reciprocal_rounding_and_bf16_before_amax() {
    let format = BlockQuantization::E4M3_CEIL32;
    for exponent in -20..=100 {
        let boundary = 448.0_f32 * 2_f32.powi(exponent);
        for bits in [
            boundary.to_bits() - 1,
            boundary.to_bits(),
            boundary.to_bits() + 1,
        ] {
            let value = f32::from_bits(bits);
            let mut input = [0.0; 32];
            input[0] = value;
            input[1] = -value;
            let raw = quantize(&input, 1, 32, format.quantizer().unwrap()).unwrap();
            assert_eq!(scale(&raw), ceil_scale(value, 1e-4, 448.0));
            let trained = format.apply(&input, 1, 32).unwrap();
            assert_eq!(scale(&trained), ceil_scale(rounded(value), 1e-4, 448.0));
            assert_eq!(trained.codes[0], 0x7e);
            assert_eq!(trained.codes[1], 0xfe);
        }
        // BF16 neighbors survive the required first rounding.
        for bits in [boundary.to_bits() - 0x10000, boundary.to_bits() + 0x10000] {
            let value = f32::from_bits(bits);
            let q = format.apply(&[value; 32], 1, 32).unwrap();
            assert_eq!(scale(&q), ceil_scale(value, 1e-4, 448.0));
        }
    }
    let x = [448.03125; 32];
    assert_eq!(
        scale(&quantize(&x, 1, 32, format.quantizer().unwrap()).unwrap()),
        2.0
    );
    assert_eq!(scale(&format.apply(&x, 1, 32).unwrap()), 1.0);
    let x = [500.0; 32];
    assert_ne!(
        format.apply(&x, 1, 32).unwrap(),
        quantize(&x, 1, 32, Quantizer::mx(SmallFloat::E4M3)).unwrap()
    );
}

#[test]
fn fp4_floors_and_fake_quant_match_the_scalar_rules() {
    for (format, floor) in [
        (BlockQuantization::E2M1E4M3Scale16, 6.0 * 2_f32.powi(-9)),
        (BlockQuantization::E2M1Ceil32, 6.0 * f32::MIN_POSITIVE),
    ] {
        for value in [
            0.0,
            f32::from_bits(1),
            f32::from_bits(floor.to_bits() - 1),
            floor,
            f32::from_bits(floor.to_bits() + 1),
            f32::from_bits(floor.to_bits() - 0x10000),
            f32::from_bits(floor.to_bits() + 0x10000),
        ] {
            let width = format.group_size() as usize;
            let mut x = vec![0.0; width];
            x[0] = value;
            x[1] = -value;
            let q = format.apply(&x, 1, width).unwrap();
            let maximum = rounded(value).abs().max(floor);
            let expected = if format == BlockQuantization::E2M1E4M3Scale16 {
                SmallFloat::E4M3.decode_f64(SmallFloat::E4M3.encode_f32(maximum / 6.0))
            } else {
                ceil_scale(maximum, floor, 6.0)
            };
            assert_eq!(scale(&q), expected);
            for (col, bits) in format
                .fake_quant(&x, 1, width)
                .unwrap()
                .into_iter()
                .enumerate()
            {
                assert_eq!(
                    f64::from(bf16_to_f32(bits)).to_bits(),
                    q.decode(0, col).to_bits()
                );
            }
        }
    }
    let format = BlockQuantization::E4M3_CEIL32;
    for value in [
        0.0,
        f32::from_bits(1e-4_f32.to_bits() - 1),
        1e-4,
        f32::from_bits(1e-4_f32.to_bits() + 1),
    ] {
        let q = format.apply(&[value; 32], 1, 32).unwrap();
        assert_eq!(scale(&q), 2_f64.powi(-22));
    }
}

#[test]
fn trained_quantizers_preserve_signed_zero_and_subnormal_codes_and_refuse_nan() {
    for format in [
        BlockQuantization::E4M3_CEIL32,
        BlockQuantization::E2M1E4M3Scale16,
        BlockQuantization::E2M1Ceil32,
    ] {
        let width = format.group_size() as usize;
        let q = format.apply(&vec![-0.0; width], 1, width).unwrap();
        let expected = if format == BlockQuantization::E4M3_CEIL32 {
            0x80
        } else {
            0x88
        };
        assert!(q.codes.iter().all(|&code| code == expected));
        assert!(
            format
                .fake_quant(&vec![-0.0; width], 1, width)
                .unwrap()
                .iter()
                .all(|&v| v == 0x8000)
        );
        for nonfinite in [
            f32::NAN,
            f32::from_bits(0xff80_0001),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
        ] {
            assert!(format.apply(&vec![nonfinite; width], 1, width).is_err());
        }
        assert!(format.apply(&vec![0.0; width - 1], 1, width - 1).is_err());
    }
    let mut x = [0.0; 32];
    x[0] = 1.0;
    x[1] = 2_f32.powi(-17);
    x[2] = -x[1];
    let q = BlockQuantization::E4M3_CEIL32.apply(&x, 1, 32).unwrap();
    assert_eq!((q.codes[1], q.codes[2]), (0x01, 0x81));
    assert_eq!(q.decode(0, 1), f64::from(x[1]));
    let q = BlockQuantization::E2M1E4M3Scale16
        .apply(&[2_f32.powi(-10); 16], 1, 16)
        .unwrap();
    assert_eq!(q.scales, Scales::Ue4M3(vec![1]));
    assert_eq!(q.codes, vec![0x11; 8]);
}

#[test]
fn ceil_declarations_require_their_floor_and_keep_distinct_ids() {
    let valid = ActivationSpec::mxfp8_ceil(1e-4_f32.to_bits());
    valid.validate("x", &[3, 64]).unwrap();
    for floor in [
        0.0_f32,
        -0.0,
        -1.0,
        f32::MIN_POSITIVE,
        f32::NAN,
        f32::INFINITY,
    ] {
        assert!(
            ActivationSpec::mxfp8_ceil(floor.to_bits())
                .validate("x", &[3, 64])
                .is_err()
        );
    }
    let query = BlockQuantizeQuery {
        width: 32,
        input: DType::F32,
        format: BlockQuantization::E4M3_CEIL32,
        output: QuantizeOutput::Codes,
    };
    assert_ne!(
        query.numerical_contract_id().unwrap(),
        BlockQuantizeQuery {
            output: QuantizeOutput::Bf16,
            ..query
        }
        .numerical_contract_id()
        .unwrap()
    );
    let mut bad = valid;
    bad.quantizer.as_mut().unwrap().calibration =
        Calibration::Dynamic(DynamicScale::Nvfp4TensorAmax);
    assert!(bad.validate("x", &[3, 64]).is_err());
}

#[test]
fn explicit_execution_claims_refuse_false_native_mma() {
    use kernel_core::contract::ActivationFormat;
    use kernel_core::precision::execution::BlockScaledExecution as Execution;
    use kernel_core::precision::format::{A16Recipe, FormatId, KernelFamily, registry};
    use kernel_core::precision::{ArithmeticSpec, DecodeOrder, Rounding};

    for format in [FormatId::Fp8Block32, FormatId::Mxfp8, FormatId::Mxfp4] {
        let plugin = registry::by_id(format).unwrap();
        for path in [
            Execution::NativeBlockScaledMma,
            Execution::ExactDecodedBf16Mma,
        ] {
            let order = plugin
                .decode_order_for_execution(KernelFamily::Dense, ActivationFormat::Mxfp8, path)
                .unwrap();
            assert_eq!(order, path.decode_order());
        }
        // Legacy format-based selection remains byte-compatible.
        assert_eq!(
            plugin
                .decode_order(
                    KernelFamily::Dense,
                    ActivationFormat::Mxfp8,
                    A16Recipe::ScaledWeight
                )
                .unwrap(),
            DecodeOrder::NativeBlockScaledMma
        );
    }
    let arithmetic = ArithmeticSpec {
        weight_operand: DType::Bf16,
        accumulator: DType::F32,
        output: DType::F32,
        decode_order: DecodeOrder::ExactScaledOperand,
        rounding: Rounding::NearestEven,
        numerical_contract: "gemm.fold128.v1".into(),
    };
    Execution::ExactDecodedBf16Mma
        .require(WeightFormat::Mxfp8, &arithmetic)
        .unwrap();
    assert!(
        Execution::NativeBlockScaledMma
            .require(WeightFormat::Mxfp8, &arithmetic)
            .is_err()
    );
    let false_native = ArithmeticSpec {
        decode_order: DecodeOrder::NativeBlockScaledMma,
        ..arithmetic
    };
    assert!(
        Execution::NativeBlockScaledMma
            .require(WeightFormat::Mxfp8, &false_native)
            .is_err()
    );
}

#[test]
fn binding_refuses_an_ocp_substitution_or_another_execution_claim() {
    use kernel_core::contract::{Epilogues, Target};
    use kernel_core::precision::capability::{
        ActivationCapability, CapabilityDocument, CapabilityInventory, GlobalScaleCapability,
        OperationCapability,
    };
    use kernel_core::precision::execution::{
        ActivationPolicy, ImplementationProvider, PrecisionSelection,
    };
    use kernel_core::precision::*;

    let usage = OperationUse {
        phase: ExecutionPhase::Prefill,
        capacity: 8,
        graph: false,
    };
    let operation = OperationSpec {
        id: "projection".into(),
        kind: OperationKind::Dense,
        target: Target::Sm120f,
        weights: vec![WeightSpec {
            codes: TensorSpan {
                tensor: "weight".into(),
                dtype: DType::Fp8E4M3,
                shape: vec![1, 32],
                offset: 0,
                byte_length: 32,
            },
            logical_shape: vec![1, 32],
            format: WeightFormat::Mxfp8,
            encoding: CodeEncoding::E4M3,
            layout: WeightLayout::row_major(),
            local_scale: Some(ScaleSpec {
                tensor: TensorSpan {
                    tensor: "scale".into(),
                    dtype: DType::Ue8M0,
                    shape: vec![1, 1],
                    offset: 0,
                    byte_length: 1,
                },
                scope: ScaleScope::Block,
                convention: ScaleConvention::Multiply,
                groups: vec![
                    GroupAxis {
                        axis: 0,
                        size: 1,
                        origin: 0,
                    },
                    GroupAxis {
                        axis: 1,
                        size: 32,
                        origin: 0,
                    },
                ],
            }),
            global_scale: None,
            transforms: vec![],
        }],
        geometry: None,
        activations: vec![ActivationSpec::mxfp8_ceil(1e-4_f32.to_bits())],
        arithmetic: ArithmeticSpec {
            weight_operand: DType::Bf16,
            accumulator: DType::F32,
            output: DType::F32,
            decode_order: DecodeOrder::ExactScaledOperand,
            rounding: Rounding::NearestEven,
            numerical_contract: "fold128.v1".into(),
        },
        epilogues: Epilogues::default(),
        uses: vec![usage],
    };
    operation.validate().unwrap();
    let capability = OperationCapability {
        kind: operation.kind,
        target: operation.target,
        usage,
        geometry: MatrixGeometry { rows: 1, k: 32 },
        operands: 1,
        weight: WeightFormat::Mxfp8,
        activation: Some(ActivationCapability::Mxfp8Ceil {
            amax_floor_bits: 1e-4_f32.to_bits(),
        }),
        epilogues: operation.epilogues,
        layout: WeightLayout::row_major(),
        local_scale_dtypes: vec![DType::Ue8M0],
        globals: GlobalScaleCapability::Absent,
        shared_local_scale: false,
        arithmetic: operation.arithmetic.clone(),
        implementation: "fold128".into(),
        stages: vec!["ceil_quantize".into(), "fold_gemm".into()],
    };
    let inventory = |capability: OperationCapability| {
        let entries = capability.stages.iter().cloned().collect();
        CapabilityInventory::new(
            CapabilityDocument {
                version: 1,
                operations: vec![capability],
            },
            entries,
            "a".repeat(64),
        )
        .unwrap()
    };
    let supplied = inventory(capability.clone());
    let binding = supplied.bind(&operation, usage).unwrap();
    let selection = PrecisionSelection {
        source: operation.clone(),
        effective: operation.clone(),
        policy: ActivationPolicy::Checkpoint,
        reuse: None,
        fallback_reasons: vec![],
        bindings: vec![execution::BoundUse {
            usage,
            implementation: binding.identity,
        }],
    };
    selection.validate().unwrap();
    let mut ocp = capability.clone();
    ocp.activation = Some(ActivationCapability::Mxfp8);
    assert!(inventory(ocp).bind(&operation, usage).is_err());
    let mut another_floor = capability;
    another_floor.activation = Some(ActivationCapability::Mxfp8Ceil {
        amax_floor_bits: 2e-4_f32.to_bits(),
    });
    assert!(inventory(another_floor).bind(&operation, usage).is_err());
    let mut false_native = operation;
    false_native.arithmetic.decode_order = DecodeOrder::NativeBlockScaledMma;
    false_native.arithmetic.weight_operand = DType::Fp8E4M3;
    assert!(supplied.bind(&false_native, usage).is_err());
    let mut substituted = selection;
    substituted.bindings[0].implementation.activations[0]
        .quantizer
        .as_mut()
        .unwrap()
        .calibration = Calibration::Dynamic(DynamicScale::MxPowerOfTwo);
    assert!(substituted.validate().is_err());
}

#[cfg(feature = "precision-serde")]
#[test]
fn ceil_scale_rule_is_describable_without_changing_the_ocp_rule() {
    use kernel_core::precision::DynamicScale;

    let document = format!(
        r#"{{"CeilPowerOfTwo":{{"amax_floor_bits":{}}}}}"#,
        1e-4_f32.to_bits()
    );
    let rule: DynamicScale = serde_json::from_str(&document).unwrap();
    assert_ne!(rule, DynamicScale::MxPowerOfTwo);
    assert_eq!(serde_json::to_string(&rule).unwrap(), document);
    assert_eq!(
        serde_json::to_string(&DynamicScale::MxPowerOfTwo).unwrap(),
        "\"MxPowerOfTwo\""
    );
    let malformed = serde_json::json!({"CeilPowerOfTwo": {
        "amax_floor_bits": 1e-4_f32.to_bits(), "input_rounding": "F32"
    }});
    assert!(serde_json::from_value::<DynamicScale>(malformed).is_err());
    let malformed = serde_json::json!({"Mxfp8Ceil": {
        "amax_floor_bits": 1e-4_f32.to_bits(), "scale_rule": "OcpFloor"
    }});
    assert!(
        serde_json::from_value::<kernel_core::precision::capability::ActivationCapability>(
            malformed
        )
        .is_err()
    );
}
