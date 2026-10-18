use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{
    ActivationFormat, DType, Epilogues, GemmQuery, NumericalContract, Validate, WeightFormat,
};
use kernel_core::precision::f32_gemm::{self, DotPath};
use kernel_core::precision::format::{
    A16Recipe, FormatId, KernelFamily, PaddingTile, SliceAxis, SliceRequest, registry,
};
use kernel_core::precision::packed::MatrixView;
use kernel_core::precision::*;

fn query(weight: WeightFormat, k: u32) -> GemmQuery {
    GemmQuery {
        m_capacity: 2,
        n: 1,
        k,
        weight,
        activation: ActivationFormat::F32,
        output: DType::F32,
        epilogues: Epilogues::default(),
        grouped: None,
    }
}

#[cfg(feature = "precision-serde")]
#[test]
fn f32_precision_documents_bind_only_exact_f32_arithmetic() {
    use kernel_core::contract::Target;
    use kernel_core::precision::capability::{
        ActivationCapability, CapabilityDocument, CapabilityInventory, GlobalScaleCapability,
        OperationCapability,
    };
    use kernel_core::precision::execution::ImplementationProvider;
    for weight in [WeightFormat::Bf16, WeightFormat::F32] {
        let q = query(weight, 3);
        let dtype = if weight == WeightFormat::F32 {
            DType::F32
        } else {
            DType::Bf16
        };
        let encoding = if weight == WeightFormat::F32 {
            CodeEncoding::F32LittleEndian
        } else {
            CodeEncoding::Bf16LittleEndian
        };
        let operation = OperationSpec {
            id: "f32_projection".into(),
            kind: OperationKind::Dense,
            target: Target::Sm120f,
            weights: vec![WeightSpec {
                codes: TensorSpan {
                    tensor: "weight".into(),
                    dtype,
                    shape: vec![1, 3],
                    offset: 0,
                    byte_length: if weight == WeightFormat::F32 { 12 } else { 6 },
                },
                logical_shape: vec![1, 3],
                format: weight,
                encoding,
                layout: WeightLayout::row_major(),
                local_scale: None,
                global_scale: None,
                transforms: vec![],
            }],
            geometry: None,
            activations: vec![ActivationSpec::f32()],
            arithmetic: ArithmeticSpec {
                weight_operand: DType::F32,
                accumulator: DType::F32,
                output: DType::F32,
                decode_order: DecodeOrder::ExactF32Operands,
                rounding: Rounding::NearestEven,
                numerical_contract: q.numerical_contract_id().unwrap().to_string(),
            },
            epilogues: Epilogues::default(),
            uses: vec![OperationUse {
                phase: ExecutionPhase::Decode,
                capacity: 2,
                graph: false,
            }],
        };
        operation.validate().unwrap();
        let parsed: OperationSpec =
            serde_json::from_slice(&serde_json::to_vec(&operation).unwrap()).unwrap();
        assert_eq!(parsed, operation);
        let cap = OperationCapability {
            kind: operation.kind,
            target: operation.target,
            usage: operation.uses[0],
            geometry: operation.matrix_geometry().unwrap(),
            operands: 1,
            weight,
            activation: Some(ActivationCapability::F32),
            epilogues: operation.epilogues,
            layout: WeightLayout::row_major(),
            local_scale_dtypes: vec![],
            globals: GlobalScaleCapability::Absent,
            shared_local_scale: false,
            arithmetic: operation.arithmetic.clone(),
            implementation: "exact-f32-fold128".into(),
            stages: vec!["exact_f32_v1".into()],
        };
        let doc = CapabilityDocument {
            version: 1,
            operations: vec![cap.clone()],
        };
        let parsed: CapabilityDocument =
            serde_json::from_slice(&serde_json::to_vec(&doc).unwrap()).unwrap();
        assert_eq!(parsed, doc);
        let inventory =
            CapabilityInventory::new(doc.clone(), doc.entries(), "0".repeat(64)).unwrap();
        inventory.bind(&operation, operation.uses[0]).unwrap();
        let absent = CapabilityInventory::new(doc, Default::default(), "0".repeat(64)).unwrap();
        assert!(absent.bind(&operation, operation.uses[0]).is_err());
        let mut narrowed = operation.clone();
        narrowed.activations[0] = ActivationSpec::bf16();
        assert!(narrowed.validate().is_err());
        assert!(inventory.bind(&narrowed, narrowed.uses[0]).is_err());
        let mut rounded = cap;
        rounded.arithmetic.output = DType::Bf16;
        assert!(rounded.validate().is_err());
    }
}

#[test]
fn f32_activation_contract_has_no_quantizer_or_bf16_rounding() {
    let activation = ActivationSpec::f32();
    activation.validate("input", &[1, 257]).unwrap();
    assert!(activation.is_f32());
    for dtype in [DType::Bf16, DType::F16, DType::Fp8E4M3] {
        let mut bad = activation.clone();
        bad.storage = dtype;
        assert!(bad.validate("input", &[1, 257]).is_err());
        bad.storage = DType::F32;
        bad.compute = dtype;
        assert!(bad.validate("input", &[1, 257]).is_err());
    }
    let mut bad = activation;
    bad.quantizer = ActivationSpec::mxfp8_ceil(1e-4_f32.to_bits()).quantizer;
    assert!(bad.validate("input", &[1, 257]).is_err());
    for weight in [WeightFormat::Bf16, WeightFormat::F32] {
        let q = query(weight, 257);
        q.validate().unwrap();
        assert!(
            q.numerical_contract_id()
                .unwrap()
                .as_str()
                .contains("exact-input.fold128")
        );
        let mut bad = q;
        bad.output = DType::Bf16;
        assert!(bad.validate().is_err());
        bad = q;
        bad.epilogues.bias = true;
        assert!(bad.validate().is_err());
    }
    assert!(query(WeightFormat::Mxfp8, 256).validate().is_err());
    let mut bad = query(WeightFormat::F32, 256);
    bad.activation = ActivationFormat::Bf16;
    assert!(bad.validate().is_err());
}

#[test]
fn raw_f32_weights_pack_and_slice_without_losing_low_bits() {
    let values = [1.0039_f32, -0.0, f32::from_bits(1), f32::MAX, -3.1401, 0.0];
    let codes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let view = MatrixView {
        tensor: "weight",
        format: WeightFormat::F32,
        rows: 2,
        k: 3,
        row_origin: 0,
        k_origin: 0,
        nibble_order: NibbleOrder::EvenLow,
        codes: &codes,
        local: None,
        global: None,
    };
    view.validate().unwrap();
    for (i, &value) in values.iter().enumerate() {
        assert_eq!(
            view.f32_operand(i / 3, i % 3).unwrap().to_bits(),
            value.to_bits()
        );
        assert_eq!(
            view.oracle_weight(i / 3, i % 3, DecodeOrder::ExactF32Operands)
                .unwrap()
                .to_bits(),
            f64::from(value).to_bits()
        );
        assert!(
            view.bf16_operand(i / 3, i % 3, DecodeOrder::ScaleThenCode)
                .is_err()
        );
    }
    assert!(
        view.validate_bf16_operands(DecodeOrder::ScaleThenCode)
            .is_err()
    );
    let plugin = registry::by_id(FormatId::F32).unwrap();
    let plan = kernel_core::precision::layout::PackingPlan {
        layout: WeightLayout::row_major(),
        format: WeightFormat::F32,
        experts: 1,
        projections: 1,
        rows: 2,
        k: 3,
        columns: kernel_core::contract::SwigluColumns::ConcatenatedHalves,
        local_scale_dtype: None,
    };
    plan.validate().unwrap();
    assert_eq!(plan.resident_bytes().unwrap(), 24);
    assert_eq!(
        plugin.layout(KernelFamily::Dense, None).unwrap().codes,
        "f32_row_major_v1"
    );
    assert_eq!(
        plugin
            .decode_order(
                KernelFamily::Dense,
                ActivationFormat::F32,
                A16Recipe::ScaledWeight
            )
            .unwrap(),
        DecodeOrder::ExactF32Operands
    );
    assert!(
        plugin
            .decode_order(
                KernelFamily::Dense,
                ActivationFormat::Bf16,
                A16Recipe::ScaledWeight
            )
            .is_err()
    );
    let packed = plugin
        .slice(
            view,
            &SliceRequest {
                axis: SliceAxis::Rows,
                range: 1..2,
                padding: PaddingTile::NONE,
                nibble_order: NibbleOrder::EvenLow,
            },
        )
        .unwrap();
    assert_eq!(packed.codes, codes[12..]);
    assert!(packed.view("slice").local.is_none());
    packed.view("slice").validate().unwrap();
    let mut invalid = codes.clone();
    invalid[..4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(
        MatrixView {
            codes: &invalid,
            ..view
        }
        .validate()
        .is_err()
    );
}

#[test]
fn bf16_components_reconstruct_f32_bits_or_require_simt() {
    let mut expanded = 0;
    let mut fallback = 0;
    for exponent in 0..255 {
        for mantissa in [0, 1, 0xffff, 0x123456, 0x400001, 0x7fffff] {
            for sign in [0, 0x80000000] {
                let value = f32::from_bits(sign | (exponent << 23) | mantissa);
                match f32_gemm::decompose(value).unwrap() {
                    Some(bits) => {
                        let terms = bits.map(bf16_to_f32);
                        assert!(terms.iter().all(|v| *v == 0.0 || v.is_normal()));
                        assert!(
                            terms
                                .iter()
                                .all(|v| v.is_sign_negative() == value.is_sign_negative())
                        );
                        let [hi, mid, lo] = terms.map(f64::from);
                        assert_eq!(((hi + mid) + lo).to_bits(), f64::from(value).to_bits());
                        expanded += 1;
                    }
                    None => fallback += 1,
                }
            }
        }
    }
    assert!(expanded > 1000 && fallback > 0);
    assert!(f32_gemm::decompose(f32::MAX).unwrap().is_some());
    assert!(f32_gemm::decompose(f32::from_bits(1)).unwrap().is_none());
    let terms = f32_gemm::decompose(f32::from_bits(0x3f812345))
        .unwrap()
        .unwrap();
    assert!(terms.iter().all(|&v| bf16_to_f32(v) != 0.0));
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(f32_gemm::decompose(value).is_err());
    }
}

#[test]
fn f32_oracle_rejects_bf16_input_rounding_and_pins_the_fold_bound() {
    let a = [1.0039_f32; 257];
    let b = [1.0_f32; 257];
    let q = query(WeightFormat::Bf16, 257);
    let reference = f32_gemm::oracle(q, 1, &a, &b).unwrap();
    assert_eq!(reference.values, [257.0 * f64::from(a[0])]);
    assert_eq!(reference.k, 257);
    assert!(
        f32_gemm::bound(reference.k, reference.sum_abs_products[0])
            < 2_f64.powi(-22) * 128.0 * reference.sum_abs_products[0]
    );
    assert_eq!(
        f32_gemm::dot_path(&a, &b, DType::Bf16).unwrap(),
        DotPath::Bf16Expansion3
    );
    let actual = f32_gemm::folded_dot(&a, &b, DType::Bf16).unwrap();
    f32_gemm::check_numeric(&[f64::from(actual)], &reference).unwrap();
    let rounded: Vec<_> = a.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect();
    let incorrect = f32_gemm::folded_dot(&rounded, &b, DType::Bf16).unwrap();
    assert!(f32_gemm::check_numeric(&[f64::from(incorrect)], &reference).is_err());
}

#[test]
fn finite_domain_exceptions_and_raw_f32_weights_use_simt_without_rounding() {
    let a = [f32::from_bits(0x00010001)];
    let b = [2_f32.powi(100)];
    assert_eq!(
        f32_gemm::dot_path(&a, &b, DType::Bf16).unwrap(),
        DotPath::SimtF32
    );
    let actual = f32_gemm::folded_dot(&a, &b, DType::Bf16).unwrap();
    assert_eq!(actual.to_bits(), a[0].mul_add(b[0], 0.0).to_bits());
    let reference = f32_gemm::oracle(query(WeightFormat::Bf16, 1), 1, &a, &b).unwrap();
    f32_gemm::check_numeric(&[f64::from(actual)], &reference).unwrap();
    assert_eq!(
        f32_gemm::dot_path(&[1.0], &[1.0039], DType::F32).unwrap(),
        DotPath::SimtF32
    );
    assert!(f32_gemm::dot_path(&[1.0], &[1.0039], DType::Bf16).is_err());
    assert_eq!(
        f32_gemm::folded_dot(&[1.0], &[1.0039], DType::F32).unwrap(),
        1.0039
    );
    assert_eq!(
        f32_gemm::dot_path(&[f32::MAX], &[2.0], DType::Bf16).unwrap(),
        DotPath::SimtF32
    );
    assert_eq!(
        f32_gemm::dot_path(&[2_f32.powi(-80)], &[2_f32.powi(-80)], DType::Bf16).unwrap(),
        DotPath::SimtF32
    );
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(f32_gemm::dot_path(&[bad], &[1.0], DType::Bf16).is_err());
        assert!(f32_gemm::dot_path(&[1.0], &[bad], DType::F32).is_err());
    }
    assert!(f32_gemm::oracle(query(WeightFormat::Bf16, 1), 1, &[1.0], &[1.0039]).is_err());
}

fn outer_rounding_witness(weight: WeightFormat, dtype: DType) {
    let k = 65_792;
    let mut a = vec![0.0_f32; k];
    a[0] = 1.0;
    for j in 1..=513 {
        a[128 * j] = 2_f32.powi(-24);
    }
    let b = vec![1.0; k];
    let actual = f32_gemm::folded_dot(&a, &b, dtype).unwrap();
    assert_eq!(actual, 1.0);
    let reference = f32_gemm::oracle(query(weight, k as u32), 1, &a, &b).unwrap();
    assert_eq!(reference.values[0], 1.0 + 513.0 * 2_f64.powi(-24));
    assert_eq!(reference.values, reference.sum_abs_products);
    assert!(
        (reference.values[0] - f64::from(actual)).abs()
            > 2_f64.powi(-22) * 128.0 * reference.sum_abs_products[0]
    );
    f32_gemm::check_numeric(&[f64::from(actual)], &reference).unwrap();
}

#[test]
fn outer_fold_rounding_is_covered_for_bf16_weights() {
    outer_rounding_witness(WeightFormat::Bf16, DType::Bf16);
}

#[test]
fn outer_fold_rounding_is_covered_for_f32_weights() {
    outer_rounding_witness(WeightFormat::F32, DType::F32);
}

#[test]
fn simt_underflow_passes_the_numerical_gate() {
    for (weight, dtype) in [
        (WeightFormat::Bf16, DType::Bf16),
        (WeightFormat::F32, DType::F32),
    ] {
        for (a, b) in [
            (2_f32.powi(-80), 2_f32.powi(-80)),
            (2_f32.powi(-75), 2_f32.powi(-75)),
            (2_f32.powi(-74), 2_f32.powi(-75)),
            (f32::from_bits(1), 1.0),
        ] {
            assert_eq!(
                f32_gemm::dot_path(&[a], &[b], dtype).unwrap(),
                DotPath::SimtF32
            );
            let actual = f32_gemm::folded_dot(&[a], &[b], dtype).unwrap();
            let exact = f64::from(a) * f64::from(b);
            assert_eq!(actual.to_bits(), (exact as f32).to_bits());
            let reference = f32_gemm::oracle(query(weight, 1), 1, &[a], &[b]).unwrap();
            assert_eq!(reference.values, [exact]);
            f32_gemm::check_numeric(&[f64::from(actual)], &reference).unwrap();
            let wrong = f64::from(actual) + 8.0 * 2_f64.powi(-149);
            assert!(f32_gemm::check_numeric(&[wrong], &reference).is_err());
        }
    }
}

#[test]
fn derived_bound_is_tight_at_short_k_and_finite_through_u32_k() {
    let u = f32_gemm::UNIT_ROUNDOFF;
    for (k, path_length) in [(1, 2), (128, 129), (4096, 160), (65_792, 642)] {
        let direct_product = ((path_length as f64) * u.ln_1p()).exp_m1();
        let allowance = f32_gemm::bound(k, 1.0);
        assert!(allowance >= direct_product);
        let nu = path_length as f64 * u;
        // The outward product remains tighter than the conventional gamma
        // estimate at these lengths, without an arbitrary F64 tolerance.
        assert!(allowance < nu / (1.0 - nu));
    }
    assert!(f32_gemm::bound(4096, 1.0) < 2_f64.powi(-22) * 128.0);
    assert!(f32_gemm::bound(u32::MAX as usize, 1.0).is_finite());
    assert!(f32_gemm::bound(0, 1.0).is_nan());
    assert!(f32_gemm::bound(1, -1.0).is_nan());
    let reference = f32_gemm::oracle(query(WeightFormat::F32, 1), 1, &[1.0], &[1.0]).unwrap();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(f32_gemm::check_numeric(&[bad], &reference).is_err());
    }
    assert!(f32_gemm::check_numeric(&[], &reference).is_err());
    let mut invalid = reference;
    invalid.k = 0;
    assert!(f32_gemm::check_numeric(&[1.0], &invalid).is_err());
}
