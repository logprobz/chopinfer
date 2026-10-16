use kernel_core::codec::{SmallFloat, bf16_to_f32};
use kernel_core::contract::{ActivationFormat, DType, WeightFormat};
use kernel_core::precision::packed::{MatrixSlice, MatrixView, ScaleBytes};
use kernel_core::precision::{ActivationSpec, DecodeOrder, NibbleOrder, TensorSpan};
use kernel_core::precision::{
    CodeEncoding, GroupAxis, ScaleConvention, ScaleScope, ScaleSpec, WeightLayout, WeightSpec,
};

fn bank_spec() -> WeightSpec {
    WeightSpec {
        codes: TensorSpan {
            tensor: "bank.weight".into(),
            dtype: DType::U8,
            shape: vec![4, 128, 128],
            offset: 0,
            byte_length: 65536,
        },
        logical_shape: vec![4, 128, 256],
        format: WeightFormat::Nvfp4,
        encoding: CodeEncoding::E2M1(NibbleOrder::EvenLow),
        layout: WeightLayout::row_major(),
        local_scale: Some(ScaleSpec {
            tensor: TensorSpan {
                tensor: "bank.local".into(),
                dtype: DType::Fp8E4M3,
                shape: vec![4, 128, 16],
                offset: 0,
                byte_length: 8192,
            },
            scope: ScaleScope::Block,
            convention: ScaleConvention::Multiply,
            groups: vec![
                GroupAxis {
                    axis: 1,
                    size: 1,
                    origin: 0,
                },
                GroupAxis {
                    axis: 2,
                    size: 16,
                    origin: 0,
                },
            ],
        }),
        global_scale: Some(ScaleSpec {
            tensor: TensorSpan {
                tensor: "bank.global".into(),
                dtype: DType::F32,
                shape: vec![4],
                offset: 0,
                byte_length: 16,
            },
            scope: ScaleScope::Expert,
            convention: ScaleConvention::Multiply,
            groups: vec![],
        }),
        transforms: vec![],
    }
}

#[test]
fn uniform_bank_descriptors_require_exact_grouping_and_scoped_companions() {
    let source = bank_spec();
    source.validate().unwrap();
    let mut bad = source.clone();
    bad.global_scale.as_mut().unwrap().scope = ScaleScope::Tensor;
    assert!(bad.validate().is_err());
    let mut bad = source.clone();
    bad.local_scale.as_mut().unwrap().groups[1].size = 32;
    assert!(bad.validate().is_err());
    let mut bad = source.clone();
    bad.codes.shape[2] = 256;
    bad.codes.byte_length *= 2;
    assert!(bad.validate().is_err());
    let mut bad = source.clone();
    bad.local_scale.as_mut().unwrap().tensor.shape[2] = 15;
    bad.local_scale.as_mut().unwrap().tensor.byte_length = 7680;
    assert!(bad.validate().is_err());
    let mut bad = source;
    bad.global_scale = None;
    assert!(bad.validate().is_err());
}

#[cfg(feature = "precision-serde")]
#[test]
fn precision_wire_values_preserve_scopes_and_refuse_unknown_execution_fields() {
    let source = bank_spec();
    let bytes = serde_json::to_vec(&source).unwrap();
    let parsed: WeightSpec = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(source, parsed);
    parsed.validate().unwrap();
    let mut value = serde_json::to_value(&source).unwrap();
    value["activation_transform"] = serde_json::json!("undeclared");
    assert!(serde_json::from_value::<WeightSpec>(value).is_err());
}

fn nv<'a>(
    codes: &'a [u8],
    local: &'a [u8],
    global: &'a [u8],
    rows: usize,
    k: usize,
) -> MatrixView<'a> {
    MatrixView {
        tensor: "unrelated.experts.up.weight",
        format: WeightFormat::Nvfp4,
        rows,
        k,
        row_origin: 0,
        k_origin: 0,
        nibble_order: NibbleOrder::EvenLow,
        codes,
        local: Some(ScaleBytes {
            dtype: DType::Fp8E4M3,
            bytes: local,
        }),
        global: Some(ScaleBytes {
            dtype: DType::F32,
            bytes: global,
        }),
    }
}

#[test]
fn every_nvfp4_code_and_signed_zero_is_preserved() {
    let codes: Vec<u8> = (0..8).map(|i| (2 * i) | ((2 * i + 1) << 4)).collect();
    let global = 0.5_f32.to_le_bytes();
    let matrix = nv(&codes, &[0x40], &global, 1, 16); // local 2, global 0.5
    matrix.validate().unwrap();
    for col in 0..16 {
        let got = bf16_to_f32(
            matrix
                .bf16_operand(0, col, DecodeOrder::ScaleThenCode)
                .unwrap(),
        );
        assert_eq!(
            got.to_bits(),
            SmallFloat::E2M1.decode_f32(col as u8).to_bits()
        );
    }
    assert!(
        bf16_to_f32(
            matrix
                .bf16_operand(0, 8, DecodeOrder::CodeThenScale)
                .unwrap()
        )
        .is_sign_negative()
    );
}

#[test]
fn post_mma_and_native_scale_contracts_cannot_use_a_bf16_operand_oracle() {
    let global = 0.5_f32.to_le_bytes();
    let matrix = nv(&[0x22; 8], &[0x40], &global, 1, 16);
    for order in [
        DecodeOrder::BlockScaleAfterMma,
        DecodeOrder::ScaleProductAfterBlockMma,
        DecodeOrder::SuccessiveScalesAfterBlockMma,
        DecodeOrder::NativeBlockScaledMma,
        DecodeOrder::NativeBlockScaledMmaThenGlobals,
    ] {
        assert!(
            matrix
                .bf16_operand(0, 0, order)
                .unwrap_err()
                .to_string()
                .contains("no scaled BF16")
        );
        assert!(matrix.validate_bf16_operands(order).is_err());
    }
}

#[test]
fn e4m3_scale_validation_rejects_raw_sign_bits_including_negative_zero() {
    let global = 1.0_f32.to_le_bytes();
    for scale in 0..=255 {
        let scale = [scale];
        let result = nv(&[0x22; 8], &scale, &global, 1, 16).validate();
        assert_eq!(result.is_ok(), scale[0] < 0x7f, "scale {:02x}", scale[0]);
        if let Err(error) = result {
            assert!(
                error
                    .to_string()
                    .contains("unrelated.experts.up.weight: local scale")
            );
        }
    }
}

#[test]
fn ue8m0_nan_and_nonpositive_global_scales_are_refused() {
    for code in 0..=255 {
        let data = [code];
        let scale = ScaleBytes {
            dtype: DType::Ue8M0,
            bytes: &data,
        };
        assert_eq!(scale.validate(1, false).is_ok(), code != 255);
    }
    for global in [0.0_f32, -0.0, -1.0, f32::INFINITY, f32::NAN] {
        assert!(
            nv(&[0; 8], &[0], &global.to_le_bytes(), 1, 16)
                .validate()
                .is_err()
        );
    }
    // A zero local scale is valid, including an all-zero exported group.
    nv(&[0; 8], &[0], &1.0_f32.to_le_bytes(), 1, 16)
        .validate()
        .unwrap();
}

#[test]
fn dense_and_grouped_decode_orders_keep_their_existing_rounding_boundaries() {
    let global = 0x41ef1c73_u32.to_le_bytes();
    let matrix = nv(&[0x33; 8], &[0x03], &global, 1, 16);
    matrix.validate().unwrap();
    assert_eq!(
        matrix
            .bf16_operand(0, 0, DecodeOrder::ScaleThenCode)
            .unwrap(),
        0x3e86
    );
    assert_eq!(
        matrix
            .bf16_operand(0, 0, DecodeOrder::CodeThenScale)
            .unwrap(),
        0x3e87
    );
}

#[test]
fn bf16_operand_bounds_check_extreme_scales_without_rejecting_valid_zero_codes() {
    let global = 2.0e38_f32.to_le_bytes();
    for order in [DecodeOrder::ScaleThenCode, DecodeOrder::CodeThenScale] {
        assert!(
            nv(&[0x77; 8], &[0x38], &global, 1, 16)
                .validate_bf16_operands(order)
                .is_err()
        );
        nv(&[0; 8], &[0x38], &global, 1, 16)
            .validate_bf16_operands(order)
            .unwrap();
    }
    let max = f32::MAX.to_le_bytes();
    assert!(
        nv(&[0; 8], &[0x40], &max, 1, 16)
            .validate_bf16_operands(DecodeOrder::ScaleThenCode)
            .is_err()
    );
    nv(&[0; 8], &[0x40], &max, 1, 16)
        .validate_bf16_operands(DecodeOrder::CodeThenScale)
        .unwrap();
}

#[test]
fn lookup_scales_retain_their_declared_bf16_or_f32_value() {
    let bf16 = 0x3f80_u16.to_le_bytes();
    let f32 = f32::from_bits(0x3f800001).to_le_bytes();
    let lookup = |scale| MatrixView {
        tensor: "table",
        format: WeightFormat::Fp8PerTensor,
        rows: 1,
        k: 2,
        row_origin: 0,
        k_origin: 0,
        nibble_order: NibbleOrder::EvenLow,
        codes: &[0x38, 0x40],
        local: Some(scale),
        global: None,
    };
    let a = lookup(ScaleBytes {
        dtype: DType::Bf16,
        bytes: &bf16,
    });
    let b = lookup(ScaleBytes {
        dtype: DType::F32,
        bytes: &f32,
    });
    a.validate().unwrap();
    b.validate().unwrap();
    assert_ne!(
        a.local.unwrap().value(0).unwrap().to_bits(),
        b.local.unwrap().value(0).unwrap().to_bits()
    );
    assert_eq!(
        a.bf16_operand(0, 1, DecodeOrder::ScaleThenCode).unwrap(),
        0x4000
    );
}

#[test]
fn packed_slicing_preserves_codes_group_origins_distinct_scales_and_padding() {
    let codes: Vec<u8> = (0..96).map(|i| (i * 19) as u8).collect();
    let scales = [
        0x30, 0x38, 0x40, 0x48, 0x50, 0x58, 0x60, 0x68, 0x70, 0x71, 0x72, 0x73,
    ];
    let global = 0.003_f32.to_le_bytes();
    let source = nv(&codes, &scales, &global, 3, 64);
    let slice = MatrixSlice {
        rows: vec![2, 0],
        k_start: 7,
        k: 34,
        padded_rows: 3,
        padded_k: 48,
        nibble_order: NibbleOrder::EvenHigh,
    };
    let packed = source.repack(&slice).unwrap();
    assert_eq!(packed.k_origin, 7);
    assert_eq!(packed.codes.len(), 72);
    assert_eq!(packed.scales.len(), 12);
    assert_eq!(packed.global.as_deref(), Some(global.as_slice()));
    let view = packed.view("packed");
    view.validate().unwrap();
    for row in 0..3 {
        for k in 0..48 {
            let actual = view
                .bf16_operand(row, k, DecodeOrder::CodeThenScale)
                .unwrap();
            if row < 2 && k < 34 {
                assert_eq!(
                    actual,
                    source
                        .bf16_operand(slice.rows[row], 7 + k, DecodeOrder::CodeThenScale)
                        .unwrap()
                );
            } else {
                assert_eq!(actual, 0);
            }
        }
    }
    // Repacking back changes only the physical nibble order.
    let back = view
        .repack(&MatrixSlice {
            rows: vec![0, 1, 2],
            k_start: 0,
            k: 48,
            padded_rows: 3,
            padded_k: 48,
            nibble_order: NibbleOrder::EvenLow,
        })
        .unwrap();
    for (a, b) in packed.codes.iter().zip(&back.codes) {
        assert_eq!(a.rotate_left(4), *b);
    }
}

#[test]
fn mxfp8_slicing_retains_ue8m0_bytes_without_requantization() {
    let source = MatrixView {
        tensor: "small.matrix",
        format: WeightFormat::Mxfp8,
        rows: 2,
        k: 64,
        row_origin: 0,
        k_origin: 0,
        nibble_order: NibbleOrder::EvenLow,
        codes: &[0x38; 128],
        local: Some(ScaleBytes {
            dtype: DType::Ue8M0,
            bytes: &[125, 126, 127, 128],
        }),
        global: None,
    };
    let packed = source
        .repack(&MatrixSlice {
            rows: vec![1],
            k_start: 16,
            k: 48,
            padded_rows: 1,
            padded_k: 64,
            nibble_order: NibbleOrder::EvenLow,
        })
        .unwrap();
    assert_eq!(packed.scales, [127, 128, 127]);
    assert_eq!(packed.codes[..48], [0x38; 48]);
    assert_eq!(packed.codes[48..], [0; 16]);
    packed.view("packed").validate().unwrap();
}

#[test]
fn tensor_spans_accept_both_scalar_shapes_and_check_overflow() {
    for shape in [vec![], vec![1]] {
        TensorSpan {
            tensor: "global".into(),
            dtype: DType::F32,
            shape,
            offset: 0,
            byte_length: 4,
        }
        .validate()
        .unwrap();
    }
    for (shape, offset, byte_length) in [
        (vec![1], u64::MAX, 4),
        (vec![0], 0, 0),
        (vec![1], 0, 2),
        (vec![u64::MAX, 2], 0, 4),
    ] {
        assert!(
            TensorSpan {
                tensor: "bad".into(),
                dtype: DType::F32,
                shape,
                offset,
                byte_length
            }
            .validate()
            .is_err()
        );
    }
}

#[test]
fn bf16_input_pointer_does_not_claim_bf16_activation_compute() {
    let mut activation = ActivationSpec::bf16();
    assert!(activation.is_bf16());
    activation.compute = DType::Fp4E2M1;
    assert!(!activation.is_bf16());
    assert!(activation.validate("projection", &[1, 256]).is_err());
    activation.format = ActivationFormat::Nvfp4;
    assert!(
        activation.validate("projection", &[1, 256]).is_err(),
        "a quantizer and calibration are required"
    );
}

#[test]
fn truncated_scale_and_code_payloads_fail_without_indexing_past_the_end() {
    let global = 1.0_f32.to_le_bytes();
    for (codes, scales) in [(&[0_u8; 7][..], &[0_u8; 1][..]), (&[0; 8][..], &[][..])] {
        assert!(nv(codes, scales, &global, 1, 16).validate().is_err());
    }
    let matrix = nv(&[0; 8], &[0x38], &global, 1, 16);
    assert!(
        matrix
            .bf16_operand(1, 0, DecodeOrder::ScaleThenCode)
            .is_err()
    );
    assert!(
        matrix
            .bf16_operand(0, usize::MAX, DecodeOrder::ScaleThenCode)
            .is_err()
    );
}
