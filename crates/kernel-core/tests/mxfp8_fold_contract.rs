use kernel_core::codec::{SmallFloat, bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{DType, NumericalContract, Validate, WeightFormat};
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmQuery, exact_bf16_values, folded_dot, oracle};
use kernel_core::precision::execution::BlockScaledExecution;
use kernel_core::precision::format::{FormatId, registry};
use kernel_core::precision::packed::{MatrixView, ScaleBytes};
use kernel_core::precision::quantizer::BlockQuantization;
use kernel_core::precision::{DecodeOrder, NibbleOrder};
use kernel_core::quantize::{QuantFormat, Quantized, ScaleFormat, Scales};

fn query(k: u32, groups: u32) -> Mxfp8FoldGemmQuery {
    Mxfp8FoldGemmQuery {
        m_capacity: 3,
        n: 32,
        k,
        groups,
        activation_group_stride: k + 32,
        activation_row_stride: groups * (k + 32) + 32,
        output: DType::F32,
        execution: BlockScaledExecution::ExactDecodedBf16Mma,
    }
}

fn source<'a>(codes: &'a [u8], scales: &'a [u8], rows: usize, k: usize) -> MatrixView<'a> {
    MatrixView {
        tensor: "weight",
        format: WeightFormat::Fp8Block32x32,
        rows,
        k,
        row_origin: 0,
        k_origin: 0,
        nibble_order: NibbleOrder::EvenLow,
        codes,
        local: Some(ScaleBytes {
            dtype: DType::Ue8M0,
            bytes: scales,
        }),
        global: None,
    }
}

#[test]
fn block32_repack_is_bitwise_lossless_with_partial_blocks_and_source_origins() {
    let (rows, k) = (65, 64);
    let codes: Vec<_> = (0..rows * k)
        .map(|i| {
            let code = i as u8;
            if code & 0x7f == 0x7f { 0x80 } else { code }
        })
        .collect();
    let scales = vec![118, 127, 132, 126, 133, 124, 137, 116, 128];
    let mut weight = source(&codes, &scales, rows, k);
    weight.row_origin = 31;
    weight.k_origin = 17;
    let plugin = registry::by_format(weight.format).unwrap();
    assert_eq!(plugin.id(), FormatId::Fp8Block32);
    assert_eq!(plugin.local_scale_groups(), (32, 32));
    let repacked = plugin.repack_mxfp8_rows(weight).unwrap();
    assert_eq!(repacked.codes, codes);
    assert_eq!(repacked.scales.len(), rows * 3);
    assert_eq!((repacked.row_origin, repacked.k_origin), (31, 17));
    let view = repacked.view("repacked");
    view.validate_bf16_operands(DecodeOrder::ExactScaledOperand)
        .unwrap();
    for row in 0..rows {
        for col in 0..k {
            assert_eq!(
                weight
                    .bf16_operand(row, col, DecodeOrder::ExactScaledOperand)
                    .unwrap(),
                view.bf16_operand(row, col, DecodeOrder::ExactScaledOperand)
                    .unwrap()
            );
        }
    }
    let mut bad = codes.clone();
    bad[0] = 0x7f;
    assert!(
        plugin
            .repack_mxfp8_rows(MatrixView {
                codes: &bad,
                ..weight
            })
            .is_err()
    );
    let bad_scales = [0xff; 9];
    assert!(
        plugin
            .repack_mxfp8_rows(MatrixView {
                local: Some(ScaleBytes {
                    dtype: DType::Ue8M0,
                    bytes: &bad_scales
                }),
                ..weight
            })
            .is_err()
    );
}

#[test]
fn block32_scale_declarations_keep_the_source_grid_and_ue8m0_dtype() {
    use kernel_core::precision::{
        CodeEncoding, GroupAxis, ScaleConvention, ScaleScope, ScaleSpec, TensorSpan, WeightLayout,
        WeightSpec,
    };
    let weight = WeightSpec {
        codes: TensorSpan {
            tensor: "weight".into(),
            dtype: DType::Fp8E4M3,
            shape: vec![65, 64],
            offset: 0,
            byte_length: 65 * 64,
        },
        logical_shape: vec![65, 64],
        format: WeightFormat::Fp8Block32x32,
        encoding: CodeEncoding::E4M3,
        layout: WeightLayout::row_major(),
        local_scale: Some(ScaleSpec {
            tensor: TensorSpan {
                tensor: "scale".into(),
                dtype: DType::Ue8M0,
                shape: vec![3, 3],
                offset: 0,
                byte_length: 9,
            },
            scope: ScaleScope::Block,
            convention: ScaleConvention::Multiply,
            groups: vec![
                GroupAxis {
                    axis: 0,
                    size: 32,
                    origin: 31,
                },
                GroupAxis {
                    axis: 1,
                    size: 32,
                    origin: 17,
                },
            ],
        }),
        global_scale: None,
        transforms: vec![],
    };
    weight.validate().unwrap();
    assert_eq!(
        registry::by_weight(&weight).unwrap().id(),
        FormatId::Fp8Block32
    );
    let mut bad = weight.clone();
    bad.local_scale.as_mut().unwrap().groups[0].size = 1;
    assert!(bad.validate().is_err());
    let mut bad = weight;
    bad.local_scale.as_mut().unwrap().tensor.dtype = DType::Fp8E4M3;
    assert!(bad.validate().is_err());
}

#[test]
fn grouped_projection_matches_bf16_weights_and_qdq_activations() {
    let query = query(160, 2);
    query.validate().unwrap();
    let (rows, n, k, groups) = (3, query.n as usize, query.k as usize, query.groups as usize);
    let activation: Vec<_> = (0..rows * query.activation_row_stride as usize)
        .map(|i| ((i * 23 % 173) as f32 - 86.0) / 19.0)
        .collect();
    let a = BlockQuantization::E4M3_CEIL32
        .apply(&activation, rows, query.activation_row_stride as usize)
        .unwrap();
    let codes: Vec<_> = (0..groups * n * k)
        .map(|i| {
            let value = ((i * 13 % 83) as f32 - 41.0) / 8.0;
            SmallFloat::E4M3.encode_f32(value)
        })
        .collect();
    let scales: Vec<_> = (0..groups * n / 32 * (k / 32))
        .map(|i| 122 + (i % 9) as u8)
        .collect();
    let original = source(&codes, &scales, groups * n, k);
    let packed = registry::by_format(original.format)
        .unwrap()
        .repack_mxfp8_rows(original)
        .unwrap();
    let b = Quantized::from_parts(
        groups * n,
        k,
        a.format,
        packed.codes,
        Scales::Ue8M0(packed.scales),
        1.0,
    )
    .unwrap();
    let result = oracle(query, rows, &a, &b).unwrap();
    let decoded_a = exact_bf16_values(&a).unwrap();
    let decoded_b = exact_bf16_values(&b).unwrap();
    let mut wrong_group_differs = false;
    for row in 0..rows {
        for group in 0..groups {
            let offset = row * a.k + group * query.activation_group_stride as usize;
            for col in 0..n {
                let index = (row * groups + group) * n + col;
                let mut expected = 0.0_f64;
                let mut wrong = 0.0_f64;
                let mut a_bf16 = Vec::new();
                let mut b_bf16 = Vec::new();
                for kk in 0..k {
                    // The conversion stores BF16 weights; QDQ stores the
                    // exact scaled activation back in BF16 before einsum.
                    let w = original
                        .bf16_operand(group * n + col, kk, DecodeOrder::ExactScaledOperand)
                        .unwrap();
                    let v = f32_to_bf16(decoded_a[offset + kk] as f32);
                    assert_eq!(
                        f64::from(bf16_to_f32(w)),
                        decoded_b[(group * n + col) * k + kk]
                    );
                    assert_eq!(f64::from(bf16_to_f32(v)), decoded_a[offset + kk]);
                    expected += f64::from(bf16_to_f32(v)) * f64::from(bf16_to_f32(w));
                    wrong += decoded_a[row * a.k + kk] * f64::from(bf16_to_f32(w));
                    a_bf16.push(bf16_to_f32(v));
                    b_bf16.push(bf16_to_f32(w));
                }
                assert_eq!(result.values[index], expected);
                wrong_group_differs |= wrong != expected;
                let folded = f64::from(folded_dot(&a_bf16, &b_bf16).unwrap());
                assert!(
                    (folded - expected).abs()
                        <= kernel_core::oracle::numeric_bound(&result, index, DType::F32)
                );
            }
        }
    }
    assert!(
        wrong_group_differs,
        "the fixture must detect a missing per-group activation offset"
    );
}

#[test]
fn fold_witness_resets_at_128_and_includes_the_final_64() {
    let mut a = vec![0.0; 576];
    let b = vec![1.0; 576];
    a[0] = 2_f32.powi(24);
    a[128..512].fill(1.0);
    a[512..].fill(2.0);
    let folded = folded_dot(&a, &b).unwrap();
    let single = a
        .iter()
        .zip(&b)
        .fold(0.0_f32, |sum, (&a, &b)| a.mul_add(b, sum));
    assert_eq!(folded, 2_f32.powi(24) + 512.0);
    assert_ne!(single, folded);
    assert_ne!(folded_dot(&a[..512], &b[..512]).unwrap(), folded);
    let mut q = query(576, 2);
    q.validate().unwrap();
    let decoded_id = q.numerical_contract_id().unwrap();
    q.execution = BlockScaledExecution::NativeBlockScaledMma;
    assert_ne!(q.numerical_contract_id().unwrap(), decoded_id);
    q.activation_row_stride -= 96;
    assert!(q.validate().is_err());
}

#[test]
fn exact_mxfp8_admission_keeps_extreme_scale_exclusions() {
    let format = QuantFormat {
        element: SmallFloat::E4M3,
        block: (1, 32),
        scale: ScaleFormat::Ue8M0,
        chunk_rows: None,
    };
    for (code, scale, valid) in [
        (0x01, 0, false),
        (0x01, 2, false),
        (0x01, 3, true),
        (0x7e, 246, true),
        (0x7e, 247, false),
        (0x7f, 127, false),
        (0xff, 127, false),
        (0, 255, false),
        (0x80, 254, true),
        (0x38, 254, true),
    ] {
        let q = Quantized::from_parts(
            1,
            32,
            format,
            vec![code; 32],
            Scales::Ue8M0(vec![scale]),
            1.0,
        )
        .unwrap();
        assert_eq!(
            exact_bf16_values(&q).is_ok(),
            valid,
            "code={code} scale={scale}"
        );
    }
}

#[test]
fn mxfp4_admission_keeps_the_scale_253_and_254_overflow_exclusions() {
    for scale in [0, 252, 253, 254, 255] {
        for code in 0..16_u8 {
            let codes = [code | (code << 4); 16];
            let scales = [scale];
            let view = MatrixView {
                format: WeightFormat::Mxfp4,
                ..source(&codes, &scales, 1, 32)
            };
            let valid = match scale {
                255 => false,
                254 => code & 7 < 4,
                253 => code & 7 < 6,
                _ => true,
            };
            assert_eq!(
                view.validate_bf16_operands(DecodeOrder::ExactScaledOperand)
                    .is_ok(),
                valid,
                "code={code} scale={scale}"
            );
        }
    }
}

#[test]
fn exact_scale_grid_admission_matches_each_decoded_operand_at_every_exponent() {
    for format in [
        WeightFormat::Mxfp4,
        WeightFormat::Mxfp8,
        WeightFormat::Fp8Block32x32,
    ] {
        let four_bits = format == WeightFormat::Mxfp4;
        for code in 0..if four_bits { 16 } else { 256 } {
            for scale in 0..=255_u8 {
                let code = code as u8;
                let codes = vec![
                    if four_bits { code | (code << 4) } else { code };
                    if four_bits { 16 } else { 32 }
                ];
                let scales = [scale];
                let view = MatrixView {
                    format,
                    ..source(&codes, &scales, 1, 32)
                };
                let expected = view
                    .bf16_operand(0, 0, DecodeOrder::ExactScaledOperand)
                    .is_ok();
                assert_eq!(
                    view.validate_bf16_operands(DecodeOrder::ExactScaledOperand)
                        .is_ok(),
                    expected,
                    "{format:?} code={code} scale={scale}"
                );
            }
        }
    }
}

#[test]
fn native_fold_payloads_are_admitted_before_whole_or_streamed_production() {
    use kernel_core::contract::SwigluColumns;
    use kernel_core::precision::WeightLayout;
    use kernel_core::precision::layout::PackingPlan;
    use kernel_core::weight_transform::{
        PackedMatrixSource, PackedOperand, PackedSection, SourceTensor, WeightRecord,
        WeightTransform,
    };
    let plan = |format, layout: &str, rows, k, local_scale_dtype| PackingPlan {
        layout: WeightLayout {
            id: layout.into(),
            version: 1,
        },
        format,
        experts: 1,
        projections: 1,
        rows,
        k,
        columns: SwigluColumns::ConcatenatedHalves,
        local_scale_dtype,
    };
    let format = WeightFormat::Mxfp8;
    for order in [
        DecodeOrder::NativeBlockScaledMma,
        DecodeOrder::ExactScaledOperand,
    ] {
        for (code, scale, valid) in [
            (0x01, 2, false),
            (0x01, 3, true),
            (0x7e, 246, true),
            (0x7e, 247, false),
            (0x7f, 127, false),
            (0xff, 127, false),
            (0, 255, false),
            (0x80, 254, true),
            (0x38, 254, true),
        ] {
            let sources = [
                vec![code; 64],
                vec![scale; if format == WeightFormat::Mxfp8 { 2 } else { 1 }],
            ];
            for section in [PackedSection::Codes, PackedSection::LocalScales] {
                let packed = PackedOperand {
                    local_encoding: None,
                    plan: plan(WeightFormat::Mxfp8, "row-major", 2, 32, Some(DType::Ue8M0)),
                    section,
                    decode_order: order,
                    matrices: vec![PackedMatrixSource {
                        expert: 0,
                        projection: 0,
                        format,
                        rows: 2,
                        k: 32,
                        row_origin: 0,
                        k_origin: 0,
                        nibble_order: NibbleOrder::EvenLow,
                        codes: 0,
                        local: Some(1),
                        local_dtype: Some(DType::Ue8M0),
                        global: None,
                        source_expert: 0,
                        source_experts: 1,
                        rows_kept: std::iter::once(0..2).collect(),
                        k_kept: 0..32,
                    }],
                };
                let record = WeightRecord {
                    transform: WeightTransform::PackedOperand(Box::new(packed.clone())),
                    sources: sources
                        .iter()
                        .map(|bytes| SourceTensor {
                            dtype: "U8".into(),
                            shape: vec![bytes.len() as u64],
                            byte_length: bytes.len() as u64,
                        })
                        .collect(),
                    axis: None,
                    ranges: vec![],
                    stored_shape: vec![],
                };
                let whole = packed.execute(
                    &record,
                    &sources.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                );
                assert_eq!(
                    whole.is_ok(),
                    valid,
                    "{format:?} {order:?} {section:?} {code} {scale}: {whole:?}"
                );
                if !valid && code & 127 != 127 && scale != 255 {
                    assert!(
                        whole
                            .as_ref()
                            .unwrap_err()
                            .to_string()
                            .contains("MXFP8 fold operands must be finite and exact in BF16")
                    );
                }
                let bytes = record.output_bytes().unwrap();
                let prepared = record.prepare(bytes).unwrap();
                let take = if section == PackedSection::Codes {
                    32
                } else {
                    1
                };
                let mut out = vec![0xa5; take];
                let streamed = prepared.produce(
                    0..take as u64,
                    &mut out,
                    &mut |index, at, dst| {
                        dst.copy_from_slice(&sources[index][at as usize..at as usize + dst.len()]);
                        Ok(())
                    },
                    &mut Vec::new(),
                );
                assert_eq!(
                    streamed.is_ok(),
                    valid,
                    "{format:?} {order:?} {section:?} {code} {scale}: {streamed:?}"
                );
                if valid {
                    assert_eq!(out, whole.unwrap()[..take]);
                } else {
                    assert_eq!(
                        out,
                        vec![0xa5; take],
                        "a refused producer must publish no bytes"
                    );
                }
            }
        }
    }
}
