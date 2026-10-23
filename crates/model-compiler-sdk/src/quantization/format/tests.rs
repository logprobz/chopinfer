#![allow(clippy::single_range_in_vec_init)]
use super::*;
use crate::quantization::packing::PackedWeightBuilder;
use kernel_core::contract::{ActivationFormat, ScaleMode, SwigluColumns, WeightFormat};
use kernel_core::precision::format::{A16Recipe, CompanionRole, KernelFamily, registry};
use kernel_core::precision::packed::{MatrixSlice, MatrixView, ScaleBytes};
use kernel_core::precision::{
    CodeEncoding, GroupAxis, NibbleOrder, ScaleConvention, ScaleScope, ScaleSpec, TensorSpan,
    WeightLayout, WeightSpec,
};

fn fixture(id: FormatId) -> (WeightSpec, Vec<Vec<u8>>) {
    let (format, dtype, groups) = match id {
        FormatId::F32 => (WeightFormat::F32, None, (0, 0)),
        FormatId::Bf16 => (WeightFormat::Bf16, None, (0, 0)),
        FormatId::Fp8PerTensor => (WeightFormat::Fp8PerTensor, Some(DType::Bf16), (0, 0)),
        FormatId::Fp8PerChannel => (WeightFormat::Fp8PerChannel, Some(DType::F32), (1, 0)),
        FormatId::Fp8Block128 => (
            WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
            Some(DType::Bf16),
            (128, 128),
        ),
        FormatId::Fp8Rows64 => (
            WeightFormat::Fp8Rows64x128(ScaleMode::ExactF32),
            Some(DType::F32),
            (64, 128),
        ),
        FormatId::Mxfp8 => (WeightFormat::Mxfp8, Some(DType::Ue8M0), (1, 32)),
        FormatId::Fp8Block32 => (WeightFormat::Fp8Block32x32, Some(DType::Ue8M0), (32, 32)),
        FormatId::Mxfp4 => (WeightFormat::Mxfp4, Some(DType::Ue8M0), (1, 32)),
        FormatId::Nvfp4 | FormatId::Nvfp4WeightOnly => {
            (WeightFormat::Nvfp4, Some(DType::Ue4M3), (1, 16))
        }
    };
    let nibble = matches!(format, WeightFormat::Mxfp4 | WeightFormat::Nvfp4);
    let codes: Vec<_> = if format == WeightFormat::F32 {
        (0..128 * 128)
            .flat_map(|i| (1.0039_f32 + i as f32).to_le_bytes())
            .collect()
    } else if format == WeightFormat::Bf16 {
        (0..128 * 128)
            .flat_map(|_| 0x3f80_u16.to_le_bytes())
            .collect()
    } else {
        (0..128 * if nibble { 64 } else { 128 })
            .map(|i| (i % 112) as u8)
            .collect()
    };
    let span = |name: &str, dtype, shape, bytes: &Vec<u8>| TensorSpan {
        tensor: name.into(),
        dtype,
        shape,
        offset: 0,
        byte_length: bytes.len() as u64,
    };
    let mut payloads = vec![codes];
    let local_scale = dtype.map(|dtype| {
        let rows = if groups.0 == 0 { 1 } else { 128 / groups.0 };
        let k = if groups.1 == 0 { 1 } else { 128 / groups.1 };
        let bytes = (0..rows * k)
            .flat_map(|_| match dtype {
                DType::Bf16 => 0x3f80_u16.to_le_bytes().to_vec(),
                DType::F32 => 1.0_f32.to_le_bytes().to_vec(),
                DType::Ue8M0 => vec![127],
                _ => vec![0x38],
            })
            .collect();
        let tensor = span("weight.scale", dtype, vec![rows, k], &bytes);
        payloads.push(bytes);
        ScaleSpec {
            tensor,
            scope: if groups.0 == 0 {
                ScaleScope::Tensor
            } else if groups.1 == 0 {
                ScaleScope::Channel
            } else {
                ScaleScope::Block
            },
            convention: ScaleConvention::Multiply,
            groups: [(0, groups.0), (1, groups.1)]
                .into_iter()
                .filter(|(_, size)| *size != 0)
                .map(|(axis, size)| GroupAxis {
                    axis,
                    size: size as u32,
                    origin: 0,
                })
                .collect(),
        }
    });
    let global_scale = (format == WeightFormat::Nvfp4).then(|| {
        let bytes = 0.75_f32.to_le_bytes().to_vec();
        let tensor = span("weight.global", DType::F32, vec![], &bytes);
        payloads.push(bytes);
        ScaleSpec {
            tensor,
            scope: ScaleScope::Tensor,
            convention: ScaleConvention::Multiply,
            groups: vec![],
        }
    });
    let weight = WeightSpec {
        codes: span(
            "weight",
            if format == WeightFormat::F32 {
                DType::F32
            } else if format == WeightFormat::Bf16 {
                DType::Bf16
            } else if nibble {
                DType::U8
            } else {
                DType::Fp8E4M3
            },
            vec![128, if nibble { 64 } else { 128 }],
            &payloads[0],
        ),
        logical_shape: vec![128, 128],
        format,
        encoding: if format == WeightFormat::F32 {
            CodeEncoding::F32LittleEndian
        } else if format == WeightFormat::Bf16 {
            CodeEncoding::Bf16LittleEndian
        } else if nibble {
            CodeEncoding::E2M1(NibbleOrder::EvenHigh)
        } else {
            CodeEncoding::E4M3
        },
        layout: WeightLayout::row_major(),
        local_scale,
        global_scale,
        transforms: vec![],
    };
    weight.validate().unwrap();
    (weight, payloads)
}

#[test]
fn registry_and_companions_cover_all_imported_formats() {
    for plugin in &registry::PLUGINS {
        let (weight, _) = fixture(plugin.id());
        let registered = registry::by_weight(&weight).unwrap();
        assert_eq!(
            registered.id(),
            if plugin.id() == FormatId::Nvfp4WeightOnly {
                FormatId::Nvfp4
            } else {
                plugin.id()
            }
        );
        let bindings = plugin.companions(&weight).unwrap();
        assert_eq!(
            bindings.len(),
            if matches!(plugin.id(), FormatId::Bf16 | FormatId::F32) {
                0
            } else if weight.format == WeightFormat::Nvfp4 {
                2
            } else {
                1
            }
        );
        if let Some(local) = &weight.local_scale {
            assert_eq!(bindings[0].role, CompanionRole::LocalScale);
            assert_eq!(bindings[0].tensor, "weight.scale");
            assert_eq!(bindings[0].source_dtype, local.tensor.dtype);
            assert_eq!(bindings[0].scope, local.scope);
            assert_eq!(
                bindings[0].launch_dtype,
                if plugin.id() == FormatId::Fp8Block128 {
                    DType::F32
                } else {
                    local.tensor.dtype
                }
            );
        }
        if weight.format == WeightFormat::Nvfp4 {
            assert_eq!(bindings[1].role, CompanionRole::GlobalScale);
            assert_eq!(bindings[1].tensor, "weight.global");
            assert_eq!(bindings[1].launch_dtype, DType::F32);
        }
    }
}

#[test]
fn every_plugin_reproduces_packed_records_and_builder_bytes() {
    for plugin in &registry::PLUGINS {
        let (weight, payloads) = fixture(plugin.id());
        for family in [KernelFamily::Dense, KernelFamily::GroupedExpert] {
            // Per-tensor and per-channel banks have no existing grouped layout.
            if family == KernelFamily::GroupedExpert
                && matches!(
                    plugin.id(),
                    FormatId::Fp8PerTensor | FormatId::Fp8PerChannel | FormatId::F32
                )
            {
                continue;
            }
            let layout = plugin
                .layout(family, weight.global_scale.as_ref().map(|s| s.scope))
                .unwrap();
            let plan = PackingPlan {
                layout: layout.weight,
                format: weight.format,
                experts: 1,
                projections: 1,
                rows: 128,
                k: 128,
                columns: SwigluColumns::ConcatenatedHalves,
                local_scale_dtype: plugin
                    .companions(&weight)
                    .unwrap()
                    .first()
                    .map(|b| b.launch_dtype),
            };
            let order = plugin
                .decode_order(
                    family,
                    if plugin.id() == FormatId::F32 {
                        ActivationFormat::F32
                    } else {
                        ActivationFormat::Bf16
                    },
                    if family == KernelFamily::GroupedExpert && plugin.id() == FormatId::Fp8Block128
                    {
                        A16Recipe::ScaleAfterBlockMma
                    } else {
                        A16Recipe::ScaledWeight
                    },
                )
                .unwrap();
            let described = matrix_source(plugin, &weight, vec![0..128], 0..128).unwrap();
            // Captured single-matrix record fields: codes/local/global source
            // order, retained coordinates and the full source row/K ranges.
            let expected = PackedMatrixSource {
                expert: 0,
                projection: 0,
                format: weight.format,
                rows: 128,
                k: 128,
                row_origin: 0,
                k_origin: 0,
                nibble_order: if matches!(weight.format, WeightFormat::Mxfp4 | WeightFormat::Nvfp4)
                {
                    NibbleOrder::EvenHigh
                } else {
                    NibbleOrder::EvenLow
                },
                codes: 0,
                local: weight.local_scale.as_ref().map(|_| 1),
                local_dtype: weight.local_scale.as_ref().map(|s| s.tensor.dtype),
                global: weight.global_scale.as_ref().map(|_| 2),
                source_expert: 0,
                source_experts: 1,
                rows_kept: vec![0..128],
                k_kept: 0..128,
            };
            assert_eq!(described, expected);
            let records = packed_records(
                plugin,
                plan.clone(),
                order,
                vec![described],
                sources(&weight),
            )
            .unwrap();
            let view = MatrixView {
                tensor: "weight",
                format: weight.format,
                rows: 128,
                k: 128,
                row_origin: 0,
                k_origin: 0,
                nibble_order: expected.nibble_order,
                codes: &payloads[0],
                local: weight.local_scale.as_ref().map(|s| ScaleBytes {
                    dtype: s.tensor.dtype,
                    bytes: &payloads[1],
                }),
                global: weight.global_scale.as_ref().map(|_| ScaleBytes {
                    dtype: DType::F32,
                    bytes: &payloads[2],
                }),
            };
            let slice = MatrixSlice {
                rows: (0..128).collect(),
                k_start: 0,
                k: 128,
                padded_rows: 128,
                padded_k: 128,
                nibble_order: NibbleOrder::EvenLow,
            };
            let mut builder = PackedWeightBuilder::new(plan.clone()).unwrap();
            builder.write_matrix(0, 0, view, &slice).unwrap();
            let packed = builder.finish().unwrap();
            let expected_sections = [packed.codes, packed.local_scales, packed.global_scales];
            assert_eq!(
                records.len(),
                expected_sections.iter().filter(|s| !s.is_empty()).count()
            );
            for record in records {
                let WeightTransform::PackedOperand(operand) = &record.transform else {
                    panic!("expected packed operand")
                };
                let captured = PackedOperand {
                    local_encoding: None,
                    plan: plan.clone(),
                    section: operand.section,
                    decode_order: order,
                    matrices: vec![expected.clone()],
                };
                assert_eq!(operand.encode().unwrap(), captured.encode().unwrap());
                let bytes = operand
                    .execute(
                        &record,
                        &payloads.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                    )
                    .unwrap();
                assert_eq!(
                    bytes,
                    expected_sections[operand.section.index()],
                    "{:?} {family:?} {:?}",
                    plugin.id(),
                    operand.section
                );
                assert_eq!(record.output_bytes().unwrap(), bytes.len() as u64);
                if plugin.id() == FormatId::F32 {
                    assert_eq!(bytes, payloads[0]);
                    assert_eq!(
                        operand.code_row_sources(&record, 3..5),
                        [(0, 3 * 128 * 4, 2 * 128 * 4)]
                    );
                    let streamed = operand
                        .execute_code_rows(&record, 3..5, &mut |index, offset, out| {
                            let start = offset as usize;
                            out.copy_from_slice(&payloads[index][start..start + out.len()]);
                            Ok(())
                        })
                        .unwrap();
                    assert_eq!(streamed, payloads[0][3 * 128 * 4..5 * 128 * 4]);
                }
            }
        }
    }
}

fn legacy_record(dtype: &str, rows: u64, columns: u64, element: u64) -> WeightRecord {
    WeightRecord {
        transform: WeightTransform::Copy,
        sources: vec![SourceTensor {
            dtype: dtype.into(),
            shape: vec![rows, columns],
            byte_length: rows * columns * element,
        }],
        axis: Some(0),
        ranges: vec![0..rows],
        stored_shape: vec![rows, columns],
    }
}

#[test]
fn legacy_fragment_strip_and_widen_records_keep_wire_parameters() {
    for (id, dtype, rows, k, width, expected) in [
        (
            FormatId::Bf16,
            "BF16",
            16,
            16,
            2,
            WeightTransform::Fragment(FragmentWeightFormat::Bf16),
        ),
        (
            FormatId::Fp8Block128,
            "F8_E4M3",
            16,
            32,
            1,
            WeightTransform::Fragment(FragmentWeightFormat::Fp8E4M3),
        ),
        (
            FormatId::Fp8Rows64,
            "F8_E4M3",
            16,
            32,
            1,
            WeightTransform::Fragment(FragmentWeightFormat::Fp8E4M3),
        ),
    ] {
        let source_dtype = if id == FormatId::Bf16 {
            DType::Bf16
        } else {
            DType::Fp8E4M3
        };
        let base = legacy_record(dtype, rows, k, width);
        let record = section_record(
            registry::by_id(id).unwrap(),
            PackedSection::Codes,
            source_dtype,
            source_dtype,
            RecordLayout::Fragment,
            base.clone(),
        )
        .unwrap();
        assert_eq!(record.transform.id(), 3);
        assert_eq!(
            record.transform.encode_params().unwrap(),
            expected.encode_params().unwrap()
        );
        assert_eq!(record.sources, base.sources);
        assert_eq!(record.ranges, base.ranges);
        assert_eq!(record.stored_shape, base.stored_shape);
    }
    for (section, cols, param) in [
        (PackedSection::Codes, 32, 1_u32),
        (PackedSection::LocalScales, 2, 2),
    ] {
        let record = section_record(
            registry::by_id(FormatId::Mxfp4).unwrap(),
            section,
            DType::U8,
            DType::U8,
            RecordLayout::Mxfp4Strips,
            legacy_record("U8", 16, cols, 1),
        )
        .unwrap();
        assert_eq!(record.transform.id(), 6);
        assert_eq!(
            record.transform.encode_params().unwrap(),
            param.to_le_bytes()
        );
        assert_eq!(record.output_bytes().unwrap(), 16 * cols);
    }
    let record = section_record(
        registry::by_id(FormatId::Fp8Block128).unwrap(),
        PackedSection::LocalScales,
        DType::Bf16,
        DType::F32,
        RecordLayout::Rows,
        legacy_record("BF16", 2, 2, 2),
    )
    .unwrap();
    assert_eq!(record.transform.id(), 2);
    assert!(record.transform.encode_params().unwrap().is_empty());
    assert_eq!(record.output_bytes().unwrap(), 16);
}

#[test]
fn malformed_sources_and_wrong_plugins_are_refused() {
    let (weight, _) = fixture(FormatId::Nvfp4);
    assert!(
        matrix_source(
            registry::by_id(FormatId::Bf16).unwrap(),
            &weight,
            vec![0..128],
            0..128
        )
        .is_err()
    );
    assert!(
        matrix_source(
            registry::by_id(FormatId::Nvfp4).unwrap(),
            &weight,
            vec![0..129],
            0..128
        )
        .is_err()
    );
    let mut missing = weight.clone();
    missing.global_scale = None;
    assert!(registry::by_weight(&missing).is_err());
    assert!(
        section_record(
            registry::by_id(FormatId::Mxfp8).unwrap(),
            PackedSection::Codes,
            DType::Fp8E4M3,
            DType::Fp8E4M3,
            RecordLayout::Fragment,
            legacy_record("F8_E4M3", 16, 32, 1)
        )
        .is_err()
    );
}

#[test]
fn imported_origins_bank_coordinates_and_bf16_lookup_scale_are_preserved() {
    let (mut weight, _) = fixture(FormatId::Nvfp4);
    weight.logical_shape.insert(0, 2);
    weight.codes.shape.insert(0, 2);
    weight.codes.byte_length *= 2;
    let local = weight.local_scale.as_mut().unwrap();
    for group in &mut local.groups {
        group.axis += 1;
        if group.size == 16 {
            group.origin = 16;
        }
    }
    local.tensor.shape.insert(0, 2);
    local.tensor.byte_length *= 2;
    let global = weight.global_scale.as_mut().unwrap();
    global.scope = ScaleScope::Expert;
    global.tensor.shape = vec![2];
    global.tensor.byte_length = 8;
    let plugin = registry::by_weight(&weight).unwrap();
    let matrix = matrix_source(plugin, &weight, vec![32..64], 16..48).unwrap();
    assert_eq!(matrix.k_origin, 16);
    assert_eq!(matrix.source_experts, 2);
    assert_eq!(matrix.rows_kept, vec![32..64]);
    assert_eq!(matrix.k_kept, 16..48);
    let binding = plugin.companions(&weight).unwrap();
    assert_eq!(
        binding[0].groups,
        weight.local_scale.as_ref().unwrap().groups
    );
    assert_eq!(binding[1].scope, ScaleScope::Expert);
    let layout = plugin
        .layout_for_weight(
            &weight,
            KernelFamily::GroupedExpert,
            kernel_core::precision::format::LayoutVariant::Standard,
        )
        .unwrap();
    assert_eq!(layout.weight.id, "moe-packed-ExpertProjection");
    let (lookup, _) = fixture(FormatId::Fp8PerTensor);
    let layout = registry::by_weight(&lookup)
        .unwrap()
        .layout_for_weight(
            &lookup,
            KernelFamily::Lookup,
            kernel_core::precision::format::LayoutVariant::Standard,
        )
        .unwrap();
    assert_eq!(layout.local_scales, Some("bf16_row_major_v1"));
    weight.local_scale.as_mut().unwrap().convention = ScaleConvention::Reciprocal;
    assert!(matrix_source(plugin, &weight, vec![0..128], 0..128).is_err());
}
