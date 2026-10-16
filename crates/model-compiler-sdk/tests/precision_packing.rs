use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::{DType, ScaleMode, SwigluColumns, WeightFormat};
use kernel_core::precision::packed::{MatrixSlice, MatrixView, ScaleBytes};
use kernel_core::precision::{DecodeOrder, NibbleOrder, WeightLayout};
use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
use kernel_core::transform::{Layout, Packed, ScaleTileLayout, Transform};
use model_compiler_sdk::quantization::packing::{
    PackedWeightBuilder, PackingPlan, uniform_f32_global,
};

fn plan(format: WeightFormat, layout: &str, rows: u32, k: u32) -> PackingPlan {
    PackingPlan {
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
        local_scale_dtype: match format {
            WeightFormat::Bf16 => None,
            WeightFormat::Nvfp4 => Some(DType::Fp8E4M3),
            WeightFormat::Mxfp8 => Some(DType::Ue8M0),
            _ => Some(DType::F32),
        },
    }
}
fn slice(rows: usize, k: usize) -> MatrixSlice {
    MatrixSlice {
        rows: (0..rows).collect(),
        k_start: 0,
        k,
        padded_rows: rows,
        padded_k: k,
        nibble_order: NibbleOrder::EvenLow,
    }
}
fn nv<'a>(
    codes: &'a [u8],
    scales: &'a [u8],
    global: &'a [u8],
    rows: usize,
    k: usize,
) -> MatrixView<'a> {
    MatrixView {
        tensor: "unrelated.expert",
        format: WeightFormat::Nvfp4,
        rows,
        k,
        row_origin: 0,
        k_origin: 0,
        nibble_order: NibbleOrder::EvenLow,
        codes,
        local: Some(ScaleBytes {
            dtype: DType::Fp8E4M3,
            bytes: scales,
        }),
        global: Some(ScaleBytes {
            dtype: DType::F32,
            bytes: global,
        }),
    }
}

#[test]
fn streamed_experts_keep_projection_globals_and_scale_tiles_exact() {
    for columns in [
        SwigluColumns::ConcatenatedHalves,
        SwigluColumns::Interleaved,
    ] {
        let mut request = plan(WeightFormat::Nvfp4, "moe-packed-ExpertProjection", 32, 128);
        request.experts = 2;
        request.projections = 2;
        request.columns = columns;
        let mut builder = PackedWeightBuilder::new(request.clone()).unwrap();
        let mut expected_codes = vec![0; 128 * 64];
        let mut expected_scales = vec![0; 128 * 8];
        let mut globals = Vec::new();
        for expert in 0..2 {
            for projection in 0..2 {
                let global = (1.0_f32 + expert as f32 + projection as f32 / 4.0).to_le_bytes();
                globals.extend(global);
                let codes: Vec<_> = (0..24 * 64)
                    .map(|i| (i + expert * 11 + projection * 31) as u8)
                    .collect();
                let scales: Vec<_> = (0..24 * 8).map(|i| (i % 120 + 1) as u8).collect();
                let mut selection = slice(24, 128);
                selection.padded_rows = 32;
                builder
                    .write_matrix(
                        expert as u32,
                        projection as u32,
                        nv(&codes, &scales, &global, 24, 128),
                        &selection,
                    )
                    .unwrap();
                for row in 0..24 {
                    let dst = expert * 64
                        + if columns == SwigluColumns::Interleaved {
                            row * 2 + projection
                        } else {
                            projection * 32 + row
                        };
                    expected_codes[dst * 64..(dst + 1) * 64]
                        .copy_from_slice(&codes[row * 64..(row + 1) * 64]);
                    expected_scales[dst * 8..(dst + 1) * 8]
                        .copy_from_slice(&scales[row * 8..(row + 1) * 8]);
                }
            }
        }
        let packed = builder.finish().unwrap();
        assert_eq!(packed.codes, expected_codes);
        assert_eq!(packed.global_scales, globals);
        let layout = ScaleTileLayout {
            rows: 128,
            k_blocks: 8,
            tile_rows: 16,
            k_blocks_per_tile: 8,
        };
        let decoded = layout
            .unpack(&Packed {
                bytes: packed.local_scales.clone(),
                layout: Layout::ScaleTile(layout),
            })
            .unwrap();
        assert_eq!(decoded, expected_scales);
        let sections = request.section_bytes().unwrap();
        assert_eq!(sections.codes as usize, packed.codes.len());
        assert_eq!(sections.local_scales as usize, packed.local_scales.len());
        assert_eq!(sections.global_scales as usize, packed.global_scales.len());
        assert_eq!(
            packed.codes.len() + packed.local_scales.len() + packed.global_scales.len(),
            request.resident_bytes().unwrap() as usize
        );
        assert!(request.resident_bytes().unwrap() < 128 * 128); // smaller than BF16 operands
    }
}

#[test]
fn shared_mx_padding_and_disjoint_projection_rows_preserve_decoded_operands() {
    let codes: Vec<_> = (0..320 * 640).map(|i| (i % 126) as u8).collect();
    let scales: Vec<_> = (0..320 * 20).map(|i| (i % 5 + 123) as u8).collect();
    let source = MatrixView {
        tensor: "shared",
        format: WeightFormat::Mxfp8,
        rows: 320,
        k: 640,
        row_origin: 0,
        k_origin: 0,
        nibble_order: NibbleOrder::EvenLow,
        codes: &codes,
        local: Some(ScaleBytes {
            dtype: DType::Ue8M0,
            bytes: &scales,
        }),
        global: None,
    };
    let mut builder =
        PackedWeightBuilder::new(plan(WeightFormat::Mxfp8, "row-major", 160, 256)).unwrap();
    let selection = MatrixSlice {
        rows: (160..240).chain(0..80).collect(),
        k_start: 160,
        k: 160,
        padded_rows: 160,
        padded_k: 256,
        nibble_order: NibbleOrder::EvenLow,
    };
    builder.write_matrix(0, 0, source, &selection).unwrap();
    let packed = builder.finish().unwrap();
    let decoded = MatrixView {
        tensor: "prepared",
        rows: 160,
        k: 256,
        row_origin: 0,
        k_origin: 160,
        codes: &packed.codes,
        local: Some(ScaleBytes {
            dtype: DType::Ue8M0,
            bytes: &packed.local_scales,
        }),
        ..source
    };
    decoded.validate().unwrap();
    for row in 0..160 {
        for k in 0..256 {
            let value = decoded
                .bf16_operand(row, k, DecodeOrder::ScaleThenCode)
                .unwrap();
            if k < 160 {
                assert_eq!(
                    value,
                    source
                        .bf16_operand(selection.rows[row], 160 + k, DecodeOrder::ScaleThenCode)
                        .unwrap()
                );
            } else {
                assert_eq!(value, 0);
            }
        }
    }
    let mut builder =
        PackedWeightBuilder::new(plan(WeightFormat::Mxfp8, "row-major", 160, 256)).unwrap();
    let mut misaligned = selection;
    misaligned.k_start += 1;
    assert!(
        builder
            .write_matrix(0, 0, source, &misaligned)
            .err()
            .unwrap()
            .to_string()
            .contains("scale-group origin")
    );
}

#[test]
fn legacy_fp8_and_bf16_use_existing_fragment_bytes_and_exact_scale_widening() {
    for format in [
        WeightFormat::Bf16,
        WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
    ] {
        let bf16 = format == WeightFormat::Bf16;
        let codes: Vec<_> = if bf16 {
            (0..128 * 256)
                .flat_map(|i| (0x3d00_u16 + (i % 100) as u16).to_le_bytes())
                .collect()
        } else {
            (0..128 * 256).map(|i| (i % 126) as u8).collect()
        };
        let scales: Vec<_> = [0x3f71_u16, 0x3f8a]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        let source = MatrixView {
            tensor: "legacy",
            format,
            rows: 128,
            k: 256,
            row_origin: 0,
            k_origin: 0,
            nibble_order: NibbleOrder::EvenLow,
            codes: &codes,
            local: (!bf16).then_some(ScaleBytes {
                dtype: DType::Bf16,
                bytes: &scales,
            }),
            global: None,
        };
        let mut builder = PackedWeightBuilder::new(plan(
            format,
            if bf16 {
                "bf16-fragment"
            } else {
                "fp8-fragment"
            },
            128,
            256,
        ))
        .unwrap();
        builder
            .write_matrix(0, 0, source, &slice(128, 256))
            .unwrap();
        let packed = builder.finish().unwrap();
        let layout = FragmentWeightLayout {
            rows: 128,
            k: 256,
            format: if bf16 {
                FragmentWeightFormat::Bf16
            } else {
                FragmentWeightFormat::Fp8E4M3
            },
        };
        assert_eq!(packed.codes, layout.pack(&codes).unwrap().bytes);
        if !bf16 {
            assert_eq!(
                packed.local_scales,
                scales
                    .chunks_exact(2)
                    .flat_map(
                        |s| bf16_to_f32(u16::from_le_bytes(s.try_into().unwrap())).to_le_bytes()
                    )
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn scalar_coalescing_requires_equal_bits_and_lookups_preserve_scale_dtype() {
    let a = 1.0_f32.to_le_bytes();
    let b = f32::from_bits(1.0_f32.to_bits() + 1).to_le_bytes();
    let scale = |bytes| ScaleBytes {
        dtype: DType::F32,
        bytes,
    };
    assert_eq!(uniform_f32_global([scale(&a), scale(&a)]).unwrap(), a);
    assert!(uniform_f32_global([scale(&a), scale(&b)]).is_err());
    let mut request = plan(WeightFormat::Nvfp4, "moe-packed-Scalar", 16, 128);
    request.experts = 2;
    let mut builder = PackedWeightBuilder::new(request).unwrap();
    let codes = vec![0x21; 16 * 64];
    let scales = vec![0x38; 16 * 8];
    builder
        .write_matrix(0, 0, nv(&codes, &scales, &a, 16, 128), &slice(16, 128))
        .unwrap();
    assert!(
        builder
            .write_matrix(1, 0, nv(&codes, &scales, &b, 16, 128), &slice(16, 128))
            .err()
            .unwrap()
            .to_string()
            .contains("bitwise equal")
    );
    assert!(builder.finish().is_err());
    for dtype in [DType::Bf16, DType::F32] {
        let scales = if dtype == DType::Bf16 {
            vec![0x80, 0x3f]
        } else {
            b.to_vec()
        };
        let codes = vec![0x38; 37 * 160];
        let source = MatrixView {
            tensor: "table",
            format: WeightFormat::Fp8PerTensor,
            rows: 37,
            k: 160,
            row_origin: 0,
            k_origin: 0,
            nibble_order: NibbleOrder::EvenLow,
            codes: &codes,
            local: Some(ScaleBytes {
                dtype,
                bytes: &scales,
            }),
            global: None,
        };
        let mut request = plan(WeightFormat::Fp8PerTensor, "row-major", 17, 160);
        request.local_scale_dtype = Some(dtype);
        let mut builder = PackedWeightBuilder::new(request).unwrap();
        let mut selection = slice(17, 160);
        selection.rows = (20..37).collect();
        builder.write_matrix(0, 0, source, &selection).unwrap();
        let packed = builder.finish().unwrap();
        assert_eq!(packed.local_scales, scales);
        assert_eq!(packed.codes, vec![0x38; 17 * 160]);
    }
}

#[test]
fn preparation_refuses_oversized_layouts_duplicate_writes_and_scale_group_remaps() {
    for (rows, k, experts) in [(u32::MAX - 1, u32::MAX - 1, u32::MAX), (16, 128, u32::MAX)] {
        let mut request = plan(WeightFormat::Nvfp4, "moe-packed-ExpertProjection", rows, k);
        request.experts = experts;
        // A portable footprint can describe a large streamed table or bank;
        // the allocating builder must enforce its working-set bound first.
        assert!(PackedWeightBuilder::new(request).is_err());
    }
    let mut request = plan(WeightFormat::Nvfp4, "row-major", 16, 128);
    request.layout.version = 2;
    assert!(PackedWeightBuilder::new(request).is_err());
    let mut builder =
        PackedWeightBuilder::new(plan(WeightFormat::Nvfp4, "row-major", 16, 128)).unwrap();
    let codes = vec![0x32; 16 * 64];
    let scales = vec![0x38; 16 * 8];
    let global = 1.0_f32.to_le_bytes();
    let source = nv(&codes, &scales, &global, 16, 128);
    builder.write_matrix(0, 0, source, &slice(16, 128)).unwrap();
    assert!(
        builder
            .write_matrix(0, 0, source, &slice(16, 128))
            .err()
            .unwrap()
            .to_string()
            .contains("already supplied")
    );
    let codes = vec![0x38; 256 * 128];
    let scales = [1.0_f32.to_le_bytes(), 2.0_f32.to_le_bytes()].concat();
    let source = MatrixView {
        tensor: "blocked",
        format: WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
        rows: 256,
        k: 128,
        row_origin: 0,
        k_origin: 0,
        nibble_order: NibbleOrder::EvenLow,
        codes: &codes,
        local: Some(ScaleBytes {
            dtype: DType::F32,
            bytes: &scales,
        }),
        global: None,
    };
    let mut selection = slice(128, 128);
    selection.rows = (0..64).chain(128..192).collect();
    assert!(
        source
            .repack(&selection)
            .err()
            .unwrap()
            .to_string()
            .contains("different source scale groups")
    );
}
