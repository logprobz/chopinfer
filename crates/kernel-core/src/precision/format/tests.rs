use super::*;
use crate::contract::ScaleMode;
use crate::precision::packed::ScaleBytes;

// Characterization of the existing dense, grouped and lookup A16 recipes.
// The post-block column is the existing block-FP8 recipe used by consumers
// whose MMA operates on raw FP8 codes converted exactly to BF16.
#[test]
fn captured_a16_decode_orders_cover_every_plugin() {
    use DecodeOrder::*;
    use FormatId::*;
    for (id, dense, grouped, lookup, post_block) in [
        (
            Bf16,
            ScaleThenCode,
            CodeThenScale,
            CodeThenScale,
            ScaleThenCode,
        ),
        (
            Fp8PerTensor,
            ScaleThenCode,
            CodeThenScale,
            CodeThenScale,
            ScaleThenCode,
        ),
        (
            Fp8PerChannel,
            ScaleThenCode,
            CodeThenScale,
            CodeThenScale,
            ScaleThenCode,
        ),
        (
            Fp8Block128,
            ScaleThenCode,
            CodeThenScale,
            CodeThenScale,
            BlockScaleAfterMma,
        ),
        (
            Fp8Rows64,
            ScaleThenCode,
            CodeThenScale,
            CodeThenScale,
            BlockScaleAfterMma,
        ),
        (
            Mxfp8,
            ExactScaledOperand,
            ExactScaledOperand,
            ExactScaledOperand,
            ExactScaledOperand,
        ),
        (
            Mxfp4,
            ExactScaledOperand,
            ExactScaledOperand,
            ExactScaledOperand,
            ExactScaledOperand,
        ),
        (
            Nvfp4,
            ExactLocalThenGlobal,
            ExactLocalThenGlobal,
            ExactWeightToBf16,
            ExactLocalThenGlobal,
        ),
        (
            Nvfp4WeightOnly,
            ExactLocalThenGlobal,
            ExactLocalThenGlobal,
            ExactWeightToBf16,
            ExactLocalThenGlobal,
        ),
    ] {
        let plugin = registry::by_id(id).unwrap();
        for (family, expected) in [
            (KernelFamily::Dense, dense),
            (KernelFamily::GroupedExpert, grouped),
            (KernelFamily::Lookup, lookup),
        ] {
            assert_eq!(
                plugin
                    .decode_order(family, ActivationFormat::Bf16, A16Recipe::ScaledWeight)
                    .unwrap(),
                expected,
                "{id:?} {family:?}"
            );
        }
        assert_eq!(
            plugin
                .decode_order(
                    KernelFamily::Dense,
                    ActivationFormat::Bf16,
                    A16Recipe::ScaleAfterBlockMma
                )
                .unwrap(),
            post_block
        );
    }
}

#[test]
fn quantized_activation_scale_order_matches_existing_recipes() {
    for id in [FormatId::Fp8Block128, FormatId::Fp8Rows64] {
        for (family, expected) in [
            (KernelFamily::Dense, DecodeOrder::ScaleProductAfterBlockMma),
            (
                KernelFamily::GroupedExpert,
                DecodeOrder::SuccessiveScalesAfterBlockMma,
            ),
        ] {
            assert_eq!(
                registry::by_id(id)
                    .unwrap()
                    .decode_order(
                        family,
                        ActivationFormat::Fp8Group128(ScaleMode::ExactF32),
                        A16Recipe::ScaledWeight
                    )
                    .unwrap(),
                expected
            );
        }
    }
    for id in [FormatId::Mxfp8, FormatId::Mxfp4] {
        assert_eq!(
            registry::by_id(id)
                .unwrap()
                .decode_order(
                    KernelFamily::Dense,
                    ActivationFormat::Mxfp8,
                    A16Recipe::ScaledWeight
                )
                .unwrap(),
            DecodeOrder::NativeBlockScaledMma
        );
    }
    assert_eq!(
        registry::by_id(FormatId::Nvfp4)
            .unwrap()
            .decode_order(
                KernelFamily::Dense,
                ActivationFormat::Nvfp4,
                A16Recipe::ScaledWeight
            )
            .unwrap(),
        DecodeOrder::NativeBlockScaledMmaThenGlobals
    );
    assert!(
        registry::by_id(FormatId::Nvfp4WeightOnly)
            .unwrap()
            .decode_order(
                KernelFamily::Dense,
                ActivationFormat::Nvfp4,
                A16Recipe::ScaledWeight
            )
            .is_err()
    );
    assert!(
        registry::by_id(FormatId::Bf16)
            .unwrap()
            .decode_order(
                KernelFamily::Dense,
                ActivationFormat::Mxfp8,
                A16Recipe::ScaledWeight
            )
            .is_err()
    );
}

// Physical IDs captured from the existing fragment, row, scale and grouped
// contracts. New format combinations use the same row/tile naming convention.
#[test]
fn layouts_preserve_existing_ids_and_explicit_variants() {
    use FormatId::*;
    for (id, layout, codes, scales) in [
        (Bf16, "bf16-fragment", "bf16_mma_fragments_16x16_v1", None),
        (
            Fp8PerTensor,
            "row-major",
            "fp8_e4m3_row_major_v1",
            Some("f32_row_major_v1"),
        ),
        (
            Fp8PerChannel,
            "row-major",
            "fp8_e4m3_row_major_v1",
            Some("f32_row_major_v1"),
        ),
        (
            Fp8Block128,
            "fp8-fragment",
            "fp8_e4m3_mma_fragments_16x32_v1",
            Some("f32_block_scale_128x128_v1"),
        ),
        (
            Fp8Rows64,
            "fp8-fragment",
            "fp8_e4m3_mma_fragments_16x32_v1",
            Some("fp8_f32_row_scale_64x128_v1"),
        ),
        (
            Mxfp8,
            "row-major",
            "mxfp8_e4m3_rows_v1",
            Some("mxfp8_ue8m0_scale_rows_v1"),
        ),
        (
            Mxfp4,
            "row-major",
            "mxfp4_e2m1_rows_v1",
            Some("mxfp4_ue8m0_scale_rows_v1"),
        ),
        (
            Nvfp4,
            "row-major",
            "nvfp4_e2m1_rows_v1",
            Some("nvfp4_e4m3_scale_rows_v1"),
        ),
        (
            Nvfp4WeightOnly,
            "row-major",
            "nvfp4_e2m1_rows_v1",
            Some("nvfp4_e4m3_scale_rows_v1"),
        ),
    ] {
        let global = matches!(id, Nvfp4 | Nvfp4WeightOnly).then_some(ScaleScope::Tensor);
        let choice = registry::by_id(id)
            .unwrap()
            .layout(KernelFamily::Dense, global)
            .unwrap();
        assert_eq!(choice.weight.id, layout);
        assert_eq!(choice.weight.version, 1);
        assert_eq!(choice.codes, codes);
        assert_eq!(choice.local_scales, scales);
        assert_eq!(choice.global_scales, global.map(|_| "f32_row_major_v1"));
        for family in [
            KernelFamily::Row,
            KernelFamily::Lookup,
            KernelFamily::LayoutCopy,
        ] {
            assert_eq!(
                registry::by_id(id)
                    .unwrap()
                    .layout(family, global)
                    .unwrap()
                    .weight,
                WeightLayout::row_major()
            );
        }
    }
    let nv = registry::by_id(Nvfp4)
        .unwrap()
        .layout(KernelFamily::GroupedExpert, Some(ScaleScope::Expert))
        .unwrap();
    assert_eq!(nv.weight.id, "moe-packed-ExpertProjection");
    assert_eq!(nv.local_scales, Some("nvfp4_e4m3_scale_tiles_16x8_v1"));
    assert_eq!(
        registry::by_id(Nvfp4)
            .unwrap()
            .layout(KernelFamily::GroupedExpert, Some(ScaleScope::Tensor))
            .unwrap()
            .weight
            .id,
        "moe-packed-Scalar"
    );
    for id in [Mxfp8, Mxfp4, Fp8Block128] {
        assert_eq!(
            registry::by_id(id)
                .unwrap()
                .layout(KernelFamily::GroupedExpert, None)
                .unwrap()
                .weight
                .id,
            "moe-packed-Absent"
        );
    }
    for (id, expected) in [
        (Fp8Block128, "fp8_f32_block_scale_128x128_v1"),
        (Fp8Rows64, "fp8_f32_row_scale_64x128_v1"),
    ] {
        assert_eq!(
            registry::by_id(id)
                .unwrap()
                .layout_variant(
                    KernelFamily::Dense,
                    None,
                    LayoutVariant::AlternateBlockScaleNames
                )
                .unwrap()
                .local_scales,
            Some(expected)
        );
    }
    for fused_gate_up in [false, true] {
        let choice = registry::by_id(Mxfp4)
            .unwrap()
            .layout_variant(
                KernelFamily::GroupedExpert,
                None,
                LayoutVariant::Mxfp4Strips { fused_gate_up },
            )
            .unwrap();
        assert_eq!(
            choice.codes,
            if fused_gate_up {
                "mxfp4_gate_up_moe_fragments_16x64_v1"
            } else {
                "mxfp4_moe_fragments_16x64_v1"
            }
        );
        assert_eq!(
            choice.local_scales,
            Some(if fused_gate_up {
                "mxfp4_gate_up_moe_fragment_scales_16x64_v2"
            } else {
                "mxfp4_moe_fragment_scales_16x64_v2"
            })
        );
    }
}

/// A dense 32x32 UE8M0 operand is resident as its MXFP8 rows; other
/// families keep the 32x32 grid.
#[test]
fn block32_dense_operands_are_resident_as_mxfp8_rows() {
    let plugin = registry::by_id(FormatId::Fp8Block32).unwrap();
    let dense = plugin.layout(KernelFamily::Dense, None).unwrap();
    assert_eq!(
        (dense.weight.id.as_str(), dense.codes, dense.local_scales),
        (
            "row-major",
            "mxfp8_e4m3_rows_v1",
            Some("mxfp8_ue8m0_scale_rows_v1")
        )
    );
    assert_eq!(
        plugin.resident_format(WeightFormat::Fp8Block32x32, KernelFamily::Dense),
        WeightFormat::Mxfp8
    );
    for family in [KernelFamily::Row, KernelFamily::GroupedExpert] {
        assert_eq!(
            plugin.layout(family, None).unwrap().local_scales,
            Some("fp8_ue8m0_block_scale_32x32_v1")
        );
        assert_eq!(
            plugin.resident_format(WeightFormat::Fp8Block32x32, family),
            WeightFormat::Fp8Block32x32
        );
    }
    for id in [FormatId::Mxfp8, FormatId::Fp8Block128] {
        let format = if id == FormatId::Mxfp8 {
            WeightFormat::Mxfp8
        } else {
            WeightFormat::Fp8Block128x128(ScaleMode::ExactF32)
        };
        assert_eq!(
            registry::by_id(id)
                .unwrap()
                .resident_format(format, KernelFamily::Dense),
            format
        );
    }
}

#[test]
fn every_format_slices_exact_codes_scales_and_origins() {
    for (id, format, group_rows, group_k, dtype) in [
        (FormatId::Bf16, WeightFormat::Bf16, 256, 256, None),
        (
            FormatId::Fp8PerTensor,
            WeightFormat::Fp8PerTensor,
            256,
            256,
            Some(DType::F32),
        ),
        (
            FormatId::Fp8PerChannel,
            WeightFormat::Fp8PerChannel,
            1,
            256,
            Some(DType::F32),
        ),
        (
            FormatId::Fp8Block128,
            WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
            128,
            128,
            Some(DType::F32),
        ),
        (
            FormatId::Fp8Rows64,
            WeightFormat::Fp8Rows64x128(ScaleMode::ExactF32),
            64,
            128,
            Some(DType::Bf16),
        ),
        (
            FormatId::Mxfp8,
            WeightFormat::Mxfp8,
            1,
            32,
            Some(DType::Ue8M0),
        ),
        (
            FormatId::Mxfp4,
            WeightFormat::Mxfp4,
            1,
            32,
            Some(DType::Ue8M0),
        ),
        (
            FormatId::Nvfp4,
            WeightFormat::Nvfp4,
            1,
            16,
            Some(DType::Ue4M3),
        ),
        (
            FormatId::Nvfp4WeightOnly,
            WeightFormat::Nvfp4,
            1,
            16,
            Some(DType::Fp8E4M3),
        ),
    ] {
        let bits = if format == WeightFormat::Bf16 {
            16
        } else if matches!(format, WeightFormat::Nvfp4 | WeightFormat::Mxfp4) {
            4
        } else {
            8
        };
        let codes: Vec<_> = (0..256 * 256 * bits / 8)
            .map(|i| {
                if bits == 16 {
                    [0x80, 0x3f][i % 2]
                } else {
                    (i % 112) as u8
                }
            })
            .collect();
        let scales: Vec<_> = (0..(256 / group_rows) * (256 / group_k))
            .flat_map(|i| match dtype {
                Some(DType::F32) => (1.0 + (i % 4) as f32).to_le_bytes().to_vec(),
                Some(DType::Bf16) => (0x3f80_u16 + (i % 4) as u16 * 128).to_le_bytes().to_vec(),
                Some(DType::Ue8M0) => vec![127 + (i % 2) as u8],
                Some(_) => vec![0x38 + (i % 2) as u8],
                None => vec![],
            })
            .collect();
        let global = 0.75_f32.to_le_bytes();
        let matrix = MatrixView {
            tensor: "fixture",
            format,
            rows: 256,
            k: 256,
            row_origin: 0,
            k_origin: 0,
            nibble_order: NibbleOrder::EvenHigh,
            codes: &codes,
            local: dtype.map(|dtype| ScaleBytes {
                dtype,
                bytes: &scales,
            }),
            global: (format == WeightFormat::Nvfp4).then_some(ScaleBytes {
                dtype: DType::F32,
                bytes: &global,
            }),
        };
        let order = registry::by_id(id)
            .unwrap()
            .decode_order(
                KernelFamily::Dense,
                ActivationFormat::Bf16,
                A16Recipe::ScaledWeight,
            )
            .unwrap();
        for (axis, range, padding) in [
            (SliceAxis::Rows, 128..144, PaddingTile { rows: 32, k: 1 }),
            (SliceAxis::K, 129..161, PaddingTile { rows: 1, k: 64 }),
        ] {
            let request = SliceRequest {
                axis,
                range: range.clone(),
                padding,
                nibble_order: NibbleOrder::EvenLow,
            };
            let result = registry::by_id(id)
                .unwrap()
                .slice(matrix, &request)
                .unwrap();
            let (row_start, col_start) = if axis == SliceAxis::Rows {
                (range.start, 0)
            } else {
                (0, range.start)
            };
            assert_eq!(result.k_origin, col_start);
            assert_eq!(
                result.row_origin,
                if group_rows == 256 { 0 } else { row_start }
            );
            let dst = result.view("sliced");
            dst.validate().unwrap();
            for row in 0..result.logical_rows {
                for col in 0..result.logical_k {
                    assert_eq!(
                        dst.bf16_operand(row, col, order).unwrap(),
                        matrix
                            .bf16_operand(row + row_start, col + col_start, order)
                            .unwrap(),
                        "{id:?} {axis:?} {row} {col}"
                    );
                    let raw = |bytes: &[u8], index: usize, even_high: bool| match bits {
                        4 => {
                            (bytes[index / 2]
                                >> if index.is_multiple_of(2) == even_high {
                                    4
                                } else {
                                    0
                                })
                                & 15
                        }
                        8 => bytes[index],
                        _ => bytes[index * 2],
                    };
                    assert_eq!(
                        raw(&result.codes, row * result.k + col, false),
                        raw(&codes, (row + row_start) * 256 + col + col_start, true)
                    );
                }
            }
            if let Some(local) = matrix.local {
                let width = local.element_bytes().unwrap();
                let (_, columns) = dst.scale_shape().unwrap();
                let source_columns = 256 / group_k;
                for row in 0..result.logical_rows {
                    for col in 0..result.logical_k {
                        let dest_index = dst.scale_index(row, col).unwrap();
                        let source_index = (row + row_start) / group_rows * source_columns
                            + (col + col_start) / group_k;
                        assert!(dest_index < result.scales.len() / width);
                        assert!(columns > 0);
                        assert_eq!(
                            &result.scales[dest_index * width..(dest_index + 1) * width],
                            &scales[source_index * width..(source_index + 1) * width]
                        );
                    }
                }
            }
            assert_eq!(result.global.as_deref(), matrix.global.map(|g| g.bytes));
            for row in 0..result.rows {
                for col in 0..result.k {
                    if row >= result.logical_rows || col >= result.logical_k {
                        assert_eq!(dst.bf16_operand(row, col, order).unwrap() & 0x7fff, 0);
                    }
                }
            }
        }
        for request in [
            SliceRequest {
                axis: SliceAxis::K,
                range: 0..257,
                padding: PaddingTile::NONE,
                nibble_order: NibbleOrder::EvenLow,
            },
            SliceRequest {
                axis: SliceAxis::Rows,
                range: 2..2,
                padding: PaddingTile::NONE,
                nibble_order: NibbleOrder::EvenLow,
            },
            SliceRequest {
                axis: SliceAxis::Rows,
                range: 1..2,
                padding: PaddingTile { rows: 0, k: 1 },
                nibble_order: NibbleOrder::EvenLow,
            },
        ] {
            assert!(
                registry::by_id(id)
                    .unwrap()
                    .slice(matrix, &request)
                    .is_err()
            );
        }
        if bits == 4 {
            assert!(
                registry::by_id(id)
                    .unwrap()
                    .slice(
                        matrix,
                        &SliceRequest {
                            axis: SliceAxis::K,
                            range: 0..3,
                            padding: PaddingTile::NONE,
                            nibble_order: NibbleOrder::EvenLow
                        }
                    )
                    .is_err()
            );
        }
    }
}
