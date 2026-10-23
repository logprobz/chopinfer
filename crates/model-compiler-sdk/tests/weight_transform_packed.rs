//! Direct loading packs prepared operands with kernel-core's weight
//! transform; every section equals this crate's prepared-weight builder.
#![allow(clippy::single_range_in_vec_init)]

use kernel_core::contract::{DType, ScaleMode, SwigluColumns, WeightFormat};
use kernel_core::precision::packed::{MatrixSlice, MatrixView, ScaleBytes};
use kernel_core::precision::{DecodeOrder, NibbleOrder, WeightLayout};
use kernel_core::weight_transform::{
    PackedMatrixSource, PackedOperand, PackedSection, SourceTensor, WeightRecord, WeightTransform,
};
use model_compiler_sdk::quantization::packing::{PackedWeightBuilder, PackingPlan};
use std::ops::Range;

/// One source matrix: codes, optional local scales and optional global.
struct Matrix {
    expert: u32,
    projection: u32,
    format: WeightFormat,
    rows: usize,
    k: usize,
    codes: Vec<u8>,
    local: Option<(DType, Vec<u8>)>,
    global: Option<Vec<u8>>,
    rows_kept: Vec<Range<u64>>,
    k_kept: Range<u64>,
}

fn bytes(count: usize, seed: usize, value: impl Fn(usize) -> u8) -> Vec<u8> {
    (0..count)
        .map(|i| value(i.wrapping_mul(31) ^ seed))
        .collect()
}

fn plan(
    format: WeightFormat,
    layout: &str,
    rows: u32,
    k: u32,
    local: Option<DType>,
) -> PackingPlan {
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
        local_scale_dtype: local,
    }
}

fn view(matrix: &Matrix) -> MatrixView<'_> {
    MatrixView {
        tensor: "fixture",
        format: matrix.format,
        rows: matrix.rows,
        k: matrix.k,
        row_origin: 0,
        k_origin: 0,
        nibble_order: NibbleOrder::EvenLow,
        codes: &matrix.codes,
        local: matrix.local.as_ref().map(|(dtype, bytes)| ScaleBytes {
            dtype: *dtype,
            bytes,
        }),
        global: matrix.global.as_ref().map(|bytes| ScaleBytes {
            dtype: DType::F32,
            bytes,
        }),
    }
}

fn check(plan: PackingPlan, order: DecodeOrder, matrices: &[Matrix]) {
    let mut builder = PackedWeightBuilder::new(plan.clone()).unwrap();
    for matrix in matrices {
        let slice = MatrixSlice {
            rows: matrix
                .rows_kept
                .iter()
                .flat_map(|r| r.start as usize..r.end as usize)
                .collect(),
            k_start: matrix.k_kept.start as usize,
            k: (matrix.k_kept.end - matrix.k_kept.start) as usize,
            padded_rows: plan.rows as usize,
            padded_k: plan.k as usize,
            nibble_order: NibbleOrder::EvenLow,
        };
        builder
            .write_matrix(matrix.expert, matrix.projection, view(matrix), &slice)
            .unwrap();
    }
    let expected = builder.finish().unwrap();
    let mut sources: Vec<Vec<u8>> = Vec::new();
    let mut described = Vec::new();
    for matrix in matrices {
        let mut push = |bytes: &Vec<u8>| {
            sources.push(bytes.clone());
            u32::try_from(sources.len() - 1).unwrap()
        };
        let codes = push(&matrix.codes);
        let local = matrix.local.as_ref().map(|(_, bytes)| push(bytes));
        let global = matrix.global.as_ref().map(&mut push);
        described.push(PackedMatrixSource {
            expert: matrix.expert,
            projection: matrix.projection,
            format: matrix.format,
            rows: matrix.rows as u64,
            k: matrix.k as u64,
            row_origin: 0,
            k_origin: 0,
            nibble_order: NibbleOrder::EvenLow,
            codes,
            local,
            local_dtype: matrix.local.as_ref().map(|(dtype, _)| *dtype),
            global,
            source_expert: 0,
            source_experts: 1,
            rows_kept: matrix.rows_kept.clone(),
            k_kept: matrix.k_kept.clone(),
        });
    }
    for (section, expected) in [
        (PackedSection::Codes, &expected.codes),
        (PackedSection::LocalScales, &expected.local_scales),
        (PackedSection::GlobalScales, &expected.global_scales),
    ] {
        let operand = PackedOperand {
            local_encoding: None,
            plan: plan.clone(),
            section,
            decode_order: order,
            matrices: described.clone(),
        };
        let decoded = PackedOperand::decode(&operand.encode().unwrap()).unwrap();
        assert_eq!(decoded, operand);
        let record = WeightRecord {
            transform: WeightTransform::PackedOperand(Box::new(operand.clone())),
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
        let views: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
        assert_eq!(
            &operand.execute(&record, &views).unwrap(),
            expected,
            "{:?} {} {section:?}",
            plan.format,
            plan.layout.id
        );
    }
}

fn e4m3(i: usize) -> u8 {
    0x30 + (i % 9) as u8
}

#[test]
fn nvfp4_rows_with_selected_rows_and_padding() {
    let (rows, k) = (40, 64);
    check(
        plan(
            WeightFormat::Nvfp4,
            "row-major",
            32,
            64,
            Some(DType::Fp8E4M3),
        ),
        DecodeOrder::ScaleThenCode,
        &[Matrix {
            expert: 0,
            projection: 0,
            format: WeightFormat::Nvfp4,
            rows,
            k,
            codes: bytes(rows * k / 2, 1, |i| i as u8),
            local: Some((DType::Fp8E4M3, bytes(rows * k / 16, 2, e4m3))),
            global: Some(1.5_f32.to_le_bytes().to_vec()),
            rows_kept: vec![4..14, 20..32],
            k_kept: 0..64,
        }],
    );
}

#[test]
fn nvfp4_expert_projection_banks_tile_scales_and_keep_globals() {
    let (rows, k) = (32, 128);
    let mut request = plan(
        WeightFormat::Nvfp4,
        "moe-packed-ExpertProjection",
        32,
        128,
        Some(DType::Fp8E4M3),
    );
    request.experts = 2;
    request.projections = 2;
    let matrices: Vec<_> = (0..4)
        .map(|index| Matrix {
            expert: index / 2,
            projection: index % 2,
            format: WeightFormat::Nvfp4,
            rows,
            k,
            codes: bytes(rows * k / 2, 10 + index as usize, |i| i as u8),
            local: Some((
                DType::Fp8E4M3,
                bytes(rows * k / 16, 20 + index as usize, e4m3),
            )),
            global: Some((1.0_f32 + index as f32).to_le_bytes().to_vec()),
            rows_kept: vec![0..32],
            k_kept: 0..128,
        })
        .collect();
    check(request, DecodeOrder::CodeThenScale, &matrices);
}

#[test]
fn mxfp8_rows_with_a_k_slice() {
    let (rows, k) = (24, 128);
    check(
        plan(WeightFormat::Mxfp8, "row-major", 32, 64, Some(DType::Ue8M0)),
        DecodeOrder::ScaleThenCode,
        &[Matrix {
            expert: 0,
            projection: 0,
            format: WeightFormat::Mxfp8,
            rows,
            k,
            codes: bytes(rows * k, 3, |i| (i as u8) & 0x77),
            local: Some((
                DType::Ue8M0,
                bytes(rows * k / 32, 4, |i| 120 + (i % 9) as u8),
            )),
            global: None,
            rows_kept: vec![0..24],
            k_kept: 32..96,
        }],
    );
}

#[test]
fn fp8_block_fragments_widen_bf16_scales() {
    let (rows, k) = (256, 256);
    let format = WeightFormat::Fp8Block128x128(ScaleMode::ExactF32);
    let bf16 = |i: usize| if i % 2 == 0 { 0x80 } else { 0x3f };
    check(
        plan(format, "fp8-fragment", 128, 256, Some(DType::F32)),
        DecodeOrder::BlockScaleAfterMma,
        &[Matrix {
            expert: 0,
            projection: 0,
            format,
            rows,
            k,
            codes: bytes(rows * k, 5, |i| (i as u8) & 0x7e),
            local: Some((DType::Bf16, (0..8).map(bf16).collect())),
            global: None,
            rows_kept: vec![128..256],
            k_kept: 0..256,
        }],
    );
}

#[test]
fn bf16_fragments_pad_rows() {
    let (rows, k) = (20, 32);
    check(
        plan(WeightFormat::Bf16, "bf16-fragment", 32, 32, None),
        DecodeOrder::BlockScaleAfterMma,
        &[Matrix {
            expert: 0,
            projection: 0,
            format: WeightFormat::Bf16,
            rows,
            k,
            codes: bytes(rows * k * 2, 6, |i| (i as u8) & 0x3f),
            local: None,
            global: None,
            rows_kept: vec![0..20],
            k_kept: 0..32,
        }],
    );
}
