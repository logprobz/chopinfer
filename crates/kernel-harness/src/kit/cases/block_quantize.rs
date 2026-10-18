//! Bitwise cases for trained BF16 quantizer inputs, including both FP4 floors.

pub mod sequence;

use crate::LabResult;
use crate::conformance::{Expected, Images, Rows, Write};
use crate::kit::{OpCase, Required, Spec};
use crate::poison::WriteMap;
use kernel_core::backend::{Need, Prepared, View};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::DType;
use kernel_core::precision::quantizer::{
    BlockQuantization, BlockQuantizeBuffers, BlockQuantizeQuery, QuantizeOutput,
};
use kernel_core::properties::Invariance;
use kernel_core::quantize::Scales;
use kernels_sm120::backend::Sm120;
use kernels_sm120::row::block_quantize::{self, LAB_CAPACITY};

fn input(query: BlockQuantizeQuery, rows: u32) -> Vec<f32> {
    let group = query.format.group_size() as usize;
    (0..rows as usize * query.width as usize)
        .map(|at| {
            let kind = (at / group) % 12;
            let lane = at % group;
            let sign = if lane % 2 == 0 { 1.0 } else { -1.0 };
            let floor = match query.format {
                BlockQuantization::E4M3Ceil32 { amax_floor_bits } => {
                    f32::from_bits(amax_floor_bits)
                }
                BlockQuantization::E2M1E4M3Scale16 => 6.0 * 2_f32.powi(-9),
                BlockQuantization::E2M1Ceil32 => 6.0 * f32::MIN_POSITIVE,
            };
            sign * match kind {
                0 => 0.0,
                1 => f32::from_bits(floor.to_bits() - 1),
                2 => floor,
                3 => f32::from_bits(floor.to_bits() + 1),
                4 => 448.03125,
                5 => f32::from_bits(448_f32.to_bits() + 1),
                6 => f32::from_bits(448_f32.to_bits() - 1),
                7 => {
                    if lane < 2 {
                        1.0
                    } else {
                        2_f32.powi(-17)
                    }
                }
                8 => floor * (lane % 7) as f32 / 12.0,
                9 => ((at * 17 % 997) as f32 + 0.37) / 13.0,
                10 => 6.0 * 2_f32.powi(lane as i32 - 16),
                _ => {
                    if lane % 3 == 0 {
                        1.0625
                    } else {
                        1.1875
                    }
                }
            }
        })
        .collect()
}

fn expected(
    query: BlockQuantizeQuery,
    images: &Images,
    live: u32,
) -> LabResult<Vec<(String, Expected)>> {
    let values = if query.input == DType::F32 {
        images.f32s("input")?
    } else {
        images.u16s("input")?.into_iter().map(bf16_to_f32).collect()
    };
    let prior = images
        .bytes("status")?
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    let (output, scales, status) = expected_rows(query, &values, &prior, live)?;
    let mut result = vec![
        ("output".into(), Expected::Bits(output)),
        (
            "status".into(),
            Expected::Bits(status.iter().flat_map(|v| v.to_le_bytes()).collect()),
        ),
    ];
    if query.output == QuantizeOutput::Codes {
        result.push(("scales".into(), Expected::Bits(scales)));
    }
    Ok(result)
}

pub fn expected_rows(
    query: BlockQuantizeQuery,
    values: &[f32],
    prior: &[u32],
    live: u32,
) -> LabResult<(Vec<u8>, Vec<u8>, Vec<u32>)> {
    let width = query.width as usize;
    let rows = values.len() / width;
    let mut status = prior.to_vec();
    let flags = query
        .row_status(&values[..live as usize * width], &status[..live as usize])
        .map_err(|e| e.to_string())?;
    status[..live as usize].copy_from_slice(&flags);
    let mut output = Vec::new();
    let mut scales = Vec::new();
    for row in 0..rows {
        if status[row] != 0 || row >= live as usize {
            let bytes = if query.output == QuantizeOutput::Bf16 {
                width * 2
            } else if matches!(query.format, BlockQuantization::E4M3Ceil32 { .. }) {
                width
            } else {
                width / 2
            };
            output.extend(vec![0; bytes]);
            scales.extend(vec![
                query.invalid_scale_code();
                width / query.format.group_size() as usize
            ]);
        } else if query.output == QuantizeOutput::Bf16 {
            output.extend(
                query
                    .format
                    .fake_quant(&values[row * width..(row + 1) * width], 1, width)
                    .map_err(|e| e.to_string())?
                    .iter()
                    .flat_map(|v| v.to_le_bytes()),
            );
        } else {
            let q = query
                .format
                .apply(&values[row * width..(row + 1) * width], 1, width)
                .map_err(|e| e.to_string())?;
            output.extend(q.codes);
            match q.scales {
                Scales::Ue8M0(v) | Scales::Ue4M3(v) => scales.extend(v),
                _ => return Err("byte scales required".into()),
            }
        }
    }
    Ok((output, scales, status))
}

pub fn case<'a>(
    query: BlockQuantizeQuery,
    capacity: u32,
    op: Prepared<BlockQuantizeBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    case_with_status(query, capacity, op, false)
}

pub fn case_with_status<'a>(
    query: BlockQuantizeQuery,
    capacity: u32,
    op: Prepared<BlockQuantizeBuffers<View>>,
    invalid: bool,
) -> LabResult<OpCase<'a>> {
    let mut values = input(query, capacity);
    let mut initial = vec![0u32; capacity as usize];
    if invalid {
        let width = query.width as usize;
        for row in 0..capacity as usize {
            let x = &mut values[row * width..(row + 1) * width];
            x.fill(1.0);
            match row % 16 {
                0 => x[0] = f32::NAN,
                1 => x[width - 1] = f32::INFINITY,
                2 => x[width / 2] = f32::NEG_INFINITY,
                3 => x[width - 1] = 2704.,
                4 => x[0] = 2688.,
                5 => x[0] = f32::from_bits(2688_f32.to_bits() + 1),
                6 => x[0] = f32::MAX,
                7 => x[0] = bf16_to_f32(0x7f7f),
                8 => {
                    initial[row] = kernel_core::precision::quantizer::SCALE_OVERFLOW;
                    x[0] = f32::NAN;
                    x[width - 1] = bf16_to_f32(0x7f7f);
                }
                9 => initial[row] = 0x8000_0000,
                10 => {
                    initial[row] = 1;
                    x[width - 1] = 2704.;
                }
                11 => x.fill(-0.0),
                12 => x.fill(1.125),
                13 => x.fill(bf16_to_f32(1)),
                14 => initial[row] = 2,
                _ => x[width - 1] = f32::NEG_INFINITY,
            }
        }
    }
    let bytes = if query.input == DType::F32 {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else {
        values
            .iter()
            .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
            .collect()
    };
    let (output_type, output_width) = match (query.output, query.format) {
        (QuantizeOutput::Bf16, _) => (DType::Bf16, query.width),
        (_, BlockQuantization::E4M3Ceil32 { .. }) => (DType::U8, query.width),
        _ => (DType::U8, query.width / 2),
    };
    let roles = BlockQuantizeBuffers {
        status_row_offset: 0,
        input: Spec::input(query.input, (Rows::Live, query.width as usize), bytes),
        output: Spec::output(output_type, (Rows::Live, output_width as usize)),
        scales: (query.output == QuantizeOutput::Codes).then(|| {
            Spec::output(
                DType::U8,
                (
                    Rows::Live,
                    (query.width / query.format.group_size()) as usize,
                ),
            )
        }),
        live_rows: Spec::live(),
        status: status_role(&initial),
    };
    Ok(OpCase::new(op, capacity, roles)?
        .oracle(move |images, live| expected(query, images, live))
        .live_counts(vec![0, 1, 3, capacity - 1, capacity]))
}

pub fn sm120() -> Sm120 {
    block_quantize::lab_entries()
        .into_iter()
        .fold(Sm120::new(), Sm120::with_block_quantize)
}

pub fn required<'a>() -> Vec<Required<'a>> {
    block_quantize::lab_entries()
        .into_iter()
        .map(|entry| {
            let query = entry.query;
            Required::new(
                format!("block quantize {query:?}"),
                Need::rows(LAB_CAPACITY).with_exact(Invariance::any_rows(LAB_CAPACITY)),
                move |backend, need| backend.block_quantize(&query, need),
                move |op| case(query, LAB_CAPACITY, op),
            )
            .required()
        })
        .collect()
}

pub fn invalid_required<'a>() -> Vec<Required<'a>> {
    block_quantize::lab_entries()
        .into_iter()
        .map(|entry| {
            let query = entry.query;
            Required::new(
                format!("block status {query:?}"),
                Need::rows(LAB_CAPACITY).with_exact(Invariance::any_rows(LAB_CAPACITY)),
                move |backend, need| backend.block_quantize(&query, need),
                move |op| case_with_status(query, LAB_CAPACITY, op, true),
            )
            .required()
        })
        .collect()
}

pub fn status_role<'a>(initial: &[u32]) -> Spec<'a> {
    Spec::state_updating(
        DType::I32,
        (Rows::Live, 1),
        initial.iter().flat_map(|v| v.to_le_bytes()).collect(),
        Write::Map(Box::new(|_, _| WriteMap::of(DType::I32).none())),
        |_, live| Ok((0..live as usize).map(|r| r * 4..r * 4 + 4).collect()),
    )
}
