//! Bitwise cases for trained BF16 quantizer inputs, including both FP4 floors.

use crate::LabResult;
use crate::conformance::{Expected, Images, Rows};
use crate::kit::{OpCase, Required, Spec};
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

fn expected(query: BlockQuantizeQuery, images: &Images) -> LabResult<Vec<(String, Expected)>> {
    let values = if query.input == DType::F32 {
        images.f32s("input")?
    } else {
        images.u16s("input")?.into_iter().map(bf16_to_f32).collect()
    };
    let width = query.width as usize;
    let rows = values.len() / width;
    if query.output == QuantizeOutput::Bf16 {
        let output = query
            .format
            .fake_quant(&values, rows, width)
            .map_err(|e| e.to_string())?;
        return Ok(vec![(
            "output".into(),
            Expected::Bits(output.iter().flat_map(|v| v.to_le_bytes()).collect()),
        )]);
    }
    let quantized = query
        .format
        .apply(&values, rows, width)
        .map_err(|e| e.to_string())?;
    let scales = match quantized.scales {
        Scales::Ue8M0(bytes) | Scales::Ue4M3(bytes) => bytes,
        _ => return Err("block quantization case needs byte scales".into()),
    };
    Ok(vec![
        ("output".into(), Expected::Bits(quantized.codes)),
        ("scales".into(), Expected::Bits(scales)),
    ])
}

pub fn case<'a>(
    query: BlockQuantizeQuery,
    capacity: u32,
    op: Prepared<BlockQuantizeBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let values = input(query, capacity);
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
    };
    Ok(OpCase::new(op, capacity, roles)?
        .oracle(move |images, _| expected(query, images))
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
