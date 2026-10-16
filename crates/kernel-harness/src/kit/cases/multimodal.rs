//! Cases of the multimodal encoder stages, with the SM120 selection of the
//! lab's vision row entries (`families::encoder_vision`), the only encoder
//! entries a lab cubin holds: the zero-column pad, the per-head q/k RMSNorm,
//! the clamped SwiGLU at two widths and the F32 GELU. The pad compares as
//! exact bits (each row's bytes copied, +0.0 appended); every other stage
//! within its F64 oracle's tolerance (`encoder_oracle`).

use super::super::{OpCase, Required, Spec, bf16_bytes, bf16_values};
use crate::LabResult;
use crate::conformance::{Expected, Images, Rows, Write};
use crate::poison::WriteMap;
use kernel_core::backend::{Need, Prepared, View};
use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::{DType, MultimodalQuery};
use kernel_core::encoder_oracle::{clamped_swiglu_rows, gelu_rows_f32, qk_rms_norm};
use kernel_core::multimodal::{EncoderHeadBuffers, EncoderRowBuffers};
use kernel_core::oracle::OracleOutput;
use kernels_sm120::backend::Sm120;

/// The token rows every encoder case serves.
pub const CAPACITY: u32 = 16;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `count` F32 values of the kit's BF16 generator times `scale`.
fn f32_values(seed: u64, count: usize, scale: f32) -> Vec<f32> {
    bf16_values(seed, count)
        .into_iter()
        .map(|b| bf16_to_f32(b) * scale)
        .collect()
}

fn widen(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v)).collect()
}

/// The case of a row stage over `capacity` rows: F32 input rows from the
/// kit's generator (times 8 for the clamped SwiGLU, so some gates and ups
/// pass the clamp) and the stage's oracle.
pub fn rows_case<'a>(
    op: MultimodalQuery,
    capacity: u32,
    prepared: Prepared<EncoderRowBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let (input, output, scale) = match op {
        MultimodalQuery::PadColumnsF32 { input, output, .. } => (input, output, 1.0),
        MultimodalQuery::SwigluClampedF32 { hidden, .. } => (2 * hidden, hidden, 8.0),
        MultimodalQuery::GeluRowsF32 { width, .. } => (width, width, 1.0),
        _ => return Err(format!("the kit has no encoder rows case of {op:?}")),
    };
    let (input, output, rows) = (input as usize, output as usize, capacity as usize);
    let roles = EncoderRowBuffers {
        input: Spec::input(
            DType::F32,
            (Rows::Live, input),
            f32_bytes(&f32_values(51, rows * input, scale)),
        ),
        output: Spec::output(DType::F32, (Rows::Live, output)),
        live_rows: Spec::live(),
        weight: None,
        bias: None,
        residual: None,
        indices: None,
        table: None,
        blend: None,
    };
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        // The pad copies each row's bytes and appends +0.0 bits: exact.
        if let MultimodalQuery::PadColumnsF32 { .. } = op {
            let bytes = images.bytes("input")?;
            let padded = bytes
                .chunks_exact(4 * input)
                .flat_map(|row| {
                    row.iter()
                        .copied()
                        .chain(std::iter::repeat_n(0, 4 * (output - input)))
                })
                .collect();
            return Ok(vec![("output".into(), Expected::Bits(padded))]);
        }
        let x = widen(&images.f32s("input")?);
        let result = match op {
            MultimodalQuery::SwigluClampedF32 { limit_bits, .. } => {
                clamped_swiglu_rows(&x, output, f64::from(f32::from_bits(limit_bits)))
            }
            _ => gelu_rows_f32(&x),
        }
        .map_err(|e| e.0.to_string())?;
        Ok(vec![("output".into(), Expected::Values(result))])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// The case of the per-head q/k RMSNorm over `capacity` rows of packed
/// q, k, v heads, rewritten in place: F32 rows and BF16 weights from the
/// kit's generator, the q and k columns of the live rows read and
/// rewritten (read-modify-write ranges, which keep their image) and every
/// other byte kept, against `encoder_oracle::qk_rms_norm`.
pub fn qk_norm_case<'a>(
    op: MultimodalQuery,
    capacity: u32,
    prepared: Prepared<EncoderHeadBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let MultimodalQuery::QkRmsNormF32 {
        q_heads,
        kv_heads,
        head_dim,
        eps_bits,
        ..
    } = op
    else {
        return Err(format!("the kit has no encoder heads case of {op:?}"));
    };
    let (q, kv, dim) = (q_heads as usize, kv_heads as usize, head_dim as usize);
    let width = (q + 2 * kv) * dim;
    let rows = capacity as usize;
    let normed = (q + kv) * dim;
    let roles = EncoderHeadBuffers {
        // The launch reads each live row's q and k and rewrites them in
        // place, so they keep their image (nothing is poisoned) and the
        // oracle's whole state judges them; v and the rows past the live
        // count must keep their bytes.
        rows: Spec::state_updating(
            DType::F32,
            (Rows::Live, width),
            f32_bytes(&f32_values(52, rows * width, 1.0)),
            Write::Map(Box::new(|_, _| WriteMap::of(DType::F32).none())),
            move |_, live| {
                Ok((0..live as usize)
                    .map(|row| row * width * 4..(row * width + normed) * 4)
                    .collect())
            },
        ),
        live_rows: Spec::live(),
        cos: None,
        sin: None,
        q_weight: Some(Spec::input(
            DType::Bf16,
            (Rows::Fixed(1), dim),
            bf16_bytes(&bf16_values(53, dim)),
        )),
        k_weight: Some(Spec::input(
            DType::Bf16,
            (Rows::Fixed(1), dim),
            bf16_bytes(&bf16_values(54, dim)),
        )),
    };
    let eps = f64::from(f32::from_bits(eps_bits));
    let oracle = move |images: &Images, live: u32| -> LabResult<Vec<(String, Expected)>> {
        let x = widen(&images.f32s("rows")?);
        let weight = |path| -> LabResult<Vec<f64>> {
            Ok(images
                .u16s(path)?
                .into_iter()
                .map(|b| f64::from(bf16_to_f32(b)))
                .collect())
        };
        let live = live as usize;
        // The live rows normed, every other element as it was.
        let mut values = OracleOutput::new(x.clone(), x.iter().map(|v| v.abs()).collect(), 1)
            .map_err(|e| e.0.to_string())?;
        if live > 0 {
            let normed = qk_rms_norm(
                &x[..live * width],
                q,
                kv,
                dim,
                &weight("q_weight")?,
                &weight("k_weight")?,
                eps,
            )
            .map_err(|e| e.0.to_string())?;
            values.values[..live * width].copy_from_slice(&normed.values);
            values.sum_abs_products[..live * width].copy_from_slice(&normed.sum_abs_products);
            values.reduction_lengths[..live * width].copy_from_slice(&normed.reduction_lengths);
        }
        Ok(vec![("rows".into(), Expected::Values(values))])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// SM120 over the vision row `entries`.
pub fn sm120(entries: Vec<kernels_sm120::encoder::Entry>) -> Sm120 {
    Sm120::new().with_encoders(entries)
}

/// Every vision row entry's stage at [`CAPACITY`] rows, each required.
pub fn required<'a>(entries: &[kernels_sm120::encoder::Entry]) -> Vec<Required<'a>> {
    let need = Need::rows(CAPACITY);
    let mut ops = Vec::new();
    for entry in entries {
        let op = entry.query;
        let name = format!("encoder {op:?}");
        let required = match op {
            MultimodalQuery::QkRmsNormF32 { .. } => Required::new(
                name,
                need,
                move |backend, need| backend.encoder_heads(&op, need),
                move |prepared| qk_norm_case(op, need.capacity, prepared),
            ),
            _ => Required::new(
                name,
                need,
                move |backend, need| backend.encoder_rows(&op, need),
                move |prepared| rows_case(op, need.capacity, prepared),
            ),
        };
        ops.push(required.required());
    }
    ops
}
