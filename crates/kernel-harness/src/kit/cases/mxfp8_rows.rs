//! Exact MXFP8 row decode cases, including every code at scale extremes.

use crate::LabResult;
use crate::conformance::{Expected, Rows};
use crate::kit::{OpCase, Required, Spec, bf16_bytes};
use kernel_core::backend::{Need, Prepared, View};
use kernel_core::contract::{DType, Validate};
use kernel_core::mxfp8_rows::{Mxfp8Rows, Mxfp8RowsBuffers, decode_rows};
use kernel_core::properties::Invariance;

pub const SCALES: [u8; 7] = [0, 1, 126, 127, 128, 254, 255];
pub const CAPACITY: u32 = 16;
pub const WIDTHS: [u32; 4] = [32, 128, 256, 6144];

/// Exhaustive cases use width 256, with one scale per complete code row.
/// Random cases vary both planes, including their non-finite encodings.
#[derive(Clone, Copy, Debug)]
pub enum Inputs {
    Exhaustive,
    Random,
}

pub fn planes(op: Mxfp8Rows, rows: u32, inputs: Inputs) -> LabResult<(Vec<u8>, Vec<u8>)> {
    op.validate().map_err(|e| e.to_string())?;
    let width = op.width as usize;
    let n = width * rows as usize;
    match inputs {
        Inputs::Exhaustive => {
            if width != 256 || (rows as usize) < SCALES.len() {
                return Err("exhaustive MXFP8 rows need width 256 and at least seven rows".into());
            }
            Ok((
                (0..n).map(|i| i as u8).collect(),
                (0..n / 32)
                    .map(|i| SCALES[(i / (width / 32)) % SCALES.len()])
                    .collect(),
            ))
        }
        Inputs::Random => {
            let mut state = 0x8c57_3a19_b4d6_021f_u64;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 24) as u8
            };
            let mut codes: Vec<_> = (0..n).map(|_| next()).collect();
            let mut scales: Vec<_> = (0..n / 32).map(|_| next()).collect();
            // Every width has a truncation witness in its first live row.
            if n > 0 {
                codes[0] = 3;
                scales[0] = 1;
            }
            Ok((codes, scales))
        }
    }
}

pub fn case(
    query: Mxfp8Rows,
    capacity: u32,
    inputs: Inputs,
    op: Prepared<Mxfp8RowsBuffers<View>>,
) -> LabResult<OpCase<'static>> {
    let width = query.width as usize;
    let (codes, scales) = planes(query, capacity, inputs)?;
    // Observe packed bits: code words admit 0xff bytes in the PDL producer,
    // and output pairs admit the NaNs and infinities this contract requires.
    // The exact oracle still checks every byte of every BF16 element.
    let roles = Mxfp8RowsBuffers {
        codes: Spec::input(DType::I32, (Rows::Live, width / 4), codes),
        scales: Spec::input(DType::U8, (Rows::Live, width / 32), scales),
        live_rows: Spec::live(),
        output: Spec::output(DType::I32, (Rows::Live, width / 2)),
    };
    Ok(
        OpCase::new(op, capacity, roles)?.oracle(move |images, live| {
            let mut output = vec![0; images.bytes("codes")?.len()];
            decode_rows(
                &query,
                images.bytes("codes")?,
                images.bytes("scales")?,
                live,
                &mut output,
            )
            .map_err(|e| e.to_string())?;
            Ok(vec![("output".into(), Expected::Bits(bf16_bytes(&output)))])
        }),
    )
}

pub fn family() -> Vec<Required<'static>> {
    std::iter::once((256, Inputs::Exhaustive))
        .chain(WIDTHS.map(|width| (width, Inputs::Random)))
        .map(|(width, inputs)| {
            let query = Mxfp8Rows { width };
            Required::new(
                format!("MXFP8 rows {width} {inputs:?}"),
                Need::rows(CAPACITY).with_exact(Invariance::any_rows(CAPACITY)),
                move |backend, need| backend.decode_mxfp8_rows(&query, need),
                move |op| case(query, CAPACITY, inputs, op),
            )
            .required()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::Backend;
    use kernel_core::codec::{SmallFloat, decode_ue8m0_f32};
    use kernel_core::mxfp8_rows::decode_value;
    use kernels_sm120::{backend::Sm120, row::mxfp8_decode::SYMBOL};

    #[test]
    fn exhaustive_inputs_cover_every_code_at_every_required_scale() {
        let (codes, scales) =
            planes(Mxfp8Rows { width: 256 }, CAPACITY, Inputs::Exhaustive).unwrap();
        let pairs = codes
            .iter()
            .enumerate()
            .map(|(i, &code)| (code, scales[i / 32]))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(pairs.len(), 256 * SCALES.len());
        for scale in SCALES {
            for code in 0..=255 {
                assert!(pairs.contains(&(code, scale)));
            }
        }
    }

    #[test]
    fn every_case_declares_and_random_inputs_reject_truncation() {
        let backend = Sm120::new().with_mxfp8_rows(SYMBOL);
        for width in WIDTHS {
            let query = Mxfp8Rows { width };
            let op = backend
                .decode_mxfp8_rows(&query, Need::rows(CAPACITY))
                .unwrap();
            case(query, CAPACITY, Inputs::Random, op)
                .unwrap()
                .validate()
                .unwrap();
            let (codes, scales) = planes(query, CAPACITY, Inputs::Random).unwrap();
            let product = SmallFloat::E4M3.decode_f32(codes[0]) * decode_ue8m0_f32(scales[0]);
            assert_eq!(decode_value(codes[0], scales[0]), 1);
            assert_eq!((product.to_bits() >> 16) as u16, 0);
        }
    }
}
