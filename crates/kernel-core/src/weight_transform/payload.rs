//! Numerical admission of record payloads while they stream.
//!
//! A gathering record (copy, widening, narrowing, fragments, scale tiles,
//! MXFP4 strips) checks the elements whose domain its transform or source
//! dtype states, as stored, without decoding them. Other transforms
//! (swapped axes, inline bytes, bundle transforms) are not checked here; a
//! packed operand checks its codes and scales through the precision rules.
//!
//! - an F8_E4M3 source, or FP8 fragments, hold no NaN code (`0x7f`, `0xff`);
//! - MXFP4 strip scales are UE8M0 bytes of at most 252. At 253 and 254 only
//!   small codes stay finite in BF16 and 255 is NaN (the device conversion
//!   traps on them); the codes are another record, so these bytes are refused;
//! - [`super::WeightTransform::CopyScales`] scales are finite with a clear
//!   sign bit, as packed operands require (`crate::precision::packed`).
//!
//! Zero padding is valid in every domain.

use super::{WeightTransformError, refuse};

/// What a record checks in the stored bytes it produces.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Payload {
    /// Nothing: the record's domain is not stated.
    Unchecked,
    /// E4M3 codes or scales: no NaN code.
    E4m3,
    /// MXFP4 UE8M0 strip scales: at most 252.
    Mxfp4Scales,
    /// Scales of `bytes` per element (4 F32, 2 BF16, 1 E4M3): finite, sign
    /// bit clear.
    Scales { bytes: u8 },
}

/// The largest MXFP4 scale byte whose every code is a finite BF16.
pub const MXFP4_SCALE_MAX: u8 = 252;

/// The source dtype's element width as a checked scale.
pub(crate) fn scale_bytes(dtype: &str) -> Option<u8> {
    match dtype {
        "F32" => Some(4),
        "BF16" => Some(2),
        "F8_E4M3" => Some(1),
        _ => None,
    }
}

impl Payload {
    /// Bytes of one checked element; a part checked on its own starts and
    /// ends on them.
    #[must_use]
    pub const fn element_bytes(self) -> u64 {
        match self {
            Self::Scales { bytes } => bytes as u64,
            _ => 1,
        }
    }

    /// Checks `bytes`, whole elements starting at stored byte `first`.
    pub fn check(self, bytes: &[u8], first: u64) -> Result<(), WeightTransformError> {
        let at = |index: usize| first + index as u64;
        match self {
            Self::Unchecked => Ok(()),
            Self::E4m3 => match find(bytes, |b| b & 0x7f == 0x7f) {
                Some(i) => refuse(format!(
                    "E4M3 element at stored byte {} is the NaN code {:#04x}",
                    at(i),
                    bytes[i]
                )),
                None => Ok(()),
            },
            Self::Mxfp4Scales => match find(bytes, |b| b > MXFP4_SCALE_MAX) {
                Some(i) => refuse(format!(
                    "MXFP4 scale at stored byte {} is {}: above {MXFP4_SCALE_MAX} it is NaN \
                     or leaves codes outside BF16 range",
                    at(i),
                    bytes[i]
                )),
                None => Ok(()),
            },
            Self::Scales { bytes: width } => {
                let width = usize::from(width);
                if !bytes.len().is_multiple_of(width) {
                    return refuse("checked scales are not whole elements");
                }
                // The sign bit and the all-ones exponent of each element's
                // top half (an E4M3 byte: sign, then 0x7f is its NaN).
                let invalid = |top: u16| match width {
                    4 | 2 => top & 0x8000 != 0 || top & 0x7f80 == 0x7f80,
                    _ => top > 0x7e,
                };
                let top = |i: usize| {
                    let element = &bytes[i * width..(i + 1) * width];
                    match width {
                        1 => u16::from(element[0]),
                        _ => u16::from_le_bytes([element[width - 2], element[width - 1]]),
                    }
                };
                match (0..bytes.len() / width).find(|&i| invalid(top(i))) {
                    Some(i) => refuse(format!(
                        "scale at stored byte {} is negative or not finite",
                        at(i * width)
                    )),
                    None => Ok(()),
                }
            }
        }
    }
}

/// The first byte `bad` holds for. Clean blocks are screened without
/// branching, which compiles to vector code for the large code payloads.
fn find(bytes: &[u8], bad: impl Fn(u8) -> bool + Copy) -> Option<usize> {
    const BLOCK: usize = 4096;
    for (index, block) in bytes.chunks(BLOCK).enumerate() {
        if block.iter().fold(false, |any, &b| any | bad(b)) {
            return block
                .iter()
                .position(|&b| bad(b))
                .map(|i| index * BLOCK + i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_refuses_only_its_nan_codes() {
        let valid: Vec<u8> = (0..=255).filter(|b| b & 0x7f != 0x7f).collect();
        Payload::E4m3.check(&valid, 0).unwrap();
        for nan in [0x7f, 0xff] {
            let mut bytes = vec![0x3c; 9000];
            bytes[8193] = nan;
            let error = Payload::E4m3.check(&bytes, 7).unwrap_err().0;
            assert!(error.contains("stored byte 8200"), "{error}");
        }
    }

    #[test]
    fn mxfp4_scales_stop_at_252() {
        Payload::Mxfp4Scales
            .check(&(0..=252).collect::<Vec<u8>>(), 0)
            .unwrap();
        for scale in [253, 254, 255] {
            assert!(Payload::Mxfp4Scales.check(&[127, scale], 0).is_err());
        }
    }

    #[test]
    fn scales_are_finite_with_a_clear_sign_bit() {
        let f32s =
            |values: &[f32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
        let f32_scales = Payload::Scales { bytes: 4 };
        f32_scales
            .check(&f32s(&[0.0, 1.5e-38, 2.0, f32::MAX, 1e-45]), 0)
            .unwrap();
        for bad in [-0.0, -1.0, f32::INFINITY, f32::NAN, -f32::NAN] {
            let error = f32_scales.check(&f32s(&[1.0, bad]), 64).unwrap_err().0;
            assert!(error.contains("stored byte 68"), "{error}");
        }
        assert!(f32_scales.check(&[0; 6], 0).is_err());
        let bf16_scales = Payload::Scales { bytes: 2 };
        bf16_scales.check(&[0x80, 0x3f, 0x7f, 0x7f], 0).unwrap();
        for bad in [[0x80, 0xbf], [0x80, 0x7f], [0xc1, 0x7f], [0x00, 0x80]] {
            assert!(bf16_scales.check(&bad, 0).is_err(), "{bad:?}");
        }
        let e4m3_scales = Payload::Scales { bytes: 1 };
        e4m3_scales
            .check(&(0..=0x7e).collect::<Vec<u8>>(), 0)
            .unwrap();
        for bad in [0x7f, 0x80, 0xb8, 0xff] {
            assert!(e4m3_scales.check(&[0x38, bad], 0).is_err(), "{bad:#x}");
        }
    }
}
