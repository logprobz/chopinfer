//! Decode dense MXFP8 rows to BF16. This conversion has its own contract
//! because row copies preserve bytes without numerical interpretation.

use crate::codec::{SmallFloat, decode_ue8m0_f64, f32_to_bf16};
use crate::contract::{Refusal, Validate};

/// Decode `[rows, width]` E4M3 codes and `[rows, width / 32]` UE8M0 scales
/// to dense BF16 rows. Width is positive and a multiple of 32. Capacity is
/// supplied by `Backend::decode_mxfp8_rows`'s `Need`; the device U32
/// `live_rows` is at most the launch's `Counts::rows`, which is at most
/// that capacity. Rows at or past `live_rows` are never written.
///
/// Numerics (`row.mxfp8-to-bf16.rne.v1`): each value is
/// `BF16_RNE(F32(E4M3(code)) * 2^(scale - 127))`. Finite products within
/// F32's range are exact, including F32 subnormals; overflow gives signed
/// infinity. The only finite rounding is F32 to BF16, nearest, ties even,
/// with gradual underflow. Signed zero is preserved, including code 0x80.
/// E4M3 subnormals are decoded without flushing. E4M3 codes 0x7f and 0xff,
/// and UE8M0 scale 0xff, produce the canonical quiet BF16 NaN 0x7fc0.
/// Every data byte is accepted: there is no trap or status word. A later
/// consumer handles non-finite values.
///
/// The three planes do not overlap. Codes and scales have byte alignment;
/// output and the live count have their natural 2- and 4-byte alignment.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Mxfp8Rows {
    pub width: u32,
}

impl Validate for Mxfp8Rows {
    fn validate(&self) -> Result<(), Refusal> {
        if self.width == 0 || !self.width.is_multiple_of(32) {
            return Err(Refusal("MXFP8 rows need a positive multiple of 32 columns"));
        }
        Ok(())
    }
}

impl Mxfp8Rows {
    pub fn code_bytes(&self, rows: u32) -> Result<u64, Refusal> {
        self.validate()?;
        Ok(u64::from(self.width) * u64::from(rows))
    }

    pub fn scale_bytes(&self, rows: u32) -> Result<u64, Refusal> {
        Ok(self.code_bytes(rows)? / 32)
    }

    pub fn output_bytes(&self, rows: u32) -> Result<u64, Refusal> {
        self.code_bytes(rows)?
            .checked_mul(2)
            .ok_or(Refusal("MXFP8 row output exceeds u64 bytes"))
    }
}

crate::scratch_record! {
    /// Dense code and scale planes, a device U32 live-row count, and BF16
    /// output. Input planes and output cover the launch's rows.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct Mxfp8RowsBuffers {
        pub codes: value,
        pub scales: value,
        pub live_rows: value,
        pub output: value,
    }
}

/// Exact scalar oracle, with a canonical NaN independent of host payloads.
pub fn decode_value(code: u8, scale: u8) -> u16 {
    if code & 0x7f == 0x7f || scale == 0xff {
        return 0x7fc0;
    }
    let product = SmallFloat::E4M3.decode_f64(code) * decode_ue8m0_f64(scale);
    f32_to_bf16(product as f32)
}

/// Decode only the live rows, preserving every other output bit. Refuses
/// malformed shapes and short buffers, never any code or scale value.
pub fn decode_rows(
    op: &Mxfp8Rows,
    codes: &[u8],
    scales: &[u8],
    live_rows: u32,
    output: &mut [u16],
) -> Result<(), Refusal> {
    let elements = op.code_bytes(live_rows)?;
    if (codes.len() as u64) < elements
        || (scales.len() as u64) < op.scale_bytes(live_rows)?
        || (output.len() as u64) < elements
    {
        return Err(Refusal("MXFP8 row buffers do not hold the live rows"));
    }
    for (at, value) in output.iter_mut().take(elements as usize).enumerate() {
        *value = decode_value(codes[at], scales[at / 32]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{decode_ue8m0_f32, f64_to_bf16};

    #[test]
    fn every_code_and_scale_matches_f32_and_direct_f64_rounding() {
        let mut checked = 0;
        for code in 0..=255 {
            for scale in 0..=255 {
                let actual = decode_value(code, scale);
                if code & 0x7f == 0x7f || scale == 255 {
                    assert_eq!(actual, 0x7fc0);
                } else {
                    let f32_product = SmallFloat::E4M3.decode_f32(code) * decode_ue8m0_f32(scale);
                    let f64_product = SmallFloat::E4M3.decode_f64(code) * decode_ue8m0_f64(scale);
                    assert_eq!(actual, f32_to_bf16(f32_product), "{code:#x} {scale}");
                    assert_eq!(actual, f64_to_bf16(f64_product), "{code:#x} {scale}");
                }
                checked += 1;
            }
        }
        assert_eq!(checked, 65536);
    }

    #[test]
    fn rne_differs_from_truncation_and_keeps_even_ties_and_signed_zero() {
        // 3 * 2^-135 is 0.75 of the least BF16 subnormal, 2^-133.
        let product = SmallFloat::E4M3.decode_f32(3) * decode_ue8m0_f32(1);
        let truncated = (product.to_bits() >> 16) as u16;
        assert_eq!(truncated, 0);
        assert_eq!(decode_value(3, 1), 1);
        assert_ne!(decode_value(3, 1), truncated);
        assert_eq!(decode_value(2, 1), 0); // 0.5 rounds to even zero.
        assert_eq!(decode_value(6, 1), 2); // 1.5 rounds to even two.
        assert_eq!(decode_value(0x83, 1), 0x8001);
        assert_eq!(decode_value(0x80, 254), 0x8000);
        assert_eq!(decode_value(1, 127), 0x3b00);
        assert_eq!(decode_value(0x7e, 254), 0x7f80);
        assert_eq!(decode_value(0xfe, 254), 0xff80);
    }

    #[test]
    fn rows_use_their_own_scales_and_leave_the_tail_poisoned() {
        let op = Mxfp8Rows { width: 64 };
        let mut codes = vec![0x38; 3 * 64];
        codes[0] = 0x80;
        codes[1] = 0x7f;
        let scales = [127, 128, 126, 255, 0, 1];
        let mut output = vec![0xa5a5; 3 * 64];
        decode_rows(&op, &codes, &scales, 2, &mut output).unwrap();
        assert_eq!(&output[..3], &[0x8000, 0x7fc0, 0x3f80]);
        assert_eq!(output[32], 0x4000);
        assert_eq!(output[64], 0x3f00);
        assert_eq!(output[96], 0x7fc0);
        assert!(output[128..].iter().all(|&v| v == 0xa5a5));
        let before = output.clone();
        decode_rows(&op, &[], &[], 0, &mut output).unwrap();
        assert_eq!(output, before);
        assert!(decode_rows(&op, &codes[..127], &scales, 2, &mut output).is_err());
        assert!(decode_rows(&op, &codes, &scales[..3], 2, &mut output).is_err());
        assert!(decode_rows(&op, &codes, &scales, 2, &mut output[..127]).is_err());
        for width in [0, 1, 31, 33] {
            assert!(Mxfp8Rows { width }.validate().is_err());
        }
        assert!(
            Mxfp8Rows {
                width: u32::MAX - 31
            }
            .output_bytes(u32::MAX)
            .is_err()
        );
    }
}
