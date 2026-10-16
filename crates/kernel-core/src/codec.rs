//! Scalar OCP MX and NVIDIA floating-point codecs. Finite encoders saturate.
//! Every tie uses IEEE round-to-nearest, ties-to-even on the encoded mantissa.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SmallFloat {
    E4M3,
    E5M2,
    E2M1,
    E2M3,
    E3M2,
    Ue4M3,
}

impl SmallFloat {
    fn shape(self) -> (u8, u8, i32) {
        match self {
            Self::E4M3 | Self::Ue4M3 => (4, 3, 7),
            Self::E5M2 => (5, 2, 15),
            Self::E2M1 => (2, 1, 1),
            Self::E2M3 => (2, 3, 1),
            Self::E3M2 => (3, 2, 3),
        }
    }

    fn total_bits(self) -> u8 {
        let (e, m, _) = self.shape();
        e + m + u8::from(self != Self::Ue4M3)
    }

    pub fn max_finite(self) -> f64 {
        let bits = self.total_bits();
        (0..(1_u16 << (bits - u8::from(self != Self::Ue4M3))))
            .map(|v| self.decode_f64(v as u8))
            .filter(|v| v.is_finite())
            .fold(0.0, f64::max)
    }

    /// Decode every bit pattern. Invalid UE4M3 high-bit patterns and NaNs return NaN.
    pub fn decode_f64(self, code: u8) -> f64 {
        let (ebits, mbits, bias) = self.shape();
        let signed = self != Self::Ue4M3;
        let total = self.total_bits();
        if code as u16 >= 1_u16 << total {
            return f64::NAN;
        }
        let sign = signed && code & (1 << (total - 1)) != 0;
        let magnitude = if signed {
            code & ((1 << (total - 1)) - 1)
        } else {
            code
        };
        let exp = (magnitude >> mbits) as i32;
        let mantissa = (magnitude & ((1 << mbits) - 1)) as u32;
        let max_exp = (1 << ebits) - 1;
        if matches!(self, Self::E4M3 | Self::Ue4M3)
            && exp == max_exp
            && mantissa == (1 << mbits) - 1
        {
            return f64::NAN;
        }
        if self == Self::E5M2 && exp == max_exp {
            return if mantissa == 0 {
                if sign {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                }
            } else {
                f64::NAN
            };
        }
        let value = if exp == 0 {
            (mantissa as f64) * 2_f64.powi(1 - bias - i32::from(mbits))
        } else {
            (1.0 + (mantissa as f64) / ((1_u32 << mbits) as f64)) * 2_f64.powi(exp - bias)
        };
        if sign { -value } else { value }
    }

    /// `decode_f64` narrowed to F32, from a table of every code built once
    /// per format: prepared weight checks decode each scale of a checkpoint.
    pub fn decode_f32(self, code: u8) -> f32 {
        use std::sync::OnceLock;
        static TABLES: [OnceLock<[f32; 256]>; 6] = [const { OnceLock::new() }; 6];
        let index = match self {
            Self::E4M3 => 0,
            Self::E5M2 => 1,
            Self::E2M1 => 2,
            Self::E2M3 => 3,
            Self::E3M2 => 4,
            Self::Ue4M3 => 5,
        };
        TABLES[index].get_or_init(|| std::array::from_fn(|code| self.decode_f64(code as u8) as f32))
            [usize::from(code)]
    }

    /// NaN encodes to a canonical positive NaN. Infinity saturates to max finite.
    pub fn encode_f32(self, value: f32) -> u8 {
        if value.is_nan() {
            return match self {
                Self::E5M2 => 0x7e,
                Self::E4M3 | Self::Ue4M3 => 0x7f,
                _ => 0,
            };
        }
        let signed = self != Self::Ue4M3;
        let sign_bit = if signed && value.is_sign_negative() {
            1 << (self.total_bits() - 1)
        } else {
            0
        };
        let x = if signed {
            value.abs() as f64
        } else {
            value.max(0.0) as f64
        };
        if x == 0.0 {
            return sign_bit;
        }
        if x.is_infinite() {
            let magnitude_bits = self.total_bits() - u8::from(signed);
            let max_code = (0..(1_u16 << magnitude_bits))
                .find(|&code| self.decode_f64(code as u8) == self.max_finite())
                .unwrap() as u8;
            return max_code | sign_bit;
        }
        let magnitude_bits = self.total_bits() - u8::from(signed);
        let mut best = 0_u8;
        let mut best_error = f64::INFINITY;
        for code in 0..(1_u16 << magnitude_bits) {
            let decoded = self.decode_f64(code as u8);
            if !decoded.is_finite() {
                continue;
            }
            let error = (x - decoded).abs();
            if error < best_error || (error == best_error && code & 1 == 0 && best & 1 != 0) {
                best = code as u8;
                best_error = error;
            }
        }
        best | sign_bit
    }
}

/// UE8M0 encodes 2^(code-127); 0xFF is NaN and there is no zero code.
pub fn decode_ue8m0_f64(code: u8) -> f64 {
    if code == 0xff {
        f64::NAN
    } else {
        2_f64.powi(i32::from(code) - 127)
    }
}

pub fn decode_ue8m0_f32(code: u8) -> f32 {
    decode_ue8m0_f64(code) as f32
}

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// IEEE RNE, preserving NaN payload's quiet bit.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// One ties-to-even rounding, without an intermediate F32 rounding. Compare
/// the nearest BF16 candidate with its neighbor in F64, where both are exact.
pub fn f64_to_bf16(value: f64) -> u16 {
    let sign = ((value.to_bits() >> 48) as u16) & 0x8000;
    let magnitude = value.abs();
    if magnitude.is_nan() {
        return sign | 0x7fc0;
    }
    // Halfway between the largest finite BF16 and the next exponent.
    if magnitude >= 2_f64.powi(128) - 2_f64.powi(119) {
        return sign | 0x7f80;
    }
    let candidate = f32_to_bf16(magnitude as f32).min(0x7f7f);
    let decoded = f64::from(bf16_to_f32(candidate));
    if decoded == magnitude {
        return sign | candidate;
    }
    let neighbor = if decoded < magnitude {
        candidate + 1
    } else {
        candidate - 1
    };
    let distance = (decoded - magnitude).abs();
    let other = (f64::from(bf16_to_f32(neighbor)) - magnitude).abs();
    sign | if other < distance || (other == distance && neighbor & 1 == 0) {
        neighbor
    } else {
        candidate
    }
}

#[cfg(test)]
mod tests {

    /// The table is `decode_f64` narrowed, bit for bit, for every code of
    /// every format, NaN payloads included.
    #[test]
    fn f32_table_equals_the_f64_decoder_for_every_code() {
        for format in [
            SmallFloat::E4M3,
            SmallFloat::E5M2,
            SmallFloat::E2M1,
            SmallFloat::E2M3,
            SmallFloat::E3M2,
            SmallFloat::Ue4M3,
        ] {
            for code in 0..=u8::MAX {
                assert_eq!(
                    format.decode_f32(code).to_bits(),
                    (format.decode_f64(code) as f32).to_bits(),
                    "{format:?} {code:#04x}"
                );
            }
        }
    }

    use super::*;

    #[test]
    fn known_codepoints() {
        assert_eq!(SmallFloat::E4M3.decode_f64(0x7e), 448.0);
        assert_eq!(SmallFloat::E4M3.encode_f32(449.0), 0x7e);
        assert!(SmallFloat::E4M3.decode_f64(0x7f).is_nan());
        assert_eq!(SmallFloat::E5M2.decode_f64(0x7b), 57344.0);
        assert!(SmallFloat::E5M2.decode_f64(0x7c).is_infinite());
        for (code, value) in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
            .into_iter()
            .enumerate()
        {
            assert_eq!(SmallFloat::E2M1.decode_f64(code as u8), value);
        }
        assert_eq!(decode_ue8m0_f64(127), 1.0);
        assert!(decode_ue8m0_f64(255).is_nan());
        assert_eq!(SmallFloat::Ue4M3.decode_f64(0x7e), 448.0);
        assert!(SmallFloat::Ue4M3.decode_f64(0xfe).is_nan());
    }

    #[test]
    fn exhaustive_decode_and_finite_round_trip() {
        for format in [
            SmallFloat::E4M3,
            SmallFloat::E5M2,
            SmallFloat::E2M1,
            SmallFloat::E2M3,
            SmallFloat::E3M2,
            SmallFloat::Ue4M3,
        ] {
            for code in 0..=255_u8 {
                let value = format.decode_f64(code);
                if value.is_finite() {
                    let back = format.encode_f32(value as f32);
                    assert_eq!(back, code, "{format:?} {code:#x} {value}");
                    assert_eq!(format.decode_f32(code) as f64, value);
                } else if format == SmallFloat::Ue4M3 && code >= 128 {
                    assert!(value.is_nan());
                }
            }
        }
    }

    #[test]
    fn rne_ties_and_saturation() {
        for format in [
            SmallFloat::E4M3,
            SmallFloat::E5M2,
            SmallFloat::E2M1,
            SmallFloat::E2M3,
            SmallFloat::E3M2,
            SmallFloat::Ue4M3,
        ] {
            let limit = 1_u16 << (format.total_bits() - u8::from(format != SmallFloat::Ue4M3));
            for code in 0..limit - 1 {
                let a = format.decode_f64(code as u8);
                let b = format.decode_f64((code + 1) as u8);
                if a.is_finite() && b.is_finite() && b > a {
                    let midpoint = ((a + b) * 0.5) as f32;
                    assert_eq!(
                        format.encode_f32(midpoint),
                        if code & 1 == 0 {
                            code as u8
                        } else {
                            (code + 1) as u8
                        },
                        "{format:?} {code}"
                    );
                }
            }
            assert_eq!(
                format.decode_f64(format.encode_f32(f32::INFINITY)),
                format.max_finite()
            );
        }
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
    }

    #[test]
    fn bf16_all_codes_round_trip_or_quiet_nan() {
        for bits in 0..=u16::MAX {
            let value = bf16_to_f32(bits);
            assert_eq!(value.to_bits(), u32::from(bits) << 16);
            let expected = if value.is_nan() { bits | 0x0040 } else { bits };
            assert_eq!(f32_to_bf16(value), expected);
        }
    }
}
