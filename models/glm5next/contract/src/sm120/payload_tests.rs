//! The payload admission's integer rules (`crate::payload`) against the
//! SM120 chained engine's admission of the same scales and codes.

use kernels_sm120::moe::chained::{admit_scales, formats};

use crate::payload::Check;

/// The integer rules agree with the engines' admission over every
/// UE4M3 code, every MXFP8 code and scale pair, and F32 edges.
#[test]
fn the_integer_rules_are_the_engines_admission() {
    for code in 0..=255_u8 {
        assert_eq!(
            Check::Ue4m3Scales.first_invalid(&[code], &[]).is_some(),
            formats::admit_ue4m3_scales(&[code]).is_err(),
            "{code:#x}"
        );
        for scale in 0..=255_u8 {
            let mut codes = [0_u8; 32];
            codes[7] = code;
            let found = Check::Mxfp8.first_invalid(&codes, &[scale]);
            assert_eq!(
                found.is_some(),
                formats::admit_mxfp8_payload(&codes, &[scale]).is_err(),
                "{code:#x} {scale}"
            );
            if let Some(at) = found {
                // Scale 255 refuses code 0 first.
                assert_eq!(at, if scale == 0xff { 0 } else { 7 });
            }
        }
    }
    for value in [
        0.0_f32,
        -0.0,
        f32::MIN_POSITIVE / 8.0,
        1.0,
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        -1.0,
        -f32::MIN_POSITIVE / 8.0,
    ] {
        let bytes = [1.0_f32, value]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            Check::F32Scales.first_invalid(&bytes, &[]),
            admit_scales(&[1.0, value]).is_err().then_some(1),
            "{value}"
        );
    }
}
