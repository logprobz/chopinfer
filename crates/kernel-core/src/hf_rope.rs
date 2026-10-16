//! HF text RoPE dtype boundaries for 64-dimensional BF16 rotations.
//! Inverse frequencies captured from Torch 2.14 CPU's F32 pow and reciprocal.
use crate::codec::{bf16_to_f32, f32_to_bf16};
use crate::rotary::{AngleRule, Axes, ComputedTable, RotaryArithmetic};

/// Arithmetic identity for capture and arbiter provenance.
pub const REFERENCE: &str = "hf-f32-angle-bf16-cossin-f32-rotation-v1";

pub const INVERSE_BITS: [[u32; 32]; 2] = [
    [
        0x3f800000, 0x3f3ff911, 0x3f0ff59a, 0x3ed7e89b, 0x3ea1e89b, 0x3e72d423, 0x3e361887,
        0x3e088d77, 0x3dcccccd, 0x3d99940d, 0x3d6655c2, 0x3d2cba15, 0x3d0186e3, 0x3cc2434f,
        0x3c91ad39, 0x3c5a7bf2, 0x3c23d70a, 0x3bf5b9b0, 0x3bb8449c, 0x3b8a2e77, 0x3b4f3e38,
        0x3b1b690d, 0x3ae91528, 0x3aaec98e, 0x3a83126f, 0x3a44948c, 0x3a136a16, 0x39dd1725,
        0x39a5cb60, 0x3978a815, 0x393a7753, 0x390bd472,
    ],
    [
        0x3f800000, 0x3f1ab32b, 0x3ebaf81b, 0x3e61f835, 0x3e088d77, 0x3da50956, 0x3d47763f,
        0x3cf11177, 0x3c91ad39, 0x3c301052, 0x3bd4ca15, 0x3b80967d, 0x3b1b690d, 0x3abbd3ed,
        0x3a6301e2, 0x3a092e02, 0x39a5cb60, 0x394860c1, 0x38f22ce2, 0x3892587e, 0x3830df52,
        0x37d5c441, 0x37812dab, 0x371c1fc4, 0x36bcb0c1, 0x36640cc6, 0x3609cf4a, 0x35a68e4c,
        0x35494c57, 0x34f3499d, 0x3493048e, 0x3431af44,
    ],
];

pub fn round_bf16(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(x))
}

/// The `AngleRule::HfF32AngleBf16CosSin` entry of one position and pair.
pub fn cos_sin(position: usize, pair: usize, theta: f64) -> (f64, f64) {
    assert!(theta == 1e4 || theta == 1e7);
    let table = ComputedTable {
        theta,
        rule: AngleRule::HfF32AngleBf16CosSin,
        axes: Axes::Linear,
    };
    table
        .cos_sin([position; 3], pair, 32)
        .expect("HF RoPE pair")
}

/// The served device rotation (`RotaryArithmetic::F32Fma`): one F32 mate
/// product, then one F32 FMA. BF16 table operands make both products exact
/// for BF16 input operands.
fn served(x: f32, y: f32, cs: (f64, f64), second: bool) -> f32 {
    RotaryArithmetic::F32Fma.element(f64::from(x), f64::from(y), cs.0, cs.1, second) as f32
}

/// Pure HF BF16 operations, retained only for reference-distance checks.
pub fn rotate_pure_hf(x: f64, y: f64, cs: (f64, f64), second: bool) -> f64 {
    let a = round_bf16(round_bf16(x as f32) * cs.0 as f32);
    let b = round_bf16(round_bf16(y as f32) * cs.1 as f32);
    round_bf16(if second { a + b } else { a - b }) as f64
}

/// Triangle bound from exact rounding errors, including subnormals. This is
/// a distance to the unserved pure-HF variant, not a relaxed kernel bound.
pub fn pure_hf_distance_bound(x: f32, y: f32, cs: (f64, f64), second: bool) -> f64 {
    let sign = if second { 1.0 } else { -1.0 };
    let (xb, yb) = (round_bf16(x) as f64, round_bf16(y) as f64);
    let a = xb * cs.0;
    let b = yb * cs.1;
    let ab = round_bf16(a as f32) as f64;
    let bb = round_bf16(b as f32) as f64;
    let hf_sum = ab + sign * bb;
    let hf = rotate_pure_hf(x as f64, y as f64, cs, second);
    let product = y * cs.1 as f32;
    let sum = (x as f64).mul_add(cs.0, sign * product as f64);
    let f32_sum = served(x, y, cs, second) as f64;
    let stored = round_bf16(f32_sum as f32) as f64;
    (x as f64 - xb).abs() * cs.0.abs()
        + (y as f64 - yb).abs() * cs.1.abs()
        + (a - ab).abs()
        + (b - bb).abs()
        + (hf_sum - hf).abs()
        + (y as f64 * cs.1 - product as f64).abs()
        + (sum - f32_sum).abs()
        + (f32_sum - stored).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_cpu_torch_frequency_and_bf16_operation_bits() {
        let fixture = include_bytes!("../../../fixtures/hf_rope_cpu.bin");
        for (i, bits) in INVERSE_BITS.iter().flatten().enumerate() {
            assert_eq!(
                *bits,
                u32::from_le_bytes(fixture[4 * i..4 * i + 4].try_into().unwrap())
            );
        }
        for row in fixture[256..].chunks_exact(18) {
            let theta = [1e4, 1e7][row[0] as usize];
            let pair = row[1] as usize;
            let position = u32::from_le_bytes(row[2..6].try_into().unwrap()) as usize;
            let words: Vec<_> = row[6..]
                .chunks_exact(2)
                .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
                .collect();
            let cs = cos_sin(position, pair, theta);
            assert_eq!(f32_to_bf16(cs.0 as f32), words[2]);
            assert_eq!(f32_to_bf16(cs.1 as f32), words[3]);
            let (x, y) = (bf16_to_f32(words[0]) as f64, bf16_to_f32(words[1]) as f64);
            assert_eq!(
                f32_to_bf16(rotate_pure_hf(x, y, cs, false) as f32),
                words[4]
            );
            assert_eq!(f32_to_bf16(rotate_pure_hf(y, x, cs, true) as f32), words[5]);
        }
    }
    #[test]
    fn distance_bound_covers_f32_inputs_and_subnormals() {
        for (x, y) in [
            (1.001_f32, -0.3333_f32),
            (f32::from_bits(1), f32::from_bits(0x800001)),
            (f32::MIN_POSITIVE, -f32::MIN_POSITIVE),
        ] {
            for pair in 0..32 {
                let cs = cos_sin(262143, pair, 1e7);
                for second in [false, true] {
                    let rotated = round_bf16(served(x, y, cs, second)) as f64;
                    let pure = rotate_pure_hf(x as f64, y as f64, cs, second);
                    assert!((rotated - pure).abs() <= pure_hf_distance_bound(x, y, cs, second));
                }
            }
        }
    }
}
