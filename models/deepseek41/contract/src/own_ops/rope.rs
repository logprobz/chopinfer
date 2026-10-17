//! The RoPE cosine and sine rows of DeepSeek V4.1's two rotary tables
//! (`inference/model.py:368-389`), numerical contract
//! `deepseek41.rope-cos-sin.v1`.
//!
//! The angle of pair i at position p is the vendor's F32 angle bit for bit:
//! `fl32(fl32(p) * f_i)`, with `f_i` the inverse frequency the pinned
//! `precompute_freqs_cis` computes in F32 on CUDA, where the reference
//! builds the model (torch 2.14, captured 2026-10-17), YaRN ramp included,
//! pinned here as bits ([`RopeTable::frequency_bits`]). Three of the 64
//! differ from a CPU evaluation of the same code: CUDA's F32 `pow` gives
//! window pairs 11 and 25 one ulp more, and CUDA divides the YaRN ramp by
//! its scalar width as a product with the F32 reciprocal, which moves
//! compressed pair 24. Cosine and sine are those angles' correctly rounded
//! F32 values; there is no exact-angle table.
//!
//! An f64 cosine or sine within 2 ulp, rounded to F32, is correctly rounded
//! at every position below [`MAX_POSITIONS`]: of every pair's cosine and
//! sine there, the closest to an F32 rounding midpoint is 3.3e-16 of it
//! (the sine at position 127,994, pair 11 of the window table), 2.44 f64
//! ulp, found by an offline scan and resolved at 160 bits. The oracle
//! uses the host's f64 `cos` and `sin`; the near cases are pinned in this
//! module's tests.

use kernel_core::backend::Need;
use kernel_core::contract::Refusal;

/// Rotated pairs per row: the 64-dimension RoPE tail.
pub const PAIRS: usize = 32;

/// Positions below this are in the contract's domain: the served 1,048,576
/// positions and 65,536 beyond. A `Range` op refuses rows past it at
/// launch. A `Positions` op cannot check device positions there: a position
/// at or past it writes NaN cosine and sine. The host keeps positions below
/// the served context, at most 1,048,576, so a served position never
/// reaches it.
pub const MAX_POSITIONS: u32 = 1_114_112;

/// One of the two rotary tables.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RopeTable {
    /// Sliding-window-only layers (compress ratio 0, DSpark): theta 10000,
    /// no YaRN.
    Window,
    /// Compressing layers and their compressed latents and index keys:
    /// theta 160000 with YaRN (factor 16, original 65536, beta 32 and 1).
    Compressed,
}

const WINDOW_BITS: [u32; PAIRS] = [
    0x3f800000, 0x3f3ff911, 0x3f0ff59a, 0x3ed7e89b, 0x3ea1e89b, 0x3e72d423, 0x3e361887, 0x3e088d77,
    0x3dcccccd, 0x3d99940d, 0x3d6655c2, 0x3d2cba16, 0x3d0186e3, 0x3cc2434f, 0x3c91ad39, 0x3c5a7bf2,
    0x3c23d70a, 0x3bf5b9b0, 0x3bb8449c, 0x3b8a2e77, 0x3b4f3e38, 0x3b1b690d, 0x3ae91528, 0x3aaec98e,
    0x3a83126f, 0x3a44948d, 0x3a136a16, 0x39dd1725, 0x39a5cb60, 0x3978a815, 0x393a7753, 0x390bd472,
];

const COMPRESSED_BITS: [u32; PAIRS] = [
    0x3f800000, 0x3f300a3a, 0x3ef21c1f, 0x3ea67d01, 0x3e64f92e, 0x3e1d7475, 0x3dd88cb4, 0x3d94e963,
    0x3d4ccccd, 0x3d0cd4fb, 0x3cc1b019, 0x3c8530ce, 0x3c372dbf, 0x3bfbed88, 0x3bad3d5e, 0x3b6e4237,
    0x3b147ae1, 0x3ab714e0, 0x3a5ebdb6, 0x3a0530ce, 0x399bb3af, 0x39305978, 0x38be904d, 0x383e9b5f,
    0x37a3d708, 0x36b443d0, 0x3677eba6, 0x362a7be8, 0x35ea77ff, 0x35a13bdc, 0x355dbf30, 0x35187c4d,
];

impl RopeTable {
    pub const ALL: [RopeTable; 2] = [RopeTable::Window, RopeTable::Compressed];

    /// The inverse frequency of each pair as F32 bits, from the pinned
    /// vendor code on CUDA (`precompute_freqs_cis`, its `torch.polar` angle
    /// at position 1).
    pub const fn frequency_bits(self) -> &'static [u32; PAIRS] {
        match self {
            RopeTable::Window => &WINDOW_BITS,
            RopeTable::Compressed => &COMPRESSED_BITS,
        }
    }

    /// A short tag for symbols and implementation names.
    pub const fn tag(self) -> &'static str {
        match self {
            RopeTable::Window => "window",
            RopeTable::Compressed => "compressed",
        }
    }

    /// The vendor's F32 angle of `pair` at `position`.
    pub fn angle(self, position: u32, pair: usize) -> f32 {
        position as f32 * f32::from_bits(self.frequency_bits()[pair])
    }

    /// The correctly rounded F32 cosine and sine of [`RopeTable::angle`].
    pub fn cos_sin(self, position: u32, pair: usize) -> (f32, f32) {
        let angle = f64::from(self.angle(position, pair));
        (angle.cos() as f32, angle.sin() as f32)
    }
}

/// Which positions an op's rows take.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RopeRows {
    /// Row r takes position `first + r`; a launch writes `Counts::rows`
    /// rows (a position-indexed table, filled once or in ranges).
    Range,
    /// Row r takes `positions[r]`; a launch writes the device count of
    /// `live_rows` rows, at most `Counts::rows` (a step's rows).
    Positions,
}

/// Cosine and sine rows of one table: `output[r][pair] = (cos, sin)` in F32
/// of the row's position, as [`RopeTable::cos_sin`], for positions below
/// [`MAX_POSITIONS`] (see there for the rest).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RopeCosSin {
    pub table: RopeTable,
    pub rows: RopeRows,
}

impl RopeCosSin {
    pub fn validate(&self, need: Need) -> Result<(), Refusal> {
        if need.capacity == 0 || need.capacity > MAX_POSITIONS {
            return Err(Refusal("RoPE rows capacity outside the position domain"));
        }
        Ok(())
    }

    /// Bytes of `rows` output rows.
    pub fn output_bytes(rows: u32) -> u64 {
        u64::from(rows) * PAIRS as u64 * 8
    }
}

kernel_core::scratch_record! {
    /// A RoPE op's buffers: the U32 `positions` per row and the device U32
    /// count of `live_rows` (both bound for `RopeRows::Positions` only), the
    /// first position of a `Range` op, and the F32 `output` rows
    /// `[row][PAIRS][cos, sin]`, 8-byte aligned.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct RopeCosSinBuffers {
        pub positions: option,
        pub live_rows: option,
        pub first: scalar(u32),
        pub output: value,
    }
}

/// The oracle's rows `[row][PAIRS][cos, sin]` for `positions`.
pub fn reference(table: RopeTable, positions: &[u32]) -> Result<Vec<f32>, Refusal> {
    let mut out = Vec::with_capacity(positions.len() * PAIRS * 2);
    for &p in positions {
        if p >= MAX_POSITIONS {
            return Err(Refusal("a RoPE position is outside the contract's domain"));
        }
        for pair in 0..PAIRS {
            let (c, s) = table.cos_sin(p, pair);
            out.extend([c, s]);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    /// SHA-256 of the vendor's F32 angle table `[position][pair]` at seqlen
    /// 2^20 (`precompute_freqs_cis` on CUDA, `torch.polar`'s angle
    /// argument), per table.
    const VENDOR_ANGLE_SHA256: [(RopeTable, &str); 2] = [
        (
            RopeTable::Window,
            "e52b4866731714b90ebf78c31eaa698d8013aff8b3324cfd10ee5e7975e1445d",
        ),
        (
            RopeTable::Compressed,
            "99c92572f98f747bbe2fa85fed689a653f4809bedc6e2568343dc13acaf9c00d",
        ),
    ];

    #[test]
    fn angles_equal_the_vendors_at_every_position_to_one_million() {
        for (table, pinned) in VENDOR_ANGLE_SHA256 {
            let mut hasher = Sha256::new();
            let mut row = [0_u8; PAIRS * 4];
            for p in 0..1_u32 << 20 {
                for pair in 0..PAIRS {
                    row[pair * 4..][..4].copy_from_slice(&table.angle(p, pair).to_le_bytes());
                }
                hasher.update(row);
            }
            let digest: String = hasher
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            assert_eq!(digest, pinned, "{table:?}");
        }
    }

    /// The frequencies by the vendor's F32 rule as CUDA evaluates it, with
    /// the host's correctly rounded `powf`: `1 / theta^(2i / 64)`, and for
    /// YaRN the ramp `(i - low) * RN32(1 / (high - low))` clamped to [0, 1],
    /// `smooth = 1 - ramp`, `f / factor * (1 - smooth) + f * smooth`.
    fn host_rule(table: RopeTable, pair: usize) -> u32 {
        let (theta, yarn) = match table {
            RopeTable::Window => (10_000_f32, false),
            RopeTable::Compressed => (160_000_f32, true),
        };
        let exponent = (2 * pair) as f32 / 64.0;
        let f = 1.0 / theta.powf(exponent);
        if !yarn {
            return f.to_bits();
        }
        let corrected = |rotations: f64| {
            64.0 * (65_536.0 / (rotations * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * f64::from(theta).ln())
        };
        let low = corrected(32.0).floor().max(0.0);
        let high = corrected(1.0).ceil().min(63.0);
        let reciprocal = 1.0 / (high - low).max(1e-3) as f32;
        let ramp = ((pair as f32 - low as f32) * reciprocal).clamp(0.0, 1.0);
        let smooth = 1.0 - ramp;
        (f / 16.0 * (1.0 - smooth) + f * smooth).to_bits()
    }

    #[test]
    fn pinned_frequencies_follow_the_vendor_rule() {
        for table in RopeTable::ALL {
            for pair in 0..PAIRS {
                // CUDA's F32 `pow` is one ulp above the correctly rounded
                // power's reciprocal at these two window pairs.
                let pow_ulp = u32::from(table == RopeTable::Window && [11, 25].contains(&pair));
                assert_eq!(
                    host_rule(table, pair) + pow_ulp,
                    table.frequency_bits()[pair],
                    "{table:?} pair {pair}"
                );
            }
        }
    }

    #[test]
    fn near_midpoint_cases_round_correctly() {
        // (table, position, pair, cosine, correctly rounded bits), from the
        // offline scan; the closest cases to an F32 rounding midpoint.
        let cases = [
            (RopeTable::Window, 127_994, 11, false, 1_045_540_322_u32),
            (RopeTable::Window, 933_959, 29, false, 1_065_352_945),
            (RopeTable::Compressed, 924_563, 9, true, 3_208_786_617),
            (RopeTable::Compressed, 1_106_563, 29, true, 1_047_848_371),
            (RopeTable::Window, 1_147_495, 6, true, 3_207_697_236),
            (RopeTable::Compressed, 1_484_639, 15, false, 1_045_540_322),
            (RopeTable::Compressed, 1_132_878, 7, true, 1_065_228_765),
            (RopeTable::Window, 1_703_397, 30, false, 1_064_822_898),
        ];
        for (table, position, pair, cosine, bits) in cases {
            let (c, s) = table.cos_sin(position, pair);
            assert_eq!(if cosine { c } else { s }.to_bits(), bits);
        }
        assert_eq!(RopeTable::Window.cos_sin(0, 3), (1.0, 0.0));
        assert!(reference(RopeTable::Window, &[MAX_POSITIONS]).is_err());
        assert_eq!(
            reference(RopeTable::Compressed, &[5, 9]).unwrap().len(),
            128
        );
    }
}
