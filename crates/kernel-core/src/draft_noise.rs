//! Gumbel draft noise: the target sampler's own noise on a drafter's logits.
//!
//! The host samples a row of a request at temperature `T > 0` as the
//! largest Gumbel score `(s(t) - M) + g(t)` over the kept tokens
//! (`crate::sampling`, step 7), with `g` from Philox4x32-10 keyed by the
//! request's seed at the counter (token ID, generated index, 0). A drafter
//! that proposes, at draft position `j`, the argmax of `x(t) / T + g(t)`
//! with `g` taken at the generated index the target samples position `j`
//! at (`generated + j`) proposes the target's own sample whenever its
//! logits rank the leading Gumbel scores as the target's processed values
//! do: the Gumbel-max coupling. Verification is unchanged, a proposal is
//! accepted only when it equals the target's sample at its row, so the
//! noise changes which tokens are proposed, never which are emitted.
//!
//! [`DraftNoise`] perturbs, in place, the live columns of each live row of
//! one rank's vocabulary shard of F32 drafter logits whose record has a
//! positive temperature: `y = x / T + g(global id)` with the noise index
//! `generated + position`. Rows whose temperature is zero (greedy), rows at
//! or past the device live count and columns at or past the live width keep
//! their bits, so a sharded argmax over the result proposes exactly what it
//! proposed before for every greedy row.
//!
//! The drafter applies the temperature only. Penalties, logit bias, the EOS
//! mask, constraint masks, min-p, top-k and top-p are not applied: a
//! proposal they would exclude is never the target's sample and is
//! rejected. Proposals are not model outputs, so the perturbation has an
//! error bound ([`bound`]) rather than an exact contract.

use crate::contract::{Refusal, Validate};
use crate::sampling::{MAX_TEMPERATURE, MAX_WIDTH, MIN_TEMPERATURE, gumbel};

/// The numerical contract of the perturbation.
pub const CONTRACT: &str = "draft-noise.philox4x32-10.gumbel.f32.v1";

/// Bytes of one row's [`NoiseRecord`].
pub const NOISE_BYTES: usize = 32;

/// The largest absolute error of the device's noise against [`gumbel`].
/// The device evaluates the device sampler's pruning score: an F32 unit
/// from the first Philox word and two F32 polynomial logarithms, with the
/// exact f64 noise for units below 1e-5 or above 0.999. The error peaks
/// near 0.999, where `-ln(u)` is about 1e-3 and the inner logarithm's
/// absolute error of about 2e-7 (series truncation and F32 rounding)
/// becomes a relative one: about 3e-4 by analysis, a third of this bound.
pub const NOISE_ERROR: f64 = 1.0 / 1024.0;

/// One rank's shard of a vocabulary split into `ranks` contiguous shards of
/// `vocab_per_rank` rows; rank `r`'s column `c` is global token `r *
/// vocab_per_rank + c`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DraftNoise {
    pub vocab_per_rank: u32,
    pub ranks: u32,
}

impl Validate for DraftNoise {
    fn validate(&self) -> Result<(), Refusal> {
        if self.vocab_per_rank == 0
            || self.ranks == 0
            || u64::from(self.vocab_per_rank) * u64::from(self.ranks) > u64::from(MAX_WIDTH)
        {
            return Err(Refusal(
                "draft noise needs 1 to 1048576 vocabulary rows over its shards",
            ));
        }
        Ok(())
    }
}

crate::scratch_record! {
    /// The draft noise's buffers on one rank: F32 `logits` rows of the
    /// query's `vocab_per_rank`, perturbed in place, of which the device
    /// U32 `live_width` (read as `max(min(live_width, vocab_per_rank), 1)`,
    /// as the sharded argmax partial reads it) are live; one
    /// [`NoiseRecord`] per row in `noise`; the device U32 `live_rows`; and
    /// the draft `position` added to every record's generated index.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct DraftNoiseBuffers {
        pub logits: value,
        pub live_width: value,
        pub noise: value,
        pub live_rows: value,
        pub position: scalar(u32),
    }
}

/// One row's noise: the request's seed, the generated index of the token
/// the target samples at draft position zero, and its temperature, zero
/// for a greedy row. Little endian: seed, generated, temperature (f64
/// bits) and a zero word.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseRecord {
    pub seed: u64,
    pub generated: u64,
    pub temperature: f64,
}

impl NoiseRecord {
    /// A row left unchanged.
    pub const GREEDY: NoiseRecord = NoiseRecord {
        seed: 0,
        generated: 0,
        temperature: 0.0,
    };

    /// Zero or a temperature the host samples at.
    pub fn validate(&self) -> Result<(), Refusal> {
        if self.temperature == 0.0
            || (MIN_TEMPERATURE..=MAX_TEMPERATURE).contains(&self.temperature)
        {
            Ok(())
        } else {
            Err(Refusal("draft noise temperature is outside the sampler's"))
        }
    }

    pub fn sampled(&self) -> bool {
        self.temperature > 0.0
    }

    pub fn encode(&self) -> [u8; NOISE_BYTES] {
        let mut bytes = [0; NOISE_BYTES];
        bytes[..8].copy_from_slice(&self.seed.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.generated.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.temperature.to_bits().to_le_bytes());
        bytes
    }

    /// The generated index the target samples draft `position` at.
    pub fn index(&self, position: u32) -> u64 {
        self.generated.wrapping_add(u64::from(position))
    }
}

/// The reference values of one row of rank `rank`'s shard after the
/// perturbation, over its `live` columns: `x / T + g` in f64 for a sampled
/// row, and the row's own values for a greedy one.
pub fn reference(
    op: &DraftNoise,
    rank: u32,
    row: &[f32],
    noise: &NoiseRecord,
    position: u32,
) -> Result<Vec<f64>, Refusal> {
    op.validate()?;
    noise.validate()?;
    if rank >= op.ranks || row.is_empty() || row.len() > op.vocab_per_rank as usize {
        return Err(Refusal("draft noise row is outside its shard"));
    }
    let first = rank * op.vocab_per_rank;
    Ok(row
        .iter()
        .zip(first..)
        .map(|(&x, token)| {
            if noise.sampled() {
                f64::from(x) / noise.temperature + gumbel(noise.seed, noise.index(position), token)
            } else {
                f64::from(x)
            }
        })
        .collect())
}

/// How far a device's perturbed value may lie from `reference`'s, for a
/// row value `x`: [`NOISE_ERROR`] plus two F32 roundings of the quotient
/// and the sum.
pub fn bound(x: f32, noise: &NoiseRecord, reference: f64) -> f64 {
    if !noise.sampled() {
        return 0.0;
    }
    let quotient = (f64::from(x) / noise.temperature).abs();
    NOISE_ERROR + 2_f64.powi(-23) * (quotient + reference.abs() + (reference - quotient).abs())
}

/// The coupled proposal of a full-vocabulary row: the largest reference
/// value, ties to the lower token ID, as the sharded argmax picks it.
pub fn proposal(logits: &[f32], noise: &NoiseRecord, position: u32) -> Result<u32, Refusal> {
    let vocab = u32::try_from(logits.len()).map_err(|_| Refusal("draft noise row is too wide"))?;
    let op = DraftNoise {
        vocab_per_rank: vocab,
        ranks: 1,
    };
    let values = reference(&op, 0, logits, noise, position)?;
    let mut best = 0_usize;
    for (token, &value) in values.iter().enumerate() {
        if value > values[best] {
            best = token;
        }
    }
    Ok(best as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(seed: u64, generated: u64, temperature: f64) -> NoiseRecord {
        NoiseRecord {
            seed,
            generated,
            temperature,
        }
    }

    #[test]
    fn records_encode_little_endian_with_a_zero_tail() {
        let bytes = record(0x0102_0304_0506_0708, 9, 0.5).encode();
        assert_eq!(bytes[..8], 0x0102_0304_0506_0708_u64.to_le_bytes());
        assert_eq!(bytes[8..16], 9_u64.to_le_bytes());
        assert_eq!(bytes[16..24], 0.5_f64.to_bits().to_le_bytes());
        assert_eq!(bytes[24..], [0; 8]);
        assert_eq!(NoiseRecord::GREEDY.encode(), [0; NOISE_BYTES]);
    }

    #[test]
    fn greedy_rows_keep_their_values_and_temperatures_are_checked() {
        let op = DraftNoise {
            vocab_per_rank: 4,
            ranks: 2,
        };
        let row = [1.5, -0.0, f32::MAX, -3.0];
        assert_eq!(
            reference(&op, 1, &row, &NoiseRecord::GREEDY, 3).unwrap(),
            row.map(f64::from)
        );
        assert_eq!(bound(1.0, &NoiseRecord::GREEDY, 1.0), 0.0);
        for temperature in [-1.0, f64::NAN, 1e-300, 101.0] {
            assert!(record(1, 2, temperature).validate().is_err());
        }
        assert!(reference(&op, 2, &row, &NoiseRecord::GREEDY, 0).is_err());
        assert!(reference(&op, 0, &[0.0; 5], &NoiseRecord::GREEDY, 0).is_err());
        assert!(
            DraftNoise {
                vocab_per_rank: 1 << 19,
                ranks: 3
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn each_shard_draws_its_global_tokens_at_the_positions_index() {
        let noise = record(0x9e37_79b9_7f4a_7c15, 41, 0.7);
        let row = [0.25_f32, -1.0, 2.0];
        let op = DraftNoise {
            vocab_per_rank: 3,
            ranks: 4,
        };
        let values = reference(&op, 2, &row, &noise, 3).unwrap();
        for (column, (&x, value)) in row.iter().zip(values).enumerate() {
            let expected = f64::from(x) / 0.7 + gumbel(noise.seed, 44, 6 + column as u32);
            assert_eq!(value, expected);
        }
        assert_eq!(noise.index(3), 44);
        assert_eq!(record(1, u64::MAX, 1.0).index(2), 1);
    }

    #[test]
    fn proposals_follow_the_noise_and_greedy_rows_take_the_argmax() {
        let logits = [0.0_f32; 64];
        let picks = (0..8)
            .map(|position| proposal(&logits, &record(7, 100, 1.0), position).unwrap())
            .collect::<Vec<_>>();
        // Flat logits leave the choice to the noise alone, which differs
        // from one generated index to the next.
        assert!(picks.windows(2).any(|pair| pair[0] != pair[1]));
        for (position, &pick) in picks.iter().enumerate() {
            let index = 100 + position as u64;
            let best = (0..64)
                .max_by(|&a, &b| gumbel(7, index, a).total_cmp(&gumbel(7, index, b)))
                .unwrap();
            assert_eq!(pick, best);
        }
        let mut ramp = [0.0_f32; 64];
        ramp[9] = 1.0;
        ramp[11] = 1.0;
        assert_eq!(proposal(&ramp, &NoiseRecord::GREEDY, 0).unwrap(), 9);
    }
}
