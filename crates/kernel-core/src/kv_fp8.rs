//! FP8 E4M3 KV cache storage.
//!
//! The SM120 KV-write kernels store `E4M3(value / scale)` with
//! `cvt.rn.satfinite`, and attention multiplies the scale back in, folded into
//! the softmax scale for K and the output for V. This module is that quantizer
//! in f64, per-plane storage statistics for choosing scales, and the
//! attention-output tolerance the rounding implies, and the stored codes
//! themselves at a given tensor scale.

use crate::codec::SmallFloat;

/// Largest finite E4M3 magnitude.
pub const E4M3_MAX: f64 = 448.0;
/// Smallest normal E4M3 magnitude; below it the step is fixed at 2^-9.
pub const E4M3_MIN_NORMAL: f64 = 1.0 / 64.0;
/// Magnitudes above this round to 448 with more than half a step of error.
pub const E4M3_SATURATION: f64 = 464.0;
/// Standard deviations of the first-order error model that
/// `attention_sigma` callers accept as a tolerance. This is a statistical
/// allowance for independent rounding errors, not a bound.
pub const TOLERANCE_SIGMAS: f64 = 6.0;

/// Nearest E4M3 value, ties to even, saturating at 448 like `cvt.rn.satfinite`.
pub fn round_e4m3(value: f64) -> f64 {
    let magnitude = value.abs();
    if magnitude == 0.0 || magnitude.is_nan() {
        return value;
    }
    if magnitude.is_infinite() {
        return E4M3_MAX.copysign(value);
    }
    let quantum = 2.0 * half_step(magnitude);
    ((magnitude / quantum).round_ties_even() * quantum)
        .min(E4M3_MAX)
        .copysign(value)
}

/// The code a KV-write kernel stores for `value` at a given tensor scale:
/// `value / scale` divided in F32, then E4M3 to nearest with ties to even,
/// saturating at 448 (`cvt.rn.satfinite`); NaN stores 0x7f.
pub fn encode(value: f32, scale: f32) -> u8 {
    SmallFloat::E4M3.encode_f32(value / scale)
}

/// The value attention reads back from a stored code: `E4M3(code) * scale`
/// in f64.
pub fn decode(code: u8, scale: f32) -> f64 {
    SmallFloat::E4M3.decode_f64(code) * f64::from(scale)
}

/// Half the E4M3 step at a magnitude in scaled units: the largest rounding
/// error of a value that does not saturate.
pub fn half_step(magnitude: f64) -> f64 {
    let exponent = if magnitude > 0.0 {
        (((magnitude.to_bits() >> 52) as i32) - 1023).max(-6)
    } else {
        -6
    };
    2.0_f64.powi(exponent - 4)
}

/// Smallest power-of-two scale that keeps `amax` at or below 448. Dividing by
/// a power of two is exact, so this scale changes only which values are
/// subnormal or saturate, never the rounding of the others.
pub fn calibrated_scale(amax: f64) -> f64 {
    if amax.is_nan() || amax <= 0.0 || amax.is_infinite() {
        return 1.0;
    }
    let ratio = amax / E4M3_MAX;
    let exponent = ((ratio.to_bits() >> 52) as i32) - 1023;
    if ratio.to_bits() & ((1 << 52) - 1) == 0 {
        ratio
    } else {
        2.0_f64.powi(exponent + 1)
    }
}

/// How rows are scaled before rounding to E4M3.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fp8Scale {
    /// One scale for the whole plane, as the kernels take it per launch.
    Tensor(f64),
    /// One `amax / 448` scale per token and head. No kernel stores this; it is
    /// the reference for what finer scaling could gain.
    TokenHead,
}

/// Storage statistics of one K or V plane, in unscaled units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Fp8Stats {
    pub elements: u64,
    pub amax: f64,
    pub sum_squares: f64,
    pub error_squares: f64,
    /// Nonzero values below the smallest normal after scaling.
    pub subnormal: u64,
    /// Values that saturate: above `E4M3_SATURATION` after scaling.
    pub saturated: u64,
}

impl Fp8Stats {
    pub fn merge(&mut self, other: &Self) {
        self.elements += other.elements;
        self.amax = self.amax.max(other.amax);
        self.sum_squares += other.sum_squares;
        self.error_squares += other.error_squares;
        self.subnormal += other.subnormal;
        self.saturated += other.saturated;
    }
}

/// Stores rows of `head_dim` values as `scale * E4M3(value / scale)` and adds
/// them to `stats`.
pub fn store(values: &[f64], head_dim: usize, scale: Fp8Scale, stats: &mut Fp8Stats) -> Vec<f64> {
    assert!(head_dim > 0 && values.len().is_multiple_of(head_dim));
    let mut stored = Vec::with_capacity(values.len());
    for head in values.chunks_exact(head_dim) {
        let scale = match scale {
            Fp8Scale::Tensor(scale) => scale,
            Fp8Scale::TokenHead => {
                let amax = head.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                if amax > 0.0 { amax / E4M3_MAX } else { 1.0 }
            }
        };
        for &value in head {
            let scaled = value / scale;
            let kept = round_e4m3(scaled) * scale;
            stats.elements += 1;
            stats.amax = stats.amax.max(value.abs());
            stats.sum_squares += value * value;
            stats.error_squares += (kept - value) * (kept - value);
            stats.subnormal += u64::from(scaled != 0.0 && scaled.abs() < E4M3_MIN_NORMAL);
            stats.saturated += u64::from(scaled.abs() > E4M3_SATURATION);
            stored.push(kept);
        }
    }
    stored
}

/// One query row against one KV head with E4M3 storage.
#[derive(Clone, Copy, Debug)]
pub struct Fp8Attention<'a> {
    /// `[qk_dim]`.
    pub query: &'a [f64],
    /// Unrounded keys `[keys, qk_dim]` and values `[keys, v_dim]`.
    pub keys: &'a [f64],
    pub values: &'a [f64],
    pub qk_dim: usize,
    pub v_dim: usize,
    pub softmax_scale: f64,
    pub k_scale: f64,
    pub v_scale: f64,
    /// Extra logit in the denominator only.
    pub sink: Option<f64>,
}

/// Exact attention output of the unrounded keys and values, and the standard
/// deviation of each output element's E4M3 storage error.
///
/// Model: every stored element's error is independent and uniform within its
/// half step `h` (variance `h^2 / 3`, an upper bound when a value is exactly
/// representable). To first order a key error moves logit j by
/// `softmax_scale * q . dk_j`, which moves output d by
/// `p_j * ds_j * (v_jd - o_d)`, and a value error moves it by `p_j * dv_jd`.
/// Saturated elements are outside the model and panic.
///
/// Independence holds for random operands. Repeated values round the same
/// way and their errors add instead of averaging out: one value row repeated
/// over n keys of equal weight has an error up to `h` against a deviation of
/// about `h / sqrt(3 n)`. The worst case to first order is
/// `sum_j p_j (|ds_j| |v_jd - o_d| + h_jd)` with `|ds_j|` at most
/// `softmax_scale * sum_e |q_e| h_je`. A single key with an E4M3 tie
/// attains it; it is not a bound for a finite perturbation through the
/// softmax.
pub fn attention_sigma(input: Fp8Attention<'_>) -> (Vec<f64>, Vec<f64>) {
    let Fp8Attention {
        query,
        keys,
        values,
        qk_dim,
        v_dim,
        softmax_scale,
        k_scale,
        v_scale,
        sink,
    } = input;
    let count = keys.len() / qk_dim;
    assert!(query.len() == qk_dim && values.len() == count * v_dim && count > 0);
    let bound = |value: f64, scale: f64| {
        let scaled = (value / scale).abs();
        assert!(scaled <= E4M3_SATURATION, "saturated E4M3 KV element");
        half_step(scaled) * scale
    };
    let logits: Vec<f64> = keys
        .chunks_exact(qk_dim)
        .map(|key| softmax_scale * key.iter().zip(query).map(|(k, q)| k * q).sum::<f64>())
        .collect();
    let max = logits.iter().copied().chain(sink).fold(f64::MIN, f64::max);
    let weights: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let total = weights.iter().sum::<f64>() + sink.map_or(0.0, |s| (s - max).exp());
    let p: Vec<f64> = weights.iter().map(|w| w / total).collect();
    let mut output = vec![0.0; v_dim];
    for (p, value) in p.iter().zip(values.chunks_exact(v_dim)) {
        for (o, v) in output.iter_mut().zip(value) {
            *o += p * v;
        }
    }
    let mut variance = vec![0.0; v_dim];
    for ((p, key), value) in p
        .iter()
        .zip(keys.chunks_exact(qk_dim))
        .zip(values.chunks_exact(v_dim))
    {
        let logit_variance = softmax_scale
            * softmax_scale
            * key
                .iter()
                .zip(query)
                .map(|(&k, q)| q * q * bound(k, k_scale).powi(2) / 3.0)
                .sum::<f64>();
        for ((var, &v), o) in variance.iter_mut().zip(value).zip(&output) {
            *var += p * p * (logit_variance * (v - o) * (v - o) + bound(v, v_scale).powi(2) / 3.0);
        }
    }
    (output, variance.into_iter().map(f64::sqrt).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ngram_oracle::splitmix64;

    #[test]
    fn stored_codes_divide_in_f32_and_saturate() {
        // 3 / 0.1 rounds in F32 before the E4M3 rounding.
        assert_eq!(encode(3.0, 0.1), SmallFloat::E4M3.encode_f32(3.0_f32 / 0.1));
        assert_eq!(decode(encode(1.5, 0.5), 0.5), 1.5);
        assert_eq!(encode(1000.0, 1.0), 0x7e);
        assert_eq!(encode(-1000.0, 1.0), 0xfe);
        assert_eq!(encode(f32::NAN, 1.0), 0x7f);
        assert_eq!(encode(-0.0, 2.0), 0x80);
        // The stored value matches the f64 quantizer wherever the F32
        // division is exact, as for a power-of-two scale.
        for value in [0.013_f32, 1.7, -300.0, 900.0] {
            assert_eq!(
                decode(encode(value, 0.5), 0.5),
                round_e4m3(f64::from(value) / 0.5) * 0.5
            );
        }
    }

    fn normal(seed: u64, count: usize, sigma: f64) -> Vec<f64> {
        (0..count as u64)
            .map(|i| {
                let a = (splitmix64(seed * 1_000_003 + 2 * i) >> 11) as f64 / (1_u64 << 53) as f64;
                let b =
                    (splitmix64(seed * 1_000_003 + 2 * i + 1) >> 11) as f64 / (1_u64 << 53) as f64;
                sigma * (-2.0 * (1.0 - a).ln()).sqrt() * (std::f64::consts::TAU * b).cos()
            })
            .collect()
    }

    #[test]
    fn rounding_matches_the_codec_and_stays_within_half_a_step() {
        for (index, value) in normal(1, 200_000, 40.0).into_iter().enumerate() {
            let value = value * 2.0_f64.powi((index % 24) as i32 - 16);
            let rounded = round_e4m3(value);
            let code = SmallFloat::E4M3.encode_f32(value as f32);
            assert_eq!(rounded, SmallFloat::E4M3.decode_f64(code), "{value}");
            if value.abs() <= E4M3_SATURATION {
                assert!((rounded - value).abs() <= half_step(value.abs()), "{value}");
            } else {
                assert_eq!(rounded.abs(), E4M3_MAX);
            }
        }
        assert_eq!(round_e4m3(1.0e9), E4M3_MAX);
        assert_eq!(round_e4m3(-f64::INFINITY), -E4M3_MAX);
        assert_eq!(half_step(0.0), 2.0_f64.powi(-10));
    }

    #[test]
    fn power_of_two_scales_only_move_the_range() {
        assert_eq!(calibrated_scale(448.0), 1.0);
        assert_eq!(calibrated_scale(449.0), 2.0);
        assert_eq!(calibrated_scale(20.0), 0.0625);
        assert_eq!(calibrated_scale(0.0), 1.0);
        let values = normal(2, 4096, 1.0);
        let (mut unit, mut quarter) = (Fp8Stats::default(), Fp8Stats::default());
        let a = store(&values, 128, Fp8Scale::Tensor(1.0), &mut unit);
        let b = store(&values, 128, Fp8Scale::Tensor(0.25), &mut quarter);
        for ((a, b), v) in a.iter().zip(&b).zip(&values) {
            if v.abs() >= E4M3_MIN_NORMAL {
                assert_eq!(a, b);
            }
        }
        assert!(unit.subnormal > quarter.subnormal && quarter.subnormal > 0);
        assert!(quarter.error_squares < unit.error_squares);
        let mut token_head = Fp8Stats::default();
        store(&values, 128, Fp8Scale::TokenHead, &mut token_head);
        assert!(token_head.subnormal * 10 < unit.subnormal);
        assert_eq!(token_head.saturated, 0);
    }

    /// The first-order model's tolerance holds on the exact attention of
    /// stored keys and values, and is not loose by more than a small factor.
    #[test]
    fn attention_error_stays_within_the_sigma_tolerance() {
        let mut worst = 0.0_f64;
        let cases = [
            // (qk_dim, v_dim, keys, logit spread, k_scale, v_scale, sink)
            (192, 128, 4096, 1.0, 1.0, 1.0, None),
            (192, 128, 1024, 3.0, 1.0, 1.0, Some(2.0)),
            (256, 256, 2048, 1.0, 0.5, 0.5, None),
            (256, 256, 777, 4.0, 0.0625, 0.03125, None),
            (128, 128, 64, 2.0, 1.0, 1.0, None),
        ];
        for (seed, &(qk_dim, v_dim, count, spread, k_scale, v_scale, sink)) in
            cases.iter().enumerate()
        {
            let seed = seed as u64 * 10;
            let query = normal(seed + 1, qk_dim, spread);
            let keys = normal(seed + 2, count * qk_dim, 1.0);
            let values = normal(seed + 3, count * v_dim, 0.3);
            let softmax_scale = 1.0 / (qk_dim as f64).sqrt();
            let mut stats = Fp8Stats::default();
            let stored_keys = store(&keys, qk_dim, Fp8Scale::Tensor(k_scale), &mut stats);
            let stored_values = store(&values, v_dim, Fp8Scale::Tensor(v_scale), &mut stats);
            let input = Fp8Attention {
                query: &query,
                keys: &keys,
                values: &values,
                qk_dim,
                v_dim,
                softmax_scale,
                k_scale,
                v_scale,
                sink,
            };
            let (exact, sigma) = attention_sigma(input);
            let (stored, _) = attention_sigma(Fp8Attention {
                keys: &stored_keys,
                values: &stored_values,
                ..input
            });
            let mut case_worst = 0.0_f64;
            for ((a, b), s) in stored.iter().zip(&exact).zip(&sigma) {
                case_worst = case_worst.max((a - b).abs() / s);
            }
            assert!(
                case_worst <= TOLERANCE_SIGMAS,
                "case {seed}: {case_worst} sigmas"
            );
            worst = worst.max(case_worst);
        }
        assert!(worst > 1.0, "tolerance model is too loose: {worst} sigmas");
    }
}
