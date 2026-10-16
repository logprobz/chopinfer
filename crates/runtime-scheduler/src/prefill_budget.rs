//! Model-neutral wall-time estimates. Each power-of-two bucket uses EWMA 1/8.
use crate::PrefillTarget;

const BUCKETS: usize = usize::BITS as usize + 1;
/// The smallest cap, and the step between candidate caps.
const CHUNK: usize = 512;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct StepTimes {
    decode_us: [u64; BUCKETS],
    previous_decode_us: [u64; BUCKETS],
    decode_samples: [u8; BUCKETS],
    prefill_ns_per_token: [u64; BUCKETS],
}

impl Default for StepTimes {
    fn default() -> Self {
        Self {
            decode_us: [0; BUCKETS],
            previous_decode_us: [0; BUCKETS],
            decode_samples: [0; BUCKETS],
            prefill_ns_per_token: [0; BUCKETS],
        }
    }
}

fn bucket(count: usize) -> usize {
    (usize::BITS - count.saturating_sub(1).leading_zeros()) as usize
}

/// A decode estimate for `rows` counts only after two samples.
fn qualified(decode_us: &[u64; BUCKETS], samples: &[u8; BUCKETS], rows: usize) -> Option<u64> {
    let index = bucket(rows);
    (decode_us[index] != 0 && samples[index] >= 2).then_some(decode_us[index])
}

/// The closest measured prefill bucket's nanoseconds per token.
fn nearest_rate(rates: &[u64; BUCKETS], tokens: usize) -> Option<u64> {
    let target = bucket(tokens);
    rates
        .iter()
        .enumerate()
        .filter(|(_, rate)| **rate != 0)
        .min_by_key(|(index, _)| index.abs_diff(target))
        .map(|(_, rate)| *rate)
}

fn update(value: &mut u64, sample: u64) {
    *value = if *value == 0 {
        sample
    } else {
        ((u128::from(*value) * 7 + u128::from(sample)) / 8) as u64
    }
    .max(1);
}

impl StepTimes {
    pub(super) fn observe(&mut self, prefill: usize, decode: usize, micros: u64) {
        if micros == 0 {
            return;
        }
        if prefill == 0 && decode > 0 {
            self.observe_decode(decode, micros);
        } else if prefill > 0 {
            let index = bucket(decode);
            if decode > 0 {
                // A known prefill rate can recover a capture-inflated decode
                // estimate even when steady traffic never runs pure decode.
                if let Some(rate) = self.prefill_rate(prefill) {
                    let predicted = (prefill as u128 * u128::from(rate) / 1000)
                        .min(u128::from(u64::MAX)) as u64;
                    if let Some(residual) = micros.checked_sub(predicted).filter(|n| *n > 0)
                        && (self.decode_samples[index] < 2 || residual < self.decode_us[index])
                    {
                        self.observe_decode(decode, residual);
                        self.decode_us[index] = self.decode_us[index].min(residual);
                    }
                }
                if self.decode_samples[index] < 2 {
                    return;
                }
                if micros <= self.decode_us[index] {
                    // The whole mixed step is shorter than our decode estimate.
                    // Disqualify it instead of discarding contradictions forever.
                    self.decode_samples[index] = 0;
                    return;
                }
            }
            let decode_time = if decode == 0 {
                0
            } else {
                self.decode_us[index]
            };
            let rate = (u128::from(micros - decode_time) * 1000 / prefill as u128)
                .clamp(1, u128::from(u64::MAX)) as u64;
            update(&mut self.prefill_ns_per_token[bucket(prefill)], rate);
        }
    }

    fn observe_decode(&mut self, rows: usize, micros: u64) {
        let index = bucket(rows);
        if self.decode_samples[index] == 0 {
            self.decode_us[index] = micros;
        } else {
            // The lower of consecutive samples excludes isolated first-use
            // capture/planning stalls. Sustained eager or graph costs still learn.
            let steady = micros.min(self.previous_decode_us[index]);
            if self.decode_samples[index] == 1 {
                self.decode_us[index] = steady;
            } else {
                update(&mut self.decode_us[index], steady);
            }
        }
        self.previous_decode_us[index] = micros;
        self.decode_samples[index] = self.decode_samples[index].saturating_add(1);
    }

    fn prefill_rate(&self, tokens: usize) -> Option<u64> {
        nearest_rate(&self.prefill_ns_per_token, tokens)
    }

    /// The largest 512-token multiple up to 8192 whose estimated prefill time
    /// fits `target` beside `decode` decode rows, or 512 when none fits or a
    /// measurement is missing.
    pub(super) fn budget(&self, decode: usize, target: PrefillTarget) -> usize {
        let Some(time) = qualified(&self.decode_us, &self.decode_samples, decode) else {
            return CHUNK;
        };
        let allowance_us = match target {
            PrefillTarget::DecodeSteps(steps) => u128::from(time) * u128::from(steps),
            PrefillTarget::ItlMillis(millis) => {
                (u128::from(millis) * 1000).saturating_sub(u128::from(time))
            }
        };
        // Use the closest measured token bucket for each candidate. Round
        // down to full 512-token chunks, never extrapolate above 8192.
        (1..=16)
            .rev()
            .map(|n| n * CHUNK)
            .find(|&tokens| {
                let rate = self.prefill_rate(tokens);
                rate.is_some_and(|rate| tokens as u128 * u128::from(rate) <= allowance_us * 1000)
            })
            .unwrap_or(CHUNK)
    }

    pub(super) fn estimates(&self) -> StepTimeEstimates {
        StepTimeEstimates {
            decode_us: self.decode_us,
            decode_samples: self.decode_samples,
            prefill_ns_per_token: self.prefill_ns_per_token,
        }
    }
}

/// A copy of the decode-aware estimator's state, for metrics and traces.
///
/// Bucket `i` covers counts in `(2^(i-1), 2^i]`; bucket 0 covers 0 and 1.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StepTimeEstimates {
    decode_us: [u64; BUCKETS],
    decode_samples: [u8; BUCKETS],
    prefill_ns_per_token: [u64; BUCKETS],
}

fn upper_bound(index: usize) -> usize {
    1_usize.checked_shl(index as u32).unwrap_or(usize::MAX)
}

impl StepTimeEstimates {
    /// Measured decode buckets as (largest row count, microseconds, samples).
    /// An estimate needs two samples before it can raise the cap.
    pub fn decode(&self) -> impl Iterator<Item = (usize, u64, u8)> + '_ {
        (0..BUCKETS)
            .filter(|index| self.decode_us[*index] != 0)
            .map(|index| {
                (
                    upper_bound(index),
                    self.decode_us[index],
                    self.decode_samples[index],
                )
            })
    }

    /// Measured prefill buckets as (largest step token count, nanoseconds per token).
    pub fn prefill(&self) -> impl Iterator<Item = (usize, u64)> + '_ {
        (0..BUCKETS)
            .filter(|index| self.prefill_ns_per_token[*index] != 0)
            .map(|index| (upper_bound(index), self.prefill_ns_per_token[index]))
    }

    /// The qualified decode estimate the cap uses for `rows`, in microseconds.
    #[must_use]
    pub fn decode_micros(&self, rows: usize) -> Option<u64> {
        qualified(&self.decode_us, &self.decode_samples, rows)
    }

    /// The step time the cap assumes for `prefill` tokens beside `decode`
    /// rows, in microseconds, or None while an estimate it needs is missing.
    #[must_use]
    pub fn predicted_micros(&self, prefill: usize, decode: usize) -> Option<u64> {
        let decode = if decode == 0 {
            0
        } else {
            self.decode_micros(decode)?
        };
        let prefill = if prefill == 0 {
            0
        } else {
            let rate = nearest_rate(&self.prefill_ns_per_token, prefill)?;
            (prefill as u128 * u128::from(rate) / 1000).min(u128::from(u64::MAX)) as u64
        };
        Some(decode.saturating_add(prefill))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// The first decode-aware policy: prefill costs at most one decode step.
    const ORIGINAL: PrefillTarget = PrefillTarget::DecodeSteps(1);
    #[test]
    fn capture_outlier_cannot_lock_the_budget_above_steady_decode_cost() {
        let mut times = StepTimes::default();
        times.observe(2048, 0, 32768);
        times.observe(0, 32, 1_000_000);
        assert_eq!(
            times.budget(32, ORIGINAL),
            512,
            "an unqualified first decode is cold"
        );
        for _ in 0..100 {
            times.observe(6144, 32, 118304);
        }
        assert_eq!(
            times.budget(32, ORIGINAL),
            1024,
            "mixed traffic recovers a 20ms decode"
        );
    }

    #[test]
    fn steady_decode_qualifies_and_isolated_recapture_does_not_inflate_it() {
        let mut times = StepTimes::default();
        times.observe(2048, 0, 32768);
        times.observe(0, 32, 20_000);
        assert_eq!(times.budget(32, ORIGINAL), 512);
        times.observe(0, 32, 20_000);
        assert_eq!(times.budget(32, ORIGINAL), 1024);
        times.observe(0, 32, 1_000_000);
        assert_eq!(times.budget(32, ORIGINAL), 1024);
        times.observe(0, 32, 20_000);
        assert_eq!(times.budget(32, ORIGINAL), 1024);
        // A sustained cost change, including an eager execution path, learns.
        for _ in 0..64 {
            times.observe(0, 32, 40_000);
        }
        assert_eq!(times.budget(32, ORIGINAL), 2048);
    }

    #[test]
    fn defaults_floor_ceiling_and_bucketed_ewma() {
        let mut times = StepTimes::default();
        assert_eq!(times.budget(32, ORIGINAL), 512);
        times.observe(2048, 0, 2048);
        times.observe(0, 32, 4096);
        times.observe(0, 32, 4096);
        assert_eq!(times.budget(32, ORIGINAL), 4096);
        assert_eq!(times.budget(64, ORIGINAL), 512);
        times.observe(2048, 32, 12288);
        assert!(times.budget(32, ORIGINAL) < 4096);
        times.observe(0, 64, 1_000_000);
        times.observe(0, 64, 1_000_000);
        assert_eq!(times.budget(64, ORIGINAL), 8192);
        times.observe(0, 1, 1);
        assert_eq!(times.budget(1, ORIGINAL), 512);
        let before = times.clone();
        times.observe(512, 256, 50);
        times.observe(0, 32, 0);
        assert_eq!(times, before);
    }

    #[test]
    fn itl_target_bounds_the_whole_mixed_step() {
        let mut times = StepTimes::default();
        // 50 us per token, 20 ms decode steps.
        times.observe(2048, 0, 102_400);
        times.observe(0, 32, 20_000);
        times.observe(0, 32, 20_000);
        let target = |millis| times.budget(32, PrefillTarget::ItlMillis(millis));
        // 20 ms decode plus 512 tokens (25.6 ms) is 45.6 ms.
        assert_eq!(target(46), 512);
        assert_eq!(target(71), 512);
        assert_eq!(target(72), 1024);
        assert_eq!(target(200), 3584);
        assert_eq!(target(1_000), 8192);
        // A target the decode step alone exceeds keeps the floor.
        assert_eq!(target(20), 512);
        assert_eq!(target(0), 512);
        let steps = |steps| times.budget(32, PrefillTarget::DecodeSteps(steps));
        assert_eq!(steps(1), 512);
        assert_eq!(steps(2), 512);
        assert_eq!(steps(3), 1024);
        assert_eq!(steps(0), 512);
        assert_eq!(
            PrefillTarget::default(),
            PrefillTarget::ItlMillis(crate::DEFAULT_PREFILL_ITL_TARGET_MS)
        );
    }

    /// Mixed steps in the 2026-09-26 TP4 soak receipts fit 16.4 ms of decode
    /// plus 27 ms and 64.4 us per prefill token. One decode step never fits
    /// the fixed 27 ms, so the original target stays at 512 tokens; the
    /// default 200 ms target settles on 2048 tokens, a 175 ms mixed step.
    #[test]
    fn default_target_settles_on_2048_tokens_under_the_measured_cost_fit() {
        let mixed = |tokens: usize| 16_400 + 27_000 + tokens as u64 * 644 / 10;
        for (target, settled) in [
            (PrefillTarget::default(), 2048),
            (PrefillTarget::DecodeSteps(1), 512),
        ] {
            let mut times = StepTimes::default();
            times.observe(0, 32, 16_400);
            times.observe(0, 32, 16_400);
            let mut selected = Vec::new();
            for step in 0..200 {
                let cap = times.budget(32, target);
                selected.push(cap);
                // Prompt tails leave smaller steps now and then.
                let tokens = if step % 8 == 7 { cap.min(1033) } else { cap };
                times.observe(tokens, 32, mixed(tokens));
                times.observe(0, 32, 16_400);
            }
            assert!(
                selected[100..].iter().all(|cap| *cap == settled),
                "{target:?}: {selected:?}"
            );
        }
        assert_eq!(mixed(2048) / 1000, 175);
    }

    #[test]
    fn estimates_report_buckets_and_the_assumed_step_time() {
        let mut times = StepTimes::default();
        assert_eq!(times.estimates().predicted_micros(0, 0), Some(0));
        assert_eq!(times.estimates().predicted_micros(512, 0), None);
        times.observe(2048, 0, 102_400);
        times.observe(0, 20, 20_000);
        let estimates = times.estimates();
        assert_eq!(estimates.decode().collect::<Vec<_>>(), [(32, 20_000, 1)]);
        assert_eq!(estimates.prefill().collect::<Vec<_>>(), [(2048, 50_000)]);
        assert_eq!(estimates.decode_micros(20), None, "one sample is cold");
        assert_eq!(estimates.predicted_micros(1024, 0), Some(51_200));
        assert_eq!(estimates.predicted_micros(1024, 20), None);
        times.observe(0, 20, 20_000);
        let estimates = times.estimates();
        assert_eq!(estimates.decode_micros(32), Some(20_000));
        assert_eq!(estimates.predicted_micros(1024, 20), Some(71_200));
        assert_eq!(estimates.predicted_micros(0, 17), Some(20_000));
    }
}
