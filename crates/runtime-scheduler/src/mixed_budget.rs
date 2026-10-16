//! The decode stall bound of `PrefillBudget::Tbt`: an online step-time model
//! and the largest prefill that fits the target beside the decoding items.
//!
//! A step of `d` decode or verify items, `P` prefill tokens and attention
//! work `A` (query-key pairs of the prefill chunks, cached context included)
//! is modelled as `B(d) + F_r + M [d > 0] + a P + c A`, where the segment
//! `r` is whether `P` reaches `SPLIT_TOKENS`. `B` is a per power-of-two item
//! bucket EWMA over steps without prefill, verify steps included. `F_r`,
//! `M`, `a` and `c` are the nonnegative least squares fit of `T - B(d)`
//! over the latest prefill steps. The fit keeps exact integer sums, so its
//! cost per observation is constant and independent of concurrency: at most
//! 31 solves of at most 5x5 systems.
use crate::MixedPolicy;
use std::collections::VecDeque;

const BUCKETS: usize = usize::BITS as usize + 1;
/// Prefill budgets are multiples of this many tokens, tails excepted.
pub(crate) const GRANULE: usize = 128;
/// Prefill steps the fit remembers.
const WINDOW: usize = 128;
/// Prefill steps needed before the fit sets a budget.
pub(crate) const MIN_FIT_SAMPLES: usize = 8;
/// Steps of at least this many prefill tokens get their own fixed term. An
/// executor may change strategy with the step size: one TP4 bundle runs a
/// prefill of 1024 or more tokens as two pipelined halves, with a higher
/// fixed and a lower per-token cost. One shared slope keeps each segment
/// identifiable from a single chunk size; where nothing changes, both terms
/// agree.
pub(crate) const SPLIT_TOKENS: usize = 1024;
/// The pull towards the previous fit, per sample in the window, in fit
/// units. An identified direction with the curvature of 128 samples moves
/// at most about 0.1% towards the previous fit.
const PRIOR_WEIGHT: f64 = 1e-3;
/// A segment's own fixed term predicts only after this many of its steps.
const SEGMENT_SAMPLES: i128 = 2;
const FEATURES: usize = 5;
/// Features in fit units: the ones of each segment, mixed, kilotokens and
/// mega-pairs; the response is in milliseconds.
const SCALE: [f64; FEATURES] = [1.0, 1.0, 1.0, 1e-3, 1e-6];

/// What one step executed, as the model sees it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StepShape {
    prefill_tokens: usize,
    prefill_pairs: u64,
    decode_items: usize,
}

impl StepShape {
    #[must_use]
    pub const fn new(prefill_tokens: usize, prefill_pairs: u64, decode_items: usize) -> Self {
        Self {
            prefill_tokens,
            prefill_pairs,
            decode_items,
        }
    }

    /// Prompt tokens the step prefilled.
    #[must_use]
    pub const fn prefill_tokens(self) -> usize {
        self.prefill_tokens
    }

    /// Query-key pairs the prefill chunks attend over: `p s + p (p + 1) / 2`
    /// for a chunk of `p` tokens after `s` cached or prefilled tokens.
    #[must_use]
    pub const fn prefill_pairs(self) -> u64 {
        self.prefill_pairs
    }

    /// Decode and verify items.
    #[must_use]
    pub const fn decode_items(self) -> usize {
        self.decode_items
    }
}

/// Attention pairs of a `tokens` chunk that starts after `context` tokens.
#[must_use]
pub(crate) fn chunk_pairs(context: usize, tokens: usize) -> u64 {
    let context = context as u128;
    let tokens = tokens as u128;
    (tokens * context + tokens * (tokens + 1) / 2).min(u128::from(u64::MAX)) as u64
}

fn bucket(count: usize) -> usize {
    (usize::BITS - count.saturating_sub(1).leading_zeros()) as usize
}

fn upper_bound(index: usize) -> usize {
    1_usize.checked_shl(index as u32).unwrap_or(usize::MAX)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Sample {
    tokens: u64,
    pairs: u64,
    /// Decode and verify items beside the prefill.
    items: usize,
    /// The step ran beside decode items without a decode estimate, so
    /// `residual` still includes their time.
    uncorrected: bool,
    /// Step time minus the decode estimate, microseconds.
    residual: i64,
}

impl Sample {
    fn features(self) -> [i128; FEATURES] {
        let large = self.tokens >= SPLIT_TOKENS as u64;
        [
            i128::from(!large),
            i128::from(large),
            i128::from(self.items > 0),
            i128::from(self.tokens),
            i128::from(self.pairs),
        ]
    }
}

/// Coefficients in integer units, so the model state compares exactly.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Fit {
    /// Per segment: below and from `SPLIT_TOKENS` prefill tokens.
    fixed_ps: [u64; 2],
    token_ps: u64,
    mixed_ps: u64,
    pair_fs: u64,
    /// Segments that have had enough steps to predict from their own terms.
    seen: [bool; 2],
}

impl Fit {
    fn residual_micros(self, tokens: usize, pairs: u64, mixed: bool) -> u64 {
        let own = usize::from(tokens >= SPLIT_TOKENS);
        // An unseen segment borrows the other's term until it has steps.
        let segment = if self.seen[own] { own } else { 1 - own };
        let picos = u128::from(self.fixed_ps[segment])
            + if mixed { u128::from(self.mixed_ps) } else { 0 }
            + tokens as u128 * u128::from(self.token_ps)
            + u128::from(pairs) * u128::from(self.pair_fs) / 1000;
        (picos / 1_000_000).min(u128::from(u64::MAX)) as u64
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct StepModel {
    decode_us: [u64; BUCKETS],
    previous_decode_us: [u64; BUCKETS],
    decode_samples: [u8; BUCKETS],
    window: VecDeque<Sample>,
    /// Exact sums over `window`: `x x^T`, `x y` and `y y`.
    xtx: [[i128; FEATURES]; FEATURES],
    xty: [i128; FEATURES],
    yty: i128,
    /// An isolated slow prefill step held back until the next one shows
    /// whether the cost really changed.
    suspect: Option<Sample>,
    fit: Option<Fit>,
}

impl Default for StepModel {
    fn default() -> Self {
        Self {
            decode_us: [0; BUCKETS],
            previous_decode_us: [0; BUCKETS],
            decode_samples: [0; BUCKETS],
            window: VecDeque::with_capacity(WINDOW),
            xtx: [[0; FEATURES]; FEATURES],
            xty: [0; FEATURES],
            yty: 0,
            suspect: None,
            fit: None,
        }
    }
}

impl StepModel {
    pub(super) fn observe(&mut self, shape: StepShape, micros: u64) {
        if micros == 0 {
            return;
        }
        if shape.prefill_tokens == 0 {
            if shape.decode_items > 0 {
                self.observe_decode(shape.decode_items, micros);
            }
            return;
        }
        let micros_signed = i64::try_from(micros).unwrap_or(i64::MAX);
        let decode = if shape.decode_items == 0 {
            Some(0)
        } else {
            self.subtractable_decode(shape.decode_items, micros_signed)
        };
        let sample = Sample {
            tokens: shape.prefill_tokens as u64,
            pairs: shape.prefill_pairs,
            items: shape.decode_items,
            uncorrected: decode.is_none(),
            residual: micros_signed - decode.unwrap_or(0) as i64,
        };
        let predicted = self.fit.map(|fit| {
            fit.residual_micros(shape.prefill_tokens, sample.pairs, shape.decode_items > 0)
        });
        let slow = |factor: u64, extra: u64| {
            predicted.is_some_and(|predicted| {
                sample.residual > 0
                    && sample.residual as u64 > predicted.saturating_mul(factor) + extra
            })
        };
        match self.suspect.take() {
            // Two slow steps in a row: the cost changed, learn both.
            Some(suspect) if slow(2, 0) => {
                self.push(suspect);
                self.push(sample);
            }
            // The held step was a first-use or capture stall.
            Some(_) => self.push(sample),
            None if slow(4, 20_000) => self.suspect = Some(sample),
            None => self.push(sample),
        }
    }

    fn observe_decode(&mut self, items: usize, micros: u64) {
        let index = bucket(items);
        if self.decode_samples[index] == 0 {
            self.decode_us[index] = micros;
        } else {
            // The lower of consecutive samples excludes isolated capture or
            // planning stalls; sustained changes still learn.
            let steady = micros.min(self.previous_decode_us[index]);
            self.decode_us[index] = if self.decode_samples[index] == 1 {
                steady
            } else {
                ((u128::from(self.decode_us[index]) * 7 + u128::from(steady)) / 8) as u64
            }
            .max(1);
        }
        self.previous_decode_us[index] = micros;
        self.decode_samples[index] = self.decode_samples[index].saturating_add(1);
        if self.decode_samples[index] == 2 {
            self.correct_residuals();
        }
    }

    /// Once a decode estimate exists, subtracts it from the prefill samples
    /// that ran beside decodes without one. Otherwise the fit's mixed term
    /// would keep that decode time and predictions would count it twice.
    fn correct_residuals(&mut self) {
        if !self.window.iter().any(|sample| sample.uncorrected) {
            return;
        }
        let mut window = std::mem::take(&mut self.window);
        for sample in &mut window {
            if sample.uncorrected
                && let Some(decode) = self.subtractable_decode(sample.items, sample.residual)
            {
                sample.residual -= decode as i64;
                sample.uncorrected = false;
            }
        }
        self.xtx = [[0; FEATURES]; FEATURES];
        self.xty = [0; FEATURES];
        self.yty = 0;
        for sample in &window {
            self.accumulate(*sample, 1);
        }
        self.window = window;
        if let Some(mut suspect) = self.suspect
            && suspect.uncorrected
            && let Some(decode) = self.subtractable_decode(suspect.items, suspect.residual)
        {
            suspect.residual -= decode as i64;
            suspect.uncorrected = false;
            self.suspect = Some(suspect);
        }
        self.refit();
    }

    fn push(&mut self, sample: Sample) {
        if self.window.len() == WINDOW
            && let Some(old) = self.window.pop_front()
        {
            self.accumulate(old, -1);
        }
        self.window.push_back(sample);
        self.accumulate(sample, 1);
        self.refit();
    }

    /// Fits the window. The first fit, which has no earlier fit to screen
    /// its samples, first drops startup stalls: a sample above four times
    /// what the other samples predict for it, plus 20 ms, as the slow-step
    /// screen does once trained. At most a quarter of the window goes, so
    /// a sustained cost is learned.
    fn refit(&mut self) {
        if self.window.len() < MIN_FIT_SAMPLES {
            self.fit = None;
            return;
        }
        if self.fit.is_none() {
            for _ in 0..self.window.len() / 4 {
                let Some(index) = self.worst_stall() else {
                    break;
                };
                if let Some(sample) = self.window.remove(index) {
                    self.accumulate(sample, -1);
                }
            }
            if self.window.len() < MIN_FIT_SAMPLES {
                return;
            }
        }
        self.fit = self.solve();
    }

    /// The window sample furthest above its leave-one-out prediction, when
    /// it counts as a stall.
    fn worst_stall(&self) -> Option<usize> {
        let mut worst: Option<(usize, f64)> = None;
        for (index, sample) in self.window.iter().enumerate() {
            let (mut xtx, mut xty, mut yty) = (self.xtx, self.xty, self.yty);
            add(&mut xtx, &mut xty, &mut yty, *sample, -1);
            let Some(fit) = self.solve_sums(&xtx, &xty, yty, self.window.len() - 1) else {
                continue;
            };
            let predicted =
                fit.residual_micros(sample.tokens as usize, sample.pairs, sample.items > 0);
            let limit = predicted.saturating_mul(4).saturating_add(20_000);
            if sample.residual > 0 && sample.residual as u64 > limit {
                let excess = sample.residual as f64 / limit as f64;
                if worst.is_none_or(|(_, most)| excess > most) {
                    worst = Some((index, excess));
                }
            }
        }
        worst.map(|(index, _)| index)
    }

    fn accumulate(&mut self, sample: Sample, sign: i128) {
        add(&mut self.xtx, &mut self.xty, &mut self.yty, sample, sign);
    }

    /// Nonnegative least squares by enumerating the active sets, with a
    /// light pull towards the previous fit. Directions the window cannot
    /// identify, such as fixed against per-token cost while every chunk has
    /// the same size, keep what earlier, more varied steps showed instead
    /// of drifting. Returns None unless some subset has finite, nonnegative
    /// coefficients.
    fn solve(&self) -> Option<Fit> {
        self.solve_sums(&self.xtx, &self.xty, self.yty, self.window.len())
    }

    /// `solve` over sums of `samples` samples.
    fn solve_sums(
        &self,
        sum_xtx: &[[i128; FEATURES]; FEATURES],
        sum_xty: &[i128; FEATURES],
        sum_yty: i128,
        samples: usize,
    ) -> Option<Fit> {
        let xtx: [[f64; FEATURES]; FEATURES] = std::array::from_fn(|i| {
            std::array::from_fn(|j| sum_xtx[i][j] as f64 * SCALE[i] * SCALE[j])
        });
        let xty: [f64; FEATURES] = std::array::from_fn(|i| sum_xty[i] as f64 * SCALE[i] * 1e-3);
        let yty = sum_yty as f64 * 1e-6;
        let prior = self.fit.map_or([0.0; FEATURES], |fit| {
            [
                fit.fixed_ps[0] as f64 / 1e9,
                fit.fixed_ps[1] as f64 / 1e9,
                fit.mixed_ps as f64 / 1e9,
                fit.token_ps as f64 / (1e9 * SCALE[3]),
                fit.pair_fs as f64 / (1e12 * SCALE[4]),
            ]
        });
        let pull = PRIOR_WEIGHT * samples as f64;
        // All coefficients zero is a feasible solution too, the optimum
        // when every term correlates negatively with the residuals.
        let zero = yty + pull * prior.iter().map(|value| value * value).sum::<f64>();
        let mut best: Option<(f64, [f64; FEATURES])> = Some((zero, [0.0; FEATURES]));
        for mask in 1_usize..(1 << FEATURES) {
            let Some(beta) = solve_subset(&xtx, &xty, &prior, pull, mask) else {
                continue;
            };
            let mut sse = yty;
            for i in 0..FEATURES {
                sse -= 2.0 * beta[i] * xty[i];
                sse += pull * (beta[i] - prior[i]).powi(2);
                for j in 0..FEATURES {
                    sse += beta[i] * xtx[i][j] * beta[j];
                }
            }
            if sse.is_finite() && best.is_none_or(|(least, _)| sse < least) {
                best = Some((sse, beta));
            }
        }
        let (_, beta) = best?;
        // Milliseconds per unit into picoseconds (femtoseconds per pair).
        let convert = |value: f64, per_unit: f64| -> Option<u64> {
            let scaled = (value * per_unit).round();
            (scaled.is_finite() && scaled >= 0.0 && scaled < u64::MAX as f64)
                .then_some(scaled as u64)
        };
        Some(Fit {
            fixed_ps: [convert(beta[0], 1e9)?, convert(beta[1], 1e9)?],
            token_ps: convert(beta[3], 1e9 * SCALE[3])?,
            mixed_ps: convert(beta[2], 1e9)?,
            pair_fs: convert(beta[4], 1e12 * SCALE[4])?,
            // A segment once seen keeps its term, held by the pull towards
            // the previous fit after its steps leave the window.
            seen: [0, 1].map(|segment| {
                self.fit.is_some_and(|fit| fit.seen[segment])
                    || sum_xtx[segment][segment] >= SEGMENT_SAMPLES
            }),
        })
    }

    /// The decode estimate to subtract from a step of `micros` beside
    /// `items` items. An estimate borrowed from another bucket that the whole
    /// step contradicts is not used; the sample waits for its own bucket.
    fn subtractable_decode(&self, items: usize, micros: i64) -> Option<u64> {
        let own = bucket(items);
        if self.decode_samples[own] >= 2 {
            return Some(self.decode_us[own]);
        }
        self.decode_micros(items)
            .filter(|&decode| i64::try_from(decode).is_ok_and(|decode| decode < micros))
    }

    /// Whether the fit sets budgets.
    pub(super) fn trained(&self) -> bool {
        self.fit.is_some()
    }

    /// The decode estimate for `items`, from the nearest bucket with two
    /// samples.
    fn decode_micros(&self, items: usize) -> Option<u64> {
        let target = bucket(items);
        (0..BUCKETS)
            .filter(|index| self.decode_samples[*index] >= 2)
            .min_by_key(|index| index.abs_diff(target))
            .map(|index| self.decode_us[index])
    }

    /// Estimated step time; None until the fit has trained.
    pub(super) fn predicted_micros(&self, shape: StepShape) -> Option<u64> {
        let fit = self.fit?;
        let decode = if shape.decode_items == 0 {
            0
        } else {
            self.decode_micros(shape.decode_items).unwrap_or(0)
        };
        Some(decode.saturating_add(fit.residual_micros(
            shape.prefill_tokens,
            shape.prefill_pairs,
            shape.decode_items > 0,
        )))
    }

    /// The largest multiple of `GRANULE` up to `maximum` tokens whose chunk
    /// after `context` tokens fits the target beside `decode` items, never
    /// below the policy floor. None until trained: the caller then uses the
    /// progress floor, bounded by the available token budget.
    pub(super) fn budget(
        &self,
        decode: usize,
        context: usize,
        maximum: usize,
        policy: MixedPolicy,
    ) -> Option<usize> {
        self.fit?;
        let target = u64::from(policy.target_millis()) * 1000;
        let predict = |tokens: usize| {
            self.predicted_micros(StepShape::new(tokens, chunk_pairs(context, tokens), decode))
        };
        let largest = (1..=maximum / GRANULE)
            .rev()
            .map(|n| n * GRANULE)
            .find(|&tokens| predict(tokens).is_some_and(|micros| micros <= target));
        let floor = policy.min_prefill_tokens().min(maximum);
        Some(largest.unwrap_or(floor).max(floor))
    }

    pub(super) fn estimates(&self) -> MixedEstimates {
        let fit = self.fit.unwrap_or_default();
        MixedEstimates {
            decode_us: self.decode_us,
            decode_samples: self.decode_samples,
            samples: self.window.len(),
            trained: self.fit.is_some(),
            fit,
        }
    }
}

/// Adds `sign` times one sample to the sums `x x^T`, `x y` and `y y`.
fn add(
    xtx: &mut [[i128; FEATURES]; FEATURES],
    xty: &mut [i128; FEATURES],
    yty: &mut i128,
    sample: Sample,
    sign: i128,
) {
    let x = sample.features();
    let y = i128::from(sample.residual);
    for i in 0..FEATURES {
        for j in 0..FEATURES {
            xtx[i][j] += sign * x[i] * x[j];
        }
        xty[i] += sign * x[i] * y;
    }
    *yty += sign * y * y;
}

/// Solves the normal equations restricted to the features in `mask`, with
/// a pull of weight `pull` towards `prior` that keeps collinear columns
/// solvable.
fn solve_subset(
    xtx: &[[f64; FEATURES]; FEATURES],
    xty: &[f64; FEATURES],
    prior: &[f64; FEATURES],
    pull: f64,
    mask: usize,
) -> Option<[f64; FEATURES]> {
    let columns: Vec<usize> = (0..FEATURES).filter(|i| mask & (1 << i) != 0).collect();
    let n = columns.len();
    let mut a = [[0.0_f64; FEATURES + 1]; FEATURES];
    for (row, &i) in columns.iter().enumerate() {
        for (column, &j) in columns.iter().enumerate() {
            a[row][column] = xtx[i][j];
        }
        a[row][row] += pull;
        a[row][n] = xty[i] + pull * prior[i];
    }
    for pivot in 0..n {
        let best = (pivot..n).max_by(|l, r| a[*l][pivot].abs().total_cmp(&a[*r][pivot].abs()))?;
        a.swap(pivot, best);
        if a[pivot][pivot].is_nan() || a[pivot][pivot].abs() <= 1e-12 {
            return None;
        }
        for row in 0..n {
            if row != pivot {
                let factor = a[row][pivot] / a[pivot][pivot];
                for column in pivot..=n {
                    a[row][column] -= factor * a[pivot][column];
                }
            }
        }
    }
    let mut beta = [0.0; FEATURES];
    for (row, &i) in columns.iter().enumerate() {
        let value = a[row][n] / a[row][row];
        if !value.is_finite() || value < 0.0 {
            return None;
        }
        beta[i] = value;
    }
    Some(beta)
}

/// A copy of the TBT model, for metrics and traces.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MixedEstimates {
    decode_us: [u64; BUCKETS],
    decode_samples: [u8; BUCKETS],
    samples: usize,
    trained: bool,
    fit: Fit,
}

impl MixedEstimates {
    /// Decode buckets as (largest item count, microseconds, samples).
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

    /// Prefill steps in the fit window.
    #[must_use]
    pub const fn samples(&self) -> usize {
        self.samples
    }

    /// Whether the fit sets budgets yet.
    #[must_use]
    pub const fn trained(&self) -> bool {
        self.trained
    }

    /// Fit terms in seconds: fixed per prefill step below and from
    /// `SPLIT_TOKENS` tokens (`small`, `large`), extra fixed when the step
    /// also decodes, per prefill token and per attention pair.
    #[must_use]
    pub fn terms_seconds(&self) -> [(&'static str, f64); 5] {
        let fit = self.fit;
        [
            ("fixed_small", fit.fixed_ps[0] as f64 * 1e-12),
            ("fixed_large", fit.fixed_ps[1] as f64 * 1e-12),
            ("mixed", fit.mixed_ps as f64 * 1e-12),
            ("token", fit.token_ps as f64 * 1e-12),
            ("pair", fit.pair_fs as f64 * 1e-15),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(target: u32) -> MixedPolicy {
        MixedPolicy::new(target, 512, 0).unwrap()
    }

    /// The model's form near a measured TP4 bundle's fits: 32.6 ms
    /// fixed below 1024 tokens, 48 ms from 1024 (two pipelined halves), and
    /// 77 us per token. A step beside decodes adds 14 ms of decode and a 20
    /// ms surcharge; attention costs 0.4 ns per pair.
    fn cost(tokens: usize, context: usize, decode: usize) -> u64 {
        let decode_part = if decode > 0 { 14_000 + 20_000 } else { 0 };
        let fixed = if tokens >= SPLIT_TOKENS {
            48_000
        } else {
            32_600
        };
        decode_part + fixed + tokens as u64 * 77 + chunk_pairs(context, tokens) * 4 / 10_000
    }

    /// PREFILL-COST's measured fits: 32.6 ms and 84 us per token below 1024
    /// tokens, 41.8 ms and 70 us per token from 1024.
    fn measured(tokens: usize) -> u64 {
        if tokens >= SPLIT_TOKENS {
            41_800 + tokens as u64 * 70
        } else {
            32_600 + tokens as u64 * 84
        }
    }

    fn trained(contexts: &[usize]) -> StepModel {
        let mut model = StepModel::default();
        model.observe(StepShape::new(0, 0, 32), 14_000);
        model.observe(StepShape::new(0, 0, 32), 14_000);
        for (step, &context) in contexts.iter().cycle().take(40).enumerate() {
            let tokens = [2048, 1536, 700, 2048, 333][step % 5];
            let decode = if step % 3 == 0 { 0 } else { 32 };
            let pairs = chunk_pairs(context, tokens);
            model.observe(
                StepShape::new(tokens, pairs, decode),
                cost(tokens, context, decode),
            );
        }
        model
    }

    #[test]
    fn untrained_model_keeps_fixed_chunking() {
        let mut model = StepModel::default();
        assert_eq!(model.budget(32, 0, 2048, policy(100)), None);
        for _ in 0..MIN_FIT_SAMPLES - 1 {
            model.observe(StepShape::new(2048, chunk_pairs(0, 2048), 32), 200_000);
        }
        assert_eq!(model.budget(32, 0, 2048, policy(100)), None);
        assert_eq!(model.predicted_micros(StepShape::new(512, 0, 32)), None);
    }

    #[test]
    fn fit_recovers_fixed_token_and_context_terms() {
        let model = trained(&[0, 8192, 32_768, 65_536]);
        let estimates = model.estimates();
        assert!(estimates.trained());
        let terms = estimates.terms_seconds();
        for ((name, value), expected) in terms.iter().zip([0.0326, 0.048, 0.020, 77e-6, 0.4e-9]) {
            assert!(
                (value - expected).abs() < expected * 2e-3,
                "{name}: {value} against {expected}"
            );
        }
        for (tokens, context) in [(640, 0), (1023, 9000), (1024, 50_000), (2048, 100_000)] {
            let predicted = model
                .predicted_micros(StepShape::new(tokens, chunk_pairs(context, tokens), 32))
                .unwrap();
            assert!(predicted.abs_diff(cost(tokens, context, 32)) < 100);
        }
    }

    #[test]
    fn budget_meets_the_target_and_shrinks_with_context() {
        let model = trained(&[0, 8192, 32_768, 65_536]);
        let at = |target, context| model.budget(32, context, 8192, policy(target)).unwrap();
        let predicted = |tokens, context| {
            model
                .predicted_micros(StepShape::new(tokens, chunk_pairs(context, tokens), 32))
                .unwrap()
        };
        let fits = |tokens, context, target: u64| {
            predicted(tokens, context) <= target * 1000
                && (tokens + GRANULE > 8192 || predicted(tokens + GRANULE, context) > target * 1000)
        };
        // 14 + 20 + 32.6 = 66.6 ms before tokens: 33.4 ms buys 384 tokens.
        assert_eq!(at(100, 0), 512, "the floor, since 512 tokens exceed 100 ms");
        let short = at(130, 0);
        assert_eq!(short, 768);
        assert!(fits(short, 0, 130));
        let long = at(130, 60_000);
        assert!(long < short && predicted(long, 60_000) <= 130_000, "{long}");
        // Two pipelined halves cost 15.4 ms more from 1024 tokens: 896
        // tokens fit 150 ms where 1024 do not.
        assert_eq!(at(150, 0), 896);
        assert!(predicted(1024, 0) > 150_000);
        let large = at(250, 0);
        assert!(large >= SPLIT_TOKENS && fits(large, 0, 250), "{large}");
        // A generous target is bounded by the maximum.
        assert_eq!(at(10_000, 0), 8192);
        // Tighter than the decode step alone: the floor keeps prefill moving.
        assert_eq!(at(1, 0), 512);
        // The floor never exceeds what the step can take.
        assert_eq!(model.budget(32, 0, 300, policy(1)), Some(300));
    }

    #[test]
    fn coefficients_stay_finite_and_nonnegative_on_degenerate_data() {
        let mut model = StepModel::default();
        // Identical samples: every column but one is collinear.
        for _ in 0..WINDOW * 2 {
            model.observe(StepShape::new(2048, chunk_pairs(0, 2048), 32), 180_000);
        }
        let estimates = model.estimates();
        assert!(estimates.trained());
        for (_, value) in estimates.terms_seconds() {
            assert!(value.is_finite() && value >= 0.0);
        }
        let predicted = model
            .predicted_micros(StepShape::new(2048, chunk_pairs(0, 2048), 32))
            .unwrap();
        assert!(predicted.abs_diff(180_000) < 200, "{predicted}");
        // Decreasing cost with more tokens cannot make any term negative.
        let mut model = StepModel::default();
        for step in 0..64_usize {
            let tokens = 128 + (step % 16) * 128;
            model.observe(
                StepShape::new(tokens, chunk_pairs(0, tokens), 32),
                300_000 - tokens as u64 * 100,
            );
        }
        for (_, value) in model.estimates().terms_seconds() {
            assert!(value.is_finite() && value >= 0.0);
        }
        assert!(model.budget(32, 0, 2048, policy(100)).is_some());
    }

    #[test]
    fn verify_steps_train_the_decode_estimate() {
        let mut model = StepModel::default();
        model.observe(StepShape::new(0, 0, 32), 20_000);
        model.observe(StepShape::new(0, 0, 32), 1_000_000);
        model.observe(StepShape::new(0, 0, 32), 20_000);
        assert_eq!(model.decode_micros(32), Some(20_000));
        // The nearest measured bucket stands in for an unmeasured one.
        assert_eq!(model.decode_micros(64), Some(20_000));
        assert_eq!(model.decode_micros(1), Some(20_000));
    }

    #[test]
    fn measured_two_slope_costs_fit_within_a_few_milliseconds() {
        let mut model = StepModel::default();
        model.observe(StepShape::new(0, 0, 32), 14_000);
        model.observe(StepShape::new(0, 0, 32), 14_000);
        for step in 0..WINDOW {
            let tokens = 256 + (step * 97) % 1793;
            model.observe(StepShape::new(tokens, 0, 32), 14_000 + measured(tokens));
        }
        for tokens in (256..=2048).step_by(128) {
            let predicted = model
                .predicted_micros(StepShape::new(tokens, 0, 32))
                .unwrap();
            let actual = 14_000 + measured(tokens);
            assert!(
                predicted.abs_diff(actual) < 5_000,
                "{tokens}: {predicted} {actual}"
            );
        }
    }

    #[test]
    fn a_window_of_one_chunk_size_keeps_the_per_token_cost() {
        let mut model = trained(&[0, 8192, 32_768]);
        // Steady state: every chunk has the same size, so the window alone
        // cannot tell fixed from per-token cost.
        for step in 0..WINDOW * 2 {
            let context = step * 640;
            let shape = StepShape::new(640, chunk_pairs(context, 640), 32);
            model.observe(shape, cost(640, context, 32));
        }
        for tokens in [256, 1536] {
            let shape = StepShape::new(tokens, chunk_pairs(0, tokens), 32);
            let predicted = model.predicted_micros(shape).unwrap();
            let actual = cost(tokens, 0, 32);
            assert!(
                predicted.abs_diff(actual) < actual / 100,
                "{tokens}: {predicted} {actual}"
            );
        }
    }

    #[test]
    fn a_startup_stall_before_the_first_fit_is_dropped() {
        let chunk = |step: usize| [2048, 1536, 512, 768][step % 4];
        let steady = |tokens: usize| 50_000 + tokens as u64 * 70;
        let run = |first: u64| {
            let mut model = StepModel::default();
            for step in 0..40 {
                let tokens = chunk(step);
                let micros = if step == 0 { first } else { steady(tokens) };
                model.observe(StepShape::new(tokens, 0, 0), micros);
            }
            model
        };
        let control = run(steady(2048));
        let stalled = run(5_000_000);
        let policy = policy(150);
        assert_eq!(
            stalled.budget(0, 0, 8192, policy),
            control.budget(0, 0, 8192, policy)
        );
        assert_eq!(control.budget(0, 0, 8192, policy), Some(1408));
        let [.., token, _] = stalled.estimates().terms_seconds();
        assert!((token.1 - 70e-6).abs() < 1e-6, "{token:?}");
        // A sustained cost is not a stall: every step at twice the cost.
        let mut model = StepModel::default();
        for step in 0..MIN_FIT_SAMPLES {
            let tokens = chunk(step);
            model.observe(StepShape::new(tokens, 0, 0), 2 * steady(tokens));
        }
        assert_eq!(model.window.len(), MIN_FIT_SAMPLES);
    }

    #[test]
    fn negative_residuals_fit_all_zero_terms_and_keep_the_floor() {
        // Decode-only steps of 32 items take 200 ms; mixed steps of one
        // item take 100 ms, contradicting the borrowed 32-item estimate.
        let mut model = StepModel::default();
        model.observe(StepShape::new(0, 0, 32), 200_000);
        model.observe(StepShape::new(0, 0, 32), 200_000);
        for step in 0..MIN_FIT_SAMPLES {
            let tokens = if step % 2 == 0 { 512 } else { 2048 };
            model.observe(
                StepShape::new(tokens, chunk_pairs(1000, tokens), 1),
                100_000,
            );
        }
        assert!(model.trained());
        // The contradicted borrowed estimate is not subtracted, so the
        // fit keeps the steps' own cost.
        assert!(model.window.iter().all(|sample| sample.residual == 100_000));
        assert!(
            model
                .estimates()
                .terms_seconds()
                .iter()
                .any(|(_, value)| *value > 0.0)
        );
        // With the 32-item bucket's own estimate the residuals are negative
        // in both segments: all-zero terms, and the floor applies.
        let mut model = StepModel::default();
        model.observe(StepShape::new(0, 0, 32), 200_000);
        model.observe(StepShape::new(0, 0, 32), 200_000);
        for step in 0..MIN_FIT_SAMPLES {
            let tokens = if step % 2 == 0 { 512 } else { 2048 };
            model.observe(
                StepShape::new(tokens, chunk_pairs(1000, tokens), 32),
                100_000,
            );
        }
        assert!(model.trained());
        assert!(
            model
                .estimates()
                .terms_seconds()
                .iter()
                .all(|(_, value)| *value == 0.0)
        );
        assert_eq!(model.budget(32, 0, 8192, policy(150)), Some(512));
    }

    #[test]
    fn a_late_decode_estimate_corrects_earlier_mixed_samples() {
        // Every step prefills at first, so no decode estimate exists and
        // the fit folds the 14 ms decode into its mixed term.
        let mut model = StepModel::default();
        for step in 0..40_usize {
            let tokens = [2048, 1536, 700, 2048, 333][step % 5];
            let decode = if step % 3 == 0 { 0 } else { 32 };
            let pairs = chunk_pairs(0, tokens);
            model.observe(
                StepShape::new(tokens, pairs, decode),
                cost(tokens, 0, decode),
            );
        }
        let [_, _, mixed, ..] = model.estimates().terms_seconds();
        assert!((mixed.1 - 0.034).abs() < 1e-4, "{mixed:?}");
        model.observe(StepShape::new(0, 0, 32), 14_000);
        model.observe(StepShape::new(0, 0, 32), 14_000);
        let [_, _, mixed, ..] = model.estimates().terms_seconds();
        assert!((mixed.1 - 0.020).abs() < 1e-4, "{mixed:?}");
        // A trained chunk size, now without the decode counted twice.
        let shape = StepShape::new(1536, chunk_pairs(0, 1536), 32);
        let predicted = model.predicted_micros(shape).unwrap();
        assert!(predicted.abs_diff(cost(1536, 0, 32)) < 100, "{predicted}");
        let mut fresh = StepModel::default();
        for sample in model.window.clone() {
            fresh.accumulate(sample, 1);
        }
        assert_eq!(
            (model.xtx, model.xty, model.yty),
            (fresh.xtx, fresh.xty, fresh.yty)
        );
    }

    #[test]
    fn isolated_stall_is_dropped_but_a_sustained_change_learns() {
        let mut model = trained(&[0]);
        let shape = StepShape::new(2048, chunk_pairs(0, 2048), 32);
        let before = model.predicted_micros(shape).unwrap();
        model.observe(shape, 5_000_000);
        model.observe(shape, cost(2048, 0, 32));
        assert!(model.predicted_micros(shape).unwrap().abs_diff(before) < 100);
        for _ in 0..WINDOW {
            model.observe(shape, 2 * cost(2048, 0, 32) + 50_000);
        }
        let after = model.predicted_micros(shape).unwrap();
        assert!(after > before * 2, "{before} -> {after}");
    }

    #[test]
    fn window_sums_stay_exact_after_eviction() {
        let mut model = trained(&[0, 32_768]);
        for step in 0..WINDOW * 3 {
            let tokens = 256 + step % 7 * 256;
            model.observe(
                StepShape::new(tokens, chunk_pairs(step, tokens), 32),
                90_000,
            );
        }
        let mut fresh = StepModel::default();
        for sample in model.window.clone() {
            fresh.accumulate(sample, 1);
        }
        assert_eq!(model.window.len(), WINDOW);
        assert_eq!(model.xtx, fresh.xtx);
        assert_eq!(model.xty, fresh.xty);
        assert_eq!(model.yty, fresh.yty);
    }
}
