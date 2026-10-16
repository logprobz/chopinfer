//! The limit rules of the quality gates, implemented once: the Python
//! tooling calls them through `eval rescore`'s `limit` op ([`call`];
//! `tools/qual/limits.py`). They were ported from Python with the same
//! results to the bit; `tests/data/limit-cases.json` keeps the boundary
//! cases.
//!
//! Spread rule (variant-spread.v1, v2 below): a check scored against a reference's
//! gate variant is held to the floors of the numerical contract, raised to
//! the reference's own spread. Each other variant, scored against the gate
//! variant like a server, has a mean |dlogprob| and a decisive top-1 rate;
//! the mean limit is the larger of the mean floor and the worst variant mean
//! times the mean factor (1 unless a gate declares one) rounded up to the
//! mean step, the top-1 limit the larger of the top-1
//! floor and the worst variant decisive rate, the near-tie limit the larger
//! of the near-tie floor and the worst variant near-tie rate rounded up to
//! the near-tie step, and the total top-1 limit the larger of the total
//! floor and the worst variant's top-1 disagreement, never rounded. The mean floor is the large-set floor for at
//! least `large_set_positions` scored positions. A gate may cap the mean
//! limit.
//!
//! Category power (variant-spread.v2): the pooled set is judged by its
//! limits alone. A category or media case is too, except a top-1 metric
//! whose limit sits within [`CATEGORY_SIGMAS`] binomial standard errors of
//! the reference BF16 run's own rate there ([`power`]): there it also
//! passes when the paired rule's top-1 clause against the BF16 run on the
//! same positions passes ([`paired_top1`]).
//!
//! Near-tie rule (top1-near-tie.v1): a scored position where the engine's
//! top-1 differs from the reference's is a flip. The reference's logprob gap
//! from its top-1 to the engine's, read from its stored top-k, makes the
//! flip a near tie when at most [`NEAR_TIE_NATS`], and decisive otherwise or
//! when the engine's token is outside the stored top-k. The top-1 limit
//! applies to the decisive rate and the near-tie limit to the near-tie rate,
//! so systematic near-tie flips fail too; all flips together are held to the
//! total limit, so the two allowances cannot add up past the reference's own
//! disagreement or the total floor.
//!
//! Paired rule (paired-margin.v1): a test and a baseline arm scored on the
//! same positions. The test arm may sit at most a mean margin farther from
//! the reference, and its extra top-1 disagreements must stay within a
//! number of standard deviations of the discordant count.
//!
//! Exact rule: two answers are equal bit for bit (floats by their IEEE 754
//! bits); the first difference is reported.

use serde::Serialize;
use serde_json::{Map, Value};

use crate::exact;
use crate::json::{is_float, python_eq, python_number, shortest_digits, truthy};

pub const MEAN_FLOOR: f64 = 0.15;
pub const LARGE_SET_MEAN_FLOOR: f64 = 0.10;
pub const LARGE_SET_POSITIONS: u64 = 1000;
pub const TOP1_FLOOR: f64 = 0.08;
pub const MEAN_STEP: f64 = 0.05;
/// The spread rule's default headroom factor on the worst variant mean.
pub const MEAN_FACTOR: f64 = 1.0;
/// The near-tie limit in nats (`limits.NEAR_TIE_NATS`).
pub const NEAR_TIE_NATS: f64 = 0.3;
/// Near-tie flips are bounded by the top-1 floor, raised to the worst
/// reference variant's near-tie rate rounded up to a whole percent.
pub const NEAR_TIE_FLOOR: f64 = TOP1_FLOOR;
pub const NEAR_TIE_STEP: f64 = 0.01;
/// All top-1 flips, near ties and decisive together, at most this share of
/// a scope, raised to the worst reference variant's exact share, never
/// rounded (`limits.TOP1_TOTAL_FLOOR`).
pub const TOP1_TOTAL_FLOOR: f64 = 0.10;
/// The fields of an answer the exact rule compares per position, then once.
pub const POSITION_FIELDS: [&str; 4] = ["token_ids", "tokens", "logprobs", "top_logprobs"];
pub const ANSWER_FIELDS: [&str; 3] = ["text", "reasoning", "finish_reason"];

/// A rule's input it cannot judge, where Python raises.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuleError(pub String);

impl std::fmt::Display for RuleError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for RuleError {}

/// Python's default `Decimal` context precision, in significant digits.
const PRECISION: usize = 28;

/// A finite decimal: its digits (most significant first, no leading zero,
/// empty for zero) times ten to `exponent`, with a sign.
struct Decimal {
    negative: bool,
    digits: Vec<u8>,
    exponent: i64,
}

impl Decimal {
    /// The shortest decimal that reads back as `value`, as Python's
    /// `Decimal(str(value))` takes it.
    fn shortest(value: f64) -> Self {
        let (digits, exponent) = shortest_digits(value);
        let mut decimal = Self {
            negative: value.is_sign_negative(),
            exponent: i64::from(exponent) + 1 - digits.len() as i64,
            digits: digits.bytes().map(|digit| digit - b'0').collect(),
        };
        decimal.trim();
        decimal
    }

    /// An exact integer, as Python's `Decimal(str(value))` takes an int.
    fn integer(negative: bool, magnitude: u128) -> Self {
        Self {
            negative: negative && magnitude != 0,
            digits: magnitude
                .to_string()
                .bytes()
                .map(|digit| digit - b'0')
                .collect(),
            exponent: 0,
        }
        .trimmed()
    }

    fn trimmed(mut self) -> Self {
        self.trim();
        self
    }

    fn trim(&mut self) {
        let zeros = self.digits.iter().take_while(|digit| **digit == 0).count();
        self.digits.drain(..zeros);
    }

    /// The exponent of the leading digit (`Decimal.adjusted`).
    fn adjusted(&self) -> i64 {
        self.exponent + self.digits.len() as i64 - 1
    }
}

impl std::fmt::Display for Decimal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let digits: String = self
            .digits
            .iter()
            .map(|digit| char::from(b'0' + digit))
            .collect();
        let sign = if self.negative { "-" } else { "" };
        if digits.is_empty() {
            write!(formatter, "{sign}0")
        } else {
            write!(formatter, "{sign}{digits}E{}", self.exponent)
        }
    }
}

/// Whether the digits `left` are less than `right`, both without leading
/// zeros.
fn less(left: &[u8], right: &[u8]) -> bool {
    left.len() < right.len() || (left.len() == right.len() && left < right)
}

/// `left` minus `right`, for `left` at least `right`, without leading zeros.
fn subtract(left: &mut Vec<u8>, right: &[u8]) {
    let offset = left.len() - right.len();
    let mut borrow = 0;
    for index in (0..left.len()).rev() {
        let take = borrow + index.checked_sub(offset).map_or(0, |index| right[index]);
        if left[index] >= take {
            left[index] -= take;
            borrow = 0;
        } else {
            left[index] = left[index] + 10 - take;
            borrow = 1;
        }
    }
    let zeros = left.iter().take_while(|digit| **digit == 0).count();
    left.drain(..zeros);
}

/// `digits` plus one.
fn increment(digits: &mut Vec<u8>) {
    for digit in digits.iter_mut().rev() {
        if *digit == 9 {
            *digit = 0;
        } else {
            *digit += 1;
            return;
        }
    }
    digits.insert(0, 1);
}

/// `digits` minus one, for a positive value.
fn decrement(digits: &mut Vec<u8>) {
    for digit in digits.iter_mut().rev() {
        if *digit == 0 {
            *digit = 9;
        } else {
            *digit -= 1;
            break;
        }
    }
    let zeros = digits.iter().take_while(|digit| **digit == 0).count();
    digits.drain(..zeros);
}

fn multiply(left: &[u8], right: &[u8]) -> Vec<u8> {
    let mut product = vec![0u32; left.len() + right.len()];
    for (i, a) in left.iter().enumerate().rev() {
        for (j, b) in right.iter().enumerate().rev() {
            product[i + j + 1] += u32::from(*a) * u32::from(*b);
        }
    }
    for index in (1..product.len()).rev() {
        product[index - 1] += product[index] / 10;
        product[index] %= 10;
    }
    let product: Vec<u8> = product.into_iter().map(|digit| digit as u8).collect();
    let zeros = product.iter().take_while(|digit| **digit == 0).count();
    product[zeros..].to_vec()
}

/// `value` rounded up to a multiple of `step` exactly as `limits.rounded_up`
/// computes it in Python's default `Decimal` context (28 digits, half even):
/// both read as their shortest decimals, so 0.15 is a multiple of 0.05 while
/// 0.1500000001 and the binary sum 0.1 + 0.05 (0.15000000000000002) round up
/// to 0.2. `divmod` truncates toward zero, only a positive remainder rounds
/// up, and a quotient of more than 28 digits is refused as Python refuses it
/// (`DivisionImpossible`), as is a zero step.
pub fn rounded_up(value: f64, step: f64) -> Result<f64, RuleError> {
    if !value.is_finite() || !step.is_finite() {
        return Err(RuleError(format!(
            "cannot round {value} up to a step of {step}"
        )));
    }
    round_up(&Decimal::shortest(value), &Decimal::shortest(step))
}

/// `rounded_up` over the decimals Python reads: a float's shortest digits
/// or an int's exact ones.
fn round_up(dividend: &Decimal, divisor: &Decimal) -> Result<f64, RuleError> {
    if divisor.digits.is_empty() {
        return Err(RuleError(format!(
            "cannot round {dividend} up to a step of {divisor}"
        )));
    }
    // `Decimal._divide`: the truncated quotient's digits and whether a
    // remainder is left.
    let (mut quotient, remainder) =
        if dividend.digits.is_empty() || dividend.adjusted() - divisor.adjusted() <= -2 {
            (Vec::new(), !dividend.digits.is_empty())
        } else if dividend.adjusted() - divisor.adjusted() > PRECISION as i64 {
            return Err(RuleError(format!(
                "{dividend} is more than 10^28 steps of {divisor}: quotient too large in divmod"
            )));
        } else {
            // Long division of the dividend's digits, padded to the divisor's
            // exponent, by the divisor scaled to the dividend's, digit by
            // digit, as an int's digits are not bounded by a double's.
            let mut scaled = divisor.digits.clone();
            scaled.extend(std::iter::repeat_n(
                0,
                (divisor.exponent - dividend.exponent).max(0) as usize,
            ));
            let padding = (dividend.exponent - divisor.exponent).max(0) as usize;
            let mut quotient = Vec::new();
            let mut rest: Vec<u8> = Vec::new();
            for digit in dividend
                .digits
                .iter()
                .copied()
                .chain(std::iter::repeat_n(0, padding))
            {
                if !rest.is_empty() || digit != 0 {
                    rest.push(digit);
                }
                let mut times = 0;
                while !less(&rest, &scaled) {
                    subtract(&mut rest, &scaled);
                    times += 1;
                }
                quotient.push(times);
            }
            let zeros = quotient.iter().take_while(|digit| **digit == 0).count();
            quotient.drain(..zeros);
            if quotient.len() > PRECISION {
                return Err(RuleError(format!(
                    "{dividend} is more than 10^28 steps of {divisor}: quotient too large in divmod"
                )));
            }
            (quotient, !rest.is_empty())
        };
    // The quotient plus one for a positive remainder, which only a positive
    // value leaves. Adding the int also turns -0 into 0.
    let mut negative = dividend.negative != divisor.negative;
    if remainder && !dividend.negative {
        if negative && !quotient.is_empty() {
            decrement(&mut quotient);
        } else {
            increment(&mut quotient);
            negative = false;
        }
    }
    if quotient.is_empty() {
        negative = false;
    }
    // Times the step, rounded to 28 digits half even.
    let mut product = multiply(&quotient, &divisor.digits);
    let mut exponent = divisor.exponent;
    if product.len() > PRECISION {
        let dropped = product.split_off(PRECISION);
        exponent += dropped.len() as i64;
        let (first, rest) = dropped.split_first().expect("a dropped digit");
        let above_half = *first > 5 || (*first == 5 && rest.iter().any(|digit| *digit != 0));
        let tie_to_odd =
            *first == 5 && rest.iter().all(|digit| *digit == 0) && product[PRECISION - 1] % 2 == 1;
        if above_half || tie_to_odd {
            increment(&mut product);
            if product.len() > PRECISION {
                product.pop();
                exponent += 1;
            }
        }
    }
    let digits: String = if product.is_empty() {
        "0".into()
    } else {
        product
            .iter()
            .map(|digit| char::from(b'0' + digit))
            .collect()
    };
    let sign = if negative != divisor.negative {
        "-"
    } else {
        ""
    };
    Ok(format!("{sign}{digits}e{exponent}")
        .parse()
        .expect("a decimal literal"))
}

/// The spread rule's floors, step and optional cap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpreadRule {
    pub mean_floor: f64,
    pub large_set_mean_floor: f64,
    pub large_set_positions: u64,
    pub top1_floor: f64,
    pub mean_step: f64,
    pub mean_cap: Option<f64>,
    pub mean_factor: f64,
    pub near_tie_floor: f64,
    pub near_tie_step: f64,
    pub total_floor: f64,
}

impl Default for SpreadRule {
    fn default() -> Self {
        Self {
            mean_floor: MEAN_FLOOR,
            large_set_mean_floor: LARGE_SET_MEAN_FLOOR,
            large_set_positions: LARGE_SET_POSITIONS,
            top1_floor: TOP1_FLOOR,
            mean_step: MEAN_STEP,
            mean_cap: None,
            mean_factor: MEAN_FACTOR,
            near_tie_floor: NEAR_TIE_FLOOR,
            near_tie_step: NEAR_TIE_STEP,
            total_floor: TOP1_TOTAL_FLOOR,
        }
    }
}

/// A mean |dlogprob|, decisive top-1 rate and near-tie rate: a variant's
/// spread, a measured score or the limits.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Spread {
    pub mean_abs_dlogprob: f64,
    pub top1_decisive: f64,
    pub top1_near_tie: f64,
    /// The total top-1 disagreement: every flip, near tie or decisive.
    pub top1_disagreement: f64,
}

/// The spread rule's limits of `positions` scored positions. `variants`
/// holds each other reference variant's spread against the gate variant
/// over the same positions.
pub fn spread_limits(
    variants: &[Spread],
    positions: u64,
    rule: &SpreadRule,
) -> Result<Spread, RuleError> {
    let rounded = variants
        .iter()
        .map(|variant| {
            Ok(Spread {
                mean_abs_dlogprob: rounded_up(
                    variant.mean_abs_dlogprob * rule.mean_factor,
                    rule.mean_step,
                )?,
                top1_near_tie: rounded_up(variant.top1_near_tie, rule.near_tie_step)?,
                ..*variant
            })
        })
        .collect::<Result<Vec<_>, RuleError>>()?;
    Ok(spread_over(&rounded, positions, rule))
}

/// The spread rule over variants whose mean and near-tie rate are already
/// rounded up to their steps.
fn spread_over(rounded: &[Spread], positions: u64, rule: &SpreadRule) -> Spread {
    let floor = if positions >= rule.large_set_positions {
        rule.large_set_mean_floor
    } else {
        rule.mean_floor
    };
    let mut mean = floor;
    let mut top1 = rule.top1_floor;
    let mut near_tie = rule.near_tie_floor;
    let mut total = rule.total_floor;
    for variant in rounded {
        mean = python_max(mean, variant.mean_abs_dlogprob);
        top1 = python_max(top1, variant.top1_decisive);
        near_tie = python_max(near_tie, variant.top1_near_tie);
        total = python_max(total, variant.top1_disagreement);
    }
    if let Some(cap) = rule.mean_cap {
        // Python's min(cap, mean) keeps the cap unless the mean is smaller.
        if mean >= cap {
            mean = cap;
        }
    }
    Spread {
        mean_abs_dlogprob: mean,
        top1_decisive: top1,
        top1_near_tie: near_tie,
        top1_disagreement: total,
    }
}

/// Python's `max` of two: the first unless the second is larger.
fn python_max(first: f64, second: f64) -> f64 {
    if second > first { second } else { first }
}

/// Whether measured scores meet spread limits.
pub fn within(measured: &Spread, limits: &Spread) -> bool {
    measured.mean_abs_dlogprob <= limits.mean_abs_dlogprob
        && measured.top1_decisive <= limits.top1_decisive
        && measured.top1_near_tie <= limits.top1_near_tie
        && measured.top1_disagreement <= limits.top1_disagreement
}

/// Whether two tokens `gap` nats apart in logprob are a near tie: the gap is
/// known and at most `limit` in magnitude.
pub fn near_tie(gap: Option<f64>, limit: f64) -> bool {
    gap.is_some_and(|gap| gap.abs() <= limit)
}

/// A flip of the reference's top-1 at one position.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Flip {
    /// The reference's logprob gap from its top-1 to the served token.
    pub gap: Option<f64>,
    pub decisive: bool,
}

/// The near-tie rule at one scored position: `None` when the engine's top-1
/// `served` is the reference's `top1`; otherwise the reference's gap from
/// its top-1 to `served`, read from `top`, its stored top-k there as
/// `[token, logprob]` pairs (`None` when either token is outside it or
/// nothing is stored), and whether the flip is decisive. A token stored
/// twice reads its last logprob, as Python's `dict` does.
pub fn flip(
    top1: &Value,
    served: &Value,
    top: Option<&[(Value, f64)]>,
    limit: f64,
) -> Option<Flip> {
    if python_eq(served, top1) {
        return None;
    }
    let logprob = |token: &Value| {
        top.unwrap_or_default()
            .iter()
            .rev()
            .find(|(stored, _)| python_eq(stored, token))
            .map(|(_, logprob)| *logprob)
    };
    let gap = logprob(top1).zip(logprob(served)).map(|(a, b)| a - b);
    Some(Flip {
        gap,
        decisive: !near_tie(gap, limit),
    })
}

/// One position's stored top-k as (token, value) pairs, best first.
pub type TopRow = Vec<(Value, f64)>;

/// Per position, the top (token ID, value) pairs whose differences are the
/// reference's own gaps in nats (`exact_reference.gap_top`): logits in the
/// runner layout (`topk_ids`, `topk_logits`), logprobs in the oracle's
/// (`top5` pairs). `None` when the scores hold neither.
pub fn gap_top(scores: &Value) -> Option<Vec<TopRow>> {
    let pairs = |row: &Value| -> Option<TopRow> {
        row.as_array()?
            .iter()
            .map(|pair| Some((pair.get(0)?.clone(), python_number(pair.get(1)?)?)))
            .collect()
    };
    if scores.get("topk_ids").is_some() {
        let ids = scores["topk_ids"].as_array()?;
        let logits = scores["topk_logits"].as_array()?;
        if ids.len() != logits.len() {
            return None;
        }
        ids.iter()
            .zip(logits)
            .map(|(ids, logits)| {
                // Each row pairs as far as both go, as Python's zip does.
                let (ids, logits) = (ids.as_array()?, logits.as_array()?);
                ids.iter()
                    .zip(logits)
                    .map(|(id, logit)| Some((id.clone(), python_number(logit)?)))
                    .collect()
            })
            .collect()
    } else {
        scores["top5"].as_array()?.iter().map(pairs).collect()
    }
}

/// Raw, near-tie and decisive top-1 counts and rates.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Top1Summary {
    pub top1_disagreements: u64,
    pub top1_disagreement: Option<f64>,
    pub top1_near_ties: u64,
    pub top1_near_tie: Option<f64>,
    pub top1_decisive_flips: u64,
    pub top1_decisive: Option<f64>,
}

/// The top-1 counts and rates over `scored` positions, from the flips
/// (`None` for agreement) there.
pub fn top1_summary(flips: &[Option<Flip>], scored: u64) -> Top1Summary {
    let flips: Vec<&Flip> = flips.iter().flatten().collect();
    let decisive = flips.iter().filter(|flip| flip.decisive).count() as u64;
    let count = flips.len() as u64;
    let rate = |numerator: u64| (scored != 0).then(|| ratio(numerator, scored));
    Top1Summary {
        top1_disagreements: count,
        top1_disagreement: rate(count),
        top1_near_ties: count - decisive,
        top1_near_tie: rate(count - decisive),
        top1_decisive_flips: decisive,
        top1_decisive: rate(decisive),
    }
}

/// `numerator / denominator` correctly rounded to the nearest double, ties
/// to even, as Python divides two ints: never through the doubles of the
/// operands, which round counts above 2^53.
fn ratio(numerator: u64, denominator: u64) -> f64 {
    const EXACT: u64 = 1 << 53;
    if numerator <= EXACT && denominator <= EXACT {
        // Both are doubles exactly, and IEEE division rounds once.
        numerator as f64 / denominator as f64
    } else if numerator == 0 {
        0.0
    } else {
        rounded_ratio(numerator, denominator)
    }
}

/// `numerator / denominator`, both positive, by integer division: a
/// quotient of 55 or 56 bits and whether a remainder is left, rounded to
/// 53 bits half even.
fn rounded_ratio(numerator: u64, denominator: u64) -> f64 {
    let (n, d) = (u128::from(numerator), u128::from(denominator));
    // At most 2^119 when shifting the numerator, 2^64 the denominator.
    let shift = 55 + d.ilog2() as i32 - n.ilog2() as i32;
    let (quotient, rest) = if shift >= 0 {
        ((n << shift) / d, (n << shift) % d != 0)
    } else {
        (n / (d << -shift), n % (d << -shift) != 0)
    };
    let dropped = quotient.ilog2() + 1 - 53;
    let mut mantissa = quotient >> dropped;
    let tail = quotient & ((1 << dropped) - 1);
    let half = 1 << (dropped - 1);
    if tail > half || (tail == half && (rest || mantissa & 1 == 1)) {
        mantissa += 1;
    }
    // mantissa * 2^(dropped - shift): at most 2^53 times a power of two
    // between 2^-116 and 2^11, both doubles, so the product is exact.
    let exponent = dropped as i32 - shift;
    mantissa as f64 * f64::from_bits(((1023 + exponent) as u64) << 52)
}

/// The paired rule's verdict.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Paired {
    pub mean_gap: f64,
    pub mean_margin: f64,
    pub top1_test_only: u64,
    pub top1_baseline_only: u64,
    pub top1_margin: f64,
    pub passed: bool,
}

/// The paired rule over two arms' (|dlogprob|, disagrees) at the same
/// positions. Under equal accuracy the difference of the discordant top-1
/// counts has standard deviation sqrt(discordant).
pub fn paired(
    test: &[(f64, bool)],
    baseline: &[(f64, bool)],
    mean_margin: f64,
    sigmas: f64,
) -> Result<Paired, RuleError> {
    if test.len() != baseline.len() {
        return Err(RuleError(format!(
            "the arms score {} and {} positions",
            test.len(),
            baseline.len()
        )));
    }
    let gaps: Vec<f64> = test.iter().zip(baseline).map(|(t, b)| t.0 - b.0).collect();
    let mean_gap =
        exact::mean(&gaps).ok_or_else(|| RuleError("the arms score no positions".into()))?;
    let test_only = test
        .iter()
        .zip(baseline)
        .filter(|(t, b)| t.1 && !b.1)
        .count() as u64;
    let baseline_only = test
        .iter()
        .zip(baseline)
        .filter(|(t, b)| b.1 && !t.1)
        .count() as u64;
    let top1 = top1_clause(test_only, baseline_only, sigmas);
    Ok(Paired {
        mean_gap,
        mean_margin,
        top1_test_only: test_only,
        top1_baseline_only: baseline_only,
        top1_margin: top1.margin,
        passed: mean_gap <= mean_margin && top1.passed,
    })
}

/// The top-1 clause of the paired rule over discordant counts.
fn top1_clause(test_only: u64, baseline_only: u64, sigmas: f64) -> PairedTop1 {
    let margin = sigmas * ((test_only + baseline_only) as f64).sqrt();
    PairedTop1 {
        test_only,
        baseline_only,
        margin,
        passed: (test_only as f64 - baseline_only as f64) <= margin,
    }
}

/// The paired rule's top-1 clause on its own.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct PairedTop1 {
    pub test_only: u64,
    pub baseline_only: u64,
    pub margin: f64,
    pub passed: bool,
}

/// The paired rule's top-1 clause over two arms' disagreements at the same
/// positions: the test arm's extra disagreements within `sigmas` standard
/// deviations of the discordant count.
pub fn paired_top1(test: &[bool], baseline: &[bool], sigmas: f64) -> Result<PairedTop1, RuleError> {
    if test.len() != baseline.len() {
        return Err(RuleError(format!(
            "the arms score {} and {} positions",
            test.len(),
            baseline.len()
        )));
    }
    let count =
        |a: &[bool], b: &[bool]| a.iter().zip(b).filter(|(a, b)| **a && !**b).count() as u64;
    Ok(top1_clause(
        count(test, baseline),
        count(baseline, test),
        sigmas,
    ))
}

/// Category power rule (variant-spread.v2): the standard deviations a
/// category's top-1 limit must sit above the reference's own BF16 rate to
/// judge it, and the paired fallback's margin where it cannot (the paired
/// rule's TOP1_SIGMAS).
pub const CATEGORY_SIGMAS: f64 = 2.0;

/// Whether a top-1 limit can judge a set of `positions`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Power {
    /// The reference BF16 run's own rate over the set.
    pub rate: f64,
    pub limit: f64,
    pub positions: u64,
    /// The binomial standard error of `rate` at `positions`.
    pub standard_error: f64,
    pub has_power: bool,
}

/// variant-spread.v2's power test: a top-1 limit judges a set when it sits
/// at least `sigmas` binomial standard errors above the reference's own
/// rate (a zero rate always does). Where it does not, an engine as accurate
/// as the reference fails the limit by chance alone about as often as it
/// passes, so the set falls back to the paired rule against the reference
/// run on the same positions.
pub fn power(rate: f64, limit: f64, positions: u64, sigmas: f64) -> Power {
    let standard_error = if positions == 0 {
        0.0
    } else {
        (rate * (1.0 - rate) / positions as f64).sqrt()
    };
    Power {
        rate,
        limit,
        positions,
        standard_error,
        has_power: rate <= 0.0 || limit - rate >= sigmas * standard_error,
    }
}

/// Equal values, with any float and the number it is compared with equal
/// only in their IEEE 754 bits; otherwise Python's `==`.
pub fn bit_equal(a: &Value, b: &Value) -> bool {
    if is_float(a) || is_float(b) {
        let number = |value: &Value| match value {
            Value::Number(number) => number.as_f64(),
            _ => None,
        };
        return match (number(a), number(b)) {
            (Some(a), Some(b)) => a.to_bits() == b.to_bits(),
            _ => false,
        };
    }
    match (a, b) {
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| bit_equal(a, b))
        }
        (Value::Object(a), Value::Object(b)) => crate::json::objects_eq(a, b, bit_equal),
        _ => python_eq(a, b),
    }
}

/// Where two answers first differ.
#[derive(Clone, Debug, PartialEq)]
pub struct Difference {
    pub field: String,
    /// The position, or `None` for an answer field.
    pub position: Option<usize>,
    pub left: Value,
    pub right: Value,
}

impl Difference {
    /// As Python's tuple `(field, position, left, right)`.
    pub fn to_json(&self) -> Value {
        Value::Array(vec![
            Value::String(self.field.clone()),
            self.position.map_or(Value::Null, Value::from),
            self.left.clone(),
            self.right.clone(),
        ])
    }
}

/// A field's values as Python's `row.get(field) or []` indexes them.
fn positions(row: &Map<String, Value>, field: &str) -> Result<Vec<Value>, RuleError> {
    match row.get(field) {
        Some(value) if !truthy(value) => Ok(Vec::new()),
        None => Ok(Vec::new()),
        Some(Value::Array(items)) => Ok(items.clone()),
        Some(Value::String(text)) => Ok(text.chars().map(|c| Value::String(c.into())).collect()),
        Some(other) => Err(RuleError(format!("{field} holds no positions: {other}"))),
    }
}

/// The exact rule: the first difference of two answers, position by
/// position over `position_fields` and then once over `answer_fields`;
/// `None` when bit-equal.
pub fn first_difference(
    left: &Map<String, Value>,
    right: &Map<String, Value>,
    position_fields: &[&str],
    answer_fields: &[&str],
) -> Result<Option<Difference>, RuleError> {
    let mut columns = Vec::new();
    for field in position_fields {
        columns.push((*field, positions(left, field)?, positions(right, field)?));
    }
    let length = columns
        .iter()
        .map(|(_, a, b)| a.len().max(b.len()))
        .max()
        .unwrap_or(0);
    for index in 0..length {
        for (field, a, b) in &columns {
            let a = a.get(index).cloned().unwrap_or(Value::Null);
            let b = b.get(index).cloned().unwrap_or(Value::Null);
            if !bit_equal(&a, &b) {
                return Ok(Some(Difference {
                    field: (*field).into(),
                    position: Some(index),
                    left: a,
                    right: b,
                }));
            }
        }
    }
    for field in answer_fields {
        let a = left.get(*field).cloned().unwrap_or(Value::Null);
        let b = right.get(*field).cloned().unwrap_or(Value::Null);
        if !bit_equal(&a, &b) {
            return Ok(Some(Difference {
                field: (*field).into(),
                position: None,
                left: a,
                right: b,
            }));
        }
    }
    Ok(None)
}

/// The arguments of a `limit` call: an object holding every `required`
/// key (an explicit null is a value; a missing key is refused) and no key
/// outside `required` and `optional`.
fn arguments<'a>(
    args: &'a Value,
    required: &[&str],
    optional: &[&str],
) -> Result<&'a Map<String, Value>, RuleError> {
    let map = args
        .as_object()
        .ok_or_else(|| RuleError("args: expected an object".into()))?;
    for key in required {
        if !map.contains_key(*key) {
            return Err(RuleError(format!("args: missing {key}")));
        }
    }
    for key in map.keys() {
        if !required.contains(&key.as_str()) && !optional.contains(&key.as_str()) {
            return Err(RuleError(format!("args: unknown argument {key}")));
        }
    }
    Ok(map)
}

/// A number operand of a `limit` call as Python holds it: an exact int
/// (JSON has 64-bit ones here) or a float. A boolean is refused at every
/// level: Python's arithmetic took one as 0 or 1, which turned a mistyped
/// rate into a limit, and no caller passes a flag where a number belongs.
#[derive(Clone, Copy, Debug)]
enum Operand {
    Int(i128),
    Float(f64),
}

impl Operand {
    fn of(value: &Value, what: &str) -> Result<Self, RuleError> {
        let invalid = || RuleError(format!("{what}: expected a number"));
        let Value::Number(number) = value else {
            return Err(invalid());
        };
        if number.is_f64() {
            return number.as_f64().map(Self::Float).ok_or_else(invalid);
        }
        number
            .as_i64()
            .map(i128::from)
            .or_else(|| number.as_u64().map(i128::from))
            .map(Self::Int)
            .ok_or_else(invalid)
    }

    /// The double Python's arithmetic converts an int to (ties to even).
    fn float(self) -> f64 {
        match self {
            Self::Int(value) => value as f64,
            Self::Float(value) => value,
        }
    }

    /// The decimal `Decimal(str(value))` reads.
    fn decimal(self) -> Result<Decimal, RuleError> {
        match self {
            Self::Int(value) => Ok(Decimal::integer(value < 0, value.unsigned_abs())),
            Self::Float(value) if value.is_finite() => Ok(Decimal::shortest(value)),
            Self::Float(value) => Err(RuleError(format!("{value} is no finite decimal"))),
        }
    }

    /// Python's product: exact between ints, otherwise of the doubles.
    fn times(self, other: Self) -> Result<Decimal, RuleError> {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => a
                .unsigned_abs()
                .checked_mul(b.unsigned_abs())
                .map(|product| Decimal::integer((a < 0) != (b < 0), product))
                .ok_or_else(|| RuleError("the product is beyond 2^128".into())),
            _ => Self::Float(self.float() * other.float()).decimal(),
        }
    }
}

/// Where a rule compares or scales a number as a double, an int operand
/// must be one the double holds exactly; where it subtracts two, their
/// difference too. Python kept ints exact there, so a larger one is
/// refused rather than judged differently.
const COMPARED: u128 = 1 << 53;
const SUBTRACTED: u128 = 1 << 52;

/// A number operand the rule reads as a double, an int at most `bound` in
/// magnitude.
fn double(value: &Value, what: &str, bound: u128) -> Result<f64, RuleError> {
    match Operand::of(value, what)? {
        Operand::Float(value) => Ok(value),
        Operand::Int(value) if value.unsigned_abs() <= bound => Ok(value as f64),
        Operand::Int(value) => Err(RuleError(format!(
            "{what}: the integer {value} is beyond the {bound} a double holds exactly here"
        ))),
    }
}

/// A number argument of a `limit` call, read as a double.
fn argument(args: &Value, key: &str) -> Result<f64, RuleError> {
    double(&args[key], key, COMPARED)
}

/// `value`, refused when the rule's arithmetic left the finite numbers,
/// which a JSON result cannot hold.
fn finite(value: f64, field: &str) -> Result<f64, RuleError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(RuleError(format!("{field}: the result is not finite")))
    }
}

/// An object holding every `required` key (an explicit null is a value);
/// other keys are left to the caller, as Python's rules read only theirs.
fn present<'a>(
    value: &'a Value,
    required: &[&str],
    what: &str,
) -> Result<&'a Map<String, Value>, RuleError> {
    let map = value
        .as_object()
        .ok_or_else(|| RuleError(format!("{what}: expected an object")))?;
    match required.iter().find(|key| !map.contains_key(**key)) {
        Some(key) => Err(RuleError(format!("{what}: missing {key}"))),
        None => Ok(map),
    }
}

const SPREAD_FIELDS: [&str; 4] = [
    "mean_abs_dlogprob",
    "top1_decisive",
    "top1_near_tie",
    "top1_disagreement",
];

/// A spread from its JSON object, every rate a double.
fn spread_of(value: &Value) -> Result<Spread, RuleError> {
    present(value, &SPREAD_FIELDS, "spread")?;
    Ok(Spread {
        mean_abs_dlogprob: argument(value, "mean_abs_dlogprob")?,
        top1_decisive: argument(value, "top1_decisive")?,
        top1_near_tie: argument(value, "top1_near_tie")?,
        top1_disagreement: argument(value, "top1_disagreement")?,
    })
}

/// A variant of the spread rule from its JSON object, its mean and
/// near-tie rate rounded up to their steps from the decimals Python read:
/// the exact product of an int mean and factor, an int rate's own digits.
fn variant_of(value: &Value, rule: &Steps) -> Result<Spread, RuleError> {
    present(value, &SPREAD_FIELDS, "spread")?;
    let mean = Operand::of(&value["mean_abs_dlogprob"], "mean_abs_dlogprob")?;
    let near_tie = Operand::of(&value["top1_near_tie"], "top1_near_tie")?;
    Ok(Spread {
        mean_abs_dlogprob: round_up(&mean.times(rule.mean_factor)?, &rule.mean_step.decimal()?)?,
        top1_decisive: argument(value, "top1_decisive")?,
        top1_near_tie: round_up(&near_tie.decimal()?, &rule.near_tie_step.decimal()?)?,
        top1_disagreement: argument(value, "top1_disagreement")?,
    })
}

/// A spread result, refused when a limit is not finite.
fn spread_value(spread: Spread) -> Result<Value, RuleError> {
    finite(spread.mean_abs_dlogprob, "mean_abs_dlogprob")?;
    finite(spread.top1_decisive, "top1_decisive")?;
    finite(spread.top1_near_tie, "top1_near_tie")?;
    finite(spread.top1_disagreement, "top1_disagreement")?;
    Ok(serde_json::to_value(spread).expect("a spread serializes"))
}

/// The operands of the spread rule's rounding, as given.
struct Steps {
    mean_step: Operand,
    mean_factor: Operand,
    near_tie_step: Operand,
}

/// A spread rule: the defaults with each given parameter replaced, and the
/// operands of its rounding as given.
fn rule_of(value: &Value) -> Result<(SpreadRule, Steps), RuleError> {
    let mut rule = SpreadRule::default();
    let mut steps = Steps {
        mean_step: Operand::Float(rule.mean_step),
        mean_factor: Operand::Float(rule.mean_factor),
        near_tie_step: Operand::Float(rule.near_tie_step),
    };
    let Some(map) = value.as_object() else {
        return value
            .is_null()
            .then_some((rule, steps))
            .ok_or_else(|| RuleError("rule: expected an object".into()));
    };
    for (key, value) in map {
        let what = format!("rule.{key}");
        let number = || double(value, &what, COMPARED);
        let operand = || Operand::of(value, &what);
        match key.as_str() {
            "mean_floor" => rule.mean_floor = number()?,
            "large_set_mean_floor" => rule.large_set_mean_floor = number()?,
            "large_set_positions" => {
                rule.large_set_positions = value
                    .as_u64()
                    .ok_or_else(|| RuleError("rule.large_set_positions: expected a count".into()))?
            }
            "top1_floor" => rule.top1_floor = number()?,
            "mean_step" => steps.mean_step = operand()?,
            "mean_cap" => rule.mean_cap = (!value.is_null()).then(number).transpose()?,
            "mean_factor" => steps.mean_factor = operand()?,
            "near_tie_floor" => rule.near_tie_floor = number()?,
            "near_tie_step" => steps.near_tie_step = operand()?,
            "total_floor" => rule.total_floor = number()?,
            other => return Err(RuleError(format!("rule: unknown parameter {other}"))),
        }
    }
    rule.mean_step = steps.mean_step.float();
    rule.mean_factor = steps.mean_factor.float();
    rule.near_tie_step = steps.near_tie_step.float();
    Ok((rule, steps))
}

/// A stored top-k as `[token, logprob]` pairs; `None` for null. The
/// logprobs of two tokens are subtracted.
fn top_of(value: &Value) -> Result<Option<Vec<(Value, f64)>>, RuleError> {
    if value.is_null() {
        return Ok(None);
    }
    let pairs = value
        .as_array()
        .ok_or_else(|| RuleError("top: expected [token, logprob] pairs".into()))?;
    pairs
        .iter()
        .map(|pair| match pair.as_array().map(Vec::as_slice) {
            Some([token, logprob]) => {
                Ok((token.clone(), double(logprob, "top logprob", SUBTRACTED)?))
            }
            _ => Err(RuleError("top: expected [token, logprob] pairs".into())),
        })
        .collect::<Result<_, _>>()
        .map(Some)
}

/// Whether Python can hash a JSON value: a list or a dict it cannot.
fn hashable(value: &Value) -> bool {
    !matches!(value, Value::Array(_) | Value::Object(_))
}

/// One position's flip as JSON (null for agreement): `top1` and `served`
/// must be given (either may be null), `top` may be null or left out.
/// Python looked a flip up in a dict of the stored top-k, so a list or
/// object token there or as the top-1, and as the served token once the
/// top-1 was found, is refused; an agreement never reached the dict.
fn flip_value(args: &Value, limit: f64) -> Result<Value, RuleError> {
    present(args, &["top1", "served"], "position")?;
    let top = top_of(args.get("top").unwrap_or(&Value::Null))?;
    let (top1, served) = (&args["top1"], &args["served"]);
    if !python_eq(served, top1) {
        let stored = top.as_deref().unwrap_or_default();
        if stored.iter().any(|(token, _)| !hashable(token))
            || !hashable(top1)
            || (!hashable(served) && stored.iter().any(|(token, _)| python_eq(token, top1)))
        {
            return Err(RuleError("flip: a token is a list or an object".into()));
        }
    }
    let flipped = flip(top1, served, top.as_deref(), limit);
    if let Some(gap) = flipped.and_then(|flip| flip.gap) {
        finite(gap, "gap")?;
    }
    Ok(serde_json::to_value(flipped).expect("a flip serializes"))
}

/// An arm of the paired rule: (|dlogprob|, disagrees) per position; the
/// two arms' values are subtracted.
fn arm_of(value: &Value, key: &str) -> Result<Vec<(f64, bool)>, RuleError> {
    let invalid = || RuleError(format!("{key}: expected [number, bool] pairs"));
    value[key]
        .as_array()
        .ok_or_else(invalid)?
        .iter()
        .map(|pair| match pair.as_array().map(Vec::as_slice) {
            Some([gap, Value::Bool(disagrees)]) => Ok((double(gap, key, SUBTRACTED)?, *disagrees)),
            _ => Err(invalid()),
        })
        .collect()
}

/// Field names of `first_difference`, or the defaults.
fn fields_of<'a>(value: &'a Value, default: &[&'a str]) -> Result<Vec<&'a str>, RuleError> {
    match value {
        Value::Null => Ok(default.to_vec()),
        Value::Array(items) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .ok_or_else(|| RuleError("fields: expected names".into()))
            })
            .collect(),
        _ => Err(RuleError("fields: expected names".into())),
    }
}

/// The rules' constants, by their Python names.
pub fn constants() -> Value {
    serde_json::json!({
        "MEAN_FLOOR": MEAN_FLOOR, "LARGE_SET_MEAN_FLOOR": LARGE_SET_MEAN_FLOOR,
        "LARGE_SET_POSITIONS": LARGE_SET_POSITIONS, "TOP1_FLOOR": TOP1_FLOOR, "MEAN_STEP": MEAN_STEP,
        "MEAN_FACTOR": MEAN_FACTOR, "NEAR_TIE_FLOOR": NEAR_TIE_FLOOR, "NEAR_TIE_STEP": NEAR_TIE_STEP,
        "TOP1_TOTAL_FLOOR": TOP1_TOTAL_FLOOR, "NEAR_TIE_NATS": NEAR_TIE_NATS,
        "POSITION_FIELDS": POSITION_FIELDS, "ANSWER_FIELDS": ANSWER_FIELDS,
    })
}

/// `eval rescore`'s `limit` op: one rule over JSON arguments, for the
/// Python tooling, which has no implementation of its own. `flips` is
/// `flip` over a list of positions at one limit. Every rule's arguments
/// are an object with its operands present (an explicit null is a value);
/// a number operand is a JSON number, never a boolean, read exactly where
/// Python's arithmetic kept an int exact (rounding, top-1 rates) and
/// refused where a double cannot hold it as Python did; a list or object
/// token is refused where Python hashed it; a result the arithmetic took
/// past the finite numbers is refused.
pub fn call(rule: &str, args: &Value) -> Result<Value, RuleError> {
    let limit = || -> Result<f64, RuleError> {
        if args.get("limit").is_none_or(Value::is_null) {
            Ok(NEAR_TIE_NATS)
        } else {
            argument(args, "limit")
        }
    };
    Ok(match rule {
        "kv_offload" => {
            arguments(args, &["off", "swap", "facts", "speculative"], &[])?;
            crate::kv_offload::judge(args).map_err(RuleError)?
        }
        "constants" => {
            if !args.is_null() {
                arguments(args, &[], &[])?;
            }
            constants()
        }
        "rounded_up" => {
            arguments(args, &["value", "step"], &[])?;
            let rounded = round_up(
                &Operand::of(&args["value"], "value")?.decimal()?,
                &Operand::of(&args["step"], "step")?.decimal()?,
            )?;
            serde_json::json!(finite(rounded, "rounded_up")?)
        }
        "spread_limits" => {
            arguments(args, &["variants", "positions"], &["rule"])?;
            let (rule, steps) = rule_of(args.get("rule").unwrap_or(&Value::Null))?;
            let variants = args["variants"]
                .as_array()
                .ok_or_else(|| RuleError("variants: expected a list".into()))?
                .iter()
                .map(|variant| variant_of(variant, &steps))
                .collect::<Result<Vec<_>, _>>()?;
            let positions = args["positions"]
                .as_u64()
                .ok_or_else(|| RuleError("positions: expected a count".into()))?;
            spread_value(spread_over(&variants, positions, &rule))?
        }
        "within" => {
            arguments(args, &["measured", "limits"], &[])?;
            serde_json::json!(within(
                &spread_of(&args["measured"])?,
                &spread_of(&args["limits"])?
            ))
        }
        "near_tie" => {
            arguments(args, &["gap"], &["limit"])?;
            let gap = (!args["gap"].is_null())
                .then(|| argument(args, "gap"))
                .transpose()?;
            serde_json::json!(near_tie(gap, limit()?))
        }
        "flip" => {
            arguments(args, &["top1", "served"], &["top", "limit"])?;
            flip_value(args, limit()?)?
        }
        "flips" => {
            arguments(args, &["positions"], &["limit"])?;
            let limit = limit()?;
            Value::Array(
                args["positions"]
                    .as_array()
                    .ok_or_else(|| RuleError("positions: expected a list".into()))?
                    .iter()
                    .map(|position| flip_value(position, limit))
                    .collect::<Result<_, _>>()?,
            )
        }
        "top1_summary" => {
            arguments(args, &["flips", "scored"], &[])?;
            let flips = args["flips"]
                .as_array()
                .ok_or_else(|| RuleError("flips: expected a list".into()))?
                .iter()
                .map(|value| match value {
                    Value::Null => Ok(None),
                    value => {
                        present(value, &["decisive"], "flip")?;
                        match value["decisive"] {
                            Value::Bool(decisive) => Ok(Some(Flip {
                                gap: python_number(&value["gap"]),
                                decisive,
                            })),
                            _ => Err(RuleError("flips: expected flips or null".into())),
                        }
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let scored = args["scored"]
                .as_u64()
                .ok_or_else(|| RuleError("scored: expected a count".into()))?;
            serde_json::to_value(top1_summary(&flips, scored)).expect("a summary serializes")
        }
        "paired" => {
            arguments(args, &["test", "baseline", "mean_margin", "sigmas"], &[])?;
            let verdict = paired(
                &arm_of(args, "test")?,
                &arm_of(args, "baseline")?,
                argument(args, "mean_margin")?,
                argument(args, "sigmas")?,
            )?;
            finite(verdict.mean_gap, "mean_gap")?;
            finite(verdict.mean_margin, "mean_margin")?;
            finite(verdict.top1_margin, "top1_margin")?;
            serde_json::to_value(verdict).expect("a verdict serializes")
        }
        "bit_equal" => {
            arguments(args, &["left", "right"], &[])?;
            serde_json::json!(bit_equal(&args["left"], &args["right"]))
        }
        "first_difference" => {
            arguments(args, &["left", "right"], &["positions", "answer"])?;
            let side = |key: &str| {
                args[key]
                    .as_object()
                    .ok_or_else(|| RuleError(format!("{key}: expected an answer object")))
            };
            let positions = fields_of(
                args.get("positions").unwrap_or(&Value::Null),
                &POSITION_FIELDS,
            )?;
            if positions.is_empty() {
                // Python's rule found no length to compare over and refused.
                return Err(RuleError("positions: expected at least one field".into()));
            }
            let answer = fields_of(args.get("answer").unwrap_or(&Value::Null), &ANSWER_FIELDS)?;
            first_difference(side("left")?, side("right")?, &positions, &answer)?
                .map_or(Value::Null, |difference| difference.to_json())
        }
        other => return Err(RuleError(format!("unknown limit rule {other:?}"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The integer quotient against IEEE division where both operands are
    /// doubles, which rounds once, over counts around powers of two; then
    /// Python's quotients past 2^53.
    #[test]
    fn integer_ratios_round_once() {
        let mut counts = vec![1u64, 2, 3, 5, 7, 10, 999, 1000, 1001, 3 << 40];
        for power in [20u32, 31, 52, 53] {
            for offset in 0..4 {
                counts.push((1u64 << power) - offset);
                counts.push((1u64 << power) + offset);
            }
        }
        counts.retain(|count| *count <= 1 << 53);
        for &numerator in &counts {
            for &denominator in &counts {
                if numerator == 0 {
                    continue;
                }
                let fast = numerator as f64 / denominator as f64;
                assert_eq!(
                    rounded_ratio(numerator, denominator).to_bits(),
                    fast.to_bits(),
                    "{numerator}/{denominator}"
                );
                assert_eq!(ratio(numerator, denominator).to_bits(), fast.to_bits());
            }
        }
        // Python's 1 / (2^53 + 1), 3 / (2^64 - 1) and (2^64 - 1) / 3.
        assert_eq!(ratio(1, (1 << 53) + 1).to_bits(), 0x3c9f_ffff_ffff_ffff);
        assert_eq!(ratio(3, u64::MAX), 3.0 / 18_446_744_073_709_551_615.0);
        assert_eq!(ratio(u64::MAX, 3), 6_148_914_691_236_517_205.0);
        assert_eq!(ratio(0, u64::MAX), 0.0);
    }

    /// variant-spread.v2's power test at the measured small categories: a
    /// 141-position total limit of 0.10 against a BF16 run with 14 flips
    /// cannot judge, one against 8 flips can, and a reference without flips
    /// always can.
    #[test]
    fn power_needs_the_limit_two_standard_errors_above_the_reference_rate() {
        let fourteen = power(14.0 / 141.0, 0.10, 141, CATEGORY_SIGMAS);
        assert!(!fourteen.has_power);
        assert!(
            (fourteen.standard_error - (14.0 / 141.0 * (127.0 / 141.0) / 141.0_f64).sqrt()).abs()
                < 1e-15
        );
        assert!(power(8.0 / 141.0, 0.10, 141, CATEGORY_SIGMAS).has_power);
        assert!(power(0.0, 0.08, 141, CATEGORY_SIGMAS).has_power);
        // 0.25 over 75 positions: a standard error of 0.05.
        assert!(power(0.25, 0.375, 75, CATEGORY_SIGMAS).has_power);
        assert!(!power(0.25, 0.34375, 75, CATEGORY_SIGMAS).has_power);
    }

    /// The paired top-1 clause on its own matches the paired rule's, and
    /// counts discordant positions only.
    #[test]
    fn paired_top1_is_the_paired_rules_top1_clause() {
        // 16 candidate flips against 14 of the reference run's, 9 shared.
        let test: Vec<bool> = (0..141).map(|i| i < 9 || (14..21).contains(&i)).collect();
        let baseline: Vec<bool> = (0..141).map(|i| i < 14).collect();
        let clause = paired_top1(&test, &baseline, CATEGORY_SIGMAS).expect("same positions");
        assert_eq!((clause.test_only, clause.baseline_only), (7, 5));
        assert_eq!(clause.margin, 2.0 * 12.0_f64.sqrt());
        assert!(clause.passed);
        let arms = |flags: &[bool]| flags.iter().map(|flag| (0.0, *flag)).collect::<Vec<_>>();
        let full =
            paired(&arms(&test), &arms(&baseline), 0.0, CATEGORY_SIGMAS).expect("same positions");
        assert_eq!(
            (
                full.top1_test_only,
                full.top1_baseline_only,
                full.top1_margin,
                full.passed
            ),
            (
                clause.test_only,
                clause.baseline_only,
                clause.margin,
                clause.passed
            )
        );
        // 28 against 14, 10 shared: 18 against 4 exceeds the margin.
        let doubled: Vec<bool> = (0..141).map(|i| i < 10 || (14..32).contains(&i)).collect();
        assert!(
            !paired_top1(&doubled, &baseline, CATEGORY_SIGMAS)
                .expect("same positions")
                .passed
        );
        assert!(paired_top1(&test[..140], &baseline, CATEGORY_SIGMAS).is_err());
    }
}
