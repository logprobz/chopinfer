//! Allocate a block's verify prefixes after drafting, before target execution.
//!
//! Confidence estimates affect work only. The ordinary target sampler and
//! acceptance walk remain the authority for emitted tokens and RNG progress.

use super::MAX_DRAFT_TOKENS;

/// One sequence, in the callback's stable item order.
#[derive(Clone, Copy, Debug)]
pub struct ConfidenceCandidate<'a> {
    /// Conditional probabilities of matching the target at each position,
    /// given that all preceding positions matched. The model adapter owns
    /// calibration; raw head scores must not be passed as probabilities.
    pub probabilities: &'a [f64],
    /// The scheduler's reserved draft count, already bounded by remaining
    /// generation tokens, context and KV capacity. The adapter must support
    /// every prefix from zero through this count.
    pub maximum_draft_tokens: usize,
}

/// Measured target cost for a graph capacity, at this concurrency.
///
/// A live row count uses the first capacity that holds it. For exact-width
/// graphs supply every attainable width, not power-of-two measurements.
#[derive(Clone, Copy, Debug)]
pub struct VerifyBucketCost {
    pub rows: usize,
    pub micros: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ConfidenceBudget {
    /// Includes the mandatory target row of every sequence, even K=0.
    pub maximum_rows: usize,
    /// Measured work shared by every possible prefix, such as the drafter.
    /// Include it once when comparing expected tokens per cycle time.
    pub fixed_micros: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConfidencePlan {
    pub drafts: Vec<usize>,
    pub rows: usize,
    pub bucket_rows: usize,
    pub expected_tokens: f64,
    pub estimated_micros: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfidenceError {
    InvalidBudget,
    InvalidCandidate,
    InvalidProbability,
    InvalidCosts,
    MissingCost,
}

/// Maximize expected emitted tokens per measured cycle time under one row
/// budget. Probabilities multiply along each sequence's prefix. Because
/// prefix survival never increases, taking the largest available survival
/// gives the best allocation at each total width. Compare *every* width:
/// crossing a graph boundary can lose before filling that graph pays.
///
/// Equal marginal gains prefer the earlier item; equal throughput prefers
/// fewer rows. This function consumes no RNG and changes no durable state.
/// It neither extrapolates an unmeasured graph cost nor interprets raw scores.
pub fn allocate(
    candidates: &[ConfidenceCandidate<'_>],
    budget: ConfidenceBudget,
    costs: &[VerifyBucketCost],
) -> Result<ConfidencePlan, ConfidenceError> {
    if candidates.is_empty()
        || candidates.len() > 256
        || budget.maximum_rows < candidates.len()
        || budget.maximum_rows > super::MAX_COST_ROWS
        || !budget.fixed_micros.is_finite()
        || budget.fixed_micros < 0.0
    {
        return Err(ConfidenceError::InvalidBudget);
    }
    let mut available = candidates.len();
    for candidate in candidates {
        if candidate.maximum_draft_tokens > candidate.probabilities.len()
            || candidate.probabilities.len() > MAX_DRAFT_TOKENS
        {
            return Err(ConfidenceError::InvalidCandidate);
        }
        if candidate
            .probabilities
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        {
            return Err(ConfidenceError::InvalidProbability);
        }
        available += candidate.maximum_draft_tokens;
    }
    let maximum = available.min(budget.maximum_rows);
    if costs.is_empty()
        || costs.len() > super::MAX_COST_ROWS
        || costs.iter().any(|cost| {
            cost.rows == 0
                || cost.rows > super::MAX_COST_ROWS
                || !cost.micros.is_finite()
                || cost.micros <= 0.0
                || !(cost.micros + budget.fixed_micros).is_finite()
                || !(maximum as f64 / (cost.micros + budget.fixed_micros)).is_finite()
        })
        || costs.windows(2).any(|pair| pair[0].rows >= pair[1].rows)
    {
        return Err(ConfidenceError::InvalidCosts);
    }
    if costs.last().expect("nonempty costs").rows < maximum {
        return Err(ConfidenceError::MissingCost);
    }
    let mut drafts = vec![0; candidates.len()];
    let mut survival = vec![1.0; candidates.len()];
    let mut expected = candidates.len() as f64;
    let mut best: Option<ConfidencePlan> = None;
    let mut bucket = 0;
    for rows in candidates.len()..=maximum {
        while costs[bucket].rows < rows {
            bucket += 1;
        }
        let micros = budget.fixed_micros + costs[bucket].micros;
        if best
            .as_ref()
            .is_none_or(|best| expected / micros > best.expected_tokens / best.estimated_micros)
        {
            best = Some(ConfidencePlan {
                drafts: drafts.clone(),
                rows,
                bucket_rows: costs[bucket].rows,
                expected_tokens: expected,
                estimated_micros: micros,
            });
        }
        if rows == maximum {
            break;
        }
        let mut next = None;
        for (item, candidate) in candidates.iter().enumerate() {
            if drafts[item] == candidate.maximum_draft_tokens {
                continue;
            }
            let gain = survival[item] * candidate.probabilities[drafts[item]];
            if next.is_none_or(|(_, best_gain)| gain > best_gain) {
                next = Some((item, gain));
            }
        }
        let (item, gain) = next.expect("maximum is bounded by available drafts");
        drafts[item] += 1;
        survival[item] = gain;
        expected += gain;
    }
    Ok(best.expect("budget holds one row per candidate"))
}

#[cfg(test)]
mod tests;
