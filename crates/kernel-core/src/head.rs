//! Streaming vocabulary-head scoring: log-sum-exp, top-k and the top-1
//! margin of one row's logits, presented chunk by chunk in vocabulary order
//! so a large head never has to be resident.

use crate::contract::Refusal;
use crate::select::Ties;

/// A scored row. Log-probabilities are `logit - log_z`.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadScore {
    pub log_z: f64,
    /// The target token's log-probability, when a target was given.
    pub target: Option<f64>,
    /// The `k` best (token, log-probability), best first.
    pub top: Vec<(usize, f64)>,
    /// Top-1 minus top-2 log-probability, when two are kept.
    pub margin: Option<f64>,
}

/// Online log-sum-exp: each chunk rescales the running sum to the new
/// maximum, `sum * exp(old_max - max)`, then adds `exp(l - max)` in
/// vocabulary order. Top-k ranks logits by `ties` as `select::topk` does, so
/// one chunk selects exactly what `topk` selects.
/// The result depends on the chunk boundaries only through the rescaling.
#[derive(Clone, Debug)]
pub struct StreamingHead {
    k: usize,
    ties: Ties,
    target: Option<usize>,
    next: usize,
    max: f64,
    sum: f64,
    target_logit: Option<f64>,
    top: Vec<(usize, f64)>,
}

impl StreamingHead {
    pub fn new(k: usize, ties: Ties, target: Option<usize>) -> Self {
        Self {
            k,
            ties,
            target,
            next: 0,
            max: f64::NEG_INFINITY,
            sum: 0.0,
            target_logit: None,
            top: Vec::with_capacity(k + 1),
        }
    }

    fn better(&self, a: (usize, f64), b: (usize, f64)) -> bool {
        self.ties.order(a, b).is_lt()
    }

    /// The logits of the next `logits.len()` vocabulary entries. Non-finite
    /// logits are refused; an empty chunk changes nothing.
    pub fn push(&mut self, logits: &[f64]) -> Result<(), Refusal> {
        if logits.iter().any(|l| !l.is_finite()) {
            return Err(Refusal("head logits must be finite"));
        }
        if logits.is_empty() {
            return Ok(());
        }
        let chunk_max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let max = self.max.max(chunk_max);
        let mut sum = self.sum * (self.max - max).exp();
        for &l in logits {
            sum += (l - max).exp();
        }
        self.max = max;
        self.sum = sum;
        let start = self.next;
        if let Some(target) = self.target
            && (start..start + logits.len()).contains(&target)
        {
            self.target_logit = Some(logits[target - start]);
        }
        for (offset, &l) in logits.iter().enumerate() {
            let candidate = (start + offset, l);
            if self.top.len() < self.k
                || self
                    .top
                    .last()
                    .is_some_and(|&last| self.better(candidate, last))
            {
                let at = self.top.partition_point(|&t| self.better(t, candidate));
                self.top.insert(at, candidate);
                self.top.truncate(self.k);
            }
        }
        self.next += logits.len();
        Ok(())
    }

    /// Refuses a head that saw no logits or never reached its target.
    pub fn finish(self) -> Result<HeadScore, Refusal> {
        if self.next == 0 || self.target.is_some() && self.target_logit.is_none() {
            return Err(Refusal("head saw no logits or not its target"));
        }
        let log_z = self.max + self.sum.ln();
        let top = self
            .top
            .iter()
            .map(|&(id, l)| (id, l - log_z))
            .collect::<Vec<_>>();
        Ok(HeadScore {
            log_z,
            target: self.target_logit.map(|l| l - log_z),
            margin: (top.len() >= 2).then(|| top[0].1 - top[1].1),
            top,
        })
    }
}
