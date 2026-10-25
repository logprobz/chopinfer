//! Bounded traces that resume bundle drafting after committed decode steps.
use super::trace::{Step, Trace};

pub const VERSION: u32 = 6;
const STEPS: usize = 1024;
const REQUESTS: usize = 32;
const ROWS: usize = 256;
const PREFILL: usize = 8192;

pub fn validate(trace: &Trace) -> Result<(), String> {
    if !(1..=REQUESTS).contains(&trace.concurrency)
        || !(1024..=1 << 20).contains(&trace.vocabulary)
        || !(2..=STEPS).contains(&trace.steps.len())
    {
        return Err("unsupported transition trace geometry".into());
    }
    let mut lengths = vec![0; trace.concurrency];
    let mut prompts = vec![None; trace.concurrency];
    let mut decoding = false;
    let mut drafted = false;
    for step in &trace.steps {
        let prefill = step.items.first().ok_or("empty transition step")?.prefill;
        if step.items.len() > REQUESTS || (prefill && decoding) {
            return Err("transition trace must finish prefill before decoding".into());
        }
        decoding |= !prefill;
        let mut seen = std::collections::BTreeSet::new();
        for item in &step.items {
            if item.slot >= trace.concurrency
                || !seen.insert(item.slot)
                || item.prefill != prefill
                || item.anchor
                || item.drafts > 7
                || !item.fixed_drafts.is_empty()
                || !item.history.is_empty()
                || item.inputs.is_empty()
                || item.inputs.len() > PREFILL
                || prompts[item.slot].is_some() == prefill
                || (prefill && (item.drafts != 0 || item.kept != item.inputs.len()))
                || (!prefill
                    && (item.inputs.len() != 1 || !(1..=item.drafts + 1).contains(&item.kept)))
                || item
                    .inputs
                    .iter()
                    .any(|&token| token as usize >= trace.vocabulary)
            {
                return Err("invalid transition trace item".into());
            }
            if prefill {
                prompts[item.slot] = Some(item.inputs.len());
            }
            let history_length = lengths[item.slot] + item.inputs.len();
            if history_length.checked_sub(item.emitted) != prompts[item.slot] {
                return Err("transition prompt/generated split drifted".into());
            }
            lengths[item.slot] += item.kept;
            drafted |= item.drafts > 0;
        }
        if step.rows() > ROWS
            || (prefill
                && step
                    .items
                    .iter()
                    .map(|item| item.inputs.len())
                    .sum::<usize>()
                    > PREFILL)
        {
            return Err("transition step exceeds the row or prefill limit".into());
        }
    }
    if !decoding || !drafted || prompts.contains(&None) {
        return Err("transition trace must prefill every request and exercise drafting".into());
    }
    Ok(())
}

/// Only successful commits advance this history. Proposed tokens are checked
/// in the exact frame before they can become inputs to a later step.
#[derive(Clone)]
pub struct History {
    vocabulary: usize,
    committed: Vec<Vec<u32>>,
}

impl History {
    pub fn tokens(&self, slot: usize) -> Result<&[u32], String> {
        self.committed
            .get(slot)
            .map(Vec::as_slice)
            .ok_or_else(|| "unknown history slot".into())
    }

    pub fn replace(&mut self, slot: usize, tokens: Vec<u32>) -> Result<(), String> {
        if tokens
            .iter()
            .any(|&token| token as usize >= self.vocabulary)
        {
            return Err("invalid restored history token".into());
        }
        *self.committed.get_mut(slot).ok_or("unknown history slot")? = tokens;
        Ok(())
    }

    pub fn new(trace: &Trace) -> Self {
        Self {
            vocabulary: trace.vocabulary,
            committed: vec![Vec::new(); trace.concurrency],
        }
    }

    pub fn materialize(&self, step: &Step) -> Result<Step, String> {
        let mut step = step.clone();
        for item in &mut step.items {
            let committed = self.committed.get(item.slot).ok_or("unknown trace slot")?;
            if !item.history.is_empty() || item.prefill != committed.is_empty() {
                return Err("transition history does not match committed state".into());
            }
            item.history = committed.clone();
            item.history.extend_from_slice(&item.inputs);
        }
        Ok(step)
    }

    /// Prepare all history updates before committing device state. The caller
    /// installs this result only after the bundle and KV commits succeed.
    pub fn prepare_commit(
        &self,
        step: &Step,
        proposals: &[(u64, Vec<u32>)],
    ) -> Result<Self, String> {
        let mut by_request = std::collections::BTreeMap::new();
        for (request, tokens) in proposals {
            if by_request.insert(*request, tokens).is_some()
                || tokens
                    .iter()
                    .any(|&token| token as usize >= self.vocabulary)
            {
                return Err("duplicate or invalid transition proposal".into());
            }
        }
        let mut next = self.clone();
        for item in &step.items {
            let committed = self.committed.get(item.slot).ok_or("unknown trace slot")?;
            let mut expected = committed.clone();
            expected.extend_from_slice(&item.inputs);
            if item.history != expected {
                return Err("materialized transition history changed before commit".into());
            }
            if item.drafts > 0 {
                let tokens = by_request
                    .remove(&(item.slot as u64 + 1))
                    .ok_or("transition proposal is absent")?;
                if tokens.len() != item.drafts || !(1..=item.drafts + 1).contains(&item.kept) {
                    return Err("transition proposal or kept count differs".into());
                }
                expected.extend_from_slice(&tokens[..item.kept - 1]);
            }
            next.committed[item.slot] = expected;
        }
        if !by_request.is_empty() {
            return Err("unexpected transition proposal".into());
        }
        Ok(next)
    }
}
