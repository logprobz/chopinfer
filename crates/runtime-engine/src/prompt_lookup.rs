//! Prompt-lookup drafting: drafts copied from earlier in the request.
//!
//! When the last few tokens of a request's history occurred before, the
//! tokens that followed that earlier occurrence are proposed as drafts. It
//! needs no model, so any executor can verify its drafts, and it does well on
//! text that repeats its context (code edits, extraction, quoting, lists).

use crate::{ExecutorError, ModelBatchItem, ModelExecutor, ModelTransaction, RequestId};
use runtime_kv::PreparedAppendBatch;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PromptLookupConfigError {
    ZeroNgram,
    NgramRangeInverted,
    WindowBelowNgram,
}

/// Match lengths and the search window of prompt-lookup drafting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PromptLookupConfig {
    minimum_ngram: usize,
    maximum_ngram: usize,
    search_window: usize,
}

impl PromptLookupConfig {
    /// Suffixes of `minimum_ngram` to `maximum_ngram` tokens are looked up in
    /// the last `search_window` tokens of history, longest suffix first.
    pub const fn new(
        minimum_ngram: usize,
        maximum_ngram: usize,
        search_window: usize,
    ) -> Result<Self, PromptLookupConfigError> {
        if minimum_ngram == 0 {
            return Err(PromptLookupConfigError::ZeroNgram);
        }
        if maximum_ngram < minimum_ngram {
            return Err(PromptLookupConfigError::NgramRangeInverted);
        }
        if search_window <= maximum_ngram {
            return Err(PromptLookupConfigError::WindowBelowNgram);
        }
        Ok(Self {
            minimum_ngram,
            maximum_ngram,
            search_window,
        })
    }

    /// Proposes up to `maximum` drafts to follow `history`.
    ///
    /// For each suffix length from the longest down, the most recent earlier
    /// occurrence of the suffix is found, and the tokens after it are
    /// proposed. The continuation may run into the suffix itself, which is
    /// what lets a repeating cycle continue. No match proposes nothing.
    #[must_use]
    pub fn propose(self, history: &[u32], maximum: usize) -> Vec<u32> {
        if maximum == 0 {
            return Vec::new();
        }
        let window = &history[history.len().saturating_sub(self.search_window)..];
        for length in (self.minimum_ngram..=self.maximum_ngram).rev() {
            if window.len() <= length {
                continue;
            }
            let suffix = &window[window.len() - length..];
            // Candidate starts end before the suffix's own start.
            let found = (0..window.len() - length)
                .rev()
                .find(|start| &window[*start..*start + length] == suffix);
            if let Some(start) = found {
                let from = start + length;
                let to = window.len().min(from + maximum);
                return window[from..to].to_vec();
            }
        }
        Vec::new()
    }
}

impl Default for PromptLookupConfig {
    fn default() -> Self {
        Self {
            minimum_ngram: 2,
            maximum_ngram: 4,
            search_window: 4096,
        }
    }
}

/// An executor that drafts by prompt lookup and delegates everything else.
///
/// Use it with `DraftSource::Host`: the engine asks it for drafts before each
/// step and the wrapped executor verifies them as ordinary input tokens.
#[derive(Clone, Debug)]
pub struct PromptLookupExecutor<E> {
    inner: E,
    config: PromptLookupConfig,
}

impl<E> PromptLookupExecutor<E> {
    #[must_use]
    pub const fn new(inner: E, config: PromptLookupConfig) -> Self {
        Self { inner, config }
    }

    #[must_use]
    pub const fn inner(&self) -> &E {
        &self.inner
    }

    #[must_use]
    pub const fn inner_mut(&mut self) -> &mut E {
        &mut self.inner
    }
}

impl<E: ModelExecutor> ModelExecutor for PromptLookupExecutor<E> {
    fn round_draft_tokens_v1(&self, proposed: usize) -> usize {
        self.inner.round_draft_tokens_v1(proposed)
    }

    fn step_rows_v1(&self, rows: usize) -> usize {
        self.inner.step_rows_v1(rows)
    }

    fn rollback_preserves_state_v1(&self) -> bool {
        self.inner.rollback_preserves_state_v1()
    }

    fn kv_offload_info_v1(&self) -> Option<crate::KvOffloadInfo> {
        self.inner.kv_offload_info_v1()
    }
    fn offload_pages_v1(
        &mut self,
        plan: runtime_kv::KvTransferPlan,
    ) -> crate::KvTransferSubmitResult {
        self.inner.offload_pages_v1(plan)
    }
    fn reload_pages_v1(
        &mut self,
        plan: runtime_kv::KvTransferPlan,
    ) -> crate::KvTransferSubmitResult {
        self.inner.reload_pages_v1(plan)
    }
    fn poll_kv_transfer_v1(
        &mut self,
        id: runtime_kv::KvTransferId,
    ) -> Result<crate::KvTransferPoll, ExecutorError> {
        self.inner.poll_kv_transfer_v1(id)
    }
    fn wait_kv_transfer_v1(
        &mut self,
        id: runtime_kv::KvTransferId,
    ) -> Result<crate::KvTransferPoll, ExecutorError> {
        self.inner.wait_kv_transfer_v1(id)
    }
    fn kv_transfer_drain_timeout_v1(&self) -> std::time::Duration {
        self.inner.kv_transfer_drain_timeout_v1()
    }
    fn drain_kv_transfers_v1(&mut self) -> Result<(), ExecutorError> {
        self.inner.drain_kv_transfers_v1()
    }

    fn admit_request_v1(&mut self, id: RequestId) -> Result<(), ExecutorError> {
        self.inner.admit_request_v1(id)
    }

    fn capture_prefix_state_v1(
        &mut self,
        id: RequestId,
        tokens: usize,
        pages: &[runtime_kv::BlockId],
    ) -> Result<(), ExecutorError> {
        self.inner.capture_prefix_state_v1(id, tokens, pages)
    }

    fn restore_prefix_state_v1(
        &mut self,
        id: RequestId,
        prefix: &[u32],
        pages: &[runtime_kv::BlockId],
    ) -> Result<(), ExecutorError> {
        self.inner.restore_prefix_state_v1(id, prefix, pages)
    }

    type TransactionState<'a>
        = E::TransactionState<'a>
    where
        Self: 'a;

    fn retire_request_v1(&mut self, id: RequestId) -> Result<(), ExecutorError> {
        self.inner.retire_request_v1(id)
    }

    fn draft_tokens_v1(
        &mut self,
        _id: RequestId,
        history: &[u32],
        maximum: usize,
    ) -> Result<Vec<u32>, ExecutorError> {
        Ok(self.config.propose(history, maximum))
    }

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        kv_plan: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        self.inner.execute(items, kv_plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup(minimum: usize, maximum: usize, window: usize) -> PromptLookupConfig {
        PromptLookupConfig::new(minimum, maximum, window).unwrap()
    }

    #[test]
    fn the_most_recent_longest_match_supplies_the_drafts() {
        let config = lookup(1, 3, 64);
        // "7 8" occurs twice before the suffix; the later one is followed by 3.
        let history = [7, 8, 1, 2, 7, 8, 3, 4, 5, 7, 8];
        assert_eq!(config.propose(&history, 2), [3, 4]);
        // A longer suffix wins over a more recent short one.
        let history = [1, 2, 3, 9, 9, 3, 5, 1, 2, 3];
        assert_eq!(config.propose(&history, 3), [9, 9, 3]);
    }

    #[test]
    fn a_cycle_continues_into_the_suffix() {
        let config = lookup(2, 2, 64);
        let history = [1, 2, 3, 1, 2, 3, 1, 2];
        assert_eq!(config.propose(&history, 5), [3, 1, 2]);
    }

    #[test]
    fn no_match_or_no_room_proposes_nothing() {
        let config = lookup(2, 3, 64);
        assert!(config.propose(&[1, 2, 3, 4], 4).is_empty());
        assert!(config.propose(&[1, 2, 1, 2], 0).is_empty());
        assert!(config.propose(&[1], 4).is_empty());
    }

    #[test]
    fn matches_outside_the_window_are_ignored() {
        let history = [5, 6, 9, 0, 0, 0, 0, 0, 0, 5, 6];
        assert_eq!(lookup(2, 2, 64).propose(&history, 1), [9]);
        assert!(lookup(2, 2, 8).propose(&history, 1).is_empty());
    }

    #[test]
    fn configuration_limits_are_checked() {
        assert_eq!(
            PromptLookupConfig::new(0, 2, 8),
            Err(PromptLookupConfigError::ZeroNgram)
        );
        assert_eq!(
            PromptLookupConfig::new(3, 2, 8),
            Err(PromptLookupConfigError::NgramRangeInverted)
        );
        assert_eq!(
            PromptLookupConfig::new(2, 4, 4),
            Err(PromptLookupConfigError::WindowBelowNgram)
        );
    }
}
