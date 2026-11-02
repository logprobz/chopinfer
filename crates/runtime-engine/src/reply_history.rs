//! Digest-only observations. This module never supplies a cache key to the
//! scheduler, owns no KV, and cannot change a capture or restore decision.
use runtime_kv::{
    KvMetrics, PrefixCacheKeyV1 as Key, PrefixChainV1, PrefixDigestTailV1,
    ReplyHistoryCounter as Count, ReplyHistoryRows as Rows,
};
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};

use crate::{FinishReason, RequestId};

pub const MAX_BYTES: usize = 64 * 1024 * 1024;
const MAX_COMPLETED: usize = 256;
const MAX_ACTIVE: usize = 256;
const MAX_DELIVERIES: usize = 256;

/// Shared observation state, independent of engine and frontend lifetimes.
#[derive(Clone, Debug)]
pub struct ReplyHistoryObserver(pub(crate) Arc<Mutex<Registry>>);
impl ReplyHistoryObserver {
    /// A stack-owned receipt. Cloning the shared registry allocates no storage.
    #[must_use]
    pub fn delivery(&self, id: RequestId) -> Option<ReplyHistoryDelivery> {
        let mut registry = self.0.lock().ok()?;
        if registry.delivery_handles == MAX_DELIVERIES {
            return None;
        }
        registry.delivery_handles += 1;
        Some(ReplyHistoryDelivery {
            observer: self.clone(),
            id,
            succeeded: false,
        })
    }
}

/// Dropping a relay before its terminal handoff discards its observation.
#[derive(Debug)]
pub struct ReplyHistoryDelivery {
    observer: ReplyHistoryObserver,
    id: RequestId,
    succeeded: bool,
}
impl ReplyHistoryDelivery {
    pub fn succeeded(mut self) {
        self.succeeded = true;
        if let Ok(mut observer) = self.observer.0.lock() {
            observer.handoff(self.id);
        }
    }
}
impl Drop for ReplyHistoryDelivery {
    fn drop(&mut self) {
        if let Ok(mut observer) = self.observer.0.lock() {
            observer.delivery_handles -= 1;
            if !self.succeeded {
                observer.discard(self.id, Count::DeliveryFailed);
            }
        }
    }
}

/// The prior token's observed contribution, never an explanation of a mismatch.
#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum ReplyRegion {
    Reasoning = 1,
    Content = 2,
    ToolCalls = 4,
    EndToken = 8,
    IgnoredEndToken = 16,
    /// A token with hidden bytes or a partial stop remains incompletely labelled.
    Partial = 32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Interval {
    start: usize,
    end: usize,
}
impl Interval {
    fn intersect(self, start: usize, end: usize) -> Self {
        Self {
            start: self.start.max(start),
            end: self.end.min(end),
        }
    }
    fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }
}

/// Stack-sized union with no allocator. A candidate contributes at most once.
#[derive(Debug)]
struct Intervals {
    ranges: [Interval; MAX_COMPLETED],
    len: usize,
}
impl Default for Intervals {
    fn default() -> Self {
        Self {
            ranges: [Interval::default(); MAX_COMPLETED],
            len: 0,
        }
    }
}
impl Intervals {
    fn push(&mut self, start: usize, end: usize) {
        if start < end {
            self.ranges[self.len] = Interval { start, end };
            self.len += 1;
        }
    }
    fn merge(&mut self) {
        self.ranges[..self.len].sort_unstable_by_key(|r| r.start);
        let mut used = 0;
        for index in 0..self.len {
            let range = self.ranges[index];
            if used > 0 && self.ranges[used - 1].end >= range.start {
                self.ranges[used - 1].end = self.ranges[used - 1].end.max(range.end);
            } else {
                self.ranges[used] = range;
                used += 1;
            }
        }
        self.len = used;
    }
    fn count(&self, metrics: &KvMetrics, kind: Rows, start: usize, end: usize) -> usize {
        let mut total = 0;
        for range in &self.ranges[..self.len] {
            let range = range.intersect(start, end);
            if range.len() > 0 {
                metrics.reply_history_rows(kind, range.start, range.end);
                total += range.len();
            }
        }
        total
    }
}

#[derive(Debug, Default)]
struct Comparison {
    pages: Intervals,
    matching: Intervals,
    candidates: Intervals,
    anchor: Interval,
    ambiguous: bool,
    unobserved: bool,
    page_counted: bool,
    anchor_counted: bool,
}

#[derive(Clone, Debug)]
struct History {
    id: RequestId,
    prompt: usize,
    block: usize,
    prompt_key: Key,
    written_key: Key,
    pages: Vec<Key>,
    fingerprints: Vec<Key>,
    regions: Vec<u8>,
    /// Set only on successful final delivery, or engine-only completion.
    order: Option<u64>,
    deferred: bool,
}
impl History {
    fn end(&self) -> usize {
        self.prompt + self.fingerprints.len()
    }
    fn bytes(&self) -> usize {
        self.pages.capacity() * size_of::<Key>()
            + self.fingerprints.capacity() * size_of::<Key>()
            + self.regions.capacity()
    }
}

#[derive(Debug)]
struct Active {
    history: History,
    tail: PrefixDigestTailV1,
    written_keys: Vec<Key>,
    comparison: Comparison,
    engine_finished: bool,
    engine_reason: Option<FinishReason>,
    frontend_cut: bool,
    handoff_required: bool,
    handoff_succeeded: bool,
    frontend_reason: Option<FinishReason>,
    tool_finish: bool,
}
impl Active {
    fn bytes(&self) -> usize {
        self.history.bytes()
            + self.tail.allocated_bytes()
            + self.written_keys.capacity() * size_of::<Key>()
    }
}

#[derive(Debug)]
pub(crate) struct Registry {
    active: Vec<Active>,
    completed: Vec<History>,
    order: u64,
    lost_history: bool,
    temporary_bytes: usize,
    delivery_handles: usize,
    metrics: KvMetrics,
}

fn reserved<T>(capacity: usize) -> Option<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(capacity).ok()?;
    Some(values)
}

fn fingerprint(prompt: Key, ordinal: usize, token: u32) -> Key {
    let mut hash = Sha256::new();
    hash.update(b"chopinfer.reply-observation.v1");
    hash.update(prompt.digest());
    hash.update((ordinal as u64).to_le_bytes());
    hash.update(token.to_le_bytes());
    Key::new(hash.finalize().into())
}

impl Registry {
    pub(crate) fn new(metrics: KvMetrics) -> Option<Self> {
        let result = Self {
            active: reserved(MAX_ACTIVE)?,
            completed: reserved(MAX_COMPLETED)?,
            order: 0,
            lost_history: false,
            temporary_bytes: 0,
            delivery_handles: 0,
            metrics,
        };
        result.publish();
        Some(result)
    }
    fn bytes(&self) -> usize {
        size_of::<Self>() + self.temporary_bytes
            + self.active.capacity() * size_of::<Active>()
            + self.completed.capacity() * size_of::<History>()
            + self.active.iter().map(Active::bytes).sum::<usize>()
            + self.completed.iter().map(History::bytes).sum::<usize>()
            // Candidate temporaries, Arc/Mutex bookkeeping, and up to 256
            // stack-owned delivery receipts (including future frame padding).
            + 64 * 1024
    }
    fn publish(&self) {
        self.metrics
            .reply_history_gauges(true, self.bytes(), self.completed.len());
    }
    fn evict(&mut self) -> bool {
        if self.completed.is_empty() {
            return false;
        }
        // Completed storage is in completion order, including deferred delivery.
        self.completed.remove(0);
        self.lost_history = true;
        self.metrics.reply_history_count(Count::Evicted, 1);
        true
    }
    pub(crate) fn reserve_temporary(&mut self, bytes: usize) -> bool {
        while bytes <= MAX_BYTES && self.bytes().saturating_add(bytes) > MAX_BYTES && self.evict() {
        }
        if self.bytes().saturating_add(bytes) > MAX_BYTES {
            self.metrics.reply_history_count(Count::Submitted, 1);
            self.metrics.reply_history_count(Count::CapacityDrop, 1);
            self.lost_history = true;
            self.publish();
            return false;
        }
        self.temporary_bytes = bytes;
        self.publish();
        true
    }
    pub(crate) fn release_temporary(&mut self) {
        self.temporary_bytes = 0;
        self.publish();
    }
    pub(crate) fn unsupported(&self) {
        self.metrics.reply_history_count(Count::Submitted, 1);
        self.metrics.reply_history_count(Count::Unsupported, 1);
    }
    pub(crate) fn submit(&mut self, id: RequestId, chain: &PrefixChainV1, maximum_reply: usize) {
        self.metrics.reply_history_count(Count::Submitted, 1);
        let prompt = chain.tokens().len();
        let Some(prompt_key) = chain.key_at(prompt) else {
            return;
        };
        let page_capacity = (prompt + maximum_reply) / chain.block_tokens() + 1;
        let required = page_capacity
            .checked_mul(size_of::<Key>())
            .and_then(|n| n.checked_add(maximum_reply.checked_mul(2 * size_of::<Key>() + 1)?))
            .and_then(|n| n.checked_add(chain.continuation_bytes()));
        let Some(required) = required.filter(|&n| n <= MAX_BYTES) else {
            self.metrics.reply_history_count(Count::CapacityDrop, 1);
            self.lost_history = true;
            return;
        };
        while self.bytes() + required > MAX_BYTES && self.evict() {}
        if self.active.len() == MAX_ACTIVE || self.bytes() + required > MAX_BYTES {
            self.metrics.reply_history_count(Count::CapacityDrop, 1);
            self.lost_history = true;
            self.publish();
            return;
        }
        let allocation = || {
            Some((
                reserved(page_capacity)?,
                reserved(maximum_reply)?,
                reserved(maximum_reply)?,
                chain.continuation().ok()?,
                reserved(maximum_reply)?,
            ))
        };
        let Some((mut pages, fingerprints, regions, tail, written_keys)) = allocation() else {
            self.metrics.reply_history_count(Count::AllocationDrop, 1);
            self.lost_history = true;
            self.publish();
            return;
        };
        pages.extend_from_slice(chain.digests());
        let history = History {
            id,
            prompt,
            block: chain.block_tokens(),
            prompt_key,
            written_key: prompt_key,
            pages,
            fingerprints,
            regions,
            order: None,
            deferred: false,
        };
        // Account actual capacities, including allocators that round reservations.
        if self.bytes()
            + history.bytes()
            + tail.allocated_bytes()
            + written_keys.capacity() * size_of::<Key>()
            > MAX_BYTES
        {
            self.metrics.reply_history_count(Count::AllocationDrop, 1);
            self.lost_history = true;
            return;
        }
        let comparison = self.compare(chain);
        self.active.push(Active {
            history,
            tail,
            written_keys,
            comparison,
            engine_finished: false,
            engine_reason: None,
            frontend_cut: false,
            handoff_required: false,
            handoff_succeeded: false,
            frontend_reason: None,
            tool_finish: false,
        });
        self.publish();
    }

    fn common_pages(&self, previous: &History, chain: &PrefixChainV1) -> usize {
        let mut low = 0;
        let mut high = (previous.pages.len() - 1).min(chain.full_blocks()) + 1;
        while low + 1 < high {
            let middle = (low + high) / 2;
            self.metrics.reply_history_count(Count::Probes, 1);
            if Some(previous.pages[middle]) == chain.chain(middle) {
                low = middle;
            } else {
                high = middle;
            }
        }
        low * chain.block_tokens()
    }

    fn compare(&self, chain: &PrefixChainV1) -> Comparison {
        let mut result = Comparison::default();
        let mut best: Option<(&History, usize)> = None;
        let mut scope = false;
        let mut ancestry = 0;
        let limit = chain.tokens().len().saturating_sub(1);
        for (index, previous) in self.completed.iter().enumerate() {
            if previous.pages.first().copied() != chain.chain(0) {
                continue;
            }
            scope = true;
            let mut common = self.common_pages(previous, chain);
            self.metrics.reply_history_count(Count::Probes, 1);
            let prompt_matches = chain.key_at(previous.prompt) == Some(previous.prompt_key);
            if prompt_matches {
                ancestry += 1;
                // Compatible nested turns are ancestry. Distinct replies to
                // one exact prompt remain ambiguous even with identical pages.
                result.ambiguous |= self.completed[..index].iter().any(|other| {
                    other.prompt == previous.prompt
                        && other.prompt_key == previous.prompt_key
                        && (other.end(), other.written_key, other.fingerprints.last())
                            != (
                                previous.end(),
                                previous.written_key,
                                previous.fingerprints.last(),
                            )
                });
                common = common.max(previous.prompt);
                let partial_end = ((common / previous.block + 1) * previous.block)
                    .min(previous.end())
                    .min(chain.tokens().len());
                while common < partial_end {
                    let ordinal = common - previous.prompt;
                    self.metrics.reply_history_count(Count::Fingerprints, 1);
                    if !chain.text_token(common)
                        || previous.fingerprints[ordinal]
                            != fingerprint(previous.prompt_key, ordinal, chain.tokens()[common])
                    {
                        break;
                    }
                    common += 1;
                }
                let end = (previous.end() - 1).min(limit);
                result.candidates.push(previous.prompt, end);
                let mut matched = common.min(end);
                // Exact boundary probes also prove short replies and partial pages.
                self.metrics.reply_history_count(Count::Probes, 1);
                if previous.end() - 1 <= limit
                    && chain.key_at(previous.end() - 1) == Some(previous.written_key)
                {
                    matched = end;
                    if end > result.anchor.end
                        || (end == result.anchor.end && previous.prompt < result.anchor.start)
                    {
                        result.anchor = Interval {
                            start: previous.prompt,
                            end,
                        };
                    }
                }
                result.matching.push(previous.prompt, matched);
                result.pages.push(
                    previous.prompt.div_ceil(previous.block) * previous.block,
                    matched / previous.block * previous.block,
                );
            }
            // Salt equality alone is never ancestry. A matching page or exact
            // prompt boundary is needed even for a before-reply diagnosis.
            if (common > 0 || prompt_matches)
                && best.is_none_or(|(old, old_common)| {
                    (common, previous.prompt, previous.order) > (old_common, old.prompt, old.order)
                })
            {
                best = Some((previous, common));
            }
        }
        result.unobserved = ancestry == 0;
        if self.lost_history {
            self.metrics.reply_history_count(Count::RetentionLoss, 1);
        }
        if result.ambiguous {
            self.metrics.reply_history_count(Count::Ambiguous, 1);
        }
        if let Some((previous, _)) = best {
            self.metrics.reply_history_count(Count::Compared, 1);
            self.diagnose(previous, chain);
        } else {
            self.metrics.reply_history_count(
                if scope {
                    Count::ScopeOnly
                } else {
                    Count::NoHistory
                },
                1,
            );
        }
        result.pages.merge();
        result.matching.merge();
        result.candidates.merge();
        result
    }

    fn diagnose(&self, previous: &History, chain: &PrefixChainV1) {
        if chain.key_at(previous.prompt) != Some(previous.prompt_key) {
            self.metrics.reply_history_count(Count::BeforeReply, 1);
            return;
        }
        let mut matched = 0;
        for (&expected, &token) in previous
            .fingerprints
            .iter()
            .zip(&chain.tokens()[previous.prompt..])
        {
            self.metrics.reply_history_count(Count::Fingerprints, 1);
            if expected != fingerprint(previous.prompt_key, matched, token)
                || !chain.text_token(previous.prompt + matched)
            {
                break;
            }
            matched += 1;
        }
        let outcome = if matched == previous.fingerprints.len() {
            Count::CompleteReply
        } else if previous.prompt + matched == chain.tokens().len() {
            Count::Shorter
        } else if matched + 1 == previous.fingerprints.len() {
            Count::FinalToken
        } else {
            Count::InsideReply
        };
        self.metrics.reply_history_count(outcome, 1);
        if matches!(outcome, Count::InsideReply | Count::FinalToken) {
            self.metrics
                .reply_history_count(region_counter(previous.regions[matched]), 1);
            let quarter = matched * 4 / previous.fingerprints.len().max(1);
            self.metrics.reply_history_count(
                match quarter {
                    0 => Count::PositionFirst,
                    3 => Count::PositionLast,
                    _ => Count::PositionMiddle,
                },
                1,
            );
        }
    }

    /// Records only successfully committed original prompt rows, with a high
    /// water mark supplied by the engine so preemption cannot count them twice.
    pub(crate) fn prefill(&mut self, id: RequestId, start: usize, end: usize, hit: usize) {
        if start >= end {
            return;
        }
        self.metrics.reply_history_rows(Rows::Prefilled, start, end);
        let Some(active) = self
            .active
            .iter_mut()
            .find(|active| active.history.id == id)
        else {
            self.metrics
                .reply_history_rows(Rows::Unobserved, start, end);
            return;
        };
        let comparison = &mut active.comparison;
        comparison
            .candidates
            .count(&self.metrics, Rows::Candidate, start, end);
        let page_start = hit.div_ceil(active.history.block) * active.history.block;
        let pages =
            comparison
                .pages
                .count(&self.metrics, Rows::Repeated, start.max(page_start), end);
        if pages > 0 && !comparison.page_counted {
            self.metrics.reply_history_count(Count::AnyPage, 1);
            comparison.page_counted = true;
        }
        // Difference of the two unions, not rounding their merged envelope:
        // overlapping reply intervals can leave an unproved interior page.
        for range in &comparison.matching.ranges[..comparison.matching.len] {
            let range = range.intersect(start, end);
            let mut cursor = range.start;
            for page in &comparison.pages.ranges[..comparison.pages.len] {
                let page = page.intersect(page_start, range.end);
                if page.len() == 0 || page.end <= cursor {
                    continue;
                }
                let gap_end = page.start.min(range.end);
                if cursor < gap_end {
                    self.metrics
                        .reply_history_rows(Rows::Rounding, cursor, gap_end);
                }
                cursor = cursor.max(page.end);
            }
            if cursor < range.end {
                self.metrics
                    .reply_history_rows(Rows::Rounding, cursor, range.end);
            }
        }
        let anchor = comparison.anchor.intersect(start, end);
        if anchor.len() > 0 {
            self.metrics
                .reply_history_rows(Rows::Eligible, anchor.start, anchor.end);
            if !comparison.anchor_counted {
                self.metrics.reply_history_count(Count::EligibleTurn, 1);
                comparison.anchor_counted = true;
            }
        }
        if comparison.unobserved {
            self.metrics
                .reply_history_rows(Rows::Unobserved, start, end);
        }
        if comparison.ambiguous {
            self.metrics.reply_history_rows(Rows::Ambiguous, start, end);
        }
    }

    pub(crate) fn append(&mut self, id: RequestId, tokens: &[u32]) {
        let Some(active) = self
            .active
            .iter_mut()
            .find(|active| active.history.id == id)
        else {
            return;
        };
        for &token in tokens {
            let history = &mut active.history;
            // Capacity was admitted up front. Never allocate on committed output.
            if history.fingerprints.len() == history.fingerprints.capacity() {
                return;
            }
            history.written_key = active.tail.key();
            active.written_keys.push(history.written_key);
            if let Some(page) = active.tail.push(token) {
                history.pages.push(page);
            }
            history.fingerprints.push(fingerprint(
                history.prompt_key,
                history.fingerprints.len(),
                token,
            ));
            history.regions.push(0);
        }
        self.metrics
            .reply_history_count(Count::Fingerprints, tokens.len());
    }

    pub(crate) fn tool_finish(&mut self, id: RequestId) {
        if let Some(active) = self
            .active
            .iter_mut()
            .find(|active| active.history.id == id)
        {
            active.tool_finish = true;
        }
    }
    pub(crate) fn require_handoff(&mut self, id: RequestId) {
        if let Some(active) = self
            .active
            .iter_mut()
            .find(|active| active.history.id == id)
        {
            active.handoff_required = true;
        }
    }
    fn handoff(&mut self, id: RequestId) {
        if let Some(active) = self
            .active
            .iter_mut()
            .find(|active| active.history.id == id)
        {
            active.handoff_succeeded = true;
            if let Some(reason) = active.frontend_reason {
                self.finish(id, reason);
            }
        }
    }
    pub(crate) fn defer(&mut self, id: RequestId) {
        if let Some(active) = self
            .active
            .iter_mut()
            .find(|active| active.history.id == id)
        {
            active.history.deferred = true;
        }
    }
    pub(crate) fn annotate(&mut self, id: RequestId, ordinal: usize, region: ReplyRegion) {
        let history = self
            .active
            .iter_mut()
            .find(|entry| entry.history.id == id)
            .map(|active| &mut active.history);
        if let Some(mask) = history.and_then(|history| history.regions.get_mut(ordinal)) {
            *mask |= region as u8;
        } else {
            self.metrics.reply_history_count(Count::LateAnnotation, 1);
        }
    }
    pub(crate) fn engine_finish(&mut self, id: RequestId, reason: FinishReason) {
        if reason == FinishReason::Cancelled {
            self.discard(id, Count::Cancelled);
            return;
        }
        let Some(active) = self
            .active
            .iter_mut()
            .find(|active| active.history.id == id)
        else {
            return;
        };
        active.engine_finished = true;
        active.engine_reason = Some(reason);
        if !active.history.deferred {
            self.finish(id, reason);
        }
    }
    pub(crate) fn delivered(&mut self, id: RequestId, reason: FinishReason, tokens: usize) {
        if let Some(active) = self
            .active
            .iter_mut()
            .find(|active| active.history.id == id)
        {
            if tokens == 0 || tokens > active.history.fingerprints.len() {
                self.discard(id, Count::DeliveryFailed);
                return;
            }
            active.frontend_cut |= !active.engine_finished
                || (reason == FinishReason::Stop
                    && active.engine_reason == Some(FinishReason::Length));
            if tokens < active.history.fingerprints.len() {
                active.frontend_cut = true;
                active.engine_finished = false;
                active.history.fingerprints.truncate(tokens);
                active.history.regions.truncate(tokens);
                active.history.written_key = active.written_keys[tokens - 1];
                active
                    .history
                    .pages
                    .truncate(active.history.end() / active.history.block + 1);
            }
        }
        self.finish(id, reason);
    }
    pub(crate) fn finish(&mut self, id: RequestId, reason: FinishReason) {
        // Finalization is idempotent, including a late conflicting callback.
        let Some(index) = self
            .active
            .iter()
            .position(|active| active.history.id == id)
        else {
            return;
        };
        if reason == FinishReason::Cancelled {
            self.discard(id, Count::DeliveryFailed);
            return;
        }
        if self.active[index].handoff_required && !self.active[index].handoff_succeeded {
            self.active[index].frontend_reason = Some(reason);
            return;
        }
        let active = self.active.remove(index);
        if active.history.fingerprints.is_empty() {
            self.publish();
            return;
        }
        let finish = if active.frontend_cut || !active.engine_finished {
            Count::FrontendStop
        } else if reason == FinishReason::Length {
            Count::Length
        } else {
            Count::Natural
        };
        self.metrics.reply_history_count(finish, 1);
        if active.tool_finish {
            self.metrics.reply_history_count(Count::ToolFinish, 1);
        }
        let mut history = active.history;
        // Drop all partial-page token material before retaining the history.
        drop(active.tail);
        drop(active.written_keys);
        self.order += 1;
        history.order = Some(self.order);
        for &mask in &history.regions {
            self.metrics.reply_history_count(
                match region_counter(mask) {
                    Count::RegionUnknown => Count::Unknown,
                    Count::RegionMixed => Count::Mixed,
                    _ => Count::Annotated,
                },
                1,
            );
            if mask & (ReplyRegion::EndToken as u8 | ReplyRegion::IgnoredEndToken as u8) != 0 {
                self.metrics.reply_history_count(Count::EndToken, 1);
            }
            if mask & ReplyRegion::IgnoredEndToken as u8 != 0 {
                self.metrics.reply_history_count(Count::IgnoredEndToken, 1);
            }
        }
        if self.completed.len() == MAX_COMPLETED {
            self.evict();
        }
        self.completed.push(history);
        self.publish();
    }
    pub(crate) fn drop_observation(&mut self, id: RequestId) {
        self.lost_history = true;
        self.discard(id, Count::CapacityDrop);
    }
    pub(crate) fn retire(&mut self, id: RequestId) {
        if self.active.iter().any(|active| {
            active.history.id == id && !active.engine_finished && active.frontend_reason.is_none()
        }) {
            self.discard(id, Count::DeliveryFailed);
        }
    }
    pub(crate) fn discard(&mut self, id: RequestId, reason: Count) {
        if let Some(index) = self.active.iter().position(|entry| entry.history.id == id) {
            self.active.remove(index);
            self.metrics.reply_history_count(reason, 1);
        }
        // Delivery failure can arrive after the engine completion event.
        if let Some(index) = self.completed.iter().position(|entry| entry.id == id) {
            self.completed.remove(index);
            self.metrics.reply_history_count(reason, 1);
        }
        self.publish();
    }
}

fn region_counter(mask: u8) -> Count {
    if mask == 0 || mask == ReplyRegion::Partial as u8 {
        Count::RegionUnknown
    } else if mask.count_ones() > 1 {
        Count::RegionMixed
    } else {
        match mask {
            1 => Count::RegionReasoning,
            2 => Count::RegionContent,
            4 => Count::RegionTool,
            8 | 16 => Count::RegionEnd,
            _ => Count::RegionUnknown,
        }
    }
}

#[cfg(test)]
#[path = "reply_history_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "reply_history_benchmark.rs"]
mod benchmark;
