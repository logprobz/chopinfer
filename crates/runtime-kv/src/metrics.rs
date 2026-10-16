use std::fmt::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::KvStats;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvCachePrecision {
    Other,
    Fp8,
}

#[derive(Debug)]
struct KvMetricsInner {
    precision: KvCachePrecision,
    tokens: AtomicU64,
    capacity_tokens: AtomicU64,
    blocks_used: AtomicU64,
    block_capacity: AtomicU64,
    active_sequences: AtomicU64,
    explicit_shared_prefix_forks: AtomicU64,
    explicit_shared_prefix_tokens: AtomicU64,
    prefix_cache_queries: AtomicU64,
    prefix_cache_hits: AtomicU64,
    prefix_cache_reused_tokens: AtomicU64,
    prefix_cache_queried_tokens: AtomicU64,
    prefix_cache_recomputed_seen_tokens: AtomicU64,
    prefix_cache_captures: AtomicU64,
    prefix_cache_capture_failures: AtomicU64,
    prefix_cache_restore_failures: AtomicU64,
    prefix_restore_resumed_decoding: AtomicU64,
    prefix_cache_evictions: AtomicU64,
    prefix_cache_anchors: AtomicU64,
    prefix_cache_blocks: AtomicU64,
    cow_copies: AtomicU64,
    cow_tokens: AtomicU64,
}

#[derive(Clone, Debug)]
pub struct KvMetrics {
    inner: Arc<KvMetricsInner>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvMetricsSnapshot {
    /// The precision this pool was constructed for.
    ///
    /// The FP8 names below are published only when this says FP8, because a
    /// pool of some other precision holding tokens is not an FP8 observation.
    pub precision: KvCachePrecision,
    pub tokens: u64,
    pub capacity_tokens: u64,
    pub blocks_used: u64,
    pub block_capacity: u64,
    pub active_sequences: u64,
    pub fp8_tokens: u64,
    pub fp8_capacity_tokens: u64,
    pub explicit_shared_prefix_forks: u64,
    pub explicit_shared_prefix_tokens: u64,
    /// Lookups, hit or miss.
    pub prefix_cache_queries: u64,
    /// Hits whose restore succeeded. A hit whose restore is refused or
    /// cancelled counts only as a query, and a refusal as a restore failure.
    pub prefix_cache_hits: u64,
    /// Prompt tokens those hits skipped, as vLLM counts hits.
    pub prefix_cache_reused_tokens: u64,
    /// Prompt tokens of every lookup, hit or miss, as vLLM counts queries.
    pub prefix_cache_queried_tokens: u64,
    /// Prompt tokens prefilled again although an earlier request's committed
    /// prefill computed the same leading full blocks (a repeat or shared
    /// prefix no anchor covered).
    pub prefix_cache_recomputed_seen_tokens: u64,
    /// Capture attempts: anchors that got snapshot pages. A capture the
    /// bundle then refuses also counts as a capture failure.
    pub prefix_cache_captures: u64,
    pub prefix_cache_capture_failures: u64,
    pub prefix_cache_restore_failures: u64,
    /// Successful restores at the request's exact committed output boundary.
    pub prefix_restore_resumed_decoding: u64,
    pub prefix_cache_evictions: u64,
    pub prefix_cache_anchors: u64,
    pub prefix_cache_blocks: u64,
    pub cow_copies: u64,
    pub cow_tokens: u64,
}

impl KvMetrics {
    #[must_use]
    pub fn new(precision: KvCachePrecision) -> Self {
        Self {
            inner: Arc::new(KvMetricsInner {
                precision,
                tokens: AtomicU64::new(0),
                capacity_tokens: AtomicU64::new(0),
                blocks_used: AtomicU64::new(0),
                block_capacity: AtomicU64::new(0),
                active_sequences: AtomicU64::new(0),
                explicit_shared_prefix_forks: AtomicU64::new(0),
                explicit_shared_prefix_tokens: AtomicU64::new(0),
                prefix_cache_queries: AtomicU64::new(0),
                prefix_cache_hits: AtomicU64::new(0),
                prefix_cache_reused_tokens: AtomicU64::new(0),
                prefix_cache_queried_tokens: AtomicU64::new(0),
                prefix_cache_recomputed_seen_tokens: AtomicU64::new(0),
                prefix_cache_captures: AtomicU64::new(0),
                prefix_cache_capture_failures: AtomicU64::new(0),
                prefix_cache_restore_failures: AtomicU64::new(0),
                prefix_restore_resumed_decoding: AtomicU64::new(0),
                prefix_cache_evictions: AtomicU64::new(0),
                prefix_cache_anchors: AtomicU64::new(0),
                prefix_cache_blocks: AtomicU64::new(0),
                cow_copies: AtomicU64::new(0),
                cow_tokens: AtomicU64::new(0),
            }),
        }
    }

    /// Records the block occupancy one pool reports about itself.
    ///
    /// The argument is a [`KvStats`], which only a pool produces, so occupancy
    /// here is the pool's own free-list arithmetic rather than a number a
    /// caller chose. A holder of this handle publishes what the pool last did;
    /// it cannot assert an occupancy the pool never reached.
    pub fn observe_occupancy(&self, stats: KvStats) {
        self.inner
            .block_capacity
            .store(saturating_u64(stats.block_capacity()), Ordering::Relaxed);
        self.inner
            .blocks_used
            .store(saturating_u64(stats.used_blocks()), Ordering::Relaxed);
        self.inner
            .active_sequences
            .store(saturating_u64(stats.active_sequences()), Ordering::Relaxed);
        self.inner
            .prefix_cache_anchors
            .store(saturating_u64(stats.anchors()), Ordering::Relaxed);
        self.inner
            .prefix_cache_blocks
            .store(saturating_u64(stats.cached_blocks()), Ordering::Relaxed);
    }

    /// Counts a snapshot the bundle failed to write; its anchor was dropped.
    pub fn prefix_capture_failed(&self) {
        self.inner
            .prefix_cache_capture_failures
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Counts a lookup whose hit the scheduler did not take; the request
    /// prefills from zero.
    pub fn prefix_cache_missed(&self, prompt_tokens: usize) {
        self.prefix_cache_lookup(prompt_tokens);
    }

    /// Counts prompt tokens a committed prefill computed although an earlier
    /// request's committed prefill had computed the same leading full blocks:
    /// the part of a repeated or shared prefix, short of the prompt's last
    /// token, that no anchor covered.
    pub fn prefix_cache_recomputed_seen(&self, tokens: usize) {
        self.inner
            .prefix_cache_recomputed_seen_tokens
            .fetch_add(saturating_u64(tokens), Ordering::Relaxed);
    }

    /// Counts a restore the bundle refused; the request prefilled from zero.
    pub fn prefix_restore_failed(&self) {
        self.inner
            .prefix_cache_restore_failures
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Counts a successful restore that resumes decoding without replay.
    pub fn prefix_restore_resumed_decoding(&self) {
        self.inner
            .prefix_restore_resumed_decoding
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn prefix_capture(&self) {
        self.inner
            .prefix_cache_captures
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn prefix_eviction(&self) {
        self.inner
            .prefix_cache_evictions
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn configure_capacity(&self, tokens: usize) {
        self.inner
            .capacity_tokens
            .store(saturating_u64(tokens), Ordering::Relaxed);
    }

    pub(crate) fn appended(&self, tokens: usize) {
        self.inner
            .tokens
            .fetch_add(saturating_u64(tokens), Ordering::Relaxed);
    }

    pub(crate) fn released(&self, tokens: usize) {
        saturating_sub(&self.inner.tokens, saturating_u64(tokens));
    }

    pub(crate) fn explicit_shared_prefix(&self, tokens: usize) {
        self.inner
            .explicit_shared_prefix_forks
            .fetch_add(1, Ordering::Relaxed);
        self.inner
            .explicit_shared_prefix_tokens
            .fetch_add(saturating_u64(tokens), Ordering::Relaxed);
    }

    /// Counts one lookup of a prompt, whether or not it finds an anchor.
    pub(crate) fn prefix_cache_lookup(&self, prompt_tokens: usize) {
        self.inner
            .prefix_cache_queries
            .fetch_add(1, Ordering::Relaxed);
        self.inner
            .prefix_cache_queried_tokens
            .fetch_add(saturating_u64(prompt_tokens), Ordering::Relaxed);
    }

    /// Counts a hit whose restore succeeded, so its tokens were not prefilled.
    pub(crate) fn prefix_cache_restored(&self, tokens: usize) {
        self.inner.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);
        self.inner
            .prefix_cache_reused_tokens
            .fetch_add(saturating_u64(tokens), Ordering::Relaxed);
    }

    pub(crate) fn cow_copy(&self, tokens: usize) {
        self.inner.cow_copies.fetch_add(1, Ordering::Relaxed);
        self.inner
            .cow_tokens
            .fetch_add(saturating_u64(tokens), Ordering::Relaxed);
    }

    #[must_use]
    pub fn snapshot(&self) -> KvMetricsSnapshot {
        let tokens = self.inner.tokens.load(Ordering::Relaxed);
        let capacity_tokens = self.inner.capacity_tokens.load(Ordering::Relaxed);
        let fp8 = self.inner.precision == KvCachePrecision::Fp8;
        KvMetricsSnapshot {
            precision: self.inner.precision,
            tokens,
            capacity_tokens,
            blocks_used: self.inner.blocks_used.load(Ordering::Relaxed),
            block_capacity: self.inner.block_capacity.load(Ordering::Relaxed),
            active_sequences: self.inner.active_sequences.load(Ordering::Relaxed),
            fp8_tokens: if fp8 { tokens } else { 0 },
            fp8_capacity_tokens: if fp8 { capacity_tokens } else { 0 },
            explicit_shared_prefix_forks: self
                .inner
                .explicit_shared_prefix_forks
                .load(Ordering::Relaxed),
            explicit_shared_prefix_tokens: self
                .inner
                .explicit_shared_prefix_tokens
                .load(Ordering::Relaxed),
            prefix_cache_queries: self.inner.prefix_cache_queries.load(Ordering::Relaxed),
            prefix_cache_hits: self.inner.prefix_cache_hits.load(Ordering::Relaxed),
            prefix_cache_reused_tokens: self
                .inner
                .prefix_cache_reused_tokens
                .load(Ordering::Relaxed),
            prefix_cache_queried_tokens: self
                .inner
                .prefix_cache_queried_tokens
                .load(Ordering::Relaxed),
            prefix_cache_recomputed_seen_tokens: self
                .inner
                .prefix_cache_recomputed_seen_tokens
                .load(Ordering::Relaxed),
            prefix_cache_captures: self.inner.prefix_cache_captures.load(Ordering::Relaxed),
            prefix_cache_capture_failures: self
                .inner
                .prefix_cache_capture_failures
                .load(Ordering::Relaxed),
            prefix_cache_restore_failures: self
                .inner
                .prefix_cache_restore_failures
                .load(Ordering::Relaxed),
            prefix_restore_resumed_decoding: self
                .inner
                .prefix_restore_resumed_decoding
                .load(Ordering::Relaxed),
            prefix_cache_evictions: self.inner.prefix_cache_evictions.load(Ordering::Relaxed),
            prefix_cache_anchors: self.inner.prefix_cache_anchors.load(Ordering::Relaxed),
            prefix_cache_blocks: self.inner.prefix_cache_blocks.load(Ordering::Relaxed),
            cow_copies: self.inner.cow_copies.load(Ordering::Relaxed),
            cow_tokens: self.inner.cow_tokens.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn detached_clone(&self) -> Self {
        let snapshot = self.snapshot();
        let metrics = Self::new(self.inner.precision);
        metrics
            .inner
            .tokens
            .store(snapshot.tokens, Ordering::Relaxed);
        metrics
            .inner
            .capacity_tokens
            .store(snapshot.capacity_tokens, Ordering::Relaxed);
        metrics
            .inner
            .blocks_used
            .store(snapshot.blocks_used, Ordering::Relaxed);
        metrics
            .inner
            .block_capacity
            .store(snapshot.block_capacity, Ordering::Relaxed);
        metrics
            .inner
            .active_sequences
            .store(snapshot.active_sequences, Ordering::Relaxed);
        metrics
            .inner
            .explicit_shared_prefix_forks
            .store(snapshot.explicit_shared_prefix_forks, Ordering::Relaxed);
        metrics
            .inner
            .explicit_shared_prefix_tokens
            .store(snapshot.explicit_shared_prefix_tokens, Ordering::Relaxed);
        metrics
            .inner
            .prefix_cache_queries
            .store(snapshot.prefix_cache_queries, Ordering::Relaxed);
        metrics
            .inner
            .prefix_cache_hits
            .store(snapshot.prefix_cache_hits, Ordering::Relaxed);
        metrics
            .inner
            .prefix_cache_reused_tokens
            .store(snapshot.prefix_cache_reused_tokens, Ordering::Relaxed);
        metrics
            .inner
            .prefix_cache_queried_tokens
            .store(snapshot.prefix_cache_queried_tokens, Ordering::Relaxed);
        for (counter, value) in [
            (
                &metrics.inner.prefix_cache_recomputed_seen_tokens,
                snapshot.prefix_cache_recomputed_seen_tokens,
            ),
            (
                &metrics.inner.prefix_restore_resumed_decoding,
                snapshot.prefix_restore_resumed_decoding,
            ),
            (
                &metrics.inner.prefix_cache_captures,
                snapshot.prefix_cache_captures,
            ),
            (
                &metrics.inner.prefix_cache_capture_failures,
                snapshot.prefix_cache_capture_failures,
            ),
            (
                &metrics.inner.prefix_cache_restore_failures,
                snapshot.prefix_cache_restore_failures,
            ),
            (
                &metrics.inner.prefix_cache_evictions,
                snapshot.prefix_cache_evictions,
            ),
            (
                &metrics.inner.prefix_cache_anchors,
                snapshot.prefix_cache_anchors,
            ),
            (
                &metrics.inner.prefix_cache_blocks,
                snapshot.prefix_cache_blocks,
            ),
        ] {
            counter.store(value, Ordering::Relaxed);
        }
        metrics
            .inner
            .cow_copies
            .store(snapshot.cow_copies, Ordering::Relaxed);
        metrics
            .inner
            .cow_tokens
            .store(snapshot.cow_tokens, Ordering::Relaxed);
        metrics
    }
}

impl Default for KvMetrics {
    fn default() -> Self {
        Self::new(KvCachePrecision::Other)
    }
}

impl KvMetricsSnapshot {
    #[must_use]
    pub fn prometheus(&self) -> String {
        let mut output = String::new();
        metric(&mut output, "chop_kv_tokens", "gauge", self.tokens);
        metric(
            &mut output,
            "chop_kv_capacity_tokens",
            "gauge",
            self.capacity_tokens,
        );
        metric(
            &mut output,
            "chop_kv_blocks_used",
            "gauge",
            self.blocks_used,
        );
        metric(
            &mut output,
            "chop_kv_block_capacity",
            "gauge",
            self.block_capacity,
        );
        metric(
            &mut output,
            "chop_kv_active_sequences",
            "gauge",
            self.active_sequences,
        );
        // The FP8 names say FP8, so they are published only by a pool that was
        // constructed for FP8. A pool of another precision publishes the
        // precision-neutral names above and nothing that claims FP8.
        if self.precision == KvCachePrecision::Fp8 {
            metric(&mut output, "chop_fp8_kv_tokens", "gauge", self.fp8_tokens);
            metric(
                &mut output,
                "chop_fp8_kv_capacity_tokens",
                "gauge",
                self.fp8_capacity_tokens,
            );
        }
        metric(
            &mut output,
            "chop_explicit_shared_prefix_forks_total",
            "counter",
            self.explicit_shared_prefix_forks,
        );
        metric(
            &mut output,
            "chop_explicit_shared_prefix_tokens_total",
            "counter",
            self.explicit_shared_prefix_tokens,
        );
        metric(
            &mut output,
            "chop_prefix_cache_queries_total",
            "counter",
            self.prefix_cache_queries,
        );
        metric(
            &mut output,
            "chop_prefix_cache_hits_total",
            "counter",
            self.prefix_cache_hits,
        );
        metric(
            &mut output,
            "chop_prefix_cache_reused_tokens_total",
            "counter",
            self.prefix_cache_reused_tokens,
        );
        metric(
            &mut output,
            "chop_prefix_cache_queried_tokens_total",
            "counter",
            self.prefix_cache_queried_tokens,
        );
        for (name, kind, value) in [
            (
                "chop_prefix_cache_recomputed_seen_tokens_total",
                "counter",
                self.prefix_cache_recomputed_seen_tokens,
            ),
            (
                "chop_prefix_restore_resumed_decoding_total",
                "counter",
                self.prefix_restore_resumed_decoding,
            ),
            (
                "chop_prefix_cache_captures_total",
                "counter",
                self.prefix_cache_captures,
            ),
            (
                "chop_prefix_cache_capture_failures_total",
                "counter",
                self.prefix_cache_capture_failures,
            ),
            (
                "chop_prefix_cache_restore_failures_total",
                "counter",
                self.prefix_cache_restore_failures,
            ),
            (
                "chop_prefix_cache_evictions_total",
                "counter",
                self.prefix_cache_evictions,
            ),
            (
                "chop_prefix_cache_anchors",
                "gauge",
                self.prefix_cache_anchors,
            ),
            (
                "chop_prefix_cache_blocks",
                "gauge",
                self.prefix_cache_blocks,
            ),
        ] {
            metric(&mut output, name, kind, value);
        }
        metric(
            &mut output,
            "chop_cow_copies_total",
            "counter",
            self.cow_copies,
        );
        metric(
            &mut output,
            "chop_cow_copied_tokens_total",
            "counter",
            self.cow_tokens,
        );
        output
    }
}

fn metric(output: &mut String, name: &str, kind: &str, value: u64) {
    writeln!(output, "# TYPE {name} {kind}\n{name} {value}")
        .expect("writing metrics to a string is infallible");
}

fn saturating_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn saturating_sub(value: &AtomicU64, decrement: u64) {
    let _ = value.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_sub(decrement))
    });
}
