use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::speculation::MAX_DRAFT_TOKENS;

pub trait MonotonicClock: std::fmt::Debug + Send + Sync + 'static {
    fn now_micros(&self) -> u64;
}

#[derive(Debug)]
pub struct SystemMonotonicClock {
    origin: Instant,
}

impl SystemMonotonicClock {
    #[must_use]
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Default for SystemMonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl MonotonicClock for SystemMonotonicClock {
    fn now_micros(&self) -> u64 {
        u64::try_from(self.origin.elapsed().as_micros()).unwrap_or(u64::MAX)
    }
}

/// What one dispatched batch was made of, counted from its items' own phases.
///
/// This is the scheduler's decision read back, not an arrival pattern: two
/// requests whose HTTP arrivals overlap may still be executed in separate
/// single-phase batches, and only these counts say which happened.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct BatchComposition {
    prefill_items: usize,
    decode_items: usize,
}

impl BatchComposition {
    pub(crate) const fn new(prefill_items: usize, decode_items: usize) -> Self {
        Self {
            prefill_items,
            decode_items,
        }
    }

    const fn is_mixed(self) -> bool {
        self.prefill_items > 0 && self.decode_items > 0
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExecutionObservations {
    pub kernel_launches: u64,
    pub kernel_micros: u64,
    pub collective_operations: u64,
    pub collective_micros: u64,
}

#[derive(Debug, Default)]
struct EngineMetricsInner {
    offload: Mutex<OffloadMetrics>,
    profiles: Arc<crate::profile::StepProfiles>,
    poisoned: AtomicU64,
    speculation_k: [AtomicU64; MAX_DRAFT_TOKENS + 1],
    compatibility: Mutex<crate::compatibility::CompatibilityMetrics>,
    lifecycle: Mutex<LifecycleMetrics>,
    prompt_tokens: AtomicU64,
    generation_tokens: AtomicU64,
    queue_micros: AtomicU64,
    queue_observations: AtomicU64,
    prefix_capture_micros: AtomicU64,
    prefix_restore_micros: AtomicU64,
    batches: AtomicU64,
    batch_items: AtomicU64,
    prefill_only_batches: AtomicU64,
    decode_only_batches: AtomicU64,
    mixed_batches: AtomicU64,
    prefill_batch_items: AtomicU64,
    decode_batch_items: AtomicU64,
    ttft_micros: AtomicU64,
    ttft_observations: AtomicU64,
    itl_micros: AtomicU64,
    itl_observations: AtomicU64,
    execution_observations: AtomicU64,
    kernel_launches: AtomicU64,
    kernel_micros: AtomicU64,
    collective_operations: AtomicU64,
    collective_micros: AtomicU64,
    errors: AtomicU64,
    refused_batches: AtomicU64,
    refused_requests: AtomicU64,
    spec_verify_items: AtomicU64,
    spec_draft_tokens: AtomicU64,
    spec_accepted_tokens: AtomicU64,
    spec_emitted_tokens: AtomicU64,
    spec_rows: AtomicU64,
    spec_drafted_positions: [AtomicU64; MAX_DRAFT_TOKENS],
    spec_accepted_positions: [AtomicU64; MAX_DRAFT_TOKENS],
    media_encodes: AtomicU64,
    media_encoded_rows: AtomicU64,
    media_cache_hits: AtomicU64,
    media_evictions: AtomicU64,
    media_reencodes: AtomicU64,
}

#[derive(Clone, Copy, Debug, Default)]
struct LifecycleMetrics {
    requests_submitted: u64,
    requests_completed: u64,
    requests_terminal: u64,
    requests_active: u64,
    sequences_active: u64,
    cancellations: u64,
}

impl LifecycleMetrics {
    fn request_terminal(&mut self, count: u64) {
        self.requests_terminal = self.requests_terminal.saturating_add(count);
        self.requests_active = self.requests_active.saturating_sub(count);
        self.sequences_active = self.sequences_active.saturating_sub(count);
    }
}

#[derive(Clone, Debug, Default)]
pub struct EngineMetrics {
    inner: Arc<EngineMetricsInner>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EngineMetricsSnapshot {
    pub offload: OffloadMetrics,
    pub requests_submitted: u64,
    pub requests_completed: u64,
    pub requests_terminal: u64,
    pub requests_active: u64,
    pub prompt_tokens: u64,
    pub generation_tokens: u64,
    pub queue_micros: u64,
    pub queue_observations: u64,
    /// Engine-thread time spent in the executor's prefix-state captures and
    /// restores, successful or not.
    pub prefix_capture_micros: u64,
    pub prefix_restore_micros: u64,
    pub sequences_active: u64,
    pub batches: u64,
    pub batch_items: u64,
    pub prefill_only_batches: u64,
    pub decode_only_batches: u64,
    pub mixed_batches: u64,
    pub prefill_batch_items: u64,
    pub decode_batch_items: u64,
    pub ttft_micros: u64,
    pub ttft_observations: u64,
    pub itl_micros: u64,
    pub itl_observations: u64,
    /// How many executions measured themselves.
    ///
    /// The kernel and collective counters below are meaningless without this:
    /// an executor that measures nothing leaves them at zero, which is not the
    /// same as an execution that launched nothing.
    pub execution_observations: u64,
    pub kernel_launches: u64,
    pub kernel_micros: u64,
    pub collective_operations: u64,
    pub collective_micros: u64,
    pub errors: u64,
    pub cancellations: u64,
    /// Verify items executed.
    pub spec_verify_items: u64,
    /// Drafts verified, the sum of every verify item's draft count.
    pub spec_draft_tokens: u64,
    /// Drafts accepted.
    pub spec_accepted_tokens: u64,
    /// Tokens verify items emitted, after stop and length cuts.
    pub spec_emitted_tokens: u64,
    /// Decode-phase rows verify items ran: drafts plus one each.
    pub spec_rows: u64,
    /// Drafts verified at each draft position, starting at the first draft.
    pub spec_drafted_positions: [u64; MAX_DRAFT_TOKENS],
    /// Drafts accepted at each draft position.
    pub spec_accepted_positions: [u64; MAX_DRAFT_TOKENS],
    /// Media items encoded into the arena.
    pub media_encodes: u64,
    /// Arena rows those encodes wrote.
    pub media_encoded_rows: u64,
    /// Media items found resident, which skipped their encoder.
    pub media_cache_hits: u64,
    /// Unpinned arena entries evicted to make room.
    pub media_evictions: u64,
    /// Media items encoded again for a request whose earlier chunk had
    /// placed them, after an eviction between its steps.
    pub media_reencodes: u64,
}

impl EngineMetrics {
    pub(crate) fn start_step_profile(&self) -> Option<crate::profile::StepTimer> {
        self.inner.profiles.begin()
    }

    /// Returns bounded sampled host timings, or None when profiling is disabled.
    #[must_use]
    pub fn step_profile_json(&self) -> Option<String> {
        self.inner.profiles.json()
    }

    pub(crate) fn poison(&self) {
        self.inner.poisoned.store(1, Ordering::Relaxed);
    }

    pub(crate) fn speculation_k(&self, k: usize) {
        self.inner.speculation_k[k.min(MAX_DRAFT_TOKENS)].fetch_add(1, Ordering::Relaxed);
    }

    /// Correlates optional JSON request traces with the transport's choice ID.
    pub fn set_request_trace_id(&self, id: crate::RequestId, trace_id: &str) {
        self.compatibility(|metrics| metrics.trace_id(id, trace_id));
    }

    pub(crate) fn request_submitted(&self, prompt_tokens: usize) {
        let mut lifecycle = self
            .inner
            .lifecycle
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        lifecycle.requests_submitted = lifecycle.requests_submitted.saturating_add(1);
        lifecycle.requests_active = lifecycle.requests_active.saturating_add(1);
        lifecycle.sequences_active = lifecycle.sequences_active.saturating_add(1);
        drop(lifecycle);
        self.inner
            .prompt_tokens
            .fetch_add(saturating_u64(prompt_tokens), Ordering::Relaxed);
    }

    pub(crate) fn request_succeeded(&self, count: usize) {
        let count = saturating_u64(count);
        let mut lifecycle = self
            .inner
            .lifecycle
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        lifecycle.requests_completed = lifecycle.requests_completed.saturating_add(count);
        lifecycle.request_terminal(count);
    }

    pub(crate) fn batch_refused(&self, count: usize) {
        self.inner.refused_batches.fetch_add(1, Ordering::Relaxed);
        self.inner
            .refused_requests
            .fetch_add(saturating_u64(count), Ordering::Relaxed);
    }

    pub(crate) fn request_failed(&self, count: usize) {
        self.inner
            .lifecycle
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .request_terminal(saturating_u64(count));
    }

    pub(crate) fn cancellation(&self) {
        let mut lifecycle = self
            .inner
            .lifecycle
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        lifecycle.cancellations = lifecycle.cancellations.saturating_add(1);
        lifecycle.request_terminal(1);
    }

    pub(crate) fn error(&self) {
        self.inner.errors.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn batch(&self, composition: BatchComposition) {
        let prefill = saturating_u64(composition.prefill_items);
        let decode = saturating_u64(composition.decode_items);
        self.inner.batches.fetch_add(1, Ordering::Relaxed);
        self.inner
            .batch_items
            .fetch_add(prefill.saturating_add(decode), Ordering::Relaxed);
        self.inner
            .prefill_batch_items
            .fetch_add(prefill, Ordering::Relaxed);
        self.inner
            .decode_batch_items
            .fetch_add(decode, Ordering::Relaxed);
        let class = if composition.is_mixed() {
            &self.inner.mixed_batches
        } else if prefill > 0 {
            &self.inner.prefill_only_batches
        } else if decode > 0 {
            &self.inner.decode_only_batches
        } else {
            return;
        };
        class.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn queue(&self, micros: u64) {
        self.inner.queue_micros.fetch_add(micros, Ordering::Relaxed);
        self.inner
            .queue_observations
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn prefix_state_call(&self, capture: bool, micros: u64) {
        let counter = if capture {
            &self.inner.prefix_capture_micros
        } else {
            &self.inner.prefix_restore_micros
        };
        counter.fetch_add(micros, Ordering::Relaxed);
    }

    pub(crate) fn generated(&self, count: usize) {
        self.inner
            .generation_tokens
            .fetch_add(saturating_u64(count), Ordering::Relaxed);
    }

    pub(crate) fn ttft(&self, micros: u64) {
        self.compatibility(|metrics| metrics.ttft.observe(micros));
        self.inner.ttft_micros.fetch_add(micros, Ordering::Relaxed);
        self.inner.ttft_observations.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one interval between output updates, matching vLLM's ITL
    /// histogram. A speculative run contributes one observation regardless
    /// of its token count; this is not mean time per output token.
    pub(crate) fn itl(&self, micros: u64) {
        self.compatibility(|metrics| metrics.itl.observe(micros));
        self.inner.itl_micros.fetch_add(micros, Ordering::Relaxed);
        self.inner.itl_observations.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one execution the executor measured for itself.
    ///
    /// Only a measured execution reaches here, so the counters below stay at
    /// zero exactly when nothing measured them, and the snapshot can leave
    /// them unpublished rather than report a zero as an observation.
    pub(crate) fn execution(&self, observations: ExecutionObservations) {
        self.compatibility(|metrics| metrics.gpu.observe(observations.kernel_micros));
        self.inner
            .execution_observations
            .fetch_add(1, Ordering::Relaxed);
        self.inner
            .kernel_launches
            .fetch_add(observations.kernel_launches, Ordering::Relaxed);
        self.inner
            .kernel_micros
            .fetch_add(observations.kernel_micros, Ordering::Relaxed);
        self.inner
            .collective_operations
            .fetch_add(observations.collective_operations, Ordering::Relaxed);
        self.inner
            .collective_micros
            .fetch_add(observations.collective_micros, Ordering::Relaxed);
    }

    /// Records one verify item that verified `drafted` drafts, accepted the
    /// first `accepted`, and emitted `emitted` tokens.
    pub(crate) fn verified(&self, drafted: usize, accepted: usize, emitted: usize) {
        let inner = &self.inner;
        inner.spec_verify_items.fetch_add(1, Ordering::Relaxed);
        inner
            .spec_draft_tokens
            .fetch_add(saturating_u64(drafted), Ordering::Relaxed);
        inner
            .spec_accepted_tokens
            .fetch_add(saturating_u64(accepted), Ordering::Relaxed);
        inner
            .spec_emitted_tokens
            .fetch_add(saturating_u64(emitted), Ordering::Relaxed);
        inner
            .spec_rows
            .fetch_add(saturating_u64(drafted.saturating_add(1)), Ordering::Relaxed);
        for position in inner.spec_drafted_positions.iter().take(drafted) {
            position.fetch_add(1, Ordering::Relaxed);
        }
        for position in inner.spec_accepted_positions.iter().take(accepted) {
            position.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Records one step's media placement.
    pub(crate) fn media(
        &self,
        encodes: usize,
        rows: u64,
        hits: usize,
        evictions: usize,
        reencodes: usize,
    ) {
        let inner = &self.inner;
        inner
            .media_encodes
            .fetch_add(saturating_u64(encodes), Ordering::Relaxed);
        inner.media_encoded_rows.fetch_add(rows, Ordering::Relaxed);
        inner
            .media_cache_hits
            .fetch_add(saturating_u64(hits), Ordering::Relaxed);
        inner
            .media_evictions
            .fetch_add(saturating_u64(evictions), Ordering::Relaxed);
        inner
            .media_reencodes
            .fetch_add(saturating_u64(reencodes), Ordering::Relaxed);
    }

    pub(crate) fn compatibility<R>(
        &self,
        observe: impl FnOnce(&mut crate::compatibility::CompatibilityMetrics) -> R,
    ) -> R {
        observe(
            &mut self
                .inner
                .compatibility
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
        )
    }

    /// Requests the reference metric set still keeps a record of: every
    /// submitted request until it finishes, fails or its frontend finishes
    /// it. Zero once every request has left.
    #[must_use]
    pub fn live_request_records(&self) -> usize {
        self.compatibility(|metrics| metrics.live_records())
    }

    /// Publishes the documented reference metric set for one local engine.
    #[must_use]
    pub fn prometheus_compatibility(&self, model: &str) -> String {
        let labels = crate::compatibility::labels(model);
        let mut output = self
            .inner
            .compatibility
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .render(&labels);
        writeln!(
            output,
            "# TYPE chop_instance_poisoned gauge\nchop_instance_poisoned{{{labels}}} {}",
            self.inner.poisoned.load(Ordering::Relaxed)
        )
        .unwrap();
        for (name, counter) in [
            ("vllm:batch_refusals_total", &self.inner.refused_batches),
            (
                "vllm:batch_refused_requests_total",
                &self.inner.refused_requests,
            ),
        ] {
            writeln!(
                output,
                "# TYPE {name} counter\n{name}{{{labels}}} {}",
                counter.load(Ordering::Relaxed)
            )
            .unwrap();
        }
        writeln!(output, "# TYPE chop_speculation_k_total counter").unwrap();
        for (k, count) in self.inner.speculation_k.iter().enumerate() {
            writeln!(
                output,
                "chop_speculation_k_total{{{labels},k=\"{k}\"}} {}",
                count.load(Ordering::Relaxed)
            )
            .unwrap();
        }
        let snapshot = self.snapshot();
        for (name, value) in [
            ("vllm:generation_tokens_total", snapshot.generation_tokens),
            (
                "vllm:spec_decode_num_drafts_total",
                snapshot.spec_verify_items,
            ),
            (
                "vllm:spec_decode_num_draft_tokens_total",
                snapshot.spec_draft_tokens,
            ),
            (
                "vllm:spec_decode_num_accepted_tokens_total",
                snapshot.spec_accepted_tokens,
            ),
        ] {
            crate::compatibility::counter(&mut output, name, &labels, value);
        }
        for (name, values) in [
            (
                "vllm:spec_decode_num_draft_tokens_per_pos_total",
                snapshot.spec_drafted_positions,
            ),
            (
                "vllm:spec_decode_num_accepted_tokens_per_pos_total",
                snapshot.spec_accepted_positions,
            ),
        ] {
            writeln!(output, "# TYPE {name} counter").unwrap();
            for (position, value) in values.iter().enumerate() {
                writeln!(output, "{name}{{{labels},position=\"{position}\"}} {value}").unwrap();
            }
        }
        output
    }

    #[must_use]
    pub fn snapshot(&self) -> EngineMetricsSnapshot {
        let lifecycle = *self
            .inner
            .lifecycle
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        macro_rules! load {
            ($field:ident) => {
                self.inner.$field.load(Ordering::Relaxed)
            };
        }
        EngineMetricsSnapshot {
            offload: *self.inner.offload.lock().unwrap_or_else(|e| e.into_inner()),
            requests_submitted: lifecycle.requests_submitted,
            requests_completed: lifecycle.requests_completed,
            requests_terminal: lifecycle.requests_terminal,
            requests_active: lifecycle.requests_active,
            prompt_tokens: load!(prompt_tokens),
            generation_tokens: load!(generation_tokens),
            queue_micros: load!(queue_micros),
            queue_observations: load!(queue_observations),
            prefix_capture_micros: load!(prefix_capture_micros),
            prefix_restore_micros: load!(prefix_restore_micros),
            sequences_active: lifecycle.sequences_active,
            batches: load!(batches),
            batch_items: load!(batch_items),
            prefill_only_batches: load!(prefill_only_batches),
            decode_only_batches: load!(decode_only_batches),
            mixed_batches: load!(mixed_batches),
            prefill_batch_items: load!(prefill_batch_items),
            decode_batch_items: load!(decode_batch_items),
            ttft_micros: load!(ttft_micros),
            ttft_observations: load!(ttft_observations),
            itl_micros: load!(itl_micros),
            itl_observations: load!(itl_observations),
            execution_observations: load!(execution_observations),
            kernel_launches: load!(kernel_launches),
            kernel_micros: load!(kernel_micros),
            collective_operations: load!(collective_operations),
            collective_micros: load!(collective_micros),
            errors: load!(errors),
            cancellations: lifecycle.cancellations,
            spec_verify_items: load!(spec_verify_items),
            spec_draft_tokens: load!(spec_draft_tokens),
            spec_accepted_tokens: load!(spec_accepted_tokens),
            spec_emitted_tokens: load!(spec_emitted_tokens),
            spec_rows: load!(spec_rows),
            spec_drafted_positions: std::array::from_fn(|position| {
                self.inner.spec_drafted_positions[position].load(Ordering::Relaxed)
            }),
            spec_accepted_positions: std::array::from_fn(|position| {
                self.inner.spec_accepted_positions[position].load(Ordering::Relaxed)
            }),
            media_encodes: load!(media_encodes),
            media_encoded_rows: load!(media_encoded_rows),
            media_cache_hits: load!(media_cache_hits),
            media_evictions: load!(media_evictions),
            media_reencodes: load!(media_reencodes),
        }
    }
}

impl EngineMetricsSnapshot {
    #[must_use]
    pub fn prometheus(&self) -> String {
        let mut output = String::new();
        self.offload.render(&mut output);
        metric(
            &mut output,
            "chop_requests_submitted_total",
            "counter",
            self.requests_submitted,
        );
        metric(
            &mut output,
            "chop_requests_completed_total",
            "counter",
            self.requests_completed,
        );
        metric(
            &mut output,
            "chop_requests_terminal_total",
            "counter",
            self.requests_terminal,
        );
        metric(
            &mut output,
            "chop_requests_active",
            "gauge",
            self.requests_active,
        );
        metric(
            &mut output,
            "chop_prompt_tokens_total",
            "counter",
            self.prompt_tokens,
        );
        metric(
            &mut output,
            "chop_generation_tokens_total",
            "counter",
            self.generation_tokens,
        );
        metric(
            &mut output,
            "chop_queue_duration_microseconds_total",
            "counter",
            self.queue_micros,
        );
        metric(
            &mut output,
            "chop_queue_duration_count",
            "counter",
            self.queue_observations,
        );
        metric(
            &mut output,
            "chop_prefix_cache_capture_microseconds_total",
            "counter",
            self.prefix_capture_micros,
        );
        metric(
            &mut output,
            "chop_prefix_cache_restore_microseconds_total",
            "counter",
            self.prefix_restore_micros,
        );
        metric(
            &mut output,
            "chop_sequences_active",
            "gauge",
            self.sequences_active,
        );
        metric(&mut output, "chop_batches_total", "counter", self.batches);
        metric(
            &mut output,
            "chop_batch_items_total",
            "counter",
            self.batch_items,
        );
        metric(
            &mut output,
            "chop_prefill_only_batches_total",
            "counter",
            self.prefill_only_batches,
        );
        metric(
            &mut output,
            "chop_decode_only_batches_total",
            "counter",
            self.decode_only_batches,
        );
        metric(
            &mut output,
            "chop_mixed_batches_total",
            "counter",
            self.mixed_batches,
        );
        metric(
            &mut output,
            "chop_prefill_batch_items_total",
            "counter",
            self.prefill_batch_items,
        );
        metric(
            &mut output,
            "chop_decode_batch_items_total",
            "counter",
            self.decode_batch_items,
        );
        metric(
            &mut output,
            "chop_ttft_microseconds_total",
            "counter",
            self.ttft_micros,
        );
        metric(
            &mut output,
            "chop_ttft_count",
            "counter",
            self.ttft_observations,
        );
        metric(
            &mut output,
            "chop_itl_microseconds_total",
            "counter",
            self.itl_micros,
        );
        metric(
            &mut output,
            "chop_itl_count",
            "counter",
            self.itl_observations,
        );
        metric(
            &mut output,
            "chop_execution_observations_total",
            "counter",
            self.execution_observations,
        );
        // An executor that measures nothing would otherwise publish four zeros
        // that read as an execution which launched no kernel and ran no
        // collective. Until one execution has measured itself, these names are
        // absent instead.
        if self.execution_observations > 0 {
            metric(
                &mut output,
                "chop_kernel_launches_total",
                "counter",
                self.kernel_launches,
            );
            metric(
                &mut output,
                "chop_kernel_duration_microseconds_total",
                "counter",
                self.kernel_micros,
            );
            metric(
                &mut output,
                "chop_collective_operations_total",
                "counter",
                self.collective_operations,
            );
            metric(
                &mut output,
                "chop_collective_duration_microseconds_total",
                "counter",
                self.collective_micros,
            );
        }
        metric(&mut output, "chop_errors_total", "counter", self.errors);
        metric(
            &mut output,
            "chop_cancellations_total",
            "counter",
            self.cancellations,
        );
        // An engine that never verified a draft publishes no speculation
        // counters, as with the kernel counters above.
        if self.spec_verify_items > 0 {
            for (name, value) in [
                ("chop_spec_verify_items_total", self.spec_verify_items),
                ("chop_spec_draft_tokens_total", self.spec_draft_tokens),
                ("chop_spec_accepted_tokens_total", self.spec_accepted_tokens),
                ("chop_spec_emitted_tokens_total", self.spec_emitted_tokens),
                ("chop_spec_rows_total", self.spec_rows),
            ] {
                metric(&mut output, name, "counter", value);
            }
            positions(
                &mut output,
                "chop_spec_drafted_position_total",
                &self.spec_drafted_positions,
            );
            positions(
                &mut output,
                "chop_spec_accepted_position_total",
                &self.spec_accepted_positions,
            );
        }
        // An engine that placed no media publishes no media counters.
        if self.media_encodes + self.media_cache_hits > 0 {
            for (name, value) in [
                ("chop_media_encodes_total", self.media_encodes),
                ("chop_media_encoded_rows_total", self.media_encoded_rows),
                ("chop_media_cache_hits_total", self.media_cache_hits),
                ("chop_media_evictions_total", self.media_evictions),
                ("chop_media_reencodes_total", self.media_reencodes),
            ] {
                metric(&mut output, name, "counter", value);
            }
        }
        output
    }
}

fn metric(output: &mut String, name: &str, kind: &str, value: u64) {
    writeln!(output, "# TYPE {name} {kind}\n{name} {value}")
        .expect("writing metrics to a string is infallible");
}

/// Writes one counter per draft position that has verified a draft.
/// Positions are numbered from 1, the first draft after the bonus input.
fn positions(output: &mut String, name: &str, values: &[u64; MAX_DRAFT_TOKENS]) {
    writeln!(output, "# TYPE {name} counter").expect("writing metrics to a string is infallible");
    for (position, value) in values.iter().enumerate() {
        writeln!(output, "{name}{{position=\"{}\"}} {value}", position + 1)
            .expect("writing metrics to a string is infallible");
    }
}

fn saturating_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::EngineMetrics;

    #[test]
    fn concurrent_snapshots_keep_request_lifecycle_counts_together() {
        let metrics = EngineMetrics::default();
        let done = Arc::new(AtomicBool::new(false));
        std::thread::scope(|scope| {
            let writer_metrics = metrics.clone();
            let writer_done = Arc::clone(&done);
            let writer = scope.spawn(move || {
                for _ in 0..10_000 {
                    writer_metrics.request_submitted(1);
                    std::thread::yield_now();
                    writer_metrics.cancellation();
                }
                writer_done.store(true, Ordering::Release);
            });
            while !done.load(Ordering::Acquire) {
                let snapshot = metrics.snapshot();
                assert_eq!(
                    snapshot.requests_submitted,
                    snapshot.requests_terminal + snapshot.requests_active
                );
                assert_eq!(snapshot.requests_terminal, snapshot.cancellations);
                assert_eq!(snapshot.requests_active, snapshot.sequences_active);
            }
            writer.join().unwrap();
        });
    }
}

const OFFLOAD_REASONS: [&str; 10] = [
    "tier_full",
    "capture",
    "out_refused",
    "in_refused",
    "out_failed",
    "in_failed",
    "restore",
    "unconfirmed",
    "unsupported",
    "staging_capacity",
];
const COPY_BUCKETS: [u64; 6] = [100, 1000, 10000, 100000, 1000000, u64::MAX];
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OffloadMetrics {
    pub copies: [u64; 2],
    pub fallbacks: [u64; 10],
    pub used_pages: u64,
    pub capacity: u64,
    pub swapped_requests: u64,
    pub oldest_pending_micros: u64,
    pub stalled_transfers: u64,
    pub stalls: u64,
    micros: [u64; 2],
    buckets: [[u64; 6]; 2],
}
impl EngineMetrics {
    pub(crate) fn offload_fallback(&self, reason: &str) {
        if let Some(index) = OFFLOAD_REASONS.iter().position(|r| *r == reason) {
            let mut metrics = self.inner.offload.lock().unwrap_or_else(|e| e.into_inner());
            if metrics.fallbacks[index] == 0 {
                eprintln!("KV offload fallback: {reason}");
            }
            metrics.fallbacks[index] += 1;
        }
    }
    pub(crate) fn offload_transfer_health(&self, oldest: u64, stalled: u64, new_stalls: u64) {
        let mut metrics = self.inner.offload.lock().unwrap_or_else(|e| e.into_inner());
        metrics.oldest_pending_micros = oldest;
        metrics.stalled_transfers = stalled;
        metrics.stalls += new_stalls;
    }
    pub(crate) fn offload_copy(&self, incoming: bool, micros: u64) {
        let mut metrics = self.inner.offload.lock().unwrap_or_else(|e| e.into_inner());
        let index = usize::from(incoming);
        metrics.copies[index] += 1;
        metrics.micros[index] += micros;
        for (bucket, bound) in COPY_BUCKETS.iter().enumerate() {
            if micros <= *bound {
                metrics.buckets[index][bucket] += 1;
            }
        }
    }
    pub(crate) fn offload_occupancy(&self, used: usize, capacity: usize, requests: usize) {
        let mut metrics = self.inner.offload.lock().unwrap_or_else(|e| e.into_inner());
        metrics.used_pages = used as u64;
        metrics.capacity = capacity as u64;
        metrics.swapped_requests = requests as u64;
    }
}
impl OffloadMetrics {
    fn render(&self, out: &mut String) {
        // Preserve disabled metrics and journals byte for byte.
        if self.capacity == 0 {
            return;
        }
        out.push_str("# TYPE chop_kv_offload_oldest_pending_seconds gauge\n");
        let _ = writeln!(
            out,
            "chop_kv_offload_oldest_pending_seconds {}",
            self.oldest_pending_micros as f64 / 1e6
        );
        metric(
            out,
            "chop_kv_offload_stalled_transfers",
            "gauge",
            self.stalled_transfers,
        );
        metric(out, "chop_kv_offload_stalls_total", "counter", self.stalls);
        for (index, direction) in ["out", "in"].iter().enumerate() {
            metric(
                out,
                &format!("chop_kv_offload_swap_{direction}_total"),
                "counter",
                self.copies[index],
            );
        }
        out.push_str("# TYPE chop_kv_offload_copy_seconds histogram\n");
        for (index, direction) in ["out", "in"].iter().enumerate() {
            for (bucket, bound) in COPY_BUCKETS.iter().enumerate() {
                let le = if *bound == u64::MAX {
                    "+Inf".to_owned()
                } else {
                    format!("{}", *bound as f64 / 1e6)
                };
                let _ = writeln!(
                    out,
                    "chop_kv_offload_copy_seconds_bucket{{direction=\"{direction}\",le=\"{le}\"}} {}",
                    self.buckets[index][bucket]
                );
            }
            let _ = writeln!(
                out,
                "chop_kv_offload_copy_seconds_sum{{direction=\"{direction}\"}} {}",
                self.micros[index] as f64 / 1e6
            );
            let _ = writeln!(
                out,
                "chop_kv_offload_copy_seconds_count{{direction=\"{direction}\"}} {}",
                self.copies[index]
            );
        }
        out.push_str("# TYPE chop_kv_offload_fallback_total counter\n");
        for (reason, count) in OFFLOAD_REASONS.iter().zip(self.fallbacks) {
            let _ = writeln!(
                out,
                "chop_kv_offload_fallback_total{{reason=\"{reason}\"}} {count}"
            );
        }
        out.push_str("# TYPE chop_kv_offload_host_pages gauge\n");
        let _ = writeln!(
            out,
            "chop_kv_offload_host_pages{{kind=\"used\"}} {}",
            self.used_pages
        );
        let _ = writeln!(
            out,
            "chop_kv_offload_host_pages{{kind=\"capacity\"}} {}",
            self.capacity
        );
        metric(
            out,
            "chop_kv_offload_swapped_requests",
            "gauge",
            self.swapped_requests,
        );
    }
}

#[cfg(test)]
mod offload_render_tests {
    use super::*;
    #[test]
    fn offload_families_are_typed_and_contiguous() {
        let mut text = String::new();
        OffloadMetrics::default().render(&mut text);
        assert!(text.is_empty());
        OffloadMetrics {
            capacity: 8,
            ..Default::default()
        }
        .render(&mut text);
        let mut seen = std::collections::BTreeSet::new();
        let mut family = "";
        for line in text.lines() {
            if let Some(header) = line.strip_prefix("# TYPE ") {
                family = header.split_whitespace().next().unwrap();
                assert!(seen.insert(family), "family repeated: {family}");
            } else {
                let name = line.split(['{', ' ']).next().unwrap();
                assert!(!family.is_empty());
                assert!(
                    name == family
                        || ["_bucket", "_sum", "_count"]
                            .iter()
                            .any(|suffix| name.strip_suffix(suffix) == Some(family)),
                    "ungrouped sample: {line}"
                );
            }
        }
        assert_eq!(seen.len(), 9);
    }
}
