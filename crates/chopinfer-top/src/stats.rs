//! Rates, ratios and latency quantiles from successive `/metrics` scrapes.
//!
//! Each poll keeps a compact [`Snapshot`] of the series chopinfer-top reads.
//! A ring of snapshots spans five minutes at the configured interval, so a
//! window's figures are the difference between the newest snapshot and the
//! one at the window's start. Counters reset when a server restarts; a new
//! startup nonce or a counter going backwards clears the ring.

use std::collections::VecDeque;

use crate::prometheus::{Exposition, Histogram};

/// The longest window, in seconds.
pub const HISTORY_SECONDS: f64 = 300.0;
/// Polls a throughput rate spans.
pub const RATE_POLLS: usize = 3;

/// The window latency quantiles and ratios cover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Window {
    Minute,
    FiveMinutes,
    SinceStart,
}

impl Window {
    pub fn seconds(self) -> f64 {
        match self {
            Self::Minute => 60.0,
            Self::FiveMinutes => HISTORY_SECONDS,
            Self::SinceStart => f64::INFINITY,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Minute => "60s",
            Self::FiveMinutes => "5m",
            Self::SinceStart => "start",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Minute => Self::FiveMinutes,
            Self::FiveMinutes => Self::SinceStart,
            Self::SinceStart => Self::Minute,
        }
    }
}

/// Counter values of one scrape. Missing series read as zero.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Counters {
    pub generation_tokens: f64,
    /// Prompt tokens computed by committed prefill-bearing steps.
    pub prefill_tokens: f64,
    pub prefix_hit_tokens: f64,
    pub prefix_query_tokens: f64,
    pub prefix_lookups: f64,
    pub prefix_lookup_hits: f64,
    pub prefix_evictions: f64,
    pub prefix_recomputed_seen_tokens: f64,
    pub prefix_capture_failures: f64,
    pub prefix_restore_failures: f64,
    pub spec_draft_tokens: f64,
    pub spec_accepted_tokens: f64,
    pub spec_draft_by_position: Vec<f64>,
    pub spec_accepted_by_position: Vec<f64>,
    /// Scheduled items by chosen K.
    pub speculation_k: Vec<(String, f64)>,
    /// Terminal sequences by `finished_reason`.
    pub finished: Vec<(String, f64)>,
    /// HTTP refusals by reason.
    pub refusals: Vec<(String, f64)>,
    pub cancellations: f64,
    pub errors: f64,
    pub preemptions: f64,
    pub batch_refusals: f64,
    pub batches: f64,
    pub prefill_only_batches: f64,
    pub decode_only_batches: f64,
    pub mixed_batches: f64,
    /// Generated tokens after each sequence's first, and the seconds
    /// between each sequence's consecutive outputs.
    pub decode_tokens: f64,
    pub decode_seconds: f64,
    /// Engine sequences submitted.
    pub requests_submitted: f64,
    /// Verify items that checked drafts, and the tokens they emitted.
    pub spec_verify_items: f64,
    pub spec_emitted_tokens: f64,
}

/// Gauge values of one scrape.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Gauges {
    pub running: Option<f64>,
    pub waiting: Option<f64>,
    /// Occupied KV blocks over capacity, from 0 to 1.
    pub kv_usage: Option<f64>,
    /// Tokens the KV page pool can hold: its pages times the tokens of a
    /// page (a context-parallel group's stripe).
    pub kv_capacity_tokens: Option<f64>,
    pub prefix_anchors: Option<f64>,
    pub prefix_blocks: Option<f64>,
    pub intake_used: Option<f64>,
    pub intake_budget: Option<f64>,
    pub transport_reserved: Option<f64>,
    pub transport_budget: Option<f64>,
    /// `starting`, `ready`, `failed` or `stopped`.
    pub worker: Option<&'static str>,
    pub poisoned: Option<bool>,
}

/// Latency histograms of one scrape.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Histograms {
    pub ttft: Option<Histogram>,
    pub inter_token: Option<Histogram>,
    pub end_to_end: Option<Histogram>,
    pub queue: Option<Histogram>,
    pub prefill: Option<Histogram>,
    pub per_output_token: Option<Histogram>,
    pub step_decode: Option<Histogram>,
    pub step_mixed: Option<Histogram>,
    pub step_prefill: Option<Histogram>,
    /// Full prompt and generated tokens per finished sequence.
    pub prompt_tokens: Option<Histogram>,
    pub output_tokens: Option<Histogram>,
    /// Prompt tokens per prefill-bearing step.
    pub step_prefill_tokens: Option<Histogram>,
    /// The prefill cap selected beside decoding requests.
    pub prefill_cap: Option<Histogram>,
    /// A TBT mixed step's execution time over its prediction.
    pub mixed_over_predicted: Option<Histogram>,
}

/// What one poll keeps of a scrape.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Snapshot {
    /// `chop_process_uptime_seconds`, when published.
    pub server_seconds: Option<f64>,
    /// Seconds on chopinfer-top's own monotonic clock.
    pub client_seconds: f64,
    pub counters: Counters,
    pub gauges: Gauges,
    pub histograms: Histograms,
}

/// Every series name chopinfer-top reads, for the contract test that
/// checks they are published or documented.
pub const SERIES: &[&str] = &[
    "chop_process_uptime_seconds",
    "vllm:generation_tokens_total",
    "chop_step_prefill_tokens",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "chop_prefix_cache_queries_total",
    "chop_prefix_cache_hits_total",
    "chop_prefix_cache_evictions_total",
    "chop_prefix_cache_recomputed_seen_tokens_total",
    "chop_prefix_cache_capture_failures_total",
    "chop_prefix_cache_restore_failures_total",
    "chop_prefix_cache_anchors",
    "chop_prefix_cache_blocks",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_draft_tokens_per_pos_total",
    "vllm:spec_decode_num_accepted_tokens_per_pos_total",
    "chop_speculation_k_total",
    "vllm:request_success_total",
    "chop_admission_refusals_total",
    "chop_cancellations_total",
    "chop_errors_total",
    "vllm:num_preemptions_total",
    "vllm:batch_refusals_total",
    "chop_batches_total",
    "chop_prefill_only_batches_total",
    "chop_decode_only_batches_total",
    "chop_mixed_batches_total",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "chop_kv_capacity_tokens",
    "chop_intake_used_bytes",
    "chop_intake_budget_bytes",
    "chop_transport_buffer_reserved_bytes",
    "chop_transport_buffer_budget_bytes",
    "chop_generation_worker_starting",
    "chop_generation_worker_ready",
    "chop_generation_worker_failed",
    "chop_generation_worker_stopped",
    "chop_instance_poisoned",
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_time_per_output_token_seconds",
    "chop_step_execution_seconds",
    "chop_request_decode_tokens_total",
    "chop_request_decode_seconds_total",
    "chop_requests_submitted_total",
    "chop_spec_verify_items_total",
    "chop_spec_emitted_tokens_total",
    "vllm:request_prompt_tokens",
    "vllm:request_generation_tokens",
    "chop_prefill_budget_selected_tokens",
    "chop_mixed_step_actual_over_predicted",
];

impl Snapshot {
    pub fn from_exposition(scrape: &Exposition, client_seconds: f64) -> Self {
        let counter = |name: &str| scrape.value(name, &[]).unwrap_or(0.0);
        let gauge = |name: &str| scrape.value(name, &[]);
        let positions = |name: &str| {
            let mut values = scrape
                .by_label(name, "position")
                .into_iter()
                .filter_map(|(position, value)| Some((position.parse::<usize>().ok()?, value)))
                .collect::<Vec<_>>();
            values.sort_by_key(|(position, _)| *position);
            values.into_iter().map(|(_, value)| value).collect()
        };
        let worker = ["starting", "ready", "failed", "stopped"]
            .into_iter()
            .find(|state| gauge(&format!("chop_generation_worker_{state}")) == Some(1.0));
        let step =
            |phase: &str| scrape.histogram("chop_step_execution_seconds", &[("phase", phase)]);
        Self {
            server_seconds: gauge("chop_process_uptime_seconds"),
            client_seconds,
            counters: Counters {
                generation_tokens: counter("vllm:generation_tokens_total"),
                prefill_tokens: counter("chop_step_prefill_tokens_sum"),
                prefix_hit_tokens: counter("vllm:prefix_cache_hits_total"),
                prefix_query_tokens: counter("vllm:prefix_cache_queries_total"),
                prefix_lookups: counter("chop_prefix_cache_queries_total"),
                prefix_lookup_hits: counter("chop_prefix_cache_hits_total"),
                prefix_evictions: counter("chop_prefix_cache_evictions_total"),
                prefix_recomputed_seen_tokens: counter(
                    "chop_prefix_cache_recomputed_seen_tokens_total",
                ),
                prefix_capture_failures: counter("chop_prefix_cache_capture_failures_total"),
                prefix_restore_failures: counter("chop_prefix_cache_restore_failures_total"),
                spec_draft_tokens: counter("vllm:spec_decode_num_draft_tokens_total"),
                spec_accepted_tokens: counter("vllm:spec_decode_num_accepted_tokens_total"),
                spec_draft_by_position: positions(
                    "vllm:spec_decode_num_draft_tokens_per_pos_total",
                ),
                spec_accepted_by_position: positions(
                    "vllm:spec_decode_num_accepted_tokens_per_pos_total",
                ),
                speculation_k: scrape.by_label("chop_speculation_k_total", "k"),
                finished: scrape.by_label("vllm:request_success_total", "finished_reason"),
                refusals: scrape.by_label("chop_admission_refusals_total", "reason"),
                cancellations: counter("chop_cancellations_total"),
                errors: counter("chop_errors_total"),
                preemptions: counter("vllm:num_preemptions_total"),
                batch_refusals: counter("vllm:batch_refusals_total"),
                batches: counter("chop_batches_total"),
                prefill_only_batches: counter("chop_prefill_only_batches_total"),
                decode_only_batches: counter("chop_decode_only_batches_total"),
                mixed_batches: counter("chop_mixed_batches_total"),
                decode_tokens: counter("chop_request_decode_tokens_total"),
                decode_seconds: counter("chop_request_decode_seconds_total"),
                requests_submitted: counter("chop_requests_submitted_total"),
                spec_verify_items: counter("chop_spec_verify_items_total"),
                spec_emitted_tokens: counter("chop_spec_emitted_tokens_total"),
            },
            gauges: Gauges {
                running: gauge("vllm:num_requests_running"),
                waiting: gauge("vllm:num_requests_waiting"),
                kv_usage: gauge("vllm:kv_cache_usage_perc"),
                kv_capacity_tokens: gauge("chop_kv_capacity_tokens"),
                prefix_anchors: gauge("chop_prefix_cache_anchors"),
                prefix_blocks: gauge("chop_prefix_cache_blocks"),
                intake_used: gauge("chop_intake_used_bytes"),
                intake_budget: gauge("chop_intake_budget_bytes"),
                transport_reserved: gauge("chop_transport_buffer_reserved_bytes"),
                transport_budget: gauge("chop_transport_buffer_budget_bytes"),
                worker,
                poisoned: gauge("chop_instance_poisoned").map(|value| value >= 1.0),
            },
            histograms: Histograms {
                ttft: scrape.histogram("vllm:time_to_first_token_seconds", &[]),
                inter_token: scrape.histogram("vllm:inter_token_latency_seconds", &[]),
                end_to_end: scrape.histogram("vllm:e2e_request_latency_seconds", &[]),
                queue: scrape.histogram("vllm:request_queue_time_seconds", &[]),
                prefill: scrape.histogram("vllm:request_prefill_time_seconds", &[]),
                per_output_token: scrape
                    .histogram("vllm:request_time_per_output_token_seconds", &[]),
                step_decode: step("decode"),
                step_mixed: step("mixed"),
                step_prefill: step("prefill"),
                prompt_tokens: scrape.histogram("vllm:request_prompt_tokens", &[]),
                output_tokens: scrape.histogram("vllm:request_generation_tokens", &[]),
                step_prefill_tokens: scrape.histogram("chop_step_prefill_tokens", &[]),
                prefill_cap: scrape.histogram("chop_prefill_budget_selected_tokens", &[]),
                mixed_over_predicted: scrape
                    .histogram("chop_mixed_step_actual_over_predicted", &[]),
            },
        }
    }

    /// Counters that only grow while one server process runs.
    fn monotonic(&self) -> [f64; 5] {
        let counters = &self.counters;
        [
            counters.generation_tokens,
            counters.prefill_tokens,
            counters.batches,
            counters.finished.iter().map(|(_, value)| value).sum(),
            self.server_seconds.unwrap_or(0.0),
        ]
    }
}

/// A selector of one value of a snapshot.
pub type Value = fn(&Snapshot) -> f64;
/// A selector of one histogram of a snapshot.
pub type HistogramOf = fn(&Snapshot) -> Option<&Histogram>;

/// The span a window's figures cover.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coverage {
    /// Seconds of history behind the figures.
    pub seconds: f64,
    /// Whether that history spans the whole window.
    pub complete: bool,
}

/// Engine wall time over a window: the shares spent executing decode-only,
/// mixed and prefill-only steps, and the rest, outside step execution
/// (scheduling and host work between steps, or no work).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepTime {
    pub decode: f64,
    pub mixed: f64,
    pub prefill: f64,
    pub outside: f64,
}

/// Decode tokens per second of decoding, per sequence, between two
/// snapshots; `None` when no decode time accrued between them.
fn stream_between(before: &Snapshot, after: &Snapshot) -> Option<f64> {
    let seconds = after.counters.decode_seconds - before.counters.decode_seconds;
    (seconds > 0.0)
        .then(|| (after.counters.decode_tokens - before.counters.decode_tokens).max(0.0) / seconds)
}

/// One instance's recent snapshots.
#[derive(Clone, Debug)]
pub struct Stats {
    ring: VecDeque<Snapshot>,
    capacity: usize,
    nonce: Option<String>,
    /// Polls dropped from the front of the ring since the history began.
    dropped: u64,
}

impl Stats {
    /// A ring holding five minutes of polls every `interval` seconds, plus
    /// one baseline: 301 at 1 s, 601 at 0.5 s.
    pub fn new(interval: f64) -> Self {
        let capacity = (HISTORY_SECONDS / interval.max(0.1)).ceil() as usize + 1;
        Self {
            ring: VecDeque::with_capacity(capacity),
            capacity,
            nonce: None,
            dropped: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.ring.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    pub fn nonce(&self) -> Option<&str> {
        self.nonce.as_deref()
    }

    /// The number of the oldest poll held, counting from 0 at the first
    /// poll of the current history; so also the number of the first
    /// [`Stats::rate_series`] value.
    pub fn oldest_poll(&self) -> u64 {
        self.dropped
    }

    /// Adds a poll. Returns true when it is from a restarted server: a new
    /// startup nonce, or a counter or the server clock went backwards. The
    /// earlier history is then dropped.
    pub fn push(&mut self, nonce: Option<&str>, snapshot: Snapshot) -> bool {
        let new_nonce = match (&self.nonce, nonce) {
            (Some(known), Some(seen)) => known != seen,
            _ => false,
        };
        let backwards = self.ring.back().is_some_and(|last| {
            last.monotonic()
                .iter()
                .zip(snapshot.monotonic())
                .any(|(before, now)| now < *before)
        });
        let restarted = new_nonce || backwards;
        if restarted {
            self.ring.clear();
            self.dropped = 0;
        }
        if nonce.is_some() {
            self.nonce = nonce.map(str::to_owned);
        }
        if self.ring.len() == self.capacity {
            self.ring.pop_front();
            self.dropped += 1;
        }
        self.ring.push_back(snapshot);
        restarted
    }

    pub fn latest(&self) -> Option<&Snapshot> {
        self.ring.back()
    }

    /// Whether every snapshot has the server clock, so it is the clock used.
    fn server_clock(&self) -> bool {
        self.ring
            .iter()
            .all(|snapshot| snapshot.server_seconds.is_some())
    }

    fn time(&self, snapshot: &Snapshot) -> f64 {
        if self.server_clock() {
            snapshot.server_seconds.unwrap_or(snapshot.client_seconds)
        } else {
            snapshot.client_seconds
        }
    }

    /// The per-second rate of `value` over the last three polls.
    pub fn rate(&self, value: Value) -> Option<f64> {
        let latest = self.ring.back()?;
        let earlier = self
            .ring
            .get(self.ring.len().saturating_sub(RATE_POLLS + 1))?;
        let seconds = self.time(latest) - self.time(earlier);
        (seconds > 0.0).then(|| ((value(latest) - value(earlier)) / seconds).max(0.0))
    }

    /// The snapshot a window starts from, `None` meaning the server's start,
    /// and how much history that covers. `None` overall without data.
    fn baseline(&self, window: Window) -> Option<(Option<&Snapshot>, Coverage)> {
        let latest = self.ring.back()?;
        let now = self.time(latest);
        let since_start = self.server_clock() && now <= window.seconds();
        if window == Window::SinceStart || since_start {
            let seconds = latest.server_seconds.unwrap_or(f64::INFINITY);
            return Some((
                None,
                Coverage {
                    seconds,
                    complete: window == Window::SinceStart,
                },
            ));
        }
        let start = now - window.seconds();
        if let Some(base) = self
            .ring
            .iter()
            .rev()
            .find(|snapshot| self.time(snapshot) <= start)
        {
            return Some((
                Some(base),
                Coverage {
                    seconds: now - self.time(base),
                    complete: true,
                },
            ));
        }
        let oldest = self.ring.front()?;
        (self.ring.len() > 1).then(|| {
            (
                Some(oldest),
                Coverage {
                    seconds: now - self.time(oldest),
                    complete: false,
                },
            )
        })
    }

    pub fn coverage(&self, window: Window) -> Option<Coverage> {
        self.baseline(window).map(|(_, coverage)| coverage)
    }

    /// How much `value` grew over `window`.
    pub fn delta(&self, window: Window, value: Value) -> Option<f64> {
        let latest = self.ring.back()?;
        let (base, _) = self.baseline(window)?;
        Some((value(latest) - base.map_or(0.0, value)).max(0.0))
    }

    /// `numerator` growth over `denominator` growth across `window`; `None`
    /// when the denominator did not grow.
    pub fn ratio(&self, window: Window, numerator: Value, denominator: Value) -> Option<f64> {
        let denominator = self.delta(window, denominator)?;
        (denominator > 0.0)
            .then(|| (self.delta(window, numerator).unwrap_or(0.0) / denominator).min(1.0))
    }

    /// The observations of a histogram within `window`.
    pub fn histogram(&self, window: Window, histogram: HistogramOf) -> Option<Histogram> {
        let latest = histogram(self.ring.back()?)?;
        match self.baseline(window)? {
            (None, _) => Some(latest.clone()),
            (Some(base), _) => match histogram(base) {
                Some(earlier) => latest.since(earlier),
                None => Some(latest.clone()),
            },
        }
    }

    /// The `q` quantile of a histogram within `window`.
    pub fn quantile(&self, window: Window, histogram: HistogramOf, q: f64) -> Option<f64> {
        self.histogram(window, histogram)?.quantile(q)
    }

    /// The per-second rate of `value` between each pair of polls.
    pub fn rate_series(&self, value: Value) -> Vec<f64> {
        self.ring
            .iter()
            .zip(self.ring.iter().skip(1))
            .map(|(before, after)| {
                let seconds = self.time(after) - self.time(before);
                if seconds > 0.0 {
                    ((value(after) - value(before)) / seconds).max(0.0)
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// The decode speed of the average sequence over the last three polls:
    /// decode tokens over the time between each sequence's outputs.
    pub fn stream_rate(&self) -> Option<f64> {
        let latest = self.ring.back()?;
        let earlier = self
            .ring
            .get(self.ring.len().saturating_sub(RATE_POLLS + 1))?;
        stream_between(earlier, latest)
    }

    /// The same across every poll held.
    pub fn stream_average(&self) -> Option<f64> {
        stream_between(self.ring.front()?, self.ring.back()?)
    }

    /// The same between each pair of polls.
    pub fn stream_series(&self) -> Vec<Option<f64>> {
        self.ring
            .iter()
            .zip(self.ring.iter().skip(1))
            .map(|(before, after)| stream_between(before, after))
            .collect()
    }

    /// How the engine's wall time over `window` divides between executing
    /// each kind of step and the rest; `None` without step time in it.
    pub fn step_time(&self, window: Window) -> Option<StepTime> {
        let seconds = self.coverage(window)?.seconds;
        if !(seconds.is_finite() && seconds > 0.0) {
            return None;
        }
        let spent = |histogram: HistogramOf| {
            self.histogram(window, histogram)
                .map_or(0.0, |histogram| histogram.sum)
        };
        let decode = spent(|s| s.histograms.step_decode.as_ref());
        let mixed = spent(|s| s.histograms.step_mixed.as_ref());
        let prefill = spent(|s| s.histograms.step_prefill.as_ref());
        let busy = decode + mixed + prefill;
        if busy <= 0.0 {
            return None;
        }
        // Steps timed at their edges can sum past the window; they then
        // fill it.
        let whole = busy.max(seconds);
        Some(StepTime {
            decode: decode / whole,
            mixed: mixed / whole,
            prefill: prefill / whole,
            outside: (seconds - busy).max(0.0) / whole,
        })
    }

    /// `value` at each poll.
    pub fn series(&self, value: fn(&Snapshot) -> Option<f64>) -> Vec<f64> {
        self.ring
            .iter()
            .map(|snapshot| value(snapshot).unwrap_or(0.0))
            .collect()
    }

    /// The share of each chosen K over `window`.
    pub fn speculation_k(&self, window: Window) -> Vec<(String, f64)> {
        let Some(latest) = self.ring.back() else {
            return Vec::new();
        };
        let base = self.baseline(window).and_then(|(base, _)| base);
        let grown = latest
            .counters
            .speculation_k
            .iter()
            .map(|(k, value)| {
                let before = base
                    .and_then(|base| {
                        base.counters
                            .speculation_k
                            .iter()
                            .find(|(seen, _)| seen == k)
                    })
                    .map_or(0.0, |(_, value)| *value);
                (k.clone(), (value - before).max(0.0))
            })
            .collect::<Vec<_>>();
        let total: f64 = grown.iter().map(|(_, value)| value).sum();
        if total <= 0.0 {
            return Vec::new();
        }
        grown
            .into_iter()
            .filter(|(_, value)| *value > 0.0)
            .map(|(k, value)| (k, value / total))
            .collect()
    }

    /// Draft acceptance at each position over `window`.
    pub fn acceptance_by_position(&self, window: Window) -> Vec<Option<f64>> {
        let Some(latest) = self.ring.back() else {
            return Vec::new();
        };
        let base = self.baseline(window).and_then(|(base, _)| base);
        let at = |values: &[f64], index: usize| values.get(index).copied().unwrap_or(0.0);
        (0..latest.counters.spec_draft_by_position.len())
            .map(|index| {
                let drafted = at(&latest.counters.spec_draft_by_position, index)
                    - base.map_or(0.0, |base| at(&base.counters.spec_draft_by_position, index));
                let accepted = at(&latest.counters.spec_accepted_by_position, index)
                    - base.map_or(0.0, |base| {
                        at(&base.counters.spec_accepted_by_position, index)
                    });
                (drafted > 0.0).then(|| (accepted / drafted).clamp(0.0, 1.0))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(seconds: f64, generated: f64) -> Snapshot {
        Snapshot {
            server_seconds: Some(seconds),
            client_seconds: seconds,
            counters: Counters {
                generation_tokens: generated,
                ..Counters::default()
            },
            ..Snapshot::default()
        }
    }

    fn histogram(fast: f64, total: f64) -> Histogram {
        Histogram {
            buckets: vec![(0.1, fast), (1.0, total), (f64::INFINITY, total)],
            sum: 0.0,
            count: total,
        }
    }

    #[test]
    fn rates_span_the_last_three_polls() {
        let mut stats = Stats::new(1.0);
        assert_eq!(stats.rate(|s| s.counters.generation_tokens), None);
        for (second, generated) in [(10.0, 0.0), (11.0, 100.0), (12.0, 200.0), (13.0, 600.0)] {
            assert!(!stats.push(Some("a"), snapshot(second, generated)));
        }
        assert_eq!(stats.rate(|s| s.counters.generation_tokens), Some(200.0));
        assert_eq!(
            stats.rate_series(|s| s.counters.generation_tokens),
            [100.0, 100.0, 400.0]
        );
    }

    #[test]
    fn a_new_nonce_or_a_counter_going_backwards_is_a_restart() {
        let mut stats = Stats::new(1.0);
        stats.push(Some("a"), snapshot(100.0, 500.0));
        stats.push(Some("a"), snapshot(101.0, 600.0));
        assert!(stats.push(Some("b"), snapshot(102.0, 700.0)));
        assert_eq!((stats.len(), stats.nonce()), (1, Some("b")));
        stats.push(Some("b"), snapshot(103.0, 800.0));
        assert!(stats.push(Some("b"), snapshot(104.0, 10.0)));
        assert_eq!(stats.len(), 1);
        // The server clock going back is a restart too.
        stats.push(Some("b"), snapshot(105.0, 20.0));
        assert!(stats.push(Some("b"), snapshot(2.0, 30.0)));
    }

    #[test]
    fn polls_are_numbered_from_the_start_of_the_history() {
        let mut stats = Stats::new(1.0);
        assert_eq!(stats.oldest_poll(), 0);
        for poll in 0..stats.capacity() + 5 {
            stats.push(Some("a"), snapshot(poll as f64, poll as f64));
        }
        assert_eq!(stats.oldest_poll(), 5);
        // A restart starts a new history.
        stats.push(Some("b"), snapshot(1.0, 1.0));
        assert_eq!(stats.oldest_poll(), 0);
    }

    #[test]
    fn the_ring_holds_five_minutes_at_every_interval() {
        for interval in [0.5, 1.0] {
            let mut stats = Stats::new(interval);
            assert_eq!(stats.capacity(), (300.0 / interval) as usize + 1);
            let polls = (360.0 / interval) as usize;
            for poll in 0..=polls {
                let second = 1000.0 + poll as f64 * interval;
                stats.push(Some("a"), snapshot(second, poll as f64 * 10.0));
            }
            assert_eq!(stats.len(), stats.capacity());
            let five = stats.coverage(Window::FiveMinutes).unwrap();
            assert!(five.complete, "{interval}");
            assert_eq!(five.seconds, 300.0);
            let generated = stats
                .delta(Window::FiveMinutes, |s| s.counters.generation_tokens)
                .unwrap();
            assert_eq!(generated, 300.0 / interval * 10.0);
        }
    }

    #[test]
    fn short_history_is_marked_incomplete() {
        // A server up 30 s: the minute window is its whole life.
        let mut stats = Stats::new(1.0);
        stats.push(Some("a"), snapshot(29.0, 10.0));
        stats.push(Some("a"), snapshot(30.0, 20.0));
        let coverage = stats.coverage(Window::Minute).unwrap();
        assert_eq!(
            coverage,
            Coverage {
                seconds: 30.0,
                complete: false
            }
        );
        assert_eq!(
            stats.delta(Window::Minute, |s| s.counters.generation_tokens),
            Some(20.0)
        );
        // A server up for hours, watched for 20 s.
        let mut stats = Stats::new(1.0);
        stats.push(Some("a"), snapshot(10_000.0, 10.0));
        assert_eq!(stats.coverage(Window::Minute), None);
        stats.push(Some("a"), snapshot(10_020.0, 50.0));
        let coverage = stats.coverage(Window::Minute).unwrap();
        assert_eq!(
            coverage,
            Coverage {
                seconds: 20.0,
                complete: false
            }
        );
        assert_eq!(
            stats.delta(Window::Minute, |s| s.counters.generation_tokens),
            Some(40.0)
        );
        assert!(stats.coverage(Window::SinceStart).unwrap().complete);
    }

    #[test]
    fn window_quantiles_use_bucket_deltas() {
        let mut stats = Stats::new(1.0);
        for second in 0..=120 {
            let mut snapshot = snapshot(1000.0 + f64::from(second), 0.0);
            // Fast requests for the first minute, slow ones after it.
            let fast = f64::from(second.min(60)) * 10.0;
            let total = f64::from(second) * 10.0;
            snapshot.histograms.ttft = Some(histogram(fast, total));
            stats.push(Some("a"), snapshot);
        }
        let ttft: HistogramOf = |s| s.histograms.ttft.as_ref();
        let minute = stats.quantile(Window::Minute, ttft, 0.5).unwrap();
        assert!(minute > 0.1, "{minute}");
        let start = stats.quantile(Window::SinceStart, ttft, 0.5).unwrap();
        assert!((start - 0.1).abs() < 1e-9, "{start}");
        let mut idle = Stats::new(1.0);
        for second in 0..3 {
            let mut snapshot = snapshot(1000.0 + f64::from(second), 0.0);
            snapshot.histograms.ttft = Some(histogram(5.0, 5.0));
            idle.push(Some("a"), snapshot);
        }
        assert_eq!(idle.quantile(Window::Minute, ttft, 0.5), None);
    }

    #[test]
    fn ratios_and_speculation_shares_cover_the_window() {
        let mut stats = Stats::new(1.0);
        for second in 0..=70 {
            let mut snapshot = snapshot(1000.0 + f64::from(second), 0.0);
            let counters = &mut snapshot.counters;
            counters.spec_draft_tokens = f64::from(second) * 10.0;
            counters.spec_accepted_tokens = f64::from(second) * 6.0;
            counters.spec_draft_by_position =
                vec![f64::from(second) * 5.0, f64::from(second) * 5.0];
            counters.spec_accepted_by_position =
                vec![f64::from(second) * 4.0, f64::from(second) * 2.0];
            counters.speculation_k = vec![
                ("0".into(), f64::from(second)),
                ("3".into(), f64::from(second) * 3.0),
            ];
            stats.push(Some("a"), snapshot);
        }
        let acceptance = stats
            .ratio(
                Window::Minute,
                |s| s.counters.spec_accepted_tokens,
                |s| s.counters.spec_draft_tokens,
            )
            .unwrap();
        assert!((acceptance - 0.6).abs() < 1e-12);
        assert_eq!(
            stats.acceptance_by_position(Window::Minute),
            [Some(0.8), Some(0.4)]
        );
        assert_eq!(
            stats.speculation_k(Window::Minute),
            [("0".to_owned(), 0.25), ("3".to_owned(), 0.75)]
        );
        assert_eq!(
            stats.ratio(
                Window::Minute,
                |s| s.counters.prefix_hit_tokens,
                |s| s.counters.prefix_query_tokens
            ),
            None
        );
    }

    /// A poll at `second` whose decode counters stand at `tokens` and
    /// `seconds`.
    fn decoding(second: f64, tokens: f64, seconds: f64) -> Snapshot {
        let mut snapshot = snapshot(second, tokens);
        snapshot.counters.decode_tokens = tokens;
        snapshot.counters.decode_seconds = seconds;
        snapshot
    }

    #[test]
    fn stream_speed_is_decode_tokens_over_decode_time() {
        let mut stats = Stats::new(1.0);
        assert_eq!(stats.stream_rate(), None);
        // Ten sequences at 8 tok/s each: 80 tokens over 10 decode-seconds a
        // poll; then one finishes and nine run at 6 tok/s.
        let polls = [
            (0.0, 0.0),
            (80.0, 10.0),
            (160.0, 20.0),
            (240.0, 30.0),
            (294.0, 39.0),
            (348.0, 48.0),
            (402.0, 57.0),
        ];
        for (second, (tokens, seconds)) in polls.into_iter().enumerate() {
            stats.push(Some("a"), decoding(second as f64, tokens, seconds));
        }
        assert_eq!(stats.stream_rate(), Some(6.0));
        assert_eq!(stats.stream_average(), Some(402.0 / 57.0));
        let series = stats.stream_series();
        assert_eq!(series.len(), 6);
        assert_eq!(series[0], Some(8.0));
        assert_eq!(series[5], Some(6.0));
    }

    #[test]
    fn a_request_between_polls_counts_and_no_decoding_is_no_figure() {
        let mut stats = Stats::new(1.0);
        stats.push(Some("a"), decoding(0.0, 0.0, 0.0));
        // A request that decoded 20 tokens in 0.5 s and finished between
        // two polls still counts.
        stats.push(Some("a"), decoding(1.0, 20.0, 0.5));
        // Nothing decodes for a poll.
        stats.push(Some("a"), decoding(2.0, 20.0, 0.5));
        assert_eq!(stats.stream_series(), [Some(40.0), None]);
        assert_eq!(stats.stream_rate(), Some(40.0));
        // A server without the counters has no figure.
        let mut old = Stats::new(1.0);
        for second in 0..5 {
            old.push(
                Some("a"),
                snapshot(f64::from(second), f64::from(second) * 100.0),
            );
        }
        assert_eq!(old.stream_rate(), None);
        assert_eq!(old.stream_average(), None);
        assert!(old.stream_series().iter().all(Option::is_none));
    }

    /// A step histogram whose `count` steps took `seconds` in all.
    fn steps(count: f64, seconds: f64) -> Option<Histogram> {
        Some(Histogram {
            buckets: vec![(1.0, count), (f64::INFINITY, count)],
            sum: seconds,
            count,
        })
    }

    #[test]
    fn step_time_divides_the_window_by_step_kind() {
        let mut stats = Stats::new(1.0);
        assert_eq!(stats.step_time(Window::Minute), None);
        for second in 0..=120 {
            let t = f64::from(second);
            let mut snapshot = snapshot(1000.0 + t, 0.0);
            // Each second: 0.25 s of decode steps and 0.3 s of mixed ones.
            snapshot.histograms.step_decode = steps(5.0 * t, 0.25 * t);
            snapshot.histograms.step_mixed = steps(t, 0.3 * t);
            stats.push(Some("a"), snapshot);
        }
        let time = stats.step_time(Window::Minute).unwrap();
        let close = |left: f64, right: f64| (left - right).abs() < 1e-9;
        assert!(close(time.decode, 0.25), "{time:?}");
        assert!(close(time.mixed, 0.3), "{time:?}");
        assert_eq!(time.prefill, 0.0);
        assert!(close(time.outside, 0.45), "{time:?}");
        // Steps that add up past the window fill it.
        let mut over = Stats::new(1.0);
        for second in 0..=70 {
            let t = f64::from(second);
            let mut snapshot = snapshot(1000.0 + t, 0.0);
            snapshot.histograms.step_decode = steps(t, 0.9 * t);
            snapshot.histograms.step_prefill = steps(t, 0.3 * t);
            over.push(Some("a"), snapshot);
        }
        let time = over.step_time(Window::Minute).unwrap();
        assert!(close(time.decode, 0.75), "{time:?}");
        assert!(close(time.prefill, 0.25), "{time:?}");
        assert_eq!(time.outside, 0.0);
        // A window without step time has no split.
        let mut idle = Stats::new(1.0);
        for second in 0..3 {
            let mut snapshot = snapshot(1000.0 + f64::from(second), 0.0);
            snapshot.histograms.step_decode = steps(4.0, 0.2);
            idle.push(Some("a"), snapshot);
        }
        assert_eq!(idle.step_time(Window::Minute), None);
    }

    #[test]
    fn snapshots_read_the_published_series() {
        let scrape = Exposition::parse(
            "chop_process_uptime_seconds 5.5\n\
             vllm:generation_tokens_total{model_name=\"m\",engine=\"0\"} 42\n\
             chop_step_prefill_tokens_sum{model_name=\"m\",engine=\"0\"} 900\n\
             vllm:spec_decode_num_draft_tokens_per_pos_total{model_name=\"m\",engine=\"0\",position=\"1\"} 4\n\
             vllm:spec_decode_num_draft_tokens_per_pos_total{model_name=\"m\",engine=\"0\",position=\"0\"} 8\n\
             chop_generation_worker_starting 0\nchop_generation_worker_ready 1\n\
             chop_instance_poisoned{model_name=\"m\",engine=\"0\"} 0\n\
             vllm:kv_cache_usage_perc{model_name=\"m\",engine=\"0\"} 0.25\n\
             chop_step_execution_seconds_bucket{model_name=\"m\",engine=\"0\",phase=\"mixed\",le=\"+Inf\"} 3\n\
             chop_request_decode_tokens_total{model_name=\"m\",engine=\"0\"} 39\n\
             chop_request_decode_seconds_total{model_name=\"m\",engine=\"0\"} 4.875\n\
             chop_requests_submitted_total 7\nchop_kv_capacity_tokens 7340032\n\
             chop_spec_verify_items_total 10\nchop_spec_emitted_tokens_total 19\n\
             vllm:request_prompt_tokens_bucket{model_name=\"m\",engine=\"0\",le=\"+Inf\"} 2\n\
             vllm:request_prompt_tokens_sum{model_name=\"m\",engine=\"0\"} 3000\n\
             chop_mixed_step_actual_over_predicted_bucket{model_name=\"m\",engine=\"0\",le=\"1.0\"} 4\n",
        );
        let snapshot = Snapshot::from_exposition(&scrape, 1.0);
        assert_eq!(snapshot.server_seconds, Some(5.5));
        assert_eq!(snapshot.counters.generation_tokens, 42.0);
        assert_eq!(snapshot.counters.prefill_tokens, 900.0);
        assert_eq!(snapshot.counters.spec_draft_by_position, [8.0, 4.0]);
        assert_eq!(snapshot.gauges.worker, Some("ready"));
        assert_eq!(snapshot.gauges.poisoned, Some(false));
        assert_eq!(snapshot.gauges.kv_usage, Some(0.25));
        assert_eq!(snapshot.gauges.running, None);
        assert!(snapshot.histograms.step_mixed.is_some());
        assert!(snapshot.histograms.step_decode.is_none());
        assert_eq!(snapshot.counters.decode_tokens, 39.0);
        assert_eq!(snapshot.counters.decode_seconds, 4.875);
        assert_eq!(snapshot.counters.requests_submitted, 7.0);
        assert_eq!(snapshot.gauges.kv_capacity_tokens, Some(7_340_032.0));
        assert_eq!(snapshot.counters.spec_verify_items, 10.0);
        assert_eq!(snapshot.counters.spec_emitted_tokens, 19.0);
        assert_eq!(snapshot.histograms.prompt_tokens.unwrap().sum, 3000.0);
        assert!(snapshot.histograms.mixed_over_predicted.is_some());
        assert!(snapshot.histograms.prefill_cap.is_none());
    }
}
