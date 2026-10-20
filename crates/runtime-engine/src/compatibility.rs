//! Reference metric names backed by engine lifecycle observations.
use std::collections::BTreeMap;
use std::fmt::Write;
use std::time::Instant;

use crate::{FinishReason, RequestId};

const TTFT: &[u64] = &[
    1_000,
    5_000,
    10_000,
    20_000,
    40_000,
    60_000,
    80_000,
    100_000,
    250_000,
    500_000,
    750_000,
    1_000_000,
    2_500_000,
    5_000_000,
    7_500_000,
    10_000_000,
    20_000_000,
    40_000_000,
    80_000_000,
    160_000_000,
    640_000_000,
    2_560_000_000,
];
const ITL: &[u64] = &[
    10_000, 25_000, 50_000, 75_000, 100_000, 150_000, 200_000, 300_000, 400_000, 500_000, 750_000,
    1_000_000, 2_500_000, 5_000_000, 7_500_000, 10_000_000, 20_000_000, 40_000_000, 80_000_000,
];
const E2E: &[u64] = &[
    300_000,
    500_000,
    800_000,
    1_000_000,
    1_500_000,
    2_000_000,
    2_500_000,
    5_000_000,
    10_000_000,
    15_000_000,
    20_000_000,
    30_000_000,
    40_000_000,
    50_000_000,
    60_000_000,
    120_000_000,
    240_000_000,
    480_000_000,
    960_000_000,
    1_920_000_000,
    7_680_000_000,
];

const ITERATION: &[u64] = &[
    1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
];
const PREEMPTIONS: &[u64] = &[1, 2, 3, 4, 5, 10, 20];
const STEP: &[u64] = &[
    5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 40_000, 50_000, 60_000, 75_000, 100_000,
    125_000, 150_000, 175_000, 200_000, 250_000, 300_000, 400_000, 500_000, 750_000, 1_000_000,
    2_500_000,
];
const STEP_PREFILL_TOKENS: &[u64] = &[
    64, 128, 256, 512, 1024, 1536, 2048, 2560, 3072, 4096, 6144, 8192, 16384,
];
const PREFILL_CAP: &[u64] = &[
    128, 256, 384, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656,
    7168, 7680, 8192,
];
/// Actual over predicted mixed-step time, in thousandths.
const PREDICTION_RATIO: &[u64] = &[500, 750, 900, 1000, 1100, 1250, 1500, 2000, 3000, 5000];
/// Step execution phases, by whether the batch had prefill and decode items.
const STEP_PHASES: [&str; 3] = ["decode", "mixed", "prefill"];

#[derive(Debug)]
pub(crate) struct Histogram {
    bounds: Vec<u64>,
    buckets: Vec<u64>,
    count: u64,
    micros: u64,
}

impl Histogram {
    fn new(bounds: &[u64]) -> Self {
        Self {
            bounds: bounds.to_vec(),
            buckets: vec![0; bounds.len()],
            count: 0,
            micros: 0,
        }
    }

    pub(crate) fn observe(&mut self, micros: u64) {
        self.count = self.count.saturating_add(1);
        self.micros = self.micros.saturating_add(micros);
        for (bound, count) in self.bounds.iter().zip(&mut self.buckets) {
            if micros <= *bound {
                *count = count.saturating_add(1);
            }
        }
    }

    fn render(&self, output: &mut String, name: &str, labels: &str) {
        self.render_scaled(output, name, labels, 1_000_000.0);
    }

    fn render_scaled(&self, output: &mut String, name: &str, labels: &str, scale: f64) {
        writeln!(output, "# TYPE {name} histogram").unwrap();
        self.samples(output, name, labels, scale);
    }

    /// Writes the series without a TYPE line, for one of several label sets.
    fn samples(&self, output: &mut String, name: &str, labels: &str, scale: f64) {
        for (bound, count) in self.bounds.iter().zip(&self.buckets) {
            writeln!(
                output,
                "{name}_bucket{{{labels},le=\"{}\"}} {count}",
                bucket_label(*bound as f64 / scale)
            )
            .unwrap();
        }
        writeln!(
            output,
            "{name}_bucket{{{labels},le=\"+Inf\"}} {}",
            self.count
        )
        .unwrap();
        writeln!(output, "{name}_count{{{labels}}} {}", self.count).unwrap();
        writeln!(
            output,
            "{name}_sum{{{labels}}} {}",
            self.micros as f64 / scale
        )
        .unwrap();
    }
}

#[derive(Debug)]
struct LiveRequest {
    submitted: u64,
    started: Option<u64>,
    first_token: Option<u64>,
    last_token: Option<u64>,
    generated: u64,
    prompt_tokens: usize,
    max_tokens: usize,
    cached_tokens: usize,
    draft_tokens: usize,
    accepted_tokens: usize,
    trace_id: Option<String>,
    frontend_completion: bool,
    terminal: bool,
    preemptions: u64,
    /// Preempted and waiting to be scheduled again.
    preempted: bool,
}

impl LiveRequest {
    fn trace_record(&self, id: RequestId, reason: &str, now: u64) -> serde_json::Value {
        let seconds = |micros: u64| micros as f64 / 1_000_000.0;
        serde_json::json!({
            "schema": "chopinfer.request.v1",
            "request_id": self.trace_id,
            "engine_request_id": id.get(),
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "queue_seconds": seconds(self.started.unwrap_or(now).saturating_sub(self.submitted)),
            "ttft_seconds": self.first_token.map(|first| seconds(first.saturating_sub(self.submitted))),
            "e2e_seconds": (reason != "error").then(|| seconds(now.saturating_sub(self.submitted))),
            "generation_tokens": self.generated,
            "finish_reason": reason,
            "speculation": {"draft_tokens": self.draft_tokens, "accepted_tokens": self.accepted_tokens}
        })
    }
}

#[derive(Debug)]
pub(crate) struct CompatibilityMetrics {
    live: BTreeMap<RequestId, LiveRequest>,
    pub(crate) trace_enabled: bool,
    prompt_tokens: u64,
    cached_tokens: u64,
    request_prompt: Histogram,
    request_generation: Histogram,
    request_max: Histogram,
    request_computed: Histogram,
    success: [u64; 3],
    pub(crate) ttft: Histogram,
    pub(crate) itl: Histogram,
    e2e: Histogram,
    queue: Histogram,
    prefill: Histogram,
    decode: Histogram,
    inference: Histogram,
    time_per_token: Histogram,
    pub(crate) iteration: Histogram,
    preemptions: Histogram,
    preemptions_total: u64,
    pub(crate) gpu: Histogram,
    /// Engine wall time from execute through commit, by `STEP_PHASES`.
    step_execution: [Histogram; 3],
    step_prefill_tokens: Histogram,
    prefill_cap: Histogram,
    /// The decode-aware estimator after the latest step that trained it.
    pub(crate) step_estimates: Option<runtime_scheduler::StepTimeEstimates>,
    /// The TBT budget's predicted time of each mixed step, and the ratio of
    /// its execution time to that prediction.
    mixed_predicted: Histogram,
    mixed_ratio: Histogram,
    /// The TBT model after the latest step that trained it.
    pub(crate) mixed_estimates: Option<runtime_scheduler::MixedEstimates>,
    /// `CHOPINFER_STEP_TRACE=1` writes one JSON line per executed step.
    pub(crate) step_trace_enabled: bool,
    pub(crate) step_rows: crate::step_rows::StepRowHistogram,
}

impl Default for CompatibilityMetrics {
    fn default() -> Self {
        Self {
            live: BTreeMap::new(),
            prompt_tokens: 0,
            cached_tokens: 0,
            request_prompt: Histogram::new(&[]),
            request_generation: Histogram::new(&[]),
            request_max: Histogram::new(&[]),
            request_computed: Histogram::new(&[]),
            trace_enabled: std::env::var("CHOPINFER_REQUEST_TRACE").is_ok_and(|value| value == "1"),
            success: [0; 3],
            ttft: Histogram::new(TTFT),
            itl: Histogram::new(ITL),
            e2e: Histogram::new(E2E),
            queue: Histogram::new(E2E),
            prefill: Histogram::new(E2E),
            decode: Histogram::new(E2E),
            inference: Histogram::new(E2E),
            time_per_token: Histogram::new(ITL),
            iteration: Histogram::new(ITERATION),
            preemptions: Histogram::new(PREEMPTIONS),
            preemptions_total: 0,
            gpu: Histogram::new(ITL),
            step_execution: std::array::from_fn(|_| Histogram::new(STEP)),
            step_prefill_tokens: Histogram::new(STEP_PREFILL_TOKENS),
            prefill_cap: Histogram::new(PREFILL_CAP),
            step_estimates: None,
            mixed_predicted: Histogram::new(STEP),
            mixed_ratio: Histogram::new(PREDICTION_RATIO),
            mixed_estimates: None,
            step_rows: crate::step_rows::StepRowHistogram::default(),
            step_trace_enabled: std::env::var("CHOPINFER_STEP_TRACE")
                .is_ok_and(|value| value == "1"),
        }
    }
}

impl CompatibilityMetrics {
    /// Records one committed step: its execution time, its prefill tokens and
    /// the decode-aware cap selected for it, if any.
    pub(crate) fn step(
        &mut self,
        prefill_tokens: usize,
        decoding: bool,
        micros: u64,
        prefill_cap: Option<usize>,
    ) {
        let phase = match (prefill_tokens > 0, decoding) {
            (true, true) => 1,
            (true, false) => 2,
            (false, _) => 0,
        };
        self.step_execution[phase].observe(micros);
        if prefill_tokens > 0 {
            self.step_prefill_tokens.observe(prefill_tokens as u64);
        }
        if let Some(cap) = prefill_cap {
            self.prefill_cap.observe(cap as u64);
        }
    }

    /// Records a TBT mixed step's predicted and actual execution time.
    pub(crate) fn mixed_step(&mut self, predicted_micros: u64, micros: u64) {
        self.mixed_predicted.observe(predicted_micros);
        self.mixed_ratio.observe(
            (u128::from(micros) * 1000 / u128::from(predicted_micros.max(1)))
                .min(u128::from(u64::MAX)) as u64,
        );
    }

    fn render_mixed_budget(&self, output: &mut String, labels: &str) {
        if self.mixed_predicted.count > 0 {
            self.mixed_predicted.render_scaled(
                output,
                "chop_mixed_step_predicted_seconds",
                labels,
                1_000_000.0,
            );
            self.mixed_ratio.render_scaled(
                output,
                "chop_mixed_step_actual_over_predicted",
                labels,
                1000.0,
            );
        }
        let Some(estimates) = &self.mixed_estimates else {
            return;
        };
        writeln!(output, "# TYPE chop_mixed_budget_term_seconds gauge").unwrap();
        for (term, seconds) in estimates.terms_seconds() {
            writeln!(
                output,
                "chop_mixed_budget_term_seconds{{{labels},term=\"{term}\"}} {seconds:e}"
            )
            .unwrap();
        }
        writeln!(output, "# TYPE chop_mixed_budget_samples gauge").unwrap();
        writeln!(
            output,
            "chop_mixed_budget_samples{{{labels}}} {}",
            estimates.samples()
        )
        .unwrap();
        writeln!(output, "# TYPE chop_mixed_budget_trained gauge").unwrap();
        writeln!(
            output,
            "chop_mixed_budget_trained{{{labels}}} {}",
            u8::from(estimates.trained())
        )
        .unwrap();
        writeln!(
            output,
            "# TYPE chop_mixed_budget_decode_estimate_seconds gauge"
        )
        .unwrap();
        for (items, micros, _) in estimates.decode() {
            writeln!(
                output,
                "chop_mixed_budget_decode_estimate_seconds{{{labels},items_le=\"{items}\"}} {}",
                micros as f64 / 1_000_000.0
            )
            .unwrap();
        }
    }

    fn render_prefill_budget(&self, output: &mut String, labels: &str) {
        writeln!(output, "# TYPE chop_step_execution_seconds histogram").unwrap();
        for (phase, histogram) in STEP_PHASES.iter().zip(&self.step_execution) {
            histogram.samples(
                output,
                "chop_step_execution_seconds",
                &format!("{labels},phase=\"{phase}\""),
                1_000_000.0,
            );
        }
        self.step_prefill_tokens
            .render_scaled(output, "chop_step_prefill_tokens", labels, 1.0);
        if self.prefill_cap.count > 0 {
            self.prefill_cap.render_scaled(
                output,
                "chop_prefill_budget_selected_tokens",
                labels,
                1.0,
            );
        }
        self.render_mixed_budget(output, labels);
        let Some(estimates) = &self.step_estimates else {
            return;
        };
        writeln!(
            output,
            "# TYPE chop_prefill_budget_decode_estimate_seconds gauge"
        )
        .unwrap();
        for (rows, micros, _) in estimates.decode() {
            writeln!(
                output,
                "chop_prefill_budget_decode_estimate_seconds{{{labels},rows_le=\"{rows}\"}} {}",
                micros as f64 / 1_000_000.0
            )
            .unwrap();
        }
        writeln!(
            output,
            "# TYPE chop_prefill_budget_decode_estimate_samples gauge"
        )
        .unwrap();
        for (rows, _, samples) in estimates.decode() {
            writeln!(
                output,
                "chop_prefill_budget_decode_estimate_samples{{{labels},rows_le=\"{rows}\"}} {samples}"
            )
            .unwrap();
        }
        writeln!(
            output,
            "# TYPE chop_prefill_budget_prefill_token_seconds gauge"
        )
        .unwrap();
        for (tokens, nanos) in estimates.prefill() {
            writeln!(
                output,
                "chop_prefill_budget_prefill_token_seconds{{{labels},tokens_le=\"{tokens}\"}} {}",
                nanos as f64 / 1_000_000_000.0
            )
            .unwrap();
        }
    }

    pub(crate) fn configure(&mut self, maximum_tokens: usize) {
        let mut bounds = Vec::new();
        let mut magnitude = 1_u64;
        'outer: loop {
            for mantissa in [1, 2, 5] {
                let Some(value) = magnitude.checked_mul(mantissa) else {
                    break 'outer;
                };
                if value > maximum_tokens as u64 {
                    break 'outer;
                }
                bounds.push(value);
            }
            let Some(next) = magnitude.checked_mul(10) else {
                break;
            };
            magnitude = next;
        }
        self.request_prompt = Histogram::new(&bounds);
        self.request_generation = Histogram::new(&bounds);
        self.request_max = Histogram::new(&bounds);
        self.request_computed = Histogram::new(&bounds);
    }

    pub(crate) fn submit(
        &mut self,
        id: RequestId,
        submitted: u64,
        prompt_tokens: usize,
        max_tokens: usize,
    ) {
        self.live.insert(
            id,
            LiveRequest {
                submitted,
                started: None,
                first_token: None,
                last_token: None,
                generated: 0,
                prompt_tokens,
                max_tokens,
                cached_tokens: 0,
                draft_tokens: 0,
                accepted_tokens: 0,
                trace_id: None,
                frontend_completion: false,
                terminal: false,
                preemptions: 0,
                preempted: false,
            },
        );
    }

    pub(crate) fn trace_id(&mut self, id: RequestId, trace_id: &str) {
        if self.trace_enabled
            && let Some(request) = self.live.get_mut(&id)
        {
            request.trace_id = Some(trace_id.to_owned());
        }
    }

    pub(crate) fn cached(&mut self, id: RequestId, tokens: usize) {
        if let Some(request) = self.live.get_mut(&id) {
            request.cached_tokens = tokens;
        }
    }

    pub(crate) fn verified(&mut self, id: RequestId, drafted: usize, accepted: usize) {
        if let Some(request) = self.live.get_mut(&id) {
            request.draft_tokens += drafted;
            request.accepted_tokens += accepted;
        }
    }

    /// A live request returned to the queue to recompute; it waits again.
    pub(crate) fn preempted(&mut self, id: RequestId) {
        self.preemptions_total = self.preemptions_total.saturating_add(1);
        if let Some(request) = self.live.get_mut(&id) {
            request.preemptions += 1;
            request.preempted = true;
        }
    }

    /// A preempted request was scheduled again.
    pub(crate) fn resumed(&mut self, id: RequestId) {
        if let Some(request) = self.live.get_mut(&id) {
            request.preempted = false;
        }
    }

    pub(crate) fn start(&mut self, id: RequestId, now: u64) {
        if let Some(request) = self.live.get_mut(&id) {
            request.started.get_or_insert(now);
        }
    }

    pub(crate) fn tokens(&mut self, id: RequestId, count: usize, now: u64) {
        if count > 0
            && let Some(request) = self.live.get_mut(&id)
        {
            if request.first_token.is_none() {
                self.prompt_tokens = self
                    .prompt_tokens
                    .saturating_add(request.prompt_tokens as u64);
                self.cached_tokens = self
                    .cached_tokens
                    .saturating_add(request.cached_tokens as u64);
            }
            request.first_token.get_or_insert(now);
            request.last_token = Some(now);
            request.generated = request.generated.saturating_add(count as u64);
        }
    }

    pub(crate) fn defer_completion(&mut self, id: RequestId) {
        if let Some(request) = self.live.get_mut(&id) {
            request.frontend_completion = true;
        }
    }

    pub(crate) fn finish(&mut self, id: RequestId, reason: FinishReason, now: u64) {
        if let Some(request) = self.live.get_mut(&id)
            && request.frontend_completion
            && reason != FinishReason::Cancelled
        {
            request.terminal = true;
            return;
        }
        self.finish_completion(id, reason, now);
    }

    pub(crate) fn finish_completion(&mut self, id: RequestId, reason: FinishReason, now: u64) {
        if let Some(request) = self.live.remove(&id) {
            let index = match reason {
                FinishReason::Stop => 0,
                FinishReason::Length => 1,
                FinishReason::Cancelled => 2,
            };
            self.success[index] = self.success[index].saturating_add(1);
            self.e2e.observe(now.saturating_sub(request.submitted));
            let started = request.started.unwrap_or(now);
            let first = request.first_token.unwrap_or(started);
            let last = request.last_token.unwrap_or(first);
            self.queue
                .observe(started.saturating_sub(request.submitted));
            self.prefill.observe(first.saturating_sub(started));
            self.decode.observe(last.saturating_sub(first));
            self.inference.observe(last.saturating_sub(started));
            self.time_per_token.observe(if request.generated > 1 {
                last.saturating_sub(first) / (request.generated - 1)
            } else {
                0
            });
            self.preemptions.observe(request.preemptions);
            self.request_prompt.observe(request.prompt_tokens as u64);
            self.request_generation.observe(request.generated);
            self.request_max.observe(request.max_tokens as u64);
            self.request_computed
                .observe(request.prompt_tokens.saturating_sub(request.cached_tokens) as u64);
            if self.trace_enabled {
                let reason = ["stop", "length", "abort"][index];
                crate::diagnostic!("{}", request.trace_record(id, reason, now));
            }
        }
    }

    pub(crate) fn fail(&mut self, id: RequestId) {
        if let Some(request) = self.live.remove(&id)
            && self.trace_enabled
        {
            crate::diagnostic!(
                "{}",
                request.trace_record(
                    id,
                    "error",
                    request
                        .last_token
                        .or(request.started)
                        .unwrap_or(request.submitted)
                )
            );
        }
    }

    pub(crate) fn render(&self, labels: &str) -> String {
        let mut output = String::new();
        if self.gpu.count > 0 {
            self.gpu
                .render(&mut output, "chop_step_gpu_seconds", labels);
        }
        self.render_prefill_budget(&mut output, labels);
        self.step_rows.render(&mut output, labels);
        writeln!(output, "# TYPE vllm:request_success_total counter").unwrap();
        for (index, reason) in ["stop", "length", "abort"].iter().enumerate() {
            writeln!(
                output,
                "vllm:request_success_total{{{labels},finished_reason=\"{reason}\"}} {}",
                self.success[index]
            )
            .unwrap();
        }
        let running = self
            .live
            .values()
            .filter(|request| request.started.is_some() && !request.preempted && !request.terminal)
            .count();
        let waiting = self
            .live
            .values()
            .filter(|request| (request.started.is_none() || request.preempted) && !request.terminal)
            .count();
        for (name, value) in [
            ("vllm:num_requests_running", running),
            ("vllm:num_requests_waiting", waiting),
        ] {
            writeln!(output, "# TYPE {name} gauge\n{name}{{{labels}}} {value}").unwrap();
        }
        self.ttft
            .render(&mut output, "vllm:time_to_first_token_seconds", labels);
        self.itl
            .render(&mut output, "vllm:inter_token_latency_seconds", labels);
        self.e2e
            .render(&mut output, "vllm:e2e_request_latency_seconds", labels);
        for (name, histogram) in [
            ("vllm:request_queue_time_seconds", &self.queue),
            ("vllm:request_prefill_time_seconds", &self.prefill),
            ("vllm:request_decode_time_seconds", &self.decode),
            ("vllm:request_inference_time_seconds", &self.inference),
            (
                "vllm:request_time_per_output_token_seconds",
                &self.time_per_token,
            ),
        ] {
            histogram.render(&mut output, name, labels);
        }
        self.iteration
            .render_scaled(&mut output, "vllm:iteration_tokens_total", labels, 1.0);
        self.preemptions
            .render_scaled(&mut output, "vllm:request_num_preemptions", labels, 1.0);
        for (name, histogram) in [
            ("vllm:request_prompt_tokens", &self.request_prompt),
            ("vllm:request_generation_tokens", &self.request_generation),
            ("vllm:request_params_max_tokens", &self.request_max),
            (
                "vllm:request_prefill_kv_computed_tokens",
                &self.request_computed,
            ),
        ] {
            histogram.render_scaled(&mut output, name, labels, 1.0);
        }
        for (name, value) in [
            ("vllm:num_preemptions_total", self.preemptions_total),
            ("vllm:prompt_tokens_total", self.prompt_tokens),
            ("vllm:prompt_tokens_cached_total", self.cached_tokens),
        ] {
            counter(&mut output, name, labels, value);
        }
        writeln!(output, "# TYPE vllm:prompt_tokens_by_source_total counter").unwrap();
        for (source, count) in [
            (
                "local_compute",
                self.prompt_tokens.saturating_sub(self.cached_tokens),
            ),
            ("local_cache_hit", self.cached_tokens),
            ("external_kv_transfer", 0),
        ] {
            writeln!(
                output,
                "vllm:prompt_tokens_by_source_total{{{labels},source=\"{source}\"}} {count}"
            )
            .unwrap();
        }
        output
    }
}

// Match prometheus_client's floatToGoString for these positive bucket bounds.
fn bucket_label(value: f64) -> String {
    if value >= 1_000_000.0 {
        let exponent = value.log10().floor() as i32;
        format!("{}e+{exponent:02}", value / 10_f64.powi(exponent))
    } else if value.fract() == 0.0 {
        format!("{value:.1}")
    } else {
        value.to_string()
    }
}

pub(crate) fn labels(model: &str) -> String {
    let model = model
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('"', "\\\"");
    format!("model_name=\"{model}\",engine=\"0\"")
}

/// One committed step, as the `chopinfer.step.v1` trace line reports it.
#[derive(Clone, Debug)]
pub(crate) struct StepTrace {
    pub(crate) batch: u64,
    pub(crate) prefill_tokens: usize,
    pub(crate) prefill_items: usize,
    /// Request IDs and prompt lengths, without token contents.
    pub(crate) completed_prompts: Vec<(u64, usize)>,
    pub(crate) decode_rows: usize,
    pub(crate) verify_items: usize,
    pub(crate) verify_rows: usize,
    pub(crate) row_capacity: Option<usize>,
    pub(crate) prefill_cap: Option<usize>,
    pub(crate) execution_micros: u64,
    /// When the step completed, in nanoseconds since the Unix epoch, read on
    /// the engine thread: diagnostics reach stderr through a queue, so the
    /// time a line is read can be later than the step.
    pub(crate) completed_unix_ns: Option<u64>,
    /// What the decode-aware estimates, or the TBT model for a mixed step,
    /// assumed before this step trained them.
    pub(crate) predicted_micros: Option<u64>,
    pub(crate) decode_estimate_micros: Option<u64>,
    /// Whether the step trained the decode-aware estimator; verify and media
    /// steps do not. The TBT model also learns from verify steps.
    pub(crate) trained: bool,
    /// Where the engine thread spent the step's host time.
    pub(crate) host: HostTiming,
    /// The selected width policy's inputs and branch before cost training.
    pub(crate) speculation: Option<crate::speculation::PlanTrace>,
}

impl StepTrace {
    pub(crate) fn record(&self) -> serde_json::Value {
        serde_json::json!({
            "schema": "chopinfer.step.v1",
            "batch": self.batch,
            "prefill_tokens": self.prefill_tokens,
            "prefill_items": self.prefill_items,
            "prompt_completions": self.completed_prompts.len(),
            "completed_prompts": self.completed_prompts,
            "decode_rows": self.decode_rows,
            "verify_items": self.verify_items,
            "verify_rows": self.verify_rows,
            "live_rows": self.decode_rows + self.verify_rows,
            "row_capacity": self.row_capacity,
            "prefill_cap": self.prefill_cap,
            "execution_us": self.execution_micros,
            "completed_unix_ns": self.completed_unix_ns,
            "predicted_us": self.predicted_micros,
            "decode_estimate_us": self.decode_estimate_micros,
            "trained": self.trained,
            "host": self.host.record(),
            "speculation": self.speculation.as_ref().map(|trace| trace.record()),
        })
    }
}

/// Host time of one executed step on the engine thread, in microseconds.
///
/// `boundary_us` is the host-side step boundary: from the previous step's
/// last device wait ending to this step's first one starting. The device is
/// idle for at most that long between the two steps. It is absent when
/// either step never waited on the device, as with CPU executors; `gap_us`
/// is then the closest measure.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct HostTiming {
    /// Outside `Engine::step` since the previous call returned: the serving
    /// loop's event delivery, control commands and admission.
    pub(crate) outside: Option<u64>,
    /// Retirement, preemption, scheduling, drafting and items.
    pub(crate) schedule: u64,
    /// The executor call, device waits and in-execute sampling included.
    pub(crate) execute: u64,
    pub(crate) bundle_execute: u64,
    pub(crate) result_collect: u64,
    pub(crate) result_convert: u64,
    pub(crate) prefix_capture: u64,
    pub(crate) bundle_log: u64,
    /// From the executor's return to the commit timing mark: validation, sampling,
    /// scheduler and model commit, and applying the tokens.
    pub(crate) commit: u64,
    /// From the previous executor return to this executor call.
    pub(crate) gap: Option<u64>,
    /// From the previous executor call to this one.
    pub(crate) period: Option<u64>,
    pub(crate) boundary: Option<u64>,
    /// Phase totals: device waits, uploads, graph launches, sampling, and
    /// host work deferred into device waits.
    pub(crate) device_wait: u64,
    pub(crate) upload: u64,
    pub(crate) launch: u64,
    pub(crate) sampling: u64,
    pub(crate) deferred: u64,
}

impl HostTiming {
    pub(crate) fn finish_trace(&mut self) {
        let phases = crate::profile::snapshot();
        self.prefix_capture = phases.totals[crate::profile::StepPhase::PrefixCapture as usize];
    }

    fn record(&self) -> serde_json::Value {
        serde_json::json!({
            "outside_us": self.outside,
            "schedule_us": self.schedule,
            "execute_us": self.execute,
            "bundle_execute_us": self.bundle_execute,
            "result_collect_us": self.result_collect,
            "result_convert_us": self.result_convert,
            "prefix_capture_us": self.prefix_capture,
            "bundle_log_us": self.bundle_log,
            "commit_us": self.commit,
            "gap_us": self.gap,
            "period_us": self.period,
            "boundary_us": self.boundary,
            "device_wait_us": self.device_wait,
            "upload_us": self.upload,
            "launch_us": self.launch,
            "sampling_us": self.sampling,
            "deferred_us": self.deferred,
        })
    }
}

fn micros(duration: std::time::Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}

/// The engine thread's clock marks between traced steps.
#[derive(Clone, Debug, Default)]
pub(crate) struct HostClock {
    step_returned: Option<Instant>,
    step_entered: Option<Instant>,
    execute_started: Option<Instant>,
    execute_returned: Option<Instant>,
    previous_execute_started: Option<Instant>,
    previous_execute_returned: Option<Instant>,
    previous_wait_ended: Option<Instant>,
    outside: Option<u64>,
}

impl HostClock {
    pub(crate) fn step_entered(&mut self) {
        let now = Instant::now();
        self.outside = self.step_returned.map(|returned| micros(now - returned));
        self.step_entered = Some(now);
        self.execute_started = None;
        self.execute_returned = None;
    }

    pub(crate) fn step_returned(&mut self) {
        self.step_returned = Some(Instant::now());
    }

    pub(crate) fn execute_started(&mut self) {
        self.execute_started = Some(Instant::now());
    }

    pub(crate) fn execute_returned(&mut self) {
        self.execute_returned = Some(Instant::now());
    }

    /// Times the executed step up to now and makes it the previous step.
    pub(crate) fn executed(&mut self) -> HostTiming {
        let now = Instant::now();
        let phases = crate::profile::snapshot();
        let entered = self.step_entered.unwrap_or(now);
        let started = self.execute_started.unwrap_or(entered);
        let returned = self.execute_returned.unwrap_or(started);
        let timing = HostTiming {
            bundle_execute: phases.totals[crate::profile::StepPhase::BundleExecute as usize],
            result_collect: phases.totals[crate::profile::StepPhase::ResultCollect as usize],
            result_convert: phases.totals[crate::profile::StepPhase::ResultConvert as usize],
            prefix_capture: 0,
            bundle_log: phases.totals[crate::profile::StepPhase::BundleLog as usize],
            outside: self.outside,
            schedule: micros(started.saturating_duration_since(entered)),
            execute: micros(returned.saturating_duration_since(started)),
            commit: micros(now.saturating_duration_since(returned)),
            gap: self
                .previous_execute_returned
                .map(|previous| micros(started.saturating_duration_since(previous))),
            period: self
                .previous_execute_started
                .map(|previous| micros(started.saturating_duration_since(previous))),
            boundary: self
                .previous_wait_ended
                .zip(phases.first_wait_started)
                .map(|(ended, started)| micros(started.saturating_duration_since(ended))),
            device_wait: phases.totals[crate::profile::StepPhase::DeviceWait as usize],
            upload: phases.totals[crate::profile::StepPhase::Upload as usize],
            launch: phases.totals[crate::profile::StepPhase::GraphLaunch as usize],
            sampling: phases.totals[crate::profile::StepPhase::Sampling as usize],
            deferred: phases.totals[crate::profile::StepPhase::Deferred as usize],
        };
        self.previous_execute_started = Some(started);
        self.previous_execute_returned = Some(returned);
        self.previous_wait_ended = phases.last_wait_ended;
        timing
    }
}

pub(crate) fn counter(output: &mut String, name: &str, labels: &str, value: u64) {
    writeln!(output, "# TYPE {name} counter\n{name}{{{labels}}} {value}").unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_trace_json_preserves_identity_and_has_no_prompt_text() {
        let mut metrics = CompatibilityMetrics {
            trace_enabled: true,
            ..Default::default()
        };
        let id = RequestId::new(7);
        metrics.submit(id, 1_000_000, 100, 10);
        metrics.trace_id(id, "req-\"escaped\n_0");
        metrics.cached(id, 80);
        metrics.start(id, 2_000_000);
        metrics.tokens(id, 3, 3_000_000);
        metrics.verified(id, 4, 2);
        let json = metrics.live[&id].trace_record(id, "length", 4_000_000);
        let parsed: serde_json::Value = serde_json::from_str(&json.to_string()).unwrap();
        assert_eq!(parsed["schema"], "chopinfer.request.v1");
        assert_eq!(parsed["request_id"], "req-\"escaped\n_0");
        assert_eq!(parsed["queue_seconds"], 1.0);
        assert_eq!(parsed["ttft_seconds"], 2.0);
        assert_eq!(parsed["generation_tokens"], 3);
        assert_eq!(parsed["cached_tokens"], 80);
        assert_eq!(parsed["speculation"]["accepted_tokens"], 2);
        assert_eq!(parsed.as_object().unwrap().len(), 11);
    }

    #[test]
    fn step_trace_reports_the_cap_and_assumed_against_actual_time() {
        let trace = StepTrace {
            batch: 9,
            speculation: None,
            prefill_tokens: 2048,
            prefill_items: 1,
            completed_prompts: vec![(7, 2048)],
            decode_rows: 31,
            verify_items: 0,
            verify_rows: 0,
            row_capacity: None,
            prefill_cap: Some(2048),
            execution_micros: 175_300,
            completed_unix_ns: Some(1_792_224_301_138_000_123),
            predicted_micros: Some(175_328),
            decode_estimate_micros: Some(16_400),
            trained: true,
            host: HostTiming {
                outside: Some(40),
                schedule: 12,
                execute: 175_000,
                commit: 30,
                gap: Some(90),
                period: None,
                boundary: Some(150),
                device_wait: 174_700,
                bundle_execute: 170_000,
                result_collect: 4_000,
                result_convert: 1_000,
                prefix_capture: 25,
                bundle_log: 12,
                ..HostTiming::default()
            },
        };
        let parsed: serde_json::Value = serde_json::from_str(&trace.record().to_string()).unwrap();
        assert_eq!(parsed["schema"], "chopinfer.step.v1");
        assert_eq!(parsed["prefill_cap"], 2048);
        assert_eq!(parsed["execution_us"], 175_300);
        assert_eq!(parsed["completed_unix_ns"], 1_792_224_301_138_000_123_u64);
        assert_eq!(parsed["predicted_us"], 175_328);
        assert_eq!(parsed["decode_estimate_us"], 16_400);
        assert_eq!(parsed["host"]["boundary_us"], 150);
        assert_eq!(parsed["host"]["device_wait_us"], 174_700);
        assert_eq!(parsed["prompt_completions"], 1);
        for (field, expected) in [
            ("bundle_execute_us", 170_000),
            ("result_collect_us", 4_000),
            ("result_convert_us", 1_000),
            ("prefix_capture_us", 25),
            ("bundle_log_us", 12),
        ] {
            assert_eq!(parsed["host"][field], expected);
        }
        assert!(parsed["host"]["period_us"].is_null());
        assert_eq!(parsed["host"].as_object().unwrap().len(), 17);
        assert_eq!(parsed["live_rows"], 31);
        assert_eq!(parsed["verify_rows"], 0);
        assert!(parsed["row_capacity"].is_null());
        assert_eq!(parsed.as_object().unwrap().len(), 19);
        let fixed = StepTrace {
            prefill_cap: None,
            predicted_micros: None,
            decode_estimate_micros: None,
            ..trace
        }
        .record();
        assert!(fixed["prefill_cap"].is_null() && fixed["predicted_us"].is_null());
    }

    #[test]
    fn frontend_stop_preserves_trace_and_phase_metrics_until_finalization() {
        let mut metrics = CompatibilityMetrics {
            trace_enabled: true,
            ..Default::default()
        };
        let id = RequestId::new(7);
        metrics.submit(id, 1_000_000, 100, 10);
        metrics.trace_id(id, "frontend-stop");
        metrics.defer_completion(id);
        metrics.cached(id, 80);
        metrics.start(id, 2_000_000);
        metrics.tokens(id, 1, 3_000_000);
        metrics.tokens(id, 4, 4_000_000);
        metrics.verified(id, 3, 3);
        metrics.finish(id, FinishReason::Length, 4_000_000);
        assert_eq!(metrics.success, [0, 0, 0]);
        assert_eq!(metrics.request_generation.count, 0);
        let trace = metrics.live[&id].trace_record(id, "stop", 5_000_000);
        assert_eq!(trace["request_id"], "frontend-stop");
        assert_eq!(trace["finish_reason"], "stop");
        assert_eq!(trace["generation_tokens"], 5);
        assert_eq!(trace["cached_tokens"], 80);
        assert_eq!(trace["speculation"]["accepted_tokens"], 3);
        metrics.finish_completion(id, FinishReason::Stop, 5_000_000);
        metrics.finish_completion(id, FinishReason::Cancelled, 6_000_000);
        assert!(metrics.live.is_empty());
        assert_eq!(metrics.success, [1, 0, 0]);
        assert_eq!(metrics.e2e.count, 1);
        assert_eq!(metrics.e2e.micros, 4_000_000);
        assert_eq!(metrics.queue.micros, 1_000_000);
        assert_eq!(metrics.prefill.micros, 1_000_000);
        assert_eq!(metrics.decode.micros, 1_000_000);
        assert_eq!(metrics.time_per_token.micros, 250_000);
        assert_eq!(metrics.request_generation.count, 1);
        assert_eq!(metrics.request_generation.micros, 5);
        assert_eq!(metrics.request_computed.micros, 20);
    }

    #[test]
    fn completed_phase_histograms_include_queue_and_single_token_requests() {
        let mut metrics = CompatibilityMetrics::default();
        let id = RequestId::new(1);
        metrics.submit(id, 1_000_000, 10, 10);
        metrics.start(id, 3_000_000);
        metrics.tokens(id, 1, 4_000_000);
        metrics.tokens(id, 3, 10_000_000);
        metrics.finish(id, FinishReason::Length, 11_000_000);
        assert_eq!(metrics.queue.micros, 2_000_000);
        assert_eq!(metrics.prefill.micros, 1_000_000);
        assert_eq!(metrics.decode.micros, 6_000_000);
        assert_eq!(metrics.inference.micros, 7_000_000);
        assert_eq!(metrics.time_per_token.micros, 2_000_000);
        metrics.submit(id, 12_000_000, 10, 10);
        metrics.start(id, 13_000_000);
        metrics.tokens(id, 1, 14_000_000);
        metrics.finish(id, FinishReason::Stop, 14_000_000);
        assert_eq!(metrics.decode.count, 2);
        assert_eq!(metrics.decode.micros, 6_000_000);
        assert_eq!(metrics.time_per_token.count, 2);
        assert_eq!(metrics.preemptions.count, 2);
    }
}
