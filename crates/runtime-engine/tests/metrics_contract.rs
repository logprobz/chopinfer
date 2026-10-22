use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use runtime_engine::{
    Engine, EngineConfig, EngineMetrics, EngineRequest, ExecutionObservations, ExecutorError,
    LogitsRow, ModelBatchItem, ModelExecutor, ModelTransaction, MonotonicClock,
    NoopModelTransactionState,
};
use runtime_kv::{KvCachePrecision, KvGeometry, KvMetrics, PreparedAppendBatch};
use runtime_model_api::ModelVocabulary;
use runtime_sampling::SamplingParameters;
use runtime_scheduler::{RequestId, SchedulerConfig};

#[derive(Debug, Default)]
struct TestClock(AtomicU64);

impl TestClock {
    fn set(&self, value: u64) {
        self.0.store(value, Ordering::Relaxed);
    }
}

impl MonotonicClock for TestClock {
    fn now_micros(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

#[derive(Debug, Default)]
struct ObservedExecutor;

impl ModelExecutor for ObservedExecutor {
    type TransactionState<'a> = NoopModelTransactionState;

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        _kv_plan: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        let rows = items
            .iter()
            .filter(|item| item.expects_logits())
            .map(|item| {
                let mut logits = vec![0.0; 8];
                logits[2] = 1.0;
                LogitsRow::new(item.request_id(), logits)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(
            ModelTransaction::no_op(rows).with_observations(ExecutionObservations {
                kernel_launches: 3,
                kernel_micros: 20,
                collective_operations: 2,
                collective_micros: 7,
            }),
        )
    }
}

/// An executor that runs rows without measuring its own execution.
#[derive(Debug, Default)]
struct MeasurelessExecutor;

impl ModelExecutor for MeasurelessExecutor {
    type TransactionState<'a> = NoopModelTransactionState;

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        _kv_plan: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        let rows = items
            .iter()
            .filter(|item| item.expects_logits())
            .map(|item| {
                let mut logits = vec![0.0; 8];
                logits[2] = 1.0;
                LogitsRow::new(item.request_id(), logits)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ModelTransaction::no_op(rows))
    }
}

#[derive(Debug, Default)]
struct MissingOutputExecutor;

impl ModelExecutor for MissingOutputExecutor {
    type TransactionState<'a> = NoopModelTransactionState;

    fn execute<'a>(
        &'a mut self,
        _items: &[ModelBatchItem],
        _kv_plan: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        Ok(ModelTransaction::no_op(Vec::new()))
    }
}

#[derive(Debug, Default)]
struct FailingExecutor;

impl ModelExecutor for FailingExecutor {
    type TransactionState<'a> = NoopModelTransactionState;

    fn execute<'a>(
        &'a mut self,
        _items: &[ModelBatchItem],
        _kv_plan: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        Err(ExecutorError::new("injected failure").unwrap())
    }
}

fn config() -> EngineConfig {
    let geometry = KvGeometry::new(4, 8).unwrap();
    EngineConfig::new(
        SchedulerConfig::new(1, 2, 8, 8, 1).unwrap(),
        geometry,
        ModelVocabulary::new(8).unwrap(),
        geometry.token_capacity(),
        2,
    )
    .unwrap()
}

/// Two sequences over one budget, so a batch may carry both phases at once.
fn mixing_config() -> EngineConfig {
    let geometry = KvGeometry::new(4, 8).unwrap();
    EngineConfig::new(
        SchedulerConfig::new(2, 4, 8, 8, 1).unwrap(),
        geometry,
        ModelVocabulary::new(8).unwrap(),
        geometry.token_capacity(),
        2,
    )
    .unwrap()
}

fn request(id: u64, maximum_new_tokens: usize) -> EngineRequest {
    EngineRequest::new(
        RequestId::new(id),
        vec![1],
        maximum_new_tokens,
        SamplingParameters::new(0.0).unwrap(),
        id,
        Vec::new(),
        ModelVocabulary::new(8).unwrap(),
    )
    .unwrap()
}

#[test]
fn engine_records_real_lifecycle_boundaries_with_an_injected_clock() {
    let clock = Arc::new(TestClock::default());
    let metrics = EngineMetrics::default();
    let kv_metrics = KvMetrics::new(KvCachePrecision::Fp8);
    clock.set(100);
    let mut engine = Engine::new_with_runtime_metrics(
        config(),
        ObservedExecutor,
        clock.clone(),
        metrics.clone(),
        kv_metrics.clone(),
    )
    .unwrap();
    engine.submit(request(1, 2)).unwrap();
    let labels = r#"model_name="fixture",engine="0""#;
    let waiting = metrics.prometheus_compatibility("fixture");
    assert!(waiting.contains(&format!("vllm:num_requests_waiting{{{labels}}} 1")));
    assert!(waiting.contains(&format!("vllm:num_requests_running{{{labels}}} 0")));

    clock.set(150);
    engine.step().unwrap();
    assert_eq!(kv_metrics.snapshot().fp8_tokens, 1);
    assert_eq!(engine.kv_metrics().snapshot(), kv_metrics.snapshot());
    let running = metrics.prometheus_compatibility("fixture");
    assert!(running.contains(&format!("vllm:num_requests_running{{{labels}}} 1")));
    assert!(running.contains(&format!("vllm:num_requests_waiting{{{labels}}} 0")));
    // The first token starts decoding; it adds no decode token or time.
    assert!(running.contains(&format!("chop_request_decode_tokens_total{{{labels}}} 0")));
    assert!(running.contains(&format!(
        "chop_request_decode_seconds_total{{{labels}}} 0\n"
    )));
    clock.set(210);
    engine.step().unwrap();

    clock.set(300);
    engine.submit(request(2, 1)).unwrap();
    engine.cancel(RequestId::new(2)).unwrap();
    assert!(engine.cancel(RequestId::new(2)).is_err());

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.requests_submitted, 2);
    assert_eq!(snapshot.requests_completed, 1);
    assert_eq!(snapshot.requests_terminal, 2);
    assert_eq!(snapshot.requests_active, 0);
    assert_eq!(snapshot.prompt_tokens, 2);
    assert_eq!(snapshot.generation_tokens, 2);
    assert_eq!(snapshot.queue_micros, 50);
    assert_eq!(snapshot.queue_observations, 1);
    assert_eq!(snapshot.sequences_active, 0);
    assert_eq!(snapshot.batches, 2);
    assert_eq!(snapshot.batch_items, 2);
    assert_eq!(snapshot.ttft_micros, 50);
    assert_eq!(snapshot.ttft_observations, 1);
    assert_eq!(snapshot.itl_micros, 60);
    assert_eq!(snapshot.itl_observations, 1);
    assert_eq!(snapshot.kernel_launches, 6);
    assert_eq!(snapshot.kernel_micros, 40);
    assert_eq!(snapshot.collective_operations, 4);
    assert_eq!(snapshot.collective_micros, 14);
    assert_eq!(snapshot.errors, 0);
    assert_eq!(snapshot.cancellations, 1);
    assert_eq!(kv_metrics.snapshot().fp8_tokens, 0);
    engine.step().unwrap();
    let after_retirement = metrics.snapshot();
    assert_eq!(after_retirement.requests_terminal, 2);
    assert_eq!(after_retirement.requests_active, 0);

    let compatible = metrics.prometheus_compatibility("fixture");
    for sample in [
        format!("vllm:request_success_total{{{labels},finished_reason=\"length\"}} 1"),
        format!("vllm:request_success_total{{{labels},finished_reason=\"abort\"}} 1"),
        format!("vllm:num_requests_running{{{labels}}} 0"),
        format!("vllm:num_requests_waiting{{{labels}}} 0"),
        format!("vllm:time_to_first_token_seconds_bucket{{{labels},le=\"+Inf\"}} 1"),
        format!("vllm:time_to_first_token_seconds_sum{{{labels}}} 0.00005"),
        format!("vllm:inter_token_latency_seconds_sum{{{labels}}} 0.00006"),
        format!("vllm:e2e_request_latency_seconds_sum{{{labels}}} 0.00011"),
        format!("vllm:e2e_request_latency_seconds_count{{{labels}}} 2"),
        format!("vllm:prompt_tokens_total{{{labels}}} 1"),
        format!("vllm:generation_tokens_total{{{labels}}} 2"),
        // The second token came 60 us after the first; the cancelled
        // request never produced one.
        format!("chop_request_decode_tokens_total{{{labels}}} 1"),
        format!("chop_request_decode_seconds_total{{{labels}}} 0.00006"),
        format!("chop_speculation_k_total{{{labels},k=\"0\"}} 1"),
        format!(
            "chop_step_rows_total{{{labels},decode_rows=\"1\",verify_rows=\"0\",row_capacity=\"unknown\"}} 1"
        ),
        format!("chop_step_gpu_seconds_count{{{labels}}} 2"),
        format!("chop_step_gpu_seconds_sum{{{labels}}} 0.00004"),
    ] {
        assert!(
            compatible.contains(&sample),
            "missing {sample}: {compatible}"
        );
    }
    assert_eq!(
        compatible
            .matches("# TYPE vllm:request_success_total")
            .count(),
        1
    );
    assert!(
        metrics
            .prometheus_compatibility("a\"b\nc")
            .contains(r#"model_name="a\"b\nc""#)
    );
    let text = snapshot.prometheus();
    for family in [
        "requests",
        "tokens",
        "queue",
        "sequences",
        "batches",
        "ttft",
        "itl",
        "kernel",
        "collective",
        "errors",
        "cancellations",
    ] {
        assert!(text.contains(family), "missing metric family {family}");
    }
}

/// A mixed batch is recorded from the phases the scheduler put in it.
///
/// The two requests below overlap in flight for both steps, so overlap alone
/// proves nothing. What separates the single-phase batch from the mixed one is
/// the phase membership of each dispatched batch, and that is what these
/// counters carry.
#[test]
fn batch_phase_membership_separates_mixed_batches_from_overlapping_requests() {
    let metrics = EngineMetrics::default();
    let mut engine = Engine::new_with_observability(
        mixing_config(),
        ObservedExecutor,
        Arc::new(TestClock::default()),
        metrics.clone(),
    )
    .unwrap();

    engine.submit(request(1, 3)).unwrap();
    engine.step().unwrap();
    let first = metrics.snapshot();
    assert_eq!(first.batches, 1);
    assert_eq!(first.prefill_only_batches, 1);
    assert_eq!(first.decode_only_batches, 0);
    assert_eq!(first.mixed_batches, 0);
    assert_eq!(first.prefill_batch_items, 1);
    assert_eq!(first.decode_batch_items, 0);

    // A prompt admitted while the first request decodes is the mixed case.
    engine.submit(request(2, 1)).unwrap();
    engine.step().unwrap();
    let second = metrics.snapshot();
    assert_eq!(second.batches, 2);
    assert_eq!(second.prefill_only_batches, 1);
    assert_eq!(second.decode_only_batches, 0);
    assert_eq!(second.mixed_batches, 1);
    assert!(
        !metrics
            .prometheus_compatibility("fixture")
            .contains("chop_step_rows_total{")
    );
    assert_eq!(second.prefill_batch_items, 2);
    assert_eq!(second.decode_batch_items, 1);

    // The second request asked for one token, so the batch after it is the
    // first one alone, still decoding: both requests were in flight across
    // this step and the one before it, and only the phases tell them apart.
    engine.step().unwrap();
    let third = metrics.snapshot();
    assert_eq!(third.batches, 3);
    assert_eq!(third.mixed_batches, 1);
    assert_eq!(third.decode_only_batches, 1);
    assert_eq!(third.prefill_batch_items, 2);
    assert_eq!(third.decode_batch_items, 2);
    assert_eq!(
        third.batch_items,
        third.prefill_batch_items + third.decode_batch_items
    );
    assert_eq!(
        third.batches,
        third.prefill_only_batches + third.decode_only_batches + third.mixed_batches
    );

    let text = third.prometheus();
    for name in [
        "chop_prefill_only_batches_total",
        "chop_decode_only_batches_total",
        "chop_mixed_batches_total",
        "chop_prefill_batch_items_total",
        "chop_decode_batch_items_total",
    ] {
        assert!(text.contains(name), "missing metric {name}");
    }
}

/// An executor that measures nothing publishes no execution counters.
///
/// `ObservedExecutor` attaches its own counts and `MeasurelessExecutor` does
/// not, and the difference has to survive to the published text: four zeros
/// would read as an execution that launched no kernel.
#[test]
fn execution_counters_are_published_only_by_an_executor_that_measured_them() {
    let metrics = EngineMetrics::default();
    let mut measureless = Engine::new_with_observability(
        config(),
        MeasurelessExecutor,
        Arc::new(TestClock::default()),
        metrics.clone(),
    )
    .unwrap();
    measureless.submit(request(1, 1)).unwrap();
    measureless.step().unwrap();
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.batches, 1);
    assert_eq!(snapshot.execution_observations, 0);
    assert_eq!(snapshot.kernel_launches, 0);
    let text = snapshot.prometheus();
    assert!(text.contains("chop_execution_observations_total 0"));
    for absent in [
        "chop_kernel_launches_total",
        "chop_kernel_duration_microseconds_total",
        "chop_collective_operations_total",
        "chop_collective_duration_microseconds_total",
    ] {
        assert!(!text.contains(absent), "{absent} is published unmeasured");
    }

    let measured = EngineMetrics::default();
    let mut engine = Engine::new_with_observability(
        config(),
        ObservedExecutor,
        Arc::new(TestClock::default()),
        measured.clone(),
    )
    .unwrap();
    engine.submit(request(2, 1)).unwrap();
    engine.step().unwrap();
    let snapshot = measured.snapshot();
    assert_eq!(snapshot.execution_observations, 1);
    assert_eq!(snapshot.kernel_launches, 3);
    assert!(
        snapshot
            .prometheus()
            .contains("chop_kernel_launches_total 3")
    );
}

/// KV occupancy is the pool's own arithmetic, republished at step boundaries.
#[test]
fn kv_occupancy_follows_the_pool_through_admission_and_release() {
    let kv_metrics = KvMetrics::new(KvCachePrecision::Other);
    let mut engine = Engine::new_with_runtime_metrics(
        config(),
        ObservedExecutor,
        Arc::new(TestClock::default()),
        EngineMetrics::default(),
        kv_metrics.clone(),
    )
    .unwrap();
    let empty = kv_metrics.snapshot();
    assert_eq!(empty.block_capacity, 8);
    assert_eq!(empty.blocks_used, 0);
    assert_eq!(empty.active_sequences, 0);
    assert_eq!(empty.capacity_tokens, 32);
    // Nothing here declares FP8, so nothing here publishes an FP8 number.
    assert_eq!(empty.fp8_tokens, 0);
    assert_eq!(empty.fp8_capacity_tokens, 0);
    let text = empty.prometheus();
    assert!(text.contains("chop_kv_block_capacity 8"));
    assert!(!text.contains("chop_fp8_kv_capacity_tokens"));

    engine.submit(request(1, 2)).unwrap();
    engine.step().unwrap();
    let occupied = kv_metrics.snapshot();
    assert_eq!(occupied.active_sequences, 1);
    assert!(occupied.blocks_used > 0);
    assert_eq!(occupied.tokens, 1);

    engine.cancel(RequestId::new(1)).unwrap();
    let released = kv_metrics.snapshot();
    assert_eq!(released.active_sequences, 0);
    assert_eq!(released.blocks_used, 0);
    assert_eq!(released.tokens, 0);
    assert_eq!(released.block_capacity, 8);
}

#[test]
fn executor_errors_retire_request_metrics() {
    let clock = Arc::new(TestClock::default());
    let metrics = EngineMetrics::default();
    let mut engine =
        Engine::new_with_observability(config(), FailingExecutor, clock, metrics.clone()).unwrap();
    engine.submit(request(9, 1)).unwrap();
    assert!(engine.step().is_err());
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.errors, 1);
    assert_eq!(snapshot.requests_completed, 0);
    assert_eq!(snapshot.requests_terminal, 1);
    assert_eq!(snapshot.requests_active, 0);
    assert_eq!(snapshot.sequences_active, 0);
}

#[test]
fn output_mismatch_retires_request_metrics_once() {
    let metrics = EngineMetrics::default();
    let mut engine = Engine::new_with_observability(
        config(),
        MissingOutputExecutor,
        Arc::new(TestClock::default()),
        metrics.clone(),
    )
    .unwrap();
    engine.submit(request(10, 1)).unwrap();
    assert!(engine.step().is_err());
    assert_eq!(metrics.snapshot().errors, 1);
    assert_eq!(metrics.snapshot().requests_completed, 0);
    assert_eq!(metrics.snapshot().requests_terminal, 1);
    engine.step().unwrap();
    assert_eq!(metrics.snapshot().requests_terminal, 1);
}

#[test]
fn sampling_failure_retires_request_metrics_once() {
    let metrics = EngineMetrics::default();
    let mut engine = Engine::new_with_observability(
        config(),
        ObservedExecutor,
        Arc::new(TestClock::default()),
        metrics.clone(),
    )
    .unwrap();
    let sampling = SamplingParameters::new(0.0)
        .unwrap()
        .with_minimum_new_tokens(1)
        .unwrap();
    let request = EngineRequest::new(
        RequestId::new(11),
        vec![1],
        1,
        sampling,
        11,
        (0..8).collect(),
        ModelVocabulary::new(8).unwrap(),
    )
    .unwrap();
    engine.submit(request).unwrap();
    assert!(engine.step().is_err());
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.errors, 1);
    assert_eq!(snapshot.requests_completed, 0);
    assert_eq!(snapshot.requests_terminal, 1);
    assert_eq!(snapshot.requests_active, 0);
    engine.step().unwrap();
    assert_eq!(metrics.snapshot().requests_terminal, 1);
}

#[test]
fn reference_histograms_and_label_sets_match_the_pinned_fixture() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("vllm_metrics.json")).unwrap();
    let metrics = EngineMetrics::default();
    let _engine = Engine::new_with_observability(
        config(),
        ObservedExecutor,
        Arc::new(TestClock::default()),
        metrics.clone(),
    )
    .unwrap();
    let text = metrics.prometheus_compatibility("fixture");
    for (name, bounds) in fixture["histograms"].as_object().unwrap() {
        assert!(text.contains(&format!("# TYPE {name} histogram\n")));
        let prefix = format!("{name}_bucket{{model_name=\"fixture\",engine=\"0\",le=\"");
        let actual: Vec<f64> = text
            .lines()
            .filter_map(|line| {
                let rest = line.strip_prefix(&prefix)?;
                let bound = rest.split('"').next().unwrap();
                (bound != "+Inf").then(|| bound.parse().unwrap())
            })
            .collect();
        let expected: Vec<f64> = bounds
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_f64().unwrap())
            .collect();
        assert_eq!(actual, expected, "{name}");
        for label in fixture["bucket_labels"][name].as_array().unwrap() {
            assert!(
                text.contains(&format!("{prefix}{}\"}} 0", label.as_str().unwrap())),
                "{name}: missing exact bucket label {label}"
            );
        }
        assert!(text.contains(&format!("{prefix}+Inf\"}} 0")));
    }
    for (name, extra) in fixture["counters"].as_object().unwrap() {
        assert!(text.contains(&format!("# TYPE {name} counter\n")));
        let line = text
            .lines()
            .find(|line| line.starts_with(&format!("{name}{{")))
            .unwrap();
        let labels: Vec<_> = line
            .split_once('{')
            .unwrap()
            .1
            .split_once('}')
            .unwrap()
            .0
            .split(',')
            .map(|label| label.split_once('=').unwrap().0)
            .collect();
        let mut expected = vec!["model_name", "engine"];
        expected.extend(
            extra
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s.as_str().unwrap()),
        );
        assert_eq!(labels, expected, "{name}");
    }
    for name in fixture["gauges"].as_array().unwrap() {
        let name = name.as_str().unwrap();
        assert!(text.contains(&format!("{name}{{model_name=\"fixture\",engine=\"0\"}} 0")));
    }
}

#[test]
fn frontend_completion_is_counted_once_after_engine_retirement() {
    let clock = Arc::new(TestClock::default());
    let metrics = EngineMetrics::default();
    let mut engine =
        Engine::new_with_observability(config(), ObservedExecutor, clock.clone(), metrics.clone())
            .unwrap();
    let id = RequestId::new(1);
    engine.submit(request(1, 1)).unwrap();
    engine.defer_completion_metrics(id);
    clock.set(10);
    engine.run_until_idle(4).unwrap();
    let labels = "model_name=\"fixture\",engine=\"0\"";
    let pending = metrics.prometheus_compatibility("fixture");
    for reason in ["stop", "length", "abort"] {
        assert!(pending.contains(&format!(
            "vllm:request_success_total{{{labels},finished_reason=\"{reason}\"}} 0\n"
        )));
    }
    for gauge in ["running", "waiting"] {
        assert!(pending.contains(&format!("vllm:num_requests_{gauge}{{{labels}}} 0\n")));
    }
    clock.set(20);
    engine.finish_completion_metrics(id, runtime_engine::FinishReason::Stop);
    engine.finish_completion_metrics(id, runtime_engine::FinishReason::Length);
    engine.finish_completion_metrics(id, runtime_engine::FinishReason::Cancelled);
    let finished = metrics.prometheus_compatibility("fixture");
    assert!(finished.contains(&format!(
        "vllm:request_success_total{{{labels},finished_reason=\"stop\"}} 1\n"
    )));
    assert!(finished.contains(&format!(
        "vllm:e2e_request_latency_seconds_count{{{labels}}} 1\n"
    )));
    assert!(finished.contains(&format!(
        "vllm:e2e_request_latency_seconds_sum{{{labels}}} 0.00002\n"
    )));

    engine.submit(request(2, 2)).unwrap();
    engine.defer_completion_metrics(RequestId::new(2));
    engine.cancel(RequestId::new(2)).unwrap();
    let cancelled = metrics.prometheus_compatibility("fixture");
    assert!(cancelled.contains(&format!(
        "vllm:request_success_total{{{labels},finished_reason=\"abort\"}} 1\n"
    )));
    assert!(cancelled.contains(&format!(
        "vllm:e2e_request_latency_seconds_count{{{labels}}} 2\n"
    )));
}

/// Advances the clock by one microsecond per prefill token plus 4096 when
/// any request decodes.
struct TimedExecutor {
    clock: Arc<TestClock>,
    prefills: Vec<usize>,
}

impl ModelExecutor for TimedExecutor {
    type TransactionState<'a> = NoopModelTransactionState;
    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        kv: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        use runtime_scheduler::BatchPhase;
        let prefill: usize = items
            .iter()
            .map(|item| match item.phase() {
                BatchPhase::Prefill { tokens } => tokens,
                _ => 0,
            })
            .sum();
        let decode = items
            .iter()
            .filter(|item| item.phase() == BatchPhase::Decode)
            .count();
        self.prefills.push(prefill);
        self.clock.0.fetch_add(
            prefill as u64 + if decode > 0 { 4096 } else { 0 },
            Ordering::Relaxed,
        );
        MeasurelessExecutor.execute(items, kv)
    }
}

/// Runs one 2048-token prompt, two decode steps, then two 8192-token
/// prompts beside that decode.
fn timed_steps(policy: runtime_scheduler::PrefillBudget) -> (Vec<usize>, EngineMetrics) {
    let clock = Arc::new(TestClock::default());
    let metrics = EngineMetrics::default();
    let config = EngineConfig::new(
        SchedulerConfig::new(4, 8, 8192, 2048, 1)
            .unwrap()
            .with_prefill_budget(policy),
        KvGeometry::new(16, 4096).unwrap(),
        ModelVocabulary::new(8).unwrap(),
        16384,
        4,
    )
    .unwrap();
    let mut engine = Engine::new_with_observability(
        config,
        TimedExecutor {
            clock: clock.clone(),
            prefills: Vec::new(),
        },
        clock,
        metrics.clone(),
    )
    .unwrap();
    let request = |id, tokens| {
        EngineRequest::new(
            RequestId::new(id),
            vec![1; tokens],
            64,
            SamplingParameters::new(0.0).unwrap(),
            id,
            Vec::new(),
            ModelVocabulary::new(8).unwrap(),
        )
        .unwrap()
    };
    engine.submit(request(1, 2048)).unwrap();
    engine.step().unwrap();
    engine.step().unwrap();
    engine.step().unwrap();
    engine.submit(request(2, 8192)).unwrap();
    engine.submit(request(3, 8192)).unwrap();
    engine.step().unwrap();
    (engine.executor().prefills.clone(), metrics)
}

#[test]
fn successful_execution_times_feed_the_decode_aware_budget() {
    use runtime_scheduler::{PrefillBudget, PrefillTarget};
    let (prefills, metrics) =
        timed_steps(PrefillBudget::DecodeAware(PrefillTarget::DecodeSteps(1)));
    assert_eq!(prefills, vec![2048, 0, 0, 4096]);

    // The policy is visible: execution time per step phase, prefill tokens
    // per step, the cap selected beside decode rows, and the estimates.
    let text = metrics.prometheus_compatibility("fixture");
    let labels = "model_name=\"fixture\",engine=\"0\"";
    for line in [
        format!("chop_step_execution_seconds_count{{{labels},phase=\"prefill\"}} 1"),
        format!("chop_step_execution_seconds_sum{{{labels},phase=\"prefill\"}} 0.002048"),
        format!("chop_step_execution_seconds_count{{{labels},phase=\"decode\"}} 2"),
        format!("chop_step_execution_seconds_sum{{{labels},phase=\"decode\"}} 0.008192"),
        format!("chop_step_execution_seconds_bucket{{{labels},phase=\"decode\",le=\"0.005\"}} 2"),
        format!("chop_step_execution_seconds_count{{{labels},phase=\"mixed\"}} 1"),
        format!("chop_step_execution_seconds_bucket{{{labels},phase=\"mixed\",le=\"0.005\"}} 0"),
        format!("chop_step_execution_seconds_bucket{{{labels},phase=\"mixed\",le=\"0.01\"}} 1"),
        format!("chop_step_prefill_tokens_count{{{labels}}} 2"),
        format!("chop_step_prefill_tokens_sum{{{labels}}} 6144"),
        // Cold 512 twice, then 4096 once the decode estimate qualifies.
        format!("chop_prefill_budget_selected_tokens_bucket{{{labels},le=\"512.0\"}} 2"),
        format!("chop_prefill_budget_selected_tokens_bucket{{{labels},le=\"3584.0\"}} 2"),
        format!("chop_prefill_budget_selected_tokens_bucket{{{labels},le=\"4096.0\"}} 3"),
        format!("chop_prefill_budget_selected_tokens_sum{{{labels}}} 5120"),
        format!("chop_prefill_budget_decode_estimate_seconds{{{labels},rows_le=\"1\"}} 0.004096"),
        format!("chop_prefill_budget_decode_estimate_samples{{{labels},rows_le=\"1\"}} 2"),
        format!(
            "chop_prefill_budget_prefill_token_seconds{{{labels},tokens_le=\"2048\"}} 0.000001"
        ),
        format!(
            "chop_prefill_budget_prefill_token_seconds{{{labels},tokens_le=\"4096\"}} 0.000001"
        ),
    ] {
        assert!(text.contains(&format!("{line}\n")), "{line}\n{text}");
    }
    for name in [
        "chop_step_execution_seconds",
        "chop_prefill_budget_decode_estimate_seconds",
    ] {
        assert_eq!(text.matches(&format!("# TYPE {name} ")).count(), 1);
    }
}

#[test]
fn fixed_budget_publishes_step_times_without_estimator_series() {
    let (prefills, metrics) = timed_steps(runtime_scheduler::PrefillBudget::Fixed);
    assert_eq!(prefills, vec![2048, 0, 0, 4096]);
    let text = metrics.prometheus_compatibility("fixture");
    let labels = "model_name=\"fixture\",engine=\"0\"";
    assert!(text.contains(&format!(
        "chop_step_execution_seconds_count{{{labels},phase=\"mixed\"}} 1\n"
    )));
    assert!(text.contains(&format!("chop_step_prefill_tokens_sum{{{labels}}} 6144\n")));
    assert!(!text.contains("chop_prefill_budget_"), "{text}");
}

#[test]
fn tbt_budget_publishes_its_model_and_predicted_against_actual_mixed_steps() {
    use runtime_scheduler::{MixedPolicy, PrefillBudget};
    let clock = Arc::new(TestClock::default());
    let metrics = EngineMetrics::default();
    let config = EngineConfig::new(
        SchedulerConfig::new(4, 64, 8192, 2048, 1)
            .unwrap()
            .with_prefill_budget(PrefillBudget::Tbt(MixedPolicy::new(1, 512, 0).unwrap())),
        KvGeometry::new(16, 4096).unwrap(),
        ModelVocabulary::new(8).unwrap(),
        16384,
        4,
    )
    .unwrap();
    let mut engine = Engine::new_with_observability(
        config,
        TimedExecutor {
            clock: clock.clone(),
            prefills: Vec::new(),
        },
        clock,
        metrics.clone(),
    )
    .unwrap();
    let request = |id, tokens, output| {
        EngineRequest::new(
            RequestId::new(id),
            vec![1; tokens],
            output,
            SamplingParameters::new(0.0).unwrap(),
            id,
            Vec::new(),
            ModelVocabulary::new(8).unwrap(),
        )
        .unwrap()
    };
    engine.submit(request(1, 16, 1000)).unwrap();
    engine.step().unwrap();
    engine.step().unwrap();
    engine.step().unwrap();
    for id in 2..20 {
        engine
            .submit(request(id, 1024 + 100 * id as usize, 1))
            .unwrap();
    }
    for _ in 0..60 {
        engine.step().unwrap();
    }
    let prefills = engine.executor().prefills.clone();
    // The cold cap uses the floor before training; the 1 ms fitted target
    // then keeps each mixed step at that same 512-token floor.
    assert_eq!(prefills[..3], [16, 0, 0]);
    // Cold fairness revisits the oldest prompt, including its 200-token
    // tail, instead of taking a full chunk from a new prompt every time.
    assert_eq!(prefills[3..7], [512, 512, 512, 200]);
    assert!(
        prefills[40..].iter().all(|&tokens| tokens <= 512),
        "{prefills:?}"
    );

    let text = metrics.prometheus_compatibility("fixture");
    let labels = "model_name=\"fixture\",engine=\"0\"";
    let value = |name: &str| -> f64 {
        let prefix = format!("{name}{{{labels}");
        text.lines()
            .find_map(|line| line.strip_prefix(&prefix))
            .and_then(|rest| rest.rsplit(' ').next())
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| panic!("{name}\n{text}"))
    };
    assert_eq!(value("chop_mixed_budget_trained"), 1.0);
    assert!(value("chop_mixed_budget_samples") >= 8.0);
    assert!(value("chop_mixed_step_predicted_seconds_count") > 10.0);
    // Cold samples now share the floor size. While the fit gains varied
    // tails, integer rounding can put actual/predicted just above one.
    // Retain a 10% error bound on both sides of the known synthetic cost.
    let ratio_count = value("chop_mixed_step_actual_over_predicted_count");
    assert!(text.contains(&format!(
        "chop_mixed_step_actual_over_predicted_bucket{{{labels},le=\"1.1\"}} {ratio_count}\n"
    )));
    assert!(text.contains(&format!(
        "chop_mixed_step_actual_over_predicted_bucket{{{labels},le=\"0.9\"}} 0\n"
    )));
    assert!(text.contains(&format!(
        "chop_mixed_budget_decode_estimate_seconds{{{labels},items_le=\"1\"}} 0.004096\n"
    )));
    let token = text
        .lines()
        .find_map(|line| {
            line.strip_prefix(&format!(
                "chop_mixed_budget_term_seconds{{{labels},term=\"token\"}} "
            ))
        })
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap();
    assert!((token - 1e-6).abs() < 1e-9, "{token}");
    assert!(text.contains(&format!(
        "chop_prefill_budget_selected_tokens_bucket{{{labels},le=\"512.0\"}}"
    )));
}
