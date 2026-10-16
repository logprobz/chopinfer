use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use runtime_engine::{
    Engine, EngineConfig, EngineError, EngineEventKind, EngineRequest, ExecutorError, FinishReason,
    LogitsRow, ModelBatchItem, ModelExecutor, ModelTransaction, ModelTransactionState,
    NoopModelTransactionState, StepStatus,
};
use runtime_kv::{KvGeometry, PreparedAppendBatch};
use runtime_model_api::ModelVocabulary;
use runtime_sampling::SamplingParameters;
use runtime_scheduler::{
    BatchPhase, Overload, RequestId, SchedulerCapacityProfile, SchedulerConfig, SubmitError,
};

#[derive(Clone, Debug, Default)]
struct DeterministicExecutor {
    scripts: BTreeMap<RequestId, Vec<u32>>,
    emitted: BTreeMap<RequestId, usize>,
    calls: usize,
    fail_call: Option<usize>,
    refuse_call: Option<usize>,
    traces: Vec<Vec<(RequestId, BatchPhase, Vec<u32>)>>,
    kv_traces: Vec<Vec<(Vec<usize>, Vec<usize>)>>,
}

impl DeterministicExecutor {
    fn scripted(scripts: &[(u64, &[u32])]) -> Self {
        Self {
            scripts: scripts
                .iter()
                .map(|(id, tokens)| (RequestId::new(*id), tokens.to_vec()))
                .collect(),
            ..Self::default()
        }
    }
}

impl ModelExecutor for DeterministicExecutor {
    type TransactionState<'a> = NoopModelTransactionState;

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        kv_plan: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        self.calls += 1;
        if self.refuse_call == Some(0) || self.refuse_call == Some(self.calls) {
            return Err(ExecutorError::batch_refused("packing check: invalid argument").unwrap());
        }
        if self.fail_call == Some(self.calls) {
            return Err(ExecutorError::new("injected executor failure").unwrap());
        }
        assert_eq!(items.len(), kv_plan.appends().len());
        for (item, append) in items.iter().zip(kv_plan.appends()) {
            let expected_append = item.phase().appended_tokens();
            assert_eq!(append.additional_tokens(), expected_append);
            assert_eq!(item.input_tokens().len(), expected_append);
        }
        self.traces.push(
            items
                .iter()
                .map(|item| {
                    (
                        item.request_id(),
                        item.phase(),
                        item.input_tokens().to_vec(),
                    )
                })
                .collect(),
        );
        self.kv_traces.push(
            kv_plan
                .appends()
                .iter()
                .map(|append| {
                    (
                        append
                            .block_table()
                            .iter()
                            .map(|block| block.index())
                            .collect(),
                        append
                            .newly_allocated()
                            .iter()
                            .map(|block| block.index())
                            .collect(),
                    )
                })
                .collect(),
        );

        let rows = items
            .iter()
            .filter(|item| item.expects_logits())
            .map(|item| {
                let emitted = self.emitted.entry(item.request_id()).or_default();
                let token = self
                    .scripts
                    .get(&item.request_id())
                    .and_then(|script| script.get(*emitted))
                    .copied()
                    .unwrap_or_else(|| (item.request_id().get() as u32 + *emitted as u32) % 16);
                *emitted += 1;
                let mut logits = vec![-10.0; 32];
                logits[token as usize] = 10.0;
                LogitsRow::new(item.request_id(), logits)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ModelTransaction::no_op(rows))
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct MissingRowExecutor;

impl ModelExecutor for MissingRowExecutor {
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
struct PanicTrace {
    active: AtomicUsize,
    committed: AtomicUsize,
    rolled_back: AtomicUsize,
}

struct PanicTransactionState {
    trace: Arc<PanicTrace>,
}

impl PanicTransactionState {
    fn new(trace: Arc<PanicTrace>) -> Self {
        trace.active.fetch_add(1, Ordering::SeqCst);
        Self { trace }
    }
}

impl ModelTransactionState for PanicTransactionState {
    fn commit(&mut self) -> Result<(), ExecutorError> {
        self.trace.committed.fetch_add(1, Ordering::SeqCst);
        self.trace.active.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }

    fn rollback(&mut self) -> Result<(), ExecutorError> {
        self.trace.rolled_back.fetch_add(1, Ordering::SeqCst);
        self.trace.active.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }
}

struct PanicOnceExecutor {
    panic_next: bool,
    trace: Arc<PanicTrace>,
}

impl ModelExecutor for PanicOnceExecutor {
    type TransactionState<'a> = PanicTransactionState;

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        _kv_plan: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        let state = PanicTransactionState::new(Arc::clone(&self.trace));
        if self.panic_next {
            self.panic_next = false;
            let _transaction = ModelTransaction::new(Vec::new(), state);
            panic!("injected executor panic");
        }
        let rows = items
            .iter()
            .filter(|item| item.expects_logits())
            .map(|item| {
                let mut logits = vec![-10.0; 32];
                logits[3] = 10.0;
                LogitsRow::new(item.request_id(), logits)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ModelTransaction::new(rows, state))
    }
}

fn config(concurrency: usize, budget: usize, retirement: usize) -> EngineConfig {
    let geometry = KvGeometry::new(4, concurrency * 16).unwrap();
    EngineConfig::new(
        SchedulerConfig::new(concurrency, concurrency * 2, budget, 8, 1).unwrap(),
        geometry,
        ModelVocabulary::new(32).unwrap(),
        geometry.token_capacity(),
        retirement,
    )
    .unwrap()
}

#[test]
fn engine_preserves_named_profile_backpressure() {
    for (name, concurrency) in [("C1", 1), ("C4", 4), ("C16", 16), ("C32", 32)] {
        let geometry = KvGeometry::new(4, concurrency * 2).unwrap();
        let profile = SchedulerCapacityProfile::new(name, concurrency).unwrap();
        let scheduler = SchedulerConfig::new(concurrency, concurrency + 1, 8, 8, 1)
            .unwrap()
            .with_capacity_profile(profile)
            .unwrap();
        let config = EngineConfig::new(
            scheduler,
            geometry,
            ModelVocabulary::new(32).unwrap(),
            geometry.token_capacity(),
            1,
        )
        .unwrap();
        let mut engine = Engine::new(config, DeterministicExecutor::default()).unwrap();
        assert_eq!(engine.capacity_profile(), profile);

        for id in 0..concurrency {
            engine
                .submit(request(id as u64 + 1, &[1], 1, &[]))
                .expect("work within the profile bound is accepted");
        }
        assert_eq!(
            engine.submit(request(concurrency as u64 + 1, &[1], 1, &[])),
            Err(EngineError::Submit(SubmitError::Overloaded(
                Overload::ProfileCapacity {
                    profile: name,
                    maximum: concurrency,
                }
            ))),
            "{name} must propagate scheduler backpressure"
        );
    }
}

/// The engine reports back exactly the vocabulary its caller declared.
///
/// It has no vocabulary of its own to fall back to, so two configurations that
/// differ only in this differ everywhere it is used.
#[test]
fn configuration_carries_the_vocabulary_its_caller_declared() {
    let geometry = KvGeometry::new(4, 16).unwrap();
    for size in [32, 154_880] {
        let config = EngineConfig::new(
            SchedulerConfig::new(1, 2, 8, 8, 1).unwrap(),
            geometry,
            ModelVocabulary::new(size).unwrap(),
            64,
            1,
        )
        .unwrap();
        assert_eq!(config.model_vocabulary().get(), size);
    }
}

#[test]
fn large_vocabulary_accepts_high_tokens_and_rejects_out_of_range_tokens() {
    let geometry = KvGeometry::new(4, 16).unwrap();
    let vocabulary = ModelVocabulary::new(154_880).unwrap();
    let config = EngineConfig::new(
        SchedulerConfig::new(1, 2, 8, 8, 1).unwrap(),
        geometry,
        vocabulary,
        64,
        1,
    )
    .unwrap();
    let mut engine = Engine::new(config, DeterministicExecutor::default()).unwrap();
    let accepted = EngineRequest::new(
        RequestId::new(1),
        vec![154_879],
        1,
        SamplingParameters::new(0.0).unwrap(),
        1,
        Vec::new(),
        vocabulary,
    )
    .unwrap();
    engine.submit(accepted).unwrap();

    let rejected = EngineRequest::new(
        RequestId::new(2),
        vec![154_880],
        1,
        SamplingParameters::new(0.0).unwrap(),
        2,
        Vec::new(),
        vocabulary,
    );
    assert_eq!(
        rejected,
        Err(
            runtime_engine::EngineRequestError::PromptTokenOutOfVocabulary {
                position: 0,
                token: 154_880,
                vocabulary: 154_880,
            }
        )
    );
}

#[test]
fn engine_rejects_model_config_drift() {
    let geometry = KvGeometry::new(4, 16).unwrap();
    let config = EngineConfig::new(
        SchedulerConfig::new(1, 2, 8, 8, 1).unwrap(),
        geometry,
        ModelVocabulary::new(32).unwrap(),
        64,
        1,
    )
    .unwrap();
    let mut engine = Engine::new(config, DeterministicExecutor::default()).unwrap();
    let drifted = EngineRequest::new(
        RequestId::new(1),
        vec![1],
        1,
        SamplingParameters::new(0.0).unwrap(),
        1,
        Vec::new(),
        ModelVocabulary::new(33).unwrap(),
    )
    .unwrap();
    assert_eq!(
        engine.submit(drifted),
        Err(EngineError::InvalidRequest(
            runtime_engine::EngineRequestError::VocabularyMismatch {
                request: 33,
                engine: 32,
            }
        ))
    );
}

#[test]
fn engine_rejects_logit_vocabulary_drift() {
    let geometry = KvGeometry::new(4, 16).unwrap();
    let vocabulary = ModelVocabulary::new(33).unwrap();
    let config = EngineConfig::new(
        SchedulerConfig::new(1, 2, 8, 8, 1).unwrap(),
        geometry,
        vocabulary,
        64,
        1,
    )
    .unwrap();
    let mut engine = Engine::new(config, DeterministicExecutor::default()).unwrap();
    engine
        .submit(
            EngineRequest::new(
                RequestId::new(1),
                vec![1],
                1,
                SamplingParameters::new(0.0).unwrap(),
                1,
                Vec::new(),
                vocabulary,
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(engine.step(), Err(EngineError::OutputMismatch));
}

#[test]
fn engine_rejects_vocabulary_above_the_sampler_limit() {
    let geometry = KvGeometry::new(4, 16).unwrap();
    let provided = runtime_sampling::MAX_VOCABULARY + 1;
    assert_eq!(
        EngineConfig::new(
            SchedulerConfig::new(1, 2, 8, 8, 1).unwrap(),
            geometry,
            ModelVocabulary::new(provided).unwrap(),
            64,
            1,
        ),
        Err(runtime_engine::EngineConfigError::VocabularyTooLarge {
            provided,
            maximum: runtime_sampling::MAX_VOCABULARY,
        })
    );
}

#[test]
fn request_rejects_stop_tokens_outside_the_declared_vocabulary() {
    let vocabulary = ModelVocabulary::new(32).unwrap();
    for minimum_new_tokens in [0, 1] {
        let sampling = SamplingParameters::new(0.0)
            .unwrap()
            .with_minimum_new_tokens(minimum_new_tokens)
            .unwrap();
        assert_eq!(
            EngineRequest::new(
                RequestId::new(1),
                vec![1],
                1,
                sampling,
                1,
                vec![31, 32],
                vocabulary,
            ),
            Err(
                runtime_engine::EngineRequestError::StopTokenOutOfVocabulary {
                    token: 32,
                    vocabulary: 32,
                }
            )
        );
    }
}

#[test]
fn request_rejects_logit_bias_outside_the_declared_vocabulary() {
    let vocabulary = ModelVocabulary::new(32).unwrap();
    let biased = |token| {
        SamplingParameters::new(1.0)
            .unwrap()
            .with_logit_bias(vec![runtime_sampling::LogitBias::new(token, 1.0).unwrap()])
            .unwrap()
    };
    assert!(
        EngineRequest::new(
            RequestId::new(1),
            vec![1],
            1,
            biased(31),
            1,
            vec![],
            vocabulary
        )
        .is_ok()
    );
    for token in [32, u32::MAX] {
        assert_eq!(
            EngineRequest::new(
                RequestId::new(1),
                vec![1],
                1,
                biased(token),
                1,
                vec![],
                vocabulary
            ),
            Err(
                runtime_engine::EngineRequestError::LogitBiasTokenOutOfVocabulary {
                    token,
                    vocabulary: 32,
                }
            )
        );
    }
}

#[test]
fn engine_rejects_a_request_above_its_configured_sequence_limit() {
    let executor = DeterministicExecutor::default();
    let mut engine = Engine::new(config(1, 8, 4), executor).unwrap();
    assert_eq!(engine.maximum_sequence_tokens(), 64);
    engine.submit(request(1, &[1, 2, 3], 61, &[])).unwrap();
    assert_eq!(
        engine.submit(request(2, &[1, 2, 3], 62, &[])),
        Err(EngineError::InvalidRequest(
            runtime_engine::EngineRequestError::SequenceTooLong {
                maximum: 64,
                requested: 65,
            }
        ))
    );
}

#[test]
fn engine_config_rejects_a_sequence_limit_above_kv_capacity() {
    let geometry = KvGeometry::new(4, 16).unwrap();
    assert_eq!(
        EngineConfig::new(
            SchedulerConfig::new(1, 2, 8, 8, 1).unwrap(),
            geometry,
            ModelVocabulary::new(32).unwrap(),
            65,
            1,
        ),
        Err(
            runtime_engine::EngineConfigError::MaximumSequenceTokensExceedKvCapacity {
                provided: 65,
                maximum: 64,
            }
        )
    );
}

#[test]
fn acquired_page_capacity_bounds_engine_admission() {
    assert_eq!(
        config(1, 8, 4).with_kv_page_capacity(0),
        Err(runtime_engine::EngineConfigError::InvalidKvPageCapacity)
    );
    let config = config(1, 8, 4).with_kv_page_capacity(2).unwrap();
    let mut engine = Engine::new(config, DeterministicExecutor::default()).unwrap();
    assert_eq!(engine.maximum_sequence_tokens(), 8);
    assert_eq!(
        engine.submit(request(1, &[1, 2, 3], 6, &[])),
        Err(EngineError::InvalidRequest(
            runtime_engine::EngineRequestError::SequenceTooLong {
                maximum: 8,
                requested: 9,
            }
        ))
    );
}

fn request(id: u64, prompt: &[u32], maximum: usize, stops: &[u32]) -> EngineRequest {
    EngineRequest::new(
        RequestId::new(id),
        prompt.to_vec(),
        maximum,
        SamplingParameters::new(0.0).unwrap(),
        id + 99,
        stops.to_vec(),
        ModelVocabulary::new(32).unwrap(),
    )
    .unwrap()
}

#[test]
fn final_prefill_emits_first_token_and_max_one_finishes_immediately() {
    let executor = DeterministicExecutor::scripted(&[(1, &[3])]);
    let mut engine = Engine::new(config(1, 8, 4), executor).unwrap();
    engine.submit(request(1, &[10, 11], 1, &[])).unwrap();
    assert_eq!(engine.step().unwrap(), StepStatus::Progressed);
    let events = engine.drain_events();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].sequence(), 0);
    assert!(matches!(
        events[0].kind(),
        EngineEventKind::Token { token_id: 3, .. }
    ));
    assert_eq!(events[1].sequence(), 1);
    assert_eq!(
        events[1].kind(),
        &EngineEventKind::Finished {
            reason: FinishReason::Length
        }
    );
    assert_eq!(engine.executor().calls, 1);
    assert_eq!(engine.kv_stats().used_blocks(), 0);
    assert_eq!(engine.step().unwrap(), StepStatus::Idle);
}

#[test]
fn decode_appends_only_prior_pending_tokens_without_off_by_one() {
    let executor = DeterministicExecutor::scripted(&[(1, &[2, 3, 4])]);
    let mut engine = Engine::new(config(1, 8, 4), executor).unwrap();
    engine.submit(request(1, &[9], 3, &[])).unwrap();
    engine.run_until_idle(8).unwrap();
    let tokens = engine
        .drain_events()
        .into_iter()
        .filter_map(|event| match event.kind() {
            EngineEventKind::Token { token_id, .. } => Some(*token_id),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tokens, [2, 3, 4]);
    assert_eq!(engine.executor().traces.len(), 3);
    assert_eq!(engine.executor().traces[0][0].2, [9]);
    assert_eq!(engine.executor().traces[1][0].2, [2]);
    assert_eq!(engine.executor().traces[2][0].2, [3]);
    assert_eq!(engine.kv_stats().used_blocks(), 0);
}

#[test]
fn decode_executor_receives_complete_page_table_without_local_mapping_state() {
    let executor = DeterministicExecutor::scripted(&[(1, &[2, 3])]);
    let mut engine = Engine::new(config(1, 8, 4), executor).unwrap();
    engine.submit(request(1, &[9, 9, 9, 9, 9], 2, &[])).unwrap();
    engine.run_until_idle(4).unwrap();

    assert_eq!(engine.executor().kv_traces.len(), 2);
    assert_eq!(engine.executor().kv_traces[0], [(vec![0, 1], vec![0, 1])]);
    assert_eq!(engine.executor().kv_traces[1], [(vec![0, 1], Vec::new())]);
    assert_eq!(engine.kv_stats().used_blocks(), 0);
}

#[test]
fn early_eos_finishes_without_an_extra_decode() {
    let executor = DeterministicExecutor::scripted(&[(1, &[7])]);
    let mut engine = Engine::new(config(1, 8, 4), executor).unwrap();
    engine.submit(request(1, &[9], 8, &[7])).unwrap();
    engine.run_until_idle(4).unwrap();
    let events = engine.drain_events();
    assert!(events.iter().any(|event| {
        event.kind()
            == &EngineEventKind::Finished {
                reason: FinishReason::Stop,
            }
    }));
    assert_eq!(engine.executor().calls, 1);
    assert_eq!(engine.kv_stats().used_blocks(), 0);
}

#[test]
fn executor_failure_rolls_back_and_emits_one_terminal_error() {
    let executor = DeterministicExecutor {
        fail_call: Some(1),
        ..DeterministicExecutor::default()
    };
    let mut engine = Engine::new(config(1, 8, 4), executor).unwrap();
    engine.submit(request(1, &[1, 2, 3, 4], 4, &[])).unwrap();
    assert!(matches!(engine.step(), Err(EngineError::Executor(_))));
    let events = engine.drain_events();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event.kind(), EngineEventKind::Error { .. }))
            .count(),
        1
    );
    assert_eq!(engine.kv_stats().used_blocks(), 0);
    assert_eq!(engine.step().unwrap(), StepStatus::Idle);
    assert!(engine.drain_events().is_empty());
    engine.audit().unwrap();
}

#[test]
fn executor_panic_rolls_back_pending_state_and_the_next_step_succeeds() {
    let trace = Arc::new(PanicTrace::default());
    let executor = PanicOnceExecutor {
        panic_next: true,
        trace: Arc::clone(&trace),
    };
    let mut engine = Engine::new(config(1, 8, 4), executor).unwrap();
    engine.submit(request(1, &[1, 2], 1, &[])).unwrap();

    let panic = catch_unwind(AssertUnwindSafe(|| engine.step()));

    assert!(panic.is_err());
    assert_eq!(trace.active.load(Ordering::SeqCst), 0);
    assert_eq!(trace.committed.load(Ordering::SeqCst), 0);
    assert_eq!(trace.rolled_back.load(Ordering::SeqCst), 1);
    assert_eq!(engine.kv_stats().used_blocks(), 0);
    engine.audit().unwrap();

    assert_eq!(engine.step().unwrap(), StepStatus::Progressed);
    assert_eq!(trace.active.load(Ordering::SeqCst), 0);
    assert_eq!(trace.committed.load(Ordering::SeqCst), 1);
    assert_eq!(trace.rolled_back.load(Ordering::SeqCst), 1);
    assert_eq!(engine.kv_stats().used_blocks(), 0);
    engine.audit().unwrap();
}

#[test]
fn executor_row_mismatch_rolls_back_before_terminal_failure() {
    let mut engine = Engine::new(config(1, 8, 4), MissingRowExecutor).unwrap();
    engine.submit(request(1, &[1, 2], 2, &[])).unwrap();
    assert_eq!(engine.step(), Err(EngineError::OutputMismatch));
    let events = engine.drain_events();
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0].kind(), EngineEventKind::Error { .. }));
    assert_eq!(engine.kv_stats().used_blocks(), 0);
    assert_eq!(engine.step().unwrap(), StepStatus::Idle);
    engine.audit().unwrap();
}

#[test]
fn cancellation_with_pending_decode_work_is_isolated() {
    let executor = DeterministicExecutor::scripted(&[(1, &[1, 2, 3]), (2, &[4, 5, 6])]);
    // The helper pins the chunk limit at 8, and the chunk limit may not exceed
    // the token budget, so this scenario uses the matching 8 token budget. Its
    // items are one token each, so the schedule is unchanged.
    let mut engine = Engine::new(config(2, 8, 4), executor).unwrap();
    engine.submit(request(1, &[10], 3, &[])).unwrap();
    engine.submit(request(2, &[11], 3, &[])).unwrap();
    engine.step().unwrap();
    engine.cancel(RequestId::new(1)).unwrap();
    engine.run_until_idle(8).unwrap();
    let events = engine.drain_events();
    let first_tokens = events
        .iter()
        .filter(|event| event.request_id() == RequestId::new(1))
        .filter(|event| matches!(event.kind(), EngineEventKind::Token { .. }))
        .count();
    assert_eq!(first_tokens, 1);
    assert!(events.iter().any(|event| {
        event.request_id() == RequestId::new(1)
            && event.kind()
                == &EngineEventKind::Finished {
                    reason: FinishReason::Cancelled,
                }
    }));
    let second_tokens = events
        .iter()
        .filter(|event| event.request_id() == RequestId::new(2))
        .filter(|event| matches!(event.kind(), EngineEventKind::Token { .. }))
        .count();
    assert_eq!(second_tokens, 3);
    assert_eq!(engine.kv_stats().used_blocks(), 0);
    engine.audit().unwrap();
}

fn deterministic_trace(concurrency: usize) -> Vec<(u64, RequestId, EngineEventKind)> {
    let mut engine =
        Engine::new(config(concurrency, 32, 3), DeterministicExecutor::default()).unwrap();
    for id in 0..concurrency as u64 {
        let prompt = vec![id as u32; id as usize % 11 + 1];
        engine.submit(request(id, &prompt, 3, &[])).unwrap();
    }
    engine.run_until_idle(128).unwrap();
    assert_eq!(engine.kv_stats().used_blocks(), 0);
    engine.audit().unwrap();
    let events = engine.drain_events();
    assert!(
        events
            .windows(2)
            .all(|window| window[0].sequence() + 1 == window[1].sequence())
    );
    events
        .into_iter()
        .map(|event| (event.sequence(), event.request_id(), event.kind().clone()))
        .collect()
}

#[test]
fn c1_c4_c16_c32_event_traces_are_stable_and_terminal_retirement_is_bounded() {
    for concurrency in [1, 4, 16, 32] {
        assert_eq!(
            deterministic_trace(concurrency),
            deterministic_trace(concurrency)
        );
    }

    let mut engine = Engine::new(config(4, 8, 1), DeterministicExecutor::default()).unwrap();
    for id in 0..4 {
        engine.submit(request(id, &[id as u32], 1, &[])).unwrap();
    }
    engine.step().unwrap();
    assert_eq!(engine.terminal_count(), 3);
    engine.run_until_idle(8).unwrap();
    assert_eq!(engine.terminal_count(), 0);
    assert_eq!(engine.kv_stats().used_blocks(), 0);
}

#[derive(Clone, Copy)]
enum RetirementOutput {
    Token,
    MissingRow,
    Refuse,
    Panic,
}

struct RetirementExecutor {
    trace: Arc<std::sync::Mutex<Vec<String>>>,
    output: RetirementOutput,
    fail_retirement: bool,
    in_place: bool,
    fail_finalize: bool,
}

impl Default for RetirementExecutor {
    fn default() -> Self {
        Self {
            trace: Arc::default(),
            output: RetirementOutput::Token,
            fail_retirement: false,
            in_place: false,
            fail_finalize: false,
        }
    }
}

struct RetirementTransaction {
    trace: Arc<std::sync::Mutex<Vec<String>>>,
    fail: bool,
}

impl RetirementTransaction {
    fn finalized(&self, action: &str) -> Result<(), ExecutorError> {
        self.trace.lock().unwrap().push(action.into());
        if self.fail {
            Err(ExecutorError::new(format!("injected {action} failure")).unwrap())
        } else {
            Ok(())
        }
    }
}

impl ModelTransactionState for RetirementTransaction {
    fn commit(&mut self) -> Result<(), ExecutorError> {
        self.finalized("commit")
    }

    fn rollback(&mut self) -> Result<(), ExecutorError> {
        self.finalized("rollback")
    }
}

impl Drop for RetirementTransaction {
    fn drop(&mut self) {
        self.trace.lock().unwrap().push("drop".into());
    }
}

impl ModelExecutor for RetirementExecutor {
    fn rollback_preserves_state_v1(&self) -> bool {
        !self.in_place
    }

    type TransactionState<'a> = RetirementTransaction;

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        _kv_plan: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        self.trace.lock().unwrap().push("execute".into());
        if matches!(self.output, RetirementOutput::Refuse) {
            return Err(ExecutorError::new("injected execute refusal").unwrap());
        }
        let state = RetirementTransaction {
            trace: Arc::clone(&self.trace),
            fail: self.fail_finalize,
        };
        if matches!(self.output, RetirementOutput::Panic) {
            self.output = RetirementOutput::Token;
            let _transaction = ModelTransaction::new(Vec::new(), state);
            panic!("injected transaction panic");
        }
        let rows = items
            .iter()
            .filter(|item| {
                item.expects_logits() && !matches!(self.output, RetirementOutput::MissingRow)
            })
            .map(|item| {
                let mut logits = vec![-10.0; 32];
                logits[7] = 10.0;
                LogitsRow::new(item.request_id(), logits).unwrap()
            })
            .collect();
        Ok(ModelTransaction::new(rows, state))
    }

    fn retire_request_v1(&mut self, id: RequestId) -> Result<(), ExecutorError> {
        self.trace
            .lock()
            .unwrap()
            .push(format!("retire {}", id.get()));
        if self.fail_retirement {
            Err(ExecutorError::new("injected retirement failure").unwrap())
        } else {
            Ok(())
        }
    }
}

fn retirement_trace(engine: &Engine<RetirementExecutor>) -> Vec<String> {
    engine.executor().trace.lock().unwrap().clone()
}

#[test]
fn length_and_stop_retire_after_transaction_commit_and_drop() {
    for (maximum, stops) in [(1, vec![]), (8, vec![7])] {
        let mut engine = Engine::new(config(1, 8, 1), RetirementExecutor::default()).unwrap();
        engine.submit(request(1, &[1], maximum, &stops)).unwrap();
        engine.step().unwrap();
        assert_eq!(
            retirement_trace(&engine),
            ["execute", "commit", "drop", "retire 1"]
        );
        assert_eq!(engine.kv_stats().used_blocks(), 0);
        assert_eq!(engine.step().unwrap(), StepStatus::Idle);
        assert_eq!(retirement_trace(&engine).len(), 4);
    }
}

#[test]
fn every_completion_is_notified_before_bounded_terminal_reaping() {
    let mut engine = Engine::new(config(4, 8, 1), RetirementExecutor::default()).unwrap();
    for id in 1..=4 {
        engine.submit(request(id, &[1], 1, &[])).unwrap();
    }
    engine.step().unwrap();
    let trace = retirement_trace(&engine);
    assert_eq!(&trace[..3], ["execute", "commit", "drop"]);
    let mut retired = trace[3..].to_vec();
    retired.sort();
    assert_eq!(retired, ["retire 1", "retire 2", "retire 3", "retire 4"]);
    assert_eq!(engine.terminal_count(), 3);
    engine.run_until_idle(8).unwrap();
    assert_eq!(retirement_trace(&engine), trace);
}

#[test]
fn queued_and_live_cancellation_notify_exactly_once() {
    let mut engine = Engine::new(config(2, 8, 1), RetirementExecutor::default()).unwrap();
    engine.submit(request(1, &[1], 4, &[])).unwrap();
    engine.submit(request(2, &[2], 4, &[])).unwrap();
    engine.cancel(RequestId::new(2)).unwrap();
    assert_eq!(retirement_trace(&engine), ["retire 2"]);
    engine.step().unwrap();
    assert_eq!(
        retirement_trace(&engine),
        ["retire 2", "execute", "commit", "drop"]
    );
    assert!(engine.kv_stats().used_blocks() > 0);
    engine.cancel(RequestId::new(1)).unwrap();
    assert_eq!(
        retirement_trace(&engine),
        ["retire 2", "execute", "commit", "drop", "retire 1"]
    );
    assert!(engine.cancel(RequestId::new(1)).is_err());
    assert_eq!(engine.step().unwrap(), StepStatus::Idle);
    assert_eq!(retirement_trace(&engine).len(), 5);
    assert_eq!(engine.kv_stats().used_blocks(), 0);
}

#[test]
fn refusal_and_output_mismatch_retire_after_error_cleanup() {
    for output in [RetirementOutput::Refuse, RetirementOutput::MissingRow] {
        let executor = RetirementExecutor {
            output,
            ..RetirementExecutor::default()
        };
        let mut engine = Engine::new(config(1, 8, 1), executor).unwrap();
        engine.submit(request(1, &[1], 2, &[])).unwrap();
        let error = engine.step().unwrap_err();
        if matches!(output, RetirementOutput::Refuse) {
            assert!(matches!(error, EngineError::Executor(_)));
            assert_eq!(retirement_trace(&engine), ["execute", "retire 1"]);
        } else {
            assert_eq!(error, EngineError::OutputMismatch);
            assert_eq!(
                retirement_trace(&engine),
                ["execute", "rollback", "drop", "retire 1"]
            );
        }
        assert_eq!(engine.kv_stats().used_blocks(), 0);
        let trace = retirement_trace(&engine);
        assert_eq!(engine.step().unwrap(), StepStatus::Idle);
        assert_eq!(retirement_trace(&engine), trace);
    }
}

#[test]
fn sampling_failure_retires_after_transaction_rollback_and_drop() {
    let mut engine = Engine::new(config(1, 8, 1), RetirementExecutor::default()).unwrap();
    engine
        .submit(
            EngineRequest::new(
                RequestId::new(1),
                vec![1],
                2,
                SamplingParameters::new(0.0)
                    .unwrap()
                    .with_minimum_new_tokens(1)
                    .unwrap(),
                1,
                (0..32).collect(),
                ModelVocabulary::new(32).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    assert!(matches!(engine.step(), Err(EngineError::Sampling(_))));
    assert_eq!(
        retirement_trace(&engine),
        ["execute", "rollback", "drop", "retire 1"]
    );
    assert_eq!(engine.kv_stats().used_blocks(), 0);
    engine.audit().unwrap();
}

/// A failed model commit or rollback leaves executor state uncertain, so the
/// step fails with an executor error, the batch publishes no token, and the
/// engine executes nothing more. Cancellation still retires requests.
#[test]
fn a_failed_model_finalization_publishes_nothing_and_blocks_execution() {
    let invalid_sampling = || {
        EngineRequest::new(
            RequestId::new(3),
            vec![1],
            2,
            SamplingParameters::new(0.0)
                .unwrap()
                .with_minimum_new_tokens(1)
                .unwrap(),
            1,
            (0..32).collect(),
            ModelVocabulary::new(32).unwrap(),
        )
        .unwrap()
    };
    for rollback in [false, true] {
        let executor = RetirementExecutor {
            fail_finalize: true,
            ..RetirementExecutor::default()
        };
        let mut engine = Engine::new(config(3, 16, 3), executor).unwrap();
        engine.submit(request(1, &[1], 1, &[])).unwrap();
        engine.submit(request(2, &[2], 4, &[])).unwrap();
        if rollback {
            engine.submit(invalid_sampling()).unwrap();
        }
        assert!(matches!(engine.step(), Err(EngineError::Executor(_))));
        let finalized = if rollback { "rollback" } else { "commit" };
        assert!(retirement_trace(&engine).contains(&finalized.to_owned()));
        assert!(engine.drain_events().iter().all(|event| !matches!(
            event.kind(),
            EngineEventKind::Token { .. }
                | EngineEventKind::Finished {
                    reason: FinishReason::Length | FinishReason::Stop
                }
        )));
        assert_eq!(engine.step(), Err(EngineError::ModelStateFailed));
        assert!(
            engine
                .metrics()
                .prometheus_compatibility("fixture")
                .contains("chop_instance_poisoned{model_name=\"fixture\",engine=\"0\"} 1")
        );
        for id in 1..=3 {
            let _ = engine.cancel(RequestId::new(id));
        }
        assert_eq!(engine.kv_stats().used_blocks(), 0);
    }
}

#[test]
fn sampling_failure_isolates_one_request_in_a_mixed_batch() {
    let mut engine = Engine::new(config(2, 16, 2), RetirementExecutor::default()).unwrap();
    let invalid = EngineRequest::new(
        RequestId::new(1),
        vec![1],
        2,
        SamplingParameters::new(0.0)
            .unwrap()
            .with_minimum_new_tokens(1)
            .unwrap(),
        1,
        (0..32).collect(),
        ModelVocabulary::new(32).unwrap(),
    )
    .unwrap();
    engine.submit(invalid).unwrap();
    engine.submit(request(2, &[2], 1, &[])).unwrap();

    assert!(matches!(engine.step(), Err(EngineError::Sampling(_))));
    let failed = engine.drain_events();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].request_id(), RequestId::new(1));
    engine.run_until_idle(8).unwrap();
    assert!(engine.drain_events().iter().any(|event| {
        event.request_id() == RequestId::new(2)
            && matches!(event.kind(), EngineEventKind::Finished { .. })
    }));
    engine.audit().unwrap();
}

#[test]
fn rollback_of_a_live_request_does_not_notify_retirement() {
    let executor = RetirementExecutor {
        output: RetirementOutput::Panic,
        ..RetirementExecutor::default()
    };
    let mut engine = Engine::new(config(1, 8, 1), executor).unwrap();
    engine.submit(request(1, &[1], 1, &[])).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| engine.step())).is_err());
    assert_eq!(retirement_trace(&engine), ["execute", "rollback", "drop"]);
    engine.step().unwrap();
    assert_eq!(
        retirement_trace(&engine),
        [
            "execute", "rollback", "drop", "execute", "commit", "drop", "retire 1"
        ]
    );
}

#[test]
fn retirement_failure_permanently_blocks_steps_after_completion_or_cancellation() {
    for cancel in [false, true] {
        let executor = RetirementExecutor {
            fail_retirement: true,
            ..RetirementExecutor::default()
        };
        let mut engine = Engine::new(config(2, 8, 1), executor).unwrap();
        engine.submit(request(1, &[1], 1, &[])).unwrap();
        engine.submit(request(2, &[2; 16], 4, &[])).unwrap();
        let result = if cancel {
            engine.cancel(RequestId::new(1))
        } else {
            engine.step().map(|_| ())
        };
        assert!(matches!(result, Err(EngineError::Executor(_))));
        let trace = retirement_trace(&engine);
        let used = engine.kv_stats().used_blocks();
        let terminals = engine.terminal_count();
        for _ in 0..3 {
            assert_eq!(engine.step(), Err(EngineError::RetirementFailed));
            assert!(
                engine
                    .metrics()
                    .prometheus_compatibility("fixture")
                    .contains("chop_instance_poisoned{model_name=\"fixture\",engine=\"0\"} 1")
            );
            assert_eq!(retirement_trace(&engine), trace);
            assert_eq!(engine.kv_stats().used_blocks(), used);
            assert_eq!(engine.terminal_count(), terminals);
        }
    }
}

#[test]
fn retirement_failure_does_not_suppress_other_completion_notifications() {
    let executor = RetirementExecutor {
        fail_retirement: true,
        ..RetirementExecutor::default()
    };
    let mut engine = Engine::new(config(4, 8, 1), executor).unwrap();
    for id in 1..=4 {
        engine.submit(request(id, &[1], 1, &[])).unwrap();
    }
    assert!(matches!(engine.step(), Err(EngineError::Executor(_))));
    let trace = retirement_trace(&engine);
    let mut retired = trace[3..].to_vec();
    retired.sort();
    assert_eq!(retired, ["retire 1", "retire 2", "retire 3", "retire 4"]);
    assert_eq!(engine.step(), Err(EngineError::RetirementFailed));
    assert!(
        engine
            .metrics()
            .prometheus_compatibility("fixture")
            .contains("chop_instance_poisoned{model_name=\"fixture\",engine=\"0\"} 1")
    );
    assert_eq!(retirement_trace(&engine), trace);
}

#[test]
fn a_refused_batch_fails_only_its_items_and_releases_all_pages() {
    let executor = DeterministicExecutor {
        refuse_call: Some(1),
        ..Default::default()
    };
    let mut engine = Engine::new(config(3, 8, 4), executor).unwrap();
    for id in 1..=3 {
        engine.submit(request(id, &[1; 4], 1, &[])).unwrap();
    }
    assert_eq!(engine.step().unwrap(), StepStatus::Progressed);
    engine.audit().unwrap();
    let errors: Vec<_> = engine.drain_events().into_iter().filter_map(|event| {
        matches!(event.kind(), EngineEventKind::Error { message } if message.contains("packing check"))
            .then_some(event.request_id())
    }).collect();
    assert_eq!(errors, vec![RequestId::new(1), RequestId::new(2)]);
    engine.submit(request(4, &[2], 1, &[])).unwrap();
    engine.run_until_idle(10).unwrap();
    engine.audit().unwrap();
    let finished: Vec<_> = engine
        .drain_events()
        .into_iter()
        .filter_map(|event| {
            matches!(event.kind(), EngineEventKind::Finished { .. }).then_some(event.request_id())
        })
        .collect();
    assert_eq!(finished, vec![RequestId::new(3), RequestId::new(4)]);
    assert_eq!(engine.kv_stats().used_blocks(), 0);
    assert_eq!(engine.metrics().snapshot().requests_active, 0);
}

#[test]
fn repeated_batch_refusals_trip_the_guard_and_block_further_execution() {
    let executor = DeterministicExecutor {
        refuse_call: Some(0),
        ..Default::default()
    };
    let mut engine = Engine::new(
        config(1, 8, 4).with_batch_refusal_limits(runtime_engine::BatchRefusalLimits {
            maximum_consecutive: 1,
            maximum_in_window: 10,
            window_seconds: 60,
        }),
        executor,
    )
    .unwrap();
    engine.submit(request(1, &[1], 1, &[])).unwrap();
    engine.step().unwrap();
    engine.submit(request(2, &[1], 1, &[])).unwrap();
    assert!(
        matches!(engine.step(), Err(EngineError::Executor(error)) if error.message().contains("consecutive refusal limit"))
    );
    engine.audit().unwrap();
    assert_eq!(engine.kv_stats().used_blocks(), 0);
    assert!(matches!(engine.step(), Err(EngineError::ModelStateFailed)));
    assert_eq!(engine.executor().calls, 2);
    let text = engine.metrics().prometheus_compatibility("fixture");
    assert!(text.contains("vllm:batch_refusals_total{model_name=\"fixture\",engine=\"0\"} 2"));
    assert!(
        text.contains("vllm:batch_refused_requests_total{model_name=\"fixture\",engine=\"0\"} 2")
    );
}

/// An in-place executor cannot roll a step back, so a request whose
/// sampling fails must not take its batch with it: the step commits, the
/// failed requests (one with room left, one whose placeholder row reaches
/// its length) end with their own errors and retire, and the healthy request
/// publishes its token and runs to completion.
#[test]
fn sampling_failure_fails_only_the_failing_in_place_requests() {
    let mut engine = Engine::new(
        config(3, 16, 3),
        RetirementExecutor {
            in_place: true,
            ..RetirementExecutor::default()
        },
    )
    .unwrap();
    let invalid = |id, maximum| {
        EngineRequest::new(
            RequestId::new(id),
            vec![1],
            maximum,
            SamplingParameters::new(0.0)
                .unwrap()
                .with_minimum_new_tokens(1)
                .unwrap(),
            1,
            (0..32).collect(),
            ModelVocabulary::new(32).unwrap(),
        )
        .unwrap()
    };
    engine.submit(invalid(1, 2)).unwrap();
    engine.submit(request(2, &[2], 3, &[])).unwrap();
    engine.submit(invalid(3, 1)).unwrap();

    assert!(matches!(engine.step(), Err(EngineError::Sampling(_))));
    let events = engine.drain_events();
    let kinds = |id| {
        events
            .iter()
            .filter(|event| event.request_id() == RequestId::new(id))
            .map(|event| event.kind().clone())
            .collect::<Vec<_>>()
    };
    for id in [1, 3] {
        assert!(
            matches!(kinds(id)[..], [EngineEventKind::Error { .. }]),
            "{id}"
        );
    }
    assert!(matches!(
        kinds(2)[..],
        [EngineEventKind::Token { token_id: 7, .. }]
    ));
    let mut trace = retirement_trace(&engine);
    assert_eq!(&trace[..3], ["execute", "commit", "drop"]);
    trace[3..].sort();
    assert_eq!(&trace[3..], ["retire 1", "retire 3"]);

    engine.run_until_idle(8).unwrap();
    let events = engine.drain_events();
    assert!(
        events
            .iter()
            .all(|event| event.request_id() == RequestId::new(2))
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event.kind(), EngineEventKind::Token { .. }))
            .count(),
        2
    );
    assert!(matches!(
        events.last().unwrap().kind(),
        EngineEventKind::Finished {
            reason: FinishReason::Length
        }
    ));
    assert_eq!(retirement_trace(&engine).last().unwrap(), "retire 2");
    assert!(!retirement_trace(&engine).contains(&"rollback".to_owned()));
    assert_eq!(engine.kv_stats().used_blocks(), 0);
    engine.audit().unwrap();
}
