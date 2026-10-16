//! Preempted requests recompute to exactly what they would have produced.
//!
//! The model's logits depend on every token a request committed, and its
//! state lives in the executor, as a sliding-window ring or a recurrent state
//! does. A preemption retires that state, so a recompute that fed the wrong
//! tokens, resampled an emitted token or skipped the retirement would change
//! the output or trip the executor's own checks.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use runtime_engine::{
    DraftRow, DraftSource, Engine, EngineConfig, EngineError, EngineEventKind, EngineRequest,
    ExecutorError, FinishReason, KeptRows, LogitsRow, ModelBatchItem, ModelExecutor,
    ModelTransaction, ModelTransactionState, SampledRow, SpeculationConfig, SpeculationMode,
    StepStatus,
};
use runtime_grammar::{
    CompiledGrammar, ConstraintState, JsonOptions, TokenVocabulary, VocabularyToken,
    json_schema_grammar,
};
use runtime_kv::{BlockId, KvGeometry, PreparedAppendBatch};
use runtime_model_api::ModelVocabulary;
use runtime_sampling::{SamplingParameters, sample_for_vocabulary};
use runtime_scheduler::{
    AdmissionPolicy, BatchPhase, OptimisticAdmission, RequestId, RequestState, SchedulerConfig,
};
use serde_json::json;

/// Token texts for the grammar; the stop token has no bytes.
const TEXTS: [&str; 24] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "[", "]", ",", " ", "[1", "1,", "],", "0]",
    ", 4", "x", "\"", "", "12", "a",
];
const STOP: u32 = 21;
const VOCABULARY: usize = 24;
const BLOCK: usize = 4;

fn vocabulary() -> ModelVocabulary {
    ModelVocabulary::new(VOCABULARY).unwrap()
}

fn grammar() -> Arc<CompiledGrammar> {
    let tokens = TEXTS
        .iter()
        .enumerate()
        .map(|(id, text)| VocabularyToken {
            bytes: text.as_bytes().to_vec(),
            special: id as u32 == STOP,
        })
        .collect();
    let vocabulary = Arc::new(TokenVocabulary::new(VOCABULARY, tokens).unwrap());
    let schema = json!({"type": "array", "items": {"type": "integer"}, "maxItems": 6});
    let grammar = json_schema_grammar(&schema, JsonOptions::default()).unwrap();
    Arc::new(CompiledGrammar::new(&grammar, vocabulary).unwrap())
}

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Hash noise over the whole prefix; the stop token is a little unlikely.
fn logits_for(prefix: &[u32]) -> Vec<f32> {
    let hash = prefix
        .iter()
        .fold(0x1234_5678_u64, |hash, token| mix(hash ^ u64::from(*token)));
    let mut logits: Vec<f32> = (0..VOCABULARY as u64)
        .map(|token| (mix(hash ^ (token << 40)) >> 40) as f32 / (1u64 << 24) as f32 * 6.0 - 3.0)
        .collect();
    logits[STOP as usize] -= 0.8;
    logits
}

#[derive(Default, Debug)]
struct Model {
    offload: Option<runtime_engine::KvOffloadInfo>,
    transfers: BTreeMap<runtime_kv::KvTransferId, (runtime_kv::KvTransferSubmission, usize)>,
    host: BTreeMap<runtime_kv::HostPageId, Vec<u32>>,
    held: BTreeSet<runtime_kv::KvTransferId>,
    delay: usize,
    failure: Option<&'static str>,
    polls: Option<Arc<std::sync::atomic::AtomicUsize>>,
    panic_poll: bool,
    mismatched_completion: bool,
    draft_failure: bool,
    drafted_since_execute: BTreeSet<RequestId>,

    /// Rows come back sampled, as a device sampler returns them.
    sampled: bool,
    /// Distinguish a prefill row from decode even for identical token history.
    phase_sensitive: bool,
    /// Committed tokens per request: the executor's own state.
    state: BTreeMap<RequestId, Vec<u32>>,
    pending: Vec<(RequestId, Vec<u32>, bool)>,
    admitted: BTreeSet<RequestId>,
    admissions: BTreeMap<RequestId, usize>,
    snapshots: BTreeMap<BlockId, Vec<u32>>,
    captures: Vec<(RequestId, usize)>,
    restores: Vec<(RequestId, usize)>,
    restored_pending: BTreeSet<RequestId>,
    resumed_drafts: usize,
    /// Retirements of requests admitted and never executed.
    retired_unexecuted: usize,
}

struct Pending<'a> {
    model: &'a mut Model,
}

impl ModelTransactionState for Pending<'_> {
    fn commit(&mut self) -> Result<(), ExecutorError> {
        self.commit_kept(&[])
    }

    fn commit_kept(&mut self, kept: &[KeptRows]) -> Result<(), ExecutorError> {
        let mut kept = kept.iter();
        for (id, inputs, verify) in self.model.pending.drain(..) {
            let count = if verify {
                kept.next().unwrap().kept_rows
            } else {
                inputs.len()
            };
            self.model
                .state
                .entry(id)
                .or_default()
                .extend_from_slice(&inputs[..count]);
        }
        Ok(())
    }

    fn rollback(&mut self) -> Result<(), ExecutorError> {
        self.model.pending.clear();
        Ok(())
    }
}

impl ModelExecutor for Model {
    type TransactionState<'a> = Pending<'a>;
    fn kv_offload_info_v1(&self) -> Option<runtime_engine::KvOffloadInfo> {
        self.offload
    }
    fn offload_pages_v1(
        &mut self,
        plan: runtime_kv::KvTransferPlan,
    ) -> runtime_engine::KvTransferSubmitResult {
        if self.failure == Some("out_fatal") {
            return Err((ExecutorError::fatal("fatal submit refusal").unwrap(), plan));
        }
        if self.failure == Some("out_refused") {
            return Err((ExecutorError::new("refused").unwrap(), plan));
        }
        self.transfers
            .insert(plan.id(), (plan.into_submission(), self.delay));
        Ok(())
    }
    fn reload_pages_v1(
        &mut self,
        plan: runtime_kv::KvTransferPlan,
    ) -> runtime_engine::KvTransferSubmitResult {
        if self.failure == Some("in_refused") {
            return Err((ExecutorError::new("refused").unwrap(), plan));
        }
        self.transfers
            .insert(plan.id(), (plan.into_submission(), self.delay));
        Ok(())
    }
    fn poll_kv_transfer_v1(
        &mut self,
        id: runtime_kv::KvTransferId,
    ) -> Result<runtime_engine::KvTransferPoll, ExecutorError> {
        use runtime_kv::{KvTransferDirection, KvTransferPoll, KvTransferStatus};
        if let Some(polls) = &self.polls {
            polls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        assert!(!self.panic_poll, "injected poll panic");
        if self.mismatched_completion {
            self.mismatched_completion = false;
            let (other, _) = self
                .transfers
                .iter()
                .find(|(key, _)| **key != id)
                .unwrap()
                .1;
            return Ok(KvTransferPoll::Finished(
                other.completion(KvTransferStatus::Complete),
            ));
        }
        if self.held.contains(&id) {
            return Ok(KvTransferPoll::Pending);
        }
        if self.failure == Some("poll") {
            return Err(ExecutorError::new("unexpected poll error").unwrap());
        }
        let (transfer, delay) = self.transfers.get_mut(&id).unwrap();
        if *delay > 0 {
            *delay -= 1;
            return Ok(KvTransferPoll::Pending);
        }
        let status = match (self.failure, transfer.direction()) {
            (Some("unconfirmed"), _) | (Some("in_unconfirmed"), KvTransferDirection::In) => {
                KvTransferStatus::Unconfirmed
            }
            (Some("out_failed"), KvTransferDirection::Out)
            | (Some("in_failed"), KvTransferDirection::In) => KvTransferStatus::FailedIdle,
            _ => KvTransferStatus::Complete,
        };
        if status == KvTransferStatus::Complete {
            for pair in transfer.pages() {
                match transfer.direction() {
                    KvTransferDirection::Out => {
                        self.host.insert(
                            pair.host,
                            self.snapshots
                                .get(&pair.device)
                                .cloned()
                                .unwrap_or_default(),
                        );
                    }
                    KvTransferDirection::In => {
                        self.snapshots
                            .insert(pair.device, self.host[&pair.host].clone());
                    }
                }
            }
        }
        Ok(KvTransferPoll::Finished(transfer.completion(status)))
    }
    fn wait_kv_transfer_v1(
        &mut self,
        id: runtime_kv::KvTransferId,
    ) -> Result<runtime_engine::KvTransferPoll, ExecutorError> {
        self.transfers.get_mut(&id).unwrap().1 = 0;
        self.poll_kv_transfer_v1(id)
    }
    fn drain_kv_transfers_v1(&mut self) -> Result<(), ExecutorError> {
        self.held.clear();
        for (_, delay) in self.transfers.values_mut() {
            *delay = 0;
        }
        Ok(())
    }

    fn admit_request_v1(&mut self, id: RequestId) -> Result<(), ExecutorError> {
        if self.failure == Some("admit") {
            return Err(ExecutorError::new("admission failed").unwrap());
        }
        assert!(
            self.admitted.insert(id),
            "a request is admitted again only after it was retired"
        );
        *self.admissions.entry(id).or_default() += 1;
        Ok(())
    }

    fn retire_request_v1(&mut self, id: RequestId) -> Result<(), ExecutorError> {
        if self.admitted.remove(&id) && self.state.remove(&id).is_none() {
            self.retired_unexecuted += 1;
        }
        self.state.remove(&id);
        Ok(())
    }

    fn capture_prefix_state_v1(
        &mut self,
        id: RequestId,
        tokens: usize,
        pages: &[BlockId],
    ) -> Result<(), ExecutorError> {
        if self.failure == Some("capture") {
            return Err(ExecutorError::new("refused capture").unwrap());
        }
        let state = self.state[&id].clone();
        assert_eq!(state.len(), tokens, "a capture names the committed length");
        self.snapshots.insert(pages[0], state);
        self.captures.push((id, tokens));
        Ok(())
    }

    fn restore_prefix_state_v1(
        &mut self,
        id: RequestId,
        prefix: &[u32],
        pages: &[BlockId],
    ) -> Result<(), ExecutorError> {
        if self.failure == Some("restore_fatal") {
            return Err(ExecutorError::fatal("restore did not finish").unwrap());
        }
        if self.failure == Some("restore") {
            return Err(ExecutorError::new("refused restore").unwrap());
        }
        assert!(self.admitted.contains(&id), "admission precedes restore");
        assert!(!self.state.contains_key(&id), "restore precedes execution");
        assert_eq!(self.snapshots[&pages[0]], prefix);
        self.state.insert(id, prefix.to_vec());
        self.restores.push((id, prefix.len()));
        self.restored_pending.insert(id);
        Ok(())
    }

    fn draft_tokens_v1(
        &mut self,
        id: RequestId,
        history: &[u32],
        maximum: usize,
    ) -> Result<Vec<u32>, ExecutorError> {
        assert!(
            self.drafted_since_execute.insert(id),
            "a bundle draft was repeated without execute"
        );
        if self.draft_failure {
            return Err(ExecutorError::new("draft failed").unwrap());
        }
        assert_eq!(self.state[&id], history[..history.len() - 1]);
        if self.restored_pending.contains(&id) {
            self.resumed_drafts += 1;
        }
        // Drafts that are sometimes the greedy continuation.
        let mut context = history.to_vec();
        let mut drafts = Vec::new();
        for index in 0..maximum {
            let logits = logits_for(&context);
            let best = (0..VOCABULARY)
                .max_by(|a, b| logits[*a].total_cmp(&logits[*b]))
                .unwrap() as u32;
            let token = if mix(context.len() as u64 + index as u64) % 3 == 0 {
                (best + 1) % VOCABULARY as u32
            } else {
                best
            };
            drafts.push(token);
            context.push(token);
        }
        Ok(drafts)
    }

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        kv_plan: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        self.drafted_since_execute.clear();
        let mut logits_rows = Vec::new();
        let mut prompt_rows = Vec::new();
        let mut sampled_rows = Vec::new();
        let mut draft_rows = Vec::new();
        for (item, append) in items.iter().zip(kv_plan.appends()) {
            assert!(self.admitted.contains(&item.request_id()));
            if self.restored_pending.contains(&item.request_id()) && self.phase_sensitive {
                assert!(!matches!(item.phase(), BatchPhase::Prefill { .. }));
            }
            let committed = self
                .state
                .get(&item.request_id())
                .cloned()
                .unwrap_or_default();
            assert_eq!(
                committed.len(),
                append.previous_tokens(),
                "model state matches the KV"
            );
            let mut inputs = item.input_tokens().to_vec();
            if let Some(draft) = item
                .draft()
                .filter(|draft| draft.source() == DraftSource::Bundle)
            {
                let tokens = self.draft_tokens_v1(
                    item.request_id(),
                    &item.sampling().unwrap().history.to_vec(),
                    draft.tokens(),
                )?;
                inputs.extend_from_slice(&tokens);
                draft_rows.push(DraftRow::new(item.request_id(), tokens));
            }
            self.restored_pending.remove(&item.request_id());
            assert_eq!(inputs.len(), append.additional_tokens());
            let drafts = if item.draft().is_some() {
                inputs[1..].to_vec()
            } else {
                Vec::new()
            };
            for position in 0..item.prompt_targets().len() {
                let mut prefix = committed.clone();
                prefix.extend_from_slice(&inputs[..=position]);
                prompt_rows.push(LogitsRow::new(item.request_id(), logits_for(&prefix)).unwrap());
            }
            if let Some(work) = item.sampling() {
                let first_row = if item.draft().is_some() {
                    0
                } else {
                    inputs.len() - 1
                };
                for input in first_row..inputs.len() {
                    let mut prefix = committed.clone();
                    prefix.extend_from_slice(&inputs[..=input]);
                    let mut logits = logits_for(&prefix);
                    if self.phase_sensitive && matches!(item.phase(), BatchPhase::Prefill { .. }) {
                        logits[0] += 0.25;
                    }
                    if !self.sampled {
                        logits_rows.push(LogitsRow::new(item.request_id(), logits).unwrap());
                        continue;
                    }
                    let row = work.for_verify_row(&drafts, input - first_row);
                    let context = row.history.context(row.emitted_tokens, &row.stop_ids);
                    let context = match row.constraint.as_ref().and_then(|row| row.mask()) {
                        Some(mask) => context.with_allowed_tokens(mask),
                        None => context,
                    };
                    let mut state = row.state;
                    let sample = sample_for_vocabulary(
                        &row.parameters,
                        &logits,
                        vocabulary(),
                        context,
                        &mut state,
                    )
                    .unwrap();
                    sampled_rows.push(SampledRow::new(
                        item.request_id(),
                        sample.token_id(),
                        sample.logprob(),
                        sample.top_logprobs().to_vec(),
                    ));
                }
            }
            self.pending
                .push((item.request_id(), inputs, item.draft().is_some()));
        }
        self.drafted_since_execute.clear();
        Ok(ModelTransaction::new(logits_rows, Pending { model: self })
            .with_prompt_rows(prompt_rows)
            .with_sampled(sampled_rows)
            .with_drafts(draft_rows))
    }
}

#[derive(Clone, Copy, Debug)]
struct Setup {
    pages: usize,
    admission: AdmissionPolicy,
    prefix: bool,
    speculation: bool,
    sampled: bool,
}

fn optimistic(maximum_preemptions: usize) -> AdmissionPolicy {
    AdmissionPolicy::Optimistic(OptimisticAdmission::new(0, maximum_preemptions).unwrap())
}

fn forced(every: usize, maximum_preemptions: usize) -> AdmissionPolicy {
    AdmissionPolicy::Optimistic(
        OptimisticAdmission::new(0, maximum_preemptions)
            .unwrap()
            .with_forced_preemption(every)
            .unwrap(),
    )
}

fn engine(setup: Setup) -> Engine<Model> {
    engine_with_scoring(setup, 4, 512)
}

fn engine_with_scoring(setup: Setup, chunk: usize, cap: usize) -> Engine<Model> {
    engine_with_scoring_source(setup, chunk, cap, DraftSource::Host)
}

fn engine_with_source(setup: Setup, source: DraftSource) -> Engine<Model> {
    engine_with_scoring_source(setup, 4, 512, source)
}

fn engine_with_scoring_source(
    setup: Setup,
    chunk: usize,
    cap: usize,
    source: DraftSource,
) -> Engine<Model> {
    engine_with_offload(setup, chunk, cap, source, None, false, None)
}

#[derive(Clone, Copy, Debug)]
struct DraftPolicy(DraftSource, runtime_engine::DraftCostModel);
impl From<DraftSource> for DraftPolicy {
    fn from(source: DraftSource) -> Self {
        Self(source, runtime_engine::DraftCostModel::PerToken)
    }
}

fn engine_with_offload(
    setup: Setup,
    chunk: usize,
    cap: usize,
    source: impl Into<DraftPolicy>,
    offload: Option<runtime_engine::KvOffloadInfo>,
    force: bool,
    failure: Option<&'static str>,
) -> Engine<Model> {
    let source = source.into();
    let mut config = EngineConfig::new(
        SchedulerConfig::new(6, 64, chunk.max(24), chunk, 1)
            .unwrap()
            .with_maximum_scored_rows(cap)
            .unwrap(),
        KvGeometry::new(BLOCK, setup.pages).unwrap(),
        vocabulary(),
        (setup.pages * BLOCK).min(96),
        64,
    )
    .unwrap()
    .with_admission(setup.admission);
    if setup.prefix {
        config = config.with_prefix_cache(1);
    }
    if setup.speculation {
        config = config.with_speculation(
            SpeculationConfig::new(3, 16, SpeculationMode::Fixed, source.0)
                .unwrap()
                .with_draft_cost_model(source.1),
        );
    }
    if force {
        config = config.with_forced_swaps();
    }
    Engine::new(
        config,
        Model {
            offload,
            delay: 2,
            failure,
            sampled: setup.sampled,
            ..Model::default()
        },
    )
    .unwrap()
}

/// Greedy, seeded, penalized and constrained requests with prompts of 3 to
/// 17 tokens and a 60-token limit most stop well before.
fn requests() -> Vec<EngineRequest> {
    let grammar = grammar();
    (0..10_u64)
        .map(|index| {
            let prompt: Vec<u32> = (0..3 + (index * 5) % 15)
                .map(|position| (mix(index * 31 + position) % 20) as u32)
                .collect();
            let parameters = match index % 4 {
                0 => SamplingParameters::new(0.0)
                    .unwrap()
                    .with_top_logprobs(3)
                    .unwrap(),
                1 => SamplingParameters::new(0.8).unwrap(),
                2 => SamplingParameters::new(1.1)
                    .unwrap()
                    .with_penalties(0.2, 0.1, 1.1)
                    .unwrap(),
                _ => SamplingParameters::new(0.7)
                    .unwrap()
                    .with_top_p(0.9)
                    .unwrap(),
            };
            let request = EngineRequest::new(
                RequestId::new(index + 1),
                prompt,
                60,
                parameters,
                mix(index) ^ 0xfeed,
                vec![STOP],
                vocabulary(),
            )
            .unwrap();
            if index % 3 == 2 {
                request
                    .with_constraint(ConstraintState::new(Arc::clone(&grammar), None, vec![STOP]))
                    .unwrap()
            } else {
                request
            }
        })
        .collect()
}

/// Token IDs and exact floating-point encodings of the sampled row.
type TokenEventBits = (u32, u64, Vec<(u32, u64)>);

#[derive(Clone, Debug, PartialEq)]
struct Output {
    tokens: Vec<TokenEventBits>,
    finish: Option<FinishReason>,
    prompt: Vec<EngineEventKind>,
}

type Outputs = BTreeMap<RequestId, Output>;

/// Runs every request, checking the event order and the engine's audit
/// after every step, and returns what each request streamed.
fn run(engine: &mut Engine<Model>, requests: &[EngineRequest]) -> Result<Outputs, EngineError> {
    run_with_tick(engine, requests, |_| {})
}
fn run_with_tick(
    engine: &mut Engine<Model>,
    requests: &[EngineRequest],
    mut tick: impl FnMut(&mut Engine<Model>),
) -> Result<Outputs, EngineError> {
    for request in requests {
        engine.submit(request.clone()).unwrap();
    }
    let mut outputs = Outputs::new();
    let mut cached = BTreeSet::new();
    for _ in 0..20_000 {
        tick(engine);
        let status = engine.step()?;
        engine.audit().unwrap();
        if !engine.prefix_cache_enabled() {
            let kv = engine.kv_metrics().snapshot();
            assert_eq!(
                (
                    kv.prefix_cache_queries,
                    kv.prefix_cache_hits,
                    kv.prefix_cache_reused_tokens,
                    kv.prefix_cache_queried_tokens
                ),
                (0, 0, 0, 0)
            );
            assert_eq!(
                (
                    kv.prefix_cache_captures,
                    kv.prefix_cache_capture_failures,
                    kv.prefix_cache_restore_failures,
                    kv.prefix_cache_evictions
                ),
                (0, 0, 0, 0)
            );
        }
        for event in engine.drain_events() {
            let output = outputs.entry(event.request_id()).or_insert(Output {
                tokens: Vec::new(),
                finish: None,
                prompt: Vec::new(),
            });
            assert!(output.finish.is_none(), "nothing follows the finish");
            match event.kind() {
                EngineEventKind::PromptCached { .. } => {
                    assert!(output.tokens.is_empty(), "cached tokens precede tokens");
                    assert!(cached.insert(event.request_id()), "reported once");
                }
                EngineEventKind::Token {
                    token_id,
                    logprob,
                    top_logprobs,
                } => output.tokens.push((
                    *token_id,
                    logprob.to_bits(),
                    top_logprobs
                        .iter()
                        .map(|top| (top.token_id(), top.logprob().to_bits()))
                        .collect(),
                )),
                EngineEventKind::Finished { reason } => output.finish = Some(*reason),
                kind @ EngineEventKind::PromptLogprobs { .. } => output.prompt.push(kind.clone()),
                kind => panic!("unexpected event {kind:?}"),
            }
        }
        if status == StepStatus::Idle {
            assert!(engine.executor().admitted.is_empty());
            assert!(engine.executor().state.is_empty());
            let stats = engine.kv_stats();
            assert_eq!(stats.used_blocks(), 0);
            assert_eq!(stats.active_sequences(), 0);
            return Ok(outputs);
        }
    }
    panic!("the engine did not finish");
}

fn preemptions(engine: &Engine<Model>) -> u64 {
    let text = engine.metrics().prometheus_compatibility("fixture");
    let line = text
        .lines()
        .find(|line| line.starts_with("vllm:num_preemptions_total{"))
        .unwrap();
    line.rsplit(' ').next().unwrap().parse().unwrap()
}

fn baseline(setup: Setup) -> Outputs {
    let mut engine = engine(Setup {
        pages: 512,
        admission: AdmissionPolicy::Reserved,
        ..setup
    });
    let outputs = run(&mut engine, &requests()).unwrap();
    assert_eq!(preemptions(&engine), 0);
    outputs
}

#[test]
fn preempted_requests_stream_what_unpreempted_runs_stream() {
    for prefix in [false, true] {
        for speculation in [false, true] {
            for sampled in [false, true] {
                let base = Setup {
                    pages: 512,
                    admission: AdmissionPolicy::Reserved,
                    prefix,
                    speculation,
                    sampled,
                };
                let expected = baseline(base);
                assert_eq!(expected.len(), 10);
                assert!(expected.values().all(|output| output.finish.is_some()));
                for (pages, admission) in [(24, optimistic(3)), (512, forced(5, 2))] {
                    let setup = Setup {
                        pages,
                        admission,
                        ..base
                    };
                    let mut engine = engine(setup);
                    let outputs = run(&mut engine, &requests()).unwrap();
                    assert!(preemptions(&engine) > 0, "{setup:?}");
                    assert_eq!(outputs, expected, "{setup:?}");
                }
            }
        }
    }
}

#[test]
fn constrained_output_stays_valid_after_recompute() {
    let setup = Setup {
        pages: 512,
        admission: forced(3, 3),
        prefix: false,
        speculation: true,
        sampled: false,
    };
    let mut engine = engine(setup);
    let outputs = run(&mut engine, &requests()).unwrap();
    assert!(preemptions(&engine) > 0);
    let mut parsed = 0;
    for (id, output) in outputs {
        if (id.get() - 1) % 3 != 2 {
            continue;
        }
        let text: String = output
            .tokens
            .iter()
            .filter(|(token, _, _)| *token != STOP)
            .map(|(token, _, _)| TEXTS[*token as usize])
            .collect();
        if output.finish == Some(FinishReason::Stop) {
            let value: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert!(value.is_array(), "{text}");
            parsed += 1;
        }
    }
    assert!(parsed > 0);
    let model = engine.executor();
    assert!(
        [3, 6, 9]
            .iter()
            .any(|id| model.admissions[&RequestId::new(*id)] > 1),
        "a constrained request was preempted"
    );
}

#[test]
fn each_recompute_restores_the_prefix_its_preemption_published() {
    let setup = Setup {
        pages: 512,
        admission: forced(4, 2),
        prefix: true,
        speculation: false,
        sampled: false,
    };
    let expected = baseline(setup);
    let mut engine = engine(setup);
    let outputs = run(&mut engine, &requests()).unwrap();
    assert_eq!(outputs, expected);
    let model = engine.executor();
    // Captures at prompt ends and shared prefixes come from first
    // admissions; the preemption captures are the ones restored at their
    // own length.
    for (id, tokens) in &model.restores {
        assert!(model.captures.contains(&(*id, *tokens)));
    }
    let preempted = preemptions(&engine) as usize;
    assert!(preempted > 0);
    assert_eq!(
        model.restores.len(),
        preempted,
        "every recompute restored its prefix: {:?}",
        model.restores
    );
    for (id, admissions) in &model.admissions {
        assert!(
            *admissions <= 3,
            "request {id:?} was admitted {admissions} times"
        );
    }
}

#[test]
fn prompt_logprob_requests_are_never_preempted() {
    let mut engine = engine(Setup {
        pages: 512,
        admission: forced(2, 5),
        prefix: false,
        speculation: false,
        sampled: false,
    });
    let pinned = requests()[0].clone().with_prompt_logprobs(2);
    let other = requests()[1].clone();
    engine.submit(pinned).unwrap();
    engine.submit(other).unwrap();
    let mut prompt_rows = 0;
    for _ in 0..1_000 {
        let status = engine.step().unwrap();
        engine.audit().unwrap();
        prompt_rows += engine
            .drain_events()
            .iter()
            .filter(|event| matches!(event.kind(), EngineEventKind::PromptLogprobs { .. }))
            .count();
        if status == StepStatus::Idle {
            break;
        }
    }
    let admissions = &engine.executor().admissions;
    assert_eq!(admissions[&RequestId::new(1)], 1);
    assert!(admissions[&RequestId::new(2)] > 1);
    assert_eq!(prompt_rows, 1);
}

#[test]
fn cancelling_a_preempted_or_recomputing_request_frees_everything() {
    // Two long requests outgrow 16 pages; the later one is preempted and
    // waits until the earlier one finishes, then recomputes.
    for recomputing in [false, true] {
        let mut engine = engine(Setup {
            pages: 16,
            admission: optimistic(3),
            prefix: false,
            speculation: false,
            sampled: false,
        });
        for id in [1, 2] {
            let request = EngineRequest::new(
                RequestId::new(id),
                (0..17).collect(),
                40,
                SamplingParameters::new(0.0).unwrap(),
                id,
                Vec::new(),
                vocabulary(),
            )
            .unwrap();
            engine.submit(request).unwrap();
        }
        let id = RequestId::new(2);
        let target = if recomputing {
            RequestState::Prefilling
        } else {
            RequestState::Queued
        };
        let mut emitted = 0;
        let mut reached = false;
        for _ in 0..200 {
            engine.step().unwrap();
            engine.audit().unwrap();
            emitted += engine
                .drain_events()
                .iter()
                .filter(|event| {
                    event.request_id() == id
                        && matches!(event.kind(), EngineEventKind::Token { .. })
                })
                .count();
            if emitted > 0 && engine.request_state(id).unwrap() == target {
                reached = true;
                break;
            }
        }
        assert!(reached, "request 2 was preempted and reached {target:?}");
        if !recomputing {
            let text = engine.metrics().prometheus_compatibility("fixture");
            let labels = r#"model_name="fixture",engine="0""#;
            assert!(text.contains(&format!("vllm:num_requests_waiting{{{labels}}} 1")));
            assert!(text.contains(&format!("vllm:num_requests_running{{{labels}}} 1")));
        }
        assert_eq!(
            engine.executor().admitted.contains(&id),
            recomputing,
            "a preempted request holds no model state until it is readmitted"
        );
        engine.cancel(id).unwrap();
        engine.audit().unwrap();
        let events = engine.drain_events();
        assert!(matches!(
            events.last().unwrap().kind(),
            EngineEventKind::Finished {
                reason: FinishReason::Cancelled
            }
        ));
        assert!(!engine.executor().admitted.contains(&id));
        assert!(!engine.executor().state.contains_key(&id));
        for _ in 0..200 {
            let status = engine.step().unwrap();
            engine.audit().unwrap();
            let _ = engine.drain_events();
            if status == StepStatus::Idle {
                break;
            }
        }
        assert!(engine.executor().admitted.is_empty());
        assert_eq!(engine.kv_stats().free_blocks(), 16);
    }
}

#[test]
fn preemption_counts_are_published_per_request() {
    let mut engine = engine(Setup {
        pages: 512,
        admission: forced(4, 2),
        prefix: false,
        speculation: false,
        sampled: false,
    });
    run(&mut engine, &requests()).unwrap();
    let text = engine.metrics().prometheus_compatibility("fixture");
    let total = preemptions(&engine);
    let sum = text
        .lines()
        .find(|line| line.starts_with("vllm:request_num_preemptions_sum{"))
        .unwrap()
        .rsplit(' ')
        .next()
        .unwrap()
        .parse::<f64>()
        .unwrap();
    assert!(total > 0);
    assert_eq!(sum, total as f64);
}

#[test]
fn a_request_preempted_before_its_first_batch_is_timed_from_that_batch() {
    // One four-token chunk per batch and five pages: request 1 decodes while
    // requests 2 and 3 are admitted, 3 waits for its first chunk, and 1's
    // growth preempts it before it ever ran.
    let config = EngineConfig::new(
        SchedulerConfig::new(3, 64, 4, 4, 1).unwrap(),
        KvGeometry::new(BLOCK, 5).unwrap(),
        vocabulary(),
        20,
        64,
    )
    .unwrap()
    .with_admission(optimistic(3));
    let mut engine = Engine::new(config, Model::default()).unwrap();
    let request = |id: u64, prompt: u32| {
        EngineRequest::new(
            RequestId::new(id),
            (0..prompt).collect(),
            12,
            SamplingParameters::new(0.0).unwrap(),
            id,
            Vec::new(),
            vocabulary(),
        )
        .unwrap()
    };
    engine.submit(request(1, 3)).unwrap();
    engine.step().unwrap();
    engine.submit(request(2, 8)).unwrap();
    engine.submit(request(3, 8)).unwrap();
    for _ in 0..500 {
        let status = engine.step().unwrap();
        engine.audit().unwrap();
        let _ = engine.drain_events();
        if status == StepStatus::Idle {
            break;
        }
    }
    assert!(engine.executor().retired_unexecuted > 0);
    assert_eq!(
        engine.metrics().snapshot().queue_observations,
        3,
        "each request's queue time ends at its first batch"
    );
    let text = engine.metrics().prometheus_compatibility("fixture");
    let labels = r#"model_name="fixture",engine="0""#;
    assert!(text.contains(&format!("vllm:num_requests_running{{{labels}}} 0")));
    assert!(text.contains(&format!("vllm:num_requests_waiting{{{labels}}} 0")));
}

#[test]
fn capped_prompt_scores_match_uncapped_with_cache_preemption_and_speculation() {
    fn collect_scores(
        chunk: usize,
        cap: usize,
        prefix: bool,
        speculation: bool,
    ) -> Vec<(u64, String)> {
        let mut engine = engine_with_scoring(
            Setup {
                pages: 512,
                admission: forced(2, 5),
                prefix,
                speculation,
                sampled: false,
            },
            chunk,
            cap,
        );
        // Warm the same prefix before requesting every prompt score.
        let prompt = (0..67).map(|i| (i % 20) as u32).collect::<Vec<_>>();
        let request = EngineRequest::new(
            RequestId::new(100),
            prompt.clone(),
            1,
            SamplingParameters::new(0.0).unwrap(),
            42,
            vec![STOP],
            vocabulary(),
        )
        .unwrap();
        run(&mut engine, &[request]).unwrap();
        // The scored request decodes after its prompt, drafting from it
        // when speculation is on, as the other request does.
        let scored = EngineRequest::new(
            RequestId::new(101),
            prompt,
            8,
            SamplingParameters::new(0.0)
                .unwrap()
                .with_top_logprobs(3)
                .unwrap(),
            42,
            vec![STOP],
            vocabulary(),
        )
        .unwrap()
        .with_prompt_logprobs(3);
        engine.submit(scored).unwrap();
        engine.submit(requests()[1].clone()).unwrap();
        let mut result = Vec::new();
        let mut positions = 0;
        let mut events = 0;
        for _ in 0..1000 {
            let status = engine.step().unwrap();
            engine.audit().unwrap();
            for event in engine.drain_events() {
                if event.request_id() != RequestId::new(101) {
                    continue;
                }
                match event.kind() {
                    EngineEventKind::PromptCached { .. } => {
                        panic!("scored prompts cannot skip rows")
                    }
                    EngineEventKind::PromptLogprobs { positions: rows } => {
                        // One event per scored chunk, never above the cap.
                        assert!(!rows.is_empty() && rows.len() <= chunk.min(cap));
                        events += 1;
                        for row in rows {
                            positions += 1;
                            assert_eq!(row.position, positions);
                            result.push((
                                row.position as u64,
                                format!(
                                    "{}:{}:{:?}",
                                    row.token_id,
                                    row.logprob.to_bits(),
                                    row.top_logprobs
                                        .iter()
                                        .map(|v| (v.token_id(), v.logprob().to_bits()))
                                        .collect::<Vec<_>>()
                                ),
                            ));
                        }
                    }
                    other => result.push((0, format!("{other:?}"))),
                }
            }
            if status == StepStatus::Idle {
                break;
            }
        }
        assert_eq!(positions, 66);
        // One event per chunk of at most `rows` tokens (captures may end a
        // chunk early): each row scores the next position, and the last
        // prompt row samples instead.
        assert!(events >= 66_usize.div_ceil(chunk.min(cap)));
        assert_eq!(
            engine.metrics().snapshot().spec_draft_tokens > 0,
            speculation
        );
        assert_eq!(engine.executor().admissions[&RequestId::new(101)], 1);
        result
    }
    let expected = collect_scores(128, 8192, false, false);
    for speculation in [false, true] {
        for prefix in [false, true] {
            for chunk in [3, 16, 128] {
                for cap in [1, 7, 512] {
                    assert_eq!(
                        collect_scores(chunk, cap, prefix, speculation),
                        expected,
                        "chunk={chunk} cap={cap} prefix={prefix} speculation={speculation}"
                    );
                }
            }
        }
    }
}

#[test]
fn exact_restore_keeps_decode_rows_instead_of_sampling_a_prefill_row() {
    for (speculation, source) in [
        (false, DraftSource::Host),
        (true, DraftSource::Host),
        (true, DraftSource::Bundle),
    ] {
        for sampled in [false, true] {
            let setup = Setup {
                pages: 512,
                admission: AdmissionPolicy::Reserved,
                prefix: true,
                speculation,
                sampled,
            };
            let mut reference = engine_with_source(setup, source);
            reference.executor_mut().phase_sensitive = true;
            let expected = run(&mut reference, &requests()).unwrap();
            let mut resumed = engine_with_source(
                Setup {
                    admission: forced(4, 2),
                    ..setup
                },
                source,
            );
            resumed.executor_mut().phase_sensitive = true;
            let actual = run(&mut resumed, &requests()).unwrap();
            assert!(preemptions(&resumed) > 0);
            assert_eq!(
                resumed.executor().restores.len(),
                preemptions(&resumed) as usize
            );
            assert_eq!(
                actual, expected,
                "speculation={speculation}, sampled={sampled}"
            );
            assert_eq!(resumed.executor().resumed_drafts > 0, speculation);
            let usage = |engine: &Engine<Model>| {
                engine
                    .metrics()
                    .prometheus_compatibility("fixture")
                    .lines()
                    .filter(|line| {
                        line.starts_with("vllm:request_prompt_tokens_sum{")
                            || line.starts_with("vllm:request_generation_tokens_sum{")
                            || line.starts_with("vllm:prompt_tokens_cached_total{")
                    })
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            };
            assert_eq!(usage(&resumed).len(), 3);
            assert_eq!(usage(&resumed), usage(&reference));
        }
    }
}

#[test]
fn prompt_logprobs_remain_exact_while_neighbors_resume() {
    for prefix in [false, true] {
        let setup = Setup {
            pages: 512,
            admission: AdmissionPolicy::Reserved,
            prefix,
            speculation: true,
            sampled: false,
        };
        let requests = [
            requests()[0].clone().with_prompt_logprobs(2),
            requests()[1].clone(),
        ];
        let expected = run(&mut engine(setup), &requests).unwrap();
        let mut resumed = engine(Setup {
            admission: forced(2, 3),
            ..setup
        });
        assert_eq!(run(&mut resumed, &requests).unwrap(), expected);
        assert_eq!(resumed.executor().admissions[&RequestId::new(1)], 1);
        assert!(resumed.executor().admissions[&RequestId::new(2)] > 1);
        assert_eq!(expected[&RequestId::new(1)].prompt.len(), 1);
    }
}

fn swap_info(capacity: usize) -> runtime_engine::KvOffloadInfo {
    runtime_engine::KvOffloadInfo {
        host_page_capacity: capacity,
        minimum_tokens: 1,
        snapshot_pages: 1,
    }
}

#[test]
fn swapped_streams_equal_uninterrupted_streams_with_both_tiers_audited() {
    for sampled in [false, true] {
        for prefix in [false, true] {
            for source in [
                DraftPolicy::from(DraftSource::Host),
                DraftPolicy::from(DraftSource::Bundle),
                DraftPolicy(
                    DraftSource::Bundle,
                    runtime_engine::DraftCostModel::PerBlock,
                ),
            ] {
                for speculation in [false, true] {
                    let base = Setup {
                        pages: 512,
                        admission: AdmissionPolicy::Reserved,
                        prefix,
                        speculation,
                        sampled,
                    };
                    let mut reference =
                        engine_with_offload(base, 8, 256, source, None, false, None);
                    let expected = run(&mut reference, &requests()).unwrap();
                    for (pages, admission, force) in
                        [(512, forced(5, 2), true), (24, optimistic(3), false)]
                    {
                        let mut engine = engine_with_offload(
                            Setup {
                                pages,
                                admission,
                                ..base
                            },
                            8,
                            256,
                            source,
                            Some(swap_info(512)),
                            force,
                            None,
                        );
                        let output = run(&mut engine, &requests()).unwrap_or_else(|e| panic!("{e:?} prefix={prefix} speculation={speculation} source={source:?} force={force} metrics={:?} ", engine.metrics().snapshot().offload));
                        assert_eq!(
                            output, expected,
                            "prefix={prefix} speculation={speculation} source={source:?} force={force}"
                        );
                        let stats = engine.metrics().snapshot().offload;
                        assert!(stats.copies[0] > 0);
                        assert!(stats.copies[1] > 0);
                        assert_eq!(stats.used_pages, 0);
                        assert_eq!(stats.swapped_requests, 0);
                    }
                }
            }
        }
    }
}

#[test]
fn confirmed_swap_failures_recompute_and_unconfirmed_work_stops() {
    for prefix in [false, true] {
        let base = Setup {
            pages: 512,
            admission: forced(5, 2),
            prefix,
            speculation: false,
            sampled: false,
        };
        let expected = baseline(base);
        for failure in [
            "tier_full",
            "capture",
            "out_refused",
            "in_refused",
            "out_failed",
            "in_failed",
            "restore",
        ] {
            let mut engine = engine_with_offload(
                base,
                8,
                256,
                DraftSource::Host,
                Some(swap_info(if failure == "tier_full" { 1 } else { 512 })),
                true,
                Some(failure),
            );
            assert_eq!(
                run(&mut engine, &requests()).unwrap(),
                expected,
                "{failure}"
            );
            assert_eq!(engine.metrics().snapshot().offload.used_pages, 0);
            if matches!(failure, "in_refused" | "in_failed") {
                assert_eq!(
                    engine.kv_metrics().snapshot().prefix_cache_restore_failures,
                    0
                );
            }
            assert!(
                engine
                    .metrics()
                    .snapshot()
                    .offload
                    .fallbacks
                    .iter()
                    .sum::<u64>()
                    > 0
            );
        }
        for failure in ["unconfirmed", "in_unconfirmed", "restore_fatal", "poll"] {
            let mut engine = engine_with_offload(
                base,
                8,
                256,
                DraftSource::Host,
                Some(swap_info(512)),
                true,
                Some(failure),
            );
            assert!(run(&mut engine, &requests()).is_err());
            assert!(matches!(
                engine.step(),
                Err(EngineError::PrefixStateUnconfirmed)
            ));
            engine.audit().unwrap();
            assert!(engine.metrics().snapshot().offload.used_pages > 0);
        }
    }
}

#[test]
fn cancellation_at_each_copy_phase_and_shutdown_drain_release_both_tiers() {
    for prefix in [false, true] {
        let setup = Setup {
            pages: 512,
            admission: forced(5, 2),
            prefix,
            speculation: true,
            sampled: false,
        };
        let mut observed_out = false;
        let mut observed_in = false;
        let mut observed_restored = false;
        for steps in 1..70 {
            let mut engine = engine_with_offload(
                setup,
                8,
                256,
                DraftSource::Host,
                Some(swap_info(512)),
                true,
                None,
            );
            for request in requests() {
                engine.submit(request).unwrap();
            }
            for _ in 0..steps {
                engine.step().unwrap();
                engine.audit().unwrap();
            }
            for (transfer, _) in engine
                .executor()
                .transfers
                .values()
                .filter(|(t, _)| !t.accepted())
            {
                observed_out |= transfer.direction() == runtime_kv::KvTransferDirection::Out;
                observed_in |= transfer.direction() == runtime_kv::KvTransferDirection::In;
            }
            observed_restored |= engine.metrics().snapshot().offload.copies[1] > 0;
            for id in 1..=10 {
                let id = RequestId::new(id);
                if engine
                    .request_state(id)
                    .is_ok_and(|state| !state.is_terminal())
                {
                    engine.cancel(id).unwrap();
                    engine.audit().unwrap();
                }
            }
            engine.drain_kv_transfers().unwrap();
            engine.run_until_idle(100).unwrap();
            engine.audit().unwrap();
            let stats = engine.metrics().snapshot().offload;
            assert_eq!((stats.used_pages, stats.swapped_requests), (0, 0));
            assert_eq!(engine.kv_stats().used_blocks(), 0);
            assert!(
                engine
                    .executor()
                    .transfers
                    .values()
                    .all(|(t, _)| t.accepted())
            );
        }
        assert!(observed_out && observed_in && observed_restored);
    }
}

#[test]
fn a_reload_that_cannot_fit_its_staging_recomputes_without_blocking_fifo() {
    let setup = Setup {
        pages: 16,
        admission: forced(5, 1),
        prefix: false,
        speculation: false,
        sampled: false,
    };
    let workload = vec![
        EngineRequest::new(
            RequestId::new(1),
            vec![1, 2, 3, 4],
            60,
            SamplingParameters::new(0.0).unwrap(),
            0,
            vec![],
            vocabulary(),
        )
        .unwrap(),
    ];
    let mut reference = engine(Setup {
        pages: 512,
        admission: AdmissionPolicy::Reserved,
        ..setup
    });
    let expected = run(&mut reference, &workload).unwrap();
    let mut swapped = engine_with_offload(
        setup,
        8,
        256,
        DraftSource::Host,
        Some(swap_info(512)),
        true,
        None,
    );
    assert_eq!(run(&mut swapped, &workload).unwrap(), expected);
    assert!(swapped.metrics().snapshot().offload.fallbacks[9] > 0);
    assert_eq!(swapped.metrics().snapshot().offload.used_pages, 0);
}

#[test]
fn disabled_unsupported_and_below_threshold_swaps_keep_the_existing_stream() {
    let setup = Setup {
        pages: 512,
        admission: forced(5, 2),
        prefix: false,
        speculation: true,
        sampled: true,
    };
    let mut reference = engine(setup);
    let expected = run(&mut reference, &requests()).unwrap();
    for info in [
        None,
        Some(runtime_engine::KvOffloadInfo {
            minimum_tokens: usize::MAX,
            ..swap_info(512)
        }),
    ] {
        let mut test = engine_with_offload(setup, 8, 256, DraftSource::Host, info, true, None);
        assert_eq!(run(&mut test, &requests()).unwrap(), expected);
        assert_eq!(test.metrics().snapshot().offload.copies, [0, 0]);
        assert_eq!(test.metrics().snapshot().offload.used_pages, 0);
        assert!(test.executor().transfers.is_empty());
    }
}

#[test]
fn cancelled_copy_completion_cannot_touch_a_reused_request_id() {
    for prefix in [false, true] {
        let setup = Setup {
            pages: 512,
            admission: forced(5, 2),
            prefix,
            speculation: false,
            sampled: false,
        };
        let mut reference = engine(Setup {
            admission: AdmissionPolicy::Reserved,
            ..setup
        });
        reference.executor_mut().phase_sensitive = true;
        let expected = run(&mut reference, &requests()).unwrap();
        let mut test = engine_with_offload(
            setup,
            8,
            256,
            DraftSource::Host,
            Some(swap_info(512)),
            true,
            None,
        );
        test.executor_mut().phase_sensitive = true;
        for request in requests() {
            test.submit(request).unwrap();
        }
        let mut pending_reload = false;
        for _ in 0..100 {
            test.step().unwrap();
            test.audit().unwrap();
            if test
                .executor()
                .transfers
                .values()
                .any(|(t, _)| !t.accepted() && t.direction() == runtime_kv::KvTransferDirection::In)
            {
                pending_reload = true;
                break;
            }
        }
        assert!(pending_reload);
        for id in 1..=10 {
            let id = RequestId::new(id);
            if test
                .request_state(id)
                .is_ok_and(|state| !state.is_terminal())
            {
                test.cancel(id).unwrap();
            }
        }
        let _ = test.drain_events();
        for (transfer, delay) in test.executor_mut().transfers.values_mut() {
            if !transfer.accepted() {
                *delay = 100_000;
            }
        }
        let old: BTreeSet<_> = test
            .executor()
            .transfers
            .iter()
            .filter_map(|(&id, (t, _))| (!t.accepted()).then_some(id))
            .collect();
        test.executor_mut().held = old.clone();
        test.executor_mut().delay = 2;
        let mut released = false;
        // Resubmit under another cache salt: the cancelled prompts' anchors
        // would otherwise serve these repeats, whose prefill after a restore
        // the phase-sensitive model reserves for own-boundary resumes.
        let again = requests()
            .into_iter()
            .map(|request| request.with_cache_salt(b"again".to_vec()))
            .collect::<Vec<_>>();
        let output = run_with_tick(&mut test, &again, |engine| {
            if !released
                && engine.executor().transfers.iter().any(|(id, (t, _))| {
                    !old.contains(id)
                        && !t.accepted()
                        && t.direction() == runtime_kv::KvTransferDirection::Out
                })
            {
                released = true;
                engine.executor_mut().held.clear();
                for id in &old {
                    engine.executor_mut().transfers.get_mut(id).unwrap().1 = 0;
                }
            }
        })
        .unwrap();
        assert!(released);
        assert_eq!(output, expected);
        assert_eq!(test.metrics().snapshot().offload.used_pages, 0);
    }
}

#[test]
fn cancelling_a_ready_queued_host_owner_releases_the_anchor() {
    let setup = Setup {
        pages: 24,
        admission: optimistic(3),
        prefix: false,
        speculation: false,
        sampled: false,
    };
    let mut test = engine_with_offload(
        setup,
        8,
        256,
        DraftSource::Host,
        Some(swap_info(512)),
        false,
        None,
    );
    for request in requests() {
        test.submit(request).unwrap();
    }
    let mut reached_ready_owner = false;
    for _ in 0..1000 {
        test.step().unwrap();
        test.audit().unwrap();
        if test.metrics().snapshot().offload.used_pages > 0
            && test
                .executor()
                .transfers
                .values()
                .all(|(transfer, _)| transfer.accepted())
        {
            reached_ready_owner = true;
            assert!((1..=10).any(|id| {
                test.request_state(RequestId::new(id))
                    .is_ok_and(|state| state == RequestState::Queued)
            }));
            for id in 1..=10 {
                let id = RequestId::new(id);
                if test
                    .request_state(id)
                    .is_ok_and(|state| !state.is_terminal())
                {
                    test.cancel(id).unwrap();
                    test.audit().unwrap();
                }
            }
            test.drain_kv_transfers().unwrap();
            test.run_until_idle(100).unwrap();
            assert_eq!(test.metrics().snapshot().offload.used_pages, 0);
            assert_eq!(test.kv_stats().free_blocks(), 24);
            break;
        }
    }
    assert!(reached_ready_owner);
}

#[test]
fn unavailable_snapshot_staging_falls_back_without_allocating_host_pages() {
    let setup = Setup {
        pages: 512,
        admission: forced(5, 2),
        prefix: false,
        speculation: false,
        sampled: false,
    };
    let expected = baseline(setup);
    let mut test = engine_with_offload(
        setup,
        8,
        256,
        DraftSource::Host,
        Some(runtime_engine::KvOffloadInfo {
            snapshot_pages: 512,
            ..swap_info(512)
        }),
        true,
        None,
    );
    assert_eq!(run(&mut test, &requests()).unwrap(), expected);
    assert!(test.metrics().snapshot().offload.fallbacks[1] > 0);
    assert_eq!(test.metrics().snapshot().offload.used_pages, 0);
    assert!(test.executor().transfers.is_empty());
}

#[test]
fn existing_public_anchors_are_copied_without_capture_fallback() {
    let workload: Vec<_> = (1..=4)
        .map(|id| {
            EngineRequest::new(
                RequestId::new(id),
                vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
                30,
                SamplingParameters::new(0.0).unwrap(),
                0,
                vec![],
                vocabulary(),
            )
            .unwrap()
        })
        .collect();
    for every in [1, 5] {
        for prefix in [false, true] {
            let setup = Setup {
                pages: 512,
                admission: forced(every, 2),
                prefix,
                speculation: false,
                sampled: false,
            };
            let mut engine = engine_with_offload(
                setup,
                8,
                256,
                DraftSource::Host,
                Some(swap_info(512)),
                false,
                None,
            );
            run(&mut engine, &workload).unwrap();
            let stats = engine.metrics().snapshot().offload;
            assert_eq!(stats.fallbacks[1], 0, "existing anchors must be offloaded");
            assert_eq!(stats.copies[0], preemptions(&engine) as u64);
            assert_eq!(
                stats.copies[1], 0,
                "a fitting own GPU anchor wins over host reload"
            );
            let kv = engine.kv_metrics().snapshot();
            assert_eq!(
                kv.prefix_restore_resumed_decoding,
                preemptions(&engine) as u64
            );
            if !prefix {
                assert_eq!(kv.prefix_cache_hits, 0);
                assert_eq!(kv.prefix_cache_queries, 0);
            }
        }
    }
}

#[test]
fn mixed_transfer_failures_and_cancellations_preserve_every_surviving_stream() {
    let mut successes = 0;
    let mut failures = 0;
    for seed in 0..24_u64 {
        let base = Setup {
            pages: 512,
            admission: AdmissionPolicy::Reserved,
            prefix: seed % 2 == 0,
            speculation: true,
            sampled: seed % 3 == 0,
        };
        let expected = baseline(base);
        let mut e = engine_with_offload(
            Setup {
                pages: 24 + seed as usize % 16,
                admission: forced(5, 3),
                ..base
            },
            8,
            256,
            DraftSource::Host,
            Some(swap_info(16 + seed as usize % 24)),
            true,
            None,
        );
        let mut tick = 0;
        let mut cancelled = BTreeSet::new();
        let output = run_with_tick(&mut e, &requests(), |e| {
            tick += 1;
            let random = mix(seed * 1009 + tick);
            e.executor_mut().delay = (random % 5) as usize;
            e.executor_mut().failure = match random % 11 {
                0 => Some("out_failed"),
                1 => Some("in_failed"),
                2 => Some("out_refused"),
                3 => Some("in_refused"),
                _ => None,
            };
            if random % 9 == 0 {
                let id = RequestId::new(random % 10 + 1);
                if e.request_state(id).is_ok_and(|state| !state.is_terminal()) {
                    e.cancel(id).unwrap();
                    cancelled.insert(id);
                    e.audit().unwrap();
                }
            }
        })
        .unwrap();
        for (id, result) in output {
            if cancelled.contains(&id) {
                assert!(expected[&id].tokens.starts_with(&result.tokens));
            } else {
                assert_eq!(result, expected[&id]);
            }
        }
        e.drain_kv_transfers().unwrap();
        e.audit().unwrap();
        successes += e.metrics().snapshot().offload.copies.iter().sum::<u64>();
        failures += e.metrics().snapshot().offload.fallbacks.iter().sum::<u64>();
        assert_eq!(e.metrics().snapshot().offload.used_pages, 0);
        assert_eq!(e.kv_stats().used_blocks(), 0);
    }
    assert!(successes > 0 && failures > 0);
}

fn engine_with_pending_copy() -> Engine<Model> {
    let setup = Setup {
        pages: 512,
        admission: forced(5, 3),
        prefix: false,
        speculation: false,
        sampled: false,
    };
    let mut e = engine_with_offload(
        setup,
        8,
        256,
        DraftSource::Host,
        Some(swap_info(512)),
        true,
        None,
    );
    e.executor_mut().delay = 100;
    for r in requests() {
        e.submit(r).unwrap();
    }
    for _ in 0..100 {
        e.step().unwrap();
        if e.executor().transfers.values().any(|(t, _)| !t.accepted()) {
            return e;
        }
    }
    panic!("no outgoing transfer");
}

#[test]
fn engine_drop_is_bounded_and_does_not_poll_while_unwinding() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    for panic_poll in [false, true] {
        let mut e = engine_with_pending_copy();
        let ids = e.executor().transfers.keys().copied().collect();
        e.executor_mut().held = ids;
        e.executor_mut().panic_poll = panic_poll;
        let calls = Arc::new(AtomicUsize::new(0));
        e.executor_mut().polls = Some(calls.clone());
        let start = std::time::Instant::now();
        drop(e);
        assert!(start.elapsed() < std::time::Duration::from_secs(1));
        assert!(calls.load(Ordering::Relaxed) > 0);
    }
    let mut e = engine_with_pending_copy();
    let calls = Arc::new(AtomicUsize::new(0));
    e.executor_mut().polls = Some(calls.clone());
    e.executor_mut().panic_poll = true;
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _e = e;
        panic!("worker panic");
    }));
    assert!(caught.is_err());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn fatal_submit_refusal_still_drains_confirmed_older_copies() {
    let mut e = engine_with_pending_copy();
    e.executor_mut().failure = Some("out_fatal");
    let mut failed = false;
    for _ in 0..100 {
        if e.step().is_err() {
            failed = true;
            break;
        }
    }
    assert!(failed);
    assert!(matches!(e.step(), Err(EngineError::ModelStateFailed)));
    e.audit().unwrap();
    assert!(
        e.metrics()
            .prometheus_compatibility("test")
            .contains("chop_instance_poisoned{model_name=\"test\",engine=\"0\"} 1")
    );
    e.executor_mut().failure = None;
    e.drain_kv_transfers().unwrap();
    assert!(e.executor().transfers.values().all(|(t, _)| t.accepted()));
}

#[test]
fn a_refused_completion_is_repolled_without_losing_either_transfer() {
    let mut e = engine_with_pending_copy();
    for _ in 0..100 {
        if e.executor().transfers.len() >= 2 {
            break;
        }
        e.step().unwrap();
    }
    assert!(e.executor().transfers.len() >= 2);
    e.executor_mut().mismatched_completion = true;
    assert!(matches!(e.step(), Err(EngineError::InvariantViolation)));
    e.audit().unwrap();
    e.drain_kv_transfers().unwrap();
    e.audit().unwrap();
    assert!(e.executor().transfers.values().all(|(t, _)| t.accepted()));
}

#[test]
fn cancelled_incoming_failed_idle_releases_both_tiers() {
    for prefix in [false, true] {
        let setup = Setup {
            pages: 512,
            admission: forced(5, 3),
            prefix,
            speculation: false,
            sampled: false,
        };
        let mut e = engine_with_offload(
            setup,
            8,
            256,
            DraftSource::Host,
            Some(swap_info(512)),
            true,
            None,
        );
        for r in requests() {
            e.submit(r).unwrap();
        }
        let mut incoming = false;
        for _ in 0..200 {
            e.step().unwrap();
            e.audit().unwrap();
            if e.executor()
                .transfers
                .values()
                .any(|(t, _)| !t.accepted() && t.direction() == runtime_kv::KvTransferDirection::In)
            {
                incoming = true;
                break;
            }
        }
        assert!(incoming);
        for id in 1..=10 {
            let id = RequestId::new(id);
            if e.request_state(id).is_ok_and(|s| !s.is_terminal()) {
                e.cancel(id).unwrap();
                e.audit().unwrap();
            }
        }
        e.executor_mut().failure = Some("in_failed");
        e.drain_kv_transfers().unwrap();
        e.audit().unwrap();
        assert!(e.metrics().snapshot().offload.fallbacks[5] > 0);
        assert_eq!(e.metrics().snapshot().offload.used_pages, 0);
        assert_eq!(e.kv_stats().used_blocks(), 0);
    }
}

#[test]
fn a_drafter_error_with_pending_copies_is_never_dropped() {
    let setup = Setup {
        pages: 28,
        admission: forced(5, 3),
        prefix: false,
        speculation: true,
        sampled: false,
    };
    let mut e = engine_with_offload(
        setup,
        8,
        256,
        DraftSource::Bundle,
        Some(swap_info(64)),
        true,
        None,
    );
    e.executor_mut().delay = 20;
    for r in requests() {
        e.submit(r).unwrap();
    }
    let mut injected = false;
    let mut failed = false;
    for _ in 0..500 {
        if e.executor().transfers.values().any(|(t, _)| !t.accepted()) {
            e.executor_mut().draft_failure = true;
            injected = true;
        }
        match e.step() {
            Err(EngineError::Executor(_)) => {
                failed = true;
                break;
            }
            Ok(_) => (),
            Err(other) => panic!("wrong failure: {other:?}"),
        }
        e.audit().unwrap();
    }
    assert!(injected && failed);
}

#[test]
fn a_confirmed_failed_outgoing_copy_keeps_the_own_gpu_resume_available() {
    for prefix in [false, true] {
        let setup = Setup {
            pages: 512,
            admission: forced(5, 2),
            prefix,
            speculation: false,
            sampled: false,
        };
        let mut e = engine_with_offload(
            setup,
            8,
            256,
            DraftSource::Host,
            Some(swap_info(512)),
            false,
            Some("out_failed"),
        );
        assert_eq!(run(&mut e, &requests()).unwrap(), baseline(setup));
        let stats = e.metrics().snapshot().offload;
        assert!(stats.fallbacks[4] > 0);
        assert_eq!(stats.copies[1], 0);
        assert!(e.kv_metrics().snapshot().prefix_restore_resumed_decoding > 0);
        assert_eq!(stats.used_pages, 0);
    }
}

#[test]
fn drain_timeout_retains_leases_without_poisoning_healthy_copies() {
    let mut e = engine_with_pending_copy();
    e.executor_mut().held = e.executor().transfers.keys().copied().collect();
    let started = std::time::Instant::now();
    assert!(matches!(
        e.drain_kv_transfers(),
        Err(EngineError::KvTransferDrainTimeout)
    ));
    assert!(started.elapsed() < std::time::Duration::from_secs(10));
    assert!(e.executor().transfers.values().any(|(t, _)| !t.accepted()));
    assert!(e.metrics().snapshot().offload.used_pages > 0);
    e.audit().unwrap();
    e.executor_mut().held.clear();
    for (_, delay) in e.executor_mut().transfers.values_mut() {
        *delay = 0;
    }
    assert!(
        e.drain_kv_transfers().is_ok(),
        "elapsed time is not unconfirmed DMA"
    );
    e.step().unwrap();
    e.audit().unwrap();
}

#[test]
fn warm_prefix_cache_captures_every_swap_under_copy_leases() {
    let second: Vec<EngineRequest> = (0..10_u64)
        .map(|i| {
            let prompt: Vec<u32> = (0..9 + i % 7)
                .map(|p| (mix(i * 77 + p + 1000) % 20) as u32)
                .collect();
            EngineRequest::new(
                RequestId::new(100 + i),
                prompt,
                40,
                SamplingParameters::new(0.0).unwrap(),
                0,
                vec![STOP],
                vocabulary(),
            )
            .unwrap()
        })
        .collect();
    for pages in [64usize, 96] {
        for delay in [0usize, 4, 20] {
            let setup = Setup {
                pages,
                admission: forced(4, 3),
                prefix: true,
                speculation: false,
                sampled: false,
            };
            let mut e = engine_with_offload(
                setup,
                8,
                256,
                DraftSource::Host,
                Some(swap_info(512)),
                false,
                None,
            );
            e.executor_mut().delay = delay;
            run(&mut e, &requests()).unwrap();
            let before = e.metrics().snapshot().offload;
            let kv0 = e.kv_metrics().snapshot();
            run(&mut e, &second).unwrap();
            let after = e.metrics().snapshot().offload;
            let kv = e.kv_metrics().snapshot();
            assert_eq!(after.fallbacks[1] - before.fallbacks[1], 0);
            assert_eq!(after.copies[0] - before.copies[0], 30);
            // Each prompt's end and final chunk start are public captures;
            // the 30 preemption swaps capture under private keys, which are
            // not cache entries.
            assert_eq!(kv.prefix_cache_captures - kv0.prefix_cache_captures, 20);
        }
    }
}

#[test]
fn admission_failure_emits_the_executor_error() {
    let mut e = engine(Setup {
        // Both worst-case reservations must fit in the same admission pass.
        pages: 64,
        admission: AdmissionPolicy::Reserved,
        prefix: false,
        speculation: false,
        sampled: false,
    });
    e.executor_mut().failure = Some("admit");
    for request in requests().into_iter().take(2) {
        e.submit(request).unwrap();
    }
    assert!(matches!(e.step(), Err(EngineError::Executor(_))));
    assert_eq!(e.metrics().snapshot().errors, 2);
    assert_eq!(e.metrics().snapshot().cancellations, 0);
    let events = e.drain_events();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event.kind(),
        EngineEventKind::Error { message } if message == "admission failed"))
            .count(),
        2
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event.kind(), EngineEventKind::Finished { .. }))
    );
    e.audit().unwrap();
}

#[test]
fn randomized_engine_interleavings_preserve_streams() {
    let n: u64 = std::env::var("CHOPINFER_ENGINE_FUZZ_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let start: u64 = std::env::var("CHOPINFER_ENGINE_FUZZ_START")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    assert!(n > 0, "the fuzz harness must execute at least one case");
    let mut exercised = [0_u64; 4];
    for case in start..start + n {
        let mut r = mix(case ^ 0xabcdef) | 1;
        let mut next = move || {
            r = mix(r);
            r
        };
        let prefix = next() % 2 == 0;
        let speculation = next() % 2 == 0;
        let sampled = next() % 2 == 0;
        let pages = 20 + (next() % 30) as usize;
        let admission = match next() % 3 {
            0 => optimistic((next() % 3) as usize),
            1 => forced(1 + (next() % 6) as usize, (next() % 3) as usize),
            _ => AdmissionPolicy::Reserved,
        };
        let mut setup = Setup {
            pages,
            admission,
            prefix,
            speculation,
            sampled,
        };
        let base = baseline(setup);
        let source = if next() % 2 == 0 {
            DraftSource::Host
        } else {
            DraftSource::Bundle
        };
        let fault =
            ["none", "out_failed", "out_refused", "delayed", "discard"][(next() % 5) as usize];
        let mut offload = (next() % 3 == 0 || fault != "none").then(|| swap_info(256));
        let force = offload.is_some() && next() % 2 == 0;
        if fault == "discard" {
            // Capture an early boundary, then pin its worst-case reservation.
            // The host reload's staging cannot fit, so the engine discards the
            // host plan while the outgoing copy can still be pending.
            offload.as_mut().unwrap().snapshot_pages = pages - 6;
            setup.admission = forced(1, 1);
        }
        let failure = match fault {
            "out_failed" | "out_refused" => Some(fault),
            _ => None,
        };
        let chunk = [4usize, 8][(next() % 2) as usize];
        let mut e = engine_with_offload(setup, chunk, 512, source, offload, force, failure);
        if fault == "delayed" || fault == "discard" {
            e.executor_mut().delay = 1 + (next() % 16) as usize;
        }
        let mut cancelled = BTreeSet::new();
        let mut ticks = 0u64;
        let seed = next();
        let result = run_with_tick(&mut e, &requests(), |engine| {
            ticks += 1;
            if fault == "delayed"
                && engine
                    .executor()
                    .transfers
                    .values()
                    .any(|(copy, _)| !copy.accepted())
            {
                exercised[2] += 1;
            }
            let roll = mix(seed ^ ticks);
            if roll % 9 == 0 {
                let id = RequestId::new(1 + (roll >> 8) % 10);
                if engine.request_state(id).is_ok_and(|s| !s.is_terminal()) {
                    engine
                        .cancel(id)
                        .unwrap_or_else(|err| panic!("case {case}: cancel {err:?}"));
                    cancelled.insert(id);
                }
            }
        });
        let outputs = result.unwrap_or_else(|err| {
            panic!("case {case} {setup:?} {source:?} off={offload:?} fault={fault}: {err:?}")
        });
        let stats = e.metrics().snapshot().offload;
        exercised[0] += stats.fallbacks[4];
        exercised[1] += stats.fallbacks[2];
        if fault == "discard" {
            exercised[3] += stats.fallbacks[9];
        }
        for (id, expected) in &base {
            let got = outputs.get(id);
            if cancelled.contains(id) {
                if let Some(got) = got {
                    assert!(
                        expected.tokens.starts_with(&got.tokens),
                        "case {case}: cancelled {id:?} streamed a non-prefix"
                    );
                }
            } else {
                assert_eq!(
                    got,
                    Some(expected),
                    "case {case} {setup:?} {source:?} off={offload:?} fault={fault} chunk={chunk}: {id:?} diverged"
                );
            }
        }
    }
    eprintln!("copy failures/refusals/pending ticks/discards: {exercised:?}");
    if n >= 200 {
        assert!(
            exercised.iter().all(|count| *count > 0),
            "missing copy fault coverage: {exercised:?}"
        );
    }
}
