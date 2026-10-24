//! Speculative decoding must not change what a request emits.
//!
//! The model here is a pure function from a token prefix to logits, so a
//! request's output without speculation is fully determined. Every test runs
//! the same requests with speculation and compares each request's events,
//! token IDs, logprobs, top logprobs and finish reasons, for exact equality.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use runtime_engine::{
    DraftRow, DraftSource, Engine, EngineConfig, EngineError, EngineEvent, EngineEventKind,
    EngineMetrics, EngineRequest, ExecutorError, KeptRows, LogitsRow, ModelBatchItem,
    ModelExecutor, ModelTransaction, ModelTransactionState, MonotonicClock, PromptLookupConfig,
    PromptLookupExecutor, SampledRow, SpeculationConfig, SpeculationMode, StepStatus,
};
use runtime_kv::{KvGeometry, PreparedAppendBatch};
use runtime_model_api::ModelVocabulary;
use runtime_sampling::{LogitBias, SamplingParameters, sample_for_vocabulary};
use runtime_scheduler::{BatchPhase, RequestId, SchedulerConfig};

const VOCABULARY: usize = 24;
const STOP: u32 = 5;
const KV_BLOCKS: usize = 256;

fn vocabulary() -> ModelVocabulary {
    ModelVocabulary::new(VOCABULARY).unwrap()
}

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// The model: logits for the token after `prefix`, with one favoured token
/// so that drafts are often right and sampling is not uniform.
fn logits_for(prefix: &[u32]) -> Vec<f32> {
    let hash = prefix
        .iter()
        .fold(0x1234_5678_u64, |hash, token| mix(hash ^ u64::from(*token)));
    let mut logits: Vec<f32> = (0..VOCABULARY as u64)
        .map(|token| (mix(hash ^ (token << 40)) >> 40) as f32 / (1u64 << 24) as f32 * 4.0 - 2.0)
        .collect();
    logits[(hash % VOCABULARY as u64) as usize] += 3.0;
    logits
}

/// A model of repetitive text: it strongly prefers the token `period`
/// positions back, so its output continues any cycle in its context.
fn copying_logits_for(prefix: &[u32], period: usize) -> Vec<f32> {
    let mut logits = logits_for(prefix);
    if let Some(previous) = prefix.len().checked_sub(period) {
        logits[prefix[previous] as usize] += 8.0;
    }
    logits
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (token, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = token;
        }
    }
    best as u32
}

#[derive(Clone, Copy, Debug)]
enum Drafter {
    /// Chains the model's own argmax, which greedy requests always accept.
    Greedy,
    /// Proposes a token next to the argmax, which greedy requests reject.
    Wrong,
    /// Hash noise, sometimes right.
    Noise,
    /// Proposes the stop token.
    Stop,
    /// The argmax chain with every third draft wrong.
    Mostly,
}

impl Drafter {
    fn propose(self, context: &[u32], count: usize) -> Vec<u32> {
        let mut context = context.to_vec();
        let mut drafts = Vec::new();
        for index in 0..count {
            let best = argmax(&logits_for(&context));
            let token = match self {
                Self::Greedy => best,
                Self::Wrong => (best + 1) % VOCABULARY as u32,
                Self::Noise => {
                    (mix(context.len() as u64 ^ u64::from(best)) % VOCABULARY as u64) as u32
                }
                Self::Stop => STOP,
                Self::Mostly if index % 3 == 2 => (best + 7) % VOCABULARY as u32,
                Self::Mostly => best,
            };
            drafts.push(token);
            context.push(token);
        }
        drafts
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Rows {
    /// Full logits rows; the engine samples on the host.
    Logits,
    /// Rows sampled by the executor, as a device sampler would.
    Sampled,
}

/// A model executor that keeps its own copy of each request's committed KV.
struct Model {
    drafter: Drafter,
    rows: Rows,
    kv: BTreeMap<RequestId, Vec<u32>>,
    pending: Vec<(RequestId, Vec<u32>, bool)>,
    corrupt_drafts: bool,
    /// When set, each execution advances this clock by `micros_per_row`
    /// per row.
    clock: Option<Arc<AtomicU64>>,
    micros_per_row: u64,
    /// Controlled whole-step timings for adaptive recovery tests.
    scripted_verify_costs: Option<VecDeque<u64>>,
    scripted_phases: Vec<bool>,
    /// When set, the model copies the token this many positions back.
    copy_period: Option<usize>,
    /// (rows, items) of every executed step without prefill that verifies.
    verify_steps: Vec<(usize, usize)>,
    /// A fake confidence result sink supplies a shorter proposal prefix.
    confidence_caps: Vec<usize>,
}

/// A clock only the model advances, so step costs are exact.
#[derive(Debug)]
struct ModelClock(Arc<AtomicU64>);

impl MonotonicClock for ModelClock {
    fn now_micros(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

impl Model {
    fn new(drafter: Drafter, rows: Rows) -> Self {
        Self {
            drafter,
            rows,
            kv: BTreeMap::new(),
            pending: Vec::new(),
            corrupt_drafts: false,
            clock: None,
            micros_per_row: 1000,
            scripted_verify_costs: None,
            scripted_phases: Vec::new(),
            copy_period: None,
            verify_steps: Vec::new(),
            confidence_caps: Vec::new(),
        }
    }
}

struct ModelState<'a> {
    model: &'a mut Model,
}

impl ModelTransactionState for ModelState<'_> {
    fn commit(&mut self) -> Result<(), ExecutorError> {
        for (id, inputs, verify) in self.model.pending.drain(..) {
            assert!(!verify, "a batch with verify items must commit kept rows");
            self.model.kv.entry(id).or_default().extend(inputs);
        }
        Ok(())
    }

    fn commit_kept(&mut self, kept: &[KeptRows]) -> Result<(), ExecutorError> {
        let mut kept = kept.iter();
        for (id, inputs, verify) in self.model.pending.drain(..) {
            let count = if verify {
                let record = kept.next().expect("one kept record per verify item");
                assert_eq!(record.request_id, id);
                assert!((1..=inputs.len()).contains(&record.kept_rows));
                record.kept_rows
            } else {
                inputs.len()
            };
            self.model
                .kv
                .entry(id)
                .or_default()
                .extend_from_slice(&inputs[..count]);
        }
        assert!(kept.next().is_none());
        Ok(())
    }

    fn rollback(&mut self) -> Result<(), ExecutorError> {
        self.model.pending.clear();
        Ok(())
    }
}

impl ModelExecutor for Model {
    type TransactionState<'a> = ModelState<'a>;

    fn retire_request_v1(&mut self, id: RequestId) -> Result<(), ExecutorError> {
        self.kv.remove(&id);
        Ok(())
    }

    fn draft_tokens_v1(
        &mut self,
        id: RequestId,
        history: &[u32],
        maximum: usize,
    ) -> Result<Vec<u32>, ExecutorError> {
        let maximum = self
            .confidence_caps
            .get(id.get() as usize - 1)
            .copied()
            .map_or(maximum, |cap| cap.min(maximum));
        Ok(self.drafter.propose(history, maximum))
    }

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        kv_plan: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        assert!(self.pending.is_empty());
        if let Some(costs) = self.scripted_verify_costs.as_mut() {
            let verifies = items
                .iter()
                .any(|item| matches!(item.phase(), BatchPhase::Verify { .. }));
            self.scripted_phases.push(verifies);
            let micros = if verifies {
                costs.pop_front().unwrap_or(64_000)
            } else {
                90_000
            };
            self.clock
                .as_ref()
                .expect("controlled clock")
                .fetch_add(micros, Ordering::Relaxed);
        }
        if items
            .iter()
            .any(|item| matches!(item.phase(), BatchPhase::Verify { .. }))
            && !items
                .iter()
                .any(|item| matches!(item.phase(), BatchPhase::Prefill { .. }))
        {
            let rows = items
                .iter()
                .map(|item| item.phase().appended_tokens())
                .sum();
            self.verify_steps.push((rows, items.len()));
        }
        let mut logits_rows = Vec::new();
        let mut sampled_rows = Vec::new();
        let mut draft_rows = Vec::new();
        for (item, append) in items.iter().zip(kv_plan.appends()) {
            let committed = self.kv.get(&item.request_id()).cloned().unwrap_or_default();
            assert_eq!(committed.len(), append.previous_tokens());
            let mut inputs = item.input_tokens().to_vec();
            let mut drafts = Vec::new();
            if let Some(draft) = item.draft() {
                assert!(
                    matches!(item.phase(), BatchPhase::Verify { draft_tokens } if draft_tokens == draft.tokens())
                );
                match draft.source() {
                    DraftSource::Host => drafts = inputs[1..].to_vec(),
                    DraftSource::Bundle => {
                        assert_eq!(inputs.len(), 1);
                        let mut context = committed.clone();
                        context.push(inputs[0]);
                        drafts = self.drafter.propose(&context, draft.tokens());
                        inputs.extend_from_slice(&drafts);
                        let mut reported = drafts.clone();
                        if self.corrupt_drafts {
                            reported.pop();
                        }
                        draft_rows.push(DraftRow::new(item.request_id(), reported));
                    }
                }
            }
            assert_eq!(inputs.len(), append.additional_tokens());
            if let Some(clock) = &self.clock
                && self.scripted_verify_costs.is_none()
            {
                clock.fetch_add(self.micros_per_row * inputs.len() as u64, Ordering::Relaxed);
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
                    let row = input - first_row;
                    let row_work = work.for_verify_row(&drafts, row);
                    // The row's sampling history is exactly the tokens the
                    // model conditions on.
                    assert_eq!(prefix, row_work.history.to_vec());
                    let logits = match self.copy_period {
                        Some(period) => copying_logits_for(&prefix, period),
                        None => logits_for(&prefix),
                    };
                    match self.rows {
                        Rows::Logits => {
                            logits_rows.push(LogitsRow::new(item.request_id(), logits).unwrap());
                        }
                        Rows::Sampled => {
                            let mut state = row_work.state;
                            let sample = sample_for_vocabulary(
                                &row_work.parameters,
                                &logits,
                                vocabulary(),
                                row_work
                                    .history
                                    .context(row_work.emitted_tokens, &row_work.stop_ids),
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
                }
            }
            self.pending
                .push((item.request_id(), inputs, item.draft().is_some()));
        }
        Ok(
            ModelTransaction::new(logits_rows, ModelState { model: self })
                .with_sampled(sampled_rows)
                .with_drafts(draft_rows),
        )
    }
}

fn parameters(kind: usize, maximum: usize) -> SamplingParameters {
    match kind % 5 {
        0 => SamplingParameters::new(0.0).unwrap(),
        1 => SamplingParameters::new(0.0)
            .unwrap()
            .with_top_logprobs(3)
            .unwrap(),
        2 => SamplingParameters::new(0.7)
            .unwrap()
            .with_top_k(6)
            .unwrap()
            .with_top_p(0.9)
            .unwrap()
            .with_top_logprobs(2)
            .unwrap(),
        3 => SamplingParameters::new(1.0)
            .unwrap()
            .with_min_p(0.05)
            .unwrap()
            .with_penalties(0.3, 0.2, 1.3)
            .unwrap()
            .with_logit_bias(vec![LogitBias::new(7, 1.5).unwrap()])
            .unwrap()
            .with_minimum_new_tokens(maximum.min(4))
            .unwrap(),
        _ => SamplingParameters::new(1.3)
            .unwrap()
            .with_penalties(-0.5, 0.0, 0.8)
            .unwrap(),
    }
}

/// Requests of mixed length, sampling and stop settings.
fn requests(count: u64) -> Vec<EngineRequest> {
    (0..count)
        .map(|index| {
            let prompt: Vec<u32> = (0..3 + index % 9)
                .map(|position| (mix(index * 31 + position) % VOCABULARY as u64) as u32)
                .collect();
            let maximum = [1, 2, 7, 25, 40][(index % 5) as usize];
            let stops = if index % 2 == 0 {
                vec![STOP]
            } else {
                Vec::new()
            };
            EngineRequest::new(
                RequestId::new(index + 1),
                prompt,
                maximum,
                parameters(index as usize / 2, maximum),
                mix(index) ^ 0xfeed,
                stops,
                vocabulary(),
            )
            .unwrap()
        })
        .collect()
}

fn config(speculation: Option<SpeculationConfig>) -> EngineConfig {
    let config = EngineConfig::new(
        SchedulerConfig::new(6, 64, 24, 4, 1).unwrap(),
        KvGeometry::new(4, KV_BLOCKS).unwrap(),
        vocabulary(),
        256,
        64,
    )
    .unwrap();
    match speculation {
        Some(speculation) => config.with_speculation(speculation),
        None => config,
    }
}

type Outputs = BTreeMap<RequestId, Vec<EngineEventKind>>;

/// Reaches the test model inside an executor.
trait HasModel {
    fn model(&self) -> &Model;
}

impl HasModel for Model {
    fn model(&self) -> &Model {
        self
    }
}

impl HasModel for PromptLookupExecutor<Model> {
    fn model(&self) -> &Model {
        self.inner()
    }
}

fn run<E: ModelExecutor + HasModel>(engine: &mut Engine<E>, requests: &[EngineRequest]) -> Outputs {
    let mut outputs = Outputs::new();
    let mut pending = requests.iter();
    // Staggered arrival mixes prefill chunks with decode and verify items.
    for step in 0..10_000 {
        if step % 3 == 0
            && let Some(request) = pending.next()
        {
            engine.submit(request.clone()).unwrap();
        }
        let status = engine.step().unwrap();
        engine.audit().unwrap();
        for event in engine.drain_events() {
            record(&mut outputs, event);
        }
        if status == StepStatus::Idle && pending.len() == 0 {
            assert_eq!(
                engine.kv_stats().free_blocks(),
                engine.kv_stats().block_capacity()
            );
            assert!(engine.executor().model().kv.is_empty());
            return outputs;
        }
    }
    panic!("the engine did not finish");
}

fn record(outputs: &mut Outputs, event: EngineEvent) {
    outputs
        .entry(event.request_id())
        .or_default()
        .push(event.kind().clone());
}

fn reference(requests: &[EngineRequest], rows: Rows) -> Outputs {
    let mut engine = Engine::new(config(None), Model::new(Drafter::Greedy, rows)).unwrap();
    run(&mut engine, requests)
}

fn speculative(
    requests: &[EngineRequest],
    drafter: Drafter,
    rows: Rows,
    speculation: SpeculationConfig,
) -> (Outputs, runtime_engine::EngineMetricsSnapshot) {
    let mut engine = Engine::new(config(Some(speculation)), Model::new(drafter, rows)).unwrap();
    let outputs = run(&mut engine, requests);
    (outputs, engine.metrics().snapshot())
}

fn speculation(
    drafts: usize,
    rows: usize,
    mode: SpeculationMode,
    source: DraftSource,
) -> SpeculationConfig {
    SpeculationConfig::new(drafts, rows, mode, source).unwrap()
}

#[test]
fn every_drafter_policy_and_row_path_emits_exactly_the_reference_output() {
    let requests = requests(20);
    for rows in [Rows::Logits, Rows::Sampled] {
        let expected = reference(&requests, rows);
        assert_eq!(expected.len(), requests.len());
        for drafter in [
            Drafter::Greedy,
            Drafter::Wrong,
            Drafter::Noise,
            Drafter::Stop,
            Drafter::Mostly,
        ] {
            for mode in [
                SpeculationMode::Fixed,
                SpeculationMode::Balanced(Default::default()),
                SpeculationMode::Adaptive,
            ] {
                for source in [DraftSource::Bundle, DraftSource::Host] {
                    for cost_model in [
                        runtime_engine::DraftCostModel::PerToken,
                        runtime_engine::DraftCostModel::PerBlock,
                    ] {
                        for (drafts, verify_rows) in [(1, 64), (4, 64), (7, 12), (16, 64)] {
                            let (actual, metrics) = speculative(
                                &requests,
                                drafter,
                                rows,
                                speculation(drafts, verify_rows, mode, source)
                                    .with_draft_cost_model(cost_model),
                            );
                            assert_eq!(
                                actual, expected,
                                "{drafter:?} {rows:?} {mode:?} {source:?} {cost_model:?} K={drafts} rows={verify_rows}"
                            );
                            assert_eq!(
                                metrics.spec_accepted_positions.iter().sum::<u64>(),
                                metrics.spec_accepted_tokens
                            );
                            assert!(metrics.spec_emitted_tokens <= metrics.spec_rows);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn balanced_sampled_batches_preserve_target_draws_and_logprob_bits() {
    for count in [8, 16] {
        let base = EngineConfig::new(
            SchedulerConfig::new(count, 64, 128, 4, 1).unwrap(),
            KvGeometry::new(4, KV_BLOCKS).unwrap(),
            vocabulary(),
            256,
            64,
        )
        .unwrap();
        for seed in [0, 1, 17, 142] {
            let requests = (0..count)
                .map(|index| {
                    EngineRequest::new(
                        RequestId::new(index as u64 + 1),
                        vec![1, 2, 3, index as u32],
                        24,
                        SamplingParameters::new(1.0)
                            .unwrap()
                            .with_top_p(1.0)
                            .unwrap()
                            .with_top_logprobs(5)
                            .unwrap(),
                        seed + index as u64,
                        Vec::new(),
                        vocabulary(),
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            for rows in [Rows::Logits, Rows::Sampled] {
                let mut reference = Engine::new(base, Model::new(Drafter::Greedy, rows)).unwrap();
                for request in &requests {
                    reference.submit(request.clone()).unwrap();
                }
                let expected = run(&mut reference, &[]);
                for drafter in [
                    Drafter::Greedy,
                    Drafter::Wrong,
                    Drafter::Mostly,
                    Drafter::Noise,
                ] {
                    for source in [DraftSource::Host, DraftSource::Bundle] {
                        let mut engine = Engine::new(
                            base.with_speculation(speculation(
                                5,
                                32,
                                SpeculationMode::Balanced(Default::default()),
                                source,
                            )),
                            Model::new(drafter, rows),
                        )
                        .unwrap();
                        for request in &requests {
                            engine.submit(request.clone()).unwrap();
                        }
                        let actual = run(&mut engine, &[]);
                        assert_eq!(
                            actual, expected,
                            "{count} {seed} {rows:?} {drafter:?} {source:?}"
                        );
                        assert_eq!(logprob_bits(&actual), logprob_bits(&expected));
                        assert!(engine.executor().verify_steps.contains(&(32, count)));
                        assert!(
                            engine
                                .executor()
                                .verify_steps
                                .iter()
                                .all(|&(rows, _)| rows <= 32)
                        );
                        assert!(engine.metrics().snapshot().spec_verify_items > 0);
                    }
                }
            }
        }
    }
}

#[test]
fn adaptive_rechecks_keep_sampled_tokens_and_logprob_bits_exact() {
    for count in [8, 16] {
        let base = EngineConfig::new(
            SchedulerConfig::new(count, 64, 128, 4, 1).unwrap(),
            KvGeometry::new(4, 2048).unwrap(),
            vocabulary(),
            256,
            64,
        )
        .unwrap();
        let requests = (0..count)
            .map(|index| {
                EngineRequest::new(
                    RequestId::new(index as u64 + 1),
                    vec![1, 2, 3, index as u32],
                    192,
                    SamplingParameters::new(1.0)
                        .unwrap()
                        .with_top_p(1.0)
                        .unwrap()
                        .with_top_logprobs(5)
                        .unwrap(),
                    710 + index as u64,
                    Vec::new(),
                    vocabulary(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        for rows in [Rows::Logits, Rows::Sampled] {
            let mut reference = Engine::new(base, Model::new(Drafter::Mostly, rows)).unwrap();
            let expected = run(&mut reference, &requests);
            for source in [DraftSource::Host, DraftSource::Bundle] {
                let clock = Arc::new(AtomicU64::new(1));
                let mut model = Model::new(Drafter::Mostly, rows);
                model.clock = Some(Arc::clone(&clock));
                model.scripted_verify_costs =
                    Some([210_000, 185_000, 208_000, 64_000, 64_000].into());
                let mut engine = Engine::new_with_observability(
                    base.with_speculation(
                        speculation(1, 32, SpeculationMode::Adaptive, source)
                            .with_draft_cost_model(runtime_engine::DraftCostModel::PerBlock),
                    ),
                    model,
                    Arc::new(ModelClock(clock)),
                    EngineMetrics::default(),
                )
                .unwrap();
                let actual = run(&mut engine, &requests);
                assert_eq!(actual, expected, "{count} {rows:?} {source:?}");
                assert_eq!(logprob_bits(&actual), logprob_bits(&expected));
                let phases = &engine.executor().scripted_phases;
                assert!(
                    phases.windows(8).any(|w| w.iter().all(|&draft| !draft)),
                    "no fallback interval"
                );
                assert!(
                    phases.windows(3).any(|w| w.iter().all(|&draft| draft)),
                    "no caught-up rechecks: {phases:?}"
                );
                assert!(engine.executor().verify_steps.len() >= 5);
            }
        }
    }
}

#[test]
fn confidence_selected_prefixes_preserve_target_tokens_and_logprob_bits() {
    use runtime_engine::confidence::{
        ConfidenceBudget, ConfidenceCandidate, VerifyBucketCost, allocate,
    };
    // Until the bundle callback exists, the fake sink returns its selected
    // prefixes through the engine's existing shorter-host-proposal path.
    // Target sampling, acceptance, stop handling and commits are production
    // engine code, including seeded draws, penalties and top logprobs.
    for count in [1, 8, 32] {
        let probabilities: Vec<_> = (0..count)
            .map(|item| {
                if count > 1 && item % 3 == 0 {
                    [0.0; 5]
                } else {
                    [0.9, 0.8, 0.1, 1.0, 1.0]
                }
            })
            .collect();
        let candidates: Vec<_> = probabilities
            .iter()
            .map(|probabilities| ConfidenceCandidate {
                probabilities,
                maximum_draft_tokens: 5,
            })
            .collect();
        let plan = allocate(
            &candidates,
            ConfidenceBudget {
                maximum_rows: count * 3,
                fixed_micros: 10.0,
            },
            &[VerifyBucketCost {
                rows: count * 3,
                micros: 20.0,
            }],
        )
        .unwrap();
        assert!(plan.drafts.iter().any(|&k| k < 5));
        let requests = if count == 1 {
            vec![
                EngineRequest::new(
                    RequestId::new(1),
                    vec![1, 2, 3],
                    40,
                    parameters(3, 40),
                    19,
                    vec![STOP],
                    vocabulary(),
                )
                .unwrap(),
            ]
        } else {
            requests(count as u64)
        };
        for rows in [Rows::Logits, Rows::Sampled] {
            let expected = reference(&requests, rows);
            for drafter in [
                Drafter::Greedy,
                Drafter::Wrong,
                Drafter::Mostly,
                Drafter::Noise,
                Drafter::Stop,
            ] {
                let mut model = Model::new(drafter, rows);
                model.confidence_caps = plan.drafts.clone();
                let mut engine = Engine::new(
                    config(Some(speculation(
                        5,
                        64,
                        SpeculationMode::Fixed,
                        DraftSource::Host,
                    ))),
                    model,
                )
                .unwrap();
                let actual = run(&mut engine, &requests);
                assert_eq!(actual, expected, "{count} {rows:?} {drafter:?}");
                let logprob_bits = |outputs: &Outputs| {
                    outputs
                        .values()
                        .flatten()
                        .filter_map(|event| {
                            if let EngineEventKind::Token {
                                logprob,
                                top_logprobs,
                                ..
                            } = event
                            {
                                Some((
                                    logprob.to_bits(),
                                    top_logprobs
                                        .iter()
                                        .map(|p| (p.token_id(), p.logprob().to_bits()))
                                        .collect::<Vec<_>>(),
                                ))
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                };
                assert_eq!(logprob_bits(&actual), logprob_bits(&expected));
                let metrics = engine.metrics().snapshot();
                assert!(metrics.spec_verify_items > 0);
                assert_eq!(
                    metrics.spec_accepted_positions.iter().sum::<u64>(),
                    metrics.spec_accepted_tokens
                );
                assert!(metrics.spec_emitted_tokens <= metrics.spec_rows);
            }
        }
    }
}

#[test]
fn outputs_equal_the_reference_for_many_seeds() {
    for seed in 0..12u64 {
        let requests: Vec<_> = requests(10)
            .into_iter()
            .enumerate()
            .map(|(index, request)| {
                let prompt: Vec<u32> = (0..5)
                    .map(|position| {
                        (mix(seed * 97 + index as u64 * 13 + position) % VOCABULARY as u64) as u32
                    })
                    .collect();
                EngineRequest::new(
                    RequestId::new(index as u64 + 1),
                    prompt,
                    30,
                    parameters(index + seed as usize, 30),
                    mix(seed ^ index as u64),
                    if index % 3 == 0 {
                        vec![STOP]
                    } else {
                        Vec::new()
                    },
                    vocabulary(),
                )
                .unwrap_or(request)
            })
            .collect();
        let expected = reference(&requests, Rows::Sampled);
        let (actual, _) = speculative(
            &requests,
            Drafter::Mostly,
            Rows::Sampled,
            speculation(5, 32, SpeculationMode::Fixed, DraftSource::Bundle),
        );
        assert_eq!(actual, expected, "seed {seed}");
    }
}

#[test]
fn greedy_requests_accept_every_argmax_draft_and_no_wrong_one() {
    let requests: Vec<_> = (0..4)
        .map(|index| {
            EngineRequest::new(
                RequestId::new(index + 1),
                vec![1, 2, 3 + index as u32],
                40,
                SamplingParameters::new(0.0).unwrap(),
                index,
                Vec::new(),
                vocabulary(),
            )
            .unwrap()
        })
        .collect();
    let fixed = speculation(4, 64, SpeculationMode::Fixed, DraftSource::Bundle);
    let (_, right) = speculative(&requests, Drafter::Greedy, Rows::Logits, fixed);
    assert!(right.spec_verify_items > 0);
    assert_eq!(right.spec_accepted_tokens, right.spec_draft_tokens);
    assert_eq!(
        right.spec_emitted_tokens,
        right.spec_draft_tokens + right.spec_verify_items
    );
    // Four requests of 40 tokens: one from prefill and the rest in runs of
    // five, so every request needs eight verify steps.
    assert_eq!(right.spec_verify_items, 4 * 8);
    assert_eq!(right.generation_tokens, 160);

    let (_, wrong) = speculative(&requests, Drafter::Wrong, Rows::Logits, fixed);
    assert_eq!(wrong.spec_accepted_tokens, 0);
    assert_eq!(wrong.spec_emitted_tokens, wrong.spec_verify_items);
    assert_eq!(wrong.generation_tokens, 160);
    // Every verify item drafts at least one token; the first draft is always
    // checked and never accepted.
    assert_eq!(wrong.spec_drafted_positions[0], wrong.spec_verify_items);
    assert_eq!(wrong.spec_accepted_positions[0], 0);
}

#[test]
fn adaptive_policy_stops_drafting_for_a_drafter_that_is_always_wrong() {
    let requests: Vec<_> = (0..2)
        .map(|index| {
            EngineRequest::new(
                RequestId::new(index + 1),
                vec![4, 4, 4],
                200,
                SamplingParameters::new(0.0).unwrap(),
                index,
                Vec::new(),
                vocabulary(),
            )
            .unwrap()
        })
        .collect();
    // Every row costs the same 1 ms, so a draft pays for itself only when it
    // is nearly always accepted, and an always-wrong drafter never is.
    let time = Arc::new(AtomicU64::new(1));
    let mut model = Model::new(Drafter::Wrong, Rows::Logits);
    model.clock = Some(Arc::clone(&time));
    let mut engine = Engine::new_with_observability(
        config(Some(speculation(
            8,
            64,
            SpeculationMode::Adaptive,
            DraftSource::Bundle,
        ))),
        model,
        Arc::new(ModelClock(time)),
        EngineMetrics::default(),
    )
    .unwrap();
    run(&mut engine, &requests);
    let metrics = engine.metrics().snapshot();
    // The first steps draft on the priors. Each rejection lowers the
    // request's estimate for its first draft by an eighth, and within about
    // ten steps per request it falls below the measured cost of a row; the
    // rest of the 199 decode steps per request are plain decodes.
    assert!(metrics.spec_verify_items > 0);
    assert!(
        metrics.spec_verify_items <= 30,
        "{}",
        metrics.spec_verify_items
    );
    assert_eq!(metrics.spec_accepted_tokens, 0);
    assert_eq!(metrics.generation_tokens, 400);
}

#[test]
fn speculation_metrics_are_published_once_a_draft_is_verified() {
    let requests = requests(4);
    let (_, metrics) = speculative(
        &requests,
        Drafter::Greedy,
        Rows::Logits,
        speculation(3, 64, SpeculationMode::Fixed, DraftSource::Bundle),
    );
    let text = metrics.prometheus();
    for name in [
        "chop_spec_verify_items_total",
        "chop_spec_draft_tokens_total",
        "chop_spec_accepted_tokens_total",
        "chop_spec_emitted_tokens_total",
        "chop_spec_rows_total",
        "chop_spec_accepted_position_total{position=\"1\"}",
        "chop_spec_drafted_position_total{position=\"3\"}",
    ] {
        assert!(text.contains(name), "{name} missing from\n{text}");
    }
    let quiet = reference(&requests, Rows::Logits);
    assert!(!quiet.is_empty());
    let mut engine = Engine::new(config(None), Model::new(Drafter::Greedy, Rows::Logits)).unwrap();
    run(&mut engine, &requests);
    assert!(
        !engine
            .metrics()
            .snapshot()
            .prometheus()
            .contains("chop_spec_")
    );
}

#[test]
fn a_draft_row_of_the_wrong_length_fails_the_batch() {
    let mut model = Model::new(Drafter::Greedy, Rows::Logits);
    model.corrupt_drafts = true;
    let mut engine = Engine::new(
        config(Some(speculation(
            3,
            64,
            SpeculationMode::Fixed,
            DraftSource::Bundle,
        ))),
        model,
    )
    .unwrap();
    engine
        .submit(
            EngineRequest::new(
                RequestId::new(1),
                vec![1, 2, 3],
                10,
                SamplingParameters::new(0.0).unwrap(),
                1,
                Vec::new(),
                vocabulary(),
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(engine.step().unwrap(), StepStatus::Progressed);
    assert_eq!(engine.step(), Err(EngineError::OutputMismatch));
    engine.audit().unwrap();
    let events = engine.drain_events();
    assert!(matches!(
        events.last().map(EngineEvent::kind),
        Some(EngineEventKind::Error { .. })
    ));
}

fn prompt_lookup(
    model: Model,
    requests: &[EngineRequest],
    lookup: PromptLookupConfig,
    speculation: SpeculationConfig,
) -> (Outputs, runtime_engine::EngineMetricsSnapshot) {
    let mut engine = Engine::new(
        config(Some(speculation)),
        PromptLookupExecutor::new(model, lookup),
    )
    .unwrap();
    let outputs = run(&mut engine, requests);
    (outputs, engine.metrics().snapshot())
}

fn copying_model(rows: Rows) -> Model {
    let mut model = Model::new(Drafter::Greedy, rows);
    model.copy_period = Some(4);
    model
}

/// Prompts that repeat a four-token cycle, with mixed sampling settings.
fn repetitive_requests(count: u64) -> Vec<EngineRequest> {
    (0..count)
        .map(|index| {
            let cycle: Vec<u32> = (0..4)
                .map(|position| (mix(index * 7 + position) % VOCABULARY as u64) as u32)
                .collect();
            let prompt: Vec<u32> = cycle
                .iter()
                .cycle()
                .take(6 + index as usize % 7)
                .copied()
                .collect();
            let maximum = [3, 16, 40][(index % 3) as usize];
            EngineRequest::new(
                RequestId::new(index + 1),
                prompt,
                maximum,
                parameters(index as usize, maximum),
                mix(index) ^ 0xbeef,
                if index % 2 == 0 {
                    vec![STOP]
                } else {
                    Vec::new()
                },
                vocabulary(),
            )
            .unwrap()
        })
        .collect()
}

#[test]
fn prompt_lookup_drafts_emit_exactly_the_reference_output() {
    for rows in [Rows::Logits, Rows::Sampled] {
        for (requests, copying) in [(requests(20), false), (repetitive_requests(12), true)] {
            let model = || {
                if copying {
                    copying_model(rows)
                } else {
                    Model::new(Drafter::Greedy, rows)
                }
            };
            let mut engine = Engine::new(config(None), model()).unwrap();
            let expected = run(&mut engine, &requests);
            for mode in [SpeculationMode::Fixed, SpeculationMode::Adaptive] {
                for lookup in [
                    PromptLookupConfig::default(),
                    PromptLookupConfig::new(1, 2, 3).unwrap(),
                ] {
                    let (actual, _) = prompt_lookup(
                        model(),
                        &requests,
                        lookup,
                        speculation(6, 32, mode, DraftSource::Host),
                    );
                    assert_eq!(actual, expected, "{rows:?} {mode:?} {lookup:?} {copying}");
                }
            }
        }
    }
}

#[test]
fn prompt_lookup_drafts_are_accepted_on_repetitive_text() {
    let requests = repetitive_requests(6);
    let greedy: Vec<_> = (0..6)
        .map(|index| {
            EngineRequest::new(
                RequestId::new(index + 1),
                vec![3, 1, 4, 1, 3, 1, 4],
                40,
                SamplingParameters::new(0.0).unwrap(),
                index,
                Vec::new(),
                vocabulary(),
            )
            .unwrap()
        })
        .collect();
    for requests in [requests, greedy] {
        let (_, metrics) = prompt_lookup(
            copying_model(Rows::Logits),
            &requests,
            PromptLookupConfig::default(),
            speculation(6, 64, SpeculationMode::Fixed, DraftSource::Host),
        );
        assert!(metrics.spec_verify_items > 0);
        assert!(metrics.spec_accepted_tokens > 0);
        let rate = metrics.spec_accepted_tokens as f64 / metrics.spec_draft_tokens as f64;
        assert!(rate > 0.5, "acceptance {rate}");
    }
}

#[test]
fn speculative_itl_matches_vllm_output_update_intervals() {
    // vLLM's IterationStats.update_from_output records one timestamp delta
    // per non-prefill output update, including multi-token speculative runs.
    // It does not divide this histogram observation by the emitted-token count.
    let prompt = (0..VOCABULARY as u32)
        .map(|token| vec![token])
        .find(|prompt| {
            let chain = Drafter::Greedy.propose(prompt, 3);
            chain[2] != chain[0] && chain[2] != chain[1]
        })
        .unwrap();
    let stop = Drafter::Greedy.propose(&prompt, 3)[2];
    for rows in [Rows::Logits, Rows::Sampled] {
        for (drafter, stops, emitted) in [
            (Drafter::Greedy, vec![], 8),
            (Drafter::Wrong, vec![], 1),
            (Drafter::Greedy, vec![stop], 2),
        ] {
            let time = Arc::new(AtomicU64::new(1));
            let mut model = Model::new(drafter, rows);
            model.clock = Some(Arc::clone(&time));
            let metrics = EngineMetrics::default();
            let mut engine = Engine::new_with_observability(
                config(Some(speculation(
                    7,
                    64,
                    SpeculationMode::Fixed,
                    DraftSource::Bundle,
                ))),
                model,
                Arc::new(ModelClock(time)),
                metrics.clone(),
            )
            .unwrap();
            engine
                .submit(
                    EngineRequest::new(
                        RequestId::new(1),
                        prompt.clone(),
                        20,
                        SamplingParameters::new(0.0).unwrap(),
                        0,
                        stops,
                        vocabulary(),
                    )
                    .unwrap(),
                )
                .unwrap();
            engine.step().unwrap();
            assert_eq!(metrics.snapshot().ttft_micros, 1000);
            assert_eq!(metrics.snapshot().itl_observations, 0);
            engine.step().unwrap();
            let snapshot = metrics.snapshot();
            assert_eq!(snapshot.generation_tokens, 1 + emitted);
            assert_eq!(snapshot.itl_observations, 1);
            assert_eq!(snapshot.itl_micros, 8000);
            let text = metrics.prometheus_compatibility("fixture");
            let labels = "model_name=\"fixture\",engine=\"0\"";
            assert!(text.contains(&format!(
                "vllm:inter_token_latency_seconds_count{{{labels}}} 1\n"
            )));
            assert!(text.contains(&format!(
                "vllm:inter_token_latency_seconds_sum{{{labels}}} 0.008\n"
            )));
            assert!(text.contains(&format!(
                "vllm:inter_token_latency_seconds_bucket{{{labels},le=\"0.01\"}} 1\n"
            )));
        }
    }
}

#[test]
fn pinned_drafts_override_the_policy_and_keep_the_output() {
    let requests: Vec<_> = (0..3)
        .map(|index| {
            let request = EngineRequest::new(
                RequestId::new(index + 1),
                vec![4, 4, 4],
                60,
                SamplingParameters::new(0.0).unwrap(),
                index,
                Vec::new(),
                vocabulary(),
            )
            .unwrap();
            // Request 1 always verifies three drafts, request 2 never drafts
            // and request 3 is left to the policy.
            match index {
                0 => request.with_pinned_draft_tokens(3),
                1 => request.with_pinned_draft_tokens(0),
                _ => request,
            }
        })
        .collect();
    let expected = reference(&requests, Rows::Logits);
    // An always-wrong drafter: the adaptive policy soon stops drafting, but
    // the pin does not, and every rejected draft still emits one token.
    let (outputs, metrics) = speculative(
        &requests,
        Drafter::Wrong,
        Rows::Logits,
        speculation(8, 64, SpeculationMode::Adaptive, DraftSource::Bundle),
    );
    assert_eq!(outputs, expected);
    assert_eq!(metrics.spec_accepted_tokens, 0);
    // Request 1 verifies in each of its decode steps but the last, which
    // has no output left to draft for; request 3 only while the policy
    // still drafts for it.
    assert!(
        (58..=78).contains(&metrics.spec_verify_items),
        "{}",
        metrics.spec_verify_items
    );
    // The pinned request drafts three positions while its output allows.
    assert!(metrics.spec_drafted_positions[2] >= 56);
}

#[test]
fn verify_warmup_waves_reach_every_planned_shape() {
    use runtime_engine::warmup::{VerifyWarmup, verify_shapes};
    let warmup = VerifyWarmup {
        item_rows: vec![2, 4, 8],
        maximum_draft_tokens: 7,
        step_rows: 64,
        drafting_items: None,
        exact_graphs: false,
        widths: Vec::new(),
    };
    let shapes = verify_shapes(6, &warmup);
    assert!(shapes.len() >= 12, "{shapes:?}");
    let config = EngineConfig::new(
        SchedulerConfig::new(6, 64, 256, 16, 1).unwrap(),
        KvGeometry::new(4, KV_BLOCKS).unwrap(),
        vocabulary(),
        256,
        64,
    )
    .unwrap()
    .with_speculation(speculation(
        7,
        64,
        SpeculationMode::Adaptive,
        DraftSource::Bundle,
    ));
    let mut engine = Engine::new(config, Model::new(Drafter::Wrong, Rows::Logits)).unwrap();
    let mut next = 1;
    for shape in &shapes {
        // One wave, as startup warmup submits it: every request back to
        // back, each pinned, with room for the deepest pin.
        let wave: Vec<_> = shape
            .drafts
            .iter()
            .map(|&drafts| {
                next += 1;
                EngineRequest::new(
                    RequestId::new(next),
                    vec![4, 4],
                    warmup.maximum_draft_tokens + 2,
                    SamplingParameters::new(0.0).unwrap(),
                    0,
                    Vec::new(),
                    vocabulary(),
                )
                .unwrap()
                .with_pinned_draft_tokens(drafts)
            })
            .collect();
        for request in wave {
            engine.submit(request).unwrap();
        }
        let before = engine.executor().verify_steps.len();
        while engine.step().unwrap() != StepStatus::Idle {}
        let _ = engine.drain_events();
        let reached = engine.executor().verify_steps[before..]
            .iter()
            .any(|&(rows, items)| {
                rows.next_power_of_two() == shape.row_bucket
                    && items.next_power_of_two() == shape.item_bucket
            });
        assert!(
            reached,
            "{shape:?}: {:?}",
            &engine.executor().verify_steps[before..]
        );
    }
    assert_eq!(engine.kv_stats().free_blocks(), KV_BLOCKS);
}

fn logprob_bits(outputs: &Outputs) -> Vec<(u64, Vec<(u32, u64)>)> {
    outputs
        .values()
        .flatten()
        .filter_map(|event| {
            if let EngineEventKind::Token {
                logprob,
                top_logprobs,
                ..
            } = event
            {
                Some((
                    logprob.to_bits(),
                    top_logprobs
                        .iter()
                        .map(|p| (p.token_id(), p.logprob().to_bits()))
                        .collect(),
                ))
            } else {
                None
            }
        })
        .collect()
}

/// Tables that draft deep, draft shallow on every item, and never draft.
fn tables() -> [runtime_engine::StepCosts; 3] {
    [
        "1:16000:0:4000:800,6:30000:0:3000:800",
        "1:16000:0:100:9000,6:30000:0:100:5000",
        "1:16000:0:90000:800,6:30000:0:90000:800",
    ]
    .map(|text| runtime_engine::StepCosts::parse(text).unwrap())
}

#[test]
fn table_policies_emit_the_reference_tokens_and_logprob_bits() {
    let requests = requests(20);
    for rows in [Rows::Logits, Rows::Sampled] {
        let expected = reference(&requests, rows);
        for drafter in [
            Drafter::Greedy,
            Drafter::Wrong,
            Drafter::Noise,
            Drafter::Stop,
            Drafter::Mostly,
        ] {
            for (index, costs) in tables().into_iter().enumerate() {
                for source in [DraftSource::Bundle, DraftSource::Host] {
                    for (drafts, verify_rows) in [(4, 64), (5, 24), (7, 12)] {
                        let (actual, metrics) = speculative(
                            &requests,
                            drafter,
                            rows,
                            speculation(drafts, verify_rows, SpeculationMode::Table(costs), source)
                                .with_draft_cost_model(runtime_engine::DraftCostModel::PerBlock),
                        );
                        assert_eq!(
                            actual, expected,
                            "{drafter:?} {rows:?} table {index} {source:?} K={drafts}"
                        );
                        assert_eq!(logprob_bits(&actual), logprob_bits(&expected));
                        if index == 2 {
                            // Drafting costs more than any expected gain.
                            assert_eq!(metrics.spec_verify_items, 0);
                        } else if matches!(drafter, Drafter::Greedy) {
                            assert!(metrics.spec_verify_items > 0);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn table_plans_do_not_depend_on_measured_step_times() {
    let requests = requests(20);
    let costs = tables()[0];
    let mut steps = Vec::new();
    // Rows cost 1 ms or 50 ms on the engine's clock; the plans are the same.
    for per_row in [1_000, 50_000] {
        let time = Arc::new(AtomicU64::new(1));
        let mut model = Model::new(Drafter::Mostly, Rows::Logits);
        model.clock = Some(Arc::clone(&time));
        model.micros_per_row = per_row;
        let mut engine = Engine::new_with_observability(
            config(Some(speculation(
                5,
                24,
                SpeculationMode::Table(costs),
                DraftSource::Bundle,
            ))),
            model,
            Arc::new(ModelClock(time)),
            EngineMetrics::default(),
        )
        .unwrap();
        let outputs = run(&mut engine, &requests);
        assert_eq!(outputs, reference(&requests, Rows::Logits));
        steps.push(engine.executor().verify_steps.clone());
    }
    assert!(!steps[0].is_empty());
    assert_eq!(steps[0], steps[1]);
}
