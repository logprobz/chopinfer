//! Constrained requests emit only what their grammar admits, with or without
//! speculation, whether the host or the executor samples.

use std::collections::BTreeMap;
use std::sync::Arc;

use runtime_engine::{
    DraftRow, DraftSource, Engine, EngineConfig, EngineError, EngineEventKind, EngineRequest,
    ExecutorError, FinishReason, KeptRows, LogitsRow, ModelBatchItem, ModelExecutor,
    ModelTransaction, ModelTransactionState, PromptLookupConfig, PromptLookupExecutor,
    RowConstraint, SampledRow, SpeculationConfig, SpeculationMode, StepStatus,
};
use runtime_grammar::{
    CompiledGrammar, ConstraintState, JsonOptions, TokenVocabulary, VocabularyToken,
    json_schema_grammar,
};
use runtime_kv::{KvGeometry, PreparedAppendBatch};
use runtime_model_api::ModelVocabulary;
use runtime_sampling::{SamplingError, SamplingParameters, sample_for_vocabulary};
use runtime_scheduler::{RequestId, SchedulerConfig};
use serde_json::{Value, json};

/// Token texts; the stop and gate tokens have no grammar bytes.
const TEXTS: [&str; 24] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "[", "]", ",", " ", "[1", "1,", "],", "0]",
    ", 4", "x", "\"", "", "12", "",
];
const STOP: u32 = 21;
const GATE: u32 = 23;
const VOCABULARY: usize = 26;
const KV_BLOCKS: usize = 256;

fn vocabulary() -> ModelVocabulary {
    ModelVocabulary::new(VOCABULARY).unwrap()
}

fn token_vocabulary() -> Arc<TokenVocabulary> {
    let tokens = TEXTS
        .iter()
        .enumerate()
        .map(|(id, text)| VocabularyToken {
            bytes: text.as_bytes().to_vec(),
            special: id as u32 == STOP || id as u32 == GATE,
        })
        .collect();
    Arc::new(TokenVocabulary::new(VOCABULARY, tokens).unwrap())
}

fn schema() -> Value {
    json!({"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 99}, "maxItems": 4})
}

fn grammar() -> Arc<CompiledGrammar> {
    let grammar = json_schema_grammar(&schema(), JsonOptions::default()).unwrap();
    Arc::new(CompiledGrammar::new(&grammar, token_vocabulary()).unwrap())
}

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Hash noise over the prefix, which favours the stop token a little so that
/// unconstrained rows also end.
fn logits_for(prefix: &[u32]) -> Vec<f32> {
    let hash = prefix
        .iter()
        .fold(0x1234_5678_u64, |hash, token| mix(hash ^ u64::from(*token)));
    let mut logits: Vec<f32> = (0..VOCABULARY as u64)
        .map(|token| (mix(hash ^ (token << 40)) >> 40) as f32 / (1u64 << 24) as f32 * 6.0 - 3.0)
        .collect();
    logits[STOP as usize] += 0.5;
    logits
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Rows {
    Logits,
    Sampled,
    /// Sampled rows that ignore the mask, as a faulty device sampler would.
    Unmasked,
    InvalidTop,
}

struct Model {
    rows: Rows,
    kv: BTreeMap<RequestId, Vec<u32>>,
    pending: Vec<(RequestId, Vec<u32>, bool)>,
    /// Updates state in place: a rollback invalidates every touched
    /// request, which may then only retire, never continue.
    in_place: bool,
    invalid: std::collections::BTreeSet<RequestId>,
    /// Requests whose rows ignore the grammar mask once their prefix
    /// reaches this length, so their sampling fails at the first disallowed
    /// token after it.
    unmasked: BTreeMap<RequestId, usize>,
}

impl Model {
    fn new(rows: Rows) -> Self {
        Self {
            rows,
            kv: BTreeMap::new(),
            pending: Vec::new(),
            in_place: false,
            invalid: std::collections::BTreeSet::new(),
            unmasked: BTreeMap::new(),
        }
    }

    /// Noisy drafts that often break the grammar.
    fn propose(context: &[u32], count: usize) -> Vec<u32> {
        (0..count)
            .map(|index| (mix(context.len() as u64 * 7 + index as u64) % 21) as u32)
            .collect()
    }
}

struct ModelState<'a> {
    model: &'a mut Model,
}

impl ModelTransactionState for ModelState<'_> {
    fn commit(&mut self) -> Result<(), ExecutorError> {
        for (id, inputs, _) in self.model.pending.drain(..) {
            self.model.kv.entry(id).or_default().extend(inputs);
        }
        Ok(())
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
                .kv
                .entry(id)
                .or_default()
                .extend_from_slice(&inputs[..count]);
        }
        Ok(())
    }

    fn rollback(&mut self) -> Result<(), ExecutorError> {
        for (id, _, _) in self.model.pending.drain(..) {
            if self.model.in_place {
                self.model.invalid.insert(id);
            }
        }
        Ok(())
    }
}

impl ModelExecutor for Model {
    type TransactionState<'a> = ModelState<'a>;

    fn rollback_preserves_state_v1(&self) -> bool {
        !self.in_place
    }

    fn retire_request_v1(&mut self, id: RequestId) -> Result<(), ExecutorError> {
        self.kv.remove(&id);
        self.invalid.remove(&id);
        Ok(())
    }

    fn draft_tokens_v1(
        &mut self,
        _id: RequestId,
        history: &[u32],
        maximum: usize,
    ) -> Result<Vec<u32>, ExecutorError> {
        Ok(Self::propose(history, maximum))
    }

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        kv_plan: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        let mut logits_rows = Vec::new();
        let mut sampled_rows = Vec::new();
        let mut draft_rows = Vec::new();
        for (item, append) in items.iter().zip(kv_plan.appends()) {
            let committed = self.kv.get(&item.request_id()).cloned().unwrap_or_default();
            assert!(
                !self.invalid.contains(&item.request_id()),
                "request {:?} continued from state a rollback invalidated",
                item.request_id()
            );
            let mut inputs = item.input_tokens().to_vec();
            let mut drafts = Vec::new();
            if let Some(draft) = item.draft() {
                match draft.source() {
                    DraftSource::Host => drafts = inputs[1..].to_vec(),
                    DraftSource::Bundle => {
                        let mut context = committed.clone();
                        context.push(inputs[0]);
                        drafts = Self::propose(&context, draft.tokens());
                        inputs.extend_from_slice(&drafts);
                        draft_rows.push(DraftRow::new(item.request_id(), drafts.clone()));
                    }
                }
            }
            assert_eq!(inputs.len(), append.additional_tokens());
            if let Some(work) = item.sampling() {
                let first_row = if item.draft().is_some() {
                    0
                } else {
                    inputs.len() - 1
                };
                for input in first_row..inputs.len() {
                    let mut prefix = committed.clone();
                    prefix.extend_from_slice(&inputs[..=input]);
                    let row_work = work.for_verify_row(&drafts, input - first_row);
                    let logits = logits_for(&prefix);
                    if self.rows == Rows::Logits {
                        logits_rows.push(LogitsRow::new(item.request_id(), logits).unwrap());
                        continue;
                    }
                    let context = row_work
                        .history
                        .context(row_work.emitted_tokens, &row_work.stop_ids);
                    let mask = row_work.constraint.as_ref().and_then(|row| row.mask());
                    let masked = self
                        .unmasked
                        .get(&item.request_id())
                        .is_none_or(|&length| prefix.len() < length);
                    let context = match (self.rows, mask) {
                        (Rows::Sampled | Rows::InvalidTop, Some(mask)) if masked => {
                            context.with_allowed_tokens(mask)
                        }
                        _ => context,
                    };
                    let mut state = row_work.state;
                    let sample = sample_for_vocabulary(
                        &row_work.parameters,
                        &logits,
                        vocabulary(),
                        context,
                        &mut state,
                    )
                    .unwrap();
                    let mut top = sample.top_logprobs().to_vec();
                    if self.rows == Rows::InvalidTop && !top.is_empty() {
                        top[0] = runtime_sampling::TokenLogprob::new(
                            sample.token_id(),
                            f64::NEG_INFINITY,
                        );
                    }
                    sampled_rows.push(SampledRow::new(
                        item.request_id(),
                        sample.token_id(),
                        sample.logprob(),
                        top,
                    ));
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

fn parameters(index: u64) -> SamplingParameters {
    match index % 3 {
        0 => SamplingParameters::new(0.0)
            .unwrap()
            .with_top_logprobs(3)
            .unwrap(),
        1 => SamplingParameters::new(1.0).unwrap(),
        _ => SamplingParameters::new(0.8)
            .unwrap()
            .with_top_p(0.9)
            .unwrap()
            .with_minimum_new_tokens(3)
            .unwrap(),
    }
}

fn requests(count: u64, grammar: &Arc<CompiledGrammar>, gated: bool) -> Vec<EngineRequest> {
    (0..count)
        .map(|index| {
            let prompt: Vec<u32> = (0..2 + index % 5)
                .map(|position| (mix(index * 31 + position) % 20) as u32)
                .collect();
            let gate = gated.then(|| vec![GATE]);
            EngineRequest::new(
                RequestId::new(index + 1),
                prompt,
                40,
                parameters(index),
                mix(index) ^ 0xfeed,
                vec![STOP],
                vocabulary(),
            )
            .unwrap()
            .with_constraint(ConstraintState::new(Arc::clone(grammar), gate, vec![STOP]))
            .unwrap()
        })
        .collect()
}

type Outputs = BTreeMap<RequestId, (Vec<u32>, Vec<f64>, Option<FinishReason>)>;

fn run(engine: &mut Engine<Model>, requests: &[EngineRequest]) -> Result<Outputs, EngineError> {
    let mut outputs = Outputs::new();
    let mut pending = requests.iter();
    for step in 0..10_000 {
        if step % 2 == 0
            && let Some(request) = pending.next()
        {
            engine.submit(request.clone()).unwrap();
        }
        let status = engine.step()?;
        for event in engine.drain_events() {
            let entry = outputs.entry(event.request_id()).or_default();
            match event.kind() {
                EngineEventKind::Token {
                    token_id, logprob, ..
                } => {
                    entry.0.push(*token_id);
                    entry.1.push(*logprob);
                }
                EngineEventKind::Finished { reason } => entry.2 = Some(*reason),
                _ => {}
            }
        }
        if status == StepStatus::Idle && pending.len() == 0 {
            assert_eq!(engine.kv_stats().free_blocks(), KV_BLOCKS);
            return Ok(outputs);
        }
    }
    panic!("the engine did not finish");
}

fn text(tokens: &[u32]) -> String {
    tokens
        .iter()
        .filter(|&&token| token != STOP)
        .map(|&token| TEXTS[token as usize])
        .collect()
}

fn valid(value: &Value) -> bool {
    let Value::Array(items) = value else {
        return false;
    };
    items.len() <= 4
        && items
            .iter()
            .all(|item| item.as_u64().is_some_and(|number| number <= 99))
}

fn check(outputs: &Outputs) {
    let mut finished = 0;
    for (tokens, logprobs, reason) in outputs.values() {
        assert!(logprobs.iter().all(|logprob| *logprob <= 1e-9));
        match reason {
            Some(FinishReason::Stop) => {
                finished += 1;
                assert_eq!(tokens.last(), Some(&STOP));
                let value: Value = serde_json::from_str(&text(tokens)).unwrap();
                assert!(valid(&value), "{value}");
            }
            Some(FinishReason::Length) => {
                // A truncated output is still a prefix the grammar admits.
                let mut state = ConstraintState::new(grammar(), None, vec![STOP]);
                for &token in tokens {
                    state.accept(token).unwrap();
                }
            }
            other => panic!("unexpected finish {other:?}"),
        }
    }
    assert!(finished > outputs.len() / 2);
}

#[test]
fn constrained_outputs_are_valid_and_speculation_changes_nothing() {
    let grammar = grammar();
    let requests = requests(24, &grammar, false);
    for rows in [Rows::Logits, Rows::Sampled] {
        let mut plain = Engine::new(config(None), Model::new(rows)).unwrap();
        let baseline = run(&mut plain, &requests).unwrap();
        check(&baseline);
        for source in [DraftSource::Host, DraftSource::Bundle] {
            let speculation =
                SpeculationConfig::new(3, 24, SpeculationMode::Fixed, source).unwrap();
            let mut engine = Engine::new(config(Some(speculation)), Model::new(rows)).unwrap();
            let outputs = run(&mut engine, &requests).unwrap();
            assert_eq!(outputs, baseline, "{rows:?} with {source:?} drafts");
        }
    }
}

#[test]
fn a_gate_holds_the_constraint_until_it_opens() {
    let grammar = grammar();
    let requests = requests(12, &grammar, true);
    let mut engine = Engine::new(config(None), Model::new(Rows::Logits)).unwrap();
    let outputs = run(&mut engine, &requests).unwrap();
    let mut opened = 0;
    for (tokens, _, reason) in outputs.values() {
        let Some(gate) = tokens.iter().position(|&token| token == GATE) else {
            continue;
        };
        opened += 1;
        let after = &tokens[gate + 1..];
        if *reason == Some(FinishReason::Stop) && after.last() == Some(&STOP) {
            let value: Value = serde_json::from_str(&text(after)).unwrap();
            assert!(valid(&value), "{value}");
        }
    }
    // Before the gate the rows are unconstrained, so outputs differ from the
    // grammar; some requests reach the gate.
    assert!(opened > 0);
    assert!(
        outputs
            .values()
            .any(|(tokens, _, _)| tokens.iter().take_while(|&&t| t != GATE).any(|&t| t == 19))
    );
}

#[test]
fn a_sampler_that_ignores_the_mask_fails_the_request() {
    let grammar = grammar();
    let requests = requests(4, &grammar, false);
    let mut engine = Engine::new(config(None), Model::new(Rows::Unmasked)).unwrap();
    let mut failed = false;
    for request in &requests {
        engine.submit(request.clone()).unwrap();
    }
    for _ in 0..200 {
        match engine.step() {
            Err(EngineError::Sampling(SamplingError::DisallowedToken(_))) => failed = true,
            Err(error) => panic!("{error}"),
            Ok(StepStatus::Idle) => break,
            Ok(StepStatus::Progressed) => {}
        }
        let _ = engine.drain_events();
    }
    assert!(failed);
}

#[test]
fn a_constraint_for_another_vocabulary_is_refused() {
    let tokens = vec![
        VocabularyToken {
            bytes: b"x".to_vec(),
            special: false,
        };
        3
    ];
    let narrow = Arc::new(TokenVocabulary::new(3, tokens).unwrap());
    let grammar = json_schema_grammar(&json!({"const": "x"}), JsonOptions::default()).unwrap();
    let compiled = Arc::new(CompiledGrammar::new(&grammar, narrow).unwrap());
    let request = EngineRequest::new(
        RequestId::new(1),
        vec![1],
        4,
        SamplingParameters::new(0.0).unwrap(),
        0,
        Vec::new(),
        vocabulary(),
    )
    .unwrap();
    assert!(
        request
            .with_constraint(ConstraintState::new(compiled, None, Vec::new()))
            .is_err()
    );
}

#[test]
fn verify_rows_carry_masks_after_their_drafts() {
    let grammar = grammar();
    let state = ConstraintState::new(Arc::clone(&grammar), None, vec![STOP]);
    let first = RowConstraint::new(state.fork());
    assert!(first.allows(10) && !first.allows(11) && !first.allows(STOP));
    // Row 2 after the drafts "[" and "1" sees the mask of that position.
    let row = first.after(&[10, 1]);
    let mut advanced = state.fork();
    advanced.accept(10).unwrap();
    advanced.accept(1).unwrap();
    assert_eq!(row, RowConstraint::new(advanced.fork()));
    assert!(row.allows(11) && row.allows(12) && !row.allows(10));
    // A draft the grammar refuses leaves the rows after it at its own mask.
    let blocked = first.after(&[10, 19, 1]);
    let mut open = state.fork();
    open.accept(10).unwrap();
    assert_eq!(blocked, RowConstraint::new(open));
}

#[test]
fn negative_infinite_top_logprobs_require_a_masked_token() {
    let request = requests(1, &grammar(), false).remove(0);
    let mut engine = Engine::new(config(None), Model::new(Rows::InvalidTop)).unwrap();
    engine.submit(request).unwrap();
    let mut rejected = false;
    for _ in 0..20 {
        match engine.step() {
            Err(EngineError::Sampling(SamplingError::InvalidDistribution)) => rejected = true,
            Err(error) => panic!("{error}"),
            Ok(StepStatus::Idle) => break,
            Ok(StepStatus::Progressed) => {}
        }
        let _ = engine.drain_events();
    }
    assert!(rejected);
}

/// A run in which a step whose sampling failed for some request is
/// progress, since the engine has already failed that request: each
/// request's outputs and the requests that ended with an error.
fn run_with_failures<E: ModelExecutor>(
    engine: &mut Engine<E>,
    requests: &[EngineRequest],
) -> (Outputs, BTreeMap<RequestId, usize>) {
    let mut outputs = Outputs::new();
    let mut errors = BTreeMap::new();
    let mut pending = requests.iter();
    for step in 0..10_000 {
        if step % 2 == 0
            && let Some(request) = pending.next()
        {
            engine.submit(request.clone()).unwrap();
        }
        let status = match engine.step() {
            Ok(status) => status,
            Err(EngineError::Sampling(_) | EngineError::Constraint(_)) => StepStatus::Progressed,
            Err(error) => panic!("{error}"),
        };
        engine.audit().unwrap();
        for event in engine.drain_events() {
            let entry = outputs.entry(event.request_id()).or_default();
            match event.kind() {
                EngineEventKind::Token {
                    token_id, logprob, ..
                } => {
                    entry.0.push(*token_id);
                    entry.1.push(*logprob);
                }
                EngineEventKind::Finished { reason } => entry.2 = Some(*reason),
                EngineEventKind::Error { .. } => {
                    *errors.entry(event.request_id()).or_default() += 1
                }
                _ => {}
            }
        }
        if status == StepStatus::Idle && pending.len() == 0 {
            assert_eq!(engine.kv_stats().free_blocks(), KV_BLOCKS);
            return (outputs, errors);
        }
    }
    panic!("the engine did not finish");
}

/// A request whose sampling fails must not take its batch with it, whether
/// or not the executor can roll a step back. Four requests sample without
/// their grammar mask after two tokens, so they fail mid-generation, in
/// verify rows or plain decode rows. Every other request, drafting or not,
/// emits exactly what it emits without them, each failing request ends
/// with one error, and no request continues from state a rollback
/// invalidated. Checked directly with host and bundle drafts, and through
/// the prompt-lookup adapter, which must pass the executor's rollback
/// contract on.
#[test]
fn sampling_failures_fail_only_their_requests() {
    let grammar = grammar();
    let requests = requests(24, &grammar, false);
    // `requests` numbers requests from 1; request i + 1 has a prompt of
    // 2 + i % 5 tokens.
    let unmasked: BTreeMap<RequestId, usize> = [1_u64, 6, 10, 17]
        .into_iter()
        .map(|index| (RequestId::new(index + 1), 2 + index as usize % 5 + 2))
        .collect();
    let healthy: Vec<_> = requests
        .iter()
        .enumerate()
        .filter(|(index, _)| !unmasked.contains_key(&RequestId::new(*index as u64 + 1)))
        .map(|(_, request)| request.clone())
        .collect();
    let mut plain = Engine::new(config(None), Model::new(Rows::Sampled)).unwrap();
    let expected = run(&mut plain, &healthy).unwrap();
    let isolated = |(outputs, errors): (Outputs, BTreeMap<RequestId, usize>), label: &str| {
        assert_eq!(
            errors,
            unmasked.keys().map(|&id| (id, 1)).collect(),
            "{label}"
        );
        for (id, output) in &outputs {
            if unmasked.contains_key(id) {
                // Two masked tokens, then the failure.
                assert!(output.0.len() >= 2 && output.2.is_none(), "{label} {id:?}");
            } else {
                assert_eq!(Some(output), expected.get(id), "{label} {id:?}");
            }
        }
        assert_eq!(outputs.len(), requests.len(), "{label}");
    };
    let model = |in_place| {
        let mut model = Model::new(Rows::Sampled);
        model.in_place = in_place;
        model.unmasked = unmasked.clone();
        model
    };
    for in_place in [true, false] {
        for source in [DraftSource::Host, DraftSource::Bundle] {
            let speculation =
                SpeculationConfig::new(3, 24, SpeculationMode::Fixed, source).unwrap();
            let mut engine = Engine::new(config(Some(speculation)), model(in_place)).unwrap();
            isolated(
                run_with_failures(&mut engine, &requests),
                &format!("in place {in_place} {source:?}"),
            );
            assert!(engine.metrics().snapshot().spec_verify_items > 0);
        }
        let speculation =
            SpeculationConfig::new(3, 24, SpeculationMode::Fixed, DraftSource::Host).unwrap();
        let mut engine = Engine::new(
            config(Some(speculation)),
            PromptLookupExecutor::new(model(in_place), PromptLookupConfig::default()),
        )
        .unwrap();
        isolated(
            run_with_failures(&mut engine, &requests),
            &format!("prompt lookup, in place {in_place}"),
        );
    }
}

#[test]
fn exhausted_grammar_fails_without_emitting_or_disabling_the_constraint() {
    for scanner in [false, true] {
        let schema = if scanner {
            json!({"$defs": {
                "a": {"type":"array", "items":{"$ref":"#/$defs/u"}},
                "b": {"type":"array", "items":{"$ref":"#/$defs/u"}, "minItems":1},
                "u": {"anyOf":[{"$ref":"#/$defs/a"},{"$ref":"#/$defs/b"}]}
            }, "$ref":"#/$defs/u"})
        } else {
            json!({"$defs":{"u":{"anyOf":[{"$ref":"#/$defs/u"},{"type":"null"}]}},"$ref":"#/$defs/u"})
        };
        let mut tokens: Vec<_> = TEXTS
            .iter()
            .enumerate()
            .map(|(id, text)| VocabularyToken {
                bytes: text.as_bytes().to_vec(),
                special: id == STOP as usize || id == GATE as usize,
            })
            .collect();
        tokens.push(VocabularyToken {
            bytes: vec![b'['; 32],
            special: false,
        });
        let bad = Arc::new(
            CompiledGrammar::new(
                &json_schema_grammar(&schema, JsonOptions::default()).unwrap(),
                Arc::new(TokenVocabulary::new(VOCABULARY, tokens).unwrap()),
            )
            .unwrap(),
        );
        let healthy = requests(2, &grammar(), false).pop().unwrap();
        let mut control = Engine::new(config(None), Model::new(Rows::Logits)).unwrap();
        let expected = run(&mut control, std::slice::from_ref(&healthy)).unwrap();
        for stop_tokens in [vec![], vec![STOP]] {
            for gate in [None, Some(vec![GATE])] {
                // Scanner work starts once the gate opens. Initial exhaustion
                // is already an error, even before a gated row can emit.
                if scanner && gate.is_some() {
                    continue;
                }
                let request = requests(1, &bad, false)
                    .pop()
                    .unwrap()
                    .with_constraint(ConstraintState::new(bad.clone(), gate, stop_tokens.clone()))
                    .unwrap();
                let mut engine = Engine::new(config(None), Model::new(Rows::Logits)).unwrap();
                let (output, errors) = run_with_failures(&mut engine, &[request, healthy.clone()]);
                assert_eq!(errors, BTreeMap::from([(RequestId::new(1), 1)]));
                assert!(output[&RequestId::new(1)].0.is_empty());
                assert_eq!(
                    output.get(&RequestId::new(2)),
                    expected.get(&RequestId::new(2))
                );
            }
        }
    }
}

#[test]
fn speculative_budget_failure_never_reuses_the_last_valid_mask() {
    let schema = json!({"$defs": {
        "a": {"type":"array", "items":{"$ref":"#/$defs/u"}},
        "b": {"type":"array", "items":{"$ref":"#/$defs/u"}, "minItems":1},
        "u": {"anyOf":[{"$ref":"#/$defs/a"},{"$ref":"#/$defs/b"}]}
    }, "$ref":"#/$defs/u"});
    let compiled = Arc::new(
        CompiledGrammar::new(
            &json_schema_grammar(&schema, JsonOptions::default()).unwrap(),
            token_vocabulary(),
        )
        .unwrap(),
    );
    let row = RowConstraint::new(ConstraintState::new(compiled, None, vec![]));
    assert!(row.allows(10));
    let failed = row.after(&[10; 32]);
    assert!(
        failed
            .mask()
            .is_some_and(|mask| mask.iter().all(|word| *word == 0))
    );
    assert!(!failed.allows(10));
}

#[test]
fn grammar_budget_reason_reaches_request_termination() {
    for (schema, scanner, depth, reason) in [
        (
            json!({"$defs":{"u":{"anyOf":[{"$ref":"#/$defs/u"},{"type":"null"}]}},"$ref":"#/$defs/u"}),
            false,
            false,
            "grammar matcher work budget exceeded (limit: 4096 units per closure)",
        ),
        (
            json!({"$defs": {
            "a": {"type":"array", "items":{"$ref":"#/$defs/u"}},
            "b": {"type":"array", "items":{"$ref":"#/$defs/u"}, "minItems":1},
            "u": {"anyOf":[{"$ref":"#/$defs/a"},{"$ref":"#/$defs/b"}]}
        }, "$ref":"#/$defs/u"}),
            true,
            false,
            "grammar scanner work budget exceeded (limit: 4096 units per closure)",
        ),
        (
            json!({"$defs":{"u":{"anyOf":[{"type":"null"},{"type":"array","items":{"$ref":"#/$defs/u"}}]}},"$ref":"#/$defs/u"}),
            false,
            true,
            "grammar matcher depth budget exceeded (limit: 2048 frames)",
        ),
    ] {
        let vocabulary = if scanner {
            let mut tokens: Vec<_> = TEXTS
                .iter()
                .enumerate()
                .map(|(id, text)| VocabularyToken {
                    bytes: text.as_bytes().to_vec(),
                    special: id == STOP as usize || id == GATE as usize,
                })
                .collect();
            tokens.push(VocabularyToken {
                bytes: vec![b'['; 32],
                special: false,
            });
            Arc::new(TokenVocabulary::new(VOCABULARY, tokens).unwrap())
        } else {
            token_vocabulary()
        };
        let compiled = Arc::new(
            CompiledGrammar::new(
                &json_schema_grammar(&schema, JsonOptions::default()).unwrap(),
                vocabulary,
            )
            .unwrap(),
        );
        for rows in [Rows::Logits, Rows::Sampled] {
            for in_place in [false, true] {
                let mut state = ConstraintState::new(compiled.clone(), None, vec![]);
                if depth {
                    for _ in 0..2046 {
                        state.accept(10).unwrap();
                    }
                }
                let request = requests(1, &grammar(), false)
                    .pop()
                    .unwrap()
                    .with_constraint(state)
                    .unwrap();
                let mut model = Model::new(rows);
                model.in_place = in_place;
                // Deliberately bypass the executor mask to test engine validation too.
                if rows == Rows::Sampled {
                    model.unmasked.insert(RequestId::new(1), 0);
                }
                let mut engine = Engine::new(config(None), model).unwrap();
                engine.submit(request).unwrap();
                let mut errors = Vec::new();
                for _ in 0..100 {
                    let step = engine.step();
                    engine.audit().unwrap();
                    for event in engine.drain_events() {
                        match event.kind() {
                            EngineEventKind::Error { message } => errors.push(message.clone()),
                            EngineEventKind::Token { .. } | EngineEventKind::Finished { .. } => {
                                panic!("budget failure must terminate with an error: {event:?}")
                            }
                            _ => {}
                        }
                    }
                    if !errors.is_empty() {
                        let error = step.unwrap_err();
                        assert!(matches!(
                            error,
                            EngineError::Constraint(runtime_grammar::GrammarError::TooLarge(_))
                        ));
                        assert!(error.to_string().contains(reason), "{error}");
                        break;
                    }
                    step.unwrap();
                }
                assert_eq!(errors, [reason], "{rows:?}, in_place={in_place}");
            }
        }
    }
}
