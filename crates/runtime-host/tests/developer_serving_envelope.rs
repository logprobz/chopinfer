#![forbid(unsafe_code)]

//! What a developer serving envelope resolves to, and what that then runs.
//!
//! The envelope is a statement in tokens: one exact prompt length, answered
//! with at most so many generated tokens. The bundle's declaration is in rows
//! of per-sequence model state. This drives both halves over the real engine,
//! scheduler and KV pool, with an executor that records the batches it was
//! asked for and returns fixed logits, so the row accounting is observed rather
//! than asserted about.
//!
//! Nothing here is a model and nothing here is a device. The manifest is a
//! fixture, the executor computes nothing, and no receipt is produced: what is
//! under test is which shapes the host admits and how many rows and pages they
//! consume.

use std::collections::BTreeMap;

use runtime_bundle::{
    BundleComponentV1, BundleGraphBucketV1, BundleKvLayoutV1, BundleManifestV1, BundleOpaqueId,
    BundleRankPlacementV1, BundleRelativePathV1, BundleResourceRequirementsV1, BundleStateLayoutV1,
    BundleStateLifetimeV1, BundleTargetV1, BundleTopologyV1,
};
use runtime_engine::{
    Engine, EngineEventKind, EngineRequest, ExecutorError, FinishReason, LogitsRow, ModelBatchItem,
    ModelExecutor, ModelTransaction, NoopModelTransactionState, StepStatus,
};
use runtime_host::{
    DeveloperServingEnvelopeV1, ServingOptionsV1, developer_serving_engine_config_v1,
    resolve_developer_serving_plan_v1,
};
use runtime_kv::PreparedAppendBatch;
use runtime_model_api::ModelVocabulary;
use runtime_sampling::SamplingParameters;
use runtime_scheduler::{BatchPhase, RequestId};

/// Rows of model state one sequence of the fixture bundle may hold.
const DECLARED_ROWS: u64 = 33;
/// Rows one declared KV page holds.
const PAGE_TOKENS: u32 = 32;
/// What the fixture's frontend declaration offers, which is far wider.
const DECLARED_TOTAL_TOKENS: usize = 200_000;
/// The prompt length the envelope admits, in token ids.
const PROMPT_TOKENS: usize = 32;
/// The most tokens one admitted request is answered with.
const GENERATED_TOKENS: usize = 2;
/// The two tokens the fixture executor answers every request with.
const SCRIPT: [u32; 2] = [7, 9];

fn id(value: &str) -> BundleOpaqueId {
    BundleOpaqueId::new(value).expect("the test identifier is valid")
}

/// One manifest whose KV layout declares these pages and these rows.
///
/// Nothing here is a model: the identifiers are invented, and the host reads
/// only what any manifest declares.
fn manifest(page_tokens: u32, declared_rows: u64) -> BundleManifestV1 {
    BundleManifestV1::new(
        id("fixture.envelope.v1"),
        BundleTargetV1::new(
            vec![id("sm_120f")],
            vec![BundleTopologyV1::new(id("tp1"), 1).expect("the test topology is valid")],
        )
        .expect("the test target is valid"),
        vec![
            BundleComponentV1::new(
                id("weights"),
                id("fixture.weights.v1"),
                id("fixture.bytes.v1"),
                BundleRelativePathV1::new("components/weights.bin")
                    .expect("the test path is relative"),
                16,
                "0".repeat(64),
            )
            .expect("the test component is valid"),
        ],
        BundleResourceRequirementsV1::new(1, Vec::new()).expect("the test requirements are valid"),
        BundleKvLayoutV1::new(
            id("fixture.kv.v1"),
            u64::from(page_tokens) * 64,
            64,
            page_tokens,
            64,
            1,
            declared_rows,
            BundleRankPlacementV1::Replicated,
            BundleStateLifetimeV1::Transaction,
        )
        .expect("the test KV layout is valid"),
        BundleStateLayoutV1::new(
            id("fixture.state.v1"),
            64,
            64,
            BundleStateLifetimeV1::Request,
        )
        .expect("the test state layout is valid"),
        vec![
            BundleGraphBucketV1::new(id("bucket.one"), vec![id("weights")], Vec::new())
                .expect("the test bucket is valid"),
        ],
        Vec::new(),
        Vec::new(),
        None,
    )
    .expect("the test manifest is valid")
}

/// One sequence, one page of scheduled tokens per step.
fn serving() -> ServingOptionsV1 {
    ServingOptionsV1 {
        maximum_context_tokens: None,
        maximum_sequences: 1,
        maximum_batch_tokens: Some(PAGE_TOKENS as usize),
        ..ServingOptionsV1::default()
    }
}

fn envelope(prompt_tokens: usize, generated_tokens: usize) -> DeveloperServingEnvelopeV1 {
    DeveloperServingEnvelopeV1::new(prompt_tokens, generated_tokens)
        .expect("the test envelope is constructible")
}

fn vocabulary() -> ModelVocabulary {
    ModelVocabulary::new(128).expect("the test vocabulary is valid")
}

/// What one scheduled batch asked of the executor, and of the KV pool.
#[derive(Clone, Debug, Eq, PartialEq)]
struct BatchTraceV1 {
    phase: BatchPhase,
    input_tokens: usize,
    appended_rows: usize,
    /// Pages this append had to allocate, which no earlier one held.
    newly_allocated: Vec<usize>,
    /// Every page the sequence holds after this append.
    block_table: Vec<usize>,
}

/// An executor that records the batches it ran and answers from a script.
///
/// It computes nothing and reaches no device. Two requests of the same shape
/// are answered with the same two tokens, so a repeated request is a repeated
/// answer rather than a coincidence.
#[derive(Debug, Default)]
struct TracingExecutorV1 {
    batches: Vec<BatchTraceV1>,
    emitted: BTreeMap<RequestId, usize>,
}

impl ModelExecutor for TracingExecutorV1 {
    type TransactionState<'a> = NoopModelTransactionState;

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        kv_plan: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        for (item, append) in items.iter().zip(kv_plan.appends()) {
            self.batches.push(BatchTraceV1 {
                phase: item.phase(),
                input_tokens: item.input_tokens().len(),
                appended_rows: append.additional_tokens(),
                newly_allocated: append
                    .newly_allocated()
                    .iter()
                    .map(|block| block.index())
                    .collect(),
                block_table: append
                    .block_table()
                    .iter()
                    .map(|block| block.index())
                    .collect(),
            });
        }
        let rows = items
            .iter()
            .filter(|item| item.expects_logits())
            .map(|item| {
                let emitted = self.emitted.entry(item.request_id()).or_default();
                let token = SCRIPT[*emitted % SCRIPT.len()];
                *emitted += 1;
                let mut logits = vec![-10.0; vocabulary().get()];
                logits[token as usize] = 10.0;
                LogitsRow::new(item.request_id(), logits)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ModelTransaction::no_op(rows))
    }
}

/// Runs one request of the admitted shape and returns the tokens it emitted.
fn run_request(
    engine: &mut Engine<TracingExecutorV1>,
    id: u64,
) -> (Vec<u32>, Option<FinishReason>) {
    let request = EngineRequest::new(
        RequestId::new(id),
        vec![1; PROMPT_TOKENS],
        GENERATED_TOKENS,
        SamplingParameters::new(0.0).expect("greedy sampling is valid"),
        id,
        Vec::new(),
        vocabulary(),
    )
    .expect("the admitted shape is a valid engine request");
    engine.submit(request).expect("the engine admits the shape");

    let mut tokens = Vec::new();
    let mut finished = None;
    // Prefill, one decode, and a retirement step, with room to spare. A shape
    // this small never needs more, and a bound is what keeps a stuck engine a
    // failing test rather than a hanging one.
    for _ in 0..8 {
        let status = engine.step().expect("the engine steps");
        for event in engine.drain_events() {
            match event.kind() {
                EngineEventKind::Token { token_id, .. } => tokens.push(*token_id),
                EngineEventKind::PromptLogprobs { .. } | EngineEventKind::PromptCached { .. } => {}
                EngineEventKind::Finished { reason } => finished = Some(*reason),
                EngineEventKind::Error { message } => panic!("the engine failed: {message}"),
            }
        }
        if finished.is_some() && status == StepStatus::Idle {
            break;
        }
    }
    (tokens, finished)
}

/// The budget is one token wider than the rows, and the run stays inside them.
///
/// This is the whole of the allowance: 32 prompt tokens answered with 2 is a
/// 34-token request, and running it writes 33 rows, over the two pages the
/// declared 33 rows already needed. The second decode step never happens,
/// because the second generated token is read out of the row the first one
/// wrote.
#[test]
fn the_admitted_shape_prefills_once_decodes_once_and_writes_the_declared_rows() {
    let manifest = manifest(PAGE_TOKENS, DECLARED_ROWS);
    let plan = resolve_developer_serving_plan_v1(
        &manifest,
        DECLARED_TOTAL_TOKENS,
        envelope(PROMPT_TOKENS, GENERATED_TOKENS),
    )
    .expect("the envelope is within what this bundle declares");
    assert_eq!(plan.total_tokens(), 34);
    assert_eq!(plan.consumed_rows(), 33);
    assert_eq!(plan.declared_rows(), 33);
    assert_eq!(plan.pages_per_sequence(), 2);

    let config = developer_serving_engine_config_v1(
        &manifest,
        vocabulary(),
        DECLARED_TOTAL_TOKENS,
        serving(),
        plan,
    )
    .expect("the resolved plan sizes an engine");
    let mut engine = Engine::new(config, TracingExecutorV1::default())
        .expect("the sized engine is constructible");
    assert_eq!(engine.maximum_sequence_tokens(), plan.total_tokens());

    let (tokens, finished) = run_request(&mut engine, 1);

    assert_eq!(tokens, SCRIPT);
    assert_eq!(finished, Some(FinishReason::Length));
    assert_eq!(
        engine.executor().batches,
        vec![
            BatchTraceV1 {
                phase: BatchPhase::Prefill {
                    tokens: PROMPT_TOKENS
                },
                input_tokens: PROMPT_TOKENS,
                appended_rows: PROMPT_TOKENS,
                newly_allocated: vec![0],
                block_table: vec![0],
            },
            BatchTraceV1 {
                phase: BatchPhase::Decode,
                input_tokens: 1,
                appended_rows: 1,
                newly_allocated: vec![1],
                block_table: vec![0, 1],
            },
        ]
    );
    // 32 rows prefilled and one decoded is 33: the rows the bundle declares,
    // over the pages it declares them in.
    let rows: usize = engine
        .executor()
        .batches
        .iter()
        .map(|batch| batch.appended_rows)
        .sum();
    assert_eq!(rows, plan.consumed_rows());
    assert_eq!(
        engine
            .executor()
            .batches
            .last()
            .expect("a decode ran")
            .block_table
            .len(),
        plan.pages_per_sequence()
    );

    // A second request of the same shape is answered the same way.
    let batches = engine.executor().batches.len();
    let (repeated, repeated_finish) = run_request(&mut engine, 2);
    assert_eq!(repeated, tokens);
    assert_eq!(repeated_finish, finished);
    assert_eq!(engine.executor().batches.len(), batches * 2);
}

/// A request the envelope does not admit never reaches the executor.
///
/// The engine holds the resolved budget and nothing beyond it, so a third
/// generated token is refused at submission rather than attempted against a
/// row of model state the bundle never declared.
#[test]
fn a_longer_request_than_the_budget_is_refused_before_any_batch_runs() {
    let manifest = manifest(PAGE_TOKENS, DECLARED_ROWS);
    let plan = resolve_developer_serving_plan_v1(
        &manifest,
        DECLARED_TOTAL_TOKENS,
        envelope(PROMPT_TOKENS, GENERATED_TOKENS),
    )
    .expect("the envelope is within what this bundle declares");
    let config = developer_serving_engine_config_v1(
        &manifest,
        vocabulary(),
        DECLARED_TOTAL_TOKENS,
        serving(),
        plan,
    )
    .expect("the resolved plan sizes an engine");
    let mut engine = Engine::new(config, TracingExecutorV1::default())
        .expect("the sized engine is constructible");

    for (prompt_tokens, generated_tokens) in [(PROMPT_TOKENS, 3), (33, 2), (35, 1)] {
        let request = EngineRequest::new(
            RequestId::new(1),
            vec![1; prompt_tokens],
            generated_tokens,
            SamplingParameters::new(0.0).expect("greedy sampling is valid"),
            1,
            Vec::new(),
            vocabulary(),
        )
        .expect("the request itself is well formed");
        assert!(
            engine.submit(request).is_err(),
            "{prompt_tokens} prompt token(s) and {generated_tokens} generated one(s) must be \
             refused by a budget of {}",
            plan.total_tokens()
        );
    }
    assert!(engine.executor().batches.is_empty());

    // The same lengths are refused a step earlier, when the envelope itself is
    // resolved against what the bundle declares.
    for (prompt_tokens, generated_tokens) in [(PROMPT_TOKENS, 3), (33, 2)] {
        assert!(
            resolve_developer_serving_plan_v1(
                &manifest,
                DECLARED_TOTAL_TOKENS,
                envelope(prompt_tokens, generated_tokens),
            )
            .is_err(),
            "{prompt_tokens} and {generated_tokens} write more rows than this bundle declares"
        );
    }
}

/// A plan is only about the bundle it was resolved against.
///
/// A plan is a public, copyable value, and nothing about the value says where
/// it came from. Sizing an engine resolves it again against the manifest in
/// hand, so a plan that a bundle declaring 1024 rows produced cannot size an
/// engine for one declaring 33: it would ask for a budget the model cannot
/// hold and reserve pages nothing declared.
#[test]
fn a_plan_from_another_bundle_never_sizes_this_one() {
    let small = manifest(PAGE_TOKENS, DECLARED_ROWS);
    let large = manifest(PAGE_TOKENS, 1024);

    // A long envelope is legitimate over the bundle that declares 1024 rows:
    // 1024 tokens over 1023 rows, in the 32 pages those rows already needed.
    let large_plan =
        resolve_developer_serving_plan_v1(&large, DECLARED_TOTAL_TOKENS, envelope(1000, 24))
            .expect("a long envelope suits a bundle that declares 1024 rows");
    assert_eq!(large_plan.total_tokens(), 1024);
    assert_eq!(large_plan.declared_rows(), 1024);
    assert_eq!(large_plan.pages_per_sequence(), 32);

    let refusal = developer_serving_engine_config_v1(
        &small,
        vocabulary(),
        DECLARED_TOTAL_TOKENS,
        serving(),
        large_plan,
    )
    .expect_err("a plan resolved against 1024 rows must not size a 33-row bundle")
    .to_string();
    assert!(refusal.contains("writes 1023 row(s)"), "{refusal}");
    assert!(refusal.contains("declares 33"), "{refusal}");

    // The same holds the other way, where the mismatch is the plan's own
    // numbers rather than a refusal of its envelope.
    let small_plan = resolve_developer_serving_plan_v1(
        &small,
        DECLARED_TOTAL_TOKENS,
        envelope(PROMPT_TOKENS, GENERATED_TOKENS),
    )
    .expect("the measured envelope suits the bundle it was measured on");
    let refusal = developer_serving_engine_config_v1(
        &large,
        vocabulary(),
        DECLARED_TOTAL_TOKENS,
        serving(),
        small_plan,
    )
    .expect_err("a plan resolved against 33 rows must not size a 1024-row bundle")
    .to_string();
    assert!(
        refusal.contains("reserve 2 page(s) per sequence"),
        "{refusal}"
    );

    // A bundle that declares the same rows in different pages is a different
    // bundle for this purpose, and its plan says so.
    let repaged = manifest(16, DECLARED_ROWS);
    let repaged_plan = resolve_developer_serving_plan_v1(
        &repaged,
        DECLARED_TOTAL_TOKENS,
        envelope(PROMPT_TOKENS, GENERATED_TOKENS),
    )
    .expect("the measured envelope fits three pages of 16 rows");
    assert_eq!(repaged_plan.page_tokens(), 16);
    assert_eq!(repaged_plan.pages_per_sequence(), 3);
    let refusal = developer_serving_engine_config_v1(
        &small,
        vocabulary(),
        DECLARED_TOTAL_TOKENS,
        serving(),
        repaged_plan,
    )
    .expect_err("a plan resolved over pages of 16 must not size a bundle paged in 32")
    .to_string();
    assert!(refusal.contains("in pages of 16"), "{refusal}");
    assert!(refusal.contains("in pages of 32"), "{refusal}");

    // The plan the bundle in hand resolves still sizes it.
    assert!(
        developer_serving_engine_config_v1(
            &small,
            vocabulary(),
            DECLARED_TOTAL_TOKENS,
            serving(),
            small_plan,
        )
        .is_ok()
    );
}

/// The context a deployment serves bounds the plan that sizes its engine.
///
/// The served context is checked again here rather than taken from the plan,
/// because a plan carries the declaration it was resolved against and not the
/// one this deployment ended up serving.
#[test]
fn a_plan_longer_than_the_served_context_never_sizes_an_engine() {
    let manifest = manifest(PAGE_TOKENS, DECLARED_ROWS);
    let plan = resolve_developer_serving_plan_v1(
        &manifest,
        DECLARED_TOTAL_TOKENS,
        envelope(PROMPT_TOKENS, GENERATED_TOKENS),
    )
    .expect("the measured envelope suits this bundle");

    let refusal = developer_serving_engine_config_v1(
        &manifest,
        vocabulary(),
        plan.total_tokens() - 1,
        serving(),
        plan,
    )
    .expect_err("a 34-token plan must not size an engine serving 33 tokens")
    .to_string();
    assert!(refusal.contains("charges 34 token(s)"), "{refusal}");
    assert!(refusal.contains("offers 33"), "{refusal}");

    // Serving exactly the budget is what the envelope path itself configures.
    assert!(
        developer_serving_engine_config_v1(
            &manifest,
            vocabulary(),
            plan.total_tokens(),
            serving(),
            plan,
        )
        .is_ok()
    );
}

#[test]
fn bounded_envelope_admits_only_positive_counts_within_all_limits() {
    let envelope = DeveloperServingEnvelopeV1::bounded(255, 64, 256).unwrap();
    assert!(envelope.is_bounded());
    assert_eq!(envelope.total_tokens(), 256);
    for (prompt, output) in [(1, 1), (32, 64), (192, 64), (255, 1)] {
        assert!(envelope.admits(prompt, output));
    }
    for (prompt, output) in [(0, 1), (1, 0), (1, 65), (193, 64), (255, 2), (256, 1)] {
        assert!(!envelope.admits(prompt, output));
    }
    for limits in [
        (0, 64, 256),
        (255, 0, 256),
        (255, 64, 0),
        (256, 64, 256),
        (1, 256, 256),
    ] {
        assert!(DeveloperServingEnvelopeV1::bounded(limits.0, limits.1, limits.2).is_err());
    }
}

#[test]
fn bounded_requests_run_variable_budgets_in_the_declared_pages() {
    let declared = manifest(32, 256);
    let envelope = DeveloperServingEnvelopeV1::bounded(255, 64, 256).unwrap();
    let plan = resolve_developer_serving_plan_v1(&declared, 256, envelope).unwrap();
    assert_eq!(plan.pages_per_sequence(), 8);
    assert_eq!(plan.consumed_rows(), 255);
    for (prompt, output) in [(1, 1), (32, 64), (192, 64), (255, 1)] {
        let config =
            developer_serving_engine_config_v1(&declared, vocabulary(), 256, serving(), plan)
                .unwrap();
        let mut engine = Engine::new(config, TracingExecutorV1::default()).unwrap();
        assert_eq!(engine.maximum_sequence_tokens(), 256);
        assert_eq!(engine.kv_stats().free_blocks(), 8);
        let request = EngineRequest::new(
            RequestId::new(1),
            vec![1; prompt],
            output,
            SamplingParameters::new(0.0).unwrap(),
            1,
            Vec::new(),
            vocabulary(),
        )
        .unwrap();
        engine.submit(request).unwrap();
        let mut emitted = 0;
        let mut finished = false;
        for _ in 0..100 {
            let status = engine.step().unwrap();
            for event in engine.drain_events() {
                match event.kind() {
                    EngineEventKind::Token { .. } => emitted += 1,
                    EngineEventKind::PromptLogprobs { .. }
                    | EngineEventKind::PromptCached { .. } => {}
                    EngineEventKind::Finished { reason } => {
                        assert_eq!(*reason, FinishReason::Length);
                        finished = true;
                    }
                    EngineEventKind::Error { message } => panic!("{message}"),
                }
            }
            if finished && status == StepStatus::Idle {
                break;
            }
        }
        assert!(finished);
        assert_eq!(emitted, output);
        let batches = &engine.executor().batches;
        assert_eq!(
            batches
                .iter()
                .map(|batch| batch.appended_rows)
                .sum::<usize>(),
            prompt + output - 1
        );
        assert_eq!(
            batches.last().unwrap().block_table.len(),
            (prompt + output - 1).div_ceil(32)
        );
        assert!(
            batches
                .iter()
                .flat_map(|batch| &batch.block_table)
                .all(|page| *page < 8)
        );
    }
    for other in [manifest(16, 256), manifest(32, 512)] {
        assert!(
            developer_serving_engine_config_v1(&other, vocabulary(), 256, serving(), plan).is_err()
        );
    }
    assert!(
        developer_serving_engine_config_v1(&declared, vocabulary(), 255, serving(), plan).is_err()
    );
}

#[test]
fn bounded_capacity_reserves_declared_pages_even_for_a_smaller_budget() {
    let declared = manifest(32, 256);
    let envelope = DeveloperServingEnvelopeV1::bounded(63, 32, 64).unwrap();
    let plan = resolve_developer_serving_plan_v1(&declared, 256, envelope).unwrap();
    assert_eq!(plan.pages_per_sequence(), 8);
    let mut options = serving();
    options.maximum_sequences = 4;
    let config =
        developer_serving_engine_config_v1(&declared, vocabulary(), 256, options, plan).unwrap();
    let engine = Engine::new(config, TracingExecutorV1::default()).unwrap();
    assert_eq!(engine.maximum_sequence_tokens(), 64);
    assert_eq!(engine.kv_stats().free_blocks(), 32);
    let too_wide = DeveloperServingEnvelopeV1::bounded(256, 1, 257).unwrap();
    assert!(resolve_developer_serving_plan_v1(&declared, 257, too_wide).is_err());
}
