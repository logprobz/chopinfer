//! Host cost of one serving step on the engine thread, on CPU.
//!
//! Runs the engine over an executor that returns device-sampled rows, and
//! delivers every event the way the worker loop does, to per-request streams
//! a tokio runtime drains. Each execute emulates a device step: it launches,
//! waits on the device for 2 ms, and reports that wait. The critical path is
//! everything else the engine thread does per step, which on a GPU is time
//! the device sits idle between steps.
//!
//! `cargo test --release -p runtime-host --lib host_step_cost_bench --
//! --ignored --nocapture`.
use super::stop_tests::declarations;
use super::*;
use runtime_engine::{
    DraftRow, DraftSource, EngineConfig, ModelBatchItem, ModelTransaction,
    NoopModelTransactionState, SampledRow, SpeculationConfig, SpeculationMode,
};
use runtime_kv::{KvGeometry, PreparedAppendBatch};
use runtime_sampling::TokenLogprob as SampleTokenLogprob;
use runtime_scheduler::SchedulerConfig;

const VOCABULARY: usize = 259;
/// How long the emulated device runs each step.
const DEVICE_STEP: Duration = Duration::from_millis(2);
const WARMUP_STEPS: usize = 8;
const MEASURED_STEPS: usize = 64;

struct BenchExecutor {
    /// The draft row every verify item gets wrong, so each accepts that many.
    wrong: usize,
    /// Wall time of the emulated device waits, deferred work included.
    waited: Rc<Cell<Duration>>,
}

/// Printable letters and spaces, never EOS.
fn token_at(id: RequestId, position: usize) -> u32 {
    let value = (id.get() as usize * 7 + position * 13) % 27;
    if value == 26 {
        u32::from(b' ') + 3
    } else {
        u32::from(b'a') + value as u32 + 3
    }
}

impl ModelExecutor for BenchExecutor {
    type TransactionState<'a> = NoopModelTransactionState;

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        _: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, runtime_engine::ExecutorError> {
        let waiting = Instant::now();
        crate::overlap::note_launch();
        crate::overlap::device_wait();
        if let Some(rest) = DEVICE_STEP.checked_sub(waiting.elapsed()) {
            thread::sleep(rest);
        }
        self.waited.set(self.waited.get() + waiting.elapsed());
        let mut sampled = Vec::new();
        let mut drafts = Vec::new();
        for item in items {
            let Some(work) = item.sampling() else {
                continue;
            };
            let start = work.emitted_tokens;
            let count = item.draft().map_or(0, |draft| draft.tokens());
            if item.draft().is_some() {
                let proposed = (0..count)
                    .map(|row| {
                        token_at(item.request_id(), start + row) ^ u32::from(row == self.wrong)
                    })
                    .collect();
                drafts.push(DraftRow::new(item.request_id(), proposed));
            }
            for row in 0..=count {
                let token = token_at(item.request_id(), start + row);
                let top = vec![SampleTokenLogprob::new(token, -0.5)];
                sampled.push(SampledRow::new(item.request_id(), token, -0.5, top));
            }
        }
        Ok(ModelTransaction::no_op(Vec::new())
            .with_sampled(sampled)
            .with_drafts(drafts))
    }
}

fn median(mut values: Vec<Duration>) -> f64 {
    values.sort();
    values[values.len() / 2].as_secs_f64() * 1e6
}

/// Median host microseconds per step on the critical path, of it the engine
/// step and the event handling, and the tokens each step emitted.
fn run(concurrency: usize, prompt: usize, draft: usize, overlap: bool) -> [f64; 4] {
    let tokenizer = super::stop_tests::tokenizer();
    // Early requests decode while later ones still prefill, 65,536 tokens
    // a step.
    let prefill_steps = (concurrency * prompt).div_ceil(65536) + 2;
    let maximum = (prefill_steps + WARMUP_STEPS + MEASURED_STEPS + 2) * (draft + 1) + 16;
    let block = 64;
    let pages = concurrency * (prompt + maximum + block) / block + 64;
    let config = EngineConfig::new(
        SchedulerConfig::new(concurrency, concurrency * 2, 65536, 8192, 1).unwrap(),
        KvGeometry::new(block, pages).unwrap(),
        ModelVocabulary::new(VOCABULARY).unwrap(),
        prompt + maximum + 8,
        64,
    )
    .unwrap();
    let config = if draft > 0 {
        config.with_speculation(
            SpeculationConfig::new(draft, 8192, SpeculationMode::Fixed, DraftSource::Bundle)
                .unwrap(),
        )
    } else {
        config
    };
    let waited = Rc::new(Cell::new(Duration::ZERO));
    let executor = BenchExecutor {
        wrong: draft.saturating_sub(2),
        waited: Rc::clone(&waited),
    };
    let mut engine = Engine::new(config, executor).unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let descriptor = Arc::new(
        FrontendDescriptorV1::from_component_bytes(declarations::KEY_VALUE.as_bytes()).unwrap(),
    );
    let mut active = BTreeMap::new();
    for index in 0..concurrency {
        let id = RequestId::new(index as u64 + 1);
        let sampling = SamplingParameters::new(0.0)
            .unwrap()
            .with_top_logprobs(1)
            .unwrap();
        let prompt_tokens = vec![u32::from(b'q') + 3; prompt];
        let vocabulary = ModelVocabulary::new(VOCABULARY).unwrap();
        let request =
            EngineRequest::new(id, prompt_tokens, maximum, sampling, 0, vec![1], vocabulary)
                .unwrap();
        engine.submit(request).unwrap();
        let mut decoder = CompletionDecoderV1::new(
            Arc::clone(&descriptor),
            PromptModeV1::Chat,
            DecoderLimitsV1::declared(&descriptor),
        );
        // A raw completion parses through the declared frontend, as admission
        // opens it.
        decoder.set_parser_source(
            runtime_frontend::BundleParserSourceV1::from_declaration(&descriptor).unwrap(),
        );
        decoder.set_decoding_options(Default::default(), true);
        let (sender, mut receiver) = tokio::sync::mpsc::channel(65536);
        runtime.spawn(async move { while receiver.recv().await.is_some() {} });
        let cancellation = CancellationSignal::new();
        active.insert(
            id,
            ActiveGeneration {
                events: Outbox::new(EventSender::new(sender), cancellation.clone(), overlap),
                cancellation,
                decoder,
                prompt_tokens: prompt as u64,
                cached_tokens: None,
                completion_tokens: 0,
                accounting: TokenAccounting::default(),
                logprob_context: [None; 4],
                prompt_logprob_context: [None; 4],
                logprobs_enabled: true,
                return_token_ids: false,
                admission: None,
                queued_since: Instant::now(),
                group_admitted: Rc::new(Cell::new(false)),
            },
        );
    }
    // Every request prefills before the measured decode steps.
    while active
        .values()
        .any(|generation: &ActiveGeneration| generation.completion_tokens == 0)
    {
        engine.step().unwrap();
        crate::overlap::run_pending();
        for event in engine.drain_events() {
            assert!(handle_engine_event(
                &tokenizer,
                &mut engine,
                &mut active,
                event
            ));
        }
    }
    let (mut critical, mut stepped, mut handled) = (Vec::new(), Vec::new(), Vec::new());
    let mut tokens = 0;
    for step in 0..WARMUP_STEPS + MEASURED_STEPS {
        let started = Instant::now();
        let waited_before = waited.get();
        engine.step().unwrap();
        crate::overlap::run_pending();
        let step_done = Instant::now();
        let events = engine.drain_events();
        let count = events
            .iter()
            .filter(|event| matches!(event.kind(), EngineEventKind::Token { .. }))
            .count();
        for event in events {
            assert!(handle_engine_event(
                &tokenizer,
                &mut engine,
                &mut active,
                event
            ));
        }
        let events_done = Instant::now();
        if step >= WARMUP_STEPS {
            let device = waited.get() - waited_before;
            critical.push(events_done - started - device);
            stepped.push(step_done - started - device);
            handled.push(events_done - step_done);
            tokens += count;
        }
    }
    assert_eq!(active.len(), concurrency, "no request finished early");
    drop(active);
    crate::overlap::run_pending();
    runtime.shutdown_timeout(Duration::from_secs(1));
    [
        median(critical),
        median(stepped),
        median(handled),
        tokens as f64 / MEASURED_STEPS as f64,
    ]
}

#[test]
#[ignore = "CPU host-cost bench; run with --release --ignored --nocapture"]
fn host_step_cost_bench() {
    eprintln!(
        "| C | Prompt | K | Tokens/step | Overlap | Critical, us | Engine step, us | Events, us |"
    );
    eprintln!("|---:|---:|---:|---:|---|---:|---:|---:|");
    for (concurrency, prompt, draft) in [
        (1, 1024, 0),
        (16, 1024, 0),
        (32, 1024, 0),
        (32, 8192, 0),
        (128, 1024, 0),
        (256, 1024, 0),
        (256, 8192, 0),
        (16, 1024, 7),
        (32, 1024, 7),
        (32, 8192, 7),
        (128, 1024, 3),
    ] {
        for overlap in [false, true] {
            let [critical, stepped, handled, tokens] = run(concurrency, prompt, draft, overlap);
            let overlap = if overlap { "on" } else { "off" };
            eprintln!(
                "| {concurrency} | {prompt} | {draft} | {tokens:.0} | {overlap} | {critical:.0} | {stepped:.0} | {handled:.0} |"
            );
        }
    }
}
