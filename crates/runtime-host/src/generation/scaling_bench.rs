//! How the serving loop's host time per step grows with concurrency and with
//! the tokens each step emits, on CPU.
//!
//! Drives the real worker loop over an executor that returns device-sampled
//! rows. Each execute emulates a device step: it launches, waits on the device
//! for 2 ms, and reports that wait. Every request's events go through the
//! serving relay to a client task that keeps up, as `chopinfer serve` delivers
//! them. Requests take the served tuner's shape: a token-ID completion with
//! one top logprob, token IDs returned, EOS ignored and a minimum length. A
//! cell runs C decoding requests whose verify items carry K proposals, all
//! accepted, so a step emits C * (K + 1) tokens.
//!
//! The table's gap is the median host time from one executor return to the
//! next executor call: commit, event delivery, the loop's intake and
//! scheduling, everything the device would wait for. With
//! `CHOPINFER_STEP_TRACE=1` the engine also writes its `chopinfer.step.v1`
//! records to stderr, each run's after one `chopinfer.bench.cell` line
//! through the same queue, so the records split by cell:
//!
//! `CHOPINFER_STEP_TRACE=1 cargo test --release -p runtime-host --lib
//! serving_loop_scaling_bench -- --ignored --nocapture 2> trace.jsonl`
use super::stop_tests::declarations;
use super::*;
use runtime_engine::{
    DraftRow, DraftSource, EngineConfig, ModelBatchItem, ModelTransaction,
    NoopModelTransactionState, SampledRow, SpeculationConfig, SpeculationMode,
};
use runtime_kv::{KvGeometry, PreparedAppendBatch};
use runtime_sampling::TokenLogprob as SampleTokenLogprob;
use runtime_scheduler::{BatchPhase, SchedulerConfig};

const VOCABULARY: usize = 259;
const EOS: u32 = 1;
/// How long the emulated device runs each step.
const DEVICE_STEP: Duration = Duration::from_millis(2);
const WARMUP_STEPS: usize = 16;
const MEASURED_STEPS: usize = 128;
/// Runs per cell, each on a new worker.
const REPEATS: usize = 3;
const PROMPT: usize = 64;

/// Printable letters and spaces, never EOS.
fn token_at(id: RequestId, position: usize) -> u32 {
    let value = (id.get() as usize * 7 + position * 13) % 27;
    if value == 26 {
        u32::from(b' ') + 3
    } else {
        u32::from(b'a') + value as u32 + 3
    }
}

struct ScalingExecutor {
    concurrency: usize,
    /// Steps in which every request decodes, still to run.
    remaining: usize,
    /// Each such step's executor entry and return.
    marks: Arc<Mutex<Vec<(Instant, Instant)>>>,
    control: SyncSender<WorkerCommand>,
}

impl ModelExecutor for ScalingExecutor {
    type TransactionState<'a> = NoopModelTransactionState;

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        _: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, runtime_engine::ExecutorError> {
        let entered = Instant::now();
        crate::overlap::note_launch();
        crate::overlap::device_wait();
        if let Some(rest) = DEVICE_STEP.checked_sub(entered.elapsed()) {
            thread::sleep(rest);
        }
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
                    .map(|row| token_at(item.request_id(), start + row))
                    .collect();
                drafts.push(DraftRow::new(item.request_id(), proposed));
            }
            for row in 0..=count {
                let token = token_at(item.request_id(), start + row);
                let top = vec![SampleTokenLogprob::new(token, -0.5)];
                sampled.push(SampledRow::new(item.request_id(), token, -0.5, top));
            }
        }
        let decoding = items.len() == self.concurrency
            && items
                .iter()
                .all(|item| !matches!(item.phase(), BatchPhase::Prefill { .. }));
        if decoding && self.remaining > 0 {
            self.remaining -= 1;
            if self.remaining == 0 {
                assert!(self.control.try_send(WorkerCommand::Shutdown).is_ok());
            }
            self.marks.lock().unwrap().push((entered, Instant::now()));
        }
        Ok(ModelTransaction::no_op(Vec::new())
            .with_sampled(sampled)
            .with_drafts(drafts))
    }
}

/// The host gaps between executor calls over the measured all-decode steps.
fn run(concurrency: usize, drafts: usize) -> Vec<Duration> {
    runtime_engine::logging::line(format!(
        "{{\"schema\":\"chopinfer.bench.cell\",\"concurrency\":{concurrency},\"drafts\":{drafts}}}"
    ));
    let mut descriptor =
        FrontendDescriptorV1::from_component_bytes(declarations::KEY_VALUE.as_bytes()).unwrap();
    descriptor.vocabulary.size = VOCABULARY as u32;
    descriptor.vocabulary.eos_token_id = EOS;
    descriptor.vocabulary.stop_token_ids = vec![EOS];
    let policy = EngineModelPolicyV1::from_declaration(&descriptor);
    let steps = WARMUP_STEPS + MEASURED_STEPS;
    let maximum = (steps + 8) * (drafts + 1);
    let (control, receiver) = mpsc::sync_channel(concurrency + 8);
    let relays = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let tokenizer = Arc::new(super::stop_tests::tokenizer());
    let echoes = PromptEchoes::new(Arc::clone(&tokenizer));
    let mut submissions = Vec::new();
    for index in 0..concurrency {
        let id = RequestId::new(index as u64 + 1);
        let request = PreparedRequest {
            request_id: format!("scaling-{index}"),
            client_request_id: None,
            model: "fixture/alpha-1".into(),
            prompt: PreparedPrompt::TokenIds(vec![u32::from(b'q') + 3; PROMPT]),
            max_new_tokens: maximum,
            stop: Vec::new(),
            decoding: runtime_server::DecodingOptions {
                ignore_eos: true,
                ..Default::default()
            },
            sampling: SamplingParameters::new(0.0)
                .unwrap()
                .with_top_logprobs(1)
                .unwrap()
                .with_minimum_new_tokens(maximum)
                .unwrap(),
            seed: index as u64,
            logprobs: runtime_server::RequestedLogprobs {
                enabled: true,
                top: 1,
                echo: false,
                prompt_top: None,
                return_token_ids: true,
            },
            chat: None,
            constraint: None,
            cache_salt: None,
            cancellation: CancellationSignal::new(),
        };
        let (client, mut events) = tokio::sync::mpsc::channel(4096);
        let (sender, forward) = tokio::sync::mpsc::channel(4096);
        relays.spawn(forward_events(
            forward,
            EventSender::new(client),
            request.cancellation.clone(),
            echoes.clone(),
        ));
        relays.spawn(async move { while events.recv().await.is_some() {} });
        let (response, _admitted) = oneshot::channel();
        submissions.push(WorkerSubmission {
            observe_delivery: false,
            skip_reply_history: false,
            id,
            request: Box::new(request),
            constraint: None,
            prefix_chain: None,
            events: WorkerEvents(sender),
            response,
        });
    }
    control
        .try_send(WorkerCommand::Submit(submissions))
        .ok()
        .unwrap();
    let marks = Arc::new(Mutex::new(Vec::new()));
    let executor = ScalingExecutor {
        concurrency,
        remaining: steps,
        marks: Arc::clone(&marks),
        control: control.clone(),
    };
    let block = 64;
    let pages = concurrency * (PROMPT + maximum + block) / block + 64;
    let config = EngineConfig::new(
        SchedulerConfig::new(concurrency, concurrency * 2, 65536, 8192, 1).unwrap(),
        KvGeometry::new(block, pages).unwrap(),
        ModelVocabulary::new(VOCABULARY).unwrap(),
        PROMPT + maximum + 8,
        64,
    )
    .unwrap();
    let config = if drafts > 0 {
        config.with_speculation(
            SpeculationConfig::new(drafts, 8192, SpeculationMode::Fixed, DraftSource::Bundle)
                .unwrap(),
        )
    } else {
        config
    };
    let engine = Engine::new(config, executor).unwrap();
    let backend = EngineBackendConfig::default();
    let worker = thread::spawn(move || {
        let state = AtomicU8::new(READY);
        worker_loop(
            tokenizer,
            engine,
            receiver,
            backend,
            policy,
            &state,
            &WorkerProgress::new(),
        )
    });
    assert!(matches!(worker.join().unwrap(), WorkerExit::Shutdown));
    relays.shutdown_timeout(Duration::from_secs(1));
    let marks = std::mem::take(&mut *marks.lock().unwrap());
    assert_eq!(marks.len(), steps, "every request decoded to the end");
    marks
        .windows(2)
        .skip(WARMUP_STEPS)
        .map(|pair| pair[1].0 - pair[0].1)
        .collect()
}

#[test]
#[ignore = "CPU serving-loop scaling bench; run with --release --ignored --nocapture"]
fn serving_loop_scaling_bench() {
    assert!(!cfg!(debug_assertions), "run with --release");
    eprintln!("| C | K | Tokens/step | Gap median, us | Gap p95, us |");
    eprintln!("|---:|---:|---:|---:|---:|");
    for drafts in [0, 3, 4, 7] {
        for concurrency in [1, 8, 16, 32, 64, 128] {
            let mut gaps = (0..REPEATS)
                .flat_map(|_| run(concurrency, drafts))
                .collect::<Vec<_>>();
            gaps.sort();
            let micros = |at: usize| gaps[at].as_secs_f64() * 1e6;
            let (median, p95) = (micros(gaps.len() / 2), micros(gaps.len() * 95 / 100));
            let tokens = concurrency * (drafts + 1);
            eprintln!("| {concurrency} | {drafts} | {tokens} | {median:.0} | {p95:.0} |");
        }
    }
    runtime_engine::logging::flush();
}
