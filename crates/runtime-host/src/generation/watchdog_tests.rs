use std::sync::atomic::AtomicBool;

use super::stop_tests::{Stall, declarations, stalling_engine, tokenizer};
use super::*;
use runtime_engine::{
    EngineConfig, LogitsRow, ModelBatchItem, ModelTransaction, NoopModelTransactionState,
};
use runtime_kv::{KvGeometry, PreparedAppendBatch};
use runtime_scheduler::{AdmissionPolicy, OptimisticAdmission, SchedulerConfig};

const X: u32 = b'x' as u32 + 3;
const CHILD: &str = "CHOPINFER_HOST_WATCHDOG_CHILD";

fn policy() -> EngineModelPolicyV1 {
    let mut descriptor =
        FrontendDescriptorV1::from_component_bytes(declarations::KEY_VALUE.as_bytes()).unwrap();
    descriptor.vocabulary.size = 259;
    descriptor.vocabulary.eos_token_id = 1;
    descriptor.vocabulary.stop_token_ids = vec![1];
    EngineModelPolicyV1::from_declaration(&descriptor)
}

fn request(prompt: Vec<u32>, max_new_tokens: usize) -> PreparedRequest {
    PreparedRequest {
        request_id: "watched".into(),
        client_request_id: None,
        model: "fixture/alpha-1".into(),
        prompt: PreparedPrompt::TokenIds(prompt),
        max_new_tokens,
        stop: vec![],
        decoding: Default::default(),
        sampling: SamplingParameters::new(0.0).unwrap(),
        seed: 0,
        logprobs: runtime_server::RequestedLogprobs {
            enabled: false,
            top: 0,
            echo: false,
            prompt_top: None,
            return_token_ids: false,
        },
        chat: None,
        constraint: None,
        cache_salt: None,
        cancellation: CancellationSignal::new(),
    }
}

/// A backend whose model runs `stall` on each batch, watched with these
/// allowances and no process exit.
fn watched(stall: Stall, watchdog: WorkerWatchdogV1) -> EngineGenerationBackend {
    EngineGenerationBackend::new(
        tokenizer(),
        stalling_engine(vec![X; 16], stall),
        EngineBackendConfig::new(4)
            .unwrap()
            .with_watchdog(Some(watchdog)),
        policy(),
    )
    .unwrap()
}

fn prompt_items(items: &[ModelBatchItem]) -> bool {
    items.iter().any(|item| item.input_tokens().len() > 1)
}

/// Blocks the calling thread until `release` is set.
fn hang(release: &AtomicBool) {
    while !release.load(Ordering::Acquire) {
        thread::sleep(Duration::from_millis(5));
    }
}

/// Prompt steps that run longer than the decode allowance but within the
/// prompt allowance leave the worker ready, and the request completes.
#[test]
fn long_prompt_steps_within_their_allowance_keep_the_worker_ready() {
    let backend = watched(
        Arc::new(|items: &[ModelBatchItem]| {
            if prompt_items(items) {
                thread::sleep(Duration::from_millis(400));
            }
        }),
        WorkerWatchdogV1::serving()
            .with_allowances(
                Duration::from_millis(100),
                Duration::from_secs(5),
                Duration::from_secs(5),
            )
            .with_exit_grace(None),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let (events, mut receiver) = tokio::sync::mpsc::channel(128);
        backend
            .submit(request(vec![X; 8], 4), EventSender::new(events))
            .await
            .unwrap();
        let mut finished = false;
        while let Some(event) = receiver.recv().await {
            assert_eq!(backend.readiness(), BackendReadiness::Ready);
            finished |= matches!(event, GenerationEvent::Finished { .. });
        }
        assert!(finished);
    });
    assert_eq!(backend.readiness(), BackendReadiness::Ready);
    assert!(!backend.progress.is_stalled());
    backend.shutdown();
}

/// A decode step that never returns fails readiness and admission within
/// the decode allowance, and shutdown leaves the stuck worker unjoined.
#[test]
fn a_stuck_decode_step_fails_readiness_and_is_not_joined() {
    let release = Arc::new(AtomicBool::new(false));
    let held = Arc::clone(&release);
    let backend = watched(
        Arc::new(move |items: &[ModelBatchItem]| {
            if !prompt_items(items) {
                hang(&held);
            }
        }),
        WorkerWatchdogV1::serving()
            .with_allowances(
                Duration::from_millis(200),
                Duration::from_secs(30),
                Duration::from_secs(30),
            )
            .with_exit_grace(None),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (events, _receiver) = tokio::sync::mpsc::channel(128);
    runtime
        .block_on(async {
            backend
                .submit(request(vec![X; 8], 4), EventSender::new(events))
                .await
        })
        .unwrap();
    let started = Instant::now();
    while backend.readiness() == BackendReadiness::Ready {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "a stuck step left the worker ready"
        );
        thread::sleep(Duration::from_millis(5));
    }
    assert!(matches!(
        backend.readiness(),
        BackendReadiness::Terminated(_)
    ));
    assert!(backend.progress.is_stalled());
    let (events, _receiver) = tokio::sync::mpsc::channel(128);
    assert_eq!(
        runtime.block_on(async {
            backend
                .submit(request(vec![X; 8], 4), EventSender::new(events))
                .await
        }),
        Err(AdmissionError::WorkerUnavailable)
    );
    let shutdown = Instant::now();
    backend.shutdown();
    assert!(shutdown.elapsed() < Duration::from_secs(2));
    assert!(matches!(
        backend.readiness(),
        BackendReadiness::Terminated(_)
    ));
    // Let the detached worker leave its step; it fails what it held.
    release.store(true, Ordering::Release);
}

/// Runs `scenario` in a child copy of this test binary with `stderr`, and
/// returns its exit code. A child still running after a minute is killed and
/// fails the test.
fn child(scenario: &str, test: &str, stderr: std::process::Stdio) -> Option<i32> {
    let started = Instant::now();
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture"])
        .env(CHILD, scenario)
        .stdout(std::process::Stdio::null())
        .stderr(stderr)
        .spawn()
        .unwrap();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status.code();
        }
        if started.elapsed() > Duration::from_secs(60) {
            child.kill().unwrap();
            let _ = child.wait();
            panic!("{scenario}: the child process did not exit");
        }
        thread::sleep(Duration::from_millis(20));
    }
}

/// In a child: a backend whose executor never returns, which the watchdog
/// fails after 200 ms and exits 100 ms later.
fn hung_backend() {
    let backend = watched(
        Arc::new(|_: &[ModelBatchItem]| hang(&AtomicBool::new(false))),
        WorkerWatchdogV1::serving()
            .with_allowances(
                Duration::from_millis(200),
                Duration::from_millis(200),
                Duration::from_millis(200),
            )
            .with_exit_grace(Some(Duration::from_millis(100))),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (events, _receiver) = tokio::sync::mpsc::channel(128);
    let _ = runtime.block_on(async {
        backend
            .submit(request(vec![X; 8], 4), EventSender::new(events))
            .await
    });
    // The watchdog exits the process long before this.
    thread::sleep(Duration::from_secs(30));
    std::process::exit(3);
}

/// A process whose executor hangs exits with status 1 so a supervisor
/// restarts it, also when stderr fails (`/dev/full`) or blocks (a full pipe
/// nobody reads, with another thread stuck inside a stderr write).
#[test]
fn a_hung_executor_exits_the_process_whatever_stderr_does() {
    const TEST: &str =
        "generation::watchdog_tests::a_hung_executor_exits_the_process_whatever_stderr_does";
    match std::env::var(CHILD).as_deref() {
        Ok("blocked") => {
            // Fill the unread pipe; this thread then blocks in the write
            // while holding the standard library's stderr lock.
            thread::spawn(|| {
                let _ = std::io::Write::write_all(&mut std::io::stderr(), &vec![b'x'; 1 << 20]);
            });
            thread::sleep(Duration::from_millis(200));
            hung_backend();
        }
        Ok(_) => hung_backend(),
        Err(_) => {
            assert_eq!(child("plain", TEST, std::process::Stdio::null()), Some(1));
            let full = std::fs::OpenOptions::new()
                .write(true)
                .open("/dev/full")
                .unwrap();
            assert_eq!(child("full", TEST, full.into()), Some(1));
            // The parent keeps the pipe open and never reads it.
            assert_eq!(
                child("blocked", TEST, std::process::Stdio::piped()),
                Some(1)
            );
        }
    }
}

/// A watchdog that panics while failing the worker still exits the process.
#[test]
fn a_panicking_watchdog_still_exits_the_process() {
    const TEST: &str = "generation::watchdog_tests::a_panicking_watchdog_still_exits_the_process";
    if std::env::var_os(CHILD).is_none() {
        assert_eq!(child("panic", TEST, std::process::Stdio::null()), Some(1));
        return;
    }
    let progress = WorkerProgress::new();
    progress.mark(WorkerPhase::Decode);
    crate::watchdog::spawn(
        WorkerWatchdogV1::serving()
            .with_allowances(
                Duration::from_millis(50),
                Duration::from_millis(50),
                Duration::from_millis(50),
            )
            .with_exit_grace(Some(Duration::from_millis(100))),
        progress,
        || panic!("injected watchdog failure"),
    )
    .unwrap();
    thread::sleep(Duration::from_secs(30));
    std::process::exit(3);
}

/// An executor that prefills slowly on its second execution, which a forced
/// preemption after the first token makes a replay of the prompt.
struct SlowReplay {
    executions: usize,
    replay: Duration,
}

impl ModelExecutor for SlowReplay {
    type TransactionState<'a> = NoopModelTransactionState;

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        _: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, runtime_engine::ExecutorError> {
        if self.executions == 1 {
            assert!(
                prompt_items(items),
                "the second execution replays the prompt"
            );
            thread::sleep(self.replay);
        }
        self.executions += 1;
        let rows = items
            .iter()
            .filter(|item| item.sampling().is_some())
            .map(|item| {
                let mut logits = vec![-100.0; 259];
                logits[X as usize] = 10.0;
                LogitsRow::new(item.request_id(), logits).unwrap()
            })
            .collect();
        Ok(ModelTransaction::no_op(rows))
    }
}

/// A request preempted after its first token replays its prompt: that step
/// gets the prompt allowance, though the request has already emitted.
#[test]
fn a_replayed_prompt_after_preemption_gets_the_prompt_allowance() {
    let admission = AdmissionPolicy::Optimistic(
        OptimisticAdmission::new(0, 1)
            .unwrap()
            .with_forced_preemption(1)
            .unwrap(),
    );
    let engine = Engine::new(
        EngineConfig::new(
            SchedulerConfig::new(1, 32, 32, 8, 1)
                .unwrap()
                .with_admission(admission),
            KvGeometry::new(4, 64).unwrap(),
            ModelVocabulary::new(259).unwrap(),
            128,
            64,
        )
        .unwrap(),
        SlowReplay {
            executions: 0,
            replay: Duration::from_millis(400),
        },
    )
    .unwrap();
    let backend = EngineGenerationBackend::new(
        tokenizer(),
        engine,
        EngineBackendConfig::new(4).unwrap().with_watchdog(Some(
            WorkerWatchdogV1::serving()
                .with_allowances(
                    Duration::from_millis(100),
                    Duration::from_secs(5),
                    Duration::from_secs(5),
                )
                .with_exit_grace(None),
        )),
        policy(),
    )
    .unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let (events, mut receiver) = tokio::sync::mpsc::channel(128);
        backend
            .submit(request(vec![X; 2], 4), EventSender::new(events))
            .await
            .unwrap();
        let mut finished = false;
        while let Some(event) = receiver.recv().await {
            finished |= matches!(event, GenerationEvent::Finished { .. });
        }
        assert!(finished);
    });
    assert_eq!(backend.readiness(), BackendReadiness::Ready);
    assert!(!backend.progress.is_stalled());
    backend.shutdown();
}

/// Building the engine is watched: a factory that does not return fails
/// startup within its allowance, without joining the stuck thread.
#[test]
fn a_stuck_engine_build_fails_startup() {
    let release = Arc::new(AtomicBool::new(false));
    let held = Arc::clone(&release);
    let started = Instant::now();
    let result = EngineGenerationBackend::try_new_with_factory::<SlowReplay, _>(
        tokenizer(),
        move || {
            hang(&held);
            Err("released".to_owned())
        },
        EngineBackendConfig::new(4).unwrap().with_watchdog(Some(
            WorkerWatchdogV1::serving()
                .with_lifecycle_allowance(Duration::from_millis(200))
                .with_exit_grace(None),
        )),
        policy(),
    );
    assert_eq!(result.err(), Some(EngineBackendStartupError::WorkerStalled));
    assert!(started.elapsed() < Duration::from_secs(10));
    release.store(true, Ordering::Release);
}

/// An executor whose release blocks until `release` is set.
struct HoldOnDrop(Arc<AtomicBool>);

impl ModelExecutor for HoldOnDrop {
    type TransactionState<'a> = NoopModelTransactionState;

    fn execute<'a>(
        &'a mut self,
        _: &[ModelBatchItem],
        _: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, runtime_engine::ExecutorError> {
        Ok(ModelTransaction::no_op(Vec::new()))
    }
}

impl Drop for HoldOnDrop {
    fn drop(&mut self) {
        hang(&self.0);
    }
}

/// Tearing down an idle worker is watched: an executor release that blocks
/// fails the worker within its allowance, and shutdown returns without
/// joining it.
#[test]
fn a_stuck_teardown_of_an_idle_worker_is_not_joined() {
    let release = Arc::new(AtomicBool::new(false));
    let engine = Engine::new(
        EngineConfig::new(
            SchedulerConfig::new(1, 32, 32, 4, 1).unwrap(),
            KvGeometry::new(4, 64).unwrap(),
            ModelVocabulary::new(259).unwrap(),
            128,
            64,
        )
        .unwrap(),
        HoldOnDrop(Arc::clone(&release)),
    )
    .unwrap();
    let backend = EngineGenerationBackend::new(
        tokenizer(),
        engine,
        EngineBackendConfig::new(4).unwrap().with_watchdog(Some(
            WorkerWatchdogV1::serving()
                .with_lifecycle_allowance(Duration::from_millis(200))
                .with_exit_grace(None),
        )),
        policy(),
    )
    .unwrap();
    // Idle well past the teardown allowance first: waiting is not work.
    thread::sleep(Duration::from_millis(400));
    assert_eq!(backend.readiness(), BackendReadiness::Ready);
    let started = Instant::now();
    backend.shutdown();
    assert!(started.elapsed() < Duration::from_secs(10));
    assert!(backend.progress.is_stalled());
    assert!(matches!(
        backend.readiness(),
        BackendReadiness::Terminated(_)
    ));
    release.store(true, Ordering::Release);
}
