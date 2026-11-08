use super::*;
use std::time::{Duration, Instant};

fn wait_for_clock_tick() {
    let start = Instant::now();
    while start.elapsed() < Duration::from_micros(10) {
        std::hint::spin_loop();
    }
}

struct TimedCommit;

impl ModelTransactionState for TimedCommit {
    fn commit(&mut self) -> Result<(), ExecutorError> {
        assert!(profile::snapshot().totals[profile::StepPhase::SchedulerCommit as usize] > 0);
        assert_eq!(
            profile::snapshot().totals[profile::StepPhase::ModelCommit as usize],
            0
        );
        wait_for_clock_tick();
        Ok(())
    }

    fn rollback(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

#[test]
fn commit_trace_separates_scheduler_and_model_scopes() {
    let _totals = profile::TraceTotals::begin();
    commit_scheduler_then_model(ModelTransaction::new(Vec::new(), TimedCommit), None, || {
        wait_for_clock_tick();
        Ok(())
    })
    .unwrap()
    .1
    .unwrap();
    let timing = compatibility::HostClock::default().executed();
    assert!(timing.scheduler_commit > 0);
    assert!(timing.model_commit > 0);
    assert_eq!(timing.publication, 0);
}

/// Samples token 7 for every row. Each execute reports the next queued
/// prepared-gather wait, as a host completing its gathers would, and first
/// blocks on the device for `device_wait`, as a host readback would.
#[derive(Default)]
struct SampledExecutor {
    gather_waits: std::collections::VecDeque<u64>,
    device_wait: Duration,
    /// Per device wait: clock reads just before its timer starts, just
    /// after, just before it ends and just after.
    wait_windows: Vec<[Instant; 4]>,
    /// Proposes token 8 last, so each verify item rejects its last draft
    /// and keeps fewer tokens than it appended.
    reject_last: bool,
}

impl ModelExecutor for SampledExecutor {
    type TransactionState<'a> = NoopModelTransactionState;

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        _: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        if !self.device_wait.is_zero() {
            let before = Instant::now();
            let wait = profile::PhaseTimer::new(profile::StepPhase::DeviceWait);
            let started = Instant::now();
            std::thread::sleep(self.device_wait);
            let ending = Instant::now();
            drop(wait);
            self.wait_windows
                .push([before, started, ending, Instant::now()]);
        }
        if let Some(nanoseconds) = self.gather_waits.pop_front() {
            profile::record_mapped_gather_exposed(nanoseconds);
        }
        let mut sampled = Vec::new();
        let mut drafts = Vec::new();
        for item in items.iter().filter(|item| item.expects_logits()) {
            let count = item.draft.map_or(0, |draft| draft.tokens);
            if count > 0 {
                let mut proposed = vec![7; count];
                if self.reject_last {
                    proposed[count - 1] = 8;
                }
                drafts.push(DraftRow::new(item.request_id(), proposed));
            }
            for _ in 0..=count {
                sampled.push(SampledRow::new(item.request_id(), 7, -0.125, Vec::new()));
            }
        }
        Ok(ModelTransaction::no_op(Vec::new())
            .with_sampled(sampled)
            .with_drafts(drafts))
    }
}

fn engine(concurrency: usize, drafts: usize, trace: bool) -> Engine<SampledExecutor> {
    let geometry = KvGeometry::new(64, 2048).unwrap();
    let mut config = EngineConfig::new(
        SchedulerConfig::new(concurrency, concurrency * (drafts + 1), 2048, 2048, 1).unwrap(),
        geometry,
        ModelVocabulary::new(128).unwrap(),
        geometry.token_capacity(),
        32,
    )
    .unwrap();
    if drafts > 0 {
        config = config.with_speculation(
            SpeculationConfig::new(
                drafts,
                concurrency * (drafts + 1),
                SpeculationMode::Fixed,
                DraftSource::Bundle,
            )
            .unwrap(),
        );
    }
    let mut engine = Engine::new(config, SampledExecutor::default()).unwrap();
    engine.step_trace = trace;
    engine
        .metrics
        .compatibility(|metrics| metrics.step_trace_enabled = trace);
    for id in 1..=concurrency {
        engine
            .submit(
                EngineRequest::new(
                    RequestId::new(id as u64),
                    vec![1],
                    1024,
                    SamplingParameters::new(0.0).unwrap(),
                    1,
                    Vec::new(),
                    ModelVocabulary::new(128).unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
    }
    engine
}

#[test]
fn host_trace_preserves_sampled_events_and_is_disabled_without_tracing() {
    for (concurrency, drafts) in [(1, 0), (8, 3), (32, 0)] {
        let run = |trace| {
            let mut engine = engine(concurrency, drafts, trace);
            let mut events = Vec::new();
            for _ in 0..6 {
                let before = profile::snapshot().totals;
                let totals = trace.then(profile::TraceTotals::begin);
                if trace {
                    engine.host_clock.step_entered();
                }
                assert_eq!(
                    engine.step_scheduled_batch(&mut |_| {}).unwrap(),
                    StepStatus::Progressed
                );
                if trace {
                    let record = engine.pending_step_trace.take().unwrap();
                    let phases = profile::snapshot().totals;
                    assert_eq!(
                        record.host.scheduler_commit,
                        phases[profile::StepPhase::SchedulerCommit as usize]
                    );
                    assert_eq!(
                        record.host.model_commit,
                        phases[profile::StepPhase::ModelCommit as usize]
                    );
                    assert_eq!(
                        record.host.publication,
                        phases[profile::StepPhase::Publication as usize]
                    );
                    assert!(
                        record.host.commit
                            >= record.host.scheduler_commit
                                + record.host.model_commit
                                + record.host.publication
                    );
                } else {
                    assert!(engine.pending_step_trace.is_none());
                    assert_eq!(profile::snapshot().totals, before);
                }
                drop(totals);
                engine.audit().unwrap();
                events.extend(engine.drain_events());
            }
            assert!(!events.is_empty());
            events
        };
        assert_eq!(run(false), run(true));
    }
}

#[test]
fn traced_steps_report_the_prepared_gather_waits_their_execution_completed() {
    let mut engine = engine(4, 0, true);
    // No prepared gather, then 2.5 us, then none again.
    engine.executor.gather_waits = [0, 2_500, 0].into();
    let mut exposed = Vec::new();
    for _ in 0..4 {
        let _totals = profile::TraceTotals::begin();
        engine.host_clock.step_entered();
        assert_eq!(
            engine.step_scheduled_batch(&mut |_| {}).unwrap(),
            StepStatus::Progressed
        );
        let record = engine.pending_step_trace.take().unwrap();
        exposed.push(record.host.mapped_gather_exposed);
        assert!(!engine.drain_events().is_empty());
    }
    assert_eq!(exposed, [0, 2, 0, 0]);
}

/// `boundary_us` runs from the previous step's last device wait ending to
/// this step's first one starting: at least the gap between the executor
/// calls, and within the clock reads around the waits' timers, however the
/// test thread is scheduled.
#[test]
fn traced_steps_bound_device_idle_between_their_device_waits() {
    let wait = Duration::from_millis(20);
    let mut engine = engine(4, 0, true);
    engine.executor.device_wait = wait;
    let mut hosts = Vec::new();
    for _ in 0..4 {
        let _totals = profile::TraceTotals::begin();
        engine.host_clock.step_entered();
        assert_eq!(
            engine.step_scheduled_batch(&mut |_| {}).unwrap(),
            StepStatus::Progressed
        );
        hosts.push(engine.pending_step_trace.take().unwrap().host);
        assert!(!engine.drain_events().is_empty());
    }
    assert_eq!(hosts[0].boundary, None, "no earlier step waited");
    for host in &hosts {
        assert!(host.device_wait >= wait.as_micros() as u64);
    }
    let windows = &engine.executor.wait_windows;
    assert_eq!(windows.len(), hosts.len());
    for (step, host) in hosts.iter().enumerate().skip(1) {
        let (boundary, gap) = (host.boundary.unwrap(), host.gap.unwrap());
        let (previous, current) = (windows[step - 1], windows[step]);
        let shortest = current[0].saturating_duration_since(previous[3]);
        let longest = current[1].saturating_duration_since(previous[2]);
        assert!(gap <= boundary);
        assert!(
            (shortest.as_micros()..=longest.as_micros()).contains(&u128::from(boundary)),
            "boundary {boundary} us outside {shortest:?} to {longest:?}"
        );
    }
}

#[test]
#[ignore = "release CPU host timing microbenchmark"]
fn release_host_phase_costs() {
    assert!(!cfg!(debug_assertions), "run with --release");
    // Verify items accept every proposal, and then all but the last.
    let shapes = [
        (0, false),
        (3, false),
        (4, false),
        (7, false),
        (3, true),
        (7, true),
    ]
    .into_iter()
    .flat_map(|(drafts, reject_last)| {
        [1, 8, 16, 32, 64, 128]
            .into_iter()
            .map(move |concurrency| (concurrency, drafts, reject_last))
    });
    for (concurrency, drafts, reject_last) in shapes {
        let mut measured = Vec::new();
        for _ in 0..5 {
            let mut engine = engine(concurrency, drafts, true);
            engine.executor.reject_last = reject_last;
            for iteration in 0..100 {
                let _totals = profile::TraceTotals::begin();
                engine.host_clock.step_entered();
                assert_eq!(
                    engine.step_scheduled_batch(&mut |_| {}).unwrap(),
                    StepStatus::Progressed
                );
                let record = engine.pending_step_trace.take().unwrap();
                let events = engine.drain_events();
                assert!(!events.is_empty());
                if iteration >= 20 {
                    assert_eq!(record.prefill_tokens, 0);
                    assert_eq!(record.decode_rows + record.verify_items, concurrency);
                    measured.push(record);
                }
            }
        }
        let quantiles = |field: fn(&compatibility::StepTrace) -> u64| {
            let mut values: Vec<_> = measured.iter().map(field).collect();
            values.sort_unstable();
            [values[values.len() / 2], values[values.len() * 95 / 100]]
        };
        println!(
            "HOST_COST {}",
            serde_json::json!({
                "concurrency": concurrency, "maximum_drafts": drafts, "samples": measured.len(),
                "rejects_last": reject_last,
                "rows": measured[0].decode_rows + measured[0].verify_rows,
                "drafting": measured[0].verify_items,
                "commit_us": quantiles(|r| r.host.commit),
                "scheduler_commit_us": quantiles(|r| r.host.scheduler_commit),
                "model_commit_us": quantiles(|r| r.host.model_commit),
                "publication_us": quantiles(|r| r.host.publication),
                "sampling_us": quantiles(|r| r.host.sampling),
                "schedule_us": quantiles(|r| r.host.schedule),
            })
        );
    }
}
