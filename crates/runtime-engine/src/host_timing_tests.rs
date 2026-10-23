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

#[derive(Default)]
struct SampledExecutor;

impl ModelExecutor for SampledExecutor {
    type TransactionState<'a> = NoopModelTransactionState;

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        _: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        let mut sampled = Vec::new();
        let mut drafts = Vec::new();
        for item in items.iter().filter(|item| item.expects_logits()) {
            let count = item.draft.map_or(0, |draft| draft.tokens);
            if count > 0 {
                drafts.push(DraftRow::new(item.request_id(), vec![7; count]));
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
            SpeculationConfig::new(drafts, 32, SpeculationMode::Fixed, DraftSource::Bundle)
                .unwrap(),
        );
    }
    let mut engine = Engine::new(config, SampledExecutor).unwrap();
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
                    engine.step_scheduled_batch().unwrap(),
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
#[ignore = "release CPU host timing microbenchmark"]
fn release_host_phase_costs() {
    assert!(!cfg!(debug_assertions), "run with --release");
    for (concurrency, drafts) in [(1, 0), (1, 5), (8, 0), (8, 4), (8, 3), (16, 0), (32, 0)] {
        let mut measured = Vec::new();
        for _ in 0..5 {
            let mut engine = engine(concurrency, drafts, true);
            for iteration in 0..100 {
                let _totals = profile::TraceTotals::begin();
                engine.host_clock.step_entered();
                assert_eq!(
                    engine.step_scheduled_batch().unwrap(),
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
