//! Step overlap (`CHOPINFER_STEP_OVERLAP=1`) moves event delivery into the
//! next step's device wait and changes nothing that is decided or delivered.
//!
//! Each scenario drives the real worker loop twice over one scripted
//! executor, overlap off and on, and compares every request's stream, every
//! batch the engine formed and the engine's counters. The executor injects
//! arrivals and cancellations at fixed steps through the worker's own
//! control channel, so both runs see them at the same step boundary.
use super::*;
use runtime_engine::{
    DraftRow, DraftSource, EngineConfig, LogitsRow, ModelBatchItem, ModelTransaction,
    NoopModelTransactionState, SampledRow, SpeculationConfig, SpeculationMode,
};
use runtime_kv::{KvGeometry, PreparedAppendBatch};
use runtime_sampling::TokenLogprob as SampleTokenLogprob;
use runtime_scheduler::{AdmissionPolicy, BatchPhase, OptimisticAdmission, SchedulerConfig};
use std::cell::RefCell;
use std::collections::BTreeSet;

use super::stop_tests::declarations;

const VOCABULARY: usize = 259;
const EOS: u32 = 1;
/// A scenario that runs longer than this has stalled.
const MAXIMUM_STEPS: usize = 2_000;

fn byte(value: u8) -> u32 {
    u32::from(value) + 3
}

fn text(value: &str) -> Vec<u32> {
    value.bytes().map(byte).collect()
}

/// One request of a scenario.
#[derive(Clone)]
struct Planned {
    /// The step after which the request reaches the worker; 0 is before the
    /// first step.
    arrive: usize,
    /// The step after which its client cancels.
    cancel: Option<usize>,
    prompt: Vec<u32>,
    maximum: usize,
    /// What the model emits at each output position; EOS past its end.
    script: Vec<u32>,
    stop: Vec<String>,
    grammar: Option<&'static str>,
    temperature: f64,
    top: usize,
    /// The event stream's depth; nothing reads it until the run ends.
    capacity: usize,
}

impl Planned {
    fn new(prompt: &str, script: &str, maximum: usize) -> Self {
        Self {
            arrive: 0,
            cancel: None,
            prompt: text(prompt),
            maximum,
            script: text(script),
            stop: Vec::new(),
            grammar: None,
            temperature: 0.0,
            top: 0,
            capacity: 4096,
        }
    }
}

#[derive(Clone)]
struct Scenario {
    /// Rows come back sampled, as from a device sampler, or as logits the
    /// engine samples.
    sampled: bool,
    /// Every execute launches work and waits on the device, as a bundle does.
    device_wait: bool,
    speculative: bool,
    sequences: usize,
    kv_pages: usize,
    prefill_chunk: usize,
    optimistic: bool,
    /// Each request's events go through the serving forwarder, whose channel
    /// is `Planned::capacity` deep, to a client that keeps up. Every execute
    /// waits, after its device wait, until the forwarders have taken
    /// everything delivered so far, so reader progress is the same at every
    /// step boundary in both runs.
    forwarded: bool,
    requests: Vec<Planned>,
}

/// What one run produced.
#[derive(Debug, PartialEq)]
struct Outcome {
    streams: Vec<Vec<GenerationEvent>>,
    admissions: Vec<Option<Result<(), AdmissionError>>>,
    batches: Vec<Vec<(u64, BatchPhase)>>,
    counters: [u64; 9],
}

struct Run {
    outcome: Outcome,
    /// Events that were still queued when an execute began its device wait.
    deferred_at_wait: usize,
    preempted: usize,
}

/// Each batch's items, and how many events were queued at device waits.
type ExecutionLog = (Vec<Vec<(u64, BatchPhase)>>, usize);

struct ScriptedExecutor {
    scripts: BTreeMap<RequestId, Vec<u32>>,
    sampled: bool,
    device_wait: bool,
    step: usize,
    arrivals: BTreeMap<usize, Vec<WorkerCommand>>,
    cancels: BTreeMap<usize, Vec<CancellationSignal>>,
    control: SyncSender<WorkerCommand>,
    log: Arc<Mutex<ExecutionLog>>,
    /// The forwarders' channels, when requests are forwarded.
    forwarding: Vec<tokio::sync::mpsc::WeakSender<GenerationEvent>>,
}

impl ScriptedExecutor {
    fn token(&self, id: RequestId, position: usize) -> u32 {
        self.scripts[&id].get(position).copied().unwrap_or(EOS)
    }

    /// The scripted token, or the first one the row's constraint admits.
    fn allowed(&self, id: RequestId, position: usize, work: &runtime_engine::SamplingWork) -> u32 {
        let token = self.token(id, position);
        match &work.constraint {
            Some(constraint) if !constraint.allows(token) => (0..VOCABULARY as u32)
                .find(|candidate| constraint.allows(*candidate))
                .unwrap_or(EOS),
            _ => token,
        }
    }
}

impl ModelExecutor for ScriptedExecutor {
    type TransactionState<'a> = NoopModelTransactionState;

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        _: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, runtime_engine::ExecutorError> {
        assert!(self.step < MAXIMUM_STEPS, "the scenario stalled");
        let mut log = self.log.lock().unwrap();
        if self.device_wait {
            log.1 += crate::overlap::pending();
            crate::overlap::note_launch();
            crate::overlap::device_wait();
            assert_eq!(crate::overlap::pending(), 0, "the device wait delivered");
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        for channel in &self.forwarding {
            while let Some(sender) = channel.upgrade() {
                if sender.capacity() == sender.max_capacity() {
                    break;
                }
                assert!(Instant::now() < deadline, "the forwarder stalled");
                thread::yield_now();
            }
        }
        log.0.push(
            items
                .iter()
                .map(|item| (item.request_id().get(), item.phase()))
                .collect(),
        );
        drop(log);
        let mut rows = Vec::new();
        let mut sampled = Vec::new();
        let mut drafts = Vec::new();
        for item in items {
            let Some(work) = item.sampling() else {
                continue;
            };
            let id = item.request_id();
            let start = work.emitted_tokens;
            let count = item.draft().map_or(0, |draft| draft.tokens());
            // The drafts match the script up to a row that depends on the
            // request and its position, so acceptance varies.
            let wrong = (id.get() as usize + start) % (count + 2);
            let proposed: Vec<u32> = (0..count)
                .map(|row| {
                    let token = self.token(id, start + row);
                    if row == wrong { byte(b'#') } else { token }
                })
                .collect();
            if item.draft().is_some() {
                drafts.push(DraftRow::new(id, proposed.clone()));
            }
            for row in 0..=count {
                let row_work = work.for_verify_row(&proposed, row);
                if self.sampled {
                    let token = self.allowed(id, start + row, &row_work);
                    let top = (0..row_work.parameters.top_logprobs())
                        .map(|rank| {
                            SampleTokenLogprob::new(
                                (token + rank as u32) % VOCABULARY as u32,
                                -0.25 - rank as f64,
                            )
                        })
                        .collect();
                    sampled.push(SampledRow::new(id, token, -0.25, top));
                } else {
                    let mut logits = vec![0.0; VOCABULARY];
                    logits[self.token(id, start + row) as usize] = 8.0;
                    logits[byte(b'z') as usize] = 1.5;
                    rows.push(LogitsRow::new(id, logits)?);
                }
            }
        }
        self.step += 1;
        for command in self.arrivals.remove(&self.step).unwrap_or_default() {
            assert!(self.control.try_send(command).is_ok(), "control has room");
        }
        for cancellation in self.cancels.remove(&self.step).unwrap_or_default() {
            cancellation.cancel();
        }
        Ok(ModelTransaction::no_op(rows)
            .with_sampled(sampled)
            .with_drafts(drafts))
    }
}

fn grammar(pattern: &str) -> Arc<CompiledGrammar> {
    let tokenizer = super::stop_tests::tokenizer();
    let vocabulary = Arc::new(
        runtime_grammar::TokenVocabulary::new(
            VOCABULARY,
            (0..VOCABULARY as u32)
                .map(|id| runtime_grammar::VocabularyToken {
                    bytes: tokenizer.token_bytes(id).unwrap().to_vec(),
                    special: id < 3,
                })
                .collect(),
        )
        .unwrap(),
    );
    Arc::new(
        CompiledGrammar::new(
            &runtime_grammar::regex_grammar(pattern).unwrap(),
            vocabulary,
        )
        .unwrap(),
    )
}

fn run(scenario: &Scenario, overlap: bool) -> Run {
    let mut descriptor =
        FrontendDescriptorV1::from_component_bytes(declarations::KEY_VALUE.as_bytes()).unwrap();
    descriptor.vocabulary.size = VOCABULARY as u32;
    descriptor.vocabulary.eos_token_id = EOS;
    descriptor.vocabulary.stop_token_ids = vec![EOS];
    let policy = EngineModelPolicyV1::from_declaration(&descriptor);
    let (control, receiver) = mpsc::sync_channel(256);
    let mut arrivals: BTreeMap<usize, Vec<WorkerCommand>> = BTreeMap::new();
    let mut cancels: BTreeMap<usize, Vec<CancellationSignal>> = BTreeMap::new();
    let mut scripts = BTreeMap::new();
    let mut streams = Vec::new();
    let mut admissions = Vec::new();
    let forwarders = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .unwrap();
    let mut forwarding = Vec::new();
    for (index, planned) in scenario.requests.iter().enumerate() {
        let id = RequestId::new(index as u64 + 1);
        scripts.insert(id, planned.script.clone());
        let cancellation = CancellationSignal::new();
        if let Some(step) = planned.cancel {
            cancels.entry(step).or_default().push(cancellation.clone());
        }
        let mut sampling = SamplingParameters::new(planned.temperature)
            .unwrap()
            .with_top_logprobs(planned.top)
            .unwrap();
        if planned.temperature > 0.0 {
            sampling = sampling.with_top_p(0.9).unwrap();
        }
        let request = PreparedRequest {
            request_id: format!("overlap-{index}"),
            client_request_id: None,
            model: "fixture/alpha-1".into(),
            prompt: PreparedPrompt::TokenIds(planned.prompt.clone()),
            max_new_tokens: planned.maximum,
            stop: planned.stop.clone(),
            decoding: runtime_server::DecodingOptions::default(),
            sampling,
            seed: 17 + index as u64,
            logprobs: runtime_server::RequestedLogprobs {
                enabled: planned.top > 0,
                top: planned.top,
                echo: false,
                prompt_top: None,
                return_token_ids: true,
            },
            chat: None,
            constraint: None,
            cache_salt: None,
            cancellation,
        };
        let (sender, events) = if scenario.forwarded {
            let (client, events) = tokio::sync::mpsc::channel(4096);
            let (sender, forward) = tokio::sync::mpsc::channel(planned.capacity);
            forwarding.push(sender.downgrade());
            forwarders.spawn(forward_events(
                forward,
                EventSender::new(client),
                request.cancellation.clone(),
            ));
            (sender, events)
        } else {
            tokio::sync::mpsc::channel(planned.capacity)
        };
        let (response, admitted) = oneshot::channel();
        streams.push(events);
        admissions.push(admitted);
        let command = WorkerCommand::Submit(vec![WorkerSubmission {
            id,
            request: Box::new(request),
            constraint: planned.grammar.map(grammar),
            events: EventSender::new(sender),
            response,
        }]);
        if planned.arrive == 0 {
            control.try_send(command).ok().unwrap();
        } else {
            arrivals.entry(planned.arrive).or_default().push(command);
        }
    }
    let log = Arc::new(Mutex::new((Vec::new(), 0)));
    let executor = ScriptedExecutor {
        scripts,
        sampled: scenario.sampled,
        device_wait: scenario.device_wait,
        step: 0,
        arrivals,
        cancels,
        control: control.clone(),
        log: Arc::clone(&log),
        forwarding,
    };
    let mut scheduler =
        SchedulerConfig::new(scenario.sequences, 64, 256, scenario.prefill_chunk, 1).unwrap();
    if scenario.optimistic {
        scheduler = scheduler.with_admission(AdmissionPolicy::Optimistic(
            OptimisticAdmission::new(0, 2).unwrap(),
        ));
    }
    let config = EngineConfig::new(
        scheduler,
        KvGeometry::new(4, scenario.kv_pages).unwrap(),
        ModelVocabulary::new(VOCABULARY).unwrap(),
        128,
        64,
    )
    .unwrap();
    let config = if scenario.speculative {
        config.with_speculation(
            SpeculationConfig::new(3, 64, SpeculationMode::Fixed, DraftSource::Bundle).unwrap(),
        )
    } else {
        config
    };
    let engine = Engine::new(config, executor).unwrap();
    let metrics = engine.metrics();
    let backend = EngineBackendConfig::default().with_step_overlap(overlap);
    let tokenizer = Arc::new(super::stop_tests::tokenizer());
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
        );
    });
    // Nothing reads a stream until every request is terminal and the worker
    // has stopped, so what a stream holds does not depend on a reader.
    let deadline = Instant::now() + Duration::from_secs(20);
    while metrics.snapshot().requests_terminal < scenario.requests.len() as u64 {
        assert!(
            Instant::now() < deadline && !worker.is_finished(),
            "the scenario did not finish"
        );
        thread::sleep(Duration::from_millis(1));
    }
    control.send(WorkerCommand::Shutdown).unwrap();
    worker.join().expect("the worker ran the scenario");
    // A forwarder ends once the worker's side of its channel is gone.
    let streams = streams
        .into_iter()
        .map(|mut events| {
            let mut seen = Vec::new();
            if scenario.forwarded {
                while let Some(event) = events.blocking_recv() {
                    seen.push(event);
                }
            }
            while let Ok(event) = events.try_recv() {
                seen.push(event);
            }
            seen
        })
        .collect();
    forwarders.shutdown_timeout(Duration::from_secs(1));
    let admissions = admissions
        .into_iter()
        .map(|mut admitted| admitted.try_recv().ok())
        .collect();
    let snapshot = metrics.snapshot();
    let (batches, deferred_at_wait) = std::mem::take(&mut *log.lock().unwrap());
    let mut decoded = BTreeSet::new();
    let mut preempted = 0;
    for batch in &batches {
        for (id, phase) in batch {
            match phase {
                BatchPhase::Prefill { .. } if decoded.contains(id) => preempted += 1,
                BatchPhase::Prefill { .. } => {}
                _ => {
                    decoded.insert(*id);
                }
            }
        }
    }
    Run {
        outcome: Outcome {
            streams,
            admissions,
            batches,
            counters: [
                snapshot.requests_submitted,
                snapshot.requests_completed,
                snapshot.requests_terminal,
                snapshot.generation_tokens,
                snapshot.batches,
                snapshot.mixed_batches,
                snapshot.errors,
                snapshot.cancellations,
                snapshot.spec_draft_tokens,
            ],
        },
        deferred_at_wait,
        preempted,
    }
}

/// Runs a scenario with overlap off and on and requires the same outcome.
fn same_with_overlap(scenario: &Scenario) -> Run {
    let off = run(scenario, false);
    let on = run(scenario, true);
    assert_eq!(
        off.deferred_at_wait, 0,
        "overlap off delivers before each step"
    );
    if scenario.device_wait {
        assert!(
            on.deferred_at_wait > 0,
            "overlap on delivers in device waits"
        );
    }
    assert_eq!(off.outcome, on.outcome);
    on
}

/// Every stream ends with a finish or a failure, except cancelled ones.
fn finished(run: &Run) -> Vec<Option<GenerationEvent>> {
    run.outcome
        .streams
        .iter()
        .map(|stream| {
            stream
                .last()
                .filter(|event| {
                    matches!(
                        event,
                        GenerationEvent::Finished { .. } | GenerationEvent::Failed { .. }
                    )
                })
                .cloned()
        })
        .collect()
}

fn content(stream: &[GenerationEvent]) -> String {
    stream
        .iter()
        .filter_map(|event| match event {
            GenerationEvent::Content(content) => Some(content.delta.as_str()),
            _ => None,
        })
        .collect()
}

/// Mixed prefill and decode, arrivals, a cancellation, a stop string, an
/// engine stop, a grammar, a length finish and seeded sampling in one run.
fn mixed(sampled: bool, device_wait: bool, speculative: bool) -> Scenario {
    let long = "a long prompt that prefills in chunks while others decode";
    let mut stop = Planned::new("stop", "abc END later", 24);
    stop.stop = vec!["END".into()];
    stop.top = 2;
    let mut eos = Planned::new("eos", "short", 24);
    eos.top = 1;
    let mut constrained = Planned::new("grammar", "yyyyyyyy", 12);
    constrained.grammar = Some("x{5}");
    let mut seeded = Planned::new("seeded", "seeded tokens go here", 10);
    seeded.temperature = 0.8;
    seeded.top = 3;
    let mut late = Planned::new(long, "late arrival text", 14);
    late.arrive = 3;
    let mut cancelled = Planned::new("cancel me", "never finishes this line", 40);
    cancelled.cancel = Some(4);
    let mut latest = Planned::new(long, "the last one", 9);
    latest.arrive = 6;
    Scenario {
        sampled,
        device_wait,
        speculative,
        sequences: 8,
        kv_pages: 512,
        prefill_chunk: 8,
        optimistic: false,
        forwarded: false,
        requests: vec![
            Planned::new("hello", "plain decode output", 16),
            stop,
            eos,
            constrained,
            seeded,
            late,
            cancelled,
            latest,
        ],
    }
}

#[test]
fn overlap_changes_no_stream_or_batch_in_mixed_serving() {
    for sampled in [false, true] {
        for device_wait in [true, false] {
            for speculative in [false, true] {
                let scenario = mixed(sampled, device_wait, speculative);
                let run = same_with_overlap(&scenario);
                let finished = finished(&run);
                let streams = &run.outcome.streams;
                assert!(
                    matches!(
                        finished[0],
                        Some(GenerationEvent::Finished {
                            reason: FinishReason::Length,
                            ..
                        })
                    ),
                    "{finished:?}"
                );
                assert_eq!(content(&streams[1]), "abc ");
                assert!(matches!(
                    finished[1],
                    Some(GenerationEvent::Finished {
                        reason: FinishReason::Stop,
                        ..
                    })
                ));
                assert_eq!(content(&streams[2]), "short");
                assert_eq!(content(&streams[3]), "xxxxx");
                assert!(finished[4].is_some() && finished[5].is_some());
                assert!(
                    finished[6].is_none(),
                    "the cancelled stream ends unfinished"
                );
                assert!(finished[7].is_some());
                assert!(
                    run.outcome.counters[5] > 0,
                    "prefill chunks ran beside decodes"
                );
                if speculative {
                    assert!(run.outcome.counters[8] > 0, "verify items ran");
                }
            }
        }
    }
}

#[test]
fn overlap_changes_no_preemption_or_recompute() {
    for sampled in [false, true] {
        for speculative in [false, true] {
            let requests = (0..6)
                .map(|index| {
                    let mut planned = Planned::new(
                        "a prompt of some length",
                        &format!("output {index} runs long enough to need more pages"),
                        40,
                    );
                    planned.arrive = index / 2;
                    planned.top = index % 3;
                    planned
                })
                .collect();
            let scenario = Scenario {
                sampled,
                device_wait: true,
                speculative,
                sequences: 6,
                kv_pages: 40,
                prefill_chunk: 16,
                optimistic: true,
                forwarded: false,
                requests,
            };
            let run = same_with_overlap(&scenario);
            assert!(run.preempted > 0, "the pool preempted and recomputed");
            assert!(finished(&run).iter().all(Option::is_some));
        }
    }
}

#[test]
fn an_overflowing_stream_is_cancelled_at_the_same_step_with_or_without_overlap() {
    for sampled in [false, true] {
        let mut slow = Planned::new(
            "slow reader",
            "many tokens for a reader that never reads",
            40,
        );
        slow.capacity = 3;
        slow.top = 1;
        let mut late = Planned::new("late", "arrives while the slow one runs", 20);
        late.arrive = 4;
        let scenario = Scenario {
            sampled,
            device_wait: true,
            speculative: false,
            sequences: 4,
            kv_pages: 256,
            prefill_chunk: 16,
            optimistic: false,
            forwarded: false,
            requests: vec![slow, Planned::new("steady", "steady output", 20), late],
        };
        let run = same_with_overlap(&scenario);
        let streams = &run.outcome.streams;
        assert_eq!(streams[0].len(), 3, "the stream holds what fit");
        assert!(!matches!(
            streams[0].last(),
            Some(GenerationEvent::Finished { .. })
        ));
        assert_eq!(content(&streams[1]), "steady output");
        assert_eq!(
            run.outcome.counters[7], 1,
            "the engine cancelled the request"
        );
    }
}

/// Through the serving forwarder with a reader that keeps up, a stream about
/// one step's events deep delivers the same events and finishes the same way
/// with overlap off and on, K3 verify steps and finish events included.
#[test]
fn a_reader_that_keeps_up_sees_the_same_stream_through_the_forwarder() {
    for sampled in [false, true] {
        for speculative in [false, true] {
            let mut stop = Planned::new("stop", "abc END later", 24);
            stop.stop = vec!["END".into()];
            let mut eos = Planned::new("eos", "short text", 24);
            let mut constrained = Planned::new("grammar", "yyyyyyyy", 12);
            constrained.grammar = Some("x{5}");
            let mut length = Planned::new("length", "a long line that runs to the limit", 18);
            let mut late = Planned::new("late", "arrives later on", 30);
            late.arrive = 3;
            for planned in [
                &mut stop,
                &mut eos,
                &mut constrained,
                &mut length,
                &mut late,
            ] {
                // Four tokens a step, each with logprobs, token IDs and
                // text, plus the finish: one step's events fill most of it.
                planned.top = 1;
                planned.capacity = 16;
            }
            let scenario = Scenario {
                sampled,
                device_wait: true,
                speculative,
                sequences: 8,
                kv_pages: 256,
                prefill_chunk: 16,
                optimistic: false,
                forwarded: true,
                requests: vec![stop, eos, constrained, length, late],
            };
            let run = same_with_overlap(&scenario);
            assert!(
                finished(&run).iter().all(Option::is_some),
                "every stream finished"
            );
            assert_eq!(run.outcome.counters[7], 0, "nothing was cancelled");
            assert_eq!(content(&run.outcome.streams[0]), "abc ");
            assert_eq!(content(&run.outcome.streams[3]), "a long line that r");
            if speculative {
                assert!(run.outcome.counters[8] > 0, "verify items ran");
            }
        }
    }
}

fn deferred_outbox(capacity: usize) -> (Outbox, tokio::sync::mpsc::Receiver<GenerationEvent>) {
    let (sender, receiver) = tokio::sync::mpsc::channel(capacity);
    let outbox = Outbox::new(EventSender::new(sender), CancellationSignal::new(), true);
    (outbox, receiver)
}

fn token(id: u32) -> GenerationEvent {
    GenerationEvent::GeneratedTokenIds(vec![id])
}

fn finish() -> GenerationEvent {
    GenerationEvent::Finished {
        reason: FinishReason::Length,
        usage: TokenUsage {
            prompt_tokens: 1,
            completion_tokens: 2,
            cached_tokens: None,
        },
    }
}

#[test]
fn a_deferred_event_is_refused_only_while_the_channel_is_full_with_everything_sent() {
    let (outbox, mut receiver) = deferred_outbox(3);
    for id in 0..3 {
        assert!(outbox.try_emit(token(id)).is_ok());
    }
    assert!(receiver.try_recv().is_err(), "queued, not sent");
    // The queue fills the channel, so the fourth is refused, after the
    // queued three were sent, as an immediate send of all four would be.
    assert!(outbox.try_emit(finish()).is_err());
    assert_eq!(crate::overlap::pending(), 0, "the queue was sent first");
    let delivered: Vec<_> = std::iter::from_fn(|| receiver.try_recv().ok()).collect();
    assert_eq!(delivered, [token(0), token(1), token(2)]);
}

#[test]
fn a_finish_is_accepted_when_the_reader_makes_room_while_the_queue_is_sent() {
    let (outbox, receiver) = deferred_outbox(2);
    let receiver = Rc::new(RefCell::new(receiver));
    let read = Rc::new(RefCell::new(Vec::new()));
    assert!(outbox.try_emit(token(7)).is_ok());
    // A reader that runs while the queue is sent takes the first event.
    let (reader, seen) = (Rc::clone(&receiver), Rc::clone(&read));
    crate::overlap::defer(move || {
        while let Ok(event) = reader.borrow_mut().try_recv() {
            seen.borrow_mut().push(event);
        }
    });
    assert!(outbox.try_emit(token(8)).is_ok());
    // Two queued events use up the channel; sending them lets the reader
    // take one, so the finish fits and is not refused.
    assert!(outbox.try_emit(finish()).is_ok());
    crate::overlap::run_pending();
    let mut receiver = receiver.borrow_mut();
    read.borrow_mut()
        .extend(std::iter::from_fn(|| receiver.try_recv().ok()));
    assert_eq!(*read.borrow(), [token(7), token(8), finish()]);
}

#[test]
fn deferred_backpressure_can_refuse_what_immediate_delivery_fits() {
    // The documented difference, on a two-event channel whose reader takes
    // each event as soon as it arrives. Delivered at once, the reader keeps
    // the channel empty and the finish fits. Queued, the two events count
    // against the channel until the queue runs, so the finish is refused
    // when no reader runs meanwhile.
    for overlap in [false, true] {
        let (sender, mut receiver) = tokio::sync::mpsc::channel(2);
        let outbox = Outbox::new(EventSender::new(sender), CancellationSignal::new(), overlap);
        let mut accepted = Vec::new();
        for event in [token(1), token(2), finish()] {
            accepted.push(outbox.try_emit(event).is_ok());
            // The reader keeps up with whatever reached the channel.
            while receiver.try_recv().is_ok() {}
        }
        crate::overlap::run_pending();
        if overlap {
            assert_eq!(accepted, [true, true, false]);
        } else {
            assert_eq!(accepted, [true, true, true]);
        }
    }
}
