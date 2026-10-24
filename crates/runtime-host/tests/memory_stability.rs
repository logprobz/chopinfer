//! Many requests through the real engine, scheduler, KV pool, prefix cache,
//! speculation and bundle executor adapter leave nothing behind, and the
//! process heap stays flat once the warm-up has filled its caches.
//!
//! Requests end every way a server ends them: by length and by a stop token,
//! by a frontend stop after their first token, and by a cancel while queued,
//! mid-prefill and mid-decode, under forced preemption with prefix-cache
//! anchors evicted for room. After each epoch the engine drains, and then
//! every per-request structure is back at its baseline: no live, terminal or
//! tracked request, no active sequence, no slot or committed history in the
//! bundle, and no reference-metrics record. The heap, counted by the
//! allocator, may move by the anchors that replace each other, never by the
//! requests served. One test, since the live-byte count is global.
// SAFETY-REVIEWED: the only unsafe code is the counting allocator below,
// which forwards to the system allocator.
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicIsize, Ordering};

use runtime_bundle_abi::{
    BundleAbiError, BundleKeptRowsV1, BundleStatusV1, OwnedLogitsRowV1, ResultSinkLimitsV1,
};
use runtime_bundle_sdk::BatchRequestV1;
use runtime_engine::{
    DraftCostModel, DraftSource, Engine, EngineConfig, EngineEventKind, EngineRequest,
    PromptLookupConfig, PromptLookupExecutor, SpeculationConfig, SpeculationMode,
};
use runtime_host::{BatchWireCapabilityV1, BundleExecutionV1, BundleModelExecutorV1};
use runtime_kv::KvGeometry;
use runtime_model_api::ModelVocabulary;
use runtime_sampling::SamplingParameters;
use runtime_scheduler::{
    AdmissionPolicy, OptimisticAdmission, RequestId, SchedulerCapacityProfile, SchedulerConfig,
};

struct Counting;

static LIVE: AtomicIsize = AtomicIsize::new(0);

// SAFETY-REVIEWED: every method forwards to the system allocator with the
// caller's pointer and layout unchanged and only updates a counter.
unsafe impl GlobalAlloc for Counting {
    // SAFETY-REVIEWED: forwards the caller's layout contract unchanged.
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY-REVIEWED: the caller guarantees a valid nonzero layout.
        let pointer = unsafe { System.alloc(layout) };
        count(pointer, layout.size() as isize)
    }

    // SAFETY-REVIEWED: forwards the caller's layout contract unchanged.
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY-REVIEWED: the caller guarantees a valid nonzero layout.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        count(pointer, layout.size() as isize)
    }

    // SAFETY-REVIEWED: forwards a pointer this allocator returned.
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::SeqCst);
        // SAFETY-REVIEWED: the pointer came from System with this layout.
        unsafe { System.dealloc(pointer, layout) }
    }

    // SAFETY-REVIEWED: forwards a pointer this allocator returned.
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY-REVIEWED: the pointer came from System with this layout.
        let next = unsafe { System.realloc(pointer, layout, size) };
        count(next, size as isize - layout.size() as isize)
    }
}

fn count(pointer: *mut u8, bytes: isize) -> *mut u8 {
    if !pointer.is_null() {
        LIVE.fetch_add(bytes, Ordering::SeqCst);
    }
    pointer
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn live() -> isize {
    LIVE.load(Ordering::SeqCst)
}

const VOCABULARY: usize = 256;
const SEQUENCES: usize = 8;
const PAGE_TOKENS: usize = 16;
const CONTEXT_TOKENS: usize = 512;
const KV_PAGES: usize = 160;
const BATCH_TOKENS: usize = 256;
const CHUNK_TOKENS: usize = 64;
const SNAPSHOT_PAGES: usize = 2;
const DRAFT_TOKENS: usize = 3;
const VERIFY_ROWS: usize = 16;
/// Requests in flight at once, beyond the engine's sequences: the rest wait
/// in its queue, where some are cancelled.
const IN_FLIGHT: usize = SEQUENCES + 4;
const PROMPTS: usize = 40;
const EPOCH_REQUESTS: usize = 240;
const WARMUP_EPOCHS: usize = 3;
const EPOCHS: usize = 10;
/// Heap the anchors that replace each other may move by between drained
/// epochs; a leak of 64 bytes a request over the measured epochs exceeds it.
const HEAP_TOLERANCE_BYTES: isize = 96 * 1024;

/// The bundle side: keeps each request's slot and committed tokens, as a
/// native bundle keeps its per-request state, and releases them on retire.
#[derive(Default)]
struct State {
    slots: BTreeMap<u64, u32>,
    committed: BTreeMap<u64, Vec<u32>>,
    retired: usize,
}

struct Execution {
    state: Rc<RefCell<State>>,
    pending: Vec<(u64, Vec<u32>, bool)>,
}

/// The model repeats the token four positions back, so prompt lookup finds
/// drafts it accepts.
fn logits_after(context: &[u32]) -> Vec<u8> {
    let favoured = context
        .len()
        .checked_sub(4)
        .map_or(7, |back| context[back] as usize)
        % VOCABULARY;
    (0..VOCABULARY)
        .flat_map(|token| f32::to_le_bytes(if token == favoured { 8.0 } else { 0.0 }))
        .collect()
}

impl BundleExecutionV1 for Execution {
    fn supports_request_retirement(&self) -> bool {
        true
    }

    fn supports_speculation(&self) -> bool {
        true
    }

    fn retire_request(&mut self, request_id: u64) -> Result<(), BundleAbiError> {
        let mut state = self.state.borrow_mut();
        state.slots.remove(&request_id);
        state.committed.remove(&request_id);
        state.retired += 1;
        Ok(())
    }

    fn capture_prefix_state(
        &mut self,
        _request_id: u64,
        _tokens: u64,
        _pages: &[u64],
    ) -> Result<(), BundleAbiError> {
        Ok(())
    }

    fn restore_prefix_state(
        &mut self,
        request_id: u64,
        token_ids: &[u32],
        _pages: &[u64],
    ) -> Result<(), BundleAbiError> {
        let mut state = self.state.borrow_mut();
        let slot = (0..SEQUENCES as u32)
            .find(|slot| !state.slots.values().any(|used| used == slot))
            .ok_or(BundleAbiError::Status(BundleStatusV1::ResourceExhausted))?;
        state.slots.insert(request_id, slot);
        state.committed.insert(request_id, token_ids.to_vec());
        Ok(())
    }

    fn execute_batch(
        &mut self,
        request: &[u8],
        _limits: ResultSinkLimitsV1,
    ) -> Result<Vec<OwnedLogitsRowV1>, BundleAbiError> {
        let request = BatchRequestV1::decode(request)
            .map_err(|_| BundleAbiError::Status(BundleStatusV1::InvalidArgument))?;
        let mut state = self.state.borrow_mut();
        let mut rows = Vec::new();
        for (item, append) in request.items().iter().zip(request.appends()) {
            let id = item.request_id();
            if !state.slots.contains_key(&id) && append.previous_tokens() == 0 {
                let slot = (0..SEQUENCES as u32)
                    .find(|slot| !state.slots.values().any(|used| used == slot))
                    .ok_or(BundleAbiError::Status(BundleStatusV1::NotFound))?;
                state.slots.insert(id, slot);
            }
            let inputs = item.input_tokens().to_vec();
            if item.expects_logits() {
                let mut context = state.committed.get(&id).cloned().unwrap_or_default();
                let first = if item.is_verify() {
                    0
                } else {
                    inputs.len() - 1
                };
                context.extend_from_slice(&inputs[..first]);
                for &input in &inputs[first..] {
                    context.push(input);
                    rows.push(OwnedLogitsRowV1::new(id, logits_after(&context)));
                }
            }
            self.pending.push((id, inputs, item.is_verify()));
        }
        Ok(rows)
    }

    fn commit(&mut self) -> Result<(), BundleAbiError> {
        let mut state = self.state.borrow_mut();
        for (id, inputs, _) in self.pending.drain(..) {
            state.committed.entry(id).or_default().extend(inputs);
        }
        Ok(())
    }

    fn commit_accepted(&mut self, records: &[BundleKeptRowsV1]) -> Result<(), BundleAbiError> {
        let mut state = self.state.borrow_mut();
        for (id, inputs, verify) in self.pending.drain(..) {
            let kept = records
                .iter()
                .find(|record| record.request_id == id)
                .filter(|_| verify)
                .map_or(inputs.len(), |record| record.kept_rows as usize);
            state
                .committed
                .entry(id)
                .or_default()
                .extend_from_slice(&inputs[..kept]);
        }
        Ok(())
    }

    fn cancel(&mut self) -> Result<(), BundleAbiError> {
        self.pending.clear();
        Ok(())
    }

    fn destroy(&mut self) -> Result<(), BundleAbiError> {
        Ok(())
    }

    fn is_poisoned(&self) -> bool {
        false
    }
}

fn engine(state: Rc<RefCell<State>>) -> Engine<impl runtime_engine::ModelExecutor> {
    let tracked = IN_FLIGHT + SEQUENCES;
    let admission = OptimisticAdmission::new(10_000, 3)
        .unwrap()
        .with_forced_preemption(7)
        .unwrap();
    let scheduler = SchedulerConfig::new(SEQUENCES, tracked, BATCH_TOKENS, CHUNK_TOKENS, 1)
        .unwrap()
        .with_capacity_profile(SchedulerCapacityProfile::new("served", IN_FLIGHT).unwrap())
        .unwrap()
        .with_admission(AdmissionPolicy::Optimistic(admission));
    let geometry = KvGeometry::new(PAGE_TOKENS, KV_PAGES).unwrap();
    let speculation = SpeculationConfig::new(
        DRAFT_TOKENS,
        VERIFY_ROWS,
        SpeculationMode::Adaptive,
        DraftSource::Host,
    )
    .unwrap()
    .with_draft_cost_model(DraftCostModel::PerBlock);
    let config = EngineConfig::new(
        scheduler,
        geometry,
        ModelVocabulary::new(VOCABULARY).unwrap(),
        CONTEXT_TOKENS,
        SEQUENCES,
    )
    .unwrap()
    .with_prefix_cache(SNAPSHOT_PAGES)
    .with_speculation(speculation);
    let execution = Execution {
        state,
        pending: Vec::new(),
    };
    let executor = BundleModelExecutorV1::new_with_batch_wire(
        execution,
        VOCABULARY,
        SEQUENCES as u64,
        BatchWireCapabilityV1::V3Speculative,
    );
    let executor = PromptLookupExecutor::new(executor, PromptLookupConfig::new(1, 4, 64).unwrap());
    Engine::new(config, executor).unwrap()
}

/// How a request ends.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Exit {
    Length,
    StopToken,
    FrontendStop,
    CancelQueued,
    CancelPrefill,
    CancelDecode,
}

fn exit(index: usize) -> Exit {
    match index % 10 {
        0 => Exit::CancelQueued,
        1 => Exit::CancelPrefill,
        2 => Exit::CancelDecode,
        3 => Exit::FrontendStop,
        4 => Exit::StopToken,
        _ => Exit::Length,
    }
}

/// One of `PROMPTS` prompts, 3 to 300 tokens; the longest take several
/// prefill chunks.
fn prompt(index: usize) -> Vec<u32> {
    let pool = index % PROMPTS;
    let length = 3 + (pool * 37) % 298;
    (0..length)
        .map(|row| 16 + ((pool * 131 + row * 17) % (VOCABULARY - 16)) as u32)
        .collect()
}

struct Live {
    exit: Exit,
    steps: usize,
    tokens: usize,
}

/// Serves `requests` requests from `first`, `IN_FLIGHT` at a time, until the
/// engine has none left. Returns how many ended each way.
fn serve(
    engine: &mut Engine<impl runtime_engine::ModelExecutor>,
    first: usize,
    requests: usize,
) -> BTreeMap<String, usize> {
    let vocabulary = ModelVocabulary::new(VOCABULARY).unwrap();
    let mut next = first;
    let mut live = BTreeMap::<RequestId, Live>::new();
    let mut ended = BTreeMap::<String, usize>::new();
    let end = |ended: &mut BTreeMap<String, usize>, how: &str| {
        *ended.entry(how.to_owned()).or_default() += 1;
    };
    while next < first + requests || !live.is_empty() {
        while next < first + requests && live.len() < IN_FLIGHT {
            let index = next;
            next += 1;
            let id = RequestId::new(index as u64 + 1);
            let how = exit(index);
            let sampling = if index % 2 == 0 {
                SamplingParameters::new(0.0).unwrap()
            } else {
                SamplingParameters::new(1.0).unwrap()
            };
            let maximum = [1, 4, 12, 30][index / 10 % 4];
            // The stop-token requests stop at the token the model favours
            // after their prompt, so they end at their first token.
            let prompt = prompt(index);
            let stops = if how == Exit::StopToken {
                vec![prompt[prompt.len().saturating_sub(4)]]
            } else {
                vec![]
            };
            let request = EngineRequest::new(
                id,
                prompt,
                maximum.max(2),
                sampling,
                index as u64,
                stops,
                vocabulary,
            )
            .unwrap();
            engine.submit(request).unwrap();
            if how == Exit::CancelQueued {
                engine.cancel(id).unwrap();
                end(&mut ended, "cancel-queued");
                continue;
            }
            live.insert(
                id,
                Live {
                    exit: how,
                    steps: 0,
                    tokens: 0,
                },
            );
        }
        engine.step().unwrap();
        for event in engine.drain_events() {
            let id = event.request_id();
            match event.kind() {
                EngineEventKind::Token { .. } => {
                    if let Some(request) = live.get_mut(&id) {
                        request.tokens += 1;
                    }
                }
                EngineEventKind::Finished { reason } => {
                    if live.remove(&id).is_some() {
                        end(&mut ended, &format!("finished-{reason:?}"));
                    }
                }
                EngineEventKind::Error { message } => panic!("request {id:?} failed: {message}"),
                _ => {}
            }
        }
        let mut leaving = Vec::new();
        for (id, request) in &mut live {
            request.steps += 1;
            let how = match request.exit {
                Exit::CancelPrefill if request.steps == 1 => Some("cancel-prefill"),
                Exit::CancelDecode if request.tokens >= 1 => Some("cancel-decode"),
                Exit::FrontendStop if request.tokens >= 1 => Some("frontend-stop"),
                _ => None,
            };
            if let Some(how) = how {
                leaving.push((*id, request.exit, how));
            }
        }
        for (id, how, label) in leaving {
            live.remove(&id);
            if how == Exit::FrontendStop {
                engine.stop(id).unwrap();
            } else {
                engine.cancel(id).unwrap();
            }
            end(&mut ended, label);
        }
        let _ = engine.drain_events();
    }
    // Ended requests are reaped a few at a time on later steps.
    for _ in 0..64 {
        if engine.terminal_count() == 0 {
            break;
        }
        engine.step().unwrap();
        let _ = engine.drain_events();
    }
    ended
}

/// What a drained engine still holds for requests.
#[derive(Debug, Eq, PartialEq)]
struct Held {
    terminal: usize,
    active_sequences: usize,
    requests_active: u64,
    metric_records: usize,
    bundle_slots: usize,
    bundle_histories: usize,
}

fn held(engine: &Engine<impl runtime_engine::ModelExecutor>, state: &RefCell<State>) -> Held {
    let state = state.borrow();
    Held {
        terminal: engine.terminal_count(),
        active_sequences: engine.kv_stats().active_sequences(),
        requests_active: engine.metrics().snapshot().requests_active,
        metric_records: engine.metrics().live_request_records(),
        bundle_slots: state.slots.len(),
        bundle_histories: state.committed.len(),
    }
}

#[test]
fn many_requests_leave_nothing_behind_and_the_heap_flat() {
    let state = Rc::new(RefCell::new(State::default()));
    let mut engine = engine(Rc::clone(&state));
    let nothing = Held {
        terminal: 0,
        active_sequences: 0,
        requests_active: 0,
        metric_records: 0,
        bundle_slots: 0,
        bundle_histories: 0,
    };
    let mut ended = BTreeMap::<String, usize>::new();
    let mut heap = Vec::new();
    for epoch in 0..EPOCHS {
        for (how, count) in serve(&mut engine, epoch * EPOCH_REQUESTS, EPOCH_REQUESTS) {
            *ended.entry(how).or_default() += count;
        }
        engine.audit().unwrap();
        assert_eq!(held(&engine, &state), nothing, "after epoch {epoch}");
        for id in 1..=((epoch + 1) * EPOCH_REQUESTS) as u64 {
            assert!(
                engine.request_state(RequestId::new(id)).is_err(),
                "request {id} is still tracked after epoch {epoch}"
            );
        }
        heap.push((live(), engine.kv_stats().anchors()));
    }
    let served = EPOCHS * EPOCH_REQUESTS;
    assert_eq!(ended.values().sum::<usize>(), served, "{ended:?}");
    for how in [
        "cancel-queued",
        "cancel-prefill",
        "cancel-decode",
        "frontend-stop",
    ] {
        assert!(
            ended.get(how).copied().unwrap_or(0) > 0,
            "no {how}: {ended:?}"
        );
    }
    assert!(
        ended
            .keys()
            .filter(|how| how.starts_with("finished-"))
            .count()
            >= 2,
        "{ended:?}"
    );
    let preemptions = engine
        .metrics()
        .prometheus_compatibility("fixture")
        .lines()
        .find_map(|line| line.strip_prefix("vllm:num_preemptions_total{"))
        .and_then(|line| line.rsplit(' ').next())
        .and_then(|count| count.parse::<u64>().ok())
        .unwrap_or(0);
    assert!(preemptions > 0, "no preemption");
    assert!(
        engine.metrics().snapshot().spec_verify_items > 0,
        "no verify item"
    );
    assert!(engine.kv_stats().anchors() > 0, "no prefix-cache anchor");
    assert!(state.borrow().retired >= served - ended["cancel-queued"]);
    let (baseline, _) = heap[WARMUP_EPOCHS - 1];
    eprintln!("heap after each epoch (bytes, anchors): {heap:?}");
    for (epoch, &(bytes, anchors)) in heap.iter().enumerate().skip(WARMUP_EPOCHS) {
        assert!(
            (bytes - baseline).abs() <= HEAP_TOLERANCE_BYTES,
            "the heap moved by {} bytes from the warm baseline after epoch {epoch} ({anchors} \
             anchors): {heap:?}",
            bytes - baseline
        );
    }
}
