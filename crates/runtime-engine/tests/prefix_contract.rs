use std::collections::{BTreeMap, BTreeSet};

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use runtime_engine::{
    Engine, EngineConfig, EngineError, EngineEventKind, EngineMetrics, EngineRequest,
    ExecutorError, LogitsRow, ModelBatchItem, ModelExecutor, ModelTransaction,
    ModelTransactionState, MonotonicClock, RequestId,
};
use runtime_kv::{BlockId, KvGeometry, PreparedAppendBatch};
use runtime_model_api::ModelVocabulary;
use runtime_sampling::SamplingParameters;
use runtime_scheduler::SchedulerConfig;

const VOCABULARY: usize = 32;
const SEED: u64 = 0x9e37_79b9_7f4a_7c15;

fn fold(state: u64, tokens: &[u32]) -> u64 {
    tokens.iter().fold(state, |state, token| {
        (state.rotate_left(9) ^ u64::from(*token)).wrapping_mul(0x100_0000_01b3)
    })
}

/// A model whose logits depend on every committed token of a request.
///
/// Its per-request state is a fold of those tokens, the way a sliding-window
/// ring or a recurrent state depends on history the paged KV does not hold. A
/// snapshot writes that state and its length into one page.
#[derive(Default)]
struct FoldingExecutor {
    committed: BTreeMap<RequestId, (u64, usize)>,
    pages: BTreeMap<usize, (u64, usize)>,
    refuse_capture: bool,
    refuse_restore: bool,
    /// Refusals report that device work may still access the pages.
    unconfirmed: bool,
    captures: Vec<(RequestId, usize)>,
    restores: Vec<(RequestId, usize)>,
    /// Requests admitted and not yet retired, as a bundle with per-request
    /// state tracks them.
    admitted: BTreeSet<RequestId>,
    input_tokens: usize,
    /// Updates state in place, so a step is never rolled back for one
    /// request's sampling failure.
    in_place: bool,
    /// The state an anchor item kept at the last multiple of
    /// [`ANCHOR_GRANULE`] inside its rows, per request, staged then committed.
    anchors: BTreeMap<RequestId, (u64, usize)>,
    anchor_items: usize,
    executions: usize,
}

/// The prefill granule of the anchor tests.
const ANCHOR_GRANULE: usize = 8;

struct FoldCommit<'a> {
    committed: &'a mut BTreeMap<RequestId, (u64, usize)>,
    staged: Vec<(RequestId, (u64, usize))>,
    anchors: &'a mut BTreeMap<RequestId, (u64, usize)>,
    staged_anchors: Vec<(RequestId, (u64, usize))>,
}

impl ModelTransactionState for FoldCommit<'_> {
    fn commit(&mut self) -> Result<(), ExecutorError> {
        for (id, state) in self.staged.drain(..) {
            self.committed.insert(id, state);
        }
        self.anchors.clear();
        self.anchors.extend(self.staged_anchors.drain(..));
        Ok(())
    }

    fn rollback(&mut self) -> Result<(), ExecutorError> {
        self.staged.clear();
        self.staged_anchors.clear();
        Ok(())
    }
}

impl FoldingExecutor {
    fn refusal(&self, message: &str) -> ExecutorError {
        if self.unconfirmed {
            ExecutorError::fatal(message).unwrap()
        } else {
            ExecutorError::new(message).unwrap()
        }
    }
}

impl ModelExecutor for FoldingExecutor {
    type TransactionState<'a> = FoldCommit<'a>;

    fn rollback_preserves_state_v1(&self) -> bool {
        !self.in_place
    }

    fn admit_request_v1(&mut self, id: RequestId) -> Result<(), ExecutorError> {
        self.admitted.insert(id);
        Ok(())
    }

    fn retire_request_v1(&mut self, id: RequestId) -> Result<(), ExecutorError> {
        self.committed.remove(&id);
        self.admitted.remove(&id);
        Ok(())
    }

    fn capture_prefix_state_v1(
        &mut self,
        id: RequestId,
        tokens: usize,
        pages: &[BlockId],
    ) -> Result<(), ExecutorError> {
        if self.refuse_capture {
            return Err(self.refusal("refused capture"));
        }
        let mut state = self.committed[&id];
        if tokens < state.1 {
            state = self.anchors[&id];
        }
        assert_eq!(
            state.1, tokens,
            "a capture names the committed length or the kept anchor"
        );
        assert_eq!(pages.len(), 1);
        self.pages.insert(pages[0].index(), state);
        self.captures.push((id, tokens));
        Ok(())
    }

    fn restore_prefix_state_v1(
        &mut self,
        id: RequestId,
        prefix: &[u32],
        pages: &[BlockId],
    ) -> Result<(), ExecutorError> {
        assert!(self.admitted.contains(&id), "admission precedes restore");
        if self.refuse_restore {
            return Err(self.refusal("refused restore"));
        }
        assert!(
            !self.committed.contains_key(&id),
            "restore precedes execution"
        );
        let state = self.pages[&pages[0].index()];
        assert_eq!(state, (fold(SEED, prefix), prefix.len()));
        self.committed.insert(id, state);
        self.restores.push((id, prefix.len()));
        Ok(())
    }

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        kv_plan: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        self.executions += 1;
        self.input_tokens += items
            .iter()
            .map(|item| item.input_tokens().len())
            .sum::<usize>();
        let mut staged = Vec::new();
        let mut staged_anchors = Vec::new();
        let mut rows = Vec::new();
        for (item, append) in items.iter().zip(kv_plan.appends()) {
            if !self.admitted.contains(&item.request_id()) {
                return Err(ExecutorError::new("the request was never admitted").unwrap());
            }
            let (state, length) = match self.committed.get(&item.request_id()) {
                Some(state) => *state,
                None => (SEED, 0),
            };
            if length != append.previous_tokens() {
                return Err(ExecutorError::new("executor state does not match the KV").unwrap());
            }
            if item.anchor() {
                let end = length + item.input_tokens().len();
                let anchor = end / ANCHOR_GRANULE * ANCHOR_GRANULE;
                assert!(
                    length < anchor && anchor < end,
                    "the anchor is inside the item"
                );
                let kept = fold(state, &item.input_tokens()[..anchor - length]);
                staged_anchors.push((item.request_id(), (kept, anchor)));
                self.anchor_items += 1;
            }
            let state = fold(state, item.input_tokens());
            staged.push((
                item.request_id(),
                (state, length + item.input_tokens().len()),
            ));
            if item.expects_logits() {
                let mut logits = vec![-10.0; VOCABULARY];
                logits[1 + (state % 30) as usize] = 10.0;
                rows.push(LogitsRow::new(item.request_id(), logits)?);
            }
        }
        Ok(ModelTransaction::new(
            rows,
            FoldCommit {
                committed: &mut self.committed,
                staged,
                anchors: &mut self.anchors,
                staged_anchors,
            },
        ))
    }
}

fn config(prefix: bool) -> EngineConfig {
    let config = EngineConfig::new(
        SchedulerConfig::new(4, 16, 16, 8, 1).unwrap(),
        KvGeometry::new(4, 64).unwrap(),
        ModelVocabulary::new(VOCABULARY).unwrap(),
        128,
        4,
    )
    .unwrap();
    if prefix {
        config.with_prefix_cache(1)
    } else {
        config
    }
}

fn request(id: u64, prompt: &[u32], maximum: usize) -> EngineRequest {
    EngineRequest::new(
        RequestId::new(id),
        prompt.to_vec(),
        maximum,
        SamplingParameters::new(0.0).unwrap(),
        id,
        Vec::new(),
        ModelVocabulary::new(VOCABULARY).unwrap(),
    )
    .unwrap()
}

/// Runs one request to completion and returns its tokens and cached count.
fn run(
    engine: &mut Engine<FoldingExecutor>,
    id: u64,
    prompt: &[u32],
    maximum: usize,
) -> (Vec<u32>, Option<usize>) {
    run_request(engine, request(id, prompt, maximum), id)
}

fn run_request(
    engine: &mut Engine<FoldingExecutor>,
    request: EngineRequest,
    id: u64,
) -> (Vec<u32>, Option<usize>) {
    engine.submit(request).unwrap();
    engine.run_until_idle(64).unwrap();
    engine.audit().unwrap();
    let mut tokens = Vec::new();
    let mut cached = None;
    for event in engine.drain_events() {
        assert_eq!(event.request_id(), RequestId::new(id));
        match event.kind() {
            EngineEventKind::Token { token_id, .. } => tokens.push(*token_id),
            EngineEventKind::PromptCached { tokens } => {
                assert!(cached.replace(*tokens).is_none());
            }
            _ => {}
        }
    }
    (tokens, cached)
}

fn cold(prompt: &[u32], maximum: usize) -> Vec<u32> {
    let mut engine = Engine::new(config(false), FoldingExecutor::default()).unwrap();
    run(&mut engine, 9, prompt, maximum).0
}

fn words(length: u32, salt: u32) -> Vec<u32> {
    (0..length)
        .map(|index| (index * 7 + salt) % 31 + 1)
        .collect()
}

#[test]
fn a_follow_up_turn_resumes_the_previous_prompt_and_matches_a_cold_run() {
    let mut engine = Engine::new(config(true), FoldingExecutor::default()).unwrap();
    let first = words(10, 0);
    let (_, cached) = run(&mut engine, 1, &first, 3);
    assert_eq!(cached, None);
    assert_eq!(
        engine.executor().captures,
        [(RequestId::new(1), 8), (RequestId::new(1), 10)],
        "the final chunk's start and the prompt end"
    );

    let mut second = first.clone();
    second.extend(words(13, 5));
    let (tokens, cached) = run(&mut engine, 2, &second, 4);
    assert_eq!(cached, Some(10));
    assert_eq!(engine.executor().restores, [(RequestId::new(2), 10)]);
    assert_eq!(tokens, cold(&second, 4));
    let kv = engine.kv_metrics().snapshot();
    assert_eq!(kv.prefix_cache_hits, 1);
    assert_eq!(kv.prefix_cache_reused_tokens, 10);
    assert_eq!(engine.kv_stats().used_blocks(), 0);
    assert!(engine.kv_stats().cached_blocks() > 0);
}

#[test]
fn a_cache_salt_shares_prefixes_only_with_the_same_salt() {
    let mut engine = Engine::new(config(true), FoldingExecutor::default()).unwrap();
    let salted = |id: u64, prompt: &[u32], salt: &[u8]| {
        request(id, prompt, 2).with_cache_salt(salt.to_vec())
    };
    let first = words(10, 2);
    let mut second = first.clone();
    second.extend(words(7, 4));
    run_request(&mut engine, salted(1, &first, b"tenant-a"), 1);
    let (_, cached) = run_request(&mut engine, salted(2, &second, b"tenant-b"), 2);
    assert_eq!(cached, None, "another salt misses");
    let (_, cached) = run(&mut engine, 3, &second, 2);
    assert_eq!(cached, None, "no salt misses a salted anchor");
    let (tokens, cached) = run_request(&mut engine, salted(4, &second, b"tenant-a"), 4);
    assert_eq!(cached, Some(10), "the same salt hits");
    assert_eq!(tokens, cold(&second, 2));
}

/// A clock that moves 5 microseconds every time it is read.
#[derive(Debug, Default)]
struct TickingClock(AtomicU64);

impl MonotonicClock for TickingClock {
    fn now_micros(&self) -> u64 {
        self.0.fetch_add(5, Ordering::Relaxed)
    }
}

#[test]
fn capture_and_restore_time_is_counted_per_call() {
    let metrics = EngineMetrics::default();
    let mut engine = Engine::new_with_observability(
        config(true),
        FoldingExecutor::default(),
        Arc::new(TickingClock::default()),
        metrics.clone(),
    )
    .unwrap();
    let first = words(10, 0);
    run(&mut engine, 1, &first, 3);
    let mut second = first.clone();
    second.extend(words(13, 5));
    run(&mut engine, 2, &second, 4);
    let captures = engine.executor().captures.len() as u64;
    assert_eq!(engine.executor().restores.len(), 1);
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.prefix_capture_micros, 5 * captures);
    assert_eq!(snapshot.prefix_restore_micros, 5);
    let text = snapshot.prometheus();
    assert!(text.contains(&format!(
        "chop_prefix_cache_capture_microseconds_total {}",
        5 * captures
    )));
    assert!(text.contains("chop_prefix_cache_restore_microseconds_total 5"));
}

/// Prompts of one chunk capture only their end, so a shared system prompt
/// hits once a second prompt has shown where the shared blocks end.
#[test]
fn a_repeated_system_prompt_hits_from_its_third_use() {
    let mut engine = Engine::new(config(true), FoldingExecutor::default()).unwrap();
    let system = words(5, 3);
    let turn = |salt| {
        let mut prompt = system.clone();
        prompt.extend(words(3, salt));
        prompt
    };
    let (_, cached) = run(&mut engine, 1, &turn(11), 1);
    assert_eq!(
        cached, None,
        "a prompt that finishes at its capture still captures"
    );
    let (_, cached) = run(&mut engine, 2, &turn(12), 2);
    assert_eq!(cached, None);
    assert!(
        engine.executor().captures.contains(&(RequestId::new(2), 4)),
        "the second use captures the shared blocks"
    );
    let third = turn(13);
    let (tokens, cached) = run(&mut engine, 3, &third, 3);
    assert_eq!(cached, Some(4));
    assert_eq!(tokens, cold(&third, 3));
}

/// A prompt longer than a chunk also captures its final chunk's start, a
/// chunk end anyway. The same prompt, or one that diverges in that final
/// chunk, hits there from its second use and matches a cold run.
#[test]
fn a_repeated_long_prompt_hits_from_its_second_use() {
    let mut engine = Engine::new(config(true), FoldingExecutor::default()).unwrap();
    let prompt = words(20, 3);
    let (first, cached) = run(&mut engine, 1, &prompt, 3);
    assert_eq!(cached, None);
    assert_eq!(
        engine.executor().captures,
        [(RequestId::new(1), 16), (RequestId::new(1), 20)]
    );
    let (tokens, cached) = run(&mut engine, 2, &prompt, 3);
    assert_eq!(cached, Some(16));
    assert_eq!(tokens, first);
    assert_eq!(tokens, cold(&prompt, 3));
    let mut fork = prompt[..17].to_vec();
    fork.extend(words(4, 9));
    let (tokens, cached) = run(&mut engine, 3, &fork, 3);
    assert_eq!(cached, Some(16));
    assert_eq!(tokens, cold(&fork, 3));
}

#[test]
fn a_refused_restore_prefills_everything_and_a_refused_capture_drops_its_anchor() {
    let mut engine = Engine::new(config(true), FoldingExecutor::default()).unwrap();
    let first = words(12, 1);
    run(&mut engine, 1, &first, 2);
    engine.executor_mut().refuse_restore = true;
    let mut second = first.clone();
    second.extend(words(3, 2));
    let (tokens, cached) = run(&mut engine, 2, &second, 3);
    assert_eq!(cached, None);
    assert_eq!(tokens, cold(&second, 3));
    let kv = engine.kv_metrics().snapshot();
    assert_eq!(kv.prefix_cache_restore_failures, 1);
    assert_eq!(kv.prefix_cache_queries, 2);
    assert_eq!(
        (kv.prefix_cache_hits, kv.prefix_cache_reused_tokens),
        (0, 0),
        "a refused restore reuses nothing"
    );

    engine.executor_mut().refuse_restore = false;
    engine.executor_mut().refuse_capture = true;
    let other = words(9, 30);
    run(&mut engine, 3, &other, 1);
    let mut longer = other.clone();
    longer.push(4);
    let (tokens, cached) = run(&mut engine, 4, &longer, 2);
    assert_eq!(cached, None, "a refused capture leaves nothing to hit");
    assert_eq!(tokens, cold(&longer, 2));
    assert!(engine.kv_metrics().snapshot().prefix_cache_capture_failures >= 1);
}

/// Steps until the engine fails, and returns that error.
fn step_until_error(engine: &mut Engine<FoldingExecutor>) -> EngineError {
    for _ in 0..64 {
        if let Err(error) = engine.step() {
            return error;
        }
    }
    panic!("the engine did not fail");
}

#[test]
fn an_unconfirmed_capture_or_restore_keeps_its_pages_and_stops_the_engine() {
    for restore in [false, true] {
        let mut engine = Engine::new(config(true), FoldingExecutor::default()).unwrap();
        let first = words(12, 1);
        if restore {
            run(&mut engine, 1, &first, 2);
            engine.executor_mut().refuse_restore = true;
        } else {
            engine.executor_mut().refuse_capture = true;
        }
        engine.executor_mut().unconfirmed = true;
        let anchors = engine.kv_metrics().snapshot().prefix_cache_anchors;
        let mut second = first.clone();
        second.extend(words(3, 2));
        engine.submit(request(2, &second, 3)).unwrap();
        let error = step_until_error(&mut engine);
        assert!(
            matches!(&error, EngineError::Executor(error) if error.is_fatal()),
            "{error:?}"
        );
        let kv = engine.kv_metrics().snapshot();
        // Neither a cold fallback nor a dropped anchor: both would free pages
        // the device may still be copying.
        assert_eq!(kv.prefix_cache_restore_failures, 0);
        assert_eq!(kv.prefix_cache_capture_failures, 0);
        assert!(kv.prefix_cache_anchors >= anchors.max(1));
        assert_eq!(engine.step(), Err(EngineError::PrefixStateUnconfirmed));
        assert_eq!(
            engine.cancel(RequestId::new(2)),
            Err(EngineError::PrefixStateUnconfirmed)
        );
        assert_eq!(
            engine.submit(request(3, &first, 1)),
            Err(EngineError::PrefixStateUnconfirmed)
        );
    }
}

/// Every request the scheduler admits reaches the executor's admission first,
/// including those that fit only once earlier hits restore and hand back the
/// pages they share.
#[test]
fn requests_admitted_after_restores_free_pages_are_admitted_by_the_executor() {
    // Ten pages of four tokens; each hit below may write three pages.
    let config = EngineConfig::new(
        SchedulerConfig::new(4, 16, 16, 8, 1).unwrap(),
        KvGeometry::new(4, 10).unwrap(),
        ModelVocabulary::new(VOCABULARY).unwrap(),
        40,
        4,
    )
    .unwrap()
    .with_prefix_cache(1);
    let mut engine = Engine::new(config, FoldingExecutor::default()).unwrap();
    let system = words(8, 3);
    run(&mut engine, 1, &system, 1);
    let mut prompts = Vec::new();
    for id in 2..6 {
        let mut prompt = system.clone();
        prompt.push(id as u32);
        engine.submit(request(id, &prompt, 4)).unwrap();
        prompts.push((id, prompt));
    }
    engine.run_until_idle(64).unwrap();
    engine.audit().unwrap();
    let mut tokens = BTreeMap::<u64, Vec<u32>>::new();
    for event in engine.drain_events() {
        if let EngineEventKind::Token { token_id, .. } = event.kind() {
            tokens
                .entry(event.request_id().get())
                .or_default()
                .push(*token_id);
        }
    }
    for (id, prompt) in prompts {
        assert_eq!(tokens[&id], cold(&prompt, 4), "request {id}");
    }
    assert_eq!(engine.executor().restores.len(), 4);
    assert!(engine.executor().admitted.is_empty());
}

#[test]
fn grouped_prefix_reuse_reduces_work_and_preserves_stateful_outputs() {
    for shared_suffix in [false, true] {
        let prompts: Vec<_> = (0..8)
            .map(|index| {
                let mut prompt = words(20, 0);
                prompt.push(if shared_suffix { 2 } else { 2 + index });
                prompt
            })
            .collect();
        let mut measurements = Vec::new();
        for group in [false, true] {
            let config = EngineConfig::new(
                SchedulerConfig::new(8, 16, 64, 8, 1).unwrap(),
                KvGeometry::new(4, 128).unwrap(),
                ModelVocabulary::new(VOCABULARY).unwrap(),
                128,
                8,
            )
            .unwrap()
            .with_prefix_cache(1);
            let mut engine = Engine::new(config, FoldingExecutor::default()).unwrap();
            let ids: Vec<_> = (1..=8).map(RequestId::new).collect();
            for (id, prompt) in ids.iter().zip(&prompts) {
                engine.submit(request(id.get(), prompt, 3)).unwrap();
            }
            if group {
                engine.share_prompt_prefixes(&ids).unwrap();
            }
            let mut tokens: BTreeMap<RequestId, Vec<u32>> = BTreeMap::new();
            let mut finished = 0;
            let mut peak = 0;
            for _ in 0..64 {
                engine.step().unwrap();
                peak = peak.max(engine.kv_stats().used_blocks());
                engine.audit().unwrap();
                for event in engine.drain_events() {
                    match event.kind() {
                        EngineEventKind::Token { token_id, .. } => tokens
                            .entry(event.request_id())
                            .or_default()
                            .push(*token_id),
                        EngineEventKind::Finished { .. } => finished += 1,
                        EngineEventKind::Error { message } => panic!("{message}"),
                        _ => {}
                    }
                }
                if finished == 8 {
                    break;
                }
            }
            assert_eq!(finished, 8);
            for (id, prompt) in ids.iter().zip(&prompts) {
                assert_eq!(tokens[id], cold(prompt, 3));
            }
            measurements.push((engine.executor().input_tokens, peak));
        }
        assert_eq!(measurements[0].0, 8 * 23);
        assert_eq!(measurements[1].0, 20 + 8 * 3);
        assert!(
            measurements[1].1 < measurements[0].1,
            "live KV peaks: {measurements:?}"
        );
    }
}

#[test]
fn grouped_followers_run_cold_after_leader_cancellation_or_capture_failure() {
    for cancel in [false, true] {
        let executor = FoldingExecutor {
            refuse_capture: !cancel,
            ..FoldingExecutor::default()
        };
        let mut engine = Engine::new(config(true), executor).unwrap();
        let prompt = words(21, 0);
        for id in 1..=3 {
            engine.submit(request(id, &prompt, 3)).unwrap();
        }
        engine
            .share_prompt_prefixes(&[RequestId::new(1), RequestId::new(2), RequestId::new(3)])
            .unwrap();
        engine.step().unwrap();
        if cancel {
            engine.cancel(RequestId::new(1)).unwrap();
        }
        engine.run_until_idle(64).unwrap();
        engine.audit().unwrap();
        let mut tokens: BTreeMap<RequestId, Vec<u32>> = BTreeMap::new();
        for event in engine.drain_events() {
            if let EngineEventKind::Token { token_id, .. } = event.kind() {
                tokens
                    .entry(event.request_id())
                    .or_default()
                    .push(*token_id);
            }
        }
        for id in 2..=3 {
            assert_eq!(tokens[&RequestId::new(id)], cold(&prompt, 3));
        }
        assert!(engine.executor().restores.is_empty());
    }
}

#[test]
fn grouped_prefixes_do_not_cross_cache_salts() {
    let mut engine = Engine::new(config(true), FoldingExecutor::default()).unwrap();
    let prompt = words(21, 0);
    engine
        .submit(request(1, &prompt, 1).with_cache_salt(vec![1]))
        .unwrap();
    engine
        .submit(request(2, &prompt, 1).with_cache_salt(vec![2]))
        .unwrap();
    engine
        .share_prompt_prefixes(&[RequestId::new(1), RequestId::new(2)])
        .unwrap();
    engine.run_until_idle(64).unwrap();
    assert_eq!(engine.executor().input_tokens, 42);
    assert!(engine.executor().restores.is_empty());
}
/// With in-place state, a request whose first sample fails on its final
/// prefill row commits its step, which queues its prompt capture, and then
/// fails alone. Its capture is dropped with it before its sequence is
/// released, so no capture reads a released sequence: the request
/// prefilled beside it completes, a later request with the failed prompt
/// prefills it again and matches a cold run, and no KV block stays in use.
/// Both a failed request with room left and one whose placeholder row
/// reaches its length (whose capture already owns its sequence and runs).
#[test]
fn a_sampling_failure_on_a_captured_final_prefill_fails_only_that_request() {
    for maximum in [4, 1] {
        let mut engine = Engine::new(
            config(true),
            FoldingExecutor {
                in_place: true,
                ..FoldingExecutor::default()
            },
        )
        .unwrap();
        let prompt = words(10, 0);
        // Every token stops and one token is required: no token samples.
        let failing = EngineRequest::new(
            RequestId::new(1),
            prompt.clone(),
            maximum,
            SamplingParameters::new(0.0)
                .unwrap()
                .with_minimum_new_tokens(1)
                .unwrap(),
            1,
            (0..VOCABULARY as u32).collect(),
            ModelVocabulary::new(VOCABULARY).unwrap(),
        )
        .unwrap();
        engine.submit(failing).unwrap();
        engine.submit(request(2, &words(9, 3), 3)).unwrap();
        // The prompt prefills in chunks; its final chunk's sample fails.
        let error = (0..8)
            .find_map(|_| engine.step().err())
            .expect("the final prefill's sample fails");
        assert!(
            matches!(error, EngineError::Sampling(_)),
            "maximum {maximum}: {error:?}"
        );
        engine.audit().unwrap();
        engine.run_until_idle(16).unwrap();
        engine.audit().unwrap();
        let mut tokens = BTreeMap::<u64, usize>::new();
        let mut errors = BTreeMap::<u64, usize>::new();
        for event in engine.drain_events() {
            match event.kind() {
                EngineEventKind::Token { .. } => {
                    *tokens.entry(event.request_id().get()).or_default() += 1;
                }
                EngineEventKind::Error { .. } => {
                    *errors.entry(event.request_id().get()).or_default() += 1;
                }
                _ => {}
            }
        }
        assert_eq!(errors, BTreeMap::from([(1, 1)]), "maximum {maximum}");
        assert_eq!(tokens, BTreeMap::from([(2, 3)]), "maximum {maximum}");
        // Only a request its placeholder completed keeps its capture.
        let captured = engine
            .executor()
            .captures
            .contains(&(RequestId::new(1), 10));
        assert_eq!(captured, maximum == 1, "maximum {maximum}");
        assert!(!engine.executor().admitted.contains(&RequestId::new(1)));
        let mut longer = prompt.clone();
        longer.extend(words(5, 7));
        let (resumed, cached) = run(&mut engine, 3, &longer, 3);
        // The final chunk's start was captured before the failure.
        assert_eq!(
            cached,
            Some(if captured { 10 } else { 8 }),
            "maximum {maximum}"
        );
        assert_eq!(resumed, cold(&longer, 3));
        assert_eq!(engine.kv_stats().used_blocks(), 0);
    }
}

#[test]
fn a_prompt_ending_chunk_keeps_its_anchor_inside_one_step() {
    let config = |anchors| {
        EngineConfig::new(
            SchedulerConfig::new(4, 32, 32, 16, 1)
                .unwrap()
                .with_prefill_granule(ANCHOR_GRANULE)
                .unwrap()
                .with_anchor_items(anchors),
            KvGeometry::new(4, 64).unwrap(),
            ModelVocabulary::new(VOCABULARY).unwrap(),
            128,
            4,
        )
        .unwrap()
        .with_prefix_cache(1)
    };
    let first = words(13, 2);
    let mut second = first.clone();
    second.extend(words(9, 6));
    let mut steps = Vec::new();
    for anchors in [0, 1] {
        let mut engine = Engine::new(config(anchors), FoldingExecutor::default()).unwrap();
        let (_, cached) = run(&mut engine, 1, &first, 2);
        assert_eq!(cached, None);
        // The anchor at 8 is captured either way: from the end of a chunk,
        // or from the state the 13-token chunk kept at 8.
        assert_eq!(engine.executor().captures, [(RequestId::new(1), 8)]);
        assert_eq!(engine.executor().anchor_items, anchors);
        steps.push(engine.executor().executions);
        let (tokens, cached) = run(&mut engine, 2, &second, 3);
        assert_eq!(cached, Some(8));
        assert_eq!(engine.executor().restores, [(RequestId::new(2), 8)]);
        assert_eq!(tokens, cold(&second, 3));
    }
    assert_eq!(
        steps[1] + 1,
        steps[0],
        "the kept anchor saves one prefill step"
    );
}
