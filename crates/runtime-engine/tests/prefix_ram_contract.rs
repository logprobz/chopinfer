//! The engine's RAM prefix tier: demotions after captures under the
//! measured headroom, promotions that admission waits for, RAM hits, the
//! cancellation triggers of design amendment A8, forced mode and metrics.

use std::collections::{BTreeMap, BTreeSet};

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use runtime_engine::{
    Engine, EngineConfig, EngineError, EngineEventKind, EngineMetrics, EngineRequest,
    ExecutorError, KvOffloadInfo, KvTransferPoll, KvTransferStatus, KvTransferSubmitResult,
    LogitsRow, ModelBatchItem, ModelExecutor, ModelTransaction, ModelTransactionState,
    MonotonicClock, PrefixRamStatusV1, PrefixRamUnavailableV1, RequestId, StepStatus,
};
use runtime_kv::{
    BlockId, HostPageId, KvGeometry, KvMetrics, KvTransferClass, KvTransferDirection, KvTransferId,
    KvTransferPlan, KvTransferSubmission, PreparedAppendBatch,
};
use runtime_model_api::ModelVocabulary;
use runtime_sampling::SamplingParameters;
use runtime_scheduler::{PrefixRamConfig, SchedulerConfig};

const BLOCK: usize = 4;
const VOCABULARY: usize = 64;
/// Bytes the fake executor's host pages hold.
const PAGE_BYTES: u64 = 100;

fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Logits that depend on every token of the history.
fn logits_for(history: &[u32]) -> Vec<f32> {
    let hash = history
        .iter()
        .fold(0x1234_5678_u64, |hash, token| mix(hash ^ u64::from(*token)));
    (0..VOCABULARY as u64)
        .map(|token| (mix(hash ^ (token << 40)) >> 40) as f32 / (1u64 << 24) as f32)
        .collect()
}

/// A prompt of `length` tokens; prompts of different salts share no page.
fn prompt(salt: u32, length: usize) -> Vec<u32> {
    (0..length as u64)
        .map(|position| {
            if position == 0 {
                salt
            } else {
                (mix(u64::from(salt) * 1_000 + position) % VOCABULARY as u64) as u32
            }
        })
        .collect()
}

/// `base` followed by `extra` tokens of its own.
fn extended(base: &[u32], extra: usize, tail: u32) -> Vec<u32> {
    let mut tokens = base.to_vec();
    tokens.extend(prompt(tail, extra));
    tokens
}

#[derive(Debug)]
struct Copy {
    submission: KvTransferSubmission,
    polls: usize,
    cancelled: bool,
}

/// An executor whose per-request state is its committed tokens, and whose
/// snapshot page holds that state. Copies move snapshot contents between
/// device pages and host pages, so a restore after a promotion reads the
/// bytes the demotion copied out.
#[derive(Debug, Default)]
struct Model {
    offload: bool,
    cancel: bool,
    host_pages: usize,
    state: BTreeMap<RequestId, Vec<u32>>,
    pending: Vec<(RequestId, Vec<u32>)>,
    admitted: BTreeSet<RequestId>,
    snapshots: BTreeMap<BlockId, Vec<u32>>,
    host: BTreeMap<HostPageId, Vec<u32>>,
    copies: BTreeMap<KvTransferId, Copy>,
    /// Classes whose copies stay pending until cancelled.
    held: Vec<KvTransferClass>,
    /// Copies of these classes and page counts stay pending until
    /// cancelled too.
    held_sizes: Vec<(KvTransferClass, usize)>,
    /// Polls each copy stays pending first.
    delay: usize,
    /// The terminal status of each class's copies; complete otherwise.
    status: Vec<(KvTransferClass, KvTransferStatus)>,
    /// The status a cancelled copy reports.
    cancelled_status: Option<KvTransferStatus>,
    /// Cancelled copies stay pending while set: their issued chunks never
    /// drain.
    unsettled: bool,
    /// The drain timeout the executor reports; five seconds otherwise.
    drain_timeout: Option<Duration>,
    refused: Vec<KvTransferClass>,
    submitted: Vec<(KvTransferClass, usize)>,
    /// The class and pages of each copy that completed.
    completed: Vec<(KvTransferClass, usize)>,
    /// The most demotions pending at once.
    most_demotions: usize,
    cancels: Vec<KvTransferId>,
    captures: Vec<(RequestId, usize)>,
    /// Each restore's request and the snapshot it read.
    restores: Vec<(RequestId, Vec<u32>)>,
}

fn model() -> Model {
    Model {
        offload: true,
        cancel: true,
        host_pages: 64,
        delay: 1,
        ..Model::default()
    }
}

struct Pending<'a> {
    model: &'a mut Model,
}

impl ModelTransactionState for Pending<'_> {
    fn commit(&mut self) -> Result<(), ExecutorError> {
        for (id, inputs) in self.model.pending.drain(..) {
            self.model.state.entry(id).or_default().extend(inputs);
        }
        Ok(())
    }

    fn rollback(&mut self) -> Result<(), ExecutorError> {
        self.model.pending.clear();
        Ok(())
    }
}

impl Model {
    fn submit(&mut self, plan: KvTransferPlan) -> KvTransferSubmitResult {
        let class = plan.class();
        if self.refused.contains(&class) {
            return Err((ExecutorError::new("refused copy").unwrap(), plan));
        }
        self.submitted.push((class, plan.pages().len()));
        self.copies.insert(
            plan.id(),
            Copy {
                submission: plan.into_submission(),
                polls: 0,
                cancelled: false,
            },
        );
        self.most_demotions = self
            .most_demotions
            .max(self.pending_copies(KvTransferClass::Demotion));
        Ok(())
    }

    fn pending_copies(&self, class: KvTransferClass) -> usize {
        self.copies
            .values()
            .filter(|copy| copy.submission.class() == class)
            .count()
    }
}

impl ModelExecutor for Model {
    type TransactionState<'a> = Pending<'a>;

    fn kv_offload_info_v1(&self) -> Option<KvOffloadInfo> {
        self.offload.then_some(KvOffloadInfo {
            host_page_capacity: self.host_pages,
            minimum_tokens: 1,
            snapshot_pages: 1,
        })
    }

    fn offload_pages_v1(&mut self, plan: KvTransferPlan) -> KvTransferSubmitResult {
        self.submit(plan)
    }

    fn reload_pages_v1(&mut self, plan: KvTransferPlan) -> KvTransferSubmitResult {
        self.submit(plan)
    }

    fn poll_kv_transfer_v1(&mut self, id: KvTransferId) -> Result<KvTransferPoll, ExecutorError> {
        let copy = self
            .copies
            .get_mut(&id)
            .expect("only submitted copies are polled");
        let class = copy.submission.class();
        if copy.cancelled {
            if self.unsettled {
                return Ok(KvTransferPoll::Pending);
            }
            let status = self
                .cancelled_status
                .unwrap_or(KvTransferStatus::FailedIdle);
            let completion = copy.submission.completion(status);
            if status != KvTransferStatus::Unconfirmed {
                self.copies.remove(&id);
            }
            return Ok(KvTransferPoll::Finished(completion));
        }
        if self.held.contains(&class)
            || self
                .held_sizes
                .contains(&(class, copy.submission.pages().len()))
        {
            return Ok(KvTransferPoll::Pending);
        }
        if copy.polls < self.delay {
            copy.polls += 1;
            return Ok(KvTransferPoll::Pending);
        }
        let status = self
            .status
            .iter()
            .find(|(held, _)| *held == class)
            .map_or(KvTransferStatus::Complete, |(_, status)| *status);
        if status == KvTransferStatus::Complete {
            for pair in copy.submission.pages() {
                match copy.submission.direction() {
                    KvTransferDirection::Out => {
                        let bytes = self.snapshots.get(&pair.device).cloned();
                        self.host.insert(pair.host, bytes.unwrap_or_default());
                    }
                    KvTransferDirection::In => {
                        let bytes = self.host[&pair.host].clone();
                        self.snapshots.insert(pair.device, bytes);
                    }
                }
            }
        }
        if status == KvTransferStatus::Complete {
            self.completed.push((class, copy.submission.pages().len()));
        }
        let completion = copy.submission.completion(status);
        if status != KvTransferStatus::Unconfirmed {
            self.copies.remove(&id);
        }
        Ok(KvTransferPoll::Finished(completion))
    }

    fn kv_offload_page_bytes_v1(&self) -> u64 {
        PAGE_BYTES
    }

    fn kv_transfer_drain_timeout_v1(&self) -> Duration {
        self.drain_timeout.unwrap_or(Duration::from_secs(5))
    }

    fn kv_transfer_cancel_v1_supported(&self) -> bool {
        self.cancel
    }

    fn cancel_kv_transfer_v1(&mut self, id: KvTransferId) -> Result<bool, ExecutorError> {
        assert!(self.cancel, "cancel needs support");
        self.cancels.push(id);
        match self.copies.get_mut(&id) {
            Some(copy) if copy.submission.class() == KvTransferClass::Demotion => {
                copy.cancelled = true;
                Ok(true)
            }
            Some(_) => Err(ExecutorError::new("only demotions cancel").unwrap()),
            None => Ok(false),
        }
    }

    fn admit_request_v1(&mut self, id: RequestId) -> Result<(), ExecutorError> {
        assert!(self.admitted.insert(id), "admitted once until retired");
        Ok(())
    }

    fn retire_request_v1(&mut self, id: RequestId) -> Result<(), ExecutorError> {
        self.admitted.remove(&id);
        self.state.remove(&id);
        Ok(())
    }

    fn capture_prefix_state_v1(
        &mut self,
        id: RequestId,
        tokens: usize,
        pages: &[BlockId],
    ) -> Result<(), ExecutorError> {
        let state = self.state[&id].clone();
        assert_eq!(state.len(), tokens, "a capture names the committed length");
        self.snapshots.insert(pages[0], state);
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
        assert!(!self.state.contains_key(&id), "restore precedes execution");
        let snapshot = self.snapshots[&pages[0]].clone();
        assert_eq!(snapshot, prefix, "a restore reads the prefix's state");
        self.state.insert(id, prefix.to_vec());
        self.restores.push((id, snapshot));
        Ok(())
    }

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        kv_plan: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        let mut rows = Vec::new();
        for (item, append) in items.iter().zip(kv_plan.appends()) {
            let id = item.request_id();
            assert!(self.admitted.contains(&id));
            let mut history = self.state.get(&id).cloned().unwrap_or_default();
            assert_eq!(history.len(), append.previous_tokens(), "state matches KV");
            history.extend_from_slice(item.input_tokens());
            if item.expects_logits() {
                rows.push(LogitsRow::new(id, logits_for(&history)).unwrap());
            }
            self.pending.push((id, item.input_tokens().to_vec()));
        }
        Ok(ModelTransaction::new(rows, Pending { model: self }))
    }
}

fn tier(forced: bool, budget: usize) -> PrefixRamConfig {
    PrefixRamConfig {
        forced,
        promotion_budget_pages: budget,
    }
}

/// Four sequences, a token budget and a prefill chunk, one snapshot page
/// per anchor.
fn config_with(
    pages: usize,
    budget: usize,
    chunk: usize,
    tier: Option<PrefixRamConfig>,
) -> EngineConfig {
    config_sequences(pages, 4, budget, chunk, tier)
}

fn config_sequences(
    pages: usize,
    sequences: usize,
    budget: usize,
    chunk: usize,
    tier: Option<PrefixRamConfig>,
) -> EngineConfig {
    let config = EngineConfig::new(
        SchedulerConfig::new(sequences, 64, budget, chunk, 1).unwrap(),
        KvGeometry::new(BLOCK, pages).unwrap(),
        ModelVocabulary::new(VOCABULARY).unwrap(),
        (pages * BLOCK).min(320),
        64,
    )
    .unwrap()
    .with_prefix_cache(1);
    match tier {
        Some(tier) => config.with_prefix_ram(tier),
        None => config,
    }
}

fn config(pages: usize, tier: Option<PrefixRamConfig>) -> EngineConfig {
    config_with(pages, 16, 8, tier)
}

#[derive(Debug)]
struct FixedClock;

impl MonotonicClock for FixedClock {
    fn now_micros(&self) -> u64 {
        1_000
    }
}

/// A clock that moves a millisecond every time it is read.
#[derive(Debug, Default)]
struct TickingClock(AtomicU64);

impl MonotonicClock for TickingClock {
    fn now_micros(&self) -> u64 {
        self.0.fetch_add(1_000, Ordering::Relaxed)
    }
}

fn engine_with_clock(
    config: EngineConfig,
    model: Model,
    clock: Arc<dyn MonotonicClock>,
) -> Engine<Model> {
    Engine::new_with_runtime_metrics(
        config,
        model,
        clock,
        EngineMetrics::default(),
        KvMetrics::default(),
    )
    .unwrap()
}

fn engine_from(config: EngineConfig, model: Model) -> Engine<Model> {
    engine_with_clock(config, model, Arc::new(FixedClock))
}

fn engine(pages: usize, tier: Option<PrefixRamConfig>, model: Model) -> Engine<Model> {
    engine_from(config(pages, tier), model)
}

fn request(id: u64, prompt: &[u32], new_tokens: usize) -> EngineRequest {
    EngineRequest::new(
        RequestId::new(id),
        prompt.to_vec(),
        new_tokens,
        SamplingParameters::new(0.0).unwrap(),
        id,
        Vec::new(),
        ModelVocabulary::new(VOCABULARY).unwrap(),
    )
    .unwrap()
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Output {
    cached: Option<usize>,
    tokens: Vec<u32>,
    finished: bool,
}

type Outputs = BTreeMap<RequestId, Output>;

/// One step, audited when it succeeds (both tiers and the transfer
/// ledger), with its events added to `outputs`.
fn try_step(engine: &mut Engine<Model>, outputs: &mut Outputs) -> Result<StepStatus, EngineError> {
    let status = engine.step();
    if status.is_ok() {
        engine.audit().unwrap();
    }
    for event in engine.drain_events() {
        let output = outputs.entry(event.request_id()).or_default();
        assert!(!output.finished, "nothing follows the finish");
        match event.kind() {
            EngineEventKind::PromptCached { tokens } => {
                assert!(output.cached.replace(*tokens).is_none(), "reported once");
                assert!(output.tokens.is_empty(), "cached tokens precede tokens");
            }
            EngineEventKind::Token { token_id, .. } => output.tokens.push(*token_id),
            EngineEventKind::Finished { .. } => output.finished = true,
            kind => panic!("unexpected event {kind:?}"),
        }
    }
    status
}

fn step(engine: &mut Engine<Model>, outputs: &mut Outputs) -> StepStatus {
    try_step(engine, outputs).unwrap()
}

/// A request's id, prompt and new tokens.
type Spec<'a> = (u64, &'a [u32], usize);

fn submit(engine: &mut Engine<Model>, requests: &[Spec<'_>]) {
    for (id, prompt, new_tokens) in requests {
        engine.submit(request(*id, prompt, *new_tokens)).unwrap();
    }
}

fn finished(outputs: &Outputs, requests: &[Spec<'_>]) -> bool {
    requests.iter().all(|(id, _, _)| {
        outputs
            .get(&RequestId::new(*id))
            .is_some_and(|output| output.finished)
    })
}

/// Submits the requests and steps until each has finished.
fn try_run(engine: &mut Engine<Model>, requests: &[Spec<'_>]) -> Result<Outputs, EngineError> {
    submit(engine, requests);
    let mut outputs = Outputs::new();
    for _ in 0..500 {
        try_step(engine, &mut outputs)?;
        if finished(&outputs, requests) {
            return Ok(outputs);
        }
    }
    panic!("the requests did not finish: {outputs:?}");
}

fn run(engine: &mut Engine<Model>, requests: &[Spec<'_>]) -> Outputs {
    try_run(engine, requests).unwrap()
}

/// Steps until no copy is pending.
fn settle(engine: &mut Engine<Model>) {
    let mut outputs = Outputs::new();
    for _ in 0..100 {
        if !engine.poll_idle_kv_transfers().unwrap() {
            engine.audit().unwrap();
            return;
        }
        step(engine, &mut outputs);
    }
    panic!("copies stayed pending");
}

fn prometheus(engine: &Engine<Model>) -> String {
    let mut text = engine.metrics().snapshot().prometheus();
    text.push_str(&engine.kv_metrics().snapshot().prometheus());
    text
}

/// The value of one sample, `name` with its labels.
fn metric(text: &str, sample: &str) -> u64 {
    text.lines()
        .find_map(|line| line.strip_prefix(sample)?.strip_prefix(' '))
        .unwrap_or_else(|| panic!("missing {sample}"))
        .parse()
        .unwrap()
}

fn demotions(text: &str, result: &str) -> u64 {
    metric(
        text,
        &format!("chop_prefix_ram_demotions_total{{result=\"{result}\"}}"),
    )
}

fn promotions(text: &str, result: &str) -> u64 {
    metric(
        text,
        &format!("chop_prefix_ram_promotions_total{{result=\"{result}\"}}"),
    )
}

const MISS_REASONS: [&str; 7] = [
    "headroom",
    "in_flight",
    "tier_full",
    "failed",
    "below_minimum",
    "not_frontier",
    "unplanned",
];

fn missed(text: &str, reason: &str) -> u64 {
    metric(
        text,
        &format!("chop_prefix_ram_missed_evictions_total{{reason=\"{reason}\"}}"),
    )
}

fn fallbacks(text: &str, reason: &str) -> u64 {
    metric(
        text,
        &format!("chop_prefix_ram_fallbacks_total{{reason=\"{reason}\"}}"),
    )
}

fn submitted(engine: &Engine<Model>, class: KvTransferClass) -> usize {
    engine
        .executor()
        .submitted
        .iter()
        .filter(|(submitted, _)| *submitted == class)
        .count()
}

fn restored(engine: &Engine<Model>, id: u64) -> Vec<Vec<u32>> {
    engine
        .executor()
        .restores
        .iter()
        .filter(|(restored, _)| *restored == RequestId::new(id))
        .map(|(_, bytes)| bytes.clone())
        .collect()
}

/// A, its 16-token anchor demoted, then evicted by a filler on a 24-page
/// pool; B extends A's prompt, so its ready host match is 16 tokens and
/// its GPU match nothing.
fn evicted_after_demotion(model: Model) -> (Engine<Model>, Vec<u32>, Vec<u32>) {
    let a = prompt(1, 16);
    let b = extended(&a, 4, 2);
    let mut engine = engine(24, Some(tier(false, 16)), model);
    run(&mut engine, &[(1, &a, 1)]);
    settle(&mut engine);
    assert_eq!(demotions(&prometheus(&engine), "ok"), 1);
    run(&mut engine, &[(3, &prompt(3, 80), 1)]);
    settle(&mut engine);
    (engine, a, b)
}

#[test]
fn a_ram_hit_restores_the_snapshot_a_gpu_hit_restores_and_is_counted() {
    // A large pool keeps A's anchor: B resumes from it on the GPU.
    let a = prompt(1, 16);
    let b = extended(&a, 4, 2);
    let mut gpu = engine(64, Some(tier(false, 16)), model());
    run(&mut gpu, &[(1, &a, 1)]);
    settle(&mut gpu);
    let gpu_outputs = run(&mut gpu, &[(2, &b, 4)]);
    assert_eq!(gpu_outputs[&RequestId::new(2)].cached, Some(16));
    let text = prometheus(&gpu);
    assert_eq!(metric(&text, "chop_prefix_cache_ram_hits_total"), 0);
    assert_eq!(promotions(&text, "ok"), 0);
    // A small pool loses A's GPU anchor, so B resumes from its host entry.
    let (mut ram, _, _) = evicted_after_demotion(model());
    let before = prometheus(&ram);
    // A's entry is the tier's one anchor.
    assert_eq!(metric(&before, "chop_prefix_ram_anchors"), 1);
    assert_eq!(metric(&before, "chop_prefix_ram_bytes"), 5 * PAGE_BYTES);
    let ram_outputs = run(&mut ram, &[(2, &b, 4)]);
    assert_eq!(ram_outputs, gpu_outputs);
    assert_eq!(restored(&ram, 2), std::slice::from_ref(&a));
    assert_eq!(restored(&ram, 2), restored(&gpu, 2));
    assert_eq!(submitted(&ram, KvTransferClass::Promotion), 1);
    let text = prometheus(&ram);
    let delta = |sample: &str| metric(&text, sample) - metric(&before, sample);
    assert_eq!(delta("chop_prefix_cache_hits_total"), 1);
    assert_eq!(delta("chop_prefix_cache_reused_tokens_total"), 16);
    assert_eq!(delta("chop_prefix_cache_ram_hits_total"), 1);
    assert_eq!(delta("chop_prefix_cache_ram_reused_tokens_total"), 16);
    assert_eq!(promotions(&text, "ok"), 1);
    assert_eq!(delta("chop_prefix_ram_restore_seconds_count"), 1);
    assert_eq!(metric(&text, "chop_prefix_ram_headroom_waits_total"), 0);
    // Bytes count the completed copies' pages: B's promotion moved A's
    // five pages in.
    for (direction, class) in [
        ("out", KvTransferClass::Demotion),
        ("in", KvTransferClass::Promotion),
    ] {
        let pages: usize = ram
            .executor()
            .completed
            .iter()
            .filter(|(completed, _)| *completed == class)
            .map(|(_, pages)| pages)
            .sum();
        let sample =
            format!("chop_kv_offload_bytes_total{{direction=\"{direction}\",kind=\"prefix\"}}");
        assert_eq!(metric(&text, &sample), pages as u64 * PAGE_BYTES);
    }
    assert_eq!(
        ram.executor()
            .completed
            .iter()
            .filter(|(class, _)| *class == KvTransferClass::Promotion)
            .collect::<Vec<_>>(),
        [&(KvTransferClass::Promotion, 5)]
    );
    assert_eq!(
        metric(
            &text,
            "chop_kv_offload_bytes_total{direction=\"out\",kind=\"swap\"}"
        ),
        0
    );
    assert_eq!(
        metric(
            &text,
            "chop_kv_offload_copy_seconds_count{direction=\"in\"}"
        ),
        1
    );
    assert_eq!(metric(&text, "chop_kv_offload_swap_in_total"), 0);
    // Three evictions found no host copy: the anchor at A's final chunk
    // start, which A's end superseded, when the filler evicted it, and the
    // filler's two anchors when B's promotion evicted them. The filler's end
    // was too large for the start rule; its final chunk start is no
    // frontier.
    let misses: Vec<_> = MISS_REASONS
        .iter()
        .map(|reason| (*reason, missed(&text, reason)))
        .collect();
    assert_eq!(
        misses,
        [
            ("headroom", 1),
            ("in_flight", 0),
            ("tier_full", 0),
            ("failed", 0),
            ("below_minimum", 0),
            ("not_frontier", 2),
            ("unplanned", 0)
        ]
    );
}

#[test]
fn a_failed_or_refused_promotion_falls_back_to_the_gpu_match_or_cold() {
    // A poisoned copy queue fails or refuses every promotion. With a
    // shorter GPU anchor (A1, A's first 8 tokens) B falls back to it, and
    // without one it runs cold.
    for refused in [false, true] {
        for shorter in [true, false] {
            let a = prompt(1, 16);
            let a1 = a[..8].to_vec();
            let b = extended(&a, 4, 2);
            let filler = prompt(3, 80);
            let mut requests: Vec<Spec<'_>> = vec![(1, &a, 1), (3, &filler, 1)];
            if shorter {
                requests.push((4, &a1, 1));
            }
            let workload = |engine: &mut Engine<Model>| {
                for request in &requests {
                    run(engine, &[*request]);
                    settle(engine);
                }
                run(engine, &[(2, &b, 4)])
            };
            let mut reference = engine(24, None, model());
            let expected = workload(&mut reference);
            let mut model = model();
            if refused {
                model.refused = vec![KvTransferClass::Promotion];
            } else {
                model.status = vec![(KvTransferClass::Promotion, KvTransferStatus::FailedIdle)];
            }
            let mut ram = engine(24, Some(tier(false, 16)), model);
            let outputs = workload(&mut ram);
            assert_eq!(outputs, expected, "refused {refused}, shorter {shorter}");
            assert_eq!(outputs[&RequestId::new(2)].cached, shorter.then_some(8));
            let text = prometheus(&ram);
            assert_eq!(promotions(&text, "failed"), 1);
            assert_eq!(promotions(&text, "ok"), 0);
            assert_eq!(fallbacks(&text, "failed"), 1);
            assert_eq!(metric(&text, "chop_prefix_cache_ram_hits_total"), 0);
            assert_eq!(metric(&text, "chop_prefix_ram_restore_seconds_count"), 0);
        }
    }
}

#[test]
fn forced_mode_turns_every_hit_into_a_ram_hit() {
    let a = prompt(1, 16);
    let followers = [extended(&a, 4, 2), extended(&a, 6, 3), extended(&a, 4, 2)];
    for settled in [true, false] {
        let workload = |engine: &mut Engine<Model>| {
            let mut outputs = run(engine, &[(1, &a, 1)]);
            for (index, prompt) in followers.iter().enumerate() {
                // Without settling, a follower finds the demotion still
                // copying and waits for it.
                if settled {
                    settle(engine);
                }
                outputs.extend(run(engine, &[(2 + index as u64, prompt, 3)]));
            }
            outputs
        };
        let mut gpu = engine(64, None, model());
        let gpu_outputs = workload(&mut gpu);
        let mut forced = engine(
            64,
            Some(tier(true, 16)),
            Model {
                delay: 3,
                ..model()
            },
        );
        assert_eq!(
            forced.prefix_ram_status(),
            PrefixRamStatusV1::Active { forced: true }
        );
        let forced_outputs = workload(&mut forced);
        assert_eq!(forced_outputs, gpu_outputs, "settled {settled}");
        let hits = gpu.kv_metrics().snapshot().prefix_cache_hits;
        assert_eq!(hits, 3);
        let kv = forced.kv_metrics().snapshot();
        assert_eq!(kv.prefix_cache_hits, hits);
        assert_eq!(kv.prefix_cache_ram_hits, hits);
        assert_eq!(kv.prefix_cache_ram_reused_tokens, 3 * 16);
        assert_eq!(submitted(&forced, KvTransferClass::Promotion) as u64, hits);
        // Every public capture was demoted: A's final chunk start and end,
        // and each follower's end.
        assert_eq!(
            submitted(&forced, KvTransferClass::Demotion),
            forced.executor().captures.len(),
            "settled {settled}: {:?}",
            forced.executor().submitted
        );
        assert_eq!(forced.executor().captures.len(), 5);
        let text = prometheus(&forced);
        assert_eq!(promotions(&text, "ok"), hits);
        assert_eq!(metric(&text, "chop_prefix_ram_restore_seconds_count"), hits);
    }
}

#[test]
fn a_forced_recapture_is_demoted_once_its_older_copy_ends() {
    let a = prompt(1, 16);
    let b = extended(&a, 4, 2);
    let workload = |engine: &mut Engine<Model>, release: &dyn Fn(&mut Engine<Model>)| {
        let mut outputs = run(engine, &[(1, &a, 1)]);
        outputs.extend(run(engine, &[(2, &a, 1)]));
        submit(engine, &[(3, &b, 3)]);
        let mut waiting = Outputs::new();
        for _ in 0..10 {
            step(engine, &mut waiting);
        }
        let waited = !waiting.contains_key(&RequestId::new(3));
        release(engine);
        while !finished(&waiting, &[(3, &b, 3)]) {
            step(engine, &mut waiting);
        }
        outputs.extend(waiting);
        (outputs, waited)
    };
    let mut gpu = engine(64, None, model());
    let (gpu_outputs, _) = workload(&mut gpu, &|_| {});
    assert_eq!(gpu_outputs[&RequestId::new(3)].cached, Some(16));
    // A's five-page demotion of its end stays copying, so A2, which
    // resumes from A's eight-token entry, recaptures A's end while it
    // copies.
    let mut forced = engine(
        64,
        Some(tier(true, 16)),
        Model {
            held_sizes: vec![(KvTransferClass::Demotion, 5)],
            ..model()
        },
    );
    let (outputs, waited) = workload(&mut forced, &|engine| {
        assert_eq!(
            engine.executor().pending_copies(KvTransferClass::Demotion),
            1
        );
        engine.executor_mut().held_sizes.clear();
    });
    // B resumed from no GPU anchor of A2's recapture: it waited for its
    // demotion, which started once the older copy ended.
    assert!(waited);
    assert_eq!(outputs, gpu_outputs);
    let kv = forced.kv_metrics().snapshot();
    assert_eq!(
        kv.prefix_cache_hits,
        gpu.kv_metrics().snapshot().prefix_cache_hits
    );
    assert_eq!(kv.prefix_cache_ram_hits, kv.prefix_cache_hits);
    assert_eq!(restored(&forced, 3), std::slice::from_ref(&a));
}

/// A pool of 12 pages whose only anchor, X's three pages, is under a
/// demotion that never finishes unless cancelled.
fn held_demotion(model: Model) -> Engine<Model> {
    let mut engine = engine(
        12,
        Some(tier(false, 16)),
        Model {
            held: vec![KvTransferClass::Demotion],
            ..model
        },
    );
    run(&mut engine, &[(1, &prompt(1, 8), 1)]);
    assert_eq!(
        engine.executor().pending_copies(KvTransferClass::Demotion),
        1
    );
    engine
}

/// The counters a cancel for headroom moves: one wait, one cancelled
/// demotion that is not a failure, and the anchor's eviction a headroom
/// miss.
fn assert_cancelled_for_headroom(engine: &Engine<Model>) {
    let cancels = &engine.executor().cancels;
    assert_eq!(cancels.len(), 1);
    assert!(!engine.executor().copies.contains_key(&cancels[0]));
    let text = prometheus(engine);
    assert_eq!(metric(&text, "chop_prefix_ram_headroom_waits_total"), 1);
    assert_eq!(demotions(&text, "cancelled"), 1);
    assert_eq!(demotions(&text, "failed"), 0);
    assert_eq!(missed(&text, "headroom"), 1);
    assert_eq!(
        metric(
            &text,
            "chop_kv_offload_fallback_total{reason=\"out_failed\"}"
        ),
        0
    );
}

#[test]
fn a_capture_without_pages_cancels_the_demotion_and_retries_once() {
    let mut engine = held_demotion(model());
    // R's nine pages fill the pool, so its snapshot needs X's pages.
    run(&mut engine, &[(2, &prompt(2, 36), 1)]);
    assert!(
        engine
            .executor()
            .captures
            .contains(&(RequestId::new(2), 36))
    );
    assert_cancelled_for_headroom(&engine);
}

#[test]
fn a_batch_without_pages_cancels_the_demotion_instead_of_waiting() {
    let mut engine = held_demotion(model());
    // R's tenth page needs X's pages.
    run(&mut engine, &[(2, &prompt(2, 40), 1)]);
    assert_cancelled_for_headroom(&engine);
}

#[test]
fn a_cancel_still_pending_at_the_drain_timeout_frees_no_pages() {
    // X's cancelled demotion never drains within the one-millisecond drain
    // timeout, and the clock passes its two-millisecond stall deadline.
    let mut engine = engine_with_clock(
        config(12, Some(tier(false, 16))),
        Model {
            held: vec![KvTransferClass::Demotion],
            unsettled: true,
            drain_timeout: Some(Duration::from_millis(1)),
            ..model()
        },
        Arc::new(TickingClock::default()),
    );
    run(&mut engine, &[(1, &prompt(1, 8), 1)]);
    // R's snapshot needs X's pages: R finishes, and its capture falls back
    // as one that found no pages, without starting again.
    run(&mut engine, &[(2, &prompt(2, 36), 1)]);
    assert!(
        !engine
            .executor()
            .captures
            .contains(&(RequestId::new(2), 36))
    );
    let cancels = engine.executor().cancels.clone();
    assert_eq!(cancels.len(), 1);
    // The copy keeps its leases and its ledger entry, which every step's
    // audit checks, and is polled on.
    assert!(engine.executor().copies.contains_key(&cancels[0]));
    let text = prometheus(&engine);
    assert_eq!(metric(&text, "chop_prefix_ram_cancel_timeouts_total"), 1);
    assert_eq!(metric(&text, "chop_prefix_ram_headroom_waits_total"), 1);
    assert_eq!(demotions(&text, "cancelled"), 0);
    // Like any overdue copy, it marks the engine's copies stalled.
    assert!(engine.has_stalled_kv_transfers());
    assert_eq!(metric(&text, "chop_kv_offload_stalled_transfers"), 1);
    // A batch that needs its pages waits as for any pending copy, and
    // cancels nothing more.
    submit(&mut engine, &[(3, &prompt(3, 40), 1)]);
    let mut outputs = Outputs::new();
    for _ in 0..20 {
        assert_eq!(step(&mut engine, &mut outputs), StepStatus::Progressed);
    }
    assert!(!outputs.contains_key(&RequestId::new(3)));
    assert_eq!(engine.executor().cancels, cancels);
    // Once its issued chunks drain, the cancel settles and the batch runs.
    engine.executor_mut().unsettled = false;
    while !finished(&outputs, &[(3, &[], 1)]) {
        step(&mut engine, &mut outputs);
    }
    assert!(!engine.executor().copies.contains_key(&cancels[0]));
    assert!(!engine.has_stalled_kv_transfers());
    let text = prometheus(&engine);
    assert_eq!(demotions(&text, "cancelled"), 1);
    assert_eq!(demotions(&text, "failed"), 0);
    assert_eq!(metric(&text, "chop_prefix_ram_cancel_timeouts_total"), 1);
    assert_eq!(metric(&text, "chop_prefix_ram_headroom_waits_total"), 1);
}

/// H's 16-token anchor demoted on a 33-page pool with a 24-token prefill
/// chunk, then X's anchor under a demotion that never finishes unless
/// cancelled, and R, alive, whose 27 pages and two snapshots (its final
/// chunk's start and its end) evicted H's GPU anchor and leave two pages
/// free: short of a five-page promotion even with R's snapshots evicted.
/// Returns B, which extends H's prompt and so prefers H's host entry.
fn tight_pool_with_held_demotion(budget: usize) -> (Engine<Model>, Vec<u32>) {
    let h = prompt(1, 16);
    let mut engine = engine_from(config_with(33, 32, 24, Some(tier(false, budget))), model());
    run(&mut engine, &[(1, &h, 1)]);
    settle(&mut engine);
    engine.executor_mut().held = vec![KvTransferClass::Demotion];
    run(&mut engine, &[(2, &prompt(2, 4), 1)]);
    assert_eq!(
        engine.executor().pending_copies(KvTransferClass::Demotion),
        1
    );
    let r = prompt(3, 105);
    submit(&mut engine, &[(3, &r, 2)]);
    let mut outputs = Outputs::new();
    while !engine
        .executor()
        .captures
        .contains(&(RequestId::new(3), 105))
    {
        step(&mut engine, &mut outputs);
    }
    assert!(!outputs.get(&RequestId::new(3)).is_some_and(|r| r.finished));
    assert_eq!(engine.kv_stats().free_blocks(), 2);
    (engine, extended(&h, 4, 4))
}

#[test]
fn an_exhausted_promotion_cancels_the_demotion_and_admits_again() {
    let (mut engine, b) = tight_pool_with_held_demotion(16);
    let outputs = run(&mut engine, &[(4, &b, 1)]);
    assert_eq!(outputs[&RequestId::new(4)].cached, Some(16));
    assert_cancelled_for_headroom(&engine);
    let text = prometheus(&engine);
    assert_eq!(metric(&text, "chop_prefix_cache_ram_hits_total"), 1);
    assert_eq!(fallbacks(&text, "pool_exhausted"), 0);
}

#[test]
fn a_promotion_over_the_budget_cancels_a_demotion_that_makes_no_progress() {
    // The demotion never progresses, as behind continuing foreground
    // copies; the five-page promotion is larger than the four-page budget,
    // so it starts only as a step's first with no demotion outstanding.
    let (mut engine, b) = tight_pool_with_held_demotion(4);
    let before = engine.executor().submitted.len();
    let outputs = run(&mut engine, &[(4, &b, 1)]);
    assert_eq!(outputs[&RequestId::new(4)].cached, Some(16));
    assert_cancelled_for_headroom(&engine);
    // The promotion followed the cancel with no demotion between them.
    let after: Vec<_> = engine.executor().submitted[before..]
        .iter()
        .map(|(class, _)| *class)
        .collect();
    assert_eq!(after.first(), Some(&KvTransferClass::Promotion));
    assert_eq!(
        metric(&prometheus(&engine), "chop_prefix_cache_ram_hits_total"),
        1
    );
}

#[test]
fn a_promotion_over_the_budget_suppresses_demotions_until_it_starts() {
    // H's 16-token anchor is demoted, then evicted by R. X's anchor is
    // under a demotion that never finishes unless cancelled.
    let h = prompt(1, 16);
    let mut engine = engine_from(
        config_sequences(32, 2, 16, 8, Some(tier(false, 4))),
        model(),
    );
    run(&mut engine, &[(1, &h, 1)]);
    settle(&mut engine);
    engine.executor_mut().held = vec![KvTransferClass::Demotion];
    run(&mut engine, &[(2, &prompt(2, 112), 1)]);
    run(&mut engine, &[(3, &prompt(3, 4), 1)]);
    assert_eq!(
        engine.executor().pending_copies(KvTransferClass::Demotion),
        1
    );
    let before = engine.executor().submitted.len();
    // B prefers H's host entry: a five-page promotion over the four-page
    // budget, which waits for no demotion to be outstanding. C and D take
    // both sequence slots, so admission does not reach B again until one
    // ends.
    let b = extended(&h, 4, 4);
    let outputs = run(
        &mut engine,
        &[(4, &b, 1), (5, &prompt(5, 4), 6), (6, &prompt(6, 4), 6)],
    );
    assert_eq!(outputs[&RequestId::new(4)].cached, Some(16));
    assert_eq!(engine.executor().cancels.len(), 1);
    // No demotion started while B waited: its promotion came next.
    let after: Vec<_> = engine.executor().submitted[before..]
        .iter()
        .map(|(class, _)| *class)
        .collect();
    assert_eq!(
        after.first(),
        Some(&KvTransferClass::Promotion),
        "{after:?}"
    );
    assert_eq!(
        metric(&prometheus(&engine), "chop_prefix_cache_ram_hits_total"),
        1
    );
}

#[test]
fn at_most_one_demotion_is_outstanding() {
    // Two four-token anchors of two pages each fall within the lead.
    let mut engine = engine(
        12,
        Some(tier(false, 16)),
        Model {
            held: vec![KvTransferClass::Demotion],
            ..model()
        },
    );
    run(&mut engine, &[(1, &prompt(1, 4), 1), (2, &prompt(2, 4), 1)]);
    let mut outputs = Outputs::new();
    for _ in 0..4 {
        step(&mut engine, &mut outputs);
        assert_eq!(submitted(&engine, KvTransferClass::Demotion), 1);
    }
    // A filler evicts the anchor that waited behind the bound.
    run(&mut engine, &[(3, &prompt(3, 32), 1)]);
    assert_eq!(missed(&prometheus(&engine), "in_flight"), 1);
    assert_eq!(submitted(&engine, KvTransferClass::Demotion), 1);
    // Once the first copy completes, the next demotion may start.
    engine.executor_mut().held.clear();
    settle(&mut engine);
    assert!(demotions(&prometheus(&engine), "ok") >= 1);
    assert_eq!(engine.executor().most_demotions, 1);
}

#[test]
fn the_start_rule_refuses_an_anchor_the_measured_bound_cannot_spare() {
    // A 64-token chunk prefills X in one step: 16 pages and a snapshot,
    // so the measured F is 34 pages. On 40 pages the 23 free cover X's own
    // 17 pages but not F as well, so X is not demoted, and a filler then
    // evicts it.
    let x = prompt(1, 64);
    let mut small = engine_from(config_with(40, 64, 64, Some(tier(false, 16))), model());
    run(&mut small, &[(1, &x, 1)]);
    settle(&mut small);
    assert_eq!(submitted(&small, KvTransferClass::Demotion), 0);
    run(&mut small, &[(2, &prompt(2, 96), 1)]);
    let text = prometheus(&small);
    assert_eq!(missed(&text, "headroom"), 1);
    assert_eq!(demotions(&text, "ok"), 0);
    // On 80 pages the 63 free cover F and X's own pages, and the eviction
    // a larger filler makes loses nothing.
    let mut large = engine_from(config_with(80, 64, 64, Some(tier(false, 16))), model());
    run(&mut large, &[(1, &x, 1)]);
    settle(&mut large);
    assert_eq!(demotions(&prometheus(&large), "ok"), 1);
    run(&mut large, &[(2, &prompt(2, 256), 1)]);
    let text = prometheus(&large);
    let evictions: u64 = ["superseded", "retained", "maximal"]
        .iter()
        .map(|class| {
            metric(
                &text,
                &format!("chop_prefix_cache_evictions_total{{class=\"{class}\"}}"),
            )
        })
        .sum();
    assert!(evictions > 0);
    assert_eq!(
        MISS_REASONS
            .iter()
            .map(|reason| missed(&text, reason))
            .sum::<u64>(),
        0
    );
}

#[test]
fn unconfirmed_copies_poison_the_engine() {
    // A demotion.
    let mut demotion = engine(
        24,
        Some(tier(false, 16)),
        Model {
            status: vec![(KvTransferClass::Demotion, KvTransferStatus::Unconfirmed)],
            ..model()
        },
    );
    submit(&mut demotion, &[(1, &prompt(1, 16), 1)]);
    let mut outputs = Outputs::new();
    let error = loop {
        if let Err(error) = try_step(&mut demotion, &mut outputs) {
            break error;
        }
    };
    assert!(matches!(error, EngineError::PrefixStateUnconfirmed));
    assert!(matches!(
        demotion.step(),
        Err(EngineError::PrefixStateUnconfirmed)
    ));
    demotion.audit().unwrap();
    assert!(demotion.metrics().snapshot().offload.used_pages > 0);
    // A promotion.
    let (mut promotion, _, b) = evicted_after_demotion(model());
    promotion.executor_mut().status =
        vec![(KvTransferClass::Promotion, KvTransferStatus::Unconfirmed)];
    assert!(matches!(
        try_run(&mut promotion, &[(2, &b, 4)]),
        Err(EngineError::PrefixStateUnconfirmed)
    ));
    assert!(matches!(
        promotion.step(),
        Err(EngineError::PrefixStateUnconfirmed)
    ));
    promotion.audit().unwrap();
    // A demotion whose cancel settles unconfirmed.
    let mut cancel = held_demotion(Model {
        cancelled_status: Some(KvTransferStatus::Unconfirmed),
        ..model()
    });
    assert!(matches!(
        try_run(&mut cancel, &[(2, &prompt(2, 36), 1)]),
        Err(EngineError::PrefixStateUnconfirmed)
    ));
    cancel.audit().unwrap();
}

#[test]
fn the_tier_stays_off_without_cancel_support_or_its_prerequisites() {
    let tiered = Some(tier(false, 16));
    let status =
        |config: EngineConfig, model: Model| engine_from(config, model).prefix_ram_status();
    assert_eq!(status(config(24, None), model()), PrefixRamStatusV1::Off);
    assert_eq!(
        status(config(24, tiered), model()),
        PrefixRamStatusV1::Active { forced: false }
    );
    let uncached = EngineConfig::new(
        SchedulerConfig::new(4, 64, 16, 8, 1).unwrap(),
        KvGeometry::new(BLOCK, 24).unwrap(),
        ModelVocabulary::new(VOCABULARY).unwrap(),
        96,
        64,
    )
    .unwrap()
    .with_prefix_ram(tier(false, 16));
    assert_eq!(
        status(uncached, model()),
        PrefixRamStatusV1::Unavailable(PrefixRamUnavailableV1::NoPrefixCache)
    );
    assert_eq!(
        status(
            config(24, tiered),
            Model {
                offload: false,
                ..model()
            }
        ),
        PrefixRamStatusV1::Unavailable(PrefixRamUnavailableV1::NoOffload)
    );
    let uncancellable = || Model {
        cancel: false,
        ..model()
    };
    assert_eq!(
        status(config(24, tiered), uncancellable()),
        PrefixRamStatusV1::Unavailable(PrefixRamUnavailableV1::NoCancel)
    );
    // Without cancel support the configured tier runs exactly as no tier:
    // the same streams, copies and metrics.
    let a = prompt(1, 16);
    let b = extended(&a, 4, 2);
    let filler = prompt(3, 80);
    let workload = |engine: &mut Engine<Model>| {
        let mut outputs = run(engine, &[(1, &a, 1)]);
        outputs.extend(run(engine, &[(3, &filler, 1)]));
        outputs.extend(run(engine, &[(2, &b, 4)]));
        outputs
    };
    let mut off = engine(24, None, uncancellable());
    let mut refused = engine(24, tiered, uncancellable());
    assert_eq!(workload(&mut refused), workload(&mut off));
    assert!(refused.executor().submitted.is_empty());
    assert_eq!(refused.executor().captures, off.executor().captures);
    assert_eq!(prometheus(&refused), prometheus(&off));
}

#[test]
fn every_tier_family_is_exported_from_startup() {
    let plain = [
        "chop_prefix_cache_ram_hits_total",
        "chop_prefix_cache_ram_reused_tokens_total",
        "chop_prefix_ram_headroom_waits_total",
        "chop_prefix_ram_cancel_timeouts_total",
        "chop_prefix_ram_evictions_total",
        "chop_prefix_ram_eviction_log_discarded_total",
        "chop_prefix_ram_anchors",
        "chop_prefix_ram_bytes",
        "chop_prefix_ram_restore_seconds_count",
        "chop_prefix_ram_restore_seconds_bucket{le=\"+Inf\"}",
    ];
    let labelled: [(&str, &str, &[&str]); 4] = [
        (
            "chop_prefix_ram_demotions_total",
            "result",
            &["ok", "failed", "cancelled", "refused", "tier_full"],
        ),
        (
            "chop_prefix_ram_promotions_total",
            "result",
            &["ok", "failed", "invalidated", "gpu_resident"],
        ),
        (
            "chop_prefix_ram_missed_evictions_total",
            "reason",
            &MISS_REASONS,
        ),
        (
            "chop_prefix_ram_fallbacks_total",
            "reason",
            &[
                "pool_exhausted",
                "capacity",
                "failed",
                "invalidated",
                "gpu_resident",
            ],
        ),
    ];
    let engines = [
        engine(
            24,
            None,
            Model {
                offload: false,
                ..model()
            },
        ),
        engine(24, None, model()),
        engine(24, Some(tier(false, 16)), model()),
        engine(24, Some(tier(true, 16)), model()),
    ];
    for engine in &engines {
        let text = prometheus(engine);
        for name in plain {
            assert_eq!(metric(&text, name), 0, "{name}");
        }
        for (name, label, values) in labelled {
            let header = format!("# TYPE {name} counter");
            assert!(text.lines().any(|line| line == header), "{name}");
            for value in values {
                assert_eq!(metric(&text, &format!("{name}{{{label}=\"{value}\"}}")), 0);
            }
        }
        assert!(
            text.lines()
                .any(|line| line == "# TYPE chop_prefix_ram_restore_seconds histogram")
        );
        for direction in ["out", "in"] {
            for kind in ["swap", "prefix"] {
                let sample = format!(
                    "chop_kv_offload_bytes_total{{direction=\"{direction}\",kind=\"{kind}\"}}"
                );
                assert_eq!(metric(&text, &sample), 0);
            }
        }
    }
}
