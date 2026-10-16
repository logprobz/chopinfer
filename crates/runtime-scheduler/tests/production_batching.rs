//! Continuous batching acceptance against a reference state machine.
//! The fixture uses a 32-token prefill chunk and a one-token decode cost.

use runtime_kv::KvGeometry;
use runtime_scheduler::{
    Batch, BatchItem, BatchOutcome, BatchPhase, Overload, RequestId, RequestSpec, RequestState,
    Scheduler, SchedulerCapacityProfile, SchedulerConfig, SubmitError,
};

/// The test prefill chunk width.
const PREFILL_CHUNK_TOKENS: usize = 32;
/// The test decode cost.
const DECODE_COST: usize = 1;
const BLOCK_TOKENS: usize = 16;

fn profile_name(concurrency: usize) -> &'static str {
    match concurrency {
        1 => "C1",
        4 => "C4",
        16 => "C16",
        32 => "C32",
        _ => panic!("unsupported test profile C{concurrency}"),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Config {
    maximum_sequences: usize,
    tracking_capacity: usize,
    token_budget: usize,
    chunk_limit: usize,
    decode_cost: usize,
    block_tokens: usize,
    block_capacity: usize,
}

impl Config {
    fn scheduler(self) -> Scheduler {
        Scheduler::new(
            SchedulerConfig::new(
                self.maximum_sequences,
                self.tracking_capacity,
                self.token_budget,
                self.chunk_limit,
                self.decode_cost,
            )
            .expect("production scheduler configuration must be valid"),
            KvGeometry::new(self.block_tokens, self.block_capacity)
                .expect("production KV geometry must be valid"),
        )
        .expect("production scheduler must construct")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RefState {
    Queued,
    Admitted,
    Prefilling,
    Decoding,
    Finished,
    Cancelled,
}

impl RefState {
    const fn is_live(self) -> bool {
        matches!(self, Self::Admitted | Self::Prefilling | Self::Decoding)
    }

    const fn is_terminal(self) -> bool {
        matches!(self, Self::Finished | Self::Cancelled)
    }

    const fn expected(self) -> RequestState {
        match self {
            Self::Queued => RequestState::Queued,
            Self::Admitted => RequestState::Admitted,
            Self::Prefilling => RequestState::Prefilling,
            Self::Decoding => RequestState::Decoding,
            Self::Finished => RequestState::Finished,
            Self::Cancelled => RequestState::Cancelled,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RefRequest {
    id: u64,
    prompt_tokens: usize,
    maximum_new_tokens: usize,
    state: RefState,
    prefilled: usize,
    emitted: usize,
    kv_tokens: usize,
    reserved_blocks: usize,
}

impl RefRequest {
    const fn maximum_kv_tokens(self) -> usize {
        self.prompt_tokens + self.maximum_new_tokens - 1
    }

    const fn remaining_prompt(self) -> usize {
        self.prompt_tokens - self.prefilled
    }
}

/// A replica-local model of admission, selection shape, and token accounting.
///
/// It never predicts which requests the scheduler picks. It checks that each
/// proposed batch is executable by the production model side and then advances
/// its own accounting on the committed outcomes, so any divergence in state is a
/// scheduler defect rather than a policy difference.
struct Reference {
    config: Config,
    requests: Vec<RefRequest>,
    active: usize,
    reserved_blocks: usize,
    violations: Vec<String>,
}

impl Reference {
    fn new(config: Config) -> Self {
        Self {
            config,
            requests: Vec::new(),
            active: 0,
            reserved_blocks: 0,
            violations: Vec::new(),
        }
    }

    fn submit(&mut self, id: u64, prompt_tokens: usize, maximum_new_tokens: usize) {
        self.requests.push(RefRequest {
            id,
            prompt_tokens,
            maximum_new_tokens,
            state: RefState::Queued,
            prefilled: 0,
            emitted: 0,
            kv_tokens: 0,
            reserved_blocks: 0,
        });
    }

    fn blocks_for(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.config.block_tokens)
    }

    fn index_of(&self, id: u64) -> usize {
        self.requests
            .iter()
            .position(|request| request.id == id)
            .unwrap_or_else(|| panic!("batch names request {id}, which the reference never saw"))
    }

    fn admit(&mut self) -> (Vec<u64>, Option<Overload>) {
        let mut admitted = Vec::new();
        let mut blocked = None;
        for index in 0..self.requests.len() {
            if self.requests[index].state != RefState::Queued {
                continue;
            }
            if self.active == self.config.maximum_sequences {
                blocked = Some(Overload::MaximumSequences);
                break;
            }
            let blocks = self.blocks_for(self.requests[index].maximum_kv_tokens());
            if blocks > self.config.block_capacity - self.reserved_blocks {
                blocked = Some(Overload::KvCapacity);
                break;
            }
            self.requests[index].state = RefState::Admitted;
            self.requests[index].reserved_blocks = blocks;
            self.active += 1;
            self.reserved_blocks += blocks;
            admitted.push(self.requests[index].id);
        }
        (admitted, blocked)
    }

    fn prefill_ready(&self) -> usize {
        self.requests
            .iter()
            .filter(|request| matches!(request.state, RefState::Admitted | RefState::Prefilling))
            .count()
    }

    fn decode_ready(&self) -> usize {
        self.requests
            .iter()
            .filter(|request| request.state == RefState::Decoding)
            .count()
    }

    /// Records every way `batch` is not executable by the production model side.
    fn check(&mut self, batch: &Batch) {
        let items = batch.items();
        if items.is_empty() {
            self.violations
                .push("scheduler proposed an empty batch while work remained".to_owned());
            return;
        }
        let appends = batch.kv_plan().appends();
        if appends.len() != items.len() {
            self.violations.push(format!(
                "batch carries {} prepared KV appends for {} items",
                appends.len(),
                items.len()
            ));
        }
        let mut seen: Vec<u64> = Vec::new();
        let mut cost = 0usize;
        for (position, item) in items.iter().enumerate() {
            let id = item.request_id().get();
            if seen.contains(&id) {
                self.violations.push(format!(
                    "request {id} appears twice in one batch, so its KV rows are not isolated"
                ));
            }
            seen.push(id);
            cost += item.token_cost();
            let request = self.requests[self.index_of(id)];
            let expected_append = self.check_item(request, item);
            if let Some(append) = appends.get(position)
                && append.additional_tokens() != expected_append
            {
                self.violations.push(format!(
                    "request {id} claims {} prepared KV tokens for an item that consumes {expected_append}",
                    append.additional_tokens()
                ));
            }
        }
        if cost > self.config.token_budget {
            self.violations.push(format!(
                "batch costs {cost} tokens against a {} token budget",
                self.config.token_budget
            ));
        }
    }

    /// Checks one item and returns the KV tokens it must append.
    fn check_item(&mut self, request: RefRequest, item: &BatchItem) -> usize {
        match item.phase() {
            BatchPhase::Verify { .. } => {
                self.violations.push(format!(
                    "request {} was scheduled to verify drafts without a draft plan",
                    request.id
                ));
                item.phase().appended_tokens()
            }
            BatchPhase::Prefill { tokens } => {
                if !matches!(request.state, RefState::Admitted | RefState::Prefilling) {
                    self.violations.push(format!(
                        "request {} was scheduled for prefill in state {:?}",
                        request.id, request.state
                    ));
                }
                let remaining = request.remaining_prompt();
                if tokens == 0 || tokens > remaining {
                    self.violations.push(format!(
                        "request {} was scheduled for {tokens} prefill tokens with {remaining} remaining",
                        request.id
                    ));
                }
                if item.token_cost() != tokens {
                    self.violations.push(format!(
                        "request {} charges {} tokens for a {tokens} token prefill chunk",
                        request.id,
                        item.token_cost()
                    ));
                }
                let is_tail = tokens == remaining;
                if !is_tail && tokens != self.config.chunk_limit {
                    self.violations.push(format!(
                        "request {} took a {tokens} token mid-prompt prefill chunk with {remaining} tokens remaining; the full prefill chunk executes a full {} token chunk or the authenticated tail, never a short interior chunk",
                        request.id, self.config.chunk_limit
                    ));
                }
                if item.expects_output() != is_tail {
                    self.violations.push(format!(
                        "request {} expects_output {} on a prefill chunk that {} the prompt",
                        request.id,
                        item.expects_output(),
                        if is_tail {
                            "completes"
                        } else {
                            "does not complete"
                        }
                    ));
                }
                tokens
            }
            BatchPhase::Decode => {
                if request.state != RefState::Decoding {
                    self.violations.push(format!(
                        "request {} was scheduled for decode in state {:?}",
                        request.id, request.state
                    ));
                }
                if item.token_cost() != self.config.decode_cost {
                    self.violations.push(format!(
                        "request {} charges {} tokens for one a decode item",
                        request.id,
                        item.token_cost()
                    ));
                }
                if !item.expects_output() {
                    self.violations.push(format!(
                        "request {} decodes without expecting an output token",
                        request.id
                    ));
                }
                1
            }
        }
    }

    /// Advances the reference on a committed batch and returns the completions.
    fn apply(&mut self, items: &[BatchItem]) -> Vec<u64> {
        let mut completed = Vec::new();
        for item in items {
            let id = item.request_id().get();
            let index = self.index_of(id);
            let request = &mut self.requests[index];
            match item.phase() {
                BatchPhase::Verify { .. } => unreachable!("check_item refuses verify items"),
                BatchPhase::Prefill { tokens } => {
                    request.prefilled += tokens;
                    request.kv_tokens += tokens;
                    if request.prefilled == request.prompt_tokens {
                        request.emitted += 1;
                        request.state = RefState::Decoding;
                    } else {
                        request.state = RefState::Prefilling;
                    }
                }
                BatchPhase::Decode => {
                    request.kv_tokens += 1;
                    request.emitted += 1;
                }
            }
            if request.emitted == request.maximum_new_tokens {
                self.retire_live(index, RefState::Finished);
                completed.push(id);
            }
        }
        completed
    }

    fn cancel(&mut self, id: u64) {
        let index = self.index_of(id);
        self.retire_live(index, RefState::Cancelled);
    }

    fn retire_live(&mut self, index: usize, state: RefState) {
        if self.requests[index].state.is_live() {
            self.active -= 1;
            self.reserved_blocks -= self.requests[index].reserved_blocks;
            self.requests[index].reserved_blocks = 0;
            self.requests[index].kv_tokens = 0;
        }
        self.requests[index].state = state;
    }

    /// Removes terminal records in arrival order, as bulk reap must.
    fn reap(&mut self, maximum: usize) -> Vec<u64> {
        let mut reaped = Vec::new();
        while reaped.len() < maximum {
            let Some(index) = self
                .requests
                .iter()
                .position(|request| request.state.is_terminal())
            else {
                break;
            };
            reaped.push(self.requests.remove(index).id);
        }
        reaped
    }

    /// Compares every tracked counter with the scheduler's public accessors.
    ///
    /// `quiescent` says no batch is pending. A pending batch already owns the
    /// blocks its prepared appends need, so pool occupancy only has to match the
    /// committed token counts between batches.
    fn compare(&mut self, scheduler: &Scheduler, stage: &str, quiescent: bool) {
        for request in &self.requests {
            let id = RequestId::new(request.id);
            let state = scheduler
                .request_state(id)
                .expect("scheduler must still track a referenced request");
            if state != request.state.expected() {
                self.violations.push(format!(
                    "{stage}: request {} is {state:?} in the scheduler and {:?} in the reference",
                    request.id, request.state
                ));
            }
            let emitted = scheduler
                .emitted_tokens(id)
                .expect("scheduler must report emitted tokens");
            if emitted != request.emitted {
                self.violations.push(format!(
                    "{stage}: request {} emitted {emitted} tokens in the scheduler and {} in the reference",
                    request.id, request.emitted
                ));
            }
            let kv_tokens = scheduler
                .kv_tokens(id)
                .expect("scheduler must report KV occupancy");
            if kv_tokens != request.kv_tokens {
                self.violations.push(format!(
                    "{stage}: request {} holds {kv_tokens} KV tokens in the scheduler and {} in the reference",
                    request.id, request.kv_tokens
                ));
            }
        }
        if quiescent {
            let used = scheduler.kv_stats().used_blocks();
            let committed: usize = self
                .requests
                .iter()
                .map(|request| self.blocks_for(request.kv_tokens))
                .sum();
            if used != committed {
                self.violations.push(format!(
                    "{stage}: KV pool reports {used} used blocks against {committed} referenced blocks"
                ));
            }
        }
        if self.reserved_blocks > self.config.block_capacity {
            self.violations.push(format!(
                "{stage}: admission reserved {} blocks against a {} block pool",
                self.reserved_blocks, self.config.block_capacity
            ));
        }
    }
}

fn outcomes_for(batch: &Batch) -> Vec<BatchOutcome> {
    batch
        .items()
        .iter()
        .map(|item| {
            BatchOutcome::success(
                item.request_id(),
                item.phase(),
                usize::from(item.expects_output()),
                false,
            )
        })
        .collect()
}

fn spec(id: u64, prompt_tokens: usize, maximum_new_tokens: usize) -> RequestSpec {
    RequestSpec::new(RequestId::new(id), prompt_tokens, maximum_new_tokens)
        .expect("production request lengths must be valid")
}

/// A mixed load: short prompts reach decode in one batch, long prompts need
/// several the full prefill chunk chunks, so both classes compete in nearly every batch.
fn workload(concurrency: usize) -> Vec<(u64, usize, usize)> {
    (0..concurrency)
        .map(|slot| {
            let prompt_tokens = if slot % 2 == 0 {
                1 + slot % 7
            } else {
                PREFILL_CHUNK_TOKENS + 1 + 29 * (slot % 5)
            };
            (slot as u64 + 1, prompt_tokens, 2 + slot % 4)
        })
        .collect()
}

struct RunReport {
    violations: Vec<String>,
    batches: usize,
    mixed_batches: usize,
}

/// Drives one concurrency level to completion against the reference.
fn drive(concurrency: usize) -> RunReport {
    let config = Config {
        maximum_sequences: concurrency,
        tracking_capacity: concurrency.max(64),
        token_budget: 2 * PREFILL_CHUNK_TOKENS,
        chunk_limit: PREFILL_CHUNK_TOKENS,
        decode_cost: DECODE_COST,
        block_tokens: BLOCK_TOKENS,
        block_capacity: 32 * concurrency,
    };
    let mut scheduler = config.scheduler();
    let mut reference = Reference::new(config);
    for (id, prompt_tokens, maximum_new_tokens) in workload(concurrency) {
        scheduler
            .submit(spec(id, prompt_tokens, maximum_new_tokens))
            .expect("production submission must be accepted");
        reference.submit(id, prompt_tokens, maximum_new_tokens);
    }

    let limit = 64 * concurrency + 256;
    let mut batches = 0usize;
    let mut mixed_batches = 0usize;
    loop {
        let batch = scheduler
            .next_batch()
            .expect("scheduler must not fail while admitted work remains");
        let Some(batch) = batch else { break };
        assert!(
            batches < limit,
            "C{concurrency} did not drain in {limit} batches"
        );
        batches += 1;

        // The scheduler admits inside `next_batch`, so the reference admits here
        // and both sides see the same candidates for this selection.
        reference.admit();
        reference.compare(&scheduler, "after selection", false);
        reference.check(&batch);
        if reference.prefill_ready() > 0 && reference.decode_ready() > 0 {
            let has_prefill = batch
                .items()
                .iter()
                .any(|item| matches!(item.phase(), BatchPhase::Prefill { .. }));
            let has_decode = batch
                .items()
                .iter()
                .any(|item| item.phase() == BatchPhase::Decode);
            mixed_batches += usize::from(has_prefill && has_decode);
        }

        let outcomes = outcomes_for(&batch);
        let completed = scheduler
            .commit_batch(batch.id(), &outcomes)
            .expect("a batch the reference accepts must commit");
        let expected = reference.apply(batch.items());
        let committed: Vec<u64> = completed.completed().iter().map(|id| id.get()).collect();
        if committed != expected {
            reference.violations.push(format!(
                "batch completed {committed:?} where the reference completed {expected:?}"
            ));
        }
        reference.compare(&scheduler, "after commit", true);
        scheduler
            .audit()
            .expect("scheduler invariants must hold after every commit");
    }

    for request in &reference.requests {
        assert_eq!(
            request.state,
            RefState::Finished,
            "C{concurrency} left request {} unfinished",
            request.id
        );
        assert_eq!(
            request.emitted, request.maximum_new_tokens,
            "C{concurrency} request {} emitted the wrong token count",
            request.id
        );
    }
    assert_eq!(
        scheduler.kv_stats().used_blocks(),
        0,
        "C{concurrency} leaked KV blocks after every request finished"
    );

    RunReport {
        violations: reference.violations,
        batches,
        mixed_batches,
    }
}

#[test]
fn production_batching_matches_the_reference_state_machine_at_c1_c4_c16_c32() {
    let mut failures = Vec::new();
    for concurrency in [1, 4, 16, 32] {
        let report = drive(concurrency);
        assert!(report.batches > 0, "C{concurrency} scheduled no batches");
        if concurrency > 1 {
            assert!(
                report.mixed_batches > 0,
                "C{concurrency} never mixed prefill and decode, so the load is not representative"
            );
        }
        for violation in report.violations {
            failures.push(format!("C{concurrency}: {violation}"));
        }
    }
    assert!(
        failures.is_empty(),
        "production batching diverged from the reference state machine:\n  {}",
        failures.join("\n  ")
    );
}

#[test]
fn named_profiles_reject_work_above_c1_c4_c16_and_c32_bounds() {
    for concurrency in [1, 4, 16, 32] {
        let profile =
            SchedulerCapacityProfile::new(profile_name(concurrency), concurrency).unwrap();
        let config = SchedulerConfig::new(
            concurrency,
            concurrency + 1,
            2 * PREFILL_CHUNK_TOKENS,
            PREFILL_CHUNK_TOKENS,
            DECODE_COST,
        )
        .unwrap()
        .with_capacity_profile(profile)
        .unwrap();
        assert_eq!(config.capacity_profile(), profile);

        let mut scheduler = Scheduler::new(
            config,
            KvGeometry::new(BLOCK_TOKENS, 2 * concurrency).unwrap(),
        )
        .unwrap();
        for id in 0..concurrency {
            scheduler
                .submit(spec(id as u64 + 1, 1, 1))
                .expect("work within the profile bound is accepted");
        }
        assert_eq!(
            scheduler.submit(spec(concurrency as u64 + 1, 1, 1)),
            Err(SubmitError::Overloaded(Overload::ProfileCapacity {
                profile: profile_name(concurrency),
                maximum: concurrency,
            })),
            "C{concurrency} must reject work above its declared bound"
        );

        scheduler.cancel(RequestId::new(1)).unwrap();
        scheduler
            .submit(spec(concurrency as u64 + 2, 1, 1))
            .expect("terminal work releases one profile slot");
        scheduler.audit().unwrap();
    }
}

#[test]
fn prefill_chunks_are_full_or_the_authenticated_tail() {
    let config = Config {
        maximum_sequences: 4,
        tracking_capacity: 8,
        token_budget: 2 * PREFILL_CHUNK_TOKENS,
        chunk_limit: PREFILL_CHUNK_TOKENS,
        decode_cost: DECODE_COST,
        block_tokens: BLOCK_TOKENS,
        block_capacity: 64,
    };
    let mut scheduler = config.scheduler();
    // One request reaches decode immediately. Three more owe four chunks each,
    // so more prefill work is ready than the budget can carry in one batch.
    scheduler.submit(spec(1, 1, 8)).expect("short prompt");
    for id in 2..=4 {
        scheduler
            .submit(spec(id, 4 * PREFILL_CHUNK_TOKENS, 2))
            .expect("long prompt");
    }

    let mut short_chunks = Vec::new();
    for _ in 0..6 {
        let Some(batch) = scheduler.next_batch().expect("scheduler must schedule") else {
            break;
        };
        for item in batch.items() {
            if let BatchPhase::Prefill { tokens } = item.phase()
                && tokens != PREFILL_CHUNK_TOKENS
                && !item.expects_output()
            {
                short_chunks.push((item.request_id().get(), tokens));
            }
        }
        let outcomes = outcomes_for(&batch);
        scheduler
            .commit_batch(batch.id(), &outcomes)
            .expect("batch must commit");
        scheduler.audit().expect("invariants must hold");
    }

    assert!(
        short_chunks.is_empty(),
        "mixed batches truncated interior prefill chunks to {short_chunks:?}; the full prefill chunk cannot execute a short interior chunk without unmasked padding"
    );
}

#[test]
fn bounded_decode_preference_keeps_both_classes_progressing() {
    // A budget that holds a full chunk and one decode must carry both classes.
    let together = Config {
        maximum_sequences: 4,
        tracking_capacity: 8,
        token_budget: PREFILL_CHUNK_TOKENS + DECODE_COST,
        chunk_limit: PREFILL_CHUNK_TOKENS,
        decode_cost: DECODE_COST,
        block_tokens: BLOCK_TOKENS,
        block_capacity: 64,
    };
    for phases in contended_phases(together, 8) {
        assert_eq!(phases.first(), Some(&BatchPhase::Decode));
        assert!(
            phases.iter().any(BatchPhase::is_decode)
                && phases.iter().any(|phase| !phase.is_decode()),
            "a batch with room for a chunk and a decode carried only {phases:?}"
        );
    }

    // A budget that holds only one class must alternate between them.
    let alternating = Config {
        maximum_sequences: 4,
        tracking_capacity: 8,
        token_budget: PREFILL_CHUNK_TOKENS,
        chunk_limit: PREFILL_CHUNK_TOKENS,
        decode_cost: PREFILL_CHUNK_TOKENS,
        block_tokens: BLOCK_TOKENS,
        block_capacity: 64,
    };
    let classes: Vec<bool> = contended_phases(alternating, 8)
        .iter()
        .map(|phases| phases.iter().any(BatchPhase::is_decode))
        .collect();
    assert!(
        classes.len() >= 4,
        "the alternating load ran only {} batches",
        classes.len()
    );
    assert!(
        classes.windows(2).all(|window| window[0] != window[1]),
        "one-class batches did not alternate: {classes:?}"
    );
}

#[test]
fn a_chunk_that_cannot_co_run_with_decode_still_runs_on_prefill_batches() {
    // With a budget of exactly one chunk, a chunk and a decode item cannot
    // share a batch. Once the short request reaches decode, the long request
    // must still take its chunks on the batches that prefer prefill, and every
    // chunk it takes stays a full the full prefill chunk chunk or the authenticated tail.
    let config = Config {
        maximum_sequences: 2,
        tracking_capacity: 8,
        token_budget: PREFILL_CHUNK_TOKENS,
        chunk_limit: PREFILL_CHUNK_TOKENS,
        decode_cost: DECODE_COST,
        block_tokens: BLOCK_TOKENS,
        block_capacity: 64,
    };
    let mut scheduler = config.scheduler();
    scheduler
        .submit(spec(1, 3 * PREFILL_CHUNK_TOKENS, 2))
        .expect("long prompt");
    scheduler.submit(spec(2, 1, 8)).expect("short prompt");

    let mut trace = Vec::new();
    let mut previous_was_decode_only = false;
    while let Some(batch) = scheduler.next_batch().expect("scheduler must schedule") {
        let long_request_owing = matches!(
            scheduler.request_state(RequestId::new(1)).unwrap(),
            RequestState::Admitted | RequestState::Prefilling
        );
        if previous_was_decode_only && long_request_owing {
            assert!(
                batch
                    .items()
                    .iter()
                    .any(|item| matches!(item.phase(), BatchPhase::Prefill { .. })),
                "prefill starved behind decode: {trace:?}"
            );
        }
        let has_prefill = batch
            .items()
            .iter()
            .any(|item| matches!(item.phase(), BatchPhase::Prefill { .. }));
        let has_decode = batch
            .items()
            .iter()
            .any(|item| item.phase() == BatchPhase::Decode);
        for item in batch.items() {
            if let BatchPhase::Prefill { tokens } = item.phase() {
                assert!(
                    tokens == PREFILL_CHUNK_TOKENS || item.expects_output(),
                    "request {} took a {tokens} token interior chunk; the full prefill chunk executes a full {} token chunk or the authenticated tail",
                    item.request_id().get(),
                    PREFILL_CHUNK_TOKENS
                );
            }
        }
        trace.push((has_prefill, has_decode));
        previous_was_decode_only = !has_prefill && has_decode;
        let outcomes = outcomes_for(&batch);
        scheduler
            .commit_batch(batch.id(), &outcomes)
            .expect("batch must commit");
        scheduler.audit().expect("invariants must hold");
    }
    assert_eq!(
        scheduler.request_state(RequestId::new(1)).unwrap(),
        RequestState::Finished,
        "the long request never drained"
    );
    assert_eq!(
        scheduler.request_state(RequestId::new(2)).unwrap(),
        RequestState::Finished,
        "the decoding request never drained"
    );
    assert!(
        trace.iter().any(|&(has_prefill, _)| has_prefill),
        "prefill never ran: {trace:?}"
    );
}

trait PhaseClass {
    fn is_decode(&self) -> bool;
}

impl PhaseClass for BatchPhase {
    fn is_decode(&self) -> bool {
        *self == BatchPhase::Decode
    }
}

/// Runs a load where prefill and decode compete, returning each batch's phases.
///
/// The opening batch is dropped: request 1 has not finished its one-token prompt
/// yet, so there is no decode candidate to compete with. Every returned batch is
/// selected with both classes ready.
fn contended_phases(config: Config, batches: usize) -> Vec<Vec<BatchPhase>> {
    let mut scheduler = config.scheduler();
    scheduler
        .submit(spec(1, 1, batches + 2))
        .expect("decode-only request");
    scheduler
        .submit(spec(2, PREFILL_CHUNK_TOKENS * (batches + 2), 2))
        .expect("prefill-heavy request");

    let mut trace: Vec<Vec<BatchPhase>> = Vec::new();
    for _ in 0..=batches {
        let Some(batch) = scheduler.next_batch().expect("scheduler must schedule") else {
            break;
        };
        trace.push(batch.items().iter().map(|item| item.phase()).collect());
        let outcomes = outcomes_for(&batch);
        scheduler
            .commit_batch(batch.id(), &outcomes)
            .expect("batch must commit");
    }
    assert_eq!(
        scheduler.request_state(RequestId::new(1)).unwrap(),
        RequestState::Decoding,
        "the decode side must survive the whole contended window"
    );
    trace.remove(0);
    trace
}

#[test]
fn kv_capacity_blocks_admission_without_oversubscribing_the_pool() {
    let config = Config {
        maximum_sequences: 4,
        tracking_capacity: 8,
        token_budget: 2 * PREFILL_CHUNK_TOKENS,
        chunk_limit: PREFILL_CHUNK_TOKENS,
        decode_cost: DECODE_COST,
        block_tokens: BLOCK_TOKENS,
        block_capacity: 4,
    };
    let mut scheduler = config.scheduler();
    let mut reference = Reference::new(config);
    for id in 1..=3 {
        scheduler
            .submit(spec(id, PREFILL_CHUNK_TOKENS, 1))
            .expect("each request must fit the pool alone");
        reference.submit(id, PREFILL_CHUNK_TOKENS, 1);
    }

    let report = scheduler.admit().expect("admission must not fail");
    let (admitted, blocked) = reference.admit();
    assert_eq!(
        report
            .admitted()
            .iter()
            .map(|id| id.get())
            .collect::<Vec<_>>(),
        admitted,
        "admission order diverged from arrival order"
    );
    assert_eq!(report.blocked(), blocked);
    assert_eq!(report.blocked(), Some(Overload::KvCapacity));
    assert_eq!(
        scheduler.kv_stats().used_blocks(),
        0,
        "reservation must not allocate blocks before a prepared append"
    );
    assert_eq!(
        scheduler.request_state(RequestId::new(3)).unwrap(),
        RequestState::Queued
    );

    let batch = scheduler
        .next_batch()
        .expect("admitted work must schedule")
        .expect("a batch must exist");
    reference.check(&batch);
    let outcomes = outcomes_for(&batch);
    scheduler
        .commit_batch(batch.id(), &outcomes)
        .expect("batch must commit");
    reference.apply(batch.items());

    let report = scheduler.admit().expect("admission must not fail");
    let (admitted, blocked) = reference.admit();
    assert_eq!(
        report
            .admitted()
            .iter()
            .map(|id| id.get())
            .collect::<Vec<_>>(),
        admitted
    );
    assert_eq!(
        admitted,
        vec![3],
        "freed capacity must admit the queued request"
    );
    assert_eq!(blocked, None);
    reference.compare(&scheduler, "after freed capacity", true);
    scheduler.audit().expect("invariants must hold");
    assert!(
        reference.violations.is_empty(),
        "KV capacity handling diverged: {:?}",
        reference.violations
    );
}

#[test]
fn cancellation_drops_the_pending_batch_and_releases_only_its_own_kv() {
    let config = Config {
        maximum_sequences: 4,
        tracking_capacity: 8,
        token_budget: 2 * PREFILL_CHUNK_TOKENS,
        chunk_limit: PREFILL_CHUNK_TOKENS,
        decode_cost: DECODE_COST,
        block_tokens: BLOCK_TOKENS,
        block_capacity: 32,
    };
    let mut scheduler = config.scheduler();
    let mut reference = Reference::new(config);
    for (id, prompt_tokens, maximum_new_tokens) in [
        (1, 2 * PREFILL_CHUNK_TOKENS, 2),
        (2, PREFILL_CHUNK_TOKENS, 2),
    ] {
        scheduler
            .submit(spec(id, prompt_tokens, maximum_new_tokens))
            .expect("submission must be accepted");
        reference.submit(id, prompt_tokens, maximum_new_tokens);
    }

    let batch = scheduler
        .next_batch()
        .expect("scheduler must schedule")
        .expect("a batch must exist");
    reference.admit();
    reference.check(&batch);
    assert!(
        batch
            .items()
            .iter()
            .any(|item| item.request_id() == RequestId::new(1)),
        "the cancelled request must be in flight"
    );

    scheduler
        .cancel(RequestId::new(1))
        .expect("cancellation must succeed");
    reference.cancel(1);
    assert!(
        scheduler
            .commit_batch(batch.id(), &outcomes_for(&batch))
            .is_err(),
        "cancellation must discard the pending batch"
    );
    reference.compare(&scheduler, "after cancellation", true);
    scheduler.audit().expect("invariants must hold");

    // The surviving request keeps its reservation and still runs to completion.
    assert_eq!(
        scheduler.request_state(RequestId::new(2)).unwrap(),
        RequestState::Admitted
    );
    for _ in 0..4 {
        let Some(batch) = scheduler.next_batch().expect("scheduler must schedule") else {
            break;
        };
        reference.check(&batch);
        let outcomes = outcomes_for(&batch);
        scheduler
            .commit_batch(batch.id(), &outcomes)
            .expect("batch must commit");
        reference.apply(batch.items());
        reference.compare(&scheduler, "after cancellation drain", true);
    }
    assert_eq!(
        scheduler.request_state(RequestId::new(2)).unwrap(),
        RequestState::Finished
    );
    assert_eq!(scheduler.kv_stats().used_blocks(), 0);
    assert!(
        reference.violations.is_empty(),
        "cancellation handling diverged: {:?}",
        reference.violations
    );
}

#[test]
fn rollback_restores_the_exact_pre_batch_state() {
    let config = Config {
        maximum_sequences: 4,
        tracking_capacity: 8,
        token_budget: 2 * PREFILL_CHUNK_TOKENS,
        chunk_limit: PREFILL_CHUNK_TOKENS,
        decode_cost: DECODE_COST,
        block_tokens: BLOCK_TOKENS,
        block_capacity: 32,
    };
    let mut scheduler = config.scheduler();
    let mut reference = Reference::new(config);
    for (id, prompt_tokens) in [(1u64, 2 * PREFILL_CHUNK_TOKENS), (2, PREFILL_CHUNK_TOKENS)] {
        scheduler
            .submit(spec(id, prompt_tokens, 3))
            .expect("submission must be accepted");
        reference.submit(id, prompt_tokens, 3);
    }

    // Occupancy is measured after admission and before selection, because a
    // pending batch legitimately owns the blocks its prepared appends need and
    // rollback does not un-admit a request.
    scheduler.admit().expect("admission must succeed");
    reference.admit();
    let before = scheduler.kv_stats();
    let first = scheduler
        .next_batch()
        .expect("scheduler must schedule")
        .expect("a batch must exist");
    reference.check(&first);
    let items = first.items().to_vec();
    scheduler
        .rollback_batch(first.id())
        .expect("rollback must succeed");
    assert_eq!(
        scheduler.kv_stats(),
        before,
        "rollback changed committed occupancy"
    );
    reference.compare(&scheduler, "after rollback", true);

    let retry = scheduler
        .next_batch()
        .expect("scheduler must reschedule")
        .expect("a batch must exist");
    assert_eq!(
        retry.items(),
        items,
        "rollback did not restore the selection"
    );
    reference.check(&retry);
    let outcomes = outcomes_for(&retry);
    scheduler
        .commit_batch(retry.id(), &outcomes)
        .expect("the retried batch must commit");
    reference.apply(retry.items());
    reference.compare(&scheduler, "after retry", true);
    scheduler.audit().expect("invariants must hold");
    assert!(
        reference.violations.is_empty(),
        "rollback handling diverged: {:?}",
        reference.violations
    );
}

#[test]
fn terminal_retirement_reaps_in_arrival_order_and_reuses_capacity() {
    let config = Config {
        maximum_sequences: 2,
        tracking_capacity: 4,
        token_budget: 2 * PREFILL_CHUNK_TOKENS,
        chunk_limit: PREFILL_CHUNK_TOKENS,
        decode_cost: DECODE_COST,
        block_tokens: BLOCK_TOKENS,
        block_capacity: 32,
    };
    let mut scheduler = config.scheduler();
    let mut reference = Reference::new(config);
    for id in 1..=4 {
        scheduler
            .submit(spec(id, PREFILL_CHUNK_TOKENS, 1))
            .expect("submission must be accepted");
        reference.submit(id, PREFILL_CHUNK_TOKENS, 1);
    }
    // Terminate out of arrival order to prove reap order comes from arrival.
    for id in [3, 1, 4, 2] {
        scheduler
            .cancel(RequestId::new(id))
            .expect("cancellation must succeed");
        reference.cancel(id);
    }
    reference.compare(&scheduler, "after cancellation", true);
    assert_eq!(scheduler.terminal_count(), 4);

    let reaped: Vec<u64> = scheduler
        .reap_terminal(2)
        .expect("reap must succeed")
        .iter()
        .map(|id| id.get())
        .collect();
    assert_eq!(reaped, reference.reap(2), "bulk reap left arrival order");
    assert_eq!(reaped, vec![1, 2]);

    // Retiring reclaims tracking capacity for new work.
    for id in 5..=6 {
        scheduler
            .submit(spec(id, PREFILL_CHUNK_TOKENS, 1))
            .expect("retired capacity must be reusable");
        reference.submit(id, PREFILL_CHUNK_TOKENS, 1);
    }
    let reaped: Vec<u64> = scheduler
        .reap_terminal(4)
        .expect("reap must succeed")
        .iter()
        .map(|id| id.get())
        .collect();
    assert_eq!(reaped, reference.reap(4));
    assert_eq!(reaped, vec![3, 4]);
    assert_eq!(scheduler.terminal_count(), 0);
    reference.compare(&scheduler, "after retirement", true);
    scheduler.audit().expect("invariants must hold");
    assert!(
        reference.violations.is_empty(),
        "retirement handling diverged: {:?}",
        reference.violations
    );
}
