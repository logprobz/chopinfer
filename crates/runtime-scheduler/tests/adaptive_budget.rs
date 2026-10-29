//! Adaptive TBT caps (`SchedulerConfig::with_adaptive_prefill_budget`):
//! under a prefill backlog, a step beside decoding requests takes the
//! largest legal cap whose predicted time, with the step model's margin,
//! keeps the target, and the backlog prefills the shortest remaining prompt
//! first, an aged prompt first on every second prefill step. A backlog that uncut
//! chunks clear within `TRANSIENT_BACKLOG_MICROS` is served as without the
//! switch. Grid requests with cuts take the cap as a timing cut; other
//! requests take it as the step's prefill budget.

use runtime_kv::{KvGeometry, PrefixChainV1};
use runtime_scheduler::{
    AdmissionPolicy, BatchOutcome, BatchPhase, MixedPolicy, OptimisticAdmission,
    PREFILL_AGE_MICROS, PrefillBudget, PrefillEnd, PrefixRequestV1, RequestId, RequestSpec,
    RequestState, Scheduler, SchedulerConfig, StepShape, TRANSIENT_BACKLOG_MICROS,
};
use std::sync::Arc;

const GRID: usize = 8192;
const GRANULE: usize = 64;
const FLOOR: usize = 512;

/// The gap target: 750 ms. With the least margin (5%), a step beside a
/// few decoders keeps it up to about 4,900 prompt tokens at no context.
const TARGET_MS: u32 = 750;

fn tbt() -> PrefillBudget {
    PrefillBudget::Tbt(MixedPolicy::new(TARGET_MS, FLOOR, 0).unwrap())
}

/// A synthetic step: decode 20 ms plus 0.5 ms per item; prefill 160 ms
/// fixed, 107 us per token and 0.1 ns per attention pair. A whole
/// 8,192-token chunk takes over 1 s, so four of them exceed the transient
/// window.
fn cost(shape: StepShape) -> u64 {
    let decode = if shape.decode_items() > 0 {
        20_000 + 500 * shape.decode_items() as u64
    } else {
        0
    };
    let prefill = if shape.prefill_tokens() == 0 {
        0
    } else {
        160_000 + 107 * shape.prefill_tokens() as u64 + shape.prefill_pairs() / 10_000
    };
    decode + prefill
}

#[derive(Clone, Copy, Debug)]
struct Setup {
    grid: bool,
    adaptive: bool,
    /// Grid timing cuts (`fixed` when not adaptive).
    cuts: bool,
    budget: PrefillBudget,
}

fn grid(adaptive: bool) -> Setup {
    Setup {
        grid: true,
        adaptive,
        cuts: adaptive,
        budget: tbt(),
    }
}

fn plain(adaptive: bool) -> Setup {
    Setup {
        grid: false,
        adaptive,
        cuts: false,
        budget: tbt(),
    }
}

fn scheduler(setup: Setup) -> Scheduler {
    let config = if setup.grid {
        SchedulerConfig::new(64, 256, GRID, GRID, 1)
            .unwrap()
            .with_prefill_granule(GRANULE)
            .unwrap()
            .with_guaranteed_prefill_chunks(GRID)
            .unwrap()
            .with_prefill_cuts(true)
            .with_prefill_timing_cuts(setup.cuts)
    } else {
        SchedulerConfig::new(64, 256, GRID, 2048, 1).unwrap()
    };
    let config = config
        .with_prefill_budget(setup.budget)
        .with_adaptive_prefill_budget(setup.adaptive);
    let mut s = Scheduler::new(config, KvGeometry::new(GRANULE, 262_144).unwrap()).unwrap();
    train(&mut s);
    s
}

/// Trains the step model from synthetic observations alone.
fn train(s: &mut Scheduler) {
    for decode in [1, 2, 4, 8, 16, 32, 64] {
        for _ in 0..2 {
            let shape = StepShape::new(0, 0, decode);
            s.observe_batch_time(shape, cost(shape));
        }
    }
    for step in 0..24_usize {
        let tokens = [512, 2048, 1024, 7552, 3000, 640, 8192, 1536][step % 8];
        let decode = [0, 1, 4, 8][step % 4];
        let context = step * 2003;
        let pairs = (tokens * context + tokens * (tokens + 1) / 2) as u64;
        let shape = StepShape::new(tokens, pairs, decode);
        s.observe_batch_time(shape, cost(shape));
    }
    assert!(s.mixed_estimates().unwrap().trained());
}

fn submit(s: &mut Scheduler, id: u64, prompt: usize, output: usize) {
    s.submit(RequestSpec::new(RequestId::new(id), prompt, output).unwrap())
        .unwrap();
}

/// One committed batch.
#[derive(Clone, Debug, Default, PartialEq)]
struct Step {
    /// (request, start, tokens).
    prefills: Vec<(u64, usize, usize)>,
    decodes: Vec<u64>,
    cap: Option<usize>,
    /// Timing-cut chunk ends, by request.
    ends: Vec<(u64, PrefillEnd)>,
    micros: u64,
}

/// Forms, commits and times one batch.
fn step(s: &mut Scheduler) -> Option<Step> {
    let batch = s.next_batch().unwrap()?;
    let mut result = Step {
        cap: batch.prefill_cap(),
        ..Step::default()
    };
    if let Some(cuts) = batch.timing_cuts() {
        result.ends = cuts
            .chunks()
            .iter()
            .map(|chunk| (chunk.request_id().get(), chunk.reason()))
            .collect();
    }
    for (item, append) in batch.items().iter().zip(batch.kv_plan().appends()) {
        let id = item.request_id().get();
        match item.phase() {
            BatchPhase::Prefill { tokens } => {
                result.prefills.push((id, append.previous_tokens(), tokens));
            }
            _ => result.decodes.push(id),
        }
    }
    let outcomes = batch
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
        .collect::<Vec<_>>();
    s.commit_batch(batch.id(), &outcomes).unwrap();
    s.audit().unwrap();
    result.micros = cost(batch.shape());
    s.observe_batch_time(batch.shape(), result.micros);
    Some(result)
}

fn prefilling(s: &Scheduler, id: u64) -> bool {
    matches!(
        s.request_state(RequestId::new(id)),
        Ok(RequestState::Queued | RequestState::Admitted | RequestState::Prefilling)
    )
}

/// Starts `count` long-running decoders from id 100 and returns once all
/// of them decode.
fn decoders(s: &mut Scheduler, count: u64) {
    for id in 100..100 + count {
        submit(s, id, 1, 100_000);
    }
    while (100..100 + count).any(|id| prefilling(s, id)) {
        step(s).unwrap();
    }
}

/// Steps until `id` has prefilled, at most `limit` steps.
fn prefill(s: &mut Scheduler, id: u64, limit: usize) -> Vec<Step> {
    let mut steps = Vec::new();
    while prefilling(s, id) {
        assert!(steps.len() < limit, "{id} did not prefill: {steps:#?}");
        steps.push(step(s).unwrap());
    }
    steps
}

/// The chunks of `id`, as (start, tokens).
fn chunks(steps: &[Step], id: u64) -> Vec<(usize, usize)> {
    steps
        .iter()
        .flat_map(|step| &step.prefills)
        .filter(|item| item.0 == id)
        .map(|&(_, start, tokens)| (start, tokens))
        .collect()
}

/// Every chunk ends on an absolute granule multiple or at `target`, and
/// never crosses a grid boundary.
fn assert_legal(chunks: &[(usize, usize)], target: usize) {
    let mut next = chunks.first().map_or(0, |chunk| chunk.0);
    for &(start, tokens) in chunks {
        assert_eq!(start, next, "{chunks:?}");
        assert!(tokens > 0, "{chunks:?}");
        let end = start + tokens;
        assert!(end == target || end.is_multiple_of(GRANULE), "{chunks:?}");
        assert!(end <= (start / GRID + 1) * GRID, "{chunks:?}");
        next = end;
    }
    assert_eq!(next, target, "{chunks:?}");
}

/// Prompt tokens `id` has left to prefill after `steps`.
fn remaining(steps: &[Step], id: u64, prompt: usize) -> usize {
    prompt - chunks(steps, id).iter().map(|chunk| chunk.1).sum::<usize>()
}

/// A step's time against the target with the least margin.
fn keeps_target(step: &Step) -> bool {
    step.micros * 105 <= u64::from(TARGET_MS) * 100_000
}

#[test]
fn the_switch_needs_the_tbt_budget() {
    let config = SchedulerConfig::new(64, 256, GRID, 2048, 1).unwrap();
    assert!(!config.adaptive_prefill_budget());
    assert!(
        !config
            .with_adaptive_prefill_budget(true)
            .adaptive_prefill_budget()
    );
    assert!(!config.with_prefill_budget(tbt()).adaptive_prefill_budget());
    assert!(
        config
            .with_prefill_budget(tbt())
            .with_adaptive_prefill_budget(true)
            .adaptive_prefill_budget()
    );
}

#[test]
fn a_backlog_keeps_every_decoder_gap_within_the_target() {
    // Two decoders beside three prompts of four grid chunks each: about
    // 13 s of uncut prefill. Every step beside the decoders keeps the
    // target until the backlog left could clear within the transient
    // window, and serves both decoders.
    let prompt = 4 * GRID;
    let mut s = scheduler(grid(true));
    decoders(&mut s, 2);
    for id in 1..=3 {
        submit(&mut s, id, prompt, 4);
    }
    let mut steps = Vec::new();
    while (1..=3).any(|id| prefilling(&s, id)) {
        assert!(steps.len() < 200, "{steps:#?}");
        let left: usize = (1..=3).map(|id| remaining(&steps, id, prompt)).sum();
        let step = step(&mut s).unwrap();
        if left > 5 * GRID {
            assert!(keeps_target(&step), "{left} left: {step:?}");
            // Both decoders, and any prompt that finished meanwhile.
            assert!(
                step.decodes.contains(&100) && step.decodes.contains(&101),
                "{step:?}"
            );
            assert!(step.cap.is_some(), "{step:?}");
        }
        steps.push(step);
    }
    for id in 1..=3 {
        assert_legal(&chunks(&steps, id), prompt);
    }
    // The cap is the largest that keeps the target, not the floor.
    assert!(
        steps
            .iter()
            .filter_map(|step| step.cap)
            .any(|cap| cap >= 3000),
        "{steps:#?}"
    );
}

#[test]
fn a_backlog_that_clears_within_the_transient_window_runs_whole_chunks() {
    // Twelve decoders beside one prompt of two grid chunks (about 2.2 s):
    // the chunks run whole, as without the switch.
    let pairs = |start: usize| (GRID * start + GRID * (GRID + 1) / 2) as u64;
    let uncut: u64 = [0, GRID]
        .iter()
        .map(|&start| cost(StepShape::new(GRID, pairs(start), 12)))
        .sum();
    assert!(uncut < TRANSIENT_BACKLOG_MICROS, "{uncut}");
    let mut s = scheduler(grid(true));
    decoders(&mut s, 12);
    submit(&mut s, 1, 2 * GRID, 4);
    let steps = prefill(&mut s, 1, 20);
    assert_eq!(chunks(&steps, 1), [(0, GRID), (GRID, GRID)]);
}

#[test]
fn light_load_serves_as_without_the_switch() {
    // Short turns beside decoders never build a backlog beyond the
    // transient window: every batch matches the scheduler without the
    // switch (grid: uncut chunks; plain: the fixed-target cap).
    for (adaptive, served) in [(grid(true), grid(false)), (plain(true), plain(false))] {
        let mut a = scheduler(adaptive);
        let mut b = scheduler(served);
        decoders(&mut a, 6);
        decoders(&mut b, 6);
        let mut next = 1;
        for round in 0..40_usize {
            if round % 4 == 0 {
                let prompt = [300, 5000, 9000, 2000][round / 4 % 4];
                submit(&mut a, next, prompt, 3);
                submit(&mut b, next, prompt, 3);
                next += 1;
            }
            let (x, y) = (step(&mut a).unwrap(), step(&mut b).unwrap());
            assert_eq!(
                (&x.prefills, &x.decodes),
                (&y.prefills, &y.decodes),
                "{adaptive:?} round {round}"
            );
        }
    }
}

#[test]
fn shortest_remaining_prefill_goes_first_under_a_backlog() {
    // A long prompt queued first and a short one after it: the short one
    // prefills before the long one takes any chunk.
    let mut s = scheduler(grid(true));
    decoders(&mut s, 1);
    submit(&mut s, 1, 6 * GRID, 4);
    submit(&mut s, 2, 2 * GRID, 4);
    let steps = prefill(&mut s, 2, 40);
    assert!(chunks(&steps, 1).is_empty(), "{steps:#?}");
    assert_legal(&chunks(&steps, 2), 2 * GRID);
}

#[test]
fn an_aged_prompt_goes_first_on_every_second_step() {
    // Shorter prompts keep arriving ahead of a long one. Once it has been
    // queued for PREFILL_AGE_MICROS of step time, the long prompt takes a
    // chunk at least on every second step.
    let prompt = 8 * GRID;
    let mut s = scheduler(grid(true));
    decoders(&mut s, 1);
    submit(&mut s, 1, prompt, 4);
    let mut steps = Vec::new();
    let mut clock = 0u64;
    let mut aged_at = None;
    let mut next = 2;
    while prefilling(&s, 1) {
        assert!(steps.len() < 400, "the long prompt starved");
        if steps.len() % 3 == 0 {
            submit(&mut s, next, 3 * GRID, 2);
            next += 1;
        }
        let step = step(&mut s).unwrap();
        clock += step.micros;
        if aged_at.is_none() && clock >= PREFILL_AGE_MICROS + 2_000_000 {
            aged_at = Some(steps.len());
        }
        steps.push(step);
    }
    let aged_at = aged_at.expect("the long prompt aged");
    // Before aging it waited behind the shorter prompts.
    assert!(
        remaining(&steps[..aged_at], 1, prompt) > prompt / 2,
        "{:?}",
        chunks(&steps[..aged_at], 1)
    );
    // After aging, no two consecutive steps without its chunk, apart
    // from the decode step after a whole chunk that ran alone.
    let served: Vec<bool> = steps[aged_at..]
        .iter()
        .map(|step| step.prefills.iter().any(|item| item.0 == 1))
        .collect();
    for window in served.windows(3) {
        assert!(window.iter().any(|&served| served), "{served:?}");
    }
    assert_legal(&chunks(&steps, 1), prompt);
}

#[test]
fn plain_requests_take_the_cadence_cap_as_their_step_budget() {
    // Two decoders beside five 8,000-token prompts: a backlog of about
    // 8 s. Each step's prefill budget is the largest that keeps the
    // target, so several 2,048-token chunks share a step.
    let mut s = scheduler(plain(true));
    decoders(&mut s, 2);
    for id in 1..=5 {
        submit(&mut s, id, 8000, 4);
    }
    let first = step(&mut s).unwrap();
    let tokens: usize = first.prefills.iter().map(|item| item.2).sum();
    assert!(tokens >= 2 * 2048, "{first:?}");
    assert!(keeps_target(&first), "{first:?}");
    assert_eq!(first.decodes.len(), 2, "{first:?}");
    let cap = first.cap.expect("capped beside decoders");
    assert!((4096..=GRID).contains(&cap), "{first:?}");
}

#[test]
fn without_decoders_there_is_no_cap() {
    let mut s = scheduler(grid(true));
    submit(&mut s, 1, 6 * GRID, 4);
    submit(&mut s, 2, 3 * GRID, 4);
    let first = step(&mut s).unwrap();
    assert_eq!(first.cap, None);
    // The shorter prompt first, a whole chunk.
    assert_eq!(first.prefills, [(2, 0, GRID)]);
}

#[test]
fn every_step_progresses_prefill_by_the_floor_or_the_prompt_end() {
    for setup in [grid(true), plain(true)] {
        let mut s = scheduler(setup);
        decoders(&mut s, 12);
        for id in 1..=3 {
            submit(&mut s, id, 5 * GRID, 4);
        }
        let mut steps = Vec::new();
        while (1..=3).any(|id| prefilling(&s, id)) {
            assert!(steps.len() < 300, "{setup:?}");
            steps.push(step(&mut s).unwrap());
        }
        for id in 1..=3 {
            let chunks = chunks(&steps, id);
            for (index, &(start, tokens)) in chunks.iter().enumerate() {
                assert!(
                    tokens >= FLOOR || start + tokens == 5 * GRID || index + 1 == chunks.len(),
                    "{setup:?} {chunks:?}"
                );
            }
        }
        // Every step that serves decoders serves all twelve.
        assert!(
            steps
                .iter()
                .filter(|step| !step.decodes.is_empty())
                .all(|step| (100..112).all(|id| step.decodes.contains(&id))),
            "{setup:?}"
        );
    }
}

#[test]
fn continuous_arrivals_starve_neither_the_long_prompt_nor_the_decoders() {
    for setup in [grid(true), plain(true)] {
        let mut s = scheduler(setup);
        decoders(&mut s, 3);
        submit(&mut s, 1, 6 * GRID, 4);
        let mut steps = Vec::new();
        let mut next = 1000;
        while prefilling(&s, 1) {
            assert!(steps.len() < 600, "{setup:?}: the long prompt starved");
            // A new short turn every step.
            submit(&mut s, next, 300, 2);
            next += 1;
            steps.push(step(&mut s).unwrap());
        }
        // No decoder went two steps without progress.
        for pair in steps.windows(2) {
            for id in 100..103 {
                assert!(
                    pair.iter().any(|step| step.decodes.contains(&id)),
                    "{setup:?} decoder {id} starved: {pair:#?}"
                );
            }
        }
        // Every short turn arrived during the prompt completes its prefill.
        for _ in 0..400 {
            if !(1000..next).any(|id| prefilling(&s, id)) {
                break;
            }
            step(&mut s).unwrap();
        }
        assert!(!(1000..next).any(|id| prefilling(&s, id)), "{setup:?}");
    }
}

#[test]
fn an_untrained_model_keeps_the_progress_floor() {
    let config = SchedulerConfig::new(64, 256, GRID, GRID, 1)
        .unwrap()
        .with_prefill_granule(GRANULE)
        .unwrap()
        .with_guaranteed_prefill_chunks(GRID)
        .unwrap()
        .with_prefill_cuts(true)
        .with_prefill_timing_cuts(true)
        .with_prefill_budget(tbt())
        .with_adaptive_prefill_budget(true);
    let mut s = Scheduler::new(config, KvGeometry::new(GRANULE, 65536).unwrap()).unwrap();
    submit(&mut s, 100, 1, 100_000);
    let mut steps = Vec::new();
    while prefilling(&s, 100) {
        steps.push(s.next_batch().unwrap().unwrap());
        let batch = steps.last().unwrap();
        let outcomes: Vec<_> = batch
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
            .collect();
        s.commit_batch(batch.id(), &outcomes).unwrap();
    }
    submit(&mut s, 1, 2 * GRID, 4);
    let batch = s.next_batch().unwrap().unwrap();
    assert_eq!(batch.prefill_cap(), Some(FLOOR));
}

/// Uncapped fusion is a different candidate from stopping at the anchor. A
/// backlog that clears within the transient window runs uncapped, and an
/// anchored item may then include its tail (653 rows, anchor at 640). Under
/// a longer backlog the cadence cap is a timing cut, which stops at the
/// anchor (640 rows), and the step keeps the target.
#[test]
fn adaptive_caps_price_the_uncapped_anchor_tail() {
    for (anchors, decode, backlog, capped, rows, anchor) in [
        (0, 1, false, false, 640, None),
        (8, 1, false, false, 653, Some(640)),
        (8, 0, false, false, 653, Some(640)),
        (8, 1, true, true, 640, None),
    ] {
        let config = SchedulerConfig::new(64, 256, GRID, GRID, 1)
            .unwrap()
            .with_prefill_granule(GRANULE)
            .unwrap()
            .with_guaranteed_prefill_chunks(GRID)
            .unwrap()
            .with_prefix_cache(1)
            .with_prefill_cuts(true)
            .with_grid_anchor_items(anchors)
            .with_grid_anchor_minimum_rows(357)
            .with_prefill_timing_cuts(true)
            .with_prefill_budget(tbt())
            .with_adaptive_prefill_budget(true);
        let mut s = Scheduler::new(config, KvGeometry::new(GRANULE, 65536).unwrap()).unwrap();
        train(&mut s);
        decoders(&mut s, decode);
        let tokens = Arc::from((0..653).map(|n| 10_000 + n).collect::<Vec<u32>>());
        let chain = PrefixChainV1::new(GRANULE, tokens, Vec::new()).unwrap();
        s.submit_with_prefix(
            RequestSpec::new(RequestId::new(1), 653, 4).unwrap(),
            Some(PrefixRequestV1::new(Arc::new(chain), true)),
        )
        .unwrap();
        if backlog {
            // A longer prompt behind it keeps the backlog past the window.
            submit(&mut s, 2, 6 * GRID, 4);
        }
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(
            batch.prefill_cap().is_some(),
            capped,
            "{anchors} {decode} {backlog}"
        );
        let item = batch
            .items()
            .iter()
            .find(|item| item.request_id() == RequestId::new(1))
            .unwrap();
        assert_eq!(
            item.phase(),
            BatchPhase::Prefill { tokens: rows },
            "{anchors} {decode} {backlog}"
        );
        assert_eq!(item.anchor_tokens(), anchor, "{anchors} {decode} {backlog}");
        assert!(cost(batch.shape()) * 105 <= u64::from(TARGET_MS) * 100_000);
    }
}

fn owing(decode_steps: u32) -> PrefillBudget {
    PrefillBudget::Tbt(MixedPolicy::new(TARGET_MS, FLOOR, decode_steps).unwrap())
}

#[test]
fn an_aged_prompt_progresses_when_decode_only_steps_follow_each_mixed_step() {
    // One owed decode-only step after each mixed step puts every step that
    // takes prefill on one scheduling-tick parity. A 65,536-token prompt
    // queued behind a stream of shorter prompts still takes a chunk on
    // every second prefill step once aged, whichever parity its backlog
    // starts on.
    for lead in 0..2 {
        for base in [grid(true), plain(true)] {
            let setup = Setup {
                budget: owing(1),
                ..base
            };
            let mut s = scheduler(setup);
            decoders(&mut s, 2);
            for _ in 0..lead {
                step(&mut s).unwrap();
            }
            let prompt = 8 * GRID;
            submit(&mut s, 1, prompt, 4);
            let mut steps = Vec::new();
            let mut clock = 0u64;
            let mut next = 1000;
            while prefilling(&s, 1) {
                assert!(
                    steps.len() < 600,
                    "{setup:?} lead {lead}: starved after {} ms of steps with {} of {prompt} tokens left",
                    clock / 1000,
                    prompt
                        - steps
                            .iter()
                            .flat_map(|(_, step): &(u64, Step)| &step.prefills)
                            .filter(|item| item.0 == 1)
                            .map(|item| item.2)
                            .sum::<usize>()
                );
                if steps.len() % 4 == 0 {
                    submit(&mut s, next, 3 * GRID, 2);
                    next += 1;
                }
                let step = step(&mut s).unwrap();
                clock += step.micros;
                steps.push((clock, step));
            }
            // Once aged, under the backlog the shorter prompts keep, it takes
            // a chunk on at least every second step that takes prefill.
            let served: Vec<bool> = steps
                .iter()
                .filter(|(at, step)| {
                    *at >= PREFILL_AGE_MICROS + 2_000_000 && !step.prefills.is_empty()
                })
                .map(|(_, step)| step.prefills.iter().any(|item| item.0 == 1))
                .collect();
            for pair in served.windows(2) {
                assert!(
                    pair.iter().any(|&served| served),
                    "{setup:?} lead {lead}: {served:?}"
                );
            }
            let steps: Vec<Step> = steps.into_iter().map(|(_, step)| step).collect();
            assert_legal(&chunks(&steps, 1), prompt);
        }
    }
}

/// xorshift64*: a fixed-seed generator for the randomized arrivals.
struct Rng(u64);

impl Rng {
    fn below(&mut self, bound: u64) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) % bound
    }
}

#[test]
fn every_waiting_prompt_progresses_within_a_bounded_number_of_steps() {
    // Randomized arrivals, prompt sizes, decoder counts and owed
    // decode-only steps k, on grid and plain requests, through backlogs and
    // transient windows. Two bounds hold on every step:
    // - while a prompt waits, at most k + 1 steps without prefill pass
    //   between steps that take prefill (the owed decode-only steps and one
    //   decode turn before a grid chunk that runs alone);
    // - once the earliest waiting prompt has been queued for
    //   PREFILL_AGE_MICROS, it misses at most one prefill-taking step per
    //   other waiting prompt, plus one: under a backlog it goes first on
    //   every second one, and a transient window's order serves each prompt
    //   that waited longer at most once before it.
    // So, by induction over arrivals, every prompt's wait is bounded in
    // steps, and once arrivals stop every prompt prefills.
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for trial in 0..18_u64 {
        let k = (trial % 3) as u32;
        let base = if trial % 2 == 0 {
            grid(true)
        } else {
            plain(true)
        };
        let setup = Setup {
            budget: owing(k),
            ..base
        };
        let mut s = scheduler(setup);
        decoders(&mut s, 1 + rng.below(6));
        // Waiting prompts: id -> clock when queued.
        let mut waiting = std::collections::BTreeMap::new();
        let (mut clock, mut next, mut steps) = (0u64, 1000, 0);
        let (mut idle, mut missed, mut others, mut current) = (0, 0, 0, None);
        while steps < 240 || !waiting.is_empty() {
            assert!(
                steps < 4000,
                "trial {trial} {setup:?}: {waiting:?} still waiting"
            );
            if steps < 240 && rng.below(4) == 0 {
                let prompt = match rng.below(20) {
                    0 => 6 * GRID + rng.below(4 * GRID as u64) as usize,
                    1..=5 => 4096 + rng.below(2 * GRID as u64) as usize,
                    _ => 64 + rng.below(4032) as usize,
                };
                submit(&mut s, next, prompt, 1 + rng.below(4) as usize);
                waiting.insert(next, clock);
                next += 1;
            }
            let earliest = waiting
                .iter()
                .next()
                .filter(|&(_, &queued)| clock - queued >= PREFILL_AGE_MICROS)
                .map(|(&id, _)| id);
            if earliest != current {
                (current, missed, others) = (earliest, 0, 0);
            }
            others = others.max(waiting.len().saturating_sub(1));
            let waited = !waiting.is_empty();
            let step = step(&mut s).unwrap();
            clock += step.micros;
            steps += 1;
            if step.prefills.is_empty() {
                idle = if waited { idle + 1 } else { 0 };
                assert!(
                    idle <= k + 1,
                    "trial {trial} {setup:?}: {idle} steps without prefill while {waiting:?} waited"
                );
            } else {
                idle = 0;
                if let Some(id) = earliest {
                    if step.prefills.iter().any(|item| item.0 == id) {
                        missed = 0;
                    } else {
                        missed += 1;
                        assert!(
                            missed <= others + 1,
                            "trial {trial} {setup:?}: aged {id} missed {missed} prefill steps beside {others} others"
                        );
                    }
                }
            }
            waiting.retain(|&id, _| prefilling(&s, id));
        }
    }
}

#[test]
fn owed_steps_rollbacks_and_cancellations_leave_every_prompt_progressing() {
    // A long prompt behind a stream of shorter ones, with owed decode-only
    // steps, live cancellations and rolled-back batches. A rolled-back
    // batch replans the same items and cap (nothing it planned commits),
    // and the long prompt and every surviving short prompt prefill.
    for k in [0, 1, 2, 3, 7] {
        for base in [grid(true), plain(true)] {
            for lead in 0..2 {
                let setup = Setup {
                    budget: owing(k),
                    ..base
                };
                let mut s = scheduler(setup);
                decoders(&mut s, 2);
                for _ in 0..lead {
                    step(&mut s).unwrap();
                }
                let prompt = 8 * GRID + 13;
                submit(&mut s, 1, prompt, 2);
                let mut waiting = std::collections::BTreeSet::from([1u64]);
                let (mut steps, mut progress, mut cancelled) = (0, 0, 0);
                while steps < 80 || !waiting.is_empty() {
                    assert!(
                        steps < 8000,
                        "k {k} {setup:?} lead {lead}: {waiting:?} still waiting"
                    );
                    if steps < 80 && steps % 2 == 0 {
                        let id = 1000 + steps as u64;
                        submit(&mut s, id, [653, 4097, 3 * GRID, 8191][(steps / 2) % 4], 2);
                        waiting.insert(id);
                    }
                    if steps < 80 && steps % 6 == 0 && steps >= 2 {
                        let id = RequestId::new(1000 + steps as u64 - 2);
                        if !s.request_state(id).unwrap().is_terminal() {
                            s.cancel(id).unwrap();
                            cancelled += 1;
                        }
                        waiting.remove(&id.get());
                    }
                    if steps % 5 == 0 {
                        let first = s.next_batch().unwrap().unwrap();
                        let (items, cap) = (first.items().to_vec(), first.prefill_cap());
                        s.rollback_batch(first.id()).unwrap();
                        let second = s.next_batch().unwrap().unwrap();
                        assert_eq!(items, second.items());
                        assert_eq!(cap, second.prefill_cap());
                        s.rollback_batch(second.id()).unwrap();
                    }
                    let step = step(&mut s).unwrap();
                    progress += step
                        .prefills
                        .iter()
                        .filter(|item| item.0 == 1)
                        .map(|item| item.2)
                        .sum::<usize>();
                    waiting.retain(|&id| prefilling(&s, id));
                    steps += 1;
                }
                assert_eq!(progress, prompt);
                assert!(cancelled > 0);
            }
        }
    }
}

/// A grid scheduler with prefix caching, anchors and optional forced
/// preemption, for the restore and preemption paths.
fn prefix_scheduler(anchors: usize, owed: u32, preempt: bool) -> Scheduler {
    let mut config = SchedulerConfig::new(64, 256, GRID, GRID, 1)
        .unwrap()
        .with_prefill_granule(GRANULE)
        .unwrap()
        .with_guaranteed_prefill_chunks(GRID)
        .unwrap()
        .with_prefix_cache(1)
        .with_prefill_cuts(true)
        .with_grid_anchor_items(anchors)
        .with_grid_anchor_minimum_rows(357)
        .with_prefill_timing_cuts(true)
        .with_prefill_budget(owing(owed))
        .with_adaptive_prefill_budget(true);
    if preempt {
        config = config.with_admission(AdmissionPolicy::Optimistic(
            OptimisticAdmission::new(0, 2)
                .unwrap()
                .with_forced_preemption(2)
                .unwrap(),
        ));
    }
    let mut s = Scheduler::new(config, KvGeometry::new(GRANULE, 262_144).unwrap()).unwrap();
    train(&mut s);
    s
}

fn submit_prefix(s: &mut Scheduler, id: u64, tokens: Vec<u32>, output: usize) {
    let chain = PrefixChainV1::new(GRANULE, Arc::from(tokens.clone()), Vec::new()).unwrap();
    s.submit_with_prefix(
        RequestSpec::new(RequestId::new(id), tokens.len(), output).unwrap(),
        Some(PrefixRequestV1::new(Arc::new(chain), true)),
    )
    .unwrap();
}

/// One step with the host's restore and capture work done around it.
fn prefix_step(s: &mut Scheduler) -> Step {
    let restores: Vec<_> = s
        .pending_prefix_restores()
        .iter()
        .map(|restore| restore.request_id())
        .collect();
    for id in restores {
        s.complete_prefix_restore(id).unwrap();
    }
    let result = step(s).unwrap();
    for capture in s.take_prefix_captures() {
        let present = s.has_capture_anchor(&capture);
        assert!(s.begin_prefix_capture(&capture).unwrap().is_some() || present);
        s.end_prefix_capture(&capture).unwrap();
    }
    s.audit().unwrap();
    result
}

#[test]
fn restored_prefixes_and_cancellations_keep_the_long_prompt_progressing() {
    // A long prompt restores a cached prefix, then prefills its remainder
    // behind a stream of shorter prompts, some cancelled, with owed
    // decode-only steps and anchors off or on.
    for anchors in [0, 8] {
        for owed in [1, 3] {
            let mut s = prefix_scheduler(anchors, owed, false);
            let tokens = |n: usize| (0..n).map(|i| 20_000 + i as u32).collect::<Vec<u32>>();
            submit_prefix(&mut s, 50, tokens(GRID + 653), 1);
            while prefilling(&s, 50) {
                prefix_step(&mut s);
            }
            decoders(&mut s, 2);
            let prompt = 8 * GRID + 653;
            submit_prefix(&mut s, 1, tokens(prompt), 2);
            s.admit().unwrap();
            let cached = s.cached_tokens(RequestId::new(1)).unwrap();
            assert!(cached >= GRID, "{cached}");
            let (mut steps, mut progress, mut cancelled) = (0, 0, 0);
            while prefilling(&s, 1) || steps < 100 {
                assert!(
                    steps < 1500,
                    "anchors {anchors} owed {owed}: the restored prompt starved"
                );
                if steps < 100 && steps % 4 == 0 {
                    submit_prefix(&mut s, 1000 + steps as u64, tokens(3 * GRID + 13), 2);
                }
                if steps < 100 && steps % 8 == 4 {
                    let id = RequestId::new(1000 + steps as u64 - 4);
                    if !s.request_state(id).unwrap().is_terminal() {
                        s.cancel(id).unwrap();
                        cancelled += 1;
                    }
                }
                progress += prefix_step(&mut s)
                    .prefills
                    .iter()
                    .filter(|item| item.0 == 1)
                    .map(|item| item.2)
                    .sum::<usize>();
                steps += 1;
            }
            assert_eq!(cached + progress, prompt);
            assert!(cancelled > 0);
        }
    }
}

#[test]
fn preempted_requests_do_not_freeze_backlog_aging() {
    // Forced preemptions while a long prompt waits behind shorter ones: it
    // still prefills, and preemption is exercised.
    for owed in [1, 3] {
        let mut s = prefix_scheduler(0, owed, true);
        decoders(&mut s, 2);
        submit(&mut s, 1, 8 * GRID + 13, 2);
        let (mut steps, mut preempted) = (0, 0);
        while prefilling(&s, 1) || steps < 80 {
            assert!(
                steps < 1500,
                "owed {owed}: preemption stalled the long prompt"
            );
            if steps < 80 && steps % 4 == 0 {
                submit(&mut s, 1000 + steps as u64, 3 * GRID + 13, 2);
            }
            preempted += s.preempt_for_next_step(0, |_, _| None).unwrap().len();
            prefix_step(&mut s);
            steps += 1;
        }
        assert!(preempted >= 2, "{preempted}");
    }
}

#[test]
fn the_fused_anchor_tail_changes_the_uncapped_cadence_price() {
    // Under a backlog, a restored long prompt has 64 rows to its next grid
    // boundary and a cold 4,927-row prompt has an anchor at 4,864 and a
    // 63-row tail. Without anchors the uncapped 4,928-row step keeps the
    // target. With anchors the uncapped item would fuse its tail (4,991
    // rows) and cross it, so the cap stops at the anchor instead.
    for anchors in [0, 8] {
        let mut s = prefix_scheduler(anchors, 0, false);
        submit_prefix(&mut s, 50, (0..8129).map(|i| 20_000 + i).collect(), 1);
        while prefilling(&s, 50) {
            prefix_step(&mut s);
        }
        decoders(&mut s, 1);
        submit_prefix(&mut s, 1, (0..4927).map(|i| 200_000 + i).collect(), 2);
        let long = (0..8 * GRID as u32)
            .map(|i| if i < 8128 { 20_000 + i } else { 900_000 + i })
            .collect();
        submit_prefix(&mut s, 2, long, 2);
        s.admit().unwrap();
        assert_eq!(s.cached_tokens(RequestId::new(2)).unwrap(), 8128);
        let restores: Vec<_> = s
            .pending_prefix_restores()
            .iter()
            .map(|restore| restore.request_id())
            .collect();
        for id in restores {
            s.complete_prefix_restore(id).unwrap();
        }
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(
            batch.prefill_cap().is_some(),
            anchors > 0,
            "anchors {anchors}: {batch:?}"
        );
        if anchors > 0 {
            let item = batch
                .items()
                .iter()
                .find(|item| item.request_id() == RequestId::new(1))
                .unwrap();
            assert_eq!(item.anchor_tokens(), None);
            assert!(matches!(item.phase(), BatchPhase::Prefill { tokens } if tokens <= 4864));
        }
    }
}
