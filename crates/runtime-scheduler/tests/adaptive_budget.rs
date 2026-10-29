//! Adaptive TBT caps (`SchedulerConfig::with_adaptive_prefill_budget`):
//! under a prefill backlog, a step beside decoding requests takes the
//! largest legal cap whose predicted time, with the step model's margin,
//! keeps the target, and the backlog prefills the shortest remaining prompt
//! first, an aged prompt first on every second step. A backlog that uncut
//! chunks clear within `TRANSIENT_BACKLOG_MICROS` is served as without the
//! switch. Grid requests with cuts take the cap as a timing cut; other
//! requests take it as the step's prefill budget.

use runtime_kv::KvGeometry;
use runtime_scheduler::{
    BatchOutcome, BatchPhase, MixedPolicy, PREFILL_AGE_MICROS, PrefillBudget, PrefillEnd,
    RequestId, RequestSpec, RequestState, Scheduler, SchedulerConfig, StepShape,
    TRANSIENT_BACKLOG_MICROS,
};

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
    use runtime_kv::PrefixChainV1;
    use runtime_scheduler::PrefixRequestV1;
    use std::sync::Arc;

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
