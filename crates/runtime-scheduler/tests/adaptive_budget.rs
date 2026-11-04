//! Adaptive TBT caps (`SchedulerConfig::with_adaptive_prefill_budget`):
//! a step beside decoding requests takes the largest legal cap whose
//! predicted time keeps its bound: the target under a sustained backlog
//! (as the fixed policy sizes it), four thirds of the target with the step
//! model's margin under a backlog that uncut chunks clear within
//! `TRANSIENT_BACKLOG_MICROS`. A chunk too long to share the step with the
//! decoding requests is always cut. A sustained backlog prefills the prompt
//! with the fewest uncut chunks left first, the earliest arrival among
//! equals, an aged prompt first on every second prefill step. Grid requests
//! with cuts take the cap as a timing cut; other requests take it as the
//! step's prefill budget.

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

/// A step's time against the target, the sustained bound.
fn keeps_target(step: &Step) -> bool {
    step.micros <= u64::from(TARGET_MS) * 1000
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
    // target, without the margin, until the backlog left could clear
    // within the transient window, and serves both decoders.
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
        // Through the transition to a transient backlog, the ceiling.
        assert!(step.decodes.is_empty() || keeps_ceiling(&step), "{step:?}");
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
fn a_transient_backlog_cuts_whole_chunks_to_share_the_step() {
    // Twelve decoders beside one prompt of two grid chunks (about 2.2 s
    // uncut): a whole chunk leaves no room for the decoders and crosses the
    // ceiling, so each is cut to the ceiling and every step that prefills
    // serves all twelve.
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
    for step in steps.iter().filter(|step| !step.prefills.is_empty()) {
        assert!(keeps_ceiling(step), "{step:?}");
        assert!((100..112).all(|id| step.decodes.contains(&id)), "{step:?}");
    }
    assert!(chunks(&steps, 1).len() > 2, "{steps:#?}");
    assert_legal(&chunks(&steps, 1), 2 * GRID);
}

#[test]
fn light_load_within_the_ceiling_serves_as_without_the_switch() {
    // Short turns beside decoders whose uncut chunks share the step within
    // the ceiling: every batch matches the grid scheduler without the
    // switch (uncut chunks, the served order).
    let (mut a, mut b) = (scheduler(grid(true)), scheduler(grid(false)));
    decoders(&mut a, 6);
    decoders(&mut b, 6);
    let mut next = 1;
    for round in 0..40_usize {
        if round % 4 == 0 {
            let prompt = [300, 5000, 2000, 4000][round / 4 % 4];
            submit(&mut a, next, prompt, 3);
            submit(&mut b, next, prompt, 3);
            next += 1;
        }
        let (x, y) = (step(&mut a).unwrap(), step(&mut b).unwrap());
        assert_eq!(
            (&x.prefills, &x.decodes),
            (&y.prefills, &y.decodes),
            "round {round}"
        );
        assert!(x.decodes.is_empty() || keeps_ceiling(&x), "{x:?}");
    }
}

#[test]
fn plain_requests_take_the_ceiling_as_a_transient_step_budget() {
    // Two decoders beside four 2,500-token prompts, about 2.5 s uncut: a
    // transient backlog. The step's prefill budget is the largest multiple
    // of 128 whose step, with the least margin, stays within four thirds
    // of the target: more than the target alone allows.
    let mut s = scheduler(plain(true));
    decoders(&mut s, 2);
    for id in 1..=4 {
        submit(&mut s, id, 2500, 4);
    }
    let first = step(&mut s).unwrap();
    let ceiling = (1..=GRID / 128)
        .map(|n| n * 128)
        .filter(|&tokens| {
            let shape = StepShape::new(tokens, (tokens * (tokens + 1) / 2) as u64, 2);
            cost(shape) * 105 * 3 <= u64::from(TARGET_MS) * 4 * 100_000
        })
        .max()
        .unwrap();
    let cap = first.cap.expect("capped beside decoders");
    assert!(cap.abs_diff(ceiling) <= 128, "{cap} {ceiling}");
    let target = (1..=GRID / 128)
        .map(|n| n * 128)
        .filter(|&tokens| cost(StepShape::new(tokens, 0, 2)) <= u64::from(TARGET_MS) * 1000)
        .max()
        .unwrap();
    assert!(cap > target, "{cap} {target}");
    assert!(keeps_ceiling(&first), "{first:?}");
    assert_eq!(first.decodes.len(), 2, "{first:?}");
}

#[test]
fn fewest_remaining_chunks_go_first_under_a_backlog() {
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

/// Manifest requests use a step budget even for a transient backlog. A
/// generous budget admits the fused tail; a 262 ms target under a longer
/// backlog stops at the anchor instead. Without decoders no cap applies.
#[test]
fn adaptive_caps_price_the_uncapped_manifest_anchor_tail() {
    // A 703-token prompt with a manifest anchor at 640 and a 63-row tail.
    // Beside one decoder a transient backlog takes no cap: the uncut step
    // keeps the ceiling, with or without the fused tail. Under a sustained
    // backlog at a 252 ms target, the unfused 640 rows (249.0 ms) would fit
    // but the fused 703 (255.7 ms) do not, so the step is capped and stops
    // at the anchor.
    for (anchors, decode, backlog, target_ms, capped, rows, anchor) in [
        (0, 1, false, TARGET_MS, false, 640, None),
        (8, 1, false, TARGET_MS, false, 703, Some(640)),
        (8, 0, false, TARGET_MS, false, 703, Some(640)),
        (8, 1, true, 252, true, 640, None),
    ] {
        let config = SchedulerConfig::new(64, 256, GRID, GRID, 1)
            .unwrap()
            .with_prefill_granule(GRANULE)
            .unwrap()
            .with_prefix_cache(1)
            .with_prefill_cuts(true)
            .with_anchor_items(anchors)
            .with_prefill_timing_cuts(true)
            .with_prefill_budget(PrefillBudget::Tbt(
                MixedPolicy::new(target_ms, FLOOR, 0).unwrap(),
            ))
            .with_adaptive_prefill_budget(true);
        let mut s = Scheduler::new(config, KvGeometry::new(GRANULE, 65536).unwrap()).unwrap();
        train(&mut s);
        decoders(&mut s, decode);
        let tokens = Arc::from((0..703).map(|n| 10_000 + n).collect::<Vec<u32>>());
        let chain = PrefixChainV1::new(GRANULE, tokens, Vec::new()).unwrap();
        s.submit_with_prefix(
            RequestSpec::new(RequestId::new(1), 703, 4).unwrap(),
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
        if backlog {
            assert!(cost(batch.shape()) <= u64::from(target_ms) * 1000);
        } else if decode > 0 {
            assert!(cost(batch.shape()) * 105 * 3 <= u64::from(target_ms) * 4 * 100_000);
        }
    }
}

#[test]
fn adaptive_caps_price_grid_anchor_tails_with_current_cadence_bounds() {
    // At 200 ms the transient ceiling with 5% margin admits 253.97 ms:
    // 640 rows cost 249.0 ms, but 703 cost 255.7 ms. Fusion therefore
    // changes an uncapped step into a 640-token cap. Under a sustained
    // backlog, 750 ms admits 5248 rows (743.4 ms), while 252 ms admits
    // 640. The long grid chunk always takes a cap beside decoders.
    for (anchors, decode, backlog, target_ms, cap, rows, anchor) in [
        (0, 1, false, 200, None, 640, None),
        (8, 1, false, 200, Some(640), 640, None),
        (8, 1, false, TARGET_MS, None, 703, Some(640)),
        (8, 0, false, 200, None, 703, Some(640)),
        (8, 1, true, TARGET_MS, Some(5248), 703, Some(640)),
        (8, 1, true, 252, Some(640), 640, None),
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
            .with_prefill_timing_cuts(true)
            .with_prefill_budget(PrefillBudget::Tbt(
                MixedPolicy::new(target_ms, FLOOR, 0).unwrap(),
            ))
            .with_adaptive_prefill_budget(true);
        let mut s = Scheduler::new(config, KvGeometry::new(GRANULE, 65536).unwrap()).unwrap();
        train(&mut s);
        decoders(&mut s, decode);
        submit_prefix(&mut s, 1, (0..703).map(|n| 10_000 + n).collect(), 4);
        if backlog {
            submit(&mut s, 2, 6 * GRID, 4);
        }
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(
            batch.prefill_cap(),
            cap,
            "{anchors} {decode} {backlog} {target_ms}"
        );
        let item = batch
            .items()
            .iter()
            .find(|item| item.request_id().get() == 1)
            .unwrap();
        assert_eq!(item.phase(), BatchPhase::Prefill { tokens: rows });
        assert_eq!(item.anchor_tokens(), anchor);
        if let Some(cuts) = batch.timing_cuts() {
            let chunk = cuts
                .chunks()
                .iter()
                .find(|chunk| chunk.request_id().get() == 1)
                .unwrap();
            assert_eq!(
                (chunk.start(), chunk.end(), chunk.forced()),
                (0, rows, false)
            );
            assert_eq!(
                chunk.reason(),
                if anchor.is_some() {
                    PrefillEnd::Prompt
                } else {
                    PrefillEnd::Capture
                }
            );
        }
        if backlog {
            assert!(cost(batch.shape()) <= u64::from(target_ms) * 1000);
        } else if decode > 0 {
            assert!(cost(batch.shape()) * 105 * 3 <= u64::from(target_ms) * 4 * 100_000);
        }
    }
}

/// Beside a scored prompt, which has no grid contract, a step keeps no
/// grid anchor, so its cap and items are those it takes without an
/// allowance: the fused tail is neither priced nor packed.
#[test]
fn adaptive_caps_price_no_grid_anchor_tail_beside_a_scored_prompt() {
    for (decode, target_ms) in [(1, 200), (1, TARGET_MS), (0, 200)] {
        let mut runs = Vec::new();
        for anchors in [8, 0] {
            let config = SchedulerConfig::new(64, 256, GRID, GRID, 1)
                .unwrap()
                .with_prefill_granule(GRANULE)
                .unwrap()
                .with_guaranteed_prefill_chunks(GRID)
                .unwrap()
                .with_prefix_cache(1)
                .with_prefill_cuts(true)
                .with_grid_anchor_items(anchors)
                .with_prefill_timing_cuts(true)
                .with_prefill_budget(PrefillBudget::Tbt(
                    MixedPolicy::new(target_ms, FLOOR, 0).unwrap(),
                ))
                .with_adaptive_prefill_budget(true);
            let mut s = Scheduler::new(config, KvGeometry::new(GRANULE, 65536).unwrap()).unwrap();
            train(&mut s);
            decoders(&mut s, decode);
            submit_prefix(&mut s, 1, (0..703).map(|n| 10_000 + n).collect(), 4);
            s.submit(
                RequestSpec::new(RequestId::new(2), 40, 2)
                    .unwrap()
                    .with_prompt_scoring(),
            )
            .unwrap();
            let batch = s.next_batch().unwrap().unwrap();
            let items: Vec<_> = batch
                .items()
                .iter()
                .map(|item| (item.request_id(), item.phase(), item.anchor_tokens()))
                .collect();
            assert!(
                items.contains(&(RequestId::new(1), BatchPhase::Prefill { tokens: 640 }, None)),
                "{decode} {target_ms} {anchors}: {items:?}"
            );
            runs.push((batch.prefill_cap(), items, batch.timing_cuts().cloned()));
        }
        assert_eq!(runs[0], runs[1], "{decode} {target_ms}");
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

/// A manifest-anchor scheduler with prefix caching and optional forced
/// preemption, for the restore and preemption paths.
fn prefix_scheduler(anchors: usize, owed: u32, preempt: bool) -> Scheduler {
    prefix_scheduler_at(anchors, owed, preempt, TARGET_MS)
}

fn prefix_scheduler_at(anchors: usize, owed: u32, preempt: bool, target_ms: u32) -> Scheduler {
    let mut config = SchedulerConfig::new(64, 256, GRID, GRID, 1)
        .unwrap()
        .with_prefill_granule(GRANULE)
        .unwrap()
        .with_maximum_scored_rows(64)
        .unwrap()
        .with_prefix_cache(1)
        .with_prefill_cuts(true)
        .with_anchor_items(anchors)
        .with_prefill_timing_cuts(true)
        .with_prefill_budget(PrefillBudget::Tbt(
            MixedPolicy::new(target_ms, FLOOR, owed).unwrap(),
        ))
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
    // A scored long prompt contributes a bounded 64-row chunk and keeps
    // a sustained backlog. The manifest-anchored 4,927-row prompt has an
    // anchor at 4,864 and a 63-row tail. At a 712 ms target, without fusion
    // the 4,928-row uncapped step fits (709.0 ms); with fusion, 4,991 rows
    // (715.8 ms) cross it. Restore and preemption are covered by the
    // preceding two tests.
    for anchors in [0, 8] {
        let mut s = prefix_scheduler_at(anchors, 0, false, 712);
        decoders(&mut s, 1);
        submit_prefix(&mut s, 1, (0..4927).map(|i| 200_000 + i).collect(), 2);
        s.submit(
            RequestSpec::new(RequestId::new(2), 8 * GRID, 2)
                .unwrap()
                .with_prompt_scoring(),
        )
        .unwrap();
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

/// A step's time against the transient ceiling, four thirds of the target,
/// with the least margin.
fn keeps_ceiling(step: &Step) -> bool {
    step.micros * 105 * 3 <= u64::from(TARGET_MS) * 4 * 100_000
}

#[test]
fn a_transient_backlog_keeps_every_decoder_gap_within_the_ceiling() {
    // Two decoders beside one 8,000-token prompt: about 1.04 s uncut, well
    // inside the transient window, and the chunk fits beside them. Every
    // step beside the decoders stays within four thirds of the target.
    let uncut = cost(StepShape::new(8000, (8000 * 8001 / 2) as u64, 2));
    assert!(uncut < TRANSIENT_BACKLOG_MICROS, "{uncut}");
    assert!(uncut * 3 > u64::from(TARGET_MS) * 4 * 1000, "{uncut}");
    let mut s = scheduler(grid(true));
    decoders(&mut s, 2);
    submit(&mut s, 1, 8000, 4);
    let steps = prefill(&mut s, 1, 20);
    for step in steps.iter().filter(|step| !step.decodes.is_empty()) {
        assert!(keeps_ceiling(step), "{step:?}");
    }
    assert!(chunks(&steps, 1).len() >= 2, "{steps:#?}");
    assert_legal(&chunks(&steps, 1), 8000);
}

#[test]
fn no_prefill_runs_alone_while_requests_decode() {
    // Two decoders beside a prompt of two whole grid chunks, a transient
    // backlog: a whole chunk leaves no room for the decoders, so it is
    // cut to share the step instead of running alone.
    let mut s = scheduler(grid(true));
    decoders(&mut s, 2);
    submit(&mut s, 1, 2 * GRID, 4);
    let steps = prefill(&mut s, 1, 30);
    for step in steps.iter().filter(|step| !step.prefills.is_empty()) {
        assert!(
            step.decodes.contains(&100) && step.decodes.contains(&101),
            "{step:?}"
        );
    }
    assert_legal(&chunks(&steps, 1), 2 * GRID);
}

#[test]
fn a_backlog_keeps_arrival_order_among_prompts_of_equal_chunk_counts() {
    // Under a backlog, a later prompt with fewer tokens but as many grid
    // chunks left does not overtake an earlier one; a prompt of one chunk
    // still goes first.
    let mut s = scheduler(grid(true));
    decoders(&mut s, 1);
    submit(&mut s, 1, 2 * GRID + 7000, 4);
    submit(&mut s, 2, 2 * GRID + 1000, 4);
    submit(&mut s, 3, 4000, 4);
    let mut steps = Vec::new();
    while (1..=3).any(|id| prefilling(&s, id)) {
        assert!(steps.len() < 100, "{steps:#?}");
        steps.push(step(&mut s).unwrap());
    }
    let first = |id: u64| {
        steps
            .iter()
            .position(|step| step.prefills.iter().any(|item| item.0 == id))
            .unwrap()
    };
    assert!(first(3) <= first(1), "{steps:#?}");
    assert!(first(1) < first(2), "{steps:#?}");
    let done = |id: u64, prompt: usize| {
        (0..steps.len())
            .find(|&n| remaining(&steps[..=n], id, prompt) == 0)
            .unwrap()
    };
    assert!(
        done(1, 2 * GRID + 7000) < done(2, 2 * GRID + 1000),
        "{steps:#?}"
    );
}

#[test]
fn a_sustained_backlog_caps_as_fixed_does() {
    // Under a sustained backlog the adaptive cap sizes a step to the
    // target as the fixed policy does, without the margin.
    let fixed = Setup {
        grid: true,
        adaptive: false,
        cuts: true,
        budget: tbt(),
    };
    let caps = [grid(true), fixed].map(|setup| {
        let mut s = scheduler(setup);
        decoders(&mut s, 4);
        for id in 1..=3 {
            submit(&mut s, id, 4 * GRID, 4);
        }
        loop {
            let step = step(&mut s).unwrap();
            if !step.prefills.is_empty() && !step.decodes.is_empty() {
                break step.cap;
            }
        }
    });
    assert!(caps[0].is_some(), "{caps:?}");
    assert_eq!(caps[0], caps[1]);
}

#[test]
fn chunks_left_count_absolute_grid_boundaries_after_a_restored_prefix() {
    // A prompt restored from the prefix cache just short of a grid boundary
    // has fewer than 8,192 tokens left but two chunks, so a later prompt of
    // one chunk goes first; a six-chunk prompt keeps the backlog sustained.
    let config = SchedulerConfig::new(64, 256, GRID, GRID, 1)
        .unwrap()
        .with_prefill_granule(GRANULE)
        .unwrap()
        .with_guaranteed_prefill_chunks(GRID)
        .unwrap()
        .with_prefill_cuts(true)
        .with_prefix_cache(1)
        .with_prefill_timing_cuts(true)
        .with_prefill_budget(tbt())
        .with_adaptive_prefill_budget(true);
    let mut s = Scheduler::new(config, KvGeometry::new(GRANULE, 262_144).unwrap()).unwrap();
    train(&mut s);
    let tokens = |n: usize, base: u32| (0..n).map(|i| base + i as u32).collect::<Vec<u32>>();
    submit_prefix(&mut s, 50, tokens(2 * GRID - 1200, 20_000), 1);
    while prefilling(&s, 50) {
        prefix_step(&mut s);
    }
    decoders(&mut s, 1);
    submit(&mut s, 3, 6 * GRID, 4);
    submit_prefix(&mut s, 1, tokens(2 * GRID + 1000, 20_000), 4);
    s.admit().unwrap();
    let cached = s.cached_tokens(RequestId::new(1)).unwrap();
    assert!(
        cached > GRID && 2 * GRID + 1000 - cached <= GRID,
        "{cached}"
    );
    submit(&mut s, 2, 8000, 4);
    let mut steps = Vec::new();
    let left =
        |steps: &[Step]| remaining(steps, 1, 2 * GRID + 1000 - cached) + remaining(steps, 2, 8000);
    while left(&steps) > 0 {
        assert!(steps.len() < 100, "{steps:#?}");
        steps.push(prefix_step(&mut s));
    }
    let first = |id: u64| {
        steps
            .iter()
            .position(|step| step.prefills.iter().any(|item| item.0 == id))
            .unwrap()
    };
    assert!(first(2) < first(1), "{steps:#?}");
    // The six-chunk prompt waits behind both.
    assert!(chunks(&steps[..=first(1)], 3).is_empty(), "{steps:#?}");
}

#[test]
fn without_a_cut_that_keeps_the_bound_the_step_takes_the_floor_beside_every_decoder() {
    // At a 170 ms target even the progress floor beside twelve decoders
    // runs about 240 ms, past the ceiling with the margin: each step takes
    // the floor and still serves all twelve decoders, under a sustained and
    // then a transient backlog.
    let policy = PrefillBudget::Tbt(MixedPolicy::new(170, FLOOR, 0).unwrap());
    let floor_step = cost(StepShape::new(FLOOR, (FLOOR * (FLOOR + 1) / 2) as u64, 12));
    assert!(floor_step * 105 * 3 > 170 * 4 * 100_000, "{floor_step}");
    let mut s = scheduler(Setup {
        budget: policy,
        ..grid(true)
    });
    decoders(&mut s, 12);
    submit(&mut s, 1, 3 * GRID, 4);
    let steps = prefill(&mut s, 1, 200);
    for step in steps.iter().filter(|step| !step.prefills.is_empty()) {
        assert_eq!(step.cap, Some(FLOOR), "{step:?}");
        assert!((100..112).all(|id| step.decodes.contains(&id)), "{step:?}");
    }
    assert_legal(&chunks(&steps, 1), 3 * GRID);
}
