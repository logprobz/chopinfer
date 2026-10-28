//! Demand-adaptive TBT caps (`SchedulerConfig::with_adaptive_prefill_budget`):
//! beside decoding requests a step takes the cap that costs blocked
//! requests the least time per prefill token, among the largest step under
//! the 250 ms stall line (at least the floor), the largest under the 1 s
//! line, and no cap. Grid requests with cuts take the cap as a timing cut;
//! other requests take it as the step's prefill budget.

use runtime_kv::KvGeometry;
use runtime_scheduler::{
    BatchOutcome, BatchPhase, MixedPolicy, PrefillBudget, PrefillEnd, RequestId, RequestSpec,
    RequestState, Scheduler, SchedulerConfig, StepShape,
};

const GRID: usize = 8192;
const GRANULE: usize = 64;
const FLOOR: usize = 512;

/// A target the fixed sizing meets with about 2,000-token chunks, so its
/// caps differ from every adaptive band.
fn tbt() -> PrefillBudget {
    PrefillBudget::Tbt(MixedPolicy::new(400, FLOOR, 0).unwrap())
}

/// A synthetic step: decode 20 ms plus 0.5 ms per item; prefill 160 ms
/// fixed, 107 us per token and 0.1 ns per attention pair. A 512-token step
/// beside a few decoders stays under 250 ms, about 7,500 tokens fit under
/// 1 s, and a whole 8,192-token chunk takes over 1 s.
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
    budget: PrefillBudget,
}

fn grid(adaptive: bool) -> Setup {
    Setup {
        grid: true,
        adaptive,
        budget: tbt(),
    }
}

fn plain(adaptive: bool) -> Setup {
    Setup {
        grid: false,
        adaptive,
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
            .with_prefill_timing_cuts(true)
    } else {
        SchedulerConfig::new(64, 256, GRID, 2048, 1).unwrap()
    };
    let config = config
        .with_prefill_budget(setup.budget)
        .with_adaptive_prefill_budget(setup.adaptive);
    let mut s = Scheduler::new(config, KvGeometry::new(GRANULE, 65536).unwrap()).unwrap();
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
fn many_decoders_beside_little_prefill_cut_to_the_floor() {
    let mut s = scheduler(grid(true));
    decoders(&mut s, 6);
    submit(&mut s, 1, 3 * GRID, 4);
    let steps = prefill(&mut s, 1, 200);
    assert_legal(&chunks(&steps, 1), 3 * GRID);
    for step in &steps {
        // Every step serves every decoder beside a floor-sized cut.
        assert_eq!(step.decodes.len(), 6, "{step:?}");
        assert_eq!(step.cap, Some(FLOOR), "{step:?}");
        for &(_, _, tokens) in &step.prefills {
            assert!(tokens <= FLOOR, "{step:?}");
        }
        assert!(step.micros < 250_000, "{step:?}");
    }
    assert!(
        steps
            .iter()
            .flat_map(|step| &step.ends)
            .any(|&(_, end)| matches!(end, PrefillEnd::Timing | PrefillEnd::Floor))
    );
}

#[test]
fn prefill_demand_keeps_chunks_under_the_one_second_line() {
    let mut s = scheduler(grid(true));
    decoders(&mut s, 1);
    for id in 1..=3 {
        submit(&mut s, id, 2 * GRID, 4);
    }
    let mut steps = Vec::new();
    while (1..=3).any(|id| prefilling(&s, id)) {
        assert!(steps.len() < 40, "{steps:#?}");
        steps.push(step(&mut s).unwrap());
    }
    for id in 1..=3 {
        assert_legal(&chunks(&steps, id), 2 * GRID);
    }
    // Wide cuts, close to whole chunks, each beside the decoder and under
    // 1 s; the decoder progresses every step.
    let wide: Vec<_> = steps.iter().filter(|step| step.cap.is_some()).collect();
    assert!(!wide.is_empty());
    for step in &steps {
        assert!(step.decodes.contains(&100), "{step:?}");
        assert!(step.micros <= 1_000_000, "{step:?}");
        if let Some(cap) = step.cap {
            assert!((7000..GRID).contains(&cap), "{step:?}");
        }
    }
    assert!(
        steps
            .iter()
            .flat_map(|step| &step.prefills)
            .any(|&(_, _, tokens)| tokens >= 7000)
    );
}

#[test]
fn waiting_requests_that_outnumber_decoders_run_whole_chunks() {
    let mut s = scheduler(grid(true));
    decoders(&mut s, 1);
    for id in 1..=48 {
        submit(&mut s, id, GRID + 64, 1);
    }
    let mut whole = 0;
    let mut previous_whole = false;
    for _ in 0..30 {
        let step = step(&mut s).unwrap();
        if previous_whole {
            // A whole chunk runs alone; the decoder takes the next step.
            assert_eq!(step.decodes, [100], "{step:?}");
        }
        previous_whole = step.prefills.iter().any(|&(_, _, tokens)| tokens == GRID);
        if previous_whole {
            assert_eq!(step.cap, None, "{step:?}");
            assert!(step.decodes.is_empty(), "{step:?}");
            whole += 1;
        }
    }
    assert!(whole >= 5, "{whole}");
}

#[test]
fn demand_moves_the_cap_between_bands_and_lifts_it_without_decoders() {
    let mut s = scheduler(grid(true));
    // Nobody decodes: whole chunks, no cap.
    submit(&mut s, 1, 2 * GRID, 100_000);
    let first = step(&mut s).unwrap();
    assert_eq!(first.prefills, [(1, 0, GRID)]);
    assert_eq!(first.cap, None);
    // One decoder joins beside two waiting prompts: wide cuts.
    submit(&mut s, 2, 1, 100_000);
    submit(&mut s, 3, 2 * GRID, 4);
    let mut caps = Vec::new();
    while prefilling(&s, 1) || prefilling(&s, 2) {
        caps.push(step(&mut s).unwrap().cap);
    }
    assert!(
        caps.iter().any(|cap| cap.is_some_and(|cap| cap > 7000)),
        "{caps:?}"
    );
    // Decoders now outnumber the one prompt left: floor cuts.
    decoders(&mut s, 6);
    let steps = prefill(&mut s, 3, 100);
    assert!(
        steps
            .iter()
            .filter(|step| !step.decodes.is_empty())
            .all(|step| step.cap == Some(FLOOR)),
        "{steps:#?}"
    );
}

#[test]
fn every_step_progresses_prefill_by_the_floor_or_the_prompt_end() {
    for setup in [grid(true), plain(true)] {
        let mut s = scheduler(setup);
        decoders(&mut s, 12);
        submit(&mut s, 1, 5000, 4);
        let steps = prefill(&mut s, 1, 100);
        let chunks = chunks(&steps, 1);
        // The floor cut (12 decoders, one prompt) progresses at least the
        // floor, or to the prompt end.
        for (index, &(start, tokens)) in chunks.iter().enumerate() {
            assert!(
                tokens >= FLOOR || start + tokens == 5000 || index + 1 == chunks.len(),
                "{setup:?} {chunks:?}"
            );
        }
        assert!(chunks.len() >= 5000 / (2 * FLOOR), "{setup:?} {chunks:?}");
        // Each decoder runs in every step.
        assert!(
            steps.iter().all(|step| step.decodes.len() == 12),
            "{setup:?}"
        );
    }
}

#[test]
fn plain_requests_take_the_adaptive_cap_as_their_step_budget() {
    // Many decoders beside one prompt: the largest step under 250 ms.
    let mut s = scheduler(plain(true));
    decoders(&mut s, 6);
    submit(&mut s, 1, 6000, 4);
    let steps = prefill(&mut s, 1, 100);
    for step in &steps {
        assert_eq!(step.decodes.len(), 6, "{step:?}");
        let cap = step.cap.expect("capped beside decoders");
        assert!((FLOOR..=640).contains(&cap), "{step:?}");
        assert!(step.micros < 250_000, "{step:?}");
    }
    // Prompts outnumber the decoder: the step budget grows past one
    // chunk, so several 2048-token chunks share a step under 1 s.
    let mut s = scheduler(plain(true));
    decoders(&mut s, 1);
    for id in 1..=4 {
        submit(&mut s, id, 4096, 4);
    }
    let step = step(&mut s).unwrap();
    let tokens: usize = step.prefills.iter().map(|item| item.2).sum();
    assert!(tokens > 2048 * 2, "{step:?}");
    assert_eq!(step.decodes, [100]);
    assert!(step.micros <= 1_000_000, "{step:?}");
}

#[test]
fn continuous_arrivals_starve_neither_the_long_prompt_nor_the_decoders() {
    for setup in [grid(true), plain(true)] {
        let mut s = scheduler(setup);
        decoders(&mut s, 3);
        submit(&mut s, 1, 6 * GRID, 4);
        let mut steps = Vec::new();
        let mut next = 2;
        while prefilling(&s, 1) {
            assert!(steps.len() < 160, "{setup:?}: the long prompt starved");
            // A new short turn every step.
            submit(&mut s, next, 300, 2);
            next += 1;
            steps.push(step(&mut s).unwrap());
        }
        // The long prompt progressed in large steps: wide grid cuts, or
        // whole 2048-token chunks.
        let long = chunks(&steps, 1);
        if setup.grid {
            assert!(long.len() <= 16, "{setup:?} {long:?}");
        } else {
            assert!(
                long[..long.len() - 1].iter().all(|chunk| chunk.1 == 2048),
                "{long:?}"
            );
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
        for _ in 0..200 {
            if !(2..next).any(|id| prefilling(&s, id)) {
                break;
            }
            step(&mut s).unwrap();
        }
        assert!(!(2..next).any(|id| prefilling(&s, id)), "{setup:?}");
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
