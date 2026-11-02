//! Timing-cut grid chunks (`SchedulerConfig::with_prefill_timing_cuts`): with
//! the TBT budget and declared cuts, a grid request prefilling beside
//! decoding requests ends its chunk at the granule multiple the cap reaches.

use std::sync::Arc;

use runtime_kv::{KvGeometry, PrefixChainV1};
use runtime_scheduler::{
    AdmissionPolicy, BatchOutcome, BatchPhase, MixedPolicy, OptimisticAdmission, PrefillBudget,
    PrefillEnd, PrefixRequestV1, RequestId, RequestSpec, RequestState, Scheduler, SchedulerConfig,
    StepShape, TimingCuts,
};

const GRID: usize = 8192;
const GRANULE: usize = 64;

#[derive(Clone, Copy, Debug)]
struct Setup {
    timing: bool,
    cuts: bool,
    grid: bool,
    grid_anchors: usize,
    budget: PrefillBudget,
    decode_cost: usize,
    admission: AdmissionPolicy,
}

fn tbt(floor: usize) -> PrefillBudget {
    PrefillBudget::Tbt(MixedPolicy::new(150, floor, 0).unwrap())
}

const fn on(budget: PrefillBudget) -> Setup {
    Setup {
        timing: true,
        cuts: true,
        grid: true,
        grid_anchors: 0,
        budget,
        decode_cost: 1,
        admission: AdmissionPolicy::Reserved,
    }
}

fn scheduler(setup: Setup) -> Scheduler {
    let config = SchedulerConfig::new(16, 256, GRID, GRID, setup.decode_cost)
        .unwrap()
        .with_prefill_granule(GRANULE)
        .unwrap()
        .with_prefix_cache(1)
        .with_lone_prefill_expansion(true);
    let config = if setup.grid {
        config.with_guaranteed_prefill_chunks(GRID).unwrap()
    } else {
        config
    };
    let config = config
        .with_prefill_cuts(setup.cuts)
        .with_grid_anchor_items(setup.grid_anchors)
        .with_prefill_budget(setup.budget)
        .with_admission(setup.admission)
        .with_prefill_timing_cuts(setup.timing);
    Scheduler::new(config, KvGeometry::new(GRANULE, 16384).unwrap()).unwrap()
}

fn tokens(id: u64, length: usize) -> Vec<u32> {
    (0..length)
        .map(|position| (id * 100_000 + position as u64) as u32)
        .collect()
}

fn submit_tokens(s: &mut Scheduler, id: u64, tokens: &[u32], output: usize, scored: bool) {
    let mut spec = RequestSpec::new(RequestId::new(id), tokens.len(), output).unwrap();
    if scored {
        spec = spec.with_prompt_scoring();
    }
    let chain = PrefixChainV1::new(GRANULE, Arc::from(tokens), Vec::new()).unwrap();
    s.submit_with_prefix(spec, Some(PrefixRequestV1::new(Arc::new(chain), !scored)))
        .unwrap();
}

fn submit(s: &mut Scheduler, id: u64, prompt: usize, output: usize) {
    submit_tokens(s, id, &tokens(id, prompt), output, false);
}

/// One committed batch: prefill items (request, start, tokens), decoding
/// requests, captured anchors (request, tokens) and the timing cuts.
#[derive(Clone, Debug, Default, PartialEq)]
struct Step {
    prefills: Vec<(u64, usize, usize)>,
    decodes: Vec<u64>,
    captures: Vec<(u64, usize)>,
    anchors: Vec<(u64, usize)>,
    cap: Option<usize>,
    cuts: Option<TimingCuts>,
}

fn step(s: &mut Scheduler) -> Option<Step> {
    let batch = s.next_batch().unwrap()?;
    let mut result = Step {
        cap: batch.prefill_cap(),
        cuts: batch.timing_cuts().cloned(),
        ..Step::default()
    };
    for (item, append) in batch.items().iter().zip(batch.kv_plan().appends()) {
        let id = item.request_id().get();
        if let Some(anchor) = item.anchor_tokens() {
            result.anchors.push((id, anchor));
        }
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
    for capture in s.take_prefix_captures() {
        result
            .captures
            .push((capture.request_id().get(), capture.tokens()));
        let exists = s.has_capture_anchor(&capture);
        assert!(s.begin_prefix_capture(&capture).unwrap().is_some() || exists);
        s.end_prefix_capture(&capture).unwrap();
    }
    s.audit().unwrap();
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
        submit(s, id, 1, 10_000);
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

fn captures(steps: &[Step], id: u64) -> Vec<usize> {
    steps
        .iter()
        .flat_map(|step| &step.captures)
        .filter(|capture| capture.0 == id)
        .map(|capture| capture.1)
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

/// A mixed workload: decoders, two long prompts and short turns arriving
/// while they prefill. Returns every step.
fn workload(setup: Setup) -> Vec<Step> {
    let mut s = scheduler(setup);
    decoders(&mut s, 2);
    submit(&mut s, 1, 2 * GRID + 100, 4);
    let mut steps = Vec::new();
    for tick in 0..400 {
        if tick == 3 {
            submit(&mut s, 2, GRID + 700, 4);
        }
        if tick % 5 == 0 && tick < 60 {
            submit(&mut s, 10 + tick, 40 + tick as usize, 2);
        }
        let Some(step) = step(&mut s) else {
            break;
        };
        steps.push(step);
        if !(1..80).any(
            |id| matches!(s.request_state(RequestId::new(id)), Ok(state) if !state.is_terminal()),
        ) {
            break;
        }
    }
    steps
}

#[test]
fn an_inert_switch_leaves_every_schedule_unchanged() {
    let base = on(tbt(512));
    let cases = [
        // Without the switch.
        (
            Setup {
                timing: false,
                ..base
            },
            base,
        ),
        // Without the bundle's cuts.
        (
            Setup {
                timing: false,
                cuts: false,
                ..base
            },
            Setup {
                cuts: false,
                ..base
            },
        ),
        // Without the TBT budget.
        (
            Setup {
                timing: false,
                budget: PrefillBudget::Fixed,
                ..base
            },
            Setup {
                budget: PrefillBudget::Fixed,
                ..base
            },
        ),
        // Without a grid.
        (
            Setup {
                timing: false,
                grid: false,
                ..base
            },
            Setup {
                grid: false,
                ..base
            },
        ),
    ];
    for (index, (off, inert)) in cases.into_iter().enumerate() {
        let expected = workload(off);
        if index > 0 {
            assert_eq!(workload(inert), expected, "case {index}");
        }
        assert!(expected.iter().all(|step| step.cuts.is_none()));
    }
    // With the switch on, the long prompts take timing cuts.
    let cut = workload(base);
    assert!(cut.iter().any(|step| step.cuts.is_some()));
    assert_ne!(
        cut,
        workload(Setup {
            timing: false,
            ..base
        })
    );
}

#[test]
fn a_declared_grid_without_the_switch_keeps_whole_chunks_beside_decodes() {
    let mut s = scheduler(Setup {
        timing: false,
        ..on(tbt(512))
    });
    decoders(&mut s, 1);
    submit(&mut s, 1, GRID + 100, 4);
    let steps = prefill(&mut s, 1, 16);
    assert_eq!(chunks(&steps, 1), [(0, GRID), (GRID, 64), (GRID + 64, 36)]);
    // A whole chunk leaves no room for the decoder.
    assert!(steps[0].decodes.is_empty() || steps[1].prefills.is_empty());
    assert!(steps.iter().all(|step| step.cap.is_none()));
}

#[test]
fn timing_cuts_let_decoders_share_every_step_and_capture_only_anchors() {
    let target = GRID + 100;
    let mut expected_captures = None;
    for timing in [false, true] {
        let mut s = scheduler(Setup {
            timing,
            ..on(tbt(512))
        });
        decoders(&mut s, 2);
        submit(&mut s, 1, target, 4);
        let steps = prefill(&mut s, 1, 64);
        let chunks = chunks(&steps, 1);
        assert_legal(&chunks, target);
        // Timing cuts publish no anchor: the captures are the grid, the
        // rounded prompt end and the prompt end either way.
        let captures = captures(&steps, 1);
        assert_eq!(captures, [GRID, GRID + 64, target]);
        assert_eq!(*expected_captures.get_or_insert(captures.clone()), captures);
        if !timing {
            assert_eq!(chunks[0], (0, GRID));
            continue;
        }
        let mut expected = (0..16).map(|n| (n * 512, 512)).collect::<Vec<_>>();
        expected.extend([(GRID, 64), (GRID + 64, 36)]);
        assert_eq!(chunks, expected);
        for step in &steps {
            assert_eq!(step.decodes.len(), 2, "{step:?}");
            assert_eq!(step.cap, Some(512));
            let cuts = step.cuts.as_ref().unwrap();
            assert_eq!((cuts.cap(), cuts.target_micros()), (512, 150_000));
            assert!(!cuts.forced());
            let [chunk] = cuts.chunks() else {
                panic!("{cuts:?}")
            };
            let reason = match chunk.end() {
                end if end < GRID => PrefillEnd::Timing,
                GRID => PrefillEnd::Grid,
                end if end == GRID + 64 => PrefillEnd::Capture,
                _ => PrefillEnd::Prompt,
            };
            assert_eq!(chunk.reason(), reason, "{chunk:?}");
            assert_eq!(chunk.request_id(), RequestId::new(1));
        }
    }
}

#[test]
fn without_decodes_a_grid_request_takes_whole_chunks() {
    let mut s = scheduler(on(tbt(512)));
    submit(&mut s, 1, 2 * GRID + 5, 4);
    let steps = prefill(&mut s, 1, 8);
    assert_eq!(chunks(&steps, 1), [(0, GRID), (GRID, GRID), (2 * GRID, 5)]);
    assert!(
        steps
            .iter()
            .all(|step| step.cuts.is_none() && step.cap.is_none())
    );
}

/// Floors below, at and between granule multiples, down to one token: a
/// chunk ends at the last multiple the cap reaches, or at the first one at
/// or past the floor when that is later, and never takes fewer tokens than
/// the floor before the rounded prompt-end anchor or the prompt end.
#[test]
fn chunks_keep_the_floor_at_absolute_granule_multiples() {
    for (floor, first, reason) in [
        (1, 64, PrefillEnd::Floor),
        (63, 64, PrefillEnd::Floor),
        (64, 64, PrefillEnd::Timing),
        (65, 128, PrefillEnd::Floor),
        (127, 128, PrefillEnd::Floor),
        (128, 128, PrefillEnd::Timing),
        (500, 512, PrefillEnd::Floor),
    ] {
        let mut s = scheduler(on(tbt(floor)));
        decoders(&mut s, 1);
        let target = 1000;
        submit(&mut s, 1, target, 4);
        let steps = prefill(&mut s, 1, 64);
        let chunks = chunks(&steps, 1);
        assert_legal(&chunks, target);
        assert_eq!(chunks[0], (0, first), "floor {floor}");
        let cuts = steps[0].cuts.as_ref().unwrap();
        assert_eq!(cuts.chunks()[0].reason(), reason, "floor {floor}");
        assert_eq!((cuts.cap(), cuts.floor()), (floor, floor), "floor {floor}");
        assert!(
            steps
                .iter()
                .all(|step| !step.cuts.as_ref().unwrap().forced())
        );
        // Only the natural ends, 960 and 1000, take fewer tokens.
        for &(start, tokens) in &chunks {
            assert!(
                tokens >= floor || [960, 1000].contains(&(start + tokens)),
                "{chunks:?}"
            );
        }
        assert!(steps.iter().all(|step| step.decodes.len() == 1));
    }
}

/// When the floor's granule multiple does not fit the room beside the
/// decoding requests, the chunk ends at the multiple the cap reaches,
/// below the floor, and still shares the step with the decoders.
#[test]
fn a_floor_multiple_beyond_the_room_rounds_down() {
    let mut s = scheduler(Setup {
        decode_cost: GRID - 500,
        ..on(tbt(512))
    });
    decoders(&mut s, 1);
    submit(&mut s, 1, 1000, 4);
    let steps = prefill(&mut s, 1, 16);
    assert_eq!(
        chunks(&steps, 1),
        [(0, 448), (448, 448), (896, 64), (960, 40)]
    );
    for step in &steps {
        let cuts = step.cuts.as_ref().unwrap();
        assert_eq!((cuts.cap(), cuts.floor()), (500, 500));
        assert!(!cuts.forced());
        assert_eq!(step.decodes, [100]);
    }
}

/// A restored prompt end off the granule: the first cut still ends on an
/// absolute multiple, so later chunks run between multiples.
#[test]
fn a_restored_start_off_the_granule_cuts_at_absolute_multiples() {
    // From 529, 1024 would leave 495 tokens, under the 512 floor: the cut
    // takes 1088. A one-token floor takes the next multiple, 576.
    for (floor, first) in [(512, 559), (1, 47)] {
        let mut s = scheduler(on(tbt(floor)));
        let first_turn = tokens(1, 529);
        submit_tokens(&mut s, 1, &first_turn, 1, false);
        prefill(&mut s, 1, 4);
        decoders(&mut s, 1);
        let mut second = first_turn.clone();
        second.extend(tokens(2, 2000));
        submit_tokens(&mut s, 2, &second, 4, false);
        s.admit().unwrap();
        assert_eq!(s.cached_tokens(RequestId::new(2)).unwrap(), 529);
        s.complete_prefix_restore(RequestId::new(2)).unwrap();
        let steps = prefill(&mut s, 2, 128);
        let chunks = chunks(&steps, 2);
        assert_eq!(chunks[0], (529, first), "floor {floor}");
        assert_legal(&chunks, second.len());
        for &(start, tokens) in &chunks {
            assert!(tokens >= floor || start + tokens >= 2496, "{chunks:?}");
        }
    }
}

/// With every row of the step budget taken by decodes, the cap is zero:
/// one granule still prefills each step, beside the decodes that fit.
#[test]
fn zero_decode_room_still_progresses_one_granule() {
    let mut s = scheduler(Setup {
        decode_cost: GRID / 2,
        ..on(tbt(512))
    });
    decoders(&mut s, 2);
    submit(&mut s, 1, 300, 4);
    let steps = prefill(&mut s, 1, 16);
    assert_eq!(
        chunks(&steps, 1),
        [(0, 64), (64, 64), (128, 64), (192, 64), (256, 44)]
    );
    for step in &steps {
        assert_eq!(step.cap, Some(0));
        assert_eq!(step.decodes.len(), 1, "{step:?}");
        assert!(step.cuts.as_ref().unwrap().forced());
    }
    // The tail ends at the prompt, still forced past the zero cap.
    let tail = steps
        .iter()
        .flat_map(|step| step.cuts.as_ref().unwrap().chunks())
        .find(|chunk| chunk.end() == 300)
        .unwrap();
    assert_eq!((tail.reason(), tail.forced()), (PrefillEnd::Prompt, true));
    // Both decoders keep decoding.
    let decoded = |id| steps.iter().filter(|step| step.decodes == [id]).count();
    assert!(decoded(100) >= 2 && decoded(101) >= 2);
}

/// Grid chunks share the cap: one chunk at the cap or floor per step,
/// and other grid chunks only within what it leaves. A floor rounded up to
/// its granule multiple overruns the cap only for a chunk alone in its step.
#[test]
fn grid_chunks_share_one_cap_per_step() {
    for floor in [512, 500] {
        grid_chunks_share_one_cap(floor);
    }
}

fn grid_chunks_share_one_cap(floor: usize) {
    let mut s = scheduler(on(tbt(floor)));
    decoders(&mut s, 1);
    submit(&mut s, 1, 4096, 4);
    submit(&mut s, 2, 4096, 4);
    submit(&mut s, 3, 40, 4);
    let mut steps = Vec::new();
    while (1..4).any(|id| prefilling(&s, id)) {
        assert!(steps.len() < 64);
        steps.push(step(&mut s).unwrap());
    }
    for step in &steps {
        let total: usize = step.prefills.iter().map(|item| item.2).sum();
        assert!(total <= 512, "{step:?}");
        assert!(total <= floor || step.prefills.len() == 1, "{step:?}");
        let cuts = step.cuts.as_ref().unwrap();
        let timed = cuts
            .chunks()
            .iter()
            .filter(|chunk| matches!(chunk.reason(), PrefillEnd::Timing | PrefillEnd::Floor))
            .count();
        assert!(timed <= 1, "{step:?}");
        assert!(step.decodes.contains(&100), "{step:?}");
    }
    for id in [1, 2] {
        assert_legal(&chunks(&steps, id), 4096);
    }
    // A 500-token floor rounds up to 512-token chunks.
    let floored = steps
        .iter()
        .flat_map(|step| step.cuts.as_ref().unwrap().chunks());
    assert_eq!(
        floored
            .filter(|chunk| chunk.reason() == PrefillEnd::Floor)
            .count()
            > 0,
        floor == 500
    );
    assert_eq!(chunks(&steps, 3), [(0, 40)]);
    // Rotation serves both long prompts while they prefill together.
    let first_end = |id| {
        steps
            .iter()
            .position(|step| step.prefills.iter().any(|item| item.0 == id))
            .unwrap()
    };
    assert!(first_end(1) < 4 && first_end(2) < 4, "{steps:#?}");
}

/// A cut with a trained model: the cap is the largest the fit predicts
/// within the target, above the floor.
#[test]
fn a_trained_model_selects_the_cap() {
    let mut s = scheduler(on(PrefillBudget::Tbt(
        MixedPolicy::new(130, 256, 0).unwrap(),
    )));
    // 14 ms decode; prefill 36 ms + 20 ms beside decodes + 72 us per token.
    let cost = |shape: StepShape| {
        let decode = if shape.decode_items() > 0 { 14_000 } else { 0 };
        if shape.prefill_tokens() == 0 {
            return decode;
        }
        decode
            + 36_000
            + if shape.decode_items() > 0 { 20_000 } else { 0 }
            + shape.prefill_tokens() as u64 * 72
            + shape.prefill_pairs() * 4 / 10_000
    };
    for _ in 0..2 {
        s.observe_batch_time(StepShape::new(0, 0, 1), cost(StepShape::new(0, 0, 1)));
    }
    for step in 0..16_usize {
        let tokens = [2048, 1536, 700, 333, 1100, 512][step % 6];
        let shape = StepShape::new(tokens, (tokens * (tokens + 1) / 2) as u64, 1);
        s.observe_batch_time(shape, cost(shape));
    }
    decoders(&mut s, 1);
    submit(&mut s, 1, 3000, 4);
    let first = step(&mut s).unwrap();
    let cap = first.cap.unwrap();
    assert!(cap > 256 && cap.is_multiple_of(128), "{cap}");
    assert_eq!(first.prefills, [(1, 0, cap)]);
    assert_eq!(first.decodes, [100]);
    let steps = prefill(&mut s, 1, 32);
    let mut all = chunks(&[first], 1);
    all.extend(chunks(&steps, 1));
    assert_legal(&all, 3000);
}

/// Shared-prefix capture points end chunks before the cap and are
/// captured; the cuts between them are not.
#[test]
fn explicit_capture_points_end_chunks_and_keep_their_anchors() {
    for timing in [false, true] {
        let mut s = scheduler(Setup {
            timing,
            ..on(tbt(512))
        });
        let first = tokens(1, 1500);
        submit_tokens(&mut s, 1, &first, 1, false);
        prefill(&mut s, 1, 4);
        decoders(&mut s, 1);
        // Shares 704 tokens, a capture point the first prompt observed.
        let mut second = first[..704].to_vec();
        second.extend(tokens(2, 2000));
        submit_tokens(&mut s, 2, &second, 4, false);
        let steps = prefill(&mut s, 2, 64);
        let chunks = chunks(&steps, 2);
        assert_legal(&chunks, second.len());
        let captures = captures(&steps, 2);
        assert!(captures.contains(&704), "{captures:?}");
        if timing {
            assert_eq!(chunks[..3], [(0, 512), (512, 192), (704, 512)]);
            let reasons = steps
                .iter()
                .filter_map(|step| step.cuts.as_ref())
                .flat_map(|cuts| cuts.chunks().iter().map(|chunk| chunk.reason()))
                .collect::<Vec<_>>();
            assert_eq!(
                reasons[..3],
                [PrefillEnd::Timing, PrefillEnd::Capture, PrefillEnd::Timing]
            );
            // Only capture points are captured, never a timing cut.
            assert!(!captures.contains(&512) && !captures.contains(&1216));
        }
    }
}

/// Scored requests keep their ordinary chunks with the switch on.
#[test]
fn scored_requests_keep_their_chunks() {
    let mut runs = Vec::new();
    for timing in [false, true] {
        let mut s = scheduler(Setup {
            timing,
            ..on(tbt(512))
        });
        decoders(&mut s, 1);
        submit_tokens(&mut s, 1, &tokens(1, 3000), 2, true);
        assert_eq!(
            s.prefill_boundary_contract(RequestId::new(1)).unwrap(),
            None
        );
        let steps = prefill(&mut s, 1, 16);
        assert!(steps.iter().all(|step| step.cuts.is_none()));
        runs.push(chunks(&steps, 1));
    }
    assert_eq!(runs[0], runs[1]);
    assert_eq!(runs[0], [(0, 2944), (2944, 56)]);
}

/// Media requests have no grid contract either, so they keep their chunks.
#[test]
fn media_requests_keep_their_chunks() {
    let mut runs = Vec::new();
    for timing in [false, true] {
        let config = SchedulerConfig::new(16, 256, GRID, GRID, 1)
            .unwrap()
            .with_prefill_granule(GRANULE)
            .unwrap()
            .with_guaranteed_prefill_chunks(GRID)
            .unwrap()
            .with_prefill_cuts(true)
            .with_media_pages(4)
            .with_prefill_budget(tbt(512))
            .with_prefill_timing_cuts(timing);
        let mut s = Scheduler::new(config, KvGeometry::new(GRANULE, 16384).unwrap()).unwrap();
        decoders(&mut s, 1);
        s.submit(
            RequestSpec::new(RequestId::new(1), 3000, 2)
                .unwrap()
                .with_media_pages(1),
        )
        .unwrap();
        assert_eq!(
            s.prefill_boundary_contract(RequestId::new(1)).unwrap(),
            None
        );
        let steps = prefill(&mut s, 1, 16);
        assert!(steps.iter().all(|step| step.cuts.is_none()));
        runs.push(chunks(&steps, 1));
    }
    assert_eq!(runs[0], runs[1]);
}

#[test]
fn cancelling_a_cut_prompt_releases_it_between_cuts() {
    let mut s = scheduler(on(tbt(512)));
    decoders(&mut s, 1);
    submit(&mut s, 1, 4000, 4);
    submit(&mut s, 2, 1500, 4);
    for _ in 0..2 {
        step(&mut s).unwrap();
    }
    s.cancel(RequestId::new(1)).unwrap();
    s.audit().unwrap();
    let steps = prefill(&mut s, 2, 16);
    assert!(chunks(&steps, 1).is_empty());
    let mut all = chunks(&steps, 2);
    all.retain(|chunk| chunk.0 > 0 || chunk.1 > 0);
    assert_eq!(all.last().map(|chunk| chunk.0 + chunk.1), Some(1500));
    assert_eq!(
        s.request_state(RequestId::new(1)),
        Ok(RequestState::Cancelled)
    );
}

/// A preempted request replays its prompt and emitted tokens with timing
/// cuts to the new prefill target.
#[test]
fn a_preempted_request_replays_through_timing_cuts() {
    let mut s = scheduler(Setup {
        admission: AdmissionPolicy::Optimistic(
            OptimisticAdmission::new(0, 1)
                .unwrap()
                .with_forced_preemption(3)
                .unwrap(),
        ),
        ..on(tbt(512))
    });
    decoders(&mut s, 1);
    submit(&mut s, 1, 1500, 8);
    let mut replays = Vec::new();
    for _ in 0..200 {
        if s.request_state(RequestId::new(1)) == Ok(RequestState::Finished) {
            break;
        }
        let preempted = s.preempt_for_next_step(0, |_, _| None).unwrap();
        for capture in s.take_prefix_captures() {
            let exists = s.has_capture_anchor(&capture);
            assert!(s.begin_prefix_capture(&capture).unwrap().is_some() || exists);
            s.end_prefix_capture(&capture).unwrap();
        }
        if preempted.contains(&RequestId::new(1)) {
            replays.clear();
        }
        let step = step(&mut s).unwrap();
        replays.extend(
            step.prefills
                .iter()
                .filter(|item| item.0 == 1)
                .map(|&(_, start, tokens)| (start, tokens)),
        );
    }
    assert_eq!(
        s.request_state(RequestId::new(1)),
        Ok(RequestState::Finished)
    );
    assert!(s.preemptions(RequestId::new(1)).unwrap() >= 1);
    // The replay covers the prompt and the three emitted tokens, the last
    // row sampling the next, and its non-final ends sit on granule
    // multiples.
    assert_legal(&replays, 1503);
    assert!(replays.len() > 1, "{replays:?}");
}

/// Continuous arrivals of new prompts cannot starve a prompt that takes
/// timing cuts: rotation alternates oldest arrival and longest wait.
#[test]
fn continuous_arrivals_do_not_starve_a_cut_prompt() {
    let mut s = scheduler(on(tbt(512)));
    decoders(&mut s, 1);
    submit(&mut s, 1, 6000, 4);
    let mut steps = 0;
    let mut next = 2;
    while prefilling(&s, 1) {
        // A new 1000-token prompt every step.
        submit(&mut s, next, 1000, 1);
        next += 1;
        step(&mut s).unwrap();
        steps += 1;
        assert!(steps < 64, "the long prompt starved");
    }
    // Twelve 512-token cuts, with at most one other prompt's chunk between
    // each.
    assert!(steps <= 24, "{steps}");
}

#[test]
fn grid_anchor_items_span_a_fitting_timing_cap_beside_decodes() {
    for prompt in [300, 511] {
        let mut s = scheduler(Setup {
            grid_anchors: 8,
            ..on(tbt(512))
        });
        decoders(&mut s, 2);
        submit(&mut s, 1, prompt, 4);
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(batch.prefill_cap(), Some(512));
        let item = batch
            .items()
            .iter()
            .find(|item| item.request_id().get() == 1)
            .unwrap();
        assert_eq!(item.phase(), BatchPhase::Prefill { tokens: prompt });
        assert_eq!(item.anchor_tokens(), Some(prompt / GRANULE * GRANULE));
        let [chunk] = batch.timing_cuts().unwrap().chunks() else {
            panic!("one timed chunk")
        };
        assert_eq!(
            (
                chunk.request_id(),
                chunk.start(),
                chunk.end(),
                chunk.reason(),
                chunk.forced()
            ),
            (RequestId::new(1), 0, prompt, PrefillEnd::Prompt, false)
        );
        s.rollback_batch(batch.id()).unwrap();
        let step = step(&mut s).unwrap();
        assert_eq!(step.prefills, [(1, 0, prompt)]);
        assert_eq!(step.decodes.len(), 2);
        assert_eq!(
            step.captures,
            [(1, prompt / GRANULE * GRANULE), (1, prompt)]
        );
    }
}

#[test]
fn grid_anchor_items_stop_at_a_cap_then_span_a_fitting_continuation() {
    let mut s = scheduler(Setup {
        grid_anchors: 8,
        ..on(tbt(512))
    });
    decoders(&mut s, 1);
    submit(&mut s, 1, 789, 4);
    let steps = prefill(&mut s, 1, 4);
    assert_eq!(chunks(&steps, 1), [(0, 512), (512, 277)]);
    assert!(steps[0].anchors.is_empty());
    assert_eq!(steps[1].anchors, [(1, 768)]);
    assert_eq!(captures(&steps, 1), [768, 789]);
    for (step, reason) in steps.iter().zip([PrefillEnd::Timing, PrefillEnd::Prompt]) {
        let chunk = &step.cuts.as_ref().unwrap().chunks()[0];
        assert_eq!(chunk.reason(), reason);
        assert!(!chunk.forced());
        assert_eq!(step.decodes, [100]);
    }
}

#[test]
fn grid_anchor_items_fit_restored_remainders_up_to_the_exact_cap() {
    for remaining in [511, 512, 513] {
        let mut s = scheduler(Setup {
            grid_anchors: 1,
            ..on(tbt(512))
        });
        decoders(&mut s, 1);
        let prefix = tokens(1, 77);
        submit_tokens(&mut s, 1, &prefix, 1, false);
        prefill(&mut s, 1, 4);
        let mut prompt = prefix;
        prompt.extend(tokens(2, remaining));
        submit_tokens(&mut s, 2, &prompt, 4, false);
        s.admit().unwrap();
        assert_eq!(s.cached_tokens(RequestId::new(2)).unwrap(), 77);
        s.complete_prefix_restore(RequestId::new(2)).unwrap();

        let step = step(&mut s).unwrap();
        let fits = remaining <= 512;
        // The anchor is at 576 for all three prompts. At 513 rows the
        // remainder exceeds the cap by one, so the chunk stops after 499.
        let rows = if fits { remaining } else { 499 };
        assert_eq!(step.cap, Some(512));
        assert_eq!(step.prefills, [(2, 77, rows)]);
        assert_eq!(step.anchors, if fits { vec![(2, 576)] } else { vec![] });
        assert_eq!(step.decodes, [100]);
        let chunk = &step.cuts.as_ref().unwrap().chunks()[0];
        assert_eq!(
            (chunk.start(), chunk.end(), chunk.forced()),
            (77, 77 + rows, false)
        );
        assert_eq!(
            chunk.reason(),
            if fits {
                PrefillEnd::Prompt
            } else {
                PrefillEnd::Capture
            }
        );
    }
}

#[test]
fn grid_anchor_items_share_the_cap_with_other_prefills() {
    for (first, second, expected, anchor) in [
        (200, 300, 300, Some(256)),
        (200, 312, 312, Some(256)),
        (200, 313, 256, None),
        (256, 300, 256, None),
        (300, 200, 200, Some(192)),
        (300, 300, 0, None),
        (512, 300, 0, None),
    ] {
        let mut s = scheduler(Setup {
            grid_anchors: 8,
            ..on(tbt(512))
        });
        decoders(&mut s, 1);
        submit(&mut s, 1, first, 4);
        submit(&mut s, 2, second, 4);
        let step = step(&mut s).unwrap();
        assert_eq!(chunks(std::slice::from_ref(&step), 1), [(0, first)]);
        assert_eq!(
            chunks(std::slice::from_ref(&step), 2),
            if expected == 0 {
                vec![]
            } else {
                vec![(0, expected)]
            }
        );
        assert_eq!(
            step.anchors
                .iter()
                .find(|item| item.0 == 2)
                .map(|item| item.1),
            anchor
        );
        assert!(step.prefills.iter().map(|item| item.2).sum::<usize>() <= 512);
    }
}

#[test]
fn a_declined_grid_fusion_keeps_its_allowance_for_a_later_item() {
    let mut s = scheduler(Setup {
        grid_anchors: 1,
        ..on(tbt(512))
    });
    decoders(&mut s, 1);
    let first = tokens(2, 125);
    submit_tokens(&mut s, 2, &first, 1, false);
    prefill(&mut s, 2, 4);
    submit(&mut s, 1, 50, 4);
    submit(&mut s, 3, 511, 4);
    let mut continuation = first;
    continuation.extend(tokens(4, 5));
    submit_tokens(&mut s, 4, &continuation, 4, false);
    s.admit().unwrap();
    assert_eq!(s.cached_tokens(RequestId::new(4)).unwrap(), 125);
    s.complete_prefix_restore(RequestId::new(4)).unwrap();
    // The first 50 tokens leave 462: the 511-row item stops at 448,
    // leaving 14 tokens and the unused allowance for the five-row item.
    let step = step(&mut s).unwrap();
    assert_eq!(step.prefills, [(1, 0, 50), (3, 0, 448), (4, 125, 5)]);
    assert_eq!(step.anchors, [(4, 128)]);
}

#[test]
fn grid_anchor_items_never_fuse_floor_or_progress_forced_chunks() {
    for (floor, decode_cost, decode_count, expected) in
        [(500, 1, 1, 512), (1, 1, 1, 64), (512, GRID / 2, 2, 64)]
    {
        let mut s = scheduler(Setup {
            grid_anchors: 8,
            decode_cost,
            ..on(tbt(floor))
        });
        decoders(&mut s, decode_count);
        submit(&mut s, 1, expected + 1, 4);
        let steps = prefill(&mut s, 1, 4);
        assert_eq!(chunks(&steps, 1), [(0, expected), (expected, 1)]);
        assert!(steps.iter().all(|step| step.anchors.is_empty()));
        assert_eq!(captures(&steps, 1), [expected, expected + 1]);
    }
}
