//! Optimistic admission and preemption by recompute.

use std::collections::BTreeMap;
use std::sync::Arc;

use runtime_kv::{KvGeometry, PrefixChainV1};
use runtime_scheduler::{
    AdmissionPolicy, BatchOutcome, BatchPhase, DraftLimits, OptimisticAdmission, PrefixRequestV1,
    RequestId, RequestSpec, RequestState, Scheduler, SchedulerConfig,
};

const BLOCK: usize = 4;

fn optimistic(watermark_millionths: u32, maximum_preemptions: usize) -> AdmissionPolicy {
    AdmissionPolicy::Optimistic(
        OptimisticAdmission::new(watermark_millionths, maximum_preemptions).unwrap(),
    )
}

fn forced(every: usize, maximum_preemptions: usize) -> AdmissionPolicy {
    AdmissionPolicy::Optimistic(
        OptimisticAdmission::new(0, maximum_preemptions)
            .unwrap()
            .with_forced_preemption(every)
            .unwrap(),
    )
}

fn scheduler(
    sequences: usize,
    pages: usize,
    admission: AdmissionPolicy,
    prefix: bool,
) -> Scheduler {
    scheduler_with(sequences, pages, admission, prefix, |config| config)
}

/// `scheduler` with a further configuration step.
fn scheduler_with(
    sequences: usize,
    pages: usize,
    admission: AdmissionPolicy,
    prefix: bool,
    configure: impl FnOnce(SchedulerConfig) -> SchedulerConfig,
) -> Scheduler {
    let config = SchedulerConfig::new(sequences, 64, 32, 8, 1)
        .unwrap()
        .with_lone_prefill_expansion(true)
        .with_admission(admission)
        .with_media_pages(8);
    let config = if prefix {
        config.with_prefix_cache(1)
    } else {
        config
    };
    Scheduler::new(configure(config), KvGeometry::new(BLOCK, pages).unwrap()).unwrap()
}

/// Token `position` of request `id`: its prompt, then what it generates.
fn tokens(id: u64, length: usize) -> Vec<u32> {
    (0..length)
        .map(|position| id as u32 * 1_000 + position as u32)
        .collect()
}

fn chain(id: u64, length: usize) -> PrefixRequestV1 {
    PrefixRequestV1::new(
        Arc::new(PrefixChainV1::new(BLOCK, Arc::from(tokens(id, length)), Vec::new()).unwrap()),
        true,
    )
}

#[derive(Clone, Copy, Debug)]
struct Work {
    prompt: usize,
    maximum: usize,
    /// Tokens the request emits before a stop token ends it.
    stops_after: usize,
    pinned: bool,
    media: bool,
}

const fn work(prompt: usize, maximum: usize, stops_after: usize) -> Work {
    Work {
        prompt,
        maximum,
        stops_after,
        pinned: false,
        media: false,
    }
}

fn submit(scheduler: &mut Scheduler, id: u64, work: Work, prefix: bool) {
    let spec = RequestSpec::new(RequestId::new(id), work.prompt, work.maximum).unwrap();
    let spec = if work.pinned {
        spec.with_reserved_admission()
    } else {
        spec
    };
    let spec = if work.media {
        spec.with_media_pages(1)
    } else {
        spec
    };
    scheduler
        .submit_with_prefix(spec, prefix.then(|| chain(id, work.prompt)))
        .unwrap();
}

#[derive(Debug, Default)]
struct Observed {
    emitted: BTreeMap<u64, usize>,
    preempted: Vec<u64>,
    peak_live: usize,
    /// Committed tokens of each capture made at a preemption.
    preemption_captures: Vec<(u64, usize)>,
    restores: Vec<(u64, usize)>,
    /// Prefill items of recomputing requests: id, start and tokens.
    replays: Vec<(u64, usize, usize)>,
}

/// Ends every due capture, as the engine does before it admits again.
fn settle_captures(scheduler: &mut Scheduler) -> Vec<(u64, usize)> {
    let mut captured = Vec::new();
    for capture in scheduler.take_prefix_captures() {
        scheduler.begin_prefix_capture(&capture).unwrap();
        scheduler.end_prefix_capture(&capture).unwrap();
        captured.push((capture.request_id().get(), capture.tokens()));
    }
    captured
}

fn check(scheduler: &Scheduler) {
    scheduler.audit().unwrap();
    assert!(
        scheduler.kv_stats().used_blocks() <= scheduler.reserved_pages(),
        "live sequences hold only reserved pages: {:?} against {}",
        scheduler.kv_stats(),
        scheduler.reserved_pages()
    );
}

/// Drives the scheduler the way the engine does until every request ends.
fn run(scheduler: &mut Scheduler, work: &BTreeMap<u64, Work>, drafts: usize) -> Observed {
    let mut observed = Observed::default();
    for _ in 0..10_000 {
        let preempted = scheduler
            .preempt_for_next_step(drafts, |id, length| Some(chain(id.get(), length)))
            .unwrap();
        observed
            .preempted
            .extend(preempted.iter().map(|id| id.get()));
        observed
            .preemption_captures
            .extend(settle_captures(scheduler));
        check(scheduler);
        scheduler.admit().unwrap();
        for restore in scheduler.pending_prefix_restores() {
            observed
                .restores
                .push((restore.request_id().get(), restore.tokens()));
            scheduler
                .complete_prefix_restore(restore.request_id())
                .unwrap();
        }
        check(scheduler);
        let limits = if drafts == 0 {
            DraftLimits::NONE
        } else {
            DraftLimits::new(drafts, 64)
        };
        let Some(batch) = scheduler
            .next_batch_with_drafts(limits, |candidates, budget| {
                let mut spare = budget.spare_draft_tokens().min(budget.spare_rows());
                candidates
                    .iter()
                    .map(|candidate| {
                        let drafts = candidate.maximum_draft_tokens().min(spare);
                        spare -= drafts;
                        drafts
                    })
                    .collect()
            })
            .unwrap()
        else {
            break;
        };
        observed.peak_live = observed
            .peak_live
            .max(scheduler.kv_stats().active_sequences());
        let mut outcomes = Vec::new();
        for (item, append) in batch.items().iter().zip(batch.kv_plan().appends()) {
            let id = item.request_id();
            let emitted = scheduler.emitted_tokens(id).unwrap();
            let request = work[&id.get()];
            if let BatchPhase::Prefill { tokens } = item.phase() {
                let start = append.previous_tokens();
                if scheduler.preemptions(id).unwrap() > 0 && emitted > 0 {
                    observed.replays.push((id.get(), start, tokens));
                }
                if item.expects_output() {
                    assert_eq!(
                        start + tokens,
                        request.prompt + emitted,
                        "the last prefill row follows every emitted token"
                    );
                }
            }
            let count = match item.phase() {
                _ if !item.expects_output() => 0,
                BatchPhase::Verify { draft_tokens } => {
                    (draft_tokens + 1).min(request.stops_after - emitted)
                }
                _ => 1,
            };
            outcomes.push(BatchOutcome::success(
                id,
                item.phase(),
                count,
                count > 0 && emitted + count == request.stops_after,
            ));
            *observed.emitted.entry(id.get()).or_default() += count;
        }
        scheduler.commit_batch(batch.id(), &outcomes).unwrap();
        settle_captures(scheduler);
        check(scheduler);
    }
    for id in work.keys() {
        assert_eq!(
            scheduler.request_state(RequestId::new(*id)).unwrap(),
            RequestState::Finished
        );
    }
    let stats = scheduler.kv_stats();
    assert_eq!(stats.active_sequences(), 0);
    assert_eq!(stats.used_blocks(), 0);
    assert_eq!(
        stats.free_blocks() + stats.cached_blocks(),
        stats.block_capacity()
    );
    assert_eq!(scheduler.reserved_pages(), 0);
    observed
}

fn submit_all(scheduler: &mut Scheduler, work: &BTreeMap<u64, Work>, prefix: bool) {
    for (id, request) in work {
        submit(scheduler, *id, *request, prefix);
    }
}

/// Twelve requests with varied prompts and early stops, far below their
/// maximum output.
fn storm() -> BTreeMap<u64, Work> {
    (1..=12)
        .map(|id| {
            let id = id as u64;
            (
                id,
                work(3 + (id as usize * 5) % 7, 40, 10 + (id as usize * 7) % 21),
            )
        })
        .collect()
}

#[test]
fn optimistic_admission_runs_what_worst_case_reservation_serializes() {
    let work: BTreeMap<_, _> = (1..=8).map(|id| (id, work(6, 100, 5))).collect();
    let mut reserved = scheduler(8, 32, AdmissionPolicy::Reserved, false);
    submit_all(&mut reserved, &work, false);
    let serial = run(&mut reserved, &work, 0);
    assert_eq!(serial.peak_live, 1, "each request reserves 27 of 32 pages");

    let mut optimistic = scheduler(8, 32, optimistic(10_000, 3), false);
    submit_all(&mut optimistic, &work, false);
    let concurrent = run(&mut optimistic, &work, 0);
    assert_eq!(concurrent.peak_live, 8);
    assert!(concurrent.preempted.is_empty());
    assert_eq!(concurrent.emitted, serial.emitted);
}

#[test]
fn a_preemption_storm_keeps_pages_exact_and_emits_each_token_once() {
    let work = storm();
    for prefix in [false, true] {
        for drafts in [0, 3] {
            let mut scheduler = scheduler(6, 16, optimistic(0, 3), prefix);
            submit_all(&mut scheduler, &work, prefix);
            let observed = run(&mut scheduler, &work, drafts);
            assert!(
                observed.preempted.len() >= 4,
                "prefix {prefix}, drafts {drafts}: {observed:?}"
            );
            for (id, request) in &work {
                assert_eq!(observed.emitted[id], request.stops_after);
                assert!(scheduler.preemptions(RequestId::new(*id)).unwrap() <= 3);
            }
            // Under this much pressure most anchors are evicted before the
            // recompute, which then prefills cold.
            assert_eq!(!observed.preemption_captures.is_empty(), prefix);
        }
    }
}

#[test]
fn recomputes_restore_the_prefix_their_preemption_published() {
    let work: BTreeMap<_, _> = (1..=6)
        .map(|id| (id, work(5 + id as usize, 30, 14 + id as usize)))
        .collect();
    for drafts in [0, 3] {
        let mut scheduler = scheduler(6, 64, forced(4, 2), true);
        submit_all(&mut scheduler, &work, true);
        let mut observed = run(&mut scheduler, &work, drafts);
        assert_eq!(observed.preempted.len(), 12);
        observed.preemption_captures.sort_unstable();
        observed.restores.sort_unstable();
        assert_eq!(
            observed.restores, observed.preemption_captures,
            "every recompute resumes where its preemption left off"
        );
        // An exact restore schedules the pending token as decode or verify.
        assert!(observed.replays.is_empty(), "{observed:?}");
        for (id, request) in &work {
            assert_eq!(observed.emitted[id], request.stops_after);
        }
    }
}

/// A drafter-like schedule: chunks and shared anchors on a page granule,
/// and steps that rewrite rows of the last page. Private preemption anchors
/// still restore exactly at any length (page-aligned ones copy the shared
/// last page only while the anchor still holds it), and pages stay exact
/// under a preemption storm.
#[test]
fn preemption_anchors_compose_with_last_page_rewrites_and_a_granule() {
    let configure = |config: SchedulerConfig| {
        config
            .with_prefill_granule(BLOCK)
            .unwrap()
            .with_last_page_rewrites(true)
    };
    let work: BTreeMap<_, _> = (1..=6)
        .map(|id| (id, work(5 + id as usize, 30, 14 + id as usize)))
        .collect();
    for drafts in [0, 3] {
        let mut scheduler = scheduler_with(6, 64, forced(4, 2), true, configure);
        submit_all(&mut scheduler, &work, true);
        let mut observed = run(&mut scheduler, &work, drafts);
        assert_eq!(observed.preempted.len(), 12);
        observed.preemption_captures.sort_unstable();
        observed.restores.sort_unstable();
        assert_eq!(observed.restores, observed.preemption_captures);
        assert!(
            observed
                .restores
                .iter()
                .any(|(_, tokens)| tokens % BLOCK == 0),
            "{observed:?}"
        );
        assert!(observed.replays.is_empty(), "{observed:?}");
        for (id, request) in &work {
            assert_eq!(observed.emitted[id], request.stops_after);
        }
    }
    let work = storm();
    for drafts in [0, 3] {
        let mut scheduler = scheduler_with(6, 16, optimistic(0, 3), true, configure);
        submit_all(&mut scheduler, &work, true);
        let observed = run(&mut scheduler, &work, drafts);
        assert!(
            observed.preempted.len() >= 4,
            "drafts {drafts}: {observed:?}"
        );
        for (id, request) in &work {
            assert_eq!(observed.emitted[id], request.stops_after);
        }
    }
}

#[test]
fn reserved_admission_never_preempts() {
    let work = storm();
    let mut scheduler = scheduler(6, 16, AdmissionPolicy::Reserved, false);
    submit_all(&mut scheduler, &work, false);
    let observed = run(&mut scheduler, &work, 3);
    assert!(observed.preempted.is_empty());
}

#[test]
fn victims_are_the_latest_arrivals() {
    // Three one-page prompts grow a page at a time until six pages are not
    // enough; the latest arrival goes first.
    let work: BTreeMap<_, _> = (1..=3).map(|id| (id, work(4, 20, 16))).collect();
    let mut scheduler = scheduler(3, 6, optimistic(0, 3), false);
    submit_all(&mut scheduler, &work, false);
    let observed = run(&mut scheduler, &work, 0);
    assert_eq!(observed.preempted.first(), Some(&3));
    assert!(!observed.preempted.contains(&1), "{observed:?}");
}

#[test]
fn a_request_that_cannot_grow_preempts_itself_when_nothing_later_is_preemptible() {
    let mut scheduler = scheduler(2, 6, optimistic(0, 3), false);
    let pinned = Work {
        pinned: true,
        ..work(4, 13, 12)
    };
    submit(&mut scheduler, 1, pinned, false);
    submit(&mut scheduler, 2, work(4, 12, 12), false);
    let mut preempted = Vec::new();
    for _ in 0..40 {
        preempted.extend(scheduler.preempt_for_next_step(0, |_, _| None).unwrap());
        check(&scheduler);
        let Some(batch) = scheduler.next_batch().unwrap() else {
            break;
        };
        let outcomes = batch
            .items()
            .iter()
            .map(|item| {
                let emitted = scheduler.emitted_tokens(item.request_id()).unwrap();
                let count = usize::from(item.expects_output());
                BatchOutcome::success(
                    item.request_id(),
                    item.phase(),
                    count,
                    count > 0 && emitted + count == 12,
                )
            })
            .collect::<Vec<_>>();
        scheduler.commit_batch(batch.id(), &outcomes).unwrap();
        check(&scheduler);
    }
    // The pinned request holds 4 of 6 pages from admission, so the other
    // request outgrows the last two and preempts itself, never the pinned one.
    assert_eq!(preempted.first(), Some(&RequestId::new(2)));
    assert!(!preempted.contains(&RequestId::new(1)));
    assert_eq!(scheduler.preemptions(RequestId::new(1)).unwrap(), 0);
    assert_eq!(
        scheduler.request_state(RequestId::new(1)).unwrap(),
        RequestState::Finished
    );
}

#[test]
fn pinned_requests_reserve_their_worst_case_and_are_never_preempted() {
    // Requests that ask for it, and requests with media, are pinned.
    let mut work = storm();
    for id in [2, 5] {
        work.get_mut(&id).unwrap().pinned = true;
    }
    work.get_mut(&9).unwrap().media = true;
    let mut scheduler = scheduler(6, 16, optimistic(0, 3), false);
    submit(&mut scheduler, 2, work[&2], false);
    scheduler.admit().unwrap();
    assert_eq!(
        scheduler.reserved_pages(),
        (work[&2].prompt + work[&2].maximum - 1).div_ceil(BLOCK)
    );
    for (id, request) in &work {
        if *id != 2 {
            submit(&mut scheduler, *id, *request, false);
        }
    }
    let observed = run(&mut scheduler, &work, 0);
    assert!(!observed.preempted.is_empty());
    for id in [2, 5, 9] {
        assert!(!observed.preempted.contains(&id));
    }
}

#[test]
fn the_starvation_bound_pins_a_request_after_its_last_preemption() {
    let work: BTreeMap<_, _> = (1..=4).map(|id| (id, work(5, 20, 12))).collect();
    for maximum in [0, 1, 2] {
        let mut scheduler = scheduler(4, 32, forced(3, maximum), false);
        submit_all(&mut scheduler, &work, false);
        let observed = run(&mut scheduler, &work, 0);
        for id in work.keys() {
            assert_eq!(
                observed.preempted.iter().filter(|seen| *seen == id).count(),
                maximum,
                "request {id} is preempted exactly the bound, then runs to its end"
            );
        }
    }
    // Under real pressure too, no request passes the bound.
    let work = storm();
    let mut scheduler = scheduler(6, 14, optimistic(0, 1), false);
    submit_all(&mut scheduler, &work, false);
    let observed = run(&mut scheduler, &work, 0);
    assert!(!observed.preempted.is_empty());
    for id in work.keys() {
        assert!(observed.preempted.iter().filter(|seen| *seen == id).count() <= 1);
    }
}

#[test]
fn a_preempted_prefix_is_published_and_the_recompute_restores_it() {
    let work = BTreeMap::from([(1, work(9, 20, 8))]);
    let mut scheduler = scheduler(2, 16, forced(3, 1), true);
    submit_all(&mut scheduler, &work, true);
    let observed = run(&mut scheduler, &work, 0);
    assert_eq!(observed.preempted, [1]);
    // Three emitted tokens: the prompt and the first two are in KV, the
    // third is the pending input.
    assert_eq!(observed.preemption_captures, [(1, 11)]);
    assert_eq!(observed.restores, [(1, 11)]);
    assert!(
        observed.replays.is_empty(),
        "an exact restore never replays"
    );
    assert_eq!(
        scheduler.cached_tokens(RequestId::new(1)).unwrap(),
        0,
        "usage keeps the first admission's cached tokens"
    );
}

#[test]
fn a_preemption_shorter_than_one_page_is_restored_exactly() {
    // One prompt token and one committed output token: two tokens, under
    // the four-token page. Recomputing them would change their logits.
    let work = BTreeMap::from([(1, work(1, 8, 6))]);
    for drafts in [0, 1] {
        let mut scheduler = scheduler(2, 16, forced(2, 1), true);
        submit_all(&mut scheduler, &work, true);
        let observed = run(&mut scheduler, &work, drafts);
        assert_eq!(observed.preempted, [1]);
        let (_, tokens) = observed.preemption_captures[0];
        assert!(tokens < BLOCK, "{observed:?}");
        assert_eq!(observed.restores, observed.preemption_captures);
        assert!(observed.replays.is_empty(), "{observed:?}");
        assert_eq!(observed.emitted[&1], 6);
    }
}

#[test]
fn a_recompute_without_prefix_caching_replays_prompt_and_emitted_tokens() {
    let work = BTreeMap::from([(1, work(9, 20, 8))]);
    let mut scheduler = scheduler(2, 16, forced(3, 1), false);
    submit_all(&mut scheduler, &work, false);
    let observed = run(&mut scheduler, &work, 0);
    assert_eq!(observed.preempted, [1]);
    assert_eq!(observed.replays, [(1, 0, 12)]);
}

#[test]
fn cancelling_a_preempted_or_recomputing_request_releases_everything() {
    for replaying in [false, true] {
        let mut scheduler = scheduler(2, 16, forced(2, 3), false);
        submit(&mut scheduler, 1, work(10, 20, 20), false);
        submit(&mut scheduler, 2, work(3, 5, 5), false);
        let id = RequestId::new(1);
        loop {
            let preempted = scheduler.preempt_for_next_step(0, |_, _| None).unwrap();
            check(&scheduler);
            if preempted.contains(&id) {
                break;
            }
            let batch = scheduler.next_batch().unwrap().unwrap();
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
            scheduler.commit_batch(batch.id(), &outcomes).unwrap();
        }
        assert_eq!(scheduler.request_state(id).unwrap(), RequestState::Queued);
        if replaying {
            // The replay of 12 tokens takes a full chunk, then its tail.
            let batch = scheduler.next_batch().unwrap().unwrap();
            let item = batch
                .items()
                .iter()
                .find(|item| item.request_id() == id)
                .unwrap();
            assert_eq!(item.phase(), BatchPhase::Prefill { tokens: 8 });
            assert!(!item.expects_output());
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
            scheduler.commit_batch(batch.id(), &outcomes).unwrap();
            assert_eq!(
                scheduler.request_state(id).unwrap(),
                RequestState::Prefilling
            );
        }
        scheduler.cancel(id).unwrap();
        check(&scheduler);
        scheduler.cancel(RequestId::new(2)).unwrap();
        check(&scheduler);
        assert_eq!(scheduler.kv_stats().free_blocks(), 16);
        assert_eq!(scheduler.reserved_pages(), 0);
    }
}

#[test]
fn the_watermark_is_kept_while_a_sequence_is_live_and_waived_when_none_is() {
    let mut scheduler = scheduler(4, 10, optimistic(200_000, 3), false);
    submit(&mut scheduler, 1, work(32, 4, 4), false);
    submit(&mut scheduler, 2, work(4, 4, 4), false);
    let report = scheduler.admit().unwrap();
    assert_eq!(report.admitted(), [RequestId::new(1)]);
    assert!(report.blocked().is_some(), "two watermark pages stay free");

    let mut scheduler = self::scheduler(4, 10, optimistic(200_000, 3), false);
    submit(&mut scheduler, 1, work(40, 1, 1), false);
    let report = scheduler.admit().unwrap();
    assert_eq!(
        report.admitted(),
        [RequestId::new(1)],
        "a request that fits the pool is not refused by the watermark"
    );
}

#[test]
fn decode_aware_pressure_finishes_fifo_work_with_exact_accounting() {
    let config = SchedulerConfig::new(8, 64, 8192, 2048, 1)
        .unwrap()
        .with_admission(optimistic(10_000, 3))
        .with_prefill_budget(runtime_scheduler::PrefillBudget::DecodeAware(
            runtime_scheduler::PrefillTarget::DecodeSteps(1),
        ));
    let mut scheduler = Scheduler::new(config, KvGeometry::new(BLOCK, 256).unwrap()).unwrap();
    let work: BTreeMap<_, _> = (1..=16).map(|id| (id, work(64, 256, 256))).collect();
    submit_all(&mut scheduler, &work, false);
    let observed = run(&mut scheduler, &work, 0);
    assert!(!observed.preempted.is_empty());
    assert!(!observed.replays.is_empty());
    assert_eq!(observed.emitted.values().sum::<usize>(), 16 * 256);
}

#[test]
fn exact_restore_decodes_but_shorter_and_refused_restores_recompute() {
    // Both aligned and partial tails, including a preemption after the first
    // output and repeated preemptions, are exercised by the cohort above.
    for (capture_exact, refuse_restore) in [(true, false), (false, false), (true, true)] {
        let mut scheduler = scheduler(1, 32, forced(3, 3), true);
        let id = RequestId::new(1);
        submit(&mut scheduler, 1, work(9, 20, 20), true);
        loop {
            let victims = scheduler
                .preempt_for_next_step(3, |id, length| Some(chain(id.get(), length)))
                .unwrap();
            if !victims.is_empty() {
                assert_eq!(victims, [id]);
                for capture in scheduler.take_prefix_captures() {
                    assert_eq!(capture.tokens(), 11);
                    if capture_exact {
                        scheduler.begin_prefix_capture(&capture).unwrap();
                    }
                    scheduler.end_prefix_capture(&capture).unwrap();
                }
                break;
            }
            let batch = scheduler.next_batch().unwrap().unwrap();
            let item = &batch.items()[0];
            let outcome =
                BatchOutcome::success(id, item.phase(), usize::from(item.expects_output()), false);
            scheduler.commit_batch(batch.id(), &[outcome]).unwrap();
            settle_captures(&mut scheduler);
            check(&scheduler);
        }
        scheduler.admit_with_lookahead(3).unwrap();
        let restores = scheduler.pending_prefix_restores();
        assert_eq!(restores.len(), 1);
        assert_eq!(restores[0].resumes_decoding(), capture_exact);
        assert_eq!(restores[0].tokens(), if capture_exact { 11 } else { 9 });
        assert!(
            scheduler.next_batch().unwrap().is_none(),
            "restore pins block scheduling"
        );
        if refuse_restore {
            scheduler.abandon_prefix_restore(id).unwrap();
        } else {
            scheduler.complete_prefix_restore(id).unwrap();
        }
        check(&scheduler);
        let exact = capture_exact && !refuse_restore;
        assert_eq!(
            scheduler
                .kv_metrics()
                .snapshot()
                .prefix_restore_resumed_decoding,
            u64::from(exact)
        );
        assert!(
            scheduler
                .kv_metrics()
                .snapshot()
                .prometheus()
                .contains(&format!(
                    "chop_prefix_restore_resumed_decoding_total {}",
                    u64::from(exact)
                ))
        );
        assert_eq!(
            scheduler.request_state(id).unwrap(),
            if exact {
                RequestState::Decoding
            } else {
                RequestState::Admitted
            }
        );
        assert_eq!(scheduler.emitted_tokens(id).unwrap(), 3);
        assert_eq!(scheduler.cached_tokens(id).unwrap(), 0);
        let batch = scheduler
            .next_batch_with_drafts(DraftLimits::new(3, 8), |candidates, _| {
                candidates
                    .iter()
                    .map(|candidate| candidate.maximum_draft_tokens())
                    .collect()
            })
            .unwrap()
            .unwrap();
        let append = &batch.kv_plan().appends()[0];
        if exact {
            assert_eq!(
                batch.items()[0].phase(),
                BatchPhase::Verify { draft_tokens: 3 }
            );
            assert_eq!(append.previous_tokens(), 11);
            assert!(
                append.copy().is_none(),
                "the restore consumed its private anchor, so the tail is unshared"
            );
        } else {
            assert!(matches!(
                batch.items()[0].phase(),
                BatchPhase::Prefill { .. }
            ));
            assert_eq!(append.previous_tokens(), if refuse_restore { 0 } else { 9 });
        }
        let item = &batch.items()[0];
        let outcome =
            BatchOutcome::success(id, item.phase(), usize::from(item.expects_output()), false);
        scheduler.commit_batch(batch.id(), &[outcome]).unwrap();
        settle_captures(&mut scheduler);
        assert_eq!(
            scheduler
                .kv_metrics()
                .snapshot()
                .prefix_cache_recomputed_seen_tokens,
            0,
            "a recompute repeats the request's own rows, not an earlier prompt's"
        );
        scheduler.cancel(id).unwrap();
        check(&scheduler);
        assert_eq!(scheduler.reserved_pages(), 0);
        assert_eq!(scheduler.kv_stats().used_blocks(), 0);
    }
}

#[test]
fn a_restore_after_the_first_output_keeps_aligned_and_partial_boundaries() {
    for prompt in [8, 9] {
        for drafts in [0, 3] {
            let work = BTreeMap::from([(1, work(prompt, 20, 10))]);
            let mut scheduler = scheduler(1, 32, forced(1, 2), true);
            submit_all(&mut scheduler, &work, true);
            let observed = run(&mut scheduler, &work, drafts);
            assert_eq!(observed.preempted, [1, 1]);
            assert_eq!(observed.restores[0], (1, prompt));
            assert_eq!(observed.restores, observed.preemption_captures);
            assert!(observed.replays.is_empty());
            assert_eq!(observed.emitted[&1], 10);
        }
    }
}

fn preempt_at_eleven(scheduler: &mut Scheduler) -> RequestId {
    let id = RequestId::new(1);
    submit(scheduler, 1, work(9, 12, 12), true);
    loop {
        let victims = scheduler
            .preempt_for_next_step(0, |id, length| Some(chain(id.get(), length)))
            .unwrap();
        if !victims.is_empty() {
            assert_eq!(victims, [id]);
            assert_eq!(settle_captures(scheduler), [(1, 11)]);
            check(scheduler);
            return id;
        }
        let batch = scheduler.next_batch().unwrap().unwrap();
        let item = &batch.items()[0];
        let outcome =
            BatchOutcome::success(id, item.phase(), usize::from(item.expects_output()), false);
        scheduler.commit_batch(batch.id(), &[outcome]).unwrap();
        settle_captures(scheduler);
        check(scheduler);
    }
}

#[test]
fn optional_draft_lookahead_never_discards_a_fitting_exact_restore() {
    for lookahead in [0, 3, 8] {
        let mut scheduler = scheduler(1, 5, forced(3, 3), true);
        let id = preempt_at_eleven(&mut scheduler);
        assert_eq!(
            scheduler
                .admit_with_lookahead(lookahead)
                .unwrap()
                .admitted(),
            [id]
        );
        let restores = scheduler.pending_prefix_restores();
        assert_eq!(
            restores.len(),
            1,
            "lookahead {lookahead} discarded the anchor"
        );
        assert_eq!(restores[0].tokens(), 11);
        assert!(restores[0].resumes_decoding());
        scheduler.complete_prefix_restore(id).unwrap();
        check(&scheduler);
        let batch = scheduler
            .next_batch_with_drafts(DraftLimits::new(3, 8), |items, _| {
                items
                    .iter()
                    .map(|item| item.maximum_draft_tokens())
                    .collect()
            })
            .unwrap()
            .unwrap();
        // The consumed private anchor leaves the tail unshared, so no copy
        // page is held: three drafts fit a reservation made for them, while
        // the restore without its lookahead (0, or 8, which does not fit)
        // decodes.
        let phase = if lookahead == 3 {
            BatchPhase::Verify { draft_tokens: 3 }
        } else {
            BatchPhase::Decode
        };
        assert_eq!(batch.items()[0].phase(), phase, "lookahead {lookahead}");
        assert_eq!(batch.kv_plan().appends()[0].previous_tokens(), 11);
        assert!(batch.kv_plan().appends()[0].copy().is_none());
        scheduler.rollback_batch(batch.id()).unwrap();
        scheduler.cancel(id).unwrap();
        check(&scheduler);
    }
}

#[test]
fn optional_drafts_do_not_preempt_a_just_restored_boundary() {
    let mut scheduler = scheduler(2, 5, forced(3, 3), true);
    let id = preempt_at_eleven(&mut scheduler);
    scheduler.admit().unwrap();
    scheduler.complete_prefix_restore(id).unwrap();
    check(&scheduler);
    // A pinned request takes the fifth page. Eight optional drafts cannot
    // fit, but the restored request can still decode its next token.
    let pinned = Work {
        pinned: true,
        ..work(1, 3, 3)
    };
    submit(&mut scheduler, 2, pinned, false);
    scheduler.admit().unwrap();
    assert_eq!(
        scheduler
            .preempt_for_next_step(8, |id, n| Some(chain(id.get(), n)))
            .unwrap(),
        []
    );
    scheduler.cancel(RequestId::new(2)).unwrap();
    assert!(settle_captures(&mut scheduler).is_empty());
    assert_eq!(
        scheduler
            .kv_metrics()
            .snapshot()
            .prefix_restore_resumed_decoding,
        1
    );
    let batch = scheduler.next_batch().unwrap().unwrap();
    assert_eq!(batch.items()[0].phase(), BatchPhase::Decode);
    assert_eq!(batch.kv_plan().appends()[0].previous_tokens(), 11);
    scheduler
        .commit_batch(
            batch.id(),
            &[BatchOutcome::success(id, BatchPhase::Decode, 1, false)],
        )
        .unwrap();
    assert_eq!(scheduler.emitted_tokens(id).unwrap(), 4);
    scheduler.cancel(id).unwrap();
    check(&scheduler);
}

#[test]
fn a_private_partial_anchor_restores_in_a_pool_with_no_page_for_a_copy() {
    // Two pages: the preempted request's one KV page and the snapshot page.
    // Its anchor is its own and dropped once restored, so the partial tail
    // becomes exclusive: no copy page is needed, and the restore must not be
    // declined for one.
    let work = BTreeMap::from([(1, work(1, 3, 3))]);
    let mut scheduler = scheduler(1, 2, forced(2, 1), true);
    submit_all(&mut scheduler, &work, true);
    let id = RequestId::new(1);
    loop {
        let victims = scheduler
            .preempt_for_next_step(0, |id, length| Some(chain(id.get(), length)))
            .unwrap();
        if !victims.is_empty() {
            assert_eq!(settle_captures(&mut scheduler), [(1, 2)]);
            break;
        }
        scheduler.admit().unwrap();
        let batch = scheduler.next_batch().unwrap().unwrap();
        let item = &batch.items()[0];
        let outcome =
            BatchOutcome::success(id, item.phase(), usize::from(item.expects_output()), false);
        scheduler.commit_batch(batch.id(), &[outcome]).unwrap();
        settle_captures(&mut scheduler);
        check(&scheduler);
    }
    scheduler.admit().unwrap();
    let restores = scheduler.pending_prefix_restores();
    assert_eq!(restores.len(), 1, "the restore fits the two pages");
    assert_eq!(restores[0].tokens(), 2);
    assert!(restores[0].resumes_decoding());
    scheduler.complete_prefix_restore(id).unwrap();
    check(&scheduler);
    let batch = scheduler.next_batch().unwrap().unwrap();
    assert_eq!(batch.items()[0].phase(), BatchPhase::Decode);
    let append = &batch.kv_plan().appends()[0];
    assert_eq!(append.previous_tokens(), 2);
    assert!(append.copy().is_none(), "the tail is appended in place");
    scheduler
        .commit_batch(
            batch.id(),
            &[BatchOutcome::success(id, BatchPhase::Decode, 1, true)],
        )
        .unwrap();
    check(&scheduler);
    assert_eq!(scheduler.emitted_tokens(id).unwrap(), 3);
}

#[test]
fn a_preemption_anchor_is_private_to_its_request() {
    let mut scheduler = scheduler(2, 16, forced(3, 3), true);
    let id = preempt_at_eleven(&mut scheduler);
    // A request with the same tokens finds no anchor at the preemption
    // boundary, only the public one at the shared prompt's end: another
    // request's KV bits and model state are not its own.
    let other = RequestId::new(2);
    scheduler
        .submit_with_prefix(RequestSpec::new(other, 12, 4).unwrap(), Some(chain(1, 12)))
        .unwrap();
    scheduler.admit().unwrap();
    let restores = scheduler.pending_prefix_restores();
    assert_eq!(restores.len(), 2);
    for restore in &restores {
        let own = restore.request_id() == id;
        assert_eq!(restore.tokens(), if own { 11 } else { 9 });
        assert_eq!(restore.resumes_decoding(), own);
    }
    for restore in restores {
        scheduler
            .complete_prefix_restore(restore.request_id())
            .unwrap();
    }
    check(&scheduler);
    let batch = scheduler.next_batch().unwrap().unwrap();
    for (item, append) in batch.items().iter().zip(batch.kv_plan().appends()) {
        if item.request_id() == id {
            assert_eq!(item.phase(), BatchPhase::Decode);
            assert_eq!(append.previous_tokens(), 11);
        } else {
            assert_eq!(
                append.previous_tokens(),
                9,
                "the other prefills after the prompt"
            );
        }
    }
    scheduler.rollback_batch(batch.id()).unwrap();
    scheduler.cancel(id).unwrap();
    scheduler.cancel(other).unwrap();
    check(&scheduler);
    assert_eq!(
        scheduler.kv_stats().anchors(),
        2,
        "only the prompt's public anchors, at its final chunk and its end"
    );
}

#[test]
fn identical_requests_each_restore_their_own_preemption_state() {
    // Two requests with the same token IDs, below one page when preempted,
    // whose KV and snapshots may differ in bits (other batch shapes): each
    // capture has its own key and pages, and each resume restores its own.
    let mut scheduler = scheduler(2, 32, forced(3, 1), true);
    for id in [1, 2] {
        scheduler
            .submit_with_prefix(
                RequestSpec::new(RequestId::new(id), 1, 8).unwrap(),
                Some(chain(1, 1)),
            )
            .unwrap();
    }
    let mut snapshots = BTreeMap::new();
    while snapshots.len() < 2 {
        scheduler
            .preempt_for_next_step(0, |_, length| Some(chain(1, length)))
            .unwrap();
        for capture in scheduler.take_prefix_captures() {
            assert!(capture.tokens() < BLOCK);
            let pages = scheduler.begin_prefix_capture(&capture).unwrap().unwrap();
            scheduler.end_prefix_capture(&capture).unwrap();
            snapshots.insert(capture.request_id().get(), (capture.tokens(), pages));
        }
        check(&scheduler);
        if snapshots.len() == 2 {
            break;
        }
        scheduler.admit().unwrap();
        let batch = scheduler.next_batch().unwrap().unwrap();
        let outcomes: Vec<_> = batch
            .items()
            .iter()
            .map(|item| {
                let count = usize::from(item.expects_output());
                BatchOutcome::success(item.request_id(), item.phase(), count, false)
            })
            .collect();
        scheduler.commit_batch(batch.id(), &outcomes).unwrap();
        settle_captures(&mut scheduler);
        check(&scheduler);
    }
    assert_ne!(snapshots[&1].1, snapshots[&2].1);
    scheduler.admit().unwrap();
    let restores = scheduler.pending_prefix_restores();
    assert_eq!(restores.len(), 2);
    for restore in &restores {
        let (tokens, pages) = &snapshots[&restore.request_id().get()];
        assert_eq!(restore.tokens(), *tokens);
        assert_eq!(restore.pages(), pages.as_slice(), "{restore:?}");
        assert!(restore.resumes_decoding());
    }
    for restore in restores {
        scheduler
            .complete_prefix_restore(restore.request_id())
            .unwrap();
    }
    check(&scheduler);
    for id in [1, 2] {
        scheduler.cancel(RequestId::new(id)).unwrap();
    }
    check(&scheduler);
    assert_eq!(scheduler.kv_stats().anchors(), 0);
}

#[test]
fn zero_token_victims_recompute_without_capturing() {
    let mut s = scheduler(2, 2, optimistic(0, 3), false);
    s.configure_offload(runtime_scheduler::OffloadConfig {
        minimum_tokens: 0,
        snapshot_pages: 1,
        enabled: true,
        force_host: false,
    });
    let a = RequestId::new(1);
    s.submit(RequestSpec::new(a, 4, 4).unwrap()).unwrap();
    let batch = s.next_batch().unwrap().unwrap();
    let item = &batch.items()[0];
    let outcome = BatchOutcome::success(a, item.phase(), usize::from(item.expects_output()), false);
    s.commit_batch(batch.id(), &[outcome]).unwrap();
    let b = RequestId::new(2);
    s.submit(RequestSpec::new(b, 4, 4).unwrap()).unwrap();
    assert_eq!(s.admit().unwrap().admitted(), &[b]);
    let victims = s.preempt_for_next_step(0, |_, _| None).unwrap();
    assert_eq!(victims, vec![b]);
    assert!(s.take_prefix_captures().is_empty());
    check(&s);
}

#[test]
fn a_ready_host_reload_fits_when_its_own_gpu_tail_does_not() {
    let mut s = scheduler(1, 4, forced(3, 3), false);
    s.configure_offload(runtime_scheduler::OffloadConfig {
        minimum_tokens: 1,
        snapshot_pages: 1,
        enabled: true,
        force_host: false,
    });
    let mut tier = s.new_host_tier(16).unwrap();
    let id = RequestId::new(1);
    submit(&mut s, 1, work(9, 4, 4), false);
    loop {
        let victims = s
            .preempt_for_next_step(0, |id, n| Some(chain(id.get(), n)))
            .unwrap();
        if !victims.is_empty() {
            break;
        }
        let batch = s.next_batch().unwrap().unwrap();
        let item = &batch.items()[0];
        let outcome =
            BatchOutcome::success(id, item.phase(), usize::from(item.expects_output()), false);
        s.commit_batch(batch.id(), &[outcome]).unwrap();
    }
    let capture = s.take_prefix_captures().pop().unwrap();
    assert_eq!(capture.tokens(), 11);
    assert!(s.begin_prefix_capture(&capture).unwrap().is_some());
    let transfer = s
        .offload_capture(&mut tier, &capture, true)
        .unwrap()
        .into_submission();
    s.end_prefix_capture(&capture).unwrap();
    s.accept_kv_transfer(
        &mut tier,
        transfer.completion(runtime_kv::KvTransferStatus::Complete),
    )
    .unwrap();
    assert!(s.has_capture_anchor(&capture));
    assert_eq!(s.admit().unwrap().admitted(), [id]);
    assert_eq!(
        s.pending_reloads(),
        [id],
        "host restore must win over cold recompute"
    );
    assert!(s.pending_prefix_restores().is_empty());
    s.audit_host_tier(&tier).unwrap();
    let reload = s.begin_reload(&mut tier, id).unwrap();
    let incoming = reload.id();
    s.accept_kv_transfer(
        &mut tier,
        reload
            .into_submission()
            .completion(runtime_kv::KvTransferStatus::Complete),
    )
    .unwrap();
    assert!(s.reload_restore(id, incoming).unwrap().resumes_decoding());
    s.complete_reload_restore(id).unwrap();
    s.discard_swap(&mut tier, id).unwrap();
    s.cancel(id).unwrap();
    s.audit_host_tier(&tier).unwrap();
    assert_eq!(s.kv_stats().free_blocks(), 4);
}

#[test]
fn impossible_reload_is_discarded_while_another_request_is_live() {
    let mut s = scheduler(2, 6, forced(3, 1), false);
    s.configure_offload(runtime_scheduler::OffloadConfig {
        minimum_tokens: 0,
        snapshot_pages: 1,
        enabled: true,
        force_host: true,
    });
    let mut tier = s.new_host_tier(32).unwrap();
    submit(
        &mut s,
        1,
        Work {
            pinned: true,
            ..work(1, 12, 12)
        },
        false,
    );
    submit(&mut s, 2, work(9, 16, 16), false);
    loop {
        let victims = s
            .preempt_for_next_step(0, |id, n| Some(chain(id.get(), n)))
            .unwrap();
        if !victims.is_empty() {
            assert_eq!(victims, [RequestId::new(2)]);
            break;
        }
        let batch = s.next_batch().unwrap().unwrap();
        let outcomes: Vec<_> = batch
            .items()
            .iter()
            .map(|i| {
                BatchOutcome::success(
                    i.request_id(),
                    i.phase(),
                    usize::from(i.expects_output()),
                    false,
                )
            })
            .collect();
        s.commit_batch(batch.id(), &outcomes).unwrap();
    }
    let c = s.take_prefix_captures().pop().unwrap();
    s.begin_prefix_capture(&c).unwrap().unwrap();
    let out = s
        .offload_capture(&mut tier, &c, true)
        .unwrap()
        .into_submission();
    s.end_prefix_capture(&c).unwrap();
    s.accept_kv_transfer(
        &mut tier,
        out.completion(runtime_kv::KvTransferStatus::Complete),
    )
    .unwrap();
    assert_eq!(s.kv_stats().active_sequences(), 1);
    assert!(s.admit().unwrap().admitted().is_empty());
    assert!(
        s.swap_needs_discard(),
        "unfit reload must not wait for the live request"
    );
    s.release_unused_swaps(&mut tier).unwrap();
    assert_eq!(tier.used_pages(), 0);
    s.audit_host_tier(&tier).unwrap();
}

/// With steps that rewrite committed rows, an own GPU restore waits while
/// an outgoing copy of its anchor is in flight: the continuation may write
/// any of the anchor's pages (an MTP drafter behind its frontier) that the
/// copy still reads. Once the copy completes it resumes from the GPU anchor.
/// Without rewrites the restore does not wait.
#[test]
fn an_own_restore_that_rewrites_rows_waits_for_its_outgoing_copy() {
    for rewrites in [true, false] {
        let mut s = scheduler_with(1, 16, forced(3, 3), true, |config| {
            config.with_last_page_rewrites(rewrites)
        });
        s.configure_offload(runtime_scheduler::OffloadConfig {
            minimum_tokens: 1,
            snapshot_pages: 1,
            enabled: true,
            force_host: false,
        });
        let mut tier = s.new_host_tier(32).unwrap();
        let id = RequestId::new(1);
        submit(&mut s, 1, work(9, 12, 12), true);
        loop {
            let victims = s
                .preempt_for_next_step(0, |id, n| Some(chain(id.get(), n)))
                .unwrap();
            if !victims.is_empty() {
                break;
            }
            let batch = s.next_batch().unwrap().unwrap();
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
                s.begin_prefix_capture(&capture).unwrap();
                s.end_prefix_capture(&capture).unwrap();
            }
        }
        let capture = s
            .take_prefix_captures()
            .into_iter()
            .find(|capture| capture.request_id() == id)
            .unwrap();
        assert!(s.begin_prefix_capture(&capture).unwrap().is_some());
        let transfer = s
            .offload_capture(&mut tier, &capture, true)
            .unwrap()
            .into_submission();
        s.end_prefix_capture(&capture).unwrap();
        assert!(s.has_capture_anchor(&capture));
        let admitted = s.admit().unwrap().admitted().to_vec();
        if rewrites {
            assert!(admitted.is_empty(), "the copy is still in flight");
            s.accept_kv_transfer(
                &mut tier,
                transfer.completion(runtime_kv::KvTransferStatus::Complete),
            )
            .unwrap();
            assert_eq!(s.admit().unwrap().admitted(), [id]);
        } else {
            assert_eq!(admitted, [id]);
            s.accept_kv_transfer(
                &mut tier,
                transfer.completion(runtime_kv::KvTransferStatus::Complete),
            )
            .unwrap();
        }
        let restores = s.pending_prefix_restores();
        assert_eq!(restores.len(), 1, "the GPU anchor restores");
        assert_eq!(restores[0].tokens(), capture.tokens());
        s.complete_prefix_restore(id).unwrap();
        s.release_unused_swaps(&mut tier).unwrap();
        s.audit_host_tier(&tier).unwrap();
        s.cancel(id).unwrap();
        check(&s);
    }
}
