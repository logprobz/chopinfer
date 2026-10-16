use std::sync::Arc;

use runtime_kv::{KvGeometry, PrefixChainV1, PrefixMediaSpanV1};
use runtime_scheduler::{
    BatchOutcome, BatchPhase, InvalidRequest, PrefixCaptureV1, PrefixRequestV1, RequestId,
    RequestSpec, RequestState, ScheduleError, Scheduler, SchedulerConfig, SubmitError,
};

const STATE_PAGES: usize = 1;

fn scheduler(chunk: usize) -> Scheduler {
    Scheduler::new(
        SchedulerConfig::new(4, 64, 32, chunk, 1)
            .unwrap()
            .with_lone_prefill_expansion(true)
            .with_prefix_cache(STATE_PAGES),
        KvGeometry::new(4, 32).unwrap(),
    )
    .unwrap()
}

fn prompt(length: u32, salt: u32) -> Vec<u32> {
    (0..length).map(|token| 100 + token + salt).collect()
}

fn submit(scheduler: &mut Scheduler, id: u64, tokens: &[u32], output: usize) {
    let chain = PrefixChainV1::new(4, Arc::from(tokens), Vec::new()).unwrap();
    scheduler
        .submit_with_prefix(
            RequestSpec::new(RequestId::new(id), tokens.len(), output).unwrap(),
            Some(PrefixRequestV1::new(Arc::new(chain), true)),
        )
        .unwrap();
}

/// Commits the next batch, ending each item as `terminal` says, and returns
/// its items and the captures it made due.
fn step(
    scheduler: &mut Scheduler,
    terminal: impl Fn(RequestId) -> bool,
) -> (Vec<(u64, BatchPhase, usize)>, Vec<PrefixCaptureV1>) {
    let batch = scheduler.next_batch().unwrap().unwrap();
    let trace = batch
        .items()
        .iter()
        .zip(batch.kv_plan().appends())
        .map(|(item, append)| {
            (
                item.request_id().get(),
                item.phase(),
                append.previous_tokens(),
            )
        })
        .collect();
    let outcomes = batch
        .items()
        .iter()
        .map(|item| {
            BatchOutcome::success(
                item.request_id(),
                item.phase(),
                usize::from(item.expects_output()),
                item.expects_output() && terminal(item.request_id()),
            )
        })
        .collect::<Vec<_>>();
    scheduler.commit_batch(batch.id(), &outcomes).unwrap();
    (trace, scheduler.take_prefix_captures())
}

fn capture_all(scheduler: &mut Scheduler, captures: &[PrefixCaptureV1]) {
    for capture in captures {
        assert!(scheduler.begin_prefix_capture(capture).unwrap().is_some());
        scheduler.end_prefix_capture(capture).unwrap();
    }
}

#[test]
fn a_follow_up_prompt_resumes_after_the_previous_prompt_end() {
    let mut scheduler = scheduler(8);
    let first = prompt(10, 0);
    submit(&mut scheduler, 1, &first, 2);
    let (trace, captures) = step(&mut scheduler, |_| false);
    assert_eq!(trace, [(1, BatchPhase::Prefill { tokens: 8 }, 0)]);
    assert_eq!(
        captures
            .iter()
            .map(PrefixCaptureV1::tokens)
            .collect::<Vec<_>>(),
        [8],
        "the final chunk's start is captured for a repeat"
    );
    capture_all(&mut scheduler, &captures);
    let (trace, captures) = step(&mut scheduler, |_| false);
    assert_eq!(trace, [(1, BatchPhase::Prefill { tokens: 2 }, 8)]);
    assert_eq!(
        captures
            .iter()
            .map(PrefixCaptureV1::tokens)
            .collect::<Vec<_>>(),
        [10],
        "the prompt end is captured"
    );
    capture_all(&mut scheduler, &captures);
    step(&mut scheduler, |_| true);
    assert_eq!(scheduler.kv_stats().used_blocks(), 0);
    assert_eq!(scheduler.kv_stats().anchors(), 2);

    let mut second = first.clone();
    second.extend([1, 2, 3]);
    submit(&mut scheduler, 2, &second, 2);
    scheduler.admit().unwrap();
    let restores = scheduler.pending_prefix_restores();
    assert_eq!(restores.len(), 1);
    assert_eq!(restores[0].tokens(), 10);
    assert_eq!(restores[0].pages().len(), STATE_PAGES);
    assert_eq!(scheduler.cached_tokens(RequestId::new(2)).unwrap(), 10);
    assert_eq!(
        scheduler.next_batch().unwrap(),
        None,
        "an unrestored request is not scheduled"
    );
    scheduler
        .complete_prefix_restore(RequestId::new(2))
        .unwrap();
    scheduler.audit().unwrap();
    let batch = scheduler.next_batch().unwrap().unwrap();
    let append = &batch.kv_plan().appends()[0];
    assert_eq!(batch.items()[0].phase(), BatchPhase::Prefill { tokens: 3 });
    assert_eq!(append.previous_tokens(), 10);
    assert!(append.copy().is_some(), "the shared partial page is copied");
    scheduler.rollback_batch(batch.id()).unwrap();
    let (_, captures) = step(&mut scheduler, |_| false);
    assert_eq!(captures[0].tokens(), 13);
    scheduler.audit().unwrap();
}

#[test]
fn a_shared_block_prefix_is_captured_at_its_end_for_the_next_request() {
    let mut scheduler = scheduler(16);
    let first = prompt(10, 0);
    submit(&mut scheduler, 1, &first, 1);
    let (_, captures) = step(&mut scheduler, |_| true);
    capture_all(&mut scheduler, &captures);
    assert_eq!(
        scheduler.kv_stats().active_sequences(),
        0,
        "a request that finished at its capture releases after it"
    );

    // The second prompt shares two full blocks, then diverges.
    let mut second = first[..8].to_vec();
    second.extend(prompt(6, 50));
    submit(&mut scheduler, 2, &second, 1);
    let (trace, captures) = step(&mut scheduler, |_| true);
    assert_eq!(
        trace,
        [(2, BatchPhase::Prefill { tokens: 8 }, 0)],
        "the chunk ends at the shared prefix"
    );
    assert_eq!(captures[0].tokens(), 8);
    capture_all(&mut scheduler, &captures);
    let (trace, captures) = step(&mut scheduler, |_| true);
    assert_eq!(trace, [(2, BatchPhase::Prefill { tokens: 6 }, 8)]);
    capture_all(&mut scheduler, &captures);

    let mut third = first[..8].to_vec();
    third.extend(prompt(5, 90));
    submit(&mut scheduler, 3, &third, 1);
    scheduler.admit().unwrap();
    assert_eq!(scheduler.cached_tokens(RequestId::new(3)).unwrap(), 8);
    scheduler.audit().unwrap();
}

#[test]
fn a_failed_restore_prefills_the_whole_prompt_and_cancel_releases_pins() {
    let mut scheduler = scheduler(16);
    let first = prompt(8, 0);
    submit(&mut scheduler, 1, &first, 1);
    let (_, captures) = step(&mut scheduler, |_| true);
    capture_all(&mut scheduler, &captures);

    let mut longer = first.clone();
    longer.push(7);
    submit(&mut scheduler, 2, &longer, 1);
    submit(&mut scheduler, 3, &longer, 1);
    scheduler.admit().unwrap();
    assert_eq!(scheduler.pending_prefix_restores().len(), 2);
    scheduler.abandon_prefix_restore(RequestId::new(2)).unwrap();
    assert_eq!(scheduler.cached_tokens(RequestId::new(2)).unwrap(), 0);
    scheduler.cancel(RequestId::new(3)).unwrap();
    scheduler.audit().unwrap();
    let (trace, captures) = step(&mut scheduler, |_| true);
    assert_eq!(trace, [(2, BatchPhase::Prefill { tokens: 9 }, 0)]);
    capture_all(&mut scheduler, &captures);
    assert_eq!(
        scheduler.request_state(RequestId::new(2)).unwrap(),
        RequestState::Finished
    );
    let metrics = scheduler.kv_metrics().snapshot();
    assert_eq!(metrics.prefix_cache_restore_failures, 1);
    assert_eq!(
        (
            metrics.prefix_cache_hits,
            metrics.prefix_cache_reused_tokens
        ),
        (0, 0),
        "neither the refused nor the cancelled restore reused a token"
    );
    submit(&mut scheduler, 4, &longer, 1);
    scheduler.admit().unwrap();
    scheduler
        .complete_prefix_restore(RequestId::new(4))
        .unwrap();
    let metrics = scheduler.kv_metrics().snapshot();
    assert_eq!(
        (
            metrics.prefix_cache_hits,
            metrics.prefix_cache_reused_tokens
        ),
        (1, 8)
    );
    scheduler.audit().unwrap();
}

#[test]
fn a_chain_that_is_not_the_prompt_is_refused() {
    let mut scheduler = scheduler(8);
    let chain = PrefixChainV1::new(4, Arc::from(prompt(6, 0).as_slice()), Vec::new()).unwrap();
    assert_eq!(
        scheduler.submit_with_prefix(
            RequestSpec::new(RequestId::new(1), 5, 1).unwrap(),
            Some(PrefixRequestV1::new(Arc::new(chain), true)),
        ),
        Err(SubmitError::Invalid(InvalidRequest::PrefixMismatch))
    );
    assert_eq!(
        scheduler.complete_prefix_restore(RequestId::new(1)),
        Err(ScheduleError::UnknownRequest(RequestId::new(1)))
    );
}

/// A prompt of one chunk captures only its end, which serves longer prompts
/// only, so an identical prompt misses on its second use, which captures
/// where its blocks end, and hits from its third. The miss counts the rows
/// the first use computed.
#[test]
fn an_identical_prompt_of_one_chunk_hits_from_its_third_use() {
    let mut scheduler = scheduler(8);
    let ids = prompt(7, 0);
    let mut sends = Vec::new();
    for id in 1..=2 {
        submit(&mut scheduler, id, &ids, 1);
        let mut trace = Vec::new();
        for _ in 0..id {
            let (items, captures) = step(&mut scheduler, |_| true);
            trace.extend(items);
            capture_new(&mut scheduler, &captures);
        }
        assert_eq!(scheduler.cached_tokens(RequestId::new(id)).unwrap(), 0);
        sends.push(trace);
    }
    assert_eq!(sends[0], [(1, BatchPhase::Prefill { tokens: 7 }, 0)]);
    assert_eq!(
        sends[1],
        [
            (2, BatchPhase::Prefill { tokens: 4 }, 0),
            (2, BatchPhase::Prefill { tokens: 3 }, 4),
        ],
        "the second use ends a chunk where its full blocks end"
    );
    assert_eq!(recomputed(&scheduler), 4);

    submit(&mut scheduler, 3, &ids, 1);
    scheduler.admit().unwrap();
    assert_eq!(scheduler.cached_tokens(RequestId::new(3)).unwrap(), 4);
    scheduler
        .complete_prefix_restore(RequestId::new(3))
        .unwrap();
    let (trace, _) = step(&mut scheduler, |_| true);
    assert_eq!(trace, [(3, BatchPhase::Prefill { tokens: 3 }, 4)]);
    assert_eq!(recomputed(&scheduler), 4);
    scheduler.audit().unwrap();
}

fn recomputed(scheduler: &Scheduler) -> u64 {
    scheduler
        .kv_metrics()
        .snapshot()
        .prefix_cache_recomputed_seen_tokens
}

/// Identical prompts admitted together prefill side by side: neither
/// computed a block before the other, so neither counts as recomputing.
#[test]
fn identical_prompts_admitted_together_recompute_nothing() {
    let mut scheduler = scheduler(8);
    let ids = prompt(21, 0);
    submit(&mut scheduler, 1, &ids, 1);
    submit(&mut scheduler, 2, &ids, 1);
    scheduler.admit().unwrap();
    assert_eq!(recomputed(&scheduler), 0);
    step_to_idle(&mut scheduler);
    assert_eq!(recomputed(&scheduler), 0);
    scheduler.audit().unwrap();
}

/// Only committed prefill counts, and only against blocks a committed
/// prefill computed: a request cancelled before its first step neither
/// counts nor lets the next one count.
#[test]
fn a_request_cancelled_before_prefill_counts_nothing() {
    let mut scheduler = scheduler(8);
    let ids = prompt(21, 0);
    submit(&mut scheduler, 1, &ids, 1);
    scheduler.admit().unwrap();
    scheduler.cancel(RequestId::new(1)).unwrap();
    submit(&mut scheduler, 2, &ids, 1);
    step_to_idle(&mut scheduler);
    assert_eq!(
        recomputed(&scheduler),
        0,
        "the cancelled request computed nothing"
    );

    submit(&mut scheduler, 3, &ids, 1);
    scheduler.admit().unwrap();
    scheduler.cancel(RequestId::new(3)).unwrap();
    assert_eq!(
        recomputed(&scheduler),
        0,
        "the cancelled repeat prefilled nothing"
    );
    scheduler.audit().unwrap();
}

/// A refused restore prefills the whole prompt again, and counts the blocks
/// earlier requests had computed.
#[test]
fn a_refused_restore_counts_what_it_recomputes() {
    let mut scheduler = scheduler(8);
    let ids = prompt(21, 0);
    submit(&mut scheduler, 1, &ids, 1);
    step_to_idle(&mut scheduler);
    assert_eq!(recomputed(&scheduler), 0);
    submit(&mut scheduler, 2, &ids, 1);
    scheduler.admit().unwrap();
    assert_eq!(scheduler.cached_tokens(RequestId::new(2)).unwrap(), 16);
    scheduler.abandon_prefix_restore(RequestId::new(2)).unwrap();
    step_to_idle(&mut scheduler);
    assert_eq!(recomputed(&scheduler), 20);
    scheduler.audit().unwrap();
}

/// A prompt longer than a chunk also captures its final chunk's start, so
/// an identical prompt hits from its second use and prefills the first
/// send's final chunk whole.
#[test]
fn a_repeated_prompt_resumes_at_its_final_chunk() {
    let mut scheduler = scheduler(8);
    let ids = prompt(21, 0);
    submit(&mut scheduler, 1, &ids, 1);
    let mut cold = Vec::new();
    for _ in 0..2 {
        let (trace, captures) = step(&mut scheduler, |_| true);
        cold.extend(trace);
        capture_all(&mut scheduler, &captures);
    }
    assert_eq!(
        cold,
        [
            (1, BatchPhase::Prefill { tokens: 16 }, 0),
            (1, BatchPhase::Prefill { tokens: 5 }, 16),
        ],
        "the final chunk's anchor adds no chunk"
    );
    assert_eq!(scheduler.kv_stats().anchors(), 2);
    assert_eq!(recomputed(&scheduler), 0);

    // The prompt-end anchor serves longer prompts only; the repeat resumes
    // at the final chunk and prefills it whole, as the first send did.
    submit(&mut scheduler, 2, &ids, 1);
    scheduler.admit().unwrap();
    assert_eq!(scheduler.cached_tokens(RequestId::new(2)).unwrap(), 16);
    scheduler
        .complete_prefix_restore(RequestId::new(2))
        .unwrap();
    let (trace, captures) = step(&mut scheduler, |_| true);
    assert_eq!(trace, [(2, BatchPhase::Prefill { tokens: 5 }, 16)]);
    capture_new(&mut scheduler, &captures);
    assert_eq!(scheduler.kv_stats().anchors(), 2);
    assert_eq!(
        recomputed(&scheduler),
        4,
        "rows 16..20 were computed before but no anchor covered them"
    );

    submit(&mut scheduler, 3, &ids, 1);
    scheduler.admit().unwrap();
    assert_eq!(scheduler.cached_tokens(RequestId::new(3)).unwrap(), 16);
    scheduler
        .complete_prefix_restore(RequestId::new(3))
        .unwrap();
    let (trace, _) = step(&mut scheduler, |_| true);
    assert_eq!(trace, [(3, BatchPhase::Prefill { tokens: 5 }, 16)]);
    assert_eq!(recomputed(&scheduler), 8);

    // A prompt that diverges before the final chunk misses, and still shows
    // the next one where the shared blocks end.
    let mut fork = ids[..12].to_vec();
    fork.extend(prompt(9, 500));
    submit(&mut scheduler, 4, &fork, 1);
    let mut trace = Vec::new();
    for _ in 0..3 {
        let (items, captures) = step(&mut scheduler, |_| true);
        trace.extend(items);
        capture_new(&mut scheduler, &captures);
    }
    assert_eq!(scheduler.cached_tokens(RequestId::new(4)).unwrap(), 0);
    assert_eq!(
        trace,
        [
            (4, BatchPhase::Prefill { tokens: 12 }, 0),
            (4, BatchPhase::Prefill { tokens: 8 }, 12),
            (4, BatchPhase::Prefill { tokens: 1 }, 20),
        ],
        "chunks end where the shared blocks end and restart there"
    );
    scheduler.audit().unwrap();
}

/// Like `capture_all`, but a capture whose anchor already exists is skipped,
/// as the engine skips it.
fn capture_new(scheduler: &mut Scheduler, captures: &[PrefixCaptureV1]) {
    for capture in captures {
        scheduler.begin_prefix_capture(capture).unwrap();
        scheduler.end_prefix_capture(capture).unwrap();
    }
}

fn submit_with_media(
    scheduler: &mut Scheduler,
    id: u64,
    tokens: &[u32],
    media: Vec<PrefixMediaSpanV1>,
) {
    let chain = PrefixChainV1::new(4, Arc::from(tokens), media).unwrap();
    scheduler
        .submit_with_prefix(
            RequestSpec::new(RequestId::new(id), tokens.len(), 1).unwrap(),
            Some(PrefixRequestV1::new(Arc::new(chain), true)),
        )
        .unwrap();
}

/// The three arms of `tools/qual/prefix_exact.py` for one scored position `p`
/// with anchor `B`: a primer that shares `[0, B)` and then diverges, a cold
/// prefill that the divergence capture ends a chunk at `B`, and a hit at `B`
/// that prefills `[B, p)` as one chunk, like the cold arm's second chunk.
#[test]
fn a_primed_prompt_is_chunked_at_the_anchor_and_then_hits_it() {
    let mut scheduler = scheduler(16);
    let ids = prompt(12, 0);
    let mut primer = ids[..4].to_vec();
    primer.push(7);
    submit(&mut scheduler, 1, &primer, 1);
    let (_, captures) = step(&mut scheduler, |_| true);
    capture_new(&mut scheduler, &captures);

    submit(&mut scheduler, 2, &ids[..7], 1);
    let (trace, captures) = step(&mut scheduler, |_| true);
    assert_eq!(trace, [(2, BatchPhase::Prefill { tokens: 4 }, 0)]);
    assert_eq!(scheduler.cached_tokens(RequestId::new(2)).unwrap(), 0);
    capture_new(&mut scheduler, &captures);
    let (trace, captures) = step(&mut scheduler, |_| true);
    assert_eq!(trace, [(2, BatchPhase::Prefill { tokens: 3 }, 4)]);
    capture_new(&mut scheduler, &captures);

    submit(&mut scheduler, 3, &ids[..7], 1);
    scheduler.admit().unwrap();
    assert_eq!(scheduler.cached_tokens(RequestId::new(3)).unwrap(), 4);
    scheduler
        .complete_prefix_restore(RequestId::new(3))
        .unwrap();
    let (trace, captures) = step(&mut scheduler, |_| true);
    assert_eq!(trace, [(3, BatchPhase::Prefill { tokens: 3 }, 4)]);
    capture_new(&mut scheduler, &captures);
    scheduler.audit().unwrap();

    // A prompt that ends past the next block boundary: the cold arm saw that
    // block too, but the hit resumes within one chunk of its end and
    // prefills it whole, as the cold arm did from the same boundary.
    let ids = prompt(12, 1000);
    let mut primer = ids[..4].to_vec();
    primer.push(7);
    submit(&mut scheduler, 4, &primer, 1);
    let (_, captures) = step(&mut scheduler, |_| true);
    capture_new(&mut scheduler, &captures);
    submit(&mut scheduler, 5, &ids[..11], 1);
    for _ in 0..2 {
        let (_, captures) = step(&mut scheduler, |_| true);
        capture_new(&mut scheduler, &captures);
    }
    submit(&mut scheduler, 6, &ids[..11], 1);
    scheduler.admit().unwrap();
    assert_eq!(scheduler.cached_tokens(RequestId::new(6)).unwrap(), 4);
    scheduler
        .complete_prefix_restore(RequestId::new(6))
        .unwrap();
    let (trace, _) = step(&mut scheduler, |_| true);
    assert_eq!(trace, [(6, BatchPhase::Prefill { tokens: 7 }, 4)]);
}

#[test]
fn the_same_tokens_over_different_media_share_no_anchor_past_the_media() {
    let mut scheduler = scheduler(16);
    let ids = prompt(12, 0);
    let image = |identity: u8| {
        vec![PrefixMediaSpanV1 {
            identity: [identity; 32],
            start: 2,
            rows: 4,
            media_row: 0,
        }]
    };
    for id in 1..=2 {
        submit_with_media(&mut scheduler, id, &ids, image(1));
        loop {
            let (_, captures) = step(&mut scheduler, |_| true);
            capture_new(&mut scheduler, &captures);
            if scheduler.request_state(RequestId::new(id)).unwrap() == RequestState::Finished {
                break;
            }
        }
    }
    submit_with_media(&mut scheduler, 3, &ids, image(1));
    scheduler.admit().unwrap();
    assert_eq!(
        scheduler.cached_tokens(RequestId::new(3)).unwrap(),
        8,
        "the same media hits the anchor the second request captured"
    );
    scheduler
        .complete_prefix_restore(RequestId::new(3))
        .unwrap();
    step(&mut scheduler, |_| true);

    submit_with_media(&mut scheduler, 4, &ids, image(2));
    scheduler.admit().unwrap();
    assert_eq!(scheduler.cached_tokens(RequestId::new(4)).unwrap(), 0);
    let (trace, _) = step(&mut scheduler, |_| true);
    assert_eq!(
        trace,
        [(4, BatchPhase::Prefill { tokens: 12 }, 0)],
        "no block of the other media was seen, so nothing caps the chunk"
    );
    scheduler.audit().unwrap();
}

#[test]
fn a_shared_prefix_that_ends_inside_media_is_not_captured() {
    let mut scheduler = scheduler(16);
    let ids = prompt(12, 0);
    let image = vec![PrefixMediaSpanV1 {
        identity: [1; 32],
        start: 2,
        rows: 8,
        media_row: 0,
    }];
    submit_with_media(&mut scheduler, 1, &ids, image.clone());
    let (_, captures) = step(&mut scheduler, |_| true);
    capture_new(&mut scheduler, &captures);

    // Two full blocks are shared and end at 8, inside the span [2, 10).
    let mut longer = ids[..11].to_vec();
    longer[10] = 7;
    submit_with_media(&mut scheduler, 2, &longer, image);
    let (trace, captures) = step(&mut scheduler, |_| true);
    assert_eq!(trace, [(2, BatchPhase::Prefill { tokens: 11 }, 0)]);
    assert_eq!(
        captures
            .iter()
            .map(PrefixCaptureV1::tokens)
            .collect::<Vec<_>>(),
        [11],
        "only the prompt end is captured"
    );
    scheduler.audit().unwrap();
}

/// Four pages of four tokens, one of them a snapshot page once an anchor
/// exists.
fn small_scheduler() -> Scheduler {
    Scheduler::new(
        SchedulerConfig::new(4, 64, 32, 16, 1)
            .unwrap()
            .with_prefix_cache(STATE_PAGES),
        KvGeometry::new(4, 4).unwrap(),
    )
    .unwrap()
}

/// Leaves an anchor at the end of a finished four-token prompt.
fn anchored(scheduler: &mut Scheduler) -> Vec<u32> {
    let first = prompt(4, 0);
    submit(scheduler, 1, &first, 1);
    let (_, captures) = step(scheduler, |_| true);
    capture_all(scheduler, &captures);
    step_to_idle(scheduler);
    assert_eq!(scheduler.kv_stats().anchors(), 1);
    first
}

fn step_to_idle(scheduler: &mut Scheduler) {
    for _ in 0..16 {
        scheduler.admit().unwrap();
        if scheduler.kv_stats().active_sequences() == 0 {
            return;
        }
        let (_, captures) = step(scheduler, |_| true);
        capture_new(scheduler, &captures);
    }
    panic!("the scheduler did not drain");
}

#[test]
fn a_hit_whose_pins_do_not_fit_runs_cold_and_the_queue_moves() {
    let mut scheduler = small_scheduler();
    let first = anchored(&mut scheduler);
    // Thirteen KV tokens fill all four pages cold, leaving no room for the
    // hit's snapshot pin.
    let mut long = first.clone();
    long.extend(prompt(9, 40));
    submit(&mut scheduler, 2, &long, 1);
    submit(&mut scheduler, 3, &prompt(3, 80), 1);
    let report = scheduler.admit().unwrap();
    assert_eq!(report.admitted(), [RequestId::new(2)]);
    assert!(scheduler.pending_prefix_restores().is_empty());
    assert_eq!(scheduler.cached_tokens(RequestId::new(2)).unwrap(), 0);
    let (trace, captures) = step(&mut scheduler, |_| true);
    assert_eq!(
        trace,
        [(2, BatchPhase::Prefill { tokens: 13 }, 0)],
        "the cold prefill evicts the anchor for its pages"
    );
    assert_eq!(scheduler.kv_stats().anchors(), 0);
    capture_new(&mut scheduler, &captures);
    step_to_idle(&mut scheduler);
    assert_eq!(
        scheduler.request_state(RequestId::new(3)).unwrap(),
        RequestState::Finished,
        "the follower is admitted after it"
    );
    let metrics = scheduler.kv_metrics().snapshot();
    assert_eq!(metrics.prefix_cache_hits, 0);
    assert_eq!(metrics.prefix_cache_reused_tokens, 0);
    scheduler.audit().unwrap();
}

#[test]
fn cancelling_a_bypassed_hit_frees_its_reservation_for_the_queue() {
    let mut scheduler = small_scheduler();
    let first = anchored(&mut scheduler);
    let mut long = first.clone();
    long.extend(prompt(9, 40));
    submit(&mut scheduler, 2, &long, 1);
    submit(&mut scheduler, 3, &prompt(12, 80), 1);
    assert_eq!(scheduler.admit().unwrap().admitted(), [RequestId::new(2)]);
    scheduler.cancel(RequestId::new(2)).unwrap();
    assert_eq!(scheduler.admit().unwrap().admitted(), [RequestId::new(3)]);
    step_to_idle(&mut scheduler);
    assert_eq!(
        scheduler.request_state(RequestId::new(3)).unwrap(),
        RequestState::Finished
    );
    assert_eq!(scheduler.kv_stats().used_blocks(), 0);
    scheduler.audit().unwrap();
}

#[test]
fn hits_that_all_copy_an_evicted_anchor_tail_return_every_page() {
    // Seven pages of four tokens, three sequences, one snapshot page.
    let mut scheduler = Scheduler::new(
        SchedulerConfig::new(3, 16, 32, 8, 1)
            .unwrap()
            .with_prefix_cache(STATE_PAGES),
        KvGeometry::new(4, 7).unwrap(),
    )
    .unwrap();
    let warm = prompt(5, 0);
    submit(&mut scheduler, 1, &warm, 1);
    let (_, captures) = step(&mut scheduler, |_| true);
    capture_all(&mut scheduler, &captures);
    assert_eq!(scheduler.kv_stats().anchors(), 1);

    let mut extended = warm.clone();
    extended.push(7);
    submit(&mut scheduler, 2, &extended, 1);
    submit(&mut scheduler, 3, &extended, 1);
    for _ in 0..2 {
        scheduler.admit().unwrap();
        for restore in scheduler.pending_prefix_restores() {
            scheduler
                .complete_prefix_restore(restore.request_id())
                .unwrap();
        }
    }
    assert_eq!(scheduler.cached_tokens(RequestId::new(3)).unwrap(), 5);
    // Dropping the anchor, as eviction does, leaves the two hits the only
    // owners of the tail, so both copy it in one batch.
    scheduler.abandon_prefix_capture(&captures[0]).unwrap();
    assert_eq!(scheduler.kv_stats().anchors(), 0);
    let batch = scheduler.next_batch().unwrap().unwrap();
    assert!(
        batch
            .kv_plan()
            .appends()
            .iter()
            .all(|append| append.copy().is_some())
    );
    scheduler.rollback_batch(batch.id()).unwrap();
    let (_, captures) = step(&mut scheduler, |_| true);
    for capture in &captures {
        scheduler.end_prefix_capture(capture).unwrap();
    }
    scheduler.audit().unwrap();
    assert_eq!(scheduler.kv_stats().active_sequences(), 0);
    assert_eq!(scheduler.kv_stats().free_blocks(), 7);

    // A request as long as the pool still fits.
    submit(&mut scheduler, 5, &prompt(27, 80), 2);
    step_to_idle(&mut scheduler);
    assert_eq!(
        scheduler.request_state(RequestId::new(5)).unwrap(),
        RequestState::Finished
    );
    scheduler.audit().unwrap();
}

/// 128 pages of 64 tokens with one snapshot page per anchor.
fn paged_scheduler(sequences: usize) -> Scheduler {
    Scheduler::new(
        SchedulerConfig::new(sequences, 256, 2048, 2048, 1)
            .unwrap()
            .with_prefix_cache(STATE_PAGES),
        KvGeometry::new(64, 128).unwrap(),
    )
    .unwrap()
}

fn submit_paged(scheduler: &mut Scheduler, id: u64, tokens: &[u32], output: usize) {
    let chain = PrefixChainV1::new(64, Arc::from(tokens), Vec::new()).unwrap();
    scheduler
        .submit_with_prefix(
            RequestSpec::new(RequestId::new(id), tokens.len(), output).unwrap(),
            Some(PrefixRequestV1::new(Arc::new(chain), true)),
        )
        .unwrap();
}

/// Leaves an anchor over a finished 1024-token prompt, 16 full pages, and
/// returns the prompt and its capture.
fn paged_anchor(scheduler: &mut Scheduler) -> (Vec<u32>, PrefixCaptureV1) {
    let warm = prompt(1024, 0);
    submit_paged(scheduler, 1, &warm, 1);
    let mut captures = Vec::new();
    while scheduler.kv_stats().anchors() == 0 {
        let (_, due) = step(scheduler, |_| true);
        capture_all(scheduler, &due);
        captures.extend(due);
    }
    scheduler.reap_terminal(16).unwrap();
    (warm, captures.pop().unwrap())
}

/// Ends every pending restore, successfully or not.
fn end_restores(scheduler: &mut Scheduler, restored: bool) {
    for restore in scheduler.pending_prefix_restores() {
        if restored {
            scheduler
                .complete_prefix_restore(restore.request_id())
                .unwrap();
        } else {
            scheduler
                .abandon_prefix_restore(restore.request_id())
                .unwrap();
        }
    }
}

/// Admits and restores until the queue stops moving, returning how many
/// requests it admitted.
fn admit_and_restore(scheduler: &mut Scheduler, restored: bool) -> usize {
    let mut admitted = 0;
    loop {
        let report = scheduler.admit().unwrap();
        end_restores(scheduler, restored);
        scheduler.audit().unwrap();
        if report.admitted().is_empty() {
            return admitted;
        }
        admitted += report.admitted().len();
    }
}

/// Runs every live request to its full output limit, checking that the live
/// pages never exceed the shared prefix plus each request's private pages.
fn run_to_limits(scheduler: &mut Scheduler, bound: usize, restored: bool) {
    for _ in 0..4096 {
        end_restores(scheduler, restored);
        if scheduler.kv_stats().active_sequences() == 0 {
            return;
        }
        let (_, captures) = step(scheduler, |_| false);
        capture_new(scheduler, &captures);
        scheduler.audit().unwrap();
        assert!(scheduler.kv_stats().used_blocks() <= bound);
    }
    panic!("the scheduler did not drain");
}

#[test]
fn restored_hits_share_one_reservation_for_their_common_prefix() {
    for sequences in [16, 32] {
        let mut scheduler = paged_scheduler(sequences);
        let (mut shared, _) = paged_anchor(&mut scheduler);
        shared.push(777);
        for id in 0..sequences as u64 {
            submit_paged(&mut scheduler, 10 + id, &shared, 128);
        }
        // Each request may write 1152 tokens, 18 pages: the 16 it shares and
        // 2 of its own.
        assert_eq!(admit_and_restore(&mut scheduler, true), sequences);
        run_to_limits(&mut scheduler, 16 + 2 * sequences, true);
        for id in 0..sequences as u64 {
            let id = RequestId::new(10 + id);
            assert_eq!(scheduler.emitted_tokens(id).unwrap(), 128);
            assert_eq!(scheduler.request_state(id).unwrap(), RequestState::Finished);
        }
        assert_eq!(scheduler.kv_stats().used_blocks(), 0);
        scheduler.audit().unwrap();
    }
}

#[test]
fn shared_prefix_credit_survives_eviction_and_cancellation() {
    let mut scheduler = paged_scheduler(32);
    let (mut shared, warm_capture) = paged_anchor(&mut scheduler);
    shared.push(777);
    for id in 0..32 {
        submit_paged(&mut scheduler, 10 + id, &shared, 128);
    }
    assert_eq!(admit_and_restore(&mut scheduler, true), 32);
    // The live requests keep the prefix pages after their anchor is gone.
    scheduler.abandon_prefix_capture(&warm_capture).unwrap();
    assert_eq!(scheduler.kv_stats().anchors(), 0);
    step(&mut scheduler, |_| false);
    scheduler.audit().unwrap();
    for id in [10, 11, 12, 13] {
        scheduler.cancel(RequestId::new(id)).unwrap();
    }
    scheduler.audit().unwrap();
    // Cancelling four owners frees their private pages but not the prefix,
    // which the other 28 still hold: 128 - 16 - 2 * 28 = 56 pages remain,
    // enough for three cold requests of 18 pages.
    for id in 100..104 {
        submit_paged(&mut scheduler, id, &prompt(1025, 5000 + id as u32), 128);
    }
    assert_eq!(admit_and_restore(&mut scheduler, true), 3);
    // The fourth is admitted once the first finish.
    run_to_limits(&mut scheduler, 128, true);
    for id in (14..42).chain(100..104) {
        assert_eq!(
            scheduler.request_state(RequestId::new(id)).unwrap(),
            RequestState::Finished
        );
    }
    assert_eq!(scheduler.kv_stats().used_blocks(), 0);
    scheduler.audit().unwrap();
}

#[test]
fn a_failed_restore_keeps_its_cold_reservation() {
    let mut scheduler = paged_scheduler(16);
    let (mut shared, _) = paged_anchor(&mut scheduler);
    shared.push(777);
    for id in 0..16 {
        submit_paged(&mut scheduler, 10 + id, &shared, 128);
    }
    // Every restore fails, so each request prefills its own 18 pages: seven
    // fit beside the anchor's snapshot pin of the next hit.
    assert_eq!(admit_and_restore(&mut scheduler, false), 7);
    run_to_limits(&mut scheduler, 128, false);
    for id in 10..26 {
        assert_eq!(
            scheduler.request_state(RequestId::new(id)).unwrap(),
            RequestState::Finished
        );
    }
    scheduler.audit().unwrap();
}

/// Admits and restores three times, returning what was admitted.
fn admit_rounds(scheduler: &mut Scheduler) -> Vec<RequestId> {
    let mut admitted = Vec::new();
    for _ in 0..3 {
        admitted.extend_from_slice(scheduler.admit().unwrap().admitted());
        end_restores(scheduler, true);
        scheduler.audit().unwrap();
    }
    admitted
}

/// Every item of the next batch, and whether its append copies a shared tail.
fn next_copies(scheduler: &mut Scheduler) -> Vec<(u64, bool)> {
    let batch = scheduler.next_batch().unwrap().unwrap();
    let copies = batch
        .items()
        .iter()
        .zip(batch.kv_plan().appends())
        .map(|(item, append)| (item.request_id().get(), append.copy().is_some()))
        .collect();
    scheduler.rollback_batch(batch.id()).unwrap();
    copies
}

fn small_prefix_scheduler(pages: usize) -> Scheduler {
    Scheduler::new(
        SchedulerConfig::new(6, 16, 32, 8, 1)
            .unwrap()
            .with_prefix_cache(STATE_PAGES),
        KvGeometry::new(4, pages).unwrap(),
    )
    .unwrap()
}

#[test]
fn hits_that_share_a_partial_tail_reserve_its_copy_until_they_write() {
    // Seven pages of four tokens. The anchor covers five tokens: one full
    // page and a partial tail both hits share.
    let mut scheduler = small_prefix_scheduler(7);
    let warm = prompt(5, 0);
    submit(&mut scheduler, 1, &warm, 1);
    let (_, captures) = step(&mut scheduler, |_| true);
    capture_all(&mut scheduler, &captures);
    let mut extended = warm.clone();
    extended.push(7);
    submit(&mut scheduler, 2, &extended, 3);
    submit(&mut scheduler, 3, &extended, 3);
    for id in 4..8 {
        submit(&mut scheduler, id, &prompt(2, 40 + id as u32), 1);
    }
    // The shared page, two pages per hit (its copy of the tail and one more)
    // and two cold pages fill the pool: the old tail stays held while both
    // hits copy it.
    assert_eq!(
        admit_rounds(&mut scheduler),
        [2, 3, 4, 5].map(RequestId::new)
    );
    assert_eq!(
        next_copies(&mut scheduler),
        [(2, true), (3, true), (4, false), (5, false)]
    );
    step_to_idle(&mut scheduler);
    for id in 2..8 {
        assert_eq!(
            scheduler.request_state(RequestId::new(id)).unwrap(),
            RequestState::Finished
        );
    }
    assert_eq!(scheduler.kv_stats().used_blocks(), 0);
    scheduler.audit().unwrap();
}

#[test]
fn a_hit_on_a_live_requests_tail_reserves_its_copy() {
    // Six pages of four tokens. Request 2 resumes from a one-page anchor,
    // writes a partial page of its own and anchors it; request 3 resumes
    // there, so both hold that page and both copy it in one batch.
    let mut scheduler = small_prefix_scheduler(6);
    let warm = prompt(4, 0);
    submit(&mut scheduler, 1, &warm, 1);
    let (_, captures) = step(&mut scheduler, |_| true);
    capture_all(&mut scheduler, &captures);
    let mut second = warm.clone();
    second.push(7);
    submit(&mut scheduler, 2, &second, 4);
    assert_eq!(admit_rounds(&mut scheduler), [RequestId::new(2)]);
    let (_, captures) = step(&mut scheduler, |_| false);
    capture_all(&mut scheduler, &captures);
    let mut third = second.clone();
    third.push(9);
    submit(&mut scheduler, 3, &third, 3);
    for id in 4..7 {
        submit(&mut scheduler, id, &prompt(2, 40 + id as u32), 1);
    }
    assert_eq!(admit_rounds(&mut scheduler), [3, 4, 5].map(RequestId::new));
    assert_eq!(
        next_copies(&mut scheduler),
        [(2, true), (3, true), (4, false), (5, false)]
    );
    step_to_idle(&mut scheduler);
    for id in 2..7 {
        assert_eq!(
            scheduler.request_state(RequestId::new(id)).unwrap(),
            RequestState::Finished
        );
    }
    assert_eq!(scheduler.kv_stats().used_blocks(), 0);
    scheduler.audit().unwrap();
}

#[test]
fn hits_on_a_page_boundary_copy_the_last_page_when_requests_rewrite_it() {
    // Seven pages of four tokens. The anchor covers one full page. When
    // steps rewrite rows of the last page (a drafter's pending row), both
    // hits copy it like a partial tail and hold the old page until they
    // write: three pages each, so one cold request fits. Without rewrites
    // they share it and every cold request fits.
    for (rewrites, admitted, copies) in [
        (true, vec![2, 3, 4], vec![(2, true), (3, true), (4, false)]),
        (
            false,
            vec![2, 3, 4, 5, 6, 7],
            (2..8).map(|id| (id, false)).collect(),
        ),
    ] {
        let mut scheduler = Scheduler::new(
            SchedulerConfig::new(6, 16, 32, 8, 1)
                .unwrap()
                .with_prefix_cache(STATE_PAGES)
                .with_last_page_rewrites(rewrites),
            KvGeometry::new(4, 7).unwrap(),
        )
        .unwrap();
        let warm = prompt(4, 0);
        submit(&mut scheduler, 1, &warm, 1);
        let (_, captures) = step(&mut scheduler, |_| true);
        capture_all(&mut scheduler, &captures);
        let mut extended = warm.clone();
        extended.push(7);
        submit(&mut scheduler, 2, &extended, 3);
        submit(&mut scheduler, 3, &extended, 3);
        for id in 4..8 {
            submit(&mut scheduler, id, &prompt(2, 40 + id as u32), 1);
        }
        assert_eq!(
            admit_rounds(&mut scheduler),
            admitted.into_iter().map(RequestId::new).collect::<Vec<_>>(),
            "rewrites {rewrites}"
        );
        assert_eq!(next_copies(&mut scheduler), copies, "rewrites {rewrites}");
        step_to_idle(&mut scheduler);
        for id in 2..8 {
            assert_eq!(
                scheduler.request_state(RequestId::new(id)).unwrap(),
                RequestState::Finished
            );
        }
        assert_eq!(scheduler.kv_stats().used_blocks(), 0);
        scheduler.audit().unwrap();
    }
}

/// A scheduler whose chunks and anchors sit on multiples of 8 tokens, with
/// 4-token pages.
fn granule_scheduler(chunk: usize) -> Scheduler {
    Scheduler::new(
        SchedulerConfig::new(4, 64, 64, chunk, 1)
            .unwrap()
            .with_lone_prefill_expansion(true)
            .with_prefill_granule(8)
            .unwrap()
            .with_prefix_cache(STATE_PAGES),
        KvGeometry::new(4, 64).unwrap(),
    )
    .unwrap()
}

#[test]
fn a_granule_must_divide_the_chunk_limit() {
    let config = SchedulerConfig::new(4, 64, 64, 16, 1).unwrap();
    assert!(config.with_prefill_granule(0).is_err());
    assert!(config.with_prefill_granule(3).is_err());
    assert_eq!(config.prefill_granule(), 1);
    assert_eq!(config.with_prefill_granule(8).unwrap().prefill_granule(), 8);
    // A bounded scored-row limit ends scored chunks, so it is a multiple too,
    // whichever limit is set first.
    let scored = config.with_maximum_scored_rows(12).unwrap();
    assert!(scored.with_prefill_granule(8).is_err());
    assert!(scored.with_prefill_granule(4).is_ok());
    let granule = config.with_prefill_granule(8).unwrap();
    assert!(granule.with_maximum_scored_rows(12).is_err());
    assert!(granule.with_maximum_scored_rows(16).is_ok());
}

#[test]
fn a_granule_ends_chunks_and_the_prompt_anchor_on_its_multiples() {
    let mut scheduler = granule_scheduler(16);
    let first = prompt(37, 0);
    submit(&mut scheduler, 1, &first, 2);
    let mut chunks = Vec::new();
    let mut captured = Vec::new();
    while chunks.iter().map(|(tokens, _)| tokens).sum::<usize>() < first.len() {
        let (trace, captures) = step(&mut scheduler, |_| false);
        for (_, phase, previous) in trace {
            let BatchPhase::Prefill { tokens } = phase else {
                panic!("a prefill chunk")
            };
            chunks.push((tokens, previous));
        }
        captured.extend(captures.iter().map(PrefixCaptureV1::tokens));
        capture_all(&mut scheduler, &captures);
    }
    // The prompt-end anchor moves to 32, the last multiple of the granule,
    // and ends a chunk there; only the prompt's own end is off the grid.
    assert_eq!(chunks, [(32, 0), (5, 32)]);
    assert_eq!(captured, [32]);
    step(&mut scheduler, |_| true);

    // A follow-up resumes at 32 and its chunks keep to the grid.
    let mut second = first.clone();
    second.extend(prompt(30, 500));
    submit(&mut scheduler, 2, &second, 2);
    scheduler.admit().unwrap();
    assert_eq!(scheduler.cached_tokens(RequestId::new(2)).unwrap(), 32);
    scheduler
        .complete_prefix_restore(RequestId::new(2))
        .unwrap();
    let mut chunks = Vec::new();
    while chunks.iter().map(|(tokens, _)| tokens).sum::<usize>() < second.len() - 32 {
        let (trace, captures) = step(&mut scheduler, |_| false);
        for (_, phase, previous) in trace {
            let BatchPhase::Prefill { tokens } = phase else {
                panic!("a prefill chunk")
            };
            chunks.push((tokens, previous));
        }
        capture_all(&mut scheduler, &captures);
    }
    assert_eq!(chunks, [(32, 32), (3, 64)]);
    scheduler.audit().unwrap();
}

/// A prompt that ends on the granule keeps its end anchor there, which only
/// longer prompts can use; its final chunk's start, a multiple of the
/// granule, serves the repeat.
#[test]
fn a_repeat_of_a_prompt_on_the_granule_resumes_at_its_final_chunk() {
    let mut scheduler = granule_scheduler(16);
    let ids = prompt(40, 0);
    let mut sends = Vec::new();
    for id in 1..=2 {
        submit(&mut scheduler, id, &ids, 1);
        scheduler.admit().unwrap();
        if id == 2 {
            assert_eq!(scheduler.cached_tokens(RequestId::new(2)).unwrap(), 32);
            scheduler
                .complete_prefix_restore(RequestId::new(2))
                .unwrap();
        }
        let mut chunks = Vec::new();
        let mut captured = Vec::new();
        while scheduler.request_state(RequestId::new(id)).unwrap() != RequestState::Finished {
            let (trace, captures) = step(&mut scheduler, |_| true);
            for (_, phase, previous) in trace {
                let BatchPhase::Prefill { tokens } = phase else {
                    panic!("a prefill chunk")
                };
                chunks.push((tokens, previous));
            }
            captured.extend(captures.iter().map(PrefixCaptureV1::tokens));
            capture_new(&mut scheduler, &captures);
        }
        sends.push((chunks, captured));
    }
    assert_eq!(sends[0], (vec![(32, 0), (8, 32)], vec![32, 40]));
    assert_eq!(
        sends[1],
        (vec![(8, 32)], vec![40]),
        "the repeat prefills the first send's final chunk"
    );
    scheduler.audit().unwrap();
}

#[test]
fn a_failed_restore_with_a_granule_captures_on_its_multiples() {
    let mut scheduler = granule_scheduler(16);
    let first = prompt(16, 0);
    submit(&mut scheduler, 1, &first, 1);
    let (_, captures) = step(&mut scheduler, |_| true);
    capture_all(&mut scheduler, &captures);

    let mut longer = first.clone();
    longer.extend(prompt(9, 300));
    submit(&mut scheduler, 2, &longer, 1);
    scheduler.admit().unwrap();
    assert_eq!(scheduler.cached_tokens(RequestId::new(2)).unwrap(), 16);
    scheduler.abandon_prefix_restore(RequestId::new(2)).unwrap();
    let mut chunks = Vec::new();
    let mut captured = Vec::new();
    while chunks.iter().map(|(tokens, _)| tokens).sum::<usize>() < longer.len() {
        let (trace, captures) = step(&mut scheduler, |_| true);
        for (_, phase, previous) in trace {
            let BatchPhase::Prefill { tokens } = phase else {
                panic!("a prefill chunk")
            };
            chunks.push((tokens, previous));
        }
        captured.extend(captures.iter().map(PrefixCaptureV1::tokens));
        capture_all(&mut scheduler, &captures);
    }
    // The cold prefill captures the prompt end's last multiple of the
    // granule, never the unaligned end itself.
    assert_eq!(chunks, [(24, 0), (1, 24)]);
    assert_eq!(captured, [24]);
    scheduler.audit().unwrap();
}

#[test]
fn prompt_ending_chunks_run_past_their_anchor_within_the_anchor_budget() {
    let mut scheduler = Scheduler::new(
        SchedulerConfig::new(4, 64, 64, 16, 1)
            .unwrap()
            .with_prefill_granule(8)
            .unwrap()
            .with_anchor_items(1)
            .with_prefix_cache(STATE_PAGES),
        KvGeometry::new(4, 64).unwrap(),
    )
    .unwrap();
    let first = prompt(45, 0);
    let second = prompt(45, 1000);
    submit(&mut scheduler, 1, &first, 1);
    submit(&mut scheduler, 2, &second, 1);
    let mut chunks = Vec::new();
    let mut captured = Vec::new();
    while scheduler.request_state(RequestId::new(1)).unwrap() != RequestState::Finished
        || scheduler.request_state(RequestId::new(2)).unwrap() != RequestState::Finished
    {
        let batch = scheduler.next_batch().unwrap().unwrap();
        let outcomes = batch
            .items()
            .iter()
            .zip(batch.kv_plan().appends())
            .map(|(item, append)| {
                chunks.push((
                    item.request_id().get(),
                    item.phase(),
                    append.previous_tokens(),
                    item.anchor_tokens(),
                ));
                BatchOutcome::success(
                    item.request_id(),
                    item.phase(),
                    usize::from(item.expects_output()),
                    item.expects_output(),
                )
            })
            .collect::<Vec<_>>();
        scheduler.commit_batch(batch.id(), &outcomes).unwrap();
        let captures = scheduler.take_prefix_captures();
        captured.extend(
            captures
                .iter()
                .map(|capture| (capture.request_id().get(), capture.tokens())),
        );
        capture_all(&mut scheduler, &captures);
        scheduler.audit().unwrap();
    }
    let prefill = |id: u64| {
        chunks
            .iter()
            .filter(|chunk| chunk.0 == id)
            .map(|&(_, phase, previous, anchor)| {
                let BatchPhase::Prefill { tokens } = phase else {
                    panic!("a prefill chunk")
                };
                (tokens, previous, anchor)
            })
            .collect::<Vec<_>>()
    };
    // The first request's last chunk runs past its anchor at 40 to the
    // prompt end and keeps its state there; with one anchor item a step,
    // the second, in the same steps, ends a chunk at 40 instead.
    assert_eq!(
        prefill(1),
        [(16, 0, None), (16, 16, None), (13, 32, Some(40))]
    );
    assert_eq!(
        prefill(2),
        [(16, 0, None), (16, 16, None), (8, 32, None), (5, 40, None)]
    );
    captured.sort_unstable();
    assert_eq!(captured, [(1, 40), (2, 40)]);

    // Either anchor serves a longer prompt, which resumes at 40.
    let mut third = first.clone();
    third.extend(prompt(10, 500));
    submit(&mut scheduler, 3, &third, 1);
    scheduler.admit().unwrap();
    assert_eq!(scheduler.cached_tokens(RequestId::new(3)).unwrap(), 40);
    scheduler.audit().unwrap();
}

/// An anchor item goes only to a chunk that reaches its prompt-end anchor
/// in this step: two prompts that stop first at the end of the blocks an
/// earlier prompt shared (32) keep none, and the third, which only has its
/// anchor (40), runs past it with one of the two. Their next chunks then run
/// past their own anchors.
#[test]
fn only_a_chunk_that_reaches_its_anchor_takes_an_anchor_item() {
    let mut scheduler = Scheduler::new(
        SchedulerConfig::new(8, 256, 256, 64, 1)
            .unwrap()
            .with_prefill_granule(8)
            .unwrap()
            .with_anchor_items(2)
            .with_prefix_cache(STATE_PAGES),
        KvGeometry::new(4, 256).unwrap(),
    )
    .unwrap();
    // Two earlier prompts whose first 32 tokens the next two share.
    for (id, salt) in [(1, 0), (2, 500)] {
        submit(&mut scheduler, id, &prompt(40, salt), 1);
        while scheduler.request_state(RequestId::new(id)).unwrap() != RequestState::Finished {
            let (_, captures) = step(&mut scheduler, |_| true);
            capture_all(&mut scheduler, &captures);
        }
    }
    let mut shared = |id: u64, salt: u32| {
        let mut tokens = prompt(32, salt);
        tokens.extend(prompt(13, 900 + salt));
        submit(&mut scheduler, id, &tokens, 1);
    };
    shared(3, 0);
    shared(4, 500);
    submit(&mut scheduler, 5, &prompt(45, 2000), 1);
    let mut chunks = Vec::new();
    for _ in 0..2 {
        let batch = scheduler.next_batch().unwrap().unwrap();
        let outcomes = batch
            .items()
            .iter()
            .map(|item| {
                chunks.push((item.request_id().get(), item.phase(), item.anchor_tokens()));
                BatchOutcome::success(
                    item.request_id(),
                    item.phase(),
                    usize::from(item.expects_output()),
                    item.expects_output(),
                )
            })
            .collect::<Vec<_>>();
        scheduler.commit_batch(batch.id(), &outcomes).unwrap();
        let captures = scheduler.take_prefix_captures();
        capture_all(&mut scheduler, &captures);
    }
    chunks.sort_by_key(|&(id, ..)| id);
    assert_eq!(
        chunks,
        [
            (3, BatchPhase::Prefill { tokens: 32 }, None),
            (3, BatchPhase::Prefill { tokens: 13 }, Some(40)),
            (4, BatchPhase::Prefill { tokens: 32 }, None),
            (4, BatchPhase::Prefill { tokens: 13 }, Some(40)),
            (5, BatchPhase::Prefill { tokens: 45 }, Some(40)),
        ]
    );
    scheduler.audit().unwrap();
}
