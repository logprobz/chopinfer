use std::cell::Cell;

use runtime_kv::KvGeometry;
use runtime_scheduler::{
    Batch, BatchOutcome, BatchPhase, DraftBudget, DraftCandidate, DraftLimits, RequestId,
    RequestSpec, RequestState, ScheduleError, Scheduler, SchedulerConfig,
};

fn scheduler(max_sequences: usize, blocks: usize, budget: usize, chunk: usize) -> Scheduler {
    Scheduler::new(
        SchedulerConfig::new(max_sequences, 256, budget, chunk, 1).unwrap(),
        KvGeometry::new(4, blocks).unwrap(),
    )
    .unwrap()
}

fn spec(id: u64, prompt: usize, output: usize) -> RequestSpec {
    RequestSpec::new(RequestId::new(id), prompt, output).unwrap()
}

fn commit(scheduler: &mut Scheduler, batch: &Batch, emitted: &[(u64, usize, bool)]) {
    let outcomes: Vec<_> = batch
        .items()
        .iter()
        .map(|item| {
            let (count, terminal) = emitted
                .iter()
                .find(|(id, _, _)| *id == item.request_id().get())
                .map_or(
                    (usize::from(item.expects_output()), false),
                    |(_, count, terminal)| (*count, *terminal),
                );
            BatchOutcome::success(item.request_id(), item.phase(), count, terminal)
        })
        .collect();
    scheduler.commit_batch(batch.id(), &outcomes).unwrap();
    scheduler.audit().unwrap();
}

/// Prefills every submitted prompt so each request is ready to decode.
fn prefill_all(scheduler: &mut Scheduler) {
    let batch = scheduler.next_batch().unwrap().unwrap();
    assert!(
        batch
            .items()
            .iter()
            .all(|item| matches!(item.phase(), BatchPhase::Prefill { .. }))
    );
    commit(scheduler, &batch, &[]);
}

fn next_with(scheduler: &mut Scheduler, limits: DraftLimits, drafts: &[usize]) -> Batch {
    scheduler
        .next_batch_with_drafts(limits, |_, _| drafts.to_vec())
        .unwrap()
        .unwrap()
}

#[test]
fn the_plan_sees_decode_items_with_their_clamps() {
    let mut scheduler = scheduler(2, 16, 16, 8);
    scheduler.submit(spec(1, 3, 10)).unwrap();
    scheduler.submit(spec(2, 3, 3)).unwrap();
    prefill_all(&mut scheduler);

    let seen = Cell::new(None);
    let batch = scheduler
        .next_batch_with_drafts(DraftLimits::new(4, 32), |candidates, budget| {
            seen.set(Some((candidates.to_vec(), budget)));
            vec![4, 1]
        })
        .unwrap()
        .unwrap();
    let (candidates, budget): (Vec<DraftCandidate>, DraftBudget) = seen.take().unwrap();
    assert_eq!(
        candidates
            .iter()
            .map(|candidate| (
                candidate.request_id().get(),
                candidate.maximum_draft_tokens()
            ))
            .collect::<Vec<_>>(),
        // Request 2 has emitted one of three tokens, so one draft plus the
        // bonus token is all its quota admits.
        [(1, 4), (2, 1)]
    );
    assert_eq!(
        candidates
            .iter()
            .map(|candidate| candidate.remaining_tokens())
            .collect::<Vec<_>>(),
        [9, 2]
    );
    assert_eq!(budget.spare_draft_tokens(), 14);
    assert_eq!(budget.spare_rows(), 30);
    assert!(!budget.prefill());
    let phases: Vec<_> = batch.items().iter().map(|item| item.phase()).collect();
    assert_eq!(
        phases,
        [
            BatchPhase::Verify { draft_tokens: 4 },
            BatchPhase::Verify { draft_tokens: 1 }
        ]
    );
    assert_eq!(batch.token_cost(), 7);
    assert_eq!(
        batch
            .kv_plan()
            .appends()
            .iter()
            .map(|append| append.additional_tokens())
            .collect::<Vec<_>>(),
        [5, 2]
    );
    scheduler.audit().unwrap();
}

#[test]
fn the_plan_knows_when_its_batch_also_prefills() {
    let mut scheduler = scheduler(2, 16, 16, 8);
    scheduler.submit(spec(1, 3, 10)).unwrap();
    prefill_all(&mut scheduler);
    scheduler.submit(spec(2, 3, 10)).unwrap();
    let seen = Cell::new(None);
    let batch = scheduler
        .next_batch_with_drafts(DraftLimits::new(4, 32), |candidates, budget| {
            seen.set(Some((candidates.len(), budget.prefill())));
            vec![0; candidates.len()]
        })
        .unwrap()
        .unwrap();
    assert!(
        batch
            .items()
            .iter()
            .any(|item| matches!(item.phase(), BatchPhase::Prefill { .. }))
    );
    assert_eq!(seen.take(), Some((1, true)));
}

#[test]
fn a_zero_draft_count_leaves_a_plain_decode() {
    let mut scheduler = scheduler(1, 8, 8, 8);
    scheduler.submit(spec(1, 3, 10)).unwrap();
    prefill_all(&mut scheduler);
    let batch = next_with(&mut scheduler, DraftLimits::new(4, 32), &[0]);
    assert_eq!(batch.items()[0].phase(), BatchPhase::Decode);
    assert_eq!(batch.token_cost(), 1);
}

#[test]
fn the_plan_is_not_called_without_limits_or_decode_items() {
    let mut scheduler = scheduler(1, 8, 8, 8);
    scheduler.submit(spec(1, 3, 10)).unwrap();
    let called = Cell::new(false);
    let batch = scheduler
        .next_batch_with_drafts(DraftLimits::new(4, 32), |_, _| {
            called.set(true);
            Vec::new()
        })
        .unwrap()
        .unwrap();
    assert!(!called.get());
    commit(&mut scheduler, &batch, &[]);
    let batch = scheduler
        .next_batch_with_drafts(DraftLimits::NONE, |_, _| {
            called.set(true);
            Vec::new()
        })
        .unwrap()
        .unwrap();
    assert!(!called.get());
    assert_eq!(batch.items()[0].phase(), BatchPhase::Decode);
}

#[test]
fn partial_acceptance_keeps_kv_at_prefilled_plus_emitted_minus_one() {
    let mut scheduler = scheduler(1, 8, 16, 8);
    let id = RequestId::new(1);
    scheduler.submit(spec(1, 3, 20)).unwrap();
    prefill_all(&mut scheduler);
    assert_eq!(scheduler.kv_tokens(id).unwrap(), 3);

    let batch = next_with(&mut scheduler, DraftLimits::new(6, 32), &[6]);
    commit(&mut scheduler, &batch, &[(1, 2, false)]);
    assert_eq!(scheduler.emitted_tokens(id).unwrap(), 3);
    assert_eq!(scheduler.kv_tokens(id).unwrap(), 5);
    assert_eq!(scheduler.kv_blocks(id).unwrap().len(), 2);

    let batch = next_with(&mut scheduler, DraftLimits::new(6, 32), &[6]);
    commit(&mut scheduler, &batch, &[(1, 7, false)]);
    assert_eq!(scheduler.emitted_tokens(id).unwrap(), 10);
    assert_eq!(scheduler.kv_tokens(id).unwrap(), 12);
    assert_eq!(scheduler.request_state(id).unwrap(), RequestState::Decoding);
}

#[test]
fn the_final_verify_step_fits_the_quota_exactly() {
    let mut scheduler = scheduler(1, 8, 16, 8);
    let id = RequestId::new(1);
    scheduler.submit(spec(1, 3, 5)).unwrap();
    prefill_all(&mut scheduler);
    let seen = Cell::new(0);
    let batch = scheduler
        .next_batch_with_drafts(DraftLimits::new(8, 32), |candidates, _| {
            seen.set(candidates[0].maximum_draft_tokens());
            vec![candidates[0].maximum_draft_tokens()]
        })
        .unwrap()
        .unwrap();
    assert_eq!(seen.get(), 3);
    commit(&mut scheduler, &batch, &[(1, 4, false)]);
    assert_eq!(scheduler.request_state(id).unwrap(), RequestState::Finished);
    assert_eq!(scheduler.kv_stats().free_blocks(), 8);
}

#[test]
fn a_stop_inside_a_verify_run_finishes_and_releases() {
    let mut scheduler = scheduler(1, 8, 16, 8);
    let id = RequestId::new(1);
    scheduler.submit(spec(1, 3, 20)).unwrap();
    prefill_all(&mut scheduler);
    let batch = next_with(&mut scheduler, DraftLimits::new(5, 32), &[5]);
    commit(&mut scheduler, &batch, &[(1, 2, true)]);
    assert_eq!(scheduler.request_state(id).unwrap(), RequestState::Finished);
    assert_eq!(scheduler.emitted_tokens(id).unwrap(), 3);
    assert_eq!(scheduler.kv_stats().free_blocks(), 8);
}

#[test]
fn verify_outcomes_outside_one_to_k_plus_one_are_refused() {
    let mut scheduler = scheduler(1, 8, 16, 8);
    scheduler.submit(spec(1, 3, 20)).unwrap();
    prefill_all(&mut scheduler);
    let batch = next_with(&mut scheduler, DraftLimits::new(3, 32), &[3]);
    let item = batch.items()[0];
    for emitted in [0, 5] {
        let outcome = BatchOutcome::success(item.request_id(), item.phase(), emitted, false);
        assert_eq!(
            scheduler
                .preflight_commit_batch(batch.id(), &[outcome])
                .err(),
            Some(ScheduleError::InvalidOutcome(RequestId::new(1)))
        );
    }
    let decode = BatchOutcome::success(item.request_id(), BatchPhase::Decode, 1, false);
    assert_eq!(
        scheduler
            .preflight_commit_batch(batch.id(), &[decode])
            .err(),
        Some(ScheduleError::OutcomeMembershipMismatch)
    );
    scheduler.rollback_batch(batch.id()).unwrap();
    scheduler.audit().unwrap();
    assert_eq!(scheduler.kv_tokens(RequestId::new(1)).unwrap(), 3);
}

#[test]
fn plans_that_do_not_fit_are_refused_before_any_kv_is_prepared() {
    let mut scheduler = scheduler(2, 16, 8, 4);
    scheduler.submit(spec(1, 3, 20)).unwrap();
    scheduler.submit(spec(2, 3, 20)).unwrap();
    prefill_all(&mut scheduler);
    let before = scheduler.kv_stats();
    // Two decode items leave six spare budget tokens; a row limit of four
    // leaves two spare rows.
    for (limits, plan) in [
        (DraftLimits::new(4, 4), vec![1]),
        (DraftLimits::new(4, 4), vec![1, 1, 1]),
        (DraftLimits::new(4, 4), vec![3, 0]),
        (DraftLimits::new(4, 4), vec![2, 1]),
        (DraftLimits::new(4, 32), vec![4, 4]),
        (DraftLimits::new(2, 32), vec![3, 0]),
    ] {
        let result = scheduler.next_batch_with_drafts(limits, |_, _| plan);
        assert_eq!(result.err(), Some(ScheduleError::InvalidDraftPlan));
        assert_eq!(scheduler.kv_stats(), before);
        scheduler.audit().unwrap();
    }
    let batch = next_with(&mut scheduler, DraftLimits::new(4, 4), &[1, 1]);
    assert_eq!(batch.token_cost(), 4);
    commit(&mut scheduler, &batch, &[(1, 1, false), (2, 2, false)]);
}

#[test]
fn a_mixed_batch_charges_verify_rows_to_the_token_budget() {
    let mut scheduler = scheduler(2, 32, 8, 4);
    scheduler.submit(spec(1, 3, 20)).unwrap();
    prefill_all(&mut scheduler);
    scheduler.submit(spec(2, 8, 20)).unwrap();
    let seen = Cell::new(None);
    let batch = scheduler
        .next_batch_with_drafts(DraftLimits::new(4, 32), |candidates, budget| {
            seen.set(Some(budget));
            vec![
                candidates[0]
                    .maximum_draft_tokens()
                    .min(budget.spare_draft_tokens()),
            ]
        })
        .unwrap()
        .unwrap();
    // One decode and one four-token prefill chunk leave three spare tokens.
    assert_eq!(seen.get().unwrap().spare_draft_tokens(), 3);
    assert_eq!(batch.token_cost(), 8);
    assert!(
        batch
            .items()
            .iter()
            .any(|item| item.phase() == BatchPhase::Verify { draft_tokens: 3 })
    );
    commit(&mut scheduler, &batch, &[(1, 2, false)]);
}

#[test]
fn many_partial_steps_return_every_block() {
    let mut scheduler = scheduler(3, 64, 32, 8);
    for id in 1..=3 {
        scheduler.submit(spec(id, 5, 40)).unwrap();
    }
    prefill_all(&mut scheduler);
    let mut step = 0usize;
    while let Some(batch) = scheduler
        .next_batch_with_drafts(DraftLimits::new(5, 32), |candidates, _| {
            candidates
                .iter()
                .map(|candidate| candidate.maximum_draft_tokens().min(1 + step % 5))
                .collect()
        })
        .unwrap()
    {
        let emitted: Vec<_> = batch
            .items()
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let rows = item.phase().appended_tokens();
                (item.request_id().get(), 1 + (step + index) % rows, false)
            })
            .collect();
        commit(&mut scheduler, &batch, &emitted);
        step += 1;
    }
    for id in 1..=3 {
        assert_eq!(
            scheduler.request_state(RequestId::new(id)).unwrap(),
            RequestState::Finished
        );
        assert_eq!(scheduler.emitted_tokens(RequestId::new(id)).unwrap(), 40);
    }
    assert_eq!(scheduler.kv_stats().free_blocks(), 64);
}
