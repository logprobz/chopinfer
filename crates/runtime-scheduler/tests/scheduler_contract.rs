use runtime_kv::{KvCachePrecision, KvGeometry, KvMetrics};
use runtime_scheduler::{
    BatchOutcome, BatchPhase, ChunkMedia, ConfigError, InvalidRequest, Overload, RequestId,
    RequestSpec, RequestState, ScheduleError, Scheduler, SchedulerConfig, SubmitError,
};

fn scheduler(max_sequences: usize, blocks: usize, budget: usize, chunk: usize) -> Scheduler {
    Scheduler::new(
        SchedulerConfig::new(max_sequences, 256, budget, chunk, 1)
            .unwrap()
            .with_lone_prefill_expansion(true),
        KvGeometry::new(4, blocks).unwrap(),
    )
    .unwrap()
}

fn spec(id: u64, prompt: usize, output: usize) -> RequestSpec {
    RequestSpec::new(RequestId::new(id), prompt, output).unwrap()
}

fn commit_next(scheduler: &mut Scheduler) -> Vec<(u64, BatchPhase)> {
    let batch = scheduler.next_batch().unwrap().unwrap();
    let trace = batch
        .items()
        .iter()
        .map(|item| (item.request_id().get(), item.phase()))
        .collect();
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
    let prepared = scheduler
        .preflight_commit_batch(batch.id(), &outcomes)
        .unwrap();
    let _ = scheduler.apply_prepared_commit(prepared).unwrap();
    trace
}

fn outcomes_for(batch: &runtime_scheduler::Batch) -> Vec<BatchOutcome> {
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

#[test]
fn dropped_preflight_token_can_be_rolled_back_and_retried() {
    let mut scheduler = scheduler(1, 4, 4, 4);
    scheduler.submit(spec(1, 2, 1)).unwrap();
    let batch = scheduler.next_batch().unwrap().unwrap();
    let before = scheduler.kv_stats();
    let outcomes = outcomes_for(&batch);

    let prepared = scheduler
        .preflight_commit_batch(batch.id(), &outcomes)
        .unwrap();

    assert_eq!(scheduler.kv_stats(), before);
    assert_eq!(scheduler.kv_tokens(RequestId::new(1)).unwrap(), 0);
    assert_eq!(scheduler.emitted_tokens(RequestId::new(1)).unwrap(), 0);
    assert_eq!(
        scheduler.request_state(RequestId::new(1)).unwrap(),
        RequestState::Admitted
    );
    drop(prepared);
    scheduler.rollback_batch(batch.id()).unwrap();

    let retry = scheduler.next_batch().unwrap().unwrap();
    let retry_outcomes = outcomes_for(&retry);
    let prepared = scheduler
        .preflight_commit_batch(retry.id(), &retry_outcomes)
        .unwrap();
    scheduler.apply_prepared_commit(prepared).unwrap();
    assert_eq!(
        scheduler.request_state(RequestId::new(1)).unwrap(),
        RequestState::Finished
    );
    scheduler.audit().unwrap();
}

#[test]
fn foreign_scheduler_token_is_rejected_without_consuming_either_batch() {
    let mut first = scheduler(1, 4, 4, 4);
    let mut second = scheduler(1, 4, 4, 4);
    first.submit(spec(1, 2, 1)).unwrap();
    second.submit(spec(1, 2, 1)).unwrap();
    let first_batch = first.next_batch().unwrap().unwrap();
    let second_batch = second.next_batch().unwrap().unwrap();
    let prepared = first
        .preflight_commit_batch(first_batch.id(), &outcomes_for(&first_batch))
        .unwrap();

    assert_eq!(
        second.apply_prepared_commit(prepared),
        Err(ScheduleError::PreparedCommitMismatch)
    );
    second.rollback_batch(second_batch.id()).unwrap();
    first.rollback_batch(first_batch.id()).unwrap();
    first.audit().unwrap();
    second.audit().unwrap();
}

#[test]
fn stale_and_exact_state_mismatched_tokens_do_not_consume_pending_state() {
    let mut scheduler = scheduler(1, 8, 4, 4);
    scheduler.submit(spec(1, 2, 2)).unwrap();
    let first = scheduler.next_batch().unwrap().unwrap();
    let stale = scheduler
        .preflight_commit_batch(first.id(), &outcomes_for(&first))
        .unwrap();
    scheduler.rollback_batch(first.id()).unwrap();

    let second = scheduler.next_batch().unwrap().unwrap();
    assert_eq!(
        scheduler.apply_prepared_commit(stale),
        Err(ScheduleError::BatchIdMismatch)
    );
    let exact = scheduler
        .preflight_commit_batch(second.id(), &outcomes_for(&second))
        .unwrap();
    scheduler.submit(spec(2, 1, 1)).unwrap();
    assert_eq!(
        scheduler.apply_prepared_commit(exact),
        Err(ScheduleError::PreparedCommitMismatch)
    );
    scheduler.rollback_batch(second.id()).unwrap();
    scheduler.audit().unwrap();
}

#[test]
fn rejects_invalid_lengths_duplicate_ids_and_tracking_overload() {
    assert_eq!(
        RequestSpec::new(RequestId::new(1), 0, 1),
        Err(InvalidRequest::ZeroPrompt)
    );
    assert_eq!(
        RequestSpec::new(RequestId::new(1), 1, 0),
        Err(InvalidRequest::ZeroMaximumNewTokens)
    );
    assert_eq!(
        RequestSpec::new(RequestId::new(1), usize::MAX, 2),
        Err(InvalidRequest::LengthOverflow)
    );

    let mut scheduler = Scheduler::new(
        SchedulerConfig::new(1, 1, 4, 4, 1).unwrap(),
        KvGeometry::new(4, 4).unwrap(),
    )
    .unwrap();
    scheduler.submit(spec(7, 2, 1)).unwrap();
    assert_eq!(
        scheduler.submit(spec(7, 2, 1)),
        Err(SubmitError::DuplicateId(RequestId::new(7)))
    );
    assert_eq!(
        scheduler.submit(spec(8, 2, 1)),
        Err(SubmitError::Overloaded(Overload::TrackingCapacity))
    );
}

#[test]
fn prefill_chunk_limit_is_bounded_by_the_token_budget() {
    // A chunk limit equal to the budget keeps every prefill item executable as
    // a full chunk or an authenticated tail. Above the budget, an interior
    // chunk could never fit and the request would stall while staying active.
    let equal = SchedulerConfig::new(1, 1, 4, 4, 1);
    assert!(
        equal.is_ok(),
        "a chunk limit equal to the token budget must be accepted"
    );
    assert_eq!(
        SchedulerConfig::new(1, 1, 4, 5, 1),
        Err(ConfigError::PrefillChunkLimitAboveTokenBudget)
    );
    assert_eq!(
        SchedulerConfig::new(1, 1, 4, usize::MAX, 1)
            .unwrap_err()
            .to_string(),
        "scheduler prefill chunk limit exceeds the token budget"
    );
}

#[test]
fn admission_reports_sequence_and_kv_overload_without_mutation() {
    let mut limited = scheduler(1, 8, 4, 4);
    limited.submit(spec(1, 4, 1)).unwrap();
    limited.submit(spec(2, 4, 1)).unwrap();
    let report = limited.admit().unwrap();
    assert_eq!(report.admitted(), &[RequestId::new(1)]);
    assert_eq!(report.blocked(), Some(Overload::MaximumSequences));
    assert_eq!(
        limited.request_state(RequestId::new(2)).unwrap(),
        RequestState::Queued
    );

    let mut kv_limited = scheduler(2, 2, 4, 4);
    kv_limited.submit(spec(1, 4, 4)).unwrap();
    kv_limited.submit(spec(2, 1, 1)).unwrap();
    let report = kv_limited.admit().unwrap();
    assert_eq!(report.admitted(), &[RequestId::new(1)]);
    assert_eq!(report.blocked(), Some(Overload::KvCapacity));
    assert_eq!(kv_limited.kv_stats().used_blocks(), 0);
}

#[test]
fn chunked_prefill_respects_budget_boundaries() {
    let mut scheduler = scheduler(2, 16, 3, 2);
    scheduler.submit(spec(1, 5, 2)).unwrap();
    for expected in [3, 2] {
        let trace = commit_next(&mut scheduler);
        assert_eq!(trace, vec![(1, BatchPhase::Prefill { tokens: expected })]);
    }
    assert_eq!(
        scheduler.request_state(RequestId::new(1)).unwrap(),
        RequestState::Decoding
    );
    assert_eq!(scheduler.kv_tokens(RequestId::new(1)).unwrap(), 5);
    assert_eq!(scheduler.emitted_tokens(RequestId::new(1)).unwrap(), 1);
}

#[test]
fn decode_and_mixed_batches_obey_cost_budget() {
    // A mixed batch must carry a full chunk or an authenticated tail plus the
    // decode item, so the budget has to hold both: 3 + 2.
    let mut scheduler = Scheduler::new(
        SchedulerConfig::new(4, 32, 5, 3, 2).unwrap(),
        KvGeometry::new(4, 32).unwrap(),
    )
    .unwrap();
    scheduler.submit(spec(1, 1, 3)).unwrap();
    commit_next(&mut scheduler);
    scheduler.submit(spec(2, 5, 1)).unwrap();
    let batch = scheduler.next_batch().unwrap().unwrap();
    assert_eq!(batch.token_cost(), 5);
    assert!(
        batch
            .items()
            .iter()
            .any(|item| item.phase() == BatchPhase::Decode)
    );
    assert!(
        batch
            .items()
            .iter()
            .any(|item| item.phase() == BatchPhase::Prefill { tokens: 3 })
    );
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

#[test]
fn prefill_and_decode_each_make_progress_under_tight_budget() {
    let mut scheduler = Scheduler::new(
        SchedulerConfig::new(4, 32, 1, 1, 1).unwrap(),
        KvGeometry::new(4, 32).unwrap(),
    )
    .unwrap();
    scheduler.submit(spec(1, 1, 8)).unwrap();
    commit_next(&mut scheduler);
    scheduler.submit(spec(2, 8, 1)).unwrap();
    let mut phases = Vec::new();
    for _ in 0..8 {
        phases.push(commit_next(&mut scheduler)[0].1);
    }
    assert!(phases.windows(2).all(|window| window[0] != window[1]));
    assert!(phases.contains(&BatchPhase::Decode));
    assert!(phases.contains(&BatchPhase::Prefill { tokens: 1 }));
}

#[test]
fn rollback_and_wrong_commit_leave_batch_atomic() {
    let mut scheduler = scheduler(2, 8, 4, 4);
    scheduler.submit(spec(1, 4, 2)).unwrap();
    let batch = scheduler.next_batch().unwrap().unwrap();
    assert_eq!(batch.kv_plan().appends().len(), 1);
    assert_eq!(batch.kv_plan().appends()[0].new_tokens(), 4);
    assert_eq!(
        batch.kv_plan().appends()[0]
            .allocated()
            .iter()
            .map(|block| block.index())
            .collect::<Vec<_>>(),
        [0]
    );
    assert_eq!(scheduler.kv_tokens(RequestId::new(1)).unwrap(), 0);
    scheduler.audit().unwrap();
    assert_eq!(scheduler.next_batch(), Err(ScheduleError::PendingBatch));
    assert_eq!(
        scheduler.commit_batch(batch.id().next(), &[]),
        Err(ScheduleError::BatchIdMismatch)
    );
    assert_eq!(scheduler.kv_tokens(RequestId::new(1)).unwrap(), 0);
    scheduler.rollback_batch(batch.id()).unwrap();
    assert_eq!(
        scheduler.request_state(RequestId::new(1)).unwrap(),
        RequestState::Admitted
    );
    let retry = scheduler.next_batch().unwrap().unwrap();
    assert_eq!(retry.items(), batch.items());
    assert_eq!(
        retry.kv_plan().appends()[0].allocated(),
        batch.kv_plan().appends()[0].allocated()
    );
    let outcomes = retry
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
    scheduler.commit_batch(retry.id(), &outcomes).unwrap();
    assert_eq!(
        scheduler.kv_blocks(RequestId::new(1)).unwrap(),
        retry.kv_plan().appends()[0].allocated()
    );
}

#[test]
fn pending_cancellation_discards_prepared_kv_for_the_whole_batch() {
    let mut scheduler = scheduler(2, 8, 4, 4);
    scheduler.submit(spec(1, 4, 1)).unwrap();
    scheduler.submit(spec(2, 4, 1)).unwrap();
    let batch = scheduler.next_batch().unwrap().unwrap();
    assert_eq!(batch.kv_plan().appends().len(), 1);
    scheduler.cancel(RequestId::new(1)).unwrap();
    assert_eq!(scheduler.kv_stats().free_blocks(), 8);
    assert_eq!(
        scheduler.commit_batch(batch.id(), &[]),
        Err(ScheduleError::NoPendingBatch)
    );
    assert_eq!(
        scheduler.request_state(RequestId::new(2)).unwrap(),
        RequestState::Admitted
    );
    scheduler.audit().unwrap();
}

#[test]
fn terminal_retirement_is_stable_and_reuses_tracking_capacity() {
    let mut scheduler = Scheduler::new(
        SchedulerConfig::new(1, 2, 4, 4, 1).unwrap(),
        KvGeometry::new(4, 4).unwrap(),
    )
    .unwrap();
    scheduler.submit(spec(10, 1, 1)).unwrap();
    assert_eq!(
        scheduler.retire_terminal(RequestId::new(10)),
        Err(ScheduleError::RequestNotTerminal(RequestId::new(10)))
    );
    let pending = scheduler.next_batch().unwrap().unwrap();
    assert_eq!(
        scheduler.retire_terminal(RequestId::new(10)),
        Err(ScheduleError::RequestPendingBatch(RequestId::new(10)))
    );
    scheduler.rollback_batch(pending.id()).unwrap();
    scheduler.finish(RequestId::new(10)).unwrap();
    scheduler.submit(spec(20, 1, 1)).unwrap();
    scheduler.finish(RequestId::new(20)).unwrap();
    assert_eq!(scheduler.reap_terminal(1).unwrap(), [RequestId::new(10)]);
    scheduler.submit(spec(30, 1, 1)).unwrap();
    assert_eq!(scheduler.reap_terminal(1).unwrap(), [RequestId::new(20)]);

    for id in 31..41 {
        scheduler.finish(RequestId::new(30)).unwrap();
        scheduler.retire_terminal(RequestId::new(30)).unwrap();
        scheduler.submit(spec(id, 1, 1)).unwrap();
        scheduler.finish(RequestId::new(id)).unwrap();
        scheduler.retire_terminal(RequestId::new(id)).unwrap();
        scheduler.submit(spec(30, 1, 1)).unwrap();
    }
    scheduler.finish(RequestId::new(30)).unwrap();
    scheduler.retire_terminal(RequestId::new(30)).unwrap();
    assert_eq!(scheduler.terminal_count(), 0);
    scheduler.audit().unwrap();
}

#[test]
fn cancellation_at_each_live_state_and_failure_release_kv() {
    let mut scheduler = scheduler(4, 16, 2, 2);
    scheduler.submit(spec(1, 2, 2)).unwrap();
    scheduler.cancel(RequestId::new(1)).unwrap();
    assert_eq!(
        scheduler.request_state(RequestId::new(1)).unwrap(),
        RequestState::Cancelled
    );

    scheduler.submit(spec(2, 2, 2)).unwrap();
    scheduler.admit().unwrap();
    scheduler.cancel(RequestId::new(2)).unwrap();

    scheduler.submit(spec(3, 4, 2)).unwrap();
    commit_next(&mut scheduler);
    assert_eq!(
        scheduler.request_state(RequestId::new(3)).unwrap(),
        RequestState::Prefilling
    );
    scheduler.cancel(RequestId::new(3)).unwrap();

    scheduler.submit(spec(4, 1, 2)).unwrap();
    commit_next(&mut scheduler);
    assert_eq!(
        scheduler.request_state(RequestId::new(4)).unwrap(),
        RequestState::Decoding
    );
    scheduler.cancel(RequestId::new(4)).unwrap();
    assert_eq!(
        scheduler.request_state(RequestId::new(4)).unwrap(),
        RequestState::Cancelled
    );

    scheduler.submit(spec(5, 1, 2)).unwrap();
    commit_next(&mut scheduler);
    scheduler.fail(RequestId::new(5)).unwrap();
    assert_eq!(
        scheduler.request_state(RequestId::new(5)).unwrap(),
        RequestState::Failed
    );
    assert_eq!(scheduler.kv_stats().used_blocks(), 0);
    scheduler.audit().unwrap();
}

#[test]
fn normal_completion_releases_kv() {
    let mut scheduler = scheduler(1, 4, 4, 4);
    scheduler.submit(spec(1, 2, 1)).unwrap();
    commit_next(&mut scheduler);
    assert_eq!(
        scheduler.request_state(RequestId::new(1)).unwrap(),
        RequestState::Finished
    );
    assert_eq!(scheduler.kv_stats().used_blocks(), 0);
    assert!(scheduler.next_batch().unwrap().is_none());
}

#[test]
fn final_prefill_emits_without_appending_and_decode_appends_pending_tokens() {
    let mut scheduler = scheduler(1, 8, 4, 4);
    scheduler.submit(spec(1, 2, 3)).unwrap();

    let prefill = scheduler.next_batch().unwrap().unwrap();
    assert!(prefill.items()[0].expects_output());
    scheduler
        .commit_batch(
            prefill.id(),
            &[BatchOutcome::success(
                RequestId::new(1),
                BatchPhase::Prefill { tokens: 2 },
                1,
                false,
            )],
        )
        .unwrap();
    assert_eq!(scheduler.kv_tokens(RequestId::new(1)).unwrap(), 2);
    assert_eq!(scheduler.emitted_tokens(RequestId::new(1)).unwrap(), 1);

    let decode = scheduler.next_batch().unwrap().unwrap();
    scheduler
        .commit_batch(
            decode.id(),
            &[BatchOutcome::success(
                RequestId::new(1),
                BatchPhase::Decode,
                1,
                false,
            )],
        )
        .unwrap();
    assert_eq!(scheduler.kv_tokens(RequestId::new(1)).unwrap(), 3);
    assert_eq!(scheduler.emitted_tokens(RequestId::new(1)).unwrap(), 2);
}

#[test]
fn explicit_terminal_outcome_finishes_without_an_extra_decode() {
    let mut scheduler = scheduler(1, 8, 4, 4);
    scheduler.submit(spec(1, 1, 8)).unwrap();
    let batch = scheduler.next_batch().unwrap().unwrap();
    scheduler
        .commit_batch(
            batch.id(),
            &[BatchOutcome::success(
                RequestId::new(1),
                BatchPhase::Prefill { tokens: 1 },
                1,
                true,
            )],
        )
        .unwrap();
    assert_eq!(
        scheduler.request_state(RequestId::new(1)).unwrap(),
        RequestState::Finished
    );
    assert_eq!(scheduler.kv_stats().used_blocks(), 0);
    assert!(scheduler.next_batch().unwrap().is_none());
}

#[test]
fn outcome_mismatch_preserves_prepared_ownership_for_rollback() {
    let mut scheduler = scheduler(2, 8, 4, 4);
    scheduler.submit(spec(1, 2, 2)).unwrap();
    scheduler.submit(spec(2, 2, 2)).unwrap();
    let batch = scheduler.next_batch().unwrap().unwrap();
    let free_while_prepared = scheduler.kv_stats().free_blocks();
    assert_eq!(
        scheduler.commit_batch(batch.id(), &[]),
        Err(ScheduleError::OutcomeCountMismatch)
    );
    let mismatched = batch
        .items()
        .iter()
        .enumerate()
        .map(|(index, item)| {
            BatchOutcome::success(
                if index == 0 {
                    RequestId::new(99)
                } else {
                    item.request_id()
                },
                item.phase(),
                usize::from(item.expects_output()),
                false,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        scheduler.commit_batch(batch.id(), &mismatched),
        Err(ScheduleError::OutcomeMembershipMismatch)
    );
    let invalid = batch
        .items()
        .iter()
        .map(|item| BatchOutcome::success(item.request_id(), item.phase(), 0, false))
        .collect::<Vec<_>>();
    assert_eq!(
        scheduler.commit_batch(batch.id(), &invalid),
        Err(ScheduleError::InvalidOutcome(
            batch
                .items()
                .iter()
                .find(|item| item.expects_output())
                .map(|item| item.request_id())
                .unwrap_or_else(|| {
                    // Both final prefill items fit this batch under the test budget.
                    RequestId::new(1)
                })
        ))
    );
    assert_eq!(scheduler.kv_stats().free_blocks(), free_while_prepared);
    scheduler.rollback_batch(batch.id()).unwrap();
    assert_eq!(scheduler.kv_stats().free_blocks(), 8);
    scheduler.audit().unwrap();
}

fn deterministic_trace(concurrency: usize) -> Vec<Vec<(u64, BatchPhase)>> {
    let mut scheduler = Scheduler::new(
        SchedulerConfig::new(concurrency, concurrency * 2, 32, 8, 1).unwrap(),
        KvGeometry::new(4, concurrency * 8).unwrap(),
    )
    .unwrap();
    for id in 0..concurrency as u64 {
        scheduler
            .submit(spec(id, (id as usize % 11) + 1, 3))
            .unwrap();
    }
    let mut trace = Vec::new();
    while scheduler.terminal_count() < concurrency {
        trace.push(commit_next(&mut scheduler));
        scheduler.audit().unwrap();
    }
    trace
}

#[test]
fn synthetic_c1_c4_c16_c32_traces_are_deterministic() {
    for concurrency in [1, 4, 16, 32] {
        let first = deterministic_trace(concurrency);
        let second = deterministic_trace(concurrency);
        assert_eq!(first, second);
        assert!(!first.is_empty());
    }
}

#[test]
fn scheduler_uses_the_injected_kv_metrics_handle() {
    let metrics = KvMetrics::new(KvCachePrecision::Fp8);
    let mut scheduler = Scheduler::new_with_kv_metrics(
        SchedulerConfig::new(1, 2, 4, 4, 1).unwrap(),
        KvGeometry::new(4, 4).unwrap(),
        metrics.clone(),
    )
    .unwrap();
    assert_eq!(metrics.snapshot().fp8_capacity_tokens, 16);
    assert_eq!(scheduler.kv_metrics().snapshot(), metrics.snapshot());

    scheduler.submit(spec(99, 1, 2)).unwrap();
    commit_next(&mut scheduler);
    assert_eq!(metrics.snapshot().fp8_tokens, 1);
    assert_eq!(scheduler.kv_metrics().snapshot(), metrics.snapshot());
    scheduler.cancel(RequestId::new(99)).unwrap();
    assert_eq!(metrics.snapshot().fp8_tokens, 0);
}

#[test]
fn scored_prefill_caps_the_whole_batch_across_chunk_sizes() {
    for chunk in [1, 7, 512, 2048, 8192] {
        for cap in [1, 3, 512] {
            let mut scheduler = Scheduler::new(
                SchedulerConfig::new(3, 16, 16384, chunk, 1)
                    .unwrap()
                    .with_maximum_scored_rows(cap)
                    .unwrap(),
                KvGeometry::new(4, 8192).unwrap(),
            )
            .unwrap();
            for id in [1, 2] {
                scheduler
                    .submit(spec(id, 2051, 1).with_prompt_scoring())
                    .unwrap();
            }
            scheduler.submit(spec(3, 17, 1)).unwrap();
            let mut totals = [0; 3];
            while let Some(batch) = scheduler.next_batch().unwrap() {
                let mut rows = 0;
                for item in batch.items() {
                    let index = item.request_id().get() as usize - 1;
                    let BatchPhase::Prefill { tokens } = item.phase() else {
                        panic!("prefill only")
                    };
                    totals[index] += tokens;
                    if index < 2 {
                        assert!(tokens <= cap.min(chunk));
                        rows += tokens;
                    } else {
                        assert_eq!(tokens, chunk.min(17 - (totals[index] - tokens)));
                        rows += usize::from(item.expects_output());
                    }
                }
                assert!(rows <= cap);
                let prepared = scheduler
                    .preflight_commit_batch(batch.id(), &outcomes_for(&batch))
                    .unwrap();
                scheduler.apply_prepared_commit(prepared).unwrap();
                scheduler.audit().unwrap();
            }
            assert_eq!(totals, [2051, 2051, 17]);
        }
    }
}

#[test]
fn scored_row_capacity_must_be_positive() {
    assert_eq!(
        SchedulerConfig::new(1, 1, 8, 8, 1)
            .unwrap()
            .with_maximum_scored_rows(0),
        Err(ConfigError::ZeroMaximumScoredRows)
    );
}

#[test]
fn scored_row_budget_is_restored_after_rollback_and_charged_for_media_overrun() {
    let mut scheduler = Scheduler::new(
        SchedulerConfig::new(2, 4, 8, 8, 1)
            .unwrap()
            .with_maximum_scored_rows(3)
            .unwrap(),
        KvGeometry::new(4, 32).unwrap(),
    )
    .unwrap();
    scheduler
        .submit(spec(1, 17, 1).with_prompt_scoring())
        .unwrap();
    scheduler
        .submit(spec(2, 17, 1).with_prompt_scoring())
        .unwrap();
    let batch = scheduler.next_batch().unwrap().unwrap();
    let items = batch.items().to_vec();
    scheduler.rollback_batch(batch.id()).unwrap();
    let retry = scheduler.next_batch().unwrap().unwrap();
    assert_eq!(retry.items(), items);
    scheduler.rollback_batch(retry.id()).unwrap();
    let overrun = scheduler
        .next_batch_planned(
            runtime_scheduler::DraftLimits::NONE,
            |_, _, _| ChunkMedia {
                encode_cost: 100,
                pages: 0,
            },
            |_, _| Vec::new(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(overrun.items().len(), 1);
    assert_eq!(
        overrun.items()[0].phase(),
        BatchPhase::Prefill { tokens: 3 }
    );
    assert_eq!(overrun.token_cost(), 103);
}

#[test]
fn scored_rows_respect_the_decode_aware_prefill_budget() {
    let mut scheduler = Scheduler::new(
        SchedulerConfig::new(3, 8, 8192, 2048, 1)
            .unwrap()
            .with_maximum_scored_rows(17)
            .unwrap()
            .with_prefill_budget(runtime_scheduler::PrefillBudget::DecodeAware(
                runtime_scheduler::PrefillTarget::DecodeSteps(1),
            )),
        KvGeometry::new(4, 4096).unwrap(),
    )
    .unwrap();
    scheduler.submit(spec(10, 1, 4)).unwrap();
    commit_next(&mut scheduler);
    scheduler
        .submit(spec(1, 2051, 1).with_prompt_scoring())
        .unwrap();
    scheduler
        .submit(spec(2, 2051, 1).with_prompt_scoring())
        .unwrap();
    let batch = scheduler.next_batch().unwrap().unwrap();
    assert!(
        batch
            .items()
            .iter()
            .any(|item| item.phase() == BatchPhase::Decode)
    );
    let rows: usize = batch
        .items()
        .iter()
        .map(|item| match item.phase() {
            BatchPhase::Prefill { tokens } => tokens,
            _ => 0,
        })
        .sum();
    assert!(rows > 0 && rows <= 17);
    scheduler.audit().unwrap();
}

#[test]
fn deferred_scored_media_chunk_is_not_retried_with_a_smaller_row_budget() {
    let mut scheduler = Scheduler::new(
        SchedulerConfig::new(2, 4, 1024, 400, 1)
            .unwrap()
            .with_maximum_scored_rows(512)
            .unwrap(),
        KvGeometry::new(4, 256).unwrap(),
    )
    .unwrap();
    scheduler.submit(spec(10, 1, 3)).unwrap();
    commit_next(&mut scheduler);
    scheduler
        .submit(spec(1, 400, 1).with_prompt_scoring())
        .unwrap();
    let batch = scheduler
        .next_batch_planned(
            runtime_scheduler::DraftLimits::NONE,
            |_, _, tokens| ChunkMedia {
                encode_cost: if tokens > 112 { 2000 } else { 0 },
                pages: 0,
            },
            |_, _| Vec::new(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(batch.items().len(), 2);
    assert_eq!(batch.items()[0].phase(), BatchPhase::Decode);
    assert_eq!(batch.items()[1].request_id(), RequestId::new(1));
    assert_eq!(
        batch.items()[1].phase(),
        BatchPhase::Prefill { tokens: 400 }
    );
    let prepared = scheduler
        .preflight_commit_batch(batch.id(), &outcomes_for(&batch))
        .unwrap();
    scheduler.apply_prepared_commit(prepared).unwrap();
    scheduler.audit().unwrap();
}

/// Prefill chunk sizes of request 1, a 1500-token scored prompt, scheduled
/// alone or next to other traffic that finishes after four tokens.
fn scored_chunks_next_to(others: &[RequestSpec]) -> Vec<usize> {
    let mut scheduler = Scheduler::new(
        SchedulerConfig::new(4, 16, 8192, 2048, 1)
            .unwrap()
            .with_maximum_scored_rows(512)
            .unwrap(),
        KvGeometry::new(4, 8192).unwrap(),
    )
    .unwrap();
    for other in others {
        scheduler.submit(*other).unwrap();
    }
    scheduler
        .submit(spec(1, 1500, 1).with_prompt_scoring())
        .unwrap();
    let mut chunks = Vec::new();
    while let Some(batch) = scheduler.next_batch().unwrap() {
        for item in batch.items() {
            if let (1, BatchPhase::Prefill { tokens }) = (item.request_id().get(), item.phase()) {
                chunks.push(tokens);
            }
        }
        let prepared = scheduler
            .preflight_commit_batch(batch.id(), &outcomes_for(&batch))
            .unwrap();
        scheduler.apply_prepared_commit(prepared).unwrap();
        scheduler.audit().unwrap();
    }
    chunks
}

#[test]
fn scored_chunk_boundaries_do_not_depend_on_co_batched_requests() {
    let alone = scored_chunks_next_to(&[]);
    assert_eq!(alone, [512, 512, 476]);
    assert_eq!(scored_chunks_next_to(&[spec(9, 10, 4)]), alone);
    assert_eq!(
        scored_chunks_next_to(&[spec(8, 300, 1).with_prompt_scoring()]),
        alone
    );
    assert_eq!(
        scored_chunks_next_to(&[spec(9, 10, 4), spec(8, 300, 1).with_prompt_scoring()]),
        alone
    );
}

#[test]
fn scored_request_is_not_starved_by_arriving_final_rows() {
    let mut scheduler = Scheduler::new(
        SchedulerConfig::new(8, 256, 64, 8, 1)
            .unwrap()
            .with_maximum_scored_rows(2)
            .unwrap(),
        KvGeometry::new(4, 4096).unwrap(),
    )
    .unwrap();
    scheduler
        .submit(spec(1, 20, 1).with_prompt_scoring())
        .unwrap();
    let mut next = 100;
    let mut scored = 0;
    for _ in 0..40 {
        // A new single-chunk request arrives every step. It sorts ahead of
        // the scored request and would take a final row the scored chunk
        // needs; the scored request, having waited longer, holds its rows.
        if scheduler.submit(spec(next, 3, 1)).is_ok() {
            next += 1;
        }
        let Some(batch) = scheduler.next_batch().unwrap() else {
            break;
        };
        let rows: usize = batch
            .items()
            .iter()
            .map(|item| match (item.request_id().get(), item.phase()) {
                (1, BatchPhase::Prefill { tokens }) => {
                    scored += tokens;
                    tokens
                }
                (_, BatchPhase::Prefill { .. }) => usize::from(item.expects_output()),
                _ => 0,
            })
            .sum();
        assert!(rows <= 2);
        let prepared = scheduler
            .preflight_commit_batch(batch.id(), &outcomes_for(&batch))
            .unwrap();
        scheduler.apply_prepared_commit(prepared).unwrap();
        if scored == 20 {
            break;
        }
    }
    assert_eq!(scored, 20);
}

/// Commits one batch as fully successful and returns its items.
fn commit_all(scheduler: &mut Scheduler) -> Vec<(u64, BatchPhase)> {
    let batch = scheduler.next_batch().unwrap().unwrap();
    let trace = batch
        .items()
        .iter()
        .map(|item| (item.request_id().get(), item.phase()))
        .collect();
    let prepared = scheduler
        .preflight_commit_batch(batch.id(), &outcomes_for(&batch))
        .unwrap();
    scheduler.apply_prepared_commit(prepared).unwrap();
    scheduler.audit().unwrap();
    trace
}

#[test]
fn ordinary_final_rows_are_not_starved_by_arriving_scored_chunks() {
    // Two slots; a new scored prompt that fills the whole row cap arrives
    // before every batch and is retired once it finishes.
    let mut scheduler = Scheduler::new(
        SchedulerConfig::new(2, 256, 8192, 2048, 1)
            .unwrap()
            .with_maximum_scored_rows(512)
            .unwrap(),
        KvGeometry::new(4, 8192).unwrap(),
    )
    .unwrap();
    scheduler.submit(spec(1, 10, 1)).unwrap();
    let mut ordinary_batch = None;
    for step in 0..40_u64 {
        scheduler
            .submit(spec(100 + step, 512, 1).with_prompt_scoring())
            .unwrap();
        let trace = commit_all(&mut scheduler);
        if trace.contains(&(1, BatchPhase::Prefill { tokens: 10 })) {
            ordinary_batch.get_or_insert(step);
        }
        for id in 100..=100 + step {
            if scheduler.request_state(RequestId::new(id)) == Ok(RequestState::Finished) {
                scheduler.retire_terminal(RequestId::new(id)).unwrap();
            }
        }
    }
    // The ordinary request waited longest, so its row was held for it.
    assert_eq!(ordinary_batch, Some(0));
    // Scored arrivals still finish one per batch after it.
    assert!(
        (100..139).all(|id| scheduler.request_state(RequestId::new(id)).is_err()),
        "every earlier scored request finished and was retired"
    );
}

#[test]
fn a_row_blocked_first_candidate_does_not_keep_prefill_behind_decode() {
    // The scored chunk fills the whole token budget, so it can only run in
    // a prefill-first step; the ordinary request ahead of it cannot take the
    // rows the scored request holds.
    let mut scheduler = Scheduler::new(
        SchedulerConfig::new(3, 256, 512, 512, 1)
            .unwrap()
            .with_maximum_scored_rows(512)
            .unwrap(),
        KvGeometry::new(4, 8192).unwrap(),
    )
    .unwrap();
    scheduler.submit(spec(10, 1, 1000)).unwrap();
    commit_all(&mut scheduler);
    scheduler
        .submit(spec(1, 1024, 1).with_prompt_scoring())
        .unwrap();
    while scheduler.kv_tokens(RequestId::new(1)).unwrap() == 0 {
        commit_all(&mut scheduler);
    }
    scheduler.submit(spec(20, 1, 1)).unwrap();
    let mut decodes = 0;
    for _ in 0..4 {
        let trace = commit_all(&mut scheduler);
        decodes += usize::from(trace.contains(&(10, BatchPhase::Decode)));
    }
    // The decoder keeps progressing, every other step at least.
    assert!(decodes >= 2);
    // Both prefill requests finished within four steps.
    for id in [1, 20] {
        assert_eq!(
            scheduler.request_state(RequestId::new(id)).unwrap(),
            RequestState::Finished,
            "request {id}"
        );
    }
}

/// A started prompt whose chunk fills the whole step budget, with a new
/// one-token prompt arriving before every step. New arrivals must not take
/// the budget the older prefill needs on every step: after four steps
/// without a chunk it goes first, on the ordinary path and on the TBT path
/// without decoding requests. Its chunks keep the configured size.
#[test]
fn a_stream_of_one_token_prompts_cannot_starve_a_started_prefill() {
    use runtime_scheduler::{MixedPolicy, PrefillBudget};
    for policy in [
        PrefillBudget::Fixed,
        PrefillBudget::Tbt(MixedPolicy::new(150, 512, 0).unwrap()),
    ] {
        let mut scheduler = Scheduler::new(
            SchedulerConfig::new(8, 256, 2048, 2048, 1)
                .unwrap()
                .with_prefill_budget(policy),
            KvGeometry::new(16, 4096).unwrap(),
        )
        .unwrap();
        scheduler.submit(spec(1, 8192, 1)).unwrap();
        assert_eq!(
            commit_all(&mut scheduler),
            [(1, BatchPhase::Prefill { tokens: 2048 })]
        );
        let old = RequestId::new(1);
        let mut waited = 0;
        let mut longest = 0;
        let mut chunks = vec![2048];
        for id in 100..164 {
            scheduler.submit(spec(id, 1, 1)).unwrap();
            let trace = commit_all(&mut scheduler);
            match trace.iter().find(|(request, _)| *request == 1) {
                Some((_, BatchPhase::Prefill { tokens })) => {
                    chunks.push(*tokens);
                    waited = 0;
                }
                _ => {
                    waited += 1;
                    longest = longest.max(waited);
                }
            }
            if scheduler.request_state(old) == Ok(RequestState::Finished) {
                break;
            }
            scheduler.reap_terminal(64).unwrap();
        }
        assert!(
            longest <= 4,
            "{policy:?}: the started prefill waited {longest} steps in a row"
        );
        assert_eq!(chunks, [2048; 4], "{policy:?}");
        assert_eq!(scheduler.request_state(old), Ok(RequestState::Finished));
    }
}
