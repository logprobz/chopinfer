use runtime_kv::{BlockId, KvError, KvGeometry, KvPool, SequenceHandle};

fn pool(block_tokens: usize, block_capacity: usize, sequence_capacity: usize) -> KvPool {
    KvPool::new(
        KvGeometry::new(block_tokens, block_capacity).unwrap(),
        sequence_capacity,
    )
    .unwrap()
}

#[test]
fn geometry_rejects_zero_and_overflow() {
    assert_eq!(KvGeometry::new(0, 1), Err(KvError::ZeroBlockTokens));
    assert_eq!(KvGeometry::new(1, 0), Err(KvError::ZeroBlockCapacity));
    assert_eq!(
        KvGeometry::new(usize::MAX, 2),
        Err(KvError::TokenCapacityOverflow)
    );
    assert_eq!(
        KvPool::new(KvGeometry::new(4, 2).unwrap(), 0),
        Err(KvError::ZeroSequenceCapacity)
    );
}

#[test]
fn exhaustion_and_quota_fail_without_partial_mutation() {
    let mut pool = pool(4, 2, 2);
    let first = pool.create_sequence(8).unwrap();
    let second = pool.create_sequence(8).unwrap();
    pool.append(first, 5).unwrap();
    let before = pool.snapshot(second).unwrap();
    assert_eq!(
        pool.append(second, 1),
        Err(KvError::PoolExhausted { needed: 1, free: 0 })
    );
    assert_eq!(pool.snapshot(second).unwrap(), before);
    assert_eq!(
        pool.append(first, 4),
        Err(KvError::QuotaExceeded {
            quota: 8,
            requested: 9
        })
    );
    assert_eq!(pool.snapshot(first).unwrap().tokens(), 5);
    pool.audit().unwrap();
}

#[test]
fn append_spans_blocks_and_reserve_is_checked() {
    let mut pool = pool(4, 4, 1);
    let handle = pool.create_sequence(12).unwrap();
    let report = pool.append(handle, 9).unwrap();
    assert_eq!(
        report
            .allocated()
            .iter()
            .map(|id| id.index())
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert!(report.copy().is_none());
    assert_eq!(pool.snapshot(handle).unwrap().tokens(), 9);
    assert_eq!(pool.reserve(handle, 3).unwrap().new_tokens(), 12);
    assert_eq!(
        pool.reserve(handle, 1),
        Err(KvError::QuotaExceeded {
            quota: 12,
            requested: 13
        })
    );
}

#[test]
fn fork_shares_full_blocks() {
    let mut pool = pool(4, 4, 2);
    let source = pool.create_sequence(8).unwrap();
    pool.append(source, 4).unwrap();
    let child = pool.fork(source, 8).unwrap();
    let shared = pool.snapshot(source).unwrap().blocks()[0];
    assert_eq!(pool.snapshot(child).unwrap().blocks(), &[shared]);
    assert_eq!(pool.block_ref_count(shared).unwrap(), 2);
    let report = pool.append(child, 4).unwrap();
    assert!(report.copy().is_none());
    assert_eq!(pool.snapshot(source).unwrap().blocks(), &[shared]);
    assert_eq!(pool.snapshot(child).unwrap().blocks()[0], shared);
    pool.audit().unwrap();
}

#[test]
fn shared_partial_block_is_copied_before_append() {
    let mut pool = pool(4, 4, 2);
    let source = pool.create_sequence(8).unwrap();
    pool.append(source, 3).unwrap();
    let child = pool.fork(source, 8).unwrap();
    let original = pool.snapshot(source).unwrap().blocks()[0];
    let report = pool.append(child, 1).unwrap();
    let copy = report.copy().unwrap();
    assert_eq!(copy.source(), original);
    assert_eq!(copy.initialized_tokens(), 3);
    assert_ne!(copy.destination(), original);
    assert_eq!(pool.snapshot(source).unwrap().blocks(), &[original]);
    assert_eq!(
        pool.snapshot(child).unwrap().blocks(),
        &[copy.destination()]
    );
    assert_eq!(pool.block_ref_count(original).unwrap(), 1);
    pool.audit().unwrap();
}

/// With last-page rewrites a full shared last page is copied like a partial
/// tail (every row initialized), and each sharer that appends gets its own.
#[test]
fn a_shared_full_last_page_is_copied_when_appends_rewrite_it() {
    let mut pool = pool(4, 5, 2).with_last_page_rewrites(true);
    let source = pool.create_sequence(12).unwrap();
    pool.append(source, 4).unwrap();
    let child = pool.fork(source, 12).unwrap();
    let original = pool.snapshot(source).unwrap().blocks()[0];
    let report = pool.append(child, 1).unwrap();
    let copy = report.copy().unwrap();
    assert_eq!(copy.source(), original);
    assert_eq!(copy.initialized_tokens(), 4);
    // The copy, then the page for the new row.
    assert_eq!(report.allocated().len(), 2);
    assert_eq!(report.allocated()[0], copy.destination());
    assert_eq!(pool.snapshot(child).unwrap().blocks(), report.allocated());
    // The last sharer writes the page it now holds alone, in place.
    assert_eq!(pool.block_ref_count(original).unwrap(), 1);
    let report = pool.append(source, 1).unwrap();
    assert!(report.copy().is_none());
    assert_eq!(pool.snapshot(source).unwrap().blocks()[0], original);
    pool.audit().unwrap();
}

#[test]
fn stale_handles_double_release_and_deterministic_reuse_are_rejected_or_stable() {
    let mut pool = pool(4, 3, 1);
    let old = pool.create_sequence(8).unwrap();
    pool.append(old, 5).unwrap();
    assert_eq!(
        pool.snapshot(old)
            .unwrap()
            .blocks()
            .iter()
            .map(|id| id.index())
            .collect::<Vec<_>>(),
        [0, 1]
    );
    pool.release(old).unwrap();
    assert_eq!(pool.release(old), Err(KvError::StaleHandle));
    let current = pool.create_sequence(8).unwrap();
    assert_ne!(old.generation(), current.generation());
    assert_eq!(pool.append(old, 1), Err(KvError::StaleHandle));
    assert_eq!(
        pool.append(current, 5)
            .unwrap()
            .allocated()
            .iter()
            .map(|id| id.index())
            .collect::<Vec<_>>(),
        [0, 1]
    );
    pool.cancel(current).unwrap();
    assert_eq!(pool.stats().free_blocks(), 3);
    pool.audit().unwrap();
}

#[test]
fn batch_append_is_atomic_and_rejects_duplicate_handles() {
    let mut pool = pool(4, 2, 2);
    let first = pool.create_sequence(8).unwrap();
    let second = pool.create_sequence(8).unwrap();
    let before = pool.stats();
    assert_eq!(
        pool.append_batch(&[(first, 5), (second, 4)]),
        Err(KvError::PoolExhausted { needed: 3, free: 2 })
    );
    assert_eq!(pool.stats(), before);
    assert_eq!(pool.snapshot(first).unwrap().tokens(), 0);
    assert_eq!(
        pool.append_batch(&[(first, 1), (first, 1)]),
        Err(KvError::DuplicateBatchHandle)
    );
    pool.audit().unwrap();
}

#[test]
fn prepared_plan_is_visible_before_exact_commit() {
    let mut pool = pool(4, 4, 2);
    let first = pool.create_sequence(8).unwrap();
    let second = pool.create_sequence(8).unwrap();
    let plan = pool
        .prepare_append_batch(&[(first, 5), (second, 3)])
        .unwrap();
    assert_eq!(pool.prepared_id(), Some(plan.id()));
    assert_eq!(plan.appends().len(), 2);
    assert_eq!(plan.appends()[0].handle(), first);
    assert_eq!(plan.appends()[0].previous_tokens(), 0);
    assert_eq!(plan.appends()[0].additional_tokens(), 5);
    assert_eq!(plan.appends()[0].new_tokens(), 5);
    assert_eq!(
        plan.appends()[0]
            .allocated()
            .iter()
            .map(|block| block.index())
            .collect::<Vec<_>>(),
        [0, 1]
    );
    assert_eq!(
        plan.appends()[1]
            .allocated()
            .iter()
            .map(|block| block.index())
            .collect::<Vec<_>>(),
        [2]
    );
    assert_eq!(pool.snapshot(first).unwrap().tokens(), 0);
    assert_eq!(pool.snapshot(second).unwrap().tokens(), 0);
    pool.audit().unwrap();

    let reports = pool.commit_prepared(plan.id()).unwrap();
    assert_eq!(reports[0].allocated(), plan.appends()[0].allocated());
    assert_eq!(reports[1].allocated(), plan.appends()[1].allocated());
    assert_eq!(
        pool.snapshot(first).unwrap().blocks(),
        plan.appends()[0].allocated()
    );
    assert_eq!(
        pool.snapshot(second).unwrap().blocks(),
        plan.appends()[1].allocated()
    );
    assert_eq!(pool.prepared_id(), None);
    pool.audit().unwrap();
}

#[test]
fn prevalidated_commit_rejects_foreign_and_stale_tokens_before_mutation() {
    let mut first = pool(4, 4, 1);
    let mut second = pool(4, 4, 1);
    let first_handle = first.create_sequence(8).unwrap();
    let second_handle = second.create_sequence(8).unwrap();
    let first_plan = first.prepare_append_batch(&[(first_handle, 4)]).unwrap();
    let second_plan = second.prepare_append_batch(&[(second_handle, 4)]).unwrap();
    let foreign = first
        .preflight_commit_prepared(first_plan.id(), &[])
        .unwrap();

    assert_eq!(
        second.apply_prepared_commit(foreign),
        Err(KvError::PreparedTransactionMismatch)
    );
    assert_eq!(second.prepared_id(), Some(second_plan.id()));
    assert_eq!(second.snapshot(second_handle).unwrap().tokens(), 0);

    let stale = second
        .preflight_commit_prepared(second_plan.id(), &[second_handle])
        .unwrap();
    second.rollback_prepared(second_plan.id()).unwrap();
    assert_eq!(
        second.apply_prepared_commit(stale),
        Err(KvError::NoPreparedTransaction)
    );
    assert_eq!(second.snapshot(second_handle).unwrap().tokens(), 0);
    first.rollback_prepared(first_plan.id()).unwrap();
    first.audit().unwrap();
    second.audit().unwrap();
}

#[test]
fn prevalidated_commit_applies_append_and_release_without_a_fallible_tail() {
    let mut pool = pool(4, 4, 1);
    let handle = pool.create_sequence(8).unwrap();
    let plan = pool.prepare_append_batch(&[(handle, 4)]).unwrap();
    let prepared = pool
        .preflight_commit_prepared(plan.id(), &[handle])
        .unwrap();

    pool.apply_prepared_commit(prepared).unwrap();

    assert_eq!(pool.prepared_id(), None);
    assert_eq!(pool.stats().active_sequences(), 0);
    assert_eq!(pool.stats().free_blocks(), 4);
    assert_eq!(pool.snapshot(handle), Err(KvError::StaleHandle));
    pool.audit().unwrap();
}

#[test]
fn prepared_multi_step_append_exposes_complete_post_append_block_table() {
    let mut pool = pool(4, 6, 1);
    let handle = pool.create_sequence(12).unwrap();

    let first = pool.prepare_append_batch(&[(handle, 5)]).unwrap();
    assert_eq!(
        first.appends()[0]
            .block_table()
            .iter()
            .map(|block| block.index())
            .collect::<Vec<_>>(),
        [0, 1]
    );
    assert_eq!(
        first.appends()[0]
            .newly_allocated()
            .iter()
            .map(|block| block.index())
            .collect::<Vec<_>>(),
        [0, 1]
    );
    pool.commit_prepared(first.id()).unwrap();

    let within_tail = pool.prepare_append_batch(&[(handle, 3)]).unwrap();
    assert_eq!(
        within_tail.appends()[0]
            .block_table()
            .iter()
            .map(|block| block.index())
            .collect::<Vec<_>>(),
        [0, 1]
    );
    assert!(within_tail.appends()[0].newly_allocated().is_empty());
    pool.commit_prepared(within_tail.id()).unwrap();

    let next_block = pool.prepare_append_batch(&[(handle, 1)]).unwrap();
    assert_eq!(
        next_block.appends()[0]
            .block_table()
            .iter()
            .map(|block| block.index())
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert_eq!(
        next_block.appends()[0]
            .newly_allocated()
            .iter()
            .map(|block| block.index())
            .collect::<Vec<_>>(),
        [2]
    );
    pool.commit_prepared(next_block.id()).unwrap();
    assert_eq!(
        pool.snapshot(handle).unwrap().blocks(),
        next_block.appends()[0].block_table()
    );
    pool.audit().unwrap();
}

#[test]
fn prepared_partial_tail_exposes_copy_before_commit() {
    let mut pool = pool(4, 4, 2);
    let source = pool.create_sequence(8).unwrap();
    pool.append(source, 3).unwrap();
    let child = pool.fork(source, 8).unwrap();
    let original = pool.snapshot(child).unwrap().blocks()[0];
    let plan = pool.prepare_append_batch(&[(child, 1)]).unwrap();
    let copy = plan.appends()[0].copy().unwrap();
    assert_eq!(copy.source(), original);
    assert_eq!(copy.destination(), plan.appends()[0].allocated()[0]);
    assert_eq!(copy.initialized_tokens(), 3);
    assert_eq!(pool.snapshot(child).unwrap().blocks(), &[original]);
    pool.commit_prepared(plan.id()).unwrap();
    assert_eq!(
        pool.snapshot(child).unwrap().blocks(),
        &[copy.destination()]
    );
    pool.audit().unwrap();
}

#[test]
fn prepared_shared_partial_tail_replaces_copy_in_complete_block_table() {
    let mut pool = pool(4, 6, 2);
    let source = pool.create_sequence(12).unwrap();
    pool.append(source, 7).unwrap();
    let child = pool.fork(source, 12).unwrap();
    let source_blocks = pool.snapshot(source).unwrap().blocks().to_vec();

    let plan = pool.prepare_append_batch(&[(child, 2)]).unwrap();
    let append = &plan.appends()[0];
    let copy = append.copy().unwrap();
    assert_eq!(copy.source(), source_blocks[1]);
    assert_eq!(copy.destination(), append.newly_allocated()[0]);
    assert_eq!(copy.initialized_tokens(), 3);
    assert_eq!(append.block_table()[0], source_blocks[0]);
    assert_eq!(append.block_table()[1], copy.destination());
    assert_eq!(append.block_table()[2], append.newly_allocated()[1]);
    assert_eq!(pool.snapshot(child).unwrap().blocks(), source_blocks);

    pool.commit_prepared(plan.id()).unwrap();
    assert_eq!(pool.snapshot(child).unwrap().blocks(), append.block_table());
    assert_eq!(pool.snapshot(source).unwrap().blocks(), source_blocks);
    pool.audit().unwrap();
}

#[test]
fn rollback_reuses_reserved_ids_and_rejects_stale_or_mismatched_plans() {
    let mut pool = pool(4, 3, 1);
    let handle = pool.create_sequence(8).unwrap();
    let first = pool.prepare_append_batch(&[(handle, 5)]).unwrap();
    assert_eq!(
        pool.commit_prepared(first.id().next()),
        Err(KvError::PreparedTransactionMismatch)
    );
    pool.audit().unwrap();
    pool.rollback_prepared(first.id()).unwrap();
    assert_eq!(
        pool.commit_prepared(first.id()),
        Err(KvError::NoPreparedTransaction)
    );
    let second = pool.prepare_append_batch(&[(handle, 5)]).unwrap();
    assert_eq!(
        first.appends()[0].allocated(),
        second.appends()[0].allocated()
    );
    pool.rollback_prepared(second.id()).unwrap();
    assert_eq!(pool.stats().free_blocks(), 3);
    pool.audit().unwrap();
}

#[test]
fn cancellation_discards_the_whole_prepared_batch_without_leaks() {
    let mut pool = pool(4, 4, 2);
    let first = pool.create_sequence(8).unwrap();
    let second = pool.create_sequence(8).unwrap();
    let plan = pool
        .prepare_append_batch(&[(first, 4), (second, 4)])
        .unwrap();
    assert_eq!(
        pool.release(first),
        Err(KvError::PreparedTransactionPending)
    );
    assert_eq!(
        pool.append(second, 1),
        Err(KvError::PreparedTransactionPending)
    );
    pool.cancel(first).unwrap();
    assert_eq!(pool.prepared_id(), None);
    assert_eq!(pool.snapshot(second).unwrap().tokens(), 0);
    assert_eq!(pool.stats().free_blocks(), 4);
    assert_eq!(
        pool.commit_prepared(plan.id()),
        Err(KvError::NoPreparedTransaction)
    );
    pool.audit().unwrap();
}

#[test]
fn deterministic_state_machine_sequences_preserve_invariants() {
    for seed in 0..32_u64 {
        let mut pool = pool(8, 24, 8);
        let mut handles = Vec::new();
        let mut state = seed | 1;
        for _ in 0..256 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            match state % 4 {
                0 if handles.len() < 8 => {
                    if let Ok(handle) = pool.create_sequence(32) {
                        handles.push(handle);
                    }
                }
                1 if !handles.is_empty() => {
                    let index = (state as usize) % handles.len();
                    let _ = pool.append(handles[index], (state as usize % 3) + 1);
                }
                2 if !handles.is_empty() && handles.len() < 8 => {
                    let index = (state as usize) % handles.len();
                    if let Ok(handle) = pool.fork(handles[index], 32) {
                        handles.push(handle);
                    }
                }
                3 if !handles.is_empty() => {
                    let index = (state as usize) % handles.len();
                    let handle = handles.swap_remove(index);
                    pool.release(handle).unwrap();
                }
                _ => {}
            }
            pool.audit().unwrap();
        }
        for handle in handles {
            pool.release(handle).unwrap();
        }
        assert_eq!(pool.stats().used_blocks(), 0);
        pool.audit().unwrap();
    }
}

/// Forks `owners - 1` children from a sequence holding one partial page, so
/// every owner shares that tail.
fn shared_partial_tail(pool: &mut KvPool, owners: usize) -> (Vec<SequenceHandle>, BlockId) {
    let first = pool.create_sequence(8).unwrap();
    pool.append(first, 1).unwrap();
    let tail = pool.snapshot(first).unwrap().blocks()[0];
    let mut handles = vec![first];
    for _ in 1..owners {
        handles.push(pool.fork(first, 8).unwrap());
    }
    (handles, tail)
}

#[test]
fn replacing_every_owner_of_a_shared_tail_frees_the_old_tail() {
    for checked in [false, true] {
        let mut pool = pool(4, 3, 2);
        for _ in 0..4 {
            let (handles, tail) = shared_partial_tail(&mut pool, 2);
            let appends: Vec<_> = handles.iter().map(|handle| (*handle, 1)).collect();
            let plan = pool.prepare_append_batch(&appends).unwrap();
            assert!(plan.appends().iter().all(|append| append.copy().is_some()));
            if checked {
                let prepared = pool.preflight_commit_prepared(plan.id(), &[]).unwrap();
                pool.apply_prepared_commit(prepared).unwrap();
            } else {
                pool.commit_prepared(plan.id()).unwrap();
            }
            pool.audit().unwrap();
            assert_eq!(pool.block_ref_count(tail).unwrap(), 0);
            assert_eq!(pool.stats().free_blocks(), 1);
            for handle in handles {
                pool.release(handle).unwrap();
            }
            pool.audit().unwrap();
            assert_eq!(pool.stats().free_blocks(), 3);
        }
    }
}

#[test]
fn a_rolled_back_all_owner_copy_keeps_the_shared_tail() {
    let mut pool = pool(4, 3, 2);
    let (handles, tail) = shared_partial_tail(&mut pool, 2);
    let plan = pool
        .prepare_append_batch(&[(handles[0], 1), (handles[1], 1)])
        .unwrap();
    pool.rollback_prepared(plan.id()).unwrap();
    pool.audit().unwrap();
    assert_eq!(pool.block_ref_count(tail).unwrap(), 2);
    assert_eq!(pool.stats().free_blocks(), 2);
    for handle in handles {
        pool.release(handle).unwrap();
    }
    pool.audit().unwrap();
    assert_eq!(pool.stats().free_blocks(), 3);
}

#[test]
fn a_released_owner_in_an_all_owner_copy_returns_every_page() {
    let mut pool = pool(4, 4, 3);
    let (handles, tail) = shared_partial_tail(&mut pool, 3);
    let appends: Vec<_> = handles.iter().map(|handle| (*handle, 1)).collect();
    let plan = pool.prepare_append_batch(&appends).unwrap();
    let prepared = pool
        .preflight_commit_prepared(plan.id(), &[handles[1]])
        .unwrap();
    pool.apply_prepared_commit(prepared).unwrap();
    pool.audit().unwrap();
    assert_eq!(pool.block_ref_count(tail).unwrap(), 0);
    assert_eq!(pool.stats().free_blocks(), 2);
    pool.release(handles[0]).unwrap();
    pool.release(handles[2]).unwrap();
    pool.audit().unwrap();
    assert_eq!(pool.stats().free_blocks(), 4);
}
