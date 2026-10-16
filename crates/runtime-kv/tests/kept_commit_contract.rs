use runtime_kv::{KvError, KvGeometry, KvPool, SequenceHandle};

fn pool(block_tokens: usize, block_capacity: usize, sequence_capacity: usize) -> KvPool {
    KvPool::new(
        KvGeometry::new(block_tokens, block_capacity).unwrap(),
        sequence_capacity,
    )
    .unwrap()
}

fn commit_kept(
    pool: &mut KvPool,
    appends: &[(SequenceHandle, usize)],
    kept: &[(SequenceHandle, usize)],
) {
    let plan = pool.prepare_append_batch(appends).unwrap();
    let prepared = pool
        .preflight_commit_prepared_kept(plan.id(), kept, &[])
        .unwrap();
    pool.apply_prepared_commit(prepared).unwrap();
    pool.audit().unwrap();
}

#[test]
fn keeping_every_token_matches_an_ordinary_commit() {
    let mut kept = pool(4, 8, 1);
    let mut whole = pool(4, 8, 1);
    let kept_handle = kept.create_sequence(16).unwrap();
    let whole_handle = whole.create_sequence(16).unwrap();
    commit_kept(&mut kept, &[(kept_handle, 3)], &[]);
    commit_kept(&mut whole, &[(whole_handle, 3)], &[]);
    commit_kept(&mut kept, &[(kept_handle, 7)], &[(kept_handle, 7)]);
    commit_kept(&mut whole, &[(whole_handle, 7)], &[]);

    assert_eq!(
        kept.snapshot(kept_handle).unwrap(),
        whole.snapshot(whole_handle).unwrap()
    );
    assert_eq!(kept.stats().free_blocks(), whole.stats().free_blocks());
}

#[test]
fn a_short_kept_prefix_frees_whole_blocks_past_it() {
    let mut pool = pool(4, 8, 1);
    let handle = pool.create_sequence(16).unwrap();
    commit_kept(&mut pool, &[(handle, 3)], &[]);
    let before = pool.snapshot(handle).unwrap();
    let free_before = pool.stats().free_blocks();

    // Tokens 3..12 need blocks 1 and 2; keeping one token needs none of them.
    commit_kept(&mut pool, &[(handle, 9)], &[(handle, 1)]);

    let after = pool.snapshot(handle).unwrap();
    assert_eq!(after.tokens(), 4);
    assert_eq!(after.blocks(), before.blocks());
    assert_eq!(pool.stats().free_blocks(), free_before);

    // Keeping two more tokens crosses into one new block and frees the other.
    commit_kept(&mut pool, &[(handle, 8)], &[(handle, 2)]);
    let after = pool.snapshot(handle).unwrap();
    assert_eq!(after.tokens(), 6);
    assert_eq!(after.blocks().len(), 2);
    assert_eq!(pool.stats().free_blocks(), free_before - 1);
}

#[test]
fn a_kept_prefix_inside_a_copied_tail_keeps_the_copy() {
    let mut pool = pool(4, 8, 2);
    let source = pool.create_sequence(16).unwrap();
    commit_kept(&mut pool, &[(source, 6)], &[]);
    let child = pool.fork(source, 16).unwrap();
    let shared_tail = pool.snapshot(source).unwrap().blocks()[1];

    let plan = pool.prepare_append_batch(&[(child, 7)]).unwrap();
    let copy = plan.appends()[0].copy().unwrap();
    assert_eq!(copy.source(), shared_tail);
    let prepared = pool
        .preflight_commit_prepared_kept(plan.id(), &[(child, 1)], &[])
        .unwrap();
    pool.apply_prepared_commit(prepared).unwrap();
    pool.audit().unwrap();

    let child_blocks = pool.snapshot(child).unwrap();
    assert_eq!(child_blocks.tokens(), 7);
    assert_eq!(child_blocks.blocks()[1], copy.destination());
    assert_eq!(pool.snapshot(source).unwrap().blocks()[1], shared_tail);
    pool.release(source).unwrap();
    pool.release(child).unwrap();
    pool.audit().unwrap();
    assert_eq!(pool.stats().free_blocks(), 8);
}

#[test]
fn kept_counts_apply_per_handle() {
    let mut pool = pool(2, 16, 2);
    let first = pool.create_sequence(16).unwrap();
    let second = pool.create_sequence(16).unwrap();
    commit_kept(&mut pool, &[(first, 5), (second, 5)], &[(second, 2)]);
    assert_eq!(pool.snapshot(first).unwrap().tokens(), 5);
    assert_eq!(pool.snapshot(second).unwrap().tokens(), 2);
    assert_eq!(pool.stats().free_blocks(), 16 - 3 - 1);
}

#[test]
fn invalid_kept_counts_are_refused_before_mutation() {
    let mut pool = pool(4, 8, 2);
    let handle = pool.create_sequence(16).unwrap();
    let other = pool.create_sequence(16).unwrap();
    let plan = pool.prepare_append_batch(&[(handle, 3)]).unwrap();
    assert_eq!(
        pool.preflight_commit_prepared_kept(plan.id(), &[(handle, 0)], &[])
            .unwrap_err(),
        KvError::InvalidKeptTokens {
            kept: 0,
            additional: 3
        }
    );
    assert_eq!(
        pool.preflight_commit_prepared_kept(plan.id(), &[(handle, 4)], &[])
            .unwrap_err(),
        KvError::InvalidKeptTokens {
            kept: 4,
            additional: 3
        }
    );
    assert_eq!(
        pool.preflight_commit_prepared_kept(plan.id(), &[(other, 1)], &[])
            .unwrap_err(),
        KvError::InvariantViolation
    );
    assert_eq!(
        pool.preflight_commit_prepared_kept(plan.id(), &[(handle, 1), (handle, 2)], &[])
            .unwrap_err(),
        KvError::DuplicateBatchHandle
    );
    assert_eq!(pool.prepared_id(), Some(plan.id()));
    pool.rollback_prepared(plan.id()).unwrap();
    pool.audit().unwrap();
    assert_eq!(pool.stats().free_blocks(), 8);
}

#[test]
fn repeated_partial_commits_return_every_block() {
    let mut pool = pool(4, 256, 4);
    let handles: Vec<_> = (0..4).map(|_| pool.create_sequence(200).unwrap()).collect();
    for step in 0..40usize {
        let appends: Vec<_> = handles.iter().map(|handle| (*handle, 5)).collect();
        let kept: Vec<_> = handles
            .iter()
            .enumerate()
            .map(|(index, handle)| (*handle, 1 + (step * 7 + index * 3) % 5))
            .collect();
        commit_kept(&mut pool, &appends, &kept);
    }
    for handle in handles {
        pool.release(handle).unwrap();
    }
    pool.audit().unwrap();
    assert_eq!(pool.stats().free_blocks(), 256);
}

#[test]
fn a_kept_commit_that_copies_every_owner_of_a_tail_frees_the_old_tail() {
    let mut pool = pool(4, 6, 2);
    for _ in 0..3 {
        let source = pool.create_sequence(16).unwrap();
        commit_kept(&mut pool, &[(source, 2)], &[]);
        let child = pool.fork(source, 16).unwrap();
        let shared_tail = pool.snapshot(source).unwrap().blocks()[0];

        // Each owner copies the tail and drafts into a second page; one
        // keeps a row inside the copy and the other crosses into its second
        // page.
        commit_kept(
            &mut pool,
            &[(source, 5), (child, 5)],
            &[(source, 1), (child, 3)],
        );
        assert_eq!(pool.block_ref_count(shared_tail).unwrap(), 0);
        assert_eq!(pool.snapshot(source).unwrap().tokens(), 3);
        assert_eq!(pool.snapshot(child).unwrap().tokens(), 5);
        assert_eq!(pool.stats().free_blocks(), 6 - 1 - 2);
        pool.release(source).unwrap();
        pool.release(child).unwrap();
        pool.audit().unwrap();
        assert_eq!(pool.stats().free_blocks(), 6);
    }
}
