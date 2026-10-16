use runtime_kv::{
    BlockId, ContextParallelPagesV1, DeclaredPageLayoutV1, HostTier, KvPool, KvTransferStatus,
    PagedKvPlanV1, PrefixCacheKeyV1, RankPageV1, RankPlacementV1, StateLifetimeV1,
};

fn layout() -> DeclaredPageLayoutV1 {
    DeclaredPageLayoutV1::new(
        256,
        4,
        16,
        256,
        4,
        RankPlacementV1::Sharded,
        StateLifetimeV1::Request,
        4_096,
    )
    .expect("the DCP4 page-layout fixture must be valid")
}

fn key(n: u64) -> PrefixCacheKeyV1 {
    let mut bytes = [0; 32];
    bytes[..8].copy_from_slice(&n.to_le_bytes());
    PrefixCacheKeyV1::new(bytes)
}

/// Every position of a `tokens`-token sequence over `table` is held by
/// exactly one group rank, in a page at the stripe's local index.
fn assert_owned_once(pages: ContextParallelPagesV1, table: &[BlockId], tokens: usize) {
    let mut holders: Vec<Option<usize>> = vec![None; tokens];
    for rank in 0..pages.size() {
        let held = pages
            .rank_pages(rank, table.iter().map(|page| page.index()), tokens)
            .unwrap();
        assert_eq!(
            held.iter().map(|page| page.tokens).sum::<usize>(),
            pages.rank_tokens(rank, tokens)
        );
        for page in held {
            let stripe = page.first_position / pages.stripe_tokens();
            assert_eq!(page.local_page, table[stripe].index());
            for (position, holder) in holders
                .iter_mut()
                .enumerate()
                .skip(page.first_position)
                .take(page.tokens)
            {
                assert_eq!(pages.owner(position), rank);
                assert!(holder.replace(rank).is_none());
            }
        }
    }
    assert!(holders.iter().all(Option::is_some));
}

#[test]
fn dcp_sharded_pool_owns_every_page_once_and_counts_capacity_across_ranks() {
    let layout = layout();
    assert_eq!(layout.rank_placement(), RankPlacementV1::Sharded);
    let per_rank_pages = 64;
    let single = ContextParallelPagesV1::new(1, layout.page_tokens(), layout.rank_count()).unwrap();
    let pages = ContextParallelPagesV1::new(4, layout.page_tokens(), layout.rank_count()).unwrap();
    // Each rank holds the same pages and bytes; the group holds four times
    // the tokens.
    let plan = PagedKvPlanV1::validate(layout, per_rank_pages as u64, 1).unwrap();
    assert_eq!(
        plan.bytes_per_rank_pool(),
        per_rank_pages as u64 * layout.page_bytes()
    );
    assert_eq!(
        pages.geometry(per_rank_pages).unwrap().token_capacity(),
        4 * single.geometry(per_rank_pages).unwrap().token_capacity()
    );

    let mut pool = KvPool::new(pages.geometry(per_rank_pages).unwrap(), 8).unwrap();
    let mut sequences = Vec::new();
    for tokens in [1, 16, 17, 63, 64, 65, 200] {
        let sequence = pool.create_sequence(1_024).unwrap();
        pool.append(sequence, tokens).unwrap();
        sequences.push((sequence, tokens));
    }
    for (sequence, tokens) in sequences {
        let snapshot = pool.snapshot(sequence).unwrap();
        assert_eq!(
            snapshot.blocks().len(),
            tokens.div_ceil(pages.stripe_tokens())
        );
        assert_owned_once(pages, snapshot.blocks(), tokens);
    }
    assert_eq!(
        pages.rank_pages(0, [], 1),
        Err(runtime_kv::ContextParallelError::ShortTable {
            pages: 0,
            tokens: 1
        })
    );
}

#[test]
fn prefix_anchors_and_copies_keep_every_position_on_its_owner() {
    let pages = ContextParallelPagesV1::new(2, 4, 4).unwrap();
    let mut pool = KvPool::new(pages.geometry(32).unwrap(), 8).unwrap();
    let source = pool.create_sequence(64).unwrap();
    // Two full stripes and a partial one that ends inside rank 1's page.
    pool.append(source, 22).unwrap();
    let state = pool
        .insert_anchor(key(1), key(1), source, 1)
        .unwrap()
        .expect("the anchor fits");
    assert_eq!(state.len(), 1, "one state page is one page on every rank");
    let hit = pool.resume_anchor(key(1), 64).unwrap();
    pool.finish_restore(hit.handle()).unwrap();
    let shared = pool.snapshot(hit.handle()).unwrap();
    assert_eq!(shared.blocks(), pool.snapshot(source).unwrap().blocks());

    // Appending to the shared partial stripe copies the whole stripe: one
    // local page on each rank, at a new index, holding the same positions.
    let prepared = pool.prepare_append_batch(&[(hit.handle(), 5)]).unwrap();
    let append = &prepared.appends()[0];
    let copy = append.copy().expect("a shared partial stripe is copied");
    assert_eq!(copy.source(), shared.blocks()[2]);
    assert_eq!(copy.initialized_tokens(), 22 % pages.stripe_tokens());
    assert_eq!(append.block_table()[..2], shared.blocks()[..2]);
    assert_eq!(append.block_table()[2], copy.destination());
    assert_owned_once(pages, append.block_table(), 27);
    let indices = |table: &[BlockId]| table.iter().map(|page| page.index()).collect::<Vec<_>>();
    let before = pages.rank_pages(1, indices(shared.blocks()), 22).unwrap();
    let after = pages
        .rank_pages(1, indices(append.block_table()), 27)
        .unwrap();
    assert_eq!(
        before
            .iter()
            .map(|page| page.first_position)
            .collect::<Vec<_>>(),
        after
            .iter()
            .map(|page| page.first_position)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        after.last(),
        Some(&RankPageV1 {
            local_page: copy.destination().index(),
            first_position: 20,
            tokens: 4,
        })
    );
    pool.commit_prepared(prepared.id()).unwrap();
    pool.audit().unwrap();
}

#[test]
fn offload_and_reload_move_whole_stripes() {
    let pages = ContextParallelPagesV1::new(4, 4, 4).unwrap();
    let mut pool = KvPool::new(pages.geometry(16).unwrap(), 8).unwrap();
    let mut host = HostTier::new(&pool, 16).unwrap();
    let sequence = pool.create_sequence(128).unwrap();
    pool.append(sequence, 37).unwrap();
    pool.insert_private_anchor(key(2), sequence, 1)
        .unwrap()
        .expect("the anchor fits");
    let (anchor, transfer) = host.begin_offload(&mut pool, key(2), Some(2)).unwrap();
    // Three stripes of sixteen tokens and one state page.
    assert_eq!(transfer.pages().len(), 4);
    pool.release(sequence).unwrap();
    pool.remove_anchor(key(2)).unwrap();
    host.complete_transfer(
        &mut pool,
        transfer
            .into_submission()
            .completion(KvTransferStatus::Complete),
    )
    .unwrap();

    let destination = pool.create_initialized_sequence(37, 128, 1).unwrap();
    let reload = host.begin_reload(&mut pool, anchor, &destination).unwrap();
    assert_eq!(reload.pages().len(), 4);
    host.complete_transfer(
        &mut pool,
        reload
            .into_submission()
            .completion(KvTransferStatus::Complete),
    )
    .unwrap();
    pool.finish_initialized_restore(destination.handle())
        .unwrap();
    let restored = pool.snapshot(destination.handle()).unwrap();
    assert_eq!(restored.blocks().len(), 3);
    assert_owned_once(pages, restored.blocks(), 37);
    host.audit(&pool).unwrap();
}
