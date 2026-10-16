use runtime_kv::{KvGeometry, KvPool, PrefixCacheKeyV1};

#[test]
fn available_anchors_serve_append_before_a_paced_copy_completes() {
    let mut pool = KvPool::new(KvGeometry::new(4, 8).unwrap(), 4).unwrap();
    let key = PrefixCacheKeyV1::new([9; 32]);
    let cached = pool.create_sequence(4).unwrap();
    pool.append(cached, 4).unwrap();
    pool.insert_anchor(key, key, cached, 1).unwrap().unwrap();
    pool.release(cached).unwrap();
    let victim = pool.create_sequence(16).unwrap();
    pool.append(victim, 16).unwrap();
    let pages = pool
        .snapshot(victim)
        .unwrap()
        .into_blocks()
        .iter()
        .map(|p| pool.page_allocation(*p).unwrap())
        .collect::<Vec<_>>();
    let lease = pool.retain_transfer_pages(&pages).unwrap();
    pool.release(victim).unwrap();
    let next = pool.create_sequence(12).unwrap();
    pool.ensure_append_capacity(&[(next, 12)]).unwrap();
    pool.append(next, 12).unwrap();
    assert!(pool.anchor_pages(key).is_none());
    assert_eq!(pool.metrics().snapshot().prefix_cache_evictions, 1);
    pool.audit().unwrap();
    pool.release_transfer_pages(&lease).unwrap();
    pool.audit().unwrap();
}

#[test]
fn captures_evict_anchors_while_released_pages_are_leased() {
    let mut pool = KvPool::new(KvGeometry::new(4, 8).unwrap(), 4).unwrap();
    let old = PrefixCacheKeyV1::new([9; 32]);
    let cached = pool.create_sequence(4).unwrap();
    pool.append(cached, 4).unwrap();
    pool.insert_anchor(old, old, cached, 1).unwrap().unwrap();
    pool.release(cached).unwrap();
    let victim = pool.create_sequence(16).unwrap();
    pool.append(victim, 16).unwrap();
    let pages = pool
        .snapshot(victim)
        .unwrap()
        .into_blocks()
        .iter()
        .map(|p| pool.page_allocation(*p).unwrap())
        .collect::<Vec<_>>();
    let _lease = pool.retain_transfer_pages(&pages).unwrap();
    pool.release(victim).unwrap();
    let live = pool.create_sequence(8).unwrap();
    pool.append(live, 8).unwrap();
    let new = PrefixCacheKeyV1::new([7; 32]);
    let result = pool.insert_anchor(new, new, live, 1).unwrap();
    assert!(
        result.is_some(),
        "a snapshot must evict instead of waiting for a copy"
    );
    assert!(pool.anchor_pages(old).is_none());
    pool.audit().unwrap();
}

#[test]
fn leased_pages_with_live_owners_do_not_delay_append_eviction() {
    for keep_anchor in [false, true] {
        let mut pool = KvPool::new(KvGeometry::new(4, 8).unwrap(), 4).unwrap();
        let old = PrefixCacheKeyV1::new([9; 32]);
        let cached = pool.create_sequence(4).unwrap();
        pool.append(cached, 4).unwrap();
        pool.insert_anchor(old, old, cached, 1).unwrap().unwrap();
        pool.release(cached).unwrap();
        let victim = pool.create_sequence(16).unwrap();
        pool.append(victim, 16).unwrap();
        let key = PrefixCacheKeyV1::new([8; 32]);
        if keep_anchor {
            pool.insert_anchor(key, key, victim, 0).unwrap().unwrap();
        }
        let pages = pool
            .snapshot(victim)
            .unwrap()
            .into_blocks()
            .iter()
            .map(|p| pool.page_allocation(*p).unwrap())
            .collect::<Vec<_>>();
        let lease = pool.retain_transfer_pages(&pages).unwrap();
        if keep_anchor {
            pool.release(victim).unwrap();
        }
        let next = pool.create_sequence(12).unwrap();
        pool.ensure_append_capacity(&[(next, 12)]).unwrap();
        pool.append(next, 12).unwrap();
        assert!(pool.anchor_pages(old).is_none());
        pool.audit().unwrap();
        pool.release_transfer_pages(&lease).unwrap();
        pool.audit().unwrap();
    }
}

#[test]
fn capped_extras_fit_shared_tails_zero_bases_and_multiple_candidates() {
    for capacity in 4..12 {
        for tokens in 1..8 {
            for base in [0, 1, 3] {
                let mut pool = KvPool::new(KvGeometry::new(4, capacity).unwrap(), 4).unwrap();
                let quota = capacity * 4;
                let first = pool.create_sequence(quota).unwrap();
                pool.append(first, tokens).unwrap();
                let second = pool.fork(first, quota).unwrap();
                let appends = [(first, base), (second, base)];
                let limits = match pool.cap_append_extras(&appends, &[7, 7]) {
                    Ok(limits) => limits,
                    Err(runtime_kv::KvError::PoolExhausted { .. }) => {
                        assert!(matches!(
                            pool.prepare_append_batch(&appends),
                            Err(runtime_kv::KvError::PoolExhausted { .. })
                        ));
                        pool.release(first).unwrap();
                        pool.release(second).unwrap();
                        pool.audit().unwrap();
                        assert_eq!(pool.stats().free_blocks(), capacity);
                        continue;
                    }
                    Err(error) => panic!("unexpected cap failure: {error}"),
                };
                for chosen in [limits.clone(), vec![0, limits[1]], vec![limits[0], 0]] {
                    let prepared = pool
                        .prepare_append_batch(&[
                            (first, base + chosen[0]),
                            (second, base + chosen[1]),
                        ])
                        .unwrap();
                    pool.audit().unwrap();
                    pool.rollback_prepared(prepared.id()).unwrap();
                    pool.audit().unwrap();
                }
                pool.release(first).unwrap();
                pool.release(second).unwrap();
                assert_eq!(pool.stats().free_blocks(), capacity);
            }
        }
    }
}
