use std::sync::Arc;

use runtime_kv::{KvError, KvGeometry, KvPool, PrefixChainV1, PrefixMediaSpanV1};

fn chain(tokens: &[u32]) -> PrefixChainV1 {
    PrefixChainV1::new(4, Arc::from(tokens), Vec::new()).expect("the fixture chain is valid")
}

fn prompt(length: u32) -> Vec<u32> {
    (0..length).map(|token| 100 + token).collect()
}

/// Publishes an anchor over `source`'s committed tokens of `tokens`.
fn capture(
    pool: &mut KvPool,
    tokens: &PrefixChainV1,
    source: runtime_kv::SequenceHandle,
    state_pages: usize,
) -> Option<Vec<runtime_kv::BlockId>> {
    let boundary = pool.snapshot(source).unwrap().tokens();
    pool.insert_content_anchor(tokens, source, boundary, state_pages)
        .expect("a live source can be anchored")
}

#[test]
fn chain_digests_name_tokens_media_and_the_exact_boundary() {
    let tokens = prompt(10);
    let first = chain(&tokens);
    let again = chain(&tokens);
    assert_eq!(first, again, "hashing is deterministic");
    assert_eq!(first.full_blocks(), 2);
    assert_eq!(
        first.key_at(8),
        first.chain(2),
        "an aligned key is its chain"
    );
    assert_ne!(first.key_at(9), first.key_at(10));
    assert_eq!(first.key_at(0), None);
    assert_eq!(first.key_at(11), None);

    let mut changed = tokens.clone();
    changed[5] += 1;
    let changed = chain(&changed);
    assert_eq!(changed.chain(1), first.chain(1));
    assert_ne!(changed.chain(2), first.chain(2));
    assert_ne!(changed.key_at(9), first.key_at(9));

    let media = |identity: u8| {
        PrefixChainV1::new(
            4,
            Arc::from(tokens.as_slice()),
            vec![PrefixMediaSpanV1 {
                identity: [identity; 32],
                start: 5,
                rows: 2,
                media_row: 0,
            }],
        )
        .unwrap()
    };
    let red = media(1);
    let blue = media(2);
    assert_eq!(
        red.chain(1),
        first.chain(1),
        "blocks before the media agree"
    );
    assert_ne!(red.chain(2), blue.chain(2), "the media identity is hashed");
    assert_ne!(red.chain(2), first.chain(2));
    assert!(red.splits_media(6));
    assert!(!red.splits_media(5));
    assert!(!red.splits_media(7));
    assert_eq!(
        PrefixChainV1::new(
            4,
            Arc::from(tokens.as_slice()),
            vec![PrefixMediaSpanV1 {
                identity: [0; 32],
                start: 9,
                rows: 2,
                media_row: 0,
            }],
        ),
        Err(KvError::InvalidPrefixMedia)
    );
}

#[test]
fn a_cache_salt_separates_every_key() {
    let tokens = prompt(10);
    let salted = |salt: &[u8]| {
        PrefixChainV1::salted(4, Arc::from(tokens.as_slice()), Vec::new(), Some(salt)).unwrap()
    };
    let plain = chain(&tokens);
    let (red, blue) = (salted(b"red"), salted(b"blue"));
    assert_eq!(red, salted(b"red"), "a salted chain is deterministic");
    assert_eq!(
        PrefixChainV1::salted(4, Arc::from(tokens.as_slice()), Vec::new(), None).unwrap(),
        plain,
        "no salt is the unsalted chain"
    );
    for boundary in 1..=10 {
        let keys = [&plain, &red, &blue].map(|chain| chain.key_at(boundary).unwrap());
        assert!(
            keys[0] != keys[1] && keys[0] != keys[2] && keys[1] != keys[2],
            "boundary {boundary}"
        );
    }
}

#[test]
fn an_anchor_outlives_its_source_and_resumes_a_longer_prompt() {
    let geometry = KvGeometry::new(4, 16).expect("the prefix fixture geometry is valid");
    let mut pool = KvPool::new(geometry, 4).expect("the prefix fixture pool is valid");
    let first = prompt(10);
    let first_chain = chain(&first);
    let source = pool.create_sequence(16).unwrap();
    pool.append(source, 10).unwrap();
    let source_blocks = pool.snapshot(source).unwrap().into_blocks();
    let state = capture(&mut pool, &first_chain, source, 2).expect("the pool has room");
    assert_eq!(state.len(), 2);
    assert_eq!(
        capture(&mut pool, &first_chain, source, 2),
        None,
        "a key that already has an anchor is not captured twice"
    );
    pool.release(source).unwrap();
    let stats = pool.stats();
    assert_eq!(stats.used_blocks(), 0, "only the anchor holds pages");
    assert_eq!(stats.cached_blocks(), 5);
    assert_eq!(stats.anchors(), 1);
    pool.audit().unwrap();

    let mut second = first.clone();
    second.extend([7, 8, 9]);
    let second_chain = chain(&second);
    assert_eq!(
        pool.lookup_anchor(&second_chain, 12),
        Some((first_chain.key_at(10).unwrap(), 10))
    );
    assert_eq!(
        pool.lookup_anchor(&second_chain, 9),
        None,
        "a hit leaves the requested number of tokens"
    );
    let hit = pool
        .resume_prefix(&second_chain, 12, 16)
        .unwrap()
        .expect("the longer prompt hits");
    assert_eq!(hit.tokens(), 10);
    assert_eq!(hit.state(), state.as_slice());
    let resumed = pool.snapshot(hit.handle()).unwrap();
    assert_eq!(resumed.tokens(), 10);
    assert_eq!(resumed.blocks(), source_blocks.as_slice());
    assert_eq!(
        pool.block_ref_count(state[0]).unwrap(),
        2,
        "the restore pins"
    );
    pool.finish_restore(hit.handle()).unwrap();
    assert_eq!(pool.block_ref_count(state[0]).unwrap(), 1);

    // The shared partial page is copied before the new sequence writes it.
    let prepared = pool.prepare_append_batch(&[(hit.handle(), 3)]).unwrap();
    let copy = prepared.appends()[0].copy().expect("the tail is shared");
    assert_eq!(copy.source(), source_blocks[2]);
    assert_eq!(copy.initialized_tokens(), 2);
    pool.commit_prepared(prepared.id()).unwrap();
    pool.audit().unwrap();

    let snapshot = pool.metrics().snapshot();
    assert_eq!(snapshot.prefix_cache_queries, 1);
    assert_eq!(snapshot.prefix_cache_hits, 1);
    assert_eq!(snapshot.prefix_cache_reused_tokens, 10);
    assert_eq!(snapshot.prefix_cache_captures, 1);

    let unrelated = chain(&prompt(3).iter().map(|token| token + 50).collect::<Vec<_>>());
    assert_eq!(pool.resume_prefix(&unrelated, 2, 16).unwrap(), None);
    assert_eq!(pool.metrics().snapshot().prefix_cache_queries, 2);
}

#[test]
fn lookup_takes_the_longest_anchor_outside_media() {
    let geometry = KvGeometry::new(4, 32).unwrap();
    let mut pool = KvPool::new(geometry, 4).unwrap();
    let tokens = prompt(14);
    let plain = chain(&tokens);
    let source = pool.create_sequence(16).unwrap();
    for step in [4, 5, 3] {
        pool.append(source, step).unwrap();
        capture(&mut pool, &plain, source, 1).expect("room for every anchor");
    }
    assert_eq!(pool.anchor_count(), 3);
    assert_eq!(
        pool.lookup_anchor(&plain, 13).map(|(_, tokens)| tokens),
        Some(12)
    );
    assert_eq!(
        pool.lookup_anchor(&plain, 11).map(|(_, tokens)| tokens),
        Some(9)
    );
    assert_eq!(
        pool.lookup_anchor(&plain, 8).map(|(_, tokens)| tokens),
        Some(4)
    );

    let media = PrefixChainV1::new(
        4,
        Arc::from(tokens.as_slice()),
        vec![PrefixMediaSpanV1 {
            identity: [3; 32],
            start: 11,
            rows: 2,
            media_row: 0,
        }],
    )
    .unwrap();
    // Media changes the digest of the third block, so only the first two
    // anchors can match, and neither splits the span.
    assert_eq!(
        pool.lookup_anchor(&media, 13).map(|(_, tokens)| tokens),
        Some(9)
    );
    pool.audit().unwrap();
}

#[test]
fn appends_evict_the_least_recently_used_anchor_first() {
    let geometry = KvGeometry::new(4, 6).unwrap();
    let mut pool = KvPool::new(geometry, 4).unwrap();
    let mut chains = Vec::new();
    for base in [0, 1000] {
        let tokens: Vec<u32> = prompt(4).iter().map(|token| token + base).collect();
        let tokens = chain(&tokens);
        let source = pool.create_sequence(8).unwrap();
        pool.append(source, 4).unwrap();
        capture(&mut pool, &tokens, source, 1).expect("room for the anchor");
        pool.release(source).unwrap();
        chains.push(tokens);
    }
    assert_eq!(pool.stats().free_blocks(), 2);
    // Reusing the first anchor makes the second the oldest.
    let mut longer = chains[0].tokens().to_vec();
    longer.push(1);
    let hit = pool
        .resume_prefix(&chain(&longer), 4, 8)
        .unwrap()
        .expect("the first anchor hits");
    pool.finish_restore(hit.handle()).unwrap();
    pool.release(hit.handle()).unwrap();

    let sequence = pool.create_sequence(16).unwrap();
    pool.append(sequence, 12)
        .expect("evicting one anchor frees enough pages");
    assert!(pool.lookup_anchor(&chain(&longer), 4).is_some());
    assert!(pool.lookup_anchor(&chains[1], 4).is_none());
    assert_eq!(pool.metrics().snapshot().prefix_cache_evictions, 1);
    pool.audit().unwrap();
}

#[test]
fn a_capture_without_room_is_skipped_and_pins_survive_eviction() {
    let geometry = KvGeometry::new(4, 4).unwrap();
    let mut pool = KvPool::new(geometry, 4).unwrap();
    let tokens = chain(&prompt(8));
    let source = pool.create_sequence(8).unwrap();
    pool.append(source, 4).unwrap();
    let state = capture(&mut pool, &tokens, source, 1).expect("one state page fits");
    pool.release(source).unwrap();
    let hit = pool
        .resume_prefix(&tokens, 7, 8)
        .unwrap()
        .expect("the anchor hits");
    // Evicting the anchor leaves the pinned snapshot with the sequence.
    assert!(pool.remove_anchor(tokens.key_at(4).unwrap()).unwrap());
    assert_eq!(pool.block_ref_count(state[0]).unwrap(), 1);
    pool.append(hit.handle(), 4).unwrap();
    assert_eq!(
        capture(&mut pool, &tokens, hit.handle(), 2),
        None,
        "two state pages do not fit beside a live sequence"
    );
    pool.release(hit.handle()).unwrap();
    assert_eq!(pool.stats().free_blocks(), 4);
    pool.audit().unwrap();
}

/// An anchor over a strict prefix of a sequence's committed tokens holds only
/// the pages of that prefix, and a later request hits it there.
#[test]
fn a_prefix_anchor_shares_only_the_pages_of_its_tokens() {
    let tokens = prompt(11);
    let chain = chain(&tokens);
    let mut pool = KvPool::new(KvGeometry::new(4, 16).unwrap(), 2).unwrap();
    let source = pool.create_sequence(16).unwrap();
    pool.append(source, 11).unwrap();
    let key = chain.key_at(8).unwrap();
    let digest = chain.chain(2).unwrap();
    assert_eq!(
        pool.insert_prefix_anchor(key, digest, source, 12, 1),
        Err(KvError::InvariantViolation),
        "an anchor cannot pass the committed tokens"
    );
    let state = pool
        .insert_prefix_anchor(key, digest, source, 8, 1)
        .unwrap()
        .unwrap();
    assert_eq!(state.len(), 1);
    let blocks = pool.snapshot(source).unwrap().blocks().to_vec();
    assert_eq!(pool.block_ref_count(blocks[0]).unwrap(), 2);
    assert_eq!(pool.block_ref_count(blocks[1]).unwrap(), 2);
    assert_eq!(
        pool.block_ref_count(blocks[2]).unwrap(),
        1,
        "the source's later page stays private"
    );
    assert_eq!(pool.lookup_anchor(&chain, 10), Some((key, 8)));
    pool.audit().unwrap();
    pool.append(source, 1).unwrap();
    assert_eq!(pool.snapshot(source).unwrap().blocks()[2], blocks[2]);
    pool.audit().unwrap();
}

/// A preemption's private anchor restores for its own request alone, and
/// from its prompt's last row on (the earliest MTP drafter frontier) every
/// page the restored request holds is its own once the anchor is dropped:
/// with last-page rewrites its first append after a public anchor at its
/// aligned prompt end copied the shared page. A drafter continuation from a
/// frontier that lags into an earlier page therefore rewrites only the
/// request's pages, while the public anchor keeps its own.
#[test]
fn an_own_restore_holds_every_page_from_its_prompt_end_alone() {
    let tokens = prompt(8);
    let chain = chain(&tokens);
    let mut pool = KvPool::new(KvGeometry::new(4, 32).unwrap(), 4)
        .unwrap()
        .with_last_page_rewrites(true);
    let source = pool.create_sequence(32).unwrap();
    pool.append(source, 8).unwrap();
    capture(&mut pool, &chain, source, 1).unwrap();
    let shared = pool.snapshot(source).unwrap().blocks().to_vec();
    // Seven decode steps; the first copies the prompt's shared last page.
    for _ in 0..7 {
        pool.append(source, 1).unwrap();
    }
    let decoded = pool.snapshot(source).unwrap().blocks().to_vec();
    assert_eq!(decoded[0], shared[0]);
    assert_ne!(decoded[1], shared[1]);
    // Preemption at 15 tokens: a private anchor, and the sequence goes.
    let own = runtime_kv::PrefixCacheKeyV1::new([7; 32]);
    pool.insert_private_anchor(own, source, 1).unwrap().unwrap();
    pool.release(source).unwrap();
    let hit = pool.resume_anchor(own, 32).unwrap();
    pool.finish_owned_restore(hit.handle()).unwrap();
    assert!(pool.remove_anchor(own).unwrap());
    let restored = pool.snapshot(hit.handle()).unwrap().blocks().to_vec();
    assert_eq!(restored, decoded);
    // Row 7 (the frontier after the prompt) and every later row.
    for block in &restored[7 / 4..] {
        assert_eq!(pool.block_ref_count(*block).unwrap(), 1, "{block:?}");
    }
    assert_eq!(
        pool.block_ref_count(shared[1]).unwrap(),
        1,
        "the anchor's page"
    );
    assert_eq!(pool.block_ref_count(shared[0]).unwrap(), 2);
    pool.audit().unwrap();
}

#[test]
fn idle_frontier_survives_busy_ancestors_under_pressure() {
    let mut pool = KvPool::new(KvGeometry::new(4, 12).unwrap(), 4)
        .unwrap()
        .with_prefix_ancestry(true);
    let idle = chain(&[900; 4]);
    let source = pool.create_sequence(4).unwrap();
    pool.append(source, 4).unwrap();
    capture(&mut pool, &idle, source, 2).unwrap();
    pool.release(source).unwrap();
    let busy = chain(&prompt(12));
    let source = pool.create_sequence(12).unwrap();
    for _ in 0..3 {
        pool.append(source, 4).unwrap();
        capture(&mut pool, &busy, source, 2).unwrap();
    }
    pool.release(source).unwrap();
    assert_eq!(pool.metrics().snapshot().prefix_cache_evictions, 0);
    let pressure = pool.create_sequence(4).unwrap();
    pool.append(pressure, 4).unwrap();
    assert!(pool.has_anchor(idle.key_at(4).unwrap()));
    pool.audit().unwrap();
}

fn content_pool(pages: usize) -> KvPool {
    KvPool::new(KvGeometry::new(4, pages).unwrap(), 8)
        .unwrap()
        .with_prefix_ancestry(true)
}

fn detached_capture(pool: &mut KvPool, chain: &PrefixChainV1, depth: usize, pages: usize) {
    let source = pool.create_sequence(depth).unwrap();
    pool.append(source, depth).unwrap();
    pool.insert_content_anchor(chain, source, depth, pages)
        .unwrap();
    pool.release(source).unwrap();
    pool.audit().unwrap();
}

fn assert_classes(pool: &KvPool, entries: &[(PrefixChainV1, usize)]) {
    use runtime_kv::PrefixAnchorClassV1::{Maximal, Retained, Superseded};
    let live: Vec<_> = entries
        .iter()
        .filter(|(c, n)| pool.has_anchor(c.key_at(*n).unwrap()))
        .collect();
    let ancestor = |a: &(PrefixChainV1, usize), b: &(PrefixChainV1, usize)| {
        a.1 < b.1 && a.0.key_at(a.1) == b.0.key_at(a.1)
    };
    let maximal: Vec<_> = live
        .iter()
        .copied()
        .filter(|a| !live.iter().any(|b| ancestor(a, b)))
        .collect();
    let mut retained = std::collections::BTreeSet::new();
    for frontier in &maximal {
        for divisor in [2, 4, 8] {
            if let Some(a) = live
                .iter()
                .filter(|a| ancestor(a, frontier) && a.1 <= frontier.1 / divisor)
                .max_by_key(|a| a.1)
            {
                retained.insert(a.0.key_at(a.1).unwrap());
            }
        }
    }
    for (i, a) in maximal.iter().enumerate() {
        for b in maximal.iter().skip(i + 1) {
            if let Some(common) = live
                .iter()
                .filter(|c| ancestor(c, a) && ancestor(c, b))
                .max_by_key(|c| c.1)
            {
                retained.insert(common.0.key_at(common.1).unwrap());
            }
        }
    }
    let mut counts = [0; 3];
    for entry in live {
        let key = entry.0.key_at(entry.1).unwrap();
        let expected = if maximal.contains(&entry) {
            Maximal
        } else if retained.contains(&key) {
            Retained
        } else {
            Superseded
        };
        assert_eq!(pool.anchor_class(key), Some(expected), "depth {}", entry.1);
        counts[match expected {
            Superseded => 0,
            Retained => 1,
            Maximal => 2,
        }] += 1;
    }
    assert_eq!(pool.metrics().snapshot().prefix_anchors_by_class, counts);
    pool.audit().unwrap();
}

#[test]
fn forks_rewinds_and_failed_descendants_reclassify_without_pressure() {
    use runtime_kv::PrefixAnchorClassV1::{Maximal, Retained};
    let mut pool = content_pool(512);
    let main = chain(&prompt(64));
    let mut entries = Vec::new();
    for depth in [4, 8, 16, 24, 32, 48, 64] {
        detached_capture(&mut pool, &main, depth, 2);
        entries.push((main.clone(), depth));
        assert_classes(&pool, &entries);
    }
    assert_eq!(pool.metrics().snapshot().prefix_cache_evictions, 0);
    assert_eq!(pool.anchor_count(), 7, "supersession alone never evicts");
    let hit = pool.resume_prefix(&main, 33, 64).unwrap().unwrap();
    assert_eq!(hit.tokens(), 32, "rewind hits a retained ancestor");
    pool.finish_restore(hit.handle()).unwrap();
    pool.release(hit.handle()).unwrap();
    let mut branch = prompt(40);
    branch[25] += 900;
    let branch = chain(&branch);
    detached_capture(&mut pool, &branch, 40, 2);
    entries.push((branch.clone(), 40));
    assert_eq!(pool.anchor_class(main.key_at(24).unwrap()), Some(Retained));
    assert_classes(&pool, &entries);
    // Abandoning a capture uses the same removal path as explicit deletion.
    pool.remove_anchor(branch.key_at(40).unwrap()).unwrap();
    assert_classes(&pool, &entries);
    for depth in [64, 48, 32, 24, 16, 8] {
        pool.remove_anchor(main.key_at(depth).unwrap()).unwrap();
        assert_classes(&pool, &entries);
    }
    assert_eq!(pool.anchor_class(main.key_at(4).unwrap()), Some(Maximal));
    pool.remove_anchor(main.key_at(4).unwrap()).unwrap();
    assert_eq!(pool.stats().free_blocks(), 512);
    pool.audit().unwrap();
}

#[test]
fn content_ancestry_ignores_cow_ids_and_discriminates_tails_salts_and_media() {
    use runtime_kv::PrefixAnchorClassV1::{Maximal, Retained};
    let mut pool = content_pool(128);
    let base = chain(&prompt(16));
    let source = pool.create_sequence(16).unwrap();
    pool.append(source, 5).unwrap();
    capture(&mut pool, &base, source, 1).unwrap();
    let before = pool.snapshot(source).unwrap();
    let append = pool.append(source, 7).unwrap();
    assert!(append.copy().is_some());
    assert_ne!(
        before.blocks()[1],
        pool.snapshot(source).unwrap().blocks()[1]
    );
    capture(&mut pool, &base, source, 1).unwrap();
    pool.release(source).unwrap();
    assert_eq!(pool.anchor_class(base.key_at(5).unwrap()), Some(Retained));
    let mut divergent = prompt(9);
    divergent[4] += 50;
    let divergent = chain(&divergent);
    detached_capture(&mut pool, &divergent, 9, 1);
    assert_eq!(base.chain(1), divergent.chain(1));
    assert_eq!(
        pool.anchor_class(divergent.key_at(9).unwrap()),
        Some(Maximal)
    );
    let salted = PrefixChainV1::salted(4, prompt(16).into(), vec![], Some(b"other")).unwrap();
    detached_capture(&mut pool, &salted, 16, 1);
    let media = PrefixChainV1::new(
        4,
        prompt(16).into(),
        vec![PrefixMediaSpanV1 {
            identity: [7; 32],
            start: 4,
            rows: 2,
            media_row: 0,
        }],
    )
    .unwrap();
    detached_capture(&mut pool, &media, 16, 1);
    let entries = [
        (base.clone(), 5),
        (base, 12),
        (divergent, 9),
        (salted, 16),
        (media, 16),
    ];
    assert_classes(&pool, &entries);
}

#[test]
fn incoming_ancestor_tie_break_and_failed_capture_do_not_pin_old_anchors() {
    for enabled in [false, true] {
        let mut pool = KvPool::new(KvGeometry::new(4, 6).unwrap(), 4)
            .unwrap()
            .with_prefix_ancestry(enabled);
        let idle = chain(&[900; 4]);
        detached_capture(&mut pool, &idle, 4, 1);
        let busy = chain(&prompt(8));
        let source = pool.create_sequence(8).unwrap();
        pool.append(source, 4).unwrap();
        capture(&mut pool, &busy, source, 1).unwrap();
        pool.append(source, 4).unwrap();
        capture(&mut pool, &busy, source, 2).unwrap();
        assert_eq!(pool.has_anchor(idle.key_at(4).unwrap()), enabled);
        if enabled {
            assert!(!pool.has_anchor(busy.key_at(4).unwrap()));
        }
        pool.remove_anchor(busy.key_at(8).unwrap()).unwrap();
        pool.metrics().prefix_capture_failed();
        // A failed replacement does not resurrect its evicted ancestor.
        if enabled {
            assert!(!pool.has_anchor(busy.key_at(4).unwrap()));
        }
        pool.release(source).unwrap();
        pool.audit().unwrap();
    }
}

#[test]
fn private_restore_and_transfer_pins_survive_candidate_eviction() {
    use runtime_kv::{PrefixAnchorClassV1::Maximal, PrefixCacheKeyV1};
    let mut pool = content_pool(6);
    let public = chain(&prompt(8));
    let source = pool.create_sequence(8).unwrap();
    pool.append(source, 4).unwrap();
    let state = capture(&mut pool, &public, source, 1).unwrap();
    let private = PrefixCacheKeyV1::new([91; 32]);
    pool.insert_private_anchor(private, source, 1)
        .unwrap()
        .unwrap();
    assert_eq!(pool.anchor_class(private), Some(Maximal));
    assert_eq!(pool.metrics().snapshot().prefix_private_anchors, 1);
    assert_eq!(pool.metrics().snapshot().prefix_anchors_by_class, [0, 0, 1]);
    pool.release(source).unwrap();
    let hit = pool.resume_anchor(public.key_at(4).unwrap(), 8).unwrap();
    let lease = pool
        .retain_transfer_pages(&[pool.page_allocation(state[0]).unwrap()])
        .unwrap();
    let pressure = pool.create_sequence(20).unwrap();
    pool.append(pressure, 20)
        .expect_err("pinned pages prevent reclaiming enough space");
    assert_eq!(pool.anchor_count(), 0);
    assert_eq!(pool.block_ref_count(state[0]).unwrap(), 2);
    pool.audit().unwrap();
    pool.finish_restore(hit.handle()).unwrap();
    pool.release(hit.handle()).unwrap();
    pool.release(pressure).unwrap();
    assert_eq!(pool.block_ref_count(state[0]).unwrap(), 1);
    pool.release_transfer_pages(&lease).unwrap();
    assert_eq!(pool.stats().free_blocks(), 6);
    let metrics = pool.metrics().snapshot();
    assert_eq!(
        metrics.prefix_cache_evictions,
        metrics.prefix_evictions_by_class.iter().sum()
    );
    assert_eq!(metrics.prefix_private_anchors, 0);
    assert_eq!(metrics.prefix_pages_freed_by_class, [0, 0, 0]);
    pool.audit().unwrap();
}

#[test]
fn random_capture_remove_touch_and_pressure_match_slow_content_oracle() {
    let mut pool = content_pool(96);
    let mut entries = Vec::new();
    let mut random = 0x12345678u64;
    for iteration in 0..1200 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let depth = 1 + (random as usize % 48);
        let branch = (random >> 16) as u32 % 5;
        let mut tokens = prompt(48);
        tokens[12] += branch;
        tokens[27] += (random >> 32) as u32 % 3;
        let identity =
            PrefixChainV1::salted(4, tokens.into(), vec![], Some(&[(iteration % 3) as u8]))
                .unwrap();
        match random % 5 {
            0 if !entries.is_empty() => {
                let (c, n): &(PrefixChainV1, usize) = &entries[random as usize % entries.len()];
                pool.remove_anchor(c.key_at(*n).unwrap()).unwrap();
            }
            1 => {
                if let Some(hit) = pool.resume_prefix(&identity, depth, 48).unwrap() {
                    pool.finish_restore(hit.handle()).unwrap();
                    pool.release(hit.handle()).unwrap();
                }
            }
            _ => {
                detached_capture(&mut pool, &identity, depth, 2);
                let key = identity.key_at(depth).unwrap();
                if !entries.iter().any(|(c, n)| c.key_at(*n) == Some(key)) {
                    entries.push((identity, depth));
                }
            }
        }
        assert_classes(&pool, &entries);
        let clone = pool.clone();
        assert_eq!(clone, pool);
        assert_eq!(clone.metrics().snapshot(), pool.metrics().snapshot());
        clone.audit().unwrap();
    }
    for (chain, depth) in entries {
        pool.remove_anchor(chain.key_at(depth).unwrap()).unwrap();
    }
    assert_eq!(pool.stats().free_blocks(), 96);
    pool.audit().unwrap();
}
