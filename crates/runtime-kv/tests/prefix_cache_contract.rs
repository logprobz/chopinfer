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
    let key = tokens
        .key_at(boundary)
        .expect("the boundary is inside the prompt");
    let digest = tokens
        .chain(boundary / 4)
        .expect("the full blocks are hashed");
    pool.insert_anchor(key, digest, source, state_pages)
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
