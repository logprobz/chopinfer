use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use runtime_kv::{
    KvError, KvGeometry, KvPool, PrefixAnchorClassV1, PrefixChainV1, PrefixEvictionTierV1,
    PrefixMediaSpanV1,
};

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
    pool.insert_anchor(
        tokens.key_at(boundary).unwrap(),
        tokens.chain(boundary / tokens.block_tokens()).unwrap(),
        source,
        state_pages,
    )
    .expect("a live source can be anchored")
}

fn content_capture(
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

/// One conversation turn on `history`: resume the frontier, prefill, and
/// capture the new prompt end with `pages` snapshot pages.
fn conversation_turn(pool: &mut KvPool, history: &[u32], pages: usize) {
    let identity = chain(history);
    let tokens = history.len();
    let source = match pool.resume_prefix(&identity, tokens - 1, tokens).unwrap() {
        Some(hit) => {
            pool.finish_restore(hit.handle()).unwrap();
            hit.handle()
        }
        None => pool.create_sequence(tokens).unwrap(),
    };
    let committed = pool.snapshot(source).unwrap().tokens();
    pool.append(source, tokens - committed).unwrap();
    pool.insert_content_anchor(&identity, source, tokens, pages)
        .unwrap()
        .unwrap();
    pool.release(source).unwrap();
}

/// A busy conversation resumes its frontier every turn, so ancestry
/// eviction takes an ancestor it moved past before an idle frontier. LRU
/// evicts the idle frontier, the oldest anchor.
#[test]
fn idle_frontier_survives_busy_ancestors_under_pressure() {
    for ancestry in [false, true] {
        let mut pool = KvPool::new(KvGeometry::new(4, 14).unwrap(), 4)
            .unwrap()
            .with_prefix_ancestry(ancestry);
        let idle = chain(&[900; 4]);
        let source = pool.create_sequence(4).unwrap();
        pool.append(source, 4).unwrap();
        content_capture(&mut pool, &idle, source, 2).unwrap();
        pool.release(source).unwrap();
        let busy = prompt(16);
        for depth in [4, 8, 12] {
            conversation_turn(&mut pool, &busy[..depth], 2);
        }
        assert_eq!(pool.metrics().snapshot().prefix_cache_evictions, 0);
        conversation_turn(&mut pool, &busy, 2);
        assert_eq!(pool.metrics().snapshot().prefix_cache_evictions, 1);
        assert_eq!(pool.has_anchor(idle.key_at(4).unwrap()), ancestry);
        assert_eq!(
            pool.has_anchor(chain(&busy).key_at(8).unwrap()),
            !ancestry,
            "the busy conversation moved past its anchor at 8"
        );
        pool.audit().unwrap();
    }
}

fn content_pool(pages: usize) -> KvPool {
    KvPool::new(KvGeometry::new(4, pages).unwrap(), 8)
        .unwrap()
        .with_prefix_ancestry(true)
}

type Entry = (PrefixChainV1, usize);
type Key = runtime_kv::PrefixCacheKeyV1;

fn entry_key(entry: &Entry) -> Key {
    entry.0.key_at(entry.1).unwrap()
}

/// Whether `a` is a strict content ancestor of `b`.
fn above(a: &Entry, b: &Entry) -> bool {
    a.1 < b.1 && a.0.key_at(a.1) == b.0.key_at(a.1)
}

#[derive(Clone, Copy, Debug, Default)]
struct Flags {
    extended: bool,
    consumed: bool,
    shared: bool,
}

/// Deliberately slow model of ancestry eviction. It recomputes the content
/// structure from scratch after every event and replays the anchors'
/// lifetime flags from capture, successful public restore and removal
/// events alone, independently of the radix index and its ordered map.
#[derive(Clone, Default)]
struct Model {
    entries: Vec<Entry>,
    flags: BTreeMap<Key, Flags>,
    last_used: BTreeMap<Key, u64>,
    clock: u64,
}

impl Model {
    fn live(&self) -> Vec<&Entry> {
        self.entries
            .iter()
            .filter(|entry| self.flags.contains_key(&entry_key(entry)))
            .collect()
    }

    fn entry(&self, key: Key) -> &Entry {
        self.entries
            .iter()
            .find(|entry| entry_key(entry) == key)
            .unwrap()
    }

    /// Resident anchors below `a` with no resident anchor between.
    fn nearest_descendants(live: &[&Entry], a: &Entry) -> usize {
        live.iter()
            .filter(|d| above(a, d) && !live.iter().any(|c| above(a, c) && above(c, d)))
            .count()
    }

    fn has_descendant(&self, key: Key) -> bool {
        let a = self.entry(key);
        self.live().iter().any(|d| above(a, d))
    }

    /// A fork after any event stays shared for its anchor's lifetime.
    fn mark_forks(&mut self) {
        let live = self.live();
        let forks: Vec<Key> = live
            .iter()
            .filter(|a| Self::nearest_descendants(&live, a) >= 2)
            .map(|a| entry_key(a))
            .collect();
        for key in forks {
            self.flags.get_mut(&key).unwrap().shared = true;
        }
    }

    fn touch(&mut self, key: Key) {
        self.clock += 1;
        self.last_used.insert(key, self.clock);
    }

    /// A capture of a new key.
    fn insert(&mut self, entry: Entry) {
        let key = entry_key(&entry);
        let consumed = self.live().iter().any(|d| {
            above(&entry, d) && {
                let flags = self.flags[&entry_key(d)];
                flags.consumed || flags.extended
            }
        });
        if !self.entries.iter().any(|known| entry_key(known) == key) {
            self.entries.push(entry);
        }
        self.flags.insert(
            key,
            Flags {
                consumed,
                ..Flags::default()
            },
        );
        self.touch(key);
        self.mark_forks();
    }

    /// A successful public restore, classified when it resumed.
    fn restore(&mut self, key: Key, branch: bool) {
        if branch {
            self.flags.get_mut(&key).unwrap().shared = true;
        } else {
            self.flags.get_mut(&key).unwrap().extended = true;
            let restored = self.entry(key).clone();
            let ancestors: Vec<Key> = self
                .live()
                .iter()
                .filter(|a| above(a, &restored))
                .map(|a| entry_key(a))
                .collect();
            for ancestor in ancestors {
                self.flags.get_mut(&ancestor).unwrap().consumed = true;
            }
        }
        self.mark_forks();
    }

    fn remove(&mut self, key: Key) {
        self.flags.remove(&key);
        self.last_used.remove(&key);
        self.mark_forks();
    }

    fn ranks(&self) -> BTreeMap<Key, (PrefixAnchorClassV1, PrefixEvictionTierV1)> {
        let live = self.live();
        let maximal = |a: &Entry| !live.iter().any(|d| above(a, d));
        let mut nominated = BTreeSet::new();
        for frontier in live.iter().filter(|a| maximal(a)) {
            for divisor in [2, 4, 8] {
                if let Some(a) = live
                    .iter()
                    .filter(|a| above(a, frontier) && a.1 <= frontier.1 / divisor)
                    .max_by_key(|a| a.1)
                {
                    nominated.insert(entry_key(a));
                }
            }
        }
        live.iter()
            .map(|a| {
                let key = entry_key(a);
                let flags = self.flags[&key];
                let fork = Self::nearest_descendants(&live, a) >= 2;
                let rank = if maximal(a) {
                    (PrefixAnchorClassV1::Maximal, PrefixEvictionTierV1::Live)
                } else if fork || flags.shared || !flags.consumed {
                    (PrefixAnchorClassV1::Retained, PrefixEvictionTierV1::Live)
                } else if nominated.contains(&key) {
                    (PrefixAnchorClassV1::Retained, PrefixEvictionTierV1::Spaced)
                } else {
                    (
                        PrefixAnchorClassV1::Superseded,
                        PrefixEvictionTierV1::Superseded,
                    )
                };
                (key, rank)
            })
            .collect()
    }

    /// The victims, in order, of an allocation of `needed` free pages,
    /// counting physical references independently of the pool's map.
    fn expected_evictions(&self, pool: &KvPool, needed: usize, replaced: Option<Key>) -> Vec<Key> {
        let mut model = self.clone();
        let mut refs = BTreeMap::new();
        let mut pages = BTreeMap::new();
        for entry in model.live() {
            let key = entry_key(entry);
            let (_, mut kv, state) = pool.anchor_pages(key).unwrap();
            kv.extend(state);
            for page in &kv {
                refs.insert(*page, pool.block_ref_count(*page).unwrap());
            }
            pages.insert(key, kv);
        }
        let mut free = pool.stats().free_blocks();
        let mut replaced = replaced;
        let mut victims = Vec::new();
        while free < needed && !model.flags.is_empty() {
            let ranks = model.ranks();
            let lowest = ranks.values().map(|(_, tier)| *tier).min().unwrap();
            let victim = match replaced.take() {
                Some(key) if lowest == PrefixEvictionTierV1::Live && ranks.contains_key(&key) => {
                    key
                }
                other => {
                    replaced = other;
                    *ranks
                        .iter()
                        .filter(|(_, (_, tier))| *tier == lowest)
                        .map(|(key, _)| key)
                        .min_by_key(|key| (model.last_used[key], **key))
                        .unwrap()
                }
            };
            for page in &pages[&victim] {
                let refs = refs.get_mut(page).unwrap();
                *refs -= 1;
                free += usize::from(*refs == 0);
            }
            model.remove(victim);
            victims.push(victim);
        }
        victims
    }

    fn check(&self, pool: &KvPool) {
        let live: BTreeSet<Key> = self.flags.keys().copied().collect();
        for entry in &self.entries {
            let key = entry_key(entry);
            assert_eq!(pool.has_anchor(key), live.contains(&key));
        }
        let mut counts = [0; 3];
        for (key, (class, tier)) in self.ranks() {
            assert_eq!(pool.anchor_class(key), Some(class), "class of {key:?}");
            assert_eq!(pool.anchor_tier(key), Some(tier), "tier of {key:?}");
            counts[class as usize] += 1;
        }
        assert_eq!(pool.metrics().snapshot().prefix_anchors_by_class, counts);
        pool.audit().unwrap();
    }
}

/// A capture from a fresh sequence over `depth` tokens of `chain`.
fn detached_capture(
    pool: &mut KvPool,
    model: &mut Model,
    chain: &PrefixChainV1,
    depth: usize,
    pages: usize,
) {
    let source = pool.create_sequence(depth).unwrap();
    pool.append(source, depth).unwrap();
    let new = !pool.has_anchor(chain.key_at(depth).unwrap());
    let state = pool
        .insert_content_anchor(chain, source, depth, pages)
        .unwrap();
    pool.release(source).unwrap();
    if new {
        assert!(state.is_some());
        model.insert((chain.clone(), depth));
    } else {
        model.touch(chain.key_at(depth).unwrap());
    }
    model.check(pool);
}

/// Resumes `limit` tokens of `chain` and finishes the restore successfully.
fn restore(pool: &mut KvPool, model: &mut Model, chain: &PrefixChainV1, limit: usize) -> usize {
    let hit = pool.resume_prefix(chain, limit, 64).unwrap().unwrap();
    let key = chain.key_at(hit.tokens()).unwrap();
    let branch = model.has_descendant(key);
    model.touch(key);
    pool.finish_restore(hit.handle()).unwrap();
    model.restore(key, branch);
    pool.release(hit.handle()).unwrap();
    model.check(pool);
    hit.tokens()
}

#[test]
fn forks_rewinds_and_failed_descendants_reclassify_without_pressure() {
    use runtime_kv::PrefixAnchorClassV1::{Maximal, Retained};
    let mut pool = content_pool(512);
    let mut model = Model::default();
    let main = chain(&prompt(64));
    for depth in [4, 8, 16, 24, 32, 48, 64] {
        detached_capture(&mut pool, &mut model, &main, depth, 2);
    }
    assert_eq!(pool.metrics().snapshot().prefix_cache_evictions, 0);
    assert_eq!(pool.anchor_count(), 7, "supersession alone never evicts");
    assert_eq!(
        restore(&mut pool, &mut model, &main, 33),
        32,
        "rewind hits an ancestor"
    );
    assert_eq!(
        pool.anchor_tier(main.key_at(32).unwrap()),
        Some(PrefixEvictionTierV1::Live),
        "a hit its descendants could not serve shares it"
    );
    let mut branch = prompt(40);
    branch[25] += 900;
    let branch = chain(&branch);
    detached_capture(&mut pool, &mut model, &branch, 40, 2);
    assert_eq!(pool.anchor_class(main.key_at(24).unwrap()), Some(Retained));
    // Abandoning a capture uses the same removal path as explicit deletion.
    pool.remove_anchor(branch.key_at(40).unwrap()).unwrap();
    model.remove(branch.key_at(40).unwrap());
    model.check(&pool);
    for depth in [64, 48, 32, 24, 16, 8] {
        pool.remove_anchor(main.key_at(depth).unwrap()).unwrap();
        model.remove(main.key_at(depth).unwrap());
        model.check(&pool);
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
    let mut model = Model::default();
    let base = chain(&prompt(16));
    let source = pool.create_sequence(16).unwrap();
    pool.append(source, 5).unwrap();
    content_capture(&mut pool, &base, source, 1).unwrap();
    model.insert((base.clone(), 5));
    let before = pool.snapshot(source).unwrap();
    let append = pool.append(source, 7).unwrap();
    assert!(append.copy().is_some());
    assert_ne!(
        before.blocks()[1],
        pool.snapshot(source).unwrap().blocks()[1]
    );
    content_capture(&mut pool, &base, source, 1).unwrap();
    model.insert((base.clone(), 12));
    pool.release(source).unwrap();
    assert_eq!(pool.anchor_class(base.key_at(5).unwrap()), Some(Retained));
    let mut divergent = prompt(9);
    divergent[4] += 50;
    let divergent = chain(&divergent);
    detached_capture(&mut pool, &mut model, &divergent, 9, 1);
    assert_eq!(base.chain(1), divergent.chain(1));
    assert_eq!(
        pool.anchor_class(divergent.key_at(9).unwrap()),
        Some(Maximal)
    );
    let salted = PrefixChainV1::salted(4, prompt(16).into(), vec![], Some(b"other")).unwrap();
    detached_capture(&mut pool, &mut model, &salted, 16, 1);
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
    detached_capture(&mut pool, &mut model, &media, 16, 1);
    model.check(&pool);
}

/// When only live anchors remain, a turn's capture replaces the frontier its
/// own sequence extended, not an unrelated idle frontier. A failed capture
/// removes only its new anchor: the replaced frontier stays evicted.
#[test]
fn capture_replaces_its_extended_frontier_and_failure_does_not_restore_it() {
    for ancestry in [false, true] {
        let mut pool = KvPool::new(KvGeometry::new(4, 6).unwrap(), 4)
            .unwrap()
            .with_prefix_ancestry(ancestry);
        let mut model = Model::default();
        let idle = chain(&[900; 4]);
        detached_capture(&mut pool, &mut model, &idle, 4, 1);
        let busy = chain(&prompt(8));
        detached_capture(&mut pool, &mut model, &busy, 4, 1);
        let hit = pool.resume_prefix(&busy, 7, 8).unwrap().unwrap();
        assert_eq!(hit.tokens(), 4);
        pool.finish_restore(hit.handle()).unwrap();
        pool.append(hit.handle(), 4).unwrap();
        assert_eq!(pool.stats().free_blocks(), 1);
        content_capture(&mut pool, &busy, hit.handle(), 2).unwrap();
        assert_eq!(pool.has_anchor(idle.key_at(4).unwrap()), ancestry);
        assert_eq!(pool.has_anchor(busy.key_at(4).unwrap()), !ancestry);
        pool.remove_anchor(busy.key_at(8).unwrap()).unwrap();
        pool.metrics().prefix_capture_failed();
        assert_eq!(pool.has_anchor(busy.key_at(4).unwrap()), !ancestry);
        pool.release(hit.handle()).unwrap();
        pool.audit().unwrap();
    }
}

/// Only a successful public restore marks ancestry: a failed restore and an
/// own-boundary restore leave the passed ancestor live.
#[test]
fn failed_and_own_boundary_restores_mark_no_ancestry() {
    let mut pool = content_pool(32);
    let mut model = Model::default();
    let tokens = chain(&prompt(16));
    detached_capture(&mut pool, &mut model, &tokens, 8, 1);
    detached_capture(&mut pool, &mut model, &tokens, 12, 1);
    let passed = tokens.key_at(8).unwrap();
    let frontier = tokens.key_at(12).unwrap();
    assert_eq!(pool.anchor_tier(passed), Some(PrefixEvictionTierV1::Live));
    let failed = pool.resume_prefix(&tokens, 15, 16).unwrap().unwrap();
    assert_eq!(failed.tokens(), 12);
    pool.release(failed.handle()).unwrap();
    model.touch(frontier);
    model.check(&pool);
    let own = pool.resume_anchor(frontier, 16).unwrap();
    pool.finish_owned_restore(own.handle()).unwrap();
    pool.release(own.handle()).unwrap();
    model.touch(frontier);
    model.check(&pool);
    assert_eq!(pool.anchor_tier(passed), Some(PrefixEvictionTierV1::Live));
    // The successful extension of the frontier moves past the anchor at 8.
    assert_eq!(restore(&mut pool, &mut model, &tokens, 15), 12);
    assert_eq!(
        pool.anchor_tier(passed),
        Some(PrefixEvictionTierV1::Superseded)
    );
}

/// A restore mark belongs to the anchor lifetime it resumed: evicting and
/// recapturing the key before the restore finishes marks nothing.
#[test]
fn a_restore_mark_dies_with_its_anchor() {
    let mut pool = content_pool(32);
    let mut model = Model::default();
    let tokens = chain(&prompt(12));
    detached_capture(&mut pool, &mut model, &tokens, 4, 1);
    detached_capture(&mut pool, &mut model, &tokens, 8, 1);
    let hit = pool.resume_prefix(&tokens, 11, 12).unwrap().unwrap();
    model.touch(tokens.key_at(8).unwrap());
    pool.remove_anchor(tokens.key_at(8).unwrap()).unwrap();
    model.remove(tokens.key_at(8).unwrap());
    detached_capture(&mut pool, &mut model, &tokens, 8, 1);
    pool.finish_restore(hit.handle()).unwrap();
    pool.release(hit.handle()).unwrap();
    model.check(&pool);
    assert_eq!(
        pool.anchor_tier(tokens.key_at(4).unwrap()),
        Some(PrefixEvictionTierV1::Live)
    );
}

/// A consumed spaced ancestor that a branch restore shares keeps its
/// Retained class but moves to the live tier, so a plain spaced ancestor
/// goes first under pressure.
#[test]
fn a_branch_restore_moves_a_spaced_ancestor_to_the_live_tier() {
    let mut pool = content_pool(40);
    let mut model = Model::default();
    let history = prompt(32);
    let mut previous = 0;
    for depth in [4, 8, 16, 32] {
        let identity = chain(&history[..depth]);
        let source = if previous == 0 {
            pool.create_sequence(depth).unwrap()
        } else {
            let hit = pool
                .resume_prefix(&identity, depth - 1, depth)
                .unwrap()
                .unwrap();
            assert_eq!(hit.tokens(), previous);
            let key = identity.key_at(previous).unwrap();
            let branch = model.has_descendant(key);
            model.touch(key);
            pool.finish_restore(hit.handle()).unwrap();
            model.restore(key, branch);
            hit.handle()
        };
        let committed = pool.snapshot(source).unwrap().tokens();
        pool.append(source, depth - committed).unwrap();
        content_capture(&mut pool, &identity, source, 1).unwrap();
        model.insert((identity, depth));
        pool.release(source).unwrap();
        model.check(&pool);
        previous = depth;
    }
    let full = chain(&history);
    for depth in [4, 8] {
        let key = full.key_at(depth).unwrap();
        assert_eq!(pool.anchor_tier(key), Some(PrefixEvictionTierV1::Spaced));
        assert_eq!(pool.anchor_class(key), Some(PrefixAnchorClassV1::Retained));
    }
    let mut diverging = prompt(12);
    diverging[10] += 500;
    assert_eq!(restore(&mut pool, &mut model, &chain(&diverging), 11), 8);
    let shared = full.key_at(8).unwrap();
    assert_eq!(pool.anchor_tier(shared), Some(PrefixEvictionTierV1::Live));
    assert_eq!(
        pool.anchor_class(shared),
        Some(PrefixAnchorClassV1::Retained)
    );
    let free = pool.stats().free_blocks();
    let expected = model.expected_evictions(&pool, free + 1, None);
    assert_eq!(expected, [full.key_at(4).unwrap()]);
    let pressure = pool.create_sequence(4 * (free + 1)).unwrap();
    pool.append(pressure, 4 * (free + 1)).unwrap();
    for victim in expected {
        model.remove(victim);
    }
    model.check(&pool);
    pool.release(pressure).unwrap();
}

/// Private and digest-only anchors stay maximal and live: their ancestry
/// cannot be reconstructed. A duplicate capture only refreshes recency.
#[test]
fn private_digest_only_and_duplicate_captures_keep_their_tier() {
    let mut pool = content_pool(32);
    let tokens = chain(&prompt(8));
    let source = pool.create_sequence(8).unwrap();
    pool.append(source, 8).unwrap();
    pool.insert_content_anchor(&tokens, source, 4, 1)
        .unwrap()
        .unwrap();
    pool.insert_anchor(
        tokens.key_at(8).unwrap(),
        tokens.chain(2).unwrap(),
        source,
        1,
    )
    .unwrap()
    .unwrap();
    let private = runtime_kv::PrefixCacheKeyV1::new([91; 32]);
    pool.insert_private_anchor(private, source, 1)
        .unwrap()
        .unwrap();
    for key in [tokens.key_at(8).unwrap(), private] {
        assert_eq!(pool.anchor_tier(key), Some(PrefixEvictionTierV1::Live));
        assert_eq!(pool.anchor_class(key), Some(PrefixAnchorClassV1::Maximal));
    }
    let before = pool.anchor_tier(tokens.key_at(4).unwrap());
    assert_eq!(pool.insert_content_anchor(&tokens, source, 4, 1), Ok(None));
    assert_eq!(pool.anchor_tier(tokens.key_at(4).unwrap()), before);
    pool.release(source).unwrap();
    pool.audit().unwrap();
}

#[test]
fn private_restore_and_transfer_pins_survive_candidate_eviction() {
    use runtime_kv::{PrefixAnchorClassV1::Maximal, PrefixCacheKeyV1};
    let mut pool = content_pool(6);
    let public = chain(&prompt(8));
    let source = pool.create_sequence(8).unwrap();
    pool.append(source, 4).unwrap();
    let state = content_capture(&mut pool, &public, source, 1).unwrap();
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
    // The restore's anchor is gone, so finishing it marks nothing.
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

struct Rng(u64);

impl Rng {
    fn below(&mut self, limit: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % limit
    }
}

fn random_chain(rng: &mut Rng) -> (PrefixChainV1, bool) {
    let mut tokens = prompt(64);
    for _ in 0..rng.below(3) {
        tokens[rng.below(64) as usize] += 1 + rng.below(2) as u32;
    }
    let mut media = Vec::new();
    if rng.below(3) == 0 {
        media.push(PrefixMediaSpanV1 {
            identity: [rng.below(2) as u8; 32],
            start: rng.below(56) as usize,
            rows: 1 + rng.below(6) as usize,
            media_row: rng.below(2) as u32,
        });
    }
    let has_media = !media.is_empty();
    let salt = [rng.below(2) as u8];
    (
        PrefixChainV1::salted(4, tokens.into(), media, (salt[0] == 1).then_some(&salt[..]))
            .unwrap(),
        has_media,
    )
}

/// A capture of `depth` tokens of `chain` from `source` with `state`
/// snapshot pages.
struct Capture<'a> {
    chain: &'a PrefixChainV1,
    source: runtime_kv::SequenceHandle,
    depth: usize,
    state: usize,
}

/// Applies a capture and checks its victims against the model.
fn checked_capture(
    pool: &mut KvPool,
    model: &mut Model,
    capture: &Capture<'_>,
    replaced: Option<Key>,
    seed: u64,
) -> Vec<Key> {
    let Capture {
        chain,
        source,
        depth,
        state,
    } = *capture;
    let key = chain.key_at(depth).unwrap();
    let duplicate = pool.has_anchor(key);
    let expected = if duplicate {
        Vec::new()
    } else {
        model.expected_evictions(pool, state, replaced)
    };
    let captured = pool
        .insert_content_anchor(chain, source, depth, state)
        .unwrap();
    for victim in &expected {
        model.remove(*victim);
    }
    if duplicate {
        model.touch(key);
    } else if captured.is_some() {
        model.insert((chain.clone(), depth));
    }
    model.check(pool);
    assert!(
        model.flags.contains_key(&key) || captured.is_none(),
        "capture of {key:?}, seed {seed}"
    );
    expected
}

#[test]
fn random_capture_restore_remove_and_pressure_match_slow_content_model() {
    let mut counts = BTreeMap::<&str, usize>::new();
    for seed in 1..=40 {
        let mut rng = Rng(0x9e3779b97f4a7c15 ^ seed);
        let capacity = 60 + rng.below(80) as usize;
        let mut pool = content_pool(capacity);
        let mut model = Model::default();
        for _ in 0..400 {
            match rng.below(7) {
                0 | 1 if !model.flags.is_empty() => {
                    let keys: Vec<Key> = model.flags.keys().copied().collect();
                    let key = keys[rng.below(keys.len() as u64) as usize];
                    pool.remove_anchor(key).unwrap();
                    model.remove(key);
                }
                2 | 3 => {
                    let (chain, _) = random_chain(&mut rng);
                    let limit = 1 + rng.below(64) as usize;
                    let Some(hit) = pool.resume_prefix(&chain, limit, 64).unwrap() else {
                        continue;
                    };
                    let key = chain.key_at(hit.tokens()).unwrap();
                    let branch = model.has_descendant(key);
                    model.touch(key);
                    match rng.below(4) {
                        0 => {
                            pool.release(hit.handle()).unwrap();
                            *counts.entry("failed restores").or_default() += 1;
                        }
                        1 => {
                            pool.finish_owned_restore(hit.handle()).unwrap();
                            pool.release(hit.handle()).unwrap();
                            *counts.entry("own restores").or_default() += 1;
                        }
                        _ => {
                            pool.finish_restore(hit.handle()).unwrap();
                            model.restore(key, branch);
                            *counts
                                .entry(if branch {
                                    "branch restores"
                                } else {
                                    "extension restores"
                                })
                                .or_default() += 1;
                            model.check(&pool);
                            // Extend the restored sequence and capture deeper
                            // when the append needs no eviction, so the
                            // capture alone exercises the replaced frontier.
                            let depth =
                                hit.tokens() + 1 + rng.below(65 - hit.tokens() as u64) as usize;
                            let pages = depth.div_ceil(4) - hit.tokens().div_ceil(4) + 1;
                            if depth <= 64
                                && !chain.splits_media(depth)
                                && pool.stats().free_blocks() > pages
                            {
                                pool.append(hit.handle(), depth - hit.tokens()).unwrap();
                                model.check(&pool);
                                let replaced =
                                    (!branch && !model.flags[&key].shared).then_some(key);
                                let state = 1 + rng.below(3) as usize;
                                let capture = Capture {
                                    chain: &chain,
                                    source: hit.handle(),
                                    depth,
                                    state,
                                };
                                let victims = checked_capture(
                                    &mut pool, &mut model, &capture, replaced, seed,
                                );
                                if victims.first() == Some(&key) {
                                    *counts.entry("replaced frontiers").or_default() += 1;
                                }
                            }
                            pool.release(hit.handle()).unwrap();
                        }
                    }
                }
                _ => {
                    let (chain, media) = random_chain(&mut rng);
                    let depth = (1 + rng.below(64) as usize..=64)
                        .find(|depth| !chain.splits_media(*depth))
                        .unwrap();
                    let source = pool.create_sequence(depth).unwrap();
                    let expected = model.expected_evictions(&pool, depth.div_ceil(4), None);
                    pool.append(source, depth).unwrap();
                    for victim in &expected {
                        model.remove(*victim);
                    }
                    model.check(&pool);
                    let ranks = model.ranks();
                    let state = 1 + rng.below(3) as usize;
                    let capture = Capture {
                        chain: &chain,
                        source,
                        depth,
                        state,
                    };
                    for victim in checked_capture(&mut pool, &mut model, &capture, None, seed) {
                        *counts
                            .entry(match ranks[&victim].1 {
                                PrefixEvictionTierV1::Superseded => "superseded victims",
                                PrefixEvictionTierV1::Spaced => "spaced victims",
                                PrefixEvictionTierV1::Live => "live victims",
                            })
                            .or_default() += 1;
                    }
                    *counts.entry("media captures").or_default() += usize::from(media);
                    pool.release(source).unwrap();
                }
            }
            model.check(&pool);
            let clone = pool.clone();
            assert_eq!(clone, pool);
            assert_eq!(clone.metrics().snapshot(), pool.metrics().snapshot());
            clone.audit().unwrap();
        }
        let keys: Vec<Key> = model.flags.keys().copied().collect();
        for key in keys {
            pool.remove_anchor(key).unwrap();
        }
        assert_eq!(pool.stats().free_blocks(), capacity);
        assert_eq!(pool.metrics().snapshot().prefix_anchors_by_class, [0; 3]);
        assert_eq!(
            pool.metrics().snapshot().prefix_snapshot_pages_by_class,
            [0; 3]
        );
        pool.audit().unwrap();
    }
    println!("{counts:?}");
    for kind in [
        "failed restores",
        "own restores",
        "branch restores",
        "extension restores",
        "replaced frontiers",
        "superseded victims",
        "spaced victims",
        "live victims",
        "media captures",
    ] {
        assert!(
            counts.get(kind).is_some_and(|count| *count > 0),
            "no {kind}: {counts:?}"
        );
    }
}

/// Equal tokens with different media identity, span extent or media row
/// branch exactly at the span start, including a span crossing a block
/// boundary, and an interior capture from a longer prompt stops at its own
/// boundary.
#[test]
fn media_only_divergence_splits_at_the_span_start() {
    let mut pool = content_pool(256);
    let mut model = Model::default();
    let tokens = prompt(32);
    let span = |identity: u8, rows: usize, media_row: u32| PrefixMediaSpanV1 {
        identity: [identity; 32],
        start: 9,
        rows,
        media_row,
    };
    let variants: Vec<PrefixChainV1> = [
        vec![span(1, 6, 0)],
        vec![span(2, 6, 0)],
        vec![span(1, 5, 0)],
        vec![span(1, 6, 1)],
        vec![],
    ]
    .into_iter()
    .map(|media| PrefixChainV1::new(4, tokens.clone().into(), media).unwrap())
    .collect();
    for variant in &variants {
        detached_capture(&mut pool, &mut model, variant, 32, 1);
    }
    for (first, second) in [(0, 1), (0, 2), (0, 3), (0, 4)] {
        assert_ne!(variants[first].key_at(32), variants[second].key_at(32));
        assert_eq!(variants[first].key_at(9), variants[second].key_at(9));
    }
    // The boundary at the span start is shared by all five; past the span
    // each variant has its own path.
    detached_capture(&mut pool, &mut model, &variants[0], 9, 1);
    assert_eq!(
        pool.anchor_class(variants[0].key_at(9).unwrap()),
        Some(PrefixAnchorClassV1::Retained)
    );
    detached_capture(&mut pool, &mut model, &variants[1], 16, 1);
    detached_capture(&mut pool, &mut model, &variants[0], 20, 1);
    detached_capture(&mut pool, &mut model, &variants[2], 8, 1);
    for variant in &variants {
        let hit = pool.lookup_anchor(variant, 31).unwrap();
        assert!(hit.1 >= 9, "{hit:?}");
    }
    for (variant, depth) in [(0, 9), (1, 32), (0, 20), (2, 8), (3, 32)] {
        let key = variants[variant].key_at(depth).unwrap();
        pool.remove_anchor(key).unwrap();
        model.remove(key);
        model.check(&pool);
    }
}
