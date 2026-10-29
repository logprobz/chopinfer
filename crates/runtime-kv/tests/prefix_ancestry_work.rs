//! The ancestry index's work per capture, restore and eviction at 131,072
//! tokens with 64-token pages: bounded by the anchors and branch points on
//! the path, not by context length. Page references are the pool's own
//! ownership work and are exact.
use std::sync::Arc;

use runtime_kv::{
    KvGeometry, KvPool, PrefixAncestryWorkV1, PrefixChainV1, PrefixMediaSpanV1, SequenceHandle,
};

const BLOCK: usize = 64;
const DEPTH: usize = 131_072;
/// Radix nodes, digests and target entries one operation may touch here.
const NODES: u64 = 48;
const DIGESTS: u64 = 64;
const TARGETS: u64 = 96;
/// Atoms compared one by one: under a block per descent step.
const ATOMS: u64 = 4 * BLOCK as u64;

fn chain(tokens: &[u32]) -> PrefixChainV1 {
    PrefixChainV1::new(BLOCK, Arc::from(tokens), Vec::new()).unwrap()
}

fn session(seed: u32, length: usize) -> Vec<u32> {
    (0..length as u32).map(|t| seed * 1_000_000 + t).collect()
}

fn measured(pool: &mut KvPool, f: impl FnOnce(&mut KvPool)) -> PrefixAncestryWorkV1 {
    let before = pool.prefix_ancestry_work();
    f(pool);
    pool.prefix_ancestry_work().since(before)
}

fn assert_bounded(work: PrefixAncestryWorkV1, copied: u64, what: &str) {
    assert!(work.nodes <= NODES, "{what}: {work:?}");
    assert!(work.digests_compared <= DIGESTS, "{what}: {work:?}");
    assert!(work.atoms_compared <= ATOMS, "{what}: {work:?}");
    assert!(work.target_updates <= TARGETS, "{what}: {work:?}");
    assert!(work.tokens_copied <= copied, "{what}: {work:?}");
}

fn pool() -> KvPool {
    KvPool::new(KvGeometry::new(BLOCK, 40_000).unwrap(), 8)
        .unwrap()
        .with_prefix_ancestry(true)
}

fn capture(pool: &mut KvPool, identity: &PrefixChainV1, source: SequenceHandle, depth: usize) {
    pool.insert_content_anchor(identity, source, depth, 1)
        .unwrap()
        .unwrap();
}

#[test]
fn ascending_captures_and_a_session_turn_do_bounded_index_work() {
    let mut pool = pool();
    let tokens = session(1, DEPTH + BLOCK);
    let identity = chain(&tokens[..DEPTH]);
    let source = pool.create_sequence(DEPTH).unwrap();
    pool.append(source, DEPTH).unwrap();
    let mut previous = 0;
    for depth in [DEPTH / 8, DEPTH / 4, DEPTH / 2, DEPTH - 7] {
        let work = measured(&mut pool, |pool| capture(pool, &identity, source, depth));
        // A capture copies its new tokens once and references its pages.
        assert_bounded(work, (depth - previous) as u64, "ascending capture");
        assert_eq!(work.page_references, depth.div_ceil(BLOCK) as u64 + 1);
        previous = depth;
    }
    pool.release(source).unwrap();
    // A later turn: an extension restore, a prefill and a deeper capture.
    let next = chain(&tokens);
    let work = measured(&mut pool, |pool| {
        let hit = pool
            .resume_prefix(&next, DEPTH, DEPTH + BLOCK)
            .unwrap()
            .unwrap();
        assert_eq!(hit.tokens(), DEPTH - 7);
        pool.finish_restore(hit.handle()).unwrap();
        pool.append(hit.handle(), BLOCK + 7).unwrap();
        capture(pool, &next, hit.handle(), DEPTH + BLOCK);
        pool.release(hit.handle()).unwrap();
    });
    assert_bounded(work, (BLOCK + 7) as u64, "session turn");
    // Evicting the passed ancestors one by one.
    for depth in [DEPTH / 8, DEPTH / 4, DEPTH / 2] {
        let key = identity.key_at(depth).unwrap();
        let work = measured(&mut pool, |pool| assert!(pool.remove_anchor(key).unwrap()));
        assert_bounded(work, 8192, "eviction");
        assert_eq!(work.page_references, depth.div_ceil(BLOCK) as u64 + 1);
    }
    pool.audit().unwrap();
}

#[test]
fn repeated_interior_insert_and_drop_copies_nothing() {
    let mut pool = pool();
    let identity = chain(&session(2, DEPTH));
    let source = pool.create_sequence(DEPTH).unwrap();
    pool.append(source, DEPTH).unwrap();
    capture(&mut pool, &identity, source, DEPTH);
    for round in 0..200usize {
        let depth = 1 + (round * 7_919 + 333) % (DEPTH - 1);
        let work = measured(&mut pool, |pool| {
            capture(pool, &identity, source, depth);
            assert!(pool.remove_anchor(identity.key_at(depth).unwrap()).unwrap());
        });
        assert_bounded(work, 0, "interior insert and drop");
    }
    pool.release(source).unwrap();
    pool.audit().unwrap();
}

#[test]
fn a_late_branch_of_a_long_prefix_compares_digests_not_tokens() {
    let mut pool = pool();
    let tokens = session(3, DEPTH);
    let first = chain(&tokens);
    let source = pool.create_sequence(DEPTH).unwrap();
    pool.append(source, DEPTH).unwrap();
    capture(&mut pool, &first, source, DEPTH);
    pool.release(source).unwrap();
    let mut branched = tokens.clone();
    for token in &mut branched[DEPTH - 100..] {
        *token += 500_000;
    }
    let second = chain(&branched);
    let source = pool.create_sequence(DEPTH).unwrap();
    pool.append(source, DEPTH).unwrap();
    let work = measured(&mut pool, |pool| capture(pool, &second, source, DEPTH));
    assert_bounded(work, 100, "late branch");
    pool.release(source).unwrap();
    pool.audit().unwrap();
}

#[test]
fn high_fanout_under_a_shared_prefix_stays_bounded() {
    let mut pool = pool();
    let system = session(4, 4096);
    let shared = chain(&system);
    let source = pool.create_sequence(4096).unwrap();
    pool.append(source, 4096).unwrap();
    capture(&mut pool, &shared, source, 4096);
    pool.release(source).unwrap();
    let mut users = Vec::new();
    for user in 0..1000u32 {
        let identity = chain(&[system.as_slice(), &[2_000_000 + user; 64]].concat());
        let work = measured(&mut pool, |pool| {
            let hit = pool
                .resume_prefix(&identity, 4096 + 63, 4096 + 64)
                .unwrap()
                .unwrap();
            pool.finish_restore(hit.handle()).unwrap();
            pool.append(hit.handle(), 64).unwrap();
            capture(pool, &identity, hit.handle(), 4096 + 64);
            pool.release(hit.handle()).unwrap();
        });
        assert_bounded(work, 64, "fanout child");
        users.push(identity);
    }
    // Recapturing the shared prefix above its thousand children.
    let key = shared.key_at(4096).unwrap();
    let work = measured(&mut pool, |pool| assert!(pool.remove_anchor(key).unwrap()));
    assert_bounded(work, 0, "fanout parent eviction");
    let source = pool.create_sequence(4096).unwrap();
    pool.append(source, 4096).unwrap();
    let work = measured(&mut pool, |pool| capture(pool, &shared, source, 4096));
    assert_bounded(work, 0, "fanout parent recapture");
    pool.release(source).unwrap();
    for identity in users.iter().step_by(7) {
        let key = identity.key_at(4096 + 64).unwrap();
        let work = measured(&mut pool, |pool| assert!(pool.remove_anchor(key).unwrap()));
        assert_bounded(work, 0, "fanout child eviction");
    }
    pool.audit().unwrap();
}

/// Edges from different insertions merge by copying up to 8,192 tokens;
/// longer neighbours keep a junction node. Lookups, splits below a junction
/// and removals keep exact content ancestry.
#[test]
fn long_edges_from_different_insertions_merge_or_keep_a_junction() {
    for (first, second) in [(4_000, 6_000), (10_000, 20_000)] {
        let mut pool = pool();
        let tokens = session(5, second + BLOCK);
        let short = chain(&tokens[..first]);
        let long = chain(&tokens[..second]);
        let source = pool.create_sequence(first).unwrap();
        pool.append(source, first).unwrap();
        capture(&mut pool, &short, source, first);
        pool.release(source).unwrap();
        let hit = pool
            .resume_prefix(&long, second - 1, second)
            .unwrap()
            .unwrap();
        assert_eq!(hit.tokens(), first);
        pool.finish_restore(hit.handle()).unwrap();
        pool.append(hit.handle(), second - first).unwrap();
        capture(&mut pool, &long, hit.handle(), second);
        let work = measured(&mut pool, |pool| {
            assert!(pool.remove_anchor(short.key_at(first).unwrap()).unwrap());
        });
        let merged = second <= 8192;
        assert_eq!(work.tokens_copied, if merged { second as u64 } else { 0 });
        pool.audit().unwrap();
        // A capture below the junction splits the lower edge in place.
        let middle = (first + second) / 2;
        let work = measured(&mut pool, |pool| capture(pool, &long, hit.handle(), middle));
        assert_bounded(work, 0, "split below a junction");
        pool.release(hit.handle()).unwrap();
        let probe = chain(&tokens);
        assert_eq!(
            pool.lookup_anchor(&probe, second + 10)
                .map(|(_, tokens)| tokens),
            Some(second)
        );
        assert_eq!(
            pool.lookup_anchor(&probe, second - 1)
                .map(|(_, tokens)| tokens),
            Some(middle)
        );
        for depth in [second, middle] {
            assert!(pool.remove_anchor(long.key_at(depth).unwrap()).unwrap());
            pool.audit().unwrap();
        }
        assert_eq!(pool.anchor_count(), 0);
        assert_eq!(pool.stats().free_blocks(), 40_000);
    }
}

/// One eight-row media span ten tokens into every block that starts in
/// `start..end`, where the span also ends by `end`.
fn media(start: usize, end: usize, identity: u8) -> Vec<PrefixMediaSpanV1> {
    (start.div_ceil(BLOCK)..end.div_ceil(BLOCK))
        .map(|block| PrefixMediaSpanV1 {
            identity: [identity; 32],
            start: block * BLOCK + 10,
            rows: 8,
            media_row: 0,
        })
        .filter(|span| span.start + 8 <= end)
        .collect()
}

#[test]
fn a_short_merge_copies_only_the_media_of_its_range() {
    let mut pool = pool();
    let tokens = session(9, DEPTH);
    let long = PrefixChainV1::new(BLOCK, Arc::from(&tokens[..]), media(0, DEPTH, 1)).unwrap();
    let source = pool.create_sequence(DEPTH).unwrap();
    pool.append(source, DEPTH).unwrap();
    for depth in [DEPTH, 4_000, 6_000] {
        capture(&mut pool, &long, source, depth);
    }
    pool.release(source).unwrap();
    // Anchors at 4,000 and 6,000 now share the long segment and its 2,048
    // media spans; dropping the deepest anchor leaves 6,000 as a leaf.
    assert!(pool.remove_anchor(long.key_at(DEPTH).unwrap()).unwrap());
    let mut branch = tokens[..6_000].to_vec();
    branch.extend(session(10, 1_000));
    let mut spans = media(0, 6_000, 1);
    spans.extend(media(6_000, 7_000, 2));
    let branch = PrefixChainV1::new(BLOCK, Arc::from(branch), spans).unwrap();
    let hit = pool.resume_prefix(&branch, 6_999, 7_000).unwrap().unwrap();
    assert_eq!(hit.tokens(), 6_000);
    pool.finish_restore(hit.handle()).unwrap();
    pool.append(hit.handle(), 1_000).unwrap();
    capture(&mut pool, &branch, hit.handle(), 7_000);
    pool.release(hit.handle()).unwrap();
    // Removing 6,000 joins 4,000..6,000 of the long segment with the
    // branch's edge: only the spans starting in 4,000..7,000 are copied.
    let work = measured(&mut pool, |pool| {
        assert!(pool.remove_anchor(long.key_at(6_000).unwrap()).unwrap());
    });
    let expected = media(4_000, 6_000, 1).len() + media(6_000, 7_000, 2).len();
    assert_eq!(expected, 47);
    assert_eq!(work.media_copied, expected as u64);
    assert_eq!(work.tokens_copied, 3_000);
    assert_bounded(work, 3_000, "short merge of a media-heavy segment");
    pool.audit().unwrap();
    // The merged edge keeps its media: the same tokens over one other span
    // at 5,002 must split there, which the audit checks against digests.
    let mut other = media(0, 6_000, 1);
    other.extend(media(6_000, 7_000, 2));
    other[media(0, 5_000, 1).len()].identity = [3; 32];
    assert_eq!(other[media(0, 5_000, 1).len()].start, 5_002);
    let other = PrefixChainV1::new(BLOCK, Arc::from(branch.tokens()), other).unwrap();
    let source = pool.create_sequence(7_000).unwrap();
    pool.append(source, 7_000).unwrap();
    let work = measured(&mut pool, |pool| capture(pool, &other, source, 7_000));
    assert_bounded(work, 2_000, "media divergence inside a merged edge");
    pool.release(source).unwrap();
    pool.audit().unwrap();
    assert_eq!(pool.anchor_count(), 3);
    for chain in [&branch, &other] {
        assert_eq!(
            pool.lookup_anchor(chain, 7_000).map(|(_, tokens)| tokens),
            Some(7_000)
        );
    }
}

fn timed(f: impl FnOnce()) -> std::time::Duration {
    let started = std::time::Instant::now();
    f();
    started.elapsed()
}

/// Sixteen 131K-token sessions captured at 1/8, 1/4, 1/2 and the end, then
/// one allocation under pressure; indexed against digest-only anchors.
fn ascending_and_pressure(indexed: bool) -> (std::time::Duration, std::time::Duration, u64) {
    let mut pool = KvPool::new(KvGeometry::new(BLOCK, 40_000).unwrap(), 4)
        .unwrap()
        .with_prefix_ancestry(indexed);
    let mut insert = std::time::Duration::ZERO;
    for seed in 0..16 {
        let identity = chain(&session(seed + 10, DEPTH));
        let source = pool.create_sequence(DEPTH).unwrap();
        pool.append(source, DEPTH).unwrap();
        for depth in [DEPTH / 8, DEPTH / 4, DEPTH / 2, DEPTH - 7] {
            insert += timed(|| {
                if indexed {
                    capture(&mut pool, &identity, source, depth);
                } else {
                    pool.insert_prefix_anchor(
                        identity.key_at(depth).unwrap(),
                        identity.chain(depth / BLOCK).unwrap(),
                        source,
                        depth,
                        1,
                    )
                    .unwrap()
                    .unwrap();
                }
            });
        }
        pool.release(source).unwrap();
    }
    let before = pool.metrics().snapshot().prefix_cache_evictions;
    let pressure = pool.create_sequence(20_000 * BLOCK).unwrap();
    let evict = timed(|| {
        pool.append(pressure, 20_000 * BLOCK).unwrap();
    });
    let evictions = pool.metrics().snapshot().prefix_cache_evictions - before;
    pool.release(pressure).unwrap();
    (insert / 64, evict, evictions)
}

#[test]
#[ignore = "release timing benchmark"]
fn long_context_bookkeeping_benchmark() {
    for indexed in [false, true] {
        let (capture_each, evict, evictions) = ascending_and_pressure(indexed);
        println!(
            "indexed={indexed}: capture {capture_each:?} each; pressure append {evict:?} for {evictions} evictions ({:?} each)",
            evict / evictions.max(1) as u32
        );
    }
    let mut pool = pool();
    let tokens = session(30, DEPTH + 64 * BLOCK);
    let identity = chain(&tokens[..DEPTH]);
    let source = pool.create_sequence(DEPTH).unwrap();
    pool.append(source, DEPTH).unwrap();
    capture(&mut pool, &identity, source, DEPTH);
    let rounds = 1000u32;
    let elapsed = timed(|| {
        for round in 0..rounds as usize {
            let depth = 1 + (round * 7_919 + 333) % (DEPTH - 1);
            capture(&mut pool, &identity, source, depth);
            pool.remove_anchor(identity.key_at(depth).unwrap()).unwrap();
        }
    });
    println!(
        "interior insert and drop at 131K: {:?} per pair",
        elapsed / rounds
    );
    pool.release(source).unwrap();
    // Sixty-four session turns of 64 tokens past 131K. Hashing the prompts
    // is the frontend's work, outside the timed pool operations.
    let turns: Vec<_> = (1..=64)
        .map(|turn| chain(&tokens[..DEPTH + turn * BLOCK]))
        .collect();
    let mut depth = DEPTH;
    let elapsed = timed(|| {
        for next in &turns {
            let hit = pool
                .resume_prefix(next, depth + BLOCK - 1, depth + BLOCK)
                .unwrap()
                .unwrap();
            pool.finish_restore(hit.handle()).unwrap();
            pool.append(hit.handle(), depth + BLOCK - hit.tokens())
                .unwrap();
            capture(&mut pool, next, hit.handle(), depth + BLOCK);
            pool.release(hit.handle()).unwrap();
            depth += BLOCK;
        }
    });
    println!(
        "session turn at 131K (restore, append 64, capture): {:?} each",
        elapsed / 64
    );
    let mut branched = tokens[..DEPTH].to_vec();
    for token in &mut branched[DEPTH - 100..] {
        *token += 500_000;
    }
    let second = chain(&branched);
    let source = pool.create_sequence(DEPTH).unwrap();
    pool.append(source, DEPTH).unwrap();
    let elapsed = timed(|| capture(&mut pool, &second, source, DEPTH));
    println!("late branch 100 tokens before 131K: {elapsed:?}");
    pool.release(source).unwrap();
    let system = session(31, 4096);
    let shared = chain(&system);
    let source = pool.create_sequence(4096).unwrap();
    pool.append(source, 4096).unwrap();
    capture(&mut pool, &shared, source, 4096);
    pool.release(source).unwrap();
    let users: Vec<_> = (0..1000u32)
        .map(|user| chain(&[system.as_slice(), &[3_000_000 + user; 64]].concat()))
        .collect();
    let elapsed = timed(|| {
        for identity in &users {
            let hit = pool
                .resume_prefix(identity, 4096 + 63, 4096 + 64)
                .unwrap()
                .unwrap();
            pool.finish_restore(hit.handle()).unwrap();
            pool.append(hit.handle(), 64).unwrap();
            capture(&mut pool, identity, hit.handle(), 4096 + 64);
            pool.release(hit.handle()).unwrap();
        }
    });
    println!(
        "fanout user turn under a shared 4K prefix: {:?} each",
        elapsed / 1000
    );
    let key = shared.key_at(4096).unwrap();
    let elapsed = timed(|| {
        pool.remove_anchor(key).unwrap();
    });
    println!("evict a shared prefix above 1,000 children: {elapsed:?}");
    pool.audit().unwrap();
}
