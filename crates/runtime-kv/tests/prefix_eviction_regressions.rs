//! Ancestry eviction against global LRU on shared prefixes, repeats and
//! interior captures. Each case names the outcome LRU gives; ancestry
//! eviction must match or keep more.
use std::sync::Arc;

use runtime_kv::{KvGeometry, KvPool, PrefixChainV1, SequenceHandle};

fn chain(tokens: &[u32]) -> PrefixChainV1 {
    PrefixChainV1::new(4, Arc::from(tokens), Vec::new()).unwrap()
}

fn pool(pages: usize, ancestry: bool) -> KvPool {
    KvPool::new(KvGeometry::new(4, pages).unwrap(), 8)
        .unwrap()
        .with_prefix_ancestry(ancestry)
}

/// Global LRU, the ancestry index alone, the RAM tier with nothing
/// demoted, or the index with its tier order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Order {
    Lru,
    Index,
    Ram,
    Ancestry,
}

fn ordered(pool: KvPool, order: Order) -> KvPool {
    pool.with_prefix_ancestry(order == Order::Ancestry)
        .with_prefix_index(order == Order::Index)
        .with_prefix_ram(order == Order::Ram)
}

fn cold_capture(pool: &mut KvPool, chain: &PrefixChainV1, depth: usize) {
    let source = pool.create_sequence(depth).unwrap();
    pool.append(source, depth).unwrap();
    pool.insert_content_anchor(chain, source, depth, 1)
        .unwrap()
        .unwrap();
    pool.release(source).unwrap();
}

/// Resumes the longest anchor before the prompt's last token, or starts
/// cold, and returns the sequence with the reused tokens.
fn resume(pool: &mut KvPool, chain: &PrefixChainV1) -> (SequenceHandle, usize) {
    let tokens = chain.tokens().len();
    match pool.resume_prefix(chain, tokens - 1, tokens + 8).unwrap() {
        Some(hit) => {
            pool.finish_restore(hit.handle()).unwrap();
            (hit.handle(), hit.tokens())
        }
        None => (pool.create_sequence(tokens + 8).unwrap(), 0),
    }
}

/// A request that resumes, prefills and captures its prompt end.
fn request(pool: &mut KvPool, chain: &PrefixChainV1) -> usize {
    let (handle, hit) = resume(pool, chain);
    let tokens = chain.tokens().len();
    pool.append(handle, tokens - hit).unwrap();
    let _ = pool
        .insert_content_anchor(chain, handle, tokens, 1)
        .unwrap();
    pool.release(handle).unwrap();
    hit
}

fn pressure(pool: &mut KvPool, pages: usize) {
    let handle = pool.create_sequence(4 * pages).unwrap();
    pool.append(handle, 4 * pages).unwrap();
    pool.release(handle).unwrap();
}

fn with_tokens(prefix: &[u32], suffix: &[u32]) -> PrefixChainV1 {
    chain(&[prefix, suffix].concat())
}

/// The review probe: a system prompt two users hit is the most recently
/// used anchor; LRU evicts the oldest one-off frontier instead of it.
#[test]
fn hot_shared_prefix_outlives_older_one_off_frontiers() {
    for ancestry in [false, true] {
        let mut pool = pool(40, ancestry);
        let stale: Vec<_> = (0..4u32).map(|i| chain(&[500 + i; 8])).collect();
        for one_off in &stale {
            cold_capture(&mut pool, one_off, 8);
        }
        let system: Vec<u32> = (1..=8).collect();
        cold_capture(&mut pool, &chain(&system), 8);
        assert_eq!(request(&mut pool, &with_tokens(&system, &[600; 4])), 8);
        assert_eq!(request(&mut pool, &with_tokens(&system, &[700; 4])), 8);
        let free = pool.stats().free_blocks();
        pressure(&mut pool, free + 1);
        assert!(pool.has_anchor(chain(&system).key_at(8).unwrap()));
        assert!(!pool.has_anchor(stale[0].key_at(8).unwrap()));
        assert_eq!(
            request(&mut pool, &with_tokens(&system, &[800; 4])),
            8,
            "ancestry {ancestry}"
        );
        pool.audit().unwrap();
    }
}

/// A cold-captured system prompt that a second user resumed as a frontier
/// is not yet a conversation moving past it: unrelated pressure must not
/// take it before an older one-off anchor.
#[test]
fn prefix_extended_once_still_serves_a_later_user() {
    for ancestry in [false, true] {
        let mut pool = pool(32, ancestry);
        let stale = chain(&[99; 8]);
        cold_capture(&mut pool, &stale, 8);
        let system: Vec<u32> = (0..8).collect();
        cold_capture(&mut pool, &chain(&system), 8);
        assert_eq!(request(&mut pool, &with_tokens(&system, &[80; 4])), 8);
        let free = pool.stats().free_blocks();
        pressure(&mut pool, free + 1);
        assert!(!pool.has_anchor(stale.key_at(8).unwrap()));
        assert_eq!(
            request(&mut pool, &with_tokens(&system, &[90; 4])),
            8,
            "ancestry {ancestry}"
        );
        pool.audit().unwrap();
    }
}

/// When only frontiers remain, a turn's capture replaces the frontier its
/// own sequence extended rather than an unrelated idle frontier. LRU evicts
/// the idle frontier here.
#[test]
fn capture_replaces_the_frontier_its_sequence_extended() {
    for ancestry in [false, true] {
        let mut pool = pool(5, ancestry);
        let idle = chain(&[20; 4]);
        cold_capture(&mut pool, &idle, 4);
        cold_capture(&mut pool, &chain(&[1, 2, 3]), 3);
        let extended = chain(&[1, 2, 3, 4]);
        let (handle, hit) = resume(&mut pool, &extended);
        assert_eq!(hit, 3);
        pool.append(handle, 1).unwrap();
        pool.insert_content_anchor(&extended, handle, 4, 1)
            .unwrap()
            .unwrap();
        pool.release(handle).unwrap();
        assert_eq!(pool.has_anchor(idle.key_at(4).unwrap()), ancestry);
        assert!(pool.has_anchor(extended.key_at(4).unwrap()));
        pool.audit().unwrap();
    }
}

/// An interior capture and the prompt end written from one sequence in one
/// step, under pressure: the end capture takes an older idle anchor, as LRU
/// does, not the interior it was written beside.
#[test]
fn prompt_end_capture_keeps_the_interior_written_beside_it() {
    for ancestry in [false, true] {
        let mut pool = pool(5, ancestry);
        let idle = chain(&[90; 4]);
        cold_capture(&mut pool, &idle, 4);
        let prompt = chain(&(0..8).collect::<Vec<_>>());
        let source = pool.create_sequence(8).unwrap();
        pool.append(source, 8).unwrap();
        pool.insert_content_anchor(&prompt, source, 4, 1)
            .unwrap()
            .unwrap();
        assert_eq!(pool.stats().free_blocks(), 0);
        pool.insert_content_anchor(&prompt, source, 8, 1)
            .unwrap()
            .unwrap();
        pool.release(source).unwrap();
        assert!(
            !pool.has_anchor(idle.key_at(4).unwrap()),
            "ancestry {ancestry}"
        );
        assert!(pool.has_anchor(prompt.key_at(4).unwrap()));
        assert!(pool.has_anchor(prompt.key_at(8).unwrap()));
        pool.audit().unwrap();
    }
}

/// A conversation turn: resume its frontier, append and capture the new
/// prompt end.
fn conversation_turn(pool: &mut KvPool, history: &mut Vec<u32>, append: &[u32]) {
    history.extend_from_slice(append);
    request(pool, &chain(history));
}

/// An exact repeat cannot restore its own prompt end, so it restores the
/// primer below it. Busy conversations then churn the pool. LRU loses the
/// primer to their newer turns; ancestry keeps it, since the repeat showed
/// its descendant cannot serve that prompt, and evicts the conversations'
/// passed ancestors instead.
#[test]
fn repeat_primer_outlives_conversation_churn() {
    let mut kept = Vec::new();
    for ancestry in [false, true] {
        let mut pool = pool(64, ancestry);
        let repeat: Vec<u32> = (300..312).collect();
        cold_capture(&mut pool, &chain(&repeat[..8]), 8);
        cold_capture(&mut pool, &chain(&repeat), 12);
        assert_eq!(request(&mut pool, &chain(&repeat)), 8);
        let mut conversations: Vec<Vec<u32>> = (0..3).map(|c| vec![1000 * (c + 1); 4]).collect();
        for turn in 0..12u32 {
            for (index, history) in conversations.iter_mut().enumerate() {
                conversation_turn(&mut pool, history, &[turn * 10 + index as u32; 4]);
            }
        }
        pool.audit().unwrap();
        kept.push(pool.has_anchor(chain(&repeat).key_at(8).unwrap()));
        let hit = request(&mut pool, &chain(&repeat));
        assert_eq!(hit > 0, kept[kept.len() - 1]);
    }
    assert_eq!(
        kept,
        [false, true],
        "LRU loses the primer, ancestry keeps it"
    );
}

struct Xorshift(u64);

impl Xorshift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// The scheduler's ordinary capture points without a grid: the shared
/// full blocks an earlier prompt produced, past the hit, then the prompt end.
fn scheduler_turn(
    pool: &mut KvPool,
    tokens: &[u32],
    salt: Option<&[u8]>,
    snapshot: usize,
    outputs: usize,
) -> usize {
    let chain = PrefixChainV1::salted(64, Arc::from(tokens), Vec::new(), salt).unwrap();
    let prompt = tokens.len();
    let seen = pool.observe_prefix(&chain);
    let (handle, hit) = match pool
        .resume_prefix(&chain, prompt - 1, prompt + outputs)
        .unwrap()
    {
        Some(hit) => {
            pool.finish_restore(hit.handle()).unwrap();
            (hit.handle(), hit.tokens())
        }
        None => (pool.create_sequence(prompt + outputs).unwrap(), 0),
    };
    let shared = (seen * 64).min((prompt - 1) / 64 * 64);
    let resumes_final_chunk = hit > 0 && prompt - hit <= 2048;
    let mut points = Vec::new();
    if shared > hit
        && !resumes_final_chunk
        && shared >= 64
        && !pool.has_anchor(chain.key_at(shared).unwrap())
    {
        points.push(shared);
    }
    if prompt > hit && prompt >= 64 {
        points.push(prompt);
    }
    for point in points {
        let committed = pool.sequence_tokens(handle).unwrap();
        pool.append(handle, point - committed).unwrap();
        let _ = pool
            .insert_content_anchor(&chain, handle, point, snapshot)
            .unwrap();
    }
    let committed = pool.sequence_tokens(handle).unwrap();
    pool.append(handle, prompt - committed + outputs).unwrap();
    pool.release(handle).unwrap();
    hit
}

fn scheduler_pool(order: Order) -> KvPool {
    ordered(
        KvPool::new(KvGeometry::new(64, 2048).unwrap(), 8).unwrap(),
        order,
    )
}

/// API-style traffic: 64 shared 256-token prefixes, each request a one-off
/// tail with no follow-up. Returns reused and prompt tokens.
fn shared_one_off(order: Order) -> (usize, usize) {
    let mut pool = scheduler_pool(order);
    let mut rng = Xorshift(11);
    let (mut reused, mut prompts) = (0, 0);
    for request in 0..2000u32 {
        let base = (rng.next() % 64) as u32;
        let mut tokens: Vec<u32> = (0..256).map(|t| base * 1000 + t).collect();
        for t in 0..1 + rng.next() % 64 {
            let _ = rng.next();
            tokens.push(1_000_000 + request * 100 + t as u32);
        }
        reused += scheduler_turn(&mut pool, &tokens, None, 53, 1);
        prompts += tokens.len();
    }
    pool.audit().unwrap();
    (reused, prompts)
}

/// Default thinking removal: every prompt drops the previous generation
/// prompt token, so the useful anchor is the shared-page point below each
/// prompt end. Eight salted conversations, five turns, six episodes.
fn default_thinking(order: Order) -> (usize, usize) {
    let mut pool = scheduler_pool(order);
    let (mut reused, mut prompts) = (0, 0);
    let mut fresh = 10_000_000u32;
    let mut token = || {
        fresh += 1;
        fresh
    };
    for episode in 0..6u8 {
        let mut sessions: Vec<Vec<u32>> = (0..8)
            .map(|_| (0..512).map(|_| token()).collect())
            .collect();
        for turn in 0..5 {
            for (index, history) in sessions.iter_mut().enumerate() {
                if turn > 0 {
                    history.pop();
                    for _ in 0..129 {
                        history.push(token());
                    }
                }
                let salt = [episode, index as u8];
                reused += scheduler_turn(&mut pool, history, Some(&salt), 53, 64);
                prompts += history.len();
            }
        }
    }
    pool.audit().unwrap();
    (reused, prompts)
}

#[test]
fn shared_one_off_traffic_reuses_at_least_what_lru_reuses() {
    let lru = shared_one_off(Order::Lru);
    let ancestry = shared_one_off(Order::Ancestry);
    println!("shared one-off reused/prompt tokens: LRU {lru:?}, ancestry {ancestry:?}");
    assert_eq!(lru, (119_296, 577_204));
    assert!(ancestry.0 >= lru.0, "{ancestry:?} < {lru:?}");
}

#[test]
fn default_thinking_reuses_at_least_what_lru_reuses() {
    let lru = default_thinking(Order::Lru);
    let ancestry = default_thinking(Order::Ancestry);
    println!("default thinking reused/prompt tokens: LRU {lru:?}, ancestry {ancestry:?}");
    assert_eq!(lru, (64_512, 184_320));
    assert!(ancestry.0 >= lru.0, "{ancestry:?} < {lru:?}");
}

/// The RAM tier indexes every public anchor while eviction keeps global
/// LRU: the index alone changes no reuse.
#[test]
fn index_without_tier_order_reuses_what_lru_reuses() {
    assert_eq!(shared_one_off(Order::Index), (119_296, 577_204));
    assert_eq!(default_thinking(Order::Index), (64_512, 184_320));
}

/// Conversations that branch, repeat and continue under pressure, run on
/// four pools in lockstep. The index-only pool and the RAM tier pool with
/// nothing demoted match LRU in every hit, victim, page count and metric;
/// the ancestry pool shows the traffic separates the two orders.
#[test]
fn index_without_tier_order_evicts_exactly_as_lru() {
    let orders = [Order::Lru, Order::Index, Order::Ram, Order::Ancestry];
    let mut pools = orders.map(|order| {
        ordered(
            KvPool::new(KvGeometry::new(4, 48).unwrap(), 8).unwrap(),
            order,
        )
    });
    let mut rng = Xorshift(7);
    let mut histories: Vec<Vec<u32>> = (0..6).map(|c| vec![1000 * (c + 1); 4]).collect();
    let mut seen = Vec::new();
    let mut diverged = false;
    for step in 0..3000u32 {
        let index = (rng.next() % 6) as usize;
        let pressure_pages = match rng.next() % 8 {
            0..=4 => {
                let tokens = 1 + (rng.next() % 6) as usize;
                histories[index].extend((0..tokens as u32).map(|t| step * 8 + t));
                0
            }
            5 => {
                // Branch from another conversation's history.
                let other = histories[(rng.next() % 6) as usize].clone();
                let keep = 4 + (rng.next() as usize) % (other.len() - 3);
                histories[index] = other[..keep].to_vec();
                histories[index].push(step * 8 + 7);
                0
            }
            6 => 0,
            _ => 1 + (rng.next() % 8) as usize,
        };
        if histories[index].len() > 72 {
            histories[index] = vec![1000 * (index as u32 + 1) + step; 4];
        }
        let prompt = chain(&histories[index]);
        let mut outcomes = Vec::new();
        for pool in &mut pools {
            let hit = if pressure_pages > 0 {
                pressure(pool, pressure_pages);
                0
            } else {
                request(pool, &prompt)
            };
            outcomes.push((hit, pool.stats(), pool.metrics().snapshot()));
        }
        if pressure_pages == 0 {
            seen.push(prompt.key_at(prompt.tokens().len()).unwrap());
        }
        assert_eq!(outcomes[1], outcomes[0], "step {step}");
        assert_eq!(outcomes[2], outcomes[0], "step {step}");
        diverged |= outcomes[3].0 != outcomes[0].0;
        if step % 25 == 0 {
            let present = |pool: &KvPool| {
                seen.iter()
                    .map(|key| pool.has_anchor(*key))
                    .collect::<Vec<_>>()
            };
            assert_eq!(present(&pools[1]), present(&pools[0]), "step {step}");
            assert_eq!(present(&pools[2]), present(&pools[0]), "step {step}");
            diverged |= present(&pools[3]) != present(&pools[0]);
            for pool in &pools {
                pool.audit().unwrap();
            }
        }
    }
    for pool in &pools {
        pool.audit().unwrap();
    }
    assert!(diverged, "the traffic never separated ancestry from LRU");
}
