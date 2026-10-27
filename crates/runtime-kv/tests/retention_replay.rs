//! Deterministic real-pool port of the retention policy scenario. One logical
//! token represents a 64-token page; page counts and snapshot costs are exact.
use runtime_kv::{KvGeometry, KvPool, PrefixChainV1};

#[derive(Debug)]
struct Outcome {
    idle_hits: Vec<usize>,
    recomputed_tokens: usize,
    evictions: [u64; 3],
    used_pages: usize,
    snapshot_pages: u64,
}

fn replay(candidate: bool) -> Outcome {
    let mut pool = KvPool::new(KvGeometry::new(1, 318_000).unwrap(), 12)
        .unwrap()
        .with_prefix_ancestry(candidate);
    let background = pool.create_sequence(80_000).unwrap();
    pool.append(background, 80_000).unwrap();
    let mut histories: Vec<Vec<u32>> = (0..9)
        .map(|session| (0..65).map(|page| session * 100_000 + page).collect())
        .collect();
    for history in &histories {
        let chain = PrefixChainV1::new(1, history.clone().into(), vec![]).unwrap();
        let source = pool.create_sequence(2000).unwrap();
        pool.append(source, 65).unwrap();
        for depth in [64, 65] {
            pool.insert_content_anchor(&chain, source, depth, 125)
                .unwrap()
                .unwrap();
        }
        pool.release(source).unwrap();
    }
    let mut idle_hits = Vec::new();
    for second in 1..=7200 {
        if second % 8 == 0 {
            for history in histories.iter_mut().take(6) {
                let previous = history.len();
                let last = *history.last().unwrap();
                history.extend([last + 1, last + 2]);
                let chain = PrefixChainV1::new(1, history.clone().into(), vec![]).unwrap();
                let source = if let Some(hit) = pool.resume_prefix(&chain, previous, 2000).unwrap()
                {
                    pool.finish_restore(hit.handle()).unwrap();
                    hit.handle()
                } else {
                    pool.create_sequence(2000).unwrap()
                };
                let committed = pool.snapshot(source).unwrap().tokens();
                pool.append(source, history.len() - committed).unwrap();
                for depth in [previous + 1, previous + 2] {
                    pool.insert_content_anchor(&chain, source, depth, 125)
                        .unwrap()
                        .unwrap();
                }
                pool.release(source).unwrap();
            }
        }
        if let Some(session) = match second {
            1800 => Some(6),
            3600 => Some(7),
            7200 => Some(8),
            _ => None,
        } {
            let chain = PrefixChainV1::new(1, histories[session].clone().into(), vec![]).unwrap();
            let hit = pool.resume_prefix(&chain, 65, 2000).unwrap();
            idle_hits.push(hit.as_ref().map_or(0, |hit| hit.tokens() * 64));
            if let Some(hit) = hit {
                pool.finish_restore(hit.handle()).unwrap();
                pool.release(hit.handle()).unwrap();
            }
            pool.audit().unwrap();
        }
    }
    let metrics = pool.metrics().snapshot();
    let outcome = Outcome {
        recomputed_tokens: idle_hits.iter().map(|hit| 4160 - hit).sum(),
        idle_hits,
        evictions: metrics.prefix_evictions_by_class,
        used_pages: 318_000 - pool.stats().free_blocks(),
        snapshot_pages: metrics.prefix_snapshot_pages_by_class.iter().sum(),
    };
    pool.audit().unwrap();
    outcome
}

#[test]
fn real_pool_retention_preserves_three_idle_frontiers() {
    let old = replay(false);
    let new = replay(true);
    println!("LRU: {old:?}\nAncestry: {new:?}");
    assert_eq!(old.idle_hits, [0, 0, 0]);
    assert_eq!(new.idle_hits, [4160, 4160, 4160]);
    assert_eq!(old.recomputed_tokens, 12_480);
    assert_eq!(new.recomputed_tokens, 0);
    assert_eq!(new.evictions[2], 0);
    assert!(old.used_pages <= 318_000 && new.used_pages <= 318_000);
    assert!(old.snapshot_pages > 200_000 && new.snapshot_pages > 200_000);
}
