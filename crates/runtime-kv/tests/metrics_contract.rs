use std::sync::Arc;

use runtime_kv::{KvCachePrecision, KvGeometry, KvMetrics, KvPool, PrefixChainV1};

#[test]
fn kv_records_fp8_explicit_shared_prefix_and_copy_on_write_at_commit() {
    let metrics = KvMetrics::new(KvCachePrecision::Fp8);
    let mut pool =
        KvPool::new_with_metrics(KvGeometry::new(4, 8).unwrap(), 2, metrics.clone()).unwrap();

    let parent = pool.create_sequence(12).unwrap();
    pool.append(parent, 3).unwrap();
    let child = pool.fork(parent, 12).unwrap();
    let report = pool.append(child, 1).unwrap();
    assert_eq!(report.copy().unwrap().initialized_tokens(), 3);

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.fp8_tokens, 7);
    assert_eq!(snapshot.fp8_capacity_tokens, 32);
    assert_eq!(snapshot.explicit_shared_prefix_forks, 1);
    assert_eq!(snapshot.explicit_shared_prefix_tokens, 3);
    assert_eq!(snapshot.cow_copies, 1);
    assert_eq!(snapshot.cow_tokens, 3);

    pool.cancel(child).unwrap();
    pool.release(parent).unwrap();
    assert_eq!(metrics.snapshot().fp8_tokens, 0);
    let text = metrics.snapshot().prometheus();
    for name in [
        "chop_fp8_kv_tokens",
        "chop_fp8_kv_capacity_tokens",
        "chop_explicit_shared_prefix_forks_total",
        "chop_explicit_shared_prefix_tokens_total",
        "chop_prefix_cache_queries_total",
        "chop_prefix_cache_hits_total",
        "chop_prefix_cache_reused_tokens_total",
        "chop_prefix_cache_queried_tokens_total",
        "chop_cow_copies_total",
        "chop_cow_copied_tokens_total",
    ] {
        assert!(text.contains(name), "missing metric {name}");
    }
    pool.audit().unwrap();
}

#[test]
fn kv_records_deterministic_prefix_cache_misses_hits_and_evictions() {
    let metrics = KvMetrics::new(KvCachePrecision::Fp8);
    let mut pool =
        KvPool::new_with_metrics(KvGeometry::new(4, 4).unwrap(), 2, metrics.clone()).unwrap();
    let tokens: Vec<u32> = (0..6).collect();
    let chain = PrefixChainV1::new(4, Arc::from(tokens.as_slice()), Vec::new()).unwrap();
    let miss = PrefixChainV1::new(4, Arc::from(&[9_u32, 9, 9, 9, 9][..]), Vec::new()).unwrap();
    let source = pool.create_sequence(8).unwrap();
    pool.append(source, 4).unwrap();
    pool.insert_anchor(chain.key_at(4).unwrap(), chain.chain(1).unwrap(), source, 1)
        .unwrap()
        .expect("the anchor fits");

    assert_eq!(pool.resume_prefix(&miss, 4, 8).unwrap(), None);
    let reused = pool
        .resume_prefix(&chain, 5, 8)
        .unwrap()
        .expect("the anchor is present");
    assert_eq!(
        metrics.snapshot().prefix_cache_reused_tokens,
        0,
        "nothing is reused before the restore succeeds"
    );
    pool.finish_restore(reused.handle()).unwrap();

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.prefix_cache_queries, 2);
    assert_eq!(snapshot.prefix_cache_hits, 1);
    assert_eq!(snapshot.prefix_cache_reused_tokens, 4);
    assert_eq!(snapshot.prefix_cache_queried_tokens, 5 + 6);
    assert_eq!(snapshot.prefix_cache_captures, 1);
    assert_eq!(snapshot.fp8_tokens, 8);
    pool.release(reused.handle()).unwrap();
    pool.release(source).unwrap();
    assert_eq!(metrics.snapshot().fp8_tokens, 0);
    pool.metrics().observe_occupancy(pool.stats());
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.blocks_used, 0);
    assert_eq!(snapshot.prefix_cache_blocks, 2);
    assert_eq!(snapshot.prefix_cache_anchors, 1);

    let filler = pool.create_sequence(16).unwrap();
    pool.append(filler, 16)
        .expect("the anchor is evicted for a live sequence");
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.prefix_cache_evictions, 1);
    let text = snapshot.prometheus();
    for name in [
        "chop_prefix_cache_recomputed_seen_tokens_total",
        "chop_prefix_cache_captures_total",
        "chop_prefix_cache_capture_failures_total",
        "chop_prefix_cache_restore_failures_total",
        "chop_prefix_cache_evictions_total",
        "chop_prefix_cache_anchors",
        "chop_prefix_cache_blocks",
    ] {
        assert!(text.contains(name), "missing metric {name}");
    }
    pool.audit().unwrap();
}
