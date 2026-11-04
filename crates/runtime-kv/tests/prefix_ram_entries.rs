//! The pool's host entries for the RAM prefix tier: demotion, per-tier
//! residency in the ancestry index, and the logical events that make an
//! entry unusable before its host tier drops it.
use std::sync::Arc;

use runtime_kv::{
    HostAnchorId, HostTier, KvGeometry, KvPool, KvTransferClass, KvTransferCompletion,
    KvTransferPlan, KvTransferStatus, PrefixAnchorClassV1, PrefixCacheKeyV1, PrefixChainV1,
    PrefixEvictionTierV1, PrefixHostInvalidationV1,
};

fn chain(tokens: &[u32]) -> PrefixChainV1 {
    PrefixChainV1::new(4, Arc::from(tokens), Vec::new()).unwrap()
}

fn ram_pool(pages: usize) -> KvPool {
    KvPool::new(KvGeometry::new(4, pages).unwrap(), 8)
        .unwrap()
        .with_prefix_ram(true)
}

/// Captures each depth of `chain` from one cold sequence, with one
/// snapshot page each.
fn capture(pool: &mut KvPool, chain: &PrefixChainV1, depths: &[usize]) {
    let tokens = *depths.iter().max().unwrap();
    let source = pool.create_sequence(tokens).unwrap();
    pool.append(source, tokens).unwrap();
    for &depth in depths {
        pool.insert_content_anchor(chain, source, depth, 1)
            .unwrap()
            .unwrap();
    }
    pool.release(source).unwrap();
}

fn completed(plan: KvTransferPlan) -> KvTransferCompletion {
    plan.into_submission()
        .completion(KvTransferStatus::Complete)
}

/// Demotes `key` and completes its copy.
fn demote(pool: &mut KvPool, tier: &mut HostTier, key: PrefixCacheKeyV1) -> HostAnchorId {
    let (id, plan) = tier.begin_offload(pool, key, None).unwrap();
    assert_eq!(plan.class(), KvTransferClass::Demotion);
    tier.complete_transfer(pool, completed(plan)).unwrap();
    tier.audit(pool).unwrap();
    id
}

fn assert_idle(pool: &KvPool, tier: &HostTier) {
    tier.audit(pool).unwrap();
    assert_eq!(tier.used_pages(), 0);
    assert_eq!(tier.pending_transfers(), 0);
}

#[test]
fn host_entries_keep_their_anchors_last_gpu_use() {
    let mut pool = ram_pool(32);
    let mut tier = HostTier::new(&pool, 32).unwrap();
    let (older, newer) = (chain(&[1; 8]), chain(&[2; 8]));
    capture(&mut pool, &older, &[8]);
    capture(&mut pool, &newer, &[8]);
    let (older, newer) = (older.key_at(8).unwrap(), newer.key_at(8).unwrap());
    // Demotion order does not matter; the last GPU use does.
    let (id, plan) = tier.begin_offload(&mut pool, newer, None).unwrap();
    let entry = pool.host_entry(newer).unwrap();
    assert_eq!((entry.host, entry.tokens, entry.ready), (id, 8, false));
    tier.audit(&pool).unwrap();
    tier.complete_transfer(&mut pool, completed(plan)).unwrap();
    assert!(pool.host_entry(newer).unwrap().ready);
    demote(&mut pool, &mut tier, older);
    let old = pool.host_entry(older).unwrap();
    assert!(old.last_used < entry.last_used);
    assert!(old.serial > entry.serial);
    // GPU eviction leaves the entries in RAM.
    assert!(pool.remove_anchor(older).unwrap());
    assert!(pool.remove_anchor(newer).unwrap());
    assert_eq!(pool.host_entry(older), Some(old));
    tier.audit(&pool).unwrap();
    assert!(tier.drop_public_entry(&mut pool, older).unwrap());
    assert!(!tier.drop_public_entry(&mut pool, older).unwrap());
    assert!(tier.drop_public_entry(&mut pool, newer).unwrap());
    assert_idle(&pool, &tier);
    assert_eq!(pool.stats().free_blocks(), 32);
}

#[test]
fn a_frontier_moved_to_ram_keeps_its_consumed_ancestor_superseded() {
    let prompt = chain(&(0..24).collect::<Vec<_>>());
    let key = |depth| prompt.key_at(depth).unwrap();
    let mut pool = ram_pool(32).with_prefix_ancestry(true);
    let mut tier = HostTier::new(&pool, 32).unwrap();
    capture(&mut pool, &prompt, &[4, 20, 24]);
    let hit = pool.resume_anchor(key(24), 24).unwrap();
    pool.finish_restore(hit.handle()).unwrap();
    pool.release(hit.handle()).unwrap();
    assert_eq!(
        pool.anchor_tier(key(20)),
        Some(PrefixEvictionTierV1::Superseded)
    );
    demote(&mut pool, &mut tier, key(24));
    assert!(pool.remove_anchor(key(24)).unwrap());
    // The node stays while RAM holds the key, so the ancestor still ranks
    // below recency.
    assert_eq!(
        pool.anchor_tier(key(20)),
        Some(PrefixEvictionTierV1::Superseded)
    );
    assert_eq!(
        pool.anchor_class(key(20)),
        Some(PrefixAnchorClassV1::Superseded)
    );
    tier.audit(&pool).unwrap();
    assert!(tier.drop_public_entry(&mut pool, key(24)).unwrap());
    assert_eq!(pool.anchor_tier(key(20)), Some(PrefixEvictionTierV1::Live));
    assert_eq!(
        pool.anchor_class(key(20)),
        Some(PrefixAnchorClassV1::Maximal)
    );
    assert_idle(&pool, &tier);
}

#[test]
fn every_host_tier_drop_updates_the_pool() {
    let prompts: Vec<_> = (1..=4).map(|n| chain(&[n; 8])).collect();
    let keys: Vec<_> = prompts.iter().map(|c| c.key_at(8).unwrap()).collect();
    let mut pool = ram_pool(32);
    let mut tier = HostTier::new(&pool, 3).unwrap();
    for prompt in &prompts {
        capture(&mut pool, prompt, &[8]);
    }
    // A confirmed idle failure drops the entry and its demotion mark.
    let (_, plan) = tier.begin_offload(&mut pool, keys[0], None).unwrap();
    tier.complete_transfer(&mut pool, plan.cancel()).unwrap();
    assert!(pool.host_entry(keys[0]).is_none());
    assert_idle(&pool, &tier);
    // An explicit drop during the copy discards the copy at completion.
    let (_, plan) = tier.begin_offload(&mut pool, keys[0], None).unwrap();
    assert!(tier.drop_public_entry(&mut pool, keys[0]).unwrap());
    tier.audit(&pool).unwrap();
    tier.complete_transfer(&mut pool, completed(plan)).unwrap();
    assert!(pool.host_entry(keys[0]).is_none());
    assert_idle(&pool, &tier);
    // A host eviction for a swap drops a host-only entry and its node.
    demote(&mut pool, &mut tier, keys[1]);
    assert!(pool.remove_anchor(keys[1]).unwrap());
    let source = pool.create_sequence(8).unwrap();
    pool.append(source, 8).unwrap();
    let private = PrefixCacheKeyV1::new([9; 32]);
    pool.insert_private_anchor(private, source, 1)
        .unwrap()
        .unwrap();
    pool.release(source).unwrap();
    let (swap, plan) = tier.begin_offload(&mut pool, private, Some(1)).unwrap();
    assert!(pool.host_entry(keys[1]).is_none());
    tier.complete_transfer(&mut pool, completed(plan)).unwrap();
    tier.audit(&pool).unwrap();
    tier.drop_anchor(swap).unwrap();
    // Unconfirmed status quarantines the demotion: it stays in flight and
    // its entry stays unready.
    let (_, plan) = tier.begin_offload(&mut pool, keys[2], None).unwrap();
    tier.complete_transfer(
        &mut pool,
        plan.into_submission()
            .completion(KvTransferStatus::Unconfirmed),
    )
    .unwrap();
    assert!(!pool.host_entry(keys[2]).unwrap().ready);
    assert_eq!(tier.pending_transfers(), 1);
    tier.audit(&pool).unwrap();
}

#[test]
fn a_resident_descendant_drops_its_ancestors_host_entry() {
    let prompt = chain(&(0..16).collect::<Vec<_>>());
    let key = |depth| prompt.key_at(depth).unwrap();
    for case in ["shadow", "host only", "in flight"] {
        let mut pool = ram_pool(32);
        let mut tier = HostTier::new(&pool, 32).unwrap();
        capture(&mut pool, &prompt, &[8]);
        let plan = if case == "in flight" {
            Some(tier.begin_offload(&mut pool, key(8), None).unwrap().1)
        } else {
            demote(&mut pool, &mut tier, key(8));
            None
        };
        if case == "host only" {
            assert!(pool.remove_anchor(key(8)).unwrap());
        }
        capture(&mut pool, &prompt, &[12]);
        assert_eq!(
            pool.host_invalidation(key(8)),
            Some(PrefixHostInvalidationV1::Descendant),
            "{case}"
        );
        tier.audit(&pool).unwrap();
        assert_eq!(
            tier.drop_invalidated_entries(&mut pool).unwrap(),
            [(key(8), PrefixHostInvalidationV1::Descendant)],
            "{case}"
        );
        assert!(pool.host_entry(key(8)).is_none(), "{case}");
        assert_eq!(pool.has_anchor(key(8)), case != "host only");
        tier.audit(&pool).unwrap();
        if let Some(plan) = plan {
            tier.complete_transfer(&mut pool, completed(plan)).unwrap();
        }
        assert_idle(&pool, &tier);
    }
}

#[test]
fn resumes_and_recaptures_report_their_shadows() {
    let prompts: Vec<_> = (1..=4).map(|n| chain(&[n; 8])).collect();
    let keys: Vec<_> = prompts.iter().map(|c| c.key_at(8).unwrap()).collect();
    let mut pool = ram_pool(48);
    let mut tier = HostTier::new(&pool, 32).unwrap();
    for (prompt, key) in prompts.iter().zip(&keys) {
        capture(&mut pool, prompt, &[8]);
        demote(&mut pool, &mut tier, *key);
    }
    // A resume of a shadow's GPU anchor.
    let hit = pool.resume_anchor(keys[0], 8).unwrap();
    pool.finish_restore(hit.handle()).unwrap();
    pool.release(hit.handle()).unwrap();
    // A prompt lookup that resumes a shadow.
    let longer = chain(&[2; 12]);
    let hit = pool.resume_prefix(&longer, 11, 12).unwrap().unwrap();
    assert_eq!(hit.tokens(), 8);
    pool.finish_restore(hit.handle()).unwrap();
    pool.release(hit.handle()).unwrap();
    // A capture of a key the GPU still holds touches it.
    let source = pool.create_sequence(8).unwrap();
    pool.append(source, 8).unwrap();
    assert_eq!(
        pool.insert_content_anchor(&prompts[2], source, 8, 1),
        Ok(None)
    );
    pool.release(source).unwrap();
    // A capture of a key only RAM holds makes a new GPU anchor.
    assert!(pool.remove_anchor(keys[3]).unwrap());
    capture(&mut pool, &prompts[3], &[8]);
    use PrefixHostInvalidationV1::{Recaptured, Resumed};
    for (key, reason) in keys.iter().zip([Resumed, Resumed, Recaptured, Recaptured]) {
        assert_eq!(pool.host_invalidation(*key), Some(reason));
    }
    tier.audit(&pool).unwrap();
    let mut dropped = tier.drop_invalidated_entries(&mut pool).unwrap();
    dropped.sort();
    let mut expected: Vec<_> = keys
        .iter()
        .copied()
        .zip([Resumed, Resumed, Recaptured, Recaptured])
        .collect();
    expected.sort();
    assert_eq!(dropped, expected);
    // Every key is now exclusive to the GPU.
    for key in &keys {
        assert!(pool.has_anchor(*key) && pool.host_entry(*key).is_none());
    }
    assert_idle(&pool, &tier);
}

#[test]
fn forced_mode_keeps_entries_through_descendants_and_resumes() {
    let prompt = chain(&(0..8).collect::<Vec<_>>());
    let key = |depth| prompt.key_at(depth).unwrap();
    let mut pool = ram_pool(32).with_prefix_ram_forced(true);
    let mut tier = HostTier::new(&pool, 32).unwrap();
    // A prompt's interior and end captured in one step, demoted after.
    capture(&mut pool, &prompt, &[4, 8]);
    demote(&mut pool, &mut tier, key(8));
    demote(&mut pool, &mut tier, key(4));
    let hit = pool.resume_anchor(key(4), 8).unwrap();
    pool.finish_restore(hit.handle()).unwrap();
    pool.release(hit.handle()).unwrap();
    assert_eq!(pool.host_invalidation(key(4)), None);
    tier.audit(&pool).unwrap();
    // A recapture still invalidates.
    let source = pool.create_sequence(8).unwrap();
    pool.append(source, 8).unwrap();
    assert_eq!(pool.insert_content_anchor(&prompt, source, 4, 1), Ok(None));
    pool.release(source).unwrap();
    assert_eq!(
        tier.drop_invalidated_entries(&mut pool).unwrap(),
        [(key(4), PrefixHostInvalidationV1::Recaptured)]
    );
    tier.audit(&pool).unwrap();
    assert!(tier.drop_public_entry(&mut pool, key(8)).unwrap());
    assert_idle(&pool, &tier);
}

#[test]
fn demotions_refuse_anything_but_an_indexed_public_frontier() {
    let prompt = chain(&(0..8).collect::<Vec<_>>());
    let key = |depth| prompt.key_at(depth).unwrap();
    // Without the tier, an unowned copy has no host entry to fill.
    let mut off = KvPool::new(KvGeometry::new(4, 32).unwrap(), 8).unwrap();
    let mut off_tier = HostTier::new(&off, 32).unwrap();
    capture(&mut off, &prompt, &[8]);
    assert!(off_tier.begin_offload(&mut off, key(8), None).is_err());

    let mut pool = ram_pool(32);
    let mut tier = HostTier::new(&pool, 32).unwrap();
    capture(&mut pool, &prompt, &[4, 8]);
    let source = pool.create_sequence(8).unwrap();
    pool.append(source, 8).unwrap();
    let (digest, private) = (
        PrefixCacheKeyV1::new([7; 32]),
        PrefixCacheKeyV1::new([9; 32]),
    );
    pool.insert_anchor(digest, digest, source, 1)
        .unwrap()
        .unwrap();
    pool.insert_private_anchor(private, source, 1)
        .unwrap()
        .unwrap();
    pool.release(source).unwrap();
    let before = pool.clone();
    for refused in [key(4), digest, private, PrefixCacheKeyV1::new([5; 32])] {
        assert!(tier.begin_offload(&mut pool, refused, None).is_err());
    }
    assert_eq!(pool, before);
    demote(&mut pool, &mut tier, key(8));
    // One entry per key, and one tier per pool.
    assert!(tier.begin_offload(&mut pool, key(8), None).is_err());
    let mut other = HostTier::new(&pool, 32).unwrap();
    assert!(pool.remove_anchor(key(8)).unwrap());
    assert!(tier.drop_public_entry(&mut pool, key(8)).unwrap());
    capture(&mut pool, &prompt, &[8]);
    assert!(other.begin_offload(&mut pool, key(8), None).is_err());
    other.audit(&pool).unwrap();
    assert_idle(&pool, &tier);
}
