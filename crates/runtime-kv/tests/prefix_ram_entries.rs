//! The pool's host entries for the RAM prefix tier: demotion, per-tier
//! residency in the ancestry index, and the logical events that make an
//! entry unusable before its host tier drops it.
use std::sync::Arc;

use runtime_kv::{
    HostAnchorId, HostTier, KvError, KvGeometry, KvPool, KvTransferClass, KvTransferCompletion,
    KvTransferPlan, KvTransferStatus, PREFIX_EVICTION_LOG_V1, PrefixAnchorClassV1,
    PrefixCacheKeyV1, PrefixChainV1, PrefixDemotionCandidateV1, PrefixEvictionTierV1,
    PrefixEvictionV1, PrefixHostInvalidationV1,
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

#[test]
fn public_drops_refuse_a_tier_the_entries_are_not_bound_to() {
    use PrefixHostInvalidationV1::Resumed;
    let prompt = chain(&(0..8).collect::<Vec<_>>());
    let key = prompt.key_at(8).unwrap();
    for case in ["drop", "drop invalidated"] {
        let mut pool = ram_pool(32);
        let mut tier = HostTier::new(&pool, 16).unwrap();
        let mut other = HostTier::new(&pool, 16).unwrap();
        capture(&mut pool, &prompt, &[8]);
        let public = demote(&mut pool, &mut tier, key);
        // The other tier's first anchor, an owned swap, takes the same id.
        let source = pool.create_sequence(8).unwrap();
        pool.append(source, 8).unwrap();
        let private = PrefixCacheKeyV1::new([9; 32]);
        pool.insert_private_anchor(private, source, 1)
            .unwrap()
            .unwrap();
        pool.release(source).unwrap();
        let (swap, plan) = other.begin_offload(&mut pool, private, Some(1)).unwrap();
        other.complete_transfer(&mut pool, completed(plan)).unwrap();
        assert_eq!(swap, public);
        // A resume leaves the public entry invalidated.
        let hit = pool.resume_anchor(key, 8).unwrap();
        pool.finish_restore(hit.handle()).unwrap();
        pool.release(hit.handle()).unwrap();
        let before = pool.clone();
        if case == "drop" {
            assert_eq!(
                other.drop_public_entry(&mut pool, key),
                Err(KvError::StaleHandle)
            );
        } else {
            assert_eq!(
                other.drop_invalidated_entries(&mut pool),
                Err(KvError::StaleHandle)
            );
        }
        assert_eq!(pool, before, "{case}");
        assert!(
            other
                .anchor(swap)
                .is_some_and(|anchor| anchor.owner == Some(1) && anchor.ready),
            "{case}"
        );
        other.audit(&pool).unwrap();
        tier.audit(&pool).unwrap();
        // The bound tier still drops its own entry, and the swap stays.
        assert_eq!(
            tier.drop_invalidated_entries(&mut pool).unwrap(),
            [(key, Resumed)]
        );
        assert!(other.anchor(swap).is_some());
        other.drop_anchor(swap).unwrap();
        assert_idle(&pool, &tier);
        assert_idle(&pool, &other);
    }
}

#[test]
fn configuration_keeps_an_index_the_host_tier_holds() {
    type Configure = fn(KvPool) -> KvPool;
    let builders: [(&str, Configure); 4] = [
        ("index", |pool| pool.with_prefix_index(true)),
        ("ancestry", |pool| pool.with_prefix_ancestry(true)),
        ("ancestry off", |pool| pool.with_prefix_ancestry(false)),
        ("ram", |pool| pool.with_prefix_ram(true)),
    ];
    let refused = |pool: &KvPool, state: &str| {
        for (name, configure) in builders {
            let copy = pool.clone();
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| configure(copy))).is_err(),
                "{state}: {name}"
            );
        }
    };
    let prompt = chain(&[1; 8]);
    let key = prompt.key_at(8).unwrap();
    let mut pool = ram_pool(32);
    let mut tier = HostTier::new(&pool, 32).unwrap();
    // A host-only key leaves no GPU anchor for the capture guard to see.
    capture(&mut pool, &prompt, &[8]);
    demote(&mut pool, &mut tier, key);
    assert!(pool.remove_anchor(key).unwrap());
    refused(&pool, "host only");
    assert!(tier.drop_public_entry(&mut pool, key).unwrap());
    // So does a demotion whose entry was dropped during its copy.
    capture(&mut pool, &prompt, &[8]);
    let (_, plan) = tier.begin_offload(&mut pool, key, None).unwrap();
    assert!(tier.drop_public_entry(&mut pool, key).unwrap());
    assert!(pool.remove_anchor(key).unwrap());
    refused(&pool, "demotion");
    tier.complete_transfer(&mut pool, completed(plan)).unwrap();
    assert_idle(&pool, &tier);
    // Once the tier holds nothing, configuration replaces the index.
    let pool = pool.with_prefix_index(true).with_prefix_ancestry(true);
    assert_idle(&pool, &tier);
}

#[test]
fn demotion_candidates_are_frontiers_in_eviction_order() {
    let interior = chain(&(0..8).collect::<Vec<_>>());
    let (short, plain, shadowed, copying) = (
        chain(&[2; 4]),
        chain(&[3; 8]),
        chain(&[4; 8]),
        chain(&[5; 8]),
    );
    let mut pool = ram_pool(48);
    let mut tier = HostTier::new(&pool, 32).unwrap();
    // Oldest first: an ancestor, its frontier, a short frontier, a plain
    // frontier, a shadow, a copy in flight, then digest-only and private
    // anchors.
    capture(&mut pool, &interior, &[4, 8]);
    capture(&mut pool, &short, &[4]);
    capture(&mut pool, &plain, &[8]);
    capture(&mut pool, &shadowed, &[8]);
    capture(&mut pool, &copying, &[8]);
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
    demote(&mut pool, &mut tier, shadowed.key_at(8).unwrap());
    let (_, plan) = tier
        .begin_offload(&mut pool, copying.key_at(8).unwrap(), None)
        .unwrap();
    let before = pool.clone();
    let candidate = |chain: &PrefixChainV1, tokens, freeable_pages| PrefixDemotionCandidateV1 {
        key: chain.key_at(tokens).unwrap(),
        tokens,
        freeable_pages,
    };
    // The frontier shares its first page with the ancestor below it.
    let all = [candidate(&interior, 8, 2), candidate(&plain, 8, 3)];
    assert_eq!(pool.demotion_candidates(usize::MAX, 8), all);
    assert_eq!(
        pool.demotion_candidates(usize::MAX, 4),
        [all[0], candidate(&short, 4, 2), all[1]]
    );
    // The ancestor frees only its snapshot page, a lead of one page; the
    // frontier after it reaches a lead of three.
    assert!(pool.demotion_candidates(1, 8).is_empty());
    assert_eq!(pool.demotion_candidates(3, 8), all[..1]);
    assert!(pool.demotion_candidates(0, 8).is_empty());
    assert_eq!(pool, before);
    tier.complete_transfer(&mut pool, completed(plan)).unwrap();
    tier.audit(&pool).unwrap();
    // Without the tier nothing is a candidate.
    let mut off = KvPool::new(KvGeometry::new(4, 16).unwrap(), 2)
        .unwrap()
        .with_prefix_index(true);
    capture(&mut off, &plain, &[8]);
    assert!(off.demotion_candidates(usize::MAX, 1).is_empty());
}

#[test]
fn host_lookup_matches_ready_usable_entries_of_the_prompt() {
    let prompt = chain(&(0..16).collect::<Vec<_>>());
    let key = |depth| prompt.key_at(depth).unwrap();
    // Forced mode keeps an interior entry beside a deeper one, so one prompt
    // matches two entries.
    let mut pool = ram_pool(48).with_prefix_ram_forced(true);
    let mut tier = HostTier::new(&pool, 32).unwrap();
    for depth in [6, 12] {
        capture(&mut pool, &prompt, &[depth]);
        demote(&mut pool, &mut tier, key(depth));
        assert!(pool.remove_anchor(key(depth)).unwrap());
    }
    assert!(pool.lookup_anchor(&prompt, 16).is_none());
    assert_eq!(pool.lookup_host_anchor(&prompt, 15), Some((key(12), 12)));
    assert_eq!(pool.lookup_host_anchor(&prompt, 11), Some((key(6), 6)));
    assert_eq!(pool.lookup_host_anchor(&prompt, 5), None);
    // The same full blocks with a different partial tail are another key.
    let diverged = chain(&[0, 1, 2, 3, 4, 99, 6, 7]);
    assert_eq!(pool.lookup_host_anchor(&diverged, 8), None);
    let geometry = PrefixChainV1::new(8, Arc::from(prompt.tokens()), Vec::new()).unwrap();
    assert_eq!(pool.lookup_host_anchor(&geometry, 15), None);
    // An unusable entry is skipped for a shorter usable one.
    capture(&mut pool, &prompt, &[12]);
    assert_eq!(
        pool.host_invalidation(key(12)),
        Some(PrefixHostInvalidationV1::Recaptured)
    );
    assert_eq!(pool.lookup_host_anchor(&prompt, 15), Some((key(6), 6)));
    tier.drop_invalidated_entries(&mut pool).unwrap();
    assert!(pool.remove_anchor(key(12)).unwrap());
    // So is an entry whose copy is still in flight.
    capture(&mut pool, &prompt, &[12]);
    let (_, plan) = tier.begin_offload(&mut pool, key(12), None).unwrap();
    assert_eq!(pool.lookup_host_anchor(&prompt, 15), Some((key(6), 6)));
    tier.complete_transfer(&mut pool, completed(plan)).unwrap();
    assert_eq!(pool.lookup_host_anchor(&prompt, 15), Some((key(12), 12)));
    tier.audit(&pool).unwrap();
}

#[test]
fn demotion_lookup_matches_usable_entries_whose_copy_is_in_flight() {
    let prompt = chain(&(0..16).collect::<Vec<_>>());
    let key = |depth| prompt.key_at(depth).unwrap();
    let mut pool = ram_pool(48).with_prefix_ram_forced(true);
    let mut tier = HostTier::new(&pool, 32).unwrap();
    capture(&mut pool, &prompt, &[6]);
    demote(&mut pool, &mut tier, key(6));
    assert!(pool.remove_anchor(key(6)).unwrap());
    assert_eq!(pool.lookup_host_demotion(&prompt, 15), None);
    capture(&mut pool, &prompt, &[12]);
    let (_, plan) = tier.begin_offload(&mut pool, key(12), None).unwrap();
    assert!(pool.remove_anchor(key(12)).unwrap());
    assert_eq!(pool.lookup_host_demotion(&prompt, 15), Some((key(12), 12)));
    assert_eq!(pool.lookup_host_demotion(&prompt, 11), None);
    assert_eq!(pool.lookup_host_anchor(&prompt, 15), Some((key(6), 6)));
    // A recapture makes the entry in flight unusable.
    capture(&mut pool, &prompt, &[12]);
    assert_eq!(pool.lookup_host_demotion(&prompt, 15), None);
    assert!(pool.remove_anchor(key(12)).unwrap());
    tier.complete_transfer(&mut pool, completed(plan)).unwrap();
    tier.drop_invalidated_entries(&mut pool).unwrap();
    assert_eq!(pool.lookup_host_demotion(&prompt, 15), None);
    assert_eq!(pool.lookup_host_anchor(&prompt, 15), Some((key(6), 6)));
    tier.audit(&pool).unwrap();
}

#[test]
fn eviction_skips_the_demotion_in_flight_and_logs_public_victims() {
    let prompts: Vec<_> = (1..=3).map(|n| chain(&[n; 8])).collect();
    let keys: Vec<_> = prompts.iter().map(|c| c.key_at(8).unwrap()).collect();
    // Three anchors of three pages fill nine of ten pages.
    let mut pool = ram_pool(10);
    let mut tier = HostTier::new(&pool, 32).unwrap();
    for prompt in &prompts {
        capture(&mut pool, prompt, &[8]);
    }
    let private = PrefixCacheKeyV1::new([9; 32]);
    let (_, plan) = tier.begin_offload(&mut pool, keys[0], None).unwrap();
    // Two pages of pressure skip the oldest anchor, whose leased pages
    // eviction cannot free, and take the next.
    let pressure = |pool: &mut KvPool, tokens| {
        let handle = pool.create_sequence(tokens).unwrap();
        let result = pool.append(handle, tokens);
        pool.release(handle).unwrap();
        result.map(|_| ())
    };
    pressure(&mut pool, 8).unwrap();
    assert!(pool.has_anchor(keys[0]) && !pool.has_anchor(keys[1]));
    let record = |key, covered| PrefixEvictionV1 {
        key,
        tokens: 8,
        covered,
        frontier: true,
    };
    assert_eq!(
        pool.drain_prefix_evictions(),
        (vec![record(keys[1], false)], 0)
    );
    // When the copy is the only victim left, allocation fails instead.
    pressure(&mut pool, 20).unwrap();
    assert!(!pool.has_anchor(keys[2]));
    assert!(matches!(
        pressure(&mut pool, 40),
        Err(KvError::PoolExhausted { .. })
    ));
    assert!(pool.has_anchor(keys[0]));
    tier.complete_transfer(&mut pool, completed(plan)).unwrap();
    // A covered eviction is logged as such; private evictions are not.
    let source = pool.create_sequence(4).unwrap();
    pool.append(source, 4).unwrap();
    pool.insert_private_anchor(private, source, 1)
        .unwrap()
        .unwrap();
    pool.release(source).unwrap();
    pressure(&mut pool, 40).unwrap();
    assert!(!pool.has_anchor(keys[0]) && !pool.has_anchor(private));
    assert_eq!(
        pool.drain_prefix_evictions(),
        (vec![record(keys[2], false), record(keys[0], true)], 0)
    );
    tier.audit(&pool).unwrap();
}

#[test]
fn the_eviction_log_keeps_its_newest_records() {
    // Each capture of four tokens and one snapshot page evicts the last.
    let mut pool = ram_pool(3);
    let total = PREFIX_EVICTION_LOG_V1 + 3;
    for n in 0..=total as u32 {
        capture(&mut pool, &chain(&[n; 4]), &[4]);
    }
    let (evictions, discarded) = pool.drain_prefix_evictions();
    assert_eq!((evictions.len(), discarded), (PREFIX_EVICTION_LOG_V1, 3));
    assert_eq!(
        evictions.last().map(|record| record.key),
        chain(&[total as u32 - 1; 4]).key_at(4)
    );
    assert_eq!(pool.drain_prefix_evictions(), (Vec::new(), 0));
    pool.audit().unwrap();
}

#[test]
fn host_tier_evicts_public_entries_by_last_gpu_use_after_a_fit_check() {
    let prompts: Vec<_> = (1..=6).map(|n| chain(&[n; 8])).collect();
    let keys: Vec<_> = prompts.iter().map(|c| c.key_at(8).unwrap()).collect();
    let big = chain(&[7; 28]);
    let mut pool = ram_pool(64);
    // Room for three anchors of three pages.
    let mut tier = HostTier::new(&pool, 9).unwrap();
    for prompt in &prompts {
        capture(&mut pool, prompt, &[8]);
    }
    capture(&mut pool, &big, &[28]);
    let swap = |pool: &mut KvPool, tier: &mut HostTier, owner: u64, tokens: usize| {
        let key = PrefixCacheKeyV1::new([owner as u8; 32]);
        let source = pool.create_sequence(tokens).unwrap();
        pool.append(source, tokens).unwrap();
        pool.insert_private_anchor(key, source, 1).unwrap().unwrap();
        pool.release(source).unwrap();
        let result = tier.begin_offload(pool, key, Some(owner));
        assert!(pool.remove_anchor(key).unwrap());
        result.map(|(_, plan)| tier.complete_transfer(pool, completed(plan)).unwrap())
    };
    let held = |pool: &KvPool| {
        keys.iter()
            .map(|key| pool.host_entry(*key).is_some())
            .collect::<Vec<_>>()
    };
    // The demotion order differs from the order of last GPU use.
    for index in [2, 1, 3] {
        demote(&mut pool, &mut tier, keys[index]);
    }
    // A newer demotion evicts the entry used longest ago, not the first
    // demoted.
    demote(&mut pool, &mut tier, keys[4]);
    assert_eq!(held(&pool), [false, false, true, true, true, false]);
    // A demotion older than every entry evicts nothing and fails.
    assert!(matches!(
        tier.begin_offload(&mut pool, keys[0], None),
        Err(KvError::PoolExhausted { .. })
    ));
    assert_eq!(held(&pool), [false, false, true, true, true, false]);
    // A copy in flight holds its pages, so a demotion that cannot fit
    // without them evicts nothing.
    let (_, plan) = tier.begin_offload(&mut pool, keys[5], None).unwrap();
    assert_eq!(held(&pool), [false, false, false, true, true, true]);
    assert!(
        tier.begin_offload(&mut pool, big.key_at(28).unwrap(), None)
            .is_err()
    );
    assert_eq!(held(&pool), [false, false, false, true, true, true]);
    tier.complete_transfer(&mut pool, completed(plan)).unwrap();
    tier.audit(&pool).unwrap();
    // A swap evicts the oldest entries whatever its own age, but only when
    // that makes it fit.
    swap(&mut pool, &mut tier, 1, 8).unwrap();
    assert_eq!(held(&pool), [false, false, false, false, true, true]);
    assert!(swap(&mut pool, &mut tier, 2, 28).is_err());
    assert_eq!(held(&pool), [false, false, false, false, true, true]);
    tier.audit(&pool).unwrap();
}

#[test]
fn the_allocation_counter_adds_every_page_taken_from_the_free_list() {
    let mut pool = ram_pool(32);
    let mut tier = HostTier::new(&pool, 32).unwrap();
    let taken = |pool: &KvPool, since: u64| pool.allocated_pages() - since;
    let source = pool.create_sequence(12).unwrap();
    pool.append(source, 6).unwrap();
    assert_eq!(taken(&pool, 0), 2);
    // A fork's first append copies the shared partial page.
    let child = pool.fork(source, 12).unwrap();
    let mark = pool.allocated_pages();
    pool.append(child, 1).unwrap();
    assert_eq!(taken(&pool, mark), 1);
    pool.release(child).unwrap();
    // A rolled-back append took its page all the same, and so does the
    // retry: the count is gross.
    let mark = pool.allocated_pages();
    let prepared = pool.prepare_append_batch(&[(source, 6)]).unwrap();
    assert_eq!(taken(&pool, mark), 1);
    pool.rollback_prepared(prepared.id()).unwrap();
    assert_eq!(taken(&pool, mark), 1);
    let prepared = pool.prepare_append_batch(&[(source, 6)]).unwrap();
    assert_eq!(taken(&pool, mark), 2);
    pool.rollback_prepared(prepared.id()).unwrap();
    assert_eq!(taken(&pool, mark), 2);
    // A snapshot takes its state page; a capture of a key that has an
    // anchor takes none.
    pool.append(source, 6).unwrap();
    let prompt = chain(&(0..12).collect::<Vec<_>>());
    let key = prompt.key_at(8).unwrap();
    let mark = pool.allocated_pages();
    pool.insert_content_anchor(&prompt, source, 8, 1)
        .unwrap()
        .unwrap();
    assert_eq!(taken(&pool, mark), 1);
    let serial = pool.anchor_serial(key);
    assert!(serial.is_some());
    assert!(
        pool.insert_content_anchor(&prompt, source, 8, 1)
            .unwrap()
            .is_none()
    );
    assert_eq!(taken(&pool, mark), 1);
    assert_eq!(pool.anchor_serial(key), serial);
    pool.release(source).unwrap();
    // Restore staging and a pending anchor take theirs.
    let mark = pool.allocated_pages();
    let staged = pool.create_initialized_sequence(8, 12, 1).unwrap();
    assert_eq!(taken(&pool, mark), 3);
    pool.release(staged.handle()).unwrap();
    demote(&mut pool, &mut tier, key);
    assert!(pool.remove_anchor(key).unwrap());
    assert_eq!(pool.anchor_serial(key), None);
    let mark = pool.allocated_pages();
    let pending = pool.create_pending_anchor(key, 1).unwrap();
    assert_eq!(taken(&pool, mark), 3);
    pool.abandon_pending_anchor(pending).unwrap();
    // A later capture of the key is another anchor lifetime.
    capture(&mut pool, &prompt, &[8]);
    assert!(
        pool.anchor_serial(key)
            .is_some_and(|later| Some(later) != serial)
    );
    tier.audit(&pool).unwrap();
}

#[test]
fn the_eviction_counter_adds_the_pages_each_eviction_frees() {
    let mut pool = ram_pool(8);
    let tier = HostTier::new(&pool, 8).unwrap();
    let prompt = chain(&(0..8).collect::<Vec<_>>());
    capture(&mut pool, &prompt, &[8]);
    let key = prompt.key_at(8).unwrap();
    assert!(tier.demotable(&pool, key));
    assert_eq!(pool.evicted_pages(), 0);
    // Releasing a sequence frees pages without an eviction.
    let other = pool.create_sequence(4).unwrap();
    pool.append(other, 4).unwrap();
    pool.release(other).unwrap();
    assert_eq!(pool.evicted_pages(), 0);
    // A sequence that needs every page evicts the anchor: two prompt pages
    // and its snapshot.
    let filler = pool.create_sequence(32).unwrap();
    pool.append(filler, 32).unwrap();
    assert_eq!(pool.evicted_pages(), 3);
    assert!(!tier.demotable(&pool, key));
    pool.release(filler).unwrap();
    assert_eq!(pool.evicted_pages(), 3);
}

#[test]
fn host_evictions_for_room_count_in_the_pool_metrics() {
    let mut pool = ram_pool(32);
    // Room for one three-page entry and a little more.
    let mut tier = HostTier::new(&pool, 5).unwrap();
    let first = chain(&(0..8).collect::<Vec<_>>());
    let second = chain(&(100..108).collect::<Vec<_>>());
    capture(&mut pool, &first, &[8]);
    demote(&mut pool, &mut tier, first.key_at(8).unwrap());
    assert_eq!(tier.public_entries(), (1, 3));
    assert_eq!(pool.metrics().snapshot().prefix_ram_evictions, 0);
    // The second demotion evicts the older entry for room.
    capture(&mut pool, &second, &[8]);
    demote(&mut pool, &mut tier, second.key_at(8).unwrap());
    assert_eq!(tier.public_entries(), (1, 3));
    assert!(pool.host_entry(first.key_at(8).unwrap()).is_none());
    let snapshot = pool.metrics().snapshot();
    assert_eq!(snapshot.prefix_ram_evictions, 1);
    assert!(
        snapshot
            .prometheus()
            .contains("\nchop_prefix_ram_evictions_total 1\n")
    );
}

#[test]
fn anchors_a_live_sequence_references_are_no_demotion_candidates() {
    let prompt = chain(&(0..8).collect::<Vec<_>>());
    let key = prompt.key_at(8).unwrap();
    let keys = |pool: &KvPool| {
        pool.demotion_candidates(usize::MAX, 1)
            .iter()
            .map(|candidate| candidate.key)
            .collect::<Vec<_>>()
    };
    for forced in [false, true] {
        let mut pool = ram_pool(32).with_prefix_ram_forced(forced);
        // An anchor captured from a live sequence shares its pages: copying
        // it would lease them all to free one snapshot page. Forced mode
        // keeps its candidates.
        let source = pool.create_sequence(12).unwrap();
        pool.append(source, 8).unwrap();
        pool.insert_content_anchor(&prompt, source, 8, 1)
            .unwrap()
            .unwrap();
        assert_eq!(keys(&pool), if forced { vec![key] } else { vec![] });
        pool.release(source).unwrap();
        assert_eq!(keys(&pool), [key]);
        // So does a sequence that resumed from it.
        let longer = chain(&(0..12).collect::<Vec<_>>());
        let resumed = pool.resume_prefix(&longer, 11, 12).unwrap().unwrap();
        pool.finish_restore(resumed.handle()).unwrap();
        assert_eq!(keys(&pool), if forced { vec![key] } else { vec![] });
        pool.release(resumed.handle()).unwrap();
        assert_eq!(keys(&pool), [key]);
    }
    // Pages shared between cached anchors do not exclude one: the end
    // anchor shares its first page with the interior anchor.
    let mut pool = ram_pool(32);
    capture(&mut pool, &prompt, &[4, 8]);
    assert_eq!(keys(&pool), [key]);
}
