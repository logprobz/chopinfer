use runtime_kv::*;
fn completed(plan: KvTransferPlan) -> KvTransferCompletion {
    plan.into_submission()
        .completion(KvTransferStatus::Complete)
}

fn key(n: u64) -> PrefixCacheKeyV1 {
    let mut bytes = [0; 32];
    bytes[..8].copy_from_slice(&n.to_le_bytes());
    PrefixCacheKeyV1::new(bytes)
}
fn pool() -> KvPool {
    KvPool::new(KvGeometry::new(4, 32).unwrap(), 8).unwrap()
}
#[test]
fn transfers_pin_both_tiers_through_cancellation_and_eviction() {
    let mut pool = pool();
    let mut host = HostTier::new(&pool, 16).unwrap();
    for n in 0..2000 {
        let seq = pool.create_sequence(20).unwrap();
        pool.append(seq, 9).unwrap();
        pool.insert_private_anchor(key(n), seq, 2).unwrap().unwrap();
        let (anchor, transfer) = host.begin_offload(&mut pool, key(n), Some(n)).unwrap();
        pool.release(seq).unwrap();
        pool.remove_anchor(key(n)).unwrap();
        host.audit(&pool).unwrap();
        assert_eq!(pool.stats().used_blocks(), 5);
        if n % 3 == 0 {
            host.drop_anchor(anchor).unwrap();
        }
        let source = transfer.into_submission();
        host.complete_transfer(&mut pool, source.completion(KvTransferStatus::Complete))
            .unwrap();
        assert!(
            host.complete_transfer(&mut pool, source.completion(KvTransferStatus::Complete))
                .is_err()
        );
        assert_eq!(pool.stats().free_blocks(), 32);
        if n % 3 != 0 {
            let dest = pool.create_initialized_sequence(9, 20, 2).unwrap();
            let copy = host.begin_reload(&mut pool, anchor, &dest).unwrap();
            if n % 3 == 1 {
                pool.release(dest.handle()).unwrap();
            }
            host.drop_anchor(anchor).unwrap();
            host.audit(&pool).unwrap();
            assert_eq!(host.used_pages(), 5);
            host.complete_transfer(&mut pool, completed(copy)).unwrap();
            if n % 3 == 2 {
                pool.finish_initialized_restore(dest.handle()).unwrap();
                let append = pool.prepare_append_batch(&[(dest.handle(), 1)]).unwrap();
                assert!(append.appends()[0].newly_allocated().is_empty());
                pool.rollback_prepared(append.id()).unwrap();
                pool.release(dest.handle()).unwrap();
            }
        }
        host.audit(&pool).unwrap();
        assert_eq!(host.used_pages(), 0);
        assert_eq!(pool.stats().free_blocks(), 32);
    }
}
#[test]
fn serials_reject_stale_leases_and_reused_pages_do_not_share() {
    let mut p = pool();
    let mut h = HostTier::new(&p, 16).unwrap();
    let a = p.create_sequence(8).unwrap();
    p.append(a, 4).unwrap();
    let old = p
        .page_allocation(p.snapshot(a).unwrap().blocks()[0])
        .unwrap();
    p.insert_private_anchor(key(1), a, 0).unwrap();
    let (ha, t) = h.begin_offload(&mut p, key(1), Some(1)).unwrap();
    h.complete_transfer(&mut p, completed(t)).unwrap();
    p.remove_anchor(key(1)).unwrap();
    p.release(a).unwrap();
    let b = p.create_sequence(8).unwrap();
    p.append(b, 4).unwrap();
    let new = p
        .page_allocation(p.snapshot(b).unwrap().blocks()[0])
        .unwrap();
    assert_eq!(old.page, new.page);
    assert_ne!(old.serial, new.serial);
    assert!(p.retain_transfer_pages(&[old]).is_err());
    p.insert_private_anchor(key(2), b, 0).unwrap();
    let (hb, t) = h.begin_offload(&mut p, key(2), Some(2)).unwrap();
    assert_eq!(t.pages().len(), 1);
    h.complete_transfer(&mut p, completed(t)).unwrap();
    h.drop_anchor(ha).unwrap();
    h.drop_anchor(hb).unwrap();
    p.remove_anchor(key(2)).unwrap();
    p.release(b).unwrap();
    h.audit(&p).unwrap();
}
#[test]
fn captures_of_shared_device_pages_copy_into_their_own_host_pages() {
    let mut p = pool();
    let mut h = HostTier::new(&p, 16).unwrap();
    let a = p.create_sequence(12).unwrap();
    p.append(a, 5).unwrap();
    let b = p.fork(a, 12).unwrap();
    p.insert_private_anchor(key(1), a, 1).unwrap();
    p.insert_private_anchor(key(2), b, 1).unwrap();
    let (ha, t) = h.begin_offload(&mut p, key(1), Some(1)).unwrap();
    h.complete_transfer(&mut p, completed(t)).unwrap();
    let (hb, t) = h.begin_offload(&mut p, key(2), Some(2)).unwrap();
    // The full page is shared on the device, yet its host copy is not.
    assert_eq!(t.pages().len(), 3);
    assert_eq!(h.used_pages(), 6);
    assert_ne!(
        h.anchor(ha).unwrap().blocks[0],
        h.anchor(hb).unwrap().blocks[0]
    );
    h.drop_anchor(ha).unwrap();
    h.complete_transfer(&mut p, completed(t)).unwrap();
    h.audit(&p).unwrap();
    h.drop_anchor(hb).unwrap();
    p.remove_anchor(key(1)).unwrap();
    p.remove_anchor(key(2)).unwrap();
    p.release(a).unwrap();
    p.release(b).unwrap();
    h.audit(&p).unwrap();
}
#[test]
fn failures_quarantine_and_foreign_pools_do_not_reuse_pages() {
    let mut p = pool();
    let mut h = HostTier::new(&p, 3).unwrap();
    let seq = p.create_sequence(12).unwrap();
    p.append(seq, 5).unwrap();
    p.insert_private_anchor(key(1), seq, 1).unwrap();
    let (a, t) = h.begin_offload(&mut p, key(1), Some(1)).unwrap();
    assert!(h.begin_offload(&mut p, key(1), Some(2)).is_err());
    let mut foreign = p.clone();
    let source = t.into_submission();
    let (_, retry) = h
        .complete_transfer(
            &mut foreign,
            source.completion(KvTransferStatus::Unconfirmed),
        )
        .unwrap_err();
    h.complete_transfer(&mut p, retry).unwrap();
    h.drop_anchor(a).unwrap();
    p.remove_anchor(key(1)).unwrap();
    p.release(seq).unwrap();
    assert!(
        h.complete_transfer(&mut p, source.completion(KvTransferStatus::FailedIdle))
            .is_err()
    );
    h.audit(&p).unwrap();
    assert_eq!(h.used_pages(), 3);
    assert_eq!(p.stats().free_blocks(), 29);
}
#[test]
fn failed_idle_copies_return_all_pages() {
    let mut p = pool();
    let mut h = HostTier::new(&p, 3).unwrap();
    let seq = p.create_sequence(12).unwrap();
    p.append(seq, 5).unwrap();
    p.insert_private_anchor(key(1), seq, 1).unwrap();
    let (_, t) = h.begin_offload(&mut p, key(1), Some(1)).unwrap();
    p.remove_anchor(key(1)).unwrap();
    p.release(seq).unwrap();
    h.complete_transfer(&mut p, t.cancel()).unwrap();
    h.audit(&p).unwrap();
    assert_eq!(h.used_pages(), 0);
    assert_eq!(p.stats().free_blocks(), 32);
}

#[test]
fn private_anchor_is_invisible_to_prompt_lookup_but_resumes_by_exact_key() {
    let mut p = pool();
    let chain = PrefixChainV1::new(4, std::sync::Arc::from([1, 2, 3, 4, 5]), vec![]).unwrap();
    let seq = p.create_sequence(12).unwrap();
    p.append(seq, 4).unwrap();
    let key = chain.key_at(4).unwrap();
    p.insert_private_anchor(key, seq, 1).unwrap();
    p.release(seq).unwrap();
    assert!(p.lookup_anchor(&chain, 4).is_none());
    assert!(p.resume_prefix(&chain, 4, 12).unwrap().is_none());
    let restored = p.resume_anchor(key, 12).unwrap();
    assert_eq!(restored.tokens(), 4);
    p.remove_anchor(key).unwrap();
    p.finish_initialized_restore(restored.handle()).unwrap();
    p.release(restored.handle()).unwrap();
    p.audit().unwrap();
    assert_eq!(p.stats().free_blocks(), 32);
}

#[test]
fn unowned_host_anchors_use_reload_recency_and_never_evict_owned_pages() {
    let mut p = pool();
    let mut h = HostTier::new(&p, 2).unwrap();
    let capture = |p: &mut KvPool, h: &mut HostTier, n, owner| {
        let seq = p.create_sequence(8).unwrap();
        p.append(seq, 4).unwrap();
        p.insert_private_anchor(key(n), seq, 0).unwrap();
        let result = h.begin_offload(p, key(n), owner);
        p.remove_anchor(key(n)).unwrap();
        p.release(seq).unwrap();
        result
    };
    let (a, t) = capture(&mut p, &mut h, 1, None).unwrap();
    h.complete_transfer(&mut p, completed(t)).unwrap();
    let (b, t) = capture(&mut p, &mut h, 2, None).unwrap();
    h.complete_transfer(&mut p, completed(t)).unwrap();
    let dest = p.create_initialized_sequence(4, 8, 0).unwrap();
    let t = h.begin_reload(&mut p, a, &dest).unwrap();
    h.complete_transfer(&mut p, completed(t)).unwrap();
    p.release(dest.handle()).unwrap();
    let (c, t) = capture(&mut p, &mut h, 3, Some(3)).unwrap();
    h.complete_transfer(&mut p, completed(t)).unwrap();
    assert!(h.anchor(a).is_some());
    assert!(h.anchor(b).is_none());
    let (d, t) = capture(&mut p, &mut h, 4, Some(4)).unwrap();
    h.complete_transfer(&mut p, completed(t)).unwrap();
    assert!(h.anchor(a).is_none());
    assert!(capture(&mut p, &mut h, 5, Some(5)).is_err());
    h.drop_anchor(c).unwrap();
    h.drop_anchor(d).unwrap();
    h.audit(&p).unwrap();
    assert_eq!(h.used_pages(), 0);
    assert_eq!(p.stats().free_blocks(), 32);
}

#[test]
fn completion_refusal_preserves_token_and_reload_cannot_be_batched_early() {
    let mut p = pool();
    let mut h = HostTier::new(&p, 16).unwrap();
    let seq = p.create_sequence(20).unwrap();
    p.append(seq, 5).unwrap();
    p.insert_private_anchor(key(7), seq, 1).unwrap();
    let (a, out) = h.begin_offload(&mut p, key(7), Some(7)).unwrap();
    let source = out.into_submission();
    let prepared = p.prepare_append_batch(&[(seq, 1)]).unwrap();
    let (_, token) = h
        .complete_transfer(&mut p, source.completion(KvTransferStatus::Complete))
        .unwrap_err();
    assert!(!source.accepted());
    p.rollback_prepared(prepared.id()).unwrap();
    h.complete_transfer(&mut p, token).unwrap();
    assert!(source.accepted());
    let dest = p.create_initialized_sequence(5, 20, 1).unwrap();
    assert!(p.append(dest.handle(), 1).is_err());
    assert!(p.fork(dest.handle(), 20).is_err());
    assert!(p.finish_initialized_restore(dest.handle()).is_err());
    assert!(p.finish_restore(dest.handle()).is_err());
    let reload = h.begin_reload(&mut p, a, &dest).unwrap();
    h.complete_transfer(&mut p, completed(reload)).unwrap();
    assert!(p.append(dest.handle(), 1).is_err());
    p.finish_initialized_restore(dest.handle()).unwrap();
    p.append(dest.handle(), 1).unwrap();
    p.release(seq).unwrap();
    p.remove_anchor(key(7)).unwrap();
    p.release(dest.handle()).unwrap();
    h.drop_anchor(a).unwrap();
    h.audit(&p).unwrap();
    assert_eq!(p.stats().free_blocks(), 32);
}

#[test]
fn different_tiers_never_issue_the_same_transfer_id() {
    let mut p = pool();
    let seq = p.create_sequence(8).unwrap();
    p.append(seq, 4).unwrap();
    p.insert_private_anchor(key(1), seq, 0).unwrap();
    let mut h1 = HostTier::new(&p, 8).unwrap();
    let mut h2 = HostTier::new(&p, 8).unwrap();
    let (_, a) = h1.begin_offload(&mut p, key(1), None).unwrap();
    let (_, b) = h2.begin_offload(&mut p, key(1), None).unwrap();
    assert_ne!(a.id(), b.id());
    let (_, token) = h1.complete_transfer(&mut p, b.cancel()).unwrap_err();
    h2.complete_transfer(&mut p, token).unwrap();
    h1.complete_transfer(&mut p, a.cancel()).unwrap();
    p.remove_anchor(key(1)).unwrap();
    p.release(seq).unwrap();
    h1.audit(&p).unwrap();
    h2.audit(&p).unwrap();
    assert_eq!(p.stats().free_blocks(), 32);
}

#[test]
fn retained_host_copy_does_not_survive_a_mutable_page_rewrite() {
    use std::collections::BTreeMap;

    // A shortened GPU anchor also makes a former interior page a mutable tail.
    for initial_tokens in [4, 8] {
        let mut p = pool().with_last_page_rewrites(true);
        let mut h = HostTier::new(&p, 16).unwrap();
        let source = p.create_sequence(12).unwrap();
        p.append(source, initial_tokens).unwrap();
        p.insert_private_anchor(key(1), source, 0).unwrap().unwrap();
        p.insert_prefix_anchor(key(2), key(2), source, 4, 0)
            .unwrap()
            .unwrap();
        let original = p.snapshot(source).unwrap().blocks()[0];
        let allocation = p.page_allocation(original).unwrap();
        let mut device = BTreeMap::new();
        for page in p.snapshot(source).unwrap().blocks() {
            device.insert(*page, [1_u8; 32]);
        }
        let mut host = BTreeMap::new();
        let (old, transfer) = h.begin_offload(&mut p, key(1), None).unwrap();
        for pair in transfer.pages() {
            host.insert(pair.host, device[&pair.device]);
        }
        h.complete_transfer(&mut p, completed(transfer)).unwrap();
        p.release(source).unwrap();
        p.remove_anchor(key(1)).unwrap();
        let resumed = p.resume_anchor(key(2), 12).unwrap();
        p.finish_owned_restore(resumed.handle()).unwrap();
        p.remove_anchor(key(2)).unwrap();
        assert_eq!(p.block_ref_count(original).unwrap(), 1);

        let append = p.prepare_append_batch(&[(resumed.handle(), 1)]).unwrap();
        assert!(append.appends()[0].copy().is_none());
        assert_eq!(p.page_allocation(original).unwrap(), allocation);
        // The bundle may rewrite committed tail rows before appending.
        device.insert(original, [2; 32]);
        for page in append.appends()[0].newly_allocated() {
            device.insert(*page, [3; 32]);
        }
        p.commit_prepared(append.id()).unwrap();
        p.insert_private_anchor(key(3), resumed.handle(), 0)
            .unwrap()
            .unwrap();
        let (new, transfer) = h.begin_offload(&mut p, key(3), None).unwrap();
        for pair in transfer.pages() {
            host.insert(pair.host, device[&pair.device]);
        }
        h.complete_transfer(&mut p, completed(transfer)).unwrap();
        let expected: Vec<_> = p
            .snapshot(resumed.handle())
            .unwrap()
            .blocks()
            .iter()
            .map(|page| device[page])
            .collect();
        p.release(resumed.handle()).unwrap();
        p.remove_anchor(key(3)).unwrap();
        let destination = p.create_initialized_sequence(5, 12, 0).unwrap();
        let reload = h.begin_reload(&mut p, new, &destination).unwrap();
        for pair in reload.pages() {
            device.insert(pair.device, host[&pair.host]);
        }
        h.complete_transfer(&mut p, completed(reload)).unwrap();
        p.finish_initialized_restore(destination.handle()).unwrap();
        let actual: Vec<_> = p
            .snapshot(destination.handle())
            .unwrap()
            .blocks()
            .iter()
            .map(|page| device[page])
            .collect();
        assert_eq!(
            actual, expected,
            "stale host copy after {initial_tokens}-token capture"
        );
        h.drop_anchor(old).unwrap();
        h.drop_anchor(new).unwrap();
        p.release(destination.handle()).unwrap();
        h.audit(&p).unwrap();
        assert_eq!(h.used_pages(), 0);
    }
}

#[test]
fn retained_host_copy_does_not_survive_an_append_after_a_shorter_prefix() {
    use std::collections::BTreeMap;

    // Without last-page rewrites a full page is still not immutable: a
    // prefix anchor ending inside it resumes with the page as its tail, and
    // the next append writes rows in place under the same allocation.
    for rewrites in [false, true] {
        for initial_tokens in [4, 8] {
            let case = format!("rewrites {rewrites}, {initial_tokens}-token capture");
            let mut p = pool().with_last_page_rewrites(rewrites);
            let mut h = HostTier::new(&p, 16).unwrap();
            let source = p.create_sequence(12).unwrap();
            p.append(source, initial_tokens).unwrap();
            p.insert_private_anchor(key(1), source, 0).unwrap().unwrap();
            p.insert_prefix_anchor(key(2), key(2), source, 2, 0)
                .unwrap()
                .unwrap();
            let original = p.snapshot(source).unwrap().blocks()[0];
            let allocation = p.page_allocation(original).unwrap();
            let mut device = BTreeMap::new();
            for page in p.snapshot(source).unwrap().blocks() {
                device.insert(*page, [1_u8; 32]);
            }
            let mut host = BTreeMap::new();
            let (old, transfer) = h.begin_offload(&mut p, key(1), Some(1)).unwrap();
            for pair in transfer.pages() {
                host.insert(pair.host, device[&pair.device]);
            }
            h.complete_transfer(&mut p, completed(transfer)).unwrap();
            p.release(source).unwrap();
            p.remove_anchor(key(1)).unwrap();
            let resumed = p.resume_anchor(key(2), 12).unwrap();
            p.finish_owned_restore(resumed.handle()).unwrap();
            p.remove_anchor(key(2)).unwrap();
            assert_eq!(p.block_ref_count(original).unwrap(), 1, "{case}");

            let append = p.prepare_append_batch(&[(resumed.handle(), 2)]).unwrap();
            assert!(append.appends()[0].copy().is_none(), "{case}");
            assert_eq!(p.page_allocation(original).unwrap(), allocation, "{case}");
            // Keep the two committed rows; write only the two appended rows.
            device.get_mut(&original).unwrap()[16..].fill(2);
            p.commit_prepared(append.id()).unwrap();
            p.insert_private_anchor(key(3), resumed.handle(), 0)
                .unwrap()
                .unwrap();
            let (new, transfer) = h.begin_offload(&mut p, key(3), Some(2)).unwrap();
            let copied = transfer.pages().len();
            for pair in transfer.pages() {
                host.insert(pair.host, device[&pair.device]);
            }
            h.complete_transfer(&mut p, completed(transfer)).unwrap();
            let expected: Vec<_> = p
                .snapshot(resumed.handle())
                .unwrap()
                .blocks()
                .iter()
                .map(|page| device[page])
                .collect();
            p.release(resumed.handle()).unwrap();
            p.remove_anchor(key(3)).unwrap();
            let destination = p.create_initialized_sequence(4, 12, 0).unwrap();
            let reload = h.begin_reload(&mut p, new, &destination).unwrap();
            for pair in reload.pages() {
                device.insert(pair.device, host[&pair.host]);
            }
            h.complete_transfer(&mut p, completed(reload)).unwrap();
            p.finish_initialized_restore(destination.handle()).unwrap();
            let actual: Vec<_> = p
                .snapshot(destination.handle())
                .unwrap()
                .blocks()
                .iter()
                .map(|page| device[page])
                .collect();
            assert_eq!(actual, expected, "{case}: stale host copy");
            assert_eq!(copied, 1, "{case}: page not copied");
            h.drop_anchor(old).unwrap();
            h.drop_anchor(new).unwrap();
            p.release(destination.handle()).unwrap();
            h.audit(&p).unwrap();
            assert_eq!(h.used_pages(), 0, "{case}");
            assert_eq!(p.stats().free_blocks(), 32, "{case}");
        }
    }
}
