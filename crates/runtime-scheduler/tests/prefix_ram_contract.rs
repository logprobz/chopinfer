//! Scheduler admission for the RAM prefix tier: the GPU match against the
//! ready host match, promotions with their reservations and waiters, the
//! engine entry points, and the per-step allocation bound.

use std::sync::Arc;

use runtime_kv::{
    HostTier, KvError, KvGeometry, KvTransferPlan, KvTransferStatus, PrefixCacheKeyV1,
    PrefixChainV1, PrefixEvictionV1, PrefixHostInvalidationV1, PrefixPublishV1,
};
use runtime_scheduler::{
    AdmissionPolicy, AdmissionReport, BatchOutcome, OffloadConfig, OptimisticAdmission,
    PrefixRamConfig, PrefixRamFallbackV1, PrefixRamHitV1, PrefixRequestV1, RequestId, RequestSpec,
    RequestState, ScheduleError, Scheduler, SchedulerConfig,
};

const BLOCK: usize = 4;
/// Pages of a 17-token anchor: five KV pages and one snapshot page.
const ANCHOR_PAGES: usize = 6;

fn tier(forced: bool, budget: usize) -> Option<PrefixRamConfig> {
    Some(PrefixRamConfig {
        forced,
        promotion_budget_pages: budget,
    })
}

fn offload(minimum_tokens: usize) -> OffloadConfig {
    OffloadConfig {
        minimum_tokens,
        snapshot_pages: 1,
        enabled: true,
        force_host: false,
    }
}

/// Prompt `salt` of `length` tokens; prompts with one salt share prefixes.
fn prompt(salt: u32, length: usize) -> Vec<u32> {
    (0..length as u32)
        .map(|token| salt * 1_000 + token)
        .collect()
}

/// `prompt(salt, base)` followed by `extra` tokens of its own.
fn extended(salt: u32, base: usize, extra: usize, tail: u32) -> Vec<u32> {
    let mut tokens = prompt(salt, base);
    tokens.extend((0..extra as u32).map(|token| 900_000 + tail * 100 + token));
    tokens
}

fn chain(tokens: &[u32]) -> PrefixChainV1 {
    PrefixChainV1::new(BLOCK, Arc::from(tokens), Vec::new()).unwrap()
}

fn key(tokens: &[u32], depth: usize) -> PrefixCacheKeyV1 {
    chain(tokens).key_at(depth).unwrap()
}

struct Harness {
    s: Scheduler,
    tier: HostTier,
}

impl Harness {
    fn new(pages: usize, ram: Option<PrefixRamConfig>) -> Self {
        Self::with_config(pages, ram, |config| config)
    }

    fn with_config(
        pages: usize,
        ram: Option<PrefixRamConfig>,
        configure: impl FnOnce(SchedulerConfig) -> SchedulerConfig,
    ) -> Self {
        let config = SchedulerConfig::new(4, 64, 64, 64, 1)
            .unwrap()
            .with_admission(AdmissionPolicy::Optimistic(
                OptimisticAdmission::new(0, 3).unwrap(),
            ))
            .with_prefix_cache(1)
            .with_prefix_ram(ram);
        let mut s =
            Scheduler::new(configure(config), KvGeometry::new(BLOCK, pages).unwrap()).unwrap();
        s.configure_offload(offload(1));
        let tier = s.new_host_tier(32).unwrap();
        Self { s, tier }
    }

    fn check(&self) {
        self.s.audit().unwrap();
        self.s.audit_host_tier(&self.tier).unwrap();
    }

    /// Submits a request that ends after its first output token.
    fn submit(&mut self, id: u64, tokens: &[u32]) {
        self.submit_spec(
            RequestSpec::new(RequestId::new(id), tokens.len(), 1).unwrap(),
            tokens,
        );
    }

    fn submit_spec(&mut self, spec: RequestSpec, tokens: &[u32]) {
        self.s
            .submit_with_prefix(
                spec,
                Some(PrefixRequestV1::new(Arc::new(chain(tokens)), true)),
            )
            .unwrap();
        self.check();
    }

    fn admit(&mut self) -> AdmissionReport {
        let report = self.s.admit().unwrap();
        self.check();
        report
    }

    fn restore_all(&mut self) {
        for restore in self.s.pending_prefix_restores() {
            self.s
                .complete_prefix_restore(restore.request_id())
                .unwrap();
            self.check();
        }
    }

    /// Runs batches, restoring and capturing, until no request is live.
    /// With `capture` false the captures are skipped, so finished requests
    /// leave no anchor.
    fn finish(&mut self, capture: bool) {
        loop {
            self.admit();
            self.restore_all();
            let Some(batch) = self.s.next_batch().unwrap() else {
                break;
            };
            let outcomes: Vec<_> = batch
                .items()
                .iter()
                .map(|item| {
                    BatchOutcome::success(
                        item.request_id(),
                        item.phase(),
                        usize::from(item.expects_output()),
                        item.expects_output(),
                    )
                })
                .collect();
            self.s.commit_batch(batch.id(), &outcomes).unwrap();
            for capture_due in self.s.take_prefix_captures() {
                if capture {
                    self.s.begin_prefix_capture(&capture_due).unwrap();
                }
                self.s.end_prefix_capture(&capture_due).unwrap();
            }
            self.check();
        }
    }

    /// Runs one request to its end, capturing its anchors.
    fn run(&mut self, id: u64, tokens: &[u32]) {
        self.submit(id, tokens);
        self.finish(true);
    }

    fn complete(&mut self, plan: KvTransferPlan, status: KvTransferStatus) {
        self.s
            .accept_kv_transfer(&mut self.tier, plan.into_submission().completion(status))
            .unwrap();
        self.check();
    }

    fn demote(&mut self, key: PrefixCacheKeyV1) {
        let plan = self.s.begin_demotion(&mut self.tier, key).unwrap();
        self.check();
        self.complete(plan, KvTransferStatus::Complete);
    }

    /// Evicts the least recently used GPU anchor: a cold request takes one
    /// page more than is free, and leaves no anchor of its own.
    fn evict_oldest(&mut self, id: u64) {
        let pages = self.s.kv_stats().free_blocks() + 1;
        self.submit(id, &prompt(500 + id as u32, pages * BLOCK));
        self.finish(false);
    }

    /// Submits each pending promotion's copy and completes it with `status`.
    fn promote(&mut self, status: KvTransferStatus) -> usize {
        let pending = self.s.pending_promotions();
        for promotion in &pending {
            let plan = self
                .s
                .begin_promotion(&mut self.tier, promotion.id)
                .unwrap();
            self.check();
            self.complete(plan, status);
        }
        pending.len()
    }

    fn step(&mut self) -> Vec<(PrefixPublishV1, usize)> {
        let settled = self.s.begin_prefix_ram_step().unwrap();
        self.check();
        settled
            .iter()
            .map(|settled| (settled.outcome, settled.pages))
            .collect()
    }

    fn anchors(&self) -> usize {
        self.s.kv_stats().anchors()
    }
}

/// A 17-token anchor of prompt `salt`, ready in RAM and gone from the GPU.
fn host_only(h: &mut Harness, salt: u32) -> PrefixCacheKeyV1 {
    let tokens = prompt(salt, 17);
    h.run(u64::from(salt), &tokens);
    let key = key(&tokens, 17);
    h.demote(key);
    key
}

#[test]
fn public_captures_are_indexed_for_demotion_only_with_the_tier() {
    for ram in [None, tier(false, 64)] {
        let mut h = Harness::new(16, ram);
        let tokens = prompt(1, 17);
        h.run(1, &tokens);
        let candidates = h.s.prefix_demotion_candidates(usize::MAX);
        assert_eq!(candidates.len(), usize::from(ram.is_some()), "{ram:?}");
        if ram.is_some() {
            assert_eq!(candidates[0].key, key(&tokens, 17));
        } else {
            assert!(h.s.begin_demotion(&mut h.tier, key(&tokens, 17)).is_err());
        }
    }
}

#[test]
fn demotions_stay_outstanding_until_confirmed_and_resumes_drop_their_shadows() {
    for status in [
        KvTransferStatus::Complete,
        KvTransferStatus::FailedIdle,
        KvTransferStatus::Unconfirmed,
    ] {
        let mut h = Harness::new(16, tier(false, 64));
        let tokens = prompt(1, 17);
        h.run(1, &tokens);
        let key = key(&tokens, 17);
        let plan = h.s.begin_demotion(&mut h.tier, key).unwrap();
        assert_eq!(h.s.outstanding_demotions(), 1);
        h.check();
        h.complete(plan, status);
        let unconfirmed = status == KvTransferStatus::Unconfirmed;
        assert_eq!(h.s.outstanding_demotions(), usize::from(unconfirmed));
        if status != KvTransferStatus::Complete {
            continue;
        }
        // A resume of the key's GPU anchor makes its host copy a shadow.
        h.run(2, &extended(1, 17, 7, 0));
        assert_eq!(
            h.s.drop_invalidated_host_entries(&mut h.tier).unwrap(),
            [(key, PrefixHostInvalidationV1::Resumed)]
        );
        assert_eq!(h.tier.used_pages(), 0);
        h.check();
    }
}

#[test]
fn the_eviction_of_a_demoted_anchor_is_logged_as_covered() {
    let mut h = Harness::new(16, tier(false, 64));
    let tokens = prompt(1, 17);
    h.run(1, &tokens);
    h.demote(key(&tokens, 17));
    h.evict_oldest(2);
    let (evictions, discarded) = h.s.drain_prefix_evictions();
    assert_eq!(
        evictions,
        [PrefixEvictionV1 {
            key: key(&tokens, 17),
            tokens: 17,
            covered: true,
            frontier: true,
        }]
    );
    assert_eq!(discarded, 0);
    h.check();
}

#[test]
fn a_waiting_request_does_not_block_later_admissions() {
    let mut h = Harness::new(16, tier(false, 64));
    let key = host_only(&mut h, 1);
    h.evict_oldest(2);
    assert_eq!(h.anchors(), 0);
    h.submit(3, &extended(1, 17, 7, 0));
    h.submit(4, &prompt(5, 9));
    let report = h.admit();
    assert_eq!(report.admitted(), [RequestId::new(4)]);
    assert_eq!(report.ram_waits().len(), 1);
    let (waiter, id) = report.ram_waits()[0];
    assert_eq!((waiter, id.key()), (RequestId::new(3), key));
    assert_eq!(h.s.request_state(waiter).unwrap(), RequestState::Queued);
    // The promotion's pages stay reserved while it waits, beside request
    // 4's prefill target.
    assert_eq!(h.s.reserved_pages(), ANCHOR_PAGES + 3);
    assert_eq!(h.s.pending_promotions()[0].pages, ANCHOR_PAGES);
    h.finish(true);
    assert_eq!(h.s.request_state(waiter).unwrap(), RequestState::Queued);
    assert_eq!(h.s.reserved_pages(), ANCHOR_PAGES);
    assert_eq!(h.promote(KvTransferStatus::Complete), 1);
    assert_eq!(h.step(), [(PrefixPublishV1::Published, ANCHOR_PAGES)]);
    assert_eq!(h.s.reserved_pages(), 0);
    let report = h.admit();
    assert_eq!(report.admitted(), [waiter]);
    assert!(report.ram_waits().is_empty() && report.ram_fallbacks().is_empty());
    assert_eq!(h.s.cached_tokens(waiter).unwrap(), 17);
    h.restore_all();
    assert_eq!(
        h.s.take_prefix_ram_hits(),
        [PrefixRamHitV1 {
            request_id: waiter,
            key,
            tokens: 17
        }]
    );
    // The resume made the host copy a shadow of the GPU anchor.
    assert_eq!(
        h.s.drop_invalidated_host_entries(&mut h.tier).unwrap(),
        [(key, PrefixHostInvalidationV1::Resumed)]
    );
    h.check();
    h.finish(true);
    assert_eq!(h.tier.used_pages(), 0);
}

#[test]
fn followers_share_one_promotion_and_each_counts_its_ram_hit() {
    let mut h = Harness::new(48, tier(false, 64));
    let key = host_only(&mut h, 1);
    h.evict_oldest(2);
    assert_eq!(h.anchors(), 0);
    h.submit(3, &extended(1, 17, 7, 0));
    h.submit(4, &extended(1, 17, 7, 1));
    let report = h.admit();
    assert!(report.admitted().is_empty());
    let waits = report.ram_waits();
    assert_eq!(waits.len(), 2);
    assert_eq!(waits[0].1, waits[1].1, "the follower joins the promotion");
    let id = waits[0].1;
    assert_eq!(h.s.pending_promotions().len(), 1);
    assert_eq!(h.s.reserved_pages(), ANCHOR_PAGES);
    // A request arriving while the copy is in flight joins it too.
    let plan = h.s.begin_promotion(&mut h.tier, id).unwrap();
    h.check();
    h.submit(5, &extended(1, 17, 7, 2));
    assert_eq!(h.admit().ram_waits(), [(RequestId::new(5), id)]);
    assert!(h.s.pending_promotions().is_empty());
    h.complete(plan, KvTransferStatus::Complete);
    assert_eq!(h.step(), [(PrefixPublishV1::Published, ANCHOR_PAGES)]);
    // A request that did not wait hits the published anchor on the GPU and
    // counts no RAM hit.
    h.submit(6, &extended(1, 17, 7, 3));
    let report = h.admit();
    assert_eq!(report.admitted().len(), 4);
    assert!(report.ram_waits().is_empty() && report.ram_fallbacks().is_empty());
    h.restore_all();
    let mut hits = h.s.take_prefix_ram_hits();
    hits.sort_by_key(|hit| hit.request_id);
    assert_eq!(
        hits.iter()
            .map(|hit| hit.request_id.get())
            .collect::<Vec<_>>(),
        [3, 4, 5]
    );
    assert!(hits.iter().all(|hit| hit.key == key && hit.tokens == 17));
    h.finish(true);
    h.s.drop_invalidated_host_entries(&mut h.tier).unwrap();
    h.check();
}
#[test]
fn a_failed_promotion_falls_back_to_the_gpu_match_or_cold_and_is_not_retried() {
    for (gpu, status) in [
        (false, KvTransferStatus::FailedIdle),
        (true, KvTransferStatus::FailedIdle),
        (true, KvTransferStatus::Complete),
    ] {
        let case = format!("gpu match {gpu}, {status:?}");
        let mut h = Harness::new(16, tier(false, 64));
        host_only(&mut h, 1);
        if gpu {
            // Shorter anchors of the same prompt stay on the GPU.
            h.run(2, &prompt(1, 8));
        }
        let anchors = h.anchors();
        h.evict_oldest(3);
        assert_eq!(h.anchors(), anchors - 1, "{case}");
        let waiter = RequestId::new(4);
        h.submit(4, &extended(1, 17, 7, 0));
        assert_eq!(h.admit().ram_waits().len(), 1, "{case}");
        let expected = if status == KvTransferStatus::Complete {
            // A recapture of the key invalidates the promotion in flight.
            let id = h.s.pending_promotions()[0].id;
            let plan = h.s.begin_promotion(&mut h.tier, id).unwrap();
            h.run(5, &prompt(1, 17));
            h.complete(plan, status);
            let reason = PrefixHostInvalidationV1::Recaptured;
            assert_eq!(
                h.step(),
                [(PrefixPublishV1::Invalidated(reason), ANCHOR_PAGES)],
                "{case}"
            );
            PrefixRamFallbackV1::Invalidated(reason)
        } else {
            h.promote(status);
            assert_eq!(
                h.step(),
                [(PrefixPublishV1::NotCopied, ANCHOR_PAGES)],
                "{case}"
            );
            PrefixRamFallbackV1::Failed
        };
        assert_eq!(h.s.reserved_pages(), 0, "{case}");
        // A failed copy keeps its source entry: the waiter never chooses it
        // again, while a request that did not fail on it still does.
        let other = RequestId::new(6);
        h.submit(6, &extended(1, 17, 7, 1));
        let report = h.admit();
        let retried = report.ram_waits().iter().any(|(id, _)| *id == waiter);
        assert!(!retried, "{case}");
        assert!(report.admitted().contains(&waiter), "{case}");
        assert_eq!(report.ram_fallbacks(), [(waiter, expected)], "{case}");
        let cached = match (gpu, status) {
            (_, KvTransferStatus::Complete) => 17,
            (true, _) => 8,
            (false, _) => 0,
        };
        assert_eq!(h.s.cached_tokens(waiter).unwrap(), cached, "{case}");
        let promoted = status == KvTransferStatus::FailedIdle;
        assert_eq!(
            report
                .ram_waits()
                .iter()
                .map(|(id, _)| *id)
                .collect::<Vec<_>>(),
            if promoted { vec![other] } else { vec![] },
            "{case}"
        );
        h.restore_all();
        assert!(h.s.take_prefix_ram_hits().is_empty(), "{case}");
        h.finish(true);
        h.s.drop_invalidated_host_entries(&mut h.tier).unwrap();
        h.check();
    }
}
#[test]
fn a_cancelled_waiter_leaves_its_promotion_to_publish_a_normal_anchor() {
    let mut h = Harness::new(16, tier(false, 64));
    let key = host_only(&mut h, 1);
    h.evict_oldest(2);
    h.submit(3, &extended(1, 17, 7, 0));
    assert_eq!(h.admit().ram_waits().len(), 1);
    let id = h.s.pending_promotions()[0].id;
    let plan = h.s.begin_promotion(&mut h.tier, id).unwrap();
    h.s.cancel(RequestId::new(3)).unwrap();
    h.check();
    assert_eq!(h.s.reserved_pages(), ANCHOR_PAGES);
    h.complete(plan, KvTransferStatus::Complete);
    assert_eq!(h.step(), [(PrefixPublishV1::Published, ANCHOR_PAGES)]);
    assert_eq!(h.anchors(), 1);
    assert_eq!(h.s.reserved_pages(), 0);
    // An untouched shadow stays; the next request is a plain GPU hit.
    assert!(
        h.s.drop_invalidated_host_entries(&mut h.tier)
            .unwrap()
            .is_empty()
    );
    h.submit(4, &extended(1, 17, 7, 1));
    let report = h.admit();
    assert_eq!(report.admitted(), [RequestId::new(4)]);
    assert!(report.ram_waits().is_empty());
    h.restore_all();
    assert_eq!(h.s.cached_tokens(RequestId::new(4)).unwrap(), 17);
    assert!(h.s.take_prefix_ram_hits().is_empty());
    assert_eq!(
        h.s.drop_invalidated_host_entries(&mut h.tier).unwrap(),
        [(key, PrefixHostInvalidationV1::Resumed)]
    );
    h.finish(true);
    h.check();
}

#[test]
fn an_unconfirmed_promotion_keeps_its_pages_and_waiters() {
    let mut h = Harness::new(16, tier(false, 64));
    host_only(&mut h, 1);
    h.evict_oldest(2);
    h.submit(3, &extended(1, 17, 7, 0));
    h.admit();
    assert_eq!(h.promote(KvTransferStatus::Unconfirmed), 1);
    for _ in 0..2 {
        assert!(h.step().is_empty());
        assert!(h.admit().admitted().is_empty());
        assert_eq!(h.s.reserved_pages(), ANCHOR_PAGES);
        assert!(h.s.pending_promotions().is_empty());
    }
}

#[test]
fn ram_wins_only_by_the_minimum_gain() {
    // h = 17 against g = 8 gains 9 tokens; 64 tokens against 60 gain 4, at
    // h / 16 = 4.
    for (base, short, minimum, ram) in [
        (17, 8, 9, true),
        (17, 8, 10, false),
        (17, 0, 17, true),
        (17, 0, 18, false),
        (64, 60, 1, true),
        (64, 61, 1, false),
    ] {
        let case = format!("h {base} g {short} minimum {minimum}");
        let mut h = Harness::new(64, tier(false, 64));
        let tokens = prompt(1, base);
        h.run(1, &tokens);
        h.demote(key(&tokens, base));
        if short > 0 {
            h.run(2, &prompt(1, short));
        }
        // The long anchor is the oldest.
        let anchors = h.anchors();
        h.evict_oldest(3);
        assert_eq!(h.anchors(), anchors - 1, "{case}");
        h.s.configure_offload(offload(minimum));
        let waiter = RequestId::new(4);
        h.submit(4, &extended(1, base, 3, 0));
        let report = h.admit();
        assert_eq!(report.ram_waits().len(), usize::from(ram), "{case}");
        if !ram {
            assert_eq!(report.admitted(), [waiter], "{case}");
            assert!(report.ram_fallbacks().is_empty(), "{case}");
            assert_eq!(h.s.cached_tokens(waiter).unwrap(), short, "{case}");
        }
    }
}
#[test]
fn a_promotion_that_does_not_fit_falls_back_without_waiting() {
    for (reserved_tokens, ram) in [(32, true), (36, false)] {
        let mut h = Harness::new(16, tier(false, 64));
        host_only(&mut h, 1);
        h.evict_oldest(2);
        // A live pinned request holds its whole reservation: 8 pages leave
        // exactly the 8 the hit needs (6 pages for its prefill target, one
        // snapshot pin and one tail copy), and 9 leave one too few.
        h.submit_spec(
            RequestSpec::new(RequestId::new(3), reserved_tokens - 1, 2)
                .unwrap()
                .with_reserved_admission(),
            &prompt(6, reserved_tokens - 1),
        );
        assert_eq!(h.admit().admitted(), [RequestId::new(3)]);
        h.submit(4, &extended(1, 17, 7, 0));
        let report = h.admit();
        if ram {
            assert_eq!(report.ram_waits().len(), 1);
            continue;
        }
        assert_eq!(report.admitted(), [RequestId::new(4)]);
        assert_eq!(
            report.ram_fallbacks(),
            [(RequestId::new(4), PrefixRamFallbackV1::Capacity)]
        );
        assert_eq!(h.s.cached_tokens(RequestId::new(4)).unwrap(), 0);
        assert!(h.s.pending_promotions().is_empty());
    }
}

#[test]
fn the_step_budget_defers_promotions_and_a_large_one_waits_for_demotions() {
    let mut h = Harness::new(24, tier(false, ANCHOR_PAGES));
    let keys = [host_only(&mut h, 1), host_only(&mut h, 2)];
    h.evict_oldest(3);
    h.evict_oldest(30);
    assert_eq!(h.anchors(), 0);
    h.submit(4, &extended(1, 17, 7, 0));
    h.submit(5, &extended(2, 17, 7, 0));
    let report = h.admit();
    assert_eq!(report.ram_waits().len(), 1);
    assert_eq!(report.ram_waits()[0].1.key(), keys[0]);
    assert!(report.admitted().is_empty());
    assert_eq!(h.s.deferred_promotion_pages(), ANCHOR_PAGES);
    // A second admission pass in the same step finds the budget spent.
    assert!(h.admit().ram_waits().is_empty());
    h.step();
    assert_eq!(h.s.deferred_promotion_pages(), 0);
    let report = h.admit();
    assert_eq!(report.ram_waits().len(), 1);
    assert_eq!(report.ram_waits()[0].1.key(), keys[1]);
    assert_eq!(h.s.reserved_pages(), 2 * ANCHOR_PAGES);

    // A promotion larger than the budget starts alone, and only with no
    // demotion outstanding.
    let mut h = Harness::new(24, tier(false, ANCHOR_PAGES - 2));
    host_only(&mut h, 1);
    let other = prompt(2, 9);
    h.run(2, &other);
    h.evict_oldest(3);
    assert_eq!(h.anchors(), 1);
    let demotion = h.s.begin_demotion(&mut h.tier, key(&other, 9)).unwrap();
    assert_eq!(h.s.outstanding_demotions(), 1);
    h.check();
    h.submit(4, &extended(1, 17, 7, 0));
    assert!(h.admit().ram_waits().is_empty());
    assert_eq!(h.s.deferred_promotion_pages(), ANCHOR_PAGES);
    h.complete(demotion, KvTransferStatus::Complete);
    assert_eq!(h.s.outstanding_demotions(), 0);
    assert_eq!(h.admit().ram_waits().len(), 1);
    assert_eq!(h.s.reserved_pages(), ANCHOR_PAGES);
}
#[test]
fn an_exhausted_pool_falls_back_to_a_recomputed_gpu_match() {
    let mut h = Harness::new(16, tier(false, 64));
    host_only(&mut h, 1);
    h.run(2, &prompt(1, 8));
    // A captured 44-token anchor evicts the older long one and fills the
    // pool.
    let big = prompt(3, 44);
    h.run(3, &big);
    assert_eq!(h.anchors(), 3);
    assert_eq!(h.s.kv_stats().free_blocks(), 0);
    // Its demotion leases its pages, so only the short anchors are
    // evictable: four pages, while the promotion needs six.
    let demotion = h.s.begin_demotion(&mut h.tier, key(&big, 44)).unwrap();
    h.check();
    let waiter = RequestId::new(4);
    h.submit(4, &extended(1, 17, 7, 0));
    let report = h.admit();
    assert!(report.ram_waits().is_empty());
    assert_eq!(report.admitted(), [waiter]);
    assert_eq!(
        report.ram_fallbacks(),
        [(waiter, PrefixRamFallbackV1::PoolExhausted)]
    );
    assert_eq!(h.s.cached_tokens(waiter).unwrap(), 8, "the GPU match stays");
    assert_eq!(h.anchors(), 3);
    h.restore_all();
    h.complete(demotion, KvTransferStatus::Complete);
    h.finish(true);
}
#[test]
fn forced_mode_turns_each_repeat_into_a_ram_hit() {
    let mut h = Harness::new(24, tier(true, 64));
    let tokens = prompt(1, 17);
    h.run(1, &tokens);
    let key = key(&tokens, 17);
    let demotion = h.s.begin_demotion(&mut h.tier, key).unwrap();
    assert_eq!(h.anchors(), 0, "a forced demotion removes its GPU anchor");
    h.check();
    // The repeat waits for the copy in flight instead of recomputing.
    h.submit(2, &extended(1, 17, 7, 0));
    let report = h.admit();
    assert!(report.admitted().is_empty() && report.ram_waits().is_empty());
    h.complete(demotion, KvTransferStatus::Complete);
    for (round, id) in [2u64, 3].into_iter().enumerate() {
        if round > 0 {
            h.submit(id, &extended(1, 17, 7, id as u32));
        }
        let report = h.admit();
        assert_eq!(report.ram_waits().len(), 1, "round {round}");
        h.promote(KvTransferStatus::Complete);
        assert_eq!(h.step(), [(PrefixPublishV1::Published, ANCHOR_PAGES)]);
        assert_eq!(h.admit().admitted(), [RequestId::new(id)]);
        assert_eq!(h.anchors(), 1);
        h.restore_all();
        assert_eq!(h.s.take_prefix_ram_hits().len(), 1, "round {round}");
        // The restore removed the promoted anchor and kept its host entry.
        assert_eq!(h.anchors(), 0, "round {round}");
        assert!(
            h.s.drop_invalidated_host_entries(&mut h.tier)
                .unwrap()
                .is_empty()
        );
        h.finish(false);
    }
}

#[test]
fn an_abandoned_restore_of_a_promoted_anchor_counts_no_ram_hit() {
    let mut h = Harness::new(16, tier(false, 64));
    host_only(&mut h, 1);
    h.evict_oldest(2);
    h.submit(3, &extended(1, 17, 7, 0));
    h.admit();
    h.promote(KvTransferStatus::Complete);
    h.step();
    h.admit();
    h.s.abandon_prefix_restore(RequestId::new(3)).unwrap();
    h.check();
    assert!(h.s.take_prefix_ram_hits().is_empty());
    h.finish(true);
}

#[test]
fn swaps_keep_priority_over_public_entries() {
    // A tier of six host pages holds one public entry. A swap takes its
    // pages; a demotion never takes a swap's.
    let config = SchedulerConfig::new(2, 64, 64, 64, 1)
        .unwrap()
        .with_admission(AdmissionPolicy::Optimistic(
            OptimisticAdmission::new(0, 3)
                .unwrap()
                .with_forced_preemption(2)
                .unwrap(),
        ))
        .with_prefix_cache(1)
        .with_prefix_ram(tier(false, 64));
    let mut s = Scheduler::new(config, KvGeometry::new(BLOCK, 32).unwrap()).unwrap();
    s.configure_offload(offload(1));
    let mut tier = s.new_host_tier(ANCHOR_PAGES).unwrap();
    let check = |s: &Scheduler, tier: &HostTier| {
        s.audit().unwrap();
        s.audit_host_tier(tier).unwrap();
    };
    let submit = |s: &mut Scheduler, id: u64, tokens: &[u32], output: usize| {
        s.submit_with_prefix(
            RequestSpec::new(RequestId::new(id), tokens.len(), output).unwrap(),
            Some(PrefixRequestV1::new(Arc::new(chain(tokens)), true)),
        )
        .unwrap();
    };
    let step = |s: &mut Scheduler, tier: &HostTier| {
        let batch = s.next_batch().unwrap().unwrap();
        let outcomes: Vec<_> = batch
            .items()
            .iter()
            .map(|item| {
                BatchOutcome::success(
                    item.request_id(),
                    item.phase(),
                    usize::from(item.expects_output()),
                    false,
                )
            })
            .collect();
        s.commit_batch(batch.id(), &outcomes).unwrap();
        for capture in s.take_prefix_captures() {
            s.begin_prefix_capture(&capture).unwrap();
            s.end_prefix_capture(&capture).unwrap();
        }
        check(s, tier);
    };
    let public = prompt(1, 17);
    submit(&mut s, 1, &public, 2);
    step(&mut s, &tier);
    s.cancel(RequestId::new(1)).unwrap();
    let plan = s.begin_demotion(&mut tier, key(&public, 17)).unwrap();
    s.accept_kv_transfer(
        &mut tier,
        plan.into_submission()
            .completion(KvTransferStatus::Complete),
    )
    .unwrap();
    check(&s, &tier);
    assert_eq!(tier.used_pages(), ANCHOR_PAGES);
    let other = prompt(3, 17);
    submit(&mut s, 3, &other, 2);
    step(&mut s, &tier);
    s.cancel(RequestId::new(3)).unwrap();
    // A decoding request is preempted and swapped out.
    let swapped = prompt(2, 8);
    let resume_chain = |_: RequestId, tokens: usize| {
        let mut all = swapped.clone();
        all.resize(tokens, 7);
        Some(PrefixRequestV1::new(Arc::new(chain(&all)), true))
    };
    submit(&mut s, 2, &swapped, 6);
    let bound = s.step_allocation_bound();
    let capture = loop {
        let victims = s.preempt_for_next_step(0, resume_chain).unwrap();
        if !victims.is_empty() {
            break s.take_prefix_captures().pop().unwrap();
        }
        step(&mut s, &tier);
    };
    assert_eq!(capture.tokens(), 9);
    assert!(s.begin_prefix_capture(&capture).unwrap().is_some());
    let out = s.offload_capture(&mut tier, &capture, true).unwrap();
    s.end_prefix_capture(&capture).unwrap();
    check(&s, &tier);
    s.accept_kv_transfer(
        &mut tier,
        out.into_submission().completion(KvTransferStatus::Complete),
    )
    .unwrap();
    check(&s, &tier);
    assert_eq!(tier.used_pages(), 4, "the swap evicted the public entry");
    // A swapped request that may resume adds its pages to the bound: three
    // KV pages and a snapshot page.
    assert_eq!(s.step_allocation_bound(), bound + 4);
    // The host tier refuses a demotion that would need the swap's pages.
    assert!(s.begin_demotion(&mut tier, key(&other, 17)).is_err());
    assert_eq!(s.outstanding_demotions(), 0);
    check(&s, &tier);
    // The swapped request resumes its own GPU boundary first.
    let report = s.admit().unwrap();
    assert_eq!(report.admitted(), [RequestId::new(2)]);
    assert!(report.ram_waits().is_empty() && s.pending_reloads().is_empty());
    assert_eq!(s.pending_prefix_restores()[0].tokens(), 9);
    s.complete_prefix_restore(RequestId::new(2)).unwrap();
    s.release_unused_swaps(&mut tier).unwrap();
    check(&s, &tier);
}
#[test]
fn the_step_allocation_bound_adds_its_terms() {
    // 64 budget tokens are 16 pages; four sequences add 8 pages of rounding
    // and tail copies, 8 public and 4 private snapshot pages, and the
    // promotion budget adds 10.
    let h = Harness::new(64, tier(false, 10));
    assert_eq!(h.s.step_allocation_bound(), 16 + 8 + 4 * (2 + 1) + 10);
    // Reserved admission never preempts.
    let reserved = Harness::with_config(64, tier(false, 10), |config| {
        config.with_admission(AdmissionPolicy::Reserved)
    });
    assert_eq!(reserved.s.step_allocation_bound(), 16 + 8 + 4 * 2 + 10);
    let off = Harness::new(64, None);
    assert_eq!(off.s.step_allocation_bound(), 16 + 8 + 4 * (2 + 1));
    // A demotion of the only anchor, six pages, starts only while the free
    // pages cover the bound of 46 beside them.
    for (pages, headroom) in [(58, true), (57, false)] {
        let mut h = Harness::new(pages, tier(false, 10));
        let tokens = prompt(1, 17);
        h.run(1, &tokens);
        let key = key(&tokens, 17);
        assert_eq!(h.s.prefix_demotion_headroom(key), headroom, "{pages}");
    }
}

/// A small deterministic generator for the seeded workload.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, bound: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % bound as u64) as usize
    }
}

/// Runs the due captures as the engine does, swapping out a preemption's
/// boundary, and returns the snapshot pages they allocated.
fn run_captures(
    s: &mut Scheduler,
    tier: &mut HostTier,
    transfers: &mut Vec<KvTransferPlan>,
) -> usize {
    let mut allocated = 0;
    for due in s.take_prefix_captures() {
        let pages = s.begin_prefix_capture(&due).unwrap();
        allocated += pages.as_ref().map_or(0, Vec::len);
        if s.capture_swaps(&due) && s.has_capture_anchor(&due) {
            match s.offload_capture(tier, &due, pages.is_some()) {
                Ok(plan) => transfers.push(plan),
                Err(ScheduleError::Kv(KvError::PoolExhausted { .. })) => {}
                Err(error) => panic!("{error}"),
            }
        }
        s.end_prefix_capture(&due).unwrap();
    }
    allocated
}

/// Submits the copies admission allocated pages for, promotions and
/// swap-ins, and returns the pages not counted before.
fn start_copies(
    s: &mut Scheduler,
    tier: &mut HostTier,
    transfers: &mut Vec<KvTransferPlan>,
    seen: &mut std::collections::BTreeSet<runtime_kv::PendingAnchorId>,
) -> usize {
    let mut allocated = 0;
    for promotion in s.pending_promotions() {
        if seen.insert(promotion.id) {
            allocated += promotion.pages;
        }
        transfers.push(s.begin_promotion(tier, promotion.id).unwrap());
    }
    for id in s.pending_reloads() {
        let plan = s.begin_reload(tier, id).unwrap();
        allocated += plan.pages().len();
        transfers.push(plan);
    }
    allocated
}

/// Drives a seeded workload of shared prompts through preemption, swaps,
/// demotions and promotions, as the engine would, and counts every page
/// each step takes from the pool: preemption and post-commit snapshots,
/// promotions, swap-ins and the batch's appends. No step exceeds the bound
/// read at its start.
#[test]
fn per_step_allocations_stay_within_the_bound() {
    let mut promoted = 0;
    for seed in 1..=24u64 {
        let mut rng = Lcg(seed);
        let config = SchedulerConfig::new(4, 64, 32, 16, 1)
            .unwrap()
            .with_admission(AdmissionPolicy::Optimistic(
                OptimisticAdmission::new(0, 2)
                    .unwrap()
                    .with_forced_preemption(3)
                    .unwrap(),
            ))
            .with_prefix_cache(1)
            .with_prefix_ram(tier(false, 16));
        let mut s = Scheduler::new(config, KvGeometry::new(BLOCK, 96).unwrap()).unwrap();
        s.configure_offload(offload(1));
        let mut tier = s.new_host_tier(64).unwrap();
        let mut transfers: Vec<KvTransferPlan> = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        let mut prompts = std::collections::BTreeMap::new();
        let mut next_id = 1u64;
        for _ in 0..160 {
            let bound = s.step_allocation_bound();
            let mut allocated = 0usize;
            for plan in transfers.drain(..) {
                let status = if rng.below(8) == 0 {
                    KvTransferStatus::FailedIdle
                } else {
                    KvTransferStatus::Complete
                };
                s.accept_kv_transfer(&mut tier, plan.into_submission().completion(status))
                    .unwrap();
            }
            s.begin_prefix_ram_step().unwrap();
            if rng.below(2) == 0 {
                let salt = rng.below(3) as u32;
                let length = 6 + rng.below(30);
                let tokens = prompt(salt, length);
                let output = 1 + rng.below(6);
                s.submit_with_prefix(
                    RequestSpec::new(RequestId::new(next_id), length, output).unwrap(),
                    Some(PrefixRequestV1::new(Arc::new(chain(&tokens)), true)),
                )
                .unwrap();
                prompts.insert(next_id, tokens);
                next_id += 1;
            }
            let resume = |id: RequestId, length: usize| {
                let mut tokens = prompts[&id.get()].clone();
                tokens.resize(length, 3);
                Some(PrefixRequestV1::new(Arc::new(chain(&tokens)), true))
            };
            s.preempt_for_next_step(0, resume).unwrap();
            allocated += run_captures(&mut s, &mut tier, &mut transfers);
            // Promotions and swap-ins allocate at admission; the batch's own
            // admission pass may start more of them.
            promoted += s.admit().unwrap().ram_waits().len();
            allocated += start_copies(&mut s, &mut tier, &mut transfers, &mut seen);
            s.release_unused_swaps(&mut tier).unwrap();
            for restore in s.pending_prefix_restores() {
                s.complete_prefix_restore(restore.request_id()).unwrap();
            }
            if let Some(batch) = s.next_batch().unwrap() {
                allocated += batch
                    .kv_plan()
                    .appends()
                    .iter()
                    .map(|append| append.allocated().len())
                    .sum::<usize>();
                let outcomes: Vec<_> = batch
                    .items()
                    .iter()
                    .map(|item| {
                        BatchOutcome::success(
                            item.request_id(),
                            item.phase(),
                            usize::from(item.expects_output()),
                            false,
                        )
                    })
                    .collect();
                s.commit_batch(batch.id(), &outcomes).unwrap();
                allocated += run_captures(&mut s, &mut tier, &mut transfers);
            }
            allocated += start_copies(&mut s, &mut tier, &mut transfers, &mut seen);
            s.release_unused_swaps(&mut tier).unwrap();
            if s.outstanding_demotions() == 0
                && let Some(candidate) = s
                    .prefix_demotion_candidates(8)
                    .into_iter()
                    .find(|candidate| s.prefix_demotion_headroom(candidate.key))
                && let Ok(plan) = s.begin_demotion(&mut tier, candidate.key)
            {
                transfers.push(plan);
            }
            s.drop_invalidated_host_entries(&mut tier).unwrap();
            for id in s.reap_terminal(usize::MAX).unwrap() {
                prompts.remove(&id.get());
            }
            s.audit().unwrap();
            s.audit_host_tier(&tier).unwrap();
            assert!(allocated <= bound, "seed {seed}: {allocated} > {bound}");
        }
    }
    assert!(promoted > 0, "the workload promotes");
}
