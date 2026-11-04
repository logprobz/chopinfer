//! Scheduler admission for the RAM prefix tier: the GPU match against the
//! ready host match, promotions with their reservations and waiters, the
//! engine entry points, and the per-step allocation bound.

use std::sync::Arc;

use runtime_kv::{
    HostTier, KvGeometry, KvTransferPlan, KvTransferStatus, PrefixCacheKeyV1, PrefixChainV1,
    PrefixEvictionV1, PrefixHostInvalidationV1,
};
use runtime_scheduler::{
    AdmissionPolicy, AdmissionReport, BatchOutcome, OffloadConfig, OptimisticAdmission,
    PrefixRamConfig, PrefixRequestV1, RequestId, RequestSpec, Scheduler, SchedulerConfig,
};

const BLOCK: usize = 4;

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
