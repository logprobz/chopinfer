//! The pool's view of the RAM prefix tier: public anchors resident in a
//! [`HostTier`](crate::HostTier). The host tier owns host page lifetime and
//! updates these maps and the ancestry residency in the same call as every
//! drop. The pool marks entries a logical event made unusable and the host
//! tier drops them later, since the pool cannot release host pages itself.
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::ancestry::{PrefixResidency, PrefixTier};
use crate::{HostAnchorId, KvError, KvPool, KvTransferId, PrefixCacheKeyV1, PrefixChainV1};

/// A public anchor resident in the host tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefixHostEntryV1 {
    /// The digest of the anchor's full blocks, which indexes it for lookup.
    pub chain: PrefixCacheKeyV1,
    pub tokens: usize,
    /// The pool's anchor clock at the anchor's last GPU use.
    pub last_used: u64,
    pub host: HostAnchorId,
    /// The copy out completed on every rank.
    pub ready: bool,
    /// Distinguishes this entry from a later demotion of the same key.
    pub serial: u64,
}

/// A public frontier the RAM tier may copy ahead of its eviction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefixDemotionCandidateV1 {
    pub key: PrefixCacheKeyV1,
    pub tokens: usize,
    /// Pages its eviction would free now: those only this anchor references.
    pub freeable_pages: usize,
}

/// A public GPU eviction with the RAM tier on, for miss accounting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefixEvictionV1 {
    pub key: PrefixCacheKeyV1,
    pub tokens: usize,
    /// A ready, usable host entry holds the key, so the eviction lost nothing.
    pub covered: bool,
    /// An indexed frontier of both tiers: the shape a demotion copies.
    pub frontier: bool,
}

/// Public evictions the pool logs between drains; older ones are discarded.
pub const PREFIX_EVICTION_LOG_V1: usize = 4096;

/// Why a host entry became unusable before its host tier dropped it.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PrefixHostInvalidationV1 {
    /// A resume touched the key's GPU anchor.
    Resumed,
    /// The key was captured again on the GPU.
    Recaptured,
    /// The key gained a resident descendant in either tier.
    Descendant,
}

/// A promotion copying a host entry back into a pending anchor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingPromotion {
    /// The entry's tokens, which stay the node's depth after its drop.
    tokens: usize,
    /// The first logical event on the key since the promotion began,
    /// which forbids its publish.
    invalidation: Option<PrefixHostInvalidationV1>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct PrefixRam {
    pub(crate) entries: BTreeMap<PrefixCacheKeyV1, PrefixHostEntryV1>,
    /// Entry keys by the digest of their full blocks.
    chains: BTreeMap<PrefixCacheKeyV1, BTreeSet<PrefixCacheKeyV1>>,
    /// Entry keys by last GPU use, oldest first.
    pub(crate) order: BTreeSet<(u64, PrefixCacheKeyV1)>,
    serial: u64,
    /// Demotions in flight: the GPU anchor lifetime whose pages the copy
    /// leases, and its transfer.
    pub(crate) demotions: BTreeMap<PrefixCacheKeyV1, (u64, KvTransferId)>,
    /// Entries a logical event made unusable, awaiting their host drop.
    invalidations: BTreeMap<PrefixCacheKeyV1, PrefixHostInvalidationV1>,
    /// Promotions in flight by key and the serial of the entry each
    /// copies. Each holds the key's node until it ends, through its
    /// entry's host eviction.
    promotions: BTreeMap<(PrefixCacheKeyV1, u64), PendingPromotion>,
    /// Public evictions since the last drain, oldest first.
    evictions: VecDeque<PrefixEvictionV1>,
    /// Evictions the log's bound discarded since the last drain.
    evictions_discarded: u64,
    /// The host tier whose anchors back the entries, bound by its first
    /// demotion.
    pub(crate) tier: Option<u64>,
    /// Debug forced mode: entries survive a descendant and a resume.
    forced: bool,
}

impl PrefixRam {
    /// Whether no key is held: no host entry, demotion or promotion.
    pub(crate) fn holds_no_key(&self) -> bool {
        self.entries.is_empty() && self.demotions.is_empty() && self.promotions.is_empty()
    }
}

impl KvPool {
    /// The RAM prefix tier: public anchors may keep host-resident entries
    /// that one [`HostTier`](crate::HostTier) owns. Implies the ancestry
    /// index. Configure before creating anchors.
    #[must_use]
    pub fn with_prefix_ram(mut self, enabled: bool) -> Self {
        self.assert_unindexed();
        self.ram = enabled.then(PrefixRam::default);
        self.configure_index()
    }

    /// Debug forced mode of the RAM tier: host entries survive gaining a
    /// resident descendant and a resume of their key, so every capture can
    /// stay in RAM. A recapture still invalidates them.
    #[must_use]
    pub fn with_prefix_ram_forced(mut self, enabled: bool) -> Self {
        match &mut self.ram {
            Some(ram) => ram.forced = enabled,
            None => assert!(!enabled, "configure the RAM tier first"),
        }
        self
    }

    #[must_use]
    pub const fn prefix_ram_enabled(&self) -> bool {
        self.ram.is_some()
    }

    #[must_use]
    pub fn host_entry(&self, key: PrefixCacheKeyV1) -> Option<PrefixHostEntryV1> {
        self.ram.as_ref()?.entries.get(&key).copied()
    }

    /// Walks the eviction order from the next victim over the anchors whose
    /// eviction frees the first `lead_pages` pages, and returns the ones a
    /// demotion may copy, in that order: indexed public frontiers of both
    /// tiers with at least `minimum_tokens` tokens, no host entry and no copy
    /// in flight. The anchor whose demotion is in flight, which eviction
    /// skips, adds nothing.
    #[must_use]
    pub fn demotion_candidates(
        &self,
        lead_pages: usize,
        minimum_tokens: usize,
    ) -> Vec<PrefixDemotionCandidateV1> {
        let (Some(ram), Some(index)) = (&self.ram, &self.ancestry) else {
            return Vec::new();
        };
        let mut covered = 0usize;
        let mut candidates = Vec::new();
        for key in self.anchor_order.values() {
            if covered >= lead_pages {
                break;
            }
            if self.demotion_in_flight(*key) {
                continue;
            }
            let anchor = &self.anchors[key];
            let freeable_pages = anchor
                .blocks
                .iter()
                .chain(&anchor.state)
                .filter(|page| self.blocks[page.index()] == 1)
                .count();
            covered += freeable_pages;
            if !anchor.private
                && anchor.indexed
                && anchor.tokens >= minimum_tokens
                && !ram.entries.contains_key(key)
                && !ram.demotions.contains_key(key)
                && !index.has_descendant(*key)
            {
                candidates.push(PrefixDemotionCandidateV1 {
                    key: *key,
                    tokens: anchor.tokens,
                    freeable_pages,
                });
            }
        }
        candidates
    }

    /// Finds the longest ready, usable host entry of this prompt that ends at
    /// most at `limit` tokens and outside every media span, as
    /// [`Self::lookup_anchor`] does for GPU anchors.
    #[must_use]
    pub fn lookup_host_anchor(
        &self,
        chain: &PrefixChainV1,
        limit: usize,
    ) -> Option<(PrefixCacheKeyV1, usize)> {
        let ram = self.ram.as_ref().filter(|ram| !ram.entries.is_empty())?;
        self.longest_match(chain, limit, &ram.chains, |key| {
            ram.entries
                .get(key)
                .filter(|entry| entry.ready && !ram.invalidations.contains_key(key))
                .map(|entry| entry.tokens)
        })
    }

    /// Starts a promotion of `key`'s ready, usable host entry and returns
    /// the entry's serial, which names the promotion. Until it ends it
    /// holds the key's node, and a logical event on the key invalidates
    /// it as it does the entry. A publish adds the GPU hold before it ends
    /// the promotion, so the node keeps its lifetime flags.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "pending anchors start promotions")
    )]
    pub(crate) fn begin_pending_promotion(
        &mut self,
        key: PrefixCacheKeyV1,
    ) -> Result<u64, KvError> {
        let ram = self.ram.as_mut().ok_or(KvError::StaleHandle)?;
        let entry = ram
            .entries
            .get(&key)
            .filter(|entry| entry.ready && !ram.invalidations.contains_key(&key))
            .ok_or(KvError::StaleHandle)?;
        let (serial, tokens) = (entry.serial, entry.tokens);
        // One promotion per entry; followers wait for its publish.
        if ram.promotions.contains_key(&(key, serial)) {
            return Err(KvError::StaleHandle);
        }
        ram.promotions.insert(
            (key, serial),
            PendingPromotion {
                tokens,
                invalidation: None,
            },
        );
        self.ancestry
            .as_mut()
            .expect("the RAM tier keeps the index")
            .add_residency(key, PrefixTier::Pending);
        Ok(serial)
    }

    /// Ends the promotion of `key`'s entry `serial`, returning the logical
    /// event that invalidated it, if any. The key's node goes once no tier
    /// holds it.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "pending anchors end promotions")
    )]
    pub(crate) fn end_pending_promotion(
        &mut self,
        key: PrefixCacheKeyV1,
        serial: u64,
    ) -> Result<Option<PrefixHostInvalidationV1>, KvError> {
        let promotion = self
            .ram
            .as_mut()
            .and_then(|ram| ram.promotions.remove(&(key, serial)))
            .ok_or(KvError::StaleHandle)?;
        if let Some(index) = &mut self.ancestry {
            index.release_residency(key, PrefixTier::Pending);
            self.reclassify_anchors();
        }
        Ok(promotion.invalidation)
    }

    /// Takes the public evictions logged since the last call, oldest first,
    /// with how many the bound of [`PREFIX_EVICTION_LOG_V1`] discarded.
    pub fn drain_prefix_evictions(&mut self) -> (Vec<PrefixEvictionV1>, u64) {
        self.ram.as_mut().map_or_else(
            || (Vec::new(), 0),
            |ram| {
                (
                    ram.evictions.drain(..).collect(),
                    std::mem::take(&mut ram.evictions_discarded),
                )
            },
        )
    }

    /// Whether `key`'s current GPU anchor is the one a demotion in flight
    /// leases.
    pub(crate) fn demotion_in_flight(&self, key: PrefixCacheKeyV1) -> bool {
        self.ram
            .as_ref()
            .and_then(|ram| ram.demotions.get(&key))
            .is_some_and(|(lifetime, _)| {
                self.anchors
                    .get(&key)
                    .is_some_and(|anchor| anchor.serial == *lifetime)
            })
    }

    /// What the eviction log records for `key`, if the tier is on and the
    /// anchor public.
    pub(crate) fn eviction_record(&self, key: PrefixCacheKeyV1) -> Option<PrefixEvictionV1> {
        let ram = self.ram.as_ref()?;
        let anchor = self.anchors.get(&key).filter(|anchor| !anchor.private)?;
        Some(PrefixEvictionV1 {
            key,
            tokens: anchor.tokens,
            covered: ram.entries.get(&key).is_some_and(|entry| entry.ready)
                && !ram.invalidations.contains_key(&key),
            frontier: anchor.indexed
                && !self
                    .ancestry
                    .as_ref()
                    .is_some_and(|index| index.has_descendant(key)),
        })
    }

    pub(crate) fn log_eviction(&mut self, record: PrefixEvictionV1) {
        let ram = self.ram.as_mut().expect("the RAM tier logs evictions");
        if ram.evictions.len() == PREFIX_EVICTION_LOG_V1 {
            ram.evictions.pop_front();
            ram.evictions_discarded += 1;
        }
        ram.evictions.push_back(record);
    }

    /// Why the key's host entry is unusable, until its host tier drops it.
    #[must_use]
    pub fn host_invalidation(&self, key: PrefixCacheKeyV1) -> Option<PrefixHostInvalidationV1> {
        self.ram.as_ref()?.invalidations.get(&key).copied()
    }

    /// Refuses a demotion of anything but an indexed public frontier
    /// without a host entry or a copy in flight.
    pub(crate) fn check_demotion(&self, key: PrefixCacheKeyV1, tier: u64) -> Result<(), KvError> {
        let ram = self.ram.as_ref().ok_or(KvError::StaleHandle)?;
        let anchor = self.anchors.get(&key).ok_or(KvError::StaleHandle)?;
        let index = self.ancestry.as_ref().ok_or(KvError::InvariantViolation)?;
        if ram.tier.is_some_and(|bound| bound != tier)
            || anchor.private
            || !anchor.indexed
            || ram.entries.contains_key(&key)
            || ram.demotions.contains_key(&key)
            || (!ram.forced && index.has_descendant(key))
        {
            return Err(KvError::StaleHandle);
        }
        Ok(())
    }

    /// Records a checked demotion's entry, not ready until its copy completes.
    pub(crate) fn record_demotion(
        &mut self,
        key: PrefixCacheKeyV1,
        tier: u64,
        host: HostAnchorId,
        transfer: KvTransferId,
    ) {
        let anchor = &self.anchors[&key];
        let (chain, tokens, last_used, lifetime) =
            (anchor.chain, anchor.tokens, anchor.last_used, anchor.serial);
        let ram = self.ram.as_mut().expect("a checked demotion");
        ram.tier = Some(tier);
        ram.serial += 1;
        ram.entries.insert(
            key,
            PrefixHostEntryV1 {
                chain,
                tokens,
                last_used,
                host,
                ready: false,
                serial: ram.serial,
            },
        );
        ram.chains.entry(chain).or_default().insert(key);
        ram.order.insert((last_used, key));
        ram.demotions.insert(key, (lifetime, transfer));
        self.ancestry
            .as_mut()
            .expect("the RAM tier keeps the index")
            .add_residency(key, PrefixTier::Host);
    }

    /// Ends a demotion's copy: a confirmed success makes its entry ready.
    pub(crate) fn finish_demotion(
        &mut self,
        key: PrefixCacheKeyV1,
        host: HostAnchorId,
        transfer: KvTransferId,
        success: bool,
    ) {
        let Some(ram) = &mut self.ram else {
            return;
        };
        if ram
            .demotions
            .get(&key)
            .is_some_and(|(_, id)| *id == transfer)
        {
            ram.demotions.remove(&key);
        }
        if success
            && let Some(entry) = ram.entries.get_mut(&key)
            && entry.host == host
        {
            entry.ready = true;
        }
    }

    /// Removes the entry the host tier dropped, and the key's node once no
    /// tier holds it.
    pub(crate) fn remove_host_entry(&mut self, key: PrefixCacheKeyV1, host: HostAnchorId) {
        let Some(ram) = &mut self.ram else {
            return;
        };
        if ram.entries.get(&key).is_none_or(|entry| entry.host != host) {
            return;
        }
        let entry = ram.entries.remove(&key).expect("entry checked");
        ram.order.remove(&(entry.last_used, key));
        if let Some(keys) = ram.chains.get_mut(&entry.chain) {
            keys.remove(&key);
            if keys.is_empty() {
                ram.chains.remove(&entry.chain);
            }
        }
        ram.invalidations.remove(&key);
        if let Some(index) = &mut self.ancestry {
            index.release_residency(key, PrefixTier::Host);
            self.reclassify_anchors();
        }
    }

    /// A logical event on `key`: its host entry, if any, becomes unusable
    /// and is reported for its host drop, and every pending promotion of
    /// the key is invalidated, even after its entry's host eviction.
    /// Forced mode exempts descendants and resumes.
    pub(crate) fn host_key_event(
        &mut self,
        key: PrefixCacheKeyV1,
        reason: PrefixHostInvalidationV1,
    ) {
        let Some(ram) = &mut self.ram else {
            return;
        };
        if ram.forced && reason != PrefixHostInvalidationV1::Recaptured {
            return;
        }
        if ram.entries.contains_key(&key) {
            ram.invalidations.entry(key).or_insert(reason);
        }
        for promotion in ram
            .promotions
            .range_mut((key, 0)..=(key, u64::MAX))
            .map(|(_, promotion)| promotion)
        {
            promotion.invalidation.get_or_insert(reason);
        }
    }

    pub(crate) fn invalidated_host_entries(
        &self,
    ) -> Vec<(PrefixCacheKeyV1, HostAnchorId, PrefixHostInvalidationV1)> {
        self.ram.as_ref().map_or_else(Vec::new, |ram| {
            ram.invalidations
                .iter()
                .map(|(key, reason)| (*key, ram.entries[key].host, *reason))
                .collect()
        })
    }

    /// Every key the ancestry index must hold, with its depth and tiers.
    pub(crate) fn expected_residency(
        &self,
    ) -> impl Iterator<Item = (PrefixCacheKeyV1, usize, PrefixResidency)> + '_ {
        let mut expected: BTreeMap<_, _> = self
            .anchors
            .iter()
            .filter(|(_, anchor)| anchor.indexed)
            .map(|(key, anchor)| (*key, (anchor.tokens, PrefixResidency::GPU)))
            .collect();
        if let Some(ram) = &self.ram {
            for (key, entry) in &ram.entries {
                expected
                    .entry(*key)
                    .or_insert((entry.tokens, PrefixResidency::default()))
                    .1
                    .host = true;
            }
            for ((key, _), promotion) in &ram.promotions {
                expected
                    .entry(*key)
                    .or_insert((promotion.tokens, PrefixResidency::default()))
                    .1
                    .pending += 1;
            }
        }
        expected
            .into_iter()
            .map(|(key, (tokens, residency))| (key, tokens, residency))
    }

    pub(crate) fn audit_ram(&self) -> Result<(), KvError> {
        let Some(ram) = &self.ram else {
            return Ok(());
        };
        let index = self.ancestry.as_ref().ok_or(KvError::InvariantViolation)?;
        let mut chained = 0usize;
        for (chain, keys) in &ram.chains {
            if keys.iter().any(|key| {
                ram.entries
                    .get(key)
                    .is_none_or(|entry| entry.chain != *chain)
            }) {
                return Err(KvError::InvariantViolation);
            }
            chained += keys.len();
        }
        if chained != ram.entries.len()
            || ram.order.len() != ram.entries.len()
            || ram.evictions.len() > PREFIX_EVICTION_LOG_V1
            || (ram.tier.is_none() && !ram.entries.is_empty())
            || ram
                .invalidations
                .keys()
                .any(|key| !ram.entries.contains_key(key))
        {
            return Err(KvError::InvariantViolation);
        }
        for (key, entry) in &ram.entries {
            let usable = !ram.invalidations.contains_key(key);
            // Every touch of a shadow's GPU anchor invalidates its entry, so
            // a usable shadow keeps the entry's last use; a key captured
            // again without the index has an unusable entry.
            let gpu_mismatch = self.anchors.get(key).is_some_and(|anchor| {
                if anchor.indexed {
                    anchor.tokens != entry.tokens
                        || (usable && !ram.forced && anchor.last_used != entry.last_used)
                } else {
                    usable
                }
            });
            if entry.tokens == 0
                || entry.serial == 0
                || entry.serial > ram.serial
                || entry.last_used > self.anchor_clock
                || !ram.order.contains(&(entry.last_used, *key))
                || gpu_mismatch
                || (usable && !ram.forced && index.has_descendant(*key))
            {
                return Err(KvError::InvariantViolation);
            }
        }
        // A promotion began from a ready, usable entry, and every logical
        // event on its key since then invalidated it with the entry.
        for ((key, serial), promotion) in &ram.promotions {
            let usable = promotion.invalidation.is_none();
            let entry = ram.entries.get(key).filter(|entry| entry.serial == *serial);
            if promotion.tokens == 0
                || *serial == 0
                || *serial > ram.serial
                || entry.is_some_and(|entry| {
                    entry.tokens != promotion.tokens
                        || !entry.ready
                        || (usable && ram.invalidations.contains_key(key))
                })
                || (usable && !ram.forced && index.has_descendant(*key))
            {
                return Err(KvError::InvariantViolation);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HostTier, KvGeometry, KvTransferStatus};

    fn demoted() -> (KvPool, HostTier, PrefixChainV1) {
        let chain = PrefixChainV1::new(4, (0..12).collect::<Vec<u32>>().into(), vec![]).unwrap();
        let mut pool = KvPool::new(KvGeometry::new(4, 16).unwrap(), 2)
            .unwrap()
            .with_prefix_ram(true);
        let mut tier = HostTier::new(&pool, 16).unwrap();
        let source = pool.create_sequence(12).unwrap();
        pool.append(source, 12).unwrap();
        pool.insert_content_anchor(&chain, source, 8, 1)
            .unwrap()
            .unwrap();
        let (_, plan) = tier
            .begin_offload(&mut pool, chain.key_at(8).unwrap(), None)
            .unwrap();
        tier.complete_transfer(
            &mut pool,
            plan.into_submission()
                .completion(KvTransferStatus::Complete),
        )
        .unwrap();
        // A descendant leaves the entry unusable until its drop.
        pool.insert_content_anchor(&chain, source, 12, 1)
            .unwrap()
            .unwrap();
        pool.release(source).unwrap();
        tier.audit(&pool).unwrap();
        (pool, tier, chain)
    }

    #[test]
    fn audits_reject_host_entries_out_of_step_with_the_index_or_the_tier() {
        type Tamper = fn(&mut PrefixRam, PrefixCacheKeyV1);
        let tampers: [(&str, Tamper); 6] = [
            ("ready", |ram, key| {
                ram.entries.get_mut(&key).unwrap().ready = false;
            }),
            ("tokens", |ram, key| {
                ram.entries.get_mut(&key).unwrap().tokens = 7;
            }),
            ("missing", |ram, key| {
                ram.entries.remove(&key);
                ram.order.clear();
                ram.chains.clear();
                ram.invalidations.clear();
            }),
            ("usable ancestor", |ram, _| ram.invalidations.clear()),
            ("unknown invalidation", |ram, _| {
                ram.invalidations.insert(
                    PrefixCacheKeyV1::new([3; 32]),
                    PrefixHostInvalidationV1::Resumed,
                );
            }),
            ("order", |ram, _| ram.order.clear()),
        ];
        for (name, tamper) in tampers {
            let (mut pool, tier, chain) = demoted();
            tamper(pool.ram.as_mut().unwrap(), chain.key_at(8).unwrap());
            assert_eq!(
                tier.audit(&pool),
                Err(KvError::InvariantViolation),
                "{name}"
            );
        }
    }

    #[test]
    fn a_logical_event_invalidates_a_promotion_after_its_entry_is_evicted() {
        use PrefixHostInvalidationV1::{Descendant, Recaptured};
        let chain = PrefixChainV1::new(4, (0..12).collect::<Vec<u32>>().into(), vec![]).unwrap();
        let key = chain.key_at(8).unwrap();
        let capture = |pool: &mut KvPool, tokens| {
            let source = pool.create_sequence(12).unwrap();
            pool.append(source, 12).unwrap();
            pool.insert_content_anchor(&chain, source, tokens, 1)
                .unwrap()
                .unwrap();
            pool.release(source).unwrap();
        };
        let residency = |pool: &KvPool| pool.ancestry.as_ref().unwrap().residency(key);
        for (case, forced, expected) in [
            ("untouched", false, None),
            ("descendant", false, Some(Descendant)),
            ("recapture", false, Some(Recaptured)),
            ("forced descendant", true, None),
        ] {
            let mut pool = KvPool::new(KvGeometry::new(4, 16).unwrap(), 2)
                .unwrap()
                .with_prefix_ram(true)
                .with_prefix_ram_forced(forced);
            let mut tier = HostTier::new(&pool, 16).unwrap();
            capture(&mut pool, 8);
            let (_, plan) = tier.begin_offload(&mut pool, key, None).unwrap();
            tier.complete_transfer(
                &mut pool,
                plan.into_submission()
                    .completion(KvTransferStatus::Complete),
            )
            .unwrap();
            assert!(pool.remove_anchor(key).unwrap());
            let serial = pool.begin_pending_promotion(key).unwrap();
            assert_eq!(serial, pool.host_entry(key).unwrap().serial);
            assert_eq!(pool.begin_pending_promotion(key), Err(KvError::StaleHandle));
            tier.audit(&pool).unwrap();
            // The promotion holds the node through its entry's host eviction,
            // and configuration may not replace it.
            assert!(tier.drop_public_entry(&mut pool, key).unwrap());
            let pending = PrefixResidency {
                pending: 1,
                ..PrefixResidency::default()
            };
            assert_eq!(residency(&pool), Some(pending), "{case}");
            tier.audit(&pool).unwrap();
            let copy = pool.clone();
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    copy.with_prefix_index(true)
                }))
                .is_err(),
                "{case}"
            );
            match case {
                "untouched" => {}
                "recapture" => capture(&mut pool, 8),
                _ => capture(&mut pool, 12),
            }
            tier.audit(&pool).unwrap();
            if case == "descendant" {
                // A usable promotion below a resident descendant is refused.
                let mut copy = pool.clone();
                let ram = copy.ram.as_mut().unwrap();
                ram.promotions.get_mut(&(key, serial)).unwrap().invalidation = None;
                assert_eq!(copy.audit(), Err(KvError::InvariantViolation));
            }
            assert_eq!(
                pool.end_pending_promotion(key, serial),
                Ok(expected),
                "{case}"
            );
            assert_eq!(
                pool.end_pending_promotion(key, serial),
                Err(KvError::StaleHandle)
            );
            // The node stays only while the GPU holds the key.
            let gpu = (case == "recapture").then_some(PrefixResidency::GPU);
            assert_eq!(residency(&pool), gpu, "{case}");
            tier.audit(&pool).unwrap();
        }
    }

    #[test]
    fn public_drops_release_only_the_public_anchor_of_the_entrys_key() {
        for case in ["owned", "other key"] {
            for drop in ["drop", "drop invalidated"] {
                let (mut pool, mut tier, chain) = demoted();
                let key = chain.key_at(8).unwrap();
                // A second anchor in the same tier: an owned swap, or the
                // public entry of another key.
                let source = pool.create_sequence(4).unwrap();
                pool.append(source, 4).unwrap();
                let other = if case == "owned" {
                    let private = PrefixCacheKeyV1::new([9; 32]);
                    pool.insert_private_anchor(private, source, 1)
                        .unwrap()
                        .unwrap();
                    tier.begin_offload(&mut pool, private, Some(1))
                } else {
                    let prompt = PrefixChainV1::new(4, vec![7u32; 4].into(), vec![]).unwrap();
                    pool.insert_content_anchor(&prompt, source, 4, 1)
                        .unwrap()
                        .unwrap();
                    tier.begin_offload(&mut pool, prompt.key_at(4).unwrap(), None)
                };
                pool.release(source).unwrap();
                let (other, plan) = other.unwrap();
                tier.complete_transfer(
                    &mut pool,
                    plan.into_submission()
                        .completion(KvTransferStatus::Complete),
                )
                .unwrap();
                pool.ram
                    .as_mut()
                    .unwrap()
                    .entries
                    .get_mut(&key)
                    .unwrap()
                    .host = other;
                let before = pool.clone();
                let result = if drop == "drop" {
                    tier.drop_public_entry(&mut pool, key).map(|_| ())
                } else {
                    tier.drop_invalidated_entries(&mut pool).map(|_| ())
                };
                assert_eq!(result, Err(KvError::InvariantViolation), "{case} {drop}");
                assert_eq!(pool, before, "{case} {drop}");
                assert!(tier.anchor(other).is_some(), "{case} {drop}");
            }
        }
    }
}
