//! Pending anchors: GPU pages a promotion fills from a public host entry
//! before the pool publishes them as a normal public anchor. A pending
//! anchor holds its pages with plain references, so it is invisible to
//! lookup and eviction and its pages count as used, not cached. It holds
//! its key's ancestry node until it is published or abandoned, through its
//! entry's host eviction, and a logical event on the key invalidates it as
//! it does the entry.
use std::ops::RangeInclusive;
use std::sync::atomic::{AtomicU64, Ordering};

use super::PrefixRam;
use crate::ancestry::{PrefixAncestry, PrefixTier};
use crate::{
    BlockId, KvError, KvPool, KvTransferId, PrefixAnchorClassV1, PrefixAnchorV1, PrefixCacheKeyV1,
    PrefixChainV1, PrefixEvictionTierV1, PrefixHostInvalidationV1,
};

static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

/// Names one pending anchor: the key it promotes, the serial of the host
/// entry it copies, and a generation no other pending anchor in the process
/// shares, so a retry of the same entry never answers to an earlier
/// attempt's id. An entry has at most one pending anchor at a time.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PendingAnchorId {
    key: PrefixCacheKeyV1,
    serial: u64,
    generation: u64,
}

impl PendingAnchorId {
    #[must_use]
    pub const fn key(self) -> PrefixCacheKeyV1 {
        self.key
    }

    /// The serial of the host entry it copies.
    #[must_use]
    pub const fn serial(self) -> u64 {
        self.serial
    }

    /// Every pending anchor of `key` whose entry serial is in `serials`.
    pub(super) fn of(key: PrefixCacheKeyV1, serials: RangeInclusive<u64>) -> RangeInclusive<Self> {
        Self {
            key,
            serial: *serials.start(),
            generation: 0,
        }..=Self {
            key,
            serial: *serials.end(),
            generation: u64::MAX,
        }
    }
}

/// How far a pending anchor's copy from the host has come.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefixPendingCopyV1 {
    /// No promotion has started.
    Waiting,
    /// A promotion's transfer is filling the pages.
    InFlight(KvTransferId),
    /// The copy completed on every rank.
    Copied,
    /// The copy failed idle; the pages hold nothing usable.
    Failed,
}

/// A pending anchor as the scheduler sees it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefixPendingAnchorV1 {
    pub tokens: usize,
    /// KV and snapshot pages it holds until it is published or abandoned.
    pub pages: usize,
    pub copy: PrefixPendingCopyV1,
    /// The first logical event on its key since it was created, which
    /// forbids its publish.
    pub invalidation: Option<PrefixHostInvalidationV1>,
}

/// What [`KvPool::publish_pending_anchor`] did with a pending anchor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefixPublishV1 {
    /// Published as a normal public anchor.
    Published,
    /// Abandoned: its copy failed or never started.
    NotCopied,
    /// Abandoned: a logical event on its key invalidated it.
    Invalidated(PrefixHostInvalidationV1),
    /// Abandoned: its key has a GPU anchor already.
    GpuResident,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PendingAnchor {
    chain: PrefixCacheKeyV1,
    /// The entry's tokens, which stay the node's depth after its drop.
    pub(super) tokens: usize,
    blocks: Vec<BlockId>,
    state: Vec<BlockId>,
    copy: PrefixPendingCopyV1,
    pub(super) invalidation: Option<PrefixHostInvalidationV1>,
}

impl KvPool {
    /// Allocates a pending anchor for a promotion of `key`'s ready, usable
    /// host entry, which gives its chain and tokens: `blocks_for(tokens)`
    /// KV pages and `state_pages` snapshot pages. It checks fit first,
    /// against free pages plus the pages evicting every evictable anchor
    /// frees, and fails with [`KvError::PoolExhausted`] before evicting
    /// anything. A key with a GPU anchor is refused, and so is a second
    /// pending anchor of one entry: followers wait for its publish.
    pub fn create_pending_anchor(
        &mut self,
        key: PrefixCacheKeyV1,
        state_pages: usize,
    ) -> Result<PendingAnchorId, KvError> {
        self.ensure_no_prepared()?;
        let ram = self.ram.as_ref().ok_or(KvError::StaleHandle)?;
        let entry = ram
            .entries
            .get(&key)
            .filter(|entry| entry.ready && !ram.invalidations.contains_key(&key))
            .ok_or(KvError::StaleHandle)?;
        let serial = entry.serial;
        if ram
            .pending
            .range(PendingAnchorId::of(key, serial..=serial))
            .next()
            .is_some()
            || self.anchors.contains_key(&key)
        {
            return Err(KvError::StaleHandle);
        }
        let (chain, tokens) = (entry.chain, entry.tokens);
        let count = self.geometry.blocks_for(tokens)?;
        let total = count
            .checked_add(state_pages)
            .ok_or(KvError::TokenCountOverflow)?;
        // Anchor-only pages are exactly those evicting every evictable
        // anchor frees: a lease, an in-flight demotion's included, adds a
        // transfer reference to each page it holds.
        let free = self.free_blocks.len();
        if total > free + self.anchor_only_pages {
            return Err(KvError::PoolExhausted {
                needed: total,
                free,
            });
        }
        let generation = NEXT_GENERATION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| KvError::SequenceGenerationExhausted)?;
        let id = PendingAnchorId {
            key,
            serial,
            generation,
        };
        self.evict_anchors_for(total);
        if total > self.free_blocks.len() {
            return Err(KvError::PoolExhausted {
                needed: total,
                free: self.free_blocks.len(),
            });
        }
        self.check_serials(total)?;
        let mut blocks = Vec::with_capacity(total);
        for _ in 0..total {
            let page = self.free_blocks.pop().expect("pages counted");
            self.allocation_serials[page.index()] += 1;
            self.set_block_references(page, 1, 0, 0);
            blocks.push(page);
        }
        let state = blocks.split_off(count);
        self.ram.as_mut().expect("tier checked").pending.insert(
            id,
            PendingAnchor {
                chain,
                tokens,
                blocks,
                state,
                copy: PrefixPendingCopyV1::Waiting,
                invalidation: None,
            },
        );
        self.ancestry
            .as_mut()
            .expect("the RAM tier keeps the index")
            .add_residency(key, PrefixTier::Pending);
        Ok(id)
    }

    /// Publishes a pending anchor whose copy completed as a normal public
    /// anchor over `identity`, which must name its key, unless a logical
    /// event invalidated it or its key has a GPU anchor; otherwise it is
    /// abandoned. Its node takes the GPU hold before the pending hold goes,
    /// so it keeps its lifetime flags, and a usable shadow left in the host
    /// tier takes the anchor's last use. Refused while its copy is in
    /// flight.
    pub fn publish_pending_anchor(
        &mut self,
        id: PendingAnchorId,
        identity: &PrefixChainV1,
    ) -> Result<PrefixPublishV1, KvError> {
        self.ensure_no_prepared()?;
        let pending = self
            .ram
            .as_ref()
            .and_then(|ram| ram.pending.get(&id))
            .ok_or(KvError::StaleHandle)?;
        let (tokens, chain) = (pending.tokens, pending.chain);
        let outcome = match (pending.copy, pending.invalidation) {
            (PrefixPendingCopyV1::InFlight(_), _) => return Err(KvError::StaleHandle),
            (PrefixPendingCopyV1::Waiting | PrefixPendingCopyV1::Failed, _) => {
                PrefixPublishV1::NotCopied
            }
            (PrefixPendingCopyV1::Copied, Some(reason)) => PrefixPublishV1::Invalidated(reason),
            (PrefixPendingCopyV1::Copied, None) if self.anchors.contains_key(&id.key) => {
                PrefixPublishV1::GpuResident
            }
            (PrefixPendingCopyV1::Copied, None) => PrefixPublishV1::Published,
        };
        if outcome != PrefixPublishV1::Published {
            self.release_pending(id);
            return Ok(outcome);
        }
        if identity.block_tokens() != self.geometry.block_tokens
            || identity.splits_media(tokens)
            || identity.key_at(tokens) != Some(id.key)
            || identity.chain(tokens / identity.block_tokens()) != Some(chain)
        {
            return Err(KvError::StaleHandle);
        }
        let ram = self.ram.as_mut().expect("pending anchor checked");
        let pending = ram.pending.remove(&id).expect("pending anchor checked");
        // Each page's pending reference becomes its anchor reference.
        for page in pending.blocks.iter().chain(&pending.state) {
            self.set_block_references(
                *page,
                self.blocks[page.index()],
                self.transfer_refs[page.index()],
                self.anchor_refs[page.index()] + 1,
            );
        }
        self.anchor_page_work += (pending.blocks.len() + pending.state.len()) as u64;
        self.anchor_chains
            .entry(pending.chain)
            .or_default()
            .insert(id.key);
        self.anchor_clock += 1;
        self.anchor_serial += 1;
        self.anchor_order.insert(
            (PrefixEvictionTierV1::Live, self.anchor_clock, id.key),
            id.key,
        );
        self.metrics
            .anchor_added(PrefixAnchorClassV1::Maximal, pending.state.len(), false);
        self.anchors.insert(
            id.key,
            PrefixAnchorV1 {
                chain: pending.chain,
                tokens,
                blocks: pending.blocks,
                state: pending.state,
                last_used: self.anchor_clock,
                created_at: std::time::Instant::now(),
                class: PrefixAnchorClassV1::Maximal,
                tier: PrefixEvictionTierV1::Live,
                serial: self.anchor_serial,
                private: false,
                indexed: true,
            },
        );
        let index = self
            .ancestry
            .as_mut()
            .expect("the RAM tier keeps the index");
        index.insert(identity, tokens, id.key);
        index.release_residency(id.key, PrefixTier::Pending);
        self.reclassify_anchors();
        let last_used = self.anchor_clock;
        let ram = self.ram.as_mut().expect("pending anchor checked");
        if !ram.invalidations.contains_key(&id.key)
            && let Some(entry) = ram.entries.get_mut(&id.key)
        {
            ram.order.remove(&(entry.last_used, id.key));
            entry.last_used = last_used;
            ram.order.insert((last_used, id.key));
        }
        Ok(PrefixPublishV1::Published)
    }

    /// Releases a pending anchor's pages and its hold on its key's node.
    /// Refused while its copy is in flight: the pages are released only
    /// after a confirmed completion, so an unconfirmed copy keeps them.
    pub fn abandon_pending_anchor(&mut self, id: PendingAnchorId) -> Result<(), KvError> {
        self.ensure_no_prepared()?;
        match self.pending_anchor(id).map(|pending| pending.copy) {
            None | Some(PrefixPendingCopyV1::InFlight(_)) => Err(KvError::StaleHandle),
            Some(_) => {
                self.release_pending(id);
                Ok(())
            }
        }
    }

    #[must_use]
    pub fn pending_anchor(&self, id: PendingAnchorId) -> Option<PrefixPendingAnchorV1> {
        self.ram
            .as_ref()?
            .pending
            .get(&id)
            .map(|pending| PrefixPendingAnchorV1 {
                tokens: pending.tokens,
                pages: pending.blocks.len() + pending.state.len(),
                copy: pending.copy,
                invalidation: pending.invalidation,
            })
    }

    fn release_pending(&mut self, id: PendingAnchorId) {
        let pending = self
            .ram
            .as_mut()
            .and_then(|ram| ram.pending.remove(&id))
            .expect("pending anchor checked");
        for page in pending
            .state
            .iter()
            .rev()
            .chain(pending.blocks.iter().rev())
        {
            self.release_block(*page);
        }
        if let Some(index) = &mut self.ancestry {
            index.release_residency(id.key, PrefixTier::Pending);
            self.reclassify_anchors();
        }
    }

    /// The KV and snapshot pages a promotion fills: those of a pending
    /// anchor no copy has started and no logical event invalidated.
    pub(crate) fn pending_copy_pages(
        &self,
        id: PendingAnchorId,
    ) -> Result<(Vec<BlockId>, Vec<BlockId>), KvError> {
        let pending = self
            .ram
            .as_ref()
            .and_then(|ram| ram.pending.get(&id))
            .filter(|pending| {
                pending.copy == PrefixPendingCopyV1::Waiting && pending.invalidation.is_none()
            })
            .ok_or(KvError::StaleHandle)?;
        Ok((pending.blocks.clone(), pending.state.clone()))
    }

    /// Records a checked promotion's transfer.
    pub(crate) fn start_pending_copy(&mut self, id: PendingAnchorId, transfer: KvTransferId) {
        self.ram
            .as_mut()
            .and_then(|ram| ram.pending.get_mut(&id))
            .expect("a checked promotion")
            .copy = PrefixPendingCopyV1::InFlight(transfer);
    }

    /// Ends a promotion's copy: a confirmed success makes it publishable.
    pub(crate) fn finish_pending_copy(
        &mut self,
        id: PendingAnchorId,
        transfer: KvTransferId,
        success: bool,
    ) {
        if let Some(pending) = self.ram.as_mut().and_then(|ram| ram.pending.get_mut(&id))
            && pending.copy == PrefixPendingCopyV1::InFlight(transfer)
        {
            pending.copy = if success {
                PrefixPendingCopyV1::Copied
            } else {
                PrefixPendingCopyV1::Failed
            };
        }
    }

    /// Every page a pending anchor holds.
    pub(crate) fn pending_pages(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.ram
            .iter()
            .flat_map(|ram| ram.pending.values())
            .flat_map(|pending| pending.blocks.iter().chain(&pending.state).copied())
    }

    /// Every pending anchor whose copy is in flight, with its transfer.
    pub(crate) fn pending_copies(
        &self,
    ) -> impl Iterator<Item = (PendingAnchorId, KvTransferId)> + '_ {
        self.ram
            .iter()
            .flat_map(|ram| ram.pending.iter())
            .filter_map(|(id, pending)| match pending.copy {
                PrefixPendingCopyV1::InFlight(transfer) => Some((*id, transfer)),
                _ => None,
            })
    }

    /// Each pending anchor began from a ready, usable entry, and every
    /// logical event on its key since then invalidated it with the entry.
    /// Its pages carry only its own reference and, during its copy, the
    /// copy's lease.
    pub(super) fn audit_pending(
        &self,
        ram: &PrefixRam,
        index: &PrefixAncestry,
    ) -> Result<(), KvError> {
        let mut previous = None;
        for (id, pending) in &ram.pending {
            // One pending anchor per entry.
            if previous.replace((id.key, id.serial)) == Some((id.key, id.serial)) {
                return Err(KvError::InvariantViolation);
            }
            let usable = pending.invalidation.is_none();
            let copying = u32::from(matches!(pending.copy, PrefixPendingCopyV1::InFlight(_)));
            let entry = ram
                .entries
                .get(&id.key)
                .filter(|entry| entry.serial == id.serial);
            if pending.tokens == 0
                || id.serial == 0
                || id.serial > ram.serial
                || self.geometry.blocks_for(pending.tokens)? != pending.blocks.len()
                || entry.is_some_and(|entry| {
                    entry.tokens != pending.tokens
                        || entry.chain != pending.chain
                        || !entry.ready
                        || (usable && ram.invalidations.contains_key(&id.key))
                })
                || (usable && !ram.forced && index.has_descendant(id.key))
                || pending.blocks.iter().chain(&pending.state).any(|page| {
                    let page = page.index();
                    page >= self.blocks.len()
                        || self.anchor_refs[page] != 0
                        || self.transfer_refs[page] != copying
                        || self.blocks[page] != 1 + copying
                })
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

    fn prompt() -> PrefixChainV1 {
        PrefixChainV1::new(4, (0..24).collect::<Vec<u32>>().into(), vec![]).unwrap()
    }

    fn capture(pool: &mut KvPool, chain: &PrefixChainV1, depths: &[usize]) {
        let tokens = *depths.iter().max().unwrap();
        let source = pool.create_sequence(tokens).unwrap();
        pool.append(source, tokens).unwrap();
        for depth in depths {
            pool.insert_content_anchor(chain, source, *depth, 1)
                .unwrap()
                .unwrap();
        }
        pool.release(source).unwrap();
    }

    fn complete(pool: &mut KvPool, tier: &mut HostTier, plan: crate::KvTransferPlan) {
        tier.complete_transfer(
            pool,
            plan.into_submission()
                .completion(KvTransferStatus::Complete),
        )
        .unwrap();
        tier.audit(pool).unwrap();
    }

    /// Moves `key`'s GPU anchor to the host tier alone.
    fn demote(pool: &mut KvPool, tier: &mut HostTier, key: PrefixCacheKeyV1) {
        let (_, plan) = tier.begin_offload(pool, key, None).unwrap();
        complete(pool, tier, plan);
        assert!(pool.remove_anchor(key).unwrap());
        tier.audit(pool).unwrap();
    }

    fn ram_pool(pages: usize) -> KvPool {
        KvPool::new(KvGeometry::new(4, pages).unwrap(), 4)
            .unwrap()
            .with_prefix_ram(true)
    }

    #[test]
    fn pending_pages_are_used_and_invisible_to_lookup_and_eviction() {
        let chain = prompt();
        let key = chain.key_at(8).unwrap();
        let mut pool = ram_pool(8);
        let mut tier = HostTier::new(&pool, 8).unwrap();
        capture(&mut pool, &chain, &[8]);
        demote(&mut pool, &mut tier, key);
        let pending = pool.create_pending_anchor(key, 1).unwrap();
        tier.audit(&pool).unwrap();
        let stats = pool.stats();
        assert_eq!((stats.free_blocks(), stats.cached_blocks()), (5, 0));
        assert_eq!(stats.used_blocks(), 3);
        assert!(!pool.has_anchor(key) && pool.lookup_anchor(&chain, 24).is_none());
        // Allocation pressure cannot take its pages.
        let handle = pool.create_sequence(24).unwrap();
        assert!(matches!(
            pool.append(handle, 24),
            Err(KvError::PoolExhausted { .. })
        ));
        pool.release(handle).unwrap();
        assert_eq!(pool.pending_anchor(pending).unwrap().pages, 3);
        pool.abandon_pending_anchor(pending).unwrap();
        tier.audit(&pool).unwrap();
        assert_eq!(pool.stats().free_blocks(), 8);
    }

    #[test]
    fn creation_checks_fit_before_evicting_anything() {
        let chain = prompt();
        let key = chain.key_at(8).unwrap();
        let mut pool = ram_pool(8);
        let mut tier = HostTier::new(&pool, 8).unwrap();
        capture(&mut pool, &chain, &[8]);
        demote(&mut pool, &mut tier, key);
        // Two anchors of three pages leave two pages free; a live sequence
        // shares the first anchor's pages, so its eviction frees only its
        // snapshot page.
        let other = PrefixChainV1::new(4, vec![7u32; 8].into(), vec![]).unwrap();
        let live = pool.create_sequence(8).unwrap();
        pool.append(live, 8).unwrap();
        pool.insert_content_anchor(&other, live, 8, 1)
            .unwrap()
            .unwrap();
        capture(
            &mut pool,
            &PrefixChainV1::new(4, vec![9u32; 8].into(), vec![]).unwrap(),
            &[8],
        );
        tier.audit(&pool).unwrap();
        assert_eq!(pool.stats().free_blocks() + pool.stats().cached_blocks(), 6);
        let before = pool.clone();
        assert_eq!(
            pool.create_pending_anchor(key, 5),
            Err(KvError::PoolExhausted { needed: 7, free: 2 })
        );
        assert_eq!(pool, before);
        let pending = pool.create_pending_anchor(key, 4).unwrap();
        assert_eq!(pool.stats().free_blocks(), 0);
        assert_eq!(pool.anchor_count(), 0);
        tier.audit(&pool).unwrap();
        pool.abandon_pending_anchor(pending).unwrap();
        pool.release(live).unwrap();
        tier.audit(&pool).unwrap();
    }

    #[test]
    fn publish_keeps_the_lifetime_flags_of_a_node_only_it_held() {
        let chain = prompt();
        let key = |depth| chain.key_at(depth).unwrap();
        let mut pool = ram_pool(32).with_prefix_ancestry(true);
        let mut tier = HostTier::new(&pool, 16).unwrap();
        // The frontier is extended, so the ancestor below it is consumed.
        capture(&mut pool, &chain, &[4, 20, 24]);
        let hit = pool.resume_anchor(key(24), 24).unwrap();
        pool.finish_restore(hit.handle()).unwrap();
        pool.release(hit.handle()).unwrap();
        demote(&mut pool, &mut tier, key(24));
        let pending = pool.create_pending_anchor(key(24), 1).unwrap();
        let plan = tier.begin_promotion(&mut pool, pending).unwrap();
        // The host tier drops the source entry; only the pending anchor
        // holds the node now.
        assert!(tier.drop_public_entry(&mut pool, key(24)).unwrap());
        complete(&mut pool, &mut tier, plan);
        assert_eq!(
            pool.publish_pending_anchor(pending, &chain),
            Ok(PrefixPublishV1::Published)
        );
        tier.audit(&pool).unwrap();
        // A new anchor between them learns that the frontier is still
        // extended, and ranks below recency.
        capture(&mut pool, &chain, &[22]);
        assert_ne!(pool.anchor_tier(key(22)), Some(PrefixEvictionTierV1::Live));
        tier.audit(&pool).unwrap();
    }

    #[test]
    fn publish_checks_its_identity_and_refuses_a_gpu_resident_key() {
        let chain = prompt();
        let key = chain.key_at(8).unwrap();
        for case in ["identity", "gpu resident"] {
            let mut pool = ram_pool(16);
            let mut tier = HostTier::new(&pool, 8).unwrap();
            capture(&mut pool, &chain, &[8]);
            demote(&mut pool, &mut tier, key);
            let pending = pool.create_pending_anchor(key, 1).unwrap();
            let plan = tier.begin_promotion(&mut pool, pending).unwrap();
            complete(&mut pool, &mut tier, plan);
            if case == "identity" {
                let before = pool.clone();
                let other = PrefixChainV1::new(4, vec![1u32; 8].into(), vec![]).unwrap();
                assert_eq!(
                    pool.publish_pending_anchor(pending, &other),
                    Err(KvError::StaleHandle)
                );
                assert_eq!(pool, before);
                continue;
            }
            // A recapture always invalidates, so clear the flag to reach
            // the GPU anchor check.
            capture(&mut pool, &chain, &[8]);
            pool.ram
                .as_mut()
                .unwrap()
                .pending
                .get_mut(&pending)
                .unwrap()
                .invalidation = None;
            assert_eq!(
                pool.publish_pending_anchor(pending, &chain),
                Ok(PrefixPublishV1::GpuResident)
            );
            assert!(pool.pending_anchor(pending).is_none());
        }
    }

    #[test]
    fn audits_reject_pending_anchors_out_of_step() {
        type Tamper = fn(&mut KvPool, PendingAnchorId, KvTransferId);
        let tampers: [(&str, Tamper); 4] = [
            ("tokens", |pool, id, _| {
                pool.ram
                    .as_mut()
                    .unwrap()
                    .pending
                    .get_mut(&id)
                    .unwrap()
                    .tokens = 4;
            }),
            ("anchor reference", |pool, id, _| {
                let page = pool.ram.as_ref().unwrap().pending[&id].blocks[0];
                let refs = pool.blocks[page.index()];
                pool.set_block_references(page, refs, 0, 1);
            }),
            ("copy without a transfer", |pool, id, finished| {
                pool.start_pending_copy(id, finished);
            }),
            ("serial", |pool, id, _| {
                let mut pending = pool.ram.as_mut().unwrap().pending.remove(&id).unwrap();
                pending.copy = PrefixPendingCopyV1::Waiting;
                pool.ram.as_mut().unwrap().pending.insert(
                    PendingAnchorId {
                        serial: u64::MAX,
                        ..id
                    },
                    pending,
                );
            }),
        ];
        let chain = prompt();
        let key = chain.key_at(8).unwrap();
        for (name, tamper) in tampers {
            let mut pool = ram_pool(16);
            let mut tier = HostTier::new(&pool, 8).unwrap();
            capture(&mut pool, &chain, &[8]);
            let (_, plan) = tier.begin_offload(&mut pool, key, None).unwrap();
            let finished = plan.id();
            complete(&mut pool, &mut tier, plan);
            assert!(pool.remove_anchor(key).unwrap());
            let pending = pool.create_pending_anchor(key, 1).unwrap();
            tier.audit(&pool).unwrap();
            tamper(&mut pool, pending, finished);
            assert_eq!(
                tier.audit(&pool),
                Err(KvError::InvariantViolation),
                "{name}"
            );
        }
    }
}
