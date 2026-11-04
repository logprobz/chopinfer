//! Model-neutral host page ownership. Completion is supplied by the executor,
//! only after every rank event completed or every copy stream was fenced.
use super::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static NEXT_TRANSFER: AtomicU64 = AtomicU64::new(1);
static NEXT_TIER: AtomicU64 = AtomicU64::new(1);
fn next_transfer() -> Result<KvTransferId, KvError> {
    NEXT_TRANSFER
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map(KvTransferId)
        .map_err(|_| KvError::SequenceGenerationExhausted)
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct DevicePageAllocation {
    pub page: BlockId,
    pub serial: u64,
}

#[derive(Debug)]
pub struct DevicePageLease {
    identity: Arc<()>,
    id: u64,
}

impl KvPool {
    pub(super) fn check_serials(&self, count: usize) -> Result<(), KvError> {
        if self
            .free_blocks
            .iter()
            .rev()
            .take(count)
            .any(|p| self.allocation_serials[p.index()] == u64::MAX)
        {
            return Err(KvError::SequenceGenerationExhausted);
        }
        Ok(())
    }

    pub fn page_allocation(&self, page: BlockId) -> Result<DevicePageAllocation, KvError> {
        if self.block_ref_count(page)? == 0 {
            return Err(KvError::StaleHandle);
        }
        Ok(DevicePageAllocation {
            page,
            serial: self.allocation_serials[page.index()],
        })
    }

    /// Retains each distinct allocation once. References force append COW and
    /// survive sequence retirement and anchor eviction. No transaction may be open.
    pub fn retain_transfer_pages(
        &mut self,
        pages: &[DevicePageAllocation],
    ) -> Result<DevicePageLease, KvError> {
        self.ensure_no_prepared()?;
        let next = self
            .next_transfer
            .checked_add(1)
            .ok_or(KvError::SequenceGenerationExhausted)?;
        let mut unique = std::collections::BTreeSet::new();
        for allocation in pages {
            if self.page_allocation(allocation.page)? != *allocation {
                return Err(KvError::StaleHandle);
            }
            if self.blocks[allocation.page.index()] == u32::MAX {
                return Err(KvError::ReferenceCountOverflow {
                    block: allocation.page,
                });
            }
            unique.insert(allocation.page);
        }
        let pages: Vec<_> = unique.into_iter().collect();
        for page in &pages {
            self.set_block_references(
                *page,
                self.blocks[page.index()] + 1,
                self.transfer_refs[page.index()] + 1,
                self.anchor_refs[page.index()],
            );
        }
        let id = self.next_transfer;
        self.next_transfer = next;
        self.transfer_pages.insert(id, pages);
        Ok(DevicePageLease {
            identity: Arc::clone(&self.identity),
            id,
        })
    }

    /// Call only after confirmed DMA completion. An unconfirmed transfer keeps
    /// this lease, even after cancellation; dropping the token does not free pages.
    pub fn release_transfer_pages(&mut self, lease: &DevicePageLease) -> Result<(), KvError> {
        self.ensure_no_prepared()?;
        if !Arc::ptr_eq(&lease.identity, &self.identity) {
            return Err(KvError::StaleHandle);
        }
        let pages = self
            .transfer_pages
            .remove(&lease.id)
            .ok_or(KvError::StaleHandle)?;
        for page in pages.into_iter().rev() {
            self.set_block_references(
                page,
                self.blocks[page.index()] - 1,
                self.transfer_refs[page.index()] - 1,
                self.anchor_refs[page.index()],
            );
            if self.blocks[page.index()] == 0 {
                self.free_blocks.push(page);
            }
        }
        Ok(())
    }

    /// Snapshot an exact anchor for a copy, without changing its LRU position.
    pub fn anchor_pages(
        &self,
        key: PrefixCacheKeyV1,
    ) -> Option<(usize, Vec<BlockId>, Vec<BlockId>)> {
        self.anchors
            .get(&key)
            .map(|a| (a.tokens, a.blocks.clone(), a.state.clone()))
    }

    /// Private keys must be unique to the request/capture, rather than a public
    /// prefix digest. They remain LRU evictable but cannot be prompt-cache hits.
    pub fn insert_private_anchor(
        &mut self,
        key: PrefixCacheKeyV1,
        source: SequenceHandle,
        state_pages: usize,
    ) -> Result<Option<Vec<BlockId>>, KvError> {
        if self.anchors.contains_key(&key) {
            return Err(KvError::StaleHandle);
        }
        self.insert_anchor_inner(key, key, source, None, state_pages, true, None)
    }

    /// Allocates committed KV and restore staging pages under the caller's
    /// admission reservation. No page-zero work is returned. Do not schedule
    /// the sequence until reload and bundle restore have both completed.
    pub fn create_initialized_sequence(
        &mut self,
        tokens: usize,
        quota: usize,
        state_pages: usize,
    ) -> Result<PrefixHitV1, KvError> {
        self.ensure_no_prepared()?;
        if tokens == 0 {
            return Err(KvError::ZeroPrefixTokens);
        }
        if tokens > quota {
            return Err(KvError::QuotaExceeded {
                quota,
                requested: tokens,
            });
        }
        let count = self.geometry.blocks_for(tokens)?;
        let total = count
            .checked_add(state_pages)
            .ok_or(KvError::TokenCountOverflow)?;
        self.evict_anchors_for(total);
        if total > self.free_blocks.len() {
            return Err(KvError::PoolExhausted {
                needed: total,
                free: self.free_blocks.len(),
            });
        }
        self.check_serials(total)?;
        let handle = self.create_sequence(quota)?;
        let mut pages = Vec::with_capacity(total);
        for _ in 0..total {
            let page = self.free_blocks.pop().expect("pages counted");
            self.allocation_serials[page.index()] += 1;
            self.set_block_references(page, 1, 0, 0);
            pages.push(page);
        }
        let state = pages.split_off(count);
        let sequence = &mut self.sequences[handle.slot];
        sequence.blocks = Arc::new(pages);
        sequence.pinned = state.clone();
        sequence.tokens = tokens;
        sequence.restoring = Some(false);
        self.metrics.appended(tokens);
        Ok(PrefixHitV1 {
            handle,
            tokens,
            state,
        })
    }

    /// Releases reload staging without counting a prefix-cache hit.
    pub fn finish_initialized_restore(&mut self, handle: SequenceHandle) -> Result<(), KvError> {
        self.ensure_no_prepared()?;
        let slot = self.validate(handle)?;
        if self.sequences[slot].restoring == Some(false) {
            return Err(KvError::StaleHandle);
        }
        self.sequences[slot].restoring = None;
        for page in std::mem::take(&mut self.sequences[slot].pinned)
            .into_iter()
            .rev()
        {
            self.release_block(page);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct HostPageId(usize);
impl HostPageId {
    pub const fn index(self) -> usize {
        self.0
    }
}
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct HostAnchorId(u64);
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct KvTransferId(u64);
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvTransferDirection {
    Out,
    In,
}
/// What a transfer is for, which orders the copy queue.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KvTransferClass {
    /// A public host entry copied back into a pending GPU anchor.
    Promotion,
    /// A swapped request's own boundary copied back for its restore.
    SwapIn,
    /// A preempted request's own boundary copied to the host.
    SwapOut,
    /// A public GPU anchor copied ahead of its eviction; never owned.
    Demotion,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvPagePair {
    pub device: BlockId,
    pub host: HostPageId,
}
#[derive(Clone, Debug)]
pub struct HostAnchor {
    pub key: PrefixCacheKeyV1,
    pub tokens: usize,
    pub blocks: Vec<HostPageId>,
    pub state: Vec<HostPageId>,
    /// The request a swap belongs to; a public entry has none.
    pub owner: Option<u64>,
    pub ready: bool,
    last_used: u64,
}
#[derive(Debug)]
struct Transfer {
    lease: DevicePageLease,
    hosts: Vec<HostPageId>,
    anchor: HostAnchorId,
    key: PrefixCacheKeyV1,
    direction: KvTransferDirection,
    class: KvTransferClass,
    quarantined: bool,
    receipt: Arc<AtomicBool>,
    destination: Option<SequenceHandle>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvTransferStatus {
    Pending,
    Complete,
    FailedIdle,
    Unconfirmed,
}

/// A single-use leased plan. Only the executor consumes it into a submission.
#[derive(Debug)]
pub struct KvTransferPlan {
    id: KvTransferId,
    direction: KvTransferDirection,
    class: KvTransferClass,
    pages: Vec<KvPagePair>,
    receipt: Arc<AtomicBool>,
}
impl KvTransferPlan {
    pub fn id(&self) -> KvTransferId {
        self.id
    }
    pub fn direction(&self) -> KvTransferDirection {
        self.direction
    }
    pub fn class(&self) -> KvTransferClass {
        self.class
    }
    pub fn pages(&self) -> &[KvPagePair] {
        &self.pages
    }
    pub fn into_submission(self) -> KvTransferSubmission {
        KvTransferSubmission(self)
    }
    /// Safe rollback of a rejected plan that never reached the copy queue.
    pub fn cancel(self) -> KvTransferCompletion {
        self.into_submission()
            .completion(KvTransferStatus::FailedIdle)
    }
}
/// Owned by the executor, never retained by the engine alongside a submission.
#[derive(Debug)]
pub struct KvTransferSubmission(KvTransferPlan);
impl KvTransferSubmission {
    pub fn id(&self) -> KvTransferId {
        self.0.id
    }
    pub fn direction(&self) -> KvTransferDirection {
        self.0.direction
    }
    pub fn class(&self) -> KvTransferClass {
        self.0.class
    }
    pub fn pages(&self) -> &[KvPagePair] {
        &self.0.pages
    }
    pub fn accepted(&self) -> bool {
        self.0.receipt.load(Ordering::Acquire)
    }
    /// The executor calls this only after querying every rank or fencing all
    /// streams. Pending work cannot produce a completion. Repeat polls may
    /// reissue a token until the tier accepts it; only one can be accepted.
    pub fn completion(&self, status: KvTransferStatus) -> KvTransferCompletion {
        assert_ne!(status, KvTransferStatus::Pending);
        KvTransferCompletion {
            id: self.id(),
            status,
            receipt: Arc::clone(&self.0.receipt),
        }
    }
}
#[derive(Debug)]
pub struct KvTransferCompletion {
    id: KvTransferId,
    status: KvTransferStatus,
    receipt: Arc<AtomicBool>,
}
impl KvTransferCompletion {
    pub fn id(&self) -> KvTransferId {
        self.id
    }
    pub fn status(&self) -> KvTransferStatus {
        self.status
    }
}
#[derive(Debug)]
pub enum KvTransferPoll {
    Pending,
    Finished(KvTransferCompletion),
}
impl KvTransferPoll {
    pub fn status(&self) -> KvTransferStatus {
        match self {
            Self::Pending => KvTransferStatus::Pending,
            Self::Finished(c) => c.status(),
        }
    }
    pub fn completion(self) -> Option<KvTransferCompletion> {
        match self {
            Self::Pending => None,
            Self::Finished(c) => Some(c),
        }
    }
}

/// One tier belongs to one logical pool. Host IDs address symmetric pages on
/// every rank. Owned anchors are swaps and are never evicted. Unowned
/// anchors are the pool's public host entries: the tier updates the pool's
/// host maps in the same call as every drop of one.
#[derive(Debug)]
pub struct HostTier {
    identity: Arc<()>,
    /// Binds the pool's public host entries to this tier.
    tier: u64,
    refs: Vec<u32>,
    free: Vec<HostPageId>,
    anchors: BTreeMap<HostAnchorId, HostAnchor>,
    transfers: BTreeMap<KvTransferId, Transfer>,
    next: u64,
}
impl HostTier {
    pub fn new(pool: &KvPool, capacity: usize) -> Result<Self, KvError> {
        if capacity > MAX_BLOCK_CAPACITY {
            return Err(KvError::BlockCapacityTooLarge {
                maximum: MAX_BLOCK_CAPACITY,
                requested: capacity,
            });
        }
        Ok(Self {
            identity: Arc::clone(&pool.identity),
            tier: NEXT_TIER.fetch_add(1, Ordering::Relaxed),
            refs: vec![0; capacity],
            free: (0..capacity).rev().map(HostPageId).collect(),
            anchors: BTreeMap::new(),
            transfers: BTreeMap::new(),
            next: 1,
        })
    }
    pub fn capacity(&self) -> usize {
        self.refs.len()
    }
    pub fn used_pages(&self) -> usize {
        self.capacity() - self.free.len()
    }
    pub fn transfer_ids(&self) -> impl Iterator<Item = KvTransferId> + '_ {
        self.transfers.keys().copied()
    }
    pub fn pending_transfers(&self) -> usize {
        self.transfers.len()
    }
    pub fn anchor(&self, id: HostAnchorId) -> Option<&HostAnchor> {
        self.anchors.get(&id)
    }
    fn check_pool(&self, pool: &KvPool) -> Result<(), KvError> {
        if Arc::ptr_eq(&self.identity, &pool.identity) {
            Ok(())
        } else {
            Err(KvError::StaleHandle)
        }
    }
    fn release(&mut self, page: HostPageId) {
        self.refs[page.0] -= 1;
        if self.refs[page.0] == 0 {
            self.free.push(page);
        }
    }
    /// Drops an owned swap anchor. Public entries go through
    /// [`Self::drop_public_entry`], which also updates the pool.
    pub fn drop_anchor(&mut self, id: HostAnchorId) -> Result<(), KvError> {
        if self
            .anchors
            .get(&id)
            .ok_or(KvError::StaleHandle)?
            .owner
            .is_none()
        {
            return Err(KvError::StaleHandle);
        }
        let anchor = self.anchors.remove(&id).expect("anchor checked");
        for page in anchor.blocks.into_iter().chain(anchor.state) {
            self.release(page);
        }
        Ok(())
    }
    /// Drops any anchor; a public entry also leaves the pool's host maps.
    /// A transfer naming the anchor keeps its pages until it completes.
    fn release_anchor(&mut self, pool: &mut KvPool, id: HostAnchorId) -> Result<(), KvError> {
        let anchor = self.anchors.remove(&id).ok_or(KvError::StaleHandle)?;
        if anchor.owner.is_none() {
            pool.remove_host_entry(anchor.key, id);
        }
        for page in anchor.blocks.into_iter().chain(anchor.state) {
            self.release(page);
        }
        Ok(())
    }
    /// Drops a key's public host entry, returning whether it had one.
    pub fn drop_public_entry(
        &mut self,
        pool: &mut KvPool,
        key: PrefixCacheKeyV1,
    ) -> Result<bool, KvError> {
        self.check_pool(pool)?;
        pool.ensure_no_prepared()?;
        let Some(entry) = pool.host_entry(key) else {
            return Ok(false);
        };
        self.release_anchor(pool, entry.host)?;
        Ok(true)
    }
    /// Drops every public entry a resume, a recapture or a new resident
    /// descendant made unusable, returning each key with its cause.
    pub fn drop_invalidated_entries(
        &mut self,
        pool: &mut KvPool,
    ) -> Result<Vec<(PrefixCacheKeyV1, PrefixHostInvalidationV1)>, KvError> {
        self.check_pool(pool)?;
        pool.ensure_no_prepared()?;
        let invalidated = pool.invalidated_host_entries();
        for (_, host, _) in &invalidated {
            self.release_anchor(pool, *host)?;
        }
        Ok(invalidated
            .into_iter()
            .map(|(key, _, reason)| (key, reason))
            .collect())
    }
    /// Capture must have completed its device writes before this call. Reserve
    /// references before submitting the returned plan to the executor.
    ///
    /// An owner makes a swap of that request's own boundary. Without one this
    /// is a demotion: the copy of an indexed public frontier into a public
    /// host entry that keeps the anchor's last GPU use, on a pool with the
    /// RAM tier. Its GPU anchor stays until eviction.
    pub fn begin_offload(
        &mut self,
        pool: &mut KvPool,
        key: PrefixCacheKeyV1,
        owner: Option<u64>,
    ) -> Result<(HostAnchorId, KvTransferPlan), KvError> {
        self.check_pool(pool)?;
        pool.ensure_no_prepared()?;
        if owner.is_some() && self.anchors.values().any(|a| a.owner == owner) {
            return Err(KvError::StaleHandle);
        }
        if owner.is_none() {
            pool.check_demotion(key, self.tier)?;
        }
        let id = next_transfer()?;
        let next = self
            .next
            .checked_add(1)
            .ok_or(KvError::SequenceGenerationExhausted)?;
        let (tokens, blocks, state) = pool.anchor_pages(key).ok_or(KvError::StaleHandle)?;
        let allocations = blocks
            .iter()
            .chain(&state)
            .map(|p| pool.page_allocation(*p))
            .collect::<Result<Vec<_>, _>>()?;
        // Copy every page; host copies are never shared by allocation.
        // Allocation identity does not version in-place writes: a prefix
        // anchor ending inside a full page resumes with that page as its
        // tail, so a later append or last-page rewrite changes its bytes
        // under the same allocation while an older host copy keeps the old.
        let needed = allocations.len();
        while self.free.len() < needed {
            let candidate = self
                .anchors
                .iter()
                .filter(|(_, a)| a.owner.is_none() && a.ready)
                .min_by_key(|(_, a)| a.last_used)
                .map(|(id, _)| *id);
            let Some(id) = candidate else { break };
            self.release_anchor(pool, id)?;
        }
        if self.free.len() < needed {
            return Err(KvError::PoolExhausted {
                needed,
                free: self.free.len(),
            });
        }
        let lease = pool.retain_transfer_pages(&allocations)?;
        let mut hosts = Vec::with_capacity(needed);
        let mut pairs = Vec::with_capacity(needed);
        for allocation in &allocations {
            let host = self.free.pop().expect("host pages counted");
            // One reference for the anchor, one for the transfer.
            self.refs[host.0] = 2;
            hosts.push(host);
            pairs.push(KvPagePair {
                device: allocation.page,
                host,
            });
        }
        let anchor = HostAnchorId(self.next);
        let receipt = Arc::new(AtomicBool::new(false));
        let class = if owner.is_some() {
            KvTransferClass::SwapOut
        } else {
            KvTransferClass::Demotion
        };
        self.next = next;
        self.anchors.insert(
            anchor,
            HostAnchor {
                key,
                tokens,
                blocks: hosts[..blocks.len()].to_vec(),
                state: hosts[blocks.len()..].to_vec(),
                owner,
                ready: false,
                last_used: id.0,
            },
        );
        self.transfers.insert(
            id,
            Transfer {
                lease,
                hosts,
                anchor,
                key,
                direction: KvTransferDirection::Out,
                class,
                quarantined: false,
                receipt: Arc::clone(&receipt),
                destination: None,
            },
        );
        if owner.is_none() {
            pool.record_demotion(key, self.tier, anchor, id);
        }
        Ok((
            anchor,
            KvTransferPlan {
                id,
                direction: KvTransferDirection::Out,
                class,
                pages: pairs,
                receipt,
            },
        ))
    }
    pub fn begin_reload(
        &mut self,
        pool: &mut KvPool,
        anchor: HostAnchorId,
        destination: &PrefixHitV1,
    ) -> Result<KvTransferPlan, KvError> {
        self.check_pool(pool)?;
        let id = next_transfer()?;
        let next = self
            .next
            .checked_add(1)
            .ok_or(KvError::SequenceGenerationExhausted)?;
        let a = self.anchors.get(&anchor).ok_or(KvError::StaleHandle)?;
        let snapshot = pool.snapshot(destination.handle)?;
        // Public entries are promoted into pending anchors, never reloaded
        // into a request's sequence.
        if a.owner.is_none()
            || !a.ready
            || a.tokens != snapshot.tokens
            || a.blocks.len() != snapshot.blocks.len()
            || a.state.len() != destination.state.len()
            || self.transfers.values().any(|t| t.anchor == anchor)
        {
            return Err(KvError::StaleHandle);
        }
        let key = a.key;
        let hosts: Vec<_> = a.blocks.iter().chain(&a.state).copied().collect();
        let allocations = snapshot
            .blocks
            .iter()
            .chain(&destination.state)
            .map(|p| pool.page_allocation(*p))
            .collect::<Result<Vec<_>, _>>()?;
        if pool.sequences[destination.handle.slot].restoring != Some(false)
            || pool.sequences[destination.handle.slot].pinned != destination.state
            || allocations.iter().any(|a| pool.blocks[a.page.index()] != 1)
        {
            return Err(KvError::StaleHandle);
        }
        for page in &hosts {
            if self.refs[page.0] == u32::MAX {
                return Err(KvError::InvariantViolation);
            }
        }
        let lease = pool.retain_transfer_pages(&allocations)?;
        for page in &hosts {
            self.refs[page.0] += 1;
        }
        let pages = allocations
            .iter()
            .zip(&hosts)
            .map(|(a, h)| KvPagePair {
                device: a.page,
                host: *h,
            })
            .collect();
        let receipt = Arc::new(AtomicBool::new(false));
        self.next = next;
        self.anchors
            .get_mut(&anchor)
            .expect("anchor checked")
            .last_used = id.0;
        self.transfers.insert(
            id,
            Transfer {
                lease,
                hosts,
                anchor,
                key,
                direction: KvTransferDirection::In,
                class: KvTransferClass::SwapIn,
                quarantined: false,
                receipt: Arc::clone(&receipt),
                destination: Some(destination.handle),
            },
        );
        Ok(KvTransferPlan {
            id,
            direction: KvTransferDirection::In,
            class: KvTransferClass::SwapIn,
            pages,
            receipt,
        })
    }
    /// Consumes executor evidence. On refusal no ownership changes; retry
    /// with the returned token or repoll the executor's retained status.
    pub fn complete_transfer(
        &mut self,
        pool: &mut KvPool,
        completion: KvTransferCompletion,
    ) -> Result<(), (KvError, KvTransferCompletion)> {
        if let Err(e) = self.accept_completion(pool, &completion) {
            return Err((e, completion));
        }
        completion.receipt.store(true, Ordering::Release);
        Ok(())
    }
    fn accept_completion(
        &mut self,
        pool: &mut KvPool,
        c: &KvTransferCompletion,
    ) -> Result<(), KvError> {
        self.check_pool(pool)?;
        pool.ensure_no_prepared()?;
        let t = self.transfers.get(&c.id).ok_or(KvError::StaleHandle)?;
        if t.quarantined
            || c.receipt.load(Ordering::Acquire)
            || !Arc::ptr_eq(&c.receipt, &t.receipt)
        {
            return Err(KvError::StaleHandle);
        }
        if c.status == KvTransferStatus::Unconfirmed {
            self.transfers
                .get_mut(&c.id)
                .expect("checked transfer")
                .quarantined = true;
            return Ok(());
        }
        pool.release_transfer_pages(&t.lease)?;
        let t = self.transfers.remove(&c.id).expect("checked transfer");
        let success = c.status == KvTransferStatus::Complete;
        if t.class == KvTransferClass::Demotion {
            pool.finish_demotion(t.key, t.anchor, c.id, success);
        }
        if success
            && t.direction == KvTransferDirection::Out
            && let Some(a) = self.anchors.get_mut(&t.anchor)
        {
            a.ready = true;
        }
        if success
            && let Some(destination) = t.destination
            && let Ok(slot) = pool.validate(destination)
        {
            pool.sequences[slot].restoring = Some(true);
        }
        if !success && self.anchors.contains_key(&t.anchor) {
            self.release_anchor(pool, t.anchor)?;
        }
        for page in t.hosts {
            self.release(page);
        }
        Ok(())
    }
    pub fn audit(&self, pool: &KvPool) -> Result<(), KvError> {
        self.check_pool(pool)?;
        pool.audit()?;
        let mut refs = vec![0u32; self.capacity()];
        for a in self.anchors.values() {
            for p in a.blocks.iter().chain(&a.state) {
                refs[p.0] += 1;
            }
        }
        for t in self.transfers.values() {
            if !pool.transfer_pages.contains_key(&t.lease.id) {
                return Err(KvError::InvariantViolation);
            }
            for p in &t.hosts {
                refs[p.0] += 1;
            }
        }
        let free: std::collections::BTreeSet<_> = self.free.iter().copied().collect();
        if refs != self.refs
            || free.len() != self.free.len()
            || refs
                .iter()
                .enumerate()
                .any(|(i, r)| (*r == 0) != free.contains(&HostPageId(i)))
        {
            return Err(KvError::InvariantViolation);
        }
        self.audit_public_entries(pool)
    }
    /// The pool's host entries are exactly this tier's public anchors, and
    /// its demotion marks exactly the demotions in flight.
    fn audit_public_entries(&self, pool: &KvPool) -> Result<(), KvError> {
        let mut public = self.anchors.iter().filter(|(_, a)| a.owner.is_none());
        let mut demotions = self
            .transfers
            .iter()
            .filter(|(_, t)| t.class == KvTransferClass::Demotion);
        let Some(ram) = pool.ram.as_ref().filter(|ram| ram.tier == Some(self.tier)) else {
            return if public.next().is_none() && demotions.next().is_none() {
                Ok(())
            } else {
                Err(KvError::InvariantViolation)
            };
        };
        let mut count = 0usize;
        for (id, a) in public {
            count += 1;
            let in_flight = self
                .transfers
                .values()
                .any(|t| t.anchor == *id && t.class == KvTransferClass::Demotion);
            if ram.entries.get(&a.key).is_none_or(|entry| {
                entry.host != *id || entry.ready != a.ready || entry.tokens != a.tokens
            }) || (!a.ready && !in_flight)
            {
                return Err(KvError::InvariantViolation);
            }
        }
        let mut marked = 0usize;
        for (id, t) in demotions {
            marked += 1;
            if ram
                .demotions
                .get(&t.key)
                .is_none_or(|(_, transfer)| transfer != id)
            {
                return Err(KvError::InvariantViolation);
            }
        }
        if count != ram.entries.len() || marked != ram.demotions.len() {
            return Err(KvError::InvariantViolation);
        }
        Ok(())
    }
}
