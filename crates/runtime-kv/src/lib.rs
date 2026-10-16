#![forbid(unsafe_code)]

mod context_parallel;
mod declared_pages;
mod offload;
pub use offload::*;
mod metrics;
mod prefix;

pub use context_parallel::*;
pub use declared_pages::*;
pub use metrics::{KvCachePrecision, KvMetrics, KvMetricsSnapshot};
pub use prefix::{PrefixChainV1, PrefixMediaSpanV1};

use prefix::PrefixRegistryV1;

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub const MAX_BLOCK_CAPACITY: usize = 1 << 24;
const MAX_SEQUENCE_CAPACITY: usize = 1 << 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvGeometry {
    block_tokens: usize,
    block_capacity: usize,
    token_capacity: usize,
}

impl KvGeometry {
    pub fn new(block_tokens: usize, block_capacity: usize) -> Result<Self, KvError> {
        if block_tokens == 0 {
            return Err(KvError::ZeroBlockTokens);
        }
        if block_capacity == 0 {
            return Err(KvError::ZeroBlockCapacity);
        }
        let token_capacity = block_tokens
            .checked_mul(block_capacity)
            .ok_or(KvError::TokenCapacityOverflow)?;
        if block_capacity > MAX_BLOCK_CAPACITY {
            return Err(KvError::BlockCapacityTooLarge {
                maximum: MAX_BLOCK_CAPACITY,
                requested: block_capacity,
            });
        }
        Ok(Self {
            block_tokens,
            block_capacity,
            token_capacity,
        })
    }

    #[must_use]
    pub const fn block_tokens(self) -> usize {
        self.block_tokens
    }

    #[must_use]
    pub const fn block_capacity(self) -> usize {
        self.block_capacity
    }

    #[must_use]
    pub const fn token_capacity(self) -> usize {
        self.token_capacity
    }

    pub fn blocks_for(self, tokens: usize) -> Result<usize, KvError> {
        if tokens == 0 {
            return Ok(0);
        }
        Ok((tokens - 1) / self.block_tokens + 1)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BlockId(usize);

impl BlockId {
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SequenceHandle {
    slot: usize,
    generation: u64,
}

impl SequenceHandle {
    #[must_use]
    pub const fn slot(self) -> usize {
        self.slot
    }

    #[must_use]
    pub const fn generation(self) -> u64 {
        self.generation
    }
}

/// A content digest that names one reusable prefix.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PrefixCacheKeyV1([u8; 32]);

impl PrefixCacheKeyV1 {
    #[must_use]
    pub const fn new(digest: [u8; 32]) -> Self {
        Self(digest)
    }

    #[must_use]
    pub const fn digest(self) -> [u8; 32] {
        self.0
    }
}

/// Chain digests the prefix registry remembers by default.
pub const PREFIX_REGISTRY_DIGESTS_V1: usize = 1 << 18;

/// One cached prefix: a boundary, the pages covering it, and the pages that
/// hold the bundle's snapshot of its own state at that boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
struct PrefixAnchorV1 {
    /// The digest of the anchor's full blocks, which indexes it for lookup.
    chain: PrefixCacheKeyV1,
    tokens: usize,
    blocks: Vec<BlockId>,
    state: Vec<BlockId>,
    last_used: u64,
    private: bool,
}

/// A request resumed from an anchor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixHitV1 {
    handle: SequenceHandle,
    tokens: usize,
    state: Vec<BlockId>,
}

impl PrefixHitV1 {
    #[must_use]
    pub const fn handle(&self) -> SequenceHandle {
        self.handle
    }

    /// Committed tokens the new sequence starts with.
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }

    /// The snapshot pages the sequence pins until its restore completes.
    #[must_use]
    pub fn state(&self) -> &[BlockId] {
        &self.state
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockCopy {
    source: BlockId,
    destination: BlockId,
    initialized_tokens: usize,
}

impl BlockCopy {
    #[must_use]
    pub const fn source(self) -> BlockId {
        self.source
    }

    #[must_use]
    pub const fn destination(self) -> BlockId {
        self.destination
    }

    #[must_use]
    pub const fn initialized_tokens(self) -> usize {
        self.initialized_tokens
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppendReport {
    new_tokens: usize,
    allocated: Vec<BlockId>,
    copy: Option<BlockCopy>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PreparedAppendId(u64);

impl PreparedAppendId {
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0.wrapping_add(1))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedAppend {
    handle: SequenceHandle,
    previous_tokens: usize,
    new_tokens: usize,
    allocated: Vec<BlockId>,
    block_table: Arc<Vec<BlockId>>,
    copy: Option<BlockCopy>,
}

impl PreparedAppend {
    #[must_use]
    pub const fn handle(&self) -> SequenceHandle {
        self.handle
    }

    #[must_use]
    pub const fn previous_tokens(&self) -> usize {
        self.previous_tokens
    }

    #[must_use]
    pub const fn additional_tokens(&self) -> usize {
        self.new_tokens - self.previous_tokens
    }

    #[must_use]
    pub const fn new_tokens(&self) -> usize {
        self.new_tokens
    }

    #[must_use]
    /// Returns the exact blocks reserved by this prepared append.
    pub fn allocated(&self) -> &[BlockId] {
        &self.allocated
    }

    #[must_use]
    /// Returns the exact blocks reserved by this prepared append.
    pub fn newly_allocated(&self) -> &[BlockId] {
        &self.allocated
    }

    #[must_use]
    /// Returns the complete ordered block table after the append commits.
    pub fn block_table(&self) -> &[BlockId] {
        &self.block_table
    }

    #[must_use]
    pub const fn copy(&self) -> Option<BlockCopy> {
        self.copy
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedAppendBatch {
    id: PreparedAppendId,
    appends: Vec<PreparedAppend>,
}

/// Proof that one exact prepared append can commit without a fallible mutation.
#[doc(hidden)]
#[derive(Debug)]
pub struct PreparedKvCommit {
    pool_identity: Arc<()>,
    prepared: PreparedAppendBatch,
    /// Appended tokens each prepared append keeps, in append order.
    kept: Vec<usize>,
    release_handles: Vec<SequenceHandle>,
}

impl PreparedAppendBatch {
    #[must_use]
    pub const fn id(&self) -> PreparedAppendId {
        self.id
    }

    #[must_use]
    pub fn appends(&self) -> &[PreparedAppend] {
        &self.appends
    }
}

impl AppendReport {
    #[must_use]
    pub const fn new_tokens(&self) -> usize {
        self.new_tokens
    }

    #[must_use]
    pub fn allocated(&self) -> &[BlockId] {
        &self.allocated
    }

    #[must_use]
    pub const fn copy(&self) -> Option<BlockCopy> {
        self.copy
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SequenceSnapshot {
    tokens: usize,
    quota: usize,
    blocks: Vec<BlockId>,
}

impl SequenceSnapshot {
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }

    #[must_use]
    pub const fn quota(&self) -> usize {
        self.quota
    }

    #[must_use]
    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    #[must_use]
    pub fn into_blocks(self) -> Vec<BlockId> {
        self.blocks
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvStats {
    block_capacity: usize,
    free_blocks: usize,
    cached_blocks: usize,
    anchors: usize,
    active_sequences: usize,
}

impl KvStats {
    #[must_use]
    pub const fn block_capacity(self) -> usize {
        self.block_capacity
    }

    #[must_use]
    pub const fn free_blocks(self) -> usize {
        self.free_blocks
    }

    /// Blocks some live sequence holds. Blocks only anchors hold are cached,
    /// not used.
    #[must_use]
    pub const fn used_blocks(self) -> usize {
        self.block_capacity - self.free_blocks - self.cached_blocks
    }

    /// Blocks held only by prefix-cache anchors, which eviction can free.
    #[must_use]
    pub const fn cached_blocks(self) -> usize {
        self.cached_blocks
    }

    #[must_use]
    pub const fn anchors(self) -> usize {
        self.anchors
    }

    #[must_use]
    pub const fn active_sequences(self) -> usize {
        self.active_sequences
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SequenceSlot {
    generation: u64,
    active: bool,
    tokens: usize,
    quota: usize,
    blocks: Arc<Vec<BlockId>>,
    /// Anchor snapshot pages a resumed sequence holds until its restore.
    pinned: Vec<BlockId>,
    /// None: usable, Some(false): awaiting DMA, Some(true): awaiting restore.
    restoring: Option<bool>,
}

#[derive(Debug)]
pub struct KvPool {
    identity: Arc<()>,
    geometry: KvGeometry,
    blocks: Vec<u32>,
    allocation_serials: Vec<u64>,
    transfer_pages: BTreeMap<u64, Vec<BlockId>>,
    next_transfer: u64,
    transfer_refs: Vec<u32>,
    transfer_only_pages: usize,
    free_blocks: Vec<BlockId>,
    sequences: Vec<SequenceSlot>,
    free_sequences: Vec<usize>,
    active_sequences: usize,
    anchors: BTreeMap<PrefixCacheKeyV1, PrefixAnchorV1>,
    /// Anchor keys by the digest of their full blocks.
    anchor_chains: BTreeMap<PrefixCacheKeyV1, Vec<PrefixCacheKeyV1>>,
    /// Anchor keys by last use, oldest first.
    anchor_order: BTreeMap<u64, PrefixCacheKeyV1>,
    anchor_clock: u64,
    /// References each block holds on behalf of anchors.
    anchor_refs: Vec<u32>,
    registry: PrefixRegistryV1,
    /// Chains of full blocks a committed prefill computed.
    produced: PrefixRegistryV1,
    produced_operations: AtomicU64,
    prepared: Option<PreparedAppendBatch>,
    next_prepared_id: u64,
    /// Whether an append may rewrite committed rows of the sequence's last
    /// page (drafter rows that need later tokens), so a shared last page is
    /// copied even when full.
    rewrites_last_page: bool,
    metrics: KvMetrics,
}

impl Clone for KvPool {
    fn clone(&self) -> Self {
        Self {
            identity: Arc::new(()),
            geometry: self.geometry,
            blocks: self.blocks.clone(),
            allocation_serials: self.allocation_serials.clone(),
            transfer_pages: self.transfer_pages.clone(),
            next_transfer: self.next_transfer,
            transfer_refs: self.transfer_refs.clone(),
            transfer_only_pages: self.transfer_only_pages,
            free_blocks: self.free_blocks.clone(),
            sequences: self.sequences.clone(),
            free_sequences: self.free_sequences.clone(),
            active_sequences: self.active_sequences,
            anchors: self.anchors.clone(),
            anchor_chains: self.anchor_chains.clone(),
            anchor_order: self.anchor_order.clone(),
            anchor_clock: self.anchor_clock,
            anchor_refs: self.anchor_refs.clone(),
            registry: self.registry.clone(),
            produced: self.produced.clone(),
            produced_operations: AtomicU64::new(self.produced_operations()),
            prepared: self.prepared.clone(),
            next_prepared_id: self.next_prepared_id,
            rewrites_last_page: self.rewrites_last_page,
            metrics: self.metrics.detached_clone(),
        }
    }
}

impl PartialEq for KvPool {
    fn eq(&self, other: &Self) -> bool {
        self.geometry == other.geometry
            && self.allocation_serials == other.allocation_serials
            && self.transfer_pages == other.transfer_pages
            && self.next_transfer == other.next_transfer
            && self.transfer_refs == other.transfer_refs
            && self.transfer_only_pages == other.transfer_only_pages
            && self.blocks == other.blocks
            && self.free_blocks == other.free_blocks
            && self.sequences == other.sequences
            && self.free_sequences == other.free_sequences
            && self.active_sequences == other.active_sequences
            && self.anchors == other.anchors
            && self.anchor_refs == other.anchor_refs
            && self.registry == other.registry
            && self.produced == other.produced
            && self.prepared == other.prepared
            && self.next_prepared_id == other.next_prepared_id
            && self.rewrites_last_page == other.rewrites_last_page
    }
}

impl Eq for KvPool {}

impl KvPool {
    pub fn new(geometry: KvGeometry, sequence_capacity: usize) -> Result<Self, KvError> {
        Self::new_with_metrics(geometry, sequence_capacity, KvMetrics::default())
    }

    pub fn new_with_metrics(
        geometry: KvGeometry,
        sequence_capacity: usize,
        metrics: KvMetrics,
    ) -> Result<Self, KvError> {
        metrics.configure_capacity(geometry.token_capacity());
        if sequence_capacity == 0 {
            return Err(KvError::ZeroSequenceCapacity);
        }
        if sequence_capacity > MAX_SEQUENCE_CAPACITY {
            return Err(KvError::SequenceCapacityTooLarge {
                maximum: MAX_SEQUENCE_CAPACITY,
                requested: sequence_capacity,
            });
        }
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(geometry.block_capacity)
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        blocks.resize(geometry.block_capacity, 0);

        let mut free_blocks = Vec::new();
        free_blocks
            .try_reserve_exact(geometry.block_capacity)
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        free_blocks.extend((0..geometry.block_capacity).rev().map(BlockId));

        let mut sequences = Vec::new();
        sequences
            .try_reserve_exact(sequence_capacity)
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        sequences.resize_with(sequence_capacity, || SequenceSlot {
            generation: 0,
            active: false,
            tokens: 0,
            quota: 0,
            blocks: Arc::new(Vec::new()),
            pinned: Vec::new(),
            restoring: None,
        });

        let mut free_sequences = Vec::new();
        free_sequences
            .try_reserve_exact(sequence_capacity)
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        free_sequences.extend((0..sequence_capacity).rev());
        let mut anchor_refs = Vec::new();
        anchor_refs
            .try_reserve_exact(geometry.block_capacity)
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        anchor_refs.resize(geometry.block_capacity, 0);
        Ok(Self {
            identity: Arc::new(()),
            geometry,
            blocks,
            allocation_serials: vec![0; geometry.block_capacity],
            transfer_pages: BTreeMap::new(),
            transfer_refs: vec![0; geometry.block_capacity],
            transfer_only_pages: 0,
            next_transfer: 1,
            free_blocks,
            sequences,
            free_sequences,
            active_sequences: 0,
            anchors: BTreeMap::new(),
            anchor_chains: BTreeMap::new(),
            anchor_order: BTreeMap::new(),
            anchor_clock: 0,
            anchor_refs,
            registry: PrefixRegistryV1::new(PREFIX_REGISTRY_DIGESTS_V1),
            produced: PrefixRegistryV1::new(PREFIX_REGISTRY_DIGESTS_V1),
            produced_operations: AtomicU64::new(0),
            prepared: None,
            next_prepared_id: 1,
            rewrites_last_page: false,
            metrics,
        })
    }

    /// Declares that an append may rewrite committed rows of the sequence's
    /// last page before it adds rows: a model whose drafter row for a
    /// position needs the next token. A shared last page is then copied
    /// before an append even when it is full, so the rewrite never reaches
    /// a page a prefix anchor or another sequence holds.
    #[must_use]
    pub const fn with_last_page_rewrites(mut self, rewrites: bool) -> Self {
        self.rewrites_last_page = rewrites;
        self
    }

    #[must_use]
    pub const fn rewrites_last_page(&self) -> bool {
        self.rewrites_last_page
    }

    #[must_use]
    pub const fn geometry(&self) -> KvGeometry {
        self.geometry
    }

    /// Rows of a sequence of `tokens` in the last page an append writes:
    /// the partial tail, or with last-page rewrites a full last page. Zero
    /// when the append writes only new pages.
    const fn written_tail_tokens(&self, tokens: usize) -> usize {
        let partial = tokens % self.geometry.block_tokens;
        if partial == 0 && self.rewrites_last_page && tokens > 0 {
            self.geometry.block_tokens
        } else {
            partial
        }
    }

    pub fn create_sequence(&mut self, quota: usize) -> Result<SequenceHandle, KvError> {
        self.ensure_no_prepared()?;
        if quota == 0 {
            return Err(KvError::ZeroQuota);
        }
        if quota > self.geometry.token_capacity {
            return Err(KvError::QuotaExceedsPool {
                quota,
                pool_tokens: self.geometry.token_capacity,
            });
        }
        let slot = self
            .free_sequences
            .pop()
            .ok_or(KvError::SequenceCapacityExhausted)?;
        let generation = match self.sequences[slot].generation.checked_add(1) {
            Some(generation) => generation,
            None => {
                self.free_sequences.push(slot);
                return Err(KvError::SequenceGenerationExhausted);
            }
        };
        let blocks = std::mem::take(&mut self.sequences[slot].blocks);
        self.sequences[slot] = SequenceSlot {
            generation,
            active: true,
            tokens: 0,
            quota,
            blocks,
            pinned: Vec::new(),
            restoring: None,
        };
        self.active_sequences += 1;
        Ok(SequenceHandle { slot, generation })
    }

    pub fn fork(
        &mut self,
        source: SequenceHandle,
        child_quota: usize,
    ) -> Result<SequenceHandle, KvError> {
        self.ensure_no_prepared()?;
        let source_index = self.validate(source)?;
        self.ensure_restored(source_index)?;
        let source_slot = &self.sequences[source_index];
        if child_quota < source_slot.tokens {
            return Err(KvError::QuotaExceeded {
                quota: child_quota,
                requested: source_slot.tokens,
            });
        }
        if child_quota > self.geometry.token_capacity {
            return Err(KvError::QuotaExceedsPool {
                quota: child_quota,
                pool_tokens: self.geometry.token_capacity,
            });
        }
        for block in source_slot.blocks.iter() {
            if self.blocks[block.index()] == u32::MAX {
                return Err(KvError::ReferenceCountOverflow { block: *block });
            }
        }
        let source_tokens = source_slot.tokens;
        let mut shared_blocks = Vec::new();
        shared_blocks
            .try_reserve_exact(source_slot.blocks.len())
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        shared_blocks.extend_from_slice(&source_slot.blocks);
        let slot = self
            .free_sequences
            .pop()
            .ok_or(KvError::SequenceCapacityExhausted)?;
        let generation = match self.sequences[slot].generation.checked_add(1) {
            Some(generation) => generation,
            None => {
                self.free_sequences.push(slot);
                return Err(KvError::SequenceGenerationExhausted);
            }
        };
        for block in &shared_blocks {
            self.retain_block(*block);
        }
        self.sequences[slot] = SequenceSlot {
            generation,
            active: true,
            tokens: source_tokens,
            quota: child_quota,
            blocks: Arc::new(shared_blocks),
            pinned: Vec::new(),
            restoring: None,
        };
        self.active_sequences += 1;
        self.metrics.explicit_shared_prefix(source_tokens);
        self.metrics.appended(source_tokens);
        Ok(SequenceHandle { slot, generation })
    }

    /// Records a prompt's full-block digests, returning how many leading
    /// blocks an earlier prompt already produced.
    pub fn observe_prefix(&mut self, chain: &PrefixChainV1) -> usize {
        self.registry.observe(chain)
    }

    /// Whether a committed prefill already computed full block `blocks` of
    /// this prompt, after the same `blocks - 1` blocks: one digest probe.
    #[must_use]
    pub fn produced(&self, chain: &PrefixChainV1, blocks: usize) -> bool {
        self.produced_operations.fetch_add(1, Ordering::Relaxed);
        self.produced.contains(chain, blocks)
    }

    /// Records that a committed prefill computed full blocks `first` through
    /// `last` of this prompt: one digest insert per block.
    pub fn record_produced(&mut self, chain: &PrefixChainV1, first: usize, last: usize) {
        let blocks = last
            .min(chain.full_blocks())
            .saturating_sub(first.max(1) - 1);
        self.produced_operations
            .fetch_add(blocks as u64, Ordering::Relaxed);
        self.produced.record(chain, first, last);
    }

    /// Probes and inserts of the produced-block registry so far, which bound
    /// its cost per committed prefill chunk.
    #[must_use]
    pub fn produced_operations(&self) -> u64 {
        self.produced_operations.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn has_anchor(&self, key: PrefixCacheKeyV1) -> bool {
        self.anchors.contains_key(&key)
    }

    #[must_use]
    pub fn anchor_count(&self) -> usize {
        self.anchors.len()
    }

    /// Finds the longest anchor of this prompt that ends at most at `limit`
    /// tokens and outside every media span.
    #[must_use]
    pub fn lookup_anchor(
        &self,
        chain: &PrefixChainV1,
        limit: usize,
    ) -> Option<(PrefixCacheKeyV1, usize)> {
        if self.anchors.is_empty() || chain.block_tokens() != self.geometry.block_tokens {
            return None;
        }
        let limit = limit.min(chain.tokens().len());
        for blocks in (0..=limit / self.geometry.block_tokens).rev() {
            let Some(keys) = chain
                .chain(blocks)
                .and_then(|digest| self.anchor_chains.get(&digest))
            else {
                continue;
            };
            let best = keys
                .iter()
                .filter_map(|key| {
                    self.anchors
                        .get(key)
                        .filter(|anchor| !anchor.private)
                        .map(|anchor| (*key, anchor.tokens))
                })
                .filter(|(key, tokens)| {
                    *tokens <= limit
                        && !chain.splits_media(*tokens)
                        && chain.key_at(*tokens) == Some(*key)
                })
                .max_by_key(|(_, tokens)| *tokens);
            if best.is_some() {
                return best;
            }
        }
        None
    }

    /// Looks up this prompt's longest anchor and resumes a new sequence from it.
    ///
    /// The sequence shares the anchor's pages and pins its snapshot pages
    /// until [`KvPool::finish_restore`]. Every call counts a query; only
    /// `finish_restore` counts the hit.
    pub fn resume_prefix(
        &mut self,
        chain: &PrefixChainV1,
        limit: usize,
        quota: usize,
    ) -> Result<Option<PrefixHitV1>, KvError> {
        self.ensure_no_prepared()?;
        self.metrics.prefix_cache_lookup(chain.tokens().len());
        let Some((key, _)) = self.lookup_anchor(chain, limit) else {
            return Ok(None);
        };
        self.resume_anchor(key, quota).map(Some)
    }

    /// Resumes an exact public or private anchor. The caller must use its own
    /// capture key for private anchors, never a prompt lookup.
    pub fn resume_anchor(
        &mut self,
        key: PrefixCacheKeyV1,
        quota: usize,
    ) -> Result<PrefixHitV1, KvError> {
        self.ensure_no_prepared()?;
        let tokens = self.anchors.get(&key).ok_or(KvError::StaleHandle)?.tokens;
        if quota < tokens {
            return Err(KvError::QuotaExceeded {
                quota,
                requested: tokens,
            });
        }
        if quota > self.geometry.token_capacity {
            return Err(KvError::QuotaExceedsPool {
                quota,
                pool_tokens: self.geometry.token_capacity,
            });
        }
        let anchor = &self.anchors[&key];
        for block in anchor.blocks.iter().chain(&anchor.state) {
            match self.blocks.get(block.index()) {
                Some(&0) | None => return Err(KvError::InvariantViolation),
                Some(&u32::MAX) => return Err(KvError::ReferenceCountOverflow { block: *block }),
                Some(_) => {}
            }
        }
        let mut shared = Vec::new();
        shared
            .try_reserve_exact(anchor.blocks.len())
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        shared.extend_from_slice(&anchor.blocks);
        let mut pinned = Vec::new();
        pinned
            .try_reserve_exact(anchor.state.len())
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        pinned.extend_from_slice(&anchor.state);
        let state = pinned.clone();
        let slot = self
            .free_sequences
            .pop()
            .ok_or(KvError::SequenceCapacityExhausted)?;
        let generation = match self.sequences[slot].generation.checked_add(1) {
            Some(generation) => generation,
            None => {
                self.free_sequences.push(slot);
                return Err(KvError::SequenceGenerationExhausted);
            }
        };
        for block in shared.iter().chain(&pinned) {
            self.retain_block(*block);
        }
        self.sequences[slot] = SequenceSlot {
            generation,
            active: true,
            tokens,
            quota,
            blocks: Arc::new(shared),
            pinned,
            restoring: None,
        };
        self.active_sequences += 1;
        self.touch_anchor(key);
        self.metrics.appended(tokens);
        Ok(PrefixHitV1 {
            handle: SequenceHandle { slot, generation },
            tokens,
            state,
        })
    }

    /// Ends a successful restore: drops the snapshot pages the resumed
    /// sequence pinned and counts the hit and its reused tokens. A failed
    /// restore releases the sequence instead.
    pub fn finish_restore(&mut self, handle: SequenceHandle) -> Result<(), KvError> {
        self.ensure_no_prepared()?;
        let slot = self.validate(handle)?;
        self.ensure_restored(slot)?;
        self.metrics
            .prefix_cache_restored(self.sequences[slot].tokens);
        self.finish_owned_restore(handle)
    }

    /// Releases an own-boundary restore without claiming a prompt-cache hit.
    pub fn finish_owned_restore(&mut self, handle: SequenceHandle) -> Result<(), KvError> {
        self.ensure_no_prepared()?;
        let slot = self.validate(handle)?;
        self.ensure_restored(slot)?;
        let pinned = std::mem::take(&mut self.sequences[slot].pinned);
        for block in pinned.into_iter().rev() {
            self.release_block(block);
        }
        Ok(())
    }

    /// Publishes an anchor over a sequence's committed tokens.
    ///
    /// The anchor takes a reference on every page of the sequence and owns
    /// `state_pages` fresh pages for the caller to fill with a snapshot. It
    /// returns those pages, or `None` when the key already has an anchor or no
    /// page is free even after evicting older anchors.
    pub fn insert_anchor(
        &mut self,
        key: PrefixCacheKeyV1,
        chain: PrefixCacheKeyV1,
        source: SequenceHandle,
        state_pages: usize,
    ) -> Result<Option<Vec<BlockId>>, KvError> {
        self.insert_anchor_inner(key, chain, source, None, state_pages, false)
    }

    /// [`Self::insert_anchor`] over the sequence's first `tokens` committed
    /// tokens: the anchor references only the pages holding them. The
    /// sequence keeps writing its later pages, which the anchor never
    /// shares; the caller's snapshot must hold the model state at `tokens`.
    pub fn insert_prefix_anchor(
        &mut self,
        key: PrefixCacheKeyV1,
        chain: PrefixCacheKeyV1,
        source: SequenceHandle,
        tokens: usize,
        state_pages: usize,
    ) -> Result<Option<Vec<BlockId>>, KvError> {
        self.insert_anchor_inner(key, chain, source, Some(tokens), state_pages, false)
    }

    fn insert_anchor_inner(
        &mut self,
        key: PrefixCacheKeyV1,
        chain: PrefixCacheKeyV1,
        source: SequenceHandle,
        prefix: Option<usize>,
        state_pages: usize,
        private: bool,
    ) -> Result<Option<Vec<BlockId>>, KvError> {
        self.ensure_no_prepared()?;
        let slot = self.validate(source)?;
        self.ensure_restored(slot)?;
        let committed = self.sequences[slot].tokens;
        let tokens = prefix.unwrap_or(committed);
        if tokens == 0 {
            return Err(KvError::ZeroPrefixTokens);
        }
        if tokens > committed {
            return Err(KvError::InvariantViolation);
        }
        let pages = self.geometry.blocks_for(tokens)?;
        if self.anchors.contains_key(&key) {
            self.touch_anchor(key);
            return Ok(None);
        }
        if let Some(block) = self.sequences[slot].blocks[..pages]
            .iter()
            .find(|block| self.blocks[block.index()] == u32::MAX)
        {
            return Err(KvError::ReferenceCountOverflow { block: *block });
        }
        if self.free_blocks.len() < state_pages {
            self.evict_anchors_for(state_pages);
        }
        if self.free_blocks.len() < state_pages {
            return Ok(None);
        }
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(pages)
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        blocks.extend_from_slice(&self.sequences[slot].blocks[..pages]);
        let mut state = Vec::new();
        state
            .try_reserve_exact(state_pages)
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        let mut chain_keys = self.anchor_chains.get(&chain).cloned().unwrap_or_default();
        chain_keys
            .try_reserve(1)
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        chain_keys.push(key);
        self.check_serials(state_pages)?;
        for _ in 0..state_pages {
            let page = self.free_blocks.pop().expect("free pages were counted");
            self.allocation_serials[page.index()] += 1;
            state.push(page);
        }
        for block in blocks.iter().chain(&state) {
            self.retain_block(*block);
            self.anchor_refs[block.index()] += 1;
        }
        self.anchor_chains.insert(chain, chain_keys);
        self.anchor_clock += 1;
        self.anchor_order.insert(self.anchor_clock, key);
        self.anchors.insert(
            key,
            PrefixAnchorV1 {
                chain,
                tokens,
                blocks,
                state: state.clone(),
                last_used: self.anchor_clock,
                private: false,
            },
        );
        if !private {
            self.metrics.prefix_capture();
        }
        Ok(Some(state))
    }

    /// Drops one anchor, returning whether it existed.
    pub fn remove_anchor(&mut self, key: PrefixCacheKeyV1) -> Result<bool, KvError> {
        self.ensure_no_prepared()?;
        Ok(self.drop_anchor(key))
    }

    /// Evicts least recently used anchors until `needed` pages are free or
    /// no anchor is left.
    fn evict_anchors_for(&mut self, needed: usize) {
        while self.free_blocks.len() < needed {
            let Some((_, key)) = self.anchor_order.first_key_value() else {
                return;
            };
            let key = *key;
            let public = !self.anchors[&key].private;
            self.drop_anchor(key);
            if public {
                self.metrics.prefix_eviction();
            }
        }
    }

    fn drop_anchor(&mut self, key: PrefixCacheKeyV1) -> bool {
        let Some(anchor) = self.anchors.remove(&key) else {
            return false;
        };
        self.anchor_order.remove(&anchor.last_used);
        if let Some(keys) = self.anchor_chains.get_mut(&anchor.chain) {
            keys.retain(|candidate| *candidate != key);
            if keys.is_empty() {
                self.anchor_chains.remove(&anchor.chain);
            }
        }
        for block in anchor.state.iter().rev().chain(anchor.blocks.iter().rev()) {
            self.anchor_refs[block.index()] -= 1;
            self.release_block(*block);
        }
        true
    }

    fn touch_anchor(&mut self, key: PrefixCacheKeyV1) {
        let Some(anchor) = self.anchors.get_mut(&key) else {
            return;
        };
        self.anchor_order.remove(&anchor.last_used);
        self.anchor_clock += 1;
        anchor.last_used = self.anchor_clock;
        self.anchor_order.insert(self.anchor_clock, key);
    }

    fn set_block_references(&mut self, block: BlockId, refs: u32, transfers: u32) {
        let index = block.index();
        self.transfer_only_pages -= usize::from(
            self.transfer_refs[index] > 0 && self.blocks[index] == self.transfer_refs[index],
        );
        self.blocks[index] = refs;
        self.transfer_refs[index] = transfers;
        self.transfer_only_pages += usize::from(transfers > 0 && refs == transfers);
    }

    fn retain_block(&mut self, block: BlockId) {
        self.set_block_references(
            block,
            self.blocks[block.index()] + 1,
            self.transfer_refs[block.index()],
        );
    }

    fn release_block(&mut self, block: BlockId) {
        self.set_block_references(
            block,
            self.blocks[block.index()] - 1,
            self.transfer_refs[block.index()],
        );
        if self.blocks[block.index()] == 0 {
            self.free_blocks.push(block);
        }
    }

    // An available anchor is preferable to waiting behind a paced copy queue.
    // The engine waits only after eviction cannot supply the required pages.
    fn evict_for_append(&mut self, needed: usize) {
        self.evict_anchors_for(needed);
    }

    pub fn append(
        &mut self,
        handle: SequenceHandle,
        additional_tokens: usize,
    ) -> Result<AppendReport, KvError> {
        let mut reports = self.append_batch(&[(handle, additional_tokens)])?;
        Ok(reports.remove(0))
    }

    pub fn reserve(
        &mut self,
        handle: SequenceHandle,
        additional_tokens: usize,
    ) -> Result<AppendReport, KvError> {
        self.append(handle, additional_tokens)
    }

    pub fn append_batch(
        &mut self,
        appends: &[(SequenceHandle, usize)],
    ) -> Result<Vec<AppendReport>, KvError> {
        let prepared = self.prepare_append_batch(appends)?;
        match self.commit_prepared(prepared.id) {
            Ok(reports) => Ok(reports),
            Err(error) => {
                if self.prepared_id() == Some(prepared.id) {
                    self.rollback_prepared(prepared.id)?;
                }
                Err(error)
            }
        }
    }

    /// Ensures a planned append can allocate before a stateful draft call.
    /// No append is prepared and no sequence is changed by this probe.
    pub fn ensure_append_capacity(
        &mut self,
        appends: &[(SequenceHandle, usize)],
    ) -> Result<(), KvError> {
        self.ensure_no_prepared()?;
        let (_, mut needed) = self.plan_appends(appends)?;
        if needed <= self.free_blocks.len() {
            return Ok(());
        }
        self.evict_for_append(needed);
        (_, needed) = self.plan_appends(appends)?;
        if needed > self.free_blocks.len() {
            return Err(KvError::PoolExhausted {
                needed,
                free: self.free_blocks.len(),
            });
        }
        Ok(())
    }

    /// Ensure base appends fit, then cap optional extras using only
    /// the free pages left. Never evict an anchor for an unplanned draft.
    pub fn cap_append_extras(
        &mut self,
        appends: &[(SequenceHandle, usize)],
        maximum_extras: &[usize],
    ) -> Result<Vec<usize>, KvError> {
        self.ensure_no_prepared()?;
        if appends.len() != maximum_extras.len() {
            return Err(KvError::DuplicateBatchHandle);
        }
        let (mut needs, mut needed) = self.plan_appends(appends)?;
        if needed > self.free_blocks.len() {
            self.evict_for_append(needed);
            (needs, needed) = self.plan_appends(appends)?;
        }
        let Some(mut spare) = self.free_blocks.len().checked_sub(needed) else {
            return Err(KvError::PoolExhausted {
                needed,
                free: self.free_blocks.len(),
            });
        };
        let mut limits = Vec::new();
        limits
            .try_reserve_exact(needs.len())
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        for ((_, slot, requested, _, copy), maximum) in needs.into_iter().zip(maximum_extras) {
            let sequence = &self.sequences[slot];
            let base_pages = self.geometry.blocks_for(requested)?;
            // A zero-length base append did not yet account for tail COW.
            let extra_copy = !copy
                && requested == sequence.tokens
                && self.written_tail_tokens(requested) != 0
                && self.blocks[sequence.blocks.last().expect("written page").index()] > 1;
            let extra = if *maximum == 0 || (extra_copy && spare == 0) {
                0
            } else {
                let available = (base_pages + spare - usize::from(extra_copy))
                    * self.geometry.block_tokens
                    - requested;
                (*maximum).min(available).min(sequence.quota - requested)
            };
            if extra > 0 {
                spare -= self.geometry.blocks_for(requested + extra)? - base_pages
                    + usize::from(extra_copy);
            }
            limits.push(extra);
        }
        Ok(limits)
    }

    pub fn prepare_append_batch(
        &mut self,
        appends: &[(SequenceHandle, usize)],
    ) -> Result<PreparedAppendBatch, KvError> {
        self.ensure_no_prepared()?;
        if appends.len() > self.sequences.len() {
            return Err(KvError::DuplicateBatchHandle);
        }
        let next_prepared_id = self
            .next_prepared_id
            .checked_add(1)
            .ok_or(KvError::PreparedIdExhausted)?;
        let (mut needs, mut total_needed) = self.plan_appends(appends)?;
        if total_needed > self.free_blocks.len() && !self.anchors.is_empty() {
            // Evicting only drops references, so a recount can only need
            // fewer pages, for example when a shared tail no longer needs its
            // copy.
            self.evict_for_append(total_needed);
            (needs, total_needed) = self.plan_appends(appends)?;
        }
        if total_needed > self.free_blocks.len() {
            return Err(KvError::PoolExhausted {
                needed: total_needed,
                free: self.free_blocks.len(),
            });
        }

        let mut prepared_appends = Vec::new();
        prepared_appends
            .try_reserve_exact(needs.len())
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        let mut free_offset = 0usize;
        for (handle, slot, requested, needed, copy_required) in needs {
            let mut allocated = Vec::new();
            allocated
                .try_reserve_exact(needed)
                .map_err(|_| KvError::MetadataAllocationFailed)?;
            for offset in 0..needed {
                allocated.push(self.free_blocks[self.free_blocks.len() - 1 - free_offset - offset]);
            }
            free_offset += needed;
            let copy = copy_required.then(|| BlockCopy {
                source: *self.sequences[slot]
                    .blocks
                    .last()
                    .expect("partial sequence has a block"),
                destination: allocated[0],
                initialized_tokens: self.written_tail_tokens(self.sequences[slot].tokens),
            });
            let required_blocks = self.geometry.blocks_for(requested)?;
            let retained_blocks = self.sequences[slot].blocks.len() - usize::from(copy_required);
            let block_table = if needed == 0 {
                Arc::clone(&self.sequences[slot].blocks)
            } else {
                let mut blocks = Vec::new();
                blocks
                    .try_reserve_exact(required_blocks)
                    .map_err(|_| KvError::MetadataAllocationFailed)?;
                blocks.extend_from_slice(&self.sequences[slot].blocks[..retained_blocks]);
                if copy_required {
                    blocks.push(allocated[0]);
                }
                blocks.extend_from_slice(&allocated[usize::from(copy_required)..]);
                Arc::new(blocks)
            };
            if block_table.len() != required_blocks {
                return Err(KvError::InvariantViolation);
            }
            prepared_appends.push(PreparedAppend {
                handle,
                previous_tokens: self.sequences[slot].tokens,
                new_tokens: requested,
                allocated,
                block_table,
                copy,
            });
        }
        let prepared = PreparedAppendBatch {
            id: PreparedAppendId(self.next_prepared_id),
            appends: prepared_appends,
        };
        let exposed = Self::try_clone_prepared(&prepared)?;
        self.check_serials(total_needed)?;
        for page in self.free_blocks.iter().rev().take(total_needed) {
            self.allocation_serials[page.index()] += 1;
        }
        self.free_blocks
            .truncate(self.free_blocks.len() - total_needed);
        self.prepared = Some(prepared);
        self.next_prepared_id = next_prepared_id;
        Ok(exposed)
    }

    /// Validates each append and counts the pages it needs, copies included.
    #[allow(clippy::type_complexity)]
    fn plan_appends(
        &self,
        appends: &[(SequenceHandle, usize)],
    ) -> Result<(Vec<(SequenceHandle, usize, usize, usize, bool)>, usize), KvError> {
        let mut seen_slots = Vec::new();
        seen_slots
            .try_reserve_exact(self.sequences.len())
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        seen_slots.resize(self.sequences.len(), false);
        let mut needs = Vec::new();
        needs
            .try_reserve_exact(appends.len())
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        let mut total_needed = 0usize;
        for (handle, additional_tokens) in appends {
            let slot = self.validate(*handle)?;
            self.ensure_restored(slot)?;
            if seen_slots[slot] {
                return Err(KvError::DuplicateBatchHandle);
            }
            seen_slots[slot] = true;
            let sequence = &self.sequences[slot];
            let requested = sequence
                .tokens
                .checked_add(*additional_tokens)
                .ok_or(KvError::TokenCountOverflow)?;
            if requested > sequence.quota {
                return Err(KvError::QuotaExceeded {
                    quota: sequence.quota,
                    requested,
                });
            }
            let required_blocks = self.geometry.blocks_for(requested)?;
            let copy = *additional_tokens > 0
                && self.written_tail_tokens(sequence.tokens) != 0
                && self.blocks[sequence
                    .blocks
                    .last()
                    .expect("partial sequence has a block")
                    .index()]
                    > 1;
            let needed = required_blocks - sequence.blocks.len() + usize::from(copy);
            total_needed = total_needed
                .checked_add(needed)
                .ok_or(KvError::TokenCountOverflow)?;
            needs.push((*handle, slot, requested, needed, copy));
        }
        Ok((needs, total_needed))
    }

    pub fn commit_prepared(&mut self, id: PreparedAppendId) -> Result<Vec<AppendReport>, KvError> {
        let Some(prepared) = self.prepared.as_ref() else {
            return Err(KvError::NoPreparedTransaction);
        };
        if prepared.id != id {
            return Err(KvError::PreparedTransactionMismatch);
        }
        let mut reports = Vec::new();
        reports
            .try_reserve_exact(prepared.appends.len())
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        let prepared = self
            .prepared
            .take()
            .expect("prepared transaction was validated");
        for append in prepared.appends {
            let slot = self.validate(append.handle)?;
            if let Some(copy) = append.copy {
                // When every owner of a shared tail copies it in one batch,
                // the last copy drops the source's final reference.
                self.release_block(copy.source);
                self.metrics.cow_copy(copy.initialized_tokens);
            }
            self.metrics.appended(append.additional_tokens());
            for block in &append.allocated {
                self.blocks[block.index()] = 1;
            }
            self.sequences[slot].blocks = append.block_table;
            self.sequences[slot].tokens = append.new_tokens;
            reports.push(AppendReport {
                new_tokens: append.new_tokens,
                allocated: append.allocated,
                copy: append.copy,
            });
        }
        Ok(reports)
    }

    /// Validates an exact prepared append and its post-commit releases.
    #[doc(hidden)]
    pub fn preflight_commit_prepared(
        &self,
        id: PreparedAppendId,
        release_handles: &[SequenceHandle],
    ) -> Result<PreparedKvCommit, KvError> {
        self.preflight_commit_prepared_kept(id, &[], release_handles)
    }

    /// Validates a prepared append that keeps only a prefix of some appends.
    ///
    /// Each `(handle, kept)` pair commits `previous + kept` tokens for that
    /// handle, with `1 <= kept <= additional`. Handles not listed keep their
    /// whole append. Blocks that start at or beyond a kept length return to
    /// the free list when the commit applies; tokens past it inside a kept
    /// block are left for the next append to overwrite.
    #[doc(hidden)]
    pub fn preflight_commit_prepared_kept(
        &self,
        id: PreparedAppendId,
        kept_tokens: &[(SequenceHandle, usize)],
        release_handles: &[SequenceHandle],
    ) -> Result<PreparedKvCommit, KvError> {
        let Some(prepared) = self.prepared.as_ref() else {
            return Err(KvError::NoPreparedTransaction);
        };
        if prepared.id != id {
            return Err(KvError::PreparedTransactionMismatch);
        }
        if cfg!(debug_assertions) {
            self.audit()?;
        }
        let mut releases = Vec::new();
        releases
            .try_reserve_exact(release_handles.len())
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        for handle in release_handles {
            self.validate(*handle)?;
            if releases.contains(handle)
                || !prepared
                    .appends
                    .iter()
                    .any(|append| append.handle == *handle)
            {
                return Err(KvError::InvariantViolation);
            }
            releases.push(*handle);
        }
        let mut kept = Vec::new();
        kept.try_reserve_exact(prepared.appends.len())
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        kept.extend(
            prepared
                .appends
                .iter()
                .map(PreparedAppend::additional_tokens),
        );
        for (index, (handle, tokens)) in kept_tokens.iter().enumerate() {
            let Some(position) = prepared
                .appends
                .iter()
                .position(|append| append.handle == *handle)
            else {
                return Err(KvError::InvariantViolation);
            };
            if kept_tokens[..index].iter().any(|(seen, _)| seen == handle) {
                return Err(KvError::DuplicateBatchHandle);
            }
            let additional = prepared.appends[position].additional_tokens();
            if *tokens == 0 || *tokens > additional {
                return Err(KvError::InvalidKeptTokens {
                    kept: *tokens,
                    additional,
                });
            }
            kept[position] = *tokens;
        }
        Ok(PreparedKvCommit {
            pool_identity: Arc::clone(&self.identity),
            prepared: Self::try_clone_prepared(prepared)?,
            kept,
            release_handles: releases,
        })
    }

    /// Applies a token only after all rejection paths have run.
    #[doc(hidden)]
    pub fn apply_prepared_commit(
        &mut self,
        prepared_commit: PreparedKvCommit,
    ) -> Result<(), KvError> {
        if !Arc::ptr_eq(&self.identity, &prepared_commit.pool_identity) {
            return Err(KvError::PreparedTransactionMismatch);
        }
        let Some(prepared) = self.prepared.as_ref() else {
            return Err(KvError::NoPreparedTransaction);
        };
        if prepared.id != prepared_commit.prepared.id
            || prepared.appends.len() != prepared_commit.prepared.appends.len()
            || prepared
                .appends
                .iter()
                .zip(&prepared_commit.prepared.appends)
                .any(|(left, right)| {
                    left.handle != right.handle
                        || left.previous_tokens != right.previous_tokens
                        || left.new_tokens != right.new_tokens
                        || left.allocated != right.allocated
                        || left.copy != right.copy
                        || !Arc::ptr_eq(&left.block_table, &right.block_table)
                })
        {
            return Err(KvError::PreparedTransactionMismatch);
        }
        for (index, handle) in prepared_commit.release_handles.iter().enumerate() {
            self.validate(*handle)?;
            if prepared_commit.release_handles[..index].contains(handle)
                || !prepared
                    .appends
                    .iter()
                    .any(|append| append.handle == *handle)
            {
                return Err(KvError::InvariantViolation);
            }
        }

        if prepared_commit.kept.len() != prepared.appends.len() {
            return Err(KvError::PreparedTransactionMismatch);
        }
        let mut kept_tables = Vec::new();
        kept_tables
            .try_reserve_exact(prepared.appends.len())
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        for (append, kept) in prepared.appends.iter().zip(&prepared_commit.kept) {
            let tokens = append.previous_tokens + kept;
            let required = self.geometry.blocks_for(tokens)?;
            if required == append.block_table.len() {
                kept_tables.push(None);
                continue;
            }
            let mut table = Vec::new();
            table
                .try_reserve_exact(required)
                .map_err(|_| KvError::MetadataAllocationFailed)?;
            table.extend_from_slice(&append.block_table[..required]);
            kept_tables.push(Some(Arc::new(table)));
        }

        let Some(prepared) = self.prepared.take() else {
            return Err(KvError::NoPreparedTransaction);
        };
        for ((append, kept), kept_table) in prepared
            .appends
            .into_iter()
            .zip(prepared_commit.kept)
            .zip(kept_tables)
        {
            let slot = append.handle.slot;
            if let Some(copy) = append.copy {
                // When every owner of a shared tail copies it in one batch,
                // the last copy drops the source's final reference.
                self.release_block(copy.source);
                self.metrics.cow_copy(copy.initialized_tokens);
            }
            self.metrics.appended(kept);
            // Allocated blocks follow the retained prefix of the table in
            // order, so the blocks a shorter table drops are the allocation's
            // tail.
            let dropped = kept_table
                .as_ref()
                .map_or(0, |table| append.block_table.len() - table.len());
            let (used, unused) = append.allocated.split_at(append.allocated.len() - dropped);
            for block in used {
                self.blocks[block.index()] = 1;
            }
            for block in unused.iter().rev() {
                self.free_blocks.push(*block);
            }
            self.sequences[slot].blocks = kept_table.unwrap_or(append.block_table);
            self.sequences[slot].tokens = append.previous_tokens + kept;
        }
        for handle in prepared_commit.release_handles {
            self.release_slot(handle.slot);
        }
        Ok(())
    }

    pub fn rollback_prepared(&mut self, id: PreparedAppendId) -> Result<(), KvError> {
        let Some(prepared) = self.prepared.as_ref() else {
            return Err(KvError::NoPreparedTransaction);
        };
        if prepared.id != id {
            return Err(KvError::PreparedTransactionMismatch);
        }
        let prepared = self
            .prepared
            .take()
            .expect("prepared transaction was validated");
        for append in prepared.appends.into_iter().rev() {
            for block in append.allocated.into_iter().rev() {
                self.free_blocks.push(block);
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn prepared_id(&self) -> Option<PreparedAppendId> {
        self.prepared.as_ref().map(|prepared| prepared.id)
    }

    #[must_use]
    pub const fn prepared(&self) -> Option<&PreparedAppendBatch> {
        self.prepared.as_ref()
    }

    pub fn release(&mut self, handle: SequenceHandle) -> Result<(), KvError> {
        let slot = self.validate(handle)?;
        self.ensure_no_prepared()?;
        self.release_slot(slot);
        Ok(())
    }

    pub fn cancel(&mut self, handle: SequenceHandle) -> Result<(), KvError> {
        let slot = self.validate(handle)?;
        if let Some(prepared) = self.prepared.as_ref() {
            if !prepared
                .appends
                .iter()
                .any(|append| append.handle == handle)
            {
                return Err(KvError::PreparedTransactionPending);
            }
            let id = prepared.id;
            self.rollback_prepared(id)?;
        }
        self.release_slot(slot);
        Ok(())
    }

    fn release_slot(&mut self, slot: usize) {
        self.metrics.released(self.sequences[slot].tokens);
        let released_blocks = std::mem::take(&mut self.sequences[slot].blocks);
        let pinned = std::mem::take(&mut self.sequences[slot].pinned);
        for block in pinned.into_iter().rev() {
            self.release_block(block);
        }
        for block in released_blocks.iter().rev().copied() {
            self.release_block(block);
        }
        self.sequences[slot].active = false;
        self.sequences[slot].tokens = 0;
        self.sequences[slot].quota = 0;
        self.free_sequences.push(slot);
        self.active_sequences -= 1;
    }

    pub fn snapshot(&self, handle: SequenceHandle) -> Result<SequenceSnapshot, KvError> {
        let slot = self.validate(handle)?;
        let sequence = &self.sequences[slot];
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(sequence.blocks.len())
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        blocks.extend_from_slice(&sequence.blocks);
        Ok(SequenceSnapshot {
            tokens: sequence.tokens,
            quota: sequence.quota,
            blocks,
        })
    }

    pub fn block_ref_count(&self, block: BlockId) -> Result<u32, KvError> {
        self.blocks
            .get(block.index())
            .copied()
            .ok_or(KvError::InvalidBlockId { block })
    }

    #[must_use]
    pub fn metrics(&self) -> KvMetrics {
        self.metrics.clone()
    }

    #[must_use]
    pub fn stats(&self) -> KvStats {
        let cached_blocks = if self.anchors.is_empty() {
            0
        } else {
            self.anchor_refs
                .iter()
                .zip(&self.blocks)
                .filter(|(anchor, total)| **anchor > 0 && anchor == total)
                .count()
        };
        KvStats {
            block_capacity: self.geometry.block_capacity,
            free_blocks: self.free_blocks.len(),
            cached_blocks,
            anchors: self.anchors.len(),
            active_sequences: self.active_sequences,
        }
    }

    pub fn audit(&self) -> Result<(), KvError> {
        let mut expected_refs = Vec::new();
        expected_refs
            .try_reserve_exact(self.blocks.len())
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        expected_refs.resize(self.blocks.len(), 0u32);
        for pages in self.transfer_pages.values() {
            for page in pages {
                expected_refs[page.index()] += 1;
            }
        }
        if expected_refs != self.transfer_refs
            || self.transfer_only_pages
                != self
                    .blocks
                    .iter()
                    .zip(&self.transfer_refs)
                    .filter(|(refs, transfers)| **transfers > 0 && refs == transfers)
                    .count()
        {
            return Err(KvError::InvariantViolation);
        }
        let mut active = 0usize;
        let mut sequence_seen = Vec::new();
        sequence_seen
            .try_reserve_exact(self.blocks.len())
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        sequence_seen.resize(self.blocks.len(), 0u32);
        let mut sequence_generation = 0u32;
        for sequence in &self.sequences {
            if !sequence.active {
                if sequence.tokens != 0
                    || sequence.quota != 0
                    || !sequence.blocks.is_empty()
                    || !sequence.pinned.is_empty()
                {
                    return Err(KvError::InvariantViolation);
                }
                continue;
            }
            active += 1;
            sequence_generation = sequence_generation
                .checked_add(1)
                .ok_or(KvError::InvariantViolation)?;
            if sequence.tokens > sequence.quota
                || self.geometry.blocks_for(sequence.tokens)? != sequence.blocks.len()
            {
                return Err(KvError::InvariantViolation);
            }
            for block in sequence.blocks.iter() {
                if block.index() >= self.blocks.len()
                    || sequence_seen[block.index()] == sequence_generation
                {
                    return Err(KvError::InvariantViolation);
                }
                sequence_seen[block.index()] = sequence_generation;
                expected_refs[block.index()] = expected_refs[block.index()]
                    .checked_add(1)
                    .ok_or(KvError::InvariantViolation)?;
            }
        }
        let mut expected_anchor_refs = Vec::new();
        expected_anchor_refs
            .try_reserve_exact(self.blocks.len())
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        expected_anchor_refs.resize(self.blocks.len(), 0u32);
        for sequence in self.sequences.iter().filter(|sequence| sequence.active) {
            for block in &sequence.pinned {
                let refs = expected_refs
                    .get_mut(block.index())
                    .ok_or(KvError::InvariantViolation)?;
                *refs = refs.checked_add(1).ok_or(KvError::InvariantViolation)?;
            }
        }
        let mut indexed = 0usize;
        for (chain, keys) in &self.anchor_chains {
            for key in keys {
                if self
                    .anchors
                    .get(key)
                    .is_none_or(|anchor| anchor.chain != *chain)
                {
                    return Err(KvError::InvariantViolation);
                }
            }
            indexed += keys.len();
        }
        if indexed != self.anchors.len() || self.anchor_order.len() != self.anchors.len() {
            return Err(KvError::InvariantViolation);
        }
        for (key, anchor) in &self.anchors {
            if anchor.tokens == 0
                || self.geometry.blocks_for(anchor.tokens)? != anchor.blocks.len()
                || self.anchor_order.get(&anchor.last_used) != Some(key)
            {
                return Err(KvError::InvariantViolation);
            }
            for block in anchor.blocks.iter().chain(&anchor.state) {
                if block.index() >= self.blocks.len() {
                    return Err(KvError::InvariantViolation);
                }
                expected_refs[block.index()] = expected_refs[block.index()]
                    .checked_add(1)
                    .ok_or(KvError::InvariantViolation)?;
                expected_anchor_refs[block.index()] += 1;
            }
        }
        if active != self.active_sequences
            || expected_refs != self.blocks
            || expected_anchor_refs != self.anchor_refs
        {
            return Err(KvError::InvariantViolation);
        }
        let mut reserved_seen = Vec::new();
        reserved_seen
            .try_reserve_exact(self.blocks.len())
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        reserved_seen.resize(self.blocks.len(), false);
        if let Some(prepared) = &self.prepared {
            let mut sequence_seen = Vec::new();
            sequence_seen
                .try_reserve_exact(self.sequences.len())
                .map_err(|_| KvError::MetadataAllocationFailed)?;
            sequence_seen.resize(self.sequences.len(), false);
            for append in &prepared.appends {
                let slot = self.validate(append.handle)?;
                if sequence_seen[slot]
                    || append.previous_tokens != self.sequences[slot].tokens
                    || append.new_tokens > self.sequences[slot].quota
                    || append.new_tokens < append.previous_tokens
                {
                    return Err(KvError::InvariantViolation);
                }
                sequence_seen[slot] = true;
                let required_blocks = self.geometry.blocks_for(append.new_tokens)?;
                let copy_count = usize::from(append.copy.is_some());
                if required_blocks - self.sequences[slot].blocks.len() + copy_count
                    != append.allocated.len()
                    || append.block_table.len() != required_blocks
                {
                    return Err(KvError::InvariantViolation);
                }
                for block in &append.allocated {
                    if block.index() >= self.blocks.len()
                        || self.blocks[block.index()] != 0
                        || reserved_seen[block.index()]
                    {
                        return Err(KvError::InvariantViolation);
                    }
                    reserved_seen[block.index()] = true;
                }
                if let Some(copy) = append.copy
                    && (append.allocated.first() != Some(&copy.destination)
                        || self.sequences[slot].blocks.last() != Some(&copy.source)
                        || self.blocks[copy.source.index()] <= 1
                        || copy.initialized_tokens
                            != self.written_tail_tokens(self.sequences[slot].tokens))
                {
                    return Err(KvError::InvariantViolation);
                }
                let retained_blocks = self.sequences[slot].blocks.len() - copy_count;
                if append.block_table[..retained_blocks]
                    != self.sequences[slot].blocks[..retained_blocks]
                {
                    return Err(KvError::InvariantViolation);
                }
                let block_table_tail = &append.block_table[retained_blocks..];
                let expected_tail = if let Some(copy) = append.copy {
                    if block_table_tail.first() != Some(&copy.destination) {
                        return Err(KvError::InvariantViolation);
                    }
                    &block_table_tail[1..]
                } else {
                    block_table_tail
                };
                if expected_tail != &append.allocated[copy_count..] {
                    return Err(KvError::InvariantViolation);
                }
            }
        }
        let mut free_seen = Vec::new();
        free_seen
            .try_reserve_exact(self.blocks.len())
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        free_seen.resize(self.blocks.len(), false);
        for block in &self.free_blocks {
            if block.index() >= self.blocks.len()
                || self.blocks[block.index()] != 0
                || free_seen[block.index()]
            {
                return Err(KvError::InvariantViolation);
            }
            free_seen[block.index()] = true;
        }
        if self.blocks.iter().enumerate().any(|(index, refs)| {
            if *refs == 0 {
                free_seen[index] == reserved_seen[index]
            } else {
                free_seen[index] || reserved_seen[index]
            }
        }) {
            return Err(KvError::InvariantViolation);
        }
        let mut free_sequence_seen = Vec::new();
        free_sequence_seen
            .try_reserve_exact(self.sequences.len())
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        free_sequence_seen.resize(self.sequences.len(), false);
        for slot in &self.free_sequences {
            if *slot >= self.sequences.len() || free_sequence_seen[*slot] {
                return Err(KvError::InvariantViolation);
            }
            free_sequence_seen[*slot] = true;
        }
        if self
            .sequences
            .iter()
            .enumerate()
            .any(|(index, slot)| slot.active == free_sequence_seen[index])
        {
            return Err(KvError::InvariantViolation);
        }
        Ok(())
    }

    fn try_clone_prepared(prepared: &PreparedAppendBatch) -> Result<PreparedAppendBatch, KvError> {
        let mut appends = Vec::new();
        appends
            .try_reserve_exact(prepared.appends.len())
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        for append in &prepared.appends {
            let mut allocated = Vec::new();
            allocated
                .try_reserve_exact(append.allocated.len())
                .map_err(|_| KvError::MetadataAllocationFailed)?;
            allocated.extend_from_slice(&append.allocated);
            appends.push(PreparedAppend {
                handle: append.handle,
                previous_tokens: append.previous_tokens,
                new_tokens: append.new_tokens,
                allocated,
                block_table: Arc::clone(&append.block_table),
                copy: append.copy,
            });
        }
        Ok(PreparedAppendBatch {
            id: prepared.id,
            appends,
        })
    }

    fn ensure_restored(&self, slot: usize) -> Result<(), KvError> {
        if self.sequences[slot].restoring.is_some() {
            Err(KvError::StaleHandle)
        } else {
            Ok(())
        }
    }
    fn ensure_no_prepared(&self) -> Result<(), KvError> {
        if self.prepared.is_some() {
            return Err(KvError::PreparedTransactionPending);
        }
        Ok(())
    }

    fn validate(&self, handle: SequenceHandle) -> Result<usize, KvError> {
        let Some(slot) = self.sequences.get(handle.slot) else {
            return Err(KvError::StaleHandle);
        };
        if !slot.active || slot.generation != handle.generation {
            return Err(KvError::StaleHandle);
        }
        Ok(handle.slot)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvError {
    ZeroBlockTokens,
    ZeroBlockCapacity,
    TokenCapacityOverflow,
    TokenCountOverflow,
    BlockCapacityTooLarge { maximum: usize, requested: usize },
    ZeroSequenceCapacity,
    SequenceCapacityTooLarge { maximum: usize, requested: usize },
    MetadataAllocationFailed,
    ZeroQuota,
    QuotaExceedsPool { quota: usize, pool_tokens: usize },
    SequenceCapacityExhausted,
    SequenceGenerationExhausted,
    StaleHandle,
    QuotaExceeded { quota: usize, requested: usize },
    PoolExhausted { needed: usize, free: usize },
    ReferenceCountOverflow { block: BlockId },
    DuplicateBatchHandle,
    PreparedTransactionPending,
    NoPreparedTransaction,
    PreparedTransactionMismatch,
    PreparedIdExhausted,
    InvalidBlockId { block: BlockId },
    ZeroPrefixTokens,
    InvalidPrefixMedia,
    InvalidKeptTokens { kept: usize, additional: usize },
    InvariantViolation,
}

impl fmt::Display for KvError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBlockTokens => formatter.write_str("KV block token count is zero"),
            Self::ZeroBlockCapacity => formatter.write_str("KV block capacity is zero"),
            Self::TokenCapacityOverflow => formatter.write_str("KV token capacity overflows usize"),
            Self::TokenCountOverflow => formatter.write_str("KV token count overflows usize"),
            Self::BlockCapacityTooLarge { maximum, requested } => write!(
                formatter,
                "KV block capacity {requested} exceeds metadata limit {maximum}"
            ),
            Self::ZeroSequenceCapacity => formatter.write_str("KV sequence capacity is zero"),
            Self::SequenceCapacityTooLarge { maximum, requested } => write!(
                formatter,
                "KV sequence capacity {requested} exceeds metadata limit {maximum}"
            ),
            Self::MetadataAllocationFailed => formatter.write_str("KV metadata allocation failed"),
            Self::ZeroQuota => formatter.write_str("KV sequence quota is zero"),
            Self::QuotaExceedsPool { quota, pool_tokens } => write!(
                formatter,
                "KV sequence quota {quota} exceeds pool token capacity {pool_tokens}"
            ),
            Self::SequenceCapacityExhausted => {
                formatter.write_str("KV sequence capacity is exhausted")
            }
            Self::SequenceGenerationExhausted => {
                formatter.write_str("KV sequence generation is exhausted")
            }
            Self::StaleHandle => formatter.write_str("KV sequence handle is stale"),
            Self::QuotaExceeded { quota, requested } => write!(
                formatter,
                "KV token request {requested} exceeds sequence quota {quota}"
            ),
            Self::PoolExhausted { needed, free } => write!(
                formatter,
                "KV append needs {needed} blocks but only {free} are free"
            ),
            Self::ReferenceCountOverflow { block } => write!(
                formatter,
                "KV block {} reference count overflows",
                block.index()
            ),
            Self::DuplicateBatchHandle => {
                formatter.write_str("KV batch contains a duplicate sequence handle")
            }
            Self::PreparedTransactionPending => {
                formatter.write_str("a KV append transaction is already prepared")
            }
            Self::NoPreparedTransaction => {
                formatter.write_str("no KV append transaction is prepared")
            }
            Self::PreparedTransactionMismatch => {
                formatter.write_str("KV prepared transaction ID does not match")
            }
            Self::PreparedIdExhausted => {
                formatter.write_str("KV prepared transaction ID is exhausted")
            }
            Self::InvalidBlockId { block } => {
                write!(formatter, "KV block {} is outside the pool", block.index())
            }
            Self::ZeroPrefixTokens => formatter.write_str("a cached KV prefix has zero tokens"),
            Self::InvalidPrefixMedia => {
                formatter.write_str("prefix media spans are unordered or outside the prompt")
            }
            Self::InvalidKeptTokens { kept, additional } => write!(
                formatter,
                "KV commit keeps {kept} of {additional} prepared tokens"
            ),
            Self::InvariantViolation => formatter.write_str("KV pool invariant audit failed"),
        }
    }
}

impl Error for KvError {}
