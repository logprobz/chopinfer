//! Content digests that name reusable prompt prefixes.
//!
//! A chain digest names the first `j` full blocks of a prompt: the root digest
//! for `j = 0`, then one SHA-256 per block over the previous digest, the
//! block's token IDs and every media span that intersects the block. A
//! boundary inside a block adds one tail digest over its partial tokens.
//! Nothing about sampling is hashed, because KV depends only on the inputs.
//! A request's cache salt, when it has one, enters the root digest, so only
//! requests with the same salt share any key.

use std::collections::BTreeMap;
use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::{KvError, PrefixCacheKeyV1};

const ROOT_DOMAIN: &[u8] = b"chopinfer.prefix.v1";
const TAIL_DOMAIN: &[u8] = b"tail";
const SALT_DOMAIN: &[u8] = b"salt";

/// One media span of a prompt, as the prefix chain hashes it.
///
/// `identity` names the media content and how it is encoded, so the same
/// placeholder tokens over different media never share a digest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefixMediaSpanV1 {
    pub identity: [u8; 32],
    pub start: usize,
    pub rows: usize,
    pub media_row: u32,
}

impl PrefixMediaSpanV1 {
    const fn end(&self) -> usize {
        self.start + self.rows
    }
}

/// The chain digests of one prompt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixChainV1 {
    block_tokens: usize,
    tokens: Arc<[u32]>,
    media: Arc<[PrefixMediaSpanV1]>,
    /// `digests[j]` names the first `j` full blocks; `digests[0]` is the root.
    digests: Vec<PrefixCacheKeyV1>,
}

impl PrefixChainV1 {
    /// Hashes every full block of `tokens`.
    ///
    /// Media spans must be ordered, must not overlap and must lie inside the
    /// prompt.
    pub fn new(
        block_tokens: usize,
        tokens: Arc<[u32]>,
        media: Vec<PrefixMediaSpanV1>,
    ) -> Result<Self, KvError> {
        Self::salted(block_tokens, tokens, media, None)
    }

    /// Hashes every full block of `tokens` under a cache salt. Chains with
    /// different salts, or with and without one, share no digest.
    pub fn salted(
        block_tokens: usize,
        tokens: Arc<[u32]>,
        media: Vec<PrefixMediaSpanV1>,
        salt: Option<&[u8]>,
    ) -> Result<Self, KvError> {
        if block_tokens == 0 {
            return Err(KvError::ZeroBlockTokens);
        }
        let mut end = 0usize;
        for span in &media {
            let span_end = span
                .start
                .checked_add(span.rows)
                .ok_or(KvError::TokenCountOverflow)?;
            if span.rows == 0 || span.start < end || span_end > tokens.len() {
                return Err(KvError::InvalidPrefixMedia);
            }
            end = span_end;
        }
        let full_blocks = tokens.len() / block_tokens;
        let mut digests = Vec::new();
        digests
            .try_reserve_exact(full_blocks + 1)
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        let mut root = Sha256::new();
        root.update(ROOT_DOMAIN);
        root.update((block_tokens as u64).to_le_bytes());
        if let Some(salt) = salt {
            root.update(SALT_DOMAIN);
            root.update((salt.len() as u64).to_le_bytes());
            root.update(salt);
        }
        digests.push(PrefixCacheKeyV1::new(root.finalize().into()));
        let mut chain = Self {
            block_tokens,
            tokens,
            media: media.into(),
            digests,
        };
        for block in 0..full_blocks {
            let start = block * block_tokens;
            let mut sha = Sha256::new();
            sha.update(chain.digests[block].digest());
            chain.hash_range(&mut sha, start, start + block_tokens);
            chain
                .digests
                .push(PrefixCacheKeyV1::new(sha.finalize().into()));
        }
        Ok(chain)
    }

    #[must_use]
    pub const fn block_tokens(&self) -> usize {
        self.block_tokens
    }

    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    #[must_use]
    pub fn full_blocks(&self) -> usize {
        self.digests.len() - 1
    }

    /// The digest of the first `blocks` full blocks.
    #[must_use]
    pub fn chain(&self, blocks: usize) -> Option<PrefixCacheKeyV1> {
        self.digests.get(blocks).copied()
    }

    /// The key of an anchor at `boundary` tokens of this prompt.
    #[must_use]
    pub fn key_at(&self, boundary: usize) -> Option<PrefixCacheKeyV1> {
        if boundary == 0 || boundary > self.tokens.len() {
            return None;
        }
        let blocks = boundary / self.block_tokens;
        let tail = boundary % self.block_tokens;
        if tail == 0 {
            return self.chain(blocks);
        }
        let start = blocks * self.block_tokens;
        let mut sha = Sha256::new();
        sha.update(self.digests[blocks].digest());
        sha.update(TAIL_DOMAIN);
        sha.update((tail as u64).to_le_bytes());
        self.hash_range(&mut sha, start, boundary);
        Some(PrefixCacheKeyV1::new(sha.finalize().into()))
    }

    /// Whether `boundary` falls strictly inside one of the prompt's media spans.
    #[must_use]
    pub fn splits_media(&self, boundary: usize) -> bool {
        self.media
            .iter()
            .any(|span| span.start < boundary && boundary < span.end())
    }

    fn hash_range(&self, sha: &mut Sha256, start: usize, end: usize) {
        for token in &self.tokens[start..end] {
            sha.update(token.to_le_bytes());
        }
        for span in self
            .media
            .iter()
            .filter(|span| span.start < end && start < span.end())
        {
            sha.update(span.identity);
            sha.update((span.start as u64).to_le_bytes());
            sha.update((span.rows as u64).to_le_bytes());
            sha.update(span.media_row.to_le_bytes());
        }
    }
}

/// A bounded, least-recently-seen set of full-block chain digests.
///
/// It holds digests only, never pages. A request that shares leading blocks
/// with an earlier one learns where the shared prefix ends from it.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct PrefixRegistryV1 {
    capacity: usize,
    clock: u64,
    seen: BTreeMap<PrefixCacheKeyV1, u64>,
    order: BTreeMap<u64, PrefixCacheKeyV1>,
}

impl PrefixRegistryV1 {
    pub(crate) const fn new(capacity: usize) -> Self {
        Self {
            capacity,
            clock: 0,
            seen: BTreeMap::new(),
            order: BTreeMap::new(),
        }
    }

    /// Counts the leading full blocks seen before, then records all of them.
    pub(crate) fn observe(&mut self, chain: &PrefixChainV1) -> usize {
        let matched = self.matched(chain);
        self.record(chain, 1, chain.full_blocks());
        matched
    }

    /// Whether the chain of the first `blocks` full blocks is recorded.
    pub(crate) fn contains(&self, chain: &PrefixChainV1, blocks: usize) -> bool {
        chain
            .chain(blocks)
            .is_some_and(|digest| self.seen.contains_key(&digest))
    }

    /// Counts the leading full blocks recorded before.
    pub(crate) fn matched(&self, chain: &PrefixChainV1) -> usize {
        (1..=chain.full_blocks())
            .take_while(|blocks| {
                chain
                    .chain(*blocks)
                    .is_some_and(|digest| self.seen.contains_key(&digest))
            })
            .count()
    }

    /// Records the chains of the first `first` through `last` full blocks,
    /// oldest records first out.
    pub(crate) fn record(&mut self, chain: &PrefixChainV1, first: usize, last: usize) {
        if self.capacity == 0 {
            return;
        }
        for blocks in first.max(1)..=last.min(chain.full_blocks()) {
            let Some(digest) = chain.chain(blocks) else {
                break;
            };
            self.clock += 1;
            if let Some(previous) = self.seen.insert(digest, self.clock) {
                self.order.remove(&previous);
            }
            self.order.insert(self.clock, digest);
            while self.seen.len() > self.capacity {
                let Some((_, oldest)) = self.order.pop_first() else {
                    break;
                };
                self.seen.remove(&oldest);
            }
        }
    }
}
