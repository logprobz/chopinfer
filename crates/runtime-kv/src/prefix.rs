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
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
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
        digests.push(root_digest(block_tokens, salt));
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

    /// Whether this is the chain [`Self::salted`] builds from these inputs.
    ///
    /// Every digest after the root follows from the root, the block size,
    /// the tokens and the media, and the root from the block size and the
    /// salt, so comparing those checks a chain built elsewhere without
    /// hashing its tokens again.
    #[must_use]
    pub fn is_salted(
        &self,
        block_tokens: usize,
        tokens: &[u32],
        media: &[PrefixMediaSpanV1],
        salt: Option<&[u8]>,
    ) -> bool {
        self.block_tokens == block_tokens
            && *self.tokens == *tokens
            && *self.media == *media
            && self.digests[0] == root_digest(block_tokens, salt)
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

    /// Already computed full-page identities, including the salted root.
    #[must_use]
    pub fn digests(&self) -> &[PrefixCacheKeyV1] {
        &self.digests
    }

    /// Incremental continuation of the final page. The caller must account
    /// for its allocation and discard it before retaining a digest-only history.
    pub fn continuation(&self) -> Result<PrefixDigestTailV1, KvError> {
        let start = self.full_blocks() * self.block_tokens;
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(self.block_tokens)
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        tokens.extend_from_slice(&self.tokens[start..]);
        let mut media = Vec::new();
        let spans = self.media.iter().filter(|span| start < span.end()).count();
        media
            .try_reserve_exact(spans)
            .map_err(|_| KvError::MetadataAllocationFailed)?;
        media.extend(self.media.iter().filter(|span| start < span.end()).copied());
        Ok(PrefixDigestTailV1 {
            block_tokens: self.block_tokens,
            previous: self.digests[self.full_blocks()],
            tokens,
            media,
        })
    }

    /// Allocation capacity needed for [`Self::continuation`].
    #[must_use]
    pub fn continuation_bytes(&self) -> usize {
        self.block_tokens * size_of::<u32>()
            + self
                .media
                .iter()
                .filter(|span| self.full_blocks() * self.block_tokens < span.end())
                .count()
                * size_of::<PrefixMediaSpanV1>()
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

    /// Whether this position is a text token, outside all media spans.
    #[must_use]
    pub fn text_token(&self, position: usize) -> bool {
        !self
            .media
            .iter()
            .any(|span| span.start <= position && position < span.end())
    }

    /// Whether `boundary` falls strictly inside one of the prompt's media spans.
    #[must_use]
    pub fn splits_media(&self, boundary: usize) -> bool {
        self.media
            .iter()
            .any(|span| span.start < boundary && boundary < span.end())
    }

    /// The token at `position` and the media span starting there, if any.
    pub(crate) fn atom(&self, position: usize) -> PrefixAtom {
        let index = self.media.partition_point(|span| span.start < position);
        PrefixAtom {
            token: self.tokens[position],
            media: self
                .media
                .get(index)
                .filter(|span| span.start == position)
                .copied(),
        }
    }

    /// The media spans starting in `start..end`, in order.
    pub(crate) fn media_starting(&self, start: usize, end: usize) -> &[PrefixMediaSpanV1] {
        let first = self.media.partition_point(|span| span.start < start);
        let last = self.media.partition_point(|span| span.start < end);
        &self.media[first..last]
    }

    fn hash_range(&self, sha: &mut Sha256, start: usize, end: usize) {
        hash_tokens(sha, &self.tokens[start..end]);
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

/// The digest of a chain's first zero blocks, which names its block size
/// and salt.
fn root_digest(block_tokens: usize, salt: Option<&[u8]>) -> PrefixCacheKeyV1 {
    let mut root = Sha256::new();
    root.update(ROOT_DOMAIN);
    root.update((block_tokens as u64).to_le_bytes());
    if let Some(salt) = salt {
        root.update(SALT_DOMAIN);
        root.update((salt.len() as u64).to_le_bytes());
        root.update(salt);
    }
    PrefixCacheKeyV1::new(root.finalize().into())
}

/// Hashes each token's little-endian bytes in order. A digest depends only on
/// the bytes, so staging them in a buffer gives the digest one update per
/// token gives, with one update per buffer.
fn hash_tokens(sha: &mut Sha256, tokens: &[u32]) {
    const STAGED: usize = 256;
    let mut bytes = [0; STAGED * size_of::<u32>()];
    for chunk in tokens.chunks(STAGED) {
        for (staged, token) in bytes.chunks_exact_mut(size_of::<u32>()).zip(chunk) {
            staged.copy_from_slice(&token.to_le_bytes());
        }
        sha.update(&bytes[..size_of_val(chunk)]);
    }
}

/// Bounded partial-page material for extending a prompt with committed text
/// output. It has no KV handles and never changes the public prefix cache.
#[derive(Debug)]
pub struct PrefixDigestTailV1 {
    block_tokens: usize,
    previous: PrefixCacheKeyV1,
    tokens: Vec<u32>,
    media: Vec<PrefixMediaSpanV1>,
}

impl PrefixDigestTailV1 {
    fn digest(&self, tail: bool) -> PrefixCacheKeyV1 {
        let mut sha = Sha256::new();
        sha.update(self.previous.digest());
        if tail {
            sha.update(TAIL_DOMAIN);
            sha.update((self.tokens.len() as u64).to_le_bytes());
        }
        hash_tokens(&mut sha, &self.tokens);
        for span in &self.media {
            sha.update(span.identity);
            sha.update((span.start as u64).to_le_bytes());
            sha.update((span.rows as u64).to_le_bytes());
            sha.update(span.media_row.to_le_bytes());
        }
        PrefixCacheKeyV1::new(sha.finalize().into())
    }

    /// Exact identity at the current boundary, including partial-page media.
    #[must_use]
    pub fn key(&self) -> PrefixCacheKeyV1 {
        if self.tokens.is_empty() {
            self.previous
        } else {
            self.digest(true)
        }
    }

    /// Appends an accepted output token and returns any newly completed page.
    pub fn push(&mut self, token: u32) -> Option<PrefixCacheKeyV1> {
        self.tokens.push(token);
        if self.tokens.len() != self.block_tokens {
            return None;
        }
        self.previous = self.digest(false);
        self.tokens.clear();
        self.media.clear();
        Some(self.previous)
    }

    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.tokens.capacity() * size_of::<u32>()
            + self.media.capacity() * size_of::<PrefixMediaSpanV1>()
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

/// Exact material for splitting a radix edge inside a logical block. Media
/// starts carry the complete span identity; public boundaries never split it.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct PrefixAtom {
    pub(crate) token: u32,
    pub(crate) media: Option<PrefixMediaSpanV1>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The digests as the module documentation defines them, each over one
    /// buffer of the previous digest, the block's token bytes and its media.
    fn defined(
        block_tokens: usize,
        tokens: &[u32],
        media: &[PrefixMediaSpanV1],
        salt: Option<&[u8]>,
    ) -> Vec<[u8; 32]> {
        let mut root = ROOT_DOMAIN.to_vec();
        root.extend((block_tokens as u64).to_le_bytes());
        if let Some(salt) = salt {
            root.extend(SALT_DOMAIN);
            root.extend((salt.len() as u64).to_le_bytes());
            root.extend(salt);
        }
        let mut digests = vec![<[u8; 32]>::from(Sha256::digest(&root))];
        for (block, chunk) in tokens.chunks_exact(block_tokens).enumerate() {
            let (start, end) = (block * block_tokens, (block + 1) * block_tokens);
            let mut buffer = digests[block].to_vec();
            buffer.extend(chunk.iter().flat_map(|token| token.to_le_bytes()));
            for span in media
                .iter()
                .filter(|span| span.start < end && start < span.end())
            {
                buffer.extend(span.identity);
                buffer.extend((span.start as u64).to_le_bytes());
                buffer.extend((span.rows as u64).to_le_bytes());
                buffer.extend(span.media_row.to_le_bytes());
            }
            digests.push(Sha256::digest(&buffer).into());
        }
        digests
    }

    fn prompt(length: usize) -> Vec<u32> {
        (0..length as u32)
            .map(|index| index.wrapping_mul(2_654_435_761) % 151_000)
            .collect()
    }

    #[test]
    fn chain_digests_hash_the_defined_bytes() {
        let media = vec![PrefixMediaSpanV1 {
            identity: [9; 32],
            start: 300,
            rows: 700,
            media_row: 4,
        }];
        for block_tokens in [1, 3, 64, 256, 257, 1024] {
            for length in [0, 1, 255, 256, 1000, 5000] {
                for salt in [None, Some(b"scope".as_slice())] {
                    let tokens = prompt(length);
                    let spans = if length >= 1000 {
                        media.clone()
                    } else {
                        Vec::new()
                    };
                    let chain = PrefixChainV1::salted(
                        block_tokens,
                        Arc::from(tokens.as_slice()),
                        spans.clone(),
                        salt,
                    )
                    .unwrap();
                    let expected = defined(block_tokens, &tokens, &spans, salt);
                    let actual = chain
                        .digests()
                        .iter()
                        .map(|key| key.digest())
                        .collect::<Vec<_>>();
                    assert_eq!(actual, expected, "{block_tokens} {length} {salt:?}");
                }
            }
        }
    }

    /// A chain is the one its own inputs build, and no other's.
    #[test]
    fn a_chain_is_salted_only_from_its_own_inputs() {
        let tokens = prompt(1000);
        let media = vec![PrefixMediaSpanV1 {
            identity: [3; 32],
            start: 10,
            rows: 20,
            media_row: 0,
        }];
        let build = |block: usize, tokens: &[u32], media: &[PrefixMediaSpanV1], salt| {
            PrefixChainV1::salted(block, Arc::from(tokens), media.to_vec(), salt).unwrap()
        };
        let salt = Some(b"scope".as_slice());
        let chain = build(64, &tokens, &media, salt);
        assert!(chain.is_salted(64, &tokens, &media, salt));
        let mut other = tokens.clone();
        other[999] += 1;
        for (block, tokens, media, salt) in [
            (32, &tokens, &media[..], salt),
            (64, &other, &media[..], salt),
            (64, &tokens, &[][..], salt),
            (64, &tokens, &media[..], None),
            (64, &tokens, &media[..], Some(b"other".as_slice())),
        ] {
            assert!(!chain.is_salted(block, tokens, media, salt));
            assert_ne!(chain, build(block, tokens, media, salt));
        }
    }

    /// One served-size digest, pinned so a change to the bytes hashed fails.
    #[test]
    fn a_long_chain_digest_is_pinned() {
        let tokens = prompt(94_080);
        let chain =
            PrefixChainV1::salted(256, Arc::from(tokens.as_slice()), Vec::new(), Some(b"salt"))
                .unwrap();
        let hex = |key: PrefixCacheKeyV1| {
            key.digest()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        assert_eq!(chain.full_blocks(), 367);
        assert_eq!(
            hex(chain.chain(367).unwrap()),
            "0c6d450b6fbb21d3e9ba4fa54a811f604755cdc528c97b0c5937cf5e43da1539"
        );
        assert_eq!(
            hex(chain.key_at(94_080).unwrap()),
            "24a8dc80e2f07a1c8b43c5ed0a286dd505486bf0c078bb5b47f0cf926b8d6415"
        );
    }
}
