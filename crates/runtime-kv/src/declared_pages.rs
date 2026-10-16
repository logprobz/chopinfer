//! Physical paging over a bundle-declared page layout.
//!
//! The layout is declared by whichever model owns it: page bytes, alignment,
//! page tokens, rank placement, and state lifetime. This module knows none of
//! those numbers ahead of time. Logical page allocation, sharing,
//! cancellation, and commit stay in `crate::KvPool`; this layer adds the
//! rank-local storage plan.

use std::error::Error;
use std::fmt;

use crate::{BlockId, MAX_BLOCK_CAPACITY, MAX_SEQUENCE_CAPACITY};

/// The largest rank count a declared layout may span.
pub const MAX_DECLARED_RANKS: u8 = 64;

/// Where one logical page is placed across ranks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RankPlacementV1 {
    Replicated,
    RankLocal,
    Sharded,
}

/// How long declared storage lives.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StateLifetimeV1 {
    Transaction,
    Request,
    Process,
}

/// A bundle-declared physical page layout.
///
/// One page holds `page_tokens` tokens in `segments_per_page` equally shaped
/// segments. A segment is the model's own unit of division inside a page, for
/// example one layer plane; the shared host never interprets what a segment
/// means, only how wide it is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeclaredPageLayoutV1 {
    bytes_per_token: u32,
    segments_per_page: u32,
    page_tokens: u32,
    alignment_bytes: u64,
    rank_count: u8,
    rank_placement: RankPlacementV1,
    state_lifetime: StateLifetimeV1,
    maximum_sequence_tokens: usize,
    segment_bytes: u64,
    page_bytes: u64,
}

impl DeclaredPageLayoutV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bytes_per_token: u32,
        segments_per_page: u32,
        page_tokens: u32,
        alignment_bytes: u64,
        rank_count: u8,
        rank_placement: RankPlacementV1,
        state_lifetime: StateLifetimeV1,
        maximum_sequence_tokens: usize,
    ) -> Result<Self, DeclaredPageLayoutError> {
        if bytes_per_token == 0 {
            return Err(DeclaredPageLayoutError::ZeroBytesPerToken);
        }
        if segments_per_page == 0 {
            return Err(DeclaredPageLayoutError::ZeroSegmentsPerPage);
        }
        if page_tokens == 0 {
            return Err(DeclaredPageLayoutError::ZeroPageTokens);
        }
        if alignment_bytes == 0 || !alignment_bytes.is_power_of_two() {
            return Err(DeclaredPageLayoutError::InvalidAlignment { alignment_bytes });
        }
        if rank_count == 0 || rank_count > MAX_DECLARED_RANKS {
            return Err(DeclaredPageLayoutError::RankCountOutOfRange { rank_count });
        }
        if maximum_sequence_tokens == 0 {
            return Err(DeclaredPageLayoutError::ZeroMaximumSequenceTokens);
        }
        let segment_bytes = u64::from(bytes_per_token)
            .checked_mul(u64::from(page_tokens))
            .ok_or(DeclaredPageLayoutError::ArithmeticOverflow)?;
        let page_bytes = segment_bytes
            .checked_mul(u64::from(segments_per_page))
            .ok_or(DeclaredPageLayoutError::ArithmeticOverflow)?;
        if !page_bytes.is_multiple_of(alignment_bytes) {
            return Err(DeclaredPageLayoutError::UnalignedPage {
                page_bytes,
                alignment_bytes,
            });
        }
        Ok(Self {
            bytes_per_token,
            segments_per_page,
            page_tokens,
            alignment_bytes,
            rank_count,
            rank_placement,
            state_lifetime,
            maximum_sequence_tokens,
            segment_bytes,
            page_bytes,
        })
    }

    #[must_use]
    pub const fn bytes_per_token(&self) -> usize {
        self.bytes_per_token as usize
    }

    #[must_use]
    pub const fn segments_per_page(&self) -> usize {
        self.segments_per_page as usize
    }

    #[must_use]
    pub const fn page_tokens(&self) -> usize {
        self.page_tokens as usize
    }

    #[must_use]
    pub const fn alignment_bytes(&self) -> u64 {
        self.alignment_bytes
    }

    #[must_use]
    pub const fn rank_count(&self) -> usize {
        self.rank_count as usize
    }

    #[must_use]
    pub const fn rank_placement(&self) -> RankPlacementV1 {
        self.rank_placement
    }

    #[must_use]
    pub const fn state_lifetime(&self) -> StateLifetimeV1 {
        self.state_lifetime
    }

    #[must_use]
    pub const fn maximum_sequence_tokens(&self) -> usize {
        self.maximum_sequence_tokens
    }

    #[must_use]
    pub const fn segment_bytes(&self) -> u64 {
        self.segment_bytes
    }

    #[must_use]
    pub const fn page_bytes(&self) -> u64 {
        self.page_bytes
    }

    #[must_use]
    pub fn maximum_logical_pages(&self) -> usize {
        self.maximum_sequence_tokens.div_ceil(self.page_tokens())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeclaredPageLayoutError {
    ZeroBytesPerToken,
    ZeroSegmentsPerPage,
    ZeroPageTokens,
    ZeroMaximumSequenceTokens,
    InvalidAlignment {
        alignment_bytes: u64,
    },
    RankCountOutOfRange {
        rank_count: u8,
    },
    UnalignedPage {
        page_bytes: u64,
        alignment_bytes: u64,
    },
    ArithmeticOverflow,
}

impl fmt::Display for DeclaredPageLayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBytesPerToken => {
                formatter.write_str("declared page layout has zero bytes per token")
            }
            Self::ZeroSegmentsPerPage => {
                formatter.write_str("declared page layout has zero segments per page")
            }
            Self::ZeroPageTokens => {
                formatter.write_str("declared page layout has zero page tokens")
            }
            Self::ZeroMaximumSequenceTokens => {
                formatter.write_str("declared page layout has zero maximum sequence tokens")
            }
            Self::InvalidAlignment { alignment_bytes } => write!(
                formatter,
                "declared page alignment {alignment_bytes} is not a nonzero power of two"
            ),
            Self::RankCountOutOfRange { rank_count } => write!(
                formatter,
                "declared rank count {rank_count} is outside 1..={MAX_DECLARED_RANKS}"
            ),
            Self::UnalignedPage {
                page_bytes,
                alignment_bytes,
            } => write!(
                formatter,
                "declared page of {page_bytes} bytes is not a multiple of {alignment_bytes}"
            ),
            Self::ArithmeticOverflow => {
                formatter.write_str("declared page layout arithmetic overflowed u64")
            }
        }
    }
}

impl Error for DeclaredPageLayoutError {}

mod ownership_sealed {
    pub trait Sealed {}
}

pub trait RankOwnershipCapability: ownership_sealed::Sealed {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnqualifiedRankOwnership {
    _private: (),
}

impl ownership_sealed::Sealed for UnqualifiedRankOwnership {}
impl RankOwnershipCapability for UnqualifiedRankOwnership {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RankOwnershipQualification {
    Unqualified,
}

/// One segment plane inside one page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageSegmentSpecV1 {
    segment: usize,
    bytes_per_token: usize,
    page_tokens: usize,
    segment_bytes: u64,
}

impl PageSegmentSpecV1 {
    #[must_use]
    pub const fn segment(self) -> usize {
        self.segment
    }

    #[must_use]
    pub const fn bytes_per_token(self) -> usize {
        self.bytes_per_token
    }

    #[must_use]
    pub const fn page_tokens(self) -> usize {
        self.page_tokens
    }

    #[must_use]
    pub const fn segment_bytes(self) -> u64 {
        self.segment_bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KvRankV1(u8);

impl KvRankV1 {
    /// Checks one rank index against a declared rank count.
    pub fn new(rank: u8, rank_count: usize) -> Result<Self, PagedKvErrorV1> {
        if usize::from(rank) >= rank_count {
            return Err(PagedKvErrorV1::RankOutOfRange { rank });
        }
        Ok(Self(rank))
    }

    #[must_use]
    pub const fn index(self) -> u8 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PhysicalPageIdV1(usize);

impl PhysicalPageIdV1 {
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

impl From<BlockId> for PhysicalPageIdV1 {
    fn from(block: BlockId) -> Self {
        Self(block.index())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RankPoolPlanV1 {
    rank: KvRankV1,
    page_capacity: u64,
    bytes: u64,
    base_offset: u64,
}

impl RankPoolPlanV1 {
    #[must_use]
    pub const fn rank(self) -> KvRankV1 {
        self.rank
    }

    #[must_use]
    pub const fn page_capacity(self) -> u64 {
        self.page_capacity
    }

    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    #[must_use]
    pub const fn base_offset(self) -> u64 {
        self.base_offset
    }
}

/// The physical plan for one declared layout at one pool capacity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PagedKvPlanV1<C: RankOwnershipCapability = UnqualifiedRankOwnership> {
    layout: DeclaredPageLayoutV1,
    page_capacity: u64,
    request_capacity: usize,
    bytes_per_rank_pool: u64,
    total_bytes: u64,
    capability: C,
}

impl PagedKvPlanV1<UnqualifiedRankOwnership> {
    pub fn validate(
        layout: DeclaredPageLayoutV1,
        page_capacity: u64,
        request_capacity: usize,
    ) -> Result<Self, PagedKvErrorV1> {
        if page_capacity == 0 {
            return Err(PagedKvErrorV1::ZeroPageCapacity);
        }
        if request_capacity == 0 {
            return Err(PagedKvErrorV1::ZeroRequestCapacity);
        }
        let bytes_per_rank_pool = layout
            .page_bytes()
            .checked_mul(page_capacity)
            .ok_or(PagedKvErrorV1::ArithmeticOverflow)?;
        let total_bytes = bytes_per_rank_pool
            .checked_mul(layout.rank_count() as u64)
            .ok_or(PagedKvErrorV1::ArithmeticOverflow)?;
        if page_capacity > MAX_BLOCK_CAPACITY as u64 {
            return Err(PagedKvErrorV1::PageCapacityTooLarge {
                maximum: MAX_BLOCK_CAPACITY as u64,
                requested: page_capacity,
            });
        }
        if request_capacity > MAX_SEQUENCE_CAPACITY {
            return Err(PagedKvErrorV1::RequestCapacityTooLarge {
                maximum: MAX_SEQUENCE_CAPACITY,
                requested: request_capacity,
            });
        }
        Ok(Self {
            layout,
            page_capacity,
            request_capacity,
            bytes_per_rank_pool,
            total_bytes,
            capability: UnqualifiedRankOwnership { _private: () },
        })
    }
}

impl<C: RankOwnershipCapability> PagedKvPlanV1<C> {
    #[must_use]
    pub const fn layout(&self) -> &DeclaredPageLayoutV1 {
        &self.layout
    }

    #[must_use]
    pub const fn page_capacity(&self) -> u64 {
        self.page_capacity
    }

    #[must_use]
    pub const fn request_capacity(&self) -> usize {
        self.request_capacity
    }

    #[must_use]
    pub const fn bytes_per_token(&self) -> usize {
        self.layout.bytes_per_token()
    }

    #[must_use]
    pub const fn segments_per_page(&self) -> usize {
        self.layout.segments_per_page()
    }

    #[must_use]
    pub const fn page_tokens(&self) -> usize {
        self.layout.page_tokens()
    }

    #[must_use]
    pub const fn bytes_per_segment_page(&self) -> u64 {
        self.layout.segment_bytes()
    }

    #[must_use]
    pub const fn bytes_per_page(&self) -> u64 {
        self.layout.page_bytes()
    }

    #[must_use]
    pub const fn bytes_per_rank_pool(&self) -> u64 {
        self.bytes_per_rank_pool
    }

    #[must_use]
    pub const fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    #[must_use]
    pub fn segment_specs(&self) -> Vec<PageSegmentSpecV1> {
        (0..self.layout.segments_per_page())
            .map(|segment| PageSegmentSpecV1 {
                segment,
                bytes_per_token: self.layout.bytes_per_token(),
                page_tokens: self.layout.page_tokens(),
                segment_bytes: self.layout.segment_bytes(),
            })
            .collect()
    }

    #[must_use]
    pub const fn ownership_qualification(&self) -> RankOwnershipQualification {
        RankOwnershipQualification::Unqualified
    }

    pub fn rank(&self, rank: u8) -> Result<KvRankV1, PagedKvErrorV1> {
        KvRankV1::new(rank, self.layout.rank_count())
    }

    #[must_use]
    pub fn rank_pools(&self) -> Vec<RankPoolPlanV1> {
        (0..self.layout.rank_count())
            .map(|rank| self.rank_pool_unchecked(rank as u8))
            .collect()
    }

    pub fn rank_pool(&self, rank: KvRankV1) -> Result<RankPoolPlanV1, PagedKvErrorV1> {
        if usize::from(rank.index()) >= self.layout.rank_count() {
            return Err(PagedKvErrorV1::RankOutOfRange { rank: rank.index() });
        }
        Ok(self.rank_pool_unchecked(rank.index()))
    }

    pub fn logical_pages_for_tokens(&self, tokens: usize) -> Result<usize, PagedKvErrorV1> {
        if tokens > self.layout.maximum_sequence_tokens() {
            return Err(PagedKvErrorV1::SequenceTokenOutOfRange {
                tokens,
                maximum: self.layout.maximum_sequence_tokens(),
            });
        }
        Ok(tokens.div_ceil(self.layout.page_tokens()))
    }

    #[doc(hidden)]
    pub fn validate_pages(&self, pages: &[PhysicalPageIdV1]) -> Result<(), PagedKvErrorV1> {
        let maximum = self.layout.maximum_logical_pages();
        if pages.len() > maximum {
            return Err(PagedKvErrorV1::PageTableTooLong {
                pages: pages.len(),
                maximum,
            });
        }
        for (position, page) in pages.iter().enumerate() {
            if page.index() as u64 >= self.page_capacity {
                return Err(PagedKvErrorV1::PageOutOfRange {
                    page: *page,
                    capacity: self.page_capacity,
                });
            }
            if pages[..position].contains(page) {
                return Err(PagedKvErrorV1::DuplicatePage { page: *page });
            }
        }
        Ok(())
    }

    pub fn page_offset(&self, page: PhysicalPageIdV1) -> Result<u64, PagedKvErrorV1> {
        self.page_pool_offset(page)
    }

    pub fn aggregate_page_offset(
        &self,
        rank: KvRankV1,
        page: PhysicalPageIdV1,
    ) -> Result<u64, PagedKvErrorV1> {
        self.rank_pool_unchecked(rank.index())
            .base_offset
            .checked_add(self.page_pool_offset(page)?)
            .ok_or(PagedKvErrorV1::ArithmeticOverflow)
    }

    pub fn row_offset(
        &self,
        page: PhysicalPageIdV1,
        segment: usize,
        token_in_page: usize,
    ) -> Result<u64, PagedKvErrorV1> {
        self.row_pool_offset(page, segment, token_in_page)
    }

    fn rank_pool_unchecked(&self, rank: u8) -> RankPoolPlanV1 {
        RankPoolPlanV1 {
            rank: KvRankV1(rank),
            page_capacity: self.page_capacity,
            bytes: self.bytes_per_rank_pool,
            base_offset: self.bytes_per_rank_pool * u64::from(rank),
        }
    }

    fn page_pool_offset(&self, page: PhysicalPageIdV1) -> Result<u64, PagedKvErrorV1> {
        if page.index() as u64 >= self.page_capacity {
            return Err(PagedKvErrorV1::PageOutOfRange {
                page,
                capacity: self.page_capacity,
            });
        }
        (page.index() as u64)
            .checked_mul(self.layout.page_bytes())
            .ok_or(PagedKvErrorV1::ArithmeticOverflow)
    }

    fn row_pool_offset(
        &self,
        page: PhysicalPageIdV1,
        segment: usize,
        token_in_page: usize,
    ) -> Result<u64, PagedKvErrorV1> {
        if segment >= self.layout.segments_per_page() {
            return Err(PagedKvErrorV1::SegmentOutOfRange {
                segment,
                segment_count: self.layout.segments_per_page(),
            });
        }
        if token_in_page >= self.layout.page_tokens() {
            return Err(PagedKvErrorV1::TokenInPageOutOfRange {
                token: token_in_page,
                page_tokens: self.layout.page_tokens(),
            });
        }
        self.page_pool_offset(page)?
            .checked_add((segment as u64) * self.layout.segment_bytes())
            .and_then(|offset| {
                offset.checked_add((token_in_page * self.layout.bytes_per_token()) as u64)
            })
            .ok_or(PagedKvErrorV1::ArithmeticOverflow)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PagedKvErrorV1 {
    ZeroPageCapacity,
    ZeroRequestCapacity,
    ArithmeticOverflow,
    PageCapacityTooLarge {
        maximum: u64,
        requested: u64,
    },
    RequestCapacityTooLarge {
        maximum: usize,
        requested: usize,
    },
    RankOutOfRange {
        rank: u8,
    },
    SegmentOutOfRange {
        segment: usize,
        segment_count: usize,
    },
    TokenInPageOutOfRange {
        token: usize,
        page_tokens: usize,
    },
    SequenceTokenOutOfRange {
        tokens: usize,
        maximum: usize,
    },
    PageTableTooLong {
        pages: usize,
        maximum: usize,
    },
    PageOutOfRange {
        page: PhysicalPageIdV1,
        capacity: u64,
    },
    DuplicatePage {
        page: PhysicalPageIdV1,
    },
}

impl fmt::Display for PagedKvErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroPageCapacity => formatter.write_str("paged KV page capacity is zero"),
            Self::ZeroRequestCapacity => formatter.write_str("paged KV request capacity is zero"),
            Self::ArithmeticOverflow => {
                formatter.write_str("paged KV planning arithmetic overflowed u64")
            }
            Self::PageCapacityTooLarge { maximum, requested } => write!(
                formatter,
                "paged KV page capacity {requested} exceeds {maximum}"
            ),
            Self::RequestCapacityTooLarge { maximum, requested } => write!(
                formatter,
                "paged KV request capacity {requested} exceeds {maximum}"
            ),
            Self::RankOutOfRange { rank } => {
                write!(formatter, "rank {rank} is outside the declared rank count")
            }
            Self::SegmentOutOfRange {
                segment,
                segment_count,
            } => write!(
                formatter,
                "page segment {segment} is outside the declared {segment_count} segments"
            ),
            Self::TokenInPageOutOfRange { token, page_tokens } => write!(
                formatter,
                "token-in-page {token} is outside the {page_tokens}-token page"
            ),
            Self::SequenceTokenOutOfRange { tokens, maximum } => write!(
                formatter,
                "sequence length {tokens} exceeds declared maximum {maximum}"
            ),
            Self::PageTableTooLong { pages, maximum } => {
                write!(formatter, "page table length {pages} exceeds {maximum}")
            }
            Self::PageOutOfRange { page, capacity } => write!(
                formatter,
                "page {} is outside pool capacity {capacity}",
                page.index()
            ),
            Self::DuplicatePage { page } => {
                write!(formatter, "page table repeats page {}", page.index())
            }
        }
    }
}

impl Error for PagedKvErrorV1 {}
