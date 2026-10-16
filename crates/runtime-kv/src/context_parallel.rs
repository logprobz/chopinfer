//! Context-parallel page ownership.
//!
//! A context-parallel group of `size` ranks splits each sequence's KV pages
//! round-robin: the sequence's physical page `j` (tokens `j * page_tokens`
//! onward) lives on group rank `j % size`. Pages are allocated in stripes of
//! one page per group rank, all at the same local page index, so one logical
//! pool page is one stripe of `size * page_tokens` tokens and the logical
//! block table is every rank's own table of local pages. The pool, prefix
//! anchors, snapshots, preemption and offload then work on stripes unchanged,
//! and at size 1 a stripe is exactly one page.

use std::error::Error;
use std::fmt;

use crate::{KvError, KvGeometry};

/// The largest context-parallel group, matching the largest declared layout.
pub const MAX_CONTEXT_PARALLEL_SIZE: usize = crate::MAX_DECLARED_RANKS as usize;

/// How a context-parallel group owns the pages of every sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContextParallelPagesV1 {
    size: usize,
    page_tokens: usize,
}

/// One page a group rank owns: its local page index and the sequence
/// positions it holds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RankPageV1 {
    pub local_page: usize,
    pub first_position: usize,
    pub tokens: usize,
}

impl ContextParallelPagesV1 {
    /// A group of `size` ranks over physical pages of `page_tokens` tokens,
    /// for a topology of `ranks` ranks that `size` divides.
    pub fn new(
        size: usize,
        page_tokens: usize,
        ranks: usize,
    ) -> Result<Self, ContextParallelError> {
        if size == 0 || size > MAX_CONTEXT_PARALLEL_SIZE {
            return Err(ContextParallelError::SizeOutOfRange { size });
        }
        if ranks == 0 || !ranks.is_multiple_of(size) {
            return Err(ContextParallelError::SizeDoesNotDivideRanks { size, ranks });
        }
        if page_tokens == 0 {
            return Err(ContextParallelError::ZeroPageTokens);
        }
        if size.checked_mul(page_tokens).is_none() {
            return Err(ContextParallelError::Overflow);
        }
        Ok(Self { size, page_tokens })
    }

    #[must_use]
    pub const fn size(self) -> usize {
        self.size
    }

    /// Tokens of one physical page on one rank.
    #[must_use]
    pub const fn page_tokens(self) -> usize {
        self.page_tokens
    }

    /// Tokens of one logical pool page: one page on every group rank.
    #[must_use]
    pub const fn stripe_tokens(self) -> usize {
        self.size * self.page_tokens
    }

    /// The logical pool over `pages_per_rank` local pages on every rank. Its
    /// token capacity counts every group rank's pages.
    pub fn geometry(self, pages_per_rank: usize) -> Result<KvGeometry, KvError> {
        KvGeometry::new(self.stripe_tokens(), pages_per_rank)
    }

    /// The group rank of a topology rank: groups are consecutive ranks.
    #[must_use]
    pub const fn group_rank(self, rank: usize) -> usize {
        rank % self.size
    }

    /// The group rank that holds sequence position `position`.
    #[must_use]
    pub const fn owner(self, position: usize) -> usize {
        position / self.page_tokens % self.size
    }

    /// How many of a sequence's first `tokens` positions group rank `rank`
    /// holds.
    #[must_use]
    pub const fn rank_tokens(self, rank: usize, tokens: usize) -> usize {
        let stripe = self.stripe_tokens();
        let tail = tokens % stripe;
        let before = rank * self.page_tokens;
        let partial = if tail > before {
            let held = tail - before;
            if held < self.page_tokens {
                held
            } else {
                self.page_tokens
            }
        } else {
            0
        };
        tokens / stripe * self.page_tokens + partial
    }

    /// The sequence position of the `row`-th position group rank `rank` holds.
    #[must_use]
    pub const fn rank_position(self, rank: usize, row: usize) -> usize {
        (row / self.page_tokens * self.size + rank) * self.page_tokens + row % self.page_tokens
    }

    /// The pages group rank `rank` holds of a sequence of `tokens` tokens
    /// over its logical page `table` (pool page indices, as the batch wire
    /// carries them), in sequence order. Each is the rank's local page at
    /// the stripe's index.
    pub fn rank_pages(
        self,
        rank: usize,
        table: impl IntoIterator<Item = usize>,
        tokens: usize,
    ) -> Result<Vec<RankPageV1>, ContextParallelError> {
        if rank >= self.size {
            return Err(ContextParallelError::RankOutOfRange {
                rank,
                size: self.size,
            });
        }
        let mut stripes = 0;
        let mut pages = Vec::new();
        for page in table {
            let first_position = (stripes * self.size + rank) * self.page_tokens;
            stripes += 1;
            if first_position < tokens {
                pages.push(RankPageV1 {
                    local_page: page,
                    first_position,
                    tokens: (tokens - first_position).min(self.page_tokens),
                });
            }
        }
        if stripes < tokens.div_ceil(self.stripe_tokens()) {
            return Err(ContextParallelError::ShortTable {
                pages: stripes,
                tokens,
            });
        }
        Ok(pages)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContextParallelError {
    SizeOutOfRange { size: usize },
    SizeDoesNotDivideRanks { size: usize, ranks: usize },
    ZeroPageTokens,
    Overflow,
    RankOutOfRange { rank: usize, size: usize },
    ShortTable { pages: usize, tokens: usize },
}

impl fmt::Display for ContextParallelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeOutOfRange { size } => write!(
                formatter,
                "context-parallel size {size} is outside 1..={MAX_CONTEXT_PARALLEL_SIZE}"
            ),
            Self::SizeDoesNotDivideRanks { size, ranks } => write!(
                formatter,
                "context-parallel size {size} does not divide the {ranks} tensor-parallel rank(s)"
            ),
            Self::ZeroPageTokens => formatter.write_str("the KV page token count is zero"),
            Self::Overflow => formatter.write_str("the context-parallel stripe overflows usize"),
            Self::RankOutOfRange { rank, size } => write!(
                formatter,
                "group rank {rank} is outside a context-parallel group of {size}"
            ),
            Self::ShortTable { pages, tokens } => write!(
                formatter,
                "a table of {pages} stripe(s) cannot hold {tokens} token(s)"
            ),
        }
    }
}

impl Error for ContextParallelError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_one_stripes_are_pages() {
        let pages = ContextParallelPagesV1::new(1, 16, 4).unwrap();
        assert_eq!(pages.stripe_tokens(), 16);
        assert!((0..4).all(|rank| pages.group_rank(rank) == 0));
        let pairs = ContextParallelPagesV1::new(2, 16, 4).unwrap();
        assert_eq!(
            (0..4)
                .map(|rank| pairs.group_rank(rank))
                .collect::<Vec<_>>(),
            [0, 1, 0, 1]
        );
        assert_eq!(pages.geometry(8).unwrap(), KvGeometry::new(16, 8).unwrap());
        for tokens in 0..64 {
            assert_eq!(pages.rank_tokens(0, tokens), tokens);
            assert_eq!(pages.rank_position(0, tokens), tokens);
            assert_eq!(pages.owner(tokens), 0);
        }
    }

    #[test]
    fn rank_rows_cover_the_sequence_once() {
        for size in [1, 2, 4] {
            let pages = ContextParallelPagesV1::new(size, 4, 4).unwrap();
            for tokens in 0..50 {
                let mut seen = vec![0; tokens];
                for rank in 0..size {
                    let held = pages.rank_tokens(rank, tokens);
                    for row in 0..held {
                        let position = pages.rank_position(rank, row);
                        assert_eq!(pages.owner(position), rank);
                        seen[position] += 1;
                    }
                }
                assert!(seen.iter().all(|count| *count == 1), "{size} {tokens}");
            }
        }
    }

    #[test]
    fn sizes_must_divide_the_ranks() {
        assert_eq!(
            ContextParallelPagesV1::new(3, 16, 4),
            Err(ContextParallelError::SizeDoesNotDivideRanks { size: 3, ranks: 4 })
        );
        assert_eq!(
            ContextParallelPagesV1::new(0, 16, 4),
            Err(ContextParallelError::SizeOutOfRange { size: 0 })
        );
        assert_eq!(
            ContextParallelPagesV1::new(2, 0, 4),
            Err(ContextParallelError::ZeroPageTokens)
        );
    }
}
