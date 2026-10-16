//! Model-neutral rank splits for heads, experts, vocabulary and MLP columns.
//!
//! These are placement calculations, not a claim that a kernel supports a
//! topology. In particular, padded heads still need masking in the consumer.

use crate::contract::Refusal;
use std::ops::Range;

/// Maximum rank count supported by the placement arithmetic, including TP16.
/// This does not change the limits of existing collective kernels or ABIs.
pub const MAX_RANKS: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorParallel {
    ranks: usize,
}

impl TensorParallel {
    /// Accept powers of two from one through [`MAX_RANKS`].
    pub fn new(ranks: usize) -> Result<Self, Refusal> {
        if !ranks.is_power_of_two() || ranks > MAX_RANKS {
            return Err(Refusal("rank count must be 1, 2, 4, 8 or 16"));
        }
        Ok(Self { ranks })
    }

    pub fn ranks(self) -> usize {
        self.ranks
    }

    /// Equal contiguous shards. Refuse zero dimensions and any remainder.
    /// Use this for attention/linear heads, expert IDs or vocabulary entries
    /// when the consumer requires exact divisibility.
    pub fn split(self, total: usize) -> Result<RankSplit, Refusal> {
        if !total.is_multiple_of(self.ranks) {
            return Err(Refusal("dimension is not divisible by rank count"));
        }
        self.padded_split(total, 1)
    }

    /// Split into chunks of `ceil(total / ranks)` source elements, then pad
    /// each rank's storage to `tile` elements. Neither padding nor an empty
    /// trailing rank owns a global source index. `total` and `tile` must be
    /// nonzero. Tile padding does not move the source shard boundaries.
    ///
    /// For example, 640 MLP columns at TP4 with tile 128 give each rank 160
    /// live columns and 256 storage columns.
    pub fn padded_split(self, total: usize, tile: usize) -> Result<RankSplit, Refusal> {
        RankSplit::new(self.ranks, total, self.ranks, tile)
    }

    /// Shard KV heads, duplicating each head across contiguous rank groups
    /// when there are fewer heads than ranks. In that case rank `r` holds
    /// head `floor(r * total / ranks)`, even for unequal duplication groups.
    /// Otherwise use a padded split with one head per padding element.
    /// Zero heads are refused. Query-head alignment remains the caller's job.
    pub fn kv_heads(self, total: usize) -> Result<RankSplit, Refusal> {
        RankSplit::new(self.ranks, total, self.ranks.min(total), 1)
    }
}

/// A rank's source slice and its allocation in concatenated rank storage.
/// Copy `live` into the start of `padded`; the remaining slots are padding.
/// Source coordinates and storage coordinates differ when there is padding
/// or duplication. Both ranges are half-open.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RankShard {
    pub live: Range<usize>,
    pub padded: Range<usize>,
}

impl RankShard {
    pub fn live_len(&self) -> usize {
        self.live.len()
    }

    pub fn padded_len(&self) -> usize {
        self.padded.len()
    }
}

/// A validated split. No allocation or model dependencies are needed.
/// Use the same index arithmetic for expert ownership and vocabulary shards.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RankSplit {
    ranks: usize,
    total: usize,
    parts: usize,
    source_stride: usize,
    padded_len: usize,
}

impl RankSplit {
    fn new(ranks: usize, total: usize, parts: usize, tile: usize) -> Result<Self, Refusal> {
        if total == 0 || tile == 0 {
            return Err(Refusal("dimension and tile must be nonzero"));
        }
        let source_stride = total.div_ceil(parts);
        let padded_len = source_stride
            .div_ceil(tile)
            .checked_mul(tile)
            .ok_or(Refusal("padded shard length overflow"))?;
        // Validate every endpoint once so rank and owner lookups cannot overflow.
        padded_len
            .checked_mul(ranks)
            .ok_or(Refusal("padded topology length overflow"))?;
        Ok(Self {
            ranks,
            total,
            parts,
            source_stride,
            padded_len,
        })
    }

    pub fn ranks(self) -> usize {
        self.ranks
    }

    /// Number of source elements, excluding duplicates and padding.
    pub fn total(self) -> usize {
        self.total
    }

    /// Uniform allocation length per rank, including tile padding.
    pub fn padded_len(self) -> usize {
        self.padded_len
    }

    pub fn rank(self, rank: usize) -> Result<RankShard, Refusal> {
        if rank >= self.ranks {
            return Err(Refusal("rank outside topology"));
        }
        let part = rank * self.parts / self.ranks;
        let start = part * self.source_stride;
        let storage_start = rank * self.padded_len;
        Ok(RankShard {
            live: start.min(self.total)..(start + self.source_stride).min(self.total),
            padded: storage_start..storage_start + self.padded_len,
        })
    }

    /// All ranks holding a live global index. Padding has no owners.
    pub fn owners(self, index: usize) -> Option<Range<usize>> {
        if index >= self.total {
            return None;
        }
        Some(self.group(index / self.source_stride))
    }

    /// Canonical owner: the first rank in the duplication group.
    pub fn owner(self, index: usize) -> Option<usize> {
        self.owners(index).map(|group| group.start)
    }

    /// Ranks sharing this rank's source shard. A nonduplicated shard has a
    /// singleton group, including an empty trailing shard of a padded split.
    pub fn duplication_group(self, rank: usize) -> Result<Range<usize>, Refusal> {
        if rank >= self.ranks {
            return Err(Refusal("rank outside topology"));
        }
        Ok(self.group(rank * self.parts / self.ranks))
    }

    fn group(self, part: usize) -> Range<usize> {
        (part * self.ranks).div_ceil(self.parts)..((part + 1) * self.ranks).div_ceil(self.parts)
    }
}
