//! Context parallelism at group sizes 2 and 4: the buckets a group launch
//! carries, the per-rank sizes its buffers follow and each rank's buffer
//! bytes (`bundle_runtime::context_parallel`), the group gathers' states as
//! the selected backend sizes them (`backend::Kernels::group_gather`). The
//! backend's entries are its own (`crate::sm120::context_parallel` on
//! SM120).

use bundle_runtime::context_parallel::{self as group, GroupDims, RankBytes};

use crate::backend::{Kernels, Selected};
use crate::contract::{PREFILL_TOKENS, Shape};
use crate::geometry::Geometry;

pub use bundle_runtime::context_parallel::{BARRIER_WORDS, bucket};

/// Rows of the largest group launch: an eager prefill's indexer piece (the
/// bundle's `step::PIECE_ROWS`). Decode and verify buckets are smaller.
pub const MAX_BUCKET: u32 = 2048;

/// Every bucket, smallest first; bucket `1 << i` is index `i`.
pub fn buckets() -> impl Iterator<Item = u32> {
    group::buckets(MAX_BUCKET)
}

/// The index of `rows`' bucket among [`buckets`].
pub fn bucket_index(rows: u32) -> Result<usize, String> {
    group::bucket_index(rows, MAX_BUCKET)
}

/// The per-rank sizes the group buffers follow.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Dims {
    /// Latent heads per rank.
    pub heads: u32,
    /// Latent width.
    pub latent: u32,
    /// Entries of a key list row: the top-k plus the incomplete pool tail.
    pub list_width: u32,
    /// Pools one selection keeps: the top-k over the pool size.
    pub kept: u32,
    /// Rows of the scratch the localized positions and lists span.
    pub scratch_rows: u32,
}

impl Dims {
    /// The dimensions of a served shape.
    pub const fn of(shape: Shape) -> Self {
        let p = shape.profile;
        Self {
            heads: p.latent_heads,
            latent: p.kv_rank,
            list_width: p.index_topk_max + p.index_pool - 1,
            kept: p.index_topk_max / p.index_pool,
            scratch_rows: PREFILL_TOKENS,
        }
    }

    /// The dimensions of geometry `g` at `ranks` ranks, for capacity
    /// planning.
    pub const fn planned(g: &Geometry, ranks: u32) -> Self {
        let s = &g.shape;
        Self {
            heads: s.latent.heads / ranks,
            latent: s.latent.kv_rank,
            list_width: g.limits.index_topk + s.indexer.pool - 1,
            kept: g.limits.index_topk / s.indexer.pool,
            scratch_rows: PREFILL_TOKENS,
        }
    }

    /// The sizes the group exchanges: the latent heads' queries and
    /// partials, and the kept pools' pairs.
    pub const fn group(self) -> GroupDims {
        GroupDims {
            heads: self.heads,
            width: self.latent,
            kept: self.kept,
            scratch_rows: self.scratch_rows,
        }
    }
}

/// Bytes of one rank's context-parallel buffers for groups of `size`
/// (`bundle_runtime::context_parallel::RankBytes`), the model's own buffer
/// being its localized key lists (scratch rows of the list width).
pub fn rank_bytes(dims: Dims, size: u32) -> Result<RankBytes, String> {
    let lists = u64::from(dims.scratch_rows) * u64::from(dims.list_width) * 4;
    RankBytes::new(
        dims.group(),
        size,
        MAX_BUCKET,
        lists,
        Selected::group_gather,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One rank's buffers at TP4: `size` received blocks of 2048 rows (0.25
    /// GiB at size 4, which the row exchange's `size * size` blocks made
    /// 1.00 GiB), beside as many send blocks and the gathered queries.
    #[test]
    fn rank_bytes_follow_the_bucket() {
        let dims = Dims::of(Shape::TP4);
        assert_eq!(dims, Dims::planned(&crate::geometry::FLASH, 4));
        assert_eq!(
            (dims.heads, dims.latent, dims.list_width, dims.kept),
            (16, 512, 2051, 512)
        );
        assert_eq!(buckets().count(), 12);
        assert_eq!(bucket_index(24).unwrap(), 5);
        assert!(bucket_index(MAX_BUCKET + 1).is_err());
        let block = 2048 * 16 * 514 * 4;
        for (size, total) in [(2, 446_034_672), (4, 816_387_400)] {
            let bytes = rank_bytes(dims, size).unwrap();
            assert_eq!(bytes.send, u64::from(size) * block);
            assert_eq!(bytes.receive, u64::from(size) * block);
            assert_eq!(
                bytes.partial_gathers[11].flag_bytes,
                2 * u64::from(size) * 4112 * 4
            );
            assert_eq!(bytes.barrier_output, u64::from(size) * 4);
            assert_eq!(bytes.queries, u64::from(size) * 2048 * 16 * 512 * 2);
            assert_eq!(bytes.gathered_pairs[11], u64::from(size) * 2048 * 1024 * 4);
            assert_eq!(bytes.total(), total, "size {size}");
        }
        assert_eq!(rank_bytes(dims, 4).unwrap().receive, 269_484_032);
    }
}
