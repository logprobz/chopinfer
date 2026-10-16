//! Qwen's context-parallel groups: the group sizes each topology serves,
//! the row buckets of a group launch, the per-rank sizes the group buffers
//! follow and the bytes of one rank's buffers
//! (`bundle_runtime::context_parallel`). The entries themselves are the
//! backend's (`crate::sm120::context_parallel` on SM120).

use bundle_runtime::context_parallel as group;

use crate::backend::{Kernels, Selected};
use crate::contract::{PREFILL_TOKENS, Ranks, SCORE_ROWS, Shape};

pub use bundle_runtime::context_parallel::{
    BARRIER_WORDS, GatherBytes, GroupDims, RankBytes, bucket,
};

/// The group sizes above 1 the TP4 topology serves. Ranks {0, 1} hold KV
/// head 0 and ranks {2, 3} KV head 1, so a group of two consecutive ranks
/// is one replica set and splits its duplicated pages. TP2 ranks hold
/// distinct KV heads and serve size 1 only.
pub const SIZES: [u32; 1] = [2];

/// The geometries that serve a context-parallel size above 1, by id: the
/// ones whose group lowering has been checked. A geometry opts in here,
/// with its cubins in the family's lock; a data-only geometry addition
/// serves size 1 and adds no family record.
pub const GEOMETRIES: [&str; 1] = ["flash"];

/// The context-parallel sizes above 1 `shape`'s topology serves: [`SIZES`]
/// at TP4 for a geometry of [`GEOMETRIES`] whose every group entry the
/// backend instantiates, none otherwise. The package's topology, the
/// cubin build and create's admission all read this one list, so any other
/// geometry builds and serves at size 1 only.
pub fn sizes(shape: Shape) -> &'static [u32] {
    if shape.ranks() == Ranks::TP4.get()
        && GEOMETRIES.contains(&shape.geometry.id)
        && Selected::context_parallel_entries(shape)
    {
        &SIZES
    } else {
        &[]
    }
}

/// Rows of the largest group launch: a prefill selection piece
/// (`SCORE_ROWS`). Attention runs in the same pieces; decode and verify
/// buckets are smaller.
pub const MAX_BUCKET: u32 = SCORE_ROWS;

/// Every bucket, smallest first; bucket `1 << i` is index `i`.
pub fn buckets() -> impl Iterator<Item = u32> {
    group::buckets(MAX_BUCKET)
}

/// The index of `rows`' bucket among [`buckets`].
pub fn bucket_index(rows: u32) -> Result<usize, String> {
    group::bucket_index(rows, MAX_BUCKET)
}

/// The per-rank sizes of `shape`'s group buffers: a TP4 rank's Q heads
/// over its KV head, the head width and the kept blocks, and a prefill
/// chunk of localized positions.
pub fn dims(shape: Shape) -> GroupDims {
    let g = shape.geometry;
    let ranks = Ranks::TP4.get();
    GroupDims {
        heads: g.attention_q_heads() / ranks * (g.attention_kv_heads() / ranks).max(1),
        width: g.head_dim(),
        kept: g.top_blocks(),
        scratch_rows: PREFILL_TOKENS,
    }
}

/// Bytes of one rank's context-parallel buffers for group size `size`
/// (`bundle_runtime::context_parallel::RankBytes`), the model's own buffer
/// being the selection's split workspace of the widest piece of any row
/// count.
pub fn rank_bytes(shape: Shape, size: u32) -> Result<RankBytes, String> {
    if !sizes(shape).contains(&size) {
        return Err(format!(
            "{} serves no context-parallel size {size}",
            shape.topology().id
        ));
    }
    let workspace = (1..=MAX_BUCKET)
        .map(|rows| Selected::selection_workspace(shape, rows))
        .max()
        .unwrap_or(0);
    RankBytes::new(dims(shape), size, MAX_BUCKET, workspace, |size, words| {
        Selected::group_gather(size, words).map_err(|refusal| refusal.to_string())
    })
}
