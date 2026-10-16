//! The owner-routed writes lab profile at a TP4 rank of the block-sparse
//! attention: six query heads over one KV head of 256 with rotary 64 in
//! M-RoPE sections [11, 11, 10], 64-token pages, and a four-head indexer of
//! 128 with blocks of 4, a 16-entry ring and 512 kept blocks; plus the
//! block-id merges of that indexer's per-rank selections at group sizes 2
//! and 4. Shared by build.rs, which builds both cubins in the
//! `owned_writes` lock family, and the context-parallel chain tests.

use kernel_build::CubinSpec;
use kernels_sm120::attention::{gated, sparse};
use kernels_sm120::owned_writes::{Entry, cubin_spec, entries};
use kernels_sm120::pooled_indexer::{block_merge, head_sum};
use kernels_sm120::{Profile, ProfileNames};

pub const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "owned-writes-lab",
    tag: "ow_lab",
});

/// The lock family of the profile's cubins in `kernels.lock.d`.
#[allow(dead_code)] // Read by the build script only.
pub const FAMILY: &str = "owned_writes";

/// The owner-routed writes' cubin (`owned_writes::cubin_spec`).
pub const CUBIN: &str = "owned_writes_ow_lab_sm120";
/// The block-id merges' cubin (`block_merge::cubin_spec`).
pub const MERGE_CUBIN: &str = "block_merge_ow_lab_sm120";

/// The group size the writes serve.
pub const RANKS: u32 = 2;

pub const GATED: gated::Config = gated::Config {
    q_heads: 6,
    kv_heads: 1,
    head_dim: 256,
    rotary: 64,
    theta: 1e7,
    sections: [11, 11, 10],
    eps: 1e-6,
    page: 64,
    max_rows: 8192,
};

pub const INDEXER: sparse::Geometry = sparse::Geometry {
    index_heads: 4,
    index_dim: 128,
    rotary_dim: 64,
    theta: 1e7,
    mrope: [11, 11, 10],
    eps: 1e-6,
    block: 4,
    ring: 16,
    top_blocks: 512,
    max_context: 262_144,
    q_heads: 6,
    kv_heads: 1,
    head_dim: 256,
    page_size: 64,
    softmax_scale: 0.0625,
};

/// The head-sum selection whose per-rank pairs the merges fold.
pub const SELECTION: head_sum::Geometry = head_sum::Geometry {
    heads: 4,
    head_dim: 128,
    group: 4,
    page_size: 64,
    max_top_k: 2048,
    always_tail: true,
};

pub fn lab_entries() -> Vec<Entry> {
    entries(PROFILE, GATED, INDEXER, RANKS)
}

pub fn merge_entries() -> Vec<block_merge::Entry> {
    block_merge::entries(PROFILE, SELECTION, &[2, 4])
}

#[allow(dead_code)] // Read by the build script only.
pub fn cubins() -> Vec<CubinSpec> {
    vec![
        cubin_spec(PROFILE, &lab_entries()).expect("lab owner-routed writes are valid"),
        block_merge::cubin_spec(PROFILE, &merge_entries()).expect("lab block merges are valid"),
    ]
}
