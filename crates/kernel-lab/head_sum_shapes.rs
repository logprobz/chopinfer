//! The head-sum selection's lab profile at the block-sparse indexer's shape:
//! four heads of 128, blocks of 4, top 2048 tokens (512 blocks) with the
//! always-selected tail, 64-token pages. Shared by build.rs, which builds
//! its cubin in the `head_sum_select` lock family, and the selection tests
//! that compare it with the block-sparse indexer.

use kernel_build::CubinSpec;
use kernels_sm120::pooled_indexer::head_sum::{Entry, Geometry, cubin_spec, entries};
use kernels_sm120::{Profile, ProfileNames};

pub const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "head-sum-lab",
    tag: "hsum_lab",
});

/// The lock family of the profile's cubin in `kernels.lock.d`.
#[allow(dead_code)] // Read by the build script only.
pub const FAMILY: &str = "head_sum_select";

/// The profile's cubin name (`pooled_indexer::head_sum::cubin_spec`).
pub const CUBIN: &str = "pooled_indexer_head_sum_hsum_lab_sm120";

pub const GEOMETRY: Geometry = Geometry {
    heads: 4,
    head_dim: 128,
    group: 4,
    page_size: 64,
    max_top_k: 2048,
    always_tail: true,
};

pub fn lab_entries() -> Vec<Entry> {
    entries(PROFILE, GEOMETRY)
}

#[allow(dead_code)] // Read by the build script only.
pub fn cubin() -> CubinSpec {
    cubin_spec(PROFILE, &lab_entries()).expect("lab head-sum entries are valid")
}
