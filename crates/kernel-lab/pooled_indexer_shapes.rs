//! The pooled indexer's lab profile: the served replicated-indexer shape
//! (32 heads of 128, pools of 4, top 2048 with the always-selected tail,
//! 64-token pages, BF16 position embedding, verify items of 6 rows). Shared
//! by build.rs, which builds its cubin in the `pooled_indexer` lock family,
//! and `tests/pooled_indexer.rs`.

use kernel_core::contract::DType;
use kernels_sm120::pooled_indexer::{Entry, Geometry, entries};
use kernels_sm120::{Profile, ProfileNames};

pub const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "pooled-indexer-lab",
    tag: "pidx_lab",
});

/// The lock family of the profile's cubin in `kernels.lock.d`.
#[allow(dead_code)] // Read by the build script only.
pub const FAMILY: &str = "pooled_indexer";

/// The profile's cubin name (`pooled_indexer::cubin_spec`).
pub const CUBIN: &str = "pooled_indexer_pidx_lab_sm120";

pub fn geometry() -> Geometry {
    Geometry {
        heads: 32,
        head_dim: 128,
        group: 4,
        page_size: 64,
        max_top_k: 2048,
        always_tail: true,
        position_dtype: DType::Bf16,
        key_norm_eps: 1e-6,
        verify_rows: 6,
    }
}

pub fn lab_entries() -> Vec<Entry> {
    entries(PROFILE, geometry())
}
