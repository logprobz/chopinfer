//! The selected attention partials lab profile at a TP4 rank of the
//! block-sparse attention: six query heads over one KV head of 256, 64-token
//! pages, blocks of 4 and 512 selected blocks. Shared by build.rs, which
//! builds its cubin in the `selected_partials` lock family, and the
//! selected partial tests; the partial merges are the paged partials
//! family's (`paged_partials_shapes.rs`).

use kernel_build::CubinSpec;
use kernels_sm120::selected_partials::{Entry, Geometry, cubin_spec, entries};
use kernels_sm120::{Profile, ProfileNames};

pub const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "selected-partials-lab",
    tag: "sp_lab",
});

/// The lock family of the profile's cubin in `kernels.lock.d`.
#[allow(dead_code)] // Read by the build script only.
pub const FAMILY: &str = "selected_partials";

/// The profile's cubin (`selected_partials::cubin_spec`).
pub const CUBIN: &str = "selected_partials_sp_lab_sm120";

pub const GEOMETRY: Geometry = Geometry {
    q_heads: 6,
    kv_heads: 1,
    head_dim: 256,
    page: 64,
    block: 4,
    top_blocks: 512,
};

pub fn lab_entries() -> Vec<Entry> {
    entries(PROFILE, GEOMETRY)
}

#[allow(dead_code)] // Read by the build script only.
pub fn cubin() -> CubinSpec {
    cubin_spec(PROFILE, &lab_entries()).expect("lab selected partial entries are valid")
}
