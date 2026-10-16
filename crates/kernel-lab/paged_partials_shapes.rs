//! The paged attention partials lab profile at a TP4 rank of a grouped-query
//! model with replicated KV heads: six query heads over one KV head of 256,
//! 64-token pages, plus the partial merge of those six heads at group sizes
//! 2 and 4. Shared by build.rs, which builds both cubins in the
//! `paged_partials` lock family, and `tests/paged_partials.rs`.

use kernel_build::CubinSpec;
use kernels_sm120::context_parallel;
use kernels_sm120::paged_partials::{Entry, Geometry, cubin_spec, entries};
use kernels_sm120::{Profile, ProfileNames};

pub const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "paged-partials-lab",
    tag: "pp_lab",
});

/// The lock family of the profile's cubins in `kernels.lock.d`.
#[allow(dead_code)] // Read by the build script only.
pub const FAMILY: &str = "paged_partials";

/// The partial entries' cubin (`paged_partials::cubin_spec`).
pub const CUBIN: &str = "paged_partials_pp_lab_sm120";
/// The merge entries' cubin (`context_parallel::cubin_spec`).
pub const MERGE_CUBIN: &str = "context_parallel_pp_lab_sm120";

pub const GEOMETRY: Geometry = Geometry {
    q_heads: 6,
    kv_heads: 1,
    head_dim: 256,
    page: 64,
};

pub fn lab_entries() -> Vec<Entry> {
    entries(PROFILE, GEOMETRY)
}

/// The partial merges of a rank's six heads of 256.
pub fn merge_entries() -> Vec<context_parallel::Entry> {
    context_parallel::entries(
        PROFILE,
        context_parallel::Geometry {
            heads: GEOMETRY.q_heads * GEOMETRY.kv_heads,
            width: GEOMETRY.head_dim,
            page_size: GEOMETRY.page,
            list_width: 1,
            capacity: 8192,
        },
    )
    .into_iter()
    .filter(|entry| matches!(entry.kind, context_parallel::Kind::PartialMerge { .. }))
    .collect()
}

#[allow(dead_code)] // Read by the build script only.
pub fn cubins() -> Vec<CubinSpec> {
    vec![
        cubin_spec(PROFILE, &lab_entries()).expect("lab paged partial entries are valid"),
        context_parallel::cubin_spec(PROFILE, &merge_entries())
            .expect("lab paged partial merges are valid"),
    ]
}
