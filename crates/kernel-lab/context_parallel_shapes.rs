//! The context-parallel lab profile at the served latent shape: 16 heads
//! merged per rank, latent width 512, 64-token pages and key lists of 2048
//! selected tokens plus a 3-token tail, at group sizes 2 and 4. Shared by
//! build.rs, which builds its cubin in the `context_parallel` lock family,
//! and `tests/context_parallel.rs`.

use kernels_sm120::context_parallel::{Entry, Geometry, entries};
use kernels_sm120::{Profile, ProfileNames};

pub const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "context-parallel-lab",
    tag: "cp_lab",
});

/// The lock family of the profile's cubin in `kernels.lock.d`.
#[allow(dead_code)] // Read by the build script only.
pub const FAMILY: &str = "context_parallel";

/// The profile's cubin name (`context_parallel::cubin_spec`).
pub const CUBIN: &str = "context_parallel_cp_lab_sm120";

pub const GEOMETRY: Geometry = Geometry {
    heads: 16,
    width: 512,
    page_size: 64,
    list_width: 2051,
    capacity: 8192,
};

pub fn lab_entries() -> Vec<Entry> {
    entries(PROFILE, GEOMETRY)
}
