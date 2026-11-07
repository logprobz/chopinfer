//! The shared snapshot copy (`kernels_sm120::snapshot_copy`) the build
//! compiles under its kernel-base symbol, in a family of its own: prefix
//! captures and restores launch it under the `prefix_snapshot_copy_kernel`
//! candidate (`instance::prefix`), and only such an instance loads it.

use kernel_build::CubinSpec;
use kernels_sm120::snapshot_copy::{cubin_spec as shared_spec, entries};

/// The lock family of the snapshot copy's cubin.
pub const FAMILY: &str = "snapshot_copy";
pub const CUBIN: &str = "snapshot_copy_sm120";

/// The cubin of the shared entry.
pub fn cubin_spec() -> CubinSpec {
    shared_spec(CUBIN, &entries())
}

/// `sm120` with the shared entry registered; an instance prepares it only
/// under the candidate.
pub fn register(sm120: kernels_sm120::backend::Sm120) -> kernels_sm120::backend::Sm120 {
    entries()
        .into_iter()
        .fold(sm120, |sm120, entry| sm120.with_snapshot_copy(entry))
}
