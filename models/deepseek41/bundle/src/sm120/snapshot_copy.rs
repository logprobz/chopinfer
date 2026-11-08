//! The shared snapshot copy (`kernels_sm120::snapshot_copy`) the build
//! compiles under its kernel-base symbol, in a family of its own: prefix
//! captures and restores launch it (`instance::prefix`), and every instance
//! loads it.

use kernel_build::CubinSpec;
use kernels_sm120::snapshot_copy::{cubin_spec as shared_spec, entries};

/// The lock family of the snapshot copy's cubin.
pub const FAMILY: &str = "snapshot_copy";
pub const CUBIN: &str = "snapshot_copy_sm120";
/// The entry's symbol, which the fake host runs as the copies it names.
#[cfg(test)]
pub const SYMBOL: &str = kernels_sm120::snapshot_copy::SYMBOL;

/// The cubin of the shared entry.
pub fn cubin_spec() -> CubinSpec {
    shared_spec(CUBIN, &entries())
}

/// `sm120` with the shared entry registered, which every instance prepares
/// (`instance::prefix::SnapshotKernel`).
pub fn register(sm120: kernels_sm120::backend::Sm120) -> kernels_sm120::backend::Sm120 {
    entries()
        .into_iter()
        .fold(sm120, |sm120, entry| sm120.with_snapshot_copy(entry))
}
