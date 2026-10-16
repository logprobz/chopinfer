//! The phased group gather (`kernels_sm120::context_parallel::phased`) at
//! its only shape: one cubin, one entry, any group of 1 to 8 ranks.

use kernel_build::CubinSpec;

pub const FAMILY: &str = "context_parallel_phased";

/// The cubin `tests/context_parallel_phased.rs` loads.
pub const CUBIN: &str = "comm_group_phased_lab_sm120";

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(
        &[CUBIN],
        vec![kernels_sm120::context_parallel::phased::cubin_spec(
            CUBIN.into(),
        )],
    )
}
