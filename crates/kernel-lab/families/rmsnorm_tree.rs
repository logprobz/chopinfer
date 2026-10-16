//! The row engine's HF RMSNorm with the 256-lane tree sum of squares at the
//! lab widths, and its reversed-distances mutant
//! (`row::rmsnorm_tree::lab_cubin_specs`).

use kernel_build::CubinSpec;

pub const FAMILY: &str = "rmsnorm_tree";

pub fn cubins() -> Vec<CubinSpec> {
    kernels_sm120::row::rmsnorm_tree::lab_cubin_specs()
}
