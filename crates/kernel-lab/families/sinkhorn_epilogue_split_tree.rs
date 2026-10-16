//! The split Sinkhorn mHC comm epilogue with the input norm's 256-lane tree
//! sum and its lab mutants (`comm::sinkhorn_split_tree::lab_cubin_specs`).

use kernel_build::CubinSpec;

pub const FAMILY: &str = "sinkhorn_epilogue_split_tree";

pub fn cubins() -> Vec<CubinSpec> {
    kernels_sm120::comm::sinkhorn_split_tree::lab_cubin_specs()
}
