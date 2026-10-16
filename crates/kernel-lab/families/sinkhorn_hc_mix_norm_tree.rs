//! The split Sinkhorn mHC mix-collapse fused with the input RMSNorm whose
//! sum of squares is the 256-lane tree, and its lab mutants
//! (`sinkhorn_hc::mix_norm_tree::lab_cubin_specs`).

use kernel_build::CubinSpec;

pub const FAMILY: &str = "sinkhorn_hc_mix_norm_tree";

pub fn cubins() -> Vec<CubinSpec> {
    kernels_sm120::sinkhorn_hc::mix_norm_tree::lab_cubin_specs()
}
