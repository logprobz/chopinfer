//! The split Sinkhorn mHC mix-collapse fused with the input RMSNorm whose
//! ordered sum is the exact parallel method, and its lab mutants
//! (`sinkhorn_hc::mix_norm_exact::lab_cubin_specs`).

use kernel_build::CubinSpec;

pub const FAMILY: &str = "sinkhorn_hc_mix_norm_exact";

pub fn cubins() -> Vec<CubinSpec> {
    kernels_sm120::sinkhorn_hc::mix_norm_exact::lab_cubin_specs()
}
