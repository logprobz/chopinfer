//! The split Sinkhorn mHC mix-collapse fused with the input RMSNorm, its
//! lab mutants and the row engine's RMSNorm it is compared with
//! (`sinkhorn_hc::mix_norm::lab_cubin_specs`).

use kernel_build::CubinSpec;

pub const FAMILY: &str = "sinkhorn_hc_mix_norm";

pub fn cubins() -> Vec<CubinSpec> {
    kernels_sm120::sinkhorn_hc::mix_norm::lab_cubin_specs()
}
