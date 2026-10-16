//! The split Sinkhorn mHC mix-collapse fused with the input RMSNorm on
//! batched loads, its lab mutants and the phase probes of it and of the
//! exact-sum entry (`sinkhorn_hc::mix_norm_batched::lab_cubin_specs`).

use kernel_build::CubinSpec;

pub const FAMILY: &str = "sinkhorn_hc_mix_norm_batched";

pub fn cubins() -> Vec<CubinSpec> {
    kernels_sm120::sinkhorn_hc::mix_norm_batched::lab_cubin_specs()
}
