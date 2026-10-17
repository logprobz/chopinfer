//! The lagged Sinkhorn mHC engine's lab cubins: its served mix-collapse, F32
//! expand and weighted head, its probe and the per-sublayer twin probe
//! (`sinkhorn_hc::lagged::lab_cubin_specs`).

use kernel_build::CubinSpec;

pub const FAMILY: &str = "sinkhorn_hc_lagged";

pub fn cubins() -> Vec<CubinSpec> {
    kernels_sm120::sinkhorn_hc::lagged::lab_cubin_specs()
}
