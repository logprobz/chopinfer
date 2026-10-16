//! The Sinkhorn mHC mix-collapse split over CTAs
//! (`sinkhorn_hc::split::lab_cubin_specs`).

use kernel_build::CubinSpec;

pub const FAMILY: &str = "sinkhorn_hc_split";

pub fn cubins() -> Vec<CubinSpec> {
    kernels_sm120::sinkhorn_hc::split::lab_cubin_specs()
}
