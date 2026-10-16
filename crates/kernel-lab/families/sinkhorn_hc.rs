//! The Sinkhorn-mixed hyper-connection engine's lab cubins (`sinkhorn_hc::lab_cubin_specs`).

use kernel_build::CubinSpec;

pub const FAMILY: &str = "sinkhorn_hc";

pub fn cubins() -> Vec<CubinSpec> {
    kernels_sm120::sinkhorn_hc::lab_cubin_specs()
}
