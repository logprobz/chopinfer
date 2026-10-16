//! The Sinkhorn mHC mix-collapse with rows tiled over a CTA at every tile
//! the kernel takes, their probes and the served tile's lab mutants
//! (`sinkhorn_hc::tiled::lab_cubin_specs`).

use kernel_build::CubinSpec;

pub const FAMILY: &str = "sinkhorn_hc_tiled";

pub fn cubins() -> Vec<CubinSpec> {
    kernels_sm120::sinkhorn_hc::tiled::lab_cubin_specs()
}
