//! The chained expert engine's served dense cubins, their block-warp cubins
//! and the shared expert's block-warp mutants (`block_warp_mutant_shapes.rs`).
//! The routed experts' cubins are the split's (`moe_chained_split`).

use kernel_build::CubinSpec;
use kernels_sm120::moe::chained;

#[path = "../block_warp_mutant_shapes.rs"]
pub mod block_warp_mutants;

pub const FAMILY: &str = "moe_chained";

pub fn cubins() -> Vec<CubinSpec> {
    let mut specs = chained::cubin_specs();
    specs.extend(chained::block_warp_cubin_specs());
    specs.push(block_warp_mutants::cubin());
    specs
}
