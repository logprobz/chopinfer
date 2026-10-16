//! The chained expert engine's lab-only decoded-tile projections
//! (`chained_decoded_shapes.rs`): the dense and shared-expert FP8 prefill
//! gate/up and the mutants.

use kernel_build::CubinSpec;

#[path = "../chained_decoded_shapes.rs"]
pub mod shapes;

pub const FAMILY: &str = "moe_chained_decoded";

pub fn cubins() -> Vec<CubinSpec> {
    let mut specs = shapes::cubins();
    specs.push(shapes::mutants_cubin());
    specs
}
