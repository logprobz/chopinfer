//! The strip-chain tiled GEMM's lab instances: the decode sweep beside the
//! entries it would replace, and the mutants (`strip_chain_shapes.rs`).

use kernel_build::CubinSpec;

#[path = "../strip_chain_shapes.rs"]
pub mod shapes;

pub use shapes::FAMILY;

pub fn cubins() -> Vec<CubinSpec> {
    let names = shapes::cubin_names();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    crate::declared(&names, shapes::cubins())
}
