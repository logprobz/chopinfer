//! The chain-split tiled GEMM's lab instances, unmodified and mutated
//! (`chain_mutant_shapes.rs`).

use kernel_build::CubinSpec;

#[path = "../chain_mutant_shapes.rs"]
pub mod shapes;

pub use shapes::FAMILY;

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(&[shapes::CUBIN], vec![shapes::cubin()])
}
