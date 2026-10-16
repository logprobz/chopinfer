//! The stacked strip-chain GEMM's and the L2 weight prefetch's lab
//! instances, beside the entries they are checked and timed against
//! (`strip_chain_stacked_shapes.rs`).

use kernel_build::CubinSpec;

#[path = "../strip_chain_stacked_shapes.rs"]
pub mod shapes;

pub use shapes::FAMILY;

pub fn cubins() -> Vec<CubinSpec> {
    let names = shapes::cubin_names();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    crate::declared(&names, shapes::cubins())
}
