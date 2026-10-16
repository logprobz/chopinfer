//! The decoded-tile GEMM's lab instances (FP8 128x128 prefill projections
//! with each weight code decoded once per CTA): the prefill sweep beside the
//! entries it would replace, and the mutants (`decoded_tile_shapes.rs`).

use kernel_build::CubinSpec;

#[path = "../decoded_tile_shapes.rs"]
pub mod shapes;

pub use shapes::FAMILY;

pub fn cubins() -> Vec<CubinSpec> {
    let names = shapes::cubin_names();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    crate::declared(&names, shapes::cubins())
}
