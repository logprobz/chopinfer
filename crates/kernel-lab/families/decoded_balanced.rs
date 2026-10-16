//! The balanced decoded-tile GEMM's lab instances (FP8 128x128 prefill
//! projections, one 128-row item per CTA with the final partial wave split
//! into 64-row halves): the prefill sweep beside the decoded tiles it would
//! replace, and the mutants (`decoded_balanced_shapes.rs`).

use kernel_build::CubinSpec;

#[path = "../decoded_balanced_shapes.rs"]
pub mod shapes;

pub use shapes::FAMILY;

pub fn cubins() -> Vec<CubinSpec> {
    let names = shapes::cubin_names();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    crate::declared(&names, shapes::cubins())
}
