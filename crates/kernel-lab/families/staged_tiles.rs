//! The staged-tile GEMM's lab instances (BF16 prefill projections in 128 x
//! 128 tiles that a loader warp stages by TMA for eight math warps, the
//! final wave split into 64-row halves): the prefill sweep beside the warp
//! tiles it would replace, and the mutants (`staged_tile_shapes.rs`).

use kernel_build::CubinSpec;

#[path = "../staged_tile_shapes.rs"]
pub mod shapes;

pub use shapes::FAMILY;

pub fn cubins() -> Vec<CubinSpec> {
    let names = shapes::cubin_names();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    crate::declared(&names, shapes::cubins())
}
