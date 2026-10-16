//! The paged attention partials and their merges at the lab profile
//! (`paged_partials_shapes.rs`).

use kernel_build::CubinSpec;

#[path = "../paged_partials_shapes.rs"]
pub mod shapes;

pub use shapes::FAMILY;

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(&[shapes::CUBIN, shapes::MERGE_CUBIN], shapes::cubins())
}
