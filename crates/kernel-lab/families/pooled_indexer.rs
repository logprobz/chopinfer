//! The pooled indexer at its lab profile (`pooled_indexer_shapes.rs`).

use kernel_build::CubinSpec;

#[path = "../pooled_indexer_shapes.rs"]
pub mod shapes;

pub use shapes::FAMILY;

pub fn cubins() -> Vec<CubinSpec> {
    let spec = kernels_sm120::pooled_indexer::cubin_spec(shapes::PROFILE, &shapes::lab_entries())
        .expect("lab pooled indexer entries are valid");
    crate::declared(&[shapes::CUBIN], vec![spec])
}
