//! The context-parallel entries at their lab profile
//! (`context_parallel_shapes.rs`).

use kernel_build::CubinSpec;

#[path = "../context_parallel_shapes.rs"]
pub mod shapes;

pub use shapes::FAMILY;

pub fn cubins() -> Vec<CubinSpec> {
    let spec = kernels_sm120::context_parallel::cubin_spec(shapes::PROFILE, &shapes::lab_entries())
        .expect("lab context-parallel entries are valid");
    crate::declared(&[shapes::CUBIN], vec![spec])
}
