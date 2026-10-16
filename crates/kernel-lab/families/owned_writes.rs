//! The owner-routed writes and block-id merges at their lab profile
//! (`owned_writes_shapes.rs`).

use kernel_build::CubinSpec;

#[path = "../owned_writes_shapes.rs"]
pub mod shapes;

pub use shapes::FAMILY;

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(&[shapes::CUBIN, shapes::MERGE_CUBIN], shapes::cubins())
}
