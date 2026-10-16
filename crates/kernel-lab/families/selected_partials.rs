//! The selected attention partials at their lab profile
//! (`selected_partials_shapes.rs`).

use kernel_build::CubinSpec;

#[path = "../selected_partials_shapes.rs"]
pub mod shapes;

pub use shapes::FAMILY;

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(&[shapes::CUBIN], vec![shapes::cubin()])
}
