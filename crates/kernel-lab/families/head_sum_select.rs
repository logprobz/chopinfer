//! The head-sum selection at its lab profile (`head_sum_shapes.rs`).

use kernel_build::CubinSpec;

#[path = "../head_sum_shapes.rs"]
pub mod shapes;

pub use shapes::FAMILY;

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(&[shapes::CUBIN], vec![shapes::cubin()])
}
