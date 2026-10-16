//! The vision encoder row entries at their served shapes
//! (`encoder_vision_shapes.rs`).

use kernel_build::CubinSpec;

#[path = "../encoder_vision_shapes.rs"]
pub mod shapes;

pub use shapes::FAMILY;

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(&[shapes::CUBIN], vec![shapes::cubin()])
}
