//! The latent attention engine at the lab shape (`latent_shapes.rs`).

use kernel_build::CubinSpec;

#[path = "../latent_shapes.rs"]
pub mod shapes;

pub use shapes::FAMILY;

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(&[&shapes::cubin_name()], vec![shapes::cubin()])
}
