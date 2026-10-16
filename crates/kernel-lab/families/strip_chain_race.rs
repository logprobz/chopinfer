//! The strip-chain stage-release race's lab instances: each served
//! multi-split schedule and the vocabulary head's two-split schedule on the
//! locked and the served kernel beside the one-split entry, and the
//! contention probe (`strip_chain_race_shapes.rs`).

use kernel_build::CubinSpec;

#[path = "../strip_chain_race_shapes.rs"]
pub mod shapes;

pub use shapes::FAMILY;

pub fn cubins() -> Vec<CubinSpec> {
    let names = shapes::cubin_names();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    crate::declared(&names, shapes::cubins())
}
