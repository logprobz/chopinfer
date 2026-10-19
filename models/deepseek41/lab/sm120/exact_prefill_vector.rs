//! The lab and bundle compile the same qualified vector declaration.
#[path = "../../bundle/src/sm120/moe_pipeline_vector.rs"]
mod frozen;
pub const CUBIN: &str = frozen::CUBIN;
pub const FAMILY: &str = frozen::FAMILY;
pub const VARIANTS: [&str; 3] = frozen::VARIANTS;
pub fn symbol(name: &str) -> String {
    frozen::symbol(name)
}
pub fn cubins() -> Vec<kernel_build::CubinSpec> {
    frozen::cubins(super::family::cubins())
}
