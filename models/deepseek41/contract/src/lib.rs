//! DeepSeek V4.1 Flash (`deepseek41`) declarations shared by the compiler,
//! the bundle and the kernel test crate: the `deepseek41.v1` serving
//! contract, its topologies, geometry, admission and projection
//! declarations.
#![forbid(unsafe_code)]
pub mod admission;
pub mod dspark_options;
pub mod encoder;
pub mod encoder_plan;
pub mod frontend;
pub mod geometry;
pub mod hc;
pub mod indexer;
pub mod latent;
pub mod latent_glue;
pub mod prefill_cone;
pub mod prefill_profile;
pub mod projections;
pub mod structure;

pub mod moe;
pub mod own_ops;
pub mod rows;
pub mod serving;
pub mod state;
pub mod weight_plan;

/// The model's one bundle ID. Every rank count it supports is a topology of
/// this bundle, never a bundle ID of its own.
pub const BUNDLE_ID: &str = "deepseek41.v1";

/// The production topology: the flash geometry at four ranks.
pub const FLASH_TP4_TOPOLOGY: &str = "deepseek41-flash-tp4";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topologies_belong_to_the_one_bundle() {
        assert_eq!(BUNDLE_ID, "deepseek41.v1");
        assert!(FLASH_TP4_TOPOLOGY.starts_with("deepseek41-"));
    }
}
