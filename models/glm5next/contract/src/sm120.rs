//! The contract's SM120 selections that name `kernels_sm120` directly: the
//! backend's facts ([`Sm120`], `crate::backend::Kernels`), the catalog the
//! shared kernels' consumers read ([`CATALOG`]), the engine entries
//! (catalogs, lock families, comm sizing), the projections' offers, copies
//! and bindings and the NVFP4 layouts' added entries, the context-parallel
//! and vision encoder entries, the stacked KDA input projections, and the
//! payload admission and state copy entries.

pub mod backend;
pub mod catalog;
pub mod context_parallel;
pub mod encoder;
pub mod engines;
pub mod payload;
#[cfg(test)]
mod payload_tests;
pub mod precision;
pub mod prepared;
pub mod projection_formats;
pub mod stacked;
pub mod state_copy;
pub mod tuning;

pub use backend::Sm120;
pub use catalog::CATALOG;
