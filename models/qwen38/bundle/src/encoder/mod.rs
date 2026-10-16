//! Qwen3.8's vision encoder (media contract, "Qwen3.8 Flash"): its image
//! and indexes, run by the shared encoder runtime
//! (`bundle_runtime::encoder`) from the vision declaration, its media
//! arena and the prefill injection.

mod acquire;
pub mod views;

pub(crate) use acquire::acquire;
pub(crate) use acquire::arena_for;
pub use acquire::configured;
#[cfg(test)]
pub use acquire::{ARENA, CONFIG};
pub use bundle_runtime::encoder::{Encoder, Rank};

pub use model_qwen38_contract::encoder::*;
