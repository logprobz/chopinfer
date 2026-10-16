//! GLM 5.3 Flash (`glm5_next`) declarations shared by the compiler, the
//! bundle and `build.rs`: geometry, admission, checkpoint quantization
//! layouts, kernel profiles, tensor catalog, weight plan, tuning tables and
//! capacity planning, and the `glm5next.v1` serving contract (rank views, KV
//! pages, request state and fit checks), and the chat format the bundle
//! exports (`frontend`).
#![forbid(unsafe_code)]
pub mod admission;
pub mod backend;
pub mod capacity;
pub mod catalog;
pub mod context_parallel;
pub mod contract;
pub mod copies;
pub mod direct;
pub mod draft_argmax;
pub mod encoder;
pub mod encoder_plan;
pub mod fit;
pub mod frontend;
pub mod geometry;
pub mod kernels;
pub mod options;
pub mod own_ops;
pub mod payload;
pub mod precision;
pub mod prepared;
pub mod projections;
pub mod quantization;
pub mod sm120;
pub mod state;
pub mod structure;
pub mod views;
pub mod weight_plan;

#[cfg(test)]
mod fp8_pins;
#[cfg(test)]
mod nvfp4_rank_image;
