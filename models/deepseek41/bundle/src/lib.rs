//! The DeepSeek V4.1 Flash bundle (`deepseek41.v1`). Generic code states the
//! program and reaches kernels only through `kernel_core::backend` and the
//! model's `OwnOps`; SM120 registration lives in the `sm120` module.
#![deny(unsafe_code)]

// The bundle API, exported only to the crate's tests: no step serves
// until every stage of the program lowers, so the shipped library exports
// nothing yet.
#[cfg(test)]
mod api;
pub mod buffers;
pub mod config;
pub mod dspark;
#[allow(
    dead_code,
    reason = "the program acquires the encoder once media steps land"
)]
pub mod encoders;
pub mod engram_rows;
#[cfg(test)]
mod fake_host_tests;
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "only the tests export the bundle API that serves the instance"
    )
)]
mod instance;
pub mod lower;
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "only the tests export the bundle API that serves the instance"
    )
)]
pub mod ops;
pub mod program;
pub mod shape;
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "only the tests export the bundle API that serves the instance"
    )
)]
pub mod sm120;

pub use model_deepseek41_contract::BUNDLE_ID;

include!(concat!(env!("OUT_DIR"), "/embedded.rs"));

// Direct checkpoint loading: each rank's text image and the replicated
// encoder image planned from the checkpoint's headers
// (`model_deepseek41_contract::weight_plan`, `encoder_plan`), the encoder's
// column permutation as the bundle's own transform.
runtime_bundle_abi::export_weight_plan_v1!(model_deepseek41_contract::weight_plan::plan);
runtime_bundle_abi::export_weight_transform_v1!(model_deepseek41_contract::encoder_plan::transform);

/// The kernels an instance prepares its ops on: a backend of the shared
/// contracts and one of the model's own ops. The target's module supplies
/// them (`Kernels::new` in `sm120`).
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "only the tests export the bundle API that serves the instance"
    )
)]
pub(crate) struct Kernels {
    pub(crate) backend: Box<dyn kernel_core::backend::Backend>,
    pub(crate) own: Box<dyn model_deepseek41_contract::own_ops::OwnOps>,
}
