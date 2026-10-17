//! The DeepSeek V4.1 Flash bundle (`deepseek41.v1`). Generic code states the
//! program and reaches kernels only through `kernel_core::backend` and the
//! model's `OwnOps`; SM120 registration lives in the `sm120` module.
#![deny(unsafe_code)]

pub mod program;
pub mod shape;
mod sm120;

pub use model_deepseek41_contract::BUNDLE_ID;
