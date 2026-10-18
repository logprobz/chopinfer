//! The DeepSeek V4.1 Flash bundle (`deepseek41.v1`). Generic code states the
//! program and reaches kernels only through `kernel_core::backend` and the
//! model's `OwnOps`; SM120 registration lives in the `sm120` module.
#![deny(unsafe_code)]

// The bundle API over the instance (`api.rs`), which the conformance
// suite drives on the fake host (`fake_host_tests::conformance`).
mod api;
pub mod buffers;
pub mod compressor;
pub mod config;
pub mod dspark;
pub mod encoders;
pub mod engram_rows;
#[cfg(test)]
mod fake_host_tests;
mod instance;
pub mod lower;
pub mod ops;
pub mod program;
pub mod shape;
pub mod sm120;
pub mod taps;

pub use model_deepseek41_contract::BUNDLE_ID;

include!(concat!(env!("OUT_DIR"), "/embedded.rs"));
include!(concat!(env!("OUT_DIR"), "/geometry_cubins.rs"));

/// Whether `cubin` serves geometry `id`: a cubin of one geometry serves it
/// alone (`GEOMETRY_CUBINS`), every other cubin serves every geometry,
/// except the cubins no instance loads (`Kernels::loads`).
pub(crate) fn serves(cubin: &str, id: &str) -> bool {
    Kernels::loads(cubin)
        && GEOMETRY_CUBINS
            .iter()
            .find(|(name, _)| *name == cubin)
            .is_none_or(|(_, geometry)| *geometry == id)
}

// Direct checkpoint loading: each rank's text image and the replicated
// encoder image planned from the checkpoint's headers
// (`model_deepseek41_contract::weight_plan`, `encoder_plan`), the encoder's
// column permutation as the bundle's own transform.
runtime_bundle_abi::export_weight_plan_v1!(model_deepseek41_contract::weight_plan::plan);
runtime_bundle_abi::export_weight_transform_v1!(model_deepseek41_contract::encoder_plan::transform);

// The chat frontend: DeepSeek V4.1 Flash's chat format, over this package's
// frontend component (`chop.frontend.v1`).
frontend_format::export_chat_format_v1!(model_deepseek41_contract::frontend::DeepSeekV41Format);

/// The kernels an instance prepares its ops on: a backend of the shared
/// contracts and one of the model's own ops. The target's module supplies
/// them (`Kernels::new` in `sm120`).
pub(crate) struct Kernels {
    pub(crate) backend: Box<dyn kernel_core::backend::Backend>,
    pub(crate) own: Box<dyn model_deepseek41_contract::own_ops::OwnOps>,
    /// The bytes of a gather of logits rows to rank 0's flags and epoch,
    /// which every rank allocates zeroed for the backend's entry.
    pub(crate) gather_state: [u64; 2],
}
