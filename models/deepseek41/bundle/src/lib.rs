//! The DeepSeek V4.1 Flash bundle (`deepseek41.v1`). Generic code states the
//! program and reaches kernels only through `kernel_core::backend` and the
//! model's `OwnOps`; SM120 registration lives in the `sm120` module.
#![deny(unsafe_code)]

pub mod buffers;
pub mod config;
#[cfg(test)]
mod fake_host_tests;
mod instance;
pub mod lower;
pub mod ops;
pub mod program;
pub mod shape;
mod sm120;

pub use model_deepseek41_contract::BUNDLE_ID;

include!(concat!(env!("OUT_DIR"), "/embedded.rs"));

use bundle_runtime::entry;
use instance::Instance;

// The shared entry bodies over the deepseek41 instance
// (`bundle_runtime::entry`): retirement frees a request's slot, a verify
// step commits the rows the host accepted, and prefix snapshots capture and
// restore a slot's rings and tails; a restore fills the host's state slot.
runtime_bundle_abi::export_bundle_api_v1!(
    with_prefix_state_slot BUNDLE_DEEPSEEK41_API_V1,
    BUNDLE_DEEPSEEK41_EXTENDED_API_V1,
    entry::create::<Instance>,
    entry::execute::<Instance>,
    entry::commit::<Instance>,
    entry::cancel::<Instance>,
    entry::destroy::<Instance>,
    entry::retire_request::<Instance>,
    [entry::commit_accepted::<Instance>],
    entry::describe_prefix_state::<Instance>,
    entry::capture_prefix_state::<Instance>,
    entry::restore_prefix_state::<Instance>,
    entry::restore_prefix_state_in_slot::<Instance>,
);

/// The kernels an instance prepares its ops on: a backend of the shared
/// contracts and one of the model's own ops. The target's module supplies
/// them (`Kernels::new` in `sm120`).
pub(crate) struct Kernels {
    pub(crate) backend: Box<dyn kernel_core::backend::Backend>,
    pub(crate) own: Box<dyn model_deepseek41_contract::own_ops::OwnOps>,
}
