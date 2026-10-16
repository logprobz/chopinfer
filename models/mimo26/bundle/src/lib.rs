//! Native MiMo V2.6 Flash TP4 bundle.

#![deny(unsafe_code)]

pub mod attention_plan;
pub mod contract;
mod contract_note;
pub mod dflash;
mod encoder_availability;
pub mod encoders;
pub mod ga_pool;
pub mod graphs;
pub mod index;
pub mod inputs;
mod instance;
pub mod mtp;
pub mod ops;
pub mod overlap;
pub mod prefix;
mod prefix_projected;
pub mod program;
pub mod program_lower;
pub mod projection;
pub mod rank_views;
pub mod retain;
pub mod ring;
mod rope_tables;
pub mod row_plan;
pub mod scratch;
pub mod selection;
pub mod sm120;
pub mod switches;
#[cfg(feature = "diagnostic-taps")]
pub mod taps;
pub mod target;
pub mod topology;
pub mod upload;

#[cfg(not(test))]
pub use bundle_runtime::pipeline;
pub use bundle_runtime::{dispatch, logits};
#[cfg(test)]
#[path = "launch_rows.rs"]
pub mod pipeline;

use bundle_runtime::entry;
pub use contract_note::kernel_contract_identity;
use instance::Instance;

#[cfg(test)]
mod fake_host_tests;

include!(concat!(env!("OUT_DIR"), "/embedded.rs"));
include!(concat!(env!("OUT_DIR"), "/geometry-cubins.rs"));

/// The bytes of the embedded cubin `name` (`mimo_tp2_ring`, ...), for
/// kernel-lab to run the bundle's own kernels.
pub fn embedded_cubin(name: &str) -> Option<&'static [u8]> {
    CUBINS
        .iter()
        .find(|cubin| cubin.name == name)
        .map(|cubin| cubin.bytes)
}

// The shared entry bodies over MiMo's instance (`bundle_runtime::entry`).
runtime_bundle_abi::export_bundle_api_v1!(
    with_prefix_cache
    BUNDLE_MIMO26_API_V1,
    BUNDLE_MIMO26_EXTENDED_API_V1,
    entry::create::<Instance>,
    entry::execute::<Instance>,
    entry::commit::<Instance>,
    entry::cancel::<Instance>,
    entry::destroy::<Instance>,
    entry::retire_request::<Instance>,
    entry::commit_accepted::<Instance>,
    entry::describe_prefix_state::<Instance>,
    entry::capture_prefix_state::<Instance>,
    entry::restore_prefix_state::<Instance>,
);

// Direct checkpoint loading: the text, drafter and encoder images' own plans,
// from the checkpoint headers.
runtime_bundle_abi::export_weight_plan_v1!(model_mimo26_contract::weight_plan::plan);

// The chat frontend: MiMo's chat template ported to Rust, over this
// package's frontend component (`chop.frontend.v1`).
frontend_format::export_chat_format_v1!(model_mimo26_contract::frontend::Mimo26Format);

#[cfg(test)]
#[path = "sm120/resource_tests.rs"]
mod resource_tests;
