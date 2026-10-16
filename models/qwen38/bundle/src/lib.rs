//! Native Qwen3.8 Flash bundle at 2 or 4 ranks: prefill, decode, and MTP
//! speculation with verify rows.
//!
//! The model program (layer plan, lowering, buffers, recurrent state) is
//! here; dispatch, graphs, input packing, upload, logits views, resources
//! and the instance table come from `bundle-runtime`. See
//! `docs/bundle-qwen38.md`.

#![deny(unsafe_code)]

pub mod precision;
mod preparation_note;
pub mod projections;

pub mod context_parallel;
pub mod contract;
pub mod drafter;
pub mod dump;
pub mod encoder;
mod instance;
pub mod launch;
pub mod lower;
mod ops;
pub mod prefix;
pub mod program;
pub mod scratch;
pub mod side;
pub mod sm120;
pub mod state;
mod target;
pub mod views;

#[cfg(not(test))]
pub use bundle_runtime::pipeline;
pub use bundle_runtime::{dispatch, inputs, logits};
#[cfg(test)]
#[path = "launch_rows.rs"]
pub mod pipeline;

use bundle_runtime::entry;
use instance::Instance;

/// Qwen3.8's graph buckets over the shared graph cache.
pub mod graphs {
    pub use bundle_runtime::graphs::{GraphKey, pieces};

    use crate::contract::Qwen;

    pub type GraphCache = bundle_runtime::graphs::GraphCache<Qwen>;

    pub fn decode_bucket(live: u32) -> Option<u32> {
        bundle_runtime::graphs::decode_bucket::<Qwen>(live)
    }
}

/// Qwen3.8's input arenas over the shared layout.
pub mod upload {
    pub use bundle_runtime::upload::{DeviceInputs, InputLayout, PhaseArena, RowWrite};

    pub type BlockTableMirror = bundle_runtime::upload::BlockTableMirror<crate::contract::Qwen>;
}

#[cfg(test)]
mod fake_host_tests;

include!(concat!(env!("OUT_DIR"), "/embedded.rs"));
include!(concat!(env!("OUT_DIR"), "/geometry-cubins.rs"));

/// The bytes of the embedded cubin `name`, for kernel-lab to run the
/// bundle's own kernels.
pub fn embedded_cubin(name: &str) -> Option<&'static [u8]> {
    CUBINS
        .iter()
        .find(|cubin| cubin.name == name)
        .map(|cubin| cubin.bytes)
}

// The shared entry bodies over Qwen3.8's instance
// (`bundle_runtime::entry`). Restore needs the request's state slot,
// so the host restores through the slot tail.
runtime_bundle_abi::export_bundle_api_v1!(
    with_prefix_state_slot BUNDLE_QWEN38_API_V1,
    BUNDLE_QWEN38_EXTENDED_API_V1,
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

// Direct checkpoint loading: the rank images' own plan, from the headers.
runtime_bundle_abi::export_weight_plan_v1!(model_qwen38_contract::weight_plan::plan);

// The chat frontend: Qwen3.8's chat format, over this package's frontend
// component (`chop.frontend.v1`).
frontend_format::export_chat_format_v1!(model_qwen38_contract::frontend::Qwen38Format);

#[cfg(test)]
mod package_tests;
