//! GLM 5.3 Flash bundle (`glm5next.v1`) at TP4: the forward program, its
//! launch records over the shared engine contracts and the lowering of the
//! merged engines' stages, the create record, the instance's acquisition
//! and the ABI and weight-plan exports. Execution, graphs, commit and the
//! drafter follow; the fake-host journals pin the program.
#![deny(unsafe_code)]

pub mod buffers;
pub mod config;
pub mod context_parallel;
pub mod drafter;
pub mod encoders;
mod instance;
pub mod launch;
pub mod lower;
pub mod ops;
mod payload;
pub mod program;
pub mod projections;
mod rank_layout;
pub mod shared_overlap;
pub mod sm120;
pub mod step;
#[cfg(not(test))]
mod submit;
#[cfg(test)]
#[path = "launch_rows.rs"]
pub(crate) mod submit;
pub mod taps;
pub mod target;

#[cfg(test)]
mod comm_epilogue_tests;
#[cfg(test)]
mod context_parallel_cache_tests;
#[cfg(test)]
mod context_parallel_served_tests;
#[cfg(test)]
mod context_parallel_tests;
#[cfg(test)]
mod copy_backend_tests;
#[cfg(test)]
mod fake_host_tests;
#[cfg(test)]
mod journal_tests;
#[cfg(test)]
mod row_ops_tests;

include!(concat!(env!("OUT_DIR"), "/embedded.rs"));

/// The bytes of the embedded cubin `name`, for kernel-lab to run the
/// bundle's own kernels.
pub fn embedded_cubin(name: &str) -> Option<&'static [u8]> {
    CUBINS
        .iter()
        .find(|cubin| cubin.name == name)
        .map(|cubin| cubin.bytes)
}

/// The embedded cubin that holds entry `symbol`, by name.
pub fn cubin_of(symbol: &str) -> Option<&'static str> {
    CUBINS
        .iter()
        .find(|cubin| cubin.symbols.contains(&symbol))
        .map(|cubin| cubin.name)
}

use bundle_runtime::entry;
use instance::Instance;

// The shared entry bodies over the glm5next instance
// (`bundle_runtime::entry`): retirement frees a request's slot, a
// verify step commits the rows the host accepted, and prefix snapshots
// capture and restore a slot's state; a restore fills the host's state slot.
runtime_bundle_abi::export_bundle_api_v1!(
    with_prefix_state_slot BUNDLE_GLM5NEXT_API_V1,
    BUNDLE_GLM5NEXT_EXTENDED_API_V1,
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

// Direct checkpoint loading: the text and encoder images' plans, from the
// headers, and the encoder image's two transforms the shared set lacks.
runtime_bundle_abi::export_weight_plan_v1!(model_glm5next_contract::direct::plan);
runtime_bundle_abi::export_weight_transform_v1!(model_glm5next_contract::encoder_plan::transform);

// The chat frontend: GLM 5.3 Flash's chat format, ported from its template in
// the contract crate, with the shared parsers over this package's frontend
// component (`chop.frontend.v1`).
frontend_format::export_chat_format_v1!(model_glm5next_contract::frontend::Glm53Format);
