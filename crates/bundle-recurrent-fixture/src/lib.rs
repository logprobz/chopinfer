//! The TP1 recurrent bundle fixture, compiled as one installable bundle.
//!
//! This is the second of the two separately compiled fixtures the draft V1 C ABI
//! is frozen against, and it is built to be unlike the first one in every way
//! the ABI has to survive. It serves exactly one rank rather than four. It
//! carries persistent recurrent state instead of a KV cache: the manifest still
//! declares a KV page layout, because a V1 manifest carries one, and this bundle
//! never allocates a physical KV pool from it. It declares a small vocabulary of
//! its own, and reads that width out of the program image it ships rather than
//! from anything compiled in.
//!
//! It requires two operations, and every execution performs both. The
//! recurrent step is a bundle custom operation, which only a bundle
//! may implement and which this package carries; it writes its row into an
//! intermediate operand. Moving that row into the logits a caller receives is
//! [`runtime_bundle::GENERIC_COPY_F32_PRIMITIVE_V1`], the model-neutral
//! primitive the shared contract crate declares, which this package leaves to
//! the host tier and carries no implementation of. A host that offers an exact
//! match for it satisfies the operation generically; a host that offers
//! nothing, or a near miss, leaves it unsupported and this fixture emits no row
//! at all.
//!
//! The transaction is the recurrent state. An execution folds the batch's tokens
//! into the committed lanes and stages the result; `commit` publishes those lanes
//! as the new persistent state, and `cancel` drops them, so a cancelled step
//! leaves the model exactly where the last commit left it.
//!
//! Nothing above this crate names any of that. The host offers model-neutral
//! services through the draft V1 C ABI, this bundle asks for the resources its
//! manifest declared, and the only things that cross the boundary are bytes and
//! opaque handles. The crate links no CUDA and holds no device pointer.

#![deny(unsafe_code)]

mod configuration;
mod contract;
mod fixture;
mod instance;
mod operation;
mod recurrent;
mod weight_plan;

pub use configuration::{
    BUNDLE_CONFIGURATION_MAGIC_V1, BundleConfigurationV1, ConfiguredKvLayoutV1, ConfiguredStateV1,
    WireErrorV1,
};
pub use contract::{
    ProgramGeometryV1, RECURRENT_ARCHITECTURE_V1, RECURRENT_BUFFER_ALIGNMENT_V1,
    RECURRENT_BUNDLE_ID_V1, RECURRENT_FIXTURE_NAME_V1, RECURRENT_FRONTEND_COMPONENT_V1,
    RECURRENT_FRONTEND_DECLARATION_V1, RECURRENT_FRONTEND_FORMAT_V1, RECURRENT_FRONTEND_PATH_V1,
    RECURRENT_FRONTEND_ROLE_V1, RECURRENT_GEOMETRY_ALIGNMENT_V1, RECURRENT_GEOMETRY_BYTES_V1,
    RECURRENT_GEOMETRY_OFFSET_V1, RECURRENT_GEOMETRY_VIEW_V1, RECURRENT_GRAPH_BUCKET_V1,
    RECURRENT_KV_ALIGNMENT_V1, RECURRENT_KV_BYTES_PER_TOKEN_V1, RECURRENT_KV_FP8_LAYOUT_ID_V1,
    RECURRENT_KV_FP8_PAGE_BYTES_V1, RECURRENT_KV_LAYOUT_ID_V1, RECURRENT_KV_PAGE_BYTES_V1,
    RECURRENT_KV_PAGE_TOKENS_V1, RECURRENT_KV_SEGMENTS_PER_PAGE_V1,
    RECURRENT_MAXIMUM_SEQUENCE_TOKENS_V1, RECURRENT_MAXIMUM_TOKENS_V1,
    RECURRENT_MINIMUM_DEVICE_MEMORY_BYTES_V1, RECURRENT_NATIVE_MODULE_PATH_V1,
    RECURRENT_PROGRAM_COMPONENT_V1, RECURRENT_PROGRAM_FORMAT_V1,
    RECURRENT_PROGRAM_IMAGE_ALIGNMENT_V1, RECURRENT_PROGRAM_IMAGE_BYTES_V1,
    RECURRENT_PROGRAM_ROLE_V1, RECURRENT_RANK_COUNT_V1, RECURRENT_STAGE_ID_V1,
    RECURRENT_STATE_ALIGNMENT_V1, RECURRENT_STATE_LAYOUT_ID_V1, RECURRENT_STATE_SEED_OFFSET_V1,
    RECURRENT_STATE_SEED_VIEW_V1, RECURRENT_STEP_OPERATION_V1, RECURRENT_STEP_VARIANT_V1,
    RECURRENT_STREAM_KIND_V1, RECURRENT_TOKEN_BOUND_V1, RECURRENT_TOPOLOGY_ID_V1,
    RECURRENT_VOCABULARY_BOUND_V1, RECURRENT_VOCABULARY_V1, program_image_bytes_v1,
    recurrent_kv_fp8_layout_v1, recurrent_kv_layout_v1, recurrent_state_layout_v1,
    recurrent_topology_v1, state_seed_lanes_v1,
};
pub use fixture::{
    emit_recurrent_fixture_v1, recurrent_bundle_draft_v1, recurrent_generic_copy_contract_v1,
    recurrent_generic_copy_key_v1, recurrent_step_contract_v1, recurrent_step_key_v1,
};
pub use instance::{
    cancel_v1, capture_prefix_state_v1, commit_accepted_v1, commit_v1, create_v1,
    describe_prefix_state_v1, destroy_v1, execute_v1, restore_prefix_state_in_slot_v1,
    restore_prefix_state_v1, retire_request_v1, unreleased_resource_count_v1,
};
pub use operation::{
    ObservedLendV1, RECURRENT_STEP_NOTHING_STAGED_V1, RECURRENT_STEP_OK_V1,
    RECURRENT_STEP_OPERANDS_V1, RECURRENT_STEP_OPERATION_ID_V1,
    RECURRENT_STEP_WRONG_OPERAND_COUNT_V1, RECURRENT_STEP_WRONG_OPERAND_LENGTH_V1,
    RecurrentStepOperationV1, StagedStepV1, StepResultV1, recurrent_step_code_name_v1,
};
pub use recurrent::{
    RECURRENT_LANES_V1, RECURRENT_STATE_BYTES_V1, RecurrentLanesV1, fold_tokens_v1,
    lanes_from_bytes_v1, lanes_to_bytes_v1, logits_row_v1,
};
pub use runtime_bundle_sdk::{
    BATCH_PHASE_DECODE_V1, BATCH_PHASE_PREFILL_V1, BATCH_REQUEST_MAGIC_V1, BatchAppendV1,
    BatchItemV1, BatchRequestV1, MAXIMUM_BATCH_WIRE_ELEMENTS_V1 as MAXIMUM_WIRE_ELEMENTS_V1,
};
pub use weight_plan::{recurrent_fixture_weight_plan_v1, write_recurrent_fixture_checkpoint_v1};

// The one native entry point an installed bundle exports. The macro owns every
// pointer decode and catches a panic before it can reach the host, so nothing in
// this crate writes unsafe code.
runtime_bundle_abi::export_bundle_api_v1!(
    with_prefix_state_slot BUNDLE_RECURRENT_FIXTURE_API_V1,
    BUNDLE_RECURRENT_FIXTURE_API_WITH_PREFIX_STATE_SLOT_V1,
    create_v1,
    execute_v1,
    commit_v1,
    cancel_v1,
    destroy_v1,
    retire_request_v1,
    [commit_accepted_v1],
    describe_prefix_state_v1,
    capture_prefix_state_v1,
    restore_prefix_state_v1,
    restore_prefix_state_in_slot_v1,
);

/// Speculation prefix retained for direct ABI conformance callers.
pub static BUNDLE_RECURRENT_FIXTURE_API_WITH_SPECULATION_V1:
    runtime_bundle_abi::BundleApiWithSpeculationV1 =
    BUNDLE_RECURRENT_FIXTURE_API_WITH_PREFIX_STATE_SLOT_V1
        .base
        .base;

// The weight plan a direct load asks for: the package plans its program image.
runtime_bundle_abi::export_weight_plan_v1!(recurrent_fixture_weight_plan_v1);

// The chat frontend the package's capability set requires
// (`chop.frontend.v1`).
frontend_format::export_chat_format_v1!(RecurrentFixtureFormatV1);

/// The chat format this package exports. Its frontend component declares
/// the fixture's vocabulary, not a chat format, so the export refuses to
/// open over it: no test serves this package a chat.
pub struct RecurrentFixtureFormatV1;

impl frontend_format::ChatFormatV1 for RecurrentFixtureFormatV1 {
    fn open(_: &frontend_format::FormatDescriptorV1) -> Result<Self, String> {
        Err("the recurrent fixture renders no chat".into())
    }

    fn opened(&self) -> frontend_format::OpenedV1 {
        frontend_format::OpenedV1::default()
    }

    fn render(
        &self,
        _: frontend_format::FormatRenderRequestV1<'_>,
    ) -> Result<frontend_format::FormatRenderedV1, frontend_format::FrontendErrorV1> {
        Err(frontend_format::FrontendErrorV1::new(
            "the recurrent fixture renders no chat",
            None,
        ))
    }
}
