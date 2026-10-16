//! A synthetic four-rank bundle for host conformance tests.
//!
//! This fixture exercises bundle installation, authenticated rank images,
//! paged KV, prefill and decode transactions, and the OpenAI frontend.
//! Each logits row is a bounded deterministic function of its synthetic
//! rank images, committed history and batch tokens. It qualifies no model
//! arithmetic or GPU route.
//!
//! The vocabulary width, prompt markers, token IDs and tokenizer policy
//! are fixed test inputs inherited from the original fixture. Its tokenizer
//! has no merges. The bundle ID and these inputs remain stable so existing
//! packages and conformance expectations keep their identities.
//!
//! `destroy` is not a rollback here. An instance with an execution still open is
//! refused with `InvalidState` and nothing is resolved, so the caller that owns
//! the transaction is the one that finishes it.
//!
//! Nothing above this crate names any of that. The host offers model-neutral
//! services through the draft V1 C ABI, this bundle asks for the resources its
//! manifest declared, and the only things that cross the boundary are bytes and
//! opaque handles. The crate links no CUDA, links no host, and holds no device
//! pointer.

#![deny(unsafe_code)]

mod chat_format;
mod configuration;
mod contract;
mod fixture;
mod frontend;
mod instance;
mod marker;
mod metadata;
mod operation;
mod synthetic;
mod tokenizer;
mod weight_plan;

pub use chat_format::Tp4FixtureFormatV1;
pub use configuration::{
    BUNDLE_CONFIGURATION_MAGIC_V1, BundleConfigurationV1, ConfiguredKvLayoutV1, ConfiguredStateV1,
    WireErrorV1,
};
pub use contract::{
    ProgramGeometryV1, TP4_FIXTURE_ARCHITECTURE_V1, TP4_FIXTURE_BUFFER_ALIGNMENT_V1,
    TP4_FIXTURE_BUNDLE_ID_V1, TP4_FIXTURE_CONTEXT_TOKENS_V1, TP4_FIXTURE_DECODE_STEP_TOKENS_V1,
    TP4_FIXTURE_EMITTING_RANK_V1, TP4_FIXTURE_FRONTEND_COMPONENT_V1,
    TP4_FIXTURE_FRONTEND_FORMAT_V1, TP4_FIXTURE_FRONTEND_PATH_V1, TP4_FIXTURE_FRONTEND_ROLE_V1,
    TP4_FIXTURE_GEOMETRY_ALIGNMENT_V1, TP4_FIXTURE_GEOMETRY_BYTES_V1,
    TP4_FIXTURE_GEOMETRY_OFFSET_V1, TP4_FIXTURE_GEOMETRY_VIEW_V1, TP4_FIXTURE_GRAPH_BUCKET_V1,
    TP4_FIXTURE_IMAGE_FILLER_OFFSET_V1, TP4_FIXTURE_IMAGE_MAGIC_OFFSET_V1,
    TP4_FIXTURE_IMAGE_MAGIC_V1, TP4_FIXTURE_IMAGE_RANK_COUNT_OFFSET_V1,
    TP4_FIXTURE_IMAGE_RANK_OFFSET_V1, TP4_FIXTURE_KV_ALIGNMENT_V1,
    TP4_FIXTURE_KV_BYTES_PER_TOKEN_V1, TP4_FIXTURE_KV_LAYOUT_ID_V1, TP4_FIXTURE_KV_PAGE_BYTES_V1,
    TP4_FIXTURE_KV_PAGE_CAPACITY_V1, TP4_FIXTURE_KV_PAGE_TOKENS_V1, TP4_FIXTURE_KV_QUOTA_TOKENS_V1,
    TP4_FIXTURE_KV_REQUEST_CAPACITY_V1, TP4_FIXTURE_KV_SEGMENTS_PER_PAGE_V1,
    TP4_FIXTURE_MARKER_COMPONENT_V1, TP4_FIXTURE_MARKER_FORMAT_V1, TP4_FIXTURE_MARKER_PATH_V1,
    TP4_FIXTURE_MARKER_ROLE_V1, TP4_FIXTURE_MARKER_SCHEMA_V1,
    TP4_FIXTURE_MAXIMUM_SEQUENCE_TOKENS_V1, TP4_FIXTURE_MAXIMUM_TOKENS_V1,
    TP4_FIXTURE_MINIMUM_DEVICE_MEMORY_BYTES_V1, TP4_FIXTURE_NAME_V1,
    TP4_FIXTURE_NATIVE_MODULE_PATH_V1, TP4_FIXTURE_PREFILL_CHUNK_TOKENS_V1,
    TP4_FIXTURE_PROGRAM_COMPONENT_V1, TP4_FIXTURE_PROGRAM_FORMAT_V1,
    TP4_FIXTURE_PROGRAM_IMAGE_ALIGNMENT_V1, TP4_FIXTURE_PROGRAM_IMAGE_BYTES_V1,
    TP4_FIXTURE_PROGRAM_ROLE_V1, TP4_FIXTURE_RANK_COUNT_V1, TP4_FIXTURE_ROW_INPUT_BYTES_V1,
    TP4_FIXTURE_ROW_INPUT_TOKENS_V1, TP4_FIXTURE_ROW_OPERATION_V1,
    TP4_FIXTURE_ROW_TOKEN_COUNT_BYTES_V1, TP4_FIXTURE_ROW_TOKEN_COUNT_ELEMENTS_V1,
    TP4_FIXTURE_ROW_VARIANT_V1, TP4_FIXTURE_STAGE_ID_V1, TP4_FIXTURE_STATE_ALIGNMENT_V1,
    TP4_FIXTURE_STATE_BYTES_V1, TP4_FIXTURE_STATE_LAYOUT_ID_V1, TP4_FIXTURE_STATE_SEED_OFFSET_V1,
    TP4_FIXTURE_STATE_SEED_VIEW_V1, TP4_FIXTURE_STREAM_KIND_V1,
    TP4_FIXTURE_TOKENIZER_CONFIGURATION_COMPONENT_V1, TP4_FIXTURE_TOKENIZER_CONFIGURATION_PATH_V1,
    TP4_FIXTURE_TOKENIZER_CONFIGURATION_ROLE_V1, TP4_FIXTURE_TOKENIZER_DEFINITION_COMPONENT_V1,
    TP4_FIXTURE_TOKENIZER_DEFINITION_PATH_V1, TP4_FIXTURE_TOKENIZER_DEFINITION_ROLE_V1,
    TP4_FIXTURE_TOKENIZER_FORMAT_V1, TP4_FIXTURE_TOPOLOGY_ID_V1, TP4_FIXTURE_VOCABULARY_BOUND_V1,
    TP4_PROMPT_TOKEN_IDS_V1, tp4_fixture_kv_layout_v1, tp4_fixture_state_layout_v1,
    tp4_fixture_topology_v1, tp4_fixture_vocabulary_v1,
};
pub use fixture::{
    emit_tp4_fixture_v1, tp4_fixture_bundle_draft_v1, tp4_fixture_row_contract_v1,
    tp4_fixture_row_key_v1,
};
pub use frontend::{
    TP4_FIXTURE_FRONTEND_SCHEMA_V1, TP4_FIXTURE_FRONTEND_VERSION_V1, TP4_FIXTURE_MODEL_ID_V1,
    TP4_FIXTURE_MODEL_OWNER_V1, tp4_fixture_begin_marker_v1, tp4_fixture_frontend_declaration_v1,
    tp4_fixture_reserved_markers_v1,
};
pub use instance::{
    cancel_v1, capture_prefix_state_v1, commit_accepted_v1, commit_v1, create_v1,
    describe_prefix_state_v1, destroy_v1, execute_v1, restore_prefix_state_v1, retire_request_v1,
    unreleased_resource_count_v1,
};
pub use marker::tp4_fixture_marker_v1;
pub use operation::{
    ObservedLendV1, RowResultV1, StagedRowV1, TP4_FIXTURE_ROW_NOTHING_STAGED_V1,
    TP4_FIXTURE_ROW_OK_V1, TP4_FIXTURE_ROW_OPERANDS_V1, TP4_FIXTURE_ROW_OPERATION_ID_V1,
    TP4_FIXTURE_ROW_TOKENS_OVER_CAPACITY_V1, TP4_FIXTURE_ROW_WRONG_OPERAND_COUNT_V1,
    TP4_FIXTURE_ROW_WRONG_OPERAND_LENGTH_V1, Tp4RowOperationV1, tp4_row_code_name_v1,
};
pub use runtime_bundle_sdk::{
    BATCH_PHASE_DECODE_V1, BATCH_PHASE_PREFILL_V1, BATCH_REQUEST_MAGIC_V1, BatchAppendV1,
    BatchItemV1, BatchRequestV1, MAXIMUM_BATCH_WIRE_ELEMENTS_V1 as MAXIMUM_WIRE_ELEMENTS_V1,
};
pub use synthetic::{
    TP4_FIXTURE_LOGIT_BOUND_V1, TP4_FIXTURE_PEAK_LOGIT_V1, TP4_FIXTURE_PEAK_TOKEN_ID_V1,
    TokenHistoryV1, fold_tokens_v1, history_from_bytes_v1, history_to_bytes_v1,
    program_image_bytes_v1, program_image_digest_v1, program_image_digests_v1, row_logits_v1,
    state_seed_history_v1, synthetic_row_v1,
};
pub use tokenizer::{
    AddedTokenV1, TP4_FIXTURE_BASE_VOCABULARY_V1, TP4_FIXTURE_BYTE_TOKEN_FIRST_ID_V1,
    TP4_FIXTURE_MERGE_COUNT_V1, TP4_FIXTURE_OUTPUT_MARKERS_V1,
    TP4_FIXTURE_TOKENIZER_MODEL_MAX_LENGTH_V1, TP4_FIXTURE_TOKENIZER_REPOSITORY_V1,
    tp4_fixture_added_token_count_v1, tp4_fixture_added_tokens_v1, tp4_fixture_base_token_v1,
    tp4_fixture_byte_alphabet_v1, tp4_fixture_filler_token_v1,
    tp4_fixture_tokenizer_configuration_v1, tp4_fixture_tokenizer_definition_v1,
    tp4_fixture_tokenizer_revision_v1, tp4_published_markers_v1,
};
pub use weight_plan::{tp4_fixture_weight_plan_v1, write_tp4_fixture_checkpoint_v1};

// The one native entry point an installed bundle exports. The macro owns every
// pointer decode and catches a panic before it can reach the host, so nothing in
// this crate writes unsafe code.
runtime_bundle_abi::export_bundle_api_v1!(
    with_prefix_cache BUNDLE_TP4_FLASH_API_V1,
    BUNDLE_TP4_FLASH_API_WITH_PREFIX_CACHE_V1,
    create_v1,
    execute_v1,
    commit_v1,
    cancel_v1,
    destroy_v1,
    retire_request_v1,
    commit_accepted_v1,
    describe_prefix_state_v1,
    capture_prefix_state_v1,
    restore_prefix_state_v1,
);

/// The speculation prefix of the exported table.
pub static BUNDLE_TP4_FLASH_API_WITH_SPECULATION_V1:
    runtime_bundle_abi::BundleApiWithSpeculationV1 = BUNDLE_TP4_FLASH_API_WITH_PREFIX_CACHE_V1.base;

// The weight plan a direct load asks for: the package plans its program image.
runtime_bundle_abi::export_weight_plan_v1!(tp4_fixture_weight_plan_v1);

// The chat frontend: the fixture's own renderer and the shared parsers, over
// this package's frontend component (`chop.frontend.v1`).
frontend_format::export_chat_format_v1!(Tp4FixtureFormatV1);
