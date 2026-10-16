//! Model-neutral bundle authoring helpers over the draft Bundle V1 ABI.
//!
//! This crate belongs on the bundle side of the dynamic-library boundary. It
//! owns wire-reading, host-error, transaction, rank-index, and diagnostic
//! mechanics, but it owns no model schedule, graph, operands, or numerical
//! policy.

mod batch;
pub mod context_parallel;
pub mod diagnostic;
mod host;
pub mod modalities;
pub mod native_note;
pub mod rank_index;
pub mod taps;
mod transaction;
pub mod warmup;

pub use batch::{
    BATCH_ITEM_FLAG_BUNDLE_DRAFTS_V3, BATCH_PHASE_DECODE_V1, BATCH_PHASE_PREFILL_V1,
    BATCH_PHASE_VERIFY_V3, BATCH_REQUEST_MAGIC_V1, BATCH_REQUEST_MAGIC_V2, BATCH_REQUEST_MAGIC_V3,
    BATCH_REQUEST_MAGIC_V5, BatchAppendCopyV2, BatchAppendV1, BatchCodecV1, BatchItemV1,
    BatchMediaEncodeV3, BatchMediaInjectV3, BatchMediaSectionV3, BatchReadErrorV1, BatchReaderV1,
    BatchRequestV1, BatchWireVersionV1, BatchWriteErrorV1, MAXIMUM_BATCH_WIRE_ELEMENTS_V1,
};
pub use host::{
    AuthenticatedComponentErrorV1, AuthenticatedComponentV1, HostServiceErrorV1,
    authenticated_component_v1, host_service_v1,
};
pub use runtime_bundle_abi as abi;
pub use transaction::{
    StreamAcquisitionFailureV1, StreamCloseFailureV1, StreamTransactionErrorV1,
    StreamTransactionEvidenceV1, StreamTransactionPhaseV1, StreamTransactionV1,
};

pub mod model_options;
