//! The component that says a package is this fixture and not a model.
//!
//! A host never reads it: it is opaque bytes under a declared role, and the role
//! itself is what the native module gates on. It exists so that anything holding
//! the package - an operator, a reviewer, an archive - can read one file and
//! learn what this bundle does and, more importantly, what it does not.
//!
//! Every claim below is one this crate can stand behind. The row is synthetic,
//! no TP4 arithmetic runs, no FP8 or NVFP4 route is qualified, no TP4 operation
//! is implemented, and the tokenizer artifacts are generated stand-ins.
//!
//! It also records what the package does not reach. One instance holds one open
//! transaction, so what it exercises is C1: one sequence, prefilled in chunks
//! and decoded one token at a time. The C4, C16, and C32 mixed prefill and
//! decode behaviour the qualification matrix asks for is not proved here, and
//! the marker says so rather than leaving a reader to infer it from TP4.

use serde_json::json;

use crate::contract::{
    TP4_FIXTURE_BUNDLE_ID_V1, TP4_FIXTURE_CONTEXT_TOKENS_V1, TP4_FIXTURE_DECODE_STEP_TOKENS_V1,
    TP4_FIXTURE_EMITTING_RANK_V1, TP4_FIXTURE_MARKER_SCHEMA_V1,
    TP4_FIXTURE_PREFILL_CHUNK_TOKENS_V1, TP4_FIXTURE_PROGRAM_IMAGE_BYTES_V1,
    TP4_FIXTURE_RANK_COUNT_V1, TP4_FIXTURE_ROW_OPERATION_V1, TP4_FIXTURE_TOPOLOGY_ID_V1,
    tp4_fixture_vocabulary_v1,
};
use crate::frontend::TP4_FIXTURE_MODEL_ID_V1;
use crate::synthetic::program_image_digests_v1;

/// The marker component this fixture ships, as the bytes it stages.
#[must_use]
pub fn tp4_fixture_marker_v1() -> Vec<u8> {
    let marker = json!({
        "schema": TP4_FIXTURE_MARKER_SCHEMA_V1,
        "bundle": TP4_FIXTURE_BUNDLE_ID_V1,
        "served_model": TP4_FIXTURE_MODEL_ID_V1,
        "execution": "fixture-only-authenticated-synthetic",
        "model_arithmetic": false,
        "fp8_qualified": false,
        "nvfp4_qualified": false,
        "model_operations_implemented": 0,
        "authenticated_payload_execution": false,
        // The two artifacts are generated here and they work: the shared
        // reader builds a tokenizer from them and binds every boundary the
        // served declaration pins. The encoding is this fixture's own, not the
        // one TP4 publishes, and it carries none of TP4's 321,649 merges.
        "tokenizer_artifacts": "generated-fixture-only",
        "tokenizer_builds": true,
        "tokenizer_is_upstream": false,
        "topology": TP4_FIXTURE_TOPOLOGY_ID_V1,
        "rank_count": TP4_FIXTURE_RANK_COUNT_V1,
        // What this package exercises, and what it does not. One instance holds
        // one open transaction, so a concurrent mixed batch never reaches it.
        "concurrency_proved": "C1",
        "proves_c4_c16_c32": false,
        "concurrency_scope": "one sequence at a time: chunked prefill and one DecodeStep per step, \
                              with no C4, C16, or C32 mixed-batch evidence",
        // The KV pages are rank-local across all four ranks and every append is
        // written to all four. The committed history is not: it lives in one
        // rank's view of the declared state, and nothing replicates it.
        "state_owning_rank": TP4_FIXTURE_EMITTING_RANK_V1,
        "state_replicated_across_ranks": false,
        "prefill_tokens": TP4_FIXTURE_PREFILL_CHUNK_TOKENS_V1,
        "decode_tokens": TP4_FIXTURE_DECODE_STEP_TOKENS_V1,
        "logit_count": tp4_fixture_vocabulary_v1(),
        "context_tokens": TP4_FIXTURE_CONTEXT_TOKENS_V1,
        "private_operation": TP4_FIXTURE_ROW_OPERATION_V1,
        "rank_image_bytes": TP4_FIXTURE_PROGRAM_IMAGE_BYTES_V1,
        "rank_images": program_image_digests_v1()
            .into_iter()
            .enumerate()
            .map(|(rank, sha256)| json!({"rank": rank, "sha256": sha256}))
            .collect::<Vec<_>>(),
    });
    serde_json::to_vec_pretty(&marker).expect("a constructed marker serializes")
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use crate::synthetic::program_image_digests_v1;

    use super::tp4_fixture_marker_v1;

    /// The marker states the execution mode and claims nothing about TP4.
    #[test]
    fn the_marker_denies_every_claim_this_fixture_must_not_make() {
        let marker: Value =
            serde_json::from_slice(&tp4_fixture_marker_v1()).expect("the marker is JSON");

        assert_eq!(marker["execution"], "fixture-only-authenticated-synthetic");
        assert_eq!(marker["model_arithmetic"], false);
        assert_eq!(marker["fp8_qualified"], false);
        assert_eq!(marker["nvfp4_qualified"], false);
        assert_eq!(marker["model_operations_implemented"], 0);
        assert_eq!(marker["authenticated_payload_execution"], false);
        assert_eq!(marker["tokenizer_artifacts"], "generated-fixture-only");
        assert_eq!(marker["tokenizer_builds"], true);
        assert_eq!(marker["tokenizer_is_upstream"], false);
        assert_eq!(marker["logit_count"], 154_880);
        assert_eq!(marker["topology"], "tp4");
        assert_eq!(marker["rank_count"], 4);
        // TP4 is the topology, not the concurrency. The marker has to keep the
        // two apart, because a reader who conflates them reads C4 evidence into
        // a package that has none.
        assert_eq!(marker["concurrency_proved"], "C1");
        assert_eq!(marker["proves_c4_c16_c32"], false);
        assert_eq!(marker["state_owning_rank"], 0);
        assert_eq!(marker["state_replicated_across_ranks"], false);
        let images = marker["rank_images"]
            .as_array()
            .expect("the marker lists its rank images");
        assert_eq!(images.len(), 4);
        for (rank, digest) in program_image_digests_v1().into_iter().enumerate() {
            assert_eq!(images[rank]["rank"], rank);
            assert_eq!(
                images[rank]["sha256"], digest,
                "the marker names every image the instance authenticates"
            );
        }
        assert_eq!(marker, {
            let repeated: Value =
                serde_json::from_slice(&tp4_fixture_marker_v1()).expect("the marker is JSON");
            repeated
        });
    }
}
