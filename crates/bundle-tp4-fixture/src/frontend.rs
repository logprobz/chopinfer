//! The OpenAI frontend declaration this fixture serves.
//!
//! This is the one part of the package an unchanged transport reads and answers
//! out of, so it is a real declaration rather than a stand-in: the schema, the
//! limits, the reasoning controls, the parser section, the tokenizer identity,
//! and the vocabulary are all present and all validated by the shared parser.
//!
//! What makes it TP4-specific is the part that is not invented. The parser
//! section reads the published TP4 reasoning and tool-call markers, which the
//! fixture's renderer ([`crate::Tp4FixtureFormatV1`]) writes with the other
//! published prompt markers, the reserved set is the 16 published markers
//! plus the published stop token, the boundary
//! identifiers are the ones `bundle-tp4-fixture` pins, the vocabulary width is
//! TP4's configured width, and the pretokenizer pattern is TP4's own. What is
//! this fixture's own is the served identity, the context bound, the limits, and
//! the tokenizer artifacts the declaration pins, and each of those names itself
//! a fixture.
//!
//! The tokenizer section pins artifacts that work: the component policy, the
//! base vocabulary, the merge count, and the added-token count all describe the
//! bytes [`crate::tokenizer`] generates, so the shared reader builds a real
//! tokenizer from them and binds every boundary identifier this declaration
//! pins. The policy it declares is the encoding this fixture chose for its own
//! artifacts, not the encoding TP4 publishes.

use crate::metadata::{
    TP4_PRETOKENIZER_PATTERN, TP4_TEXT_PROMPT_SPECIAL_TOKENS, tokenizer_metadata,
};
use serde_json::{Value, json};

use crate::contract::{TP4_FIXTURE_CONTEXT_TOKENS_V1, tp4_fixture_vocabulary_v1};
use crate::tokenizer::{
    TP4_FIXTURE_BASE_VOCABULARY_V1, TP4_FIXTURE_BYTE_TOKEN_FIRST_ID_V1, TP4_FIXTURE_MERGE_COUNT_V1,
    TP4_FIXTURE_TOKENIZER_CLASS_V1, TP4_FIXTURE_TOKENIZER_MODEL_MAX_LENGTH_V1,
    TP4_FIXTURE_TOKENIZER_REPOSITORY_V1, tp4_fixture_added_token_count_v1,
    tp4_fixture_tokenizer_configuration_v1, tp4_fixture_tokenizer_definition_v1,
    tp4_fixture_tokenizer_revision_v1, tp4_published_markers_v1,
};

/// The schema the shared frontend parser reads, and the only one it reads.
pub const TP4_FIXTURE_FRONTEND_SCHEMA_V1: &str = "chop-frontend-openai-v2";

/// The version of that schema this declaration is written against.
pub const TP4_FIXTURE_FRONTEND_VERSION_V1: u32 = 2;

/// The synthetic identity this fixture answers to.
pub const TP4_FIXTURE_MODEL_ID_V1: &str = "fixture/tp4";

/// Who the served identity is owned by.
pub const TP4_FIXTURE_MODEL_OWNER_V1: &str = "bundle-fixture";

/// The one marker the prompt opens a conversation with.
///
/// TP4's published beginning-of-sequence pair, concatenated. The declaration
/// reserves both halves, so a client cannot write either of them itself.
#[must_use]
pub fn tp4_fixture_begin_marker_v1() -> String {
    let metadata = tokenizer_metadata();
    let prefix = metadata.bos_prefix();
    format!("{}{}", prefix[0], prefix[1])
}

/// Every marker a client is forbidden from writing into a prompt.
#[must_use]
pub fn tp4_fixture_reserved_markers_v1() -> Vec<String> {
    let mut reserved = TP4_TEXT_PROMPT_SPECIAL_TOKENS
        .iter()
        .map(|marker| (*marker).to_owned())
        .collect::<Vec<_>>();
    reserved.push(tokenizer_metadata().eos_token().to_owned());
    reserved
}

/// The declaration this package ships, as the bytes the transport reads.
#[must_use]
pub fn tp4_fixture_frontend_declaration_v1() -> Vec<u8> {
    let metadata = tokenizer_metadata();
    let marker = |index: usize| Value::from(TP4_TEXT_PROMPT_SPECIAL_TOKENS[index]);
    let boundaries = tp4_published_markers_v1()
        .into_iter()
        .map(|(id, token)| json!({"text": token, "id": id}))
        .collect::<Vec<_>>();
    let prompt_special_tokens = TP4_TEXT_PROMPT_SPECIAL_TOKENS
        .iter()
        .map(|token| Value::from(*token))
        .collect::<Vec<_>>();
    // What the shared parsers read: the key-value tool grammar over the
    // published markers, apart so the declaration below nests within the
    // JSON macro's recursion limit. The fixture's renderer writes the prompt
    // markers in code ([`crate::Tp4FixtureFormatV1`]).
    let parser = json!({
        "tool_grammar": "xml-key-value-arguments-v1",
        "json_encoding": "spaced-json-v1",
        "reasoning_open": marker(6),
        "reasoning_close": marker(7),
        "tool_call_open": marker(8),
        "tool_call_close": marker(9),
        "parameter_open": marker(12),
        "parameter_name_close": marker(13),
        "parameter_value_open": marker(14),
        "parameter_close": marker(15),
        "reserved_markers": tp4_fixture_reserved_markers_v1(),
        "turn_open": [marker(4), marker(5), marker(2), marker(3)],
        "assistant_header": marker(4)
    });
    let declaration = json!({
        "schema": TP4_FIXTURE_FRONTEND_SCHEMA_V1,
        "version": TP4_FIXTURE_FRONTEND_VERSION_V1,
        "models": [{"id": TP4_FIXTURE_MODEL_ID_V1, "owned_by": TP4_FIXTURE_MODEL_OWNER_V1}],
        "context": {"maximum_total_tokens": TP4_FIXTURE_CONTEXT_TOKENS_V1},
        "limits": {
            "maximum_prompt_bytes": 65_536,
            "maximum_completion_bytes": 65_536,
            "maximum_generated_tokens": 64,
            "maximum_client_stops": 4,
            "maximum_messages": 64,
            "maximum_tools": 8,
            "maximum_tool_calls": 8,
            "maximum_tool_parameters": 16,
            "maximum_name_bytes": 64,
            "maximum_json_bytes": 65_536,
            "maximum_json_depth": 16,
            "maximum_json_values": 4_096
        },
        "reasoning": {
            "disable_value": "off",
            "default_effort": "standard",
            "efforts": [
                {"name": "standard", "preamble": ""},
                {"name": "deep", "preamble": "Reason step by step before answering.\n"}
            ],
            "end_markers": [marker(7), marker(8)]
        },
        "tokenizer": {
            "repository": TP4_FIXTURE_TOKENIZER_REPOSITORY_V1,
            "revision": tp4_fixture_tokenizer_revision_v1(),
            "definition": file_identity_v1(
                "tokenizer.json",
                &tp4_fixture_tokenizer_definition_v1(),
            ),
            "configuration": file_identity_v1(
                "tokenizer_config.json",
                &tp4_fixture_tokenizer_configuration_v1(),
            ),
            // The encoding this fixture generated its own artifacts under. It
            // is not TP4's published encoding, and the merge count says so:
            // TP4 publishes 321,649 merges and these artifacts carry none.
            "policy": {
                "normalizer": "identity-sequence",
                "pretokenizer_byte_level": {"add_prefix_space": false, "trim_offsets": true, "use_regex": false},
                "post_processor_byte_level": {"add_prefix_space": true, "trim_offsets": false, "use_regex": true},
                "decoder_byte_level": {"add_prefix_space": true, "trim_offsets": true, "use_regex": true},
                "merge_results": {"kind": "sequential", "first_id": TP4_FIXTURE_BASE_VOCABULARY_V1},
                "merge_records": "space-separated-string",
                "base_tokens": {
                    "kind": "literal-prefix-then-byte-level",
                    "byte_start_id": TP4_FIXTURE_BYTE_TOKEN_FIRST_ID_V1
                },
                "config_tokens": {"kind": "added-token", "normalized": true},
                "add_bos_token": {"kind": "required", "value": false},
                "bos_token": {"kind": "required", "value": {"kind": "contract-value"}},
                "unk_token": {"kind": "required-null"},
                "tokenizer_class": TP4_FIXTURE_TOKENIZER_CLASS_V1,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "ignore_merges": "absent",
                "add_eos_token": false,
                "legacy": true,
                "empty_sentencepiece_kwargs": true
            },
            "base_vocabulary_size": TP4_FIXTURE_BASE_VOCABULARY_V1,
            "merge_count": TP4_FIXTURE_MERGE_COUNT_V1,
            "added_token_count": tp4_fixture_added_token_count_v1(),
            "model_max_length": TP4_FIXTURE_TOKENIZER_MODEL_MAX_LENGTH_V1,
            "bos_token": metadata.bos_prefix()[0],
            "eos_token": metadata.eos_token(),
            "pad_token": metadata.pad_token(),
            "pretokenizer_patterns": [TP4_PRETOKENIZER_PATTERN]
        },
        "tools": {
            "name_bytes": {"ascii_alphanumeric": true, "extra": "_-"},
            "arguments": "json-object-integral-numbers"
        },
        "vocabulary": {
            "size": tp4_fixture_vocabulary_v1(),
            "bos_token_id": metadata.bos_prefix_ids()[0],
            "eos_token_id": metadata.eos_token_id(),
            "stop_token_ids": [metadata.eos_token_id()],
            "boundary_tokens": boundaries,
            "prompt_special_tokens": prompt_special_tokens
        },
        "multimodal": {
            "arena": {"rows": 64, "page_rows": 4, "hidden": 1, "element": "bf16"},
            "encoders": [{"name": "fixture-copy-image", "payload_element": "bf16", "payload_columns": 3, "maximum_payload_bytes": 1048576, "cost_per_row_milli": 1000}],
            "modalities": [{
                "kind": "image", "part_types": ["image_url"], "maximum_items": 1,
                "maximum_bytes": 1048576, "maximum_tokens": 1,
                "recipe": {"tag": "image.patch-grid.v1", "parameters": {
                    "factor": 1, "minimum_pixels": 1, "maximum_pixels": 1,
                    "maximum_aspect_ratio": 1.0, "patch": 1, "temporal_patch": 1,
                    "merge": 1, "value_scale": 1.0,
                    "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]
                }},
                "encoder": 0, "render": {"pad": "<|observation|>"}
            }]
        },
        "response": {"reasoning_field": "reasoning_content"},
        "format": {"parser": parser}
    });
    serde_json::to_vec_pretty(&declaration).expect("a constructed declaration serializes")
}

/// One pinned artifact identity, exactly as the declaration carries it.
fn file_identity_v1(name: &str, bytes: &[u8]) -> Value {
    json!({
        "name": name,
        "byte_length": bytes.len(),
        "sha256": runtime_weights::sha256_bytes(bytes),
    })
}

#[cfg(test)]
mod tests {
    use crate::metadata::{TP4_TEXT_PROMPT_SPECIAL_TOKENS, tokenizer_metadata};
    use serde_json::Value;

    use crate::contract::{TP4_FIXTURE_CONTEXT_TOKENS_V1, tp4_fixture_vocabulary_v1};
    use crate::tokenizer::{
        TP4_FIXTURE_TOKENIZER_MODEL_MAX_LENGTH_V1, tp4_fixture_tokenizer_configuration_v1,
        tp4_fixture_tokenizer_definition_v1,
    };

    use super::{
        TP4_FIXTURE_MODEL_ID_V1, tp4_fixture_frontend_declaration_v1,
        tp4_fixture_reserved_markers_v1,
    };

    fn declaration() -> Value {
        serde_json::from_slice(&tp4_fixture_frontend_declaration_v1())
            .expect("the emitted declaration is JSON")
    }

    /// The served identity is a fixture, and it says so in the one field a
    /// client reads it from.
    #[test]
    fn the_served_identity_is_the_fixture() {
        let declaration = declaration();

        let models = declaration["models"].as_array().expect("models are listed");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["id"], TP4_FIXTURE_MODEL_ID_V1);
        assert!(
            TP4_FIXTURE_MODEL_ID_V1.contains("fixture"),
            "the served identity has to be recognisable as a fixture"
        );
    }

    /// Every published marker is reserved, and the parser reads the published
    /// reasoning and tool-call markers.
    #[test]
    fn the_parser_reads_the_published_fixture_markers() {
        let declaration = declaration();
        let parser = &declaration["format"]["parser"];
        let metadata = tokenizer_metadata();

        assert_eq!(parser["tool_grammar"], "xml-key-value-arguments-v1");
        for (field, index) in [
            ("reasoning_open", 6_usize),
            ("reasoning_close", 7),
            ("tool_call_open", 8),
            ("tool_call_close", 9),
            ("parameter_open", 12),
            ("parameter_name_close", 13),
            ("parameter_value_open", 14),
            ("parameter_close", 15),
            ("assistant_header", 4),
        ] {
            assert_eq!(
                parser[field], TP4_TEXT_PROMPT_SPECIAL_TOKENS[index],
                "{field} must be the published marker at {index}"
            );
        }
        assert!(declaration["format"].get("template").is_none());

        let reserved = tp4_fixture_reserved_markers_v1();
        assert_eq!(reserved.len(), TP4_TEXT_PROMPT_SPECIAL_TOKENS.len() + 1);
        assert!(reserved.iter().any(|entry| entry == metadata.eos_token()));
        assert_eq!(
            parser["reserved_markers"]
                .as_array()
                .expect("the reserved set is a list")
                .len(),
            reserved.len()
        );
    }

    /// The vocabulary the declaration pins is TP4's width and TP4's identifiers.
    #[test]
    fn the_declared_vocabulary_is_the_published_width_and_identifiers() {
        let declaration = declaration();
        let vocabulary = &declaration["vocabulary"];
        let metadata = tokenizer_metadata();

        assert_eq!(vocabulary["size"], tp4_fixture_vocabulary_v1());
        assert_eq!(vocabulary["size"], 154_880);
        assert_eq!(vocabulary["eos_token_id"], metadata.eos_token_id());
        assert_eq!(vocabulary["bos_token_id"], metadata.bos_prefix_ids()[0]);
        assert_eq!(
            vocabulary["boundary_tokens"]
                .as_array()
                .expect("boundaries are listed")
                .len(),
            TP4_TEXT_PROMPT_SPECIAL_TOKENS.len() + 1
        );
        assert_eq!(
            vocabulary["prompt_special_tokens"]
                .as_array()
                .expect("prompt markers are listed")
                .len(),
            TP4_TEXT_PROMPT_SPECIAL_TOKENS.len()
        );
    }

    /// The tokenizer section pins exactly the bytes this package ships, and the
    /// context it offers is one the pinned reader could carry.
    #[test]
    fn the_tokenizer_section_pins_the_generated_bytes_this_package_ships() {
        let declaration = declaration();
        let tokenizer = &declaration["tokenizer"];

        for (field, bytes) in [
            ("definition", tp4_fixture_tokenizer_definition_v1()),
            ("configuration", tp4_fixture_tokenizer_configuration_v1()),
        ] {
            assert_eq!(tokenizer[field]["byte_length"], bytes.len());
            assert_eq!(
                tokenizer[field]["sha256"],
                runtime_weights::sha256_bytes(&bytes),
                "the declaration must pin the {field} this package ships"
            );
        }
        assert_eq!(
            tokenizer["model_max_length"],
            TP4_FIXTURE_TOKENIZER_MODEL_MAX_LENGTH_V1
        );
        assert_eq!(
            declaration["context"]["maximum_total_tokens"],
            TP4_FIXTURE_CONTEXT_TOKENS_V1
        );
        // A deployment may serve less context than its tokenizer reads, never
        // more, and the two numbers are read back out of the document here.
        let context = declaration["context"]["maximum_total_tokens"]
            .as_u64()
            .expect("the served context is a number");
        let readable = tokenizer["model_max_length"]
            .as_u64()
            .expect("the pinned reader's bound is a number");
        assert!(context <= readable, "{context} exceeds {readable}");
    }
}
