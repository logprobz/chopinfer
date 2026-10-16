#![forbid(unsafe_code)]
//! The host-read sections of `chop-frontend-openai-v2`: closed, bounded, and
//! blind to the format object.

use runtime_frontend_contract::{
    FRONTEND_FORMAT_MAX_BYTES, FrontendComponentErrorV1, FrontendDescriptorV1, FrontendNameBytesV1,
    FrontendReasoningFieldV1, FrontendToolArgumentsV1,
};
use serde_json::{Value, json};

/// A declaration with every host section and an empty format.
fn declaration() -> Value {
    json!({
        "schema": "chop-frontend-openai-v2",
        "version": 2,
        "models": [{"id": "fixture/v2", "owned_by": "fixture"}],
        "context": {"maximum_total_tokens": 512},
        "limits": {
            "maximum_prompt_bytes": 65536, "maximum_completion_bytes": 65536,
            "maximum_generated_tokens": 4096, "maximum_client_stops": 4,
            "maximum_messages": 64, "maximum_tools": 8, "maximum_tool_calls": 4,
            "maximum_tool_parameters": 16, "maximum_name_bytes": 64,
            "maximum_json_bytes": 4096, "maximum_json_depth": 8, "maximum_json_values": 256
        },
        "reasoning": {
            "disable_value": "none",
            "default_effort": "low",
            "efforts": [{"name": "low", "preamble": ""}],
            "thinking_keywords": ["enable_thinking", "thinking"],
            "end_markers": ["</t>", "<c>"]
        },
        "tokenizer": {
            "repository": "fixture/v2",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "definition": {"name": "tokenizer.json", "byte_length": 4096,
                "sha256": "abcdef1111111111111111111111111111111111111111111111111111111111"},
            "configuration": {"name": "tokenizer_config.json", "byte_length": 512,
                "sha256": "fedcba2222222222222222222222222222222222222222222222222222222222"},
            "policy": {
                "normalizer": "identity-sequence",
                "pretokenizer_byte_level": {"add_prefix_space": false, "trim_offsets": true, "use_regex": false},
                "post_processor_byte_level": {"add_prefix_space": true, "trim_offsets": false, "use_regex": true},
                "decoder_byte_level": {"add_prefix_space": true, "trim_offsets": true, "use_regex": true},
                "merge_results": {"kind": "sequential", "first_id": 300},
                "merge_records": "space-separated-string",
                "base_tokens": {"kind": "literal-prefix-then-byte-level", "byte_start_id": 44},
                "config_tokens": {"kind": "added-token", "normalized": true},
                "add_bos_token": {"kind": "required", "value": false},
                "bos_token": {"kind": "required", "value": {"kind": "contract-value"}},
                "unk_token": {"kind": "required-null"},
                "tokenizer_class": "PreTrainedTokenizerFast",
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "ignore_merges": "absent",
                "add_eos_token": false,
                "legacy": true,
                "empty_sentencepiece_kwargs": true
            },
            "base_vocabulary_size": 512,
            "merge_count": 212,
            "added_token_count": 6,
            "model_max_length": 512,
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "pad_token": "[EOS]",
            "pretokenizer_patterns": ["\\s+"]
        },
        "tools": {
            "name_bytes": {"ascii_alphanumeric": true, "extra": "_-"},
            "arguments": "json-object"
        },
        "vocabulary": {
            "size": 4096,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "stop_token_ids": [1],
            "boundary_tokens": [
                {"text": "[BOS]", "id": 0},
                {"text": "[EOS]", "id": 1},
                {"text": "</t>", "id": 2},
                {"text": "<c>", "id": 3}
            ],
            "prompt_special_tokens": ["[BOS]", "[EOS]"]
        },
        "response": {"reasoning_field": "reasoning"},
        "format": {}
    })
}

fn read(value: &Value) -> Result<FrontendDescriptorV1, FrontendComponentErrorV1> {
    FrontendDescriptorV1::from_component_bytes(&serde_json::to_vec(value).unwrap())
}

/// `declaration()` changed by `edit`, which must be refused as `expected`.
fn refused(edit: impl FnOnce(&mut Value), expected: fn(&FrontendComponentErrorV1) -> bool) {
    let mut value = declaration();
    edit(&mut value);
    let error = read(&value).expect_err("the changed declaration must be refused");
    assert!(expected(&error), "unexpected refusal: {error}");
}

fn syntax(error: &FrontendComponentErrorV1) -> bool {
    matches!(error, FrontendComponentErrorV1::Syntax(_))
}

fn invalid(error: &FrontendComponentErrorV1) -> bool {
    matches!(error, FrontendComponentErrorV1::Invalid(_))
}

#[test]
fn the_host_sections_read_and_the_format_passes_unread() {
    let descriptor = read(&declaration()).unwrap();
    assert_eq!(
        descriptor.reasoning.thinking_keywords,
        ["enable_thinking", "thinking"]
    );
    assert_eq!(descriptor.reasoning.end_markers, ["</t>", "<c>"]);
    assert_eq!(
        descriptor.tools.arguments,
        FrontendToolArgumentsV1::JsonObject
    );
    assert_eq!(
        descriptor.response.reasoning_field,
        FrontendReasoningFieldV1::Reasoning
    );
    // Whatever the format holds is the format implementation's to read.
    let mut value = declaration();
    value["format"] = json!({"template": {"anything": [1, {"x": null}]}, "other": "y"});
    let descriptor = read(&value).unwrap();
    assert_eq!(descriptor.format, value["format"]);
    // Writing the declaration back keeps every section.
    let written = serde_json::to_value(&descriptor).unwrap();
    assert_eq!(read(&written).unwrap(), descriptor);
}

#[test]
fn defaults_hold_where_the_schema_names_them() {
    let mut value = declaration();
    value["reasoning"]
        .as_object_mut()
        .unwrap()
        .remove("thinking_keywords");
    assert_eq!(
        read(&value).unwrap().reasoning.thinking_keywords,
        ["enable_thinking"]
    );
    // An empty end-marker list leaves constraints ungated.
    value["reasoning"]["end_markers"] = json!([]);
    assert!(read(&value).unwrap().reasoning.end_markers.is_empty());
}

#[test]
fn only_v2_is_read() {
    refused(
        |value| value["schema"] = json!("chop-frontend-openai-v1"),
        |error| matches!(error, FrontendComponentErrorV1::Schema { .. }),
    );
    refused(
        |value| value["version"] = json!(1),
        |error| matches!(error, FrontendComponentErrorV1::Version { .. }),
    );
}

#[test]
fn every_section_is_closed() {
    // A v1 template, an unknown section, and an omitted v2 section.
    refused(|value| value["template"] = json!({}), syntax);
    refused(|value| value["extra"] = json!(true), syntax);
    for section in ["response", "format"] {
        refused(
            |value| {
                value.as_object_mut().unwrap().remove(section);
            },
            syntax,
        );
    }
    refused(
        |value| {
            value["reasoning"]
                .as_object_mut()
                .unwrap()
                .remove("end_markers");
        },
        syntax,
    );
    refused(|value| value["response"]["extra"] = json!(1), syntax);
    refused(
        |value| value["response"]["reasoning_field"] = json!("thoughts"),
        syntax,
    );
    refused(|value| value["tools"]["arguments"] = json!("json"), syntax);
    refused(
        |value| value["tools"]["name_charset"] = json!("ascii-alphanumeric-dash-underscore"),
        syntax,
    );
    refused(
        |value| value["tools"]["name_bytes"]["extra_bytes"] = json!(""),
        syntax,
    );
    refused(
        |value| value["reasoning"]["thinking_keywords"] = json!([1]),
        syntax,
    );
}

#[test]
fn the_format_is_a_bounded_object() {
    refused(|value| value["format"] = json!([]), invalid);
    refused(|value| value["format"] = json!("chatml"), invalid);
    // The bound is the compact JSON's length: the object's braces, the key
    // and its quotes, the colon, and the string's quotes take 10 bytes.
    let fill = |value: &mut Value, length: usize| {
        value["format"] = json!({"pad": "x".repeat(length - 10)});
    };
    let mut value = declaration();
    fill(&mut value, FRONTEND_FORMAT_MAX_BYTES);
    assert!(read(&value).is_ok());
    refused(|value| fill(value, FRONTEND_FORMAT_MAX_BYTES + 1), invalid);
}

#[test]
fn thinking_keywords_are_distinct_bounded_names() {
    refused(
        |value| value["reasoning"]["thinking_keywords"] = json!([""]),
        invalid,
    );
    refused(
        |value| value["reasoning"]["thinking_keywords"] = json!(["x".repeat(129)]),
        invalid,
    );
    refused(
        |value| value["reasoning"]["thinking_keywords"] = json!(["thinking", "thinking"]),
        invalid,
    );
    refused(
        |value| {
            value["reasoning"]["thinking_keywords"] =
                json!((0..17).map(|index| format!("k{index}")).collect::<Vec<_>>());
        },
        invalid,
    );
    // The host does not interpret a keyword; the format does.
    let mut value = declaration();
    value["reasoning"]["thinking_keywords"] = json!(["ponder"]);
    assert!(read(&value).is_ok());
}

#[test]
fn end_markers_are_distinct_declared_boundary_tokens() {
    refused(
        |value| value["reasoning"]["end_markers"] = json!(["</think>"]),
        invalid,
    );
    refused(
        |value| value["reasoning"]["end_markers"] = json!(["</t>", "</t>"]),
        invalid,
    );
    refused(
        |value| {
            value["reasoning"]["end_markers"] = json!(vec!["</t>"; 129]);
        },
        invalid,
    );
}

#[test]
fn the_tool_name_byte_set_is_data() {
    let declared = read(&declaration()).unwrap().tools.name_bytes;
    assert_eq!(declared.describe(), "[A-Za-z0-9_-]");
    for byte in 0..=u8::MAX {
        let expected = byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-';
        assert_eq!(declared.permits(byte), expected, "{byte}");
    }
    let only = FrontendNameBytesV1 {
        ascii_alphanumeric: false,
        extra: ".:".into(),
    };
    assert_eq!(only.describe(), "[.:]");
    assert!(only.permits(b':') && !only.permits(b'a'));
    for extra in ["-_", "__", "_ ", "_a", "[", "]", "^", "\\", "é"] {
        refused(
            |value| value["tools"]["name_bytes"]["extra"] = json!(extra),
            invalid,
        );
    }
    refused(
        |value| {
            value["tools"]["name_bytes"] = json!({"ascii_alphanumeric": false, "extra": ""});
        },
        invalid,
    );
    let mut value = declaration();
    value["tools"]["name_bytes"] = json!({"ascii_alphanumeric": false, "extra": ".:-"});
    value["tools"]["arguments"] = json!("json-object-integral-numbers");
    let descriptor = read(&value).unwrap();
    assert_eq!(descriptor.tools.name_bytes.describe(), "[.:-]");
    assert_eq!(
        descriptor.tools.arguments,
        FrontendToolArgumentsV1::JsonObjectIntegralNumbers
    );
}

#[test]
fn the_response_field_names_its_body_field() {
    assert_eq!(FrontendReasoningFieldV1::Reasoning.name(), "reasoning");
    assert_eq!(
        FrontendReasoningFieldV1::ReasoningContent.name(),
        "reasoning_content"
    );
    let mut value = declaration();
    value["response"]["reasoning_field"] = json!("reasoning_content");
    assert_eq!(
        read(&value).unwrap().response.reasoning_field,
        FrontendReasoningFieldV1::ReasoningContent
    );
}

#[test]
fn media_render_holds_only_what_the_host_expands() {
    let modality = |render: Value| {
        json!({
            "arena": {"rows": 64, "page_rows": 64, "hidden": 8, "element": "bf16"},
            "encoders": [{"name": "image", "payload_element": "bf16", "payload_columns": 4,
                "maximum_payload_bytes": 64, "cost_per_row_milli": 1}],
            "modalities": [{"kind": "image", "part_types": ["image_url"], "maximum_items": 1,
                "maximum_bytes": 64, "maximum_tokens": 8, "encoder": 0,
                "recipe": {"tag": "image.patch-grid.v1", "parameters": {}},
                "render": render}]
        })
    };
    let mut value = declaration();
    value["multimodal"] = modality(json!({"pad": "[EOS]"}));
    assert!(read(&value).is_ok());
    for label in [
        json!({"pad": "[EOS]", "prefix": []}),
        json!({"pad": "[EOS]", "suffix": []}),
        json!({"pad": "[EOS]", "numbered_label": {"prefix": "", "suffix": ""}}),
    ] {
        refused(|value| value["multimodal"] = modality(label), syntax);
    }
    // The pad is a prompt special token.
    refused(
        |value| value["multimodal"] = modality(json!({"pad": "</t>"})),
        invalid,
    );
}

/// `declaration()` with `section` in place of the format, serialized, then
/// with `from` replaced by `to` in its text: JSON a `Value` cannot hold.
fn raw(section: &str, value: Value, from: &str, to: &str) -> Vec<u8> {
    let mut declaration = declaration();
    if section == "format" {
        declaration["format"] = value;
    } else {
        declaration["multimodal"] = value;
    }
    let text = serde_json::to_string(&declaration).unwrap();
    assert!(text.contains(from), "{from} is in {text}");
    text.replacen(from, to, 1).into_bytes()
}

#[test]
fn a_member_named_twice_is_refused_at_any_depth() {
    let duplicate = |error: &FrontendComponentErrorV1| {
        matches!(error, FrontendComponentErrorV1::Syntax(message)
            if message.contains("duplicate"))
    };
    let format = json!({"template": {"json_encoding": "x", "forms": [{"a": 1}]}});
    let multimodal = json!({
        "arena": {"rows": 64, "page_rows": 64, "hidden": 8, "element": "bf16"},
        "encoders": [{"name": "image", "payload_element": "bf16", "payload_columns": 4,
            "maximum_payload_bytes": 64, "cost_per_row_milli": 1}],
        "modalities": [{"kind": "image", "part_types": ["image_url"], "maximum_items": 1,
            "maximum_bytes": 64, "maximum_tokens": 8, "encoder": 0,
            "recipe": {"tag": "image.patch-grid.v1", "parameters": {"a": 1}},
            "render": {"pad": "[EOS]"}}]
    });
    for (section, value, from, to) in [
        // Opaque to the host, so only a check of the text can see these.
        (
            "format",
            format.clone(),
            r#""json_encoding":"x""#,
            r#""json_encoding":"y","json_encoding":"x""#,
        ),
        ("format", format.clone(), r#"{"a":1}"#, r#"{"a":0,"a":1}"#),
        (
            "format",
            format.clone(),
            r#""template":"#,
            r#""template":{},"template":"#,
        ),
        (
            "multimodal",
            multimodal.clone(),
            r#""parameters":{"a":1}"#,
            r#""parameters":{"a":0,"a":1}"#,
        ),
        // A typed section refused it already.
        (
            "multimodal",
            multimodal.clone(),
            r#""maximum_items":1"#,
            r#""maximum_items":2,"maximum_items":1"#,
        ),
        (
            "format",
            format.clone(),
            r#""schema":"#,
            r#""schema":"x","schema":"#,
        ),
    ] {
        let bytes = raw(section, value, from, to);
        let error = FrontendDescriptorV1::from_component_bytes(&bytes)
            .expect_err(&format!("{to} must be refused"));
        assert!(duplicate(&error), "{to}: {error}");
    }
    // The same member in different objects is no duplicate.
    let bytes = raw("format", format, r#"{"a":1}"#, r#"{"a":1},{"a":1}"#);
    FrontendDescriptorV1::from_component_bytes(&bytes).unwrap();
}
