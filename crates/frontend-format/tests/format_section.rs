#![forbid(unsafe_code)]
//! The `format` object as this library reads it: closed, exactly the parser
//! section, and tied to the host sections it shares a declaration with.
#[allow(dead_code)]
mod common;

use frontend_format::FormatDescriptorV1;
use runtime_frontend_contract::{FrontendComponentErrorV1, FrontendDescriptorV1};
use serde_json::{Value, json};

fn key_value() -> Value {
    serde_json::from_str(common::KEY_VALUE).unwrap()
}

/// `key_value()` changed by `edit`: accepted by the host, and refused by the
/// format as `expected`.
fn refused(edit: impl FnOnce(&mut Value), expected: fn(&FrontendComponentErrorV1) -> bool) {
    let mut value = key_value();
    edit(&mut value);
    let bytes = serde_json::to_vec(&value).unwrap();
    FrontendDescriptorV1::from_component_bytes(&bytes).expect("the host sections read");
    let error = FormatDescriptorV1::from_component_bytes(&bytes)
        .expect_err("the changed format must be refused");
    assert!(expected(&error), "unexpected refusal: {error}");
}

fn syntax(error: &FrontendComponentErrorV1) -> bool {
    matches!(error, FrontendComponentErrorV1::Syntax(_))
}

fn invalid(error: &FrontendComponentErrorV1) -> bool {
    matches!(error, FrontendComponentErrorV1::Invalid(_))
}

#[test]
fn the_format_reads_into_the_parser_section() {
    let declared = FormatDescriptorV1::from_component_bytes(common::KEY_VALUE.as_bytes()).unwrap();
    let parser = &declared.parser;
    assert_eq!(parser.reasoning_close, "</t>");
    assert_eq!(parser.reserved_markers.len(), 6);
    // The host sections it dereferences to cannot pass for the declaration.
    assert!(declared.format.is_null());
    assert!(declared.validate().is_ok());
    // Writing the format back gives a declaration that reads the same: the
    // parser section with its defaults written out.
    let written = declared.clone().into_descriptor();
    assert_eq!(FormatDescriptorV1::new(written).unwrap(), declared);
    let host = FrontendDescriptorV1::from_component_bytes(common::KEY_VALUE.as_bytes()).unwrap();
    assert_eq!(FormatDescriptorV1::new(host).unwrap(), declared);
}

#[test]
fn the_format_is_closed() {
    refused(|value| value["format"]["extra"] = json!({}), syntax);
    refused(
        |value| value["format"]["parser"]["extra"] = json!(1),
        syntax,
    );
    refused(
        |value| {
            value["format"]["parser"]
                .as_object_mut()
                .unwrap()
                .remove("json_encoding");
        },
        syntax,
    );
    // A template or media labels are a renderer's, not the declaration's.
    refused(
        |value| value["format"]["template"] = json!({"json_encoding": "spaced-json-v1"}),
        syntax,
    );
    refused(
        |value| value["format"]["media"] = json!({"image": {"prefix": [], "suffix": []}}),
        syntax,
    );
    refused(|value| value["format"] = json!({}), syntax);
}

#[test]
fn reasoning_end_markers_are_the_parsers_own() {
    // The tool-call open is no boundary token, so the reasoning close alone
    // ends reasoning.
    refused(
        |value| value["reasoning"]["end_markers"] = json!([]),
        invalid,
    );
    refused(
        |value| value["reasoning"]["end_markers"] = json!(["</t>", "<t>"]),
        invalid,
    );
}

#[test]
fn only_the_keywords_the_format_reads_are_declared() {
    refused(
        |value| value["reasoning"]["thinking_keywords"] = json!(["think"]),
        invalid,
    );
    let mut value = key_value();
    value["reasoning"]["thinking_keywords"] = json!(["thinking", "enable_thinking"]);
    assert!(FormatDescriptorV1::from_component_bytes(&serde_json::to_vec(&value).unwrap()).is_ok());
}

#[test]
fn a_stop_token_needs_its_text() {
    refused(
        |value| value["vocabulary"]["stop_token_ids"] = json!([1, 7]),
        invalid,
    );
}

#[test]
fn reserved_markers_are_checked() {
    // Empty, declared twice, and too many.
    for markers in [json!([""]), json!(["[U]", "[U]"]), json!(vec!["x"; 129])] {
        refused(
            |value| value["format"]["parser"]["reserved_markers"] = markers,
            invalid,
        );
    }
}

#[test]
fn a_descriptor_changed_in_place_is_checked_again() {
    let mut declared =
        FormatDescriptorV1::from_component_bytes(common::KEY_VALUE.as_bytes()).unwrap();
    declared.parser.reasoning_close = String::new();
    assert!(invalid(&declared.validate().unwrap_err()));
    declared.parser = FormatDescriptorV1::from_component_bytes(common::KEY_VALUE.as_bytes())
        .unwrap()
        .parser;
    declared.validate().unwrap();
}

#[test]
fn a_format_member_named_twice_is_refused() {
    // The fixture's text with one member written twice, the first copy
    // changed: a `Value` keeps only the last copy, which reads.
    for (from, to) in [
        (
            r#""json_encoding": "python-json-v1""#,
            r#""json_encoding": "not-an-encoding", "json_encoding": "python-json-v1""#,
        ),
        (
            r#""reasoning_close": "</t>""#,
            r#""reasoning_close": "</x>", "reasoning_close": "</t>""#,
        ),
        (r#""parser": {"#, r#""parser": {}, "parser": {"#),
    ] {
        assert!(common::KEY_VALUE.contains(from), "{from}");
        let text = common::KEY_VALUE.replacen(from, to, 1);
        let error = FormatDescriptorV1::from_component_bytes(text.as_bytes())
            .expect_err("a duplicate member must be refused");
        assert!(
            matches!(&error, FrontendComponentErrorV1::Syntax(message)
                if message.contains("duplicate")),
            "{to}: {error}"
        );
    }
}
