#![forbid(unsafe_code)]
//! The typed tool grammar: a function element whose parameters close their
//! open tag with `" string="true">` for a raw string value or
//! `" string="false">` for a JSON value, and a call block read with the
//! blank line written before it.
#[allow(dead_code)]
mod common;

use std::sync::Arc;

use frontend_format::{
    FormatDescriptorV1, FrontendJsonV1, FrontendToolChoiceV1, FrontendToolGrammarV1,
    FrontendToolV1, PromptModeV1, parse_completion_v1, tool_constraint_v1,
};
use runtime_grammar::{CompiledGrammar, GrammarMatcher, TokenVocabulary, VocabularyToken};

fn descriptor() -> FormatDescriptorV1 {
    FormatDescriptorV1::from_component_bytes(common::TYPED_PARAMETERS.as_bytes()).unwrap()
}

/// One call to `name` with each `(key, json, value)` parameter, as the
/// template writes it.
fn call(name: &str, parameters: &[(&str, bool, &str)]) -> String {
    let mut text = format!("<invoke name=\"{name}\">\n");
    for (key, json, value) in parameters {
        text += &format!(
            "<parameter name=\"{key}\" string=\"{}\">{value}</parameter>\n",
            !json
        );
    }
    text + "</invoke>\n"
}

#[test]
fn the_grammar_requires_a_json_close_of_its_own() {
    let valid = descriptor();
    let mut missing = valid.clone();
    missing.parser.parameter_json_name_close.clear();
    assert!(
        missing
            .validate()
            .unwrap_err()
            .to_string()
            .contains("declared by the typed tool grammar")
    );
    let mut prefix = valid.clone();
    prefix.parser.parameter_json_name_close = "\" string=\"true\">x".into();
    assert!(
        prefix
            .validate()
            .unwrap_err()
            .to_string()
            .contains("neither begins nor is begun")
    );
    let mut other = valid;
    other.parser.tool_grammar = FrontendToolGrammarV1::XmlFunctionParametersV1;
    assert!(
        other
            .validate()
            .unwrap_err()
            .to_string()
            .contains("and only by it")
    );
}

#[test]
fn the_json_close_writes_back_and_is_absent_elsewhere() {
    let descriptor = descriptor();
    let written = descriptor.clone().into_descriptor();
    assert_eq!(
        written.format["parser"]["parameter_json_name_close"],
        "\" string=\"false\">"
    );
    let bytes = serde_json::to_vec(&written).unwrap();
    assert_eq!(
        FormatDescriptorV1::from_declaration_bytes(&bytes).unwrap(),
        descriptor
    );
    // A declaration of another grammar writes no such field, as before.
    let other = FormatDescriptorV1::from_component_bytes(common::FUNCTION_PARAMETERS.as_bytes())
        .unwrap()
        .into_descriptor();
    assert!(
        other.format["parser"]
            .get("parameter_json_name_close")
            .is_none()
    );
}

#[test]
fn each_close_types_its_value() {
    let completion = format!(
        "Checking.\n\n<calls>\n{}</calls>[EOS]",
        call(
            "f",
            &[
                ("s", false, "5"),
                ("n", true, "1.50"),
                ("l", true, "[1,  \"</parameter>\"]"),
                ("o", true, "{\"k\": 1, \"k\": 2}"),
                ("bad", true, "{not json"),
                ("s", false, "last"),
            ],
        )
    );
    let parsed = parse_completion_v1(&descriptor(), &completion, PromptModeV1::Chat).unwrap();
    // The blank line before the call block is the block's, not the text's.
    assert_eq!(parsed.content, "Checking.");
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].name, "f");
    // A raw value is the string it spells whatever it looks like, a JSON
    // value the value it spells in the declared encoding (a marker inside a
    // string literal is content, a repeated key takes its last value), JSON
    // that does not read is kept as text, and a repeated parameter takes its
    // last value.
    assert_eq!(
        parsed.tool_calls[0].arguments,
        r#"{"s": "last", "n": 1.5, "l": [1, "</parameter>"], "o": {"k": 2}, "bad": "{not json"}"#
    );
}

#[test]
fn a_block_open_without_its_blank_line_is_text() {
    let completion = "Hi<calls>\n</calls>[EOS]";
    let parsed = parse_completion_v1(&descriptor(), completion, PromptModeV1::Chat).unwrap();
    assert!(parsed.tool_calls.is_empty());
    assert_eq!(parsed.content, "Hi<calls>\n</calls>");
}

/// Whether the grammar `choice` imposes on `tools` admits `text` whole.
fn admits(tools: &[FrontendToolV1], choice: &FrontendToolChoiceV1, text: &str) -> bool {
    let grammar = tool_constraint_v1(&descriptor(), tools, choice)
        .unwrap()
        .expect("the choice constrains");
    let tokens = (0..=255_u8)
        .map(|byte| VocabularyToken {
            bytes: vec![byte],
            special: false,
        })
        .collect();
    let vocabulary = Arc::new(TokenVocabulary::new(256, tokens).unwrap());
    let compiled = Arc::new(CompiledGrammar::new(&grammar, vocabulary).unwrap());
    let mut matcher = GrammarMatcher::new(compiled);
    matcher.accept_bytes(text.as_bytes()).is_ok() && matcher.is_complete()
}

#[test]
fn a_forced_call_writes_strings_raw_and_everything_else_as_json() {
    let parameters = serde_json::json!({
        "type": "object",
        "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        "required": ["city", "days"],
        "additionalProperties": false
    });
    let tool = FrontendToolV1 {
        name: "f".into(),
        description: None,
        parameters: serde_json::from_value::<FrontendJsonV1>(parameters).unwrap(),
        parameters_present: true,
        strict: None,
    };
    let tools = [tool];
    let block = |body: String| format!("\n\n<calls>\n{body}</calls>");
    let good = block(call("f", &[("city", false, "Oslo"), ("days", true, "3")]));
    assert!(admits(&tools, &FrontendToolChoiceV1::Required, &good));
    assert!(admits(
        &tools,
        &FrontendToolChoiceV1::Named("f".into()),
        &good
    ));
    // An integer written as raw text, or a string written as JSON, is not
    // the call the schema asks for.
    for wrong in [
        block(call("f", &[("city", false, "Oslo"), ("days", false, "3")])),
        block(call(
            "f",
            &[("city", true, "\"Oslo\""), ("days", true, "3")],
        )),
    ] {
        assert!(
            !admits(&tools, &FrontendToolChoiceV1::Required, &wrong),
            "{wrong}"
        );
    }
    // What the grammar writes, the parser reads back.
    let parsed =
        parse_completion_v1(&descriptor(), &format!("{good}[EOS]"), PromptModeV1::Chat).unwrap();
    assert_eq!(
        parsed.tool_calls[0].arguments,
        r#"{"city": "Oslo", "days": 3}"#
    );
}
