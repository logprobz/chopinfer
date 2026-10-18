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

#[test]
fn a_tool_choice_prefill_opens_the_declared_calls_and_never_thinks() {
    let policy = |prefill: &str, refused: bool| {
        let mut descriptor = descriptor();
        descriptor.tools.choice = Some(runtime_frontend_contract::FrontendToolChoicePolicyV1 {
            none_omits_tools: true,
            named_renders_its_tool_only: true,
            forced_prefill: prefill.into(),
            forced_refused_in_thinking: refused,
            forced_without_tools_is_auto: false,
        });
        descriptor.validate().map_err(|error| error.to_string())
    };
    policy("\n\n<calls>\n", true).unwrap();
    policy("", false).unwrap();
    assert!(
        policy("<calls>", true)
            .unwrap_err()
            .contains("must begin format.parser tool_call_open")
    );
    assert!(
        policy("\n\n<calls>\n", false)
            .unwrap_err()
            .contains("needs forced_refused_in_thinking")
    );
}

fn tool(parameters: serde_json::Value) -> FrontendToolV1 {
    FrontendToolV1 {
        name: "f".into(),
        description: None,
        parameters: serde_json::from_value::<FrontendJsonV1>(parameters).unwrap(),
        parameters_present: true,
        strict: None,
    }
}

fn block(calls: &[String]) -> String {
    format!("\n\n<calls>\n{}</calls>", calls.concat())
}

#[test]
fn a_forced_call_writes_only_names_the_parser_reads() {
    let required = tool(serde_json::json!({
        "type": "object", "properties": {"a.b": {"type": "integer"}},
        "required": ["a.b"], "additionalProperties": false
    }));
    let error = tool_constraint_v1(&descriptor(), &[required], &FrontendToolChoiceV1::Required)
        .unwrap_err();
    assert_eq!(error.parameter.as_deref(), Some("tools"));
    assert!(error.message.contains("\"a.b\""), "{}", error.message);
    // An optional name the parser cannot read is never written.
    let optional = [tool(serde_json::json!({
        "type": "object", "properties": {"a.b": {"type": "integer"}, "c": {"type": "integer"}},
        "required": ["c"], "additionalProperties": false
    }))];
    let required = FrontendToolChoiceV1::Required;
    assert!(admits(
        &optional,
        &required,
        &block(&[call("f", &[("c", true, "1")])])
    ));
    assert!(!admits(
        &optional,
        &required,
        &block(&[call("f", &[("a.b", true, "1"), ("c", true, "1")])])
    ));
}

/// A constrained number keeps its type when read back: a fraction or an
/// exponent is admitted only as a finite binary64, at any depth, while an
/// integer keeps every digit. Unconstrained, a number beyond binary64 reads
/// back as its text.
#[test]
fn a_constrained_number_keeps_its_type() {
    let tools = [tool(serde_json::json!({
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "z": {"type": "array", "items": {"type": "number"}},
            "w": {"type": "object"}
        },
        "required": ["x"],
        "additionalProperties": false
    }))];
    let admitted = |parameters: &[(&str, bool, &str)]| {
        admits(
            &tools,
            &FrontendToolChoiceV1::Required,
            &block(&[call("f", parameters)]),
        )
    };
    let long = |digits: usize| format!("1{}", "0".repeat(digits - 1));
    // Every finite spelling below 1e308 is admitted, including underflow and
    // zero under any exponent.
    for value in [
        "1.5e99".to_owned(),
        "-0.25E-7".to_owned(),
        long(400),
        format!("{}.5", long(308)),
        "1e-100".to_owned(),
        "1e300".to_owned(),
        "1e307".to_owned(),
        "-9.99e307".to_owned(),
        "1.7976931348623157e307".to_owned(),
        "1E+0000307".to_owned(),
        "0.5e308".to_owned(),
        "1e-400".to_owned(),
        "0e400".to_owned(),
        "-0.00E+999".to_owned(),
        format!("{}e291", long(17)),
        "1234567890.5e298".to_owned(),
        format!("{}e-400", long(308)),
    ] {
        assert!(admitted(&[("x", true, &value)]), "{value}");
    }
    // Magnitudes from 1e308 are refused: the overflowing ones and, by the
    // conservative boundary, the finite ones within a decade below.
    for value in [
        "1e308".to_owned(),
        "1.7976931348623157e308".to_owned(),
        "1.7976931348623159e308".to_owned(),
        "1e309".to_owned(),
        "1e400".to_owned(),
        "-1e400".to_owned(),
        "10e307".to_owned(),
        format!("{}.5", long(309)),
        format!("{}e292", long(17)),
        "1234567890.5e299".to_owned(),
        // A digit count weighed against an exponent is refused past them.
        format!("{}e-1", long(309)),
        "0.001e309".to_owned(),
        format!("{}e1", long(18)),
    ] {
        assert!(!admitted(&[("x", true, &value)]), "{value}");
    }
    assert!(admitted(&[("x", true, "1"), ("z", true, "[1.5e10]")]));
    assert!(!admitted(&[("x", true, "1"), ("z", true, "[1e400]")]));
    assert!(admitted(&[
        ("x", true, "1"),
        ("w", true, r#"{"k": [2.5e-3]}"#)
    ]));
    assert!(!admitted(&[
        ("x", true, "1"),
        ("w", true, r#"{"k": [1e400]}"#)
    ]));
    // Every admitted number reads back as a number.
    let text = block(&[call("f", &[("x", true, "1.5e99"), ("z", true, "[2.5]")])]);
    let parsed =
        parse_completion_v1(&descriptor(), &format!("{text}[EOS]"), PromptModeV1::Chat).unwrap();
    assert_eq!(
        parsed.tool_calls[0].arguments,
        r#"{"x": 1.5e+99, "z": [2.5]}"#
    );
    for value in ["1e-400", "-9.99e307", "1E+0000307", "0e400"] {
        let text = block(&[call("f", &[("x", true, value)])]);
        let parsed =
            parse_completion_v1(&descriptor(), &format!("{text}[EOS]"), PromptModeV1::Chat)
                .unwrap();
        let arguments: serde_json::Value =
            serde_json::from_str(&parsed.tool_calls[0].arguments).unwrap();
        assert!(arguments["x"].is_number(), "{value}");
    }
    // An unconstrained completion keeps a number beyond binary64 as text.
    let text = block(&[call("f", &[("x", true, "1e400")])]);
    let parsed =
        parse_completion_v1(&descriptor(), &format!("{text}[EOS]"), PromptModeV1::Chat).unwrap();
    assert_eq!(parsed.tool_calls[0].arguments, r#"{"x": "1e400"}"#);
}

#[test]
fn a_function_element_outside_a_call_block_is_text() {
    let completion = format!("Hi{}[EOS]", call("f", &[("x", false, "1")]));
    let parsed = parse_completion_v1(&descriptor(), &completion, PromptModeV1::Chat).unwrap();
    assert!(parsed.tool_calls.is_empty());
    assert!(parsed.content.starts_with("Hi<invoke name=\"f\">"));
}

#[test]
fn a_raw_value_runs_to_its_own_close() {
    let value = "a <parameter name=\"y\" string=\"true\">b</invoke>\nc";
    let completion = format!("{}[EOS]", block(&[call("f", &[("x", false, value)])]));
    let parsed = parse_completion_v1(&descriptor(), &completion, PromptModeV1::Chat).unwrap();
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(
        parsed.tool_calls[0].arguments,
        serde_json::json!({ "x": value })
            .to_string()
            .replace("\":", "\": ")
    );
}

#[test]
fn forced_calls_share_one_call_block() {
    let tools = [tool(serde_json::json!({
        "type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]
    }))];
    let one = call("f", &[("x", false, "a")]);
    let two = call("f", &[("x", false, "b")]);
    for choice in [
        FrontendToolChoiceV1::Required,
        FrontendToolChoiceV1::Named("f".into()),
    ] {
        assert!(admits(&tools, &choice, &block(std::slice::from_ref(&one))));
        assert!(admits(&tools, &choice, &block(&[one.clone(), two.clone()])));
        assert!(!admits(
            &tools,
            &choice,
            &format!(
                "{}{}",
                block(std::slice::from_ref(&one)),
                block(std::slice::from_ref(&two))
            )
        ));
    }
    let parsed = parse_completion_v1(
        &descriptor(),
        &format!("{}[EOS]", block(&[one, two])),
        PromptModeV1::Chat,
    )
    .unwrap();
    assert_eq!(parsed.tool_calls.len(), 2);
}

/// A declared namespace separator qualifies a parsed call's name, which is
/// read back whole; a separator that could begin a marker ending a name is
/// refused.
#[test]
fn a_qualified_call_reads_back_by_its_qualified_name() {
    let qualified = |separator: &str| {
        let mut value: serde_json::Value = serde_json::from_str(common::TYPED_PARAMETERS).unwrap();
        value["tools"]["namespaces"] = serde_json::json!({"separator": separator});
        FormatDescriptorV1::from_component_bytes(value.to_string().as_bytes())
    };
    let namespaced = qualified("::").unwrap();
    let text = format!("{}[EOS]", block(&[call("ns::f", &[("x", false, "v")])]));
    let parsed = parse_completion_v1(&namespaced, &text, PromptModeV1::Chat).unwrap();
    assert_eq!(parsed.tool_calls[0].name, "ns::f");
    let twice = format!("{}[EOS]", block(&[call("a::b::f", &[("x", false, "v")])]));
    assert!(parse_completion_v1(&namespaced, &twice, PromptModeV1::Chat).is_err());
    // Without the declaration a qualified name is no name.
    assert!(parse_completion_v1(&descriptor(), &text, PromptModeV1::Chat).is_err());
    let error = qualified("\"").unwrap_err().to_string();
    assert!(error.contains("ends a name"), "{error}");
}
