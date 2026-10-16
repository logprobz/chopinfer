//! Forced calls with literal values: a string enumeration or constant value
//! the tool-choice grammar writes as it is, which the served parser must
//! read back whole.

use std::sync::Arc;

use runtime_grammar::{CompiledGrammar, GrammarMatcher, TokenVocabulary, VocabularyToken};
use serde_json::{Value, json};

use crate::descriptor::FormatDescriptorV1;
use crate::format_decode::completion_calls_v1;
use crate::request::{FrontendToolChoiceV1, FrontendToolV1};
use crate::tool_grammar::tool_constraint_v1;

/// One call to `f` with the parameter `x` holding `value`, written with the
/// declaration's markers and wrappers exactly, whitespace included, as the
/// tool-choice grammar writes it.
#[must_use]
pub fn forced_call(descriptor: &FormatDescriptorV1, value: &str) -> String {
    let parser = &descriptor.parser;
    [
        parser.tool_call_open.as_str(),
        &parser.function_open,
        "f",
        &parser.function_name_close,
        &parser.parameter_open,
        "x",
        &parser.parameter_name_close,
        &parser.parameter_value_open,
        value,
        &parser.parameter_close,
        &parser.function_close,
        &parser.tool_call_close,
    ]
    .concat()
}

/// Whether the grammar `tool_choice: required` imposes on `tools` admits
/// `text` whole, over a vocabulary of one token per byte; `None` when the
/// tools are refused.
#[must_use]
pub fn required_admits(
    descriptor: &FormatDescriptorV1,
    tools: &[FrontendToolV1],
    text: &str,
) -> Option<bool> {
    let grammar = tool_constraint_v1(descriptor, tools, &FrontendToolChoiceV1::Required)
        .ok()?
        .expect("a required tool choice constrains");
    let tokens = (0..=255_u8)
        .map(|byte| VocabularyToken {
            bytes: vec![byte],
            special: false,
        })
        .collect();
    let vocabulary = Arc::new(TokenVocabulary::new(256, tokens).expect("a byte vocabulary"));
    let compiled =
        Arc::new(CompiledGrammar::new(&grammar, vocabulary).expect("the grammar compiles"));
    let mut matcher = GrammarMatcher::new(compiled);
    Some(matcher.accept_bytes(text.as_bytes()).is_ok() && matcher.is_complete())
}

fn tool(parameters: Value) -> FrontendToolV1 {
    FrontendToolV1 {
        name: "f".into(),
        description: None,
        parameters: serde_json::from_value(parameters).expect("a schema"),
        parameters_present: true,
        strict: None,
    }
}

/// Checks a declaration's forced literals: under `tool_choice: required`,
/// an enumeration, a constant and an additional-property enumeration value
/// that spells no marker builds a grammar that writes the call, which the
/// served parser reads back as that value, under `string`, the `str` alias
/// and an unknown type name; a value that spells the parameter close is
/// refused.
///
/// # Panics
///
/// When any of these does not hold.
pub fn assert_forced_literals_read_back(descriptor: &FormatDescriptorV1) {
    let parser = &descriptor.parser;
    let served = Arc::new(descriptor.clone());
    for kind in ["string", "str", "unknown-type"] {
        for value in ["plain", "\"quoted\"", "two words", ""] {
            for parameters in [
                json!({"type": "object", "properties": {"x": {"type": kind, "enum": [value]}},
                       "required": ["x"], "additionalProperties": false}),
                json!({"type": "object", "properties": {"x": {"type": kind, "const": value}},
                       "required": ["x"]}),
                json!({"type": "object", "required": ["x"],
                       "additionalProperties": {"type": kind, "enum": [value]}}),
            ] {
                let tools = [tool(parameters)];
                let text = forced_call(descriptor, value);
                let label = format!("{kind} {value:?} {}", tools[0].parameters.to_json_value());
                assert_eq!(
                    required_admits(descriptor, &tools, &text),
                    Some(true),
                    "{label}: the grammar refuses or does not write the call"
                );
                let calls = completion_calls_v1(&served, &text, &tools)
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
                let read: Vec<(String, Value)> = calls
                    .into_iter()
                    .map(|call| (call.name, serde_json::from_str(&call.arguments).unwrap()))
                    .collect();
                assert_eq!(read, [("f".to_owned(), json!({"x": value}))], "{label}");
            }
        }
        let value = format!("a{}b", parser.parameter_close.trim());
        let tools = [tool(
            json!({"type": "object", "properties": {"x": {"type": kind, "enum": [value]}},
                   "required": ["x"]}),
        )];
        assert_eq!(
            required_admits(descriptor, &tools, &forced_call(descriptor, &value)),
            None,
            "{kind} {value:?}: a value that spells the parameter close is offered"
        );
    }
}
