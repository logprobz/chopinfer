#![forbid(unsafe_code)]
//! A parameter's value reads back as the template wrote it: the shared
//! parsers remove only the framing the declared markers carry, and they
//! resolve a value's schema by one policy, additional properties included.
//! Every completion is read whole and at every byte split, with tool calls
//! delivered whole and streamed.

mod common;

use std::sync::Arc;

use common::{FUNCTION_PARAMETERS, KEY_VALUE};
use frontend_format::{
    DeclaredParserV1, FormatDescriptorV1, FormatEventV1, FormatInputV1, FormatParserOptionsV1,
    FrontendToolChoiceV1, FrontendToolV1, PromptModeV1, tool_constraint_v1,
};
use runtime_grammar::{CompiledGrammar, GrammarMatcher, TokenVocabulary, VocabularyToken};
use serde_json::{Value, json};

/// The function-parameter fixture, whose markers frame each value in one
/// newline (`>\n` and `\n</parameter>\n`).
fn framed() -> FormatDescriptorV1 {
    FormatDescriptorV1::from_component_bytes(FUNCTION_PARAMETERS.as_bytes()).unwrap()
}

/// The same fixture with markers that frame nothing (`>` and
/// `</parameter>`), as a template that writes a value right after its name.
fn unframed() -> FormatDescriptorV1 {
    let mut descriptor = framed();
    let parser = &mut descriptor.parser;
    parser.tool_call_open = "<tool_call>".into();
    parser.function_name_close = ">".into();
    parser.function_close = "</function>".into();
    parser.parameter_name_close = ">".into();
    parser.parameter_close = "</parameter>".into();
    descriptor.validate().unwrap();
    descriptor
}

fn key_value() -> FormatDescriptorV1 {
    FormatDescriptorV1::from_component_bytes(KEY_VALUE.as_bytes()).unwrap()
}

fn tool(parameters: Value) -> FrontendToolV1 {
    FrontendToolV1 {
        name: "f".into(),
        description: None,
        parameters: serde_json::from_value(parameters).unwrap(),
        parameters_present: true,
        strict: None,
    }
}

/// One call to `f` with the parameter `x` holding `value`, as each
/// fixture's template writes it.
fn call(descriptor: &FormatDescriptorV1, value: &str) -> String {
    let parser = &descriptor.parser;
    if parser.function_open.is_empty() {
        return format!(
            "{}f{}x{}{}{value}{}{}",
            parser.tool_call_open,
            parser.parameter_open,
            parser.parameter_name_close,
            parser.parameter_value_open,
            parser.parameter_close,
            parser.tool_call_close
        );
    }
    format!(
        "{}{}f{}{}x{}{value}{}{}{}",
        parser.tool_call_open,
        parser.function_open,
        parser.function_name_close,
        parser.parameter_open,
        parser.parameter_name_close,
        parser.parameter_close,
        parser.function_close,
        parser.tool_call_close
    )
}

/// Each call's name and arguments.
type CallsV1 = Vec<(String, String)>;

/// The calls one parser delivers for `text` and the stop, pushed in two
/// parts at the byte `split`, which may fall inside a character, or in one
/// part when it is `None`: whole, and when streamed, also as joined deltas.
fn calls(
    descriptor: &FormatDescriptorV1,
    tools: &[FrontendToolV1],
    text: &str,
    split: Option<usize>,
    stream: bool,
) -> (CallsV1, CallsV1) {
    let mut parser = DeclaredParserV1::new(
        Arc::new(descriptor.clone()),
        FormatParserOptionsV1 {
            mode: PromptModeV1::Chat,
            reasoning_open: false,
            extract_tools: true,
            stream_tool_calls: stream,
            raw: false,
            ignore_eos: false,
            engine_tokens: false,
            tool_schemas: tools.to_vec(),
        },
    );
    let completion = format!("{text}[EOS]");
    let bytes = completion.as_bytes();
    let mut events = Vec::new();
    if let Some(split) = split {
        events = parser
            .push(FormatInputV1 {
                completion: &bytes[..split],
                held: false,
                last: false,
                stop_floor: Some(0),
            })
            .unwrap_or_else(|error| panic!("{text:?} split {split}: {error}"));
    }
    events.extend(
        parser
            .push(FormatInputV1 {
                completion: bytes,
                held: false,
                last: true,
                stop_floor: Some(0),
            })
            .unwrap_or_else(|error| panic!("{text:?} split {split:?}: {error}")),
    );
    let (mut whole, mut streamed) = (CallsV1::new(), CallsV1::new());
    for event in events {
        match event {
            FormatEventV1::ToolCalls { calls, .. } => {
                whole.extend(calls.into_iter().map(|call| (call.name, call.arguments)));
            }
            FormatEventV1::ToolCallDeltas(deltas) => {
                for delta in deltas {
                    match delta.name {
                        Some(name) => streamed.push((name, delta.arguments)),
                        None => streamed[delta.index].1.push_str(&delta.arguments),
                    }
                }
            }
            _ => {}
        }
    }
    (whole, streamed)
}

/// `text` reads as one call to `f` with `arguments` delivered in one part
/// and split at every byte, an empty first part included, whole and
/// streamed.
fn assert_reads(
    descriptor: &FormatDescriptorV1,
    tools: &[FrontendToolV1],
    text: &str,
    arguments: &Value,
) {
    let expected = vec![("f".to_owned(), arguments.to_string())];
    let normalize = |calls: CallsV1| -> CallsV1 {
        calls
            .into_iter()
            .map(|(name, arguments)| {
                let value: Value = serde_json::from_str(&arguments).unwrap();
                (name, value.to_string())
            })
            .collect()
    };
    for split in std::iter::once(None).chain((0..=text.len()).map(Some)) {
        for stream in [false, true] {
            let (whole, streamed) = calls(descriptor, tools, text, split, stream);
            assert_eq!(
                normalize(whole),
                expected,
                "{text:?} split {split:?} stream {stream}"
            );
            if stream {
                assert_eq!(normalize(streamed), expected, "{text:?} split {split:?}");
            }
        }
    }
}

/// Values whose first or last bytes are newlines, or that are empty.
const NEWLINE_VALUES: [&str; 9] = [
    "",
    "\n",
    "\r\n",
    "\n\n",
    "\n\ntext\n\n",
    "\ntext",
    "text\n",
    "a\r\nb\r\n",
    " padded ",
];

#[test]
fn an_unframed_string_value_keeps_every_byte() {
    let descriptor = unframed();
    let tools = [tool(
        json!({"type": "object", "properties": {"x": {"type": "string"}}}),
    )];
    for value in NEWLINE_VALUES.into_iter().chain(["e\u{301}"]) {
        assert_reads(
            &descriptor,
            &tools,
            &call(&descriptor, value),
            &json!({"x": value}),
        );
    }
}

#[test]
fn a_framed_string_value_loses_only_its_framing() {
    let descriptor = framed();
    let tools = [tool(
        json!({"type": "object", "properties": {"x": {"type": "string"}}}),
    )];
    for value in NEWLINE_VALUES {
        assert_reads(
            &descriptor,
            &tools,
            &call(&descriptor, value),
            &json!({"x": value}),
        );
    }
    // A completion without the framing reads the same value.
    let bare =
        "<tool_call>\n<function=f>\n<parameter=x>text</parameter>\n</function>\n</tool_call>";
    assert_reads(&descriptor, &tools, bare, &json!({"x": "text"}));
}

#[test]
fn a_key_value_string_value_keeps_every_byte() {
    let descriptor = key_value();
    let tools = [tool(
        json!({"type": "object", "properties": {"x": {"type": "string"}}}),
    )];
    for value in NEWLINE_VALUES {
        assert_reads(
            &descriptor,
            &tools,
            &call(&descriptor, value),
            &json!({"x": value}),
        );
    }
}

/// A typed additional property is read by its schema, as a declared
/// property is, in each grammar.
#[test]
fn a_typed_additional_property_is_coerced_by_its_schema() {
    for descriptor in [framed(), unframed(), key_value()] {
        for (schema, text, value) in [
            (json!({"type": "integer"}), "42", json!(42)),
            (json!({"type": "number"}), "0.25", json!(0.25)),
            (json!({"type": "boolean"}), "true", json!(true)),
            (
                json!({"type": "object"}),
                r#"{"a": [1, true, null], "b": "</parameter>"}"#,
                json!({"a": [1, true, null], "b": "</parameter>"}),
            ),
            (json!({"type": "string"}), "42", json!("42")),
        ] {
            let tools = [tool(
                json!({"type": "object", "additionalProperties": schema}),
            )];
            assert_reads(
                &descriptor,
                &tools,
                &call(&descriptor, text),
                &json!({"x": value}),
            );
            // A required name outside the properties resolves the same way.
            let tools = [tool(
                json!({"type": "object", "properties": {"y": {"type": "string"}},
                                     "required": ["x"], "additionalProperties": schema}),
            )];
            assert_reads(
                &descriptor,
                &tools,
                &call(&descriptor, text),
                &json!({"x": value}),
            );
        }
    }
}

/// A name that pattern properties may type, or that no schema types, is
/// free text: its value is a string.
#[test]
fn an_untyped_parameter_is_a_string() {
    for descriptor in [framed(), unframed(), key_value()] {
        for parameters in [
            json!({"type": "object", "patternProperties": {"^x$": {"type": "integer"}},
                   "additionalProperties": {"type": "integer"}}),
            json!({"type": "object", "additionalProperties": true}),
            json!({"type": "object"}),
            json!({"type": "object", "properties": {"x": {}}}),
        ] {
            let tools = [tool(parameters)];
            assert_reads(
                &descriptor,
                &tools,
                &call(&descriptor, "42"),
                &json!({"x": "42"}),
            );
        }
    }
}

/// A value of a string type, written by name or by an alias the coercion
/// reads as a string, is raw text in every grammar: a leading quote, brace,
/// bracket or digit does not make it JSON. Before this held, an alias such
/// as `str` scanned the value as JSON, so `"é` opened a string literal that
/// never closed and the call lost its argument.
#[test]
fn a_string_type_or_alias_keeps_its_raw_text() {
    let types = [
        json!("string"),
        json!("str"),
        json!("text"),
        json!("varchar"),
        json!("char"),
        json!("enum"),
        json!(" STR "),
        json!(["str", "null"]),
        json!("unknown-type"),
    ];
    let values = [
        "\"é",
        "\"quoted\"",
        "{\"a\": 1}",
        "{unclosed",
        "[1, 2]",
        "[",
        "42",
        "1e5",
        "true",
        "null",
    ];
    for descriptor in [framed(), unframed(), key_value()] {
        for kind in &types {
            let schemas = [
                json!({"type": "object", "properties": {"x": {"type": kind}}}),
                json!({"type": "object", "additionalProperties": {"type": kind}}),
            ];
            for parameters in schemas {
                let tools = [tool(parameters)];
                for value in values {
                    // A nullable alias reads the text `null` as null, as
                    // a declared string-or-null does.
                    let expected = if value == "null" && kind.is_array() {
                        json!({"x": null})
                    } else {
                        json!({"x": value})
                    };
                    assert_reads(&descriptor, &tools, &call(&descriptor, value), &expected);
                }
            }
        }
    }
}

/// Whether the grammar `tool_choice: required` imposes on `tools` admits
/// `text` whole, over a vocabulary of one token per byte; `None` when the
/// tools are refused.
fn required_admits(
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
    let vocabulary = Arc::new(TokenVocabulary::new(256, tokens).unwrap());
    let compiled = Arc::new(CompiledGrammar::new(&grammar, vocabulary).unwrap());
    let mut matcher = GrammarMatcher::new(compiled);
    Some(matcher.accept_bytes(text.as_bytes()).is_ok() && matcher.is_complete())
}

/// A string enumeration or constant value is written as it is, so a forced
/// call can only offer one the parser reads back whole, whatever layout
/// whitespace the declared wrappers carry. A value that spells
/// a marker ending a raw value, or that opens a string literal a JSON scan
/// never closes, refuses the tools; every value offered reads back, under a
/// string type, its aliases and an unknown type name, as a property and as
/// an additional property, in all three families.
/// `descriptor` as declared, and with layout whitespace before its tool
/// call open, after its tool call close, and both: the parser reads the bare
/// wrappers, so the same calls parse.
fn wrapper_layouts(descriptor: &FormatDescriptorV1) -> Vec<FormatDescriptorV1> {
    [(false, false), (true, false), (false, true), (true, true)]
        .into_iter()
        .map(|(leading, trailing)| {
            let mut layout = descriptor.clone();
            let parser = &mut layout.parser;
            if leading {
                parser.tool_call_open = format!("\n{}", parser.tool_call_open);
            }
            if trailing {
                parser.tool_call_close = format!("{}\n", parser.tool_call_close);
            }
            layout.validate().unwrap();
            layout
        })
        .collect()
}

#[test]
fn a_forced_enumeration_value_reads_back_or_is_refused() {
    let families = [framed(), unframed(), key_value()];
    for descriptor in families.iter().flat_map(wrapper_layouts) {
        let parser = descriptor.parser.clone();
        let mut markers: Vec<&str> = [
            &parser.parameter_open,
            &parser.parameter_close,
            &parser.function_close,
            &parser.tool_call_close,
        ]
        .map(|marker| marker.trim())
        .into_iter()
        .filter(|marker| !marker.is_empty())
        .collect();
        markers.dedup();
        // The markers that end a raw value in this family.
        let ending: Vec<&str> = if parser.function_open.is_empty() {
            vec![parser.parameter_close.trim(), parser.tool_call_close.trim()]
        } else {
            vec![
                parser.parameter_close.trim(),
                parser.function_close.trim(),
                parser.parameter_open.trim(),
            ]
        };
        for kind in ["string", "str", " STR ", "text", "enum", "unknown-type"] {
            let mut values: Vec<String> = markers
                .iter()
                .flat_map(|marker| [format!("\"before{marker}after\""), format!("x{marker}")])
                .collect();
            values.extend(["\"quoted\"", "\"open", "plain", ""].map(str::to_owned));
            for value in &values {
                let schemas = [
                    json!({"type": "object", "properties": {"x": {"type": kind, "enum": [value]}},
                           "required": ["x"], "additionalProperties": false}),
                    json!({"type": "object", "properties": {"x": {"type": kind, "const": value}},
                           "required": ["x"]}),
                    json!({"type": "object", "required": ["x"],
                           "additionalProperties": {"type": kind, "enum": [value]}}),
                ];
                for parameters in schemas {
                    let tools = [tool(parameters)];
                    let text = call(&descriptor, value);
                    let label = format!("{kind} {value:?}");
                    match required_admits(&descriptor, &tools, &text) {
                        None => assert!(
                            ending.iter().any(|marker| value.contains(marker)),
                            "{label}: refused, though it spells no ending marker"
                        ),
                        Some(admitted) => {
                            assert!(admitted, "{label}: the grammar does not write it");
                            assert!(
                                !ending.iter().any(|marker| value.contains(marker)),
                                "{label}: offered, though it spells an ending marker"
                            );
                            assert_reads(&descriptor, &tools, &text, &json!({"x": value}));
                        }
                    }
                }
            }
        }
        // A JSON value's scan reads a quote as a string literal, so a value
        // whose literal never closes cannot be offered either.
        let tools = [tool(json!({"type": "object",
                                 "properties": {"x": {"type": "integer", "enum": ["\"open"]}},
                                 "required": ["x"]}))];
        assert_eq!(
            required_admits(&descriptor, &tools, &call(&descriptor, "\"open")),
            None
        );
    }
}

/// One call to `f` with `arguments` in order, as each fixture's template
/// writes it.
fn call_with(descriptor: &FormatDescriptorV1, arguments: &[(&str, &str)]) -> String {
    let parser = &descriptor.parser;
    let mut text = [
        parser.tool_call_open.as_str(),
        &parser.function_open,
        "f",
        &parser.function_name_close,
    ]
    .concat();
    for (name, value) in arguments {
        text.extend([
            parser.parameter_open.as_str(),
            name,
            &parser.parameter_name_close,
            &parser.parameter_value_open,
            value,
            &parser.parameter_close,
        ]);
    }
    text.extend([parser.function_close.as_str(), &parser.tool_call_close]);
    text
}

/// An additional property never takes a declared name, which the parser
/// reads by the declared schema instead. Before this held, a forced call
/// could write the additional enumeration value `"open` under the integer
/// `x`, which scans the quote as a JSON string that never closes, and the
/// call read back as `f({})`.
#[test]
fn an_additional_property_never_takes_a_declared_name() {
    for descriptor in [framed(), unframed(), key_value()] {
        let tools = [tool(json!({"type": "object",
                                 "properties": {"x": {"type": "integer"}},
                                 "additionalProperties": {"type": "str", "enum": ["\"open"]}}))];
        let admits = |arguments: &[(&str, &str)]| {
            required_admits(&descriptor, &tools, &call_with(&descriptor, arguments)).unwrap()
        };
        assert!(!admits(&[("x", "\"open")]));
        assert!(!admits(&[("x", "42"), ("x", "\"open")]));
        for (arguments, expected) in [
            (&[("x", "42")][..], json!({"x": 42})),
            (&[("y", "\"open")], json!({"y": "\"open"})),
            (
                &[("x", "42"), ("xx", "\"open")],
                json!({"x": 42, "xx": "\"open"}),
            ),
        ] {
            assert!(admits(arguments), "{arguments:?}");
            assert_reads(
                &descriptor,
                &tools,
                &call_with(&descriptor, arguments),
                &expected,
            );
        }
    }
}
