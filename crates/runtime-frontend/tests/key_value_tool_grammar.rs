#![forbid(unsafe_code)]
//! The key-value tool grammar: `<tool_call>name<arg_key>k</arg_key>
//! <arg_value>v</arg_value></tool_call>`, with no function element, raw
//! string values and JSON for every other value.
use std::sync::Arc;

use frontend_format::test_support::fixtures::KEY_VALUE;
use runtime_frontend::{
    AssistantCompletionV1, CompletionDecoderV1, CompletionUpdateV1, DecoderLimitsV1,
    FrontendDescriptorV1, GeneratedTokenV1, ParsedToolCallV1, PromptModeV1, parse_completion_v1,
};
use runtime_grammar::{CompiledGrammar, ConstraintState, TokenVocabulary, VocabularyToken};
use runtime_server::{FrontendToolChoiceV1, FrontendToolV1, ServerFrontendV1};
use serde_json::{Value, json};

const STOP: &str = "[EOS]";

/// The key-value fixture, whose reasoning closes with `</t>`, with room for
/// many calls.
fn descriptor() -> FrontendDescriptorV1 {
    let mut d = FrontendDescriptorV1::from_component_bytes(KEY_VALUE.as_bytes()).unwrap();
    d.limits.maximum_tool_calls = 64;
    d.validate().unwrap();
    d
}

fn tool(name: &str, parameters: Value) -> FrontendToolV1 {
    FrontendToolV1 {
        name: name.to_owned(),
        description: None,
        parameters: serde_json::from_value(parameters).unwrap(),
        parameters_present: true,
        strict: None,
    }
}

fn tools() -> Vec<FrontendToolV1> {
    vec![
        tool(
            "get_weather",
            json!({"type": "object", "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer", "minimum": 1, "maximum": 14},
                "metric": {"type": "boolean"},
                "ratio": {"type": "number"},
                "options": {"type": "object", "properties": {
                    "units": {"enum": ["c", "f"]}, "deep": {"type": "object", "properties": {
                        "x": {"type": "array", "items": {"type": ["integer", "null"]}}
                    }, "additionalProperties": false}
                }, "additionalProperties": false},
                "hours": {"type": "array", "items": {"type": "integer"}},
                "note": {"type": ["string", "null"]}
            }, "required": ["city", "days"]}),
        ),
        tool(
            "ping",
            json!({"type": "object", "properties": {}, "additionalProperties": false}),
        ),
        tool(
            "lookup",
            json!({"type": "object", "properties": {
                "payload": {"type": "object"},
                "list": {"type": "array"},
                "text": {"type": "string"}
            }, "required": ["payload"]}),
        ),
        FrontendToolV1 {
            strict: Some(false),
            ..tool(
                "loose",
                json!({"type": "object", "properties": {"payload": {"type": "object"}}}),
            )
        },
    ]
}

const CALLS: &str = concat!(
    "plan</t>Checking.",
    "<tool_call>get_weather<arg_key>city</arg_key><arg_value> New\nYork </arg_value>",
    "<arg_key>days</arg_key>\n<arg_value>3</arg_value>",
    "<arg_key>metric</arg_key><arg_value>true</arg_value>",
    "<arg_key>ratio</arg_key><arg_value>2.5</arg_value>",
    "<arg_key>options</arg_key><arg_value>{\"units\": \"c\", \"deep\": {\"x\": [1, null]}}</arg_value>",
    "<arg_key>hours</arg_key><arg_value>[\"1\", 2]</arg_value>",
    "<arg_key>note</arg_key><arg_value>null</arg_value></tool_call>",
    "<tool_call>ping</tool_call>",
    "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value>",
    "<arg_key>days</arg_key><arg_value>-0</arg_value></tool_call>[EOS]"
);

fn expected_calls() -> Vec<ParsedToolCallV1> {
    let call = |name: &str, arguments: &str| ParsedToolCallV1 {
        name: name.to_owned(),
        arguments: arguments.to_owned(),
    };
    vec![
        call(
            "get_weather",
            concat!(
                r#"{"city": " New\nYork ", "days": 3, "metric": true, "ratio": 2.5, "#,
                r#""options": {"units": "c", "deep": {"x": [1, null]}}, "hours": [1, 2], "#,
                r#""note": null}"#
            ),
        ),
        call("ping", "{}"),
        call("get_weather", r#"{"city": "北京", "days": 0}"#),
    ]
}

/// Decodes `chunks` as successive tokens with the request's tool schemas, and
/// returns the finished completion and every call's streamed arguments.
fn decode(
    d: &Arc<FrontendDescriptorV1>,
    chunks: &[&[u8]],
    stream: bool,
) -> Result<(AssistantCompletionV1, Vec<(String, String)>), String> {
    let mut decoder = CompletionDecoderV1::new(
        Arc::clone(d),
        PromptModeV1::Thinking,
        DecoderLimitsV1::declared(d),
    );
    decoder.set_tool_schemas(&tools());
    decoder.set_decoding_options(
        runtime_server::DecodingOptions {
            stream_tool_calls: stream,
            ..Default::default()
        },
        false,
    );
    let mut streamed: Vec<(String, String)> = Vec::new();
    let mut finished = None;
    let mut reasoning = String::new();
    let mut content = String::new();
    for (ordinal, bytes) in chunks.iter().filter(|bytes| !bytes.is_empty()).enumerate() {
        for update in decoder.push(GeneratedTokenV1::new(ordinal as u64, 42, bytes.to_vec()))? {
            match update {
                CompletionUpdateV1::Reasoning(part) => reasoning.push_str(&part.text),
                CompletionUpdateV1::Content(part) => content.push_str(&part.text),
                CompletionUpdateV1::ToolCallDeltas(deltas) => {
                    for delta in deltas {
                        if delta.index == streamed.len() {
                            streamed.push((delta.name.clone().unwrap(), String::new()));
                        }
                        assert!(delta.name.is_none() || streamed[delta.index].1.is_empty());
                        streamed[delta.index].1.push_str(&delta.arguments);
                    }
                }
                CompletionUpdateV1::Finished(end) => finished = end.completion,
                CompletionUpdateV1::ToolCalls(_) => {}
            }
        }
    }
    let completion = finished.ok_or("no finish")?;
    assert_eq!(
        completion.reasoning_content.as_deref(),
        Some(reasoning.as_str())
    );
    assert_eq!(completion.content, content);
    Ok((completion, streamed))
}

fn calls_of(text: &str) -> Vec<ParsedToolCallV1> {
    let d = Arc::new(descriptor());
    decode(&d, &[text.as_bytes()], false).unwrap().0.tool_calls
}

#[test]
fn whole_completions_read_names_raw_values_and_schema_types() {
    let d = Arc::new(descriptor());
    let (completion, _) = decode(&d, &[CALLS.as_bytes()], false).unwrap();
    assert_eq!(completion.reasoning_content.as_deref(), Some("plan"));
    assert_eq!(completion.content, "Checking.");
    assert_eq!(completion.tool_calls, expected_calls());
    // Without a schema every value stays the raw string it was.
    let untyped = parse_completion_v1(&d, CALLS, PromptModeV1::Thinking).unwrap();
    assert_eq!(untyped.tool_calls.len(), 3);
    assert_eq!(
        untyped.tool_calls[0].arguments,
        concat!(
            r#"{"city": " New\nYork ", "days": "3", "metric": "true", "ratio": "2.5", "#,
            r#""options": "{\"units\": \"c\", \"deep\": {\"x\": [1, null]}}", "#,
            r#""hours": "[\"1\", 2]", "note": "null"}"#
        )
    );
}

#[test]
fn streamed_calls_match_the_whole_parse_at_every_byte_split() {
    let d = Arc::new(descriptor());
    let bytes = CALLS.as_bytes();
    let expected = decode(&d, &[bytes], false).unwrap().0;
    let check = |chunks: &[&[u8]], label: &str| {
        for stream in [false, true] {
            let (completion, streamed) = decode(&d, chunks, stream).unwrap();
            assert_eq!(completion, expected, "{label}");
            if stream {
                let calls: Vec<_> = expected
                    .tool_calls
                    .iter()
                    .map(|call| (call.name.clone(), call.arguments.clone()))
                    .collect();
                assert_eq!(streamed, calls, "{label}");
            }
        }
    };
    for split in 0..=bytes.len() {
        check(
            &[&bytes[..split], &bytes[split..]],
            &format!("split {split}"),
        );
    }
    let single: Vec<&[u8]> = bytes.chunks(1).collect();
    check(&single, "one byte per token");
}

#[test]
fn malformed_arguments_keep_only_complete_pairs() {
    let arguments = |text: &str| -> Vec<String> {
        calls_of(text)
            .into_iter()
            .map(|call| format!("{}{}", call.name, call.arguments))
            .collect()
    };
    // A key without a value element is skipped, and parsing resumes after it.
    assert_eq!(
        arguments(
            "</t><tool_call>get_weather<arg_key>city</arg_key>x<arg_key>days</arg_key> <arg_value>2</arg_value></tool_call>[EOS]"
        ),
        [r#"get_weather{"days": 2}"#]
    );
    // An unclosed value or key ends the arguments at the call close.
    assert_eq!(
        arguments(
            "</t><tool_call>get_weather<arg_key>days</arg_key><arg_value>2</arg_value><arg_key>city</arg_key><arg_value>P</tool_call>[EOS]"
        ),
        [r#"get_weather{"days": 2}"#]
    );
    assert_eq!(
        arguments("</t><tool_call>ping<arg_key>days</tool_call>[EOS]"),
        ["ping{}"]
    );
    // A repeated key keeps its first position and its last value.
    assert_eq!(
        arguments(
            "</t><tool_call>get_weather<arg_key>days</arg_key><arg_value>1</arg_value><arg_key>city</arg_key><arg_value>A</arg_value><arg_key>days</arg_key><arg_value>5</arg_value></tool_call>[EOS]"
        ),
        [r#"get_weather{"days": 5, "city": "A"}"#]
    );
    // A schema type the text does not satisfy leaves the raw string.
    assert_eq!(
        arguments(
            "</t><tool_call>get_weather<arg_key>days</arg_key><arg_value>soon</arg_value></tool_call>[EOS]"
        ),
        [r#"get_weather{"days": "soon"}"#]
    );
    // A region without a name calls nothing.
    assert!(
        arguments(
            "</t><tool_call> <arg_key>days</arg_key><arg_value>1</arg_value></tool_call>[EOS]"
        )
        .is_empty()
    );
    // A key the name rule refuses fails the completion.
    let d = Arc::new(descriptor());
    let bad = "</t><tool_call>ping<arg_key>a b</arg_key><arg_value>1</arg_value></tool_call>[EOS]";
    assert!(decode(&d, &[bad.as_bytes()], false).is_err());
}

#[test]
fn a_call_open_at_the_end_keeps_its_complete_arguments() {
    let d = Arc::new(descriptor());
    let text =
        "</t><tool_call>get_weather<arg_key>days</arg_key><arg_value>4</arg_value><arg_key>ci";
    for end in [true, false] {
        let mut decoder = CompletionDecoderV1::new(
            Arc::clone(&d),
            PromptModeV1::Thinking,
            DecoderLimitsV1::declared(&d),
        );
        decoder.set_tool_schemas(&tools());
        let mut calls = Vec::new();
        let pushed = if end {
            decoder.push(GeneratedTokenV1::new(
                0,
                42,
                format!("{text}{STOP}").into_bytes(),
            ))
        } else {
            decoder.push(GeneratedTokenV1::new(0, 42, text.as_bytes().to_vec()))
        };
        let mut updates = pushed.unwrap();
        if !end {
            updates.extend(decoder.finish_length().unwrap());
        }
        for update in updates {
            if let CompletionUpdateV1::ToolCalls(group) = update {
                calls.extend(group.calls);
            }
        }
        assert_eq!(
            calls,
            [ParsedToolCallV1 {
                name: "get_weather".into(),
                arguments: r#"{"days": 4}"#.into(),
            }],
            "end token {end}"
        );
    }
}

/// Single bytes, the call markers as special tokens, multi-byte spellings of
/// the argument markers, and the stop token.
const MARKERS: [(&str, bool); 7] = [
    ("<tool_call>", true),
    ("</tool_call>", true),
    ("<arg_key>", false),
    ("</arg_key>", false),
    ("<arg_value>", false),
    ("</arg_value>", false),
    (STOP, true),
];

fn vocabulary() -> (Arc<TokenVocabulary>, u32) {
    let mut tokens: Vec<VocabularyToken> = (0..=255_u8)
        .map(|byte| VocabularyToken {
            bytes: vec![byte],
            special: false,
        })
        .collect();
    for (text, special) in MARKERS {
        tokens.push(VocabularyToken {
            bytes: text.as_bytes().to_vec(),
            special,
        });
    }
    let stop = tokens.len() as u32 - 1;
    (
        Arc::new(TokenVocabulary::new(tokens.len(), tokens).unwrap()),
        stop,
    )
}

fn walk(grammar: &Arc<CompiledGrammar>, stop: u32, seed: &mut u64) -> Option<String> {
    let vocabulary = Arc::clone(grammar.vocabulary());
    let mut state = ConstraintState::new(Arc::clone(grammar), None, vec![stop]);
    let mut mask = vec![0_u32; vocabulary.mask_words()];
    let mut text = Vec::new();
    let mut below = |bound: usize| {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed % bound as u64) as usize
    };
    for _ in 0..20_000 {
        assert!(state.fill_mask(&mut mask).unwrap());
        let allowed: Vec<u32> = (0..vocabulary.size() as u32)
            .filter(|&token| mask[token as usize / 32] & (1 << (token % 32)) != 0)
            .collect();
        let long: Vec<u32> = allowed.iter().copied().filter(|&t| t >= 256).collect();
        let token = if !long.is_empty() && below(2) == 0 {
            long[below(long.len())]
        } else {
            allowed[below(allowed.len())]
        };
        state.accept(token).unwrap();
        if token == stop {
            return Some(String::from_utf8(text).expect("the grammar emits UTF-8"));
        }
        let bytes = vocabulary
            .ordinary_bytes(token)
            .unwrap_or_else(|| MARKERS[token as usize - 256].0.as_bytes());
        text.extend_from_slice(bytes);
    }
    None
}

/// The walks include free JSON strings, which may spell either close marker
/// as content.
#[test]
fn forced_calls_follow_the_key_value_form_and_parse_back_typed() {
    let d = descriptor();
    let frontend = runtime_frontend::BundleFrontendV1::with_export(
        d.clone(),
        &runtime_frontend::test_support::fixture_export_v1(),
    )
    .unwrap();
    let shared = Arc::new(d);
    let (vocabulary, stop) = vocabulary();
    let tools = tools();
    let mut seed = 0x9e37_79b9_u64;
    for choice in [
        FrontendToolChoiceV1::Required,
        FrontendToolChoiceV1::Named("get_weather".to_owned()),
    ] {
        let grammar = frontend.tool_constraint(&tools, &choice).unwrap().unwrap();
        let compiled = Arc::new(CompiledGrammar::new(&grammar, Arc::clone(&vocabulary)).unwrap());
        let mut parsed = 0;
        let mut several = false;
        for _ in 0..40 {
            let Some(text) = walk(&compiled, stop, &mut seed) else {
                continue;
            };
            let completion = format!("</t>{text}{STOP}");
            let (completion, _) = decode(&shared, &[completion.as_bytes()], false)
                .unwrap_or_else(|error| panic!("{error}: {text:?}"));
            assert!(completion.content.is_empty(), "{text:?}");
            several |= completion.tool_calls.len() > 1;
            for call in &completion.tool_calls {
                let arguments: Value = serde_json::from_str(&call.arguments).unwrap();
                match call.name.as_str() {
                    "get_weather" => {
                        assert!(arguments["city"].is_string(), "{text:?}");
                        let days = arguments["days"].as_i64().unwrap();
                        assert!((1..=14).contains(&days), "{text:?}");
                        for (key, typed) in [
                            (
                                "metric",
                                arguments.get("metric").is_none_or(Value::is_boolean),
                            ),
                            // A number too large for a double stays its text.
                            (
                                "ratio",
                                arguments.get("ratio").is_none_or(|ratio| {
                                    ratio.is_number()
                                        || ratio
                                            .as_str()
                                            .and_then(|text| text.parse::<f64>().ok())
                                            .is_some_and(f64::is_infinite)
                                }),
                            ),
                            (
                                "options",
                                arguments.get("options").is_none_or(Value::is_object),
                            ),
                            ("hours", arguments.get("hours").is_none_or(Value::is_array)),
                        ] {
                            assert!(typed, "{key}: {text:?}");
                        }
                    }
                    "ping" => assert_eq!(call.arguments, "{}"),
                    "lookup" => {
                        // JSON with a number too large for a double stays its
                        // text, as an overflowing number value does.
                        let json = |value: &Value, kind: fn(&Value) -> bool| {
                            kind(value)
                                || value.as_str().is_some_and(|text| {
                                    serde_json::from_str::<runtime_server::FrontendJsonV1>(text)
                                        .is_err_and(|error| {
                                            error.to_string().contains("non-finite JSON number")
                                        })
                                })
                        };
                        assert!(json(&arguments["payload"], Value::is_object), "{text:?}");
                        for (key, typed) in [
                            (
                                "list",
                                arguments
                                    .get("list")
                                    .is_none_or(|list| json(list, Value::is_array)),
                            ),
                            ("text", arguments.get("text").is_none_or(Value::is_string)),
                        ] {
                            assert!(typed, "{key}: {text:?}");
                        }
                    }
                    "loose" => assert!(arguments.is_object()),
                    other => panic!("unexpected function {other}"),
                }
                if let FrontendToolChoiceV1::Named(name) = &choice {
                    assert_eq!(&call.name, name);
                }
            }
            parsed += 1;
        }
        assert!(parsed > 10, "{choice:?}: only {parsed} walks finished");
        if choice == FrontendToolChoiceV1::Required {
            assert!(several, "no walk produced several calls");
        }
    }
}

/// A typed value's JSON string literals may spell either close marker; the
/// decoder reads them as content, as the tool-choice grammar writes them. A
/// raw value, and any value of a tool that does not constrain its arguments,
/// ends at the first close marker whatever it holds.
#[test]
fn close_markers_inside_json_strings_are_content() {
    let d = Arc::new(descriptor());
    let cases = [
        (
            r#"<tool_call>lookup<arg_key>payload</arg_key><arg_value>{"message":"before</arg_value>after"}</arg_value></tool_call>"#,
            r#"{"payload": {"message": "before</arg_value>after"}}"#,
        ),
        (
            r#"<tool_call>lookup<arg_key>payload</arg_key><arg_value>{"message":"before</tool_call>after"}</arg_value></tool_call>"#,
            r#"{"payload": {"message": "before</tool_call>after"}}"#,
        ),
        (
            r#"<tool_call>lookup<arg_key>payload</arg_key><arg_value>{"m":"a\"</arg_value>\\","n":["<arg_key>x</arg_key>"]}</arg_value><arg_key>list</arg_key><arg_value>["</tool_call>", {"k": "</arg_value>"}]</arg_value></tool_call>"#,
            r#"{"payload": {"m": "a\"</arg_value>\\", "n": ["<arg_key>x</arg_key>"]}, "list": ["</tool_call>", {"k": "</arg_value>"}]}"#,
        ),
        (
            r#"<tool_call>lookup<arg_key>text</arg_key><arg_value>a"b</arg_value>c</arg_value><arg_key>payload</arg_key><arg_value>{}</arg_value></tool_call>"#,
            r#"{"text": "a\"b", "payload": {}}"#,
        ),
        (
            r#"<tool_call>loose<arg_key>payload</arg_key><arg_value>{"m":"x</arg_value>y"}</arg_value></tool_call>"#,
            r#"{"payload": "{\"m\":\"x"}"#,
        ),
    ];
    for (call, arguments) in cases {
        let text = format!("</t>{call}after{STOP}");
        let bytes = text.as_bytes();
        let check = |chunks: &[&[u8]], label: &str| {
            for stream in [false, true] {
                let (completion, streamed) = decode(&d, chunks, stream).unwrap();
                assert_eq!(completion.content, "after", "{label}: {call}");
                assert_eq!(completion.tool_calls.len(), 1, "{label}: {call}");
                assert_eq!(completion.tool_calls[0].arguments, arguments, "{label}");
                if stream {
                    assert_eq!(streamed[0].1, arguments, "{label}");
                }
            }
        };
        check(&[bytes], "whole");
        check(&bytes.chunks(1).collect::<Vec<_>>(), "one byte per token");
        for split in 0..=bytes.len() {
            check(
                &[&bytes[..split], &bytes[split..]],
                &format!("split {split}"),
            );
        }
    }
    // An unterminated string keeps the call open to the end of generation,
    // which keeps only the complete arguments.
    let open = r#"</t><tool_call>lookup<arg_key>list</arg_key><arg_value>[1]</arg_value><arg_key>payload</arg_key><arg_value>{"m":"x</arg_value></tool_call>tail[EOS]"#;
    let (completion, _) = decode(&d, &[open.as_bytes()], false).unwrap();
    assert_eq!(completion.content, "");
    assert_eq!(completion.tool_calls[0].arguments, r#"{"list": [1]}"#);
}

/// The forced-call grammar admits a JSON string that spells the close
/// markers, so the decoder has to read it back as content.
#[test]
fn the_tool_choice_grammar_admits_close_markers_inside_json_strings() {
    let frontend = runtime_frontend::BundleFrontendV1::with_export(
        descriptor(),
        &runtime_frontend::test_support::fixture_export_v1(),
    )
    .unwrap();
    let (vocabulary, stop) = vocabulary();
    let grammar = frontend
        .tool_constraint(&tools(), &FrontendToolChoiceV1::Named("lookup".to_owned()))
        .unwrap()
        .unwrap();
    let compiled = Arc::new(CompiledGrammar::new(&grammar, Arc::clone(&vocabulary)).unwrap());
    for inner in ["</arg_value>", "</tool_call>"] {
        let text = format!(
            r#"<tool_call>lookup<arg_key>payload</arg_key><arg_value>{{"message":"before{inner}after"}}</arg_value></tool_call>"#
        );
        let mut state = ConstraintState::new(Arc::clone(&compiled), None, vec![stop]);
        let call_open = MARKERS
            .iter()
            .position(|(marker, _)| *marker == "<tool_call>")
            .unwrap() as u32
            + 256;
        state.accept(call_open).unwrap();
        for byte in &text.as_bytes()["<tool_call>".len()..] {
            state.accept(u32::from(*byte)).unwrap();
        }
        state.accept(stop).unwrap();
    }
}
