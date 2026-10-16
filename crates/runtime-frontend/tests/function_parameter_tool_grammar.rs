#![forbid(unsafe_code)]
//! The function-parameter tool grammar: `<tool_call><function=name>
//! <parameter=k>v</parameter></function></tool_call>`, with raw string values
//! and JSON for every other value. A closing marker inside a JSON value's
//! string literal is content.
use std::sync::Arc;

use frontend_format::test_support::fixtures::FUNCTION_PARAMETERS;
use runtime_frontend::{
    AssistantCompletionV1, CompletionDecoderV1, CompletionUpdateV1, DecoderLimitsV1,
    FrontendDescriptorV1, GeneratedTokenV1, PromptModeV1,
};
use runtime_grammar::{CompiledGrammar, ConstraintState, TokenVocabulary, VocabularyToken};
use runtime_server::{FrontendToolChoiceV1, FrontendToolV1, ServerFrontendV1};
use serde_json::{Value, json};

const STOP: &str = "[EOS]";

/// The cases tools/qual's agentic raw re-parse reads too
/// (tools/qual/test_agentic_markup.py), with this file's markers and tools.
use frontend_format::test_support::FUNCTION_PARAMETER_CASES as SHARED_CASES;

fn shared_cases() -> Value {
    serde_json::from_str(SHARED_CASES).unwrap()
}

/// The function-parameter fixture, whose parser section is the shared
/// cases': the markers with the whitespace a template writes around them,
/// which the decoder does not require. A split here may be one byte per
/// token, so the completion has room for that many tokens.
fn descriptor() -> FrontendDescriptorV1 {
    let mut d = FrontendDescriptorV1::from_component_bytes(FUNCTION_PARAMETERS.as_bytes()).unwrap();
    d.tokenizer.model_max_length = 8192;
    d.context.maximum_total_tokens = 8192;
    d.limits.maximum_generated_tokens = 8192;
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

/// The request tools of the shared cases, converted as a request converts
/// them.
fn tools() -> Vec<FrontendToolV1> {
    let cases = shared_cases();
    cases["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| {
            let function = &tool["function"];
            FrontendToolV1 {
                name: function["name"].as_str().unwrap().to_owned(),
                description: None,
                parameters: serde_json::from_value(
                    function.get("parameters").cloned().unwrap_or(json!({})),
                )
                .unwrap(),
                parameters_present: function.get("parameters").is_some(),
                strict: function.get("strict").and_then(Value::as_bool),
            }
        })
        .collect()
}

/// Decodes `chunks` as successive tokens with `tools`, and returns the
/// finished completion and every call's streamed arguments.
fn decode(
    d: &Arc<FrontendDescriptorV1>,
    tools: &[FrontendToolV1],
    chunks: &[&[u8]],
    stream: bool,
) -> Result<(AssistantCompletionV1, Vec<(String, String)>), String> {
    let mut decoder = CompletionDecoderV1::new(
        Arc::clone(d),
        PromptModeV1::Thinking,
        DecoderLimitsV1::declared(d),
    );
    decoder.set_tool_schemas(tools);
    decoder.set_decoding_options(
        runtime_server::DecodingOptions {
            stream_tool_calls: stream,
            ..Default::default()
        },
        false,
    );
    let mut streamed: Vec<(String, String)> = Vec::new();
    let mut finished = None;
    let mut content = String::new();
    for (ordinal, bytes) in chunks.iter().filter(|bytes| !bytes.is_empty()).enumerate() {
        for update in decoder.push(GeneratedTokenV1::new(ordinal as u64, 42, bytes.to_vec()))? {
            match update {
                CompletionUpdateV1::Content(part) => content.push_str(&part.text),
                CompletionUpdateV1::ToolCallDeltas(deltas) => {
                    for delta in deltas {
                        if delta.index == streamed.len() {
                            streamed.push((delta.name.clone().unwrap(), String::new()));
                        }
                        streamed[delta.index].1.push_str(&delta.arguments);
                    }
                }
                CompletionUpdateV1::Finished(end) => finished = end.completion,
                CompletionUpdateV1::Reasoning(_) | CompletionUpdateV1::ToolCalls(_) => {}
            }
        }
    }
    let completion = finished.ok_or("no finish")?;
    assert_eq!(completion.content, content);
    Ok((completion, streamed))
}

/// One call in the template's layout, with one value per parameter.
fn call(name: &str, parameters: &[(&str, &str)]) -> String {
    let mut text = format!("<tool_call>\n<function={name}>\n");
    for (parameter, value) in parameters {
        text.push_str(&format!("<parameter={parameter}>\n{value}\n</parameter>\n"));
    }
    text + "</function>\n</tool_call>"
}

/// Checks `text` whole, one byte per token and split at every byte, with and
/// without streamed calls.
fn check_every_split(tools: &[FrontendToolV1], text: &str, content: &str, calls: &[(&str, &str)]) {
    let d = Arc::new(descriptor());
    let bytes = text.as_bytes();
    let expected: Vec<(String, String)> = calls
        .iter()
        .map(|(name, arguments)| ((*name).to_owned(), (*arguments).to_owned()))
        .collect();
    let check = |chunks: &[&[u8]], label: &str| {
        for stream in [false, true] {
            let (completion, streamed) = decode(&d, tools, chunks, stream).unwrap();
            assert_eq!(completion.content, content, "{label}: {text:?}");
            let parsed: Vec<_> = completion
                .tool_calls
                .iter()
                .map(|call| (call.name.clone(), call.arguments.clone()))
                .collect();
            assert_eq!(parsed, expected, "{label}: {text:?}");
            if stream {
                assert_eq!(streamed, expected, "{label}, streamed: {text:?}");
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

/// A typed value's JSON string literals may spell any closing marker or the
/// next parameter's open; the decoder reads them as content, as the
/// tool-choice grammar writes them.
#[test]
fn closing_markers_inside_json_strings_are_content() {
    let cases = [
        (
            call("read", &[("filter", r#"{"pattern": "</parameter>"}"#)]),
            r#"{"filter": {"pattern": "</parameter>"}}"#,
        ),
        (
            call("read", &[("filter", r#"{"pattern": "</function>"}"#)]),
            r#"{"filter": {"pattern": "</function>"}}"#,
        ),
        (
            call("read", &[("filter", r#"{"pattern": "<parameter=x>"}"#)]),
            r#"{"filter": {"pattern": "<parameter=x>"}}"#,
        ),
        (
            call(
                "read",
                &[
                    (
                        "filter",
                        r#"{"m": "a\"</parameter>\\", "n": ["</function>\n</tool_call>"]}"#,
                    ),
                    ("text", "plain"),
                    ("list", r#"["<parameter=y>\n", {"k": "\n</parameter>\n"}]"#),
                ],
            ),
            r#"{"filter": {"m": "a\"</parameter>\\", "n": ["</function>\n</tool_call>"]}, "text": "plain", "list": ["<parameter=y>\n", {"k": "\n</parameter>\n"}]}"#,
        ),
    ];
    for (text, arguments) in cases {
        check_every_split(
            &tools(),
            &format!("plan</think>{text}after{STOP}"),
            "after",
            &[("read", arguments)],
        );
    }
}

/// Each shared case parses to its content and calls, so the re-parse in
/// tools/qual reads the same texts against what this parser returns.
#[test]
fn the_shared_cases_parse_to_their_calls() {
    let cases = shared_cases();
    let stop = cases["stop"].as_str().unwrap();
    assert_eq!(stop, STOP);
    for case in cases["cases"].as_array().unwrap() {
        let tools = if case.get("undeclared").and_then(Value::as_bool) == Some(true) {
            Vec::new()
        } else {
            tools()
        };
        let calls: Vec<(&str, &str)> = case["calls"]
            .as_array()
            .unwrap()
            .iter()
            .map(|call| (call[0].as_str().unwrap(), call[1].as_str().unwrap()))
            .collect();
        check_every_split(
            &tools,
            &format!("plan</think>{}{STOP}", case["text"].as_str().unwrap()),
            case["content"].as_str().unwrap(),
            &calls,
        );
    }
}

/// A raw value, any value of a tool that does not constrain its arguments,
/// and any value of a function the request did not declare end at the first
/// closing marker whatever they hold, as before.
#[test]
fn raw_values_end_at_the_first_closing_marker() {
    let cases = [
        (
            "read",
            call(
                "read",
                &[("text", r#"a "b</parameter>c"#), ("filter", "{}")],
            ),
            r#"{"text": "a \"b", "filter": {}}"#,
        ),
        (
            "loose",
            call("loose", &[("filter", r#"{"pattern": "</parameter>"}"#)]),
            r#"{"filter": "{\"pattern\": \""}"#,
        ),
        (
            "loose",
            call("loose", &[("filter", r#"{"pattern": "</function>"}"#)]),
            "{}",
        ),
    ];
    for (name, text, arguments) in cases {
        check_every_split(
            &tools(),
            &format!("plan</think>{text}after{STOP}"),
            "after",
            &[(name, arguments)],
        );
    }
    // An undeclared function's values are raw.
    let text = call("read", &[("filter", r#"{"pattern": "</parameter>"}"#)]);
    check_every_split(
        &[],
        &format!("plan</think>{text}after{STOP}"),
        "after",
        &[("read", r#"{"filter": "{\"pattern\": \""}"#)],
    );
}

/// An unterminated string keeps the call open to the end of generation, which
/// keeps only the complete parameters.
#[test]
fn an_unterminated_json_string_keeps_the_call_open() {
    let text = concat!(
        "plan</think><tool_call>\n<function=read>\n<parameter=list>\n[1]\n</parameter>\n",
        "<parameter=filter>\n{\"m\": \"x\n</parameter>\n</function>\n</tool_call>tail[EOS]"
    );
    check_every_split(&tools(), text, "", &[("read", r#"{"list": [1]}"#)]);
}

/// The forced-call grammar admits a JSON string that spells the closing
/// markers, so the decoder has to read it back as content.
#[test]
fn the_tool_choice_grammar_admits_closing_markers_inside_json_strings() {
    let frontend = runtime_frontend::BundleFrontendV1::with_export(
        descriptor(),
        &runtime_frontend::test_support::fixture_export_v1(),
    )
    .unwrap();
    let mut tokens: Vec<VocabularyToken> = (0..=255_u8)
        .map(|byte| VocabularyToken {
            bytes: vec![byte],
            special: false,
        })
        .collect();
    tokens.push(VocabularyToken {
        bytes: STOP.as_bytes().to_vec(),
        special: true,
    });
    let stop = tokens.len() as u32 - 1;
    let vocabulary = Arc::new(TokenVocabulary::new(tokens.len(), tokens).unwrap());
    let grammar = frontend
        .tool_constraint(&tools(), &FrontendToolChoiceV1::Named("read".to_owned()))
        .unwrap()
        .unwrap();
    let compiled = Arc::new(CompiledGrammar::new(&grammar, vocabulary).unwrap());
    for inner in ["</parameter>", "</function>", "<parameter=x>"] {
        let text = call(
            "read",
            &[("filter", &format!(r#"{{"pattern":"{inner}"}}"#))],
        );
        let mut state = ConstraintState::new(Arc::clone(&compiled), None, vec![stop]);
        for byte in text.as_bytes() {
            state.accept(u32::from(*byte)).unwrap();
        }
        state.accept(stop).unwrap();
    }
}

/// Splits `bytes` into tokens of `size` bytes.
fn tokens(bytes: &[u8], size: usize) -> Vec<&[u8]> {
    bytes.chunks(size).collect()
}

/// Streams `chunks` as engine tokens, as the host does, ends them with the
/// stop token, and returns the one call's streamed arguments.
fn stream_engine_tokens(
    d: &Arc<FrontendDescriptorV1>,
    tools: &[FrontendToolV1],
    chunks: &[&[u8]],
) -> String {
    let mut decoder = CompletionDecoderV1::new(
        Arc::clone(d),
        PromptModeV1::Thinking,
        DecoderLimitsV1::declared(d),
    );
    decoder.set_tool_schemas(tools);
    decoder.set_decoding_options(
        runtime_server::DecodingOptions {
            stream_tool_calls: true,
            ..Default::default()
        },
        false,
    );
    let stop = d.vocabulary.stop_token_ids[0];
    let tokens = chunks
        .iter()
        .map(|bytes| (42, bytes.to_vec()))
        .chain([(stop, STOP.as_bytes().to_vec())]);
    let mut streamed = String::new();
    let mut finished = false;
    for (ordinal, (id, bytes)) in tokens.enumerate() {
        let token = GeneratedTokenV1::new(ordinal as u64, id, bytes);
        for update in decoder.push_engine_token(token).unwrap() {
            match update {
                CompletionUpdateV1::ToolCallDeltas(deltas) => {
                    for delta in deltas {
                        assert_eq!(delta.index, 0);
                        streamed.push_str(&delta.arguments);
                    }
                }
                CompletionUpdateV1::Finished(_) => finished = true,
                _ => {}
            }
        }
    }
    assert!(finished);
    streamed
}

/// A JSON string that spells the function close many times costs one pass:
/// each close is checked from where the previous check stopped, and a
/// streamed region reads only the bytes each token adds. Walking the
/// arguments again for every close took seconds here, and over a minute
/// streamed.
#[test]
fn many_closes_inside_one_json_string_decode_in_linear_time() {
    let mut d = descriptor();
    // As large as the released declarations allow.
    d.limits.maximum_completion_bytes = 1 << 26;
    d.limits.maximum_json_bytes = 1 << 24;
    d.limits.maximum_generated_tokens = 1 << 20;
    let d = Arc::new(d);
    let tools = [tool(
        "read",
        json!({"type": "object", "properties": {"p": {"type": "object"}}}),
    )];
    let text = |markers: usize| {
        format!(
            "plan</think><tool_call><function=read><parameter=p>{{\"x\":\"{}\"}}</parameter></function></tool_call>",
            "</function>".repeat(markers)
        )
    };
    let arguments =
        |markers: usize| format!(r#"{{"p": {{"x": "{}"}}}}"#, "</function>".repeat(markers));
    // A tenth of a second at most on an idle host, but a loaded CI host can
    // take ten times that, so the cost is judged by growth: four times the
    // closes must take well under sixteen times as long, which walking the
    // arguments again for each close took (4 s whole and 2.6 s in 64-byte
    // tokens at the larger sizes; one-byte tokens took minutes). Each size
    // takes its fastest of three runs, and the smaller run counts as at
    // least 20 ms so timer noise on a fast run cannot fail the ratio.
    let fastest = |run: &dyn Fn()| {
        (0..3)
            .map(|_| {
                let started = std::time::Instant::now();
                run();
                started.elapsed()
            })
            .min()
            .unwrap()
    };
    let linear = |what: &str, small: std::time::Duration, large: std::time::Duration| {
        let floor = std::time::Duration::from_millis(20);
        assert!(
            large < 8 * small.max(floor),
            "{what}: four times the closes took {large:?} against {small:?}"
        );
        assert!(
            large < std::time::Duration::from_secs(30),
            "{what} took {large:?}"
        );
    };

    let whole = |markers: usize| {
        let whole = format!("{}{STOP}", text(markers));
        let (completion, _) = decode(&d, &tools, &[whole.as_bytes()], false).unwrap();
        assert_eq!(completion.tool_calls[0].arguments, arguments(markers));
    };
    linear(
        "one token",
        fastest(&|| whole(4_000)),
        fastest(&|| whole(16_000)),
    );

    for (markers, size) in [(1_600, 64), (16_000, 1)] {
        let streamed = |markers: usize| {
            let streamed =
                stream_engine_tokens(&d, &tools, &tokens(text(markers).as_bytes(), size));
            assert_eq!(streamed, arguments(markers));
        };
        linear(
            &format!("{size}-byte tokens"),
            fastest(&|| streamed(markers / 4)),
            fastest(&|| streamed(markers)),
        );
    }
}

/// A region that names several calls streams each as it closes, whatever the
/// token boundaries.
#[test]
fn a_region_with_several_calls_streams_each() {
    let first = call("read", &[("filter", r#"{"a": "</function>"}"#)]);
    let second = call("read", &[("text", "b"), ("list", "[1]")]);
    let text = format!(
        "plan</think>{}\n{}after{STOP}",
        first.strip_suffix("</tool_call>").unwrap(),
        second.strip_prefix("<tool_call>\n").unwrap(),
    );
    check_every_split(
        &tools(),
        &text,
        "after",
        &[
            ("read", r#"{"filter": {"a": "</function>"}}"#),
            ("read", r#"{"text": "b", "list": [1]}"#),
        ],
    );
}

/// Bytes that are not UTF-8 inside a region decode with replacement
/// characters, whatever the token boundaries.
#[test]
fn invalid_bytes_inside_a_region_decode_as_replacements() {
    let d = Arc::new(descriptor());
    let mut text = b"plan</think>".to_vec();
    text.extend_from_slice(call("read", &[("text", "a\u{1}b"), ("list", "[1]")]).as_bytes());
    text.extend_from_slice(b"after[EOS]");
    let invalid = text.iter().position(|byte| *byte == 1).unwrap();
    text[invalid] = 0xff;
    let expected = r#"{"text": "a�b", "list": [1]}"#;
    for size in [usize::MAX, 1, 2, 3, 5] {
        for stream in [false, true] {
            let (completion, streamed) =
                decode(&d, &tools(), &tokens(&text, size), stream).unwrap();
            assert_eq!(completion.tool_calls[0].arguments, expected, "{size}");
            if stream {
                assert_eq!(
                    streamed,
                    [("read".to_owned(), expected.to_owned())],
                    "{size}"
                );
            }
        }
    }
}
