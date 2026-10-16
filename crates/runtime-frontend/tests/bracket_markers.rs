#![forbid(unsafe_code)]
//! Decoding a completion in markers that share no character with any other
//! fixture's.
use frontend_format::test_support::edit_format;
use frontend_format::test_support::fixtures::KEY_VALUE;
use std::sync::Arc;

use runtime_frontend::{
    CompletionDecoderV1, CompletionUpdateV1, DecoderLimitsV1, FrontendDescriptorV1,
    GeneratedTokenV1, PromptModeV1,
};
use runtime_server::FrontendToolV1;

/// The key-value fixture's host sections with a function-parameter parser
/// over bracket markers, which no reserved marker covers.
fn descriptor() -> FrontendDescriptorV1 {
    let mut d = FrontendDescriptorV1::from_component_bytes(KEY_VALUE.as_bytes()).unwrap();
    d.reasoning.end_markers = Vec::new();
    d.format = serde_json::json!({"parser": {
        "tool_grammar": "xml-function-parameters-v1", "json_encoding": "spaced-json-v1",
        "reasoning_open": "t[", "reasoning_close": "]t", "tool_call_open": "call[",
        "tool_call_close": "]call", "function_open": "fn[", "function_name_close": "]",
        "function_close": "]fn", "parameter_open": "p[", "parameter_name_close": "]",
        "parameter_close": "]p",
        "reserved_markers": ["[BOS]", "[EOS]", "[U]", "[A]", "<t>", "</t>", "<r>", "</r>", "<calls>"],
        "turn_open": ["[A]", "[U]"], "assistant_header": "[A]"
    }});
    d.validate().unwrap();
    d
}

/// `d` with its parser section changed by `edit`.
fn with_parser(
    d: &FrontendDescriptorV1,
    edit: impl FnOnce(&mut frontend_format::FrontendParserDeclarationV1),
) -> FrontendDescriptorV1 {
    edit_format(d, |d| edit(&mut d.parser))
}

/// The content and tool-call names a completion yields, whole and streamed.
fn answers(d: &FrontendDescriptorV1, completion: &str) -> Vec<(String, usize)> {
    let whole =
        runtime_frontend::parse_completion_v1(d, completion, PromptModeV1::Thinking).unwrap();
    let mut seen = vec![(whole.content, whole.tool_calls.len())];
    let d = Arc::new(d.clone());
    for split in 1..completion.len() {
        let mut decoder = CompletionDecoderV1::new(
            Arc::clone(&d),
            PromptModeV1::Thinking,
            DecoderLimitsV1::declared(&d),
        );
        let (mut content, mut calls) = (String::new(), 0);
        for (ordinal, bytes) in [
            &completion.as_bytes()[..split],
            &completion.as_bytes()[split..],
        ]
        .into_iter()
        .enumerate()
        {
            for update in decoder
                .push(GeneratedTokenV1::new(ordinal as u64, 200, bytes.to_vec()))
                .unwrap()
            {
                match update {
                    CompletionUpdateV1::Content(part) => content.push_str(&part.text),
                    CompletionUpdateV1::ToolCalls(tools) => calls += tools.calls.len(),
                    _ => {}
                }
            }
        }
        seen.push((content, calls));
    }
    seen.dedup();
    seen
}

#[test]
fn only_a_declared_parser_drops_a_blank_answer_before_a_tool_call() {
    let mut d = descriptor();
    let call = "thought]t\n\ncall[fn[lookup]]fn]call[EOS]";
    let text = "thought]t\n\nanswer[EOS]";
    let blank = "thought]t\n\n[EOS]";
    assert_eq!(answers(&d, call), [("\n\n".to_owned(), 1)]);
    d = with_parser(&d, |parser| parser.drop_blank_tool_call_content = true);
    assert_eq!(answers(&d, call), [(String::new(), 1)]);
    assert_eq!(answers(&d, text), [("\n\nanswer".to_owned(), 0)]);
    assert_eq!(answers(&d, blank), [("\n\n".to_owned(), 0)]);
    assert_eq!(
        answers(&d, "thought]t\n\nfirst\n\ncall[fn[lookup]]fn]call[EOS]"),
        [("\n\nfirst\n\n".to_owned(), 1)]
    );
}

#[test]
fn strict_tools_accept_required_names_without_property_schemas() {
    use runtime_server::{FrontendToolChoiceV1, ServerFrontendV1};
    let frontend = runtime_frontend::BundleFrontendV1::with_export(
        descriptor(),
        &runtime_frontend::test_support::fixture_export_v1(),
    )
    .unwrap();
    let tool = FrontendToolV1 {
        name: "lookup".into(),
        description: None,
        parameters: serde_json::from_value(serde_json::json!({
            "type": "object", "required": ["query"],
            "additionalProperties": {"type": "integer", "multipleOf": 2},
            "dependentRequired": {"query": ["context"]}
        }))
        .unwrap(),
        parameters_present: true,
        strict: Some(true),
    };
    assert!(
        frontend
            .tool_constraint(&[tool], &FrontendToolChoiceV1::Required)
            .unwrap()
            .is_some()
    );
}

#[test]
fn minimum_tokens_preserves_end_marker_text_in_reasoning_and_tool_arguments() {
    for (mode, parts) in [
        (
            PromptModeV1::Thinking,
            vec!["[EO", "S]", "]tanswer", "[EOS]"],
        ),
        (
            PromptModeV1::Chat,
            vec!["call[fn[lookup]p[text]", "[EOS]", "]p]fn]call", "[EOS]"],
        ),
    ] {
        for minimum in [0, 1, 2, 3, 64] {
            let descriptor = Arc::new(descriptor());
            let mut decoder = CompletionDecoderV1::new(
                descriptor.clone(),
                mode,
                DecoderLimitsV1::declared(&descriptor),
            );
            decoder.set_minimum_new_tokens(minimum);
            let mut reasoning = String::new();
            let mut arguments = Vec::new();
            for (ordinal, part) in parts.iter().enumerate() {
                let updates = decoder
                    .push_engine_token(GeneratedTokenV1::new(
                        ordinal as u64,
                        if ordinal == 3 {
                            descriptor.vocabulary.eos_token_id
                        } else {
                            200
                        },
                        part.as_bytes().to_vec(),
                    ))
                    .unwrap();
                assert_eq!(
                    updates
                        .iter()
                        .any(|update| matches!(update, CompletionUpdateV1::Finished(_))),
                    ordinal == 3
                );
                for update in updates {
                    match update {
                        CompletionUpdateV1::Reasoning(part) => reasoning.push_str(&part.text),
                        CompletionUpdateV1::ToolCalls(calls) => {
                            arguments.extend(calls.calls.into_iter().map(|call| call.arguments))
                        }
                        _ => {}
                    }
                }
            }
            if mode == PromptModeV1::Thinking {
                assert_eq!(reasoning, "[EOS]");
            } else {
                assert_eq!(arguments, vec![r#"{"text": "[EOS]"}"#]);
            }
        }
    }
}

#[test]
fn sampled_grammar_stops_have_uniform_visibility_before_and_after_the_minimum() {
    for minimum in [0, 1, 64] {
        for include in [false, true] {
            for skip_special in [false, true] {
                let descriptor = Arc::new(descriptor());
                let mut decoder = CompletionDecoderV1::new(
                    descriptor.clone(),
                    PromptModeV1::Chat,
                    DecoderLimitsV1::declared(&descriptor),
                );
                decoder.set_minimum_new_tokens(minimum);
                decoder.set_decoding_options(
                    runtime_server::DecodingOptions {
                        include_stop_str_in_output: include,
                        skip_special_tokens: skip_special,
                        ..Default::default()
                    },
                    false,
                );
                let mut updates = decoder
                    .push_engine_token(GeneratedTokenV1::new(0, 200, b"{}".to_vec()))
                    .unwrap();
                // The engine may sample this ID early when only the grammar's
                // terminal transition remains. Its bytes are not parsed anew.
                updates.extend(
                    decoder
                        .push_engine_token(
                            GeneratedTokenV1::new(1, 1, b"[EOS]".to_vec()).with_special(true),
                        )
                        .unwrap(),
                );
                let mut delivered = String::new();
                let mut finished = Vec::new();
                for update in updates {
                    match update {
                        CompletionUpdateV1::Content(part) => delivered.push_str(&part.text),
                        CompletionUpdateV1::Finished(done) => {
                            finished.push((done.reason, done.generated_token_count))
                        }
                        _ => {}
                    }
                }
                assert_eq!(
                    delivered,
                    if include && !skip_special {
                        "{}[EOS]"
                    } else {
                        "{}"
                    },
                    "minimum={minimum}, include={include}, skip={skip_special}"
                );
                assert_eq!(
                    finished,
                    vec![(runtime_frontend::CompletionFinishV1::Stop, 2)]
                );
            }
        }
    }
}

#[test]
fn invalid_utf8_tool_regions_preserve_text_at_every_split() {
    let d = Arc::new(descriptor());
    let bytes = b"thought]tcall[fn[f]p[x]\xe4 ]p]fn]callhello[EOS]";
    for split in 1..bytes.len() {
        let mut decoder = CompletionDecoderV1::new(
            Arc::clone(&d),
            PromptModeV1::Thinking,
            DecoderLimitsV1::declared(&d),
        );
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut completion = None;
        for (ordinal, chunk) in [&bytes[..split], &bytes[split..]].into_iter().enumerate() {
            for update in decoder
                .push(GeneratedTokenV1::new(ordinal as u64, 200, chunk.to_vec()))
                .unwrap()
            {
                match update {
                    CompletionUpdateV1::Content(part) => content.push_str(&part.text),
                    CompletionUpdateV1::Reasoning(part) => reasoning.push_str(&part.text),
                    CompletionUpdateV1::Finished(done) => completion = done.completion,
                    _ => {}
                }
            }
        }
        let completion = completion.expect("completion at every split");
        assert_eq!(content, "hello", "split {split}");
        assert_eq!(reasoning, "thought");
        assert_eq!(completion.content, content);
        assert_eq!(completion.reasoning_content.as_deref(), Some("thought"));
        assert_eq!(completion.tool_calls.len(), 1);
        assert_eq!(completion.tool_calls[0].name, "f");
        assert_eq!(completion.tool_calls[0].arguments, "{\"x\": \"� \"}");
    }
}

#[test]
fn incomplete_prompt_never_echoes_or_matches_prompt_stops() {
    let d = Arc::new(descriptor());
    let mut decoder = CompletionDecoderV1::with_client_stops(
        Arc::clone(&d),
        PromptModeV1::Chat,
        DecoderLimitsV1::declared(&d),
        &["\n".into()],
    )
    .unwrap();
    decoder.set_decoding_options(Default::default(), true);
    decoder.prime_prompt_bytes(b"Q: secret\nA: \xe4").unwrap();
    let mut text = String::new();
    for update in decoder
        .push(GeneratedTokenV1::new(0, 200, b"\xb8\xad ok".to_vec()))
        .unwrap()
        .into_iter()
        .chain(decoder.finish_length().unwrap())
    {
        if let CompletionUpdateV1::Content(part) = update {
            text.push_str(&part.text);
        }
    }
    assert_eq!(text, "中 ok");
}

#[test]
fn replacement_of_prompt_context_cannot_complete_a_client_stop() {
    let d = Arc::new(descriptor());
    let mut decoder = CompletionDecoderV1::with_client_stops(
        Arc::clone(&d),
        PromptModeV1::Chat,
        DecoderLimitsV1::declared(&d),
        &["�".into()],
    )
    .unwrap();
    decoder.set_decoding_options(Default::default(), true);
    decoder.prime_prompt_bytes(&[0xe4]).unwrap();
    let updates = decoder
        .push_engine_token(GeneratedTokenV1::new(0, 200, b"A".to_vec()))
        .unwrap();
    assert!(
        !updates
            .iter()
            .any(|update| matches!(update, CompletionUpdateV1::Finished(_)))
    );
    assert!(decoder.finish_length().is_ok());
}

fn decode_parity(
    d: &Arc<FrontendDescriptorV1>,
    chunks: Vec<&[u8]>,
    stream: bool,
) -> Result<
    (
        String,
        String,
        Option<runtime_frontend::AssistantCompletionV1>,
    ),
    String,
> {
    let mut dec = CompletionDecoderV1::new(
        d.clone(),
        PromptModeV1::Thinking,
        DecoderLimitsV1::declared(d),
    );
    dec.set_decoding_options(
        runtime_server::DecodingOptions {
            stream_tool_calls: stream,
            ..Default::default()
        },
        false,
    );
    let mut deltas = std::collections::BTreeMap::<usize, String>::new();
    let mut delivered = Vec::new();
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut done = None;
    for (i, c) in chunks.into_iter().filter(|c| !c.is_empty()).enumerate() {
        for u in dec.push(GeneratedTokenV1::new(i as u64, 200, c.to_vec()))? {
            match u {
                CompletionUpdateV1::Content(x) => content += &x.text,
                CompletionUpdateV1::Reasoning(x) => reasoning += &x.text,
                CompletionUpdateV1::ToolCallDeltas(xs) => {
                    for x in xs {
                        *deltas.entry(x.index).or_default() += &x.arguments;
                    }
                }
                CompletionUpdateV1::ToolCalls(x) => delivered.extend(x.calls),
                CompletionUpdateV1::Finished(x) => done = x.completion,
            }
        }
    }
    if stream {
        for (i, c) in delivered.iter().enumerate() {
            if deltas.get(&i) != Some(&c.arguments) {
                return Err(format!("tool delta mismatch: {deltas:?} {delivered:?}"));
            }
        }
    }
    Ok((content, reasoning, done))
}
#[test]
fn corrupted_delimiters_have_chunk_independent_tool_arguments() {
    let d = Arc::new(descriptor());
    let base = b"thought]tbefore call[fn[f]p[x]value]p]fn]call after[EOS]";
    let sequences: Vec<&[u8]> = vec![
        b"\xff",
        b"\xc0\xaf",
        b"\xe4",
        b"\xe4\xb8",
        b"\xed\xa0\x80",
        b"\xf4\x90\x80\x80",
        b"\xef\xbf\xbd",
    ];
    let mut count = 0;
    let mut bad = 0;
    // Corrupt every position, including delimiters, and exercise every split.
    for pos in 0..base.len() {
        for seq in &sequences {
            let mut bytes = base[..pos].to_vec();
            bytes.extend_from_slice(seq);
            bytes.extend_from_slice(&base[pos..]);
            let reference = String::from_utf8_lossy(&bytes);
            let expected = decode_parity(&d, vec![reference.as_bytes()], false);
            for stream in [false, true] {
                for split in 0..=bytes.len() {
                    count += 1;
                    let actual = decode_parity(&d, vec![&bytes[..split], &bytes[split..]], stream);
                    if actual != expected {
                        bad += 1;
                        if bad < 12 {
                            println!(
                                "MISMATCH pos={pos} seq={seq:?} split={split} stream={stream} actual={actual:?} expected={expected:?}"
                            );
                        }
                    }
                }
                count += 1;
                let actual = decode_parity(&d, bytes.chunks(1).collect(), stream);
                if actual != expected {
                    bad += 1;
                    if bad < 12 {
                        println!(
                            "BYTE MISMATCH pos={pos} seq={seq:?} stream={stream} actual={actual:?} expected={expected:?}"
                        );
                    }
                }
            }
        }
    }
    assert_eq!(bad, 0, "{count} parity comparisons");
}

#[test]
fn repeated_parameters_do_not_rewrite_already_streamed_arguments() {
    let d = Arc::new(descriptor());
    let bytes = b"thought]tcall[fn[f]p[x]old]pp[x]new]p]fn]call[EOS]";
    let expected = decode_parity(&d, vec![bytes], false);
    assert_eq!(
        expected.as_ref().unwrap().2.as_ref().unwrap().tool_calls[0].arguments,
        r#"{"x": "new"}"#
    );
    for split in 0..=bytes.len() {
        assert_eq!(
            decode_parity(&d, vec![&bytes[..split], &bytes[split..]], true),
            expected
        );
    }
    assert_eq!(decode_parity(&d, bytes.chunks(1).collect(), true), expected);
}
