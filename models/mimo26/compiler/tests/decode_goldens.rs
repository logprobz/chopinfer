use runtime_frontend::test_support::decode::{
    Calls, End, assert_call_endings, assert_call_splits, assert_no_calls, deliver, deliver_ending,
    owned, tool,
};
use std::sync::Arc;

use model_mimo26::frontend::descriptor;
use runtime_frontend::test_support::documents::PairedDecoderV1;
use runtime_frontend::{
    AssistantCompletionV1, CompletionFinishV1, CompletionUpdateV1, DecoderLimitsV1,
    GeneratedTokenV1, PromptModeV1, parse_completion_v1,
};
use runtime_server::{FrontendJsonV1, FrontendToolV1};

fn decode_chunks(
    text: &str,
    mode: PromptModeV1,
    chunks: &[&[u8]],
    tools: &[FrontendToolV1],
) -> (AssistantCompletionV1, String, String, CompletionFinishV1) {
    let descriptor = Arc::new(descriptor());
    let limits = DecoderLimitsV1::declared(&descriptor);
    let mut decoder = PairedDecoderV1::new(descriptor, mode, limits);
    decoder.set_tool_schemas(tools);
    let mut reasoning = String::new();
    let mut content = String::new();
    let mut finished = None;
    for (ordinal, bytes) in chunks.iter().enumerate() {
        let updates = decoder
            .push(GeneratedTokenV1::new(ordinal as u64, 42, bytes.to_vec()))
            .unwrap_or_else(|error| panic!("decode {text:?} at chunk {ordinal}: {error}"));
        for update in updates {
            match update {
                CompletionUpdateV1::Reasoning(span) => reasoning.push_str(&span.text),
                CompletionUpdateV1::Content(span) => content.push_str(&span.text),
                CompletionUpdateV1::Finished(end) => {
                    finished = Some((end.completion.unwrap(), end.reason))
                }
                CompletionUpdateV1::ToolCalls(_) | CompletionUpdateV1::ToolCallDeltas(_) => {}
            }
        }
    }
    let (completion, reason) = finished.expect("completion terminated");
    (completion, reasoning, content, reason)
}

/// Calls parse as the reference parser reads them, not only as rendered:
/// newlines around the wrappers are optional, text between wrappers is
/// skipped, a parameter may end where the next one opens, and text may
/// follow a call. MiMo's parameter markers write no newlines around a value
/// (template line 52), so a value keeps every byte between them.
#[test]
fn tool_calls_parse_with_the_reference_whitespace_tolerance() {
    let tools = [tool("ping", r#"{"s":{"type":"string"}}"#)];
    let x = r#"{"s": "x"}"#;
    let cases: [(&str, &str, Calls<'_>); 5] = [
        (
            "<tool_call>\n<function=ping>\n<parameter=s>\nx\n</parameter>\n</function>\n</tool_call><|im_end|>",
            "",
            &[("ping", r#"{"s": "\nx\n"}"#)],
        ),
        (
            "<tool_call><function=ping><parameter=s>x</parameter></function></tool_call> trailing<|im_end|>",
            " trailing",
            &[("ping", x)],
        ),
        (
            "<tool_call><function=ping><parameter=s>x</parameter></function></tool_call>\n\n<tool_call><function=ping></function></tool_call>\n<|im_end|>",
            "",
            &[("ping", x), ("ping", "{}")],
        ),
        (
            "<function=ping><parameter=s>x<parameter=t>y</parameter></function><|im_end|>",
            "",
            &[("ping", r#"{"s": "x", "t": "y"}"#)],
        ),
        (
            "<tool_call><function=ping><parameter=s>\n\nx\n\n</parameter></function></tool_call><|im_end|>",
            "",
            &[("ping", r#"{"s": "\n\nx\n\n"}"#)],
        ),
    ];
    assert_call_splits(descriptor(), &cases, &tools);
}

#[test]
fn reasoning_and_tool_call_cases_match_the_declared_grammar() {
    let declaration = descriptor();
    let cases = [
        (
            "<think>plan</think>answer<|im_end|>",
            PromptModeV1::Thinking,
            Some("plan"),
            "answer",
            0,
        ),
        (
            "plan</think>answer<|im_end|>",
            PromptModeV1::Thinking,
            Some("plan"),
            "answer",
            0,
        ),
        (
            "<think></think>answer<|im_end|>",
            PromptModeV1::Thinking,
            Some(""),
            "answer",
            0,
        ),
        (
            "<think>plan</think></think>answer<|im_end|>",
            PromptModeV1::Thinking,
            Some("plan"),
            "answer",
            0,
        ),
        (
            "<think>literal</think>answer<|im_end|>",
            PromptModeV1::Chat,
            None,
            "<think>literal</think>answer",
            0,
        ),
        ("plain<|endoftext|>", PromptModeV1::Chat, None, "plain", 0),
        (
            "plain<|mimo_audio_eod|>",
            PromptModeV1::Chat,
            None,
            "plain",
            0,
        ),
        (
            "<think>reason<tool_call><function=ping></function></tool_call><|im_end|>",
            PromptModeV1::Thinking,
            Some("reason"),
            "",
            1,
        ),
        (
            "before<tool_call><function=ping></function></tool_call><|im_end|>",
            PromptModeV1::Chat,
            None,
            "before",
            1,
        ),
        (
            "<tool_call><function=ping></function></tool_call><tool_call><function=ping></function></tool_call><|im_end|>",
            PromptModeV1::Chat,
            None,
            "",
            2,
        ),
        (
            "before<function=ping></function><|im_end|>",
            PromptModeV1::Chat,
            None,
            "before",
            1,
        ),
        (
            "<tool_call><function=ping></function><tool_call><function=ping></function></tool_call><|im_end|>",
            PromptModeV1::Chat,
            None,
            "",
            2,
        ),
    ];
    for (text, mode, expected_reasoning, expected_content, calls) in cases {
        let parsed = parse_completion_v1(&declaration, text, mode)
            .unwrap_or_else(|error| panic!("parse {text:?}: {error}"));
        assert_eq!(
            parsed.reasoning_content.as_deref(),
            expected_reasoning,
            "{text}"
        );
        assert_eq!(parsed.content, expected_content, "{text}");
        assert_eq!(parsed.tool_calls.len(), calls, "{text}");
        let (streamed, reasoning, content, reason) =
            decode_chunks(text, mode, &[text.as_bytes()], &[]);
        assert_eq!(streamed, parsed, "{text}");
        assert_eq!(reasoning, expected_reasoning.unwrap_or_default(), "{text}");
        assert_eq!(content, expected_content, "{text}");
        assert_eq!(
            reason,
            if calls == 0 {
                CompletionFinishV1::Stop
            } else {
                CompletionFinishV1::ToolCalls
            }
        );
    }
}

#[test]
fn tool_parameters_use_the_request_schema() {
    let tools = [tool(
        "lookup",
        r#"{"text":{"type":"string"},"count":{"type":"integer"},"enabled":{"type":"boolean"},"ratio":{"type":"number"},"filters":{"type":"object"},"items":{"type":"array"},"missing":{"type":"null"}}"#,
    )];
    let text = "<tool_call><function=lookup><parameter=text>42</parameter><parameter=count>2</parameter><parameter=enabled>true</parameter><parameter=ratio>0.5</parameter><parameter=filters>{\"lang\": \"fr\"}</parameter><parameter=items>[1, 2]</parameter><parameter=missing>null</parameter></function></tool_call><|im_end|>";
    let (parsed, _, _, reason) =
        decode_chunks(text, PromptModeV1::Chat, &[text.as_bytes()], &tools);
    assert_eq!(reason, CompletionFinishV1::ToolCalls);
    assert_eq!(parsed.tool_calls.len(), 1);
    let arguments: FrontendJsonV1 = serde_json::from_str(&parsed.tool_calls[0].arguments).unwrap();
    let expected: FrontendJsonV1 = serde_json::from_str(r#"{"text":"42","count":2,"enabled":true,"ratio":0.5,"filters":{"lang":"fr"},"items":[1,2],"missing":null}"#).unwrap();
    assert_eq!(arguments, expected);
}

#[test]
fn streaming_handles_every_byte_split_inside_markers_and_utf8() {
    let text = "<think>Café</think>Hi<tool_call><function=ping><parameter=text>é</parameter></function></tool_call><|im_end|>";
    let tools = [tool("ping", r#"{"text":{"type":"string"}}"#)];
    let (expected, _, _, _) =
        decode_chunks(text, PromptModeV1::Thinking, &[text.as_bytes()], &tools);
    for split in 1..text.len() {
        let (parsed, reasoning, content, reason) = decode_chunks(
            text,
            PromptModeV1::Thinking,
            &[&text.as_bytes()[..split], &text.as_bytes()[split..]],
            &tools,
        );
        assert_eq!(parsed, expected, "split at {split}");
        assert_eq!(reasoning, "Café", "split at {split}");
        assert_eq!(content, "Hi", "split at {split}");
        assert_eq!(reason, CompletionFinishV1::ToolCalls);
    }
}

#[test]
fn tool_arguments_wait_for_the_function_close() {
    let declaration = Arc::new(descriptor());
    let mut decoder = PairedDecoderV1::new(
        declaration.clone(),
        PromptModeV1::Chat,
        DecoderLimitsV1::declared(&declaration),
    );
    decoder.set_decoding_options(
        runtime_server::DecodingOptions {
            stream_tool_calls: true,
            ..Default::default()
        },
        false,
    );
    decoder.set_tool_schemas(&[tool(
        "lookup",
        r#"{"city":{"type":"string"},"count":{"type":"integer"}}"#,
    )]);
    let chunks = [
        "<tool_call><function=lookup><parameter=city>Pa",
        "ris",
        "</parameter><parameter=count>1",
        "2",
        "</parameter></function></tool_call><|im_end|>",
    ];
    let mut arguments = String::new();
    let mut names = 0;
    for (ordinal, chunk) in chunks.iter().enumerate() {
        let updates = decoder
            .push(GeneratedTokenV1::new(
                ordinal as u64,
                42,
                chunk.as_bytes().to_vec(),
            ))
            .unwrap();
        for update in updates {
            if let CompletionUpdateV1::ToolCallDeltas(deltas) = update {
                for delta in deltas {
                    assert_eq!(delta.index, 0);
                    names += usize::from(delta.name.is_some());
                    arguments.push_str(&delta.arguments);
                }
            }
        }
        if ordinal < chunks.len() - 1 {
            assert_eq!(names, 1);
            assert!(arguments.is_empty());
        }
    }
    assert_eq!(names, 1);
    assert_eq!(arguments, r#"{"city": "Paris", "count": 12}"#);
}

/// A generation that ends inside or right after a call, without its end
/// token, finishes the call as the reference parser does when generation
/// ends: both delivery modes keep only the complete parameters.
#[test]
fn a_call_cut_short_finishes_in_both_stream_modes() {
    let tools = [tool(
        "ping",
        r#"{"s":{"type":"string"},"n":{"type":"integer"}}"#,
    )];
    let x = r#"{"s": "x"}"#;
    let cases: [(&str, Calls<'_>); 5] = [
        (
            "<tool_call><function=ping></function></tool_call>",
            &[("ping", "{}")],
        ),
        (
            "<tool_call><function=ping><parameter=s>x</parameter></function>",
            &[("ping", x)],
        ),
        (
            "<tool_call><function=ping><parameter=s>x",
            &[("ping", "{}")],
        ),
        (
            "<tool_call><function=ping><parameter=n>12",
            &[("ping", "{}")],
        ),
        ("<tool_call><function=pi", &[]),
    ];
    assert_call_endings(descriptor(), &cases, &tools);

    // A client stop inside a call ends it the same way, and bytes that may
    // still become the stop are not sent as argument text.
    for stream in [false, true] {
        let delivered = deliver_ending(
            descriptor(),
            PromptModeV1::Chat,
            &[
                b"<tool_call><function=ping><parameter=s>xE".as_slice(),
                b"ND",
            ],
            stream,
            &tools,
            &["END".to_owned()],
            End::Token,
        )
        .unwrap_or_else(|error| panic!("client stop, stream {stream}: {error}"));
        let expected = owned(&[("ping", "{}")]);
        assert_eq!(delivered.calls, expected, "stream {stream}");
        if stream {
            assert_eq!(delivered.streamed, expected);
        }
        assert_eq!(delivered.reason, Some(CompletionFinishV1::ToolCalls));
    }

    // Reasoning cut short is unchanged.
    for stream in [false, true] {
        let delivered = deliver_ending(
            descriptor(),
            PromptModeV1::Thinking,
            &[b"<think>plan"],
            stream,
            &tools,
            &[],
            End::Length,
        )
        .unwrap();
        assert_eq!(delivered.reasoning, "plan");
        assert!(delivered.calls.is_empty());
        assert_eq!(delivered.reason, Some(CompletionFinishV1::Length));
    }
}

/// Inside reasoning, a repeated reasoning open is markup rather than
/// reasoning text, as the reference parser reads it, whichever chunk it
/// arrives in. After reasoning ends the same bytes are answer text.
#[test]
fn repeated_reasoning_opens_are_not_reasoning_text() {
    let cases = [
        (
            "<think>first<think>second</think>answer<|im_end|>",
            "firstsecond",
            "answer",
        ),
        (
            "first<think><think>second</think>answer<|im_end|>",
            "firstsecond",
            "answer",
        ),
        ("plan</think>a<think>b<|im_end|>", "plan", "a<think>b"),
    ];
    for (text, reasoning, content) in cases {
        let whole = parse_completion_v1(&descriptor(), text, PromptModeV1::Thinking)
            .unwrap_or_else(|error| panic!("parse {text:?}: {error}"));
        assert_eq!(
            whole.reasoning_content.as_deref(),
            Some(reasoning),
            "{text:?}"
        );
        assert_eq!(whole.content, content, "{text:?}");
        for split in 1..text.len() {
            let bytes = text.as_bytes();
            let delivered = deliver(
                descriptor(),
                PromptModeV1::Thinking,
                &[&bytes[..split], &bytes[split..]],
                false,
                &[],
            )
            .unwrap_or_else(|error| panic!("{text:?} split {split}: {error}"));
            assert_eq!(delivered.reasoning, reasoning, "{text:?} split {split}");
            assert_eq!(delivered.content, content, "{text:?} split {split}");
        }
    }
    // A length finish inside a partial open keeps its bytes as reasoning.
    let delivered = deliver_ending(
        descriptor(),
        PromptModeV1::Thinking,
        &[b"<think>plan<thi"],
        false,
        &[],
        &[],
        End::Length,
    )
    .unwrap();
    assert_eq!(delivered.reasoning, "plan<thi");
    assert_eq!(delivered.reason, Some(CompletionFinishV1::Length));
}

/// A request whose tool choice leaves no function to call reads tool-shaped
/// output as no call, the way the reference suppresses tool extraction: the
/// markup is consumed, the text around it is the answer, and nothing names a
/// function, whatever the name.
#[test]
fn tool_shaped_output_is_no_call_when_the_request_calls_none() {
    let tools = [tool("ping", r#"{"s":{"type":"string"}}"#)];
    let cases = [
        (
            "<tool_call><function=ping><parameter=s>x</parameter></function></tool_call><|im_end|>",
            End::Token,
            "",
            CompletionFinishV1::Stop,
        ),
        (
            "before<tool_call><function=ping><parameter=s>x</parameter></function></tool_call>after<|im_end|>",
            End::Token,
            "beforeafter",
            CompletionFinishV1::Stop,
        ),
        (
            "<tool_call><function=unknown><parameter=s>x</parameter></function></tool_call><|im_end|>",
            End::Token,
            "",
            CompletionFinishV1::Stop,
        ),
        (
            "<tool_call><function=ping><parameter=s>x",
            End::Length,
            "",
            CompletionFinishV1::Length,
        ),
    ];
    assert_no_calls(descriptor(), &cases, &tools);
}
