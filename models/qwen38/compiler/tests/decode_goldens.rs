//! Completions decoded through this model's declared call layout, whose
//! delimiters carry the template's newlines.

use runtime_frontend::test_support::decode::{
    Calls, End, assert_call_endings, assert_call_splits, assert_no_calls, deliver, deliver_ending,
    owned, tool,
};

use frontend_format::parse_completion_v1;
use model_qwen38::frontend::descriptor;
use runtime_frontend::{CompletionFinishV1, PromptModeV1};

/// Calls parse as the reference parser reads them, not only as rendered:
/// template newlines are optional, text between wrappers is skipped, a
/// parameter may end where the next one opens, and text may follow a call.
#[test]
fn tool_calls_parse_with_the_reference_whitespace_tolerance() {
    let tools = [tool("ping", r#"{"s":{"type":"string"}}"#)];
    let x = r#"{"s": "x"}"#;
    let cases: [(&str, &str, Calls<'_>); 5] = [
        (
            "<tool_call><function=ping><parameter=s>x</parameter></function></tool_call><|im_end|>",
            "",
            &[("ping", x)],
        ),
        (
            "<tool_call>\n<function=ping>\n<parameter=s>\nx\n</parameter>\n</function>\n</tool_call> trailing<|im_end|>",
            " trailing",
            &[("ping", x)],
        ),
        (
            "<tool_call>\n<function=ping>\n<parameter=s>\nx\n</parameter>\n</function>\n</tool_call>\n\n<tool_call>\n<function=ping>\n</function>\n</tool_call><|im_end|>",
            "",
            &[("ping", x), ("ping", "{}")],
        ),
        (
            "<function=ping>\n<parameter=s>\nx\n<parameter=t>\ny\n</parameter>\n</function><|im_end|>",
            "",
            &[("ping", r#"{"s": "x", "t": "y"}"#)],
        ),
        (
            "<tool_call>\n<function=ping>\n<parameter=s>\n\nx\n\n</parameter>\n</function>\n</tool_call><|im_end|>",
            "",
            &[("ping", r#"{"s": "\nx\n"}"#)],
        ),
    ];
    assert_call_splits(descriptor(), &cases, &tools);
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
            "<tool_call>\n<function=ping>\n</function>\n</tool_call>",
            &[("ping", "{}")],
        ),
        (
            "<tool_call>\n<function=ping>\n<parameter=s>\nx\n</parameter>\n</function>\n",
            &[("ping", x)],
        ),
        (
            "<tool_call>\n<function=ping>\n<parameter=s>\nx\n",
            &[("ping", "{}")],
        ),
        (
            "<tool_call>\n<function=ping>\n<parameter=n>\n12\n",
            &[("ping", "{}")],
        ),
        ("<tool_call>\n<function=pi", &[]),
    ];
    assert_call_endings(descriptor(), &cases, &tools);

    // A client stop inside a call ends it the same way, and bytes that may
    // still become the stop are not sent as argument text.
    for stream in [false, true] {
        let delivered = deliver_ending(
            descriptor(),
            PromptModeV1::Chat,
            &[
                b"<tool_call>\n<function=ping>\n<parameter=s>\nxE".as_slice(),
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
        let format = frontend_format::FormatDescriptorV1::new(descriptor()).unwrap();
        let whole = parse_completion_v1(&format, text, PromptModeV1::Thinking)
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
            "<tool_call>\n<function=ping>\n<parameter=s>\nx\n</parameter>\n</function>\n</tool_call><|im_end|>",
            End::Token,
            "",
            CompletionFinishV1::Stop,
        ),
        (
            "before<tool_call>\n<function=ping>\n<parameter=s>\nx\n</parameter>\n</function>\n</tool_call>after<|im_end|>",
            End::Token,
            "beforeafter",
            CompletionFinishV1::Stop,
        ),
        (
            "<tool_call>\n<function=unknown>\n<parameter=s>\nx\n</parameter>\n</function>\n</tool_call><|im_end|>",
            End::Token,
            "",
            CompletionFinishV1::Stop,
        ),
        (
            "<tool_call>\n<function=ping>\n<parameter=s>\nx",
            End::Length,
            "",
            CompletionFinishV1::Length,
        ),
    ];
    assert_no_calls(descriptor(), &cases, &tools);
}
