//! What a client receives when MiMo quotes its own chat-template markers.
//!
//! `<think>`, `</think>`, `<tool_call>` and `</tool_call>` are added tokens
//! that the tokenizer does not mark special, so any text that spells one
//! encodes to its single identifier, and the decoder keeps its bytes under
//! `skip_special_tokens`. The declared parser then reads those bytes as
//! structure wherever its state expects them, as the reference parser reads
//! the identifiers: a `<think>` inside reasoning is dropped, a `</think>`
//! ends reasoning, and a `<tool_call>` opens a tool region whose text before
//! a function open is skipped. Inside a parameter value every one of them is
//! kept, and so are the newlines MiMo's parameter markers do not write.
//!
//! The completions below reconstruct two served calls whose reasoning ended
//! where the model quoted these markers.

use std::path::Path;
use std::sync::Arc;

use model_mimo26::frontend::descriptor;
use model_mimo26::tokenizer::load_tokenizer;
use runtime_frontend::test_support::decode::tool;
use runtime_frontend::test_support::documents::PairedDecoderV1;
use runtime_frontend::{
    CompletionFinishV1, CompletionUpdateV1, DecoderLimitsV1, GeneratedTokenV1, PromptModeV1,
};
use runtime_server::{DecodingOptions, FrontendToolV1};
use runtime_tokenizer::Tokenizer;

const MARKERS: [(&str, u32); 4] = [
    ("<think>", 151_667),
    ("</think>", 151_668),
    ("<tool_call>", 151_657),
    ("</tool_call>", 151_658),
];

fn tokenizer() -> Option<Tokenizer> {
    let root = std::env::var("CHOPINFER_MIMO_CHECKPOINT")
        .or_else(|_| std::env::var("CHOPINFER_MIMO_TOKENIZER_DIR"))
        .ok()?;
    Some(load_tokenizer(Path::new(&root)).unwrap())
}

/// The model's completion as the tokenizer encodes its text, the way the
/// reference tokenizer encodes training text: every added token matches.
fn encode(t: &Tokenizer, text: &str) -> Vec<u32> {
    let d = descriptor();
    let added: Vec<&str> = d
        .vocabulary
        .prompt_special_tokens
        .iter()
        .map(String::as_str)
        .collect();
    t.encode(text, &added).unwrap()
}

/// `marker` in ordinary sub-word tokens: its bracket, then the rest.
fn spell(t: &Tokenizer, marker: &str) -> Vec<u32> {
    let split = if marker.starts_with("</") { 2 } else { 1 };
    let mut ids = t.encode(&marker[..split], &[]).unwrap();
    ids.extend(t.encode(&marker[split..], &[]).unwrap());
    ids
}

fn tools() -> Vec<FrontendToolV1> {
    vec![
        tool(
            "write_file",
            r#"{"path":{"type":"string"},"content":{"type":"string"}}"#,
        ),
        tool("read_file", r#"{"path":{"type":"string"}}"#),
    ]
}

#[derive(Debug, Default)]
struct Received {
    reasoning: String,
    content: String,
    calls: Vec<(String, String)>,
    /// The calls as a streamed client assembles them from their deltas.
    streamed: Vec<(String, String)>,
    reason: Option<CompletionFinishV1>,
}

/// Streams `ids` as the engine sampled them, with the serving defaults a
/// streamed chat completion in thinking mode decodes with.
fn serve(t: &Tokenizer, ids: &[u32], options: DecodingOptions) -> Received {
    let d = Arc::new(descriptor());
    let mut decoder = PairedDecoderV1::new(
        Arc::clone(&d),
        PromptModeV1::Thinking,
        DecoderLimitsV1::declared(&d),
    );
    let stream = options.stream_tool_calls;
    decoder.set_decoding_options(options, false);
    decoder.set_tool_schemas(&tools());
    let mut received = Received::default();
    for (ordinal, id) in ids.iter().enumerate() {
        let token =
            GeneratedTokenV1::new(ordinal as u64, *id, t.token_bytes(*id).unwrap().to_vec())
                .with_special(t.token_is_special(*id).unwrap());
        for update in decoder.push_engine_token(token).unwrap() {
            match update {
                CompletionUpdateV1::Reasoning(span) => received.reasoning.push_str(&span.text),
                CompletionUpdateV1::Content(span) => received.content.push_str(&span.text),
                CompletionUpdateV1::ToolCalls(group) => received.calls.extend(
                    group
                        .calls
                        .into_iter()
                        .map(|call| (call.name, call.arguments)),
                ),
                CompletionUpdateV1::Finished(end) => received.reason = Some(end.reason),
                CompletionUpdateV1::ToolCallDeltas(deltas) => {
                    for delta in deltas {
                        if delta.index == received.streamed.len() {
                            received.streamed.push((String::new(), String::new()));
                        }
                        let (name, arguments) = &mut received.streamed[delta.index];
                        name.push_str(delta.name.as_deref().unwrap_or_default());
                        arguments.push_str(&delta.arguments);
                    }
                }
            }
        }
        if received.reason.is_some() {
            assert_eq!(ordinal + 1, ids.len(), "the completion ended early");
        }
    }
    // A streamed client assembles the same calls the groups deliver.
    let expected = if stream { &received.calls[..] } else { &[] };
    assert_eq!(received.streamed, expected);
    received
}

fn streamed() -> DecodingOptions {
    DecodingOptions {
        stream_tool_calls: true,
        ..DecodingOptions::default()
    }
}

#[test]
fn quoted_markers_encode_to_their_added_token_ids() {
    let Some(t) = tokenizer() else { return };
    for (marker, id) in MARKERS {
        assert!(!t.token_is_special(id).unwrap(), "{marker}");
        assert_eq!(t.token_bytes(id).unwrap(), marker.as_bytes());
        let quoted = encode(&t, &format!("the marker `{marker}`, quoted"));
        assert_eq!(
            quoted.iter().filter(|&&each| each == id).count(),
            1,
            "{marker}"
        );
        // Spelled in ordinary sub-words, the same bytes take several tokens.
        let spelled = spell(&t, marker);
        assert!(
            spelled.len() > 1 && !spelled.contains(&id),
            "{marker} {spelled:?}"
        );
        assert_eq!(t.decode_bytes(&spelled).unwrap(), marker.as_bytes());
    }
}

/// Call 9147578e: reasoning held three empty backtick pairs and ended in
/// "markers ``, `", the content was "`, `", and two tool calls followed.
#[test]
fn quoted_markers_end_reasoning_as_call_9147578e_received() {
    let Some(t) = tokenizer() else { return };
    let completion = "The opener `<think>` starts reasoning, so a reply that quotes `<think>` \
        must type it. The tool description says the response must contain the markers \
        `<think>`, `</think>`, `<tool_call>`. I will write them into the file as typed.</think>\n\n\
        <tool_call>\n<function=write_file>\n<parameter=path>\nnotes.md\n</parameter>\n\
        <parameter=content>\nOpen with <think>, close with </think>, then call \
        <tool_call>...</tool_call>.\n</parameter>\n</function>\n</tool_call>\n\
        <tool_call>\n<function=read_file>\n<parameter=path>\nnotes.md\n</parameter>\n\
        </function>\n</tool_call><|im_end|>";
    let ids = encode(&t, completion);
    for (marker, id) in MARKERS {
        assert!(ids.contains(&id), "{marker}");
    }
    let received = serve(&t, &ids, streamed());
    assert_eq!(
        received.reasoning,
        "The opener `` starts reasoning, so a reply that quotes `` must type it. The tool \
         description says the response must contain the markers ``, `"
    );
    assert_eq!(received.reasoning.matches("``").count(), 3);
    // The quoted `<tool_call>` opened a region, which skipped the rest of
    // the reasoning and the real close up to the first function open.
    assert_eq!(received.content, "`, `");
    assert_eq!(
        received.calls,
        [
            (
                "write_file".to_owned(),
                r#"{"path": "\nnotes.md\n", "content": "\nOpen with <think>, close with </think>, then call <tool_call>...</tool_call>.\n"}"#
                    .to_owned()
            ),
            ("read_file".to_owned(), r#"{"path": "\nnotes.md\n"}"#.to_owned()),
        ]
    );
    assert_eq!(received.reason, Some(CompletionFinishV1::ToolCalls));
}

/// Call c26e5660: reasoning ended in "The same applies to " with a double
/// space where `<think>` was, the content was ", ", and one tool call
/// followed. Had the model also quoted `</tool_call>`, that would have
/// closed the region and returned the prose after it as content.
#[test]
fn quoted_markers_end_reasoning_as_call_c26e5660_received() {
    let Some(t) = tokenizer() else { return };
    let head = "So the client escapes the leading bracket: <think> arrives as \"<\", a \
        zero-width space, then \"think>\". The same applies to </think>, <tool_call>";
    let tail = ". I will read the file next.</think>\n\n<tool_call>\n<function=read_file>\n\
        <parameter=path>\nnotes.md\n</parameter>\n</function>\n</tool_call><|im_end|>";
    for (quoted_close, content) in [
        ("", ", "),
        (" and </tool_call>", ", . I will read the file next."),
    ] {
        let completion = format!("{head}{quoted_close}{tail}");
        let received = serve(&t, &encode(&t, &completion), streamed());
        assert_eq!(
            received.reasoning,
            "So the client escapes the leading bracket:  arrives as \"<\", a zero-width space, \
             then \"think>\". The same applies to "
        );
        assert_eq!(received.content, content);
        assert_eq!(
            received.calls,
            [(
                "read_file".to_owned(),
                r#"{"path": "\nnotes.md\n"}"#.to_owned()
            )]
        );
        assert_eq!(received.reason, Some(CompletionFinishV1::ToolCalls));
    }
}

/// The parser reads marker bytes, not identifiers: the same text in
/// sub-words is read as the marker too, and only a spelling that breaks the
/// bytes, such as a zero-width space after the bracket, stays reasoning
/// text. The reference parser differs here: once a completion carries
/// token ids, it reads a marker only from its id and keeps a sub-word
/// spelling as text, so for these ids it gives "quote `<think>` and
/// `</think>` here" as reasoning and no content.
#[test]
fn quoted_markers_parse_by_their_bytes() {
    let Some(t) = tokenizer() else { return };
    let mut ids = Vec::new();
    for piece in ["quote `", "<think>", "` and `", "</think>", "` here"] {
        if piece.starts_with('<') {
            ids.extend(spell(&t, piece));
        } else {
            ids.extend(t.encode(piece, &[]).unwrap());
        }
    }
    ids.push(descriptor().vocabulary.eos_token_id);
    for (marker, id) in MARKERS {
        assert!(!ids.contains(&id), "{marker}");
    }
    let received = serve(&t, &ids, streamed());
    assert_eq!(received.reasoning, "quote `` and `");
    assert_eq!(received.content, "` here");

    let escaped = "quote `<\u{200b}think>` and `<\u{200b}/think>` here</think>done<|im_end|>";
    let received = serve(&t, &encode(&t, escaped), streamed());
    assert_eq!(
        received.reasoning,
        "quote `<\u{200b}think>` and `<\u{200b}/think>` here"
    );
    assert_eq!(received.content, "done");
}

/// Inside a parameter value every marker identifier keeps its text, and
/// only the `<|...|>` tokens the tokenizer marks special are dropped under
/// the default `skip_special_tokens`, as the reference decoder drops them.
#[test]
fn marker_ids_inside_tool_arguments_keep_their_text() {
    let Some(t) = tokenizer() else { return };
    let value = "a <think> b </think> c <tool_call> d </tool_call> e <|im_start|> f";
    let completion = format!(
        "plan</think><tool_call>\n<function=write_file>\n<parameter=path>\nx\n</parameter>\n\
         <parameter=content>\n{value}\n</parameter>\n</function>\n</tool_call><|im_end|>"
    );
    let ids = encode(&t, &completion);
    for (skip, kept) in [
        (
            true,
            "\na <think> b </think> c <tool_call> d </tool_call> e  f\n",
        ),
        (false, &*format!("\n{value}\n")),
    ] {
        for stream in [false, true] {
            let options = DecodingOptions {
                stream_tool_calls: stream,
                skip_special_tokens: skip,
                ..DecodingOptions::default()
            };
            let received = serve(&t, &ids, options);
            assert_eq!(received.reasoning, "plan");
            assert_eq!(received.content, "");
            let arguments = serde_json::json!({"path": "\nx\n", "content": kept});
            let (name, actual) = &received.calls[0];
            assert_eq!(name, "write_file");
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(actual).unwrap(),
                arguments,
                "skip={skip} stream={stream}"
            );
        }
    }
}
