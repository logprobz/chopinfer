//! Helpers for completion golden tests over a caller-supplied declaration.
//!
//! Every completion decodes twice, calling the shared parser directly and
//! through the frontend documents, and the two must agree.
use crate::test_support::documents::PairedDecoderV1;
use crate::{
    CompletionFinishV1, CompletionUpdateV1, DecoderLimitsV1, FrontendDescriptorV1,
    GeneratedTokenV1, PromptModeV1, parse_completion_v1,
};
use runtime_server::{DecodingOptions, FrontendToolV1};
use std::sync::Arc;
pub fn tool(name: &str, properties: &str) -> FrontendToolV1 {
    FrontendToolV1 {
        name: name.to_owned(),
        description: None,
        parameters: serde_json::from_str(&format!(
            "{{\"type\":\"object\",\"properties\":{properties}}}"
        ))
        .unwrap(),
        parameters_present: true,
        strict: None,
    }
}

#[derive(Debug, Default)]
pub struct Delivered {
    pub reasoning: String,
    pub content: String,
    pub calls: Vec<(String, String)>,
    pub streamed: Vec<(String, String)>,
    pub reason: Option<CompletionFinishV1>,
}

#[derive(Clone, Copy, Debug)]
pub enum End {
    Token,
    Length,
    EngineStop,
}

pub fn decoder(
    descriptor: FrontendDescriptorV1,
    mode: PromptModeV1,
    stream: bool,
    tools: &[FrontendToolV1],
    stops: &[String],
) -> Result<PairedDecoderV1, String> {
    let declaration = Arc::new(descriptor);
    let limits = DecoderLimitsV1::declared(&declaration);
    let mut decoder = if stops.is_empty() {
        PairedDecoderV1::new(declaration, mode, limits)
    } else {
        PairedDecoderV1::with_client_stops(declaration, mode, limits, stops)?
    };
    decoder.set_decoding_options(
        DecodingOptions {
            stream_tool_calls: stream,
            ..Default::default()
        },
        false,
    );
    decoder.set_tool_schemas(tools);
    Ok(decoder)
}

pub fn run(mut decoder: PairedDecoderV1, chunks: &[&[u8]], end: End) -> Result<Delivered, String> {
    let mut updates = Vec::new();
    for (ordinal, bytes) in chunks.iter().enumerate() {
        updates.extend(decoder.push(GeneratedTokenV1::new(ordinal as u64, 42, bytes.to_vec()))?);
    }
    match end {
        End::Token => {}
        End::Length => updates.extend(decoder.finish_length()?),
        End::EngineStop => updates.extend(decoder.finish_stop()?),
    }
    let mut delivered = Delivered::default();
    for update in updates {
        match update {
            CompletionUpdateV1::Reasoning(span) => delivered.reasoning.push_str(&span.text),
            CompletionUpdateV1::Content(span) => delivered.content.push_str(&span.text),
            CompletionUpdateV1::ToolCalls(group) => delivered.calls.extend(
                group
                    .calls
                    .into_iter()
                    .map(|call| (call.name, call.arguments)),
            ),
            CompletionUpdateV1::ToolCallDeltas(deltas) => {
                for delta in deltas {
                    match delta.name {
                        Some(name) => delivered.streamed.push((name, delta.arguments)),
                        None => delivered.streamed[delta.index].1.push_str(&delta.arguments),
                    }
                }
            }
            CompletionUpdateV1::Finished(done) => delivered.reason = Some(done.reason),
        }
    }
    Ok(delivered)
}

pub fn owned(calls: Calls<'_>) -> Vec<(String, String)> {
    calls
        .iter()
        .map(|(name, arguments)| ((*name).to_owned(), (*arguments).to_owned()))
        .collect()
}

pub type Calls<'a> = &'a [(&'a str, &'a str)];

pub fn deliver(
    descriptor: FrontendDescriptorV1,
    mode: PromptModeV1,
    chunks: &[&[u8]],
    stream: bool,
    tools: &[FrontendToolV1],
) -> Result<Delivered, String> {
    deliver_ending(descriptor, mode, chunks, stream, tools, &[], End::Token)
}

pub fn deliver_ending(
    descriptor: FrontendDescriptorV1,
    mode: PromptModeV1,
    chunks: &[&[u8]],
    stream: bool,
    tools: &[FrontendToolV1],
    stops: &[String],
    end: End,
) -> Result<Delivered, String> {
    run(
        decoder(descriptor, mode, stream, tools, stops)?,
        chunks,
        end,
    )
}

/// Check every declared completion case in both streaming modes.
pub fn assert_call_splits(
    descriptor: FrontendDescriptorV1,
    cases: &[(&str, &str, Calls<'_>)],
    tools: &[FrontendToolV1],
) {
    for (text, content, calls) in cases.iter().copied() {
        let calls = owned(calls);
        let whole = parse_completion_v1(&descriptor.clone(), text, PromptModeV1::Chat)
            .unwrap_or_else(|error| panic!("parse {text:?}: {error}"));
        assert_eq!(whole.content, content, "{text:?}");
        assert_eq!(whole.tool_calls.len(), calls.len(), "{text:?}");
        for stream in [false, true] {
            for split in 1..text.len() {
                let bytes = text.as_bytes();
                let delivered = deliver(
                    descriptor.clone(),
                    PromptModeV1::Chat,
                    &[&bytes[..split], &bytes[split..]],
                    stream,
                    tools,
                )
                .unwrap_or_else(|error| panic!("{text:?} split {split}: {error}"));
                assert_eq!(delivered.content, content, "{text:?} split {split}");
                assert_eq!(delivered.calls, calls, "{text:?} split {split}");
                if stream {
                    assert_eq!(delivered.streamed, calls, "{text:?} split {split}");
                }
                assert_eq!(delivered.reason, Some(CompletionFinishV1::ToolCalls));
            }
        }
    }
}

/// Check every declared completion case in both streaming modes.
pub fn assert_call_endings(
    descriptor: FrontendDescriptorV1,
    cases: &[(&str, Calls<'_>)],
    tools: &[FrontendToolV1],
) {
    for (text, calls) in cases.iter().copied() {
        let with_end_token = format!("{text}<|im_end|>");
        for (end, text, base) in [
            (End::Length, text, CompletionFinishV1::Length),
            (End::EngineStop, text, CompletionFinishV1::Stop),
            (
                End::Token,
                with_end_token.as_str(),
                CompletionFinishV1::Stop,
            ),
        ] {
            for stream in [false, true] {
                let expected = owned(calls);
                let delivered = deliver_ending(
                    descriptor.clone(),
                    PromptModeV1::Chat,
                    &[text.as_bytes()],
                    stream,
                    tools,
                    &[],
                    end,
                )
                .unwrap_or_else(|error| panic!("{text:?} {end:?} stream {stream}: {error}"));
                let label = format!("{text:?} {end:?} stream {stream}");
                assert_eq!(delivered.calls, expected, "{label}");
                if stream {
                    assert_eq!(delivered.streamed, expected, "{label}");
                }
                assert_eq!(delivered.content, "", "{label}");
                let reason = if expected.is_empty() {
                    base
                } else {
                    CompletionFinishV1::ToolCalls
                };
                assert_eq!(delivered.reason, Some(reason), "{label}");
            }
        }
    }
}

/// Check every declared completion case in both streaming modes.
pub fn assert_no_calls(
    descriptor: FrontendDescriptorV1,
    cases: &[(&str, End, &str, CompletionFinishV1)],
    tools: &[FrontendToolV1],
) {
    for (text, end, content, reason) in cases.iter().copied() {
        for stream in [false, true] {
            for split in 1..text.len() {
                let bytes = text.as_bytes();
                let mut decoder =
                    decoder(descriptor.clone(), PromptModeV1::Chat, stream, tools, &[]).unwrap();
                decoder.set_tool_extraction(false);
                let delivered = run(decoder, &[&bytes[..split], &bytes[split..]], end)
                    .unwrap_or_else(|error| panic!("{text:?} split {split}: {error}"));
                let label = format!("{text:?} stream {stream} split {split}");
                assert!(delivered.calls.is_empty(), "{label}");
                assert!(delivered.streamed.is_empty(), "{label}");
                assert_eq!(delivered.content, content, "{label}");
                assert_eq!(delivered.reason, Some(reason), "{label}");
            }
        }
    }
}
