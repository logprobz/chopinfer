#![forbid(unsafe_code)]
//! Streaming vectors for where a thinking prompt leaves reasoning.
//!
//! Each vector renders one request through the test fixture format's
//! documents and decodes token chunks the way admission does: through the frontend that
//! rendered the prompt, beginning in reasoning exactly when the render
//! reported that the prompt left it open. A thinking prompt that continues a
//! final assistant turn which already closed its reasoning leaves reasoning
//! closed, so the completion continues the answer.

use std::sync::Arc;

use frontend_format::test_support::fixtures::FUNCTION_PARAMETERS;
use runtime_frontend::{
    BundleChatStateV1, BundleFrontendV1, CompletionDecoderV1, CompletionFinishV1,
    CompletionUpdateV1, DecoderLimitsV1, FrontendDescriptorV1, GeneratedTokenV1, PromptModeV1,
};
use runtime_server::{
    FrontendChatRequestV1, FrontendMessageV1, FrontendRoleV1, FrontendToolChoiceV1, FrontendToolV1,
    ServerFrontendV1,
};

/// The function-parameter fixture, rendered by the fixture format.
fn fixture() -> FrontendDescriptorV1 {
    FrontendDescriptorV1::from_component_bytes(FUNCTION_PARAMETERS.as_bytes()).unwrap()
}

fn message(role: FrontendRoleV1, content: &str, reasoning: Option<&str>) -> FrontendMessageV1 {
    FrontendMessageV1 {
        role,
        content: Some(content.to_owned()),
        parts: None,
        reasoning_content: reasoning.map(str::to_owned),
        tool_calls: Vec::new(),
        tool_call_id: None,
        tools: Vec::new(),
    }
}

/// A question, and, when `continued`, the start of the assistant's answer
/// after reasoning it already closed, which the prompt continues.
fn request(thinking: bool, continued: bool, tools: bool) -> FrontendChatRequestV1 {
    let mut messages = vec![message(FrontendRoleV1::User, "question", None)];
    if continued {
        messages.push(message(FrontendRoleV1::Assistant, "Par", Some("thought")));
    }
    FrontendChatRequestV1 {
        messages,
        continue_final_message: continued,
        add_generation_prompt: !continued,
        tools: if tools { vec![lookup()] } else { Vec::new() },
        tool_choice: if tools {
            FrontendToolChoiceV1::Auto
        } else {
            FrontendToolChoiceV1::None
        },
        response_format: None,
        reasoning_effort: None,
        enable_thinking: Some(thinking),
        thinking: None,
        nested_reasoning_effort: None,
        preserve_thinking: None,
        clear_thinking: None,
        add_vision_id: None,
        tool_objects: Vec::new(),
        chat_template_kwargs: None,
    }
}

/// The one tool a tool-call vector offers.
fn lookup() -> FrontendToolV1 {
    FrontendToolV1 {
        name: "lookup".to_owned(),
        description: None,
        parameters: serde_json::from_value(serde_json::json!({
            "type": "object",
            "properties": {"query": {"type": "string"}}
        }))
        .unwrap(),
        parameters_present: true,
        strict: None,
    }
}

/// One vector: a request, the prompt it renders and where that prompt leaves
/// reasoning, the completion's token chunks before its stop token, and the
/// reasoning and content the chunks deliver.
struct VectorV1 {
    name: &'static str,
    declaration: fn() -> FrontendDescriptorV1,
    thinking: bool,
    continued: bool,
    prompt: &'static str,
    reasoning_open: bool,
    /// Whether the request offers the `lookup` tool.
    tools: bool,
    chunks: &'static [&'static str],
    reasoning: &'static [&'static str],
    content: &'static [&'static str],
    /// The tool calls delivered, as name and arguments.
    calls: &'static [(&'static str, &'static str)],
}

const VECTORS: &[VectorV1] = &[
    VectorV1 {
        name: "thinking continuation after closed reasoning",
        declaration: fixture,
        thinking: true,
        continued: true,
        prompt: "<|u|>user\nquestion\n<|a|>\n<think>thought</think>Par",
        reasoning_open: false,
        chunks: &["is", " big", "."],
        reasoning: &[],
        content: &["is", " big", "."],
        tools: false,
        calls: &[],
    },
    VectorV1 {
        name: "thinking generation prompt",
        declaration: fixture,
        thinking: true,
        continued: false,
        prompt: "<|u|>user\nquestion\n<|a|>\n<think>",
        reasoning_open: true,
        chunks: &["hm", "</think>", "ok"],
        reasoning: &["hm"],
        content: &["ok"],
        tools: false,
        calls: &[],
    },
    VectorV1 {
        name: "chat continuation",
        declaration: fixture,
        thinking: false,
        continued: true,
        prompt: "<|u|>user\nquestion\n<|a|>\n<think>thought</think>Par",
        reasoning_open: false,
        chunks: &["is", " big", "."],
        reasoning: &[],
        content: &["is", " big", "."],
        tools: false,
        calls: &[],
    },
    VectorV1 {
        name: "thinking continuation with a tool call",
        declaration: fixture,
        thinking: true,
        continued: true,
        prompt: "<|u|>tools\n{\"name\": \"lookup\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\"}}}}\n<|u|>user\nquestion\n<|a|>\n<think>thought</think>Par",
        reasoning_open: false,
        tools: true,
        chunks: &[
            "Checking.",
            "<tool_call>\n<function=lookup>\n<parameter=query>\nx\n</parameter>\n</function>\n</tool_call>",
        ],
        reasoning: &[],
        content: &["Checking."],
        calls: &[("lookup", "{\"query\": \"x\"}")],
    },
];

#[test]
fn a_continued_turn_that_closed_its_reasoning_continues_the_answer() {
    for vector in VECTORS {
        let name = vector.name;
        let descriptor = Arc::new((vector.declaration)());
        let frontend = BundleFrontendV1::with_export(
            descriptor.as_ref().clone(),
            &runtime_frontend::test_support::fixture_export_v1(),
        )
        .unwrap();
        let rendered = frontend
            .render_chat(request(vector.thinking, vector.continued, vector.tools))
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
        assert_eq!(rendered.prompt, vector.prompt, "{name}");
        let state = BundleChatStateV1::from_state(&rendered.state).unwrap();
        let mode = if vector.thinking {
            PromptModeV1::Thinking
        } else {
            PromptModeV1::Chat
        };
        assert_eq!(state.mode(), mode, "{name}");
        assert_eq!(state.reasoning_open(), vector.reasoning_open, "{name}");

        // Admission's settings, from the chat's state.
        let mut decoder = CompletionDecoderV1::new(
            Arc::clone(&descriptor),
            state.mode(),
            DecoderLimitsV1::declared(&descriptor),
        );
        decoder.set_parser_source(state.parser_source().unwrap().clone());
        decoder.set_tool_schemas(state.tools());
        decoder.set_tool_extraction(state.extracts_tool_calls());
        decoder.set_reasoning_open(state.reasoning_open());
        let mut reasoning = Vec::new();
        let mut content = Vec::new();
        let mut calls = Vec::new();
        let mut finished = None;
        // The text chunks, then the spelling of the fixture's completion
        // end, which ends the completion.
        let tokens = vector
            .chunks
            .iter()
            .map(|chunk| (200, chunk.as_bytes()))
            .chain([(200, b"[EOS]".as_slice())]);
        for (ordinal, (id, bytes)) in tokens.enumerate() {
            let token = GeneratedTokenV1::new(ordinal as u64, id, bytes.to_vec());
            for update in decoder.push(token).unwrap() {
                match update {
                    CompletionUpdateV1::Reasoning(text) => reasoning.push(text.text),
                    CompletionUpdateV1::Content(text) => content.push(text.text),
                    CompletionUpdateV1::ToolCalls(group) => calls.extend(
                        group
                            .calls
                            .into_iter()
                            .map(|call| (call.name, call.arguments)),
                    ),
                    CompletionUpdateV1::Finished(done) => finished = Some(done),
                    other => panic!("{name}: {other:?}"),
                }
            }
        }
        assert_eq!(reasoning, vector.reasoning, "{name}");
        assert_eq!(content, vector.content, "{name}");
        let finished = finished.unwrap_or_else(|| panic!("{name}: the completion finished"));
        let expected: Vec<_> = vector
            .calls
            .iter()
            .map(|(name, arguments)| ((*name).to_owned(), (*arguments).to_owned()))
            .collect();
        assert_eq!(calls, expected, "{name}");
        let reason = if expected.is_empty() {
            CompletionFinishV1::Stop
        } else {
            CompletionFinishV1::ToolCalls
        };
        assert_eq!(finished.reason, reason, "{name}");
    }
}
