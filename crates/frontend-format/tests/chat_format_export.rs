#![forbid(unsafe_code)]
//! The generic export serves a `ChatFormatV1` and its `format.parser`
//! section: it opens only over a parser declaration, answers the format's
//! open, sets `reasoning_open` from the prompt, and refuses a render that
//! leaves parser state. That every answer crosses the documents as the
//! format and the shared parser give it directly is runtime-frontend's
//! paired check (`test_support::documents`).
#[allow(dead_code)]
mod common;

use frontend_format::{
    ChatFormatV1, FormatDescriptorV1, FormatRenderRequestV1, FormatRenderedV1, FrontendErrorV1,
    OpenedV1, PromptModeV1, PromptSegmentV1, chat_format_export_v1,
};
use runtime_bundle_abi::frontend::FrontendCallErrorV1;
use runtime_bundle_abi::frontend_document as document;
use serde_json::{Value, json};

const MAXIMUM: usize = 1 << 20;

/// The function-parameter fixture, whose reasoning markers are `<think>`
/// and `</think>`.
fn parser_component() -> Vec<u8> {
    common::FUNCTION_PARAMETERS.as_bytes().to_vec()
}

/// An answer, or its refusal's message and parameter.
fn answer(result: Result<Vec<u8>, FrontendCallErrorV1>) -> Result<Vec<u8>, String> {
    result.map_err(|error| match error {
        FrontendCallErrorV1::Refused(bytes) => {
            let refusal = document::RefusalV1::decode(&bytes).unwrap();
            format!("{} ({:?})", refusal.message, refusal.parameter)
        }
        other => panic!("not an answer or refusal: {other}"),
    })
}

fn message(role: &str, content: Option<&str>) -> document::MessageV1 {
    document::MessageV1 {
        role: role.into(),
        content: content.map(str::to_owned),
        parts: None,
        reasoning: None,
        reasoning_content: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
        tools: Vec::new(),
    }
}

fn request(messages: Vec<document::MessageV1>) -> document::RenderRequestV1 {
    document::RenderRequestV1 {
        messages,
        continue_final_message: false,
        add_generation_prompt: true,
        tools: Vec::new(),
        tool_objects: Vec::new(),
        response_format: None,
        reasoning_effort: None,
        chat_template_kwargs: None,
    }
}

#[test]
fn the_generic_export_serves_only_a_parser_declaration() {
    let mut declaration: Value = serde_json::from_str(common::FUNCTION_PARAMETERS).unwrap();
    declaration["format"] = json!({"template": {"json_encoding": "python-json-v1"}});
    let error = chat_format_export_v1::<ScriptedV1<1>>()
        .open(&serde_json::to_vec(&declaration).unwrap())
        .map(|_| ())
        .unwrap_err();
    let refusal = answer(Err(error)).unwrap_err();
    assert!(refusal.contains("template"), "{refusal}");
}

/// A format whose open and render answers a test sets by type.
struct ScriptedV1<const KIND: u8>;

impl<const KIND: u8> ChatFormatV1 for ScriptedV1<KIND> {
    fn open(_: &FormatDescriptorV1) -> Result<Self, String> {
        if KIND == 0 {
            return Err("this format writes no image labels".into());
        }
        Ok(Self)
    }

    fn opened(&self) -> OpenedV1 {
        OpenedV1 {
            reads_tool_objects: true,
            ..OpenedV1::default()
        }
    }

    fn render(
        &self,
        request: FormatRenderRequestV1<'_>,
    ) -> Result<FormatRenderedV1, FrontendErrorV1> {
        // The prompt is the last message, rendered in thinking mode, and the
        // render reports the opposite of what its prompt leaves open.
        let prompt = request
            .messages
            .last()
            .and_then(|message| message.content.clone())
            .unwrap_or_default();
        Ok(FormatRenderedV1 {
            segments: vec![PromptSegmentV1::Text(prompt.clone())],
            prompt,
            media: Vec::new(),
            mode: PromptModeV1::Thinking,
            effort: "low".into(),
            reasoning_open: request.continue_final_message,
            parser_state: if KIND == 2 { vec![1] } else { Vec::new() },
        })
    }
}

#[test]
fn the_export_answers_the_format_open_and_sets_reasoning_open_from_the_prompt() {
    let component = parser_component();
    let error = chat_format_export_v1::<ScriptedV1<0>>()
        .open(&component)
        .map(|_| ())
        .unwrap_err();
    assert_eq!(
        answer(Err(error)).unwrap_err(),
        "this format writes no image labels (None)"
    );

    let (frontend, opened) = chat_format_export_v1::<ScriptedV1<1>>()
        .open(&component)
        .unwrap();
    assert!(
        document::OpenedV1::decode(&opened)
            .unwrap()
            .reads_tool_objects
    );
    for (text, continued, open) in [
        ("<|a|>\n<think>", false, true),
        ("<|a|>\n<think>", true, true),
        ("<|a|>\n<think>plan</think>answer", true, false),
        ("<|a|>\n<think>plan</think>answer", false, true),
    ] {
        let mut render = request(vec![message("user", Some(text))]);
        render.continue_final_message = continued;
        let rendered =
            document::RenderedV1::decode(&frontend.render(&render.encode(), MAXIMUM).unwrap())
                .unwrap();
        assert_eq!(rendered.reasoning_open, open, "{text} {continued}");
    }

    let (frontend, _) = chat_format_export_v1::<ScriptedV1<2>>()
        .open(&component)
        .unwrap();
    assert_eq!(
        answer(frontend.render(&request(vec![message("user", Some("x"))]).encode(), MAXIMUM))
            .unwrap_err(),
        "a chat format render leaves no parser state: the shared parser reads format.parser (None)"
    );
}
