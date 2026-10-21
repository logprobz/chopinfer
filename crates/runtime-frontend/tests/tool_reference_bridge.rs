#![forbid(unsafe_code)]

//! A `tool_reference` part crosses the bundle boundary only to an export that
//! declares it renders them.
//!
//! Two in-process exports serve the same declaration. One's open answer sets
//! `tool_references`; its render writes each part of the render document it
//! receives into the prompt, so the prompt shows what crossed. The other's
//! does not, and the transport refuses the same request before it renders.

use frontend_format::test_support::fixtures::KEY_VALUE;
use runtime_bundle_abi::frontend::{FrontendExportV1, FrontendParserV1, FrontendV1};
use runtime_bundle_abi::frontend_document as document;
use runtime_frontend::{
    BundleFrontendV1, FrontendDescriptorV1, FrontendMediaArenaV1, FrontendMediaElementV1,
    FrontendMediaEncoderV1, FrontendMediaKindV1, FrontendMediaModalityV1, FrontendMediaRecipeV1,
    FrontendMediaRenderV1, FrontendMultimodalV1,
};
use runtime_server::{PreparedPrompt, ServerFrontendV1, prepare_chat_request_v1};

/// An export whose open answer declares `tool_references` as `REFERENCES`.
struct ReferencesExportV1<const REFERENCES: bool>;

struct NoParserV1;

impl FrontendParserV1 for NoParserV1 {
    fn push(&mut self, _: &[u8], _: bool) -> Result<Vec<u8>, Vec<u8>> {
        Err(document::RefusalV1::new("this export parses nothing", None).encode())
    }
}

impl<const REFERENCES: bool> FrontendV1 for ReferencesExportV1<REFERENCES> {
    type Parser = NoParserV1;

    fn open(_: &[u8]) -> Result<(Self, Vec<u8>), Vec<u8>> {
        let opened = document::OpenedV1 {
            tool_references: REFERENCES,
            ..document::OpenedV1::default()
        };
        Ok((Self, opened.encode()))
    }

    /// Writes each message as its role and its parts, or its content when
    /// it has none, a reference as `@name`.
    fn render(&self, request: &[u8]) -> Result<Vec<u8>, Vec<u8>> {
        let request = document::RenderRequestV1::decode(request).unwrap();
        let prompt = request
            .messages
            .iter()
            .map(|message| {
                let content = match &message.parts {
                    Some(parts) => parts
                        .iter()
                        .map(|part| match part {
                            document::PartV1::Text(text) => text.clone(),
                            document::PartV1::Media(media) => format!("[media {}]", media.index),
                            document::PartV1::ToolReference(name) => format!("@{name}"),
                        })
                        .collect(),
                    None => message.content.clone().unwrap_or_default(),
                };
                format!("{}:{content}", message.role)
            })
            .collect::<Vec<_>>()
            .join("|");
        Ok(document::RenderedV1 {
            segments: vec![document::SegmentV1::Text(prompt)],
            mode: document::ModeV1::Chat,
            effort: "low".into(),
            reasoning_open: false,
            parser_state: Vec::new(),
        }
        .encode())
    }

    fn constraint(&self, _: &[u8]) -> Result<Vec<u8>, Vec<u8>> {
        Ok(Vec::new())
    }

    fn open_parser(&self, _: &[u8]) -> Result<Self::Parser, Vec<u8>> {
        Ok(NoParserV1)
    }
}

fn frontend<const REFERENCES: bool>() -> BundleFrontendV1 {
    frontend_of::<REFERENCES>(
        FrontendDescriptorV1::from_component_bytes(KEY_VALUE.as_bytes()).unwrap(),
    )
}

fn frontend_of<const REFERENCES: bool>(descriptor: FrontendDescriptorV1) -> BundleFrontendV1 {
    BundleFrontendV1::with_export(
        descriptor,
        &FrontendExportV1::in_process::<ReferencesExportV1<REFERENCES>>(),
    )
    .unwrap()
}

/// A tool result answering one call with references by `name` and by
/// `tool_name`.
fn body() -> serde_json::Map<String, serde_json::Value> {
    let serde_json::Value::Object(body) = serde_json::json!({
        "messages": [
            {"role": "user", "content": "Find a tool."},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_s", "type": "function",
                "function": {"name": "tool_search", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_s", "content": [
                {"type": "tool_reference", "name": "lookup"},
                {"type": "tool_reference", "tool_name": "convert"},
            ]},
        ],
    }) else {
        unreachable!("the body is an object")
    };
    body
}

#[test]
fn a_declaring_export_receives_each_reference_by_name() {
    let frontend = frontend::<true>();
    assert!(frontend.tool_references());
    let prepared = prepare_chat_request_v1(&frontend, 1 << 20, body(), 8).unwrap();
    assert_eq!(
        prepared.prompt,
        PreparedPrompt::RenderedChat("user:Find a tool.|assistant:|tool:@lookup@convert".into())
    );
}

#[test]
fn an_export_without_the_flag_is_never_sent_one() {
    let frontend = frontend::<false>();
    assert!(!frontend.tool_references());
    let error = prepare_chat_request_v1(&frontend, 1 << 20, body(), 8).unwrap_err();
    assert_eq!(
        error.message,
        "tool_reference content parts are not supported by this model"
    );
    assert_eq!(error.parameter.as_deref(), Some("messages"));
}

/// A reference's name is client text a format may write, so a declaration
/// with media refuses one that carries its media placeholder, as it refuses
/// such text.
#[test]
fn a_reference_carrying_a_media_placeholder_is_refused() {
    let mut descriptor = FrontendDescriptorV1::from_component_bytes(KEY_VALUE.as_bytes()).unwrap();
    descriptor.multimodal = Some(FrontendMultimodalV1 {
        arena: FrontendMediaArenaV1 {
            rows: 64,
            page_rows: 64,
            hidden: 8,
            element: FrontendMediaElementV1::Bf16,
        },
        encoders: vec![FrontendMediaEncoderV1 {
            name: "image".into(),
            payload_element: FrontendMediaElementV1::Bf16,
            payload_columns: 4,
            maximum_payload_bytes: 64,
            cost_per_row_milli: 1,
        }],
        modalities: vec![FrontendMediaModalityV1 {
            kind: FrontendMediaKindV1::Image,
            part_types: vec!["image_url".into()],
            maximum_items: 2,
            maximum_bytes: 64,
            maximum_tokens: 8,
            recipe: FrontendMediaRecipeV1 {
                tag: "image.patch-grid.v1".into(),
                parameters: serde_json::json!({}),
            },
            encoder: 0,
            render: FrontendMediaRenderV1 {
                pad: "[A]".into(),
                grid: None,
            },
            positions: None,
            layout: None,
            formats: None,
        }],
    });
    let frontend = frontend_of::<true>(descriptor);
    let prepared = prepare_chat_request_v1(&frontend, 1 << 20, body(), 8).unwrap();
    assert_eq!(
        prepared.prompt,
        PreparedPrompt::RenderedChat("user:Find a tool.|assistant:|tool:@lookup@convert".into())
    );
    let mut padded = body();
    padded["messages"][2]["content"][0]["name"] = "look[A]up".into();
    let error = prepare_chat_request_v1(&frontend, 1 << 20, padded, 8).unwrap_err();
    assert_eq!(error.message, "client text contains a media placeholder");
    assert_eq!(error.parameter.as_deref(), Some("messages"));
}
