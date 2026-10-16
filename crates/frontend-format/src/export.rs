//! The documents a chat format's export reads and writes over
//! `chop.frontend.v1`.
//!
//! [`crate::ChatFormatExportV1`] is the bundle half: it opens over the
//! frontend component's bytes, and these read each request document the
//! host sends, run the format's renderer or the shared parser, and write the
//! answer or refusal document back. They know nothing of the host past the
//! documents in `runtime_bundle_abi::frontend_document`, so the export
//! serves the host the same way in process as from a loaded package.

use std::sync::Arc;

use runtime_bundle_abi::frontend::FrontendParserV1;
use runtime_bundle_abi::frontend_document as document;

use crate::descriptor::FormatDescriptorV1;
use crate::format::{
    FormatEventV1, FormatParserOptionsV1, FormatRenderRequestV1, FormatRenderedV1, ParsedToolCallV1,
};
use crate::format_decode::DeclaredParserV1;
use crate::request::{
    FrontendContentPartV1, FrontendErrorV1, FrontendJsonV1, FrontendMediaPartKindV1,
    FrontendMediaPartV1, FrontendMessageV1, FrontendRoleV1, FrontendToolCallV1,
    FrontendToolChoiceV1, FrontendToolV1, PromptSegmentV1,
};

/// One request's shared parser behind the export.
pub struct DeclaredExportParserV1 {
    parser: DeclaredParserV1,
}

/// Answers one render request document with what `render` renders for it,
/// or its refusal.
pub(crate) fn render_document_v1(
    request: &[u8],
    render: impl FnOnce(FormatRenderRequestV1<'_>) -> Result<FormatRenderedV1, FrontendErrorV1>,
) -> Result<Vec<u8>, Vec<u8>> {
    let request = document::RenderRequestV1::decode(request).map_err(document_refusal_v1)?;
    let rendered = RenderInputV1::read(request)?
        .render(render)
        .map_err(|error| refusal_v1(error.message, error.parameter.as_deref()))?;
    let segments = rendered
        .segments
        .into_iter()
        .map(|segment| match segment {
            PromptSegmentV1::Text(text) => Ok(document::SegmentV1::Text(text)),
            // The renderer's media list holds the parts it placed, each
            // carrying its request index (see `media_part_v1`).
            PromptSegmentV1::Media(index) => rendered
                .media
                .get(index)
                .and_then(|media| media.source.parse().ok())
                .map(document::SegmentV1::Media)
                .ok_or_else(|| refusal_v1("rendered media lost its request index", None)),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(document::RenderedV1 {
        segments,
        mode: rendered.mode,
        effort: rendered.effort,
        reasoning_open: rendered.reasoning_open,
        parser_state: rendered.parser_state,
    }
    .encode())
}

/// Answers one constraint request document with the grammar the tool
/// choice imposes in the format `descriptor`'s parser section declares.
pub(crate) fn constraint_document_v1(
    descriptor: &FormatDescriptorV1,
    request: &[u8],
) -> Result<Vec<u8>, Vec<u8>> {
    match document::ConstraintRequestV1::decode(request).map_err(document_refusal_v1)? {
        document::ConstraintRequestV1::ToolChoice { tools, choice } => {
            let tools = tools
                .into_iter()
                .map(tool_v1)
                .collect::<Result<Vec<_>, _>>()?;
            let choice = match choice {
                document::ToolChoiceV1::Auto => FrontendToolChoiceV1::Auto,
                document::ToolChoiceV1::None => FrontendToolChoiceV1::None,
                document::ToolChoiceV1::Required => FrontendToolChoiceV1::Required,
                document::ToolChoiceV1::Named(name) => FrontendToolChoiceV1::Named(name),
            };
            let grammar = crate::tool_grammar::tool_constraint_v1(descriptor, &tools, &choice)
                .map_err(|error| refusal_v1(error.message, error.parameter.as_deref()))?;
            Ok(grammar.map_or_else(Vec::new, |grammar| grammar.encode_wire()))
        }
    }
}

/// Opens the shared parser of one request's completion over `descriptor`.
pub(crate) fn open_parser_document_v1(
    descriptor: &Arc<FormatDescriptorV1>,
    request: &[u8],
) -> Result<DeclaredExportParserV1, Vec<u8>> {
    let request = document::ParserOpenV1::decode(request).map_err(document_refusal_v1)?;
    let tool_schemas = request
        .tool_schemas
        .into_iter()
        .map(tool_v1)
        .collect::<Result<Vec<_>, _>>()?;
    // The shared parser reads everything from its options; a render
    // leaves it no state.
    if !request.parser_state.is_empty() {
        return Err(refusal_v1(
            "the declared parser takes no parser state",
            None,
        ));
    }
    Ok(DeclaredExportParserV1 {
        parser: DeclaredParserV1::new(
            Arc::clone(descriptor),
            FormatParserOptionsV1 {
                mode: request.mode,
                reasoning_open: request.reasoning_open,
                extract_tools: request.extract_tools,
                stream_tool_calls: request.stream_tool_calls,
                raw: request.raw,
                ignore_eos: request.ignore_eos,
                engine_tokens: request.engine_tokens,
                tool_schemas,
            },
        ),
    })
}

impl FrontendParserV1 for DeclaredExportParserV1 {
    fn push(&mut self, text: &[u8], last: bool) -> Result<Vec<u8>, Vec<u8>> {
        let push = document::PushV1::decode(text).map_err(document_refusal_v1)?;
        let offset = |value: u64| {
            usize::try_from(value).map_err(|_| refusal_v1("push offset exceeds this host", None))
        };
        let keep = offset(push.keep)?;
        let stop_floor = push.stop_floor.map(offset).transpose()?;
        let events = self
            .parser
            .push_suffix(keep, push.suffix, push.held, last, stop_floor)
            .map_err(|message| refusal_v1(message, None))?;
        let mut answer = Vec::new();
        events_document_v1(&events, self.parser.cursor()).encode_into(&mut answer);
        Ok(answer)
    }
}

/// A render request as a format reads it.
struct RenderInputV1 {
    messages: Vec<FrontendMessageV1>,
    continue_final_message: bool,
    add_generation_prompt: bool,
    tools: Vec<FrontendToolV1>,
    tool_objects: Vec<FrontendJsonV1>,
    response_format: Option<FrontendJsonV1>,
    reasoning_effort: Option<String>,
    keywords: TemplateKeywordsV1,
}

/// The `chat_template_kwargs` entries a format reads typed.
#[derive(Default)]
struct TemplateKeywordsV1 {
    enable_thinking: Option<bool>,
    thinking: Option<bool>,
    reasoning_effort: Option<String>,
    preserve_thinking: Option<bool>,
    clear_thinking: Option<FrontendJsonV1>,
    add_vision_id: Option<FrontendJsonV1>,
}

impl RenderInputV1 {
    fn read(request: document::RenderRequestV1) -> Result<Self, Vec<u8>> {
        Ok(Self {
            messages: request
                .messages
                .into_iter()
                .map(message_v1)
                .collect::<Result<_, _>>()?,
            continue_final_message: request.continue_final_message,
            add_generation_prompt: request.add_generation_prompt,
            tools: request
                .tools
                .into_iter()
                .map(tool_v1)
                .collect::<Result<_, _>>()?,
            tool_objects: request
                .tool_objects
                .iter()
                .map(|object| json_v1(object, "tools"))
                .collect::<Result<_, _>>()?,
            response_format: request
                .response_format
                .as_deref()
                .map(|format| json_v1(format, "response_format"))
                .transpose()?,
            reasoning_effort: request.reasoning_effort,
            keywords: request
                .chat_template_kwargs
                .as_deref()
                .map(keywords_v1)
                .transpose()?
                .unwrap_or_default(),
        })
    }

    /// Renders this request with `render`.
    fn render(
        self,
        render: impl FnOnce(FormatRenderRequestV1<'_>) -> Result<FormatRenderedV1, FrontendErrorV1>,
    ) -> Result<FormatRenderedV1, FrontendErrorV1> {
        let Self {
            messages,
            continue_final_message,
            add_generation_prompt,
            tools,
            tool_objects,
            response_format,
            reasoning_effort,
            keywords,
        } = self;
        render(FormatRenderRequestV1 {
            messages,
            continue_final_message,
            add_generation_prompt,
            tools: &tools,
            tool_objects,
            response_format,
            reasoning_effort,
            enable_thinking: keywords.enable_thinking,
            thinking: keywords.thinking,
            nested_reasoning_effort: keywords.reasoning_effort,
            preserve_thinking: keywords.preserve_thinking,
            clear_thinking: keywords.clear_thinking,
            add_vision_id: keywords.add_vision_id,
        })
    }
}

/// Reads the keywords a format reads typed from the
/// `chat_template_kwargs` object. A typed keyword set to null is absent, an
/// opaque one (`clear_thinking`, `add_vision_id`) keeps its null, and one this
/// format does not read is ignored, as a template ignores it.
fn keywords_v1(text: &str) -> Result<TemplateKeywordsV1, Vec<u8>> {
    const PARAMETER: &str = "chat_template_kwargs";
    let FrontendJsonV1::Object(entries) = json_v1(text, PARAMETER)? else {
        return Err(refusal_v1(
            "chat_template_kwargs must be an object",
            Some(PARAMETER),
        ));
    };
    let mut keywords = TemplateKeywordsV1::default();
    for (name, value) in entries {
        // The opaque keywords keep a null: a template that reads one
        // decides what it means, as the typed request carries it.
        if value == FrontendJsonV1::Null
            && !matches!(name.as_str(), "clear_thinking" | "add_vision_id")
        {
            continue;
        }
        let flag = |value: &FrontendJsonV1| match value {
            FrontendJsonV1::Bool(flag) => Ok(*flag),
            _ => Err(refusal_v1(
                format!("chat_template_kwargs.{name} must be a boolean"),
                Some(PARAMETER),
            )),
        };
        match name.as_str() {
            "enable_thinking" => keywords.enable_thinking = Some(flag(&value)?),
            "thinking" => keywords.thinking = Some(flag(&value)?),
            "preserve_thinking" => keywords.preserve_thinking = Some(flag(&value)?),
            "reasoning_effort" => match value {
                FrontendJsonV1::String(effort) => keywords.reasoning_effort = Some(effort),
                _ => {
                    return Err(refusal_v1(
                        "chat_template_kwargs.reasoning_effort must be a string",
                        Some(PARAMETER),
                    ));
                }
            },
            "clear_thinking" => keywords.clear_thinking = Some(value),
            "add_vision_id" => keywords.add_vision_id = Some(value),
            _ => {}
        }
    }
    Ok(keywords)
}

fn message_v1(message: document::MessageV1) -> Result<FrontendMessageV1, Vec<u8>> {
    let role = match message.role.as_str() {
        "system" => FrontendRoleV1::System,
        "developer" => FrontendRoleV1::Developer,
        "user" => FrontendRoleV1::User,
        "assistant" => FrontendRoleV1::Assistant,
        "tool" => FrontendRoleV1::Tool,
        _ => {
            return Err(refusal_v1(
                format!("message role '{}' is not supported", message.role),
                Some("messages"),
            ));
        }
    };
    Ok(FrontendMessageV1 {
        role,
        content: message.content,
        parts: message.parts.map(|parts| {
            parts
                .into_iter()
                .map(|part| match part {
                    document::PartV1::Text(text) => FrontendContentPartV1::Text(text),
                    document::PartV1::Media(media) => {
                        FrontendContentPartV1::Media(media_part_v1(&media))
                    }
                    document::PartV1::ToolReference(name) => {
                        FrontendContentPartV1::ToolReference { name }
                    }
                })
                .collect()
        }),
        // A format reads `reasoning_content`, the field the host forwards;
        // `reasoning` waits for a host that forwards both.
        reasoning_content: message.reasoning_content,
        tool_calls: message
            .tool_calls
            .into_iter()
            .map(|call| FrontendToolCallV1 {
                id: call.id,
                name: call.name,
                arguments: call.arguments,
            })
            .collect(),
        tool_call_id: message.tool_call_id,
        tools: message
            .tools
            .iter()
            .map(|tool| json_v1(tool, "messages"))
            .collect::<Result<_, _>>()?,
    })
}

/// A media part as a format reads it. A format never reads a source (the
/// host keeps the media), so the request index rides there and maps each
/// placed part back to the request.
fn media_part_v1(media: &document::MediaPartV1) -> FrontendMediaPartV1 {
    FrontendMediaPartV1 {
        kind: match media.kind {
            document::MediaKindV1::Image => FrontendMediaPartKindV1::Image,
            document::MediaKindV1::Audio => FrontendMediaPartKindV1::Audio,
            document::MediaKindV1::Video => FrontendMediaPartKindV1::Video,
        },
        part_type: media.part_type.clone(),
        source: media.index.to_string(),
        format: None,
    }
}

fn tool_v1(tool: document::ToolV1) -> Result<FrontendToolV1, Vec<u8>> {
    Ok(FrontendToolV1 {
        parameters: json_v1(&tool.parameters, "tools")?,
        name: tool.name,
        description: tool.description,
        parameters_present: tool.parameters_present,
        strict: tool.strict,
    })
}

/// Reads one JSON value the host sent as its text.
fn json_v1(text: &str, parameter: &str) -> Result<FrontendJsonV1, Vec<u8>> {
    serde_json::from_str(text)
        .map_err(|error| refusal_v1(format!("invalid JSON: {error}"), Some(parameter)))
}

fn events_document_v1(events: &[FormatEventV1], cursor: usize) -> document::EventsV1<'_> {
    let range = |range: &std::ops::Range<usize>| range.start as u64..range.end as u64;
    document::EventsV1 {
        events: events
            .iter()
            .map(|event| match event {
                FormatEventV1::Reasoning(span) => document::EventV1::Reasoning(range(span)),
                FormatEventV1::Content(span) => document::EventV1::Content(range(span)),
                FormatEventV1::ToolCalls {
                    calls: group,
                    range: span,
                } => document::EventV1::ToolCalls {
                    range: range(span),
                    calls: calls(group),
                },
                FormatEventV1::ToolCallDeltas(deltas) => document::EventV1::ToolCallDeltas(
                    deltas
                        .iter()
                        .map(|delta| document::DeltaV1 {
                            index: u32::try_from(delta.index)
                                .expect("tool calls are bounded far below u32"),
                            name: delta.name.as_deref(),
                            arguments: &delta.arguments,
                        })
                        .collect(),
                ),
                FormatEventV1::Finish { reason, completion } => document::EventV1::Finish {
                    reason: *reason,
                    completion: completion
                        .as_ref()
                        .map(|completion| document::CompletionV1 {
                            content: &completion.content,
                            reasoning_content: completion.reasoning_content.as_deref(),
                            tool_calls: calls(&completion.tool_calls),
                        }),
                },
            })
            .collect(),
        cursor: cursor as u64,
    }
}

fn calls(calls: &[ParsedToolCallV1]) -> Vec<document::CallV1<'_>> {
    calls
        .iter()
        .map(|call| document::CallV1 {
            name: &call.name,
            arguments: &call.arguments,
        })
        .collect()
}

pub(crate) fn refusal_v1(message: impl Into<String>, parameter: Option<&str>) -> Vec<u8> {
    document::RefusalV1::new(message, parameter).encode()
}

fn document_refusal_v1(error: document::FrontendDocumentErrorV1) -> Vec<u8> {
    refusal_v1(error.to_string(), None)
}

#[cfg(test)]
mod tests {
    use runtime_bundle_abi::frontend::FrontendCallErrorV1;

    use super::*;
    use crate::test_support::FixtureFormatV1;
    use crate::test_support::fixtures as test_common;

    /// The fixture format's export, over the key-value fixture.
    fn export() -> runtime_bundle_abi::frontend::FrontendExportV1 {
        crate::chat_format_export_v1::<FixtureFormatV1>()
    }

    fn frontend() -> runtime_bundle_abi::frontend::FrontendHandleV1 {
        export().open(test_common::KEY_VALUE.as_bytes()).unwrap().0
    }

    fn refusal(error: FrontendCallErrorV1) -> document::RefusalV1 {
        match error {
            FrontendCallErrorV1::Refused(bytes) => document::RefusalV1::decode(&bytes).unwrap(),
            other => panic!("not a refusal: {other}"),
        }
    }

    fn user(text: &str) -> document::MessageV1 {
        document::MessageV1 {
            role: "user".into(),
            content: Some(text.into()),
            parts: None,
            reasoning: None,
            reasoning_content: None,
            tool_calls: vec![],
            tool_call_id: None,
            tools: vec![],
        }
    }

    fn request(messages: Vec<document::MessageV1>) -> document::RenderRequestV1 {
        document::RenderRequestV1 {
            messages,
            continue_final_message: false,
            add_generation_prompt: true,
            tools: vec![],
            tool_objects: vec![],
            response_format: None,
            reasoning_effort: None,
            chat_template_kwargs: None,
        }
    }

    fn parser_open(mode: document::ModeV1) -> document::ParserOpenV1 {
        document::ParserOpenV1 {
            mode,
            reasoning_open: mode == document::ModeV1::Thinking,
            extract_tools: true,
            stream_tool_calls: false,
            raw: false,
            ignore_eos: false,
            engine_tokens: true,
            tool_schemas: vec![],
            parser_state: vec![],
        }
    }

    fn push(keep: u64, suffix: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        document::PushV1 {
            keep,
            held: false,
            stop_floor: Some(0),
            suffix,
        }
        .encode_into(&mut bytes);
        bytes
    }

    #[test]
    fn the_open_answer_is_the_formats() {
        let (_, opened) = export().open(test_common::KEY_VALUE.as_bytes()).unwrap();
        assert_eq!(
            document::OpenedV1::decode(&opened),
            Ok(document::OpenedV1::default())
        );
    }

    #[test]
    fn open_refuses_a_component_that_does_not_read() {
        let error = export().open(b"{}").map(|_| ()).unwrap_err();
        assert!(
            refusal(error)
                .message
                .starts_with("invalid frontend component")
        );
    }

    #[test]
    fn render_answers_what_the_format_renders() {
        let rendered = frontend()
            .render(&request(vec![user("hello")]).encode(), 1 << 20)
            .unwrap();
        let rendered = document::RenderedV1::decode(&rendered).unwrap();
        assert_eq!(
            rendered.segments,
            [document::SegmentV1::Text("[U]user\nhello\n[A]\n".into())]
        );
        assert_eq!(rendered.mode, document::ModeV1::Chat);
        assert_eq!(rendered.effort, "low");
        assert!(!rendered.reasoning_open);
        assert!(rendered.parser_state.is_empty());
    }

    #[test]
    fn render_refusals_keep_their_message_and_parameter() {
        let mut bad_effort = request(vec![user("hello")]);
        bad_effort.reasoning_effort = Some("extreme".into());
        let refused = refusal(
            frontend()
                .render(&bad_effort.encode(), 1 << 20)
                .unwrap_err(),
        );
        assert_eq!(refused.parameter.as_deref(), Some("reasoning_effort"));
        assert!(
            refused
                .message
                .starts_with("top-level reasoning_effort must be")
        );

        let mut role = user("hello");
        role.role = "latest_reminder".into();
        let refused = refusal(
            frontend()
                .render(&request(vec![role]).encode(), 1 << 20)
                .unwrap_err(),
        );
        assert_eq!(
            refused,
            document::RefusalV1::new(
                "message role 'latest_reminder' is not supported",
                Some("messages")
            )
        );

        for (kwargs, message) in [
            ("[]", "chat_template_kwargs must be an object"),
            (
                r#"{"enable_thinking":"yes"}"#,
                "chat_template_kwargs.enable_thinking must be a boolean",
            ),
            (
                r#"{"reasoning_effort":1}"#,
                "chat_template_kwargs.reasoning_effort must be a string",
            ),
        ] {
            let mut keywords = request(vec![user("hello")]);
            keywords.chat_template_kwargs = Some(kwargs.into());
            assert_eq!(
                refusal(frontend().render(&keywords.encode(), 1 << 20).unwrap_err()),
                document::RefusalV1::new(message, Some("chat_template_kwargs"))
            );
        }
        // Null keywords are absent, and keywords this format does not read
        // are ignored.
        let mut keywords = request(vec![user("hello")]);
        keywords.chat_template_kwargs = Some(r#"{"enable_thinking":null,"other":[1]}"#.into());
        assert!(frontend().render(&keywords.encode(), 1 << 20).is_ok());

        let refused = refusal(frontend().render(b"not a document", 1 << 20).unwrap_err());
        assert_eq!(refused.parameter, None);
        assert!(refused.message.starts_with("frontend document:"));
    }

    /// A format that reads no `tool_reference` part admits none in its open
    /// answer; one that reaches it anyway is refused by name, in a tool
    /// message or another, and the refusal crosses whole.
    #[test]
    fn a_format_that_reads_no_tool_reference_refuses_one() {
        let reference = || Some(vec![document::PartV1::ToolReference("lookup".into())]);
        let tool = document::MessageV1 {
            role: "tool".into(),
            content: Some(String::new()),
            parts: reference(),
            tool_call_id: Some("call-1".into()),
            ..user("")
        };
        let asked = document::MessageV1 {
            parts: reference(),
            ..user("")
        };
        let (frontend, opened) = export().open(test_common::KEY_VALUE.as_bytes()).unwrap();
        assert!(!document::OpenedV1::decode(&opened).unwrap().tool_references);
        for messages in [vec![user("hello"), tool.clone()], vec![asked.clone()]] {
            assert_eq!(
                refusal(
                    frontend
                        .render(&request(messages).encode(), 1 << 20)
                        .unwrap_err()
                ),
                document::RefusalV1::new(
                    "tool_reference content parts are not supported by this model",
                    Some("messages")
                )
            );
        }
    }

    #[test]
    fn the_parser_reads_pushed_suffixes() {
        let frontend = frontend();
        // Caller tokens, so the end marker's spelling ends the completion.
        let mut open = parser_open(document::ModeV1::Thinking);
        open.engine_tokens = false;
        let mut parser = frontend.open_parser(&open.encode()).unwrap();
        let answer = parser
            .push(&push(0, b"plan</t>he"), false, 1 << 20)
            .unwrap();
        let events = document::EventsV1::decode(&answer).unwrap();
        assert_eq!(
            events.events,
            [
                document::EventV1::Reasoning(0..4),
                document::EventV1::Content(8..10)
            ]
        );
        assert_eq!(events.cursor, 10);
        let answer = parser.push(&push(10, b"llo[EOS]"), true, 1 << 20).unwrap();
        let events = document::EventsV1::decode(&answer).unwrap();
        assert_eq!(events.events[0], document::EventV1::Content(10..13));
        assert_eq!(
            events.events[1],
            document::EventV1::Finish {
                reason: document::FinishReasonV1::Stop,
                completion: Some(document::CompletionV1 {
                    content: "hello",
                    reasoning_content: Some("plan"),
                    tool_calls: vec![],
                }),
            }
        );
        assert_eq!(events.cursor, 18);
    }

    #[test]
    fn the_parser_refuses_what_it_cannot_follow() {
        let frontend = frontend();
        let mut parser = frontend
            .open_parser(&parser_open(document::ModeV1::Chat).encode())
            .unwrap();
        let refused = refusal(parser.push(&push(1, b"x"), false, 1 << 20).unwrap_err());
        assert_eq!(
            refused.message,
            "invalid conversation sequence: a push keeps bytes the parser never received"
        );
        let mut stateful = parser_open(document::ModeV1::Chat);
        stateful.parser_state = vec![1];
        assert_eq!(
            refusal(
                frontend
                    .open_parser(&stateful.encode())
                    .map(|_| ())
                    .unwrap_err()
            )
            .message,
            "the declared parser takes no parser state"
        );
    }

    #[test]
    fn constraints_and_argument_checks_cross_as_documents() {
        let tool = document::ToolV1 {
            name: "lookup".into(),
            description: None,
            parameters: r#"{"type":"object","properties":{"city":{"type":"string"}}}"#.into(),
            parameters_present: true,
            strict: None,
        };
        let constraint = |choice| {
            frontend().constraint(
                &document::ConstraintRequestV1::ToolChoice {
                    tools: vec![tool.clone()],
                    choice,
                }
                .encode(),
                1 << 20,
            )
        };
        assert!(constraint(document::ToolChoiceV1::None).unwrap().is_empty());
        assert!(constraint(document::ToolChoiceV1::Auto).unwrap().is_empty());
        // The parser section's tool grammar forces a call.
        assert!(
            !constraint(document::ToolChoiceV1::Required)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            refusal(constraint(document::ToolChoiceV1::Named("absent".into())).unwrap_err())
                .parameter
                .as_deref(),
            Some("tool_choice")
        );
    }
}
