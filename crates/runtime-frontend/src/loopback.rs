//! The `frontend-format` library linked into test builds (`test-support`).
//!
//! A host serves its package's own frontend export and links no format.
//! Tests serve chat formats from the library instead, and this is the one
//! place that names it. A test frontend opens a chat format's export, an
//! in-process loopback of the ABI, and reaches the format only through the
//! `chop.frontend.v1` documents. A test decoder given no bundle frontend
//! calls the shared parser directly ([`LibraryParserV1`]), tests compare a
//! decoder against the shared whole-completion parse
//! ([`parse_completion_v1`]), and they compare the documents against a chat
//! format called directly (`LibraryRendererV1`).

use std::sync::Arc;

use runtime_bundle_abi::frontend::FrontendExportV1;

use crate::decode::{AssistantCompletionV1, ParsedToolCallV1, PromptModeV1};
use crate::descriptor::FrontendDescriptorV1;
use crate::format::{
    FormatEventV1, FormatInputV1, FormatParserOptionsV1, FormatParserV1, FormatToolCallDeltaV1,
};

/// The export a test opens to parse completions: a chat format that renders
/// nothing. A chat format's export parses and builds tool constraints from
/// `format.parser` with the shared parsers whatever its renderer, so this
/// one parses as the model's own export does.
pub(crate) fn parser_export_v1() -> FrontendExportV1 {
    frontend_format::chat_format_export_v1::<ParserOnlyFormatV1>()
}

/// A chat format with no renderer: [`parser_export_v1`].
struct ParserOnlyFormatV1;

impl frontend_format::ChatFormatV1 for ParserOnlyFormatV1 {
    fn open(_: &frontend_format::FormatDescriptorV1) -> Result<Self, String> {
        Ok(Self)
    }

    fn opened(&self) -> runtime_bundle_abi::frontend_document::OpenedV1 {
        runtime_bundle_abi::frontend_document::OpenedV1::default()
    }

    fn render(
        &self,
        _: frontend_format::FormatRenderRequestV1<'_>,
    ) -> Result<frontend_format::FormatRenderedV1, frontend_format::FrontendErrorV1> {
        Err(frontend_format::FrontendErrorV1::new(
            "this test export only parses; the model's own export renders",
            None,
        ))
    }
}

/// The shared parser, called directly.
pub(crate) struct LibraryParserV1(frontend_format::DeclaredParserV1);

impl LibraryParserV1 {
    /// The shared parser over `descriptor`'s format, which is read and
    /// not checked: a decoder without a parser source may drive a
    /// declaration a test assembled.
    pub(crate) fn new(
        descriptor: &FrontendDescriptorV1,
        options: FormatParserOptionsV1,
    ) -> Result<Self, String> {
        let descriptor = frontend_format::FormatDescriptorV1::read(descriptor.clone())
            .map_err(|error| error.to_string())?;
        let FormatParserOptionsV1 {
            mode,
            reasoning_open,
            extract_tools,
            stream_tool_calls,
            raw,
            ignore_eos,
            engine_tokens,
            tool_schemas,
        } = options;
        Ok(Self(frontend_format::DeclaredParserV1::new(
            Arc::new(descriptor),
            frontend_format::FormatParserOptionsV1 {
                mode,
                reasoning_open,
                extract_tools,
                stream_tool_calls,
                raw,
                ignore_eos,
                engine_tokens,
                tool_schemas: tool_schemas.iter().map(tool_v1).collect(),
            },
        )))
    }
}

impl FormatParserV1 for LibraryParserV1 {
    fn push(&mut self, input: FormatInputV1<'_>) -> Result<Vec<FormatEventV1>, String> {
        let FormatInputV1 {
            completion,
            held,
            last,
            stop_floor,
        } = input;
        let events = self.0.push(frontend_format::FormatInputV1 {
            completion,
            held,
            last,
            stop_floor,
        })?;
        Ok(events.into_iter().map(event_v1).collect())
    }

    fn cursor(&self) -> usize {
        self.0.cursor()
    }
}

fn event_v1(event: frontend_format::FormatEventV1) -> FormatEventV1 {
    match event {
        frontend_format::FormatEventV1::Reasoning(range) => FormatEventV1::Reasoning(range),
        frontend_format::FormatEventV1::Content(range) => FormatEventV1::Content(range),
        frontend_format::FormatEventV1::ToolCalls { calls, range } => FormatEventV1::ToolCalls {
            calls: calls.into_iter().map(call_v1).collect(),
            range,
        },
        frontend_format::FormatEventV1::ToolCallDeltas(deltas) => FormatEventV1::ToolCallDeltas(
            deltas
                .into_iter()
                .map(
                    |frontend_format::FormatToolCallDeltaV1 {
                         index,
                         name,
                         arguments,
                     }| FormatToolCallDeltaV1 {
                        index,
                        name,
                        arguments,
                    },
                )
                .collect(),
        ),
        frontend_format::FormatEventV1::Finish { reason, completion } => FormatEventV1::Finish {
            reason,
            completion: completion.map(completion_v1),
        },
    }
}

fn call_v1(call: frontend_format::ParsedToolCallV1) -> ParsedToolCallV1 {
    let frontend_format::ParsedToolCallV1 { name, arguments } = call;
    ParsedToolCallV1 { name, arguments }
}

/// Reads one whole completion with the shared parser, and reports it as a
/// decoder does, for tests that compare the two.
pub fn parse_completion_v1(
    descriptor: &FrontendDescriptorV1,
    text: &str,
    mode: PromptModeV1,
) -> Result<AssistantCompletionV1, String> {
    let descriptor = frontend_format::FormatDescriptorV1::read(descriptor.clone())
        .map_err(|error| error.to_string())?;
    frontend_format::parse_completion_v1(&descriptor, text, mode).map(completion_v1)
}

fn completion_v1(completion: frontend_format::AssistantCompletionV1) -> AssistantCompletionV1 {
    let frontend_format::AssistantCompletionV1 {
        content,
        reasoning_content,
        tool_calls,
    } = completion;
    AssistantCompletionV1 {
        content,
        reasoning_content,
        tool_calls: tool_calls.into_iter().map(call_v1).collect(),
    }
}

/// One offered tool, as the library reads it. Each conversion from a host
/// type names every field, so a field added to one fails to compile here.
fn tool_v1(tool: &runtime_server::FrontendToolV1) -> frontend_format::FrontendToolV1 {
    let runtime_server::FrontendToolV1 {
        name,
        description,
        parameters,
        parameters_present,
        strict,
    } = tool;
    frontend_format::FrontendToolV1 {
        name: name.clone(),
        description: description.clone(),
        parameters: json_v1(parameters),
        parameters_present: *parameters_present,
        strict: *strict,
    }
}

/// One JSON value, as the library reads it: the same tree, keys in order and
/// numbers unchanged.
fn json_v1(value: &runtime_server::FrontendJsonV1) -> frontend_format::FrontendJsonV1 {
    use frontend_format::FrontendJsonV1 as Library;
    use runtime_server::FrontendJsonV1 as Host;
    match value {
        Host::Null => Library::Null,
        Host::Bool(flag) => Library::Bool(*flag),
        Host::Number(number) => Library::Number(number.clone()),
        Host::String(text) => Library::String(text.clone()),
        Host::Array(items) => Library::Array(items.iter().map(json_v1).collect()),
        Host::Object(entries) => Library::Object(
            entries
                .iter()
                .map(|(name, value)| (name.clone(), json_v1(value)))
                .collect(),
        ),
    }
}

#[cfg(any(test, feature = "test-support"))]
pub(crate) use renderer::LibraryRendererV1;

#[cfg(any(test, feature = "test-support"))]
mod renderer {
    use std::sync::Arc;

    use runtime_server::{
        FrontendContentPartV1, FrontendErrorV1, FrontendJsonV1, FrontendMediaPartKindV1,
        FrontendMediaPartV1, FrontendMessageV1, FrontendRoleV1, FrontendToolCallV1,
        FrontendToolChoiceV1, FrontendToolV1, PromptSegmentV1,
    };

    use super::{json_v1, tool_v1};
    use crate::descriptor::{FrontendComponentErrorV1, FrontendDescriptorV1};
    use crate::format::{FormatRenderRequestV1, FormatRenderedV1, FormatRendererV1};

    /// A model's chat format called directly: the library path the
    /// document path is compared against.
    #[derive(Debug)]
    pub(crate) struct LibraryRendererV1(Box<dyn LibraryFormatV1>);

    impl LibraryRendererV1 {
        /// The chat format `F` opened over `descriptor`, a `format.parser`
        /// declaration this checks.
        pub(crate) fn chat_format<F: frontend_format::ChatFormatV1>(
            descriptor: &FrontendDescriptorV1,
        ) -> Result<Self, FrontendComponentErrorV1> {
            let descriptor = frontend_format::FormatDescriptorV1::new(descriptor.clone())?;
            let format = F::open(&descriptor).map_err(FrontendComponentErrorV1::Invalid)?;
            Ok(Self(Box::new(LibraryChatFormatV1 {
                descriptor: Arc::new(descriptor),
                format,
            })))
        }
    }

    /// What the library path calls: the library's types on both sides.
    trait LibraryFormatV1: std::fmt::Debug + Send + Sync {
        fn render(
            &self,
            request: frontend_format::FormatRenderRequestV1<'_>,
        ) -> Result<frontend_format::FormatRenderedV1, frontend_format::FrontendErrorV1>;

        fn tool_constraint(
            &self,
            tools: &[frontend_format::FrontendToolV1],
            choice: &frontend_format::FrontendToolChoiceV1,
        ) -> Result<Option<runtime_grammar::Grammar>, frontend_format::FrontendErrorV1>;

        fn opened(&self) -> runtime_bundle_abi::frontend_document::OpenedV1;
    }

    /// A chat format called directly, with what its export adds to a
    /// render: the export decides `reasoning_open` from the prompt with the
    /// parser section's markers, and refuses a render that leaves parser
    /// state (`frontend_format::ChatFormatExportV1`). Its tool constraints
    /// come from the parser section, as the export's do.
    struct LibraryChatFormatV1<F> {
        descriptor: Arc<frontend_format::FormatDescriptorV1>,
        format: F,
    }

    impl<F> std::fmt::Debug for LibraryChatFormatV1<F> {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter
                .debug_struct("LibraryChatFormatV1")
                .field("format", &std::any::type_name::<F>())
                .finish_non_exhaustive()
        }
    }

    impl<F: frontend_format::ChatFormatV1> LibraryFormatV1 for LibraryChatFormatV1<F> {
        fn render(
            &self,
            request: frontend_format::FormatRenderRequestV1<'_>,
        ) -> Result<frontend_format::FormatRenderedV1, frontend_format::FrontendErrorV1> {
            let continue_final_message = request.continue_final_message;
            let mut rendered = self.format.render(request)?;
            if !rendered.parser_state.is_empty() {
                return Err(frontend_format::FrontendErrorV1::new(
                    "a chat format render leaves no parser state: the shared parser reads \
                     format.parser",
                    None,
                ));
            }
            let parser = &self.descriptor.parser;
            rendered.reasoning_open = frontend_format::toolkit::reasoning_open_v1(
                rendered.mode,
                continue_final_message,
                &parser.reasoning_open,
                &parser.reasoning_close,
                &rendered.prompt,
            );
            Ok(rendered)
        }

        fn tool_constraint(
            &self,
            tools: &[frontend_format::FrontendToolV1],
            choice: &frontend_format::FrontendToolChoiceV1,
        ) -> Result<Option<runtime_grammar::Grammar>, frontend_format::FrontendErrorV1> {
            frontend_format::tool_constraint_v1(&self.descriptor, tools, choice)
        }

        fn opened(&self) -> runtime_bundle_abi::frontend_document::OpenedV1 {
            self.format.opened()
        }
    }

    impl FormatRendererV1 for LibraryRendererV1 {
        fn render(
            &self,
            request: FormatRenderRequestV1<'_>,
        ) -> Result<FormatRenderedV1, FrontendErrorV1> {
            let FormatRenderRequestV1 {
                messages,
                continue_final_message,
                add_generation_prompt,
                tools,
                tool_objects,
                response_format,
                reasoning_effort,
                enable_thinking,
                thinking,
                nested_reasoning_effort,
                preserve_thinking,
                clear_thinking,
                add_vision_id,
                // The library's request carries its keywords typed.
                chat_template_kwargs: _,
            } = request;
            // The library's request carries only the common roles.
            if let Some(name) = messages.iter().find_map(|message| match &message.role {
                FrontendRoleV1::Declared(name) => Some(name),
                _ => None,
            }) {
                return Err(FrontendErrorV1::new(
                    format!("message role '{name}' is not supported"),
                    Some("messages"),
                ));
            }
            let tools: Vec<_> = tools.iter().map(tool_v1).collect();
            let json = |value: Option<FrontendJsonV1>| value.as_ref().map(json_v1);
            let rendered = self
                .0
                .render(frontend_format::FormatRenderRequestV1 {
                    messages: messages.iter().map(message_v1).collect(),
                    continue_final_message,
                    add_generation_prompt,
                    tools: &tools,
                    tool_objects: tool_objects.iter().map(json_v1).collect(),
                    response_format: json(response_format),
                    reasoning_effort,
                    enable_thinking,
                    thinking,
                    nested_reasoning_effort,
                    preserve_thinking,
                    clear_thinking: json(clear_thinking),
                    add_vision_id: json(add_vision_id),
                })
                .map_err(error_v1)?;
            let frontend_format::FormatRenderedV1 {
                prompt,
                segments,
                media,
                mode,
                effort,
                reasoning_open,
                parser_state,
            } = rendered;
            Ok(FormatRenderedV1 {
                prompt,
                segments: segments
                    .into_iter()
                    .map(|segment| match segment {
                        frontend_format::PromptSegmentV1::Text(text) => PromptSegmentV1::Text(text),
                        frontend_format::PromptSegmentV1::Media(index) => {
                            PromptSegmentV1::Media(index)
                        }
                    })
                    .collect(),
                media: media.into_iter().map(media_from_v1).collect(),
                mode,
                effort,
                reasoning_open,
                parser_state,
            })
        }

        fn tool_constraint(
            &self,
            tools: &[FrontendToolV1],
            choice: &FrontendToolChoiceV1,
        ) -> Result<Option<runtime_grammar::Grammar>, FrontendErrorV1> {
            let tools: Vec<_> = tools.iter().map(tool_v1).collect();
            let choice = match choice {
                FrontendToolChoiceV1::Auto => frontend_format::FrontendToolChoiceV1::Auto,
                FrontendToolChoiceV1::None => frontend_format::FrontendToolChoiceV1::None,
                FrontendToolChoiceV1::Required => frontend_format::FrontendToolChoiceV1::Required,
                FrontendToolChoiceV1::Named(name) => {
                    frontend_format::FrontendToolChoiceV1::Named(name.clone())
                }
            };
            self.0.tool_constraint(&tools, &choice).map_err(error_v1)
        }

        fn opened(&self) -> runtime_bundle_abi::frontend_document::OpenedV1 {
            self.0.opened()
        }
    }

    fn error_v1(error: frontend_format::FrontendErrorV1) -> FrontendErrorV1 {
        let frontend_format::FrontendErrorV1 { message, parameter } = error;
        FrontendErrorV1 { message, parameter }
    }

    fn message_v1(message: &FrontendMessageV1) -> frontend_format::FrontendMessageV1 {
        let FrontendMessageV1 {
            role,
            content,
            parts,
            reasoning_content,
            tool_calls,
            tool_call_id,
            tools,
        } = message;
        frontend_format::FrontendMessageV1 {
            role: match role {
                FrontendRoleV1::System => frontend_format::FrontendRoleV1::System,
                FrontendRoleV1::Developer => frontend_format::FrontendRoleV1::Developer,
                FrontendRoleV1::User => frontend_format::FrontendRoleV1::User,
                FrontendRoleV1::Assistant => frontend_format::FrontendRoleV1::Assistant,
                FrontendRoleV1::Tool => frontend_format::FrontendRoleV1::Tool,
                FrontendRoleV1::Declared(_) => {
                    unreachable!("a declared role is refused before conversion")
                }
            },
            content: content.clone(),
            parts: parts.as_ref().map(|parts| {
                parts
                    .iter()
                    .map(|part| match part {
                        FrontendContentPartV1::Text(text) => {
                            frontend_format::FrontendContentPartV1::Text(text.clone())
                        }
                        FrontendContentPartV1::Media(media) => {
                            frontend_format::FrontendContentPartV1::Media(media_v1(media))
                        }
                        FrontendContentPartV1::ToolReference { name } => {
                            frontend_format::FrontendContentPartV1::ToolReference {
                                name: name.clone(),
                            }
                        }
                    })
                    .collect()
            }),
            reasoning_content: reasoning_content.clone(),
            tool_calls: tool_calls
                .iter()
                .map(
                    |FrontendToolCallV1 {
                         id,
                         name,
                         arguments,
                     }| {
                        frontend_format::FrontendToolCallV1 {
                            id: id.clone(),
                            name: name.clone(),
                            arguments: arguments.clone(),
                        }
                    },
                )
                .collect(),
            tool_call_id: tool_call_id.clone(),
            tools: tools.iter().map(json_v1).collect(),
        }
    }

    fn media_kind_v1(kind: FrontendMediaPartKindV1) -> frontend_format::FrontendMediaPartKindV1 {
        match kind {
            FrontendMediaPartKindV1::Image => frontend_format::FrontendMediaPartKindV1::Image,
            FrontendMediaPartKindV1::Audio => frontend_format::FrontendMediaPartKindV1::Audio,
            FrontendMediaPartKindV1::Video => frontend_format::FrontendMediaPartKindV1::Video,
        }
    }

    fn media_v1(media: &FrontendMediaPartV1) -> frontend_format::FrontendMediaPartV1 {
        let FrontendMediaPartV1 {
            kind,
            part_type,
            source,
            format,
        } = media;
        frontend_format::FrontendMediaPartV1 {
            kind: media_kind_v1(*kind),
            part_type: part_type.clone(),
            source: source.clone(),
            format: format.clone(),
        }
    }

    fn media_from_v1(media: frontend_format::FrontendMediaPartV1) -> FrontendMediaPartV1 {
        let frontend_format::FrontendMediaPartV1 {
            kind,
            part_type,
            source,
            format,
        } = media;
        FrontendMediaPartV1 {
            kind: match kind {
                frontend_format::FrontendMediaPartKindV1::Image => FrontendMediaPartKindV1::Image,
                frontend_format::FrontendMediaPartKindV1::Audio => FrontendMediaPartKindV1::Audio,
                frontend_format::FrontendMediaPartKindV1::Video => FrontendMediaPartKindV1::Video,
            },
            part_type,
            source,
            format,
        }
    }
}
