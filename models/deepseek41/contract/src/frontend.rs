//! DeepSeek V4.1 Flash's chat format, ported from the encoder pinned with
//! the weights.
//!
//! The checkpoint ships no chat template. Its prompt-format reference is
//! `encoding/encoding.py` at revision 1bae0e10 (990 lines, sha256
//! 13231c59...99e2), which [`encoding`] ports function by function, citing
//! its lines. [`DeepSeekV41Format`] serves that port: it reads a request into
//! the messages the encoder reads and returns the encoder's prompt. No Jinja
//! and no Python runs at serving time.
//!
//! The request reaches the encoder as follows (the corpus reference in
//! `models/deepseek41/frontend_cases.py` maps its HF input the same way):
//! - Thinking mode and the effort come from the declared reasoning controls
//!   (`select_mode_v1`). The effort is passed in thinking mode only, where
//!   the encoder writes it; a chat prompt never reads it.
//! - The request's tools and `response_format` belong to the leading system
//!   message, where the encoder renders them (vendor README, "Tools are
//!   defined on the `system` message"). A request without one gets an empty
//!   leading system message to carry them.
//! - `continue_final_message` renders the final assistant turn without its
//!   end-of-sentence token (`wo_eos`). The encoder writes the generation
//!   header itself after a final user or later system message, so
//!   `add_generation_prompt` changes nothing.
//! - `preserve_thinking` turns off `drop_thinking`, which the encoder also
//!   turns off whenever tools are declared.
//! - A user's media parts are image blocks; images are served only when the
//!   declaration declares them, with the encoder's placeholder as the pad.
//!
//! The roles the encoder reads that the transport carries are system, user,
//! assistant and tool. A developer message, a message's own tools on a role
//! other than system, and media outside a user message are refused (see
//! `models/deepseek41/frontend-divergences.json`). The encoder's
//! `latest_reminder` role and `task` flag are not served.

pub mod encoding;

use frontend_format::toolkit::{
    field_v1, media_kind_v1, media_placeholder_v1, resource_v1, screen_media_text_v1,
    select_mode_v1, split_media_segments_v1, validate_name_v1,
};
use frontend_format::{
    ChatFormatV1, ChatRenderV1, FormatDescriptorV1, FormatRenderRequestV1, FormatRenderedV1,
    FrontendContentPartV1, FrontendErrorV1, FrontendJsonEncodingV1, FrontendJsonV1,
    FrontendMediaKindV1, FrontendMessageV1, FrontendParserDeclarationV1, FrontendRoleV1,
    FrontendToolGrammarV1, FrontendToolV1, OpenedV1, PromptModeV1,
};

use encoding::{Block, Message, ReasoningEffort, Role, ThinkingMode, ToolCall};

/// The efforts the encoder names (lines 448-453), in its order.
const EFFORTS: [&str; 3] = ["low", "high", "max"];

/// The reasoning keywords this format reads to switch thinking.
const THINKING_KEYWORDS: [&str; 2] = ["enable_thinking", "thinking"];

/// The parser section for the markers this renderer writes.
///
/// The encoder writes tool calls in DSML (lines 428-437, 524-555): a
/// `calls` block of `invoke` elements whose parameters carry
/// `string="true|false"`. Neither shared tool grammar reads that flag, so
/// until the DSML parser family exists this declares the function-parameter
/// grammar over the same markers, which reads string parameters only. Tool
/// calls with non-string arguments do not parse back until then.
#[must_use]
pub fn parser_declaration() -> FrontendParserDeclarationV1 {
    let text = |value: &str| value.to_owned();
    FrontendParserDeclarationV1 {
        tool_grammar: FrontendToolGrammarV1::XmlFunctionParametersV1,
        json_encoding: FrontendJsonEncodingV1::PythonJsonV1,
        reasoning_open: text(encoding::THINKING_START),
        reasoning_close: text(encoding::THINKING_END),
        answer_prefix: String::new(),
        tool_call_open: text("\n\n<｜DSML｜ calls>\n"),
        tool_call_close: text("</｜DSML｜ calls>"),
        tool_call_separator: String::new(),
        function_open: text("<｜DSML｜ invoke name=\""),
        function_name_close: text("\">\n"),
        function_close: text("</｜DSML｜ invoke>\n"),
        parameter_open: text("<｜DSML｜ parameter name=\""),
        parameter_name_close: text("\" string=\"true\">"),
        parameter_value_open: String::new(),
        parameter_close: text("</｜DSML｜ parameter>\n"),
        drop_blank_tool_call_content: false,
        reserved_markers: Vec::new(),
        turn_open: [encoding::ASSISTANT, encoding::SYSTEM, encoding::USER]
            .map(text)
            .to_vec(),
        assistant_header: text(encoding::ASSISTANT),
    }
}

/// The preamble a declared effort names: the encoder's effort prefix for
/// its budget without the blank line after it (lines 443-446).
#[must_use]
pub fn effort_preamble(name: &str) -> Option<String> {
    let budget = encoding::budget_of(name)?;
    let text = encoding::reasoning_effort_text(budget);
    Some(text.trim_end_matches('\n').to_owned())
}

/// Why `descriptor`'s reasoning controls or parser section are not the ones
/// this renderer writes its prompt for, if they are not.
fn reasoning_mismatch(descriptor: &FormatDescriptorV1) -> Option<String> {
    let reasoning = &descriptor.reasoning;
    let names = reasoning.effort_names();
    if names != EFFORTS {
        return Some(format!(
            "declares efforts {names:?}; the encoder reads {EFFORTS:?}"
        ));
    }
    if let Some(effort) = reasoning
        .efforts
        .iter()
        .find(|effort| Some(&effort.preamble) != effort_preamble(&effort.name).as_ref())
    {
        return Some(format!(
            "declares effort '{}' with preamble {:?}; the encoder writes {:?}",
            effort.name,
            effort.preamble,
            effort_preamble(&effort.name)
        ));
    }
    if reasoning.default_effort != encoding::DEFAULT_REASONING_EFFORT {
        return Some(format!(
            "declares default effort '{}'; the encoder's default is '{}'",
            reasoning.default_effort,
            encoding::DEFAULT_REASONING_EFFORT
        ));
    }
    if reasoning.thinking_keywords != THINKING_KEYWORDS {
        return Some(format!(
            "declares thinking keywords {:?}; this format reads {THINKING_KEYWORDS:?}",
            reasoning.thinking_keywords
        ));
    }
    // The effort use as the declaration spells it.
    let effort_use = serde_json::to_value(reasoning.effort_use).ok()?;
    if effort_use != "when-thinking" {
        return Some(format!(
            "declares effort use {effort_use}; the encoder writes the effort only while thinking"
        ));
    }
    if descriptor.parser != parser_declaration() {
        return Some(
            "declares a format.parser section other than the markers this renderer writes".into(),
        );
    }
    None
}

/// DeepSeek V4.1 Flash's chat format over its checked declaration.
#[derive(Debug)]
pub struct DeepSeekV41Format {
    descriptor: FormatDescriptorV1,
}

impl ChatFormatV1 for DeepSeekV41Format {
    /// Refuses a declaration this encoder cannot render: reasoning controls
    /// or a parser section other than the ones the renderer writes its prompt
    /// for, or a served media kind other than images with the encoder's
    /// placeholder as their pad.
    fn open(descriptor: &FormatDescriptorV1) -> Result<Self, String> {
        if let Some(mismatch) = reasoning_mismatch(descriptor) {
            return Err(format!(
                "DeepSeek V4.1 Flash's chat format: the declaration {mismatch}"
            ));
        }
        for modality in descriptor
            .multimodal
            .iter()
            .flat_map(|multimodal| &multimodal.modalities)
        {
            if modality.kind != FrontendMediaKindV1::Image {
                return Err("DeepSeek V4.1 Flash's encoder writes images only".into());
            }
            if modality.render.pad != encoding::IMAGE_PLACEHOLDER {
                return Err(format!(
                    "DeepSeek V4.1 Flash's encoder writes images as {}, not {}",
                    encoding::IMAGE_PLACEHOLDER,
                    modality.render.pad
                ));
            }
        }
        Ok(Self {
            descriptor: descriptor.clone(),
        })
    }

    /// The encoder reads each tool's function object as sent (lines 84-98)
    /// and every assistant turn's `reasoning_content` (line 606).
    fn opened(&self) -> OpenedV1 {
        OpenedV1 {
            reads_tool_objects: true,
            reasoning_content_fallback: true,
            tool_call_reasoning_fallback: false,
            extra_roles: Vec::new(),
            tool_references: false,
        }
    }

    fn render(
        &self,
        request: FormatRenderRequestV1<'_>,
    ) -> Result<FormatRenderedV1, FrontendErrorV1> {
        let refuse = |message: String| FrontendErrorV1::new(message, Some("messages"));
        let (mode, effort) = select_mode_v1(&self.descriptor.reasoning, &request)?;
        let thinking_mode = match mode {
            PromptModeV1::Thinking => ThinkingMode::Thinking,
            PromptModeV1::Chat => ThinkingMode::Chat,
        };
        let effort_name = effort.name.clone();
        let effort =
            (mode == PromptModeV1::Thinking).then(|| ReasoningEffort::Named(effort_name.clone()));
        let drop_thinking = !request.preserve_thinking.unwrap_or(false);
        let mut media = Vec::new();
        let messages = self.messages(&request, &mut media).map_err(refuse)?;
        let (prompt, order) = encoding::encode_messages(
            &messages,
            thinking_mode,
            drop_thinking,
            effort.as_ref(),
            &media_placeholder_v1,
        )
        .map_err(|error| refuse(field_v1(error.0)))?;
        if prompt.len() > self.descriptor.limits.maximum_prompt_bytes {
            return Err(refuse(resource_v1("prompt exceeds the byte limit")));
        }
        let media = order
            .into_iter()
            .map(|index| media[index].clone())
            .collect();
        let (prompt, segments, media) =
            split_media_segments_v1(&self.descriptor, prompt, media).map_err(refuse)?;
        Ok(FormatRenderedV1 {
            prompt,
            segments,
            media,
            mode,
            effort: effort_name,
            // The export decides it from the prompt.
            reasoning_open: false,
            parser_state: Vec::new(),
        })
    }
}

impl DeepSeekV41Format {
    /// The request as the encoder's messages, after the bounds every format
    /// charges: message, tool, call and parameter counts, names in the
    /// declared byte set, and no media boundary in client text. `media`
    /// receives the request's media parts, which image blocks index.
    fn messages(
        &self,
        request: &FormatRenderRequestV1<'_>,
        media: &mut Vec<frontend_format::FrontendMediaPartV1>,
    ) -> Result<Vec<Message>, String> {
        let descriptor = &self.descriptor;
        let limits = descriptor.limits;
        if request.messages.len() > limits.maximum_messages
            || request.tools.len() > limits.maximum_tools
        {
            return Err(resource_v1("too many messages or tools"));
        }
        if request.messages.is_empty() {
            return Err(field_v1("at least one message is required"));
        }
        let chat = request
            .messages
            .iter()
            .cloned()
            .map(frontend_format::toolkit::chat_message_v1)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.message)?;
        screen_media_text_v1(
            descriptor,
            &ChatRenderV1 {
                mode: PromptModeV1::Chat,
                preamble: "",
                messages: &chat,
                continue_final_message: request.continue_final_message,
                add_generation_prompt: request.add_generation_prompt,
                tools: request.tools,
                response_format: request.response_format.as_ref(),
                preserve_thinking: true,
            },
            &request.tool_objects,
        )?;
        let mut names = std::collections::BTreeSet::new();
        for tool in request.tools {
            validate_name_v1(descriptor, &tool.name, "tool")?;
            if !names.insert(tool.name.as_str()) {
                return Err(field_v1(format!("duplicate tool name '{}'", tool.name)));
            }
        }
        let mut messages = request
            .messages
            .iter()
            .map(|message| self.message(message, media))
            .collect::<Result<Vec<_>, _>>()?;
        if request.continue_final_message {
            match messages.last_mut() {
                Some(last) if last.role == Role::Assistant => last.wo_eos = true,
                _ => {
                    return Err(field_v1(
                        "continue_final_message requires a final assistant message",
                    ));
                }
            }
        }
        let tools = tool_objects(request.tools, &request.tool_objects)?;
        if !tools.is_empty() || request.response_format.is_some() {
            if messages[0].role != Role::System {
                messages.insert(0, Message::new(Role::System, Some(String::new())));
            }
            let system = &mut messages[0];
            if !tools.is_empty() {
                if !system.tools.is_empty() {
                    return Err(field_v1(
                        "tools are offered both by the request and by its system message",
                    ));
                }
                system.tools = tools;
            }
            system.response_format = request.response_format.clone();
        }
        Ok(messages)
    }

    /// One request message as the encoder reads it.
    fn message(
        &self,
        message: &FrontendMessageV1,
        media: &mut Vec<frontend_format::FrontendMediaPartV1>,
    ) -> Result<Message, String> {
        let descriptor = &self.descriptor;
        let limits = descriptor.limits;
        let role = match message.role {
            FrontendRoleV1::System => Role::System,
            FrontendRoleV1::User => Role::User,
            FrontendRoleV1::Assistant => Role::Assistant,
            FrontendRoleV1::Tool => Role::Tool,
            FrontendRoleV1::Developer => {
                return Err(field_v1("developer role is not declared"));
            }
        };
        if !message.tools.is_empty() && role != Role::System {
            return Err(field_v1(
                "message tools are declared only on system messages",
            ));
        }
        let mut result = Message::new(role, message.content.clone());
        if let Some(parts) = &message.parts {
            let mut blocks = Vec::new();
            for part in parts {
                match part {
                    FrontendContentPartV1::Text(text) => blocks.push(Block::Text(text.clone())),
                    FrontendContentPartV1::Media(item) if role == Role::User => {
                        media_kind_v1(descriptor, item)?;
                        blocks.push(Block::Image(media.len()));
                        media.push(item.clone());
                    }
                    FrontendContentPartV1::Media(_) => {
                        return Err(field_v1("media parts are served in user messages only"));
                    }
                    FrontendContentPartV1::ToolReference { .. } => {
                        return Err(frontend_format::TOOL_REFERENCES_UNSUPPORTED_V1.into());
                    }
                }
            }
            if role == Role::User {
                result.blocks = Some(blocks);
                result.content = None;
            } else {
                // Lines 387-393: another role's list content is its text
                // blocks joined.
                let texts: Vec<String> = blocks
                    .into_iter()
                    .map(|block| match block {
                        Block::Text(text) => text,
                        _ => String::new(),
                    })
                    .collect();
                result.content = Some(texts.join("\n\n"));
            }
        }
        result.reasoning_content = message.reasoning_content.clone();
        result.tool_call_id = message.tool_call_id.clone();
        if message.tool_calls.len() > limits.maximum_tool_calls {
            return Err(resource_v1("too many tool calls"));
        }
        for call in &message.tool_calls {
            validate_name_v1(descriptor, &call.name, "tool call")?;
            if call.arguments.len() > limits.maximum_json_bytes {
                return Err(resource_v1("JSON input exceeds the byte limit"));
            }
            if let Ok(FrontendJsonV1::Object(entries)) =
                serde_json::from_str::<FrontendJsonV1>(&call.arguments)
                && entries.len() > limits.maximum_tool_parameters
            {
                return Err(resource_v1("too many tool parameters"));
            }
            result.tool_calls.push(ToolCall {
                id: call.id.clone(),
                name: call.name.clone(),
                namespace: None,
                arguments: call.arguments.clone(),
            });
        }
        result.tools = message
            .tools
            .iter()
            .map(|tool| wrap_function(tool.clone()))
            .collect();
        Ok(result)
    }
}

/// The offered tools as OpenAI tool objects around each function object
/// the client sent, or around the declared fields when the request carries
/// no objects (a request not read through the transport).
fn tool_objects(
    tools: &[FrontendToolV1],
    objects: &[FrontendJsonV1],
) -> Result<Vec<FrontendJsonV1>, String> {
    if objects.is_empty() {
        return Ok(tools
            .iter()
            .map(|tool| wrap_function(FrontendJsonV1::Object(declared_function(tool))))
            .collect());
    }
    if objects.len() != tools.len() {
        return Err(field_v1("every offered tool needs its function object"));
    }
    objects
        .iter()
        .map(|object| match object {
            FrontendJsonV1::Object(_) => Ok(wrap_function(object.clone())),
            _ => Err(field_v1("a tool's function must be an object")),
        })
        .collect()
}

/// A function object as the OpenAI tool object the encoder reads.
fn wrap_function(function: FrontendJsonV1) -> FrontendJsonV1 {
    // A message's own tools already arrive as tool objects.
    if let FrontendJsonV1::Object(entries) = &function
        && entries.iter().any(|(key, _)| key == "function")
    {
        return function;
    }
    FrontendJsonV1::Object(vec![
        ("type".to_owned(), FrontendJsonV1::String("function".into())),
        ("function".to_owned(), function),
    ])
}

/// A tool's function object from its declared fields: the name, and the
/// description and parameters it has.
fn declared_function(tool: &FrontendToolV1) -> Vec<(String, FrontendJsonV1)> {
    let mut function = vec![("name".to_owned(), FrontendJsonV1::String(tool.name.clone()))];
    if let Some(description) = &tool.description {
        function.push((
            "description".to_owned(),
            FrontendJsonV1::String(description.clone()),
        ));
    }
    if tool.parameters_present {
        function.push(("parameters".to_owned(), tool.parameters.clone()));
    }
    function
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effort_preambles_are_the_encoder_prefix() {
        assert_eq!(
            effort_preamble("high").unwrap(),
            "Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the \
             reasoning)"
        );
        assert_eq!(effort_preamble("medium"), None);
    }
}
