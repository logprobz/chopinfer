//! MiMo v2.6's chat format: a structural port of the checkpoint's chat
//! template, `chat_template.jinja` at the revision and sha256
//! `fixtures/mimo26/frontend/provenance.json` pins (95 lines).
//!
//! Each function ports one macro or block and cites the template lines it
//! ports. HF's `apply_chat_template` wraps the template: it ends a
//! continued final message where its content ends, and the processor
//! expands each media pad. The bundle exports this format with
//! `frontend_format::export_chat_format_v1!`; the declaration's
//! `format.parser` section reads its completions with the shared
//! function-parameter parser.

use std::sync::Arc;

use frontend_format::toolkit::{
    append_v1, field_v1, media_kind_v1, media_placeholder_v1, parse_json_object_v1,
    reserved_media_text_v1, resource_v1, screen_media_text_v1, select_mode_v1,
    split_media_segments_v1, tojson_v1, validate_name_v1,
};
use frontend_format::{
    AssistantTurnV1, ChatFormatV1, ChatMessageV1, ChatRenderV1, FormatDescriptorV1,
    FormatRenderRequestV1, FormatRenderedV1, FrontendContentPartV1, FrontendEffortUseV1,
    FrontendErrorV1, FrontendJsonEncodingV1, FrontendJsonV1, FrontendMediaPartKindV1,
    FrontendMediaPartV1, FrontendMessageV1, FrontendParserDeclarationV1, FrontendRoleV1,
    FrontendToolCallV1, FrontendToolGrammarV1, FrontendToolV1, OpenedV1, PromptModeV1,
    TOOL_REFERENCES_UNSUPPORTED_V1, ToolResultV1,
};

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

/// The labels `render_content` writes around each media kind's pad (lines
/// 16-21), by kind, with the pad the label holds.
///
/// Line 21 writes `<|vision_start|><|video_pad|><|vision_end|>`, and HF's
/// processor replaces the pad with `<|mimo_video_start|>`, the timestamped
/// frames and `<|mimo_video_end|>` (`fixtures/mimo26/capture/frontend_media.py`).
/// The host expands the pad into the frames alone, so the label carries the
/// processor's wrapper and the served tokens are the processor's.
const MEDIA_LABELS: [(FrontendMediaPartKindV1, &str, &str, &str, &str); 3] = [
    (
        FrontendMediaPartKindV1::Image,
        "image",
        "<|vision_start|>",
        "<|image_pad|>",
        "<|vision_end|>",
    ),
    (
        FrontendMediaPartKindV1::Audio,
        "audio",
        "<|mimo_audio_start|>",
        "<|audio_pad|>",
        "<|mimo_audio_end|>",
    ),
    (
        FrontendMediaPartKindV1::Video,
        "video",
        "<|vision_start|><|mimo_video_start|>",
        "<|video_pad|>",
        "<|mimo_video_end|><|vision_end|>",
    ),
];

/// The `format.parser` section MiMo v2.6's completions read through: the
/// markers the template writes around reasoning (lines 44, 63 and 93) and
/// tool calls (lines 52 and 56), its arguments in Python's JSON (`tojson`),
/// and the boundary token that opens every turn.
#[must_use]
pub fn parser_declaration() -> FrontendParserDeclarationV1 {
    FrontendParserDeclarationV1 {
        tool_grammar: FrontendToolGrammarV1::XmlFunctionParametersV1,
        json_encoding: FrontendJsonEncodingV1::PythonJsonV1,
        reasoning_open: THINK_OPEN.into(),
        reasoning_close: THINK_CLOSE.into(),
        answer_prefix: String::new(),
        tool_call_open: "<tool_call>".into(),
        tool_call_close: "</tool_call>".into(),
        tool_call_separator: String::new(),
        function_open: "<function=".into(),
        function_name_close: ">".into(),
        function_close: "</function>".into(),
        parameter_open: "<parameter=".into(),
        parameter_name_close: ">".into(),
        parameter_json_name_close: String::new(),
        parameter_value_open: String::new(),
        parameter_close: "</parameter>".into(),
        drop_blank_tool_call_content: false,
        reserved_markers: Vec::new(),
        turn_open: vec![IM_START.into()],
        assistant_header: format!("{IM_START}assistant"),
    }
}

/// MiMo v2.6's chat template, ported.
#[derive(Debug)]
pub struct Mimo26Format {
    descriptor: Arc<FormatDescriptorV1>,
    /// The prefix and suffix of each media kind the declaration serves.
    labels: Vec<(FrontendMediaPartKindV1, &'static str, &'static str)>,
}

impl ChatFormatV1 for Mimo26Format {
    /// Opens over a declaration whose parser section is the one the
    /// template writes ([`parser_declaration`]) and whose reasoning controls
    /// are the ones it reads, labelling exactly the media kinds it serves,
    /// each with the pad the template writes for it.
    fn open(descriptor: &FormatDescriptorV1) -> Result<Self, String> {
        if descriptor.parser != parser_declaration() {
            return Err(
                "MiMo's chat format reads its completions through its own format.parser section"
                    .into(),
            );
        }
        // Line 92: the template thinks unless `enable_thinking` is false,
        // and reads no other keyword and no effort. Another policy would
        // serve a generation prompt the template does not write for the
        // same request, or an effort preamble it never writes.
        let reasoning = &descriptor.reasoning;
        if reasoning.thinking_keywords != ["enable_thinking"]
            || !reasoning.default_enabled
            || !reasoning.allow_disabled
            || reasoning.effort_use != FrontendEffortUseV1::Ignored
            || reasoning
                .efforts
                .iter()
                .any(|effort| !effort.preamble.is_empty())
        {
            return Err(
                "MiMo's template thinks unless enable_thinking is false, and reads no \
                 other keyword and no effort"
                    .into(),
            );
        }
        let mut labels = Vec::new();
        for modality in descriptor
            .multimodal
            .iter()
            .flat_map(|multimodal| &multimodal.modalities)
        {
            let kind = serde_json::to_value(modality.kind).map_err(|error| error.to_string())?;
            let Some((part, _, prefix, pad, suffix)) = MEDIA_LABELS
                .iter()
                .find(|(_, name, ..)| kind.as_str() == Some(*name))
            else {
                return Err(format!("MiMo's template labels no {kind} media"));
            };
            if modality.render.pad != *pad {
                return Err(format!(
                    "MiMo's template writes the {kind} pad {pad:?}, not {:?}",
                    modality.render.pad
                ));
            }
            labels.push((*part, *prefix, *suffix));
        }
        Ok(Self {
            descriptor: Arc::new(descriptor.clone()),
            labels,
        })
    }

    /// The template reads no history reasoning but `reasoning_content`
    /// (line 62), no client tool objects and no role beyond the five.
    fn opened(&self) -> OpenedV1 {
        OpenedV1::default()
    }

    fn render(
        &self,
        request: FormatRenderRequestV1<'_>,
    ) -> Result<FormatRenderedV1, FrontendErrorV1> {
        // The template reads only `enable_thinking` (line 92), as the
        // declaration names it; a top-level effort sets it as vLLM does.
        let (mode, effort) = select_mode_v1(&self.descriptor.reasoning, &request)?;
        let messages = request
            .messages
            .into_iter()
            .map(read_message)
            .collect::<Result<Vec<_>, _>>()?;
        let (prompt, segments, media) = self
            .render_prompt(
                &messages,
                request.tools,
                request.response_format.as_ref(),
                mode,
                request.continue_final_message,
                request.add_generation_prompt,
            )
            .map_err(|message| FrontendErrorV1::new(message, Some("messages")))?;
        Ok(FormatRenderedV1 {
            prompt,
            segments,
            media,
            mode,
            effort: effort.name.clone(),
            // The export decides it from the prompt.
            reasoning_open: false,
            parser_state: Vec::new(),
        })
    }
}

/// One request message as the template reads it.
#[derive(Debug)]
enum Message {
    /// A system, developer, user or tool message: its role, its content as
    /// text, its parts when it sent a list, and the tools it carries
    /// (lines 78-86).
    Plain {
        role: FrontendRoleV1,
        content: String,
        parts: Option<Vec<FrontendContentPartV1>>,
        tools: Vec<FrontendJsonV1>,
        tool_call_id: Option<String>,
    },
    /// An assistant turn (lines 60-68).
    Assistant(AssistantTurnV1),
}

/// Reads one message. A message of any role but the assistant's needs
/// string content, which the transport writes for a parts list too, and a
/// tool result needs the call ID it answers.
fn read_message(message: FrontendMessageV1) -> Result<Message, FrontendErrorV1> {
    let FrontendMessageV1 {
        role,
        content,
        parts,
        reasoning_content,
        tool_calls,
        tool_call_id,
        tools,
    } = message;
    // The template has no tool_reference form, so the open answer admits
    // none; refuse any that reaches the render anyway.
    if parts
        .iter()
        .flatten()
        .any(|part| matches!(part, FrontendContentPartV1::ToolReference { .. }))
    {
        return Err(FrontendErrorV1::new(
            TOOL_REFERENCES_UNSUPPORTED_V1,
            Some("messages"),
        ));
    }
    if role == FrontendRoleV1::Assistant {
        if !tools.is_empty() {
            return Err(FrontendErrorV1::new(
                "assistant message tools are invalid",
                Some("messages"),
            ));
        }
        return Ok(Message::Assistant(AssistantTurnV1 {
            content,
            reasoning_content,
            tool_calls,
        }));
    }
    // Only a user's parts render as parts (lines 9-27), with or without the
    // tools it carries (line 78); every other role's content is its text.
    let parts = parts.filter(|_| role == FrontendRoleV1::User);
    if role == FrontendRoleV1::Tool && tools.is_empty() && tool_call_id.is_none() {
        return Err(FrontendErrorV1::new(
            "tool message requires tool_call_id",
            Some("messages"),
        ));
    }
    let content = match (content, &parts) {
        (Some(content), _) => content,
        (None, Some(_)) => String::new(),
        (None, None) => {
            return Err(FrontendErrorV1::new(
                "message content must be a string",
                Some("messages"),
            ));
        }
    };
    Ok(Message::Plain {
        role,
        content,
        parts,
        tools,
        tool_call_id,
    })
}

impl Mimo26Format {
    /// The whole template (lines 70-95), with the media the prompt places.
    fn render_prompt(
        &self,
        messages: &[Message],
        tools: &[FrontendToolV1],
        response_format: Option<&FrontendJsonV1>,
        mode: PromptModeV1,
        continue_final_message: bool,
        add_generation_prompt: bool,
    ) -> Result<
        (
            String,
            Vec<frontend_format::PromptSegmentV1>,
            Vec<FrontendMediaPartV1>,
        ),
        String,
    > {
        let descriptor = &*self.descriptor;
        self.screen(messages, tools, response_format)?;
        // Each user message's parts, rendered before anything is written.
        let mut media = Vec::new();
        let mut bodies = Vec::with_capacity(messages.len());
        for message in messages {
            bodies.push(match message {
                Message::Plain {
                    parts: Some(parts), ..
                } => Some(self.render_content(parts, &mut media)?),
                _ => None,
            });
        }
        if continue_final_message
            && !matches!(messages.last(), Some(Message::Assistant(turn)) if turn.tool_calls.is_empty())
        {
            return Err(field_v1(
                "continue_final_message requires a final assistant message without tool calls",
            ));
        }
        // HF's continue_final_message continues the final message's content,
        // and refuses a message that has none.
        if continue_final_message
            && matches!(messages.last(), Some(Message::Assistant(turn)) if turn.content.is_none())
        {
            return Err(field_v1(
                "continue_final_message requires a final assistant message with content",
            ));
        }
        if messages.len() > descriptor.limits.maximum_messages
            || tools.len() > descriptor.limits.maximum_tools
        {
            return Err(resource_v1("too many messages or tools"));
        }
        let maximum = descriptor.limits.maximum_prompt_bytes;
        let mut output = String::new();
        // Lines 70-72.
        if !tools.is_empty() {
            append_v1(&mut output, IM_START, maximum)?;
            append_v1(&mut output, "system\n", maximum)?;
            let tools = tools
                .iter()
                .map(|tool| (Some(tool.name.as_str()), tool_object(tool)));
            self.render_tools(&mut output, tools, "tool")?;
            append_v1(&mut output, IM_END, maximum)?;
        }
        // Lines 74-88.
        for (index, (message, body)) in messages.iter().zip(&bodies).enumerate() {
            match message {
                Message::Assistant(turn) => {
                    let close = !(continue_final_message && index + 1 == messages.len());
                    self.render_assistant_message(&mut output, turn, close)?;
                }
                Message::Plain {
                    role,
                    content,
                    tools,
                    ..
                } => {
                    let body = body.as_deref().unwrap_or(content);
                    self.render_message(&mut output, *role, body, tools)?;
                }
            }
        }
        // Lines 90-95. A continued final message ends the prompt.
        if !continue_final_message && add_generation_prompt {
            append_v1(&mut output, IM_START, maximum)?;
            append_v1(&mut output, "assistant\n", maximum)?;
            if mode == PromptModeV1::Chat {
                append_v1(&mut output, THINK_OPEN, maximum)?;
                append_v1(&mut output, THINK_CLOSE, maximum)?;
            }
        }
        split_media_segments_v1(descriptor, output, media)
    }

    /// Refuses client text anywhere that holds a media boundary or pad.
    fn screen(
        &self,
        messages: &[Message],
        tools: &[FrontendToolV1],
        response_format: Option<&FrontendJsonV1>,
    ) -> Result<(), String> {
        let messages: Vec<ChatMessageV1> = messages
            .iter()
            .map(|message| match message {
                Message::Assistant(turn) => ChatMessageV1::Assistant(turn.clone()),
                Message::Plain {
                    role,
                    content,
                    parts,
                    tools,
                    tool_call_id,
                } => match (role, parts) {
                    // A message's text parts join into its content.
                    _ if !tools.is_empty() => ChatMessageV1::WithTools {
                        role: *role,
                        content: content.clone(),
                        tools: tools.clone(),
                    },
                    (_, Some(parts)) => ChatMessageV1::UserParts(parts.clone()),
                    (FrontendRoleV1::Tool, None) => ChatMessageV1::Tool(ToolResultV1 {
                        tool_call_id: tool_call_id.clone().unwrap_or_default(),
                        content: content.clone(),
                    }),
                    _ => ChatMessageV1::User(content.clone()),
                },
            })
            .collect();
        screen_media_text_v1(
            &self.descriptor,
            &ChatRenderV1 {
                mode: PromptModeV1::Thinking,
                preamble: "",
                messages: &messages,
                continue_final_message: false,
                add_generation_prompt: false,
                tools,
                response_format,
                preserve_thinking: true,
            },
            &[],
        )
    }

    /// `render_content` of a parts list (lines 9-27): text parts as
    /// written, and each media part's label around a placeholder for the
    /// media the prompt places. A run of text between media may not spell
    /// a media boundary or pad.
    fn render_content(
        &self,
        parts: &[FrontendContentPartV1],
        media: &mut Vec<FrontendMediaPartV1>,
    ) -> Result<String, String> {
        let descriptor = &*self.descriptor;
        let reserved = |text: &str| reserved_media_text_v1(descriptor, text);
        let mut content = String::new();
        let mut text_start = 0;
        for part in parts {
            match part {
                // Line 23.
                FrontendContentPartV1::Text(text) => content.push_str(text),
                FrontendContentPartV1::ToolReference { .. } => {
                    return Err(TOOL_REFERENCES_UNSUPPORTED_V1.to_owned());
                }
                // Lines 16-21.
                FrontendContentPartV1::Media(item) => {
                    if reserved(&content[text_start..]) {
                        return Err(field_v1("reserved media string in joined client text"));
                    }
                    media_kind_v1(descriptor, item)?;
                    let (_, prefix, suffix) = self
                        .labels
                        .iter()
                        .find(|(kind, ..)| *kind == item.kind)
                        .expect("a served media kind is labelled");
                    content.push_str(prefix);
                    content.push_str(&media_placeholder_v1(media.len()));
                    media.push(item.clone());
                    content.push_str(suffix);
                    text_start = content.len();
                }
            }
        }
        if reserved(&content[text_start..]) {
            return Err(field_v1("reserved media string in joined client text"));
        }
        Ok(content)
    }

    /// `render_tools` (lines 29-35): each tool as `tojson` writes it, with
    /// the name of an offered tool checked before it is written.
    fn render_tools<'a>(
        &self,
        output: &mut String,
        tools: impl Iterator<Item = (Option<&'a str>, FrontendJsonV1)>,
        context: &str,
    ) -> Result<(), String> {
        let maximum = self.descriptor.limits.maximum_prompt_bytes;
        append_v1(
            output,
            "You are provided with the following tools:\n\n<tools>",
            maximum,
        )?;
        for (name, tool) in tools {
            if let Some(name) = name {
                validate_name_v1(&self.descriptor, name, context)?;
            }
            append_v1(output, "\n", maximum)?;
            append_v1(output, &tojson_v1(&tool, &Default::default()), maximum)?;
        }
        append_v1(output, "\n</tools>", maximum)
    }

    /// A system, developer, user or tool message (lines 77-87).
    fn render_message(
        &self,
        output: &mut String,
        role: FrontendRoleV1,
        body: &str,
        tools: &[FrontendJsonV1],
    ) -> Result<(), String> {
        let maximum = self.descriptor.limits.maximum_prompt_bytes;
        append_v1(output, IM_START, maximum)?;
        append_v1(output, role_name(role), maximum)?;
        append_v1(output, "\n", maximum)?;
        append_v1(output, body, maximum)?;
        if !tools.is_empty() {
            if !body.is_empty() {
                append_v1(output, "\n\n", maximum)?;
            }
            if tools.len() > self.descriptor.limits.maximum_tools {
                return Err(resource_v1("too many message tools"));
            }
            self.render_tools(
                output,
                tools.iter().map(|tool| (None, tool.clone())),
                "message tool",
            )?;
        }
        append_v1(output, IM_END, maximum)
    }

    /// `render_assistant_message` (lines 60-68). A continued final message
    /// ends where its content ends, as HF's `continue_final_message` cuts
    /// the render, so `close` is false for it.
    fn render_assistant_message(
        &self,
        output: &mut String,
        turn: &AssistantTurnV1,
        close: bool,
    ) -> Result<(), String> {
        let maximum = self.descriptor.limits.maximum_prompt_bytes;
        // Lines 61-63.
        append_v1(output, IM_START, maximum)?;
        append_v1(output, "assistant\n", maximum)?;
        append_v1(output, THINK_OPEN, maximum)?;
        append_v1(
            output,
            turn.reasoning_content.as_deref().unwrap_or_default(),
            maximum,
        )?;
        append_v1(output, THINK_CLOSE, maximum)?;
        append_v1(output, turn.content.as_deref().unwrap_or_default(), maximum)?;
        // Lines 64-66.
        if turn.tool_calls.len() > self.descriptor.limits.maximum_tool_calls {
            return Err(resource_v1("too many tool calls"));
        }
        self.render_tool_calls(output, &turn.tool_calls)?;
        // Line 67.
        if close {
            append_v1(output, IM_END, maximum)?;
        }
        Ok(())
    }

    /// `render_tool_calls` (lines 37-58). The transport carries only
    /// function calls (lines 39-40), whose arguments are a JSON object
    /// (lines 47-54); a custom call's raw input (lines 41-46) never reaches
    /// here.
    fn render_tool_calls(
        &self,
        output: &mut String,
        calls: &[FrontendToolCallV1],
    ) -> Result<(), String> {
        let descriptor = &*self.descriptor;
        let maximum = descriptor.limits.maximum_prompt_bytes;
        for call in calls {
            validate_name_v1(descriptor, &call.name, "tool call")?;
            let arguments =
                parse_json_object_v1(&call.arguments, descriptor.limits, "tool arguments")?;
            if arguments.len() > descriptor.limits.maximum_tool_parameters {
                return Err(resource_v1("too many tool parameters"));
            }
            // Line 44.
            append_v1(output, "<tool_call><function=", maximum)?;
            append_v1(output, &call.name, maximum)?;
            append_v1(output, ">", maximum)?;
            // Lines 51-53.
            for (name, value) in &arguments {
                validate_name_v1(descriptor, name, "tool parameter")?;
                append_v1(output, "<parameter=", maximum)?;
                append_v1(output, name, maximum)?;
                append_v1(output, ">", maximum)?;
                append_v1(output, &render_value(value), maximum)?;
                append_v1(output, "</parameter>", maximum)?;
            }
            // Line 56.
            append_v1(output, "</function></tool_call>", maximum)?;
        }
        Ok(())
    }
}

/// `render_value` (lines 1-7): a string as written, anything else as
/// `tojson(ensure_ascii=False)` writes it.
fn render_value(value: &FrontendJsonV1) -> String {
    match value {
        FrontendJsonV1::String(text) => text.clone(),
        value => tojson_v1(value, &Default::default()),
    }
}

/// One offered tool as the transport hands it to the template: a function
/// tool object with the function fields the request sent. The transport
/// reads a null field as an absent one, which `tojson` (line 32) then
/// leaves out.
fn tool_object(tool: &FrontendToolV1) -> FrontendJsonV1 {
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
    if let Some(strict) = tool.strict {
        function.push(("strict".to_owned(), FrontendJsonV1::Bool(strict)));
    }
    FrontendJsonV1::Object(vec![
        (
            "type".to_owned(),
            FrontendJsonV1::String("function".to_owned()),
        ),
        ("function".to_owned(), FrontendJsonV1::Object(function)),
    ])
}

/// `message.role` (line 79).
const fn role_name(role: FrontendRoleV1) -> &'static str {
    match role {
        FrontendRoleV1::System => "system",
        FrontendRoleV1::Developer => "developer",
        FrontendRoleV1::User => "user",
        FrontendRoleV1::Assistant => "assistant",
        FrontendRoleV1::Tool => "tool",
    }
}
