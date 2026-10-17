//! Qwen3.8's chat format, ported from its chat template.
//!
//! The template is the checkpoint's `chat_template.jinja` at the revision
//! and digest `fixtures/qwen38/frontend/provenance.json` pins. Each function
//! below ports the template lines it cites; line numbers are the template's.
//! The template's `| trim` is Python's `str.strip` ([`python_strip_v1`]).
//! The bundle exports [`Qwen38Format`] with `export_chat_format_v1!`, and the
//! shared parsers read its completions through the declaration's
//! `format.parser` ([`parser_declaration`]).
//!
//! What the template leaves to the caller follows HF's `apply_chat_template`:
//! `continue_final_message` cuts the prompt after the final message's
//! content and strips the whitespace before the cut, and refuses a final
//! message without content. The transport refuses before the render what
//! the template raises on for shapes it never sends (media outside user
//! messages, unknown content parts and roles, an empty conversation); the
//! render refuses the rest.
//! `models/qwen38/frontend-divergences.json` lists where a request's outcome
//! differs from the template's on purpose.

use frontend_format::toolkit::{
    TojsonOptionsV1, append_v1, chat_message_v1, field_v1, json_truthy, media_kind_v1,
    media_placeholder_v1, parse_json_object_v1, preserve_thinking_v1, python_rstrip_v1,
    python_strip_v1, reserved_media_text_v1, resource_v1, screen_media_text_v1, select_mode_v1,
    split_media_segments_v1, tojson_v1, validate_name_v1,
};
use frontend_format::{
    AssistantTurnV1, ChatFormatV1, ChatMessageV1, ChatRenderV1, FormatDescriptorV1,
    FormatRenderRequestV1, FormatRenderedV1, FrontendContentPartV1, FrontendEffortUseV1,
    FrontendErrorV1, FrontendJsonEncodingV1, FrontendJsonV1, FrontendMediaPartKindV1,
    FrontendMediaPartV1, FrontendParserDeclarationV1, FrontendToolCallV1, FrontendToolGrammarV1,
    FrontendToolV1, OpenedV1, PromptModeV1, TOOL_REFERENCES_UNSUPPORTED_V1,
};

/// The reasoning instructions the template writes for each effort (lines
/// 51-55); `medium` writes none.
const EFFORTS: [(&str, &str); 3] = [
    (
        "xhigh",
        "Reasoning effort is set to xhigh. Please think carefully through the task, validate key \
         assumptions, consider plausible alternatives, and prioritize correctness, consistency, \
         and clarity in the final answer.",
    ),
    ("medium", ""),
    (
        "low",
        "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly \
         to the conclusion without unnecessary elaboration.",
    ),
];

/// The effort an absent `reasoning_effort` selects (line 47).
const DEFAULT_EFFORT: &str = "xhigh";

/// The tool list's header (line 62) and the instructions after it (lines
/// 67-68).
const TOOLS_HEADER: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const TOOLS_FOOTER: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the \
following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n\
<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\n\
This is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n\
</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified \
format: an inner <function=...></function> block must be nested within <tool_call></tool_call> \
XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your \
function call in natural language BEFORE the function call, but NOT after\n- If there is no \
function call available, answer the question like normal with your current knowledge and do not \
tell the user about function calls\n</IMPORTANT>";

/// What a tool result's content must be wrapped in to read as one (line 93).
const TOOL_RESPONSE: (&str, &str) = ("<tool_response>", "</tool_response>");

/// The pad the template writes for each media kind (lines 18 and 29), which
/// the served declaration must name for its modality.
const IMAGE_PAD: &str = "<|image_pad|>";
const VIDEO_PAD: &str = "<|video_pad|>";

/// The `format.parser` section Qwen3.8's completions read through: the
/// markers the template writes around reasoning (line 117) and tool calls
/// (lines 126-143), and the boundary token that opens every turn.
#[must_use]
pub fn parser_declaration() -> FrontendParserDeclarationV1 {
    FrontendParserDeclarationV1 {
        tool_grammar: FrontendToolGrammarV1::XmlFunctionParametersV1,
        json_encoding: FrontendJsonEncodingV1::PythonJsonV1,
        reasoning_open: "<think>".into(),
        reasoning_close: "</think>".into(),
        answer_prefix: "\n\n".into(),
        tool_call_open: "<tool_call>\n".into(),
        tool_call_close: "</tool_call>".into(),
        tool_call_separator: "\n".into(),
        function_open: "<function=".into(),
        function_name_close: ">\n".into(),
        function_close: "</function>\n".into(),
        parameter_open: "<parameter=".into(),
        parameter_name_close: ">\n".into(),
        parameter_json_name_close: String::new(),
        parameter_value_open: String::new(),
        parameter_close: "\n</parameter>\n".into(),
        drop_blank_tool_call_content: true,
        reserved_markers: Vec::new(),
        turn_open: vec!["<|im_start|>".into()],
        assistant_header: "<|im_start|>assistant".into(),
    }
}

/// The bundle's frontend export, linked into this program: what the
/// compiler's tests serve a declaration through.
#[must_use]
pub fn export() -> runtime_bundle_abi::frontend::FrontendExportV1 {
    frontend_format::chat_format_export_v1::<Qwen38Format>()
}

/// Qwen3.8's chat template, over the declaration a package serves.
pub struct Qwen38Format {
    descriptor: FormatDescriptorV1,
}

impl ChatFormatV1 for Qwen38Format {
    /// Opens over a declaration whose parser section, reasoning controls
    /// and media pads are the ones this template writes, and which serves no
    /// media kind but images and video.
    fn open(descriptor: &FormatDescriptorV1) -> Result<Self, String> {
        if descriptor.parser != parser_declaration() {
            return Err(
                "the Qwen3.8 chat format reads its completions through its own \
                        format.parser section"
                    .into(),
            );
        }
        let reasoning = &descriptor.reasoning;
        let efforts: Vec<(&str, &str)> = reasoning
            .efforts
            .iter()
            .map(|effort| (effort.name.as_str(), effort.preamble.as_str()))
            .collect();
        if efforts != EFFORTS
            || reasoning.default_effort != DEFAULT_EFFORT
            || reasoning.thinking_keywords != ["enable_thinking"]
        {
            return Err(
                "the Qwen3.8 chat format declares the template's efforts, default \
                        effort and enable_thinking keyword"
                    .into(),
            );
        }
        // Lines 46 and 165: the template thinks unless `enable_thinking` is
        // false, and reads the effort only while thinking. Another policy
        // would serve a generation prompt or effort the template does not
        // write for the same request.
        if !reasoning.default_enabled
            || !reasoning.allow_disabled
            || reasoning.effort_use != FrontendEffortUseV1::WhenThinking
        {
            return Err(
                "the Qwen3.8 chat format thinks by default, lets enable_thinking \
                        disable thinking, and reads the effort only while thinking"
                    .into(),
            );
        }
        for modality in descriptor
            .multimodal
            .iter()
            .flat_map(|multimodal| &multimodal.modalities)
        {
            let kind = serde_json::to_value(modality.kind).map_err(|error| error.to_string())?;
            let pad = match kind.as_str() {
                Some("image") => IMAGE_PAD,
                Some("video") => VIDEO_PAD,
                _ => return Err(format!("the Qwen3.8 chat format writes no {kind} input")),
            };
            if modality.render.pad != pad {
                return Err(format!(
                    "the Qwen3.8 chat format pads {kind} input with {pad}"
                ));
            }
        }
        Ok(Self {
            descriptor: descriptor.clone(),
        })
    }

    /// The template reads `reasoning_content` from history (line 112),
    /// which the transport forwards on turns with tool calls only.
    fn opened(&self) -> OpenedV1 {
        OpenedV1 {
            tool_call_reasoning_fallback: true,
            ..OpenedV1::default()
        }
    }

    fn render(
        &self,
        request: FormatRenderRequestV1<'_>,
    ) -> Result<FormatRenderedV1, FrontendErrorV1> {
        let (mode, effort) = select_mode_v1(&self.descriptor.reasoning, &request)?;
        let preserve_thinking = preserve_thinking_v1(false, &request)?;
        let add_vision_id = request.add_vision_id.as_ref().is_some_and(json_truthy);
        let messages = request
            .messages
            .into_iter()
            .map(chat_message_v1)
            .collect::<Result<Vec<_>, _>>()?;
        let refused = |message: String| FrontendErrorV1::new(message, Some("messages"));
        screen_media_text_v1(
            &self.descriptor,
            &ChatRenderV1 {
                mode,
                preamble: "",
                messages: &messages,
                continue_final_message: request.continue_final_message,
                add_generation_prompt: request.add_generation_prompt,
                tools: request.tools,
                response_format: request.response_format.as_ref(),
                preserve_thinking,
            },
            &request.tool_objects,
        )
        .map_err(refused)?;
        let mut media = Vec::new();
        let conversation = Conversation::read(
            &self.descriptor,
            messages,
            add_vision_id,
            &mut media,
            request.continue_final_message,
            request.tools.len(),
        )
        .map_err(refused)?;
        let instructions = match mode {
            PromptModeV1::Thinking => instructions(&effort.name),
            PromptModeV1::Chat => "",
        };
        let mut prompt = Prompt {
            text: String::new(),
            maximum: self.descriptor.limits.maximum_prompt_bytes,
        };
        self.system_block(&mut prompt, &conversation, instructions, request.tools)
            .map_err(refused)?;
        self.messages(&mut prompt, &conversation, preserve_thinking)
            .map_err(refused)?;
        if request.continue_final_message {
            // HF's apply_chat_template cuts the prompt after the final
            // message's content; this template trims it, so the cut strips
            // the whitespace before it.
            let length = python_rstrip_v1(&prompt.text).len();
            prompt.text.truncate(length);
        } else if request.add_generation_prompt {
            generation_prompt(&mut prompt, mode).map_err(refused)?;
        }
        let (prompt, segments, media) =
            split_media_segments_v1(&self.descriptor, prompt.text, media).map_err(refused)?;
        Ok(FormatRenderedV1 {
            prompt,
            segments,
            media,
            mode,
            effort: effort.name.clone(),
            // The export decides this from the prompt.
            reasoning_open: false,
            parser_state: Vec::new(),
        })
    }
}

impl Qwen38Format {
    /// The system turn (lines 57-87): the reasoning instructions, the tool
    /// list and the first message's system content, each when present.
    fn system_block(
        &self,
        prompt: &mut Prompt,
        conversation: &Conversation,
        instructions: &str,
        tools: &[FrontendToolV1],
    ) -> Result<(), String> {
        let system = match conversation.messages.first() {
            Some(Message::System(content)) => python_strip_v1(content),
            _ => "",
        };
        if !tools.is_empty() {
            // Lines 58-75.
            prompt.push("<|im_start|>system\n")?;
            if !instructions.is_empty() {
                prompt.push(instructions)?;
                prompt.push("\n\n")?;
            }
            prompt.push(TOOLS_HEADER)?;
            for tool in tools {
                prompt.push("\n")?;
                prompt.push(&self.tool_json(tool)?)?;
            }
            prompt.push(TOOLS_FOOTER)?;
            if !system.is_empty() {
                prompt.push("\n\n")?;
                prompt.push(system)?;
            }
            return prompt.push("<|im_end|>\n");
        }
        // Lines 77-86.
        if system.is_empty() && instructions.is_empty() {
            return Ok(());
        }
        prompt.push("<|im_start|>system\n")?;
        prompt.push(instructions)?;
        if !instructions.is_empty() && !system.is_empty() {
            prompt.push("\n\n")?;
        }
        prompt.push(system)?;
        prompt.push("<|im_end|>\n")
    }

    /// `tool | tojson` (line 65) of an offered tool as the transport carries
    /// it: its function wrapped in its type, with only the fields the client
    /// sent.
    fn tool_json(&self, tool: &FrontendToolV1) -> Result<String, String> {
        validate_name_v1(&self.descriptor, &tool.name, "tool")?;
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
        let tool = FrontendJsonV1::Object(vec![
            (
                "type".to_owned(),
                FrontendJsonV1::String("function".to_owned()),
            ),
            ("function".to_owned(), FrontendJsonV1::Object(function)),
        ]);
        Ok(tojson_v1(&tool, &TojsonOptionsV1::default()))
    }

    /// The message loop (lines 102-162).
    fn messages(
        &self,
        prompt: &mut Prompt,
        conversation: &Conversation,
        preserve_thinking: bool,
    ) -> Result<(), String> {
        let messages = &conversation.messages;
        for (index, message) in messages.iter().enumerate() {
            let last = index + 1 == messages.len();
            match message {
                // Lines 104-107: the system block wrote it. The others were
                // refused before anything was written.
                Message::System(_) | Message::Developer | Message::WithTools => {}
                // Line 109.
                Message::User(content) => {
                    prompt.push("<|im_start|>user\n")?;
                    prompt.push(python_strip_v1(content))?;
                    prompt.push("<|im_end|>\n")?;
                }
                Message::Assistant(turn) => {
                    // Lines 116-120: reasoning renders after the last user
                    // query, and before it unless preserve_thinking is false.
                    let reasoning = preserve_thinking || index > conversation.last_query;
                    self.assistant(
                        prompt,
                        turn,
                        reasoning,
                        !(conversation.continue_final_message && last),
                    )?;
                }
                // Lines 147-158: a run of results is one user turn, which
                // a result with no message before it does not open.
                Message::Tool(content) => {
                    if index > 0 && !matches!(messages[index - 1], Message::Tool(_)) {
                        prompt.push("<|im_start|>user")?;
                    }
                    prompt.push("\n<tool_response>\n")?;
                    prompt.push(python_strip_v1(content))?;
                    prompt.push("\n</tool_response>")?;
                    if last || !matches!(messages[index + 1], Message::Tool(_)) {
                        prompt.push("<|im_end|>\n")?;
                    }
                }
            }
        }
        Ok(())
    }

    /// One assistant turn (lines 110-146). A continued final turn, which
    /// has no calls, ends at its content without its turn end.
    fn assistant(
        &self,
        prompt: &mut Prompt,
        turn: &AssistantTurnV1,
        reasoning: bool,
        close: bool,
    ) -> Result<(), String> {
        let content = python_strip_v1(turn.content.as_deref().unwrap_or_default());
        prompt.push("<|im_start|>assistant\n")?;
        if reasoning {
            // Lines 111-115 and 117.
            prompt.push("<think>\n")?;
            prompt.push(python_strip_v1(
                turn.reasoning_content.as_deref().unwrap_or_default(),
            ))?;
            prompt.push("\n</think>\n\n")?;
        }
        prompt.push(content)?;
        if turn.tool_calls.len() > self.descriptor.limits.maximum_tool_calls {
            return Err(resource_v1("too many tool calls"));
        }
        for (position, call) in turn.tool_calls.iter().enumerate() {
            // Lines 126-134.
            if position != 0 {
                prompt.push("\n")?;
            } else if !content.is_empty() {
                prompt.push("\n\n")?;
            }
            self.tool_call(prompt, call)?;
        }
        if close {
            prompt.push("<|im_end|>\n")?;
        }
        Ok(())
    }

    /// One tool call (lines 122-143). The transport carries its arguments
    /// as the client's JSON text; each parameter's value is written as is
    /// when it is a string, and as `tojson` writes it otherwise (line 138).
    fn tool_call(&self, prompt: &mut Prompt, call: &FrontendToolCallV1) -> Result<(), String> {
        let limits = self.descriptor.limits;
        validate_name_v1(&self.descriptor, &call.name, "tool call")?;
        let arguments = parse_json_object_v1(&call.arguments, limits, "tool arguments")?;
        if arguments.len() > limits.maximum_tool_parameters {
            return Err(resource_v1("too many tool parameters"));
        }
        prompt.push("<tool_call>\n<function=")?;
        prompt.push(&call.name)?;
        prompt.push(">\n")?;
        for (name, value) in arguments {
            validate_name_v1(&self.descriptor, &name, "tool parameter")?;
            prompt.push("<parameter=")?;
            prompt.push(&name)?;
            prompt.push(">\n")?;
            match value {
                FrontendJsonV1::String(text) => prompt.push(&text)?,
                value => prompt.push(&tojson_v1(&value, &TojsonOptionsV1::default()))?,
            }
            prompt.push("\n</parameter>\n")?;
        }
        prompt.push("</function>\n</tool_call>")
    }
}

/// The generation prompt (lines 163-170): an open think block, or an empty
/// closed one when thinking is disabled.
fn generation_prompt(prompt: &mut Prompt, mode: PromptModeV1) -> Result<(), String> {
    prompt.push("<|im_start|>assistant\n")?;
    prompt.push(match mode {
        PromptModeV1::Chat => "<think>\n\n</think>\n\n",
        PromptModeV1::Thinking => "<think>\n",
    })
}

/// The reasoning instructions an effort writes (lines 51-55).
fn instructions(effort: &str) -> &'static str {
    EFFORTS
        .iter()
        .find(|(name, _)| *name == effort)
        .map_or("", |(_, text)| text)
}

/// The prompt being written, under the declared byte bound.
struct Prompt {
    text: String,
    maximum: usize,
}

impl Prompt {
    fn push(&mut self, text: &str) -> Result<(), String> {
        append_v1(&mut self.text, text, self.maximum)
    }
}

/// One message as the template renders it, with a user's content parts
/// already written by `render_content`.
enum Message {
    System(String),
    User(String),
    Assistant(AssistantTurnV1),
    Tool(String),
    /// A developer message, which the template has no role for (line 160).
    Developer,
    /// A message that carries tools, which the template ignores.
    WithTools,
}

/// The conversation after the checks the template makes before it writes
/// anything.
struct Conversation {
    messages: Vec<Message>,
    /// The last user message that is not a wrapped tool result (lines
    /// 88-98).
    last_query: usize,
    continue_final_message: bool,
}

impl Conversation {
    /// Reads the request's messages, writing each user message's content
    /// parts and collecting their media in order, and refuses a
    /// conversation the template raises on or this format does not render.
    fn read(
        descriptor: &FormatDescriptorV1,
        messages: Vec<ChatMessageV1>,
        add_vision_id: bool,
        media: &mut Vec<FrontendMediaPartV1>,
        continue_final_message: bool,
        tools: usize,
    ) -> Result<Self, String> {
        let mut counts = [0_usize; 2];
        let mut read = Vec::with_capacity(messages.len());
        for message in messages {
            read.push(match message {
                ChatMessageV1::System(content) => Message::System(content),
                ChatMessageV1::User(content) => Message::User(content),
                ChatMessageV1::UserParts(parts) => Message::User(render_content(
                    descriptor,
                    &parts,
                    add_vision_id,
                    &mut counts,
                    media,
                )?),
                ChatMessageV1::Assistant(turn) => Message::Assistant(turn),
                ChatMessageV1::Tool(result) => Message::Tool(result.content),
                ChatMessageV1::Developer(_) => Message::Developer,
                ChatMessageV1::WithTools { .. } => Message::WithTools,
            });
        }
        // HF's apply_chat_template continues a final message's content.
        if continue_final_message {
            let Some(Message::Assistant(turn)) = read.last() else {
                return Err(continue_refusal());
            };
            if !turn.tool_calls.is_empty() {
                return Err(continue_refusal());
            }
            if turn.content.is_none() {
                return Err(field_v1(
                    "continue_final_message requires a final assistant message with content",
                ));
            }
        }
        if read.len() > descriptor.limits.maximum_messages
            || tools > descriptor.limits.maximum_tools
        {
            return Err(resource_v1("too many messages or tools"));
        }
        // Lines 88-101: the last user message that is not a tool result
        // wrapped in its tags.
        let last_query = read
            .iter()
            .rposition(|message| {
                let Message::User(content) = message else {
                    return false;
                };
                let content = python_strip_v1(content);
                !(content.starts_with(TOOL_RESPONSE.0) && content.ends_with(TOOL_RESPONSE.1))
            })
            .ok_or_else(|| field_v1("a user query is required"))?;
        // Lines 104-107 and 159-161, checked for every message before
        // anything is written.
        for (index, message) in read.iter().enumerate() {
            match message {
                Message::System(_) if index != 0 => {
                    return Err(field_v1("system message must be first"));
                }
                Message::Developer => return Err(field_v1("developer role is not declared")),
                Message::WithTools => return Err(field_v1("message tools are not declared")),
                _ => {}
            }
        }
        Ok(Self {
            messages: read,
            last_query,
            continue_final_message,
        })
    }
}

fn continue_refusal() -> String {
    field_v1("continue_final_message requires a final assistant message without tool calls")
}

/// `render_content` of a user message's parts (lines 3-41, called at line
/// 103): text as written, and each media item as its vision markers around
/// a placeholder the segments replace with the item's pad, after a
/// numbered label when the request sets `add_vision_id`. Images and videos
/// are numbered separately through the whole conversation (lines 13 and
/// 24); the last-query scan (line 92) reads the same text.
///
/// Client text may not carry a media placeholder or pad, even split across
/// two adjacent text parts, which render as one string.
fn render_content(
    descriptor: &FormatDescriptorV1,
    parts: &[FrontendContentPartV1],
    add_vision_id: bool,
    counts: &mut [usize; 2],
    media: &mut Vec<FrontendMediaPartV1>,
) -> Result<String, String> {
    let reserved = |text: &str| reserved_media_text_v1(descriptor, text);
    let joined = || field_v1("reserved media string in joined client text");
    let mut content = String::new();
    let mut text_start = 0;
    for part in parts {
        match part {
            // Lines 30-31.
            FrontendContentPartV1::Text(text) => content.push_str(text),
            // Line 33: any other item raises. The template reads no
            // tool_reference part, which the conversation's reading refuses
            // first.
            FrontendContentPartV1::ToolReference { .. } => {
                return Err(TOOL_REFERENCES_UNSUPPORTED_V1.into());
            }
            // Lines 8-29.
            FrontendContentPartV1::Media(item) => {
                if reserved(&content[text_start..]) {
                    return Err(joined());
                }
                media_kind_v1(descriptor, item)?;
                let (count, label) = match item.kind {
                    FrontendMediaPartKindV1::Image => (&mut counts[0], "Picture "),
                    FrontendMediaPartKindV1::Video => (&mut counts[1], "Video "),
                    FrontendMediaPartKindV1::Audio => {
                        return Err(field_v1("media part type is not declared by this bundle"));
                    }
                };
                *count += 1;
                if add_vision_id {
                    append_v1(
                        &mut content,
                        &format!("{label}{count}: "),
                        descriptor.limits.maximum_prompt_bytes,
                    )?;
                    if reserved(&content[text_start..]) {
                        return Err(joined());
                    }
                }
                content.push_str("<|vision_start|>");
                content.push_str(&media_placeholder_v1(media.len()));
                content.push_str("<|vision_end|>");
                media.push(item.clone());
                text_start = content.len();
            }
        }
    }
    if reserved(&content[text_start..]) {
        return Err(joined());
    }
    Ok(content)
}
