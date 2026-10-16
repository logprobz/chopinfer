//! GLM 5.3 Flash's chat format, ported from its pinned chat template.
//!
//! The template is the checkpoint's `chat_template.jinja` at revision
//! eb9eb208 (261 lines, sha256 0c4099f3...). Each function below ports one of
//! its macros or blocks and names the template lines it ports, so a template
//! change maps to the code it touches.
//!
//! The transport hands the render what the template reads: string content
//! (a content list's text parts joined), a user's media parts in order, each
//! offered tool's function object as the client sent it,
//! `reasoning_content` on every assistant turn, and a tool result's
//! `tool_reference` parts (the `tool_name` alias read as `name`). The
//! template's branches for
//! shapes the transport refuses (a list of outputs, a bare string in a
//! content list), and the requests this format refuses although the template
//! renders them, are listed in `models/glm5next/frontend-divergences.json`;
//! the HF corpus (`fixtures/glm5next/frontend/corpus.json`) checks both.
//!
//! Beyond the template, the render keeps the bounds every format charges: the
//! prompt's byte limit, the message, tool, call and parameter counts, names
//! in the declared byte set, and no media boundary in client text.

use frontend_format::toolkit::{
    append_v1, chat_message_v1, defers_loading_v1, field_v1, media_kind_v1, media_placeholder_v1,
    parse_json_object_v1, preserve_thinking_v1, python_capitalize_v1, python_split_v1,
    python_strip_v1, reasoning_open_v1, reserved_media_text_v1, resource_v1, screen_media_text_v1,
    select_mode_v1, split_media_segments_v1, tojson_v1, validate_name_v1, write_tool_object_v1,
};
use frontend_format::{
    AssistantTurnV1, ChatFormatV1, ChatMessageV1, ChatRenderV1, FormatDescriptorV1,
    FormatRenderRequestV1, FormatRenderedV1, FrontendContentPartV1, FrontendErrorV1,
    FrontendJsonEncodingV1, FrontendJsonV1, FrontendMediaKindV1, FrontendMediaPartKindV1,
    FrontendMediaPartV1, FrontendMessageV1, FrontendParserDeclarationV1, FrontendRoleV1,
    FrontendToolCallV1, FrontendToolGrammarV1, FrontendToolV1, OpenedV1, PromptSegmentV1,
    TOOL_REFERENCES_UNSUPPORTED_V1, ToolResultV1,
};

/// Line 1.
const BEGIN: &str = "[gMASK]<sop>";
/// Lines 32-38, through the newline the tool loop's first tag keeps.
const TOOLS_HEADER: &str = "<|system|>\n# Tools\n\nYou may call one or more functions to assist \
                            with the user query.\n\nYou are provided with function signatures \
                            within <tools></tools> XML tags:\n<tools>\n";
/// Lines 47-50.
const TOOLS_FOOTER: &str = "</tools>\n\nFor each function call, output the function name and \
                            arguments within the following XML format:\n<tool_call>\
                            {function-name}<arg_key>{arg-key-1}</arg_key><arg_value>\
                            {arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>\
                            {arg-value-2}</arg_value>...</tool_call>";
/// `emit_image` and `emit_video` (lines 51-52): the labels around the pad.
/// `emit_audio` (line 53) writes no pad between its labels, so an audio
/// item has no place in the prompt and audio is not served.
const IMAGE_LABELS: (&str, &str, &str) = ("<|begin_of_image|>", "<|image|>", "<|end_of_image|>");
const VIDEO_LABELS: (&str, &str, &str) = ("<|begin_of_video|>", "<|video|>", "<|end_of_video|>");

/// The template's efforts (line 2): any other renders as `max`, as does a
/// request that names none.
const TEMPLATE_EFFORTS: [&str; 3] = ["low", "high", "max"];

/// The reasoning keywords a declaration reads: the template reads neither,
/// so a request that switches thinking off through one is refused (see
/// `frontend-divergences.json`).
const THINKING_KEYWORDS: [&str; 2] = ["enable_thinking", "thinking"];

/// The parser section of the markers this renderer writes (lines 50, 141,
/// 150-163, 259-260): the parsers read completions, and the export decides
/// whether a prompt leaves reasoning open, with these.
#[must_use]
pub fn parser_declaration() -> FrontendParserDeclarationV1 {
    let text = |value: &str| value.to_owned();
    FrontendParserDeclarationV1 {
        tool_grammar: FrontendToolGrammarV1::XmlKeyValueArgumentsV1,
        json_encoding: FrontendJsonEncodingV1::PythonJsonV1,
        reasoning_open: text("<think>"),
        reasoning_close: text("</think>"),
        answer_prefix: String::new(),
        tool_call_open: text("<tool_call>"),
        tool_call_close: text("</tool_call>"),
        tool_call_separator: String::new(),
        function_open: String::new(),
        function_name_close: String::new(),
        function_close: String::new(),
        parameter_open: text("<arg_key>"),
        parameter_name_close: text("</arg_key>"),
        parameter_value_open: text("<arg_value>"),
        parameter_close: text("</arg_value>"),
        drop_blank_tool_call_content: false,
        reserved_markers: Vec::new(),
        turn_open: ["<|assistant|>", "<|observation|>", "<|system|>", "<|user|>"]
            .map(text)
            .to_vec(),
        assistant_header: text("<|assistant|>"),
    }
}

/// Why `descriptor`'s reasoning controls or parser section are not the ones
/// this renderer writes its prompt for, if they are not. The render always
/// opens `<think>` and writes the effort the template reads; the export
/// reads the parser section to decide whether the prompt leaves reasoning
/// open, so a declaration that differs would parse a completion in another
/// state than the prompt left it. (The library itself holds the reasoning
/// end markers to the parser section's reasoning close and tool-call open.)
fn reasoning_mismatch(descriptor: &FormatDescriptorV1) -> Option<String> {
    let reasoning = &descriptor.reasoning;
    let names = reasoning.effort_names();
    if names != TEMPLATE_EFFORTS {
        return Some(format!(
            "declares efforts {names:?}; the template reads {TEMPLATE_EFFORTS:?}"
        ));
    }
    if let Some(effort) = reasoning.efforts.iter().find(|effort| {
        effort.preamble != format!("Reasoning Effort: {}", python_capitalize_v1(&effort.name))
    }) {
        return Some(format!(
            "declares effort '{}' with preamble {:?}; the template writes \
             \"Reasoning Effort: {}\"",
            effort.name,
            effort.preamble,
            python_capitalize_v1(&effort.name)
        ));
    }
    if reasoning.default_effort != "max" {
        return Some(format!(
            "declares default effort '{}'; the template renders max when none is named",
            reasoning.default_effort
        ));
    }
    if !reasoning.default_enabled || reasoning.allow_disabled {
        return Some(
            "declares reasoning that can be off; the template always opens <think>".into(),
        );
    }
    if reasoning.thinking_keywords != THINKING_KEYWORDS {
        return Some(format!(
            "declares thinking keywords {:?}; this format reads {THINKING_KEYWORDS:?}",
            reasoning.thinking_keywords
        ));
    }
    if reasoning.disable_value != "none" {
        return Some(format!(
            "declares disable value '{}'; this format reads 'none'",
            reasoning.disable_value
        ));
    }
    // The effort use as the declaration spells it.
    let effort_use = serde_json::to_value(reasoning.effort_use).ok()?;
    if effort_use != "when-thinking" {
        return Some(format!(
            "declares effort use {effort_use}; the template reads the effort a request names"
        ));
    }
    if descriptor.parser != parser_declaration() {
        return Some(
            "declares a format.parser section other than the markers this renderer writes".into(),
        );
    }
    None
}

/// GLM 5.3 Flash's chat format over its checked declaration.
#[derive(Debug)]
pub struct Glm53Format {
    descriptor: FormatDescriptorV1,
}

impl ChatFormatV1 for Glm53Format {
    /// Refuses a declaration this template cannot render: reasoning
    /// controls or a parser section other than the ones the renderer writes
    /// its prompt for ([`reasoning_mismatch`]), or a served media kind it
    /// writes no labels for.
    fn open(descriptor: &FormatDescriptorV1) -> Result<Self, String> {
        if let Some(mismatch) = reasoning_mismatch(descriptor) {
            return Err(format!(
                "GLM 5.3 Flash's chat format: the declaration {mismatch}"
            ));
        }
        for modality in descriptor
            .multimodal
            .iter()
            .flat_map(|multimodal| &multimodal.modalities)
        {
            // The kind as the declaration spells it.
            let (kind, labels) = match modality.kind {
                FrontendMediaKindV1::Image => ("image", IMAGE_LABELS),
                FrontendMediaKindV1::Video => ("video", VIDEO_LABELS),
                FrontendMediaKindV1::Audio => {
                    return Err("GLM 5.3 Flash's template writes no audio media".into());
                }
            };
            if modality.render.pad != labels.1 {
                return Err(format!(
                    "GLM 5.3 Flash's template writes {kind} media as {}, not {}",
                    labels.1, modality.render.pad
                ));
            }
        }
        Ok(Self {
            descriptor: descriptor.clone(),
        })
    }

    /// The template reads each tool's function object as sent (lines 6-17,
    /// 39-46), every assistant turn's `reasoning_content` (line 143) and a
    /// tool result's `tool_reference` parts (lines 18-31, 81-82).
    fn opened(&self) -> OpenedV1 {
        OpenedV1 {
            reads_tool_objects: true,
            reasoning_content_fallback: true,
            tool_call_reasoning_fallback: false,
            extra_roles: Vec::new(),
            tool_references: true,
        }
    }

    fn render(
        &self,
        request: FormatRenderRequestV1<'_>,
    ) -> Result<FormatRenderedV1, FrontendErrorV1> {
        let (mode, effort) = select_mode_v1(&self.descriptor.reasoning, &request)?;
        // Line 4: `clear_thinking`, read as a boolean.
        let clear_thinking = !preserve_thinking_v1(true, &request)?;
        let mut messages = request.messages;
        let references = self.take_tool_references(&mut messages, request.tools)?;
        // Read before the parts are joined into the content.
        let continued = request
            .continue_final_message
            .then(|| continued_text(messages.last()))
            .flatten();
        let messages = messages
            .into_iter()
            .map(chat_message_v1)
            .collect::<Result<Vec<_>, _>>()?;
        let chat = ChatRenderV1 {
            mode,
            preamble: "",
            messages: &messages,
            continue_final_message: request.continue_final_message,
            add_generation_prompt: request.add_generation_prompt,
            tools: request.tools,
            response_format: request.response_format.as_ref(),
            preserve_thinking: !clear_thinking,
        };
        let (prompt, segments, media) = self
            .render_chat(
                &chat,
                &effort.name,
                clear_thinking,
                &request.tool_objects,
                &references,
                continued.as_deref(),
            )
            .map_err(|message| FrontendErrorV1::new(message, Some("messages")))?;
        let reasoning_open = reasoning_open_v1(
            mode,
            request.continue_final_message,
            "<think>",
            "</think>",
            &prompt,
        );
        Ok(FormatRenderedV1 {
            prompt,
            segments,
            media,
            mode,
            effort: effort.name.clone(),
            reasoning_open,
            parser_state: Vec::new(),
        })
    }
}

/// One prompt being written under the declared byte bound.
struct Prompt<'a> {
    descriptor: &'a FormatDescriptorV1,
    text: String,
}

impl Prompt<'_> {
    fn write(&mut self, value: &str) -> Result<(), String> {
        append_v1(
            &mut self.text,
            value,
            self.descriptor.limits.maximum_prompt_bytes,
        )
    }
}

impl Glm53Format {
    /// The whole template (lines 1-261) over the request's conversation,
    /// split into text and media segments.
    fn render_chat(
        &self,
        chat: &ChatRenderV1<'_>,
        effort: &str,
        clear_thinking: bool,
        tool_objects: &[FrontendJsonV1],
        references: &[Option<Vec<String>>],
        continued: Option<&str>,
    ) -> Result<(String, Vec<PromptSegmentV1>, Vec<FrontendMediaPartV1>), String> {
        let descriptor = &self.descriptor;
        screen_media_text_v1(descriptor, chat, tool_objects)?;
        // A user's media parts are placed first, in message order, so each
        // item's placeholder indexes the media list.
        let mut media = Vec::new();
        let messages = chat
            .messages
            .iter()
            .map(|message| match message {
                ChatMessageV1::UserParts(parts) => self
                    .visible_parts(parts, &mut media)
                    .map(ChatMessageV1::User),
                message => Ok(message.clone()),
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.check_conversation(chat, &messages)?;
        let mut prompt = Prompt {
            descriptor,
            text: String::new(),
        };
        prompt.write(BEGIN)?;
        effort_preamble(&mut prompt, effort)?;
        if !chat.tools.is_empty() {
            self.tool_list(&mut prompt, chat.tools, tool_objects)?;
        }
        let results = Results {
            references,
            functions: function_objects(chat.tools, tool_objects),
        };
        self.message_loop(&mut prompt, &messages, &results, clear_thinking)?;
        // transformers continues the final message where its text ends in
        // the render, which for this template is the prompt's end, and
        // refuses one whose stripped text ([`continued_text`]) the template
        // did not keep (text before an inline `<think>`, lines 145-147).
        if let Some(text) = continued
            && !prompt.text.contains(python_strip_v1(text))
        {
            return Err(field_v1(
                "continue_final_message requires a final message the template keeps whole",
            ));
        }
        // Lines 259-261. A continued final message ends the prompt, so it
        // never takes the generation prompt (transformers refuses the two
        // together).
        if chat.add_generation_prompt && !chat.continue_final_message {
            prompt.write("<|assistant|><think>")?;
        }
        split_media_segments_v1(descriptor, prompt.text, media)
    }

    /// Each tool message's `tool_reference` names (lines 81-82), taken out
    /// of the message so the rest reads as a text result. A message holds
    /// them alone: the template reads a content list whose first part is a
    /// reference as references and drops its other parts, and one that
    /// begins with text as text, dropping the references. They need offered
    /// tools, since the template defines `tool_references_to_response` only
    /// beside its tool list (lines 5, 18-31).
    fn take_tool_references(
        &self,
        messages: &mut [FrontendMessageV1],
        tools: &[FrontendToolV1],
    ) -> Result<Vec<Option<Vec<String>>>, FrontendErrorV1> {
        let refuse = |message: String| FrontendErrorV1::new(message, Some("messages"));
        messages
            .iter_mut()
            .map(|message| {
                let Some(parts) = &message.parts else {
                    return Ok(None);
                };
                if !parts
                    .iter()
                    .any(|part| matches!(part, FrontendContentPartV1::ToolReference { .. }))
                {
                    return Ok(None);
                }
                if message.role != FrontendRoleV1::Tool {
                    return Err(refuse(
                        "only tool messages may contain tool_reference parts".into(),
                    ));
                }
                let names = parts
                    .iter()
                    .map(|part| match part {
                        FrontendContentPartV1::ToolReference { name } => Some(name.clone()),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        refuse(field_v1(
                            "a tool message's tool_reference parts cannot share it with other \
                             content",
                        ))
                    })?;
                if tools.is_empty() {
                    return Err(refuse(field_v1(
                        "tool_reference parts require offered tools",
                    )));
                }
                if names
                    .iter()
                    .any(|name| reserved_media_text_v1(&self.descriptor, name))
                {
                    return Err(refuse(field_v1(
                        "reserved media boundary or placeholder in client text",
                    )));
                }
                message.parts = None;
                message.content.get_or_insert_with(String::new);
                Ok(Some(names))
            })
            .collect()
    }

    /// What the conversation must be before anything is written: a
    /// continued final turn has content and no tool calls, the counts fit the declared
    /// bounds, and no message is one the template would drop or misread (a
    /// developer message, which it renders as nothing, or a message's own
    /// tools, which it ignores).
    fn check_conversation(
        &self,
        chat: &ChatRenderV1<'_>,
        messages: &[ChatMessageV1],
    ) -> Result<(), String> {
        let limits = self.descriptor.limits;
        if chat.continue_final_message
            && !matches!(messages.last(), Some(ChatMessageV1::Assistant(turn)) if turn.tool_calls.is_empty())
        {
            return Err(field_v1(
                "continue_final_message requires a final assistant message without tool calls",
            ));
        }
        // transformers continues the final message's content, and refuses
        // one that has none.
        if chat.continue_final_message
            && matches!(messages.last(), Some(ChatMessageV1::Assistant(turn)) if turn.content.is_none())
        {
            return Err(field_v1(
                "continue_final_message requires a final assistant message with content",
            ));
        }
        if messages.len() > limits.maximum_messages || chat.tools.len() > limits.maximum_tools {
            return Err(resource_v1("too many messages or tools"));
        }
        for message in messages {
            match message {
                ChatMessageV1::Developer(_) => {
                    return Err(field_v1("developer role is not declared"));
                }
                ChatMessageV1::WithTools { .. } => {
                    return Err(field_v1("message tools are not declared"));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// `visible_text` (lines 54-74) of a user's content list: each text
    /// part's text, and `emit_image` or `emit_video` (51-52) for a media
    /// part, whose pad is the item's placeholder. No run of joined client
    /// text may form a media boundary or pad.
    fn visible_parts(
        &self,
        parts: &[FrontendContentPartV1],
        media: &mut Vec<FrontendMediaPartV1>,
    ) -> Result<String, String> {
        let descriptor = &self.descriptor;
        let reserved = |text: &str| reserved_media_text_v1(descriptor, text);
        let mut content = String::new();
        let mut text_start = 0;
        for part in parts {
            match part {
                FrontendContentPartV1::Text(text) => content.push_str(text),
                // Only a tool message holds one, and it was taken out.
                FrontendContentPartV1::ToolReference { .. } => {
                    return Err(TOOL_REFERENCES_UNSUPPORTED_V1.into());
                }
                FrontendContentPartV1::Media(item) => {
                    if reserved(&content[text_start..]) {
                        return Err(field_v1("reserved media string in joined client text"));
                    }
                    media_kind_v1(descriptor, item)?;
                    let (open, _, close) = match item.kind {
                        FrontendMediaPartKindV1::Image => IMAGE_LABELS,
                        FrontendMediaPartKindV1::Video => VIDEO_LABELS,
                        // `open` refused a declaration that serves audio.
                        FrontendMediaPartKindV1::Audio => {
                            return Err(field_v1("media part type is not declared by this bundle"));
                        }
                    };
                    content.push_str(open);
                    content.push_str(&media_placeholder_v1(media.len()));
                    media.push(item.clone());
                    content.push_str(close);
                    text_start = content.len();
                }
            }
        }
        if reserved(&content[text_start..]) {
            return Err(field_v1("reserved media string in joined client text"));
        }
        Ok(content)
    }

    /// The tool list (lines 5, 32-50): each offered tool's function object
    /// (`tool_to_json`) on its own line, except a deferred one.
    fn tool_list(
        &self,
        prompt: &mut Prompt<'_>,
        tools: &[FrontendToolV1],
        tool_objects: &[FrontendJsonV1],
    ) -> Result<(), String> {
        prompt.write(TOOLS_HEADER)?;
        if tool_objects.is_empty() {
            // A request that carries no function objects (one not read
            // through the transport) has each tool's declared fields.
            for tool in tools {
                validate_name_v1(&self.descriptor, &tool.name, "tool")?;
                prompt.write(&tool_to_json(&declared_function(tool)))?;
                prompt.write("\n")?;
            }
            return prompt.write(TOOLS_FOOTER);
        }
        if tool_objects.len() != tools.len() {
            return Err(field_v1("every offered tool needs its function object"));
        }
        for (tool, object) in tools.iter().zip(tool_objects) {
            validate_name_v1(&self.descriptor, &tool.name, "tool")?;
            // Lines 40-42: the transport passes each tool's function.
            let FrontendJsonV1::Object(fields) = object else {
                return Err(field_v1("a tool's function must be an object"));
            };
            // Line 43.
            if defers_loading_v1(fields) {
                continue;
            }
            // Line 44, and the newline after it.
            let mut line = String::new();
            write_tool_object_v1(
                &mut line,
                fields,
                FrontendJsonEncodingV1::PythonJsonV1,
                self.descriptor.limits.maximum_prompt_bytes,
            )?;
            prompt.write(&line)?;
            prompt.write("\n")?;
        }
        prompt.write(TOOLS_FOOTER)
    }

    /// The message loop (lines 132-258): `ns.last_user_index` (132-137),
    /// then each message under its role.
    fn message_loop(
        &self,
        prompt: &mut Prompt<'_>,
        messages: &[ChatMessageV1],
        results: &Results<'_>,
        clear_thinking: bool,
    ) -> Result<(), String> {
        let last_user_index = messages
            .iter()
            .rposition(|message| matches!(message, ChatMessageV1::User(_)));
        for (index, message) in messages.iter().enumerate() {
            match message {
                // Line 139.
                ChatMessageV1::User(content) => {
                    prompt.write("<|user|>")?;
                    prompt.write(content)?;
                }
                ChatMessageV1::Assistant(turn) => {
                    // Line 149: earlier reasoning renders unless cleared.
                    let keep_reasoning =
                        !clear_thinking || last_user_index.is_none_or(|last| index > last);
                    self.assistant_turn(prompt, turn, keep_reasoning)?;
                }
                ChatMessageV1::Tool(_) => {
                    // Line 166: a run of results renders whole at its first.
                    if index == 0 || !matches!(messages[index - 1], ChatMessageV1::Tool(_)) {
                        observation(prompt, messages, results, index)?;
                    }
                }
                // Lines 255-256.
                ChatMessageV1::System(content) => {
                    prompt.write("<|system|>")?;
                    prompt.write(content)?;
                }
                // Refused by `check_conversation`; user parts were made text.
                ChatMessageV1::Developer(_)
                | ChatMessageV1::WithTools { .. }
                | ChatMessageV1::UserParts(_) => {}
            }
        }
        Ok(())
    }

    /// One assistant turn (lines 140-164): the reasoning block, the stripped
    /// answer and the tool calls.
    fn assistant_turn(
        &self,
        prompt: &mut Prompt<'_>,
        turn: &AssistantTurnV1,
        keep_reasoning: bool,
    ) -> Result<(), String> {
        prompt.write("<|assistant|>")?;
        // Line 142: the content's visible text, empty when it is null.
        let mut content = turn.content.as_deref().unwrap_or_default();
        // Lines 143-148: `reasoning_content`, else the text inside a
        // `<think>` the content closes.
        let mut reasoning = None;
        if let Some(text) = &turn.reasoning_content {
            reasoning = Some(text.as_str());
        } else if content.contains("</think>") {
            let before = python_split_v1(content, "</think>")[0];
            reasoning = python_split_v1(before, "<think>").last().copied();
            content = python_split_v1(content, "</think>")
                .last()
                .copied()
                .unwrap_or_default();
        }
        // Lines 149-153.
        prompt.write("<think>")?;
        if keep_reasoning && let Some(reasoning) = reasoning {
            prompt.write(reasoning)?;
        }
        prompt.write("</think>")?;
        // Lines 154-156, with Python's whitespace.
        prompt.write(python_strip_v1(content))?;
        // Lines 157-164.
        if turn.tool_calls.len() > self.descriptor.limits.maximum_tool_calls {
            return Err(resource_v1("too many tool calls"));
        }
        for call in &turn.tool_calls {
            self.tool_call(prompt, call)?;
        }
        Ok(())
    }

    /// One tool call (lines 159-163): the name, then each argument's key and
    /// value, a string as itself and any other value as `tojson`.
    fn tool_call(&self, prompt: &mut Prompt<'_>, call: &FrontendToolCallV1) -> Result<(), String> {
        let limits = self.descriptor.limits;
        validate_name_v1(&self.descriptor, &call.name, "tool call")?;
        let arguments = parse_json_object_v1(&call.arguments, limits, "tool arguments")?;
        if arguments.len() > limits.maximum_tool_parameters {
            return Err(resource_v1("too many tool parameters"));
        }
        prompt.write("<tool_call>")?;
        prompt.write(&call.name)?;
        for (key, value) in &arguments {
            validate_name_v1(&self.descriptor, key, "tool parameter")?;
            prompt.write("<arg_key>")?;
            prompt.write(key)?;
            prompt.write("</arg_key><arg_value>")?;
            match value {
                FrontendJsonV1::String(text) => prompt.write(text)?,
                value => prompt.write(&tojson_v1(value, &Default::default()))?,
            }
            prompt.write("</arg_value>")?;
        }
        prompt.write("</tool_call>")
    }
}

/// What an observation block reads beyond its messages: each message's
/// `tool_reference` names, and the offered tools' function objects that
/// `tool_references_to_response` looks them up in.
struct Results<'a> {
    references: &'a [Option<Vec<String>>],
    functions: Vec<Vec<(String, FrontendJsonV1)>>,
}

/// One tool message of an observation block: its result, and its
/// `tool_reference` names when it holds them.
struct ToolMessage<'a> {
    result: &'a ToolResultV1,
    references: Option<&'a [String]>,
}

/// One observation block (lines 165-254): the run of tool results from
/// `block_start`, in the order of the calls before it when every result and
/// call can be matched by id (181-224), otherwise as sent.
fn observation(
    prompt: &mut Prompt<'_>,
    messages: &[ChatMessageV1],
    results: &Results<'_>,
    block_start: usize,
) -> Result<(), String> {
    prompt.write("<|observation|>")?;
    // Lines 168-176: the block runs while messages are tool results.
    let block: Vec<ToolMessage<'_>> = messages[block_start..]
        .iter()
        .zip(&results.references[block_start..])
        .map_while(|(message, references)| match message {
            ChatMessageV1::Tool(result) => Some(ToolMessage {
                result,
                references: references.as_deref(),
            }),
            _ => None,
        })
        .collect();
    // Lines 177-180: the calls of the assistant turn just before it.
    let calls = block_start
        .checked_sub(1)
        .and_then(|previous| match &messages[previous] {
            ChatMessageV1::Assistant(turn) if !turn.tool_calls.is_empty() => {
                Some(turn.tool_calls.as_slice())
            }
            _ => None,
        });
    match calls {
        // Lines 225-248.
        Some(calls) if can_sort(&block, calls) => {
            for call in calls {
                let call_id = id_of_call(call);
                for message in &block {
                    if id_of_result(message) == call_id {
                        render_tool_response(prompt, message, &results.functions)?;
                    }
                }
            }
        }
        // Lines 249-253.
        _ => {
            for message in &block {
                render_tool_response(prompt, message, &results.functions)?;
            }
        }
    }
    Ok(())
}

/// The text transformers continues in a final message: the last text part
/// of a content list (it finds the continuation there), else the content.
/// `None` for a message without content, which `check_conversation` refuses.
fn continued_text(message: Option<&FrontendMessageV1>) -> Option<String> {
    let message = message?;
    match &message.parts {
        Some(parts) => parts.iter().rev().find_map(|part| match part {
            FrontendContentPartV1::Text(text) => Some(text.clone()),
            _ => None,
        }),
        None => message.content.clone(),
    }
}

/// Lines 2-3: the effort the template reads, capitalized, in a system block.
fn effort_preamble(prompt: &mut Prompt<'_>, effort: &str) -> Result<(), String> {
    let effective = if ["low", "high"].contains(&effort) {
        effort
    } else {
        "max"
    };
    prompt.write("<|system|>Reasoning Effort: ")?;
    prompt.write(&python_capitalize_v1(effective))
}

/// `tool_to_json` (lines 6-17) of a function object built from a tool's
/// declared fields.
fn tool_to_json(fields: &[(String, FrontendJsonV1)]) -> String {
    let mut line = String::new();
    write_tool_object_v1(
        &mut line,
        fields,
        FrontendJsonEncodingV1::PythonJsonV1,
        usize::MAX,
    )
    .expect("an unbounded line takes every object");
    line
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

/// The offered tools' function objects as the template's `tools` holds
/// them: the client's objects as sent, or each tool's declared fields.
fn function_objects(
    tools: &[FrontendToolV1],
    tool_objects: &[FrontendJsonV1],
) -> Vec<Vec<(String, FrontendJsonV1)>> {
    if tool_objects.is_empty() {
        return tools.iter().map(declared_function).collect();
    }
    tool_objects
        .iter()
        .filter_map(|object| match object {
            FrontendJsonV1::Object(fields) => Some(fields.clone()),
            _ => None,
        })
        .collect()
}

/// `render_tool_response` (lines 78-94): a result of references, or of
/// text in `tool_response` (75-77).
fn render_tool_response(
    prompt: &mut Prompt<'_>,
    message: &ToolMessage<'_>,
    functions: &[Vec<(String, FrontendJsonV1)>],
) -> Result<(), String> {
    if let Some(names) = message.references {
        return tool_references_to_response(prompt, names, functions);
    }
    prompt.write("<tool_response>")?;
    prompt.write(&message.result.content)?;
    prompt.write("</tool_response>")
}

/// `tool_references_to_response` (lines 18-31): for each referenced name,
/// every offered tool of that name (a deferred one too) as `tool_to_json`,
/// each on its own line, in one `tool_response`. A name no tool has writes
/// nothing.
fn tool_references_to_response(
    prompt: &mut Prompt<'_>,
    names: &[String],
    functions: &[Vec<(String, FrontendJsonV1)>],
) -> Result<(), String> {
    prompt.write("<tool_response><tools>\n")?;
    for name in names {
        for fields in functions {
            let named = fields.iter().any(|(key, value)| {
                key == "name" && matches!(value, FrontendJsonV1::String(text) if text == name)
            });
            if named {
                prompt.write(&tool_to_json(fields))?;
                prompt.write("\n")?;
            }
        }
    }
    prompt.write("</tools></tool_response>")
}

/// `id_of` (lines 95-101) of a tool result: its `tool_call_id`.
fn id_of_result<'a>(message: &ToolMessage<'a>) -> &'a str {
    &message.result.tool_call_id
}

/// `id_of` (lines 95-101) of a tool call: its `id`.
fn id_of_call(call: &FrontendToolCallV1) -> &str {
    &call.id
}

/// `has_dup_tool_result_id` (lines 105-121): whether two results of the
/// block name `target`.
fn has_dup_tool_result_id(block: &[ToolMessage<'_>], target: &str) -> bool {
    block
        .iter()
        .filter(|result| id_of_result(result) == target)
        .nth(1)
        .is_some()
}

/// `tc_id_exists` (lines 122-131).
fn tc_id_exists(calls: &[FrontendToolCallV1], target: &str) -> bool {
    calls.iter().any(|call| id_of_call(call) == target)
}

/// Lines 181-224: whether every result names a different call that exists,
/// and every call has a distinct, non-empty id.
fn can_sort(block: &[ToolMessage<'_>], calls: &[FrontendToolCallV1]) -> bool {
    // Lines 200-209 (a list of outputs, 188-199, never reaches the render).
    let results_match = block.iter().all(|result| {
        let id = id_of_result(result);
        !id.is_empty() && !has_dup_tool_result_id(block, id) && tc_id_exists(calls, id)
    });
    // Lines 211-223.
    let calls_distinct = calls.iter().enumerate().all(|(index, call)| {
        let id = id_of_call(call);
        !id.is_empty()
            && !calls[index + 1..]
                .iter()
                .any(|other| id_of_call(other) == id)
    });
    results_match && calls_distinct
}
