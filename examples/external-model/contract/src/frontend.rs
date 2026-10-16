//! The example's chat format: the renderer its bundle exports, written with
//! the `frontend-format` toolkit as a new model writes its own.
//!
//! A model ports its chat template here, one function per block of the
//! template, and declares in its frontend component's `format.parser`
//! section only what the shared parsers read of a completion. The renderer
//! writes the role markers in code; the bundle exports it with
//! `frontend_format::export_chat_format_v1!`.
//!
//! The example's template is small: the beginning of sequence, the
//! reasoning preamble in thinking mode, the offered tools and response
//! format as one system block, each message under its role marker, a tool
//! result group under the tool marker, and a generation transition that
//! opens reasoning in thinking mode and closes it otherwise. Tool calls use
//! the key-value grammar (`xml-key-value-arguments-v1`) over the example's
//! markers: `<calls>name<param>key<name><value>value</param></calls>`.

use frontend_format::toolkit::{
    append_v1, chat_message_v1, field_v1, json_to_string_v1, parse_json_object_v1,
    reasoning_open_v1, reject_non_integral_numbers_v1, reserved_media_text_v1, resource_v1,
    screen_media_text_v1, select_mode_v1, sequence_v1, split_media_segments_v1, validate_chat_v1,
    validate_name_v1, write_json_v1,
};
use frontend_format::{
    AssistantTurnV1, ChatFormatV1, ChatMessageV1, ChatRenderV1, FormatDescriptorV1,
    FormatRenderRequestV1, FormatRenderedV1, FrontendContentPartV1, FrontendErrorV1,
    FrontendJsonEncodingV1, FrontendJsonV1, FrontendToolCallV1, OpenedV1, PromptModeV1,
    TOOL_REFERENCES_UNSUPPORTED_V1,
};

use crate::MARKERS;

const BEGIN: &str = MARKERS[0];
const END: &str = MARKERS[1];
const SYSTEM: &str = MARKERS[2];
const USER: &str = MARKERS[3];
const ASSISTANT: &str = MARKERS[4];
const TOOL: &str = MARKERS[5];
const THINKING_OPEN: &str = MARKERS[6];
const THINKING_CLOSE: &str = MARKERS[7];
const CALLS_OPEN: &str = MARKERS[8];
const CALLS_CLOSE: &str = MARKERS[9];
const RESULT_OPEN: &str = MARKERS[10];
const RESULT_CLOSE: &str = MARKERS[11];
const PARAMETER_OPEN: &str = MARKERS[12];
const PARAMETER_NAME_CLOSE: &str = MARKERS[13];
const PARAMETER_VALUE_OPEN: &str = MARKERS[14];
const PARAMETER_CLOSE: &str = MARKERS[15];

/// The example's chat format, over its declaration.
pub struct ExampleFormat {
    descriptor: FormatDescriptorV1,
    /// How the prompt writes JSON: the parser section's encoding, which
    /// parsed tool-call arguments are read back in.
    encoding: FrontendJsonEncodingV1,
}

impl ChatFormatV1 for ExampleFormat {
    fn open(descriptor: &FormatDescriptorV1) -> Result<Self, String> {
        // The example serves text only, so it writes no media labels.
        if descriptor.multimodal.is_some() {
            return Err("the example's chat format renders no media".into());
        }
        let encoding = descriptor.parser.json_encoding;
        Ok(Self {
            descriptor: descriptor.clone(),
            encoding,
        })
    }

    fn opened(&self) -> OpenedV1 {
        OpenedV1::default()
    }

    fn render(
        &self,
        request: FormatRenderRequestV1<'_>,
    ) -> Result<FormatRenderedV1, FrontendErrorV1> {
        let descriptor = &self.descriptor;
        let (mode, effort) = select_mode_v1(&descriptor.reasoning, &request)?;
        // The prompt writes every number back as its exact integer.
        for tool in request.tools {
            reject_non_integral_numbers_v1(&tool.parameters, "tools")
                .map_err(|message| FrontendErrorV1::new(message, Some("tools")))?;
        }
        if let Some(response_format) = &request.response_format {
            reject_non_integral_numbers_v1(response_format, "response_format")
                .map_err(|message| FrontendErrorV1::new(message, Some("response_format")))?;
        }
        let continue_final_message = request.continue_final_message;
        let messages = request
            .messages
            .into_iter()
            .map(|message| chat_message_v1(message).and_then(|message| self.text(message)))
            .collect::<Result<Vec<_>, _>>()?;
        let render = ChatRenderV1 {
            mode,
            preamble: &effort.preamble,
            messages: &messages,
            continue_final_message,
            add_generation_prompt: request.add_generation_prompt,
            tools: request.tools,
            response_format: request.response_format.as_ref(),
            preserve_thinking: true,
        };
        let refused = |message| FrontendErrorV1::new(message, Some("messages"));
        screen_media_text_v1(descriptor, &render, &request.tool_objects).map_err(refused)?;
        validate_chat_v1(descriptor, &render).map_err(refused)?;
        let mut writer = Writer {
            descriptor,
            encoding: self.encoding,
            prompt: String::new(),
        };
        writer.conversation(&render).map_err(refused)?;
        let (prompt, segments, media) =
            split_media_segments_v1(descriptor, writer.prompt, Vec::new()).map_err(refused)?;
        let reasoning_open = reasoning_open_v1(
            mode,
            continue_final_message,
            THINKING_OPEN,
            THINKING_CLOSE,
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

impl ExampleFormat {
    /// A user message's content parts as the one text they join to; the
    /// example serves no media.
    fn text(&self, message: ChatMessageV1) -> Result<ChatMessageV1, FrontendErrorV1> {
        let ChatMessageV1::UserParts(parts) = message else {
            return Ok(message);
        };
        let refused = |message| FrontendErrorV1::new(message, Some("messages"));
        let mut content = String::new();
        for part in parts {
            match part {
                FrontendContentPartV1::Text(text) => content.push_str(&text),
                FrontendContentPartV1::Media(_) => {
                    return Err(refused(field_v1(
                        "this bundle does not support media inputs",
                    )));
                }
                // The conversation checks refuse a reference before this;
                // the example's open answer admits none.
                FrontendContentPartV1::ToolReference { .. } => {
                    return Err(refused(TOOL_REFERENCES_UNSUPPORTED_V1.into()));
                }
            }
        }
        if reserved_media_text_v1(&self.descriptor, &content) {
            return Err(refused(field_v1(
                "reserved media string in joined client text",
            )));
        }
        Ok(ChatMessageV1::User(content))
    }
}

/// One prompt in progress.
struct Writer<'a> {
    descriptor: &'a FormatDescriptorV1,
    encoding: FrontendJsonEncodingV1,
    prompt: String,
}

impl Writer<'_> {
    fn append(&mut self, text: &str) -> Result<(), String> {
        let maximum = self.descriptor.limits.maximum_prompt_bytes;
        append_v1(&mut self.prompt, text, maximum)
    }

    /// The whole conversation, then the generation transition.
    fn conversation(&mut self, render: &ChatRenderV1<'_>) -> Result<(), String> {
        let thinking = render.mode == PromptModeV1::Thinking;
        self.append(BEGIN)?;
        if thinking {
            self.append(render.preamble)?;
        }
        self.request_controls(render)?;
        let messages = render.messages;
        let mut index = 0;
        while index < messages.len() {
            let more = index + 1 < messages.len();
            let next_is_assistant = messages
                .get(index + 1)
                .is_none_or(|next| matches!(next, ChatMessageV1::Assistant(_)));
            let transition = next_is_assistant && (more || render.add_generation_prompt);
            match &messages[index] {
                ChatMessageV1::System(content) => {
                    self.append(SYSTEM)?;
                    self.append(content)?;
                }
                ChatMessageV1::Developer(content) => {
                    self.append(SYSTEM)?;
                    self.append(content)?;
                    if transition {
                        self.transition(thinking)?;
                    }
                }
                ChatMessageV1::User(content) => {
                    self.append(USER)?;
                    self.append(content)?;
                    if transition {
                        self.transition(thinking)?;
                    }
                }
                ChatMessageV1::Assistant(turn) => {
                    let close = more || !render.continue_final_message;
                    self.assistant(turn, thinking, close)?;
                }
                ChatMessageV1::Tool(_) => {
                    index = self.tool_results(messages, index)?;
                    if index < messages.len() || render.add_generation_prompt {
                        self.transition(thinking)?;
                    }
                    continue;
                }
                // The conversation checks refuse both before anything is
                // written.
                ChatMessageV1::UserParts(_) | ChatMessageV1::WithTools { .. } => {
                    return Err(sequence_v1("the example renders text messages only"));
                }
            }
            index += 1;
        }
        Ok(())
    }

    /// The offered tools and the response format, as one system block
    /// before the first message.
    fn request_controls(&mut self, render: &ChatRenderV1<'_>) -> Result<(), String> {
        let encoding = self.encoding;
        if !render.tools.is_empty() {
            self.append(SYSTEM)?;
            self.append("Tools\n")?;
            for (index, tool) in render.tools.iter().enumerate() {
                if index != 0 {
                    self.append("\n")?;
                }
                let mut entries =
                    vec![("name".to_owned(), FrontendJsonV1::String(tool.name.clone()))];
                if let Some(description) = &tool.description {
                    entries.push((
                        "description".to_owned(),
                        FrontendJsonV1::String(description.clone()),
                    ));
                }
                entries.push(("parameters".to_owned(), tool.parameters.clone()));
                if let Some(strict) = tool.strict {
                    entries.push(("strict".to_owned(), FrontendJsonV1::Bool(strict)));
                }
                let mut schema = String::new();
                write_json_v1(
                    &FrontendJsonV1::Object(entries),
                    encoding,
                    "tool schema",
                    &mut schema,
                )?;
                self.append(&schema)?;
            }
            self.append("\n")?;
        }
        if let Some(response_format) = render.response_format {
            if !render.tools.is_empty() {
                self.append("\n")?;
            }
            self.append("Format\n")?;
            self.append(&json_to_string_v1(
                response_format,
                encoding,
                "response format",
            )?)?;
        }
        Ok(())
    }

    /// The assistant marker, opening reasoning in thinking mode and closing
    /// it otherwise.
    fn transition(&mut self, thinking: bool) -> Result<(), String> {
        self.append(ASSISTANT)?;
        self.append(if thinking {
            THINKING_OPEN
        } else {
            THINKING_CLOSE
        })
    }

    /// One assistant turn after its transition: reasoning in thinking mode,
    /// the answer, its calls, and the end of sequence unless the request
    /// continues it.
    fn assistant(
        &mut self,
        turn: &AssistantTurnV1,
        thinking: bool,
        close: bool,
    ) -> Result<(), String> {
        if thinking {
            self.append(turn.reasoning_content.as_deref().unwrap_or_default())?;
            self.append(THINKING_CLOSE)?;
        }
        self.append(turn.content.as_deref().unwrap_or_default())?;
        for call in &turn.tool_calls {
            self.tool_call(call)?;
        }
        if close {
            self.append(END)?;
        }
        Ok(())
    }

    /// One call in the key-value grammar: the name, then each argument's key
    /// and value, a string raw and any other value as JSON.
    fn tool_call(&mut self, call: &FrontendToolCallV1) -> Result<(), String> {
        let limits = self.descriptor.limits;
        let arguments = parse_json_object_v1(
            &call.arguments,
            limits,
            &format!("arguments for tool '{}'", call.name),
        )?;
        if arguments.len() > limits.maximum_tool_parameters {
            return Err(resource_v1(format!(
                "tool '{}' has too many parameters",
                call.name
            )));
        }
        self.append(CALLS_OPEN)?;
        self.append(&call.name)?;
        for (name, value) in &arguments {
            validate_name_v1(self.descriptor, name, "tool parameter")?;
            self.append(PARAMETER_OPEN)?;
            self.append(name)?;
            self.append(PARAMETER_NAME_CLOSE)?;
            self.append(PARAMETER_VALUE_OPEN)?;
            match value {
                FrontendJsonV1::String(value) => self.append(value)?,
                value => {
                    let encoding = self.encoding;
                    self.append(&json_to_string_v1(value, encoding, "tool argument")?)?;
                }
            }
            self.append(PARAMETER_CLOSE)?;
        }
        self.append(CALLS_CLOSE)
    }

    /// The tool result group that starts at `start`, answered in the order
    /// of the calls before it; returns the index after the group. The
    /// conversation checks put an assistant with calls before the group and
    /// one result for each call in it.
    fn tool_results(&mut self, messages: &[ChatMessageV1], start: usize) -> Result<usize, String> {
        self.append(TOOL)?;
        let end = start
            + messages[start..]
                .iter()
                .take_while(|message| matches!(message, ChatMessageV1::Tool(_)))
                .count();
        let Some(ChatMessageV1::Assistant(assistant)) = start.checked_sub(1).map(|i| &messages[i])
        else {
            return Err(sequence_v1("tool message does not follow an assistant"));
        };
        for (index, call) in assistant.tool_calls.iter().enumerate() {
            if index != 0 {
                self.append("\n")?;
            }
            let result = messages[start..end]
                .iter()
                .find_map(|message| match message {
                    ChatMessageV1::Tool(result) if result.tool_call_id == call.id => {
                        Some(result.content.as_str())
                    }
                    _ => None,
                })
                .ok_or_else(|| {
                    sequence_v1(format!("missing tool result for call '{}'", call.id))
                })?;
            self.append(RESULT_OPEN)?;
            self.append(result)?;
            self.append(RESULT_CLOSE)?;
        }
        Ok(end)
    }
}
