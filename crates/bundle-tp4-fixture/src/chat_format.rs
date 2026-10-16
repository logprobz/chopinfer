//! The chat format this fixture's module exports: its own renderer, written
//! with the `frontend-format` toolkit as a model's renderer is.
//!
//! It writes the prompt the fixture has always served, from the published
//! TP4 markers: the beginning-of-sequence pair, the reasoning preamble in
//! thinking mode, the offered tools and response format as one system block,
//! each message under its role marker, a tool result group under the
//! observation marker, and a generation transition that opens reasoning in
//! thinking mode and closes it otherwise. An assistant history turn writes
//! its tool calls in the key-value grammar its `format.parser` section
//! declares, so the history the model reads is the syntax it generates. An
//! image renders bare, with no labels, as its pad.

use frontend_format::toolkit::{
    append_v1, chat_message_v1, field_v1, json_to_string_v1, media_kind_v1, media_placeholder_v1,
    parse_json_object_v1, reasoning_open_v1, reject_non_integral_numbers_v1,
    reserved_media_text_v1, resource_v1, screen_media_text_v1, select_mode_v1, sequence_v1,
    split_media_segments_v1, validate_chat_v1, validate_name_v1, write_json_v1,
};
use frontend_format::{
    AssistantTurnV1, ChatFormatV1, ChatMessageV1, ChatRenderV1, FormatDescriptorV1,
    FormatRenderRequestV1, FormatRenderedV1, FrontendContentPartV1, FrontendErrorV1,
    FrontendJsonEncodingV1, FrontendJsonV1, FrontendToolCallV1, OpenedV1, PromptModeV1,
    TOOL_REFERENCES_UNSUPPORTED_V1,
};

use crate::frontend::tp4_fixture_begin_marker_v1;
use crate::metadata::{TP4_TEXT_PROMPT_SPECIAL_TOKENS as MARKERS, tokenizer_metadata};

const SYSTEM: &str = MARKERS[2];
const USER: &str = MARKERS[3];
const ASSISTANT: &str = MARKERS[4];
const OBSERVATION: &str = MARKERS[5];
const THINKING_OPEN: &str = MARKERS[6];
const THINKING_CLOSE: &str = MARKERS[7];
const TOOL_CALL_OPEN: &str = MARKERS[8];
const TOOL_CALL_CLOSE: &str = MARKERS[9];
const TOOL_RESPONSE_OPEN: &str = MARKERS[10];
const TOOL_RESPONSE_CLOSE: &str = MARKERS[11];
const ARGUMENT_KEY_OPEN: &str = MARKERS[12];
const ARGUMENT_KEY_CLOSE: &str = MARKERS[13];
const ARGUMENT_VALUE_OPEN: &str = MARKERS[14];
const ARGUMENT_VALUE_CLOSE: &str = MARKERS[15];

/// The fixture's chat format, over its declaration.
pub struct Tp4FixtureFormatV1 {
    descriptor: FormatDescriptorV1,
    /// How the prompt writes JSON: the parser section's encoding, which
    /// parsed tool-call arguments are read back in.
    encoding: FrontendJsonEncodingV1,
}

impl ChatFormatV1 for Tp4FixtureFormatV1 {
    fn open(descriptor: &FormatDescriptorV1) -> Result<Self, String> {
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
            .map(chat_message_v1)
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
        let messages_error = |message| FrontendErrorV1::new(message, Some("messages"));
        let (prompt, media) = self
            .prompt(&render, &request.tool_objects)
            .map_err(messages_error)?;
        let (prompt, segments, media) =
            split_media_segments_v1(descriptor, prompt, media).map_err(messages_error)?;
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

impl Tp4FixtureFormatV1 {
    /// The prompt with a placeholder for each media item, and the items.
    fn prompt(
        &self,
        render: &ChatRenderV1<'_>,
        tool_objects: &[FrontendJsonV1],
    ) -> Result<(String, Vec<frontend_format::FrontendMediaPartV1>), String> {
        let descriptor = &self.descriptor;
        screen_media_text_v1(descriptor, render, tool_objects)?;
        let mut media = Vec::new();
        let mut messages = Vec::with_capacity(render.messages.len());
        for message in render.messages {
            let ChatMessageV1::UserParts(parts) = message else {
                messages.push(message.clone());
                continue;
            };
            let mut content = String::new();
            let mut text_start = 0;
            for part in parts {
                match part {
                    FrontendContentPartV1::Text(text) => content.push_str(text),
                    // The conversation checks refuse a reference before
                    // this; the fixture's open answer admits none.
                    FrontendContentPartV1::ToolReference { .. } => {
                        return Err(TOOL_REFERENCES_UNSUPPORTED_V1.to_owned());
                    }
                    FrontendContentPartV1::Media(item) => {
                        // Adjacent text parts render as one string.
                        if reserved_media_text_v1(descriptor, &content[text_start..]) {
                            return Err(field_v1("reserved media string in joined client text"));
                        }
                        media_kind_v1(descriptor, item)?;
                        content.push_str(&media_placeholder_v1(media.len()));
                        media.push(item.clone());
                        text_start = content.len();
                    }
                }
            }
            if reserved_media_text_v1(descriptor, &content[text_start..]) {
                return Err(field_v1("reserved media string in joined client text"));
            }
            messages.push(ChatMessageV1::User(content));
        }
        let render = ChatRenderV1 {
            messages: &messages,
            ..*render
        };
        validate_chat_v1(descriptor, &render)?;
        let mut writer = WriterV1 {
            descriptor,
            encoding: self.encoding,
            prompt: String::new(),
        };
        writer.conversation(&render)?;
        Ok((writer.prompt, media))
    }
}

/// One prompt in progress.
struct WriterV1<'a> {
    descriptor: &'a FormatDescriptorV1,
    encoding: FrontendJsonEncodingV1,
    prompt: String,
}

impl WriterV1<'_> {
    fn append(&mut self, text: &str) -> Result<(), String> {
        append_v1(
            &mut self.prompt,
            text,
            self.descriptor.limits.maximum_prompt_bytes,
        )
    }

    fn conversation(&mut self, render: &ChatRenderV1<'_>) -> Result<(), String> {
        let thinking = render.mode == PromptModeV1::Thinking;
        self.append(&tp4_fixture_begin_marker_v1())?;
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
                // The media parts are joined above, and the conversation
                // checks refuse message tools before anything is written.
                ChatMessageV1::UserParts(_) | ChatMessageV1::WithTools { .. } => {
                    return Err(sequence_v1("the fixture renders text messages only"));
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
        let mut separate = false;
        if !render.tools.is_empty() {
            self.append(&format!("{SYSTEM}\n# Tools\n"))?;
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
                let mut rendered = String::new();
                write_json_v1(
                    &FrontendJsonV1::Object(entries),
                    encoding,
                    "tool schema",
                    &mut rendered,
                )?;
                self.append(&rendered)?;
            }
            self.append("\n")?;
            separate = true;
        }
        if let Some(response_format) = render.response_format {
            if separate {
                self.append("\n")?;
            }
            self.append("# Response format\n")?;
            self.append(&json_to_string_v1(
                response_format,
                encoding,
                "response format",
            )?)?;
        }
        Ok(())
    }

    fn transition(&mut self, thinking: bool) -> Result<(), String> {
        self.append(ASSISTANT)?;
        self.append(if thinking {
            THINKING_OPEN
        } else {
            THINKING_CLOSE
        })
    }

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
            self.append(tokenizer_metadata().eos_token())?;
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
        self.append(TOOL_CALL_OPEN)?;
        self.append(&call.name)?;
        for (name, value) in &arguments {
            validate_name_v1(self.descriptor, name, "tool parameter")?;
            self.append(ARGUMENT_KEY_OPEN)?;
            self.append(name)?;
            self.append(ARGUMENT_KEY_CLOSE)?;
            self.append(ARGUMENT_VALUE_OPEN)?;
            match value {
                FrontendJsonV1::String(value) => self.append(value)?,
                value => self.append(&json_to_string_v1(value, self.encoding, "tool argument")?)?,
            }
            self.append(ARGUMENT_VALUE_CLOSE)?;
        }
        self.append(TOOL_CALL_CLOSE)
    }

    /// The tool result group that starts at `start`, answered in the order
    /// of the calls before it; returns the index after the group. The
    /// conversation checks put an assistant with calls before the group and
    /// one result for each call in it.
    fn tool_results(&mut self, messages: &[ChatMessageV1], start: usize) -> Result<usize, String> {
        self.append(OBSERVATION)?;
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
            self.append(TOOL_RESPONSE_OPEN)?;
            self.append(result)?;
            self.append(TOOL_RESPONSE_CLOSE)?;
        }
        Ok(end)
    }
}

#[cfg(test)]
mod tests {
    use frontend_format::{
        ChatFormatV1, FormatDescriptorV1, FormatRenderRequestV1, FrontendMessageV1, FrontendRoleV1,
        FrontendToolCallV1,
    };

    use super::Tp4FixtureFormatV1;
    use crate::frontend::tp4_fixture_frontend_declaration_v1;

    fn message(role: FrontendRoleV1, content: &str) -> FrontendMessageV1 {
        FrontendMessageV1 {
            role,
            content: Some(content.into()),
            parts: None,
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tools: Vec::new(),
        }
    }

    /// A history turn's calls render in the key-value grammar the parser
    /// section reads, and their results answer them in call order.
    #[test]
    fn a_history_turn_writes_its_calls_in_the_parsed_grammar() {
        let descriptor =
            FormatDescriptorV1::from_component_bytes(&tp4_fixture_frontend_declaration_v1())
                .unwrap();
        let format = Tp4FixtureFormatV1::open(&descriptor).unwrap();
        let mut call = message(FrontendRoleV1::Assistant, "Ok.");
        call.tool_calls = vec![FrontendToolCallV1 {
            id: "a".into(),
            name: "get_weather".into(),
            arguments: r#"{"city": "Paris", "days": 2}"#.into(),
        }];
        let mut result = message(FrontendRoleV1::Tool, "sunny");
        result.tool_call_id = Some("a".into());
        let rendered = format
            .render(FormatRenderRequestV1 {
                messages: vec![message(FrontendRoleV1::User, "Weather?"), call, result],
                continue_final_message: false,
                add_generation_prompt: true,
                tools: &[],
                tool_objects: Vec::new(),
                response_format: None,
                reasoning_effort: None,
                enable_thinking: None,
                thinking: None,
                nested_reasoning_effort: None,
                preserve_thinking: None,
                clear_thinking: None,
                add_vision_id: None,
            })
            .unwrap();
        assert_eq!(
            rendered.prompt,
            "[gMASK]<sop><|user|>Weather?<|assistant|></think>Ok.<tool_call>get_weather\
             <arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key>\
             <arg_value>2</arg_value></tool_call><|endoftext|><|observation|>\
             <tool_response>sunny</tool_response><|assistant|></think>"
        );
    }
}
