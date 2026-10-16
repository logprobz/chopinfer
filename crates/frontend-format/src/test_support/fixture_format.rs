//! A minimal chat format for tests that render through a `format.parser`
//! declaration, such as the parser fixtures.
//!
//! It stands for a model's own renderer ([`crate::ChatFormatV1`]) where a
//! test needs some prompt but asserts nothing about its layout. It reads the
//! declaration's parser section for every marker it writes, so its prompts
//! read back through the shared parsers, and it writes the turns in one fixed
//! layout:
//!
//! - a user-side turn (system, developer, user, tool) opens with the first
//!   `turn_open` token that is not the assistant header, then the role name
//!   and a newline; a tool turn names its call ID after the role;
//! - an assistant turn opens with `assistant_header` and a newline, then its
//!   reasoning between the parser's reasoning markers, its content and its
//!   tool calls in the parser's tool grammar;
//! - every turn ends with a newline, except a continued final assistant turn;
//! - the generation prompt is the assistant header and a newline, followed by
//!   the reasoning open in thinking mode;
//! - the selected effort's preamble, the offered tools (one JSON object per
//!   line) and the response format each come first as a system turn;
//! - each media item is written bare, with no labels, so the format opens
//!   over any modality;
//! - client text may not carry a reserved marker or a media placeholder,
//!   even split across adjacent text parts.

use crate::descriptor::{FormatDescriptorV1, FrontendParserDeclarationV1, FrontendToolGrammarV1};
use crate::format::{FormatRenderRequestV1, FormatRenderedV1};
use crate::request::{
    FrontendContentPartV1, FrontendErrorV1, FrontendJsonV1, FrontendMessageV1, FrontendRoleV1,
    FrontendToolCallV1, FrontendToolV1, TOOL_REFERENCES_UNSUPPORTED_V1,
};
use crate::toolkit::{
    append_v1, field_v1, json_to_string_v1, media_kind_v1, media_placeholder_v1,
    parse_json_object_v1, preserve_thinking_v1, reasoning_open_v1, reserved_free_v1,
    reserved_media_text_v1, select_mode_v1, split_media_segments_v1, validate_name_v1,
};
use crate::{ChatFormatV1, OpenedV1, PromptModeV1};

/// The test chat format: see the module documentation for its layout.
#[derive(Debug)]
pub struct FixtureFormatV1 {
    descriptor: FormatDescriptorV1,
    user_header: String,
}

impl FixtureFormatV1 {
    fn parser(&self) -> &FrontendParserDeclarationV1 {
        &self.descriptor.parser
    }
}

impl ChatFormatV1 for FixtureFormatV1 {
    fn open(descriptor: &FormatDescriptorV1) -> Result<Self, String> {
        let parser = &descriptor.parser;
        let user_header = parser
            .turn_open
            .iter()
            .find(|token| **token != parser.assistant_header)
            .ok_or("the fixture format needs a turn_open token besides the assistant header")?
            .clone();
        Ok(Self {
            descriptor: descriptor.clone(),
            user_header,
        })
    }

    fn opened(&self) -> OpenedV1 {
        OpenedV1::default()
    }

    fn render(
        &self,
        request: FormatRenderRequestV1<'_>,
    ) -> Result<FormatRenderedV1, FrontendErrorV1> {
        let (mode, effort) = select_mode_v1(&self.descriptor.reasoning, &request)?;
        let preserve_thinking = preserve_thinking_v1(false, &request)?;
        let mut writer = WriterV1 {
            format: self,
            prompt: String::new(),
            media: Vec::new(),
        };
        writer
            .conversation(&request, mode, &effort.preamble, preserve_thinking)
            .map_err(|message| FrontendErrorV1::new(message, Some("messages")))?;
        let WriterV1 { prompt, media, .. } = writer;
        let (prompt, segments, media) = split_media_segments_v1(&self.descriptor, prompt, media)
            .map_err(|message| FrontendErrorV1::new(message, Some("messages")))?;
        let parser = self.parser();
        let reasoning_open = reasoning_open_v1(
            mode,
            request.continue_final_message,
            &parser.reasoning_open,
            &parser.reasoning_close,
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

/// One render in progress.
struct WriterV1<'a> {
    format: &'a FixtureFormatV1,
    prompt: String,
    media: Vec<crate::request::FrontendMediaPartV1>,
}

impl WriterV1<'_> {
    fn conversation(
        &mut self,
        request: &FormatRenderRequestV1<'_>,
        mode: PromptModeV1,
        preamble: &str,
        preserve_thinking: bool,
    ) -> Result<(), String> {
        if mode == PromptModeV1::Thinking && !preamble.is_empty() {
            self.user_turn("system", preamble)?;
        }
        if !request.tools.is_empty() {
            let mut list = String::new();
            for (index, tool) in request.tools.iter().enumerate() {
                if index != 0 {
                    list.push('\n');
                }
                list.push_str(&self.tool(tool)?);
            }
            self.user_turn("tools", &list)?;
        }
        if let Some(response_format) = &request.response_format {
            self.screen_json(response_format, "response_format")?;
            let text = self.json(response_format, "response_format")?;
            self.user_turn("format", &text)?;
        }
        let last_user = request
            .messages
            .iter()
            .rposition(|message| message.role == FrontendRoleV1::User);
        let count = request.messages.len();
        for (index, message) in request.messages.iter().enumerate() {
            let continued = request.continue_final_message && index + 1 == count;
            if continued && message.role != FrontendRoleV1::Assistant {
                return Err(field_v1(
                    "continue_final_message requires a final assistant message",
                ));
            }
            let reasoning = preserve_thinking || last_user.is_none_or(|last| index > last);
            self.message(message, reasoning, continued)?;
        }
        let continued = request.continue_final_message && count > 0;
        if request.add_generation_prompt && !continued {
            let parser = self.format.parser();
            let mut header = format!("{}\n", parser.assistant_header);
            if mode == PromptModeV1::Thinking {
                header.push_str(&parser.reasoning_open);
            }
            self.append(&header)?;
        }
        Ok(())
    }

    fn message(
        &mut self,
        message: &FrontendMessageV1,
        reasoning: bool,
        continued: bool,
    ) -> Result<(), String> {
        if !message.tools.is_empty() {
            return Err(field_v1("the fixture format renders no message tools"));
        }
        // The format's open answer admits no reference, in any message.
        if message
            .parts
            .iter()
            .flatten()
            .any(|part| matches!(part, FrontendContentPartV1::ToolReference { .. }))
        {
            return Err(TOOL_REFERENCES_UNSUPPORTED_V1.to_owned());
        }
        let role = match message.role {
            FrontendRoleV1::System => "system",
            FrontendRoleV1::Developer => "developer",
            FrontendRoleV1::User => "user",
            FrontendRoleV1::Assistant => return self.assistant(message, reasoning, continued),
            FrontendRoleV1::Tool => {
                let id = message
                    .tool_call_id
                    .as_deref()
                    .ok_or_else(|| field_v1("tool message requires tool_call_id"))?;
                self.screen(id, "tool_call_id")?;
                let content = message.content.as_deref().unwrap_or_default();
                return self.user_turn(&format!("tool {id}"), content);
            }
        };
        let Some(parts) = &message.parts else {
            return self.user_turn(role, message.content.as_deref().unwrap_or_default());
        };
        let header = format!("{}{role}\n", self.format.user_header);
        self.append(&header)?;
        // Adjacent text parts render as one string, so they are screened
        // joined, as a model's renderer screens them.
        let mut joined = String::new();
        for part in parts {
            match part {
                FrontendContentPartV1::Text(text) => {
                    joined.push_str(text);
                    self.screen(&joined, "content")?;
                    self.append(text)?;
                }
                FrontendContentPartV1::Media(media) => {
                    joined.clear();
                    media_kind_v1(&self.format.descriptor, media)?;
                    let placeholder = media_placeholder_v1(self.media.len());
                    self.append(&placeholder)?;
                    self.media.push(media.clone());
                }
                // Refused above.
                FrontendContentPartV1::ToolReference { .. } => {
                    return Err(TOOL_REFERENCES_UNSUPPORTED_V1.to_owned());
                }
            }
        }
        self.append("\n")
    }

    fn assistant(
        &mut self,
        message: &FrontendMessageV1,
        reasoning: bool,
        continued: bool,
    ) -> Result<(), String> {
        let parser = self.format.parser();
        let mut turn = format!("{}\n", parser.assistant_header);
        if let Some(text) = message.reasoning_content.as_deref().filter(|_| reasoning) {
            self.screen(text, "reasoning_content")?;
            turn.push_str(&parser.reasoning_open);
            turn.push_str(text);
            turn.push_str(&parser.reasoning_close);
        }
        if let Some(content) = &message.content {
            self.screen(content, "content")?;
            turn.push_str(content);
        }
        if continued && !message.tool_calls.is_empty() {
            return Err(field_v1(
                "continue_final_message requires a final assistant message without tool calls",
            ));
        }
        for (index, call) in message.tool_calls.iter().enumerate() {
            if index != 0 {
                turn.push_str(&parser.tool_call_separator);
            }
            turn.push_str(&self.call(call)?);
        }
        if !continued {
            turn.push('\n');
        }
        self.append(&turn)
    }

    /// One tool call in the parser's tool grammar.
    fn call(&self, call: &FrontendToolCallV1) -> Result<String, String> {
        let descriptor = &self.format.descriptor;
        let parser = self.format.parser();
        validate_name_v1(descriptor, &call.name, "tool call")?;
        self.screen(&call.id, "tool call id")?;
        let arguments =
            parse_json_object_v1(&call.arguments, descriptor.limits, "tool call arguments")?;
        let key_value = parser.tool_grammar == FrontendToolGrammarV1::XmlKeyValueArgumentsV1;
        let mut text = parser.tool_call_open.clone();
        if !key_value {
            text.push_str(&parser.function_open);
        }
        text.push_str(&call.name);
        if !key_value {
            text.push_str(&parser.function_name_close);
        }
        for (name, value) in &arguments {
            validate_name_v1(descriptor, name, "tool call parameter")?;
            self.screen_json(value, "tool call arguments")?;
            text.push_str(&parser.parameter_open);
            text.push_str(name);
            text.push_str(&parser.parameter_name_close);
            text.push_str(&parser.parameter_value_open);
            match value {
                FrontendJsonV1::String(value) => text.push_str(value),
                value => text.push_str(&self.json(value, "tool call arguments")?),
            }
            text.push_str(&parser.parameter_close);
        }
        if !key_value {
            text.push_str(&parser.function_close);
        }
        text.push_str(&parser.tool_call_close);
        Ok(text)
    }

    /// One offered tool as a JSON object of its name, description and
    /// parameters.
    fn tool(&self, tool: &FrontendToolV1) -> Result<String, String> {
        validate_name_v1(&self.format.descriptor, &tool.name, "tool")?;
        let mut fields = vec![("name".to_owned(), FrontendJsonV1::String(tool.name.clone()))];
        if let Some(description) = &tool.description {
            fields.push((
                "description".to_owned(),
                FrontendJsonV1::String(description.clone()),
            ));
        }
        if tool.parameters_present {
            fields.push(("parameters".to_owned(), tool.parameters.clone()));
        }
        let object = FrontendJsonV1::Object(fields);
        self.screen_json(&object, "tools")?;
        self.json(&object, "tools")
    }

    fn user_turn(&mut self, role: &str, text: &str) -> Result<(), String> {
        self.screen(text, "content")?;
        let turn = format!("{}{role}\n{text}\n", self.format.user_header);
        self.append(&turn)
    }

    fn json(&self, value: &FrontendJsonV1, context: &str) -> Result<String, String> {
        let encoding = self.format.parser().json_encoding;
        json_to_string_v1(value, encoding, context)
    }

    /// Refuses client text that carries a reserved marker, a media
    /// placeholder or a media pad.
    fn screen(&self, text: &str, context: &str) -> Result<(), String> {
        reserved_free_v1(text, context, &self.format.parser().reserved_markers)?;
        if reserved_media_text_v1(&self.format.descriptor, text) {
            return Err(field_v1(
                "reserved media boundary or placeholder in client text",
            ));
        }
        Ok(())
    }

    fn screen_json(&self, value: &FrontendJsonV1, context: &str) -> Result<(), String> {
        match value {
            FrontendJsonV1::String(text) => self.screen(text, context),
            FrontendJsonV1::Array(items) => items
                .iter()
                .try_for_each(|item| self.screen_json(item, context)),
            FrontendJsonV1::Object(entries) => entries.iter().try_for_each(|(name, value)| {
                self.screen(name, context)?;
                self.screen_json(value, context)
            }),
            FrontendJsonV1::Null | FrontendJsonV1::Bool(_) | FrontendJsonV1::Number(_) => Ok(()),
        }
    }

    fn append(&mut self, text: &str) -> Result<(), String> {
        let maximum = self.format.descriptor.limits.maximum_prompt_bytes;
        append_v1(&mut self.prompt, text, maximum)
    }
}
