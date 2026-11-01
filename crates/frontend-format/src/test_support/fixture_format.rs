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
//!
//! Under `vocabulary.explicit_special_tokens` the format names its special
//! tokens: each declared boundary token the layout writes (in a turn header,
//! a reasoning marker, a tool-call marker or an effort preamble) is a special
//! segment, and client text is plain text, so it may spell any of them and
//! only a media placeholder is refused in it.

use crate::descriptor::{FormatDescriptorV1, FrontendParserDeclarationV1, FrontendToolGrammarV1};
use crate::format::{FormatRenderRequestV1, FormatRenderedV1};
use crate::request::{
    FrontendContentPartV1, FrontendErrorV1, FrontendJsonV1, FrontendMessageV1, FrontendRoleV1,
    FrontendToolCallV1, FrontendToolV1, TOOL_REFERENCES_UNSUPPORTED_V1,
};
use crate::toolkit::{
    MarkedPromptV1, field_v1, json_to_string_v1, media_kind_v1, media_placeholder_v1,
    parse_json_object_v1, preserve_thinking_v1, reasoning_open_segments_v1, reasoning_open_v1,
    reserved_free_v1, reserved_media_text_v1, select_mode_v1, split_marked_media_segments_v1,
    validate_name_v1, validate_parameter_name_v1,
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

    /// Whether the format names its special tokens.
    fn explicit(&self) -> bool {
        self.descriptor.vocabulary.explicit_special_tokens
    }

    /// Writes text of the layout to `piece`: under explicit special tokens,
    /// each declared boundary token in it, longest first, as a special token.
    fn layout(&self, piece: &mut MarkedPromptV1, text: &str) -> Result<(), String> {
        if !self.explicit() {
            return piece.push_plain(text, usize::MAX);
        }
        let (mut plain, mut at) = (0, 0);
        while let Some(character) = text[at..].chars().next() {
            let token = self
                .descriptor
                .vocabulary
                .boundary_tokens
                .iter()
                .map(|token| token.text.as_str())
                .filter(|token| !token.is_empty() && text[at..].starts_with(token))
                .max_by_key(|token| token.len());
            let Some(token) = token else {
                at += character.len_utf8();
                continue;
            };
            piece.push_plain(&text[plain..at], usize::MAX)?;
            piece.push_special(token, usize::MAX)?;
            at += token.len();
            plain = at;
        }
        piece.push_plain(&text[plain..], usize::MAX)
    }
}

/// Client text, which is plain text under explicit special tokens too.
fn client(piece: &mut MarkedPromptV1, text: &str) -> Result<(), String> {
    piece.push_plain(text, usize::MAX)
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
            prompt: MarkedPromptV1::new(),
            media: Vec::new(),
        };
        writer
            .conversation(&request, mode, &effort.preamble, preserve_thinking)
            .map_err(|message| FrontendErrorV1::new(message, Some("messages")))?;
        let WriterV1 { prompt, media, .. } = writer;
        // Without a special token, this splits as `split_media_segments_v1`.
        let (prompt, segments, media) =
            split_marked_media_segments_v1(&self.descriptor, prompt, media)
                .map_err(|message| FrontendErrorV1::new(message, Some("messages")))?;
        let parser = self.parser();
        let reasoning_open = if self.explicit() {
            reasoning_open_segments_v1(
                mode,
                request.continue_final_message,
                &parser.reasoning_open,
                &parser.reasoning_close,
                &segments,
            )
        } else {
            reasoning_open_v1(
                mode,
                request.continue_final_message,
                &parser.reasoning_open,
                &parser.reasoning_close,
                &prompt,
            )
        };
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
    prompt: MarkedPromptV1,
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
            let mut text = MarkedPromptV1::new();
            self.format.layout(&mut text, preamble)?;
            self.user_turn("system", None, &text)?;
        }
        if !request.tools.is_empty() {
            let mut list = String::new();
            for (index, tool) in request.tools.iter().enumerate() {
                if index != 0 {
                    list.push('\n');
                }
                list.push_str(&self.tool(tool)?);
            }
            self.client_turn("tools", None, &list)?;
        }
        if let Some(response_format) = &request.response_format {
            self.screen_json(response_format, "response_format")?;
            let text = self.json(response_format, "response_format")?;
            self.client_turn("format", None, &text)?;
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
            self.layout(&header)?;
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
                return self.client_turn("tool", Some(id), content);
            }
        };
        let Some(parts) = &message.parts else {
            return self.client_turn(role, None, message.content.as_deref().unwrap_or_default());
        };
        let header = format!("{}{role}\n", self.format.user_header);
        self.layout(&header)?;
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
        self.layout("\n")
    }

    fn assistant(
        &mut self,
        message: &FrontendMessageV1,
        reasoning: bool,
        continued: bool,
    ) -> Result<(), String> {
        let format = self.format;
        let parser = format.parser();
        let mut turn = MarkedPromptV1::new();
        format.layout(&mut turn, &format!("{}\n", parser.assistant_header))?;
        if let Some(text) = message.reasoning_content.as_deref().filter(|_| reasoning) {
            self.screen(text, "reasoning_content")?;
            format.layout(&mut turn, &parser.reasoning_open)?;
            client(&mut turn, text)?;
            format.layout(&mut turn, &parser.reasoning_close)?;
        }
        if let Some(content) = &message.content {
            self.screen(content, "content")?;
            client(&mut turn, content)?;
        }
        if continued && !message.tool_calls.is_empty() {
            return Err(field_v1(
                "continue_final_message requires a final assistant message without tool calls",
            ));
        }
        for (index, call) in message.tool_calls.iter().enumerate() {
            if index != 0 {
                format.layout(&mut turn, &parser.tool_call_separator)?;
            }
            self.call(&mut turn, call)?;
        }
        if !continued {
            format.layout(&mut turn, "\n")?;
        }
        self.push(&turn)
    }

    /// Writes one tool call in the parser's tool grammar to `text`.
    fn call(&self, text: &mut MarkedPromptV1, call: &FrontendToolCallV1) -> Result<(), String> {
        let format = self.format;
        let descriptor = &format.descriptor;
        let parser = format.parser();
        validate_name_v1(descriptor, &call.name, "tool call")?;
        self.screen(&call.id, "tool call id")?;
        let arguments =
            parse_json_object_v1(&call.arguments, descriptor.limits, "tool call arguments")?;
        let key_value = parser.tool_grammar == FrontendToolGrammarV1::XmlKeyValueArgumentsV1;
        format.layout(text, &parser.tool_call_open)?;
        if !key_value {
            format.layout(text, &parser.function_open)?;
        }
        client(text, &call.name)?;
        if !key_value {
            format.layout(text, &parser.function_name_close)?;
        }
        for (name, value) in &arguments {
            validate_parameter_name_v1(descriptor, name, "tool call parameter")?;
            self.screen_json(value, "tool call arguments")?;
            format.layout(text, &parser.parameter_open)?;
            client(text, name)?;
            format.layout(text, &parser.parameter_name_close)?;
            format.layout(text, &parser.parameter_value_open)?;
            match value {
                FrontendJsonV1::String(value) => client(text, value)?,
                value => client(text, &self.json(value, "tool call arguments")?)?,
            }
            format.layout(text, &parser.parameter_close)?;
        }
        if !key_value {
            format.layout(text, &parser.function_close)?;
        }
        format.layout(text, &parser.tool_call_close)
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

    /// One user-side turn of client text, which may name a tool call ID
    /// after its role.
    fn client_turn(&mut self, role: &str, id: Option<&str>, text: &str) -> Result<(), String> {
        let mut body = MarkedPromptV1::new();
        client(&mut body, text)?;
        self.user_turn(role, id, &body)
    }

    fn user_turn(
        &mut self,
        role: &str,
        id: Option<&str>,
        text: &MarkedPromptV1,
    ) -> Result<(), String> {
        self.screen(text.text(), "content")?;
        let format = self.format;
        let mut turn = MarkedPromptV1::new();
        format.layout(&mut turn, &format!("{}{role}", format.user_header))?;
        if let Some(id) = id {
            format.layout(&mut turn, " ")?;
            client(&mut turn, id)?;
        }
        format.layout(&mut turn, "\n")?;
        turn.push_prompt(text, usize::MAX)?;
        format.layout(&mut turn, "\n")?;
        self.push(&turn)
    }

    fn json(&self, value: &FrontendJsonV1, context: &str) -> Result<String, String> {
        let encoding = self.format.parser().json_encoding;
        json_to_string_v1(value, encoding, context)
    }

    /// Refuses client text that carries a media placeholder or a media pad
    /// (`reserved_media_text_v1`), or, unless client text is plain text, a
    /// reserved marker.
    fn screen(&self, text: &str, context: &str) -> Result<(), String> {
        if !self.format.explicit() {
            reserved_free_v1(text, context, &self.format.parser().reserved_markers)?;
        }
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

    /// Appends client text to the prompt.
    fn append(&mut self, text: &str) -> Result<(), String> {
        let maximum = self.format.descriptor.limits.maximum_prompt_bytes;
        self.prompt.push_plain(text, maximum)
    }

    /// Appends text of the layout to the prompt.
    fn layout(&mut self, text: &str) -> Result<(), String> {
        let mut piece = MarkedPromptV1::new();
        self.format.layout(&mut piece, text)?;
        self.push(&piece)
    }

    fn push(&mut self, piece: &MarkedPromptV1) -> Result<(), String> {
        let maximum = self.format.descriptor.limits.maximum_prompt_bytes;
        self.prompt.push_prompt(piece, maximum)
    }
}
