//! The byte documents that cross the `chop.frontend.v1` entries.
//!
//! [`crate::frontend`] fixes how a document crosses; this module fixes what
//! each one holds:
//!
//! | Entry | Request | Answer |
//! |---|---|---|
//! | `open` | the authenticated frontend component | [`OpenedV1`] |
//! | `render` | [`RenderRequestV1`] | [`RenderedV1`] |
//! | `constraint` | [`ConstraintRequestV1`] | a `runtime-grammar` wire document, or empty |
//! | `parser_open` | [`ParserOpenV1`] | none |
//! | `parser_push` | [`PushV1`] | [`EventsV1`] |
//!
//! Every refusal is a [`RefusalV1`]. The ABI layer itself refuses a sink of
//! another layout with plain UTF-8 text, which [`RefusalV1::read`] accepts
//! as a message with no parameter.
//!
//! Each document is a little-endian byte string: an 8-byte magic, a `u32`
//! layout ([`FRONTEND_DOCUMENT_LAYOUT_V1`]), then fixed-width fields. A flag
//! is one byte, 0 or 1; an optional field is a flag and, when it is 1, the
//! value; a byte string or text is a `u32` length and its bytes; a list is a
//! `u32` count and its entries; a byte offset is a `u64`. Text is UTF-8. A
//! JSON value crosses as its JSON text, which the reader parses again.
//! Readers refuse another magic or layout, an unknown tag, a flag other than
//! 0 or 1, text that is not UTF-8, a count past
//! [`FRONTEND_DOCUMENT_ENTRIES_MAX_V1`] or past the bytes left, parser state
//! past [`FRONTEND_PARSER_STATE_BYTES_MAX_V1`], and trailing bytes. A parser
//! push runs once per generated token, so the push request and the event
//! answer borrow their bytes rather than copy them.

use std::fmt;
use std::ops::Range;

const OPENED_MAGIC: &[u8; 8] = b"CHOPFOPN";
const RENDER_REQUEST_MAGIC: &[u8; 8] = b"CHOPFRRQ";
const RENDERED_MAGIC: &[u8; 8] = b"CHOPFRND";
const REFUSAL_MAGIC: &[u8; 8] = b"CHOPFREF";
const CONSTRAINT_MAGIC: &[u8; 8] = b"CHOPFCRQ";
const PARSER_OPEN_MAGIC: &[u8; 8] = b"CHOPFPOP";
const PUSH_MAGIC: &[u8; 8] = b"CHOPFPSH";
const EVENTS_MAGIC: &[u8; 8] = b"CHOPFEVT";

/// The layout of every frontend document.
pub const FRONTEND_DOCUMENT_LAYOUT_V1: u32 = 1;
/// Most entries in any one list of a frontend document.
pub const FRONTEND_DOCUMENT_ENTRIES_MAX_V1: usize = 1 << 20;
/// The longest parser state a render may return or a parser open carry.
pub const FRONTEND_PARSER_STATE_BYTES_MAX_V1: usize = 1 << 20;
/// Most message roles an open answer may declare beyond the five every
/// frontend reads.
pub const FRONTEND_EXTRA_ROLES_MAX_V1: usize = 16;
/// The longest role name an open answer may declare.
pub const FRONTEND_ROLE_NAME_BYTES_MAX_V1: usize = 64;

/// Why a frontend document did not decode.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrontendDocumentErrorV1(pub String);

impl fmt::Display for FrontendDocumentErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "frontend document: {}", self.0)
    }
}

impl std::error::Error for FrontendDocumentErrorV1 {}

type DocumentResultV1<T> = Result<T, FrontendDocumentErrorV1>;

fn fail<T>(reason: impl Into<String>) -> DocumentResultV1<T> {
    Err(FrontendDocumentErrorV1(reason.into()))
}

/// The mode a prompt rendered in.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModeV1 {
    Chat,
    Thinking,
}

/// One media kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaKindV1 {
    Image,
    Audio,
    Video,
}

/// One media part, by its place among the request's media parts. The bytes
/// and their source stay with the host.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaPartV1 {
    /// The part's index among every media part of the request, in order.
    pub index: u32,
    pub kind: MediaKindV1,
    /// The content-part type the client sent, such as `image_url`.
    pub part_type: String,
}

/// One content part of a message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PartV1 {
    Text(String),
    Media(MediaPartV1),
    /// A tool message's `tool_reference` part: the name of the tool it
    /// references. The host sends one only to a frontend whose open answer
    /// declares [`OpenedV1::tool_references`].
    ToolReference(String),
}

/// One tool call a conversation already contains.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolCallV1 {
    pub id: String,
    pub name: String,
    /// The arguments as the client sent them.
    pub arguments: String,
}

/// One message of a conversation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageV1 {
    /// The role as the client spelled it.
    pub role: String,
    pub content: Option<String>,
    pub parts: Option<Vec<PartV1>>,
    /// The message's `reasoning` field.
    pub reasoning: Option<String>,
    /// The message's `reasoning_content` field.
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCallV1>,
    pub tool_call_id: Option<String>,
    /// Message-level tools, as JSON texts.
    pub tools: Vec<String>,
}

/// One function tool a request offered.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolV1 {
    pub name: String,
    pub description: Option<String>,
    /// The parameter schema as JSON text.
    pub parameters: String,
    /// Whether the client sent a parameter schema.
    pub parameters_present: bool,
    pub strict: Option<bool>,
}

/// One chat request to render.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderRequestV1 {
    pub messages: Vec<MessageV1>,
    pub continue_final_message: bool,
    pub add_generation_prompt: bool,
    /// Every offered tool, whatever the tool choice.
    pub tools: Vec<ToolV1>,
    /// Each offered tool's function object as JSON text, when the host
    /// forwards them; otherwise empty.
    pub tool_objects: Vec<String>,
    /// The `response_format` as JSON text.
    pub response_format: Option<String>,
    /// The top-level `reasoning_effort`.
    pub reasoning_effort: Option<String>,
    /// The `chat_template_kwargs` object as JSON text.
    pub chat_template_kwargs: Option<String>,
}

/// One piece of a rendered prompt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SegmentV1 {
    Text(String),
    /// The media part with this request index.
    Media(u32),
}

/// One rendered prompt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderedV1 {
    pub segments: Vec<SegmentV1>,
    pub mode: ModeV1,
    /// The selected reasoning effort's declared name.
    pub effort: String,
    /// Whether the prompt leaves reasoning open, so the completion begins in
    /// reasoning.
    pub reasoning_open: bool,
    /// Opaque state the host hands back when it opens this request's parser.
    pub parser_state: Vec<u8>,
}

/// What a frontend reads of a chat request besides the fields every
/// request document carries, which the host's transport then carries for
/// it: the `open` answer.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OpenedV1 {
    /// Whether the render reads each offered tool's function object as the
    /// client sent it, so the transport carries the objects whole.
    pub reads_tool_objects: bool,
    /// Whether every assistant history turn reads `reasoning_content` when
    /// it has no `reasoning`.
    pub reasoning_content_fallback: bool,
    /// Whether an assistant history turn that carries tool calls reads
    /// `reasoning_content` when it has no `reasoning`.
    pub tool_call_reasoning_fallback: bool,
    /// The message roles the render reads besides system, developer, user,
    /// assistant and tool, which the transport then admits as text messages.
    /// The host bounds them ([`FRONTEND_EXTRA_ROLES_MAX_V1`],
    /// [`FRONTEND_ROLE_NAME_BYTES_MAX_V1`]).
    pub extra_roles: Vec<String>,
    /// Whether the render reads `tool_reference` content parts in tool
    /// messages, which the transport then admits there and carries as
    /// [`PartV1::ToolReference`].
    pub tool_references: bool,
}

/// A refusal of any entry: the message and the request parameter it blames.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefusalV1 {
    pub message: String,
    pub parameter: Option<String>,
}

/// Whether a request permitted tool calls, or forced them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolChoiceV1 {
    Auto,
    None,
    Required,
    Named(String),
}

/// What the `constraint` entry is asked.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConstraintRequestV1 {
    /// The grammar `choice` imposes on calls to `tools`. The answer is a
    /// grammar wire document, or empty when the choice imposes nothing.
    ToolChoice {
        tools: Vec<ToolV1>,
        choice: ToolChoiceV1,
    },
}

/// What one completion parser is fixed to when it opens.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParserOpenV1 {
    pub mode: ModeV1,
    pub reasoning_open: bool,
    pub extract_tools: bool,
    pub stream_tool_calls: bool,
    pub raw: bool,
    pub ignore_eos: bool,
    pub engine_tokens: bool,
    pub tool_schemas: Vec<ToolV1>,
    /// The state the render returned, or empty when nothing rendered.
    pub parser_state: Vec<u8>,
}

/// The completion bytes one push releases.
///
/// The parser holds the bytes of earlier pushes. The host only appends,
/// except that a client stop may cut the completion short before the last
/// push, so a push keeps the first `keep` of them and appends `suffix`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PushV1<'a> {
    pub keep: u64,
    /// Whether the host holds bytes after these.
    pub held: bool,
    /// `None` until the completion passes the request's minimum token
    /// count; then the byte offset an end marker must end after.
    pub stop_floor: Option<u64>,
    pub suffix: &'a [u8],
}

/// Why a completion ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReasonV1 {
    Stop,
    Length,
    ToolCalls,
}

/// One tool call a parser read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CallV1<'a> {
    pub name: &'a str,
    pub arguments: &'a str,
}

/// One streamed piece of a tool call's arguments.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeltaV1<'a> {
    /// The call's index among every call of the completion.
    pub index: u32,
    /// The call's name, on its first delta only.
    pub name: Option<&'a str>,
    pub arguments: &'a str,
}

/// The whole answer, at the completion end.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompletionV1<'a> {
    pub content: &'a str,
    pub reasoning_content: Option<&'a str>,
    pub tool_calls: Vec<CallV1<'a>>,
}

/// One thing a parser settled. Ranges are byte offsets into the completion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EventV1<'a> {
    Reasoning(Range<u64>),
    Content(Range<u64>),
    ToolCalls {
        range: Range<u64>,
        calls: Vec<CallV1<'a>>,
    },
    ToolCallDeltas(Vec<DeltaV1<'a>>),
    Finish {
        reason: FinishReasonV1,
        completion: Option<CompletionV1<'a>>,
    },
}

/// What one push settled, and where the bytes the parser has not settled
/// begin.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EventsV1<'a> {
    pub events: Vec<EventV1<'a>>,
    pub cursor: u64,
}

struct WriterV1<'a>(&'a mut Vec<u8>);

impl WriterV1<'_> {
    fn start(&mut self, magic: &[u8; 8]) {
        self.0.extend_from_slice(magic);
        self.u32(FRONTEND_DOCUMENT_LAYOUT_V1);
    }
    fn u8(&mut self, value: u8) {
        self.0.push(value);
    }
    fn u32(&mut self, value: u32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }
    fn u64(&mut self, value: u64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }
    fn flag(&mut self, value: bool) {
        self.u8(u8::from(value));
    }
    fn count(&mut self, count: usize) {
        self.u32(u32::try_from(count).expect("a frontend document list fits a u32 count"));
    }
    fn bytes(&mut self, bytes: &[u8]) {
        self.count(bytes.len());
        self.0.extend_from_slice(bytes);
    }
    fn text(&mut self, text: &str) {
        self.bytes(text.as_bytes());
    }
    fn optional_text(&mut self, text: Option<&str>) {
        self.flag(text.is_some());
        if let Some(text) = text {
            self.text(text);
        }
    }
    fn texts(&mut self, texts: &[String]) {
        self.count(texts.len());
        for text in texts {
            self.text(text);
        }
    }
    fn mode(&mut self, mode: ModeV1) {
        self.u8(match mode {
            ModeV1::Chat => 0,
            ModeV1::Thinking => 1,
        });
    }
    fn tools(&mut self, tools: &[ToolV1]) {
        self.count(tools.len());
        for tool in tools {
            self.text(&tool.name);
            self.optional_text(tool.description.as_deref());
            self.text(&tool.parameters);
            self.flag(tool.parameters_present);
            self.flag(tool.strict.is_some());
            if let Some(strict) = tool.strict {
                self.flag(strict);
            }
        }
    }
    fn range(&mut self, range: &Range<u64>) {
        self.u64(range.start);
        self.u64(range.end);
    }
    fn calls(&mut self, calls: &[CallV1<'_>]) {
        self.count(calls.len());
        for call in calls {
            self.text(call.name);
            self.text(call.arguments);
        }
    }
}

struct ReaderV1<'a> {
    bytes: &'a [u8],
}

impl<'a> ReaderV1<'a> {
    fn open(bytes: &'a [u8], magic: &[u8; 8], document: &str) -> DocumentResultV1<Self> {
        let mut reader = Self { bytes };
        if reader.take(8)? != magic {
            return fail(format!("not a {document} document"));
        }
        let layout = reader.u32()?;
        if layout != FRONTEND_DOCUMENT_LAYOUT_V1 {
            return fail(format!(
                "{document} layout {layout}; this build reads layout {FRONTEND_DOCUMENT_LAYOUT_V1}"
            ));
        }
        Ok(reader)
    }
    fn take(&mut self, count: usize) -> DocumentResultV1<&'a [u8]> {
        if count > self.bytes.len() {
            return fail("document ends early");
        }
        let (head, tail) = self.bytes.split_at(count);
        self.bytes = tail;
        Ok(head)
    }
    fn u8(&mut self) -> DocumentResultV1<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> DocumentResultV1<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }
    fn u64(&mut self) -> DocumentResultV1<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }
    fn flag(&mut self) -> DocumentResultV1<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            other => fail(format!("flag {other} is neither 0 nor 1")),
        }
    }
    /// A list count, which each entry's first byte bounds by the bytes left.
    fn count(&mut self) -> DocumentResultV1<usize> {
        let count = self.u32()? as usize;
        if count > FRONTEND_DOCUMENT_ENTRIES_MAX_V1 {
            return fail(format!(
                "list of {count} entries exceeds {FRONTEND_DOCUMENT_ENTRIES_MAX_V1}"
            ));
        }
        if count > self.bytes.len() {
            return fail("document ends early");
        }
        Ok(count)
    }
    fn bytes(&mut self) -> DocumentResultV1<&'a [u8]> {
        let length = self.u32()? as usize;
        self.take(length)
    }
    fn parser_state(&mut self) -> DocumentResultV1<Vec<u8>> {
        let state = self.bytes()?;
        if state.len() > FRONTEND_PARSER_STATE_BYTES_MAX_V1 {
            return fail(format!(
                "parser state of {} bytes exceeds {FRONTEND_PARSER_STATE_BYTES_MAX_V1}",
                state.len()
            ));
        }
        Ok(state.to_vec())
    }
    fn text(&mut self) -> DocumentResultV1<&'a str> {
        std::str::from_utf8(self.bytes()?)
            .map_err(|_| FrontendDocumentErrorV1("text is not UTF-8".into()))
    }
    fn owned_text(&mut self) -> DocumentResultV1<String> {
        self.text().map(str::to_owned)
    }
    fn optional_text(&mut self) -> DocumentResultV1<Option<String>> {
        if self.flag()? {
            self.owned_text().map(Some)
        } else {
            Ok(None)
        }
    }
    fn texts(&mut self) -> DocumentResultV1<Vec<String>> {
        let count = self.count()?;
        (0..count).map(|_| self.owned_text()).collect()
    }
    fn mode(&mut self) -> DocumentResultV1<ModeV1> {
        match self.u8()? {
            0 => Ok(ModeV1::Chat),
            1 => Ok(ModeV1::Thinking),
            other => fail(format!("unknown mode {other}")),
        }
    }
    fn tools(&mut self) -> DocumentResultV1<Vec<ToolV1>> {
        let count = self.count()?;
        let mut tools = Vec::with_capacity(count);
        for _ in 0..count {
            tools.push(ToolV1 {
                name: self.owned_text()?,
                description: self.optional_text()?,
                parameters: self.owned_text()?,
                parameters_present: self.flag()?,
                strict: if self.flag()? {
                    Some(self.flag()?)
                } else {
                    None
                },
            });
        }
        Ok(tools)
    }
    fn range(&mut self) -> DocumentResultV1<Range<u64>> {
        let start = self.u64()?;
        let end = self.u64()?;
        if start > end {
            return fail(format!("range {start}..{end} is inverted"));
        }
        Ok(start..end)
    }
    fn calls(&mut self) -> DocumentResultV1<Vec<CallV1<'a>>> {
        let count = self.count()?;
        let mut calls = Vec::with_capacity(count);
        for _ in 0..count {
            calls.push(CallV1 {
                name: self.text()?,
                arguments: self.text()?,
            });
        }
        Ok(calls)
    }
    fn finish(self) -> DocumentResultV1<()> {
        if self.bytes.is_empty() {
            Ok(())
        } else {
            fail("trailing bytes")
        }
    }
}

fn encode_with(magic: &[u8; 8], body: impl FnOnce(&mut WriterV1<'_>)) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut writer = WriterV1(&mut bytes);
    writer.start(magic);
    body(&mut writer);
    bytes
}

impl OpenedV1 {
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        encode_with(OPENED_MAGIC, |writer| {
            writer.flag(self.reads_tool_objects);
            writer.flag(self.reasoning_content_fallback);
            writer.flag(self.tool_call_reasoning_fallback);
            writer.texts(&self.extra_roles);
            writer.flag(self.tool_references);
        })
    }

    pub fn decode(bytes: &[u8]) -> DocumentResultV1<Self> {
        let mut reader = ReaderV1::open(bytes, OPENED_MAGIC, "open answer")?;
        let opened = Self {
            reads_tool_objects: reader.flag()?,
            reasoning_content_fallback: reader.flag()?,
            tool_call_reasoning_fallback: reader.flag()?,
            extra_roles: reader.texts()?,
            tool_references: reader.flag()?,
        };
        reader.finish()?;
        Ok(opened)
    }
}

impl RenderRequestV1 {
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        encode_with(RENDER_REQUEST_MAGIC, |writer| {
            writer.count(self.messages.len());
            for message in &self.messages {
                writer.text(&message.role);
                writer.optional_text(message.content.as_deref());
                writer.flag(message.parts.is_some());
                if let Some(parts) = &message.parts {
                    writer.count(parts.len());
                    for part in parts {
                        match part {
                            PartV1::Text(text) => {
                                writer.u8(0);
                                writer.text(text);
                            }
                            PartV1::Media(media) => {
                                writer.u8(1);
                                writer.u32(media.index);
                                writer.u8(match media.kind {
                                    MediaKindV1::Image => 0,
                                    MediaKindV1::Audio => 1,
                                    MediaKindV1::Video => 2,
                                });
                                writer.text(&media.part_type);
                            }
                            PartV1::ToolReference(name) => {
                                writer.u8(2);
                                writer.text(name);
                            }
                        }
                    }
                }
                writer.optional_text(message.reasoning.as_deref());
                writer.optional_text(message.reasoning_content.as_deref());
                writer.count(message.tool_calls.len());
                for call in &message.tool_calls {
                    writer.text(&call.id);
                    writer.text(&call.name);
                    writer.text(&call.arguments);
                }
                writer.optional_text(message.tool_call_id.as_deref());
                writer.texts(&message.tools);
            }
            writer.flag(self.continue_final_message);
            writer.flag(self.add_generation_prompt);
            writer.tools(&self.tools);
            writer.texts(&self.tool_objects);
            writer.optional_text(self.response_format.as_deref());
            writer.optional_text(self.reasoning_effort.as_deref());
            writer.optional_text(self.chat_template_kwargs.as_deref());
        })
    }

    /// Decodes a render request. Media parts must carry the request indices
    /// 0, 1, 2 and on, in order.
    pub fn decode(bytes: &[u8]) -> DocumentResultV1<Self> {
        let mut reader = ReaderV1::open(bytes, RENDER_REQUEST_MAGIC, "render request")?;
        let count = reader.count()?;
        let mut messages = Vec::with_capacity(count);
        let mut media = 0_u32;
        for _ in 0..count {
            let role = reader.owned_text()?;
            let content = reader.optional_text()?;
            let parts = if reader.flag()? {
                let count = reader.count()?;
                let mut parts = Vec::with_capacity(count);
                for _ in 0..count {
                    parts.push(match reader.u8()? {
                        0 => PartV1::Text(reader.owned_text()?),
                        1 => {
                            let index = reader.u32()?;
                            if index != media {
                                return fail(format!(
                                    "media part {index} is not request media part {media}"
                                ));
                            }
                            media += 1;
                            let kind = match reader.u8()? {
                                0 => MediaKindV1::Image,
                                1 => MediaKindV1::Audio,
                                2 => MediaKindV1::Video,
                                other => return fail(format!("unknown media kind {other}")),
                            };
                            PartV1::Media(MediaPartV1 {
                                index,
                                kind,
                                part_type: reader.owned_text()?,
                            })
                        }
                        2 => PartV1::ToolReference(reader.owned_text()?),
                        other => return fail(format!("unknown content part tag {other}")),
                    });
                }
                Some(parts)
            } else {
                None
            };
            let reasoning = reader.optional_text()?;
            let reasoning_content = reader.optional_text()?;
            let count = reader.count()?;
            let mut tool_calls = Vec::with_capacity(count);
            for _ in 0..count {
                tool_calls.push(ToolCallV1 {
                    id: reader.owned_text()?,
                    name: reader.owned_text()?,
                    arguments: reader.owned_text()?,
                });
            }
            messages.push(MessageV1 {
                role,
                content,
                parts,
                reasoning,
                reasoning_content,
                tool_calls,
                tool_call_id: reader.optional_text()?,
                tools: reader.texts()?,
            });
        }
        let request = Self {
            messages,
            continue_final_message: reader.flag()?,
            add_generation_prompt: reader.flag()?,
            tools: reader.tools()?,
            tool_objects: reader.texts()?,
            response_format: reader.optional_text()?,
            reasoning_effort: reader.optional_text()?,
            chat_template_kwargs: reader.optional_text()?,
        };
        reader.finish()?;
        Ok(request)
    }

    /// How many media parts the request carries.
    #[must_use]
    pub fn media_count(&self) -> usize {
        self.messages
            .iter()
            .flat_map(|message| message.parts.iter().flatten())
            .filter(|part| matches!(part, PartV1::Media(_)))
            .count()
    }
}

impl RenderedV1 {
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        encode_with(RENDERED_MAGIC, |writer| {
            writer.count(self.segments.len());
            for segment in &self.segments {
                match segment {
                    SegmentV1::Text(text) => {
                        writer.u8(0);
                        writer.text(text);
                    }
                    SegmentV1::Media(index) => {
                        writer.u8(1);
                        writer.u32(*index);
                    }
                }
            }
            writer.mode(self.mode);
            writer.text(&self.effort);
            writer.flag(self.reasoning_open);
            writer.bytes(&self.parser_state);
        })
    }

    pub fn decode(bytes: &[u8]) -> DocumentResultV1<Self> {
        let mut reader = ReaderV1::open(bytes, RENDERED_MAGIC, "rendered prompt")?;
        let count = reader.count()?;
        let mut segments = Vec::with_capacity(count);
        for _ in 0..count {
            segments.push(match reader.u8()? {
                0 => SegmentV1::Text(reader.owned_text()?),
                1 => SegmentV1::Media(reader.u32()?),
                other => return fail(format!("unknown segment tag {other}")),
            });
        }
        let rendered = Self {
            segments,
            mode: reader.mode()?,
            effort: reader.owned_text()?,
            reasoning_open: reader.flag()?,
            parser_state: reader.parser_state()?,
        };
        reader.finish()?;
        Ok(rendered)
    }
}

impl RefusalV1 {
    #[must_use]
    pub fn new(message: impl Into<String>, parameter: Option<&str>) -> Self {
        Self {
            message: message.into(),
            parameter: parameter.map(str::to_owned),
        }
    }

    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        encode_with(REFUSAL_MAGIC, |writer| {
            writer.text(&self.message);
            writer.optional_text(self.parameter.as_deref());
        })
    }

    pub fn decode(bytes: &[u8]) -> DocumentResultV1<Self> {
        let mut reader = ReaderV1::open(bytes, REFUSAL_MAGIC, "refusal")?;
        let refusal = Self {
            message: reader.owned_text()?,
            parameter: reader.optional_text()?,
        };
        reader.finish()?;
        Ok(refusal)
    }

    /// Reads a refusal an entry returned: a refusal document, or the plain
    /// UTF-8 text the ABI layer refuses a sink of another layout with.
    pub fn read(bytes: &[u8]) -> DocumentResultV1<Self> {
        if bytes.starts_with(REFUSAL_MAGIC) {
            return Self::decode(bytes);
        }
        match std::str::from_utf8(bytes) {
            Ok(text) if !text.is_empty() => Ok(Self::new(text, None)),
            _ => fail("a refusal is neither a refusal document nor UTF-8 text"),
        }
    }
}

impl ConstraintRequestV1 {
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        encode_with(CONSTRAINT_MAGIC, |writer| match self {
            Self::ToolChoice { tools, choice } => {
                writer.u8(0);
                writer.tools(tools);
                match choice {
                    ToolChoiceV1::Auto => writer.u8(0),
                    ToolChoiceV1::None => writer.u8(1),
                    ToolChoiceV1::Required => writer.u8(2),
                    ToolChoiceV1::Named(name) => {
                        writer.u8(3);
                        writer.text(name);
                    }
                }
            }
        })
    }

    pub fn decode(bytes: &[u8]) -> DocumentResultV1<Self> {
        let mut reader = ReaderV1::open(bytes, CONSTRAINT_MAGIC, "constraint request")?;
        let request = match reader.u8()? {
            0 => Self::ToolChoice {
                tools: reader.tools()?,
                choice: match reader.u8()? {
                    0 => ToolChoiceV1::Auto,
                    1 => ToolChoiceV1::None,
                    2 => ToolChoiceV1::Required,
                    3 => ToolChoiceV1::Named(reader.owned_text()?),
                    other => return fail(format!("unknown tool choice {other}")),
                },
            },
            other => return fail(format!("unknown constraint request {other}")),
        };
        reader.finish()?;
        Ok(request)
    }
}

impl ParserOpenV1 {
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        encode_with(PARSER_OPEN_MAGIC, |writer| {
            writer.mode(self.mode);
            writer.flag(self.reasoning_open);
            writer.flag(self.extract_tools);
            writer.flag(self.stream_tool_calls);
            writer.flag(self.raw);
            writer.flag(self.ignore_eos);
            writer.flag(self.engine_tokens);
            writer.tools(&self.tool_schemas);
            writer.bytes(&self.parser_state);
        })
    }

    pub fn decode(bytes: &[u8]) -> DocumentResultV1<Self> {
        let mut reader = ReaderV1::open(bytes, PARSER_OPEN_MAGIC, "parser open")?;
        let open = Self {
            mode: reader.mode()?,
            reasoning_open: reader.flag()?,
            extract_tools: reader.flag()?,
            stream_tool_calls: reader.flag()?,
            raw: reader.flag()?,
            ignore_eos: reader.flag()?,
            engine_tokens: reader.flag()?,
            tool_schemas: reader.tools()?,
            parser_state: reader.parser_state()?,
        };
        reader.finish()?;
        Ok(open)
    }
}

impl<'a> PushV1<'a> {
    /// Writes this push into `bytes`, replacing what it held.
    pub fn encode_into(&self, bytes: &mut Vec<u8>) {
        bytes.clear();
        let mut writer = WriterV1(bytes);
        writer.start(PUSH_MAGIC);
        writer.u64(self.keep);
        writer.flag(self.held);
        writer.flag(self.stop_floor.is_some());
        writer.u64(self.stop_floor.unwrap_or(0));
        writer.bytes(self.suffix);
    }

    pub fn decode(bytes: &'a [u8]) -> DocumentResultV1<Self> {
        let mut reader = ReaderV1::open(bytes, PUSH_MAGIC, "parser push")?;
        let keep = reader.u64()?;
        let held = reader.flag()?;
        let floored = reader.flag()?;
        let floor = reader.u64()?;
        if !floored && floor != 0 {
            return fail("an absent stop floor is not zero");
        }
        let push = Self {
            keep,
            held,
            stop_floor: floored.then_some(floor),
            suffix: reader.bytes()?,
        };
        reader.finish()?;
        Ok(push)
    }
}

impl<'a> EventsV1<'a> {
    /// Writes these events into `bytes`, replacing what it held.
    pub fn encode_into(&self, bytes: &mut Vec<u8>) {
        bytes.clear();
        let mut writer = WriterV1(bytes);
        writer.start(EVENTS_MAGIC);
        writer.u64(self.cursor);
        writer.count(self.events.len());
        for event in &self.events {
            match event {
                EventV1::Reasoning(range) => {
                    writer.u8(0);
                    writer.range(range);
                }
                EventV1::Content(range) => {
                    writer.u8(1);
                    writer.range(range);
                }
                EventV1::ToolCalls { range, calls } => {
                    writer.u8(2);
                    writer.range(range);
                    writer.calls(calls);
                }
                EventV1::ToolCallDeltas(deltas) => {
                    writer.u8(3);
                    writer.count(deltas.len());
                    for delta in deltas {
                        writer.u32(delta.index);
                        writer.optional_text(delta.name);
                        writer.text(delta.arguments);
                    }
                }
                EventV1::Finish { reason, completion } => {
                    writer.u8(4);
                    writer.u8(match reason {
                        FinishReasonV1::Stop => 0,
                        FinishReasonV1::Length => 1,
                        FinishReasonV1::ToolCalls => 2,
                    });
                    writer.flag(completion.is_some());
                    if let Some(completion) = completion {
                        writer.text(completion.content);
                        writer.optional_text(completion.reasoning_content);
                        writer.calls(&completion.tool_calls);
                    }
                }
            }
        }
    }

    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.encode_into(&mut bytes);
        bytes
    }

    pub fn decode(bytes: &'a [u8]) -> DocumentResultV1<Self> {
        let mut reader = ReaderV1::open(bytes, EVENTS_MAGIC, "parser events")?;
        let cursor = reader.u64()?;
        let count = reader.count()?;
        let mut events = Vec::with_capacity(count);
        for _ in 0..count {
            events.push(match reader.u8()? {
                0 => EventV1::Reasoning(reader.range()?),
                1 => EventV1::Content(reader.range()?),
                2 => EventV1::ToolCalls {
                    range: reader.range()?,
                    calls: reader.calls()?,
                },
                3 => {
                    let count = reader.count()?;
                    let mut deltas = Vec::with_capacity(count);
                    for _ in 0..count {
                        let index = reader.u32()?;
                        let name = if reader.flag()? {
                            Some(reader.text()?)
                        } else {
                            None
                        };
                        deltas.push(DeltaV1 {
                            index,
                            name,
                            arguments: reader.text()?,
                        });
                    }
                    EventV1::ToolCallDeltas(deltas)
                }
                4 => {
                    let reason = match reader.u8()? {
                        0 => FinishReasonV1::Stop,
                        1 => FinishReasonV1::Length,
                        2 => FinishReasonV1::ToolCalls,
                        other => return fail(format!("unknown finish reason {other}")),
                    };
                    let completion = if reader.flag()? {
                        Some(CompletionV1 {
                            content: reader.text()?,
                            reasoning_content: if reader.flag()? {
                                Some(reader.text()?)
                            } else {
                                None
                            },
                            tool_calls: reader.calls()?,
                        })
                    } else {
                        None
                    };
                    EventV1::Finish { reason, completion }
                }
                other => return fail(format!("unknown event tag {other}")),
            });
        }
        reader.finish()?;
        Ok(Self { events, cursor })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool() -> ToolV1 {
        ToolV1 {
            name: "lookup".into(),
            description: Some("Find a city".into()),
            parameters: r#"{"type":"object","properties":{"city":{"type":"string"}}}"#.into(),
            parameters_present: true,
            strict: Some(false),
        }
    }

    fn render_request() -> RenderRequestV1 {
        RenderRequestV1 {
            messages: vec![
                MessageV1 {
                    role: "user".into(),
                    content: Some("look".into()),
                    parts: Some(vec![
                        PartV1::Text("look".into()),
                        PartV1::Media(MediaPartV1 {
                            index: 0,
                            kind: MediaKindV1::Image,
                            part_type: "image_url".into(),
                        }),
                        PartV1::Media(MediaPartV1 {
                            index: 1,
                            kind: MediaKindV1::Audio,
                            part_type: "input_audio".into(),
                        }),
                    ]),
                    reasoning: None,
                    reasoning_content: None,
                    tool_calls: vec![],
                    tool_call_id: None,
                    tools: vec![],
                },
                MessageV1 {
                    role: "assistant".into(),
                    content: None,
                    parts: None,
                    reasoning: Some("r".into()),
                    reasoning_content: Some("rc".into()),
                    tool_calls: vec![ToolCallV1 {
                        id: "call-1".into(),
                        name: "lookup".into(),
                        arguments: r#"{"city":"Paris"}"#.into(),
                    }],
                    tool_call_id: None,
                    tools: vec![r#"{"type":"function"}"#.into()],
                },
                MessageV1 {
                    role: "tool".into(),
                    content: Some("sunny".into()),
                    parts: None,
                    reasoning: None,
                    reasoning_content: None,
                    tool_calls: vec![],
                    tool_call_id: Some("call-1".into()),
                    tools: vec![],
                },
                MessageV1 {
                    role: "tool".into(),
                    content: Some(String::new()),
                    parts: Some(vec![
                        PartV1::ToolReference("lookup".into()),
                        PartV1::ToolReference(String::new()),
                    ]),
                    reasoning: None,
                    reasoning_content: None,
                    tool_calls: vec![],
                    tool_call_id: Some("call-2".into()),
                    tools: vec![],
                },
            ],
            continue_final_message: false,
            add_generation_prompt: true,
            tools: vec![
                tool(),
                ToolV1 {
                    description: None,
                    strict: None,
                    parameters_present: false,
                    ..tool()
                },
            ],
            tool_objects: vec![r#"{"name":"lookup"}"#.into()],
            response_format: Some(r#"{"type":"text"}"#.into()),
            reasoning_effort: Some("high".into()),
            chat_template_kwargs: Some(r#"{"enable_thinking":true}"#.into()),
        }
    }

    fn opened() -> OpenedV1 {
        OpenedV1 {
            reads_tool_objects: true,
            reasoning_content_fallback: false,
            tool_call_reasoning_fallback: true,
            extra_roles: vec!["observation".into(), "ipython".into()],
            tool_references: true,
        }
    }

    fn rendered() -> RenderedV1 {
        RenderedV1 {
            segments: vec![
                SegmentV1::Text("<|user|>look".into()),
                SegmentV1::Media(0),
                SegmentV1::Text(String::new()),
            ],
            mode: ModeV1::Thinking,
            effort: "high".into(),
            reasoning_open: true,
            parser_state: vec![1, 2, 3],
        }
    }

    fn parser_open() -> ParserOpenV1 {
        ParserOpenV1 {
            mode: ModeV1::Chat,
            reasoning_open: false,
            extract_tools: true,
            stream_tool_calls: true,
            raw: false,
            ignore_eos: true,
            engine_tokens: true,
            tool_schemas: vec![tool()],
            parser_state: vec![],
        }
    }

    fn events() -> EventsV1<'static> {
        EventsV1 {
            events: vec![
                EventV1::Reasoning(0..4),
                EventV1::Content(12..14),
                EventV1::ToolCallDeltas(vec![
                    DeltaV1 {
                        index: 0,
                        name: Some("lookup"),
                        arguments: "{\"city\": \"Pa",
                    },
                    DeltaV1 {
                        index: 0,
                        name: None,
                        arguments: "ris\"}",
                    },
                ]),
                EventV1::ToolCalls {
                    range: 14..60,
                    calls: vec![CallV1 {
                        name: "lookup",
                        arguments: "{\"city\": \"Paris\"}",
                    }],
                },
                EventV1::Finish {
                    reason: FinishReasonV1::ToolCalls,
                    completion: Some(CompletionV1 {
                        content: "ok",
                        reasoning_content: Some("plan"),
                        tool_calls: vec![CallV1 {
                            name: "lookup",
                            arguments: "{\"city\": \"Paris\"}",
                        }],
                    }),
                },
                EventV1::Finish {
                    reason: FinishReasonV1::Length,
                    completion: None,
                },
            ],
            cursor: 60,
        }
    }

    fn constraints() -> [ConstraintRequestV1; 4] {
        [
            ConstraintRequestV1::ToolChoice {
                tools: vec![tool()],
                choice: ToolChoiceV1::Auto,
            },
            ConstraintRequestV1::ToolChoice {
                tools: vec![],
                choice: ToolChoiceV1::None,
            },
            ConstraintRequestV1::ToolChoice {
                tools: vec![tool()],
                choice: ToolChoiceV1::Required,
            },
            ConstraintRequestV1::ToolChoice {
                tools: vec![tool()],
                choice: ToolChoiceV1::Named("lookup".into()),
            },
        ]
    }

    /// Whether bytes decode as one document kind.
    type DecodesV1 = fn(&[u8]) -> bool;

    /// Every encoded document, with whether bytes decode as its kind.
    fn documents() -> Vec<(Vec<u8>, DecodesV1)> {
        let mut documents: Vec<(Vec<u8>, DecodesV1)> = vec![
            (opened().encode(), |bytes| OpenedV1::decode(bytes).is_ok()),
            (render_request().encode(), |bytes| {
                RenderRequestV1::decode(bytes).is_ok()
            }),
            (rendered().encode(), |bytes| {
                RenderedV1::decode(bytes).is_ok()
            }),
            (RefusalV1::new("no", Some("tools")).encode(), |bytes| {
                RefusalV1::decode(bytes).is_ok()
            }),
            (parser_open().encode(), |bytes| {
                ParserOpenV1::decode(bytes).is_ok()
            }),
            (events().encode(), |bytes| EventsV1::decode(bytes).is_ok()),
        ];
        for constraint in constraints() {
            documents.push((constraint.encode(), |bytes| {
                ConstraintRequestV1::decode(bytes).is_ok()
            }));
        }
        let mut push = Vec::new();
        PushV1 {
            keep: 3,
            held: true,
            stop_floor: Some(2),
            suffix: b"abc",
        }
        .encode_into(&mut push);
        documents.push((push, |bytes| PushV1::decode(bytes).is_ok()));
        documents
    }

    #[test]
    fn documents_round_trip() {
        for opened in [opened(), OpenedV1::default()] {
            assert_eq!(OpenedV1::decode(&opened.encode()), Ok(opened.clone()));
        }
        let request = render_request();
        assert_eq!(RenderRequestV1::decode(&request.encode()), Ok(request));
        assert_eq!(render_request().media_count(), 2);
        let rendered = rendered();
        assert_eq!(RenderedV1::decode(&rendered.encode()), Ok(rendered));
        for refusal in [
            RefusalV1::new("bad effort", Some("reasoning_effort")),
            RefusalV1::new("", None),
        ] {
            assert_eq!(RefusalV1::decode(&refusal.encode()), Ok(refusal.clone()));
            assert_eq!(RefusalV1::read(&refusal.encode()), Ok(refusal));
        }
        for constraint in constraints() {
            assert_eq!(
                ConstraintRequestV1::decode(&constraint.encode()),
                Ok(constraint)
            );
        }
        let open = parser_open();
        assert_eq!(ParserOpenV1::decode(&open.encode()), Ok(open));
        for push in [
            PushV1 {
                keep: 0,
                held: false,
                stop_floor: None,
                suffix: b"",
            },
            PushV1 {
                keep: 9,
                held: true,
                stop_floor: Some(0),
                suffix: b"\xff\xfe",
            },
        ] {
            let mut bytes = vec![7; 3];
            push.encode_into(&mut bytes);
            assert_eq!(PushV1::decode(&bytes), Ok(push));
        }
        let events = events();
        let bytes = events.encode();
        assert_eq!(EventsV1::decode(&bytes), Ok(events));
    }

    #[test]
    fn a_plain_text_refusal_reads_as_a_message() {
        let text = b"frontend sink layout 2; this build reads layout 1";
        assert_eq!(
            RefusalV1::read(text),
            Ok(RefusalV1::new(
                "frontend sink layout 2; this build reads layout 1",
                None
            ))
        );
        assert!(RefusalV1::read(b"").is_err());
        assert!(RefusalV1::read(b"\xff").is_err());
        // A refusal document that does not decode is not read as text.
        let mut broken = RefusalV1::new("no", None).encode();
        broken.push(0);
        assert!(RefusalV1::read(&broken).is_err());
    }

    #[test]
    fn decoding_refuses_truncated_trailing_and_foreign_documents() {
        let documents = documents();
        for (bytes, decodes) in &documents {
            assert!(decodes(bytes));
            for cut in 0..bytes.len() {
                assert!(!decodes(&bytes[..cut]), "cut at {cut}");
            }
            let mut trailing = bytes.clone();
            trailing.push(0);
            assert!(!decodes(&trailing));
            let mut layout = bytes.clone();
            layout[8] = 2;
            assert!(!decodes(&layout));
        }
        // No document decodes as another kind.
        for (index, (bytes, _)) in documents.iter().enumerate() {
            for (other, (_, decodes)) in documents.iter().enumerate() {
                if bytes[..8] != documents[other].0[..8] {
                    assert!(!decodes(bytes), "{index} as {other}");
                }
            }
        }
    }

    #[test]
    fn decoding_refuses_bad_flags_tags_counts_and_text() {
        // A flag that is neither 0 nor 1: the push's `held`.
        let mut push = Vec::new();
        PushV1 {
            keep: 0,
            held: false,
            stop_floor: None,
            suffix: b"",
        }
        .encode_into(&mut push);
        let mut bad = push.clone();
        bad[20] = 2;
        assert!(PushV1::decode(&bad).is_err());
        // An absent stop floor with a value.
        let mut bad = push.clone();
        bad[22] = 1;
        assert!(PushV1::decode(&bad).is_err());
        // An unknown event tag, an inverted range and an unknown reason.
        let header = |count: u32| {
            let mut bytes = EVENTS_MAGIC.to_vec();
            bytes.extend_from_slice(&FRONTEND_DOCUMENT_LAYOUT_V1.to_le_bytes());
            bytes.extend_from_slice(&0_u64.to_le_bytes());
            bytes.extend_from_slice(&count.to_le_bytes());
            bytes
        };
        let mut unknown = header(1);
        unknown.push(5);
        assert!(EventsV1::decode(&unknown).is_err());
        let mut inverted = header(1);
        inverted.push(1);
        inverted.extend_from_slice(&4_u64.to_le_bytes());
        inverted.extend_from_slice(&3_u64.to_le_bytes());
        assert!(EventsV1::decode(&inverted).is_err());
        let mut reason = header(1);
        reason.extend_from_slice(&[4, 3, 0]);
        assert!(EventsV1::decode(&reason).is_err());
        // A count past the bound or past the bytes left.
        let huge = header(u32::MAX);
        assert!(EventsV1::decode(&huge).is_err());
        let mut long = header(4);
        long.extend_from_slice(&[0; 3]);
        assert!(EventsV1::decode(&long).is_err());
        // Text that is not UTF-8.
        let mut text = header(1);
        text.extend_from_slice(&[3]);
        text.extend_from_slice(&1_u32.to_le_bytes());
        text.extend_from_slice(&0_u32.to_le_bytes());
        text.push(0);
        text.extend_from_slice(&1_u32.to_le_bytes());
        text.push(0xff);
        assert!(EventsV1::decode(&text).is_err());
        text.pop();
        text.push(b'x');
        assert!(EventsV1::decode(&text).is_ok());
        // A constraint request asks only for a tool choice's grammar; the
        // host checks tool-call arguments itself.
        let mut arguments = CONSTRAINT_MAGIC.to_vec();
        arguments.extend_from_slice(&FRONTEND_DOCUMENT_LAYOUT_V1.to_le_bytes());
        arguments.push(1);
        arguments.extend_from_slice(&2_u32.to_le_bytes());
        arguments.extend_from_slice(b"{}");
        assert!(ConstraintRequestV1::decode(&arguments).is_err());
    }

    /// A tool reference crosses as content part tag 2 and its name; tag 3 is
    /// unknown.
    #[test]
    fn a_tool_reference_part_is_tag_two_and_its_name() {
        let request = |parts| RenderRequestV1 {
            messages: vec![MessageV1 {
                role: "tool".into(),
                content: None,
                parts: Some(parts),
                reasoning: None,
                reasoning_content: None,
                tool_calls: vec![],
                tool_call_id: None,
                tools: vec![],
            }],
            continue_final_message: false,
            add_generation_prompt: false,
            tools: vec![],
            tool_objects: vec![],
            response_format: None,
            reasoning_effort: None,
            chat_template_kwargs: None,
        };
        let text = request(vec![PartV1::Text("lookup".into())]).encode();
        let reference = request(vec![PartV1::ToolReference("lookup".into())]).encode();
        let tag = text
            .iter()
            .zip(&reference)
            .position(|(text, reference)| text != reference)
            .unwrap();
        assert_eq!((text[tag], reference[tag]), (0, 2));
        assert_eq!(text[tag + 1..], reference[tag + 1..]);
        assert_eq!(
            RenderRequestV1::decode(&reference),
            Ok(request(vec![PartV1::ToolReference("lookup".into())]))
        );
        assert_eq!(
            request(vec![PartV1::ToolReference("lookup".into())]).media_count(),
            0
        );
        let mut unknown = reference;
        unknown[tag] = 3;
        assert_eq!(
            RenderRequestV1::decode(&unknown),
            Err(FrontendDocumentErrorV1("unknown content part tag 3".into()))
        );
    }

    #[test]
    fn parser_state_past_its_limit_is_refused() {
        let mut rendered = rendered();
        rendered.parser_state = vec![7; FRONTEND_PARSER_STATE_BYTES_MAX_V1];
        assert_eq!(RenderedV1::decode(&rendered.encode()), Ok(rendered.clone()));
        let mut open = parser_open();
        open.parser_state = rendered.parser_state.clone();
        assert_eq!(ParserOpenV1::decode(&open.encode()), Ok(open.clone()));
        rendered.parser_state.push(7);
        open.parser_state.push(7);
        let refused = format!(
            "parser state of {} bytes exceeds {FRONTEND_PARSER_STATE_BYTES_MAX_V1}",
            FRONTEND_PARSER_STATE_BYTES_MAX_V1 + 1
        );
        assert!(
            RenderedV1::decode(&rendered.encode()).is_err_and(|error| error.0.contains(&refused))
        );
        assert!(
            ParserOpenV1::decode(&open.encode()).is_err_and(|error| error.0.contains(&refused))
        );
    }

    #[test]
    fn media_parts_carry_the_request_indices_in_order() {
        let mut request = render_request();
        let Some(parts) = &mut request.messages[0].parts else {
            unreachable!()
        };
        parts.swap(1, 2);
        assert!(RenderRequestV1::decode(&request.encode()).is_err());
        let mut request = render_request();
        let Some(parts) = &mut request.messages[0].parts else {
            unreachable!()
        };
        parts.remove(1);
        assert!(RenderRequestV1::decode(&request.encode()).is_err());
    }
}
