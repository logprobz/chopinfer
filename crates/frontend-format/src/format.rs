//! What a chat format and the shared parser read and settle, in plain data.
//!
//! A host releases completion bytes to a parser and reads back events over
//! byte ranges of the completion, never tokens. A test that links this
//! library calls a [`crate::ChatFormatV1`] and [`crate::DeclaredParserV1`]
//! with these types directly; a host that loads a package reaches the same
//! code through the `chop.frontend.v1` documents, which carry these
//! fields.

use std::ops::Range;

use crate::chat::PromptModeV1;
use crate::request::{
    FrontendJsonV1, FrontendMediaPartV1, FrontendMessageV1, FrontendToolV1, PromptSegmentV1,
};

/// Why one generation ended.
pub use runtime_bundle_abi::frontend_document::FinishReasonV1 as CompletionFinishV1;

/// One tool call read out of a completion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsedToolCallV1 {
    pub name: String,
    /// The arguments, in the JSON encoding the declaration named.
    pub arguments: String,
}

/// One whole assistant answer, after the completion terminated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssistantCompletionV1 {
    pub content: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ParsedToolCallV1>,
}

/// What one completion parser is fixed to when it opens.
///
/// Each field is a request property the host decided; none names a model.
#[derive(Clone, Debug)]
pub struct FormatParserOptionsV1 {
    /// The mode the prompt rendered in.
    pub mode: PromptModeV1,
    /// Whether the prompt leaves reasoning open, so the completion begins in
    /// reasoning.
    pub reasoning_open: bool,
    /// Whether tool calls are read out of the completion. A request that
    /// offers no function still parses tool-shaped output, but delivers no
    /// call.
    pub extract_tools: bool,
    /// Whether tool-call arguments stream as deltas while they grow.
    pub stream_tool_calls: bool,
    /// Whether the completion is raw text with no structure to read.
    pub raw: bool,
    /// Whether end markers are ordinary text, as `ignore_eos` asks.
    pub ignore_eos: bool,
    /// Whether the engine sampled the tokens, so a stop it confirmed arrives
    /// by ID and the ordinary spelling of an ID-backed end marker is text.
    pub engine_tokens: bool,
    /// The request's tool schemas, for parameters typed by schema.
    pub tool_schemas: Vec<FrontendToolV1>,
}

/// The completion bytes the host released to the parser.
#[derive(Clone, Copy, Debug)]
pub struct FormatInputV1<'a> {
    /// The completion's raw bytes from its start, through the last byte the
    /// host released. Released bytes are stable: they do not end in an
    /// incomplete character the next token could still complete, and a
    /// possible client stop is withheld. They are not necessarily valid
    /// UTF-8: an invalid sequence that later bytes follow is released as it
    /// was generated. Each push extends the bytes of the one before, except
    /// that a client stop may cut them short before the last push.
    pub completion: &'a [u8],
    /// Whether the host holds bytes after `completion`, so a completion end
    /// found here is followed by more and is malformed.
    pub held: bool,
    /// Whether generation ended, so no byte follows `completion`.
    pub last: bool,
    /// `None` until the completion passes the request's minimum token
    /// count; then the byte offset an end marker must end after to stop it.
    pub stop_floor: Option<usize>,
}

/// One tool-call delta, by the call's index among the delivered calls.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FormatToolCallDeltaV1 {
    pub index: usize,
    pub name: Option<String>,
    pub arguments: String,
}

/// One thing a parser settled about the completion.
///
/// Every range is a pair of byte offsets into the raw completion bytes, not
/// into decoded text. The host decodes text from those bytes, replacing any
/// invalid sequence, and attributes them to the tokens that produced them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FormatEventV1 {
    /// Reasoning text, as these completion bytes.
    Reasoning(Range<usize>),
    /// Answer text, as these completion bytes.
    Content(Range<usize>),
    /// A group of tool calls, read from these completion bytes.
    ToolCalls {
        calls: Vec<ParsedToolCallV1>,
        range: Range<usize>,
    },
    ToolCallDeltas(Vec<FormatToolCallDeltaV1>),
    /// The completion ended; the last event of a parser.
    Finish {
        reason: CompletionFinishV1,
        completion: Option<AssistantCompletionV1>,
    },
}

/// One chat request as the format renders it, after the host's checks.
///
/// The reasoning controls and template keywords arrive as the client sent
/// them; what they mean is the format's decision.
#[derive(Clone, Debug)]
pub struct FormatRenderRequestV1<'a> {
    pub messages: Vec<FrontendMessageV1>,
    pub continue_final_message: bool,
    pub add_generation_prompt: bool,
    /// Every offered tool, whatever the tool choice.
    pub tools: &'a [FrontendToolV1],
    /// Each offered tool's function object as the client sent it, when the
    /// transport reads them; otherwise empty.
    pub tool_objects: Vec<FrontendJsonV1>,
    pub response_format: Option<FrontendJsonV1>,
    pub reasoning_effort: Option<String>,
    pub enable_thinking: Option<bool>,
    pub thinking: Option<bool>,
    pub nested_reasoning_effort: Option<String>,
    pub preserve_thinking: Option<bool>,
    pub clear_thinking: Option<FrontendJsonV1>,
    pub add_vision_id: Option<FrontendJsonV1>,
}

/// One rendered prompt, and what a parser of its completion needs.
#[derive(Clone, Debug)]
pub struct FormatRenderedV1 {
    pub prompt: String,
    pub segments: Vec<PromptSegmentV1>,
    /// The request's media parts, in the order the segments place them.
    pub media: Vec<FrontendMediaPartV1>,
    pub mode: PromptModeV1,
    /// The selected reasoning effort's declared name.
    pub effort: String,
    /// Whether the prompt leaves reasoning open, so the completion begins
    /// in reasoning and a constraint waits for a reasoning end marker.
    pub reasoning_open: bool,
    /// Opaque state the format reads back when a parser of this prompt's
    /// completion opens.
    pub parser_state: Vec<u8>,
}
