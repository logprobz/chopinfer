//! The seam between the host layer and one model's format.
//!
//! The host layer owns what is the same for every model: tokens, ordinals,
//! stop IDs, client stops, complete UTF-8, limits and token contributions. A
//! format owns what a model's template decides: which bytes of a completion
//! are reasoning, answer text or tool calls. The two meet only through the
//! plain data here, which is what will cross the bundle ABI: a parser reads
//! bytes the host already released and returns events over byte ranges of the
//! completion, never tokens.

use std::fmt;
use std::ops::Range;

use runtime_server::{
    FrontendErrorV1, FrontendJsonV1, FrontendMediaPartV1, FrontendMessageV1, FrontendToolChoiceV1,
    FrontendToolV1, PromptSegmentV1,
};

use crate::decode::{AssistantCompletionV1, CompletionFinishV1, ParsedToolCallV1, PromptModeV1};

/// What one completion parser is fixed to when it opens.
///
/// Each field is a request property the host decided; none names a model.
#[derive(Clone, Debug)]
pub(crate) struct FormatParserOptionsV1 {
    /// The mode the prompt rendered in.
    pub(crate) mode: PromptModeV1,
    /// Whether the prompt leaves reasoning open, so the completion begins in
    /// reasoning.
    pub(crate) reasoning_open: bool,
    /// Whether tool calls are read out of the completion. A request that
    /// offers no function still parses tool-shaped output, but delivers no
    /// call.
    pub(crate) extract_tools: bool,
    /// Whether tool-call arguments stream as deltas while they grow.
    pub(crate) stream_tool_calls: bool,
    /// Whether the completion is raw text with no structure to read.
    pub(crate) raw: bool,
    /// Whether end markers are ordinary text, as `ignore_eos` asks.
    pub(crate) ignore_eos: bool,
    /// Whether the engine sampled the tokens, so a stop it confirmed arrives
    /// by ID and the ordinary spelling of an ID-backed end marker is text.
    pub(crate) engine_tokens: bool,
    /// The request's tool schemas, for parameters typed by schema.
    pub(crate) tool_schemas: Vec<FrontendToolV1>,
}

/// The completion bytes the host released to the parser.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FormatInputV1<'a> {
    /// The completion's raw bytes from its start, through the last byte the
    /// host released. Released bytes are stable: they do not end in an
    /// incomplete character the next token could still complete, and a
    /// possible client stop is withheld. They are not necessarily valid
    /// UTF-8: an invalid sequence that later bytes follow is released as it
    /// was generated. Each push extends the bytes of the one before, except
    /// that a client stop may cut them short before the last push.
    pub(crate) completion: &'a [u8],
    /// Whether the host holds bytes after `completion`, so a completion end
    /// found here is followed by more and is malformed.
    pub(crate) held: bool,
    /// Whether generation ended, so no byte follows `completion`.
    pub(crate) last: bool,
    /// `None` until the completion passes the request's minimum token
    /// count; then the byte offset an end marker must end after to stop it.
    pub(crate) stop_floor: Option<usize>,
}

/// One tool-call delta, by the call's index among the delivered calls.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FormatToolCallDeltaV1 {
    pub(crate) index: usize,
    pub(crate) name: Option<String>,
    pub(crate) arguments: String,
}

/// One thing a parser settled about the completion.
///
/// Every range is a pair of byte offsets into the raw completion bytes, not
/// into decoded text. The host decodes text from those bytes, replacing any
/// invalid sequence, and attributes them to the tokens that produced them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum FormatEventV1 {
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

/// Reads one completion's structure from the bytes the host released.
pub(crate) trait FormatParserV1: Send {
    /// Reads the bytes `input` adds, and returns what they settled.
    fn push(&mut self, input: FormatInputV1<'_>) -> Result<Vec<FormatEventV1>, String>;

    /// Where the bytes the parser has not settled begin. The host matches
    /// client stops from here.
    fn cursor(&self) -> usize;
}

/// One chat request as a format renders it, after the host's checks.
///
/// The reasoning controls and template keywords arrive as the client sent
/// them; what they mean is the format's decision.
#[derive(Clone, Debug)]
pub(crate) struct FormatRenderRequestV1<'a> {
    pub(crate) messages: Vec<FrontendMessageV1>,
    pub(crate) continue_final_message: bool,
    pub(crate) add_generation_prompt: bool,
    /// Every offered tool, whatever the tool choice.
    pub(crate) tools: &'a [FrontendToolV1],
    /// Each offered tool's function object as the client sent it, when the
    /// transport reads them; otherwise empty.
    pub(crate) tool_objects: Vec<FrontendJsonV1>,
    pub(crate) response_format: Option<FrontendJsonV1>,
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) enable_thinking: Option<bool>,
    pub(crate) thinking: Option<bool>,
    pub(crate) nested_reasoning_effort: Option<String>,
    pub(crate) preserve_thinking: Option<bool>,
    pub(crate) clear_thinking: Option<FrontendJsonV1>,
    pub(crate) add_vision_id: Option<FrontendJsonV1>,
    /// The client's whole `chat_template_kwargs` object, when the transport
    /// carried one; the typed keywords above are read from it too.
    pub(crate) chat_template_kwargs: Option<FrontendJsonV1>,
}

/// One rendered prompt, and what a parser of its completion needs.
#[derive(Clone, Debug)]
pub(crate) struct FormatRenderedV1 {
    pub(crate) prompt: String,
    pub(crate) segments: Vec<PromptSegmentV1>,
    /// The request's media parts, in the order the segments place them.
    pub(crate) media: Vec<FrontendMediaPartV1>,
    pub(crate) mode: PromptModeV1,
    /// The selected reasoning effort's declared name.
    pub(crate) effort: String,
    /// Whether the prompt leaves reasoning open, so the completion begins
    /// in reasoning and a constraint waits for a reasoning end marker.
    pub(crate) reasoning_open: bool,
    /// Opaque state the format reads back when a parser of this prompt's
    /// completion opens.
    pub(crate) parser_state: Vec<u8>,
}

/// Renders chat prompts and tool constraints in one model's format.
pub(crate) trait FormatRendererV1: fmt::Debug + Send + Sync {
    fn render(
        &self,
        request: FormatRenderRequestV1<'_>,
    ) -> Result<FormatRenderedV1, FrontendErrorV1>;

    /// The grammar `choice` imposes on calls to `tools`, or `None` when it
    /// imposes nothing.
    fn tool_constraint(
        &self,
        tools: &[FrontendToolV1],
        choice: &FrontendToolChoiceV1,
    ) -> Result<Option<runtime_grammar::Grammar>, FrontendErrorV1>;

    /// What the format reads of a request besides the fields every request
    /// document carries, which the transport then carries for it.
    fn opened(&self) -> runtime_bundle_abi::frontend_document::OpenedV1;
}
