//! The host half of a bundle's frontend export.
//!
//! [`NativeFormatV1`] renders prompts, asks for tool constraints and opens
//! completion parsers through a `chop.frontend.v1` export, in the documents
//! `runtime_bundle_abi::frontend_document` defines. The format behind the
//! export is the bundle's code, so nothing it answers is trusted: each
//! answer is checked against the request and the installed declaration
//! before the host acts on it, and an answer that fails a check fails the
//! call closed, with the decoder's malformed or sequence refusal for a
//! parser.

use std::fmt;
use std::sync::Arc;

use runtime_bundle_abi::frontend::{
    FrontendCallErrorV1, FrontendExportV1, FrontendHandleV1, FrontendParserHandleV1,
};
use runtime_bundle_abi::frontend_document as document;
use runtime_server::{
    FrontendContentPartV1, FrontendErrorV1, FrontendJsonV1, FrontendMediaPartKindV1,
    FrontendMediaPartV1, FrontendMessageV1, FrontendRoleV1, FrontendToolChoiceV1, FrontendToolV1,
    PromptSegmentV1,
};

use crate::checks::{malformed_v1, resource_v1, sequence_v1, validate_function_name_v1};
use crate::decode::{AssistantCompletionV1, CompletionFinishV1, ParsedToolCallV1, PromptModeV1};
use crate::descriptor::{FrontendComponentErrorV1, FrontendDescriptorV1, FrontendMediaKindV1};
use crate::format::{
    FormatEventV1, FormatInputV1, FormatParserOptionsV1, FormatParserV1, FormatRenderRequestV1,
    FormatRenderedV1, FormatRendererV1, FormatToolCallDeltaV1,
};

/// The refusal families a decoder reports. A parser refusal in one of them
/// passes through as the format worded it; any other is malformed.
const DECODER_FAMILIES_V1: [&str; 5] = [
    "malformed completion: ",
    "invalid conversation sequence: ",
    "conversation resource limit: ",
    "invalid conversation field: ",
    "invalid JSON: ",
];

/// Bytes a render answer may spend past its prompt text and parser state:
/// the document header, mode, effort, flags and lengths.
const RENDER_OVERHEAD_BYTES_V1: usize = 64;

/// Bytes a parser's event answer may hold per released completion byte, per
/// tool call, and in all. One push's answer holds each released byte a few
/// times at most: in a text event, and in a tool call's arguments, escaped
/// up to 6 bytes per byte, in its deltas, the delivered call and the
/// finish's completion, which also repeats the text with an invalid byte
/// replaced by 3. 64 per byte covers that sum. Each call adds its name and
/// fixed fields to each of the three, and the fixed allowance covers the
/// events themselves.
const EVENT_BYTES_PER_RELEASED_BYTE_V1: usize = 64;
const EVENT_BYTES_PER_TOOL_CALL_V1: usize = 64;
const EVENT_OVERHEAD_BYTES_V1: usize = 64 * 1024;

/// Bytes a constraint answer may hold per byte of its request, plus a fixed
/// allowance. The declared tool grammars measured at most 126 wire bytes per
/// request byte (a forced call whose parameters are integers bounded at
/// plus and minus 2^53 - 1: about 64 per byte for the expression, held
/// twice by the call and the calls that may follow it); the factor allows
/// 8 times that, so a constraint the shared tool grammars build within the
/// request limits is never refused here and still fails, when it is too
/// large, where it always has: at compilation.
const CONSTRAINT_BYTES_PER_REQUEST_BYTE_V1: usize = 1024;
const CONSTRAINT_OVERHEAD_BYTES_V1: usize = 1 << 20;

/// A format behind a bundle's frontend export.
pub(crate) struct NativeFormatV1 {
    /// The installed declaration the answers are checked against.
    descriptor: Arc<FrontendDescriptorV1>,
    frontend: FrontendHandleV1,
    /// What the format said it reads of a request when it opened.
    opened: document::OpenedV1,
}

impl fmt::Debug for NativeFormatV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeFormatV1")
            .finish_non_exhaustive()
    }
}

impl NativeFormatV1 {
    /// Opens `export`'s frontend over `component`, the bytes `descriptor`
    /// was read from, or over `descriptor` written back out when it was
    /// assembled in code.
    pub(crate) fn from_export(
        export: &FrontendExportV1,
        descriptor: Arc<FrontendDescriptorV1>,
        component: Option<&[u8]>,
    ) -> Result<Self, FrontendComponentErrorV1> {
        let written;
        let component = match component {
            Some(component) => component,
            None => {
                written = serde_json::to_vec(descriptor.as_ref())
                    .map_err(|error| FrontendComponentErrorV1::Invalid(error.to_string()))?;
                &written
            }
        };
        Self::open(export, component, descriptor).map_err(FrontendComponentErrorV1::Invalid)
    }

    /// Opens `export`'s frontend over `component`, checked against
    /// `descriptor`, the declaration the host read from the same package.
    pub(crate) fn open(
        export: &FrontendExportV1,
        component: &[u8],
        descriptor: Arc<FrontendDescriptorV1>,
    ) -> Result<Self, String> {
        let (frontend, opened) = export.open(component).map_err(|error| match error {
            FrontendCallErrorV1::Refused(bytes) => document::RefusalV1::read(&bytes).map_or_else(
                |error| format!("the bundle frontend refused to open: {error}"),
                |refusal| refusal.message,
            ),
            error => format!("the bundle frontend did not open: {error}"),
        })?;
        let opened = document::OpenedV1::decode(&opened)
            .map_err(|error| format!("the bundle frontend answered a malformed open: {error}"))?;
        check_extra_roles_v1(&opened.extra_roles)
            .map_err(|error| format!("the bundle frontend answered a malformed open: {error}"))?;
        Ok(Self {
            descriptor,
            frontend,
            opened,
        })
    }

    /// Opens a parser of one completion, with the state its render returned.
    pub(crate) fn open_parser(
        &self,
        options: FormatParserOptionsV1,
        parser_state: &[u8],
    ) -> Result<NativeParserV1, String> {
        let request = document::ParserOpenV1 {
            mode: mode_document_v1(options.mode),
            reasoning_open: options.reasoning_open,
            extract_tools: options.extract_tools,
            stream_tool_calls: options.stream_tool_calls,
            raw: options.raw,
            ignore_eos: options.ignore_eos,
            engine_tokens: options.engine_tokens,
            tool_schemas: options.tool_schemas.iter().map(tool_document_v1).collect(),
            parser_state: parser_state.to_vec(),
        };
        let handle = self
            .frontend
            .open_parser(&request.encode())
            .map_err(parser_failure_v1)?;
        Ok(NativeParserV1 {
            handle,
            descriptor: Arc::clone(&self.descriptor),
            mode: options.mode,
            extract_tools: options.extract_tools,
            stream_tool_calls: options.stream_tool_calls,
            request: Vec::new(),
            delivered: 0,
            cursor: 0,
            text_end: 0,
            calls: Vec::new(),
            streamed: Vec::new(),
            finished: false,
            failed: false,
        })
    }

    fn render_failure(&self, error: FrontendCallErrorV1) -> FrontendErrorV1 {
        match error {
            FrontendCallErrorV1::Refused(bytes) => match document::RefusalV1::read(&bytes) {
                Ok(refusal) => FrontendErrorV1 {
                    message: refusal.message,
                    parameter: refusal.parameter,
                },
                Err(error) => FrontendErrorV1::new(
                    format!("the bundle frontend refused with a malformed document: {error}"),
                    None,
                ),
            },
            error => FrontendErrorV1::new(error.to_string(), None),
        }
    }

    /// Rebuilds a rendered prompt from the format's answer, checking that
    /// each media segment names a request media part, at most once and in
    /// increasing order, and that it selects a declared effort.
    ///
    /// A format may leave a request media part out of the prompt, as one
    /// that renders a user message carrying message-level tools as plain
    /// content does. The rendered media are the placed parts, in order, as
    /// the format reports them.
    fn rendered(
        &self,
        answer: &[u8],
        request_media: Vec<FrontendMediaPartV1>,
    ) -> Result<FormatRenderedV1, FrontendErrorV1> {
        let malformed = |reason: &str| {
            FrontendErrorV1::new(
                format!("the bundle frontend answered a malformed render: {reason}"),
                None,
            )
        };
        let rendered =
            document::RenderedV1::decode(answer).map_err(|error| malformed(&error.to_string()))?;
        let mut prompt = String::new();
        let mut segments = Vec::with_capacity(rendered.segments.len());
        let mut media = Vec::new();
        // The request index the next media segment must reach.
        let mut next = 0;
        for segment in rendered.segments {
            match segment {
                document::SegmentV1::Text(text) => {
                    prompt.push_str(&text);
                    segments.push(PromptSegmentV1::Text(text));
                }
                document::SegmentV1::Media(index) => {
                    let index = usize::try_from(index).unwrap_or(usize::MAX);
                    let part = request_media.get(index).filter(|_| index >= next).ok_or_else(|| {
                        malformed(
                            "media segments must name request media parts, at most once and in increasing order",
                        )
                    })?;
                    prompt
                        .push_str(self.pad(part.kind).ok_or_else(|| {
                            malformed("a media segment has no declared modality")
                        })?);
                    segments.push(PromptSegmentV1::Media(media.len()));
                    media.push(part.clone());
                    next = index + 1;
                }
            }
        }
        // The toolkit's prompt writer refuses a prompt past the declared
        // length, with or without media, before a format answers; so does
        // the host.
        if prompt.len() > self.descriptor.limits.maximum_prompt_bytes {
            return Err(prompt_too_long_v1(!media.is_empty()));
        }
        if self.descriptor.reasoning.effort(&rendered.effort).is_none() {
            return Err(malformed("the effort is not a declared effort"));
        }
        // The flag starts the completion in reasoning and gates its
        // constraints, so only a thinking prompt may leave reasoning open.
        if rendered.reasoning_open && rendered.mode == document::ModeV1::Chat {
            return Err(malformed("a chat prompt leaves reasoning open"));
        }
        Ok(FormatRenderedV1 {
            prompt,
            segments,
            media,
            mode: match rendered.mode {
                document::ModeV1::Chat => PromptModeV1::Chat,
                document::ModeV1::Thinking => PromptModeV1::Thinking,
            },
            effort: rendered.effort,
            reasoning_open: rendered.reasoning_open,
            parser_state: rendered.parser_state,
        })
    }

    /// The placeholder text the declaration renders for one media kind.
    fn pad(&self, kind: FrontendMediaPartKindV1) -> Option<&str> {
        let kind = match kind {
            FrontendMediaPartKindV1::Image => FrontendMediaKindV1::Image,
            FrontendMediaPartKindV1::Audio => FrontendMediaKindV1::Audio,
            FrontendMediaPartKindV1::Video => FrontendMediaKindV1::Video,
        };
        self.descriptor
            .multimodal
            .as_ref()?
            .modalities
            .iter()
            .find(|modality| modality.kind == kind)
            .map(|modality| modality.render.pad.as_str())
    }
}

impl FormatRendererV1 for NativeFormatV1 {
    fn render(
        &self,
        request: FormatRenderRequestV1<'_>,
    ) -> Result<FormatRenderedV1, FrontendErrorV1> {
        let (request, media) = render_document_v1(&request);
        // The prompt text the declaration allows, the parser state, and the
        // segments around each media part: past that, the prompt is too long
        // and the document is refused before it is copied.
        let longest_effort = self
            .descriptor
            .reasoning
            .effort_names()
            .iter()
            .map(|name| name.len())
            .max()
            .unwrap_or(0);
        let maximum = self
            .descriptor
            .limits
            .maximum_prompt_bytes
            .saturating_add(document::FRONTEND_PARSER_STATE_BYTES_MAX_V1)
            .saturating_add(longest_effort)
            .saturating_add(media.len().saturating_add(1).saturating_mul(10))
            .saturating_add(RENDER_OVERHEAD_BYTES_V1);
        let answer =
            self.frontend
                .render(&request.encode(), maximum)
                .map_err(|error| match error {
                    FrontendCallErrorV1::Oversized { .. } => prompt_too_long_v1(!media.is_empty()),
                    error => self.render_failure(error),
                })?;
        self.rendered(&answer, media)
    }

    fn tool_constraint(
        &self,
        tools: &[FrontendToolV1],
        choice: &FrontendToolChoiceV1,
    ) -> Result<Option<runtime_grammar::Grammar>, FrontendErrorV1> {
        let request = document::ConstraintRequestV1::ToolChoice {
            tools: tools.iter().map(tool_document_v1).collect(),
            choice: match choice {
                FrontendToolChoiceV1::Auto => document::ToolChoiceV1::Auto,
                FrontendToolChoiceV1::None => document::ToolChoiceV1::None,
                FrontendToolChoiceV1::Required => document::ToolChoiceV1::Required,
                FrontendToolChoiceV1::Named(name) => document::ToolChoiceV1::Named(name.clone()),
            },
        };
        let request = request.encode();
        let maximum = request
            .len()
            .saturating_mul(CONSTRAINT_BYTES_PER_REQUEST_BYTE_V1)
            .saturating_add(CONSTRAINT_OVERHEAD_BYTES_V1);
        let answer = self
            .frontend
            .constraint(&request, maximum)
            .map_err(|error| self.render_failure(error))?;
        if answer.is_empty() {
            return Ok(None);
        }
        // Decoding checks the document's shape and compile depth; the
        // grammar budgets apply at compilation, as for a local grammar.
        runtime_grammar::Grammar::decode_wire(&answer)
            .map(Some)
            .map_err(|error| {
                FrontendErrorV1::new(
                    format!("the bundle frontend answered a malformed constraint: {error}"),
                    None,
                )
            })
    }

    fn opened(&self) -> document::OpenedV1 {
        self.opened.clone()
    }
}

/// Where a completion decoder opens its format parser: the bundle frontend
/// that rendered the prompt, and the parser state its render returned.
///
/// A rendered chat's state carries one ([`crate::BundleChatStateV1`]); a
/// completion that rendered nothing opens with
/// [`crate::BundleFrontendV1::parser_source`] or [`Self::from_declaration`].
#[derive(Clone)]
pub struct BundleParserSourceV1 {
    format: Arc<NativeFormatV1>,
    state: Arc<[u8]>,
}

impl fmt::Debug for BundleParserSourceV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BundleParserSourceV1")
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl BundleParserSourceV1 {
    /// A source over a package's frontend export, opened over the frontend
    /// component a host authenticated, with no rendered state. This is the
    /// serving path: the package's own code reads the package's own bytes.
    pub fn from_export(
        export: &FrontendExportV1,
        bytes: &[u8],
    ) -> Result<Self, FrontendComponentErrorV1> {
        let descriptor = Arc::new(FrontendDescriptorV1::from_component_bytes(bytes)?);
        let format = NativeFormatV1::from_export(export, descriptor, Some(bytes))?;
        Ok(Self::new(Arc::new(format)))
    }

    /// A source over an in-process export, opened over one declaration
    /// written back as JSON, with no rendered state. For a declaration
    /// assembled in code, as tests do: it opens a chat format's export that
    /// renders nothing, since such an export parses from the parser section
    /// alone, as a model's own export does.
    #[cfg(any(test, feature = "test-support"))]
    pub fn from_declaration(
        descriptor: &FrontendDescriptorV1,
    ) -> Result<Self, FrontendComponentErrorV1> {
        descriptor.validate()?;
        let format = NativeFormatV1::from_export(
            &crate::loopback::parser_export_v1(),
            Arc::new(descriptor.clone()),
            None,
        )?;
        Ok(Self::new(Arc::new(format)))
    }

    pub(crate) fn new(format: Arc<NativeFormatV1>) -> Self {
        Self {
            format,
            state: Arc::from([]),
        }
    }

    /// The same frontend, with the state one render returned.
    pub(crate) fn with_state(&self, state: Vec<u8>) -> Self {
        Self {
            format: Arc::clone(&self.format),
            state: state.into(),
        }
    }

    /// The rendered state a parser opens with.
    pub(crate) fn state(&self) -> &[u8] {
        &self.state
    }

    pub(crate) fn open(&self, options: FormatParserOptionsV1) -> Result<NativeParserV1, String> {
        self.format.open_parser(options, &self.state)
    }
}

/// One completion's parser behind a bundle's frontend export.
pub(crate) struct NativeParserV1 {
    handle: FrontendParserHandleV1,
    descriptor: Arc<FrontendDescriptorV1>,
    mode: PromptModeV1,
    extract_tools: bool,
    stream_tool_calls: bool,
    /// The push document, reused across pushes.
    request: Vec<u8>,
    /// How many completion bytes the format holds.
    delivered: usize,
    cursor: usize,
    /// Where the last reported range ended.
    text_end: usize,
    /// The calls delivered so far.
    calls: Vec<ParsedToolCallV1>,
    /// Each streamed call's name and the arguments its deltas added up to.
    streamed: Vec<ParsedToolCallV1>,
    finished: bool,
    /// Whether an answer failed a check, so the parser takes no more.
    failed: bool,
}

impl FormatParserV1 for NativeParserV1 {
    fn push(&mut self, input: FormatInputV1<'_>) -> Result<Vec<FormatEventV1>, String> {
        if self.failed {
            return Err(sequence_v1("bytes pushed after the format parser failed"));
        }
        if self.finished {
            return Err(malformed_v1("bytes received after terminal completion"));
        }
        let keep = self.delivered.min(input.completion.len());
        document::PushV1 {
            keep: keep as u64,
            held: input.held,
            stop_floor: input.stop_floor.map(|floor| floor as u64),
            suffix: &input.completion[keep..],
        }
        .encode_into(&mut self.request);
        let limits = &self.descriptor.limits;
        let per_call = (self.descriptor.tools)
            .maximum_function_name_bytes(limits.maximum_name_bytes)
            .saturating_add(EVENT_BYTES_PER_TOOL_CALL_V1)
            .saturating_mul(3);
        let maximum = input
            .completion
            .len()
            .saturating_mul(EVENT_BYTES_PER_RELEASED_BYTE_V1)
            .saturating_add(limits.maximum_tool_calls.saturating_mul(per_call))
            .saturating_add(EVENT_OVERHEAD_BYTES_V1);
        let answer = self.handle.push(&self.request, input.last, maximum);
        self.delivered = input.completion.len();
        let events = answer
            .map_err(parser_failure_v1)
            .and_then(|answer| self.read_events(&answer, &input));
        self.failed = events.is_err();
        events
    }

    fn cursor(&self) -> usize {
        self.cursor
    }
}

impl NativeParserV1 {
    /// Reads one event document and checks it against what was pushed.
    fn read_events(
        &mut self,
        answer: &[u8],
        input: &FormatInputV1<'_>,
    ) -> Result<Vec<FormatEventV1>, String> {
        let answer = document::EventsV1::decode(answer).map_err(|error| {
            malformed_v1(format!(
                "the format answered a malformed event document: {error}"
            ))
        })?;
        let mut events = Vec::with_capacity(answer.events.len());
        for event in answer.events {
            if self.finished {
                return Err(malformed_v1("the format reported an event after finishing"));
            }
            events.push(match event {
                document::EventV1::Reasoning(range) => {
                    if self.mode != PromptModeV1::Thinking {
                        return Err(malformed_v1("the format reported reasoning in chat mode"));
                    }
                    FormatEventV1::Reasoning(self.range(range, input)?)
                }
                document::EventV1::Content(range) => {
                    FormatEventV1::Content(self.range(range, input)?)
                }
                document::EventV1::ToolCalls { range, calls } => {
                    let range = self.range(range, input)?;
                    FormatEventV1::ToolCalls {
                        calls: self.deliver(&calls)?,
                        range,
                    }
                }
                document::EventV1::ToolCallDeltas(deltas) => {
                    FormatEventV1::ToolCallDeltas(self.stream(&deltas)?)
                }
                document::EventV1::Finish { reason, completion } => {
                    self.finish(reason, completion, input)?
                }
            });
        }
        let cursor = usize::try_from(answer.cursor).unwrap_or(usize::MAX);
        if cursor < self.cursor || cursor > input.completion.len() {
            return Err(malformed_v1(
                "the format moved its cursor back or past the released bytes",
            ));
        }
        // The host scans for client stops from the cursor, so one inside a
        // character or ill-formed run would read its tail as replacement
        // text.
        if splits_decoding_unit_v1(input.completion, cursor) {
            return Err(malformed_v1(
                "the format moved its cursor inside a UTF-8 character or ill-formed byte run",
            ));
        }
        self.cursor = cursor;
        // A finished completion has read every released byte.
        if self.finished && cursor != input.completion.len() {
            return Err(malformed_v1(
                "the format finished before reading every released byte",
            ));
        }
        if input.last && !self.finished {
            return Err(malformed_v1("the format did not finish on the final push"));
        }
        Ok(events)
    }

    /// One reported range: within the released bytes, after every range
    /// reported before it, and with neither end inside a decoding unit.
    fn range(
        &mut self,
        range: std::ops::Range<u64>,
        input: &FormatInputV1<'_>,
    ) -> Result<std::ops::Range<usize>, String> {
        let start = usize::try_from(range.start).unwrap_or(usize::MAX);
        let end = usize::try_from(range.end).unwrap_or(usize::MAX);
        if start < self.text_end || end > input.completion.len() {
            return Err(malformed_v1(
                "the format reported a range that overlaps an earlier one or passes the released bytes",
            ));
        }
        if splits_decoding_unit_v1(input.completion, start)
            || splits_decoding_unit_v1(input.completion, end)
        {
            return Err(malformed_v1(
                "the format reported a range that splits a UTF-8 character or ill-formed byte run",
            ));
        }
        self.text_end = end;
        Ok(start..end)
    }

    /// One delivered group of calls, each a declared name with a JSON
    /// object of arguments, matching what its deltas streamed.
    fn deliver(&mut self, calls: &[document::CallV1<'_>]) -> Result<Vec<ParsedToolCallV1>, String> {
        if !self.extract_tools {
            return Err(malformed_v1(
                "the format delivered tool calls the request does not read",
            ));
        }
        if calls.is_empty() {
            return Err(malformed_v1(
                "the format delivered an empty tool-call group",
            ));
        }
        if self.calls.len() + calls.len() > self.descriptor.limits.maximum_tool_calls {
            return Err(malformed_v1(
                "the format delivered more tool calls than the declaration allows",
            ));
        }
        let mut delivered = Vec::with_capacity(calls.len());
        for call in calls {
            self.check_name(call.name)?;
            // Only the shape: the transport charges the arguments against
            // the declared limits when it checks the call, as it always has.
            if !json_object_text_v1(call.arguments) {
                return Err(malformed_v1(
                    "the format delivered tool-call arguments that are not a JSON object",
                ));
            }
            let call = ParsedToolCallV1 {
                name: call.name.to_owned(),
                arguments: call.arguments.to_owned(),
            };
            if self
                .streamed
                .get(self.calls.len())
                .is_some_and(|streamed| streamed.arguments != call.arguments)
            {
                return Err(malformed_v1(
                    "the format's tool-call deltas do not add up to the delivered call",
                ));
            }
            self.calls.push(call.clone());
            delivered.push(call);
        }
        Ok(delivered)
    }

    /// One group of deltas: each extends the call it names, the next call
    /// opens with its name, and none reaches a call already delivered.
    fn stream(
        &mut self,
        deltas: &[document::DeltaV1<'_>],
    ) -> Result<Vec<FormatToolCallDeltaV1>, String> {
        if !self.extract_tools || !self.stream_tool_calls {
            return Err(malformed_v1(
                "the format streamed tool calls the request does not stream",
            ));
        }
        let mut streamed = Vec::with_capacity(deltas.len());
        for delta in deltas {
            let index = delta.index as usize;
            if index < self.calls.len() || index > self.streamed.len() {
                return Err(malformed_v1(
                    "the format streamed a delta outside the open tool calls",
                ));
            }
            match (index == self.streamed.len(), delta.name) {
                (true, Some(name)) => {
                    self.check_name(name)?;
                    self.streamed.push(ParsedToolCallV1 {
                        name: name.to_owned(),
                        arguments: delta.arguments.to_owned(),
                    });
                }
                (false, None) => self.streamed[index].arguments.push_str(delta.arguments),
                _ => {
                    return Err(malformed_v1(
                        "the format named a tool call on other than its first delta",
                    ));
                }
            }
            streamed.push(FormatToolCallDeltaV1 {
                index,
                name: delta.name.map(str::to_owned),
                arguments: delta.arguments.to_owned(),
            });
        }
        Ok(streamed)
    }

    fn check_name(&self, name: &str) -> Result<(), String> {
        validate_function_name_v1(&self.descriptor, name, "parsed function")
            .map_err(|_| malformed_v1("the format named a tool outside the declared name charset"))
    }

    /// The completion's end: once, at the end of the released bytes, with a
    /// reason and answer that agree with the calls delivered.
    fn finish(
        &mut self,
        reason: document::FinishReasonV1,
        completion: Option<document::CompletionV1<'_>>,
        input: &FormatInputV1<'_>,
    ) -> Result<FormatEventV1, String> {
        if input.held {
            return Err(malformed_v1(
                "the format finished before the end of the released bytes",
            ));
        }
        self.finished = true;
        let reason = match reason {
            document::FinishReasonV1::Stop => CompletionFinishV1::Stop,
            document::FinishReasonV1::Length => CompletionFinishV1::Length,
            document::FinishReasonV1::ToolCalls => CompletionFinishV1::ToolCalls,
        };
        // Every streamed call ends delivered whole, which checks its
        // arguments and that its deltas add up to them.
        if self.streamed.len() > self.calls.len() {
            return Err(malformed_v1(
                "the format finished with a streamed tool call it never delivered",
            ));
        }
        if (reason == CompletionFinishV1::ToolCalls) == self.calls.is_empty() {
            return Err(malformed_v1(
                "the format's finish reason disagrees with the calls it delivered",
            ));
        }
        let completion = completion
            .map(|completion| {
                if completion.reasoning_content.is_some() && self.mode != PromptModeV1::Thinking {
                    return Err(malformed_v1("the format reported reasoning in chat mode"));
                }
                let tool_calls: Vec<ParsedToolCallV1> = completion
                    .tool_calls
                    .iter()
                    .map(|call| ParsedToolCallV1 {
                        name: call.name.to_owned(),
                        arguments: call.arguments.to_owned(),
                    })
                    .collect();
                if tool_calls != self.calls {
                    return Err(malformed_v1(
                        "the format's completion disagrees with the calls it delivered",
                    ));
                }
                Ok(AssistantCompletionV1 {
                    content: completion.content.to_owned(),
                    reasoning_content: completion.reasoning_content.map(str::to_owned),
                    tool_calls,
                })
            })
            .transpose()?;
        Ok(FormatEventV1::Finish { reason, completion })
    }
}

/// Whether `offset` falls inside a decoding unit of `bytes`: a valid
/// character or a maximal ill-formed subpart, which replacement decoding
/// reads as one U+FFFD. A unit is at most four bytes and a lead byte never
/// continues an earlier unit, so only the units the three bytes before
/// `offset` begin can contain it.
fn splits_decoding_unit_v1(bytes: &[u8], offset: usize) -> bool {
    (offset.saturating_sub(3)..offset).any(|start| {
        let window = &bytes[start..bytes.len().min(start + 4)];
        let unit = window.utf8_chunks().next().map_or(0, |chunk| {
            chunk
                .valid()
                .chars()
                .next()
                .map_or(chunk.invalid().len(), char::len_utf8)
        });
        start + unit > offset
    })
}

/// The refusal a format's prompt writer gives a prompt past the declared
/// length.
fn prompt_too_long_v1(media: bool) -> FrontendErrorV1 {
    FrontendErrorV1::new(
        resource_v1(if media {
            "rendered media prompt exceeds maximum_prompt_bytes"
        } else {
            "prompt exceeds the byte limit"
        }),
        Some("messages"),
    )
}

/// A failed parser call as the decoder reports it.
fn parser_failure_v1(error: FrontendCallErrorV1) -> String {
    match error {
        FrontendCallErrorV1::Refused(bytes) => match document::RefusalV1::read(&bytes) {
            Ok(refusal)
                if DECODER_FAMILIES_V1
                    .iter()
                    .any(|family| refusal.message.starts_with(family)) =>
            {
                refusal.message
            }
            Ok(refusal) => malformed_v1(refusal.message),
            Err(error) => malformed_v1(format!(
                "the format refused with a malformed document: {error}"
            )),
        },
        // As the decoder reports a panic of a parser it calls directly.
        FrontendCallErrorV1::Panicked => malformed_v1("decoder panicked"),
        FrontendCallErrorV1::ParserFinished => {
            sequence_v1("bytes pushed after the format parser finished")
        }
        error => malformed_v1(error.to_string()),
    }
}

/// The roles every request may carry, which a frontend cannot declare again.
const COMMON_ROLES_V1: [&str; 5] = ["system", "developer", "user", "assistant", "tool"];

/// Checks the extra roles an open answer declared: at most
/// [`document::FRONTEND_EXTRA_ROLES_MAX_V1`], each a distinct name of ASCII
/// letters, digits, `_` and `-` of at most
/// [`document::FRONTEND_ROLE_NAME_BYTES_MAX_V1`] bytes, none a common role.
fn check_extra_roles_v1(roles: &[String]) -> Result<(), String> {
    if roles.len() > document::FRONTEND_EXTRA_ROLES_MAX_V1 {
        return Err(format!(
            "it declares {} extra roles, past the {} a frontend may",
            roles.len(),
            document::FRONTEND_EXTRA_ROLES_MAX_V1
        ));
    }
    for (index, role) in roles.iter().enumerate() {
        let named = !role.is_empty()
            && role.len() <= document::FRONTEND_ROLE_NAME_BYTES_MAX_V1
            && role
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-');
        if !named {
            return Err(format!("extra role {role:?} is not a role name"));
        }
        if COMMON_ROLES_V1.contains(&role.as_str()) {
            return Err(format!("extra role {role:?} is a common role"));
        }
        if roles[..index].contains(role) {
            return Err(format!("extra role {role:?} is declared twice"));
        }
    }
    Ok(())
}

fn mode_document_v1(mode: PromptModeV1) -> document::ModeV1 {
    match mode {
        PromptModeV1::Chat => document::ModeV1::Chat,
        PromptModeV1::Thinking => document::ModeV1::Thinking,
    }
}

fn tool_document_v1(tool: &FrontendToolV1) -> document::ToolV1 {
    document::ToolV1 {
        name: tool.name.clone(),
        description: tool.description.clone(),
        parameters: json_text_v1(&tool.parameters),
        parameters_present: tool.parameters_present,
        strict: tool.strict,
    }
}

/// The render request document, and the request's media parts in order.
fn render_document_v1(
    request: &FormatRenderRequestV1<'_>,
) -> (document::RenderRequestV1, Vec<FrontendMediaPartV1>) {
    let mut media = Vec::new();
    let messages = request
        .messages
        .iter()
        .map(|message| message_document_v1(message, &mut media))
        .collect();
    let keywords = [
        (
            "enable_thinking",
            request.enable_thinking.map(FrontendJsonV1::Bool),
        ),
        ("thinking", request.thinking.map(FrontendJsonV1::Bool)),
        (
            "reasoning_effort",
            request
                .nested_reasoning_effort
                .clone()
                .map(FrontendJsonV1::String),
        ),
        (
            "preserve_thinking",
            request.preserve_thinking.map(FrontendJsonV1::Bool),
        ),
        ("clear_thinking", request.clear_thinking.clone()),
        ("add_vision_id", request.add_vision_id.clone()),
    ]
    .into_iter()
    .filter_map(|(name, value)| value.map(|value| (name.to_owned(), value)))
    .collect::<Vec<_>>();
    // The client's object goes whole, every key as sent, when the transport
    // carried one. A request assembled without one carries the typed
    // keywords set.
    let chat_template_kwargs = match &request.chat_template_kwargs {
        Some(kwargs) => Some(json_text_v1(kwargs)),
        None => (!keywords.is_empty()).then(|| json_text_v1(&FrontendJsonV1::Object(keywords))),
    };
    let document = document::RenderRequestV1 {
        messages,
        continue_final_message: request.continue_final_message,
        add_generation_prompt: request.add_generation_prompt,
        tools: request.tools.iter().map(tool_document_v1).collect(),
        tool_objects: request.tool_objects.iter().map(json_text_v1).collect(),
        response_format: request.response_format.as_ref().map(json_text_v1),
        reasoning_effort: request.reasoning_effort.clone(),
        chat_template_kwargs,
    };
    (document, media)
}

fn message_document_v1(
    message: &FrontendMessageV1,
    media: &mut Vec<FrontendMediaPartV1>,
) -> document::MessageV1 {
    document::MessageV1 {
        role: match &message.role {
            FrontendRoleV1::System => "system",
            FrontendRoleV1::Developer => "developer",
            FrontendRoleV1::User => "user",
            FrontendRoleV1::Assistant => "assistant",
            FrontendRoleV1::Tool => "tool",
            FrontendRoleV1::Declared(name) => name,
        }
        .to_owned(),
        content: message.content.clone(),
        parts: message.parts.as_ref().map(|parts| {
            parts
                .iter()
                .map(|part| match part {
                    FrontendContentPartV1::Text(text) => document::PartV1::Text(text.clone()),
                    FrontendContentPartV1::Media(part) => {
                        let index = u32::try_from(media.len())
                            .expect("a request carries far fewer media parts than u32");
                        media.push(part.clone());
                        document::PartV1::Media(document::MediaPartV1 {
                            index,
                            kind: match part.kind {
                                FrontendMediaPartKindV1::Image => document::MediaKindV1::Image,
                                FrontendMediaPartKindV1::Audio => document::MediaKindV1::Audio,
                                FrontendMediaPartKindV1::Video => document::MediaKindV1::Video,
                            },
                            part_type: part.part_type.clone(),
                        })
                    }
                    FrontendContentPartV1::ToolReference { name } => {
                        document::PartV1::ToolReference(name.clone())
                    }
                })
                .collect()
        }),
        // The transport resolves the history reasoning field into
        // `reasoning_content` today.
        reasoning: None,
        reasoning_content: message.reasoning_content.clone(),
        tool_calls: message
            .tool_calls
            .iter()
            .map(|call| document::ToolCallV1 {
                id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            })
            .collect(),
        tool_call_id: message.tool_call_id.clone(),
        tools: message.tools.iter().map(json_text_v1).collect(),
    }
}

/// Whether `text` is one JSON object by JSON's syntax alone. It applies no
/// size, depth or duplicate-key bound: those are the transport's, which
/// checks every delivered call as it did before calls crossed documents.
/// It reads iteratively, so no nesting exhausts the stack.
fn json_object_text_v1(text: &str) -> bool {
    let bytes = text.as_bytes();
    let space = |mut at: usize| {
        while matches!(bytes.get(at), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            at += 1;
        }
        at
    };
    let digits = |mut at: usize| {
        let start = at;
        while bytes.get(at).is_some_and(u8::is_ascii_digit) {
            at += 1;
        }
        (at > start).then_some(at)
    };
    let string = |mut at: usize| -> Option<usize> {
        if bytes.get(at) != Some(&b'"') {
            return None;
        }
        at += 1;
        loop {
            match *bytes.get(at)? {
                b'"' => return Some(at + 1),
                b'\\' => match *bytes.get(at + 1)? {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => at += 2,
                    b'u' if bytes.get(at + 2..at + 6)?.iter().all(u8::is_ascii_hexdigit) => {
                        at += 6;
                    }
                    _ => return None,
                },
                0..=0x1f => return None,
                _ => at += 1,
            }
        }
    };
    let number = |mut at: usize| -> Option<usize> {
        if bytes.get(at) == Some(&b'-') {
            at += 1;
        }
        at = match bytes.get(at)? {
            b'0' => at + 1,
            b'1'..=b'9' => digits(at)?,
            _ => return None,
        };
        if bytes.get(at) == Some(&b'.') {
            at = digits(at + 1)?;
        }
        if matches!(bytes.get(at), Some(b'e' | b'E')) {
            at += 1;
            if matches!(bytes.get(at), Some(b'+' | b'-')) {
                at += 1;
            }
            at = digits(at)?;
        }
        Some(at)
    };
    // A member's key and colon, to where its value may begin.
    let key = |at: usize| -> Option<usize> {
        let at = space(string(at)?);
        (bytes.get(at) == Some(&b':')).then_some(at + 1)
    };
    let mut at = space(0);
    if bytes.get(at) != Some(&b'{') {
        return false;
    }
    // Each open container: `true` for an object.
    let mut open: Vec<bool> = Vec::new();
    loop {
        // One value.
        at = space(at);
        let Some(&first) = bytes.get(at) else {
            return false;
        };
        let end = match first {
            b'{' | b'[' => {
                let object = first == b'{';
                let close = if object { b'}' } else { b']' };
                let inner = space(at + 1);
                if bytes.get(inner) == Some(&close) {
                    Some(inner + 1)
                } else {
                    open.push(object);
                    at = inner;
                    if object {
                        let Some(value) = key(at) else {
                            return false;
                        };
                        at = value;
                    }
                    continue;
                }
            }
            b'"' => string(at),
            b't' => bytes[at..].starts_with(b"true").then_some(at + 4),
            b'f' => bytes[at..].starts_with(b"false").then_some(at + 5),
            b'n' => bytes[at..].starts_with(b"null").then_some(at + 4),
            _ => number(at),
        };
        let Some(end) = end else {
            return false;
        };
        at = end;
        // Close what the value ends, or move to the next member.
        loop {
            at = space(at);
            let Some(&object) = open.last() else {
                return at == bytes.len();
            };
            match bytes.get(at) {
                Some(b',') => {
                    at = space(at + 1);
                    if object {
                        let Some(value) = key(at) else {
                            return false;
                        };
                        at = value;
                    }
                    break;
                }
                Some(b'}') if object => {
                    open.pop();
                    at += 1;
                }
                Some(b']') if !object => {
                    open.pop();
                    at += 1;
                }
                _ => return false,
            }
        }
    }
}

/// One JSON value as compact JSON text: keys in their order and numbers in
/// their digits, so reading it back yields the same value.
pub(crate) fn json_text_v1(value: &FrontendJsonV1) -> String {
    let mut text = String::new();
    write_json_text_v1(value, &mut text);
    text
}

fn write_json_text_v1(value: &FrontendJsonV1, text: &mut String) {
    match value {
        FrontendJsonV1::Null => text.push_str("null"),
        FrontendJsonV1::Bool(value) => text.push_str(if *value { "true" } else { "false" }),
        FrontendJsonV1::Number(number) => text.push_str(&number.to_string()),
        FrontendJsonV1::String(value) => write_json_string_v1(value, text),
        FrontendJsonV1::Array(values) => {
            text.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    text.push(',');
                }
                write_json_text_v1(value, text);
            }
            text.push(']');
        }
        FrontendJsonV1::Object(entries) => {
            text.push('{');
            for (index, (name, value)) in entries.iter().enumerate() {
                if index > 0 {
                    text.push(',');
                }
                write_json_string_v1(name, text);
                text.push(':');
                write_json_text_v1(value, text);
            }
            text.push('}');
        }
    }
}

fn write_json_string_v1(value: &str, text: &mut String) {
    text.push_str(&serde_json::to_string(value).expect("a string serializes"));
}

#[cfg(test)]
#[path = "native_tests.rs"]
mod tests;
