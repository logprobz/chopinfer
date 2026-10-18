//! Reading one completion back out of the markers a bundle declared.
//!
//! Generation arrives one token at a time, and a client is waiting, so the
//! parser emits reasoning and content as soon as the bytes for them can no
//! longer turn out to be the start of a structural marker. What counts as a
//! marker is the installed declaration's: the same strings the renderer wrote
//! into the prompt are the ones read back out of the answer, so a bundle whose
//! template differs decodes differently through this same code.
//!
//! Nothing a completion carries is trusted. A boundary the declaration
//! reserved is a refusal wherever it is not the one marker this state expects,
//! and every tool call is parsed against the declared grammar rather than
//! believed.
//!
//! The host layer in `decode.rs` releases the bytes this reads, in the
//! declaration's `format.parser` section: its tool regions stream in the
//! shared tool-call grammar the section names, and generation that ends
//! inside one delivers the calls it finished.

use std::str;
use std::sync::Arc;

use crate::chat::PromptModeV1;
use crate::chatml_decode::{
    ChatMlTerminalsV1, RegionCallV1, RegionScanV1, ToolRegionV1, call_arguments_v1,
};
use crate::descriptor::{FormatDescriptorV1, FrontendParserDeclarationV1, FrontendToolGrammarV1};
use crate::format::{
    AssistantCompletionV1, CompletionFinishV1, FormatEventV1, FormatInputV1, FormatParserOptionsV1,
    FormatToolCallDeltaV1, ParsedToolCallV1,
};
use crate::request::FrontendToolV1;
use crate::toolkit::{malformed_v1, resource_v1, sequence_v1, validate_function_name_v1};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TextStateV1 {
    Reasoning,
    Content,
    Tool,
    Complete,
}

#[derive(Clone, Copy)]
enum MarkerV1 {
    LiteralTerminator,
    /// A repeated reasoning open inside reasoning, which is markup.
    ThinkStart,
    ThinkEnd,
    ToolStart,
    Terminator,
    Forbidden,
}

/// An open ChatML tool region: its text and its scan, kept across tokens so
/// each token reads only the bytes it added.
struct OpenRegionV1 {
    /// Where the region begins in the completion bytes.
    start: usize,
    /// Where the bytes `text` decodes end.
    end: usize,
    text: String,
    /// Whether `text` is those bytes unchanged, so later bytes extend it.
    exact: bool,
    scan: RegionScanV1,
    /// How many named calls are closed for good and were streamed whole.
    streamed: usize,
}

impl OpenRegionV1 {
    fn new(start: usize) -> Self {
        Self {
            start,
            end: start,
            text: String::new(),
            exact: true,
            scan: RegionScanV1::default(),
            streamed: 0,
        }
    }

    /// Brings `text` to the replacement decoding of `bytes[start..end]`.
    ///
    /// Bytes that extend valid text decode on their own; any other change
    /// reads the region again from its start.
    fn read(&mut self, bytes: &[u8], end: usize) {
        if !self.exact || end < self.end {
            *self = Self::new(self.start);
        }
        let added = &bytes[self.end..end];
        match str::from_utf8(added) {
            Ok(added) => self.text.push_str(added),
            Err(_) => {
                self.text.push_str(&String::from_utf8_lossy(added));
                self.exact = false;
            }
        }
        self.end = end;
    }
}

/// Parses one completion against the markers its bundle declared.
pub struct DeclaredParserV1 {
    descriptor: Arc<FormatDescriptorV1>,
    /// What the shared parser reads: the declaration's `format.parser`.
    parser: Arc<FrontendParserDeclarationV1>,
    mode: PromptModeV1,
    state: TextStateV1,
    /// The completion bytes released so far.
    bytes: Vec<u8>,
    /// Whether the host holds bytes after `bytes`.
    held: bool,
    stop_floor: Option<usize>,
    /// Where an end marker the text spells can still begin: the search for
    /// one resumes here, so a long tool region costs its bytes once.
    terminator_from: std::cell::Cell<usize>,
    cursor: usize,
    tool_start: Option<usize>,
    /// Where the answer text began, so a held blank answer can be found.
    content_start: usize,
    /// Engine tokens carry stop IDs; their ordinary marker spellings are text.
    engine_tokens: bool,
    ignore_eos: bool,
    stream_tool_calls: bool,
    raw: bool,
    tool_schemas: Vec<FrontendToolV1>,
    /// Whether the request lets tool calls be read out of the completion.
    extract_tools: bool,
    streamed_tools: Vec<ParsedToolCallV1>,
    /// The calls delivered so far, in order.
    calls: Vec<ParsedToolCallV1>,
    /// Whether the answer text follows a tool region, so blank text is held.
    after_tools: bool,
    reasoning_text: String,
    content_text: String,
    /// The open ChatML tool region, read as it grows.
    region: Option<OpenRegionV1>,
}

impl DeclaredParserV1 {
    /// Reads the bytes `input` adds, and returns what they settled.
    pub fn push(&mut self, input: FormatInputV1<'_>) -> Result<Vec<FormatEventV1>, String> {
        // The host only appends, except that a client stop may cut the
        // completion short, so this copy follows it with the bytes it adds.
        let keep = self.bytes.len().min(input.completion.len());
        self.push_suffix(
            keep,
            &input.completion[keep..],
            input.held,
            input.last,
            input.stop_floor,
        )
    }

    /// Where the bytes the parser has not settled begin. The host matches
    /// client stops from here.
    #[must_use]
    pub const fn cursor(&self) -> usize {
        self.cursor
    }

    /// Reads a push that keeps the first `keep` released bytes and appends
    /// `suffix`, as a push across the bundle ABI carries it.
    pub fn push_suffix(
        &mut self,
        keep: usize,
        suffix: &[u8],
        held: bool,
        last: bool,
        stop_floor: Option<usize>,
    ) -> Result<Vec<FormatEventV1>, String> {
        if self.state == TextStateV1::Complete {
            return Err(malformed_v1("bytes received after terminal completion"));
        }
        if keep > self.bytes.len() {
            return Err(sequence_v1("a push keeps bytes the parser never received"));
        }
        if keep < self.bytes.len() {
            let longest = self.terminators().iter().map(|text| text.len()).max();
            let from = keep.saturating_sub(longest.unwrap_or(1).saturating_sub(1));
            self.terminator_from
                .set(self.terminator_from.get().min(from));
        }
        self.bytes.truncate(keep);
        self.bytes.extend_from_slice(suffix);
        self.held = held;
        self.stop_floor = stop_floor;
        let valid_end = self.bytes.len();
        if last {
            self.finish(valid_end)
        } else {
            self.advance_text(valid_end)
        }
    }

    /// The shared parser over `descriptor`'s parser section.
    #[must_use]
    pub fn new(descriptor: Arc<FormatDescriptorV1>, options: FormatParserOptionsV1) -> Self {
        let FormatParserOptionsV1 {
            mode,
            reasoning_open,
            extract_tools,
            stream_tool_calls,
            raw,
            ignore_eos,
            engine_tokens,
            tool_schemas,
        } = options;
        let parser = Arc::new(descriptor.parser.clone());
        Self {
            descriptor,
            parser,
            mode,
            state: if reasoning_open {
                TextStateV1::Reasoning
            } else {
                TextStateV1::Content
            },
            bytes: Vec::new(),
            held: false,
            stop_floor: None,
            terminator_from: std::cell::Cell::new(0),
            cursor: 0,
            tool_start: None,
            content_start: 0,
            engine_tokens,
            ignore_eos,
            stream_tool_calls,
            raw,
            tool_schemas,
            extract_tools,
            streamed_tools: Vec::new(),
            calls: Vec::new(),
            after_tools: false,
            reasoning_text: String::new(),
            content_text: String::new(),
            region: None,
        }
    }

    /// Where held text that a generation ended in begins an unfinished call
    /// opening, for the typed grammar: the longest tail of the pending text
    /// that is a proper prefix of the tool call open and already spells a
    /// reserved marker, which text may not carry. The vendor's stream
    /// parser has entered its calls by then and drops the unfinished block,
    /// so it is not text; a shorter tail is.
    fn unfinished_open(&self, end: usize) -> Option<usize> {
        if self.parser.tool_grammar != FrontendToolGrammarV1::XmlTypedParametersV1 {
            return None;
        }
        let open = self.parser.tool_call_open.as_bytes();
        let pending = &self.bytes[self.cursor..end];
        (1..open.len().min(pending.len() + 1))
            .rev()
            .find(|&length| {
                let prefix = &open[..length];
                pending.ends_with(prefix)
                    && self
                        .parser
                        .reserved_markers
                        .iter()
                        .any(|marker| find_bytes_v1(prefix, marker.as_bytes()).is_some())
            })
            .map(|length| end - length)
    }

    /// Reads the last bytes, and ends the completion where generation ended.
    fn finish(&mut self, valid_end: usize) -> Result<Vec<FormatEventV1>, String> {
        let mut updates = self.advance_text(valid_end)?;
        if self.state == TextStateV1::Complete {
            return Ok(updates);
        }
        if self.state == TextStateV1::Tool {
            // Generation ended inside a call, which is finished the way the
            // reference parser finishes one, whether or not it streams.
            let start = self
                .tool_start
                .ok_or_else(|| malformed_v1("missing tool start"))?;
            let end = self.bytes.len();
            self.finish_region(start, end, &mut updates)?;
        } else if let Some(start) = self.unfinished_open(self.bytes.len()) {
            // An unfinished call opening is structure, not text: the text
            // before it is sent as before a call.
            if start > self.cursor && !self.holds_blank_answer(start) {
                updates.push(self.text_update(self.cursor, start)?);
            }
        } else if self.cursor < self.bytes.len() && !self.drops_blank_answer(self.bytes.len()) {
            updates.push(self.text_update(self.cursor, self.bytes.len())?);
        }
        self.cursor = self.bytes.len();
        self.state = TextStateV1::Complete;
        updates.push(FormatEventV1::Finish {
            reason: self.finish_reason(CompletionFinishV1::Length),
            completion: None,
        });
        Ok(updates)
    }

    /// Whether a completion end at `end` is the last byte of the completion.
    fn ends_completion(&self, end: usize) -> bool {
        !self.held && end == self.bytes.len()
    }

    /// Why this generation ended: after a delivered call, because of it.
    fn finish_reason(&self, otherwise: CompletionFinishV1) -> CompletionFinishV1 {
        if self.calls.is_empty() {
            otherwise
        } else {
            CompletionFinishV1::ToolCalls
        }
    }

    fn advance_text(&mut self, valid_end: usize) -> Result<Vec<FormatEventV1>, String> {
        let mut updates = Vec::new();
        loop {
            if self.state == TextStateV1::Tool {
                if self.advance_region(valid_end, &mut updates)? {
                    continue;
                }
                return Ok(updates);
            }

            let marker = self.next_marker(valid_end);
            let Some((position, marker, length)) = marker else {
                let hold = self.longest_marker_prefix(valid_end);
                let safe_end = valid_end.saturating_sub(hold);
                if safe_end > self.cursor && !self.holds_blank_answer(safe_end) {
                    updates.push(self.text_update(self.cursor, safe_end)?);
                    self.cursor = safe_end;
                }
                return Ok(updates);
            };
            // A blank answer before a tool call is dropped rather than sent,
            // and so is blank text between or after tool calls.
            if position > self.cursor
                && !(matches!(marker, MarkerV1::ToolStart) && self.holds_blank_answer(position))
                && !(matches!(marker, MarkerV1::Terminator) && self.drops_blank_answer(position))
            {
                updates.push(self.text_update(self.cursor, position)?);
            }
            match marker {
                MarkerV1::ThinkStart => self.cursor = position + length,
                MarkerV1::ThinkEnd => {
                    self.cursor = position + length;
                    self.content_start = self.cursor;
                    self.state = TextStateV1::Content;
                }
                MarkerV1::ToolStart => {
                    self.cursor = position;
                    self.tool_start = Some(position);
                    self.state = TextStateV1::Tool;
                }
                MarkerV1::LiteralTerminator => {
                    updates.push(self.text_update(position, position + length)?);
                    self.cursor = position + length;
                }
                MarkerV1::Terminator => {
                    let terminator_end = position + length;
                    if !self.ends_completion(terminator_end) {
                        return Err(malformed_v1("trailing bytes after the completion end"));
                    }
                    let completion = self.completion();
                    let reason = if completion.tool_calls.is_empty() {
                        CompletionFinishV1::Stop
                    } else {
                        CompletionFinishV1::ToolCalls
                    };
                    updates.push(FormatEventV1::Finish {
                        reason,
                        completion: Some(completion),
                    });
                    self.cursor = terminator_end;
                    self.state = TextStateV1::Complete;
                    return Ok(updates);
                }
                MarkerV1::Forbidden => {
                    return Err(malformed_v1(
                        "completion contains an unexpected reserved boundary",
                    ));
                }
            }
        }
    }

    /// The whole answer at the completion end: the text and calls the
    /// incremental parser delivered. Text already delivered stays text,
    /// including end-marker spellings emitted before the minimum token count.
    fn completion(&self) -> AssistantCompletionV1 {
        AssistantCompletionV1 {
            content: self.content_text.clone(),
            reasoning_content: (self.mode == PromptModeV1::Thinking)
                .then(|| self.reasoning_text.clone()),
            tool_calls: self.calls.clone(),
        }
    }

    /// Advances one ChatML tool region, and reports whether text follows it.
    ///
    /// The region ends at the wrapper that returns to text, at the completion
    /// end, or at a client stop. Its calls stream as they grow; each finished
    /// region delivers its calls once, before any text that follows it.
    fn advance_region(
        &mut self,
        valid_end: usize,
        updates: &mut Vec<FormatEventV1>,
    ) -> Result<bool, String> {
        let start = self
            .tool_start
            .ok_or_else(|| malformed_v1("missing tool start"))?;
        let end = self.next_terminator(valid_end);
        let limit = end.map_or_else(
            || valid_end - self.longest_marker_prefix(valid_end),
            |(position, _)| position,
        );
        let mut open = match self.region.take() {
            Some(open) if open.start == start => open,
            _ => OpenRegionV1::new(start),
        };
        open.read(&self.bytes, limit);
        let parser = Arc::clone(&self.parser);
        open.scan.scan(
            ChatMlTerminalsV1::new(&parser),
            &open.text,
            &self.tool_schemas,
        );
        let region = open.scan.region();
        if let Some(used) = region.end {
            let used = source_offset(&self.bytes[start..limit], used);
            self.deliver_region(start, start + used, region, &open.text, updates)?;
            self.cursor = start + used;
            self.content_start = self.cursor;
            self.after_tools = self.extract_tools;
            self.tool_start = None;
            self.state = TextStateV1::Content;
            return Ok(true);
        }
        if self.stream_tool_calls {
            let settled = open.scan.settled_calls();
            self.stream_region(region, &open.text, settled, &mut open.streamed, updates)?;
        }
        let Some((terminator_start, terminator_length)) = end else {
            self.region = Some(open);
            return Ok(false);
        };
        let terminator_end = terminator_start + terminator_length;
        if !self.ends_completion(terminator_end) {
            return Err(malformed_v1("trailing bytes after the completion end"));
        }
        self.deliver_region(start, terminator_start, region, &open.text, updates)?;
        updates.push(FormatEventV1::Finish {
            reason: self.finish_reason(CompletionFinishV1::Stop),
            completion: Some(self.completion()),
        });
        self.cursor = terminator_end;
        self.state = TextStateV1::Complete;
        Ok(false)
    }

    /// Finishes the open region at `end`, when generation stops inside it.
    fn finish_region(
        &mut self,
        start: usize,
        end: usize,
        updates: &mut Vec<FormatEventV1>,
    ) -> Result<(), String> {
        let text = String::from_utf8_lossy(&self.bytes[start..end]).into_owned();
        let parser = Arc::clone(&self.parser);
        let region = crate::chatml_decode::scan_tool_region(
            ChatMlTerminalsV1::new(&parser),
            &text,
            &self.tool_schemas,
        );
        self.region = None;
        self.deliver_region(start, end, &region, &text, updates)
    }

    /// Streams the calls of an open region as far as they can no longer change.
    ///
    /// The first `settled` calls stay as they are whatever text follows, so
    /// the named ones among the first `streamed`, already sent whole, would
    /// send nothing again and are skipped.
    fn stream_region(
        &mut self,
        region: &ToolRegionV1,
        text: &str,
        settled: usize,
        streamed: &mut usize,
        updates: &mut Vec<FormatEventV1>,
    ) -> Result<(), String> {
        if !self.extract_tools {
            return Ok(());
        }
        let descriptor = Arc::clone(&self.descriptor);
        let parser = Arc::clone(&self.parser);
        let terminals = ChatMlTerminalsV1::new(&parser);
        let mut deltas = Vec::new();
        for (offset, call) in named_calls(region).enumerate().skip(*streamed) {
            validate_function_name_v1(&descriptor, &call.name, "parsed function")?;
            let arguments = call_arguments_v1(
                &descriptor,
                terminals,
                call,
                text,
                &self.tool_schemas,
                !call.closed,
            )?;
            self.stream_call(
                self.calls.len() + offset,
                &call.name,
                &arguments,
                &mut deltas,
            )?;
        }
        *streamed = region.calls[..settled]
            .iter()
            .filter(|call| !call.name.is_empty())
            .count();
        if !deltas.is_empty() {
            updates.push(FormatEventV1::ToolCallDeltas(deltas));
        }
        Ok(())
    }

    /// Sends one call up to `arguments`, and returns what the client now holds.
    ///
    /// Only confirmed final arguments reach the client; a changed prefix is
    /// an error rather than a stream-dependent replacement for the parsed call.
    fn stream_call(
        &mut self,
        index: usize,
        name: &str,
        arguments: &str,
        deltas: &mut Vec<FormatToolCallDeltaV1>,
    ) -> Result<String, String> {
        if index == self.streamed_tools.len() {
            deltas.push(FormatToolCallDeltaV1 {
                index,
                name: Some(name.to_owned()),
                arguments: arguments.to_owned(),
            });
            self.streamed_tools.push(ParsedToolCallV1 {
                name: name.to_owned(),
                arguments: arguments.to_owned(),
            });
            return Ok(arguments.to_owned());
        }
        let sent = &mut self.streamed_tools[index].arguments;
        let suffix = match arguments.strip_prefix(sent.as_str()) {
            Some(suffix) => suffix.to_owned(),
            None => {
                return Err(malformed_v1(
                    "streamed tool arguments changed after confirmation",
                ));
            }
        };
        if !suffix.is_empty() {
            sent.push_str(&suffix);
            deltas.push(FormatToolCallDeltaV1 {
                index,
                name: None,
                arguments: suffix,
            });
        }
        Ok(sent.clone())
    }

    /// Finishes every call of a region and delivers them as one group.
    ///
    /// A request that calls no function gets none: its region was markup.
    fn deliver_region(
        &mut self,
        start: usize,
        end: usize,
        region: &ToolRegionV1,
        text: &str,
        updates: &mut Vec<FormatEventV1>,
    ) -> Result<(), String> {
        if !self.extract_tools {
            return Ok(());
        }
        let descriptor = Arc::clone(&self.descriptor);
        let parser = Arc::clone(&self.parser);
        let terminals = ChatMlTerminalsV1::new(&parser);
        let mut calls = Vec::new();
        let mut deltas = Vec::new();
        for (offset, call) in named_calls(region).enumerate() {
            if self.calls.len() + offset >= descriptor.limits.maximum_tool_calls {
                return Err(resource_v1("too many completion tool calls"));
            }
            validate_function_name_v1(&descriptor, &call.name, "parsed function")?;
            let mut arguments = call_arguments_v1(
                &descriptor,
                terminals,
                call,
                text,
                &self.tool_schemas,
                false,
            )?;
            if self.stream_tool_calls {
                arguments = self.stream_call(
                    self.calls.len() + offset,
                    &call.name,
                    &arguments,
                    &mut deltas,
                )?;
            }
            calls.push(ParsedToolCallV1 {
                name: call.name.clone(),
                arguments,
            });
        }
        if !deltas.is_empty() {
            updates.push(FormatEventV1::ToolCallDeltas(deltas));
        }
        if !calls.is_empty() {
            updates.push(FormatEventV1::ToolCalls {
                calls: calls.clone(),
                range: start..end,
            });
            self.calls.extend(calls);
        }
        Ok(())
    }

    fn terminators(&self) -> Vec<&str> {
        if self.ignore_eos {
            return Vec::new();
        }
        self.descriptor
            .vocabulary
            .stop_token_ids
            .iter()
            .filter_map(|id| {
                self.descriptor
                    .vocabulary
                    .boundary_tokens
                    .iter()
                    .find(|token| token.id == *id)
                    .map(|token| token.text.as_str())
            })
            .collect()
    }

    fn next_terminator(&self, valid_end: usize) -> Option<(usize, usize)> {
        let terminators = self.terminators();
        let found = terminators
            .iter()
            .filter_map(|text| {
                self.find_textual_stop(text.as_bytes(), valid_end)
                    .map(|position| (position, text.len()))
            })
            .min_by_key(|(position, _)| *position);
        // With none found, none begins before the last bytes a longer text
        // could complete one in.
        if found.is_none() && self.stop_floor.is_some() && !self.engine_tokens {
            let longest = terminators.iter().map(|text| text.len()).max().unwrap_or(1);
            let from = valid_end.saturating_sub(longest - 1);
            self.terminator_from
                .set(self.terminator_from.get().max(from));
        }
        found
    }

    /// The earliest thing in the pending bytes that is not ordinary text.
    ///
    /// Order decides ties: the marker this state is waiting for is considered
    /// before the ones it forbids, so a boundary that is legitimate here is
    /// never read as a violation of itself.
    fn next_marker(&self, valid_end: usize) -> Option<(usize, MarkerV1, usize)> {
        if self.raw {
            return None;
        }
        let terminals = ChatMlTerminalsV1::new(&self.parser);
        let haystack = &self.bytes[self.cursor..valid_end];
        let mut best: Option<(usize, MarkerV1, usize)> = None;
        let mut consider = |needle: &[u8], marker: MarkerV1| {
            if needle.is_empty() {
                return;
            }
            if let Some(relative) = find_bytes_v1(haystack, needle) {
                let position = self.cursor + relative;
                let marker = if matches!(marker, MarkerV1::Terminator)
                    && self.find_textual_stop(needle, valid_end) != Some(position)
                {
                    MarkerV1::LiteralTerminator
                } else {
                    marker
                };
                let candidate = (position, marker, needle.len());
                if best
                    .as_ref()
                    .is_none_or(|(position, _, _)| candidate.0 < *position)
                {
                    best = Some(candidate);
                }
            }
        };
        match self.state {
            TextStateV1::Reasoning => {
                consider(terminals.reasoning_open.as_bytes(), MarkerV1::ThinkStart);
                consider(terminals.reasoning_close.as_bytes(), MarkerV1::ThinkEnd);
                consider(terminals.tool_open.as_bytes(), MarkerV1::ToolStart);
                for terminator in self.terminators() {
                    consider(terminator.as_bytes(), MarkerV1::Terminator);
                }
            }
            TextStateV1::Content => {
                if self.mode == PromptModeV1::Thinking {
                    consider(terminals.reasoning_close.as_bytes(), MarkerV1::ThinkEnd);
                }
                consider(terminals.tool_open.as_bytes(), MarkerV1::ToolStart);
                consider(self.bare_function_open().as_bytes(), MarkerV1::ToolStart);
                for terminator in self.terminators() {
                    consider(terminator.as_bytes(), MarkerV1::Terminator);
                }
            }
            TextStateV1::Tool | TextStateV1::Complete => {}
        }
        let expected = self.expected_markers();
        for reserved in &self.parser.reserved_markers {
            // A declaration may reserve the very markers this state is waiting
            // for. Those are structure here, not a violation, and the arms
            // above already classified them.
            if expected.contains(&reserved.as_str()) {
                continue;
            }
            // A marker this state is waiting for arrives one token at a time,
            // so a shorter reserved marker inside the part that has landed is
            // not yet a violation: those bytes may still complete the marker.
            // Only the markers legitimate in this state protect anything, so
            // the same text in ordinary content, or in a state that forbids the
            // marker containing it, is still refused.
            if expected.iter().any(|marker| {
                marker.contains(reserved.as_str())
                    && find_bytes_v1(haystack, reserved.as_bytes()).is_some_and(|position| {
                        inside_partial_prefix_v1(haystack, position, marker)
                    })
            }) {
                continue;
            }
            consider(reserved.as_bytes(), MarkerV1::Forbidden);
        }
        best
    }

    /// The function open that begins a call without its tool wrapper in
    /// text, as the reference parsers read one; empty for the typed grammar,
    /// whose calls open only inside the call block.
    fn bare_function_open(&self) -> &str {
        if self.parser.tool_grammar == FrontendToolGrammarV1::XmlTypedParametersV1 {
            ""
        } else {
            self.parser.function_open.trim()
        }
    }

    /// The markers that are structure rather than a violation in this state.
    fn expected_markers(&self) -> Vec<&str> {
        let terminals = ChatMlTerminalsV1::new(&self.parser);
        let mut expected = match self.state {
            TextStateV1::Reasoning => {
                vec![
                    terminals.reasoning_open,
                    terminals.reasoning_close,
                    terminals.tool_open,
                ]
            }
            TextStateV1::Content if self.mode == PromptModeV1::Thinking => {
                vec![
                    terminals.reasoning_close,
                    terminals.tool_open,
                    self.bare_function_open(),
                ]
            }
            TextStateV1::Content => {
                vec![terminals.tool_open, self.bare_function_open()]
            }
            TextStateV1::Tool | TextStateV1::Complete => Vec::new(),
        };
        expected.extend(self.terminators());
        expected
    }

    /// How many trailing bytes could still turn out to start a marker.
    fn longest_marker_prefix(&self, valid_end: usize) -> usize {
        let terminals = ChatMlTerminalsV1::new(&self.parser);
        let haystack = &self.bytes[self.cursor..valid_end];
        let mut longest = 0;
        let mut consider = |needle: &[u8]| {
            let maximum = needle.len().min(haystack.len());
            for length in (1..=maximum).rev() {
                if haystack.ends_with(&needle[..length]) {
                    longest = longest.max(length);
                    break;
                }
            }
        };
        if !self.raw && matches!(self.state, TextStateV1::Reasoning | TextStateV1::Content) {
            if self.state == TextStateV1::Reasoning {
                consider(terminals.reasoning_open.as_bytes());
            }
            if self.mode == PromptModeV1::Thinking {
                consider(terminals.reasoning_close.as_bytes());
            }
            consider(terminals.tool_open.as_bytes());
            if self.state == TextStateV1::Content {
                consider(self.bare_function_open().as_bytes());
            }
            for terminator in self.terminators() {
                consider(terminator.as_bytes());
            }
        }
        longest
    }

    // A buffered tool region may still contain a marker from before the
    // minimum. Only matches completed by later tokens can stop generation.
    fn find_textual_stop(&self, needle: &[u8], valid_end: usize) -> Option<usize> {
        let minimum_end = self.stop_floor?;
        if needle.is_empty() {
            return None;
        }
        // Whole-text parsing still needs suffix matching. On the engine path,
        // a declared stop ID is authoritative, even when ordinary tokens spell
        // the same bytes.
        if self.engine_tokens {
            return None;
        }
        let start = self
            .cursor
            .max(minimum_end.saturating_sub(needle.len() - 1))
            .max(self.terminator_from.get());
        find_bytes_v1(self.bytes.get(start..valid_end)?, needle).map(|relative| start + relative)
    }

    /// Whether the answer so far is only whitespace a declaration holds back
    /// until it learns whether text or a tool call follows.
    fn holds_blank_answer(&self, end: usize) -> bool {
        self.state == TextStateV1::Content
            && self.cursor == self.content_start
            && (self.after_tools || self.parser.drop_blank_tool_call_content)
            && str::from_utf8(&self.bytes[self.cursor..end])
                .is_ok_and(|text| text.trim().is_empty())
    }

    /// Whether held blank text after a tool call ends with the answer, unsent.
    fn drops_blank_answer(&self, end: usize) -> bool {
        self.after_tools && self.holds_blank_answer(end)
    }

    fn text_update(&mut self, start: usize, end: usize) -> Result<FormatEventV1, String> {
        let text = String::from_utf8_lossy(&self.bytes[start..end]).into_owned();
        match self.state {
            TextStateV1::Reasoning => self.reasoning_text.push_str(&text),
            TextStateV1::Content => self.content_text.push_str(&text),
            TextStateV1::Tool | TextStateV1::Complete => {}
        }
        Ok(match self.state {
            TextStateV1::Reasoning => FormatEventV1::Reasoning(start..end),
            TextStateV1::Content => FormatEventV1::Content(start..end),
            TextStateV1::Tool | TextStateV1::Complete => {
                return Err(malformed_v1("structural bytes cannot be emitted as text"));
            }
        })
    }
}

/// Reads one whole completion against the markers a bundle declared.
pub fn parse_completion_v1(
    descriptor: &FormatDescriptorV1,
    text: &str,
    mode: PromptModeV1,
) -> Result<AssistantCompletionV1, String> {
    if text.len() > descriptor.limits.maximum_completion_bytes {
        return Err(resource_v1("completion exceeds the byte limit"));
    }
    // One completion reads the same whole as it does token by token.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        parse_whole_text_v1(descriptor, text, mode)
    }))
    .unwrap_or_else(|_| Err(malformed_v1("decoder panicked")))
}

/// The token ID a whole-completion parse gives the one token that carries
/// the whole text.
const WHOLE_TEXT_TOKEN_V1: u32 = u32::MAX;

/// Reads one whole completion as a host decoder reads one caller token,
/// [`WHOLE_TEXT_TOKEN_V1`], that carries all of it. The token is checked as
/// the decoder checks it, against a declaration that need not have been
/// validated: its ID against the declared boundaries, the token cap, and
/// an end by stop ID. Then the bytes through the last complete character are
/// released, and the end marker or stop ends the completion.
/// The calls the served parser reads from `text`, a completion that ends
/// with a declared stop token after it, with the request's tool schemas
/// `tools`: the calls a client receives for that generation.
pub(crate) fn completion_calls_v1(
    descriptor: &Arc<FormatDescriptorV1>,
    text: &str,
    tools: &[FrontendToolV1],
) -> Result<Vec<ParsedToolCallV1>, String> {
    let vocabulary = &descriptor.vocabulary;
    let stop = vocabulary.stop_token_ids.first().and_then(|id| {
        vocabulary
            .boundary_tokens
            .iter()
            .find(|boundary| boundary.id == *id)
            .map(|boundary| boundary.text.as_str())
    });
    let completion = format!("{text}{}", stop.unwrap_or_default());
    let mut parser = DeclaredParserV1::new(
        Arc::clone(descriptor),
        FormatParserOptionsV1 {
            mode: PromptModeV1::Chat,
            reasoning_open: false,
            extract_tools: true,
            stream_tool_calls: false,
            raw: false,
            ignore_eos: false,
            engine_tokens: false,
            tool_schemas: tools.to_vec(),
        },
    );
    let events = parser.push(FormatInputV1 {
        completion: completion.as_bytes(),
        held: false,
        last: true,
        stop_floor: Some(0),
    })?;
    Ok(events
        .into_iter()
        .flat_map(|event| match event {
            FormatEventV1::ToolCalls { calls, .. } => calls,
            _ => Vec::new(),
        })
        .collect())
}

fn parse_whole_text_v1(
    descriptor: &FormatDescriptorV1,
    text: &str,
    mode: PromptModeV1,
) -> Result<AssistantCompletionV1, String> {
    let mut bytes = text.as_bytes();
    if bytes.is_empty() {
        return Err(malformed_v1("generated token has no decoded bytes"));
    }
    let vocabulary = &descriptor.vocabulary;
    if vocabulary
        .boundary_tokens
        .iter()
        .find(|boundary| boundary.id == WHOLE_TEXT_TOKEN_V1)
        .is_some_and(|boundary| bytes != boundary.text.as_bytes())
    {
        return Err(malformed_v1(
            "special token ID has mismatched decoded bytes",
        ));
    }
    if descriptor.limits.maximum_generated_tokens < 1 {
        return Err(resource_v1(
            "incremental completion exceeds the token limit",
        ));
    }
    // A stop ID ends the completion, and its bytes are not output.
    let stop = vocabulary.stop_token_ids.contains(&WHOLE_TEXT_TOKEN_V1);
    if stop {
        bytes = &[];
    }
    // A host holds a trailing replacement character, as the text so far
    // may still be completing a character.
    let released = if String::from_utf8_lossy(bytes).ends_with('\u{fffd}') {
        0
    } else {
        bytes.len()
    };
    let mut parser = DeclaredParserV1::new(
        Arc::new(descriptor.clone()),
        FormatParserOptionsV1 {
            mode,
            reasoning_open: mode == PromptModeV1::Thinking,
            extract_tools: true,
            stream_tool_calls: false,
            raw: false,
            ignore_eos: false,
            engine_tokens: false,
            tool_schemas: Vec::new(),
        },
    );
    parser
        .push(FormatInputV1 {
            completion: &bytes[..released],
            held: !stop && released < bytes.len(),
            last: stop,
            stop_floor: Some(0),
        })?
        .into_iter()
        .find_map(|event| match event {
            FormatEventV1::Finish { completion, .. } => completion,
            _ => None,
        })
        .ok_or_else(|| malformed_v1("missing completion end token"))
}

fn find_bytes_v1(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Whether a marker occurrence lies inside a trailing partial `opener`.
fn inside_partial_prefix_v1(haystack: &[u8], marker_position: usize, opener: &str) -> bool {
    let opener = opener.as_bytes();
    (0..=marker_position).any(|start| {
        let candidate = &haystack[start..];
        candidate.len() < opener.len()
            && opener.starts_with(candidate)
            && marker_position < start + candidate.len()
    })
}

/// The calls of a region that named a function; an empty name is no call.
fn named_calls(region: &ToolRegionV1) -> impl Iterator<Item = &RegionCallV1> {
    region.calls.iter().filter(|call| !call.name.is_empty())
}

// Map a character boundary in replacement-decoded text back to original bytes.
// Ranges always refer to those original bytes.
fn source_offset(bytes: &[u8], decoded_offset: usize) -> usize {
    let mut source = 0;
    let mut decoded = 0;
    for chunk in bytes.utf8_chunks() {
        if decoded_offset <= decoded + chunk.valid().len() {
            return source + decoded_offset - decoded;
        }
        source += chunk.valid().len() + chunk.invalid().len();
        decoded += chunk.valid().len();
        if !chunk.invalid().is_empty() {
            decoded += '\u{fffd}'.len_utf8();
        }
    }
    source
}
