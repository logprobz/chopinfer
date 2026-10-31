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
//!
//! A declaration may name ID-only markers (`vocabulary.id_only_markers`),
//! and the host then marks the bytes of the tokens generated as them. Every
//! marker the parser reads, in text and in tool regions, is then read as the
//! tokens it is made of ([`crate::marks`]), so a marker spelled by ordinary
//! tokens is text.

use std::borrow::Cow;
use std::ops::Range;
use std::str;
use std::sync::Arc;

use crate::chat::PromptModeV1;
use crate::chatml_decode::{
    ChatMlTerminalsV1, RegionCallV1, RegionMarksV1, RegionScanV1, ToolRegionV1, call_arguments_v1,
};
use crate::descriptor::{FormatDescriptorV1, FrontendParserDeclarationV1, FrontendToolGrammarV1};
use crate::format::{
    AssistantCompletionV1, CompletionFinishV1, FormatEventV1, FormatInputV1, FormatParserOptionsV1,
    FormatToolCallDeltaV1, ParsedToolCallV1,
};
use crate::marks::{MarkMatchV1, SourceMapV1};
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
    /// How `text` maps back to the bytes, when it is not them unchanged.
    map: Option<SourceMapV1>,
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
            map: None,
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
                self.map = Some(SourceMapV1::new(&bytes[self.start..end]));
            }
        }
        self.end = end;
    }
}

/// A tool region's decoded text, with where the region begins in the
/// completion bytes and how the text maps back to them when it is not those
/// bytes unchanged.
#[derive(Clone, Copy)]
struct RegionTextV1<'a> {
    start: usize,
    text: &'a str,
    map: Option<&'a SourceMapV1>,
}

impl RegionTextV1<'_> {
    /// The markers `marks` lets through in this text.
    fn marks<'a>(&'a self, marks: MarkMatchV1<'a>) -> RegionMarksV1<'a> {
        RegionMarksV1::new(marks, self.start, self.map)
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
    /// Whether pushes carry marks; the first push decides.
    marked: Option<bool>,
    /// The marks among `bytes`, ascending: the byte ranges of the tokens
    /// generated as the declaration's ID-only markers.
    marks: Vec<Range<usize>>,
}

/// What the marks of a parser let through: every marker by its spelling
/// unless its pushes carry marks.
fn mark_match_v1<'a>(
    descriptor: &'a FormatDescriptorV1,
    marked: Option<bool>,
    marks: &'a [Range<usize>],
) -> MarkMatchV1<'a> {
    if marked == Some(true) {
        MarkMatchV1::marked(&descriptor.vocabulary.id_only_markers, marks)
    } else {
        MarkMatchV1::SPELLED
    }
}

impl DeclaredParserV1 {
    /// Reads the bytes `input` adds, and returns what they settled.
    pub fn push(&mut self, input: FormatInputV1<'_>) -> Result<Vec<FormatEventV1>, String> {
        // The host only appends, except that a client stop may cut the
        // completion short, so this copy follows it with the bytes it adds.
        let keep = self.bytes.len().min(input.completion.len());
        let marks = input
            .marks
            .map(|marks| &marks[marks.partition_point(|mark| mark.end <= keep)..]);
        self.push_suffix(
            keep,
            &input.completion[keep..],
            input.held,
            input.last,
            input.stop_floor,
            marks,
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
    ///
    /// `marks` are the push's marks ([`FormatInputV1::marks`]): each ends
    /// after `keep` and within the bytes `suffix` adds, and may start before
    /// `keep`, so a token that an earlier push released in part crosses
    /// again whole. The parser keeps the marks that end within the kept
    /// bytes and drops the rest. A push without marks after one with them,
    /// or the reverse, and a marked push to a parser whose declaration names
    /// no ID-only markers, are refused as out of sequence.
    pub fn push_suffix(
        &mut self,
        keep: usize,
        suffix: &[u8],
        held: bool,
        last: bool,
        stop_floor: Option<usize>,
        marks: Option<&[Range<usize>]>,
    ) -> Result<Vec<FormatEventV1>, String> {
        if self.state == TextStateV1::Complete {
            return Err(malformed_v1("bytes received after terminal completion"));
        }
        if keep > self.bytes.len() {
            return Err(sequence_v1("a push keeps bytes the parser never received"));
        }
        let reread = self.take_marks(keep, keep + suffix.len(), marks)?;
        let changed = (keep < self.bytes.len()).then_some(keep);
        if let Some(changed) = changed.into_iter().chain(reread).min() {
            let longest = self.terminators().iter().map(|text| text.len()).max();
            let from = changed.saturating_sub(longest.unwrap_or(1).saturating_sub(1));
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

    /// Keeps the marks that end within the first `keep` bytes and adds
    /// `marks`, after checking them against the `released` bytes the push
    /// leaves. Returns where the marks of bytes the parser already read
    /// changed, if they did; the tool region is then read again.
    fn take_marks(
        &mut self,
        keep: usize,
        released: usize,
        marks: Option<&[Range<usize>]>,
    ) -> Result<Option<usize>, String> {
        let marked = marks.is_some();
        if marked && self.descriptor.vocabulary.id_only_markers.is_empty() {
            return Err(sequence_v1(
                "a push carries marks, and the declaration names no ID-only markers",
            ));
        }
        match self.marked {
            Some(true) if !marked => {
                return Err(sequence_v1("a push without marks follows one with marks"));
            }
            Some(false) if marked => {
                return Err(sequence_v1("a push with marks follows one without"));
            }
            _ => {}
        }
        let Some(marks) = marks else {
            self.marked = Some(false);
            return Ok(None);
        };
        let kept = self.marks.partition_point(|mark| mark.end <= keep);
        let mut previous = kept.checked_sub(1).map_or(0, |last| self.marks[last].end);
        for mark in marks {
            let Range { start, end } = *mark;
            if start >= end || start < previous || end <= keep || end > released {
                return Err(sequence_v1(format!(
                    "mark {start}..{end} is empty, overlaps or precedes a mark, ends within \
                     the {keep} kept bytes, or ends past the {released} released bytes"
                )));
            }
            previous = end;
        }
        self.marked = Some(true);
        let reread = self.marks[kept..]
            .first()
            .into_iter()
            .chain(marks.first().filter(|mark| mark.start < keep))
            .map(|mark| mark.start)
            .min();
        self.marks.truncate(kept);
        self.marks
            .try_reserve(marks.len())
            .map_err(|_| resource_v1("completion marks allocation failed"))?;
        self.marks.extend_from_slice(marks);
        if reread.is_some() {
            self.region = None;
        }
        Ok(reread)
    }

    /// What this parser's marks let through as markers.
    fn mark_match(&self) -> MarkMatchV1<'_> {
        mark_match_v1(&self.descriptor, self.marked, &self.marks)
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
            marked: None,
            marks: Vec::new(),
        }
    }

    /// Where held text that a generation ended in begins an unfinished call
    /// opening, for the typed grammar: the longest tail of the pending text
    /// that is a proper prefix of the tool call open and already spells a
    /// reserved marker, which text may not carry, with the newlines before
    /// it. The vendor's stream parser has entered its calls by then and
    /// drops the unfinished block, so it is not text; a shorter tail is.
    fn unfinished_open(&self, end: usize) -> Option<usize> {
        if self.parser.tool_grammar != FrontendToolGrammarV1::XmlTypedParametersV1 {
            return None;
        }
        let open = ChatMlTerminalsV1::new(&self.parser).tool_open;
        let pending = &self.bytes[self.cursor..end];
        let marks = self.mark_match();
        (1..open.len().min(pending.len() + 1))
            .rev()
            .find(|&length| {
                let start = end - length;
                pending.ends_with(&open.as_bytes()[..length])
                    && marks.admits(start, open, length)
                    && self.parser.reserved_markers.iter().any(|marker| {
                        find_marker_v1(&self.bytes, start, end, marker, marks).is_some()
                    })
            })
            .map(|length| self.open_lead(end - length))
    }

    /// Where a calls open at `position` begins with the newlines it takes:
    /// in the typed grammar, the run of newlines before it in the pending
    /// text, which the open drops as the grammar's reference stream parser
    /// does.
    /// The other grammars trim the whitespace around their open themselves.
    fn open_lead(&self, position: usize) -> usize {
        if self.parser.tool_grammar != FrontendToolGrammarV1::XmlTypedParametersV1 {
            return position;
        }
        position
            - self.bytes[self.cursor..position]
                .iter()
                .rev()
                .take_while(|byte| **byte == b'\n')
                .count()
    }

    /// Reads the last bytes, and ends the completion where generation ended:
    /// at the token budget, after or inside a call too, whose complete
    /// parameters are delivered. A host that stopped generation there reports
    /// the stop instead.
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
            reason: CompletionFinishV1::Length,
            completion: None,
        });
        Ok(updates)
    }

    /// Whether a completion end at `end` is the last byte of the completion.
    fn ends_completion(&self, end: usize) -> bool {
        !self.held && end == self.bytes.len()
    }

    /// Why a generation that an end marker stopped ended: after a delivered
    /// call, because of it.
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
        let text = RegionTextV1 {
            start,
            text: &open.text,
            map: open.map.as_ref(),
        };
        open.scan.scan(
            ChatMlTerminalsV1::new(&parser),
            text.text,
            &self.tool_schemas,
            text.marks(self.mark_match()),
        );
        let region = open.scan.region();
        if let Some(used) = region.end {
            let used = source_offset(&self.bytes[start..limit], used);
            self.deliver_region(text, start + used, region, updates)?;
            self.cursor = start + used;
            self.content_start = self.cursor;
            self.after_tools = self.extract_tools;
            self.tool_start = None;
            self.state = TextStateV1::Content;
            return Ok(true);
        }
        if self.stream_tool_calls {
            let settled = open.scan.settled_calls();
            self.stream_region(region, text, settled, &mut open.streamed, updates)?;
        }
        let Some((terminator_start, terminator_length)) = end else {
            self.region = Some(open);
            return Ok(false);
        };
        let terminator_end = terminator_start + terminator_length;
        if !self.ends_completion(terminator_end) {
            return Err(malformed_v1("trailing bytes after the completion end"));
        }
        self.deliver_region(text, terminator_start, region, updates)?;
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
        let bytes = &self.bytes[start..end];
        let decoded = String::from_utf8_lossy(bytes);
        let map = matches!(decoded, Cow::Owned(_)).then(|| SourceMapV1::new(bytes));
        let text = decoded.into_owned();
        let parser = Arc::clone(&self.parser);
        let text = RegionTextV1 {
            start,
            text: &text,
            map: map.as_ref(),
        };
        let region = crate::chatml_decode::scan_tool_region(
            ChatMlTerminalsV1::new(&parser),
            text.text,
            &self.tool_schemas,
            text.marks(self.mark_match()),
        );
        self.region = None;
        self.deliver_region(text, end, &region, updates)
    }

    /// Streams the calls of an open region as far as they can no longer change.
    ///
    /// The first `settled` calls stay as they are whatever text follows, so
    /// the named ones among the first `streamed`, already sent whole, would
    /// send nothing again and are skipped.
    fn stream_region(
        &mut self,
        region: &ToolRegionV1,
        text: RegionTextV1<'_>,
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
        let marks = text.marks(mark_match_v1(&descriptor, self.marked, &self.marks));
        let mut deltas = Vec::new();
        for (offset, call) in named_calls(region).enumerate().skip(*streamed) {
            validate_function_name_v1(&descriptor, &call.name, "parsed function")?;
            let arguments = call_arguments_v1(
                &descriptor,
                terminals,
                call,
                text.text,
                &self.tool_schemas,
                !call.closed,
                marks,
            )?;
            Self::stream_call(
                &mut self.streamed_tools,
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
        streamed_tools: &mut Vec<ParsedToolCallV1>,
        index: usize,
        name: &str,
        arguments: &str,
        deltas: &mut Vec<FormatToolCallDeltaV1>,
    ) -> Result<String, String> {
        if index == streamed_tools.len() {
            deltas.push(FormatToolCallDeltaV1 {
                index,
                name: Some(name.to_owned()),
                arguments: arguments.to_owned(),
            });
            streamed_tools.push(ParsedToolCallV1 {
                name: name.to_owned(),
                arguments: arguments.to_owned(),
            });
            return Ok(arguments.to_owned());
        }
        let sent = &mut streamed_tools[index].arguments;
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
        text: RegionTextV1<'_>,
        end: usize,
        region: &ToolRegionV1,
        updates: &mut Vec<FormatEventV1>,
    ) -> Result<(), String> {
        if !self.extract_tools {
            return Ok(());
        }
        let descriptor = Arc::clone(&self.descriptor);
        let parser = Arc::clone(&self.parser);
        let terminals = ChatMlTerminalsV1::new(&parser);
        let marks = text.marks(mark_match_v1(&descriptor, self.marked, &self.marks));
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
                text.text,
                &self.tool_schemas,
                false,
                marks,
            )?;
            if self.stream_tool_calls {
                arguments = Self::stream_call(
                    &mut self.streamed_tools,
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
                range: text.start..end,
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
                self.find_textual_stop(text, valid_end)
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
    /// never read as a violation of itself. A marker spelling the marks
    /// reject is ordinary text, and the search goes on after it.
    fn next_marker(&self, valid_end: usize) -> Option<(usize, MarkerV1, usize)> {
        if self.raw {
            return None;
        }
        let terminals = ChatMlTerminalsV1::new(&self.parser);
        let marks = self.mark_match();
        let find =
            |needle: &str| find_marker_v1(&self.bytes, self.cursor, valid_end, needle, marks);
        let mut best: Option<(usize, MarkerV1, usize)> = None;
        let mut consider = |needle: &str, marker: MarkerV1| {
            if needle.is_empty() {
                return;
            }
            if let Some(position) = find(needle) {
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
                consider(terminals.reasoning_open, MarkerV1::ThinkStart);
                consider(terminals.reasoning_close, MarkerV1::ThinkEnd);
                consider(terminals.tool_open, MarkerV1::ToolStart);
                for terminator in self.terminators() {
                    consider(terminator, MarkerV1::Terminator);
                }
            }
            TextStateV1::Content => {
                if self.mode == PromptModeV1::Thinking {
                    consider(terminals.reasoning_close, MarkerV1::ThinkEnd);
                }
                consider(terminals.tool_open, MarkerV1::ToolStart);
                consider(self.bare_function_open(), MarkerV1::ToolStart);
                for terminator in self.terminators() {
                    consider(terminator, MarkerV1::Terminator);
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
                    && find(reserved).is_some_and(|position| {
                        self.inside_partial_prefix(position, marker, valid_end)
                    })
            }) {
                continue;
            }
            consider(reserved, MarkerV1::Forbidden);
        }
        // A calls open takes the newlines before it.
        best.map(|(position, marker, length)| match marker {
            MarkerV1::ToolStart => {
                let start = self.open_lead(position);
                (start, marker, length + position - start)
            }
            _ => (position, marker, length),
        })
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

    /// Whether a marker occurrence at `position` lies inside a trailing
    /// partial `opener` that the marks let through so far.
    fn inside_partial_prefix(&self, position: usize, opener: &str, valid_end: usize) -> bool {
        let marks = self.mark_match();
        (self.cursor..=position).any(|start| {
            let candidate = &self.bytes[start..valid_end];
            candidate.len() < opener.len()
                && opener.as_bytes().starts_with(candidate)
                && marks.admits(start, opener, candidate.len())
        })
    }

    /// How many trailing bytes could still turn out to start a marker,
    /// including the newlines a calls open would take. The bytes of a
    /// marker the marks already rule out are not held.
    fn longest_marker_prefix(&self, valid_end: usize) -> usize {
        let terminals = ChatMlTerminalsV1::new(&self.parser);
        let haystack = &self.bytes[self.cursor..valid_end];
        let marks = self.mark_match();
        let mut longest = 0;
        let mut consider = |needle: &str| {
            let maximum = needle.len().min(haystack.len());
            let length = (1..=maximum)
                .rev()
                .find(|&length| {
                    haystack.ends_with(&needle.as_bytes()[..length])
                        && marks.admits(valid_end - length, needle, length)
                })
                .unwrap_or_default();
            longest = longest.max(length);
            length
        };
        let mut open = 0;
        if !self.raw && matches!(self.state, TextStateV1::Reasoning | TextStateV1::Content) {
            if self.state == TextStateV1::Reasoning {
                consider(terminals.reasoning_open);
            }
            if self.mode == PromptModeV1::Thinking {
                consider(terminals.reasoning_close);
            }
            // A calls open takes the newlines before it, so they are held
            // with any start of it.
            let prefix = consider(terminals.tool_open);
            open = valid_end - self.open_lead(valid_end - prefix);
            if self.state == TextStateV1::Content {
                consider(self.bare_function_open());
            }
            for terminator in self.terminators() {
                consider(terminator);
            }
        }
        longest.max(open)
    }

    // A buffered tool region may still contain a marker from before the
    // minimum. Only matches completed by later tokens can stop generation.
    fn find_textual_stop(&self, needle: &str, valid_end: usize) -> Option<usize> {
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
        find_marker_v1(&self.bytes, start, valid_end, needle, self.mark_match())
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

/// Reads one whole completion against the markers a bundle declared, as
/// text without token boundaries: every marker by its spelling.
pub fn parse_completion_v1(
    descriptor: &FormatDescriptorV1,
    text: &str,
    mode: PromptModeV1,
) -> Result<AssistantCompletionV1, String> {
    parse_whole_v1(descriptor, text, None, mode)
}

/// [`parse_completion_v1`] for a completion whose `marks` are the byte
/// ranges of the tokens generated as the declaration's ID-only markers
/// (`vocabulary.id_only_markers`), ascending, disjoint and not empty
/// ([`FormatInputV1::marks`]), so a marker spelled by ordinary tokens is
/// text. A declaration that names no ID-only markers refuses marks.
pub fn parse_marked_completion_v1(
    descriptor: &FormatDescriptorV1,
    text: &str,
    marks: &[Range<usize>],
    mode: PromptModeV1,
) -> Result<AssistantCompletionV1, String> {
    parse_whole_v1(descriptor, text, Some(marks), mode)
}

fn parse_whole_v1(
    descriptor: &FormatDescriptorV1,
    text: &str,
    marks: Option<&[Range<usize>]>,
    mode: PromptModeV1,
) -> Result<AssistantCompletionV1, String> {
    if text.len() > descriptor.limits.maximum_completion_bytes {
        return Err(resource_v1("completion exceeds the byte limit"));
    }
    // One completion reads the same whole as it does token by token.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        parse_whole_text_v1(descriptor, text, marks, mode)
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
        marks: None,
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
    marks: Option<&[Range<usize>]>,
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
    // The marks of held bytes are held with them.
    let marks = marks.map(|marks| &marks[..marks.partition_point(|mark| mark.end <= released)]);
    parser
        .push(FormatInputV1 {
            completion: &bytes[..released],
            held: !stop && released < bytes.len(),
            last: stop,
            stop_floor: Some(0),
            marks,
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

/// The first occurrence of `marker` in `bytes[from..end]` that `marks` lets
/// through, as a completion offset.
fn find_marker_v1(
    bytes: &[u8],
    from: usize,
    end: usize,
    marker: &str,
    marks: MarkMatchV1<'_>,
) -> Option<usize> {
    let mut start = from;
    loop {
        let position = start + find_bytes_v1(bytes.get(start..end)?, marker.as_bytes())?;
        if marks.accepts(position, marker) {
            return Some(position);
        }
        start = position + 1;
    }
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
