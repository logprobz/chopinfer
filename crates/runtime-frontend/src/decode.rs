//! Reading one completion back out of the tokens a generation produced.
//!
//! Generation arrives one token at a time, and a client is waiting, so the
//! decoder releases each byte as soon as it is complete UTF-8 and cannot
//! still turn out to begin a client stop. The format parser the bundle
//! declared reads those bytes as reasoning, content and tool calls (the
//! bundle's frontend export, see `native.rs`), and this host layer turns what
//! it reports into text and the tokens that produced it.
//!
//! Nothing a completion carries is trusted. A special token identifier must
//! decode to the text the vocabulary gave it, ordinals must follow each
//! other, and every limit is checked here, whatever the format reads.

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use runtime_server::FrontendToolV1;

use crate::checks::{field_v1, malformed_v1, resource_v1, sequence_v1};
use crate::descriptor::FrontendDescriptorV1;
use crate::format::{FormatEventV1, FormatInputV1, FormatParserOptionsV1, FormatParserV1};
use crate::native::BundleParserSourceV1;

/// Whether a rendered prompt asks the model to reason before it answers.
pub use runtime_bundle_abi::frontend_document::ModeV1 as PromptModeV1;

/// Why one generation ended.
pub use runtime_bundle_abi::frontend_document::FinishReasonV1 as CompletionFinishV1;

/// What one generation may spend before it is refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderLimitsV1 {
    max_completion_bytes: usize,
    max_generated_tokens: usize,
}

impl DecoderLimitsV1 {
    /// Bounds one generation, no wider than the declaration allows.
    pub fn new(
        descriptor: &FrontendDescriptorV1,
        max_completion_bytes: usize,
        max_generated_tokens: usize,
    ) -> Result<Self, String> {
        let declared = descriptor.limits;
        if max_completion_bytes == 0 || max_completion_bytes > declared.maximum_completion_bytes {
            return Err(resource_v1("completion byte limit is invalid"));
        }
        if max_generated_tokens == 0 || max_generated_tokens > declared.maximum_generated_tokens {
            return Err(resource_v1("completion token limit is invalid"));
        }
        Ok(Self {
            max_completion_bytes,
            max_generated_tokens,
        })
    }

    /// Everything the declaration allows.
    #[must_use]
    pub const fn declared(descriptor: &FrontendDescriptorV1) -> Self {
        Self {
            max_completion_bytes: descriptor.limits.maximum_completion_bytes,
            max_generated_tokens: descriptor.limits.maximum_generated_tokens,
        }
    }

    #[must_use]
    pub const fn max_completion_bytes(self) -> usize {
        self.max_completion_bytes
    }

    #[must_use]
    pub const fn max_generated_tokens(self) -> usize {
        self.max_generated_tokens
    }
}

/// One token the engine produced, with the bytes it decoded to.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneratedTokenV1 {
    pub ordinal: u64,
    pub token_id: u32,
    pub bytes: Vec<u8>,
    pub special: bool,
}

impl GeneratedTokenV1 {
    #[must_use]
    pub const fn new(ordinal: u64, token_id: u32, bytes: Vec<u8>) -> Self {
        Self {
            ordinal,
            token_id,
            bytes,
            special: false,
        }
    }

    /// Supplies the special flag from the authenticated tokenizer artifact.
    #[must_use]
    pub const fn with_special(mut self, special: bool) -> Self {
        self.special = special;
        self
    }
}

/// Which bytes of which token one emitted span came from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenContributionV1 {
    pub ordinal: u64,
    pub token_id: u32,
    pub byte_start: usize,
    pub byte_end: usize,
}

/// One span of decoded text and the tokens that produced it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IncrementalTextV1 {
    pub text: String,
    pub contributions: Vec<TokenContributionV1>,
}

impl IncrementalTextV1 {}

/// One tool call read out of a completion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsedToolCallV1 {
    pub name: String,
    /// The arguments, in the JSON encoding the declaration named.
    pub arguments: String,
}

/// One group of tool calls and the tokens that produced it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IncrementalToolCallsV1 {
    pub calls: Vec<ParsedToolCallV1>,
    pub contributions: Vec<TokenContributionV1>,
}

/// One whole assistant answer, after the completion terminated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssistantCompletionV1 {
    pub content: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ParsedToolCallV1>,
}

/// The terminal update, with the parsed answer when there was one.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompletedCompletionV1 {
    pub reason: CompletionFinishV1,
    pub completion: Option<AssistantCompletionV1>,
    pub generated_token_count: usize,
}

/// One thing the decoder learned from the tokens it was given.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompletionUpdateV1 {
    Reasoning(IncrementalTextV1),
    Content(IncrementalTextV1),
    ToolCalls(IncrementalToolCallsV1),
    ToolCallDeltas(Vec<runtime_server::ToolCallDelta>),
    Finished(CompletedCompletionV1),
}

#[derive(Clone, Debug)]
struct TokenRangeV1 {
    ordinal: u64,
    token_id: u32,
    start: usize,
    end: usize,
}

/// Decodes one generation against the format its bundle declared.
///
/// This is the host layer, the same for every model: it checks each token,
/// keeps the completion bytes, releases them as complete UTF-8 with any
/// possible client stop withheld, and turns the byte ranges the format parser
/// reports into text and the tokens that produced it.
pub struct CompletionDecoderV1 {
    descriptor: Arc<FrontendDescriptorV1>,
    mode: PromptModeV1,
    limits: DecoderLimitsV1,
    /// Whether the completion ended, or a panic retired the decoder.
    complete: bool,
    bytes: Vec<u8>,
    stable_end: usize,
    prompt_context_bytes: usize,
    tokens: Vec<TokenRangeV1>,
    next_ordinal: Option<u64>,
    client_stops: Vec<Vec<u8>>,
    minimum_new_tokens: usize,
    /// Whether the first token came through [`Self::push_engine_token`].
    /// Engine tokens carry stop IDs; their ordinary marker spellings are text.
    engine_tokens: Option<bool>,
    /// A configuration change the opened parser cannot see, which refuses
    /// every later push and finish.
    misuse: Option<&'static str>,
    decoding: runtime_server::DecodingOptions,
    raw: bool,
    tool_schemas: Vec<FrontendToolV1>,
    /// Whether the request lets tool calls be read out of the completion.
    extract_tools: bool,
    /// Whether the completion begins in reasoning, as the render that
    /// produced the prompt reported; `None` begins in reasoning exactly in
    /// thinking mode.
    reasoning_open: Option<bool>,
    /// The bundle frontend the parser opens from. Without one a host has
    /// no format to parse with and refuses the completion; a test build
    /// calls the shared parser directly.
    source: Option<BundleParserSourceV1>,
    /// The format parser, opened from the settings above when the first
    /// token or finish reaches it. Its settings are fixed from then on.
    parser: Option<Box<dyn FormatParserV1>>,
}

impl CompletionDecoderV1 {
    #[must_use]
    pub fn new(
        descriptor: Arc<FrontendDescriptorV1>,
        mode: PromptModeV1,
        limits: DecoderLimitsV1,
    ) -> Self {
        Self {
            descriptor,
            mode,
            limits,
            complete: false,
            bytes: Vec::new(),
            stable_end: 0,
            prompt_context_bytes: 0,
            tokens: Vec::new(),
            next_ordinal: None,
            client_stops: Vec::new(),
            minimum_new_tokens: 0,
            engine_tokens: None,
            misuse: None,
            decoding: runtime_server::DecodingOptions::default(),
            raw: false,
            tool_schemas: Vec::new(),
            extract_tools: true,
            reasoning_open: None,
            source: None,
            parser: None,
        }
    }

    /// Opens the parser from a bundle frontend, normally the one that
    /// rendered the prompt. A decoder parses only through one; without it
    /// a host refuses the completion, and a test build calls the declared
    /// format directly.
    ///
    /// Like [`Self::set_decoding_options`], this belongs before the first
    /// token; set later, the next push or finish is refused.
    pub fn set_parser_source(&mut self, source: BundleParserSourceV1) {
        if self.refuse_late("parser source set after decoding began") {
            return;
        }
        self.source = Some(source);
    }

    /// Sets transport decoding controls without changing the declared grammar.
    ///
    /// The format parser reads these when decoding begins, so they belong
    /// before the first token. Set later, they are not applied, and the next
    /// push or finish is refused as a sequence error.
    pub fn set_decoding_options(&mut self, options: runtime_server::DecodingOptions, raw: bool) {
        if self.refuse_late("decoding options set after decoding began") {
            return;
        }
        self.decoding = options;
        self.raw = raw;
    }

    /// Primes byte decoding before generation, without parsing prompt markup.
    /// Retain only an incomplete trailing character, never preceding prompt text.
    pub fn prime_prompt_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        if !self.tokens.is_empty() || self.prompt_context_bytes != 0 {
            return Err(sequence_v1("prompt context must precede generation"));
        }
        if let Some(chunk) = bytes.utf8_chunks().last() {
            let tail = chunk.invalid();
            if std::str::from_utf8(tail).is_err_and(|error| error.error_len().is_none()) {
                self.bytes.extend_from_slice(tail);
                self.prompt_context_bytes = tail.len();
            }
        }
        Ok(())
    }

    /// Suppresses textual stops through the minimum generation length.
    ///
    /// The host applies the minimum to each token, so unlike the other
    /// settings it still takes effect once decoding began.
    pub fn set_minimum_new_tokens(&mut self, minimum: usize) {
        self.minimum_new_tokens = minimum;
    }

    /// Supplies the request's tool schemas, which the shared parser reads
    /// to type tool-call parameters.
    ///
    /// Like [`Self::set_decoding_options`], this belongs before the first
    /// token; set later, the next push or finish is refused.
    pub fn set_tool_schemas(&mut self, schemas: &[FrontendToolV1]) {
        if self.refuse_late("tool schemas set after decoding began") {
            return;
        }
        self.tool_schemas = schemas.to_vec();
    }

    /// Sets whether tool calls are read out of this completion.
    ///
    /// A request that offers no function to call still parses tool-shaped
    /// output, so its markup is not answer text, but delivers no call.
    /// Like [`Self::set_decoding_options`], this belongs before the first
    /// token; set later, the next push or finish is refused.
    pub fn set_tool_extraction(&mut self, extract: bool) {
        if self.refuse_late("tool extraction set after decoding began") {
            return;
        }
        self.extract_tools = extract;
    }

    /// Sets whether the completion begins in reasoning: whether the prompt
    /// left reasoning open, as its render reported
    /// ([`crate::BundleChatStateV1::reasoning_open`]). Unset, it begins in
    /// reasoning exactly in thinking mode.
    ///
    /// Like [`Self::set_decoding_options`], this belongs before the first
    /// token; set later, the next push or finish is refused.
    pub fn set_reasoning_open(&mut self, open: bool) {
        if self.refuse_late("reasoning state set after decoding began") {
            return;
        }
        self.reasoning_open = Some(open);
    }

    /// Records a setter call the opened parser would not see, and reports
    /// whether it is one.
    fn refuse_late(&mut self, misuse: &'static str) -> bool {
        if self.parser.is_none() {
            return false;
        }
        self.misuse.get_or_insert(misuse);
        true
    }

    /// Refuses a decoder whose configuration changed after decoding began.
    fn check_configuration(&self) -> Result<(), String> {
        self.misuse
            .map_or(Ok(()), |misuse| Err(sequence_v1(misuse)))
    }

    /// Stops this generation early on any of the strings a client named.
    pub fn with_client_stops(
        descriptor: Arc<FrontendDescriptorV1>,
        mode: PromptModeV1,
        limits: DecoderLimitsV1,
        stops: &[String],
    ) -> Result<Self, String> {
        let maximum = descriptor.limits.maximum_client_stops;
        if stops.is_empty() || stops.len() > maximum {
            return Err(field_v1(format!(
                "client stops must contain 1 through {maximum} strings"
            )));
        }
        let mut distinct = BTreeSet::new();
        let mut client_stops = Vec::with_capacity(stops.len());
        for stop in stops {
            if stop.is_empty() || !distinct.insert(stop.as_str()) {
                return Err(field_v1(
                    "client stop strings must be nonempty and distinct",
                ));
            }
            client_stops.push(stop.as_bytes().to_vec());
        }
        let mut decoder = Self::new(descriptor, mode, limits);
        decoder.client_stops = client_stops;
        Ok(decoder)
    }

    /// Decodes one token a caller produced.
    ///
    /// A decoder takes all its tokens through this or through
    /// [`Self::push_engine_token`]; a token through the other one is refused
    /// as a sequence error, as is every push and finish after it.
    pub fn push(&mut self, token: GeneratedTokenV1) -> Result<Vec<CompletionUpdateV1>, String> {
        self.guard_decode(|decoder| decoder.push_inner(token, false))
    }

    /// The engine has already applied minimum-token suppression when sampling.
    /// A stop it emits can therefore terminate a completed grammar early.
    /// Ordinary spellings of ID-backed end markers are not implicit stops.
    /// See [`Self::push`] for why a decoder takes one push kind.
    pub fn push_engine_token(
        &mut self,
        token: GeneratedTokenV1,
    ) -> Result<Vec<CompletionUpdateV1>, String> {
        self.guard_decode(|decoder| decoder.push_inner(token, true))
    }

    // A decoder may have partially mutated its buffers before unwinding. Retire
    // it and let the host's existing error path cancel only this request.
    fn guard_decode(
        &mut self,
        decode: impl FnOnce(&mut Self) -> Result<Vec<CompletionUpdateV1>, String>,
    ) -> Result<Vec<CompletionUpdateV1>, String> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| decode(self))) {
            Ok(result) => result,
            Err(_) => {
                self.complete = true;
                Err(malformed_v1("decoder panicked"))
            }
        }
    }

    fn push_inner(
        &mut self,
        mut token: GeneratedTokenV1,
        engine_sampled: bool,
    ) -> Result<Vec<CompletionUpdateV1>, String> {
        if self.complete {
            return Err(malformed_v1("token received after terminal completion"));
        }
        self.check_configuration()?;
        // The parser reads the push kind when it opens, so one decoder takes
        // every token the same way.
        if *self.engine_tokens.get_or_insert(engine_sampled) != engine_sampled {
            self.misuse = Some("engine and caller tokens mixed in one completion");
            self.check_configuration()?;
        }
        if token.bytes.is_empty() {
            return Err(malformed_v1("generated token has no decoded bytes"));
        }
        if let Some(expected) = self.next_ordinal
            && token.ordinal != expected
        {
            return Err(sequence_v1(format!(
                "generated token ordinal {0} does not follow {1}",
                token.ordinal,
                expected.saturating_sub(1)
            )));
        }
        // A boundary the vocabulary named must decode to the text it named it
        // for, or the model emitted a structure under an identifier that does
        // not mean it.
        if let Some(boundary) = self
            .descriptor
            .vocabulary
            .boundary_tokens
            .iter()
            .find(|boundary| boundary.id == token.token_id)
            && token.bytes != boundary.text.as_bytes()
        {
            return Err(malformed_v1(
                "special token ID has mismatched decoded bytes",
            ));
        }
        let next_token_count = self
            .tokens
            .len()
            .checked_add(1)
            .ok_or_else(|| resource_v1("incremental token count overflow"))?;
        if next_token_count > self.limits.max_generated_tokens {
            return Err(resource_v1(
                "incremental completion exceeds the token limit",
            ));
        }
        let eos = self
            .descriptor
            .vocabulary
            .stop_token_ids
            .contains(&token.token_id);
        let token_stop = (engine_sampled || next_token_count > self.minimum_new_tokens)
            && (self.decoding.stop_token_ids.contains(&token.token_id)
                || (eos && !self.decoding.ignore_eos));
        if (token_stop && !self.decoding.include_stop_str_in_output)
            || (token.special && self.decoding.skip_special_tokens)
        {
            token.bytes.clear();
        }
        let end = self
            .bytes
            .len()
            .checked_add(token.bytes.len())
            .ok_or_else(|| resource_v1("incremental completion byte length overflow"))?;
        if end.saturating_sub(self.prompt_context_bytes) > self.limits.max_completion_bytes {
            return Err(resource_v1("incremental completion exceeds the byte limit"));
        }
        let start = self.bytes.len();
        self.bytes.extend_from_slice(&token.bytes);
        self.stable_end += complete_utf8_prefix(&self.bytes[self.stable_end..]);
        self.tokens.push(TokenRangeV1 {
            ordinal: token.ordinal,
            token_id: token.token_id,
            start,
            end,
        });
        self.next_ordinal = Some(
            token
                .ordinal
                .checked_add(1)
                .ok_or_else(|| resource_v1("generated token ordinal overflow"))?,
        );
        if token_stop {
            return self.finish_stop();
        }
        self.advance(false)
    }

    /// Flushes a stop confirmed by the engine, including grammar termination
    /// before the minimum token count. Token accounting is already complete.
    pub fn finish_stop(&mut self) -> Result<Vec<CompletionUpdateV1>, String> {
        let mut updates = self.finish_length()?;
        if let Some(CompletionUpdateV1::Finished(done)) = updates.last_mut()
            && done.reason == CompletionFinishV1::Length
        {
            done.reason = CompletionFinishV1::Stop;
        }
        Ok(updates)
    }

    pub fn finish_length(&mut self) -> Result<Vec<CompletionUpdateV1>, String> {
        self.guard_decode(Self::finish_length_inner)
    }

    fn finish_length_inner(&mut self) -> Result<Vec<CompletionUpdateV1>, String> {
        if self.complete {
            return Err(malformed_v1(
                "length finish received after terminal completion",
            ));
        }
        self.check_configuration()?;
        self.bytes.truncate(self.stable_end);
        self.advance(true)
    }

    fn advance(&mut self, final_bytes: bool) -> Result<Vec<CompletionUpdateV1>, String> {
        let valid_end = self.stable_end;
        let cursor = self.parser()?.cursor();
        // Match and withhold stops before any parser can consume their bytes.
        // This also protects reasoning and streamed tool arguments.
        if let Some((position, length)) = self.next_client_stop(cursor, valid_end) {
            let end = if self.decoding.include_stop_str_in_output {
                position + length
            } else {
                position
            };
            self.bytes.truncate(end);
            self.stable_end = end;
            self.client_stops.clear();
            return self.finish_stop();
        }
        let parse_end = if final_bytes {
            valid_end
        } else {
            valid_end - self.longest_stop_prefix(cursor, valid_end)
        };
        self.parse(parse_end, final_bytes)
    }

    /// The format parser, opened from the request's settings on first use.
    fn parser(&mut self) -> Result<&mut dyn FormatParserV1, String> {
        if self.parser.is_none() {
            let options = FormatParserOptionsV1 {
                mode: self.mode,
                reasoning_open: self
                    .reasoning_open
                    .unwrap_or(self.mode == PromptModeV1::Thinking),
                extract_tools: self.extract_tools,
                stream_tool_calls: self.decoding.stream_tool_calls,
                raw: self.raw,
                ignore_eos: self.decoding.ignore_eos,
                engine_tokens: self.engine_tokens == Some(true),
                tool_schemas: std::mem::take(&mut self.tool_schemas),
            };
            self.parser = Some(match &self.source {
                Some(source) => Box::new(source.open(options)?),
                None => unsourced_parser_v1(&self.descriptor, options)?,
            });
        }
        Ok(self.parser.as_deref_mut().expect("the parser opened above"))
    }

    /// Releases the bytes through `end` to the parser, and reads back what
    /// it settled.
    fn parse(&mut self, end: usize, last: bool) -> Result<Vec<CompletionUpdateV1>, String> {
        // An end marker stops generation only once the minimum token count
        // has passed, and only when it ends after the minimum's last byte.
        let stop_floor = (self.tokens.len() > self.minimum_new_tokens).then(|| {
            self.minimum_new_tokens
                .checked_sub(1)
                .map_or(0, |index| self.tokens[index].end)
        });
        self.parser()?;
        let parser = self.parser.as_mut().expect("opened format parser");
        let events = parser.push(FormatInputV1 {
            completion: &self.bytes[..end],
            held: end < self.bytes.len(),
            last,
            stop_floor,
        })?;
        Ok(events.into_iter().map(|event| self.update(event)).collect())
    }

    /// One parser event as the update it reports.
    fn update(&mut self, event: FormatEventV1) -> CompletionUpdateV1 {
        match event {
            FormatEventV1::Reasoning(range) => CompletionUpdateV1::Reasoning(self.text(range)),
            FormatEventV1::Content(range) => CompletionUpdateV1::Content(self.text(range)),
            FormatEventV1::ToolCalls { calls, range } => {
                CompletionUpdateV1::ToolCalls(IncrementalToolCallsV1 {
                    calls,
                    contributions: self.contributions(range.start, range.end),
                })
            }
            FormatEventV1::ToolCallDeltas(deltas) => CompletionUpdateV1::ToolCallDeltas(
                deltas
                    .into_iter()
                    .map(|delta| runtime_server::ToolCallDelta {
                        index: delta.index,
                        name: delta.name,
                        arguments: delta.arguments,
                    })
                    .collect(),
            ),
            FormatEventV1::Finish { reason, completion } => {
                self.complete = true;
                CompletionUpdateV1::Finished(CompletedCompletionV1 {
                    reason,
                    completion,
                    generated_token_count: self.tokens.len(),
                })
            }
        }
    }

    fn text(&self, range: Range<usize>) -> IncrementalTextV1 {
        IncrementalTextV1 {
            text: String::from_utf8_lossy(&self.bytes[range.clone()]).into_owned(),
            contributions: self.contributions(range.start, range.end),
        }
    }

    fn longest_stop_prefix(&self, cursor: usize, valid_end: usize) -> usize {
        if self.client_stops.is_empty() {
            return 0;
        }
        let haystack = &self.bytes[cursor..valid_end];
        let decoded = String::from_utf8_lossy(haystack);
        let mut longest = 0;
        for stop in &self.client_stops {
            for (start, _) in decoded.char_indices() {
                let suffix = &decoded.as_bytes()[start..];
                // Hold a proper prefix even before the minimum: the match
                // can complete in the first eligible token.
                if suffix.len() < stop.len() && stop.starts_with(suffix) {
                    longest = longest.max(haystack.len() - source_offset(haystack, start));
                    break;
                }
            }
        }
        longest
    }

    fn next_client_stop(&self, cursor: usize, valid_end: usize) -> Option<(usize, usize)> {
        if self.tokens.len() <= self.minimum_new_tokens || self.client_stops.is_empty() {
            return None;
        }
        let minimum_end = self
            .minimum_new_tokens
            .checked_sub(1)
            .map_or(self.prompt_context_bytes, |index| self.tokens[index].end);
        let haystack = &self.bytes[cursor..valid_end];
        let decoded = String::from_utf8_lossy(haystack);
        self.client_stops
            .iter()
            .enumerate()
            .filter_map(|(index, stop)| {
                decoded.char_indices().find_map(|(relative, _)| {
                    if !decoded.as_bytes()[relative..].starts_with(stop) {
                        return None;
                    }
                    let start = cursor + source_offset(haystack, relative);
                    let end = cursor + source_offset(haystack, relative + stop.len());
                    (end > minimum_end).then_some((start, end - start, index))
                })
            })
            .min_by_key(|(position, length, index)| (position + length, *index))
            .map(|(position, length, _)| (position, length))
    }

    fn contributions(&self, start: usize, end: usize) -> Vec<TokenContributionV1> {
        // Only the tokens overlapping the span are visited, so a delta costs
        // its own length rather than the length of the whole completion.
        let first = self.tokens.partition_point(|token| token.end <= start);
        self.tokens[first..]
            .iter()
            .take_while(|token| token.start < end)
            .filter_map(|token| {
                let overlap_start = start.max(token.start);
                let overlap_end = end.min(token.end);
                (overlap_start < overlap_end).then(|| TokenContributionV1 {
                    ordinal: token.ordinal,
                    token_id: token.token_id,
                    byte_start: overlap_start - token.start,
                    byte_end: overlap_end - token.start,
                })
            })
            .collect()
    }
}

// DecodeStream publishes a pending fragment only when it no longer ends in
// replacement text, including literal U+FFFD. Finishing does not flush it.
fn complete_utf8_prefix(bytes: &[u8]) -> usize {
    if String::from_utf8_lossy(bytes).ends_with('�') {
        0
    } else {
        bytes.len()
    }
}

/// The parser of a decoder given no parser source: a test build's declared
/// format, called directly, over a declaration a test may have assembled.
#[cfg(any(test, feature = "test-support"))]
fn unsourced_parser_v1(
    descriptor: &FrontendDescriptorV1,
    options: FormatParserOptionsV1,
) -> Result<Box<dyn FormatParserV1>, String> {
    Ok(Box::new(crate::loopback::LibraryParserV1::new(
        descriptor, options,
    )?))
}

/// A host decoder parses only through a bundle frontend: no format is linked
/// in to parse a completion without one.
#[cfg(not(any(test, feature = "test-support")))]
fn unsourced_parser_v1(
    _descriptor: &FrontendDescriptorV1,
    _options: FormatParserOptionsV1,
) -> Result<Box<dyn FormatParserV1>, String> {
    Err(sequence_v1(
        "the decoder has no bundle frontend to parse the completion with",
    ))
}

// Map a character boundary in replacement-decoded text back to original bytes.
// Contributions always refer to those original bytes.
pub(crate) fn source_offset(bytes: &[u8], decoded_offset: usize) -> usize {
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

#[cfg(test)]
pub(crate) use frontend_format::test_support::fixtures as test_common;

#[cfg(test)]
mod host_tests {
    use super::*;

    /// A format parser with a bug.
    struct PanickingParserV1;

    impl FormatParserV1 for PanickingParserV1 {
        fn push(&mut self, _input: FormatInputV1<'_>) -> Result<Vec<FormatEventV1>, String> {
            panic!("format parser bug")
        }

        fn cursor(&self) -> usize {
            0
        }
    }

    #[test]
    fn decoder_panic_becomes_a_terminal_request_error() {
        let descriptor = Arc::new(
            FrontendDescriptorV1::from_component_bytes(test_common::KEY_VALUE.as_bytes()).unwrap(),
        );
        let mut decoder = CompletionDecoderV1::new(
            Arc::clone(&descriptor),
            PromptModeV1::Chat,
            DecoderLimitsV1::declared(&descriptor),
        );
        decoder.parser = Some(Box::new(PanickingParserV1));
        assert!(
            decoder
                .push_engine_token(GeneratedTokenV1::new(0, 200, b"hello".to_vec()))
                .is_err()
        );
        assert!(decoder.complete);
        assert!(
            decoder
                .push_engine_token(GeneratedTokenV1::new(1, 200, b"again".to_vec()))
                .is_err()
        );
        let mut healthy = CompletionDecoderV1::new(
            Arc::clone(&descriptor),
            PromptModeV1::Chat,
            DecoderLimitsV1::declared(&descriptor),
        );
        assert!(
            healthy
                .push_engine_token(GeneratedTokenV1::new(0, 200, b"hello".to_vec()))
                .is_ok()
        );
        assert!(healthy.finish_length().is_ok());
    }

    fn key_value_decoder() -> CompletionDecoderV1 {
        let descriptor = Arc::new(
            FrontendDescriptorV1::from_component_bytes(test_common::KEY_VALUE.as_bytes()).unwrap(),
        );
        CompletionDecoderV1::new(
            Arc::clone(&descriptor),
            PromptModeV1::Chat,
            DecoderLimitsV1::declared(&descriptor),
        )
    }

    /// A thinking completion begins in reasoning unless the render that
    /// produced its prompt reported that the prompt already closed it.
    #[test]
    fn a_thinking_completion_begins_where_its_render_left_reasoning() {
        let descriptor = Arc::new(
            FrontendDescriptorV1::from_component_bytes(test_common::KEY_VALUE.as_bytes()).unwrap(),
        );
        for (open, reasoning) in [(None, true), (Some(true), true), (Some(false), false)] {
            let mut decoder = CompletionDecoderV1::new(
                Arc::clone(&descriptor),
                PromptModeV1::Thinking,
                DecoderLimitsV1::declared(&descriptor),
            );
            if let Some(open) = open {
                decoder.set_reasoning_open(open);
            }
            let mut updates = decoder
                .push(GeneratedTokenV1::new(0, 200, b"hello".to_vec()))
                .unwrap();
            updates.extend(decoder.finish_length().unwrap());
            let texts: Vec<_> = updates
                .iter()
                .filter_map(|update| match update {
                    CompletionUpdateV1::Reasoning(text) => Some((true, text.text.as_str())),
                    CompletionUpdateV1::Content(text) => Some((false, text.text.as_str())),
                    _ => None,
                })
                .collect();
            assert_eq!(texts, [(reasoning, "hello")], "{open:?}");
        }
    }

    #[test]
    fn a_setting_changed_after_decoding_began_is_refused() {
        type Late = fn(&mut CompletionDecoderV1);
        let late: [(Late, &str); 4] = [
            (
                |decoder| {
                    decoder.set_decoding_options(
                        runtime_server::DecodingOptions {
                            ignore_eos: true,
                            ..Default::default()
                        },
                        false,
                    );
                },
                "decoding options set after decoding began",
            ),
            (
                |decoder| decoder.set_tool_schemas(&[]),
                "tool schemas set after decoding began",
            ),
            (
                |decoder| decoder.set_tool_extraction(false),
                "tool extraction set after decoding began",
            ),
            (
                |decoder| decoder.set_reasoning_open(false),
                "reasoning state set after decoding began",
            ),
        ];
        for (set, message) in late {
            let mut decoder = key_value_decoder();
            // Settings made before the first token are the parser's own.
            set(&mut decoder);
            decoder
                .push(GeneratedTokenV1::new(0, 200, b"hello".to_vec()))
                .unwrap();
            set(&mut decoder);
            let refusal = sequence_v1(message);
            assert_eq!(
                decoder.push(GeneratedTokenV1::new(1, 200, b" there".to_vec())),
                Err(refusal.clone())
            );
            assert_eq!(decoder.finish_length(), Err(refusal.clone()));
            assert_eq!(decoder.finish_stop(), Err(refusal));
        }
        // The minimum is applied by the host to each token, so it may change.
        let mut decoder = key_value_decoder();
        decoder
            .push(GeneratedTokenV1::new(0, 200, b"hello".to_vec()))
            .unwrap();
        decoder.set_minimum_new_tokens(4);
        assert!(
            decoder
                .push(GeneratedTokenV1::new(1, 200, b" there".to_vec()))
                .is_ok()
        );
        assert!(decoder.finish_length().is_ok());
    }

    #[test]
    fn a_decoder_takes_one_push_kind() {
        let refusal = sequence_v1("engine and caller tokens mixed in one completion");
        let mut caller = key_value_decoder();
        caller
            .push(GeneratedTokenV1::new(0, 200, b"hello".to_vec()))
            .unwrap();
        assert_eq!(
            caller.push_engine_token(GeneratedTokenV1::new(1, 200, b" there".to_vec())),
            Err(refusal.clone())
        );
        assert_eq!(
            caller.push(GeneratedTokenV1::new(1, 200, b" there".to_vec())),
            Err(refusal.clone())
        );
        assert_eq!(caller.finish_length(), Err(refusal.clone()));

        let mut engine = key_value_decoder();
        engine
            .push_engine_token(GeneratedTokenV1::new(0, 200, b"hello".to_vec()))
            .unwrap();
        assert_eq!(
            engine.push(GeneratedTokenV1::new(1, 200, b" there".to_vec())),
            Err(refusal.clone())
        );
        assert_eq!(engine.finish_stop(), Err(refusal));
    }
}
