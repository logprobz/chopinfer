#![forbid(unsafe_code)]

//! Reading a completion back is the declaration's job too.
//!
//! These are the contract tests the model crate's incremental decoder carried,
//! rewritten against declared markers. They run on the parser fixtures, one on
//! each shared tool grammar, and on one whose delimiters all contain a shorter
//! reserved marker, so what they pin is the decoder rather than one model's
//! punctuation: streaming a completion in arbitrary
//! byte splits must produce the same answer as parsing it whole, a boundary
//! must never leak into visible text, and a structure that does not close must
//! never be reported as one that did.

use std::sync::Arc;

use frontend_format::test_support::fixtures::{FUNCTION_PARAMETERS, KEY_VALUE};
use frontend_format::{FormatDescriptorV1, FrontendParserDeclarationV1, FrontendToolGrammarV1};
use runtime_frontend::{
    CompletionDecoderV1, CompletionFinishV1, CompletionUpdateV1, DecoderLimitsV1,
    FrontendDescriptorV1, GeneratedTokenV1, PromptModeV1, parse_completion_v1,
};
use serde_json::json;

/// One declaration, and the completion markers a test writes with.
struct FixtureV1 {
    name: &'static str,
    descriptor: Arc<FrontendDescriptorV1>,
    parser: FrontendParserDeclarationV1,
}

impl FixtureV1 {
    fn new(name: &'static str, declaration: &str) -> Self {
        let format = FormatDescriptorV1::from_component_bytes(declaration.as_bytes())
            .expect("the fixture format is readable");
        Self {
            name,
            descriptor: Arc::new(
                FrontendDescriptorV1::from_component_bytes(declaration.as_bytes())
                    .expect("the fixture declaration is readable"),
            ),
            parser: format.parser,
        }
    }

    fn reserved(&self) -> &[String] {
        &self.parser.reserved_markers
    }

    /// The text of the declaration's stop token, which ends a completion.
    fn end(&self) -> &str {
        let stop = self.descriptor.vocabulary.stop_token_ids[0];
        &self
            .descriptor
            .vocabulary
            .boundary_tokens
            .iter()
            .find(|token| token.id == stop)
            .expect("the stop token has a boundary text")
            .text
    }

    fn thinking_close(&self) -> &str {
        &self.parser.reasoning_close
    }

    fn tool_open(&self) -> &str {
        &self.parser.tool_call_open
    }

    /// What opens a call to `name`, before its first argument.
    fn call_open(&self, name: &str) -> String {
        let parser = &self.parser;
        match parser.tool_grammar {
            FrontendToolGrammarV1::XmlKeyValueArgumentsV1 => {
                format!("{}{name}", parser.tool_call_open)
            }
            FrontendToolGrammarV1::XmlFunctionParametersV1
            | FrontendToolGrammarV1::XmlTypedParametersV1 => format!(
                "{}{}{name}{}",
                parser.tool_call_open, parser.function_open, parser.function_name_close
            ),
        }
    }

    /// One argument of a call.
    fn argument(&self, parameter: &str, value: &str) -> String {
        let parser = &self.parser;
        format!(
            "{}{parameter}{}{}{value}{}",
            parser.parameter_open,
            parser.parameter_name_close,
            parser.parameter_value_open,
            parser.parameter_close
        )
    }

    /// What closes a call after its last argument.
    fn call_close(&self) -> String {
        let parser = &self.parser;
        match parser.tool_grammar {
            FrontendToolGrammarV1::XmlKeyValueArgumentsV1 => parser.tool_call_close.clone(),
            FrontendToolGrammarV1::XmlFunctionParametersV1
            | FrontendToolGrammarV1::XmlTypedParametersV1 => {
                format!("{}{}", parser.function_close, parser.tool_call_close)
            }
        }
    }

    /// One tool call with one argument, in the declaration's tool grammar.
    fn tool_block(&self, name: &str, parameter: &str, value: &str) -> String {
        format!(
            "{}{}{}",
            self.call_open(name),
            self.argument(parameter, value),
            self.call_close()
        )
    }

    fn decoder(&self, mode: PromptModeV1) -> CompletionDecoderV1 {
        CompletionDecoderV1::new(
            Arc::clone(&self.descriptor),
            mode,
            DecoderLimitsV1::declared(&self.descriptor),
        )
    }

    fn with_stops(&self, stops: &[&str]) -> CompletionDecoderV1 {
        let stops = stops
            .iter()
            .map(|stop| (*stop).to_owned())
            .collect::<Vec<_>>();
        CompletionDecoderV1::with_client_stops(
            Arc::clone(&self.descriptor),
            PromptModeV1::Chat,
            DecoderLimitsV1::declared(&self.descriptor),
            &stops,
        )
        .expect("the fixture stops are within the declared bound")
    }
}

/// A key-value declaration whose every structural delimiter contains one
/// shorter reserved marker, `<|`.
///
/// That shape is legal, and it is the one a streaming decoder can get wrong:
/// the short marker lands before the delimiter that contains it, so a decoder
/// that judged it on sight would refuse a completion that parses whole. It has
/// to hold, and it still has to refuse `<|` wherever no delimiter this state
/// waits for could grow out of it.
fn nested() -> String {
    let mut declaration: serde_json::Value = serde_json::from_str(FUNCTION_PARAMETERS).unwrap();
    declaration["vocabulary"]["boundary_tokens"] = json!([
        {"text": "<|s|>", "id": 0},
        {"text": "<|end|>", "id": 1},
        {"text": "<|/think|>", "id": 2},
        {"text": "<|calls|>", "id": 3},
        {"text": "<|u|>", "id": 4},
        {"text": "<|a|>", "id": 5}
    ]);
    declaration["vocabulary"]["prompt_special_tokens"] = json!([
        "<|s|>",
        "<|end|>",
        "<|/think|>",
        "<|calls|>",
        "<|u|>",
        "<|a|>"
    ]);
    declaration["tokenizer"]["eos_token"] = json!("<|end|>");
    declaration["tokenizer"]["pad_token"] = json!("<|end|>");
    declaration["reasoning"]["end_markers"] = json!(["<|/think|>", "<|calls|>"]);
    declaration["format"] = json!({"parser": {
        "tool_grammar": "xml-key-value-arguments-v1",
        "json_encoding": "python-json-v1",
        "reasoning_open": "<|think|>",
        "reasoning_close": "<|/think|>",
        "tool_call_open": "<|calls|>",
        "tool_call_close": "<|/calls|>",
        "parameter_open": "<|k|>",
        "parameter_name_close": "<|/k|>",
        "parameter_value_open": "<|v|>",
        "parameter_close": "<|/v|>",
        "reserved_markers": ["<|"],
        "turn_open": ["<|a|>", "<|u|>"],
        "assistant_header": "<|a|>"
    }});
    declaration.to_string()
}

fn fixtures() -> Vec<FixtureV1> {
    vec![
        FixtureV1::new("key-value", KEY_VALUE),
        FixtureV1::new("function-parameters", FUNCTION_PARAMETERS),
        FixtureV1::new("nested", &nested()),
    ]
}

fn feed(decoder: &mut CompletionDecoderV1, parts: &[&[u8]]) -> Vec<CompletionUpdateV1> {
    let mut updates = Vec::new();
    for (ordinal, bytes) in parts.iter().filter(|bytes| !bytes.is_empty()).enumerate() {
        let next = decoder
            .push(GeneratedTokenV1::new(ordinal as u64, 200, bytes.to_vec()))
            .expect("the fixture tokens decode");
        let finished = next
            .iter()
            .any(|update| matches!(update, CompletionUpdateV1::Finished(_)));
        updates.extend(next);
        if finished {
            break;
        }
    }
    updates
}

fn text(updates: &[CompletionUpdateV1]) -> (String, String) {
    let mut reasoning = String::new();
    let mut content = String::new();
    for update in updates {
        match update {
            CompletionUpdateV1::Reasoning(part) => reasoning.push_str(&part.text),
            CompletionUpdateV1::Content(part) => content.push_str(&part.text),
            _ => {}
        }
    }
    (reasoning, content)
}

fn finished(updates: &[CompletionUpdateV1]) -> Option<&runtime_frontend::CompletedCompletionV1> {
    updates.iter().find_map(|update| match update {
        CompletionUpdateV1::Finished(done) => Some(done),
        _ => None,
    })
}

/// A token split anywhere inside a scalar still yields the whole text once.
#[test]
fn every_utf8_byte_split_is_lossless_and_owned() {
    for fixture in fixtures() {
        let source = "Aé水🦀Z";
        for split in 0..=source.len() {
            let updates = feed(
                &mut fixture.decoder(PromptModeV1::Chat),
                &[
                    &source.as_bytes()[..split],
                    &source.as_bytes()[split..],
                    fixture.end().as_bytes(),
                ],
            );
            assert_eq!(text(&updates).1, source, "{}: split {split}", fixture.name);
            let contributions = updates
                .iter()
                .filter_map(|update| match update {
                    CompletionUpdateV1::Content(part) => Some(part.contributions.as_slice()),
                    _ => None,
                })
                .flatten()
                .collect::<Vec<_>>();
            assert!(
                contributions
                    .iter()
                    .all(|entry| entry.byte_start < entry.byte_end),
                "{}: an emitted span owned no bytes",
                fixture.name
            );
            assert_eq!(
                finished(&updates).map(|done| done.reason),
                Some(CompletionFinishV1::Stop),
                "{}: split {split}",
                fixture.name
            );
        }
    }
}

/// A completion streamed in any two pieces parses as the whole one does.
#[test]
fn every_structural_delimiter_split_matches_the_whole_parse() {
    for fixture in fixtures() {
        let whole = format!(
            "answer{}{}",
            fixture.tool_block("lookup", "city", "Paris"),
            fixture.end()
        );
        let expected = parse_completion_v1(&fixture.descriptor, &whole, PromptModeV1::Chat)
            .unwrap_or_else(|error| panic!("{}: {error}", fixture.name));
        assert_eq!(expected.content, "answer", "{}", fixture.name);
        assert_eq!(expected.tool_calls.len(), 1);
        assert_eq!(expected.tool_calls[0].name, "lookup");
        assert_eq!(expected.tool_calls[0].arguments, r#"{"city": "Paris"}"#);

        for split in 0..=whole.len() {
            let updates = feed(
                &mut fixture.decoder(PromptModeV1::Chat),
                &[&whole.as_bytes()[..split], &whole.as_bytes()[split..]],
            );
            let done = finished(&updates)
                .unwrap_or_else(|| panic!("{}: no finish at split {split}", fixture.name));
            assert_eq!(
                done.completion.as_ref(),
                Some(&expected),
                "{}: split {split}",
                fixture.name
            );
            assert_eq!(done.reason, CompletionFinishV1::ToolCalls);
        }

        let thinking = format!("reason{}answer{}", fixture.thinking_close(), fixture.end());
        for split in 0..=thinking.len() {
            let updates = feed(
                &mut fixture.decoder(PromptModeV1::Thinking),
                &[&thinking.as_bytes()[..split], &thinking.as_bytes()[split..]],
            );
            assert_eq!(
                text(&updates),
                ("reason".to_owned(), "answer".to_owned()),
                "{}: split {split}",
                fixture.name
            );
        }
    }
}

/// A client stop matches at any byte boundary and leaks nothing after it.
#[test]
fn client_stops_match_at_every_byte_and_token_boundary_without_leaking() {
    for fixture in fixtures() {
        let source = "before🦀stopafter";
        for first in 0..=source.len() {
            for second in first..=source.len() {
                let updates = feed(
                    &mut fixture.with_stops(&["🦀stop"]),
                    &[
                        &source.as_bytes()[..first],
                        &source.as_bytes()[first..second],
                        &source.as_bytes()[second..],
                    ],
                );
                assert_eq!(
                    text(&updates).1,
                    "before",
                    "{}: splits {first}, {second}",
                    fixture.name
                );
                assert_eq!(
                    finished(&updates).map(|done| done.reason),
                    Some(CompletionFinishV1::Stop)
                );
            }
        }
    }
}

/// Overlapping stops resolve to the earliest match, not the longest.
#[test]
fn client_stops_resolve_overlaps_prefixes_and_ordinary_token_interiors() {
    let fixture = FixtureV1::new("key-value", KEY_VALUE);
    let updates = feed(
        &mut fixture.with_stops(&["bcde", "abc", "abcd", "def"]),
        &[b"zero abcdef tail"],
    );
    assert_eq!(text(&updates).1, "zero ");

    let shared = feed(
        &mut fixture.with_stops(&["catalog", "cat", "cater"]),
        &[b"prefix-catapult"],
    );
    assert_eq!(text(&shared).1, "prefix-");

    let overlap = feed(&mut fixture.with_stops(&["aaa", "aaab"]), &[b"aaaaa"]);
    assert_eq!(text(&overlap).1, "");
}

/// Completed stops take priority over earlier incomplete prefixes.
#[test]
fn a_complete_stop_precedes_an_earlier_incomplete_one() {
    let fixture = FixtureV1::new("key-value", KEY_VALUE);
    for (stops, bytes, expected) in [
        (["abc", "b"], b"ab".as_slice(), "a"),
        (["abcdef", "cd"], b"abcd".as_slice(), "ab"),
    ] {
        let updates = fixture
            .with_stops(&stops)
            .push(GeneratedTokenV1::new(0, 200, bytes.to_vec()))
            .unwrap();
        assert_eq!(text(&updates).1, expected);
        assert_eq!(finished(&updates).unwrap().reason, CompletionFinishV1::Stop);
        assert_eq!(finished(&updates).unwrap().generated_token_count, 1);
    }
}

/// A stop that splits one token leaves that token's own bytes accounted for.
#[test]
fn a_client_stop_preserves_partial_token_byte_ownership() {
    let fixture = FixtureV1::new("key-value", KEY_VALUE);
    let updates = feed(
        &mut fixture.with_stops(&["STOP"]),
        &[b"left-ST", b"OP-right"],
    );
    let content = updates
        .iter()
        .find_map(|update| match update {
            CompletionUpdateV1::Content(part) => Some(part),
            _ => None,
        })
        .expect("the visible prefix is emitted");
    assert_eq!(content.text, "left-");
    assert_eq!(content.contributions.len(), 1);
    assert_eq!(content.contributions[0].ordinal, 0);
    assert_eq!(content.contributions[0].token_id, 200);
    assert_eq!(content.contributions[0].byte_start, 0);
    assert_eq!(content.contributions[0].byte_end, 5);
}

/// Ending on length flushes held text and invalid bytes use replacement text.
#[test]
fn length_flushes_a_proper_stop_prefix_and_invalid_utf8_is_replaced() {
    let fixture = FixtureV1::new("key-value", KEY_VALUE);
    let mut decoder = fixture.with_stops(&["ending"]);
    let mut updates = decoder
        .push(GeneratedTokenV1::new(0, 200, b"safe end".to_vec()))
        .expect("the held prefix decodes");
    assert_eq!(text(&updates).1, "safe ");
    updates.extend(decoder.finish_length().expect("length flushes"));
    assert_eq!(text(&updates).1, "safe end");

    let mut invalid = fixture.with_stops(&["stop"]);
    let updates = invalid
        .push(GeneratedTokenV1::new(0, 200, b"ok\xffstop".to_vec()))
        .unwrap();
    assert_eq!(text(&updates).1, "ok�");
    assert_eq!(finished(&updates).unwrap().reason, CompletionFinishV1::Stop);
}

/// A client stop inside an open call ends the call where it stopped.
///
/// The host matches stops from the parser's cursor, which stays at the call
/// open while the call is pending, so the stop is found before the parser
/// reads the argument it cuts. The call is then finished as generation ending
/// inside it finishes one: its complete arguments are delivered, the pending
/// one is dropped, and none of the structure reaches the client as text.
#[test]
fn a_client_stop_inside_a_pending_argument_delivers_the_complete_ones() {
    for fixture in fixtures() {
        let parser = &fixture.parser;
        let source = format!(
            "answer{}{}{}city{}{}PaSTOP tail",
            fixture.call_open("lookup"),
            fixture.argument("days", "3"),
            parser.parameter_open,
            parser.parameter_name_close,
            parser.parameter_value_open,
        );
        for split in 0..=source.len() {
            let mut decoder = fixture.with_stops(&["STOP"]);
            let updates = feed(
                &mut decoder,
                &[&source.as_bytes()[..split], &source.as_bytes()[split..]],
            );
            let calls = updates
                .iter()
                .filter_map(|update| match update {
                    CompletionUpdateV1::ToolCalls(calls) => Some(calls.calls.clone()),
                    _ => None,
                })
                .flatten()
                .collect::<Vec<_>>();
            assert_eq!(calls.len(), 1, "{}: split {split}", fixture.name);
            assert_eq!(calls[0].name, "lookup");
            assert_eq!(
                calls[0].arguments, r#"{"days": "3"}"#,
                "{}: split {split}",
                fixture.name
            );
            assert_eq!(
                text(&updates),
                (String::new(), "answer".to_owned()),
                "{}: split {split}",
                fixture.name
            );
            assert_eq!(
                finished(&updates).map(|done| done.reason),
                Some(CompletionFinishV1::ToolCalls),
                "{}: split {split}",
                fixture.name
            );
        }
    }
}

/// The stop list is bounded and distinct, at the bound the bundle declared.
#[test]
fn client_stop_validation_is_bounded_and_distinct() {
    for fixture in fixtures() {
        let maximum = fixture.descriptor.limits.maximum_client_stops;
        for stops in [
            Vec::<String>::new(),
            vec![String::new()],
            vec!["x".to_owned(), "x".to_owned()],
            (0..=maximum).map(|index| index.to_string()).collect(),
        ] {
            assert!(
                CompletionDecoderV1::with_client_stops(
                    Arc::clone(&fixture.descriptor),
                    PromptModeV1::Chat,
                    DecoderLimitsV1::declared(&fixture.descriptor),
                    &stops,
                )
                .is_err(),
                "{}: {stops:?} must be refused",
                fixture.name
            );
        }
    }
}

/// Text that only looks like the start of a marker is ordinary content.
#[test]
fn benign_markup_and_broken_prefixes_are_content() {
    for fixture in fixtures() {
        let mut decoder = fixture.decoder(PromptModeV1::Chat);
        // The opener's first byte, then something that is not the rest of it.
        let opener = fixture.tool_open();
        let teaser = format!("<div>x</div>{}not-it", &opener[..1]);
        let mut updates = decoder
            .push(GeneratedTokenV1::new(0, 200, teaser.clone().into_bytes()))
            .expect("a broken prefix is content");
        updates.extend(decoder.finish_length().expect("length flushes"));
        assert_eq!(text(&updates).1, teaser, "{}", fixture.name);
        assert_eq!(
            finished(&updates).map(|done| done.reason),
            Some(CompletionFinishV1::Length)
        );
    }
}

/// Length may end a generation, and holds back an incomplete scalar.
#[test]
fn length_finish_flushes_text_but_not_a_partial_scalar() {
    for fixture in fixtures() {
        let mut safe = fixture.decoder(PromptModeV1::Thinking);
        safe.push(GeneratedTokenV1::new(0, 200, b"safe".to_vec()))
            .expect("plain reasoning decodes");
        assert_eq!(
            finished(&safe.finish_length().expect("length flushes")).map(|done| done.reason),
            Some(CompletionFinishV1::Length)
        );

        let mut partial = fixture.decoder(PromptModeV1::Chat);
        partial
            .push(GeneratedTokenV1::new(0, 200, vec![0xf0, 0x9f]))
            .expect("an incomplete scalar is held");
        assert_eq!(text(&partial.finish_length().unwrap()).1, "");
    }
}

/// A boundary in the wrong place, out-of-order tokens, and trailing bytes.
#[test]
fn invalid_order_boundaries_and_terminal_sequences_are_rejected() {
    for fixture in fixtures() {
        // Every reserved marker except the ones this state is waiting for and
        // the ones those could still grow out of. A marker that is a prefix of
        // the completion end or the tool opener has to be held rather than
        // refused, which is what the fixture with shorter substrings pins.
        let expected = [fixture.end(), fixture.tool_open()];
        for reserved in fixture.reserved() {
            if expected
                .iter()
                .any(|marker| marker.starts_with(reserved.as_str()))
            {
                continue;
            }
            let mut decoder = fixture.decoder(PromptModeV1::Chat);
            assert!(
                decoder
                    .push(GeneratedTokenV1::new(0, 200, reserved.as_bytes().to_vec()))
                    .is_err(),
                "{}: accepted the reserved marker {reserved}",
                fixture.name
            );
        }

        let mut ordered = fixture.decoder(PromptModeV1::Chat);
        ordered
            .push(GeneratedTokenV1::new(5, 200, b"x".to_vec()))
            .expect("the first token decodes");
        assert!(
            ordered
                .push(GeneratedTokenV1::new(7, 200, b"y".to_vec()))
                .is_err(),
            "{}: accepted a skipped ordinal",
            fixture.name
        );

        let mut duplicate = fixture.decoder(PromptModeV1::Chat);
        duplicate
            .push(GeneratedTokenV1::new(
                0,
                200,
                fixture.end().as_bytes().to_vec(),
            ))
            .expect("the completion ends");
        assert!(
            duplicate
                .push(GeneratedTokenV1::new(
                    1,
                    200,
                    fixture.end().as_bytes().to_vec()
                ))
                .is_err(),
            "{}: accepted a token after the end",
            fixture.name
        );

        let mut trailing = fixture.decoder(PromptModeV1::Chat);
        assert!(
            trailing
                .push(GeneratedTokenV1::new(
                    0,
                    200,
                    format!("{}x", fixture.end()).into_bytes()
                ))
                .is_err(),
            "{}: accepted bytes after the end",
            fixture.name
        );
    }
}

/// A declared boundary identifier must decode to the text it was declared for.
#[test]
fn a_boundary_identifier_must_decode_to_its_declared_text() {
    for fixture in fixtures() {
        let boundary = fixture
            .descriptor
            .vocabulary
            .boundary_tokens
            .first()
            .expect("the fixture declares a boundary token")
            .clone();
        let mut decoder = fixture.decoder(PromptModeV1::Chat);
        assert!(
            decoder
                .push(GeneratedTokenV1::new(0, boundary.id, b"not that".to_vec()))
                .is_err(),
            "{}: accepted {} under a boundary identifier",
            fixture.name,
            boundary.text
        );
    }
}

/// A generation is charged against the limits it was built with.
#[test]
fn resource_limits_are_checked() {
    for fixture in fixtures() {
        let limits = DecoderLimitsV1::new(&fixture.descriptor, 3, 1)
            .expect("a narrower bound is inside the declared one");
        let mut bytes =
            CompletionDecoderV1::new(Arc::clone(&fixture.descriptor), PromptModeV1::Chat, limits);
        assert!(
            bytes
                .push(GeneratedTokenV1::new(0, 200, b"four".to_vec()))
                .is_err()
        );

        let mut tokens =
            CompletionDecoderV1::new(Arc::clone(&fixture.descriptor), PromptModeV1::Chat, limits);
        tokens
            .push(GeneratedTokenV1::new(0, 200, b"a".to_vec()))
            .expect("the first token fits");
        assert!(
            tokens
                .push(GeneratedTokenV1::new(1, 200, b"b".to_vec()))
                .is_err()
        );

        // A deployment cannot ask for more than the bundle declared.
        let declared = fixture.descriptor.limits;
        assert!(
            DecoderLimitsV1::new(
                &fixture.descriptor,
                declared.maximum_completion_bytes + 1,
                1
            )
            .is_err()
        );
        assert!(
            DecoderLimitsV1::new(
                &fixture.descriptor,
                1,
                declared.maximum_generated_tokens + 1
            )
            .is_err()
        );
    }
}

/// A whole-completion parse refuses what a streamed one refuses.
#[test]
fn a_whole_completion_parse_refuses_the_same_shapes() {
    for fixture in fixtures() {
        let parse = |text: &str, mode| parse_completion_v1(&fixture.descriptor, text, mode);

        assert!(parse("plain text", PromptModeV1::Chat).is_err());
        assert!(parse(&format!("ok{}trailing", fixture.end()), PromptModeV1::Chat).is_err());
        // A thinking marker in a chat completion is a boundary out of place.
        assert!(
            parse(
                &format!("bad{}x{}", fixture.thinking_close(), fixture.end()),
                PromptModeV1::Chat
            )
            .is_err()
        );
        assert!(
            parse(
                &"x".repeat(fixture.descriptor.limits.maximum_completion_bytes + 1),
                PromptModeV1::Chat
            )
            .is_err()
        );

        // A call with no arguments parses as an empty object.
        let empty = format!(
            "{}{}{}",
            fixture.call_open("ping"),
            fixture.call_close(),
            fixture.end()
        );
        let parsed = parse(&empty, PromptModeV1::Chat)
            .unwrap_or_else(|error| panic!("{}: {error}", fixture.name));
        assert_eq!(parsed.tool_calls[0].arguments, "{}");
        assert_eq!(parsed.tool_calls[0].name, "ping");
    }
}

/// A reserved marker shorter than the delimiter containing it is structure
/// while that delimiter is still arriving, and a violation everywhere else.
///
/// This is the shape a streaming decoder gets wrong. The nested declaration's
/// delimiters all contain `<|`, so decoding one byte at a time reaches `<|`
/// before the delimiter it belongs to. Judging it on sight would refuse a
/// completion that parses whole, and every split has to agree with the whole.
#[test]
fn a_shorter_reserved_substring_holds_only_where_a_delimiter_could_grow() {
    let fixture = FixtureV1::new("nested", &nested());
    assert_eq!(fixture.reserved(), ["<|"]);
    for delimiter in [fixture.thinking_close(), fixture.end(), fixture.tool_open()] {
        assert!(delimiter.contains("<|"), "{delimiter} shares no substring");
    }

    // Chat termination, thinking close, and tool open, each streamed at every
    // split, must reach exactly what parsing the whole string reaches.
    let tools = fixture.tool_block("lookup", "city", "Oslo");
    for (mode, whole) in [
        (PromptModeV1::Chat, format!("answer{}", fixture.end())),
        (
            PromptModeV1::Thinking,
            format!("why{}answer{}", fixture.thinking_close(), fixture.end()),
        ),
        (
            PromptModeV1::Chat,
            format!("answer{tools}{}", fixture.end()),
        ),
        (
            PromptModeV1::Thinking,
            format!(
                "why{}answer{tools}{}",
                fixture.thinking_close(),
                fixture.end()
            ),
        ),
    ] {
        let expected = parse_completion_v1(&fixture.descriptor, &whole, mode)
            .unwrap_or_else(|error| panic!("{whole}: {error}"));
        for split in 0..=whole.len() {
            let updates = feed(
                &mut fixture.decoder(mode),
                &[&whole.as_bytes()[..split], &whole.as_bytes()[split..]],
            );
            let done =
                finished(&updates).unwrap_or_else(|| panic!("{whole}: no finish at split {split}"));
            assert_eq!(
                done.completion.as_ref(),
                Some(&expected),
                "{whole}: split {split}"
            );
        }
        // One byte at a time is the worst case, and it has to agree too.
        let parts: Vec<&[u8]> = whole.as_bytes().chunks(1).collect();
        let updates = feed(&mut fixture.decoder(mode), &parts);
        assert_eq!(
            finished(&updates).and_then(|done| done.completion.as_ref()),
            Some(&expected),
            "{whole}: byte at a time"
        );
    }

    // The same substring is refused as soon as no delimiter this state waits
    // for could grow out of it.
    for (mode, text) in [
        (PromptModeV1::Chat, "plain <|x"),
        (PromptModeV1::Thinking, "plain <|x"),
        // The thinking close is not a delimiter chat is waiting for, so its
        // own substring is a violation there even though it completes.
        (PromptModeV1::Chat, "plain <|/think|>"),
    ] {
        let mut decoder = fixture.decoder(mode);
        assert!(
            decoder
                .push(GeneratedTokenV1::new(0, 200, text.as_bytes().to_vec()))
                .is_err(),
            "{mode:?} accepted {text}"
        );
    }

    // Held on its own, because either delimiter could still grow out of it.
    for mode in [PromptModeV1::Chat, PromptModeV1::Thinking] {
        let mut decoder = fixture.decoder(mode);
        assert!(
            decoder
                .push(GeneratedTokenV1::new(0, 200, b"ok <|".to_vec()))
                .expect("an unfinished delimiter is held")
                .iter()
                .all(|update| !matches!(update, CompletionUpdateV1::Finished(_))),
            "{mode:?} finished on an unfinished delimiter"
        );
        // And nothing of it leaked into the visible text.
        let updates = feed(&mut fixture.decoder(mode), &[b"ok <|"]);
        assert_eq!(text(&updates).0.len() + text(&updates).1.len(), 3);
    }
}

/// A delimiter and the shorter marker inside it tie, and structure wins.
#[test]
fn a_delimiter_outranks_the_reserved_marker_it_starts_with() {
    let fixture = FixtureV1::new("nested", &nested());
    // `<|` sits at the first byte of `<|end|>`, so both match at one position.
    let updates = feed(
        &mut fixture.decoder(PromptModeV1::Chat),
        &[format!("said{}", fixture.end()).as_bytes()],
    );
    assert_eq!(text(&updates).1, "said");
    assert_eq!(
        finished(&updates).map(|done| done.reason),
        Some(CompletionFinishV1::Stop)
    );

    // The tool opener also starts with it, and opens a tool call rather than
    // being read as the violation it contains.
    let opened = feed(
        &mut fixture.decoder(PromptModeV1::Chat),
        &[fixture.tool_open().as_bytes()],
    );
    assert!(opened.is_empty(), "the opener emitted something");
}

#[test]
fn minimum_tokens_defers_client_stops_without_losing_text() {
    let fixture = FixtureV1::new("key-value", KEY_VALUE);
    let mut decoder = fixture.with_stops(&["END"]);
    decoder.set_minimum_new_tokens(2);
    let updates = feed(&mut decoder, &[b"END", b" middle ", b"END"]);
    assert_eq!(text(&updates).1, "END middle ");
    assert_eq!(
        finished(&updates).map(|done| done.reason),
        Some(CompletionFinishV1::Stop)
    );
}

#[test]
fn explicit_token_stops_are_counted_and_optionally_visible() {
    let fixture = FixtureV1::new("key-value", KEY_VALUE);
    for visible in [false, true] {
        let mut decoder = fixture.decoder(PromptModeV1::Chat);
        decoder.set_decoding_options(
            runtime_server::DecodingOptions {
                stop_token_ids: vec![200],
                include_stop_str_in_output: visible,
                ..Default::default()
            },
            true,
        );
        decoder.set_minimum_new_tokens(1);
        let updates = feed(&mut decoder, &[b"first", b"last"]);
        assert_eq!(
            text(&updates).1,
            if visible { "firstlast" } else { "first" }
        );
        let done = finished(&updates).unwrap();
        assert_eq!(done.reason, CompletionFinishV1::Stop);
        assert_eq!(done.generated_token_count, 2);
    }
}

#[test]
fn raw_completion_can_ignore_eos_and_control_special_token_visibility() {
    for fixture in fixtures() {
        let eos = fixture.descriptor.vocabulary.eos_token_id;
        let marker = fixture
            .descriptor
            .vocabulary
            .boundary_tokens
            .iter()
            .find(|entry| entry.id == eos)
            .unwrap();
        for skip in [false, true] {
            let mut decoder = fixture.decoder(PromptModeV1::Chat);
            decoder.set_decoding_options(
                runtime_server::DecodingOptions {
                    ignore_eos: true,
                    skip_special_tokens: skip,
                    ..Default::default()
                },
                true,
            );
            let mut updates = decoder
                .push(
                    GeneratedTokenV1::new(0, eos, marker.text.as_bytes().to_vec())
                        .with_special(true),
                )
                .unwrap();
            assert!(finished(&updates).is_none());
            updates.extend(
                decoder
                    .push(GeneratedTokenV1::new(1, 200, b"ok".to_vec()))
                    .unwrap(),
            );
            updates.extend(decoder.finish_length().unwrap());
            assert_eq!(
                text(&updates).1,
                if skip {
                    "ok".to_owned()
                } else {
                    format!("{}ok", marker.text)
                }
            );
            assert_eq!(finished(&updates).unwrap().generated_token_count, 2);
        }
    }
}

#[test]
fn client_stop_strings_can_be_included_without_the_trailing_text() {
    for fixture in fixtures() {
        let mut decoder = fixture.with_stops(&["END"]);
        decoder.set_decoding_options(
            runtime_server::DecodingOptions {
                include_stop_str_in_output: true,
                ..Default::default()
            },
            true,
        );
        let updates = feed(&mut decoder, &[b"before E", b"ND trailing"]);
        assert_eq!(text(&updates).1, "before END");
        assert_eq!(finished(&updates).unwrap().reason, CompletionFinishV1::Stop);
    }
}

#[test]
fn replacement_text_preserves_source_bytes_and_stop_matching() {
    for fixture in fixtures() {
        for bytes in [&b"a\x80b\xe2\x82x"[..], &b"\xf0\x80\x80z\xc3x"[..]] {
            for split in 0..=bytes.len() {
                let mut decoder = fixture.decoder(PromptModeV1::Chat);
                let mut updates = feed(&mut decoder, &[&bytes[..split], &bytes[split..]]);
                updates.extend(decoder.finish_length().unwrap());
                assert_eq!(text(&updates).1, String::from_utf8_lossy(bytes));
                let owned: usize = updates
                    .iter()
                    .filter_map(|update| match update {
                        CompletionUpdateV1::Content(part) => Some(
                            part.contributions
                                .iter()
                                .map(|c| c.byte_end - c.byte_start)
                                .sum::<usize>(),
                        ),
                        _ => None,
                    })
                    .sum();
                assert_eq!(owned, bytes.len());
            }
        }
        let mut decoder = fixture.with_stops(&["�x"]);
        let updates = feed(&mut decoder, &[b"ok\x80", b"xafter"]);
        assert_eq!(text(&updates).1, "ok");
        assert_eq!(finished(&updates).unwrap().reason, CompletionFinishV1::Stop);
        let mut decoder = fixture.decoder(PromptModeV1::Chat);
        assert!(feed(&mut decoder, &[&[0xe2, 0x82]]).is_empty());
        assert_eq!(
            text(
                &decoder
                    .push(GeneratedTokenV1::new(1, 200, vec![0xac]))
                    .unwrap()
            )
            .1,
            "€"
        );
    }
}

#[test]
fn length_and_token_stops_do_not_flush_partial_scalars() {
    for fixture in fixtures() {
        let mut decoder = fixture.with_stops(&["�"]);
        assert!(feed(&mut decoder, &[b"\xe2\x82"]).is_empty());
        let updates = decoder.finish_length().unwrap();
        assert_eq!(text(&updates).1, "");
        assert_eq!(
            finished(&updates).unwrap().reason,
            CompletionFinishV1::Length
        );
        let mut decoder = fixture.decoder(PromptModeV1::Chat);
        decoder.set_decoding_options(
            runtime_server::DecodingOptions {
                stop_token_ids: vec![201],
                ..Default::default()
            },
            true,
        );
        assert!(feed(&mut decoder, &[b"\xe2\x82"]).is_empty());
        let updates = decoder
            .push(GeneratedTokenV1::new(1, 201, b"stop".to_vec()))
            .unwrap();
        assert_eq!(text(&updates).1, "");
        assert_eq!(finished(&updates).unwrap().reason, CompletionFinishV1::Stop);
    }
}

#[test]
fn minimum_tokens_defers_textual_terminators_from_ordinary_tokens() {
    for fixture in fixtures() {
        for engine_sampled in [false, true] {
            let mut decoder = fixture.decoder(PromptModeV1::Chat);
            decoder.set_minimum_new_tokens(3);
            for (ordinal, bytes) in [fixture.end(), " middle ", fixture.end()]
                .iter()
                .enumerate()
            {
                let token = GeneratedTokenV1::new(ordinal as u64, 200, bytes.as_bytes().to_vec());
                let updates = if engine_sampled {
                    decoder.push_engine_token(token)
                } else {
                    decoder.push(token)
                }
                .unwrap();
                assert!(
                    finished(&updates).is_none(),
                    "{} stopped at {}",
                    fixture.name,
                    ordinal + 1
                );
                assert_eq!(text(&updates).1, *bytes);
            }
            let token = GeneratedTokenV1::new(3, 200, fixture.end().as_bytes().to_vec());
            let updates = if engine_sampled {
                decoder.push_engine_token(token)
            } else {
                decoder.push(token)
            }
            .unwrap();
            if engine_sampled {
                assert!(finished(&updates).is_none());
                assert_eq!(text(&updates).1, fixture.end());
                let updates = decoder
                    .push_engine_token(
                        GeneratedTokenV1::new(
                            4,
                            fixture.descriptor.vocabulary.eos_token_id,
                            fixture.end().as_bytes().to_vec(),
                        )
                        .with_special(true),
                    )
                    .unwrap();
                assert_eq!(finished(&updates).unwrap().reason, CompletionFinishV1::Stop);
            } else {
                assert_eq!(finished(&updates).unwrap().reason, CompletionFinishV1::Stop);
            }
        }
    }
}

#[test]
fn client_stops_require_a_token_after_the_minimum_and_can_cross_its_boundary() {
    let fixture = FixtureV1::new("key-value", KEY_VALUE);
    for (parts, expected) in [
        (vec![b"END".as_slice(), b"END", b"END", b"END"], "ENDENDEND"),
        (
            vec![b"END".as_slice(), b" middle ", b"E", b"ND"],
            "END middle ",
        ),
        (vec![b"a".as_slice(), b"E", b"N", b"D"], "a"),
    ] {
        for include in [false, true] {
            let mut decoder = fixture.with_stops(&["END"]);
            decoder.set_decoding_options(
                runtime_server::DecodingOptions {
                    include_stop_str_in_output: include,
                    ..Default::default()
                },
                false,
            );
            decoder.set_minimum_new_tokens(3);
            let mut delivered = String::new();
            for (ordinal, bytes) in parts.iter().enumerate() {
                let updates = decoder
                    .push(GeneratedTokenV1::new(ordinal as u64, 200, bytes.to_vec()))
                    .unwrap();
                assert_eq!(finished(&updates).is_some(), ordinal == 3);
                delivered.push_str(&text(&updates).1);
            }
            assert_eq!(
                delivered,
                format!("{expected}{}", if include { "END" } else { "" })
            );
        }
    }
}

#[test]
fn engine_end_marker_spellings_stay_literal_across_the_minimum_boundary() {
    for fixture in fixtures() {
        for minimum in [0, 1, 2, 3, 10] {
            let mut decoder = fixture.decoder(PromptModeV1::Chat);
            decoder.set_minimum_new_tokens(minimum);
            let marker = fixture.end().as_bytes();
            let split = marker.len() / 2;
            let mut delivered = String::new();
            for (ordinal, bytes) in [b"hi".as_slice(), &marker[..split], &marker[split..], b"z"]
                .iter()
                .enumerate()
            {
                let updates = decoder
                    .push_engine_token(GeneratedTokenV1::new(ordinal as u64, 200, bytes.to_vec()))
                    .unwrap();
                assert!(
                    finished(&updates).is_none(),
                    "{} minimum={minimum}, ordinal={ordinal}",
                    fixture.name
                );
                delivered.push_str(&text(&updates).1);
            }
            let updates = decoder.finish_length().unwrap();
            delivered.push_str(&text(&updates).1);
            assert_eq!(delivered, format!("hi{}z", fixture.end()));
            assert_eq!(
                finished(&updates).unwrap().reason,
                CompletionFinishV1::Length
            );
            assert_eq!(finished(&updates).unwrap().generated_token_count, 4);
        }
    }
}
