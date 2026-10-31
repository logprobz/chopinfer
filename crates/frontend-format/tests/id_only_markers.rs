#![forbid(unsafe_code)]
#![allow(clippy::single_range_in_vec_init)]
//! Markers read only from their token IDs: with a declaration that names
//! ID-only markers and pushes that mark the tokens generated as them, a
//! marker is a marker only where its ID-only spellings are marked tokens
//! and its other bytes ordinary ones, so a marker spelled by ordinary
//! tokens is text. Text without marks reads by spelling, as before.
#[allow(dead_code)]
mod common;

use std::ops::Range;
use std::sync::Arc;

use frontend_format::{
    AssistantCompletionV1, CompletionFinishV1, DeclaredParserV1, FormatDescriptorV1, FormatEventV1,
    FormatInputV1, FormatParserOptionsV1, ParsedToolCallV1, PromptModeV1, parse_completion_v1,
    parse_marked_completion_v1,
};
use serde_json::{Value, json};

/// The typed-parameter fixture with its tool markers built around the
/// boundary token `|T|`, the one reserved marker, and `<think>`, `</think>`
/// and `|T|` declared ID-only.
fn descriptor() -> FormatDescriptorV1 {
    let mut declaration: Value = serde_json::from_str(common::TYPED_PARAMETERS).unwrap();
    let vocabulary = &mut declaration["vocabulary"];
    for (text, id) in [("<think>", 5), ("|T|", 6)] {
        vocabulary["boundary_tokens"]
            .as_array_mut()
            .unwrap()
            .push(json!({"text": text, "id": id}));
    }
    vocabulary["id_only_markers"] = json!(["<think>", "</think>", "|T|"]);
    let parser = &mut declaration["format"]["parser"];
    for (field, marker) in [
        ("tool_call_open", "\n\n<|T| calls>\n"),
        ("tool_call_close", "</|T| calls>"),
        ("function_open", "<|T| invoke name=\""),
        ("function_close", "</|T| invoke>\n"),
        ("parameter_open", "<|T| parameter name=\""),
        ("parameter_close", "</|T| parameter>\n"),
    ] {
        parser[field] = json!(marker);
    }
    parser["reserved_markers"] = json!(["|T|"]);
    FormatDescriptorV1::from_declaration_bytes(&serde_json::to_vec(&declaration).unwrap()).unwrap()
}

/// A completion written with each marked token between `⟦` and `⟧`: its
/// text and its marks.
fn marked(written: &str) -> (String, Vec<Range<usize>>) {
    let (mut text, mut marks, mut start) = (String::new(), Vec::new(), None);
    for character in written.chars() {
        match character {
            '⟦' => start = Some(text.len()),
            '⟧' => marks.push(start.take().unwrap()..text.len()),
            other => text.push(other),
        }
    }
    (text, marks)
}

fn options(mode: PromptModeV1, stream_tool_calls: bool) -> FormatParserOptionsV1 {
    FormatParserOptionsV1 {
        mode,
        reasoning_open: mode == PromptModeV1::Thinking,
        extract_tools: true,
        stream_tool_calls,
        raw: false,
        ignore_eos: false,
        engine_tokens: false,
        tool_schemas: Vec::new(),
    }
}

/// `completion` pushed as the host releases it: each marked token whole,
/// each other character alone, each push naming the marks released so far.
/// The events, or the first refusal.
fn pushed(
    descriptor: &FormatDescriptorV1,
    completion: &str,
    marks: &[Range<usize>],
    mode: PromptModeV1,
    stream_tool_calls: bool,
    last: bool,
) -> Result<Vec<FormatEventV1>, String> {
    let mut parser = DeclaredParserV1::new(
        Arc::new(descriptor.clone()),
        options(mode, stream_tool_calls),
    );
    let mut ends = Vec::new();
    let mut at = 0;
    while let Some(character) = completion[at..].chars().next() {
        at = match marks.iter().find(|mark| mark.start == at) {
            Some(mark) => mark.end,
            None => at + character.len_utf8(),
        };
        ends.push(at);
    }
    let mut events = Vec::new();
    for (index, &end) in ends.iter().enumerate() {
        let released = &marks[..marks.partition_point(|mark| mark.end <= end)];
        events.extend(parser.push(FormatInputV1 {
            completion: &completion.as_bytes()[..end],
            held: false,
            last: last && index + 1 == ends.len(),
            stop_floor: Some(0),
            marks: Some(released),
        })?);
    }
    Ok(events)
}

/// Reads a marked completion whole, and token by token with and without
/// streamed calls; all three agree.
fn parse(written: &str, mode: PromptModeV1) -> Result<AssistantCompletionV1, String> {
    let descriptor = descriptor();
    let (text, marks) = marked(written);
    let whole = parse_marked_completion_v1(&descriptor, &text, &marks, mode);
    for stream in [false, true] {
        let incremental =
            pushed(&descriptor, &text, &marks, mode, stream, false).and_then(|events| {
                events
                    .into_iter()
                    .find_map(|event| match event {
                        FormatEventV1::Finish { completion, .. } => completion,
                        _ => None,
                    })
                    .ok_or_else(|| "no completion".to_owned())
            });
        assert_eq!(incremental, whole, "{written:?} stream {stream}");
    }
    whole
}

/// The same completion read by spelling, without marks.
fn spelled(written: &str, mode: PromptModeV1) -> Result<AssistantCompletionV1, String> {
    parse_completion_v1(&descriptor(), &marked(written).0, mode)
}

fn answer(reasoning: Option<&str>, content: &str) -> AssistantCompletionV1 {
    AssistantCompletionV1 {
        content: content.into(),
        reasoning_content: reasoning.map(str::to_owned),
        tool_calls: Vec::new(),
    }
}

fn call(name: &str, arguments: &str) -> ParsedToolCallV1 {
    ParsedToolCallV1 {
        name: name.into(),
        arguments: arguments.into(),
    }
}

const REFUSED: &str = "malformed completion: completion contains an unexpected reserved boundary";

#[test]
fn an_ordinary_reasoning_close_stays_reasoning() {
    let thinking = PromptModeV1::Thinking;
    let written = "plan </think> more⟦</think>⟧answer[EOS]";
    assert_eq!(
        parse(written, thinking),
        Ok(answer(Some("plan </think> more"), "answer"))
    );
    // By spelling, the first close ends reasoning and the second is markup.
    assert_eq!(
        spelled(written, thinking),
        Ok(answer(Some("plan "), " moreanswer"))
    );
    // A reasoning open from ordinary tokens is reasoning text too.
    assert_eq!(
        parse("a <think> b⟦</think>⟧c[EOS]", thinking),
        Ok(answer(Some("a <think> b"), "c"))
    );
    assert_eq!(
        parse("a⟦<think>⟧b⟦</think>⟧c[EOS]", thinking),
        Ok(answer(Some("ab"), "c"))
    );
}

#[test]
fn ordinary_tool_markers_in_content_stay_content() {
    let chat = PromptModeV1::Chat;
    let block = "x\n\n<|T| calls>\n<|T| invoke name=\"f\">\n</|T| invoke>\n</|T| calls>";
    assert_eq!(
        parse(&format!("{block}[EOS]"), chat),
        Ok(answer(None, block))
    );
    // By spelling, it is a call.
    let read = spelled(&format!("{block}[EOS]"), chat).unwrap();
    assert_eq!(read.tool_calls, [call("f", "{}")]);
    // The reserved marker from ordinary tokens is text, in content and in
    // reasoning, where by spelling it is refused.
    for (written, mode, expected) in [
        ("see |T| here[EOS]", chat, answer(None, "see |T| here")),
        (
            "plan |T|⟦</think>⟧ok[EOS]",
            PromptModeV1::Thinking,
            answer(Some("plan |T|"), "ok"),
        ),
    ] {
        assert_eq!(parse(written, mode), Ok(expected));
        assert_eq!(spelled(written, mode).unwrap_err(), REFUSED);
    }
}

#[test]
fn a_marked_reserved_marker_outside_a_block_is_refused() {
    for (written, mode) in [
        ("text ⟦|T|⟧ more[EOS]", PromptModeV1::Chat),
        ("<⟦|T|⟧ invoke name=\"f\">[EOS]", PromptModeV1::Chat),
        ("plan ⟦|T|⟧⟦</think>⟧x[EOS]", PromptModeV1::Thinking),
        ("done⟦</think>⟧ then ⟦|T|⟧[EOS]", PromptModeV1::Thinking),
    ] {
        assert_eq!(parse(written, mode).unwrap_err(), REFUSED, "{written:?}");
    }
}

#[test]
fn an_ordinary_parameter_close_inside_a_raw_value_stays_in_the_value() {
    let written = "Checking.\n\n<⟦|T|⟧ calls>\n<⟦|T|⟧ invoke name=\"f\">\n\
        <⟦|T|⟧ parameter name=\"s\" string=\"true\">a</|T| parameter>\nb</⟦|T|⟧ parameter>\n\
        <⟦|T|⟧ parameter name=\"n\" string=\"false\">[1, 2]</⟦|T|⟧ parameter>\n\
        </⟦|T|⟧ invoke>\n</⟦|T|⟧ calls>[EOS]";
    let read = parse(written, PromptModeV1::Chat).unwrap();
    assert_eq!(read.content, "Checking.");
    assert_eq!(
        read.tool_calls,
        [call("f", r#"{"s": "a</|T| parameter>\nb", "n": [1, 2]}"#)]
    );
    // By spelling, the ordinary close ends the value.
    assert_eq!(
        spelled(written, PromptModeV1::Chat).unwrap().tool_calls,
        [call("f", r#"{"s": "a", "n": [1, 2]}"#)]
    );
    // An ordinary function or calls close inside the call is text as well,
    // and an ordinary parameter open is no parameter.
    let written = "\n\n<⟦|T|⟧ calls>\n<⟦|T|⟧ invoke name=\"g\">\n\
        <|T| parameter name=\"x\" string=\"true\">1</|T| parameter>\n</|T| invoke>\n\
        <⟦|T|⟧ parameter name=\"y\" string=\"true\">2</⟦|T|⟧ parameter>\n</|T| calls>\n\
        </⟦|T|⟧ invoke>\n</⟦|T|⟧ calls>[EOS]";
    assert_eq!(
        parse(written, PromptModeV1::Chat).unwrap().tool_calls,
        [call("g", r#"{"y": "2"}"#)]
    );
}

/// A push that keeps fewer bytes than the parser read, as a client stop
/// cuts a completion, drops the marks past them: bytes that follow it are
/// read with the push's own marks alone.
#[test]
fn a_cut_drops_the_marks_past_the_kept_bytes() {
    let descriptor = Arc::new(descriptor());
    let mut parser =
        DeclaredParserV1::new(Arc::clone(&descriptor), options(PromptModeV1::Chat, false));
    let (text, marks) = marked("ok <⟦|T|⟧ ca");
    let events = parser
        .push_suffix(0, text.as_bytes(), false, false, Some(0), Some(&marks))
        .unwrap();
    // The marked start of a calls open is held.
    assert_eq!(events, [FormatEventV1::Content(0..3)]);
    let rest = "|T| calls>\n<|T| invoke name=\"f\">\n</|T| invoke>\n</|T| calls>[EOS]";
    let events = parser
        .push_suffix(4, rest.as_bytes(), false, false, Some(0), Some(&[]))
        .unwrap();
    let Some(FormatEventV1::Finish { reason, completion }) = events.last() else {
        panic!("no finish: {events:?}");
    };
    assert_eq!(*reason, CompletionFinishV1::Stop);
    assert_eq!(
        completion.as_ref(),
        Some(&answer(
            None,
            &format!("ok <{}", rest.strip_suffix("[EOS]").unwrap())
        ))
    );

    // A client stop cut the last push through the marked token: the cut
    // completion names no mark, and the held `<` is text.
    let mut parser = DeclaredParserV1::new(descriptor, options(PromptModeV1::Chat, false));
    parser
        .push(FormatInputV1 {
            completion: text.as_bytes(),
            held: false,
            last: false,
            stop_floor: Some(0),
            marks: Some(&marks),
        })
        .unwrap();
    let events = parser
        .push(FormatInputV1 {
            completion: &text.as_bytes()[..4],
            held: false,
            last: true,
            stop_floor: Some(0),
            marks: Some(&[]),
        })
        .unwrap();
    assert_eq!(events[0], FormatEventV1::Content(3..4));
}

/// A token an earlier push released in part crosses again whole, its mark
/// starting before the kept bytes.
#[test]
fn a_token_released_in_part_crosses_again_whole() {
    let descriptor = Arc::new(descriptor());
    let mut parser = DeclaredParserV1::new(descriptor, options(PromptModeV1::Chat, false));
    // `|T` of a `|T|` token, then the push that sends it whole.
    let events = parser
        .push_suffix(0, b"x\n\n<|T", false, false, Some(0), Some(&[4..6]))
        .unwrap();
    assert_eq!(events, [FormatEventV1::Content(0..1)]);
    let (rest, mut marks) = marked(
        "T| calls>\n<⟦|T|⟧ invoke name=\"f\">\n<⟦|T|⟧ parameter name=\"s\" string=\"true\">v</⟦|T|⟧ parameter>\n</⟦|T|⟧ invoke>\n</⟦|T|⟧ calls>[EOS]",
    );
    for mark in &mut marks {
        *mark = mark.start + 5..mark.end + 5;
    }
    marks.insert(0, 4..7);
    let events = parser
        .push_suffix(5, rest.as_bytes(), false, false, Some(0), Some(&marks))
        .unwrap();
    let Some(FormatEventV1::Finish { reason, completion }) = events.last() else {
        panic!("no finish: {events:?}");
    };
    assert_eq!(*reason, CompletionFinishV1::ToolCalls);
    let completion = completion.as_ref().unwrap();
    assert_eq!(completion.content, "x");
    assert_eq!(completion.tool_calls, [call("f", r#"{"s": "v"}"#)]);
}

/// Only text that can still become a marker is held: the start of a calls
/// open whose reserved marker is an ordinary spelling streams at once, and
/// one that generation ends in is text.
#[test]
fn only_a_marked_prefix_is_held() {
    let descriptor = descriptor();
    let content = |events: &[FormatEventV1], text: &str| -> String {
        events
            .iter()
            .filter_map(|event| match event {
                FormatEventV1::Content(range) => Some(&text[range.clone()]),
                _ => None,
            })
            .collect()
    };
    for (written, streamed, finished) in [
        ("hi <⟦|T|⟧", "hi ", "hi "),
        ("hi <|T|", "hi <|T|", "hi <|T|"),
        ("hi\n\n<⟦|T|⟧ ca", "hi", "hi"),
        ("hi\n\n<|T| ca", "hi\n\n<|T| ca", "hi\n\n<|T| ca"),
    ] {
        let (text, marks) = marked(written);
        for (last, expected) in [(false, streamed), (true, finished)] {
            let events =
                pushed(&descriptor, &text, &marks, PromptModeV1::Chat, false, last).unwrap();
            assert_eq!(content(&events, &text), expected, "{written:?} last {last}");
        }
    }
}

/// Where every ID-only spelling is its token, the marked read is the
/// spelled read, and text without marks reads by spelling as before.
#[test]
fn fully_marked_text_reads_as_its_spelling() {
    let descriptor = descriptor();
    let spellings = ["<think>", "</think>", "|T|"];
    for (text, mode) in [
        ("plain answer[EOS]", PromptModeV1::Chat),
        ("plan<think>more</think>answer[EOS]", PromptModeV1::Thinking),
        (
            "Hi\n\n<|T| calls>\n<|T| invoke name=\"f\">\n<|T| parameter name=\"s\" \
             string=\"true\">a b</|T| parameter>\n<|T| parameter name=\"n\" \
             string=\"false\">{\"k\": [1]}</|T| parameter>\n</|T| invoke>\n<|T| invoke \
             name=\"g\">\n</|T| invoke>\n</|T| calls>[EOS]",
            PromptModeV1::Chat,
        ),
        ("bad |T| here[EOS]", PromptModeV1::Chat),
        ("</think>after[EOS]", PromptModeV1::Chat),
    ] {
        let mut marks = Vec::new();
        let mut at = 0;
        while at < text.len() {
            match spellings
                .iter()
                .find(|spelling| text[at..].starts_with(**spelling))
            {
                Some(spelling) => {
                    marks.push(at..at + spelling.len());
                    at += spelling.len();
                }
                None => at += 1,
            }
        }
        assert_eq!(
            parse_marked_completion_v1(&descriptor, text, &marks, mode),
            parse_completion_v1(&descriptor, text, mode),
            "{text:?}"
        );
    }
    // Without marks, the ID-only declaration reads by spelling.
    assert_eq!(
        parse_completion_v1(
            &descriptor,
            "plan</think>answer[EOS]",
            PromptModeV1::Thinking
        ),
        Ok(answer(Some("plan"), "answer"))
    );
}

#[test]
fn marks_follow_the_push_sequence() {
    let descriptor = Arc::new(descriptor());
    let parser =
        || DeclaredParserV1::new(Arc::clone(&descriptor), options(PromptModeV1::Chat, false));
    let sequence = |message: &str| format!("invalid conversation sequence: {message}");

    // A declaration without ID-only markers takes no marks.
    let typed =
        FormatDescriptorV1::from_component_bytes(common::TYPED_PARAMETERS.as_bytes()).unwrap();
    assert_eq!(
        parse_marked_completion_v1(&typed, "x[EOS]", &[], PromptModeV1::Chat).unwrap_err(),
        sequence("a push carries marks, and the declaration names no ID-only markers")
    );
    assert!(parse_completion_v1(&typed, "x[EOS]", PromptModeV1::Chat).is_ok());

    // The first push decides whether the parser's pushes carry marks.
    let mut marked = parser();
    marked
        .push_suffix(0, b"ab", false, false, Some(0), Some(&[]))
        .unwrap();
    assert_eq!(
        marked
            .push_suffix(2, b"c", false, false, Some(0), None)
            .unwrap_err(),
        sequence("a push without marks follows one with marks")
    );
    let mut unmarked = parser();
    unmarked
        .push_suffix(0, b"ab", false, false, Some(0), None)
        .unwrap();
    assert_eq!(
        unmarked
            .push_suffix(2, b"c", false, false, Some(0), Some(&[]))
            .unwrap_err(),
        sequence("a push with marks follows one without")
    );

    // Each mark is not empty, follows the marks kept and the one before,
    // ends after the kept bytes and within the released ones. The first
    // push holds the marked start of a calls open.
    let first = || {
        let mut parser = parser();
        let events = parser
            .push_suffix(0, b"ab<|T|", false, false, Some(0), Some(&[3..6]))
            .unwrap();
        assert_eq!(events, [FormatEventV1::Content(0..2)]);
        parser
    };
    for (keep, marks) in [
        (6, vec![7..7]),
        (6, vec![7..9, 8..10]),
        (6, vec![5..8]),
        (6, vec![1..6]),
        (6, vec![7..20]),
        (4, vec![1..4]),
    ] {
        let error = first()
            .push_suffix(keep, b"cdefg", false, false, Some(0), Some(&marks))
            .unwrap_err();
        assert!(
            error.starts_with(&sequence("mark ")),
            "{keep} {marks:?}: {error}"
        );
    }
    // Dropping the mark past the kept bytes frees its place.
    assert_eq!(
        first().push_suffix(4, b"T| ca", false, false, Some(0), Some(&[3..6])),
        Ok(Vec::new())
    );
}

/// In a tool region whose bytes are not valid UTF-8, each marker the scan
/// finds in the decoded text is checked against the marks of the bytes it
/// came from, whole or token by token.
#[test]
fn marks_hold_across_invalid_bytes_in_a_tool_region() {
    let descriptor = Arc::new(descriptor());
    let (before, mut marks) = marked(
        "\n\n<⟦|T|⟧ calls>\n<⟦|T|⟧ invoke name=\"f\">\n<⟦|T|⟧ parameter name=\"s\" string=\"true\">a",
    );
    let (after, later) =
        marked("b</|T| parameter>\nc</⟦|T|⟧ parameter>\n</⟦|T|⟧ invoke>\n</⟦|T|⟧ calls>[EOS]");
    let mut completion = before.into_bytes();
    completion.push(0xff);
    let offset = completion.len();
    completion.extend_from_slice(after.as_bytes());
    marks.extend(
        later
            .into_iter()
            .map(|mark| mark.start + offset..mark.end + offset),
    );
    let mut ends = Vec::new();
    let mut at = 0;
    while at < completion.len() {
        at = marks
            .iter()
            .find(|mark| mark.start == at)
            .map_or(at + 1, |mark| mark.end);
        ends.push(at);
    }
    for ends in [vec![completion.len()], ends] {
        let mut parser =
            DeclaredParserV1::new(Arc::clone(&descriptor), options(PromptModeV1::Chat, true));
        let mut finished = None;
        for end in ends {
            let released = &marks[..marks.partition_point(|mark| mark.end <= end)];
            for event in parser
                .push(FormatInputV1 {
                    completion: &completion[..end],
                    held: false,
                    last: false,
                    stop_floor: Some(0),
                    marks: Some(released),
                })
                .unwrap()
            {
                if let FormatEventV1::Finish { completion, .. } = event {
                    finished = completion;
                }
            }
        }
        let calls = finished.expect("the completion finished").tool_calls;
        assert_eq!(calls.len(), 1);
        let arguments: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(arguments, json!({"s": "a\u{fffd}b</|T| parameter>\nc"}));
    }
}
