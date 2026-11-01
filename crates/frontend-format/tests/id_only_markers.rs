#![forbid(unsafe_code)]
#![allow(clippy::single_range_in_vec_init)]
//! Markers read only from their token IDs: with a declaration that names
//! ID-only markers and pushes that mark the tokens generated as them, a
//! marker is a marker only where its ID-only spellings are marked tokens
//! and its other bytes ordinary ones, so a marker spelled by ordinary
//! tokens is text. Text without marks reads by spelling, as before. A tool
//! choice's grammar writes those spellings as the tokens, so what it admits
//! reads back as it was written.
#[allow(dead_code)]
mod common;

use std::ops::Range;
use std::sync::Arc;

use frontend_format::{
    AssistantCompletionV1, CompletionFinishV1, DeclaredParserV1, FormatDescriptorV1, FormatEventV1,
    FormatInputV1, FormatParserOptionsV1, FrontendToolChoiceV1, FrontendToolV1, ParsedToolCallV1,
    PromptModeV1, parse_completion_v1, parse_marked_completion_v1, tool_constraint_v1,
};
use runtime_grammar::{
    CompiledGrammar, ConstraintState, GRAMMAR_WIRE_LAYOUT_V1, Grammar, GrammarMatcher, PrefixPart,
    TokenVocabulary, VocabularyToken,
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

/// A cumulative push of `completion` naming `marks`, more to follow.
fn input<'a>(completion: &'a [u8], marks: &'a [Range<usize>]) -> FormatInputV1<'a> {
    FormatInputV1 {
        completion,
        held: false,
        last: false,
        stop_floor: Some(0),
        marks: Some(marks),
    }
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

/// A token an earlier push released in part crosses without a mark, and
/// its mark, once the token is whole, starts before the kept bytes; the
/// held start of the marker is no text meanwhile. The same holds through
/// the bundle ABI's push form, through cumulative pushes, and for a calls
/// open.
#[test]
fn a_token_released_in_part_is_marked_once_whole() {
    let descriptor = Arc::new(descriptor());
    let thinking = || {
        DeclaredParserV1::new(
            Arc::clone(&descriptor),
            options(PromptModeV1::Thinking, false),
        )
    };
    let finished = [
        FormatEventV1::Content(12..18),
        FormatEventV1::Finish {
            reason: CompletionFinishV1::Stop,
            completion: Some(answer(Some("plan"), "answer")),
        },
    ];
    // `</thi` of a `</think>` token, held, then the rest with its mark.
    let mut parser = thinking();
    let events = parser
        .push_suffix(0, b"plan</thi", true, false, Some(0), Some(&[]))
        .unwrap();
    assert_eq!(events, [FormatEventV1::Reasoning(0..4)]);
    let events = parser
        .push_suffix(9, b"nk>answer[EOS]", false, false, Some(0), Some(&[4..12]))
        .unwrap();
    assert_eq!(events, finished);

    let completion = b"plan</think>answer[EOS]";
    let mut parser = thinking();
    let events = parser
        .push(FormatInputV1 {
            completion: &completion[..9],
            held: true,
            last: false,
            stop_floor: Some(0),
            marks: Some(&[]),
        })
        .unwrap();
    assert_eq!(events, [FormatEventV1::Reasoning(0..4)]);
    let events = parser
        .push(FormatInputV1 {
            completion,
            held: false,
            last: false,
            stop_floor: Some(0),
            marks: Some(&[4..12]),
        })
        .unwrap();
    assert_eq!(events, finished);
    assert_eq!(
        parse_marked_completion_v1(
            &descriptor,
            "plan</think>answer[EOS]",
            &[4..12],
            PromptModeV1::Thinking
        ),
        Ok(answer(Some("plan"), "answer"))
    );

    // `|T` of the `|T|` that a calls open holds, then the rest.
    let mut parser =
        DeclaredParserV1::new(Arc::clone(&descriptor), options(PromptModeV1::Chat, false));
    let events = parser
        .push_suffix(0, b"x\n\n<|T", true, false, Some(0), Some(&[]))
        .unwrap();
    assert_eq!(events, [FormatEventV1::Content(0..1)]);
    let (rest, mut marks) = marked(
        "| calls>\n<⟦|T|⟧ invoke name=\"f\">\n<⟦|T|⟧ parameter name=\"s\" \
         string=\"true\">v</⟦|T|⟧ parameter>\n</⟦|T|⟧ invoke>\n</⟦|T|⟧ calls>[EOS]",
    );
    for mark in &mut marks {
        *mark = mark.start + 6..mark.end + 6;
    }
    marks.insert(0, 4..7);
    let events = parser
        .push_suffix(6, rest.as_bytes(), false, false, Some(0), Some(&marks))
        .unwrap();
    let Some(FormatEventV1::Finish { reason, completion }) = events.last() else {
        panic!("no finish: {events:?}");
    };
    assert_eq!(*reason, CompletionFinishV1::ToolCalls);
    let completion = completion.as_ref().unwrap();
    assert_eq!(completion.content, "x");
    assert_eq!(completion.tool_calls, [call("f", r#"{"s": "v"}"#)]);
}

/// A mark covers one whole token and never changes: a mark sent for part
/// of a token overlaps the whole mark that follows, which is refused.
#[test]
fn a_mark_for_part_of_a_token_is_refused_when_the_token_completes() {
    let descriptor = Arc::new(descriptor());
    let thinking = || {
        DeclaredParserV1::new(
            Arc::clone(&descriptor),
            options(PromptModeV1::Thinking, false),
        )
    };
    let mut parser = thinking();
    parser
        .push_suffix(0, b"plan</thi", true, false, Some(0), Some(&[4..9]))
        .unwrap();
    assert_eq!(
        parser
            .push_suffix(9, b"nk>answer[EOS]", false, false, Some(0), Some(&[4..12]))
            .unwrap_err(),
        "invalid conversation sequence: mark 4..12 starts before the mark before it ends, at 9"
    );
    let completion = b"plan</think>answer[EOS]";
    let mut parser = thinking();
    parser.push(input(&completion[..9], &[4..9])).unwrap();
    assert_eq!(
        parser.push(input(completion, &[4..12])).unwrap_err(),
        "invalid conversation sequence: the marks of the kept bytes are not the marks the \
         parser kept"
    );
}

/// A client stop that cuts through a mark drops it, and the push that cuts
/// names the mark again, whole, when its token still stands.
#[test]
fn a_cut_through_a_mark_names_it_again_whole() {
    let descriptor = Arc::new(descriptor());
    let mut parser = DeclaredParserV1::new(descriptor, options(PromptModeV1::Chat, false));
    let (text, marks) = marked("ok <⟦|T|⟧ ca");
    let events = parser
        .push_suffix(0, text.as_bytes(), false, false, Some(0), Some(&marks))
        .unwrap();
    assert_eq!(events, [FormatEventV1::Content(0..3)]);
    let (rest, mut marks) =
        marked("T| calls>\n<⟦|T|⟧ invoke name=\"f\">\n</⟦|T|⟧ invoke>\n</⟦|T|⟧ calls>[EOS]");
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
    assert_eq!(completion.content, "ok ");
    assert_eq!(completion.tool_calls, [call("f", "{}")]);
}

/// Every mark list is checked whole before the parser reads any of it, and
/// a cumulative push names the kept marks unchanged.
#[test]
fn malformed_mark_lists_are_refused() {
    let descriptor = Arc::new(descriptor());
    let thinking = || {
        DeclaredParserV1::new(
            Arc::clone(&descriptor),
            options(PromptModeV1::Thinking, false),
        )
    };
    let sequence = |message: &str| format!("invalid conversation sequence: {message}");
    let text = "plan</think>answer[EOS]";
    let completion = text.as_bytes();
    for (marks, refusal) in [
        (
            vec![100..101],
            "mark 100..101 ends past the 23 completion bytes",
        ),
        (vec![0..0], "mark 0..0 is empty or inverted"),
        (
            vec![Range { start: 5, end: 4 }],
            "mark 5..4 is empty or inverted",
        ),
        (
            vec![4..12, 8..14],
            "mark 8..14 starts before the mark before it ends, at 12",
        ),
        (
            vec![8..12, 4..6],
            "mark 4..6 starts before the mark before it ends, at 12",
        ),
        (
            vec![4..12, 20..24],
            "mark 20..24 ends past the 23 completion bytes",
        ),
    ] {
        assert_eq!(
            parse_marked_completion_v1(&descriptor, text, &marks, PromptModeV1::Thinking)
                .unwrap_err(),
            sequence(refusal),
            "{marks:?}"
        );
        assert_eq!(
            thinking().push(input(completion, &marks)).unwrap_err(),
            sequence(refusal),
            "{marks:?}"
        );
    }
    // A kept mark that a later push leaves out or changes is refused; the
    // push that names it unchanged reads on.
    let mut parser = thinking();
    parser.push(input(&completion[..14], &[4..12])).unwrap();
    for marks in [vec![], vec![4..11], vec![3..12]] {
        assert_eq!(
            parser.push(input(completion, &marks)).unwrap_err(),
            sequence("the marks of the kept bytes are not the marks the parser kept"),
            "{marks:?}"
        );
    }
    let events = parser.push(input(completion, &[4..12])).unwrap();
    assert!(matches!(
        events.last(),
        Some(FormatEventV1::Finish { completion: Some(completion), .. })
            if *completion == answer(Some("plan"), "answer")
    ));
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

/// The vocabulary the fixture's grammars compile over: each byte as the
/// token [`BYTE`] plus its value, and `<think>` (5) and `|T|` ([`MARKER`])
/// as ordinary tokens, as a tokenizer may leave them, so their bytes alone
/// would fit wherever a grammar admits the text.
fn vocabulary() -> Arc<TokenVocabulary> {
    let mut tokens = vec![
        VocabularyToken {
            bytes: Vec::new(),
            special: false,
        };
        300
    ];
    for (id, text, special) in [
        (0, "<|s|>", true),
        (1, "[EOS]", true),
        (2, "</think>", false),
        (3, "<|u|>", true),
        (4, "<|a|>", true),
        (5, "<think>", false),
        (MARKER, "|T|", false),
    ] {
        tokens[id as usize] = VocabularyToken {
            bytes: text.as_bytes().to_vec(),
            special,
        };
    }
    for byte in 0..=255_u8 {
        tokens[(BYTE + u32::from(byte)) as usize] = VocabularyToken {
            bytes: vec![byte],
            special: false,
        };
    }
    Arc::new(TokenVocabulary::new(300, tokens).unwrap())
}

const BYTE: u32 = 16;
const MARKER: u32 = 6;

/// A completion written as [`marked`] writes it, as tokens: each marked
/// spelling the marker token, every other byte its own token.
fn tokens(written: &str) -> Vec<u32> {
    let (text, marks) = marked(written);
    let mut tokens = Vec::new();
    let mut at = 0;
    while at < text.len() {
        if let Some(mark) = marks.iter().find(|mark| mark.start == at) {
            assert_eq!(&text[mark.clone()], "|T|", "{written:?}");
            tokens.push(MARKER);
            at = mark.end;
        } else {
            tokens.push(BYTE + u32::from(text.as_bytes()[at]));
            at += 1;
        }
    }
    tokens
}

fn compiled(tools: &[FrontendToolV1], choice: &FrontendToolChoiceV1) -> Arc<CompiledGrammar> {
    let grammar = tool_constraint_v1(&descriptor(), tools, choice)
        .unwrap()
        .expect("the choice constrains");
    // The grammar crosses the bundle boundary as it is.
    assert_eq!(
        Grammar::decode_wire(&grammar.encode_wire()).unwrap(),
        grammar
    );
    Arc::new(CompiledGrammar::new(&grammar, vocabulary()).unwrap())
}

/// How many of `tokens` the grammar admits in turn, each also in the mask
/// before it, and whether it then matches whole.
fn admitted(compiled: &Arc<CompiledGrammar>, tokens: &[u32]) -> (usize, bool) {
    let mut matcher = GrammarMatcher::new(Arc::clone(compiled));
    let mut mask = vec![0; compiled.vocabulary().mask_words()];
    for (index, &token) in tokens.iter().enumerate() {
        matcher.fill_token_bitmask(&mut mask).unwrap();
        let masked = mask[token as usize / 32] & (1 << (token % 32)) != 0;
        let accepted = matcher.accept_token(token).is_ok();
        assert_eq!(masked, accepted, "token {index}: {token}");
        if !accepted {
            return (index, false);
        }
    }
    (tokens.len(), matcher.is_complete())
}

fn admits(compiled: &Arc<CompiledGrammar>, written: &str) -> bool {
    let tokens = tokens(written);
    admitted(compiled, &tokens) == (tokens.len(), true)
}

fn string_tool(strict: Option<bool>) -> FrontendToolV1 {
    FrontendToolV1 {
        name: "f".into(),
        description: None,
        parameters: serde_json::from_value(json!({
            "type": "object", "properties": {"x": {"type": "string"}},
            "required": ["x"], "additionalProperties": false
        }))
        .unwrap(),
        parameters_present: true,
        strict,
    }
}

/// One call to `f` with the raw value `value`, its markers' `|T|` marked.
fn forced(value: &str) -> String {
    format!(
        "\n\n<⟦|T|⟧ calls>\n<⟦|T|⟧ invoke name=\"f\">\n\
         <⟦|T|⟧ parameter name=\"x\" string=\"true\">{value}</⟦|T|⟧ parameter>\n\
         </⟦|T|⟧ invoke>\n</⟦|T|⟧ calls>"
    )
}

/// A forced call's grammar writes each marker's ID-only spelling as the
/// marker token: the call with marker tokens is admitted and reads back as
/// the call, and the same call with the markers spelled by ordinary tokens,
/// which the parser reads as text, is refused at the first of them.
#[test]
fn a_forced_call_admits_its_markers_only_as_tokens() {
    let tools = [string_tool(None)];
    for choice in [
        FrontendToolChoiceV1::Required,
        FrontendToolChoiceV1::Named("f".into()),
    ] {
        let compiled = compiled(&tools, &choice);
        let written = forced("v");
        assert!(admits(&compiled, &written), "{choice:?}");
        assert_eq!(
            parse(&format!("{written}[EOS]"), PromptModeV1::Chat)
                .unwrap()
                .tool_calls,
            [call("f", r#"{"x": "v"}"#)]
        );
        let spelled = written.replace(['⟦', '⟧'], "");
        assert_eq!(
            admitted(&compiled, &tokens(&spelled)),
            ("\n\n<".len(), false),
            "{choice:?}"
        );
        assert!(
            parse(&format!("{spelled}[EOS]"), PromptModeV1::Chat)
                .unwrap()
                .tool_calls
                .is_empty()
        );
    }
}

/// A raw value may spell any marker with ordinary tokens, the parameter
/// close included, and reads back holding it; the marker token itself is
/// refused inside the value.
#[test]
fn a_raw_value_may_spell_markers_with_ordinary_tokens() {
    let compiled = compiled(&[string_tool(None)], &FrontendToolChoiceV1::Required);
    let value = "a</|T| parameter>\n|T| b";
    let written = forced(value);
    assert!(admits(&compiled, &written));
    assert_eq!(
        parse(&format!("{written}[EOS]"), PromptModeV1::Chat)
            .unwrap()
            .tool_calls,
        [call("f", r#"{"x": "a</|T| parameter>\n|T| b"}"#)]
    );
    assert!(!admits(&compiled, &forced("a⟦|T|⟧b")));
}

/// Before a strict call under `auto`, free text may spell the calls open
/// with ordinary tokens, as the parser reads it as text; the marker token
/// opens the call.
#[test]
fn free_text_before_a_strict_call_may_spell_the_open() {
    let compiled = compiled(&[string_tool(Some(true))], &FrontendToolChoiceV1::Auto);
    let text = "Say <|T| calls> plainly.";
    assert!(admits(&compiled, text));
    assert!(!admits(&compiled, "Say ⟦|T|⟧ here."));
    let written = format!("{text}{}", forced("v"));
    assert!(admits(&compiled, &written));
    let read = parse(&format!("{written}[EOS]"), PromptModeV1::Chat).unwrap();
    assert_eq!(read.content, text);
    assert_eq!(read.tool_calls, [call("f", r#"{"x": "v"}"#)]);
}

/// A forced choice's prefill the prompt ends with holds the marker token,
/// which the constraint reads as that token; read as text, the prefill
/// does not continue the grammar.
#[test]
fn the_forced_prefill_reads_its_marker_as_the_token() {
    let compiled = compiled(&[string_tool(None)], &FrontendToolChoiceV1::Required);
    let prefill = [
        PrefixPart::Text(b"\n\n<"),
        PrefixPart::Token(MARKER, b"|T|"),
        PrefixPart::Text(b" calls>\n"),
    ];
    let mut state = ConstraintState::new(Arc::clone(&compiled), None, vec![1])
        .with_prefix_parts(&prefill)
        .unwrap();
    let body = forced("v");
    let body = &body[body.find("<⟦|T|⟧ invoke").unwrap()..];
    for token in tokens(body) {
        state.accept(token).unwrap();
    }
    assert!(state.allows(1).unwrap());
    assert!(
        ConstraintState::new(compiled, None, vec![1])
            .with_prefix(b"\n\n<|T| calls>\n")
            .is_err()
    );
}

/// A string enumeration value that spells the parameter close with
/// ordinary tokens reads back whole under ID-only markers, so the grammar
/// offers it, written as ordinary tokens.
#[test]
fn an_enumeration_value_spelling_a_marker_reads_back() {
    let value = "a</|T| parameter>\nb";
    let tool = FrontendToolV1 {
        parameters: serde_json::from_value(json!({
            "type": "object", "properties": {"x": {"type": "string", "enum": [value]}},
            "required": ["x"], "additionalProperties": false
        }))
        .unwrap(),
        ..string_tool(None)
    };
    let compiled = compiled(&[tool], &FrontendToolChoiceV1::Required);
    let written = forced(value);
    assert!(admits(&compiled, &written));
    assert!(!admits(&compiled, &forced("a</⟦|T|⟧ parameter>\nb")));
    assert_eq!(
        parse(&format!("{written}[EOS]"), PromptModeV1::Chat)
            .unwrap()
            .tool_calls,
        [call("f", r#"{"x": "a</|T| parameter>\nb"}"#)]
    );
}

/// A declaration without ID-only markers writes its grammar with no token
/// element, so its grammar document is the layout it always was.
#[test]
fn without_id_only_markers_the_grammar_names_no_tokens() {
    let plain =
        FormatDescriptorV1::from_component_bytes(common::TYPED_PARAMETERS.as_bytes()).unwrap();
    for choice in [FrontendToolChoiceV1::Required, FrontendToolChoiceV1::Auto] {
        let grammar = tool_constraint_v1(&plain, &[string_tool(Some(true))], &choice)
            .unwrap()
            .unwrap();
        assert!(grammar.named_tokens().is_empty());
        assert_eq!(
            grammar.encode_wire()[8..12],
            GRAMMAR_WIRE_LAYOUT_V1.to_le_bytes()
        );
    }
    let grammar = tool_constraint_v1(
        &descriptor(),
        &[string_tool(None)],
        &FrontendToolChoiceV1::Required,
    )
    .unwrap()
    .unwrap();
    assert_eq!(grammar.named_tokens(), [MARKER]);
}
