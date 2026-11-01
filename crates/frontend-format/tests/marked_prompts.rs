#![forbid(unsafe_code)]
//! A prompt that names its special tokens: the marked prompt a format
//! writes under `vocabulary.explicit_special_tokens`, its split into text,
//! media and special segments, and the media pads client text may spell
//! there.
#[allow(dead_code)]
mod common;

use frontend_format::toolkit::{
    MarkedPromptV1, append_v1, media_placeholder_v1, reserved_media_text_v1,
    split_marked_media_segments_v1, split_media_segments_v1,
};
use frontend_format::{
    FormatDescriptorV1, FrontendMediaPartKindV1, FrontendMediaPartV1, PromptSegmentV1,
};
use serde_json::{Value, json};

const MAXIMUM: usize = 1 << 20;

/// The key-value fixture with one image modality, whose pad `[I]` is a
/// prompt special token, and explicit special tokens when `explicit`.
fn descriptor(explicit: bool) -> FormatDescriptorV1 {
    let mut declaration: Value = serde_json::from_str(common::KEY_VALUE).unwrap();
    let vocabulary = &mut declaration["vocabulary"];
    vocabulary["boundary_tokens"]
        .as_array_mut()
        .unwrap()
        .push(json!({"text": "[I]", "id": 6}));
    vocabulary["prompt_special_tokens"]
        .as_array_mut()
        .unwrap()
        .push(json!("[I]"));
    vocabulary["explicit_special_tokens"] = json!(explicit);
    declaration["multimodal"] = json!({
        "arena": {"rows": 64, "page_rows": 64, "hidden": 8, "element": "bf16"},
        "encoders": [{"name": "image", "payload_element": "bf16", "payload_columns": 4,
            "maximum_payload_bytes": 64, "cost_per_row_milli": 1}],
        "modalities": [{"kind": "image", "part_types": ["image_url"], "maximum_items": 4,
            "maximum_bytes": 64, "maximum_tokens": 8, "encoder": 0,
            "recipe": {"tag": "image.patch-grid.v1", "parameters": {}},
            "render": {"pad": "[I]"}}]
    });
    FormatDescriptorV1::from_declaration_bytes(&serde_json::to_vec(&declaration).unwrap()).unwrap()
}

fn image(index: usize) -> FrontendMediaPartV1 {
    FrontendMediaPartV1 {
        kind: FrontendMediaPartKindV1::Image,
        part_type: "image_url".into(),
        source: index.to_string(),
        format: None,
    }
}

fn text(text: &str) -> PromptSegmentV1 {
    PromptSegmentV1::Text(text.into())
}

fn special(text: &str) -> PromptSegmentV1 {
    PromptSegmentV1::Special(text.into())
}

/// Writes `pieces` to a marked prompt: `Ok` plain text, `Err` a special
/// token.
fn marked(pieces: &[Result<&str, &str>]) -> MarkedPromptV1 {
    let mut prompt = MarkedPromptV1::new();
    for piece in pieces {
        match piece {
            Ok(text) => prompt.push_plain(text, MAXIMUM).unwrap(),
            Err(token) => prompt.push_special(token, MAXIMUM).unwrap(),
        }
    }
    prompt
}

#[test]
fn a_marked_prompt_charges_the_byte_bound_and_records_its_special_tokens() {
    let mut prompt = MarkedPromptV1::new();
    prompt.push_special("[U]", 16).unwrap();
    prompt.push_plain("hi [A]", 16).unwrap();
    let mut turn = MarkedPromptV1::new();
    turn.push_special("[A]", 16).unwrap();
    turn.push_plain("\n", 16).unwrap();
    prompt.push_prompt(&turn, 16).unwrap();
    assert_eq!(prompt.text(), "[U]hi [A][A]\n");
    assert_eq!(prompt.specials(), [0..3, 9..12]);

    // Each push refuses a prompt past the bound as `append_v1` does, and
    // writes nothing then.
    let mut appended = prompt.text().to_owned();
    let expected = append_v1(&mut appended, "1234", 16).unwrap_err();
    for push in [
        MarkedPromptV1::push_plain as fn(&mut MarkedPromptV1, &str, usize) -> _,
        MarkedPromptV1::push_special,
    ] {
        let mut longer = prompt.clone();
        assert_eq!(push(&mut longer, "1234", 16).unwrap_err(), expected);
        assert_eq!(longer, prompt);
    }
    let mut longer = prompt.clone();
    let mut piece = MarkedPromptV1::new();
    piece.push_special("1234", 16).unwrap();
    assert_eq!(longer.push_prompt(&piece, 16).unwrap_err(), expected);
    assert_eq!(longer, prompt);

    assert_eq!(
        MarkedPromptV1::new().push_special("", 16).unwrap_err(),
        "invalid conversation field: a prompt special token cannot be empty"
    );
}

#[test]
fn a_marked_prompt_splits_into_special_text_and_media_segments() {
    let descriptor = descriptor(true);
    let placeholder = |index| media_placeholder_v1(index);
    // Client text that spells a special token stays in its text segment.
    let prompt = marked(&[
        Err("[U]"),
        Ok("user\nlook "),
        Ok(&placeholder(0)),
        Ok(" [A] and "),
        Ok("more\n"),
        Err("[A]"),
        Err("<t>"),
    ]);
    let (padded, segments, media) =
        split_marked_media_segments_v1(&descriptor, prompt, vec![image(0)]).unwrap();
    assert_eq!(padded, "[U]user\nlook [I] [A] and more\n[A]<t>");
    assert_eq!(
        segments,
        [
            special("[U]"),
            text("user\nlook "),
            PromptSegmentV1::Media(0),
            text(" [A] and more\n"),
            special("[A]"),
            special("<t>"),
        ]
    );
    assert_eq!(media, [image(0)]);

    // A media segment keeps a text segment on each side, empty or not; a
    // special segment gets none it does not need.
    let prompt = marked(&[
        Err("[U]"),
        Ok(&placeholder(0)),
        Err("[A]"),
        Ok(&placeholder(1)),
    ]);
    let (padded, segments, _) =
        split_marked_media_segments_v1(&descriptor, prompt, vec![image(0), image(1)]).unwrap();
    assert_eq!(padded, "[U][I][A][I]");
    assert_eq!(
        segments,
        [
            special("[U]"),
            text(""),
            PromptSegmentV1::Media(0),
            text(""),
            special("[A]"),
            text(""),
            PromptSegmentV1::Media(1),
            text(""),
        ]
    );
    let (_, segments, _) =
        split_marked_media_segments_v1(&descriptor, MarkedPromptV1::new(), Vec::new()).unwrap();
    assert_eq!(segments, [text("")]);
}

/// A prompt with no special token splits exactly as the text splitter
/// splits its text.
#[test]
fn a_prompt_without_special_tokens_splits_as_text_does() {
    let descriptor = descriptor(false);
    let placeholder = |index| media_placeholder_v1(index);
    for (pieces, media) in [
        (vec!["[U]user\nhi\n"], 0),
        (vec![""], 0),
        (vec!["[U]", &placeholder(0), "x", &placeholder(1)], 2),
        (vec![&placeholder(0)], 1),
    ] {
        let media: Vec<_> = (0..media).map(image).collect();
        let pieces: Vec<_> = pieces.into_iter().map(Ok).collect();
        let expected = split_media_segments_v1(
            &descriptor,
            pieces.iter().flatten().copied().collect(),
            media.clone(),
        );
        assert_eq!(
            split_marked_media_segments_v1(&descriptor, marked(&pieces), media),
            expected,
            "{pieces:?}"
        );
    }
}

#[test]
fn a_marked_split_refuses_what_the_prompt_cannot_carry() {
    let descriptor = descriptor(true);
    let placeholder = media_placeholder_v1(0);
    let refusal = |prompt: MarkedPromptV1, media: usize| {
        split_marked_media_segments_v1(&descriptor, prompt, (0..media).map(image).collect())
            .unwrap_err()
    };
    assert_eq!(
        refusal(marked(&[Err("[U]"), Err("<x>")]), 0),
        "invalid conversation field: rendered special token '<x>' is not a declared boundary \
         token"
    );
    // A placeholder lies whole in plain text.
    let (open, close) = placeholder.split_at(placeholder.find('0').unwrap());
    assert_eq!(
        refusal(marked(&[Ok(open), Err("[A]"), Ok(close)]), 1),
        "invalid conversation field: media render boundary was lost"
    );
    for (pieces, media) in [(vec![Ok(placeholder.as_str())], 0), (vec![Ok("x")], 1)] {
        assert_eq!(
            refusal(marked(&pieces), media),
            "invalid conversation field: unexpected media boundary in rendered prompt"
        );
    }
    // The prompt with pads must fit the bound.
    let prompt = marked(&[Err("[U]"), Ok(&placeholder), Ok("tail")]);
    let mut bounded = descriptor.clone();
    bounded.limits.maximum_prompt_bytes = "[U][I]tail".len();
    assert!(split_marked_media_segments_v1(&bounded, prompt.clone(), vec![image(0)]).is_ok());
    bounded.limits.maximum_prompt_bytes -= 1;
    assert_eq!(
        split_marked_media_segments_v1(&bounded, prompt, vec![image(0)]).unwrap_err(),
        "conversation resource limit: rendered media prompt exceeds maximum_prompt_bytes"
    );
}

/// A media pad is reserved in client text unless client text is plain
/// text; a placeholder boundary is reserved either way.
#[test]
fn explicit_special_tokens_free_the_media_pads_in_client_text() {
    let placeholder = media_placeholder_v1(0);
    for (explicit, pad) in [(false, true), (true, false)] {
        let descriptor = descriptor(explicit);
        assert_eq!(reserved_media_text_v1(&descriptor, "a [I] b"), pad);
        assert!(reserved_media_text_v1(&descriptor, &placeholder));
        assert!(!reserved_media_text_v1(&descriptor, "a [A] b"));
    }
}
