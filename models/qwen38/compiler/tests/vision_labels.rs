//! Qwen3.8's numbered media labels: with `add_vision_id` set by the
//! template's truth test, each image is written after `Picture N: `, counted
//! through the whole conversation (template line 13) and from one again in
//! every render. The labels are ordinary text, so client text still may not
//! carry a pad or a media placeholder beside them.
use model_qwen38_contract::frontend::Qwen38Format;
use runtime_frontend::test_support::documents::PairedFrontendV1;
use runtime_server::{
    FrontendChatRequestV1, FrontendContentPartV1, FrontendJsonV1, FrontendMediaPartKindV1,
    FrontendMediaPartV1, FrontendMessageV1, FrontendRoleV1, FrontendToolChoiceV1, ServerFrontendV1,
};

/// The frontend of a package with the encoder images, through its chat
/// format called directly and through its export.
fn frontend() -> PairedFrontendV1 {
    PairedFrontendV1::chat_format::<Qwen38Format>(model_qwen38::frontend::multimodal_descriptor())
        .unwrap()
}

fn image() -> FrontendContentPartV1 {
    FrontendContentPartV1::Media(FrontendMediaPartV1 {
        kind: FrontendMediaPartKindV1::Image,
        part_type: "image_url".into(),
        source: "data:image/png;base64,AAAA".into(),
        format: None,
    })
}

fn text(text: &str) -> FrontendContentPartV1 {
    FrontendContentPartV1::Text(text.into())
}

/// One user turn for each entry of `turns`, asking for a generation.
fn request(
    turns: Vec<Vec<FrontendContentPartV1>>,
    add_vision_id: Option<FrontendJsonV1>,
) -> FrontendChatRequestV1 {
    FrontendChatRequestV1 {
        messages: turns
            .into_iter()
            .map(|parts| FrontendMessageV1 {
                role: FrontendRoleV1::User,
                content: None,
                parts: Some(parts),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tools: Vec::new(),
            })
            .collect(),
        continue_final_message: false,
        add_generation_prompt: true,
        tools: Vec::new(),
        tool_choice: FrontendToolChoiceV1::None,
        response_format: None,
        reasoning_effort: None,
        enable_thinking: None,
        thinking: None,
        nested_reasoning_effort: None,
        preserve_thinking: None,
        clear_thinking: None,
        add_vision_id,
        tool_objects: Vec::new(),
        chat_template_kwargs: None,
    }
}

/// An image in each of two turns.
fn two_turns(add_vision_id: Option<FrontendJsonV1>) -> FrontendChatRequestV1 {
    request(
        vec![vec![image(), text("first")], vec![image(), text("last")]],
        add_vision_id,
    )
}

#[test]
fn images_count_through_the_conversation_and_afresh_for_each_render() {
    let frontend = frontend();
    let plain = frontend.render_chat(two_turns(None)).unwrap();
    assert!(!plain.prompt.contains("Picture "), "{}", plain.prompt);
    let numbered = frontend
        .render_chat(two_turns(Some(FrontendJsonV1::Bool(true))))
        .unwrap();
    let first = numbered
        .prompt
        .find("Picture 1: <|vision_start|>")
        .expect("the first image is numbered one");
    let second = numbered
        .prompt
        .find("Picture 2: <|vision_start|>")
        .expect("the second turn's image is numbered two");
    assert!(first < second);
    // The labels are all the keyword adds.
    assert_eq!(
        numbered
            .prompt
            .replacen("Picture 1: ", "", 1)
            .replacen("Picture 2: ", "", 1),
        plain.prompt
    );
    assert_eq!(numbered.media, plain.media);
    assert_eq!(numbered.media.len(), 2);
    // The next render counts from one again.
    let again = frontend
        .render_chat(two_turns(Some(FrontendJsonV1::Bool(true))))
        .unwrap();
    assert_eq!(again.prompt, numbered.prompt);
    assert_eq!(again.segments, numbered.segments);
}

#[test]
fn the_keyword_is_read_by_the_template_truth_test() {
    let frontend = frontend();
    let plain = frontend.render_chat(two_turns(None)).unwrap();
    let numbered = frontend
        .render_chat(two_turns(Some(FrontendJsonV1::Bool(true))))
        .unwrap();
    for (value, labelled) in [
        ("null", false),
        ("false", false),
        ("0", false),
        ("-0.0", false),
        (r#""""#, false),
        ("[]", false),
        ("{}", false),
        ("1", true),
        ("-2", true),
        (r#""false""#, true),
        ("[false]", true),
        (r#"{"x":null}"#, true),
    ] {
        let rendered = frontend
            .render_chat(two_turns(Some(serde_json::from_str(value).unwrap())))
            .unwrap();
        let expected = if labelled { &numbered } else { &plain };
        assert_eq!(rendered.prompt, expected.prompt, "{value}");
        assert_eq!(rendered.segments, expected.segments, "{value}");
    }
}

#[test]
fn labelled_media_keeps_the_client_text_refusals() {
    let frontend = frontend();
    let numbered = || Some(FrontendJsonV1::Bool(true));
    for reserved in ["<|image_pad|>", "\u{e000}chopinfer-media:0:\u{e001}"] {
        let error = frontend
            .render_chat(request(
                vec![vec![image(), text("first"), text(reserved)]],
                numbered(),
            ))
            .unwrap_err();
        assert_eq!(error.parameter.as_deref(), Some("messages"), "{reserved}");
    }
    // Two text parts render as one string, so a pad split across them is
    // refused too, before and after a labelled image.
    for parts in [
        vec![text("<|image"), text("_pad|>"), image()],
        vec![image(), text("<|image"), text("_pad|>")],
    ] {
        let error = frontend
            .render_chat(request(vec![parts], numbered()))
            .unwrap_err();
        assert!(
            error.message.contains("reserved media string"),
            "{}",
            error.message
        );
    }
}
