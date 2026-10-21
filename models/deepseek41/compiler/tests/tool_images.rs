//! Tool images retain the vendor's prompt order through the host frontend.

use model_deepseek41::frontend::multimodal_descriptor;
use model_deepseek41_contract::frontend::DeepSeekV41Format;
use runtime_frontend::test_support::documents::PairedFrontendV1;
use runtime_server::{PreparedPrompt, PromptSegmentV1, prepare_chat_request_v1};
use serde_json::{Value, json};

fn image(value: &str) -> Value {
    json!({"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{value}")}})
}

fn history() -> Value {
    json!({"thinking": {"type": "disabled"}, "messages": [
        {"role": "user", "content": [image("AA==")]},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "look", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "look", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "b", "content": [
            {"type": "text", "text": "second"}, image("AQ==")]},
        {"role": "user", "content": [image("Ag==")]},
        {"role": "tool", "tool_call_id": "a", "content": [
            {"type": "text", "text": "first"}, image("Aw=="), image("BA=="),
            {"type": "text", "text": "end"}]}
    ]})
}

#[test]
fn tool_images_follow_sorted_prompt_order_in_the_host_media_list() {
    let frontend =
        PairedFrontendV1::chat_format::<DeepSeekV41Format>(multimodal_descriptor()).unwrap();
    let prepared = prepare_chat_request_v1(
        &frontend,
        1 << 20,
        history().as_object().unwrap().clone(),
        1024,
    )
    .unwrap();
    let PreparedPrompt::MediaChat { segments, media } = prepared.prompt else {
        panic!("tool images must reach media preparation");
    };
    let expected = ["AA==", "Aw==", "BA==", "Ag==", "AQ=="];
    assert_eq!(
        media
            .iter()
            .map(|item| item.source.as_str())
            .collect::<Vec<_>>(),
        expected.map(|value| format!("data:image/png;base64,{value}"))
    );
    let mut prompt = String::new();
    let mut indices = Vec::new();
    for segment in segments {
        match segment {
            PromptSegmentV1::Text(text) => prompt.push_str(&text),
            PromptSegmentV1::Media(index) => {
                indices.push(index);
                prompt.push_str(&format!("[{}]", expected[index]));
            }
        }
    }
    assert_eq!(indices, [0, 1, 2, 3, 4]);
    assert!(prompt.contains("<tool_result>first\n\n[Aw==]\n\n[BA==]\n\nend</tool_result>\n\n[Ag==]\n\n<tool_result>second\n\n[AQ==]</tool_result>"), "{prompt}");
}

#[test]
fn image_count_admission_includes_user_and_tool_images() {
    let mut descriptor = multimodal_descriptor();
    descriptor.multimodal.as_mut().unwrap().modalities[0].maximum_items = 4;
    let frontend = PairedFrontendV1::chat_format::<DeepSeekV41Format>(descriptor).unwrap();
    let error = prepare_chat_request_v1(
        &frontend,
        1 << 20,
        history().as_object().unwrap().clone(),
        1024,
    )
    .unwrap_err();
    assert_eq!(error.parameter.as_deref(), Some("image"));
    assert_eq!(
        error.message,
        "At most 4 image(s) may be provided in one request; this request carried 5."
    );
}

#[test]
fn tool_images_keep_text_screening_and_other_role_refusals() {
    let frontend =
        PairedFrontendV1::chat_format::<DeepSeekV41Format>(multimodal_descriptor()).unwrap();
    for role in ["assistant", "system", "developer"] {
        let body = json!({"messages": [{"role": role, "content": [image("AA==")]}]});
        let error =
            prepare_chat_request_v1(&frontend, 1 << 20, body.as_object().unwrap().clone(), 1024)
                .unwrap_err();
        assert_eq!(error.message, "only user messages may contain media");
    }
    for marker in ["<｜deepseek_image｜>", "\u{e000}chopinfer-media:0:\u{e001}"] {
        let mut body = history();
        body["messages"][2]["content"][0]["text"] = marker.into();
        let error =
            prepare_chat_request_v1(&frontend, 1 << 20, body.as_object().unwrap().clone(), 1024)
                .unwrap_err();
        assert!(
            error.message.contains("placeholder") || error.message.contains("media boundary"),
            "{}",
            error.message
        );
    }
}

/// GIF and WebP images, other spellings of the JPEG type and OpenAI `file`
/// parts render as the same segments PNG `image_url` parts do, in user and
/// tool messages, and keep their sources for media preparation in sorted
/// prompt order; other `file` shapes are refused by name.
#[test]
fn image_formats_and_file_parts_render_as_image_parts() {
    let frontend =
        PairedFrontendV1::chat_format::<DeepSeekV41Format>(multimodal_descriptor()).unwrap();
    let prepare = |body: &Value| {
        prepare_chat_request_v1(&frontend, 1 << 20, body.as_object().unwrap().clone(), 1024)
    };
    let segments_and_media = |body: &Value| {
        let PreparedPrompt::MediaChat { segments, media } = prepare(body).unwrap().prompt else {
            panic!("images must reach media preparation");
        };
        let media = media
            .into_iter()
            .map(|item| (item.part_type, item.source))
            .collect::<Vec<_>>();
        (segments, media)
    };
    let url = |mime: &str, data: &str| format!("data:{mime};base64,{data}");
    let image_url = |mime: &str, data: &str| json!({"type": "image_url", "image_url": {"url": url(mime, data)}});
    let file = |mime: &str, data: &str| json!({"type": "file", "file": {"file_data": url(mime, data), "filename": "a"}});
    let (png, _) = segments_and_media(&history());
    let mut body = history();
    body["messages"][0]["content"][0] = image_url("image/gif", "AA==");
    body["messages"][2]["content"][1] = file("image/webp", "AQ==");
    body["messages"][3]["content"][0] = image_url("image/JPG", "Ag==");
    body["messages"][4]["content"][1] = file("image/png", "Aw==");
    body["messages"][4]["content"][2] = image_url("image/jpg", "BA==");
    let (segments, media) = segments_and_media(&body);
    assert_eq!(segments, png);
    let expected = [
        ("image_url", url("image/gif", "AA==")),
        ("file", url("image/png", "Aw==")),
        ("image_url", url("image/jpg", "BA==")),
        ("image_url", url("image/JPG", "Ag==")),
        ("file", url("image/webp", "AQ==")),
    ]
    .map(|(part, source)| (part.to_owned(), source));
    assert_eq!(media, expected);

    for (part, message) in [
        (
            json!({"type": "file", "file": {"file_id": "file-1"}}),
            "file_id is not supported; pass the image as a base64 data URL in file_data",
        ),
        (
            json!({"type": "file", "file_data": url("image/png", "AA==")}),
            r#"a file part carries its fields in a "file" object: {"type": "file", "file": {"file_data": ...}}"#,
        ),
        (
            file("application/pdf", "AA=="),
            "unsupported media MIME type or invalid base64",
        ),
        (
            image_url("image/svg+xml", "AA=="),
            "unsupported media MIME type or invalid base64",
        ),
    ] {
        let body = json!({"messages": [{"role": "user", "content": [part]}]});
        assert_eq!(prepare(&body).unwrap_err().message, message);
    }
}
