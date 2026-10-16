#![forbid(unsafe_code)]
//! Request numbers cross the documents as their JSON text, and a format
//! writes every digit back into the prompt: here the fixture format, which
//! writes each offered tool as Python-style JSON.
#[allow(dead_code)]
mod common;

use frontend_format::chat_format_export_v1;
use frontend_format::test_support::FixtureFormatV1;
use runtime_bundle_abi::frontend_document as document;

#[test]
fn request_numbers_render_with_every_digit() {
    let (frontend, _) = chat_format_export_v1::<FixtureFormatV1>()
        .open(common::FUNCTION_PARAMETERS.as_bytes())
        .unwrap();
    let request = document::RenderRequestV1 {
        messages: vec![document::MessageV1 {
            role: "user".into(),
            content: Some("hi".into()),
            parts: None,
            reasoning: None,
            reasoning_content: None,
            tool_calls: vec![],
            tool_call_id: None,
            tools: vec![],
        }],
        continue_final_message: false,
        add_generation_prompt: true,
        tools: vec![document::ToolV1 {
            name: "lookup".into(),
            description: None,
            parameters: concat!(
                r#"{"type":"object","properties":{"#,
                r#""n":{"type":"integer","maximum":18446744073709551617},"#,
                r#""r":{"type":"number","default":1e-7,"maximum":1.7976931348623157e308}}}"#
            )
            .into(),
            parameters_present: true,
            strict: None,
        }],
        tool_objects: vec![],
        response_format: None,
        reasoning_effort: None,
        chat_template_kwargs: None,
    };
    let rendered = frontend.render(&request.encode(), 1 << 20).unwrap();
    let rendered = document::RenderedV1::decode(&rendered).unwrap();
    let [document::SegmentV1::Text(prompt)] = rendered.segments.as_slice() else {
        panic!("one text segment: {:?}", rendered.segments);
    };
    for written in [
        r#""maximum": 18446744073709551617"#,
        r#""default": 1e-07"#,
        r#""maximum": 1.7976931348623157e+308"#,
    ] {
        assert!(prompt.contains(written), "{written} in {prompt}");
    }
}
