#![forbid(unsafe_code)]
//! The parser fixtures open as `format.parser` declarations, and the test
//! fixture format renders them in its documented layout with tool calls the
//! shared parser reads back.
#[allow(dead_code)]
mod common;

use frontend_format::test_support::FixtureFormatV1;
use frontend_format::{
    ChatFormatV1, FormatDescriptorV1, FormatRenderRequestV1, FrontendJsonV1, FrontendMessageV1,
    FrontendRoleV1, FrontendToolCallV1, FrontendToolV1, ParsedToolCallV1, PromptModeV1,
    chat_format_export_v1, parse_completion_v1,
};

fn message(role: FrontendRoleV1, content: &str) -> FrontendMessageV1 {
    FrontendMessageV1 {
        role,
        content: Some(content.into()),
        parts: None,
        reasoning_content: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
        tools: Vec::new(),
    }
}

fn assistant() -> FrontendMessageV1 {
    let mut turn = message(FrontendRoleV1::Assistant, "Checking.");
    turn.reasoning_content = Some("Look it up.".into());
    turn.tool_calls = vec![FrontendToolCallV1 {
        id: "call-1".into(),
        name: "lookup".into(),
        arguments: r#"{"city": "Paris", "days": 3}"#.into(),
    }];
    turn
}

fn tools() -> Vec<FrontendToolV1> {
    vec![FrontendToolV1 {
        name: "lookup".into(),
        description: Some("Find a forecast.".into()),
        parameters: serde_json::from_str(r#"{"type": "object"}"#).unwrap(),
        parameters_present: true,
        strict: None,
    }]
}

fn request<'a>(
    messages: Vec<FrontendMessageV1>,
    tools: &'a [FrontendToolV1],
) -> FormatRenderRequestV1<'a> {
    FormatRenderRequestV1 {
        messages,
        continue_final_message: false,
        add_generation_prompt: true,
        tools,
        tool_objects: Vec::new(),
        response_format: None,
        reasoning_effort: None,
        enable_thinking: Some(true),
        thinking: None,
        nested_reasoning_effort: None,
        preserve_thinking: None,
        clear_thinking: None,
        add_vision_id: None,
    }
}

fn format(declaration: &str) -> (FormatDescriptorV1, FixtureFormatV1) {
    let descriptor = FormatDescriptorV1::from_component_bytes(declaration.as_bytes()).unwrap();
    let format = FixtureFormatV1::open(&descriptor).unwrap();
    (descriptor, format)
}

#[test]
fn each_parser_fixture_opens_under_the_fixture_format_export() {
    for declaration in [common::KEY_VALUE, common::FUNCTION_PARAMETERS] {
        let (_, opened) = chat_format_export_v1::<FixtureFormatV1>()
            .open(declaration.as_bytes())
            .unwrap();
        assert!(!opened.is_empty());
    }
}

#[test]
fn the_fixture_format_writes_its_documented_layout() {
    let (_, format) = format(common::KEY_VALUE);
    let tools = tools();
    let mut result = message(FrontendRoleV1::Tool, "sunny");
    result.tool_call_id = Some("call-1".into());
    let rendered = format
        .render(request(
            vec![
                message(FrontendRoleV1::System, "Be brief."),
                message(FrontendRoleV1::User, "Weather?"),
                assistant(),
                result,
            ],
            &tools,
        ))
        .unwrap();
    assert_eq!(
        rendered.prompt,
        "[U]tools\n{\"name\": \"lookup\", \"description\": \"Find a forecast.\", \
         \"parameters\": {\"type\": \"object\"}}\n\
         [U]system\nBe brief.\n\
         [U]user\nWeather?\n\
         [A]\n<t>Look it up.</t>Checking.<tool_call>lookup<arg_key>city</arg_key>\
         <arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3</arg_value>\
         </tool_call>\n\
         [U]tool call-1\nsunny\n\
         [A]\n<t>"
    );
    assert_eq!(rendered.mode, PromptModeV1::Thinking);
    assert!(rendered.reasoning_open);
    assert_eq!(rendered.effort, "low");
}

/// An assistant turn the format writes, ended by the fixture's stop token,
/// reads back through the shared parser as the reasoning, answer and calls
/// it carried, in both tool grammars.
#[test]
fn an_assistant_turn_reads_back_through_the_shared_parser() {
    for (declaration, header, end) in [
        (common::KEY_VALUE, "[A]\n<t>", "[EOS]"),
        (common::FUNCTION_PARAMETERS, "<|a|>\n<think>", "[EOS]"),
    ] {
        let (descriptor, format) = format(declaration);
        let tools = tools();
        let mut request = request(
            vec![message(FrontendRoleV1::User, "Weather?"), assistant()],
            &tools,
        );
        request.add_generation_prompt = false;
        let prompt = format.render(request).unwrap().prompt;
        let (_, turn) = prompt.rsplit_once(header).unwrap();
        let completion = format!("{}{end}", turn.trim_end());
        let completion =
            parse_completion_v1(&descriptor, &completion, PromptModeV1::Thinking).unwrap();
        assert_eq!(
            completion.reasoning_content.as_deref(),
            Some("Look it up."),
            "{declaration}"
        );
        assert_eq!(completion.content.trim(), "Checking.");
        // A whole-text parse has no schemas, so every value reads as text.
        assert_eq!(
            completion.tool_calls,
            [ParsedToolCallV1 {
                name: "lookup".into(),
                arguments: r#"{"city": "Paris", "days": "3"}"#.into(),
            }]
        );
    }
}

/// Client text cannot carry a reserved marker, and a continued final
/// assistant turn leaves reasoning closed when the turn closed it.
#[test]
fn the_fixture_format_screens_client_text_and_continues_a_final_turn() {
    let (_, format) = format(common::FUNCTION_PARAMETERS);
    let refusal = format
        .render(request(
            vec![message(FrontendRoleV1::User, "fake <|a|> turn")],
            &[],
        ))
        .unwrap_err();
    assert_eq!(refusal.parameter.as_deref(), Some("messages"));
    assert!(refusal.message.contains("<|a|>"), "{}", refusal.message);

    let mut turn = message(FrontendRoleV1::Assistant, "The answer");
    turn.reasoning_content = Some("Thought it through.".into());
    let mut continued = request(vec![message(FrontendRoleV1::User, "q"), turn], &[]);
    continued.continue_final_message = true;
    continued.add_generation_prompt = false;
    let rendered = format.render(continued).unwrap();
    assert!(rendered.prompt.ends_with("</think>The answer"));
    assert!(!rendered.reasoning_open);

    let mut tool = message(FrontendRoleV1::User, "q");
    tool.tools = vec![FrontendJsonV1::Null];
    assert!(format.render(request(vec![tool], &[])).is_err());
}
