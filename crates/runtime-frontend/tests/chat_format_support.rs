#![forbid(unsafe_code)]
//! The test support serves a model's own chat format. A `format.parser`
//! declaration pairs a chat format called directly with its export
//! ([`PairedFrontendV1::chat_format`]), and its decoder pair parses through
//! a chat format's export, so the shared decode helpers run over it
//! unchanged. These run on the function-parameter fixture, whose parser
//! section reads the shared function-parameter cases.
use frontend_format::test_support::FixtureFormatV1;
use frontend_format::test_support::fixtures::FUNCTION_PARAMETERS;
use runtime_frontend::test_support::decode::{Calls, assert_call_splits, deliver};
use runtime_frontend::test_support::documents::PairedFrontendV1;
use runtime_frontend::{
    BundleParserSourceV1, FrontendDescriptorV1, PromptModeV1, parse_completion_v1,
};
use runtime_server::{
    FrontendChatRequestV1, FrontendMessageV1, FrontendRoleV1, FrontendToolCallV1,
    FrontendToolChoiceV1, FrontendToolV1, ServerFrontendV1,
};
use serde_json::Value;

use frontend_format::test_support::FUNCTION_PARAMETER_CASES as CASES;

fn cases() -> Value {
    serde_json::from_str(CASES).unwrap()
}

fn parser_declaration() -> FrontendDescriptorV1 {
    FrontendDescriptorV1::from_component_bytes(FUNCTION_PARAMETERS.as_bytes()).unwrap()
}

fn tools() -> Vec<FrontendToolV1> {
    cases()["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| {
            let function = &tool["function"];
            FrontendToolV1 {
                name: function["name"].as_str().unwrap().to_owned(),
                description: function["description"].as_str().map(str::to_owned),
                parameters: serde_json::from_value(function["parameters"].clone()).unwrap(),
                parameters_present: !function["parameters"].is_null(),
                strict: function["strict"].as_bool(),
            }
        })
        .collect()
}

fn message(role: FrontendRoleV1, content: &str) -> FrontendMessageV1 {
    FrontendMessageV1 {
        role,
        content: Some(content.to_owned()),
        parts: None,
        reasoning_content: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
        tools: Vec::new(),
    }
}

fn chat(messages: Vec<FrontendMessageV1>) -> FrontendChatRequestV1 {
    FrontendChatRequestV1 {
        messages,
        continue_final_message: false,
        add_generation_prompt: true,
        tools: Vec::new(),
        tool_choice: FrontendToolChoiceV1::Auto,
        response_format: None,
        reasoning_effort: None,
        enable_thinking: None,
        thinking: None,
        nested_reasoning_effort: None,
        preserve_thinking: None,
        clear_thinking: None,
        add_vision_id: None,
        tool_objects: Vec::new(),
        chat_template_kwargs: None,
    }
}

fn requests() -> Vec<FrontendChatRequestV1> {
    let tools = tools();
    let name = tools[0].name.clone();
    let mut call = message(FrontendRoleV1::Assistant, "");
    call.reasoning_content = Some("Look it up.".into());
    call.tool_calls = vec![FrontendToolCallV1 {
        id: "call_0".into(),
        name,
        arguments: "{}".into(),
    }];
    let mut result = message(FrontendRoleV1::Tool, "found");
    result.tool_call_id = Some("call_0".into());
    let mut continued = chat(vec![
        message(FrontendRoleV1::User, "Hi"),
        message(FrontendRoleV1::Assistant, "Hel"),
    ]);
    continued.continue_final_message = true;
    continued.add_generation_prompt = false;
    let mut chat_mode = chat(vec![message(FrontendRoleV1::User, "Hi")]);
    chat_mode.enable_thinking = Some(false);
    let mut tool_turn = chat(vec![
        message(FrontendRoleV1::System, "Be brief."),
        message(FrontendRoleV1::User, "Find it."),
        call,
        result,
    ]);
    tool_turn.tools = tools;
    vec![
        chat(vec![message(FrontendRoleV1::User, "Hi")]),
        chat_mode,
        continued,
        tool_turn,
    ]
}

/// Every render and tool constraint agrees on both paths; the pair asserts
/// it on each call.
#[test]
fn a_chat_format_pairs_with_its_export() {
    let format = PairedFrontendV1::chat_format::<FixtureFormatV1>(parser_declaration()).unwrap();
    for request in requests() {
        format.render_chat(request).unwrap();
    }
    let tools = tools();
    for (choice, constrained) in [
        (FrontendToolChoiceV1::Auto, false),
        (FrontendToolChoiceV1::None, false),
        (FrontendToolChoiceV1::Required, true),
        (FrontendToolChoiceV1::Named(tools[0].name.clone()), true),
    ] {
        assert_eq!(
            format.tool_constraint(&tools, &choice).unwrap().is_some(),
            constrained,
            "{choice:?}"
        );
    }
}

/// A completion's text with the content and calls it reads back.
type SplitCase = (String, String, Vec<(String, String)>);

#[test]
fn a_parser_declaration_decodes_through_the_shared_helpers() {
    let parser = parser_declaration();
    BundleParserSourceV1::from_declaration(&parser).unwrap();
    let tools = tools();
    let cases = cases();
    let owned: Vec<SplitCase> = cases["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|case| {
            case["calls"]
                .as_array()
                .is_some_and(|calls| !calls.is_empty())
                && case.get("undeclared").and_then(Value::as_bool) != Some(true)
        })
        .map(|case| {
            let calls = case["calls"]
                .as_array()
                .unwrap()
                .iter()
                .map(|call| {
                    (
                        call[0].as_str().unwrap().to_owned(),
                        call[1].as_str().unwrap().to_owned(),
                    )
                })
                .collect();
            // A whole-text parse ends at the stop token's text.
            (
                format!(
                    "{}{}",
                    case["text"].as_str().unwrap(),
                    cases["stop"].as_str().unwrap()
                ),
                case["content"].as_str().unwrap().to_owned(),
                calls,
            )
        })
        .collect();
    // The helper parses whole text without the tools' schemas, so it runs
    // the cases that read alike without them.
    let owned: Vec<_> = owned
        .into_iter()
        .filter(|(text, content, _)| {
            parse_completion_v1(&parser, text, PromptModeV1::Chat)
                .is_ok_and(|parsed| parsed.content == *content)
        })
        .collect();
    assert!(!owned.is_empty());
    let calls: Vec<Vec<(&str, &str)>> = owned
        .iter()
        .map(|(_, _, calls)| {
            calls
                .iter()
                .map(|(name, arguments)| (name.as_str(), arguments.as_str()))
                .collect()
        })
        .collect();
    let split: Vec<(&str, &str, Calls<'_>)> = owned
        .iter()
        .zip(&calls)
        .map(|((text, content, _), calls)| (text.as_str(), content.as_str(), calls.as_slice()))
        .collect();
    assert_call_splits(parser.clone(), &split, &tools);
    // Streamed and whole, each completion reads as its whole parse does.
    for (text, content, calls) in &owned {
        for stream in [false, true] {
            let parsed = deliver(
                parser.clone(),
                PromptModeV1::Chat,
                &[text.as_bytes()],
                stream,
                &tools,
            )
            .unwrap();
            assert_eq!(parsed.content, *content, "{text:?}");
            assert!(
                parsed
                    .calls
                    .iter()
                    .map(|(name, _)| name)
                    .eq(calls.iter().map(|(name, _)| name)),
                "{text:?}"
            );
        }
    }
}
