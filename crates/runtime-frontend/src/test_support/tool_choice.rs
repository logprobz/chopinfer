//! What a tool choice decides for one rendered chat: which functions a
//! completion may call.

use crate::BundleChatStateV1;
use runtime_server::{
    FrontendChatRequestV1, FrontendJsonV1, FrontendMessageV1, FrontendRoleV1, FrontendToolChoiceV1,
    FrontendToolV1, ServerFrontendV1,
};

fn ping() -> FrontendToolV1 {
    FrontendToolV1 {
        name: "ping".to_owned(),
        description: None,
        parameters: serde_json::from_str::<FrontendJsonV1>(
            r#"{"type":"object","properties":{"s":{"type":"string"}}}"#,
        )
        .unwrap(),
        parameters_present: true,
        strict: None,
    }
}

fn request(tools: Vec<FrontendToolV1>, tool_choice: FrontendToolChoiceV1) -> FrontendChatRequestV1 {
    FrontendChatRequestV1 {
        messages: vec![FrontendMessageV1 {
            role: FrontendRoleV1::User,
            content: Some("Hello".to_owned()),
            parts: None,
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tools: Vec::new(),
        }],
        continue_final_message: false,
        add_generation_prompt: true,
        tools,
        tool_choice,
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

/// A completion calls a function only when the request offers one and does
/// not choose none; the decoder reads the rest as markup.
pub fn only_an_offered_function_can_be_called(frontend: &impl ServerFrontendV1) {
    for (tools, choice, extracts) in [
        (vec![ping()], FrontendToolChoiceV1::Auto, true),
        (vec![ping()], FrontendToolChoiceV1::Required, true),
        (
            vec![ping()],
            FrontendToolChoiceV1::Named("ping".to_owned()),
            true,
        ),
        (vec![ping()], FrontendToolChoiceV1::None, false),
        (Vec::new(), FrontendToolChoiceV1::Auto, false),
        (Vec::new(), FrontendToolChoiceV1::None, false),
    ] {
        let label = format!("{} tools, {choice:?}", tools.len());
        let rendered = frontend
            .render_chat(request(tools, choice))
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        let state = BundleChatStateV1::from_state(&rendered.state).expect("this frontend's state");
        assert_eq!(state.extracts_tool_calls(), extracts, "{label}");
        assert_eq!(
            rendered.allowed_function_names.is_empty(),
            !extracts,
            "{label}"
        );
    }
}

/// Choosing none keeps the offered tools in the prompt, as vLLM renders them
/// unless it is configured to exclude them; only the calls change.
pub fn every_tool_choice_renders_the_offered_tools(
    frontend: &impl ServerFrontendV1,
    tools_marker: &str,
) {
    let prompt = |tools: Vec<FrontendToolV1>, choice| {
        frontend.render_chat(request(tools, choice)).unwrap().prompt
    };
    let auto = prompt(vec![ping()], FrontendToolChoiceV1::Auto);
    assert!(
        auto.contains(tools_marker) && auto.contains("\"ping\""),
        "{auto}"
    );
    for choice in [
        FrontendToolChoiceV1::None,
        FrontendToolChoiceV1::Required,
        FrontendToolChoiceV1::Named("ping".to_owned()),
    ] {
        assert_eq!(prompt(vec![ping()], choice.clone()), auto, "{choice:?}");
    }
    assert!(!prompt(Vec::new(), FrontendToolChoiceV1::None).contains(tools_marker));
}

/// A frontend whose template has no `tool_reference` handling declares none,
/// so the transport refuses a tool result made of them, by either name key,
/// before it renders.
pub fn tool_references_are_refused(frontend: &impl ServerFrontendV1) {
    assert!(!frontend.tool_references());
    for key in ["name", "tool_name"] {
        let body = serde_json::json!({
            "messages": [
                {"role": "user", "content": "Find a tool."},
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "call_s", "type": "function",
                    "function": {"name": "ping", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_s",
                 "content": [{"type": "tool_reference", key: "ping"}]},
            ],
            "tools": [{"type": "function", "function": {"name": "ping"}}],
        });
        let serde_json::Value::Object(body) = body else {
            unreachable!("the body is an object")
        };
        let error = runtime_server::prepare_chat_request_v1(frontend, 1 << 20, body, 1)
            .map(|_| ())
            .unwrap_err();
        assert_eq!(
            error.message, "tool_reference content parts are not supported by this model",
            "{key}"
        );
        assert_eq!(error.parameter.as_deref(), Some("messages"), "{key}");
    }
}
