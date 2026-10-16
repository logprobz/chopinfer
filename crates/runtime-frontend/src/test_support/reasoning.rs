//! Requests for reasoning-control tests.
use runtime_server::{
    FrontendChatRequestV1, FrontendMessageV1, FrontendRoleV1, FrontendToolChoiceV1,
};
pub fn ask(
    keywords: &[(&str, bool)],
    top: Option<&str>,
    nested: Option<&str>,
) -> FrontendChatRequestV1 {
    let keyword = |name: &str| {
        keywords
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| *value)
    };
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
        tools: Vec::new(),
        tool_choice: FrontendToolChoiceV1::None,
        response_format: None,
        reasoning_effort: top.map(str::to_owned),
        enable_thinking: keyword("enable_thinking"),
        thinking: keyword("thinking"),
        nested_reasoning_effort: nested.map(str::to_owned),
        preserve_thinking: None,
        clear_thinking: None,
        add_vision_id: None,
        tool_objects: Vec::new(),
        chat_template_kwargs: None,
    }
}
