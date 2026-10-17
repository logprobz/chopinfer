//! The vendor's `encoding/test_encoding.py` cases that state a whole prompt
//! or an exact refusal, run against the port.
use frontend_format::FrontendJsonV1;
use model_deepseek41_contract::frontend::encoding::{
    IMAGE_PLACEHOLDER, Message, ReasoningEffort, Role, ThinkingMode, ToolCall, encode_messages,
};

const EFFORT_75: &str = "<｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, \
                         the more thorough the reasoning)\n\n";

/// `V41_TOOL_CALL_OUTPUT` (test_encoding.py lines 31-41).
const TOOL_CALL_OUTPUT: &str = "  reason  </think>summary\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke \
                                name=\"lookup\">\n<｜DSML｜ parameter name=\"query\" \
                                string=\"true\">value</｜DSML｜ parameter>\n<｜DSML｜ parameter \
                                name=\"limit\" string=\"false\">2</｜DSML｜ parameter>\n</｜DSML｜ \
                                invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>";

fn text(role: Role, content: &str) -> Message {
    Message::new(role, Some(content.to_owned()))
}

fn encode(
    messages: &[Message],
    mode: ThinkingMode,
    effort: Option<ReasoningEffort>,
) -> Result<String, String> {
    encode_messages(messages, mode, true, effort.as_ref(), &|_| {
        IMAGE_PLACEHOLDER.to_owned()
    })
    .map(|(prompt, _)| prompt)
    .map_err(|error| error.0)
}

fn call(name: &str, arguments: &str) -> ToolCall {
    ToolCall {
        name: name.to_owned(),
        arguments: arguments.to_owned(),
        ..ToolCall::default()
    }
}

/// `make_tool_call_messages` (lines 63-79).
fn tool_call_messages() -> Vec<Message> {
    let mut assistant = text(Role::Assistant, "summary");
    assistant.reasoning_content = Some("  reason  ".into());
    assistant.tool_calls = vec![call("lookup", r#"{"query":"value","limit":2}"#)];
    vec![text(Role::User, "question"), assistant]
}

/// `make_tool` (lines 44-60) with `namespace` at the tool level.
fn tool(namespace: Option<&str>) -> FrontendJsonV1 {
    let mut tool = serde_json::json!({
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"}, "limit": {"type": "integer"}}},
        },
    });
    if let Some(namespace) = namespace {
        tool["namespace"] = serde_json::from_str(namespace).unwrap();
    }
    serde_json::from_value(tool).unwrap()
}

#[test]
fn reasoning_effort_maps_to_a_budget() {
    // Lines 191-218.
    for (effort, budget) in [
        (None, 75),
        (Some(ReasoningEffort::Named("low".into())), 50),
        (Some(ReasoningEffort::Named("high".into())), 75),
        (Some(ReasoningEffort::Named("max".into())), 100),
        (Some(ReasoningEffort::Budget(1)), 1),
        (Some(ReasoningEffort::Budget(42)), 42),
        (Some(ReasoningEffort::Budget(100)), 100),
    ] {
        assert_eq!(
            encode(
                &[text(Role::User, "question")],
                ThinkingMode::Thinking,
                effort
            )
            .unwrap(),
            format!(
                "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: {budget} (range 1-100, the \
                 higher the value, the more thorough the reasoning)\n\n<｜User｜>question\
                 <｜Assistant｜><think>"
            )
        );
    }
    // Lines 238-257: chat mode writes no effort, and a bad effort refuses.
    assert_eq!(
        encode(
            &[text(Role::User, "hello")],
            ThinkingMode::Chat,
            Some(ReasoningEffort::Named("max".into()))
        )
        .unwrap(),
        "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>"
    );
    for effort in [
        ReasoningEffort::Budget(-1),
        ReasoningEffort::Budget(0),
        ReasoningEffort::Budget(101),
        ReasoningEffort::Named("medium".into()),
    ] {
        let error = encode(
            &[text(Role::User, "question")],
            ThinkingMode::Thinking,
            Some(effort),
        )
        .unwrap_err();
        assert!(error.contains("int within [1,100]"), "{error}");
    }
}

#[test]
fn system_messages_lead_and_interject() {
    // Lines 274-285.
    assert_eq!(
        encode(
            &[
                text(Role::System, "You are a helpful assistant."),
                text(Role::User, "hello")
            ],
            ThinkingMode::Chat,
            None
        )
        .unwrap(),
        "<｜begin▁of▁sentence｜><｜System｜>You are a helpful assistant.<｜User｜>hello\
         <｜Assistant｜></think>"
    );
    // Lines 288-310.
    let mut answer = text(Role::Assistant, "a1");
    answer.reasoning_content = Some("r1".into());
    assert_eq!(
        encode(
            &[
                text(Role::System, "sys"),
                text(Role::User, "q1"),
                answer,
                text(Role::System, "mid sys"),
            ],
            ThinkingMode::Thinking,
            Some(ReasoningEffort::Budget(88)),
        )
        .unwrap(),
        "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 88 (range 1-100, the higher the \
         value, the more thorough the reasoning)\n\nsys<｜User｜>q1<｜Assistant｜></think>a1\
         <｜end▁of▁sentence｜><｜System｜>mid sys<｜Assistant｜><think>"
    );
}

#[test]
fn tool_calls_render_as_spaced_dsml() {
    // Lines 335-340: the assistant turn after the user's.
    assert_eq!(
        encode(&tool_call_messages(), ThinkingMode::Thinking, None).unwrap(),
        format!(
            "<｜begin▁of▁sentence｜>{EFFORT_75}<｜User｜>question<｜Assistant｜><think>\
             {TOOL_CALL_OUTPUT}"
        )
    );
    // Lines 403-430: a namespace from the call or a qualified name.
    let expected = TOOL_CALL_OUTPUT.replace("name=\"lookup\"", "name=\"search::lookup\"");
    for (name, namespace) in [("lookup", Some("search")), ("search::lookup", None)] {
        let mut messages = tool_call_messages();
        messages[1].tool_calls[0].name = name.into();
        messages[1].tool_calls[0].namespace = namespace.map(str::to_owned);
        let prompt = encode(&messages, ThinkingMode::Thinking, None).unwrap();
        assert!(prompt.ends_with(&expected), "{prompt}");
        let chat = encode(&messages, ThinkingMode::Chat, None).unwrap();
        assert!(
            chat.ends_with(expected.split_once("</think>").unwrap().1),
            "{chat}"
        );
    }
    // Lines 483-502: ambiguous namespaces refuse.
    for (name, namespace, error) in [
        (
            "search::lookup",
            Some("files"),
            "Conflicting tool namespaces",
        ),
        ("search::nested::lookup", None, "Tool name must not contain"),
        (
            "lookup",
            Some("search::nested"),
            "Tool namespace must not contain",
        ),
    ] {
        let mut messages = tool_call_messages();
        messages[1].tool_calls[0].name = name.into();
        messages[1].tool_calls[0].namespace = namespace.map(str::to_owned);
        let refusal = encode(&messages, ThinkingMode::Thinking, None).unwrap_err();
        assert!(refusal.contains(error), "{refusal}");
    }
}

#[test]
fn namespaced_tool_schemas_qualify_the_name() {
    // Lines 380-399, with the namespace on the tool.
    for (namespace, description) in [
        (r#""search""#, "Look up a value"),
        (
            r#"{"name": "search", "description": "Search tools."}"#,
            "Search tools.\\nLook up a value",
        ),
    ] {
        let mut system = text(Role::System, "system");
        system.tools = vec![tool(Some(namespace))];
        let prompt = encode(&[system], ThinkingMode::Chat, None).unwrap();
        let schema = format!(
            "{{\"name\": \"search::lookup\", \"description\": \"{description}\", \"parameters\": \
             {{\"type\": \"object\", \"properties\": {{\"query\": {{\"type\": \"string\"}}, \
             \"limit\": {{\"type\": \"integer\"}}}}}}}}"
        );
        assert!(prompt.contains(&schema), "{prompt}");
        assert!(!prompt.contains("\"namespace\":"));
    }
}

#[test]
fn reasoning_before_the_last_user_turn_drops_without_tools() {
    // Lines 508-546.
    let mut first = text(Role::Assistant, "a1");
    first.reasoning_content = Some("r1".into());
    let mut second = Message::new(Role::Assistant, None);
    second.reasoning_content = Some("r2".into());
    second.tool_calls = vec![ToolCall {
        id: "c1".into(),
        ..call("lookup", "{}")
    }];
    let mut result = text(Role::Tool, "value");
    result.tool_call_id = Some("c1".into());
    let mut messages = vec![
        text(Role::User, "q1"),
        first,
        text(Role::User, "q2"),
        second,
        result,
    ];
    let prompt = encode(&messages, ThinkingMode::Thinking, None).unwrap();
    assert!(prompt.contains("<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>"));
    assert!(!prompt.contains("r1"));
    assert!(prompt.contains("<think>r2</think>"));
    assert!(prompt.ends_with("<tool_result>value</tool_result><｜Assistant｜><think>"));
    messages.push(text(Role::User, "q3"));
    let next = encode(&messages, ThinkingMode::Thinking, None).unwrap();
    assert!(!next.contains("r2"));
    assert!(next.ends_with("q3<｜Assistant｜><think>"));
}

#[test]
fn image_placeholders_in_text_refuse() {
    // Lines 107-114.
    for role in [Role::User, Role::Tool] {
        let message = text(role, &format!("hi {IMAGE_PLACEHOLDER}"));
        assert!(encode(&[message], ThinkingMode::Chat, None).is_err());
    }
}

#[test]
fn task_tokens_follow_their_message() {
    // Lines 560-575.
    let mut user = text(Role::User, "q");
    user.task = Some("action".into());
    assert_eq!(
        encode(
            &[user, text(Role::Assistant, "Search")],
            ThinkingMode::Chat,
            None
        )
        .unwrap(),
        "<｜begin▁of▁sentence｜><｜User｜>q<｜Assistant｜></think><｜action｜>Search\
         <｜end▁of▁sentence｜>"
    );
}
