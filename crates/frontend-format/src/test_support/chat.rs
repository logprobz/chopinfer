//! Read chat capture fixtures with their declared field conventions.
use crate::{AssistantTurnV1, ChatMessageV1, ToolResultV1};
use crate::{FrontendJsonV1, FrontendRoleV1, FrontendToolCallV1, FrontendToolV1};
use serde::Deserialize;
use serde_json::Value;
#[derive(Deserialize)]
pub struct Golden {
    pub id: String,
    pub messages: Vec<Value>,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default)]
    pub chat_template_kwargs: Value,
    pub token_ids: Vec<u32>,
    pub rendered_text: String,
}

pub fn message(
    value: Value,
    reasoning_key: &str,
    arguments: fn(&Value) -> String,
) -> ChatMessageV1 {
    let role = value["role"].as_str().expect("message role");
    let content = value["content"].as_str().unwrap_or_default().to_owned();
    let tools = value["tools"].as_array().map_or_else(Vec::new, |tools| {
        tools
            .iter()
            .cloned()
            .map(|tool| serde_json::from_value::<FrontendJsonV1>(tool).unwrap())
            .collect()
    });
    if !tools.is_empty() {
        let role = match role {
            "system" => FrontendRoleV1::System,
            "developer" => FrontendRoleV1::Developer,
            "user" => FrontendRoleV1::User,
            "tool" => FrontendRoleV1::Tool,
            _ => panic!("unexpected role with tools: {role}"),
        };
        return ChatMessageV1::WithTools {
            role,
            content,
            tools,
        };
    }
    match role {
        "system" => ChatMessageV1::System(content),
        "developer" => ChatMessageV1::Developer(content),
        "user" => ChatMessageV1::User(content),
        "tool" => ChatMessageV1::Tool(ToolResultV1 {
            tool_call_id: value["tool_call_id"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
            content,
        }),
        "assistant" => ChatMessageV1::Assistant(AssistantTurnV1 {
            content: Some(content),
            reasoning_content: value[reasoning_key].as_str().map(str::to_owned),
            tool_calls: value["tool_calls"]
                .as_array()
                .map_or_else(Vec::new, |calls| {
                    calls
                        .iter()
                        .map(|call| FrontendToolCallV1 {
                            id: call["id"].as_str().unwrap().to_owned(),
                            name: call["function"]["name"].as_str().unwrap().to_owned(),
                            arguments: arguments(&call["function"]["arguments"]),
                        })
                        .collect()
                }),
        }),
        _ => panic!("unknown role: {role}"),
    }
}

pub fn tool(value: Value) -> FrontendToolV1 {
    let function = &value["function"];
    FrontendToolV1 {
        name: function["name"].as_str().unwrap().to_owned(),
        description: function["description"].as_str().map(str::to_owned),
        parameters: serde_json::from_value(function["parameters"].clone()).unwrap(),
        parameters_present: !function["parameters"].is_null(),
        strict: function["strict"].as_bool(),
    }
}
