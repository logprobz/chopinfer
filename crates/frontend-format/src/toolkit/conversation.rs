//! The conversation as a chat format reads it from a request, and the
//! checks a conversation passes before anything is written.

use std::collections::BTreeSet;

use crate::chat::{AssistantTurnV1, ChatMessageV1, ChatRenderV1, ToolResultV1};
use crate::descriptor::FormatDescriptorV1;
use crate::request::{
    FrontendContentPartV1, FrontendErrorV1, FrontendJsonV1, FrontendMessageV1, FrontendRoleV1,
    TOOL_REFERENCES_UNSUPPORTED_V1,
};
use crate::toolkit::json::{
    field_v1, parse_json_object_v1, reserved_free_v1, resource_v1, sequence_v1, validate_json_v1,
};
use crate::toolkit::prompt::validate_function_name_v1;

/// One request message as a renderer reads it: plain content for every
/// role but a user's media parts, and a tool result's call ID. It has no
/// `tool_reference` form, so a part of one is refused with
/// [`TOOL_REFERENCES_UNSUPPORTED_V1`].
pub fn chat_message_v1(message: FrontendMessageV1) -> Result<ChatMessageV1, FrontendErrorV1> {
    let FrontendMessageV1 {
        role,
        content,
        parts,
        reasoning_content,
        tool_calls,
        tool_call_id,
        tools,
    } = message;
    // A format that reads this has no reference form, so its open answer
    // never admits one; refuse any that reaches it anyway.
    if parts
        .iter()
        .flatten()
        .any(|part| matches!(part, FrontendContentPartV1::ToolReference { .. }))
    {
        return Err(FrontendErrorV1::new(
            TOOL_REFERENCES_UNSUPPORTED_V1,
            Some("messages"),
        ));
    }
    if !tools.is_empty() && role != FrontendRoleV1::Assistant {
        return Ok(ChatMessageV1::WithTools {
            role,
            content: plain_v1(content)?,
            tools,
        });
    }
    if !tools.is_empty() {
        return Err(FrontendErrorV1::new(
            "assistant message tools are invalid",
            Some("messages"),
        ));
    }
    match role {
        FrontendRoleV1::System => Ok(ChatMessageV1::System(plain_v1(content)?)),
        FrontendRoleV1::Developer => Ok(ChatMessageV1::Developer(plain_v1(content)?)),
        FrontendRoleV1::User => match parts {
            Some(parts) => Ok(ChatMessageV1::UserParts(parts)),
            None => Ok(ChatMessageV1::User(plain_v1(content)?)),
        },
        FrontendRoleV1::Assistant => Ok(ChatMessageV1::Assistant(AssistantTurnV1 {
            content,
            reasoning_content,
            tool_calls,
        })),
        FrontendRoleV1::Tool => Ok(ChatMessageV1::Tool(ToolResultV1 {
            tool_call_id: tool_call_id.ok_or_else(|| {
                FrontendErrorV1::new("tool message requires tool_call_id", Some("messages"))
            })?,
            content: plain_v1(content)?,
        })),
    }
}

fn plain_v1(content: Option<String>) -> Result<String, FrontendErrorV1> {
    content
        .ok_or_else(|| FrontendErrorV1::new("message content must be a string", Some("messages")))
}

/// Refuses one conversation this model cannot render before anything is
/// written.
pub fn validate_chat_v1(
    descriptor: &FormatDescriptorV1,
    render: &ChatRenderV1<'_>,
) -> Result<(), String> {
    let limits = descriptor.limits;
    let reserved = &descriptor.parser.reserved_markers;
    let messages = render.messages;
    if render.continue_final_message
        && !matches!(messages.last(), Some(ChatMessageV1::Assistant(turn)) if turn.tool_calls.is_empty())
    {
        return Err(field_v1(
            "continue_final_message requires a final assistant message without tool calls",
        ));
    }
    if messages.is_empty() {
        return Err(sequence_v1("at least one message is required"));
    }
    if messages.len() > limits.maximum_messages {
        return Err(resource_v1("too many messages"));
    }
    if render.tools.len() > limits.maximum_tools {
        return Err(resource_v1("too many tools"));
    }
    let mut tool_names = BTreeSet::new();
    for tool in render.tools {
        validate_function_name_v1(descriptor, &tool.name, "tool")?;
        if !tool_names.insert(tool.name.as_str()) {
            return Err(field_v1(format!("duplicate tool name '{}'", tool.name)));
        }
        if let Some(description) = &tool.description {
            reserved_free_v1(description, "tool description", reserved)?;
        }
        if !matches!(tool.parameters, FrontendJsonV1::Object(_)) {
            return Err(field_v1(format!(
                "parameters for tool '{}' must be a JSON object",
                tool.name
            )));
        }
        validate_json_v1(&tool.parameters, "tool schema", limits, reserved)?;
    }
    if let Some(response_format) = render.response_format {
        if !matches!(response_format, FrontendJsonV1::Object(_)) {
            return Err(field_v1("response format must be a JSON object"));
        }
        validate_json_v1(response_format, "response format", limits, reserved)?;
    }

    let mut seen_system = false;
    let mut all_call_ids = BTreeSet::new();
    let mut index = 0;
    while index < messages.len() {
        match &messages[index] {
            ChatMessageV1::System(content) => {
                if index != 0 || seen_system {
                    return Err(sequence_v1("system role is allowed only once at the start"));
                }
                seen_system = true;
                reserved_free_v1(content, "system content", reserved)?;
            }
            ChatMessageV1::Developer(content) => {
                if content.is_empty() {
                    return Err(field_v1("developer content cannot be empty"));
                }
                reserved_free_v1(content, "developer content", reserved)?;
            }
            ChatMessageV1::User(content) => {
                reserved_free_v1(content, "user content", reserved)?;
            }
            ChatMessageV1::UserParts(_) => {
                return Err(field_v1("media parts require segmented rendering"));
            }
            ChatMessageV1::Assistant(turn) => {
                if index == 0
                    || !matches!(
                        messages[index - 1],
                        ChatMessageV1::User(_)
                            | ChatMessageV1::Developer(_)
                            | ChatMessageV1::Tool(_)
                    )
                {
                    return Err(sequence_v1(
                        "assistant must follow user, developer, or tool messages",
                    ));
                }
                if let Some(content) = &turn.content {
                    reserved_free_v1(content, "assistant content", reserved)?;
                }
                if let Some(reasoning) = &turn.reasoning_content {
                    reserved_free_v1(reasoning, "assistant reasoning", reserved)?;
                }
                if turn.tool_calls.len() > limits.maximum_tool_calls {
                    return Err(resource_v1("too many tool calls"));
                }
                for call in &turn.tool_calls {
                    validate_function_name_v1(descriptor, &call.name, "tool call")?;
                    if call.id.is_empty() || !all_call_ids.insert(call.id.as_str()) {
                        return Err(field_v1(format!(
                            "empty or duplicate tool call ID '{}'",
                            call.id
                        )));
                    }
                    reserved_free_v1(&call.id, "tool call ID", reserved)?;
                    let arguments =
                        parse_json_object_v1(&call.arguments, limits, "tool arguments")?;
                    validate_json_v1(
                        &FrontendJsonV1::Object(arguments),
                        "tool arguments",
                        limits,
                        reserved,
                    )?;
                }
                if !turn.tool_calls.is_empty()
                    && index + 1 < messages.len()
                    && !matches!(messages[index + 1], ChatMessageV1::Tool(_))
                {
                    return Err(sequence_v1(
                        "assistant tool calls must be followed by their tool results",
                    ));
                }
            }
            ChatMessageV1::Tool(_) => {
                if index == 0
                    || !matches!(
                        messages[index - 1],
                        ChatMessageV1::Assistant(_) | ChatMessageV1::Tool(_)
                    )
                {
                    return Err(sequence_v1(
                        "tool messages must immediately follow an assistant tool call",
                    ));
                }
                let group_start = index;
                while index < messages.len() && matches!(messages[index], ChatMessageV1::Tool(_)) {
                    index += 1;
                }
                let assistant_index = group_start
                    .checked_sub(1)
                    .ok_or_else(|| sequence_v1("tool group has no assistant"))?;
                let ChatMessageV1::Assistant(assistant) = &messages[assistant_index] else {
                    return Err(sequence_v1("tool group does not follow an assistant"));
                };
                if assistant.tool_calls.is_empty() {
                    return Err(sequence_v1(
                        "tool group follows an assistant without tool calls",
                    ));
                }
                let expected: BTreeSet<&str> = assistant
                    .tool_calls
                    .iter()
                    .map(|call| call.id.as_str())
                    .collect();
                let mut actual = BTreeSet::new();
                for message in &messages[group_start..index] {
                    let ChatMessageV1::Tool(tool) = message else {
                        unreachable!("tool group contains only tool messages");
                    };
                    reserved_free_v1(&tool.tool_call_id, "tool result ID", reserved)?;
                    reserved_free_v1(&tool.content, "tool result content", reserved)?;
                    if !actual.insert(tool.tool_call_id.as_str()) {
                        return Err(field_v1(format!(
                            "duplicate tool result ID '{}'",
                            tool.tool_call_id
                        )));
                    }
                }
                if actual != expected {
                    return Err(sequence_v1(
                        "tool result IDs do not exactly match the preceding calls",
                    ));
                }
                if index < messages.len() && !matches!(messages[index], ChatMessageV1::Assistant(_))
                {
                    return Err(sequence_v1(
                        "a complete tool result group must be followed by an assistant",
                    ));
                }
                continue;
            }
            ChatMessageV1::WithTools { .. } => {
                return Err(field_v1("message-level tools require ChatML"));
            }
        }
        index += 1;
    }
    Ok(())
}
