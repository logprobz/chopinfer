//! Convert Responses items to the existing, model-owned chat frontend.
use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::ApiError;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    pub input: Value,
    pub model: Option<String>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub min_p: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub seed: Option<i64>,
    pub logit_bias: Option<Value>,
    pub stop: Option<Value>,
    pub stop_token_ids: Option<Vec<u32>>,
    pub min_tokens: Option<usize>,
    pub ignore_eos: Option<bool>,
    pub skip_special_tokens: Option<bool>,
    pub include_stop_str_in_output: Option<bool>,
    pub chat_template_kwargs: Option<Value>,
    pub include_reasoning: Option<bool>,
    pub reasoning: Option<Value>,
    pub text: Option<Value>,
    pub tools: Option<Vec<Value>>,
    pub tool_choice: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
    pub top_logprobs: Option<usize>,
    pub include: Option<Vec<String>>,
    pub stream: Option<bool>,
    pub store: Option<bool>,
    pub background: Option<bool>,
    pub previous_response_id: Option<String>,
    pub metadata: Option<Value>,
    pub user: Option<String>,
    pub truncation: Option<String>,
    pub service_tier: Option<String>,
    pub prompt_cache_key: Option<String>,
}

impl Request {
    pub fn parse(body: &[u8], max_string_bytes: usize) -> Result<Self, ApiError> {
        crate::json_limits::validate(body)
            .map_err(|error| ApiError::invalid(error.to_string(), None))?;
        let raw: Value = serde_json::from_slice(body)
            .map_err(|error| ApiError::invalid(error.to_string(), None))?;
        validate_strings(&raw, max_string_bytes)?;
        let request: Self = serde_json::from_value(raw)
            .map_err(|error| ApiError::invalid(error.to_string(), None))?;
        if request
            .truncation
            .as_deref()
            .is_some_and(|v| v != "disabled")
        {
            return Err(invalid(
                "automatic truncation is not supported",
                "truncation",
            ));
        }
        if request
            .service_tier
            .as_deref()
            .is_some_and(|v| !matches!(v, "auto" | "default"))
        {
            return Err(invalid(
                "only auto and default service tiers are supported",
                "service_tier",
            ));
        }
        if let Some(include) = &request.include {
            for field in include {
                if field != "message.output_text.logprobs" {
                    return Err(invalid(
                        "only message.output_text.logprobs is supported",
                        "include",
                    ));
                }
            }
        }
        if let Some(metadata) = &request.metadata {
            let fields = metadata
                .as_object()
                .ok_or_else(|| invalid("metadata must be an object", "metadata"))?;
            if fields.len() > 16
                || fields.iter().any(|(key, value)| {
                    key.chars().count() > 64
                        || value.as_str().is_none_or(|s| s.chars().count() > 512)
                })
            {
                return Err(invalid(
                    "metadata permits 16 string pairs with 64 character keys and 512 character values",
                    "metadata",
                ));
            }
        }
        Ok(request)
    }

    /// The chat request this one stands for. With `separator`, the frontend
    /// qualifies function names ([`crate::ServerFrontendV1::tool_namespace_separator`]):
    /// a `namespace` tool's functions are offered in that namespace, and a
    /// `function_call` item may name its namespace.
    pub fn chat(
        &self,
        mut history: Vec<Value>,
        reasoning_field: &str,
        separator: Option<&str>,
        tool_result_images: bool,
    ) -> Result<Value, ApiError> {
        history.extend(input_messages(
            &self.input,
            reasoning_field,
            separator,
            tool_result_images,
        )?);
        if let Some(instructions) = &self.instructions {
            history.insert(0, json!({"role":"system", "content":instructions}));
        }
        let mut chat = json!({"messages":history, "stream":self.stream.unwrap_or(false)});
        macro_rules! copy {
            ($($field:ident),* $(,)?) => {$(if let Some(value) = &self.$field {chat[stringify!($field)] = json!(value);})*};
        }
        copy!(
            model,
            temperature,
            top_p,
            top_k,
            min_p,
            presence_penalty,
            frequency_penalty,
            repetition_penalty,
            seed,
            logit_bias,
            stop,
            stop_token_ids,
            min_tokens,
            ignore_eos,
            skip_special_tokens,
            include_stop_str_in_output,
            chat_template_kwargs,
            include_reasoning,
            parallel_tool_calls,
            user
        );
        if let Some(maximum) = self.max_output_tokens {
            chat["max_completion_tokens"] = json!(maximum);
        }
        if let Some(reasoning) = &self.reasoning {
            let fields = reasoning
                .as_object()
                .ok_or_else(|| invalid("reasoning must be an object", "reasoning"))?;
            for (key, value) in fields {
                match key.as_str() {
                    "effort" if value.is_string() || value.is_null() => {
                        chat["reasoning_effort"] = value.clone()
                    }
                    "summary" | "generate_summary" if value.is_null() => {}
                    _ => return Err(invalid("only reasoning.effort is supported", "reasoning")),
                }
            }
        }
        if let Some(text) = &self.text {
            let fields = text
                .as_object()
                .ok_or_else(|| invalid("text must be an object", "text"))?;
            if fields.keys().any(|key| key != "format") {
                return Err(invalid("only text.format is supported", "text"));
            }
            if let Some(format) = fields.get("format").filter(|v| !v.is_null()) {
                chat["response_format"] = match format["type"].as_str() {
                    Some("text" | "json_object") => {
                        supported_fields(format, &["type"], "text.format")?;
                        format.clone()
                    }
                    Some("json_schema") => {
                        supported_fields(
                            format,
                            &["type", "name", "schema", "strict"],
                            "text.format",
                        )?;
                        let mut schema = format.clone();
                        schema
                            .as_object_mut()
                            .expect("format object")
                            .remove("type");
                        json!({"type":"json_schema", "json_schema":schema})
                    }
                    _ => return Err(invalid("unsupported text.format", "text.format")),
                };
            }
        }
        if let Some(tools) = &self.tools {
            let mut converted = Vec::new();
            let mut namespaces = std::collections::BTreeSet::new();
            for tool in tools {
                if tool["type"] != "namespace" || separator.is_none() {
                    converted.push(json!({"type":"function", "function":function_tool(tool)?}));
                    continue;
                }
                supported_fields(tool, &["type", "name", "description", "tools"], "tools")?;
                let name = bare_name(&tool["name"], "a namespace")?;
                if !namespaces.insert(name) {
                    return Err(invalid(
                        format!("duplicate namespace name '{name}'"),
                        "tools",
                    ));
                }
                let mut namespace = json!({"name":name});
                match &tool["description"] {
                    Value::Null => {}
                    Value::String(_) => namespace["description"] = tool["description"].clone(),
                    _ => {
                        return Err(invalid("a namespace description must be a string", "tools"));
                    }
                }
                let functions = tool["tools"]
                    .as_array()
                    .ok_or_else(|| invalid("a namespace tool needs a tools list", "tools"))?;
                for function in functions {
                    let function = function_tool(function)?;
                    bare_name(&function["name"], "a function in a namespace")?;
                    converted.push(json!({"type":"function", "namespace":namespace,
                        "function":function}));
                }
            }
            chat["tools"] = json!(converted);
        }
        if let Some(choice) = &self.tool_choice {
            chat["tool_choice"] = if choice.is_string() {
                choice.clone()
            } else if choice["type"] == "function" && choice["name"].is_string() {
                supported_fields(choice, &["type", "name"], "tool_choice")?;
                json!({"type":"function", "function":{"name":choice["name"]}})
            } else {
                return Err(invalid("unsupported tool_choice", "tool_choice"));
            };
        }
        let top = self.top_logprobs.unwrap_or(0);
        if top > 20 {
            return Err(invalid(
                "top_logprobs must be between 0 and 20",
                "top_logprobs",
            ));
        }
        let logprobs = self
            .include
            .as_ref()
            .is_some_and(|items| items.iter().any(|s| s == "message.output_text.logprobs"));
        if logprobs {
            chat["logprobs"] = json!(true);
            chat["top_logprobs"] = json!(top);
        }
        Ok(chat)
    }
}

/// The name of a `namespace` tool or of a function in one: 1 to 128 bytes of
/// ASCII letters, digits, `_` or `-`. It is checked before the functions are
/// flattened, so neither a namespace without functions nor a function name
/// that already holds a separator escapes it.
fn bare_name<'a>(name: &'a Value, what: &str) -> Result<&'a str, ApiError> {
    match name.as_str() {
        Some(name)
            if (1..=128).contains(&name.len())
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')) =>
        {
            Ok(name)
        }
        _ => Err(invalid(
            format!("{what} name must be 1 to 128 ASCII letters, digits, '_' or '-'"),
            "tools",
        )),
    }
}

/// One function tool as the chat function object it stands for.
fn function_tool(tool: &Value) -> Result<Value, ApiError> {
    if tool["type"] != "function" || !tool["name"].is_string() {
        return Err(invalid("only named function tools are supported", "tools"));
    }
    let mut function = tool.clone();
    function
        .as_object_mut()
        .expect("function object")
        .remove("type");
    if function
        .as_object()
        .expect("function object")
        .keys()
        .any(|key| {
            !matches!(
                key.as_str(),
                "name" | "description" | "parameters" | "strict"
            )
        })
    {
        return Err(invalid("unsupported function tool field", "tools"));
    }
    Ok(function)
}

fn validate_strings(value: &Value, maximum: usize) -> Result<(), ApiError> {
    match value {
        Value::String(s) if s.len() > maximum => {
            return Err(invalid("request string exceeds the byte limit", "input"));
        }
        Value::Array(values) => {
            for value in values {
                validate_strings(value, maximum)?;
            }
        }
        Value::Object(fields) => {
            for (key, value) in fields {
                if key.len() > maximum {
                    return Err(invalid("request key exceeds the byte limit", "input"));
                }
                validate_strings(value, maximum)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// The chat messages of a Responses input. With `separator`, a
/// `function_call` item may name its namespace, which qualifies its name.
pub(super) fn input_messages(
    input: &Value,
    reasoning_field: &str,
    separator: Option<&str>,
    tool_result_images: bool,
) -> Result<Vec<Value>, ApiError> {
    if let Some(text) = input.as_str() {
        return Ok(vec![json!({"role":"user", "content":text})]);
    }
    let items = input
        .as_array()
        .ok_or_else(|| invalid("input must be a string or item list", "input"))?;
    let mut messages: Vec<Value> = Vec::new();
    for item in items {
        let kind = match item.get("type") {
            None => "message",
            Some(Value::String(kind)) => kind.as_str(),
            Some(_) => return Err(invalid("input item type must be a string", "input")),
        };
        let fields: &[&str] = match kind {
            "message" => &["type", "id", "status", "role", "content"],
            "function_call" if separator.is_some() => &[
                "type",
                "id",
                "status",
                "call_id",
                "name",
                "namespace",
                "arguments",
            ],
            "function_call" => &["type", "id", "status", "call_id", "name", "arguments"],
            "function_call_output" => &["type", "id", "status", "call_id", "output"],
            "reasoning" => &[
                "type",
                "id",
                "status",
                "content",
                "summary",
                "encrypted_content",
            ],
            _ => return Err(invalid("unsupported input item type", "input")),
        };
        supported_fields(item, fields, "input")?;
        match kind {
            "message" => {
                let role = string(item, "role")?;
                if !matches!(role, "user" | "system" | "developer" | "assistant") { return Err(invalid("unsupported message role", "input")); }
                let content = text_parts(&item["content"], &["input_text", "output_text"])?;
                if role == "assistant" && messages.last().is_some_and(|m| m["role"] == "assistant" && m["content"].is_null()) {
                    messages.last_mut().expect("assistant")["content"] = json!(content);
                } else { messages.push(json!({"role":role, "content":content})); }
            },
            "function_call" => {
                let name = match (&item["namespace"], separator) {
                    (Value::Null, _) | (_, None) => string(item, "name")?.to_owned(),
                    (Value::String(namespace), Some(separator)) => {
                        format!("{namespace}{separator}{}", string(item, "name")?)
                    }
                    _ => return Err(invalid("input item namespace must be a string", "input")),
                };
                let call = json!({"id":string(item,"call_id")?, "type":"function", "function":{"name":name, "arguments":string(item,"arguments")?}});
                let message = assistant(&mut messages);
                if message.get("tool_calls").is_none() { message["tool_calls"] = json!([]); }
                message["tool_calls"].as_array_mut().expect("calls").push(call);
            },
            "function_call_output" => messages.push(json!({"role":"tool", "tool_call_id":string(item,"call_id")?, "content":tool_output(&item["output"], tool_result_images)?})),
            "reasoning" => {
                if !item["encrypted_content"].is_null() { return Err(invalid("encrypted reasoning is not supported", "input")); }
                let summary = if item["summary"].is_null() { None }
                    else { Some(text_parts(&item["summary"], &["summary_text"])? ) };
                let text = if item["content"].is_null() {
                    summary.ok_or_else(|| invalid("reasoning requires content or summary", "input"))?
                } else { text_parts(&item["content"], &["reasoning_text"])? };
                assistant(&mut messages)[reasoning_field] = json!(text);
            },
            _ => return Err(invalid("unsupported input item type", "input")),
        }
    }
    Ok(messages)
}

fn assistant(messages: &mut Vec<Value>) -> &mut Value {
    if messages.last().is_none_or(|m| m["role"] != "assistant") {
        messages.push(json!({"role":"assistant", "content":null}));
    }
    messages.last_mut().expect("assistant inserted")
}
fn text_parts(value: &Value, types: &[&str]) -> Result<String, ApiError> {
    if let Some(text) = value.as_str() {
        return Ok(text.to_owned());
    }
    let parts = value
        .as_array()
        .ok_or_else(|| invalid("content must be text or a text part list", "input"))?;
    let mut text = String::new();
    for part in parts {
        if !part["type"]
            .as_str()
            .is_some_and(|kind| types.contains(&kind))
        {
            return Err(invalid("only text content parts are supported", "input"));
        }
        let fields: &[&str] = if part["type"] == "output_text" {
            &["type", "text", "annotations", "logprobs"]
        } else {
            &["type", "text"]
        };
        supported_fields(part, fields, "input")?;
        text.push_str(string(part, "text")?);
    }
    Ok(text)
}

fn tool_output(value: &Value, images: bool) -> Result<Value, ApiError> {
    // Preserve text-only mapping unless this frontend declares image results.
    if !images || !value.is_array() {
        return text_parts(value, &["input_text"]).map(Value::String);
    }
    value
        .as_array()
        .expect("array checked")
        .iter()
        .map(|part| match part["type"].as_str() {
            Some("input_text") => {
                supported_fields(part, &["type", "text"], "input")?;
                Ok(json!({"type": "text", "text": string(part, "text")?}))
            }
            Some("input_image") => {
                supported_fields(part, &["type", "image_url", "detail"], "input")?;
                let mut image = json!({"url": string(part, "image_url")?});
                if let Some(detail) = part.get("detail") {
                    image["detail"] = detail.clone();
                }
                Ok(json!({"type": "image_url", "image_url": image}))
            }
            _ => Err(invalid(
                "tool output supports input_text and input_image parts",
                "input",
            )),
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Value::Array)
}

fn supported_fields(value: &Value, allowed: &[&str], param: &str) -> Result<(), ApiError> {
    let fields = value
        .as_object()
        .ok_or_else(|| invalid("expected an object", param))?;
    if let Some(field) = fields
        .keys()
        .find(|field| !allowed.contains(&field.as_str()))
    {
        return Err(invalid(format!("unsupported field {field}"), param));
    }
    Ok(())
}

fn string<'a>(value: &'a Value, field: &str) -> Result<&'a str, ApiError> {
    value[field]
        .as_str()
        .ok_or_else(|| invalid(format!("input item requires string {field}"), "input"))
}
pub(super) fn invalid(message: impl Into<String>, param: &str) -> ApiError {
    ApiError::invalid(message, Some(param))
}

#[cfg(test)]
mod tool_output_tests {
    use super::*;

    #[test]
    fn image_results_map_to_chat_parts_only_when_declared() {
        let output = json!([
            {"type": "input_text", "text": "first"},
            {"type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "high"},
            {"type": "input_text", "text": "last"}
        ]);
        let input = json!([{"type": "function_call_output", "call_id": "c", "output": output}]);
        let mapped = input_messages(&input, "reasoning_content", None, true).unwrap();
        assert_eq!(
            mapped,
            [json!({"role": "tool", "tool_call_id": "c", "content": [
                {"type": "text", "text": "first"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA==", "detail": "high"}},
                {"type": "text", "text": "last"}
            ]})]
        );
        let error = input_messages(&input, "reasoning_content", None, false).unwrap_err();
        assert_eq!(error.message, "only text content parts are supported");
        assert_eq!(
            tool_output(
                &json!([{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}]),
                false
            )
            .unwrap(),
            json!("ab")
        );
        for output in [
            json!([{"type": "input_image", "file_id": "f"}]),
            json!([{"type": "input_file", "file_id": "f"}]),
            json!([{"type": "input_image", "image_url": 1}]),
        ] {
            assert!(tool_output(&output, true).is_err());
        }
    }
}
