//! The port of the checkpoint's `encoding/encoding.py` (990 lines, sha256
//! 13231c59...99e2), the prompt-format reference pinned with the weights.
//!
//! Each function ports the vendor function of the same name and cites its
//! lines. It works on [`Message`], the dictionaries the vendor reads, with the
//! fields it reads typed: string content or a list of content blocks, an
//! assistant turn's reasoning and tool calls, a system message's tools and
//! response format, and the `task` and `wo_eos` flags. A vendor assertion or
//! exception is an [`EncodingError`] carrying its message.
//!
//! Python's semantics the port keeps: `json.dumps(ensure_ascii=False)` with
//! its default separators and number spelling (`to_json`), `json.loads` of
//! tool-call arguments up to twice (`encode_arguments_to_dsml`), dictionary
//! key order, stable sorting, and the truthiness of the values the vendor
//! tests with `if`.

use frontend_format::toolkit::{json_to_string_v1, json_truthy};
use frontend_format::{FrontendJsonEncodingV1, FrontendJsonV1};

/// Lines 29-39 and 426.
pub const BOS: &str = "<｜begin▁of▁sentence｜>";
pub const EOS: &str = "<｜end▁of▁sentence｜>";
pub const THINKING_START: &str = "<think>";
pub const THINKING_END: &str = "</think>";
pub const DSML: &str = "｜DSML｜";
pub const USER: &str = "<｜User｜>";
pub const ASSISTANT: &str = "<｜Assistant｜>";
pub const LATEST_REMINDER: &str = "<｜latest_reminder｜>";
pub const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";
pub const SYSTEM: &str = "<｜System｜>";

/// Lines 43-50: the task tokens, by task name.
const TASKS: [(&str, &str); 6] = [
    ("action", "<｜action｜>"),
    ("query", "<｜query｜>"),
    ("authority", "<｜authority｜>"),
    ("domain", "<｜domain｜>"),
    ("title", "<｜title｜>"),
    ("read_url", "<｜read_url｜>"),
];

/// Lines 64-66, before the schema.
const RESPONSE_FORMAT: &str =
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n";

/// Lines 443-446, around the budget.
pub const REASONING_EFFORT_OPEN: &str = "Reasoning Effort: ";
pub const REASONING_EFFORT_CLOSE: &str =
    " (range 1-100, the higher the value, the more thorough the reasoning)\n\n";

/// Lines 448-453: the named efforts and their budgets, and the default.
pub const REASONING_EFFORT_MAPPINGS: [(&str, i64); 3] = [("low", 50), ("high", 75), ("max", 100)];
pub const DEFAULT_REASONING_EFFORT: &str = "high";

/// Lines 481-506 with the DSML names of lines 428-430 substituted, before
/// the tool schemas and after them.
const TOOLS_HEAD: &str = "## Tools\n\nYou have access to a set of tools to help answer the user's \
question. You can invoke tools by writing a \"<｜DSML｜ calls>\" block like the following:\n\n\
<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"$TOOL_NAME\">\n<｜DSML｜ parameter name=\"$PARAMETER_NAME\" \
string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>\n...\n</｜DSML｜ invoke>\n\
<｜DSML｜ invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\n\n\
String parameters should be specified as is and set `string=\"true\"`. For all other types \
(numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n\
If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside \
<think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after \
</think> with tool calls or final response.\n\n### Available Tool Schemas\n\n";
const TOOLS_TAIL: &str = "\n\nYou MUST strictly follow the above defined tool name and parameter \
schemas to invoke tool calls.\n";

/// A refusal: the vendor's assertion or exception message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodingError(pub String);

impl std::fmt::Display for EncodingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for EncodingError {}

type Result<T> = std::result::Result<T, EncodingError>;

fn error<T>(message: impl Into<String>) -> Result<T> {
    Err(EncodingError(message.into()))
}

/// `thinking_mode`: "chat" or "thinking".
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThinkingMode {
    Chat,
    Thinking,
}

/// `reasoning_effort`: a name of [`REASONING_EFFORT_MAPPINGS`] or a budget.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReasoningEffort {
    Named(String),
    Budget(i64),
}

/// A message's role.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
    LatestReminder,
}

/// One content block of a list content.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Block {
    Text(String),
    /// An image, by the caller's index of it.
    Image(usize),
    /// A tool result, as `merge_tool_messages` writes it.
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

/// One tool call of an assistant turn, in OpenAI's form: the function's
/// name and arguments text, a call ID and a call-level namespace.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub namespace: Option<String>,
    pub arguments: String,
}

/// One message as the vendor's dictionaries hold it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: Option<String>,
    /// A list content (`content` as a list, or `content_blocks`).
    pub blocks: Option<Vec<Block>>,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
    /// OpenAI tool objects (`{"type": "function", "function": {...}}`).
    pub tools: Vec<FrontendJsonV1>,
    pub response_format: Option<FrontendJsonV1>,
    pub task: Option<String>,
    pub wo_eos: bool,
}

impl Message {
    #[must_use]
    pub fn new(role: Role, content: Option<String>) -> Self {
        Self {
            role,
            content,
            blocks: None,
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tools: Vec::new(),
            response_format: None,
            task: None,
            wo_eos: false,
        }
    }
}

/// `to_json` (lines 76-81): `json.dumps(value, ensure_ascii=False)`.
fn to_json(value: &FrontendJsonV1) -> Result<String> {
    let value = python_ints(value);
    json_to_string_v1(&value, FrontendJsonEncodingV1::PythonJsonV1, "tool schema")
        .map_err(EncodingError)
}

/// `json.loads` reads `-0` as the integer 0, which `json.dumps` writes as `0`.
fn python_ints(value: &FrontendJsonV1) -> FrontendJsonV1 {
    match value {
        FrontendJsonV1::Number(number) if number.to_string() == "-0" => {
            FrontendJsonV1::Number(0.into())
        }
        FrontendJsonV1::Array(values) => {
            FrontendJsonV1::Array(values.iter().map(python_ints).collect())
        }
        FrontendJsonV1::Object(entries) => FrontendJsonV1::Object(
            entries
                .iter()
                .map(|(key, value)| (key.clone(), python_ints(value)))
                .collect(),
        ),
        value => value.clone(),
    }
}

fn get<'a>(entries: &'a [(String, FrontendJsonV1)], key: &str) -> Option<&'a FrontendJsonV1> {
    entries
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
}

/// Sets `key` where it stands, or appends it, as a dictionary assignment does.
fn set(entries: &mut Vec<(String, FrontendJsonV1)>, key: &str, value: FrontendJsonV1) {
    match entries.iter_mut().find(|(name, _)| name == key) {
        Some((_, slot)) => *slot = value,
        None => entries.push((key.to_owned(), value)),
    }
}

/// `tools_from_openai_format` (lines 84-98): each tool's function object
/// with its namespace-qualified name, and a namespace description before
/// its own.
fn tools_from_openai_format(tools: &[FrontendJsonV1]) -> Result<Vec<FrontendJsonV1>> {
    let mut functions = Vec::new();
    for tool in tools {
        let FrontendJsonV1::Object(tool) = tool else {
            return error("a tool must be an object");
        };
        let Some(FrontendJsonV1::Object(function)) = get(tool, "function") else {
            return error("a tool needs a function object");
        };
        let mut function = function.clone();
        if let Some(namespace) = get(tool, "namespace")
            && *namespace != FrontendJsonV1::Null
        {
            set(&mut function, "namespace", namespace.clone());
        }
        let name = tool_name_for_encoding(&function)?;
        set(&mut function, "name", FrontendJsonV1::String(name));
        let namespace = function
            .iter()
            .position(|(key, _)| key == "namespace")
            .map(|index| function.remove(index).1);
        if let Some(FrontendJsonV1::Object(namespace)) = namespace
            && let Some(description) = get(&namespace, "description")
            && json_truthy(description)
        {
            let FrontendJsonV1::String(description) = description else {
                return error("a namespace description must be a string");
            };
            let own = match get(&function, "description") {
                Some(FrontendJsonV1::String(text)) => text.clone(),
                Some(value) if json_truthy(value) => {
                    return error("a tool description must be a string");
                }
                _ => String::new(),
            };
            set(
                &mut function,
                "description",
                FrontendJsonV1::String(format!("{description}\n{own}")),
            );
        }
        functions.push(FrontendJsonV1::Object(function));
    }
    Ok(functions)
}

/// `_split_tool_name` (lines 101-113).
fn split_tool_name(name: &str, namespace: Option<&str>) -> Result<(Option<String>, String)> {
    let (mut namespace, mut name) = (namespace.map(str::to_owned), name.to_owned());
    if let Some((prefix, bare)) = name.clone().split_once("::") {
        if namespace
            .as_deref()
            .is_some_and(|namespace| namespace != prefix)
        {
            return error(format!(
                "Conflicting tool namespaces: {} != {prefix}",
                namespace.unwrap_or_default()
            ));
        }
        namespace = Some(prefix.to_owned());
        name = bare.to_owned();
    }
    if name.contains("::") {
        return error(format!("Tool name must not contain '::': {name}"));
    }
    if namespace
        .as_deref()
        .is_some_and(|namespace| namespace.contains("::"))
    {
        return error(format!(
            "Tool namespace must not contain '::': {}",
            namespace.unwrap_or_default()
        ));
    }
    Ok((namespace, name))
}

/// `_tool_name_for_encoding` (lines 116-121) of a function object.
fn tool_name_for_encoding(function: &[(String, FrontendJsonV1)]) -> Result<String> {
    let namespace = match get(function, "namespace") {
        None | Some(FrontendJsonV1::Null) => None,
        Some(FrontendJsonV1::String(text)) => Some(text.as_str()),
        Some(FrontendJsonV1::Object(entries)) => match get(entries, "name") {
            Some(FrontendJsonV1::String(text)) => Some(text.as_str()),
            _ => return error("a namespace object needs a string name"),
        },
        Some(_) => return error("a tool namespace must be a string or an object"),
    };
    let Some(FrontendJsonV1::String(name)) = get(function, "name") else {
        return error("a tool needs a string name");
    };
    qualified(split_tool_name(name, namespace)?)
}

fn qualified((namespace, name): (Option<String>, String)) -> Result<String> {
    Ok(match namespace {
        None => name,
        Some(namespace) => format!("{namespace}::{name}"),
    })
}

/// `tool_calls_from_openai_format` (lines 124-136): each call's name split
/// from its namespace.
fn tool_calls_from_openai_format(calls: &[ToolCall]) -> Result<Vec<ToolCall>> {
    calls
        .iter()
        .map(|call| {
            let (namespace, name) = split_tool_name(&call.name, call.namespace.as_deref())?;
            Ok(ToolCall {
                id: call.id.clone(),
                name,
                namespace,
                arguments: call.arguments.clone(),
            })
        })
        .collect()
}

/// `merge_tool_messages` (lines 184-230): each tool message becomes a
/// `tool_result` block of a user message, and consecutive user messages
/// join into one unless the earlier names a task.
fn merge_tool_messages(messages: &[Message]) -> Vec<Message> {
    let mut merged: Vec<Message> = Vec::new();
    for message in messages {
        let message = message.clone();
        let mergeable = |last: Option<&Message>| {
            last.is_some_and(|last| last.role == Role::User && last.blocks.is_some())
        };
        match message.role {
            Role::Tool => {
                let block = Block::ToolResult {
                    tool_use_id: message.tool_call_id.unwrap_or_default(),
                    content: message.content.unwrap_or_default(),
                };
                if mergeable(merged.last()) {
                    merged
                        .last_mut()
                        .and_then(|last| last.blocks.as_mut())
                        .expect("a mergeable user message has blocks")
                        .push(block);
                } else {
                    let mut user = Message::new(Role::User, None);
                    user.blocks = Some(vec![block]);
                    merged.push(user);
                }
            }
            Role::User => {
                let blocks = message.blocks.clone().unwrap_or_else(|| {
                    vec![Block::Text(message.content.clone().unwrap_or_default())]
                });
                if mergeable(merged.last()) && merged.last().is_some_and(|last| last.task.is_none())
                {
                    merged
                        .last_mut()
                        .and_then(|last| last.blocks.as_mut())
                        .expect("a mergeable user message has blocks")
                        .extend(blocks);
                } else {
                    let mut user = message;
                    user.blocks = Some(blocks);
                    merged.push(user);
                }
            }
            _ => merged.push(message),
        }
    }
    merged
}

/// `sort_tool_results_by_call_order` (lines 233-266): a user message's
/// tool results in the order of the calls of the assistant turn before it,
/// a result of an unknown call first, ties as they stood.
fn sort_tool_results_by_call_order(messages: &mut [Message]) {
    let mut order: Vec<(String, usize)> = Vec::new();
    for message in messages {
        if message.role == Role::Assistant && !message.tool_calls.is_empty() {
            order.clear();
            for (index, call) in message.tool_calls.iter().enumerate() {
                if !call.id.is_empty() {
                    match order.iter_mut().find(|(id, _)| *id == call.id) {
                        Some((_, slot)) => *slot = index,
                        None => order.push((call.id.clone(), index)),
                    }
                }
            }
        } else if message.role == Role::User
            && let Some(blocks) = message.blocks.as_mut()
            && !blocks.is_empty()
        {
            let results: Vec<Block> = blocks
                .iter()
                .filter(|block| matches!(block, Block::ToolResult { .. }))
                .cloned()
                .collect();
            if results.len() > 1 && !order.is_empty() {
                let rank = |block: &Block| match block {
                    Block::ToolResult { tool_use_id, .. } => order
                        .iter()
                        .find(|(id, _)| id == tool_use_id)
                        .map_or(0, |(_, index)| *index),
                    _ => 0,
                };
                let mut sorted = results;
                sorted.sort_by_key(rank);
                let mut sorted = sorted.into_iter();
                for block in blocks.iter_mut() {
                    if matches!(block, Block::ToolResult { .. }) {
                        *block = sorted.next().expect("one sorted block per result");
                    }
                }
            }
        }
    }
}

/// `_validate_no_image_sp_tokens` (lines 355-367).
fn validate_no_image_sp_tokens(message: &Message) -> Result<()> {
    if message
        .content
        .as_deref()
        .is_some_and(|content| content.contains(IMAGE_PLACEHOLDER))
    {
        return error(format!(
            "Message content contains image special token '{IMAGE_PLACEHOLDER}'. Images should be \
             provided as image content blocks."
        ));
    }
    if message
        .reasoning_content
        .as_deref()
        .is_some_and(|reasoning| reasoning.contains(IMAGE_PLACEHOLDER))
    {
        return error(format!(
            "reasoning_content contains image special token '{IMAGE_PLACEHOLDER}'"
        ));
    }
    Ok(())
}

/// `process_image_messages` and `_process_image_blocks` (lines 323-396):
/// each image block becomes a text block of `placeholder(n)` for the n-th
/// image in prompt order, whose caller index is the n-th of the returned
/// list.
fn process_image_messages(
    messages: &mut [Message],
    placeholder: &dyn Fn(usize) -> String,
) -> Result<Vec<usize>> {
    let mut images = Vec::new();
    for message in messages {
        validate_no_image_sp_tokens(message)?;
        let Some(blocks) = message.blocks.as_mut().filter(|blocks| !blocks.is_empty()) else {
            continue;
        };
        for block in blocks.iter_mut() {
            match block {
                Block::Image(index) => {
                    images.push(*index);
                    *block = Block::Text(placeholder(images.len() - 1));
                }
                Block::Text(text) if text.contains(IMAGE_PLACEHOLDER) => {
                    return error(format!(
                        "Text block contains image placeholder '{IMAGE_PLACEHOLDER}'. Images \
                         should be separate content blocks."
                    ));
                }
                _ => {}
            }
        }
        // Lines 387-393: a list content's text is its text blocks joined.
        if message.content.is_none() {
            let texts: Vec<&str> = blocks
                .iter()
                .filter_map(|block| match block {
                    Block::Text(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            message.content = Some(texts.join("\n\n"));
        }
    }
    Ok(images)
}

/// `render_reasoning_effort` (lines 456-474): the effort is checked for
/// every message, and written before the first in thinking mode.
fn render_reasoning_effort(
    index: usize,
    mode: ThinkingMode,
    effort: Option<&ReasoningEffort>,
) -> Result<String> {
    let budget = match effort {
        None => budget_of(DEFAULT_REASONING_EFFORT),
        Some(ReasoningEffort::Named(name)) => budget_of(name),
        Some(ReasoningEffort::Budget(budget)) => (1..=100).contains(budget).then_some(*budget),
    };
    let Some(budget) = budget else {
        let shown = match effort {
            Some(ReasoningEffort::Named(name)) => name.clone(),
            Some(ReasoningEffort::Budget(budget)) => budget.to_string(),
            None => String::new(),
        };
        return error(format!(
            "Invalid reasoning effort for deepseek_v41: {shown}, should be int within [1,100] or \
             ['low', 'high', 'max']"
        ));
    };
    if index == 0 && mode == ThinkingMode::Thinking {
        return Ok(reasoning_effort_text(budget));
    }
    Ok(String::new())
}

/// The effort prefix of `budget` (lines 443-446).
#[must_use]
pub fn reasoning_effort_text(budget: i64) -> String {
    format!("{REASONING_EFFORT_OPEN}{budget}{REASONING_EFFORT_CLOSE}")
}

/// The budget a named effort maps to (lines 448-452).
#[must_use]
pub fn budget_of(name: &str) -> Option<i64> {
    REASONING_EFFORT_MAPPINGS
        .iter()
        .find(|(effort, _)| *effort == name)
        .map(|(_, budget)| *budget)
}

/// `render_tools` (lines 509-521).
fn render_tools(tools: &[FrontendJsonV1]) -> Result<String> {
    let schemas = tools.iter().map(to_json).collect::<Result<Vec<_>>>()?;
    Ok(format!("{TOOLS_HEAD}{}{TOOLS_TAIL}", schemas.join("\n")))
}

/// `encode_arguments_to_dsml` (lines 524-555): the arguments as JSON (read
/// up to twice, for double-encoded text), else as one string argument
/// named `arguments`; each a string as written and any other value as JSON.
fn encode_arguments_to_dsml(call: &ToolCall) -> Result<String> {
    let mut arguments = FrontendJsonV1::String(call.arguments.clone());
    for _ in 0..2 {
        let FrontendJsonV1::String(text) = &arguments else {
            break;
        };
        match json_loads(text) {
            Some(value) => arguments = value,
            None => break,
        }
    }
    let entries = match arguments {
        FrontendJsonV1::Object(entries) => entries,
        _ => vec![(
            "arguments".to_owned(),
            FrontendJsonV1::String(call.arguments.clone()),
        )],
    };
    let mut parameters = Vec::new();
    for (key, value) in &entries {
        let (string, value) = match value {
            FrontendJsonV1::String(text) => ("true", text.clone()),
            value => ("false", to_json(value)?),
        };
        parameters.push(format!(
            "<{DSML} parameter name=\"{key}\" string=\"{string}\">{value}</{DSML} parameter>"
        ));
    }
    Ok(parameters.join("\n"))
}

/// `json.loads(text)`, or `None` where Python raises. A repeated key keeps
/// its first position and takes its last value, as a Python dict does; the
/// workspace's `serde_json` map keeps insertion order the same way.
pub fn json_loads(text: &str) -> Option<FrontendJsonV1> {
    serde_json::from_str::<serde_json::Value>(text)
        .ok()
        .map(from_json_value)
}

/// A `serde_json` value as the request's JSON value, keys in map order.
fn from_json_value(value: serde_json::Value) -> FrontendJsonV1 {
    match value {
        serde_json::Value::Null => FrontendJsonV1::Null,
        serde_json::Value::Bool(value) => FrontendJsonV1::Bool(value),
        serde_json::Value::Number(number) => FrontendJsonV1::Number(number),
        serde_json::Value::String(text) => FrontendJsonV1::String(text),
        serde_json::Value::Array(values) => {
            FrontendJsonV1::Array(values.into_iter().map(from_json_value).collect())
        }
        serde_json::Value::Object(entries) => FrontendJsonV1::Object(
            entries
                .into_iter()
                .map(|(key, value)| (key, from_json_value(value)))
                .collect(),
        ),
    }
}

/// `find_last_user_index` (lines 562-582): the last user or later system
/// message, passing over a user message of tool results alone; -1 for none.
fn find_last_user_index(messages: &[Message]) -> isize {
    for index in (0..messages.len()).rev() {
        let message = &messages[index];
        if message.role == Role::User || (message.role == Role::System && index > 0) {
            if let Some(blocks) = &message.blocks
                && !blocks.is_empty()
                && !blocks
                    .iter()
                    .any(|block| matches!(block, Block::Text(_) | Block::Image(_)))
            {
                continue;
            }
            return index as isize;
        }
    }
    -1
}

/// `render_message` (lines 585-742): one message and the transition after it.
fn render_message(
    index: usize,
    messages: &[Message],
    mode: ThinkingMode,
    drop_thinking: bool,
    effort: Option<&ReasoningEffort>,
) -> Result<String> {
    let message = &messages[index];
    let last_user = find_last_user_index(messages);
    let position = index as isize;
    let tools = if message.tools.is_empty() {
        Vec::new()
    } else {
        tools_from_openai_format(&message.tools)?
    };
    let calls = tool_calls_from_openai_format(&message.tool_calls)?;

    // Lines 615-619.
    let effort_prompt = render_reasoning_effort(index, mode, effort)?;
    let mut prompt = String::new();
    if index == 0 && (!effort_prompt.is_empty() || message.role == Role::System) {
        prompt.push_str(SYSTEM);
    }
    prompt.push_str(&effort_prompt);

    match message.role {
        // Lines 621-629.
        Role::System => {
            if index > 0 {
                prompt.push_str(SYSTEM);
            }
            prompt.push_str(message.content.as_deref().unwrap_or_default());
            if !tools.is_empty() {
                prompt.push_str("\n\n");
                prompt.push_str(&render_tools(&tools)?);
            }
            if let Some(format) = &message.response_format
                && json_truthy(format)
            {
                prompt.push_str("\n\n");
                prompt.push_str(RESPONSE_FORMAT);
                prompt.push_str(&to_json(format)?);
            }
        }
        // Lines 631-657.
        Role::User => {
            prompt.push_str(USER);
            match message.blocks.as_deref() {
                Some(blocks) if !blocks.is_empty() => {
                    let parts: Vec<String> = blocks
                        .iter()
                        .map(|block| match block {
                            Block::Text(text) => text.clone(),
                            Block::ToolResult { content, .. } => {
                                format!("<tool_result>{content}</tool_result>")
                            }
                            Block::Image(_) => "[Unsupported image]".to_owned(),
                        })
                        .collect();
                    prompt.push_str(&parts.join("\n\n"));
                }
                _ => prompt.push_str(message.content.as_deref().unwrap_or_default()),
            }
        }
        // Lines 659-660.
        Role::LatestReminder => {
            prompt.push_str(LATEST_REMINDER);
            prompt.push_str(message.content.as_deref().unwrap_or("None"));
        }
        // Lines 662-663.
        Role::Tool => {
            return error(
                "deepseek_v41 merges tool messages into user; please preprocess with \
                 merge_tool_messages()",
            );
        }
        // Lines 665-708.
        Role::Assistant => {
            let mut tool_content = String::new();
            if !calls.is_empty() {
                let invocations = calls
                    .iter()
                    .map(|call| {
                        let name = qualified((call.namespace.clone(), call.name.clone()))?;
                        Ok(format!(
                            "<{DSML} invoke name=\"{name}\">\n{}\n</{DSML} invoke>",
                            encode_arguments_to_dsml(call)?
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                tool_content = format!(
                    "\n\n<{DSML} calls>\n{}\n</{DSML} calls>",
                    invocations.join("\n")
                );
            }
            let previous_task = index > 0 && messages[index - 1].task.is_some();
            let mut thinking = String::new();
            if mode == ThinkingMode::Thinking
                && !previous_task
                && (!drop_thinking || position > last_user)
            {
                thinking.push_str(message.reasoning_content.as_deref().unwrap_or_default());
                thinking.push_str(THINKING_END);
            }
            prompt.push_str(&thinking);
            prompt.push_str(message.content.as_deref().unwrap_or_default());
            prompt.push_str(&tool_content);
            if !message.wo_eos {
                prompt.push_str(EOS);
            }
        }
    }

    // Lines 712-714.
    if let Some(next) = messages.get(index + 1)
        && !matches!(next.role, Role::Assistant | Role::LatestReminder)
    {
        return Ok(prompt);
    }
    if let Some(task) = &message.task {
        // Lines 716-729.
        let Some((_, token)) = TASKS.iter().find(|(name, _)| name == task) else {
            return error(format!(
                "Invalid task: '{task}'. Valid tasks are: {:?}",
                TASKS.map(|(name, _)| name)
            ));
        };
        if task != "action" {
            prompt.push_str(token);
        } else {
            prompt.push_str(ASSISTANT);
            prompt.push_str(if mode == ThinkingMode::Thinking {
                THINKING_START
            } else {
                THINKING_END
            });
            prompt.push_str(token);
        }
    } else if message.role == Role::User || (message.role == Role::System && index > 0) {
        // Lines 731-740.
        prompt.push_str(ASSISTANT);
        let thinking = mode == ThinkingMode::Thinking;
        // Lines 735-738: `not drop_thinking and thinking` or `drop_thinking and thinking and
        // index >= last_user_idx`.
        if thinking && (!drop_thinking || position >= last_user) {
            prompt.push_str(THINKING_START);
        } else {
            prompt.push_str(THINKING_END);
        }
    }
    Ok(prompt)
}

/// `_drop_thinking_messages` (lines 749-767): before the last user message,
/// assistant turns lose their reasoning; the roles the vendor keeps stay.
fn drop_thinking_messages(messages: &[Message]) -> Vec<Message> {
    let last_user = find_last_user_index(messages);
    let mut result = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        let kept = matches!(
            message.role,
            Role::User | Role::System | Role::Tool | Role::LatestReminder
        );
        if kept || index as isize >= last_user {
            result.push(message.clone());
        } else if message.role == Role::Assistant {
            let mut message = message.clone();
            message.reasoning_content = None;
            result.push(message);
        }
    }
    result
}

/// `_encode_messages_text` (lines 770-807) without a context: BOS, then
/// each message, with earlier reasoning dropped in thinking mode unless a
/// message declares tools or `drop_thinking` is off.
fn encode_messages_text(
    messages: &[Message],
    mode: ThinkingMode,
    drop_thinking: bool,
    effort: Option<&ReasoningEffort>,
) -> Result<String> {
    let mut prompt = BOS.to_owned();
    let drop_thinking = drop_thinking && !messages.iter().any(|message| !message.tools.is_empty());
    let dropped;
    let messages = if mode == ThinkingMode::Thinking && drop_thinking {
        dropped = drop_thinking_messages(messages);
        dropped.as_slice()
    } else {
        messages
    };
    for index in 0..messages.len() {
        prompt.push_str(&render_message(
            index,
            messages,
            mode,
            drop_thinking,
            effort,
        )?);
    }
    Ok(prompt)
}

/// `encode_messages` (lines 810-844) without a context, with
/// `add_default_bos_token`: the prompt, and the caller's image indices in
/// prompt order. `placeholder(n)` is the text the n-th image renders as
/// (the vendor writes [`IMAGE_PLACEHOLDER`]).
pub fn encode_messages(
    messages: &[Message],
    mode: ThinkingMode,
    drop_thinking: bool,
    effort: Option<&ReasoningEffort>,
    placeholder: &dyn Fn(usize) -> String,
) -> Result<(String, Vec<usize>)> {
    for message in messages {
        validate_no_image_sp_tokens(message)?;
    }
    let mut messages = merge_tool_messages(messages);
    sort_tool_results_by_call_order(&mut messages);
    let images = process_image_messages(&mut messages, placeholder)?;
    let prompt = encode_messages_text(&messages, mode, drop_thinking, effort)?;
    Ok((prompt, images))
}
