use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use runtime_sampling::{LogitBias, SamplingDefaults, SamplingParameters};
use serde::Deserialize;

use crate::backend::{
    CancellationSignal, ChatMetadata, DecodingOptions, PreparedPrompt, PreparedRequest,
    RequestedLogprobs,
};
use crate::constraint::{RequestedConstraint, requested_constraint};
use crate::error::ApiError;
use crate::frontend::{
    FrontendChatRequestV1, FrontendContentPartV1, FrontendErrorV1, FrontendJsonV1,
    FrontendMediaPartKindV1, FrontendMediaPartV1, FrontendMessageV1, FrontendRoleV1,
    FrontendToolCallV1, FrontendToolChoiceV1, FrontendToolV1, ServerFrontendV1,
};
use crate::{MAX_REQUEST_HEADER_BYTES, RequestAcceptanceProfile, RequestSurfaceV1};

const DEFAULT_MAX_NEW_TOKENS: usize = 16;
const MAX_NEW_TOKENS: usize = 1_048_576;
const MAX_MESSAGES: usize = 4096;
const MAX_TOOLS: usize = 128;
const MAX_STOPS: usize = 4;
const MAX_HEADERS: usize = 64;

/// Every chat field the single-turn plain chat surface admits.
///
/// This is an allowlist rather than a list of refusals so that a field added
/// to [`ChatRequest`] later reaches the narrow surface only when someone puts
/// it here. `tools` is admitted as a field and then required to be empty,
/// because a client that offers no tools and a client that omits the field are
/// asking for the same thing; `tool_choice`, `parallel_tool_calls`,
/// `response_format`, `reasoning_effort`, and `chat_template_kwargs` are
/// absent because each of them selects behaviour this surface has not
/// validated.
const SINGLE_TURN_PLAIN_CHAT_FIELDS_V1: &[&str] = &[
    "model",
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "logit_bias",
    "seed",
    "user",
    "min_tokens",
    "stop",
    "stop_token_ids",
    "ignore_eos",
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "include_stop_str_in_output",
    "logprobs",
    "top_logprobs",
    "prompt_logprobs",
    "return_token_ids",
    "n",
    "stream",
    "stream_options",
    "tools",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Endpoint {
    Completion,
    Chat,
}

/// What one arriving request is called, and how it can be cancelled.
pub(crate) struct RequestIdentityV1 {
    pub(crate) request_id: String,
    pub(crate) client_request_id: Option<String>,
    pub(crate) generated_seed: u64,
    pub(crate) cancellation: CancellationSignal,
}

pub(crate) struct ParsedRequest {
    pub(crate) truncation: crate::truncation::Truncation,
    pub(crate) choices: usize,
    pub(crate) seeded: bool,
    pub(crate) additional_prompts: Vec<PreparedPrompt>,
    pub(crate) prepared: PreparedRequest,
    pub(crate) stream: bool,
    pub(crate) include_usage: bool,
    pub(crate) continuous_usage_stats: bool,
    pub(crate) use_remaining_context: bool,
    pub(crate) parallel_tool_calls: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StopInput {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
    #[serde(default)]
    continuous_usage_stats: bool,
}

#[derive(Deserialize)]
struct ChatTemplateKwargs {
    thinking: Option<bool>,
    enable_thinking: Option<bool>,
    reasoning_effort: Option<String>,
    preserve_thinking: Option<bool>,
    // Opaque here: a template that does not read it ignores any value.
    clear_thinking: Option<FrontendJsonV1>,
    add_vision_id: Option<FrontendJsonV1>,
}

/// The typed keywords of [`ChatTemplateKwargs`] for a frontend that switches
/// thinking by the thinking object ([`ServerFrontendV1::reads_thinking_object`]):
/// its thinking keywords and effort are ignored unread, in their places, so
/// a positional array still reads the others where they stand. The struct
/// keeps the typed one's name, which a refusal names.
mod object_switch {
    use serde::Deserialize;
    use serde::de::IgnoredAny;

    use crate::frontend::FrontendJsonV1;

    #[derive(Deserialize)]
    pub(super) struct ChatTemplateKwargs {
        #[serde(rename = "thinking")]
        _thinking: Option<IgnoredAny>,
        #[serde(rename = "enable_thinking")]
        _enable_thinking: Option<IgnoredAny>,
        #[serde(rename = "reasoning_effort")]
        _reasoning_effort: Option<IgnoredAny>,
        pub(super) preserve_thinking: Option<bool>,
        pub(super) clear_thinking: Option<FrontendJsonV1>,
        pub(super) add_vision_id: Option<FrontendJsonV1>,
    }
}

thread_local! {
    /// Set while a chat request is read for a frontend that switches
    /// thinking by the thinking object, so [`TemplateKwargsInput`] ignores
    /// the thinking keywords instead of checking their types.
    static THINKING_KEYWORDS_IGNORED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Reads `body` as `T` with the thinking keywords ignored when `ignored`.
fn decode_chat<'a, T: Deserialize<'a>>(body: &'a [u8], ignored: bool) -> Result<T, ApiError> {
    let _ignoring = IgnoringThinkingKeywords::set(ignored);
    decode(body)
}

/// Sets [`THINKING_KEYWORDS_IGNORED`] while it lives and restores the value
/// it replaced on drop, so a decode inside another, or one that panics,
/// leaves the flag as it found it.
struct IgnoringThinkingKeywords {
    previous: bool,
}

impl IgnoringThinkingKeywords {
    fn set(ignored: bool) -> Self {
        Self {
            previous: THINKING_KEYWORDS_IGNORED.replace(ignored),
        }
    }
}

impl Drop for IgnoringThinkingKeywords {
    fn drop(&mut self) {
        THINKING_KEYWORDS_IGNORED.set(self.previous);
    }
}

/// The typed `chat_template_kwargs`: every typed keyword checked, or, while
/// the thinking keywords are ignored ([`decode_chat`]), all but those.
struct TemplateKwargsInput(ChatTemplateKwargs);

impl<'de> Deserialize<'de> for TemplateKwargsInput {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        if !THINKING_KEYWORDS_IGNORED.get() {
            return ChatTemplateKwargs::deserialize(deserializer).map(Self);
        }
        let kwargs = object_switch::ChatTemplateKwargs::deserialize(deserializer)?;
        Ok(Self(ChatTemplateKwargs {
            thinking: None,
            enable_thinking: None,
            reasoning_effort: None,
            preserve_thinking: kwargs.preserve_thinking,
            clear_thinking: kwargs.clear_thinking,
            add_vision_id: kwargs.add_vision_id,
        }))
    }
}

#[derive(Deserialize)]
struct FunctionDefinition {
    name: String,
    description: Option<String>,
    parameters: Option<FrontendJsonV1>,
    strict: Option<bool>,
}

/// Each offered tool's function object exactly as the client sent it, read
/// again only for a frontend whose template renders the objects as sent.
#[derive(Deserialize)]
struct RawToolsV1 {
    tools: Option<Vec<RawToolV1>>,
}

#[derive(Deserialize)]
struct RawToolV1 {
    function: FrontendJsonV1,
}

/// Each offered tool's namespace, read only for a frontend that qualifies
/// function names: the tool's own, else its function's.
#[derive(Deserialize)]
struct RawNamespacesV1 {
    tools: Option<Vec<RawNamespaceToolV1>>,
}

#[derive(Deserialize)]
struct RawNamespaceToolV1 {
    namespace: Option<crate::frontend::LastValueJsonV1>,
    function: Option<RawNamespaceFunctionV1>,
}

#[derive(Deserialize)]
struct RawNamespaceFunctionV1 {
    namespace: Option<crate::frontend::LastValueJsonV1>,
}

/// The client's `chat_template_kwargs` as sent, read again after the typed
/// read so that the typed keywords keep their refusals. It refuses nothing
/// the typed read accepted: a key named twice keeps its first place and its
/// last value, and a number keeps its digits.
#[derive(Deserialize)]
struct RawTemplateKwargsV1 {
    chat_template_kwargs: Option<crate::frontend::LastValueJsonV1>,
}

#[derive(Deserialize)]
struct ToolDefinition {
    #[serde(rename = "type")]
    kind: String,
    function: FunctionDefinition,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HistoricalFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HistoricalCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: HistoricalFunction,
}

#[derive(Deserialize)]
struct MessageInput {
    role: String,
    content: Option<FrontendJsonV1>,
    reasoning: Option<String>,
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<HistoricalCall>>,
    tool_call_id: Option<String>,
    tools: Option<Vec<FrontendJsonV1>>,
}

#[derive(Deserialize)]
struct CompletionRequest {
    truncate_prompt_tokens: Option<i64>,
    truncation_side: Option<String>,
    model: Option<String>,
    prompt: FrontendJsonV1,
    max_tokens: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<i64>,
    min_p: Option<f64>,
    presence_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
    repetition_penalty: Option<f64>,
    logit_bias: Option<FrontendJsonV1>,
    seed: Option<i64>,
    #[serde(rename = "user")]
    _user: Option<String>,
    min_tokens: Option<usize>,
    stop: Option<StopInput>,
    stop_token_ids: Option<Vec<u32>>,
    ignore_eos: Option<bool>,
    skip_special_tokens: Option<bool>,
    spaces_between_special_tokens: Option<bool>,
    include_stop_str_in_output: Option<bool>,
    logprobs: Option<usize>,
    prompt_logprobs: Option<usize>,
    return_token_ids: Option<bool>,
    // The installed tokenizer contract inserts no automatic BOS or EOS.
    #[serde(rename = "add_special_tokens")]
    _add_special_tokens: Option<bool>,
    n: Option<usize>,
    best_of: Option<FrontendJsonV1>,
    echo: Option<bool>,
    suffix: Option<FrontendJsonV1>,
    response_format: Option<FrontendJsonV1>,
    structured_outputs: Option<FrontendJsonV1>,
    cache_salt: Option<String>,
    #[serde(default, deserialize_with = "nullable_bool")]
    stream: bool,
    stream_options: Option<StreamOptions>,
    #[serde(flatten)]
    extra: BTreeMap<String, FrontendJsonV1>,
}

#[derive(Deserialize)]
struct ChatRequest {
    truncate_prompt_tokens: Option<i64>,
    truncation_side: Option<String>,
    model: Option<String>,
    messages: Vec<MessageInput>,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<i64>,
    min_p: Option<f64>,
    presence_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
    repetition_penalty: Option<f64>,
    logit_bias: Option<FrontendJsonV1>,
    seed: Option<i64>,
    #[serde(rename = "user")]
    _user: Option<String>,
    min_tokens: Option<usize>,
    stop: Option<StopInput>,
    stop_token_ids: Option<Vec<u32>>,
    ignore_eos: Option<bool>,
    skip_special_tokens: Option<bool>,
    spaces_between_special_tokens: Option<bool>,
    include_stop_str_in_output: Option<bool>,
    #[serde(default, deserialize_with = "nullable_bool")]
    logprobs: bool,
    top_logprobs: Option<usize>,
    prompt_logprobs: Option<usize>,
    return_token_ids: Option<bool>,
    // The installed tokenizer contract inserts no automatic BOS or EOS.
    #[serde(rename = "add_special_tokens")]
    _add_special_tokens: Option<bool>,
    n: Option<usize>,
    #[serde(default, deserialize_with = "nullable_bool")]
    stream: bool,
    stream_options: Option<StreamOptions>,
    tools: Option<Vec<ToolDefinition>>,
    tool_choice: Option<FrontendJsonV1>,
    parallel_tool_calls: Option<bool>,
    response_format: Option<FrontendJsonV1>,
    structured_outputs: Option<FrontendJsonV1>,
    reasoning_effort: Option<String>,
    chat_template_kwargs: Option<TemplateKwargsInput>,
    #[serde(default)]
    continue_final_message: bool,
    add_generation_prompt: Option<bool>,
    include_reasoning: Option<bool>,
    cache_salt: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, FrontendJsonV1>,
}

pub(crate) fn validate_headers(
    headers: &axum::http::HeaderMap,
) -> Result<Option<String>, ApiError> {
    if headers.len() > MAX_HEADERS {
        return Err(ApiError::invalid("too many request headers", None));
    }
    let bytes = headers.iter().try_fold(2_usize, |total, (name, value)| {
        total
            .checked_add(name.as_str().len())?
            .checked_add(2)?
            .checked_add(value.as_bytes().len())?
            .checked_add(2)
    });
    if bytes.is_none_or(|bytes| bytes > MAX_REQUEST_HEADER_BYTES) {
        return Err(ApiError::invalid(
            "request headers exceed the byte limit",
            None,
        ));
    }
    let name = "X-Client-Request-Id";
    let Some(value) = headers.get(name) else {
        return Ok(None);
    };
    let raw = value.as_bytes();
    if raw.is_empty() || raw.len() > 512 || !raw.is_ascii() {
        return Err(ApiError::invalid(
            format!("{name} must be nonempty ASCII and at most 512 bytes"),
            Some(name),
        ));
    }
    let value = value
        .to_str()
        .map_err(|_| ApiError::invalid(format!("{name} must be valid ASCII"), Some(name)))?;
    Ok(Some(value.to_owned()))
}

pub(crate) fn validate_content_type(headers: &axum::http::HeaderMap) -> Result<(), ApiError> {
    let Some(value) = headers.get(axum::http::header::CONTENT_TYPE) else {
        return Err(ApiError::invalid(
            "Content-Type must be application/json",
            None,
        ));
    };
    let value = value
        .to_str()
        .map_err(|_| ApiError::invalid("invalid Content-Type header", None))?;
    let media_type = value.split(';').next().unwrap_or_default().trim();
    if !media_type.eq_ignore_ascii_case("application/json") {
        return Err(ApiError::invalid(
            "Content-Type must be application/json",
            None,
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn parse(
    frontend: &dyn ServerFrontendV1,
    sampling_overrides: SamplingDefaults,
    endpoint: Endpoint,
    acceptance_profile: RequestAcceptanceProfile,
    surface: RequestSurfaceV1,
    max_string_bytes: usize,
    body: &[u8],
    identity: RequestIdentityV1,
) -> Result<ParsedRequest, ApiError> {
    let one_token = matches!(
        acceptance_profile,
        RequestAcceptanceProfile::OneToken { .. }
    );
    if one_token && endpoint != Endpoint::Completion {
        return Err(ApiError::invalid(
            "one-token acceptance only supports /v1/completions",
            None,
        ));
    }
    let exact_shape = match acceptance_profile {
        RequestAcceptanceProfile::Bounded {
            maximum_generated_tokens,
            ..
        }
        | RequestAcceptanceProfile::ExactShape {
            maximum_generated_tokens,
            ..
        } => Some(maximum_generated_tokens),
        RequestAcceptanceProfile::General
        | RequestAcceptanceProfile::ProfiledStreaming { .. }
        | RequestAcceptanceProfile::OneToken { .. } => None,
    };
    match endpoint {
        // A prompt string is the same request on every surface, so completions
        // carry only the bounds they always carried.
        Endpoint::Completion => {
            if one_token {
                validate_one_token_completion_shape(body)?;
            }
            let request: CompletionRequest = decode(body)?;
            if one_token {
                validate_one_token_completion(&request)?;
            }
            if let Some(maximum) = exact_shape {
                validate_exact_shape_generation(
                    request.max_tokens,
                    "max_tokens",
                    request.stream,
                    maximum,
                )?;
            }
            prepare_completion(
                frontend,
                sampling_overrides,
                request,
                identity,
                max_string_bytes,
            )
        }
        Endpoint::Chat => {
            // An exact shape was measured on one conversation surface and
            // carries it, so the narrow validation runs for it here as well as
            // for a deployment that named the surface on its own.
            let plain = surface == RequestSurfaceV1::SingleTurnPlainChat || exact_shape.is_some();
            if plain {
                validate_single_turn_plain_chat_shape(body)?;
            }
            let request: ChatRequest = decode_chat(body, frontend.reads_thinking_object())?;
            if plain {
                validate_single_turn_plain_chat(&request)?;
            }
            let tool_objects = if frontend.reads_tool_objects() {
                let raw: RawToolsV1 = decode(body)?;
                let objects: Vec<FrontendJsonV1> = raw
                    .tools
                    .unwrap_or_default()
                    .into_iter()
                    .map(|tool| tool.function)
                    .collect();
                for object in &objects {
                    validate_json_strings(object, "tools", max_string_bytes)?;
                }
                objects
            } else {
                Vec::new()
            };
            let tool_namespaces = if frontend.tool_namespace_separator().is_some() {
                let raw: RawNamespacesV1 = decode(body)?;
                raw.tools
                    .unwrap_or_default()
                    .into_iter()
                    .map(|tool| {
                        let present = |namespace: Option<crate::frontend::LastValueJsonV1>| {
                            namespace
                                .map(|namespace| namespace.0)
                                .filter(|namespace| *namespace != FrontendJsonV1::Null)
                        };
                        present(tool.namespace).or_else(|| {
                            tool.function
                                .and_then(|function| present(function.namespace))
                        })
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let raw: RawTemplateKwargsV1 = decode(body)?;
            let template_kwargs = raw
                .chat_template_kwargs
                .map(|kwargs| kwargs.0)
                .and_then(crate::frontend::forwarded_template_kwargs_v1);
            if let Some(maximum) = exact_shape {
                let (declared, parameter) = match request.max_completion_tokens {
                    Some(declared) => (Some(declared), "max_completion_tokens"),
                    None => (request.max_tokens, "max_tokens"),
                };
                validate_exact_shape_generation(declared, parameter, request.stream, maximum)?;
            }
            prepare_chat(
                frontend,
                sampling_overrides,
                request,
                (tool_objects, template_kwargs, tool_namespaces),
                identity,
                max_string_bytes,
            )
        }
    }
}

/// Requires the generation length this deployment measured, stated as itself.
///
/// The count is in generated tokens. A request that omits it is refused rather
/// than defaulted, because a default is a length nothing here has evidence for,
/// and the transport's own default would be charged against a budget sized for
/// this shape. Streaming is refused for the same reason: what this profile
/// admits was measured as one whole answer.
///
/// This runs before the prompt is rendered or tokenized, so a request outside
/// the shape reaches neither the frontend nor the generation backend.
fn validate_exact_shape_generation(
    declared: Option<usize>,
    parameter: &'static str,
    stream: bool,
    maximum_generated_tokens: usize,
) -> Result<(), ApiError> {
    if stream {
        return Err(ApiError::invalid(
            "stream must be false for exact-shape acceptance",
            Some("stream"),
        ));
    }
    let Some(declared) = declared else {
        return Err(ApiError::invalid(
            format!(
                "{parameter} must be stated, between 1 and {maximum_generated_tokens}, for \
                 exact-shape acceptance"
            ),
            Some(parameter),
        ));
    };
    if declared == 0 || declared > maximum_generated_tokens {
        return Err(ApiError::invalid(
            format!(
                "{parameter} must be between 1 and {maximum_generated_tokens} for exact-shape \
                 acceptance"
            ),
            Some(parameter),
        ));
    }
    Ok(())
}

fn validate_one_token_completion_shape(body: &[u8]) -> Result<(), ApiError> {
    let value: FrontendJsonV1 = decode(body)?;
    let FrontendJsonV1::Object(entries) = value else {
        return Err(ApiError::invalid("request body must be an object", None));
    };
    for (name, _) in entries {
        if !matches!(
            name.as_str(),
            "model" | "prompt" | "temperature" | "max_tokens" | "stream"
        ) {
            return Err(ApiError::invalid(
                format!("{name} is not supported by one-token acceptance"),
                Some(&name),
            ));
        }
    }
    Ok(())
}

fn validate_one_token_completion(request: &CompletionRequest) -> Result<(), ApiError> {
    if request.temperature != Some(0.0) {
        return Err(ApiError::invalid(
            "temperature must be 0 for one-token acceptance",
            Some("temperature"),
        ));
    }
    if request.max_tokens != Some(1) {
        return Err(ApiError::invalid(
            "max_tokens must be 1 for one-token acceptance",
            Some("max_tokens"),
        ));
    }
    if request.stream {
        return Err(ApiError::invalid(
            "stream must be false for one-token acceptance",
            Some("stream"),
        ));
    }
    let unsupported = [
        (request.top_p.is_some(), "top_p"),
        (request.top_k.is_some(), "top_k"),
        (request.min_p.is_some(), "min_p"),
        (request.presence_penalty.is_some(), "presence_penalty"),
        (request.frequency_penalty.is_some(), "frequency_penalty"),
        (request.repetition_penalty.is_some(), "repetition_penalty"),
        (request.logit_bias.is_some(), "logit_bias"),
        (request.seed.is_some(), "seed"),
        (request.min_tokens.is_some(), "min_tokens"),
        (request.stop.is_some(), "stop"),
        (request.logprobs.is_some(), "logprobs"),
        (request.prompt_logprobs.is_some(), "prompt_logprobs"),
        (request.n.is_some(), "n"),
        (request.best_of.is_some(), "best_of"),
        (request.echo.is_some(), "echo"),
        (request.suffix.is_some(), "suffix"),
        (request.stream_options.is_some(), "stream_options"),
    ];
    if let Some((_, parameter)) = unsupported.into_iter().find(|(present, _)| *present) {
        return Err(ApiError::invalid(
            format!("{parameter} is not supported by one-token acceptance"),
            Some(parameter),
        ));
    }
    Ok(())
}

/// Refuses every chat field the narrow surface has no evidence for.
///
/// This runs before the request is decoded, so a field that selects tools,
/// structured output, or reasoning is named as itself rather than reported as
/// whatever the renderer made of it.
fn validate_single_turn_plain_chat_shape(body: &[u8]) -> Result<(), ApiError> {
    let value: FrontendJsonV1 = decode(body)?;
    let FrontendJsonV1::Object(entries) = value else {
        return Err(ApiError::invalid("request body must be an object", None));
    };
    for (name, _) in entries {
        if !SINGLE_TURN_PLAIN_CHAT_FIELDS_V1.contains(&name.as_str()) {
            return Err(ApiError::invalid(
                format!("{name} is not supported by the single-turn plain chat surface"),
                Some(&name),
            ));
        }
    }
    Ok(())
}

/// Requires the one conversation shape this surface was validated on.
///
/// That shape is one user message, optionally preceded by one system message,
/// each carrying a plain string. An assistant or tool message is a conversation
/// that already continued, a content array is a multimodal payload, and a tool
/// definition is a grammar; none of the three is served here.
fn validate_single_turn_plain_chat(request: &ChatRequest) -> Result<(), ApiError> {
    if request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
    {
        return Err(ApiError::invalid(
            "tool definitions are not supported by the single-turn plain chat surface",
            Some("tools"),
        ));
    }
    let expected: &[&str] = match request.messages.len() {
        1 => &["user"],
        2 => &["system", "user"],
        _ => {
            return Err(ApiError::invalid(
                "the single-turn plain chat surface serves one user message, optionally preceded \
                 by one system message",
                Some("messages"),
            ));
        }
    };
    for (message, role) in request.messages.iter().zip(expected) {
        if message.role != *role {
            return Err(ApiError::invalid(
                format!(
                    "the single-turn plain chat surface serves a {} message here, not a '{}' one",
                    role, message.role
                ),
                Some("messages"),
            ));
        }
        if message.reasoning_content.is_some()
            || message.reasoning.is_some()
            || message.tool_calls.is_some()
            || message.tool_call_id.is_some()
            || message.tools.is_some()
        {
            return Err(ApiError::invalid(
                "message history is not supported by the single-turn plain chat surface",
                Some("messages"),
            ));
        }
        if !matches!(message.content, Some(FrontendJsonV1::String(_))) {
            return Err(ApiError::invalid(
                "message content must be a string for the single-turn plain chat surface",
                Some("messages"),
            ));
        }
    }
    Ok(())
}

fn decode<'a, T: Deserialize<'a>>(body: &'a [u8]) -> Result<T, ApiError> {
    crate::json_limits::validate(body)
        .map_err(|error| ApiError::invalid(format!("invalid JSON request: {error}"), None))?;
    let mut deserializer = serde_json::Deserializer::from_slice(body);
    let value = T::deserialize(&mut deserializer).map_err(|error| {
        let message = error.to_string();
        ApiError::invalid(
            format!("invalid JSON request: {message}"),
            json_error_parameter(&message),
        )
    })?;
    deserializer
        .end()
        .map_err(|error| ApiError::invalid(format!("invalid JSON request: {error}"), None))?;
    Ok(value)
}

fn json_error_parameter(message: &str) -> Option<&str> {
    for marker in ["unknown field `", "duplicate field `", "missing field `"] {
        if let Some(rest) = message.split(marker).nth(1) {
            return rest.split('`').next();
        }
    }
    None
}

fn completion_prompt(
    prompt: FrontendJsonV1,
    max_string_bytes: usize,
) -> Result<PreparedPrompt, ApiError> {
    Ok(match prompt {
        FrontendJsonV1::String(value) => {
            validate_string(&value, "prompt", max_string_bytes)?;
            PreparedPrompt::Raw(value)
        }
        FrontendJsonV1::Array(ids) if !ids.is_empty() => {
            if ids.len() > 1_048_576 {
                return Err(ApiError::invalid(
                    "token-ID prompt exceeds the token limit",
                    Some("prompt"),
                ));
            }
            let mut tokens = Vec::with_capacity(ids.len());
            for id in ids {
                let FrontendJsonV1::Number(id) = id else {
                    return Err(ApiError::invalid(
                        "token-ID prompt must contain unsigned integers",
                        Some("prompt"),
                    ));
                };
                let Some(id) = id.as_u64().and_then(|id| u32::try_from(id).ok()) else {
                    return Err(ApiError::invalid(
                        "token-ID prompt must contain unsigned 32-bit integers",
                        Some("prompt"),
                    ));
                };
                tokens.push(id);
            }
            PreparedPrompt::TokenIds(tokens)
        }
        _ => {
            return Err(ApiError::invalid(
                "prompt must be a single string or a nonempty token-ID array",
                Some("prompt"),
            ));
        }
    })
}

fn prepare_completion(
    frontend: &dyn ServerFrontendV1,
    sampling_overrides: SamplingDefaults,
    request: CompletionRequest,
    identity: RequestIdentityV1,
    max_string_bytes: usize,
) -> Result<ParsedRequest, ApiError> {
    let RequestIdentityV1 {
        request_id,
        client_request_id,
        generated_seed,
        cancellation,
    } = identity;
    validate_extensions(&request.extra)?;
    let cache_salt = validate_cache_salt(request.cache_salt.clone())?;
    validate_endpoint_extensions(&request.extra, Endpoint::Completion)?;
    let model = request
        .model
        .as_deref()
        .unwrap_or(&frontend.models()[0].id)
        .to_owned();
    validate_model(frontend, &model)?;
    let prompts = match request.prompt {
        FrontendJsonV1::Array(values)
            if values.first().is_some_and(|value| {
                matches!(value, FrontendJsonV1::String(_) | FrontendJsonV1::Array(_))
            }) =>
        {
            let strings = matches!(values[0], FrontendJsonV1::String(_));
            if values.len() > 128
                || values
                    .iter()
                    .any(|value| strings != matches!(value, FrontendJsonV1::String(_)))
            {
                return Err(ApiError::invalid(
                    "prompt batch must contain at most 128 homogeneous prompts",
                    Some("prompt"),
                ));
            }
            values
        }
        prompt => vec![prompt],
    };
    let mut prompts = prompts
        .into_iter()
        .map(|prompt| completion_prompt(prompt, max_string_bytes))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter();
    let prompt = prompts.next().expect("nonempty prompt list");
    let additional_prompts = prompts.collect();
    validate_choices(request.n)?;
    if request.suffix.is_some() {
        return Err(ApiError::invalid("suffix is not supported", Some("suffix")));
    }
    let stops = validate_stops(request.stop, max_string_bytes)?;
    let top_logprobs = request.logprobs.unwrap_or(0);
    if top_logprobs > runtime_sampling::MAX_TOP_LOGPROBS {
        return Err(ApiError::invalid(
            "logprobs must be between 0 and 20",
            Some("logprobs"),
        ));
    }
    if request
        .prompt_logprobs
        .is_some_and(|top| top > runtime_sampling::MAX_TOP_LOGPROBS)
    {
        return Err(ApiError::invalid(
            "prompt_logprobs must be between 0 and 20",
            Some("prompt_logprobs"),
        ));
    }
    if request.stream && request.prompt_logprobs.is_some_and(|top| top > 0) {
        return Err(ApiError::invalid(
            "prompt_logprobs are not available when stream=true",
            Some("prompt_logprobs"),
        ));
    }
    let prompt_top = request
        .prompt_logprobs
        .or_else(|| request.echo.unwrap_or(false).then_some(top_logprobs));
    let constraint = requested_output_constraint(
        request.structured_outputs.as_ref(),
        request.response_format.as_ref(),
        max_string_bytes,
        false,
    )?
    .map(|requested| requested.grammar())
    .transpose()?;
    let sampling = prepare_sampling(
        frontend
            .sampling_defaults()
            .overridden_by(sampling_overrides),
        request.temperature,
        request.top_p,
        request.top_k,
        request.min_p,
        request.presence_penalty,
        request.frequency_penalty,
        request.repetition_penalty,
        request.logit_bias,
        top_logprobs,
    )?
    .with_minimum_new_tokens(validate_minimum(request.min_tokens, request.max_tokens)?)
    .map_err(|error| sampling_error(error, "min_tokens"))?;
    let has_stream_options = request.stream_options.is_some();
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    let continuous_usage_stats = include_usage
        && request
            .stream_options
            .as_ref()
            .is_some_and(|options| options.continuous_usage_stats);
    if has_stream_options && !request.stream {
        return Err(ApiError::invalid(
            "stream_options requires stream=true",
            Some("stream_options"),
        ));
    }
    Ok(ParsedRequest {
        truncation: crate::truncation::Truncation::new(
            request.truncate_prompt_tokens,
            request.truncation_side.as_deref(),
        )?,
        choices: request.n.unwrap_or(1),
        seeded: request.seed.is_some_and(|seed| seed != -1),
        additional_prompts,
        prepared: PreparedRequest {
            request_id,
            client_request_id,
            model,
            prompt,
            max_new_tokens: validate_max(request.max_tokens, "max_tokens")?,
            decoding: prepare_decoding(
                request.stop_token_ids,
                request.ignore_eos,
                request.skip_special_tokens,
                request.spaces_between_special_tokens,
                request.include_stop_str_in_output,
                request.stream,
            )?,
            stop: stops,
            sampling,
            seed: request
                .seed
                .filter(|seed| *seed != -1)
                .map_or(generated_seed, |seed| seed as u64),
            logprobs: RequestedLogprobs {
                enabled: request.logprobs.is_some(),
                top: top_logprobs,
                echo: request.echo.unwrap_or(false),
                prompt_top,
                return_token_ids: request.return_token_ids.unwrap_or(false),
            },
            chat: None,
            constraint,
            cache_salt,
            cancellation,
        },
        stream: request.stream,
        include_usage,
        continuous_usage_stats,
        use_remaining_context: false,
        parallel_tool_calls: true,
    })
}

fn prepare_chat(
    frontend: &dyn ServerFrontendV1,
    sampling_overrides: SamplingDefaults,
    request: ChatRequest,
    (mut tool_objects, chat_template_kwargs, tool_namespaces): (
        Vec<FrontendJsonV1>,
        Option<FrontendJsonV1>,
        Vec<Option<FrontendJsonV1>>,
    ),
    identity: RequestIdentityV1,
    max_string_bytes: usize,
) -> Result<ParsedRequest, ApiError> {
    let RequestIdentityV1 {
        request_id,
        client_request_id,
        generated_seed,
        cancellation,
    } = identity;
    validate_extensions(&request.extra)?;
    let cache_salt = validate_cache_salt(request.cache_salt.clone())?;
    validate_endpoint_extensions(&request.extra, Endpoint::Chat)?;
    let model = request
        .model
        .as_deref()
        .unwrap_or(&frontend.models()[0].id)
        .to_owned();
    validate_model(frontend, &model)?;
    validate_choices(request.n)?;
    if request.top_logprobs.is_some_and(|top| top > 0) && !request.logprobs {
        return Err(ApiError::invalid(
            "top_logprobs requires logprobs=true",
            Some("top_logprobs"),
        ));
    }
    let top_logprobs = request.top_logprobs.unwrap_or(0);
    if top_logprobs > runtime_sampling::MAX_TOP_LOGPROBS {
        return Err(ApiError::invalid(
            "top_logprobs must be between 0 and 20",
            Some("top_logprobs"),
        ));
    }
    if request
        .prompt_logprobs
        .is_some_and(|top| top > runtime_sampling::MAX_TOP_LOGPROBS)
    {
        return Err(ApiError::invalid(
            "prompt_logprobs must be between 0 and 20",
            Some("prompt_logprobs"),
        ));
    }
    if request.stream && request.prompt_logprobs.is_some_and(|top| top > 0) {
        return Err(ApiError::invalid(
            "prompt_logprobs are not available when stream=true",
            Some("prompt_logprobs"),
        ));
    }
    if request.continue_final_message && request.add_generation_prompt == Some(true) {
        return Err(ApiError::invalid(
            "continue_final_message and add_generation_prompt cannot both be true",
            Some("add_generation_prompt"),
        ));
    }
    let mut supplied_tools = request.tools.unwrap_or_default();
    if let Some(separator) = frontend.tool_namespace_separator() {
        qualify_tools(
            &mut supplied_tools,
            &mut tool_objects,
            &tool_namespaces,
            separator,
        )?;
    }
    let tool_choice = parse_tool_choice(
        request.tool_choice,
        &supplied_tools,
        frontend.forced_choice_without_tools_is_auto(),
    )?;
    let thinking_switch = if frontend.reads_thinking_object() {
        Some(thinking_object_switch(request.extra.get("thinking"))?)
    } else {
        None
    };
    // A frontend that constrains only strict requests renders the other
    // response formats and leaves their decoding as its vendor API does.
    let strict_only = frontend.constrains_strict_only();
    let requested = requested_output_constraint(
        request.structured_outputs.as_ref(),
        request.response_format.as_ref(),
        max_string_bytes,
        strict_only,
    )?;
    if matches!(tool_choice, FrontendToolChoiceV1::Named(_))
        && requested
            .as_ref()
            .is_some_and(|requested| requested.excludes_named_tools)
    {
        return Err(ApiError::invalid(
            "You can only either use constraints for structured outputs or tools, not both.",
            Some("structured_outputs"),
        ));
    }
    let messages = convert_messages(
        request.messages,
        frontend,
        frontend.reasoning_field(),
        max_string_bytes,
    )?;
    let tools = convert_tools(supplied_tools, max_string_bytes)?;
    // A forced tool choice, or a strict tool, constrains the output in the
    // model's own call format and replaces any response format constraint,
    // as the reference does. A frontend that constrains only strict
    // requests leaves a strict tool under auto unconstrained.
    let tool_constraint = if matches!(
        tool_choice,
        FrontendToolChoiceV1::Required | FrontendToolChoiceV1::Named(_)
    ) || (!strict_only
        && tool_choice == FrontendToolChoiceV1::Auto
        && tools.iter().any(|tool| tool.strict == Some(true)))
    {
        frontend
            .tool_constraint(&tools, &tool_choice)
            .map_err(|error| frontend_error(error, "tool_choice"))?
            .map(Arc::new)
    } else {
        None
    };
    let constraint = match tool_constraint {
        Some(grammar) => Some(grammar),
        None => requested.map(|requested| requested.grammar()).transpose()?,
    };
    let named_tool_choice = matches!(tool_choice, FrontendToolChoiceV1::Named(_));
    // Each keyword reaches the frontend as sent: which ones the template
    // reads, and which wins, is the model's declaration.
    let (
        enable_thinking,
        thinking,
        nested_reasoning_effort,
        preserve_thinking,
        clear_thinking,
        add_vision_id,
    ) = match request.chat_template_kwargs.map(|options| options.0) {
        Some(options) => (
            options.enable_thinking,
            options.thinking,
            options.reasoning_effort,
            options.preserve_thinking,
            options.clear_thinking,
            options.add_vision_id,
        ),
        None => (None, None, None, None, None, None),
    };
    // A frontend that reads the thinking object switches by it alone: the
    // keywords and the effort in chat_template_kwargs are ignored, and an
    // explicit switch overrides the effort, so the effort that disables
    // thinking under an enabled object names no effort. The client's object
    // reaches the frontend without them, carrying the switch instead.
    let chat_template_kwargs = match (thinking_switch, chat_template_kwargs) {
        (Some(switch), Some(FrontendJsonV1::Object(entries))) => {
            let mut entries: Vec<_> = entries
                .into_iter()
                .filter(|(name, _)| {
                    !matches!(
                        name.as_str(),
                        "enable_thinking" | "thinking" | "reasoning_effort"
                    )
                })
                .collect();
            if let Some(switch) = switch {
                for name in ["enable_thinking", "thinking"] {
                    entries.push((name.to_owned(), FrontendJsonV1::Bool(switch)));
                }
            }
            Some(FrontendJsonV1::Object(entries))
        }
        (_, kwargs) => kwargs,
    };
    let (enable_thinking, thinking, nested_reasoning_effort, reasoning_effort) =
        match thinking_switch {
            Some(switch) => (
                switch,
                switch,
                None,
                request
                    .reasoning_effort
                    .filter(|effort| !(switch == Some(true) && effort == "none")),
            ),
            None => (
                enable_thinking,
                thinking,
                nested_reasoning_effort,
                request.reasoning_effort,
            ),
        };
    // What the reasoning controls mean, how a conversation renders, and which
    // tools survive a choice are all this model's decisions, not the
    // transport's.
    let rendered = frontend
        .render_chat(FrontendChatRequestV1 {
            messages,
            continue_final_message: request.continue_final_message,
            add_generation_prompt: request
                .add_generation_prompt
                .unwrap_or(!request.continue_final_message),
            tools,
            tool_choice,
            response_format: request.response_format,
            reasoning_effort,
            enable_thinking,
            thinking,
            nested_reasoning_effort,
            preserve_thinking,
            clear_thinking,
            add_vision_id,
            tool_objects,
            chat_template_kwargs,
        })
        .map_err(|error| frontend_error(error, "messages"))?;
    let prompt = if rendered.media.is_empty() {
        PreparedPrompt::RenderedChat(rendered.prompt)
    } else {
        PreparedPrompt::MediaChat {
            segments: rendered.segments,
            media: rendered.media,
        }
    };
    let sampling = prepare_sampling(
        frontend
            .sampling_defaults()
            .overridden_by(sampling_overrides),
        request.temperature,
        request.top_p,
        request.top_k,
        request.min_p,
        request.presence_penalty,
        request.frequency_penalty,
        request.repetition_penalty,
        request.logit_bias,
        top_logprobs,
    )?
    .with_minimum_new_tokens(validate_minimum(
        request.min_tokens,
        Some(
            request
                .max_completion_tokens
                .or(request.max_tokens)
                .unwrap_or(MAX_NEW_TOKENS),
        ),
    )?)
    .map_err(|error| sampling_error(error, "min_tokens"))?;
    let has_stream_options = request.stream_options.is_some();
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    let continuous_usage_stats = include_usage
        && request
            .stream_options
            .as_ref()
            .is_some_and(|options| options.continuous_usage_stats);
    if has_stream_options && !request.stream {
        return Err(ApiError::invalid(
            "stream_options requires stream=true",
            Some("stream_options"),
        ));
    }
    Ok(ParsedRequest {
        truncation: crate::truncation::Truncation::new(
            request.truncate_prompt_tokens,
            request.truncation_side.as_deref(),
        )?,
        choices: request.n.unwrap_or(1),
        seeded: request.seed.is_some_and(|seed| seed != -1),
        additional_prompts: Vec::new(),
        prepared: PreparedRequest {
            request_id,
            client_request_id,
            model,
            prompt,
            max_new_tokens: validate_max(
                request.max_completion_tokens.or(request.max_tokens),
                if request.max_completion_tokens.is_some() {
                    "max_completion_tokens"
                } else {
                    "max_tokens"
                },
            )?,
            decoding: prepare_decoding(
                request.stop_token_ids,
                request.ignore_eos,
                request.skip_special_tokens,
                request.spaces_between_special_tokens,
                request.include_stop_str_in_output,
                request.stream,
            )?,
            stop: validate_stops(request.stop, max_string_bytes)?,
            sampling,
            seed: request
                .seed
                .filter(|seed| *seed != -1)
                .map_or(generated_seed, |seed| seed as u64),
            logprobs: RequestedLogprobs {
                enabled: request.logprobs,
                top: top_logprobs,
                echo: false,
                prompt_top: request.prompt_logprobs,
                return_token_ids: request.return_token_ids.unwrap_or(false),
            },
            chat: Some(ChatMetadata {
                include_reasoning: request.include_reasoning.unwrap_or(true),
                allowed_function_names: rendered.allowed_function_names,
                named_tool_choice,
                state: rendered.state,
            }),
            constraint,
            cache_salt,
            cancellation,
        },
        stream: request.stream,
        include_usage,
        continuous_usage_stats,
        use_remaining_context: request.max_tokens.is_none()
            && request.max_completion_tokens.is_none(),
        parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
    })
}

/// Converts one request's messages into the frontend's neutral shape.
///
/// Only what any chat transport can check happens here: the count, the field
/// combinations each role permits, and the byte bounds. What the messages mean
/// is the frontend's decision.
fn convert_messages(
    messages: Vec<MessageInput>,
    frontend: &dyn ServerFrontendV1,
    reasoning_field: &str,
    max_string_bytes: usize,
) -> Result<Vec<FrontendMessageV1>, ApiError> {
    if messages.is_empty() || messages.len() > MAX_MESSAGES {
        return Err(ApiError::invalid(
            "messages must contain between 1 and 4096 items",
            Some("messages"),
        ));
    }
    messages
        .into_iter()
        .map(|message| convert_message(message, frontend, reasoning_field, max_string_bytes))
        .collect()
}

fn convert_message(
    message: MessageInput,
    frontend: &dyn ServerFrontendV1,
    reasoning_field: &str,
    max_string_bytes: usize,
) -> Result<FrontendMessageV1, ApiError> {
    let MessageInput {
        role,
        content: raw_content,
        reasoning,
        reasoning_content,
        tool_calls,
        tool_call_id,
        tools,
    } = message;
    // Either reasoning field as sent, before the selected one is read.
    let sent_reasoning = reasoning.is_some() || reasoning_content.is_some();
    let reasoning_content = if reasoning_field != "reasoning" {
        reasoning_content
    } else if role == "assistant"
        && (frontend.reasoning_content_fallback()
            || (tool_calls.is_some() && frontend.tool_call_reasoning_fallback()))
    {
        reasoning.or(reasoning_content)
    } else {
        reasoning
    };
    let tools = tools.unwrap_or_default();
    for tool in &tools {
        validate_json_strings(tool, "messages", max_string_bytes)?;
    }
    let (content, parts) = match raw_content {
        Some(FrontendJsonV1::String(value)) => {
            validate_string(&value, "messages", max_string_bytes)?;
            (Some(value), None)
        }
        Some(FrontendJsonV1::Null) | None => (None, None),
        Some(FrontendJsonV1::Array(items)) => {
            let parts = convert_content_parts(items, &role, frontend, max_string_bytes)?;
            let text = parts
                .iter()
                .filter_map(|part| match part {
                    FrontendContentPartV1::Text(text) => Some(text.as_str()),
                    FrontendContentPartV1::Media(_)
                    | FrontendContentPartV1::ToolReference { .. } => None,
                })
                .collect::<String>();
            (Some(text), Some(parts))
        }
        Some(_) => {
            return Err(ApiError::invalid(
                "message content must be a string or content-part array",
                Some("messages"),
            ));
        }
    };
    if let Some(reasoning) = &reasoning_content {
        validate_string(reasoning, "messages", max_string_bytes)?;
    }
    match role.as_str() {
        "system" => require_plain_message(reasoning_content, tool_calls, tool_call_id, content)
            .map(|content| plain_message(FrontendRoleV1::System, content, parts, tools)),
        "developer" => require_plain_message(reasoning_content, tool_calls, tool_call_id, content)
            .map(|content| plain_message(FrontendRoleV1::Developer, content, parts, tools)),
        "user" => require_plain_message(reasoning_content, tool_calls, tool_call_id, content)
            .map(|content| plain_message(FrontendRoleV1::User, content, parts, tools)),
        "assistant" => {
            if tool_call_id.is_some() {
                return Err(ApiError::invalid(
                    "assistant messages cannot contain tool_call_id",
                    Some("messages"),
                ));
            }
            let calls = tool_calls
                .unwrap_or_default()
                .into_iter()
                .map(|call| {
                    if call.kind != "function" {
                        return Err(ApiError::invalid(
                            "only function tool calls are supported",
                            Some("messages"),
                        ));
                    }
                    validate_string(&call.id, "messages", max_string_bytes)?;
                    validate_string(&call.function.name, "messages", max_string_bytes)?;
                    validate_string(&call.function.arguments, "messages", max_string_bytes)?;
                    Ok(FrontendToolCallV1 {
                        id: call.id,
                        name: call.function.name,
                        arguments: call.function.arguments,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(FrontendMessageV1 {
                role: FrontendRoleV1::Assistant,
                content,
                parts,
                reasoning_content,
                tool_calls: calls,
                tool_call_id: None,
                tools,
            })
        }
        "tool" => {
            if reasoning_content.is_some() || tool_calls.is_some() {
                return Err(ApiError::invalid(
                    "tool messages contain only content and tool_call_id",
                    Some("messages"),
                ));
            }
            let content = content.ok_or_else(|| {
                ApiError::invalid("tool message content must be a string", Some("messages"))
            })?;
            let call_id = tool_call_id.ok_or_else(|| {
                ApiError::invalid("tool message requires tool_call_id", Some("messages"))
            })?;
            Ok(FrontendMessageV1 {
                role: FrontendRoleV1::Tool,
                content: Some(content),
                parts,
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: Some(call_id),
                tools,
            })
        }
        name if frontend.extra_roles().iter().any(|role| role == name) => {
            // A role the frontend declared carries text alone; what it means
            // is the format's to render. Neither reasoning field belongs to
            // it, whichever one the response selects.
            if parts.is_some() || !tools.is_empty() {
                return Err(ApiError::invalid(
                    format!("{name} messages contain only string content"),
                    Some("messages"),
                ));
            }
            if sent_reasoning {
                return Err(ApiError::invalid(
                    "message contains fields that are not valid for its role",
                    Some("messages"),
                ));
            }
            require_plain_message(reasoning_content, tool_calls, tool_call_id, content).map(
                |content| FrontendMessageV1 {
                    role: FrontendRoleV1::Declared(name.to_owned()),
                    content: Some(content),
                    parts: None,
                    reasoning_content: None,
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                    tools: Vec::new(),
                },
            )
        }
        _ => Err(ApiError::invalid(
            unknown_role_message_v1(frontend.extra_roles()),
            Some("messages"),
        )),
    }
}

/// The refusal of a message role neither the transport nor the frontend
/// reads, naming the roles a frontend declared.
fn unknown_role_message_v1(declared: &[String]) -> String {
    if declared.is_empty() {
        "message role must be system, developer, user, assistant, or tool".to_owned()
    } else {
        format!(
            "message role must be system, developer, user, assistant, tool, or one this model \
             declares: {}",
            declared.join(", ")
        )
    }
}

fn plain_message(
    role: FrontendRoleV1,
    content: String,
    parts: Option<Vec<FrontendContentPartV1>>,
    tools: Vec<FrontendJsonV1>,
) -> FrontendMessageV1 {
    FrontendMessageV1 {
        role,
        content: Some(content),
        parts,
        reasoning_content: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
        tools,
    }
}

fn convert_content_parts(
    items: Vec<FrontendJsonV1>,
    role: &str,
    frontend: &dyn ServerFrontendV1,
    maximum: usize,
) -> Result<Vec<FrontendContentPartV1>, ApiError> {
    if items.is_empty() || items.len() > 128 {
        return Err(ApiError::invalid(
            "content parts must contain between 1 and 128 items",
            Some("messages"),
        ));
    }
    items
        .into_iter()
        .map(|item| {
            let FrontendJsonV1::Object(mut fields) = item else {
                return Err(ApiError::invalid(
                    "content part must be an object",
                    Some("messages"),
                ));
            };
            let kind = take_string(&mut fields, "type")?;
            let part = match kind.as_str() {
                "text" => FrontendContentPartV1::Text(take_string(&mut fields, "text")?),
                "image_url" | "audio_url" | "video_url" | "input_audio" => {
                    if role != "user"
                        && !(role == "tool" && kind == "image_url" && frontend.tool_result_images())
                    {
                        return Err(ApiError::invalid(
                            "only user messages may contain media",
                            Some("messages"),
                        ));
                    }
                    // Refused before anything reads or fetches the media.
                    let modality = match kind.as_str() {
                        "image_url" => FrontendMediaPartKindV1::Image,
                        "video_url" => FrontendMediaPartKindV1::Video,
                        _ => FrontendMediaPartKindV1::Audio,
                    };
                    let served = frontend.served_media();
                    if !served.contains(&modality) {
                        let message = if frontend.has_selected_modalities() {
                            format!(
                                "{0} input is not served by this server, which serves {1} \
                                 (--modalities); --modalities all or a list naming {0} serves \
                                 it when the checkpoint provides it",
                                modality.name(),
                                crate::frontend::served_modalities(frontend).join(",")
                            )
                        } else if served.is_empty() {
                            "this bundle does not support media inputs".into()
                        } else {
                            format!(
                                "{} input is not supported by this server, which serves {}",
                                modality.name(),
                                crate::frontend::served_modalities(frontend).join(",")
                            )
                        };
                        return Err(ApiError::invalid_owned(message, Some("messages".into())));
                    }
                    let media = if kind == "input_audio" {
                        let mut input = take_object(&mut fields, "input_audio")?;
                        let data = take_string(&mut input, "data")?;
                        let format = take_string(&mut input, "format")?;
                        let remote = data.starts_with("http://") || data.starts_with("https://");
                        if data.len() > maximum || format != "wav"
                            || if remote { !frontend.allows_remote_media() } else { !valid_base64(&data) } {
                            return Err(ApiError::invalid(
                                "input_audio requires padded base64 WAV data or an enabled HTTP(S) WAV URL",
                                Some("messages"),
                            ));
                        }
                        reject_fields(&input)?;
                        FrontendMediaPartV1 {
                            kind: FrontendMediaPartKindV1::Audio,
                            part_type: kind,
                            source: data,
                            format: Some(format),
                        }
                    } else {
                        let mut input = take_object(&mut fields, &kind)?;
                        let url = take_string(&mut input, "url")?;
                        validate_string(&url, "messages", maximum)?;
                        if kind == "image_url"
                            && let Some(detail) = take_optional_string(&mut input, "detail")?
                            && !["auto", "low", "high", "original"].contains(&detail.as_str())
                        {
                            return Err(ApiError::invalid(
                                "invalid image detail",
                                Some("messages"),
                            ));
                        }
                        reject_fields(&input)?;
                        let source = validate_media_url(&url, &kind, frontend)?;
                        FrontendMediaPartV1 {
                            kind: modality,
                            part_type: kind,
                            source,
                            format: None,
                        }
                    };
                    FrontendContentPartV1::Media(media)
                }
                "tool_reference" => {
                    // Only a frontend that renders references receives them,
                    // and only in a tool result.
                    if !frontend.tool_references() {
                        return Err(ApiError::invalid(
                            "tool_reference content parts are not supported by this model",
                            Some("messages"),
                        ));
                    }
                    if role != "tool" {
                        return Err(ApiError::invalid(
                            "only tool messages may contain tool_reference parts",
                            Some("messages"),
                        ));
                    }
                    let name = tool_reference_name(&mut fields)?;
                    validate_string(&name, "messages", maximum)?;
                    FrontendContentPartV1::ToolReference { name }
                }
                _ => {
                    return Err(ApiError::invalid(
                        format!("unsupported content part type '{kind}'"),
                        Some("messages"),
                    ));
                }
            };
            reject_fields(&fields)?;
            if let FrontendContentPartV1::Text(text) = &part {
                validate_string(text, "messages", maximum)?;
            }
            Ok(part)
        })
        .collect()
}

/// A `tool_reference` part's name: `name`, or the Anthropic API's
/// `tool_name`, or both when they agree.
fn tool_reference_name(fields: &mut Vec<(String, FrontendJsonV1)>) -> Result<String, ApiError> {
    let required = || {
        ApiError::invalid(
            "a tool_reference part requires a string name",
            Some("messages"),
        )
    };
    let string = |value| match value {
        None => Ok(None),
        Some(FrontendJsonV1::String(value)) => Ok(Some(value)),
        Some(_) => Err(required()),
    };
    match (
        string(take_field(fields, "name"))?,
        string(take_field(fields, "tool_name"))?,
    ) {
        (Some(name), Some(tool_name)) if name != tool_name => Err(ApiError::invalid(
            "a tool_reference part's name and tool_name differ",
            Some("messages"),
        )),
        (Some(name), _) | (None, Some(name)) => Ok(name),
        (None, None) => Err(required()),
    }
}

fn take_object(
    fields: &mut Vec<(String, FrontendJsonV1)>,
    key: &str,
) -> Result<Vec<(String, FrontendJsonV1)>, ApiError> {
    match take_field(fields, key) {
        Some(FrontendJsonV1::Object(object)) => Ok(object),
        _ => Err(ApiError::invalid(
            format!("{key} must be an object"),
            Some("messages"),
        )),
    }
}

fn take_optional_string(
    fields: &mut Vec<(String, FrontendJsonV1)>,
    key: &str,
) -> Result<Option<String>, ApiError> {
    match take_field(fields, key) {
        Some(FrontendJsonV1::String(value)) => Ok(Some(value)),
        None => Ok(None),
        _ => Err(ApiError::invalid(
            format!("{key} must be a string"),
            Some("messages"),
        )),
    }
}

fn take_string(fields: &mut Vec<(String, FrontendJsonV1)>, key: &str) -> Result<String, ApiError> {
    take_optional_string(fields, key)?
        .ok_or_else(|| ApiError::invalid(format!("{key} is required"), Some("messages")))
}

fn take_field(fields: &mut Vec<(String, FrontendJsonV1)>, key: &str) -> Option<FrontendJsonV1> {
    fields
        .iter()
        .position(|(name, _)| name == key)
        .map(|position| fields.remove(position).1)
}

fn reject_fields(fields: &[(String, FrontendJsonV1)]) -> Result<(), ApiError> {
    if let Some((field, _)) = fields.first() {
        Err(ApiError::invalid(
            format!("unknown content part field '{field}'"),
            Some("messages"),
        ))
    } else {
        Ok(())
    }
}

fn valid_base64(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.is_empty() || bytes.len() % 4 != 0 {
        return false;
    }
    let pad = bytes.iter().rev().take_while(|byte| **byte == b'=').count();
    pad <= 2
        && bytes[..bytes.len() - pad]
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || *byte == b'+' || *byte == b'/')
}

fn validate_media_url(
    url: &str,
    kind: &str,
    frontend: &dyn ServerFrontendV1,
) -> Result<String, ApiError> {
    if (url.starts_with("http://") || url.starts_with("https://")) && frontend.allows_remote_media()
    {
        return Ok(url.to_owned());
    }
    if let Some(rest) = url.strip_prefix("data:") {
        let (header, payload) = rest
            .split_once(',')
            .ok_or_else(|| ApiError::invalid("invalid data URL", Some("messages")))?;
        let mime = header
            .strip_suffix(";base64")
            .ok_or_else(|| ApiError::invalid("data URL must use base64", Some("messages")))?;
        let allowed: &[&str] = if kind == "image_url" {
            &["image/png", "image/jpeg"]
        } else if kind == "video_url" {
            &["video/mp4"]
        } else {
            &["audio/wav", "audio/x-wav"]
        };
        if !allowed.contains(&mime) || !valid_base64(payload) {
            return Err(ApiError::invalid(
                "unsupported media MIME type or invalid base64",
                Some("messages"),
            ));
        }
        return Ok(url.to_owned());
    }
    if let Some(path) = url.strip_prefix("file://") {
        let root = frontend.allowed_local_media_path().ok_or_else(|| {
            ApiError::invalid(
                "local media files require --allowed-local-media-path",
                Some("messages"),
            )
        })?;
        let root = root
            .canonicalize()
            .map_err(|_| ApiError::invalid("invalid local media allowlist", Some("messages")))?;
        let denied = || {
            ApiError::invalid(
                "local media file is not allowed or unavailable",
                Some("messages"),
            )
        };
        let path = std::path::Path::new(path);
        if !path.starts_with(&root)
            || path
                .components()
                .any(|part| matches!(part, std::path::Component::ParentDir))
        {
            return Err(denied());
        }
        let path = path.canonicalize().map_err(|_| denied())?;
        if !path.starts_with(&root) || !path.is_file() {
            return Err(denied());
        }
        return Ok(format!("file://{}", path.display()));
    }
    Err(ApiError::invalid(
        "media URL must be a data URL, allowed local file, or enabled HTTP(S) URL",
        Some("messages"),
    ))
}

fn require_plain_message(
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<HistoricalCall>>,
    tool_call_id: Option<String>,
    content: Option<String>,
) -> Result<String, ApiError> {
    if reasoning_content.is_some() || tool_calls.is_some() || tool_call_id.is_some() {
        return Err(ApiError::invalid(
            "message contains fields that are not valid for its role",
            Some("messages"),
        ));
    }
    content.ok_or_else(|| ApiError::invalid("message content must be a string", Some("messages")))
}

/// Names each offered tool by its qualified name, for a frontend that
/// qualifies function names by `separator`
/// ([`ServerFrontendV1::tool_namespace_separator`]): the namespace the
/// request gives it (`namespaces`, one per tool), then the separator, then
/// its name. A name already qualified keeps its namespace, which a given
/// one must equal. A namespace description goes before the function's on
/// its own line. Each function object forwarded (`objects`) takes the same
/// name and description and loses its namespace.
fn qualify_tools(
    tools: &mut [ToolDefinition],
    objects: &mut [FrontendJsonV1],
    namespaces: &[Option<FrontendJsonV1>],
    separator: &str,
) -> Result<(), ApiError> {
    let invalid = |message: String| ApiError::invalid_owned(message, Some("tools".to_owned()));
    for (index, tool) in tools.iter_mut().enumerate() {
        let (namespace, description) = match namespaces.get(index).cloned().flatten() {
            None => (None, None),
            Some(FrontendJsonV1::String(name)) => (Some(name), None),
            Some(FrontendJsonV1::Object(entries)) => {
                let field = |key: &str| {
                    entries
                        .iter()
                        .find(|(name, _)| name == key)
                        .map(|(_, value)| value.clone())
                };
                let Some(FrontendJsonV1::String(name)) = field("name") else {
                    return Err(invalid(
                        "a tool namespace object needs a string name".into(),
                    ));
                };
                let description = match field("description") {
                    None | Some(FrontendJsonV1::Null) => None,
                    Some(FrontendJsonV1::String(text)) => Some(text),
                    Some(_) => {
                        return Err(invalid(
                            "a tool namespace description must be a string".into(),
                        ));
                    }
                };
                (Some(name), description)
            }
            Some(_) => {
                return Err(invalid(
                    "a tool namespace must be a name or an object with a name".into(),
                ));
            }
        };
        let function = &mut tool.function;
        let namespace = match function.name.split_once(separator) {
            Some((prefix, _)) if namespace.as_deref().is_some_and(|name| name != prefix) => {
                return Err(invalid(format!(
                    "the tool {:?} names the namespace {prefix:?} and is given {:?}",
                    function.name,
                    namespace.unwrap_or_default()
                )));
            }
            Some(_) => None,
            None => namespace,
        };
        if let Some(namespace) = namespace {
            function.name = format!("{namespace}{separator}{}", function.name);
        }
        if let Some(prefix) = description.filter(|text| !text.is_empty()) {
            function.description = Some(format!(
                "{prefix}\n{}",
                function.description.as_deref().unwrap_or_default()
            ));
        }
        if let Some(FrontendJsonV1::Object(entries)) = objects.get_mut(index) {
            entries.retain(|(key, _)| key != "namespace");
            for (key, value) in [
                ("name", Some(&function.name)),
                ("description", function.description.as_ref()),
            ] {
                let Some(value) = value else { continue };
                let value = FrontendJsonV1::String(value.clone());
                match entries.iter_mut().find(|(name, _)| name == key) {
                    Some((_, slot)) => *slot = value,
                    None => entries.push((key.to_owned(), value)),
                }
            }
        }
    }
    Ok(())
}

fn convert_tools(
    tools: Vec<ToolDefinition>,
    max_string_bytes: usize,
) -> Result<Vec<FrontendToolV1>, ApiError> {
    if tools.len() > MAX_TOOLS {
        return Err(ApiError::invalid("too many tools", Some("tools")));
    }
    tools
        .into_iter()
        .map(|tool| {
            if tool.kind != "function" {
                return Err(ApiError::invalid(
                    "only function tools are supported",
                    Some("tools"),
                ));
            }
            validate_string(&tool.function.name, "tools", max_string_bytes)?;
            if let Some(description) = &tool.function.description {
                validate_string(description, "tools", max_string_bytes)?;
            }
            let parameters_present = tool.function.parameters.is_some();
            let parameters = tool
                .function
                .parameters
                .unwrap_or_else(|| FrontendJsonV1::Object(Vec::new()));
            validate_json_strings(&parameters, "tools", max_string_bytes)?;
            Ok(FrontendToolV1 {
                name: tool.function.name,
                description: tool.function.description,
                parameters,
                parameters_present,
                strict: tool.function.strict,
            })
        })
        .collect()
}

/// Bounds every string inside one JSON value the request carried.
fn validate_json_strings(
    value: &FrontendJsonV1,
    parameter: &'static str,
    max_string_bytes: usize,
) -> Result<(), ApiError> {
    match value {
        FrontendJsonV1::Null | FrontendJsonV1::Bool(_) | FrontendJsonV1::Number(_) => Ok(()),
        FrontendJsonV1::String(text) => validate_string(text, parameter, max_string_bytes),
        FrontendJsonV1::Array(values) => values
            .iter()
            .try_for_each(|value| validate_json_strings(value, parameter, max_string_bytes)),
        FrontendJsonV1::Object(entries) => entries.iter().try_for_each(|(name, value)| {
            validate_string(name, parameter, max_string_bytes)?;
            validate_json_strings(value, parameter, max_string_bytes)
        }),
    }
}

/// The top-level `thinking` object's switch, for a frontend that reads it
/// ([`ServerFrontendV1::reads_thinking_object`]): a `type` of `enabled` or
/// its alias `adaptive` is true and `disabled` false, and null or absent
/// is no switch. `budget_tokens` is checked and not read.
fn thinking_object_switch(value: Option<&FrontendJsonV1>) -> Result<Option<bool>, ApiError> {
    let entries = match value {
        None | Some(FrontendJsonV1::Null) => return Ok(None),
        Some(FrontendJsonV1::Object(entries)) => entries,
        Some(_) => {
            return Err(ApiError::invalid(
                "thinking must be an object",
                Some("thinking"),
            ));
        }
    };
    let field = |name: &str| {
        entries
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value)
    };
    match field("budget_tokens") {
        None | Some(FrontendJsonV1::Null) => {}
        Some(FrontendJsonV1::Number(number)) if number.as_u64().is_some() => {}
        Some(_) => {
            return Err(ApiError::invalid(
                "thinking.budget_tokens must be a non-negative integer",
                Some("thinking"),
            ));
        }
    }
    match field("type") {
        Some(FrontendJsonV1::String(kind)) if kind == "enabled" || kind == "adaptive" => {
            Ok(Some(true))
        }
        Some(FrontendJsonV1::String(kind)) if kind == "disabled" => Ok(Some(false)),
        _ => Err(ApiError::invalid(
            "thinking.type must be enabled, adaptive or disabled",
            Some("thinking"),
        )),
    }
}

/// The request's tool choice. A frontend for which a forced choice with no
/// offered tools serves as `auto` gets `auto` where the transport would
/// otherwise refuse it.
fn parse_tool_choice(
    choice: Option<FrontendJsonV1>,
    tools: &[ToolDefinition],
    forced_without_tools_is_auto: bool,
) -> Result<FrontendToolChoiceV1, ApiError> {
    const USAGE: &str =
        r#"Correct usage: `{"type": "function", "function": {"name": "my_function"}}`"#;
    let forced = match choice {
        None | Some(FrontendJsonV1::Null) if !tools.is_empty() => {
            return Ok(FrontendToolChoiceV1::Auto);
        }
        None | Some(FrontendJsonV1::Null) => return Ok(FrontendToolChoiceV1::None),
        Some(FrontendJsonV1::String(value)) if value == "auto" => {
            return Ok(FrontendToolChoiceV1::Auto);
        }
        Some(FrontendJsonV1::String(value)) if value == "none" => {
            return Ok(FrontendToolChoiceV1::None);
        }
        Some(FrontendJsonV1::String(value)) if value == "required" => {
            FrontendToolChoiceV1::Required
        }
        Some(FrontendJsonV1::Object(entries)) => {
            let function = entries
                .iter()
                .find(|(key, _)| key == "function")
                .map(|(_, value)| value);
            let Some(FrontendJsonV1::Object(function)) = function else {
                return Err(ApiError::invalid_owned(
                    format!("Invalid value for `function` in `tool_choice`! {USAGE}"),
                    Some("tool_choice.function".to_owned()),
                ));
            };
            let name = function
                .iter()
                .find(|(key, _)| key == "name")
                .map(|(_, value)| value);
            match name {
                Some(FrontendJsonV1::String(name)) if !name.is_empty() => {
                    FrontendToolChoiceV1::Named(name.clone())
                }
                _ => {
                    return Err(ApiError::invalid_owned(
                        format!("Invalid `name` in `function` in `tool_choice`! {USAGE}"),
                        Some("tool_choice.function.name".to_owned()),
                    ));
                }
            }
        }
        Some(_) => {
            return Err(ApiError::invalid(
                "Invalid value for `tool_choice`! Only named tools, \"none\", \"auto\" or \"required\" are supported.",
                Some("tool_choice"),
            ));
        }
    };
    if tools.is_empty() && forced_without_tools_is_auto {
        return Ok(FrontendToolChoiceV1::Auto);
    }
    if tools.is_empty() {
        return Err(ApiError::invalid(
            "When using `tool_choice`, `tools` must be set.",
            Some("tool_choice"),
        ));
    }
    if let FrontendToolChoiceV1::Named(name) = &forced
        && !tools.iter().any(|tool| tool.function.name == *name)
    {
        return Err(ApiError::invalid(
            "The tool specified in `tool_choice` does not match any of the specified `tools`",
            Some("tool_choice"),
        ));
    }
    Ok(forced)
}

/// Reads the structured-output fields after bounding their strings.
fn requested_output_constraint(
    structured_outputs: Option<&FrontendJsonV1>,
    response_format: Option<&FrontendJsonV1>,
    max_string_bytes: usize,
    strict_only: bool,
) -> Result<Option<RequestedConstraint>, ApiError> {
    if let Some(format) = response_format {
        validate_json_strings(format, "response_format", max_string_bytes)?;
    }
    if let Some(outputs) = structured_outputs {
        validate_json_strings(outputs, "structured_outputs", max_string_bytes)?;
    }
    requested_constraint(structured_outputs, response_format, strict_only)
}

fn frontend_error(error: FrontendErrorV1, fallback: &'static str) -> ApiError {
    // A frontend names the request parameter it blames when it can; otherwise
    // the transport reports the one it was rendering.
    let parameter = error.parameter.as_deref().unwrap_or(fallback).to_owned();
    ApiError::invalid_owned(error.message, Some(parameter))
}

#[allow(clippy::too_many_arguments)]
fn prepare_sampling(
    defaults: SamplingDefaults,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<i64>,
    min_p: Option<f64>,
    presence_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
    repetition_penalty: Option<f64>,
    logit_bias: Option<FrontendJsonV1>,
    top_logprobs: usize,
) -> Result<SamplingParameters, ApiError> {
    let temperature = temperature.or(defaults.temperature).unwrap_or(1.0);
    if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
        return Err(ApiError::invalid(
            "temperature must be between 0 and 2",
            Some("temperature"),
        ));
    }
    // vLLM normalizes positive temperatures before deciding greedy versus random.
    let temperature = if temperature > 0.0 {
        temperature.max(0.01)
    } else {
        temperature
    };
    let presence_penalty = presence_penalty.unwrap_or(0.0);
    if !presence_penalty.is_finite() || !(-2.0..=2.0).contains(&presence_penalty) {
        return Err(ApiError::invalid(
            "presence_penalty must be between -2 and 2",
            Some("presence_penalty"),
        ));
    }
    let frequency_penalty = frequency_penalty.unwrap_or(0.0);
    if !frequency_penalty.is_finite() || !(-2.0..=2.0).contains(&frequency_penalty) {
        return Err(ApiError::invalid(
            "frequency_penalty must be between -2 and 2",
            Some("frequency_penalty"),
        ));
    }
    let repetition_penalty = repetition_penalty
        .or(defaults.repetition_penalty)
        .unwrap_or(1.0);
    if !(runtime_sampling::MIN_REPETITION_PENALTY..=runtime_sampling::MAX_REPETITION_PENALTY)
        .contains(&repetition_penalty)
    {
        return Err(ApiError::invalid(
            "repetition_penalty must be between 0.01 and 10",
            Some("repetition_penalty"),
        ));
    }
    SamplingParameters::new(temperature)
        .map_err(|error| sampling_error(error, "temperature"))?
        .with_top_p(top_p.or(defaults.top_p).unwrap_or(1.0))
        .map_err(|error| sampling_error(error, "top_p"))?
        .with_top_k(match top_k.or(defaults.top_k).unwrap_or(-1) {
            -1 | 0 => 0,
            value if value > 0 => usize::try_from(value)
                .map_err(|_| ApiError::invalid("top_k is too large", Some("top_k")))?,
            _ => {
                return Err(ApiError::invalid(
                    "top_k must be -1 or nonnegative",
                    Some("top_k"),
                ));
            }
        })
        .map_err(|error| sampling_error(error, "top_k"))?
        .with_min_p(min_p.or(defaults.min_p).unwrap_or(0.0))
        .map_err(|error| sampling_error(error, "min_p"))?
        .with_penalties(presence_penalty, frequency_penalty, repetition_penalty)
        .map_err(|error| sampling_error(error, "presence_penalty"))?
        .with_logit_bias(convert_logit_bias(logit_bias)?)
        .map_err(|error| sampling_error(error, "logit_bias"))?
        .with_top_logprobs(top_logprobs)
        .map_err(|error| sampling_error(error, "top_logprobs"))
}

fn convert_logit_bias(value: Option<FrontendJsonV1>) -> Result<Vec<LogitBias>, ApiError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let FrontendJsonV1::Object(entries) = value else {
        return Err(ApiError::invalid(
            "logit_bias must be an object",
            Some("logit_bias"),
        ));
    };
    entries
        .into_iter()
        .map(|(token, value)| {
            let token_id = token.parse::<u32>().map_err(|_| {
                ApiError::invalid("logit_bias keys must be token IDs", Some("logit_bias"))
            })?;
            let FrontendJsonV1::Number(number) = value else {
                return Err(ApiError::invalid(
                    "logit_bias values must be numbers",
                    Some("logit_bias"),
                ));
            };
            let bias = number
                .as_f64()
                .ok_or_else(|| ApiError::invalid("invalid logit_bias value", Some("logit_bias")))?;
            LogitBias::new(token_id, bias.clamp(-100.0, 100.0))
                .map_err(|error| ApiError::invalid(error.to_string(), Some("logit_bias")))
        })
        .collect()
}

fn sampling_error(error: runtime_sampling::SamplingError, parameter: &'static str) -> ApiError {
    ApiError::invalid(error.to_string(), Some(parameter))
}

fn prepare_decoding(
    ids: Option<Vec<u32>>,
    ignore_eos: Option<bool>,
    skip_special_tokens: Option<bool>,
    spaces_between_special_tokens: Option<bool>,
    include_stop_str_in_output: Option<bool>,
    stream_tool_calls: bool,
) -> Result<DecodingOptions, ApiError> {
    let mut stop_token_ids = ids.unwrap_or_default();
    if stop_token_ids.len() > runtime_sampling::MAX_STOP_IDS {
        return Err(ApiError::invalid(
            "too many stop token IDs",
            Some("stop_token_ids"),
        ));
    }
    stop_token_ids.sort_unstable();
    stop_token_ids.dedup();
    Ok(DecodingOptions {
        stream_tool_calls,
        stop_token_ids,
        ignore_eos: ignore_eos.unwrap_or(false),
        skip_special_tokens: skip_special_tokens.unwrap_or(true),
        spaces_between_special_tokens: spaces_between_special_tokens.unwrap_or(true),
        include_stop_str_in_output: include_stop_str_in_output.unwrap_or(false),
        // Client requests never pin their drafts.
        warmup_draft_tokens: None,
    })
}

fn validate_endpoint_extensions(
    fields: &BTreeMap<String, FrontendJsonV1>,
    endpoint: Endpoint,
) -> Result<(), ApiError> {
    if endpoint == Endpoint::Chat
        && fields.get("echo").is_some_and(|value| {
            !matches!(value, FrontendJsonV1::Bool(false) | FrontendJsonV1::Null)
        })
    {
        return Err(ApiError::invalid("chat echo is unavailable", Some("echo")));
    }
    Ok(())
}

/// vLLM refuses an empty salt, which would otherwise read as no salt.
fn validate_cache_salt(salt: Option<String>) -> Result<Option<String>, ApiError> {
    match salt {
        Some(salt) if salt.is_empty() => Err(ApiError::invalid(
            "cache_salt must be a non-empty string if provided",
            Some("cache_salt"),
        )),
        salt => Ok(salt),
    }
}

fn validate_extensions(fields: &BTreeMap<String, FrontendJsonV1>) -> Result<(), ApiError> {
    // Unknown extensions are ignored as in the reference API. Known controls
    // that this runtime cannot honor must still fail rather than silently run.
    const UNSUPPORTED: &[&str] = &[
        "allowed_token_ids",
        "bad_words",
        "chat_template",
        "documents",
        "ec_transfer_params",
        "kv_transfer_params",
        "length_penalty",
        "logprob_token_ids",
        "media_io_kwargs",
        "mm_processor_kwargs",
        "model_options",
        "priority",
        "prompt_embeds",
        "repetition_detection",
        "request_id",
        "return_assistant_tokens_mask",
        "return_prompt_text",
        "return_token_offsets",
        "return_tokens_as_token_ids",
        "routed_experts_prompt_start",
        "session_id",
        "stream_interval",
        "thinking_token_budget",
        "use_beam_search",
        "vllm_xargs",
        "watermarking",
    ];
    if let Some(name) = fields
        .keys()
        .find(|name| UNSUPPORTED.contains(&name.as_str()))
    {
        return Err(ApiError::invalid(
            format!("{name} is not supported"),
            Some(name),
        ));
    }
    Ok(())
}

fn validate_model(frontend: &dyn ServerFrontendV1, model: &str) -> Result<(), ApiError> {
    if frontend
        .models()
        .iter()
        .any(|candidate| candidate.id == model)
    {
        Ok(())
    } else {
        Err(ApiError::model_not_found(model))
    }
}

fn validate_choices(value: Option<usize>) -> Result<(), ApiError> {
    if (1..=128).contains(&value.unwrap_or(1)) {
        Ok(())
    } else {
        Err(ApiError::invalid("n must be between 1 and 128", Some("n")))
    }
}

fn validate_minimum(value: Option<usize>, maximum: Option<usize>) -> Result<usize, ApiError> {
    let value = value.unwrap_or(0);
    if value > maximum.unwrap_or(DEFAULT_MAX_NEW_TOKENS) {
        return Err(ApiError::invalid(
            "min_tokens exceeds maximum completion tokens",
            Some("min_tokens"),
        ));
    }
    Ok(value)
}

fn validate_max(value: Option<usize>, parameter: &'static str) -> Result<usize, ApiError> {
    let value = value.unwrap_or(DEFAULT_MAX_NEW_TOKENS);
    if value == 0 || value > MAX_NEW_TOKENS {
        Err(ApiError::invalid(
            "maximum completion tokens must be between 1 and 1048576",
            Some(parameter),
        ))
    } else {
        Ok(value)
    }
}

fn validate_stops(
    stop: Option<StopInput>,
    max_string_bytes: usize,
) -> Result<Vec<String>, ApiError> {
    let values = match stop {
        None => return Ok(Vec::new()),
        Some(StopInput::One(value)) => vec![value],
        Some(StopInput::Many(values)) => values,
    };
    if values.len() > MAX_STOPS {
        return Err(ApiError::invalid(
            "stop must contain at most 4 strings",
            Some("stop"),
        ));
    }
    let mut distinct = BTreeSet::new();
    for value in &values {
        if value.is_empty() || !distinct.insert(value) {
            return Err(ApiError::invalid(
                "stop strings must be nonempty and distinct",
                Some("stop"),
            ));
        }
        validate_string(value, "stop", max_string_bytes)?;
    }
    Ok(values)
}

fn validate_string(
    value: &str,
    parameter: &'static str,
    max_string_bytes: usize,
) -> Result<(), ApiError> {
    if value.len() > max_string_bytes {
        Err(ApiError::invalid(
            "string exceeds the byte limit",
            Some(parameter),
        ))
    } else {
        Ok(())
    }
}

fn nullable_bool<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
    Ok(Option::<bool>::deserialize(deserializer)?.unwrap_or(false))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Panics;

    impl<'de> Deserialize<'de> for Panics {
        fn deserialize<D: serde::Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
            panic!("the decode panics");
        }
    }

    /// The flag's value while the body decodes.
    struct Seen(bool);

    impl<'de> Deserialize<'de> for Seen {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            serde::de::IgnoredAny::deserialize(deserializer)?;
            Ok(Self(THINKING_KEYWORDS_IGNORED.get()))
        }
    }

    #[test]
    fn a_decode_that_panics_still_clears_the_ignored_thinking_keywords() {
        let decoded = std::panic::catch_unwind(|| decode_chat::<Panics>(b"{}", true));
        assert!(decoded.is_err());
        assert!(!THINKING_KEYWORDS_IGNORED.get());
    }

    #[test]
    fn a_decode_inside_another_restores_the_outer_flag() {
        for (outer, inner) in [(true, false), (false, true), (true, true)] {
            let guard = IgnoringThinkingKeywords::set(outer);
            let Ok(Seen(seen)) = decode_chat::<Seen>(b"{}", inner) else {
                panic!("the body decodes");
            };
            assert_eq!(seen, inner);
            assert_eq!(THINKING_KEYWORDS_IGNORED.get(), outer);
            let panicked = std::panic::catch_unwind(|| decode_chat::<Panics>(b"{}", inner));
            assert!(panicked.is_err());
            assert_eq!(THINKING_KEYWORDS_IGNORED.get(), outer);
            drop(guard);
            assert!(!THINKING_KEYWORDS_IGNORED.get());
        }
    }
}
