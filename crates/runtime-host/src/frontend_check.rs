//! `bundle check-frontend`: a built package's frontend against the model's
//! own vectors.
//!
//! A model proves its frontend, including a format it implements itself, with
//! three kinds of vector, all read from JSON files it keeps beside its other
//! goldens:
//!
//! - render vectors, the `chat-goldens.json` shape: one chat request and the
//!   text and token ids the reference template renders for it, or the
//!   refusal the model gives it;
//! - tokenizer vectors, the `tokenizer-golden.json` shape: one text and the
//!   ids the reference tokenizer encodes it to;
//! - streaming vectors: one chat request, the token ids a model generated for
//!   it, and the reasoning, content, tool calls and finish reason a served
//!   chat response reports for them.
//!
//! Every vector runs through what `serve` runs, not a copy of it. The package
//! is verified, its module opened and its frontend export served over the
//! authenticated component as a serve does. A request is parsed and rendered
//! by the transport as `/tokenize` renders one, and its prompt is tokenized
//! as admission tokenizes it. A completion decodes through the decoder
//! admission builds, fed the way the generation worker feeds it, and its
//! calls and finish reason are read with the transport's own checks.

use std::fmt;
use std::sync::Arc;

use runtime_bundle::BundleManifestV1;
use runtime_bundle_abi::frontend::FrontendExportV1;
use runtime_frontend::{
    BundleFrontendV1, BundleParserSourceV1, CompletionUpdateV1, FrontendDescriptorV1,
    FrontendMediaKindV1,
};
use runtime_server::{
    ChatEventFoldV1, FinishReason, FrontendMediaPartKindV1, GenerationEvent, PreparedPrompt,
    PromptSegmentV1, ServerFrontendV1, prepare_chat_request_v1,
};
use runtime_tokenizer::Tokenizer;
use serde_json::{Map, Value};

use crate::deployment::BundleDeploymentV1;
use crate::generation::{
    DEFAULT_TOKENIZER_INPUT_BYTES, EngineModelPolicyV1, TokenAccounting, completion_decoder_v1,
    generated_token_v1, generation_events_v1, prepare_owned_prompt, request_stop_ids_v1,
};

/// The request fields a render case or a streaming vector's request may
/// carry besides `messages`, as an OpenAI chat request names them.
const REQUEST_FIELDS_V1: [&str; 8] = [
    "tools",
    "tool_choice",
    "chat_template_kwargs",
    "reasoning_effort",
    "thinking",
    "response_format",
    "add_generation_prompt",
    "continue_final_message",
];

/// What a render vector renders media parts with: a render places media
/// and never reads them, so a `file://` source the capture named becomes a
/// one-byte inline source of the same kind.
const RENDERED_MEDIA_SOURCES_V1: [(&str, &str); 3] = [
    ("image_url", "data:image/png;base64,AA=="),
    ("video_url", "data:video/mp4;base64,AA=="),
    ("audio_url", "data:audio/wav;base64,AA=="),
];

/// The kinds of vector, as each case line names them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontendVectorKindV1 {
    Render,
    Tokenizer,
    Streaming,
}

impl FrontendVectorKindV1 {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Render => "render",
            Self::Tokenizer => "tokenizer",
            Self::Streaming => "streaming",
        }
    }
}

/// One chat request a vector names: `messages` and any of
/// [`REQUEST_FIELDS_V1`], kept as JSON in the order the file wrote them.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorRequestV1 {
    fields: Map<String, Value>,
    /// Whether the fields are sent exactly as written, without the capture
    /// normalization of [`VectorRequestV1::body`].
    raw: bool,
}

/// One render vector.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderVectorV1 {
    pub id: String,
    pub request: VectorRequestV1,
    pub expect: ExpectedRenderV1,
}

/// What a render vector's request must produce.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExpectedRenderV1 {
    /// The prompt text and token ids the reference template renders.
    Rendered { text: String, token_ids: Vec<u32> },
    /// A refusal by the transport or the frontend.
    Refused(ExpectedRefusalV1),
}

/// The refusal a render vector expects: any refusal, or one with the
/// message and parameter the vector states.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExpectedRefusalV1 {
    pub message: Option<String>,
    pub parameter: Option<String>,
}

/// One tokenizer vector. The file names none, so its id is its position.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenizerVectorV1 {
    pub id: String,
    pub text: String,
    pub token_ids: Vec<u32>,
    pub decoded: Option<String>,
}

/// One tool call a streaming vector expects, with its arguments as the
/// format writes them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpectedToolCallV1 {
    pub name: String,
    pub arguments: String,
}

/// What a streaming vector's tokens must decode to.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpectedCompletionV1 {
    pub reasoning: String,
    pub content: String,
    pub tool_calls: Vec<ExpectedToolCallV1>,
    pub finish_reason: FinishReason,
}

/// The streamed deltas one step publishes for one tool call: the name
/// when the step opens the call, and the argument text the step adds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpectedDeltaV1 {
    pub index: usize,
    pub name: Option<String>,
    pub arguments: String,
}

/// One step of a streaming vector: tokens, and what the response publishes
/// once they are delivered.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExpectedStepV1 {
    pub token_ids: Vec<u32>,
    pub reasoning: String,
    pub content: String,
    /// The step's streamed tool-call deltas, merged per call in the order
    /// the calls opened.
    pub tool_call_deltas: Vec<ExpectedDeltaV1>,
}

/// One streaming vector.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamingVectorV1 {
    pub id: String,
    pub request: VectorRequestV1,
    pub token_ids: Vec<u32>,
    pub expect: ExpectedCompletionV1,
    /// What each step publishes, when the vector pins it; the steps' tokens
    /// are `token_ids` in order.
    pub steps: Option<Vec<ExpectedStepV1>>,
    /// The template's history text after the completion's stop token, which
    /// a round-trip vector records. The check reads it as a string and
    /// never decodes it: a served completion ends at its stop token.
    pub separator: Option<String>,
}

/// One case's verdict: `Ok` or the first difference found.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrontendCaseOutcomeV1 {
    pub kind: FrontendVectorKindV1,
    pub id: String,
    pub result: Result<(), String>,
}

impl fmt::Display for FrontendCaseOutcomeV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.result {
            Ok(()) => write!(formatter, "pass {} {}", self.kind.name(), self.id),
            Err(difference) => write!(
                formatter,
                "FAIL {} {}: {difference}",
                self.kind.name(),
                self.id
            ),
        }
    }
}

/// Reads a render vector file: a JSON array of cases.
///
/// A case expects `rendered_text` and `token_ids`, or a `refusal` in their
/// place. A `raw` case is sent exactly as written and may carry no field
/// besides its id, `raw`, its expectation and the request fields.
pub fn read_render_vectors_v1(bytes: &[u8]) -> Result<Vec<RenderVectorV1>, String> {
    cases_v1(bytes)?
        .into_iter()
        .enumerate()
        .map(|(index, mut case)| {
            let id = case_id_v1(&mut case, index)?;
            let context = |error: String| format!("render case {id}: {error}");
            let raw = match case.remove("raw") {
                None => false,
                Some(Value::Bool(raw)) => raw,
                Some(_) => return Err(context("raw must be a boolean".into())),
            };
            let expect = match case.remove("refusal") {
                None => ExpectedRenderV1::Rendered {
                    text: take_string_v1(&mut case, "rendered_text").map_err(context)?,
                    token_ids: take_ids_v1(&mut case, "token_ids").map_err(context)?,
                },
                Some(Value::Object(refusal)) => {
                    if let Some(field) = ["rendered_text", "token_ids"]
                        .into_iter()
                        .find(|field| case.contains_key(*field))
                    {
                        return Err(context(format!("a refusal case cannot carry {field}")));
                    }
                    ExpectedRenderV1::Refused(refusal_v1(refusal).map_err(context)?)
                }
                Some(_) => return Err(context("refusal must be an object".into())),
            };
            // A capture may carry fields of its own, such as a category, so
            // only the request fields are read from what remains. A raw case
            // is a request as a client wrote it, so any other field is
            // refused rather than dropped.
            let mut request = request_v1(case, raw).map_err(context)?;
            request.raw = raw;
            Ok(RenderVectorV1 {
                id,
                request,
                expect,
            })
        })
        .collect()
}

/// A render case's expected refusal: an optional string `message` and
/// `parameter`, and nothing else.
fn refusal_v1(mut refusal: Map<String, Value>) -> Result<ExpectedRefusalV1, String> {
    let mut text = |field: &str| match refusal.remove(field) {
        None => Ok(None),
        Some(Value::String(text)) => Ok(Some(text)),
        Some(_) => Err(format!("refusal.{field} must be a string")),
    };
    let message = text("message")?;
    let parameter = text("parameter")?;
    if let Some(field) = refusal.keys().next() {
        return Err(format!("unknown field refusal.{field}"));
    }
    Ok(ExpectedRefusalV1 { message, parameter })
}

/// Reads a tokenizer vector file: a JSON array of `{text, token_ids}`, each
/// with an optional `decoded`.
pub fn read_tokenizer_vectors_v1(bytes: &[u8]) -> Result<Vec<TokenizerVectorV1>, String> {
    cases_v1(bytes)?
        .into_iter()
        .enumerate()
        .map(|(index, mut case)| {
            let id = format!("#{index}");
            let text = take_string_v1(&mut case, "text")
                .map_err(|error| format!("tokenizer case {id}: {error}"))?;
            let token_ids = take_ids_v1(&mut case, "token_ids")
                .map_err(|error| format!("tokenizer case {id}: {error}"))?;
            let decoded = match case.remove("decoded") {
                None => None,
                Some(Value::String(decoded)) => Some(decoded),
                Some(_) => return Err(format!("tokenizer case {id}: decoded must be a string")),
            };
            Ok(TokenizerVectorV1 {
                id,
                text,
                token_ids,
                decoded,
            })
        })
        .collect()
}

/// Reads a streaming vector file: a JSON array of `{id, request, token_ids,
/// expect}` with optional `steps` and `separator`. Every field is this
/// schema's own, so any other is refused.
pub fn read_streaming_vectors_v1(bytes: &[u8]) -> Result<Vec<StreamingVectorV1>, String> {
    cases_v1(bytes)?
        .into_iter()
        .enumerate()
        .map(|(index, mut case)| {
            let id = case_id_v1(&mut case, index)?;
            let context = |error: String| format!("streaming case {id}: {error}");
            let request = match case.remove("request") {
                Some(Value::Object(request)) => request_v1(request, true).map_err(context)?,
                _ => return Err(context("request must be an object".into())),
            };
            let token_ids = take_ids_v1(&mut case, "token_ids").map_err(context)?;
            let expect = match case.remove("expect") {
                Some(Value::Object(expect)) => expected_v1(expect).map_err(context)?,
                _ => return Err(context("expect must be an object".into())),
            };
            let steps = match case.remove("steps") {
                None => None,
                Some(Value::Array(steps)) => Some(steps_v1(steps, &token_ids).map_err(context)?),
                Some(_) => return Err(context("steps must be an array".into())),
            };
            let separator = match case.remove("separator") {
                None => None,
                Some(Value::String(separator)) => Some(separator),
                Some(_) => return Err(context("separator must be a string".into())),
            };
            if let Some(field) = case.keys().next() {
                return Err(context(format!("unknown field {field}")));
            }
            Ok(StreamingVectorV1 {
                id,
                request,
                token_ids,
                expect,
                steps,
                separator,
            })
        })
        .collect()
}

/// A vector's steps, whose tokens must be the vector's `token_ids` in order.
fn steps_v1(steps: Vec<Value>, token_ids: &[u32]) -> Result<Vec<ExpectedStepV1>, String> {
    let steps = steps
        .into_iter()
        .enumerate()
        .map(|(index, step)| {
            let Value::Object(mut step) = step else {
                return Err(format!("step {index} is not an object"));
            };
            let context = |error: String| format!("step {index}: {error}");
            let token_ids = take_ids_v1(&mut step, "token_ids").map_err(context)?;
            if token_ids.is_empty() {
                return Err(context("token_ids must not be empty".into()));
            }
            let mut text = |field: &str| match step.remove(field) {
                None => Ok(String::new()),
                Some(Value::String(text)) => Ok(text),
                Some(_) => Err(context(format!("{field} must be a string"))),
            };
            let reasoning = text("reasoning")?;
            let content = text("content")?;
            let tool_call_deltas = match step.remove("tool_call_deltas") {
                None => Vec::new(),
                Some(Value::Array(deltas)) => deltas
                    .into_iter()
                    .map(delta_v1)
                    .collect::<Option<_>>()
                    .ok_or_else(|| {
                        context(
                            "tool_call_deltas must hold {index, name, arguments} objects with an \
                             index and string name and arguments"
                                .into(),
                        )
                    })?,
                Some(_) => return Err(context("tool_call_deltas must be an array".into())),
            };
            if let Some(field) = step.keys().next() {
                return Err(context(format!("unknown field {field}")));
            }
            Ok(ExpectedStepV1 {
                token_ids,
                reasoning,
                content,
                tool_call_deltas,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if steps.is_empty() {
        return Err("steps must not be empty".into());
    }
    if steps.iter().flat_map(|step| &step.token_ids).ne(token_ids) {
        return Err("the steps' token_ids must be the vector's token_ids in order".into());
    }
    Ok(steps)
}

fn delta_v1(delta: Value) -> Option<ExpectedDeltaV1> {
    let Value::Object(mut delta) = delta else {
        return None;
    };
    let index = usize::try_from(delta.remove("index")?.as_u64()?).ok()?;
    let name = match delta.remove("name") {
        None => None,
        Some(Value::String(name)) => Some(name),
        Some(_) => return None,
    };
    let arguments = match delta.remove("arguments") {
        None => String::new(),
        Some(Value::String(arguments)) => arguments,
        Some(_) => return None,
    };
    delta.is_empty().then_some(ExpectedDeltaV1 {
        index,
        name,
        arguments,
    })
}

fn cases_v1(bytes: &[u8]) -> Result<Vec<Map<String, Value>>, String> {
    let cases: Vec<Value> = serde_json::from_slice(bytes)
        .map_err(|error| format!("a vector file must be a JSON array: {error}"))?;
    cases
        .into_iter()
        .enumerate()
        .map(|(index, case)| match case {
            Value::Object(case) => Ok(case),
            _ => Err(format!("case #{index} is not an object")),
        })
        .collect()
}

fn case_id_v1(case: &mut Map<String, Value>, index: usize) -> Result<String, String> {
    match case.remove("id") {
        Some(Value::String(id)) if !id.is_empty() => Ok(id),
        _ => Err(format!("case #{index} needs a non-empty string id")),
    }
}

fn take_string_v1(case: &mut Map<String, Value>, field: &str) -> Result<String, String> {
    match case.remove(field) {
        Some(Value::String(text)) => Ok(text),
        _ => Err(format!("{field} must be a string")),
    }
}

fn take_ids_v1(case: &mut Map<String, Value>, field: &str) -> Result<Vec<u32>, String> {
    serde_json::from_value(case.remove(field).unwrap_or(Value::Null))
        .map_err(|_| format!("{field} must be an array of token ids"))
}

/// The request fields of `case`; with `strict`, any other field is refused.
fn request_v1(mut case: Map<String, Value>, strict: bool) -> Result<VectorRequestV1, String> {
    let mut fields = Map::new();
    match case.remove("messages") {
        Some(messages @ Value::Array(_)) => {
            fields.insert("messages".into(), messages);
        }
        _ => return Err("messages must be an array".into()),
    }
    for name in REQUEST_FIELDS_V1 {
        if let Some(value) = case.remove(name) {
            fields.insert(name.into(), value);
        }
    }
    if strict && let Some(field) = case.keys().next() {
        return Err(format!("unknown request field {field}"));
    }
    Ok(VectorRequestV1 { fields, raw: false })
}

fn expected_v1(mut expect: Map<String, Value>) -> Result<ExpectedCompletionV1, String> {
    let mut text = |field: &str| match expect.remove(field) {
        None => Ok(String::new()),
        Some(Value::String(text)) => Ok(text),
        Some(_) => Err(format!("expect.{field} must be a string")),
    };
    let reasoning = text("reasoning")?;
    let content = text("content")?;
    let tool_calls = match expect.remove("tool_calls") {
        None => Vec::new(),
        Some(Value::Array(calls)) => calls
            .into_iter()
            .map(|call| match call {
                Value::Object(mut call) => {
                    let name = take_string_v1(&mut call, "name");
                    let arguments = take_string_v1(&mut call, "arguments");
                    match (name, arguments, call.is_empty()) {
                        (Ok(name), Ok(arguments), true) => {
                            Ok(ExpectedToolCallV1 { name, arguments })
                        }
                        _ => Err(()),
                    }
                }
                _ => Err(()),
            })
            .collect::<Result<_, ()>>()
            .map_err(|()| {
                "expect.tool_calls must hold {name, arguments} objects with string values"
                    .to_owned()
            })?,
        Some(_) => return Err("expect.tool_calls must be an array".into()),
    };
    let finish_reason = match expect.remove("finish_reason") {
        Some(Value::String(reason)) => match reason.as_str() {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            "tool_calls" => FinishReason::ToolCalls,
            _ => return Err("expect.finish_reason must be stop, length or tool_calls".into()),
        },
        _ => return Err("expect.finish_reason must be stop, length or tool_calls".into()),
    };
    if let Some(field) = expect.keys().next() {
        return Err(format!("unknown field expect.{field}"));
    }
    Ok(ExpectedCompletionV1 {
        reasoning,
        content,
        tool_calls,
        finish_reason,
    })
}

impl VectorRequestV1 {
    /// The chat request body this vector stands for, read the way the model
    /// golden tests read a capture. Tool-call arguments given as JSON values
    /// become their compact JSON text, as a client sends them. A tool result
    /// without `tool_call_id`, and a tool call without `id` or `type`, get an
    /// empty id and type `function`. With `history_reasoning`, an assistant
    /// turn's `reasoning_content` is sent under that field when the turn has
    /// none. A `file://` media source becomes an inline one, since a render
    /// places media without reading them. Everything else is sent as the
    /// vector wrote it.
    ///
    /// A raw request is sent exactly as written, with none of these changes,
    /// so a missing id or a reasoning field tests what a client sent. Its
    /// media sources are not rewritten either: a raw vector writes an inline
    /// source a client could send, and a `file://` one is refused as the
    /// transport refuses it without a local media allowlist.
    fn body(&self, history_reasoning: Option<&str>) -> Map<String, Value> {
        let mut body = self.fields.clone();
        if self.raw {
            return body;
        }
        if let Some(Value::Array(messages)) = body.get_mut("messages") {
            for message in messages.iter_mut().filter_map(Value::as_object_mut) {
                if let Some(field) = history_reasoning
                    && message.get("role").and_then(Value::as_str) == Some("assistant")
                    && !message.contains_key(field)
                    && let Some(reasoning) = message.remove("reasoning_content")
                {
                    message.insert(field.to_owned(), reasoning);
                }
                if message.get("role").and_then(Value::as_str) == Some("tool") {
                    message
                        .entry("tool_call_id")
                        .or_insert_with(|| Value::String(String::new()));
                }
                if let Some(Value::Array(calls)) = message.get_mut("tool_calls") {
                    for call in calls.iter_mut().filter_map(Value::as_object_mut) {
                        call.entry("id")
                            .or_insert_with(|| Value::String(String::new()));
                        call.entry("type")
                            .or_insert_with(|| Value::String("function".into()));
                        if let Some(Value::Object(function)) = call.get_mut("function")
                            && let Some(arguments) = function.get_mut("arguments")
                            && !arguments.is_string()
                        {
                            *arguments = Value::String(arguments.to_string());
                        }
                    }
                }
                if let Some(Value::Array(parts)) = message.get_mut("content") {
                    for part in parts.iter_mut().filter_map(Value::as_object_mut) {
                        for (kind, inline) in RENDERED_MEDIA_SOURCES_V1 {
                            if let Some(Value::Object(media)) = part.get_mut(kind)
                                && let Some(url) = media.get_mut("url")
                                && url.as_str().is_some_and(|url| url.starts_with("file://"))
                            {
                                *url = Value::String(inline.to_owned());
                            }
                        }
                    }
                }
            }
        }
        body
    }
}

/// One served frontend, its generation policy and its tokenizer: what a
/// check runs vectors against.
pub struct FrontendCheckV1 {
    pub(crate) frontend: BundleFrontendV1,
    pub(crate) policy: EngineModelPolicyV1,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) max_string_bytes: usize,
    max_output_bytes: usize,
}

impl FrontendCheckV1 {
    /// The frontend a serve of `manifest`'s package serves out of
    /// `deployment`, which `export` was opened for.
    pub(crate) fn from_deployment(
        manifest: &BundleManifestV1,
        export: &FrontendExportV1,
        deployment: BundleDeploymentV1,
        config: runtime_server::ServerConfig,
    ) -> Result<Self, String> {
        let media_inputs = crate::video::media_inputs_required_v1(manifest);
        let frontend = crate::command::served_frontend_v1(export, &deployment, media_inputs)
            .map_err(|error| error.to_string())?;
        let policy = deployment.policy().with_media_inputs(media_inputs);
        Ok(Self {
            frontend,
            policy,
            tokenizer: deployment.into_tokenizer(),
            max_string_bytes: config.max_string_bytes(),
            max_output_bytes: config.max_output_bytes(),
        })
    }

    /// A check of a frontend declaration that `export` serves, without a
    /// package: a model's own tests run its vectors through the declared
    /// format's loopback this way. `component` is the declaration's bytes,
    /// and media are served when the declaration has any.
    pub fn from_component_v1(
        export: &FrontendExportV1,
        component: &[u8],
        tokenizer: Tokenizer,
    ) -> Result<Self, String> {
        let declaration =
            FrontendDescriptorV1::from_component_bytes(component).map_err(|e| e.to_string())?;
        let media_inputs = declaration.multimodal.is_some();
        let policy = EngineModelPolicyV1::from_declaration(&declaration)
            .with_media_inputs(media_inputs)
            .with_parser_source(
                BundleParserSourceV1::from_export(export, component).map_err(|e| e.to_string())?,
            );
        let frontend =
            BundleFrontendV1::from_export(export, component).map_err(|error| error.to_string())?;
        let config = runtime_server::ServerConfig::default();
        Ok(Self {
            frontend,
            policy,
            tokenizer,
            max_string_bytes: config.max_string_bytes(),
            max_output_bytes: config.max_output_bytes(),
        })
    }

    /// Runs every vector, in file order: render, then tokenizer, then
    /// streaming.
    #[must_use]
    pub fn run_v1(
        &self,
        render: &[RenderVectorV1],
        tokenizer: &[TokenizerVectorV1],
        streaming: &[StreamingVectorV1],
    ) -> Vec<FrontendCaseOutcomeV1> {
        let outcome = |kind, id: &str, result| FrontendCaseOutcomeV1 {
            kind,
            id: id.to_owned(),
            result,
        };
        render
            .iter()
            .map(|vector| {
                outcome(
                    FrontendVectorKindV1::Render,
                    &vector.id,
                    self.check_render_v1(vector),
                )
            })
            .chain(tokenizer.iter().map(|vector| {
                outcome(
                    FrontendVectorKindV1::Tokenizer,
                    &vector.id,
                    self.check_tokenizer_v1(vector),
                )
            }))
            .chain(streaming.iter().map(|vector| {
                outcome(
                    FrontendVectorKindV1::Streaming,
                    &vector.id,
                    self.check_streaming_v1(vector),
                )
            }))
            .collect()
    }

    /// Renders the vector's request as `/tokenize` does and compares the
    /// prompt text, then its token ids, or checks that the transport or the
    /// frontend refused it as the vector expects.
    ///
    /// The text matches when it equals the expected text or when the
    /// declared tokenizer's round trip of it does, because a capture holds
    /// tokenizer-decoded text and a normalizing tokenizer (NFC) normalizes
    /// it. A prompt with media is compared before the host expands each
    /// media placeholder to the encoder's rows, as the reference template
    /// tokenizes it.
    pub fn check_render_v1(&self, vector: &RenderVectorV1) -> Result<(), String> {
        let prepared = prepare_chat_request_v1(
            &self.frontend,
            self.max_string_bytes,
            vector.request.body(self.history_reasoning_v1()),
            1,
        );
        let (expected_text, expected_ids, prepared) = match (&vector.expect, prepared) {
            (ExpectedRenderV1::Rendered { text, token_ids }, Ok(prepared)) => {
                (text, token_ids, prepared)
            }
            (ExpectedRenderV1::Rendered { .. }, Err(error)) => {
                return Err(format!("the request was refused: {}", error.message));
            }
            (ExpectedRenderV1::Refused(expected), Err(error)) => {
                return expected.check_v1(&error.message, error.parameter.as_deref());
            }
            (ExpectedRenderV1::Refused(_), Ok(_)) => {
                return Err("the request rendered, but a refusal was expected".into());
            }
        };
        let (text, ids) = match prepared.prompt {
            PreparedPrompt::MediaChat { segments, media } => {
                let text = self.media_prompt_text_v1(&segments, &media)?;
                let ids = self.encode_v1(&text);
                (text, ids)
            }
            PreparedPrompt::RenderedChat(text) => {
                let ids = prepare_owned_prompt(
                    &self.tokenizer,
                    PreparedPrompt::RenderedChat(text.clone()),
                    &self.policy,
                    DEFAULT_TOKENIZER_INPUT_BYTES,
                )
                .map_err(|error| format!("the prompt was refused: {error:?}"));
                (text, ids)
            }
            _ => return Err("the transport prepared no rendered chat".into()),
        };
        // The text is compared first, so a prompt the tokenizer refuses still
        // reports how its text differs.
        let round_trip = ids.as_deref().ok().map(|ids| self.round_trip_v1(ids));
        if &text != expected_text
            && !matches!(&round_trip, Some(Ok(decoded)) if decoded == expected_text)
        {
            return Err(format!(
                "rendered text {}",
                text_difference_v1(expected_text, &text)
            ));
        }
        ids_match_v1(expected_ids, &ids?)
    }

    /// Encodes the vector's text as `/tokenize` encodes a prompt, and
    /// decodes the ids as `/detokenize` does.
    pub fn check_tokenizer_v1(&self, vector: &TokenizerVectorV1) -> Result<(), String> {
        let ids = self.encode_v1(&vector.text)?;
        ids_match_v1(&vector.token_ids, &ids)?;
        if let Some(decoded) = &vector.decoded {
            let actual = self.round_trip_v1(&ids)?;
            if &actual != decoded {
                return Err(format!(
                    "decoded text {}",
                    text_difference_v1(decoded, &actual)
                ));
            }
        }
        Ok(())
    }

    /// Decodes the vector's tokens once with tool calls delivered whole and
    /// once streamed. Each delivery must report the expected completion and,
    /// where the vector pins its steps, publish each step's reasoning and
    /// content, and in the streamed delivery its tool-call deltas.
    pub fn check_streaming_v1(&self, vector: &StreamingVectorV1) -> Result<(), String> {
        for stream in [false, true] {
            let delivery = if stream { "streamed" } else { "whole" };
            let (decoded, steps) = self
                .decode_v1(vector, stream)
                .map_err(|error| format!("{delivery}: {error}"))?;
            if let Some(difference) = decoded.difference(&vector.expect) {
                return Err(format!("{delivery}: {difference}"));
            }
            for (index, (published, expected)) in
                steps.iter().zip(vector.steps.iter().flatten()).enumerate()
            {
                if let Some(difference) = published.difference(expected, stream) {
                    return Err(format!("{delivery}: step {index} {difference}"));
                }
            }
        }
        Ok(())
    }

    /// Decodes the vector's tokens as the generation worker does and checks
    /// each event as the transport does, returning the completion a chat
    /// response reports and what each of the vector's steps published.
    fn decode_v1(
        &self,
        vector: &StreamingVectorV1,
        stream: bool,
    ) -> Result<(ExpectedCompletionV1, Vec<ExpectedStepV1>), String> {
        let mut body = vector.request.body(self.history_reasoning_v1());
        body.insert("stream".into(), Value::Bool(stream));
        // The completion may take every vector token: one that ends without
        // a stop token ends at this limit, as a served one does.
        let prepared = prepare_chat_request_v1(
            &self.frontend,
            self.max_string_bytes,
            body,
            vector.token_ids.len().max(1),
        )
        .map_err(|error| format!("the request was refused: {}", error.message))?;
        let prompt = prepare_owned_prompt(
            &self.tokenizer,
            prepared.prompt.clone(),
            &self.policy,
            DEFAULT_TOKENIZER_INPUT_BYTES,
        )
        .map_err(|error| format!("the prompt was refused: {error:?}"))?;
        let mut decoder = completion_decoder_v1(&self.tokenizer, &self.policy, &prepared, &prompt)
            .map_err(|error| format!("admission refused the request: {error:?}"))?;
        let (mut stop_ids, ignored) = request_stop_ids_v1(&self.policy, &prepared.decoding);
        stop_ids.retain(|id| !ignored.contains(id));
        let frontend: Arc<dyn ServerFrontendV1> = Arc::new(self.frontend.clone());
        let mut delivery = DeliveryV1 {
            fold: ChatEventFoldV1::new(frontend, &prepared, self.max_output_bytes, stream)
                .map_err(|error| format!("the transport refused the request: {}", error.message))?,
            accounting: TokenAccounting::default(),
            prompt_tokens: u64::try_from(prompt.len()).map_err(|_| "the prompt is too long")?,
            step: ExpectedStepV1::default(),
            finish: None,
        };
        // Where each step ends, as a count of tokens; without steps, the
        // whole completion is one.
        let ends: Vec<usize> = match &vector.steps {
            Some(steps) => steps
                .iter()
                .scan(0, |end, step| {
                    *end += step.token_ids.len();
                    Some(*end)
                })
                .collect(),
            None => vec![vector.token_ids.len()],
        };
        let mut steps = Vec::with_capacity(ends.len());
        for (ordinal, &token_id) in vector.token_ids.iter().enumerate() {
            if delivery.finish.is_some() {
                return Err(format!("token {ordinal} follows the end of the completion"));
            }
            let generated = ordinal + 1;
            let token = generated_token_v1(&self.tokenizer, ordinal as u64, token_id)
                .map_err(|error| format!("token {ordinal} ({token_id}): {error}"))?;
            let updates = decoder
                .push_engine_token(token)
                .map_err(|error| format!("token {ordinal} ({token_id}): {error}"))?;
            delivery.deliver(updates, generated)?;
            // The engine ends a generation at a stop token, after
            // delivering it; the worker then finishes the decoder.
            if delivery.finish.is_none() && stop_ids.binary_search(&token_id).is_ok() {
                let updates = decoder
                    .finish_stop()
                    .map_err(|error| format!("finishing at a stop token: {error}"))?;
                delivery.deliver(updates, generated)?;
            }
            // The last step also holds what the completion's end delivers.
            if ends.contains(&generated) && generated < vector.token_ids.len() {
                steps.push(std::mem::take(&mut delivery.step));
            }
        }
        if delivery.finish.is_none() {
            let updates = decoder
                .finish_length()
                .map_err(|error| format!("finishing at the token limit: {error}"))?;
            delivery.deliver(updates, vector.token_ids.len())?;
        }
        steps.push(std::mem::take(&mut delivery.step));
        let finish_reason = delivery.finish.ok_or("the decoder reported no finish")?;
        let fold = delivery.fold;
        let decoded = ExpectedCompletionV1 {
            reasoning: fold.reasoning().to_owned(),
            content: fold.content().to_owned(),
            tool_calls: fold
                .tool_calls()
                .iter()
                .map(|call| ExpectedToolCallV1 {
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                })
                .collect(),
            finish_reason,
        };
        Ok((decoded, steps))
    }

    /// The text of a media prompt as the frontend rendered it, with each
    /// media segment as its declared placeholder.
    pub(crate) fn media_prompt_text_v1(
        &self,
        segments: &[PromptSegmentV1],
        media: &[runtime_server::FrontendMediaPartV1],
    ) -> Result<String, String> {
        let modalities = self
            .policy
            .declaration()
            .multimodal
            .as_ref()
            .map(|multimodal| multimodal.modalities.as_slice())
            .unwrap_or_default();
        let mut text = String::new();
        for segment in segments {
            match segment {
                PromptSegmentV1::Text(segment) => text.push_str(segment),
                PromptSegmentV1::Media(index) => {
                    let kind = match media.get(*index).map(|part| part.kind) {
                        Some(FrontendMediaPartKindV1::Image) => FrontendMediaKindV1::Image,
                        Some(FrontendMediaPartKindV1::Audio) => FrontendMediaKindV1::Audio,
                        Some(FrontendMediaPartKindV1::Video) => FrontendMediaKindV1::Video,
                        None => return Err("a media segment names no media part".into()),
                    };
                    let modality = modalities
                        .iter()
                        .find(|modality| modality.kind == kind)
                        .ok_or("a media segment has no declared modality")?;
                    text.push_str(&modality.render.pad);
                }
            }
        }
        Ok(text)
    }

    /// The field a capture's history `reasoning_content` is sent under.
    ///
    /// A capture names the history reasoning in the field its reference
    /// template reads. A frontend that reads `reasoning_content` from history
    /// at all, on every assistant turn or on turns with tool calls, renders
    /// that template, so its captures' reasoning is sent under the served
    /// reasoning field. One that never reads it gets the field as captured,
    /// and ignores it as served.
    fn history_reasoning_v1(&self) -> Option<&'static str> {
        let field = self.frontend.reasoning_field();
        (field != "reasoning_content"
            && (self.frontend.reasoning_content_fallback()
                || self.frontend.tool_call_reasoning_fallback()))
        .then_some(field)
    }

    pub(crate) fn encode_v1(&self, text: &str) -> Result<Vec<u32>, String> {
        self.tokenizer
            .encode_with_input_limit(
                text,
                &self.policy.prompt_special_tokens(),
                DEFAULT_TOKENIZER_INPUT_BYTES,
            )
            .map_err(|error| format!("the tokenizer refused the text: {error}"))
    }

    fn round_trip_v1(&self, ids: &[u32]) -> Result<String, String> {
        self.tokenizer
            .decode_bytes(ids)
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .map_err(|error| format!("the tokenizer cannot decode the ids: {error}"))
    }
}

/// One decoded completion on its way to the transport: its updates become
/// the events the generation worker delivers, each checked by the
/// transport's fold, and the current step records what they publish.
struct DeliveryV1 {
    fold: ChatEventFoldV1,
    accounting: TokenAccounting,
    prompt_tokens: u64,
    step: ExpectedStepV1,
    finish: Option<FinishReason>,
}

impl DeliveryV1 {
    /// Delivers one push's updates after `generated` tokens; the worker
    /// stops delivering at the completion's end.
    fn deliver(
        &mut self,
        updates: Vec<CompletionUpdateV1>,
        generated: usize,
    ) -> Result<(), String> {
        for update in updates {
            let finished = matches!(update, CompletionUpdateV1::Finished(_));
            let events = generation_events_v1(
                &mut self.accounting,
                generated,
                (self.prompt_tokens, None),
                update,
            )
            .map_err(|error| format!("the host cannot deliver the decoded output: {error:?}"))?;
            for event in events {
                if let Some(reason) = self.fold.event(&event).map_err(|error| {
                    format!("the transport refused the completion: {}", error.message)
                })? {
                    self.finish = Some(reason);
                }
                self.step.publish(&event);
            }
            if finished {
                break;
            }
        }
        Ok(())
    }
}

impl ExpectedRefusalV1 {
    /// Whether a refusal with `message` and `parameter` is the one expected:
    /// each stated field must equal the refusal's.
    fn check_v1(&self, message: &str, parameter: Option<&str>) -> Result<(), String> {
        let message_matches = self
            .message
            .as_deref()
            .is_none_or(|expected| expected == message);
        let parameter_matches = self.parameter.is_none() || self.parameter.as_deref() == parameter;
        if message_matches && parameter_matches {
            return Ok(());
        }
        Err(format!(
            "the request was refused with {message:?} (parameter {parameter:?}), expected {} \
             (parameter {})",
            self.message.as_ref().map_or_else(
                || "any message".to_owned(),
                |message| format!("{message:?}")
            ),
            self.parameter
                .as_ref()
                .map_or_else(|| "any".to_owned(), |parameter| format!("{parameter:?}")),
        ))
    }
}

impl ExpectedStepV1 {
    /// Records what one checked event publishes.
    fn publish(&mut self, event: &GenerationEvent) {
        match event {
            GenerationEvent::Reasoning(reasoning) => self.reasoning.push_str(&reasoning.delta),
            GenerationEvent::Content(content) => self.content.push_str(&content.delta),
            GenerationEvent::ToolCallDeltas(deltas) => {
                for delta in deltas {
                    match self
                        .tool_call_deltas
                        .iter_mut()
                        .find(|merged| merged.index == delta.index)
                    {
                        Some(merged) => {
                            merged.arguments.push_str(&delta.arguments);
                            if merged.name.is_none() {
                                merged.name.clone_from(&delta.name);
                            }
                        }
                        None => self.tool_call_deltas.push(ExpectedDeltaV1 {
                            index: delta.index,
                            name: delta.name.clone(),
                            arguments: delta.arguments.clone(),
                        }),
                    }
                }
            }
            _ => {}
        }
    }

    /// The first way `self`, what a step published, differs from
    /// `expected`; tool-call deltas are compared only when they streamed.
    fn difference(&self, expected: &Self, streamed: bool) -> Option<String> {
        if self.reasoning != expected.reasoning {
            return Some(format!(
                "reasoning {}",
                text_difference_v1(&expected.reasoning, &self.reasoning)
            ));
        }
        if self.content != expected.content {
            return Some(format!(
                "content {}",
                text_difference_v1(&expected.content, &self.content)
            ));
        }
        (streamed && self.tool_call_deltas != expected.tool_call_deltas).then(|| {
            format!(
                "tool-call deltas: expected {:?}, published {:?}",
                expected.tool_call_deltas, self.tool_call_deltas
            )
        })
    }
}

impl ExpectedCompletionV1 {
    /// The first way `self`, what the tokens decoded to, differs from
    /// `expected`.
    fn difference(&self, expected: &Self) -> Option<String> {
        if self.reasoning != expected.reasoning {
            return Some(format!(
                "reasoning {}",
                text_difference_v1(&expected.reasoning, &self.reasoning)
            ));
        }
        if self.content != expected.content {
            return Some(format!(
                "content {}",
                text_difference_v1(&expected.content, &self.content)
            ));
        }
        if self.tool_calls.len() != expected.tool_calls.len() {
            return Some(format!(
                "{} tool call(s) expected, {} decoded",
                expected.tool_calls.len(),
                self.tool_calls.len()
            ));
        }
        for (index, (actual, expected)) in
            self.tool_calls.iter().zip(&expected.tool_calls).enumerate()
        {
            if actual.name != expected.name {
                return Some(format!(
                    "tool call {index} names {:?}, expected {:?}",
                    actual.name, expected.name
                ));
            }
            if actual.arguments != expected.arguments {
                return Some(format!(
                    "tool call {index} arguments {}",
                    text_difference_v1(&expected.arguments, &actual.arguments)
                ));
            }
        }
        (self.finish_reason != expected.finish_reason).then(|| {
            format!(
                "finish reason {}, expected {}",
                finish_reason_name_v1(self.finish_reason),
                finish_reason_name_v1(expected.finish_reason)
            )
        })
    }
}

const fn finish_reason_name_v1(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
    }
}

/// Where `actual` first departs from `expected`, with a little of each.
fn text_difference_v1(expected: &str, actual: &str) -> String {
    const CONTEXT: usize = 24;
    let at = expected
        .char_indices()
        .zip(actual.chars())
        .find(|((_, expected), actual)| expected != actual)
        .map_or_else(|| expected.len().min(actual.len()), |((at, _), _)| at);
    let excerpt = |text: &str| -> String { text[at..].chars().take(CONTEXT).collect() };
    format!(
        "differs at byte {at}: expected {:?}, got {:?}",
        excerpt(expected),
        excerpt(actual)
    )
}

fn ids_match_v1(expected: &[u32], actual: &[u32]) -> Result<(), String> {
    if expected == actual {
        return Ok(());
    }
    let at = expected
        .iter()
        .zip(actual)
        .position(|(expected, actual)| expected != actual)
        .unwrap_or_else(|| expected.len().min(actual.len()));
    Err(format!(
        "token ids differ at index {at} of {} expected and {} produced: expected {:?}, got {:?}",
        expected.len(),
        actual.len(),
        expected.get(at),
        actual.get(at)
    ))
}

#[cfg(test)]
mod tests {
    use runtime_bundle_abi::frontend::{FrontendParserV1, FrontendV1};
    use runtime_bundle_abi::frontend_document as document;

    use super::*;

    #[test]
    fn render_cases_read_the_golden_shape_and_ignore_capture_fields() {
        let cases = read_render_vectors_v1(
            br#"[{"id":"a","category":"x","messages":[{"role":"user","content":"hi"}],
                 "tools":[],"chat_template_kwargs":{"enable_thinking":false},
                 "rendered_text":"T","token_ids":[1,2]}]"#,
        )
        .unwrap();
        assert_eq!(cases[0].id, "a");
        assert_eq!(
            cases[0].expect,
            ExpectedRenderV1::Rendered {
                text: "T".into(),
                token_ids: vec![1, 2]
            }
        );
        assert!(!cases[0].request.raw);
        let keys: Vec<_> = cases[0].request.fields.keys().cloned().collect();
        assert_eq!(keys, ["messages", "tools", "chat_template_kwargs"]);
        assert!(read_render_vectors_v1(br#"[{"id":"a","messages":[],"token_ids":[]}]"#).is_err());
    }

    #[test]
    fn render_cases_read_raw_requests_and_expected_refusals() {
        let cases = read_render_vectors_v1(
            br#"[{"id":"a","raw":true,"messages":[],"tools":[],"rendered_text":"T","token_ids":[]},
                 {"id":"b","raw":false,"category":"x","messages":[],"refusal":{}},
                 {"id":"c","raw":true,"messages":[],
                  "refusal":{"message":"m","parameter":"messages"}},
                 {"id":"d","messages":[],"refusal":{"parameter":"tools"}}]"#,
        )
        .unwrap();
        assert!(cases[0].request.raw);
        assert!(!cases[1].request.raw);
        assert_eq!(
            cases[1].expect,
            ExpectedRenderV1::Refused(ExpectedRefusalV1::default())
        );
        assert_eq!(
            cases[2].expect,
            ExpectedRenderV1::Refused(ExpectedRefusalV1 {
                message: Some("m".into()),
                parameter: Some("messages".into()),
            })
        );
        assert_eq!(
            cases[3].expect,
            ExpectedRenderV1::Refused(ExpectedRefusalV1 {
                message: None,
                parameter: Some("tools".into()),
            })
        );
        for (bad, refusal) in [
            (
                r#"{"raw":1,"messages":[],"refusal":{}}"#,
                "raw must be a boolean",
            ),
            (
                r#"{"raw":true,"category":"x","messages":[],"refusal":{}}"#,
                "unknown request field category",
            ),
            (
                r#"{"raw":true,"model":"m","messages":[],"rendered_text":"","token_ids":[]}"#,
                "unknown request field model",
            ),
            (
                r#"{"messages":[],"refusal":"no"}"#,
                "refusal must be an object",
            ),
            (
                r#"{"messages":[],"refusal":{"message":1}}"#,
                "refusal.message must be a string",
            ),
            (
                r#"{"messages":[],"refusal":{"parameter":null}}"#,
                "refusal.parameter must be a string",
            ),
            (
                r#"{"messages":[],"refusal":{"code":"x"}}"#,
                "unknown field refusal.code",
            ),
            (
                r#"{"messages":[],"refusal":{},"rendered_text":""}"#,
                "a refusal case cannot carry rendered_text",
            ),
            (
                r#"{"messages":[],"refusal":{},"token_ids":[]}"#,
                "a refusal case cannot carry token_ids",
            ),
            (
                r#"{"raw":true,"messages":[],"token_ids":[]}"#,
                "rendered_text must be a string",
            ),
        ] {
            let refused =
                read_render_vectors_v1(format!(r#"[{{"id":"x",{}]"#, &bad[1..]).as_bytes())
                    .unwrap_err();
            assert_eq!(refused, format!("render case x: {refusal}"), "{bad}");
        }
    }

    #[test]
    fn streaming_vectors_read_a_separator_string() {
        let vector = |separator: &str| {
            read_streaming_vectors_v1(
                format!(
                    r#"[{{"id":"s","request":{{"messages":[]}},"token_ids":[3],
                      "expect":{{"finish_reason":"stop"}}{separator}}}]"#
                )
                .as_bytes(),
            )
        };
        assert_eq!(vector("").unwrap()[0].separator, None);
        assert_eq!(
            vector(r#","separator":"\n""#).unwrap()[0]
                .separator
                .as_deref(),
            Some("\n")
        );
        for bad in [
            r#","separator":1"#,
            r#","separator":null"#,
            r#","separator":["\n"]"#,
        ] {
            assert_eq!(
                vector(bad).unwrap_err(),
                "streaming case s: separator must be a string",
                "{bad}"
            );
        }
    }

    #[test]
    fn streaming_vectors_refuse_unknown_fields() {
        let good = br#"[{"id":"s","request":{"messages":[]},"token_ids":[3],
            "expect":{"content":"x","tool_calls":[{"name":"f","arguments":"{}"}],
                      "finish_reason":"tool_calls"}}]"#;
        let vectors = read_streaming_vectors_v1(good).unwrap();
        assert_eq!(vectors[0].expect.finish_reason, FinishReason::ToolCalls);
        assert_eq!(vectors[0].expect.reasoning, "");
        for bad in [
            r#"[{"id":"s","request":{"messages":[],"stream":true},"token_ids":[],"expect":{"finish_reason":"stop"}}]"#,
            r#"[{"id":"s","request":{"messages":[]},"token_ids":[],"expect":{"finish_reason":"stop","extra":1}}]"#,
            r#"[{"id":"s","request":{"messages":[]},"token_ids":[],"expect":{"finish_reason":"done"}}]"#,
            r#"[{"id":"s","request":{"messages":[]},"token_ids":[],"expect":{"finish_reason":"stop"},"x":1}]"#,
        ] {
            assert!(read_streaming_vectors_v1(bad.as_bytes()).is_err(), "{bad}");
        }
    }

    /// A raw request is sent exactly as the vector wrote it: no ids or
    /// types are filled, no reasoning field moves, arguments stay JSON values
    /// and media sources stay as written.
    #[test]
    fn a_raw_body_is_the_request_as_written() {
        let fields: Map<String, Value> = serde_json::from_str(
            r#"{"messages":[
              {"role":"assistant","content":"a","reasoning_content":"r",
               "tool_calls":[{"function":{"name":"h","arguments":{"b":1.50}}}]},
              {"role":"tool","content":"r"},
              {"role":"user","content":[{"type":"image_url","image_url":{"url":"file:///p.png"}}]}],
              "tools":[]}"#,
        )
        .unwrap();
        let request = VectorRequestV1 {
            fields: fields.clone(),
            raw: true,
        };
        assert_eq!(request.body(Some("reasoning")), fields);
        assert_eq!(request.body(None), fields);
    }

    #[test]
    fn a_body_reads_captures_as_the_golden_tests_do() {
        let request = request_v1(
            serde_json::from_str(
                r#"{"messages":[
                  {"role":"assistant","content":"","tool_calls":[{"id":"c","type":"function",
                   "function":{"name":"f","arguments":{"b": 1.50, "a": [1, 2]}}},
                   {"id":"d","type":"function","function":{"name":"g","arguments":"{\"x\": 1}"}}]},
                  {"role":"user","content":[{"type":"image_url","image_url":{"url":"file:///p.png"}},
                   {"type":"image_url","image_url":{"url":"https://example.invalid/p.png"}}]},
                  {"role":"assistant","tool_calls":[{"function":{"name":"h","arguments":"{}"}}]},
                  {"role":"tool","content":"r"}]}"#,
            )
            .unwrap(),
            true,
        )
        .unwrap();
        let body = Value::Object(request.body(None));
        let calls = &body["messages"][0]["tool_calls"];
        assert_eq!(calls[0]["function"]["arguments"], r#"{"b":1.50,"a":[1,2]}"#);
        assert_eq!(calls[1]["function"]["arguments"], r#"{"x": 1}"#);
        let parts = &body["messages"][1]["content"];
        assert_eq!(parts[0]["image_url"]["url"], "data:image/png;base64,AA==");
        assert_eq!(
            parts[1]["image_url"]["url"],
            "https://example.invalid/p.png"
        );
        let call = &body["messages"][2]["tool_calls"][0];
        assert_eq!(call["id"], "");
        assert_eq!(call["type"], "function");
        assert_eq!(body["messages"][3]["tool_call_id"], "");
        assert!(body["messages"][1].get("tool_call_id").is_none());

        let history = request_v1(
            serde_json::from_str(
                r#"{"messages":[{"role":"assistant","content":"a","reasoning_content":"r"},
                  {"role":"assistant","content":"b","reasoning":"s","reasoning_content":"t"}]}"#,
            )
            .unwrap(),
            true,
        )
        .unwrap();
        let renamed = Value::Object(history.body(Some("reasoning")));
        assert_eq!(renamed["messages"][0]["reasoning"], "r");
        assert!(renamed["messages"][0].get("reasoning_content").is_none());
        assert_eq!(renamed["messages"][1]["reasoning"], "s");
        assert_eq!(renamed["messages"][1]["reasoning_content"], "t");
        let kept = Value::Object(history.body(None));
        assert_eq!(kept["messages"][0]["reasoning_content"], "r");
        assert!(kept["messages"][0].get("reasoning").is_none());
    }

    /// A package frontend that declares the extra role `observation` and
    /// renders the roles and template keywords its render document carries.
    /// It refuses a message whose content is `refuse`, and parses a
    /// completion as content delivered at its end.
    struct EchoFrontendV1;

    struct ContentParserV1;

    impl FrontendParserV1 for ContentParserV1 {
        fn push(&mut self, input: &[u8], last: bool) -> Result<Vec<u8>, Vec<u8>> {
            let pushed = document::PushV1::decode(input).unwrap();
            let end = pushed.keep + pushed.suffix.len() as u64;
            let mut events = Vec::new();
            if last {
                if end > 0 {
                    events.push(document::EventV1::Content(0..end));
                }
                events.push(document::EventV1::Finish {
                    reason: document::FinishReasonV1::Stop,
                    completion: None,
                });
            }
            Ok(document::EventsV1 {
                events,
                cursor: if last { end } else { 0 },
            }
            .encode())
        }
    }

    impl FrontendV1 for EchoFrontendV1 {
        type Parser = ContentParserV1;

        fn open(_: &[u8]) -> Result<(Self, Vec<u8>), Vec<u8>> {
            let opened = document::OpenedV1 {
                extra_roles: vec!["observation".into()],
                ..document::OpenedV1::default()
            };
            Ok((Self, opened.encode()))
        }

        fn render(&self, request: &[u8]) -> Result<Vec<u8>, Vec<u8>> {
            let request = document::RenderRequestV1::decode(request).unwrap();
            if request
                .messages
                .iter()
                .any(|message| message.content.as_deref() == Some("refuse"))
            {
                return Err(
                    document::RefusalV1::new("this frontend refuses it", Some("messages")).encode(),
                );
            }
            let mut text: Vec<_> = request
                .messages
                .iter()
                .map(|message| format!("{}:{}", message.role, message.content.as_deref().unwrap()))
                .collect();
            text.extend(request.chat_template_kwargs);
            Ok(document::RenderedV1 {
                segments: vec![document::SegmentV1::Text(text.join("|"))],
                mode: document::ModeV1::Chat,
                effort: "low".into(),
                reasoning_open: false,
                parser_state: Vec::new(),
            }
            .encode())
        }

        fn constraint(&self, _: &[u8]) -> Result<Vec<u8>, Vec<u8>> {
            Ok(Vec::new())
        }

        fn open_parser(&self, _: &[u8]) -> Result<Self::Parser, Vec<u8>> {
            Ok(ContentParserV1)
        }
    }

    /// A check of [`EchoFrontendV1`] over the test tokenizer, whose only
    /// special tokens are `[BOS]` and `[EOS]`, so the declaration's prompt
    /// special tokens are narrowed to them and its prompts tokenize.
    fn echo_check() -> FrontendCheckV1 {
        use crate::generation::stop_tests::{declarations, tokenizer};
        let special = r#""prompt_special_tokens": ["[BOS]", "[EOS]", "[U]", "[A]", "<t>", "</t>"]"#;
        assert!(declarations::KEY_VALUE.contains(special));
        let declaration = declarations::KEY_VALUE
            .replace(special, r#""prompt_special_tokens": ["[BOS]", "[EOS]"]"#);
        FrontendCheckV1::from_component_v1(
            &FrontendExportV1::in_process::<EchoFrontendV1>(),
            declaration.as_bytes(),
            tokenizer(),
        )
        .unwrap()
    }

    /// One render vector over `request`, a JSON object's members.
    fn render_vector(request: &str) -> RenderVectorV1 {
        read_render_vectors_v1(format!(r#"[{{"id":"v",{request}}}]"#).as_bytes())
            .unwrap()
            .remove(0)
    }

    /// A raw vector sends the request as written, so a tool result without
    /// `tool_call_id` is refused as a client's would be, where a capture
    /// vector fills the id and renders it.
    #[test]
    fn a_raw_vector_checks_the_request_as_written() {
        let check = echo_check();
        let messages =
            r#""messages":[{"role":"user","content":"q"},{"role":"tool","content":"r"}]"#;
        let rendered = "user:q|tool:r";
        let ids = check.encode_v1(rendered).unwrap();
        let expect = format!(r#""rendered_text":"{rendered}","token_ids":{ids:?}"#);
        check
            .check_render_v1(&render_vector(&format!("{messages},{expect}")))
            .unwrap();
        assert_eq!(
            check.check_render_v1(&render_vector(&format!(
                r#""raw":true,{messages},{expect}"#
            ))),
            Err("the request was refused: tool message requires tool_call_id".into())
        );
        check
            .check_render_v1(&render_vector(&format!(
                r#""raw":true,{messages},
                   "refusal":{{"message":"tool message requires tool_call_id","parameter":"messages"}}"#
            )))
            .unwrap();
        let answered = r#""messages":[{"role":"user","content":"q"},
            {"role":"tool","tool_call_id":"c","content":"r"}]"#;
        check
            .check_render_v1(&render_vector(&format!(
                r#""raw":true,{answered},{expect}"#
            )))
            .unwrap();
    }

    /// A refusal vector passes when the transport or the frontend refuses
    /// the request with the message and parameter it states, and fails when
    /// the request renders or is refused otherwise.
    #[test]
    fn a_refusal_vector_passes_only_the_refusal_it_names() {
        let check = echo_check();
        let role = r#""messages":[{"role":"other","content":"q"}]"#;
        let refused = r#""messages":[{"role":"user","content":"refuse"}]"#;
        let renders = r#""messages":[{"role":"user","content":"q"}]"#;
        let role_message = "message role must be system, developer, user, assistant, tool, or \
                            one this model declares: observation";
        let frontend_message = "this frontend refuses it";
        for (request, refusal) in [
            (role, "{}".to_owned()),
            (role, format!(r#"{{"message":"{role_message}"}}"#)),
            (role, r#"{"parameter":"messages"}"#.to_owned()),
            (refused, "{}".to_owned()),
            (
                refused,
                format!(r#"{{"message":"{frontend_message}","parameter":"messages"}}"#),
            ),
        ] {
            let vector = render_vector(&format!(r#"{request},"refusal":{refusal}"#));
            assert_eq!(
                check.check_render_v1(&vector),
                Ok(()),
                "{request} {refusal}"
            );
            let raw = render_vector(&format!(r#""raw":true,{request},"refusal":{refusal}"#));
            assert_eq!(
                check.check_render_v1(&raw),
                Ok(()),
                "raw {request} {refusal}"
            );
        }
        for (request, refusal, failure) in [
            (
                renders,
                "{}",
                "the request rendered, but a refusal was expected".to_owned(),
            ),
            (
                refused,
                r#"{"message":"another message"}"#,
                format!(
                    r#"the request was refused with "{frontend_message}" (parameter Some("messages")), expected "another message" (parameter any)"#
                ),
            ),
            (
                refused,
                r#"{"message":"this frontend refuses it","parameter":"tools"}"#,
                format!(
                    r#"the request was refused with "{frontend_message}" (parameter Some("messages")), expected "{frontend_message}" (parameter "tools")"#
                ),
            ),
            (
                role,
                r#"{"parameter":"tools"}"#,
                format!(
                    r#"the request was refused with "{role_message}" (parameter Some("messages")), expected any message (parameter "tools")"#
                ),
            ),
        ] {
            let vector = render_vector(&format!(r#"{request},"refusal":{refusal}"#));
            assert_eq!(
                check.check_render_v1(&vector),
                Err(failure),
                "{request} {refusal}"
            );
        }
    }

    /// A streaming vector's separator is never decoded: the same tokens
    /// decode to the same completion whatever the separator says.
    #[test]
    fn a_streaming_separator_is_not_decoded() {
        // The declaration's one stop token, its end of sequence.
        const EOS: u32 = 1;
        let check = echo_check();
        let x = u32::from(b'x') + 3;
        for separator in ["", r#","separator":"\n""#, r#","separator":"y[EOS]z""#] {
            let vector = read_streaming_vectors_v1(
                format!(
                    r#"[{{"id":"s","request":{{"messages":[{{"role":"user","content":"q"}}]}},
                      "token_ids":[{x},{EOS}],"expect":{{"content":"x","finish_reason":"stop"}}
                      {separator}}}]"#
                )
                .as_bytes(),
            )
            .unwrap()
            .remove(0);
            assert_eq!(check.check_streaming_v1(&vector), Ok(()), "{separator}");
        }
    }

    /// A check renders a vector's request through the request path serve
    /// runs, so a package frontend receives the whole `chat_template_kwargs`
    /// object and the extra roles it declares, and the refusal of any other
    /// role names them.
    #[test]
    fn a_vector_request_carries_the_template_keywords_and_declared_roles() {
        use crate::generation::stop_tests::{declarations, tokenizer};
        let check = FrontendCheckV1::from_component_v1(
            &FrontendExportV1::in_process::<EchoFrontendV1>(),
            declarations::KEY_VALUE.as_bytes(),
            tokenizer(),
        )
        .unwrap();
        let vector = |role: &str| {
            let case = format!(
                r#"[{{"id":"r","messages":[{{"role":"user","content":"q"}},{{"role":"{role}","content":"seen"}}],
                  "chat_template_kwargs":{{"custom":[1.50,null],"enable_thinking":false}},
                  "rendered_text":"","token_ids":[]}}]"#
            );
            read_render_vectors_v1(case.as_bytes()).unwrap().remove(0)
        };
        // The request as check_render_v1 prepares it.
        let prepared = prepare_chat_request_v1(
            &check.frontend,
            check.max_string_bytes,
            vector("observation")
                .request
                .body(check.history_reasoning_v1()),
            1,
        )
        .unwrap();
        assert_eq!(
            prepared.prompt,
            PreparedPrompt::RenderedChat(
                r#"user:q|observation:seen|{"custom":[1.50,null],"enable_thinking":false}"#.into()
            )
        );
        assert_eq!(
            check.check_render_v1(&vector("other")),
            Err(
                "the request was refused: message role must be system, developer, user, \
                 assistant, tool, or one this model declares: observation"
                    .into()
            )
        );
    }

    #[test]
    fn steps_must_cover_the_vector_tokens_in_order() {
        let vector = |steps: &str| {
            read_streaming_vectors_v1(
                format!(
                    r#"[{{"id":"s","request":{{"messages":[]}},"token_ids":[1,2,3],
                      "expect":{{"finish_reason":"stop"}},"steps":{steps}}}]"#
                )
                .as_bytes(),
            )
        };
        let steps = vector(
            r#"[{"token_ids":[1],"reasoning":"r"},{"token_ids":[2,3],"content":"c",
                "tool_call_deltas":[{"index":0,"name":"f","arguments":"{"},{"index":1}]}]"#,
        )
        .unwrap()
        .remove(0)
        .steps
        .unwrap();
        assert_eq!(steps[0].reasoning, "r");
        assert_eq!(steps[1].token_ids, [2, 3]);
        assert_eq!(
            steps[1].tool_call_deltas,
            [
                ExpectedDeltaV1 {
                    index: 0,
                    name: Some("f".into()),
                    arguments: "{".into()
                },
                ExpectedDeltaV1 {
                    index: 1,
                    name: None,
                    arguments: String::new()
                }
            ]
        );
        for (bad, refusal) in [
            (r#"[{"token_ids":[1,2]}]"#, "the steps' token_ids must be"),
            (r#"[{"token_ids":[1,3,2]}]"#, "the steps' token_ids must be"),
            (r#"[]"#, "steps must not be empty"),
            (
                r#"[{"token_ids":[]},{"token_ids":[1,2,3]}]"#,
                "token_ids must not be empty",
            ),
            (
                r#"[{"token_ids":[1,2,3],"extra":1}]"#,
                "unknown field extra",
            ),
            (
                r#"[{"token_ids":[1,2,3],"tool_call_deltas":[{"name":"f"}]}]"#,
                "tool_call_deltas must hold",
            ),
        ] {
            let refused = vector(bad).unwrap_err();
            assert!(refused.contains(refusal), "{bad}: {refused}");
        }
    }

    #[test]
    fn a_step_publishes_its_deltas_merged_per_call() {
        let mut step = ExpectedStepV1::default();
        let delta = |index, name: Option<&str>, arguments: &str| runtime_server::ToolCallDelta {
            index,
            name: name.map(str::to_owned),
            arguments: arguments.to_owned(),
        };
        step.publish(&GenerationEvent::ToolCallDeltas(vec![
            delta(0, Some("f"), ""),
            delta(0, None, "{\"a\""),
        ]));
        step.publish(&GenerationEvent::ToolCallDeltas(vec![
            delta(0, None, ": 1}"),
            delta(1, Some("g"), "{}"),
        ]));
        assert_eq!(
            step.tool_call_deltas,
            [
                ExpectedDeltaV1 {
                    index: 0,
                    name: Some("f".into()),
                    arguments: "{\"a\": 1}".into()
                },
                ExpectedDeltaV1 {
                    index: 1,
                    name: Some("g".into()),
                    arguments: "{}".into()
                }
            ]
        );
        let expected = ExpectedStepV1 {
            tool_call_deltas: Vec::new(),
            ..step.clone()
        };
        assert!(step.difference(&expected, false).is_none());
        assert!(
            step.difference(&expected, true)
                .unwrap()
                .starts_with("tool-call deltas: expected []")
        );
    }

    #[test]
    fn differences_name_the_first_departure() {
        assert_eq!(
            text_difference_v1("abcdef", "abXdef"),
            r#"differs at byte 2: expected "cdef", got "Xdef""#
        );
        assert_eq!(
            text_difference_v1("ab", "abc"),
            r#"differs at byte 2: expected "", got "c""#
        );
        assert_eq!(
            ids_match_v1(&[1, 2, 3], &[1, 2]).unwrap_err(),
            "token ids differ at index 2 of 3 expected and 2 produced: expected Some(3), got None"
        );
        assert!(ids_match_v1(&[1], &[1]).is_ok());
    }
}
