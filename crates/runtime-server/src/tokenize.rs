use std::sync::Arc;

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::Request;
use axum::response::{IntoResponse, Response};
use serde_json::{Map, Value, json};

use crate::backend::CancellationSignal;
use crate::error::{ApiError, insert_request_id};
use crate::intake::IntakeCharge;
use crate::request::{self, Endpoint, RequestIdentityV1};
use crate::response::{CancelOnDrop, admission_error, next_request_id};
use crate::{
    FrontendErrorV1, PreparedPrompt, PreparedRequest, RequestAcceptanceProfile, RequestSurfaceV1,
    SamplingDefaults, ServerFrontendV1, ServerState,
};

pub(crate) async fn tokenize(
    State(state): State<Arc<ServerState>>,
    request: Request<Body>,
) -> Response {
    handle(state, request, false).await
}

pub(crate) async fn detokenize(
    State(state): State<Arc<ServerState>>,
    request: Request<Body>,
) -> Response {
    handle(state, request, true).await
}

async fn handle(state: Arc<ServerState>, request: Request<Body>, decode: bool) -> Response {
    let mut waiting = state.waiting.enter(crate::waiting::Stage::Intake);
    let id = next_request_id();
    let peer = request.extensions().get::<std::net::SocketAddr>().copied();
    let result = read(&state, request, decode).await;
    // Tokenizing a large body is synchronous work, kept off the async workers.
    let result = match result {
        Ok((body, intake)) => {
            waiting.move_to(crate::waiting::Stage::Preparation);
            let (state, request_id) = (Arc::clone(&state), id.clone());
            let cancellation = CancellationSignal::new();
            let leaving = CancelOnDrop(Some(cancellation.clone()));
            let result = tokio::task::spawn_blocking(move || {
                let _intake = intake;
                prepare(&state, &body, decode, &request_id, &cancellation)
            })
            .await
            .unwrap_or_else(|_| Err(ApiError::backend("tokenization failed", None)));
            leaving.disarm();
            result
        }
        Err(error) => Err(error),
    };
    match result {
        Ok(value) => {
            let mut response = Json(value).into_response();
            insert_request_id(&mut response, &id);
            response
        }
        Err(error) => {
            error.log_model_refusal(peer, state.frontend.models());
            error.into_response(&id)
        }
    }
}

async fn read(
    state: &ServerState,
    request: Request<Body>,
    decode: bool,
) -> Result<(axum::body::Bytes, IntakeCharge), ApiError> {
    request::validate_headers(request.headers())?;
    request::validate_content_type(request.headers())?;
    // Neither endpoint expands the body; only a chat to tokenize may carry
    // media.
    let limit = if decode {
        state.max_body_bytes
    } else {
        state.max_media_body_bytes
    };
    let reserve = crate::intake::reservation(request.headers(), limit, false);
    state
        .intake
        .read(request.into_body(), limit, state.wait_for_capacity, reserve)
        .await
}

fn prepare(
    state: &ServerState,
    body: &[u8],
    decode: bool,
    id: &str,
    cancellation: &CancellationSignal,
) -> Result<Value, ApiError> {
    let mut value: Value =
        serde_json::from_slice(body).map_err(|error| ApiError::invalid(error.to_string(), None))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| ApiError::invalid("request body must be an object", None))?;
    let model = match object.get("model") {
        None | Some(Value::Null) => state.frontend.models()[0].id.as_str(),
        Some(Value::String(model)) => model.as_str(),
        _ => return Err(ApiError::invalid("model must be a string", Some("model"))),
    };
    if !state
        .frontend
        .models()
        .iter()
        .any(|entry| entry.id == model)
    {
        return Err(ApiError::model_not_found(model));
    }
    let model = model.to_owned();
    if decode {
        let tokens: Vec<u32> = serde_json::from_value(
            object.remove("tokens").unwrap_or(Value::Null),
        )
        .map_err(|_| ApiError::invalid("tokens must be a token-ID array", Some("tokens")))?;
        if tokens.len() > 1_048_576 {
            return Err(ApiError::invalid("too many token IDs", Some("tokens")));
        }
        let prompt = state.backend.detokenize(&tokens).map_err(admission_error)?;
        return Ok(json!({"prompt": prompt}));
    }
    let strings = bool_option(
        object.remove("return_token_strs"),
        false,
        "return_token_strs",
    )?;
    let special = bool_option(
        object.remove("add_special_tokens"),
        !object.contains_key("messages"),
        "add_special_tokens",
    )?;
    let tokens = if object.contains_key("messages") {
        let prepared = parse_chat(
            state.frontend.as_ref(),
            state.sampling_overrides,
            state.max_string_bytes,
            std::mem::take(object),
            (&model, 1),
            id,
            cancellation,
        )?;
        match state
            .backend
            .prepare_prompt_cancellable(prepared.prompt, cancellation)
            .map_err(admission_error)?
            .0
        {
            PreparedPrompt::TokenIds(tokens) | PreparedPrompt::MediaTokens { tokens, .. } => tokens,
            _ => return Err(ApiError::invalid("tokenization is unavailable", None)),
        }
    } else {
        let prompt = object
            .get("prompt")
            .and_then(Value::as_str)
            .ok_or_else(|| ApiError::invalid("prompt must be a string", Some("prompt")))?;
        if prompt.len() > state.max_string_bytes {
            return Err(ApiError::invalid(
                "prompt exceeds the string byte limit",
                Some("prompt"),
            ));
        }
        state
            .backend
            .tokenize(prompt, special)
            .map_err(admission_error)?
    };
    let token_strs = if strings {
        Some(
            state
                .backend
                .token_strings(&tokens)
                .map_err(admission_error)?,
        )
    } else {
        None
    };
    Ok(
        json!({"count": tokens.len(), "max_model_len": state.maximum_context_tokens, "tokens": tokens, "token_strs": token_strs}),
    )
}

/// One chat body as `/tokenize` renders it, through the transport's own
/// request parsing and the frontend's render, asking for `max_tokens`.
fn parse_chat(
    frontend: &dyn ServerFrontendV1,
    sampling_overrides: SamplingDefaults,
    max_string_bytes: usize,
    mut object: Map<String, Value>,
    (model, max_tokens): (&str, usize),
    id: &str,
    cancellation: &CancellationSignal,
) -> Result<PreparedRequest, ApiError> {
    object.insert("model".to_owned(), json!(model));
    object.insert("max_tokens".to_owned(), json!(max_tokens));
    let body =
        serde_json::to_vec(&object).map_err(|error| ApiError::invalid(error.to_string(), None))?;
    request::parse(
        frontend,
        sampling_overrides,
        Endpoint::Chat,
        RequestAcceptanceProfile::General,
        RequestSurfaceV1::Full,
        max_string_bytes,
        &body,
        RequestIdentityV1 {
            request_id: id.to_owned(),
            client_request_id: None,
            generated_seed: 0,
            cancellation: cancellation.clone(),
        },
    )
    .map(|parsed| parsed.prepared)
}

/// Renders one chat request body exactly as `/tokenize` does, without a
/// listener or a backend: the transport's request parsing, then `frontend`'s
/// render. The model defaults to the first served identity, and the request
/// asks for `max_tokens` completion tokens.
///
/// The prepared request carries the rendered prompt, still untokenized, and
/// the chat state, decoding options and stops a backend admits it with. A
/// frontend check reads a vector's request this way.
pub fn prepare_chat_request_v1(
    frontend: &dyn ServerFrontendV1,
    max_string_bytes: usize,
    mut body: Map<String, Value>,
    max_tokens: usize,
) -> Result<PreparedRequest, FrontendErrorV1> {
    let model = match body.remove("model") {
        None | Some(Value::Null) => frontend.models()[0].id.clone(),
        Some(Value::String(model)) => model,
        Some(_) => {
            return Err(FrontendErrorV1::new(
                "model must be a string",
                Some("model"),
            ));
        }
    };
    parse_chat(
        frontend,
        SamplingDefaults::EMPTY,
        max_string_bytes,
        body,
        (&model, max_tokens),
        &next_request_id(),
        &CancellationSignal::new(),
    )
    .map_err(|error| FrontendErrorV1 {
        message: error.message,
        parameter: error.parameter,
    })
}

fn bool_option(value: Option<Value>, default: bool, name: &str) -> Result<bool, ApiError> {
    match value {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(value)) => Ok(value),
        _ => Err(ApiError::invalid("expected a boolean", Some(name))),
    }
}
