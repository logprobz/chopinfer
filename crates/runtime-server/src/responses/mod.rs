//! OpenAI Responses transport, sharing chat preparation and event validation.
mod request;
mod store;
mod stream;

use axum::{
    Json,
    body::{Body, Bytes},
    extract::{Path, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};
use std::{convert::Infallible, sync::Arc};
use tokio::sync::mpsc;
use tokio_stream::{StreamExt, wrappers::ReceiverStream};

use crate::{
    CancellationSignal, ServerState,
    error::{ApiError, insert_request_id},
};
pub(crate) use store::Store;

pub(crate) struct Context {
    request: request::Request,
    history: Vec<Value>,
    initial: Value,
    reasoning_field: String,
    /// The separator of a qualified function name, for a frontend that
    /// qualifies names ([`crate::ServerFrontendV1::tool_namespace_separator`]).
    separator: Option<String>,
    reservation: Option<StoredGuard>,
}
impl Context {
    pub(crate) fn prepare(
        state: &ServerState,
        body: &[u8],
        id: &str,
        charge: &mut crate::intake::IntakeCharge,
    ) -> Result<(Vec<u8>, Self), ApiError> {
        let request = request::Request::parse(body, state.max_string_bytes)?;
        if request.background.unwrap_or(false) {
            if !state.responses.enabled() || !request.store.unwrap_or(true) {
                let mut error = request::invalid(
                    "background requires enabled Responses storage and store=true",
                    "background",
                );
                error.kind = "invalid_request_error";
                return Err(error);
            }
            if request.stream.unwrap_or(false) {
                return Err(request::invalid(
                    "background streaming and event replay are not supported",
                    "stream",
                ));
            }
        }
        let history = request
            .previous_response_id
            .as_deref()
            .map(|id| state.responses.history(id, &state.intake, charge))
            .transpose()?
            .unwrap_or_default();
        let reasoning_field = state.frontend.reasoning_field().to_owned();
        let separator = state.frontend.tool_namespace_separator().map(str::to_owned);
        let chat = request.chat(
            history,
            &reasoning_field,
            separator.as_deref(),
            state.frontend.tool_result_images(),
        )?;
        let mut history = chat["messages"].as_array().expect("chat messages").clone();
        // Instructions belong to this response only, including when it is
        // continued by previous_response_id.
        if request.instructions.is_some() {
            history.remove(0);
        }
        let bytes = serde_json::to_vec(&chat).expect("chat JSON");
        if bytes.len() > state.max_body_bytes {
            return Err(request::invalid(
                "expanded response history exceeds the request byte limit",
                "input",
            ));
        }
        charge.extend(&state.intake, bytes.len())?;
        let initial = json!({"id":format!("resp_{}", id.trim_start_matches("req_")), "object":"response", "created_at":0,
            "status":"in_progress", "error":null, "incomplete_details":null, "output":[], "usage":null,
            "instructions":request.instructions, "metadata":request.metadata, "model":request.model,
            "background":request.background.unwrap_or(false), "parallel_tool_calls":request.parallel_tool_calls.unwrap_or(true),
            "max_output_tokens":request.max_output_tokens, "previous_response_id":request.previous_response_id,
            "reasoning":request.reasoning, "text":request.text, "tools":request.tools.clone().unwrap_or_default(),
            "tool_choice":request.tool_choice.clone().unwrap_or_else(|| json!(if request.tools.as_ref().is_none_or(Vec::is_empty) {"none"} else {"auto"})),
            "temperature":request.temperature, "top_p":request.top_p, "top_logprobs":request.top_logprobs.unwrap_or(0),
            "truncation":"disabled", "service_tier":request.service_tier.clone().unwrap_or_else(|| "auto".to_owned()), "user":request.user});
        // The reference accepts this field but does not use it for caching.
        let _ = &request.prompt_cache_key;
        Ok((
            bytes,
            Self {
                request,
                history,
                initial,
                reasoning_field,
                separator,
                reservation: None,
            },
        ))
    }
    pub(crate) async fn reserve(
        &mut self,
        state: Arc<ServerState>,
        cancellation: CancellationSignal,
    ) -> Result<(), ApiError> {
        if self.request.background.unwrap_or(false) {
            self.initial["status"] = json!("queued");
        }
        let mut guard = StoredGuard { state, id: None };
        if guard.state.responses.enabled() && self.request.store.unwrap_or(true) {
            guard
                .state
                .responses
                .reserve_waiting(
                    &self.initial,
                    &self.history,
                    cancellation,
                    guard.state.max_output_bytes,
                    guard
                        .state
                        .wait_for_capacity
                        .then_some(&guard.state.drain_expired),
                )
                .await?;
            self.history.clear();
            guard.id = self.initial["id"].as_str().map(str::to_owned);
        }
        self.reservation = Some(guard);
        Ok(())
    }

    pub(crate) fn effective(&mut self, prepared: &crate::PreparedRequest, created: u64) {
        self.initial["created_at"] = json!(created);
        self.initial["model"] = json!(prepared.model);
        self.initial["max_output_tokens"] = json!(prepared.max_new_tokens);
        self.initial["temperature"] = json!(prepared.sampling.temperature());
        self.initial["top_p"] = json!(prepared.sampling.top_p());
        self.initial["top_logprobs"] = if prepared.logprobs.enabled {
            json!(prepared.logprobs.top)
        } else {
            Value::Null
        };
    }
}

struct StoredGuard {
    state: Arc<ServerState>,
    id: Option<String>,
}
impl Drop for StoredGuard {
    fn drop(&mut self) {
        if let Some(id) = &self.id {
            self.state.responses.abandon(id);
        }
    }
}
impl StoredGuard {
    fn finish(&mut self, response: Value, output: Vec<Value>) {
        if let Some(id) = self.id.take() {
            self.state.responses.finish(&id, response, output);
        }
    }
}

pub(crate) async fn adapt(
    state: Arc<ServerState>,
    chat: Response,
    mut context: Context,
    cancellation: CancellationSignal,
    request_id: String,
) -> Response {
    let background = context.request.background.unwrap_or(false);
    let mut guard = context
        .reservation
        .take()
        .expect("storage checked before submission");
    let initial = context.initial.clone();
    let stream = context.request.stream.unwrap_or(false);
    let (sender, receiver) = mpsc::channel(1);
    let output = stream.then_some(sender);
    let run = async move {
        let mut events = stream::Events::new(context.initial, context.separator.clone());
        let result = consume(
            chat,
            &mut events,
            &context.reasoning_field,
            output.as_ref(),
            &cancellation,
            state.stream_write_timeout,
        )
        .await;
        if let Err(error) = &result {
            let event = events.error(error);
            let _ = send(output.as_ref(), vec![event], state.stream_write_timeout).await;
        }
        let history = request::input_messages(
            &events.response["output"],
            &context.reasoning_field,
            context.separator.as_deref(),
            false,
        )
        .unwrap_or_default();
        guard.finish(events.response, history);
        result
    };
    if background {
        tokio::spawn(async move {
            let _ = run.await;
        });
        return json_response(initial, &request_id);
    }
    if stream {
        tokio::spawn(async move {
            let _ = run.await;
        });
        let mut response = Response::new(Body::from_stream(ReceiverStream::new(receiver)));
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("text/event-stream"),
        );
        response.headers_mut().insert(
            header::CACHE_CONTROL,
            header::HeaderValue::from_static("no-cache"),
        );
        insert_request_id(&mut response, &request_id);
        response
    } else {
        match run.await {
            Ok(value) => json_response(value, &request_id),
            Err(error) => error.into_response(&request_id),
        }
    }
}

type Sender = mpsc::Sender<Result<Bytes, Infallible>>;
async fn send(
    sender: Option<&Sender>,
    events: Vec<Value>,
    timeout: std::time::Duration,
) -> Result<(), ApiError> {
    let Some(sender) = sender else {
        return Ok(());
    };
    for event in events {
        let frame = format!(
            "event: {}\ndata: {}\n\n",
            event["type"].as_str().expect("event type"),
            event
        );
        if !matches!(
            tokio::time::timeout(timeout, sender.send(Ok(Bytes::from(frame)))).await,
            Ok(Ok(()))
        ) {
            return Err(ApiError::backend(
                "Responses stream reader disconnected or timed out",
                None,
            ));
        }
    }
    Ok(())
}
async fn consume(
    chat: Response,
    events: &mut stream::Events,
    reasoning_field: &str,
    sender: Option<&Sender>,
    cancellation: &CancellationSignal,
    timeout: std::time::Duration,
) -> Result<Value, ApiError> {
    let mut body = chat.into_body().into_data_stream();
    send(sender, events.start(), timeout).await?;
    loop {
        let chunk = tokio::select! {
            biased;
            () = cancellation.cancelled() => return Err(ApiError::backend("response cancelled", None)),
            () = async { if let Some(sender) = sender {sender.closed().await} else {std::future::pending::<()>().await} } => return Err(ApiError::backend("Responses stream reader disconnected", None)),
            chunk = body.next() => chunk,
        };
        let Some(Ok(chunk)) = chunk else {
            return Err(ApiError::backend("chat stream closed unexpectedly", None));
        };
        // This is the in-process chat body: its sender emits one complete SSE
        // frame per Bytes, without network rechunking or unbounded buffering.
        let data = std::str::from_utf8(&chunk)
            .ok()
            .and_then(|text| text.strip_prefix("data: "))
            .map(str::trim)
            .ok_or_else(|| ApiError::backend("invalid internal chat frame", None))?;
        if data == "[DONE]" {
            let completed = events.complete()?;
            send(sender, vec![completed], timeout).await?;
            return Ok(events.response.clone());
        }
        let value: Value = serde_json::from_str(data)
            .map_err(|_| ApiError::backend("invalid internal chat JSON", None))?;
        if value["error"].is_object() {
            let message = value["error"]["message"]
                .as_str()
                .unwrap_or("generation failed");
            return Err(match value["error"]["code"].as_u64() {
                Some(429) => ApiError::capacity(message),
                Some(503) => ApiError::unavailable(message),
                Some(400) => ApiError::invalid(message, value["error"]["param"].as_str()),
                _ => ApiError::backend(message, value["error"]["code"].as_str().map(str::to_owned)),
            });
        }
        let deltas = events.chunk(value, reasoning_field)?;
        send(sender, deltas, timeout).await?;
    }
}

pub(crate) async fn retrieve(
    State(state): State<Arc<ServerState>>,
    Path(id): Path<String>,
    uri: axum::http::Uri,
) -> Response {
    let request_id = crate::response::next_request_id();
    let result = state.responses.get(&id).and_then(|response| {
        if uri
            .query()
            .is_some_and(|query| !query.is_empty() && query != "stream=false")
        {
            Err(request::invalid(
                "stored event replay is not supported",
                "stream",
            ))
        } else {
            Ok(response)
        }
    });
    let response = match result {
        Ok(value) => json_response(value, &request_id),
        Err(error) => error.into_response(&request_id),
    };
    state.admission_metrics.stored_response(response.status());
    response
}
pub(crate) async fn cancel(
    State(state): State<Arc<ServerState>>,
    Path(id): Path<String>,
) -> Response {
    let request_id = crate::response::next_request_id();
    let response = match state.responses.cancel(&id) {
        Ok(value) => json_response(value, &request_id),
        Err(error) => error.into_response(&request_id),
    };
    state.admission_metrics.stored_response(response.status());
    response
}
fn json_response(value: Value, id: &str) -> Response {
    let mut response = (StatusCode::OK, Json(value)).into_response();
    insert_request_id(&mut response, id);
    response
}
