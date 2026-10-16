use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::body::{Body, Bytes, to_bytes};
use axum::extract::State;
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use axum::http::{HeaderValue, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use tokio::sync::{OwnedSemaphorePermit, mpsc};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::ServerState;
use crate::backend::{
    AdmissionError, BackendReadiness, BackendToolCall, CancellationSignal, ChatMetadata,
    EventSender, FinishReason, GeneratedTokenLogprob, GenerationEvent, PromptTokenLogprob,
    ReasoningContent, RequestedLogprobs, TokenUsage, VisibleContent,
};
use crate::error::{ApiError, insert_request_id};
use crate::frontend::{FrontendErrorV1, ServerFrontendV1};
use crate::request::{self, Endpoint};

static REQUEST_SEQUENCE: AtomicU64 = AtomicU64::new(1);
const MODEL_CREATED: u64 = 1_785_483_024;

struct ActiveRequest {
    _permit: Arc<OwnedSemaphorePermit>,
    cancellation: CancellationSignal,
    drain_expired: CancellationSignal,
    completed: bool,
}

impl ActiveRequest {
    fn new(
        permit: OwnedSemaphorePermit,
        cancellation: CancellationSignal,
        drain_expired: CancellationSignal,
    ) -> Self {
        Self {
            _permit: Arc::new(permit),
            cancellation,
            drain_expired,
            completed: false,
        }
    }

    async fn next_event(
        &self,
        events: &mut mpsc::Receiver<GenerationEvent>,
    ) -> Option<GenerationEvent> {
        tokio::select! {
            biased;
            () = self.drain_expired.cancelled() => {
                self.cancellation.cancel();
                Some(GenerationEvent::Failed {
                    message: "server drain deadline expired".to_owned(),
                    code: Some("shutdown_timeout".to_owned()),
                })
            }
            event = events.recv() => event,
        }
    }

    fn complete(&mut self) {
        self.completed = true;
    }
}

impl Drop for ActiveRequest {
    fn drop(&mut self) {
        if !self.completed {
            self.cancellation.cancel();
        }
    }
}

/// Cancels a request whose handler is dropped before it reaches the engine.
pub(crate) struct CancelOnDrop(pub(crate) Option<CancellationSignal>);

impl CancelOnDrop {
    pub(crate) fn disarm(mut self) {
        self.0 = None;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(cancellation) = self.0.take() {
            cancellation.cancel();
        }
    }
}

#[derive(Clone)]
struct ResponseMetadata {
    /// The served model's frontend, carried so a streaming task can still ask
    /// it about a tool call after the request handler has returned.
    frontend: Arc<dyn ServerFrontendV1>,
    choice_index: usize,
    request_id: String,
    response_id: String,
    created: u64,
    model: String,
    endpoint: Endpoint,
    logprobs: RequestedLogprobs,
    include_usage: bool,
    responses: bool,
    parallel_tool_calls: bool,
    continuous_usage_stats: bool,
    prompt_tokens: u64,
    completion_tokens: u64,
    chat: Option<ChatMetadata>,
    max_new_tokens: usize,
    max_output_bytes: usize,
    stream_write_timeout: std::time::Duration,
}

/// The sending half of one event stream body.
///
/// A client that stops reading fills the transport, and a send then waits.
/// Each send waits at most the write timeout, and no longer than the drain
/// deadline, so a stalled reader cannot hold its admission capacity: the
/// stream task ends, which cancels the request and releases it.
struct SseSender {
    frames: mpsc::Sender<Result<Bytes, Infallible>>,
    write_timeout: std::time::Duration,
    drain_expired: CancellationSignal,
}

impl SseSender {
    async fn frame(&self, frame: Bytes) -> bool {
        let send = tokio::time::timeout(self.write_timeout, self.frames.send(Ok(frame)));
        if self.drain_expired.is_cancelled() {
            // After the deadline a terminal frame is still worth one bounded try.
            return matches!(send.await, Ok(Ok(())));
        }
        tokio::select! {
            biased;
            sent = send => matches!(sent, Ok(Ok(()))),
            () = self.drain_expired.cancelled() => false,
        }
    }

    async fn closed(&self) {
        self.frames.closed().await;
    }
}

struct OutputBudget {
    used: usize,
    maximum: usize,
}

struct GeneratedTokenBudget {
    observed: usize,
    maximum: usize,
}

struct SemanticEventBudget {
    observed: usize,
    maximum: usize,
}

impl SemanticEventBudget {
    fn new(maximum_tokens: usize) -> Result<Self, ApiError> {
        let maximum = maximum_tokens
            .checked_mul(3)
            .ok_or_else(|| invalid_backend_event("backend semantic-event limit overflowed"))?;
        Ok(Self {
            observed: 0,
            maximum,
        })
    }

    fn observe(&mut self) -> Result<(), ApiError> {
        self.observed = self
            .observed
            .checked_add(1)
            .ok_or_else(|| invalid_backend_event("backend semantic-event count overflowed"))?;
        if self.observed > self.maximum {
            return Err(invalid_backend_event(
                "backend data-bearing semantic events exceed three per generated token",
            ));
        }
        Ok(())
    }
}

impl GeneratedTokenBudget {
    const fn new(maximum: usize) -> Self {
        Self {
            observed: 0,
            maximum,
        }
    }

    fn observe(&mut self, count: usize) -> Result<(), ApiError> {
        let observed = self
            .observed
            .checked_add(count)
            .ok_or_else(|| invalid_backend_event("backend generated-token count overflowed"))?;
        if observed > self.maximum {
            return Err(invalid_backend_event(
                "backend generated-token count exceeds max_new_tokens",
            ));
        }
        self.observed = observed;
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct ValidatedUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    cached_tokens: Option<u64>,
}

impl OutputBudget {
    const fn new(maximum: usize) -> Self {
        Self { used: 0, maximum }
    }

    fn charge(&mut self, bytes: usize, context: &str) -> Result<(), ApiError> {
        let used = self
            .used
            .checked_add(bytes)
            .ok_or_else(|| invalid_backend_event("backend output size overflowed"))?;
        if used > self.maximum {
            return Err(invalid_backend_event(&format!(
                "backend {context} exceeds the configured output byte budget"
            )));
        }
        self.used = used;
        Ok(())
    }
}

/// One response's backend events, checked as the transport checks them:
/// the output, generated-token, logprob and semantic-event budgets, each
/// event's own shape, streamed tool calls against the final ones, and the
/// finish reason and usage those events allow.
///
/// Both response paths check every event here and keep only what they
/// present; [`ChatEventFoldV1`] checks a frontend vector's events here too.
struct EventFold {
    frontend: Arc<dyn ServerFrontendV1>,
    endpoint: Endpoint,
    chat: Option<ChatMetadata>,
    logprobs: RequestedLogprobs,
    max_new_tokens: usize,
    max_output_bytes: usize,
    output_budget: OutputBudget,
    generated_tokens: GeneratedTokenBudget,
    logprob_tokens: GeneratedTokenBudget,
    semantic_events: SemanticEventBudget,
    /// How many final tool calls the response has delivered.
    tool_count: usize,
    /// The calls streamed so far, which the final calls must repeat.
    streamed_tools: Vec<BackendToolCall>,
}

/// Where one final tool-call group falls among the response's calls.
struct ToolCallGroupV1 {
    /// The response's call count after this group.
    next: usize,
    /// Whether the group repeats calls that were already streamed.
    streamed: bool,
}

impl EventFold {
    fn new(
        frontend: Arc<dyn ServerFrontendV1>,
        endpoint: Endpoint,
        chat: Option<ChatMetadata>,
        logprobs: RequestedLogprobs,
        max_new_tokens: usize,
        max_output_bytes: usize,
    ) -> Result<Self, ApiError> {
        Ok(Self {
            frontend,
            endpoint,
            chat,
            logprobs,
            max_new_tokens,
            max_output_bytes,
            output_budget: OutputBudget::new(max_output_bytes),
            generated_tokens: GeneratedTokenBudget::new(max_new_tokens),
            logprob_tokens: GeneratedTokenBudget::new(max_new_tokens),
            semantic_events: SemanticEventBudget::new(max_new_tokens)?,
            tool_count: 0,
            streamed_tools: Vec::new(),
        })
    }

    fn for_response(metadata: &ResponseMetadata) -> Result<Self, ApiError> {
        Self::new(
            Arc::clone(&metadata.frontend),
            metadata.endpoint,
            metadata.chat.clone(),
            metadata.logprobs,
            metadata.max_new_tokens,
            metadata.max_output_bytes,
        )
    }

    fn token_logprobs(&mut self, records: &[GeneratedTokenLogprob]) -> Result<(), ApiError> {
        validate_token_logprobs(
            records,
            self.logprobs,
            &mut self.logprob_tokens,
            &mut self.output_budget,
        )
    }

    fn content(&mut self, content: &VisibleContent) -> Result<(), ApiError> {
        self.semantic_events.observe()?;
        self.generated_tokens
            .observe(content.generated_token_count)?;
        validate_content_event(content, &mut self.output_budget)
    }

    fn reasoning(&mut self, reasoning: &ReasoningContent) -> Result<(), ApiError> {
        self.semantic_events.observe()?;
        self.generated_tokens
            .observe(reasoning.generated_token_count)?;
        validate_reasoning_event(self.chat.as_ref(), reasoning, &mut self.output_budget)
    }

    fn tool_deltas(&mut self, deltas: &[crate::ToolCallDelta]) -> Result<(), ApiError> {
        validate_tool_deltas(
            self.frontend.as_ref(),
            self.chat.as_ref(),
            deltas,
            &mut self.streamed_tools,
            &mut self.output_budget,
        )
    }

    /// One final tool-call group. Calls that were streamed were charged to
    /// the output budget as they streamed, so the group is charged to a
    /// budget of its own, and must repeat the streamed calls at its indices.
    fn tool_calls(
        &mut self,
        calls: &[BackendToolCall],
        generated_token_count: usize,
    ) -> Result<ToolCallGroupV1, ApiError> {
        self.semantic_events.observe()?;
        self.generated_tokens.observe(generated_token_count)?;
        let streamed = !self.streamed_tools.is_empty();
        let mut final_budget = OutputBudget::new(self.max_output_bytes);
        let first = self.tool_count;
        let next = validate_tool_calls_event(
            self.frontend.as_ref(),
            self.chat.as_ref(),
            calls,
            first,
            if streamed {
                &mut final_budget
            } else {
                &mut self.output_budget
            },
        )?;
        // Calls separated by text arrive as one group each, so a group
        // matches the streamed calls at its own indices.
        if streamed && self.streamed_tools.get(first..next) != Some(calls) {
            return Err(invalid_backend_event(
                "streamed tool arguments differ from final calls",
            ));
        }
        self.tool_count = next;
        Ok(ToolCallGroupV1 { next, streamed })
    }

    fn token_usage(&mut self, generated_token_count: usize) -> Result<(), ApiError> {
        if generated_token_count == 0 {
            return Err(invalid_backend_event(
                "backend token-only accounting delta must be positive",
            ));
        }
        self.generated_tokens.observe(generated_token_count)
    }

    fn failure(&mut self, message: &str, code: Option<&str>) -> Result<(), ApiError> {
        validate_failure_event(message, code, &mut self.output_budget)
    }

    /// The finish reason and usage the response reports, which the calls
    /// delivered or streamed and the tokens observed must allow.
    fn finish(
        &self,
        reason: FinishReason,
        usage: TokenUsage,
    ) -> Result<(FinishReason, ValidatedUsage), ApiError> {
        let reason = finish_reason_for(
            self.endpoint,
            self.chat.as_ref(),
            reason,
            self.tool_count.max(self.streamed_tools.len()),
        );
        let usage = validate_usage(self.max_new_tokens, usage, self.generated_tokens.observed)
            .and_then(|usage| {
                validate_logprob_count(self.logprobs, self.logprob_tokens.observed, usage)
            });
        reason.and_then(|reason| usage.map(|usage| (reason, usage)))
    }
}

/// One chat response's backend events, checked and folded exactly as the
/// transport checks them, without a listener.
///
/// Each event goes through the checks a served response applies to it, and
/// the reasoning, content and final tool calls are kept as a non-streamed
/// response reports them. Built `streaming`, the fold also checks streamed
/// tool-call deltas and requires the final calls to repeat them, as a
/// streamed response does; otherwise deltas are ignored, as a non-streamed
/// response ignores them. A frontend check folds a vector's decoded events
/// this way.
pub struct ChatEventFoldV1 {
    fold: EventFold,
    streaming: bool,
    reasoning: String,
    content: String,
    tool_calls: Vec<BackendToolCall>,
}

impl ChatEventFoldV1 {
    /// A fold for one prepared chat request, under the transport's output
    /// byte budget `max_output_bytes`.
    pub fn new(
        frontend: Arc<dyn ServerFrontendV1>,
        request: &crate::PreparedRequest,
        max_output_bytes: usize,
        streaming: bool,
    ) -> Result<Self, FrontendErrorV1> {
        let fold = EventFold::new(
            frontend,
            Endpoint::Chat,
            request.chat.clone(),
            request.logprobs,
            request.max_new_tokens,
            max_output_bytes,
        )
        .map_err(rejected_event)?;
        Ok(Self {
            fold,
            streaming,
            reasoning: String::new(),
            content: String::new(),
            tool_calls: Vec::new(),
        })
    }

    /// Checks and folds one event; a terminal event returns the finish
    /// reason the response reports. Events that carry no output (prompt
    /// metadata, token ids) are not checked here.
    pub fn event(
        &mut self,
        event: &GenerationEvent,
    ) -> Result<Option<FinishReason>, FrontendErrorV1> {
        let fold = &mut self.fold;
        match event {
            GenerationEvent::GeneratedTokenLogprobs(records) => fold.token_logprobs(records),
            GenerationEvent::Content(content) => fold.content(content).map(|()| {
                self.content.push_str(&content.delta);
            }),
            GenerationEvent::Reasoning(reasoning) => fold.reasoning(reasoning).map(|()| {
                self.reasoning.push_str(&reasoning.delta);
            }),
            GenerationEvent::ToolCallDeltas(deltas) if self.streaming => fold.tool_deltas(deltas),
            GenerationEvent::ToolCalls {
                calls,
                generated_token_count,
            } => fold.tool_calls(calls, *generated_token_count).map(|_| {
                self.tool_calls.extend(calls.iter().cloned());
            }),
            GenerationEvent::TokenUsageDelta {
                generated_token_count,
            } => fold.token_usage(*generated_token_count),
            GenerationEvent::Finished { reason, usage } => {
                return fold
                    .finish(*reason, *usage)
                    .map(|(reason, _)| Some(reason))
                    .map_err(rejected_event);
            }
            GenerationEvent::Failed { message, code } => {
                fold.failure(message, code.as_deref())
                    .map_err(rejected_event)?;
                return Err(FrontendErrorV1::new(message.clone(), None));
            }
            GenerationEvent::AdmissionFailed(error) => Err(admission_error(error.clone())),
            _ => Ok(()),
        }
        .map(|()| None)
        .map_err(rejected_event)
    }

    /// The reasoning delivered so far.
    #[must_use]
    pub fn reasoning(&self) -> &str {
        &self.reasoning
    }

    /// The visible content delivered so far.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// The final tool calls delivered so far, in order.
    #[must_use]
    pub fn tool_calls(&self) -> &[BackendToolCall] {
        &self.tool_calls
    }
}

fn rejected_event(error: ApiError) -> FrontendErrorV1 {
    FrontendErrorV1 {
        message: error.message,
        parameter: error.parameter,
    }
}

pub(crate) async fn health(State(state): State<Arc<ServerState>>) -> Response {
    let request_id = next_request_id();
    let (status, mut body) = match state.backend.readiness() {
        BackendReadiness::Ready => (StatusCode::OK, json!({"status": "ready"})),
        BackendReadiness::NotReady(reason) | BackendReadiness::Terminated(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"status": "not_ready", "reason": reason}),
        ),
    };
    body["transport"] = state.transport.value();
    body["waiting"] = state.waiting.value();
    body["queue_depth"] = body["waiting"]["total"].clone();
    body["modalities"] = json!(crate::frontend::served_modalities(&*state.frontend));
    if let Some(serving) = &state.serving {
        body["serving"] = serving.clone();
    }
    json_response(status, body, &request_id)
}

pub(crate) async fn version() -> Json<Value> {
    Json(json!({"version": env!("CARGO_PKG_VERSION")}))
}

pub(crate) async fn models(State(state): State<Arc<ServerState>>) -> Response {
    let request_id = next_request_id();
    let modalities = crate::frontend::served_modalities(&*state.frontend);
    let images = state.frontend.maximum_images_per_request();
    let data = state
        .frontend
        .models()
        .iter()
        .map(|model| {
            let mut entry = json!({
                "id": model.id,
                "object": "model",
                "created": MODEL_CREATED,
                "owned_by": model.owned_by,
                "root": model.id,
                "parent": Value::Null,
                "max_model_len": state.maximum_context_tokens,
                "modalities": modalities
            });
            // Clients that recover from image limits read these, in the
            // shape other OpenAI-compatible model lists use.
            if let Some(images) = images {
                entry["architecture"] = json!({"input_modalities": modalities});
                entry["max_images_per_request"] = json!(images);
            }
            if let Some(serving) = &state.serving
                && let Some(options) = serving.get("model_options")
            {
                entry["model_options"] = options.clone();
                entry["model_options_qualified"] = serving["model_options_qualified"].clone();
            }
            entry
        })
        .collect::<Vec<_>>();
    json_response(
        StatusCode::OK,
        json!({"object": "list", "data": data}),
        &request_id,
    )
}

pub(crate) async fn completions(
    State(state): State<Arc<ServerState>>,
    request: Request<Body>,
) -> Response {
    generate(state, request, Endpoint::Completion, false).await
}

pub(crate) async fn chat_completions(
    State(state): State<Arc<ServerState>>,
    request: Request<Body>,
) -> Response {
    generate(state, request, Endpoint::Chat, false).await
}

pub(crate) async fn responses(
    State(state): State<Arc<ServerState>>,
    request: Request<Body>,
) -> Response {
    generate(state, request, Endpoint::Chat, true).await
}

async fn generate(
    state: Arc<ServerState>,
    request: Request<Body>,
    endpoint: Endpoint,
    responses: bool,
) -> Response {
    let response = generate_inner(Arc::clone(&state), request, endpoint, responses).await;
    state.admission_metrics.response(response.status());
    response
}

async fn await_admission<T>(
    state: &ServerState,
    deadline: Option<tokio::time::Instant>,
    future: impl std::future::Future<Output = T>,
) -> Result<T, ApiError> {
    tokio::select! {
        biased;
        () = state.drain_expired.cancelled() => Err(ApiError::backend("server drain deadline expired", Some("shutdown_timeout".into()))),
        () = async { match deadline {
            Some(deadline) => tokio::time::sleep_until(deadline).await,
            None => std::future::pending().await,
        }} => Err(ApiError::capacity("server queue wait deadline expired")),
        result = future => Ok(result),
    }
}

async fn generate_inner(
    state: Arc<ServerState>,
    request: Request<Body>,
    endpoint: Endpoint,
    responses: bool,
) -> Response {
    let mut waiting = state.waiting.enter(crate::waiting::Stage::Intake);
    let deadline = state
        .queue_wait
        .map(|wait| tokio::time::Instant::now() + wait);
    let peer = request.extensions().get::<std::net::SocketAddr>().copied();
    let request_id = next_request_id();
    let client_request_id = match request::validate_headers(request.headers()) {
        Ok(value) => value.or_else(|| {
            request
                .extensions()
                .get::<crate::http_options::HttpRequestId>()
                .map(|id| id.0.clone())
        }),
        Err(error) => return error.into_response(&request_id),
    };
    if let Err(error) = request::validate_content_type(request.headers()) {
        return error.into_response(&request_id);
    }
    // A request that could not be admitted now is refused before its body is
    // read or its prompt prepared. The acquisition below still decides.
    if !state.wait_for_capacity && state.in_flight.available_permits() == 0 {
        return capacity_error(&state).into_response(&request_id);
    }
    // Chat may carry media, so it reads up to the media limit. Only stored
    // Responses history expands a body.
    let limit = if endpoint == Endpoint::Chat && !responses {
        state.max_media_body_bytes
    } else {
        state.max_body_bytes
    };
    let reserve = crate::intake::reservation(request.headers(), limit, responses);
    let (body, intake) = match await_admission(
        &state,
        deadline,
        state
            .intake
            .read(request.into_body(), limit, state.wait_for_capacity, reserve),
    )
    .await
    .and_then(|result| result)
    {
        Ok(read) => read,
        Err(error) => return error.into_response(&request_id),
    };
    waiting.move_to(crate::waiting::Stage::Preparation);
    let cancellation = CancellationSignal::new();
    let identity = request::RequestIdentityV1 {
        request_id: request_id.clone(),
        client_request_id,
        generated_seed: generated_seed(),
        cancellation: cancellation.clone(),
    };
    // Parsing, chat rendering and tokenization are synchronous and can take
    // long for a large body, so they run off the async workers. A client that
    // leaves meanwhile drops this handler, and the guard tells the blocking
    // preparation to stop waiting on media work for it.
    let preparing = Arc::clone(&state);
    let leaving = CancelOnDrop(Some(cancellation.clone()));
    let prepared = tokio::task::spawn_blocking(move || {
        let mut intake = intake;
        let prepared = (|| {
            let (body, context) = if responses {
                let (body, context) = crate::responses::Context::prepare(
                    &preparing,
                    &body,
                    &identity.request_id,
                    &mut intake,
                )?;
                (Bytes::from(body), Some(context))
            } else {
                (body, None)
            };
            prepare_request(&preparing, endpoint, &body, identity)
                .map(|prepared| (prepared, context))
        })();
        (prepared, intake)
    });
    let prepared = match await_admission(&state, deadline, prepared).await {
        Ok(result) => result,
        Err(error) => return error.into_response(&request_id),
    };
    leaving.disarm();
    let (prepared, intake) = match prepared {
        Ok(prepared) => prepared,
        Err(_) => {
            return ApiError::backend("request preparation failed", None)
                .into_response(&request_id);
        }
    };
    let ((mut parsed, count, prepared_prompts, prompt_tokens), mut responses_context) =
        match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                error.log_model_refusal(peer, state.frontend.models());
                return error.into_response(&request_id);
            }
        };
    waiting.expand(count);
    waiting.move_to(crate::waiting::Stage::Staging);
    if count > state.max_in_flight {
        return ApiError::invalid(
            "prompt batch times n exceeds the server request limit",
            Some("n"),
        )
        .into_response(&request_id);
    }
    let acquired = if state.wait_for_capacity {
        match await_admission(
            &state,
            deadline,
            Arc::clone(&state.in_flight).acquire_many_owned(count as u32),
        )
        .await
        {
            Ok(result) => result.map_err(|_| ()),
            Err(error) => return error.into_response(&request_id),
        }
    } else {
        Arc::clone(&state.in_flight)
            .try_acquire_many_owned(count as u32)
            .map_err(|_| ())
    };
    let permit = match acquired {
        Ok(permit) => permit,
        Err(_) if state.in_flight.is_closed() => {
            return crate::lifecycle::draining_response(&request_id);
        }
        Err(_) => return capacity_error(&state).into_response(&request_id),
    };
    let mut active = ActiveRequest::new(permit, cancellation, state.drain_expired.clone());
    let metadata = ResponseMetadata {
        frontend: Arc::clone(&state.frontend),
        choice_index: 0,
        response_id: response_id(endpoint, &request_id),
        request_id: request_id.clone(),
        created: unix_time(),
        model: parsed.prepared.model.clone(),
        endpoint,
        logprobs: parsed.prepared.logprobs,
        include_usage: responses || parsed.include_usage,
        responses,
        parallel_tool_calls: parsed.parallel_tool_calls,
        continuous_usage_stats: parsed.continuous_usage_stats,
        prompt_tokens: prompt_tokens as u64,
        completion_tokens: 0,
        chat: parsed.prepared.chat.clone(),
        max_new_tokens: parsed.prepared.max_new_tokens,
        max_output_bytes: state.max_output_bytes,
        stream_write_timeout: state.stream_write_timeout,
    };
    if let Some(context) = &mut responses_context {
        waiting.move_to(crate::waiting::Stage::Storage);
        context.effective(&parsed.prepared, metadata.created);
        if let Err(error) = await_admission(
            &state,
            deadline,
            context.reserve(Arc::clone(&state), active.cancellation.clone()),
        )
        .await
        .and_then(|result| result)
        {
            return error.into_response(&request_id);
        }
    }
    drop(intake);
    let response_cancellation = active.cancellation.clone();
    let mut choices = Vec::with_capacity(count);
    let mut pending_submissions = Vec::with_capacity(count);
    for index in 0..count {
        let (template, tokens) = &prepared_prompts[index / parsed.choices];
        let mut prepared = template.clone();
        prepared.seed = if parsed.seeded {
            prepared.seed.wrapping_add((index % parsed.choices) as u64)
        } else {
            generated_seed()
        };
        prepared.request_id = format!("{request_id}_{index}");
        let (event_tx, event_rx) = mpsc::channel(state.event_buffer_capacity);
        let sender = EventSender::new(event_tx);
        pending_submissions.push((prepared, sender));
        let mut choice_metadata = metadata.clone();
        choice_metadata.choice_index = index;
        choice_metadata.prompt_tokens = *tokens as u64;
        choice_metadata.max_new_tokens = template.max_new_tokens;
        let choice_active = ActiveRequest {
            _permit: Arc::clone(&active._permit),
            cancellation: active.cancellation.clone(),
            drain_expired: active.drain_expired.clone(),
            completed: false,
        };
        choices.push((choice_metadata, event_rx, choice_active));
    }
    // The submissions own the prepared prompts now. Prepared media payloads
    // hold host memory the backend releases as prefill passes each item, so
    // the handler keeps no copy while it waits for the response.
    drop(prepared_prompts);
    parsed.prepared.prompt = crate::PreparedPrompt::TokenIds(Vec::new());
    let (initial_response, initial_admitted) = tokio::sync::oneshot::channel();
    let submission_state = Arc::clone(&state);
    let cancellation = active.cancellation.clone();
    let waiting = waiting.into_backend();
    tokio::spawn(async move {
        let senders: Vec<_> = pending_submissions
            .iter()
            .map(|(_, events)| events.clone())
            .collect();
        let admissions = submission_state.backend.submit_batch(pending_submissions);
        // Any admitted choice must be drained, even if an earlier index
        // is still queued on another replica or waiting for scheduler budget.
        let initial_response = std::sync::Mutex::new(Some(initial_response));
        let mut pending: Vec<_> = admissions
            .into_iter()
            .zip(senders)
            .zip(waiting)
            .map(|((admission, events), waiting)| {
                let initial_response = &initial_response;
                Some(Box::pin(async move {
                    let result = admission.await;
                    drop(waiting);
                    let initial = initial_response
                        .lock()
                        .expect("admission response lock")
                        .take();
                    if let Some(initial) = initial {
                        let _ = initial.send(result);
                    } else if let Err(error) = result {
                        let _ = events.emit(GenerationEvent::AdmissionFailed(error)).await;
                    }
                }))
            })
            .collect();
        tokio::select! {
            () = cancellation.cancelled() => {},
            () = std::future::poll_fn(|cx| {
                let mut complete = true;
                for future in &mut pending {
                    if let Some(future) = future
                        && future.as_mut().poll(cx).is_pending()
                    {
                        complete = false;
                        continue;
                    }
                    *future = None;
                }
                if complete { std::task::Poll::Ready(()) } else { std::task::Poll::Pending }
            }) => {},
        }
    });
    let result = match await_admission(&state, deadline, initial_admitted).await {
        Ok(result) => result.unwrap_or(Err(AdmissionError::WorkerUnavailable)),
        Err(error) => return error.into_response(&request_id),
    };
    if let Err(error) = result {
        return admission_error(error).into_response(&request_id);
    }
    active.complete();
    if count == 1 {
        let (metadata, events, active) = choices.pop().expect("one choice");
        if let Some(context) = responses_context {
            let response = streaming_response(metadata, events, active);
            return crate::responses::adapt(
                state,
                response,
                context,
                response_cancellation,
                request_id,
            )
            .await;
        }
        return if parsed.stream {
            streaming_response(metadata, events, active)
        } else {
            nonstreaming_response(metadata, events, active).await
        };
    }
    if parsed.stream {
        multiple_streaming_response(metadata, choices, parsed.choices)
    } else {
        multiple_nonstreaming_response(metadata, choices, parsed.choices).await
    }
}

type PreparedRequestParts = (
    request::ParsedRequest,
    usize,
    Vec<(crate::PreparedRequest, usize)>,
    usize,
);

/// Parses one body and prepares each of its prompts.
fn prepare_request(
    state: &ServerState,
    endpoint: Endpoint,
    body: &[u8],
    identity: request::RequestIdentityV1,
) -> Result<PreparedRequestParts, ApiError> {
    let mut parsed = request::parse(
        state.frontend.as_ref(),
        state.sampling_overrides,
        endpoint,
        state.acceptance_profile,
        state.request_surface,
        state.max_string_bytes,
        body,
        identity,
    )?;
    let count = parsed
        .choices
        .saturating_mul(parsed.additional_prompts.len() + 1);
    if count > 128 {
        return Err(ApiError::invalid(
            "prompt count times n must not exceed 128",
            Some("n"),
        ));
    }
    let mut prompts = vec![parsed.prepared.prompt.clone()];
    prompts.append(&mut parsed.additional_prompts);
    let mut prepared_prompts = Vec::with_capacity(prompts.len());
    let mut prompt_tokens = 0;
    for prompt in prompts {
        parsed.prepared.prompt = prompt;
        let tokens = prepare_prompt(state, &mut parsed)?;
        prompt_tokens += tokens;
        prepared_prompts.push((parsed.prepared.clone(), tokens));
    }
    Ok((parsed, count, prepared_prompts, prompt_tokens))
}

fn capacity_error(state: &ServerState) -> ApiError {
    let message = state.acceptance_profile.profile_name().map_or_else(
        || "server request capacity is exhausted".to_owned(),
        |profile| format!("server profile {profile} request capacity is exhausted"),
    );
    ApiError::capacity(message)
}

fn prepare_prompt(
    state: &ServerState,
    parsed: &mut request::ParsedRequest,
) -> Result<usize, ApiError> {
    let truncation_limit = parsed.truncation.limit(
        state.maximum_context_tokens,
        if parsed.use_remaining_context {
            0
        } else {
            parsed.prepared.max_new_tokens
        },
    )?;
    let prompt = std::mem::replace(
        &mut parsed.prepared.prompt,
        crate::PreparedPrompt::TokenIds(Vec::new()),
    );
    let prompt_tokens = match state
        .backend
        .prepare_prompt_cancellable(prompt, &parsed.prepared.cancellation)
    {
        Ok((prepared, prompt_tokens)) => {
            parsed.prepared.prompt = prepared;
            prompt_tokens
        }
        Err(error) => return Err(admission_error(error)),
    };
    let prompt_tokens =
        parsed
            .truncation
            .apply(&mut parsed.prepared.prompt, prompt_tokens, truncation_limit)?;
    // Which ids a prompt became is the tokenizer's answer rather than the
    // request's, so a profile that admits one exact prompt length is enforced
    // here, on the ids the engine would run.
    match state.acceptance_profile {
        crate::RequestAcceptanceProfile::OneToken {
            prompt_tokens: expected,
        } if prompt_tokens != expected => {
            return Err(ApiError::invalid(
                format!("prompt must encode to exactly {expected} tokens for one-token acceptance"),
                Some("prompt"),
            ));
        }
        crate::RequestAcceptanceProfile::ExactShape {
            prompt_tokens: expected,
            ..
        } if prompt_tokens != expected => {
            return Err(ApiError::invalid(
                format!(
                    "prompt must encode to exactly {expected} tokens for exact-shape acceptance, \
                     and this one encodes to {prompt_tokens}"
                ),
                Some("prompt"),
            ));
        }
        crate::RequestAcceptanceProfile::Bounded {
            maximum_prompt_tokens,
            ..
        } if prompt_tokens == 0 || prompt_tokens > maximum_prompt_tokens => {
            return Err(ApiError::invalid(
                format!(
                    "prompt must encode to between 1 and {maximum_prompt_tokens} tokens, and this one encodes to {prompt_tokens}"
                ),
                Some("prompt"),
            ));
        }
        _ => {}
    }
    if parsed.use_remaining_context {
        parsed.prepared.max_new_tokens = state.maximum_context_tokens.saturating_sub(prompt_tokens);
        if parsed.prepared.max_new_tokens == 0 {
            return Err(ApiError::context_length(
                "prompt leaves no room for generated tokens",
            ));
        }
        if parsed.prepared.sampling.minimum_new_tokens() > parsed.prepared.max_new_tokens {
            return Err(ApiError::invalid(
                "min_tokens exceeds remaining context",
                Some("min_tokens"),
            ));
        }
    }
    if let Err(error) = crate::context::validate_total_tokens(
        prompt_tokens,
        parsed.prepared.max_new_tokens,
        state.maximum_context_tokens,
    ) {
        return Err(ApiError::context_length(error.to_string()));
    }
    Ok(prompt_tokens)
}

type ChoiceStream = (
    ResponseMetadata,
    mpsc::Receiver<GenerationEvent>,
    ActiveRequest,
);

/// Cached prompt tokens are counted as prompt tokens are, once per prompt:
/// each prompt's first choice reports them.
fn counts_cached_tokens(index: usize, choices_per_prompt: usize) -> bool {
    index % choices_per_prompt.max(1) == 0
}

fn cached_tokens(usage: &Value) -> u64 {
    usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0)
}

async fn multiple_nonstreaming_response(
    metadata: ResponseMetadata,
    choices: Vec<ChoiceStream>,
    choices_per_prompt: usize,
) -> Response {
    let mut tasks = tokio::task::JoinSet::new();
    for (metadata, events, active) in choices {
        tasks.spawn(nonstreaming_response(metadata, events, active));
    }
    let mut result = Value::Null;
    let mut outputs = Vec::new();
    let mut completion_tokens = 0;
    let mut cached = 0;
    while let Some(response) = tasks.join_next().await {
        let Ok(response) = response else {
            return invalid_backend_event("choice task failed").into_response(&metadata.request_id);
        };
        if !response.status().is_success() {
            return response;
        }
        let Ok(body) = to_bytes(response.into_body(), usize::MAX).await else {
            return invalid_backend_event("choice response failed")
                .into_response(&metadata.request_id);
        };
        let Ok(mut value) = serde_json::from_slice::<Value>(&body) else {
            return invalid_backend_event("invalid choice response")
                .into_response(&metadata.request_id);
        };
        completion_tokens += value["usage"]["completion_tokens"].as_u64().unwrap_or(0);
        let index = value["choices"][0]["index"].as_u64().unwrap_or(0) as usize;
        if counts_cached_tokens(index, choices_per_prompt) {
            cached += cached_tokens(&value["usage"]);
        }
        outputs.append(value["choices"].as_array_mut().expect("choice array"));
        result = value;
    }
    outputs.sort_by_key(|choice| choice["index"].as_u64());
    result["choices"] = json!(outputs);
    result["usage"] = usage_value(ValidatedUsage {
        prompt_tokens: metadata.prompt_tokens,
        completion_tokens,
        total_tokens: metadata.prompt_tokens + completion_tokens,
        cached_tokens: Some(cached),
    });
    json_response(StatusCode::OK, result, &metadata.request_id)
}

fn multiple_streaming_response(
    metadata: ResponseMetadata,
    choices: Vec<ChoiceStream>,
    choices_per_prompt: usize,
) -> Response {
    let (frames, body_rx) = mpsc::channel::<Result<Bytes, Infallible>>(1);
    let body_tx = SseSender {
        frames,
        write_timeout: metadata.stream_write_timeout,
        drain_expired: choices
            .first()
            .map_or_else(CancellationSignal::new, |(_, _, active)| {
                active.drain_expired.clone()
            }),
    };
    let request_id = metadata.request_id.clone();
    tokio::spawn(async move {
        let count = choices.len();
        let (tx, mut rx) = mpsc::channel(count);
        let mut tasks = tokio::task::JoinSet::new();
        for (mut choice_metadata, events, active) in choices {
            let tx = tx.clone();
            choice_metadata.include_usage = true;
            choice_metadata.continuous_usage_stats = true;
            let index = choice_metadata.choice_index;
            let response = streaming_response(choice_metadata, events, active);
            tasks.spawn(async move {
                let mut stream = response.into_body().into_data_stream();
                while let Some(Ok(frame)) = stream.next().await {
                    if tx.send((index, frame)).await.is_err() {
                        return;
                    }
                }
            });
        }
        drop(tx);
        let mut generated = vec![0_u64; count];
        let mut cached = 0;
        let mut finished = 0;
        loop {
            let item = tokio::select! {
                _ = body_tx.closed() => return,
                item = rx.recv() => item,
            };
            let Some((index, frame)) = item else {
                break;
            };
            if frame.as_ref() == b"data: [DONE]\n\n" {
                continue;
            }
            // Internal streams emit exactly one complete SSE frame per body item.
            let Some(data) = frame.strip_prefix(b"data: ") else {
                return;
            };
            let Ok(mut value) = serde_json::from_slice::<Value>(data) else {
                return;
            };
            if value.get("error").is_some() {
                let _ = send_sse(&body_tx, &value).await;
                let _ = send_done(&body_tx).await;
                return;
            }
            generated[index] = value["usage"]["completion_tokens"].as_u64().unwrap_or(0);
            if value["choices"].as_array().is_some_and(Vec::is_empty) {
                finished += 1;
                if counts_cached_tokens(index, choices_per_prompt) {
                    cached += cached_tokens(&value["usage"]);
                }
                continue;
            }
            if metadata.include_usage {
                value["usage"] = if metadata.continuous_usage_stats {
                    let completion_tokens = generated.iter().sum();
                    usage_value(ValidatedUsage {
                        prompt_tokens: metadata.prompt_tokens,
                        completion_tokens,
                        total_tokens: metadata.prompt_tokens + completion_tokens,
                        cached_tokens: None,
                    })
                } else {
                    Value::Null
                };
            } else {
                value
                    .as_object_mut()
                    .expect("stream object")
                    .remove("usage");
            }
            if !send_sse(&body_tx, &value).await {
                return;
            }
        }
        while tasks.join_next().await.is_some() {}
        if finished != count {
            let _ = send_sse(
                &body_tx,
                &invalid_backend_event("choice stream ended early").value(),
            )
            .await;
        } else if metadata.include_usage {
            let completion_tokens = generated.iter().sum();
            let _ = send_sse(
                &body_tx,
                &json!({
                    "id": metadata.response_id, "object": stream_object(metadata.endpoint),
                    "created": metadata.created, "model": metadata.model, "choices": [],
                    "usage": usage_value(ValidatedUsage { prompt_tokens: metadata.prompt_tokens,
                        completion_tokens, total_tokens: metadata.prompt_tokens + completion_tokens,
                        cached_tokens: Some(cached) }),
                }),
            )
            .await;
        }
        let _ = send_done(&body_tx).await;
    });
    let mut response = Response::new(Body::from_stream(ReceiverStream::new(body_rx)));
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response
        .headers_mut()
        .insert(CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    insert_request_id(&mut response, &request_id);
    response
}

pub(crate) fn admission_error(error: AdmissionError) -> ApiError {
    match error {
        AdmissionError::AtCapacity => ApiError::capacity("generation backend is at capacity"),
        AdmissionError::InvalidPrompt => {
            ApiError::invalid("prompt tokenization failed", Some("prompt"))
        }
        AdmissionError::InvalidRequest => ApiError::invalid("generation request is invalid", None),
        AdmissionError::WorkerUnavailable => {
            ApiError::unavailable("generation worker is unavailable")
        }
        AdmissionError::Unavailable(message) => ApiError::unavailable(message),
        AdmissionError::Rejected(message) => ApiError::invalid(message, None),
        AdmissionError::InvalidParameter { message, parameter } => {
            ApiError::invalid(message, Some(parameter))
        }
    }
}

async fn nonstreaming_response(
    metadata: ResponseMetadata,
    mut events: mpsc::Receiver<GenerationEvent>,
    mut active: ActiveRequest,
) -> Response {
    let mut token_ids = Vec::new();
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tokens = Vec::new();
    let mut tool_calls = Vec::new();
    let mut prompt: Option<(Vec<u32>, String, Vec<String>)> = None;
    let mut prompt_logprobs = Vec::new();
    let mut fold = match EventFold::for_response(&metadata) {
        Ok(fold) => fold,
        Err(error) => return error.into_response(&metadata.request_id),
    };
    while let Some(event) = active.next_event(&mut events).await {
        match event {
            GenerationEvent::AdmissionFailed(error) => {
                return admission_error(error).into_response(&metadata.request_id);
            }
            GenerationEvent::GeneratedTokenIds(ids) => {
                if !metadata.logprobs.return_token_ids
                    || ids.is_empty()
                    || token_ids.len().saturating_add(ids.len()) > metadata.max_new_tokens
                {
                    return invalid_backend_event("invalid generated token IDs")
                        .into_response(&metadata.request_id);
                }
                token_ids.extend(ids);
            }
            GenerationEvent::Prompt {
                token_ids,
                text,
                tokens,
            } => {
                if token_ids.len() != tokens.len()
                    || prompt.is_some()
                    || (metadata.logprobs.prompt_top.is_none()
                        && !metadata.logprobs.return_token_ids)
                {
                    return invalid_backend_event("invalid prompt metadata")
                        .into_response(&metadata.request_id);
                }
                prompt = Some((token_ids, text, tokens));
            }
            GenerationEvent::PromptLogprobs(rows) => {
                prompt_logprobs.extend(rows);
            }
            GenerationEvent::GeneratedTokenLogprobs(records) => {
                if let Err(error) = fold.token_logprobs(&records) {
                    return error.into_response(&metadata.request_id);
                }
                tokens.extend(records);
            }
            GenerationEvent::Content(content_event) => {
                if let Err(error) = fold.content(&content_event) {
                    return error.into_response(&metadata.request_id);
                }
                content.push_str(&content_event.delta);
            }
            GenerationEvent::Reasoning(reasoning_event) => {
                if let Err(error) = fold.reasoning(&reasoning_event) {
                    return error.into_response(&metadata.request_id);
                }
                reasoning.push_str(&reasoning_event.delta);
            }
            GenerationEvent::ToolCallDeltas(_) => {}
            GenerationEvent::ToolCalls {
                calls,
                generated_token_count,
            } => {
                if let Err(error) = fold.tool_calls(&calls, generated_token_count) {
                    return error.into_response(&metadata.request_id);
                }
                tool_calls.extend(calls.into_iter().map(|call| (call.name, call.arguments)));
            }
            GenerationEvent::TokenUsageDelta {
                generated_token_count,
            } => {
                if let Err(error) = fold.token_usage(generated_token_count) {
                    return error.into_response(&metadata.request_id);
                }
            }
            GenerationEvent::Finished { reason, usage } => {
                let finished = fold.finish(reason, usage);
                active.complete();
                let (reason, usage) = match finished {
                    Ok(result) => result,
                    Err(error) => return error.into_response(&metadata.request_id),
                };
                let mut value = match metadata.endpoint {
                    Endpoint::Completion => completion_value(
                        &metadata,
                        &content,
                        &tokens,
                        prompt.as_ref(),
                        &prompt_logprobs,
                        reason,
                        usage,
                    ),
                    Endpoint::Chat => chat_value(
                        &metadata,
                        ChatValueInput {
                            content: &content,
                            reasoning: &reasoning,
                            tokens: &tokens,
                            tool_calls: &tool_calls,
                            prompt: prompt.as_ref(),
                            prompt_logprobs: &prompt_logprobs,
                            reason,
                            usage,
                        },
                    ),
                };
                if metadata.logprobs.return_token_ids {
                    if token_ids.len() as u64 != usage.completion_tokens {
                        return invalid_backend_event("generated token IDs differ from usage")
                            .into_response(&metadata.request_id);
                    }
                    if expose_token_ids(&metadata) {
                        value["choices"][0]["token_ids"] = json!(token_ids);
                    }
                    if metadata.endpoint == Endpoint::Completion {
                        value["choices"][0]["prompt_token_ids"] = value["prompt_token_ids"].clone();
                    }
                }
                return json_response(StatusCode::OK, value, &metadata.request_id);
            }
            GenerationEvent::Failed { message, code } => {
                active.complete();
                if let Err(error) = fold.failure(&message, code.as_deref()) {
                    return error.into_response(&metadata.request_id);
                }
                return ApiError::backend(message, code).into_response(&metadata.request_id);
            }
        }
    }
    ApiError::backend(
        "generation backend closed without a terminal event",
        Some("incomplete_generation".to_owned()),
    )
    .into_response(&metadata.request_id)
}

fn completion_value(
    metadata: &ResponseMetadata,
    content: &str,
    tokens: &[GeneratedTokenLogprob],
    prompt: Option<&(Vec<u32>, String, Vec<String>)>,
    prompt_logprobs: &[PromptTokenLogprob],
    reason: FinishReason,
    usage: ValidatedUsage,
) -> Value {
    let mut value = json!({
        "id": metadata.response_id,
        "object": "text_completion",
        "created": metadata.created,
        "model": metadata.model,
        "choices": [{
            "text": if metadata.logprobs.echo { format!("{}{}", prompt.map_or("", |item| item.1.as_str()), content) } else { content.to_owned() },
            "index": metadata.choice_index,
            "logprobs": completion_logprobs_with_prompt(tokens, metadata.logprobs, prompt, prompt_logprobs),
            "finish_reason": finish_reason(reason),
        }],
        "usage": usage_value(usage),
    });
    if let Some((ids, text, _)) = prompt {
        value["prompt_token_ids"] = json!(ids);
        value["prompt_text"] = json!(text);
        if metadata.logprobs.prompt_top.is_some() {
            value["prompt_logprobs"] = prompt_logprob_positions(ids, prompt_logprobs);
        }
    }
    value
}

struct ChatValueInput<'a> {
    content: &'a str,
    reasoning: &'a str,
    tokens: &'a [GeneratedTokenLogprob],
    tool_calls: &'a [(String, String)],
    prompt: Option<&'a (Vec<u32>, String, Vec<String>)>,
    prompt_logprobs: &'a [PromptTokenLogprob],
    reason: FinishReason,
    usage: ValidatedUsage,
}

fn chat_value(metadata: &ResponseMetadata, input: ChatValueInput<'_>) -> Value {
    let ChatValueInput {
        content,
        reasoning,
        tokens,
        tool_calls,
        prompt,
        prompt_logprobs,
        reason,
        usage,
    } = input;
    let mut message = serde_json::Map::new();
    message.insert("role".to_owned(), json!("assistant"));
    message.insert(
        "content".to_owned(),
        if content.is_empty() && !tool_calls.is_empty() {
            Value::Null
        } else {
            json!(content)
        },
    );
    if !reasoning.is_empty()
        && metadata
            .chat
            .as_ref()
            .is_none_or(|chat| chat.include_reasoning)
    {
        message.insert(
            metadata.frontend.reasoning_field().to_owned(),
            json!(reasoning),
        );
    }
    if !tool_calls.is_empty() {
        message.insert(
            "tool_calls".to_owned(),
            Value::Array(
                tool_calls
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| metadata.parallel_tool_calls || *index == 0)
                    .map(|(index, (name, arguments))| {
                        tool_call_value(metadata, index, name, arguments)
                    })
                    .collect(),
            ),
        );
    }
    let mut value = json!({
        "id": metadata.response_id,
        "object": "chat.completion",
        "created": metadata.created,
        "model": metadata.model,
        "choices": [{
            "index": metadata.choice_index,
            "message": Value::Object(message),
            "logprobs": chat_logprobs(metadata, tokens),
            "finish_reason": finish_reason(reason),
        }],
        "usage": usage_value(usage),
    });
    if let Some((ids, _, _)) = prompt {
        if metadata.logprobs.return_token_ids {
            value["prompt_token_ids"] = json!(ids);
        }
        if metadata.logprobs.prompt_top.is_some() {
            value["prompt_logprobs"] = prompt_logprob_positions(ids, prompt_logprobs);
        }
    }
    value
}

fn prompt_logprob_positions(ids: &[u32], rows: &[PromptTokenLogprob]) -> Value {
    let mut positions = vec![Value::Null; ids.len()];
    for row in rows {
        if row.position < positions.len() && ids[row.position] == row.token_id {
            positions[row.position] = prompt_logprob_value(row);
        }
    }
    Value::Array(positions)
}

fn prompt_logprob_value(row: &PromptTokenLogprob) -> Value {
    let mut entries = serde_json::Map::new();
    entries.insert(
        row.token_id.to_string(),
        json!({"logprob": row.logprob, "decoded_token": row.token}),
    );
    for (rank, (id, top)) in row.top_logprobs.iter().enumerate() {
        entries.insert(
            id.to_string(),
            json!({"logprob": top.logprob, "rank": rank + 1, "decoded_token": top.token}),
        );
    }
    Value::Object(entries)
}

/// Completion IDs, scores and decoded text arrive as separate backend events.
/// Hold one complete chunk so Finished can mark the last token's chunk.
#[derive(Default)]
struct CompletionIdStream {
    prompt: Option<(Vec<u32>, String, Vec<String>)>,
    prompt_ids: Option<Vec<u32>>,
    positions: Vec<PromptTokenLogprob>,
    ids: Vec<u32>,
    records: Vec<GeneratedTokenLogprob>,
    text_offset: usize,
    chunk: Option<Value>,
}

impl CompletionIdStream {
    fn has_pending_tokens(&self) -> bool {
        !self.ids.is_empty() || !self.records.is_empty()
    }

    fn stage(&mut self, metadata: &ResponseMetadata, text: &str) -> Option<Value> {
        let logprobs = if let Some(prompt) = self.prompt.take() {
            metadata.logprobs.enabled.then(|| {
                completion_logprobs_with_prompt(
                    &self.records,
                    metadata.logprobs,
                    Some(&prompt),
                    &self.positions,
                )
            })
        } else {
            metadata
                .logprobs
                .enabled
                .then(|| completion_logprobs(&self.records, self.text_offset))
        };
        self.records.clear();
        self.positions.clear();
        let mut chunk = stream_chunk(metadata, json!(text), None, logprobs);
        chunk["choices"][0]["token_ids"] = json!(std::mem::take(&mut self.ids));
        chunk["choices"][0]["prompt_token_ids"] = json!(self.prompt_ids.take());
        // vLLM suppresses echoed text with return_token_ids. Subsequent
        // offsets count only the generated text, even after prompt rows.
        self.text_offset += text.chars().count();
        self.chunk.replace(chunk)
    }
}

fn streaming_response(
    mut metadata: ResponseMetadata,
    mut events: mpsc::Receiver<GenerationEvent>,
    mut active: ActiveRequest,
) -> Response {
    let (frames, body_rx) = mpsc::channel::<Result<Bytes, Infallible>>(1);
    let body_tx = SseSender {
        frames,
        write_timeout: metadata.stream_write_timeout,
        drain_expired: active.drain_expired.clone(),
    };
    let response_metadata = metadata.clone();
    tokio::spawn(async move {
        let mut token_ids = GeneratedTokenBudget::new(metadata.max_new_tokens);
        let mut chat_started = false;
        let mut reasoning_tokens = 0_u64;
        let mut tool_index = 0_usize;
        let mut completion_offset = 0_usize;
        let mut completion_ids = (metadata.endpoint == Endpoint::Completion
            && metadata.logprobs.return_token_ids)
            .then(CompletionIdStream::default);
        let mut echo_prompt: Option<(Vec<u32>, String, Vec<String>)> = None;
        let mut stream_prompt_ids: Option<Vec<u32>> = None;
        let mut echo_positions = Vec::new();
        // Records of the latest generated tokens, which ride on the next chunk.
        let mut pending_logprobs = Vec::new();
        let mut fold = match EventFold::for_response(&metadata) {
            Ok(fold) => fold,
            Err(error) => {
                let _ = send_sse(&body_tx, &error.value()).await;
                let _ = send_done(&body_tx).await;
                return;
            }
        };
        if !metadata.logprobs.return_token_ids
            && !send_chat_start(&body_tx, &metadata, &mut chat_started, None).await
        {
            return;
        }
        loop {
            let event = tokio::select! {
                _ = body_tx.closed() => return,
                event = active.next_event(&mut events) => event,
            };
            let Some(event) = event else {
                let error = ApiError::backend(
                    "generation backend closed without a terminal event",
                    Some("incomplete_generation".to_owned()),
                );
                let _ = send_sse(&body_tx, &error.value()).await;
                let _ = send_done(&body_tx).await;
                return;
            };
            metadata.completion_tokens = metadata.completion_tokens.saturating_add(match &event {
                GenerationEvent::Content(event) => event.generated_token_count as u64,
                GenerationEvent::Reasoning(event) => event.generated_token_count as u64,
                GenerationEvent::ToolCalls {
                    generated_token_count,
                    ..
                }
                | GenerationEvent::TokenUsageDelta {
                    generated_token_count,
                } => *generated_token_count as u64,
                _ => 0,
            });
            match event {
                GenerationEvent::AdmissionFailed(error) => {
                    let _ = send_sse(&body_tx, &admission_error(error).value()).await;
                    let _ = send_done(&body_tx).await;
                    return;
                }
                GenerationEvent::GeneratedTokenIds(ids) => {
                    if !metadata.logprobs.return_token_ids
                        || ids.is_empty()
                        || token_ids.observe(ids.len()).is_err()
                    {
                        let _ = send_sse(
                            &body_tx,
                            &invalid_backend_event("invalid generated token IDs").value(),
                        )
                        .await;
                        let _ = send_done(&body_tx).await;
                        return;
                    }
                    if let Some(stream) = &mut completion_ids {
                        if !metadata.logprobs.enabled
                            && stream.has_pending_tokens()
                            && let Some(chunk) = stream.stage(&metadata, "")
                            && !send_sse(&body_tx, &chunk).await
                        {
                            return;
                        }
                        stream.ids.extend(ids);
                        continue;
                    }
                    if expose_token_ids(&metadata) {
                        let delta = if metadata.endpoint == Endpoint::Chat {
                            json!({})
                        } else {
                            json!("")
                        };
                        let logprobs = take_stream_logprobs(
                            &metadata,
                            &mut pending_logprobs,
                            completion_offset,
                        );
                        let mut chunk = stream_chunk(&metadata, delta, None, logprobs);
                        chunk["choices"][0]["token_ids"] = json!(ids);
                        if !send_sse(&body_tx, &chunk).await {
                            return;
                        }
                    }
                }
                GenerationEvent::Prompt {
                    token_ids,
                    text,
                    tokens,
                } => {
                    if metadata.endpoint == Endpoint::Chat && metadata.logprobs.return_token_ids {
                        stream_prompt_ids = Some(token_ids.clone());
                        if !send_chat_start(
                            &body_tx,
                            &metadata,
                            &mut chat_started,
                            stream_prompt_ids.as_deref(),
                        )
                        .await
                        {
                            return;
                        }
                    }
                    if let Some(stream) = &mut completion_ids {
                        stream.prompt_ids = Some(token_ids.clone());
                        if metadata.logprobs.echo {
                            stream.prompt = Some((token_ids, text, tokens));
                        }
                        continue;
                    }
                    if metadata.logprobs.echo {
                        echo_prompt = Some((token_ids, text, tokens));
                    }
                }
                GenerationEvent::PromptLogprobs(rows) => {
                    if let Some(stream) = &mut completion_ids {
                        stream.positions.extend(rows);
                    } else {
                        echo_positions.extend(rows);
                    }
                }
                GenerationEvent::GeneratedTokenLogprobs(records) => {
                    if let Err(error) = fold.token_logprobs(&records) {
                        let _ = send_sse(&body_tx, &error.value()).await;
                        let _ = send_done(&body_tx).await;
                        return;
                    }
                    if let Some(stream) = &mut completion_ids {
                        if stream.has_pending_tokens()
                            && let Some(chunk) = stream.stage(&metadata, "")
                            && !send_sse(&body_tx, &chunk).await
                        {
                            return;
                        }
                        stream.records = records;
                        continue;
                    }
                    // A step whose tokens produced no chunk: a completion
                    // still sends one with empty text, while chat skips it
                    // and its logprobs, as vLLM's reasoning and tool parsers
                    // do for markers.
                    if !pending_logprobs.is_empty() && metadata.endpoint == Endpoint::Completion {
                        if let Some(prompt) = echo_prompt.take() {
                            completion_offset = prompt.1.chars().count();
                            if !send_echo_prompt(&body_tx, &metadata, &prompt, &echo_positions)
                                .await
                            {
                                return;
                            }
                        }
                        let logprobs = take_stream_logprobs(
                            &metadata,
                            &mut pending_logprobs,
                            completion_offset,
                        );
                        if !send_sse(
                            &body_tx,
                            &stream_chunk(&metadata, json!(""), None, logprobs),
                        )
                        .await
                        {
                            return;
                        }
                    }
                    pending_logprobs = records;
                }
                GenerationEvent::Content(content_event) => {
                    if let Some(prompt) = echo_prompt.take() {
                        completion_offset = prompt.1.chars().count();
                        if !send_echo_prompt(&body_tx, &metadata, &prompt, &echo_positions).await {
                            return;
                        }
                    }
                    if let Err(error) = fold.content(&content_event) {
                        let _ = send_sse(&body_tx, &error.value()).await;
                        let _ = send_done(&body_tx).await;
                        return;
                    }
                    if let Some(stream) = &mut completion_ids {
                        if let Some(chunk) = stream.stage(&metadata, &content_event.delta)
                            && !send_sse(&body_tx, &chunk).await
                        {
                            return;
                        }
                        continue;
                    }
                    let next_completion_offset = if metadata.endpoint == Endpoint::Completion {
                        match completion_offset.checked_add(content_event.delta.chars().count()) {
                            Some(offset) => offset,
                            None => {
                                let error =
                                    invalid_backend_event("completion token offsets overflowed");
                                let _ = send_sse(&body_tx, &error.value()).await;
                                let _ = send_done(&body_tx).await;
                                return;
                            }
                        }
                    } else {
                        completion_offset
                    };
                    if !send_chat_start(
                        &body_tx,
                        &metadata,
                        &mut chat_started,
                        stream_prompt_ids.as_deref(),
                    )
                    .await
                    {
                        return;
                    }
                    let delta = match metadata.endpoint {
                        Endpoint::Completion => json!(content_event.delta),
                        Endpoint::Chat => json!({"content": content_event.delta}),
                    };
                    let logprobs =
                        take_stream_logprobs(&metadata, &mut pending_logprobs, completion_offset);
                    let chunk = stream_chunk(&metadata, delta, None, logprobs);
                    if !send_sse(&body_tx, &chunk).await {
                        return;
                    }
                    if metadata.endpoint == Endpoint::Completion {
                        completion_offset = next_completion_offset;
                    }
                }
                GenerationEvent::Reasoning(reasoning_event) => {
                    reasoning_tokens += reasoning_event.generated_token_count as u64;
                    if let Err(error) = fold.reasoning(&reasoning_event) {
                        let _ = send_sse(&body_tx, &error.value()).await;
                        let _ = send_done(&body_tx).await;
                        return;
                    }
                    if !send_chat_start(
                        &body_tx,
                        &metadata,
                        &mut chat_started,
                        stream_prompt_ids.as_deref(),
                    )
                    .await
                    {
                        return;
                    }
                    if metadata
                        .chat
                        .as_ref()
                        .is_some_and(|chat| !chat.include_reasoning)
                    {
                        continue;
                    }
                    let mut delta = serde_json::Map::new();
                    delta.insert(
                        metadata.frontend.reasoning_field().to_owned(),
                        json!(reasoning_event.delta),
                    );
                    let logprobs =
                        take_stream_logprobs(&metadata, &mut pending_logprobs, completion_offset);
                    let chunk = stream_chunk(&metadata, Value::Object(delta), None, logprobs);
                    if !send_sse(&body_tx, &chunk).await {
                        return;
                    }
                }
                GenerationEvent::ToolCallDeltas(deltas) => {
                    if let Err(error) = fold.tool_deltas(&deltas) {
                        let _ = send_sse(&body_tx, &error.value()).await;
                        let _ = send_done(&body_tx).await;
                        return;
                    }
                    let values = tool_delta_values(&metadata, deltas);
                    if !values.is_empty() {
                        let logprobs = take_stream_logprobs(
                            &metadata,
                            &mut pending_logprobs,
                            completion_offset,
                        );
                        let chunk =
                            stream_chunk(&metadata, json!({"tool_calls": values}), None, logprobs);
                        if !send_sse(&body_tx, &chunk).await {
                            return;
                        }
                    }
                }
                GenerationEvent::ToolCalls {
                    calls,
                    generated_token_count,
                } => {
                    let group = match fold.tool_calls(&calls, generated_token_count) {
                        Ok(group) => group,
                        Err(error) => {
                            let _ = send_sse(&body_tx, &error.value()).await;
                            let _ = send_done(&body_tx).await;
                            return;
                        }
                    };
                    let next_tool_index = group.next;
                    if !send_chat_start(
                        &body_tx,
                        &metadata,
                        &mut chat_started,
                        stream_prompt_ids.as_deref(),
                    )
                    .await
                    {
                        return;
                    }
                    // Streamed calls were sent as their deltas arrived; the
                    // fold checked that this group repeats them.
                    if group.streamed {
                        tool_index = next_tool_index;
                        continue;
                    }
                    let values = calls
                        .iter()
                        .enumerate()
                        .filter(|(offset, _)| metadata.parallel_tool_calls || tool_index + offset == 0)
                        .map(|(offset, call)| {
                            let index = tool_index + offset;
                            let value = tool_call_value(
                                &metadata,
                                index,
                                &call.name,
                                &call.arguments,
                            );
                            json!({"index": index, "id": value["id"], "type": "function", "function": value["function"]})
                        })
                        .collect::<Vec<_>>();
                    tool_index = next_tool_index;
                    if metadata.parallel_tool_calls || tool_index == calls.len() {
                        let logprobs = take_stream_logprobs(
                            &metadata,
                            &mut pending_logprobs,
                            completion_offset,
                        );
                        let chunk =
                            stream_chunk(&metadata, json!({"tool_calls": values}), None, logprobs);
                        if !send_sse(&body_tx, &chunk).await {
                            return;
                        }
                    }
                }
                GenerationEvent::TokenUsageDelta {
                    generated_token_count,
                } => {
                    if let Err(error) = fold.token_usage(generated_token_count) {
                        let _ = send_sse(&body_tx, &error.value()).await;
                        let _ = send_done(&body_tx).await;
                        return;
                    }
                }
                GenerationEvent::Finished { reason, usage } => {
                    if let Some(prompt) = echo_prompt.take() {
                        completion_offset = prompt.1.chars().count();
                        if !send_echo_prompt(&body_tx, &metadata, &prompt, &echo_positions).await {
                            return;
                        }
                    }
                    if metadata.logprobs.return_token_ids
                        && token_ids.observed as u64 != usage.completion_tokens
                    {
                        let _ = send_sse(
                            &body_tx,
                            &invalid_backend_event("generated token IDs differ from usage").value(),
                        )
                        .await;
                        let _ = send_done(&body_tx).await;
                        return;
                    }
                    let finished = fold.finish(reason, usage);
                    active.complete();
                    let (reason, usage) = match finished {
                        Ok(result) => result,
                        Err(error) => {
                            let _ = send_sse(&body_tx, &error.value()).await;
                            let _ = send_done(&body_tx).await;
                            return;
                        }
                    };
                    if !send_chat_start(
                        &body_tx,
                        &metadata,
                        &mut chat_started,
                        stream_prompt_ids.as_deref(),
                    )
                    .await
                    {
                        return;
                    }
                    let empty_delta = if metadata.endpoint == Endpoint::Chat {
                        json!({})
                    } else {
                        json!("")
                    };
                    let logprobs =
                        take_stream_logprobs(&metadata, &mut pending_logprobs, completion_offset);
                    let terminal = if let Some(stream) = &mut completion_ids {
                        if (stream.has_pending_tokens() || stream.chunk.is_none())
                            && let Some(chunk) = stream.stage(&metadata, "")
                            && !send_sse(&body_tx, &chunk).await
                        {
                            return;
                        }
                        let mut chunk = stream
                            .chunk
                            .take()
                            .expect("completion has a terminal chunk");
                        chunk["choices"][0]["finish_reason"] = json!(finish_reason(reason));
                        chunk
                    } else {
                        stream_chunk(
                            &metadata,
                            empty_delta,
                            Some(finish_reason(reason)),
                            logprobs,
                        )
                    };
                    if !send_sse(&body_tx, &terminal).await {
                        return;
                    }
                    if metadata.include_usage {
                        let mut usage_chunk = json!({
                            "id": metadata.response_id,
                            "object": stream_object(metadata.endpoint),
                            "created": metadata.created,
                            "model": metadata.model,
                            "choices": [],
                            "usage": usage_value(usage),
                        });
                        if metadata.responses {
                            usage_chunk["usage"]["completion_tokens_details"] =
                                json!({"reasoning_tokens": reasoning_tokens});
                        }
                        if !send_sse(&body_tx, &usage_chunk).await {
                            return;
                        }
                    }
                    let _ = send_done(&body_tx).await;
                    return;
                }
                GenerationEvent::Failed { message, code } => {
                    active.complete();
                    if let Err(error) = fold.failure(&message, code.as_deref()) {
                        let _ = send_sse(&body_tx, &error.value()).await;
                        let _ = send_done(&body_tx).await;
                        return;
                    }
                    let error = ApiError::backend(message, code);
                    let _ = send_sse(&body_tx, &error.value()).await;
                    let _ = send_done(&body_tx).await;
                    return;
                }
            }
        }
    });
    let mut response = Response::new(Body::from_stream(ReceiverStream::new(body_rx)));
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response
        .headers_mut()
        .insert(CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    insert_request_id(&mut response, &response_metadata.request_id);
    response
}

async fn send_echo_prompt(
    body_tx: &SseSender,
    metadata: &ResponseMetadata,
    prompt: &(Vec<u32>, String, Vec<String>),
    positions: &[PromptTokenLogprob],
) -> bool {
    let logprobs = metadata
        .logprobs
        .enabled
        .then(|| completion_logprobs_with_prompt(&[], metadata.logprobs, Some(prompt), positions));
    send_sse(
        body_tx,
        &stream_chunk(metadata, json!(prompt.1), None, logprobs),
    )
    .await
}

fn expose_token_ids(metadata: &ResponseMetadata) -> bool {
    metadata.logprobs.return_token_ids
        && metadata
            .chat
            .as_ref()
            .is_none_or(|chat| chat.include_reasoning)
}

fn stream_chunk(
    metadata: &ResponseMetadata,
    delta_or_text: Value,
    finish: Option<&str>,
    logprobs: Option<Value>,
) -> Value {
    let choice = match metadata.endpoint {
        Endpoint::Completion => json!({
            "text": delta_or_text,
            "index": metadata.choice_index,
            "logprobs": logprobs,
            "finish_reason": finish,
        }),
        Endpoint::Chat => json!({
            "index": metadata.choice_index,
            "delta": delta_or_text,
            "logprobs": logprobs,
            "finish_reason": finish,
        }),
    };
    let mut value = json!({
        "id": metadata.response_id,
        "object": stream_object(metadata.endpoint),
        "created": metadata.created,
        "model": metadata.model,
        "choices": [choice],
    });
    if expose_token_ids(metadata) {
        value["choices"][0]["token_ids"] = json!([]);
    }
    if metadata.include_usage {
        value["usage"] = if metadata.continuous_usage_stats {
            usage_value(ValidatedUsage {
                prompt_tokens: metadata.prompt_tokens,
                completion_tokens: metadata.completion_tokens,
                total_tokens: metadata.prompt_tokens + metadata.completion_tokens,
                cached_tokens: None,
            })
        } else {
            Value::Null
        };
    }
    value
}

async fn send_sse(sender: &SseSender, value: &Value) -> bool {
    let mut frame = String::from("data: ");
    match serde_json::to_string(value) {
        Ok(json) => frame.push_str(&json),
        Err(_) => return false,
    }
    frame.push_str("\n\n");
    sender.frame(Bytes::from(frame)).await
}

async fn send_chat_start(
    sender: &SseSender,
    metadata: &ResponseMetadata,
    started: &mut bool,
    prompt_ids: Option<&[u32]>,
) -> bool {
    if metadata.endpoint != Endpoint::Chat || *started {
        return true;
    }
    if metadata.logprobs.return_token_ids && prompt_ids.is_none() {
        let error = invalid_backend_event("backend omitted requested prompt token IDs");
        let _ = send_sse(sender, &error.value()).await;
        let _ = send_done(sender).await;
        return false;
    }
    let mut initial = stream_chunk(
        metadata,
        json!({"role": "assistant", "content": ""}),
        None,
        None,
    );
    if let Some(ids) = prompt_ids {
        initial["prompt_token_ids"] = json!(ids);
    }
    if send_sse(sender, &initial).await {
        *started = true;
        true
    } else {
        false
    }
}

async fn send_done(sender: &SseSender) -> bool {
    sender.frame(Bytes::from_static(b"data: [DONE]\n\n")).await
}

fn validate_content_event(
    content: &VisibleContent,
    output_budget: &mut OutputBudget,
) -> Result<(), ApiError> {
    if content.delta.is_empty() {
        return Err(invalid_backend_event(
            "backend visible-content delta must not be empty",
        ));
    }
    output_budget.charge(content.delta.len(), "content event")
}

/// Checks one event of generated-token logprob records, counts its tokens
/// and charges its size to the output budget.
fn validate_token_logprobs(
    records: &[GeneratedTokenLogprob],
    requested: RequestedLogprobs,
    tokens: &mut GeneratedTokenBudget,
    output_budget: &mut OutputBudget,
) -> Result<(), ApiError> {
    if !requested.enabled {
        return Err(invalid_backend_event(
            "backend returned unrequested logprob data",
        ));
    }
    if records.is_empty()
        || records.iter().any(|record| {
            !record.logprob.is_finite()
                || record.top_logprobs.len() > requested.top.saturating_add(1)
                || record
                    .top_logprobs
                    .iter()
                    .any(|entry| !entry.logprob.is_finite() && entry.logprob != f64::NEG_INFINITY)
        })
    {
        return Err(invalid_backend_event(
            "backend returned invalid or excessive token logprob data",
        ));
    }
    tokens.observe(records.len())?;
    let mut bytes = 0_usize;
    for record in records {
        bytes = bytes
            .checked_add(record.token.len())
            .and_then(|bytes| bytes.checked_add(record.bytes.len()))
            .ok_or_else(|| invalid_backend_event("backend logprob event size overflowed"))?;
        for entry in &record.top_logprobs {
            bytes = bytes
                .checked_add(entry.token.len())
                .and_then(|bytes| bytes.checked_add(entry.bytes.len()))
                .ok_or_else(|| invalid_backend_event("backend logprob event size overflowed"))?;
        }
    }
    output_budget.charge(bytes, "logprob event")
}

/// Requested logprobs cover every generated token, as vLLM's do.
fn validate_logprob_count(
    logprobs: RequestedLogprobs,
    observed: usize,
    usage: ValidatedUsage,
) -> Result<ValidatedUsage, ApiError> {
    if logprobs.enabled && observed as u64 != usage.completion_tokens {
        return Err(invalid_backend_event(
            "generated token logprobs differ from usage",
        ));
    }
    Ok(usage)
}

fn validate_reasoning_event(
    chat: Option<&ChatMetadata>,
    reasoning: &ReasoningContent,
    output_budget: &mut OutputBudget,
) -> Result<(), ApiError> {
    if reasoning.delta.is_empty() {
        return Err(invalid_backend_event(
            "backend reasoning event delta must not be empty",
        ));
    }
    let Some(chat) = chat else {
        return Err(invalid_backend_event(
            "completion backend emitted a reasoning event",
        ));
    };
    if !chat.state.reasoning_enabled() {
        return Err(invalid_backend_event(
            "reasoning event requires thinking mode",
        ));
    }
    output_budget.charge(reasoning.delta.len(), "reasoning output")
}

fn validate_tool_calls_event(
    frontend: &dyn ServerFrontendV1,
    chat: Option<&ChatMetadata>,
    calls: &[BackendToolCall],
    existing_count: usize,
    output_budget: &mut OutputBudget,
) -> Result<usize, ApiError> {
    let (next_count, bytes) = check_tool_call_group(frontend, chat, calls, existing_count)?;
    output_budget.charge(bytes, "tool-call group")?;
    Ok(next_count)
}

/// Checks one tool-call group a backend emitted after `existing_count`
/// calls, and returns the new call count and the group's bytes.
fn check_tool_call_group(
    frontend: &dyn ServerFrontendV1,
    chat: Option<&ChatMetadata>,
    calls: &[BackendToolCall],
    existing_count: usize,
) -> Result<(usize, usize), ApiError> {
    let Some(chat) = chat else {
        return Err(invalid_backend_event(
            "completion backend emitted a tool call",
        ));
    };
    let maximum = frontend.maximum_tool_calls();
    if calls.is_empty() || calls.len() > maximum {
        return Err(invalid_backend_event(
            "backend tool-call group must contain 1 through 128 calls",
        ));
    }
    let next_count = existing_count
        .checked_add(calls.len())
        .ok_or_else(|| invalid_backend_event("backend tool-call count overflowed"))?;
    if next_count > maximum {
        return Err(invalid_backend_event(
            "backend emitted more tool calls than the codec limit",
        ));
    }
    let mut bytes = 0_usize;
    for call in calls {
        if !function_available(chat, &call.name) {
            return Err(invalid_backend_event(
                "backend tool call names an unavailable function",
            ));
        }
        // Only this model knows whether the name is well formed and whether the
        // arguments parse the way it serializes them.
        frontend
            .validate_tool_call(&call.name, &call.arguments)
            .map_err(|error| invalid_backend_event(&error.message))?;
        bytes = bytes
            .checked_add(call.name.len())
            .and_then(|bytes| bytes.checked_add(call.arguments.len()))
            .ok_or_else(|| invalid_backend_event("backend tool-call group size overflowed"))?;
    }
    Ok((next_count, bytes))
}

/// Whether a backend may name `name` in a call for this request.
///
/// A request that offers no function never yields a call. Otherwise the name
/// is the model's output, which the reference serves whether or not it was
/// offered; only a named choice fixes which function the constraint allowed.
fn function_available(chat: &ChatMetadata, name: &str) -> bool {
    !chat.allowed_function_names.is_empty()
        && (!chat.named_tool_choice
            || chat
                .allowed_function_names
                .iter()
                .any(|allowed| allowed == name))
}

fn validate_tool_deltas(
    frontend: &dyn ServerFrontendV1,
    chat: Option<&ChatMetadata>,
    deltas: &[crate::ToolCallDelta],
    calls: &mut Vec<BackendToolCall>,
    budget: &mut OutputBudget,
) -> Result<(), ApiError> {
    let chat = chat.ok_or_else(|| invalid_backend_event("completion emitted tool deltas"))?;
    if deltas.is_empty() || deltas.len() > frontend.maximum_tool_calls() {
        return Err(invalid_backend_event("invalid tool delta count"));
    }
    for delta in deltas {
        if let Some(name) = &delta.name {
            if delta.index != calls.len()
                || calls.len() >= frontend.maximum_tool_calls()
                || !function_available(chat, name)
            {
                return Err(invalid_backend_event("invalid tool delta name or index"));
            }
            budget.charge(name.len(), "tool delta name")?;
            calls.push(BackendToolCall {
                name: name.clone(),
                arguments: String::new(),
            });
        }
        let call = calls
            .get_mut(delta.index)
            .ok_or_else(|| invalid_backend_event("tool delta precedes name"))?;
        budget.charge(delta.arguments.len(), "tool argument delta")?;
        call.arguments.push_str(&delta.arguments);
    }
    Ok(())
}

/// The stream chunk values of one checked group of tool-call deltas.
fn tool_delta_values(metadata: &ResponseMetadata, deltas: Vec<crate::ToolCallDelta>) -> Vec<Value> {
    deltas
        .into_iter()
        .filter(|delta| metadata.parallel_tool_calls || delta.index == 0)
        .map(|delta| {
            let mut value =
                json!({"index": delta.index, "function": {"arguments": delta.arguments}});
            if let Some(name) = delta.name {
                let full = tool_call_value(metadata, delta.index, &name, "");
                value["id"] = full["id"].clone();
                value["type"] = json!("function");
                value["function"]["name"] = json!(name);
            }
            value
        })
        .collect()
}

fn finish_reason_for(
    endpoint: Endpoint,
    chat: Option<&ChatMetadata>,
    reason: FinishReason,
    tool_call_count: usize,
) -> Result<FinishReason, ApiError> {
    let tools_allowed = chat.is_some_and(|chat| !chat.allowed_function_names.is_empty());
    match (endpoint, reason, tool_call_count, tools_allowed) {
        (Endpoint::Completion, FinishReason::ToolCalls, _, _) => Err(invalid_backend_event(
            "completion backend returned a tool_calls finish reason",
        )),
        (Endpoint::Chat, FinishReason::ToolCalls, _, false) => Err(invalid_backend_event(
            "tool_calls finish reason requires current tools",
        )),
        (Endpoint::Chat, FinishReason::ToolCalls, 0, true) => Err(invalid_backend_event(
            "tool_calls finish reason requires an observed tool call",
        )),
        (Endpoint::Chat, _, count, true)
            if count > 0 && chat.is_some_and(|chat| chat.named_tool_choice) =>
        {
            Ok(FinishReason::Stop)
        }
        (Endpoint::Chat, _, count, true) if count > 0 => Ok(FinishReason::ToolCalls),
        (_, reason, _, _) => Ok(reason),
    }
}

fn validate_usage(
    max_new_tokens: usize,
    usage: TokenUsage,
    observed_events: usize,
) -> Result<ValidatedUsage, ApiError> {
    let max_new_tokens = u64::try_from(max_new_tokens)
        .map_err(|_| invalid_backend_event("max_new_tokens does not fit usage accounting"))?;
    if usage.completion_tokens > max_new_tokens {
        return Err(invalid_backend_event(
            "backend completion token usage exceeds max_new_tokens",
        ));
    }
    let observed_events = u64::try_from(observed_events)
        .map_err(|_| invalid_backend_event("observed semantic event count overflowed"))?;
    if usage.completion_tokens != observed_events {
        return Err(invalid_backend_event(
            "backend completion token usage differs from observed token accounting",
        ));
    }
    let total_tokens = usage
        .prompt_tokens
        .checked_add(usage.completion_tokens)
        .ok_or_else(|| invalid_backend_event("backend total token usage overflowed"))?;
    if usage
        .cached_tokens
        .is_some_and(|cached| cached >= usage.prompt_tokens.max(1))
    {
        return Err(invalid_backend_event(
            "backend cached prompt tokens are not fewer than the prompt tokens",
        ));
    }
    Ok(ValidatedUsage {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens,
        cached_tokens: usage.cached_tokens,
    })
}

fn validate_failure_event(
    message: &str,
    code: Option<&str>,
    output_budget: &mut OutputBudget,
) -> Result<(), ApiError> {
    let bytes = message
        .len()
        .checked_add(code.map_or(0, str::len))
        .ok_or_else(|| invalid_backend_event("backend failure event size overflowed"))?;
    output_budget.charge(bytes, "failure event")
}

fn invalid_backend_event(message: &str) -> ApiError {
    ApiError::backend(message, Some("invalid_backend_event".to_owned()))
}

/// Whether generated-token logprobs are returned. vLLM hides them together
/// with the reasoning when `include_reasoning` is false.
fn expose_logprobs(metadata: &ResponseMetadata) -> bool {
    metadata.logprobs.enabled
        && metadata
            .chat
            .as_ref()
            .is_none_or(|chat| chat.include_reasoning)
}

/// vLLM's floor for reported logprobs.
fn clamp_logprob(logprob: f64) -> f64 {
    logprob.max(-9999.0)
}

/// Completion logprobs for generated tokens. As in vLLM, each offset adds the
/// previous token's label length to the first one, so hidden stop tokens and
/// corrected byte labels shift offsets instead of following the text.
fn completion_logprobs(tokens: &[GeneratedTokenLogprob], text_offset: usize) -> Value {
    let mut offsets = Vec::with_capacity(tokens.len());
    let mut offset = text_offset;
    for token in tokens {
        offsets.push(offset);
        offset += token.token.chars().count();
    }
    json!({
        "tokens": tokens.iter().map(|token| token.token.as_str()).collect::<Vec<_>>(),
        "token_logprobs": tokens.iter().map(|token| clamp_logprob(token.logprob)).collect::<Vec<_>>(),
        "top_logprobs": tokens.iter().map(completion_top_logprobs).collect::<Vec<_>>(),
        "text_offset": offsets,
    })
}

/// Every entry of the position, the generated token first, keyed by label.
fn completion_top_logprobs(token: &GeneratedTokenLogprob) -> Value {
    Value::Object(
        token
            .top_logprobs
            .iter()
            .map(|entry| (entry.token.clone(), json!(clamp_logprob(entry.logprob))))
            .collect(),
    )
}

fn completion_logprobs_with_prompt(
    tokens: &[GeneratedTokenLogprob],
    requested: RequestedLogprobs,
    prompt: Option<&(Vec<u32>, String, Vec<String>)>,
    positions: &[PromptTokenLogprob],
) -> Value {
    if !requested.enabled {
        return Value::Null;
    }
    let Some((ids, _, prompt_tokens)) = prompt.filter(|_| requested.echo) else {
        return completion_logprobs(tokens, 0);
    };
    let mut token_scores = vec![Value::Null; ids.len()];
    let mut top_scores = vec![Value::Null; ids.len()];
    let mut offsets = Vec::with_capacity(ids.len() + tokens.len());
    let mut offset = 0;
    for token in prompt_tokens {
        offsets.push(offset);
        offset += token.chars().count();
    }
    for row in positions {
        if row.position < ids.len() && ids[row.position] == row.token_id {
            token_scores[row.position] = json!(clamp_logprob(row.logprob));
            top_scores[row.position] = Value::Object(
                std::iter::once((row.token.clone(), json!(clamp_logprob(row.logprob))))
                    .chain(
                        row.top_logprobs
                            .iter()
                            .take(requested.top)
                            .map(|(_, top)| (top.token.clone(), json!(clamp_logprob(top.logprob)))),
                    )
                    .collect(),
            );
        }
    }
    let mut value = json!({"tokens": prompt_tokens, "token_logprobs": token_scores, "top_logprobs": top_scores, "text_offset": offsets});
    let generated = completion_logprobs(tokens, offset);
    for field in ["tokens", "token_logprobs", "top_logprobs", "text_offset"] {
        if let (Some(target), Some(values)) =
            (value[field].as_array_mut(), generated[field].as_array())
        {
            target.extend(values.iter().cloned());
        }
    }
    value
}

fn chat_logprobs(metadata: &ResponseMetadata, tokens: &[GeneratedTokenLogprob]) -> Value {
    if !expose_logprobs(metadata) {
        return Value::Null;
    }
    json!({
        "content": tokens.iter().map(|token| chat_token_logprob(token, metadata.logprobs.top)).collect::<Vec<_>>(),
        "refusal": Value::Null,
    })
}

/// vLLM's chat row: the first `top` entries of the position, which put the
/// generated token first when it was not the most likely one.
fn chat_token_logprob(token: &GeneratedTokenLogprob, top: usize) -> Value {
    json!({
        "token": token.token,
        "bytes": token.bytes,
        "logprob": clamp_logprob(token.logprob),
        "top_logprobs": token.top_logprobs.iter().take(top).map(|entry| json!({
            "token": entry.token,
            "bytes": entry.bytes,
            "logprob": clamp_logprob(entry.logprob),
        })).collect::<Vec<_>>(),
    })
}

/// The logprobs a stream chunk carries: those of the tokens generated since
/// the previous chunk. Records are taken even when hidden, so they never
/// reach a later chunk.
fn take_stream_logprobs(
    metadata: &ResponseMetadata,
    pending: &mut Vec<GeneratedTokenLogprob>,
    text_offset: usize,
) -> Option<Value> {
    let records = std::mem::take(pending);
    if records.is_empty() || !expose_logprobs(metadata) {
        return None;
    }
    Some(match metadata.endpoint {
        Endpoint::Chat => chat_logprobs(metadata, &records),
        Endpoint::Completion => completion_logprobs(&records, text_offset),
    })
}

fn tool_call_value(
    metadata: &ResponseMetadata,
    index: usize,
    name: &str,
    arguments: &str,
) -> Value {
    json!({
        "id": format!("call_{}_{}_{index}", opaque_suffix(&metadata.request_id), metadata.choice_index),
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        }
    })
}

fn usage_value(usage: ValidatedUsage) -> Value {
    json!({
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "prompt_tokens_details": {"cached_tokens": usage.cached_tokens.unwrap_or(0)},
    })
}

fn finish_reason(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
    }
}

fn stream_object(endpoint: Endpoint) -> &'static str {
    match endpoint {
        Endpoint::Completion => "text_completion",
        Endpoint::Chat => "chat.completion.chunk",
    }
}

fn json_response(status: StatusCode, body: Value, request_id: &str) -> Response {
    let mut response = (status, Json(body)).into_response();
    insert_request_id(&mut response, request_id);
    response
}

pub(crate) fn next_request_id() -> String {
    let sequence = REQUEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!("req_{:016x}_{sequence:016x}", unix_nanos())
}

fn response_id(endpoint: Endpoint, request_id: &str) -> String {
    let prefix = match endpoint {
        Endpoint::Completion => "cmpl",
        Endpoint::Chat => "chatcmpl",
    };
    format!("{prefix}_{}", opaque_suffix(request_id))
}

fn opaque_suffix(request_id: &str) -> &str {
    request_id.strip_prefix("req_").unwrap_or(request_id)
}

fn generated_seed() -> u64 {
    REQUEST_SEQUENCE
        .fetch_add(1, Ordering::Relaxed)
        .rotate_left(17)
        ^ u64::try_from(unix_nanos()).unwrap_or(u64::MAX)
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn unix_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A refused prepared item answers 400 with the backend's own message,
    /// not the tokenization one.
    #[test]
    fn rejected_admissions_answer_bad_request_with_their_message() {
        let message = "media item 1: geometry [2, 2, 2, 0] for encoder 0 is not one its \
                       media recipes produce";
        let error = admission_error(AdmissionError::Rejected(message.to_owned()));
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.message, message);
        assert_eq!(error.value()["error"]["message"], message);
        let tokenization = admission_error(AdmissionError::InvalidPrompt);
        assert_ne!(error.message, tokenization.message);
    }
}
