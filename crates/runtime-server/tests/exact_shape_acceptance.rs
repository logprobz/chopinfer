//! What the exact-shape acceptance profile admits, driven through the router.
//!
//! Every request here goes through the routes a bound deployment serves: the
//! router is built by `runtime_server::router`, handed to `axum::serve` over an
//! in-memory connection, and spoken to in HTTP. Nothing is bound and no port is
//! involved.
//!
//! The profile is one deployment's evidence expressed as a bound: a prompt that
//! encodes to exactly so many ids, answered with a generation length the
//! request states and that the deployment measured. A request outside that
//! shape is refused here rather than rendered, tokenized, or generated, because
//! dropping the requirement instead of enforcing it would expose exactly the
//! behaviour nothing validated.

use std::any::Any;
use std::future::{pending, poll_fn};
use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use axum::serve::Listener;
use runtime_server::{
    AdmissionError, BackendReadiness, BoxFuture, ChatStateV1, EventSender, FinishReason,
    FrontendChatRequestV1, FrontendErrorV1, FrontendModelV1, FrontendRenderedChatV1,
    FrontendRoleV1, GenerationBackend, GenerationEvent, PreparedPrompt, PreparedRequest,
    RequestAcceptanceProfile, RequestSurfaceV1, ServerCapacityProfile, ServerConfig,
    ServerConfigError, ServerFrontendV1, TokenUsage, VisibleContent, router,
};
use serde_json::Value;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tokio::time::timeout;

const MODEL: &str = "fixture.model.v1";
/// The prompt length this deployment says it measured, in token ids.
const PROMPT_TOKENS: usize = 32;
/// The longest answer it says it measured, in generated tokens.
const GENERATED_TOKENS: usize = 2;
/// Prompt plus generated tokens, which is what a request is charged.
const TOTAL_TOKENS: usize = PROMPT_TOKENS + GENERATED_TOKENS;
/// What the frontend declares, which is wider than what this deployment serves.
const DECLARED_TOTAL_TOKENS: usize = 256;
/// What the fixture's rendered chat prompt costs before its content.
const CHAT_PROMPT_PREFIX: &str = "user:";

/// The one deployment under test: one measured shape and nothing wider.
fn exact_shape_config() -> ServerConfig {
    ServerConfig::default()
        .with_exact_shape_acceptance_profile(PROMPT_TOKENS, GENERATED_TOKENS)
        .expect("the measured shape is a usable profile")
}

/// A prompt whose fake tokenization is exactly this many ids.
fn prompt_of(tokens: usize) -> String {
    "a".repeat(tokens)
}

/// A chat message whose rendered prompt tokenizes to exactly this many ids.
fn chat_content_of(tokens: usize) -> String {
    prompt_of(tokens - CHAT_PROMPT_PREFIX.len())
}

fn completion(prompt: &str, tail: &str) -> String {
    format!(r#"{{"model":"{MODEL}","prompt":"{prompt}"{tail}}}"#)
}

fn chat(content: &str, tail: &str) -> String {
    format!(r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"{content}"}}]{tail}}}"#)
}

/// A chat state whose only fact is whether reasoning was asked for.
#[derive(Clone, Copy, Debug)]
struct FakeChatStateV1;

impl ChatStateV1 for FakeChatStateV1 {
    fn reasoning_enabled(&self) -> bool {
        false
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A frontend for a model that does not exist, which renders anything.
///
/// It refuses nothing, so every refusal these tests see came from the
/// transport's acceptance profile rather than from a model that disagreed.
#[derive(Debug)]
struct PermissiveFrontendV1 {
    models: Vec<FrontendModelV1>,
    renders: AtomicUsize,
}

impl PermissiveFrontendV1 {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            models: vec![FrontendModelV1::new(MODEL, "fixture-owner")],
            renders: AtomicUsize::new(0),
        })
    }

    fn renders(&self) -> usize {
        self.renders.load(Ordering::Relaxed)
    }
}

impl ServerFrontendV1 for PermissiveFrontendV1 {
    fn models(&self) -> &[FrontendModelV1] {
        &self.models
    }

    fn maximum_total_tokens(&self) -> usize {
        DECLARED_TOTAL_TOKENS
    }

    fn maximum_tool_calls(&self) -> usize {
        2
    }

    fn render_chat(
        &self,
        request: FrontendChatRequestV1,
    ) -> Result<FrontendRenderedChatV1, FrontendErrorV1> {
        self.renders.fetch_add(1, Ordering::Relaxed);
        let prompt = request
            .messages
            .iter()
            .map(|message| {
                format!(
                    "{}:{}",
                    match &message.role {
                        FrontendRoleV1::System => "system",
                        FrontendRoleV1::Developer => "developer",
                        FrontendRoleV1::User => "user",
                        FrontendRoleV1::Assistant => "assistant",
                        FrontendRoleV1::Tool => "tool",
                        FrontendRoleV1::Declared(name) => name.as_str(),
                    },
                    message.content.as_deref().unwrap_or_default()
                )
            })
            .collect::<Vec<_>>()
            .join("|");
        Ok(FrontendRenderedChatV1 {
            prompt,
            segments: Vec::new(),
            media: Vec::new(),
            allowed_function_names: Vec::new(),
            state: Arc::new(FakeChatStateV1),
        })
    }

    fn validate_tool_call(&self, _name: &str, _arguments: &str) -> Result<(), FrontendErrorV1> {
        Ok(())
    }
}

/// A backend that tokenizes one character to one token and answers in order.
///
/// The generated tokens are fixed, so two identical requests answer with
/// identical content, and the answer is as long as the request asked for.
struct ScriptedBackendV1 {
    submissions: Mutex<Vec<PreparedRequest>>,
    tokenizations: AtomicUsize,
}

impl ScriptedBackendV1 {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            submissions: Mutex::new(Vec::new()),
            tokenizations: AtomicUsize::new(0),
        })
    }

    fn submissions(&self) -> Vec<PreparedRequest> {
        self.submissions
            .lock()
            .expect("the submission log is usable")
            .clone()
    }

    fn submission_count(&self) -> usize {
        self.submissions
            .lock()
            .expect("the submission log is usable")
            .len()
    }

    fn tokenizations(&self) -> usize {
        self.tokenizations.load(Ordering::Relaxed)
    }
}

impl GenerationBackend for ScriptedBackendV1 {
    fn readiness(&self) -> BackendReadiness {
        BackendReadiness::Ready
    }

    fn maximum_sequence_tokens(&self) -> usize {
        DECLARED_TOTAL_TOKENS
    }

    fn prompt_token_count(&self, prompt: &PreparedPrompt) -> Result<usize, AdmissionError> {
        self.tokenizations.fetch_add(1, Ordering::Relaxed);
        Ok(match prompt {
            PreparedPrompt::Raw(prompt) | PreparedPrompt::RenderedChat(prompt) => {
                if prompt == "zero-token-fixture" {
                    0
                } else {
                    prompt.chars().count()
                }
            }
            PreparedPrompt::TokenIds(ids) => ids.len(),
            PreparedPrompt::MediaChat { .. } | PreparedPrompt::MediaTokens { .. } => {
                return Err(AdmissionError::InvalidPrompt);
            }
        })
    }

    fn submit(
        &self,
        request: PreparedRequest,
        events: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        let generated = request.max_new_tokens;
        self.submissions
            .lock()
            .expect("the submission log is usable")
            .push(request);
        Box::pin(async move {
            tokio::spawn(async move {
                for index in 0..generated {
                    let event = GenerationEvent::Content(VisibleContent {
                        delta: format!("t{index}"),
                        generated_token_count: 1,
                    });
                    if events.emit(event).await.is_err() {
                        return;
                    }
                }
                let _ = events
                    .emit(GenerationEvent::Finished {
                        reason: FinishReason::Length,
                        usage: TokenUsage {
                            prompt_tokens: PROMPT_TOKENS as u64,
                            completion_tokens: generated as u64,
                            cached_tokens: None,
                        },
                    })
                    .await;
            });
            Ok(())
        })
    }
}

/// One direction of one in-memory connection.
#[derive(Clone, Default)]
struct ChannelV1(Arc<Mutex<ChannelStateV1>>);

#[derive(Default)]
struct ChannelStateV1 {
    bytes: Vec<u8>,
    closed: bool,
    reader: Option<Waker>,
}

impl ChannelV1 {
    fn state(&self) -> std::sync::MutexGuard<'_, ChannelStateV1> {
        self.0.lock().expect("the connection is usable")
    }

    fn write(&self, bytes: &[u8]) {
        let mut state = self.state();
        state.bytes.extend_from_slice(bytes);
        if let Some(reader) = state.reader.take() {
            reader.wake();
        }
    }

    fn close(&self) {
        let mut state = self.state();
        state.closed = true;
        if let Some(reader) = state.reader.take() {
            reader.wake();
        }
    }

    fn poll_read(&self, context: &Context<'_>, buffer: &mut ReadBuf<'_>) -> Poll<io::Result<()>> {
        let mut state = self.state();
        if state.bytes.is_empty() {
            if state.closed {
                return Poll::Ready(Ok(()));
            }
            state.reader = Some(context.waker().clone());
            return Poll::Pending;
        }
        let taken = state.bytes.len().min(buffer.remaining());
        buffer.put_slice(&state.bytes[..taken]);
        state.bytes.drain(..taken);
        Poll::Ready(Ok(()))
    }

    /// Everything written to this direction, once the writer has finished.
    fn poll_whole(&self, context: &Context<'_>) -> Poll<Vec<u8>> {
        let mut state = self.state();
        if state.closed {
            return Poll::Ready(std::mem::take(&mut state.bytes));
        }
        state.reader = Some(context.waker().clone());
        Poll::Pending
    }
}

/// The served end of one in-memory connection.
struct ConnectionV1 {
    incoming: ChannelV1,
    outgoing: ChannelV1,
}

impl AsyncRead for ConnectionV1 {
    fn poll_read(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        self.incoming.poll_read(context, buffer)
    }
}

impl AsyncWrite for ConnectionV1 {
    fn poll_write(
        self: Pin<&mut Self>,
        _context: &mut Context<'_>,
        bytes: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.outgoing.write(bytes);
        Poll::Ready(Ok(bytes.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.outgoing.close();
        Poll::Ready(Ok(()))
    }
}

impl Drop for ConnectionV1 {
    fn drop(&mut self) {
        self.outgoing.close();
    }
}

/// One accepted connection, handed over in memory instead of on a port.
struct InMemoryListenerV1 {
    connections: UnboundedReceiver<ConnectionV1>,
}

impl Listener for InMemoryListenerV1 {
    type Io = ConnectionV1;
    type Addr = &'static str;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        match self.connections.recv().await {
            Some(connection) => (connection, "in-process"),
            None => pending().await,
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        Ok("in-process")
    }
}

/// The transport under test, reached over in-memory connections.
struct InProcessServerV1 {
    connections: UnboundedSender<ConnectionV1>,
    backend: Arc<ScriptedBackendV1>,
    frontend: Arc<PermissiveFrontendV1>,
}

impl InProcessServerV1 {
    fn start(config: ServerConfig) -> Self {
        let backend = ScriptedBackendV1::new();
        let frontend = PermissiveFrontendV1::new();
        let router =
            router(backend.clone(), frontend.clone(), config).expect("the deployment is servable");
        let (connections, incoming) = unbounded_channel();
        tokio::spawn(async move {
            let _ = axum::serve(
                InMemoryListenerV1 {
                    connections: incoming,
                },
                router,
            )
            .await;
        });
        Self {
            connections,
            backend,
            frontend,
        }
    }

    /// A deployment that admits exactly the measured shape and nothing else.
    fn exact_shape() -> Self {
        Self::start(exact_shape_config())
    }

    /// The same deployment, with every widening a caller could name afterwards.
    fn widened_exact_shape() -> Self {
        Self::start(
            exact_shape_config()
                .with_request_surface(RequestSurfaceV1::Full)
                .with_maximum_context_tokens(4096)
                .with_capacity_profile(
                    ServerCapacityProfile::new("fixture-profile", 4, 4)
                        .expect("the fixture profile is usable"),
                )
                .with_one_token_acceptance_profile(1),
        )
    }

    async fn post(&self, path: &str, body: &str) -> ResponseV1 {
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: in-process\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let incoming = ChannelV1::default();
        let outgoing = ChannelV1::default();
        incoming.write(request.as_bytes());
        self.connections
            .send(ConnectionV1 {
                incoming,
                outgoing: outgoing.clone(),
            })
            .expect("the served router is still running");
        let raw = timeout(
            Duration::from_secs(5),
            poll_fn(|context| outgoing.poll_whole(context)),
        )
        .await
        .expect("the response arrives");
        ResponseV1::parse(&raw)
    }

    /// Sends one request and requires nothing downstream of the profile to run.
    async fn refuses(&self, path: &str, body: &str, parameter: &str) -> ResponseV1 {
        let submissions = self.backend.submission_count();

        let response = self.post(path, body).await;

        assert_eq!(response.status, 400, "{body} answered {}", response.body);
        let error = response.json();
        assert_eq!(error["error"]["param"], parameter, "{}", response.body);
        assert_eq!(error["error"]["type"], "BadRequestError");
        assert_eq!(
            self.backend.submission_count(),
            submissions,
            "{body} reached generation"
        );
        response
    }
}

struct ResponseV1 {
    status: u16,
    body: String,
}

impl ResponseV1 {
    fn parse(raw: &[u8]) -> Self {
        let text = String::from_utf8_lossy(raw).into_owned();
        let (head, body) = text.split_once("\r\n\r\n").unwrap_or((text.as_str(), ""));
        let status = head
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|code| code.parse().ok())
            .unwrap_or(0);
        Self {
            status,
            body: body.to_owned(),
        }
    }

    fn json(&self) -> Value {
        serde_json::from_str(&self.body).expect("the response body is JSON")
    }
}

/// The measured shape is served on both endpoints, and answers deterministically.
///
/// Two generated tokens are the point: a prompt is prefilled and then decoded
/// from, which is more than one step. The second request repeats the first
/// exactly, so what is admitted is a shape rather than one lucky request.
#[tokio::test]
async fn the_measured_shape_is_served_on_chat_and_completions() {
    let server = InProcessServerV1::exact_shape();

    let first = server
        .post(
            "/v1/completions",
            &completion(&prompt_of(PROMPT_TOKENS), r#","max_tokens":2"#),
        )
        .await;
    assert_eq!(first.status, 200, "{}", first.body);
    assert_eq!(first.json()["choices"][0]["text"], "t0t1");
    assert_eq!(first.json()["usage"]["completion_tokens"], 2);

    let second = server
        .post(
            "/v1/completions",
            &completion(&prompt_of(PROMPT_TOKENS), r#","max_tokens":2"#),
        )
        .await;
    assert_eq!(second.status, 200, "{}", second.body);
    assert_eq!(
        second.json()["choices"][0]["text"],
        first.json()["choices"][0]["text"]
    );

    let rendered = server
        .post(
            "/v1/chat/completions",
            &chat(&chat_content_of(PROMPT_TOKENS), r#","max_tokens":2"#),
        )
        .await;
    assert_eq!(rendered.status, 200, "{}", rendered.body);
    assert_eq!(rendered.json()["choices"][0]["message"]["content"], "t0t1");

    // What reached generation is the shape, in the same units the profile
    // states it in: the prompt the tokenizer produced and the length asked for.
    let submissions = server.backend.submissions();
    assert_eq!(submissions.len(), 3);
    for submission in &submissions {
        assert_eq!(submission.max_new_tokens, GENERATED_TOKENS);
    }
    assert_eq!(
        submissions[0].prompt,
        PreparedPrompt::Raw(prompt_of(PROMPT_TOKENS))
    );
    assert_eq!(
        submissions[2].prompt,
        PreparedPrompt::RenderedChat(format!(
            "{CHAT_PROMPT_PREFIX}{}",
            chat_content_of(PROMPT_TOKENS)
        ))
    );
}

/// A shorter answer is inside the bound, so it is served.
#[tokio::test]
async fn a_shorter_answer_than_the_bound_is_served() {
    let server = InProcessServerV1::exact_shape();

    let response = server
        .post(
            "/v1/completions",
            &completion(&prompt_of(PROMPT_TOKENS), r#","max_tokens":1"#),
        )
        .await;

    assert_eq!(response.status, 200, "{}", response.body);
    assert_eq!(response.json()["choices"][0]["text"], "t0");
    assert_eq!(response.json()["usage"]["completion_tokens"], 1);
    assert_eq!(server.backend.submissions()[0].max_new_tokens, 1);
}

/// A prompt of another length is refused on the ids it became.
///
/// Both refusals here would fit the token budget: 20 prompt tokens answered
/// with 14, and 33 answered with 1, are 34 tokens each. The budget is not what
/// this deployment measured; one exact prompt length answered with at most two
/// tokens is.
#[tokio::test]
async fn a_prompt_of_another_length_is_refused_after_tokenization() {
    let server = InProcessServerV1::exact_shape();

    for (prompt_tokens, max_tokens) in [(20, 14), (33, 1), (PROMPT_TOKENS - 1, 2)] {
        let body = completion(
            &prompt_of(prompt_tokens),
            &format!(r#","max_tokens":{max_tokens}"#),
        );
        // A generation bound outside the measured one is refused before the
        // prompt is ever tokenized, so only the prompt length is under test.
        if max_tokens > GENERATED_TOKENS {
            server.refuses("/v1/completions", &body, "max_tokens").await;
            continue;
        }
        let response = server.refuses("/v1/completions", &body, "prompt").await;
        assert!(
            response.json()["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("exactly 32 tokens")),
            "{}",
            response.body
        );
    }

    // The same holds for a rendered conversation, whose length is the
    // renderer's answer rather than the client's.
    let tail = server
        .refuses(
            "/v1/chat/completions",
            &chat(&chat_content_of(PROMPT_TOKENS + 3), r#","max_tokens":2"#),
            "prompt",
        )
        .await;
    assert_eq!(tail.status, 400);
}

/// A generation length nothing measured is refused before anything runs.
///
/// A missing length is refused rather than defaulted: the transport's own
/// default is a length this deployment has no evidence for, and it would be
/// charged against a budget sized for two tokens.
#[tokio::test]
async fn a_generation_length_outside_the_bound_is_refused_before_the_tokenizer() {
    let server = InProcessServerV1::exact_shape();
    let prompt = prompt_of(PROMPT_TOKENS);

    for (tail, parameter) in [
        ("", "max_tokens"),
        (r#","max_tokens":3"#, "max_tokens"),
        (r#","max_tokens":0"#, "max_tokens"),
        (r#","max_tokens":2,"stream":true"#, "stream"),
    ] {
        let renders = server.frontend.renders();
        let tokenizations = server.backend.tokenizations();

        server
            .refuses("/v1/completions", &completion(&prompt, tail), parameter)
            .await;

        assert_eq!(
            server.frontend.renders(),
            renders,
            "{tail} reached rendering"
        );
        assert_eq!(
            server.backend.tokenizations(),
            tokenizations,
            "{tail} reached the tokenizer"
        );
    }

    // A chat request states its length under either name, and both are bounded.
    let content = chat_content_of(PROMPT_TOKENS);
    for (tail, parameter) in [
        ("", "max_tokens"),
        (r#","max_completion_tokens":3"#, "max_completion_tokens"),
    ] {
        let renders = server.frontend.renders();
        server
            .refuses("/v1/chat/completions", &chat(&content, tail), parameter)
            .await;
        assert_eq!(
            server.frontend.renders(),
            renders,
            "{tail} reached rendering"
        );
    }
}

/// The profile carries the narrow conversation surface with it.
///
/// Tools, history, and multimodal content are refused because the surface is
/// part of the same deployment decision, not a separate switch a caller could
/// have left open.
#[tokio::test]
async fn the_profile_admits_only_plain_single_turn_conversations() {
    let server = InProcessServerV1::exact_shape();
    let content = chat_content_of(PROMPT_TOKENS);

    for (body, parameter) in [
        (
            format!(
                r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"{content}"}}],"max_tokens":2,"tools":[{{"type":"function","function":{{"name":"lookup","parameters":{{}}}}}}]}}"#
            ),
            "tools",
        ),
        (
            format!(
                r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"a"}},{{"role":"assistant","content":"b"}},{{"role":"user","content":"{content}"}}],"max_tokens":2}}"#
            ),
            "messages",
        ),
        (
            format!(
                r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":[{{"type":"text","text":"{content}"}}]}}],"max_tokens":2}}"#
            ),
            "messages",
        ),
        (
            format!(
                r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"{content}"}}],"max_tokens":2,"reasoning_effort":"high"}}"#
            ),
            "reasoning_effort",
        ),
    ] {
        let renders = server.frontend.renders();
        server
            .refuses("/v1/chat/completions", &body, parameter)
            .await;
        assert_eq!(
            server.frontend.renders(),
            renders,
            "{body} reached rendering"
        );
    }
}

/// The shape, the surface, and the budget are one configured decision.
#[test]
fn the_profile_sets_the_surface_and_the_budget_with_the_shape() {
    let configured = ServerConfig::default()
        .with_exact_shape_acceptance_profile(PROMPT_TOKENS, GENERATED_TOKENS)
        .expect("the measured shape is a usable profile");

    assert_eq!(
        configured.acceptance_profile(),
        RequestAcceptanceProfile::ExactShape {
            prompt_tokens: PROMPT_TOKENS,
            maximum_generated_tokens: GENERATED_TOKENS,
        }
    );
    assert_eq!(
        configured.request_surface(),
        RequestSurfaceV1::SingleTurnPlainChat
    );
    // The budget is the shape's own sum, in the unit both halves are counted in.
    assert_eq!(configured.maximum_context_tokens(), Some(TOTAL_TOKENS));
    assert!(!configured.acceptance_profile().streaming_allowed());
    assert_eq!(configured.acceptance_profile().profile_name(), None);

    assert_eq!(
        ServerConfig::default().with_exact_shape_acceptance_profile(0, GENERATED_TOKENS),
        Err(ServerConfigError::ZeroExactPromptTokens)
    );
    assert_eq!(
        ServerConfig::default().with_exact_shape_acceptance_profile(PROMPT_TOKENS, 0),
        Err(ServerConfigError::ZeroMaximumGeneratedTokens)
    );
    assert_eq!(
        ServerConfig::default().with_exact_shape_acceptance_profile(usize::MAX, 1),
        Err(ServerConfigError::ExactShapeBudgetOverflow {
            prompt_tokens: usize::MAX,
            maximum_generated_tokens: 1,
        })
    );
}

/// The deployments that were already served are not this one.
///
/// One-token acceptance is the narrower profile this one is modelled on, and it
/// keeps every property it had: one request in flight, completions only, and
/// the whole request surface unless a caller narrows it separately.
#[test]
fn one_token_acceptance_is_unchanged_beside_the_new_profile() {
    let one_token = ServerConfig::default().with_one_token_acceptance_profile(PROMPT_TOKENS);

    assert_eq!(
        one_token.acceptance_profile(),
        RequestAcceptanceProfile::OneToken {
            prompt_tokens: PROMPT_TOKENS
        }
    );
    assert_eq!(one_token.max_in_flight(), 1);
    assert_eq!(one_token.maximum_context_tokens(), None);
    assert_eq!(one_token.request_surface(), RequestSurfaceV1::Full);
    assert!(!one_token.acceptance_profile().streaming_allowed());

    // A deployment that names neither still serves what it always served.
    let general = ServerConfig::default();
    assert_eq!(
        general.acceptance_profile(),
        RequestAcceptanceProfile::General
    );
    assert_eq!(general.request_surface(), RequestSurfaceV1::Full);
    assert_eq!(general.maximum_context_tokens(), None);
    assert!(general.acceptance_profile().streaming_allowed());
}

/// A declared shape is not undone by anything named after it.
///
/// Every widening a caller could reach for is applied here, in the order that
/// would undo the coupling if it could: a full surface, a longer context, a
/// capacity profile that replaces the acceptance profile, and a different
/// acceptance profile outright. What the deployment serves is unchanged.
#[test]
fn a_declared_shape_survives_every_later_setter() {
    let widened = exact_shape_config()
        .with_request_surface(RequestSurfaceV1::Full)
        .with_maximum_context_tokens(4096)
        .with_capacity_profile(
            ServerCapacityProfile::new("fixture-profile", 4, 4).expect("the profile is usable"),
        )
        .with_one_token_acceptance_profile(1);

    assert_eq!(
        widened.acceptance_profile(),
        RequestAcceptanceProfile::ExactShape {
            prompt_tokens: PROMPT_TOKENS,
            maximum_generated_tokens: GENERATED_TOKENS,
        }
    );
    assert_eq!(
        widened.request_surface(),
        RequestSurfaceV1::SingleTurnPlainChat
    );
    assert_eq!(widened.maximum_context_tokens(), Some(TOTAL_TOKENS));
    // The queue bounds a capacity profile carries are not a request shape, so
    // they still apply.
    assert_eq!(widened.max_in_flight(), 1);
    assert_eq!(widened.event_buffer_capacity(), 4);

    // Naming the widenings first changes nothing either.
    let reordered = ServerConfig::default()
        .with_request_surface(RequestSurfaceV1::Full)
        .with_maximum_context_tokens(4096)
        .with_exact_shape_acceptance_profile(PROMPT_TOKENS, GENERATED_TOKENS)
        .expect("the measured shape is a usable profile");
    assert_eq!(reordered.acceptance_profile(), widened.acceptance_profile());
    assert_eq!(reordered.request_surface(), widened.request_surface());
    assert_eq!(
        reordered.maximum_context_tokens(),
        widened.maximum_context_tokens()
    );

    // Serving less than was measured is still allowed, because it is narrowing.
    assert_eq!(
        exact_shape_config()
            .with_maximum_context_tokens(20)
            .maximum_context_tokens(),
        Some(20)
    );
}

/// A widened configuration still refuses everything the shape refuses.
///
/// This is the same deployment as `the_profile_admits_only_plain_single_turn_conversations`,
/// built through every setter that would have undone the coupling, and driven
/// over the real router. Nothing reaches the renderer or generation.
#[tokio::test]
async fn a_widened_configuration_still_refuses_what_the_shape_refuses() {
    let server = InProcessServerV1::widened_exact_shape();
    let content = chat_content_of(PROMPT_TOKENS);

    for (body, parameter) in [
        (
            format!(
                r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"{content}"}}],"max_tokens":2,"tools":[{{"type":"function","function":{{"name":"lookup","parameters":{{}}}}}}]}}"#
            ),
            "tools",
        ),
        (
            format!(
                r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"a"}},{{"role":"assistant","content":"b"}},{{"role":"user","content":"{content}"}}],"max_tokens":2}}"#
            ),
            "messages",
        ),
        (
            format!(
                r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":[{{"type":"text","text":"{content}"}}]}}],"max_tokens":2}}"#
            ),
            "messages",
        ),
        (
            format!(
                r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"{content}"}}],"max_tokens":2,"reasoning_effort":"high"}}"#
            ),
            "reasoning_effort",
        ),
    ] {
        let renders = server.frontend.renders();
        let tokenizations = server.backend.tokenizations();

        server
            .refuses("/v1/chat/completions", &body, parameter)
            .await;

        assert_eq!(
            server.frontend.renders(),
            renders,
            "{body} reached rendering"
        );
        assert_eq!(
            server.backend.tokenizations(),
            tokenizations,
            "{body} reached the tokenizer"
        );
    }

    // The generation bound is the shape's, not the one-token profile's and not
    // the 4096-token context named beside it.
    for tail in ["", r#","max_tokens":3"#] {
        server
            .refuses("/v1/chat/completions", &chat(&content, tail), "max_tokens")
            .await;
    }
    assert_eq!(server.backend.submissions().len(), 0);

    // And the measured shape is still served.
    let served = server
        .post(
            "/v1/chat/completions",
            &chat(&content, r#","max_tokens":2"#),
        )
        .await;
    assert_eq!(served.status, 200, "{}", served.body);
    assert_eq!(served.json()["choices"][0]["message"]["content"], "t0t1");
    assert_eq!(
        server.backend.submissions()[0].max_new_tokens,
        GENERATED_TOKENS
    );
}

fn bounded_config() -> ServerConfig {
    ServerConfig::default()
        .with_bounded_acceptance_profile(255, 64, 256)
        .unwrap()
}

#[tokio::test]
async fn bounded_router_admits_variable_budgets_and_refuses_before_enqueue() {
    let server = InProcessServerV1::start(bounded_config());
    for (prompt, output) in [(1, 1), (32, 64), (192, 64), (255, 1)] {
        let body = completion(&prompt_of(prompt), &format!(r#","max_tokens":{output}"#));
        let response = server.post("/v1/completions", &body).await;
        assert_eq!(response.status, 200, "{}", response.body);
        assert_eq!(
            server.backend.submissions().last().unwrap().max_new_tokens,
            output
        );
        assert_eq!(response.json()["usage"]["completion_tokens"], output);
    }
    for (prompt, output) in [(0, 1), (1, 0), (1, 65), (193, 64), (255, 2), (256, 1)] {
        let count = server.backend.submission_count();
        let body = completion(&prompt_of(prompt), &format!(r#","max_tokens":{output}"#));
        let response = server.post("/v1/completions", &body).await;
        assert_eq!(response.status, 400, "{}", response.body);
        assert_eq!(server.backend.submission_count(), count);
    }
    for (prompt, output, status) in [
        (32, 64, 200),
        (192, 64, 200),
        (255, 1, 200),
        (193, 64, 400),
        (255, 2, 400),
        (256, 1, 400),
    ] {
        let count = server.backend.submission_count();
        let body = chat(
            &chat_content_of(prompt),
            &format!(r#","max_completion_tokens":{output}"#),
        );
        let response = server.post("/v1/chat/completions", &body).await;
        assert_eq!(response.status, status, "{}", response.body);
        assert_eq!(
            server.backend.submission_count(),
            count + usize::from(status == 200)
        );
    }
}

#[test]
fn bounded_profile_preserves_its_limits_across_setters() {
    let config = bounded_config()
        .with_maximum_context_tokens(4096)
        .with_request_surface(RequestSurfaceV1::Full)
        .with_one_token_acceptance_profile(1);
    assert_eq!(config.maximum_context_tokens(), Some(256));
    assert_eq!(
        config.request_surface(),
        RequestSurfaceV1::SingleTurnPlainChat
    );
    assert_eq!(
        config.acceptance_profile(),
        RequestAcceptanceProfile::Bounded {
            maximum_prompt_tokens: 255,
            maximum_generated_tokens: 64,
            maximum_total_tokens: 256,
        }
    );
    assert!(!config.acceptance_profile().streaming_allowed());
    for (prompt, output, total) in [
        (0, 64, 256),
        (255, 0, 256),
        (255, 64, 0),
        (256, 64, 256),
        (1, 256, 256),
    ] {
        assert_eq!(
            ServerConfig::default().with_bounded_acceptance_profile(prompt, output, total),
            Err(ServerConfigError::InvalidBoundedLimits)
        );
    }
}

#[tokio::test]
async fn bounded_profile_refuses_zero_tokenization_and_unsupported_surfaces() {
    let server = InProcessServerV1::start(
        bounded_config()
            .with_request_surface(RequestSurfaceV1::Full)
            .with_maximum_context_tokens(4096)
            .with_capacity_profile(ServerCapacityProfile::new("fixture", 4, 4).unwrap())
            .with_one_token_acceptance_profile(1),
    );
    server
        .refuses(
            "/v1/completions",
            &completion("zero-token-fixture", r#","max_tokens":1"#),
            "prompt",
        )
        .await;
    assert_eq!(server.backend.tokenizations(), 1);
    for (body, param) in [
        (
            chat(
                "hello",
                r#","max_tokens":1,"tools":[{"type":"function","function":{"name":"lookup","parameters":{}}}]"#,
            ),
            "tools",
        ),
        (chat("hello", r#","max_tokens":1,"stream":true"#), "stream"),
        (chat("hello", ""), "max_tokens"),
        (
            format!(
                r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"a"}},{{"role":"assistant","content":"b"}},{{"role":"user","content":"c"}}],"max_tokens":1}}"#
            ),
            "messages",
        ),
        (
            format!(
                r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":[{{"type":"image_url","image_url":{{"url":"https://example.test/image"}}}}]}}],"max_tokens":1}}"#
            ),
            "messages",
        ),
    ] {
        let renders = server.frontend.renders();
        let tokenizations = server.backend.tokenizations();
        server.refuses("/v1/chat/completions", &body, param).await;
        assert_eq!(server.frontend.renders(), renders);
        assert_eq!(server.backend.tokenizations(), tokenizations);
    }
}
