//! What each request surface admits, driven through the real router.
//!
//! Every request here goes through the same routes a bound deployment serves:
//! the router is built by `runtime_server::router`, handed to `axum::serve`
//! over an in-memory connection, and spoken to in HTTP. Nothing is bound, and
//! no port is involved, so these tests say nothing about a listener; they say
//! what the transport admits.
//!
//! The frontend and the backend are fakes. A refusal that never reaches either
//! of them is the point of the narrow surface: a deployment that has measured
//! only plain single-turn chat must not render, tokenize, or generate anything
//! else, and dropping the requirement instead of enforcing it would expose
//! exactly the behaviour nothing validated.

use std::any::Any;
use std::future::{pending, poll_fn};
use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use axum::serve::Listener;
use runtime_server::{
    AdmissionError, BackendReadiness, BoxFuture, CancellationSignal, ChatStateV1, EventSender,
    FinishReason, FrontendChatRequestV1, FrontendContentPartV1, FrontendErrorV1, FrontendModelV1,
    FrontendRenderedChatV1, FrontendRoleV1, FrontendToolChoiceV1, GeneratedTokenLogprob,
    GenerationBackend, GenerationEvent, PreparedPrompt, PreparedRequest, PromptSegmentV1,
    PromptTokenLogprob, RequestSurfaceV1, SINGLE_TURN_PLAIN_CHAT_SURFACE_NAME_V1,
    ServerCapacityProfile, ServerConfig, ServerFrontendV1, TokenUsage, VisibleContent,
};
use serde_json::Value;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tokio::time::timeout;

const MODEL: &str = "fixture.model.v1";
/// Wide enough that the default completion bound fits, so a refusal in these
/// tests is the surface's rather than the context limit's.
const MAXIMUM_TOTAL_TOKENS: usize = 8192;
const PROMPT_TOKENS: usize = 4;

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
/// transport's surface rather than from a model that happened to disagree.
#[derive(Debug)]
struct PermissiveFrontendV1 {
    models: Vec<FrontendModelV1>,
    renders: AtomicUsize,
    requests: Mutex<Vec<FrontendChatRequestV1>>,
    reasoning_field: &'static str,
    media_enabled: AtomicBool,
    selected_modalities: AtomicBool,
    remote_media_enabled: AtomicBool,
    /// The media kinds served, when not every kind media inputs declare.
    served_media: Mutex<Option<Vec<runtime_server::FrontendMediaPartKindV1>>>,
    local_media_root: Option<std::path::PathBuf>,
    /// While set, every tool choice with tools constrains to one grammar.
    constrains_tools: AtomicBool,
    /// While set, the frontend declares a vendor API's request mapping: the
    /// thinking object, strict-only constraints and forced choices without
    /// tools served as auto.
    vendor_mapping: AtomicBool,
    /// While set, the frontend declares GIF and WebP images, `file` parts
    /// and images in tool results.
    declared_images: AtomicBool,
}

impl PermissiveFrontendV1 {
    fn new_with_reasoning_field(reasoning_field: &'static str) -> Arc<Self> {
        Arc::new(Self {
            models: vec![FrontendModelV1::new(MODEL, "fixture-owner")],
            renders: AtomicUsize::new(0),
            requests: Mutex::new(Vec::new()),
            reasoning_field,
            media_enabled: AtomicBool::new(false),
            selected_modalities: AtomicBool::new(false),
            remote_media_enabled: AtomicBool::new(false),
            served_media: Mutex::new(None),
            local_media_root: Some(std::env::temp_dir().join("request-surface-local-media")),
            constrains_tools: AtomicBool::new(false),
            vendor_mapping: AtomicBool::new(false),
            declared_images: AtomicBool::new(false),
        })
    }

    fn renders(&self) -> usize {
        self.renders.load(Ordering::Relaxed)
    }
}

impl ServerFrontendV1 for PermissiveFrontendV1 {
    fn sampling_defaults(&self) -> runtime_server::SamplingDefaults {
        runtime_server::SamplingDefaults {
            top_p: Some(0.95),
            ..Default::default()
        }
    }

    fn allowed_local_media_path(&self) -> Option<&std::path::Path> {
        self.local_media_root.as_deref()
    }

    fn allows_remote_media(&self) -> bool {
        self.remote_media_enabled.load(Ordering::Relaxed)
    }

    fn supports_media(&self) -> bool {
        self.media_enabled.load(Ordering::Relaxed)
    }

    fn has_selected_modalities(&self) -> bool {
        self.selected_modalities.load(Ordering::Relaxed)
    }

    fn tool_result_images(&self) -> bool {
        self.declared_images.load(Ordering::Relaxed)
    }

    fn image_formats(&self) -> Vec<String> {
        let formats: &[&str] = if self.declared_images.load(Ordering::Relaxed) {
            &["png", "jpeg", "gif", "webp"]
        } else {
            &["png", "jpeg"]
        };
        formats.iter().map(|&format| format.to_owned()).collect()
    }

    fn image_file_parts(&self) -> bool {
        self.declared_images.load(Ordering::Relaxed)
    }

    fn served_media(&self) -> Vec<runtime_server::FrontendMediaPartKindV1> {
        match &*self.served_media.lock().unwrap() {
            Some(kinds) if self.supports_media() => kinds.clone(),
            _ if self.supports_media() => runtime_server::FrontendMediaPartKindV1::ALL.to_vec(),
            _ => Vec::new(),
        }
    }

    fn reasoning_field(&self) -> &'static str {
        self.reasoning_field
    }

    fn reads_thinking_object(&self) -> bool {
        self.vendor_mapping.load(Ordering::Relaxed)
    }

    fn constrains_strict_only(&self) -> bool {
        self.vendor_mapping.load(Ordering::Relaxed)
    }

    fn forced_choice_without_tools_is_auto(&self) -> bool {
        self.vendor_mapping.load(Ordering::Relaxed)
    }

    fn tool_constraint(
        &self,
        tools: &[runtime_server::FrontendToolV1],
        choice: &FrontendToolChoiceV1,
    ) -> Result<Option<runtime_grammar::Grammar>, FrontendErrorV1> {
        if !self.constrains_tools.load(Ordering::Relaxed) {
            return match choice {
                FrontendToolChoiceV1::Auto | FrontendToolChoiceV1::None => Ok(None),
                _ => Err(FrontendErrorV1::new(
                    "forced tool choices are not supported by this model's template",
                    Some("tool_choice"),
                )),
            };
        }
        assert!(!tools.is_empty());
        Ok(Some(tool_grammar()))
    }

    fn models(&self) -> &[FrontendModelV1] {
        &self.models
    }

    fn maximum_total_tokens(&self) -> usize {
        MAXIMUM_TOTAL_TOKENS
    }

    fn maximum_tool_calls(&self) -> usize {
        2
    }

    fn render_chat(
        &self,
        request: FrontendChatRequestV1,
    ) -> Result<FrontendRenderedChatV1, FrontendErrorV1> {
        self.renders.fetch_add(1, Ordering::Relaxed);
        self.requests
            .lock()
            .expect("request log is usable")
            .push(request.clone());
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
        let allowed_function_names = match request.tool_choice {
            FrontendToolChoiceV1::None => Vec::new(),
            _ => request.tools.iter().map(|tool| tool.name.clone()).collect(),
        };
        let mut media = Vec::new();
        let mut segments = Vec::new();
        for message in &request.messages {
            if let Some(parts) = &message.parts {
                for part in parts {
                    match part {
                        FrontendContentPartV1::Text(text) => {
                            segments.push(PromptSegmentV1::Text(text.clone()))
                        }
                        FrontendContentPartV1::Media(item) => {
                            segments.push(PromptSegmentV1::Media(media.len()));
                            media.push(item.clone());
                        }
                        FrontendContentPartV1::ToolReference { name } => {
                            segments.push(PromptSegmentV1::Text(format!("ref:{name}")))
                        }
                    }
                }
            }
        }
        Ok(FrontendRenderedChatV1 {
            prompt,
            segments,
            media,
            allowed_function_names,
            state: Arc::new(FakeChatStateV1),
        })
    }

    fn validate_tool_call(&self, _name: &str, _arguments: &str) -> Result<(), FrontendErrorV1> {
        Ok(())
    }
}

/// A backend that counts every time the transport reached it.
struct CountingBackendV1 {
    prompts: std::sync::Mutex<Vec<PreparedRequest>>,
    tokenizations: AtomicUsize,
    /// While set, admitted requests generate nothing and never finish.
    hold: AtomicBool,
    held: std::sync::Mutex<Vec<EventSender>>,
    /// The next request streams this many 1 KiB content events first.
    flood: AtomicUsize,
    /// While set, choice `i` reports `1 + i % 2` cached prompt tokens.
    report_cached: AtomicBool,
    replay: Mutex<Option<Vec<GenerationEvent>>>,
    prompt_count: AtomicUsize,
    tokenize_prompts: AtomicBool,
    /// While set, a prompt over the caller's ceiling is refused before it
    /// is prepared, as a backend that counts media from headers does.
    refuse_early: AtomicBool,
}

impl CountingBackendV1 {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            prompts: std::sync::Mutex::new(Vec::new()),
            tokenizations: AtomicUsize::new(0),
            hold: AtomicBool::new(false),
            held: std::sync::Mutex::new(Vec::new()),
            flood: AtomicUsize::new(0),
            report_cached: AtomicBool::new(false),
            replay: Mutex::new(None),
            prompt_count: AtomicUsize::new(PROMPT_TOKENS),
            tokenize_prompts: AtomicBool::new(false),
            refuse_early: AtomicBool::new(false),
        })
    }

    fn cache_salts(&self) -> Vec<Option<String>> {
        self.prompts
            .lock()
            .expect("the prompt log is usable")
            .iter()
            .map(|request| request.cache_salt.clone())
            .collect()
    }

    fn prompts(&self) -> Vec<PreparedPrompt> {
        self.prompts
            .lock()
            .expect("the prompt log is usable")
            .iter()
            .map(|request| request.prompt.clone())
            .collect()
    }

    fn submissions(&self) -> usize {
        self.prompts.lock().expect("the prompt log is usable").len()
    }

    fn tokenizations(&self) -> usize {
        self.tokenizations.load(Ordering::Relaxed)
    }
}

impl GenerationBackend for CountingBackendV1 {
    fn readiness(&self) -> BackendReadiness {
        BackendReadiness::Ready
    }

    fn maximum_sequence_tokens(&self) -> usize {
        MAXIMUM_TOTAL_TOKENS
    }

    fn prompt_token_count(&self, _prompt: &PreparedPrompt) -> Result<usize, AdmissionError> {
        self.tokenizations.fetch_add(1, Ordering::Relaxed);
        Ok(self.prompt_count.load(Ordering::Relaxed))
    }

    fn prepare_prompt_within(
        &self,
        prompt: PreparedPrompt,
        cancellation: &CancellationSignal,
        maximum_prompt_tokens: Option<usize>,
    ) -> Result<(PreparedPrompt, usize), AdmissionError> {
        let prompt_tokens = self.prompt_count.load(Ordering::Relaxed);
        if self.refuse_early.load(Ordering::Relaxed)
            && maximum_prompt_tokens.is_some_and(|ceiling| prompt_tokens > ceiling)
        {
            return Err(AdmissionError::PromptTooLong { prompt_tokens });
        }
        self.prepare_prompt_cancellable(prompt, cancellation)
    }

    fn prepare_prompt(
        &self,
        prompt: PreparedPrompt,
    ) -> Result<(PreparedPrompt, usize), AdmissionError> {
        if !self.tokenize_prompts.load(Ordering::Relaxed) {
            let count = self.prompt_token_count(&prompt)?;
            return Ok((prompt, count));
        }
        self.tokenizations.fetch_add(1, Ordering::Relaxed);
        let tokens = match prompt {
            PreparedPrompt::Raw(text) | PreparedPrompt::RenderedChat(text) => {
                self.tokenize(&text, false)?
            }
            PreparedPrompt::TokenIds(tokens) => tokens,
            media => return Ok((media, PROMPT_TOKENS)),
        };
        let count = tokens.len();
        Ok((PreparedPrompt::TokenIds(tokens), count))
    }

    fn tokenize(&self, text: &str, _special: bool) -> Result<Vec<u32>, AdmissionError> {
        Ok(text.bytes().map(u32::from).collect())
    }

    fn token_strings(&self, tokens: &[u32]) -> Result<Vec<String>, AdmissionError> {
        tokens
            .iter()
            .map(|id| match id {
                32 => Ok("Ġ".to_owned()),
                10 => Ok("Ċ".to_owned()),
                _ => self.detokenize(&[*id]),
            })
            .collect()
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, AdmissionError> {
        let bytes = tokens
            .iter()
            .map(|token| u8::try_from(*token).map_err(|_| AdmissionError::InvalidPrompt))
            .collect::<Result<Vec<_>, _>>()?;
        String::from_utf8(bytes).map_err(|_| AdmissionError::InvalidPrompt)
    }

    fn submit(
        &self,
        request: PreparedRequest,
        events: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        let prompt_tokens = if self.tokenize_prompts.load(Ordering::Relaxed) {
            match &request.prompt {
                PreparedPrompt::TokenIds(tokens) => tokens.len(),
                _ => PROMPT_TOKENS,
            }
        } else {
            PROMPT_TOKENS
        };
        let cached_tokens = self
            .report_cached
            .load(Ordering::Relaxed)
            .then(|| request.request_id.rsplit('_').next()?.parse::<u64>().ok())
            .flatten()
            .map(|index| 1 + index % 2);
        let prompt_top = request.logprobs.prompt_top;
        let return_token_ids = request.logprobs.return_token_ids;
        let logprobs = request.logprobs.enabled;
        self.prompts
            .lock()
            .expect("the prompt log is usable")
            .push(request);
        if let Some(scripted) = self.replay.lock().unwrap().take() {
            return Box::pin(async move {
                tokio::spawn(async move {
                    for event in scripted {
                        if events.emit(event).await.is_err() {
                            return;
                        }
                    }
                });
                Ok(())
            });
        }
        if self.hold.load(Ordering::Relaxed) {
            self.held
                .lock()
                .expect("the held log is usable")
                .push(events);
            return Box::pin(async { Ok(()) });
        }
        let flood = self.flood.swap(0, Ordering::Relaxed);
        Box::pin(async move {
            tokio::spawn(async move {
                let mut scripted = Vec::new();
                if prompt_top.is_some() || return_token_ids {
                    scripted.push(GenerationEvent::Prompt {
                        token_ids: vec![7, 8, 9, 10],
                        text: "prompt".to_owned(),
                        tokens: vec!["a".to_owned(); PROMPT_TOKENS],
                    });
                }
                if prompt_top.is_some() {
                    scripted.push(GenerationEvent::PromptLogprobs(
                        (1..PROMPT_TOKENS)
                            .map(|position| PromptTokenLogprob {
                                position,
                                token_id: 7 + position as u32,
                                token: "a".to_owned(),
                                logprob: -0.2,
                                top_logprobs: Vec::new(),
                            })
                            .collect(),
                    ));
                }
                if logprobs {
                    scripted.push(GenerationEvent::GeneratedTokenLogprobs(vec![
                        GeneratedTokenLogprob {
                            token_id: 11,
                            token: "hi".to_owned(),
                            bytes: b"hi".to_vec(),
                            logprob: -0.1,
                            top_logprobs: Vec::new(),
                        },
                    ]));
                }
                if return_token_ids {
                    scripted.push(GenerationEvent::GeneratedTokenIds(vec![11]));
                }
                scripted.extend((0..flood).map(|_| {
                    GenerationEvent::Content(VisibleContent {
                        delta: "x".repeat(1024),
                        generated_token_count: 1,
                    })
                }));
                scripted.extend([
                    GenerationEvent::Content(VisibleContent {
                        delta: "hi".to_owned(),
                        generated_token_count: 1,
                    }),
                    GenerationEvent::Finished {
                        reason: FinishReason::Stop,
                        usage: TokenUsage {
                            prompt_tokens: prompt_tokens as u64,
                            completion_tokens: flood as u64 + 1,
                            cached_tokens,
                        },
                    },
                ]);
                for event in scripted {
                    if events.emit(event).await.is_err() {
                        return;
                    }
                }
            });
            Ok(())
        })
    }
}

/// One direction of one in-memory connection.
///
/// The reader is whoever holds the other end: the served router reads what a
/// test wrote, and a test reads what the router answered. There is exactly one
/// of each, so one parked waker is the whole of the wakeup accounting.
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
                // An empty read is end of stream, which is what a closed
                // connection is.
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
///
/// It is an ordinary asynchronous stream as far as the server is concerned, so
/// the router, its codec, and its streaming responses all run exactly as they
/// do over a socket. Nothing is bound and no address exists.
struct ConnectionV1 {
    incoming: ChannelV1,
    outgoing: ChannelV1,
    /// A client that has stopped reading: nothing written is ever accepted.
    stalled: bool,
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
        if self.stalled {
            return Poll::Pending;
        }
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
        // A dropped connection is a closed one, whether or not it was shut
        // down first, so a test reading the answer is never left waiting.
        self.outgoing.close();
    }
}

/// One accepted connection, handed over in memory instead of accepted on a port.
struct InMemoryListenerV1 {
    connections: UnboundedReceiver<ConnectionV1>,
}

impl Listener for InMemoryListenerV1 {
    type Io = ConnectionV1;
    type Addr = &'static str;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        match self.connections.recv().await {
            Some(connection) => (connection, "in-process"),
            // The test has stopped connecting. A listener never returns, so
            // this waits rather than reporting a connection it did not accept.
            None => pending().await,
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        Ok("in-process")
    }
}

struct EmptyHttpMetrics;
impl runtime_server::MetricsSnapshotProvider for EmptyHttpMetrics {
    fn snapshot(&self, _monotonic_micros: u64) -> String {
        String::new()
    }
}

/// The transport under test, reached over in-memory connections.
struct InProcessServerV1 {
    connections: UnboundedSender<ConnectionV1>,
    backend: Arc<CountingBackendV1>,
    frontend: Arc<PermissiveFrontendV1>,
}

impl InProcessServerV1 {
    fn start(config: ServerConfig) -> Self {
        Self::start_with_reasoning_field(config, "reasoning_content")
    }

    fn start_with_reasoning_field(config: ServerConfig, reasoning_field: &'static str) -> Self {
        Self::start_with_options(
            config,
            reasoning_field,
            runtime_server::HttpOptions::default(),
        )
    }

    fn start_with_options(
        config: ServerConfig,
        reasoning_field: &'static str,
        options: runtime_server::HttpOptions,
    ) -> Self {
        let backend = CountingBackendV1::new();
        let frontend = PermissiveFrontendV1::new_with_reasoning_field(reasoning_field);
        let router = runtime_server::router_with_options(
            backend.clone(),
            frontend.clone(),
            config,
            Arc::new(runtime_server::SystemMonotonicClock::new()),
            Arc::new(EmptyHttpMetrics),
            options,
        )
        .expect("the fake deployment is servable");
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

    /// Starts a deployment that admits only plain single-turn chat.
    fn plain() -> Self {
        Self::start(
            ServerConfig::default().with_request_surface(RequestSurfaceV1::SingleTurnPlainChat),
        )
    }

    /// Starts a deployment that admits every surface the transport validates.
    fn full() -> Self {
        Self::start(ServerConfig::default())
    }

    async fn post(&self, path: &str, body: &str) -> ResponseV1 {
        answer(&self.open(path, body.len(), body.as_bytes())).await
    }

    /// Opens one connection announcing a `length`-byte body and sends `body`,
    /// which may be only its start. Returns the direction the answer comes on.
    fn open(&self, path: &str, length: usize, body: &[u8]) -> ChannelV1 {
        self.connect(path, length, body, false)
    }

    fn connect(&self, path: &str, length: usize, body: &[u8], stalled: bool) -> ChannelV1 {
        let head = format!(
            "POST {path} HTTP/1.1\r\nHost: in-process\r\nContent-Type: application/json\r\n\
             Content-Length: {length}\r\nConnection: close\r\n\r\n"
        );
        let incoming = ChannelV1::default();
        let outgoing = ChannelV1::default();
        // What was sent is there before the router accepts the connection,
        // and the connection asked to be closed, so the answer ends by itself.
        incoming.write(head.as_bytes());
        incoming.write(body);
        self.connections
            .send(ConnectionV1 {
                incoming,
                outgoing: outgoing.clone(),
                stalled,
            })
            .expect("the served router is still running");
        outgoing
    }

    async fn raw(&self, method: &str, path: &str, headers: &str, body: &str) -> ResponseV1 {
        let incoming = ChannelV1::default();
        let outgoing = ChannelV1::default();
        incoming.write(format!("{method} {path} HTTP/1.1\r\nHost: in-process\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{headers}\r\n{body}", body.len()).as_bytes());
        self.connections
            .send(ConnectionV1 {
                incoming,
                outgoing: outgoing.clone(),
                stalled: false,
            })
            .unwrap();
        answer(&outgoing).await
    }

    async fn get(&self, path: &str) -> ResponseV1 {
        let incoming = ChannelV1::default();
        let outgoing = ChannelV1::default();
        incoming.write(
            format!("GET {path} HTTP/1.1\r\nHost: in-process\r\nConnection: close\r\n\r\n")
                .as_bytes(),
        );
        self.connections
            .send(ConnectionV1 {
                incoming,
                outgoing: outgoing.clone(),
                stalled: false,
            })
            .expect("the served router is still running");
        answer(&outgoing).await
    }

    /// Sends one chat request and requires the surface to have refused it.
    ///
    /// A refusal is only a refusal if nothing downstream ran, so this checks
    /// the frontend rendered nothing, the backend tokenized nothing, and the
    /// backend was never submitted to.
    async fn refuses_chat(&self, body: &str, parameter: &str) {
        let renders = self.frontend.renders();
        let tokenizations = self.backend.tokenizations();
        let submissions = self.backend.submissions();

        let response = self.post("/v1/chat/completions", body).await;

        assert_eq!(response.status, 400, "{body} answered {}", response.body);
        let error = response.json();
        assert_eq!(error["error"]["param"], parameter, "{}", response.body);
        assert_eq!(error["error"]["type"], "BadRequestError");
        assert_eq!(self.frontend.renders(), renders, "{body} reached rendering");
        assert_eq!(
            self.backend.tokenizations(),
            tokenizations,
            "{body} reached the tokenizer"
        );
        assert_eq!(
            self.backend.submissions(),
            submissions,
            "{body} reached generation"
        );
    }
}

/// The whole answer on one connection's outgoing direction.
async fn answer(outgoing: &ChannelV1) -> ResponseV1 {
    let raw = timeout(
        Duration::from_secs(5),
        poll_fn(|context| outgoing.poll_whole(context)),
    )
    .await
    .expect("the response arrives");
    ResponseV1::parse(&raw)
}

struct ResponseV1 {
    status: u16,
    headers: String,
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
        let body = if head
            .to_ascii_lowercase()
            .contains("transfer-encoding: chunked")
        {
            dechunk(body)
        } else {
            body.to_owned()
        };
        Self {
            status,
            headers: head.to_owned(),
            body,
        }
    }

    fn json(&self) -> Value {
        serde_json::from_str(&self.body).expect("the response body is JSON")
    }

    /// Every `data:` payload of a server-sent event stream, in arrival order.
    fn events(&self) -> Vec<String> {
        self.body
            .split("\n\n")
            .filter_map(|block| block.trim().strip_prefix("data: ").map(str::to_owned))
            .collect()
    }
}

/// Reassembles a chunked body, which is how a streamed response arrives.
fn dechunk(body: &str) -> String {
    let mut rest = body;
    let mut decoded = String::new();
    while let Some((header, remainder)) = rest.split_once("\r\n") {
        let size = usize::from_str_radix(header.split(';').next().unwrap_or_default().trim(), 16)
            .expect("a chunk header is a hexadecimal size");
        if size == 0 {
            break;
        }
        let (chunk, remainder) = remainder.split_at(size);
        decoded.push_str(chunk);
        rest = remainder.strip_prefix("\r\n").unwrap_or(remainder);
    }
    decoded
}

fn chat(body: &str) -> String {
    format!(r#"{{"model":"{MODEL}",{body}}}"#)
}

/// One user message, optionally preceded by one system message, is the surface.
///
/// That is the whole of what plain single-turn chat means here, and it is
/// stated as a passing request rather than left to the refusals below.
#[tokio::test]
async fn plain_chat_serves_one_user_turn_after_an_optional_system_turn() {
    let server = InProcessServerV1::plain();

    for (body, rendered) in [
        (
            r#""messages":[{"role":"user","content":"hello"}]"#,
            "user:hello",
        ),
        (
            r#""messages":[{"role":"system","content":"rules"},{"role":"user","content":"hello"}]"#,
            "system:rules|user:hello",
        ),
        // An empty tool list offers no tools, which is what this surface is.
        (
            r#""messages":[{"role":"user","content":"hello"}],"tools":[]"#,
            "user:hello",
        ),
        // The bounds a completion always carried are unchanged by the surface.
        (
            r#""messages":[{"role":"user","content":"hello"}],"max_completion_tokens":8,"temperature":0,"seed":7,"stop":["x"],"n":1,"logprobs":true,"top_logprobs":2"#,
            "user:hello",
        ),
    ] {
        let response = server.post("/v1/chat/completions", &chat(body)).await;
        assert_eq!(response.status, 200, "{body} answered {}", response.body);
        assert_eq!(response.json()["choices"][0]["message"]["content"], "hi");
        assert_eq!(
            server.backend.prompts().last(),
            Some(&PreparedPrompt::RenderedChat(rendered.to_owned())),
            "{body} rendered something else"
        );
    }
}

/// A streamed plain chat request is served, and a refused one still refuses.
///
/// Streaming is a delivery choice rather than a conversation feature, so the
/// surface neither forbids it nor lets a request in through it.
#[tokio::test]
async fn the_surface_holds_for_streaming_and_nonstreaming_alike() {
    let server = InProcessServerV1::plain();

    let streamed = server
        .post(
            "/v1/chat/completions",
            &chat(r#""messages":[{"role":"user","content":"hello"}],"stream":true"#),
        )
        .await;
    assert_eq!(streamed.status, 200, "{}", streamed.body);
    assert!(
        streamed
            .headers
            .to_ascii_lowercase()
            .contains("content-type: text/event-stream"),
        "{}",
        streamed.headers
    );
    let events = streamed.events();
    assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
    let deltas = events
        .iter()
        .filter_map(|event| serde_json::from_str::<Value>(event).ok())
        .filter_map(|event| {
            event["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect::<Vec<_>>();
    assert_eq!(deltas.concat(), "hi");
    assert_eq!(server.backend.submissions(), 1);

    for body in [
        r#""messages":[{"role":"user","content":"x"}],"stream":true,"tools":[{"type":"function","function":{"name":"lookup"}}]"#,
        r#""messages":[{"role":"user","content":"x"}],"stream":true,"tool_choice":"auto""#,
    ] {
        let parameter = if body.contains("tool_choice") {
            "tool_choice"
        } else {
            "tools"
        };
        server.refuses_chat(&chat(body), parameter).await;
    }
    // The streamed refusals admitted nothing beyond the one served request.
    assert_eq!(server.backend.submissions(), 1);
}

#[tokio::test]
async fn chat_prompt_logprobs_and_ids_use_vllm_root_fields() {
    let server = InProcessServerV1::full();
    let response = server.post("/v1/chat/completions", &chat(
        r#""messages":[{"role":"user","content":"x"}],"max_tokens":1,"temperature":0,"prompt_logprobs":2,"return_token_ids":true"#,
    )).await;
    assert_eq!(response.status, 200, "{}", response.body);
    let body = response.json();
    assert_eq!(body["prompt_token_ids"], serde_json::json!([7, 8, 9, 10]));
    let rows = body["prompt_logprobs"]
        .as_array()
        .expect("prompt rows at response root");
    assert_eq!(rows.len(), PROMPT_TOKENS);
    assert!(rows[0].is_null());
    assert_eq!(rows[1]["8"]["logprob"], -0.2);
    assert!(body["choices"][0].get("prompt_logprobs").is_none());
    assert_eq!(server.backend.submissions(), 1);

    let ids_only = server.post("/v1/chat/completions", &chat(
        r#""messages":[{"role":"user","content":"x"}],"max_tokens":1,"return_token_ids":true"#,
    )).await;
    assert_eq!(ids_only.status, 200, "{}", ids_only.body);
    assert_eq!(
        ids_only.json()["prompt_token_ids"],
        serde_json::json!([7, 8, 9, 10])
    );
    assert!(ids_only.json().get("prompt_logprobs").is_none());

    server
        .refuses_chat(
            &chat(r#""messages":[{"role":"user","content":"x"}],"prompt_logprobs":21"#),
            "prompt_logprobs",
        )
        .await;
    server
        .refuses_chat(
            &chat(
                r#""messages":[{"role":"user","content":"x"}],"prompt_logprobs":2,"stream":true"#,
            ),
            "prompt_logprobs",
        )
        .await;
}

#[tokio::test]
async fn a_cache_salt_reaches_the_backend_and_an_empty_one_is_refused() {
    let server = InProcessServerV1::full();
    let response = server
        .post(
            "/v1/chat/completions",
            &chat(r#""messages":[{"role":"user","content":"x"}],"max_tokens":1,"cache_salt":"tenant-a""#),
        )
        .await;
    assert_eq!(response.status, 200, "{}", response.body);
    let response = server
        .post(
            "/v1/completions",
            &format!(
                r#"{{"model":"{MODEL}","prompt":"x","max_tokens":1,"cache_salt":"tenant-b"}}"#
            ),
        )
        .await;
    assert_eq!(response.status, 200, "{}", response.body);
    let response = server
        .post(
            "/v1/chat/completions",
            &chat(r#""messages":[{"role":"user","content":"x"}],"max_tokens":1,"cache_salt":null"#),
        )
        .await;
    assert_eq!(response.status, 200, "{}", response.body);
    assert_eq!(
        server.backend.cache_salts(),
        [
            Some("tenant-a".to_owned()),
            Some("tenant-b".to_owned()),
            None
        ]
    );
    server
        .refuses_chat(
            &chat(r#""messages":[{"role":"user","content":"x"}],"cache_salt":"""#),
            "cache_salt",
        )
        .await;
}

#[tokio::test]
async fn chat_forwards_final_message_continuation_to_the_frontend() {
    let server = InProcessServerV1::full();
    let response = server.post("/v1/chat/completions", &chat(
        r#""messages":[{"role":"user","content":"x"},{"role":"assistant","content":"answer"}],"continue_final_message":true,"max_tokens":1,"prompt_logprobs":2,"return_token_ids":true"#,
    )).await;
    assert_eq!(response.status, 200, "{}", response.body);
    assert_eq!(
        response.json()["prompt_token_ids"],
        serde_json::json!([7, 8, 9, 10])
    );
    assert!(server.frontend.requests.lock().unwrap()[0].continue_final_message);

    let narrow = InProcessServerV1::plain();
    narrow
        .refuses_chat(
            &chat(r#""messages":[{"role":"user","content":"x"}],"continue_final_message":true"#),
            "continue_final_message",
        )
        .await;
}

#[tokio::test]
async fn streamed_chat_returns_prompt_ids_in_its_first_chunk() {
    let server = InProcessServerV1::plain();
    let response = server.post("/v1/chat/completions", &chat(
        r#""messages":[{"role":"user","content":"x"}],"max_tokens":1,"return_token_ids":true,"stream":true"#,
    )).await;
    assert_eq!(response.status, 200, "{}", response.body);
    let events = response.events();
    let first: Value = serde_json::from_str(&events[0]).expect("role chunk JSON");
    assert_eq!(first["prompt_token_ids"], serde_json::json!([7, 8, 9, 10]));
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
    assert!(events[1..events.len() - 1].iter().all(|event| {
        serde_json::from_str::<Value>(event)
            .ok()
            .is_none_or(|value| value.get("prompt_token_ids").is_none())
    }));
}

/// A tool definition or a tool choice is refused before anything runs.
#[tokio::test]
async fn plain_chat_refuses_tool_definitions_and_tool_choices() {
    let server = InProcessServerV1::plain();

    let user = r#""messages":[{"role":"user","content":"x"}]"#;
    for (body, parameter) in [
        (
            format!(
                r#"{user},"tools":[{{"type":"function","function":{{"name":"lookup","parameters":{{}}}}}}]"#
            ),
            "tools",
        ),
        (format!(r#"{user},"tool_choice":"auto""#), "tool_choice"),
        (format!(r#"{user},"tool_choice":"none""#), "tool_choice"),
        (format!(r#"{user},"tool_choice":"required""#), "tool_choice"),
        (
            format!(r#"{user},"parallel_tool_calls":true"#),
            "parallel_tool_calls",
        ),
    ] {
        server.refuses_chat(&chat(&body), parameter).await;
    }
}

/// A conversation that already continued is not a single turn.
#[tokio::test]
async fn plain_chat_refuses_assistant_and_tool_history() {
    let server = InProcessServerV1::plain();

    for body in [
        // An assistant turn, alone or as history.
        r#""messages":[{"role":"assistant","content":"earlier"}]"#,
        r#""messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]"#,
        r#""messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"user","content":"c"}]"#,
        // A tool result, which only exists after a tool call.
        r#""messages":[{"role":"tool","content":"result","tool_call_id":"call-1"}]"#,
        r#""messages":[{"role":"user","content":"a"},{"role":"tool","content":"r","tool_call_id":"c"}]"#,
        // A tool call carried on a message rather than declared as a tool.
        r#""messages":[{"role":"user","content":"a","tool_calls":[{"id":"c","type":"function","function":{"name":"n","arguments":"{}"}}]}]"#,
        // Reasoning history, which this surface has not validated either.
        r#""messages":[{"role":"user","content":"a","reasoning_content":"why"}]"#,
        // More than one user turn, and a system turn that is not the first.
        r#""messages":[{"role":"user","content":"a"},{"role":"user","content":"b"}]"#,
        r#""messages":[{"role":"user","content":"a"},{"role":"system","content":"b"}]"#,
        r#""messages":[{"role":"system","content":"a"},{"role":"system","content":"b"},{"role":"user","content":"c"}]"#,
        // A system turn with no user turn is not a request to answer.
        r#""messages":[{"role":"system","content":"a"}]"#,
        // The developer role renders differently and nothing measured it.
        r#""messages":[{"role":"developer","content":"a"}]"#,
        r#""messages":[{"role":"developer","content":"a"},{"role":"user","content":"b"}]"#,
        r#""messages":[]"#,
    ] {
        server.refuses_chat(&chat(body), "messages").await;
    }
}

/// Content this surface cannot render as one string is refused as itself.
#[tokio::test]
async fn plain_chat_refuses_multimodal_and_absent_content() {
    let server = InProcessServerV1::plain();

    for body in [
        r#""messages":[{"role":"user","content":[{"type":"text","text":"a"}]}]"#,
        r#""messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:x"}}]}]"#,
        r#""messages":[{"role":"user","content":null}]"#,
        r#""messages":[{"role":"user"}]"#,
        r#""messages":[{"role":"user","content":7}]"#,
        r#""messages":[{"role":"system","content":["a"]},{"role":"user","content":"b"}]"#,
    ] {
        server.refuses_chat(&chat(body), "messages").await;
    }
}

#[tokio::test]
async fn full_chat_accepts_text_parts_and_carries_ordered_media() {
    let server = InProcessServerV1::full();
    let text = chat(
        r#""messages":[{"role":"user","content":[{"type":"text","text":"ab"},{"type":"text","text":"cd"}]}],"max_tokens":1"#,
    );
    let response = server.post("/v1/chat/completions", &text).await;
    assert_eq!(response.status, 200, "{}", response.body);
    assert!(matches!(
        server.backend.prompts()[0],
        PreparedPrompt::RenderedChat(_)
    ));
    server.frontend.media_enabled.store(true, Ordering::Relaxed);
    let media = chat(
        r#""messages":[{"role":"user","content":[{"type":"text","text":"before"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA","detail":"auto"}},{"type":"text","text":"after"}]}],"max_tokens":1"#,
    );
    let response = server.post("/v1/chat/completions", &media).await;
    assert_eq!(response.status, 200, "{}", response.body);
    let prompts = server.backend.prompts();
    let PreparedPrompt::MediaChat { segments, media } = &prompts[1] else {
        panic!("media prompt retained")
    };
    assert_eq!(
        segments,
        &vec![
            PromptSegmentV1::Text("before".to_owned()),
            PromptSegmentV1::Media(0),
            PromptSegmentV1::Text("after".to_owned())
        ]
    );
    assert_eq!(media.len(), 1);
    let video = chat(
        r#""messages":[{"role":"user","content":[{"type":"video_url","video_url":{"url":"data:video/mp4;base64,AAAA"}}]}],"max_tokens":1"#,
    );
    let response = server.post("/v1/chat/completions", &video).await;
    assert_eq!(response.status, 200, "{}", response.body);
    let prompts = server.backend.prompts();
    let PreparedPrompt::MediaChat { media, .. } = &prompts[2] else {
        panic!("video prompt retained")
    };
    assert_eq!(
        media[0].kind,
        runtime_server::FrontendMediaPartKindV1::Video
    );
}

#[tokio::test]
async fn remote_media_parts_require_opt_in_and_preserve_the_source_for_preparation() {
    let server = InProcessServerV1::full();
    server.frontend.media_enabled.store(true, Ordering::Relaxed);
    for kind in ["image_url", "audio_url", "video_url", "input_audio"] {
        let url = "https://media.example/asset?signature=abc";
        let part = if kind == "input_audio" {
            serde_json::json!({"type": kind, "input_audio": {"data": url, "format": "wav"}})
        } else {
            serde_json::json!({"type": kind, kind: {"url": url}})
        };
        let body = chat(&format!(
            r#""messages":[{{"role":"user","content":[{part}]}}],"max_tokens":1"#
        ));
        server
            .frontend
            .remote_media_enabled
            .store(false, Ordering::Relaxed);
        server.refuses_chat(&body, "messages").await;
        server
            .frontend
            .remote_media_enabled
            .store(true, Ordering::Relaxed);
        let response = server.post("/v1/chat/completions", &body).await;
        assert_eq!(response.status, 200, "{}", response.body);
        let prompts = server.backend.prompts();
        let PreparedPrompt::MediaChat { media, .. } = prompts.last().unwrap() else {
            panic!("media retained");
        };
        assert_eq!(media[0].source, url);
        assert_eq!(media[0].part_type, kind);
    }
}

/// A media part of a modality the server does not serve is refused by its
/// modality and the flag that serves it, before anything reads or fetches
/// it; `/health` and `/v1/models` report the served modalities.
#[tokio::test]
async fn media_of_a_modality_the_server_does_not_serve_is_refused_by_name() {
    use runtime_server::FrontendMediaPartKindV1::Image;
    let server = InProcessServerV1::full();
    server
        .frontend
        .selected_modalities
        .store(true, Ordering::Relaxed);
    let part = |part: &str| {
        chat(&format!(
            r#""messages":[{{"role":"user","content":[{part}]}}],"max_tokens":1"#
        ))
    };
    let image = part(r#"{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}"#);
    let refused = [
        (
            "video",
            part(r#"{"type":"video_url","video_url":{"url":"https://example.com/a.mp4"}}"#),
        ),
        (
            "audio",
            part(r#"{"type":"audio_url","audio_url":{"url":"https://example.com/a.wav"}}"#),
        ),
        (
            "audio",
            part(r#"{"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}}"#),
        ),
    ];
    let refusal = |served: &'static str| {
        let server = &server;
        let image = image.clone();
        let refused = refused.clone();
        async move {
            for (modality, body) in std::iter::once(("image", image)).chain(refused) {
                if served.contains(modality) {
                    continue;
                }
                server.refuses_chat(&body, "messages").await;
                let message =
                    server.post("/v1/chat/completions", &body).await.json()["error"]["message"]
                        .clone();
                assert_eq!(
                    message,
                    format!(
                        "{modality} input is not served by this server, which serves {served} \
                         (--modalities); --modalities all or a list naming {modality} serves \
                         it when the checkpoint provides it"
                    ),
                );
            }
            assert_eq!(
                server.get("/health").await.json()["modalities"],
                serde_json::json!(served.split(',').collect::<Vec<_>>())
            );
            let models = server.get("/v1/models").await.json();
            assert_eq!(
                models["data"][0]["modalities"],
                serde_json::json!(served.split(',').collect::<Vec<_>>())
            );
        }
    };
    refusal("text").await;
    server.frontend.media_enabled.store(true, Ordering::Relaxed);
    refusal("text,image,video,audio").await;
    *server.frontend.served_media.lock().unwrap() = Some(vec![Image]);
    refusal("text,image").await;
    let response = server.post("/v1/chat/completions", &image).await;
    assert_eq!(response.status, 200, "{}", response.body);
}

#[tokio::test]
async fn unsupported_media_does_not_invent_an_operator_selection() {
    let server = InProcessServerV1::full();
    server.frontend.media_enabled.store(true, Ordering::Relaxed);
    *server.frontend.served_media.lock().unwrap() =
        Some(vec![runtime_server::FrontendMediaPartKindV1::Image]);
    let body = chat(
        r#""messages":[{"role":"user","content":[{"type":"video_url","video_url":{"url":"not-a-url"}}]}],"max_tokens":1"#,
    );
    let response = server.post("/v1/chat/completions", &body).await;
    assert_eq!(response.status, 400);
    assert_eq!(
        response.json()["error"]["message"],
        "video input is not supported by this server, which serves text,image"
    );
    assert_eq!(server.frontend.renders(), 0);
}

#[tokio::test]
async fn media_parts_fail_closed_for_unsupported_bundles_and_urls() {
    let server = InProcessServerV1::full();
    let image = r#""messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}],"max_tokens":1"#;
    server.refuses_chat(&chat(image), "messages").await;
    let response = server
        .post("/v1/chat/completions", &chat(image))
        .await
        .json();
    assert_eq!(
        response["error"]["message"],
        "this bundle does not support media inputs"
    );
    server.frontend.media_enabled.store(true, Ordering::Relaxed);
    for part in [
        r#"{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}"#,
        r#"{"type":"audio_url","audio_url":{"url":"file:///tmp/a.wav"}}"#,
        r#"{"type":"video_url","video_url":{"url":"data:video/mp4;base64,AA=A"}}"#,
        r#"{"type":"input_audio","input_audio":{"data":"AA=A","format":"wav"}}"#,
        r#"{"type":"unknown"}"#,
    ] {
        let body = chat(&format!(
            r#""messages":[{{"role":"user","content":[{part}]}}],"max_tokens":1"#
        ));
        server.refuses_chat(&body, "messages").await;
    }
    let system_media = chat(
        r#""messages":[{"role":"system","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]},{"role":"user","content":"hello"}],"max_tokens":1"#,
    );
    server.refuses_chat(&system_media, "messages").await;
}

/// Image formats, `file` parts and images in tool results follow the
/// model's declaration. Undeclared, GIF and WebP data URLs are refused by
/// format and a `file` part as an unsupported type; declared, a nested
/// `file` part carries an image data URL as an image part in user and tool
/// messages, and its other shapes are refused by name. A data URL's media
/// type compares without regard to case or parameters for every model.
#[tokio::test]
async fn image_formats_and_file_parts_follow_the_declaration() {
    let server = InProcessServerV1::full();
    server.frontend.media_enabled.store(true, Ordering::Relaxed);
    let user = |part: &str| {
        chat(&format!(
            r#""messages":[{{"role":"user","content":[{part}]}}],"max_tokens":1"#
        ))
    };
    let tool = |part: &str| {
        chat(&format!(
            r#""messages":[{{"role":"user","content":"a"}},{{"role":"assistant","content":null,"tool_calls":[{{"id":"c","type":"function","function":{{"name":"n","arguments":"{{}}"}}}}]}},{{"role":"tool","tool_call_id":"c","content":[{part}]}}],"max_tokens":1"#
        ))
    };
    let image = |url: &str| format!(r#"{{"type":"image_url","image_url":{{"url":"{url}"}}}}"#);
    let file =
        r#"{"type":"file","file":{"file_data":"data:image/png;base64,AAAA","filename":"a.png"}}"#;
    let refusal = |body: String| {
        let server = &server;
        async move {
            server.refuses_chat(&body, "messages").await;
            server.post("/v1/chat/completions", &body).await.json()["error"]["message"]
                .as_str()
                .unwrap()
                .to_owned()
        }
    };
    let accepted = |body: String| {
        let server = &server;
        async move {
            let response = server.post("/v1/chat/completions", &body).await;
            assert_eq!(response.status, 200, "{body}: {}", response.body);
        }
    };

    for url in [
        "data:IMAGE/PNG;base64,AAAA",
        "data:image/jpg;base64,AAAA",
        "data:image/Jpeg;name=a.jpg;base64,AAAA",
    ] {
        accepted(user(&image(url))).await;
    }
    assert_eq!(
        refusal(user(&image("data:image/gif;base64,AAAA"))).await,
        "image/gif images are not accepted by this model, which accepts image/png, image/jpeg"
    );
    assert_eq!(
        refusal(user(&image("data:image/svg+xml;base64,AAAA"))).await,
        "unsupported media MIME type or invalid base64"
    );
    assert_eq!(
        refusal(user(file)).await,
        "unsupported content part type 'file'"
    );
    assert_eq!(
        refusal(tool(&image("data:image/png;base64,AAAA"))).await,
        "only user messages may contain media"
    );

    server
        .frontend
        .declared_images
        .store(true, Ordering::Relaxed);
    for url in ["data:image/gif;base64,AAAA", "data:image/WEBP;base64,AAAA"] {
        accepted(user(&image(url))).await;
    }
    accepted(user(file)).await;
    accepted(tool(file)).await;
    accepted(tool(&image("data:image/webp;base64,AAAA"))).await;
    {
        let requests = server.frontend.requests.lock().unwrap();
        let parts = |request: &FrontendChatRequestV1| {
            request
                .messages
                .iter()
                .flat_map(|message| message.parts.iter().flatten())
                .filter_map(|part| match part {
                    runtime_server::FrontendContentPartV1::Media(media) => Some(media.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let user_file = parts(&requests[requests.len() - 3]);
        assert_eq!(user_file.len(), 1);
        assert_eq!(
            user_file[0].kind,
            runtime_server::FrontendMediaPartKindV1::Image
        );
        assert_eq!(user_file[0].part_type, "file");
        assert_eq!(user_file[0].source, "data:image/png;base64,AAAA");
        assert_eq!(parts(&requests[requests.len() - 2])[0].part_type, "file");
    }
    for (part, message) in [
        (
            r#"{"type":"file","file":{"file_id":"file-1"}}"#,
            "file_id is not supported; pass the image as a base64 data URL in file_data",
        ),
        (
            r#"{"type":"file","file_data":"data:image/png;base64,AAAA"}"#,
            r#"a file part carries its fields in a "file" object: {"type": "file", "file": {"file_data": ...}}"#,
        ),
        (
            r#"{"type":"file","file":{"file_data":"data:application/pdf;base64,AAAA"}}"#,
            "unsupported media MIME type or invalid base64",
        ),
        (
            r#"{"type":"file","file":{"file_data":"https://example.com/a.png"}}"#,
            "file_data must be a base64 image data URL",
        ),
        (
            r#"{"type":"file","file":{"file_data":"data:image/png;base64,AAAA","format":"png"}}"#,
            "unknown content part field 'format'",
        ),
        (r#"{"type":"file","file":{}}"#, "file_data is required"),
    ] {
        assert_eq!(refusal(user(part)).await, message, "{part}");
    }
}

/// A Responses `function_call_output` with an `input_image` reaches the
/// frontend as the chat tool message with that image does, so a model that
/// declares images in tool results renders both alike.
#[tokio::test]
async fn responses_tool_output_images_reach_the_frontend_as_chat_tool_images() {
    let server = InProcessServerV1::full();
    server.frontend.media_enabled.store(true, Ordering::Relaxed);
    server
        .frontend
        .declared_images
        .store(true, Ordering::Relaxed);
    let url = "data:image/webp;base64,AAAA";
    let chat_body = chat(&format!(
        r#""messages":[{{"role":"user","content":"a"}},{{"role":"assistant","content":null,"tool_calls":[{{"id":"c","type":"function","function":{{"name":"n","arguments":"{{}}"}}}}]}},{{"role":"tool","tool_call_id":"c","content":[{{"type":"text","text":"seen"}},{{"type":"image_url","image_url":{{"url":"{url}"}}}}]}}],"max_tokens":1"#
    ));
    let response = server.post("/v1/chat/completions", &chat_body).await;
    assert_eq!(response.status, 200, "{}", response.body);
    let responses_body = serde_json::json!({"model": MODEL, "max_output_tokens": 1, "input": [
        {"role": "user", "content": "a"},
        {"type": "function_call", "call_id": "c", "name": "n", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c", "output": [
            {"type": "input_text", "text": "seen"},
            {"type": "input_image", "image_url": url}]}
    ]});
    let response = server
        .post("/v1/responses", &responses_body.to_string())
        .await;
    assert_eq!(response.status, 200, "{}", response.body);
    let requests = server.frontend.requests.lock().unwrap();
    let [.., from_chat, from_responses] = &requests[..] else {
        panic!("both requests render");
    };
    assert_eq!(from_responses.messages, from_chat.messages);
}

/// Everything else a chat request can select is refused by name.
///
/// The surface is an allowlist, so a field nothing here validated is named as
/// itself rather than quietly dropped or handed to a renderer.
#[tokio::test]
async fn plain_chat_refuses_every_field_it_did_not_validate() {
    let server = InProcessServerV1::plain();

    let user = r#""messages":[{"role":"user","content":"x"}]"#;
    for (body, parameter) in [
        (
            format!(r#"{user},"response_format":{{"type":"json_object"}}"#),
            "response_format",
        ),
        (
            format!(r#"{user},"reasoning_effort":"high""#),
            "reasoning_effort",
        ),
        (
            format!(r#"{user},"chat_template_kwargs":{{"thinking":true}}"#),
            "chat_template_kwargs",
        ),
    ] {
        server.refuses_chat(&chat(&body), parameter).await;
    }
}

/// The narrow surface bounds chat, and leaves completions as they were.
///
/// A completion is a prompt string on every surface, so the text and token
/// input path keeps the bounds it always had.
#[tokio::test]
async fn completions_are_unchanged_by_the_narrow_chat_surface() {
    let server = InProcessServerV1::plain();

    let response = server
        .post(
            "/v1/completions",
            &format!(
                r#"{{"model":"{MODEL}","prompt":"hello","max_tokens":8,"temperature":0,"seed":3,"stop":["x"],"logprobs":2}}"#
            ),
        )
        .await;
    assert_eq!(response.status, 200, "{}", response.body);
    assert_eq!(
        server.backend.prompts().last(),
        Some(&PreparedPrompt::Raw("hello".to_owned()))
    );

    // And the bounds it always carried still refuse what they always refused.
    let refused = server
        .post(
            "/v1/completions",
            &format!(r#"{{"model":"{MODEL}","prompt":"hello","temperature":2.5}}"#),
        )
        .await;
    assert_eq!(refused.status, 400);
    assert_eq!(refused.json()["error"]["param"], "temperature");
    assert_eq!(server.backend.submissions(), 1);
}

#[tokio::test]
async fn completions_accept_valid_token_id_arrays_and_refuse_other_arrays() {
    let server = InProcessServerV1::full();
    let accepted = server
        .post(
            "/v1/completions",
            &format!(r#"{{"model":"{MODEL}","prompt":[151643,151645],"max_tokens":8}}"#),
        )
        .await;
    assert_eq!(accepted.status, 200, "{}", accepted.body);
    assert_eq!(
        server.backend.prompts().last(),
        Some(&PreparedPrompt::TokenIds(vec![151643, 151645]))
    );
    for prompt in ["[]", "[-1]", "[1.5]", "[\"1\", 2]", "[4294967296]"] {
        let refused = server
            .post(
                "/v1/completions",
                &format!(r#"{{"model":"{MODEL}","prompt":{prompt},"max_tokens":8}}"#),
            )
            .await;
        assert_eq!(refused.status, 400, "{prompt}: {}", refused.body);
        assert_eq!(refused.json()["error"]["param"], "prompt");
    }
}

#[tokio::test]
async fn the_default_body_limit_admits_a_full_length_token_array() {
    let server = InProcessServerV1::full();
    let prompt = std::iter::repeat_n("151643", 1_048_448)
        .collect::<Vec<_>>()
        .join(",");
    let response = server
        .post(
            "/v1/completions",
            &format!(r#"{{"model":"{MODEL}","prompt":[{prompt}],"max_tokens":128}}"#),
        )
        .await;
    assert_eq!(response.status, 200, "{}", response.body);
    let prompts = server.backend.prompts();
    let Some(PreparedPrompt::TokenIds(ids)) = prompts.last() else {
        panic!("the backend did not receive token IDs");
    };
    assert_eq!(ids.len(), 1_048_448);
    assert_eq!(ids[0], 151643);
    assert_eq!(ids[ids.len() - 1], 151643);
}

#[tokio::test]
async fn configured_body_and_string_limits_refuse_oversized_requests() {
    let config = ServerConfig::new(2, 32, 128, 1024)
        .expect("small body limit")
        .with_max_string_bytes(4)
        .expect("small string limit");
    let server = InProcessServerV1::start(config);
    let string_refusal = server
        .post(
            "/v1/completions",
            &format!(r#"{{"model":"{MODEL}","prompt":"hello"}}"#),
        )
        .await;
    assert_eq!(string_refusal.status, 400);
    assert_eq!(string_refusal.json()["error"]["param"], "prompt");

    let body_refusal = server
        .post(
            "/v1/completions",
            &format!(r#"{{"model":"{MODEL}","prompt":[{}]}}"#, "1,".repeat(100)),
        )
        .await;
    assert_eq!(body_refusal.status, 400);
    assert_eq!(
        body_refusal.json()["error"]["message"],
        "request body exceeds the byte limit"
    );
    assert_eq!(server.backend.submissions(), 0);
}

/// With every generation slot taken, a request is refused before its prompt
/// is rendered or tokenized, not after.
#[tokio::test]
async fn over_capacity_requests_are_refused_before_preparation() {
    let server = InProcessServerV1::start(ServerConfig::new(1, 32, 4096, 1024).unwrap());
    server.backend.hold.store(true, Ordering::Relaxed);
    let body = format!(r#"{{"model":"{MODEL}","prompt":"hello"}}"#);
    let _held = server.open("/v1/completions", body.len(), body.as_bytes());
    timeout(Duration::from_secs(5), async {
        while server.backend.submissions() == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the first request is admitted");
    let tokenizations = server.backend.tokenizations();

    let refused = server.post("/v1/completions", &body).await;

    assert_eq!(refused.status, 429, "{}", refused.body);
    assert_eq!(server.backend.tokenizations(), tokenizations);
    assert_eq!(server.backend.submissions(), 1);
}

/// Bodies still arriving are charged against one intake budget, so stalled
/// uploads cannot buffer without bound, and each must arrive in time.
#[tokio::test]
async fn stalled_bodies_are_bounded_by_the_intake_budget_and_deadline() {
    let config = ServerConfig::new(8, 32, 64 * 1024, 1024)
        .unwrap()
        .with_intake_budget_bytes(128 * 1024)
        .with_body_read_timeout(Duration::from_millis(500));
    let server = InProcessServerV1::start(config);
    let stalled = [0, 1].map(|_| server.open("/v1/completions", 64 * 1024, &[b' '; 60 * 1024]));
    tokio::time::sleep(Duration::from_millis(100)).await;

    let body = format!(r#"{{"model":"{MODEL}","prompt":"hello"}}"#);
    let refused = server.post("/v1/completions", &body).await;
    assert_eq!(refused.status, 429, "{}", refused.body);
    assert_eq!(server.get("/health").await.status, 200);
    assert_eq!(server.backend.tokenizations(), 0);

    for outgoing in &stalled {
        let expired = answer(outgoing).await;
        assert_eq!(expired.status, 408, "{}", expired.body);
    }
    let admitted = server.post("/v1/completions", &body).await;
    assert_eq!(admitted.status, 200, "{}", admitted.body);
}

/// A stream whose client stopped reading cannot hold admission capacity: its
/// sends give up after the write timeout, which cancels it, and the next
/// request is admitted while the stalled connection is still open.
#[tokio::test]
async fn a_stalled_stream_reader_releases_its_capacity() {
    let config = ServerConfig::new(1, 32, 4096, 16 << 20)
        .unwrap()
        .with_stream_write_timeout(Duration::from_millis(200));
    let server = InProcessServerV1::start(config);
    // More than the transport buffers, so the writer really blocks.
    server.backend.flood.store(4096, Ordering::Relaxed);
    let body = format!(r#"{{"model":"{MODEL}","prompt":"hello","max_tokens":8000,"stream":true}}"#);
    let _stalled = server.connect("/v1/completions", body.len(), body.as_bytes(), true);
    // Connection acceptance does not order request admission. Wait until the
    // stalled request consumes the flood before a retry can reach the backend.
    timeout(Duration::from_secs(5), async {
        while server.backend.submissions() == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the stalled stream is admitted");
    let plain = format!(r#"{{"model":"{MODEL}","prompt":"hello"}}"#);
    let admitted = timeout(Duration::from_secs(5), async {
        loop {
            let response = server.post("/v1/completions", &plain).await;
            if response.status == 200 {
                return response;
            }
            assert_eq!(response.status, 429, "{}", response.body);
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("the stalled stream releases its capacity");
    assert_eq!(admitted.json()["choices"][0]["text"], "hi");
    assert_eq!(server.backend.submissions(), 2);
}

#[tokio::test]
async fn chat_accepts_declared_reasoning_and_message_tools() {
    let server =
        InProcessServerV1::start_with_reasoning_field(ServerConfig::default(), "reasoning");
    let response = server
        .post(
            "/v1/chat/completions",
            &format!(
                r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"question"}},{{"role":"assistant","content":"answer","reasoning":"chosen","reasoning_content":"ignored"}},{{"role":"user","content":"next","tools":[{{"type":"function","function":{{"name":"lookup"}}}}]}}],"chat_template_kwargs":{{"enable_thinking":false}},"max_tokens":8}}"#
            ),
        )
        .await;
    assert_eq!(response.status, 200, "{}", response.body);
    let requests = server
        .frontend
        .requests
        .lock()
        .expect("request log is usable");
    let request = requests.last().expect("chat reached the frontend");
    assert_eq!(
        request.messages[1].reasoning_content.as_deref(),
        Some("chosen")
    );
    assert_eq!(request.messages[2].tools.len(), 1);
    // Each template keyword arrives as sent, for the frontend to read.
    assert_eq!(request.enable_thinking, Some(false));
    assert_eq!(request.thinking, None);
}

/// The full surface serves everything it served before.
///
/// Every request the narrow surface refuses is admitted here, so the narrowing
/// is a deployment's declaration rather than a change to the transport.
#[tokio::test]
async fn the_full_surface_still_serves_tools_history_and_reasoning() {
    let server = InProcessServerV1::full();

    let bodies = [
        r#""messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{}}}]"#,
        r#""messages":[{"role":"user","content":"x"}],"tool_choice":"auto""#,
        r#""messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"user","content":"c"}]"#,
        r#""messages":[{"role":"tool","content":"r","tool_call_id":"c"}]"#,
        r#""messages":[{"role":"developer","content":"a"},{"role":"user","content":"b"}]"#,
        r#""messages":[{"role":"user","content":"x"}],"reasoning_effort":"high""#,
        r#""messages":[{"role":"user","content":"x"}],"chat_template_kwargs":{"thinking":true}"#,
        r#""messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_object"}"#,
    ];
    for body in bodies {
        let response = server.post("/v1/chat/completions", &chat(body)).await;
        assert_eq!(response.status, 200, "{body} answered {}", response.body);
    }
    assert_eq!(server.backend.submissions(), bodies.len());
    assert_eq!(server.frontend.renders(), bodies.len());
}

/// A surface is one configured fact, and it composes with the others.
#[tokio::test]
async fn a_configured_surface_is_carried_beside_every_other_bound() {
    let profile = ServerCapacityProfile::new("fixture-profile", 4, 4).expect("a usable profile");
    let configured = ServerConfig::default()
        .with_request_surface(RequestSurfaceV1::SingleTurnPlainChat)
        .with_capacity_profile(profile)
        .with_maximum_context_tokens(256);

    assert_eq!(
        configured.request_surface(),
        RequestSurfaceV1::SingleTurnPlainChat
    );
    assert_eq!(configured.max_in_flight(), 4);
    assert_eq!(configured.maximum_context_tokens(), Some(256));
    assert!(configured.acceptance_profile().streaming_allowed());
    assert_eq!(
        configured.acceptance_profile().profile_name(),
        Some("fixture-profile")
    );

    // The order the two are named in does not change either of them.
    let reversed = ServerConfig::default()
        .with_capacity_profile(profile)
        .with_request_surface(RequestSurfaceV1::SingleTurnPlainChat);
    assert_eq!(
        reversed.acceptance_profile(),
        configured.acceptance_profile()
    );
    assert_eq!(reversed.request_surface(), configured.request_surface());

    // Naming no surface serves every one the transport validates.
    assert_eq!(
        ServerConfig::default().request_surface(),
        RequestSurfaceV1::Full
    );
    assert_eq!(
        ServerConfig::new(4, 4, 1024, 1024)
            .expect("a usable configuration")
            .request_surface(),
        RequestSurfaceV1::Full
    );
    assert_eq!(RequestSurfaceV1::default(), RequestSurfaceV1::Full);

    // The surface names itself the same way in a receipt as in configuration.
    assert_eq!(
        RequestSurfaceV1::SingleTurnPlainChat.name(),
        SINGLE_TURN_PLAIN_CHAT_SURFACE_NAME_V1
    );
    assert_eq!(RequestSurfaceV1::Full.name(), "full");
    assert!(RequestSurfaceV1::Full.tools_admitted());
    assert!(RequestSurfaceV1::Full.chat_history_admitted());
    assert!(!RequestSurfaceV1::SingleTurnPlainChat.tools_admitted());
    assert!(!RequestSurfaceV1::SingleTurnPlainChat.chat_history_admitted());
}

/// One-token acceptance keeps its own refusals under a narrowed surface.
#[tokio::test]
async fn one_token_acceptance_is_unchanged_by_the_surface() {
    let server = InProcessServerV1::start(
        ServerConfig::default()
            .with_one_token_acceptance_profile(PROMPT_TOKENS)
            .with_request_surface(RequestSurfaceV1::SingleTurnPlainChat),
    );

    let refused = server
        .post(
            "/v1/chat/completions",
            &chat(r#""messages":[{"role":"user","content":"x"}]"#),
        )
        .await;
    assert_eq!(refused.status, 400);
    assert!(
        refused.json()["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("one-token acceptance")),
        "{}",
        refused.body
    );

    let completion = server
        .post(
            "/v1/completions",
            &format!(r#"{{"model":"{MODEL}","prompt":"hello","max_tokens":1,"temperature":0}}"#),
        )
        .await;
    assert_eq!(completion.status, 200, "{}", completion.body);
    assert_eq!(server.backend.submissions(), 1);
}

#[tokio::test]
async fn production_sampling_fields_reach_the_backend() {
    let server = InProcessServerV1::full();
    for (path, input) in [
        ("/v1/completions", r#""prompt":"hello""#),
        (
            "/v1/chat/completions",
            r#""messages":[{"role":"user","content":"hello"}],"max_completion_tokens":8"#,
        ),
    ] {
        let body = format!(
            r#"{{"model":"{MODEL}",{input},"max_tokens":4,"min_tokens":2,"seed":-1,"top_k":-1,"user":"client"}}"#
        );
        let response = server.post(path, &body).await;
        assert_eq!(response.status, 200, "{}", response.body);
        let requests = server.backend.prompts.lock().unwrap();
        let request = requests.last().unwrap();
        assert_ne!(request.seed, u64::MAX);
        assert_eq!(request.sampling.minimum_new_tokens(), 2);
        assert_eq!(request.sampling.top_p(), 0.95);
        assert_eq!(request.sampling.top_k(), 0);
        assert_eq!(
            request.max_new_tokens,
            if path.contains("chat") { 8 } else { 4 }
        );
    }
    let response = server
        .post(
            "/v1/completions",
            &format!(r#"{{"model":"{MODEL}","prompt":"x","max_tokens":2,"min_tokens":3}}"#),
        )
        .await;
    assert_eq!(response.status, 400);
    assert_eq!(response.json()["error"]["param"], "min_tokens");
}

#[tokio::test]
async fn continuous_usage_counts_tokens_and_finishes_with_usage() {
    let server = InProcessServerV1::full();
    let response = server.post("/v1/chat/completions", &format!(r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}],"max_tokens":8,"stream":true,"stream_options":{{"include_usage":true,"continuous_usage_stats":true}}}}"#)).await;
    assert_eq!(response.status, 200, "{}", response.body);
    let events: Vec<Value> = response
        .events()
        .iter()
        .filter(|event| event.as_str() != "[DONE]")
        .map(|event| serde_json::from_str(event).unwrap())
        .collect();
    assert_eq!(events[0]["usage"]["completion_tokens"], 0);
    assert_eq!(events.last().unwrap()["choices"], serde_json::json!([]));
    for event in &events[1..] {
        assert_eq!(event["usage"]["completion_tokens"], 1);
        assert_eq!(event["usage"]["prompt_tokens"], PROMPT_TOKENS);
        assert_eq!(event["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
    }
}

#[tokio::test]
async fn tokenizer_routes_round_trip_without_generation_and_validate_ids() {
    let server = InProcessServerV1::full();
    let response = server
        .post(
            "/tokenize",
            r#"{"prompt":"hello","return_token_strs":true}"#,
        )
        .await;
    assert_eq!(response.status, 200, "{}", response.body);
    let value = response.json();
    assert_eq!(value["count"], 5);
    assert_eq!(value["max_model_len"], MAXIMUM_TOTAL_TOKENS);
    assert_eq!(
        value["token_strs"],
        serde_json::json!(["h", "e", "l", "l", "o"])
    );
    let response = server
        .post(
            "/detokenize",
            &serde_json::json!({"tokens": value["tokens"]}).to_string(),
        )
        .await;
    assert_eq!(response.status, 200);
    assert_eq!(response.json()["prompt"], "hello");
    assert_eq!(server.backend.submissions(), 0);
    assert_eq!(
        server
            .post("/detokenize", r#"{"tokens":[-1]}"#)
            .await
            .status,
        400
    );
    assert_eq!(
        server
            .post("/tokenize", r#"{"model":"absent","prompt":"hello"}"#)
            .await
            .status,
        404
    );
}

#[tokio::test]
async fn default_lengths_models_and_unknown_extensions_follow_the_reference() {
    let server = InProcessServerV1::full();
    let response = server
        .post(
            "/v1/completions",
            r#"{"prompt":"hello","stop":[],"future_extension":true}"#,
        )
        .await;
    assert_eq!(response.status, 200, "{}", response.body);
    assert_eq!(
        server
            .backend
            .prompts
            .lock()
            .unwrap()
            .last()
            .unwrap()
            .max_new_tokens,
        16
    );
    let response = server.post("/v1/chat/completions", r#"{"messages":[{"role":"user","content":"hi"}],"future_extension":true,"parallel_tool_calls":false}"#).await;
    assert_eq!(response.status, 200, "{}", response.body);
    assert_eq!(
        server
            .backend
            .prompts
            .lock()
            .unwrap()
            .last()
            .unwrap()
            .max_new_tokens,
        MAXIMUM_TOTAL_TOKENS - PROMPT_TOKENS
    );
    let response = server
        .post(
            "/v1/completions",
            r#"{"prompt":"hello","use_beam_search":true}"#,
        )
        .await;
    assert_eq!(response.status, 400);
    assert_eq!(response.json()["error"]["code"], 400);
    assert_eq!(response.json()["error"]["param"], "use_beam_search");
}

/// A backend that refuses an over-context prompt before preparing it is
/// answered exactly as the context check after preparation answers.
#[tokio::test]
async fn an_early_context_refusal_reads_as_the_late_one() {
    for (endpoint, body, prompt_tokens) in [
        (
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":16}"#,
            MAXIMUM_TOTAL_TOKENS - 8,
        ),
        (
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}]}"#,
            MAXIMUM_TOTAL_TOKENS,
        ),
        (
            "/v1/completions",
            r#"{"prompt":"hello","max_tokens":16}"#,
            MAXIMUM_TOTAL_TOKENS + 5,
        ),
    ] {
        let mut answers = Vec::new();
        for early in [false, true] {
            let server = InProcessServerV1::full();
            server
                .backend
                .prompt_count
                .store(prompt_tokens, Ordering::Relaxed);
            server.backend.refuse_early.store(early, Ordering::Relaxed);
            let response = server.post(endpoint, body).await;
            assert_eq!(response.status, 400, "{body}: {}", response.body);
            assert_eq!(server.backend.submissions(), 0);
            // Only the late check prepared the prompt.
            assert_eq!(server.backend.tokenizations() == 0, early, "{body}");
            answers.push(response.json()["error"].clone());
        }
        assert_eq!(answers[0], answers[1], "{body}");
    }
}

#[tokio::test]
async fn request_stop_controls_survive_admission() {
    let server = InProcessServerV1::full();
    let response = server.post("/v1/completions", r#"{"prompt":"hello","stop_token_ids":[9,8,9],"ignore_eos":true,"skip_special_tokens":false,"include_stop_str_in_output":true}"#).await;
    assert_eq!(response.status, 200, "{}", response.body);
    let requests = server.backend.prompts.lock().unwrap();
    let options = &requests.last().unwrap().decoding;
    assert_eq!(options.stop_token_ids, [8, 9]);
    assert!(options.ignore_eos);
    assert!(!options.skip_special_tokens);
    assert!(options.include_stop_str_in_output);
}

#[tokio::test]
async fn reference_optional_booleans_and_temperature_bounds_are_preserved() {
    let server = InProcessServerV1::full();
    for (endpoint, body) in [
        (
            "/v1/completions",
            r#"{"prompt":"hello","stream":null,"temperature":2,"max_tokens":1}"#,
        ),
        (
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}],"stream":null,"logprobs":null,"top_logprobs":0,"max_tokens":1}"#,
        ),
    ] {
        let response = server.post(endpoint, body).await;
        assert_eq!(response.status, 200, "{}", response.body);
    }
    let response = server
        .post(
            "/v1/completions",
            r#"{"prompt":"hello","temperature":2.5,"max_tokens":1}"#,
        )
        .await;
    assert_eq!(response.status, 400);
    assert_eq!(response.json()["error"]["param"], "temperature");
}

#[tokio::test]
async fn multiple_choices_share_preparation_and_offset_seeds() {
    for endpoint in ["/v1/completions", "/v1/chat/completions"] {
        for stream in [false, true] {
            let server = InProcessServerV1::full();
            let mut body =
                serde_json::json!({"n": 3, "seed": -2, "max_tokens": 4, "stream": stream});
            if endpoint == "/v1/completions" {
                body["prompt"] = serde_json::json!("hello");
            } else {
                body["messages"] = serde_json::json!([{"role": "user", "content": "hello"}]);
            }
            if stream {
                body["stream_options"] =
                    serde_json::json!({"include_usage": true, "continuous_usage_stats": true});
            }
            let response = server.post(endpoint, &body.to_string()).await;
            assert_eq!(response.status, 200, "{}", response.body);
            assert_eq!(server.backend.tokenizations(), 1);
            let requests = server.backend.prompts.lock().unwrap();
            assert_eq!(
                requests.iter().map(|r| r.seed).collect::<Vec<_>>(),
                [u64::MAX - 1, u64::MAX, 0]
            );
            assert_ne!(requests[0].request_id, requests[1].request_id);
            drop(requests);
            let usage = if stream {
                let events = response.events();
                assert_eq!(
                    events
                        .iter()
                        .filter(|event| event.as_str() == "[DONE]")
                        .count(),
                    1
                );
                let values: Vec<Value> = events
                    .iter()
                    .filter_map(|event| serde_json::from_str(event).ok())
                    .collect();
                let mut terminals: Vec<u64> = values
                    .iter()
                    .flat_map(|value| value["choices"].as_array().unwrap())
                    .filter(|choice| !choice["finish_reason"].is_null())
                    .map(|choice| choice["index"].as_u64().unwrap())
                    .collect();
                terminals.sort();
                assert_eq!(terminals, [0, 1, 2]);
                for pair in values.windows(2) {
                    assert!(
                        pair[0]["usage"]["completion_tokens"].as_u64()
                            <= pair[1]["usage"]["completion_tokens"].as_u64()
                    );
                }
                values.last().unwrap()["usage"].clone()
            } else {
                let value = response.json();
                assert_eq!(
                    value["choices"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|choice| choice["index"].as_u64().unwrap())
                        .collect::<Vec<_>>(),
                    [0, 1, 2]
                );
                value["usage"].clone()
            };
            assert_eq!(usage["prompt_tokens"], PROMPT_TOKENS);
            assert_eq!(usage["completion_tokens"], 3);
            assert_eq!(usage["total_tokens"], PROMPT_TOKENS + 3);
        }
    }
}

#[tokio::test]
async fn completion_batches_index_choices_and_count_each_prompt_once() {
    for prompt in [
        serde_json::json!(["one", "two"]),
        serde_json::json!([[7, 8], [9]]),
    ] {
        for stream in [false, true] {
            let server = InProcessServerV1::full();
            server.backend.report_cached.store(true, Ordering::Relaxed);
            let mut body =
                serde_json::json!({"prompt": prompt, "n": 2, "seed": 23, "stream": stream});
            if stream {
                body["stream_options"] = serde_json::json!({"include_usage": true});
            }
            let response = server.post("/v1/completions", &body.to_string()).await;
            assert_eq!(response.status, 200, "{}", response.body);
            assert_eq!(server.backend.tokenizations(), 2);
            assert_eq!(
                server
                    .backend
                    .prompts
                    .lock()
                    .unwrap()
                    .iter()
                    .map(|r| r.seed)
                    .collect::<Vec<_>>(),
                [23, 24, 23, 24]
            );
            let usage = if stream {
                let events: Vec<Value> = response
                    .events()
                    .iter()
                    .filter_map(|event| serde_json::from_str(event).ok())
                    .collect();
                let mut indexes: Vec<_> = events
                    .iter()
                    .flat_map(|event| event["choices"].as_array().unwrap())
                    .filter(|choice| !choice["finish_reason"].is_null())
                    .map(|choice| choice["index"].as_u64().unwrap())
                    .collect();
                indexes.sort();
                assert_eq!(indexes, [0, 1, 2, 3]);
                events.last().unwrap()["usage"].clone()
            } else {
                let value = response.json();
                assert_eq!(value["choices"].as_array().unwrap().len(), 4);
                value["usage"].clone()
            };
            assert_eq!(usage["prompt_tokens"], PROMPT_TOKENS * 2);
            assert_eq!(usage["completion_tokens"], 4);
            // Each prompt's first choice (index 0 and 2) reports 1 cached token.
            assert_eq!(
                usage["prompt_tokens_details"]["cached_tokens"], 2,
                "{usage}"
            );
        }
    }
    let server = InProcessServerV1::full();
    for prompt in [
        serde_json::json!([]),
        serde_json::json!(["x", [1]]),
        serde_json::json!([[1], []]),
        serde_json::json!([[1], [-1]]),
    ] {
        let response = server
            .post(
                "/v1/completions",
                &serde_json::json!({"prompt": prompt}).to_string(),
            )
            .await;
        assert_eq!(response.status, 400, "{}", response.body);
    }
    assert_eq!(server.backend.submissions(), 0);
}

#[tokio::test]
async fn special_token_flags_preserve_the_installed_tokenizer_contract() {
    let server = InProcessServerV1::full();
    for flag in [
        serde_json::json!(true),
        serde_json::json!(false),
        Value::Null,
    ] {
        for chat in [false, true] {
            let mut body = serde_json::json!({"add_special_tokens": flag, "return_token_ids": true, "max_tokens": 1});
            let endpoint = if chat {
                body["messages"] = serde_json::json!([{"role":"user","content":"hello"}]);
                "/v1/chat/completions"
            } else {
                body["prompt"] = serde_json::json!("hello");
                "/v1/completions"
            };
            let response = server.post(endpoint, &body.to_string()).await;
            assert_eq!(response.status, 200, "{}", response.body);
            assert_eq!(
                response.json()["prompt_token_ids"],
                serde_json::json!([7, 8, 9, 10])
            );
        }
    }
    let response = server
        .post(
            "/v1/completions",
            r#"{"prompt":"hello","add_special_tokens":"yes"}"#,
        )
        .await;
    assert_eq!(response.status, 400);
}

#[tokio::test]
async fn structured_output_fields_become_request_constraints() {
    let server = InProcessServerV1::full();
    let schema = serde_json::json!({"type": "object", "properties": {"a": {"type": "integer"}}});
    let expected = runtime_grammar::json_schema_grammar(&schema, Default::default()).unwrap();
    let chat = |extra: &str| {
        format!(r#"{{"messages":[{{"role":"user","content":"hi"}}],"max_tokens":4,{extra}}}"#)
    };
    let cases = [
        (
            "/v1/chat/completions",
            chat(&format!(
                r#""response_format":{{"type":"json_schema","json_schema":{{"name":"x","strict":true,"schema":{schema}}}}}"#
            )),
            Some(expected.clone()),
        ),
        (
            "/v1/chat/completions",
            chat(&format!(r#""structured_outputs":{{"json":{schema}}}"#)),
            Some(expected),
        ),
        (
            "/v1/completions",
            r#"{"prompt":"x","response_format":{"type":"json_object"}}"#.to_owned(),
            Some(runtime_grammar::json_object_grammar(Default::default())),
        ),
        (
            "/v1/completions",
            r#"{"prompt":"x","structured_outputs":{"choice":["yes","no"]}}"#.to_owned(),
            Some(runtime_grammar::choice_grammar(&["yes".to_owned(), "no".to_owned()]).unwrap()),
        ),
        (
            "/v1/chat/completions",
            chat(r#""structured_outputs":{"regex":"\\d+"}"#),
            Some(runtime_grammar::regex_grammar(r"\d+").unwrap()),
        ),
        (
            "/v1/chat/completions",
            chat(r#""response_format":{"type":"text"},"structured_outputs":null"#),
            None,
        ),
        (
            "/v1/chat/completions",
            chat(&format!(
                r#""structured_outputs":{{"json":{schema},"disable_additional_properties":true,"whitespace_pattern":"[ ]?"}}"#
            )),
            Some(runtime_grammar::json_schema_grammar(&schema, Default::default()).unwrap()),
        ),
    ];
    for (path, body, expected) in cases {
        let response = server.post(path, &body).await;
        assert_eq!(response.status, 200, "{body}: {}", response.body);
        let requests = server.backend.prompts.lock().unwrap();
        let constraint = requests.last().unwrap().constraint.clone();
        assert_eq!(constraint.as_deref(), expected.as_ref(), "{body}");
    }
}

#[tokio::test]
async fn json_whitespace_options_reach_chat_and_completion_masks() {
    use runtime_grammar::{CompiledGrammar, GrammarMatcher, TokenVocabulary, VocabularyToken};
    let server = InProcessServerV1::full();
    let vocabulary = std::sync::Arc::new(
        TokenVocabulary::new(
            1,
            vec![VocabularyToken {
                bytes: b" ".to_vec(),
                special: false,
            }],
        )
        .unwrap(),
    );
    for path in ["/v1/chat/completions", "/v1/completions"] {
        for schema in [false, true] {
            for response_format in [false, true] {
                for disable in [None, Some(false), Some(true)] {
                    let mut body = serde_json::json!({"max_tokens":1});
                    if path == "/v1/completions" {
                        body["prompt"] = serde_json::json!("x");
                    } else {
                        body["messages"] = serde_json::json!([{"role":"user","content":"x"}]);
                    }
                    if response_format {
                        body["response_format"] = if schema {
                            serde_json::json!({"type":"json_schema", "json_schema":{"name":"x","schema":{"type":"object"}}})
                        } else {
                            serde_json::json!({"type":"json_object"})
                        };
                    } else {
                        body["structured_outputs"] = if schema {
                            serde_json::json!({"json":{"type":"object"}})
                        } else {
                            serde_json::json!({"json_object":true})
                        };
                    }
                    if let Some(disable) = disable {
                        body["structured_outputs"]["disable_any_whitespace"] =
                            serde_json::json!(disable);
                    }
                    let response = server.post(path, &body.to_string()).await;
                    assert_eq!(response.status, 200, "{}", response.body);
                    let grammar = server
                        .backend
                        .prompts
                        .lock()
                        .unwrap()
                        .last()
                        .unwrap()
                        .constraint
                        .clone()
                        .unwrap();
                    let compiled = std::sync::Arc::new(
                        CompiledGrammar::new(&grammar, vocabulary.clone()).unwrap(),
                    );
                    let mut matcher = GrammarMatcher::new(compiled);
                    matcher.accept_bytes(b"{").unwrap();
                    if disable == Some(true) {
                        assert!(!matcher.accepts_token(0).unwrap());
                    } else {
                        matcher.accept_bytes(" ".repeat(257).as_bytes()).unwrap();
                        assert_eq!(matcher.accepts_token(0).unwrap(), disable == Some(false));
                    }
                }
            }
        }
    }
}

#[tokio::test]
async fn vllm_schema_keywords_reach_request_constraints() {
    let server = InProcessServerV1::full();
    for schema in [
        serde_json::json!({"type": "integer", "multipleOf": 2}),
        serde_json::json!({"type": "string", "format": "email"}),
    ] {
        for response_format in [false, true] {
            let mut body = serde_json::json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1});
            if response_format {
                body["response_format"] = serde_json::json!({"type": "json_schema", "json_schema": {"name": "result", "strict": true, "schema": schema}});
            } else {
                body["structured_outputs"] = serde_json::json!({"json": schema});
            }
            let response = server.post("/v1/chat/completions", &body.to_string()).await;
            assert_eq!(response.status, 200, "{body}: {}", response.body);
            let requests = server.backend.prompts.lock().unwrap();
            assert!(requests.last().unwrap().constraint.is_some());
        }
    }
}

#[tokio::test]
async fn unsupported_structured_outputs_are_refused_with_their_field() {
    let server = InProcessServerV1::full();
    let cases = [
        (
            r#""structured_outputs":{"grammar":"root ::= \"x\""}"#,
            "structured_outputs",
            "not supported",
        ),
        (
            r#""structured_outputs":{"structural_tag":"{}"}"#,
            "structured_outputs",
            "not supported",
        ),
        (
            r#""structured_outputs":{"regex":"a","choice":["a"]}"#,
            "structured_outputs",
            "one kind",
        ),
        (
            r#""structured_outputs":{}"#,
            "structured_outputs",
            "none are specified",
        ),
        (
            r#""structured_outputs":{"regex":"(a"}"#,
            "structured_outputs",
            "regex",
        ),
        (
            r#""structured_outputs":{"unknown":1}"#,
            "structured_outputs",
            "unknown",
        ),
        (
            r#""structured_outputs":{"json_object":true,"disable_additional_properties":1}"#,
            "structured_outputs",
            "disable_additional_properties must be a boolean",
        ),
        (
            r#""structured_outputs":{"json_object":true,"whitespace_pattern":42}"#,
            "structured_outputs",
            "whitespace_pattern must be a string",
        ),
        (
            r#""structured_outputs":{"choice":["a"]},"response_format":{"type":"json_object"}"#,
            "response_format",
            "one kind",
        ),
        (
            r#""response_format":{"type":"json_schema"}"#,
            "response_format",
            "json_schema",
        ),
        (
            r#""response_format":{"type":"json_schema","json_schema":{"name":"x"}}"#,
            "structured_outputs",
            "none are specified",
        ),
        (
            r#""response_format":{"type":"structural_tag","format":{}}"#,
            "response_format",
            "structural_tag",
        ),
        (
            r#""response_format":{"type":"yaml"}"#,
            "response_format",
            "yaml",
        ),
    ];
    for (extra, parameter, message) in cases {
        for (path, body) in [
            (
                "/v1/chat/completions",
                format!(r#"{{"messages":[{{"role":"user","content":"hi"}}],{extra}}}"#),
            ),
            ("/v1/completions", format!(r#"{{"prompt":"hi",{extra}}}"#)),
        ] {
            let response = server.post(path, &body).await;
            assert_eq!(response.status, 400, "{body}: {}", response.body);
            let error = &response.json()["error"];
            assert_eq!(error["param"], parameter, "{body}");
            assert!(
                error["message"].as_str().unwrap().contains(message),
                "{body}: {error}"
            );
        }
    }
}

/// What the fake frontend's tool choices constrain to, when it constrains.
fn tool_grammar() -> runtime_grammar::Grammar {
    runtime_grammar::choice_grammar(&["<call>".to_owned()]).unwrap()
}

/// A declared vendor mapping constrains only strict requests, serves a
/// forced choice without tools as auto, and switches thinking by the
/// top-level object alone; without it, the same requests are served as
/// before.
#[tokio::test]
async fn a_declared_vendor_mapping_shapes_constraints_choices_and_thinking() {
    let schema = serde_json::json!({"type": "object", "properties": {"a": {"type": "integer"}}});
    let schema_grammar = runtime_grammar::json_schema_grammar(&schema, Default::default()).unwrap();
    let object_grammar = runtime_grammar::json_object_grammar(Default::default());
    let tools = r#""tools":[{"type":"function","function":{"name":"f","strict":true,"parameters":{"type":"object"}}}]"#;
    let format = |strict: &str| {
        format!(
            r#""response_format":{{"type":"json_schema","json_schema":{{"name":"x"{strict},"schema":{schema}}}}}"#
        )
    };
    // (request fields, constraint without the mapping, with it)
    let cases = [
        (
            r#""response_format":{"type":"json_object"}"#.to_owned(),
            Some(object_grammar.clone()),
            None,
        ),
        (format(""), Some(schema_grammar.clone()), None),
        (
            format(r#","strict":false"#),
            Some(schema_grammar.clone()),
            None,
        ),
        (
            format(r#","strict":true"#),
            Some(schema_grammar.clone()),
            Some(schema_grammar.clone()),
        ),
        (
            format!(r#""structured_outputs":{{"json":{schema}}}"#),
            Some(schema_grammar.clone()),
            Some(schema_grammar.clone()),
        ),
        (
            r#""structured_outputs":{"json_object":true}"#.to_owned(),
            Some(object_grammar.clone()),
            Some(object_grammar.clone()),
        ),
        // An explicit constraint stands beside a response format that
        // constrains nothing under the mapping.
        (
            r#""structured_outputs":{"json_object":true},"response_format":{"type":"json_object"}"#
                .to_owned(),
            Some(object_grammar.clone()),
            Some(object_grammar.clone()),
        ),
        (
            format!(
                r#""structured_outputs":{{"json":{{"type":"object"}}}},{}"#,
                format("")
            ),
            Some(schema_grammar.clone()),
            Some(
                runtime_grammar::json_schema_grammar(
                    &serde_json::json!({"type": "object"}),
                    Default::default(),
                )
                .unwrap(),
            ),
        ),
        (tools.to_owned(), Some(tool_grammar()), None),
        (
            format!(r#"{tools},"tool_choice":"auto",{}"#, format("")),
            Some(tool_grammar()),
            None,
        ),
        (
            format!(r#"{tools},"tool_choice":"required""#),
            Some(tool_grammar()),
            Some(tool_grammar()),
        ),
        (
            format!(r#"{tools},"tool_choice":{{"type":"function","function":{{"name":"f"}}}}"#),
            Some(tool_grammar()),
            Some(tool_grammar()),
        ),
    ];
    for mapped in [false, true] {
        let server = InProcessServerV1::full();
        server
            .frontend
            .constrains_tools
            .store(true, Ordering::Relaxed);
        server
            .frontend
            .vendor_mapping
            .store(mapped, Ordering::Relaxed);
        for (extra, unmapped_constraint, mapped_constraint) in &cases {
            let body = format!(
                r#"{{"messages":[{{"role":"user","content":"hi"}}],"max_tokens":4,{extra}}}"#
            );
            let response = server.post("/v1/chat/completions", &body).await;
            assert_eq!(response.status, 200, "{body}: {}", response.body);
            let constraint = server
                .backend
                .prompts
                .lock()
                .unwrap()
                .last()
                .unwrap()
                .constraint
                .clone();
            let expected = if mapped {
                mapped_constraint
            } else {
                unmapped_constraint
            };
            assert_eq!(constraint.as_deref(), expected.as_ref(), "{mapped}: {body}");
        }
        // A forced choice without tools is auto, or refused as before.
        for choice in [
            r#""required""#,
            r#"{"type":"function","function":{"name":"f"}}"#,
        ] {
            for tools in ["", r#","tools":[]"#] {
                let body = format!(
                    r#"{{"messages":[{{"role":"user","content":"hi"}}],"tool_choice":{choice}{tools}}}"#
                );
                let response = server.post("/v1/chat/completions", &body).await;
                if mapped {
                    assert_eq!(response.status, 200, "{body}: {}", response.body);
                    let requests = server.frontend.requests.lock().unwrap();
                    let request = requests.last().unwrap();
                    assert_eq!(request.tool_choice, FrontendToolChoiceV1::Auto, "{body}");
                    assert!(request.tools.is_empty());
                    let prompts = server.backend.prompts.lock().unwrap();
                    assert!(prompts.last().unwrap().constraint.is_none(), "{body}");
                } else {
                    assert_eq!(response.status, 400, "{body}: {}", response.body);
                    assert_eq!(response.json()["error"]["param"], "tool_choice");
                }
            }
        }
        // The thinking object switches thinking, overriding the effort and
        // the keywords, which a mapped frontend no longer reads.
        let kwargs = r#""chat_template_kwargs":{"enable_thinking":true,"thinking":true,"reasoning_effort":"max","preserve_thinking":true}"#;
        let thinking_cases = [
            (
                format!(r#""thinking":{{"type":"disabled","budget_tokens":8}},{kwargs}"#),
                (Some(false), Some(false), None, None),
                (Some(true), Some(true), Some("max"), None),
            ),
            (
                r#""thinking":{"type":"adaptive"},"reasoning_effort":"none""#.to_owned(),
                (Some(true), Some(true), None, None),
                (None, None, None, Some("none")),
            ),
            (
                r#""thinking":{"type":"enabled"},"reasoning_effort":"low""#.to_owned(),
                (Some(true), Some(true), None, Some("low")),
                (None, None, None, Some("low")),
            ),
            (
                format!(r#""reasoning_effort":"none",{kwargs}"#),
                (None, None, None, Some("none")),
                (Some(true), Some(true), Some("max"), Some("none")),
            ),
            (
                r#""thinking":null"#.to_owned(),
                (None, None, None, None),
                (None, None, None, None),
            ),
        ];
        for (extra, mapped_switch, unmapped_switch) in &thinking_cases {
            let body = format!(r#"{{"messages":[{{"role":"user","content":"hi"}}],{extra}}}"#);
            let response = server.post("/v1/chat/completions", &body).await;
            assert_eq!(response.status, 200, "{body}: {}", response.body);
            let requests = server.frontend.requests.lock().unwrap();
            let request = requests.last().unwrap();
            let seen = (
                request.enable_thinking,
                request.thinking,
                request.nested_reasoning_effort.as_deref(),
                request.reasoning_effort.as_deref(),
            );
            assert_eq!(
                &seen,
                if mapped {
                    mapped_switch
                } else {
                    unmapped_switch
                },
                "{mapped}: {body}"
            );
            if extra.contains("preserve_thinking") {
                assert_eq!(request.preserve_thinking, Some(true));
                // The forwarded object carries what the typed keywords say.
                let Some(runtime_server::FrontendJsonV1::Object(entries)) =
                    &request.chat_template_kwargs
                else {
                    panic!("{body}: no forwarded object");
                };
                let names: Vec<_> = entries.iter().map(|(name, _)| name.as_str()).collect();
                let expected: &[&str] = match (mapped, mapped_switch.0) {
                    (false, _) => &[
                        "enable_thinking",
                        "thinking",
                        "reasoning_effort",
                        "preserve_thinking",
                    ],
                    (true, Some(_)) => &["preserve_thinking", "enable_thinking", "thinking"],
                    (true, None) => &["preserve_thinking"],
                };
                assert_eq!(names, expected, "{mapped}: {body}");
                if mapped && let Some(switch) = mapped_switch.0 {
                    assert!(
                        entries[1..].iter().all(
                            |(_, value)| *value == runtime_server::FrontendJsonV1::Bool(switch)
                        )
                    );
                }
            }
        }
        for thinking in [
            r#"{"type":"on"}"#,
            r#"{"budget_tokens":8}"#,
            r#"true"#,
            r#"{"type":"enabled","budget_tokens":-1}"#,
            r#"{"type":"enabled","budget_tokens":1.5}"#,
        ] {
            let body = format!(
                r#"{{"messages":[{{"role":"user","content":"hi"}}],"thinking":{thinking}}}"#
            );
            let response = server.post("/v1/chat/completions", &body).await;
            if mapped {
                assert_eq!(response.status, 400, "{body}: {}", response.body);
                assert_eq!(response.json()["error"]["param"], "thinking", "{body}");
            } else {
                assert_eq!(response.status, 200, "{body}: {}", response.body);
            }
        }
        // The thinking keywords a mapped frontend ignores are not checked;
        // the keywords it still reads are, positionally too.
        for (kwargs, mapped_status, unmapped_status) in [
            (
                r#"{"enable_thinking":"ignored","thinking":1,"reasoning_effort":3}"#,
                200,
                400,
            ),
            (r#"[1, "x", 2, true, null, null]"#, 200, 400),
            (
                r#"{"enable_thinking":"ignored","preserve_thinking":"x"}"#,
                400,
                400,
            ),
            (r#"[null, null, null, "x", null, null]"#, 400, 400),
        ] {
            let body = format!(
                r#"{{"messages":[{{"role":"user","content":"hi"}}],"thinking":{{"type":"enabled"}},"chat_template_kwargs":{kwargs}}}"#
            );
            let response = server.post("/v1/chat/completions", &body).await;
            let expected = if mapped {
                mapped_status
            } else {
                unmapped_status
            };
            assert_eq!(
                response.status, expected,
                "{mapped}: {body}: {}",
                response.body
            );
            if expected == 200 {
                let requests = server.frontend.requests.lock().unwrap();
                let request = requests.last().unwrap();
                assert_eq!(
                    (request.enable_thinking, request.thinking),
                    (Some(true), Some(true))
                );
            }
        }
    }
}

#[tokio::test]
async fn tool_choices_are_validated_like_the_reference() {
    let server = InProcessServerV1::full();
    let tools = r#""tools":[{"type":"function","function":{"name":"get_time","parameters":{"type":"object","properties":{}}}}]"#;
    let cases = [
        // This fake frontend declares no tool-call format, so forcing is refused.
        (
            format!(r#"{tools},"tool_choice":"required""#),
            "tool_choice",
            "not supported",
        ),
        (
            format!(
                r#"{tools},"tool_choice":{{"type":"function","function":{{"name":"get_time"}}}}"#
            ),
            "tool_choice",
            "not supported",
        ),
        (
            format!(r#"{tools},"tool_choice":{{"type":"function","function":{{"name":"other"}}}}"#),
            "tool_choice",
            "does not match",
        ),
        (
            r#""tool_choice":"required""#.to_owned(),
            "tool_choice",
            "`tools` must be set",
        ),
        (
            format!(r#"{tools},"tool_choice":"sometimes""#),
            "tool_choice",
            "Invalid value",
        ),
        (
            format!(r#"{tools},"tool_choice":{{"type":"function","function":{{}}}}"#),
            "tool_choice.function.name",
            "name",
        ),
        (
            format!(
                r#"{tools},"tool_choice":{{"type":"function","function":{{"name":"get_time"}}}},"structured_outputs":{{"regex":"x"}}"#
            ),
            "structured_outputs",
            "not both",
        ),
    ];
    for (extra, parameter, message) in cases {
        let body = format!(r#"{{"messages":[{{"role":"user","content":"hi"}}],{extra}}}"#);
        let response = server.post("/v1/chat/completions", &body).await;
        assert_eq!(response.status, 400, "{body}: {}", response.body);
        let error = &response.json()["error"];
        assert_eq!(error["param"], parameter, "{body}");
        assert!(
            error["message"].as_str().unwrap().contains(message),
            "{body}: {error}"
        );
    }
}

// Frozen reference vLLM (b98186076) completion echo, replayed without a model.
fn echo_capture() -> Value {
    let receipt: Value = serde_json::from_str(include_str!(
        "../../../fixtures/server/completion-echo-vllm.json"
    ))
    .unwrap();
    receipt["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["case"] == "completion-echo")
        .unwrap()
        .clone()
}

fn captured_prompt_events(capture: &Value, logprobs: &Value) -> Vec<GenerationEvent> {
    let ids: Vec<u32> =
        serde_json::from_value(capture["stream_ids"][0]["choices"][0]["prompt_token_ids"].clone())
            .unwrap();
    let labels = logprobs["tokens"].as_array().unwrap();
    let rows = (1..ids.len())
        .map(|position| {
            let top = &capture["json"]["choices"][0]["prompt_logprobs"][position];
            let mut ranked = logprobs["top_logprobs"][position]
                .as_object()
                .unwrap()
                .iter()
                .collect::<Vec<_>>();
            ranked.sort_by(|(_, a), (_, b)| b.as_f64().unwrap().total_cmp(&a.as_f64().unwrap()));
            PromptTokenLogprob {
                position,
                token_id: ids[position],
                token: labels[position].as_str().unwrap().to_owned(),
                logprob: logprobs["token_logprobs"][position].as_f64().unwrap(),
                top_logprobs: ranked
                    .into_iter()
                    .take(2)
                    .enumerate()
                    .map(|(rank, (token, value))| {
                        let id = top
                            .as_object()
                            .unwrap()
                            .iter()
                            .find(|(_, entry)| entry["decoded_token"] == *token)
                            .map_or(500_000 + rank as u32, |(id, _)| id.parse().unwrap());
                        let token = token.clone();
                        let logprob = value.as_f64().unwrap();
                        (
                            id,
                            runtime_server::TokenLogprob {
                                bytes: token.as_bytes().to_vec(),
                                token,
                                logprob,
                            },
                        )
                    })
                    .collect(),
            }
        })
        .collect();
    vec![
        GenerationEvent::Prompt {
            tokens: labels[..ids.len()]
                .iter()
                .map(|v| v.as_str().unwrap().to_owned())
                .collect(),
            token_ids: ids,
            text: capture["request"]["prompt"].as_str().unwrap().to_owned(),
        },
        GenerationEvent::PromptLogprobs(rows),
    ]
}

/// The engine sends one prompt-row event per scored chunk, so the response
/// must be the same however the rows are split across events. The chunk
/// sizes themselves (the scored row cap) are the scheduler's and engine's,
/// covered in `runtime-engine`'s scored-prompt tests.
#[tokio::test]
async fn captured_echo_prompt_rows_include_scored_token_across_split_events() {
    let capture = echo_capture();
    let expected = &capture["json"]["choices"][0];
    for (top, rows_per_event) in [0, 2]
        .into_iter()
        .flat_map(|top| [1, 3, 512].map(|rows| (top, rows)))
    {
        let server = InProcessServerV1::full();
        server.backend.prompt_count.store(11, Ordering::Relaxed);
        let mut events = captured_prompt_events(&capture, &expected["logprobs"]);
        if top == 0
            && let GenerationEvent::PromptLogprobs(rows) = &mut events[1]
        {
            for row in rows {
                row.top_logprobs.clear();
            }
        }
        let GenerationEvent::PromptLogprobs(rows) = events.pop().unwrap() else {
            panic!("prompt rows")
        };
        events.extend(
            rows.chunks(rows_per_event)
                .map(|rows| GenerationEvent::PromptLogprobs(rows.to_vec())),
        );
        events.push(GenerationEvent::Finished {
            reason: FinishReason::Stop,
            usage: TokenUsage {
                prompt_tokens: 11,
                completion_tokens: 0,
                cached_tokens: None,
            },
        });
        *server.backend.replay.lock().unwrap() = Some(events);
        let request = serde_json::json!({"model": MODEL, "prompt": capture["request"]["prompt"],
            "echo": true, "max_tokens": 1, "logprobs": top, "prompt_logprobs": top});
        let response = server.post("/v1/completions", &request.to_string()).await;
        assert_eq!(response.status, 200, "{}", response.body);
        let body = response.json();
        let actual = &body["choices"][0]["logprobs"];
        assert!(actual["top_logprobs"][0].is_null());
        for position in 1..11 {
            let token = expected["logprobs"]["tokens"][position].as_str().unwrap();
            let score = &expected["logprobs"]["token_logprobs"][position];
            assert_eq!(&actual["top_logprobs"][position][token], score);
            assert_eq!(
                actual["top_logprobs"][position]
                    .as_object()
                    .unwrap()
                    .keys()
                    .next()
                    .unwrap(),
                token
            );
            if top == 2 {
                assert_eq!(
                    actual["top_logprobs"][position],
                    expected["logprobs"]["top_logprobs"][position]
                );
            } else {
                assert_eq!(
                    actual["top_logprobs"][position].as_object().unwrap().len(),
                    1
                );
            }
            let id =
                capture["stream_ids"][0]["choices"][0]["prompt_token_ids"][position].to_string();
            assert_eq!(&body["prompt_logprobs"][position][&id]["logprob"], score);
        }
    }
}

#[tokio::test]
async fn captured_echo_stream_ids_keep_text_and_scores_together() {
    replay_echo_stream_ids(true, true, None).await;
}

#[tokio::test]
async fn completion_stream_ids_preserve_hidden_tokens_and_optional_scores() {
    for echo in [false, true] {
        for logprobs in [false, true] {
            for hidden in [None, Some(0), Some(3)] {
                replay_echo_stream_ids(echo, logprobs, hidden).await;
            }
        }
    }
}

async fn replay_echo_stream_ids(echo: bool, logprobs: bool, hidden: Option<usize>) {
    let capture = echo_capture();
    let mut expected = capture["stream_ids"].as_array().unwrap().clone();
    let server = InProcessServerV1::full();
    server.backend.prompt_count.store(11, Ordering::Relaxed);
    let mut events = captured_prompt_events(&capture, &expected[0]["choices"][0]["logprobs"]);
    if !logprobs {
        events.truncate(1);
    }
    for (index, chunk) in expected.iter_mut().enumerate() {
        let choice = &chunk["choices"][0];
        let scores = choice["logprobs"].clone();
        let position = scores["tokens"].as_array().unwrap().len() - 1;
        let token = scores["tokens"][position].as_str().unwrap().to_owned();
        let id = choice["token_ids"][0].as_u64().unwrap() as u32;
        let top = scores["top_logprobs"][position]
            .as_object()
            .unwrap()
            .iter()
            .map(|(token, score)| runtime_server::TokenLogprob {
                token: token.clone(),
                bytes: token.as_bytes().to_vec(),
                logprob: score.as_f64().unwrap(),
            })
            .collect();
        if logprobs {
            events.push(GenerationEvent::GeneratedTokenLogprobs(vec![
                GeneratedTokenLogprob {
                    token_id: id,
                    bytes: token.as_bytes().to_vec(),
                    token,
                    logprob: scores["token_logprobs"][position].as_f64().unwrap(),
                    top_logprobs: top,
                },
            ]));
        }
        events.push(GenerationEvent::GeneratedTokenIds(vec![id]));
        if hidden == Some(index) {
            events.push(GenerationEvent::TokenUsageDelta {
                generated_token_count: 1,
            });
        } else {
            events.push(GenerationEvent::Content(VisibleContent {
                delta: choice["text"].as_str().unwrap().to_owned(),
                generated_token_count: 1,
            }));
        }
        if !echo && index == 0 {
            for field in ["tokens", "token_logprobs", "top_logprobs"] {
                chunk["choices"][0]["logprobs"][field] =
                    serde_json::json!([scores[field][position]]);
            }
            chunk["choices"][0]["logprobs"]["text_offset"] = serde_json::json!([0]);
        }
        if hidden == Some(index) {
            chunk["choices"][0]["text"] = "".into();
        }
        if hidden.is_some_and(|hidden| hidden < index) {
            chunk["choices"][0]["logprobs"]["text_offset"] = serde_json::json!([index - 1]);
        }
        if !logprobs {
            chunk["choices"][0]["logprobs"] = Value::Null;
        }
    }
    events.push(GenerationEvent::Finished {
        reason: FinishReason::Length,
        usage: TokenUsage {
            prompt_tokens: 11,
            completion_tokens: 4,
            cached_tokens: None,
        },
    });
    *server.backend.replay.lock().unwrap() = Some(events);
    let mut request = capture["request"].clone();
    request["model"] = MODEL.into();
    request["echo"] = echo.into();
    request["logprobs"] = if logprobs { 2.into() } else { Value::Null };
    request["stream"] = true.into();
    request["return_token_ids"] = true.into();
    let response = server.post("/v1/completions", &request.to_string()).await;
    assert_eq!(response.status, 200, "{}", response.body);
    let frames = response.events();
    assert_eq!(frames.last().unwrap(), "[DONE]");
    let actual = frames[..frames.len() - 1]
        .iter()
        .map(|frame| serde_json::from_str::<Value>(frame).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(actual.len(), expected.len(), "{actual:?}");
    for (actual, expected) in actual.iter().zip(expected) {
        for field in [
            "text",
            "logprobs",
            "token_ids",
            "prompt_token_ids",
            "finish_reason",
        ] {
            assert_eq!(
                actual["choices"][0][field], expected["choices"][0][field],
                "{field}"
            );
        }
    }
}

#[tokio::test]
async fn wide_json_is_refused_before_preparation() {
    let server = InProcessServerV1::start(ServerConfig::default());
    let fields = (0..4097)
        .map(|i| format!(r#""k{i}":0"#))
        .collect::<Vec<_>>()
        .join(",");
    let body = format!(r#"{{"model":"{MODEL}","prompt":"hello","best_of":{{{fields}}}}}"#);
    let response = server.post("/v1/completions", &body).await;
    assert_eq!(response.status, 400, "{}", response.body);
    assert_eq!(server.backend.tokenizations(), 0);
}

#[test]
fn ordered_json_bounds_width_and_preserves_order_and_duplicates() {
    use runtime_server::FrontendJsonV1;
    let fields = (0..4097)
        .map(|i| format!(r#""k{i}":0"#))
        .collect::<Vec<_>>()
        .join(",");
    assert!(serde_json::from_str::<FrontendJsonV1>(&format!("{{{fields}}}")).is_err());
    let value: FrontendJsonV1 = serde_json::from_str(r#"{"z":0,"a":1}"#).unwrap();
    let FrontendJsonV1::Object(entries) = value else {
        panic!("object expected")
    };
    assert_eq!(
        entries
            .iter()
            .map(|(key, _)| key.as_str())
            .collect::<Vec<_>>(),
        ["z", "a"]
    );
    assert!(serde_json::from_str::<FrontendJsonV1>(r#"{"z":0,"z":1}"#).is_err());
}

#[test]
fn ordered_json_refuses_a_repeated_key_before_reading_its_value() {
    use runtime_server::FrontendJsonV1;
    for text in [r#"{"x":0,"x":1}"#, r#"{"x":0,"x":{"y":0,"y":1}}"#] {
        assert_eq!(
            serde_json::from_str::<FrontendJsonV1>(text)
                .unwrap_err()
                .to_string(),
            "duplicate field `x` at line 1 column 10",
            "{text}"
        );
    }
}

#[tokio::test]
async fn positive_temperatures_use_the_reference_floor() {
    let server = InProcessServerV1::full();
    for (endpoint, input) in [
        ("/v1/completions", r#""prompt":"hello""#),
        (
            "/v1/chat/completions",
            r#""messages":[{"role":"user","content":"hello"}]"#,
        ),
    ] {
        for temperature in [0.0_f64, 1e-40, 1e-6, 0.009999, 0.01, 1.0] {
            let body =
                format!(r#"{{{input},"temperature":{temperature},"top_p":1,"max_tokens":1}}"#);
            let response = server.post(endpoint, &body).await;
            assert_eq!(response.status, 200, "{}", response.body);
            let requests = server.backend.prompts.lock().unwrap();
            let parameters = &requests.last().unwrap().sampling;
            let expected = if temperature > 0.0 {
                temperature.max(0.01)
            } else {
                0.0
            };
            assert_eq!(parameters.temperature(), expected);
            // Near-tied logits must use the effective distribution too.
            for seed in 0..32 {
                let draw = |p: &runtime_sampling::SamplingParameters| {
                    runtime_sampling::sample(
                        p,
                        &[0.0, 0.005],
                        runtime_sampling::SampleContext::new(&[], 0, &[]),
                        &mut runtime_sampling::SamplingState::new(seed),
                    )
                    .unwrap()
                    .token_id()
                };
                assert_eq!(
                    draw(parameters),
                    draw(&runtime_sampling::SamplingParameters::new(expected).unwrap())
                );
            }
        }
    }
}

#[tokio::test]
async fn seed_sentinel_uses_unseeded_admission_for_every_choice() {
    for endpoint in ["/v1/completions", "/v1/chat/completions"] {
        for n in [1, 2] {
            let server = InProcessServerV1::full();
            for seed in [None, Some(-1_i64), Some(0), Some(123), Some(-2)] {
                for _ in 0..2 {
                    let mut body = serde_json::json!({"max_tokens":1,"n":n});
                    body["seed"] = serde_json::json!(seed);
                    if endpoint == "/v1/completions" {
                        body["prompt"] = serde_json::json!(["hello", "again"]);
                    } else {
                        body["messages"] = serde_json::json!([{"role":"user","content":"hello"}]);
                    }
                    let response = server.post(endpoint, &body.to_string()).await;
                    assert_eq!(response.status, 200, "{}", response.body);
                    let mut requests = server.backend.prompts.lock().unwrap();
                    for (index, request) in requests.iter().enumerate() {
                        match seed {
                            Some(value) if value != -1 => assert_eq!(
                                request.seed,
                                (value as u64).wrapping_add((index % n) as u64)
                            ),
                            _ => assert_ne!(
                                request.seed,
                                u64::MAX,
                                "unseeded requests must not use the sentinel bits"
                            ),
                        }
                    }
                    requests.clear();
                }
            }
        }
    }
}

#[tokio::test]
async fn token_strings_use_vocabulary_spelling() {
    let server = InProcessServerV1::full();
    let response = server
        .post("/tokenize", r#"{"prompt":" \n","return_token_strs":true}"#)
        .await;
    assert_eq!(response.status, 200, "{}", response.body);
    assert_eq!(response.json()["token_strs"], serde_json::json!(["Ġ", "Ċ"]));
    for body in [
        serde_json::json!({"prompt":" "}),
        serde_json::json!({"prompt":" ","return_token_strs":false}),
    ] {
        let response = server.post("/tokenize", &body.to_string()).await;
        assert_eq!(response.status, 200, "{}", response.body);
        assert!(response.json()["token_strs"].is_null());
    }
}

#[tokio::test]
async fn spacing_option_is_accepted_on_both_generation_routes() {
    let server = InProcessServerV1::full();
    for spaces in [false, true] {
        for stream in [false, true] {
            for chat in [false, true] {
                let mut body = serde_json::json!({"spaces_between_special_tokens":spaces,"stream":stream,"max_tokens":1});
                let endpoint = if chat {
                    body["messages"] = serde_json::json!([{"role":"user","content":"hello"}]);
                    "/v1/chat/completions"
                } else {
                    body["prompt"] = serde_json::json!("hello");
                    "/v1/completions"
                };
                let response = server.post(endpoint, &body.to_string()).await;
                assert_eq!(response.status, 200, "{}", response.body);
                assert_eq!(
                    server
                        .backend
                        .prompts
                        .lock()
                        .unwrap()
                        .last()
                        .unwrap()
                        .decoding
                        .spaces_between_special_tokens,
                    spaces
                );
            }
        }
    }
}

#[tokio::test]
async fn spacing_option_rejects_non_boolean_values() {
    let server = InProcessServerV1::full();
    let response = server
        .post(
            "/v1/completions",
            r#"{"prompt":"hello","spaces_between_special_tokens":"no"}"#,
        )
        .await;
    assert_eq!(response.status, 400);
}

#[tokio::test]
async fn cors_opt_in_and_preflight_match_browser_contract() {
    let server = InProcessServerV1::start_with_options(
        ServerConfig::default(),
        "reasoning_content",
        runtime_server::HttpOptions {
            enable_cors: true,
            ..Default::default()
        },
    );
    let response = server.raw("OPTIONS", "/v1/chat/completions", "Origin: https://client.example\r\nAccess-Control-Request-Method: POST\r\nAccess-Control-Request-Headers: authorization, content-type\r\n", "").await;
    assert_eq!(response.status, 200);
    assert!(response.headers.contains("access-control-allow-origin: *"));
    assert!(
        response
            .headers
            .contains("access-control-allow-headers: authorization, content-type")
    );
    assert_eq!(server.backend.submissions(), 0);
    let response = server
        .raw("GET", "/missing", "Origin: https://client.example\r\n", "")
        .await;
    assert_eq!(response.status, 404);
    assert!(response.headers.contains("access-control-allow-origin: *"));
}

#[tokio::test]
async fn cors_restricts_preflight_and_supports_credentials() {
    let server = InProcessServerV1::start_with_options(
        ServerConfig::default(),
        "reasoning_content",
        runtime_server::HttpOptions {
            enable_cors: true,
            allowed_origins: vec!["https://client.example".into()],
            allowed_methods: vec!["POST".into()],
            allowed_headers: vec!["authorization".into()],
            allow_credentials: true,
            ..runtime_server::HttpOptions::default()
        },
    );
    for (origin, method, headers, status) in [
        (
            "https://client.example",
            "POST",
            "AUTHORIZATION, content-type",
            200,
        ),
        ("https://other.example", "POST", "authorization", 400),
        ("https://client.example", "DELETE", "authorization", 400),
        ("https://client.example", "POST", "x-secret", 400),
    ] {
        let response = server.raw("OPTIONS", "/v1/completions", &format!("Origin: {origin}\r\nAccess-Control-Request-Method: {method}\r\nAccess-Control-Request-Headers: {headers}\r\n"), "").await;
        assert_eq!(response.status, status);
        assert!(
            response
                .headers
                .contains("access-control-allow-credentials: true")
        );
        if status == 200 {
            assert!(
                response
                    .headers
                    .contains("access-control-allow-origin: https://client.example")
            );
            assert!(response.headers.contains("vary: Origin"));
        }
    }
}

#[tokio::test]
async fn api_key_guards_api_paths_and_exempts_operational_routes() {
    let server = InProcessServerV1::start_with_options(
        ServerConfig::default(),
        "reasoning_content",
        runtime_server::HttpOptions {
            enable_cors: true,
            api_key: Some(runtime_server::ApiKey::new("secret")),
            ..Default::default()
        },
    );
    for token in [
        "",
        "Bearer wrong",
        "Basic secret",
        "Bearer secret-extra",
        "Bearer",
    ] {
        let response = server
            .raw(
                "POST",
                "/v1/completions",
                &format!("Authorization: {token}\r\nOrigin: https://client.example\r\n"),
                "{}",
            )
            .await;
        assert_eq!(response.status, 401);
        assert_eq!(response.json(), serde_json::json!({"error":"Unauthorized"}));
        assert!(response.headers.contains("access-control-allow-origin: *"));
    }
    assert_eq!(server.backend.tokenizations(), 0);
    assert_eq!(server.backend.submissions(), 0);
    for scheme in ["Bearer", "bearer", "bEaReR"] {
        assert_eq!(
            server
                .raw(
                    "GET",
                    "/v1/models",
                    &format!("Authorization: {scheme} secret\r\n"),
                    ""
                )
                .await
                .status,
            200
        );
    }
    for path in ["/health", "/ping", "/metrics", "/version"] {
        assert_eq!(server.get(path).await.status, 200);
    }
    assert_eq!(
        server
            .raw(
                "OPTIONS",
                "/v1/completions",
                "Origin: https://client.example\r\nAccess-Control-Request-Method: POST\r\n",
                ""
            )
            .await
            .status,
        200
    );
    assert_eq!(
        server
            .raw("OPTIONS", "/v1/completions", "", "")
            .await
            .status,
        405
    );
    assert_eq!(
        InProcessServerV1::full().get("/v1/models").await.status,
        200
    );
}

#[tokio::test]
async fn request_id_headers_cover_errors_preflight_and_generation() {
    let server = InProcessServerV1::start_with_options(
        ServerConfig::default(),
        "reasoning_content",
        runtime_server::HttpOptions {
            enable_cors: true,
            enable_request_id_headers: true,
            ..Default::default()
        },
    );
    for (method, path, body, status) in [
        ("GET", "/version", "", 200),
        ("GET", "/missing", "", 404),
        ("POST", "/version", "", 405),
        ("POST", "/v1/completions", r#"{"prompt":"hello"}"#, 200),
        (
            "POST",
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hello"}],"stream":true}"#,
            200,
        ),
        ("OPTIONS", "/v1/completions", "", 200),
        ("POST", "/v1/completions", "{}", 400),
    ] {
        let response = server.raw(method, path, "X-Request-Id: caller-123\r\nOrigin: https://client.example\r\nAccess-Control-Request-Method: POST\r\n", body).await;
        assert_eq!(response.status, status, "{}", response.body);
        assert!(response.headers.contains("x-request-id: caller-123"));
    }
    for prepared in server.backend.prompts.lock().unwrap().iter() {
        assert_eq!(prepared.client_request_id.as_deref(), Some("caller-123"));
    }
    let first = server.get("/version").await;
    let second = server.get("/version").await;
    let id = |response: &ResponseV1| {
        response
            .headers
            .lines()
            .find(|line| line.starts_with("x-request-id:"))
            .unwrap()
            .to_owned()
    };
    assert_ne!(id(&first), id(&second));
    let generated = server
        .post("/v1/completions", r#"{"prompt":"hello"}"#)
        .await;
    let generated_id = id(&generated)
        .strip_prefix("x-request-id: ")
        .unwrap()
        .to_owned();
    assert_eq!(
        server
            .backend
            .prompts
            .lock()
            .unwrap()
            .last()
            .unwrap()
            .client_request_id
            .as_deref(),
        Some(generated_id.as_str())
    );
    let protected = InProcessServerV1::start_with_options(
        ServerConfig::default(),
        "reasoning_content",
        runtime_server::HttpOptions {
            enable_request_id_headers: true,
            api_key: Some(runtime_server::ApiKey::new("secret")),
            ..Default::default()
        },
    );
    let response = protected
        .raw("GET", "/v1/models", "X-Request-Id: denied\r\n", "")
        .await;
    assert_eq!(response.status, 401);
    assert!(response.headers.contains("x-request-id: denied"));
}

/// A host-named serving identity with every member set.
fn fixture_identity() -> runtime_server::ServingIdentity {
    runtime_server::ServingIdentity {
        model_options: Default::default(),
        model_options_qualified: None,
        bundle_id: "fixture.bundle.v1".into(),
        package_identity: Some("a".repeat(64)),
        checkpoint_identity: Some("b".repeat(64)),
        metadata_identity: Some("c".repeat(64)),
        startup_nonce: "e".repeat(32),
        profile: runtime_server::ServingProfile {
            name: Some("fixture-tp4".into()),
            tensor_parallel_size: 4,
            context_parallel_size: Some(1),
            speculation: runtime_server::ServingSpeculation {
                mode: "adaptive".into(),
                tokens: 3,
                drafter: Some("mtp".into()),
            },
            kv_cache_dtype: "bf16".into(),
            kv_cache_planes: Vec::new(),
            prefix_cache: true,
            max_sequences: 32,
        },
        kv_page_tokens: 64,
        version: "1.2.3".into(),
        source_commit: None,
    }
}

/// The identity the host names reaches `/health` and `/ping` unchanged, with
/// the names `/v1/models` lists; without one, health reports readiness and
/// the served modalities and waiting counters.
#[tokio::test]
async fn health_reports_the_serving_identity_the_host_named() {
    let server = InProcessServerV1::full();
    let health = server.get("/health").await;
    assert_eq!(health.status, 200);
    assert_eq!(
        health.json(),
        serde_json::json!({"status": "ready", "modalities": ["text"], "transport":{"buffer_budget_bytes":536870912,"buffer_reserved_bytes":0}, "queue_depth":0, "waiting":{"total":0,"pre_backend":0,"transport":0,"intake":0,"preparation":0,"staging":0,"storage":0,"backend":0}})
    );

    let identity = fixture_identity();
    let server = InProcessServerV1::start_with_options(
        ServerConfig::default(),
        "reasoning_content",
        runtime_server::HttpOptions {
            served_model_names: vec!["primary".into(), "alias".into()],
            serving_identity: Some(identity),
            ..Default::default()
        },
    );
    let expected = serde_json::json!({
        "transport":{"buffer_budget_bytes":536870912,"buffer_reserved_bytes":0}, "queue_depth":0, "waiting":{"total":0,"pre_backend":0,"transport":0,"intake":0,"preparation":0,"staging":0,"storage":0,"backend":0},
        "status": "ready",
        "modalities": ["text"],
        "serving": {
            "bundle_id": "fixture.bundle.v1",
            "package_identity": "a".repeat(64),
            "checkpoint_identity": "b".repeat(64),
            "metadata_identity": "c".repeat(64),
            "startup_nonce": "e".repeat(32),
            "served_model_names": ["primary", "alias"],
            "profile": {
                "name": "fixture-tp4",
                "tensor_parallel_size": 4,
                "context_parallel_size": 1,
                "speculation": {"mode": "adaptive", "tokens": 3, "drafter": "mtp"},
                "kv_cache_dtype": "bf16",
                "prefix_cache": true,
                "max_sequences": 32
            },
            "kv_page_tokens": 64,
            "version": "1.2.3",
            "source_commit": null
        }
    });
    for path in ["/health", "/ping"] {
        let response = server.get(path).await;
        assert_eq!(response.status, 200);
        assert_eq!(response.json(), expected);
    }
}

/// Every response, refusals included, names the process that wrote it, so a
/// client can tie each scored response to the identity `/health` reported.
#[tokio::test]
async fn every_response_carries_the_serving_nonce_and_checkpoint() {
    let nonce = format!(
        "{}: {}",
        runtime_server::STARTUP_NONCE_HEADER,
        "e".repeat(32)
    );
    let checkpoint = format!(
        "{}: {}",
        runtime_server::CHECKPOINT_IDENTITY_HEADER,
        "b".repeat(64)
    );
    let server = InProcessServerV1::start_with_options(
        ServerConfig::default(),
        "reasoning_content",
        runtime_server::HttpOptions {
            serving_identity: Some(fixture_identity()),
            ..Default::default()
        },
    );
    for (method, path, body, status) in [
        ("POST", "/v1/completions", r#"{"prompt":"hello"}"#, 200),
        ("GET", "/v1/models", "", 200),
        ("GET", "/health", "", 200),
        ("POST", "/v1/completions", "{", 400),
        ("GET", "/missing", "", 404),
    ] {
        let response = server.raw(method, path, "", body).await;
        assert_eq!(response.status, status, "{path}: {}", response.body);
        assert!(response.headers.contains(&nonce), "{path}");
        assert!(response.headers.contains(&checkpoint), "{path}");
    }
    let protected = InProcessServerV1::start_with_options(
        ServerConfig::default(),
        "reasoning_content",
        runtime_server::HttpOptions {
            api_key: Some(runtime_server::ApiKey::new("secret")),
            serving_identity: Some(fixture_identity()),
            ..Default::default()
        },
    );
    let response = protected.raw("GET", "/v1/models", "", "").await;
    assert_eq!(response.status, 401);
    assert!(response.headers.contains(&nonce));
    // Without a host-named identity there is nothing to report.
    let response = InProcessServerV1::full().get("/health").await;
    assert!(!response.headers.contains("x-chopinfer-"));
}

#[tokio::test]
async fn served_aliases_replace_packaged_names_without_changing_frontend() {
    let server = InProcessServerV1::start_with_options(
        ServerConfig::default(),
        "reasoning_content",
        runtime_server::HttpOptions {
            served_model_names: vec!["primary".into(), "alias".into()],
            ..Default::default()
        },
    );
    let models = server.get("/v1/models").await.json();
    assert_eq!(models["data"][0]["id"], "primary");
    assert_eq!(models["data"][1]["id"], "alias");
    assert_eq!(models["data"].as_array().unwrap().len(), 2);
    assert_eq!(models["data"][0]["owned_by"], "fixture-owner");
    for name in ["primary", "alias"] {
        let response = server
            .post(
                "/v1/completions",
                &format!(r#"{{"model":"{name}","prompt":"hello"}}"#),
            )
            .await;
        assert_eq!(response.status, 200);
        assert_eq!(response.json()["model"], name);
        let response = server
            .post(
                "/v1/chat/completions",
                &format!(
                    r#"{{"model":"{name}","messages":[{{"role":"user","content":"hello"}}]}}"#
                ),
            )
            .await;
        assert_eq!(response.status, 200);
        assert_eq!(response.json()["model"], name);
        assert_eq!(
            server
                .post(
                    "/tokenize",
                    &format!(r#"{{"model":"{name}","prompt":"hello"}}"#)
                )
                .await
                .status,
            200
        );
    }
    let response = server
        .post("/v1/completions", r#"{"prompt":"hello"}"#)
        .await;
    assert_eq!(response.json()["model"], "primary");
    assert_eq!(
        server
            .post(
                "/v1/completions",
                &format!(r#"{{"model":"{MODEL}","prompt":"hello"}}"#)
            )
            .await
            .status,
        404
    );
    assert_eq!(server.frontend.models()[0].id, MODEL);
}

#[tokio::test]
async fn prompt_truncation_keeps_requested_tokens_for_text_chat_and_batches() {
    let server = InProcessServerV1::full();
    server
        .backend
        .tokenize_prompts
        .store(true, Ordering::Relaxed);
    for (prompt, side, expected) in [
        (r#""abcdef""#, "left", vec![100, 101, 102]),
        ("[1,2,3,4,5]", "right", vec![1, 2, 3]),
    ] {
        let response = server.post("/v1/completions", &format!(r#"{{"prompt":{prompt},"truncate_prompt_tokens":3,"truncation_side":"{side}","max_tokens":1}}"#)).await;
        assert_eq!(response.status, 200, "{}", response.body);
        assert_eq!(response.json()["usage"]["prompt_tokens"], 3);
        assert_eq!(
            server.backend.prompts().last(),
            Some(&PreparedPrompt::TokenIds(expected))
        );
    }
    let response = server.post("/v1/chat/completions", r#"{"messages":[{"role":"user","content":"abcdef"}],"truncate_prompt_tokens":3,"max_completion_tokens":1,"stream":true,"stream_options":{"include_usage":true}}"#).await;
    assert_eq!(response.status, 200, "{}", response.body);
    assert!(response.body.contains("\"prompt_tokens\":3"));
    assert_eq!(
        server.backend.prompts().last(),
        Some(&PreparedPrompt::TokenIds(vec![100, 101, 102]))
    );
    let response = server
        .post(
            "/v1/completions",
            r#"{"prompt":[[1,2,3],[4,5,6]],"truncate_prompt_tokens":2,"max_tokens":1}"#,
        )
        .await;
    assert_eq!(response.status, 200);
    assert_eq!(response.json()["usage"]["prompt_tokens"], 4);
    let prompts = server.backend.prompts();
    assert_eq!(
        &prompts[prompts.len() - 2..],
        &[
            PreparedPrompt::TokenIds(vec![2, 3]),
            PreparedPrompt::TokenIds(vec![5, 6])
        ]
    );
}

#[tokio::test]
async fn prompt_truncation_enforces_effective_context_and_explicit_refusals() {
    let server = InProcessServerV1::start(ServerConfig::default().with_maximum_context_tokens(6));
    server
        .backend
        .tokenize_prompts
        .store(true, Ordering::Relaxed);
    let response = server
        .post(
            "/v1/completions",
            r#"{"prompt":[1,2,3,4,5,6,7,8],"truncate_prompt_tokens":-1,"max_tokens":1}"#,
        )
        .await;
    assert_eq!(response.status, 200, "{}", response.body);
    assert_eq!(
        server.backend.prompts().last(),
        Some(&PreparedPrompt::TokenIds(vec![4, 5, 6, 7, 8]))
    );
    assert_eq!(response.json()["usage"]["prompt_tokens"], 5);
    for fields in [
        r#""truncate_prompt_tokens":-2"#,
        r#""truncate_prompt_tokens":0"#,
        r#""truncate_prompt_tokens":6"#,
        r#""truncate_prompt_tokens":1.5"#,
        r#""truncation_side":"middle""#,
        r#""allowed_token_ids":[1,2]"#,
        r#""model_options":{"budget":8}"#,
        r#""bad_words":["bad"]"#,
    ] {
        let before = server.backend.submissions();
        let response = server
            .post(
                "/v1/completions",
                &format!(r#"{{"prompt":"abc","max_tokens":1,{fields}}}"#),
            )
            .await;
        assert_eq!(response.status, 400, "{fields}: {}", response.body);
        assert_eq!(server.backend.submissions(), before);
    }
    let response = server
        .post(
            "/v1/completions",
            r#"{"prompt":"abc","truncate_prompt_tokens":null,"max_tokens":1}"#,
        )
        .await;
    assert_eq!(response.status, 200);
}

#[tokio::test]
async fn request_id_opt_in_preserves_client_trace_identity() {
    for enabled in [false, true] {
        let server = InProcessServerV1::start_with_options(
            ServerConfig::default(),
            "reasoning_content",
            runtime_server::HttpOptions {
                enable_request_id_headers: enabled,
                ..Default::default()
            },
        );
        for id in ["proxy-id".to_owned(), "a".repeat(600)] {
            let response = server
                .raw(
                    "POST",
                    "/v1/completions",
                    &format!("X-Request-Id: {id}\r\nX-Client-Request-Id: client-id\r\n"),
                    r#"{"prompt":"hello"}"#,
                )
                .await;
            assert_eq!(response.status, 200, "{}", response.body);
            assert_eq!(
                server
                    .backend
                    .prompts
                    .lock()
                    .unwrap()
                    .last()
                    .unwrap()
                    .client_request_id
                    .as_deref(),
                Some("client-id")
            );
        }
    }
}

#[tokio::test]
async fn invalid_request_ids_are_replaced_consistently() {
    let server = InProcessServerV1::start_with_options(
        ServerConfig::default(),
        "reasoning_content",
        runtime_server::HttpOptions {
            enable_request_id_headers: true,
            ..Default::default()
        },
    );
    for id in ["a".repeat(600), "a\tb".to_owned(), String::new()] {
        let response = server
            .raw(
                "POST",
                "/v1/completions",
                &format!("X-Request-Id: {id}\r\n"),
                r#"{"prompt":"hello"}"#,
            )
            .await;
        assert_eq!(response.status, 200, "{}", response.body);
        let echoed = response
            .headers
            .lines()
            .find_map(|line| line.strip_prefix("x-request-id: "))
            .unwrap();
        assert_ne!(echoed, id);
        assert_eq!(
            server
                .backend
                .prompts
                .lock()
                .unwrap()
                .last()
                .unwrap()
                .client_request_id
                .as_deref(),
            Some(echoed)
        );
    }
}

#[tokio::test]
async fn cors_is_off_without_operator_opt_in() {
    let server = InProcessServerV1::full();
    for (method, path, headers, status) in [
        (
            "GET",
            "/v1/models",
            "Origin: https://client.example\r\n",
            200,
        ),
        (
            "OPTIONS",
            "/v1/completions",
            "Origin: https://client.example\r\nAccess-Control-Request-Method: POST\r\n",
            405,
        ),
    ] {
        let response = server.raw(method, path, headers, "").await;
        assert_eq!(response.status, status);
        assert!(!response.headers.contains("access-control-"));
    }
    assert_eq!(server.backend.submissions(), 0);
}

#[tokio::test]
async fn streaming_completions_refuse_unrepresentable_prompt_logprobs() {
    let server = InProcessServerV1::full();
    for echo in [false, true] {
        for prompt in [serde_json::json!("hello"), serde_json::json!([7, 8])] {
            let body = serde_json::json!({"model":MODEL,"prompt":prompt,"stream":true,"prompt_logprobs":2,"echo":echo});
            let response = server.post("/v1/completions", &body.to_string()).await;
            assert_eq!(response.status, 400, "{}", response.body);
            assert_eq!(response.json()["error"]["param"], "prompt_logprobs");
        }
    }
    assert_eq!(server.backend.submissions(), 0);
}

#[tokio::test]
async fn repeated_client_request_ids_do_not_alias_or_exhaust_responses() {
    let server = InProcessServerV1::start_with_options(
        ServerConfig::new(256, 32, 65536, 4096)
            .unwrap()
            .with_responses_store(128, 64 * 1024 * 1024),
        "reasoning_content",
        runtime_server::HttpOptions {
            enable_request_id_headers: true,
            ..Default::default()
        },
    );
    let body = serde_json::json!({"model":MODEL,"input":"hello","max_output_tokens":1}).to_string();
    let mut ids = std::collections::BTreeSet::new();
    for _ in 0..130 {
        let response = server
            .raw("POST", "/v1/responses", "X-Request-Id: retry-id\r\n", &body)
            .await;
        assert_eq!(response.status, 200, "{}", response.body);
        let id = response.json()["id"].as_str().unwrap().to_owned();
        assert!(ids.insert(id.clone()), "duplicate response ID: {id}");
        let stored = server.get(&format!("/v1/responses/{id}")).await;
        assert_eq!(stored.status, 200, "{}", stored.body);
        assert_eq!(stored.json(), response.json());
    }
}

#[tokio::test]
async fn impossible_completion_batch_is_a_client_error() {
    let server = InProcessServerV1::start(ServerConfig::new(3, 32, 4096, 1024).unwrap());
    let body = serde_json::json!({"model":MODEL,"prompt":["one","two"],"n":2,"max_tokens":1});
    let response = server.post("/v1/completions", &body.to_string()).await;
    assert_eq!(response.status, 400, "{}", response.body);
    assert_eq!(server.backend.submissions(), 0);
}

#[tokio::test]
async fn extreme_repetition_penalties_are_client_errors() {
    let server = InProcessServerV1::full();
    for penalty in [1e-308, 0.009, 10.01, 1e308] {
        let body = serde_json::json!({"model":MODEL,"prompt":"hello","repetition_penalty":penalty});
        let response = server.post("/v1/completions", &body.to_string()).await;
        assert_eq!(response.status, 400, "{}", response.body);
        assert_eq!(response.json()["error"]["param"], "repetition_penalty");
    }
    assert_eq!(server.backend.submissions(), 0);
}

#[tokio::test]
async fn local_media_outside_allowlist_does_not_reveal_existence() {
    let server = InProcessServerV1::full();
    server.frontend.media_enabled.store(true, Ordering::Relaxed);
    let root = server.frontend.local_media_root.as_ref().unwrap();
    std::fs::create_dir_all(root).unwrap();
    let existing = std::env::temp_dir().join(format!("media-outside-{}", std::process::id()));
    std::fs::write(&existing, b"exists").unwrap();
    let missing = existing.with_extension("missing");
    let mut errors = Vec::new();
    for path in [&existing, &missing] {
        let body = serde_json::json!({"model":MODEL,"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":format!("file://{}", path.display())}}
        ]}]});
        let response = server.post("/v1/chat/completions", &body.to_string()).await;
        assert_eq!(response.status, 400);
        errors.push(response.json());
    }
    std::fs::remove_file(existing).unwrap();
    assert_eq!(errors[0], errors[1]);
    assert_eq!(server.backend.submissions(), 0);
}

#[tokio::test]
async fn backend_failure_codes_survive_http_and_streaming_errors() {
    let server = InProcessServerV1::full();
    for (code, status) in [
        ("shutdown_timeout", 503),
        ("worker_unavailable", 503),
        ("fixture_failure", 500),
    ] {
        for stream in [false, true] {
            *server.backend.replay.lock().unwrap() = Some(vec![GenerationEvent::Failed {
                message: "generation failed".into(),
                code: Some(code.into()),
            }]);
            let body = serde_json::json!({"model":MODEL,"prompt":"hello","stream":stream});
            let response = server.post("/v1/completions", &body.to_string()).await;
            assert_eq!(
                response.status,
                if stream { 200 } else { status },
                "{}",
                response.body
            );
            if !stream && status == 503 {
                assert!(response.headers.contains("retry-after: 1"));
            }
            let value: Value = if stream {
                serde_json::from_str(
                    response
                        .body
                        .lines()
                        .find_map(|line| {
                            line.strip_prefix("data: ")
                                .filter(|data| data.contains("\"error\""))
                        })
                        .unwrap(),
                )
                .unwrap()
            } else {
                response.json()
            };
            assert_eq!(value["error"]["code"], code);
        }
    }
}

#[tokio::test]
async fn default_queue_holds_more_than_1024_waiting_requests() {
    let server = InProcessServerV1::full();
    server.backend.hold.store(true, Ordering::Relaxed);
    let body = serde_json::json!({"model":MODEL,"prompt":"hello","max_tokens":1}).to_string();
    let running = ServerConfig::default().max_in_flight();
    let clients: Vec<_> = (0..running + 1025)
        .map(|_| server.open("/v1/completions", body.len(), body.as_bytes()))
        .collect();
    timeout(Duration::from_secs(10), async {
        while server.backend.submissions() < running {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        clients.iter().all(|client| client.state().bytes.is_empty()),
        "a waiting request was refused"
    );
    let health = server.get("/health").await.json();
    assert_eq!(health["queue_depth"], 1025);
    assert_eq!(health["waiting"]["backend"], 0);
    let metrics = server.get("/metrics").await;
    assert!(metrics.body.contains("chop_requests_waiting 1025\n"));
    server.backend.hold.store(false, Ordering::Relaxed);
    let held = std::mem::take(&mut *server.backend.held.lock().unwrap());
    for events in held {
        events
            .emit(GenerationEvent::Finished {
                reason: FinishReason::Stop,
                usage: TokenUsage {
                    prompt_tokens: PROMPT_TOKENS as u64,
                    completion_tokens: 0,
                    cached_tokens: Some(0),
                },
            })
            .await
            .unwrap();
    }
    for client in clients {
        let response = answer(&client).await;
        assert_eq!(response.status, 200, "{}", response.body);
    }
    assert_eq!(server.backend.submissions(), running + 1025);
    assert_eq!(server.get("/health").await.json()["queue_depth"], 0);
    assert!(
        server
            .get("/metrics")
            .await
            .body
            .contains("chop_requests_waiting 0\n")
    );
}

/// Two stalled bodies reserve their declared lengths, the whole budget, so a
/// third request waits for room, holding none of its body meanwhile.
#[tokio::test]
async fn unbounded_intake_waits_without_holding_another_body() {
    let config = ServerConfig::new(8, 32, 64 * 1024, 1024)
        .unwrap()
        .with_unbounded_queue()
        .with_intake_budget_bytes(128 * 1024)
        .with_body_read_timeout(Duration::from_millis(500));
    let server = InProcessServerV1::start(config);
    let stalled = [0, 1].map(|_| server.open("/v1/completions", 64 * 1024, &[b' '; 60 * 1024]));
    tokio::time::sleep(Duration::from_millis(100)).await;
    let body = serde_json::json!({"model":MODEL,"prompt":"hello","max_tokens":1}).to_string();
    let waiting = server.open("/v1/completions", body.len(), body.as_bytes());
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(waiting.state().bytes.is_empty());
    assert_eq!(server.backend.tokenizations(), 0);
    let metrics = server.get("/metrics").await;
    assert!(metrics.body.contains("chop_intake_used_bytes 131072\n"));
    for outgoing in &stalled {
        assert_eq!(answer(outgoing).await.status, 408);
    }
    assert_eq!(answer(&waiting).await.status, 200);
}

/// Without a configured body limit, every endpoint reads at most 64 MiB;
/// the media limit defaults to the same, and the ceiling is 1 GiB.
#[test]
fn body_limits_default_to_64_mib_under_a_1_gib_ceiling() {
    let defaults = ServerConfig::default();
    assert_eq!(defaults.max_body_bytes(), 64 << 20);
    assert_eq!(defaults.max_media_body_bytes(), 64 << 20);
    assert_eq!(defaults.intake_budget_bytes(), 1 << 30);
    assert!(ServerConfig::new(1, 1, 1 << 30, 1024).is_ok());
    assert!(ServerConfig::new(1, 1, (1 << 30) + 1, 1024).is_err());
    assert!(defaults.with_max_media_body_bytes((1 << 30) + 1).is_err());
    assert!(defaults.with_max_media_body_bytes(0).is_err());
    // A budget below a body limit is raised to it, so the largest body can
    // still arrive alone.
    let raised = ServerConfig::new(1, 1, 64 * 1024, 1024)
        .unwrap()
        .with_intake_budget_bytes(16 * 1024)
        .with_max_media_body_bytes(320 * 1024)
        .unwrap();
    assert_eq!(raised.intake_budget_bytes(), 320 * 1024);
}

/// A chat body may use the media limit; completions and Responses keep the
/// text limit. A waiting request that declares its length reserves only
/// that, so a large media limit does not make small requests wait.
#[tokio::test]
async fn chat_reads_up_to_the_media_limit_and_small_bodies_reserve_their_length() {
    let config = ServerConfig::new(8, 32, 64 * 1024, 1024)
        .unwrap()
        .with_unbounded_queue()
        .with_max_media_body_bytes(320 * 1024)
        .unwrap()
        .with_body_read_timeout(Duration::from_millis(500));
    let server = InProcessServerV1::start(config);
    // About 100 KiB: unknown fields are ignored, but read.
    let padding = vec!["x".repeat(10 * 1024); 10];
    let chat = serde_json::json!({"model":MODEL,"max_tokens":1,"padding":padding,
        "messages":[{"role":"user","content":"hello"}]});
    let response = server.post("/v1/chat/completions", &chat.to_string()).await;
    assert_eq!(response.status, 200, "{}", response.body);
    // This backend cannot tokenize; refusing for that means the body was read.
    let tokenized = server.post("/tokenize", &chat.to_string()).await;
    assert_eq!(
        tokenized.json()["error"]["message"],
        "tokenization is unavailable",
        "{}",
        tokenized.body
    );
    let completion =
        serde_json::json!({"model":MODEL,"max_tokens":1,"padding":padding,"prompt":"hello"});
    let responses =
        serde_json::json!({"model":MODEL,"max_output_tokens":1,"padding":padding,"input":"hello"});
    let detokenize = serde_json::json!({"model":MODEL,"padding":padding,"tokens":[1]});
    for (path, body) in [
        ("/v1/completions", completion),
        ("/v1/responses", responses),
        ("/detokenize", detokenize),
    ] {
        let refused = server.post(path, &body.to_string()).await;
        assert_eq!(refused.status, 400, "{path}: {}", refused.body);
        assert_eq!(
            refused.json()["error"]["message"],
            "request body exceeds the byte limit",
            "{path}"
        );
    }

    // A stalled chat body reserves its declared 300 KiB of the 320 KiB
    // budget; two small requests still fit beside it and are read at once.
    let stalled = server.open("/v1/chat/completions", 300 * 1024, &[b' '; 1024]);
    tokio::time::sleep(Duration::from_millis(100)).await;
    let small = serde_json::json!({"model":MODEL,"prompt":"hello","max_tokens":1}).to_string();
    let first = server.open("/v1/completions", small.len(), small.as_bytes());
    let second = server.open("/v1/completions", small.len(), small.as_bytes());
    assert_eq!(answer(&first).await.status, 200);
    assert_eq!(answer(&second).await.status, 200);
    assert_eq!(answer(&stalled).await.status, 408);
}

#[tokio::test]
async fn default_queue_waits_for_response_storage() {
    let server = InProcessServerV1::start(
        ServerConfig::new(8, 32, 65536, 1024)
            .unwrap()
            .with_unbounded_queue()
            .with_responses_store(1, 65536),
    );
    server.backend.hold.store(true, Ordering::Relaxed);
    let body = serde_json::json!({"model":MODEL,"input":"hello","max_output_tokens":1}).to_string();
    let first = server.open("/v1/responses", body.len(), body.as_bytes());
    timeout(Duration::from_secs(5), async {
        while server.backend.submissions() == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let second = server.open("/v1/responses", body.len(), body.as_bytes());
    tokio::time::sleep(Duration::from_millis(100)).await;
    let was_waiting = second.state().bytes.is_empty();
    assert_eq!(server.get("/health").await.json()["waiting"]["storage"], 1);
    server.backend.hold.store(false, Ordering::Relaxed);
    let held = std::mem::take(&mut *server.backend.held.lock().unwrap());
    for events in held {
        events
            .emit(GenerationEvent::Finished {
                reason: FinishReason::Stop,
                usage: TokenUsage {
                    prompt_tokens: PROMPT_TOKENS as u64,
                    completion_tokens: 0,
                    cached_tokens: Some(0),
                },
            })
            .await
            .unwrap();
    }
    assert_eq!(answer(&first).await.status, 200);
    assert!(was_waiting, "storage capacity refused the queued request");
    assert_eq!(answer(&second).await.status, 200);
}

#[tokio::test]
async fn explicit_queue_deadline_expires_waiting_but_not_running_requests() {
    let server = InProcessServerV1::start(
        ServerConfig::new(1, 32, 4096, 1024)
            .unwrap()
            .with_unbounded_queue()
            .with_queue_wait(Duration::from_millis(100)),
    );
    server.backend.hold.store(true, Ordering::Relaxed);
    let body = serde_json::json!({"model":MODEL,"prompt":"hello","max_tokens":1}).to_string();
    let running = server.open("/v1/completions", body.len(), body.as_bytes());
    timeout(Duration::from_secs(5), async {
        while server.backend.submissions() == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let waiting = server.post("/v1/completions", &body).await;
    assert_eq!(waiting.status, 429, "{}", waiting.body);
    assert!(waiting.body.contains("queue wait deadline"));
    assert_eq!(server.get("/health").await.json()["queue_depth"], 0);
    assert_eq!(server.backend.submissions(), 1);
    assert!(running.state().bytes.is_empty());
    let held = std::mem::take(&mut *server.backend.held.lock().unwrap());
    for events in held {
        events
            .emit(GenerationEvent::Finished {
                reason: FinishReason::Stop,
                usage: TokenUsage {
                    prompt_tokens: PROMPT_TOKENS as u64,
                    completion_tokens: 0,
                    cached_tokens: Some(0),
                },
            })
            .await
            .unwrap();
    }
    assert_eq!(answer(&running).await.status, 200);
}

#[tokio::test]
async fn unenforced_schema_keywords_return_400_before_admission() {
    let server = InProcessServerV1::full();
    for (keyword, schema) in [
        ("not", serde_json::json!({"not":{"type":"null"}})),
        (
            "minimum",
            serde_json::json!({"type":"number","minimum":0.5}),
        ),
        (
            "minLength",
            serde_json::json!({"type":"string","minLength":5000}),
        ),
        (
            "multipleOf",
            serde_json::json!({"type":"integer","minimum":0,"multipleOf":7}),
        ),
    ] {
        for response_format in [false, true] {
            let mut body =
                serde_json::json!({"messages":[{"role":"user","content":"hi"}],"max_tokens":1});
            if response_format {
                body["response_format"] = serde_json::json!({"type":"json_schema","json_schema":{"name":"result","schema":schema}});
            } else {
                body["structured_outputs"] = serde_json::json!({"json":schema});
            }
            let response = server.post("/v1/chat/completions", &body.to_string()).await;
            assert_eq!(response.status, 400, "{}", response.body);
            assert!(response.body.contains(keyword), "{}", response.body);
            assert!(server.backend.prompts.lock().unwrap().is_empty());
        }
    }
}

#[test]
fn ordered_json_preserves_large_integer_digits() {
    use runtime_server::FrontendJsonV1;
    for input in [
        "18446744073709551617",
        "-9223372036854775809",
        "1234567890123456789012345678901234567890",
    ] {
        let value: FrontendJsonV1 = serde_json::from_str(input).unwrap();
        assert_eq!(
            serde_json::to_string(&value.to_json_value()).unwrap(),
            input
        );
    }
}

#[test]
fn ordered_json_preserves_number_like_object_keys_and_rejects_nonfinite_floats() {
    use runtime_server::FrontendJsonV1;
    use serde::Deserialize;
    let text = r#"{"$serde_json::private::Number":"18446744073709551617"}"#;
    let parsed: FrontendJsonV1 = serde_json::from_str(text).unwrap();
    assert!(matches!(parsed, FrontendJsonV1::Object(_)));
    let value = parsed.to_json_value();
    assert_eq!(
        serde_json::from_value::<FrontendJsonV1>(value.clone()).unwrap(),
        parsed
    );
    assert_eq!(FrontendJsonV1::deserialize(&value).unwrap(), parsed);
    for text in ["NaN", "Infinity", "-Infinity", "1e400", "-1e400"] {
        assert!(
            serde_json::from_str::<FrontendJsonV1>(text).is_err(),
            "{text}"
        );
    }
}
