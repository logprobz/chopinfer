use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use runtime_sampling::SamplingParameters;
use tokio::sync::{mpsc, watch};

use crate::frontend::ChatStateV1;
use crate::frontend::{FrontendMediaPartV1, PromptSegmentV1};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BackendReadiness {
    Ready,
    /// Temporarily unavailable; polling may make the backend ready again.
    NotReady(String),
    /// Failed or stopped; the serving supervisor must terminate.
    Terminated(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdmissionError {
    AtCapacity,
    InvalidPrompt,
    InvalidRequest,
    WorkerUnavailable,
    Unavailable(String),
    Rejected(String),
    /// A client error that names the request parameter at fault.
    InvalidParameter {
        message: String,
        parameter: &'static str,
    },
    /// The prompt takes `prompt_tokens` tokens, more than the caller's
    /// ceiling: found before the costly part of its preparation.
    PromptTooLong {
        prompt_tokens: usize,
    },
}

pub const NO_READY_LOCAL_REPLICAS_REASON: &str = "no local generation replica is ready";
pub const NO_AVAILABLE_LOCAL_REPLICAS_REASON: &str =
    "no local generation replica accepted the request";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalReplicaPoolError;

impl std::fmt::Display for LocalReplicaPoolError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("local replica pool must contain at least one replica")
    }
}

impl std::error::Error for LocalReplicaPoolError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparedPrompt {
    Raw(String),
    RenderedChat(String),
    MediaChat {
        segments: Vec<PromptSegmentV1>,
        media: Vec<FrontendMediaPartV1>,
    },
    /// Tokenized placeholders and preprocessed media, kept together through admission.
    MediaTokens {
        tokens: Vec<u32>,
        media: Vec<MediaInputV1>,
        spans: Vec<MediaSpanV1>,
        /// Grid positions for spans whose modality declares a rule.
        positions: Vec<PositionSpanV1>,
    },
    TokenIds(Vec<u32>),
}

/// The engine media handoff fixed by the media contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaInputV1 {
    pub key: [u8; 32],
    pub encoder: u16,
    pub geometry: [u32; 4],
    pub output_rows: u32,
    pub payload: MediaPayloadV1,
}

/// The owner of a prepared media item's payload: its length, and its bytes
/// written when they are needed. An owner may keep the payload in a smaller
/// form than its bytes, such as an image's 8-bit canvas.
pub trait MediaPayloadSourceV1: Send + Sync {
    /// The payload's length in bytes.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Appends exactly [`Self::len`] bytes to `out`.
    fn write_to(&self, out: &mut Vec<u8>);
}

/// Bytes held as they are.
struct FlatPayloadV1<T>(T);

impl<T: AsRef<[u8]> + Send + Sync> MediaPayloadSourceV1 for FlatPayloadV1<T> {
    fn len(&self) -> usize {
        self.0.as_ref().len()
    }

    fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self.0.as_ref());
    }
}

/// A prepared media item's payload, shared by every copy of the item. The
/// owner (for example bytes held under the host's memory budget) drops with
/// the last copy; the engine's `MediaPayloadV1` can take this one as its
/// owner, so it lives until the engine drops it once prefill passes the item.
#[derive(Clone)]
pub struct MediaPayloadV1(Arc<dyn MediaPayloadSourceV1>);

impl MediaPayloadV1 {
    /// The bytes `owner` holds, kept alive until the last copy drops.
    #[must_use]
    pub fn new(owner: impl AsRef<[u8]> + Send + Sync + 'static) -> Self {
        Self(Arc::new(FlatPayloadV1(owner)))
    }

    /// The payload `source` writes, kept alive until the last copy drops.
    #[must_use]
    pub fn from_source(source: impl MediaPayloadSourceV1 + 'static) -> Self {
        Self(Arc::new(source))
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Appends the payload's [`Self::len`] bytes to `out`.
    pub fn write_to(&self, out: &mut Vec<u8>) {
        self.0.write_to(out);
    }

    /// The payload's bytes, written into a new buffer.
    #[must_use]
    pub fn to_vec(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.len());
        self.write_to(&mut bytes);
        bytes
    }

    /// Whether another copy of this payload is alive, so dropping this one
    /// would not release it.
    #[must_use]
    pub fn is_shared(&self) -> bool {
        Arc::strong_count(&self.0) > 1
    }

    /// Whether both are copies of one payload.
    #[must_use]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl From<Vec<u8>> for MediaPayloadV1 {
    fn from(bytes: Vec<u8>) -> Self {
        Self::new(bytes)
    }
}

impl PartialEq for MediaPayloadV1 {
    fn eq(&self, other: &Self) -> bool {
        self.ptr_eq(other) || (self.len() == other.len() && self.to_vec() == other.to_vec())
    }
}

impl Eq for MediaPayloadV1 {}

impl std::fmt::Debug for MediaPayloadV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "MediaPayloadV1({} bytes)", self.len())
    }
}

/// One contiguous placeholder run in expanded prompt token IDs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaSpanV1 {
    pub media: u16,
    pub prompt_row: u32,
    pub rows: u32,
    pub media_row: u32,
}

/// One media span whose rows take `grid-3axis.v1` positions; it repeats the
/// engine's `PositionSpanV1` field for field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PositionSpanV1 {
    pub prompt_row: u32,
    pub rows: u32,
    /// Temporal groups, then merged blocks along height and width.
    pub grid: [u32; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestedLogprobs {
    pub enabled: bool,
    pub top: usize,
    pub echo: bool,
    pub prompt_top: Option<usize>,
    pub return_token_ids: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PromptTokenLogprob {
    pub position: usize,
    pub token_id: u32,
    pub token: String,
    pub logprob: f64,
    pub top_logprobs: Vec<(u32, TokenLogprob)>,
}

/// What the frontend decided about one rendered chat.
///
/// The transport reads the allowed function names and whether reasoning is
/// legitimate. Everything else the frontend decided rides in `state`, which
/// only that frontend interprets.
#[derive(Clone, Debug)]
pub struct ChatMetadata {
    pub include_reasoning: bool,
    pub allowed_function_names: Vec<String>,
    /// A named tool choice finishes with `stop`, as the reference reports
    /// it, rather than `tool_calls`.
    pub named_tool_choice: bool,
    pub state: Arc<dyn ChatStateV1>,
}

#[derive(Clone, Debug)]
pub struct DecodingOptions {
    pub stream_tool_calls: bool,
    pub stop_token_ids: Vec<u32>,
    pub ignore_eos: bool,
    pub skip_special_tokens: bool,
    /// Byte-level tokenizers concatenate added tokens for either value.
    pub spaces_between_special_tokens: bool,
    pub include_stop_str_in_output: bool,
    /// Draft tokens every decode step of the request verifies, fixed in
    /// place of the speculation policy. Only startup warmup sets it, to
    /// reach every verify graph shape before serving; the HTTP surface never
    /// does.
    pub warmup_draft_tokens: Option<usize>,
}

impl Default for DecodingOptions {
    fn default() -> Self {
        Self {
            stream_tool_calls: false,
            stop_token_ids: Vec::new(),
            ignore_eos: false,
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
            include_stop_str_in_output: false,
            warmup_draft_tokens: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PreparedRequest {
    pub request_id: String,
    pub client_request_id: Option<String>,
    pub model: String,
    pub prompt: PreparedPrompt,
    pub max_new_tokens: usize,
    pub stop: Vec<String>,
    pub decoding: DecodingOptions,
    pub sampling: SamplingParameters,
    pub seed: u64,
    pub logprobs: RequestedLogprobs,
    pub chat: Option<ChatMetadata>,
    /// The grammar the generated text must match; for chat, after the
    /// reasoning section.
    pub constraint: Option<Arc<runtime_grammar::Grammar>>,
    /// Limits prefix-cache reuse to requests with the same salt, as vLLM's
    /// `cache_salt` does.
    pub cache_salt: Option<String>,
    pub cancellation: CancellationSignal,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TokenLogprob {
    pub token: String,
    pub bytes: Vec<u8>,
    pub logprob: f64,
}

/// The logprob record of one generated token, whether its text is content,
/// reasoning, tool-call markup or a hidden stop token.
///
/// `top_logprobs` follows vLLM's per-position order: the generated token
/// first, then the requested top alternatives by rank without repeating it,
/// so it holds `top` entries, or `top + 1` when the generated token ranks
/// below them. `token` is the vLLM label and `bytes` its UTF-8 encoding.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratedTokenLogprob {
    pub token_id: u32,
    pub token: String,
    pub bytes: Vec<u8>,
    pub logprob: f64,
    pub top_logprobs: Vec<TokenLogprob>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisibleContent {
    pub delta: String,
    pub generated_token_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReasoningContent {
    pub delta: String,
    pub generated_token_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendToolCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolCallDelta {
    pub index: usize,
    pub name: Option<String>,
    pub arguments: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Prompt tokens resumed from the prefix cache. `None` when the backend
    /// serves without one, which reports zero.
    pub cached_tokens: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GenerationEvent {
    AdmissionFailed(AdmissionError),
    GeneratedTokenIds(Vec<u32>),
    /// Logprob records of newly generated tokens, in generation order, sent
    /// before any text those tokens produce.
    GeneratedTokenLogprobs(Vec<GeneratedTokenLogprob>),
    Prompt {
        token_ids: Vec<u32>,
        text: String,
        tokens: Vec<String>,
    },
    PromptLogprobs(Vec<PromptTokenLogprob>),
    Content(VisibleContent),
    Reasoning(ReasoningContent),
    ToolCallDeltas(Vec<ToolCallDelta>),
    ToolCalls {
        calls: Vec<BackendToolCall>,
        generated_token_count: usize,
    },
    TokenUsageDelta {
        generated_token_count: usize,
    },
    Finished {
        reason: FinishReason,
        usage: TokenUsage,
    },
    Failed {
        message: String,
        code: Option<String>,
    },
}

#[derive(Clone, Debug)]
pub struct CancellationSignal {
    inner: Arc<CancellationInner>,
}

#[derive(Debug)]
struct CancellationInner {
    cancelled: AtomicBool,
    sender: watch::Sender<bool>,
}

impl Default for CancellationSignal {
    fn default() -> Self {
        Self::new()
    }
}

impl CancellationSignal {
    #[must_use]
    pub fn new() -> Self {
        let (sender, _) = watch::channel(false);
        Self {
            inner: Arc::new(CancellationInner {
                cancelled: AtomicBool::new(false),
                sender,
            }),
        }
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Acquire)
    }

    pub async fn cancelled(&self) {
        let mut receiver = self.inner.sender.subscribe();
        while !*receiver.borrow_and_update() {
            if receiver.changed().await.is_err() {
                return;
            }
        }
    }

    /// Cancels this request, which a backend may do when its own delivery
    /// path breaks.
    pub fn cancel(&self) {
        if !self.inner.cancelled.swap(true, Ordering::AcqRel) {
            self.inner.sender.send_replace(true);
        }
    }
}

#[derive(Clone, Debug)]
pub struct EventSender {
    sender: mpsc::Sender<GenerationEvent>,
}

impl EventSender {
    #[must_use]
    pub fn new(sender: mpsc::Sender<GenerationEvent>) -> Self {
        Self { sender }
    }

    pub async fn emit(
        &self,
        event: GenerationEvent,
    ) -> Result<(), mpsc::error::SendError<GenerationEvent>> {
        self.sender.send(event).await
    }

    /// Emits without waiting, so a generation loop never blocks on a slow
    /// reader. A backend outside this crate drives that loop.
    pub fn try_emit(
        &self,
        event: GenerationEvent,
    ) -> Result<(), mpsc::error::TrySendError<GenerationEvent>> {
        self.sender.try_send(event)
    }

    #[must_use]
    /// Whether the reader is gone, so a backend can stop generating.
    pub fn is_closed(&self) -> bool {
        self.sender.is_closed()
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.sender.capacity()
    }
}

pub trait GenerationBackend: Send + Sync + 'static {
    fn readiness(&self) -> BackendReadiness;

    fn maximum_sequence_tokens(&self) -> usize;

    /// Establishes or restores an isolated startup generation session.
    fn startup_warmup(&self, _begin: bool) -> BoxFuture<'_, Result<(), String>> {
        Box::pin(async { Ok(()) })
    }

    /// Maximum simultaneous startup requests for the given token reservation.
    fn startup_wave_capacity(&self, _tokens: usize) -> usize {
        usize::MAX
    }

    fn prompt_token_count(&self, prompt: &PreparedPrompt) -> Result<usize, AdmissionError>;

    /// Prepares a prompt once before admission; backends may return token IDs.
    fn prepare_prompt(
        &self,
        prompt: PreparedPrompt,
    ) -> Result<(PreparedPrompt, usize), AdmissionError> {
        let count = self.prompt_token_count(&prompt)?;
        Ok((prompt, count))
    }

    /// Prepares a prompt for a request whose client may leave while it is
    /// prepared. A backend whose preparation waits on slow work, such as a
    /// video decode, stops waiting once `cancellation` fires.
    fn prepare_prompt_cancellable(
        &self,
        prompt: PreparedPrompt,
        cancellation: &CancellationSignal,
    ) -> Result<(PreparedPrompt, usize), AdmissionError> {
        let _ = cancellation;
        self.prepare_prompt(prompt)
    }

    /// [`Self::prepare_prompt_cancellable`] for a prompt that may take at
    /// most `maximum_prompt_tokens` tokens. A backend that can count a
    /// prompt's tokens before the costly part of its preparation, such as
    /// decoding its images, refuses a longer one with
    /// [`AdmissionError::PromptTooLong`] first. The default prepares the
    /// prompt and leaves the count to the caller.
    fn prepare_prompt_within(
        &self,
        prompt: PreparedPrompt,
        cancellation: &CancellationSignal,
        maximum_prompt_tokens: Option<usize>,
    ) -> Result<(PreparedPrompt, usize), AdmissionError> {
        let _ = maximum_prompt_tokens;
        self.prepare_prompt_cancellable(prompt, cancellation)
    }

    /// Encodes text for the public tokenizer endpoint.
    fn tokenize(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>, AdmissionError> {
        match self.prepare_prompt(PreparedPrompt::Raw(text.to_owned()))?.0 {
            PreparedPrompt::TokenIds(ids) => Ok(ids),
            _ => Err(AdmissionError::Rejected(
                "tokenization is unavailable".to_owned(),
            )),
        }
    }

    /// Looks up vocabulary spellings without decoding the token bytes.
    fn token_strings(&self, _tokens: &[u32]) -> Result<Vec<String>, AdmissionError> {
        Err(AdmissionError::Rejected(
            "token strings are unavailable".to_owned(),
        ))
    }

    fn detokenize(&self, _tokens: &[u32]) -> Result<String, AdmissionError> {
        Err(AdmissionError::Rejected(
            "detokenization is unavailable".to_owned(),
        ))
    }

    /// Attempts to transfer ownership of one request to this backend.
    ///
    /// `Ok(())` means the backend owns the request and its event stream. Any
    /// `Err` means the backend accepted no ownership and emitted no events.
    /// Resolves when execution is admitted. Keep this future pending while
    /// the request waits in a control or scheduler queue; the server counts it.
    fn submit(
        &self,
        request: PreparedRequest,
        events: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>>;

    /// Transfers a request's choices together, with one admission result per
    /// input in the same order. Callers must poll all results concurrently and
    /// drain admitted choices while others wait for scheduler capacity.
    /// Backends with a worker queue should enqueue the group as one command.
    fn submit_batch(
        &self,
        requests: Vec<(PreparedRequest, EventSender)>,
    ) -> Vec<BoxFuture<'_, Result<(), AdmissionError>>> {
        requests
            .into_iter()
            .map(|(request, events)| self.submit(request, events))
            .collect()
    }
}

/// A fixed process-start pool of same-model local generation replicas.
pub struct LocalReplicaPool<B: GenerationBackend + ?Sized> {
    replicas: Vec<Arc<B>>,
    next_start: AtomicUsize,
}

impl<B: GenerationBackend + ?Sized> LocalReplicaPool<B> {
    pub fn new(replicas: Vec<Arc<B>>) -> Result<Self, LocalReplicaPoolError> {
        if replicas.is_empty() {
            return Err(LocalReplicaPoolError);
        }
        Ok(Self {
            replicas,
            next_start: AtomicUsize::new(0),
        })
    }

    #[must_use]
    pub fn replicas(&self) -> &[Arc<B>] {
        &self.replicas
    }
}

impl<B: GenerationBackend + ?Sized> LocalReplicaPool<B> {
    fn submit_remaining(
        &self,
        start: usize,
        initial: Option<AdmissionError>,
        request: PreparedRequest,
        events: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        Box::pin(async move {
            let first = usize::from(initial.is_some());
            let mut ready_count = first;
            let mut full_count = usize::from(matches!(initial, Some(AdmissionError::AtCapacity)));
            for offset in first..self.replicas.len() {
                if request.cancellation.is_cancelled() || events.is_closed() {
                    return Err(AdmissionError::WorkerUnavailable);
                }
                let index = start.wrapping_add(offset) % self.replicas.len();
                let replica = &self.replicas[index];
                if replica.readiness() != BackendReadiness::Ready {
                    continue;
                }
                ready_count += 1;
                match replica.submit(request.clone(), events.clone()).await {
                    Ok(()) => return Ok(()),
                    Err(AdmissionError::AtCapacity) => full_count += 1,
                    Err(AdmissionError::WorkerUnavailable | AdmissionError::Unavailable(_)) => {}
                    Err(error) => return Err(error),
                }
            }
            if ready_count == 0 {
                Err(AdmissionError::Unavailable(
                    NO_READY_LOCAL_REPLICAS_REASON.to_owned(),
                ))
            } else if full_count == ready_count {
                Err(AdmissionError::AtCapacity)
            } else {
                Err(AdmissionError::Unavailable(
                    NO_AVAILABLE_LOCAL_REPLICAS_REASON.to_owned(),
                ))
            }
        })
    }
}

impl<B: GenerationBackend + ?Sized> GenerationBackend for LocalReplicaPool<B> {
    fn readiness(&self) -> BackendReadiness {
        let mut recoverable = false;
        for replica in &self.replicas {
            match replica.readiness() {
                BackendReadiness::Ready => return BackendReadiness::Ready,
                BackendReadiness::NotReady(_) => recoverable = true,
                BackendReadiness::Terminated(_) => {}
            }
        }
        if recoverable {
            BackendReadiness::NotReady(NO_READY_LOCAL_REPLICAS_REASON.to_owned())
        } else {
            BackendReadiness::Terminated(NO_READY_LOCAL_REPLICAS_REASON.to_owned())
        }
    }

    fn maximum_sequence_tokens(&self) -> usize {
        self.replicas
            .iter()
            .map(|replica| replica.maximum_sequence_tokens())
            .min()
            .expect("local replica pool is non-empty")
    }

    fn prompt_token_count(&self, prompt: &PreparedPrompt) -> Result<usize, AdmissionError> {
        let expected = self.replicas[0].prompt_token_count(prompt)?;
        for replica in &self.replicas[1..] {
            if replica.prompt_token_count(prompt)? != expected {
                return Err(AdmissionError::InvalidRequest);
            }
        }
        Ok(expected)
    }

    fn prepare_prompt(
        &self,
        prompt: PreparedPrompt,
    ) -> Result<(PreparedPrompt, usize), AdmissionError> {
        self.prepare_prompt_cancellable(prompt, &CancellationSignal::new())
    }

    fn prepare_prompt_cancellable(
        &self,
        prompt: PreparedPrompt,
        cancellation: &CancellationSignal,
    ) -> Result<(PreparedPrompt, usize), AdmissionError> {
        self.prepare_prompt_within(prompt, cancellation, None)
    }

    fn prepare_prompt_within(
        &self,
        prompt: PreparedPrompt,
        cancellation: &CancellationSignal,
        maximum_prompt_tokens: Option<usize>,
    ) -> Result<(PreparedPrompt, usize), AdmissionError> {
        let (prepared, count) =
            self.replicas[0].prepare_prompt_within(prompt, cancellation, maximum_prompt_tokens)?;
        for replica in &self.replicas[1..] {
            if replica.prompt_token_count(&prepared)? != count {
                return Err(AdmissionError::InvalidRequest);
            }
        }
        Ok((prepared, count))
    }

    fn tokenize(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, AdmissionError> {
        self.replicas[0].tokenize(text, add_special_tokens)
    }

    fn token_strings(&self, tokens: &[u32]) -> Result<Vec<String>, AdmissionError> {
        self.replicas[0].token_strings(tokens)
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, AdmissionError> {
        self.replicas[0].detokenize(tokens)
    }

    fn submit_batch(
        &self,
        requests: Vec<(PreparedRequest, EventSender)>,
    ) -> Vec<BoxFuture<'_, Result<(), AdmissionError>>> {
        // There is no fallback destination, so keep no duplicate prompt or media.
        if self.replicas.len() == 1 && self.replicas[0].readiness() == BackendReadiness::Ready {
            return self.replicas[0]
                .submit_batch(requests)
                .into_iter()
                .map(|admission| {
                    Box::pin(async move {
                        match admission.await {
                            Err(
                                AdmissionError::WorkerUnavailable | AdmissionError::Unavailable(_),
                            ) => Err(AdmissionError::Unavailable(
                                NO_AVAILABLE_LOCAL_REPLICAS_REASON.to_owned(),
                            )),
                            result => result,
                        }
                    }) as BoxFuture<'_, Result<(), AdmissionError>>
                })
                .collect();
        }
        let mut groups: Vec<Vec<_>> = (0..self.replicas.len()).map(|_| Vec::new()).collect();
        let mut results: Vec<Option<BoxFuture<'_, Result<(), AdmissionError>>>> =
            (0..requests.len()).map(|_| None).collect();
        for (index, (request, events)) in requests.into_iter().enumerate() {
            let start = self.next_start.fetch_add(1, Ordering::Relaxed) % self.replicas.len();
            let ready = (0..self.replicas.len())
                .map(|offset| start.wrapping_add(offset) % self.replicas.len())
                .find(|replica| self.replicas[*replica].readiness() == BackendReadiness::Ready);
            if let Some(replica) = ready {
                groups[replica].push((index, request, events));
            } else {
                results[index] = Some(Box::pin(async {
                    Err(AdmissionError::Unavailable(
                        NO_READY_LOCAL_REPLICAS_REASON.to_owned(),
                    ))
                }));
            }
        }
        for (replica_index, (replica, group)) in self.replicas.iter().zip(groups).enumerate() {
            if group.is_empty() {
                continue;
            }
            let requests = group
                .iter()
                .map(|(_, request, events)| (request.clone(), events.clone()))
                .collect();
            for ((index, request, events), admission) in
                group.into_iter().zip(replica.submit_batch(requests))
            {
                results[index] = Some(Box::pin(async move {
                    match admission.await {
                        Err(
                            error @ (AdmissionError::AtCapacity
                            | AdmissionError::WorkerUnavailable
                            | AdmissionError::Unavailable(_)),
                        ) => {
                            self.submit_remaining(replica_index, Some(error), request, events)
                                .await
                        }
                        result => result,
                    }
                }));
            }
        }
        results
            .into_iter()
            .map(|result| result.expect("one result per choice"))
            .collect()
    }

    fn submit(
        &self,
        request: PreparedRequest,
        events: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        let start = self.next_start.fetch_add(1, Ordering::Relaxed) % self.replicas.len();
        self.submit_remaining(start, None, request, events)
    }
}
