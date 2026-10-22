#![forbid(unsafe_code)]

//! The model-independent OpenAI-compatible serving boundary.
//!
//! This crate owns the transport: request shape and parameter validation,
//! routing, admission and in-flight accounting, event buffering, streaming and
//! non-streaming response shapes, usage accounting, and replica fan-out. It is a
//! socket-capable library with no binary, model executor, checkpoint loader, or
//! fallback generation backend.
//!
//! It names no model. A tokenizer, a chat template, a completion grammar, a
//! vocabulary, and a context limit all reach it through
//! [`ServerFrontendV1`], and generation reaches it through
//! [`GenerationBackend`].
//!
//! How much of that surface one deployment admits is [`RequestSurfaceV1`]. A
//! deployment serves what it has evidence for, so the features it cannot
//! account for are refused here rather than rendered by a frontend that would
//! have accepted them.

mod aliases;
mod backend;
mod constraint;
mod context;
mod error;
mod frontend;
mod http_options;
mod identity;
mod intake;
pub use http_options::{ApiKey, CHECKPOINT_IDENTITY_HEADER, HttpOptions, STARTUP_NONCE_HEADER};
pub use identity::{
    ServingIdentity, ServingKvPlane, ServingKvScale, ServingProfile, ServingSpeculation,
};
mod json_limits;
mod lifecycle;
mod metrics;
pub use lifecycle::{serve_with_shutdown, serve_with_shutdown_and_http_options};
pub use runtime_sampling::SamplingDefaults;
mod request;
mod response;
pub use response::ChatEventFoldV1;
mod responses;
mod tokenize;
pub use tokenize::prepare_chat_request_v1;
mod transport;
mod truncation;
mod waiting;

pub use backend::{
    AdmissionError, BackendReadiness, BackendToolCall, BoxFuture, CancellationSignal, ChatMetadata,
    DecodingOptions, EventSender, FinishReason, GeneratedTokenLogprob, GenerationBackend,
    GenerationEvent, LocalReplicaPool, LocalReplicaPoolError, MediaInputV1, MediaPayloadSourceV1,
    MediaPayloadV1, MediaSpanV1, NO_AVAILABLE_LOCAL_REPLICAS_REASON,
    NO_READY_LOCAL_REPLICAS_REASON, PositionSpanV1, PreparedPrompt, PreparedRequest,
    PromptTokenLogprob, ReasoningContent, RequestedLogprobs, TokenLogprob, TokenUsage,
    ToolCallDelta, VisibleContent,
};
pub use frontend::{
    ChatStateV1, FrontendChatRequestV1, FrontendContentPartV1, FrontendErrorV1, FrontendJsonV1,
    FrontendMediaPartKindV1, FrontendMediaPartV1, FrontendMessageV1, FrontendModelV1,
    FrontendRenderedChatV1, FrontendRoleV1, FrontendToolCallV1, FrontendToolChoiceV1,
    FrontendToolV1, PromptSegmentV1, ServerFrontendV1, forwarded_template_kwargs_v1,
};
pub use metrics::{
    MetricsSnapshotProvider, MonotonicClock, SystemMonotonicClock, metrics_snapshot_provider,
};

use std::fmt;
use std::io;
use std::sync::Arc;

use axum::Router;
use axum::extract::State;
use axum::http::{HeaderValue, header};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use tokio::net::TcpListener;
use tokio::sync::Semaphore;

pub const MAX_IN_FLIGHT_REQUESTS: usize = 4_096;
pub const MAX_EVENT_BUFFER_CAPACITY: usize = 4_096;
/// The largest request body any endpoint may accept.
pub const MAX_REQUEST_BODY_BYTES: usize = 1024 * 1024 * 1024;
/// The body limit of a deployment that sets none, and of every endpoint that
/// carries no media whatever the deployment sets for media.
pub const DEFAULT_REQUEST_BODY_BYTES: usize = 64 * 1024 * 1024;
pub const DEFAULT_REQUEST_STRING_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_REQUEST_STRING_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_GENERATION_OUTPUT_BYTES: usize = 64 * 1024 * 1024;
pub const DEFAULT_GENERATION_OUTPUT_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_REQUEST_HEADER_BYTES: usize = 64 * 1024;
/// Request bytes all connections together may hold before admission.
pub const DEFAULT_INTAKE_BUDGET_BYTES: usize = 1024 * 1024 * 1024;
/// How long one request body may take to arrive.
pub const DEFAULT_BODY_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
/// How long a streamed response waits for a client that stopped reading.
pub const DEFAULT_STREAM_WRITE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Which requests this deployment accepts at all.
///
/// `OneToken` is a deliberately narrow admission mode for a deployment that is
/// proving one decode step. Its expected prompt length is a property of the
/// served model, so the caller supplies it rather than the transport assuming
/// one.
///
/// `ExactShape` is the same idea for a deployment that is proving more than one
/// decode step: a prompt that encodes to exactly `prompt_tokens` ids, and a
/// generated-token count the request states explicitly, between one and
/// `maximum_generated_tokens`. Both numbers are counts of tokens, and both are
/// the caller's: this transport measures neither. It is reached through
/// [`ServerConfig::with_exact_shape_acceptance_profile`], which sets the
/// request surface and the context budget with it, because those three are one
/// deployment's evidence rather than three settings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestAcceptanceProfile {
    General,
    ProfiledStreaming {
        name: &'static str,
    },
    OneToken {
        prompt_tokens: usize,
    },
    Bounded {
        maximum_prompt_tokens: usize,
        maximum_generated_tokens: usize,
        maximum_total_tokens: usize,
    },
    ExactShape {
        prompt_tokens: usize,
        maximum_generated_tokens: usize,
    },
}

impl RequestAcceptanceProfile {
    #[must_use]
    pub const fn profile_name(self) -> Option<&'static str> {
        match self {
            Self::ProfiledStreaming { name } => Some(name),
            Self::General
            | Self::OneToken { .. }
            | Self::ExactShape { .. }
            | Self::Bounded { .. } => None,
        }
    }

    #[must_use]
    pub const fn streaming_allowed(self) -> bool {
        !matches!(
            self,
            Self::OneToken { .. } | Self::ExactShape { .. } | Self::Bounded { .. }
        )
    }
}

/// What a deployment serves every request surface this transport validates.
pub const FULL_REQUEST_SURFACE_NAME_V1: &str = "full";
/// What a deployment serves only plain single-turn chat and completions.
pub const SINGLE_TURN_PLAIN_CHAT_SURFACE_NAME_V1: &str = "single-turn-plain-chat";

/// Which conversation features one deployment admits at the transport boundary.
///
/// A deployment serves what it has evidence for, and this is where that
/// decision is enforced rather than left to a model's renderer. [`Self::Full`]
/// is every surface this crate validates. [`Self::SingleTurnPlainChat`] is the
/// narrowest chat surface there is: one user message, optionally preceded by
/// one system message, with no tool definitions, no tool choice, no assistant
/// or tool history, no reasoning controls, no structured output, and no
/// multimodal content. Completions are unchanged under either, because a
/// prompt string is the same request on both.
///
/// Nothing here names a model. A deployment that has measured only plain chat
/// selects the narrow surface; one with evidence for more keeps the full one.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RequestSurfaceV1 {
    #[default]
    Full,
    SingleTurnPlainChat,
}

impl RequestSurfaceV1 {
    /// What a receipt calls this surface.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Full => FULL_REQUEST_SURFACE_NAME_V1,
            Self::SingleTurnPlainChat => SINGLE_TURN_PLAIN_CHAT_SURFACE_NAME_V1,
        }
    }

    /// Whether a chat request may declare tools or choose between them.
    #[must_use]
    pub const fn tools_admitted(self) -> bool {
        matches!(self, Self::Full)
    }

    /// Whether a chat request may carry a conversation of more than one turn.
    ///
    /// A surface that admits no history admits no assistant or tool message
    /// either, because those roles exist only in a conversation that continued.
    #[must_use]
    pub const fn chat_history_admitted(self) -> bool {
        matches!(self, Self::Full)
    }
}

/// Bounded transport settings for one named serving profile.
///
/// These are configured queue bounds. They are not measured throughput or
/// device-capacity results.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServerCapacityProfile {
    name: &'static str,
    max_in_flight: usize,
    event_buffer_capacity: usize,
}

impl ServerCapacityProfile {
    pub const fn new(
        name: &'static str,
        max_in_flight: usize,
        event_buffer_capacity: usize,
    ) -> Result<Self, ServerConfigError> {
        if name.is_empty() {
            return Err(ServerConfigError::EmptyCapacityProfileName);
        }
        if max_in_flight == 0 {
            return Err(ServerConfigError::ZeroMaxInFlight);
        }
        if event_buffer_capacity == 0 {
            return Err(ServerConfigError::ZeroEventBufferCapacity);
        }
        if max_in_flight > MAX_IN_FLIGHT_REQUESTS {
            return Err(ServerConfigError::MaxInFlightTooLarge {
                provided: max_in_flight,
                maximum: MAX_IN_FLIGHT_REQUESTS,
            });
        }
        if event_buffer_capacity > MAX_EVENT_BUFFER_CAPACITY {
            return Err(ServerConfigError::EventBufferCapacityTooLarge {
                provided: event_buffer_capacity,
                maximum: MAX_EVENT_BUFFER_CAPACITY,
            });
        }
        Ok(Self {
            name,
            max_in_flight,
            event_buffer_capacity,
        })
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        self.name
    }

    #[must_use]
    pub const fn max_in_flight(self) -> usize {
        self.max_in_flight
    }

    #[must_use]
    pub const fn event_buffer_capacity(self) -> usize {
        self.event_buffer_capacity
    }

    #[must_use]
    pub const fn streaming_enabled(self) -> bool {
        true
    }
}

/// The one request shape a deployment declared it measured.
///
/// This is kept beside the configured profile rather than only inside it,
/// because it is the fact nothing later undoes: every effective bound this
/// config reports is derived from it while it is set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExactRequestShapeV1 {
    bounded_total_tokens: Option<usize>,
    prompt_tokens: usize,
    maximum_generated_tokens: usize,
}

impl ExactRequestShapeV1 {
    /// Prompt plus generated tokens, checked against overflow when declared.
    const fn total_tokens(self) -> usize {
        match self.bounded_total_tokens {
            Some(total) => total,
            None => self.prompt_tokens + self.maximum_generated_tokens,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ServerConfig {
    wait_for_capacity: bool,
    queue_wait: Option<std::time::Duration>,
    max_in_flight: usize,
    event_buffer_capacity: usize,
    responses_store_entries: usize,
    responses_store_bytes: usize,
    max_body_bytes: usize,
    /// The body limit of the endpoints that carry media (chat completions
    /// and tokenize); `max_body_bytes` unless a deployment sizes it.
    max_media_body_bytes: usize,
    max_string_bytes: usize,
    max_output_bytes: usize,
    /// Request bytes buffered or being prepared across all connections before
    /// admission; never less than one largest body.
    intake_budget_bytes: usize,
    transport_budget_bytes: usize,
    body_read_timeout: std::time::Duration,
    header_read_timeout: std::time::Duration,
    max_connections: Option<usize>,
    stream_write_timeout: std::time::Duration,
    /// The context limit this deployment enforces, if it is stricter than the
    /// frontend's own. `None` uses the frontend's limit unchanged.
    maximum_context_tokens: Option<usize>,
    acceptance_profile: RequestAcceptanceProfile,
    /// Which request surface this deployment admits at all.
    request_surface: RequestSurfaceV1,
    sampling_overrides: SamplingDefaults,
    /// The exact shape this deployment declared, if it declared one.
    ///
    /// Once set, nothing clears it: the accessors below answer from it, so a
    /// later call that would widen the surface, the profile, or the budget back
    /// does not widen what this deployment serves.
    exact_shape: Option<ExactRequestShapeV1>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServerConfigError {
    EmptyCapacityProfileName,
    ZeroMaxInFlight,
    ZeroEventBufferCapacity,
    ZeroMaxBodyBytes,
    ZeroMaxStringBytes,
    ZeroMaxOutputBytes,
    MaxInFlightTooLarge {
        provided: usize,
        maximum: usize,
    },
    EventBufferCapacityTooLarge {
        provided: usize,
        maximum: usize,
    },
    MaxBodyBytesTooLarge {
        provided: usize,
        maximum: usize,
    },
    MaxStringBytesTooLarge {
        provided: usize,
        maximum: usize,
    },
    MaxOutputBytesTooLarge {
        provided: usize,
        maximum: usize,
    },
    InvalidBoundedLimits,
    ZeroExactPromptTokens,
    ZeroMaximumGeneratedTokens,
    ExactShapeBudgetOverflow {
        prompt_tokens: usize,
        maximum_generated_tokens: usize,
    },
}

impl fmt::Display for ServerConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::EmptyCapacityProfileName => "capacity profile name must not be empty",
            Self::ZeroMaxInFlight => "max_in_flight must be greater than zero",
            Self::ZeroEventBufferCapacity => "event_buffer_capacity must be greater than zero",
            Self::ZeroMaxBodyBytes => "max_body_bytes must be greater than zero",
            Self::ZeroMaxStringBytes => "max_string_bytes must be greater than zero",
            Self::ZeroMaxOutputBytes => "max_output_bytes must be greater than zero",
            Self::MaxInFlightTooLarge { .. } => "max_in_flight exceeds the supported maximum",
            Self::EventBufferCapacityTooLarge { .. } => {
                "event_buffer_capacity exceeds the supported maximum"
            }
            Self::MaxBodyBytesTooLarge { .. } => "max_body_bytes exceeds the supported maximum",
            Self::MaxStringBytesTooLarge { .. } => "max_string_bytes exceeds the supported maximum",
            Self::MaxOutputBytesTooLarge { .. } => "max_output_bytes exceeds the supported maximum",
            Self::InvalidBoundedLimits => {
                "bounded serving limits must be positive and fit the total capacity"
            }
            Self::ZeroExactPromptTokens => "an exact prompt length must be at least one token",
            Self::ZeroMaximumGeneratedTokens => {
                "a bounded generation length must allow at least one token"
            }
            Self::ExactShapeBudgetOverflow { .. } => {
                "the prompt and generated token counts overflow a token budget"
            }
        })
    }
}

impl std::error::Error for ServerConfigError {}

impl ServerConfig {
    /// Applies deployment overrides after model defaults and before request values.
    pub fn with_sampling_overrides(
        mut self,
        overrides: SamplingDefaults,
    ) -> Result<Self, runtime_sampling::SamplingError> {
        overrides.validate()?;
        self.sampling_overrides = overrides;
        Ok(self)
    }

    pub const fn new(
        max_in_flight: usize,
        event_buffer_capacity: usize,
        max_body_bytes: usize,
        max_output_bytes: usize,
    ) -> Result<Self, ServerConfigError> {
        if max_in_flight == 0 {
            return Err(ServerConfigError::ZeroMaxInFlight);
        }
        if event_buffer_capacity == 0 {
            return Err(ServerConfigError::ZeroEventBufferCapacity);
        }
        if max_body_bytes == 0 {
            return Err(ServerConfigError::ZeroMaxBodyBytes);
        }
        if max_output_bytes == 0 {
            return Err(ServerConfigError::ZeroMaxOutputBytes);
        }
        if max_in_flight > MAX_IN_FLIGHT_REQUESTS {
            return Err(ServerConfigError::MaxInFlightTooLarge {
                provided: max_in_flight,
                maximum: MAX_IN_FLIGHT_REQUESTS,
            });
        }
        if event_buffer_capacity > MAX_EVENT_BUFFER_CAPACITY {
            return Err(ServerConfigError::EventBufferCapacityTooLarge {
                provided: event_buffer_capacity,
                maximum: MAX_EVENT_BUFFER_CAPACITY,
            });
        }
        if max_body_bytes > MAX_REQUEST_BODY_BYTES {
            return Err(ServerConfigError::MaxBodyBytesTooLarge {
                provided: max_body_bytes,
                maximum: MAX_REQUEST_BODY_BYTES,
            });
        }
        if max_output_bytes > MAX_GENERATION_OUTPUT_BYTES {
            return Err(ServerConfigError::MaxOutputBytesTooLarge {
                provided: max_output_bytes,
                maximum: MAX_GENERATION_OUTPUT_BYTES,
            });
        }
        Ok(Self {
            wait_for_capacity: false,
            queue_wait: None,
            max_in_flight,
            event_buffer_capacity,
            max_body_bytes,
            max_media_body_bytes: max_body_bytes,
            responses_store_entries: 0,
            responses_store_bytes: 0,
            max_string_bytes: DEFAULT_REQUEST_STRING_BYTES,
            max_output_bytes,
            intake_budget_bytes: DEFAULT_INTAKE_BUDGET_BYTES,
            transport_budget_bytes: transport::DEFAULT_BUDGET,
            body_read_timeout: DEFAULT_BODY_READ_TIMEOUT,
            header_read_timeout: std::time::Duration::from_secs(30),
            max_connections: None,
            stream_write_timeout: DEFAULT_STREAM_WRITE_TIMEOUT,
            maximum_context_tokens: None,
            acceptance_profile: RequestAcceptanceProfile::General,
            request_surface: RequestSurfaceV1::Full,
            sampling_overrides: SamplingDefaults::EMPTY,
            exact_shape: None,
        })
    }

    /// Enables an in-memory Responses store with bounded entries and reserved
    /// serialized bytes. Zero in either bound disables storage (the default).
    #[must_use]
    pub const fn with_responses_store(mut self, entries: usize, bytes: usize) -> Self {
        self.responses_store_entries = entries;
        self.responses_store_bytes = bytes;
        self
    }

    /// Sets the body limit of the endpoints that carry media, which a
    /// deployment sizes from the media it accepts.
    pub const fn with_max_media_body_bytes(
        mut self,
        limit: usize,
    ) -> Result<Self, ServerConfigError> {
        if limit == 0 {
            return Err(ServerConfigError::ZeroMaxBodyBytes);
        }
        if limit > MAX_REQUEST_BODY_BYTES {
            return Err(ServerConfigError::MaxBodyBytesTooLarge {
                provided: limit,
                maximum: MAX_REQUEST_BODY_BYTES,
            });
        }
        self.max_media_body_bytes = limit;
        Ok(self)
    }

    #[must_use]
    pub const fn max_media_body_bytes(self) -> usize {
        self.max_media_body_bytes
    }

    pub const fn with_max_string_bytes(mut self, limit: usize) -> Result<Self, ServerConfigError> {
        if limit == 0 {
            return Err(ServerConfigError::ZeroMaxStringBytes);
        }
        if limit > MAX_REQUEST_STRING_BYTES {
            return Err(ServerConfigError::MaxStringBytesTooLarge {
                provided: limit,
                maximum: MAX_REQUEST_STRING_BYTES,
            });
        }
        self.max_string_bytes = limit;
        Ok(self)
    }

    #[must_use]
    pub const fn max_string_bytes(self) -> usize {
        self.max_string_bytes
    }

    /// Serves a stricter context limit than the frontend's own.
    ///
    /// A deployment that declared an exact shape is already bounded by that
    /// shape's own budget, so this narrows it further or does nothing; it never
    /// lengthens it.
    #[must_use]
    pub const fn with_maximum_context_tokens(mut self, maximum_context_tokens: usize) -> Self {
        self.maximum_context_tokens = Some(maximum_context_tokens);
        self
    }

    /// Waits for internal staging capacity without bounding the waiting queue.
    #[must_use]
    pub const fn with_unbounded_queue(mut self) -> Self {
        self.wait_for_capacity = true;
        self
    }

    #[must_use]
    pub const fn has_unbounded_queue(self) -> bool {
        self.wait_for_capacity
    }

    /// Sets an optional deadline from request arrival through backend admission.
    #[must_use]
    pub const fn with_queue_wait(mut self, timeout: std::time::Duration) -> Self {
        self.queue_wait = if timeout.is_zero() {
            None
        } else {
            Some(timeout)
        };
        self
    }

    /// Applies bounded admission and event delivery for a named profile.
    ///
    /// The queue bounds are this profile's. The acceptance profile is too,
    /// unless this deployment already declared an exact request shape, which
    /// nothing here widens back.
    #[must_use]
    pub const fn with_capacity_profile(mut self, profile: ServerCapacityProfile) -> Self {
        self.wait_for_capacity = false;
        self.max_in_flight = profile.max_in_flight;
        self.event_buffer_capacity = profile.event_buffer_capacity;
        self.acceptance_profile =
            RequestAcceptanceProfile::ProfiledStreaming { name: profile.name };
        self
    }

    /// Admits only the named request surface.
    ///
    /// This narrows what a request may contain; it does not change what the
    /// frontend renders or what the backend runs. A deployment that has
    /// evidence for one surface serves that one and refuses the rest at the
    /// transport boundary, where dropping the requirement would otherwise
    /// expose behaviour nothing validated.
    ///
    /// A deployment that declared an exact request shape declared the surface
    /// that shape was measured on with it, and naming a wider one here does not
    /// widen it.
    #[must_use]
    pub const fn with_request_surface(mut self, surface: RequestSurfaceV1) -> Self {
        self.request_surface = surface;
        self
    }

    /// Accepts only single-token completions of the given prompt length.
    ///
    /// A deployment that already declared an exact request shape keeps it.
    #[must_use]
    pub const fn with_one_token_acceptance_profile(mut self, prompt_tokens: usize) -> Self {
        self.max_in_flight = 1;
        self.acceptance_profile = RequestAcceptanceProfile::OneToken { prompt_tokens };
        self
    }

    /// Accepts only plain single-turn requests of exactly one token shape.
    ///
    /// Three facts are set here together because a deployment has evidence for
    /// them together: a prompt that encodes to exactly `prompt_tokens` ids, a
    /// generated-token count the request states and that is at most
    /// `maximum_generated_tokens`, and the conversation surface those two were
    /// measured on. The context budget follows from the first two, in the same
    /// unit both are counted in: a request may charge at most
    /// `prompt_tokens + maximum_generated_tokens` tokens against this
    /// deployment, and the acceptance profile then refuses every division of
    /// that budget except the one shape.
    ///
    /// Setting them in one call is the point, and so is that no later call
    /// takes any of them back. A deployment that could widen the surface,
    /// replace the profile, or lengthen the budget afterwards would serve a
    /// request shape nothing validated, so a shape declared here is what every
    /// bound below answers with from then on: a wider surface, another
    /// acceptance profile, or a longer context named later is not applied.
    /// Narrowing still works, because serving less than was measured is always
    /// allowed.
    pub const fn with_exact_shape_acceptance_profile(
        mut self,
        prompt_tokens: usize,
        maximum_generated_tokens: usize,
    ) -> Result<Self, ServerConfigError> {
        if prompt_tokens == 0 {
            return Err(ServerConfigError::ZeroExactPromptTokens);
        }
        if maximum_generated_tokens == 0 {
            return Err(ServerConfigError::ZeroMaximumGeneratedTokens);
        }
        let Some(total_tokens) = prompt_tokens.checked_add(maximum_generated_tokens) else {
            return Err(ServerConfigError::ExactShapeBudgetOverflow {
                prompt_tokens,
                maximum_generated_tokens,
            });
        };
        self.maximum_context_tokens = Some(total_tokens);
        self.acceptance_profile = RequestAcceptanceProfile::ExactShape {
            prompt_tokens,
            maximum_generated_tokens,
        };
        self.request_surface = RequestSurfaceV1::SingleTurnPlainChat;
        self.exact_shape = Some(ExactRequestShapeV1 {
            bounded_total_tokens: None,
            prompt_tokens,
            maximum_generated_tokens,
        });
        Ok(self)
    }

    /// Declare positive prompt and generation bounds with a shared total capacity.
    pub const fn with_bounded_acceptance_profile(
        mut self,
        maximum_prompt_tokens: usize,
        maximum_generated_tokens: usize,
        maximum_total_tokens: usize,
    ) -> Result<Self, ServerConfigError> {
        if maximum_prompt_tokens == 0
            || maximum_generated_tokens == 0
            || maximum_total_tokens < 2
            || maximum_prompt_tokens >= maximum_total_tokens
            || maximum_generated_tokens >= maximum_total_tokens
        {
            return Err(ServerConfigError::InvalidBoundedLimits);
        }
        self.maximum_context_tokens = Some(maximum_total_tokens);
        self.request_surface = RequestSurfaceV1::SingleTurnPlainChat;
        self.exact_shape = Some(ExactRequestShapeV1 {
            bounded_total_tokens: Some(maximum_total_tokens),
            prompt_tokens: maximum_prompt_tokens,
            maximum_generated_tokens,
        });
        Ok(self)
    }

    #[must_use]
    pub const fn max_in_flight(self) -> usize {
        self.max_in_flight
    }

    #[must_use]
    pub const fn event_buffer_capacity(self) -> usize {
        self.event_buffer_capacity
    }

    #[must_use]
    pub const fn max_body_bytes(self) -> usize {
        self.max_body_bytes
    }

    #[must_use]
    pub const fn max_output_bytes(self) -> usize {
        self.max_output_bytes
    }

    /// Bounds the request bytes held before admission across all connections.
    /// A budget below a body limit is raised to it, so every admissible body
    /// can still arrive alone.
    #[must_use]
    pub const fn with_intake_budget_bytes(mut self, bytes: usize) -> Self {
        self.intake_budget_bytes = bytes;
        self
    }

    #[must_use]
    pub const fn intake_budget_bytes(self) -> usize {
        let largest = if self.max_media_body_bytes > self.max_body_bytes {
            self.max_media_body_bytes
        } else {
            self.max_body_bytes
        };
        if self.intake_budget_bytes < largest {
            largest
        } else {
            self.intake_budget_bytes
        }
    }

    /// Sets the deadline for each HTTP request head, including keep-alive requests.
    #[must_use]
    pub const fn with_header_read_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.header_read_timeout = timeout;
        self
    }

    /// Bounds accepted connections independently of request intake.
    #[must_use]
    pub const fn with_max_connections(mut self, limit: usize) -> Self {
        self.max_connections = Some(if limit == 0 { 1 } else { limit });
        self
    }

    /// Bounds how long one request body may take to arrive.
    #[must_use]
    pub const fn with_body_read_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.body_read_timeout = timeout;
        self
    }

    #[must_use]
    pub const fn body_read_timeout(self) -> std::time::Duration {
        self.body_read_timeout
    }

    /// Bounds how long a streamed response waits on a client that stopped
    /// reading; the request is then cancelled and its capacity released.
    #[must_use]
    pub const fn with_stream_write_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.stream_write_timeout = timeout;
        self
    }

    #[must_use]
    pub const fn stream_write_timeout(self) -> std::time::Duration {
        self.stream_write_timeout
    }

    /// The context limit this deployment enforces.
    ///
    /// A declared exact shape caps this at its own budget, so a longer limit
    /// named beside it is not served and a shorter one still is.
    #[must_use]
    pub const fn maximum_context_tokens(self) -> Option<usize> {
        let Some(shape) = self.exact_shape else {
            return self.maximum_context_tokens;
        };
        let budget = shape.total_tokens();
        Some(match self.maximum_context_tokens {
            Some(configured) if configured < budget => configured,
            _ => budget,
        })
    }

    /// Which requests this deployment accepts at all.
    ///
    /// A declared exact shape is the answer while it is set, whatever profile
    /// was named afterwards.
    #[must_use]
    pub const fn acceptance_profile(self) -> RequestAcceptanceProfile {
        match self.exact_shape {
            Some(shape) if shape.bounded_total_tokens.is_some() => {
                RequestAcceptanceProfile::Bounded {
                    maximum_prompt_tokens: shape.prompt_tokens,
                    maximum_generated_tokens: shape.maximum_generated_tokens,
                    maximum_total_tokens: shape.total_tokens(),
                }
            }
            Some(shape) => RequestAcceptanceProfile::ExactShape {
                prompt_tokens: shape.prompt_tokens,
                maximum_generated_tokens: shape.maximum_generated_tokens,
            },
            None => self.acceptance_profile,
        }
    }

    /// Which conversation features this deployment admits.
    ///
    /// A declared exact shape carries the surface it was measured on, so that
    /// surface is the answer while the shape is set.
    #[must_use]
    pub const fn request_surface(self) -> RequestSurfaceV1 {
        match self.exact_shape {
            Some(_) => RequestSurfaceV1::SingleTurnPlainChat,
            None => self.request_surface,
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            wait_for_capacity: true,
            queue_wait: None,
            max_in_flight: 256,
            responses_store_entries: 0,
            responses_store_bytes: 0,
            event_buffer_capacity: 32,
            max_body_bytes: DEFAULT_REQUEST_BODY_BYTES,
            max_media_body_bytes: DEFAULT_REQUEST_BODY_BYTES,
            max_string_bytes: DEFAULT_REQUEST_STRING_BYTES,
            max_output_bytes: DEFAULT_GENERATION_OUTPUT_BYTES,
            intake_budget_bytes: DEFAULT_INTAKE_BUDGET_BYTES,
            transport_budget_bytes: transport::DEFAULT_BUDGET,
            body_read_timeout: DEFAULT_BODY_READ_TIMEOUT,
            header_read_timeout: std::time::Duration::from_secs(30),
            max_connections: None,
            stream_write_timeout: DEFAULT_STREAM_WRITE_TIMEOUT,
            maximum_context_tokens: None,
            acceptance_profile: RequestAcceptanceProfile::General,
            request_surface: RequestSurfaceV1::Full,
            sampling_overrides: SamplingDefaults::EMPTY,
            exact_shape: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServerBuildError {
    ZeroBackendSequenceLimit,
    ZeroFrontendContextLimit,
    ContextLimitExceedsBackend { server: usize, backend: usize },
}

impl fmt::Display for ServerBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBackendSequenceLimit => {
                formatter.write_str("generation backend sequence limit must be positive")
            }
            Self::ZeroFrontendContextLimit => {
                formatter.write_str("frontend context limit must be positive")
            }
            Self::ContextLimitExceedsBackend { server, backend } => write!(
                formatter,
                "server context limit {server} exceeds generation backend sequence limit {backend}"
            ),
        }
    }
}

impl std::error::Error for ServerBuildError {}

struct ServerState {
    waiting: Arc<waiting::Waiting>,
    transport: Arc<transport::Budget>,
    backend: Arc<dyn GenerationBackend>,
    frontend: Arc<dyn ServerFrontendV1>,
    in_flight: Arc<Semaphore>,
    max_in_flight: usize,
    wait_for_capacity: bool,
    queue_wait: Option<std::time::Duration>,
    drain_expired: CancellationSignal,
    intake: intake::Intake,
    event_buffer_capacity: usize,
    max_body_bytes: usize,
    max_media_body_bytes: usize,
    max_string_bytes: usize,
    max_output_bytes: usize,
    stream_write_timeout: std::time::Duration,
    maximum_context_tokens: usize,
    acceptance_profile: RequestAcceptanceProfile,
    request_surface: RequestSurfaceV1,
    sampling_overrides: SamplingDefaults,
    metrics: metrics::MetricsEndpoint,
    admission_metrics: metrics::AdmissionMetrics,
    responses: responses::Store,
    /// The `serving` member of every health response, when the host named
    /// what it serves.
    serving: Option<serde_json::Value>,
}

pub fn router(
    backend: Arc<dyn GenerationBackend>,
    frontend: Arc<dyn ServerFrontendV1>,
    config: ServerConfig,
) -> Result<Router, ServerBuildError> {
    router_with_metrics(
        backend,
        frontend,
        config,
        Arc::new(SystemMonotonicClock::new()),
        Arc::new(metrics::EmptyMetricsSnapshotProvider),
    )
}

/// Builds a server with an independently supplied metrics snapshot and clock.
pub fn router_with_metrics(
    backend: Arc<dyn GenerationBackend>,
    frontend: Arc<dyn ServerFrontendV1>,
    config: ServerConfig,
    clock: Arc<dyn MonotonicClock>,
    metrics_provider: Arc<dyn MetricsSnapshotProvider>,
) -> Result<Router, ServerBuildError> {
    router_with_options(
        backend,
        frontend,
        config,
        clock,
        metrics_provider,
        HttpOptions::default(),
    )
}

/// Builds the HTTP router with deployment policy.
pub fn router_with_options(
    backend: Arc<dyn GenerationBackend>,
    frontend: Arc<dyn ServerFrontendV1>,
    config: ServerConfig,
    clock: Arc<dyn MonotonicClock>,
    metrics_provider: Arc<dyn MetricsSnapshotProvider>,
    options: HttpOptions,
) -> Result<Router, ServerBuildError> {
    configured_router(backend, frontend, config, clock, metrics_provider, options)
        .map(|(router, _)| router)
}

fn configured_router(
    backend: Arc<dyn GenerationBackend>,
    frontend: Arc<dyn ServerFrontendV1>,
    config: ServerConfig,
    clock: Arc<dyn MonotonicClock>,
    metrics_provider: Arc<dyn MetricsSnapshotProvider>,
    options: HttpOptions,
) -> Result<(Router, Arc<transport::Budget>), ServerBuildError> {
    router_with_lifecycle(
        backend,
        aliases::wrap(frontend, &options.served_model_names),
        config,
        clock,
        metrics_provider,
        Arc::new(Semaphore::new(config.max_in_flight)),
        CancellationSignal::new(),
        options.serving_identity.as_ref(),
    )
    .map(|(router, transport)| (options.layer(router), transport))
}

#[allow(clippy::too_many_arguments)]
fn router_with_lifecycle(
    backend: Arc<dyn GenerationBackend>,
    frontend: Arc<dyn ServerFrontendV1>,
    config: ServerConfig,
    clock: Arc<dyn MonotonicClock>,
    metrics_provider: Arc<dyn MetricsSnapshotProvider>,
    in_flight: Arc<Semaphore>,
    drain_expired: CancellationSignal,
    serving: Option<&ServingIdentity>,
) -> Result<(Router, Arc<transport::Budget>), ServerBuildError> {
    let backend_limit = backend.maximum_sequence_tokens();
    if backend_limit == 0 {
        return Err(ServerBuildError::ZeroBackendSequenceLimit);
    }
    let frontend_limit = frontend.maximum_total_tokens();
    if frontend_limit == 0 {
        return Err(ServerBuildError::ZeroFrontendContextLimit);
    }
    // A deployment may serve less than its model allows, never more. What it
    // configured is read through the accessor rather than off the field,
    // because a declared exact shape bounds it whatever was named afterwards.
    let server_limit = config
        .maximum_context_tokens()
        .map_or(frontend_limit, |configured| configured.min(frontend_limit));
    if server_limit > backend_limit {
        return Err(ServerBuildError::ContextLimitExceedsBackend {
            server: server_limit,
            backend: backend_limit,
        });
    }
    error::initialize_refusal_logger();
    let serving = serving.map(|identity| identity::health_member(identity, frontend.models()));
    let transport = transport::Budget::new(config.transport_budget_bytes);
    let state = Arc::new(ServerState {
        waiting: Arc::clone(&transport.waiting),
        transport: Arc::clone(&transport),
        backend,
        frontend,
        in_flight: Arc::clone(&in_flight),
        max_in_flight: config.max_in_flight,
        wait_for_capacity: config.wait_for_capacity,
        queue_wait: config.queue_wait,
        drain_expired,
        intake: intake::Intake::new(config.intake_budget_bytes(), config.body_read_timeout()),
        event_buffer_capacity: config.event_buffer_capacity,
        max_body_bytes: config.max_body_bytes,
        max_media_body_bytes: config.max_media_body_bytes,
        max_string_bytes: config.max_string_bytes,
        max_output_bytes: config.max_output_bytes,
        stream_write_timeout: config.stream_write_timeout,
        maximum_context_tokens: server_limit,
        acceptance_profile: config.acceptance_profile(),
        request_surface: config.request_surface(),
        sampling_overrides: config.sampling_overrides,
        metrics: metrics::MetricsEndpoint::new(metrics_provider, clock),
        admission_metrics: metrics::AdmissionMetrics::default(),
        responses: responses::Store::new(
            config.responses_store_entries,
            config.responses_store_bytes,
        ),
        serving,
    });
    let router = Router::new()
        .route("/health", get(response::health))
        .route("/ping", get(response::health).post(response::health))
        .route("/version", get(response::version))
        .route("/tokenize", post(tokenize::tokenize))
        .route("/detokenize", post(tokenize::detokenize))
        .route("/metrics", get(metrics_snapshot))
        .route("/debug/step-profile", get(step_profile))
        .route("/v1/models", get(response::models))
        .route("/v1/completions", post(response::completions))
        .route("/v1/chat/completions", post(response::chat_completions))
        .route("/v1/responses", post(response::responses))
        .route("/v1/responses/{response_id}", get(responses::retrieve))
        .route(
            "/v1/responses/{response_id}/cancel",
            post(responses::cancel),
        )
        .with_state(Arc::clone(&state))
        .layer(axum::middleware::from_fn_with_state(
            state,
            lifecycle::admission,
        ));
    Ok((router, transport))
}

async fn step_profile(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let Some(json) = state.metrics.step_profile() else {
        return axum::http::StatusCode::NOT_FOUND.into_response();
    };
    let mut response = json.into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

async fn metrics_snapshot(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let mut text = state.waiting.prometheus();
    text.push_str(&state.transport.prometheus());
    text.push_str(&state.intake.prometheus());
    text.push_str(&state.admission_metrics.prometheus());
    text.push_str(&state.metrics.snapshot());
    if let Some(serving) = &state.serving
        && let Some(options) = serving
            .get("model_options")
            .and_then(serde_json::Value::as_object)
    {
        text.push_str("# TYPE chop_model_option_info gauge\n");
        for (name, value) in options {
            let value = value
                .to_string()
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n");
            text.push_str(&format!(
                "chop_model_option_info{{name=\"{name}\",value=\"{value}\"}} 1\n"
            ));
        }
    }
    let mut response = text.into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

/// Serves with an independently supplied metrics snapshot and clock.
pub async fn serve_with_metrics(
    listener: TcpListener,
    backend: Arc<dyn GenerationBackend>,
    frontend: Arc<dyn ServerFrontendV1>,
    config: ServerConfig,
    clock: Arc<dyn MonotonicClock>,
    metrics_provider: Arc<dyn MetricsSnapshotProvider>,
) -> io::Result<()> {
    let (router, transport) = configured_router(
        backend,
        frontend,
        config,
        clock,
        metrics_provider,
        HttpOptions::default(),
    )
    .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    let _refusal_log_flush = error::RefusalLogFlushGuard;
    transport::serve(listener, router, config, transport, std::future::pending()).await
}

pub async fn serve(
    listener: TcpListener,
    backend: Arc<dyn GenerationBackend>,
    frontend: Arc<dyn ServerFrontendV1>,
    config: ServerConfig,
) -> io::Result<()> {
    serve_with_metrics(
        listener,
        backend,
        frontend,
        config,
        Arc::new(SystemMonotonicClock::new()),
        Arc::new(metrics::EmptyMetricsSnapshotProvider),
    )
    .await
}
