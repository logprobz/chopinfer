#![forbid(unsafe_code)]

mod compatibility;
mod device_results;
#[cfg(test)]
mod host_timing_tests;
pub use device_results::{AcceptedPrefix, AsyncInputDependency, OutputCycle};
pub mod logging;
mod media;
mod metrics;
mod positions;
pub mod profile;
mod prompt_lookup;
mod speculation;
mod step_rows;
pub mod warmup;

pub use media::{
    MAX_MEDIA_ENCODERS, MediaArenaV1, MediaConfigError, MediaConfigV1, MediaEncodeV1,
    MediaInjectV1, MediaInputV1, MediaPayloadSourceV1, MediaPayloadV1, MediaRequestError,
    MediaSpanV1,
};
pub use metrics::{
    EngineMetrics, EngineMetricsSnapshot, ExecutionObservations, MonotonicClock,
    SystemMonotonicClock,
};
pub use positions::{PositionAxesError, PositionAxesV1, PositionSpanV1};
pub use prompt_lookup::{PromptLookupConfig, PromptLookupConfigError, PromptLookupExecutor};
pub use speculation::confidence;
pub use speculation::{
    DraftCostModel, DraftDepths, DraftDepthsError, DraftSource, MAX_DRAFT_DEPTHS, MAX_DRAFT_TOKENS,
    SpeculationConfig, SpeculationConfigError, SpeculationMode, StepCost, StepCosts,
    StepCostsError,
};

use media::{inject_runs, validate_media};
use speculation::{AcceptanceEstimate, PlanCandidate, SpeculationPolicy};

use metrics::BatchComposition;

use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use runtime_grammar::{ConstraintState, GrammarError};
use runtime_kv::{
    BlockId, KvGeometry, KvMetrics, KvStats, PrefixChainV1, PrefixMediaSpanV1, PreparedAppendBatch,
};
use runtime_model_api::ModelVocabulary;
use runtime_sampling::{
    MAX_HISTORY, MAX_STOP_IDS, MAX_VOCABULARY, SampleContext, SamplingError, SamplingParameters,
    SamplingState, TokenLogprob, sample_for_vocabulary,
};
use runtime_scheduler::{
    AdmissionPolicy, Batch, BatchOutcome, BatchPhase, ChunkMedia, DraftBudget, DraftCandidate,
    DraftLimits, PrefixRequestV1, RequestSpec, RequestState, ScheduleError, Scheduler,
    SchedulerCapacityProfile, SchedulerConfig, SubmitError,
};

pub use runtime_scheduler::RequestId;

mod refusal;
pub use refusal::BatchRefusalLimits;

mod offload;

const MAX_EXECUTOR_ERROR_BYTES: usize = 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineConfig {
    scheduler: SchedulerConfig,
    kv_geometry: KvGeometry,
    model_vocabulary: ModelVocabulary,
    maximum_sequence_tokens: usize,
    terminal_retirement_per_step: usize,
    speculation: Option<SpeculationConfig>,
    media: Option<MediaConfigV1>,
    refusal_limits: BatchRefusalLimits,
    position_axes: bool,
    force_host_swap: bool,
}

impl EngineConfig {
    /// The resolved per-request sequence limit used by this engine.
    #[must_use]
    pub const fn maximum_sequence_tokens(&self) -> usize {
        self.maximum_sequence_tokens
    }

    /// Force captured preemptions through host memory for validation.
    #[must_use]
    pub const fn with_forced_swaps(mut self) -> Self {
        self.force_host_swap = true;
        self
    }

    /// Limits safe batch refusals before the worker must restart. A zero limit
    /// stops on the first refusal.
    #[must_use]
    pub const fn with_batch_refusal_limits(mut self, limits: BatchRefusalLimits) -> Self {
        self.refusal_limits = limits;
        self
    }

    /// Accepts media inputs into an arena of this geometry. Without it, the
    /// engine refuses every request that carries media.
    #[must_use]
    pub const fn with_media(mut self, media: MediaConfigV1) -> Self {
        self.scheduler = self.scheduler.with_media_pages(media.arena_pages());
        self.media = Some(media);
        self
    }

    #[must_use]
    pub const fn media(self) -> Option<MediaConfigV1> {
        self.media
    }

    /// Accepts requests with explicit 3-axis positions. Without it, the
    /// engine refuses every request that carries position spans.
    #[must_use]
    pub const fn with_position_axes(mut self) -> Self {
        self.position_axes = true;
        self
    }

    #[must_use]
    pub const fn position_axes(self) -> bool {
        self.position_axes
    }

    /// Enables speculative decoding with these limits.
    #[must_use]
    pub const fn with_speculation(mut self, speculation: SpeculationConfig) -> Self {
        self.speculation = Some(speculation);
        self
    }

    #[must_use]
    pub const fn speculation(self) -> Option<SpeculationConfig> {
        self.speculation
    }

    /// Applies an optional bundle declaration to prefill scheduling and captures.
    pub fn with_guaranteed_prefill_chunks(
        mut self,
        tokens: usize,
    ) -> Result<Self, runtime_scheduler::ConfigError> {
        self.scheduler = self.scheduler.with_guaranteed_prefill_chunks(tokens)?;
        Ok(self)
    }

    /// Enables prefix caching. Each anchor holds a bundle snapshot of
    /// `state_pages` KV pages; the executor must capture and restore them.
    #[must_use]
    pub const fn with_prefix_cache(mut self, state_pages: usize) -> Self {
        self.scheduler = self.scheduler.with_prefix_cache(state_pages);
        self
    }

    #[must_use]
    pub const fn prefix_cache(self) -> Option<usize> {
        self.scheduler.prefix_state_pages()
    }

    /// Logits rows the scheduler allows in a prefill phase that scores
    /// prompt rows.
    #[must_use]
    pub const fn maximum_scored_rows(self) -> usize {
        self.scheduler.maximum_scored_rows()
    }

    /// Selects how the scheduler reserves pages; see admission contract.
    #[must_use]
    pub const fn with_admission(mut self, admission: AdmissionPolicy) -> Self {
        self.scheduler = self.scheduler.with_admission(admission);
        self
    }

    #[must_use]
    pub const fn admission(self) -> AdmissionPolicy {
        self.scheduler.admission()
    }

    #[must_use]
    pub const fn prefill_budget(self) -> runtime_scheduler::PrefillBudget {
        self.scheduler.prefill_budget()
    }

    /// Bind logical page IDs to the physical capacity acquired during bundle create.
    pub fn with_kv_page_capacity(mut self, pages: usize) -> Result<Self, EngineConfigError> {
        let geometry = KvGeometry::new(self.kv_geometry.block_tokens(), pages)
            .map_err(|_| EngineConfigError::InvalidKvPageCapacity)?;
        self.maximum_sequence_tokens = self.maximum_sequence_tokens.min(geometry.token_capacity());
        self.kv_geometry = geometry;
        Ok(self)
    }
    /// Builds one engine configuration for the vocabulary its caller declares.
    ///
    /// There is no default vocabulary. The size belongs to whatever model the
    /// caller is serving, so the engine takes it and never names one itself.
    pub fn new(
        scheduler: SchedulerConfig,
        kv_geometry: KvGeometry,
        model_vocabulary: ModelVocabulary,
        maximum_sequence_tokens: usize,
        terminal_retirement_per_step: usize,
    ) -> Result<Self, EngineConfigError> {
        if model_vocabulary.get() > MAX_VOCABULARY {
            return Err(EngineConfigError::VocabularyTooLarge {
                provided: model_vocabulary.get(),
                maximum: MAX_VOCABULARY,
            });
        }
        if maximum_sequence_tokens == 0 {
            return Err(EngineConfigError::ZeroMaximumSequenceTokens);
        }
        if maximum_sequence_tokens > MAX_HISTORY {
            return Err(EngineConfigError::MaximumSequenceTokensTooLarge {
                provided: maximum_sequence_tokens,
                maximum: MAX_HISTORY,
            });
        }
        if maximum_sequence_tokens > kv_geometry.token_capacity() {
            return Err(EngineConfigError::MaximumSequenceTokensExceedKvCapacity {
                provided: maximum_sequence_tokens,
                maximum: kv_geometry.token_capacity(),
            });
        }
        if terminal_retirement_per_step == 0 {
            return Err(EngineConfigError::ZeroTerminalRetirement);
        }
        Ok(Self {
            scheduler,
            kv_geometry,
            model_vocabulary,
            maximum_sequence_tokens,
            terminal_retirement_per_step,
            speculation: None,
            media: None,
            refusal_limits: BatchRefusalLimits::DEFAULT,
            position_axes: false,
            force_host_swap: false,
        })
    }

    #[must_use]
    pub const fn model_vocabulary(self) -> ModelVocabulary {
        self.model_vocabulary
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EngineRequest {
    id: RequestId,
    prompt_tokens: Vec<u32>,
    maximum_new_tokens: usize,
    sampling: SamplingParameters,
    seed: u64,
    stop_ids: Vec<u32>,
    termination_stop_ids: Vec<u32>,
    model_vocabulary: ModelVocabulary,
    prompt_logprobs: Option<usize>,
    media: Vec<MediaInputV1>,
    spans: Vec<MediaSpanV1>,
    positions: Option<PositionAxesV1>,
    constraint: Option<ConstraintState>,
    cache_salt: Option<Vec<u8>>,
    pinned_draft_tokens: Option<usize>,
}

impl EngineRequest {
    /// Builds one request whose tokens are checked against a declared vocabulary.
    pub fn new(
        id: RequestId,
        prompt_tokens: Vec<u32>,
        maximum_new_tokens: usize,
        sampling: SamplingParameters,
        seed: u64,
        mut stop_ids: Vec<u32>,
        model_vocabulary: ModelVocabulary,
    ) -> Result<Self, EngineRequestError> {
        if prompt_tokens.is_empty() {
            return Err(EngineRequestError::EmptyPrompt);
        }
        if let Some((position, token)) =
            prompt_tokens
                .iter()
                .copied()
                .enumerate()
                .find(|(_, token)| {
                    usize::try_from(*token).map_or(true, |token| token >= model_vocabulary.get())
                })
        {
            return Err(EngineRequestError::PromptTokenOutOfVocabulary {
                position,
                token,
                vocabulary: model_vocabulary.get(),
            });
        }
        if maximum_new_tokens == 0 {
            return Err(EngineRequestError::ZeroMaximumNewTokens);
        }
        let history_limit = prompt_tokens
            .len()
            .checked_add(maximum_new_tokens)
            .ok_or(EngineRequestError::LengthOverflow)?;
        if history_limit > MAX_HISTORY {
            return Err(EngineRequestError::HistoryTooLarge);
        }
        if sampling.minimum_new_tokens() > maximum_new_tokens {
            return Err(EngineRequestError::MinimumExceedsMaximum);
        }
        if stop_ids.len() > MAX_STOP_IDS {
            return Err(EngineRequestError::TooManyStopIds);
        }
        if let Some(token) = stop_ids.iter().copied().find(|token| {
            usize::try_from(*token).map_or(true, |token| token >= model_vocabulary.get())
        }) {
            return Err(EngineRequestError::StopTokenOutOfVocabulary {
                token,
                vocabulary: model_vocabulary.get(),
            });
        }
        stop_ids.sort_unstable();
        if stop_ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(EngineRequestError::DuplicateStopId);
        }
        // A device sampler refuses an out-of-vocabulary bias for the whole
        // batch, so the request is refused here instead.
        if let Some(token) = sampling
            .logit_bias()
            .iter()
            .map(|bias| bias.token_id())
            .find(|token| {
                usize::try_from(*token).map_or(true, |token| token >= model_vocabulary.get())
            })
        {
            return Err(EngineRequestError::LogitBiasTokenOutOfVocabulary {
                token,
                vocabulary: model_vocabulary.get(),
            });
        }
        Ok(Self {
            id,
            prompt_tokens,
            maximum_new_tokens,
            sampling,
            seed,
            termination_stop_ids: stop_ids.clone(),
            stop_ids,
            model_vocabulary,
            prompt_logprobs: None,
            media: Vec::new(),
            spans: Vec::new(),
            positions: None,
            constraint: None,
            cache_salt: None,
            pinned_draft_tokens: None,
        })
    }

    /// Ignores these IDs for generation termination while retaining them for
    /// minimum-length processing. Explicit client stops should not be ignored.
    #[must_use]
    pub fn with_ignored_stop_ids(mut self, ignored: &[u32]) -> Self {
        self.termination_stop_ids.retain(|id| !ignored.contains(id));
        self
    }

    /// Attaches media items and the prompt spans their rows replace.
    ///
    /// Spans must be in prompt order, must not overlap, and must lie inside
    /// both the prompt and their item's output rows. Each item's key becomes
    /// its [`MediaInputV1::identity`].
    pub fn with_media(
        mut self,
        mut media: Vec<MediaInputV1>,
        spans: Vec<MediaSpanV1>,
    ) -> Result<Self, EngineRequestError> {
        validate_media(self.prompt_tokens.len(), &media, &spans)
            .map_err(EngineRequestError::Media)?;
        for item in &mut media {
            item.key = item.identity();
        }
        self.media = media;
        self.spans = spans;
        Ok(self)
    }

    /// Gives the media spans in `spans` grid positions (`grid-3axis.v1`),
    /// after [`Self::with_media`]. Each span must equal one media span. Every
    /// row of the request, generated ones included, then carries explicit
    /// positions; no spans keeps the implied 1D positions.
    pub fn with_position_axes(
        mut self,
        spans: Vec<PositionSpanV1>,
    ) -> Result<Self, EngineRequestError> {
        if spans.is_empty() {
            self.positions = None;
            return Ok(self);
        }
        if spans.iter().any(|span| {
            !self
                .spans
                .iter()
                .any(|media| media.prompt_row == span.prompt_row && media.rows == span.rows)
        }) {
            return Err(EngineRequestError::PositionAxes(
                PositionAxesError::NotMediaSpan,
            ));
        }
        let maximum_rows = self
            .prompt_tokens
            .len()
            .checked_add(self.maximum_new_tokens)
            .ok_or(EngineRequestError::LengthOverflow)?;
        self.positions = Some(
            PositionAxesV1::new(self.prompt_tokens.len(), maximum_rows, &spans)
                .map_err(EngineRequestError::PositionAxes)?,
        );
        Ok(self)
    }

    #[must_use]
    pub fn with_prompt_logprobs(mut self, top: usize) -> Self {
        self.prompt_logprobs = Some(top);
        self
    }

    /// Pins the draft tokens this request verifies in every decode step, in
    /// place of the speculation policy's choice, within the step's draft
    /// budget and the request's own draft limit. Startup warmup uses it to
    /// reach every verify graph shape before serving; client requests never
    /// carry it. Without speculation it has no effect.
    #[must_use]
    pub fn with_pinned_draft_tokens(mut self, tokens: usize) -> Self {
        self.pinned_draft_tokens = Some(tokens);
        self
    }

    /// Salts this request's prefix-cache keys: it shares cached prefixes only
    /// with requests that carry the same salt.
    #[must_use]
    pub fn with_cache_salt(mut self, salt: Vec<u8>) -> Self {
        self.cache_salt = Some(salt);
        self
    }

    /// Constrains every sampled row to the tokens `constraint` admits. Its
    /// masks must span the declared vocabulary.
    pub fn with_constraint(
        mut self,
        constraint: ConstraintState,
    ) -> Result<Self, EngineRequestError> {
        let width = constraint.matcher().grammar().vocabulary().size();
        if width != self.model_vocabulary.get() {
            return Err(EngineRequestError::VocabularyMismatch {
                request: width,
                engine: self.model_vocabulary.get(),
            });
        }
        self.constraint = Some(constraint);
        Ok(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutorError {
    message: String,
    fatal: bool,
    batch_refused: bool,
}

impl ExecutorError {
    pub fn new(message: impl Into<String>) -> Result<Self, ExecutorErrorKind> {
        let message = message.into();
        if message.is_empty() {
            return Err(ExecutorErrorKind::EmptyMessage);
        }
        if message.len() > MAX_EXECUTOR_ERROR_BYTES {
            return Err(ExecutorErrorKind::MessageTooLong);
        }
        Ok(Self {
            message,
            fatal: false,
            batch_refused: false,
        })
    }

    /// A batch refused before device work, with no pending transaction and a
    /// usable executor. All other execute errors must use `new` or `fatal`.
    pub fn batch_refused(message: impl Into<String>) -> Result<Self, ExecutorErrorKind> {
        Self::new(message).map(|error| Self {
            batch_refused: true,
            ..error
        })
    }

    #[must_use]
    pub const fn is_batch_refused(&self) -> bool {
        self.batch_refused
    }

    /// An error after which the executor cannot say its device work on the
    /// call's pages finished. A prefix capture or restore that fails this
    /// way keeps its pages and stops the engine.
    pub fn fatal(message: impl Into<String>) -> Result<Self, ExecutorErrorKind> {
        Self::new(message).map(|error| Self {
            fatal: true,
            ..error
        })
    }

    #[must_use]
    pub const fn is_fatal(&self) -> bool {
        self.fatal
    }

    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for ExecutorError {}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelBatchItem {
    prefill_boundary: Option<PrefillBoundary>,
    output_cycle: Option<OutputCycle>,
    request_id: RequestId,
    phase: BatchPhase,
    input_tokens: Vec<u32>,
    expects_logits: bool,
    prompt_targets: Vec<Option<u32>>,
    prompt_top_logprobs: usize,
    sampling: Option<SamplingWork>,
    draft: Option<DraftWork>,
    media_encodes: Vec<MediaEncodeV1>,
    media_injects: Vec<MediaInjectV1>,
    position_axes: Vec<[i32; 3]>,
    anchor: bool,
}

/// A bundle-negotiated fixed prefill grid and this request's prompt end.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillBoundary {
    pub chunk_tokens: usize,
    pub prompt_tokens: usize,
}

/// The drafts a verify item carries or asks the executor for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DraftWork {
    tokens: usize,
    source: DraftSource,
}

impl DraftWork {
    #[must_use]
    pub const fn new(tokens: usize, source: DraftSource) -> Self {
        Self { tokens, source }
    }

    /// K, the number of drafts the item verifies.
    #[must_use]
    pub const fn tokens(self) -> usize {
        self.tokens
    }

    /// With `DraftSource::Host` the drafts follow the bonus token in the
    /// item's input tokens; with `DraftSource::Bundle` the executor drafts
    /// them and reports them in a `DraftRow`.
    #[must_use]
    pub const fn source(self) -> DraftSource {
        self.source
    }
}

/// Drafts an executor produced for one bundle-drafted verify item.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DraftRow {
    request_id: RequestId,
    tokens: Vec<u32>,
}

impl DraftRow {
    #[must_use]
    pub const fn new(request_id: RequestId, tokens: Vec<u32>) -> Self {
        Self { request_id, tokens }
    }

    #[must_use]
    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

/// How many of a verify item's rows the model keeps at commit.
///
/// `kept_rows` counts the item's inputs whose state survives: the bonus
/// input and each accepted draft. It is between 1 and the draft count plus
/// one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KeptRows {
    pub request_id: RequestId,
    pub kept_rows: usize,
}

/// A request's tokens as one sampled row sees them: its committed history,
/// shared with the engine rather than copied, then the drafts a verify row
/// follows.
///
/// A step's rows all read one history, so cloning a row costs its drafts,
/// not its context.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SamplingHistory {
    committed: Arc<Vec<u32>>,
    drafts: Vec<u32>,
}

impl SamplingHistory {
    #[must_use]
    pub const fn new(committed: Arc<Vec<u32>>) -> Self {
        Self {
            committed,
            drafts: Vec::new(),
        }
    }

    /// The prompt and the tokens the request has emitted.
    #[must_use]
    pub fn committed(&self) -> &[u32] {
        &self.committed
    }

    /// Drafts this row follows, after the committed tokens.
    #[must_use]
    pub fn drafts(&self) -> &[u32] {
        &self.drafts
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.committed.len() + self.drafts.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &u32> {
        self.committed.iter().chain(&self.drafts)
    }

    #[must_use]
    pub fn to_vec(&self) -> Vec<u32> {
        self.iter().copied().collect()
    }

    /// The history followed by `drafts` more tokens.
    #[must_use]
    pub fn with_drafts(&self, drafts: &[u32]) -> Self {
        let mut extended = Vec::with_capacity(self.drafts.len() + drafts.len());
        extended.extend_from_slice(&self.drafts);
        extended.extend_from_slice(drafts);
        Self {
            committed: Arc::clone(&self.committed),
            drafts: extended,
        }
    }

    /// The sampling context of a row with this history.
    #[must_use]
    pub fn context<'a>(&'a self, emitted_tokens: usize, stop_ids: &'a [u32]) -> SampleContext<'a> {
        SampleContext::new(&self.committed, emitted_tokens, stop_ids)
            .with_history_tail(&self.drafts)
    }
}

impl From<Vec<u32>> for SamplingHistory {
    fn from(tokens: Vec<u32>) -> Self {
        Self::new(Arc::new(tokens))
    }
}

/// Request-local sampling inputs copied at the model execution boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingWork {
    pub parameters: SamplingParameters,
    pub state: SamplingState,
    pub history: SamplingHistory,
    pub emitted_tokens: usize,
    pub stop_ids: Vec<u32>,
    /// The row's constraint, for a request that has one.
    pub constraint: Option<RowConstraint>,
}

/// One sampled row's constraint: the tokens it admits and the position it
/// was derived from, so the rows after drafts can be derived too.
#[derive(Clone, Debug, PartialEq)]
pub struct RowConstraint {
    mask: Option<Arc<[u32]>>,
    error: Option<GrammarError>,
    state: ConstraintState,
}

impl RowConstraint {
    /// The row of a request at `state`.
    #[must_use]
    pub fn new(state: ConstraintState) -> Self {
        let words = state.matcher().grammar().vocabulary().mask_words();
        let mut mask = vec![0_u32; words];
        let (mask, error) = match state.fill_mask(&mut mask) {
            Ok(true) => (Some(mask.into()), None),
            Ok(false) => (None, None),
            // Keep the original refusal as well as preventing any sample.
            Err(error) => (Some(vec![0_u32; words].into()), Some(error)),
        };
        Self { mask, error, state }
    }

    /// Why this row failed, including a failure in a speculative fork.
    #[must_use]
    pub fn error(&self) -> Option<&GrammarError> {
        self.state.matcher().failure().or(self.error.as_ref())
    }

    /// Allowed tokens, bit `t % 32` of word `t / 32`; `None` when the row
    /// is unconstrained, before a gate opens or after the grammar ends.
    #[must_use]
    pub fn mask(&self) -> Option<&[u32]> {
        self.mask.as_deref()
    }

    /// Whether the row admits `token`.
    #[must_use]
    pub fn allows(&self, token: u32) -> bool {
        self.mask.as_deref().is_none_or(|mask| {
            mask.get(token as usize / 32)
                .is_some_and(|word| word & (1 << (token % 32)) != 0)
        })
    }

    /// The row after `drafts` are accepted. A draft the constraint refuses
    /// can never be accepted, because the row that would verify it excludes
    /// it; the rows after it keep that row's mask.
    #[must_use]
    pub fn after(&self, drafts: &[u32]) -> Self {
        if drafts.is_empty() {
            return self.clone();
        }
        let mut state = self.state.fork();
        for &token in drafts {
            // Resource failures latch in the request state. Rebuilding below
            // then produces a failed row, not the preceding row's valid mask.
            if !matches!(state.allows(token), Ok(true)) || state.accept(token).is_err() {
                break;
            }
        }
        Self::new(state)
    }
}

impl SamplingWork {
    /// The sampling work of row `row` of a verify item with these drafts.
    ///
    /// Row `row` samples the token after the bonus input and the first `row`
    /// drafts, so it sees those drafts in its history, counts them as
    /// emitted, and draws from the random stream `row` tokens later. When
    /// those drafts are accepted, this is exactly the work the step that
    /// emits the row's token would have had without speculation.
    #[must_use]
    pub fn for_verify_row(&self, drafts: &[u32], row: usize) -> Self {
        Self {
            parameters: self.parameters.clone(),
            state: self.state.after(row as u64),
            history: self.history.with_drafts(&drafts[..row]),
            emitted_tokens: self.emitted_tokens + row,
            stop_ids: self.stop_ids.clone(),
            constraint: self
                .constraint
                .as_ref()
                .map(|constraint| constraint.after(&drafts[..row])),
        }
    }
}

impl ModelBatchItem {
    #[must_use]
    pub const fn prefill_boundary(&self) -> Option<PrefillBoundary> {
        self.prefill_boundary
    }

    #[must_use]
    pub const fn with_prefill_boundary(mut self, boundary: PrefillBoundary) -> Self {
        self.prefill_boundary = Some(boundary);
        self
    }

    /// Present only when the executor opts into checked device results.
    #[must_use]
    pub fn output_cycle(&self) -> Option<&OutputCycle> {
        self.output_cycle.as_ref()
    }

    #[must_use]
    pub fn with_output_cycle(mut self, cycle: OutputCycle) -> Self {
        self.output_cycle = Some(cycle);
        self
    }

    /// Describes one scheduled item of a batch an executor is about to run.
    ///
    /// The engine builds these from its own scheduler decisions. The
    /// constructor is public so an executor adapter can be driven directly, and
    /// it interprets none of the four fields.
    #[must_use]
    pub const fn new(
        request_id: RequestId,
        phase: BatchPhase,
        input_tokens: Vec<u32>,
        expects_logits: bool,
    ) -> Self {
        Self {
            prefill_boundary: None,
            output_cycle: None,
            request_id,
            phase,
            input_tokens,
            expects_logits,
            prompt_targets: Vec::new(),
            prompt_top_logprobs: 0,
            sampling: None,
            draft: None,
            media_encodes: Vec::new(),
            media_injects: Vec::new(),
            position_axes: Vec::new(),
            anchor: false,
        }
    }

    /// Asks the model to keep its state at the prompt-end anchor inside
    /// this prompt-ending prefill item, which the commit captures
    /// ([`runtime_scheduler::BatchItem::anchor_tokens`]).
    #[must_use]
    pub const fn with_anchor(mut self, anchor: bool) -> Self {
        self.anchor = anchor;
        self
    }

    #[must_use]
    pub const fn anchor(&self) -> bool {
        self.anchor
    }

    /// Explicit (T, H, W) positions, one per input row of this phase: the
    /// prefill tokens, the decode row, or every verify row.
    #[must_use]
    pub fn with_position_axes(mut self, axes: Vec<[i32; 3]>) -> Self {
        self.position_axes = axes;
        self
    }

    /// Empty when the item's rows take the implied 1D positions.
    #[must_use]
    pub fn position_axes(&self) -> &[[i32; 3]] {
        &self.position_axes
    }

    #[must_use]
    pub fn with_media(mut self, encodes: Vec<MediaEncodeV1>, injects: Vec<MediaInjectV1>) -> Self {
        self.media_encodes = encodes;
        self.media_injects = injects;
        self
    }

    /// Encoder runs this item's prefill chunk needs before it executes.
    #[must_use]
    pub fn media_encodes(&self) -> &[MediaEncodeV1] {
        &self.media_encodes
    }

    /// This item's input rows that take arena rows.
    #[must_use]
    pub fn media_injects(&self) -> &[MediaInjectV1] {
        &self.media_injects
    }

    #[must_use]
    pub fn with_sampling(mut self, sampling: SamplingWork) -> Self {
        self.sampling = Some(sampling);
        self
    }

    #[must_use]
    pub const fn with_draft(mut self, draft: DraftWork) -> Self {
        self.draft = Some(draft);
        self
    }

    /// Set exactly for verify items.
    #[must_use]
    pub const fn draft(&self) -> Option<&DraftWork> {
        self.draft.as_ref()
    }

    #[must_use]
    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    #[must_use]
    pub const fn phase(&self) -> BatchPhase {
        self.phase
    }

    #[must_use]
    pub fn input_tokens(&self) -> &[u32] {
        &self.input_tokens
    }

    #[must_use]
    pub const fn expects_logits(&self) -> bool {
        self.expects_logits
    }

    #[must_use]
    pub fn prompt_targets(&self) -> &[Option<u32>] {
        &self.prompt_targets
    }

    #[must_use]
    pub const fn prompt_top_logprobs(&self) -> usize {
        self.prompt_top_logprobs
    }

    #[must_use]
    pub fn with_prompt_targets(mut self, targets: Vec<Option<u32>>, top: usize) -> Self {
        self.prompt_targets = targets;
        self.prompt_top_logprobs = top;
        self
    }

    #[must_use]
    pub const fn sampling(&self) -> Option<&SamplingWork> {
        self.sampling.as_ref()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogitsRow {
    request_id: RequestId,
    logits: Vec<f32>,
    scored: Option<(u32, f64, Vec<TokenLogprob>)>,
}

impl LogitsRow {
    pub fn new(request_id: RequestId, logits: Vec<f32>) -> Result<Self, ExecutorError> {
        if logits.is_empty() {
            return Err(ExecutorError::new("executor returned an empty logits row")
                .expect("fixed executor error is valid"));
        }
        if logits.len() > MAX_VOCABULARY {
            return Err(
                ExecutorError::new("executor logits row exceeds the vocabulary limit")
                    .expect("fixed executor error is valid"),
            );
        }
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(ExecutorError::new("executor returned a non-finite logit")
                .expect("fixed executor error is valid"));
        }
        Ok(Self {
            request_id,
            logits,
            scored: None,
        })
    }

    #[must_use]
    pub fn scored_prompt(
        request_id: RequestId,
        token_id: u32,
        logprob: f64,
        top: Vec<TokenLogprob>,
    ) -> Self {
        Self {
            request_id,
            logits: Vec::new(),
            scored: Some((token_id, logprob, top)),
        }
    }

    #[must_use]
    pub fn unscored_prompt(request_id: RequestId) -> Self {
        Self {
            request_id,
            logits: Vec::new(),
            scored: None,
        }
    }

    #[must_use]
    pub fn prompt_score(&self) -> Option<(u32, f64, &[TokenLogprob])> {
        self.scored
            .as_ref()
            .map(|(token, score, top)| (*token, *score, top.as_slice()))
    }

    #[must_use]
    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    #[must_use]
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
}

/// A device executor's selected token and raw-logit logprobs, or the
/// request-local reason it could not sample the row.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledRow {
    request_id: RequestId,
    token_id: u32,
    logprob: f64,
    top_logprobs: Vec<TokenLogprob>,
    failure: Option<SamplingError>,
}

impl SampledRow {
    #[must_use]
    pub const fn new(
        request_id: RequestId,
        token_id: u32,
        logprob: f64,
        top_logprobs: Vec<TokenLogprob>,
    ) -> Self {
        Self {
            request_id,
            token_id,
            logprob,
            top_logprobs,
            failure: None,
        }
    }

    /// A row the sampler could not sample for a reason of its request alone,
    /// such as no token surviving the request's masks. The engine fails
    /// that request as it fails one whose host sampling does, and the step's
    /// other requests keep their rows. Execution and malformed-output
    /// failures are executor errors, never failed rows.
    #[must_use]
    pub const fn failed(request_id: RequestId, error: SamplingError) -> Self {
        Self {
            request_id,
            token_id: 0,
            logprob: 0.0,
            top_logprobs: Vec::new(),
            failure: Some(error),
        }
    }

    #[must_use]
    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }
    #[must_use]
    pub const fn failure(&self) -> Option<SamplingError> {
        self.failure
    }
    #[must_use]
    pub const fn token_id(&self) -> u32 {
        self.token_id
    }
    #[must_use]
    pub const fn logprob(&self) -> f64 {
        self.logprob
    }
    #[must_use]
    pub fn top_logprobs(&self) -> &[TokenLogprob] {
        &self.top_logprobs
    }
}

/// Retains the executor state needed to finalize one model execution.
///
/// The engine calls `commit` only after its scheduler and KV commit succeeds.
/// It calls `rollback` exactly once when a pending transaction is explicitly
/// rolled back or dropped. An error means the model state is uncertain: the
/// engine publishes nothing from that batch and stops executing.
pub trait ModelTransactionState {
    fn commit(&mut self) -> Result<(), ExecutorError>;

    /// Commits a transaction that holds verify items.
    ///
    /// `kept` has one record per verify item, in item order; every other
    /// item commits whole. The default commits everything, which is correct
    /// for an executor that keeps no per-token state of its own.
    fn commit_kept(&mut self, _kept: &[KeptRows]) -> Result<(), ExecutorError> {
        self.commit()
    }

    fn rollback(&mut self) -> Result<(), ExecutorError>;
}

/// Validated logits and the rollback state for one model execution.
pub struct ModelTransaction<S: ModelTransactionState> {
    rows: Vec<LogitsRow>,
    prompt_rows: Vec<LogitsRow>,
    sampled: Vec<SampledRow>,
    drafts: Vec<DraftRow>,
    accepted_prefixes: Vec<AcceptedPrefix>,
    state: Option<S>,
    /// Absent until an executor measures its own execution.
    ///
    /// An executor that measures nothing is not an executor that launched no
    /// kernels, and the difference is the whole value of these counters. It is
    /// kept here rather than flattened to zeros so the metrics side can leave
    /// them unpublished instead of reporting a zero nothing observed.
    observations: Option<ExecutionObservations>,
    /// Prepared target-row capacity reported by the executor, when known.
    row_capacity: Option<usize>,
}

impl<S: ModelTransactionState> ModelTransaction<S> {
    #[must_use]
    pub const fn new(rows: Vec<LogitsRow>, state: S) -> Self {
        Self {
            rows,
            prompt_rows: Vec::new(),
            sampled: Vec::new(),
            drafts: Vec::new(),
            accepted_prefixes: Vec::new(),
            state: Some(state),
            observations: None,
            row_capacity: None,
        }
    }

    #[must_use]
    pub fn logits(&self) -> &[LogitsRow] {
        &self.rows
    }

    #[must_use]
    pub fn prompt_rows(&self) -> &[LogitsRow] {
        &self.prompt_rows
    }

    #[must_use]
    pub fn with_prompt_rows(mut self, rows: Vec<LogitsRow>) -> Self {
        self.prompt_rows = rows;
        self
    }

    #[must_use]
    pub fn sampled(&self) -> &[SampledRow] {
        &self.sampled
    }

    #[must_use]
    pub fn with_sampled(mut self, sampled: Vec<SampledRow>) -> Self {
        self.sampled = sampled;
        self
    }

    #[must_use]
    pub fn drafts(&self) -> &[DraftRow] {
        &self.drafts
    }

    #[must_use]
    pub fn accepted_prefixes(&self) -> &[AcceptedPrefix] {
        &self.accepted_prefixes
    }

    /// Attaches checked device decisions beside all computed sampler rows.
    #[must_use]
    pub fn with_accepted_prefixes(mut self, prefixes: Vec<AcceptedPrefix>) -> Self {
        self.accepted_prefixes = prefixes;
        self
    }

    /// Attaches the drafts of bundle-drafted verify items.
    #[must_use]
    pub fn with_drafts(mut self, drafts: Vec<DraftRow>) -> Self {
        self.drafts = drafts;
        self
    }

    /// Reports the prepared target-row capacity actually used by a pure
    /// decode/verify execution. Leave unknown or multi-program batches unset.
    #[must_use]
    pub const fn with_row_capacity(mut self, capacity: Option<usize>) -> Self {
        self.row_capacity = capacity;
        self
    }

    #[must_use]
    pub const fn row_capacity(&self) -> Option<usize> {
        self.row_capacity
    }

    /// Attaches model-neutral execution counts measured by the executor.
    #[must_use]
    pub const fn with_observations(mut self, observations: ExecutionObservations) -> Self {
        self.observations = Some(observations);
        self
    }

    #[must_use]
    pub const fn observations(&self) -> ExecutionObservations {
        match self.observations {
            Some(observations) => observations,
            None => ExecutionObservations {
                kernel_launches: 0,
                kernel_micros: 0,
                collective_operations: 0,
                collective_micros: 0,
            },
        }
    }

    /// The executor's own measurement, or nothing if it measured none.
    #[must_use]
    pub const fn measured_observations(&self) -> Option<ExecutionObservations> {
        self.observations
    }

    /// Finalization is private to the engine coordinator.
    ///
    /// ```compile_fail
    /// use runtime_engine::{ModelTransaction, NoopModelTransactionState};
    ///
    /// let transaction = ModelTransaction::new(Vec::new(), NoopModelTransactionState);
    /// transaction.commit();
    /// ```
    fn commit(mut self) -> Result<(), ExecutorError> {
        self.state
            .take()
            .expect("pending model transaction has finalizer state")
            .commit()
    }

    fn commit_kept(mut self, kept: &[KeptRows]) -> Result<(), ExecutorError> {
        self.state
            .take()
            .expect("pending model transaction has finalizer state")
            .commit_kept(kept)
    }

    fn rollback(mut self) -> Result<(), ExecutorError> {
        self.rollback_pending()
    }

    fn rollback_pending(&mut self) -> Result<(), ExecutorError> {
        self.state
            .take()
            .map_or(Ok(()), |mut state| state.rollback())
    }
}

impl<S: ModelTransactionState> Drop for ModelTransaction<S> {
    /// Every path that drops a pending transaction fails its step with an
    /// error that stops execution, so a rollback error adds nothing. The one
    /// failure a caller may continue after rolls back explicitly.
    fn drop(&mut self) {
        let _ = self.rollback_pending();
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NoopModelTransactionState;

impl ModelTransactionState for NoopModelTransactionState {
    fn commit(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn rollback(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

impl ModelTransaction<NoopModelTransactionState> {
    #[must_use]
    pub const fn no_op(rows: Vec<LogitsRow>) -> Self {
        Self::new(rows, NoopModelTransactionState)
    }
}

pub use runtime_kv::{KvTransferPoll, KvTransferStatus};
/// A refused submission returns its single-use plan, so the engine can cancel it.
pub type KvTransferSubmitResult = Result<(), (ExecutorError, runtime_kv::KvTransferPlan)>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvOffloadInfo {
    pub host_page_capacity: usize,
    pub minimum_tokens: usize,
    pub snapshot_pages: usize,
}

/// Executes one prepared logical batch and returns an uncommitted transaction.
///
/// Production integrations must implement this trait explicitly. The engine
/// has no fallback executor and makes no model-execution claim. An executor
/// error must leave no pending model state because no transaction is returned.
pub trait ModelExecutor {
    /// Explicit per-executor opt-in. Legacy executors receive no cycle metadata.
    fn device_results_v1(&self) -> bool {
        false
    }

    fn async_input_dependency_v1(&self) -> AsyncInputDependency {
        AsyncInputDependency::HostValues
    }

    /// Whether rollback restores every item's state for a later retry.
    /// In-place executors are never rolled back for one request's sampling
    /// failure: the step commits and only that request fails and retires.
    /// Any rollback they do see retires the whole batch.
    fn rollback_preserves_state_v1(&self) -> bool {
        true
    }

    fn kv_offload_info_v1(&self) -> Option<KvOffloadInfo> {
        None
    }
    /// Submit after HostTier retained both tiers. Submission errors mean no
    /// copy was queued. Poll until every rank completes before releasing pages.
    fn offload_pages_v1(&mut self, plan: runtime_kv::KvTransferPlan) -> KvTransferSubmitResult {
        Err((executor_error_fixed("KV offload is unsupported"), plan))
    }
    fn reload_pages_v1(&mut self, plan: runtime_kv::KvTransferPlan) -> KvTransferSubmitResult {
        Err((executor_error_fixed("KV reload is unsupported"), plan))
    }
    fn poll_kv_transfer_v1(
        &mut self,
        _id: runtime_kv::KvTransferId,
    ) -> Result<KvTransferPoll, ExecutorError> {
        Err(executor_error_fixed("KV transfers are unsupported"))
    }
    fn wait_kv_transfer_v1(
        &mut self,
        _id: runtime_kv::KvTransferId,
    ) -> Result<KvTransferPoll, ExecutorError> {
        Err(executor_error_fixed("KV transfers are unsupported"))
    }
    /// Shutdown polling budget, including pacing of the current copy backlog.
    fn kv_transfer_drain_timeout_v1(&self) -> std::time::Duration {
        std::time::Duration::from_secs(5)
    }
    fn drain_kv_transfers_v1(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }

    /// Reserves per-request state after KV admission, before restore or execute.
    /// Failure must leave no resources owned by this request.
    fn admit_request_v1(&mut self, _id: RequestId) -> Result<(), ExecutorError> {
        Ok(())
    }

    /// Reduces a proposed K to a captured verify-row bucket before scheduling.
    fn round_draft_tokens_v1(&self, proposed: usize) -> usize {
        proposed
    }

    /// The rows a decode or verify step of `rows` rows executes. A captured
    /// graph pads its step to a row bucket, so steps within one bucket cost
    /// about the same and the speculation policy prices them together. The
    /// default executes every row count as it is.
    fn step_rows_v1(&self, rows: usize) -> usize {
        rows
    }

    /// Releases model state after a request becomes terminal.
    ///
    /// The transaction has committed or rolled back before this runs. Queued
    /// cancellations may name requests the executor has never seen. An error
    /// permanently blocks scheduling and execution on this engine.
    fn retire_request_v1(&mut self, _id: RequestId) -> Result<(), ExecutorError> {
        Ok(())
    }

    /// Writes the executor state of `id` after its committed `tokens` into
    /// `pages` of the KV pool.
    ///
    /// The engine calls this only with prefix caching enabled, right after a
    /// prefill commit and before the request retires. The pages belong to no
    /// sequence. An error leaves the executor unchanged and drops the anchor.
    /// A [fatal](ExecutorError::fatal) error instead keeps the anchor's pages
    /// and stops the engine.
    fn capture_prefix_state_v1(
        &mut self,
        _id: RequestId,
        _tokens: usize,
        _pages: &[BlockId],
    ) -> Result<(), ExecutorError> {
        Err(executor_error_fixed("prefix state capture is unsupported"))
    }

    /// Creates the state of a request this executor has not seen, as if it
    /// had committed `prefix`, from pages an earlier capture wrote.
    ///
    /// The engine calls this after admission and before the request's first
    /// execution. An error must leave no state for the request, which then
    /// prefills its whole prompt. A [fatal](ExecutorError::fatal) error
    /// instead keeps the snapshot pages pinned and stops the engine.
    fn restore_prefix_state_v1(
        &mut self,
        _id: RequestId,
        _prefix: &[u32],
        _pages: &[BlockId],
    ) -> Result<(), ExecutorError> {
        Err(executor_error_fixed("prefix state restore is unsupported"))
    }

    /// Proposes up to `maximum` drafts to follow `history`.
    ///
    /// The engine calls this only when speculation uses `DraftSource::Host`,
    /// and verifies as many drafts as it returns. `history` is the request's
    /// prompt and emitted tokens, ending with the token the drafts follow.
    /// The default proposes none.
    fn draft_tokens_v1(
        &mut self,
        _id: RequestId,
        _history: &[u32],
        _maximum: usize,
    ) -> Result<Vec<u32>, ExecutorError> {
        Ok(Vec::new())
    }

    type TransactionState<'a>: ModelTransactionState
    where
        Self: 'a;

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        kv_plan: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    Length,
    Stop,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EngineEventKind {
    /// Prompt tokens the request resumed from the prefix cache. It comes
    /// before the request's first token, and only for a hit.
    PromptCached {
        tokens: usize,
    },
    PromptLogprobs {
        positions: Vec<PromptTokenLogprob>,
    },
    Token {
        token_id: u32,
        logprob: f64,
        top_logprobs: Vec<TokenLogprob>,
    },
    Finished {
        reason: FinishReason,
    },
    Error {
        message: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct PromptTokenLogprob {
    pub position: usize,
    pub token_id: u32,
    pub logprob: f64,
    pub top_logprobs: Vec<TokenLogprob>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EngineEvent {
    sequence: u64,
    request_id: RequestId,
    kind: EngineEventKind,
}

impl EngineEvent {
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    #[must_use]
    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    #[must_use]
    pub const fn kind(&self) -> &EngineEventKind {
        &self.kind
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StepStatus {
    Progressed,
    Idle,
}

#[derive(Clone, Debug)]
struct RequestRecord {
    prompt_tokens: Vec<u32>,
    /// Prefill progress through `history`: the prompt, and after a
    /// preemption also the emitted tokens the request recomputes.
    prompt_offset: usize,
    prompt_logprobs: Option<usize>,
    pending_decode_token: Option<u32>,
    sampling: SamplingParameters,
    sampling_state: SamplingState,
    /// Shared with the sampling work of the step in flight, which is
    /// released before the step's tokens are appended.
    history: Arc<Vec<u32>>,
    emitted_tokens: usize,
    maximum_new_tokens: usize,
    stop_ids: Vec<u32>,
    termination_stop_ids: Vec<u32>,
    submitted_micros: u64,
    last_token_micros: Option<u64>,
    /// Draft acceptance so far; set when speculation is enabled.
    acceptance: Option<AcceptanceEstimate>,
    /// The request verified drafts in its last step, so a drafter that
    /// catches up lazily holds all but that step's kept rows.
    verified_last: bool,
    media: Vec<MediaInputV1>,
    spans: Vec<MediaSpanV1>,
    /// Explicit positions for every row; `None` keeps the implied 1D ones.
    positions: Option<PositionAxesV1>,
    /// Whether the request's prefill has yet to pass each media item, which
    /// the arena counts so that eviction takes items nobody needs first.
    media_wanted: Vec<bool>,
    /// Whether a step of the request has placed each media item before, so
    /// placing it again after an eviction counts as a re-encode.
    media_placed: Vec<bool>,
    /// The prompt row after each media item's last span.
    media_ends: Vec<usize>,
    /// The constraint after the committed tokens.
    constraint: Option<ConstraintState>,
    /// Prompt tokens resumed from the prefix cache at first admission.
    cached_tokens: usize,
    cache_salt: Option<Vec<u8>>,
    /// Draft tokens every decode step verifies, fixed by startup warmup.
    pinned_draft_tokens: Option<usize>,
    /// Scheduled in a batch at least once.
    scheduled: bool,
    preemptions: usize,
    /// Preempted and not yet scheduled again.
    resuming: bool,
}

#[derive(Clone, Debug)]
struct StagedToken {
    token_id: u32,
    logprob: f64,
    top_logprobs: Vec<TokenLogprob>,
}

/// The tokens one item emits in a step: one for a prefill tail or decode
/// item, and the accepted run for a verify item.
#[derive(Clone, Debug)]
struct StagedRun {
    tokens: Vec<StagedToken>,
    state: SamplingState,
    terminal: bool,
    /// Drafts the item verified; zero for items that are not verify items.
    drafted: usize,
    /// Leading drafts that matched the target's samples.
    accepted: usize,
    /// Whether the run ended on a draft the target rejected, rather than on a
    /// stop, the length limit, or the last row.
    rejected: bool,
    /// The constraint after the run's tokens.
    constraint: Option<ConstraintState>,
}

#[derive(Clone, Debug)]
struct PreparedSampleApply {
    tokens: Vec<u32>,
    state: SamplingState,
    constraint: Option<ConstraintState>,
}

#[derive(Clone, Debug)]
struct PreparedRequestApply {
    request_id: RequestId,
    prompt_offset: usize,
    emitted_tokens: usize,
    sample: Option<PreparedSampleApply>,
}

#[derive(Clone, Debug)]
struct PreparedEngineApply {
    requests: Vec<PreparedRequestApply>,
    events: VecDeque<EngineEvent>,
    next_event_sequence: u64,
}

/// Rolls back the pending batch, and the arena changes made for it, unless
/// the batch commits.
struct PendingBatchGuard<'a> {
    scheduler: &'a mut Scheduler,
    arena: Option<&'a mut MediaArenaV1>,
    batch_id: runtime_scheduler::BatchId,
    armed: bool,
}

impl<'a> PendingBatchGuard<'a> {
    const fn new(
        scheduler: &'a mut Scheduler,
        arena: Option<&'a mut MediaArenaV1>,
        batch_id: runtime_scheduler::BatchId,
    ) -> Self {
        Self {
            scheduler,
            arena,
            batch_id,
            armed: true,
        }
    }

    fn scheduler_mut(&mut self) -> &mut Scheduler {
        self.scheduler
    }

    fn arena_mut(&mut self) -> Option<&mut MediaArenaV1> {
        self.arena.as_deref_mut()
    }

    fn rollback(&mut self) -> Result<(), ScheduleError> {
        if self.armed {
            if let Some(arena) = self.arena.as_deref_mut() {
                arena.rollback_step();
            }
            self.scheduler.rollback_batch(self.batch_id)?;
            self.armed = false;
        }
        Ok(())
    }

    fn apply(
        &mut self,
        prepared: runtime_scheduler::PreparedBatchCommit,
    ) -> Result<runtime_scheduler::CommitReport, ScheduleError> {
        let report = self.scheduler.apply_prepared_commit(prepared)?;
        if let Some(arena) = self.arena.as_deref_mut() {
            arena.commit_step();
        }
        self.armed = false;
        Ok(report)
    }
}

impl Drop for PendingBatchGuard<'_> {
    fn drop(&mut self) {
        let _ = self.rollback();
    }
}

/// Host-only startup state. The saved scheduler owns the published KV metrics;
/// its clone detaches those counters while retaining identical admission rules.
/// Scheduler pages and offload tiers are host-side accounting metadata. The
/// executor keeps the existing physical KV buffers and pinned offload arenas.
#[derive(Debug)]
struct StartupCheckpoint {
    host_clock: compatibility::HostClock,
    offload: Option<offload::OffloadState>,
    scheduler: Scheduler,
    speculation: Option<SpeculationPolicy>,
    metrics: EngineMetrics,
    prefix_block_tokens: Option<usize>,
    refusal_guard: refusal::RefusalGuard,
    next_event_sequence: u64,
}

#[derive(Debug)]
pub struct Engine<E: ModelExecutor> {
    startup_checkpoint: Option<StartupCheckpoint>,
    offload: Option<offload::OffloadState>,
    executor: E,
    refusal_guard: refusal::RefusalGuard,
    pending_retirements: Vec<RequestId>,
    retirement_failed: bool,
    /// A prefix capture or restore failed without confirming that its device
    /// work finished, so no page it named may be reused.
    prefix_state_unconfirmed: bool,
    /// A model commit or rollback failed, so executor state is uncertain.
    model_state_failed: bool,
    scheduler: Scheduler,
    requests: BTreeMap<RequestId, RequestRecord>,
    events: VecDeque<EngineEvent>,
    next_event_sequence: u64,
    terminal_retirement_per_step: usize,
    maximum_sequence_tokens: usize,
    model_vocabulary: ModelVocabulary,
    clock: Arc<dyn MonotonicClock>,
    metrics: EngineMetrics,
    speculation: Option<SpeculationPolicy>,
    media: Option<MediaConfigV1>,
    arena: Option<MediaArenaV1>,
    position_axes: bool,
    /// Page tokens of the pool when prefix caching is enabled.
    prefix_block_tokens: Option<usize>,
    /// `CHOPINFER_STEP_TRACE=1`: host timings go into each step's trace line.
    step_trace: bool,
    pending_step_trace: Option<compatibility::StepTrace>,
    /// Never restored by warmup or prefix state, so late outputs cannot alias.
    next_output_cycle: u64,
    host_clock: compatibility::HostClock,
    batch_observer: BatchObserver,
    startup_step_observer: StartupStepObserver,
}

/// Told, before each formed batch executes, whether it prefills prompt
/// tokens (and with them encodes any media they need).
#[derive(Default)]
struct BatchObserver(Option<Box<dyn Fn(bool) + Send>>);

impl fmt::Debug for BatchObserver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(if self.0.is_some() {
            "BatchObserver(set)"
        } else {
            "BatchObserver(unset)"
        })
    }
}

/// Told each step's shape while startup warmup runs.
#[derive(Default)]
struct StartupStepObserver(Option<Box<dyn Fn(warmup::StartupStep) + Send>>);

impl fmt::Debug for StartupStepObserver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(if self.0.is_some() {
            "StartupStepObserver(set)"
        } else {
            "StartupStepObserver(unset)"
        })
    }
}

impl<E: ModelExecutor> Engine<E> {
    /// Calls `observer` with each batch's kind once the step has formed it,
    /// after any preemption, restore and drafting, and before it executes:
    /// true when it prefills prompt tokens, a replay after preemption
    /// included. A caller timing the step learns what it now waits for.
    pub fn observe_batches(&mut self, observer: Box<dyn Fn(bool) + Send>) {
        self.batch_observer = BatchObserver(Some(observer));
    }

    /// Calls `observer` with each committed step's shape and accepted draft
    /// tokens between [`Self::begin_startup_warmup`] and
    /// [`Self::finish_startup_warmup`], and never otherwise.
    pub fn observe_startup_steps(&mut self, observer: Box<dyn Fn(warmup::StartupStep) + Send>) {
        self.startup_step_observer = StartupStepObserver(Some(observer));
    }

    /// Isolates startup generations before any live request is submitted.
    pub fn begin_startup_warmup(&mut self) -> Result<(), String> {
        if self.startup_checkpoint.is_some()
            || !self.requests.is_empty()
            || self.next_event_sequence != 0
        {
            return Err("startup warmup requires an idle engine".to_owned());
        }
        let scheduler = self.scheduler.clone();
        let offload = self
            .offload
            .as_ref()
            .map(|state| state.isolated(&scheduler))
            .transpose()
            .map_err(|error| error.to_string())?;
        let metrics = EngineMetrics::default();
        metrics.compatibility(|metrics| metrics.configure(self.maximum_sequence_tokens));
        self.startup_checkpoint = Some(StartupCheckpoint {
            host_clock: self.host_clock.clone(),
            offload: std::mem::replace(&mut self.offload, offload),
            scheduler: std::mem::replace(&mut self.scheduler, scheduler),
            speculation: self.speculation.clone(),
            metrics: std::mem::replace(&mut self.metrics, metrics),
            prefix_block_tokens: self.prefix_block_tokens.take(),
            refusal_guard: self.refusal_guard.clone(),
            next_event_sequence: self.next_event_sequence,
        });
        Ok(())
    }

    /// Restores live state only after warmup requests and device work retire.
    pub fn finish_startup_warmup(&mut self) -> Result<(), String> {
        self.ensure_retirement_ready()
            .map_err(|error| error.to_string())?;
        self.drain_kv_transfers()
            .map_err(|error| error.to_string())?;
        while self.retire_terminals().map_err(|error| error.to_string())? > 0 {}
        if !self.requests.is_empty() || self.model_state_failed || self.prefix_state_unconfirmed {
            return Err("startup warmup cleanup did not leave a healthy idle engine".to_owned());
        }
        let saved = self
            .startup_checkpoint
            .take()
            .ok_or("startup warmup was not started")?;
        self.host_clock = saved.host_clock;
        self.scheduler = saved.scheduler;
        self.offload = saved.offload;
        self.speculation = saved.speculation;
        self.metrics = saved.metrics;
        self.prefix_block_tokens = saved.prefix_block_tokens;
        self.refusal_guard = saved.refusal_guard;
        self.next_event_sequence = saved.next_event_sequence;
        self.events.clear();
        self.publish_kv_occupancy();
        Ok(())
    }

    /// KV page size for conservative startup reservations.
    pub fn startup_page_tokens(&self) -> usize {
        self.scheduler.kv_geometry().block_tokens()
    }

    pub fn new(config: EngineConfig, executor: E) -> Result<Self, EngineError> {
        Self::new_with_observability(
            config,
            executor,
            Arc::new(SystemMonotonicClock::new()),
            EngineMetrics::default(),
        )
    }

    pub fn new_with_observability(
        config: EngineConfig,
        executor: E,
        clock: Arc<dyn MonotonicClock>,
        metrics: EngineMetrics,
    ) -> Result<Self, EngineError> {
        Self::new_with_runtime_metrics(config, executor, clock, metrics, KvMetrics::default())
    }

    pub fn new_with_runtime_metrics(
        config: EngineConfig,
        executor: E,
        clock: Arc<dyn MonotonicClock>,
        metrics: EngineMetrics,
        kv_metrics: KvMetrics,
    ) -> Result<Self, EngineError> {
        let mut scheduler =
            Scheduler::new_with_kv_metrics(config.scheduler, config.kv_geometry, kv_metrics)
                .map_err(EngineError::Scheduler)?;
        let mut pending_retirements = Vec::new();
        pending_retirements
            .try_reserve_exact(config.scheduler.maximum_sequences())
            .map_err(|_| EngineError::MetadataAllocationFailed)?;
        metrics.compatibility(|metrics| metrics.configure(config.maximum_sequence_tokens));
        let step_trace = metrics.compatibility(|metrics| metrics.step_trace_enabled);
        let speculation = config.speculation.map(|speculation| {
            SpeculationPolicy::new(speculation)
                .with_draft_rounding(|count| executor.round_draft_tokens_v1(count).min(count))
                .with_step_rows(|rows| executor.step_rows_v1(rows).max(rows))
                .with_diagnostics(step_trace)
        });
        let offload = executor
            .kv_offload_info_v1()
            .map(|info| {
                scheduler.configure_offload(runtime_scheduler::OffloadConfig {
                    minimum_tokens: info.minimum_tokens,
                    snapshot_pages: info.snapshot_pages,
                    enabled: true,
                    force_host: config.force_host_swap,
                });
                scheduler
                    .new_host_tier(info.host_page_capacity)
                    .map(|tier| offload::OffloadState::new(tier, info, config.force_host_swap))
            })
            .transpose()
            .map_err(EngineError::Scheduler)?;
        let engine = Self {
            startup_checkpoint: None,
            offload,
            executor,
            refusal_guard: refusal::RefusalGuard::new(config.refusal_limits),
            pending_retirements,
            retirement_failed: false,
            prefix_state_unconfirmed: false,
            model_state_failed: false,
            scheduler,
            requests: BTreeMap::new(),
            events: VecDeque::new(),
            next_event_sequence: 0,
            terminal_retirement_per_step: config.terminal_retirement_per_step,
            maximum_sequence_tokens: config.maximum_sequence_tokens,
            model_vocabulary: config.model_vocabulary,
            clock,
            metrics,
            speculation,
            media: config.media,
            position_axes: config.position_axes,
            arena: config.media.map(MediaArenaV1::new),
            prefix_block_tokens: config
                .prefix_cache()
                .map(|_| config.kv_geometry.block_tokens()),
            step_trace,
            pending_step_trace: None,
            next_output_cycle: 1,
            host_clock: compatibility::HostClock::default(),
            batch_observer: BatchObserver::default(),
            startup_step_observer: StartupStepObserver::default(),
        };
        // An engine that has served nothing still holds a pool, so its
        // occupancy is published before the first request rather than read as
        // an unset gauge.
        engine.publish_kv_occupancy();
        Ok(engine)
    }

    pub fn submit(&mut self, request: EngineRequest) -> Result<(), EngineError> {
        self.ensure_retirement_ready()?;
        if request.model_vocabulary != self.model_vocabulary {
            return Err(EngineError::InvalidRequest(
                EngineRequestError::VocabularyMismatch {
                    request: request.model_vocabulary.get(),
                    engine: self.model_vocabulary.get(),
                },
            ));
        }
        let requested = request
            .prompt_tokens
            .len()
            .checked_add(request.maximum_new_tokens)
            .ok_or(EngineError::InvalidRequest(
                EngineRequestError::LengthOverflow,
            ))?;
        if requested > self.maximum_sequence_tokens {
            return Err(EngineError::InvalidRequest(
                EngineRequestError::SequenceTooLong {
                    maximum: self.maximum_sequence_tokens,
                    requested,
                },
            ));
        }
        let media_pages = if request.media.is_empty() {
            0
        } else {
            let media =
                self.media
                    .ok_or(EngineError::InvalidRequest(EngineRequestError::Media(
                        MediaRequestError::Unsupported,
                    )))?;
            media
                .step_pages(
                    &request.media,
                    &request.spans,
                    self.scheduler.prefill_granule(),
                )
                .map_err(|error| EngineError::InvalidRequest(EngineRequestError::Media(error)))?
        };
        if request.positions.is_some() && !self.position_axes {
            return Err(EngineError::InvalidRequest(
                EngineRequestError::PositionAxes(PositionAxesError::Unsupported),
            ));
        }
        let spec = RequestSpec::new(
            request.id,
            request.prompt_tokens.len(),
            request.maximum_new_tokens,
        )
        .map_err(|error| EngineError::InvalidRequest(error.into()))?
        .with_media_pages(media_pages);
        // A recompute would need the media encoded again, or would report
        // prompt rows twice, so these requests are never preempted.
        let spec = if !request.media.is_empty()
            || !request.spans.is_empty()
            || request.positions.is_some()
            || request.prompt_logprobs.is_some()
        {
            spec.with_reserved_admission()
        } else {
            spec
        };
        let spec = if request.prompt_logprobs.is_some() {
            spec.with_prompt_scoring()
        } else {
            spec
        };
        let prefix = match self.prefix_block_tokens {
            Some(block_tokens) => Some(prefix_request(block_tokens, &request)?),
            None => None,
        };
        self.scheduler
            .submit_with_prefix(spec, prefix)
            .map_err(EngineError::Submit)?;

        let mut history = Vec::new();
        if history
            .try_reserve_exact(request.prompt_tokens.len() + request.maximum_new_tokens)
            .is_err()
        {
            self.scheduler
                .cancel(request.id)
                .map_err(EngineError::Scheduler)?;
            self.pending_retirements.push(request.id);
            self.notify_retirements()?;
            self.scheduler
                .retire_terminal(request.id)
                .map_err(EngineError::Scheduler)?;
            return Err(EngineError::MetadataAllocationFailed);
        }
        history.extend_from_slice(&request.prompt_tokens);
        let prompt_tokens = request.prompt_tokens.len();
        let submitted_micros = self.clock.now_micros();
        self.requests.insert(
            request.id,
            RequestRecord {
                prompt_tokens: request.prompt_tokens,
                prompt_offset: 0,
                prompt_logprobs: request.prompt_logprobs,
                pending_decode_token: None,
                sampling: request.sampling,
                sampling_state: SamplingState::new(request.seed),
                history: Arc::new(history),
                emitted_tokens: 0,
                maximum_new_tokens: request.maximum_new_tokens,
                stop_ids: request.stop_ids,
                termination_stop_ids: request.termination_stop_ids,
                submitted_micros,
                last_token_micros: None,
                acceptance: self.speculation.as_ref().map(SpeculationPolicy::prior),
                verified_last: false,
                media_wanted: vec![true; request.media.len()],
                media_placed: vec![false; request.media.len()],
                media_ends: media_ends(request.media.len(), &request.spans),
                media: request.media,
                spans: request.spans,
                positions: request.positions,
                constraint: request.constraint,
                cached_tokens: 0,
                cache_salt: request.cache_salt,
                pinned_draft_tokens: request.pinned_draft_tokens,
                scheduled: false,
                preemptions: 0,
                resuming: false,
            },
        );
        if let (Some(arena), Some(record)) = (self.arena.as_mut(), self.requests.get(&request.id)) {
            for item in &record.media {
                arena.want(item.key);
            }
        }
        self.metrics.request_submitted(prompt_tokens);
        self.metrics.compatibility(|metrics| {
            metrics.submit(
                request.id,
                submitted_micros,
                prompt_tokens,
                request.maximum_new_tokens,
            )
        });
        Ok(())
    }

    /// Runs one scheduled batch, then publishes what the KV pool now holds.
    ///
    /// Occupancy is a gauge over a pool this thread owns, so it is republished
    /// at every step boundary, including the ones that failed: a reader on the
    /// metrics side sees the pool as the last transition left it rather than
    /// as the last successful one did.
    pub fn step(&mut self) -> Result<StepStatus, EngineError> {
        self.ensure_retirement_ready()?;
        if self.model_state_failed {
            return Err(EngineError::ModelStateFailed);
        }
        let _profile = self.metrics.start_step_profile();
        let _totals = self.step_trace.then(profile::TraceTotals::begin);
        if self.step_trace {
            self.host_clock.step_entered();
        }
        let result = self.step_and_retire();
        if self.step_trace {
            if let Some(mut record) = self.pending_step_trace.take() {
                record.host.finish_trace();
                crate::diagnostic!("{}", record.record());
            }
            self.host_clock.step_returned();
        }
        result
    }

    fn step_and_retire(&mut self) -> Result<StepStatus, EngineError> {
        self.poll_transfers(false)?;
        let status = self.step_scheduled_batch();
        // The batch function has released every model transaction borrow.
        // Anchors capture executor state before a finished request retires.
        let captured = self.capture_prefixes();
        let terminal_transition = !self.pending_retirements.is_empty();
        let retirement = self.notify_retirements();
        self.publish_kv_occupancy();
        retirement?;
        captured?;
        if terminal_transition {
            self.retire_terminals()?;
        }
        status
    }

    fn step_scheduled_batch(&mut self) -> Result<StepStatus, EngineError> {
        if let Some(policy) = self.speculation.as_mut() {
            policy.clear_diagnostics();
        }
        let scheduling_timer = profile::PhaseTimer::new(profile::StepPhase::Scheduling);
        let retired_before = self.retire_terminals()?;
        self.preempt()?;
        self.restore_prefixes()?;
        let mut host_drafts = Vec::new();
        let mut draft_error = None;
        // Host drafting happens while the scheduler forms the batch.
        let planning_started_micros = self.clock.now_micros();
        let next = {
            let executor = &mut self.executor;
            let requests = &self.requests;
            let vocabulary = self.model_vocabulary;
            let media = self.media;
            let arena = self.arena.as_ref();
            let speculation = self.speculation.as_mut();
            let limits = speculation
                .as_ref()
                .map_or(DraftLimits::NONE, |policy| policy.config().limits());
            self.scheduler.next_batch_planned(
                limits,
                |id, offset, tokens| match (media, arena, requests.get(&id)) {
                    (Some(media), Some(arena), Some(record)) => ChunkMedia {
                        encode_cost: media_encode_cost(media, arena, record, offset, tokens),
                        pages: media.chunk_pages(
                            &record.media,
                            &record.spans,
                            offset,
                            offset + tokens,
                        ),
                    },
                    _ => ChunkMedia::default(),
                },
                |candidates, budget| match speculation {
                    Some(policy) => plan_drafts(
                        policy,
                        executor,
                        requests,
                        vocabulary,
                        candidates,
                        budget,
                        &mut host_drafts,
                        &mut draft_error,
                    ),
                    None => vec![0; candidates.len()],
                },
            )
        };
        if let Some(error) = draft_error {
            // The plan fell back to plain decodes, but a drafter that failed
            // is an executor failure; the batch is withdrawn unexecuted.
            if let Ok(Some(batch)) = &next {
                self.scheduler
                    .rollback_batch(batch.id())
                    .map_err(EngineError::Scheduler)?;
            }
            self.metrics.error();
            return Err(EngineError::Executor(error));
        }
        let next = match next {
            Err(ScheduleError::Kv(runtime_kv::KvError::PoolExhausted { .. }))
            | Err(ScheduleError::Overloaded(_))
                if self.has_transfers() =>
            {
                self.poll_transfers(true)?;
                return Ok(StepStatus::Progressed);
            }
            other => other.map_err(EngineError::Scheduler)?,
        };
        let Some(batch) = next else {
            if self.has_transfers() {
                self.poll_transfers(true)?;
                return Ok(StepStatus::Progressed);
            }
            return Ok(if retired_before > 0 {
                StepStatus::Progressed
            } else {
                StepStatus::Idle
            });
        };
        if let Some(observer) = &self.batch_observer.0 {
            observer(
                batch
                    .items()
                    .iter()
                    .any(|item| matches!(item.phase(), BatchPhase::Prefill { .. })),
            );
        }
        let batch_started_micros = self.clock.now_micros();
        self.metrics.batch(batch_composition(&batch));
        for item in batch.items() {
            if !matches!(item.phase(), BatchPhase::Prefill { .. }) {
                self.metrics
                    .speculation_k(item.phase().appended_tokens().saturating_sub(1));
            }
        }
        for item in batch.items() {
            let Some(request) = self.requests.get_mut(&item.request_id()) else {
                continue;
            };
            if !request.scheduled {
                request.scheduled = true;
                self.metrics.compatibility(|metrics| {
                    metrics.start(item.request_id(), batch_started_micros)
                });
                self.metrics
                    .queue(batch_started_micros.saturating_sub(request.submitted_micros));
            }
            if request.resuming {
                request.resuming = false;
                self.metrics
                    .compatibility(|metrics| metrics.resumed(item.request_id()));
            }
        }
        let draft_source = self
            .speculation
            .as_ref()
            .map(|policy| policy.config().source());
        let executor = &mut self.executor;
        let scheduler = &mut self.scheduler;
        let requests = &mut self.requests;
        let events = &mut self.events;
        let next_event_sequence = &mut self.next_event_sequence;
        let mut pending_batch = PendingBatchGuard::new(scheduler, self.arena.as_mut(), batch.id());
        let placement = match (self.media, pending_batch.arena_mut()) {
            (Some(media), Some(arena)) => place_media(arena, media, requests, &batch),
            _ => Ok(MediaPlacement::default()),
        };
        let placement = match placement {
            Ok(placement) => placement,
            Err(error) => {
                self.metrics.error();
                fail_batch_parts(
                    &mut pending_batch,
                    &mut self.pending_retirements,
                    &self.metrics,
                    events,
                    next_event_sequence,
                    &batch,
                    &error.to_string(),
                )?;
                return Err(error);
            }
        };
        let mut items = match model_items_parts(
            requests,
            &batch,
            draft_source,
            &host_drafts,
            placement.items,
        ) {
            Ok(items) => items,
            Err(error) => {
                self.metrics.error();
                fail_batch_parts(
                    &mut pending_batch,
                    &mut self.pending_retirements,
                    &self.metrics,
                    events,
                    next_event_sequence,
                    &batch,
                    &error.to_string(),
                )?;
                return Err(error);
            }
        };
        for item in &mut items {
            if matches!(item.phase, BatchPhase::Prefill { .. }) {
                item.prefill_boundary = pending_batch
                    .scheduler_mut()
                    .prefill_boundary_contract(item.request_id)
                    .map_err(EngineError::Scheduler)?
                    .map(|(chunk_tokens, prompt_tokens)| PrefillBoundary {
                        chunk_tokens,
                        prompt_tokens,
                    });
            }
        }
        if executor.device_results_v1() {
            let epoch = self.next_output_cycle;
            self.next_output_cycle = epoch
                .checked_add(1)
                .ok_or(EngineError::InvariantViolation)?;
            for item in &mut items {
                let record = requests
                    .get(&item.request_id)
                    .expect("scheduled request exists");
                item.output_cycle = Some(OutputCycle {
                    epoch,
                    remaining_tokens: record.maximum_new_tokens - record.emitted_tokens,
                    termination_stop_ids: record.termination_stop_ids.clone(),
                });
            }
        }
        drop(scheduling_timer);
        let execution_started_micros = self.clock.now_micros();
        if self.step_trace {
            self.host_clock.execute_started();
        }
        let rollback_preserves_state = executor.rollback_preserves_state_v1();
        let transaction_result = executor.execute(&items, batch.kv_plan());
        if self.step_trace {
            self.host_clock.execute_returned();
        }
        let transaction = match transaction_result {
            Ok(transaction) => transaction,
            Err(error) => {
                let message = error.message().to_owned();
                self.metrics.error();
                fail_batch_parts(
                    &mut pending_batch,
                    &mut self.pending_retirements,
                    &self.metrics,
                    events,
                    next_event_sequence,
                    &batch,
                    &message,
                )?;
                if error.is_batch_refused() {
                    self.metrics.batch_refused(batch.items().len());
                    if let Some(reason) = self.refusal_guard.refused(self.clock.now_micros()) {
                        self.model_state_failed = true;
                        return Err(EngineError::Executor(executor_error_fixed(reason)));
                    }
                    return Ok(StepStatus::Progressed);
                }
                return Err(EngineError::Executor(error));
            }
        };
        let observations = transaction.measured_observations();
        let row_capacity = transaction.row_capacity();
        if !outputs_match(
            &items,
            transaction.logits(),
            transaction.prompt_rows(),
            transaction.sampled(),
            transaction.drafts(),
            transaction.accepted_prefixes(),
            self.model_vocabulary,
        ) {
            drop(transaction);
            let message = "executor output rows do not match the pending batch";
            self.metrics.error();
            fail_batch_parts(
                &mut pending_batch,
                &mut self.pending_retirements,
                &self.metrics,
                events,
                next_event_sequence,
                &batch,
                message,
            )?;
            return Err(EngineError::OutputMismatch);
        }

        let sampling_timer = profile::PhaseTimer::new(profile::StepPhase::Sampling);
        // Requests whose sampling failed in a step that commits anyway: an
        // in-place executor cannot roll back, so every other item commits
        // and only these requests fail after the commit.
        let mut failed: Vec<(RequestId, RequestSampleError)> = Vec::new();
        let staged = loop {
            let skipped: Vec<RequestId> = failed.iter().map(|(id, _)| *id).collect();
            match stage_samples(
                requests,
                &items,
                transaction.logits(),
                transaction.sampled(),
                transaction.drafts(),
                transaction.accepted_prefixes(),
                self.model_vocabulary,
                &skipped,
            ) {
                Err(SampleStageError::Request(id, error))
                    if !rollback_preserves_state && !skipped.contains(&id) =>
                {
                    failed.push((id, error));
                }
                result => break result,
            }
        };
        let staged = match staged {
            Ok(staged) => staged,
            Err(SampleStageError::Request(id, error)) => {
                if let Err(rollback) = transaction.rollback() {
                    self.model_state_failed = true;
                    self.metrics.poison();
                    let message = rollback.message().to_owned();
                    self.metrics.error();
                    fail_batch_parts(
                        &mut pending_batch,
                        &mut self.pending_retirements,
                        &self.metrics,
                        events,
                        next_event_sequence,
                        &batch,
                        &message,
                    )?;
                    return Err(EngineError::Executor(rollback));
                }
                self.metrics.error();
                fail_request_parts(
                    &mut pending_batch,
                    &mut self.pending_retirements,
                    &self.metrics,
                    events,
                    next_event_sequence,
                    id,
                    &error.to_string(),
                )?;
                return Err(error.into());
            }
            Err(SampleStageError::Global(error)) => {
                drop(transaction);
                let message = error.to_string();
                self.metrics.error();
                fail_batch_parts(
                    &mut pending_batch,
                    &mut self.pending_retirements,
                    &self.metrics,
                    events,
                    next_event_sequence,
                    &batch,
                    &message,
                )?;
                return Err(error);
            }
        };
        drop(sampling_timer);
        // The tokens are staged, so the rows no longer read the histories
        // they share with the requests, which the apply appends to.
        release_histories(&mut items);
        let mut outcomes = Vec::new();
        let mut kept = Vec::new();
        if outcomes.try_reserve_exact(batch.items().len()).is_err()
            || kept.try_reserve_exact(batch.items().len()).is_err()
        {
            drop(transaction);
            let error = EngineError::MetadataAllocationFailed;
            let message = error.to_string();
            self.metrics.error();
            fail_batch_parts(
                &mut pending_batch,
                &mut self.pending_retirements,
                &self.metrics,
                events,
                next_event_sequence,
                &batch,
                &message,
            )?;
            return Err(error);
        }
        for (item, run) in batch.items().iter().zip(&staged) {
            // A failed request commits one placeholder row, which the
            // scheduler and a verify commit accept, and publishes nothing.
            let emitted = if failed.iter().any(|(id, _)| *id == item.request_id()) {
                1
            } else {
                run.as_ref().map_or(0, |run| run.tokens.len())
            };
            outcomes.push(BatchOutcome::success(
                item.request_id(),
                item.phase(),
                emitted,
                run.as_ref().is_some_and(|run| run.terminal),
            ));
            if matches!(item.phase(), BatchPhase::Verify { .. }) {
                kept.push(KeptRows {
                    request_id: item.request_id(),
                    kept_rows: emitted,
                });
            }
        }
        if let Some(block_tokens) = self.prefix_block_tokens {
            for (item, run) in items.iter().zip(&staged) {
                let Some(run) = run else {
                    continue;
                };
                let record = &requests[&item.request_id];
                if matches!(item.phase, BatchPhase::Prefill { .. })
                    || failed.iter().any(|(id, _)| *id == item.request_id)
                    || !(run.terminal
                        || record.emitted_tokens + run.tokens.len() == record.maximum_new_tokens)
                    || !record.media.is_empty()
                    || !record.spans.is_empty()
                    || record.positions.is_some()
                    || pending_batch
                        .scheduler_mut()
                        .prefill_boundary_contract(item.request_id)
                        .map_err(EngineError::Scheduler)?
                        .is_none()
                {
                    continue;
                }
                let mut history = record.history.to_vec();
                history.extend(run.tokens.iter().map(|token| token.token_id));
                history.pop();
                let chain = PrefixChainV1::salted(
                    block_tokens,
                    Arc::from(history),
                    Vec::new(),
                    record.cache_salt.as_deref(),
                )
                .map_err(|e| EngineError::Scheduler(ScheduleError::Kv(e)))?;
                pending_batch
                    .scheduler_mut()
                    .prepare_decode_capture(
                        item.request_id,
                        PrefixRequestV1::new(Arc::new(chain), true),
                    )
                    .map_err(EngineError::Scheduler)?;
            }
        }
        let verified: Vec<_> = staged
            .iter()
            .map(|run| {
                run.as_ref()
                    .filter(|run| run.drafted > 0)
                    .map(|run| (run.drafted, run.accepted, run.rejected, run.tokens.len()))
            })
            .collect();
        let prepared_engine = match preflight_apply(
            requests,
            events,
            *next_event_sequence,
            &batch,
            &items,
            transaction.prompt_rows(),
            staged,
        ) {
            Ok(prepared) => prepared,
            Err(error) => {
                drop(transaction);
                let message = error.to_string();
                self.metrics.error();
                fail_batch_parts(
                    &mut pending_batch,
                    &mut self.pending_retirements,
                    &self.metrics,
                    events,
                    next_event_sequence,
                    &batch,
                    &message,
                )?;
                return Err(error);
            }
        };
        let scheduler_timer = profile::PhaseTimer::traced(profile::StepPhase::SchedulerCommit);
        let prepared_scheduler = match pending_batch
            .scheduler_mut()
            .preflight_commit_batch(batch.id(), &outcomes)
        {
            Ok(prepared) => prepared,
            Err(error) => {
                drop(transaction);
                let message = error.to_string();
                self.metrics.error();
                fail_batch_parts(
                    &mut pending_batch,
                    &mut self.pending_retirements,
                    &self.metrics,
                    events,
                    next_event_sequence,
                    &batch,
                    &message,
                )?;
                return Err(EngineError::Scheduler(error));
            }
        };
        drop(scheduler_timer);
        let (report, model_commit) = commit_scheduler_then_model_observed(
            transaction,
            (!kept.is_empty()).then_some(kept.as_slice()),
            || pending_batch.apply(prepared_scheduler),
            &self.metrics,
        )
        .map_err(EngineError::Scheduler)?;
        self.pending_retirements
            .extend_from_slice(report.completed());
        if let Err(error) = model_commit {
            // The scheduler has committed and the model may not have, so
            // nothing this batch sampled is published and execution stops.
            self.model_state_failed = true;
            self.metrics.poison();
            self.metrics.error();
            return Err(EngineError::Executor(error));
        }
        let publication_timer = profile::PhaseTimer::traced(profile::StepPhase::Publication);
        // Tokens each item published: none for a failed request's placeholder.
        let is_failed = |id: RequestId| failed.iter().any(|(failed, _)| *failed == id);
        let published = |outcome: &BatchOutcome| {
            if is_failed(outcome.request_id()) {
                0
            } else {
                outcome.emitted_tokens()
            }
        };
        let completed_at = self.clock.now_micros();
        let completed_unix_ns = self.step_trace.then(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |since| {
                    u64::try_from(since.as_nanos()).unwrap_or(u64::MAX)
                })
        });
        let execution_micros = completed_at.saturating_sub(execution_started_micros);
        let rows = step_rows::StepRows::from_items(&items, row_capacity);
        let step_rows::StepRows {
            prefill_tokens,
            prefill_items,
            decode_rows,
            verify_items,
            ..
        } = rows;
        let step_trace = self.metrics.compatibility(|metrics| {
            metrics.step_rows.observe(rows);
            metrics.step(
                prefill_tokens,
                decode_rows + verify_items > 0,
                execution_micros,
                batch.prefill_cap(),
            );
            let mut iteration_tokens = 0_u64;
            for ((item, outcome), verified) in items.iter().zip(&outcomes).zip(&verified) {
                if let Some((drafted, accepted, _, _)) = *verified {
                    metrics.verified(item.request_id(), drafted, accepted);
                }
                let emitted = published(outcome);
                metrics.tokens(item.request_id(), emitted, completed_at);
                iteration_tokens += emitted as u64;
                if matches!(item.phase(), BatchPhase::Prefill { .. }) {
                    iteration_tokens += item.phase().appended_tokens() as u64;
                }
            }
            metrics.iteration.observe(iteration_tokens);
            metrics.step_trace_enabled
        });
        for event in &prepared_engine.events {
            if let EngineEventKind::Finished { reason } = event.kind() {
                self.metrics.compatibility(|metrics| {
                    metrics.finish(event.request_id(), *reason, completed_at)
                });
            }
        }
        apply_committed_parts(requests, events, next_event_sequence, prepared_engine);
        drop(pending_batch);
        drop(publication_timer);
        let assumed = self.scheduler.step_time_estimates();
        let trained = verify_items == 0 && placement.pinned.is_empty();
        if trained {
            self.scheduler
                .observe_step_time(prefill_tokens, decode_rows, execution_micros);
            if let Some(estimates) = self.scheduler.step_time_estimates() {
                self.metrics
                    .compatibility(|metrics| metrics.step_estimates = Some(estimates));
            }
        }
        // The TBT model also learns from verify steps; a media encode's time
        // is neither prefill nor decode work.
        if placement.pinned.is_empty() {
            self.scheduler
                .observe_batch_time(batch.shape(), execution_micros);
            if let Some(estimates) = self.scheduler.mixed_estimates() {
                self.metrics.compatibility(|metrics| {
                    metrics.mixed_estimates = Some(estimates);
                    if let Some(predicted) = batch.predicted_micros() {
                        metrics.mixed_step(predicted, execution_micros);
                    }
                });
            }
        }
        if step_trace {
            let record = compatibility::StepTrace {
                batch: batch.id().get(),
                prefill_tokens,
                prefill_items,
                completed_prompts: items
                    .iter()
                    .filter(|item| {
                        matches!(item.phase(), BatchPhase::Prefill { .. }) && item.expects_logits()
                    })
                    .filter_map(|item| {
                        requests
                            .get(&item.request_id())
                            .map(|record| (item.request_id().get(), record.prompt_tokens.len()))
                    })
                    .collect(),
                decode_rows,
                verify_items,
                verify_rows: rows.verify_rows,
                verified: verified.iter().flatten().fold(
                    compatibility::VerifiedTokens::default(),
                    |total, &(drafted, accepted, _, emitted)| compatibility::VerifiedTokens {
                        drafted: total.drafted + drafted,
                        accepted: total.accepted + accepted,
                        emitted: total.emitted + emitted,
                    },
                ),
                row_capacity: rows.row_capacity,
                prefill_cap: batch.prefill_cap(),
                execution_micros,
                completed_unix_ns,
                predicted_micros: batch.predicted_micros().or_else(|| {
                    assumed.and_then(|estimates| {
                        estimates.predicted_micros(prefill_tokens, decode_rows)
                    })
                }),
                decode_estimate_micros: assumed
                    .and_then(|estimates| estimates.decode_micros(decode_rows)),
                trained,
                host: self.host_clock.executed(),
                speculation: self
                    .speculation
                    .as_ref()
                    .and_then(SpeculationPolicy::decision),
            };
            self.pending_step_trace = Some(record);
        }
        if self.startup_checkpoint.is_some()
            && let Some(observer) = &self.startup_step_observer.0
        {
            observer(warmup::StartupStep {
                prefill_tokens,
                decode_rows,
                verify_items,
                verify_rows: rows.verify_rows,
                accepted_tokens: verified
                    .iter()
                    .flatten()
                    .map(|&(_, accepted, _, _)| accepted)
                    .sum(),
            });
        }

        if let Some(arena) = self.arena.as_mut() {
            // Pins last one step: the next step pins what its chunks cover.
            for (id, media, key) in &placement.pinned {
                arena.unpin(key);
                if let Some(record) = requests.get_mut(id) {
                    record.media_placed[*media] = true;
                }
            }
            for item in batch.items() {
                if let Some(record) = requests.get_mut(&item.request_id()) {
                    let passed = record.prompt_offset;
                    release_passed_media(arena, record, passed);
                }
            }
            self.metrics.media(
                placement.encodes,
                placement.encoded_rows,
                placement.hits,
                placement.evictions,
                placement.reencodes,
            );
        }
        let emitted_micros = self.clock.now_micros();
        for ((item, outcome), verified) in items.iter().zip(&outcomes).zip(&verified) {
            if item.expects_logits()
                && !is_failed(item.request_id())
                && let Some(request) = requests.get_mut(&item.request_id())
            {
                if let Some(previous) = request.last_token_micros {
                    self.metrics.itl(emitted_micros.saturating_sub(previous));
                } else {
                    self.metrics
                        .ttft(emitted_micros.saturating_sub(request.submitted_micros));
                }
                request.last_token_micros = Some(emitted_micros);
                self.metrics.generated(outcome.emitted_tokens());
                if let Some((drafted, accepted, rejected, emitted)) = *verified {
                    self.metrics.verified(drafted, accepted, emitted);
                    if let (Some(policy), Some(acceptance)) =
                        (self.speculation.as_mut(), request.acceptance.as_mut())
                    {
                        // A run cut by a stop or the length limit says nothing
                        // about the drafts after the cut.
                        let observed = if rejected { drafted } else { accepted };
                        policy.observe(acceptance, observed, accepted);
                    }
                }
            }
        }
        let prefilled = batch
            .items()
            .iter()
            .any(|item| matches!(item.phase(), BatchPhase::Prefill { .. }));
        if let Some(policy) = self.speculation.as_mut().filter(|_| prefilled) {
            // A step that also prefilled trains no step time; the drafts it
            // accepted must not count toward the next timed step.
            policy.untimed_step();
        } else if let Some(policy) = self.speculation.as_mut() {
            let drafts = batch
                .items()
                .iter()
                .map(|item| item.phase().appended_tokens() - 1)
                .collect::<Vec<_>>();
            let fresh = batch
                .items()
                .iter()
                .filter(|item| {
                    matches!(item.phase(), BatchPhase::Verify { .. })
                        && requests
                            .get(&item.request_id())
                            .is_some_and(|record| !record.verified_last)
                })
                .count();
            policy.observe_step(
                &drafts,
                fresh,
                emitted_micros.saturating_sub(planning_started_micros),
            );
        }
        for item in batch.items() {
            if let Some(record) = requests.get_mut(&item.request_id()) {
                record.verified_last = matches!(item.phase(), BatchPhase::Verify { .. });
            }
        }
        if let Some(observations) = observations {
            self.metrics.execution(observations);
        }
        // A failed request's placeholder row may have completed it.
        let failed_completed = report
            .completed()
            .iter()
            .filter(|&&id| is_failed(id))
            .count();
        self.metrics
            .request_succeeded(report.completed().len() - failed_completed);
        self.refusal_guard.committed();
        let Some((_, first_error)) = failed.first().cloned() else {
            return Ok(StepStatus::Progressed);
        };
        // The step committed; only the failed requests end, with their own
        // errors, and their state is retired before any reuse.
        for (id, error) in &failed {
            let state = self
                .scheduler
                .request_state(*id)
                .map_err(EngineError::Scheduler)?;
            if !state.is_terminal() {
                self.scheduler.fail(*id).map_err(EngineError::Scheduler)?;
                self.pending_retirements.push(*id);
            }
            self.metrics.error();
            self.metrics.request_failed(1);
            self.metrics.compatibility(|metrics| metrics.fail(*id));
            push_event_parts(
                events,
                next_event_sequence,
                *id,
                EngineEventKind::Error {
                    message: error.to_string(),
                },
            )?;
        }
        Err(first_error.into())
    }

    /// Grows reservations for the next step and preempts what the pool
    /// cannot hold.
    ///
    /// Each preempted request's committed prefix is captured as an anchor
    /// while its model state still exists; then that state is retired, so
    /// the request recomputes from its tokens, or restores the anchor, when
    /// it is admitted again. Admission and the batch follow in this step.
    fn preempt(&mut self) -> Result<(), EngineError> {
        let lookahead = self
            .speculation
            .as_ref()
            .map_or(0, |policy| policy.config().limits().maximum_draft_tokens());
        self.update_offload_support();
        let requests = &self.requests;
        let block_tokens = self.prefix_block_tokens.or_else(|| {
            self.offload
                .as_ref()
                .map(|_| self.scheduler.kv_geometry().block_tokens())
        });
        let mut invalid_identity = false;
        let preempted = self
            .scheduler
            .preempt_for_next_step(lookahead, |id, tokens| {
                let record = requests.get(&id)?;
                // Media requests reserve their worst case and remain pinned.
                // A text-only resume chain must never erase media identity.
                if !record.media.is_empty()
                    || !record.spans.is_empty()
                    || record.positions.is_some()
                {
                    invalid_identity = true;
                    return None;
                }
                let chain = PrefixChainV1::salted(
                    block_tokens?,
                    Arc::from(record.history.get(..tokens)?),
                    Vec::new(),
                    record.cache_salt.as_deref(),
                )
                .ok()?;
                Some(PrefixRequestV1::new(Arc::new(chain), true))
            })
            .map_err(EngineError::Scheduler)?;
        if invalid_identity {
            return Err(EngineError::InvariantViolation);
        }
        if preempted.is_empty() {
            return Ok(());
        }
        for id in &preempted {
            let record = self
                .requests
                .get_mut(id)
                .ok_or(EngineError::InvariantViolation)?;
            record.prompt_offset = 0;
            record.preemptions += 1;
            record.resuming = true;
            record.verified_last = false;
            self.metrics.compatibility(|metrics| metrics.preempted(*id));
        }
        self.capture_prefixes()?;
        self.pending_retirements.extend_from_slice(&preempted);
        self.notify_retirements()
    }

    /// Admits queued requests and restores the executor state of every one
    /// that resumed from an anchor. A refused restore prefills from zero.
    ///
    /// A restore hands back the request's snapshot pins and the prefix pages
    /// it shares, which can admit more requests, so admission repeats until
    /// it admits nothing. The batch's own admission then finds nothing new,
    /// and every admitted request has passed the executor's admission.
    fn restore_prefixes(&mut self) -> Result<(), EngineError> {
        loop {
            if !self.admit_and_restore()? {
                return Ok(());
            }
        }
    }

    /// One admission pass and the restores it made due; returns whether a
    /// further pass could admit more.
    fn admit_and_restore(&mut self) -> Result<bool, EngineError> {
        let lookahead = self
            .speculation
            .as_ref()
            .map_or(0, |policy| policy.config().limits().maximum_draft_tokens());
        let admitted = self
            .scheduler
            .admit_with_lookahead(lookahead)
            .map_err(EngineError::Scheduler)?;
        let mut first_error = None;
        for &id in admitted.admitted() {
            if let Err(error) = self.executor.admit_request_v1(id) {
                self.metrics.error();
                self.scheduler.fail(id).map_err(EngineError::Scheduler)?;
                self.pending_retirements.push(id);
                self.metrics.request_failed(1);
                self.metrics.compatibility(|metrics| metrics.fail(id));
                self.push_event(
                    id,
                    EngineEventKind::Error {
                        message: error.to_string(),
                    },
                )?;
                first_error.get_or_insert(error);
            }
        }
        if let Some(error) = first_error {
            return Err(EngineError::Executor(error));
        }
        self.start_reloads()?;
        let retry_admission = self.scheduler.swap_needs_discard();
        if retry_admission {
            self.metrics.offload_fallback("staging_capacity");
        }
        self.release_unused_swaps()?;
        // Without restores nothing hands back pages before the next batch.
        if self.prefix_block_tokens.is_none() && self.offload.is_none() {
            return Ok(false);
        }
        for restore in self.scheduler.pending_prefix_restores() {
            let id = restore.request_id();
            let record = self
                .requests
                .get_mut(&id)
                .ok_or(EngineError::InvariantViolation)?;
            let prefix = record
                .history
                .get(..restore.tokens())
                .ok_or(EngineError::InvariantViolation)?;
            let started = self.clock.now_micros();
            let restored = self
                .executor
                .restore_prefix_state_v1(id, prefix, restore.pages());
            self.metrics
                .prefix_state_call(false, self.clock.now_micros().saturating_sub(started));
            match restored {
                // A recompute's hit is not reported: the client saw the
                // request's first admission.
                Ok(()) if record.preemptions > 0 => {
                    self.scheduler
                        .complete_prefix_restore(id)
                        .map_err(EngineError::Scheduler)?;
                    record.prompt_offset = if restore.resumes_decoding() {
                        record.prompt_tokens.len()
                    } else {
                        restore.tokens()
                    };
                }
                Ok(()) => {
                    self.scheduler
                        .complete_prefix_restore(id)
                        .map_err(EngineError::Scheduler)?;
                    record.prompt_offset = restore.tokens();
                    record.cached_tokens = restore.tokens();
                    self.metrics
                        .compatibility(|metrics| metrics.cached(id, restore.tokens()));
                    self.push_event(
                        id,
                        EngineEventKind::PromptCached {
                            tokens: restore.tokens(),
                        },
                    )?;
                }
                Err(error) if error.is_fatal() => {
                    self.prefix_state_unconfirmed = true;
                    self.metrics.poison();
                    return Err(EngineError::Executor(error));
                }
                Err(_) => self
                    .scheduler
                    .abandon_prefix_restore(id)
                    .map_err(EngineError::Scheduler)?,
            }
        }
        Ok(!admitted.admitted().is_empty() || retry_admission)
    }

    /// Captures every anchor the last commit made due. A refused capture only
    /// drops its anchor; a fatal one keeps it and stops the engine.
    fn capture_prefixes(&mut self) -> Result<(), EngineError> {
        let _phase = profile::PhaseTimer::traced(profile::StepPhase::PrefixCapture);
        let mut submission_error = None;
        for capture in self.scheduler.take_prefix_captures() {
            if submission_error.is_some() {
                self.scheduler
                    .end_prefix_capture(&capture)
                    .map_err(EngineError::Scheduler)?;
                continue;
            }
            if let Some(pages) = self
                .scheduler
                .begin_prefix_capture(&capture)
                .map_err(EngineError::Scheduler)?
            {
                let started = self.clock.now_micros();
                let captured = self.executor.capture_prefix_state_v1(
                    capture.request_id(),
                    capture.tokens(),
                    &pages,
                );
                self.metrics
                    .prefix_state_call(true, self.clock.now_micros().saturating_sub(started));
                match captured {
                    Ok(()) => {
                        if self.scheduler.capture_swaps(&capture)
                            && let Err(error) = self.start_offload(&capture, true)
                        {
                            if !self.model_state_failed {
                                return Err(error);
                            }
                            submission_error = Some(error);
                        }
                    }
                    Err(error) if error.is_fatal() => {
                        self.prefix_state_unconfirmed = true;
                        self.metrics.poison();
                        return Err(EngineError::Executor(error));
                    }
                    Err(_) => {
                        if self.scheduler.capture_swaps(&capture) {
                            self.metrics.offload_fallback("capture");
                        }
                        self.scheduler
                            .abandon_prefix_capture(&capture)
                            .map_err(EngineError::Scheduler)?;
                    }
                }
            } else if self.scheduler.capture_swaps(&capture) {
                if self.scheduler.has_capture_anchor(&capture) {
                    if let Err(error) = self.start_offload(&capture, false) {
                        if !self.model_state_failed {
                            return Err(error);
                        }
                        submission_error = Some(error);
                    }
                } else {
                    self.metrics.offload_fallback("capture");
                }
            }
            self.scheduler
                .end_prefix_capture(&capture)
                .map_err(EngineError::Scheduler)?;
        }
        submission_error.map_or(Ok(()), Err)
    }

    pub fn run_until_idle(&mut self, maximum_steps: usize) -> Result<(), EngineError> {
        for _ in 0..maximum_steps {
            if self.step()? == StepStatus::Idle {
                return Ok(());
            }
        }
        Err(EngineError::StepLimitReached)
    }

    /// Lets a frontend that recognizes text stops own the public completion
    /// reason. Call after submit and before stepping, then finalize on delivery
    /// or cancellation, even if the engine request has already retired.
    pub fn defer_completion_metrics(&self, id: RequestId) {
        self.metrics
            .compatibility(|metrics| metrics.defer_completion(id));
    }

    /// Records the frontend's final reason once, including for retired runs.
    pub fn finish_completion_metrics(&self, id: RequestId, reason: FinishReason) {
        self.metrics.compatibility(|metrics| {
            metrics.finish_completion(id, reason, self.clock.now_micros())
        });
    }

    /// Retires live work after a successful frontend text stop.
    pub fn stop(&mut self, id: RequestId) -> Result<(), EngineError> {
        self.terminate(id, FinishReason::Stop)
    }

    pub fn cancel(&mut self, id: RequestId) -> Result<(), EngineError> {
        self.terminate(id, FinishReason::Cancelled)
    }

    fn terminate(&mut self, id: RequestId, reason: FinishReason) -> Result<(), EngineError> {
        self.ensure_retirement_ready()?;
        self.scheduler.cancel(id).map_err(EngineError::Scheduler)?;
        self.release_unused_swaps()?;
        self.pending_retirements.push(id);
        if reason == FinishReason::Cancelled {
            self.metrics.cancellation();
        } else {
            self.metrics.request_succeeded(1);
        }
        self.metrics
            .compatibility(|metrics| metrics.finish(id, reason, self.clock.now_micros()));
        let retirement = self.notify_retirements();
        self.publish_kv_occupancy();
        retirement?;
        self.push_event(id, EngineEventKind::Finished { reason })?;
        let retired = self.retire_terminals();
        self.publish_kv_occupancy();
        retired?;
        Ok(())
    }

    fn ensure_retirement_ready(&self) -> Result<(), EngineError> {
        if self.retirement_failed {
            Err(EngineError::RetirementFailed)
        } else if self.prefix_state_unconfirmed {
            Err(EngineError::PrefixStateUnconfirmed)
        } else {
            Ok(())
        }
    }

    fn notify_retirements(&mut self) -> Result<(), EngineError> {
        let mut first_error = None;
        for id in &self.pending_retirements {
            if let (Some(arena), Some(record)) = (self.arena.as_mut(), self.requests.get_mut(id)) {
                release_passed_media(arena, record, usize::MAX);
            }
        }
        for id in self.pending_retirements.drain(..) {
            if let Err(error) = self.executor.retire_request_v1(id) {
                self.retirement_failed = true;
                self.metrics.poison();
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), |error| Err(EngineError::Executor(error)))
    }

    /// Republishes the pool's own occupancy through the shared KV handle.
    ///
    /// The engine runs on the thread that owns the pool, so this is the only
    /// place a metrics reader on another thread can learn what the pool holds.
    fn publish_kv_occupancy(&self) {
        if let Some(state) = &self.offload {
            self.metrics.offload_occupancy(
                state.tier.used_pages(),
                state.tier.capacity(),
                self.scheduler.swapped_requests(),
            );
        }
        self.scheduler
            .kv_metrics()
            .observe_occupancy(self.scheduler.kv_stats());
    }

    #[must_use]
    pub fn drain_events(&mut self) -> Vec<EngineEvent> {
        self.events.drain(..).collect()
    }

    #[must_use]
    pub const fn executor(&self) -> &E {
        &self.executor
    }

    #[must_use]
    pub const fn executor_mut(&mut self) -> &mut E {
        &mut self.executor
    }

    #[must_use]
    pub const fn maximum_sequence_tokens(&self) -> usize {
        self.maximum_sequence_tokens
    }

    #[must_use]
    pub const fn capacity_profile(&self) -> SchedulerCapacityProfile {
        self.scheduler.capacity_profile()
    }

    #[must_use]
    pub const fn model_vocabulary(&self) -> ModelVocabulary {
        self.model_vocabulary
    }

    #[must_use]
    pub const fn prefix_cache_enabled(&self) -> bool {
        self.prefix_block_tokens.is_some()
    }

    #[must_use]
    pub fn kv_stats(&self) -> KvStats {
        self.scheduler.kv_stats()
    }

    #[must_use]
    pub fn metrics(&self) -> EngineMetrics {
        self.metrics.clone()
    }

    #[must_use]
    pub fn kv_metrics(&self) -> KvMetrics {
        self.scheduler.kv_metrics()
    }

    #[must_use]
    pub fn terminal_count(&self) -> usize {
        self.scheduler.terminal_count()
    }

    /// Share cacheable prefixes within one newly submitted group, before stepping.
    pub fn share_prompt_prefixes(&mut self, ids: &[RequestId]) -> Result<(), EngineError> {
        self.scheduler
            .share_prompt_prefixes(ids)
            .map_err(EngineError::Scheduler)
    }

    pub fn request_state(
        &self,
        id: RequestId,
    ) -> Result<runtime_scheduler::RequestState, EngineError> {
        self.scheduler
            .request_state(id)
            .map_err(EngineError::Scheduler)
    }

    pub fn audit(&self) -> Result<(), EngineError> {
        self.scheduler.audit().map_err(EngineError::Scheduler)?;
        if let Some(offload) = &self.offload {
            offload.audit_transfers()?;
            self.scheduler
                .audit_host_tier(&offload.tier)
                .map_err(EngineError::Scheduler)?;
        }
        for (id, record) in &self.requests {
            let state = self
                .scheduler
                .request_state(*id)
                .map_err(EngineError::Scheduler)?;
            if record.prompt_offset > record.history.len()
                || record.history.len() != record.prompt_tokens.len() + record.emitted_tokens
                || (record.pending_decode_token.is_some() != (record.emitted_tokens > 0))
                || (state == RequestState::Decoding && record.pending_decode_token.is_none())
                || record.media_wanted.len() != record.media.len()
                || record.media_placed.len() != record.media.len()
            {
                return Err(EngineError::InvariantViolation);
            }
        }
        if let Some(arena) = &self.arena {
            // No pin outlives its step; wants follow the live prefills.
            let mut wanted = BTreeMap::new();
            for record in self.requests.values() {
                for (item, wants) in record.media.iter().zip(&record.media_wanted) {
                    if *wants {
                        *wanted.entry(item.key).or_insert(0u32) += 1;
                    }
                }
            }
            if !arena.audit(&BTreeMap::new()) || !arena.wanted_matches(&wanted) {
                return Err(EngineError::InvariantViolation);
            }
        }
        Ok(())
    }

    fn retire_terminals(&mut self) -> Result<usize, EngineError> {
        let retired = self
            .scheduler
            .reap_terminal(self.terminal_retirement_per_step)
            .map_err(EngineError::Scheduler)?;
        for id in &retired {
            self.requests.remove(id);
        }
        Ok(retired.len())
    }

    fn push_event(&mut self, id: RequestId, kind: EngineEventKind) -> Result<(), EngineError> {
        let sequence = self.next_event_sequence;
        self.next_event_sequence = self
            .next_event_sequence
            .checked_add(1)
            .ok_or(EngineError::EventSequenceExhausted)?;
        self.events.push_back(EngineEvent {
            sequence,
            request_id: id,
            kind,
        });
        Ok(())
    }
}

/// Hashes a request's prompt, media and position semantics for the prefix cache.
///
/// A request that asks for prompt logprobs needs every prompt row, so it never
/// resumes from an anchor, though its prompt can still become one.
fn prefix_request(
    block_tokens: usize,
    request: &EngineRequest,
) -> Result<PrefixRequestV1, EngineError> {
    let media = request
        .spans
        .iter()
        .map(|span| {
            let item = request
                .media
                .get(usize::from(span.media))
                .ok_or(EngineError::InvariantViolation)?;
            Ok(PrefixMediaSpanV1 {
                identity: request.positions.as_ref().map_or(item.key, |positions| {
                    positions.prefix_identity(span.prompt_row, item.key)
                }),
                start: span.prompt_row as usize,
                rows: span.rows as usize,
                media_row: span.media_row,
            })
        })
        .collect::<Result<Vec<_>, EngineError>>()?;
    let chain = PrefixChainV1::salted(
        block_tokens,
        Arc::from(request.prompt_tokens.as_slice()),
        media,
        request.cache_salt.as_deref(),
    )
    .map_err(|error| EngineError::Scheduler(ScheduleError::Kv(error)))?;
    Ok(PrefixRequestV1::new(
        Arc::new(chain),
        request.prompt_logprobs.is_none(),
    ))
}

fn executor_error_fixed(message: &'static str) -> ExecutorError {
    ExecutorError {
        message: message.to_owned(),
        fatal: false,
        batch_refused: false,
    }
}

/// Counts one dispatched batch by the phase the scheduler gave each item.
///
/// Membership is read off the batch itself, so a batch that carries both a
/// prompt chunk and a decode step is recorded as one mixed batch rather than
/// inferred from how many requests happened to be in flight.
fn batch_composition(batch: &Batch) -> BatchComposition {
    let mut prefill_items = 0;
    let mut decode_items = 0;
    for item in batch.items() {
        match item.phase() {
            BatchPhase::Prefill { .. } => prefill_items += 1,
            BatchPhase::Decode | BatchPhase::Verify { .. } => decode_items += 1,
        }
    }
    BatchComposition::new(prefill_items, decode_items)
}

/// Rows an item must yield: none without output, one per decode or prefill
/// tail, and one per verify row.
fn expected_output_rows(item: &ModelBatchItem) -> usize {
    if item.expects_logits {
        item.draft.map_or(1, |draft| draft.tokens + 1)
    } else {
        0
    }
}

fn outputs_match(
    items: &[ModelBatchItem],
    rows: &[LogitsRow],
    prompt_rows: &[LogitsRow],
    sampled: &[SampledRow],
    drafts: &[DraftRow],
    accepted_prefixes: &[AcceptedPrefix],
    model_vocabulary: ModelVocabulary,
) -> bool {
    let expected_prompt = items
        .iter()
        .flat_map(|item| std::iter::repeat_n(item.request_id(), item.prompt_targets().len()));
    if !expected_prompt.eq(prompt_rows.iter().map(LogitsRow::request_id))
        || prompt_rows.iter().any(|row| {
            row.logits().len() != model_vocabulary.get()
                && row.prompt_score().is_none()
                && !row.logits().is_empty()
        })
    {
        return false;
    }
    let bundle_drafted = |item: &ModelBatchItem| {
        item.draft
            .is_some_and(|draft| draft.source == DraftSource::Bundle)
    };
    let items_by_request = by_request(items, ModelBatchItem::request_id);
    let drafts_by_request = by_request(drafts, DraftRow::request_id);
    let accepted_by_request = by_request(accepted_prefixes, |prefix| prefix.request_id);
    let sampled_by_request = by_request(sampled, SampledRow::request_id);
    if accepted_by_request
        .values()
        .any(|prefixes| prefixes.len() != 1)
        || accepted_prefixes.iter().any(|prefix| {
            drafts_by_request.contains_key(&prefix.request_id)
                || !items_by_request
                    .get(&prefix.request_id)
                    .is_some_and(|items| {
                        items.len() == 1
                            && prefix.matches(
                                items[0],
                                sampled_by_request
                                    .get(&prefix.request_id)
                                    .map_or(&[], Vec::as_slice),
                            )
                    })
        })
    {
        return false;
    }
    let drafts_match = drafts_by_request.values().all(|drafts| drafts.len() == 1)
        && drafts.iter().all(|draft| {
            items_by_request
                .get(&draft.request_id)
                .is_some_and(|items| {
                    items.iter().any(|item| {
                        bundle_drafted(item)
                            && item
                                .draft
                                .is_some_and(|work| work.tokens == draft.tokens.len())
                    })
                })
                && draft
                    .tokens
                    .iter()
                    .all(|token| (*token as usize) < model_vocabulary.get())
        })
        && items
            .iter()
            .filter(|item| bundle_drafted(item))
            .all(|item| {
                drafts_by_request.contains_key(&item.request_id)
                    || accepted_by_request.contains_key(&item.request_id)
            });
    if !drafts_match {
        return false;
    }
    let expected_total: usize = items.iter().map(expected_output_rows).sum();
    let mut expected = items.iter().filter(|item| item.expects_logits);
    let rows_by_request = by_request(rows, LogitsRow::request_id);
    expected_total == rows.len() + sampled.len()
        && expected.all(|item| {
            let expected_rows = expected_output_rows(item);
            let item_rows = rows_by_request
                .get(&item.request_id)
                .map_or(&[][..], Vec::as_slice);
            let item_samples = sampled_by_request
                .get(&item.request_id)
                .map_or(&[][..], Vec::as_slice);
            // An item's rows are all logits rows or all sampled rows.
            (item_rows.len() == expected_rows && item_samples.is_empty()
                || item_samples.len() == expected_rows && item_rows.is_empty())
                && item_rows
                    .iter()
                    .all(|row| row.logits.len() == model_vocabulary.get())
                && item_samples.iter().all(|row| {
                    row.failure.is_some()
                        || row.token_id < model_vocabulary.get() as u32
                            && row.logprob.is_finite()
                            && top_count_matches(item, row.top_logprobs.len(), model_vocabulary)
                            && row.top_logprobs.iter().all(|top| {
                                top.token_id() < model_vocabulary.get() as u32
                                    && (top.logprob().is_finite()
                                        || (top.logprob() == f64::NEG_INFINITY
                                            && item
                                                .sampling
                                                .as_ref()
                                                .is_some_and(|work| work.constraint.is_some())))
                            })
                })
        })
}

/// Each request's entries in their order, found in one pass rather than by
/// scanning every entry for every item.
fn by_request<T>(entries: &[T], id: impl Fn(&T) -> RequestId) -> BTreeMap<RequestId, Vec<&T>> {
    let mut grouped: BTreeMap<RequestId, Vec<&T>> = BTreeMap::new();
    for entry in entries {
        grouped.entry(id(entry)).or_default().push(entry);
    }
    grouped
}

/// A row reports the requested number of top logprobs, including masked tails.
fn top_count_matches(item: &ModelBatchItem, count: usize, vocabulary: ModelVocabulary) -> bool {
    let Some(work) = item.sampling.as_ref() else {
        return count == 0;
    };
    let requested = work.parameters.top_logprobs().min(vocabulary.get());
    count == requested
}

/// Chooses each candidate's draft count and, for host drafts, fetches them.
///
/// Any drafter failure is left in `draft_error` and every count falls back to
/// zero, so the scheduler still receives a valid plan.
#[allow(clippy::too_many_arguments)]
fn plan_drafts<E: ModelExecutor>(
    policy: &mut SpeculationPolicy,
    executor: &mut E,
    requests: &BTreeMap<RequestId, RequestRecord>,
    vocabulary: ModelVocabulary,
    candidates: &[DraftCandidate],
    budget: DraftBudget,
    host_drafts: &mut Vec<(RequestId, Vec<u32>)>,
    draft_error: &mut Option<ExecutorError>,
) -> Vec<usize> {
    let none = vec![0; candidates.len()];
    if budget.spare_draft_tokens() == 0 || budget.spare_rows() == 0 {
        return none;
    }
    let mut plan_candidates = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let Some((acceptance, verified_last)) = requests
            .get(&candidate.request_id())
            .and_then(|record| Some((record.acceptance.as_ref()?, record.verified_last)))
        else {
            return none;
        };
        plan_candidates.push(PlanCandidate {
            maximum_draft_tokens: candidate.maximum_draft_tokens(),
            acceptance,
            fresh: !verified_last,
            remaining_tokens: candidate.remaining_tokens(),
        });
    }
    let mut drafts = policy.plan(&plan_candidates, budget);
    pin_drafts(
        requests,
        candidates,
        budget,
        policy.config().maximum_draft_items(),
        &mut drafts,
    );
    if drafts.iter().all(|&count| count == 0) {
        return drafts;
    }
    for count in &mut drafts {
        if *count > 0 {
            *count = executor.round_draft_tokens_v1(*count).min(*count);
        }
    }
    if policy.config().source() != DraftSource::Host {
        return drafts;
    }
    for (candidate, count) in candidates.iter().zip(drafts.iter_mut()) {
        if *count == 0 {
            continue;
        }
        let Some(record) = requests.get(&candidate.request_id()) else {
            return none;
        };
        match executor.draft_tokens_v1(candidate.request_id(), &record.history, *count) {
            Ok(mut tokens) => {
                if tokens
                    .iter()
                    .any(|token| *token as usize >= vocabulary.get())
                {
                    *draft_error = Some(
                        ExecutorError::new("host drafter proposed a token outside the vocabulary")
                            .expect("fixed executor error is valid"),
                    );
                    return none;
                }
                tokens.truncate(*count);
                *count = tokens.len();
                if !tokens.is_empty() {
                    host_drafts.push((candidate.request_id(), tokens));
                }
            }
            Err(error) => {
                *draft_error = Some(error);
                return none;
            }
        }
    }
    drafts
}

/// Replaces the planned drafts of requests that pin theirs, clamped to each
/// candidate's limit, and keeps the step within the draft budget and the
/// drafting item limit: pinned counts are granted in candidate order, and
/// the policy's own drafts give way first.
fn pin_drafts(
    requests: &BTreeMap<RequestId, RequestRecord>,
    candidates: &[DraftCandidate],
    budget: DraftBudget,
    item_limit: Option<usize>,
    drafts: &mut [usize],
) {
    let pins: Vec<Option<usize>> = candidates
        .iter()
        .map(|candidate| {
            requests
                .get(&candidate.request_id())
                .and_then(|record| record.pinned_draft_tokens)
                .map(|tokens| tokens.min(candidate.maximum_draft_tokens()))
        })
        .collect();
    if pins.iter().all(Option::is_none) {
        return;
    }
    let mut left = budget.spare_draft_tokens().min(budget.spare_rows());
    let mut items = item_limit.unwrap_or(usize::MAX);
    for pinned in [true, false] {
        for (draft, pin) in drafts.iter_mut().zip(&pins) {
            if pin.is_some() != pinned {
                continue;
            }
            let wanted = pin.unwrap_or(*draft);
            *draft = if items == 0 { 0 } else { wanted.min(left) };
            if *draft > 0 {
                left -= *draft;
                items -= 1;
            }
        }
    }
}

fn model_items_parts(
    requests: &BTreeMap<RequestId, RequestRecord>,
    batch: &Batch,
    draft_source: Option<DraftSource>,
    host_drafts: &[(RequestId, Vec<u32>)],
    media: Vec<(Vec<MediaEncodeV1>, Vec<MediaInjectV1>)>,
) -> Result<Vec<ModelBatchItem>, EngineError> {
    let mut media = media.into_iter();
    let mut items = Vec::new();
    items
        .try_reserve_exact(batch.items().len())
        .map_err(|_| EngineError::MetadataAllocationFailed)?;
    for item in batch.items() {
        let record = requests
            .get(&item.request_id())
            .ok_or(EngineError::InvariantViolation)?;
        let mut draft = None;
        let input_tokens = match item.phase() {
            // The prompt, then the emitted tokens a preempted request
            // recomputes; the history holds both in order.
            BatchPhase::Prefill { tokens } => {
                let end = record
                    .prompt_offset
                    .checked_add(tokens)
                    .ok_or(EngineError::InvariantViolation)?;
                record
                    .history
                    .get(record.prompt_offset..end)
                    .ok_or(EngineError::InvariantViolation)?
                    .to_vec()
            }
            BatchPhase::Decode => vec![
                record
                    .pending_decode_token
                    .ok_or(EngineError::InvariantViolation)?,
            ],
            BatchPhase::Verify { draft_tokens } => {
                let source = draft_source.ok_or(EngineError::InvariantViolation)?;
                let mut input = Vec::new();
                input
                    .try_reserve_exact(draft_tokens + 1)
                    .map_err(|_| EngineError::MetadataAllocationFailed)?;
                input.push(
                    record
                        .pending_decode_token
                        .ok_or(EngineError::InvariantViolation)?,
                );
                if source == DraftSource::Host {
                    let (_, drafts) = host_drafts
                        .iter()
                        .find(|(id, _)| *id == item.request_id())
                        .ok_or(EngineError::InvariantViolation)?;
                    if drafts.len() != draft_tokens {
                        return Err(EngineError::InvariantViolation);
                    }
                    input.extend_from_slice(drafts);
                }
                draft = Some(DraftWork::new(draft_tokens, source));
                input
            }
        };
        // The first input row's sequence index: the prompt offset of a
        // prefill chunk, or the pending token, which is the last of the
        // history, for a decode or verify item.
        let position_axes = match &record.positions {
            None => Vec::new(),
            Some(positions) => {
                let first = match item.phase() {
                    BatchPhase::Prefill { .. } => record.prompt_offset,
                    BatchPhase::Decode | BatchPhase::Verify { .. } => record
                        .history
                        .len()
                        .checked_sub(1)
                        .ok_or(EngineError::InvariantViolation)?,
                };
                let rows = match item.phase() {
                    BatchPhase::Verify { draft_tokens } => draft_tokens + 1,
                    _ => input_tokens.len(),
                };
                positions.rows(first, rows)
            }
        };
        let prompt_targets = if record.prompt_logprobs.is_some()
            && matches!(item.phase(), BatchPhase::Prefill { .. })
        {
            (record.prompt_offset..record.prompt_offset + input_tokens.len())
                .map(|position| record.prompt_tokens.get(position + 1).copied())
                .collect()
        } else {
            Vec::new()
        };
        items.push(ModelBatchItem {
            prefill_boundary: None,
            output_cycle: None,
            request_id: item.request_id(),
            phase: item.phase(),
            input_tokens,
            expects_logits: item.expects_output(),
            prompt_targets,
            prompt_top_logprobs: record.prompt_logprobs.unwrap_or(0),
            sampling: item.expects_output().then(|| SamplingWork {
                parameters: record.sampling.clone(),
                state: record.sampling_state,
                history: SamplingHistory::new(Arc::clone(&record.history)),
                emitted_tokens: record.emitted_tokens,
                stop_ids: record.stop_ids.clone(),
                constraint: record
                    .constraint
                    .as_ref()
                    .map(|state| RowConstraint::new(state.fork())),
            }),
            draft,
            media_encodes: Vec::new(),
            media_injects: Vec::new(),
            position_axes,
            anchor: item.anchor_tokens().is_some(),
        });
        if let Some((encodes, injects)) = media.next() {
            let item = items.last_mut().expect("item was just pushed");
            item.media_encodes = encodes;
            item.media_injects = injects;
        }
    }
    Ok(items)
}

/// The media work of one step.
#[derive(Debug, Default)]
struct MediaPlacement {
    /// Encode records and inject runs per batch item, in batch order; empty
    /// when no item carries media.
    items: Vec<(Vec<MediaEncodeV1>, Vec<MediaInjectV1>)>,
    /// Media items pinned this step, by request, item and key; the commit
    /// releases each pin and records the item as placed.
    pinned: Vec<(RequestId, usize, [u8; 32])>,
    encodes: usize,
    encoded_rows: u64,
    /// Items a request found resident the first time it placed them.
    hits: usize,
    evictions: usize,
    /// Items encoded again for a request that had placed them before.
    reencodes: usize,
}

/// Token-equivalent cost of the encodes a prefill chunk would start.
///
/// It counts each media item with a span in the chunk that is not resident,
/// since a resident item is a cache hit. An item repeated in one request is
/// counted once.
fn media_encode_cost(
    media: MediaConfigV1,
    arena: &MediaArenaV1,
    record: &RequestRecord,
    offset: usize,
    tokens: usize,
) -> usize {
    let end = offset + tokens;
    let mut cost = 0usize;
    // Each distinct item the chunk covers that the arena lacks encodes once.
    let mut covered = std::collections::HashSet::<[u8; 32]>::new();
    for span in &record.spans {
        if (span.prompt_row as usize) >= end || (span.prompt_row + span.rows) as usize <= offset {
            continue;
        }
        let Some(item) = record.media.get(usize::from(span.media)) else {
            continue;
        };
        if covered.insert(item.key) && !arena.resident(&item.key) {
            cost = cost.saturating_add(media.encode_cost(item.encoder, item.output_rows));
        }
    }
    cost
}

/// The prompt row after each of `items` media items' last span.
fn media_ends(items: usize, spans: &[MediaSpanV1]) -> Vec<usize> {
    let mut ends = vec![0; items];
    for span in spans {
        if let Some(end) = ends.get_mut(usize::from(span.media)) {
            *end = (*end).max((span.prompt_row + span.rows) as usize);
        }
    }
    ends
}

/// Pins every media item a prefill chunk covers for this step and builds its
/// encodes and inject runs. An item a request placed in an earlier step and
/// that was evicted since is encoded again. The scheduler keeps the pages the
/// step covers within the arena, so pinning finds room; the pending batch
/// guard rolls the arena back on failure.
fn place_media(
    arena: &mut MediaArenaV1,
    media: MediaConfigV1,
    requests: &BTreeMap<RequestId, RequestRecord>,
    batch: &Batch,
) -> Result<MediaPlacement, EngineError> {
    let mut placement = MediaPlacement::default();
    if !batch.items().iter().any(|item| {
        matches!(item.phase(), BatchPhase::Prefill { .. })
            && requests
                .get(&item.request_id())
                .is_some_and(|record| !record.spans.is_empty())
    }) {
        return Ok(placement);
    }
    let page_rows = media.page_rows();
    for item in batch.items() {
        let mut encodes = Vec::new();
        let mut injects = Vec::new();
        let record = requests
            .get(&item.request_id())
            .ok_or(EngineError::InvariantViolation)?;
        if let BatchPhase::Prefill { tokens } = item.phase() {
            let start = record.prompt_offset;
            let end = start + tokens;
            for span in &record.spans {
                let span_start = span.prompt_row as usize;
                let span_end = span_start + span.rows as usize;
                if span_end <= start || span_start >= end {
                    continue;
                }
                let index = usize::from(span.media);
                let input = &record.media[index];
                let pinned_now = placement
                    .pinned
                    .iter()
                    .any(|(id, media, _)| *id == item.request_id() && *media == index);
                if !pinned_now {
                    let pinned = arena
                        .pin(input.key, input.output_rows)
                        .ok_or(EngineError::InvariantViolation)?;
                    placement.evictions += pinned.evicted;
                    if pinned.encode {
                        placement.encodes += 1;
                        placement.encoded_rows += u64::from(input.output_rows);
                        if record.media_placed[index] {
                            placement.reencodes += 1;
                        }
                        encodes.push(MediaEncodeV1 {
                            encoder: input.encoder,
                            geometry: input.geometry,
                            output_rows: input.output_rows,
                            pages: pinned.pages,
                            payload: input.payload.clone(),
                        });
                    } else if !record.media_placed[index] {
                        placement.hits += 1;
                    }
                    placement.pinned.push((item.request_id(), index, input.key));
                }
                let pages = arena
                    .pages(&input.key)
                    .ok_or(EngineError::InvariantViolation)?;
                let from = span_start.max(start);
                let to = span_end.min(end);
                inject_runs(
                    pages,
                    page_rows,
                    (from - start) as u32,
                    span.media_row + (from - span_start) as u32,
                    (to - from) as u32,
                    &mut injects,
                );
            }
        }
        placement.items.push((encodes, injects));
    }
    Ok(placement)
}

/// Releases what the request kept for the media items whose spans all end
/// at or before prompt row `passed`: the arena may evict them first, and
/// their payloads, which only an encode reads, are dropped, so their host
/// memory returns before the request retires. A prefix hit moves the prompt
/// offset only once its restore succeeds, so a refused restore, which
/// prefills from the start, still finds every payload.
fn release_passed_media(arena: &mut MediaArenaV1, record: &mut RequestRecord, passed: usize) {
    for index in 0..record.media.len() {
        if record.media_wanted[index] && record.media_ends[index] <= passed {
            arena.unwant(&record.media[index].key);
            record.media_wanted[index] = false;
            record.media[index].payload = MediaPayloadV1::from(Vec::new());
        }
    }
}

#[derive(Clone, Debug)]
enum RequestSampleError {
    Sampling(SamplingError),
    Constraint(GrammarError),
}

impl fmt::Display for RequestSampleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sampling(error) => error.fmt(formatter),
            Self::Constraint(error) => error.fmt(formatter),
        }
    }
}

impl From<SamplingError> for RequestSampleError {
    fn from(error: SamplingError) -> Self {
        Self::Sampling(error)
    }
}

impl From<RequestSampleError> for EngineError {
    fn from(error: RequestSampleError) -> Self {
        match error {
            RequestSampleError::Sampling(error) => Self::Sampling(error),
            RequestSampleError::Constraint(error) => Self::Constraint(error),
        }
    }
}

fn constraint_sample_error(
    id: RequestId,
    error: GrammarError,
    fallback: SamplingError,
) -> SampleStageError {
    let error = match error {
        GrammarError::TooLarge(_) => RequestSampleError::Constraint(error),
        _ => RequestSampleError::Sampling(fallback),
    };
    SampleStageError::Request(id, error)
}

#[derive(Debug)]
enum SampleStageError {
    Request(RequestId, RequestSampleError),
    Global(EngineError),
}

/// Samples each item's rows and applies acceptance to verify items.
///
/// Row j of a verify item is sampled as the step after the bonus input and
/// the first j drafts would have been: its history holds those drafts, its
/// emitted count and random stream are j tokens further on. The run takes
/// row 0, then row j + 1 only while row j's sample equals draft j, so every
/// emitted token is the target's own sample given the tokens before it. A
/// stop token or the request's length limit ends the run early.
#[allow(
    clippy::too_many_arguments,
    reason = "keeps the independently validated result channels explicit"
)]
fn stage_samples(
    requests: &BTreeMap<RequestId, RequestRecord>,
    items: &[ModelBatchItem],
    rows: &[LogitsRow],
    sampled_rows: &[SampledRow],
    drafts: &[DraftRow],
    accepted_prefixes: &[AcceptedPrefix],
    model_vocabulary: ModelVocabulary,
    skipped: &[RequestId],
) -> Result<Vec<Option<StagedRun>>, SampleStageError> {
    let mut staged = Vec::new();
    staged
        .try_reserve_exact(items.len())
        .map_err(|_| SampleStageError::Global(EngineError::MetadataAllocationFailed))?;
    let samples_by_request = by_request(sampled_rows, SampledRow::request_id);
    let rows_by_request = by_request(rows, LogitsRow::request_id);
    let drafts_by_request = by_request(drafts, DraftRow::request_id);
    for item in items {
        if !item.expects_logits || skipped.contains(&item.request_id) {
            staged.push(None);
            continue;
        }
        let record = requests
            .get(&item.request_id)
            .expect("scheduled request has engine state");
        if let Some(prefix) = accepted_prefixes
            .iter()
            .find(|prefix| prefix.request_id == item.request_id)
        {
            let samples = samples_by_request
                .get(&item.request_id)
                .map_or(&[][..], Vec::as_slice);
            staged.push(Some(prefix.stage(record, samples)?));
            continue;
        }
        let draft_tokens: &[u32] = match item.draft {
            None => &[],
            Some(work) => match work.source {
                DraftSource::Host => &item.input_tokens[1..],
                DraftSource::Bundle => drafts_by_request
                    .get(&item.request_id)
                    .and_then(|drafts| drafts.first())
                    .map(|draft| draft.tokens.as_slice())
                    .expect("executor draft membership was checked"),
            },
        };
        let item_samples = samples_by_request
            .get(&item.request_id)
            .map_or(&[][..], Vec::as_slice);
        let item_rows = rows_by_request
            .get(&item.request_id)
            .map_or(&[][..], Vec::as_slice);
        let extended_history = if item_samples.is_empty() && !draft_tokens.is_empty() {
            let mut history = Vec::new();
            history
                .try_reserve_exact(record.history.len() + draft_tokens.len())
                .map_err(|_| SampleStageError::Global(EngineError::MetadataAllocationFailed))?;
            history.extend_from_slice(&record.history);
            history.extend_from_slice(draft_tokens);
            history
        } else {
            Vec::new()
        };
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(draft_tokens.len() + 1)
            .map_err(|_| SampleStageError::Global(EngineError::MetadataAllocationFailed))?;
        let mut state = record.sampling_state;
        let mut accepted = 0usize;
        let mut terminal = false;
        let mut rejected = false;
        // The constraint follows the run token by token; it is at the row's
        // position because the run continues only over matched drafts.
        let mut constraint = record.constraint.as_ref().map(ConstraintState::fork);
        let first_row = item
            .sampling
            .as_ref()
            .and_then(|work| work.constraint.as_ref());
        for row in 0..=draft_tokens.len() {
            let derived;
            let row_constraint = match (&constraint, first_row) {
                (None, _) => None,
                (Some(_), Some(first)) if row == 0 => Some(first),
                (Some(current), _) => {
                    derived = RowConstraint::new(current.fork());
                    Some(&derived)
                }
            };
            if let Some(error) = row_constraint.and_then(RowConstraint::error) {
                return Err(constraint_sample_error(
                    item.request_id,
                    error.clone(),
                    SamplingError::AllTokensMasked,
                ));
            }
            let token = if let Some(sampled) = item_samples.get(row) {
                if let Some(error) = sampled.failure() {
                    return Err(SampleStageError::Request(item.request_id, error.into()));
                }
                state = record.sampling_state.after(row as u64 + 1);
                StagedToken {
                    token_id: sampled.token_id(),
                    logprob: sampled.logprob(),
                    top_logprobs: sampled.top_logprobs().to_vec(),
                }
            } else {
                let logits = item_rows
                    .get(row)
                    .expect("executor output membership was checked");
                let history = if row == 0 {
                    record.history.as_slice()
                } else {
                    &extended_history[..record.history.len() + row]
                };
                let mut row_state = record.sampling_state.after(row as u64);
                let context =
                    SampleContext::new(history, record.emitted_tokens + row, &record.stop_ids);
                let context = match row_constraint.and_then(RowConstraint::mask) {
                    Some(mask) => context.with_allowed_tokens(mask),
                    None => context,
                };
                let sampled = sample_for_vocabulary(
                    &record.sampling,
                    logits.logits(),
                    model_vocabulary,
                    context,
                    &mut row_state,
                )
                .map_err(|error| SampleStageError::Request(item.request_id, error.into()))?;
                state = row_state;
                StagedToken {
                    token_id: sampled.token_id(),
                    logprob: sampled.logprob(),
                    top_logprobs: sampled.top_logprobs().to_vec(),
                }
            };
            for top in &token.top_logprobs {
                if top.logprob() == f64::NEG_INFINITY {
                    let allowed = constraint
                        .as_ref()
                        .map_or(Ok(true), |current| current.allows(top.token_id()))
                        .map_err(|error| {
                            constraint_sample_error(
                                item.request_id,
                                error,
                                SamplingError::DisallowedToken(top.token_id()),
                            )
                        })?;
                    if allowed {
                        return Err(SampleStageError::Request(
                            item.request_id,
                            SamplingError::InvalidDistribution.into(),
                        ));
                    }
                }
            }
            if let Some(constraint) = &mut constraint {
                constraint.accept(token.token_id).map_err(|error| {
                    constraint_sample_error(
                        item.request_id,
                        error,
                        SamplingError::DisallowedToken(token.token_id),
                    )
                })?;
            }
            let matched = draft_tokens.get(row) == Some(&token.token_id);
            accepted += usize::from(matched);
            rejected = !matched && row < draft_tokens.len();
            let stop = record
                .termination_stop_ids
                .binary_search(&token.token_id)
                .is_ok();
            tokens.push(token);
            if stop {
                terminal = true;
                break;
            }
            if record.emitted_tokens + tokens.len() == record.maximum_new_tokens {
                break;
            }
            if !matched {
                break;
            }
        }
        staged.push(Some(StagedRun {
            tokens,
            state,
            terminal,
            drafted: draft_tokens.len(),
            accepted,
            rejected,
            constraint,
        }));
    }
    Ok(staged)
}

fn preflight_apply(
    requests: &BTreeMap<RequestId, RequestRecord>,
    events: &mut VecDeque<EngineEvent>,
    next_event_sequence: u64,
    batch: &Batch,
    model_items: &[ModelBatchItem],
    prompt_rows: &[LogitsRow],
    staged: Vec<Option<StagedRun>>,
) -> Result<PreparedEngineApply, EngineError> {
    if staged.len() != batch.items().len() || model_items.len() != batch.items().len() {
        return Err(EngineError::InvariantViolation);
    }
    let mut event_count = 0usize;
    for ((item, model_item), sample) in batch.items().iter().zip(model_items).zip(&staged) {
        event_count = event_count
            .checked_add(usize::from(
                model_item.prompt_targets().iter().any(Option::is_some),
            ))
            .ok_or(EngineError::EventSequenceExhausted)?;
        let record = requests
            .get(&item.request_id())
            .ok_or(EngineError::InvariantViolation)?;
        if let BatchPhase::Prefill { tokens } = item.phase() {
            let prompt_offset = record
                .prompt_offset
                .checked_add(tokens)
                .ok_or(EngineError::InvariantViolation)?;
            if prompt_offset > record.history.len() {
                return Err(EngineError::InvariantViolation);
            }
        }
        if let Some(run) = sample {
            let emitted = record
                .emitted_tokens
                .checked_add(run.tokens.len())
                .ok_or(EngineError::InvariantViolation)?;
            if run.tokens.is_empty()
                || emitted > record.maximum_new_tokens
                || record.history.len() + run.tokens.len() > record.history.capacity()
            {
                return Err(EngineError::InvariantViolation);
            }
            event_count = event_count
                .checked_add(
                    usize::from(run.terminal || emitted == record.maximum_new_tokens)
                        + run.tokens.len(),
                )
                .ok_or(EngineError::EventSequenceExhausted)?;
        }
    }
    let sequence_count =
        u64::try_from(event_count).map_err(|_| EngineError::EventSequenceExhausted)?;
    next_event_sequence
        .checked_add(sequence_count)
        .ok_or(EngineError::EventSequenceExhausted)?;
    events
        .try_reserve(event_count)
        .map_err(|_| EngineError::MetadataAllocationFailed)?;
    let mut prepared_requests = Vec::new();
    prepared_requests
        .try_reserve_exact(batch.items().len())
        .map_err(|_| EngineError::MetadataAllocationFailed)?;
    let mut prepared_events = VecDeque::new();
    prepared_events
        .try_reserve(event_count)
        .map_err(|_| EngineError::MetadataAllocationFailed)?;
    let mut sequence = next_event_sequence;
    let mut prompt_rows = prompt_rows.iter();
    for ((item, model_item), sample) in batch.items().iter().zip(model_items).zip(staged) {
        let record = &requests[&item.request_id()];
        let mut prompt_positions = Vec::new();
        for (index, target) in model_item.prompt_targets().iter().enumerate() {
            let row = prompt_rows.next().ok_or(EngineError::InvariantViolation)?;
            if let Some(token_id) = target {
                let (logprob, top_logprobs) =
                    if let Some((scored_token, logprob, top)) = row.prompt_score() {
                        if scored_token != *token_id
                            || top.len() != model_item.prompt_top_logprobs()
                            || !logprob.is_finite()
                        {
                            return Err(EngineError::OutputMismatch);
                        }
                        (logprob, top.to_vec())
                    } else {
                        runtime_sampling::score_token_logprobs(
                            row.logits(),
                            *token_id,
                            model_item.prompt_top_logprobs(),
                        )
                        .map_err(|_| EngineError::OutputMismatch)?
                    };
                prompt_positions.push(PromptTokenLogprob {
                    position: record.prompt_offset + index + 1,
                    token_id: *token_id,
                    logprob,
                    top_logprobs,
                });
            }
        }
        if !prompt_positions.is_empty() {
            prepared_events.push_back(EngineEvent {
                sequence,
                request_id: item.request_id(),
                kind: EngineEventKind::PromptLogprobs {
                    positions: prompt_positions,
                },
            });
            sequence += 1;
        }
        let prompt_offset = match item.phase() {
            BatchPhase::Prefill { tokens } => record
                .prompt_offset
                .checked_add(tokens)
                .ok_or(EngineError::InvariantViolation)?,
            BatchPhase::Decode | BatchPhase::Verify { .. } => record.prompt_offset,
        };
        let terminal = sample.as_ref().is_some_and(|run| run.terminal);
        let emitted_tokens = record
            .emitted_tokens
            .checked_add(sample.as_ref().map_or(0, |run| run.tokens.len()))
            .ok_or(EngineError::InvariantViolation)?;
        let prepared_sample = if let Some(run) = sample {
            let mut tokens = Vec::new();
            tokens
                .try_reserve_exact(run.tokens.len())
                .map_err(|_| EngineError::MetadataAllocationFailed)?;
            for token in run.tokens {
                tokens.push(token.token_id);
                prepared_events.push_back(EngineEvent {
                    sequence,
                    request_id: item.request_id(),
                    kind: EngineEventKind::Token {
                        token_id: token.token_id,
                        logprob: token.logprob,
                        top_logprobs: token.top_logprobs,
                    },
                });
                sequence += 1;
            }
            if terminal || emitted_tokens == record.maximum_new_tokens {
                prepared_events.push_back(EngineEvent {
                    sequence,
                    request_id: item.request_id(),
                    kind: EngineEventKind::Finished {
                        reason: if terminal {
                            FinishReason::Stop
                        } else {
                            FinishReason::Length
                        },
                    },
                });
                sequence += 1;
            }
            Some(PreparedSampleApply {
                tokens,
                state: run.state,
                constraint: run.constraint,
            })
        } else {
            None
        };
        prepared_requests.push(PreparedRequestApply {
            request_id: item.request_id(),
            prompt_offset,
            emitted_tokens,
            sample: prepared_sample,
        });
    }
    Ok(PreparedEngineApply {
        requests: prepared_requests,
        events: prepared_events,
        next_event_sequence: sequence,
    })
}

fn release_histories(items: &mut [ModelBatchItem]) {
    for work in items.iter_mut().filter_map(|item| item.sampling.as_mut()) {
        work.history = SamplingHistory::default();
    }
}

/// The history to append to. The step's sampling work, which shares it, is
/// released before this; a copy that outlived it keeps the preflighted
/// capacity, so the append still cannot reallocate.
fn unique_history(history: &mut Arc<Vec<u32>>) -> &mut Vec<u32> {
    if Arc::get_mut(history).is_none() {
        let mut copy = Vec::with_capacity(history.capacity());
        copy.extend_from_slice(history);
        *history = Arc::new(copy);
    }
    Arc::get_mut(history).expect("the history was just made unique")
}

fn apply_committed_parts(
    requests: &mut BTreeMap<RequestId, RequestRecord>,
    events: &mut VecDeque<EngineEvent>,
    next_event_sequence: &mut u64,
    prepared: PreparedEngineApply,
) {
    for update in prepared.requests {
        let record = requests
            .get_mut(&update.request_id)
            .expect("commit preflight checked request membership");
        record.prompt_offset = update.prompt_offset;
        record.emitted_tokens = update.emitted_tokens;
        if let Some(sample) = update.sample {
            record.sampling_state = sample.state;
            record.pending_decode_token = sample.tokens.last().copied();
            unique_history(&mut record.history).extend_from_slice(&sample.tokens);
            if sample.constraint.is_some() {
                record.constraint = sample.constraint;
            }
        }
    }
    for event in prepared.events {
        events.push_back(event);
    }
    *next_event_sequence = prepared.next_event_sequence;
}

fn fail_batch_parts(
    pending_batch: &mut PendingBatchGuard<'_>,
    pending_retirements: &mut Vec<RequestId>,
    metrics: &EngineMetrics,
    events: &mut VecDeque<EngineEvent>,
    next_event_sequence: &mut u64,
    batch: &Batch,
    message: &str,
) -> Result<(), EngineError> {
    pending_batch.rollback().map_err(EngineError::Scheduler)?;
    for item in batch.items() {
        pending_batch
            .scheduler_mut()
            .fail(item.request_id())
            .map_err(EngineError::Scheduler)?;
        pending_retirements.push(item.request_id());
        metrics.request_failed(1);
        metrics.compatibility(|metrics| metrics.fail(item.request_id()));
    }
    let event_count =
        u64::try_from(batch.items().len()).map_err(|_| EngineError::MetadataAllocationFailed)?;
    next_event_sequence
        .checked_add(event_count)
        .ok_or(EngineError::EventSequenceExhausted)?;
    events
        .try_reserve(batch.items().len())
        .map_err(|_| EngineError::MetadataAllocationFailed)?;
    for item in batch.items() {
        push_event_parts(
            events,
            next_event_sequence,
            item.request_id(),
            EngineEventKind::Error {
                message: message.to_owned(),
            },
        )?;
    }
    Ok(())
}

fn fail_request_parts(
    pending_batch: &mut PendingBatchGuard<'_>,
    pending_retirements: &mut Vec<RequestId>,
    metrics: &EngineMetrics,
    events: &mut VecDeque<EngineEvent>,
    next_event_sequence: &mut u64,
    id: RequestId,
    message: &str,
) -> Result<(), EngineError> {
    pending_batch.rollback().map_err(EngineError::Scheduler)?;
    pending_batch
        .scheduler_mut()
        .fail(id)
        .map_err(EngineError::Scheduler)?;
    pending_retirements.push(id);
    metrics.request_failed(1);
    metrics.compatibility(|metrics| metrics.fail(id));
    push_event_parts(
        events,
        next_event_sequence,
        id,
        EngineEventKind::Error {
            message: message.to_owned(),
        },
    )
}

fn push_event_parts(
    events: &mut VecDeque<EngineEvent>,
    next_event_sequence: &mut u64,
    id: RequestId,
    kind: EngineEventKind,
) -> Result<(), EngineError> {
    let sequence = *next_event_sequence;
    *next_event_sequence = next_event_sequence
        .checked_add(1)
        .ok_or(EngineError::EventSequenceExhausted)?;
    events.push_back(EngineEvent {
        sequence,
        request_id: id,
        kind,
    });
    Ok(())
}

/// Commits the scheduler, then the model. `kept` is present exactly when the
/// batch holds verify items. A scheduler failure rolls the model back; a
/// model failure comes back beside the committed scheduler result.
fn commit_scheduler_then_model<S, T, F>(
    transaction: ModelTransaction<S>,
    kept: Option<&[KeptRows]>,
    scheduler_commit: F,
) -> Result<(T, Result<(), ExecutorError>), ScheduleError>
where
    S: ModelTransactionState,
    F: FnOnce() -> Result<T, ScheduleError>,
{
    let result = {
        let _timer = profile::PhaseTimer::traced(profile::StepPhase::SchedulerCommit);
        scheduler_commit()?
    };
    let _timer = profile::PhaseTimer::traced(profile::StepPhase::ModelCommit);
    let model = match kept {
        Some(kept) => transaction.commit_kept(kept),
        None => transaction.commit(),
    };
    Ok((result, model))
}

fn commit_scheduler_then_model_observed<S, T, F>(
    transaction: ModelTransaction<S>,
    kept: Option<&[KeptRows]>,
    scheduler_commit: F,
    metrics: &EngineMetrics,
) -> Result<(T, Result<(), ExecutorError>), ScheduleError>
where
    S: ModelTransactionState,
    F: FnOnce() -> Result<T, ScheduleError>,
{
    commit_scheduler_then_model(transaction, kept, scheduler_commit).inspect_err(|_| {
        metrics.error();
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineConfigError {
    VocabularyTooLarge { provided: usize, maximum: usize },
    InvalidKvPageCapacity,
    ZeroMaximumSequenceTokens,
    MaximumSequenceTokensTooLarge { provided: usize, maximum: usize },
    MaximumSequenceTokensExceedKvCapacity { provided: usize, maximum: usize },
    ZeroTerminalRetirement,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineRequestError {
    EmptyPrompt,
    PromptTokenOutOfVocabulary {
        position: usize,
        token: u32,
        vocabulary: usize,
    },
    VocabularyMismatch {
        request: usize,
        engine: usize,
    },
    StopTokenOutOfVocabulary {
        token: u32,
        vocabulary: usize,
    },
    LogitBiasTokenOutOfVocabulary {
        token: u32,
        vocabulary: usize,
    },
    ZeroMaximumNewTokens,
    LengthOverflow,
    HistoryTooLarge,
    SequenceTooLong {
        maximum: usize,
        requested: usize,
    },
    MinimumExceedsMaximum,
    TooManyStopIds,
    DuplicateStopId,
    SchedulerRequest,
    Media(MediaRequestError),
    PositionAxes(PositionAxesError),
}

impl From<runtime_scheduler::InvalidRequest> for EngineRequestError {
    fn from(_: runtime_scheduler::InvalidRequest) -> Self {
        Self::SchedulerRequest
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutorErrorKind {
    EmptyMessage,
    MessageTooLong,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EngineError {
    Scheduler(ScheduleError),
    Submit(SubmitError),
    InvalidRequest(EngineRequestError),
    Executor(ExecutorError),
    Sampling(SamplingError),
    Constraint(GrammarError),
    OutputMismatch,
    MetadataAllocationFailed,
    EventSequenceExhausted,
    StepLimitReached,
    InvariantViolation,
    RetirementFailed,
    PrefixStateUnconfirmed,
    KvTransferDrainTimeout,
    ModelStateFailed,
}

impl fmt::Display for EngineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scheduler(error) => write!(formatter, "scheduler failed: {error}"),
            Self::Submit(error) => write!(formatter, "request submission failed: {error}"),
            Self::InvalidRequest(error) => {
                write!(formatter, "engine request is invalid: {error:?}")
            }
            Self::Executor(error) => write!(formatter, "executor failed: {error}"),
            Self::Sampling(error) => write!(formatter, "sampling failed: {error}"),
            Self::Constraint(error) => write!(formatter, "constraint failed: {error}"),
            Self::OutputMismatch => {
                formatter.write_str("executor output rows do not match the pending batch")
            }
            Self::MetadataAllocationFailed => {
                formatter.write_str("engine metadata allocation failed")
            }
            Self::EventSequenceExhausted => formatter.write_str("event sequence is exhausted"),
            Self::StepLimitReached => formatter.write_str("run-until-idle step limit was reached"),
            Self::RetirementFailed => {
                formatter.write_str("request retirement failed; engine execution is blocked")
            }
            Self::KvTransferDrainTimeout => {
                formatter.write_str("KV transfer shutdown drain timed out; leases remain retained")
            }
            Self::PrefixStateUnconfirmed => formatter
                .write_str("a prefix state copy may still be running; engine execution is blocked"),
            Self::InvariantViolation => formatter.write_str("engine invariant audit failed"),
            Self::ModelStateFailed => formatter
                .write_str("a model commit or rollback failed; engine execution is blocked"),
        }
    }
}

impl Error for EngineError {}

#[cfg(test)]
mod transaction_tests {
    use std::cell::{Cell, RefCell};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::rc::Rc;

    use super::*;

    #[derive(Clone, Debug, Default)]
    struct AlwaysFails {
        retired: Vec<RequestId>,
        emit_rows: bool,
    }

    impl ModelExecutor for AlwaysFails {
        fn retire_request_v1(&mut self, id: RequestId) -> Result<(), ExecutorError> {
            self.retired.push(id);
            Ok(())
        }

        type TransactionState<'a> = NoopModelTransactionState;

        fn execute<'a>(
            &'a mut self,
            items: &[ModelBatchItem],
            _kv_plan: &PreparedAppendBatch,
        ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
            if self.emit_rows {
                return Ok(ModelTransaction::no_op(
                    items
                        .iter()
                        .filter(|item| item.expects_logits())
                        .map(|item| LogitsRow::new(item.request_id(), vec![1.0; 8]).unwrap())
                        .collect(),
                ));
            }
            Err(ExecutorError::new("injected failure").unwrap())
        }
    }

    #[test]
    fn a_mismatched_stop_draft_counts_as_a_rejection() {
        let vocabulary = ModelVocabulary::new(8).unwrap();
        let config = EngineConfig::new(
            SchedulerConfig::new(1, 2, 8, 8, 1).unwrap(),
            KvGeometry::new(4, 8).unwrap(),
            vocabulary,
            16,
            4,
        )
        .unwrap();
        let mut engine = Engine::new(config, AlwaysFails::default()).unwrap();
        let id = RequestId::new(1);
        engine
            .submit(
                EngineRequest::new(
                    id,
                    vec![0],
                    8,
                    SamplingParameters::new(0.0).unwrap(),
                    0,
                    vec![3],
                    vocabulary,
                )
                .unwrap(),
            )
            .unwrap();
        for (drafts, samples, accepted, rejected) in [
            (vec![1], vec![3], 0, true),
            (vec![1, 2], vec![1, 3], 1, true),
            (vec![3], vec![3], 1, false),
            (vec![1], vec![1, 3], 1, false),
        ] {
            let mut inputs = vec![0];
            inputs.extend_from_slice(&drafts);
            let item = ModelBatchItem::new(id, BatchPhase::Decode, inputs, true)
                .with_draft(DraftWork::new(drafts.len(), DraftSource::Host));
            let sampled: Vec<_> = samples
                .into_iter()
                .map(|token| SampledRow::new(id, token, -0.5, Vec::new()))
                .collect();
            let staged = stage_samples(
                &engine.requests,
                &[item],
                &[],
                &sampled,
                &[],
                &[],
                vocabulary,
                &[],
            )
            .unwrap();
            let run = staged[0].as_ref().unwrap();
            assert!(run.terminal);
            assert_eq!(run.accepted, accepted);
            assert_eq!(run.rejected, rejected);
        }
    }

    #[test]
    fn unexpected_position_identity_returns_a_preemption_error() {
        let vocabulary = ModelVocabulary::new(8).unwrap();
        let config = EngineConfig::new(
            SchedulerConfig::new(1, 2, 8, 8, 1).unwrap(),
            KvGeometry::new(4, 8).unwrap(),
            vocabulary,
            16,
            4,
        )
        .unwrap()
        .with_prefix_cache(1)
        .with_admission(AdmissionPolicy::Optimistic(
            runtime_scheduler::OptimisticAdmission::new(0, 2)
                .unwrap()
                .with_forced_preemption(1)
                .unwrap(),
        ));
        let mut e = Engine::new(
            config,
            AlwaysFails {
                emit_rows: true,
                ..AlwaysFails::default()
            },
        )
        .unwrap();
        let id = RequestId::new(1);
        e.submit(
            EngineRequest::new(
                id,
                vec![1; 4],
                8,
                SamplingParameters::new(0.0).unwrap(),
                0,
                vec![],
                vocabulary,
            )
            .unwrap(),
        )
        .unwrap();
        e.step().unwrap();
        // Inject a broken pinning invariant after normal admission.
        e.requests.get_mut(&id).unwrap().positions = Some(PositionAxesV1::new(4, 12, &[]).unwrap());
        assert!(matches!(e.preempt(), Err(EngineError::InvariantViolation)));
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum FinalizeEvent {
        EventPreflight,
        KvCommitValidation,
        SchedulerCommit,
        ModelCommit,
        ModelRollback,
        EventPublish,
    }

    struct RecordingState {
        events: Rc<RefCell<Vec<FinalizeEvent>>>,
        active: Rc<Cell<usize>>,
    }

    impl RecordingState {
        fn transaction(
            events: Rc<RefCell<Vec<FinalizeEvent>>>,
            active: Rc<Cell<usize>>,
        ) -> ModelTransaction<Self> {
            active.set(active.get() + 1);
            ModelTransaction::new(Vec::new(), Self { events, active })
        }

        fn finalized(&self, event: FinalizeEvent) {
            self.events.borrow_mut().push(event);
            self.active.set(self.active.get() - 1);
        }
    }

    impl ModelTransactionState for RecordingState {
        fn commit(&mut self) -> Result<(), ExecutorError> {
            self.finalized(FinalizeEvent::ModelCommit);
            Ok(())
        }

        fn rollback(&mut self) -> Result<(), ExecutorError> {
            self.finalized(FinalizeEvent::ModelRollback);
            Ok(())
        }
    }

    /// Emits one logits row per sampled item, and may keep the items.
    #[derive(Default)]
    struct Retaining {
        offload: bool,
        retain: bool,
        kept: Vec<ModelBatchItem>,
    }

    impl ModelExecutor for Retaining {
        fn kv_offload_info_v1(&self) -> Option<KvOffloadInfo> {
            self.offload.then_some(KvOffloadInfo {
                host_page_capacity: 16,
                minimum_tokens: 1,
                snapshot_pages: 1,
            })
        }
        type TransactionState<'a> = NoopModelTransactionState;

        fn execute<'a>(
            &'a mut self,
            items: &[ModelBatchItem],
            _kv_plan: &PreparedAppendBatch,
        ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
            if self.retain {
                self.kept.extend_from_slice(items);
            }
            Ok(ModelTransaction::no_op(
                items
                    .iter()
                    .filter(|item| item.expects_logits())
                    .map(|item| {
                        let mut logits = vec![0.0; 8];
                        logits[item.sampling().unwrap().history.len() % 6 + 2] = 4.0;
                        LogitsRow::new(item.request_id(), logits).unwrap()
                    })
                    .collect(),
            ))
        }
    }

    #[test]
    fn tbt_cold_and_prefill_only_caps_preserve_fake_backend_tokens() {
        let run = |policy| {
            let geometry = KvGeometry::new(4, 256).unwrap();
            let config = EngineConfig::new(
                SchedulerConfig::new(4, 8, 64, 32, 1)
                    .unwrap()
                    .with_prefill_budget(policy),
                geometry,
                ModelVocabulary::new(8).unwrap(),
                geometry.token_capacity(),
                8,
            )
            .unwrap();
            let mut engine = Engine::new(config, Retaining::default()).unwrap();
            for id in 1..=4 {
                engine
                    .submit(
                        EngineRequest::new(
                            RequestId::new(id),
                            vec![1; id as usize * 13],
                            12,
                            SamplingParameters::new(0.0).unwrap(),
                            1,
                            Vec::new(),
                            ModelVocabulary::new(8).unwrap(),
                        )
                        .unwrap(),
                    )
                    .unwrap();
            }
            let mut tokens = vec![Vec::new(); 4];
            let mut finished = 0;
            for step in 0..200 {
                let status = engine.step().unwrap();
                engine.audit().unwrap();
                for event in engine.drain_events() {
                    match event.kind {
                        EngineEventKind::Token { token_id, .. } => {
                            tokens[event.request_id.get() as usize - 1].push(token_id);
                        }
                        EngineEventKind::Finished { reason } => {
                            assert_eq!(reason, FinishReason::Length);
                            finished += 1;
                        }
                        other => panic!("unexpected event: {other:?}"),
                    }
                }
                if status == StepStatus::Idle {
                    break;
                }
                assert!(step < 199, "requests stopped progressing");
            }
            assert_eq!(finished, 4);
            assert!(tokens.iter().all(|tokens| tokens.len() == 12));
            assert_eq!(engine.kv_stats().active_sequences(), 0);
            tokens
        };
        let fixed = run(runtime_scheduler::PrefillBudget::Fixed);
        for steps in [0, 2] {
            assert_eq!(
                run(runtime_scheduler::PrefillBudget::Tbt(
                    runtime_scheduler::MixedPolicy::new(150, 4, steps).unwrap(),
                )),
                fixed
            );
        }
    }

    /// Emits, for every row, the token that follows a history of its
    /// length; its host drafter proposes the same tokens, so every draft is
    /// accepted. Records whether each batch prefilled and verified.
    #[derive(Default)]
    struct Successor {
        batches: Vec<(bool, bool)>,
    }

    fn successor(length: usize) -> u32 {
        (length % 6 + 2) as u32
    }

    impl ModelExecutor for Successor {
        fn draft_tokens_v1(
            &mut self,
            _id: RequestId,
            history: &[u32],
            maximum: usize,
        ) -> Result<Vec<u32>, ExecutorError> {
            Ok((0..maximum).map(|k| successor(history.len() + k)).collect())
        }

        type TransactionState<'a> = NoopModelTransactionState;

        fn execute<'a>(
            &'a mut self,
            items: &[ModelBatchItem],
            _kv_plan: &PreparedAppendBatch,
        ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
            let phase = |verify: bool| {
                items.iter().any(|item| {
                    matches!(item.phase(), BatchPhase::Verify { .. }) == verify
                        && (verify || matches!(item.phase(), BatchPhase::Prefill { .. }))
                })
            };
            self.batches.push((phase(false), phase(true)));
            let mut rows = Vec::new();
            for item in items {
                let Some(work) = item.sampling() else {
                    continue;
                };
                let drafts = match item.draft() {
                    Some(_) => item.input_tokens()[1..].to_vec(),
                    None => Vec::new(),
                };
                for row in 0..=drafts.len() {
                    let mut logits = vec![0.0; 8];
                    let length = work.for_verify_row(&drafts, row).history.len();
                    logits[successor(length) as usize] = 4.0;
                    rows.push(LogitsRow::new(item.request_id(), logits).unwrap());
                }
            }
            Ok(ModelTransaction::no_op(rows))
        }
    }

    /// Drafts a step accepted beside a prefill train no step time, so they
    /// must not inflate the realized throughput of the next timed step.
    #[test]
    fn drafts_accepted_beside_a_prefill_do_not_carry_into_the_next_step() {
        let vocabulary = ModelVocabulary::new(8).unwrap();
        let config = EngineConfig::new(
            SchedulerConfig::new(2, 4, 16, 16, 1).unwrap(),
            KvGeometry::new(4, 16).unwrap(),
            vocabulary,
            64,
            4,
        )
        .unwrap()
        .with_speculation(
            SpeculationConfig::new(2, 16, SpeculationMode::Fixed, DraftSource::Host).unwrap(),
        );
        let mut engine = Engine::new(config, Successor::default()).unwrap();
        let request = |id| {
            EngineRequest::new(
                RequestId::new(id),
                vec![1, 2, 3, 4],
                12,
                SamplingParameters::new(0.0).unwrap(),
                1,
                Vec::new(),
                vocabulary,
            )
            .unwrap()
        };
        engine.submit(request(1)).unwrap();
        // The first request prefills, then verifies alone.
        while !engine
            .executor
            .batches
            .last()
            .is_some_and(|&(_, verify)| verify)
        {
            engine.step().unwrap();
            let _ = engine.drain_events();
        }
        assert_eq!(engine.speculation.as_ref().unwrap().pending_accepted(), 0);
        let first = RequestId::new(1);
        let before = engine.requests[&first].history.len();
        engine.submit(request(2)).unwrap();
        engine.step().unwrap();
        let _ = engine.drain_events();
        // The second request's prefill shares a step with the first's
        // verification, which accepted both of its drafts.
        assert_eq!(engine.executor.batches.last(), Some(&(true, true)));
        assert_eq!(engine.requests[&first].history.len(), before + 3);
        assert_eq!(engine.speculation.as_ref().unwrap().pending_accepted(), 0);
    }

    #[test]
    fn startup_steps_are_observed_only_during_startup_warmup() {
        let config = EngineConfig::new(
            SchedulerConfig::new(2, 4, 16, 16, 1).unwrap(),
            KvGeometry::new(4, 16).unwrap(),
            ModelVocabulary::new(8).unwrap(),
            64,
            4,
        )
        .unwrap();
        let mut engine = Engine::new(config, Retaining::default()).unwrap();
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let log = std::sync::Arc::clone(&seen);
        engine.observe_startup_steps(Box::new(move |step| log.lock().unwrap().push(step)));
        let submit = |engine: &mut Engine<Retaining>, ids: std::ops::RangeInclusive<u64>| {
            for id in ids {
                engine
                    .submit(
                        EngineRequest::new(
                            RequestId::new(id),
                            vec![1, 2, 3, 4],
                            3,
                            SamplingParameters::new(0.0).unwrap(),
                            1,
                            Vec::new(),
                            ModelVocabulary::new(8).unwrap(),
                        )
                        .unwrap(),
                    )
                    .unwrap();
            }
        };
        let run = |engine: &mut Engine<Retaining>| {
            while engine.step().unwrap() != StepStatus::Idle {
                let _ = engine.drain_events();
            }
        };
        engine.begin_startup_warmup().unwrap();
        submit(&mut engine, 1..=2);
        run(&mut engine);
        engine.finish_startup_warmup().unwrap();
        let steps = seen.lock().unwrap().clone();
        // Both prompts prefill in the first step; the rest decode both rows
        // until each has its three tokens.
        assert_eq!(
            steps.first(),
            Some(&warmup::StartupStep {
                prefill_tokens: 8,
                ..warmup::StartupStep::default()
            })
        );
        assert!(steps.len() > 1);
        assert!(steps[1..].iter().all(|step| step.prefill_tokens == 0
            && step.decode_rows == 2
            && step.verify_items == 0
            && step.accepted_tokens == 0));
        // Served steps are not reported.
        submit(&mut engine, 3..=3);
        run(&mut engine);
        assert_eq!(*seen.lock().unwrap(), steps);
    }

    #[test]
    fn startup_warmup_restores_policy_tbt_metrics_and_prefix_state() {
        let geometry = KvGeometry::new(4, 16).unwrap();
        let config = EngineConfig::new(
            SchedulerConfig::new(2, 4, 16, 16, 1)
                .unwrap()
                .with_prefill_budget(runtime_scheduler::PrefillBudget::Tbt(
                    runtime_scheduler::MixedPolicy::new(100, 4, 0).unwrap(),
                )),
            geometry,
            ModelVocabulary::new(8).unwrap(),
            64,
            4,
        )
        .unwrap()
        .with_prefix_cache(1)
        .with_speculation(
            SpeculationConfig::new(2, 16, SpeculationMode::Adaptive, DraftSource::Bundle).unwrap(),
        );
        let mut engine = Engine::new(
            config,
            Retaining {
                offload: true,
                ..Retaining::default()
            },
        )
        .unwrap();
        let scheduler = engine.scheduler.clone();
        let policy = engine.speculation.clone();
        let metrics = engine.metrics();
        let before = metrics.snapshot();
        let kv = engine.kv_metrics();
        let kv_before = kv.snapshot();
        engine.begin_startup_warmup().unwrap();
        engine.step_trace = true;
        engine
            .metrics
            .compatibility(|metrics| metrics.step_trace_enabled = true);
        engine.audit().unwrap();
        for id in 1..=2 {
            engine
                .submit(
                    EngineRequest::new(
                        RequestId::new(id),
                        vec![1, 2, 3, 4],
                        6,
                        SamplingParameters::new(0.0).unwrap(),
                        1,
                        Vec::new(),
                        ModelVocabulary::new(8).unwrap(),
                    )
                    .unwrap()
                    .with_pinned_draft_tokens(0),
                )
                .unwrap();
        }
        while engine.step().unwrap() != StepStatus::Idle {
            let _ = engine.drain_events();
        }
        assert_ne!(engine.scheduler, scheduler);
        assert_ne!(
            engine.scheduler.mixed_estimates(),
            scheduler.mixed_estimates()
        );
        assert_ne!(engine.speculation, policy);
        assert_ne!(engine.metrics.snapshot(), before);
        assert_eq!(engine.kv_stats().anchors(), 0);
        assert!(engine.host_clock.clone().executed().gap.is_some());
        engine.finish_startup_warmup().unwrap();
        let timing = engine.host_clock.clone().executed();
        assert_eq!(timing.gap, None);
        assert_eq!(timing.period, None);
        assert_eq!(timing.outside, None);
        engine.audit().unwrap();
        assert_eq!(engine.scheduler, scheduler);
        assert_eq!(engine.speculation, policy);
        assert_eq!(metrics.snapshot(), before);
        assert_eq!(kv.snapshot(), kv_before);
        assert_eq!(engine.kv_stats().anchors(), 0);
    }

    #[test]
    fn sampling_shares_the_history_the_engine_appends_to_without_copying() {
        for retain in [false, true] {
            let geometry = KvGeometry::new(4, 16).unwrap();
            let config = EngineConfig::new(
                SchedulerConfig::new(1, 1, 16, 16, 1).unwrap(),
                geometry,
                ModelVocabulary::new(8).unwrap(),
                geometry.token_capacity(),
                1,
            )
            .unwrap();
            let executor = Retaining {
                retain,
                ..Retaining::default()
            };
            let mut engine = Engine::new(config, executor).unwrap();
            let id = RequestId::new(1);
            let request = EngineRequest::new(
                id,
                vec![1, 2, 3],
                6,
                SamplingParameters::new(0.0).unwrap(),
                1,
                Vec::new(),
                ModelVocabulary::new(8).unwrap(),
            )
            .unwrap();
            engine.submit(request).unwrap();
            let reserved = engine.requests[&id].history.capacity();
            let buffer = engine.requests[&id].history.as_ptr();
            let mut expected = vec![1, 2, 3];
            for _ in 0..5 {
                assert_eq!(engine.step(), Ok(StepStatus::Progressed));
                expected.push((expected.len() % 6 + 2) as u32);
                let history = &engine.requests[&id].history;
                assert_eq!(**history, expected);
                // Kept rows force a copy, which keeps the reserved capacity
                // the next commit's preflight checks.
                assert_eq!(history.capacity(), reserved);
                assert_eq!(Arc::strong_count(history), 1);
                if !retain {
                    assert_eq!(history.as_ptr(), buffer, "appended in place");
                }
            }
            let kept = &engine.executor().kept;
            assert_eq!(kept.len(), if retain { 5 } else { 0 });
            // A kept row still reads the history of its own step.
            for (step, item) in kept.iter().enumerate() {
                let history = &item.sampling().unwrap().history;
                assert_eq!(history.to_vec(), expected[..3 + step]);
            }
        }
    }

    #[test]
    fn event_failure_does_not_suppress_request_retirement() {
        for emit_rows in [false, true] {
            let geometry = KvGeometry::new(4, 4).unwrap();
            let config = EngineConfig::new(
                SchedulerConfig::new(1, 1, 4, 4, 1).unwrap(),
                geometry,
                ModelVocabulary::new(8).unwrap(),
                geometry.token_capacity(),
                1,
            )
            .unwrap();
            let metrics = EngineMetrics::default();
            let mut engine = Engine::new_with_observability(
                config,
                AlwaysFails {
                    emit_rows,
                    ..AlwaysFails::default()
                },
                Arc::new(SystemMonotonicClock::new()),
                metrics.clone(),
            )
            .unwrap();
            engine
                .submit(
                    EngineRequest::new(
                        RequestId::new(1),
                        vec![1],
                        1,
                        SamplingParameters::new(0.0).unwrap(),
                        1,
                        Vec::new(),
                        ModelVocabulary::new(8).unwrap(),
                    )
                    .unwrap(),
                )
                .unwrap();
            engine.next_event_sequence = u64::MAX;

            assert_eq!(engine.step(), Err(EngineError::EventSequenceExhausted));
            assert_eq!(engine.executor().retired, vec![RequestId::new(1)]);
            assert_eq!(metrics.snapshot().errors, 1);
            assert_eq!(metrics.snapshot().requests_completed, 0);
            assert_eq!(metrics.snapshot().requests_terminal, 1);
            assert_eq!(metrics.snapshot().requests_active, 0);
            assert_eq!(engine.terminal_count(), 0);
            engine.audit().unwrap();
        }
    }

    #[test]
    fn cancellation_notifies_before_event_sequence_failure() {
        let geometry = KvGeometry::new(4, 4).unwrap();
        let config = EngineConfig::new(
            SchedulerConfig::new(1, 1, 4, 4, 1).unwrap(),
            geometry,
            ModelVocabulary::new(8).unwrap(),
            geometry.token_capacity(),
            1,
        )
        .unwrap();
        let mut engine = Engine::new(config, AlwaysFails::default()).unwrap();
        let id = RequestId::new(1);
        engine
            .submit(
                EngineRequest::new(
                    id,
                    vec![1],
                    1,
                    SamplingParameters::new(0.0).unwrap(),
                    1,
                    Vec::new(),
                    ModelVocabulary::new(8).unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        engine.next_event_sequence = u64::MAX;
        assert_eq!(engine.cancel(id), Err(EngineError::EventSequenceExhausted));
        assert_eq!(engine.executor().retired, vec![id]);
        assert_eq!(engine.terminal_count(), 1);
    }

    #[test]
    fn scheduler_preflight_failure_rolls_model_back_once() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let active = Rc::new(Cell::new(0));
        let transaction = RecordingState::transaction(Rc::clone(&events), Rc::clone(&active));

        let result = Err::<(), _>(ScheduleError::NoPendingBatch);
        drop(transaction);

        assert_eq!(result, Err(ScheduleError::NoPendingBatch));
        assert_eq!(*events.borrow(), [FinalizeEvent::ModelRollback]);
        assert_eq!(active.get(), 0);
    }

    #[test]
    fn scheduler_commit_failure_records_error_without_a_terminal_request() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let active = Rc::new(Cell::new(0));
        let transaction = RecordingState::transaction(Rc::clone(&events), Rc::clone(&active));
        let metrics = EngineMetrics::default();

        let result = commit_scheduler_then_model_observed(
            transaction,
            None,
            || Err::<(), _>(ScheduleError::PreparedCommitMismatch),
            &metrics,
        );

        assert_eq!(result, Err(ScheduleError::PreparedCommitMismatch));
        assert_eq!(metrics.snapshot().errors, 1);
        assert_eq!(metrics.snapshot().requests_completed, 0);
        assert_eq!(metrics.snapshot().requests_terminal, 0);
        assert_eq!(*events.borrow(), [FinalizeEvent::ModelRollback]);
        assert_eq!(active.get(), 0);
    }

    #[test]
    fn success_order_is_event_preflight_scheduler_model_then_publish() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let active = Rc::new(Cell::new(0));
        events.borrow_mut().push(FinalizeEvent::EventPreflight);
        let transaction = RecordingState::transaction(Rc::clone(&events), Rc::clone(&active));
        let scheduler_events = Rc::clone(&events);

        commit_scheduler_then_model(transaction, None, || {
            scheduler_events
                .borrow_mut()
                .push(FinalizeEvent::SchedulerCommit);
            Ok(())
        })
        .unwrap()
        .1
        .unwrap();
        events.borrow_mut().push(FinalizeEvent::EventPublish);

        assert_eq!(
            *events.borrow(),
            [
                FinalizeEvent::EventPreflight,
                FinalizeEvent::SchedulerCommit,
                FinalizeEvent::ModelCommit,
                FinalizeEvent::EventPublish,
            ]
        );
        assert_eq!(active.get(), 0);
    }

    #[test]
    fn kv_commit_validation_failure_rolls_model_back_before_commit() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let active = Rc::new(Cell::new(0));
        let transaction = RecordingState::transaction(Rc::clone(&events), Rc::clone(&active));
        let validation_events = Rc::clone(&events);

        let result = commit_scheduler_then_model(transaction, None, || {
            validation_events
                .borrow_mut()
                .push(FinalizeEvent::KvCommitValidation);
            Err::<(), _>(ScheduleError::Kv(
                runtime_kv::KvError::PreparedTransactionMismatch,
            ))
        });

        assert_eq!(
            result,
            Err(ScheduleError::Kv(
                runtime_kv::KvError::PreparedTransactionMismatch
            ))
        );
        assert_eq!(
            *events.borrow(),
            [
                FinalizeEvent::KvCommitValidation,
                FinalizeEvent::ModelRollback
            ]
        );
        assert_eq!(active.get(), 0);
    }

    #[test]
    fn event_preflight_reserves_capacity_and_constructs_top_logprobs() {
        let geometry = KvGeometry::new(4, 4).unwrap();
        let mut scheduler =
            Scheduler::new(SchedulerConfig::new(1, 1, 4, 4, 1).unwrap(), geometry).unwrap();
        let request_id = RequestId::new(7);
        scheduler
            .submit(RequestSpec::new(request_id, 1, 1).unwrap())
            .unwrap();
        let batch = scheduler.next_batch().unwrap().unwrap();
        let items = vec![ModelBatchItem {
            prefill_boundary: None,
            output_cycle: None,
            request_id,
            phase: BatchPhase::Prefill { tokens: 1 },
            input_tokens: vec![1],
            expects_logits: true,
            prompt_targets: Vec::new(),
            prompt_top_logprobs: 0,
            sampling: None,
            draft: None,
            media_encodes: Vec::new(),
            media_injects: Vec::new(),
            position_axes: Vec::new(),
            anchor: false,
        }];
        let mut history = Vec::with_capacity(2);
        history.push(1);
        let mut requests = BTreeMap::new();
        requests.insert(
            request_id,
            RequestRecord {
                prompt_tokens: vec![1],
                prompt_offset: 0,
                prompt_logprobs: None,
                pending_decode_token: None,
                sampling: SamplingParameters::new(1.0)
                    .unwrap()
                    .with_top_logprobs(2)
                    .unwrap(),
                sampling_state: SamplingState::new(9),
                history: Arc::new(history),
                emitted_tokens: 0,
                maximum_new_tokens: 1,
                stop_ids: Vec::new(),
                termination_stop_ids: Vec::new(),
                submitted_micros: 0,
                last_token_micros: None,
                acceptance: None,
                verified_last: false,
                media: Vec::new(),
                spans: Vec::new(),
                positions: None,
                media_wanted: Vec::new(),
                media_placed: Vec::new(),
                media_ends: Vec::new(),
                constraint: None,
                cached_tokens: 0,
                cache_salt: None,
                pinned_draft_tokens: None,
                scheduled: false,
                preemptions: 0,
                resuming: false,
            },
        );
        let mut logits = vec![-1.0; 8];
        logits[3] = 2.0;
        let rows = vec![LogitsRow::new(request_id, logits).unwrap()];
        let staged = stage_samples(
            &requests,
            &items,
            &rows,
            &[],
            &[],
            &[],
            ModelVocabulary::new(8).unwrap(),
            &[],
        )
        .unwrap();
        let mut events = VecDeque::new();

        let prepared =
            preflight_apply(&requests, &mut events, 0, &batch, &items, &[], staged).unwrap();

        assert!(events.capacity() >= prepared.events.len());
        assert!(events.is_empty());
        assert_eq!(prepared.events.len(), 2);
        assert!(matches!(
            prepared.events.front().map(EngineEvent::kind),
            Some(EngineEventKind::Token { top_logprobs, .. }) if top_logprobs.len() == 2
        ));
        assert_eq!(prepared.next_event_sequence, 2);

        let record = &requests[&request_id];
        let sampled_item =
            ModelBatchItem::new(request_id, BatchPhase::Prefill { tokens: 1 }, vec![1], true)
                .with_sampling(SamplingWork {
                    parameters: record.sampling.clone(),
                    state: record.sampling_state,
                    history: SamplingHistory::new(Arc::clone(&record.history)),
                    emitted_tokens: 0,
                    stop_ids: Vec::new(),
                    constraint: None,
                });
        let sampled = SampledRow::new(
            request_id,
            3,
            -0.2,
            vec![TokenLogprob::new(3, -0.2), TokenLogprob::new(0, -3.2)],
        );
        let vocabulary = ModelVocabulary::new(8).unwrap();
        assert!(outputs_match(
            std::slice::from_ref(&sampled_item),
            &[],
            &[],
            std::slice::from_ref(&sampled),
            &[],
            &[],
            vocabulary
        ));
        assert!(!outputs_match(
            std::slice::from_ref(&sampled_item),
            &rows,
            &[],
            std::slice::from_ref(&sampled),
            &[],
            &[],
            vocabulary
        ));
        let staged = stage_samples(
            &requests,
            &[sampled_item],
            &[],
            &[sampled],
            &[],
            &[],
            vocabulary,
            &[],
        )
        .unwrap();
        let selected = staged[0].as_ref().unwrap();
        assert_eq!(selected.tokens[0].token_id, 3);
        assert_eq!(selected.state.generated(), 1);
    }

    #[test]
    fn panic_drops_a_pending_transaction_and_rolls_back_once() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let active = Rc::new(Cell::new(0));
        let transaction_events = Rc::clone(&events);
        let transaction_active = Rc::clone(&active);

        let result = catch_unwind(AssertUnwindSafe(move || {
            let _transaction = RecordingState::transaction(transaction_events, transaction_active);
            panic!("injected panic before commit");
        }));

        assert!(result.is_err());
        assert_eq!(*events.borrow(), [FinalizeEvent::ModelRollback]);
        assert_eq!(active.get(), 0);
    }

    #[test]
    fn explicit_rollback_does_not_finalize_again_on_drop() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let active = Rc::new(Cell::new(0));
        RecordingState::transaction(Rc::clone(&events), Rc::clone(&active))
            .rollback()
            .unwrap();

        assert_eq!(*events.borrow(), [FinalizeEvent::ModelRollback]);
        assert_eq!(active.get(), 0);
    }
}
