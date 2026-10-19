#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use runtime_kv::{
    BlockId, KvError, KvGeometry, KvMetrics, KvPool, KvStats, PrefixCacheKeyV1, PrefixChainV1,
    PreparedAppendBatch, PreparedAppendId, PreparedKvCommit, SequenceHandle,
};

mod offload;
pub use offload::{OffloadConfig, ResumePlan};

const MAX_TRACKED_REQUESTS: usize = 1 << 20;
/// Steps a started prompt may go without a chunk on the uncapped prefill
/// path before it sorts ahead of requests that never ran.
const STARVED_PREFILL_TICKS: u64 = 4;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RequestId(u64);

impl RequestId {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestSpec {
    id: RequestId,
    prompt_tokens: usize,
    maximum_new_tokens: usize,
    maximum_kv_tokens: usize,
    media_pages: usize,
    reserved_admission: bool,
    prompt_scoring: bool,
}

impl RequestSpec {
    pub fn new(
        id: RequestId,
        prompt_tokens: usize,
        maximum_new_tokens: usize,
    ) -> Result<Self, InvalidRequest> {
        if prompt_tokens == 0 {
            return Err(InvalidRequest::ZeroPrompt);
        }
        if maximum_new_tokens == 0 {
            return Err(InvalidRequest::ZeroMaximumNewTokens);
        }
        let maximum_kv_tokens = prompt_tokens
            .checked_add(maximum_new_tokens - 1)
            .ok_or(InvalidRequest::LengthOverflow)?;
        Ok(Self {
            id,
            prompt_tokens,
            maximum_new_tokens,
            maximum_kv_tokens,
            media_pages: 0,
            reserved_admission: false,
            prompt_scoring: false,
        })
    }

    /// Scores every prompt row. Such requests cannot replay published rows.
    #[must_use]
    pub const fn with_prompt_scoring(mut self) -> Self {
        self.prompt_scoring = true;
        self.reserved_admission = true;
        self
    }

    /// Reserves the request's worst case at admission and never preempts it,
    /// whatever the scheduler's admission policy. For requests whose
    /// recompute would repeat work already reported, such as prompt rows.
    #[must_use]
    pub const fn with_reserved_admission(mut self) -> Self {
        self.reserved_admission = true;
        self
    }

    #[must_use]
    pub const fn reserved_admission(self) -> bool {
        self.reserved_admission
    }

    /// The most media arena pages one prefill step of the request can cover:
    /// the pages of the media items overlapping its smallest chunk. Images
    /// stream through the arena chunk by chunk, so this, not the request's
    /// total, must fit the arena. Started media prefills hold up to these
    /// pages each while media lie ahead of them: another starts only when
    /// its own fit beside theirs. A request with media is never preempted.
    #[must_use]
    pub const fn with_media_pages(mut self, pages: usize) -> Self {
        self.media_pages = pages;
        self
    }

    #[must_use]
    pub const fn media_pages(self) -> usize {
        self.media_pages
    }

    #[must_use]
    pub const fn id(self) -> RequestId {
        self.id
    }

    #[must_use]
    pub const fn prompt_tokens(self) -> usize {
        self.prompt_tokens
    }

    #[must_use]
    pub const fn maximum_new_tokens(self) -> usize {
        self.maximum_new_tokens
    }

    #[must_use]
    pub const fn maximum_kv_tokens(self) -> usize {
        self.maximum_kv_tokens
    }
}

const DEFAULT_CAPACITY_PROFILE_NAME: &str = "default-unmeasured";

/// A named bound on total nonterminal work accepted by one scheduler.
///
/// The bound is an admission policy. It carries no throughput, latency, or
/// device-memory measurement and makes no qualification claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchedulerCapacityProfile {
    name: &'static str,
    maximum_in_flight: usize,
}

impl SchedulerCapacityProfile {
    pub const fn new(name: &'static str, maximum_in_flight: usize) -> Result<Self, ConfigError> {
        if name.is_empty() {
            return Err(ConfigError::EmptyCapacityProfileName);
        }
        if maximum_in_flight == 0 {
            return Err(ConfigError::ZeroProfileCapacity);
        }
        Ok(Self {
            name,
            maximum_in_flight,
        })
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        self.name
    }

    #[must_use]
    pub const fn maximum_in_flight(self) -> usize {
        self.maximum_in_flight
    }
}

/// How admission reserves KV pages for a request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionPolicy {
    /// Every request reserves its worst case, prompt plus maximum output, at
    /// admission and runs to its end without preemption.
    Reserved,
    /// Requests reserve what their next step needs. When the pool runs short,
    /// the latest arrivals are preempted and later recompute their tokens.
    Optimistic(OptimisticAdmission),
}

/// The limits of optimistic admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OptimisticAdmission {
    watermark_millionths: u32,
    maximum_preemptions: usize,
    forced_preemption_interval: Option<usize>,
}

impl OptimisticAdmission {
    /// A 1% watermark and at most three preemptions per request.
    pub const DEFAULT: Self = Self {
        watermark_millionths: 10_000,
        maximum_preemptions: 3,
        forced_preemption_interval: None,
    };

    /// `watermark_millionths` of the pool's pages stay free of reservations
    /// when a request is admitted while another is live. A request preempted
    /// `maximum_preemptions` times is readmitted with its worst case reserved
    /// and is not preempted again, so no request starves.
    pub const fn new(
        watermark_millionths: u32,
        maximum_preemptions: usize,
    ) -> Result<Self, ConfigError> {
        if watermark_millionths >= 1_000_000 {
            return Err(ConfigError::InvalidAdmissionWatermark);
        }
        Ok(Self {
            watermark_millionths,
            maximum_preemptions,
            forced_preemption_interval: None,
        })
    }

    /// Qualification only: also preempts every preemptible decoding request
    /// each time it has emitted another `every` tokens.
    pub const fn with_forced_preemption(mut self, every: usize) -> Result<Self, ConfigError> {
        if every == 0 {
            return Err(ConfigError::ZeroForcedPreemptionInterval);
        }
        self.forced_preemption_interval = Some(every);
        Ok(self)
    }

    #[must_use]
    pub const fn watermark_millionths(self) -> u32 {
        self.watermark_millionths
    }

    #[must_use]
    pub const fn maximum_preemptions(self) -> usize {
        self.maximum_preemptions
    }

    #[must_use]
    pub const fn forced_preemption_interval(self) -> Option<usize> {
        self.forced_preemption_interval
    }
}

/// How prefill shares a step with decoding requests.
/// With a declared executor capacity, a lone active unscored prefill may
/// use the whole step token budget.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PrefillBudget {
    /// Configured chunks for competing requests and the whole step token budget.
    #[default]
    Fixed,
    /// While requests decode, cap the step's prefill tokens by measured
    /// execution times so the estimated mixed step meets the target.
    DecodeAware(PrefillTarget),
    /// While requests decode, cap the step's prefill tokens so a fitted
    /// model of step time (decode, fixed, per-token and attention-context
    /// terms) meets a decode stall target. Use the progress floor until
    /// trained. Competing prefills keep the configured chunks and budget.
    Tbt(MixedPolicy),
}

/// The default `PrefillBudget::Tbt` target: an estimated mixed step of 100 ms.
pub const DEFAULT_DECODE_TBT_TARGET_MS: u32 = 100;
/// The default `PrefillBudget::Tbt` progress floor, in prefill tokens.
pub const DEFAULT_MIXED_MIN_PREFILL_TOKENS: usize = 512;
/// The most decode-only steps `PrefillBudget::Tbt` may put between chunks.
pub const MAXIMUM_MIXED_DECODE_STEPS: u32 = 8;

/// The decode stall bound of `PrefillBudget::Tbt`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MixedPolicy {
    target_millis: u32,
    min_prefill_tokens: usize,
    decode_steps: u32,
}

impl MixedPolicy {
    /// `target_millis` bounds the estimated time of a step that prefills
    /// while requests decode; a step always takes at least
    /// `min_prefill_tokens` (or the prompt tail) so prefill progresses; and
    /// `decode_steps` decode-only steps follow each such step.
    pub const fn new(
        target_millis: u32,
        min_prefill_tokens: usize,
        decode_steps: u32,
    ) -> Result<Self, ConfigError> {
        if target_millis == 0
            || min_prefill_tokens == 0
            || decode_steps > MAXIMUM_MIXED_DECODE_STEPS
        {
            return Err(ConfigError::InvalidMixedPolicy);
        }
        Ok(Self {
            target_millis,
            min_prefill_tokens,
            decode_steps,
        })
    }

    #[must_use]
    pub const fn target_millis(self) -> u32 {
        self.target_millis
    }

    #[must_use]
    pub const fn min_prefill_tokens(self) -> usize {
        self.min_prefill_tokens
    }

    #[must_use]
    pub const fn decode_steps(self) -> u32 {
        self.decode_steps
    }
}

impl Default for MixedPolicy {
    fn default() -> Self {
        Self {
            target_millis: DEFAULT_DECODE_TBT_TARGET_MS,
            min_prefill_tokens: DEFAULT_MIXED_MIN_PREFILL_TOKENS,
            decode_steps: 0,
        }
    }
}

/// The default decode-aware target: an estimated mixed step of 200 ms.
pub const DEFAULT_PREFILL_ITL_TARGET_MS: u32 = 200;

/// What a decode-aware mixed step's estimated execution time must fit.
///
/// Both forms are approximate: the cap never falls below 512 tokens and
/// rounds down to 512-token multiples.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefillTarget {
    /// The whole mixed step, decode rows included, takes at most this many
    /// milliseconds, a per-token latency bound for the decoding requests.
    ItlMillis(u32),
    /// The prefill part takes at most this many decode steps. `DecodeSteps(1)`
    /// is the first decode-aware policy.
    DecodeSteps(u32),
}

impl Default for PrefillTarget {
    fn default() -> Self {
        Self::ItlMillis(DEFAULT_PREFILL_ITL_TARGET_MS)
    }
}

mod mixed_budget;
use mixed_budget::StepModel;
pub use mixed_budget::{MixedEstimates, StepShape};
mod prefill_budget;
pub use prefill_budget::StepTimeEstimates;
use prefill_budget::StepTimes;

/// The logits-row accounting of one batch's prefill items.
struct PrefillRows<'a> {
    /// The scored row capacity, which also bounds a scored request's chunk.
    maximum_scored_rows: usize,
    /// Chunk ends short of the prefill target fall on its multiples.
    granule: usize,
    /// Logits rows left in this batch: every scored row and every final row.
    scored_budget: &'a mut usize,
    /// The prefill request whose tokens and rows other items leave free
    /// until it is visited.
    reserve: &'a mut Option<Reservation>,
    /// Media arena pages one step's chunks may cover together.
    media_arena: usize,
    /// Media arena pages left in this batch.
    media_budget: &'a mut usize,
}

/// A prefill item's claim on the batch, held for it until it is visited.
#[derive(Clone, Copy, Debug)]
struct Reservation {
    index: usize,
    tokens: usize,
    rows: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchedulerConfig {
    maximum_sequences: usize,
    tracking_capacity: usize,
    scheduled_token_budget: usize,
    prefill_chunk_limit: usize,
    guaranteed_prefill_chunks: Option<usize>,
    lone_prefill_expansion: bool,
    maximum_scored_rows: usize,
    decode_cost: usize,
    capacity_profile: SchedulerCapacityProfile,
    media_pages: usize,
    prefix_state_pages: Option<usize>,
    admission: AdmissionPolicy,
    prefill_budget: PrefillBudget,
    prefill_granule: usize,
    rewrites_last_page: bool,
    anchor_items: usize,
}

impl SchedulerConfig {
    pub fn new(
        maximum_sequences: usize,
        tracking_capacity: usize,
        scheduled_token_budget: usize,
        prefill_chunk_limit: usize,
        decode_cost: usize,
    ) -> Result<Self, ConfigError> {
        if maximum_sequences == 0 {
            return Err(ConfigError::ZeroMaximumSequences);
        }
        if tracking_capacity < maximum_sequences {
            return Err(ConfigError::TrackingBelowSequenceCapacity);
        }
        if tracking_capacity > MAX_TRACKED_REQUESTS {
            return Err(ConfigError::TrackingCapacityTooLarge);
        }
        if scheduled_token_budget == 0 {
            return Err(ConfigError::ZeroScheduledTokenBudget);
        }
        if prefill_chunk_limit == 0 {
            return Err(ConfigError::ZeroPrefillChunkLimit);
        }
        if prefill_chunk_limit > scheduled_token_budget {
            return Err(ConfigError::PrefillChunkLimitAboveTokenBudget);
        }
        if decode_cost == 0 || decode_cost > scheduled_token_budget {
            return Err(ConfigError::InvalidDecodeCost);
        }
        Ok(Self {
            maximum_sequences,
            tracking_capacity,
            scheduled_token_budget,
            prefill_chunk_limit,
            guaranteed_prefill_chunks: None,
            lone_prefill_expansion: false,
            maximum_scored_rows: usize::MAX,
            decode_cost,
            capacity_profile: SchedulerCapacityProfile {
                name: DEFAULT_CAPACITY_PROFILE_NAME,
                maximum_in_flight: tracking_capacity,
            },
            media_pages: 0,
            prefix_state_pages: None,
            admission: AdmissionPolicy::Reserved,
            prefill_budget: PrefillBudget::Fixed,
            prefill_granule: 1,
            rewrites_last_page: false,
            anchor_items: 0,
        })
    }

    /// Allows a lone unscored prefill to use the step budget instead of the
    /// configured chunk. Disabled by default. Enable only when the executor
    /// declares a prefill capacity that bounds the step budget.
    #[must_use]
    pub const fn with_lone_prefill_expansion(mut self, enabled: bool) -> Self {
        self.lone_prefill_expansion = enabled;
        self
    }

    /// Opt unscored text prefills into a fixed absolute chunk grid. Every
    /// grid boundary and prompt end is complete state and may be captured.
    /// Shared prefixes cannot introduce additional prefill boundaries.
    /// Scored and media requests keep their ordinary complete-row path.
    pub const fn with_guaranteed_prefill_chunks(
        mut self,
        tokens: usize,
    ) -> Result<Self, ConfigError> {
        if tokens == 0 {
            return Err(ConfigError::ZeroPrefillChunkLimit);
        }
        if tokens > self.scheduled_token_budget {
            return Err(ConfigError::PrefillChunkLimitAboveTokenBudget);
        }
        if !tokens.is_multiple_of(self.prefill_granule) {
            return Err(ConfigError::PrefillChunkLimitOffGranule);
        }
        self.guaranteed_prefill_chunks = Some(tokens);
        self.prefill_chunk_limit = tokens;
        self.lone_prefill_expansion = false;
        Ok(self)
    }

    #[must_use]
    pub const fn guaranteed_prefill_chunks(self) -> Option<usize> {
        self.guaranteed_prefill_chunks
    }

    /// Positions a chunk may end at before the prompt end: multiples of
    /// `tokens`. Public prefix anchors, and so prefix hits, sit on them too,
    /// so a model whose prefill arithmetic follows fixed position blocks (a
    /// chunked recurrence) computes the same rows whatever the chunking or
    /// the anchor. A preemption's private anchor taken while its request
    /// decodes may sit anywhere; it resumes that decode exactly. One, the
    /// default, allows any position. A host sets the
    /// value its bundle manifest declares; the chunk limit and a bounded
    /// scored-row limit must be multiples of it.
    pub const fn with_prefill_granule(mut self, tokens: usize) -> Result<Self, ConfigError> {
        if tokens == 0 || !self.prefill_chunk_limit.is_multiple_of(tokens) {
            return Err(ConfigError::PrefillChunkLimitOffGranule);
        }
        if !Self::scored_rows_on_granule(self.maximum_scored_rows, tokens) {
            return Err(ConfigError::ScoredRowsOffGranule);
        }
        self.prefill_granule = tokens;
        Ok(self)
    }

    /// Unbounded, or a multiple of the granule, so a scored chunk ends on it.
    const fn scored_rows_on_granule(rows: usize, granule: usize) -> bool {
        rows == usize::MAX || rows.is_multiple_of(granule)
    }

    #[must_use]
    pub const fn prefill_granule(self) -> usize {
        self.prefill_granule
    }

    /// Declares that a request's next step may rewrite committed rows of its
    /// last page (drafter rows that need the following tokens). A prefix hit
    /// then copies the page holding the anchor's last row even when it is
    /// full, and reserves that copy like the copy of a shared partial tail.
    /// A host sets the value its bundle manifest declares.
    #[must_use]
    pub const fn with_last_page_rewrites(mut self, rewrites: bool) -> Self {
        self.rewrites_last_page = rewrites;
        self
    }

    #[must_use]
    pub const fn rewrites_last_page(self) -> bool {
        self.rewrites_last_page
    }

    /// Prompt-ending prefill items per step whose model state the bundle
    /// keeps at the last multiple of the prefill granule inside the item.
    /// Up to this many requests a step prefill past their prompt-end anchor
    /// (the last granule multiple before an unaligned prompt end) in one
    /// chunk and capture it after the step ([`BatchItem::anchor_tokens`]);
    /// the others end a chunk there. Zero, the default, ends every such
    /// chunk. A host sets the value its bundle manifest declares.
    #[must_use]
    pub const fn with_anchor_items(mut self, items: usize) -> Self {
        self.anchor_items = items;
        self
    }

    #[must_use]
    pub const fn anchor_items(self) -> usize {
        self.anchor_items
    }

    /// Selects the mixed-step prefill policy. Fixed preserves competing chunks.
    #[must_use]
    pub const fn with_prefill_budget(mut self, policy: PrefillBudget) -> Self {
        self.prefill_budget = policy;
        self
    }

    pub const fn prefill_budget(self) -> PrefillBudget {
        self.prefill_budget
    }

    /// Bounds the logits rows of a prefill phase that scores prompt rows.
    /// A scored request's chunk never exceeds it. Unbounded by default; a
    /// host sets the value its bundle manifest declares.
    pub fn with_maximum_scored_rows(mut self, rows: usize) -> Result<Self, ConfigError> {
        if rows == 0 {
            return Err(ConfigError::ZeroMaximumScoredRows);
        }
        if !Self::scored_rows_on_granule(rows, self.prefill_granule) {
            return Err(ConfigError::ScoredRowsOffGranule);
        }
        self.maximum_scored_rows = rows;
        Ok(self)
    }

    #[must_use]
    pub const fn maximum_scored_rows(self) -> usize {
        self.maximum_scored_rows
    }

    /// Selects how admission reserves pages. The default is
    /// [`AdmissionPolicy::Reserved`].
    #[must_use]
    pub const fn with_admission(mut self, admission: AdmissionPolicy) -> Self {
        self.admission = admission;
        self
    }

    #[must_use]
    pub const fn admission(self) -> AdmissionPolicy {
        self.admission
    }

    /// Enables prefix caching with a bundle snapshot of `state_pages` KV
    /// pages per anchor. Without it, nothing is looked up or captured.
    #[must_use]
    pub const fn with_prefix_cache(mut self, state_pages: usize) -> Self {
        self.prefix_state_pages = Some(state_pages);
        self
    }

    #[must_use]
    pub const fn prefix_state_pages(self) -> Option<usize> {
        self.prefix_state_pages
    }

    /// Media arena pages the prefill chunks of one step may cover together.
    /// Zero, the default, admits no request that needs media pages.
    #[must_use]
    pub const fn with_media_pages(mut self, pages: usize) -> Self {
        self.media_pages = pages;
        self
    }

    #[must_use]
    pub const fn media_pages(self) -> usize {
        self.media_pages
    }

    pub const fn with_capacity_profile(
        mut self,
        profile: SchedulerCapacityProfile,
    ) -> Result<Self, ConfigError> {
        if profile.maximum_in_flight < self.maximum_sequences {
            return Err(ConfigError::ProfileBelowSequenceCapacity);
        }
        if profile.maximum_in_flight > self.tracking_capacity {
            return Err(ConfigError::ProfileAboveTrackingCapacity);
        }
        self.capacity_profile = profile;
        Ok(self)
    }

    #[must_use]
    pub const fn maximum_sequences(self) -> usize {
        self.maximum_sequences
    }

    #[must_use]
    pub const fn tracking_capacity(self) -> usize {
        self.tracking_capacity
    }

    #[must_use]
    pub const fn scheduled_token_budget(self) -> usize {
        self.scheduled_token_budget
    }

    #[must_use]
    pub const fn prefill_chunk_limit(self) -> usize {
        self.prefill_chunk_limit
    }

    #[must_use]
    pub const fn decode_cost(self) -> usize {
        self.decode_cost
    }

    #[must_use]
    pub const fn capacity_profile(self) -> SchedulerCapacityProfile {
        self.capacity_profile
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestState {
    Queued,
    Admitted,
    Prefilling,
    Decoding,
    Finished,
    Cancelled,
    Failed,
}

impl RequestState {
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Finished | Self::Cancelled | Self::Failed)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchPhase {
    Prefill {
        tokens: usize,
    },
    Decode,
    /// A decode step that also verifies `draft_tokens` drafted tokens.
    ///
    /// The item appends `draft_tokens + 1` KV tokens and emits between one
    /// and `draft_tokens + 1` tokens. `draft_tokens` is at least one; a step
    /// without drafts is `Decode`.
    Verify {
        draft_tokens: usize,
    },
}

impl BatchPhase {
    /// KV tokens this phase appends when it is prepared.
    #[must_use]
    pub const fn appended_tokens(self) -> usize {
        match self {
            Self::Prefill { tokens } => tokens,
            Self::Decode => 1,
            Self::Verify { draft_tokens } => draft_tokens + 1,
        }
    }
}

/// What the media of one prefill chunk need: the token-equivalent cost of
/// the encodes it would start, and the media arena pages of every media item
/// it covers, resident or not, since the step pins each of them.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ChunkMedia {
    pub encode_cost: usize,
    pub pages: usize,
}

/// Bounds on the drafts one batch may verify.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DraftLimits {
    maximum_draft_tokens: usize,
    maximum_verify_tokens: usize,
}

impl DraftLimits {
    /// No drafts: every decode item stays a one-token step.
    pub const NONE: Self = Self {
        maximum_draft_tokens: 0,
        maximum_verify_tokens: 0,
    };

    /// `maximum_verify_tokens` bounds the decode-phase rows of one batch: one
    /// per plain decode item and `draft_tokens + 1` per verify item.
    #[must_use]
    pub const fn new(maximum_draft_tokens: usize, maximum_verify_tokens: usize) -> Self {
        Self {
            maximum_draft_tokens,
            maximum_verify_tokens,
        }
    }

    #[must_use]
    pub const fn maximum_draft_tokens(self) -> usize {
        self.maximum_draft_tokens
    }

    #[must_use]
    pub const fn maximum_verify_tokens(self) -> usize {
        self.maximum_verify_tokens
    }
}

/// One decode item a draft plan may extend into a verify item.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DraftCandidate {
    request_id: RequestId,
    maximum_draft_tokens: usize,
    remaining_tokens: usize,
}

impl DraftCandidate {
    #[must_use]
    pub const fn request_id(self) -> RequestId {
        self.request_id
    }

    /// The most drafts this request can verify now. It is bounded by the
    /// batch limits and by the request's remaining output tokens, so that
    /// every emitted token still fits the request's KV quota.
    #[must_use]
    pub const fn maximum_draft_tokens(self) -> usize {
        self.maximum_draft_tokens
    }

    /// Output tokens the request may still emit, this step's included.
    #[must_use]
    pub const fn remaining_tokens(self) -> usize {
        self.remaining_tokens
    }
}

/// Room a draft plan has left in the batch being formed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DraftBudget {
    spare_draft_tokens: usize,
    spare_rows: usize,
    prefill: bool,
}

impl DraftBudget {
    #[must_use]
    pub const fn new(spare_draft_tokens: usize, spare_rows: usize) -> Self {
        Self {
            spare_draft_tokens,
            spare_rows,
            prefill: false,
        }
    }

    /// The same room in a batch that also prefills.
    #[must_use]
    pub const fn with_prefill(mut self, prefill: bool) -> Self {
        self.prefill = prefill;
        self
    }

    /// The batch also holds prefill items, so its time is not a decode
    /// step's.
    #[must_use]
    pub const fn prefill(self) -> bool {
        self.prefill
    }

    /// Drafts the token budget can still pay for, at the decode cost each.
    #[must_use]
    pub const fn spare_draft_tokens(self) -> usize {
        self.spare_draft_tokens
    }

    /// Drafts the decode-phase row limit still admits.
    #[must_use]
    pub const fn spare_rows(self) -> usize {
        self.spare_rows
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchItem {
    request_id: RequestId,
    phase: BatchPhase,
    token_cost: usize,
    expects_output: bool,
    anchor: Option<usize>,
}

impl BatchItem {
    #[must_use]
    pub const fn request_id(self) -> RequestId {
        self.request_id
    }

    #[must_use]
    pub const fn phase(self) -> BatchPhase {
        self.phase
    }

    #[must_use]
    pub const fn token_cost(self) -> usize {
        self.token_cost
    }

    #[must_use]
    pub const fn expects_output(self) -> bool {
        self.expects_output
    }

    /// For a prompt-ending prefill item that runs past its prompt-end
    /// anchor, the anchor's tokens: the model keeps its state there and the
    /// commit captures it ([`SchedulerConfig::with_anchor_items`]).
    #[must_use]
    pub const fn anchor_tokens(self) -> Option<usize> {
        self.anchor
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchOutcome {
    request_id: RequestId,
    phase: BatchPhase,
    emitted_tokens: usize,
    terminal: bool,
}

impl BatchOutcome {
    #[must_use]
    pub const fn success(
        request_id: RequestId,
        phase: BatchPhase,
        emitted_tokens: usize,
        terminal: bool,
    ) -> Self {
        Self {
            request_id,
            phase,
            emitted_tokens,
            terminal,
        }
    }

    #[must_use]
    pub const fn request_id(self) -> RequestId {
        self.request_id
    }

    #[must_use]
    pub const fn phase(self) -> BatchPhase {
        self.phase
    }

    #[must_use]
    pub const fn emitted_tokens(self) -> usize {
        self.emitted_tokens
    }

    #[must_use]
    pub const fn terminal(self) -> bool {
        self.terminal
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BatchId(u64);

impl BatchId {
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0.wrapping_add(1))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Batch {
    id: BatchId,
    items: Vec<BatchItem>,
    token_cost: usize,
    kv_plan: PreparedAppendBatch,
    prefill_cap: Option<usize>,
    shape: StepShape,
    predicted_micros: Option<u64>,
}

impl Batch {
    #[must_use]
    pub const fn id(&self) -> BatchId {
        self.id
    }

    #[must_use]
    pub fn items(&self) -> &[BatchItem] {
        &self.items
    }

    #[must_use]
    pub const fn token_cost(&self) -> usize {
        self.token_cost
    }

    #[must_use]
    pub const fn kv_plan(&self) -> &PreparedAppendBatch {
        &self.kv_plan
    }

    /// The prefill token cap the decode-aware or TBT budget selected for this
    /// batch, whether or not prefill used it. Both select a cap only beside a
    /// decoding request. None under the fixed budget and for prefill-only
    /// batches.
    #[must_use]
    pub const fn prefill_cap(&self) -> Option<usize> {
        self.prefill_cap
    }

    /// The batch's prefill tokens, attention pairs and decode items.
    #[must_use]
    pub const fn shape(&self) -> StepShape {
        self.shape
    }

    /// The TBT model's estimated execution time of a batch that prefills
    /// beside decoding requests; None otherwise or while untrained.
    #[must_use]
    pub const fn predicted_micros(&self) -> Option<u64> {
        self.predicted_micros
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmissionReport {
    admitted: Vec<RequestId>,
    blocked: Option<Overload>,
}

impl AdmissionReport {
    #[must_use]
    pub fn admitted(&self) -> &[RequestId] {
        &self.admitted
    }

    #[must_use]
    pub const fn blocked(&self) -> Option<Overload> {
        self.blocked
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommitReport {
    completed: Vec<RequestId>,
}

impl CommitReport {
    #[must_use]
    pub fn completed(&self) -> &[RequestId] {
        &self.completed
    }
}

/// The prompt a request brings to the prefix cache.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixRequestV1 {
    chain: Arc<PrefixChainV1>,
    lookup: bool,
}

impl PrefixRequestV1 {
    /// `lookup` is false for a request that must compute every prompt row,
    /// such as one that asks for prompt logprobs. It still creates anchors.
    #[must_use]
    pub const fn new(chain: Arc<PrefixChainV1>, lookup: bool) -> Self {
        Self { chain, lookup }
    }
}

/// A request resumed from an anchor whose bundle state is not restored yet.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixRestoreV1 {
    request_id: RequestId,
    tokens: usize,
    pages: Vec<BlockId>,
    resumes_decoding: bool,
}

impl PrefixRestoreV1 {
    #[must_use]
    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    /// Committed tokens the anchor covers, including generated inputs.
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }

    /// Whether a successful restore resumes the request's pending decode
    /// input without replay. Shorter prefix hits still require prefill.
    #[must_use]
    pub const fn resumes_decoding(&self) -> bool {
        self.resumes_decoding
    }

    /// The pages that hold the bundle snapshot to restore from.
    #[must_use]
    pub fn pages(&self) -> &[BlockId] {
        &self.pages
    }
}

/// A committed prefill boundary the host should capture as an anchor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixCaptureV1 {
    request_id: RequestId,
    tokens: usize,
    key: PrefixCacheKeyV1,
    chain: PrefixCacheKeyV1,
    handle: SequenceHandle,
    /// The request finished at this commit, and its sequence is released only
    /// once the capture ends.
    deferred: bool,
    swap: bool,
    private: bool,
    /// The anchor covers a strict prefix of the sequence's committed tokens,
    /// whose state the bundle kept inside the step ([`BatchItem::anchor_tokens`]).
    interior: bool,
}

impl PrefixCaptureV1 {
    #[must_use]
    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PreparedCommitItem {
    record_index: usize,
    prefilled_tokens: usize,
    emitted_tokens: usize,
    state: RequestState,
    terminal: bool,
    /// Whether this commit publishes complete state at capture_tokens.
    capture: bool,
    capture_tokens: usize,
    /// The anchor the item kept its state at inside its rows.
    anchor: Option<usize>,
    /// Whether the commit replaces a tail page the request shared,
    /// releasing the page reserved for its copy.
    release_tail: bool,
}

/// All scheduler state needed to finish a batch without another fallible step.
#[derive(Debug)]
pub struct PreparedBatchCommit {
    scheduler_identity: Arc<()>,
    pending_generation: u64,
    id: BatchId,
    preflight_state: SchedulerPreflightState,
    kv_commit: PreparedKvCommit,
    items: Vec<PreparedCommitItem>,
    completed: Vec<RequestId>,
    next_tick: u64,
    prefer_decode_after: bool,
    mixed_decode_steps_after: u32,
    active_sequences_after: usize,
    reserved_blocks_after: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RequestRecord {
    spec: RequestSpec,
    guaranteed_prefill_chunks: Option<usize>,
    state: RequestState,
    handle: Option<SequenceHandle>,
    reserved_blocks: usize,
    /// KV tokens `reserved_blocks` covers, with the shared pages and pins it
    /// accounts for; zero while the request holds no sequence.
    reserved_tokens: usize,
    /// Prefill progress towards the prompt and the replayed tokens.
    prefilled_tokens: usize,
    emitted_tokens: usize,
    /// Emitted tokens the prefill recomputes after a preemption. They follow
    /// the prompt as inputs and are not sampled again.
    replayed_tokens: usize,
    preemptions: usize,
    arrival: u64,
    last_scheduled: u64,
    /// The scheduling tick when the request was queued.
    queued_tick: u64,
    prefix: Option<PrefixRequestV1>,
    decode_capture: Option<usize>,
    /// A sibling that must commit this cacheable boundary before admission.
    prefix_dependency: Option<(RequestId, usize)>,
    /// Prompt boundaries to capture anchors at, ascending.
    captures: Vec<usize>,
    /// The prompt-end capture point when it lies before the prompt end and
    /// the bundle can keep state inside a chunk: a chunk may run past it.
    end_anchor: Option<usize>,
    /// Anchor snapshot pages pinned until the bundle restores from them.
    restore: Option<Vec<BlockId>>,
    cached_tokens: usize,
    /// Prompt position up to which committed prefill was compared with the
    /// blocks earlier requests computed. A recompute after preemption, or
    /// after a refused restore, repeats rows already compared and counts
    /// only past it.
    compared_tokens: usize,
    own_restore: bool,
    /// The private anchor of the request's last preemption, which only this
    /// request restores; `None` once restored, abandoned or never captured.
    own_anchor: Option<PrefixCacheKeyV1>,
    /// An outgoing copy may still read this boundary after its host owner
    /// was discarded. Rewriting resumes must wait for confirmed completion.
    own_anchor_transfer: Option<runtime_kv::KvTransferId>,
    /// Full pages a restored request shares from its anchor. They are
    /// reserved once for all live requests that share them, not in
    /// `reserved_blocks`.
    shared_blocks: Vec<BlockId>,
    /// The hit resumed inside a page (or at a page end, when requests
    /// rewrite rows of their last page), and `reserved_blocks` holds one
    /// more page until the first commit replaces that shared tail: the tail
    /// stays held while every sharer writes its own copy.
    shared_tail: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PendingBatch {
    generation: u64,
    id: BatchId,
    items: Vec<BatchItem>,
    prefer_decode_after: bool,
    /// The owed decode-only steps once this batch commits.
    mixed_decode_steps_after: u32,
    kv_plan_id: PreparedAppendId,
}

#[derive(Debug)]
struct SchedulerPreflightState {
    requests: Vec<RequestRecord>,
    active_sequences: usize,
    reserved_blocks: usize,
    next_arrival: u64,
    next_batch_id: u64,
    next_pending_generation: u64,
    scheduling_tick: u64,
    prefer_decode: bool,
    pending: PendingBatch,
}

impl SchedulerPreflightState {
    fn capture(scheduler: &Scheduler, pending: &PendingBatch) -> Result<Self, ScheduleError> {
        let mut requests = Vec::new();
        requests
            .try_reserve_exact(scheduler.requests.len())
            .map_err(|_| ScheduleError::MetadataAllocationFailed)?;
        requests.extend_from_slice(&scheduler.requests);
        let mut items = Vec::new();
        items
            .try_reserve_exact(pending.items.len())
            .map_err(|_| ScheduleError::MetadataAllocationFailed)?;
        items.extend_from_slice(&pending.items);
        Ok(Self {
            requests,
            active_sequences: scheduler.active_sequences,
            reserved_blocks: scheduler.reserved_blocks,
            next_arrival: scheduler.next_arrival,
            next_batch_id: scheduler.next_batch_id,
            next_pending_generation: scheduler.next_pending_generation,
            scheduling_tick: scheduler.scheduling_tick,
            prefer_decode: scheduler.prefer_decode,
            pending: PendingBatch {
                generation: pending.generation,
                id: pending.id,
                items,
                prefer_decode_after: pending.prefer_decode_after,
                mixed_decode_steps_after: pending.mixed_decode_steps_after,
                kv_plan_id: pending.kv_plan_id,
            },
        })
    }

    fn matches(&self, scheduler: &Scheduler) -> bool {
        self.requests == scheduler.requests
            && self.active_sequences == scheduler.active_sequences
            && self.reserved_blocks == scheduler.reserved_blocks
            && self.next_arrival == scheduler.next_arrival
            && self.next_batch_id == scheduler.next_batch_id
            && self.next_pending_generation == scheduler.next_pending_generation
            && self.scheduling_tick == scheduler.scheduling_tick
            && self.prefer_decode == scheduler.prefer_decode
            && scheduler.pending.as_ref() == Some(&self.pending)
    }
}

#[derive(Debug)]
pub struct Scheduler {
    identity: Arc<()>,
    config: SchedulerConfig,
    kv: KvPool,
    requests: Vec<RequestRecord>,
    active_sequences: usize,
    reserved_blocks: usize,
    next_arrival: u64,
    next_batch_id: u64,
    next_pending_generation: u64,
    scheduling_tick: u64,
    prefer_decode: bool,
    pending: Option<PendingBatch>,
    captures: Vec<PrefixCaptureV1>,
    /// Live restored requests sharing each full prefix page.
    shared_blocks: BTreeMap<BlockId, usize>,
    /// Pages an optimistic admission leaves free of reservations.
    watermark_pages: usize,
    step_times: StepTimes,
    step_model: StepModel,
    /// Decode-only steps `PrefillBudget::Tbt` still owes after a mixed step.
    mixed_decode_steps: u32,
    offload: Option<OffloadConfig>,
    resumes: BTreeMap<RequestId, ResumePlan>,
}

impl Clone for Scheduler {
    fn clone(&self) -> Self {
        Self {
            identity: Arc::new(()),
            config: self.config,
            kv: self.kv.clone(),
            requests: self.requests.clone(),
            active_sequences: self.active_sequences,
            reserved_blocks: self.reserved_blocks,
            next_arrival: self.next_arrival,
            next_batch_id: self.next_batch_id,
            next_pending_generation: self.next_pending_generation,
            scheduling_tick: self.scheduling_tick,
            prefer_decode: self.prefer_decode,
            pending: self.pending.clone(),
            captures: self.captures.clone(),
            shared_blocks: self.shared_blocks.clone(),
            watermark_pages: self.watermark_pages,
            step_times: self.step_times.clone(),
            step_model: self.step_model.clone(),
            mixed_decode_steps: self.mixed_decode_steps,
            offload: self.offload,
            resumes: self.resumes.clone(),
        }
    }
}

impl PartialEq for Scheduler {
    fn eq(&self, other: &Self) -> bool {
        self.config == other.config
            && self.kv == other.kv
            && self.requests == other.requests
            && self.active_sequences == other.active_sequences
            && self.reserved_blocks == other.reserved_blocks
            && self.next_arrival == other.next_arrival
            && self.next_batch_id == other.next_batch_id
            && self.next_pending_generation == other.next_pending_generation
            && self.scheduling_tick == other.scheduling_tick
            && self.prefer_decode == other.prefer_decode
            && self.pending == other.pending
            && self.captures == other.captures
            && self.shared_blocks == other.shared_blocks
            && self.watermark_pages == other.watermark_pages
            && self.step_times == other.step_times
            && self.step_model == other.step_model
            && self.mixed_decode_steps == other.mixed_decode_steps
            && self.offload == other.offload
            && self.resumes == other.resumes
    }
}

impl Eq for Scheduler {}

impl Scheduler {
    pub fn new(config: SchedulerConfig, geometry: KvGeometry) -> Result<Self, ScheduleError> {
        Self::new_with_kv_metrics(config, geometry, KvMetrics::default())
    }

    pub fn new_with_kv_metrics(
        config: SchedulerConfig,
        geometry: KvGeometry,
        kv_metrics: KvMetrics,
    ) -> Result<Self, ScheduleError> {
        let kv = KvPool::new_with_metrics(geometry, config.maximum_sequences, kv_metrics)
            .map_err(ScheduleError::Kv)?
            .with_last_page_rewrites(config.rewrites_last_page);
        let mut requests = Vec::new();
        requests
            .try_reserve_exact(config.tracking_capacity)
            .map_err(|_| ScheduleError::MetadataAllocationFailed)?;
        let watermark_pages = match config.admission {
            AdmissionPolicy::Reserved => 0,
            AdmissionPolicy::Optimistic(optimistic) => {
                let pages =
                    geometry.block_capacity() as u128 * u128::from(optimistic.watermark_millionths);
                usize::try_from(pages.div_ceil(1_000_000))
                    .map_err(|_| ScheduleError::InvariantViolation)?
            }
        };
        Ok(Self {
            identity: Arc::new(()),
            config,
            kv,
            requests,
            active_sequences: 0,
            reserved_blocks: 0,
            next_arrival: 0,
            next_batch_id: 1,
            next_pending_generation: 1,
            scheduling_tick: 0,
            prefer_decode: false,
            pending: None,
            captures: Vec::new(),
            shared_blocks: BTreeMap::new(),
            watermark_pages,
            step_times: StepTimes::default(),
            step_model: StepModel::default(),
            mixed_decode_steps: 0,
            offload: None,
            resumes: BTreeMap::new(),
        })
    }

    /// Records successful engine wall time, including completion of device work.
    /// Verify steps and media encodes are excluded by the engine: their extra
    /// work cannot be attributed to ordinary decode or text prefill.
    pub fn observe_step_time(&mut self, prefill_tokens: usize, decode_rows: usize, micros: u64) {
        if matches!(self.config.prefill_budget, PrefillBudget::DecodeAware(_)) {
            self.step_times.observe(prefill_tokens, decode_rows, micros);
        }
    }

    /// Records the successful engine wall time of a committed batch for the
    /// TBT model. Verify steps train it; media encodes are excluded by the
    /// engine, since their encode time is not prefill or decode work.
    pub fn observe_batch_time(&mut self, shape: StepShape, micros: u64) {
        if matches!(self.config.prefill_budget, PrefillBudget::Tbt(_)) {
            self.step_model.observe(shape, micros);
        }
    }

    /// The TBT model's current state; None under other budgets.
    #[must_use]
    pub fn mixed_estimates(&self) -> Option<MixedEstimates> {
        matches!(self.config.prefill_budget, PrefillBudget::Tbt(_))
            .then(|| self.step_model.estimates())
    }

    /// The decode-aware estimator's current state; None under the fixed budget.
    #[must_use]
    pub fn step_time_estimates(&self) -> Option<StepTimeEstimates> {
        matches!(self.config.prefill_budget, PrefillBudget::DecodeAware(_))
            .then(|| self.step_times.estimates())
    }

    pub fn submit(&mut self, spec: RequestSpec) -> Result<(), SubmitError> {
        self.submit_with_prefix(spec, None)
    }

    /// Queues a request with the prompt chain the prefix cache matches.
    ///
    /// The chain is ignored unless prefix caching is enabled, and refused
    /// when it does not cover exactly the prompt.
    pub fn submit_with_prefix(
        &mut self,
        spec: RequestSpec,
        prefix: Option<PrefixRequestV1>,
    ) -> Result<(), SubmitError> {
        if prefix.as_ref().is_some_and(|prefix| {
            prefix.chain.tokens().len() != spec.prompt_tokens
                || prefix.chain.block_tokens() != self.kv.geometry().block_tokens()
        }) {
            return Err(SubmitError::Invalid(InvalidRequest::PrefixMismatch));
        }
        let prefix = prefix.filter(|_| self.config.prefix_state_pages.is_some());
        if self.requests.iter().any(|record| record.spec.id == spec.id) {
            return Err(SubmitError::DuplicateId(spec.id));
        }
        if self.requests.len() == self.config.tracking_capacity {
            return Err(SubmitError::Overloaded(Overload::TrackingCapacity));
        }
        let in_flight = self
            .requests
            .iter()
            .filter(|record| !record.state.is_terminal())
            .count();
        if in_flight >= self.config.capacity_profile.maximum_in_flight {
            return Err(SubmitError::Overloaded(Overload::ProfileCapacity {
                profile: self.config.capacity_profile.name,
                maximum: self.config.capacity_profile.maximum_in_flight,
            }));
        }
        if spec.maximum_kv_tokens > self.kv.geometry().token_capacity() {
            return Err(SubmitError::Invalid(InvalidRequest::ExceedsKvPool {
                maximum: self.kv.geometry().token_capacity(),
                requested: spec.maximum_kv_tokens,
            }));
        }
        if spec.media_pages > self.config.media_pages {
            return Err(SubmitError::Invalid(InvalidRequest::ExceedsMediaArena {
                maximum: self.config.media_pages,
                requested: spec.media_pages,
            }));
        }
        let arrival = self.next_arrival;
        self.next_arrival = self
            .next_arrival
            .checked_add(1)
            .ok_or(SubmitError::RequestOrdinalExhausted)?;
        let guaranteed_prefill_chunks = self
            .config
            .guaranteed_prefill_chunks
            .filter(|_| !spec.prompt_scoring && spec.media_pages == 0);
        self.requests.push(RequestRecord {
            spec,
            guaranteed_prefill_chunks,
            state: RequestState::Queued,
            handle: None,
            reserved_blocks: 0,
            reserved_tokens: 0,
            prefilled_tokens: 0,
            emitted_tokens: 0,
            replayed_tokens: 0,
            preemptions: 0,
            arrival,
            last_scheduled: 0,
            queued_tick: self.scheduling_tick,
            prefix,
            decode_capture: None,
            prefix_dependency: None,
            captures: Vec::new(),
            end_anchor: None,
            restore: None,
            cached_tokens: 0,
            compared_tokens: 0,
            own_restore: false,
            own_anchor: None,
            own_anchor_transfer: None,
            shared_blocks: Vec::new(),
            shared_tail: false,
        });
        Ok(())
    }

    /// Plan cache reuse among newly submitted siblings before any is admitted.
    /// Matching full-page keys include token, salt, media, and position identity.
    /// Distinct prompts and requests needing all prompt scores stay independent.
    pub fn share_prompt_prefixes(&mut self, ids: &[RequestId]) -> Result<(), ScheduleError> {
        self.ensure_no_pending()?;
        if self.config.prefix_state_pages.is_none() {
            return Ok(());
        }
        let mut indices = ids
            .iter()
            .map(|id| self.record_index(*id))
            .collect::<Result<Vec<_>, _>>()?;
        indices.sort_unstable();
        indices.dedup();
        if indices.iter().any(|index| {
            self.requests[*index].state != RequestState::Queued
                || self.requests[*index].preemptions != 0
        }) {
            return Err(ScheduleError::InvariantViolation);
        }
        let block = self.kv.geometry().block_tokens();
        for (position, &follower) in indices.iter().enumerate() {
            let Some(prefix) = self.requests[follower]
                .prefix
                .as_ref()
                .filter(|prefix| prefix.lookup)
            else {
                continue;
            };
            let chain = Arc::clone(&prefix.chain);
            for &leader in &indices[..position] {
                let Some(prior) = &self.requests[leader].prefix else {
                    continue;
                };
                // Leave a prompt token to compute the follower's first logits.
                let maximum = ((chain.tokens().len() - 1) / block).min(prior.chain.full_blocks());
                let shared = (1..=maximum)
                    .take_while(|blocks| chain.chain(*blocks) == prior.chain.chain(*blocks))
                    .last()
                    .unwrap_or(0);
                let granule = self.config.prefill_granule;
                let boundary = (1..=shared)
                    .rev()
                    .map(|blocks| blocks * block)
                    .find(|tokens| {
                        tokens.is_multiple_of(granule)
                            && self.requests[leader]
                                .guaranteed_prefill_chunks
                                .is_none_or(|chunk| {
                                    tokens.is_multiple_of(chunk)
                                        || *tokens == self.requests[leader].spec.prompt_tokens
                                })
                            && !chain.splits_media(*tokens)
                            && !prior.chain.splits_media(*tokens)
                    });
                let Some(boundary) = boundary else {
                    continue;
                };
                if !self
                    .kv
                    .has_anchor(chain.key_at(boundary).expect("full prefix block"))
                {
                    self.requests[follower].prefix_dependency =
                        Some((self.requests[leader].spec.id, boundary));
                    self.requests[leader].captures.push(boundary);
                    self.requests[leader].captures.sort_unstable();
                    self.requests[leader].captures.dedup();
                }
                break;
            }
        }
        Ok(())
    }

    pub fn admit(&mut self) -> Result<AdmissionReport, ScheduleError> {
        self.admit_with_lookahead(0)
    }

    /// Admits requests with space for the pending input and up to `lookahead`
    /// drafts when an anchor restores their own exact decode boundary.
    /// Cold and shorter-prefix recomputes reserve only their prefill target.
    pub fn admit_with_lookahead(
        &mut self,
        lookahead: usize,
    ) -> Result<AdmissionReport, ScheduleError> {
        if self.pending.is_some() {
            return Err(ScheduleError::PendingBatch);
        }
        let mut admitted = Vec::new();
        admitted
            .try_reserve_exact(self.config.maximum_sequences - self.active_sequences)
            .map_err(|_| ScheduleError::MetadataAllocationFailed)?;
        let mut blocked = None;
        for index in 0..self.requests.len() {
            if self.requests[index].state != RequestState::Queued {
                continue;
            }
            if let Some((leader, boundary)) = self.requests[index].prefix_dependency
                && self.record(leader).is_ok_and(|record| {
                    !record.state.is_terminal() && record.prefilled_tokens < boundary
                })
            {
                // The leader's capture is published after its committed step.
                // A cancelled leader or a failed capture lets the follower run cold.
                continue;
            }
            if self.active_sequences == self.config.maximum_sequences {
                blocked = Some(Overload::MaximumSequences);
                break;
            }
            let spec = self.requests[index].spec;
            let target = Self::prefill_target(&self.requests[index]);
            let mut lookup = self.requests[index]
                .prefix
                .as_ref()
                .filter(|prefix| prefix.lookup)
                .map(|prefix| Arc::clone(&prefix.chain));
            let resume = self.resumes.get(&spec.id).cloned();
            let mut own_gpu = resume
                .as_ref()
                .map(|plan| plan.key)
                .or(self.requests[index].own_anchor)
                .filter(|key| self.kv.anchor_pages(*key).is_some());
            let mut host = resume
                .as_ref()
                .filter(|plan| own_gpu.is_none() && plan.host.is_some());
            // The request's own preemption anchor is dropped once restored, so
            // its partial tail page becomes the request's alone, and needs no
            // copy, unless another owner or a transfer also holds it.
            let consumed_tail = resume.is_none()
                && own_gpu.is_some()
                && own_gpu == self.requests[index].own_anchor
                && own_gpu
                    .and_then(|key| self.kv.anchor_pages(key))
                    .and_then(|(_, pages, _)| pages.last().copied())
                    .is_some_and(|page| self.kv.block_ref_count(page) == Ok(1));
            let hit_tokens = own_gpu
                .and_then(|key| self.kv.anchor_pages(key).map(|a| a.0))
                .or_else(|| host.map(|plan| plan.tokens))
                .or_else(|| {
                    lookup
                        .as_ref()
                        .and_then(|chain| self.kv.lookup_anchor(chain, target - 1))
                        .map(|(_, tokens)| tokens)
                });
            let exact = hit_tokens.is_some_and(|tokens| {
                Self::restores_decode_boundary(&self.requests[index], tokens)
            });
            // A pinned request reserves its worst case. Any other reserves its
            // prefill target and grows before each decode step, keeping the
            // watermark free while another sequence is live.
            let pinned = self.pinned(&self.requests[index]);
            let mut reserved_tokens = if pinned {
                spec.maximum_kv_tokens
            } else if exact {
                target.saturating_add(lookahead).min(spec.maximum_kv_tokens)
            } else {
                target
            };
            let watermark = if pinned || self.active_sequences == 0 {
                0
            } else {
                self.watermark_pages
            };
            let mut blocks = self
                .kv
                .geometry()
                .blocks_for(reserved_tokens)
                .map_err(ScheduleError::Kv)?;
            // A hit leaves the last input uncommitted. A new request prefills
            // it for its first logits; an exact resume decodes it.
            let available = (self.kv.geometry().block_capacity() - self.reserved_blocks)
                .saturating_sub(watermark);
            let block_tokens = self.kv.geometry().block_tokens();
            let rewrites = self.config.rewrites_last_page;
            let copies_tail = |tokens: usize, own: bool| {
                Self::copies_tail(block_tokens, rewrites, tokens) && !(own && consumed_tail)
            };
            let hit_pages = |tokens: usize, pins: usize, own: bool| {
                pins + usize::from(copies_tail(tokens, own))
            };
            let state_pages = self
                .offload
                .map_or(self.config.prefix_state_pages.unwrap_or(0), |c| {
                    c.snapshot_pages
                });
            let mut restore_pages = hit_tokens.map_or(0, |tokens| {
                if host.is_some() {
                    state_pages
                } else {
                    hit_pages(tokens, state_pages, own_gpu.is_some())
                }
            });
            // Drafts are optional. Keep a fitting restore even when its
            // lookahead and staging pins would exceed the available pages.
            if exact && !pinned && blocks + restore_pages > available {
                reserved_tokens = target;
                blocks = self
                    .kv
                    .geometry()
                    .blocks_for(target)
                    .map_err(ScheduleError::Kv)?;
            }
            // A surviving GPU anchor may need a shared-tail page that an
            // exclusive host reload does not. Try that before cold recompute.
            if own_gpu.is_some()
                && blocks + restore_pages > available
                && resume.as_ref().is_some_and(|plan| plan.host.is_some())
            {
                own_gpu = None;
                host = resume.as_ref();
                restore_pages = state_pages;
            }
            if host.is_some() && blocks + restore_pages > self.kv.geometry().block_capacity() {
                self.resumes
                    .get_mut(&spec.id)
                    .expect("resume exists")
                    .discard = true;
                blocked = Some(Overload::KvCapacity);
                break;
            }
            if host.is_some_and(|plan| !plan.ready) || blocks > available {
                blocked = Some(Overload::KvCapacity);
                break;
            }
            // A continuation that rewrites committed rows (an MTP drafter
            // behind its frontier, up to a page before the last) may write
            // any page of its own anchor, which an outgoing copy of that
            // anchor may still be reading: resume only once the copy is done
            // and its lease on the pages is gone.
            if rewrites && own_gpu.is_some() && self.requests[index].own_anchor_transfer.is_some() {
                blocked = Some(Overload::KvCapacity);
                break;
            }
            // A hit reserves its cold pages and its snapshot pins until its
            // restore, and only then gives back the full pages it shares. One
            // that resumes inside a page also holds a page for the copy of that
            // shared tail until its first commit, unless the tail was only its
            // own consumed anchor's. When the hit's extra pages
            // do not fit, the request waits for restores already pending,
            // which lower the reservation. With none pending it runs cold:
            // waiting could last forever, since nothing else may release a
            // reservation, while a cold sequence gets its pages by evicting
            // anchors.
            if host.is_some() && blocks + restore_pages > available {
                // A reload cannot fit while live reservations own its staging.
                // If it can never fit by itself, discard it for recompute.
                // Only transient pressure remains; impossible reloads were
                // discarded above, independently of other live requests.
                blocked = Some(Overload::KvCapacity);
                break;
            }
            if let Some(chain) = &lookup
                && hit_tokens.is_some()
                && blocks + restore_pages > available
            {
                if self.requests.iter().any(|record| record.restore.is_some()) {
                    blocked = Some(Overload::KvCapacity);
                    break;
                }
                if own_gpu.is_none() && host.is_none() {
                    self.kv.metrics().prefix_cache_missed(chain.tokens().len());
                }
                lookup = None;
            }
            let own_hit = own_gpu.is_some() && blocks + restore_pages <= available;
            let hit = if let Some(key) = own_gpu.filter(|_| own_hit) {
                Some(
                    self.kv
                        .resume_anchor(key, spec.maximum_kv_tokens)
                        .map_err(ScheduleError::Kv)?,
                )
            } else if let Some(plan) = host {
                let hit = match self.kv.create_initialized_sequence(
                    plan.tokens,
                    spec.maximum_kv_tokens,
                    state_pages,
                ) {
                    Ok(hit) => hit,
                    Err(KvError::PoolExhausted { .. }) => {
                        blocked = Some(Overload::KvCapacity);
                        break;
                    }
                    Err(error) => return Err(ScheduleError::Kv(error)),
                };
                self.resumes
                    .get_mut(&spec.id)
                    .expect("resume exists")
                    .destination = Some(hit.clone());
                Some(hit)
            } else {
                match &lookup {
                    Some(chain) => self
                        .kv
                        .resume_prefix(chain, target - 1, spec.maximum_kv_tokens)
                        .map_err(ScheduleError::Kv)?,
                    None => None,
                }
            };
            let (handle, prefilled, restore) = match hit {
                Some(hit) => (hit.handle(), hit.tokens(), Some(hit.state().to_vec())),
                None => (
                    self.kv
                        .create_sequence(spec.maximum_kv_tokens)
                        .map_err(ScheduleError::Kv)?,
                    0,
                    None,
                ),
            };
            let blocks = blocks
                + restore.as_ref().map_or(0, |pins| {
                    if host.is_some() {
                        pins.len()
                    } else {
                        hit_pages(prefilled, pins.len(), own_hit)
                    }
                });
            // A recompute captures nothing new: its preemption published its
            // prefix, and its first admission already captured its prompt.
            self.requests[index].captures = if self.requests[index].replayed_tokens > 0 {
                Vec::new()
            } else {
                self.capture_points(index, prefilled)
            };
            let granule = self.config.prefill_granule;
            let prompt = self.requests[index].spec.prompt_tokens;
            let end = prompt / granule * granule;
            self.requests[index].end_anchor = (self.config.anchor_items > 0
                && end < prompt
                && self.requests[index].captures.contains(&end))
            .then_some(end);
            self.requests[index].prefilled_tokens = prefilled;
            if self.requests[index].preemptions == 0 {
                self.requests[index].cached_tokens = prefilled;
            }
            self.requests[index].shared_tail =
                host.is_none() && restore.is_some() && copies_tail(prefilled, own_hit);
            self.requests[index].own_restore = own_gpu.is_some() || host.is_some();
            self.requests[index].restore = restore;
            self.requests[index].state = RequestState::Admitted;
            self.requests[index].prefix_dependency = None;
            self.requests[index].handle = Some(handle);
            self.requests[index].reserved_blocks = blocks;
            self.requests[index].reserved_tokens = reserved_tokens;
            self.active_sequences += 1;
            self.reserved_blocks += blocks;
            admitted.push(self.requests[index].spec.id);
        }
        Ok(AdmissionReport { admitted, blocked })
    }

    pub fn next_batch(&mut self) -> Result<Option<Batch>, ScheduleError> {
        self.next_batch_with_drafts(DraftLimits::NONE, |_, _| Vec::new())
    }

    /// Forms the next batch and lets `plan` turn decode items into verify items.
    ///
    /// `plan` runs once, after the batch's items are chosen and before any KV
    /// is prepared, and only when `limits` allows drafts and the batch holds a
    /// decode item. It receives the decode items in batch order and returns a
    /// draft count for each. A count of zero leaves the item a plain decode.
    /// A plan that returns the wrong length, exceeds a candidate's maximum, or
    /// overdraws the budget is refused with `InvalidDraftPlan`.
    pub fn next_batch_with_drafts<F>(
        &mut self,
        limits: DraftLimits,
        plan: F,
    ) -> Result<Option<Batch>, ScheduleError>
    where
        F: FnOnce(&[DraftCandidate], DraftBudget) -> Vec<usize>,
    {
        self.next_batch_planned(limits, |_, _, _| ChunkMedia::default(), plan)
    }

    /// Forms the next batch with draft plans and per-chunk media demands.
    ///
    /// `chunk_media(request, offset, tokens)` gives what the media of a
    /// prefill chunk of `tokens` prompt tokens from `offset` need. The start
    /// gate also asks it for a request's whole remaining prompt, so a range
    /// must report at least the pages of any range inside it. A chunk is
    /// charged its tokens plus its encode cost. The first chunk whose tokens
    /// fit the remaining budget but whose total does not is added last, after
    /// every item that fits, and leaves no budget for drafts; any other such
    /// chunk waits. Its tokens are held back from the items after it, so only
    /// the encode cost overruns: the tokens of a batch never exceed the
    /// budget.
    ///
    /// The chunks of one step cover at most the configured media pages
    /// together, since the step pins every media item they cover; pins last
    /// one step. A chunk whose media exceed the whole arena is shortened at a
    /// granule boundary to the longest one that fits, and a chunk that does
    /// not fit what the step has left waits, like a token overrun. `plan` is
    /// the draft plan of `next_batch_with_drafts`.
    pub fn next_batch_planned<C, F>(
        &mut self,
        limits: DraftLimits,
        mut chunk_media: C,
        plan: F,
    ) -> Result<Option<Batch>, ScheduleError>
    where
        C: FnMut(RequestId, usize, usize) -> ChunkMedia,
        F: FnOnce(&[DraftCandidate], DraftBudget) -> Vec<usize>,
    {
        if self.pending.is_some() {
            return Err(ScheduleError::PendingBatch);
        }
        let admission = self.admit_with_lookahead(limits.maximum_draft_tokens)?;
        let mut decode = self.candidates(true)?;
        let mut prefill = self.candidates(false)?;
        // A media prompt that started prefilling keeps the items its chunks
        // straddle in the arena between steps. Another starts, in arrival
        // order, only while its step demand fits the arena beside those of
        // the started ones. That bounds what they hold between chunks, so
        // they rarely evict each other's items to encode them again. Started
        // prompts are never held back. A prompt's demand counts only the
        // media ahead of its prefill, so a turn whose images all lie in its
        // restored prefix demands nothing.
        let mut demand = |record: &RequestRecord| {
            let remaining = Self::prefill_target(record).saturating_sub(record.prefilled_tokens);
            if record.spec.media_pages == 0 || remaining == 0 {
                return 0;
            }
            chunk_media(record.spec.id, record.prefilled_tokens, remaining)
                .pages
                .min(record.spec.media_pages)
        };
        let mut started_media = 0usize;
        for record in &self.requests {
            if record.state == RequestState::Prefilling {
                started_media = started_media.saturating_add(demand(record));
            }
        }
        let mut media_cutoff = u64::MAX;
        for &index in &prefill {
            let record = &self.requests[index];
            if record.state == RequestState::Prefilling {
                continue;
            }
            started_media = started_media.saturating_add(demand(record));
            if started_media > self.config.media_pages {
                media_cutoff = record.arrival;
                break;
            }
        }
        prefill.retain(|index| {
            let record = &self.requests[*index];
            record.state == RequestState::Prefilling
                || record.arrival < media_cutoff
                || demand(record) == 0
        });
        // Until the model trains, tbt caps a mixed step at the progress
        // floor and neither owes nor takes decode-only steps.
        let trained = matches!(self.config.prefill_budget, PrefillBudget::Tbt(_))
            && self.step_model.trained();
        let owed_decode_step = trained && !decode.is_empty() && self.mixed_decode_steps > 0;
        let decode_room = self
            .config
            .scheduled_token_budget
            .saturating_sub(decode.len().saturating_mul(self.config.decode_cost));
        let selected = match self.config.prefill_budget {
            PrefillBudget::DecodeAware(target) if !decode.is_empty() => Some(
                self.step_times
                    .budget(decode.len(), target)
                    .min(decode_room),
            ),
            // Without decode work there is no TBT to protect. Use the whole
            // token budget independently of timings or training.
            PrefillBudget::Tbt(_) if decode.is_empty() => None,
            PrefillBudget::Tbt(policy) => {
                if owed_decode_step {
                    // A decode-only step owed after the last mixed step.
                    prefill.clear();
                }
                // The longest context among the candidates bounds the
                // attention work of any chunk this step takes.
                let context = prefill
                    .iter()
                    .map(|&index| self.requests[index].prefilled_tokens)
                    .max()
                    .unwrap_or(0);
                // Beside decoding requests, bound the step, including the
                // first observations before a fit exists. This is a token
                // bound, not a latency guarantee.
                Some(
                    self.step_model
                        .budget(decode.len(), context, decode_room, policy)
                        .unwrap_or_else(|| policy.min_prefill_tokens().min(decode_room)),
                )
            }
            _ => None,
        };
        // A declared chunk cannot shrink with a timing estimate. It waits
        // for a step with enough room, using the existing whole-chunk
        // admission and starvation protection.
        let selected = selected.filter(|_| self.config.guaranteed_prefill_chunks.is_none());
        let aware = selected.is_some();
        let mut prefill_budget = selected.unwrap_or(self.config.scheduled_token_budget);
        // Tiny custom budgets retain the alternating progress guarantee.
        if prefill_budget == 0 {
            prefill_budget = self
                .config
                .prefill_chunk_limit
                .min(self.config.scheduled_token_budget);
        }
        // Expand only with an explicit capacity and no other active request,
        // including a request waiting for its prefix restore. Mixed steps keep their
        // configured chunks and timing caps. Capture points, scored rows
        // and the prefill granule still bound each item below. Scored
        // requests retain their traffic-independent event boundaries.
        let chunk_limit = if let Some(chunk) = self.config.guaranteed_prefill_chunks {
            chunk
        } else if self.config.lone_prefill_expansion
            && self.active_sequences == 1
            && prefill.len() == 1
            && !self.requests[prefill[0]].spec.prompt_scoring
        {
            self.config.scheduled_token_budget
        } else if aware {
            self.config.prefill_chunk_limit.min(prefill_budget)
        } else {
            self.config.prefill_chunk_limit
        };
        // Each item uses the full chunk selected for this step or the
        // authenticated tail of its prompt. A request that could not
        // take either within a fresh batch budget waits for a later batch
        // instead of being truncated to whatever budget is left.
        let maximum_scored_rows = self.config.maximum_scored_rows;
        let granule = self.config.prefill_granule;
        prefill.retain(|index| {
            let record = &self.requests[*index];
            Self::chunk_tokens(record, chunk_limit, maximum_scored_rows, granule, false)
                <= self.config.scheduled_token_budget
        });
        if prefill.is_empty() && decode.is_empty() {
            if let Some(overload) = admission.blocked {
                return Err(ScheduleError::Overloaded(overload));
            }
            return Ok(None);
        }
        if matches!(self.config.prefill_budget, PrefillBudget::Tbt(_)) && aware {
            // A capped step fits few chunks, and new arrivals, which never
            // ran, would otherwise always sort first. Alternate steps serve
            // the earliest arrival first, so a stream of new prompts cannot
            // starve older work, and the prefill that has waited longest
            // since its last chunk or its queueing, so new turns still start.
            if self.scheduling_tick % 2 == 0 {
                prefill.sort_by_key(|index| self.requests[*index].arrival);
            } else {
                prefill.sort_by_key(|index| Self::waiting_since(&self.requests[*index]));
            }
        } else {
            // Requests that never ran sort first, in arrival order, then by
            // when each last ran. A started prompt that has not run for
            // STARVED_PREFILL_TICKS steps also goes ahead of the requests
            // queued since its last chunk, so a stream of new prompts cannot
            // take the budget its chunk needs on every step. Until then the
            // order, and with it which item takes an anchor, is unchanged; a
            // preempted request keeps the plain order.
            let tick = self.scheduling_tick;
            prefill.sort_by_key(|index| {
                let record = &self.requests[*index];
                let starved = record.preemptions == 0
                    && record.last_scheduled > 0
                    && record.prefilled_tokens > 0
                    && tick.saturating_sub(record.last_scheduled) >= STARVED_PREFILL_TICKS;
                if record.last_scheduled == 0 {
                    (false, record.queued_tick, record.arrival)
                } else {
                    (!starved, record.last_scheduled, record.arrival)
                }
            });
        }
        decode.sort_by_key(|index| {
            let record = &self.requests[*index];
            (record.last_scheduled, record.arrival)
        });
        // A chunk ends on a granule multiple or at its prefill target, so a
        // cap below the granule may reach no legal chunk end, and the capped
        // prefill would never run beside decoding requests. On its
        // prefill-first turn such a step prefills alone, against the whole
        // step budget, from the first candidate's smallest legal chunk.
        let dedicated = aware
            && !self.prefer_decode
            && !decode.is_empty()
            && prefill.first().is_some_and(|&index| {
                Self::chunk_tokens(
                    &self.requests[index],
                    chunk_limit,
                    maximum_scored_rows,
                    granule,
                    false,
                ) > prefill_budget
            });
        if dedicated {
            decode.clear();
            prefill_budget = self.config.scheduled_token_budget;
        }
        let prefill_cap = aware.then_some(prefill_budget);
        // A prompt-ending chunk that reaches its prompt-end anchor runs past
        // it while the bundle's anchor items last (`push_prefill`).
        let mut anchors = self.config.anchor_items;
        let media_arena = self.config.media_pages;
        let mut media_budget = media_arena;

        let mut items = Vec::new();
        items
            .try_reserve_exact(prefill.len() + decode.len())
            .map_err(|_| ScheduleError::MetadataAllocationFailed)?;
        let mut budget = self.config.scheduled_token_budget;
        // Ordinary prefill keeps its original batching. A scored phase also
        // gathers ordinary final-token rows into the same logits buffer.
        let mut scored_budget = if prefill
            .iter()
            .any(|&index| self.requests[index].spec.prompt_scoring)
        {
            maximum_scored_rows
        } else {
            usize::MAX
        };
        // While a scored request waits, rows are scarce and a scored chunk is
        // never cut. The candidate that has waited longest, since its last
        // chunk or since it was queued, then holds its tokens and rows until
        // it is visited, so no request sorted ahead of it (new arrivals sort
        // first) can defer it indefinitely, whether it scores or not.
        let mut reserve = if scored_budget == usize::MAX {
            None
        } else {
            prefill
                .iter()
                .min_by_key(|&&index| Self::waiting_since(&self.requests[index]))
                .map(|&index| {
                    let record = &self.requests[index];
                    let tokens = Self::chunk_tokens(
                        record,
                        chunk_limit,
                        maximum_scored_rows,
                        granule,
                        false,
                    );
                    Reservation {
                        index,
                        tokens,
                        rows: Self::logit_rows(record, tokens),
                    }
                })
        };
        // The first chunk whose encode overruns the budget waits for the end
        // of the batch, after every item that fits.
        let mut overrun: Option<(usize, usize, usize)> = None;
        let mut prefill_offset = 0usize;
        let mut decode_offset = 0usize;
        let both = !prefill.is_empty() && !decode.is_empty();

        // A mixed step pairs one decode with the first prefill candidate the
        // held rows admit, when its tokens fit beside the decode and the
        // held tokens. A holder that cannot run beside a decode is served by
        // the alternating prefill-first step instead, so no mixed step is
        // formed while it waits.
        let held = |index: usize| match reserve {
            Some(reservation) if reservation.index != index => {
                (reservation.tokens, reservation.rows)
            }
            _ => (0, 0),
        };
        let mix_position = prefill.iter().position(|&index| {
            let record = &self.requests[index];
            let tokens =
                Self::chunk_tokens(record, chunk_limit, maximum_scored_rows, granule, false);
            Self::logit_rows(record, tokens) <= scored_budget.saturating_sub(held(index).1)
        });
        let can_mix = both
            && budget > self.config.decode_cost
            && reserve
                .is_none_or(|reservation| reservation.tokens <= budget - self.config.decode_cost)
            && mix_position.is_some_and(|position| {
                let index = prefill[position];
                let record = &self.requests[index];
                Self::chunk_tokens(record, chunk_limit, maximum_scored_rows, granule, false)
                    + held(index).0
                    <= budget - self.config.decode_cost
            });
        macro_rules! push_prefill_at {
            ($index:expr) => {
                Self::push_prefill(
                    &self.requests,
                    &mut items,
                    $index,
                    &mut budget,
                    chunk_limit,
                    &mut prefill_budget,
                    PrefillRows {
                        maximum_scored_rows,
                        granule,
                        scored_budget: &mut scored_budget,
                        reserve: &mut reserve,
                        media_arena,
                        media_budget: &mut media_budget,
                    },
                    0,
                    &mut chunk_media,
                    &mut overrun,
                    &mut anchors,
                )
            };
        }
        if can_mix {
            Self::push_decode(
                &self.requests,
                &mut items,
                decode[decode_offset],
                &mut budget,
                self.config.decode_cost,
            );
            decode_offset += 1;
            let position = mix_position.unwrap_or(0);
            push_prefill_at!(prefill[position]);
            if position == 0 {
                prefill_offset = 1;
            }
            // The holder takes its place before further decodes; a candidate
            // pushed out of order above is declined when visited again.
            while reserve.is_some() && prefill_offset < prefill.len() {
                push_prefill_at!(prefill[prefill_offset]);
                prefill_offset += 1;
            }
        } else if both {
            if self.prefer_decode && budget >= self.config.decode_cost {
                Self::push_decode(
                    &self.requests,
                    &mut items,
                    decode[decode_offset],
                    &mut budget,
                    self.config.decode_cost,
                );
                decode_offset += 1;
            } else {
                // Prefill first: candidates in order until one is pushed or
                // deferred, and until the holder has been visited.
                while prefill_offset < prefill.len() {
                    let pushed = push_prefill_at!(prefill[prefill_offset]);
                    prefill_offset += 1;
                    if (pushed || overrun.is_some()) && reserve.is_none() {
                        break;
                    }
                }
            }
        }

        while budget > 0 && (prefill_offset < prefill.len() || decode_offset < decode.len()) {
            let decode_fits = decode_offset < decode.len() && budget >= self.config.decode_cost;
            let prefill_fits = prefill_offset < prefill.len();
            if decode_fits {
                Self::push_decode(
                    &self.requests,
                    &mut items,
                    decode[decode_offset],
                    &mut budget,
                    self.config.decode_cost,
                );
                decode_offset += 1;
            } else if prefill_fits {
                push_prefill_at!(prefill[prefill_offset]);
                prefill_offset += 1;
            } else {
                break;
            }
        }

        if let Some((index, tokens, token_cost)) = overrun {
            let record = &self.requests[index];
            let remaining = Self::prefill_target(record) - record.prefilled_tokens;
            items.push(BatchItem {
                request_id: record.spec.id,
                phase: BatchPhase::Prefill { tokens },
                token_cost,
                expects_output: tokens == remaining,
                anchor: None,
            });
            budget = 0;
        }
        self.apply_draft_plan(&mut items, budget, limits, plan)?;
        let shape = self.step_shape(&items)?;
        // What the model expects of the batch as packed, for metrics.
        let predicted_micros = (matches!(self.config.prefill_budget, PrefillBudget::Tbt(_))
            && shape.prefill_tokens() > 0
            && shape.decode_items() > 0)
            .then(|| self.step_model.predicted_micros(shape))
            .flatten();
        // Owed decode-only steps follow a trained mixed step and lapse when
        // no request decodes. The count changes only when the batch commits.
        let mixed_decode_steps_after = match self.config.prefill_budget {
            PrefillBudget::Tbt(_) if !trained || shape.decode_items() == 0 => 0,
            PrefillBudget::Tbt(_) if owed_decode_step => self.mixed_decode_steps - 1,
            PrefillBudget::Tbt(policy) if shape.prefill_tokens() > 0 => policy.decode_steps(),
            _ => self.mixed_decode_steps,
        };

        let id = BatchId(self.next_batch_id);
        let next_batch_id = self
            .next_batch_id
            .checked_add(1)
            .ok_or(ScheduleError::BatchIdExhausted)?;
        let pending_generation = self.next_pending_generation;
        let next_pending_generation = self
            .next_pending_generation
            .checked_add(1)
            .ok_or(ScheduleError::PendingGenerationExhausted)?;
        let mut pending_items = Vec::new();
        pending_items
            .try_reserve_exact(items.len())
            .map_err(|_| ScheduleError::MetadataAllocationFailed)?;
        pending_items.extend_from_slice(&items);
        let mut appends = Vec::new();
        appends
            .try_reserve_exact(items.len())
            .map_err(|_| ScheduleError::MetadataAllocationFailed)?;
        for item in &items {
            let record = self.record(item.request_id)?;
            let handle = record.handle.ok_or(ScheduleError::InvariantViolation)?;
            appends.push((handle, item.phase.appended_tokens()));
        }
        let prefer_decode_after = if both || dedicated {
            !self.prefer_decode
        } else {
            self.prefer_decode
        };
        let kv_plan = self
            .kv
            .prepare_append_batch(&appends)
            .map_err(ScheduleError::Kv)?;
        let batch = Batch {
            id,
            token_cost: items.iter().map(|item| item.token_cost).sum(),
            items,
            kv_plan,
            prefill_cap,
            shape,
            predicted_micros,
        };
        self.pending = Some(PendingBatch {
            generation: pending_generation,
            id,
            items: pending_items,
            prefer_decode_after,
            mixed_decode_steps_after,
            kv_plan_id: batch.kv_plan.id(),
        });
        self.next_batch_id = next_batch_id;
        self.next_pending_generation = next_pending_generation;
        Ok(Some(batch))
    }

    pub fn preflight_commit_batch(
        &self,
        id: BatchId,
        outcomes: &[BatchOutcome],
    ) -> Result<PreparedBatchCommit, ScheduleError> {
        let Some(batch) = self.pending.as_ref() else {
            return Err(ScheduleError::NoPendingBatch);
        };
        if batch.id != id {
            return Err(ScheduleError::BatchIdMismatch);
        }
        if outcomes.len() != batch.items.len() {
            return Err(ScheduleError::OutcomeCountMismatch);
        }
        let next_tick = self
            .scheduling_tick
            .checked_add(1)
            .ok_or(ScheduleError::SchedulingTickExhausted)?;
        let mut prepared_items = Vec::new();
        prepared_items
            .try_reserve_exact(batch.items.len())
            .map_err(|_| ScheduleError::MetadataAllocationFailed)?;
        let mut completed = Vec::new();
        completed
            .try_reserve_exact(batch.items.len())
            .map_err(|_| ScheduleError::MetadataAllocationFailed)?;
        let mut terminal_handles = Vec::new();
        terminal_handles
            .try_reserve_exact(batch.items.len())
            .map_err(|_| ScheduleError::MetadataAllocationFailed)?;
        let mut kept_tokens = Vec::new();
        kept_tokens
            .try_reserve_exact(batch.items.len())
            .map_err(|_| ScheduleError::MetadataAllocationFailed)?;
        let prepared = self
            .kv
            .prepared()
            .ok_or(ScheduleError::InvariantViolation)?;
        if prepared.id() != batch.kv_plan_id || prepared.appends().len() != batch.items.len() {
            return Err(ScheduleError::InvariantViolation);
        }
        let mut terminal_sequences = 0usize;
        let mut terminal_reserved_blocks = 0usize;
        for ((item, outcome), append) in batch.items.iter().zip(outcomes).zip(prepared.appends()) {
            if outcome.request_id != item.request_id || outcome.phase != item.phase {
                return Err(ScheduleError::OutcomeMembershipMismatch);
            }
            let emitted_valid = match item.phase {
                BatchPhase::Verify { draft_tokens } => {
                    item.expects_output && (1..=draft_tokens + 1).contains(&outcome.emitted_tokens)
                }
                BatchPhase::Prefill { .. } | BatchPhase::Decode => {
                    outcome.emitted_tokens == usize::from(item.expects_output)
                }
            };
            if !emitted_valid || (outcome.terminal && outcome.emitted_tokens == 0) {
                return Err(ScheduleError::InvalidOutcome(item.request_id));
            }
            let record_index = self.record_index(item.request_id)?;
            if prepared_items
                .iter()
                .any(|prepared: &PreparedCommitItem| prepared.record_index == record_index)
            {
                return Err(ScheduleError::InvariantViolation);
            }
            let record = &self.requests[record_index];
            if record.state.is_terminal()
                || record.handle != Some(append.handle())
                || append.additional_tokens() != item.phase.appended_tokens()
                || self.kv.snapshot(append.handle()).is_err()
            {
                return Err(ScheduleError::InvariantViolation);
            }
            let (prefilled_tokens, state) = match item.phase {
                BatchPhase::Prefill { tokens } => {
                    let prefilled_tokens = record
                        .prefilled_tokens
                        .checked_add(tokens)
                        .ok_or(ScheduleError::InvariantViolation)?;
                    let target = Self::prefill_target(record);
                    if prefilled_tokens > target {
                        return Err(ScheduleError::InvariantViolation);
                    }
                    let state = if prefilled_tokens == target {
                        RequestState::Decoding
                    } else {
                        RequestState::Prefilling
                    };
                    (prefilled_tokens, state)
                }
                BatchPhase::Decode | BatchPhase::Verify { .. } => {
                    if record.state != RequestState::Decoding {
                        return Err(ScheduleError::InvariantViolation);
                    }
                    (record.prefilled_tokens, RequestState::Decoding)
                }
            };
            // A verify item keeps the KV of the inputs whose outputs were
            // emitted: its bonus input and each accepted draft. That is one
            // KV token per emitted token, which keeps KV at prefilled plus
            // emitted minus one.
            if let BatchPhase::Verify { draft_tokens } = item.phase
                && outcome.emitted_tokens <= draft_tokens
            {
                kept_tokens.push((append.handle(), outcome.emitted_tokens));
            }
            let next_emitted = record
                .emitted_tokens
                .checked_add(outcome.emitted_tokens)
                .ok_or(ScheduleError::InvariantViolation)?;
            if next_emitted > record.spec.maximum_new_tokens {
                return Err(ScheduleError::InvalidOutcome(item.request_id));
            }
            let terminal = outcome.terminal || next_emitted == record.spec.maximum_new_tokens;
            let decode_capture = record.decode_capture.filter(|&tokens| {
                terminal
                    && !matches!(item.phase, BatchPhase::Prefill { .. })
                    && tokens == record.spec.prompt_tokens + next_emitted - 1
            });
            let capture = decode_capture.is_some()
                || (matches!(item.phase, BatchPhase::Prefill { .. })
                    && record.captures.contains(&prefilled_tokens));
            let anchor = item
                .anchor
                .filter(|anchor| record.captures.contains(anchor));
            prepared_items.push(PreparedCommitItem {
                record_index,
                prefilled_tokens,
                emitted_tokens: next_emitted,
                state: if terminal {
                    RequestState::Finished
                } else {
                    state
                },
                terminal,
                capture,
                capture_tokens: decode_capture.unwrap_or(prefilled_tokens),
                anchor,
                release_tail: record.shared_tail,
            });
            if terminal {
                completed.push(record.spec.id);
            }
            // A request that finishes at a capture point, or with an anchor
            // kept inside its last chunk, keeps its sequence until the
            // capture ends.
            if terminal {
                if !capture && anchor.is_none() {
                    terminal_handles.push(record.handle.ok_or(ScheduleError::InvariantViolation)?);
                }
                terminal_sequences = terminal_sequences
                    .checked_add(1)
                    .ok_or(ScheduleError::InvariantViolation)?;
                terminal_reserved_blocks = terminal_reserved_blocks
                    .checked_add(record.reserved_blocks)
                    .ok_or(ScheduleError::InvariantViolation)?;
            }
        }
        let active_sequences_after = self
            .active_sequences
            .checked_sub(terminal_sequences)
            .ok_or(ScheduleError::InvariantViolation)?;
        // Every committed item has replaced a shared tail, and a
        // terminal one shares nothing any more.
        let mut terminal_records = Vec::new();
        terminal_records
            .try_reserve_exact(prepared_items.len())
            .map_err(|_| ScheduleError::MetadataAllocationFailed)?;
        terminal_records.extend(
            prepared_items
                .iter()
                .filter(|item| item.terminal)
                .map(|item| item.record_index),
        );
        let released_tails = prepared_items
            .iter()
            .filter(|item| item.release_tail && !item.terminal)
            .count();
        let reserved_blocks_after = self
            .reserved_blocks
            .checked_sub(terminal_reserved_blocks + released_tails)
            .and_then(|reserved| reserved.checked_sub(self.shared_released(&terminal_records)))
            .ok_or(ScheduleError::InvariantViolation)?;
        let prefer_decode_after = batch.prefer_decode_after;
        let mixed_decode_steps_after = batch.mixed_decode_steps_after;
        let preflight_state = SchedulerPreflightState::capture(self, batch)?;
        let kv_commit = self
            .kv
            .preflight_commit_prepared_kept(batch.kv_plan_id, &kept_tokens, &terminal_handles)
            .map_err(ScheduleError::Kv)?;
        Ok(PreparedBatchCommit {
            scheduler_identity: Arc::clone(&self.identity),
            pending_generation: batch.generation,
            id,
            preflight_state,
            kv_commit,
            items: prepared_items,
            completed,
            next_tick,
            prefer_decode_after,
            mixed_decode_steps_after,
            active_sequences_after,
            reserved_blocks_after,
        })
    }

    /// Applies a token after rejecting every mismatch without consuming state.
    pub fn apply_prepared_commit(
        &mut self,
        prepared: PreparedBatchCommit,
    ) -> Result<CommitReport, ScheduleError> {
        if !Arc::ptr_eq(&self.identity, &prepared.scheduler_identity) {
            return Err(ScheduleError::PreparedCommitMismatch);
        }
        let Some(batch) = self.pending.as_ref() else {
            return Err(ScheduleError::NoPendingBatch);
        };
        if batch.id != prepared.id {
            return Err(ScheduleError::BatchIdMismatch);
        }
        if batch.generation != prepared.pending_generation
            || !prepared.preflight_state.matches(self)
        {
            return Err(ScheduleError::PreparedCommitMismatch);
        }
        self.captures
            .try_reserve(prepared.items.len())
            .map_err(|_| ScheduleError::MetadataAllocationFailed)?;
        self.kv
            .apply_prepared_commit(prepared.kv_commit)
            .map_err(ScheduleError::Kv)?;

        self.pending = None;
        self.scheduling_tick = prepared.next_tick;
        self.count_recomputed_prefill(&prepared.items);
        for item in &prepared.items {
            let record = &mut self.requests[item.record_index];
            // A prompt-ending item captures either its end or the anchor it
            // kept inside its rows, never both: the prompt end is a capture
            // point only when it is its own anchor.
            let captured = match item.anchor {
                Some(anchor) => Some((anchor, true)),
                None => item.capture.then_some((item.capture_tokens, false)),
            };
            if let Some((tokens, interior)) = captured
                && let (Some(prefix), Some(handle)) = (&record.prefix, record.handle)
                && let (Some(key), Some(chain)) = (
                    prefix.chain.key_at(tokens),
                    prefix.chain.chain(tokens / prefix.chain.block_tokens()),
                )
            {
                // The capture list was reserved for every item of the batch.
                self.captures.push(PrefixCaptureV1 {
                    request_id: record.spec.id,
                    swap: false,
                    private: false,
                    interior,
                    tokens,
                    key,
                    chain,
                    handle,
                    deferred: item.terminal,
                });
            }
            record.last_scheduled = prepared.next_tick;
            record.prefilled_tokens = item.prefilled_tokens;
            record.emitted_tokens = item.emitted_tokens;
            record.state = item.state;
            if item.terminal {
                record.handle = None;
                record.reserved_blocks = 0;
                record.reserved_tokens = 0;
            }
        }
        for item in &prepared.items {
            let record = &mut self.requests[item.record_index];
            if item.release_tail && !item.terminal {
                record.reserved_blocks -= 1;
            }
            record.shared_tail = false;
            if item.terminal {
                self.release_shared(item.record_index);
            }
        }
        self.active_sequences = prepared.active_sequences_after;
        self.reserved_blocks = prepared.reserved_blocks_after;
        self.prefer_decode = prepared.prefer_decode_after;
        self.mixed_decode_steps = prepared.mixed_decode_steps_after;
        Ok(CommitReport {
            completed: prepared.completed,
        })
    }

    /// Counts the committed prompt rows whose full blocks an earlier
    /// request's committed prefill already computed, then records the blocks
    /// this commit completed. Every item is compared before any is recorded,
    /// so requests prefilled in one step do not count each other. Restored
    /// rows are never prefilled, and a row is compared once per request. Each
    /// item probes and records only the blocks its own rows touch, so the
    /// cost of a prompt is linear in its length.
    fn count_recomputed_prefill(&mut self, items: &[PreparedCommitItem]) {
        let mut recomputed = 0;
        for item in items {
            let record = &self.requests[item.record_index];
            let Some(prefix) = record.prefix.as_ref().filter(|prefix| prefix.lookup) else {
                continue;
            };
            let block_tokens = prefix.chain.block_tokens();
            let end = item
                .prefilled_tokens
                .min(record.spec.prompt_tokens.saturating_sub(1))
                .min(prefix.chain.full_blocks() * block_tokens);
            let mut row = record.prefilled_tokens.max(record.compared_tokens);
            while row < end {
                let block = row / block_tokens + 1;
                let block_end = (block * block_tokens).min(end);
                if self.kv.produced(&prefix.chain, block) {
                    recomputed += block_end - row;
                }
                row = block_end;
            }
        }
        for item in items {
            let record = &mut self.requests[item.record_index];
            if item.prefilled_tokens <= record.prefilled_tokens {
                continue;
            }
            record.compared_tokens = record.compared_tokens.max(item.prefilled_tokens);
            if let Some(prefix) = &record.prefix {
                let block_tokens = prefix.chain.block_tokens();
                self.kv.record_produced(
                    &prefix.chain,
                    record.prefilled_tokens / block_tokens + 1,
                    item.prefilled_tokens / block_tokens,
                );
            }
        }
        if recomputed > 0 {
            self.kv.metrics().prefix_cache_recomputed_seen(recomputed);
        }
    }

    pub fn commit_batch(
        &mut self,
        id: BatchId,
        outcomes: &[BatchOutcome],
    ) -> Result<CommitReport, ScheduleError> {
        let prepared = self.preflight_commit_batch(id, outcomes)?;
        self.apply_prepared_commit(prepared)
    }

    pub fn rollback_batch(&mut self, id: BatchId) -> Result<(), ScheduleError> {
        let Some(batch) = self.pending.as_ref() else {
            return Err(ScheduleError::NoPendingBatch);
        };
        if batch.id != id {
            return Err(ScheduleError::BatchIdMismatch);
        }
        self.kv
            .rollback_prepared(batch.kv_plan_id)
            .map_err(ScheduleError::Kv)?;
        self.pending = None;
        Ok(())
    }

    /// Requests resumed from an anchor whose bundle state the host must
    /// restore before they can be scheduled.
    #[must_use]
    pub fn pending_prefix_restores(&self) -> Vec<PrefixRestoreV1> {
        self.requests
            .iter()
            .filter(|record| {
                self.resumes
                    .get(&record.spec.id)
                    .is_none_or(|plan| plan.destination.is_none())
            })
            .filter_map(|record| {
                record.restore.as_ref().map(|pages| PrefixRestoreV1 {
                    request_id: record.spec.id,
                    tokens: record.prefilled_tokens,
                    pages: pages.clone(),
                    resumes_decoding: Self::restores_decode_boundary(
                        record,
                        record.prefilled_tokens,
                    ),
                })
            })
            .collect()
    }

    /// Records that the bundle restored a resumed request, releasing its pins.
    /// An exact restore of its own output boundary resumes in `Decoding`;
    /// other hits retain their prefill progress and replay the remaining inputs.
    ///
    /// The request's full prefix pages are shared and never written again, so
    /// from here on they are reserved once for every live request that shares
    /// them, and the request reserves only the pages it may still add, plus
    /// the page for its copy of a shared tail until its first commit.
    pub fn complete_prefix_restore(&mut self, id: RequestId) -> Result<(), ScheduleError> {
        self.ensure_no_pending()?;
        let index = self.record_index(id)?;
        let record = &self.requests[index];
        let (Some(handle), Some(_)) = (record.handle, record.restore.as_ref()) else {
            return Err(ScheduleError::InvariantViolation);
        };
        let block_tokens = self.kv.geometry().block_tokens();
        let tokens = record.prefilled_tokens;
        let mut table = self
            .kv
            .snapshot(handle)
            .map_err(ScheduleError::Kv)?
            .into_blocks();
        // A full last page the request rewrites is copied, not shared.
        let full = if record.shared_tail {
            tokens.div_ceil(block_tokens) - 1
        } else {
            tokens / block_tokens
        };
        table.truncate(full);
        let private = self
            .kv
            .geometry()
            .blocks_for(record.reserved_tokens)
            .map_err(ScheduleError::Kv)?
            .checked_sub(full)
            .ok_or(ScheduleError::InvariantViolation)?
            + usize::from(record.shared_tail);
        let added = table
            .iter()
            .filter(|block| !self.shared_blocks.contains_key(block))
            .count();
        // A restore only ever lowers the reservation: pages it newly reserves
        // once were already part of the request's cold reservation.
        let reserved = self.reserved_blocks - record.reserved_blocks + private + added;
        if reserved > self.reserved_blocks {
            return Err(ScheduleError::InvariantViolation);
        }
        if record.own_restore {
            self.kv
                .finish_owned_restore(handle)
                .map_err(ScheduleError::Kv)?;
        } else {
            self.kv.finish_restore(handle).map_err(ScheduleError::Kv)?;
        }
        for block in &table {
            *self.shared_blocks.entry(*block).or_default() += 1;
        }
        let record = &mut self.requests[index];
        record.restore = None;
        record.reserved_blocks = private;
        record.shared_blocks = table;
        if Self::restores_decode_boundary(record, tokens) {
            // Normalize to the uninterrupted counters: KV contains the prompt
            // and every emitted token except the pending decode input.
            record.prefilled_tokens = record.spec.prompt_tokens;
            record.replayed_tokens = 0;
            record.state = RequestState::Decoding;
            self.kv.metrics().prefix_restore_resumed_decoding();
        }
        self.reserved_blocks = reserved;
        // The sequence now holds the pages; the snapshot has served.
        self.drop_own_anchor(index)
    }

    /// Gives a resumed request whose restore failed a fresh sequence, so it
    /// prefills its whole prompt.
    pub fn abandon_prefix_restore(&mut self, id: RequestId) -> Result<(), ScheduleError> {
        self.ensure_no_pending()?;
        let index = self.record_index(id)?;
        let (Some(handle), Some(pages)) = (
            self.requests[index].handle,
            self.requests[index].restore.take(),
        ) else {
            return Err(ScheduleError::InvariantViolation);
        };
        self.kv.release(handle).map_err(ScheduleError::Kv)?;
        if self.config.prefix_state_pages.is_some() && !self.requests[index].own_restore {
            self.kv.metrics().prefix_restore_failed();
        }
        let fresh = self
            .kv
            .create_sequence(self.requests[index].spec.maximum_kv_tokens)
            .map_err(ScheduleError::Kv)?;
        let record = &mut self.requests[index];
        // A cold sequence reserves its worst case alone: no pins, and no
        // page for a copy of a shared tail.
        let released = pages.len() + usize::from(record.shared_tail);
        record.handle = Some(fresh);
        record.reserved_blocks -= released;
        record.shared_tail = false;
        record.prefilled_tokens = 0;
        if record.preemptions == 0 {
            record.cached_tokens = 0;
        }
        self.reserved_blocks -= released;
        if let Some(prefix) = &record.prefix
            && record.replayed_tokens == 0
        {
            let prompt = record.spec.prompt_tokens;
            let end = prompt / self.config.prefill_granule * self.config.prefill_granule;
            if end >= prefix.chain.block_tokens() && !record.captures.contains(&end) {
                record.captures.push(end);
                record.captures.sort_unstable();
            }
            record.end_anchor =
                (self.config.anchor_items > 0 && end < prompt && record.captures.contains(&end))
                    .then_some(end);
        }
        if self.requests[index].guaranteed_prefill_chunks.is_some() {
            self.requests[index].captures = self.capture_points(index, 0);
            self.requests[index].end_anchor = None;
        }
        self.drop_own_anchor(index)
    }

    /// Grows every decoding request's reservation to cover its next step,
    /// preempting requests when the pool cannot, and returns the preempted
    /// requests in the order they were preempted.
    ///
    /// Only [`AdmissionPolicy::Optimistic`] preempts; otherwise this returns
    /// nothing. A step appends one token plus up to `lookahead` drafts.
    /// Decoding requests grow in arrival order. When one cannot, the
    /// latest-arrived live request that is not pinned is preempted, which may
    /// be the request itself. A request is pinned when its spec asks for
    /// reserved admission or it was preempted the configured maximum number of
    /// times; a request awaiting a prefix restore is never preempted.
    ///
    /// A preempted request is queued again at its arrival position. Its
    /// prefill target becomes its prompt followed by every token it emitted,
    /// whose last row samples its next token. With prefix caching,
    /// `resume_prefix(id, tokens)` gives the chain over those first `tokens`
    /// tokens, which the request looks up when it is readmitted, and the
    /// preemption makes a due capture of its committed KV under a private key
    /// the request owns: it restores that anchor when it is readmitted, and no
    /// other request can. The capture holds the request's sequence until it ends, so the caller
    /// takes and runs the captures, and then retires the request's model
    /// state, before it admits or forms the next batch.
    pub fn preempt_for_next_step<F>(
        &mut self,
        lookahead: usize,
        mut resume_prefix: F,
    ) -> Result<Vec<RequestId>, ScheduleError>
    where
        F: FnMut(RequestId, usize) -> Option<PrefixRequestV1>,
    {
        self.ensure_no_pending()?;
        let AdmissionPolicy::Optimistic(optimistic) = self.config.admission else {
            return Ok(Vec::new());
        };
        let mut preempted = Vec::new();
        if let Some(every) = optimistic.forced_preemption_interval {
            for index in 0..self.requests.len() {
                let record = &self.requests[index];
                if record.state == RequestState::Decoding
                    && !self.pinned(record)
                    && record.emitted_tokens >= (record.preemptions + 1).saturating_mul(every)
                {
                    preempted.push(self.preempt(index, &mut resume_prefix)?);
                }
            }
        }
        let mut decoding = Vec::new();
        decoding
            .try_reserve_exact(self.active_sequences)
            .map_err(|_| ScheduleError::MetadataAllocationFailed)?;
        decoding.extend(
            (0..self.requests.len())
                .filter(|index| self.requests[*index].state == RequestState::Decoding),
        );
        decoding.sort_by_key(|index| self.requests[*index].arrival);
        let geometry = self.kv.geometry();
        // Reserve every mandatory decode row before giving spare pages to
        // drafts, so an earlier request's drafts cannot preempt a later one.
        for optional in [false, true] {
            for &index in &decoding {
                let record = &self.requests[index];
                if record.state != RequestState::Decoding {
                    continue;
                }
                let needed = Self::committed_tokens(record)
                    .saturating_add(1)
                    .saturating_add(if optional { lookahead } else { 0 })
                    .min(record.spec.maximum_kv_tokens);
                if needed <= record.reserved_tokens {
                    continue;
                }
                let grown = needed
                    .div_ceil(geometry.block_tokens())
                    .saturating_mul(geometry.block_tokens())
                    .min(record.spec.maximum_kv_tokens);
                let growth = geometry.blocks_for(grown).map_err(ScheduleError::Kv)?
                    - geometry
                        .blocks_for(record.reserved_tokens)
                        .map_err(ScheduleError::Kv)?;
                if optional && growth > geometry.block_capacity() - self.reserved_blocks {
                    continue;
                }
                let mut grows = true;
                while geometry.block_capacity() - self.reserved_blocks < growth {
                    // The growing request itself is a candidate: needing growth
                    // means it is not pinned.
                    let victim = self
                        .latest_preemptible()
                        .ok_or(ScheduleError::InvariantViolation)?;
                    preempted.push(self.preempt(victim, &mut resume_prefix)?);
                    if victim == index {
                        grows = false;
                        break;
                    }
                }
                if grows {
                    let record = &mut self.requests[index];
                    record.reserved_tokens = grown;
                    record.reserved_blocks += growth;
                    self.reserved_blocks += growth;
                }
            }
        }
        Ok(preempted)
    }

    /// Times a request was preempted.
    pub fn preemptions(&self, id: RequestId) -> Result<usize, ScheduleError> {
        Ok(self.record(id)?.preemptions)
    }

    /// Pages reserved for live requests. Once due captures have ended, every
    /// page a live sequence holds is within it.
    #[must_use]
    pub const fn reserved_pages(&self) -> usize {
        self.reserved_blocks
    }

    /// The granule chunks that stop short of the prefill target end on.
    #[must_use]
    pub const fn prefill_granule(&self) -> usize {
        self.config.prefill_granule
    }

    /// The live request a preemption takes first: the latest arrival that is
    /// neither pinned nor awaiting a restore.
    fn latest_preemptible(&self) -> Option<usize> {
        self.requests
            .iter()
            .enumerate()
            .filter(|(_, record)| {
                record.handle.is_some()
                    && !record.state.is_terminal()
                    && record.restore.is_none()
                    && !self.pinned(record)
            })
            .max_by_key(|(_, record)| record.arrival)
            .map(|(index, _)| index)
    }

    /// Returns a live request to the queue to recompute its tokens later.
    fn preempt<F>(
        &mut self,
        index: usize,
        resume_prefix: &mut F,
    ) -> Result<RequestId, ScheduleError>
    where
        F: FnMut(RequestId, usize) -> Option<PrefixRequestV1>,
    {
        let record = &self.requests[index];
        let (Some(handle), None) = (record.handle, record.restore.as_ref()) else {
            return Err(ScheduleError::InvariantViolation);
        };
        let tokens = self
            .kv
            .snapshot(handle)
            .map_err(ScheduleError::Kv)?
            .tokens();
        self.captures
            .try_reserve(1)
            .map_err(|_| ScheduleError::MetadataAllocationFailed)?;
        // A cold admission can leave an older private boundary behind. Its
        // copy leases remain valid, but do not protect this new boundary.
        self.drop_own_anchor(index)?;
        let released = self.shared_released(&[index]);
        self.release_shared(index);
        let block_tokens = self.kv.geometry().block_tokens();
        let caching = self.config.prefix_state_pages.is_some();
        let swap = self
            .offload
            .is_some_and(|config| config.enabled && tokens > 0 && tokens >= config.minimum_tokens);
        let record = &mut self.requests[index];
        self.reserved_blocks -= record.reserved_blocks + released;
        self.active_sequences -= 1;
        record.handle = None;
        record.reserved_blocks = 0;
        record.reserved_tokens = 0;
        record.shared_tail = false;
        record.state = RequestState::Queued;
        record.prefilled_tokens = 0;
        record.replayed_tokens = record.emitted_tokens;
        record.preemptions += 1;
        record.captures.clear();
        let target = Self::prefill_target(record);
        if (caching || swap) && record.replayed_tokens > 0 {
            record.prefix = resume_prefix(record.spec.id, target).filter(|prefix| {
                prefix.chain.tokens().len() == target && prefix.chain.block_tokens() == block_tokens
            });
        }
        if !caching && let Some(prefix) = &mut record.prefix {
            prefix.lookup = false;
        }
        // The committed KV and model state become a private anchor under a
        // key this preemption owns, so the recompute restores its own state
        // instead of prefilling it, and no other request restores it: one
        // with the same tokens may hold other KV bits or state. This holds
        // below one page too: a prefill of those tokens would change their
        // logits, while the restore resumes the decode bit for bit.
        let id = record.spec.id;
        // A request without a prefix chain (startup warmup, which runs
        // without prefix caching) is never restored from an anchor, so only
        // a swap without prefix caching captures it.
        let capture = tokens > 0
            && match &record.prefix {
                Some(prefix) => (caching || swap) && !prefix.chain.splits_media(tokens),
                None => swap && !caching,
            };
        if capture {
            let mut key = [0u8; 32];
            key[..8].copy_from_slice(b"KVO-own1");
            key[8..16].copy_from_slice(&record.arrival.to_le_bytes());
            key[16..24].copy_from_slice(&(record.preemptions as u64).to_le_bytes());
            key[24..].copy_from_slice(&id.get().to_le_bytes());
            let key = PrefixCacheKeyV1::new(key);
            record.own_anchor = Some(key);
            self.captures.push(PrefixCaptureV1 {
                request_id: id,
                tokens,
                key,
                chain: key,
                handle,
                deferred: true,
                swap,
                private: true,
                interior: false,
            });
        } else {
            self.kv.release(handle).map_err(ScheduleError::Kv)?;
        }
        Ok(id)
    }

    /// Prompt tokens a request resumed from the prefix cache.
    pub fn cached_tokens(&self, id: RequestId) -> Result<usize, ScheduleError> {
        Ok(self.record(id)?.cached_tokens)
    }

    /// The chunk grid and complete prefill target promised for this request.
    /// The host passes this only to an executor that opted into the policy.
    pub fn prefill_boundary_contract(
        &self,
        id: RequestId,
    ) -> Result<Option<(usize, usize)>, ScheduleError> {
        let record = self.record(id)?;
        Ok(record
            .guaranteed_prefill_chunks
            .map(|chunk| (chunk, Self::prefill_target(record))))
    }

    /// Supplies a complete accepted text history for a terminal decode capture.
    /// The final sampled token has no KV yet and must be excluded. Only the
    /// opt-in profile publishes these public response-end anchors.
    pub fn prepare_decode_capture(
        &mut self,
        id: RequestId,
        prefix: PrefixRequestV1,
    ) -> Result<(), ScheduleError> {
        let index = self.record_index(id)?;
        let record = &mut self.requests[index];
        if self.config.prefix_state_pages.is_none() || record.guaranteed_prefill_chunks.is_none() {
            return Ok(());
        }
        let Some(prior) = &record.prefix else {
            return Ok(());
        };
        if self.pending.as_ref().is_none_or(|batch| {
            !batch.items.iter().any(|item| {
                item.request_id == id && !matches!(item.phase, BatchPhase::Prefill { .. })
            })
        }) {
            return Err(ScheduleError::InvalidOutcome(id));
        }
        let tokens = prefix.chain.tokens().len();
        let prompt = record.spec.prompt_tokens;
        if tokens < prompt || prefix.chain.key_at(prompt) != prior.chain.key_at(prompt) {
            return Err(ScheduleError::InvalidOutcome(id));
        }
        record.decode_capture = Some(tokens);
        record.prefix = Some(prefix);
        Ok(())
    }

    /// Takes the anchors the last commits made due, in commit order.
    pub fn take_prefix_captures(&mut self) -> Vec<PrefixCaptureV1> {
        std::mem::take(&mut self.captures)
    }

    /// Publishes a due capture's anchor and returns the snapshot pages the
    /// bundle must fill, or `None` when there is nothing to capture.
    pub fn begin_prefix_capture(
        &mut self,
        capture: &PrefixCaptureV1,
    ) -> Result<Option<Vec<BlockId>>, ScheduleError> {
        self.ensure_no_pending()?;
        if capture.private {
            let Some(state_pages) = self
                .offload
                .map(|config| config.snapshot_pages)
                .or(self.config.prefix_state_pages)
            else {
                return Ok(None);
            };
            return self
                .kv
                .insert_private_anchor(capture.key, capture.handle, state_pages)
                .map_err(ScheduleError::Kv);
        }
        let Some(state_pages) = self.config.prefix_state_pages else {
            return Ok(None);
        };
        if capture.interior {
            return self
                .kv
                .insert_prefix_anchor(
                    capture.key,
                    capture.chain,
                    capture.handle,
                    capture.tokens,
                    state_pages,
                )
                .map_err(ScheduleError::Kv);
        }
        self.kv
            .insert_anchor(capture.key, capture.chain, capture.handle, state_pages)
            .map_err(ScheduleError::Kv)
    }

    /// Drops the anchor of a capture the bundle failed to write.
    pub fn abandon_prefix_capture(
        &mut self,
        capture: &PrefixCaptureV1,
    ) -> Result<(), ScheduleError> {
        self.kv
            .remove_anchor(capture.key)
            .map_err(ScheduleError::Kv)?;
        if let Ok(index) = self.record_index(capture.request_id)
            && self.requests[index].own_anchor == Some(capture.key)
        {
            self.requests[index].own_anchor = None;
            self.requests[index].own_anchor_transfer = None;
        }
        if !capture.private || self.config.prefix_state_pages.is_some() {
            self.kv.metrics().prefix_capture_failed();
        }
        Ok(())
    }

    /// Ends a capture, releasing the sequence of a request that finished at it.
    pub fn end_prefix_capture(&mut self, capture: &PrefixCaptureV1) -> Result<(), ScheduleError> {
        self.ensure_no_pending()?;
        let owned = self.record(capture.request_id).is_ok_and(|record| {
            !record.state.is_terminal() && record.own_anchor == Some(capture.key)
        });
        if capture.private && !owned && !self.resumes.contains_key(&capture.request_id) {
            self.kv
                .remove_anchor(capture.key)
                .map_err(ScheduleError::Kv)?;
        }
        if capture.deferred {
            self.kv.release(capture.handle).map_err(ScheduleError::Kv)?;
        }
        Ok(())
    }

    /// How far the shared reservation falls when these requests release
    /// their full prefix pages.
    fn shared_released(&self, records: &[usize]) -> usize {
        let mut released = BTreeMap::<BlockId, usize>::new();
        for index in records {
            for block in &self.requests[*index].shared_blocks {
                *released.entry(*block).or_default() += 1;
            }
        }
        released
            .iter()
            .filter(|(block, count)| self.shared_blocks.get(block) == Some(count))
            .count()
    }

    /// Drops a request's full prefix pages from the shared reservation.
    fn release_shared(&mut self, index: usize) {
        for block in std::mem::take(&mut self.requests[index].shared_blocks) {
            if let Some(sharers) = self.shared_blocks.get_mut(&block) {
                *sharers -= 1;
                if *sharers == 0 {
                    self.shared_blocks.remove(&block);
                }
            }
        }
    }

    fn ensure_no_pending(&self) -> Result<(), ScheduleError> {
        if self.pending.is_some() {
            return Err(ScheduleError::PendingBatch);
        }
        Ok(())
    }

    pub fn cancel(&mut self, id: RequestId) -> Result<(), ScheduleError> {
        self.rollback_pending_for_mutation()?;
        self.transition_terminal(id, RequestState::Cancelled)
    }

    pub fn finish(&mut self, id: RequestId) -> Result<(), ScheduleError> {
        self.rollback_pending_for_mutation()?;
        self.transition_terminal(id, RequestState::Finished)
    }

    pub fn fail(&mut self, id: RequestId) -> Result<(), ScheduleError> {
        self.rollback_pending_for_mutation()?;
        self.transition_terminal(id, RequestState::Failed)
    }

    pub fn request_state(&self, id: RequestId) -> Result<RequestState, ScheduleError> {
        Ok(self.record(id)?.state)
    }

    pub fn kv_tokens(&self, id: RequestId) -> Result<usize, ScheduleError> {
        let record = self.record(id)?;
        match record.handle {
            Some(handle) => self
                .kv
                .snapshot(handle)
                .map(|snapshot| snapshot.tokens())
                .map_err(ScheduleError::Kv),
            None => Ok(0),
        }
    }

    pub fn emitted_tokens(&self, id: RequestId) -> Result<usize, ScheduleError> {
        Ok(self.record(id)?.emitted_tokens)
    }

    pub fn kv_blocks(&self, id: RequestId) -> Result<Vec<BlockId>, ScheduleError> {
        let record = self.record(id)?;
        match record.handle {
            Some(handle) => self
                .kv
                .snapshot(handle)
                .map(|snapshot| snapshot.into_blocks())
                .map_err(ScheduleError::Kv),
            None => Ok(Vec::new()),
        }
    }

    pub fn retire_terminal(&mut self, id: RequestId) -> Result<(), ScheduleError> {
        if self
            .pending
            .as_ref()
            .is_some_and(|batch| batch.items.iter().any(|item| item.request_id == id))
        {
            return Err(ScheduleError::RequestPendingBatch(id));
        }
        let index = self.record_index(id)?;
        if !self.requests[index].state.is_terminal() {
            return Err(ScheduleError::RequestNotTerminal(id));
        }
        self.requests.remove(index);
        Ok(())
    }

    pub fn reap_terminal(&mut self, maximum: usize) -> Result<Vec<RequestId>, ScheduleError> {
        let target = maximum.min(self.terminal_count());
        let mut retired = Vec::new();
        retired
            .try_reserve_exact(target)
            .map_err(|_| ScheduleError::MetadataAllocationFailed)?;
        let mut index = 0usize;
        while index < self.requests.len() && retired.len() < target {
            if self.requests[index].state.is_terminal() {
                retired.push(self.requests[index].spec.id);
                self.requests.remove(index);
            } else {
                index += 1;
            }
        }
        Ok(retired)
    }

    #[must_use]
    pub fn kv_stats(&self) -> KvStats {
        self.kv.stats()
    }

    #[must_use]
    pub const fn capacity_profile(&self) -> SchedulerCapacityProfile {
        self.config.capacity_profile
    }

    #[must_use]
    pub fn kv_metrics(&self) -> KvMetrics {
        self.kv.metrics()
    }

    #[must_use]
    pub fn terminal_count(&self) -> usize {
        self.requests
            .iter()
            .filter(|record| record.state.is_terminal())
            .count()
    }

    pub fn audit(&self) -> Result<(), ScheduleError> {
        self.kv.audit().map_err(ScheduleError::Kv)?;
        let active = self
            .requests
            .iter()
            .filter(|record| record.handle.is_some())
            .count();
        let reserved: usize = self
            .requests
            .iter()
            .map(|record| record.reserved_blocks)
            .sum();
        let mut shared_blocks = BTreeMap::<BlockId, usize>::new();
        for record in &self.requests {
            if (!record.shared_blocks.is_empty() && record.restore.is_some())
                || ((!record.shared_blocks.is_empty() || record.shared_tail)
                    && record.handle.is_none())
            {
                return Err(ScheduleError::InvariantViolation);
            }
            for block in &record.shared_blocks {
                *shared_blocks.entry(*block).or_default() += 1;
            }
        }
        if shared_blocks != self.shared_blocks {
            return Err(ScheduleError::InvariantViolation);
        }
        let geometry = self.kv.geometry();
        for record in &self.requests {
            if record.replayed_tokens > record.emitted_tokens
                || record.reserved_tokens > record.spec.maximum_kv_tokens
            {
                return Err(ScheduleError::InvariantViolation);
            }
            let Some(handle) = record.handle else {
                if record.reserved_tokens != 0
                    || (record.state == RequestState::Queued && record.prefilled_tokens != 0)
                {
                    return Err(ScheduleError::InvariantViolation);
                }
                continue;
            };
            let tokens = self
                .kv
                .snapshot(handle)
                .map_err(ScheduleError::Kv)?
                .tokens();
            // The reservation covers the committed KV, in pages net of the
            // shared prefix pages reserved once, plus snapshot pins until the
            // restore and the copy of a shared tail until the first commit.
            let pages = geometry
                .blocks_for(record.reserved_tokens)
                .map_err(ScheduleError::Kv)?;
            let expected_pages = match &record.restore {
                Some(pins) => pages + pins.len(),
                None => pages
                    .checked_sub(record.shared_blocks.len())
                    .ok_or(ScheduleError::InvariantViolation)?,
            } + usize::from(record.shared_tail);
            if tokens != Self::committed_tokens(record)
                || tokens > record.reserved_tokens
                || record.reserved_blocks != expected_pages
                || (self.pinned(record) && record.reserved_tokens != record.spec.maximum_kv_tokens)
            {
                return Err(ScheduleError::InvariantViolation);
            }
        }
        if active != self.active_sequences
            || reserved + self.shared_blocks.len() != self.reserved_blocks
            || self.reserved_blocks > self.kv.geometry().block_capacity()
            || self.requests.iter().any(|record| {
                let live = matches!(
                    record.state,
                    RequestState::Admitted | RequestState::Prefilling | RequestState::Decoding
                );
                live != record.handle.is_some()
                    || live != (record.reserved_blocks > 0)
                    || record.prefilled_tokens > Self::prefill_target(record)
                    || record.emitted_tokens > record.spec.maximum_new_tokens
            })
        {
            return Err(ScheduleError::InvariantViolation);
        }
        if let Some(batch) = &self.pending {
            let prepared = self
                .kv
                .prepared()
                .ok_or(ScheduleError::InvariantViolation)?;
            if prepared.id() != batch.kv_plan_id || prepared.appends().len() != batch.items.len() {
                return Err(ScheduleError::InvariantViolation);
            }
            let mut seen = Vec::new();
            seen.try_reserve_exact(self.requests.len())
                .map_err(|_| ScheduleError::MetadataAllocationFailed)?;
            seen.resize(self.requests.len(), false);
            for (item, append) in batch.items.iter().zip(prepared.appends()) {
                let index = self.record_index(item.request_id)?;
                if seen[index]
                    || self.requests[index].handle != Some(append.handle())
                    || append.additional_tokens() != item.phase.appended_tokens()
                {
                    return Err(ScheduleError::InvariantViolation);
                }
                seen[index] = true;
            }
        } else if self.kv.prepared_id().is_some() {
            return Err(ScheduleError::InvariantViolation);
        }
        Ok(())
    }

    fn candidates(&self, decode: bool) -> Result<Vec<usize>, ScheduleError> {
        let mut candidates = Vec::new();
        candidates
            .try_reserve_exact(self.active_sequences)
            .map_err(|_| ScheduleError::MetadataAllocationFailed)?;
        for (index, record) in self.requests.iter().enumerate() {
            let matches = if decode {
                record.state == RequestState::Decoding
            } else {
                matches!(
                    record.state,
                    RequestState::Admitted | RequestState::Prefilling
                ) && record.restore.is_none()
            };
            if matches {
                candidates.push(index);
            }
        }
        Ok(candidates)
    }

    /// The pending token follows this request's committed output boundary.
    const fn restores_decode_boundary(record: &RequestRecord, tokens: usize) -> bool {
        record.emitted_tokens > 0 && tokens == record.spec.prompt_tokens + record.emitted_tokens - 1
    }

    /// The prompt-end anchor the request's next chunk can run past: the
    /// next capture point is that anchor, and the chunk limit reaches the
    /// prefill target.
    fn spannable_anchor(
        record: &RequestRecord,
        chunk_limit: usize,
        maximum_scored_rows: usize,
    ) -> Option<usize> {
        if record.guaranteed_prefill_chunks.is_some() {
            return None;
        }
        let start = record.prefilled_tokens;
        let anchor = record.end_anchor.filter(|&anchor| anchor > start)?;
        let next = record
            .captures
            .iter()
            .copied()
            .find(|&boundary| boundary > start);
        (next == Some(anchor)
            && Self::prefill_target(record) - start
                <= Self::request_chunk_limit(record, chunk_limit, maximum_scored_rows))
        .then_some(anchor)
    }

    /// Tokens to prefill: the prompt and already emitted tokens to replay.
    const fn prefill_target(record: &RequestRecord) -> usize {
        record.spec.prompt_tokens + record.replayed_tokens
    }

    /// Committed KV tokens: every prefilled token and every emitted token
    /// after the replayed ones except the last, the next decode input.
    /// Whether a hit restoring `tokens` holds a page for the copy of the
    /// shared page its first write reaches: a partial tail, or a full last
    /// page when the request rewrites rows of its last page.
    const fn copies_tail(block_tokens: usize, rewrites: bool, tokens: usize) -> bool {
        !tokens.is_multiple_of(block_tokens) || (rewrites && tokens > 0)
    }

    const fn committed_tokens(record: &RequestRecord) -> usize {
        record.prefilled_tokens + (record.emitted_tokens - record.replayed_tokens).saturating_sub(1)
    }

    /// Whether the request reserves its worst case and is never preempted.
    const fn pinned(&self, record: &RequestRecord) -> bool {
        match self.config.admission {
            AdmissionPolicy::Reserved => true,
            AdmissionPolicy::Optimistic(optimistic) => {
                record.spec.reserved_admission
                    || record.spec.media_pages > 0
                    || record.preemptions >= optimistic.maximum_preemptions
            }
        }
    }

    /// Tokens the next chunk may take: up to the prefill target, or to the
    /// next capture point, where a chunk must end (except a prompt-end
    /// anchor the chunk runs past, `spans`).
    fn chunk_remaining(record: &RequestRecord, spans: bool) -> usize {
        let remaining = Self::prefill_target(record) - record.prefilled_tokens;
        record
            .captures
            .iter()
            .filter(|boundary| !(spans && record.end_anchor == Some(**boundary)))
            .find(|boundary| **boundary > record.prefilled_tokens)
            .map_or(remaining, |boundary| {
                remaining.min(boundary - record.prefilled_tokens)
            })
    }

    /// The prompt boundaries a newly admitted request captures anchors at.
    ///
    /// The prompt end (with a prefill granule, its last multiple of it) is
    /// captured when it is at least one page long. When that is the prompt
    /// end itself it serves longer prompts only: a request prefills at least
    /// its last prompt token, so the same prompt again looks up anchors
    /// before its end. For such repeats the start of the final prefill chunk
    /// (with a granule, its last multiple of it) is captured too. This uses
    /// the configured chunk even when a lone request can take more tokens,
    /// so repeats retain their anchors and bounded final replay. The
    /// end of the leading full blocks an earlier prompt already produced
    /// (with a granule, their last multiple of it) is captured when it lies
    /// past the request's hit, outside media, and has no anchor yet, so that
    /// the next request sharing them hits there. A request that resumes
    /// within one chunk of its end captures no such blocks: the anchor it
    /// resumed from serves the prompts that share them, and a capture would
    /// split the final chunk, which would then differ from the first send's.
    fn capture_points(&mut self, index: usize, hit: usize) -> Vec<usize> {
        let Some(chain) = self.requests[index]
            .prefix
            .as_ref()
            .map(|prefix| Arc::clone(&prefix.chain))
        else {
            return Vec::new();
        };
        if let Some(chunk) = self.requests[index].guaranteed_prefill_chunks {
            let prompt = self.requests[index].spec.prompt_tokens;
            let mut points = Vec::new();
            let mut next = hit / chunk * chunk;
            while let Some(end) = next.checked_add(chunk).filter(|&end| end < prompt) {
                if !chain.splits_media(end) {
                    points.push(end);
                }
                next = end;
            }
            if prompt > hit && !chain.splits_media(prompt) {
                points.push(prompt);
            }
            return points;
        }
        let block_tokens = self.kv.geometry().block_tokens();
        let prompt = self.requests[index].spec.prompt_tokens;
        let chunk = Self::request_chunk_limit(
            &self.requests[index],
            self.config.prefill_chunk_limit,
            self.config.maximum_scored_rows,
        );
        // A readmitted request already recorded its blocks, so the registry
        // would only find its own first admission.
        let seen = if self.requests[index].preemptions == 0 {
            self.kv.observe_prefix(&chain)
        } else {
            0
        };
        let mut points = self.requests[index]
            .captures
            .iter()
            .copied()
            .filter(|tokens| *tokens > hit)
            .collect::<Vec<_>>();
        let granule = self.config.prefill_granule;
        let absent = |tokens: usize| {
            tokens >= block_tokens
                && !chain.splits_media(tokens)
                && chain
                    .key_at(tokens)
                    .is_some_and(|key| !self.kv.has_anchor(key))
        };
        let shared = (seen * block_tokens).min((prompt - 1) / block_tokens * block_tokens)
            / granule
            * granule;
        let resumes_final_chunk = hit > 0 && prompt - hit <= chunk;
        if shared > hit && !resumes_final_chunk && absent(shared) {
            points.push(shared);
        }
        // The prompt end, or with a granule the last multiple of it: an
        // anchor sits where a chunk may end.
        let end = prompt / granule * granule;
        if end > hit && end >= block_tokens {
            points.push(end);
        }
        // Chunks restart at the hit and at every capture before the prompt
        // end, so the final chunk starts a whole number of chunks after the
        // last of them.
        let start = points
            .iter()
            .copied()
            .filter(|tokens| *tokens < prompt)
            .fold(hit, usize::max);
        let repeat = (start + prompt.saturating_sub(start + 1) / chunk * chunk) / granule * granule;
        if repeat > start && absent(repeat) {
            points.push(repeat);
        }
        points.sort_unstable();
        points.dedup();
        points
    }

    /// The chunk limit of one request's prefill items.
    ///
    /// A scored request also cuts at the scored row capacity, never at what
    /// other items left of it, so its chunk boundaries (and the rounding of
    /// its prompt scores) do not depend on the traffic it shares a batch with.
    const fn request_chunk_limit(
        record: &RequestRecord,
        chunk_limit: usize,
        maximum_scored_rows: usize,
    ) -> usize {
        if record.spec.prompt_scoring && maximum_scored_rows < chunk_limit {
            maximum_scored_rows
        } else {
            chunk_limit
        }
    }

    /// Logits rows a prefill item of `tokens` fills: every row of a scored
    /// chunk, otherwise the final prompt row only.
    const fn logit_rows(record: &RequestRecord, tokens: usize) -> usize {
        if record.spec.prompt_scoring {
            tokens
        } else if tokens == Self::prefill_target(record) - record.prefilled_tokens {
            1
        } else {
            0
        }
    }

    /// How long a request has waited for prefill: since its last scheduled
    /// batch, or since it was queued, with earlier arrivals first on a tie.
    const fn waiting_since(record: &RequestRecord) -> (u64, u64) {
        let tick = if record.last_scheduled > 0 {
            record.last_scheduled
        } else {
            record.queued_tick
        };
        (tick, record.arrival)
    }

    /// Tokens one prefill item must take for the model side to execute it.
    ///
    /// Use the full chunk selected for this step or the authenticated tail
    /// up to the next capture point. Competing fixed-mode requests keep their chunk;
    /// decode-aware mode selects a smaller chunk before packing, never from
    /// the leftover packing budget. A chunk that stops short of the prefill
    /// target ends on a multiple of `granule` in sequence position: the last
    /// one it reaches, or the next one when a limit below the granule
    /// reaches none (chunks start on multiples, and capture points are
    /// multiples, so that end is within the remaining tokens).
    fn chunk_tokens(
        record: &RequestRecord,
        chunk_limit: usize,
        maximum_scored_rows: usize,
        granule: usize,
        spans: bool,
    ) -> usize {
        if let Some(chunk) = record.guaranteed_prefill_chunks {
            let start = record.prefilled_tokens;
            let end = (start / chunk)
                .saturating_add(1)
                .saturating_mul(chunk)
                .min(Self::prefill_target(record));
            return end - start;
        }
        let remaining = Self::chunk_remaining(record, spans);
        let limit = Self::request_chunk_limit(record, chunk_limit, maximum_scored_rows);
        let tokens = remaining.min(limit);
        let start = record.prefilled_tokens;
        let end = start + tokens;
        if end == Self::prefill_target(record) {
            return tokens;
        }
        let aligned = end / granule * granule;
        if aligned > start {
            aligned - start
        } else {
            ((start / granule + 1) * granule).min(start + remaining) - start
        }
    }

    /// Pushes one prefill item when the model side can execute it.
    ///
    /// The item takes its request's chunk limit or the tail, never more than
    /// the budget left after `reserved_budget` is held back for a later item.
    /// A chunk the leftover token or logits-row budget would truncate is
    /// declined, and the request waits for a later batch. A request is pushed
    /// at most once per batch: one already in `items` or deferred as the
    /// encode overrun is declined. While `anchors` lasts, a chunk that can
    /// run past its prompt-end anchor to the prompt end does so when that
    /// fits, and takes one; otherwise it ends at the anchor. A chunk whose
    /// media cover more pages than the whole arena ends at the last granule
    /// boundary that fits it; one whose media do not fit the pages the batch
    /// has left is declined. Returns whether an item was pushed.
    #[allow(clippy::too_many_arguments)]
    fn push_prefill(
        requests: &[RequestRecord],
        items: &mut Vec<BatchItem>,
        index: usize,
        budget: &mut usize,
        chunk_limit: usize,
        prefill_budget: &mut usize,
        rows: PrefillRows<'_>,
        reserved_budget: usize,
        chunk_media: &mut dyn FnMut(RequestId, usize, usize) -> ChunkMedia,
        overrun: &mut Option<(usize, usize, usize)>,
        anchors: &mut usize,
    ) -> bool {
        let record = &requests[index];
        if overrun.is_some_and(|(deferred, ..)| deferred == index)
            || items.iter().any(|item| item.request_id == record.spec.id)
        {
            return false;
        }
        let (held_tokens, held_rows) = match *rows.reserve {
            Some(reservation) if reservation.index == index => {
                *rows.reserve = None;
                (0, 0)
            }
            Some(reservation) => (reservation.tokens, reservation.rows),
            None => (0, 0),
        };
        let scored_budget = rows.scored_budget;
        let media_budget = rows.media_budget;
        let remaining = Self::prefill_target(record) - record.prefilled_tokens;
        let available = budget.saturating_sub(reserved_budget.saturating_add(held_tokens));
        if let Some(anchor) = Self::spannable_anchor(record, chunk_limit, rows.maximum_scored_rows)
            .filter(|_| *anchors > 0)
        {
            let tokens = remaining;
            let scored_rows = Self::logit_rows(record, tokens);
            if scored_rows <= scored_budget.saturating_sub(held_rows)
                && tokens <= available
                && tokens <= *prefill_budget
            {
                let media = chunk_media(record.spec.id, record.prefilled_tokens, tokens);
                let token_cost = tokens.saturating_add(media.encode_cost);
                if token_cost <= available && media.pages <= *media_budget {
                    items.push(BatchItem {
                        request_id: record.spec.id,
                        phase: BatchPhase::Prefill { tokens },
                        token_cost,
                        expects_output: true,
                        anchor: Some(anchor),
                    });
                    *anchors -= 1;
                    *prefill_budget -= tokens;
                    *budget -= token_cost;
                    *scored_budget -= scored_rows;
                    *media_budget -= media.pages;
                    return true;
                }
            }
        }
        let chunk = |limit| {
            Self::chunk_tokens(record, limit, rows.maximum_scored_rows, rows.granule, false)
        };
        let tokens = chunk(chunk_limit);
        if tokens == 0 {
            return false;
        }
        let Some((tokens, media)) =
            Self::fit_media(record, tokens, rows.media_arena, chunk, chunk_media)
        else {
            return false;
        };
        let scored_rows = Self::logit_rows(record, tokens);
        if scored_rows > scored_budget.saturating_sub(held_rows) {
            return false;
        }
        if tokens > available || tokens > *prefill_budget || media.pages > *media_budget {
            return false;
        }
        let token_cost = tokens.saturating_add(media.encode_cost);
        if token_cost > available {
            // One encode per batch may overrun the budget, so a large item is
            // never starved. It is added after every item that fits. Only the
            // encode overruns: its prompt tokens are held back now, so the
            // items after it cannot take the batch past its token budget.
            if overrun.is_none() {
                *prefill_budget -= tokens;
                *overrun = Some((index, tokens, token_cost));
                *budget -= tokens;
                *scored_budget -= scored_rows;
                *media_budget -= media.pages;
            }
            return false;
        }
        items.push(BatchItem {
            request_id: record.spec.id,
            phase: BatchPhase::Prefill { tokens },
            token_cost,
            expects_output: tokens == remaining,
            anchor: None,
        });
        *prefill_budget -= tokens;
        *budget -= token_cost;
        *scored_budget -= scored_rows;
        *media_budget -= media.pages;
        true
    }

    /// The chunk of `tokens` with its media demand, or, when its media cover
    /// more than `arena` pages, the longest chunk `chunk(limit)` forms whose
    /// media fit. Chunks keep their granule alignment, and a longer limit
    /// covers the media of every shorter one, so the search is over limits.
    /// `None` only when even the shortest chunk does not fit, which
    /// submission refuses (`RequestSpec::with_media_pages`).
    fn fit_media(
        record: &RequestRecord,
        tokens: usize,
        arena: usize,
        chunk: impl Fn(usize) -> usize,
        chunk_media: &mut dyn FnMut(RequestId, usize, usize) -> ChunkMedia,
    ) -> Option<(usize, ChunkMedia)> {
        let (id, offset) = (record.spec.id, record.prefilled_tokens);
        let media = chunk_media(id, offset, tokens);
        if media.pages <= arena {
            return Some((tokens, media));
        }
        let mut fitting = None;
        let (mut low, mut high) = (1, tokens - 1);
        while low <= high {
            let limit = low + (high - low) / 2;
            let candidate = chunk(limit);
            let media = chunk_media(id, offset, candidate);
            if media.pages <= arena {
                fitting = Some((candidate, media));
                low = limit + 1;
            } else {
                high = limit - 1;
            }
        }
        fitting
    }

    /// The prefill tokens, attention pairs and decode items of `items`.
    fn step_shape(&self, items: &[BatchItem]) -> Result<StepShape, ScheduleError> {
        let mut tokens = 0usize;
        let mut pairs = 0u64;
        let mut decode = 0usize;
        for item in items {
            match item.phase {
                BatchPhase::Prefill { tokens: chunk } => {
                    let record = self.record(item.request_id)?;
                    tokens += chunk;
                    pairs = pairs
                        .saturating_add(mixed_budget::chunk_pairs(record.prefilled_tokens, chunk));
                }
                BatchPhase::Decode | BatchPhase::Verify { .. } => decode += 1,
            }
        }
        Ok(StepShape::new(tokens, pairs, decode))
    }

    /// Offers the batch's decode items to `plan` and applies its draft counts.
    fn apply_draft_plan<F>(
        &mut self,
        items: &mut [BatchItem],
        budget: usize,
        limits: DraftLimits,
        plan: F,
    ) -> Result<(), ScheduleError>
    where
        F: FnOnce(&[DraftCandidate], DraftBudget) -> Vec<usize>,
    {
        if limits.maximum_draft_tokens == 0 {
            return Ok(());
        }
        let mut candidates = Vec::new();
        candidates
            .try_reserve_exact(items.len())
            .map_err(|_| ScheduleError::MetadataAllocationFailed)?;
        for item in items.iter() {
            if item.phase != BatchPhase::Decode {
                continue;
            }
            let record = self.record(item.request_id)?;
            // The KV quota holds prompt plus maximum output minus one tokens,
            // and a verify item appends one more token than it drafts, so
            // drafts are bounded by the output tokens left after the bonus,
            // and by the tokens the request's reservation still covers.
            let remaining = record
                .spec
                .maximum_new_tokens
                .checked_sub(record.emitted_tokens)
                .ok_or(ScheduleError::InvariantViolation)?;
            let reserved = record
                .reserved_tokens
                .saturating_sub(Self::committed_tokens(record) + 1);
            candidates.push(DraftCandidate {
                request_id: item.request_id,
                maximum_draft_tokens: limits
                    .maximum_draft_tokens
                    .min(remaining.saturating_sub(1))
                    .min(reserved),
                remaining_tokens: remaining,
            });
        }
        if candidates.is_empty() {
            return Ok(());
        }
        let mut draft_budget = DraftBudget {
            spare_draft_tokens: budget / self.config.decode_cost,
            spare_rows: limits
                .maximum_verify_tokens
                .saturating_sub(candidates.len()),
            prefill: items
                .iter()
                .any(|item| matches!(item.phase, BatchPhase::Prefill { .. })),
        };
        // Base appends may evict; optional drafts use only the remaining free
        // capacity. A stateful drafter is called once and its batch will fit.
        if self.offload.is_some() {
            let mut maxima = candidates.iter();
            let mut base = Vec::with_capacity(items.len());
            let mut extras = Vec::with_capacity(items.len());
            for item in items.iter() {
                base.push((
                    self.record(item.request_id)?
                        .handle
                        .expect("admitted candidate"),
                    item.phase.appended_tokens(),
                ));
                extras.push(if item.phase == BatchPhase::Decode {
                    maxima
                        .next()
                        .expect("decode candidate")
                        .maximum_draft_tokens
                } else {
                    0
                });
            }
            let limits = self
                .kv
                .cap_append_extras(&base, &extras)
                .map_err(ScheduleError::Kv)?;
            let mut candidates_mut = candidates.iter_mut();
            let mut total = 0usize;
            for (item, limit) in items.iter().zip(limits) {
                if item.phase == BatchPhase::Decode {
                    candidates_mut
                        .next()
                        .expect("decode candidate")
                        .maximum_draft_tokens = limit;
                    total += limit;
                }
            }
            draft_budget.spare_draft_tokens = draft_budget.spare_draft_tokens.min(total);
            draft_budget.spare_rows = draft_budget.spare_rows.min(total);
        }
        let drafts = plan(&candidates, draft_budget);
        if drafts.len() != candidates.len() {
            return Err(ScheduleError::InvalidDraftPlan);
        }
        let mut total = 0usize;
        for (candidate, draft_tokens) in candidates.iter().zip(&drafts) {
            if *draft_tokens > candidate.maximum_draft_tokens {
                return Err(ScheduleError::InvalidDraftPlan);
            }
            total = total
                .checked_add(*draft_tokens)
                .ok_or(ScheduleError::InvalidDraftPlan)?;
        }
        if total > draft_budget.spare_draft_tokens || total > draft_budget.spare_rows {
            return Err(ScheduleError::InvalidDraftPlan);
        }
        let mut drafts = drafts.into_iter();
        for item in items.iter_mut() {
            if item.phase != BatchPhase::Decode {
                continue;
            }
            let draft_tokens = drafts.next().ok_or(ScheduleError::InvariantViolation)?;
            if draft_tokens > 0 {
                item.phase = BatchPhase::Verify { draft_tokens };
                item.token_cost = (draft_tokens + 1) * self.config.decode_cost;
            }
        }
        Ok(())
    }

    fn push_decode(
        requests: &[RequestRecord],
        items: &mut Vec<BatchItem>,
        index: usize,
        budget: &mut usize,
        decode_cost: usize,
    ) {
        items.push(BatchItem {
            request_id: requests[index].spec.id,
            phase: BatchPhase::Decode,
            token_cost: decode_cost,
            expects_output: true,
            anchor: None,
        });
        *budget -= decode_cost;
    }

    fn transition_terminal(
        &mut self,
        id: RequestId,
        state: RequestState,
    ) -> Result<(), ScheduleError> {
        let index = self.record_index(id)?;
        if self.requests[index].state.is_terminal() {
            return Err(ScheduleError::TerminalRequest(id));
        }
        if let Some(handle) = self.requests[index].handle.take() {
            // A capture the last commit queued, not yet run, would read this
            // sequence after its release: it is dropped with the request.
            self.captures
                .retain(|capture| capture.handle != handle || capture.deferred);
            self.kv.release(handle).map_err(ScheduleError::Kv)?;
            let released = self.shared_released(&[index]);
            self.release_shared(index);
            self.requests[index].shared_tail = false;
            self.active_sequences -= 1;
            self.reserved_blocks -= self.requests[index].reserved_blocks + released;
            self.requests[index].reserved_blocks = 0;
            self.requests[index].reserved_tokens = 0;
        }
        self.requests[index].state = state;
        self.drop_own_anchor(index)
    }

    /// Removes the private anchor of the request's last preemption: it was
    /// restored, abandoned, or the request ended.
    fn drop_own_anchor(&mut self, index: usize) -> Result<(), ScheduleError> {
        self.requests[index].own_anchor_transfer = None;
        if let Some(key) = self.requests[index].own_anchor.take()
            && !self.resumes.contains_key(&self.requests[index].spec.id)
        {
            self.kv.remove_anchor(key).map_err(ScheduleError::Kv)?;
        }
        Ok(())
    }

    fn rollback_pending_for_mutation(&mut self) -> Result<(), ScheduleError> {
        if let Some(batch) = self.pending.as_ref() {
            self.kv
                .rollback_prepared(batch.kv_plan_id)
                .map_err(ScheduleError::Kv)?;
            self.pending = None;
        }
        Ok(())
    }

    fn record(&self, id: RequestId) -> Result<&RequestRecord, ScheduleError> {
        self.requests
            .iter()
            .find(|record| record.spec.id == id)
            .ok_or(ScheduleError::UnknownRequest(id))
    }

    fn record_index(&self, id: RequestId) -> Result<usize, ScheduleError> {
        self.requests
            .iter()
            .position(|record| record.spec.id == id)
            .ok_or(ScheduleError::UnknownRequest(id))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InvalidRequest {
    ZeroPrompt,
    ZeroMaximumNewTokens,
    LengthOverflow,
    ExceedsKvPool { maximum: usize, requested: usize },
    ExceedsMediaArena { maximum: usize, requested: usize },
    PrefixMismatch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Overload {
    TrackingCapacity,
    ProfileCapacity {
        profile: &'static str,
        maximum: usize,
    },
    MaximumSequences,
    KvCapacity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfigError {
    EmptyCapacityProfileName,
    ZeroProfileCapacity,
    ZeroMaximumSequences,
    TrackingBelowSequenceCapacity,
    TrackingCapacityTooLarge,
    ProfileBelowSequenceCapacity,
    ProfileAboveTrackingCapacity,
    ZeroScheduledTokenBudget,
    ZeroPrefillChunkLimit,
    ZeroMaximumScoredRows,
    PrefillChunkLimitAboveTokenBudget,
    InvalidDecodeCost,
    InvalidAdmissionWatermark,
    ZeroForcedPreemptionInterval,
    InvalidMixedPolicy,
    PrefillChunkLimitOffGranule,
    ScoredRowsOffGranule,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCapacityProfileName => {
                formatter.write_str("scheduler capacity profile name is empty")
            }
            Self::ZeroProfileCapacity => {
                formatter.write_str("scheduler capacity profile has zero in-flight capacity")
            }
            Self::ZeroMaximumSequences => {
                formatter.write_str("scheduler maximum sequence count is zero")
            }
            Self::TrackingBelowSequenceCapacity => {
                formatter.write_str("scheduler tracking capacity is below maximum sequences")
            }
            Self::TrackingCapacityTooLarge => {
                formatter.write_str("scheduler tracking capacity exceeds the metadata limit")
            }
            Self::ProfileBelowSequenceCapacity => {
                formatter.write_str("scheduler profile capacity is below maximum sequences")
            }
            Self::ProfileAboveTrackingCapacity => {
                formatter.write_str("scheduler profile capacity exceeds tracking capacity")
            }
            Self::ZeroScheduledTokenBudget => formatter.write_str("scheduler token budget is zero"),
            Self::ZeroMaximumScoredRows => {
                formatter.write_str("scheduler scored row limit is zero")
            }
            Self::ZeroPrefillChunkLimit => {
                formatter.write_str("scheduler prefill chunk limit is zero")
            }
            Self::PrefillChunkLimitAboveTokenBudget => {
                formatter.write_str("scheduler prefill chunk limit exceeds the token budget")
            }
            Self::InvalidDecodeCost => {
                formatter.write_str("scheduler decode cost is zero or exceeds the token budget")
            }
            Self::InvalidAdmissionWatermark => {
                formatter.write_str("admission watermark must be below the whole pool")
            }
            Self::ZeroForcedPreemptionInterval => {
                formatter.write_str("forced preemption interval is zero")
            }
            Self::InvalidMixedPolicy => formatter.write_str(
                "mixed-step target and progress floor must be positive, with at most 8 decode steps",
            ),
            Self::PrefillChunkLimitOffGranule => formatter.write_str(
                "scheduler prefill granule is zero or does not divide the prefill chunk limit",
            ),
            Self::ScoredRowsOffGranule => formatter.write_str(
                "scheduler prefill granule does not divide the maximum scored rows",
            ),
        }
    }
}

impl Error for ConfigError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SubmitError {
    DuplicateId(RequestId),
    Invalid(InvalidRequest),
    Overloaded(Overload),
    RequestOrdinalExhausted,
}

impl fmt::Display for SubmitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateId(id) => {
                write!(formatter, "request ID {} is already tracked", id.get())
            }
            Self::Invalid(reason) => write!(formatter, "request is invalid: {reason:?}"),
            Self::Overloaded(reason) => write!(formatter, "scheduler is overloaded: {reason:?}"),
            Self::RequestOrdinalExhausted => {
                formatter.write_str("scheduler request ordinal is exhausted")
            }
        }
    }
}

impl Error for SubmitError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScheduleError {
    Kv(KvError),
    Overloaded(Overload),
    MetadataAllocationFailed,
    PendingBatch,
    NoPendingBatch,
    BatchIdMismatch,
    OutcomeCountMismatch,
    OutcomeMembershipMismatch,
    InvalidOutcome(RequestId),
    BatchIdExhausted,
    PendingGenerationExhausted,
    SchedulingTickExhausted,
    UnknownRequest(RequestId),
    TerminalRequest(RequestId),
    RequestNotTerminal(RequestId),
    RequestPendingBatch(RequestId),
    PreparedCommitMismatch,
    InvalidDraftPlan,
    InvariantViolation,
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kv(error) => write!(formatter, "KV operation failed: {error}"),
            Self::Overloaded(reason) => write!(formatter, "scheduler is overloaded: {reason:?}"),
            Self::MetadataAllocationFailed => {
                formatter.write_str("scheduler metadata allocation failed")
            }
            Self::PendingBatch => formatter.write_str("a scheduler batch is already pending"),
            Self::NoPendingBatch => formatter.write_str("no scheduler batch is pending"),
            Self::BatchIdMismatch => {
                formatter.write_str("scheduler batch ID does not match the pending batch")
            }
            Self::OutcomeCountMismatch => {
                formatter.write_str("batch outcomes do not match the pending item count")
            }
            Self::OutcomeMembershipMismatch => {
                formatter.write_str("batch outcomes do not match pending request phases")
            }
            Self::InvalidOutcome(id) => write!(
                formatter,
                "batch outcome for request ID {} is invalid",
                id.get()
            ),
            Self::BatchIdExhausted => formatter.write_str("scheduler batch ID is exhausted"),
            Self::PendingGenerationExhausted => {
                formatter.write_str("scheduler pending generation is exhausted")
            }
            Self::SchedulingTickExhausted => {
                formatter.write_str("scheduler fairness tick is exhausted")
            }
            Self::UnknownRequest(id) => write!(formatter, "request ID {} is not tracked", id.get()),
            Self::TerminalRequest(id) => {
                write!(formatter, "request ID {} is already terminal", id.get())
            }
            Self::RequestNotTerminal(id) => {
                write!(formatter, "request ID {} is not terminal", id.get())
            }
            Self::RequestPendingBatch(id) => write!(
                formatter,
                "request ID {} belongs to the pending batch",
                id.get()
            ),
            Self::PreparedCommitMismatch => {
                formatter.write_str("prepared commit does not match this scheduler state")
            }
            Self::InvalidDraftPlan => {
                formatter.write_str("draft plan does not fit the batch it extends")
            }
            Self::InvariantViolation => formatter.write_str("scheduler invariant audit failed"),
        }
    }
}

impl Error for ScheduleError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_new_preemption_replaces_old_anchor(evicted: bool) {
        let mut s = Scheduler::new(
            SchedulerConfig::new(2, 8, 32, 8, 1)
                .unwrap()
                .with_prefix_cache(1)
                .with_last_page_rewrites(true)
                .with_admission(AdmissionPolicy::Optimistic(
                    OptimisticAdmission::new(0, 3).unwrap(),
                )),
            KvGeometry::new(4, 10).unwrap(),
        )
        .unwrap();
        s.configure_offload(OffloadConfig {
            minimum_tokens: 1,
            snapshot_pages: 3,
            enabled: true,
            force_host: false,
        });
        let mut tier = s.new_host_tier(32).unwrap();
        let pinned = RequestId::new(1);
        let id = RequestId::new(2);
        s.submit(
            RequestSpec::new(pinned, 1, 25)
                .unwrap()
                .with_reserved_admission(),
        )
        .unwrap();
        s.submit(RequestSpec::new(id, 4, 16).unwrap()).unwrap();
        let commit = |s: &mut Scheduler| {
            let b = s.next_batch().unwrap().unwrap();
            let outcomes: Vec<_> = b
                .items()
                .iter()
                .map(|i| {
                    BatchOutcome::success(
                        i.request_id(),
                        i.phase(),
                        usize::from(i.expects_output()),
                        false,
                    )
                })
                .collect();
            s.commit_batch(b.id(), &outcomes).unwrap();
        };
        let mut chain = |_: RequestId, tokens: usize| {
            Some(PrefixRequestV1::new(
                Arc::new(PrefixChainV1::new(4, Arc::from(vec![1; tokens]), Vec::new()).unwrap()),
                true,
            ))
        };
        commit(&mut s);
        s.preempt(1, &mut chain).unwrap();
        let capture = s.take_prefix_captures().pop().unwrap();
        let inserted = s.begin_prefix_capture(&capture).unwrap().is_some();
        let transfer = s
            .offload_capture(&mut tier, &capture, inserted)
            .unwrap()
            .into_submission();
        s.end_prefix_capture(&capture).unwrap();
        let old = s.requests[1].own_anchor.unwrap();
        s.discard_swap(&mut tier, id).unwrap();
        if evicted {
            s.kv.remove_anchor(old).unwrap();
        } else {
            s.accept_kv_transfer(
                &mut tier,
                transfer.completion(runtime_kv::KvTransferStatus::Complete),
            )
            .unwrap();
        }
        // Seven pages stay reserved for the neighbor. The old snapshot and
        // copied tail do not fit, so this admission must recompute cold.
        assert_eq!(s.admit().unwrap().admitted(), [id]);
        assert!(s.pending_prefix_restores().is_empty());
        assert_eq!(s.requests[1].prefilled_tokens, 0);
        assert_eq!(s.kv.has_anchor(old), !evicted);
        assert_eq!(s.requests[1].own_anchor_transfer.is_some(), evicted);
        commit(&mut s);
        s.configure_offload(OffloadConfig {
            minimum_tokens: 1,
            snapshot_pages: 3,
            enabled: false,
            force_host: false,
        });
        s.preempt(1, &mut chain).unwrap();
        assert_eq!(
            (
                s.kv.has_anchor(old),
                s.requests[1].own_anchor_transfer.is_some()
            ),
            (false, false)
        );
        let capture = s.take_prefix_captures().pop().unwrap();
        assert!(s.begin_prefix_capture(&capture).unwrap().is_some());
        s.end_prefix_capture(&capture).unwrap();
        s.cancel(pinned).unwrap();
        assert_eq!(tier.pending_transfers(), usize::from(evicted));
        assert_eq!(s.admit().unwrap().admitted(), [id]);
        assert!(s.pending_prefix_restores()[0].resumes_decoding());
        s.complete_prefix_restore(id).unwrap();
        s.cancel(id).unwrap();
        if evicted {
            s.accept_kv_transfer(
                &mut tier,
                transfer.completion(runtime_kv::KvTransferStatus::Complete),
            )
            .unwrap();
        }
        s.audit().unwrap();
        s.audit_host_tier(&tier).unwrap();
        assert_eq!(s.kv_stats().free_blocks(), 10);
    }

    #[test]
    fn new_preemption_releases_previous_anchor_after_cold_admission() {
        check_new_preemption_replaces_old_anchor(false);
    }

    #[test]
    fn new_preemption_does_not_wait_on_a_previous_anchors_copy() {
        check_new_preemption_replaces_old_anchor(true);
    }

    /// The recomputed-row count probes and records each block a chunk
    /// touches once, so every chunk of a long prompt costs the same.
    #[test]
    fn recomputed_row_counting_is_linear_in_the_prompt() {
        let mut scheduler = Scheduler::new(
            SchedulerConfig::new(2, 8, 64, 64, 1)
                .unwrap()
                .with_prefix_cache(1),
            KvGeometry::new(4, 512).unwrap(),
        )
        .unwrap();
        let first: Arc<[u32]> = (0..1_021).map(|token| token % 97 + 3).collect();
        // The second prompt diverges before the first one's final chunk, so
        // it misses and compares every row it shares.
        let second: Arc<[u32]> = first[..900].iter().copied().chain(0..121).collect();
        for (id, tokens) in [(1, first), (2, second)] {
            let chain = PrefixChainV1::new(4, Arc::clone(&tokens), Vec::new()).unwrap();
            scheduler
                .submit_with_prefix(
                    RequestSpec::new(RequestId::new(id), tokens.len(), 1).unwrap(),
                    Some(PrefixRequestV1::new(Arc::new(chain), true)),
                )
                .unwrap();
            let mut chunks = Vec::new();
            while let Some(batch) = scheduler.next_batch().unwrap() {
                let before = scheduler.kv.produced_operations();
                let item = &batch.items()[0];
                let BatchPhase::Prefill { tokens } = item.phase() else {
                    panic!("one prefill item per step");
                };
                let outcome = BatchOutcome::success(
                    item.request_id(),
                    item.phase(),
                    usize::from(item.expects_output()),
                    item.expects_output(),
                );
                scheduler.commit_batch(batch.id(), &[outcome]).unwrap();
                chunks.push((tokens, scheduler.kv.produced_operations() - before));
                for capture in scheduler.take_prefix_captures() {
                    scheduler.begin_prefix_capture(&capture).unwrap();
                    scheduler.end_prefix_capture(&capture).unwrap();
                }
            }
            for (tokens, operations) in &chunks {
                // At most one probe and one record per block the chunk touches.
                assert!(
                    *operations <= 2 * (*tokens as u64 / 4 + 1),
                    "send {id}: {chunks:?}"
                );
            }
            assert!(chunks.len() >= 16, "send {id}: {chunks:?}");
        }
        assert_eq!(
            scheduler
                .kv_metrics()
                .snapshot()
                .prefix_cache_recomputed_seen_tokens,
            900
        );
    }

    #[test]
    fn unplanned_drafts_do_not_evict_prefix_anchors() {
        for offload in [false, true] {
            let mut scheduler = Scheduler::new(
                SchedulerConfig::new(2, 8, 32, 8, 1).unwrap(),
                KvGeometry::new(4, 8).unwrap(),
            )
            .unwrap();
            if offload {
                scheduler.configure_offload(OffloadConfig {
                    minimum_tokens: 1,
                    snapshot_pages: 1,
                    enabled: true,
                    force_host: false,
                });
            }
            let id = RequestId::new(1);
            scheduler
                .submit(RequestSpec::new(id, 4, 16).unwrap())
                .unwrap();
            let batch = scheduler.next_batch().unwrap().unwrap();
            scheduler
                .commit_batch(
                    batch.id(),
                    &[BatchOutcome::success(
                        id,
                        BatchPhase::Prefill { tokens: 4 },
                        1,
                        false,
                    )],
                )
                .unwrap();
            let cached = scheduler.kv.create_sequence(24).unwrap();
            scheduler.kv.append(cached, 24).unwrap();
            let key = runtime_kv::PrefixCacheKeyV1::new([5; 32]);
            scheduler
                .kv
                .insert_anchor(key, key, cached, 0)
                .unwrap()
                .unwrap();
            scheduler.kv.release(cached).unwrap();

            let batch = scheduler
                .next_batch_with_drafts(DraftLimits::new(7, 16), |candidates, _| {
                    // The drafter declines to draft (for example, no n-gram match).
                    vec![0; candidates.len()]
                })
                .unwrap()
                .unwrap();
            assert!(
                scheduler.kv.has_anchor(key),
                "unplanned drafts evicted an anchor"
            );
            scheduler.rollback_batch(batch.id()).unwrap();
            scheduler.kv.audit().unwrap();
        }
    }

    #[test]
    fn leased_capacity_runs_base_decodes_when_optional_drafts_do_not_fit() {
        let mut scheduler = Scheduler::new(
            SchedulerConfig::new(2, 8, 32, 8, 1).unwrap(),
            KvGeometry::new(4, 8).unwrap(),
        )
        .unwrap();
        scheduler.configure_offload(OffloadConfig {
            minimum_tokens: 1,
            snapshot_pages: 1,
            enabled: true,
            force_host: false,
        });
        let id = RequestId::new(1);
        scheduler
            .submit(RequestSpec::new(id, 4, 16).unwrap())
            .unwrap();
        let batch = scheduler.next_batch().unwrap().unwrap();
        scheduler
            .commit_batch(
                batch.id(),
                &[BatchOutcome::success(
                    id,
                    BatchPhase::Prefill { tokens: 4 },
                    1,
                    false,
                )],
            )
            .unwrap();
        // Six pages are now held only by a delayed outgoing copy. The one
        // remaining free page fits decode, but not seven speculative tokens.
        let victim = scheduler.kv.create_sequence(24).unwrap();
        scheduler.kv.append(victim, 24).unwrap();
        let pages = scheduler
            .kv
            .snapshot(victim)
            .unwrap()
            .into_blocks()
            .iter()
            .map(|p| scheduler.kv.page_allocation(*p).unwrap())
            .collect::<Vec<_>>();
        let lease = scheduler.kv.retain_transfer_pages(&pages).unwrap();
        scheduler.kv.release(victim).unwrap();
        let mut called = false;
        let batch = scheduler
            .next_batch_with_drafts(DraftLimits::new(7, 16), |candidates, budget| {
                called = true;
                assert_eq!(budget.spare_draft_tokens(), 3);
                assert!(candidates.iter().all(|c| c.maximum_draft_tokens() == 3));
                vec![0; candidates.len()]
            })
            .unwrap()
            .unwrap();
        assert!(called);
        assert_eq!(batch.items()[0].phase(), BatchPhase::Decode);
        scheduler
            .commit_batch(
                batch.id(),
                &[BatchOutcome::success(id, BatchPhase::Decode, 1, false)],
            )
            .unwrap();
        scheduler.kv.release_transfer_pages(&lease).unwrap();
        scheduler.audit().unwrap();
    }

    #[test]
    fn push_prefill_declines_a_request_already_in_the_batch_or_deferred() {
        let mut scheduler = Scheduler::new(
            SchedulerConfig::new(2, 4, 64, 8, 1).unwrap(),
            KvGeometry::new(4, 64).unwrap(),
        )
        .unwrap();
        scheduler
            .submit(RequestSpec::new(RequestId::new(1), 20, 1).unwrap())
            .unwrap();
        scheduler.admit().unwrap();
        let index = scheduler.candidates(false).unwrap()[0];
        let push = |items: &mut Vec<BatchItem>, overrun: &mut Option<(usize, usize, usize)>| {
            let (mut budget, mut prefill_budget, mut scored_budget) = (64, 64, usize::MAX);
            Scheduler::push_prefill(
                &scheduler.requests,
                items,
                index,
                &mut budget,
                8,
                &mut prefill_budget,
                PrefillRows {
                    maximum_scored_rows: usize::MAX,
                    granule: 1,
                    scored_budget: &mut scored_budget,
                    reserve: &mut None,
                    media_arena: 0,
                    media_budget: &mut 0,
                },
                0,
                &mut |_, _, _| ChunkMedia::default(),
                overrun,
                &mut 0,
            )
        };
        let mut items = Vec::new();
        assert!(push(&mut items, &mut None));
        assert!(!push(&mut items, &mut None));
        assert_eq!(items.len(), 1);
        assert!(!push(&mut Vec::new(), &mut Some((index, 8, 108))));
    }
}
