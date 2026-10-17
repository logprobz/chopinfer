use std::cell::Cell;
use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use runtime_engine::{
    Engine, EngineError, EngineEvent, EngineEventKind, EngineMetrics, EngineRequest,
    FinishReason as EngineFinishReason, MediaInputV1 as EngineMediaInputV1,
    MediaPayloadV1 as EngineMediaPayloadV1, MediaSpanV1 as EngineMediaSpanV1, ModelExecutor,
    PositionSpanV1 as EnginePositionSpanV1, RequestId, StepStatus,
};
use runtime_frontend::{
    AUDIO_MAXIMUM_SECONDS_V1, AUDIO_SAMPLE_RATE_V1, BundleChatStateV1, BundleParserSourceV1,
    CompletionDecoderV1, CompletionFinishV1, CompletionUpdateV1, DecoderLimitsV1,
    FrontendDescriptorV1, FrontendMultimodalV1, FrontendPositionRuleV1, FrontendVocabularyV1,
    GeneratedTokenV1, PromptModeV1, TokenContributionV1,
};
use runtime_grammar::{CompiledGrammar, ConstraintState};
use runtime_kv::KvMetrics;
use runtime_model_api::ModelVocabulary;
use runtime_sampling::SamplingParameters;
use runtime_scheduler::RequestState;
use runtime_tokenizer::Tokenizer;
use tokio::sync::oneshot;

use crate::media_memory::{
    MediaMemoryV1, REQUEST_MEDIA_PREPARATION_BYTES_V1, RequestMediaMemoryV1,
};
pub use crate::watchdog::WorkerWatchdogV1;
use crate::watchdog::{WorkerPhase, WorkerProgress};

use runtime_server::{
    AdmissionError, BackendReadiness, BackendToolCall, BoxFuture, CancellationSignal, EventSender,
    FinishReason, FrontendMediaPartKindV1, FrontendMediaPartV1, GeneratedTokenLogprob,
    GenerationBackend, GenerationEvent, MediaInputV1, MediaSpanV1, MetricsSnapshotProvider,
    PositionSpanV1, PreparedPrompt, PreparedRequest, PromptSegmentV1,
    PromptTokenLogprob as ServerPromptTokenLogprob, ReasoningContent, TokenLogprob, TokenUsage,
    VisibleContent,
};

/// `runtime_server::MAX_IN_FLIGHT_REQUESTS`, which bounds the in-flight limit
/// a served deployment sizes this channel to.
const MAX_CONTROL_CAPACITY: usize = 4_096;
pub const DEFAULT_TOKENIZER_INPUT_BYTES: usize = 32 * 1024 * 1024;
const DEFAULT_FORWARDING_CAPACITY: usize = 4_096;
const STARTING: u8 = 0;
const READY: u8 = 1;
const FAILED: u8 = 2;
const STOPPED: u8 = 3;
const KV_STALLED: u8 = 4;
const FAILURE_MESSAGE: &str = "generation worker failed";
const FAILURE_CODE: &str = "generation_worker_failed";
const DECODE_MESSAGE: &str = "model output could not be decoded";
const DECODE_CODE: &str = "invalid_model_output";

/// What the installed declaration tells the generation side.
///
/// The frontend reads the same component for the serving half. Nothing here
/// is compiled in: a deployment that installs a different bundle runs a
/// different vocabulary, stops on different identifiers, and trusts a
/// different set of prompt boundary tokens without a rebuild.
#[derive(Clone, Debug)]
pub struct EngineModelPolicyV1 {
    descriptor: Arc<FrontendDescriptorV1>,
    vocabulary: ModelVocabulary,
    stop_token_ids: Vec<u32>,
    prompt_special_tokens: Vec<String>,
    prompt_rows: bool,
    media_inputs: bool,
    /// The bundle frontend a completion without a rendered chat parses
    /// through. A deployment always sets it. Without one such a completion
    /// has no format to parse with: a host refuses it, and a test build
    /// calls the declared format directly.
    parser_source: Option<BundleParserSourceV1>,
}

/// Two policies are equal when they read the same declaration the same
/// way, whichever frontend instance their completions parse through.
impl PartialEq for EngineModelPolicyV1 {
    fn eq(&self, other: &Self) -> bool {
        self.descriptor == other.descriptor
            && self.vocabulary == other.vocabulary
            && self.stop_token_ids == other.stop_token_ids
            && self.prompt_special_tokens == other.prompt_special_tokens
            && self.prompt_rows == other.prompt_rows
            && self.media_inputs == other.media_inputs
    }
}

impl EngineModelPolicyV1 {
    /// Reads the generation half of one frontend declaration.
    #[must_use]
    pub fn from_declaration(descriptor: &FrontendDescriptorV1) -> Self {
        Self {
            descriptor: Arc::new(descriptor.clone()),
            // The declaration was validated to name a vocabulary in range, so
            // there is no size left here for this to refuse.
            vocabulary: ModelVocabulary::new(descriptor.vocabulary.size as usize)
                .expect("a validated declaration names a usable vocabulary size"),
            stop_token_ids: descriptor.vocabulary.stop_token_ids.clone(),
            prompt_special_tokens: descriptor.vocabulary.prompt_special_tokens.clone(),
            prompt_rows: false,
            media_inputs: false,
            parser_source: None,
        }
    }

    #[must_use]
    pub fn with_prompt_rows(mut self, enabled: bool) -> Self {
        self.prompt_rows = enabled;
        self
    }

    #[must_use]
    pub fn with_media_inputs(mut self, enabled: bool) -> Self {
        self.media_inputs = enabled;
        self
    }

    /// The declaration this policy and its frontend both read.
    #[must_use]
    pub fn declaration(&self) -> &Arc<FrontendDescriptorV1> {
        &self.descriptor
    }

    /// Parses a completion without a rendered chat through `source`, as a
    /// raw completion parses through the frontend the deployment serves.
    #[must_use]
    pub fn with_parser_source(mut self, source: BundleParserSourceV1) -> Self {
        self.parser_source = Some(source);
        self
    }

    /// Where a completion without a rendered chat opens its parser.
    #[must_use]
    pub const fn parser_source(&self) -> Option<&BundleParserSourceV1> {
        self.parser_source.as_ref()
    }

    #[must_use]
    pub const fn vocabulary(&self) -> ModelVocabulary {
        self.vocabulary
    }

    #[must_use]
    pub fn stop_token_ids(&self) -> &[u32] {
        &self.stop_token_ids
    }

    /// Exactly the special tokens a rendered prompt may carry.
    ///
    /// A prompt is encoded against this list and nothing wider, so a client
    /// cannot write a boundary token into its own conversation.
    #[must_use]
    pub fn prompt_special_tokens(&self) -> Vec<&str> {
        self.prompt_special_tokens
            .iter()
            .map(String::as_str)
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineBackendConfig {
    control_capacity: usize,
    forwarding_capacity: usize,
    queue_wait: Duration,
    tokenizer_input_bytes: usize,
    step_overlap: bool,
    watchdog: Option<WorkerWatchdogV1>,
}

impl EngineBackendConfig {
    pub const fn new(control_capacity: usize) -> Result<Self, EngineBackendConfigError> {
        if control_capacity == 0 {
            return Err(EngineBackendConfigError::ZeroControlCapacity);
        }
        if control_capacity > MAX_CONTROL_CAPACITY {
            return Err(EngineBackendConfigError::ControlCapacityTooLarge);
        }
        Ok(Self {
            control_capacity,
            forwarding_capacity: DEFAULT_FORWARDING_CAPACITY,
            queue_wait: Duration::ZERO,
            tokenizer_input_bytes: DEFAULT_TOKENIZER_INPUT_BYTES,
            step_overlap: false,
            watchdog: None,
        })
    }

    #[must_use]
    pub const fn forwarding_capacity(self) -> usize {
        self.forwarding_capacity
    }

    #[must_use]
    pub const fn control_capacity(self) -> usize {
        self.control_capacity
    }

    #[must_use]
    pub const fn queue_wait(self) -> Duration {
        self.queue_wait
    }

    /// Refuses a request still queued after `queue_wait`; zero, the default,
    /// lets queued requests wait for capacity without a limit.
    pub const fn with_queue_wait(mut self, queue_wait: Duration) -> Self {
        self.queue_wait = queue_wait;
        self
    }

    pub const fn with_tokenizer_input_bytes(mut self, tokenizer_input_bytes: usize) -> Self {
        self.tokenizer_input_bytes = tokenizer_input_bytes;
        self
    }

    /// Delivers each step's events while the device runs the next step
    /// (`CHOPINFER_STEP_OVERLAP=1`) instead of before scheduling it. Stop
    /// detection and every other decision the engine makes from a step's
    /// tokens still happen first, and the tokens and logprobs of every step
    /// a request runs are unchanged. What a client does with its stream can
    /// have different consequences, because each event reaches it up to one
    /// step later: see `Outbox`.
    pub const fn with_step_overlap(mut self, step_overlap: bool) -> Self {
        self.step_overlap = step_overlap;
        self
    }

    #[must_use]
    pub const fn step_overlap(self) -> bool {
        self.step_overlap
    }

    /// Watches the worker's progress from another thread (see
    /// `WorkerWatchdogV1`). Without it, a worker that never returns from a
    /// step stays ready and `shutdown` waits for it without a bound.
    pub const fn with_watchdog(mut self, watchdog: Option<WorkerWatchdogV1>) -> Self {
        self.watchdog = watchdog;
        self
    }

    #[must_use]
    pub const fn watchdog(self) -> Option<WorkerWatchdogV1> {
        self.watchdog
    }
}

impl Default for EngineBackendConfig {
    fn default() -> Self {
        Self {
            control_capacity: 256,
            forwarding_capacity: DEFAULT_FORWARDING_CAPACITY,
            queue_wait: Duration::ZERO,
            tokenizer_input_bytes: DEFAULT_TOKENIZER_INPUT_BYTES,
            step_overlap: false,
            watchdog: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineBackendConfigError {
    ZeroControlCapacity,
    ControlCapacityTooLarge,
    ZeroForwardingCapacity,
    ForwardingCapacityTooLarge,
}

impl fmt::Display for EngineBackendConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::ZeroControlCapacity => "engine backend control capacity must be positive",
            Self::ControlCapacityTooLarge => "engine backend control capacity exceeds its limit",
            Self::ZeroForwardingCapacity => "engine backend forwarding capacity must be positive",
            Self::ForwardingCapacityTooLarge => {
                "engine backend forwarding capacity exceeds its limit"
            }
        })
    }
}

impl std::error::Error for EngineBackendConfigError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EngineBackendStartupError {
    ThreadSpawn,
    WorkerDidNotStart,
    /// Building the engine overran its watchdog allowance; the worker thread
    /// is left as it is.
    WorkerStalled,
    ModelVocabularyMismatch {
        expected: usize,
        actual: usize,
    },
    /// The caller's own engine construction refused, on the worker thread.
    ///
    /// A worker that owns a device or a loaded module builds its engine where
    /// it will run it, so that construction can fail for reasons only the
    /// worker can see. Carrying the reason here is what lets the caller report
    /// it instead of reporting a worker that did not answer.
    EngineUnavailable(String),
}

impl fmt::Display for EngineBackendStartupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ThreadSpawn => {
                formatter.write_str("generation worker thread could not be started")
            }
            Self::WorkerDidNotStart => {
                formatter.write_str("generation worker did not report startup")
            }
            Self::WorkerStalled => {
                formatter.write_str("generation worker stalled while building its engine")
            }
            Self::ModelVocabularyMismatch { expected, actual } => write!(
                formatter,
                "generation engine vocabulary {actual} does not match model vocabulary {expected}"
            ),
            Self::EngineUnavailable(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for EngineBackendStartupError {}

pub struct EngineGenerationBackend {
    tokenizer: Arc<Tokenizer>,
    grammars: Arc<crate::grammar::GrammarServiceV1>,
    policy: EngineModelPolicyV1,
    tokenizer_input_bytes: usize,
    control: SyncSender<WorkerCommand>,
    state: Arc<AtomicU8>,
    next_request_id: AtomicU64,
    forwarding_capacity: usize,
    maximum_sequence_tokens: usize,
    startup_page_tokens: usize,
    startup_pages: usize,
    metrics: EngineMetrics,
    kv_metrics: KvMetrics,
    worker: Mutex<Option<JoinHandle<()>>>,
    progress: Arc<WorkerProgress>,
    media_root: Option<PathBuf>,
    remote_media: runtime_media::remote::RemoteMediaPolicy,
    media_memory: Arc<MediaMemoryV1>,
    media_request_bytes: u64,
    /// Images this backend prepared, by source, for the turns that send them
    /// again.
    image_cache: Arc<crate::media_cache::PreparedImageCacheV1>,
    video: Option<crate::video::VideoDecodeServiceV1>,
    video_unavailable: Option<String>,
}

impl EngineGenerationBackend {
    pub fn new<E>(
        tokenizer: Tokenizer,
        engine: Engine<E>,
        config: EngineBackendConfig,
        policy: EngineModelPolicyV1,
    ) -> Result<Self, EngineBackendStartupError>
    where
        E: ModelExecutor + Send + 'static,
    {
        Self::new_with_factory(tokenizer, move || engine, config, policy)
    }

    pub fn new_with_factory<E, F>(
        tokenizer: Tokenizer,
        engine_factory: F,
        config: EngineBackendConfig,
        policy: EngineModelPolicyV1,
    ) -> Result<Self, EngineBackendStartupError>
    where
        E: ModelExecutor + 'static,
        F: FnOnce() -> Engine<E> + Send + 'static,
    {
        Self::try_new_with_factory(tokenizer, move || Ok(engine_factory()), config, policy)
    }

    /// Starts the worker with an engine its own thread may fail to build.
    ///
    /// A worker that opens a device, or loads a module that only its thread may
    /// touch, has to build its engine where it will run it. That construction
    /// can refuse for reasons the calling thread cannot see, so the factory
    /// returns its own refusal and this reports that refusal rather than a
    /// worker that never answered.
    pub fn try_new_with_factory<E, F>(
        tokenizer: Tokenizer,
        engine_factory: F,
        config: EngineBackendConfig,
        policy: EngineModelPolicyV1,
    ) -> Result<Self, EngineBackendStartupError>
    where
        E: ModelExecutor + 'static,
        F: FnOnce() -> Result<Engine<E>, String> + Send + 'static,
    {
        validate_media_recipes_v1(policy.declaration())
            .map_err(EngineBackendStartupError::EngineUnavailable)?;
        let model_vocabulary = policy.vocabulary();
        let tokenizer = Arc::new(tokenizer);
        let worker_tokenizer = Arc::clone(&tokenizer);
        let worker_policy = policy.clone();
        let (control, receiver) = mpsc::sync_channel(config.control_capacity);
        let (startup_sender, startup_receiver) = mpsc::sync_channel(1);
        let state = Arc::new(AtomicU8::new(STARTING));
        let worker_state = Arc::clone(&state);
        let progress = WorkerProgress::new();
        let worker_progress = Arc::clone(&progress);
        // The watchdog starts before the engine is built, so building it is
        // observed too.
        if let Some(watchdog) = config.watchdog {
            let failed = Arc::clone(&state);
            crate::watchdog::spawn(watchdog, Arc::clone(&progress), move || {
                failed.store(FAILED, Ordering::Release);
            })
            .map_err(|_| EngineBackendStartupError::ThreadSpawn)?;
        }
        let worker = thread::Builder::new()
            .name("bundle-engine".to_owned())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    worker_progress.mark(WorkerPhase::Startup);
                    let mut engine = match engine_factory() {
                        Ok(engine) => engine,
                        Err(refusal) => {
                            let _ = startup_sender
                                .send(Err(EngineBackendStartupError::EngineUnavailable(refusal)));
                            return WorkerExit::Shutdown;
                        }
                    };
                    let engine_vocabulary = engine.model_vocabulary();
                    if engine_vocabulary != model_vocabulary {
                        let _ = startup_sender.send(Err(
                            EngineBackendStartupError::ModelVocabularyMismatch {
                                expected: model_vocabulary.get(),
                                actual: engine_vocabulary.get(),
                            },
                        ));
                        return WorkerExit::Shutdown;
                    }
                    // The engine is built here and never leaves this thread, so
                    // its counters are handed back now: these handles are the
                    // only way a metrics reader on another thread sees what
                    // this worker's scheduler, engine and KV pool record.
                    let started = WorkerStartupV1 {
                        maximum_sequence_tokens: engine.maximum_sequence_tokens(),
                        startup_page_tokens: engine.startup_page_tokens(),
                        startup_pages: engine.kv_stats().free_blocks(),
                        metrics: engine.metrics(),
                        kv_metrics: engine.kv_metrics(),
                    };
                    // Each step is timed again once its batch is formed, by
                    // what the batch does.
                    let batches = Arc::clone(&worker_progress);
                    engine.observe_batches(Box::new(move |prefill| batches.batch(prefill)));
                    // A watchdog that failed a slow startup keeps it failed.
                    if worker_state
                        .compare_exchange(STARTING, READY, Ordering::AcqRel, Ordering::Acquire)
                        .is_err()
                        || startup_sender.send(Ok(started)).is_err()
                    {
                        return WorkerExit::Shutdown;
                    }
                    worker_loop(
                        worker_tokenizer,
                        engine,
                        receiver,
                        config,
                        worker_policy,
                        &worker_state,
                        &worker_progress,
                    )
                }));
                let stalled = worker_progress.is_stalled();
                worker_progress.end();
                match result {
                    Ok(WorkerExit::Shutdown) if !stalled => {
                        worker_state.store(STOPPED, Ordering::Release);
                    }
                    _ => worker_state.store(FAILED, Ordering::Release),
                }
            })
            .map_err(|_| EngineBackendStartupError::ThreadSpawn)?;
        // Waiting for the engine ends when the watchdog fails its build.
        let started = loop {
            match startup_receiver.recv_timeout(Duration::from_millis(10)) {
                Err(RecvTimeoutError::Timeout) if !progress.is_stalled() => {}
                Err(RecvTimeoutError::Timeout) => {
                    state.store(FAILED, Ordering::Release);
                    return Err(EngineBackendStartupError::WorkerStalled);
                }
                Ok(Ok(started)) => break started,
                Ok(Err(error)) => {
                    state.store(FAILED, Ordering::Release);
                    drop(control);
                    join_unless_stalled(worker, &progress);
                    return Err(error);
                }
                Err(RecvTimeoutError::Disconnected) => {
                    state.store(FAILED, Ordering::Release);
                    drop(control);
                    join_unless_stalled(worker, &progress);
                    return Err(EngineBackendStartupError::WorkerDidNotStart);
                }
            }
        };
        if started.maximum_sequence_tokens == 0 {
            let _ = control.send(WorkerCommand::Shutdown);
            join_unless_stalled(worker, &progress);
            return Err(EngineBackendStartupError::WorkerDidNotStart);
        }
        let media_memory = MediaMemoryV1::process();
        let image_cache = Arc::new(crate::media_cache::PreparedImageCacheV1::default());
        media_memory.register(Arc::downgrade(&image_cache) as _);
        Ok(Self {
            grammars: Arc::new(crate::grammar::GrammarServiceV1::new(
                Arc::clone(&tokenizer),
                model_vocabulary.get(),
                boundary_token_ids_v1(&policy),
            )),
            tokenizer,
            policy,
            tokenizer_input_bytes: config.tokenizer_input_bytes,
            control,
            state,
            next_request_id: AtomicU64::new(1),
            forwarding_capacity: config.forwarding_capacity,
            maximum_sequence_tokens: started.maximum_sequence_tokens,
            startup_page_tokens: started.startup_page_tokens,
            startup_pages: started.startup_pages,
            metrics: started.metrics,
            kv_metrics: started.kv_metrics,
            worker: Mutex::new(Some(worker)),
            progress,
            media_root: None,
            remote_media: Default::default(),
            media_memory,
            media_request_bytes: REQUEST_MEDIA_PREPARATION_BYTES_V1,
            image_cache,
            video: None,
            video_unavailable: None,
        })
    }

    /// Media memory budgets other than the process's, for tests.
    #[cfg(test)]
    pub(crate) fn with_media_memory(
        mut self,
        pool: Arc<MediaMemoryV1>,
        request_bytes: u64,
    ) -> Self {
        pool.register(Arc::downgrade(&self.image_cache) as _);
        self.media_memory = pool;
        self.media_request_bytes = request_bytes;
        self
    }

    pub fn with_remote_media(mut self, policy: runtime_media::remote::RemoteMediaPolicy) -> Self {
        self.remote_media = policy;
        self
    }

    fn read_media_source(
        &self,
        part: &FrontendMediaPartV1,
        limits: runtime_media::MediaLimits,
        cancellation: &CancellationSignal,
        request_deadline: Instant,
    ) -> Result<runtime_media::MediaSource, AdmissionError> {
        // Source-rate WAV duration is checked after probing, against the recipe
        // rate. The byte bound still applies while reading the source.
        let limits = if part.kind == FrontendMediaPartKindV1::Audio {
            runtime_media::MediaLimits {
                maximum_audio_frames: u64::MAX,
                ..limits
            }
        } else {
            limits
        };
        let result = if part.source.starts_with("http://") || part.source.starts_with("https://") {
            runtime_media::remote::fetch_with_deadline(
                &part.source,
                &self.remote_media,
                limits,
                request_deadline,
                || cancellation.is_cancelled(),
            )
        } else {
            runtime_media::read_source(&part.source, self.media_root.as_deref(), limits)
        };
        result.map_err(|error| AdmissionError::Rejected(error.to_string()))
    }

    /// Enables declared media preprocessing and starts video decode when needed.
    pub fn with_media_processing(
        mut self,
        allowed_root: Option<PathBuf>,
        device_ordinal: i32,
    ) -> Self {
        self.media_root = allowed_root;
        let has_video =
            self.policy
                .declaration()
                .multimodal
                .as_ref()
                .is_some_and(|media| {
                    media.modalities.iter().any(|modality| {
                        modality.kind == runtime_frontend::FrontendMediaKindV1::Video
                    })
                });
        if has_video {
            #[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
            match crate::video::VideoDecodeServiceV1::start(device_ordinal) {
                Ok(service) => self.video = Some(service),
                Err(error) => self.video_unavailable = Some(error.to_string()),
            }
            #[cfg(not(all(target_os = "linux", feature = "linux-cuda-host")))]
            {
                let _ = device_ordinal;
                self.video_unavailable = Some("video decoding requires a CUDA host".to_owned());
            }
        }
        self
    }

    fn preprocess_media_part_v1(
        &self,
        part: &FrontendMediaPartV1,
        modality: &runtime_frontend::FrontendMediaModalityV1,
        cancellation: &CancellationSignal,
        request_deadline: Instant,
        memory: &mut RequestMediaMemoryV1,
    ) -> Result<PreparedMediaInputV1, AdmissionError> {
        if part.kind == FrontendMediaPartKindV1::Video {
            return self.preprocess_video_part_v1(
                part,
                modality,
                cancellation,
                request_deadline,
                memory,
            );
        }
        let parameters = &modality.recipe.parameters;
        let limits = part_limits_v1(modality);
        let source_reserved = if part.kind == FrontendMediaPartKindV1::Audio {
            // Readers can grow their byte buffer geometrically up to the bound.
            let bytes = (limits.maximum_bytes as u64)
                .saturating_mul(2)
                .saturating_add(1);
            memory.reserve(bytes)?;
            bytes
        } else {
            0
        };
        let source = if part.part_type == "input_audio"
            && !part.source.starts_with("http://")
            && !part.source.starts_with("https://")
        {
            let bytes = runtime_media::decode_base64(&part.source, limits.maximum_bytes)
                .map_err(|error| AdmissionError::Rejected(error.to_string()))?;
            runtime_media::MediaSource {
                mime: "audio/wav".to_owned(),
                bytes,
            }
        } else {
            self.read_media_source(part, limits, cancellation, request_deadline)?
        };
        let cached = (part.kind == FrontendMediaPartKindV1::Image).then(|| {
            crate::media_cache::PreparedImageCacheV1::key(&source.mime, modality, &source.bytes)
        });
        if let Some(prepared) = cached.and_then(|key| self.image_cache.get(&key)) {
            return Ok(prepared);
        }
        let declared = self
            .policy
            .declaration()
            .multimodal
            .as_ref()
            .ok_or(AdmissionError::InvalidPrompt)?;
        let encoder = &declared.encoders[usize::from(modality.encoder)];
        let reserved;
        let (geometry, output_rows, values, columns) = match part.kind {
            FrontendMediaPartKindV1::Image => {
                let recipe = declared_image_recipe_v1(&modality.recipe)?;
                if modality.layout.is_some() {
                    check_image_span_v1(&source, modality, recipe, limits)?;
                }
                let grid;
                (grid, reserved) = prepare_image_v1(&source, recipe, limits, memory)?;
                let span = match modality.layout {
                    None => grid.tokens,
                    Some(_) => image_span_rows_v1(modality, recipe_merge_v1(recipe), grid.grid)
                        .ok_or(AdmissionError::InvalidPrompt)?,
                };
                (
                    [grid.grid[0], grid.grid[1], grid.grid[2], 0],
                    u32::try_from(span).map_err(|_| AdmissionError::InvalidPrompt)?,
                    grid.rows,
                    grid.columns,
                )
            }
            FrontendMediaPartKindV1::Audio => {
                if source.mime != "audio/wav" && source.mime != "audio/x-wav" {
                    return Err(AdmissionError::InvalidPrompt);
                }
                let recipe = audio_recipe_v1(parameters)?;
                let mel;
                (mel, reserved) = prepare_audio_v1(
                    &source,
                    recipe,
                    limits,
                    &|| cancellation.is_cancelled() || Instant::now() >= request_deadline,
                    source_reserved,
                    memory,
                )?;
                (
                    [
                        u32::try_from(mel.frames).map_err(|_| AdmissionError::InvalidPrompt)?,
                        u32::try_from(mel.bins).map_err(|_| AdmissionError::InvalidPrompt)?,
                        0,
                        0,
                    ],
                    u32::try_from(mel.tokens).map_err(|_| AdmissionError::InvalidPrompt)?,
                    mel.values,
                    mel.bins,
                )
            }
            FrontendMediaPartKindV1::Video => unreachable!(),
        };
        drop(source);
        if columns != encoder.payload_columns as usize || output_rows > modality.maximum_tokens {
            return Err(AdmissionError::InvalidPrompt);
        }
        let payload = media_values_to_bytes_v1(values, encoder.payload_element)?;
        if payload.len() > encoder.maximum_payload_bytes {
            return Err(AdmissionError::InvalidPrompt);
        }
        let key = runtime_media::content_key(
            &modality.recipe.tag,
            &serde_json::to_vec(parameters).map_err(|_| AdmissionError::InvalidPrompt)?,
            &payload,
        );
        let prepared: PreparedMediaInputV1 = MediaInputV1 {
            key,
            encoder: modality.encoder,
            geometry,
            output_rows,
            payload: memory.retain(reserved, payload)?,
        }
        .into();
        if let Some(key) = cached {
            self.image_cache
                .insert(key, PreparedMediaInputV1::clone(&prepared));
        }
        Ok(prepared)
    }

    /// The prepared image an inline (data URL) part names, when this backend
    /// prepared its source before. A request whose parts all hit needs no
    /// preparation memory.
    fn cached_inline_image_v1(
        &self,
        part: &FrontendMediaPartV1,
        declared: &FrontendMultimodalV1,
    ) -> Option<PreparedMediaInputV1> {
        if part.kind != FrontendMediaPartKindV1::Image || !part.source.starts_with("data:") {
            return None;
        }
        let modality = declared.modalities.iter().find(|modality| {
            modality.kind == runtime_frontend::FrontendMediaKindV1::Image
                && modality.part_types.contains(&part.part_type)
        })?;
        let source =
            runtime_media::read_source(&part.source, None, part_limits_v1(modality)).ok()?;
        self.image_cache
            .get(&crate::media_cache::PreparedImageCacheV1::key(
                &source.mime,
                modality,
                &source.bytes,
            ))
    }

    fn preprocess_video_part_v1(
        &self,
        part: &FrontendMediaPartV1,
        modality: &runtime_frontend::FrontendMediaModalityV1,
        cancellation: &CancellationSignal,
        request_deadline: Instant,
        memory: &mut RequestMediaMemoryV1,
    ) -> Result<PreparedMediaInputV1, AdmissionError> {
        {
            let service = self.video.as_ref().ok_or_else(|| {
                AdmissionError::Unavailable(
                    self.video_unavailable
                        .clone()
                        .unwrap_or_else(|| "video decoder is unavailable".to_owned()),
                )
            })?;
            let parameters = &modality.recipe.parameters;
            let image = parameters
                .get("image")
                .ok_or(AdmissionError::InvalidPrompt)?;
            let limits = video_limits_v1(modality, image);
            let source = self.read_media_source(part, limits, cancellation, request_deadline)?;
            if source.mime != "video/mp4" {
                return Err(AdmissionError::InvalidPrompt);
            }
            let video_limits = crate::video::VideoLimitsV1 {
                maximum_samples: 1 << 16,
                maximum_width: 4096,
                maximum_height: 4096,
                sampling: runtime_media::video_frames::FrameSampling {
                    maximum_frames: VIDEO_MAXIMUM_FRAMES_V1,
                    seconds_per_frame: VIDEO_SECONDS_PER_FRAME_V1,
                },
            };
            let prepared = crate::video::prepare_video_v1(&source.bytes, &video_limits)
                .map_err(video_admission_error_v1)?;
            // The reservation holds the frames' share of the decode pool's
            // frame budget until their patches are built.
            let (frames, _reservation) = service
                .decode(prepared, video_limits, &|| cancellation.is_cancelled())
                .map_err(video_admission_error_v1)?
                .into_parts();
            let frames = frames
                .into_iter()
                .map(|frame| {
                    runtime_media::RgbImage::new(frame.width, frame.height, frame.rgb, limits)
                        .map(|rgb| (f64::from(frame.label_seconds), rgb))
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| AdmissionError::Rejected(error.to_string()))?;
            let grids = runtime_media::recipe::video_grids(
                &frames,
                image_recipe_v1(image)?,
                media_u64_v1(parameters, "total_max_pixels")?,
                limits,
            )
            .map_err(|error| AdmissionError::Rejected(error.to_string()))?;
            let declared = self
                .policy
                .declaration()
                .multimodal
                .as_ref()
                .ok_or(AdmissionError::InvalidPrompt)?;
            let encoder = &declared.encoders[usize::from(modality.encoder)];
            let output_rows = u32::try_from(grids.iter().map(|grid| grid.grid.tokens).sum::<u64>())
                .map_err(|_| AdmissionError::InvalidPrompt)?;
            let columns = grids.first().map_or(0, |grid| grid.grid.columns);
            if columns != encoder.payload_columns as usize || output_rows > modality.maximum_tokens
            {
                return Err(AdmissionError::InvalidPrompt);
            }
            let geometry = video_geometry_v1(&grids)?;
            let rendered_grids = grids
                .iter()
                .map(|grid| {
                    Ok((
                        grid.timestamp.clone(),
                        u32::try_from(grid.grid.tokens)
                            .map_err(|_| AdmissionError::InvalidPrompt)?,
                    ))
                })
                .collect::<Result<Vec<_>, AdmissionError>>()?;
            let mut values = Vec::new();
            values
                .try_reserve_exact(grids.iter().map(|grid| grid.grid.rows.len()).sum())
                .map_err(|_| payload_allocation_failed_v1())?;
            for grid in grids {
                values.extend(grid.grid.rows);
            }
            let payload = media_values_to_bytes_v1(values, encoder.payload_element)?;
            if payload.len() > encoder.maximum_payload_bytes {
                return Err(AdmissionError::InvalidPrompt);
            }
            let key = runtime_media::content_key(
                &modality.recipe.tag,
                &serde_json::to_vec(parameters).map_err(|_| AdmissionError::InvalidPrompt)?,
                &payload,
            );
            Ok(PreparedMediaInputV1 {
                input: MediaInputV1 {
                    key,
                    encoder: modality.encoder,
                    geometry,
                    output_rows,
                    payload: memory.retain(0, payload)?,
                },
                grids: rendered_grids,
            })
        }
    }

    /// What this worker publishes about itself, for one metrics endpoint.
    ///
    /// Every number behind this handle is recorded by the engine, the
    /// scheduler or the KV pool the worker is running; nothing here counts
    /// requests a second time on the transport side. The handle keeps the
    /// worker's lifecycle state, so a worker that failed or stopped is
    /// published as failed or stopped rather than as a healthy idle one.
    #[must_use]
    pub fn runtime_metrics(&self, served_model_names: &[String]) -> NativeRuntimeMetricsV1 {
        NativeRuntimeMetricsV1 {
            model_name: served_model_names
                .first()
                .unwrap_or(&self.policy.descriptor.models[0].id)
                .clone(),
            engine: self.metrics.clone(),
            kv: self.kv_metrics.clone(),
            state: Arc::clone(&self.state),
        }
    }

    /// Cancels remaining work and joins the worker before releasing host services.
    ///
    /// A worker the watchdog declared stalled is left running and never
    /// joined: it may still own device work, and the process is about to
    /// exit for its supervisor.
    pub fn shutdown(&self) {
        if let Ok(mut worker) = self.worker.lock()
            && let Some(worker) = worker.take()
        {
            let mut command = WorkerCommand::Shutdown;
            loop {
                if self.progress.is_stalled() {
                    break;
                }
                match self.control.try_send(command) {
                    Ok(()) | Err(TrySendError::Disconnected(_)) => break,
                    Err(TrySendError::Full(pending)) => {
                        command = pending;
                        thread::sleep(Duration::from_millis(10));
                    }
                }
            }
            join_unless_stalled(worker, &self.progress);
        }
        if self.state.load(Ordering::Acquire) != FAILED {
            self.state.store(STOPPED, Ordering::Release);
        }
    }

    /// Whether the watchdog failed this worker. The process is then about
    /// to exit, and nothing should wait on the worker or on stderr.
    pub(crate) fn is_stalled(&self) -> bool {
        self.progress.is_stalled()
    }

    fn allocate_request_id(&self) -> Result<RequestId, AdmissionError> {
        self.next_request_id
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map(RequestId::new)
            .map_err(|_| AdmissionError::WorkerUnavailable)
    }
}

impl GenerationBackend for EngineGenerationBackend {
    fn startup_warmup(&self, begin: bool) -> BoxFuture<'_, Result<(), String>> {
        let (sender, receiver) = oneshot::channel();
        let sent = self
            .control
            .send(WorkerCommand::StartupWarmup(begin, sender));
        Box::pin(async move {
            sent.map_err(|_| "generation worker unavailable".to_owned())?;
            receiver
                .await
                .map_err(|_| "generation worker failed during warmup cleanup".to_owned())?
        })
    }

    fn startup_wave_capacity(&self, tokens: usize) -> usize {
        self.startup_pages / tokens.max(1).div_ceil(self.startup_page_tokens)
    }

    fn readiness(&self) -> BackendReadiness {
        match self.state.load(Ordering::Acquire) {
            READY => BackendReadiness::Ready,
            STARTING => BackendReadiness::NotReady("generation worker is starting".to_owned()),
            FAILED => BackendReadiness::Terminated("generation worker failed".to_owned()),
            KV_STALLED => BackendReadiness::NotReady(
                "KV transfer is overdue; polling retained leases".to_owned(),
            ),
            _ => BackendReadiness::Terminated("generation worker is stopped".to_owned()),
        }
    }

    fn maximum_sequence_tokens(&self) -> usize {
        self.maximum_sequence_tokens
    }

    fn prompt_token_count(&self, prompt: &PreparedPrompt) -> Result<usize, AdmissionError> {
        if matches!(prompt, PreparedPrompt::MediaChat { .. }) {
            return self.prepare_prompt(prompt.clone()).map(|(_, count)| count);
        }
        if let PreparedPrompt::MediaTokens { tokens, .. } = prompt {
            validate_token_ids(&self.tokenizer, tokens)?;
            return Ok(tokens.len());
        }
        if let PreparedPrompt::TokenIds(ids) = prompt {
            validate_token_ids(&self.tokenizer, ids)?;
            return Ok(ids.len());
        }
        tokenize_prompt(
            &self.tokenizer,
            prompt,
            &self.policy,
            self.tokenizer_input_bytes,
        )
        .map(|tokens| tokens.len())
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
        if let PreparedPrompt::MediaChat { segments, media } = &prompt {
            if !self.policy.media_inputs {
                return Err(AdmissionError::InvalidPrompt);
            }
            let declared = self
                .policy
                .declaration()
                .multimodal
                .as_ref()
                .ok_or(AdmissionError::InvalidPrompt)?;
            // Images prepared before need no preparation memory, so a request
            // whose parts all hit skips the allowance wait. Any other waits
            // holding none of its hits, and stops looking at its first miss:
            // a held hit is not idle, so the pool could not reclaim it for
            // this or another waiter. Preparation finds the hits again.
            let mut cached: std::collections::VecDeque<_> = media
                .iter()
                .map(|part| self.cached_inline_image_v1(part, declared))
                .collect::<Option<_>>()
                .unwrap_or_default();
            let allowance = if cached.len() == media.len() {
                0
            } else {
                self.media_request_bytes
            };
            let mut memory = RequestMediaMemoryV1::wait(
                Arc::clone(&self.media_memory),
                allowance,
                cancellation,
            )?;
            let request_deadline = Instant::now() + self.remote_media.request_timeout;
            let (tokens, inputs, spans, positions) = expand_media_chat_v1(
                declared,
                &self.policy.declaration().vocabulary,
                segments,
                media,
                |text| {
                    self.tokenizer
                        .encode_with_input_limit(
                            text,
                            &self.policy.prompt_special_tokens(),
                            self.tokenizer_input_bytes,
                        )
                        .map_err(|_| AdmissionError::InvalidPrompt)
                },
                |part, modality| match cached.pop_front() {
                    Some(prepared) => Ok(prepared),
                    None => self.preprocess_media_part_v1(
                        part,
                        modality,
                        cancellation,
                        request_deadline,
                        &mut memory,
                    ),
                },
            )?;
            let count = tokens.len();
            return Ok((
                PreparedPrompt::MediaTokens {
                    tokens,
                    media: inputs,
                    spans,
                    positions,
                },
                count,
            ));
        }
        let tokens = prepare_owned_prompt(
            &self.tokenizer,
            prompt,
            &self.policy,
            self.tokenizer_input_bytes,
        )?;
        let count = tokens.len();
        Ok((PreparedPrompt::TokenIds(tokens), count))
    }

    fn tokenize(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>, AdmissionError> {
        // This tokenizer contract does not insert BOS or EOS automatically.
        self.tokenizer
            .encode_with_input_limit(
                text,
                &self.policy.prompt_special_tokens(),
                self.tokenizer_input_bytes,
            )
            .map_err(|_| AdmissionError::InvalidPrompt)
    }

    fn token_strings(&self, tokens: &[u32]) -> Result<Vec<String>, AdmissionError> {
        tokens
            .iter()
            .map(|id| {
                self.tokenizer
                    .token_string(*id)
                    .map(str::to_owned)
                    .map_err(|_| AdmissionError::InvalidPrompt)
            })
            .collect()
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, AdmissionError> {
        self.tokenizer
            .decode_bytes(tokens)
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .map_err(|_| AdmissionError::InvalidPrompt)
    }

    fn submit(
        &self,
        request: PreparedRequest,
        events: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        self.submit_batch(vec![(request, events)])
            .pop()
            .expect("one submission")
    }

    fn submit_batch(
        &self,
        requests: Vec<(PreparedRequest, EventSender)>,
    ) -> Vec<BoxFuture<'_, Result<(), AdmissionError>>> {
        let mut results: Vec<BoxFuture<'_, Result<(), AdmissionError>>> =
            Vec::with_capacity(requests.len());
        let mut submissions = Vec::with_capacity(requests.len());
        for (request, events) in requests {
            let id =
                validate_logit_bias(&request.sampling, self.policy.vocabulary()).and_then(|()| {
                    if self.state.load(Ordering::Acquire) == READY {
                        self.allocate_request_id()
                    } else {
                        Err(AdmissionError::WorkerUnavailable)
                    }
                });
            let id = match id {
                Ok(id) => id,
                Err(error) => {
                    results.push(Box::pin(async move { Err(error) }));
                    continue;
                }
            };
            let (forward_sender, forward_receiver) =
                tokio::sync::mpsc::channel(self.forwarding_capacity);
            tokio::spawn(forward_events(
                forward_receiver,
                events,
                request.cancellation.clone(),
            ));
            let (response, admitted) = oneshot::channel();
            submissions.push(WorkerSubmission {
                id,
                request: Box::new(request),
                constraint: None,
                events: EventSender::new(forward_sender),
                response,
            });
            results.push(Box::pin(async move {
                admitted
                    .await
                    .unwrap_or(Err(AdmissionError::WorkerUnavailable))
            }));
        }
        let control = self.control.clone();
        let state = Arc::clone(&self.state);
        if submissions
            .iter()
            .any(|submission| submission.request.constraint.is_some())
        {
            let grammars = Arc::clone(&self.grammars);
            tokio::spawn(async move {
                // Compile before publishing any of the group to the worker.
                // A failed member retains its own admission error.
                let compiled = tokio::task::spawn_blocking(move || {
                    submissions
                        .into_iter()
                        .filter_map(|mut submission| {
                            if submission.request.cancellation.is_cancelled() {
                                return None;
                            }
                            if let Some(grammar) = &submission.request.constraint {
                                let cancellation = &submission.request.cancellation;
                                match grammars.compile(grammar, &|| cancellation.is_cancelled()) {
                                    Ok(compiled) => submission.constraint = Some(compiled),
                                    // Nobody reads a cancelled request's answer.
                                    Err(_) if cancellation.is_cancelled() => return None,
                                    Err(error) => {
                                        let _ = submission.response.send(Err(
                                            AdmissionError::Rejected(format!(
                                                "constraint: {error}"
                                            )),
                                        ));
                                        return None;
                                    }
                                }
                            }
                            Some(submission)
                        })
                        .collect()
                })
                .await;
                if let Ok(submissions) = compiled {
                    enqueue_submissions(control, state, submissions);
                }
            });
        } else {
            enqueue_submissions(control, state, submissions);
        }
        results
    }
}

/// Joins the worker thread once it has returned. A worker the watchdog
/// failed is never joined: it may still own device work, and the process is
/// about to exit for its supervisor.
fn join_unless_stalled(worker: JoinHandle<()>, progress: &WorkerProgress) {
    while !worker.is_finished() {
        if progress.is_stalled() {
            crate::signals::try_write_stderr(
                "shutdown: the generation worker is stalled; not joining it\n",
            );
            return;
        }
        thread::sleep(Duration::from_millis(10));
    }
    let _ = worker.join();
}

/// Publish the whole group before the worker can take its next engine step.
/// A full control channel waits, including when cancelled commands still
/// occupy it. Scheduler admission and its replies remain per choice.
fn enqueue_submissions(
    control: SyncSender<WorkerCommand>,
    state: Arc<AtomicU8>,
    submissions: Vec<WorkerSubmission>,
) {
    if submissions.is_empty() {
        return;
    }
    let command = WorkerCommand::Submit(submissions);
    let Err(TrySendError::Full(mut command)) = control.try_send(command) else {
        return;
    };
    tokio::spawn(async move {
        let mut backoff = Duration::from_micros(250);
        loop {
            let WorkerCommand::Submit(submissions) = &command else {
                unreachable!("only submissions are retried");
            };
            if state.load(Ordering::Acquire) != READY
                || submissions
                    .iter()
                    .all(|submission| submission.request.cancellation.is_cancelled())
            {
                return;
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_millis(10));
            match control.try_send(command) {
                Ok(()) | Err(TrySendError::Disconnected(_)) => return,
                Err(TrySendError::Full(pending)) => command = pending,
            }
        }
    });
}

impl Drop for EngineGenerationBackend {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Relays one request's events from the worker's bounded channel to its
/// client, waiting on the client, until either side ends or it is cancelled.
/// Cancellation also ends a send that waits on a full client channel; events
/// still go out one at a time, in the order the worker queued them.
async fn forward_events(
    mut forward_receiver: tokio::sync::mpsc::Receiver<GenerationEvent>,
    events: EventSender,
    cancellation: CancellationSignal,
) {
    loop {
        let event = tokio::select! {
            biased;
            () = cancellation.cancelled() => return,
            event = forward_receiver.recv() => event,
        };
        let Some(event) = event else {
            return;
        };
        let sent = tokio::select! {
            biased;
            () = cancellation.cancelled() => return,
            sent = events.emit(event) => sent,
        };
        if sent.is_err() {
            cancellation.cancel();
            return;
        }
    }
}

/// What the worker reports about the engine it built on its own thread.
struct WorkerStartupV1 {
    maximum_sequence_tokens: usize,
    startup_page_tokens: usize,
    startup_pages: usize,
    metrics: EngineMetrics,
    kv_metrics: KvMetrics,
}

/// The metrics one generation worker publishes, in Prometheus text.
///
/// This is the whole of what a served process reports: its uptime, which
/// lifecycle state its worker is in, and the counters the engine and the KV
/// pool behind it recorded. There is no separate transport-side tally, and no
/// measurement here is derived from a caller saying a run went well.
///
/// Absent by construction, until a producer exists: KV evictions, copy-on-write
/// bytes, and peak device memory. The pool counts copy-on-write copies and the
/// tokens they initialized, which are not bytes, and nothing in the runtime
/// observes device residency yet. They are left unpublished rather than
/// published as an observed zero.
#[derive(Clone, Debug)]
pub struct NativeRuntimeMetricsV1 {
    model_name: String,
    engine: EngineMetrics,
    kv: KvMetrics,
    state: Arc<AtomicU8>,
}

impl NativeRuntimeMetricsV1 {
    /// Renders one complete snapshot, at the monotonic time it was asked for.
    #[must_use]
    pub fn prometheus(&self, monotonic_micros: u64) -> String {
        let state = self.state.load(Ordering::Acquire);
        let mut output = format!(
            "# TYPE chop_process_uptime_seconds gauge\nchop_process_uptime_seconds {}.{:06}\n",
            monotonic_micros / 1_000_000,
            monotonic_micros % 1_000_000
        );
        // A worker that never started, failed, or has stopped publishes zero
        // requests exactly as a healthy idle one does. Which of those it is
        // only these say, so they are published beside the counters rather
        // than left for a reader to assume.
        for (name, value) in [
            ("chop_generation_worker_starting", STARTING),
            ("chop_generation_worker_ready", READY),
            ("chop_generation_worker_failed", FAILED),
            ("chop_generation_worker_stopped", STOPPED),
        ] {
            output.push_str(&format!(
                "# TYPE {name} gauge\n{name} {}\n",
                u8::from(state == value)
            ));
        }
        output.push_str(&self.engine.snapshot().prometheus());
        let kv = self.kv.snapshot();
        output.push_str(&kv.prometheus());
        output.push_str(&self.engine.prometheus_compatibility(&self.model_name));
        let model = self
            .model_name
            .replace('\\', "\\\\")
            .replace('\n', "\\n")
            .replace('"', "\\\"");
        let usage = if kv.block_capacity == 0 {
            0.0
        } else {
            kv.blocks_used as f64 / kv.block_capacity as f64
        };
        output.push_str(&format!("# TYPE vllm:kv_cache_usage_perc gauge\nvllm:kv_cache_usage_perc{{model_name=\"{model}\",engine=\"0\"}} {usage}\n"));
        // vLLM counts prefix-cache queries and hits in prompt tokens.
        for (name, value) in [
            (
                "vllm:prefix_cache_queries_total",
                kv.prefix_cache_queried_tokens,
            ),
            (
                "vllm:prefix_cache_hits_total",
                kv.prefix_cache_reused_tokens,
            ),
        ] {
            output.push_str(&format!(
                "# TYPE {name} counter\n{name}{{model_name=\"{model}\",engine=\"0\"}} {value}\n"
            ));
        }
        output
    }
}

impl MetricsSnapshotProvider for NativeRuntimeMetricsV1 {
    fn step_profile(&self) -> Option<String> {
        self.engine.step_profile_json()
    }
    fn snapshot(&self, monotonic_micros: u64) -> String {
        self.prometheus(monotonic_micros)
    }
}

struct WorkerSubmission {
    id: RequestId,
    request: Box<PreparedRequest>,
    constraint: Option<Arc<CompiledGrammar>>,
    events: EventSender,
    response: oneshot::Sender<Result<(), AdmissionError>>,
}

enum WorkerCommand {
    StartupWarmup(bool, oneshot::Sender<Result<(), String>>),
    Submit(Vec<WorkerSubmission>),
    Shutdown,
}

enum WorkerExit {
    Shutdown,
    Fatal,
}

/// Which generated tokens a generation's delivered events have charged.
#[derive(Default)]
pub(crate) struct TokenAccounting {
    frontier: usize,
    last_contribution: Option<(u64, usize)>,
}

/// Delivers one generation's events: at once, or with step overlap queued
/// for the next device wait (see `overlap`).
///
/// Backpressure counts queued events. An event is refused when the stream is
/// closed, or when the channel's free slots are used up by the events still
/// queued for it. Before refusing for capacity, the outbox sends everything
/// queued and checks again, so an event, a finish included, is refused only
/// while the channel is really full; the serving loop then cancels the
/// request, as for an immediate refusal.
///
/// What overlap bounds is the backlog and the delay, not outcomes:
/// - At most one step's events of a stream are queued at a time (about 26 at
///   most: eight K7 tokens with logprobs, token IDs and text, and a finish).
///   Each reaches its channel at most one step later than at once: in the
///   next step's device wait, after that step returns, or at a flush.
/// - Backpressure counts the queued events, which a reader cannot yet take.
///   A reader running close to the forwarding capacity therefore sees a
///   fuller channel than with immediate delivery, and the difference can
///   compound over its pauses: a request can be cancelled that immediate
///   delivery would have completed. With the default capacity of 4,096
///   events this needs a reader thousands of events behind.
/// - A client that cancels or disconnects because of what it has read acts
///   up to one step later, so the batches after that point can differ, and
///   with `n > 1` the choices sharing its cancellation are affected alike.
///
/// Slots only free up until the queue runs, so a queued send fails only if
/// the reader leaves first; the job then drops the generation's later events
/// and cancels it.
struct Outbox {
    sender: Rc<EventSender>,
    deferred: Option<Rc<Queued>>,
    cancellation: CancellationSignal,
}

/// A deferring outbox's events not yet sent, and whether a send failed.
#[derive(Default)]
struct Queued {
    unsent: Cell<usize>,
    refused: Cell<bool>,
}

impl Outbox {
    fn new(sender: EventSender, cancellation: CancellationSignal, deferred: bool) -> Self {
        Self {
            sender: Rc::new(sender),
            deferred: deferred.then(Rc::default),
            cancellation,
        }
    }

    fn try_emit(&self, event: GenerationEvent) -> Result<(), ()> {
        let Some(queued) = &self.deferred else {
            return self.sender.try_emit(event).map_err(drop);
        };
        if queued.refused.get() || self.sender.is_closed() {
            return Err(());
        }
        if self.sender.capacity() <= queued.unsent.get() {
            // Refuse only a channel that is full with everything sent.
            crate::overlap::run_pending();
            if queued.refused.get() || self.sender.capacity() <= queued.unsent.get() {
                return Err(());
            }
        }
        queued.unsent.set(queued.unsent.get() + 1);
        let sender = Rc::clone(&self.sender);
        let queued = Rc::clone(queued);
        let cancellation = self.cancellation.clone();
        crate::overlap::defer(move || {
            queued.unsent.set(queued.unsent.get() - 1);
            if !queued.refused.get() && sender.try_emit(event).is_err() {
                queued.refused.set(true);
                cancellation.cancel();
            }
        });
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.sender.is_closed()
    }
}

struct ActiveGeneration {
    events: Outbox,
    cancellation: CancellationSignal,
    decoder: CompletionDecoderV1,
    prompt_tokens: u64,
    /// Prompt tokens resumed from the prefix cache, when it is enabled.
    cached_tokens: Option<u64>,
    completion_tokens: usize,
    accounting: TokenAccounting,
    logprob_context: [Option<u32>; 4],
    prompt_logprob_context: [Option<u32>; 4],
    logprobs_enabled: bool,
    return_token_ids: bool,
    admission: Option<oneshot::Sender<Result<(), AdmissionError>>>,
    queued_since: Instant,
    group_admitted: Rc<Cell<bool>>,
}

/// When the worker delivers a step's events.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Delivery {
    /// Before the next step is scheduled.
    Immediate,
    /// While the device runs the next step (`CHOPINFER_STEP_OVERLAP=1`).
    Deferred,
}

fn update_transfer_readiness<E: ModelExecutor>(engine: &Engine<E>, state: &AtomicU8) {
    let next = if engine.has_stalled_kv_transfers() {
        KV_STALLED
    } else {
        READY
    };
    let _ = state.fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
        matches!(current, READY | KV_STALLED).then_some(next)
    });
}

fn shutdown_worker<E: ModelExecutor>(
    engine: &mut Engine<E>,
    active: &mut BTreeMap<RequestId, ActiveGeneration>,
) -> WorkerExit {
    cancel_all(engine, active);
    match engine.drain_kv_transfers() {
        Ok(()) => WorkerExit::Shutdown,
        Err(EngineError::KvTransferDrainTimeout) => {
            eprintln!(
                "KV transfer shutdown drain timed out; retaining DMA leases for arena cleanup"
            );
            WorkerExit::Shutdown
        }
        Err(error) => {
            eprintln!("KV transfer shutdown drain failed: {error}");
            WorkerExit::Fatal
        }
    }
}

fn worker_loop<E>(
    tokenizer: Arc<Tokenizer>,
    mut engine: Engine<E>,
    receiver: Receiver<WorkerCommand>,
    config: EngineBackendConfig,
    policy: EngineModelPolicyV1,
    state: &AtomicU8,
    progress: &WorkerProgress,
) -> WorkerExit
where
    E: ModelExecutor,
{
    let queue_wait = config.queue_wait;
    let tokenizer_input_bytes = config.tokenizer_input_bytes;
    let deliver = if config.step_overlap {
        Delivery::Deferred
    } else {
        Delivery::Immediate
    };
    // Events queued for a device wait are delivered on every way out.
    let _flush = crate::overlap::FlushOnDrop;
    let mut active = BTreeMap::new();
    let mut submissions = Vec::new();
    loop {
        progress.mark(WorkerPhase::Decode);
        if active.is_empty() {
            crate::overlap::run_pending();
            let pending = match engine.poll_idle_kv_transfers() {
                Ok(pending) => pending,
                Err(error) => {
                    eprintln!("generation worker idle KV poll failed: {error}");
                    state.store(FAILED, Ordering::Release);
                    return WorkerExit::Fatal;
                }
            };
            update_transfer_readiness(&engine, state);
            progress.idle();
            let received = if pending {
                receiver.recv_timeout(Duration::from_millis(1))
            } else {
                receiver.recv().map_err(|_| RecvTimeoutError::Disconnected)
            };
            match received {
                Ok(WorkerCommand::Shutdown) | Err(RecvTimeoutError::Disconnected) => {
                    progress.mark(WorkerPhase::Shutdown);
                    return shutdown_worker(&mut engine, &mut active);
                }
                Ok(WorkerCommand::StartupWarmup(begin, response)) => {
                    progress.mark(WorkerPhase::Warmup);
                    progress.warming(begin);
                    crate::overlap::run_pending();
                    let result = if begin {
                        engine.begin_startup_warmup()
                    } else {
                        cancel_all(&mut engine, &mut active);
                        engine.finish_startup_warmup()
                    };
                    let failed = result.is_err();
                    let _ = response.send(result);
                    if failed {
                        state.store(FAILED, Ordering::Release);
                        return WorkerExit::Fatal;
                    }
                }
                Ok(WorkerCommand::Submit(batch)) => submissions.push(batch),
                Err(RecvTimeoutError::Timeout) => continue,
            }
        }

        loop {
            match receiver.try_recv() {
                Ok(WorkerCommand::Shutdown) | Err(TryRecvError::Disconnected) => {
                    progress.mark(WorkerPhase::Shutdown);
                    return shutdown_worker(&mut engine, &mut active);
                }
                Ok(WorkerCommand::StartupWarmup(begin, response)) => {
                    progress.mark(WorkerPhase::Warmup);
                    progress.warming(begin);
                    crate::overlap::run_pending();
                    let result = if begin {
                        engine.begin_startup_warmup()
                    } else {
                        cancel_all(&mut engine, &mut active);
                        engine.finish_startup_warmup()
                    };
                    let failed = result.is_err();
                    let _ = response.send(result);
                    if failed {
                        state.store(FAILED, Ordering::Release);
                        return WorkerExit::Fatal;
                    }
                }
                Ok(WorkerCommand::Submit(batch)) => submissions.push(batch),
                Err(TryRecvError::Empty) => break,
            }
        }

        // Cancelled requests leave the engine before new ones enter it, so a
        // request takes the queue place a cancelled one has freed, as its
        // HTTP permit already has.
        let cancelled = active
            .iter()
            .filter_map(|(id, generation)| {
                (generation.cancellation.is_cancelled() || generation.events.is_closed())
                    .then_some(*id)
            })
            .collect::<Vec<_>>();
        for id in cancelled {
            active.remove(&id);
            if !cancel_live_request(&mut engine, id) {
                state.store(FAILED, Ordering::Release);
                fail_all(&mut engine, &mut active);
                return WorkerExit::Fatal;
            }
            let _ = engine.drain_events();
        }
        for batch in submissions.drain(..) {
            let group_admitted = Rc::new(Cell::new(false));
            let queued_since = Instant::now();
            let mut ids = Vec::with_capacity(batch.len());
            for submission in batch {
                let WorkerSubmission {
                    id,
                    request,
                    constraint,
                    events,
                    response,
                } = submission;
                // A request cancelled before the worker read it never enters the
                // engine; its client is gone and nobody reads the answer.
                if request.cancellation.is_cancelled() || events.is_closed() {
                    continue;
                }
                admit(
                    &tokenizer,
                    &mut engine,
                    &mut active,
                    id,
                    *request,
                    constraint,
                    events,
                    response,
                    tokenizer_input_bytes,
                    &policy,
                    deliver,
                );
                if let Some(generation) = active.get_mut(&id) {
                    generation.group_admitted = Rc::clone(&group_admitted);
                    generation.queued_since = queued_since;
                    ids.push(id);
                }
            }
            if engine.share_prompt_prefixes(&ids).is_err() {
                state.store(FAILED, Ordering::Release);
                fail_all(&mut engine, &mut active);
                return WorkerExit::Fatal;
            }
        }
        let expired = active
            .iter()
            .filter_map(|(id, generation)| {
                (generation.admission.is_some()
                    && !generation.group_admitted.get()
                    && !queue_wait.is_zero()
                    && generation.queued_since.elapsed() >= queue_wait
                    && matches!(engine.request_state(*id), Ok(RequestState::Queued)))
                .then_some(*id)
            })
            .collect::<Vec<_>>();
        for id in expired {
            if let Some(mut generation) = active.remove(&id) {
                if !cancel_live_request(&mut engine, id) {
                    state.store(FAILED, Ordering::Release);
                    fail_all(&mut engine, &mut active);
                    return WorkerExit::Fatal;
                }
                let _ = engine.drain_events();
                if let Some(response) = generation.admission.take() {
                    let _ = response.send(Err(AdmissionError::AtCapacity));
                }
            }
        }
        if active.is_empty() {
            continue;
        }

        // Until the step forms its batch (preemption, restore, drafting), a
        // request with no token yet may still need its prompt; the engine
        // then reports whether the batch prefills, a replay included.
        progress.mark(
            if active
                .values()
                .any(|generation: &ActiveGeneration| generation.completion_tokens == 0)
            {
                WorkerPhase::Prompt
            } else {
                WorkerPhase::Decode
            },
        );
        let step = engine.step();
        if progress.is_stalled() {
            // The watchdog has already failed this worker; a step that
            // returned late does not make it trusted again.
            fail_all(&mut engine, &mut active);
            return WorkerExit::Fatal;
        }
        // The last step's events that no device wait delivered go now,
        // before this step's.
        crate::overlap::run_pending();
        update_transfer_readiness(&engine, state);
        if let Err(error) = &step {
            eprintln!("generation engine step failed: {error:?}");
        }
        for (id, generation) in &mut active {
            if generation.admission.is_some()
                && !matches!(engine.request_state(*id), Ok(RequestState::Queued))
                && let Some(response) = generation.admission.take()
            {
                generation.group_admitted.set(true);
                let _ = response.send(Ok(()));
            }
        }
        let events = engine.drain_events();
        for event in events {
            if let EngineEventKind::Error { message } = event.kind() {
                eprintln!("generation request failed: {message}");
            }
            if !handle_engine_event(&tokenizer, &mut engine, &mut active, event) {
                state.store(FAILED, Ordering::Release);
                fail_all(&mut engine, &mut active);
                return WorkerExit::Fatal;
            }
        }
        match step {
            Ok(StepStatus::Progressed) => {}
            Ok(StepStatus::Idle) if active.is_empty() => {}
            // Sampling and constraint errors have already failed their requests
            // and emitted request errors. Executor and invariant errors leave
            // device or scheduler state uncertain, so they still stop this worker.
            Err(EngineError::Sampling(_) | EngineError::Constraint(_)) => {}
            Ok(StepStatus::Idle) | Err(_) => {
                eprintln!(
                    "generation worker stopping: {}",
                    stop_reason(&step, active.len())
                );
                state.store(FAILED, Ordering::Release);
                fail_all(&mut engine, &mut active);
                return WorkerExit::Fatal;
            }
        }
    }
}

/// Why a failed or stalled step stops the worker, for the one log line that
/// precedes the restart.
fn stop_reason(step: &Result<StepStatus, EngineError>, active: usize) -> String {
    match step {
        Err(EngineError::Executor(error)) => format!(
            "the model executor failed a batch and its requests failed; the worker stops \
             because device state after a failed execute is not trusted: {}",
            error.message()
        ),
        Err(error) => format!("the engine step failed: {error:?}"),
        Ok(_) => format!("the engine went idle with {active} request(s) still active"),
    }
}

#[allow(clippy::too_many_arguments)]
fn admit<E: ModelExecutor>(
    tokenizer: &Tokenizer,
    engine: &mut Engine<E>,
    active: &mut BTreeMap<RequestId, ActiveGeneration>,
    id: RequestId,
    mut request: PreparedRequest,
    constraint: Option<Arc<CompiledGrammar>>,
    events: EventSender,
    response: oneshot::Sender<Result<(), AdmissionError>>,
    tokenizer_input_bytes: usize,
    policy: &EngineModelPolicyV1,
    deliver: Delivery,
) {
    let prepared = std::mem::replace(&mut request.prompt, PreparedPrompt::TokenIds(Vec::new()));
    let (prompt, media, spans, positions) = match prepared {
        PreparedPrompt::MediaTokens {
            tokens,
            media,
            spans,
            positions,
        } => {
            if let Err(error) =
                admit_prepared_media_v1(policy.declaration().multimodal.as_ref(), &media)
            {
                let _ = response.send(Err(error));
                return;
            }
            let media = media
                .into_iter()
                .map(|item| EngineMediaInputV1 {
                    key: item.key,
                    encoder: item.encoder,
                    geometry: item.geometry,
                    output_rows: item.output_rows,
                    payload: EngineMediaPayloadV1::new(item.payload),
                })
                .collect::<Vec<_>>();
            let spans = spans
                .into_iter()
                .map(|span| EngineMediaSpanV1 {
                    media: span.media,
                    prompt_row: span.prompt_row,
                    rows: span.rows,
                    media_row: span.media_row,
                })
                .collect::<Vec<_>>();
            let positions = positions
                .into_iter()
                .map(|span| EnginePositionSpanV1 {
                    prompt_row: span.prompt_row,
                    rows: span.rows,
                    grid: span.grid,
                })
                .collect::<Vec<_>>();
            (tokens, media, spans, positions)
        }
        prompt => match prepare_owned_prompt(tokenizer, prompt, policy, tokenizer_input_bytes) {
            Ok(prompt) => (prompt, Vec::new(), Vec::new(), Vec::new()),
            Err(_) => {
                let _ = response.send(Err(AdmissionError::InvalidPrompt));
                return;
            }
        },
    };
    if request.logprobs.prompt_top.is_some() && !policy.prompt_rows {
        let _ = response.send(Err(AdmissionError::Rejected(
            "bundle does not support prompt logprobs".to_owned(),
        )));
        return;
    }
    let prompt_echo = (request.logprobs.prompt_top.is_some() || request.logprobs.return_token_ids)
        .then(|| prompt.clone());
    if request.sampling.top_logprobs() != request.logprobs.top {
        let _ = response.send(Err(AdmissionError::InvalidRequest));
        return;
    }
    let declaration = policy.declaration();
    let decoder = match completion_decoder_v1(tokenizer, policy, &request, &prompt) {
        Ok(decoder) => decoder,
        Err(error) => {
            let _ = response.send(Err(error));
            return;
        }
    };
    let chat_state = request
        .chat
        .as_ref()
        .and_then(|chat| BundleChatStateV1::from_state(&chat.state));
    let (stop_ids, ignored_stop_ids) = request_stop_ids_v1(policy, &request.decoding);
    let prompt_tokens = match u64::try_from(prompt.len()) {
        Ok(count) => count,
        Err(_) => {
            let _ = response.send(Err(AdmissionError::InvalidRequest));
            return;
        }
    };
    // A forced choice's constraint begins after the prefill the prompt ends
    // with, which the completion continues.
    let prefix = chat_state.map_or("", BundleChatStateV1::completion_prefix);
    let constraint = match constraint
        .map(|grammar| {
            let gate = constraint_gate_v1(chat_state, declaration);
            ConstraintState::new(grammar, gate, stop_ids.clone()).with_prefix(prefix.as_bytes())
        })
        .transpose()
    {
        Ok(constraint) => constraint,
        Err(_) => {
            let _ = response.send(Err(AdmissionError::InvalidRequest));
            return;
        }
    };
    let engine_request = match EngineRequest::new(
        id,
        prompt,
        request.max_new_tokens,
        request.sampling.clone(),
        request.seed,
        stop_ids,
        policy.vocabulary(),
    )
    .map(|request| request.with_ignored_stop_ids(&ignored_stop_ids))
    .and_then(|request| request.with_media(media, spans))
    .and_then(|request| request.with_position_axes(positions))
    .and_then(|request| match constraint {
        Some(constraint) => request.with_constraint(constraint),
        None => Ok(request),
    }) {
        Ok(engine_request) => {
            let engine_request = match &request.cache_salt {
                Some(salt) => engine_request.with_cache_salt(salt.as_bytes().to_vec()),
                None => engine_request,
            };
            let engine_request = match request.decoding.warmup_draft_tokens {
                Some(tokens) => engine_request.with_pinned_draft_tokens(tokens),
                None => engine_request,
            };
            if let Some(top) = request.logprobs.prompt_top {
                engine_request.with_prompt_logprobs(top)
            } else {
                engine_request
            }
        }
        Err(_) => {
            let _ = response.send(Err(AdmissionError::InvalidRequest));
            return;
        }
    };
    if let Err(error) = engine.submit(engine_request) {
        let refusal = match error {
            EngineError::Submit(runtime_scheduler::SubmitError::Overloaded(_)) => {
                AdmissionError::AtCapacity
            }
            EngineError::InvalidRequest(_) | EngineError::Submit(_) => {
                AdmissionError::InvalidRequest
            }
            _ => AdmissionError::WorkerUnavailable,
        };
        let _ = response.send(Err(refusal));
        return;
    }
    engine.metrics().set_request_trace_id(
        id,
        request
            .client_request_id
            .as_deref()
            .unwrap_or(&request.request_id),
    );
    engine.defer_completion_metrics(id);
    if let Some(ids) = prompt_echo {
        let tokens = ids
            .iter()
            .map(|id| {
                String::from_utf8_lossy(tokenizer.token_bytes(*id).expect("validated token"))
                    .into_owned()
            })
            .collect::<Vec<_>>();
        let text = tokenizer
            .decode_utf8(&ids)
            .unwrap_or_else(|_| tokens.concat());
        if events
            .try_emit(GenerationEvent::Prompt {
                token_ids: ids,
                text,
                tokens,
            })
            .is_err()
        {
            let _ = engine.cancel(id);
            let _ = response.send(Err(AdmissionError::WorkerUnavailable));
            return;
        }
    }
    active.insert(
        id,
        ActiveGeneration {
            events: Outbox::new(
                events,
                request.cancellation.clone(),
                deliver == Delivery::Deferred,
            ),
            cancellation: request.cancellation,
            decoder,
            prompt_tokens,
            cached_tokens: engine.prefix_cache_enabled().then_some(0),
            completion_tokens: 0,
            accounting: TokenAccounting::default(),
            logprob_context: [None; 4],
            prompt_logprob_context: [None; 4],
            logprobs_enabled: request.logprobs.enabled,
            return_token_ids: request.logprobs.return_token_ids,
            admission: Some(response),
            queued_since: Instant::now(),
            group_admitted: Rc::new(Cell::new(false)),
        },
    );
}

/// The single-token markers that end a reasoning section, as the
/// declaration names them: such as its close marker, and a tool-call open
/// marker that also ends it implicitly.
/// The tokens a chat's constraint waits for. A chat whose render left
/// reasoning open at the end of the prompt is constrained only after one of
/// the declared reasoning end tokens, as the reference gates it; any other
/// completion is constrained from its first token.
fn constraint_gate_v1(
    chat: Option<&BundleChatStateV1>,
    declaration: &FrontendDescriptorV1,
) -> Option<Vec<u32>> {
    chat.filter(|chat| chat.reasoning_open())
        .map(|_| reasoning_gate_tokens(declaration))
        .filter(|tokens| !tokens.is_empty())
}

fn reasoning_gate_tokens(declaration: &FrontendDescriptorV1) -> Vec<u32> {
    let markers = &declaration.reasoning.end_markers;
    declaration
        .vocabulary
        .boundary_tokens
        .iter()
        .filter(|token| markers.contains(&token.text))
        .map(|token| token.id)
        .collect()
}

/// The decoder one admitted request decodes its generated tokens with.
///
/// It reads the declared limits and the request's client stops. It parses
/// through the frontend that rendered the prompt, or the deployment's for a
/// completion without a chat, and applies the request's decoding options,
/// the prompt's bytes and the chat's tools and reasoning state. A frontend
/// check builds its decoder here too, so it decodes as admission does.
pub(crate) fn completion_decoder_v1(
    tokenizer: &Tokenizer,
    policy: &EngineModelPolicyV1,
    request: &PreparedRequest,
    prompt: &[u32],
) -> Result<CompletionDecoderV1, AdmissionError> {
    let declaration = policy.declaration();
    // The frontend and this backend read the same installed declaration, so
    // recovering the rendered prompt mode from the chat state is reading back
    // what that one declaration decided.
    let chat_state = request
        .chat
        .as_ref()
        .and_then(|chat| BundleChatStateV1::from_state(&chat.state));
    let mode = chat_state.map_or(PromptModeV1::Chat, BundleChatStateV1::mode);
    let limits = DecoderLimitsV1::new(
        declaration,
        declaration.limits.maximum_completion_bytes,
        request.max_new_tokens,
    )
    .map_err(|_| AdmissionError::InvalidRequest)?;
    let mut decoder = if request.stop.is_empty() {
        Ok(CompletionDecoderV1::new(
            Arc::clone(declaration),
            mode,
            limits,
        ))
    } else {
        CompletionDecoderV1::with_client_stops(Arc::clone(declaration), mode, limits, &request.stop)
    }
    .map_err(|_| AdmissionError::InvalidRequest)?;
    // The completion parses through the frontend that rendered its prompt,
    // and one that rendered nothing through the frontend the deployment
    // serves.
    if let Some(source) = chat_state
        .and_then(BundleChatStateV1::parser_source)
        .or(policy.parser_source())
    {
        decoder.set_parser_source(source.clone());
    }
    decoder.set_minimum_new_tokens(request.sampling.minimum_new_tokens());
    decoder.set_decoding_options(request.decoding.clone(), request.chat.is_none());
    let prompt_bytes = prompt.iter().try_fold(Vec::new(), |mut bytes, id| {
        if !request.decoding.skip_special_tokens || !tokenizer.token_is_special(*id)? {
            bytes.extend_from_slice(tokenizer.token_bytes(*id)?);
        }
        Ok::<_, runtime_tokenizer::TokenizerError>(bytes)
    });
    prompt_bytes
        .map_err(|error| error.to_string())
        .and_then(|bytes| decoder.prime_prompt_bytes(&bytes))
        .map_err(|_| AdmissionError::InvalidPrompt)?;
    if let Some(state) = chat_state {
        decoder.set_tool_schemas(state.tools());
        decoder.set_tool_extraction(state.extracts_tool_calls());
        // The completion begins in reasoning exactly when its render left
        // reasoning open at the end of the prompt.
        decoder.set_reasoning_open(state.reasoning_open());
        decoder
            .prime_completion_prefix(state.completion_prefix())
            .map_err(|_| AdmissionError::InvalidPrompt)?;
    }
    Ok(decoder)
}

/// The token identifiers the served declaration names as boundaries: the
/// markers its completion parser reads as structure.
pub(crate) fn boundary_token_ids_v1(policy: &EngineModelPolicyV1) -> Vec<u32> {
    policy
        .declaration()
        .vocabulary
        .boundary_tokens
        .iter()
        .map(|token| token.id)
        .collect()
}

/// The token ids that end one request's generation, sorted and unique, and
/// the declared ones among them that `ignore_eos` keeps generating past.
pub(crate) fn request_stop_ids_v1(
    policy: &EngineModelPolicyV1,
    decoding: &runtime_server::DecodingOptions,
) -> (Vec<u32>, Vec<u32>) {
    let mut stop_ids = decoding.stop_token_ids.clone();
    stop_ids.extend_from_slice(policy.stop_token_ids());
    let ignored_stop_ids = if decoding.ignore_eos {
        policy
            .stop_token_ids()
            .iter()
            .copied()
            .filter(|id| !decoding.stop_token_ids.contains(id))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    stop_ids.sort_unstable();
    stop_ids.dedup();
    (stop_ids, ignored_stop_ids)
}

/// One generated token as the decoder receives it: its bytes and whether
/// the tokenizer marks it special.
pub(crate) fn generated_token_v1(
    tokenizer: &Tokenizer,
    ordinal: u64,
    token_id: u32,
) -> Result<GeneratedTokenV1, runtime_tokenizer::TokenizerError> {
    let bytes = tokenizer.token_bytes(token_id)?.to_vec();
    let special = tokenizer.token_is_special(token_id)?;
    Ok(GeneratedTokenV1::new(ordinal, token_id, bytes).with_special(special))
}

fn tokenize_prompt(
    tokenizer: &Tokenizer,
    prompt: &PreparedPrompt,
    policy: &EngineModelPolicyV1,
    tokenizer_input_bytes: usize,
) -> Result<Vec<u32>, AdmissionError> {
    if let PreparedPrompt::MediaChat { segments, media } = prompt {
        return expand_media_chat_v1(
            policy
                .declaration()
                .multimodal
                .as_ref()
                .ok_or(AdmissionError::InvalidPrompt)?,
            &policy.declaration().vocabulary,
            segments,
            media,
            |text| {
                tokenizer
                    .encode_with_input_limit(
                        text,
                        &policy.prompt_special_tokens(),
                        tokenizer_input_bytes,
                    )
                    .map_err(|_| AdmissionError::InvalidPrompt)
            },
            |_part, _modality| {
                Err(AdmissionError::Rejected(
                    "media preprocessing is not available in this build".to_owned(),
                ))
            },
        )
        .map(|(tokens, ..)| tokens);
    }
    match prompt {
        // Raw prompts may already contain a chat template, just as token-ID
        // prompts may contain the declared boundary IDs.
        PreparedPrompt::Raw(prompt) | PreparedPrompt::RenderedChat(prompt) => {
            if prompt.len() > tokenizer_input_bytes {
                return Err(AdmissionError::InvalidPrompt);
            }
            tokenizer.encode_with_input_limit(
                prompt,
                &policy.prompt_special_tokens(),
                tokenizer_input_bytes,
            )
        }
        PreparedPrompt::TokenIds(ids) => {
            validate_token_ids(tokenizer, ids)?;
            Ok(ids.clone())
        }
        PreparedPrompt::MediaChat { .. } => unreachable!("handled above"),
        PreparedPrompt::MediaTokens { tokens, .. } => Ok(tokens.clone()),
    }
    .map_err(|_| AdmissionError::InvalidPrompt)
}

/// Splices trusted pad IDs into text encoded with the declared prompt tokens.
/// The probe closure is the only media dependency: runtime-media supplies it
/// when media decoding and recipe execution are installed.
#[derive(Clone)]
pub(crate) struct PreparedMediaInputV1 {
    pub(crate) input: MediaInputV1,
    /// (formatted timestamp, encoder rows) for each temporal grid.
    grids: Vec<(String, u32)>,
}

impl From<MediaInputV1> for PreparedMediaInputV1 {
    fn from(input: MediaInputV1) -> Self {
        Self {
            input,
            grids: Vec::new(),
        }
    }
}

type ExpandedMediaPromptV1 = (
    Vec<u32>,
    Vec<MediaInputV1>,
    Vec<MediaSpanV1>,
    Vec<PositionSpanV1>,
);

/// The merged grid `[t, h / merge, w / merge]` whose rows take positions
/// under the modality's rule, or `None` for implied 1D positions.
fn position_grid_v1(
    modality: &runtime_frontend::FrontendMediaModalityV1,
    input: &MediaInputV1,
) -> Result<Option<[u32; 3]>, AdmissionError> {
    match modality.positions {
        None => Ok(None),
        Some(FrontendPositionRuleV1::Grid3AxisV1) => {
            let merge = u32::try_from(media_u64_v1(&modality.recipe.parameters, "merge")?)
                .map_err(|_| AdmissionError::InvalidPrompt)?;
            let [t, h, w, _] = input.geometry;
            if merge == 0
                || modality.render.grid.is_some()
                || h % merge != 0
                || w % merge != 0
                || u64::from(t) * u64::from(h / merge) * u64::from(w / merge)
                    != u64::from(input.output_rows)
            {
                return Err(AdmissionError::InvalidPrompt);
            }
            Ok(Some([t, h / merge, w / merge]))
        }
    }
}

/// The limits one image or audio part is prepared under.
fn part_limits_v1(
    modality: &runtime_frontend::FrontendMediaModalityV1,
) -> runtime_media::MediaLimits {
    runtime_media::MediaLimits {
        maximum_bytes: modality.maximum_bytes,
        maximum_pixels: MAXIMUM_DECODED_IMAGE_PIXELS_V1,
        maximum_audio_frames: AUDIO_SAMPLE_RATE_V1 * AUDIO_MAXIMUM_SECONDS_V1,
        maximum_frames: VIDEO_MAXIMUM_FRAMES_V1,
        maximum_tokens: u64::from(modality.maximum_tokens),
        maximum_aspect_ratio: modality
            .recipe
            .parameters
            .get("maximum_aspect_ratio")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(MAXIMUM_ASPECT_RATIO_V1),
    }
}

/// The limits one video part is prepared under; `image` is its recipe's
/// frame recipe.
fn video_limits_v1(
    modality: &runtime_frontend::FrontendMediaModalityV1,
    image: &serde_json::Value,
) -> runtime_media::MediaLimits {
    runtime_media::MediaLimits {
        maximum_bytes: modality.maximum_bytes,
        maximum_pixels: MAXIMUM_DECODED_FRAME_PIXELS_V1,
        maximum_audio_frames: 0,
        maximum_frames: VIDEO_MAXIMUM_FRAMES_V1,
        maximum_tokens: u64::from(modality.maximum_tokens),
        maximum_aspect_ratio: image
            .get("maximum_aspect_ratio")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(MAXIMUM_ASPECT_RATIO_V1),
    }
}

/// Whether frames of `height x width` patches are a size the recipe
/// resizes some source to (`runtime_media::recipe::emits_size`).
fn emits_patches_v1(
    height: u32,
    width: u32,
    recipe: runtime_media::recipe::ImageRecipe,
    limits: runtime_media::MediaLimits,
) -> bool {
    match (
        height.checked_mul(recipe.patch),
        width.checked_mul(recipe.patch),
    ) {
        (Some(height), Some(width)) => {
            runtime_media::recipe::emits_size(height, width, recipe, limits)
        }
        _ => false,
    }
}

/// Whether preparing a part of `modality` can produce a record of
/// `geometry`: exactly the outputs of the modality's recipe under the
/// limits a part is prepared with, its token limit included.
fn modality_emits_v1(
    modality: &runtime_frontend::FrontendMediaModalityV1,
    geometry: [u32; 4],
) -> bool {
    use runtime_frontend::FrontendMediaKindV1;
    let parameters = &modality.recipe.parameters;
    let tokens_fit = |tokens: u64| tokens <= u64::from(modality.maximum_tokens);
    match modality.kind {
        FrontendMediaKindV1::Image => {
            let Ok(recipe) = declared_image_recipe_v1(&modality.recipe) else {
                return false;
            };
            let limits = part_limits_v1(modality);
            let [t, h, w, rest] = geometry;
            let (temporal_patch, merge, emits) = match recipe {
                ImageRecipeV1::PatchGrid(recipe) => (
                    recipe.temporal_patch,
                    recipe.merge,
                    emits_patches_v1(h, w, recipe, limits),
                ),
                ImageRecipeV1::TokenBudget(recipe, _) => (
                    recipe.temporal_patch,
                    recipe.merge,
                    match (h.checked_mul(recipe.patch), w.checked_mul(recipe.patch)) {
                        (Some(h), Some(w)) => {
                            runtime_media::recipe::padded_emits_size(h, w, recipe, limits)
                        }
                        _ => false,
                    },
                ),
                // A superset below the recipe's minimum pixels: the
                // upscale's truncation and the solve's rounding emit small
                // canvases that are not their own plan and have no closed
                // form (`centered_pad_emits_size`); every admitted canvas is
                // still aligned, within the token budget and the pixels.
                ImageRecipeV1::CenteredPad(recipe, _) => (
                    1,
                    recipe.downsample,
                    match (h.checked_mul(recipe.patch), w.checked_mul(recipe.patch)) {
                        (Some(h), Some(w)) => {
                            runtime_media::recipe::centered_pad_emits_size(h, w, recipe, limits)
                        }
                        _ => false,
                    },
                ),
            };
            t == 1
                && rest == 0
                && temporal_patch as usize <= limits.maximum_frames
                && emits
                && match modality.layout {
                    None => tokens_fit(u64::from(h / merge) * u64::from(w / merge)),
                    Some(_) => {
                        image_span_rows_v1(modality, merge, [1, h, w]).is_some_and(tokens_fit)
                    }
                }
        }
        FrontendMediaKindV1::Video => {
            let (Some(image), Ok(total)) = (
                parameters.get("image"),
                media_u64_v1(parameters, "total_max_pixels"),
            ) else {
                return false;
            };
            let Ok(recipe) = image_recipe_v1(image) else {
                return false;
            };
            let limits = video_limits_v1(modality, image);
            let [t, h, w, rest] = geometry;
            let patch = recipe.temporal_patch as usize;
            // `t` grids come from one more than `t - 1` temporal patches of
            // frames up to `t` of them, the last repeated if short.
            let first = (t as usize).saturating_sub(1) * patch + 1;
            let last = (t as usize * patch).min(limits.maximum_frames);
            let grid = u64::from(h / recipe.merge.max(1)) * u64::from(w / recipe.merge.max(1));
            t >= 1
                && rest == 0
                && tokens_fit(u64::from(t) * grid)
                && (first..=last).any(|frames| {
                    let frame = runtime_media::recipe::video_frame_recipe(recipe, total, frames);
                    emits_patches_v1(h, w, frame, limits)
                })
        }
        FrontendMediaKindV1::Audio => {
            let Ok(recipe) = audio_recipe_v1(parameters) else {
                return false;
            };
            let limits = part_limits_v1(modality);
            let [frames, bins, c, rest] = geometry;
            bins as usize == recipe.mel_bins
                && c == 0
                && rest == 0
                && runtime_media::audio::emitted_frames(recipe, limits).contains(&u64::from(frames))
                && runtime_media::audio::strided_group_tokens(frames as usize, recipe)
                    .is_ok_and(|tokens| tokens_fit(tokens as u64))
        }
    }
}

/// Row count preparing a part with this geometry produces, including any
/// delimiters. Geometry validity is checked separately by `modality_emits_v1`.
fn modality_output_rows_v1(
    modality: &runtime_frontend::FrontendMediaModalityV1,
    geometry: [u32; 4],
) -> Option<u64> {
    use runtime_frontend::FrontendMediaKindV1;
    let parameters = &modality.recipe.parameters;
    let [t, h, w, _] = geometry;
    match modality.kind {
        FrontendMediaKindV1::Image => {
            let recipe = declared_image_recipe_v1(&modality.recipe).ok()?;
            image_span_rows_v1(modality, recipe_merge_v1(recipe), [t, h, w])
        }
        FrontendMediaKindV1::Video => {
            let recipe = image_recipe_v1(parameters.get("image")?).ok()?;
            let rows =
                u64::from(h.checked_div(recipe.merge)?) * u64::from(w.checked_div(recipe.merge)?);
            u64::from(t).checked_mul(rows)
        }
        FrontendMediaKindV1::Audio => {
            let recipe = audio_recipe_v1(parameters).ok()?;
            runtime_media::audio::strided_group_tokens(t as usize, recipe)
                .ok()
                .map(|rows| rows as u64)
        }
    }
}

/// Refuses prepared media whose geometry or row count its encoder's
/// modalities cannot produce, before it reaches the engine.
fn admit_prepared_media_v1(
    declared: Option<&FrontendMultimodalV1>,
    media: &[MediaInputV1],
) -> Result<(), AdmissionError> {
    if media.is_empty() {
        return Ok(());
    }
    let declared = declared.ok_or_else(|| {
        AdmissionError::Rejected("media inputs are not supported by this model".to_owned())
    })?;
    for (index, input) in media.iter().enumerate() {
        let admitted = declared
            .modalities
            .iter()
            .filter(|modality| modality.encoder == input.encoder)
            .any(|modality| {
                modality_emits_v1(modality, input.geometry)
                    && input.output_rows <= modality.maximum_tokens
                    && modality_output_rows_v1(modality, input.geometry)
                        == Some(u64::from(input.output_rows))
            });
        if !admitted {
            return Err(AdmissionError::Rejected(format!(
                "media item {index}: geometry {:?} with {} output rows for encoder {} is not one its media \
                 recipes produce",
                input.geometry, input.output_rows, input.encoder
            )));
        }
    }
    Ok(())
}

const VIDEO_MAXIMUM_FRAMES_V1: usize = 32;
const VIDEO_SECONDS_PER_FRAME_V1: u32 = 1;

/// Refuse declarations the implemented recipe versions cannot honor.
pub fn validate_media_recipes_v1(descriptor: &FrontendDescriptorV1) -> Result<(), String> {
    match &descriptor.multimodal {
        Some(media) => validate_multimodal_recipes_v1(media),
        None => Ok(()),
    }
}

fn validate_multimodal_recipes_v1(media: &FrontendMultimodalV1) -> Result<(), String> {
    let has = |matches: &dyn Fn(&runtime_frontend::FrontendMediaModalityV1) -> bool| {
        media.modalities.iter().any(matches)
    };
    for tag in [TOKEN_BUDGET_TAG_V1, CENTERED_PAD_TAG_V1] {
        if has(&|modality| modality.recipe.tag == tag)
            && has(&|modality| modality.kind == runtime_frontend::FrontendMediaKindV1::Video)
        {
            return Err(format!(
                "{tag} prepares single images and has no video form: video is not supported \
                 alongside it"
            ));
        }
    }
    for modality in &media.modalities {
        validate_media_recipe_v1(&modality.recipe)?;
        validate_recipe_outputs_v1(modality)?;
    }
    Ok(())
}

/// Refuses an image or video recipe under which a size it emits is not its
/// own resize, at every frame budget a video can take, so that prepared
/// media is admitted exactly as parts emit it (`modality_emits_v1`).
fn validate_recipe_outputs_v1(
    modality: &runtime_frontend::FrontendMediaModalityV1,
) -> Result<(), String> {
    use runtime_frontend::FrontendMediaKindV1;
    use runtime_media::recipe::{check_emits_own_resizes, video_frame_recipe};
    let parameters = &modality.recipe.parameters;
    let tag = &modality.recipe.tag;
    let unreadable = |_| format!("{tag}: the image recipe parameters cannot be read");
    match modality.kind {
        FrontendMediaKindV1::Image => match declared_image_recipe_v1(&modality.recipe) {
            Ok(ImageRecipeV1::PatchGrid(recipe)) => {
                check_emits_own_resizes(recipe, part_limits_v1(modality))
                    .map_err(|error| format!("{tag}: {error}"))
            }
            Ok(ImageRecipeV1::TokenBudget(recipe, _)) => {
                validate_token_budget_outputs_v1(modality, recipe)
                    .map_err(|error| format!("{tag}: {error}"))
            }
            Ok(ImageRecipeV1::CenteredPad(recipe, _)) => {
                validate_centered_pad_outputs_v1(modality, recipe)
                    .map_err(|error| format!("{tag}: {error}"))
            }
            Err(error) => Err(unreadable(error)),
        },
        FrontendMediaKindV1::Video => {
            let image = parameters
                .get("image")
                .ok_or_else(|| format!("{tag}: no image recipe"))?;
            let recipe = image_recipe_v1(image).map_err(unreadable)?;
            let total = media_u64_v1(parameters, "total_max_pixels")
                .map_err(|_| format!("{tag}: no total_max_pixels"))?;
            let limits = video_limits_v1(modality, image);
            for frames in 1..=limits.maximum_frames {
                check_emits_own_resizes(video_frame_recipe(recipe, total, frames), limits)
                    .map_err(|error| format!("{tag} at {frames} frames: {error}"))?;
            }
            Ok(())
        }
        FrontendMediaKindV1::Audio => Ok(()),
    }
}

/// Refuses a token-budget recipe the host cannot prepare every image of:
/// its temporal patch above the host's frame limit, or its largest canvas
/// above the decoded pixel limit or the modality's token limit.
fn validate_token_budget_outputs_v1(
    modality: &runtime_frontend::FrontendMediaModalityV1,
    recipe: runtime_media::recipe::TokenBudgetRecipe,
) -> Result<(), String> {
    runtime_media::recipe::check_token_budget_recipe(recipe).map_err(|error| error.to_string())?;
    let limits = part_limits_v1(modality);
    if recipe.temporal_patch as usize > limits.maximum_frames {
        return Err(format!(
            "temporal_patch {} exceeds the host's {} frames",
            recipe.temporal_patch, limits.maximum_frames
        ));
    }
    if recipe.maximum_tokens > u64::from(modality.maximum_tokens) {
        return Err(format!(
            "maximum_tokens {} exceeds the modality's {} tokens",
            recipe.maximum_tokens, modality.maximum_tokens
        ));
    }
    // `check_token_budget_recipe` bounds the canvas pixels below 2^53.
    let block = u64::from(recipe.patch) * u64::from(recipe.merge);
    let canvas = recipe.maximum_tokens * block * block;
    if canvas > limits.maximum_pixels {
        return Err(format!(
            "a canvas of {canvas} pixels exceeds the host's {} decoded pixels",
            limits.maximum_pixels
        ));
    }
    Ok(())
}

/// Refuses a centered-pad recipe the host cannot prepare every image of
/// as declared: one the recipe check refuses, a modality without learned
/// delimiters (the budget counts their rows, `rows * (columns + 1) + 2`),
/// a budget above the modality's token limit, or a largest canvas above
/// the decoded pixel limit. A canvas within the budget has at most
/// `maximum_tokens - 3` cells of `(patch * downsample)^2` pixels.
fn validate_centered_pad_outputs_v1(
    modality: &runtime_frontend::FrontendMediaModalityV1,
    recipe: runtime_media::recipe::CenteredPadRecipe,
) -> Result<(), String> {
    runtime_media::recipe::check_centered_pad_recipe(recipe).map_err(|error| error.to_string())?;
    modality.validate_centered_pad_v1()
}

/// Refuses token-budget parameters it does not read, a missing or unknown
/// alpha policy, and a supplied aspect limit that is not a finite number of
/// at least 1 (an omitted one is the host default).
fn validate_token_budget_parameters_v1(parameters: &serde_json::Value) -> Result<(), String> {
    validate_padded_parameters_v1(TOKEN_BUDGET_TAG_V1, &TOKEN_BUDGET_PARAMETERS_V1, parameters)
}

/// Refuses parameters of `tag` outside `known`, a missing or unknown alpha
/// policy, and a supplied aspect limit that is not a finite number of at
/// least 1 (an omitted one is the host default).
fn validate_padded_parameters_v1(
    tag: &str,
    known: &[&str],
    parameters: &serde_json::Value,
) -> Result<(), String> {
    // The resize, padding and interpolation are fixed: an annotation that
    // would describe another would be silently ignored.
    let unknown = parameters
        .as_object()
        .and_then(|parameters| parameters.keys().find(|key| !known.contains(&key.as_str())));
    if let Some(key) = unknown {
        return Err(format!("{tag} has no parameter {key}"));
    }
    if declared_alpha_v1(parameters).is_none() {
        return Err(format!(
            "{tag} requires alpha_background \"white\" (composite) or \"none\" (discard)"
        ));
    }
    if let Some(ratio) = parameters.get("maximum_aspect_ratio")
        && !ratio
            .as_f64()
            .is_some_and(|ratio| ratio.is_finite() && ratio >= 1.0)
    {
        return Err(format!(
            "{tag} requires maximum_aspect_ratio to be a finite number of at least 1, \
             declared {ratio}"
        ));
    }
    Ok(())
}

/// The declared `alpha_background` of a padded recipe: "white"
/// composites transparency onto white, "none" discards alpha as PIL's
/// `convert("RGB")` does.
fn declared_alpha_v1(parameters: &serde_json::Value) -> Option<runtime_media::Alpha> {
    match parameters.get("alpha_background")?.as_str()? {
        "white" => Some(runtime_media::Alpha::CompositeWhite),
        "none" => Some(runtime_media::Alpha::Discard),
        _ => None,
    }
}

fn validate_media_recipe_v1(
    recipe: &runtime_frontend::FrontendMediaRecipeV1,
) -> Result<(), String> {
    let parameters = &recipe.parameters;
    let supported = match recipe.tag.as_str() {
        "audio.log-mel.v1" => runtime_frontend::audio_log_mel_v1_fixed_parameters(),
        "video.timestamped-grids.v1" => serde_json::json!({
            "maximum_frames": VIDEO_MAXIMUM_FRAMES_V1,
            "sampling": "uniform", "repeat_last_if_odd": true, "timestamp": "mm:ss"
        }),
        "image.patch-grid.v1" => {
            return validate_image_fixed_parameters_v1(parameters);
        }
        TOKEN_BUDGET_TAG_V1 => return validate_token_budget_parameters_v1(parameters),
        CENTERED_PAD_TAG_V1 => return recipe.validate_centered_pad_v1(),
        _ => return Ok(()),
    };
    for (key, expected) in supported.as_object().expect("object literal") {
        require_media_value_v1(&recipe.tag, parameters, key, expected, true)?;
    }
    if recipe.tag == "video.timestamped-grids.v1" {
        // V1 declares one second per sampled frame implicitly when omitted.
        require_media_value_v1(
            &recipe.tag,
            parameters,
            "seconds_per_frame",
            &serde_json::json!(VIDEO_SECONDS_PER_FRAME_V1),
            false,
        )?;
        let image = parameters.get("image").ok_or("video recipe needs image")?;
        for (outer, inner) in [
            ("minimum_pixels_per_frame", "minimum_pixels"),
            ("maximum_pixels_per_frame", "maximum_pixels"),
        ] {
            let expected = image
                .get(inner)
                .ok_or_else(|| format!("video image needs {inner}"))?;
            require_media_value_v1(&recipe.tag, parameters, outer, expected, true)?;
        }
        validate_image_fixed_parameters_v1(image)?;
    }
    Ok(())
}

fn require_media_value_v1(
    tag: &str,
    parameters: &serde_json::Value,
    key: &str,
    expected: &serde_json::Value,
    required: bool,
) -> Result<(), String> {
    let Some(actual) = parameters.get(key) else {
        return if required {
            Err(format!("{tag} requires {key}={expected}"))
        } else {
            Ok(())
        };
    };
    let equal = if actual.is_number() && expected.is_number() {
        actual.as_f64() == expected.as_f64()
    } else {
        actual == expected
    };
    if !equal {
        return Err(format!(
            "{tag} requires {key}={expected}, declared {actual}"
        ));
    }
    Ok(())
}

fn validate_image_fixed_parameters_v1(parameters: &serde_json::Value) -> Result<(), String> {
    // These optional annotations describe behavior already fixed by the decoder
    // and interpolation recipe. Declarations may omit them, but cannot change it.
    let antialias = parameters
        .get("interpolation")
        .and_then(serde_json::Value::as_str)
        == Some("bicubic-antialias");
    let small_side = if parameters
        .get("upscale_small_side")
        .and_then(serde_json::Value::as_bool)
        == Some(false)
    {
        serde_json::json!(0)
    } else {
        parameters
            .get("factor")
            .cloned()
            .unwrap_or(serde_json::Value::Null)
    };
    for (key, expected) in [
        ("antialias", serde_json::json!(antialias)),
        ("alpha_background", serde_json::json!("white")),
        ("exif_orientation", serde_json::json!(true)),
        ("small_side_minimum", small_side),
    ] {
        require_media_value_v1("image.patch-grid.v1", parameters, key, &expected, false)?;
    }
    Ok(())
}

fn media_u64_v1(parameters: &serde_json::Value, key: &str) -> Result<u64, AdmissionError> {
    parameters
        .get(key)
        .and_then(|value| {
            value.as_u64().or_else(|| {
                let number = value.as_f64()?;
                (number.is_finite()
                    && number >= 0.0
                    && number < u64::MAX as f64
                    && number.fract() == 0.0)
                    .then_some(number as u64)
            })
        })
        .ok_or(AdmissionError::InvalidPrompt)
}

/// The host could not allocate a media buffer: a service condition, retried
/// later, rather than a crash.
fn payload_allocation_failed_v1() -> AdmissionError {
    AdmissionError::Unavailable("host memory for the media payload is exhausted".to_owned())
}

fn media_values_to_bytes_v1(
    values: Vec<f32>,
    element: runtime_frontend::FrontendMediaElementV1,
) -> Result<Vec<u8>, AdmissionError> {
    let width = match element {
        runtime_frontend::FrontendMediaElementV1::Bf16 => 2,
        runtime_frontend::FrontendMediaElementV1::F32 => 4,
    };
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(values.len() * width)
        .map_err(|_| payload_allocation_failed_v1())?;
    match element {
        runtime_frontend::FrontendMediaElementV1::Bf16 => {
            bytes.extend(values.into_iter().flat_map(|value| {
                let bits = value.to_bits();
                let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
                ((rounded >> 16) as u16).to_le_bytes()
            }));
        }
        runtime_frontend::FrontendMediaElementV1::F32 => {
            bytes.extend(values.into_iter().flat_map(f32::to_le_bytes));
        }
    }
    Ok(bytes)
}

/// A video the decoder refused is the client's fault; a decode the service
/// could not run or finish in time is the service's.
fn video_admission_error_v1(error: crate::video::VideoErrorV1) -> AdmissionError {
    use crate::video::VideoErrorV1;
    match error {
        VideoErrorV1::Unavailable(message) => AdmissionError::Unavailable(message),
        error @ (VideoErrorV1::Busy(_) | VideoErrorV1::TimedOut(_) | VideoErrorV1::Cancelled) => {
            AdmissionError::Unavailable(error.to_string())
        }
        error => AdmissionError::Rejected(error.to_string()),
    }
}

/// Largest decoded image, before the recipe resizes it to its own pixel
/// budget. vLLM resizes any image PIL opens, and PIL opens up to twice its
/// `MAX_IMAGE_PIXELS` (warning above it); this is `MAX_IMAGE_PIXELS`, which
/// bounds a decode at 256 MiB of RGB. The JPEG decoder keeps its own lower
/// hard limit.
const MAXIMUM_DECODED_IMAGE_PIXELS_V1: u64 = runtime_frontend::MAXIMUM_DECODED_IMAGE_PIXELS_V1;

/// Largest decoded video frame, the JPEG decoder's hard limit.
const MAXIMUM_DECODED_FRAME_PIXELS_V1: u64 = 1 << 24;

/// vLLM's resize refuses aspect ratios above 200 and accepts the rest.
const MAXIMUM_ASPECT_RATIO_V1: f64 = 200.0;

/// Source frames are checked separately from the recipe's output frames.
fn audio_source_limits_v1(
    bytes: &[u8],
    sample_rate: u32,
    limits: runtime_media::MediaLimits,
) -> Result<runtime_media::MediaLimits, AdmissionError> {
    limits
        .check_bytes(bytes.len())
        .map_err(|e| AdmissionError::Rejected(e.to_string()))?;
    let info =
        runtime_media::wav::probe(bytes).map_err(|e| AdmissionError::Rejected(e.to_string()))?;
    if sample_rate == 0 {
        return Err(AdmissionError::InvalidPrompt);
    }
    let frames = u128::from(limits.maximum_audio_frames) * u128::from(info.sample_rate)
        / u128::from(sample_rate);
    Ok(runtime_media::MediaLimits {
        maximum_audio_frames: u64::try_from(frames).map_err(|_| AdmissionError::InvalidPrompt)?,
        ..limits
    })
}

/// Reserve decoded samples, resampled samples, FFT/filter workspace, features
/// and payload conversion before allocating them. Source reads reserve their
/// byte bound earlier; callers with an existing source can reserve it here.
fn prepare_audio_v1(
    source: &runtime_media::MediaSource,
    recipe: runtime_media::audio::AudioRecipe,
    limits: runtime_media::MediaLimits,
    cancelled: &dyn Fn() -> bool,
    source_reserved: u64,
    memory: &mut RequestMediaMemoryV1,
) -> Result<(runtime_media::audio::LogMel, u64), AdmissionError> {
    if cancelled() {
        return Err(AdmissionError::Unavailable(
            "audio preparation cancelled".into(),
        ));
    }
    let rejected = |e: runtime_media::MediaError| AdmissionError::Rejected(e.to_string());
    let source_limits = audio_source_limits_v1(&source.bytes, recipe.sample_rate, limits)?;
    let info = runtime_media::wav::probe(&source.bytes).map_err(rejected)?;
    source_limits.check_audio(info.frames).map_err(rejected)?;
    let samples =
        (info.frames * u64::from(recipe.sample_rate)).div_ceil(u64::from(info.sample_rate));
    limits.check_audio(samples).map_err(rejected)?;
    if recipe.hop == 0 {
        return Err(AdmissionError::InvalidPrompt);
    }
    let frames = samples / recipe.hop as u64 + 1;
    let tokens =
        runtime_media::audio::strided_group_tokens(frames as usize, recipe).map_err(rejected)?;
    limits.check_tokens(tokens as u64).map_err(rejected)?;
    // The workspace includes dense filters while they are converted to sparse
    // filters, cached twiddles, and the transform's reusable scratch arrays.
    let bytes = u128::from(source_reserved.max(source.bytes.capacity() as u64))
        + 4 * (u128::from(info.frames) + u128::from(samples))
        + 8 * u128::from(frames) * recipe.mel_bins as u128
        + 512 * recipe.n_fft as u128
        + recipe.mel_bins as u128 * (16 * (recipe.n_fft as u128 / 2 + 1) + 128)
        + 4096;
    let reserved = u64::try_from(bytes).map_err(|_| AdmissionError::InvalidPrompt)?;
    memory.reserve(reserved - source_reserved)?;
    let result = runtime_media::wav::decode_cancellable(&source.bytes, source_limits, cancelled)
        .and_then(|decoded| {
            runtime_media::audio::log_mel_cancellable(&decoded, recipe, limits, cancelled)
        });
    let mel = result.map_err(|error| {
        if cancelled() {
            AdmissionError::Unavailable("audio preparation cancelled".into())
        } else {
            rejected(error)
        }
    })?;
    Ok((mel, reserved))
}

/// Reserves `memory` for an image's preparation from its header before it
/// decodes, then decodes it and turns it into patch rows. Returns the grid and
/// the bytes reserved.
fn prepare_image_v1(
    source: &runtime_media::MediaSource,
    recipe: impl Into<ImageRecipeV1>,
    limits: runtime_media::MediaLimits,
    memory: &mut RequestMediaMemoryV1,
) -> Result<(runtime_media::recipe::PatchGrid, u64), AdmissionError> {
    let recipe = recipe.into();
    let rejected = |error: runtime_media::MediaError| AdmissionError::Rejected(error.to_string());
    let orientation = image_orientation_v1(recipe);
    let (width, height) = match runtime_media::probe_oriented(&source.bytes, limits, orientation)
        .map_err(rejected)?
    {
        runtime_media::Probe::Png { width, height }
        | runtime_media::Probe::Jpeg { width, height } => (width, height),
        runtime_media::Probe::Wav { .. } => return Err(AdmissionError::InvalidPrompt),
    };
    let bytes = match recipe {
        ImageRecipeV1::PatchGrid(recipe) => {
            runtime_media::recipe::image_preparation_bytes(width, height, recipe, limits)
        }
        ImageRecipeV1::TokenBudget(recipe, _) => {
            runtime_media::recipe::padded_preparation_bytes(width, height, recipe, limits)
        }
        ImageRecipeV1::CenteredPad(recipe, _) => {
            runtime_media::recipe::centered_pad_preparation_bytes(&source.bytes, recipe, limits)
        }
    };
    let reserved = bytes
        .map_err(rejected)?
        .saturating_add(source.bytes.len() as u64);
    memory.reserve(reserved)?;
    let alpha = match recipe {
        ImageRecipeV1::PatchGrid(_) => runtime_media::Alpha::CompositeWhite,
        ImageRecipeV1::TokenBudget(_, alpha) | ImageRecipeV1::CenteredPad(_, alpha) => alpha,
    };
    let decoded = decode_image_v1(source, limits, alpha, orientation)?;
    let grid = match recipe {
        ImageRecipeV1::PatchGrid(recipe) => {
            runtime_media::recipe::image_grid(&decoded, recipe, limits)
        }
        ImageRecipeV1::TokenBudget(recipe, _) => {
            runtime_media::recipe::padded_token_budget_grid(&decoded, recipe, limits)
        }
        ImageRecipeV1::CenteredPad(recipe, _) => {
            runtime_media::recipe::centered_pad_grid(&decoded, recipe, limits)
        }
    };
    Ok((grid.map_err(rejected)?, reserved))
}

/// The EXIF orientation policy of a recipe's JPEG sources: the centered
/// pad keeps the stored pixels, as its reference's `Image.open` and
/// `convert("RGB")` do; the others transpose into the tagged orientation.
fn image_orientation_v1(recipe: ImageRecipeV1) -> runtime_media::Orientation {
    match recipe {
        ImageRecipeV1::CenteredPad(..) => runtime_media::Orientation::Stored,
        ImageRecipeV1::PatchGrid(_) | ImageRecipeV1::TokenBudget(..) => {
            runtime_media::Orientation::Exif
        }
    }
}

/// Decodes by MIME type; `alpha` applies to PNG transparency (JPEG has
/// none), `orientation` to a JPEG's EXIF orientation.
fn decode_image_v1(
    source: &runtime_media::MediaSource,
    limits: runtime_media::MediaLimits,
    alpha: runtime_media::Alpha,
    orientation: runtime_media::Orientation,
) -> Result<runtime_media::RgbImage, AdmissionError> {
    match source.mime.as_str() {
        "image/png" => runtime_media::png::decode_with_alpha(&source.bytes, limits, alpha),
        "image/jpeg" => runtime_media::jpeg::decode_oriented(&source.bytes, limits, orientation),
        _ => return Err(AdmissionError::InvalidPrompt),
    }
    .map_err(|error| AdmissionError::Rejected(error.to_string()))
}

/// The tag of images fit into a zero-padded canvas within a token budget
/// (`runtime_media::recipe::padded_token_budget_grid`).
pub const TOKEN_BUDGET_TAG_V1: &str = "image.zero-padded-token-budget.v1";

/// Every parameter [`TOKEN_BUDGET_TAG_V1`] reads, `maximum_aspect_ratio`
/// through the part limits.
const TOKEN_BUDGET_PARAMETERS_V1: [&str; 10] = [
    "alpha_background",
    "patch",
    "temporal_patch",
    "merge",
    "minimum_tokens",
    "maximum_tokens",
    "value_scale",
    "mean",
    "std",
    "maximum_aspect_ratio",
];

/// The tag of images fitted at their aspect ratio into a patch-aligned
/// canvas within a delimited token budget and centered on a constant color
/// (`runtime_media::recipe::centered_pad_grid`).
pub const CENTERED_PAD_TAG_V1: &str = "image.centered-pad.v1";

/// An image modality's recipe, by its tag. The patch grid composites
/// transparency onto white; the padded recipes handle it as declared.
#[derive(Clone, Copy, Debug)]
enum ImageRecipeV1 {
    PatchGrid(runtime_media::recipe::ImageRecipe),
    TokenBudget(
        runtime_media::recipe::TokenBudgetRecipe,
        runtime_media::Alpha,
    ),
    CenteredPad(
        runtime_media::recipe::CenteredPadRecipe,
        runtime_media::Alpha,
    ),
}

impl From<runtime_media::recipe::ImageRecipe> for ImageRecipeV1 {
    fn from(recipe: runtime_media::recipe::ImageRecipe) -> Self {
        ImageRecipeV1::PatchGrid(recipe)
    }
}

/// The rows an image part's span takes, from its patch grid `[t, h, w]`:
/// the merged cells under the modality's layout
/// ([`runtime_frontend::FrontendMediaModalityV1::span_rows`]). A partial
/// merge block is a cell, as the encoder pads it. The encoder writes every
/// row of the span, delimiters included, so the span stays one pad run.
fn image_span_rows_v1(
    modality: &runtime_frontend::FrontendMediaModalityV1,
    merge: u32,
    grid: [u32; 3],
) -> Option<u64> {
    let [t, h, w] = grid;
    if merge == 0 {
        return None;
    }
    modality.span_rows([
        u64::from(t),
        u64::from(h.div_ceil(merge)),
        u64::from(w.div_ceil(merge)),
    ])
}

fn recipe_merge_v1(recipe: ImageRecipeV1) -> u32 {
    match recipe {
        ImageRecipeV1::PatchGrid(recipe) => recipe.merge,
        ImageRecipeV1::TokenBudget(recipe, _) => recipe.merge,
        ImageRecipeV1::CenteredPad(recipe, _) => recipe.downsample,
    }
}

/// The patch grid an image of `width x height` takes under `recipe`, as its
/// preparation will make it, planned from the size alone.
fn planned_image_grid_v1(
    width: u32,
    height: u32,
    recipe: ImageRecipeV1,
    limits: runtime_media::MediaLimits,
) -> runtime_media::MediaResult<[u32; 3]> {
    let (h, w, patch) = match recipe {
        ImageRecipeV1::PatchGrid(recipe) => {
            let (h, w) = runtime_media::recipe::smart_resize(height, width, recipe, limits)?;
            (h, w, recipe.patch)
        }
        ImageRecipeV1::TokenBudget(recipe, _) => {
            let size = runtime_media::recipe::padded_size(height, width, recipe, limits)?;
            (size.canvas.0, size.canvas.1, recipe.patch)
        }
        ImageRecipeV1::CenteredPad(recipe, _) => {
            let plan = runtime_media::recipe::centered_pad_plan(height, width, recipe, limits)?;
            (plan.canvas.0, plan.canvas.1, recipe.patch)
        }
    };
    Ok([1, h / patch, w / patch])
}

/// Refuses, before the image decodes, a part whose span under the
/// modality's layout (its cells and the layout's own rows) would exceed the
/// modality's token limit.
fn check_image_span_v1(
    source: &runtime_media::MediaSource,
    modality: &runtime_frontend::FrontendMediaModalityV1,
    recipe: ImageRecipeV1,
    limits: runtime_media::MediaLimits,
) -> Result<(), AdmissionError> {
    let rejected = |error: runtime_media::MediaError| AdmissionError::Rejected(error.to_string());
    let orientation = image_orientation_v1(recipe);
    let (width, height) = match runtime_media::probe_oriented(&source.bytes, limits, orientation)
        .map_err(rejected)?
    {
        runtime_media::Probe::Png { width, height }
        | runtime_media::Probe::Jpeg { width, height } => (width, height),
        runtime_media::Probe::Wav { .. } => return Err(AdmissionError::InvalidPrompt),
    };
    let grid = planned_image_grid_v1(width, height, recipe, limits).map_err(rejected)?;
    let merge = recipe_merge_v1(recipe);
    let span = image_span_rows_v1(modality, merge, grid).ok_or(AdmissionError::InvalidPrompt)?;
    if span > u64::from(modality.maximum_tokens) {
        let cells = u64::from(grid[0])
            * u64::from(grid[1].div_ceil(merge))
            * u64::from(grid[2].div_ceil(merge));
        return Err(AdmissionError::Rejected(format!(
            "a {width} x {height} image spans {span} rows ({cells} cells and {} learned delimiter rows), \
             above the modality's {} tokens",
            span - cells,
            modality.maximum_tokens
        )));
    }
    Ok(())
}

fn declared_image_recipe_v1(
    recipe: &runtime_frontend::FrontendMediaRecipeV1,
) -> Result<ImageRecipeV1, AdmissionError> {
    if recipe.tag == TOKEN_BUDGET_TAG_V1 {
        let alpha = declared_alpha_v1(&recipe.parameters).ok_or(AdmissionError::InvalidPrompt)?;
        Ok(ImageRecipeV1::TokenBudget(
            token_budget_recipe_v1(&recipe.parameters)?,
            alpha,
        ))
    } else if recipe.tag == CENTERED_PAD_TAG_V1 {
        let alpha = declared_alpha_v1(&recipe.parameters).ok_or(AdmissionError::InvalidPrompt)?;
        Ok(ImageRecipeV1::CenteredPad(
            centered_pad_recipe_v1(&recipe.parameters)?,
            alpha,
        ))
    } else {
        image_recipe_v1(&recipe.parameters).map(ImageRecipeV1::PatchGrid)
    }
}

fn media_vector_v1(parameters: &serde_json::Value, key: &str) -> Result<[f32; 3], AdmissionError> {
    let values = parameters
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or(AdmissionError::InvalidPrompt)?;
    let [a, b, c] = values.as_slice() else {
        return Err(AdmissionError::InvalidPrompt);
    };
    let value = |v: &serde_json::Value| v.as_f64().map(|v| v as f32);
    match (value(a), value(b), value(c)) {
        (Some(a), Some(b), Some(c)) => Ok([a, b, c]),
        _ => Err(AdmissionError::InvalidPrompt),
    }
}

/// Parse the parameters of [`TOKEN_BUDGET_TAG_V1`].
pub fn token_budget_recipe_v1(
    parameters: &serde_json::Value,
) -> Result<runtime_media::recipe::TokenBudgetRecipe, AdmissionError> {
    let u32_at = |name| {
        u32::try_from(media_u64_v1(parameters, name)?).map_err(|_| AdmissionError::InvalidPrompt)
    };
    Ok(runtime_media::recipe::TokenBudgetRecipe {
        patch: u32_at("patch")?,
        temporal_patch: u32_at("temporal_patch")?,
        merge: u32_at("merge")?,
        minimum_tokens: media_u64_v1(parameters, "minimum_tokens")?,
        maximum_tokens: media_u64_v1(parameters, "maximum_tokens")?,
        value_scale: parameters
            .get("value_scale")
            .and_then(serde_json::Value::as_f64)
            .ok_or(AdmissionError::InvalidPrompt)?,
        mean: media_vector_v1(parameters, "mean")?,
        std: media_vector_v1(parameters, "std")?,
    })
}

/// Parse the parameters of [`CENTERED_PAD_TAG_V1`]: `pad` is three 8-bit
/// values, `value_divisor` a number.
pub fn centered_pad_recipe_v1(
    parameters: &serde_json::Value,
) -> Result<runtime_media::recipe::CenteredPadRecipe, AdmissionError> {
    let u32_at = |name| {
        u32::try_from(media_u64_v1(parameters, name)?).map_err(|_| AdmissionError::InvalidPrompt)
    };
    let pad = parameters
        .get("pad")
        .and_then(serde_json::Value::as_array)
        .ok_or(AdmissionError::InvalidPrompt)?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| u8::try_from(value).ok())
                .ok_or(AdmissionError::InvalidPrompt)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(runtime_media::recipe::CenteredPadRecipe {
        patch: u32_at("patch")?,
        downsample: u32_at("downsample")?,
        minimum_pixels: media_u64_v1(parameters, "minimum_pixels")?,
        maximum_tokens: media_u64_v1(parameters, "maximum_tokens")?,
        pad: pad.try_into().map_err(|_| AdmissionError::InvalidPrompt)?,
        value_divisor: parameters
            .get("value_divisor")
            .and_then(serde_json::Value::as_f64)
            .ok_or(AdmissionError::InvalidPrompt)? as f32,
        mean: media_vector_v1(parameters, "mean")?,
        std: media_vector_v1(parameters, "std")?,
    })
}

/// Parse the parameters consumed by the generic image patch-grid recipe.
/// `interpolation` ("bilinear" or "bicubic-antialias") and
/// `upscale_small_side` are optional and default to "bilinear" and true.
pub fn image_recipe_v1(
    parameters: &serde_json::Value,
) -> Result<runtime_media::recipe::ImageRecipe, AdmissionError> {
    let u32_at = |name| {
        u32::try_from(media_u64_v1(parameters, name)?).map_err(|_| AdmissionError::InvalidPrompt)
    };
    let vector = |name| -> Result<[f32; 3], AdmissionError> {
        let values = parameters
            .get(name)
            .and_then(serde_json::Value::as_array)
            .ok_or(AdmissionError::InvalidPrompt)?;
        let [a, b, c] = values.as_slice() else {
            return Err(AdmissionError::InvalidPrompt);
        };
        Ok([
            a.as_f64().ok_or(AdmissionError::InvalidPrompt)? as f32,
            b.as_f64().ok_or(AdmissionError::InvalidPrompt)? as f32,
            c.as_f64().ok_or(AdmissionError::InvalidPrompt)? as f32,
        ])
    };
    let interpolation = match parameters.get("interpolation") {
        None => runtime_media::recipe::Interpolation::Bilinear,
        Some(value) => match value.as_str() {
            Some("bilinear") => runtime_media::recipe::Interpolation::Bilinear,
            Some("bicubic-antialias") => runtime_media::recipe::Interpolation::BicubicAntialias,
            _ => return Err(AdmissionError::InvalidPrompt),
        },
    };
    let upscale_small_side = match parameters.get("upscale_small_side") {
        None => true,
        Some(value) => value.as_bool().ok_or(AdmissionError::InvalidPrompt)?,
    };
    Ok(runtime_media::recipe::ImageRecipe {
        factor: u32_at("factor")?,
        minimum_pixels: media_u64_v1(parameters, "minimum_pixels")?,
        maximum_pixels: media_u64_v1(parameters, "maximum_pixels")?,
        upscale_small_side,
        interpolation,
        patch: u32_at("patch")?,
        temporal_patch: u32_at("temporal_patch")?,
        merge: u32_at("merge")?,
        value_scale: parameters
            .get("value_scale")
            .and_then(serde_json::Value::as_f64)
            .ok_or(AdmissionError::InvalidPrompt)?,
        mean: vector("mean")?,
        std: vector("std")?,
    })
}

/// Parse the parameters consumed by the generic log-mel recipe.
pub fn audio_recipe_v1(
    parameters: &serde_json::Value,
) -> Result<runtime_media::audio::AudioRecipe, AdmissionError> {
    let size = |name| {
        usize::try_from(media_u64_v1(parameters, name)?).map_err(|_| AdmissionError::InvalidPrompt)
    };
    let u32_at = |name| {
        u32::try_from(media_u64_v1(parameters, name)?).map_err(|_| AdmissionError::InvalidPrompt)
    };
    let float = |name| {
        parameters
            .get(name)
            .and_then(serde_json::Value::as_f64)
            .ok_or(AdmissionError::InvalidPrompt)
    };
    Ok(runtime_media::audio::AudioRecipe {
        sample_rate: u32_at("sample_rate")?,
        sinc_width: u32_at("sinc_width")?,
        rolloff: float("rolloff")?,
        n_fft: size("n_fft")?,
        window: size("window")?,
        hop: size("hop")?,
        mel_bins: size("mel_bins")?,
        f_min: float("f_min")?,
        f_max: float("f_max")?,
        log_floor: float("log_floor")? as f32,
        first_kernel: size("first_kernel")?,
        second_kernel: size("second_kernel")?,
        second_stride: size("second_stride")?,
        pool: size("pool")?,
        group: size("group")?,
    })
}

fn expand_media_chat_v1<E, P>(
    declared: &FrontendMultimodalV1,
    vocabulary: &FrontendVocabularyV1,
    segments: &[PromptSegmentV1],
    media: &[FrontendMediaPartV1],
    mut encode: E,
    mut probe: P,
) -> Result<ExpandedMediaPromptV1, AdmissionError>
where
    E: FnMut(&str) -> Result<Vec<u32>, AdmissionError>,
    P: FnMut(
        &FrontendMediaPartV1,
        &runtime_frontend::FrontendMediaModalityV1,
    ) -> Result<PreparedMediaInputV1, AdmissionError>,
{
    let mut inputs = Vec::with_capacity(media.len());
    let mut pads = Vec::with_capacity(media.len());
    let mut layouts = Vec::with_capacity(media.len());
    let mut position_grids = Vec::with_capacity(media.len());
    for part in media {
        let kind = match part.kind {
            FrontendMediaPartKindV1::Image => runtime_frontend::FrontendMediaKindV1::Image,
            FrontendMediaPartKindV1::Audio => runtime_frontend::FrontendMediaKindV1::Audio,
            FrontendMediaPartKindV1::Video => runtime_frontend::FrontendMediaKindV1::Video,
        };
        let modality = declared
            .modalities
            .iter()
            .find(|item| item.kind == kind && item.part_types.contains(&part.part_type))
            .ok_or(AdmissionError::InvalidPrompt)?;
        let prepared = probe(part, modality)?;
        let input = prepared.input;
        if let Some(grid) = &modality.render.grid {
            let total = prepared.grids.iter().try_fold(0_u32, |total, (_, rows)| {
                if *rows == 0 {
                    None
                } else {
                    total.checked_add(*rows)
                }
            });
            if prepared.grids.is_empty() || total != Some(input.output_rows) {
                return Err(AdmissionError::InvalidPrompt);
            }
            layouts.push(Some((grid, prepared.grids)));
        } else {
            if !prepared.grids.is_empty() {
                return Err(AdmissionError::InvalidPrompt);
            }
            layouts.push(None);
        }
        if input.encoder != modality.encoder
            || input.output_rows == 0
            || input.output_rows > modality.maximum_tokens
            || input.payload.len()
                > declared.encoders[usize::from(input.encoder)].maximum_payload_bytes
        {
            return Err(AdmissionError::InvalidPrompt);
        }
        let pad = vocabulary
            .boundary_tokens
            .iter()
            .find(|token| token.text == modality.render.pad)
            .ok_or(AdmissionError::InvalidPrompt)?
            .id;
        pads.push(pad);
        position_grids.push(position_grid_v1(modality, &input)?);
        inputs.push(input);
    }
    let mut tokens = Vec::new();
    let mut spans = Vec::new();
    let mut positions = Vec::new();
    for segment in segments {
        match segment {
            PromptSegmentV1::Text(text) => tokens.extend(encode(text)?),
            PromptSegmentV1::Media(index) => {
                let input = inputs.get(*index).ok_or(AdmissionError::InvalidPrompt)?;
                let media_index =
                    u16::try_from(*index).map_err(|_| AdmissionError::InvalidPrompt)?;
                let mut append_rows = |rows: u32,
                                       media_row: u32,
                                       tokens: &mut Vec<u32>|
                 -> Result<(), AdmissionError> {
                    let prompt_row =
                        u32::try_from(tokens.len()).map_err(|_| AdmissionError::InvalidPrompt)?;
                    tokens.resize(
                        tokens
                            .len()
                            .checked_add(rows as usize)
                            .ok_or(AdmissionError::InvalidPrompt)?,
                        pads[*index],
                    );
                    spans.push(MediaSpanV1 {
                        media: media_index,
                        prompt_row,
                        rows,
                        media_row,
                    });
                    Ok(())
                };
                if let Some((render, grids)) = &layouts[*index] {
                    let prefix = render.prefix.concat();
                    let suffix = render.suffix.concat();
                    let mut media_row = 0;
                    for (timestamp, rows) in grids {
                        tokens.extend(encode(&format!("{timestamp}{prefix}"))?);
                        append_rows(*rows, media_row, &mut tokens)?;
                        media_row += rows;
                        tokens.extend(encode(&suffix)?);
                    }
                } else {
                    let prompt_row =
                        u32::try_from(tokens.len()).map_err(|_| AdmissionError::InvalidPrompt)?;
                    append_rows(input.output_rows, 0, &mut tokens)?;
                    if let Some(grid) = position_grids[*index] {
                        positions.push(PositionSpanV1 {
                            prompt_row,
                            rows: input.output_rows,
                            grid,
                        });
                    }
                }
            }
        }
    }
    Ok((tokens, inputs, spans, positions))
}

pub(crate) fn prepare_owned_prompt(
    tokenizer: &Tokenizer,
    prompt: PreparedPrompt,
    policy: &EngineModelPolicyV1,
    tokenizer_input_bytes: usize,
) -> Result<Vec<u32>, AdmissionError> {
    match prompt {
        PreparedPrompt::TokenIds(ids) => {
            validate_token_ids(tokenizer, &ids)?;
            Ok(ids)
        }
        PreparedPrompt::MediaTokens { tokens, .. } => {
            validate_token_ids(tokenizer, &tokens)?;
            Ok(tokens)
        }
        prompt => tokenize_prompt(tokenizer, &prompt, policy, tokenizer_input_bytes),
    }
}

/// Refuses bias entries outside the model vocabulary with vLLM's message.
fn validate_logit_bias(
    sampling: &SamplingParameters,
    vocabulary: ModelVocabulary,
) -> Result<(), AdmissionError> {
    let invalid = sampling
        .logit_bias()
        .iter()
        .map(|bias| bias.token_id())
        .filter(|token| usize::try_from(*token).map_or(true, |token| token >= vocabulary.get()))
        .map(|token| token.to_string())
        .collect::<Vec<_>>();
    if invalid.is_empty() {
        return Ok(());
    }
    Err(AdmissionError::InvalidParameter {
        message: format!(
            "token_id(s) [{}] in logit_bias contain out-of-vocab token ids. Vocabulary size: {}",
            invalid.join(", "),
            vocabulary.get()
        ),
        parameter: "logit_bias",
    })
}

fn validate_token_ids(tokenizer: &Tokenizer, ids: &[u32]) -> Result<(), AdmissionError> {
    if ids.iter().any(|id| tokenizer.token_bytes(*id).is_err()) {
        Err(AdmissionError::InvalidPrompt)
    } else {
        Ok(())
    }
}

// Events can outlive their engine request, including the last sampled token.
// Only failures retiring a live request indicate a worker-level failure.
fn cancel_live_request<E: ModelExecutor>(engine: &mut Engine<E>, id: RequestId) -> bool {
    engine.finish_completion_metrics(id, EngineFinishReason::Cancelled);
    retire_live_request(engine, id, false)
}

fn retire_live_request<E: ModelExecutor>(
    engine: &mut Engine<E>,
    id: RequestId,
    stopped: bool,
) -> bool {
    match engine.request_state(id) {
        Ok(state) if state.is_terminal() => true,
        Err(EngineError::Scheduler(runtime_scheduler::ScheduleError::UnknownRequest(retired)))
            if retired == id =>
        {
            true
        }
        Ok(_) if stopped => engine.stop(id).is_ok(),
        Ok(_) => engine.cancel(id).is_ok(),
        Err(_) => false,
    }
}

fn handle_engine_event<E: ModelExecutor>(
    tokenizer: &Tokenizer,
    engine: &mut Engine<E>,
    active: &mut BTreeMap<RequestId, ActiveGeneration>,
    event: EngineEvent,
) -> bool {
    let id = event.request_id();
    let Some(mut generation) = active.remove(&id) else {
        return true;
    };
    match event.kind() {
        EngineEventKind::PromptCached { tokens } => {
            generation.cached_tokens = Some(*tokens as u64);
            active.insert(id, generation);
            true
        }
        EngineEventKind::PromptLogprobs { positions } => {
            let rows = positions
                .iter()
                .map(|position| {
                    let context = generation
                        .prompt_logprob_context
                        .iter()
                        .flatten()
                        .map(|id| tokenizer.token_bytes(*id).map_err(|_| ()))
                        .collect::<Result<Vec<_>, ()>>()?;
                    let token = token_label(
                        tokenizer.token_bytes(position.token_id).map_err(|_| ())?,
                        &context,
                    );
                    generation.prompt_logprob_context.rotate_left(1);
                    generation.prompt_logprob_context[3] = Some(position.token_id);
                    let top_logprobs = position
                        .top_logprobs
                        .iter()
                        .map(|top| {
                            let bytes = tokenizer
                                .token_bytes(top.token_id())
                                .map_err(|_| ())?
                                .to_vec();
                            Ok((
                                top.token_id(),
                                TokenLogprob {
                                    token: token_label(&bytes, &context),
                                    bytes,
                                    logprob: top.logprob(),
                                },
                            ))
                        })
                        .collect::<Result<Vec<_>, ()>>()?;
                    Ok(ServerPromptTokenLogprob {
                        position: position.position,
                        token_id: position.token_id,
                        token,
                        logprob: position.logprob,
                        top_logprobs,
                    })
                })
                .collect::<Result<Vec<_>, ()>>();
            if rows.is_ok_and(|rows| {
                generation
                    .events
                    .try_emit(GenerationEvent::PromptLogprobs(rows))
                    .is_ok()
            }) {
                active.insert(id, generation);
                true
            } else {
                fail_generation(&generation, DECODE_MESSAGE, DECODE_CODE);
                cancel_live_request(engine, id)
            }
        }
        EngineEventKind::Token {
            token_id,
            logprob,
            top_logprobs,
        } => {
            let ordinal = match u64::try_from(generation.completion_tokens) {
                Ok(ordinal) => ordinal,
                Err(_) => {
                    fail_generation(&generation, DECODE_MESSAGE, DECODE_CODE);
                    return cancel_live_request(engine, id);
                }
            };
            let decoded = match generated_token_v1(tokenizer, ordinal, *token_id) {
                Ok(token) => token,
                Err(_) => {
                    fail_generation(&generation, DECODE_MESSAGE, DECODE_CODE);
                    return cancel_live_request(engine, id);
                }
            };
            let record = if generation.logprobs_enabled {
                match generated_logprob(
                    tokenizer,
                    *token_id,
                    &decoded.bytes,
                    *logprob,
                    top_logprobs,
                    generation.logprob_context,
                ) {
                    Ok(record) => Some(record),
                    Err(()) => {
                        fail_generation(&generation, DECODE_MESSAGE, DECODE_CODE);
                        return cancel_live_request(engine, id);
                    }
                }
            } else {
                None
            };
            generation.completion_tokens = match generation.completion_tokens.checked_add(1) {
                Some(count) => count,
                None => {
                    fail_generation(&generation, DECODE_MESSAGE, DECODE_CODE);
                    return cancel_live_request(engine, id);
                }
            };
            if let Some(record) = record
                && generation
                    .events
                    .try_emit(GenerationEvent::GeneratedTokenLogprobs(vec![record]))
                    .is_err()
            {
                handle_delivery_error(&generation, UpdateDeliveryError::Overflow);
                return cancel_live_request(engine, id);
            }
            if generation.return_token_ids
                && generation
                    .events
                    .try_emit(GenerationEvent::GeneratedTokenIds(vec![*token_id]))
                    .is_err()
            {
                handle_delivery_error(&generation, UpdateDeliveryError::Overflow);
                return cancel_live_request(engine, id);
            }
            generation.logprob_context.rotate_left(1);
            generation.logprob_context[3] = Some(*token_id);
            let updates = match generation.decoder.push_engine_token(decoded) {
                Ok(updates) => updates,
                Err(_) => {
                    fail_generation(&generation, DECODE_MESSAGE, DECODE_CODE);
                    return cancel_live_request(engine, id);
                }
            };
            // Client stop strings and legacy suffixes without stop IDs can
            // finish before the engine sees a stop token. The frontend owns
            // the public reason, including at the engine token limit.
            let terminal = updates
                .iter()
                .any(|update| matches!(update, CompletionUpdateV1::Finished(_)));
            if let Err(error) = deliver_engine_updates(engine, id, &mut generation, updates) {
                handle_delivery_error(&generation, error);
                return cancel_live_request(engine, id);
            }
            if terminal {
                return retire_live_request(engine, id, true);
            }
            active.insert(id, generation);
            true
        }
        EngineEventKind::Finished {
            reason: reason @ (EngineFinishReason::Length | EngineFinishReason::Stop),
        } => {
            let updates = match if *reason == EngineFinishReason::Stop {
                generation.decoder.finish_stop()
            } else {
                generation.decoder.finish_length()
            } {
                Ok(updates) => updates,
                Err(_) => {
                    fail_generation(&generation, DECODE_MESSAGE, DECODE_CODE);
                    return cancel_live_request(engine, id);
                }
            };
            if let Err(error) = deliver_engine_updates(engine, id, &mut generation, updates) {
                handle_delivery_error(&generation, error);
                return cancel_live_request(engine, id);
            }
            true
        }
        EngineEventKind::Finished {
            reason: EngineFinishReason::Cancelled,
        } => true,
        EngineEventKind::Error { message } => {
            fail_generation(&generation, message, FAILURE_CODE);
            true
        }
    }
}

fn deliver_engine_updates<E: ModelExecutor>(
    engine: &Engine<E>,
    id: RequestId,
    generation: &mut ActiveGeneration,
    updates: Vec<CompletionUpdateV1>,
) -> Result<(), UpdateDeliveryError> {
    let reason = updates.iter().find_map(|update| match update {
        CompletionUpdateV1::Finished(done) => Some(match done.reason {
            CompletionFinishV1::Stop | CompletionFinishV1::ToolCalls => EngineFinishReason::Stop,
            CompletionFinishV1::Length => EngineFinishReason::Length,
        }),
        _ => None,
    });
    deliver_updates(generation, updates)?;
    if let Some(reason) = reason {
        engine.finish_completion_metrics(id, reason);
    }
    Ok(())
}

/// The logprob record vLLM reports for one generated token: labels read with
/// the preceding sampled tokens as context, and the position's entries with
/// the generated token first and each token ID once.
fn generated_logprob(
    tokenizer: &Tokenizer,
    token_id: u32,
    bytes: &[u8],
    logprob: f64,
    top_logprobs: &[runtime_sampling::TokenLogprob],
    context: [Option<u32>; 4],
) -> Result<GeneratedTokenLogprob, ()> {
    let context = context
        .iter()
        .flatten()
        .map(|id| tokenizer.token_bytes(*id).map_err(|_| ()))
        .collect::<Result<Vec<_>, ()>>()?;
    let token = token_label(bytes, &context);
    let mut entries = vec![TokenLogprob {
        token: token.clone(),
        bytes: token.as_bytes().to_vec(),
        logprob,
    }];
    for entry in top_logprobs {
        if entry.token_id() == token_id {
            continue;
        }
        let label = token_label(
            tokenizer.token_bytes(entry.token_id()).map_err(|_| ())?,
            &context,
        );
        entries.push(TokenLogprob {
            bytes: label.as_bytes().to_vec(),
            token: label,
            logprob: entry.logprob(),
        });
    }
    Ok(GeneratedTokenLogprob {
        token_id,
        bytes: token.as_bytes().to_vec(),
        token,
        logprob,
        top_logprobs: entries,
    })
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum UpdateDeliveryError {
    Overflow,
    InvalidOutput,
}

fn deliver_updates(
    generation: &mut ActiveGeneration,
    updates: Vec<CompletionUpdateV1>,
) -> Result<(), UpdateDeliveryError> {
    for update in updates {
        let finished = matches!(update, CompletionUpdateV1::Finished(_));
        for event in generation_events_v1(
            &mut generation.accounting,
            generation.completion_tokens,
            (generation.prompt_tokens, generation.cached_tokens),
            update,
        )? {
            generation
                .events
                .try_emit(event)
                .map_err(|_| UpdateDeliveryError::Overflow)?;
        }
        if finished {
            return Ok(());
        }
    }
    Ok(())
}

/// The transport events one decoder update becomes after `completion_tokens`
/// generated tokens, as the worker delivers them.
///
/// Each text or call event is charged the generated tokens it newly accounts
/// for. A finish first reports the tokens no event accounted for, then the
/// finish with the request's usage (`prompt_tokens`, `cached_tokens`). A
/// frontend check delivers its decoded updates through this too.
pub(crate) fn generation_events_v1(
    accounting: &mut TokenAccounting,
    completion_tokens: usize,
    (prompt_tokens, cached_tokens): (u64, Option<u64>),
    update: CompletionUpdateV1,
) -> Result<Vec<GenerationEvent>, UpdateDeliveryError> {
    Ok(vec![match update {
        CompletionUpdateV1::ToolCallDeltas(deltas) => GenerationEvent::ToolCallDeltas(deltas),
        CompletionUpdateV1::Reasoning(text) => {
            let generated_token_count =
                account(accounting, completion_tokens, &text.contributions)?;
            GenerationEvent::Reasoning(ReasoningContent {
                delta: text.text,
                generated_token_count,
            })
        }
        CompletionUpdateV1::Content(text) => {
            let generated_token_count =
                account(accounting, completion_tokens, &text.contributions)?;
            GenerationEvent::Content(VisibleContent {
                delta: text.text,
                generated_token_count,
            })
        }
        CompletionUpdateV1::ToolCalls(tool) => {
            let generated_token_count =
                account(accounting, completion_tokens, &tool.contributions)?;
            let calls = tool
                .calls
                .into_iter()
                .map(|call| BackendToolCall {
                    name: call.name,
                    arguments: call.arguments,
                })
                .collect();
            GenerationEvent::ToolCalls {
                calls,
                generated_token_count,
            }
        }
        CompletionUpdateV1::Finished(done) => {
            let unaccounted = unaccounted_tokens(accounting, completion_tokens)?;
            let completion_tokens =
                u64::try_from(completion_tokens).map_err(|_| UpdateDeliveryError::InvalidOutput)?;
            let finished = GenerationEvent::Finished {
                reason: match done.reason {
                    CompletionFinishV1::Stop => FinishReason::Stop,
                    CompletionFinishV1::Length => FinishReason::Length,
                    CompletionFinishV1::ToolCalls => FinishReason::ToolCalls,
                },
                usage: TokenUsage {
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                },
            };
            return Ok(if unaccounted > 0 {
                vec![
                    GenerationEvent::TokenUsageDelta {
                        generated_token_count: unaccounted,
                    },
                    finished,
                ]
            } else {
                vec![finished]
            });
        }
    }])
}

impl From<()> for UpdateDeliveryError {
    fn from((): ()) -> Self {
        Self::InvalidOutput
    }
}

fn account(
    accounting: &mut TokenAccounting,
    completion_tokens: usize,
    contributions: &[TokenContributionV1],
) -> Result<usize, ()> {
    let mut greatest = None;
    for contribution in contributions {
        let ordinal = usize::try_from(contribution.ordinal).map_err(|_| ())?;
        if ordinal >= completion_tokens || contribution.byte_start >= contribution.byte_end {
            return Err(());
        }
        if let Some((last_ordinal, last_byte_end)) = accounting.last_contribution
            && (contribution.ordinal < last_ordinal
                || (contribution.ordinal == last_ordinal
                    && contribution.byte_start < last_byte_end))
        {
            return Err(());
        }
        accounting.last_contribution = Some((contribution.ordinal, contribution.byte_end));
        greatest = Some(ordinal);
    }
    let Some(greatest) = greatest else {
        return Err(());
    };
    let next_frontier = accounting.frontier.max(greatest.checked_add(1).ok_or(())?);
    let newly_accounted = next_frontier.checked_sub(accounting.frontier).ok_or(())?;
    accounting.frontier = next_frontier;
    Ok(newly_accounted)
}

fn unaccounted_tokens(accounting: &TokenAccounting, completion_tokens: usize) -> Result<usize, ()> {
    completion_tokens.checked_sub(accounting.frontier).ok_or(())
}

fn handle_delivery_error(generation: &ActiveGeneration, error: UpdateDeliveryError) {
    match error {
        UpdateDeliveryError::Overflow => generation.cancellation.cancel(),
        UpdateDeliveryError::InvalidOutput => {
            fail_generation(generation, DECODE_MESSAGE, DECODE_CODE);
        }
    }
}

// vLLM corrects byte-fallback labels using up to four preceding sampled
// tokens. Alternatives at the same position are never context for one another.
fn token_label(bytes: &[u8], context: &[&[u8]]) -> String {
    let text = String::from_utf8_lossy(bytes);
    if !text.ends_with('\u{fffd}') {
        return text.into_owned();
    }
    for count in 1..=context.len().min(4) {
        let context = &context[context.len() - count..];
        let mut combined = context.concat();
        combined.extend_from_slice(bytes);
        let full = String::from_utf8_lossy(&combined);
        if full.ends_with('\u{fffd}') {
            continue;
        }
        let clean_end = context
            .iter()
            .rposition(|part| !String::from_utf8_lossy(part).ends_with('\u{fffd}'))
            .map_or(0, |index| index + 1);
        let clean = context[..clean_end].concat();
        let prefix = String::from_utf8_lossy(&clean);
        let common: usize = full
            .chars()
            .zip(prefix.chars())
            .take_while(|(a, b)| a == b)
            .map(|(ch, _)| ch.len_utf8())
            .sum();
        return full[common..].to_owned();
    }
    String::new()
}

fn fail_generation(generation: &ActiveGeneration, message: &str, code: &str) {
    let _ = generation.events.try_emit(GenerationEvent::Failed {
        message: message.to_owned(),
        code: Some(code.to_owned()),
    });
}

fn cancel_all<E: ModelExecutor>(
    engine: &mut Engine<E>,
    active: &mut BTreeMap<RequestId, ActiveGeneration>,
) {
    let ids = active.keys().copied().collect::<Vec<_>>();
    active.clear();
    for id in ids {
        let _ = cancel_live_request(engine, id);
    }
    let _ = engine.drain_events();
}

fn fail_all<E: ModelExecutor>(
    engine: &mut Engine<E>,
    active: &mut BTreeMap<RequestId, ActiveGeneration>,
) {
    for generation in active.values() {
        fail_generation(generation, FAILURE_MESSAGE, FAILURE_CODE);
    }
    cancel_all(engine, active);
}

fn video_geometry_v1(
    grids: &[runtime_media::recipe::TimedGrid],
) -> Result<[u32; 4], AdmissionError> {
    let first = grids
        .first()
        .ok_or(AdmissionError::InvalidPrompt)?
        .grid
        .grid;
    if first.contains(&0)
        || grids
            .iter()
            .any(|grid| grid.grid.grid[0] == 0 || grid.grid.grid[1..] != first[1..])
    {
        return Err(AdmissionError::InvalidPrompt);
    }
    let temporal = grids
        .iter()
        .try_fold(0_u32, |sum, grid| sum.checked_add(grid.grid.grid[0]))
        .ok_or(AdmissionError::InvalidPrompt)?;
    Ok([temporal, first[1], first[2], 0])
}

#[cfg(test)]
mod tests {
    /// A chat whose render left reasoning open waits for a declared
    /// reasoning end token before its constraint applies; a chat prompt, a
    /// thinking prompt that continues a final assistant turn which already
    /// closed its reasoning, and a completion with no rendered chat are
    /// constrained from the start.
    #[test]
    fn a_constraint_waits_for_reasoning_only_where_the_prompt_left_it_open() {
        use runtime_server::ServerFrontendV1;
        let descriptor = runtime_frontend::FrontendDescriptorV1::from_component_bytes(
            super::stop_tests::declarations::FUNCTION_PARAMETERS.as_bytes(),
        )
        .unwrap();
        let frontend = runtime_frontend::BundleFrontendV1::with_export(
            descriptor.clone(),
            &runtime_frontend::test_support::fixture_export_v1(),
        )
        .unwrap();
        let gate = |thinking, continued| {
            let mut request = runtime_frontend::test_support::reasoning::ask(
                &[("enable_thinking", thinking)],
                None,
                None,
            );
            if continued {
                request.messages.push(runtime_server::FrontendMessageV1 {
                    role: runtime_server::FrontendRoleV1::Assistant,
                    content: Some("The answer".to_owned()),
                    parts: None,
                    reasoning_content: Some("Thought it through.".to_owned()),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                    tools: Vec::new(),
                });
                request.continue_final_message = true;
                request.add_generation_prompt = false;
            }
            let rendered = frontend.render_chat(request).unwrap();
            super::constraint_gate_v1(
                super::BundleChatStateV1::from_state(&rendered.state),
                &descriptor,
            )
        };
        let tokens = super::reasoning_gate_tokens(&descriptor);
        assert!(!tokens.is_empty());
        assert_eq!(gate(true, false), Some(tokens));
        assert_eq!(gate(false, false), None);
        assert_eq!(gate(true, true), None);
        assert_eq!(gate(false, true), None);
        assert_eq!(super::constraint_gate_v1(None, &descriptor), None);
    }

    #[test]
    fn media_recipe_versions_refuse_unsupported_declarations() {
        for (tag, parameters) in [
            (
                "audio.log-mel.v1",
                serde_json::json!({"sample_rate": 24000.0, "maximum_seconds": 600.0,
                    "channels": 1, "resampler": "sinc-hann", "window_kind": "periodic-hann",
                    "power": 1, "mel_scale": "htk", "mel_norm": null, "center": true,
                    "pad_mode": "reflect", "segment_frames": 6000}),
            ),
            (
                "video.timestamped-grids.v1",
                serde_json::json!({
                    "maximum_frames": 32, "seconds_per_frame": 1,
                    "sampling": "uniform", "repeat_last_if_odd": true, "timestamp": "mm:ss",
                    "minimum_pixels_per_frame": 8192, "maximum_pixels_per_frame": 262144,
                    "image": {"minimum_pixels": 8192, "maximum_pixels": 262144}
                }),
            ),
        ] {
            let recipe = runtime_frontend::FrontendMediaRecipeV1 {
                tag: tag.into(),
                parameters,
            };
            assert!(validate_media_recipe_v1(&recipe).is_ok());
            for key in recipe.parameters.as_object().unwrap().keys() {
                if key == "image" {
                    continue;
                }
                if key != "seconds_per_frame" {
                    let mut missing = recipe.clone();
                    missing.parameters.as_object_mut().unwrap().remove(key);
                    assert!(validate_media_recipe_v1(&missing).is_err(), "missing {key}");
                }
                let mut changed = recipe.clone();
                changed.parameters[key] = serde_json::json!("unsupported");
                assert!(
                    validate_media_recipe_v1(&changed)
                        .unwrap_err()
                        .contains(key)
                );
            }
        }
        let recipe = runtime_frontend::FrontendMediaRecipeV1 {
            tag: "video.timestamped-grids.v1".into(),
            parameters: serde_json::json!({"maximum_frames": 16}),
        };
        assert!(validate_media_recipe_v1(&recipe).is_err());
        let recipe = runtime_frontend::FrontendMediaRecipeV1 {
            tag: "audio.log-mel.v1".into(),
            parameters: serde_json::json!({"sample_rate": 16000}),
        };
        assert!(validate_media_recipe_v1(&recipe).is_err());
    }

    #[test]
    fn image_recipe_annotations_cannot_override_implemented_behavior() {
        let parameters = serde_json::json!({"factor": 32, "small_side_minimum": 32,
            "interpolation": "bilinear", "antialias": false,
            "alpha_background": "white", "exif_orientation": true});
        assert!(validate_image_fixed_parameters_v1(&parameters).is_ok());
        for key in [
            "small_side_minimum",
            "antialias",
            "alpha_background",
            "exif_orientation",
        ] {
            let mut changed = parameters.clone();
            changed[key] = serde_json::json!("unsupported");
            assert!(
                validate_image_fixed_parameters_v1(&changed)
                    .unwrap_err()
                    .contains(key)
            );
        }
        assert_eq!(
            media_u64_v1(&serde_json::json!({"rate": 24000.0}), "rate").unwrap(),
            24000
        );
        assert!(media_u64_v1(&serde_json::json!({"rate": 24000.5}), "rate").is_err());
    }

    #[test]
    fn byte_fallback_logprob_labels_use_sampled_context() {
        assert_eq!(token_label(b"\x80", &[]), "");
        assert_eq!(token_label(b"\xe2\x82", &[]), "");
        assert_eq!(token_label(b"\xac", &[b"\xe2", b"\x82"]), "€");
        assert_eq!(token_label(b"a\x80b", &[]), "a�b");
        assert_eq!(token_label(b"\xac", &[b"a", b"\xe2", b"\x82"]), "€");
    }

    #[test]
    fn video_encoder_geometry_preserves_spatial_dimensions() {
        let grid = |t, h, w| runtime_media::recipe::TimedGrid {
            timestamp: "00:00".into(),
            grid: runtime_media::recipe::PatchGrid {
                grid: [t, h, w],
                columns: 1536,
                rows: Vec::new(),
                tokens: 1,
            },
        };
        assert_eq!(
            super::video_geometry_v1(&[grid(1, 8, 16), grid(1, 8, 16)]).unwrap(),
            [2, 8, 16, 0]
        );
        assert!(super::video_geometry_v1(&[grid(1, 8, 16), grid(1, 16, 8)]).is_err());
        assert!(super::video_geometry_v1(&[grid(u32::MAX, 8, 16), grid(1, 8, 16)]).is_err());
        assert!(super::video_geometry_v1(&[]).is_err());
    }

    #[test]
    fn transposed_images_with_equal_patch_bytes_get_different_media_identities() {
        // Eight 32x32 blocks in the same traversal order, as a 64x128 and a
        // 128x64 image, pack to the same bytes under a 16-pixel, merge-2 patch recipe.
        let image = |width: u32, height: u32| {
            let mut data = vec![0; (width * height * 3) as usize];
            for y in 0..height {
                for x in 0..width {
                    let block = (y / 32) * (width / 32) + x / 32;
                    let at = ((y * width + x) * 3) as usize;
                    data[at..at + 3].copy_from_slice(&[
                        (block * 29) as u8,
                        (255 - block * 23) as u8,
                        (block * 17) as u8,
                    ]);
                }
            }
            runtime_media::RgbImage {
                width,
                height,
                data,
            }
        };
        let recipe = runtime_media::recipe::ImageRecipe {
            factor: 32,
            minimum_pixels: 8192,
            maximum_pixels: 1 << 20,
            upscale_small_side: true,
            interpolation: runtime_media::recipe::Interpolation::Bilinear,
            patch: 16,
            temporal_patch: 2,
            merge: 2,
            value_scale: 1.0,
            mean: [123.675, 116.28, 103.53],
            std: [58.395, 57.12, 57.375],
        };
        let limits = runtime_media::MediaLimits {
            maximum_bytes: 1 << 26,
            maximum_pixels: 1 << 20,
            maximum_audio_frames: 0,
            maximum_frames: VIDEO_MAXIMUM_FRAMES_V1,
            maximum_tokens: 1024,
            maximum_aspect_ratio: 200.0,
        };
        let input = |width, height| {
            let grid =
                runtime_media::recipe::image_grid(&image(width, height), recipe, limits).unwrap();
            let payload = grid.bf16_bytes();
            EngineMediaInputV1 {
                key: runtime_media::content_key("image.patch-grid.v1", b"{}", &payload),
                encoder: 0,
                geometry: [grid.grid[0], grid.grid[1], grid.grid[2], 0],
                output_rows: u32::try_from(grid.tokens).unwrap(),
                payload: payload.into(),
            }
        };
        let (tall, wide) = (input(64, 128), input(128, 64));
        assert_eq!(tall.payload, wide.payload);
        assert_eq!(tall.output_rows, wide.output_rows);
        assert_eq!(tall.key, wide.key, "the content key alone aliases them");
        assert_ne!(tall.geometry, wide.geometry);
        assert_ne!(tall.identity(), wide.identity());
        assert_eq!(tall.identity(), input(64, 128).identity());
    }

    use super::*;

    /// A solid RGB PNG with stored (uncompressed) deflate blocks.
    fn png_chunk(kind: &[u8], payload: &[u8], output: &mut Vec<u8>) {
        fn crc32(bytes: &[u8]) -> u32 {
            !bytes.iter().fold(!0_u32, |crc, &byte| {
                (0..8).fold(crc ^ u32::from(byte), |crc, _| {
                    (crc >> 1) ^ (0xedb8_8320 & (crc & 1).wrapping_neg())
                })
            })
        }
        output.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        let start = output.len();
        output.extend_from_slice(kind);
        output.extend_from_slice(payload);
        let crc = crc32(&output[start..]);
        output.extend_from_slice(&crc.to_be_bytes());
    }

    fn png_header(width: u32, height: u32) -> Vec<u8> {
        let mut header = width.to_be_bytes().to_vec();
        header.extend_from_slice(&height.to_be_bytes());
        header.extend_from_slice(&[8, 2, 0, 0, 0]);
        header
    }

    pub(super) fn solid_png(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
        let row = [&[0][..], &rgb.repeat(width as usize)].concat();
        let raw = row.repeat(height as usize);
        let mut zlib = vec![0x78, 0x01];
        let blocks = raw.chunks(65_535).collect::<Vec<_>>();
        for (index, block) in blocks.iter().enumerate() {
            zlib.push(u8::from(index + 1 == blocks.len()));
            let length = block.len() as u16;
            zlib.extend_from_slice(&length.to_le_bytes());
            zlib.extend_from_slice(&(!length).to_le_bytes());
            zlib.extend_from_slice(block);
        }
        let (a, b) = raw.iter().fold((1_u32, 0_u32), |(a, b), &byte| {
            let a = (a + u32::from(byte)) % 65_521;
            (a, (b + a) % 65_521)
        });
        zlib.extend_from_slice(&((b << 16) | a).to_be_bytes());
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png_chunk(b"IHDR", &png_header(width, height), &mut png);
        png_chunk(b"IDAT", &zlib, &mut png);
        png_chunk(b"IEND", &[], &mut png);
        png
    }

    /// A small PNG that declares `width` x `height` and whose pixel data
    /// does not inflate: its header passes, its decode fails.
    fn declared_png(width: u32, height: u32) -> runtime_media::MediaSource {
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png_chunk(b"IHDR", &png_header(width, height), &mut png);
        png_chunk(b"IDAT", &[0x78, 0x01, 0xff, 0xff], &mut png);
        png_chunk(b"IEND", &[], &mut png);
        runtime_media::MediaSource {
            mime: "image/png".to_owned(),
            bytes: png,
        }
    }

    fn patch_recipe(maximum_pixels: u64) -> runtime_media::recipe::ImageRecipe {
        image_recipe_v1(&serde_json::json!({
            "factor": 32, "minimum_pixels": 8192, "maximum_pixels": maximum_pixels,
            "patch": 16, "temporal_patch": 2, "merge": 2, "value_scale": 1.0,
            "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]
        }))
        .unwrap()
    }

    fn image_limits() -> runtime_media::MediaLimits {
        runtime_media::MediaLimits {
            maximum_bytes: 64 << 20,
            maximum_pixels: MAXIMUM_DECODED_IMAGE_PIXELS_V1,
            maximum_audio_frames: 0,
            maximum_frames: VIDEO_MAXIMUM_FRAMES_V1,
            maximum_tokens: 16_384,
            maximum_aspect_ratio: MAXIMUM_ASPECT_RATIO_V1,
        }
    }

    fn pcm_wav_source(rate: u32, frames: usize) -> runtime_media::MediaSource {
        let size = (frames * 2) as u32;
        let mut bytes = b"RIFF".to_vec();
        bytes.extend((size + 36).to_le_bytes());
        bytes.extend(b"WAVEfmt ");
        bytes.extend(16_u32.to_le_bytes());
        bytes.extend(1_u16.to_le_bytes());
        bytes.extend(1_u16.to_le_bytes());
        bytes.extend(rate.to_le_bytes());
        bytes.extend((rate * 2).to_le_bytes());
        bytes.extend(2_u16.to_le_bytes());
        bytes.extend(16_u16.to_le_bytes());
        bytes.extend(b"data");
        bytes.extend(size.to_le_bytes());
        bytes.resize(size as usize + 44, 0);
        runtime_media::MediaSource {
            mime: "audio/wav".into(),
            bytes,
        }
    }

    #[test]
    fn audio_duration_limit_uses_source_seconds() {
        let limits = runtime_media::MediaLimits {
            maximum_audio_frames: AUDIO_SAMPLE_RATE_V1 * AUDIO_MAXIMUM_SECONDS_V1,
            ..image_limits()
        };
        for rate in [16_000, 24_000, 44_100, 48_000] {
            let frames = rate as usize * 600;
            for count in [frames - 1, frames, frames + 1] {
                let source = pcm_wav_source(rate, count);
                let source_limits = audio_source_limits_v1(&source.bytes, 24_000, limits).unwrap();
                let result = runtime_media::probe(&source.bytes, source_limits);
                assert_eq!(result.is_ok(), count <= frames, "{rate} Hz, {count} frames");
            }
        }
    }

    fn test_audio_recipe() -> runtime_media::audio::AudioRecipe {
        runtime_media::audio::AudioRecipe {
            sample_rate: 24_000,
            sinc_width: 6,
            rolloff: 0.99,
            n_fft: 960,
            window: 960,
            hop: 240,
            mel_bins: 128,
            f_min: 0.0,
            f_max: 12_000.0,
            log_floor: 1e-7,
            first_kernel: 3,
            second_kernel: 3,
            second_stride: 2,
            pool: 2,
            group: 4,
        }
    }

    #[test]
    fn audio_reserves_memory_before_decoding_and_releases_it() {
        use crate::media_memory::MediaMemoryV1;
        let source = pcm_wav_source(48_000, 1920);
        let limits = runtime_media::MediaLimits {
            maximum_audio_frames: 24_000 * 600,
            ..image_limits()
        };
        let pool = Arc::new(MediaMemoryV1::new(0));
        let mut memory = RequestMediaMemoryV1::new(Arc::clone(&pool), u64::MAX);
        assert!(matches!(
            prepare_audio_v1(
                &source,
                test_audio_recipe(),
                limits,
                &|| false,
                0,
                &mut memory
            ),
            Err(AdmissionError::Unavailable(_))
        ));
        assert_eq!(pool.in_use(), 0);
        let pool = Arc::new(MediaMemoryV1::new(1 << 24));
        let mut memory = RequestMediaMemoryV1::new(Arc::clone(&pool), 1);
        assert!(matches!(
            prepare_audio_v1(
                &source,
                test_audio_recipe(),
                limits,
                &|| false,
                0,
                &mut memory
            ),
            Err(AdmissionError::Rejected(_))
        ));
        assert_eq!(pool.in_use(), 0);
        drop(memory);
        let mut memory = RequestMediaMemoryV1::new(Arc::clone(&pool), 1 << 24);
        let (mel, reserved) = prepare_audio_v1(
            &source,
            test_audio_recipe(),
            limits,
            &|| false,
            0,
            &mut memory,
        )
        .unwrap();
        assert_eq!(pool.in_use(), reserved);
        assert!(
            reserved >= source.bytes.len() as u64 + 4 * (1920 + 960) + mel.values.len() as u64 * 8
        );
        let payload = memory
            .retain(
                reserved,
                mel.values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            )
            .unwrap();
        assert_eq!(pool.in_use(), payload.len() as u64);
        drop(memory);
        drop(payload);
        assert_eq!(pool.in_use(), 0);
        let mut memory = RequestMediaMemoryV1::new(Arc::clone(&pool), 1 << 24);
        assert!(
            prepare_audio_v1(
                &source,
                test_audio_recipe(),
                limits,
                &|| true,
                0,
                &mut memory
            )
            .is_err()
        );
        drop(memory);
        assert_eq!(pool.in_use(), 0);
        // Cancellation during preparation and decoder failures both release
        // the reservation when the request leaves preparation.
        let mut memory = RequestMediaMemoryV1::new(Arc::clone(&pool), 1 << 24);
        let polls = std::cell::Cell::new(0);
        assert!(matches!(
            prepare_audio_v1(
                &source,
                test_audio_recipe(),
                limits,
                &|| {
                    polls.set(polls.get() + 1);
                    polls.get() >= 8
                },
                0,
                &mut memory
            ),
            Err(AdmissionError::Unavailable(_))
        ));
        assert!(pool.in_use() > 0);
        drop(memory);
        assert_eq!(pool.in_use(), 0);
        let mut invalid = pcm_wav_source(24_000, 1920);
        invalid.bytes[20..22].copy_from_slice(&3_u16.to_le_bytes());
        invalid.bytes[28..32].copy_from_slice(&96_000_u32.to_le_bytes());
        invalid.bytes[32..34].copy_from_slice(&4_u16.to_le_bytes());
        invalid.bytes[34..36].copy_from_slice(&32_u16.to_le_bytes());
        invalid.bytes[44..48].copy_from_slice(&f32::NAN.to_le_bytes());
        let mut memory = RequestMediaMemoryV1::new(Arc::clone(&pool), 1 << 24);
        assert!(matches!(
            prepare_audio_v1(
                &invalid,
                test_audio_recipe(),
                limits,
                &|| false,
                0,
                &mut memory
            ),
            Err(AdmissionError::Rejected(_))
        ));
        assert!(pool.in_use() > 0);
        drop(memory);
        assert_eq!(pool.in_use(), 0);
    }

    /// An image reserves its preparation bound from its header before it
    /// decodes. With the process budget held elsewhere it is refused as
    /// unavailable without decoding; a prepared image keeps only its payload;
    /// the request releases everything when it drops.
    #[test]
    fn images_reserve_host_memory_before_they_decode() {
        use crate::media_memory::{
            MediaMemoryV1, PROCESS_MEDIA_PREPARATION_BYTES_V1, REQUEST_MEDIA_PREPARATION_BYTES_V1,
            RequestMediaMemoryV1,
        };
        let limits = image_limits();
        let recipe = patch_recipe(16_777_216);
        let near_limit = declared_png(9459, 9459);
        let cost = runtime_media::recipe::image_preparation_bytes(9459, 9459, recipe, limits)
            .unwrap()
            + near_limit.bytes.len() as u64;
        // A 16 MiP image decoded at the pixel bound fits one request's budget
        // and two fit the process's, not three.
        assert!(cost <= REQUEST_MEDIA_PREPARATION_BYTES_V1);
        assert!(2 * cost <= PROCESS_MEDIA_PREPARATION_BYTES_V1);
        assert!(3 * cost > PROCESS_MEDIA_PREPARATION_BYTES_V1);

        let pool = Arc::new(MediaMemoryV1::new(PROCESS_MEDIA_PREPARATION_BYTES_V1));
        let request =
            || RequestMediaMemoryV1::new(Arc::clone(&pool), REQUEST_MEDIA_PREPARATION_BYTES_V1);
        // Other requests' preparations hold all but less than one image.
        let mut others = RequestMediaMemoryV1::new(Arc::clone(&pool), u64::MAX);
        others
            .reserve(PROCESS_MEDIA_PREPARATION_BYTES_V1 - cost + 1)
            .unwrap();
        let mut memory = request();
        assert!(matches!(
            prepare_image_v1(&near_limit, recipe, limits, &mut memory),
            Err(AdmissionError::Unavailable(_))
        ));
        drop(others);
        // With room it reserves, then its pixel data fails to inflate.
        let refused = prepare_image_v1(&near_limit, recipe, limits, &mut memory);
        assert!(
            matches!(&refused, Err(AdmissionError::Rejected(message)) if message.contains("inflate")),
            "{refused:?}"
        );
        assert_eq!(pool.in_use(), cost);
        drop(memory);
        assert_eq!(pool.in_use(), 0);

        let mut memory = request();
        let small = runtime_media::MediaSource {
            mime: "image/png".to_owned(),
            bytes: solid_png(256, 256, [1, 2, 3]),
        };
        let (grid, reserved) = prepare_image_v1(&small, recipe, limits, &mut memory).unwrap();
        assert_eq!(pool.in_use(), reserved);
        // The payload keeps its bytes leased after the request's preparation
        // ends, until its last copy drops.
        let payload = memory.retain(reserved, grid.bf16_bytes()).unwrap();
        assert_eq!(pool.in_use(), payload.len() as u64);
        drop(memory);
        assert_eq!(pool.in_use(), payload.len() as u64);
        drop(payload);
        assert_eq!(pool.in_use(), 0);
    }

    /// Concurrent requests of near-limit images, four per request with a
    /// 16 MiP recipe and thirty-two with a 1 MiP one, never take more than
    /// the process budget, fail as client errors or as unavailable, and
    /// leave nothing reserved.
    #[test]
    fn concurrent_near_limit_images_stay_inside_the_process_budget() {
        use crate::media_memory::{
            MediaMemoryV1, PROCESS_MEDIA_PREPARATION_BYTES_V1, REQUEST_MEDIA_PREPARATION_BYTES_V1,
            RequestMediaMemoryV1,
        };
        let limits = image_limits();
        let pool = Arc::new(MediaMemoryV1::new(PROCESS_MEDIA_PREPARATION_BYTES_V1));
        let near_limit = declared_png(9459, 9459);
        let outcomes = std::thread::scope(|scope| {
            let workers = (0..16)
                .map(|worker| {
                    let (pool, near_limit) = (Arc::clone(&pool), &near_limit);
                    scope.spawn(move || {
                        let (recipe, images) = if worker % 2 == 0 {
                            (patch_recipe(16_777_216), 4)
                        } else {
                            (patch_recipe(1_048_576), 32)
                        };
                        let mut outcomes = Vec::new();
                        for _ in 0..8 {
                            let mut memory = RequestMediaMemoryV1::new(
                                Arc::clone(&pool),
                                REQUEST_MEDIA_PREPARATION_BYTES_V1,
                            );
                            for _ in 0..images {
                                let outcome =
                                    prepare_image_v1(near_limit, recipe, limits, &mut memory);
                                assert!(pool.in_use() <= PROCESS_MEDIA_PREPARATION_BYTES_V1);
                                if let Err(error) = outcome {
                                    outcomes.push(error);
                                    break;
                                }
                            }
                        }
                        outcomes
                    })
                })
                .collect::<Vec<_>>();
            workers
                .into_iter()
                .flat_map(|worker| worker.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(outcomes.len(), 16 * 8);
        assert!(outcomes.iter().all(|error| matches!(
            error,
            AdmissionError::Rejected(_) | AdmissionError::Unavailable(_)
        )));
        assert_eq!(pool.in_use(), 0);
    }

    #[test]
    fn image_recipe_reads_optional_resize_keys() {
        use runtime_media::recipe::Interpolation;
        let mut parameters = serde_json::json!({
            "factor": 32, "minimum_pixels": 65536, "maximum_pixels": 16777216,
            "patch": 16, "temporal_patch": 2, "merge": 2, "value_scale": 0.00392156862745098,
            "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]
        });
        let recipe = image_recipe_v1(&parameters).unwrap();
        assert_eq!(recipe.interpolation, Interpolation::Bilinear);
        assert!(recipe.upscale_small_side);
        parameters["interpolation"] = "bicubic-antialias".into();
        parameters["upscale_small_side"] = false.into();
        let recipe = image_recipe_v1(&parameters).unwrap();
        assert_eq!(recipe.interpolation, Interpolation::BicubicAntialias);
        assert!(!recipe.upscale_small_side);
        parameters["interpolation"] = "bilinear".into();
        assert_eq!(
            image_recipe_v1(&parameters).unwrap().interpolation,
            Interpolation::Bilinear
        );
        for (key, value) in [
            ("interpolation", serde_json::json!("bicubic")),
            ("interpolation", serde_json::json!(null)),
            ("upscale_small_side", serde_json::json!("false")),
        ] {
            let mut bad = parameters.clone();
            bad[key] = value;
            assert!(matches!(
                image_recipe_v1(&bad),
                Err(AdmissionError::InvalidPrompt)
            ));
        }
    }

    #[test]
    fn images_decode_by_mime_above_the_resize_budget() {
        let limits = runtime_media::MediaLimits {
            maximum_bytes: 64 << 20,
            maximum_pixels: MAXIMUM_DECODED_IMAGE_PIXELS_V1,
            maximum_audio_frames: 0,
            maximum_frames: VIDEO_MAXIMUM_FRAMES_V1,
            maximum_tokens: 1024,
            maximum_aspect_ratio: MAXIMUM_ASPECT_RATIO_V1,
        };
        let source = |mime: &str, bytes: Vec<u8>| runtime_media::MediaSource {
            mime: mime.to_owned(),
            bytes,
        };
        // 2048x1536 is three times the 1 MiP resize budget and must still be
        // accepted, as vLLM resizes it rather than refusing it.
        let png = solid_png(2048, 1536, [0, 0, 255]);
        let image = decode_image_v1(
            &source("image/png", png.clone()),
            limits,
            runtime_media::Alpha::CompositeWhite,
            runtime_media::Orientation::Exif,
        )
        .unwrap();
        assert_eq!((image.width, image.height), (2048, 1536));
        assert_eq!(&image.data[..3], [0, 0, 255]);
        let recipe = image_recipe_v1(&serde_json::json!({
            "factor": 32, "minimum_pixels": 8192, "maximum_pixels": 1048576,
            "patch": 16, "temporal_patch": 2, "merge": 2, "value_scale": 1.0,
            "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]
        }))
        .unwrap();
        let grid = runtime_media::recipe::image_grid(&image, recipe, limits).unwrap();
        assert_eq!(grid.grid, [1, 54, 72]);
        // A 160:1 strip resizes to 32x6400 in vLLM and must not be refused.
        let strip = decode_image_v1(
            &source("image/png", solid_png(6400, 40, [9; 3])),
            limits,
            runtime_media::Alpha::CompositeWhite,
            runtime_media::Orientation::Exif,
        );
        let grid = runtime_media::recipe::image_grid(&strip.unwrap(), recipe, limits).unwrap();
        assert_eq!(grid.grid, [1, 2, 400]);
        // An 18 MP 4500x4000 image is admitted and resized to 4320x3840 by a
        // 16 MiP recipe, as vLLM's processor resizes it. PIL's
        // MAX_IMAGE_PIXELS is the largest decode.
        let large = image_recipe_v1(&serde_json::json!({
            "factor": 32, "minimum_pixels": 65536, "maximum_pixels": 16777216,
            "patch": 16, "temporal_patch": 2, "merge": 2, "value_scale": 1.0,
            "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]
        }))
        .unwrap();
        assert_eq!(
            runtime_media::recipe::smart_resize(4000, 4500, large, limits),
            Ok((3840, 4320))
        );
        assert!(limits.check_image(9459, 9459).is_ok());
        assert!(limits.check_image(9460, 9459).is_err());
        // The MIME selects the decoder, so PNG bytes labeled JPEG reach the
        // JPEG decoder and fail there; non-image MIME types never decode.
        assert!(matches!(
            decode_image_v1(&source("image/jpeg", png.clone()), limits, runtime_media::Alpha::CompositeWhite, runtime_media::Orientation::Exif),
            Err(AdmissionError::Rejected(message)) if message.contains("JPEG")
        ));
        assert!(matches!(
            decode_image_v1(
                &source("audio/wav", png),
                limits,
                runtime_media::Alpha::CompositeWhite,
                runtime_media::Orientation::Exif
            ),
            Err(AdmissionError::InvalidPrompt)
        ));
    }

    #[test]
    fn video_input_errors_map_to_client_and_service_statuses() {
        assert!(matches!(
            video_admission_error_v1(crate::video::VideoErrorV1::Decode("bad input".to_owned())),
            AdmissionError::Rejected(_)
        ));
        assert!(matches!(
            video_admission_error_v1(crate::video::VideoErrorV1::Unavailable(
                "no decoder".to_owned()
            )),
            AdmissionError::Unavailable(_)
        ));
        for error in [
            crate::video::VideoErrorV1::Busy("full".to_owned()),
            crate::video::VideoErrorV1::TimedOut(std::time::Duration::from_secs(30)),
            crate::video::VideoErrorV1::Cancelled,
        ] {
            assert!(matches!(
                video_admission_error_v1(error),
                AdmissionError::Unavailable(_)
            ));
        }
    }

    /// Prepared media is admitted exactly when preparing a part could have
    /// produced it: the resizes of a sweep of sources, video grids of 1 to 32
    /// frames and the log-mel frame counts are admitted; grids under
    /// the minimum pixels or past the aspect limit, over the maximum, with a
    /// nonzero unused field, and frame counts outside the clip bounds are
    /// refused.
    #[test]
    fn prepared_media_is_admitted_exactly_as_parts_emit_it() {
        let modality = |kind: &str, encoder: u16, tag: &str, parameters: serde_json::Value| {
            serde_json::json!({
                "kind": kind, "part_types": [format!("{kind}_url")], "maximum_items": 1,
                "maximum_bytes": 64, "maximum_tokens": 8192, "encoder": encoder,
                "recipe": {"tag": tag, "parameters": parameters},
                "render": {"pad": format!("<{kind}>")}
            })
        };
        let image = |minimum: u64, maximum: u64| {
            serde_json::json!({
                "factor": 32, "minimum_pixels": minimum, "maximum_pixels": maximum,
                "patch": 16, "temporal_patch": 2, "merge": 2, "value_scale": 1.0,
                "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]
            })
        };
        let declared: FrontendMultimodalV1 = serde_json::from_value(serde_json::json!({
            "arena": {"rows": 64, "page_rows": 64, "hidden": 8, "element": "bf16"},
            "encoders": [],
            "modalities": [
                modality("image", 0, "image.patch-grid.v1", image(8192, 1_048_576)),
                modality("video", 0, "video.timestamped-grids.v1", serde_json::json!({
                    "maximum_frames": 32, "total_max_pixels": 268_435_456,
                    "image": image(8192, 262_144)
                })),
                modality("audio", 1, "audio.log-mel.v1", serde_json::json!({
                    "sample_rate": 24_000, "maximum_seconds": 600, "sinc_width": 6,
                    "rolloff": 0.99, "n_fft": 960, "window": 960, "hop": 240, "mel_bins": 128,
                    "f_min": 0.0, "f_max": 12_000.0, "log_floor": 1e-7, "first_kernel": 3,
                    "second_kernel": 3, "second_stride": 2, "pool": 2, "group": 4
                })),
            ]
        }))
        .unwrap();
        let input = |encoder, geometry: [u32; 4]| MediaInputV1 {
            key: [0; 32],
            encoder,
            geometry,
            output_rows: if encoder == 1 {
                runtime_media::audio::strided_group_tokens(
                    geometry[0] as usize,
                    audio_recipe_v1(&declared.modalities[2].recipe.parameters).unwrap(),
                )
                .unwrap() as u32
            } else {
                geometry[0] * (geometry[1] / 2) * (geometry[2] / 2)
            },
            payload: vec![0_u8; 4].into(),
        };
        let admits = |encoder, geometry| {
            admit_prepared_media_v1(Some(&declared), &[input(encoder, geometry)]).is_ok()
        };
        let [image, video, audio] = &declared.modalities[..] else {
            panic!("three modalities");
        };

        // Image resizes of sources around every bound, both orientations.
        let recipe = image_recipe_v1(&image.recipe.parameters).unwrap();
        let limits = part_limits_v1(image);
        let descriptor_recipes = declared.modalities.iter().map(validate_recipe_outputs_v1);
        assert!(descriptor_recipes.collect::<Result<Vec<_>, _>>().is_ok());
        let mut sides: Vec<u32> = (1..=40).collect();
        for edge in [64, 128, 256, 512, 1024, 2048, 4096, 8192] {
            sides.extend([edge - 1, edge, edge + 1]);
        }
        let mut admitted = 0;
        for &height in &sides {
            for &width in &sides {
                if let Ok((h, w)) =
                    runtime_media::recipe::smart_resize(height, width, recipe, limits)
                {
                    assert!(
                        admits(0, [1, h / 16, w / 16, 0]),
                        "{height}x{width} -> {h}x{w}"
                    );
                    admitted += 1;
                }
            }
        }
        assert!(admitted > 1000);

        // Video grids of 1 to 32 frames (odd counts repeat the last), small
        // frames scaled up to the minimum and wide ones.
        let parameters = &video.recipe.parameters;
        let frame = image_recipe_v1(&parameters["image"]).unwrap();
        let limits = video_limits_v1(video, &parameters["image"]);
        for [width, height] in [[40, 30], [2, 400], [400, 2], [640, 480], [4096, 64]] {
            let rgb = runtime_media::RgbImage::new(
                width,
                height,
                vec![9; (width * height * 3) as usize],
                limits,
            )
            .unwrap();
            for count in [1, 2, 3, 16, 17, 31, 32] {
                let frames: Vec<_> = (0..count).map(|i| (f64::from(i), rgb.clone())).collect();
                let grids = runtime_media::recipe::video_grids(&frames, frame, 268_435_456, limits)
                    .unwrap();
                let [_, h, w] = grids[0].grid.grid;
                let t = grids.iter().map(|grid| grid.grid.grid[0]).sum();
                assert!(
                    admits(0, [t, h, w, 0]),
                    "{count} frames of {width}x{height}"
                );
            }
        }

        // Every frame count a clip of up to 600 s yields, and none other.
        let recipe = audio_recipe_v1(&audio.recipe.parameters).unwrap();
        let frames = runtime_media::audio::emitted_frames(recipe, part_limits_v1(audio));
        assert_eq!(frames, 3..=60_001);
        for count in [3, 4, 1595, 6001, 60_001] {
            assert!(admits(1, [count, 128, 0, 0]), "{count}");
        }

        // Never emitted: frames under the minimum pixels ([2, 2, 2] is 1,024
        // pixels a frame), past the 200 aspect limit, over the maximum, of
        // odd blocks or 17 grids; unused fields set; too short or long a
        // clip, or other bins.
        for geometry in [
            [2, 2, 2, 0],
            [1, 2, 2, 0],
            [1, 2, 2048, 0],
            [1, 2048, 2, 0],
            [1, 66, 64, 0],
            [2, 34, 32, 0],
            [1, 3, 4, 0],
            [17, 2, 64, 0],
            [1, 64, 64, 1],
        ] {
            assert!(!admits(0, geometry), "{geometry:?}");
        }
        for geometry in [
            [1, 128, 0, 0],
            [2, 128, 0, 0],
            [60_002, 128, 0, 0],
            [100, 64, 0, 0],
            [100, 128, 1, 0],
            [100, 128, 0, 1],
        ] {
            assert!(!admits(1, geometry), "{geometry:?}");
        }
        assert!(!admits(2, [1, 2, 2, 0]));
        assert!(admit_prepared_media_v1(None, &[]).is_ok());

        // A refusal names the item, its geometry and its encoder, never
        // tokenization; the transport answers it with 400 and this message.
        let refusal =
            |declared, media: &[MediaInputV1]| match admit_prepared_media_v1(declared, media)
                .unwrap_err()
            {
                AdmissionError::Rejected(message) => message,
                other => panic!("{other:?}"),
            };
        let media = [input(0, [1, 32, 32, 0]), input(0, [2, 2, 2, 0])];
        assert_eq!(
            refusal(Some(&declared), &media),
            "media item 1: geometry [2, 2, 2, 0] with 2 output rows for encoder 0 is not one its media recipes \
             produce"
        );
        assert_eq!(
            refusal(None, &media[..1]),
            "media inputs are not supported by this model"
        );
    }

    /// The token-budget recipe prepares a part as runtime-media pads it,
    /// with transparency handled as declared, admits exactly its canvases
    /// (its one-block fallback included) as prepared media, and refuses
    /// declarations the host cannot honor, beside video among them.
    #[test]
    fn token_budget_images_prepare_and_admit_their_canvases() {
        let declared = |parameters: serde_json::Value, video: bool| {
            let modality = |kind: &str, tag: &str, parameters: serde_json::Value| {
                serde_json::json!({
                    "kind": kind, "part_types": [format!("{kind}_url")], "maximum_items": 1,
                    "maximum_bytes": 1 << 20, "maximum_tokens": 8192, "encoder": 0,
                    "recipe": {"tag": tag, "parameters": parameters},
                    "render": {"pad": format!("<{kind}>")}
                })
            };
            let mut modalities = vec![modality("image", TOKEN_BUDGET_TAG_V1, parameters)];
            if video {
                modalities.push(modality(
                    "video",
                    "video.timestamped-grids.v1",
                    serde_json::json!({}),
                ));
            }
            serde_json::from_value::<FrontendMultimodalV1>(serde_json::json!({
                "arena": {"rows": 64, "page_rows": 64, "hidden": 8, "element": "bf16"},
                "encoders": [],
                "modalities": modalities
            }))
            .unwrap()
        };
        let parameters = serde_json::json!({
            "patch": 14, "temporal_patch": 2, "merge": 2, "minimum_tokens": 16,
            "maximum_tokens": 8000, "value_scale": 0.00392156862745098,
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711], "alpha_background": "none"
        });
        let with = |key: &str, value: serde_json::Value| {
            let mut parameters = parameters.clone();
            parameters[key] = value;
            declared(parameters, false)
        };
        let media = declared(parameters.clone(), false);
        validate_multimodal_recipes_v1(&media).unwrap();
        let modality = &media.modalities[0];
        let limits = part_limits_v1(modality);
        // A PNG whose tRNS color key is its own color: fully transparent
        // when composited, its color kept when alpha is discarded.
        let png = solid_png(301, 199, [200, 100, 50]);
        let mut keyed = png[..33].to_vec();
        png_chunk(b"tRNS", &[0, 200, 0, 100, 0, 50], &mut keyed);
        keyed.extend_from_slice(&png[33..]);
        let source = runtime_media::MediaSource {
            mime: "image/png".to_owned(),
            bytes: keyed,
        };
        let pool = Arc::new(MediaMemoryV1::new(1 << 30));
        let mut memory = RequestMediaMemoryV1::new(Arc::clone(&pool), u64::MAX);
        let mut prepared = |modality: &runtime_frontend::FrontendMediaModalityV1, source| {
            let recipe = declared_image_recipe_v1(&modality.recipe).unwrap();
            prepare_image_v1(source, recipe, part_limits_v1(modality), &mut memory)
                .unwrap()
                .0
        };
        let padded = |alpha| {
            let decoded =
                decode_image_v1(&source, limits, alpha, runtime_media::Orientation::Exif).unwrap();
            let ImageRecipeV1::TokenBudget(recipe, _) =
                declared_image_recipe_v1(&modality.recipe).unwrap()
            else {
                panic!("the tag selects the token-budget recipe");
            };
            let grid = runtime_media::recipe::padded_token_budget_grid(&decoded, recipe, limits);
            (decoded.data[..3].to_vec(), grid.unwrap())
        };
        let (discarded, expected) = padded(runtime_media::Alpha::Discard);
        assert_eq!(discarded, [200, 100, 50]);
        let grid = prepared(modality, &source);
        assert_eq!(grid, expected);
        // Declared white compositing turns the image white.
        let (composited, white) = padded(runtime_media::Alpha::CompositeWhite);
        assert_eq!(composited, [255; 3]);
        let declared_white = with("alpha_background", "white".into());
        assert_eq!(prepared(&declared_white.modalities[0], &source), white);
        assert_ne!(white, expected);
        // 199 x 301 pads to 224 x 308: 16 x 22 patches, 88 tokens.
        assert_eq!((grid.grid, grid.tokens), ([1, 16, 22], 88));
        let emits = |geometry| modality_emits_v1(modality, geometry);
        assert!(emits([1, 16, 22, 0]));
        // Under learned delimiters the span is the 8 x 11 merged cells, a
        // newline after each of the 8 cell rows, and a start and an end row:
        // 98 rows, which the token limit counts.
        let recipe = declared_image_recipe_v1(&modality.recipe).unwrap();
        let merge = recipe_merge_v1(recipe);
        assert_eq!(merge, 2);
        assert_eq!(
            image_span_rows_v1(modality, merge, grid.grid),
            Some(grid.tokens)
        );
        assert_eq!(
            planned_image_grid_v1(301, 199, recipe, limits).unwrap(),
            grid.grid
        );
        let mut delimited = modality.clone();
        delimited.layout = Some(runtime_frontend::FrontendMediaLayoutV1::LearnedDelimitersV1);
        assert_eq!(image_span_rows_v1(&delimited, merge, grid.grid), Some(98));
        // A partial merge block is a cell: 40 x 40 patches merged by 3 are
        // 14 x 14 cells, 14 * 15 + 2 rows.
        assert_eq!(image_span_rows_v1(&delimited, 3, [1, 40, 40]), Some(212));
        delimited.maximum_tokens = 98;
        assert!(modality_emits_v1(&delimited, [1, 16, 22, 0]));
        check_image_span_v1(&source, &delimited, recipe, limits).unwrap();
        delimited.maximum_tokens = 97;
        assert!(!modality_emits_v1(&delimited, [1, 16, 22, 0]));
        // refused by name before the image decodes, counting the delimiters
        let Err(AdmissionError::Rejected(message)) =
            check_image_span_v1(&source, &delimited, recipe, limits)
        else {
            panic!("an over-long span is refused by name");
        };
        assert!(
            message.contains("spans 98 rows (88 cells and 10 learned delimiter rows), above the modality's 97 tokens"),
            "{message}"
        );
        // Unaligned, below 16 or above 8,000 tokens, two grids, or a
        // nonzero unused field.
        for geometry in [
            [1, 16, 21, 0],
            [1, 2, 2, 0],
            [1, 6, 4, 0],
            [1, 162, 200, 0],
            [2, 16, 22, 0],
            [1, 16, 22, 1],
        ] {
            assert!(!emits(geometry), "{geometry:?}");
        }
        // Under an aspect limit of 300,000, a 1 x 224,001 source is too wide
        // for any canvas within 8,000 tokens and keeps HF's one-block
        // fallback, 2 x 2 patches below the 16-token minimum: it is admitted.
        let wide = with("maximum_aspect_ratio", 300_000.0.into());
        validate_multimodal_recipes_v1(&wide).unwrap();
        let strip = runtime_media::MediaSource {
            mime: "image/png".to_owned(),
            bytes: solid_png(224_001, 1, [9; 3]),
        };
        assert_eq!(prepared(&wide.modalities[0], &strip).grid, [1, 2, 2]);
        assert!(modality_emits_v1(&wide.modalities[0], [1, 2, 2, 0]));
        // Declarations the host cannot honor are refused at load.
        let mut missing = parameters.clone();
        missing.as_object_mut().unwrap().remove("alpha_background");
        let mut narrow = declared(parameters.clone(), false);
        narrow.modalities[0].maximum_tokens = 4000;
        let refusals = [
            (declared(missing, false), "requires alpha_background"),
            (
                with("alpha_background", "black".into()),
                "requires alpha_background",
            ),
            (
                with("interpolation", "bilinear".into()),
                "has no parameter interpolation",
            ),
            (
                with("maximum_aspect_ratio", "2".into()),
                "finite number of at least 1",
            ),
            (
                with("maximum_aspect_ratio", serde_json::Value::Null),
                "finite number",
            ),
            (with("maximum_aspect_ratio", 0.5.into()), "finite number"),
            // Outside the domain whose canvases admission is shown to
            // admit: 21 x 28 would pad to one block at 16 to 16 tokens, and
            // 1 x 200 to 200 x 200 with patch 100 at 2 to 16 tokens.
            (with("temporal_patch", 33.into()), "temporal_patch 2 only"),
            (with("patch", 100.into()), "patch 14, merge 2"),
            (
                with("maximum_tokens", 9000.into()),
                "maximum_tokens 256 to 8000",
            ),
            (
                with("maximum_tokens", 16.into()),
                "maximum_tokens 256 to 8000",
            ),
            (with("minimum_tokens", 17.into()), "minimum_tokens 1 to 16"),
            (narrow, "exceeds the modality's 4000 tokens"),
            (
                with("value_scale", (-1.0).into()),
                "invalid image normalization",
            ),
            (with("mean", serde_json::json!([0.5])), "cannot be read"),
            (declared(parameters, true), "no video form"),
        ];
        for (media, message) in refusals {
            let error = validate_multimodal_recipes_v1(&media).unwrap_err();
            assert!(error.contains(message), "{error}");
        }
    }

    #[test]
    fn centered_pad_prepared_rows_must_match_the_delimited_span() {
        let mut declared: FrontendMultimodalV1 = serde_json::from_value(serde_json::json!({
            "arena": {"rows": 1024, "page_rows": 64, "hidden": 8, "element": "bf16"},
            "encoders": [],
            "modalities": [{
                "kind": "image", "part_types": ["image_url"], "maximum_items": 1,
                "maximum_bytes": 1 << 20, "maximum_tokens": 1024, "encoder": 0,
                "recipe": {"tag": CENTERED_PAD_TAG_V1, "parameters": {
                    "patch": 14, "downsample": 3, "minimum_pixels": 295936, "maximum_tokens": 1024,
                    "pad": [127, 127, 127], "value_divisor": 255.0,
                    "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "alpha_background": "none"
                }},
                "layout": "learned-delimiters.v1", "render": {"pad": "<image>"}
            }]
        }))
        .unwrap();
        validate_multimodal_recipes_v1(&declared).unwrap();
        let mut input = MediaInputV1 {
            key: [0; 32],
            encoder: 0,
            geometry: [1, 39, 39, 0],
            output_rows: 184,
            payload: vec![0_u8; 39 * 39 * 588 * 2].into(),
        };
        assert_eq!(
            image_span_rows_v1(&declared.modalities[0], 3, [1, 39, 39]),
            Some(184)
        );
        admit_prepared_media_v1(Some(&declared), &[input.clone()]).unwrap();
        for rows in [1025, 185, 183, 0] {
            input.output_rows = rows;
            assert!(
                admit_prepared_media_v1(Some(&declared), &[input.clone()]).is_err(),
                "{rows}"
            );
        }
        input.output_rows = 184;
        declared.modalities[0].maximum_tokens = 183;
        assert!(admit_prepared_media_v1(Some(&declared), &[input]).is_err());
    }

    #[test]
    fn centered_pad_reserves_cmyk_decode_storage_before_decoding() {
        let source = runtime_media::MediaSource {
            mime: "image/jpeg".into(),
            bytes: include_bytes!(
                "../../runtime-media/tests/fixtures/centered-pad-jpeg-cmyk-256x256.jpg"
            )
            .to_vec(),
        };
        let recipe = runtime_media::recipe::CenteredPadRecipe {
            patch: 1,
            downsample: 1,
            minimum_pixels: 0,
            maximum_tokens: 4,
            pad: [127; 3],
            value_divisor: 255.0,
            mean: [0.5; 3],
            std: [0.5; 3],
        };
        let limits = runtime_media::MediaLimits {
            maximum_bytes: 1 << 20,
            maximum_pixels: 1 << 24,
            maximum_audio_frames: 0,
            maximum_frames: 1,
            maximum_tokens: 4,
            maximum_aspect_ratio: 200.0,
        };
        let wrapped = ImageRecipeV1::CenteredPad(recipe, runtime_media::Alpha::Discard);
        // The previous 1,273,686-byte preparation bound plus this source
        // fits, but the actual decoder buffers alone need 1,507,328 bytes.
        let pool = Arc::new(MediaMemoryV1::new(1_500_000));
        let mut memory = RequestMediaMemoryV1::new(Arc::clone(&pool), u64::MAX);
        assert!(matches!(
            prepare_image_v1(&source, wrapped, limits, &mut memory),
            Err(AdmissionError::Unavailable(_))
        ));
        let pool = Arc::new(MediaMemoryV1::new(2_000_000));
        let mut memory = RequestMediaMemoryV1::new(pool, u64::MAX);
        let (grid, reserved) = prepare_image_v1(&source, wrapped, limits, &mut memory).unwrap();
        assert_eq!(grid.grid, [1, 1, 1]);
        assert!(reserved >= 1_507_328 + source.bytes.len() as u64);
    }

    #[test]
    fn centered_pad_refuses_16_bit_png_before_preparation() {
        // A complete 1 x 1, 16-bit RGB PNG. This recipe has no matching
        // 16-bit decode path and must not silently truncate the samples.
        let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
        let mut header = png_header(1, 1);
        header[8] = 16;
        png_chunk(b"IHDR", &header, &mut bytes);
        png_chunk(
            b"IDAT",
            &[
                0x78, 0x01, 1, 7, 0, 0xf8, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 1,
            ],
            &mut bytes,
        );
        png_chunk(b"IEND", &[], &mut bytes);
        let source = runtime_media::MediaSource {
            mime: "image/png".into(),
            bytes,
        };
        let recipe = runtime_media::recipe::CenteredPadRecipe {
            patch: 14,
            downsample: 3,
            minimum_pixels: 295936,
            maximum_tokens: 1024,
            pad: [127; 3],
            value_divisor: 255.0,
            mean: [0.5; 3],
            std: [0.5; 3],
        };
        let limits = runtime_media::MediaLimits {
            maximum_bytes: 1 << 20,
            maximum_pixels: 1 << 24,
            maximum_audio_frames: 0,
            maximum_frames: 1,
            maximum_tokens: 1024,
            maximum_aspect_ratio: 200.0,
        };
        let mut memory = RequestMediaMemoryV1::new(Arc::new(MediaMemoryV1::new(0)), u64::MAX);
        let error = prepare_image_v1(
            &source,
            ImageRecipeV1::CenteredPad(recipe, runtime_media::Alpha::Discard),
            limits,
            &mut memory,
        )
        .unwrap_err();
        assert!(
            matches!(error, AdmissionError::Rejected(ref message) if message == "unsupported PNG IHDR"),
            "{error:?}"
        );
    }

    /// The centered-pad recipe prepares a part as runtime-media centers it,
    /// with transparency handled as declared, counts its span under learned
    /// delimiters, admits its canvases as prepared media (one that rounding
    /// leaves below the minimum pixels and past the source's aspect
    /// included), and refuses declarations the host cannot honor.
    #[test]
    fn centered_pad_images_prepare_and_admit_their_canvases() {
        let declared = |parameters: serde_json::Value, layout: bool, video: bool| {
            let modality = |kind: &str, tag: &str, parameters: serde_json::Value| {
                let mut modality = serde_json::json!({
                    "kind": kind, "part_types": [format!("{kind}_url")], "maximum_items": 1,
                    "maximum_bytes": 1 << 20, "maximum_tokens": 1024, "encoder": 0,
                    "recipe": {"tag": tag, "parameters": parameters},
                    "render": {"pad": format!("<{kind}>")}
                });
                if layout && kind == "image" {
                    modality["layout"] = "learned-delimiters.v1".into();
                }
                modality
            };
            let mut modalities = vec![modality("image", CENTERED_PAD_TAG_V1, parameters)];
            if video {
                modalities.push(modality(
                    "video",
                    "video.timestamped-grids.v1",
                    serde_json::json!({}),
                ));
            }
            serde_json::from_value::<FrontendMultimodalV1>(serde_json::json!({
                "arena": {"rows": 64, "page_rows": 64, "hidden": 8, "element": "bf16"},
                "encoders": [],
                "modalities": modalities
            }))
            .unwrap()
        };
        let parameters = serde_json::json!({
            "patch": 14, "downsample": 3, "minimum_pixels": 295_936, "maximum_tokens": 1024,
            "pad": [127, 127, 127], "value_divisor": 255.0, "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5], "alpha_background": "none"
        });
        let with = |key: &str, value: serde_json::Value| {
            let mut parameters = parameters.clone();
            parameters[key] = value;
            declared(parameters, true, false)
        };
        let media = declared(parameters.clone(), true, false);
        validate_multimodal_recipes_v1(&media).unwrap();
        let modality = &media.modalities[0];
        let limits = part_limits_v1(modality);
        let ImageRecipeV1::CenteredPad(recipe, alpha) =
            declared_image_recipe_v1(&modality.recipe).unwrap()
        else {
            panic!("the tag selects the centered-pad recipe");
        };
        assert_eq!(alpha, runtime_media::Alpha::Discard);
        assert_eq!(recipe.pad, [127; 3]);
        assert_eq!(recipe.value_divisor, 255.0);
        // A PNG whose tRNS color key is its own color: fully transparent
        // when composited, its color kept when alpha is discarded.
        let png = solid_png(301, 199, [200, 100, 50]);
        let mut keyed = png[..33].to_vec();
        png_chunk(b"tRNS", &[0, 200, 0, 100, 0, 50], &mut keyed);
        keyed.extend_from_slice(&png[33..]);
        let source = runtime_media::MediaSource {
            mime: "image/png".to_owned(),
            bytes: keyed,
        };
        let pool = Arc::new(MediaMemoryV1::new(1 << 30));
        let mut memory = RequestMediaMemoryV1::new(Arc::clone(&pool), u64::MAX);
        let mut prepared = |modality: &runtime_frontend::FrontendMediaModalityV1| {
            let recipe = declared_image_recipe_v1(&modality.recipe).unwrap();
            prepare_image_v1(&source, recipe, part_limits_v1(modality), &mut memory)
                .unwrap()
                .0
        };
        let centered = |alpha| {
            let decoded =
                decode_image_v1(&source, limits, alpha, runtime_media::Orientation::Stored)
                    .unwrap();
            runtime_media::recipe::centered_pad_grid(&decoded, recipe, limits).unwrap()
        };
        let grid = prepared(modality);
        assert_eq!(grid, centered(runtime_media::Alpha::Discard));
        let declared_white = with("alpha_background", "white".into());
        let white = prepared(&declared_white.modalities[0]);
        assert_eq!(white, centered(runtime_media::Alpha::CompositeWhite));
        assert_ne!(white, grid);
        // 301 x 199 scales up to 448 x 672 pixels, as the reference plans
        // it: 32 x 48 patches and 11 x 16 tokens.
        assert_eq!((grid.grid, grid.tokens), ([1, 32, 48], 176));
        let wrapped = ImageRecipeV1::CenteredPad(recipe, alpha);
        assert_eq!(recipe_merge_v1(wrapped), 3);
        assert_eq!(
            planned_image_grid_v1(301, 199, wrapped, limits).unwrap(),
            grid.grid
        );
        // The span is the recipe's budget count: 11 * (16 + 1) + 2.
        assert_eq!(image_span_rows_v1(modality, 3, grid.grid), Some(189));
        check_image_span_v1(&source, modality, wrapped, limits).unwrap();
        // A JPEG tagged with EXIF orientation 6 keeps its stored 8 x 16, as
        // the reference reads it: planned and prepared at 770 x 392 pixels
        // (55 x 28 patches), not as the rotated 16 x 8. Other recipes keep
        // transposing into the tagged orientation.
        let jpeg = runtime_media::MediaSource {
            mime: "image/jpeg".to_owned(),
            bytes: include_bytes!(
                "../../runtime-media/tests/fixtures/centered-pad-jpeg-exif6-8x16.jpg"
            )
            .to_vec(),
        };
        assert_eq!(
            image_orientation_v1(wrapped),
            runtime_media::Orientation::Stored
        );
        let token_budget = token_budget_recipe_v1(&serde_json::json!({
            "patch": 14, "temporal_patch": 2, "merge": 2, "minimum_tokens": 16,
            "maximum_tokens": 8000, "value_scale": 1.0, "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0]
        }))
        .unwrap();
        assert_eq!(
            image_orientation_v1(ImageRecipeV1::TokenBudget(token_budget, alpha)),
            runtime_media::Orientation::Exif
        );
        let mut jpeg_memory = RequestMediaMemoryV1::new(Arc::clone(&pool), u64::MAX);
        let (stored, _) = prepare_image_v1(&jpeg, wrapped, limits, &mut jpeg_memory).unwrap();
        assert_eq!(stored.grid, [1, 55, 28]);
        assert_eq!(
            planned_image_grid_v1(8, 16, wrapped, limits).unwrap(),
            stored.grid
        );
        check_image_span_v1(&jpeg, modality, wrapped, limits).unwrap();
        let emits = |geometry| modality_emits_v1(modality, geometry);
        assert!(emits([1, 32, 48, 0]));
        // A 79 x 15000 source solves to 569 x 2 patches (7966 x 28 pixels):
        // under the minimum pixels and past the 200:1 aspect limit, it is
        // still admitted.
        let tall = runtime_media::recipe::centered_pad_plan(15000, 79, recipe, limits).unwrap();
        assert_eq!(tall.canvas, (7966, 28));
        assert!(emits([1, 569, 2, 0]));
        // Over the budget (34 x 30 tokens span 1,056), two grids, or a
        // nonzero unused field.
        for geometry in [
            [1, 102, 90, 0],
            [2, 32, 48, 0],
            [1, 32, 48, 1],
            [1, 0, 48, 0],
        ] {
            assert!(!emits(geometry), "{geometry:?}");
        }
        // Declarations the host cannot honor are refused at load.
        let mut missing = parameters.clone();
        missing.as_object_mut().unwrap().remove("alpha_background");
        let mut narrow = declared(parameters.clone(), true, false);
        narrow.modalities[0].maximum_tokens = 1000;
        let mut wide = with("maximum_tokens", 100_000.into());
        wide.modalities[0].maximum_tokens = 200_000;
        let refusals = [
            (declared(missing, true, false), "requires alpha_background"),
            (
                with("interpolation", "bicubic".into()),
                "has no parameter interpolation",
            ),
            (
                declared(parameters.clone(), false, false),
                "learned-delimiters.v1 layout",
            ),
            (narrow, "exceeds the modality's 1000 tokens"),
            (with("patch", 0.into()), "invalid centered pad recipe"),
            (
                with("std", serde_json::json!([0.5, 0.0, 0.5])),
                "normalization",
            ),
            (wide, "decoded pixels"),
            (
                declared(parameters.clone(), true, true),
                "has no video form",
            ),
        ];
        for (media, message) in refusals {
            let error = validate_multimodal_recipes_v1(&media).unwrap_err();
            assert!(error.contains(message), "{error}");
        }
        // Parameters that do not parse never select a recipe.
        let mut bad_pad = modality.recipe.clone();
        bad_pad.parameters["pad"] = serde_json::json!([127, 127, 300]);
        assert!(declared_image_recipe_v1(&bad_pad).is_err());
    }

    /// A descriptor whose recipe can emit a size that is not its own resize
    /// is refused at load: with 8,192 pixels as both bounds, a 36 x 280 image
    /// floors to 32 x 224 (7,168 pixels), a grid the image recipe emits but
    /// would not admit as prepared media.
    #[test]
    fn recipes_whose_outputs_are_not_their_own_resize_are_refused() {
        let declared = |kind: &str, tag: &str, parameters: serde_json::Value| {
            serde_json::from_value::<FrontendMultimodalV1>(serde_json::json!({
                "arena": {"rows": 64, "page_rows": 64, "hidden": 8, "element": "bf16"},
                "encoders": [],
                "modalities": [{
                    "kind": kind, "part_types": ["part"], "maximum_items": 1,
                    "maximum_bytes": 64, "maximum_tokens": 8192, "encoder": 0,
                    "recipe": {"tag": tag, "parameters": parameters},
                    "render": {"pad": "<pad>"}
                }]
            }))
            .unwrap()
        };
        let image = |minimum: u64, maximum: u64| {
            serde_json::json!({
                "factor": 32, "minimum_pixels": minimum, "maximum_pixels": maximum,
                "patch": 16, "temporal_patch": 2, "merge": 2, "value_scale": 1.0,
                "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]
            })
        };
        let narrow = declared("image", "image.patch-grid.v1", image(8192, 8192));
        let modality = &narrow.modalities[0];
        let recipe = image_recipe_v1(&modality.recipe.parameters).unwrap();
        let limits = part_limits_v1(modality);
        let (h, w) = runtime_media::recipe::smart_resize(36, 280, recipe, limits).unwrap();
        assert_eq!((h, w), (32, 224));
        let error = validate_recipe_outputs_v1(modality).unwrap_err();
        assert!(
            error.contains("floor below its 8192 minimum pixels"),
            "{error}"
        );
        let wide = declared("image", "image.patch-grid.v1", image(8192, 1_048_576));
        assert!(validate_recipe_outputs_v1(&wide.modalities[0]).is_ok());

        // A video's frame budget shrinks with its frame count: sharing
        // 268,435,456 pixels, every count up to 32 keeps the 262,144-pixel
        // maximum; sharing 1,048,576, ten or more frames take too narrow a
        // budget over the 8,192-pixel minimum.
        let video = |total: u64| {
            declared(
                "video",
                "video.timestamped-grids.v1",
                serde_json::json!({
                    "maximum_frames": 32, "total_max_pixels": total,
                    "image": image(8192, 262_144)
                }),
            )
        };
        assert!(validate_recipe_outputs_v1(&video(268_435_456).modalities[0]).is_ok());
        let error = validate_recipe_outputs_v1(&video(1 << 20).modalities[0]).unwrap_err();
        assert!(
            error.contains("at 10 frames: image resize recipe can floor below"),
            "{error}"
        );
    }

    #[test]
    fn media_expansion_splices_pad_runs_and_records_exact_spans() {
        use runtime_frontend::{
            FrontendBoundaryTokenV1, FrontendMediaArenaV1, FrontendMediaElementV1,
            FrontendMediaEncoderV1, FrontendMediaKindV1, FrontendMediaModalityV1,
            FrontendMediaRecipeV1, FrontendMediaRenderV1,
        };
        let mut declared = FrontendMultimodalV1 {
            arena: FrontendMediaArenaV1 {
                rows: 64,
                page_rows: 64,
                hidden: 8,
                element: FrontendMediaElementV1::Bf16,
            },
            encoders: vec![FrontendMediaEncoderV1 {
                name: "image-encoder".to_owned(),
                payload_element: FrontendMediaElementV1::Bf16,
                payload_columns: 4,
                maximum_payload_bytes: 64,
                cost_per_row_milli: 1,
            }],
            modalities: vec![FrontendMediaModalityV1 {
                kind: FrontendMediaKindV1::Image,
                part_types: vec!["image_url".to_owned()],
                maximum_items: 1,
                maximum_bytes: 64,
                maximum_tokens: 4,
                recipe: FrontendMediaRecipeV1 {
                    tag: "image.patch-grid.v1".to_owned(),
                    parameters: serde_json::json!({}),
                },
                encoder: 0,
                render: FrontendMediaRenderV1 {
                    pad: "<pad>".to_owned(),
                    grid: None,
                },
                positions: None,
                layout: None,
            }],
        };
        let vocabulary = FrontendVocabularyV1 {
            size: 256,
            bos_token_id: 0,
            eos_token_id: 1,
            stop_token_ids: vec![1],
            boundary_tokens: vec![FrontendBoundaryTokenV1 {
                text: "<pad>".to_owned(),
                id: 7,
            }],
            prompt_special_tokens: vec!["<pad>".to_owned()],
        };
        let mut media = vec![FrontendMediaPartV1 {
            kind: FrontendMediaPartKindV1::Image,
            part_type: "image_url".to_owned(),
            source: "data:image/png;base64,AAAA".to_owned(),
            format: None,
        }];
        let segments = vec![
            PromptSegmentV1::Text("ab".to_owned()),
            PromptSegmentV1::Media(0),
            PromptSegmentV1::Text("z".to_owned()),
        ];
        let expand = |rows| {
            expand_media_chat_v1(
                &declared,
                &vocabulary,
                &segments,
                &media,
                |text| Ok(text.bytes().map(u32::from).collect()),
                move |_, _| {
                    Ok(MediaInputV1 {
                        key: [1; 32],
                        encoder: 0,
                        geometry: [1, 1, 1, 1],
                        output_rows: rows,
                        payload: vec![0_u8; 8].into(),
                    }
                    .into())
                },
            )
        };
        let (tokens, inputs, spans, positions) = expand(3).expect("three image tokens fit");
        assert!(positions.is_empty(), "no rule keeps implied positions");
        assert_eq!(tokens, [97, 98, 7, 7, 7, 122]);
        assert_eq!(inputs[0].output_rows, 3);
        assert_eq!(
            spans,
            [MediaSpanV1 {
                media: 0,
                prompt_row: 2,
                rows: 3,
                media_row: 0
            }]
        );
        assert!(expand(5).is_err());
        // A learned-delimiter span is still one pad run: the encoder writes
        // its delimiter rows with its cells.
        let mut delimited = declared.clone();
        delimited.modalities[0].layout =
            Some(runtime_frontend::FrontendMediaLayoutV1::LearnedDelimitersV1);
        let (tokens, _, spans, _) = expand_media_chat_v1(
            &delimited,
            &vocabulary,
            &segments,
            &media,
            |text| Ok(text.bytes().map(u32::from).collect()),
            |_, _| {
                Ok(MediaInputV1 {
                    key: [1; 32],
                    encoder: 0,
                    geometry: [1, 1, 1, 1],
                    output_rows: 4,
                    payload: vec![0_u8; 8].into(),
                }
                .into())
            },
        )
        .expect("four placed rows fit");
        assert_eq!(tokens, [97, 98, 7, 7, 7, 7, 122]);
        assert_eq!(spans[0].rows, 4);
        declared.modalities[0].kind = FrontendMediaKindV1::Video;
        declared.modalities[0].part_types = vec!["video_url".into()];
        declared.modalities[0].render.grid = Some(runtime_frontend::FrontendMediaGridRenderV1 {
            prefix: vec!["<g>".into()],
            suffix: vec!["</g>".into()],
            timestamp: "mm:ss".into(),
        });
        media[0].kind = FrontendMediaPartKindV1::Video;
        media[0].part_type = "video_url".into();
        let missing_grids = expand_media_chat_v1(
            &declared,
            &vocabulary,
            &segments,
            &media,
            |text| Ok(text.bytes().map(u32::from).collect()),
            |_, _| {
                Ok(MediaInputV1 {
                    key: [1; 32],
                    encoder: 0,
                    geometry: [2, 2, 4, 0],
                    output_rows: 4,
                    payload: vec![0_u8; 8].into(),
                }
                .into())
            },
        );
        assert!(
            missing_grids.is_err(),
            "video must not silently omit declared grid timestamps and markers"
        );
        let expand_video = |grids: Vec<(String, u32)>| {
            expand_media_chat_v1(
                &declared,
                &vocabulary,
                &segments,
                &media,
                |text| Ok(text.bytes().map(u32::from).collect()),
                |_, _| {
                    Ok(PreparedMediaInputV1 {
                        input: MediaInputV1 {
                            key: [1; 32],
                            encoder: 0,
                            geometry: [2, 2, 4, 0],
                            output_rows: 4,
                            payload: vec![0_u8; 8].into(),
                        },
                        grids: grids.clone(),
                    })
                },
            )
        };
        let (tokens, inputs, spans, _) =
            expand_video(vec![("00:00".into(), 2), ("00:09".into(), 2)])
                .expect("each video grid retains its timestamp and markers");
        let mut expected = "ab00:00<g>".bytes().map(u32::from).collect::<Vec<_>>();
        expected.extend([7, 7]);
        expected.extend("</g>00:09<g>".bytes().map(u32::from));
        expected.extend([7, 7]);
        expected.extend("</g>z".bytes().map(u32::from));
        assert_eq!(tokens, expected);
        assert_eq!(inputs.len(), 1);
        assert_eq!(
            spans,
            [
                MediaSpanV1 {
                    media: 0,
                    prompt_row: 10,
                    rows: 2,
                    media_row: 0
                },
                MediaSpanV1 {
                    media: 0,
                    prompt_row: 24,
                    rows: 2,
                    media_row: 2
                },
            ]
        );
        assert!(expand_video(vec![("00:00".into(), 3), ("00:09".into(), 2)]).is_err());
    }

    #[test]
    fn grid_position_rule_records_spans_whose_rows_take_grid_positions() {
        use runtime_frontend::{
            FrontendBoundaryTokenV1, FrontendMediaArenaV1, FrontendMediaElementV1,
            FrontendMediaEncoderV1, FrontendMediaKindV1, FrontendMediaModalityV1,
            FrontendMediaRecipeV1, FrontendMediaRenderV1,
        };
        let declared = FrontendMultimodalV1 {
            arena: FrontendMediaArenaV1 {
                rows: 64,
                page_rows: 64,
                hidden: 8,
                element: FrontendMediaElementV1::Bf16,
            },
            encoders: vec![FrontendMediaEncoderV1 {
                name: "image-encoder".to_owned(),
                payload_element: FrontendMediaElementV1::Bf16,
                payload_columns: 4,
                maximum_payload_bytes: 64,
                cost_per_row_milli: 1,
            }],
            modalities: vec![FrontendMediaModalityV1 {
                kind: FrontendMediaKindV1::Image,
                part_types: vec!["image_url".to_owned()],
                maximum_items: 2,
                maximum_bytes: 64,
                maximum_tokens: 16,
                recipe: FrontendMediaRecipeV1 {
                    tag: "image.patch-grid.v1".to_owned(),
                    parameters: serde_json::json!({ "merge": 2 }),
                },
                encoder: 0,
                render: FrontendMediaRenderV1 {
                    pad: "<pad>".to_owned(),
                    grid: None,
                },
                positions: Some(FrontendPositionRuleV1::Grid3AxisV1),
                layout: None,
            }],
        };
        let vocabulary = FrontendVocabularyV1 {
            size: 256,
            bos_token_id: 0,
            eos_token_id: 1,
            stop_token_ids: vec![1],
            boundary_tokens: vec![FrontendBoundaryTokenV1 {
                text: "<pad>".to_owned(),
                id: 7,
            }],
            prompt_special_tokens: vec!["<pad>".to_owned()],
        };
        let part = FrontendMediaPartV1 {
            kind: FrontendMediaPartKindV1::Image,
            part_type: "image_url".to_owned(),
            source: "data:image/png;base64,AAAA".to_owned(),
            format: None,
        };
        let media = vec![part.clone(), part];
        let segments = vec![
            PromptSegmentV1::Text("ab".to_owned()),
            PromptSegmentV1::Media(0),
            PromptSegmentV1::Text("z".to_owned()),
            PromptSegmentV1::Media(1),
            PromptSegmentV1::Text("q".to_owned()),
        ];
        // Patch grids of each image in order, and the rows each reports.
        let expand = |grids: [([u32; 3], u32); 2]| {
            let mut next = grids.into_iter();
            expand_media_chat_v1(
                &declared,
                &vocabulary,
                &segments,
                &media,
                |text| Ok(text.bytes().map(u32::from).collect()),
                move |_, _| {
                    let ([t, h, w], rows) = next.next().expect("two images");
                    Ok(MediaInputV1 {
                        key: [h as u8; 32],
                        encoder: 0,
                        geometry: [t, h, w, 0],
                        output_rows: rows,
                        payload: vec![0_u8; 8].into(),
                    }
                    .into())
                },
            )
        };
        let (tokens, inputs, spans, positions) =
            expand([([1, 4, 6], 6), ([1, 2, 2], 1)]).expect("both grids merge by two");
        assert_eq!(tokens, [97, 98, 7, 7, 7, 7, 7, 7, 122, 7, 113]);
        assert_eq!(
            positions,
            [
                PositionSpanV1 {
                    prompt_row: 2,
                    rows: 6,
                    grid: [1, 2, 3]
                },
                PositionSpanV1 {
                    prompt_row: 9,
                    rows: 1,
                    grid: [1, 1, 1]
                },
            ]
        );
        // The engine takes the spans as rendered and gives every row, and
        // the first generated row, its position.
        let engine_spans = spans
            .iter()
            .map(|span| EngineMediaSpanV1 {
                media: span.media,
                prompt_row: span.prompt_row,
                rows: span.rows,
                media_row: span.media_row,
            })
            .collect::<Vec<_>>();
        let engine_positions = positions
            .iter()
            .map(|span| EnginePositionSpanV1 {
                prompt_row: span.prompt_row,
                rows: span.rows,
                grid: span.grid,
            })
            .collect::<Vec<_>>();
        EngineRequest::new(
            RequestId::new(1),
            tokens.clone(),
            4,
            SamplingParameters::new(0.0).unwrap(),
            1,
            Vec::new(),
            ModelVocabulary::new(256).unwrap(),
        )
        .and_then(|request| {
            request.with_media(
                inputs
                    .iter()
                    .map(|input| EngineMediaInputV1 {
                        key: input.key,
                        encoder: input.encoder,
                        geometry: input.geometry,
                        output_rows: input.output_rows,
                        payload: EngineMediaPayloadV1::new(input.payload.clone()),
                    })
                    .collect(),
                engine_spans,
            )
        })
        .and_then(|request| request.with_position_axes(engine_positions.clone()))
        .expect("the engine accepts the rendered spans");
        let axes = runtime_engine::PositionAxesV1::new(11, 16, &engine_positions).unwrap();
        assert_eq!(
            axes.rows(0, 12),
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 3],
                [2, 2, 4],
                [2, 3, 2],
                [2, 3, 3],
                [2, 3, 4],
                [5, 5, 5],
                [6, 6, 6],
                [7, 7, 7],
                [8, 8, 8],
            ]
        );
        // A grid the merge does not divide, or rows that are not the merged
        // grid's product, are refused.
        assert!(expand([([1, 3, 6], 6), ([1, 2, 2], 1)]).is_err());
        assert!(expand([([1, 4, 6], 5), ([1, 2, 2], 1)]).is_err());
    }

    fn contribution(ordinal: u64, byte_start: usize, byte_end: usize) -> TokenContributionV1 {
        TokenContributionV1 {
            ordinal,
            token_id: ordinal as u32 + 10,
            byte_start,
            byte_end,
        }
    }

    #[test]
    fn ordinal_frontier_accounts_split_and_withheld_tokens_once() {
        let mut accounting = TokenAccounting::default();
        assert_eq!(account(&mut accounting, 1, &[contribution(0, 0, 1)]), Ok(1));
        assert_eq!(account(&mut accounting, 1, &[contribution(0, 1, 2)]), Ok(0));
        assert_eq!(
            account(
                &mut accounting,
                2,
                &[contribution(0, 2, 3), contribution(1, 0, 1)],
            ),
            Ok(1)
        );
        assert_eq!(unaccounted_tokens(&accounting, 2), Ok(0));
    }

    #[test]
    fn ordinal_frontier_includes_hidden_boundaries_and_leaves_terminal_tokens() {
        let mut accounting = TokenAccounting::default();
        assert_eq!(account(&mut accounting, 1, &[contribution(0, 0, 1)]), Ok(1));
        assert_eq!(account(&mut accounting, 3, &[contribution(2, 0, 1)]), Ok(2));
        assert_eq!(unaccounted_tokens(&accounting, 4), Ok(1));

        let no_visible_stop = TokenAccounting::default();
        assert_eq!(unaccounted_tokens(&no_visible_stop, 1), Ok(1));

        let mut partial_stop = TokenAccounting::default();
        assert_eq!(
            account(&mut partial_stop, 1, &[contribution(0, 0, 2)]),
            Ok(1)
        );
        assert_eq!(unaccounted_tokens(&partial_stop, 1), Ok(0));
    }

    #[test]
    fn ordinal_frontier_rejects_nonmonotonic_or_overlapping_contributions() {
        let mut accounting = TokenAccounting::default();
        assert_eq!(account(&mut accounting, 2, &[contribution(1, 0, 1)]), Ok(2));
        assert!(account(&mut accounting, 2, &[contribution(0, 0, 1)]).is_err());

        let mut overlapping = TokenAccounting::default();
        assert_eq!(
            account(&mut overlapping, 1, &[contribution(0, 0, 2)]),
            Ok(1)
        );
        assert!(account(&mut overlapping, 1, &[contribution(0, 1, 3)]).is_err());
    }

    /// Requests that ask for equal grammars at the same time get one
    /// compiled grammar, not one compilation each.
    #[test]
    fn equal_constraints_compiled_at_once_share_one_grammar() {
        let service = crate::grammar::GrammarServiceV1::new(
            Arc::new(super::stop_tests::tokenizer()),
            259,
            [],
        );
        let schema = serde_json::json!({"type": "object", "properties": {
            "name": {"type": "string", "maxLength": 24},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
            "count": {"type": "integer"}
        }, "required": ["name"]});
        let grammar =
            runtime_grammar::json_schema_grammar(&schema, runtime_grammar::JsonOptions::default())
                .unwrap();
        let barrier = std::sync::Barrier::new(8);
        let compiled = thread::scope(|scope| {
            let requests = (0..8)
                .map(|_| {
                    let grammar = Arc::new(grammar.clone());
                    let (service, barrier) = (&service, &barrier);
                    scope.spawn(move || {
                        barrier.wait();
                        service.compile(&grammar, &|| false).unwrap()
                    })
                })
                .collect::<Vec<_>>();
            requests
                .into_iter()
                .map(|request| request.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(
            compiled
                .iter()
                .all(|grammar| Arc::ptr_eq(grammar, &compiled[0]))
        );
        // The cache charges the grammar and its compiled form, and listens
        // for the compiled form growing.
        assert_eq!(
            service.kept_bytes(),
            grammar.clone().retained_bytes() + compiled[0].retained_bytes()
        );
        assert!(!compiled[0].observe_growth(Box::new(|| {})));
    }

    /// A declared boundary the tokenizer leaves ordinary is never answer
    /// text: a grammar that names no special token (a response format or
    /// structured output) does not admit it inside a JSON string, where its
    /// bytes fit. A grammar that names special tokens writes the model's own
    /// markup, as a tool choice does, and still admits it there, as do
    /// grammars served for a declaration that names no such boundary.
    #[test]
    fn answer_grammars_never_admit_a_declared_boundary() {
        let tokenizer = Arc::new(super::stop_tests::tokenizer_with(&["<t>"]));
        let (marker, quote) = (259, u32::from(b'"') + 3);
        let after_key_open = |service: &crate::grammar::GrammarServiceV1,
                              grammar: &runtime_grammar::Grammar| {
            let compiled = service
                .compile(&Arc::new(grammar.clone()), &|| false)
                .unwrap();
            let mut matcher = runtime_grammar::GrammarMatcher::new(compiled);
            matcher.accept_bytes(b"{\"").unwrap();
            let mut mask = vec![0; matcher.grammar().vocabulary().mask_words()];
            matcher.fill_token_bitmask(&mut mask).unwrap();
            let allows = |token: u32| mask[token as usize / 32] & (1 << (token % 32)) != 0;
            assert_eq!(allows(marker), matcher.accepts_token(marker).unwrap());
            (allows(marker), allows(quote))
        };
        let answer = runtime_grammar::json_object_grammar(runtime_grammar::JsonOptions::default());
        let markup = answer.clone().with_special_tokens([b"<call>".to_vec()]);
        let declared =
            crate::grammar::GrammarServiceV1::new(Arc::clone(&tokenizer), 260, [1, marker]);
        let undeclared = crate::grammar::GrammarServiceV1::new(Arc::clone(&tokenizer), 260, []);
        assert_eq!(after_key_open(&declared, &answer), (false, true));
        assert_eq!(after_key_open(&declared, &markup), (true, true));
        assert_eq!(after_key_open(&undeclared, &answer), (true, true));
    }

    /// A kept grammar that classifies on first use is charged for what that
    /// grew, also when the classification is refused for exhausting its
    /// budget after the scanner grew.
    #[test]
    fn a_refused_first_use_classification_is_charged_to_the_cache() {
        let brackets = "[".repeat(32);
        let service = crate::grammar::GrammarServiceV1::new(
            Arc::new(super::stop_tests::tokenizer_with(&[&brackets])),
            260,
            [],
        );
        // Too many positions to classify ahead, so classification happens
        // on first use, where the recursive arrays exhaust the budget.
        let schema = serde_json::json!({"$defs": {
            "a": {"type": "array", "items": {"$ref": "#/$defs/u"}},
            "b": {"type": "array", "items": {"$ref": "#/$defs/u"}, "minItems": 1},
            "u": {"anyOf": [{"$ref": "#/$defs/a"}, {"$ref": "#/$defs/b"}]}
        }, "anyOf": [{"$ref": "#/$defs/u"}, {"const": "x".repeat(9000)}]});
        let key = Arc::new(
            runtime_grammar::json_schema_grammar(&schema, runtime_grammar::JsonOptions::default())
                .unwrap(),
        );
        let compiled = service.compile(&key, &|| false).unwrap();
        assert_eq!(compiled.precompute(1).unwrap(), 0);
        let charged = || key.retained_bytes() + compiled.retained_bytes();
        let before = charged();
        assert_eq!(service.kept_bytes(), before);
        let refused = runtime_grammar::GrammarMatcher::new(Arc::clone(&compiled))
            .fill_token_bitmask(&mut vec![0; compiled.vocabulary().mask_words()]);
        assert!(refused.is_err(), "{refused:?}");
        assert!(charged() > before);
        assert_eq!(service.kept_bytes(), charged());
    }

    /// A client that stops reading leaves the relay waiting on its full
    /// channel; cancelling the request ends that wait, and what the client
    /// did receive arrived in order.
    #[test]
    fn cancellation_interrupts_a_send_blocked_on_a_full_client() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let (client, mut reader) = tokio::sync::mpsc::channel(1);
            let (worker, forward) = tokio::sync::mpsc::channel(8);
            let cancellation = CancellationSignal::new();
            let relay = tokio::spawn(forward_events(
                forward,
                EventSender::new(client),
                cancellation.clone(),
            ));
            for tokens in [vec![1], vec![2], vec![3]] {
                worker
                    .send(GenerationEvent::GeneratedTokenIds(tokens))
                    .await
                    .unwrap();
            }
            // The first event fills the client's channel and the second
            // waits for room.
            while worker.capacity() < 7 {
                tokio::task::yield_now().await;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
            assert!(!relay.is_finished());
            cancellation.cancel();
            tokio::time::timeout(Duration::from_secs(5), relay)
                .await
                .expect("cancellation ends the blocked send")
                .unwrap();
            let mut received = Vec::new();
            while let Ok(event) = reader.try_recv() {
                let GenerationEvent::GeneratedTokenIds(tokens) = event else {
                    panic!("unexpected event");
                };
                received.extend(tokens);
            }
            assert_eq!(received, [1]);
        });
    }

    /// Ordinary backpressure is kept: a slow reader still receives every
    /// event, in order.
    #[test]
    fn a_slow_reader_receives_every_event_in_order() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let (client, mut reader) = tokio::sync::mpsc::channel(1);
            let (worker, forward) = tokio::sync::mpsc::channel(64);
            let relay = tokio::spawn(forward_events(
                forward,
                EventSender::new(client),
                CancellationSignal::new(),
            ));
            for token in 0..32 {
                worker
                    .send(GenerationEvent::GeneratedTokenIds(vec![token]))
                    .await
                    .unwrap();
            }
            drop(worker);
            let mut received = Vec::new();
            while let Some(event) = reader.recv().await {
                tokio::time::sleep(Duration::from_millis(1)).await;
                let GenerationEvent::GeneratedTokenIds(tokens) = event else {
                    panic!("unexpected event");
                };
                received.extend(tokens);
            }
            relay.await.unwrap();
            assert_eq!(received, (0..32).collect::<Vec<_>>());
        });
    }
}

#[cfg(test)]
#[path = "generation/overlap_bench.rs"]
mod overlap_bench;
#[cfg(test)]
#[path = "generation/overlap_tests.rs"]
mod overlap_tests;
#[cfg(test)]
#[path = "generation/stop_tests.rs"]
pub(crate) mod stop_tests;

#[cfg(test)]
#[path = "generation/media_memory_tests.rs"]
mod media_memory_tests;

#[cfg(test)]
#[path = "generation/watchdog_tests.rs"]
mod watchdog_tests;
