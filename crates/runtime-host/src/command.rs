//! The command surface over one explicitly selected bundle.
//!
//! Four commands act on a bundle, and this is the only implementation of them.
//! `bundle verify` runs the canonical verifier over a directory, `bundle
//! install` copies a verified directory into an explicit local install root
//! without replacing anything, `serve` loads one selected bundle at process
//! start and serves it over the OpenAI transport, and `conformance-run` drives
//! one bounded batch through that same loaded bundle and exits. Every front end
//! parses through here, so none of them carries a second parser, a second
//! verifier, a second installer, or a second load.
//!
//! Nothing here names a model. A selection is a directory, or an install root
//! and an identifier, and everything a run needs afterwards comes out of the
//! package that selection verified: the topology, the page layout, the served
//! identities, the tokenizer, and the context policy. Local checkpoint selection
//! discovers bounded installed preparation helpers, then selects one canonical
//! package. There is no hot reload or checkpoint-code execution.

use crate::preparation::{LocalModel, ServeSelection};
use std::any::Any;
use std::ffi::OsString;
use std::fmt;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::rc::Rc;
use std::sync::Arc;

use runtime_bundle::{
    BundleError, BundleHostCompatibilityV1, BundleKvDtypeV1, BundleManifestV1, BundleOpaqueId,
    BundlePackageV1, BundleTopologyV1, MODEL_BUNDLE_V1_ABI_VERSION,
};
use runtime_bundle_abi::{
    BUNDLE_LOGITS_ELEMENT_V1_BF16, BundleAbiError, BundleStatusV1, HostServiceResultV1,
    LoadedBundleInstanceV1, LogitsRowSinkV1, OwnedLogitsRowV1, OwnedLogitsViewV1,
    ResultSinkLimitsV1,
};
use runtime_engine::{
    DraftCostModel, DraftSource, Engine, EngineConfig, EngineMetrics, ExecutorError,
    ModelBatchItem, ModelExecutor, ModelTransaction, PromptLookupConfig, RequestId,
    SpeculationConfig, SpeculationMode,
};
use runtime_kv::{KvCachePrecision, KvGeometry, KvMetrics, KvPool, PreparedAppendBatch};
use runtime_model_api::ModelVocabulary;
use runtime_scheduler::{
    AdmissionPolicy, BatchPhase, OptimisticAdmission, SchedulerCapacityProfile, SchedulerConfig,
};
use runtime_server::{
    GenerationBackend, MetricsSnapshotProvider, RequestSurfaceV1, ServerConfig, ServerFrontendV1,
    SystemMonotonicClock,
};
use tokio::net::TcpListener;

use crate::deployment::deployment_from_package_v1;
use crate::device::HostDeviceBackendV1;
use crate::executor::{
    BundleExecutionV1, BundleModelExecutorV1, BundleTransactionStateV1, batch_wire_capability_v1,
    encode_batch_request_with_capability_v1,
};
use crate::generation::{
    DEFAULT_TOKENIZER_INPUT_BYTES, EngineBackendConfig, EngineGenerationBackend,
};
use crate::install::{
    BundleSelectionV1, HostLoadErrorV1, InstalledBundleHostV1, load_verified_process_bundle_v1,
    verify_selected_bundle_v1,
};
use crate::kernel;
use crate::kv_cache_dtype::{KvCacheDtypeV1, resolve_kv_cache_dtype_v1};
use crate::reference_forcing::{
    REFERENCE_TOKENS_ENV_V1, ReferenceTokensV1, refuse_reference_tokens_variable_v1,
};
use crate::services::{HostServicesV1Backend, ServingKvPolicyV1};

/// The device architecture a command assumes when the caller names none.
///
/// This repository targets one architecture, so assuming it is not a guess
/// about the bundle: a bundle that does not declare it is refused rather than
/// adapted, and `--architecture` names another one explicitly.
pub const DEFAULT_HOST_ARCHITECTURE_V1: &str = "sm_120f";

/// The most logits one fixture row may carry.
///
/// This is the host's own bound on what it will collect, not a property of any
/// model. A bundle that emits more is refused before the row is kept.
pub const MAX_FIXTURE_LOGITS_PER_ROW_V1: u64 = 1 << 20;

/// The request identifier one fixture batch schedules.
const FIXTURE_REQUEST_ID_V1: u64 = 1;

/// Where a served deployment listens when the caller names no address.
///
/// Loopback is the safe default: a bundle is served to this machine unless an
/// operator says otherwise with `--listen`.
pub const DEFAULT_LISTEN_ADDRESS_V1: &str = "127.0.0.1:8310";

/// How many sequences a served deployment admits at once by default.
pub const DEFAULT_SERVED_SEQUENCES_V1: usize = 8;

/// The largest token budget one scheduled batch takes by default.
///
/// A deployment that declares a shorter sequence limit gets that instead, and
/// `--max-batch-tokens` names another.
pub const DEFAULT_SERVED_BATCH_TOKENS_V1: usize = 8192;
pub const DEFAULT_PREFILL_CHUNK_TOKENS_V1: usize = 2048;
/// Internal staging depth; excess requests wait at the transport.
pub const PREPARED_QUEUE_DEPTH_V1: usize = 1024;
/// Queued requests wait without a limit unless `--queue-wait-ms` sets one.
pub const DEFAULT_QUEUE_WAIT_MS_V1: u64 = 0;

/// The most characters a fixture request label may carry.
const MAX_FIXTURE_LABEL_BYTES_V1: usize = 64;

/// What the receipt calls the host-memory device services.
///
/// Both names are public and unconditional, because a caller that needs to
/// refuse one of them needs to be able to say which one it got on a build that
/// cannot reach the other. A build with no driver still has to be able to name
/// the driver it does not have.
pub const HOST_MEMORY_SERVICES_V1: &str = "host-memory";

/// What the receipt calls the driver device services.
pub const DRIVER_SERVICES_V1: &str = "cuda-driver";

/// The command words, without the program that carries them.
///
/// Both front ends print this after their own `usage:` line, because both run
/// the same commands.
pub const HOST_COMMAND_USAGE_V1: &str = "\
profiles list FILE
profiles show FILE NAME
bundle verify --bundle DIR
bundle check-frontend --bundle DIR [--topology ID] [--render FILE]... [--tokenizer FILE]...
  [--streaming FILE]... (bundle check-frontend --help describes the vector files)
kernel build --manifest FILE [--cache-dir DIR]
kernel inspect --manifest FILE [--cache-dir DIR]
serve --model DIR [--tensor-parallel-size N] [--model-cache-dir DIR] [--activation-policy auto|checkpoint|bf16]
  [--modalities auto|all|text[,image][,video][,audio]] [--load-threads N]
verify --model DIR [--manifest FILE[#ENTRY]] [--allow-unrecorded] [--jobs N]
eval list|run|verify-references --model DIR ... (eval --help lists the options)
serve --bundle DIR [--checkpoint DIR [--load-threads N]] [--profile NAME|FILE#NAME]
  [--listen ADDR] [--topology ID] [--devices 0,1,...]
  [--max-context-tokens N] [--max-sequences N] [--max-batch-tokens N]
  [--prefill-chunk-tokens N] [--queue-depth N] [--queue-wait-ms N] [--drain-timeout-ms N]
  [--kv-memory-fraction F] [--kv-pages N] [--kv-cache-dtype auto|bf16|fp8|mixed]
  [--context-parallel-size N]
  [--model-option NAME=VALUE]... [--list-model-options]
  [--speculative-tokens N] [--speculative-rows N] [--speculation fixed|adaptive]
  [--draft-cost per-token|per-block] [--prefix-cache on|off] [--verify-warmup on|off]
  [--prefill-budget fixed|decode-aware|tbt] [--prefill-itl-target-ms N | --prefill-decode-steps N]
  [--decode-tbt-target-ms N] [--mixed-min-prefill-tokens N] [--mixed-decode-steps N]
  [--kv-offload-gib N] [--kv-offload-min-tokens N] [--kv-offload-gbps F] [--kv-offload-prefix off]
  [--admission reserved|optimistic] [--admission-watermark F] [--max-preemptions N]
  [--debug-preempt-every N] [--debug-swap-every N]
  [--max-request-body-bytes N] [--max-request-string-bytes N] [--max-images-per-request N]
  [--served-model-name NAME [NAME ...]]
  [--responses-store-bytes N]
  [--max-tokenizer-input-bytes N]
  [--allowed-local-media-path DIR] [--allowed-media-domains HOST[,HOST...]]
  [--media-allow-private-addresses on|off] [--media-connect-timeout-ms N]
  [--media-fetch-timeout-ms N] [--media-request-timeout-ms N] [--media-max-redirects N]
  [--allowed-origins JSON] [--allowed-methods JSON] [--allowed-headers JSON]
  [--allow-credentials] [--api-key KEY] [--enable-request-id-headers]
  [--fake-host-services]
conformance-run --bundle DIR [--checkpoint DIR] --fixture-request LABEL
  [--topology ID] [--fixture-tokens COUNT] [--scenario commit|cancel|drop]
  [--kv-memory-fraction F]
  [--fake-host-services]
every command except verify accepts --architecture ID and --capability ID (repeatable)";

/// What `bundle check-frontend --help` prints: the command and the schema of
/// each vector file it reads.
pub const CHECK_FRONTEND_USAGE_V1: &str = r#"bundle check-frontend --bundle DIR [--topology ID] [--render FILE]... [--tokenizer FILE]...
  [--streaming FILE]...

Checks a built package's chat frontend against the model's vectors, through
the code serve runs: the package is verified, its native module opened and its
frontend export served, and its tokenizer built from the package's own
components. It needs no GPU and no checkpoint. A package with more than one
topology needs --topology. Each case prints one line, "pass" or "FAIL" with
the first difference; any failure, or files that hold no case, exits
nonzero.

Each FILE is a JSON array of cases.

--render: {"id", "messages", "tools", "chat_template_kwargs", "rendered_text",
  "token_ids"}, the chat-goldens.json shape, with optional "tool_choice",
  "reasoning_effort", "response_format", "add_generation_prompt" (default
  true) and "continue_final_message" (default false). Other fields are
  ignored. The case becomes an OpenAI chat request that the transport parses
  and the frontend renders, as /tokenize renders one. Messages are read as
  the model golden tests read them: tool-call arguments given as JSON values
  are sent as their compact JSON text, a tool result without "tool_call_id"
  and a call without "id" or "type" get an empty id and type "function", an
  assistant turn's "reasoning_content" is sent under the served reasoning
  field when the frontend reads "reasoning_content" from history at all, and
  a file:// media URL is sent as an inline one, since a render places media
  without reading them. The rendered text must equal "rendered_text", or its
  tokenizer round trip must (a normalizing tokenizer captures normalized
  text), and its ids
  must equal "token_ids". A prompt with media is tokenized before each media
  placeholder is expanded.
  A case may hold "refusal": {"message", "parameter"}, both optional strings,
  in place of "rendered_text" and "token_ids": it passes only when the
  transport or the frontend refuses the request, with the stated message and
  parameter; {} accepts any refusal. A case with "raw": true is sent exactly
  as written, with none of the changes above (a file:// media URL is refused
  as a client's would be), and may hold no field besides "id", "raw", its
  expectation, "messages" and the request fields.

--tokenizer: {"text", "token_ids"} with an optional "decoded", the
  tokenizer-golden.json shape. The text is encoded as /tokenize encodes a
  prompt, and the ids decoded as /detokenize decodes them.

--streaming: {"id", "request", "token_ids", "expect"} with optional "steps"
  and "separator".
  "request" holds "messages" and any of the optional render request fields.
  "token_ids" is the generated completion; a stop token ends it, and without
  one it ends at the token limit. "expect" is {"reasoning", "content",
  "tool_calls": [{"name", "arguments"}], "finish_reason"}, where reasoning
  and content default to "", tool_calls to [], arguments is the text the
  format writes, and finish_reason is "stop", "length" or "tool_calls" as a
  chat response reports it. "steps" is [{"token_ids", "reasoning", "content",
  "tool_call_deltas": [{"index", "name", "arguments"}]}]: the steps' token ids
  are "token_ids" in order, and each step holds what the response publishes
  once its tokens are delivered (the last step also holds what the end
  delivers). Text fields default to "" and deltas to []; a step's deltas are
  merged per call, with "name" only where the step opens the call. The ids
  decode through the decoder admission builds and every event is checked as a
  served response checks it, once with tool calls delivered whole and once
  streamed. Each delivery must match "expect" and each step's reasoning and
  content; the streamed one must also match each step's deltas. "separator"
  is a string, the template's history text after a round trip's stop token;
  it is never decoded."#;

/// Which device services one run uses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceServicesV1 {
    /// The strongest services this build has: the driver where one is linked,
    /// and the host-memory services where none is.
    Compiled,
    /// The host-memory services, demanded by name.
    HostMemory,
}

impl DeviceServicesV1 {
    /// What a receipt calls the services this choice actually reaches.
    ///
    /// A build that links no driver has only host memory, and saying so is what
    /// keeps a run against host memory from reading as a run against a device.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::HostMemory => HOST_MEMORY_SERVICES_V1,
            #[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
            Self::Compiled => DRIVER_SERVICES_V1,
            #[cfg(not(all(target_os = "linux", feature = "linux-cuda-host")))]
            Self::Compiled => HOST_MEMORY_SERVICES_V1,
        }
    }
}

/// What a build without host-memory services says when one is asked for.
#[cfg(not(feature = "fake-host-services"))]
const NO_HOST_MEMORY_SERVICES_V1: &str = "this host was built without host-memory device services, so --fake-host-services has nothing \
     to select";

/// One bounded conformance run of the selected bundle.
///
/// This is not the serving command. It drives one batch through a loaded
/// bundle so a conformance host can prove the boundary without a transport,
/// and it exits when that batch has been finalized and the instance released.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixtureRequestV1 {
    /// What the receipt calls this run. It selects no behaviour.
    pub label: String,
    /// How many tokens the one prefilled sequence carries, where the caller
    /// named a count instead of taking the bundle's declared page.
    pub prefill_tokens: Option<usize>,
    /// What this run does with the batch it drives.
    pub scenario: FixtureScenarioV1,
}

/// What one conformance run does with the batch it drove.
///
/// The scenario is named on the command line and selects nothing else: the same
/// selection, the same plan, and the same encoded batch drive all of them, so
/// what differs is only how the execution ends. Each one ends the same way
/// regardless, with the instance released and this host owning nothing on the
/// bundle's behalf.
///
/// The list is closed and bounded, and every scenario lasts one process: there
/// is no scenario that leaves an instance alive for a later invocation to find.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FixtureScenarioV1 {
    /// Execute one batch, commit it, and destroy the instance.
    #[default]
    Commit,
    /// Execute one batch, cancel it, and replay the same batch.
    ///
    /// The replay is what makes the cancel observable without knowing what any
    /// of the numbers mean: a bundle that rolled the cancelled execution back
    /// produces the row it produced before, and one that kept part of it
    /// produces something else.
    Cancel,
    /// Execute one batch and drop the instance with it still pending.
    ///
    /// Nothing here finalizes that execution. The ABI wrapper owes the cancel
    /// and the destroy, and the resource count afterwards says whether it paid.
    Drop,
}

impl FixtureScenarioV1 {
    /// What `--scenario` names, and what the receipt reports.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Commit => "commit",
            Self::Cancel => "cancel",
            Self::Drop => "drop",
        }
    }

    /// What the receipt calls a run that ended this way.
    #[must_use]
    pub const fn status(self) -> &'static str {
        match self {
            Self::Commit => "committed",
            Self::Cancel => "cancelled",
            Self::Drop => "dropped",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "commit" => Some(Self::Commit),
            "cancel" => Some(Self::Cancel),
            "drop" => Some(Self::Drop),
            _ => None,
        }
    }
}

/// What one driven scenario did, before the receipt names the bundle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixtureScenarioReportV1 {
    /// How many times this run executed the bundle's batch.
    pub executions: usize,
    /// How many logits rows those executions emitted in total.
    pub rows: usize,
    /// The width of each of them.
    pub logits_per_row: u64,
    /// What this host still owned on the bundle's behalf when the run ended.
    pub live_resources: usize,
}

/// What a served deployment is allowed to admit at once.
///
/// Every one of these is a deployment choice rather than a model property: the
/// bundle declares its page layout and sequence limit, and the frontend
/// declares its context, so these only narrow what those already allow.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServingOptionsV1 {
    pub maximum_context_tokens: Option<usize>,
    pub maximum_sequences: usize,
    pub maximum_batch_tokens: Option<usize>,
    pub prefill_chunk_tokens: Option<usize>,
    pub queue_depth: Option<usize>,
    /// How long a request may wait for admission; zero waits without limit.
    pub queue_wait_ms: u64,
    pub drain_timeout_ms: u64,
    pub maximum_request_body_bytes: Option<usize>,
    pub maximum_request_string_bytes: Option<usize>,
    /// Enables at most 128 stored Responses within this serialized byte budget.
    pub responses_store_bytes: usize,
    pub maximum_tokenizer_input_bytes: Option<usize>,
    /// The most image parts one request may carry; the package's declared
    /// count by default. Images stream through the media arena, so this may
    /// exceed the declaration.
    pub maximum_images_per_request: Option<usize>,
    pub kv_memory_fraction_millionths: u32,
    pub kv_pages: Option<u64>,
    /// Ranks of one context-parallel group, which split every sequence's KV
    /// pages; 1, the default, keeps every page on every rank.
    pub context_parallel_size: usize,
    /// Which declared serving KV layout to run; `auto`, the package's
    /// preference, by default.
    pub kv_cache_dtype: KvCacheDtypeV1,
    /// Enables prefix caching for a bundle that declares it; on by default.
    pub prefix_cache: bool,
    pub debug_swap: bool,
    pub kv_offload: crate::offload::KvOffloadOptionsV1,
    /// With speculation, startup warmup also reaches every verify graph
    /// shape; on by default.
    pub verify_warmup: bool,
    /// How the scheduler reserves pages; optimistic by default.
    pub admission: AdmissionPolicy,
    pub prefill_budget: runtime_scheduler::PrefillBudget,
}

impl Default for ServingOptionsV1 {
    fn default() -> Self {
        Self {
            maximum_context_tokens: None,
            maximum_sequences: DEFAULT_SERVED_SEQUENCES_V1,
            maximum_batch_tokens: None,
            prefill_chunk_tokens: None,
            queue_depth: None,
            queue_wait_ms: DEFAULT_QUEUE_WAIT_MS_V1,
            drain_timeout_ms: 30_000,
            maximum_request_body_bytes: None,
            maximum_request_string_bytes: None,
            responses_store_bytes: 0,
            maximum_tokenizer_input_bytes: None,
            maximum_images_per_request: None,
            kv_memory_fraction_millionths: 900_000,
            kv_pages: None,
            context_parallel_size: 1,
            kv_cache_dtype: KvCacheDtypeV1::Auto,
            prefix_cache: true,
            debug_swap: false,
            kv_offload: crate::offload::KvOffloadOptionsV1::default(),
            verify_warmup: true,
            admission: AdmissionPolicy::Optimistic(OptimisticAdmission::DEFAULT),
            prefill_budget: runtime_scheduler::PrefillBudget::Fixed,
        }
    }
}

/// The exact or bounded request shape a developer deployment admits, counted in tokens.
///
/// `new` requires an exact prompt count and bounds generated tokens.
/// `bounded` accepts positive counts up to separate prompt and generation
/// maxima, subject to a shared total capacity.
///
/// This is a request, not a permission. What a bundle actually lets it serve is
/// [`resolve_developer_serving_plan_v1`], which compares it against the rows,
/// the page size, and the context the selected bundle declares. Nothing here
/// names a model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeveloperServingEnvelopeV1 {
    bounded_total_tokens: Option<usize>,
    prompt_tokens: usize,
    maximum_generated_tokens: usize,
}

impl DeveloperServingEnvelopeV1 {
    /// One envelope over a prompt length and a generation bound.
    pub fn new(
        prompt_tokens: usize,
        maximum_generated_tokens: usize,
    ) -> Result<Self, HostCommandErrorV1> {
        if prompt_tokens == 0 {
            return Err(usage_error(
                "a developer serving envelope must admit a prompt of at least one token",
            ));
        }
        if maximum_generated_tokens == 0 {
            return Err(usage_error(
                "a developer serving envelope must generate at least one token",
            ));
        }
        if prompt_tokens
            .checked_add(maximum_generated_tokens)
            .is_none()
        {
            return Err(usage_error(format!(
                "a prompt of {prompt_tokens} token(s) and {maximum_generated_tokens} generated \
                 token(s) overflow a token budget on this platform"
            )));
        }
        Ok(Self {
            bounded_total_tokens: None,
            prompt_tokens,
            maximum_generated_tokens,
        })
    }

    /// Admit positive prompt and generation counts within generic limits.
    /// The total capacity must accommodate each maximum with a one-token partner.
    pub fn bounded(
        maximum_prompt_tokens: usize,
        maximum_generated_tokens: usize,
        maximum_total_tokens: usize,
    ) -> Result<Self, HostCommandErrorV1> {
        if maximum_prompt_tokens == 0
            || maximum_generated_tokens == 0
            || maximum_total_tokens < 2
            || maximum_prompt_tokens >= maximum_total_tokens
            || maximum_generated_tokens >= maximum_total_tokens
        {
            return Err(usage_error(
                "bounded serving limits must be positive and fit the total capacity",
            ));
        }
        Ok(Self {
            bounded_total_tokens: Some(maximum_total_tokens),
            prompt_tokens: maximum_prompt_tokens,
            maximum_generated_tokens,
        })
    }

    /// Whether the prompt count is a maximum rather than an exact length.
    #[must_use]
    pub const fn is_bounded(self) -> bool {
        self.bounded_total_tokens.is_some()
    }

    /// Whether tokenized request counts fit this envelope.
    #[must_use]
    pub const fn admits(self, prompt_tokens: usize, generated_tokens: usize) -> bool {
        let prompt_matches = if self.is_bounded() {
            prompt_tokens > 0 && prompt_tokens <= self.prompt_tokens
        } else {
            prompt_tokens == self.prompt_tokens
        };
        prompt_matches
            && generated_tokens > 0
            && generated_tokens <= self.maximum_generated_tokens
            && match prompt_tokens.checked_add(generated_tokens) {
                Some(total) => total <= self.total_tokens(),
                None => false,
            }
    }

    /// The exact prompt count, or the maximum prompt count for a bounded form.
    #[must_use]
    pub const fn prompt_tokens(self) -> usize {
        self.prompt_tokens
    }

    /// The most tokens one admitted request may be answered with.
    #[must_use]
    pub const fn maximum_generated_tokens(self) -> usize {
        self.maximum_generated_tokens
    }

    /// The maximum total request budget, in prompt plus generated tokens.
    ///
    /// This is the public number a transport validates a request against. It is
    /// larger than the model state one request consumes, and
    /// [`DeveloperServingPlanV1::consumed_rows`] is where that difference is
    /// accounted for.
    #[must_use]
    pub const fn total_tokens(self) -> usize {
        match self.bounded_total_tokens {
            Some(total) => total,
            None => self.prompt_tokens + self.maximum_generated_tokens,
        }
    }
}

/// What one bundle's own declaration lets one envelope serve.
///
/// Two counts are kept apart here, in two different units, because conflating
/// them is what would over-allocate or over-promise:
///
/// - `total_tokens` counts prompt tokens plus generated tokens. It is what a
///   request declares and what the transport charges it.
/// - `consumed_rows` counts rows of per-sequence model state, which is what the
///   bundle's KV layout declares a maximum for.
///
/// The two differ by exactly one, and only in the safe direction. A prefill of
/// `P` prompt tokens writes `P` rows and reads the first generated token out of
/// the last of them. Every further generated token is written back as one row
/// before the step that reads it, so answering with `G` tokens writes `G - 1`
/// rows: the last token is emitted from a row that already exists and is never
/// appended. A request is therefore `P + G` tokens over `P + G - 1` rows, and a
/// deployment may serve one token more than the bundle declares rows without
/// ever writing a row the bundle did not declare.
///
/// That is the whole of the allowance. It is one token, it is bounded by the
/// declared row count rather than chosen, and it buys no extra storage: a plan
/// that would need a page the declared rows do not already need is refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeveloperServingPlanV1 {
    envelope: DeveloperServingEnvelopeV1,
    total_tokens: usize,
    consumed_rows: usize,
    declared_rows: usize,
    page_tokens: usize,
    pages_per_sequence: usize,
}

impl DeveloperServingPlanV1 {
    /// The shape this plan admits.
    #[must_use]
    pub const fn envelope(self) -> DeveloperServingEnvelopeV1 {
        self.envelope
    }

    /// Prompt plus generated tokens, which is what a request is charged.
    #[must_use]
    pub const fn total_tokens(self) -> usize {
        self.total_tokens
    }

    /// Upper bound on rows of model state an admitted request writes.
    #[must_use]
    pub const fn consumed_rows(self) -> usize {
        self.consumed_rows
    }

    /// Rows of model state the bundle declares one sequence may hold.
    #[must_use]
    pub const fn declared_rows(self) -> usize {
        self.declared_rows
    }

    /// Rows of model state one declared KV page holds.
    #[must_use]
    pub const fn page_tokens(self) -> usize {
        self.page_tokens
    }

    /// Pages one sequence reserves, which the declared rows already needed.
    #[must_use]
    pub const fn pages_per_sequence(self) -> usize {
        self.pages_per_sequence
    }
}

/// Resolves one developer envelope against one bundle's own declaration.
///
/// Three facts refuse an envelope, and every one of them is the bundle's rather
/// than this host's: the rows its KV layout declares per sequence, the rows one
/// of its declared pages holds, and the total tokens its frontend declaration
/// offers. An envelope that would write a row the bundle did not declare, that
/// would reserve a page those declared rows do not already need, or that is
/// longer than the served declaration allows is refused here, before anything
/// is loaded.
pub fn resolve_developer_serving_plan_v1(
    manifest: &BundleManifestV1,
    declared_total_tokens: usize,
    envelope: DeveloperServingEnvelopeV1,
) -> Result<DeveloperServingPlanV1, HostCommandErrorV1> {
    let kv_layout = manifest.kv_layout();
    let page_tokens = usize::try_from(kv_layout.page_tokens())
        .map_err(|_| run_error("the declared page token count does not fit this platform"))?;
    let declared_rows = usize::try_from(kv_layout.maximum_sequence_tokens())
        .map_err(|_| run_error("the declared sequence limit does not fit this platform"))?;
    let total_tokens = envelope.total_tokens();
    // One generated token is emitted from a row that already exists, so the
    // budget may exceed the declared rows by exactly that one and no more.
    let consumed_rows = total_tokens - 1;
    if consumed_rows > declared_rows {
        return Err(usage_error(format!(
            "a prompt of {} token(s) answered with {} token(s) writes {consumed_rows} row(s) of \
             model state, and the selected bundle declares {declared_rows}",
            envelope.prompt_tokens(),
            envelope.maximum_generated_tokens(),
        )));
    }
    if total_tokens > declared_total_tokens {
        return Err(usage_error(format!(
            "a prompt of {} token(s) answered with {} token(s) charges {total_tokens} token(s), \
             and the served declaration offers {declared_total_tokens}",
            envelope.prompt_tokens(),
            envelope.maximum_generated_tokens(),
        )));
    }
    let declared_pages = declared_rows.div_ceil(page_tokens);
    let required_pages = total_tokens.div_ceil(page_tokens);
    if required_pages > declared_pages
        || (!envelope.is_bounded() && required_pages != declared_pages)
    {
        return Err(usage_error(format!(
            "{total_tokens} token(s) over pages of {page_tokens} row(s) reserve \
             {required_pages} page(s) per sequence, and the declared {declared_rows} row(s) \
             account for {declared_pages}"
        )));
    }
    let pages_per_sequence = if envelope.is_bounded() {
        declared_pages
    } else {
        required_pages
    };
    Ok(DeveloperServingPlanV1 {
        envelope,
        total_tokens,
        consumed_rows,
        declared_rows,
        page_tokens,
        pages_per_sequence,
    })
}

/// The engine sizing one resolved plan earns over one bundle.
///
/// This is the same sizing the serving command builds its engine with, so a
/// caller that runs that engine over its own executor runs the one a served
/// deployment would rather than a second one shaped like it.
///
/// A plan is a value, and a value carries no evidence about where it came from.
/// This one is resolved again, here, against the manifest it is about to size
/// and against the context that manifest's deployment actually serves. A plan
/// that a different bundle's rows or pages produced does not survive that, so
/// it cannot size an engine for this one.
pub fn developer_serving_engine_config_v1(
    manifest: &BundleManifestV1,
    vocabulary: ModelVocabulary,
    served_context_tokens: usize,
    serving: ServingOptionsV1,
    plan: DeveloperServingPlanV1,
) -> Result<EngineConfig, HostCommandErrorV1> {
    engine_config_v1(
        manifest,
        vocabulary,
        served_context_tokens,
        serving,
        Some(plan),
    )
}

/// Resolves one plan again against the bundle and context it will size.
///
/// The envelope is the only part of a plan that is a caller's statement; every
/// other number in it was derived from a bundle's declaration. Deriving them
/// again from the manifest in hand and requiring the same answer is what binds
/// the plan to this bundle: a plan resolved against 1024 declared rows names
/// 1024, and a bundle that declares 33 produces a different plan or no plan at
/// all.
fn revalidated_plan_v1(
    manifest: &BundleManifestV1,
    served_context_tokens: usize,
    plan: DeveloperServingPlanV1,
) -> Result<DeveloperServingPlanV1, HostCommandErrorV1> {
    let resolved =
        resolve_developer_serving_plan_v1(manifest, served_context_tokens, plan.envelope())?;
    if resolved != plan {
        return Err(usage_error(format!(
            "this serving plan charges {} token(s) over {} row(s) of a bundle declaring {} row(s) \
             in pages of {}, and the selected bundle resolves the same envelope to {} token(s) \
             over {} row(s) of {} row(s) in pages of {}",
            plan.total_tokens(),
            plan.consumed_rows(),
            plan.declared_rows(),
            plan.page_tokens(),
            resolved.total_tokens(),
            resolved.consumed_rows(),
            resolved.declared_rows(),
            resolved.page_tokens(),
        )));
    }
    Ok(resolved)
}

/// The identities of a direct package and the checkpoint metadata it was
/// selected for, as preparation returned and the host checked them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectPackageIdentityV1 {
    pub package_identity: String,
    pub metadata_identity: String,
}

/// The vector files one `bundle check-frontend` reads, each kind in the
/// order the command line named them.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FrontendVectorFilesV1 {
    pub render: Vec<PathBuf>,
    pub tokenizer: Vec<PathBuf>,
    pub streaming: Vec<PathBuf>,
}

/// One parsed command over one selected bundle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HostCommandV1 {
    Display {
        text: String,
    },
    Verify {
        selection: BundleSelectionV1,
        host: BundleHostCompatibilityV1,
    },
    /// `bundle check-frontend`: the selected package's frontend against the
    /// model's vector files.
    CheckFrontend {
        selection: BundleSelectionV1,
        host: BundleHostCompatibilityV1,
        topology: Option<String>,
        vectors: FrontendVectorFilesV1,
    },
    KernelBuild {
        manifest: PathBuf,
        cache_dir: Option<PathBuf>,
    },
    KernelInspect {
        manifest: PathBuf,
        cache_dir: Option<PathBuf>,
    },
    /// `verify --model DIR`: the checkpoint's files against their recorded
    /// SHA-256 values.
    CheckpointVerify {
        options: crate::checkpoint_verify::VerifyOptionsV1,
    },
    Serve {
        model_options: std::collections::BTreeMap<String, String>,
        list_model_options: bool,
        /// The serving profile whose options are defaults, reported by
        /// `/health`.
        profile_name: Option<String>,
        profile_drafter: Option<String>,
        qualified_max_sequences: Option<usize>,
        /// The direct package `serve --model` prepared, reported by `/health`.
        direct_package: Option<Box<DirectPackageIdentityV1>>,
        http_options: Box<runtime_server::HttpOptions>,
        devices: Option<Box<crate::DeviceSelectionV1>>,
        selection: ServeSelection,
        host: BundleHostCompatibilityV1,
        topology: Option<String>,
        device: DeviceServicesV1,
        listen: SocketAddr,
        serving: Box<ServingOptionsV1>,
        speculative_tokens: usize,
        /// Most decode-phase rows of one verify step, below the bundle's
        /// declared maximum; the declaration when unset.
        speculative_rows: Option<u16>,
        speculation_mode: SpeculationMode,
        draft_cost_model: DraftCostModel,
        allowed_local_media_path: Option<PathBuf>,
        remote_media: Box<runtime_media::remote::RemoteMediaPolicy>,
        /// Qualification-only teacher forcing; see `reference_forcing`.
        debug_reference_tokens: Option<PathBuf>,
        /// `--checkpoint`: the checkpoint the bundle's weight plans read, in
        /// place of the package's compiled rank images.
        direct_checkpoint: Option<PathBuf>,
        /// `--load-threads`: packing workers per rank of a direct load.
        load_threads: Option<std::num::NonZeroU16>,
    },
    ConformanceRun {
        selection: BundleSelectionV1,
        host: BundleHostCompatibilityV1,
        topology: Option<String>,
        device: DeviceServicesV1,
        request: FixtureRequestV1,
        kv_memory_fraction_millionths: u32,
        /// `--checkpoint`: the checkpoint the bundle's weight plans load.
        direct_checkpoint: Option<PathBuf>,
    },
}

/// Why one command was refused.
#[derive(Debug)]
pub enum HostCommandErrorV1 {
    /// The invocation or its serving profile is invalid.
    Usage(String),
    /// The canonical parser, verifier, or installer refused the bundle.
    Bundle(BundleError),
    /// Process-start loading refused the selected bundle.
    Load(HostLoadErrorV1),
    /// The run reached the bundle and could not finish.
    Run(String),
}

impl fmt::Display for HostCommandErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(message) | Self::Run(message) => formatter.write_str(message),
            Self::Bundle(error) => error.fmt(formatter),
            Self::Load(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for HostCommandErrorV1 {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Usage(_) | Self::Run(_) => None,
            Self::Bundle(error) => Some(error),
            Self::Load(error) => Some(error),
        }
    }
}

impl From<BundleError> for HostCommandErrorV1 {
    fn from(error: BundleError) -> Self {
        Self::Bundle(error)
    }
}

impl From<HostLoadErrorV1> for HostCommandErrorV1 {
    fn from(error: HostLoadErrorV1) -> Self {
        Self::Load(error)
    }
}

/// What one fixture request did, in the order the host did it.
///
/// The scenario, the status it reached, and the resources left behind are all
/// here, because a harness that only sees an exit status cannot tell a run that
/// cancelled from one that quietly committed instead.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixtureRunReceiptV1 {
    pub bundle_id: String,
    pub topology_id: String,
    pub rank_count: u32,
    pub device_services: &'static str,
    /// The architecture this host answered the bundle's target with.
    pub architecture: String,
    /// The bundle ABI version this host implements.
    pub abi_version: u32,
    pub label: String,
    pub scenario: FixtureScenarioV1,
    pub prefill_tokens: usize,
    pub executions: usize,
    pub rows: usize,
    pub logits_per_row: u64,
    pub live_resources: usize,
}

impl fmt::Display for FixtureRunReceiptV1 {
    /// Every value here is a bounded identifier, a label of the same charset,
    /// or a number, so the receipt needs no escaping to stay one JSON line.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{{\"status\":\"{}\",\"scenario\":\"{}\",\"bundle\":\"{}\",\"topology\":\"{}\",\
             \"ranks\":{},\"device\":\"{}\",\"architecture\":\"{}\",\"abi_version\":{},\
             \"request\":\"{}\",\"prefill_tokens\":{},\"executions\":{},\"rows\":{},\
             \"logits_per_row\":{},\"live_resources\":{}}}",
            self.scenario.status(),
            self.scenario.as_str(),
            self.bundle_id,
            self.topology_id,
            self.rank_count,
            self.device_services,
            self.architecture,
            self.abi_version,
            self.label,
            self.prefill_tokens,
            self.executions,
            self.rows,
            self.logits_per_row,
            self.live_resources
        )
    }
}

/// Parses the whole surface, with every command named.
///
/// This is what an operator's program uses: nothing is assumed, so an
/// invocation that names no command is refused rather than run as one.
pub fn parse_host_command_v1(arguments: &[String]) -> Result<HostCommandV1, HostCommandErrorV1> {
    parse_named_command_v1(arguments, false)
}

/// Parses the same surface, with a bare flag list meaning `conformance-run`.
///
/// The conformance host exists to drive one batch through one bundle, so that
/// is what it does when its invocation names no command. Serving is still
/// reached by naming `serve`, because the two are different commands and only
/// the verb says which one was meant.
pub fn parse_conformance_host_command_v1(
    arguments: &[String],
) -> Result<HostCommandV1, HostCommandErrorV1> {
    parse_named_command_v1(arguments, true)
}

fn terminal_help_v1(arguments: &[String]) -> bool {
    let mut words = arguments.iter().peekable();
    while let Some(word) = words.next() {
        if matches!(word.as_str(), "--help" | "-h") {
            return words.peek().is_none();
        }
        if !word.starts_with('-') || VALUELESS_FLAGS_V1.contains(&word.as_str()) {
            continue;
        }
        words.next();
        if word == "--served-model-name" {
            while words.peek().is_some_and(|name| !name.starts_with("--")) {
                words.next();
            }
        }
    }
    false
}

fn parse_named_command_v1(
    arguments: &[String],
    bare_flags_are_conformance_run: bool,
) -> Result<HostCommandV1, HostCommandErrorV1> {
    if terminal_help_v1(arguments) {
        let text = if arguments.starts_with(&["bundle".into(), "check-frontend".into()]) {
            CHECK_FRONTEND_USAGE_V1
        } else {
            HOST_COMMAND_USAGE_V1
        };
        return Ok(HostCommandV1::Display { text: text.into() });
    }
    let command = arguments.first().ok_or_else(|| usage_error("no command"))?;
    match command.as_str() {
        "profiles" => Ok(HostCommandV1::Display {
            text: crate::profiles::inspect(&arguments[1..])?,
        }),
        "bundle" => parse_bundle_command_v1(&arguments[1..]),
        "kernel" => parse_kernel_command_v1(&arguments[1..]),
        "serve" => parse_serve_command_v1(&arguments[1..]),
        "verify" => parse_checkpoint_verify_command_v1(&arguments[1..]),
        "conformance-run" => parse_conformance_run_command_v1(&arguments[1..]),
        _ if command.starts_with('-') && bare_flags_are_conformance_run => {
            parse_conformance_run_command_v1(arguments)
        }
        _ => Err(usage_error(format!("unsupported command '{command}'"))),
    }
}

/// Parses the `bundle` commands, which act on a bundle without loading it.
pub fn parse_bundle_command_v1(arguments: &[String]) -> Result<HostCommandV1, HostCommandErrorV1> {
    let command = arguments
        .first()
        .ok_or_else(|| usage_error("bundle requires a command"))?;
    if command == "check-frontend" {
        return parse_check_frontend_command_v1(&arguments[1..]);
    }
    let parsed = ArgumentsV1::parse(&arguments[1..])?;
    match command.as_str() {
        "verify" => {
            parsed.reject_unsupported("bundle verify", &["--bundle"])?;
            Ok(HostCommandV1::Verify {
                host: parsed.host_compatibility()?,
                selection: parsed.selection("bundle verify")?,
            })
        }
        "install" => Err(usage_error(
            "bundle install is gone: packages carry no rank images; serve a package directory \
             with --bundle DIR --checkpoint DIR",
        )),
        _ => Err(usage_error(format!(
            "unsupported bundle command '{command}'"
        ))),
    }
}

/// Parses `bundle check-frontend`. The vector flags repeat, so they are taken
/// out first and the rest is read as every other bundle command reads it.
fn parse_check_frontend_command_v1(
    arguments: &[String],
) -> Result<HostCommandV1, HostCommandErrorV1> {
    let mut vectors = FrontendVectorFilesV1::default();
    let mut rest = Vec::new();
    let mut words = arguments.iter();
    while let Some(word) = words.next() {
        let files = match word.as_str() {
            "--render" => &mut vectors.render,
            "--tokenizer" => &mut vectors.tokenizer,
            "--streaming" => &mut vectors.streaming,
            _ => {
                rest.push(word.clone());
                continue;
            }
        };
        let file = words
            .next()
            .filter(|file| !file.is_empty() && !file.starts_with("--"))
            .ok_or_else(|| usage_error(format!("{word} requires a FILE")))?;
        files.push(PathBuf::from(file));
    }
    if vectors.render.is_empty() && vectors.tokenizer.is_empty() && vectors.streaming.is_empty() {
        return Err(usage_error(
            "bundle check-frontend requires at least one --render, --tokenizer or --streaming FILE",
        ));
    }
    let parsed = ArgumentsV1::parse(&rest)?;
    parsed.reject_unsupported("bundle check-frontend", &["--bundle", "--topology"])?;
    Ok(HostCommandV1::CheckFrontend {
        host: parsed.host_compatibility()?,
        selection: parsed.selection("bundle check-frontend")?,
        topology: parsed.topology.clone(),
        vectors,
    })
}

/// Parses `verify --model DIR [--manifest FILE[#ENTRY]] [--allow-unrecorded] [--jobs N]`.
fn parse_checkpoint_verify_command_v1(
    arguments: &[String],
) -> Result<HostCommandV1, HostCommandErrorV1> {
    let mut root = None;
    let mut manifest = None;
    let mut allow_unrecorded = false;
    let mut jobs = None;
    let mut words = arguments.iter();
    while let Some(word) = words.next() {
        let mut value = || {
            words
                .next()
                .filter(|value| !value.is_empty())
                .ok_or_else(|| usage_error(format!("verify {word} needs a value")))
        };
        match word.as_str() {
            "--model" if root.is_none() => root = Some(PathBuf::from(value()?)),
            "--manifest" if manifest.is_none() => {
                let spec = value()?;
                manifest = Some(match spec.rsplit_once('#') {
                    Some((file, entry)) if !file.is_empty() && !entry.is_empty() => {
                        (PathBuf::from(file), Some(entry.to_owned()))
                    }
                    Some(_) => {
                        return Err(usage_error("--manifest takes FILE or FILE#ENTRY"));
                    }
                    None => (PathBuf::from(spec), None),
                });
            }
            "--allow-unrecorded" if !allow_unrecorded => allow_unrecorded = true,
            "--jobs" if jobs.is_none() => {
                jobs = Some(
                    value()?
                        .parse::<usize>()
                        .ok()
                        .filter(|jobs| (1..=256).contains(jobs))
                        .ok_or_else(|| usage_error("--jobs takes a count from 1 to 256"))?,
                );
            }
            _ => {
                return Err(usage_error(format!(
                    "verify does not take {word} here; verify --model DIR [--manifest FILE[#ENTRY]] [--allow-unrecorded] [--jobs N]"
                )));
            }
        }
    }
    let root = root.ok_or_else(|| usage_error("verify requires --model DIR"))?;
    Ok(HostCommandV1::CheckpointVerify {
        options: crate::checkpoint_verify::VerifyOptionsV1 {
            root,
            manifest,
            allow_unrecorded,
            jobs: jobs.unwrap_or_else(|| crate::direct::process_cpus_v1().clamp(1, 8)),
        },
    })
}

fn parse_kernel_command_v1(arguments: &[String]) -> Result<HostCommandV1, HostCommandErrorV1> {
    let command = arguments
        .first()
        .ok_or_else(|| usage_error("kernel requires a command"))?;
    if command != "build" && command != "inspect" {
        return Err(usage_error(format!(
            "unsupported kernel command: {}",
            command
        )));
    }
    if !matches!(arguments.len(), 3 | 5)
        || arguments[1] != "--manifest"
        || (arguments.len() == 5 && arguments[3] != "--cache-dir")
    {
        return Err(usage_error(format!(
            "kernel {} requires --manifest FILE",
            command
        )));
    }
    let manifest = PathBuf::from(&arguments[2]);
    let cache_dir = arguments.get(4).map(PathBuf::from);
    Ok(if command == "build" {
        HostCommandV1::KernelBuild {
            manifest,
            cache_dir,
        }
    } else {
        HostCommandV1::KernelInspect {
            manifest,
            cache_dir,
        }
    })
}

fn resolve_api_key_v1(
    cli: Option<runtime_server::ApiKey>,
    chopinfer: Option<String>,
    vllm: Option<String>,
) -> Option<runtime_server::ApiKey> {
    cli.or_else(|| {
        chopinfer
            .filter(|value| !value.is_empty())
            .or_else(|| vllm.filter(|value| !value.is_empty()))
            .map(|value| runtime_server::ApiKey::new(&value))
    })
}

fn model_options_environment_v1(value: Option<&str>) -> Result<Vec<String>, HostCommandErrorV1> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let mut names = std::collections::BTreeSet::new();
    let mut args = Vec::new();
    for assignment in value.split(',') {
        let (name, value) = assignment
            .split_once('=')
            .ok_or_else(|| usage_error("CHOPINFER_MODEL_OPTIONS requires name=value,..."))?;
        if !runtime_bundle::model_options::valid_name(name)
            || value.len() > 1024
            || !names.insert(name)
        {
            return Err(usage_error(
                "CHOPINFER_MODEL_OPTIONS contains an invalid or repeated option",
            ));
        }
        args.extend(["--model-option".into(), assignment.into()]);
    }
    Ok(args)
}

fn parse_serve_command_v1(arguments: &[String]) -> Result<HostCommandV1, HostCommandErrorV1> {
    let env_defaults =
        model_options_environment_v1(std::env::var("CHOPINFER_MODEL_OPTIONS").ok().as_deref())?;
    let arguments = crate::profiles::apply_defaults(&env_defaults, arguments);
    let resolved = crate::profiles::resolve(&arguments)?;
    let parsed = ArgumentsV1::parse(&resolved.arguments)?;
    parsed.reject_unsupported(
        "serve",
        &[
            "--bundle",
            "--model",
            "--tensor-parallel-size",
            "--model-cache-dir",
            "--model-option",
            "--list-model-options",
            "--activation-policy",
            "--modalities",
            "--listen",
            "--served-model-name",
            "--enable-request-id-headers",
            "--api-key",
            "--allowed-origins",
            "--allowed-methods",
            "--allowed-headers",
            "--allow-credentials",
            "--topology",
            "--devices",
            "--max-context-tokens",
            "--kv-memory-fraction",
            "--kv-pages",
            "--context-parallel-size",
            "--kv-cache-dtype",
            "--speculative-tokens",
            "--speculative-rows",
            "--speculation",
            "--draft-cost",
            "--prefix-cache",
            "--verify-warmup",
            "--prefill-budget",
            "--prefill-itl-target-ms",
            "--prefill-decode-steps",
            "--decode-tbt-target-ms",
            "--mixed-min-prefill-tokens",
            "--mixed-decode-steps",
            "--kv-offload-gib",
            "--kv-offload-min-tokens",
            "--kv-offload-gbps",
            "--kv-offload-prefix",
            "--admission",
            "--admission-watermark",
            "--max-preemptions",
            "--debug-preempt-every",
            "--debug-swap-every",
            "--max-sequences",
            "--max-batch-tokens",
            "--prefill-chunk-tokens",
            "--queue-depth",
            "--queue-wait-ms",
            "--drain-timeout-ms",
            "--max-request-body-bytes",
            "--max-request-string-bytes",
            "--responses-store-bytes",
            "--max-tokenizer-input-bytes",
            "--max-images-per-request",
            "--allowed-local-media-path",
            "--allowed-media-domains",
            "--media-allow-private-addresses",
            "--media-request-timeout-ms",
            "--media-fetch-timeout-ms",
            "--media-connect-timeout-ms",
            "--media-max-redirects",
            "--debug-reference-tokens",
            "--checkpoint",
            "--load-threads",
            "--fake-host-services",
        ],
    )?;
    let direct_checkpoint = match &parsed.direct_checkpoint {
        Some(_) if parsed.model.is_some() => {
            return Err(usage_error(
                "--model loads its own checkpoint; --checkpoint names the checkpoint of a --bundle",
            ));
        }
        Some(path) if path.as_os_str().is_empty() => {
            return Err(usage_error("--checkpoint needs a directory"));
        }
        checkpoint => checkpoint.clone(),
    };
    let load_threads = match parsed.load_threads {
        Some(_) if direct_checkpoint.is_none() && parsed.model.is_none() => {
            return Err(usage_error(
                "--load-threads requires --model or --checkpoint",
            ));
        }
        Some(threads) => Some(
            u16::try_from(threads)
                .ok()
                .and_then(std::num::NonZeroU16::new)
                .ok_or_else(|| usage_error("--load-threads takes a count from 1 to 65535"))?,
        ),
        None => None,
    };
    let listen = match &parsed.listen {
        Some(listen) => listen.clone(),
        None => DEFAULT_LISTEN_ADDRESS_V1.to_owned(),
    };
    let listen = listen.parse::<SocketAddr>().map_err(|error| {
        usage_error(format!(
            "--listen takes an address and port, got '{listen}': {error}"
        ))
    })?;
    let maximum_sequences = parsed
        .maximum_sequences
        .unwrap_or(DEFAULT_SERVED_SEQUENCES_V1);
    if maximum_sequences == 0 {
        return Err(usage_error("--max-sequences must admit at least one"));
    }
    if parsed.kv_pages == Some(0) {
        return Err(usage_error("--kv-pages must allow at least one page"));
    }
    let budget_bytes = parsed
        .kv_offload_gib
        .unwrap_or(0)
        .checked_mul(1024 * 1024 * 1024)
        .ok_or_else(|| usage_error("--kv-offload-gib overflows bytes"))?;
    if budget_bytes != 0 && parsed.optimistic_admission == Some(false) {
        return Err(usage_error(
            "--kv-offload-gib requires --admission optimistic",
        ));
    }
    if parsed.kv_offload_prefix == Some(true) {
        return Err(usage_error(
            "--kv-offload-prefix on requires the deferred host prefix cache",
        ));
    }
    let kv_offload = crate::offload::KvOffloadOptionsV1 {
        budget_bytes,
        minimum_tokens: parsed.kv_offload_min_tokens.unwrap_or(64),
        bytes_per_second: parsed.kv_offload_rate,
    };
    if parsed.debug_swap_every.is_some()
        && (budget_bytes == 0 || parsed.optimistic_admission == Some(false))
    {
        return Err(usage_error(
            "--debug-swap-every requires positive --kv-offload-gib and --admission optimistic",
        ));
    }
    if parsed.debug_swap_every == Some(0) {
        return Err(usage_error("--debug-swap-every must be positive"));
    }
    if parsed.debug_swap_every.is_some() && parsed.debug_preempt_every.is_some() {
        return Err(usage_error(
            "choose --debug-swap-every or --debug-preempt-every",
        ));
    }
    let admission = match parsed.optimistic_admission {
        Some(true) | None => {
            let defaults = OptimisticAdmission::DEFAULT;
            let optimistic = OptimisticAdmission::new(
                parsed
                    .admission_watermark_millionths
                    .unwrap_or(defaults.watermark_millionths()),
                parsed
                    .maximum_preemptions
                    .unwrap_or(defaults.maximum_preemptions()),
            )
            .map_err(|error| usage_error(error.to_string()))?;
            let optimistic = match parsed.debug_swap_every.or(parsed.debug_preempt_every) {
                Some(every) => optimistic
                    .with_forced_preemption(every)
                    .map_err(|_| usage_error("--debug-preempt-every must be positive"))?,
                None => optimistic,
            };
            AdmissionPolicy::Optimistic(optimistic)
        }
        Some(false) => {
            for (flag, given) in [
                (
                    "--admission-watermark",
                    parsed.admission_watermark_millionths.is_some(),
                ),
                ("--max-preemptions", parsed.maximum_preemptions.is_some()),
                (
                    "--debug-preempt-every",
                    parsed.debug_preempt_every.is_some(),
                ),
            ] {
                if given {
                    return Err(usage_error(format!(
                        "{flag} requires --admission optimistic"
                    )));
                }
            }
            AdmissionPolicy::Reserved
        }
    };
    // `serve --model` gets its speculation and scheduling defaults from the
    // family's serving profile only after its package is selected, and then
    // parses the whole serve again with them. Only checks that depend on
    // those defaults (speculation needs tokens, a TBT target needs the TBT
    // budget) wait for that parse; every explicit value is checked here.
    let profile_pending = parsed.model.is_some();
    let prefill_budget = prefill_budget_v1(&parsed, profile_pending)?;
    if !profile_pending
        && parsed.speculation_mode.is_some()
        && parsed.speculative_tokens.unwrap_or(0) == 0
    {
        return Err(usage_error(
            "--speculation requires positive --speculative-tokens",
        ));
    }
    let mut http_options = parsed.http_options.clone();
    http_options.api_key = resolve_api_key_v1(
        http_options.api_key,
        std::env::var("CHOPINFER_API_KEY").ok(),
        std::env::var("VLLM_API_KEY").ok(),
    );
    http_options.validate().map_err(usage_error)?;
    if !profile_pending
        && parsed.speculative_rows.is_some()
        && parsed.speculative_tokens.unwrap_or(0) == 0
    {
        return Err(usage_error(
            "--speculative-rows requires positive --speculative-tokens",
        ));
    }
    if parsed.model.is_none() && parsed.activation_policy.is_some() {
        return Err(usage_error(
            "--activation-policy applies to --model; an explicit package serves the precision \
             it was emitted with",
        ));
    }
    if parsed.model.is_none() && parsed.modalities.is_some() {
        return Err(usage_error(
            "--modalities applies to --model; an explicit package serves the modalities it was \
             emitted with",
        ));
    }
    Ok(HostCommandV1::Serve {
        model_options: parsed.model_options.clone(),
        list_model_options: parsed.list_model_options,
        profile_name: resolved.profile,
        profile_drafter: resolved.drafter,
        qualified_max_sequences: resolved.qualified_max_sequences,
        direct_package: None,
        http_options: Box::new(http_options),
        devices: parsed
            .devices
            .clone()
            .map(|ordinals| Box::new(crate::DeviceSelectionV1(ordinals))),
        host: parsed.host_compatibility()?,
        topology: parsed.topology.clone(),
        device: parsed.device_services(),
        listen,
        serving: Box::new(ServingOptionsV1 {
            maximum_context_tokens: parsed.maximum_context_tokens,
            maximum_sequences,
            maximum_batch_tokens: parsed.maximum_batch_tokens,
            prefill_chunk_tokens: parsed.prefill_chunk_tokens,
            queue_depth: parsed.queue_depth,
            queue_wait_ms: parsed.queue_wait_ms.unwrap_or(DEFAULT_QUEUE_WAIT_MS_V1),
            drain_timeout_ms: parsed.drain_timeout_ms.unwrap_or(30_000),
            maximum_request_body_bytes: parsed.maximum_request_body_bytes,
            maximum_request_string_bytes: parsed.maximum_request_string_bytes,
            responses_store_bytes: parsed.responses_store_bytes.unwrap_or(0),
            maximum_tokenizer_input_bytes: parsed.maximum_tokenizer_input_bytes,
            maximum_images_per_request: match parsed.maximum_images_per_request {
                Some(count) if count == 0 || count > usize::from(u16::MAX) => {
                    return Err(usage_error(
                        "--max-images-per-request takes a count from 1 to 65535",
                    ));
                }
                count => count,
            },
            kv_memory_fraction_millionths: parsed.kv_memory_fraction_millionths.unwrap_or(900_000),
            kv_pages: parsed.kv_pages.map(|pages| pages as u64),
            context_parallel_size: match parsed.context_parallel_size {
                Some(0) => return Err(usage_error("--context-parallel-size must be at least 1")),
                size => size.unwrap_or(1),
            },
            kv_cache_dtype: parsed.kv_cache_dtype.unwrap_or_default(),
            prefix_cache: parsed.prefix_cache.unwrap_or(true),
            debug_swap: parsed.debug_swap_every.is_some(),
            kv_offload,
            verify_warmup: parsed.verify_warmup.unwrap_or(true),
            admission,
            prefill_budget,
        }),
        speculative_tokens: parsed.speculative_tokens.unwrap_or(0),
        speculative_rows: parsed
            .speculative_rows
            .map(|rows| {
                u16::try_from(rows)
                    .map_err(|_| usage_error("--speculative-rows must be at most 65535"))
            })
            .transpose()?,
        speculation_mode: parsed.speculation_mode.unwrap_or(SpeculationMode::Adaptive),
        draft_cost_model: parsed.draft_cost_model.unwrap_or_default(),
        allowed_local_media_path: parsed.allowed_local_media_path.clone(),
        remote_media: Box::new(parsed.remote_media.clone()),
        debug_reference_tokens: parsed.debug_reference_tokens.clone(),
        direct_checkpoint,
        load_threads,
        selection: match parsed.serve_selection()? {
            ServeSelection::Model(model) => ServeSelection::Model(Box::new(LocalModel {
                serve_arguments: crate::profiles::serving_options(&resolved.arguments),
                ..*model
            })),
            bundle => bundle,
        },
    })
}

fn parse_conformance_run_command_v1(
    arguments: &[String],
) -> Result<HostCommandV1, HostCommandErrorV1> {
    let parsed = ArgumentsV1::parse(arguments)?;
    parsed.reject_unsupported(
        "conformance-run",
        &[
            "--bundle",
            "--topology",
            "--fixture-request",
            "--fixture-tokens",
            "--scenario",
            "--kv-memory-fraction",
            "--checkpoint",
            "--fake-host-services",
        ],
    )?;
    if parsed
        .direct_checkpoint
        .as_ref()
        .is_some_and(|path| path.as_os_str().is_empty())
    {
        return Err(usage_error("--checkpoint needs a directory"));
    }
    let label = parsed.fixture_request.clone().ok_or_else(|| {
        usage_error(
            "conformance-run requires --fixture-request LABEL, naming the one bounded batch it \
             drives through the selected bundle",
        )
    })?;
    validate_label_v1(&label)?;
    Ok(HostCommandV1::ConformanceRun {
        host: parsed.host_compatibility()?,
        topology: parsed.topology.clone(),
        device: parsed.device_services(),
        request: FixtureRequestV1 {
            label,
            prefill_tokens: parsed.fixture_tokens,
            // A run that names no scenario commits, which is the positive run.
            scenario: parsed.scenario.unwrap_or_default(),
        },
        kv_memory_fraction_millionths: parsed.kv_memory_fraction_millionths.unwrap_or(900_000),
        selection: parsed.selection("conformance-run")?,
        direct_checkpoint: parsed.direct_checkpoint.clone(),
    })
}

/// Runs one invocation of a front-end program and reports what happened.
///
/// Both programs are this function with a different parser: one requires every
/// command to be named, and the other reads a bare flag list as its conformance
/// run. Neither carries any other logic, so they cannot drift apart.
pub fn run_command_line_v1(
    program: &str,
    arguments: impl Iterator<Item = OsString>,
    parse: fn(&[String]) -> Result<HostCommandV1, HostCommandErrorV1>,
) -> ExitCode {
    let arguments = match arguments
        .map(std::ffi::OsString::into_string)
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(arguments) => arguments,
        Err(argument) => {
            return refuse_v1(
                program,
                &format!(
                    "argument {} is not UTF-8\n{HOST_COMMAND_USAGE_V1}",
                    argument.to_string_lossy()
                ),
            );
        }
    };
    if arguments == ["--version"] || arguments == ["-V"] {
        let version = env!("CARGO_PKG_VERSION");
        let revision = env!("CHOPINFER_BUILD_REVISION");
        if revision.is_empty() {
            println!("{program} {version}");
        } else {
            println!("{program} {version} ({revision})");
        }
        return ExitCode::SUCCESS;
    }
    if let Some((command, rest)) = arguments.split_first()
        && command == "eval"
    {
        // Qualification against a running server: its own exit codes
        // (docs/cli.md), so it bypasses the receipt-or-refusal path.
        let revision = env!("CHOPINFER_BUILD_REVISION");
        let tool = runtime_eval::Tool {
            name: format!("{program} eval"),
            version: env!("CARGO_PKG_VERSION").into(),
            revision: (!revision.is_empty()).then(|| revision.into()),
        };
        // A checkpoint directory names its model through the installed
        // helpers, as `serve --model` selects them.
        let host = runtime_eval::Host {
            checkpoint_model: Box::new(|path| {
                crate::preparation::checkpoint_model(path).map_err(|error| error.to_string())
            }),
            serve_listen: Some(DEFAULT_LISTEN_ADDRESS_V1.into()),
            ..runtime_eval::Host::current()
        };
        return runtime_eval::command_line(&tool, &host, rest);
    }
    match parse(&arguments).and_then(|command| {
        log_resolved_serve_v1(&command);
        run_host_command_v1(&command)
    }) {
        Ok(receipt) => {
            println!("{receipt}");
            ExitCode::SUCCESS
        }
        Err(error) => refuse_v1(program, &error.to_string()),
    }
}

fn log_resolved_serve_v1(command: &HostCommandV1) {
    if let HostCommandV1::Serve {
        selection,
        topology,
        devices,
        listen,
        serving,
        speculative_tokens,
        speculation_mode,
        draft_cost_model,
        http_options,
        qualified_max_sequences,
        profile_drafter,
        ..
    } = command
    {
        let mode = if *speculative_tokens == 0 {
            "off".into()
        } else {
            format!("{speculation_mode:?}")
        };
        eprintln!(
            "resolved served model names: {:?}",
            http_options.served_model_names
        );
        eprintln!(
            "resolved serve: bundle={selection:?} topology={topology:?} devices={devices:?} listen={listen} settings={serving:?} speculative_tokens={speculative_tokens} speculation={mode} draft_cost={draft_cost_model:?} profile_drafter={profile_drafter:?}"
        );
        if let Some(limit) = qualified_max_sequences
            && serving.maximum_sequences > *limit
        {
            eprintln!(
                "warning: --max-sequences {} exceeds the profile's qualified limit {limit}; this override is not qualified",
                serving.maximum_sequences
            );
        }
    }
}

fn refuse_v1(program: &str, message: &str) -> ExitCode {
    eprintln!("{program}: {message}");
    ExitCode::FAILURE
}

/// Runs one parsed command and returns the receipt it earned.
///
/// A command run this way serves every request surface the transport
/// validates. A program that has evidence for less names the one it has with
/// [`run_host_command_with_surface_v1`].
pub fn run_host_command_v1(command: &HostCommandV1) -> Result<String, HostCommandErrorV1> {
    run_host_command_with_surface_v1(command, RequestSurfaceV1::Full)
}

/// Runs one parsed command, serving only the request surface named here.
///
/// The surface bounds `serve` and nothing else, because no other command
/// admits a request. It is the calling program's declaration rather than the
/// host's: this host serves whichever bundle it is pointed at and holds no
/// evidence about any of them, so it neither chooses nor narrows what is named
/// here. What the surface means, and what it refuses, is
/// [`runtime_server::RequestSurfaceV1`].
pub fn run_host_command_with_surface_v1(
    command: &HostCommandV1,
    surface: RequestSurfaceV1,
) -> Result<String, HostCommandErrorV1> {
    run_host_command_from_v1(command, surface, None)
}

/// When a serve started: `serve --model` hands the start of the whole
/// invocation to the serve of its selected package, so readiness includes
/// discovery and package selection; any other serve starts now.
fn serve_started_v1(inherited: Option<std::time::Instant>) -> std::time::Instant {
    inherited.unwrap_or_else(std::time::Instant::now)
}

fn run_host_command_from_v1(
    command: &HostCommandV1,
    surface: RequestSurfaceV1,
    started: Option<std::time::Instant>,
) -> Result<String, HostCommandErrorV1> {
    let _logs = runtime_engine::logging::FlushOnDrop::default();
    match command {
        HostCommandV1::Display { text } => Ok(text.clone()),
        HostCommandV1::Verify { selection, host } => {
            let package = verify_selected_bundle_v1(selection, host)?;
            let manifest = package.manifest();
            Ok(format!(
                "verified bundle '{}' at {}: {} components, {} planned components, {} native module",
                manifest.id(),
                package.root().display(),
                manifest.components().len(),
                manifest.planned_components().len(),
                if manifest.native_module().is_some() {
                    "one"
                } else {
                    "no"
                },
            ))
        }
        HostCommandV1::CheckFrontend {
            selection,
            host,
            topology,
            vectors,
        } => check_frontend_v1(selection, host, topology.as_deref(), vectors),
        HostCommandV1::KernelBuild {
            manifest,
            cache_dir,
        } => match cache_dir {
            Some(root) => kernel::cached(manifest, root, true),
            None => kernel::build(manifest),
        }
        .map_err(run_error),
        HostCommandV1::KernelInspect {
            manifest,
            cache_dir,
        } => match cache_dir {
            Some(root) => kernel::cached(manifest, root, false),
            None => kernel::inspect(manifest),
        }
        .map_err(run_error),
        HostCommandV1::CheckpointVerify { options } => {
            let started = std::time::Instant::now();
            let report =
                crate::checkpoint_verify::verify_checkpoint_v1(options).map_err(run_error)?;
            let mut text = report.lines.join("\n");
            text.push('\n');
            text.push_str(&report.summary(started.elapsed().as_secs_f64()));
            if report.passed(options.allow_unrecorded) {
                Ok(text)
            } else {
                if report.unrecorded != 0 && !options.allow_unrecorded {
                    text.push_str(
                        "\nfiles without a recorded digest fail; name a manifest with --manifest, \
                         verify a hub snapshot or an hf download --local-dir directory, or pass \
                         --allow-unrecorded to accept files other than weights",
                    );
                }
                Err(run_error(text))
            }
        }
        HostCommandV1::Serve {
            model_options,
            list_model_options,
            profile_name,
            profile_drafter,
            qualified_max_sequences: _,
            direct_package,
            http_options,
            devices,
            selection,
            host,
            topology,
            device,
            listen,
            serving,
            speculative_tokens,
            speculative_rows,
            speculation_mode,
            draft_cost_model,
            allowed_local_media_path,
            remote_media,
            debug_reference_tokens,
            direct_checkpoint,
            load_threads,
        } => {
            let started = serve_started_v1(started);
            if let Some(threads) = load_threads {
                crate::direct::select_threads_per_rank_v1(usize::from(threads.get()))
                    .map_err(run_error)?;
            }
            if let Some(root) = direct_checkpoint {
                let opened = std::time::Instant::now();
                crate::direct::select_checkpoint_v1(root).map_err(run_error)?;
                eprintln!(
                    "startup direct_checkpoint: {:.6}s, {}",
                    opened.elapsed().as_secs_f64(),
                    root.display()
                );
            }
            let selection = match selection {
                ServeSelection::Bundle(bundle) => bundle.clone(),
                ServeSelection::Model(model) => {
                    server_config_v1(**serving, surface, None)?;
                    refuse_reference_tokens_variable_v1(std::env::var_os(REFERENCE_TOKENS_ENV_V1))
                        .map_err(usage_error)?;
                    crate::startup::development_skip_warmup(
                        std::env::var_os("CHOPINFER_DEV_SKIP_WARMUP").as_deref(),
                    )
                    .map_err(usage_error)?;
                    if serving
                        .maximum_tokenizer_input_bytes
                        .is_some_and(|n| n == 0 || n > 64 * 1024 * 1024)
                        || serving.maximum_batch_tokens == Some(0)
                        || serving.prefill_chunk_tokens == Some(0)
                    {
                        return Err(usage_error(
                            "invalid token or batch bound for local model serving",
                        ));
                    }
                    // The options are judged against the family's profile
                    // before any package work, as they are after it.
                    let check = |profile: &runtime_bundle::preparation::DirectProfile| {
                        let resolution = crate::profiles::direct_arguments(
                            profile,
                            model.tensor_parallel_size,
                            &model.serve_arguments,
                            Path::new("/direct-package"),
                        )?;
                        parse_serve_command_v1(&resolution.arguments[1..])?;
                        Ok(())
                    };
                    let (package, lease) = crate::preparation::prepare_direct(model, &check)
                        .map_err(|e| run_error(e.to_string()))?;
                    if *list_model_options {
                        let selected = BundleSelectionV1::Directory(package.bundle.clone());
                        let verified = crate::install::verify_selected_bundle_v1(&selected, host)?;
                        verified.manifest().resolve_model_options(model_options)?;
                        return serde_json::to_string_pretty(&verified.manifest().model_options())
                            .map_err(|e| run_error(e.to_string()));
                    }
                    let opened = std::time::Instant::now();
                    crate::direct::select_checkpoint_v1(&model.path).map_err(run_error)?;
                    let identity = crate::direct::selected_checkpoint_v1()
                        .map_err(run_error)?
                        .ok_or_else(|| run_error("no direct-load checkpoint is selected"))?
                        .identity_hex();
                    // The package was selected for these headers and metadata
                    // (configuration, index, tokenizer); a checkpoint that
                    // changed since then is not the one it describes.
                    let metadata: String =
                        runtime_weights::checkpoint::metadata_identity(&model.path)
                            .map_err(|e| run_error(format!("checkpoint metadata: {e}")))?
                            .iter()
                            .map(|b| format!("{b:02x}"))
                            .collect();
                    if identity != package.checkpoint_identity
                        || metadata != package.metadata_identity
                    {
                        return Err(run_error(format!(
                            "the checkpoint changed after its package was selected: \
                             headers {identity} and metadata {metadata}, package selected for \
                             {} and {}",
                            package.checkpoint_identity, package.metadata_identity
                        )));
                    }
                    eprintln!(
                        "startup direct_checkpoint: {:.6}s, {}",
                        opened.elapsed().as_secs_f64(),
                        model.path.display()
                    );
                    // Serve the package as `serve --bundle` serves it, with
                    // the family's serving profile (the one the package path
                    // uses for this checkpoint and rank count) as defaults
                    // under the explicit options. The selected checkpoint
                    // makes its load direct.
                    let resolution = crate::profiles::direct_arguments(
                        &package.profile,
                        model.tensor_parallel_size,
                        &model.serve_arguments,
                        &package.bundle,
                    )?;
                    eprintln!(
                        "startup direct_profile: {} (explicit options override it)",
                        package.profile.name
                    );
                    let mut command = parse_serve_command_v1(&resolution.arguments[1..])?;
                    if let HostCommandV1::Serve {
                        profile_name,
                        profile_drafter,
                        qualified_max_sequences,
                        direct_package,
                        ..
                    } = &mut command
                    {
                        *profile_name = resolution.profile;
                        *profile_drafter = resolution.drafter;
                        *qualified_max_sequences = resolution.qualified_max_sequences;
                        *direct_package = Some(Box::new(DirectPackageIdentityV1 {
                            package_identity: package.package_identity.clone(),
                            metadata_identity: package.metadata_identity.clone(),
                        }));
                    }
                    log_resolved_serve_v1(&command);
                    let result = run_host_command_from_v1(&command, surface, Some(started));
                    drop(lease);
                    return result;
                }
            };
            let result = serve_v1(
                started,
                (profile_name.as_deref(), profile_drafter.as_deref()),
                direct_package.as_deref(),
                devices.as_ref().map(|devices| devices.0.as_slice()),
                &selection,
                host,
                topology.as_deref(),
                *device,
                *listen,
                **serving,
                (*speculative_tokens, speculative_rows.map(usize::from)),
                *speculation_mode,
                *draft_cost_model,
                surface,
                None,
                allowed_local_media_path.as_deref(),
                remote_media,
                debug_reference_tokens.as_deref(),
                http_options.as_ref().clone(),
                model_options,
                *list_model_options,
            );
            if let Err(error) = &result {
                eprintln!("startup/serving failure: {error}");
            }
            result
        }
        HostCommandV1::ConformanceRun {
            selection,
            host,
            topology,
            device,
            request,
            kv_memory_fraction_millionths,
            direct_checkpoint,
        } => {
            if let Some(root) = direct_checkpoint {
                crate::direct::select_checkpoint_v1(root).map_err(run_error)?;
            }
            conformance_run_v1(
                selection,
                host,
                topology.as_deref(),
                *device,
                request,
                *kv_memory_fraction_millionths,
            )
            .map(|receipt| receipt.to_string())
        }
    }
}

/// The decode-phase rows one verify step may carry: the bundle's `declared`
/// maximum, or fewer when `--speculative-rows` asks, but more than the
/// `tokens` one item drafts.
fn verify_rows_v1(
    declared: usize,
    tokens: usize,
    requested: Option<usize>,
) -> Result<usize, HostCommandErrorV1> {
    let rows = requested.unwrap_or(declared);
    if rows > declared || rows <= tokens {
        return Err(usage_error(format!(
            "--speculative-rows must exceed --speculative-tokens and stay within the bundle maximum of {declared}"
        )));
    }
    Ok(rows)
}

use runtime_bundle_sdk::warmup::served_verify_rows as served_verify_rows_v1;

/// The row buckets of steps with at most `rows` rows. The shared SM120
/// bundle runtime captures one decode or verify graph per power of two rows
/// (`bundle_runtime::graphs`), so a step runs its next power of two.
fn step_buckets_v1(rows: usize) -> Vec<usize> {
    std::iter::successors(Some(1_usize), |bucket| bucket.checked_mul(2))
        .take_while(|&bucket| bucket < rows.max(1).saturating_mul(2))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn speculation_config_v1(
    manifest: &BundleManifestV1,
    (tokens, rows): (usize, Option<usize>),
    maximum_sequences: usize,
    mode: SpeculationMode,
    draft_cost_model: DraftCostModel,
) -> Result<Option<SpeculationConfig>, HostCommandErrorV1> {
    if tokens == 0 {
        return Ok(None);
    }
    let declaration = manifest.speculation().ok_or_else(|| {
        run_error("--speculative-tokens requires a bundle speculation declaration")
    })?;
    if tokens > declaration.maximum_draft_tokens() as usize {
        return Err(usage_error(format!(
            "--speculative-tokens exceeds the bundle maximum of {}",
            declaration.maximum_draft_tokens()
        )));
    }
    let served_rows = manifest
        .resource_requirements()
        .required_capabilities()
        .iter()
        .any(|id| id.as_str() == runtime_bundle_abi::SPECULATIVE_NARROW_ROWS_CAPABILITY_V1);
    let declared = served_verify_rows_v1(
        declaration.maximum_verify_tokens() as usize,
        maximum_sequences,
        runtime_bundle_sdk::warmup::narrows_rows(served_rows),
    );
    let rows = verify_rows_v1(declared, tokens, rows)?;
    let source = if declaration.bundle_drafter() {
        DraftSource::Bundle
    } else {
        DraftSource::Host
    };
    // The bundle's per-step drafting state bounds how many items draft.
    let items = declaration
        .maximum_draft_items()
        .map(|items| items as usize);
    SpeculationConfig::new(tokens, rows, mode, source)
        .map(|config| {
            Some(
                config
                    .with_draft_cost_model(draft_cost_model)
                    .with_maximum_draft_items(items),
            )
        })
        .map_err(|error| run_error(format!("invalid speculation limits: {error:?}")))
}

/// What `/health` reports the server serves: values startup already
/// established, gathered once.
fn serving_identity_v1(
    manifest: &BundleManifestV1,
    (profile, direct_package): (Option<&str>, Option<&DirectPackageIdentityV1>),
    (tensor_parallel_size, context_parallel_size): (usize, Option<usize>),
    kv_cache: crate::kv_cache_dtype::KvCacheChoiceV1,
    speculation: Option<SpeculationConfig>,
    serving: &ServingOptionsV1,
) -> Result<runtime_server::ServingIdentity, HostCommandErrorV1> {
    let prefix_capability = manifest
        .resource_requirements()
        .required_capabilities()
        .iter()
        .any(|capability| capability.as_str() == runtime_bundle_abi::PREFIX_CACHE_CAPABILITY_V1);
    let revision = env!("CHOPINFER_BUILD_REVISION");
    Ok(runtime_server::ServingIdentity {
        bundle_id: manifest.id().as_str().to_owned(),
        package_identity: direct_package.map(|package| package.package_identity.clone()),
        checkpoint_identity: crate::direct::selected_checkpoint_v1()
            .map_err(run_error)?
            .map(|checkpoint| checkpoint.identity_hex()),
        metadata_identity: direct_package.map(|package| package.metadata_identity.clone()),
        startup_nonce: startup_nonce_v1()?,
        profile: runtime_server::ServingProfile {
            name: profile.map(str::to_owned),
            tensor_parallel_size,
            context_parallel_size,
            speculation: runtime_server::ServingSpeculation {
                mode: match speculation.map(SpeculationConfig::mode) {
                    None => "off",
                    Some(SpeculationMode::Fixed) => "fixed",
                    Some(SpeculationMode::Adaptive) => "adaptive",
                }
                .to_owned(),
                tokens: speculation.map_or(0, SpeculationConfig::maximum_draft_tokens),
                drafter: speculation
                    .and(manifest.speculation())
                    .filter(|declaration| declaration.bundle_drafter())
                    .and_then(|declaration| declaration.drafter_kind())
                    .map(str::to_owned),
            },
            kv_cache_dtype: kv_cache.label().to_owned(),
            prefix_cache: serving.prefix_cache && prefix_capability,
            max_sequences: serving.maximum_sequences,
        },
        kv_page_tokens: manifest.kv_layout().page_tokens(),
        version: env!("CARGO_PKG_VERSION").to_owned(),
        source_commit: (!revision.is_empty()).then(|| revision.to_owned()),
        model_options: Default::default(),
        model_options_qualified: None,
    })
}

/// 128 random bits for one server process, in hex.
fn startup_nonce_v1() -> Result<String, HostCommandErrorV1> {
    let mut bytes = [0_u8; 16];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut file| std::io::Read::read_exact(&mut file, &mut bytes))
        .map_err(|error| run_error(format!("cannot draw the startup nonce: {error}")))?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn validate_profile_drafter_v1(
    manifest: &BundleManifestV1,
    expected: Option<&str>,
    tokens: usize,
) -> Result<(), HostCommandErrorV1> {
    let Some(expected) = expected.filter(|_| tokens > 0) else {
        return Ok(());
    };
    let actual = match manifest.speculation() {
        Some(declaration) if declaration.bundle_drafter() => declaration.drafter_kind(),
        _ => Some("none"),
    };
    if actual != Some(expected) {
        let actual = actual.unwrap_or("unspecified");
        return Err(usage_error(format!(
            "profile drafter '{expected}' does not match package drafter '{actual}'; select a matching package or repackage it with a drafter_kind declaration"
        )));
    }
    Ok(())
}

fn offload_startup_probe_v1() -> bool {
    std::env::var_os("CHOPINFER_KV_OFFLOAD_STARTUP_CHECK").is_some_and(|v| v == "1")
}
fn offload_capability_v1(manifest: &BundleManifestV1) -> bool {
    let capabilities = manifest.resource_requirements().required_capabilities();
    [
        crate::offload::KV_OFFLOAD_CAPABILITY_V1,
        runtime_bundle_abi::PREFIX_CACHE_CAPABILITY_V1,
    ]
    .iter()
    .all(|name| capabilities.iter().any(|c| c.as_str() == *name))
}
/// Opens a verified package's native module and its frontend export, and
/// reads the deployment that export serves, refusing a package that
/// declares no frontend before any of its native code runs.
fn open_served_deployment_v1(
    package: &BundlePackageV1,
    host: &BundleHostCompatibilityV1,
    config: ServerConfig,
) -> Result<
    (
        runtime_bundle_abi::BundleLibraryV1,
        runtime_bundle_abi::frontend::FrontendExportV1,
        crate::deployment::BundleDeploymentV1,
    ),
    HostCommandErrorV1,
> {
    let refused = |error: crate::deployment::BundleDeploymentErrorV1| {
        run_error(format!("the selected bundle cannot be served: {error}"))
    };
    crate::deployment::require_frontend_capability_v1(package.manifest()).map_err(refused)?;
    let library = crate::install::open_native_module_v1(package, host)?;
    let export =
        crate::deployment::frontend_export_v1(package.manifest(), &library).map_err(refused)?;
    let deployment = deployment_from_package_v1(package, &export, config).map_err(refused)?;
    Ok((library, export, deployment))
}

/// The frontend the transport serves out of `deployment`, opened over the
/// package's authenticated frontend component by its export. Without the
/// media capability, declared media are refused with the request.
pub(crate) fn served_frontend_v1(
    export: &runtime_bundle_abi::frontend::FrontendExportV1,
    deployment: &crate::deployment::BundleDeploymentV1,
    media_enabled: bool,
) -> Result<runtime_frontend::BundleFrontendV1, HostCommandErrorV1> {
    let frontend =
        runtime_frontend::BundleFrontendV1::from_export(export, deployment.frontend_component())
            .map_err(|error| run_error(format!("invalid frontend: {error}")))?
            .with_maximum_total_tokens(deployment.frontend().maximum_total_tokens());
    Ok(if media_enabled {
        frontend
    } else {
        frontend.with_served_media(&[])
    })
}

/// Runs `bundle check-frontend`: every vector file is read first, then the
/// package is opened as serve opens it and each case runs against what it
/// serves. Each case prints its line as it finishes.
fn check_frontend_v1(
    selection: &BundleSelectionV1,
    host: &BundleHostCompatibilityV1,
    topology: Option<&str>,
    files: &FrontendVectorFilesV1,
) -> Result<String, HostCommandErrorV1> {
    use crate::frontend_check::{
        FrontendCheckV1, read_render_vectors_v1, read_streaming_vectors_v1,
        read_tokenizer_vectors_v1,
    };
    fn read_all<T>(
        paths: &[PathBuf],
        read: fn(&[u8]) -> Result<Vec<T>, String>,
    ) -> Result<Vec<T>, HostCommandErrorV1> {
        let mut cases = Vec::new();
        for path in paths {
            let bytes = std::fs::read(path)
                .map_err(|error| usage_error(format!("cannot read {}: {error}", path.display())))?;
            cases.extend(read(&bytes).map_err(|error| {
                usage_error(format!("{} is not a vector file: {error}", path.display()))
            })?);
        }
        Ok(cases)
    }
    let render = read_all(&files.render, read_render_vectors_v1)?;
    let tokenizer = read_all(&files.tokenizer, read_tokenizer_vectors_v1)?;
    let streaming = read_all(&files.streaming, read_streaming_vectors_v1)?;
    // A check of nothing proves nothing, so it is refused before the package
    // is read or any of its code runs.
    if render.is_empty() && tokenizer.is_empty() && streaming.is_empty() {
        return Err(usage_error(
            "bundle check-frontend found no cases in its vector files; a check needs at least one",
        ));
    }
    let config = server_config_v1(ServingOptionsV1::default(), RequestSurfaceV1::Full, None)?;
    let package = select_topology_v1(
        crate::install::verify_selected_bundle_v1(selection, host)?,
        topology,
        None,
    )?;
    let (_library, export, deployment) = open_served_deployment_v1(&package, host, config)?;
    let check = FrontendCheckV1::from_deployment(package.manifest(), &export, deployment, config)
        .map_err(run_error)?;
    let outcomes = check.run_v1(&render, &tokenizer, &streaming);
    for outcome in &outcomes {
        println!("{outcome}");
    }
    let failed = outcomes
        .iter()
        .filter(|outcome| outcome.result.is_err())
        .count();
    let manifest = package.manifest();
    let checked = format!(
        "{} frontend vector(s) of bundle '{}', topology {}",
        outcomes.len(),
        manifest.id(),
        manifest.target().topologies()[0].id()
    );
    if failed == 0 {
        Ok(format!("checked {checked}: all passed"))
    } else {
        Err(run_error(format!("{failed} of {checked} failed")))
    }
}

#[allow(clippy::too_many_arguments)]
fn serve_v1(
    readiness_started: std::time::Instant,
    (profile_name, profile_drafter): (Option<&str>, Option<&str>),
    direct_package: Option<&DirectPackageIdentityV1>,
    devices: Option<&[i32]>,
    selection: &BundleSelectionV1,
    host: &BundleHostCompatibilityV1,
    topology: Option<&str>,
    device: DeviceServicesV1,
    listen: SocketAddr,
    serving: ServingOptionsV1,
    speculative: (usize, Option<usize>),
    speculation_mode: SpeculationMode,
    draft_cost_model: DraftCostModel,
    surface: RequestSurfaceV1,
    envelope: Option<DeveloperServingEnvelopeV1>,
    allowed_local_media_path: Option<&Path>,
    remote_media: &runtime_media::remote::RemoteMediaPolicy,
    debug_reference_tokens: Option<&Path>,
    http_options: runtime_server::HttpOptions,
    model_options: &std::collections::BTreeMap<String, String>,
    list_model_options: bool,
) -> Result<String, HostCommandErrorV1> {
    // Refused before anything loads, so a launcher that still exports the
    // variable fails at startup rather than serving forced tokens.
    refuse_reference_tokens_variable_v1(std::env::var_os(REFERENCE_TOKENS_ENV_V1))
        .map_err(usage_error)?;
    let tokenizer_input_bytes = serving
        .maximum_tokenizer_input_bytes
        .unwrap_or(DEFAULT_TOKENIZER_INPUT_BYTES);
    if tokenizer_input_bytes == 0 || tokenizer_input_bytes > 64 * 1024 * 1024 {
        return Err(usage_error(
            "--max-tokenizer-input-bytes must be between 1 and 67108864",
        ));
    }
    crate::signals::install()
        .map_err(|error| run_error(format!("cannot install signal handlers: {error}")))?;
    let server_config = server_config_v1(serving, surface, envelope)?;
    let BundleSelectionV1::Directory(path) = selection;
    let _cache_lease = runtime_bundle::preparation::CacheLease::for_bundle(
        &path.canonicalize().map_err(|e| run_error(e.to_string()))?,
    )
    .map_err(|e| run_error(e.to_string()))?;
    let package = select_topology_v1(
        crate::install::verify_selected_bundle_v1(selection, host)?,
        topology,
        devices.map(<[i32]>::len),
    )?;
    let resolved_options = package.manifest().resolve_model_options(model_options)?;
    if list_model_options {
        return serde_json::to_string_pretty(&package.manifest().model_options())
            .map_err(|e| run_error(e.to_string()));
    }
    validate_profile_drafter_v1(package.manifest(), profile_drafter, speculative.0)?;
    if crate::offload::allocation_enabled(
        serving.kv_offload.budget_bytes,
        offload_capability_v1(package.manifest()),
        offload_startup_probe_v1(),
    ) {
        let memory = std::fs::read_to_string("/proc/meminfo")
            .map_err(|e| run_error(format!("cannot check host KV memory: {e}")))?;
        let available = crate::offload::mem_available_v1(&memory).map_err(run_error)?;
        crate::offload::preflight_host_memory(serving.kv_offload.budget_bytes, available)
            .map_err(run_error)?;
    }
    let kv_cache = resolve_kv_cache_dtype_v1(package.manifest(), serving.kv_cache_dtype)
        .map_err(usage_error)?;
    println!("{}", kv_cache.describe(package.manifest()));
    let speculation = speculation_config_v1(
        package.manifest(),
        speculative,
        serving.maximum_sequences,
        speculation_mode,
        draft_cost_model,
    )?;
    let bundle_id = package.manifest().id().as_str().to_owned();
    // The served frontend is the package's own export, so the module opens
    // here, before the deployment; the generation worker builds its engine
    // over the same library.
    let (library, frontend_export, deployment) =
        open_served_deployment_v1(&package, host, server_config)?;
    let topology = ServedTopologyV1::new(package.manifest(), topology, devices, device)?
        .with_context_parallel(package.manifest(), serving.context_parallel_size)?;
    let mut identity = serving_identity_v1(
        package.manifest(),
        (profile_name, direct_package),
        (
            topology.device_ordinals.len(),
            topology.reported_context_parallel(package.manifest())?,
        ),
        kv_cache,
        speculation,
        &serving,
    )?;
    identity.model_options = resolved_options;
    identity.model_options_qualified = package
        .manifest()
        .model_options()
        .map(|options| options.is_qualified(&identity.model_options));
    eprintln!(
        "startup serving_identity: {}",
        serde_json::to_string(&identity).map_err(|error| run_error(error.to_string()))?
    );
    crate::services::record_tap_producer_v1(&identity.startup_nonce);
    let mut http_options = http_options;
    http_options.serving_identity = Some(identity);
    remote_media
        .validate()
        .map_err(|error| usage_error(error.to_string()))?;
    let media_enabled = crate::video::media_inputs_required_v1(package.manifest());
    let mut frontend = crate::deployment::configure_served_images_v1(
        served_frontend_v1(&frontend_export, &deployment, media_enabled)?,
        serving.maximum_images_per_request,
    )
    .map_err(usage_error)?;
    if let Some(images) = frontend.maximum_images_per_request() {
        eprintln!("startup images_per_request: {images}");
    }
    if let Some(root) = allowed_local_media_path {
        let root = root.canonicalize().map_err(|error| {
            usage_error(format!(
                "--allowed-local-media-path cannot be resolved: {error}"
            ))
        })?;
        if !root.is_dir() {
            return Err(usage_error(
                "--allowed-local-media-path must name a directory",
            ));
        }
        frontend = frontend.with_allowed_local_media_path(root);
    }
    // A direct package serves the modalities `--modalities` selected, so
    // refusals name the flag.
    frontend = frontend.with_selected_modalities(direct_package.is_some());
    // Warmup waves capture a bundle's decode graph buckets, which only a
    // package with batch limits declares; any other package warms up with
    // one request.
    let warmup_sequences = if package.manifest().batch_limits().is_some() {
        serving.maximum_sequences
    } else {
        1
    };
    // Verify graphs are captured per row and item bucket, which only a
    // package with batch limits declares.
    let verify_warmup = speculation
        .filter(|_| serving.verify_warmup && package.manifest().batch_limits().is_some())
        .map(|speculation| runtime_engine::warmup::VerifyWarmup {
            // Without declared buckets any depth runs as it is.
            item_rows: runtime_bundle_sdk::warmup::item_rows(
                &package
                    .manifest()
                    .speculation()
                    .map(|declaration| {
                        declaration
                            .row_buckets()
                            .iter()
                            .map(|&rows| rows as usize)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default(),
                speculation.maximum_draft_tokens(),
            ),
            maximum_draft_tokens: speculation.maximum_draft_tokens(),
            step_rows: speculation.maximum_verify_tokens(),
            drafting_items: speculation.maximum_draft_items(),
        });
    let policy = deployment
        .policy()
        .with_media_inputs(media_enabled)
        .with_prompt_rows(
            package
                .manifest()
                .resource_requirements()
                .required_capabilities()
                .iter()
                .any(|capability| {
                    capability.as_str() == crate::executor::PROMPT_ROWS_CAPABILITY_V1
                }),
        );
    // An envelope is checked against what this package declares rather than
    // against what the transport was already configured to admit, because the
    // declaration is the only thing here that knows the model's own rows.
    let plan = envelope
        .map(|envelope| {
            resolve_developer_serving_plan_v1(
                package.manifest(),
                deployment.declaration().context.maximum_total_tokens,
                envelope,
            )
        })
        .transpose()?;
    let mut engine_config = engine_config_v1(
        package.manifest(),
        policy.vocabulary(),
        frontend.maximum_total_tokens(),
        serving,
        plan,
    )?;
    if crate::video::media_inputs_required_v1(package.manifest()) {
        let media = deployment
            .declaration()
            .multimodal
            .as_ref()
            .ok_or_else(|| {
                run_error("media capability requires a multimodal frontend declaration")
            })?;
        let pages = media.arena.rows / media.arena.page_rows;
        let mut config = runtime_engine::MediaConfigV1::new(pages as usize, media.arena.page_rows)
            .map_err(|error| run_error(format!("invalid media arena: {error:?}")))?;
        for encoder in &media.encoders {
            config = config
                .with_encoder(
                    encoder.maximum_payload_bytes as u64,
                    encoder.cost_per_row_milli,
                )
                .map_err(|error| run_error(format!("invalid media encoder: {error:?}")))?;
        }
        engine_config = engine_config.with_media(config);
    }
    let position_axes = crate::executor::position_axes_capability_v1(package.manifest());
    require_position_axes_capability_v1(
        deployment.declaration().multimodal.as_ref(),
        position_axes,
    )
    .map_err(run_error)?;
    if position_axes {
        engine_config = engine_config.with_position_axes();
    }
    if let Some(speculation) = speculation {
        engine_config = engine_config.with_speculation(speculation);
    }
    // A developer plan may include one final emitted token without a model row.
    let served_context_tokens = engine_config
        .maximum_sequence_tokens()
        .min(package.manifest().kv_layout().maximum_sequence_tokens() as usize);
    let engine = ServedEngineV1 {
        selection: selection.clone(),
        host: host.clone(),
        topology: topology.clone(),
        device,
        config: engine_config,
        vocabulary: policy.vocabulary().get(),
        maximum_rows: serving.maximum_sequences as u64,
        serving_kv_policy: ServingKvPolicyV1 {
            sampling_vocabulary: Some(policy.vocabulary().get()),
            maximum_context_tokens: Some(served_context_tokens),
            state_reserved_bytes: 0,
            fraction_millionths: serving.kv_memory_fraction_millionths,
            maximum_pages: serving.kv_pages,
        },
        kv_dtype: kv_cache.dtype,
        speculation,
        prefix_cache: serving.prefix_cache,
        kv_offload: serving.kv_offload,
        debug_reference_tokens: debug_reference_tokens.map(Path::to_path_buf),
        model_options: model_options.clone(),
        package,
        library,
    };
    // The engine is built on the worker thread, because the module it runs is
    // loaded there and stays there.
    let backend = EngineGenerationBackend::try_new_with_factory(
        deployment.into_tokenizer(),
        move || engine.open(),
        serving_backend_config_v1(&serving)?.with_tokenizer_input_bytes(tokenizer_input_bytes),
        policy,
    )
    .map_err(|error| run_error(format!("the generation worker did not start: {error}")))?;
    let backend = if media_enabled {
        backend.with_media_processing(
            allowed_local_media_path.map(Path::to_path_buf),
            topology.video_device(),
        )
    } else {
        backend
    };
    let (frontend, backend) = crate::configure_remote_media_v1(frontend, backend, remote_media)
        .map_err(|error| usage_error(error.to_string()))?;
    let server_config = crate::deployment::configure_media_body_limit_v1(
        server_config,
        &frontend,
        serving.maximum_request_body_bytes.is_some(),
    )
    .map_err(usage_error)?;
    eprintln!(
        "startup request_body_bytes: media={} other={}",
        server_config.max_media_body_bytes(),
        server_config.max_body_bytes()
    );
    let frontend: Arc<dyn ServerFrontendV1> = Arc::new(frontend);
    let server_config = constrain_server_context_v1(
        server_config,
        frontend.maximum_total_tokens(),
        backend.maximum_sequence_tokens(),
        serving.maximum_context_tokens.is_some() || envelope.is_some(),
    )?;
    // What the metrics endpoint answers with is this worker's own engine, KV
    // and lifecycle state, taken before the backend is handed to the transport.
    let metrics = Arc::new(backend.runtime_metrics(&http_options.served_model_names));
    let backend = Arc::new(backend);
    let result = serve_transport_v1(
        readiness_started,
        listen,
        backend.clone(),
        frontend,
        server_config,
        &bundle_id,
        &topology,
        device,
        metrics,
        std::time::Duration::from_millis(serving.drain_timeout_ms),
        http_options,
        warmup_sequences,
        verify_warmup,
    );
    if backend.is_stalled() {
        // The watchdog has set an exit deadline. Exit now without waiting on
        // stderr, exit handlers or the stuck worker.
        crate::signals::try_write_stderr(
            "fatal: generation worker stalled; exiting for supervisor restart\n",
        );
        crate::signals::exit_now(1);
    }
    if let Err(error) = &result
        && !matches!(backend.readiness(), runtime_server::BackendReadiness::Ready)
    {
        // A failed native rank may also block destruction. Let the supervisor
        // restart this process without waiting on a poisoned worker join.
        eprintln!("{error}");
        eprintln!("fatal: generation worker failed; exiting for supervisor restart");
        runtime_engine::logging::flush();
        std::process::exit(1);
    }
    backend.shutdown();
    result
}

/// Binds the transport and serves it until it stops.
///
/// Nothing about the transport is reimplemented here: `runtime-server` owns the
/// routes, the admission accounting, and the response shapes. This binds one
/// listener, says what it bound, and hands both over, along with the metrics
/// the running worker publishes about itself.
#[allow(clippy::too_many_arguments)]
fn serve_transport_v1(
    readiness_started: std::time::Instant,
    listen: SocketAddr,
    backend: Arc<dyn GenerationBackend>,
    frontend: Arc<dyn ServerFrontendV1>,
    config: ServerConfig,
    bundle_id: &str,
    topology: &ServedTopologyV1,
    device: DeviceServicesV1,
    metrics: Arc<dyn MetricsSnapshotProvider>,
    drain_timeout: std::time::Duration,
    http_options: runtime_server::HttpOptions,
    warmup_sequences: usize,
    verify_warmup: Option<runtime_engine::warmup::VerifyWarmup>,
) -> Result<String, HostCommandErrorV1> {
    let skip_warmup = crate::startup::development_skip_warmup(
        std::env::var_os("CHOPINFER_DEV_SKIP_WARMUP").as_deref(),
    )
    .map_err(usage_error)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| run_error(format!("cannot start the serving runtime: {error}")))?;
    let result = runtime.block_on(async move {
        let model = frontend
            .models()
            .first()
            .ok_or_else(|| run_error("frontend has no model"))?
            .id
            .clone();
        let warmup_limit = config
            .maximum_context_tokens()
            .unwrap_or(frontend.maximum_total_tokens());
        tokio::select! {
            result = crate::startup::warmup(
                backend.as_ref(),
                model,
                warmup_limit,
                config.acceptance_profile(),
                warmup_sequences,
                skip_warmup,
                verify_warmup,
            ) => {
                result.map_err(|error| run_error(format!("startup warmup failed: {error}")))?;
            }
            () = crate::signals::shutdown() => return Ok("stopped during startup".to_owned()),
        }
        let listener = TcpListener::bind(listen)
            .await
            .map_err(|error| run_error(format!("cannot listen on {listen}: {error}")))?;
        let address = listener
            .local_addr()
            .map_err(|error| run_error(format!("the listener has no address: {error}")))?;
        eprintln!("startup readiness: {:.6}s (elapsed, includes preparation, verification, upload and warmup)", readiness_started.elapsed().as_secs_f64());
        // A served process runs until it is stopped, so what it bound is
        // reported when it binds rather than when it returns.
        println!(
            "{{\"status\":\"listening\",\"address\":\"{address}\",\"bundle\":\"{bundle_id}\",\
             \"topology\":\"{}\",\"ranks\":{},\"device\":\"{}\",\"context_tokens\":{}}}",
            topology.id,
            topology.device_ordinals.len(),
            device.name(),
            config
                .maximum_context_tokens()
                .unwrap_or(frontend.maximum_total_tokens()),
        );
        // Keep HTTP alive for retryable refusals until admitted responses finish
        // or the drain deadline expires. Only then tear down the serving runtime.
        runtime_server::serve_with_shutdown_and_http_options(
            listener,
            backend,
            frontend,
            config,
            Arc::new(SystemMonotonicClock::new()),
            metrics,
            crate::signals::shutdown(),
            drain_timeout,
            http_options,
        )
        .await
        .map_err(|error| run_error(format!("serving stopped: {error}")))?;
        Ok(format!(
            "served bundle '{bundle_id}' on {address} until the listener stopped"
        ))
    });
    runtime.shutdown_timeout(std::time::Duration::from_secs(1));
    result
}

/// Everything the generation worker needs to open its own engine.
///
/// It is carried rather than captured piecemeal so the thread boundary is one
/// value, and so nothing that is not `Send` can be smuggled across it.
struct ServedEngineV1 {
    model_options: std::collections::BTreeMap<String, String>,
    selection: BundleSelectionV1,
    host: BundleHostCompatibilityV1,
    /// The verified package and its authenticated open component handles.
    package: BundlePackageV1,
    /// The package's native module, opened before the deployment, whose
    /// frontend export the transport serves.
    library: runtime_bundle_abi::BundleLibraryV1,
    topology: ServedTopologyV1,
    device: DeviceServicesV1,
    config: EngineConfig,
    vocabulary: usize,
    maximum_rows: u64,
    serving_kv_policy: ServingKvPolicyV1,
    /// The declared KV layout `--kv-cache-dtype` resolved to; it reaches the
    /// instance on either physical-KV path.
    kv_dtype: BundleKvDtypeV1,
    speculation: Option<SpeculationConfig>,
    prefix_cache: bool,
    kv_offload: crate::offload::KvOffloadOptionsV1,
    debug_reference_tokens: Option<PathBuf>,
}

trait ServingPoolControlV1: crate::offload::KvOffloadExecutorV1 {
    fn initialize_offload(
        &self,
        options: crate::offload::KvOffloadOptionsV1,
    ) -> Result<Option<crate::offload::HostArenaPlanV1>, String>;
    fn pages(&self) -> Option<u64>;
    fn prepare(
        &self,
        batch: &PreparedAppendBatch,
    ) -> Result<(), runtime_bundle_abi::BundleStatusV1>;
}

impl<D: HostDeviceBackendV1> ServingPoolControlV1 for crate::services::HostServicesV1Backend<D> {
    fn initialize_offload(
        &self,
        options: crate::offload::KvOffloadOptionsV1,
    ) -> Result<Option<crate::offload::HostArenaPlanV1>, String> {
        if options.budget_bytes == 0 {
            return Ok(None);
        }
        let memory = std::fs::read_to_string("/proc/meminfo")
            .map_err(|e| format!("cannot check host KV memory: {e}"))?;
        self.initialize_kv_offload(options, crate::offload::mem_available_v1(&memory)?)
    }
    fn pages(&self) -> Option<u64> {
        self.serving_kv_pages()
    }
    fn prepare(
        &self,
        batch: &PreparedAppendBatch,
    ) -> Result<(), runtime_bundle_abi::BundleStatusV1> {
        self.prepare_serving_batch(batch)
    }
}

impl ServedEngineV1 {
    /// Loads the selected bundle on this thread and builds the engine over it.
    ///
    /// This runs on the generation worker, which is where the bundle has to
    /// be loaded: the process records one selection, and the loaded host
    /// stays on the thread that loaded it. The module itself was opened
    /// before the deployment, and this load takes that library.
    fn open(self) -> Result<Engine<ServedExecutorV1>, String> {
        let expected = self.package.manifest().clone();
        let loaded = load_verified_process_bundle_v1(
            &self.selection,
            &self.host,
            self.package,
            Some(self.library),
        )
        .map_err(|error| format!("the selected bundle did not load: {error}"))?;
        let loaded = loaded
            .with_model_options(&self.model_options)
            .map_err(|e| e.to_string())?;
        same_selection_v1(&expected, loaded.manifest(), &self.selection)?;
        let batch_wire = batch_wire_capability_v1(loaded.manifest());
        let batch_wire = if self.speculation.is_some() {
            if batch_wire == crate::executor::BatchWireCapabilityV1::V3Media {
                crate::executor::BatchWireCapabilityV1::V3MediaSpeculative
            } else {
                crate::executor::BatchWireCapabilityV1::V3Speculative
            }
        } else {
            batch_wire
        };
        require_speculation_callback_v1(
            self.speculation.is_some(),
            loaded.library().supports_speculation(),
        )?;
        let prefix_capability = loaded
            .manifest()
            .resource_requirements()
            .required_capabilities()
            .iter()
            .any(|capability| {
                capability.as_str() == runtime_bundle_abi::PREFIX_CACHE_CAPABILITY_V1
            });
        require_prefix_cache_callbacks_v1(
            prefix_capability,
            loaded.library().supports_prefix_cache(),
            loaded.library().supports_request_retirement(),
            batch_wire,
        )?;
        require_prefix_state_slot_callback_v1(
            loaded
                .manifest()
                .resource_requirements()
                .required_capabilities()
                .iter()
                .any(|capability| {
                    capability.as_str() == runtime_bundle_abi::PREFIX_CACHE_STATE_SLOT_CAPABILITY_V1
                }),
            prefix_capability && !loaded.manifest().recurrent_state_layouts().is_empty(),
            loaded.library().supports_prefix_state_slot(),
        )?;
        let serving_kv = loaded
            .manifest()
            .resource_requirements()
            .required_capabilities()
            .iter()
            .any(|capability| capability.as_str() == runtime_bundle_abi::SERVING_KV_CAPABILITY_V1);
        let fp8_kv = self.kv_dtype == BundleKvDtypeV1::Fp8E4m3;
        let state_slots = if loaded.manifest().recurrent_state_layouts().is_empty() {
            None
        } else {
            if !loaded.library().supports_request_retirement() {
                return Err("recurrent state requires request retirement".to_owned());
            }
            Some(crate::recurrent::StateSlotPoolV1::new(
                loaded.manifest(),
                self.maximum_rows as usize,
                // One item verifies at most its drafts plus the bonus row;
                // the verify-token bound spans the whole batch.
                self.speculation
                    .map_or(1, |config| config.maximum_draft_tokens() + 1),
            )?)
        };
        let mut serving_policy = self.serving_kv_policy;
        serving_policy.state_reserved_bytes = state_slots
            .as_ref()
            .map_or(0, |pool| pool.reserved_bytes_per_rank());
        let offload_capability = loaded
            .manifest()
            .resource_requirements()
            .required_capabilities()
            .iter()
            .any(|c| c.as_str() == crate::offload::KV_OFFLOAD_CAPABILITY_V1)
            && prefix_capability;
        let allocate_offload = crate::offload::allocation_enabled(
            self.kv_offload.budget_bytes,
            offload_capability,
            offload_startup_probe_v1(),
        );
        if self.kv_offload.budget_bytes != 0 && !offload_capability {
            println!(
                "KV offload: bundle capability absent; victims recompute; arena allocation {}",
                if allocate_offload {
                    "startup probe only"
                } else {
                    "skipped"
                }
            );
        }
        let create_started = std::time::Instant::now();
        let (services, pool_control, mut instance) = open_with_request_retirement_v1(
            loaded.manifest(),
            loaded.library().supports_request_retirement(),
            || {
                open_instance_v1(
                    loaded,
                    &self.topology,
                    self.device,
                    serving_kv.then_some(serving_policy),
                    self.kv_dtype,
                    self.speculation,
                    u32::try_from(self.maximum_rows).unwrap_or(u32::MAX),
                )
            },
        )?;
        eprintln!(
            "startup bundle create: {:.3}s",
            create_started.elapsed().as_secs_f64()
        );
        let config = if serving_kv {
            let pages = pool_control
                .pages()
                .ok_or("the serving bundle did not acquire its KV pool during create")?;
            let page_tokens = u64::from(loaded.manifest().kv_layout().page_tokens());
            let group = self.topology.context_parallel_size as u64;
            println!(
                "serving KV capacity: {} pages, {} tokens",
                pages,
                pages.saturating_mul(page_tokens).saturating_mul(group)
            );
            if group > 1 {
                println!(
                    "context parallel: {group} ranks per group, each holding {pages} pages of \
                     {page_tokens} tokens"
                );
            }
            if fp8_kv {
                println!("serving KV precision: FP8 E4M3 K and V");
            }
            self.config
                .with_kv_page_capacity(
                    usize::try_from(pages)
                        .map_err(|_| "serving KV capacity does not fit this host")?,
                )
                .map_err(|error| format!("the serving KV capacity is unusable: {error:?}"))?
        } else {
            self.config
        };
        let snapshot_pages = if prefix_capability
            && (self.prefix_cache || (offload_capability && self.kv_offload.budget_bytes != 0))
        {
            usize::try_from(instance.describe_prefix_state().map_err(|error| {
                format!("the bundle did not describe its prefix state: {error}")
            })?)
            .map_err(|_| "the prefix snapshot does not fit this host")?
        } else {
            0
        };
        let config = if self.prefix_cache && prefix_capability {
            let pages = snapshot_pages;
            println!("prefix cache: {pages} snapshot pages per anchor");
            config.with_prefix_cache(pages)
        } else {
            if self.prefix_cache {
                println!(
                    "prefix cache: the bundle does not declare {}, so nothing is cached",
                    runtime_bundle_abi::PREFIX_CACHE_CAPABILITY_V1
                );
            }
            config
        };
        let offload_plan = if allocate_offload {
            pool_control.initialize_offload(self.kv_offload)?
        } else {
            None
        };
        if let Some(plan) = offload_plan {
            println!(
                "host KV capacity: {} pages per rank, {} bytes across {} ranks, {} tokens per rank",
                plan.pages_per_rank,
                plan.allocated_bytes,
                plan.rank_count,
                plan.pages_per_rank * loaded.manifest().kv_layout().page_tokens() as usize
            );
        }
        let model_executor = BundleModelExecutorV1::new_with_batch_wire(
            instance,
            self.vocabulary,
            self.maximum_rows,
            batch_wire,
        )
        .with_in_place_state(
            loaded
                .manifest()
                .resource_requirements()
                .required_capabilities()
                .iter()
                .any(|id| {
                    id.as_str() == runtime_bundle_abi::RECURRENT_STATE_IN_PLACE_CAPABILITY_V1
                }),
        )
        .with_position_axes(crate::executor::position_axes_capability_v1(
            loaded.manifest(),
        ))
        .with_row_buckets(if self.speculation.is_some() {
            loaded
                .manifest()
                .speculation()
                .map_or(&[], |speculation| speculation.row_buckets())
        } else {
            &[]
        })
        .with_step_buckets(&step_buckets_v1(
            self.speculation
                .map_or(0, |speculation| speculation.maximum_verify_tokens())
                .max(self.maximum_rows as usize),
        ));
        let model_executor = if let Some(plan) = offload_plan.filter(|_| offload_capability) {
            model_executor.with_kv_offload(
                pool_control.clone(),
                runtime_engine::KvOffloadInfo {
                    host_page_capacity: plan.pages_per_rank,
                    minimum_tokens: self.kv_offload.minimum_tokens,
                    snapshot_pages,
                },
            )
        } else {
            model_executor
        };
        let model_executor = if let Some(pool) = state_slots {
            model_executor.with_state_slots(pool)
        } else {
            model_executor
        };
        let reference = self
            .debug_reference_tokens
            .as_deref()
            .map(ReferenceTokensV1::from_file)
            .transpose()?
            .map(Rc::new);
        #[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
        let model_executor = if let Ok(cuda_services) =
            Rc::clone(&services)
                .downcast::<crate::services::HostServicesV1Backend<crate::cuda::CudaHostDeviceV1>>()
        {
            model_executor.with_view_sampler(cuda_services)
        } else {
            model_executor
        };
        let model_executor = if let Some(reference) = &reference {
            let model_executor = model_executor
                .with_reference_forcing(Rc::clone(reference))
                .ok_or_else(|| {
                    "--debug-reference-tokens forces tokens in the device sampler, which these services lack"
                        .to_owned()
                })?;
            println!(
                "debug: --debug-reference-tokens forces drafts and sampled tokens onto {} reference sequences; outputs are not model samples",
                reference.sequence_count()
            );
            model_executor
        } else {
            model_executor
        };
        let executor = ServedExecutorV1 {
            executor: model_executor,
            _services: services,
            pool_control: serving_kv.then_some(pool_control),
            reference,
        };
        Engine::new_with_runtime_metrics(
            config,
            executor,
            Arc::new(runtime_engine::SystemMonotonicClock::new()),
            EngineMetrics::default(),
            kv_metrics_v1(self.kv_dtype),
        )
        .map_err(|error| format!("the serving engine is unusable: {error}"))
    }
}

/// The engine's KV metrics for the served layout's element: an FP8 layout
/// publishes the FP8 token and capacity gauges; a BF16 or mixed-plane one,
/// whose planes are not all FP8, does not.
fn kv_metrics_v1(dtype: BundleKvDtypeV1) -> KvMetrics {
    match dtype {
        BundleKvDtypeV1::Fp8E4m3 => KvMetrics::new(KvCachePrecision::Fp8),
        BundleKvDtypeV1::Bf16 | BundleKvDtypeV1::MixedPlanes => KvMetrics::default(),
    }
}

fn open_with_request_retirement_v1<T>(
    manifest: &BundleManifestV1,
    supports_request_retirement: bool,
    open: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    if manifest
        .resource_requirements()
        .required_capabilities()
        .iter()
        .any(|capability| {
            capability.as_str() == runtime_bundle_abi::REQUEST_RETIREMENT_CAPABILITY_V1
        })
        && !supports_request_retirement
    {
        return Err(
            "the selected bundle requires request retirement but has no callback".to_owned(),
        );
    }
    open()
}

/// Refuses a bundle that declares prefix caching without what it relies on.
///
/// A hit shares a partly filled page, which needs copy-on-write records, and
/// restored bundle state is freed by retirement like any other.
fn require_prefix_cache_callbacks_v1(
    declared: bool,
    callbacks: bool,
    retirement: bool,
    batch_wire: crate::executor::BatchWireCapabilityV1,
) -> Result<(), String> {
    if !declared {
        return Ok(());
    }
    if !callbacks {
        return Err(
            "the selected bundle requires prefix caching but has no prefix-state callbacks"
                .to_owned(),
        );
    }
    if !retirement || batch_wire == crate::executor::BatchWireCapabilityV1::V1 {
        return Err(
            "the selected bundle requires prefix caching without request retirement and \
             copy-on-write batch records"
                .to_owned(),
        );
    }
    Ok(())
}

/// Refuses a declared state-slot restore without its callback, or without
/// the prefix cache and request state slots it restores into.
fn require_prefix_state_slot_callback_v1(
    declared: bool,
    prefix_cache_with_state: bool,
    callback: bool,
) -> Result<(), String> {
    if !declared {
        return Ok(());
    }
    if !prefix_cache_with_state {
        return Err(format!(
            "the selected bundle requires {} without a prefix cache and request state",
            runtime_bundle_abi::PREFIX_CACHE_STATE_SLOT_CAPABILITY_V1
        ));
    }
    if !callback {
        return Err(
            "the selected bundle requires a state-slot prefix restore but has no \
             restore_prefix_state_in_slot_v1 callback"
                .to_owned(),
        );
    }
    Ok(())
}

/// A declared media position rule needs a bundle that reads explicit
/// positions; otherwise its requests would reach it with none.
fn require_position_axes_capability_v1(
    multimodal: Option<&runtime_frontend::FrontendMultimodalV1>,
    capability: bool,
) -> Result<(), String> {
    let declared = multimodal
        .into_iter()
        .flat_map(|media| &media.modalities)
        .any(|modality| modality.positions.is_some());
    if declared && !capability {
        return Err(format!(
            "a media position rule requires the {} capability",
            crate::executor::POSITION_AXES_CAPABILITY_V1
        ));
    }
    Ok(())
}

fn require_speculation_callback_v1(enabled: bool, present: bool) -> Result<(), String> {
    if enabled && !present {
        return Err(
            "the selected bundle enables speculation but has no commit_accepted_v1 callback"
                .to_owned(),
        );
    }
    Ok(())
}

/// Refuses a package that is not the one the deployment was read from.
///
/// Verifying a directory and loading it are two passes over a path, and what
/// is at that path in between is not this host's to decide. The dangerous case
/// is a second bundle that verifies just as well: its module would run under
/// the first one's frontend, tokenizer, vocabulary, and engine sizing. The
/// validated manifest binds every component path, length, and digest, along
/// with the target, the layouts, and the module's own identity, so comparing
/// the whole of it is what makes the two passes one package.
fn same_selection_v1(
    expected: &BundleManifestV1,
    loaded: &BundleManifestV1,
    selection: &BundleSelectionV1,
) -> Result<(), String> {
    if loaded == expected {
        return Ok(());
    }
    Err(format!(
        "the selection '{}' changed between verification and loading: the deployment was read \
         from bundle '{}' and the module loaded belongs to bundle '{}'",
        selection.display(),
        expected.id(),
        loaded.id(),
    ))
}

/// The bundle executor, holding the services it borrows through.
///
/// The host owns every device resource the bundle reaches, and it owns them
/// through this table. Keeping the table beside the executor is what makes the
/// resources outlive the instance that borrows them, whichever backend issued
/// them.
struct ServedExecutorV1<E: BundleExecutionV1 = LoadedBundleInstanceV1> {
    executor: BundleModelExecutorV1<E>,
    _services: Rc<dyn Any>,
    pool_control: Option<Rc<dyn ServingPoolControlV1>>,
    /// Debug teacher forcing; host drafts follow these sequences when set.
    reference: Option<Rc<ReferenceTokensV1>>,
}

impl<E: BundleExecutionV1> ModelExecutor for ServedExecutorV1<E> {
    fn rollback_preserves_state_v1(&self) -> bool {
        self.executor.rollback_preserves_state_v1()
    }

    fn kv_offload_info_v1(&self) -> Option<runtime_engine::KvOffloadInfo> {
        self.executor.kv_offload_info_v1()
    }
    fn offload_pages_v1(
        &mut self,
        plan: runtime_kv::KvTransferPlan,
    ) -> runtime_engine::KvTransferSubmitResult {
        self.executor.offload_pages_v1(plan)
    }
    fn reload_pages_v1(
        &mut self,
        plan: runtime_kv::KvTransferPlan,
    ) -> runtime_engine::KvTransferSubmitResult {
        self.executor.reload_pages_v1(plan)
    }
    fn poll_kv_transfer_v1(
        &mut self,
        id: runtime_kv::KvTransferId,
    ) -> Result<runtime_engine::KvTransferPoll, ExecutorError> {
        self.executor.poll_kv_transfer_v1(id)
    }
    fn wait_kv_transfer_v1(
        &mut self,
        id: runtime_kv::KvTransferId,
    ) -> Result<runtime_engine::KvTransferPoll, ExecutorError> {
        self.executor.wait_kv_transfer_v1(id)
    }
    fn kv_transfer_drain_timeout_v1(&self) -> std::time::Duration {
        self.executor.kv_transfer_drain_timeout_v1()
    }
    fn drain_kv_transfers_v1(&mut self) -> Result<(), ExecutorError> {
        self.executor.drain_kv_transfers_v1()
    }
    fn admit_request_v1(&mut self, id: RequestId) -> Result<(), ExecutorError> {
        self.executor.admit_request_v1(id)
    }

    fn round_draft_tokens_v1(&self, proposed: usize) -> usize {
        self.executor.round_draft_tokens_v1(proposed)
    }

    fn step_rows_v1(&self, rows: usize) -> usize {
        self.executor.step_rows_v1(rows)
    }

    fn draft_tokens_v1(
        &mut self,
        _id: RequestId,
        history: &[u32],
        maximum: usize,
    ) -> Result<Vec<u32>, ExecutorError> {
        if let Some(reference) = &self.reference {
            let continuation = reference.continuation(history);
            return Ok(continuation[..continuation.len().min(maximum)].to_vec());
        }
        Ok(PromptLookupConfig::default().propose(history, maximum))
    }

    fn retire_request_v1(&mut self, id: runtime_engine::RequestId) -> Result<(), ExecutorError> {
        self.executor.retire_request_v1(id)
    }

    fn capture_prefix_state_v1(
        &mut self,
        id: RequestId,
        tokens: usize,
        pages: &[runtime_kv::BlockId],
    ) -> Result<(), ExecutorError> {
        self.executor.capture_prefix_state_v1(id, tokens, pages)
    }

    fn restore_prefix_state_v1(
        &mut self,
        id: RequestId,
        prefix: &[u32],
        pages: &[runtime_kv::BlockId],
    ) -> Result<(), ExecutorError> {
        self.executor.restore_prefix_state_v1(id, prefix, pages)
    }

    type TransactionState<'a>
        = BundleTransactionStateV1<'a, E>
    where
        Self: 'a;

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        kv_plan: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        if let Some(pool) = &self.pool_control {
            pool.prepare(kv_plan).map_err(|status| {
                ExecutorError::new(format!("serving KV page work failed: {status:?}"))
                    .expect("bounded error")
            })?;
        }
        self.executor.execute(items, kv_plan)
    }
}

/// Creates one instance over whichever device services this build has.
type OpenServedInstanceV1 = (
    Rc<dyn Any>,
    Rc<dyn ServingPoolControlV1>,
    LoadedBundleInstanceV1,
);

/// Opens the served instance for `maximum_sequences` live sequences. A
/// package that requires `chop.served-items.v1` receives the count in its
/// create record. A bundle may still size its slots and graph buckets for its
/// package maximum until it adopts the count (docs/bundle-runtime.md).
fn open_instance_v1(
    loaded: &'static InstalledBundleHostV1,
    topology: &ServedTopologyV1,
    device: DeviceServicesV1,
    policy: Option<ServingKvPolicyV1>,
    kv_dtype: BundleKvDtypeV1,
    speculation: Option<SpeculationConfig>,
    maximum_sequences: u32,
) -> Result<OpenServedInstanceV1, String> {
    let kv = ServedKvV1 { policy, kv_dtype };
    match device {
        DeviceServicesV1::HostMemory => {
            open_host_memory_instance_v1(loaded, topology, kv, speculation, maximum_sequences)
        }
        DeviceServicesV1::Compiled => {
            open_compiled_instance_v1(loaded, topology, kv, speculation, maximum_sequences)
        }
    }
}

/// The serving pool policy, if the bundle acquires the pool, and the KV
/// layout the instance runs either way.
#[derive(Clone, Copy)]
struct ServedKvV1 {
    policy: Option<ServingKvPolicyV1>,
    kv_dtype: BundleKvDtypeV1,
}

fn speculation_service_v1(
    config: Option<SpeculationConfig>,
) -> runtime_bundle_abi::BundleRawSpeculationV1 {
    let mut record = runtime_bundle_abi::BundleRawSpeculationV1::disabled();
    if let Some(config) = config {
        record.maximum_draft_tokens = config.maximum_draft_tokens() as u32;
        record.maximum_verify_tokens = config.maximum_verify_tokens() as u32;
    }
    record
}

#[cfg(feature = "fake-host-services")]
fn open_host_memory_instance_v1(
    loaded: &'static InstalledBundleHostV1,
    topology: &ServedTopologyV1,
    kv: ServedKvV1,
    speculation: Option<SpeculationConfig>,
    maximum_sequences: u32,
) -> Result<OpenServedInstanceV1, String> {
    let (services, instance) = loaded
        .create_served_instance(
            crate::fake::FakeHostDeviceV1::new(),
            &topology.id,
            topology.device_ordinals.clone(),
            kv.policy,
            speculation_service_v1(speculation),
            Some(maximum_sequences),
            Some(kv.kv_dtype),
            topology.context_parallel_size,
        )
        .map_err(|error| format!("the selected bundle refused this host: {error}"))?;
    Ok((services.clone(), services, instance))
}

#[cfg(not(feature = "fake-host-services"))]
fn open_host_memory_instance_v1(
    _loaded: &'static InstalledBundleHostV1,
    _topology: &ServedTopologyV1,
    _kv: ServedKvV1,
    _speculation: Option<SpeculationConfig>,
    _maximum_sequences: u32,
) -> Result<OpenServedInstanceV1, String> {
    Err(NO_HOST_MEMORY_SERVICES_V1.to_owned())
}

#[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
fn open_compiled_instance_v1(
    loaded: &'static InstalledBundleHostV1,
    topology: &ServedTopologyV1,
    kv: ServedKvV1,
    speculation: Option<SpeculationConfig>,
    maximum_sequences: u32,
) -> Result<OpenServedInstanceV1, String> {
    let started = std::time::Instant::now();
    let device = crate::cuda::CudaHostDeviceV1::new()
        .map_err(|error| format!("cannot open the device driver: {error}"))?;
    eprintln!(
        "startup driver initialization: {:.3}s",
        started.elapsed().as_secs_f64()
    );
    let (services, instance) = loaded
        .create_served_instance(
            device,
            &topology.id,
            topology.device_ordinals.clone(),
            kv.policy,
            speculation_service_v1(speculation),
            Some(maximum_sequences),
            Some(kv.kv_dtype),
            topology.context_parallel_size,
        )
        .map_err(|error| format!("the selected bundle refused this host: {error}"))?;
    Ok((services.clone(), services, instance))
}

#[cfg(not(all(target_os = "linux", feature = "linux-cuda-host")))]
fn open_compiled_instance_v1(
    loaded: &'static InstalledBundleHostV1,
    topology: &ServedTopologyV1,
    kv: ServedKvV1,
    speculation: Option<SpeculationConfig>,
    maximum_sequences: u32,
) -> Result<OpenServedInstanceV1, String> {
    open_host_memory_instance_v1(loaded, topology, kv, speculation, maximum_sequences)
}

/// The one topology a run serves, and the device behind each of its ranks.
#[derive(Clone, Debug)]
struct ServedTopologyV1 {
    id: String,
    device_ordinals: Vec<i32>,
    /// Ranks of one context-parallel group; 1 keeps every KV page on every
    /// rank.
    context_parallel_size: usize,
}

impl ServedTopologyV1 {
    fn video_device(&self) -> i32 {
        self.device_ordinals[0]
    }

    fn new(
        manifest: &BundleManifestV1,
        requested: Option<&str>,
        devices: Option<&[i32]>,
        backend: DeviceServicesV1,
    ) -> Result<Self, HostCommandErrorV1> {
        let topology = resolve_topology_v1(manifest, requested, devices.map(<[i32]>::len))?;
        let selected = Self {
            id: topology.id().as_str().to_owned(),
            context_parallel_size: 1,
            device_ordinals: crate::topology::serving_devices(
                topology.id().as_str(),
                topology.rank_count(),
                devices,
                backend == DeviceServicesV1::Compiled,
            )
            .map_err(run_error)?,
        };
        #[cfg(not(all(target_os = "linux", feature = "linux-cuda-host")))]
        if backend == DeviceServicesV1::Compiled {
            host_memory_fallback_v1(
                manifest
                    .resource_requirements()
                    .minimum_device_memory_bytes(),
            )?;
        }
        #[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
        if backend == DeviceServicesV1::Compiled {
            for warning in crate::topology::memory_warnings(
                &selected.device_ordinals,
                manifest
                    .resource_requirements()
                    .minimum_device_memory_bytes(),
            ) {
                eprintln!("{warning}");
            }
        }
        Ok(selected)
    }

    /// Serves with `size` ranks per context-parallel group, a size that
    /// divides the topology's ranks and that the bundle declares for it.
    fn with_context_parallel(
        mut self,
        manifest: &BundleManifestV1,
        size: usize,
    ) -> Result<Self, HostCommandErrorV1> {
        let topology = resolve_topology_v1(manifest, Some(&self.id), None)?;
        runtime_kv::ContextParallelPagesV1::new(
            size,
            manifest.kv_layout().page_tokens() as usize,
            topology.rank_count() as usize,
        )
        .map_err(|error| usage_error(format!("--context-parallel-size {size}: {error}")))?;
        if !u32::try_from(size).is_ok_and(|size| topology.serves_context_parallel(size)) {
            let declared = std::iter::once(1)
                .chain(topology.context_parallel_sizes().iter().copied())
                .map(|size| size.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(usage_error(format!(
                "--context-parallel-size {size}: the selected bundle declares context-parallel \
                 size(s) {declared} for topology {}",
                self.id
            )));
        }
        self.context_parallel_size = size;
        Ok(self)
    }

    /// The context-parallel size the serving identity reports: where the
    /// topology declares groups, or serves with them; none otherwise.
    fn reported_context_parallel(
        &self,
        manifest: &BundleManifestV1,
    ) -> Result<Option<usize>, HostCommandErrorV1> {
        let topology = resolve_topology_v1(manifest, Some(&self.id), None)?;
        Ok(
            (self.context_parallel_size > 1 || !topology.context_parallel_sizes().is_empty())
                .then_some(self.context_parallel_size),
        )
    }
}

/// A build without the device driver serves `Compiled` runs on its
/// host-memory device services, whose ranks report
/// [`crate::fake::DEFAULT_FAKE_DEVICE_MEMORY_BYTES_V1`] bytes of device
/// memory. A package that needs more cannot be served there: refuse it before
/// any weight loads, naming the missing driver, rather than after a full load
/// with a bare status from the serving KV pool. Smaller packages run on it as
/// before, and the run says which services it uses.
#[cfg(not(all(target_os = "linux", feature = "linux-cuda-host")))]
fn host_memory_fallback_v1(minimum_device_memory_bytes: u64) -> Result<(), HostCommandErrorV1> {
    #[cfg(feature = "fake-host-services")]
    let available = crate::fake::DEFAULT_FAKE_DEVICE_MEMORY_BYTES_V1;
    #[cfg(not(feature = "fake-host-services"))]
    let available = 0;
    if minimum_device_memory_bytes > available {
        return Err(run_error(format!(
            "this chopinfer build links no device driver, so it serves on host-memory device \
             services of {available} bytes per rank, and the package needs at least \
             {minimum_device_memory_bytes} device bytes per rank; build chopinfer with \
             `--features runtime-host/linux-cuda-host` to serve it on GPUs"
        )));
    }
    eprintln!(
        "startup device services: host memory, {available} device bytes per rank (this build \
         links no device driver)"
    );
    Ok(())
}

/// What the transport is allowed to admit, before the bundle narrows it.
fn server_config_v1(
    serving: ServingOptionsV1,
    surface: RequestSurfaceV1,
    envelope: Option<DeveloperServingEnvelopeV1>,
) -> Result<ServerConfig, HostCommandErrorV1> {
    if serving.maximum_sequences == 0 {
        return Err(usage_error("--max-sequences must admit at least one"));
    }
    let defaults = ServerConfig::default();
    // Keep backend staging bounded; the default transport waits for a slot.
    let in_flight = serving
        .maximum_sequences
        .checked_add(serving.queue_depth.unwrap_or(PREPARED_QUEUE_DEPTH_V1))
        .ok_or_else(|| usage_error("--queue-depth exceeds the supported request capacity"))?;
    let config = ServerConfig::new(
        in_flight,
        defaults.event_buffer_capacity(),
        serving
            .maximum_request_body_bytes
            .unwrap_or(defaults.max_body_bytes()),
        defaults.max_output_bytes(),
    )
    .map_err(|error| usage_error(format!("serving request limits are unusable: {error}")))?
    .with_max_string_bytes(
        serving
            .maximum_request_string_bytes
            .unwrap_or(defaults.max_string_bytes()),
    )
    .map_err(|error| usage_error(format!("--max-request-string-bytes is unusable: {error}")))?;
    let config = if serving.queue_depth.is_none() {
        config.with_unbounded_queue()
    } else {
        config
    };
    let config = config
        .with_queue_wait(std::time::Duration::from_millis(serving.queue_wait_ms))
        .with_responses_store(128, serving.responses_store_bytes);
    if let Some(envelope) = envelope {
        // The envelope already says what a request may charge, so a second
        // context limit beside it would be a second answer to that question.
        if serving
            .maximum_context_tokens
            .is_some_and(|maximum| maximum != envelope.total_tokens())
        {
            return Err(usage_error(format!(
                "a developer envelope of {} prompt token(s) and {} generated token(s) charges {} \
                 token(s), and this deployment also asks to serve {} context token(s)",
                envelope.prompt_tokens(),
                envelope.maximum_generated_tokens(),
                envelope.total_tokens(),
                option_tokens_v1(serving.maximum_context_tokens),
            )));
        }
        return (if envelope.is_bounded() {
            config.with_bounded_acceptance_profile(
                envelope.prompt_tokens(),
                envelope.maximum_generated_tokens(),
                envelope.total_tokens(),
            )
        } else {
            config.with_exact_shape_acceptance_profile(
                envelope.prompt_tokens(),
                envelope.maximum_generated_tokens(),
            )
        })
        .map_err(|error| {
            usage_error(format!(
                "the developer serving envelope is unusable: {error}"
            ))
        });
    }
    let config = config.with_request_surface(surface);
    Ok(match serving.maximum_context_tokens {
        Some(maximum) => {
            if maximum == 0 {
                return Err(usage_error("--max-context-tokens must allow at least one"));
            }
            config.with_maximum_context_tokens(maximum)
        }
        None => config,
    })
}

/// Uses the acquired engine capacity for admission after the bundle has created its pool.
fn constrain_server_context_v1(
    config: ServerConfig,
    frontend_tokens: usize,
    engine_tokens: usize,
    requested_limit: bool,
) -> Result<ServerConfig, HostCommandErrorV1> {
    let requested = config.maximum_context_tokens().unwrap_or(frontend_tokens);
    if requested_limit && requested > engine_tokens {
        return Err(run_error(format!(
            "the requested context of {requested} token(s) exceeds the acquired KV capacity of {engine_tokens} token(s)"
        )));
    }
    Ok(config.with_maximum_context_tokens(requested.min(engine_tokens)))
}

/// Names an optional context limit, saying when the bundle's own was taken.
fn option_tokens_v1(value: Option<usize>) -> String {
    value.map_or_else(
        || "the whole declared context".to_owned(),
        |value| value.to_string(),
    )
}

/// Sizes the engine from what the bundle and its declaration allow.
///
/// The page size, the sequence limit, and the vocabulary are the bundle's. The
/// concurrency and the batch budget are the deployment's, and they only narrow
/// what the bundle already declared: options past its declared batch limits
/// are refused.
fn engine_config_v1(
    manifest: &BundleManifestV1,
    vocabulary: ModelVocabulary,
    served_context_tokens: usize,
    serving: ServingOptionsV1,
    plan: Option<DeveloperServingPlanV1>,
) -> Result<EngineConfig, HostCommandErrorV1> {
    let kv_layout = manifest.kv_layout();
    let page_tokens = usize::try_from(kv_layout.page_tokens())
        .map_err(|_| run_error("the declared page token count does not fit this platform"))?;
    // A context-parallel group's logical page is a stripe of one page on each
    // of its ranks (runtime_kv::ContextParallelPagesV1).
    if serving.context_parallel_size != 1 && plan.is_some() {
        return Err(usage_error(
            "a developer serving envelope serves at --context-parallel-size 1",
        ));
    }
    let page_tokens = page_tokens
        .checked_mul(serving.context_parallel_size)
        .ok_or_else(|| usage_error("--context-parallel-size overflows the page token count"))?;
    let declared_tokens = usize::try_from(kv_layout.maximum_sequence_tokens())
        .map_err(|_| run_error("the declared sequence limit does not fit this platform"))?;
    let maximum_sequence_tokens = match plan {
        // A plan is resolved again against this manifest before it sizes
        // anything, so the rows and pages it names are this bundle's. The one
        // token it exceeds them by is the last generated token, which is
        // emitted from a row rather than written to one.
        Some(plan) => revalidated_plan_v1(manifest, served_context_tokens, plan)?.total_tokens(),
        // A deployment serves the shorter of what the bundle can hold and what
        // its declaration offers, never more than either.
        None => declared_tokens.min(served_context_tokens),
    };
    if maximum_sequence_tokens == 0 {
        return Err(run_error(
            "the selected bundle declares no sequence tokens to serve",
        ));
    }
    // A batch holds at most one item per live sequence, and its tokens never
    // exceed its token budget (a media encode may overrun the budget, its
    // prompt tokens may not), so these two bounds keep every batch the
    // scheduler forms within what the bundle executes.
    let limits = manifest.batch_limits();
    let prefill_limit = limits.map_or(usize::MAX, |limits| {
        usize::try_from(limits.maximum_prefill_tokens()).unwrap_or(usize::MAX)
    });
    if let Some(limits) = limits
        && u64::try_from(serving.maximum_sequences).unwrap_or(u64::MAX)
            > u64::from(limits.maximum_sequences())
    {
        return Err(usage_error(format!(
            "--max-sequences {} exceeds the {} sequence(s) the selected bundle runs at once",
            serving.maximum_sequences,
            limits.maximum_sequences()
        )));
    }
    if let Some(requested) = serving.maximum_batch_tokens
        && requested > prefill_limit
    {
        return Err(usage_error(format!(
            "--max-batch-tokens {requested} exceeds the {prefill_limit} prompt token(s) the selected bundle prefills per batch"
        )));
    }
    let batch_tokens = serving
        .maximum_batch_tokens
        .unwrap_or_else(|| {
            DEFAULT_SERVED_BATCH_TOKENS_V1
                .min(maximum_sequence_tokens)
                .min(prefill_limit)
        })
        .min(maximum_sequence_tokens);
    if batch_tokens == 0 {
        return Err(usage_error("--max-batch-tokens must schedule at least one"));
    }
    let prefill_chunk_tokens = match serving.prefill_chunk_tokens {
        Some(chunk) if chunk > batch_tokens => {
            return Err(usage_error(
                "--prefill-chunk-tokens exceeds the effective batch budget",
            ));
        }
        Some(chunk) => chunk,
        None => DEFAULT_PREFILL_CHUNK_TOKENS_V1.min(batch_tokens),
    };
    let pages_per_sequence = plan.map_or_else(
        || maximum_sequence_tokens.div_ceil(page_tokens),
        DeveloperServingPlanV1::pages_per_sequence,
    );
    let pages = pages_per_sequence
        .checked_mul(serving.maximum_sequences)
        .ok_or_else(|| {
            usage_error("--max-sequences asks for more pages than this host can hold")
        })?;
    let geometry = KvGeometry::new(page_tokens, pages)
        .map_err(|error| run_error(format!("the served page pool is unusable: {error}")))?;
    let in_flight = serving
        .maximum_sequences
        .checked_add(serving.queue_depth.unwrap_or(PREPARED_QUEUE_DEPTH_V1))
        .ok_or_else(|| usage_error("--queue-depth exceeds the supported request capacity"))?;
    let tracking_capacity = in_flight
        .checked_add(serving.maximum_sequences)
        .ok_or_else(|| usage_error("--queue-depth exceeds the supported tracking capacity"))?;
    let scheduler = SchedulerConfig::new(
        serving.maximum_sequences,
        tracking_capacity,
        batch_tokens,
        prefill_chunk_tokens,
        1,
    )
    // A legacy bundle's configured chunk may be its only capacity bound.
    // Expand only after an explicit manifest limit has bounded batch_tokens.
    .map(|config| config.with_lone_prefill_expansion(limits.is_some()))
    .and_then(|config| {
        config.with_maximum_scored_rows(limits.map_or(
            runtime_bundle::DEFAULT_MAXIMUM_SCORED_ROWS,
            runtime_bundle::BundleBatchLimitsV1::maximum_scored_rows,
        ) as usize)
    })
    .and_then(|config| {
        config.with_prefill_granule(
            limits.map_or(1, runtime_bundle::BundleBatchLimitsV1::prefill_granule) as usize,
        )
    })
    .map(|config| {
        config.with_last_page_rewrites(
            limits.is_some_and(runtime_bundle::BundleBatchLimitsV1::rewrites_last_page),
        )
    })
    .map(|config| {
        config.with_anchor_items(
            limits.map_or(0, runtime_bundle::BundleBatchLimitsV1::anchor_items) as usize,
        )
    })
    .and_then(|config| {
        config.with_capacity_profile(SchedulerCapacityProfile::new("served", in_flight)?)
    })
    .map_err(|error| run_error(format!("the served schedule is unusable: {error}")))?
    .with_admission(serving.admission)
    .with_prefill_budget(serving.prefill_budget);
    EngineConfig::new(
        scheduler,
        geometry,
        vocabulary,
        maximum_sequence_tokens,
        serving.maximum_sequences,
    )
    .map_err(|error| run_error(format!("the serving engine is unusable: {error:?}")))
    .and_then(|config| {
        batch_refusal_limits_v1(|name| {
            std::env::var_os(name).map(|value| value.to_string_lossy().into_owned())
        })
        .map(|limits| {
            let config = config.with_batch_refusal_limits(limits);
            if serving.debug_swap {
                config.with_forced_swaps()
            } else {
                config
            }
        })
    })
}

/// The generation worker's queue for a served deployment.
///
/// Each submitted choice holds an in-flight permit. Extra HTTP requests wait
/// outside this bounded staging channel unless the operator bounded the queue.
fn serving_backend_config_v1(
    serving: &ServingOptionsV1,
) -> Result<EngineBackendConfig, HostCommandErrorV1> {
    let in_flight = serving
        .maximum_sequences
        .saturating_add(serving.queue_depth.unwrap_or(PREPARED_QUEUE_DEPTH_V1));
    let step_overlap = crate::overlap::step_overlap_from_environment();
    if step_overlap {
        eprintln!(
            "step overlap on ({}=1): each step's events are delivered while the device runs the next step",
            crate::overlap::STEP_OVERLAP
        );
    }
    Ok(EngineBackendConfig::new(in_flight)
        .map_err(|error| run_error(format!("the generation worker is unusable: {error}")))?
        .with_queue_wait(std::time::Duration::from_millis(serving.queue_wait_ms))
        .with_step_overlap(step_overlap)
        // A worker stuck in a step fails readiness and exits the process for
        // its supervisor rather than staying ready.
        .with_watchdog(Some(crate::WorkerWatchdogV1::serving())))
}

fn batch_refusal_limits_v1(
    read: impl Fn(&str) -> Option<String>,
) -> Result<runtime_engine::BatchRefusalLimits, HostCommandErrorV1> {
    let mut limits = runtime_engine::BatchRefusalLimits::DEFAULT;
    for (name, limit, maximum, minimum) in [
        (
            "CHOPINFER_BATCH_REFUSAL_MAX_CONSECUTIVE",
            &mut limits.maximum_consecutive,
            10_000,
            0,
        ),
        (
            "CHOPINFER_BATCH_REFUSAL_MAX_IN_WINDOW",
            &mut limits.maximum_in_window,
            10_000,
            0,
        ),
        (
            "CHOPINFER_BATCH_REFUSAL_WINDOW_SECONDS",
            &mut limits.window_seconds,
            86_400,
            1,
        ),
    ] {
        if let Some(value) = read(name) {
            *limit = value
                .parse::<u32>()
                .ok()
                .filter(|value| (minimum..=maximum).contains(value))
                .ok_or_else(|| {
                    usage_error(format!(
                        "{name} must be an integer from {minimum} through {maximum}"
                    ))
                })?;
        }
    }
    Ok(limits)
}

/// Loads the one bundle this process runs, then drives one scenario through it.
///
/// This is the conformance mode, not the serving command: it exits when the
/// named scenario has finalized its batch and released the instance, and it
/// binds nothing.
fn conformance_run_v1(
    selection: &BundleSelectionV1,
    host: &BundleHostCompatibilityV1,
    topology: Option<&str>,
    device: DeviceServicesV1,
    request: &FixtureRequestV1,
    kv_memory_fraction_millionths: u32,
) -> Result<FixtureRunReceiptV1, HostCommandErrorV1> {
    let package = select_topology_v1(verify_selected_bundle_v1(selection, host)?, topology, None)?;
    let loaded = load_verified_process_bundle_v1(selection, host, package, None)?;
    let mut plan = FixturePlanV1::new(loaded.manifest(), host, topology, request)?;
    plan.kv_memory_fraction_millionths = kv_memory_fraction_millionths;
    match device {
        DeviceServicesV1::HostMemory => run_host_memory_device_v1(loaded, &plan),
        DeviceServicesV1::Compiled => run_compiled_device_v1(loaded, &plan),
    }
}

#[cfg(feature = "fake-host-services")]
fn run_host_memory_device_v1(
    loaded: &'static InstalledBundleHostV1,
    plan: &FixturePlanV1,
) -> Result<FixtureRunReceiptV1, HostCommandErrorV1> {
    run_over_device_v1(
        crate::fake::FakeHostDeviceV1::new(),
        HOST_MEMORY_SERVICES_V1,
        loaded,
        plan,
    )
}

#[cfg(not(feature = "fake-host-services"))]
fn run_host_memory_device_v1(
    _loaded: &'static InstalledBundleHostV1,
    _plan: &FixturePlanV1,
) -> Result<FixtureRunReceiptV1, HostCommandErrorV1> {
    Err(HostCommandErrorV1::Run(
        NO_HOST_MEMORY_SERVICES_V1.to_owned(),
    ))
}

#[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
fn run_compiled_device_v1(
    loaded: &'static InstalledBundleHostV1,
    plan: &FixturePlanV1,
) -> Result<FixtureRunReceiptV1, HostCommandErrorV1> {
    let device = crate::cuda::CudaHostDeviceV1::new().map_err(|error| {
        HostCommandErrorV1::Run(format!("cannot open the device driver: {error}"))
    })?;
    run_over_device_v1(device, DRIVER_SERVICES_V1, loaded, plan)
}

/// A build with no driver has one kind of device services, and says so.
///
/// The receipt names the services every run used, so a run against host memory
/// is never read as a run against a device.
#[cfg(not(all(target_os = "linux", feature = "linux-cuda-host")))]
fn run_compiled_device_v1(
    loaded: &'static InstalledBundleHostV1,
    plan: &FixturePlanV1,
) -> Result<FixtureRunReceiptV1, HostCommandErrorV1> {
    run_host_memory_device_v1(loaded, plan)
}

struct FixtureExecutionV1<D: HostDeviceBackendV1> {
    backend: Rc<HostServicesV1Backend<D>>,
    instance: LoadedBundleInstanceV1,
}

struct FixtureResultSinkV1<D: HostDeviceBackendV1> {
    backend: Rc<HostServicesV1Backend<D>>,
    limits: ResultSinkLimitsV1,
    rows: Vec<OwnedLogitsRowV1>,
}

impl<D: HostDeviceBackendV1> LogitsRowSinkV1 for FixtureResultSinkV1<D> {
    fn limits(&self) -> ResultSinkLimitsV1 {
        self.limits
    }

    fn emit_row(&mut self, row: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        self.rows.push(row);
        Ok(())
    }

    fn emit_view(&mut self, view: OwnedLogitsViewV1) -> HostServiceResultV1<()> {
        let raw = view.raw();
        if raw.rank != 0
            || raw.first_item_index != 0
            || raw.row_count != 1
            || u64::from(raw.live_width) > self.limits.maximum_logits_per_row
            || !self.rows.is_empty()
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let bytes = self.backend.read_logits_view_row(view, 0)?;
        let logit_bits = if raw.element_type == BUNDLE_LOGITS_ELEMENT_V1_BF16 {
            bytes
                .chunks_exact(2)
                .flat_map(|value| {
                    (u32::from(u16::from_le_bytes([value[0], value[1]])) << 16).to_le_bytes()
                })
                .collect()
        } else {
            bytes
        };
        self.rows
            .push(OwnedLogitsRowV1::new(FIXTURE_REQUEST_ID_V1, logit_bits));
        Ok(())
    }
}

impl<D: HostDeviceBackendV1> BundleExecutionV1 for FixtureExecutionV1<D> {
    fn execute_batch(
        &mut self,
        request: &[u8],
        limits: ResultSinkLimitsV1,
    ) -> Result<Vec<OwnedLogitsRowV1>, BundleAbiError> {
        let mut sink = FixtureResultSinkV1 {
            backend: Rc::clone(&self.backend),
            limits,
            rows: Vec::new(),
        };
        self.instance.execute(request, &mut sink)?;
        Ok(sink.rows)
    }

    fn commit(&mut self) -> Result<(), BundleAbiError> {
        self.instance.commit()
    }

    fn cancel(&mut self) -> Result<(), BundleAbiError> {
        self.instance.cancel()
    }

    fn destroy(&mut self) -> Result<(), BundleAbiError> {
        self.instance.destroy()
    }

    fn is_poisoned(&self) -> bool {
        self.instance.is_poisoned()
    }
}

/// Creates one instance, drives the named scenario, and releases everything.
///
/// The backend outlives the instance on purpose: it is the host's own record of
/// what the bundle borrowed, so reading it after the instance is gone is what
/// makes the cleanup a result rather than an assumption.
fn run_over_device_v1<D: HostDeviceBackendV1 + 'static>(
    device: D,
    device_services: &'static str,
    loaded: &'static InstalledBundleHostV1,
    plan: &FixturePlanV1,
) -> Result<FixtureRunReceiptV1, HostCommandErrorV1> {
    let serving_kv = loaded
        .manifest()
        .resource_requirements()
        .required_capabilities()
        .iter()
        .any(|capability| capability.as_str() == runtime_bundle_abi::SERVING_KV_CAPABILITY_V1);
    let create_started = std::time::Instant::now();
    let (backend, instance) = loaded
        .create_instance_with_serving_kv(
            device,
            &plan.topology_id,
            plan.device_ordinals()?,
            serving_kv.then_some(ServingKvPolicyV1 {
                sampling_vocabulary: None,
                maximum_context_tokens: None,
                state_reserved_bytes: 0,
                fraction_millionths: plan.kv_memory_fraction_millionths,
                maximum_pages: None,
            }),
        )
        .map_err(HostCommandErrorV1::Load)?;
    eprintln!(
        "startup bundle create: {:.3}s",
        create_started.elapsed().as_secs_f64()
    );
    if serving_kv && backend.serving_kv_pages().is_none() {
        return Err(run_error(
            "the conformance bundle did not acquire its serving KV pool during create",
        ));
    }
    if serving_kv {
        let pages = backend.serving_kv_pages().expect("checked above");
        if pages
            < plan
                .prefill_tokens
                .div_ceil(loaded.manifest().kv_layout().page_tokens() as usize) as u64
        {
            return Err(run_error(
                "the conformance batch exceeds the serving KV pool",
            ));
        }
        backend
            .prepare_serving_batch(&plan.kv_plan)
            .map_err(|status| {
                run_error(format!(
                    "the conformance KV pages could not be prepared: {status:?}"
                ))
            })?;
    }
    let execution = FixtureExecutionV1 {
        backend: Rc::clone(&backend),
        instance,
    };
    let report = drive_fixture_scenario_v1(execution, &plan.request, plan.scenario, || {
        backend.live_resource_count()
    })?;
    Ok(FixtureRunReceiptV1 {
        bundle_id: loaded.manifest().id().as_str().to_owned(),
        topology_id: plan.topology_id.clone(),
        rank_count: plan.rank_count,
        device_services,
        architecture: plan.architecture.clone(),
        abi_version: MODEL_BUNDLE_V1_ABI_VERSION,
        label: plan.label.clone(),
        scenario: plan.scenario,
        prefill_tokens: plan.prefill_tokens,
        executions: report.executions,
        rows: report.rows,
        logits_per_row: report.logits_per_row,
        live_resources: report.live_resources,
    })
}

/// Drives one scenario over one bundle instance, and gives everything back.
///
/// The instance is taken by value because letting it go is part of two of the
/// three scenarios: `drop` leaves an execution pending and relies on the ABI
/// wrapper to cancel and release it, and the other two destroy it first. Either
/// way nothing holds a resource when `live_resources` is read, so the count it
/// returns is the state after cleanup rather than a snapshot taken during it.
///
/// Nothing here is model-specific. The batch is the same bytes each time, the
/// only comparison is between two rows this same bundle emitted, and the only
/// number checked against zero is this host's own count of what it lent out.
pub fn drive_fixture_scenario_v1<E: BundleExecutionV1>(
    mut instance: E,
    request: &[u8],
    scenario: FixtureScenarioV1,
    live_resources: impl Fn() -> usize,
) -> Result<FixtureScenarioReportV1, HostCommandErrorV1> {
    // One row is scheduled per execution, so one row is all each one collects.
    let limits = ResultSinkLimitsV1::new(
        1,
        MAX_FIXTURE_LOGITS_PER_ROW_V1,
        MAX_FIXTURE_LOGITS_PER_ROW_V1,
    );
    let executed = execute_fixture_batch_v1(&mut instance, request, limits)?;
    let logits_per_row = match fixture_logits_per_row_v1(&executed) {
        Ok(logits_per_row) => logits_per_row,
        Err(refusal) => return Err(cancel_refused_run_v1(&mut instance, refusal)),
    };
    let mut rows = executed.len();
    let mut executions = 1;
    match scenario {
        FixtureScenarioV1::Commit => finalize_fixture_batch_v1(&mut instance, true)?,
        FixtureScenarioV1::Cancel => {
            finalize_fixture_batch_v1(&mut instance, false)?;
            let replayed = execute_fixture_batch_v1(&mut instance, request, limits)?;
            rows += replayed.len();
            executions += 1;
            if let Err(refusal) = same_fixture_rows_v1(&executed, &replayed) {
                return Err(cancel_refused_run_v1(&mut instance, refusal));
            }
            // The replay is finalized the same way, so this scenario leaves
            // nothing pending and nothing committed.
            finalize_fixture_batch_v1(&mut instance, false)?;
        }
        FixtureScenarioV1::Drop => {
            // Nothing finalizes this execution. Letting the instance go is the
            // scenario, and what the wrapper does with it is the result.
            drop(instance);
            return released_fixture_instance_v1(
                scenario,
                executions,
                rows,
                logits_per_row,
                live_resources(),
            );
        }
    }
    instance.destroy().map_err(|error| {
        run_error(format!(
            "the bundle could not release the fixture instance: {error}"
        ))
    })?;
    drop(instance);
    released_fixture_instance_v1(scenario, executions, rows, logits_per_row, live_resources())
}

fn execute_fixture_batch_v1<E: BundleExecutionV1>(
    instance: &mut E,
    request: &[u8],
    limits: ResultSinkLimitsV1,
) -> Result<Vec<OwnedLogitsRowV1>, HostCommandErrorV1> {
    instance.execute_batch(request, limits).map_err(|error| {
        HostCommandErrorV1::Run(format!("the bundle refused the fixture batch: {error}"))
    })
}

fn finalize_fixture_batch_v1<E: BundleExecutionV1>(
    instance: &mut E,
    commit: bool,
) -> Result<(), HostCommandErrorV1> {
    let (result, verb) = if commit {
        (instance.commit(), "commit")
    } else {
        (instance.cancel(), "cancel")
    };
    result.map_err(|error| {
        run_error(format!(
            "the bundle could not {verb} the fixture batch: {error}"
        ))
    })
}

/// A cancelled execution has to leave the bundle where it found it.
///
/// The replayed batch is byte for byte the cancelled one, so a bundle that
/// rolled that execution back emits the row it emitted before. One that kept
/// part of it emits something else. Neither claim needs this host to know what
/// a single one of those bytes means.
fn same_fixture_rows_v1(
    cancelled: &[OwnedLogitsRowV1],
    replayed: &[OwnedLogitsRowV1],
) -> Result<(), String> {
    if cancelled.len() != replayed.len() {
        return Err(format!(
            "replaying a cancelled batch emitted {} rows where the cancelled one emitted {}, so \
             the cancel did not roll that execution back",
            replayed.len(),
            cancelled.len()
        ));
    }
    for (index, (before, after)) in cancelled.iter().zip(replayed).enumerate() {
        if before.request_id() != after.request_id() || before.logit_bits() != after.logit_bits() {
            return Err(format!(
                "row {index} of the replayed batch is not the row the cancelled batch emitted, so \
                 the cancel did not roll that execution back"
            ));
        }
    }
    Ok(())
}

/// Reports one finished scenario, once this host owns nothing on its behalf.
fn released_fixture_instance_v1(
    scenario: FixtureScenarioV1,
    executions: usize,
    rows: usize,
    logits_per_row: u64,
    live_resources: usize,
) -> Result<FixtureScenarioReportV1, HostCommandErrorV1> {
    if live_resources != 0 {
        return Err(run_error(format!(
            "the {} scenario ended with {live_resources} host resource(s) the bundle never gave \
             back",
            scenario.as_str()
        )));
    }
    Ok(FixtureScenarioReportV1 {
        executions,
        rows,
        logits_per_row,
        live_resources,
    })
}

/// Checks the one row this batch scheduled, without knowing a vocabulary.
///
/// The width is the bundle's to declare, so this only refuses a row nothing
/// could read: a wrong count, a row for another request, or a byte length that
/// is not whole 32-bit logits.
fn fixture_logits_per_row_v1(rows: &[OwnedLogitsRowV1]) -> Result<u64, String> {
    let [row] = rows else {
        return Err(format!(
            "the bundle emitted {} logits rows for one scheduled output",
            rows.len()
        ));
    };
    if row.request_id() != FIXTURE_REQUEST_ID_V1 {
        return Err(format!(
            "the bundle emitted a row for request {} where request {FIXTURE_REQUEST_ID_V1} was \
             scheduled",
            row.request_id()
        ));
    }
    let bits = row.logit_bits().len();
    if bits == 0 || !bits.is_multiple_of(4) {
        return Err(format!(
            "the bundle emitted {bits} logit bytes, which are not whole 32-bit logits"
        ));
    }
    Ok(bits as u64 / 4)
}

fn cancel_refused_run_v1(
    instance: &mut impl BundleExecutionV1,
    refusal: String,
) -> HostCommandErrorV1 {
    match instance.cancel() {
        Ok(()) => HostCommandErrorV1::Run(refusal),
        Err(error) => HostCommandErrorV1::Run(format!(
            "{refusal}; cancelling that execution also failed: {error}"
        )),
    }
}

/// One bounded fixture batch, encoded once for whichever device runs it.
#[derive(Debug)]
struct FixturePlanV1 {
    label: String,
    topology_id: String,
    rank_count: u32,
    /// The architecture this host answered the bundle's target with, carried so
    /// the receipt says which host ran the batch.
    architecture: String,
    scenario: FixtureScenarioV1,
    prefill_tokens: usize,
    request: Vec<u8>,
    kv_plan: PreparedAppendBatch,
    kv_memory_fraction_millionths: u32,
}

impl FixturePlanV1 {
    /// Builds the batch out of the manifest the selection verified.
    ///
    /// The token count is the bundle's own declared page unless the caller
    /// named one, and the page plan is prepared through the host's logical KV
    /// allocator so the encoded request carries a real page table rather than
    /// an invented one. That pool is scaffolding for this one batch: a served
    /// deployment keeps its pool for the engine's lifetime instead.
    ///
    /// One plan serves every scenario, because the batch a run cancels has to
    /// be the batch it would otherwise have committed.
    fn new(
        manifest: &BundleManifestV1,
        host: &BundleHostCompatibilityV1,
        topology: Option<&str>,
        request: &FixtureRequestV1,
    ) -> Result<Self, HostCommandErrorV1> {
        let topology = resolve_topology_v1(manifest, topology, None)?;
        let kv_layout = manifest.kv_layout();
        let page_tokens = usize::try_from(kv_layout.page_tokens())
            .map_err(|_| run_error("the declared page token count does not fit this platform"))?;
        let prefill_tokens = request.prefill_tokens.unwrap_or(page_tokens);
        if prefill_tokens == 0 {
            return Err(usage_error("--fixture-tokens must be at least one token"));
        }
        if prefill_tokens as u64 > kv_layout.maximum_sequence_tokens() {
            return Err(usage_error(format!(
                "the selected bundle declares at most {} sequence tokens, {prefill_tokens} were \
                 requested",
                kv_layout.maximum_sequence_tokens()
            )));
        }
        let geometry = KvGeometry::new(page_tokens, prefill_tokens.div_ceil(page_tokens))
            .map_err(|error| run_error(format!("the declared page layout is unusable: {error}")))?;
        let mut pool = KvPool::new(geometry, 1)
            .map_err(|error| run_error(format!("the fixture page pool is unusable: {error}")))?;
        let sequence = pool
            .create_sequence(prefill_tokens)
            .map_err(|error| run_error(format!("the fixture sequence does not fit: {error}")))?;
        let kv_plan = pool
            .prepare_append_batch(&[(sequence, prefill_tokens)])
            .map_err(|error| run_error(format!("the fixture append does not fit: {error}")))?;
        let items = vec![ModelBatchItem::new(
            RequestId::new(FIXTURE_REQUEST_ID_V1),
            BatchPhase::Prefill {
                tokens: prefill_tokens,
            },
            vec![0; prefill_tokens],
            true,
        )];
        let request_bytes = encode_batch_request_with_capability_v1(
            &items,
            &kv_plan,
            batch_wire_capability_v1(manifest),
        )
        .map_err(|error| run_error(error.message().to_owned()))?;
        Ok(Self {
            label: request.label.clone(),
            topology_id: topology.id().as_str().to_owned(),
            rank_count: topology.rank_count(),
            architecture: host.architecture().as_str().to_owned(),
            scenario: request.scenario,
            prefill_tokens,
            request: request_bytes,
            kv_plan,
            kv_memory_fraction_millionths: 900_000,
        })
    }

    /// One device per rank, in rank order.
    fn device_ordinals(&self) -> Result<Vec<i32>, HostCommandErrorV1> {
        (0..self.rank_count)
            .map(|rank| {
                i32::try_from(rank).map_err(|_| run_error("the declared rank count is too large"))
            })
            .collect()
    }
}

/// Picks the one topology this run serves: the one `--topology` names, the
/// only one declared, or the one with a rank per `--devices` entry.
fn resolve_topology_v1<'a>(
    manifest: &'a BundleManifestV1,
    requested: Option<&str>,
    ranks: Option<usize>,
) -> Result<&'a BundleTopologyV1, HostCommandErrorV1> {
    let declared = manifest.target().topologies();
    match requested {
        Some(requested) => declared
            .iter()
            .find(|topology| topology.id().as_str() == requested)
            .ok_or_else(|| {
                usage_error(format!(
                    "the selected bundle declares no topology '{requested}'; it declares {}",
                    declared_topologies_v1(declared)
                ))
            }),
        None => match declared {
            [only] => Ok(only),
            _ => {
                let mut matching = declared.iter().filter(|topology| {
                    ranks.is_some_and(|ranks| topology.rank_count() as usize == ranks)
                });
                match (matching.next(), matching.next()) {
                    (Some(topology), None) => Ok(topology),
                    _ => Err(usage_error(format!(
                        "the selected bundle declares {} topologies, so --topology ID or one \
                         --devices entry per rank must name one: {}",
                        declared.len(),
                        declared_topologies_v1(declared)
                    ))),
                }
            }
        },
    }
}

/// The verified `package` as it serves the topology [`resolve_topology_v1`]
/// picks, with that topology's resources.
fn select_topology_v1(
    package: BundlePackageV1,
    requested: Option<&str>,
    ranks: Option<usize>,
) -> Result<BundlePackageV1, HostCommandErrorV1> {
    let topology = resolve_topology_v1(package.manifest(), requested, ranks)?
        .id()
        .as_str()
        .to_owned();
    package.for_topology(&topology).map_err(|error| {
        run_error(format!(
            "the selected bundle cannot serve {topology}: {error}"
        ))
    })
}

fn declared_topologies_v1(topologies: &[BundleTopologyV1]) -> String {
    topologies
        .iter()
        .map(|topology| topology.id().as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Every flag either front end accepts, read once each.
///
/// Parsing is shared, and each command then refuses the flags it does not
/// accept, so an argument that belongs to another command is a refusal rather
/// than something quietly ignored.
#[derive(Debug, Default, Eq, PartialEq)]
struct ArgumentsV1 {
    model_options: std::collections::BTreeMap<String, String>,
    list_model_options: bool,
    http_options: runtime_server::HttpOptions,
    http_flags: Vec<String>,
    devices: Option<Vec<i32>>,
    bundle: Option<String>,
    model: Option<PathBuf>,
    tensor_parallel_size: Option<usize>,
    model_cache_dir: Option<PathBuf>,
    activation_policy: Option<String>,
    modalities: Option<runtime_bundle::preparation::ModalitySelection>,
    architecture: Option<String>,
    capabilities: Vec<String>,
    topology: Option<String>,
    listen: Option<String>,
    maximum_context_tokens: Option<usize>,
    kv_memory_fraction_millionths: Option<u32>,
    kv_pages: Option<usize>,
    context_parallel_size: Option<usize>,
    kv_cache_dtype: Option<KvCacheDtypeV1>,
    speculative_tokens: Option<usize>,
    speculative_rows: Option<usize>,
    speculation_mode: Option<SpeculationMode>,
    draft_cost_model: Option<DraftCostModel>,
    prefix_cache: Option<bool>,
    kv_offload_gib: Option<u64>,
    kv_offload_min_tokens: Option<usize>,
    kv_offload_rate: Option<u64>,
    kv_offload_prefix: Option<bool>,
    verify_warmup: Option<bool>,
    optimistic_admission: Option<bool>,
    prefill_budget: Option<runtime_scheduler::PrefillBudget>,
    prefill_itl_target_ms: Option<usize>,
    prefill_decode_steps: Option<usize>,
    decode_tbt_target_ms: Option<usize>,
    mixed_min_prefill_tokens: Option<usize>,
    mixed_decode_steps: Option<usize>,
    admission_watermark_millionths: Option<u32>,
    maximum_preemptions: Option<usize>,
    debug_preempt_every: Option<usize>,
    debug_swap_every: Option<usize>,
    maximum_sequences: Option<usize>,
    maximum_batch_tokens: Option<usize>,
    prefill_chunk_tokens: Option<usize>,
    queue_depth: Option<usize>,
    queue_wait_ms: Option<u64>,
    drain_timeout_ms: Option<u64>,
    maximum_request_body_bytes: Option<usize>,
    maximum_request_string_bytes: Option<usize>,
    responses_store_bytes: Option<usize>,
    maximum_tokenizer_input_bytes: Option<usize>,
    maximum_images_per_request: Option<usize>,
    allowed_local_media_path: Option<PathBuf>,
    remote_media: runtime_media::remote::RemoteMediaPolicy,
    remote_media_flags: Vec<&'static str>,
    debug_reference_tokens: Option<PathBuf>,
    direct_checkpoint: Option<PathBuf>,
    load_threads: Option<usize>,
    fixture_request: Option<String>,
    fixture_tokens: Option<usize>,
    scenario: Option<FixtureScenarioV1>,
    host_memory_device: bool,
}

pub(crate) const VALUELESS_FLAGS_V1: &[&str] = &[
    "--enable-request-id-headers",
    "--allow-credentials",
    "--fake-host-services",
    "--list-model-options",
];

/// Removed flags, refused with what replaced them.
const REMOVED_FLAGS_V1: &[(&str, &str)] = &[
    (
        "--fold-ranks",
        "--fold-ranks is gone: rank folding is no longer supported; use one distinct device per rank",
    ),
    (
        "--install-root",
        "--install-root is gone: select a package directory with --bundle DIR",
    ),
    (
        "--root",
        "--root is gone: bundle install no longer exists; select a package directory with --bundle DIR",
    ),
    (
        "--source",
        "--source is gone: bundle install no longer exists; select a package directory with --bundle DIR",
    ),
    (
        "--copy-fallback",
        "--copy-fallback is gone: bundle install no longer exists",
    ),
    (
        "--load",
        "--load is gone: packages carry no rank images; --checkpoint DIR loads the checkpoint through the bundle's weight plan",
    ),
    (
        "--video-input",
        "--video-input is gone: --modalities selects video, which auto leaves out; --modalities all or a list naming video serves it",
    ),
    (
        "--refresh-model",
        "--refresh-model is gone: --model loads the checkpoint directly and reads no tensor contents to refresh; check its contents with chopinfer verify --model DIR",
    ),
];

impl ArgumentsV1 {
    fn parse(arguments: &[String]) -> Result<Self, HostCommandErrorV1> {
        let mut parsed = Self::default();
        let mut remaining = arguments.iter().peekable();
        while let Some(flag) = remaining.next() {
            if let Some((_, message)) = REMOVED_FLAGS_V1.iter().find(|(name, _)| name == flag) {
                return Err(usage_error(*message));
            }
            if VALUELESS_FLAGS_V1.contains(&flag.as_str()) {
                let slot = match flag.as_str() {
                    "--enable-request-id-headers" => {
                        &mut parsed.http_options.enable_request_id_headers
                    }
                    "--allow-credentials" => &mut parsed.http_options.allow_credentials,
                    "--fake-host-services" => &mut parsed.host_memory_device,
                    "--list-model-options" => &mut parsed.list_model_options,
                    _ => unreachable!("the valueless flag table and parser must agree"),
                };
                if *slot {
                    return Err(given_more_than_once_v1(flag));
                }
                *slot = true;
                if flag == "--allow-credentials" {
                    parsed.http_options.enable_cors = true;
                }
                if matches!(
                    flag.as_str(),
                    "--enable-request-id-headers" | "--allow-credentials"
                ) {
                    parsed.http_flags.push(flag.clone());
                }
                continue;
            }
            let value = remaining
                .next()
                .ok_or_else(|| usage_error(format!("{flag} requires a value")))?;
            let remote_flag = match flag.as_str() {
                "--allowed-media-domains" => Some("--allowed-media-domains"),
                "--media-allow-private-addresses" => Some("--media-allow-private-addresses"),
                "--media-request-timeout-ms" => Some("--media-request-timeout-ms"),
                "--media-fetch-timeout-ms" => Some("--media-fetch-timeout-ms"),
                "--media-connect-timeout-ms" => Some("--media-connect-timeout-ms"),
                "--media-max-redirects" => Some("--media-max-redirects"),
                _ => None,
            };
            if let Some(flag) = remote_flag {
                if flag != "--allowed-media-domains" && parsed.remote_media_flags.contains(&flag) {
                    return Err(given_more_than_once_v1(flag));
                }
                parsed.remote_media_flags.push(flag);
            }
            match flag.as_str() {
                "--served-model-name" => {
                    if value.starts_with("--") {
                        return Err(usage_error(
                            "--served-model-name requires at least one name",
                        ));
                    }
                    parsed.http_options.served_model_names.push(value.clone());
                    while remaining.peek().is_some_and(|name| !name.starts_with("--")) {
                        parsed
                            .http_options
                            .served_model_names
                            .push(remaining.next().expect("peeked name").clone());
                    }
                    parsed.http_flags.push(flag.clone());
                }
                "--model-option" => {
                    let (name, value) = value
                        .split_once('=')
                        .ok_or_else(|| usage_error("--model-option requires NAME=VALUE"))?;
                    if !runtime_bundle::model_options::valid_name(name) || value.len() > 1024 {
                        return Err(usage_error("invalid model option name or value"));
                    }
                    if parsed
                        .model_options
                        .insert(name.into(), value.into())
                        .is_some()
                    {
                        return Err(usage_error(format!(
                            "model option '{name}' given more than once"
                        )));
                    }
                }
                "--api-key" => {
                    if value.is_empty() {
                        return Err(usage_error("--api-key must not be empty"));
                    }
                    set_once_v1(
                        &mut parsed.http_options.api_key,
                        runtime_server::ApiKey::new(value),
                        flag,
                    )?;
                    parsed.http_flags.push(flag.clone());
                }
                "--allowed-origins" | "--allowed-methods" | "--allowed-headers" => {
                    parsed.http_options.enable_cors = true;
                    if parsed.http_flags.contains(flag) {
                        return Err(given_more_than_once_v1(flag));
                    }
                    parsed.http_flags.push(flag.clone());
                    let values: Vec<String> = serde_json::from_str(value).map_err(|_| {
                        usage_error(format!("{flag} requires a JSON array of strings"))
                    })?;
                    match flag.as_str() {
                        "--allowed-origins" => parsed.http_options.allowed_origins = values,
                        "--allowed-methods" => parsed.http_options.allowed_methods = values,
                        _ => parsed.http_options.allowed_headers = values,
                    }
                }
                "--devices" => set_once_v1(
                    &mut parsed.devices,
                    crate::topology::parse_devices(value).map_err(usage_error)?,
                    flag,
                )?,
                "--bundle" => set_once_v1(&mut parsed.bundle, value.clone(), flag)?,
                "--model" => set_once_v1(&mut parsed.model, PathBuf::from(value), flag)?,
                "--activation-policy" => {
                    set_once_v1(&mut parsed.activation_policy, value.clone(), flag)?
                }
                "--modalities" => set_once_v1(
                    &mut parsed.modalities,
                    runtime_bundle::preparation::ModalitySelection::parse(value)
                        .map_err(usage_error)?,
                    flag,
                )?,
                "--model-cache-dir" => {
                    set_once_v1(&mut parsed.model_cache_dir, PathBuf::from(value), flag)?
                }
                "--tensor-parallel-size" => set_once_v1(
                    &mut parsed.tensor_parallel_size,
                    count_v1(flag, value)?,
                    flag,
                )?,
                "--architecture" => set_once_v1(&mut parsed.architecture, value.clone(), flag)?,
                "--capability" => {
                    if parsed.capabilities.contains(value) {
                        return Err(given_more_than_once_v1(&format!("--capability {value}")));
                    }
                    parsed.capabilities.push(value.clone());
                }
                "--topology" => set_once_v1(&mut parsed.topology, value.clone(), flag)?,
                "--listen" => set_once_v1(&mut parsed.listen, value.clone(), flag)?,
                "--max-context-tokens" => set_once_v1(
                    &mut parsed.maximum_context_tokens,
                    count_v1(flag, value)?,
                    flag,
                )?,
                "--kv-memory-fraction" => set_once_v1(
                    &mut parsed.kv_memory_fraction_millionths,
                    parse_kv_fraction_v1(value)?,
                    flag,
                )?,
                "--kv-pages" => set_once_v1(&mut parsed.kv_pages, count_v1(flag, value)?, flag)?,
                "--context-parallel-size" => set_once_v1(
                    &mut parsed.context_parallel_size,
                    count_v1(flag, value)?,
                    flag,
                )?,
                "--prefill-budget" => {
                    let policy = match value.as_str() {
                        "fixed" => runtime_scheduler::PrefillBudget::Fixed,
                        "decode-aware" => runtime_scheduler::PrefillBudget::DecodeAware(
                            runtime_scheduler::PrefillTarget::default(),
                        ),
                        "tbt" => runtime_scheduler::PrefillBudget::Tbt(
                            runtime_scheduler::MixedPolicy::default(),
                        ),
                        _ => {
                            return Err(usage_error(
                                "--prefill-budget must be fixed, decode-aware or tbt",
                            ));
                        }
                    };
                    set_once_v1(&mut parsed.prefill_budget, policy, flag)?;
                }
                "--prefill-itl-target-ms" => set_once_v1(
                    &mut parsed.prefill_itl_target_ms,
                    count_v1(flag, value)?,
                    flag,
                )?,
                "--prefill-decode-steps" => set_once_v1(
                    &mut parsed.prefill_decode_steps,
                    count_v1(flag, value)?,
                    flag,
                )?,
                "--decode-tbt-target-ms" => set_once_v1(
                    &mut parsed.decode_tbt_target_ms,
                    count_v1(flag, value)?,
                    flag,
                )?,
                "--mixed-min-prefill-tokens" => set_once_v1(
                    &mut parsed.mixed_min_prefill_tokens,
                    count_v1(flag, value)?,
                    flag,
                )?,
                "--mixed-decode-steps" => {
                    set_once_v1(&mut parsed.mixed_decode_steps, count_v1(flag, value)?, flag)?
                }
                "--kv-offload-gib" => set_once_v1(
                    &mut parsed.kv_offload_gib,
                    count_v1(flag, value)? as u64,
                    flag,
                )?,
                "--kv-offload-min-tokens" => set_once_v1(
                    &mut parsed.kv_offload_min_tokens,
                    count_v1(flag, value)?,
                    flag,
                )?,
                "--kv-offload-gbps" => {
                    let rate = value.parse::<f64>().map_err(|_| {
                        usage_error("--kv-offload-gbps must be a positive finite rate")
                    })? * 1e9;
                    if !rate.is_finite() || rate < 1.0 || rate >= u64::MAX as f64 {
                        return Err(usage_error(
                            "--kv-offload-gbps must be a positive finite rate",
                        ));
                    }
                    set_once_v1(&mut parsed.kv_offload_rate, rate as u64, flag)?;
                }
                "--kv-offload-prefix" => {
                    let enabled = match value.as_str() {
                        "off" => false,
                        "on" => true,
                        _ => return Err(usage_error("--kv-offload-prefix must be on or off")),
                    };
                    set_once_v1(&mut parsed.kv_offload_prefix, enabled, flag)?;
                }
                "--admission" => {
                    let optimistic = match value.as_str() {
                        "optimistic" => true,
                        "reserved" => false,
                        _ => {
                            return Err(usage_error("--admission must be reserved or optimistic"));
                        }
                    };
                    set_once_v1(&mut parsed.optimistic_admission, optimistic, flag)?;
                }
                "--admission-watermark" => set_once_v1(
                    &mut parsed.admission_watermark_millionths,
                    parse_admission_watermark_v1(value)?,
                    flag,
                )?,
                "--max-preemptions" => set_once_v1(
                    &mut parsed.maximum_preemptions,
                    count_v1(flag, value)?,
                    flag,
                )?,
                "--debug-swap-every" => {
                    set_once_v1(&mut parsed.debug_swap_every, count_v1(flag, value)?, flag)?
                }
                "--debug-preempt-every" => set_once_v1(
                    &mut parsed.debug_preempt_every,
                    count_v1(flag, value)?,
                    flag,
                )?,
                "--speculative-tokens" => {
                    set_once_v1(&mut parsed.speculative_tokens, count_v1(flag, value)?, flag)?
                }
                "--speculative-rows" => {
                    set_once_v1(&mut parsed.speculative_rows, count_v1(flag, value)?, flag)?
                }
                "--draft-cost" => {
                    let model = match value.as_str() {
                        "per-token" => DraftCostModel::PerToken,
                        "per-block" => DraftCostModel::PerBlock,
                        _ => {
                            return Err(usage_error("--draft-cost must be per-token or per-block"));
                        }
                    };
                    set_once_v1(&mut parsed.draft_cost_model, model, flag)?;
                }
                "--speculation" => {
                    let mode = match value.as_str() {
                        "fixed" => SpeculationMode::Fixed,
                        "adaptive" => SpeculationMode::Adaptive,
                        _ => return Err(usage_error("--speculation must be fixed or adaptive")),
                    };
                    set_once_v1(&mut parsed.speculation_mode, mode, flag)?;
                }
                "--kv-cache-dtype" => {
                    let dtype = KvCacheDtypeV1::parse(value).ok_or_else(|| {
                        usage_error("--kv-cache-dtype must be auto, bf16, fp8 or mixed")
                    })?;
                    set_once_v1(&mut parsed.kv_cache_dtype, dtype, flag)?;
                }
                "--prefix-cache" => {
                    let enabled = match value.as_str() {
                        "on" => true,
                        "off" => false,
                        _ => return Err(usage_error("--prefix-cache must be on or off")),
                    };
                    set_once_v1(&mut parsed.prefix_cache, enabled, flag)?;
                }
                "--verify-warmup" => {
                    let enabled = match value.as_str() {
                        "on" => true,
                        "off" => false,
                        _ => return Err(usage_error("--verify-warmup must be on or off")),
                    };
                    set_once_v1(&mut parsed.verify_warmup, enabled, flag)?;
                }
                "--max-sequences" => {
                    set_once_v1(&mut parsed.maximum_sequences, count_v1(flag, value)?, flag)?;
                }
                "--max-batch-tokens" => {
                    set_once_v1(
                        &mut parsed.maximum_batch_tokens,
                        count_v1(flag, value)?,
                        flag,
                    )?;
                }
                "--prefill-chunk-tokens" => set_once_v1(
                    &mut parsed.prefill_chunk_tokens,
                    count_v1(flag, value)?,
                    flag,
                )?,
                "--queue-depth" => {
                    set_once_v1(&mut parsed.queue_depth, count_v1(flag, value)?, flag)?;
                }
                "--drain-timeout-ms" => set_once_v1(
                    &mut parsed.drain_timeout_ms,
                    value
                        .parse()
                        .map_err(|_| usage_error("--drain-timeout-ms must be an integer"))?,
                    flag,
                )?,
                "--queue-wait-ms" => set_once_v1(
                    &mut parsed.queue_wait_ms,
                    value
                        .parse()
                        .map_err(|_| usage_error("--queue-wait-ms must be an integer"))?,
                    flag,
                )?,
                "--max-request-body-bytes" => set_once_v1(
                    &mut parsed.maximum_request_body_bytes,
                    count_v1(flag, value)?,
                    flag,
                )?,
                "--max-request-string-bytes" => set_once_v1(
                    &mut parsed.maximum_request_string_bytes,
                    count_v1(flag, value)?,
                    flag,
                )?,
                "--responses-store-bytes" => set_once_v1(
                    &mut parsed.responses_store_bytes,
                    count_v1(flag, value)?,
                    flag,
                )?,
                "--max-tokenizer-input-bytes" => set_once_v1(
                    &mut parsed.maximum_tokenizer_input_bytes,
                    count_v1(flag, value)?,
                    flag,
                )?,
                "--max-images-per-request" => set_once_v1(
                    &mut parsed.maximum_images_per_request,
                    count_v1(flag, value)?,
                    flag,
                )?,
                "--allowed-media-domains" => {
                    parsed
                        .remote_media
                        .allowed_domains
                        .extend(value.split(',').map(str::to_owned));
                    parsed
                        .remote_media
                        .validate()
                        .map_err(|error| usage_error(error.to_string()))?;
                }
                "--media-allow-private-addresses" => {
                    parsed.remote_media.allow_private_addresses = match value.as_str() {
                        "on" => true,
                        "off" => false,
                        _ => {
                            return Err(usage_error(
                                "--media-allow-private-addresses takes on or off",
                            ));
                        }
                    };
                }
                "--media-fetch-timeout-ms"
                | "--media-connect-timeout-ms"
                | "--media-request-timeout-ms" => {
                    let duration = std::time::Duration::from_millis(count_v1(flag, value)? as u64);
                    if duration.is_zero() || duration > std::time::Duration::from_secs(300) {
                        return Err(usage_error("media timeout must be between 1 and 300000 ms"));
                    }
                    if flag == "--media-fetch-timeout-ms" {
                        parsed.remote_media.total_timeout = duration;
                    } else if flag == "--media-request-timeout-ms" {
                        parsed.remote_media.request_timeout = duration;
                    } else {
                        parsed.remote_media.connect_timeout = duration;
                    }
                }
                "--media-max-redirects" => {
                    parsed.remote_media.maximum_redirects = count_v1(flag, value)?;
                    parsed
                        .remote_media
                        .validate()
                        .map_err(|error| usage_error(error.to_string()))?;
                }
                "--allowed-local-media-path" => set_once_v1(
                    &mut parsed.allowed_local_media_path,
                    PathBuf::from(value),
                    flag,
                )?,
                "--debug-reference-tokens" => set_once_v1(
                    &mut parsed.debug_reference_tokens,
                    PathBuf::from(value),
                    flag,
                )?,
                "--checkpoint" => {
                    set_once_v1(&mut parsed.direct_checkpoint, PathBuf::from(value), flag)?
                }
                "--load-threads" => {
                    set_once_v1(&mut parsed.load_threads, count_v1(flag, value)?, flag)?
                }
                "--fixture-request" => {
                    set_once_v1(&mut parsed.fixture_request, value.clone(), flag)?;
                }
                "--fixture-tokens" => {
                    set_once_v1(&mut parsed.fixture_tokens, count_v1(flag, value)?, flag)?;
                }
                "--scenario" => {
                    let scenario = FixtureScenarioV1::parse(value).ok_or_else(|| {
                        usage_error(format!(
                            "--scenario takes one of commit, cancel, drop, got '{value}'"
                        ))
                    })?;
                    set_once_v1(&mut parsed.scenario, scenario, flag)?;
                }
                _ => return Err(usage_error(format!("unsupported argument '{flag}'"))),
            }
        }
        Ok(parsed)
    }

    /// Every flag this invocation carried, named as it was written.
    fn present_flags(&self) -> Vec<&'static str> {
        let mut present = self.remote_media_flags.clone();
        for (flag, carried) in [
            ("--bundle", self.bundle.is_some()),
            ("--model", self.model.is_some()),
            (
                "--tensor-parallel-size",
                self.tensor_parallel_size.is_some(),
            ),
            ("--model-option", !self.model_options.is_empty()),
            ("--list-model-options", self.list_model_options),
            ("--model-cache-dir", self.model_cache_dir.is_some()),
            ("--activation-policy", self.activation_policy.is_some()),
            ("--modalities", self.modalities.is_some()),
            ("--topology", self.topology.is_some()),
            ("--devices", self.devices.is_some()),
            ("--listen", self.listen.is_some()),
            (
                "--max-context-tokens",
                self.maximum_context_tokens.is_some(),
            ),
            ("--max-sequences", self.maximum_sequences.is_some()),
            (
                "--kv-memory-fraction",
                self.kv_memory_fraction_millionths.is_some(),
            ),
            ("--kv-pages", self.kv_pages.is_some()),
            (
                "--context-parallel-size",
                self.context_parallel_size.is_some(),
            ),
            ("--kv-cache-dtype", self.kv_cache_dtype.is_some()),
            ("--speculative-tokens", self.speculative_tokens.is_some()),
            ("--speculative-rows", self.speculative_rows.is_some()),
            ("--speculation", self.speculation_mode.is_some()),
            ("--draft-cost", self.draft_cost_model.is_some()),
            ("--kv-offload-gib", self.kv_offload_gib.is_some()),
            (
                "--kv-offload-min-tokens",
                self.kv_offload_min_tokens.is_some(),
            ),
            ("--kv-offload-gbps", self.kv_offload_rate.is_some()),
            ("--kv-offload-prefix", self.kv_offload_prefix.is_some()),
            ("--prefix-cache", self.prefix_cache.is_some()),
            ("--verify-warmup", self.verify_warmup.is_some()),
            ("--prefill-budget", self.prefill_budget.is_some()),
            (
                "--prefill-itl-target-ms",
                self.prefill_itl_target_ms.is_some(),
            ),
            (
                "--prefill-decode-steps",
                self.prefill_decode_steps.is_some(),
            ),
            (
                "--decode-tbt-target-ms",
                self.decode_tbt_target_ms.is_some(),
            ),
            (
                "--mixed-min-prefill-tokens",
                self.mixed_min_prefill_tokens.is_some(),
            ),
            ("--mixed-decode-steps", self.mixed_decode_steps.is_some()),
            ("--admission", self.optimistic_admission.is_some()),
            (
                "--admission-watermark",
                self.admission_watermark_millionths.is_some(),
            ),
            ("--max-preemptions", self.maximum_preemptions.is_some()),
            ("--debug-preempt-every", self.debug_preempt_every.is_some()),
            ("--debug-swap-every", self.debug_swap_every.is_some()),
            (
                "--responses-store-bytes",
                self.responses_store_bytes.is_some(),
            ),
            ("--max-batch-tokens", self.maximum_batch_tokens.is_some()),
            (
                "--prefill-chunk-tokens",
                self.prefill_chunk_tokens.is_some(),
            ),
            ("--queue-depth", self.queue_depth.is_some()),
            ("--queue-wait-ms", self.queue_wait_ms.is_some()),
            ("--drain-timeout-ms", self.drain_timeout_ms.is_some()),
            (
                "--max-request-body-bytes",
                self.maximum_request_body_bytes.is_some(),
            ),
            (
                "--max-request-string-bytes",
                self.maximum_request_string_bytes.is_some(),
            ),
            (
                "--max-tokenizer-input-bytes",
                self.maximum_tokenizer_input_bytes.is_some(),
            ),
            (
                "--max-images-per-request",
                self.maximum_images_per_request.is_some(),
            ),
            (
                "--debug-reference-tokens",
                self.debug_reference_tokens.is_some(),
            ),
            ("--checkpoint", self.direct_checkpoint.is_some()),
            ("--load-threads", self.load_threads.is_some()),
            ("--fixture-request", self.fixture_request.is_some()),
            ("--fixture-tokens", self.fixture_tokens.is_some()),
            ("--scenario", self.scenario.is_some()),
            ("--fake-host-services", self.host_memory_device),
        ] {
            if carried {
                present.push(flag);
            }
        }
        present
    }

    fn reject_unsupported(
        &self,
        command: &str,
        accepted: &[&'static str],
    ) -> Result<(), HostCommandErrorV1> {
        for flag in &self.http_flags {
            if !accepted.contains(&flag.as_str()) {
                return Err(usage_error(format!("{command} does not accept {flag}")));
            }
        }

        for flag in self.present_flags() {
            if !accepted.contains(&flag) {
                return Err(usage_error(format!("{command} does not accept {flag}")));
            }
        }
        Ok(())
    }

    /// Which device services this invocation asked for.
    fn device_services(&self) -> DeviceServicesV1 {
        if self.host_memory_device {
            DeviceServicesV1::HostMemory
        } else {
            DeviceServicesV1::Compiled
        }
    }

    fn serve_selection(&self) -> Result<ServeSelection, HostCommandErrorV1> {
        let Some(path) = &self.model else {
            if self.tensor_parallel_size.is_some() || self.model_cache_dir.is_some() {
                return Err(usage_error("model preparation options require --model"));
            }
            return self.selection("serve").map(ServeSelection::Bundle);
        };
        if path.as_os_str().is_empty() || self.bundle.is_some() || self.topology.is_some() {
            return Err(usage_error(
                "--model requires a directory and is exclusive with bundle, topology and profile selection",
            ));
        }
        let ranks = self
            .tensor_parallel_size
            .unwrap_or(crate::preparation::DEFAULT_TENSOR_PARALLEL_SIZE as usize);
        if !(1..=64).contains(&ranks)
            || self
                .devices
                .as_ref()
                .is_some_and(|devices| devices.len() != ranks)
        {
            return Err(usage_error(
                "--tensor-parallel-size must be 1..64 and match the number of selected device ordinals",
            ));
        }
        Ok(ServeSelection::Model(Box::new(LocalModel {
            path: path.clone(),
            cache_root: self.model_cache_dir.clone(),
            tensor_parallel_size: ranks as u32,
            activation_policy: self.activation_policy.clone(),
            modalities: self.modalities.clone().unwrap_or_default(),
            serve_arguments: Vec::new(),
        })))
    }

    /// The package directory this invocation named.
    fn selection(&self, command: &str) -> Result<BundleSelectionV1, HostCommandErrorV1> {
        let bundle = self.bundle.clone().ok_or_else(|| {
            usage_error(format!(
                "{command} requires --bundle, naming a package directory"
            ))
        })?;
        Ok(BundleSelectionV1::Directory(PathBuf::from(bundle)))
    }

    /// What this host tells the verifier about itself.
    fn host_compatibility(&self) -> Result<BundleHostCompatibilityV1, HostCommandErrorV1> {
        let architecture = self
            .architecture
            .clone()
            .unwrap_or_else(|| DEFAULT_HOST_ARCHITECTURE_V1.to_owned());
        let capabilities = self
            .capabilities
            .iter()
            .map(String::as_str)
            .chain(std::iter::once(
                runtime_bundle_abi::REQUEST_RETIREMENT_CAPABILITY_V1,
            ))
            .chain(std::iter::once(
                runtime_bundle_abi::SERVING_KV_CAPABILITY_V1,
            ))
            .chain(std::iter::once(crate::executor::PROMPT_ROWS_CAPABILITY_V1))
            .chain(std::iter::once(crate::executor::SPECULATIVE_CAPABILITY_V1))
            .chain(std::iter::once(
                runtime_bundle_abi::SPECULATIVE_NARROW_ROWS_CAPABILITY_V1,
            ))
            .chain(std::iter::once(crate::executor::MEDIA_INPUTS_CAPABILITY_V1))
            .chain(std::iter::once(
                crate::executor::POSITION_AXES_CAPABILITY_V1,
            ))
            .chain(std::iter::once(crate::offload::KV_OFFLOAD_CAPABILITY_V1))
            .chain(std::iter::once(
                runtime_bundle_abi::PREFIX_CACHE_CAPABILITY_V1,
            ))
            .chain(std::iter::once(
                runtime_bundle_abi::PREFIX_CACHE_STATE_SLOT_CAPABILITY_V1,
            ))
            .chain(std::iter::once(
                runtime_bundle_abi::RECURRENT_STATE_CAPABILITY_V1,
            ))
            .chain(std::iter::once(
                runtime_bundle_abi::RECURRENT_STATE_PREALLOCATED_CAPABILITY_V1,
            ))
            .chain(std::iter::once(runtime_bundle_abi::FP8_KV_CAPABILITY_V1))
            .chain(std::iter::once(
                runtime_bundle::MIXED_PLANE_KV_CAPABILITY_V1,
            ))
            .chain(std::iter::once(
                runtime_bundle_abi::HOST_RESIDENT_VIEWS_CAPABILITY_V1,
            ))
            .chain(std::iter::once(
                runtime_bundle_abi::HOST_MAPPED_GATHER_CAPABILITY_V1,
            ))
            .chain(std::iter::once(
                runtime_bundle_abi::SERVED_ITEMS_CAPABILITY_V1,
            ))
            .chain(std::iter::once(
                runtime_bundle_abi::SERVED_CONTEXT_CAPABILITY_V1,
            ))
            .chain(std::iter::once(
                runtime_bundle_abi::MODEL_OPTIONS_CAPABILITY_V1,
            ))
            .chain(std::iter::once(
                runtime_bundle_abi::RECURRENT_STATE_IN_PLACE_CAPABILITY_V1,
            ))
            .chain(std::iter::once(
                runtime_bundle_abi::STATE_PER_ITEM_CAPABILITY_V1,
            ))
            // Packages that load directly may carry no rank images.
            .chain(std::iter::once(
                runtime_bundle_abi::weight_plan::WEIGHT_PLAN_CAPABILITY_V1,
            ))
            // The served frontend is the module's own export.
            .chain(std::iter::once(
                runtime_bundle_abi::frontend::FRONTEND_CAPABILITY_V1,
            ))
            .map(BundleOpaqueId::new)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(
            BundleHostCompatibilityV1::new(architecture, MODEL_BUNDLE_V1_ABI_VERSION)?
                .with_capabilities(capabilities),
        )
    }
}

/// The largest `--prefill-itl-target-ms`, one minute.
const MAXIMUM_PREFILL_ITL_TARGET_MS_V1: usize = 60_000;
/// The largest `--prefill-decode-steps`.
const MAXIMUM_PREFILL_DECODE_STEPS_V1: usize = 64;

/// Applies the target flags of `--prefill-budget decode-aware` or `tbt`.
/// The prefill budget of `parsed`. With `profile_pending` (`serve --model`,
/// whose profile may still supply `--prefill-budget`), options that need a
/// budget the invocation does not name are checked for their own values
/// only; the parse after the profile's defaults checks the pairing.
fn prefill_budget_v1(
    parsed: &ArgumentsV1,
    profile_pending: bool,
) -> Result<runtime_scheduler::PrefillBudget, HostCommandErrorV1> {
    use runtime_scheduler::{PrefillBudget, PrefillTarget};
    let budget_pending = profile_pending && parsed.prefill_budget.is_none();
    if matches!(parsed.prefill_budget, Some(PrefillBudget::Tbt(_))) {
        if let Some(flag) = [
            ("--prefill-itl-target-ms", parsed.prefill_itl_target_ms),
            ("--prefill-decode-steps", parsed.prefill_decode_steps),
        ]
        .into_iter()
        .find_map(|(flag, value)| value.map(|_| flag))
        {
            return Err(usage_error(format!(
                "{flag} requires --prefill-budget decode-aware"
            )));
        }
        return mixed_policy_v1(parsed).map(PrefillBudget::Tbt);
    }
    if let Some(flag) = [
        ("--decode-tbt-target-ms", parsed.decode_tbt_target_ms),
        (
            "--mixed-min-prefill-tokens",
            parsed.mixed_min_prefill_tokens,
        ),
        ("--mixed-decode-steps", parsed.mixed_decode_steps),
    ]
    .into_iter()
    .find_map(|(flag, value)| value.map(|_| flag))
    {
        if !budget_pending {
            return Err(usage_error(format!("{flag} requires --prefill-budget tbt")));
        }
        mixed_policy_v1(parsed)?;
    }
    let target = match (parsed.prefill_itl_target_ms, parsed.prefill_decode_steps) {
        (Some(_), Some(_)) => {
            return Err(usage_error(
                "--prefill-itl-target-ms and --prefill-decode-steps are exclusive",
            ));
        }
        (Some(millis), None) => {
            if !(1..=MAXIMUM_PREFILL_ITL_TARGET_MS_V1).contains(&millis) {
                return Err(usage_error(format!(
                    "--prefill-itl-target-ms must be 1 to {MAXIMUM_PREFILL_ITL_TARGET_MS_V1}"
                )));
            }
            Some((
                "--prefill-itl-target-ms",
                PrefillTarget::ItlMillis(millis as u32),
            ))
        }
        (None, Some(steps)) => {
            if !(1..=MAXIMUM_PREFILL_DECODE_STEPS_V1).contains(&steps) {
                return Err(usage_error(format!(
                    "--prefill-decode-steps must be 1 to {MAXIMUM_PREFILL_DECODE_STEPS_V1}"
                )));
            }
            Some((
                "--prefill-decode-steps",
                PrefillTarget::DecodeSteps(steps as u32),
            ))
        }
        (None, None) => None,
    };
    match (parsed.prefill_budget.unwrap_or_default(), target) {
        (PrefillBudget::DecodeAware(_), Some((_, target))) => {
            Ok(PrefillBudget::DecodeAware(target))
        }
        (_, Some((flag, _)))
            if budget_pending
                && [
                    parsed.decode_tbt_target_ms,
                    parsed.mixed_min_prefill_tokens,
                    parsed.mixed_decode_steps,
                ]
                .iter()
                .any(Option::is_some) =>
        {
            // A TBT option needs the tbt budget and a decode-aware target the
            // decode-aware one, so no profile's budget can serve both.
            Err(usage_error(format!(
                "{flag} and the TBT options need different --prefill-budget values"
            )))
        }
        (budget, Some(_)) if budget_pending => Ok(budget),
        (_, Some((flag, _))) => Err(usage_error(format!(
            "{flag} requires --prefill-budget decode-aware"
        ))),
        (budget, None) => Ok(budget),
    }
}

/// The `--prefill-budget tbt` policy from its flags and defaults.
fn mixed_policy_v1(
    parsed: &ArgumentsV1,
) -> Result<runtime_scheduler::MixedPolicy, HostCommandErrorV1> {
    use runtime_scheduler::{MAXIMUM_MIXED_DECODE_STEPS, MixedPolicy};
    let defaults = MixedPolicy::default();
    let target = parsed
        .decode_tbt_target_ms
        .unwrap_or(defaults.target_millis() as usize);
    if !(1..=MAXIMUM_PREFILL_ITL_TARGET_MS_V1).contains(&target) {
        return Err(usage_error(format!(
            "--decode-tbt-target-ms must be 1 to {MAXIMUM_PREFILL_ITL_TARGET_MS_V1}"
        )));
    }
    let floor = parsed
        .mixed_min_prefill_tokens
        .unwrap_or(defaults.min_prefill_tokens());
    if floor == 0 {
        return Err(usage_error("--mixed-min-prefill-tokens must be positive"));
    }
    let steps = parsed.mixed_decode_steps.unwrap_or(0);
    if steps > MAXIMUM_MIXED_DECODE_STEPS as usize {
        return Err(usage_error(format!(
            "--mixed-decode-steps must be 0 to {MAXIMUM_MIXED_DECODE_STEPS}"
        )));
    }
    MixedPolicy::new(target as u32, floor, steps as u32)
        .map_err(|error| usage_error(error.to_string()))
}

fn count_v1(flag: &str, value: &str) -> Result<usize, HostCommandErrorV1> {
    value
        .parse::<usize>()
        .map_err(|_| usage_error(format!("{flag} takes a count, got '{value}'")))
}

fn parse_kv_fraction_v1(value: &str) -> Result<u32, HostCommandErrorV1> {
    let fraction = value
        .parse::<f64>()
        .map_err(|_| usage_error("--kv-memory-fraction must be a number in (0, 1]"))?;
    if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0 {
        return Err(usage_error("--kv-memory-fraction must be in (0, 1]"));
    }
    let millionths = (fraction * 1_000_000.0).round() as u32;
    if millionths == 0 {
        return Err(usage_error("--kv-memory-fraction is too small"));
    }
    Ok(millionths)
}

fn parse_admission_watermark_v1(value: &str) -> Result<u32, HostCommandErrorV1> {
    let fraction = value
        .parse::<f64>()
        .map_err(|_| usage_error("--admission-watermark must be a number in [0, 1)"))?;
    if !fraction.is_finite() || !(0.0..1.0).contains(&fraction) {
        return Err(usage_error("--admission-watermark must be in [0, 1)"));
    }
    Ok(((fraction * 1_000_000.0).round() as u32).min(999_999))
}

fn set_once_v1<T>(slot: &mut Option<T>, value: T, flag: &str) -> Result<(), HostCommandErrorV1> {
    if slot.is_some() {
        return Err(given_more_than_once_v1(flag));
    }
    *slot = Some(value);
    Ok(())
}

fn given_more_than_once_v1(flag: &str) -> HostCommandErrorV1 {
    usage_error(format!("{flag} is given more than once"))
}

/// A label rides into the receipt, so it carries the same bounded charset an
/// identifier does and nothing that would need escaping.
fn validate_label_v1(label: &str) -> Result<(), HostCommandErrorV1> {
    if label.is_empty()
        || label.len() > MAX_FIXTURE_LABEL_BYTES_V1
        || !label.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._-".contains(&byte)
        })
    {
        return Err(usage_error(format!(
            "fixture request label '{label}' must contain 1..={MAX_FIXTURE_LABEL_BYTES_V1} \
             lowercase ASCII letters, digits, '.', '_', or '-'"
        )));
    }
    Ok(())
}

fn usage_error(message: impl Into<String>) -> HostCommandErrorV1 {
    HostCommandErrorV1::Usage(format!("{}\n{HOST_COMMAND_USAGE_V1}", message.into()))
}

fn run_error(message: impl Into<String>) -> HostCommandErrorV1 {
    HostCommandErrorV1::Run(message.into())
}

#[cfg(test)]
mod tests {
    /// A build without the device driver refuses, before loading weights, a
    /// package needing more device memory than its host-memory services
    /// report, and still serves one that fits them.
    #[cfg(all(
        feature = "fake-host-services",
        not(all(target_os = "linux", feature = "linux-cuda-host"))
    ))]
    #[test]
    fn host_memory_fallback_refuses_packages_larger_than_its_device() {
        let available = crate::fake::DEFAULT_FAKE_DEVICE_MEMORY_BYTES_V1;
        assert!(super::host_memory_fallback_v1(available).is_ok());
        let error = super::host_memory_fallback_v1(94 << 30)
            .unwrap_err()
            .to_string();
        assert!(error.contains("links no device driver"), "{error}");
        assert!(error.contains("runtime-host/linux-cuda-host"), "{error}");
    }

    #[test]
    fn help_lists_media_controls_and_verification_workers() {
        for option in [
            "--jobs",
            "--allowed-local-media-path",
            "--allowed-media-domains",
            "--media-allow-private-addresses",
            "--media-request-timeout-ms",
            "--media-fetch-timeout-ms",
            "--media-connect-timeout-ms",
            "--media-max-redirects",
        ] {
            assert!(
                super::HOST_COMMAND_USAGE_V1.contains(option),
                "help omits {option}"
            );
        }
    }

    use std::path::PathBuf;

    use runtime_bundle::{
        BundleBatchLimitsV1, BundleComponentV1, BundleGraphBucketV1, BundleKvDtypeV1,
        BundleKvFormatV1, BundleKvLayoutV1, BundleManifestV1, BundleOpaqueId,
        BundleRankPlacementV1, BundleRelativePathV1, BundleResourceRequirementsV1,
        BundleStateLayoutV1, BundleStateLifetimeV1, BundleTargetV1, BundleTopologyV1,
    };

    use runtime_engine::{EngineConfig, SpeculationMode};
    use runtime_kv::KvGeometry;
    use runtime_model_api::ModelVocabulary;
    use runtime_scheduler::{
        AdmissionPolicy, OptimisticAdmission, SchedulerCapacityProfile, SchedulerConfig,
    };
    use runtime_server::{
        MAX_IN_FLIGHT_REQUESTS, RequestAcceptanceProfile, RequestSurfaceV1, ServerConfig,
    };

    use crate::executor::BATCH_REQUEST_MAGIC_V1;

    use super::{
        ArgumentsV1, BundleHostCompatibilityV1, BundleSelectionV1, DEFAULT_HOST_ARCHITECTURE_V1,
        DEFAULT_LISTEN_ADDRESS_V1, DEFAULT_SERVED_SEQUENCES_V1, DeveloperServingEnvelopeV1,
        DeviceServicesV1, FixturePlanV1, FixtureRequestV1, FixtureScenarioV1, HostCommandErrorV1,
        HostCommandV1, KvCacheDtypeV1, PREPARED_QUEUE_DEPTH_V1, ServingOptionsV1, engine_config_v1,
        parse_bundle_command_v1, parse_conformance_host_command_v1, parse_host_command_v1,
        resolve_developer_serving_plan_v1, resolve_kv_cache_dtype_v1, resolve_topology_v1,
        same_selection_v1, server_config_v1, serving_backend_config_v1, validate_label_v1,
    };

    fn arguments(words: &[&str]) -> Vec<String> {
        words.iter().map(|word| (*word).to_owned()).collect()
    }

    #[test]
    fn responses_storage_is_an_explicit_serving_option() {
        let HostCommandV1::Serve { serving, .. } = parse_host_command_v1(&arguments(&[
            "serve",
            "--bundle",
            "fixture",
            "--responses-store-bytes",
            "268435456",
        ]))
        .unwrap() else {
            panic!("serve command");
        };
        assert_eq!(serving.responses_store_bytes, 268435456);
        assert_eq!(ServingOptionsV1::default().responses_store_bytes, 0);
        assert!(
            refusal(&[
                "serve",
                "--bundle",
                "fixture",
                "--responses-store-bytes",
                "bad"
            ])
            .contains("count")
        );
        assert!(
            refusal(&[
                "serve",
                "--bundle",
                "fixture",
                "--responses-store-bytes",
                "10",
                "--responses-store-bytes",
                "20"
            ])
            .contains("more than once")
        );
    }

    fn refusal(words: &[&str]) -> String {
        match parse_host_command_v1(&arguments(words)) {
            Err(error) => error.to_string(),
            Ok(command) => panic!("this invocation must be refused, it parsed as {command:?}"),
        }
    }

    #[test]
    fn removed_rank_flag_explains_distinct_devices() {
        let refusal = refusal(&["serve", "--bundle", "fixture", "--fold-ranks"]);
        assert!(refusal.contains("--fold-ranks is gone"), "{refusal}");
        assert!(
            refusal.contains("one distinct device per rank"),
            "{refusal}"
        );
    }

    #[test]
    fn serve_devices_parse_in_rank_order_and_are_serve_only() {
        let command = parse_host_command_v1(&arguments(&[
            "serve",
            "--bundle",
            "/fixture",
            "--devices",
            "3,1,0,2",
        ]))
        .unwrap();
        let HostCommandV1::Serve { devices, .. } = command else {
            panic!("serve command expected")
        };
        assert_eq!(devices.unwrap().0, [3, 1, 0, 2]);
        for flags in [
            vec!["--devices", "0,0"],
            vec!["--devices", "-1"],
            vec!["--devices", "0", "--devices", "1"],
        ] {
            let mut args = arguments(&["serve", "--bundle", "/fixture"]);
            args.extend(flags.into_iter().map(str::to_owned));
            assert!(parse_host_command_v1(&args).is_err());
        }
        let error = parse_host_command_v1(&arguments(&[
            "conformance-run",
            "--bundle",
            "/fixture",
            "--fixture-request",
            "test",
            "--devices",
            "0",
        ]))
        .unwrap_err();
        assert!(error.to_string().contains("--devices"));
    }

    #[test]
    fn nvdec_uses_the_selected_rank_zero_device() {
        for devices in [
            vec![7],
            vec![1, 0],
            vec![3, 1, 0, 2],
            vec![7, 6, 5, 4, 3, 2, 1, 0],
        ] {
            let topology = super::ServedTopologyV1 {
                id: "test".to_owned(),
                device_ordinals: devices.clone(),
                context_parallel_size: 1,
            };
            assert_eq!(topology.video_device(), devices[0]);
        }
    }

    #[test]
    fn an_explicit_package_takes_no_activation_policy() {
        let refusal = parse_host_command_v1(&arguments(&[
            "serve",
            "--bundle",
            "fixture",
            "--activation-policy",
            "bf16",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            refusal.contains("--activation-policy applies to --model"),
            "{refusal}"
        );
    }

    #[test]
    fn the_modalities_flag_names_the_package_a_model_serves() {
        use runtime_bundle::preparation::Modality::*;
        use runtime_bundle::preparation::ModalitySelection::{All, Auto, Only};
        let model = |flags: &[&str]| -> Result<super::LocalModel, String> {
            let mut words = arguments(&["serve", "--model", "checkpoint"]);
            words.extend(flags.iter().map(|flag| flag.to_string()));
            match parse_host_command_v1(&words).map_err(|error| error.to_string())? {
                HostCommandV1::Serve {
                    selection: super::ServeSelection::Model(model),
                    ..
                } => Ok(*model),
                _ => panic!("expected a model"),
            }
        };
        // Auto, the default, is sent to the helper, which serves text and
        // image of what the checkpoint provides; all serves every modality.
        assert_eq!(model(&[]).unwrap().modalities, Auto);
        assert_eq!(model(&["--modalities", "auto"]).unwrap().modalities, Auto);
        assert_eq!(model(&["--modalities", "all"]).unwrap().modalities, All);
        let text = model(&["--modalities", "text", "--max-sequences", "3"]).unwrap();
        assert_eq!(text.modalities, Only([Text].into()));
        // It selects the package, so the package's serving takes no copy.
        assert_eq!(text.serve_arguments, ["--max-sequences", "3"]);
        assert_eq!(
            model(&["--modalities", "image,text,video"])
                .unwrap()
                .modalities,
            Only([Text, Image, Video].into())
        );
        for (flags, refusal) in [
            (&["--modalities", "image"][..], "must include text"),
            (&["--modalities", "text,text"], "names text twice"),
            (&["--modalities", "text,speech"], "comma list"),
            (
                &["--modalities", "auto", "--modalities", "text"],
                "--modalities",
            ),
        ] {
            let error = model(flags).unwrap_err();
            assert!(error.contains(refusal), "{flags:?}: {error}");
        }
        let refusal = parse_host_command_v1(&arguments(&[
            "serve",
            "--bundle",
            "fixture",
            "--modalities",
            "text",
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            refusal.contains("--modalities applies to --model"),
            "{refusal}"
        );
    }

    #[test]
    fn each_serve_attempt_starts_its_own_readiness_clock() {
        // Two sequential attempts in one process each start their own clock;
        // only the package serve of a `--model` attempt inherits one.
        let first = super::serve_started_v1(None);
        std::thread::sleep(std::time::Duration::from_millis(5));
        let second = super::serve_started_v1(None);
        assert!(second > first);
        assert_eq!(super::serve_started_v1(Some(first)), first);
        // Two failing serve attempts go through the same runner one after
        // the other; neither leaves a start behind for the next.
        for _ in 0..2 {
            let command = parse_host_command_v1(&arguments(&[
                "serve",
                "--bundle",
                "/nonexistent/bundle",
                "--fake-host-services",
            ]))
            .unwrap();
            assert!(super::run_host_command_v1(&command).is_err());
        }
        let third = super::serve_started_v1(None);
        assert!(third > second);
    }

    #[test]
    fn model_speculation_options_wait_for_the_profile_defaults() {
        // The family's profile supplies the tokens and the TBT budget after
        // the package is selected; the direct parse checks these in full.
        for flags in [
            &["--speculation", "fixed"][..],
            &["--speculative-rows", "8"],
            &["--decode-tbt-target-ms", "120"],
        ] {
            let mut words = arguments(&["serve", "--model", "checkpoint"]);
            words.extend(flags.iter().map(|w| (*w).to_owned()));
            assert!(parse_host_command_v1(&words).is_ok(), "{flags:?}");
            let mut words = arguments(&["serve", "--bundle", "fixture"]);
            words.extend(flags.iter().map(|w| (*w).to_owned()));
            assert!(parse_host_command_v1(&words).is_err(), "{flags:?}");
        }
        // Values that do not depend on the profile are refused before any
        // preparation, as are pairings the invocation itself names.
        for (flags, reason) in [
            (
                &["--decode-tbt-target-ms", "0"][..],
                "--decode-tbt-target-ms must be 1 to",
            ),
            (&["--mixed-min-prefill-tokens", "0"], "must be positive"),
            (
                &["--prefill-itl-target-ms", "0"],
                "--prefill-itl-target-ms must be 1 to",
            ),
            (
                &[
                    "--prefill-itl-target-ms",
                    "5",
                    "--prefill-decode-steps",
                    "2",
                ],
                "are exclusive",
            ),
            (
                &["--prefill-budget", "fixed", "--decode-tbt-target-ms", "100"],
                "requires --prefill-budget tbt",
            ),
            (
                &["--prefill-budget", "tbt", "--prefill-decode-steps", "2"],
                "requires --prefill-budget decode-aware",
            ),
            // No budget serves both: the refusal needs no profile.
            (
                &[
                    "--decode-tbt-target-ms",
                    "100",
                    "--prefill-decode-steps",
                    "2",
                ],
                "need different --prefill-budget values",
            ),
            (
                &["--mixed-decode-steps", "1", "--prefill-itl-target-ms", "5"],
                "need different --prefill-budget values",
            ),
        ] {
            let mut words = arguments(&["serve", "--model", "checkpoint"]);
            words.extend(flags.iter().map(|w| (*w).to_owned()));
            let refusal = parse_host_command_v1(&words).unwrap_err().to_string();
            assert!(refusal.contains(reason), "{flags:?}: {refusal}");
        }
    }

    #[test]
    fn local_model_selection_is_exclusive_and_forwards_serving_options() {
        let command = parse_host_command_v1(&arguments(&[
            "serve",
            "--model",
            "checkpoint",
            "--model-cache-dir",
            "cache",
            "--activation-policy",
            "bf16",
            "--tensor-parallel-size",
            "2",
            "--load-threads",
            "6",
            "--devices",
            "1,0",
            "--served-model-name",
            "experiment",
            "--max-sequences",
            "3",
        ]))
        .unwrap();
        let HostCommandV1::Serve {
            selection,
            devices,
            http_options,
            serving,
            qualified_max_sequences,
            direct_checkpoint,
            load_threads,
            ..
        } = command
        else {
            panic!("expected serve")
        };
        let super::ServeSelection::Model(model) = selection else {
            panic!("expected model")
        };
        assert_eq!(model.tensor_parallel_size, 2);
        // The checkpoint is the model directory, selected once its package is.
        assert_eq!(direct_checkpoint, None);
        // Serve options, not the model's, override the family's profile.
        assert_eq!(
            model.serve_arguments,
            [
                "--devices",
                "1,0",
                "--served-model-name",
                "experiment",
                "--max-sequences",
                "3"
            ]
        );
        assert_eq!(load_threads.map(|n| n.get()), Some(6));
        assert_eq!(model.activation_policy.as_deref(), Some("bf16"));
        assert_eq!(model.cache_root, Some(PathBuf::from("cache")));
        assert_eq!(devices.unwrap().0, vec![1, 0]);
        assert_eq!(http_options.served_model_names, vec!["experiment"]);
        assert_eq!(serving.maximum_sequences, 3);
        assert_eq!(qualified_max_sequences, None);
        for flags in [
            vec!["--bundle", "fixture"],
            vec!["--topology", "tp2"],
            vec!["--install-root", "root"],
            vec!["--tensor-parallel-size", "0"],
            vec!["--devices", "0,1"],
            vec!["--refresh-model"],
            vec!["--load", "package"],
            vec!["--load", "direct"],
            vec!["--checkpoint", "other"],
        ] {
            let mut words = arguments(&["serve", "--model", "checkpoint"]);
            words.extend(flags.into_iter().map(String::from));
            assert!(parse_host_command_v1(&words).is_err());
        }
        let mut words = arguments(&["serve", "--model", "checkpoint", "--refresh-model"]);
        let refusal = parse_host_command_v1(&words).unwrap_err().to_string();
        assert!(refusal.contains("chopinfer verify --model"), "{refusal}");
        words.truncate(3);
        words.extend(["--checkpoint".into(), "other".into()]);
        let refusal = parse_host_command_v1(&words).unwrap_err().to_string();
        assert!(
            refusal.contains("--model loads its own checkpoint"),
            "{refusal}"
        );
        for flags in [
            vec!["--refresh-model"],
            vec!["--model-cache-dir", "cache"],
            vec!["--tensor-parallel-size", "4"],
        ] {
            let mut words = arguments(&["serve", "--bundle", "fixture"]);
            words.extend(flags.into_iter().map(String::from));
            assert!(parse_host_command_v1(&words).is_err());
        }
    }

    #[test]
    fn a_selection_is_a_package_directory() {
        let selected = parse_host_command_v1(&arguments(&[
            "serve",
            "--bundle",
            "/build/fixture.recurrent.tp1.v1",
        ]))
        .expect("a directory selection parses");
        let HostCommandV1::Serve {
            selection,
            device,
            listen,
            serving,
            ..
        } = selected
        else {
            panic!("this invocation names serve");
        };
        assert_eq!(
            selection,
            super::ServeSelection::Bundle(BundleSelectionV1::Directory(PathBuf::from(
                "/build/fixture.recurrent.tp1.v1"
            )))
        );
        assert_eq!(device, DeviceServicesV1::Compiled);
        // Serving a machine it was not told to leave is the safe default.
        assert_eq!(listen, DEFAULT_LISTEN_ADDRESS_V1.parse().unwrap());
        assert!(listen.ip().is_loopback());
        assert_eq!(*serving, ServingOptionsV1::default());

        let directory = parse_bundle_command_v1(&arguments(&[
            "verify",
            "--bundle",
            "/build/fixture.recurrent.tp1.v1",
        ]))
        .expect("a directory selection parses");
        let HostCommandV1::Verify { selection, .. } = directory else {
            panic!("this invocation names verify");
        };
        assert_eq!(
            selection,
            BundleSelectionV1::Directory(PathBuf::from("/build/fixture.recurrent.tp1.v1"))
        );
        // Installed bundles, rank-image loads and bundle install are gone,
        // and each refusal names what replaced it.
        for (words, reason) in [
            (
                &["serve", "--install-root", "/opt", "--bundle", "id"][..],
                "--install-root is gone",
            ),
            (
                &["serve", "--bundle", "a", "--load", "package"],
                "--load is gone",
            ),
            (
                &["bundle", "install", "--source", "a", "--root", "b"],
                "is gone",
            ),
            (&["bundle", "install"], "bundle install is gone"),
        ] {
            assert!(refusal(words).contains(reason), "{words:?}");
        }
    }

    /// Serving and the conformance run are two commands, not two spellings.
    ///
    /// The operator's program names every command. The conformance host reads
    /// a bare flag list as its own run, because that is what it exists for, and
    /// still reaches serving only when serving is named.
    #[test]
    fn a_bare_flag_list_is_a_conformance_run_only_for_the_conformance_host() {
        let bare = parse_conformance_host_command_v1(&arguments(&[
            "--bundle",
            "/build/bundle",
            "--fixture-request",
            "external-bundle-host",
            "--fake-host-services",
        ]))
        .expect("the conformance host reads a bare flag list");
        let HostCommandV1::ConformanceRun { device, .. } = bare else {
            panic!("a bare flag list names the conformance run");
        };
        assert_eq!(device, DeviceServicesV1::HostMemory);

        // The same invocation names no command for the operator's program.
        assert!(
            refusal(&[
                "--bundle",
                "/build/bundle",
                "--fixture-request",
                "external-bundle-host",
            ])
            .contains("unsupported command '--bundle'")
        );

        // Both programs run the same four commands.
        for parse in [parse_host_command_v1, parse_conformance_host_command_v1] {
            assert!(matches!(
                parse(&arguments(&["bundle", "verify", "--bundle", "a"])),
                Ok(HostCommandV1::Verify { .. })
            ));
            assert!(matches!(
                parse(&arguments(&["serve", "--bundle", "a"])),
                Ok(HostCommandV1::Serve { .. })
            ));
            assert!(matches!(
                parse(&arguments(&[
                    "conformance-run",
                    "--bundle",
                    "a",
                    "--fixture-request",
                    "run"
                ])),
                Ok(HostCommandV1::ConformanceRun { .. })
            ));
        }

        assert!(
            refusal(&["install", "--source", "a", "--root", "b"]).contains("unsupported command")
        );
        assert!(refusal(&[]).contains("no command"));
        assert!(refusal(&["bundle"]).contains("bundle requires a command"));
        assert!(refusal(&["bundle", "publish"]).contains("unsupported bundle command"));
    }

    /// Vector files that hold no case check nothing, so the run is refused
    /// before the package is read: the bundle named here does not exist.
    #[test]
    fn check_frontend_refuses_vector_files_without_cases() {
        let file = std::env::temp_dir().join(format!(
            "chopinfer-empty-vectors-{}.json",
            std::process::id()
        ));
        std::fs::write(&file, b"[]").unwrap();
        let command = parse_host_command_v1(&arguments(&[
            "bundle",
            "check-frontend",
            "--bundle",
            "missing-package",
            "--render",
            file.to_str().unwrap(),
            "--streaming",
            file.to_str().unwrap(),
        ]))
        .unwrap();
        let refused = super::run_host_command_v1(&command)
            .unwrap_err()
            .to_string();
        std::fs::remove_file(&file).unwrap();
        assert!(
            refused.starts_with("bundle check-frontend found no cases in its vector files"),
            "{refused}"
        );
    }

    /// `bundle check-frontend` takes each kind of vector file any number of
    /// times, in order, and the selection flags every bundle command takes.
    #[test]
    fn check_frontend_reads_repeated_vector_files() {
        for parse in [parse_host_command_v1, parse_conformance_host_command_v1] {
            let Ok(HostCommandV1::CheckFrontend {
                selection,
                topology,
                vectors,
                ..
            }) = parse(&arguments(&[
                "bundle",
                "check-frontend",
                "--render",
                "r1.json",
                "--bundle",
                "pkg",
                "--streaming",
                "s.json",
                "--render",
                "r2.json",
                "--topology",
                "tp4",
            ]))
            else {
                panic!("check-frontend parses");
            };
            assert_eq!(selection, BundleSelectionV1::Directory("pkg".into()));
            assert_eq!(topology.as_deref(), Some("tp4"));
            assert_eq!(
                vectors,
                super::FrontendVectorFilesV1 {
                    render: vec!["r1.json".into(), "r2.json".into()],
                    tokenizer: Vec::new(),
                    streaming: vec!["s.json".into()],
                }
            );
        }
        for (words, expected) in [
            (
                &["bundle", "check-frontend", "--bundle", "a"][..],
                "requires at least one --render, --tokenizer or --streaming FILE",
            ),
            (
                &["bundle", "check-frontend", "--bundle", "a", "--render"],
                "--render requires a FILE",
            ),
            (
                &["bundle", "check-frontend", "--tokenizer", "--bundle", "a"],
                "--tokenizer requires a FILE",
            ),
            (
                &["bundle", "check-frontend", "--render", "r.json"],
                "bundle check-frontend requires --bundle",
            ),
            (
                &[
                    "bundle",
                    "check-frontend",
                    "--bundle",
                    "a",
                    "--render",
                    "r.json",
                    "--listen",
                    "127.0.0.1:0",
                ],
                "bundle check-frontend does not accept --listen",
            ),
        ] {
            let refused = refusal(words);
            assert!(refused.contains(expected), "{words:?}: {refused}");
        }
        // Its help describes the vector files rather than every command.
        let Ok(HostCommandV1::Display { text }) =
            parse_host_command_v1(&arguments(&["bundle", "check-frontend", "--help"]))
        else {
            panic!("help displays");
        };
        assert_eq!(text, super::CHECK_FRONTEND_USAGE_V1);
        assert!(super::HOST_COMMAND_USAGE_V1.contains("bundle check-frontend --bundle DIR"));
    }

    /// The image count is a request count of at least one, and no larger
    /// than a declaration can state.
    #[test]
    fn images_per_request_takes_a_count_from_one() {
        for value in ["0", "65536"] {
            let refused = refusal(&["serve", "--bundle", "a", "--max-images-per-request", value]);
            assert!(
                refused.contains("--max-images-per-request takes a count from 1 to 65535"),
                "{value}: {refused}"
            );
        }
        let refused = refusal(&[
            "serve",
            "--bundle",
            "a",
            "--max-images-per-request",
            "4",
            "--max-images-per-request",
            "8",
        ]);
        assert!(refused.contains("--max-images-per-request"), "{refused}");
    }

    /// A served deployment listens where it was told, and nowhere else.
    #[test]
    fn serving_binds_the_address_it_was_given() {
        let named = parse_host_command_v1(&arguments(&[
            "serve",
            "--bundle",
            "a",
            "--listen",
            "0.0.0.0:9001",
            "--max-context-tokens",
            "1024",
            "--max-sequences",
            "2",
            "--max-batch-tokens",
            "64",
            "--prefill-chunk-tokens",
            "16",
            "--queue-depth",
            "4",
            "--queue-wait-ms",
            "250",
            "--max-request-body-bytes",
            "33554432",
            "--max-request-string-bytes",
            "16777216",
            "--max-tokenizer-input-bytes",
            "8388608",
            "--max-images-per-request",
            "32",
        ]))
        .expect("a named address parses");
        let HostCommandV1::Serve {
            listen, serving, ..
        } = named
        else {
            panic!("this invocation names serve");
        };
        assert_eq!(listen, "0.0.0.0:9001".parse().unwrap());
        assert_eq!(
            *serving,
            ServingOptionsV1 {
                maximum_context_tokens: Some(1024),
                maximum_sequences: 2,
                maximum_batch_tokens: Some(64),
                prefill_chunk_tokens: Some(16),
                queue_depth: Some(4),
                queue_wait_ms: 250,
                drain_timeout_ms: 30_000,
                maximum_request_body_bytes: Some(33_554_432),
                maximum_request_string_bytes: Some(16_777_216),
                responses_store_bytes: 0,
                maximum_tokenizer_input_bytes: Some(8_388_608),
                maximum_images_per_request: Some(32),
                kv_memory_fraction_millionths: 900_000,
                kv_pages: None,
                context_parallel_size: 1,
                kv_cache_dtype: KvCacheDtypeV1::Auto,
                prefix_cache: true,
                debug_swap: false,
                kv_offload: crate::offload::KvOffloadOptionsV1::default(),
                verify_warmup: true,
                admission: AdmissionPolicy::Optimistic(OptimisticAdmission::DEFAULT),
                prefill_budget: runtime_scheduler::PrefillBudget::Fixed,
            }
        );

        for (words, expected) in [
            (
                vec!["serve", "--bundle", "a", "--listen", "8310"],
                "--listen takes an address and port",
            ),
            (
                vec!["serve", "--bundle", "a", "--max-sequences", "0"],
                "--max-sequences must admit at least one",
            ),
            (
                vec!["serve", "--bundle", "a", "--max-sequences", "many"],
                "--max-sequences takes a count",
            ),
            (
                vec!["serve", "--bundle", "a", "--fixture-request", "run"],
                "serve does not accept --fixture-request",
            ),
            (
                vec!["conformance-run", "--bundle", "a", "--listen", "1.2.3.4:5"],
                "conformance-run does not accept --listen",
            ),
            (
                vec!["conformance-run", "--bundle", "a"],
                "conformance-run requires --fixture-request",
            ),
        ] {
            let refusal = refusal(&words);
            assert!(refusal.contains(expected), "{refusal}");
        }
    }

    #[test]
    fn tbt_flags_parse_and_require_tbt() {
        use runtime_scheduler::{MixedPolicy, PrefillBudget};
        let parse = |extra: &[&str]| {
            let mut words = vec!["serve", "--bundle", "a"];
            words.extend_from_slice(extra);
            parse_host_command_v1(&arguments(&words)).map(|command| match command {
                HostCommandV1::Serve { serving, .. } => serving.prefill_budget,
                _ => panic!("serve"),
            })
        };
        assert_eq!(
            parse(&["--prefill-budget", "tbt"]).unwrap(),
            PrefillBudget::Tbt(MixedPolicy::new(100, 512, 0).unwrap())
        );
        assert_eq!(
            parse(&[
                "--decode-tbt-target-ms",
                "75",
                "--prefill-budget",
                "tbt",
                "--mixed-min-prefill-tokens",
                "256",
                "--mixed-decode-steps",
                "2",
            ])
            .unwrap(),
            PrefillBudget::Tbt(MixedPolicy::new(75, 256, 2).unwrap())
        );
        for (words, expected) in [
            (
                vec!["--decode-tbt-target-ms", "100"],
                "--decode-tbt-target-ms requires --prefill-budget tbt",
            ),
            (
                vec![
                    "--prefill-budget",
                    "decode-aware",
                    "--mixed-decode-steps",
                    "1",
                ],
                "--mixed-decode-steps requires --prefill-budget tbt",
            ),
            (
                vec!["--prefill-budget", "tbt", "--prefill-itl-target-ms", "100"],
                "--prefill-itl-target-ms requires --prefill-budget decode-aware",
            ),
            (
                vec!["--prefill-budget", "tbt", "--decode-tbt-target-ms", "0"],
                "--decode-tbt-target-ms must be 1 to 60000",
            ),
            (
                vec!["--prefill-budget", "tbt", "--mixed-min-prefill-tokens", "0"],
                "--mixed-min-prefill-tokens must be positive",
            ),
            (
                vec!["--prefill-budget", "tbt", "--mixed-decode-steps", "9"],
                "--mixed-decode-steps must be 0 to 8",
            ),
            (
                vec!["--prefill-budget", "slo"],
                "--prefill-budget must be fixed, decode-aware or tbt",
            ),
        ] {
            let mut full = vec!["serve", "--bundle", "a"];
            full.extend_from_slice(&words);
            let refusal = refusal(&full);
            assert!(refusal.contains(expected), "{refusal}");
        }
    }

    #[test]
    fn decode_aware_targets_parse_and_require_decode_aware() {
        use runtime_scheduler::{PrefillBudget, PrefillTarget};
        let budget = |extra: &[&str]| {
            let mut words = vec!["serve", "--bundle", "a"];
            words.extend_from_slice(extra);
            match parse_host_command_v1(&arguments(&words)).unwrap() {
                HostCommandV1::Serve { serving, .. } => serving.prefill_budget,
                _ => panic!("serve"),
            }
        };
        assert_eq!(budget(&[]), PrefillBudget::Fixed);
        assert_eq!(
            budget(&["--prefill-budget", "decode-aware"]),
            PrefillBudget::DecodeAware(PrefillTarget::ItlMillis(200))
        );
        assert_eq!(
            budget(&[
                "--prefill-budget",
                "decode-aware",
                "--prefill-itl-target-ms",
                "150"
            ]),
            PrefillBudget::DecodeAware(PrefillTarget::ItlMillis(150))
        );
        assert_eq!(
            budget(&[
                "--prefill-decode-steps",
                "1",
                "--prefill-budget",
                "decode-aware"
            ]),
            PrefillBudget::DecodeAware(PrefillTarget::DecodeSteps(1))
        );
        for (words, expected) in [
            (
                vec!["--prefill-itl-target-ms", "200"],
                "--prefill-itl-target-ms requires --prefill-budget decode-aware",
            ),
            (
                vec!["--prefill-budget", "fixed", "--prefill-decode-steps", "2"],
                "--prefill-decode-steps requires --prefill-budget decode-aware",
            ),
            (
                vec![
                    "--prefill-budget",
                    "decode-aware",
                    "--prefill-itl-target-ms",
                    "200",
                    "--prefill-decode-steps",
                    "2",
                ],
                "--prefill-itl-target-ms and --prefill-decode-steps are exclusive",
            ),
            (
                vec![
                    "--prefill-budget",
                    "decode-aware",
                    "--prefill-itl-target-ms",
                    "0",
                ],
                "--prefill-itl-target-ms must be 1 to 60000",
            ),
            (
                vec![
                    "--prefill-budget",
                    "decode-aware",
                    "--prefill-itl-target-ms",
                    "60001",
                ],
                "--prefill-itl-target-ms must be 1 to 60000",
            ),
            (
                vec![
                    "--prefill-budget",
                    "decode-aware",
                    "--prefill-decode-steps",
                    "65",
                ],
                "--prefill-decode-steps must be 1 to 64",
            ),
            (
                vec![
                    "--prefill-budget",
                    "decode-aware",
                    "--prefill-decode-steps",
                    "x",
                ],
                "--prefill-decode-steps takes a count",
            ),
        ] {
            let mut all = vec!["serve", "--bundle", "a"];
            all.extend(words);
            let refusal = refusal(&all);
            assert!(refusal.contains(expected), "{refusal}");
        }
        for flag in ["--prefill-itl-target-ms", "--prefill-decode-steps"] {
            assert!(
                refusal(&["conformance-run", "--bundle", "a", flag, "2"])
                    .contains(&format!("does not accept {flag}"))
            );
        }
        assert!(
            super::HOST_COMMAND_USAGE_V1
                .contains("--prefill-itl-target-ms N | --prefill-decode-steps N")
        );
    }

    #[test]
    fn prefill_budget_option_is_explicit_and_serve_only() {
        for (value, expected) in [
            ("fixed", runtime_scheduler::PrefillBudget::Fixed),
            (
                "decode-aware",
                runtime_scheduler::PrefillBudget::DecodeAware(
                    runtime_scheduler::PrefillTarget::default(),
                ),
            ),
        ] {
            let HostCommandV1::Serve { serving, .. } = parse_host_command_v1(&arguments(&[
                "serve",
                "--bundle",
                "a",
                "--prefill-budget",
                value,
            ]))
            .unwrap() else {
                panic!("serve");
            };
            assert_eq!(serving.prefill_budget, expected);
        }
        assert_eq!(
            ServingOptionsV1::default().prefill_budget,
            runtime_scheduler::PrefillBudget::Fixed
        );
        assert!(
            refusal(&["serve", "--bundle", "a", "--prefill-budget", "adaptive"])
                .contains("--prefill-budget must be fixed, decode-aware or tbt")
        );
        assert!(
            refusal(&[
                "conformance-run",
                "--bundle",
                "a",
                "--prefill-budget",
                "fixed"
            ])
            .contains("does not accept --prefill-budget")
        );
        assert!(
            parse_host_command_v1(&arguments(&[
                "serve",
                "--bundle",
                "a",
                "--prefill-budget",
                "fixed",
                "--prefill-budget",
                "decode-aware"
            ]))
            .is_err()
        );
    }

    #[test]
    fn forced_swaps_require_an_enabled_optimistic_tier() {
        for options in [
            vec!["--debug-swap-every", "16"],
            vec!["--kv-offload-gib", "1", "--debug-swap-every", "0"],
            vec![
                "--kv-offload-gib",
                "1",
                "--debug-swap-every",
                "16",
                "--admission",
                "reserved",
            ],
            vec![
                "--kv-offload-gib",
                "1",
                "--debug-swap-every",
                "16",
                "--debug-preempt-every",
                "16",
            ],
        ] {
            let mut args = vec!["serve", "--bundle", "a"];
            args.extend(options);
            assert!(parse_host_command_v1(&arguments(&args)).is_err());
        }
        let HostCommandV1::Serve { serving, .. } = parse_host_command_v1(&arguments(&[
            "serve",
            "--bundle",
            "a",
            "--kv-offload-gib",
            "1",
            "--debug-swap-every",
            "16",
        ]))
        .unwrap() else {
            panic!("serve")
        };
        assert!(serving.debug_swap);
        assert_eq!(
            serving.admission,
            AdmissionPolicy::Optimistic(
                OptimisticAdmission::DEFAULT
                    .with_forced_preemption(16)
                    .unwrap()
            )
        );
    }

    #[test]
    fn offload_options_default_off_and_validate_serve_configuration() {
        let HostCommandV1::Serve { serving, .. } =
            parse_host_command_v1(&arguments(&["serve", "--bundle", "a"])).unwrap()
        else {
            panic!("serve");
        };
        assert_eq!(
            serving.kv_offload,
            crate::offload::KvOffloadOptionsV1::default()
        );
        let HostCommandV1::Serve { serving, .. } = parse_host_command_v1(&arguments(&[
            "serve",
            "--bundle",
            "a",
            "--kv-offload-gib",
            "16",
            "--kv-offload-min-tokens",
            "128",
            "--kv-offload-gbps",
            "10.5",
            "--kv-offload-prefix",
            "off",
        ]))
        .unwrap() else {
            panic!("serve");
        };
        assert_eq!(serving.kv_offload.budget_bytes, 16 * 1024 * 1024 * 1024);
        assert_eq!(serving.kv_offload.minimum_tokens, 128);
        assert_eq!(serving.kv_offload.bytes_per_second, Some(10_500_000_000));
        for options in [
            vec!["--kv-offload-gib", "1", "--admission", "reserved"],
            vec!["--kv-offload-gib", "18446744073709551615"],
            vec!["--kv-offload-gib", "1", "--kv-offload-gib", "2"],
            vec!["--kv-offload-prefix", "on"],
            vec!["--kv-offload-prefix", "yes"],
        ] {
            let mut args = vec!["serve", "--bundle", "a"];
            args.extend(options);
            assert!(parse_host_command_v1(&arguments(&args)).is_err());
        }
        for rate in ["0", "-1", "NaN", "inf", "1e30", "1e-30"] {
            assert!(
                parse_host_command_v1(&arguments(&[
                    "serve",
                    "--bundle",
                    "a",
                    "--kv-offload-gbps",
                    rate
                ]))
                .is_err()
            );
        }
        assert!(
            parse_host_command_v1(&arguments(&[
                "conformance-run",
                "--bundle",
                "a",
                "--kv-offload-gib",
                "1"
            ]))
            .is_err()
        );
    }

    #[test]
    fn admission_defaults_to_optimistic_and_takes_its_limits() {
        let serving = |words: &[&str]| match parse_host_command_v1(&arguments(words)).unwrap() {
            HostCommandV1::Serve { serving, .. } => serving,
            _ => panic!("serve parses as serve"),
        };
        let defaults = serving(&["serve", "--bundle", "a"]);
        assert_eq!(
            defaults.admission,
            AdmissionPolicy::Optimistic(OptimisticAdmission::DEFAULT)
        );
        assert_eq!(
            serving(&["serve", "--bundle", "a", "--admission", "reserved"]).admission,
            AdmissionPolicy::Reserved
        );
        assert_eq!(
            defaults.queue_wait_ms, 0,
            "queued requests wait for capacity"
        );
        assert_eq!(
            serving(&["serve", "--bundle", "a", "--queue-wait-ms", "0"]).queue_wait_ms,
            0
        );
        assert_eq!(
            serving(&["serve", "--bundle", "a", "--admission", "optimistic"]).admission,
            AdmissionPolicy::Optimistic(OptimisticAdmission::DEFAULT)
        );
        assert_eq!(
            serving(&[
                "serve",
                "--bundle",
                "a",
                "--admission",
                "optimistic",
                "--admission-watermark",
                "0.05",
                "--max-preemptions",
                "0",
                "--debug-preempt-every",
                "16",
            ])
            .admission,
            AdmissionPolicy::Optimistic(
                OptimisticAdmission::new(50_000, 0)
                    .unwrap()
                    .with_forced_preemption(16)
                    .unwrap()
            )
        );
        for (words, expected) in [
            (
                vec!["serve", "--bundle", "a", "--admission", "lazy"],
                "--admission must be reserved or optimistic",
            ),
            (
                vec![
                    "serve",
                    "--bundle",
                    "a",
                    "--admission",
                    "reserved",
                    "--max-preemptions",
                    "2",
                ],
                "--max-preemptions requires --admission optimistic",
            ),
            (
                vec![
                    "serve",
                    "--bundle",
                    "a",
                    "--admission",
                    "reserved",
                    "--debug-preempt-every",
                    "4",
                ],
                "--debug-preempt-every requires --admission optimistic",
            ),
            (
                vec![
                    "serve",
                    "--bundle",
                    "a",
                    "--admission",
                    "optimistic",
                    "--admission-watermark",
                    "1",
                ],
                "--admission-watermark must be in [0, 1)",
            ),
            (
                vec![
                    "serve",
                    "--bundle",
                    "a",
                    "--admission",
                    "optimistic",
                    "--debug-preempt-every",
                    "0",
                ],
                "--debug-preempt-every must be positive",
            ),
            (
                vec![
                    "conformance-run",
                    "--bundle",
                    "a",
                    "--fixture-request",
                    "run",
                    "--admission",
                    "optimistic",
                ],
                "conformance-run does not accept --admission",
            ),
        ] {
            let refusal = refusal(&words);
            assert!(refusal.contains(expected), "{refusal}");
        }
    }

    #[test]
    fn the_kv_cache_dtype_flag_is_auto_by_default() {
        let dtype = |words: &[&str]| match parse_host_command_v1(&arguments(words)).unwrap() {
            HostCommandV1::Serve { serving, .. } => serving.kv_cache_dtype,
            _ => panic!("serve parses as serve"),
        };
        assert_eq!(dtype(&["serve", "--bundle", "a"]), KvCacheDtypeV1::Auto);
        for (value, expected) in [
            ("auto", KvCacheDtypeV1::Auto),
            ("bf16", KvCacheDtypeV1::Bf16),
            ("bfloat16", KvCacheDtypeV1::Bf16),
            ("fp8", KvCacheDtypeV1::Fp8),
            ("fp8_e4m3", KvCacheDtypeV1::Fp8),
            ("mixed", KvCacheDtypeV1::Mixed),
            ("mixed_planes", KvCacheDtypeV1::Mixed),
        ] {
            let words = ["serve", "--bundle", "a", "--kv-cache-dtype", value];
            assert_eq!(dtype(&words), expected);
        }
        assert!(
            refusal(&["serve", "--bundle", "a", "--kv-cache-dtype", "fp16"])
                .contains("--kv-cache-dtype must be auto, bf16, fp8 or mixed")
        );
        assert!(
            refusal(&[
                "conformance-run",
                "--bundle",
                "a",
                "--fixture-request",
                "x",
                "--kv-cache-dtype",
                "fp8"
            ])
            .contains("does not accept --kv-cache-dtype")
        );
    }

    #[test]
    fn the_prefix_cache_and_verify_warmup_flags_are_on_or_off_and_default_on() {
        let serving = |words: &[&str]| match parse_host_command_v1(&arguments(words)).unwrap() {
            HostCommandV1::Serve { serving, .. } => serving,
            _ => panic!("serve parses as serve"),
        };
        assert!(serving(&["serve", "--bundle", "a"]).prefix_cache);
        assert!(serving(&["serve", "--bundle", "a", "--prefix-cache", "on"]).prefix_cache);
        assert!(!serving(&["serve", "--bundle", "a", "--prefix-cache", "off"]).prefix_cache);
        assert!(serving(&["serve", "--bundle", "a"]).verify_warmup);
        assert!(!serving(&["serve", "--bundle", "a", "--verify-warmup", "off"]).verify_warmup);
        assert!(
            refusal(&["serve", "--bundle", "a", "--verify-warmup", "no"])
                .contains("--verify-warmup must be on or off")
        );
        assert!(
            refusal(&["serve", "--bundle", "a", "--prefix-cache", "yes"])
                .contains("--prefix-cache must be on or off")
        );
        // `--modalities` selects video; the runtime switch is gone.
        for words in [
            &["serve", "--model", "m", "--video-input", "off"][..],
            &["serve", "--bundle", "a", "--video-input", "on"],
        ] {
            assert!(refusal(words).contains("--video-input is gone: --modalities selects video"));
        }
        assert!(
            refusal(&[
                "conformance-run",
                "--bundle",
                "a",
                "--fixture-request",
                "x",
                "--prefix-cache",
                "on"
            ])
            .contains("does not accept --prefix-cache")
        );
    }

    #[test]
    fn a_declared_prefix_cache_needs_its_callbacks_retirement_and_copy_records() {
        use crate::executor::BatchWireCapabilityV1;
        assert!(
            super::require_prefix_cache_callbacks_v1(
                false,
                false,
                false,
                BatchWireCapabilityV1::V1
            )
            .is_ok()
        );
        assert!(
            super::require_prefix_cache_callbacks_v1(
                true,
                true,
                true,
                BatchWireCapabilityV1::V3Media
            )
            .is_ok()
        );
        assert!(
            super::require_prefix_cache_callbacks_v1(
                true,
                false,
                true,
                BatchWireCapabilityV1::V3Media
            )
            .unwrap_err()
            .contains("no prefix-state callbacks")
        );
        for (retirement, wire) in [
            (false, BatchWireCapabilityV1::V2CopyOnWrite),
            (true, BatchWireCapabilityV1::V1),
        ] {
            assert!(
                super::require_prefix_cache_callbacks_v1(true, true, retirement, wire)
                    .unwrap_err()
                    .contains("copy-on-write")
            );
        }
    }

    #[test]
    fn a_declared_state_slot_restore_needs_its_callback_and_request_state() {
        use super::require_prefix_state_slot_callback_v1 as require;
        assert!(require(false, false, false).is_ok());
        assert!(require(true, true, true).is_ok());
        assert!(
            require(true, false, true)
                .unwrap_err()
                .contains("without a prefix cache and request state")
        );
        assert!(
            require(true, true, false)
                .unwrap_err()
                .contains("restore_prefix_state_in_slot_v1")
        );
    }

    #[test]
    fn speculation_flags_require_a_positive_depth_and_a_known_mode() {
        assert!(
            refusal(&["serve", "--bundle", "a", "--speculation", "fixed"])
                .contains("requires positive --speculative-tokens")
        );
        assert!(
            refusal(&["serve", "--bundle", "a", "--speculation", "unknown"])
                .contains("must be fixed or adaptive")
        );
        let fixed = parse_host_command_v1(&arguments(&[
            "serve",
            "--bundle",
            "a",
            "--speculative-tokens",
            "3",
            "--speculation",
            "fixed",
        ]))
        .unwrap();
        assert!(matches!(
            fixed,
            HostCommandV1::Serve {
                speculative_tokens: 3,
                speculation_mode: SpeculationMode::Fixed,
                ..
            }
        ));
        let off = parse_host_command_v1(&arguments(&["serve", "--bundle", "a"])).unwrap();
        assert!(matches!(
            off,
            HostCommandV1::Serve {
                speculative_tokens: 0,
                ..
            }
        ));
        assert!(
            refusal(&[
                "conformance-run",
                "--bundle",
                "a",
                "--fixture-request",
                "x",
                "--speculative-tokens",
                "2"
            ])
            .contains("does not accept --speculative-tokens")
        );
    }

    #[test]
    fn a_package_loads_the_checkpoint_it_names() {
        let direct = parse_host_command_v1(&arguments(&[
            "serve",
            "--bundle",
            "a",
            "--checkpoint",
            "/tmp/checkpoint",
            "--load-threads",
            "12",
        ]))
        .unwrap();
        assert!(matches!(
            direct,
            HostCommandV1::Serve {
                direct_checkpoint: Some(path),
                load_threads: Some(threads),
                ..
            } if path == std::path::Path::new("/tmp/checkpoint") && threads.get() == 12
        ));
        // A package that plans no weights serves without a checkpoint.
        assert!(matches!(
            parse_host_command_v1(&arguments(&["serve", "--bundle", "a"])).unwrap(),
            HostCommandV1::Serve {
                direct_checkpoint: None,
                ..
            }
        ));
        for (words, reason) in [
            (
                &["serve", "--bundle", "a", "--load", "direct"][..],
                "--load is gone",
            ),
            (
                &["serve", "--bundle", "a", "--checkpoint", ""],
                "--checkpoint needs a directory",
            ),
            (
                &["serve", "--bundle", "a", "--load-threads", "8"],
                "--load-threads requires --model or --checkpoint",
            ),
            (
                &[
                    "serve",
                    "--bundle",
                    "a",
                    "--checkpoint",
                    "/x",
                    "--load-threads",
                    "0",
                ],
                "from 1 to 65535",
            ),
            (
                &["serve", "--model", "/m", "--checkpoint", "/x"],
                "--model loads its own checkpoint",
            ),
        ] {
            assert!(refusal(words).contains(reason), "{words:?}");
        }
    }

    #[test]
    fn reference_forcing_is_an_explicit_serve_flag() {
        let forced = parse_host_command_v1(&arguments(&[
            "serve",
            "--bundle",
            "a",
            "--debug-reference-tokens",
            "/tmp/references.json",
        ]))
        .unwrap();
        assert!(matches!(
            forced,
            HostCommandV1::Serve {
                debug_reference_tokens: Some(path),
                ..
            } if path == std::path::Path::new("/tmp/references.json")
        ));
        let plain = parse_host_command_v1(&arguments(&["serve", "--bundle", "a"])).unwrap();
        assert!(matches!(
            plain,
            HostCommandV1::Serve {
                debug_reference_tokens: None,
                ..
            }
        ));
        assert!(
            refusal(&[
                "conformance-run",
                "--bundle",
                "a",
                "--fixture-request",
                "x",
                "--debug-reference-tokens",
                "/tmp/references.json"
            ])
            .contains("does not accept --debug-reference-tokens")
        );
    }

    /// Runs the serve command in a child process with the retired variable
    /// set, since the environment is process-wide.
    #[test]
    fn serving_refuses_the_retired_reference_forcing_variable() {
        const CHILD: &str = "CHOPINFER_HOST_REFERENCE_VARIABLE_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "command::tests::serving_refuses_the_retired_reference_forcing_variable",
                ])
                .env(CHILD, "1")
                .env(
                    super::REFERENCE_TOKENS_ENV_V1,
                    "/nonexistent/references.json",
                )
                .status()
                .unwrap();
            assert!(status.success(), "child exited with {status}");
            return;
        }
        let command = parse_host_command_v1(&arguments(&[
            "serve",
            "--bundle",
            "/nonexistent/bundle",
            "--fake-host-services",
        ]))
        .unwrap();
        let refusal = super::run_host_command_v1(&command)
            .unwrap_err()
            .to_string();
        assert!(
            refusal.contains(super::REFERENCE_TOKENS_ENV_V1)
                && refusal.contains("--debug-reference-tokens"),
            "{refusal}"
        );
    }

    #[test]
    fn drafter_cost_shape_is_explicit_and_model_neutral() {
        let parsed = parse_host_command_v1(&arguments(&[
            "serve",
            "--bundle",
            "a",
            "--speculative-tokens",
            "7",
            "--draft-cost",
            "per-block",
        ]))
        .unwrap();
        assert!(matches!(
            parsed,
            HostCommandV1::Serve {
                draft_cost_model: runtime_engine::DraftCostModel::PerBlock,
                ..
            }
        ));
        assert!(
            refusal(&["serve", "--bundle", "a", "--draft-cost", "unknown"])
                .contains("must be per-token or per-block")
        );
    }

    #[test]
    fn a_media_position_rule_requires_the_position_capability() {
        use runtime_frontend::{
            FrontendMediaArenaV1, FrontendMediaElementV1, FrontendMediaKindV1,
            FrontendMediaModalityV1, FrontendMediaRecipeV1, FrontendMediaRenderV1,
            FrontendMultimodalV1, FrontendPositionRuleV1,
        };
        let mut media = FrontendMultimodalV1 {
            arena: FrontendMediaArenaV1 {
                rows: 64,
                page_rows: 64,
                hidden: 8,
                element: FrontendMediaElementV1::Bf16,
            },
            encoders: vec![],
            modalities: vec![FrontendMediaModalityV1 {
                kind: FrontendMediaKindV1::Image,
                part_types: vec!["image_url".to_owned()],
                maximum_items: 1,
                maximum_bytes: 64,
                maximum_tokens: 4,
                recipe: FrontendMediaRecipeV1 {
                    tag: "image.patch-grid.v1".to_owned(),
                    parameters: serde_json::json!({ "merge": 2 }),
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
        for capability in [false, true] {
            assert!(super::require_position_axes_capability_v1(None, capability).is_ok());
            assert!(super::require_position_axes_capability_v1(Some(&media), capability).is_ok());
        }
        media.modalities[0].positions = Some(FrontendPositionRuleV1::Grid3AxisV1);
        assert!(super::require_position_axes_capability_v1(Some(&media), true).is_ok());
        let refusal = super::require_position_axes_capability_v1(Some(&media), false).unwrap_err();
        assert!(refusal.contains(crate::executor::POSITION_AXES_CAPABILITY_V1));
    }

    #[test]
    fn enabled_speculation_requires_the_abi_callback_before_create() {
        assert!(
            super::require_speculation_callback_v1(true, false)
                .unwrap_err()
                .contains("commit_accepted_v1")
        );
        assert!(super::require_speculation_callback_v1(false, false).is_ok());
        assert!(super::require_speculation_callback_v1(true, true).is_ok());
    }

    #[test]
    fn help_spelling_as_a_value_does_not_hide_explicit_flags() {
        for words in [
            vec!["serve", "--bundle", "fixture", "--api-key", "-h"],
            vec![
                "serve",
                "--bundle",
                "fixture",
                "--served-model-name",
                "a",
                "-h",
            ],
        ] {
            assert!(matches!(
                parse_host_command_v1(&arguments(&words)).unwrap(),
                HostCommandV1::Serve { .. }
            ));
        }
        assert!(matches!(
            parse_host_command_v1(&arguments(&["serve", "--help"])).unwrap(),
            HostCommandV1::Display { .. }
        ));
    }

    #[test]
    fn shared_valueless_flags_are_parsed_without_consuming_values() {
        for flag in super::VALUELESS_FLAGS_V1 {
            assert!(
                super::ArgumentsV1::parse(&[flag.to_string()]).is_ok(),
                "{flag}"
            );
            assert!(
                super::ArgumentsV1::parse(&[flag.to_string(), flag.to_string()]).is_err(),
                "{flag}"
            );
        }
    }

    #[test]
    fn profile_drafter_is_checked_against_the_verified_manifest() {
        let base = manifest_requiring(
            vec![topology("tp1", 1)],
            8,
            DECLARED_ROWS_V1,
            &"0".repeat(64),
            vec![id("chop.speculative.v1")],
        );
        let declaration =
            runtime_bundle::BundleSpeculationV1::new(3, true, 8, vec![4, 8], Some(id("weights")))
                .unwrap();
        let legacy = base.clone().with_speculation(declaration.clone()).unwrap();
        let named = base
            .with_speculation(declaration.with_drafter_kind(Some(id("first"))).unwrap())
            .unwrap();
        assert!(super::validate_profile_drafter_v1(&named, Some("first"), 3).is_ok());
        let error = super::validate_profile_drafter_v1(&named, Some("second"), 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("'second'") && error.contains("'first'"));
        assert!(
            super::validate_profile_drafter_v1(&legacy, Some("first"), 3)
                .unwrap_err()
                .to_string()
                .contains("unspecified")
        );
        assert!(super::validate_profile_drafter_v1(&named, Some("second"), 0).is_ok());
        assert!(super::validate_profile_drafter_v1(&named, None, 3).is_ok());
        let decoded =
            BundleManifestV1::from_canonical_json(&named.to_canonical_json().unwrap()).unwrap();
        assert_eq!(decoded.speculation().unwrap().drafter_kind(), Some("first"));
        let decoded =
            BundleManifestV1::from_canonical_json(&legacy.to_canonical_json().unwrap()).unwrap();
        assert_eq!(decoded.speculation().unwrap().drafter_kind(), None);
        let plain = manifest(vec![topology("tp1", 1)], 8);
        assert!(super::validate_profile_drafter_v1(&plain, Some("first"), 3).is_err());
    }

    #[test]
    fn speculation_is_off_by_default_and_refuses_a_bundle_without_limits() {
        let manifest = manifest(vec![topology("tp1", 1)], 8);
        assert!(
            super::speculation_config_v1(
                &manifest,
                (0, None),
                32,
                SpeculationMode::Adaptive,
                runtime_engine::DraftCostModel::PerToken
            )
            .unwrap()
            .is_none()
        );
        assert!(
            super::speculation_config_v1(
                &manifest,
                (2, None),
                32,
                SpeculationMode::Fixed,
                runtime_engine::DraftCostModel::PerToken
            )
            .unwrap_err()
            .to_string()
            .contains("requires a bundle speculation declaration")
        );
    }

    /// The drafting-items bound comes from the bundle's declaration; a
    /// declaration without one leaves every item free to draft.
    #[test]
    fn the_drafting_item_bound_is_the_bundles_declaration() {
        for items in [None, Some(32)] {
            let mut speculation = runtime_bundle::BundleSpeculationV1::new(
                3,
                true,
                256,
                vec![2, 4],
                Some(BundleOpaqueId::new("weights").unwrap()),
            )
            .unwrap();
            if let Some(items) = items {
                speculation = speculation.with_maximum_draft_items(items).unwrap();
            }
            let manifest = manifest_requiring(
                vec![topology("tp1", 1)],
                8,
                DECLARED_ROWS_V1,
                &"0".repeat(64),
                vec![BundleOpaqueId::new(crate::executor::SPECULATIVE_CAPABILITY_V1).unwrap()],
            )
            .with_speculation(speculation)
            .unwrap();
            let config = super::speculation_config_v1(
                &manifest,
                (3, None),
                64,
                SpeculationMode::Fixed,
                runtime_engine::DraftCostModel::PerToken,
            )
            .unwrap()
            .unwrap();
            assert_eq!(
                config.maximum_draft_items(),
                items.map(|items| items as usize)
            );
        }
    }

    /// `/health` reports what startup resolved: the profile, the direct
    /// package, and the settings the package serves with.
    #[test]
    fn the_serving_identity_reports_what_startup_resolved() {
        let declaration =
            runtime_bundle::BundleSpeculationV1::new(3, true, 8, vec![4, 8], Some(id("weights")))
                .unwrap()
                .with_drafter_kind(Some(id("first")))
                .unwrap();
        let speculative = manifest_requiring(
            vec![topology("tp1", 1)],
            8,
            DECLARED_ROWS_V1,
            &"0".repeat(64),
            vec![
                id(crate::executor::SPECULATIVE_CAPABILITY_V1),
                id(runtime_bundle_abi::PREFIX_CACHE_CAPABILITY_V1),
            ],
        )
        .with_speculation(declaration)
        .unwrap();
        let speculation = super::speculation_config_v1(
            &speculative,
            (2, None),
            8,
            SpeculationMode::Adaptive,
            runtime_engine::DraftCostModel::PerToken,
        )
        .unwrap();
        let package = super::DirectPackageIdentityV1 {
            package_identity: "a".repeat(64),
            metadata_identity: "c".repeat(64),
        };
        let serving = ServingOptionsV1 {
            maximum_sequences: 8,
            ..ServingOptionsV1::default()
        };
        let identity = super::serving_identity_v1(
            &speculative,
            (Some("fixture-tp4"), Some(&package)),
            (4, None),
            resolve_kv_cache_dtype_v1(&speculative, KvCacheDtypeV1::Auto).unwrap(),
            speculation,
            &serving,
        )
        .unwrap();
        assert_eq!(identity.bundle_id, "fixture.plan.v1");
        assert_eq!(identity.package_identity, Some("a".repeat(64)));
        assert_eq!(identity.metadata_identity, Some("c".repeat(64)));
        // The checkpoint is the one this process selected for direct loads.
        assert_eq!(
            identity.checkpoint_identity,
            super::super::direct::selected_checkpoint_v1()
                .unwrap()
                .map(|checkpoint| checkpoint.identity_hex())
        );
        assert_eq!(
            identity.profile,
            runtime_server::ServingProfile {
                name: Some("fixture-tp4".into()),
                tensor_parallel_size: 4,
                context_parallel_size: None,
                speculation: runtime_server::ServingSpeculation {
                    mode: "adaptive".into(),
                    tokens: 2,
                    drafter: Some("first".into()),
                },
                kv_cache_dtype: "bf16".into(),
                prefix_cache: true,
                max_sequences: 8,
            }
        );
        assert_eq!(
            identity.kv_page_tokens,
            speculative.kv_layout().page_tokens()
        );
        assert_eq!(identity.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(identity.startup_nonce.len(), 32);
        assert!(
            identity
                .startup_nonce
                .bytes()
                .all(|b| b.is_ascii_hexdigit())
        );
        let revision = env!("CHOPINFER_BUILD_REVISION");
        assert_eq!(
            identity.source_commit.as_deref(),
            Some(revision).filter(|revision| !revision.is_empty())
        );

        let first_nonce = identity.startup_nonce;

        // Without speculation, a profile or the prefix capability.
        let plain = manifest(vec![topology("tp1", 1)], 8);
        let identity = super::serving_identity_v1(
            &plain,
            (None, None),
            (1, None),
            resolve_kv_cache_dtype_v1(&plain, KvCacheDtypeV1::Auto).unwrap(),
            None,
            &ServingOptionsV1::default(),
        )
        .unwrap();
        assert_eq!(identity.package_identity, None);
        assert_eq!(identity.metadata_identity, None);
        assert_eq!(identity.kv_page_tokens, 8);
        // Each process draws its own nonce.
        assert_eq!(identity.startup_nonce.len(), 32);
        assert_ne!(identity.startup_nonce, first_nonce);
        assert_eq!(identity.profile.name, None);
        assert_eq!(
            identity.profile.speculation,
            runtime_server::ServingSpeculation {
                mode: "off".into(),
                tokens: 0,
                drafter: None,
            }
        );
        assert!(!identity.profile.prefix_cache);
    }

    #[test]
    fn remote_media_flags_are_explicit_and_bounded() {
        let parse = |extra: &[&str]| {
            let mut args = vec!["serve", "--bundle", "a"];
            args.extend(extra);
            parse_host_command_v1(&arguments(&args))
        };
        let HostCommandV1::Serve { remote_media, .. } = parse(&[]).unwrap() else {
            panic!("serve");
        };
        assert!(remote_media.allowed_domains.is_empty());
        assert!(!remote_media.allow_private_addresses);
        assert_eq!(remote_media.request_timeout.as_secs(), 30);
        let HostCommandV1::Serve { remote_media, .. } = parse(&[
            "--allowed-media-domains",
            "media.example,cdn.example",
            "--media-allow-private-addresses",
            "on",
            "--media-fetch-timeout-ms",
            "1234",
            "--media-request-timeout-ms",
            "2345",
            "--media-connect-timeout-ms",
            "123",
            "--media-max-redirects",
            "0",
        ])
        .unwrap() else {
            panic!("serve");
        };
        assert_eq!(
            remote_media.allowed_domains,
            ["media.example", "cdn.example"]
        );
        assert!(remote_media.allow_private_addresses);
        assert_eq!(remote_media.total_timeout.as_millis(), 1234);
        assert_eq!(remote_media.request_timeout.as_millis(), 2345);
        assert_eq!(remote_media.connect_timeout.as_millis(), 123);
        assert_eq!(remote_media.maximum_redirects, 0);
        for args in [
            ["--allowed-media-domains", "*"],
            ["--allowed-media-domains", "user@host"],
            ["--media-allow-private-addresses", "yes"],
            ["--media-fetch-timeout-ms", "0"],
            ["--media-request-timeout-ms", "0"],
            ["--media-request-timeout-ms", "300001"],
            ["--media-connect-timeout-ms", "300001"],
            ["--media-max-redirects", "11"],
        ] {
            assert!(parse(&args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn steps_are_priced_in_power_of_two_row_buckets() {
        assert_eq!(super::step_buckets_v1(32), [1, 2, 4, 8, 16, 32]);
        assert_eq!(super::step_buckets_v1(33).last(), Some(&64));
        assert_eq!(super::step_buckets_v1(256).last(), Some(&256));
        assert_eq!(super::step_buckets_v1(0), [1]);
    }

    #[test]
    fn served_verify_rows_preserve_small_instances_and_allow_wide_drafting() {
        for (sequences, rows) in [
            (1, 32),
            (32, 32),
            (33, 256),
            (64, 256),
            (65, 256),
            (128, 256),
            (129, 256),
            (256, 256),
        ] {
            assert_eq!(super::served_verify_rows_v1(256, sequences, true), rows);
            assert_eq!(super::served_verify_rows_v1(256, sequences, false), 256);
        }
        assert_eq!(super::served_verify_rows_v1(32, 256, true), 32);
    }

    #[test]
    fn speculative_rows_narrow_the_declared_verify_rows() {
        assert_eq!(super::verify_rows_v1(256, 7, None).unwrap(), 256);
        assert_eq!(super::verify_rows_v1(256, 7, Some(32)).unwrap(), 32);
        for refused in [257, 7] {
            assert!(
                super::verify_rows_v1(256, 7, Some(refused))
                    .unwrap_err()
                    .to_string()
                    .contains("--speculative-rows"),
                "{refused}"
            );
        }
        let parsed = parse_host_command_v1(&arguments(&[
            "serve",
            "--bundle",
            "a",
            "--speculative-tokens",
            "7",
            "--speculative-rows",
            "32",
        ]))
        .unwrap();
        assert!(matches!(
            parsed,
            HostCommandV1::Serve {
                speculative_rows: Some(32_u16),
                ..
            }
        ));
        assert!(
            parse_host_command_v1(&arguments(&[
                "serve",
                "--bundle",
                "a",
                "--speculative-rows",
                "32"
            ]))
            .is_err()
        );
    }

    #[test]
    fn local_media_files_require_an_explicit_serve_flag() {
        let command = parse_host_command_v1(&arguments(&[
            "serve",
            "--bundle",
            "a",
            "--allowed-local-media-path",
            "/tmp/media",
        ]))
        .expect("media root parses");
        let HostCommandV1::Serve {
            allowed_local_media_path,
            ..
        } = command
        else {
            panic!("serve command expected");
        };
        assert_eq!(allowed_local_media_path, Some(PathBuf::from("/tmp/media")));
        assert!(
            refusal(&[
                "serve",
                "--bundle",
                "a",
                "--allowed-local-media-path",
                "/tmp/a",
                "--allowed-local-media-path",
                "/tmp/b"
            ])
            .contains("more than once")
        );
    }

    #[test]
    fn serving_kv_flags_bound_the_host_pool() {
        let parsed = parse_host_command_v1(&arguments(&[
            "serve",
            "--bundle",
            "a",
            "--kv-memory-fraction",
            "0.75",
            "--kv-pages",
            "128",
        ]))
        .unwrap();
        let HostCommandV1::Serve { serving, .. } = parsed else {
            panic!("the command serves");
        };
        assert_eq!(serving.kv_memory_fraction_millionths, 750_000);
        assert_eq!(serving.kv_pages, Some(128));
        for fraction in ["0", "1.1", "NaN", "invalid"] {
            assert!(
                parse_host_command_v1(&arguments(&[
                    "serve",
                    "--bundle",
                    "a",
                    "--kv-memory-fraction",
                    fraction,
                ]))
                .is_err()
            );
        }
        assert!(
            parse_host_command_v1(&arguments(&["serve", "--bundle", "a", "--kv-pages", "0",]))
                .is_err()
        );
    }

    #[test]
    fn conformance_runs_load_a_checkpoint_directly() {
        let words = [
            "conformance-run",
            "--bundle",
            "a",
            "--fixture-request",
            "one",
        ];
        let mut direct = words.to_vec();
        direct.extend(["--checkpoint", "/x"]);
        assert!(matches!(
            parse_host_command_v1(&arguments(&direct)).unwrap(),
            HostCommandV1::ConformanceRun {
                direct_checkpoint: Some(path),
                ..
            } if path == std::path::Path::new("/x")
        ));
        assert!(matches!(
            parse_host_command_v1(&arguments(&words)).unwrap(),
            HostCommandV1::ConformanceRun {
                direct_checkpoint: None,
                ..
            }
        ));
    }

    #[test]
    fn conformance_kv_fraction_bounds_the_host_pool() {
        let parsed = parse_host_command_v1(&arguments(&[
            "conformance-run",
            "--bundle",
            "a",
            "--fixture-request",
            "g7",
            "--kv-memory-fraction",
            "0.5",
        ]))
        .unwrap();
        let HostCommandV1::ConformanceRun {
            kv_memory_fraction_millionths,
            ..
        } = parsed
        else {
            panic!("the command runs conformance");
        };
        assert_eq!(kv_memory_fraction_millionths, 500_000);
    }

    #[test]
    fn fixture_reads_bf16_device_view_as_full_f32_logits_row() {
        use runtime_bundle_abi::{
            BUNDLE_LOGITS_ELEMENT_V1_BF16, BundleDeviceBufferViewV1, BundleLogitsViewV1,
            DeviceBufferSpanV1, HostServicesBackendV1, LogitsRowSinkV1, OwnedLogitsViewV1,
            ResultSinkLimitsV1,
        };

        use crate::fake::FakeHostDeviceV1;
        use crate::services::{HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1};

        let backend = std::rc::Rc::new(HostServicesV1Backend::new(
            FakeHostDeviceV1::new(),
            HostInstanceConfigV1::new(HostTopologyV1::new(vec![0]).unwrap()),
        ));
        let context = backend.select_context(0, 0).unwrap();
        let pool = backend.select_memory_pool(context).unwrap();
        let stream = backend.create_stream(context, 0).unwrap();
        let buffer = backend
            .allocate_device_buffer(pool, stream, 4, 256)
            .unwrap();
        backend
            .copy_host_to_device(
                stream,
                DeviceBufferSpanV1::new(buffer, 0, 4),
                &[0x80, 0x3f, 0x00, 0xc0],
            )
            .unwrap();
        let view = BundleLogitsViewV1 {
            struct_size: std::mem::size_of::<BundleLogitsViewV1>() as u32,
            reserved: 0,
            context,
            rank: 0,
            element_type: BUNDLE_LOGITS_ELEMENT_V1_BF16,
            buffer: BundleDeviceBufferViewV1 {
                struct_size: std::mem::size_of::<BundleDeviceBufferViewV1>() as u32,
                reserved: 0,
                buffer,
                offset: 0,
                length: 4,
            },
            row_stride_elements: 2,
            row_count: 1,
            live_width: 2,
            first_item_index: 0,
            reserved_tail: 0,
        };
        let mut sink = super::FixtureResultSinkV1 {
            backend: std::rc::Rc::clone(&backend),
            limits: ResultSinkLimitsV1::new(1, 2, 2),
            rows: Vec::new(),
        };
        sink.emit_view(OwnedLogitsViewV1::new(view)).unwrap();
        assert_eq!(sink.rows[0].request_id(), super::FIXTURE_REQUEST_ID_V1);
        assert_eq!(
            sink.rows[0].logit_bits(),
            [1.0_f32.to_le_bytes(), (-2.0_f32).to_le_bytes()].concat()
        );
        backend.release_device_buffer(buffer).unwrap();
        backend.release_stream(stream).unwrap();
        backend.release_memory_pool(pool).unwrap();
        backend.release_context(context).unwrap();
    }

    #[test]
    fn explicit_million_token_context_requires_one_full_sequence_of_kv() {
        let config = server_config_v1(
            ServingOptionsV1 {
                maximum_context_tokens: Some(1_048_576),
                ..Default::default()
            },
            RequestSurfaceV1::Full,
            None,
        )
        .unwrap();
        for capacity in [1, 524_288, 1_048_575] {
            let error = super::constrain_server_context_v1(config, 1_048_576, capacity, true)
                .unwrap_err()
                .to_string();
            assert!(error.contains("requested context of 1048576"), "{error}");
            assert!(
                error.contains(&format!("acquired KV capacity of {capacity}")),
                "{error}"
            );
        }
        let accepted =
            super::constrain_server_context_v1(config, 1_048_576, 1_048_576, true).unwrap();
        assert_eq!(accepted.maximum_context_tokens(), Some(1_048_576));
    }

    /// Before the frontend sizes the media limit, every endpoint shares the
    /// text default; an explicit body limit is both limits.
    #[test]
    fn the_body_limit_flag_sets_every_endpoint_limit() {
        let defaults =
            server_config_v1(ServingOptionsV1::default(), RequestSurfaceV1::Full, None).unwrap();
        assert_eq!(defaults.max_body_bytes(), 64 << 20);
        assert_eq!(defaults.max_media_body_bytes(), 64 << 20);
        let explicit = server_config_v1(
            ServingOptionsV1 {
                maximum_request_body_bytes: Some(8 << 20),
                ..Default::default()
            },
            RequestSurfaceV1::Full,
            None,
        )
        .unwrap();
        assert_eq!(explicit.max_body_bytes(), 8 << 20);
        assert_eq!(explicit.max_media_body_bytes(), 8 << 20);
    }

    #[test]
    fn acquired_kv_capacity_bounds_transport_admission() {
        let config = super::constrain_server_context_v1(
            server_config_v1(ServingOptionsV1::default(), RequestSurfaceV1::Full, None).unwrap(),
            1024,
            256,
            false,
        )
        .unwrap();
        assert_eq!(config.maximum_context_tokens(), Some(256));
        let refusal = super::constrain_server_context_v1(config, 1024, 128, true)
            .unwrap_err()
            .to_string();
        assert!(refusal.contains("acquired KV capacity of 128"), "{refusal}");
    }

    /// Every command reads each flag once and refuses what it cannot use.
    #[test]
    fn arguments_are_read_once_each_and_never_borrowed_from_another_command() {
        for (words, expected) in [
            (
                vec!["bundle", "verify", "--bundle", "a", "--bundle", "b"],
                "--bundle is given more than once",
            ),
            (
                vec![
                    "bundle",
                    "verify",
                    "--bundle",
                    "a",
                    "--fixture-request",
                    "run",
                ],
                "bundle verify does not accept --fixture-request",
            ),
            (
                vec!["serve", "--bundle", "a", "--fixture-request", "run"],
                "serve does not accept --fixture-request",
            ),
            (
                vec!["bundle", "verify", "--bundle"],
                "--bundle requires a value",
            ),
            (
                vec!["bundle", "verify", "--unknown", "a"],
                "unsupported argument '--unknown'",
            ),
            (vec!["bundle", "verify"], "bundle verify requires --bundle"),
            (
                vec![
                    "conformance-run",
                    "--bundle",
                    "a",
                    "--fixture-request",
                    "run",
                    "--fixture-tokens",
                    "many",
                ],
                "--fixture-tokens takes a count",
            ),
            (
                vec![
                    "serve",
                    "--bundle",
                    "a",
                    "--fake-host-services",
                    "--fake-host-services",
                ],
                "--fake-host-services is given more than once",
            ),
            (
                vec![
                    "serve",
                    "--bundle",
                    "a",
                    "--capability",
                    "peer-access",
                    "--capability",
                    "peer-access",
                ],
                "--capability peer-access is given more than once",
            ),
        ] {
            let refusal = refusal(&words);
            assert!(refusal.contains(expected), "{refusal}");
        }
    }

    /// A label rides into a receipt, so it is checked before anything loads.
    #[test]
    fn a_fixture_label_is_bounded_and_needs_no_escaping() {
        validate_label_v1("external-bundle-host").expect("the harness label is valid");
        for label in ["", "Upper", "a b", "a\"b", &"a".repeat(65)] {
            let error = validate_label_v1(label).expect_err("this label must be refused");
            assert!(error.to_string().contains("fixture request label"));
        }
    }

    /// The architecture and capabilities a bundle is checked against are the
    /// caller's to state, and the pinned target is what a silent caller means.
    #[test]
    fn the_host_descriptor_is_stated_rather_than_discovered() {
        let assumed = ArgumentsV1::default()
            .host_compatibility()
            .expect("the assumed descriptor is valid");
        assert_eq!(
            assumed.architecture().as_str(),
            DEFAULT_HOST_ARCHITECTURE_V1
        );
        assert_eq!(assumed.abi_version(), 1);
        assert_eq!(
            assumed
                .capabilities()
                .map(BundleOpaqueId::as_str)
                .collect::<Vec<_>>(),
            vec![
                runtime_bundle_abi::frontend::FRONTEND_CAPABILITY_V1,
                runtime_bundle_abi::HOST_MAPPED_GATHER_CAPABILITY_V1,
                runtime_bundle_abi::HOST_RESIDENT_VIEWS_CAPABILITY_V1,
                runtime_bundle_abi::FP8_KV_CAPABILITY_V1,
                runtime_bundle::MIXED_PLANE_KV_CAPABILITY_V1,
                crate::offload::KV_OFFLOAD_CAPABILITY_V1,
                crate::executor::MEDIA_INPUTS_CAPABILITY_V1,
                runtime_bundle_abi::MODEL_OPTIONS_CAPABILITY_V1,
                crate::executor::POSITION_AXES_CAPABILITY_V1,
                runtime_bundle_abi::PREFIX_CACHE_STATE_SLOT_CAPABILITY_V1,
                runtime_bundle_abi::PREFIX_CACHE_CAPABILITY_V1,
                crate::executor::PROMPT_ROWS_CAPABILITY_V1,
                runtime_bundle_abi::RECURRENT_STATE_IN_PLACE_CAPABILITY_V1,
                runtime_bundle_abi::RECURRENT_STATE_PREALLOCATED_CAPABILITY_V1,
                runtime_bundle_abi::RECURRENT_STATE_CAPABILITY_V1,
                runtime_bundle_abi::REQUEST_RETIREMENT_CAPABILITY_V1,
                runtime_bundle_abi::SERVED_CONTEXT_CAPABILITY_V1,
                runtime_bundle_abi::SERVED_ITEMS_CAPABILITY_V1,
                runtime_bundle_abi::SERVING_KV_CAPABILITY_V1,
                runtime_bundle_abi::SPECULATIVE_NARROW_ROWS_CAPABILITY_V1,
                crate::executor::SPECULATIVE_CAPABILITY_V1,
                runtime_bundle_abi::STATE_PER_ITEM_CAPABILITY_V1,
                runtime_bundle_abi::weight_plan::WEIGHT_PLAN_CAPABILITY_V1,
            ]
        );

        let stated = ArgumentsV1 {
            architecture: Some("sm_90a".to_owned()),
            capabilities: vec!["peer-access".to_owned(), "collective".to_owned()],
            ..ArgumentsV1::default()
        }
        .host_compatibility()
        .expect("the stated descriptor is valid");
        assert_eq!(stated.architecture().as_str(), "sm_90a");
        assert_eq!(
            stated.capabilities().collect::<Vec<_>>(),
            vec![
                &BundleOpaqueId::new(runtime_bundle_abi::frontend::FRONTEND_CAPABILITY_V1).unwrap(),
                &BundleOpaqueId::new(runtime_bundle_abi::HOST_MAPPED_GATHER_CAPABILITY_V1).unwrap(),
                &BundleOpaqueId::new(runtime_bundle_abi::HOST_RESIDENT_VIEWS_CAPABILITY_V1)
                    .unwrap(),
                &BundleOpaqueId::new(runtime_bundle_abi::FP8_KV_CAPABILITY_V1).unwrap(),
                &BundleOpaqueId::new(runtime_bundle::MIXED_PLANE_KV_CAPABILITY_V1).unwrap(),
                &BundleOpaqueId::new(crate::offload::KV_OFFLOAD_CAPABILITY_V1).unwrap(),
                &BundleOpaqueId::new(crate::executor::MEDIA_INPUTS_CAPABILITY_V1).unwrap(),
                &BundleOpaqueId::new(runtime_bundle_abi::MODEL_OPTIONS_CAPABILITY_V1).unwrap(),
                &BundleOpaqueId::new(crate::executor::POSITION_AXES_CAPABILITY_V1).unwrap(),
                &BundleOpaqueId::new(runtime_bundle_abi::PREFIX_CACHE_STATE_SLOT_CAPABILITY_V1)
                    .unwrap(),
                &BundleOpaqueId::new(runtime_bundle_abi::PREFIX_CACHE_CAPABILITY_V1).unwrap(),
                &BundleOpaqueId::new(crate::executor::PROMPT_ROWS_CAPABILITY_V1).unwrap(),
                &BundleOpaqueId::new(runtime_bundle_abi::RECURRENT_STATE_IN_PLACE_CAPABILITY_V1)
                    .unwrap(),
                &BundleOpaqueId::new(
                    runtime_bundle_abi::RECURRENT_STATE_PREALLOCATED_CAPABILITY_V1
                )
                .unwrap(),
                &BundleOpaqueId::new(runtime_bundle_abi::RECURRENT_STATE_CAPABILITY_V1).unwrap(),
                &BundleOpaqueId::new(runtime_bundle_abi::REQUEST_RETIREMENT_CAPABILITY_V1).unwrap(),
                &BundleOpaqueId::new(runtime_bundle_abi::SERVED_CONTEXT_CAPABILITY_V1).unwrap(),
                &BundleOpaqueId::new(runtime_bundle_abi::SERVED_ITEMS_CAPABILITY_V1).unwrap(),
                &BundleOpaqueId::new(runtime_bundle_abi::SERVING_KV_CAPABILITY_V1).unwrap(),
                &BundleOpaqueId::new(runtime_bundle_abi::SPECULATIVE_NARROW_ROWS_CAPABILITY_V1)
                    .unwrap(),
                &BundleOpaqueId::new(crate::executor::SPECULATIVE_CAPABILITY_V1).unwrap(),
                &BundleOpaqueId::new(runtime_bundle_abi::STATE_PER_ITEM_CAPABILITY_V1).unwrap(),
                &BundleOpaqueId::new(runtime_bundle_abi::weight_plan::WEIGHT_PLAN_CAPABILITY_V1)
                    .unwrap(),
                &BundleOpaqueId::new("collective").expect("the test capability is valid"),
                &BundleOpaqueId::new("peer-access").expect("the test capability is valid"),
            ]
        );

        let refused = ArgumentsV1 {
            architecture: Some("SM_120A".to_owned()),
            ..ArgumentsV1::default()
        }
        .host_compatibility()
        .expect_err("an unusable architecture identifier is refused");
        assert!(matches!(refused, HostCommandErrorV1::Bundle(_)));
    }

    #[test]
    fn required_retirement_is_checked_before_device_setup() {
        let legacy = manifest(vec![topology("tp1", 1)], 8);
        let mut json: serde_json::Value =
            serde_json::from_slice(&legacy.to_canonical_json().unwrap()).unwrap();
        json["resource_requirements"]["required_capabilities"] =
            serde_json::json!([runtime_bundle_abi::REQUEST_RETIREMENT_CAPABILITY_V1]);
        let required =
            BundleManifestV1::from_canonical_json(&serde_json::to_vec(&json).unwrap()).unwrap();
        for (manifest, present, accepted) in [
            (&legacy, false, true),
            (&required, true, true),
            (&required, false, false),
        ] {
            let opened = std::cell::Cell::new(false);
            let result = super::open_with_request_retirement_v1(manifest, present, || {
                opened.set(true);
                Ok(())
            });
            assert_eq!(opened.get(), accepted);
            assert_eq!(result.is_ok(), accepted);
            if !accepted {
                assert!(result.unwrap_err().contains("has no callback"));
            }
        }
    }

    struct RetirementExecution {
        trace: std::rc::Rc<std::cell::RefCell<Vec<String>>>,
        fail_retirement: bool,
    }

    impl crate::BundleExecutionV1 for RetirementExecution {
        fn execute_batch(
            &mut self,
            _: &[u8],
            _: runtime_bundle_abi::ResultSinkLimitsV1,
        ) -> Result<Vec<runtime_bundle_abi::OwnedLogitsRowV1>, runtime_bundle_abi::BundleAbiError>
        {
            self.trace.borrow_mut().push("execute".to_owned());
            Ok(vec![runtime_bundle_abi::OwnedLogitsRowV1::new(
                7,
                [0.0_f32, 1.0, 0.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            )])
        }
        fn commit(&mut self) -> Result<(), runtime_bundle_abi::BundleAbiError> {
            self.trace.borrow_mut().push("commit".to_owned());
            Ok(())
        }
        fn cancel(&mut self) -> Result<(), runtime_bundle_abi::BundleAbiError> {
            self.trace.borrow_mut().push("cancel".to_owned());
            Ok(())
        }
        fn destroy(&mut self) -> Result<(), runtime_bundle_abi::BundleAbiError> {
            Ok(())
        }
        fn is_poisoned(&self) -> bool {
            false
        }
        fn supports_request_retirement(&self) -> bool {
            true
        }
        fn retire_request(&mut self, id: u64) -> Result<(), runtime_bundle_abi::BundleAbiError> {
            self.trace.borrow_mut().push(format!("retire {id}"));
            if self.fail_retirement {
                Err(runtime_bundle_abi::BundleAbiError::Lifecycle(
                    "retirement fixture",
                ))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn all_executor_wrappers_forward_offload_and_preserve_rejected_plans() {
        use runtime_engine::{KvOffloadInfo, ModelExecutor};
        use runtime_kv::*;
        use std::cell::{Cell, RefCell};
        use std::collections::BTreeMap;
        use std::rc::Rc;
        type Status = runtime_bundle_abi::BundleStatusV1;
        #[derive(Default)]
        struct Copies {
            submissions: RefCell<BTreeMap<KvTransferId, KvTransferSubmission>>,
            unavailable: Cell<bool>,
            reject: Cell<bool>,
            fail_poll: Cell<bool>,
            fail_drain: Cell<bool>,
            calls: RefCell<Vec<&'static str>>,
        }
        impl crate::offload::KvOffloadExecutorV1 for Copies {
            fn available(&self) -> bool {
                !self.unavailable.get()
            }
            fn submit(&self, p: KvTransferPlan) -> Result<(), (Status, KvTransferPlan)> {
                self.calls.borrow_mut().push("submit");
                if self.reject.get() {
                    return Err((Status::InvalidArgument, p));
                }
                self.submissions
                    .borrow_mut()
                    .insert(p.id(), p.into_submission());
                Ok(())
            }
            fn poll(&self, id: KvTransferId) -> Result<KvTransferPoll, Status> {
                self.calls.borrow_mut().push("poll");
                if self.fail_poll.get() {
                    return Err(Status::InternalError);
                }
                Ok(KvTransferPoll::Finished(
                    self.submissions.borrow()[&id].completion(KvTransferStatus::Complete),
                ))
            }
            fn wait(&self, id: KvTransferId) -> Result<KvTransferPoll, Status> {
                self.calls.borrow_mut().push("wait");
                self.poll(id)
            }
            fn drain_timeout(&self) -> std::time::Duration {
                std::time::Duration::from_secs(77)
            }
            fn drain(&self) -> Result<(), Status> {
                self.calls.borrow_mut().push("drain");
                if self.fail_drain.get() {
                    Err(Status::InternalError)
                } else {
                    Ok(())
                }
            }
        }
        let copies = Rc::new(Copies::default());
        let info = KvOffloadInfo {
            host_page_capacity: 16,
            minimum_tokens: 64,
            snapshot_pages: 2,
        };
        let served = super::ServedExecutorV1 {
            executor: crate::BundleModelExecutorV1::new(
                RetirementExecution {
                    trace: Rc::new(RefCell::new(Vec::new())),
                    fail_retirement: false,
                },
                3,
                1,
            )
            .with_kv_offload(copies.clone(), info),
            _services: Rc::new(()),
            pool_control: None,
            reference: None,
        };
        let mut executor = runtime_engine::PromptLookupExecutor::new(
            served,
            runtime_engine::PromptLookupConfig::default(),
        );
        assert_eq!(
            executor.kv_transfer_drain_timeout_v1(),
            std::time::Duration::from_secs(77)
        );
        assert_eq!(executor.kv_offload_info_v1(), Some(info));
        let mut pool = KvPool::new(KvGeometry::new(4, 32).unwrap(), 4).unwrap();
        let mut host = HostTier::new(&pool, 16).unwrap();
        let seq = pool.create_sequence(12).unwrap();
        pool.append(seq, 5).unwrap();
        let key = PrefixCacheKeyV1::new([3; 32]);
        pool.insert_private_anchor(key, seq, 2).unwrap();
        let (anchor, plan) = host.begin_offload(&mut pool, key, Some(1)).unwrap();
        let id = plan.id();
        let (error, plan) = executor.reload_pages_v1(plan).unwrap_err();
        assert!(!error.is_fatal());
        assert!(copies.calls.borrow().is_empty());
        copies.reject.set(true);
        let (error, plan) = executor.offload_pages_v1(plan).unwrap_err();
        assert!(!error.is_fatal());
        copies.reject.set(false);
        executor.offload_pages_v1(plan).unwrap();
        copies.fail_poll.set(true);
        assert!(executor.poll_kv_transfer_v1(id).is_err());
        copies.fail_poll.set(false);
        host.complete_transfer(
            &mut pool,
            executor
                .poll_kv_transfer_v1(id)
                .unwrap()
                .completion()
                .unwrap(),
        )
        .unwrap();
        let dest = pool.create_initialized_sequence(5, 12, 2).unwrap();
        let plan = host.begin_reload(&mut pool, anchor, &dest).unwrap();
        let id = plan.id();
        let (_, plan) = executor.offload_pages_v1(plan).unwrap_err();
        executor.reload_pages_v1(plan).unwrap();
        host.complete_transfer(
            &mut pool,
            executor
                .wait_kv_transfer_v1(id)
                .unwrap()
                .completion()
                .unwrap(),
        )
        .unwrap();
        pool.finish_initialized_restore(dest.handle()).unwrap();
        copies.unavailable.set(true);
        assert_eq!(executor.kv_offload_info_v1(), None);
        executor.drain_kv_transfers_v1().unwrap();
        copies.fail_drain.set(true);
        assert!(executor.drain_kv_transfers_v1().unwrap_err().is_fatal());
        assert!(copies.calls.borrow().contains(&"wait"));
        pool.release(dest.handle()).unwrap();
        pool.remove_anchor(key).unwrap();
        pool.release(seq).unwrap();
        host.drop_anchor(anchor).unwrap();
        host.audit(&pool).unwrap();
        assert_eq!(pool.stats().free_blocks(), 32);
    }

    #[test]
    fn served_executor_forwards_retirement_and_failure_after_commit() {
        for fail_retirement in [false, true] {
            let trace = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
            let executor = super::ServedExecutorV1 {
                executor: crate::BundleModelExecutorV1::new(
                    RetirementExecution {
                        trace: trace.clone(),
                        fail_retirement,
                    },
                    3,
                    1,
                ),
                _services: std::rc::Rc::new(()),
                pool_control: None,
                reference: None,
            };
            assert!(executor.executor.supports_request_retirement());
            let geometry = KvGeometry::new(4, 4).unwrap();
            let vocabulary = ModelVocabulary::new(3).unwrap();
            let config = EngineConfig::new(
                SchedulerConfig::new(1, 2, 4, 4, 1).unwrap(),
                geometry,
                vocabulary,
                geometry.token_capacity(),
                1,
            )
            .unwrap();
            let mut engine = runtime_engine::Engine::new(config, executor).unwrap();
            engine
                .submit(
                    runtime_engine::EngineRequest::new(
                        runtime_engine::RequestId::new(7),
                        vec![0],
                        1,
                        runtime_sampling::SamplingParameters::new(0.0).unwrap(),
                        1,
                        Vec::new(),
                        vocabulary,
                    )
                    .unwrap(),
                )
                .unwrap();
            assert_eq!(engine.step().is_err(), fail_retirement);
            assert_eq!(*trace.borrow(), vec!["execute", "commit", "retire 7"]);
            if fail_retirement {
                assert_eq!(
                    engine.step(),
                    Err(runtime_engine::EngineError::RetirementFailed)
                );
                assert_eq!(*trace.borrow(), vec!["execute", "commit", "retire 7"]);
            }
        }
    }

    #[test]
    fn served_executor_uses_prompt_lookup_for_host_drafts() {
        let trace = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut executor = super::ServedExecutorV1 {
            executor: crate::BundleModelExecutorV1::new(
                RetirementExecution {
                    trace,
                    fail_retirement: false,
                },
                3,
                1,
            ),
            _services: std::rc::Rc::new(()),
            pool_control: None,
            reference: None,
        };
        let history = [1, 2, 3, 1, 2];
        assert_eq!(
            runtime_engine::ModelExecutor::draft_tokens_v1(
                &mut executor,
                runtime_engine::RequestId::new(7),
                &history,
                2,
            )
            .unwrap(),
            vec![3, 1]
        );
    }

    #[test]
    fn served_executor_drafts_the_reference_continuation_when_forcing() {
        let trace = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut executor = super::ServedExecutorV1 {
            executor: crate::BundleModelExecutorV1::new(
                RetirementExecution {
                    trace,
                    fail_retirement: false,
                },
                3,
                1,
            ),
            _services: std::rc::Rc::new(()),
            pool_control: None,
            reference: Some(std::rc::Rc::new(crate::ReferenceTokensV1::new(vec![vec![
                1, 2, 3, 1, 2, 0, 0, 2,
            ]]))),
        };
        let mut draft = |history: &[u32], maximum| {
            runtime_engine::ModelExecutor::draft_tokens_v1(
                &mut executor,
                runtime_engine::RequestId::new(7),
                history,
                maximum,
            )
            .unwrap()
        };
        // Prompt lookup would propose [3, 1]; the reference continues [0, 0, 2].
        assert_eq!(draft(&[1, 2, 3, 1, 2], 2), vec![0, 0]);
        assert_eq!(draft(&[1, 2, 3, 1, 2], 7), vec![0, 0, 2]);
        assert!(draft(&[1, 2, 3, 1, 1], 2).is_empty());
    }

    fn id(value: &str) -> BundleOpaqueId {
        BundleOpaqueId::new(value).expect("the test identifier is valid")
    }

    /// One manifest that declares the given topologies and page size.
    ///
    /// Nothing here is a model: the identifiers are invented, and the run reads
    /// only what any manifest declares.
    fn manifest(topologies: Vec<BundleTopologyV1>, page_tokens: u32) -> BundleManifestV1 {
        manifest_with_component_digest(topologies, page_tokens, DECLARED_ROWS_V1, &"0".repeat(64))
    }

    /// The rows one sequence may hold, when a test does not say otherwise.
    const DECLARED_ROWS_V1: u64 = 1024;

    /// One manifest whose KV layout declares this page size and this row count.
    fn manifest_with_rows(page_tokens: u32, declared_rows: u64) -> BundleManifestV1 {
        manifest_with_component_digest(
            vec![topology("tp1", 1)],
            page_tokens,
            declared_rows,
            &"0".repeat(64),
        )
    }

    fn manifest_with_component_digest(
        topologies: Vec<BundleTopologyV1>,
        page_tokens: u32,
        declared_rows: u64,
        component_sha256: &str,
    ) -> BundleManifestV1 {
        manifest_requiring(
            topologies,
            page_tokens,
            declared_rows,
            component_sha256,
            Vec::new(),
        )
    }

    fn manifest_requiring(
        topologies: Vec<BundleTopologyV1>,
        page_tokens: u32,
        declared_rows: u64,
        component_sha256: &str,
        capabilities: Vec<BundleOpaqueId>,
    ) -> BundleManifestV1 {
        BundleManifestV1::new(
            id("fixture.plan.v1"),
            BundleTargetV1::new(vec![id("sm_120f")], topologies).expect("the test target is valid"),
            vec![
                BundleComponentV1::new(
                    id("weights"),
                    id("fixture.weights.v1"),
                    id("fixture.bytes.v1"),
                    BundleRelativePathV1::new("components/weights.bin")
                        .expect("the test path is relative"),
                    16,
                    component_sha256,
                )
                .expect("the test component is valid"),
            ],
            BundleResourceRequirementsV1::new(1, capabilities)
                .expect("the test requirements are valid"),
            BundleKvLayoutV1::new(
                id("fixture.kv.v1"),
                u64::from(page_tokens) * 64,
                64,
                page_tokens,
                64,
                1,
                declared_rows,
                BundleRankPlacementV1::Replicated,
                BundleStateLifetimeV1::Transaction,
            )
            .expect("the test KV layout is valid"),
            BundleStateLayoutV1::new(
                id("fixture.state.v1"),
                64,
                64,
                BundleStateLifetimeV1::Request,
            )
            .expect("the test state layout is valid"),
            vec![
                BundleGraphBucketV1::new(id("bucket.one"), vec![id("weights")], Vec::new())
                    .expect("the test bucket is valid"),
            ],
            Vec::new(),
            Vec::new(),
            None,
        )
        .expect("the test manifest is valid")
    }

    fn kv_formats(manifest: BundleManifestV1, preferred: BundleKvDtypeV1) -> BundleManifestV1 {
        let bf16 = manifest.kv_layout().clone();
        let fp8 = BundleKvLayoutV1::new(
            id("fixture.kv-fp8.v1"),
            u64::from(bf16.page_tokens()) * 32,
            32,
            bf16.page_tokens(),
            32,
            1,
            bf16.maximum_sequence_tokens(),
            BundleRankPlacementV1::Replicated,
            BundleStateLifetimeV1::Transaction,
        )
        .expect("the FP8 test layout is valid");
        let formats = vec![
            BundleKvFormatV1::new(BundleKvDtypeV1::Bf16, bf16.clone()),
            BundleKvFormatV1::new(BundleKvDtypeV1::Fp8E4m3, fp8.clone()),
        ];
        let json = String::from_utf8(manifest.to_canonical_json().unwrap()).unwrap();
        let preferred_layout = if preferred == BundleKvDtypeV1::Bf16 {
            &bf16
        } else {
            &fp8
        };
        let json = json.replace(
            &String::from_utf8(serde_json::to_vec(&bf16).unwrap()).unwrap(),
            &String::from_utf8(serde_json::to_vec(preferred_layout).unwrap()).unwrap(),
        );
        BundleManifestV1::from_canonical_json(json.as_bytes())
            .expect("the relaid manifest is valid")
            .with_kv_formats(formats)
            .expect("both KV formats are valid")
    }

    #[test]
    fn only_an_fp8_served_layout_publishes_the_fp8_kv_gauges() {
        let published = |dtype| super::kv_metrics_v1(dtype).snapshot().prometheus();
        let fp8 = published(BundleKvDtypeV1::Fp8E4m3);
        let bf16 = published(BundleKvDtypeV1::Bf16);
        let mixed = published(BundleKvDtypeV1::MixedPlanes);
        for gauge in ["chop_fp8_kv_tokens", "chop_fp8_kv_capacity_tokens"] {
            assert!(fp8.contains(gauge), "{fp8}");
            assert!(!bf16.contains(gauge), "{bf16}");
            assert!(!mixed.contains(gauge), "{mixed}");
        }
        assert!(bf16.contains("chop_kv_capacity_tokens"));
    }

    #[test]
    fn auto_serves_the_packages_preferred_kv_layout_and_explicit_dtypes_must_be_declared() {
        use crate::kv_cache_dtype::kv_layout_of_v1;
        let resolve = |manifest: &BundleManifestV1, requested| {
            resolve_kv_cache_dtype_v1(manifest, requested).map(|choice| choice.dtype)
        };
        let (bf16, fp8) = (BundleKvDtypeV1::Bf16, BundleKvDtypeV1::Fp8E4m3);
        // A manifest without the list, and one that predates it with an FP8
        // layout, each serve their one layout.
        let plain = manifest_with_rows(16, DECLARED_ROWS_V1);
        assert_eq!(resolve(&plain, KvCacheDtypeV1::Auto), Ok(bf16));
        assert_eq!(resolve(&plain, KvCacheDtypeV1::Bf16), Ok(bf16));
        let refusal = resolve(&plain, KvCacheDtypeV1::Fp8).unwrap_err();
        assert!(
            refusal.contains("declares no fp8 KV layout; it supports bf16"),
            "{refusal}"
        );
        let legacy_fp8 = manifest_requiring(
            vec![topology("tp1", 1)],
            16,
            DECLARED_ROWS_V1,
            &"0".repeat(64),
            vec![id(runtime_bundle_abi::FP8_KV_CAPABILITY_V1)],
        );
        assert_eq!(resolve(&legacy_fp8, KvCacheDtypeV1::Auto), Ok(fp8));
        assert!(resolve(&legacy_fp8, KvCacheDtypeV1::Bf16).is_err());
        // Both declared: auto follows the preference, explicit either way.
        let prefers_bf16 = kv_formats(manifest_with_rows(16, DECLARED_ROWS_V1), bf16);
        let prefers_fp8 = kv_formats(manifest_with_rows(16, DECLARED_ROWS_V1), fp8);
        assert_eq!(resolve(&prefers_bf16, KvCacheDtypeV1::Auto), Ok(bf16));
        assert_eq!(resolve(&prefers_fp8, KvCacheDtypeV1::Auto), Ok(fp8));
        for manifest in [&prefers_bf16, &prefers_fp8] {
            assert_eq!(resolve(manifest, KvCacheDtypeV1::Bf16), Ok(bf16));
            assert_eq!(resolve(manifest, KvCacheDtypeV1::Fp8), Ok(fp8));
            assert_eq!(
                kv_layout_of_v1(manifest, fp8).map(|layout| layout.id().as_str()),
                Some("fixture.kv-fp8.v1")
            );
        }
        let described = resolve_kv_cache_dtype_v1(&prefers_bf16, KvCacheDtypeV1::Auto)
            .unwrap()
            .describe(&prefers_bf16);
        assert_eq!(
            described,
            "serving KV cache dtype: bf16 (auto, the package's preference; the package supports bf16, fp8)"
        );
    }

    /// `manifest` preferring a mixed-plane layout of two planes, with its BF16
    /// layout as the other format.
    fn mixed_planes(manifest: BundleManifestV1) -> BundleManifestV1 {
        use runtime_bundle::{
            BundleKvCodeV1, BundleKvPlaneV1, BundleKvScaleFormatV1, BundleKvScaleV1,
            MIXED_PLANE_KV_CAPABILITY_V1,
        };
        let bf16 = manifest.kv_layout().clone();
        let native = BundleKvLayoutV1::new(
            id("fixture.kv-native.v1"),
            u64::from(bf16.page_tokens()) * 24,
            8,
            bf16.page_tokens(),
            24,
            1,
            bf16.maximum_sequence_tokens(),
            BundleRankPlacementV1::Replicated,
            BundleStateLifetimeV1::Transaction,
        )
        .expect("the native test layout is valid");
        let scale = |format, block| Some(BundleKvScaleV1::new(format, block).unwrap());
        let planes = vec![
            BundleKvPlaneV1::new(
                id("window"),
                BundleKvCodeV1::Fp8E4m3,
                scale(BundleKvScaleFormatV1::Ue8m0, 32),
            )
            .unwrap(),
            BundleKvPlaneV1::new(
                id("compressed"),
                BundleKvCodeV1::Fp4E2m1,
                scale(BundleKvScaleFormatV1::E4m3, 16),
            )
            .unwrap(),
        ];
        let mut value = serde_json::to_value(&manifest).unwrap();
        let capabilities = value["resource_requirements"]["required_capabilities"]
            .as_array_mut()
            .unwrap();
        capabilities.push(MIXED_PLANE_KV_CAPABILITY_V1.into());
        capabilities.sort_by(|left, right| left.as_str().cmp(&right.as_str()));
        value["kv_layout"] = serde_json::to_value(&native).unwrap();
        serde_json::from_value::<BundleManifestV1>(value)
            .unwrap()
            .with_kv_formats(vec![
                BundleKvFormatV1::new(BundleKvDtypeV1::Bf16, bf16),
                BundleKvFormatV1::mixed_planes(planes, native).unwrap(),
            ])
            .expect("both KV formats are valid")
    }

    #[test]
    fn mixed_selects_a_declared_mixed_plane_layout_and_records_its_planes() {
        use crate::kv_cache_dtype::kv_layout_of_v1;
        let resolve = |manifest: &BundleManifestV1, requested| {
            resolve_kv_cache_dtype_v1(manifest, requested).map(|choice| choice.dtype)
        };
        let (bf16, mixed) = (BundleKvDtypeV1::Bf16, BundleKvDtypeV1::MixedPlanes);
        let native = mixed_planes(manifest_with_rows(16, DECLARED_ROWS_V1));
        assert_eq!(resolve(&native, KvCacheDtypeV1::Auto), Ok(mixed));
        assert_eq!(resolve(&native, KvCacheDtypeV1::Mixed), Ok(mixed));
        assert_eq!(resolve(&native, KvCacheDtypeV1::Bf16), Ok(bf16));
        assert!(resolve(&native, KvCacheDtypeV1::Fp8).is_err());
        assert_eq!(
            kv_layout_of_v1(&native, mixed).map(|layout| layout.id().as_str()),
            Some("fixture.kv-native.v1")
        );
        let described = resolve_kv_cache_dtype_v1(&native, KvCacheDtypeV1::Auto)
            .unwrap()
            .describe(&native);
        assert_eq!(
            described,
            "serving KV cache dtype: mixed [window fp8_e4m3/ue8m0x32, compressed fp4_e2m1/e4m3x16] \
             (auto, the package's preference; the package supports bf16, mixed)"
        );
        // A BF16 choice of the same package records no planes.
        let described = resolve_kv_cache_dtype_v1(&native, KvCacheDtypeV1::Bf16)
            .unwrap()
            .describe(&native);
        assert_eq!(
            described,
            "serving KV cache dtype: bf16 (requested; the package supports bf16, mixed)"
        );
        // Packages without the element refuse it by name.
        for manifest in [
            manifest_with_rows(16, DECLARED_ROWS_V1),
            kv_formats(manifest_with_rows(16, DECLARED_ROWS_V1), bf16),
        ] {
            let refusal = resolve(&manifest, KvCacheDtypeV1::Mixed).unwrap_err();
            assert!(refusal.contains("declares no mixed KV layout"), "{refusal}");
        }
        // This host serves the element; the capability it requires is the
        // host's own.
        let host = ArgumentsV1::default().host_compatibility().unwrap();
        assert!(
            native
                .capabilities_of_every_topology()
                .all(|capability| { host.capabilities().any(|offered| offered == capability) })
        );
    }

    fn topology(name: &str, ranks: u32) -> BundleTopologyV1 {
        BundleTopologyV1::new(id(name), ranks).expect("the test topology is valid")
    }

    fn request(prefill_tokens: Option<usize>) -> FixtureRequestV1 {
        FixtureRequestV1 {
            label: "external-bundle-host".to_owned(),
            prefill_tokens,
            scenario: FixtureScenarioV1::Commit,
        }
    }

    fn host() -> BundleHostCompatibilityV1 {
        BundleHostCompatibilityV1::new(DEFAULT_HOST_ARCHITECTURE_V1, 1)
            .expect("the test host descriptor is valid")
    }

    /// A run serves one topology the bundle declares: the one it names, the
    /// only one, or the one with a rank per device it names. It never
    /// guesses.
    #[test]
    fn a_run_names_one_of_the_topologies_the_bundle_declares() {
        let single = manifest(vec![topology("tp1", 1)], 8);
        let several = manifest(vec![topology("tp1", 1), topology("tp4", 4)], 8);

        assert_eq!(
            resolve_topology_v1(&single, None, None)
                .expect("one declared topology needs no argument")
                .id()
                .as_str(),
            "tp1"
        );
        assert_eq!(
            resolve_topology_v1(&several, Some("tp4"), Some(1))
                .expect("a named topology resolves")
                .id()
                .as_str(),
            "tp4"
        );
        assert_eq!(
            resolve_topology_v1(&several, None, Some(4))
                .expect("one topology has a rank per device")
                .id()
                .as_str(),
            "tp4"
        );

        for devices in [None, Some(2)] {
            let unnamed = resolve_topology_v1(&several, None, devices)
                .expect_err("two declared topologies need an argument")
                .to_string();
            assert!(
                unnamed.contains(
                    "--topology ID or one --devices entry per rank must name one: tp1, tp4"
                ),
                "{unnamed}"
            );
        }
        let undeclared = resolve_topology_v1(&single, Some("tp4"), None)
            .expect_err("an undeclared topology resolves to nothing")
            .to_string();
        assert!(
            undeclared.contains("declares no topology 'tp4'"),
            "{undeclared}"
        );
    }

    /// `--context-parallel-size` defaults to 1, must divide the topology's
    /// ranks and must be a size the bundle declares for that topology.
    #[test]
    fn a_context_parallel_size_is_one_the_topology_declares() {
        let parsed = |words: &[&str]| {
            let mut all = vec!["serve", "--bundle", "fixture"];
            all.extend_from_slice(words);
            match parse_host_command_v1(&arguments(&all)) {
                Ok(HostCommandV1::Serve { serving, .. }) => Ok(serving.context_parallel_size),
                Ok(_) => panic!("this invocation names serve"),
                Err(error) => Err(error.to_string()),
            }
        };
        assert_eq!(parsed(&[]), Ok(1));
        assert_eq!(parsed(&["--context-parallel-size", "2"]), Ok(2));
        assert!(
            parsed(&["--context-parallel-size", "0"])
                .unwrap_err()
                .contains("at least 1")
        );

        let declared = topology("tp4", 4)
            .with_context_parallel_sizes(vec![2])
            .unwrap();
        let manifest = manifest(vec![topology("tp1", 1), declared], 8);
        let served = |id: &str, size| {
            super::ServedTopologyV1 {
                id: id.to_owned(),
                device_ordinals: Vec::new(),
                context_parallel_size: 1,
            }
            .with_context_parallel(&manifest, size)
            .map(|topology| topology.context_parallel_size)
            .map_err(|error| error.to_string())
        };
        assert_eq!(served("tp4", 1), Ok(1));
        assert_eq!(served("tp4", 2), Ok(2));
        assert_eq!(served("tp1", 1), Ok(1));
        // The identity reports the size, 1 included, only where groups are declared.
        let reported = |id: &str, size| {
            super::ServedTopologyV1 {
                id: id.to_owned(),
                device_ordinals: Vec::new(),
                context_parallel_size: 1,
            }
            .with_context_parallel(&manifest, size)
            .and_then(|topology| topology.reported_context_parallel(&manifest))
            .unwrap()
        };
        assert_eq!(reported("tp4", 1), Some(1));
        assert_eq!(reported("tp4", 2), Some(2));
        assert_eq!(reported("tp1", 1), None);
        let undeclared = served("tp4", 4).unwrap_err();
        assert!(
            undeclared.contains("declares context-parallel size(s) 1, 2 for topology tp4"),
            "{undeclared}"
        );
        let uneven = served("tp4", 3).unwrap_err();
        assert!(
            uneven.contains("does not divide the 4 tensor-parallel rank(s)"),
            "{uneven}"
        );
        assert!(served("tp1", 2).unwrap_err().contains("does not divide"));
    }

    /// The engine's logical page is a stripe of one declared page per group
    /// rank, so a group of 2 over 16-token pages sizes the engine as 32-token
    /// pages would, and a developer envelope stays at size 1.
    #[test]
    fn a_context_parallel_engine_pages_in_stripes() {
        let vocabulary = ModelVocabulary::new(128).expect("the test vocabulary is valid");
        let striped = ServingOptionsV1 {
            context_parallel_size: 2,
            ..ServingOptionsV1::default()
        };
        assert_eq!(
            engine_config_v1(
                &manifest_with_rows(16, 1024),
                vocabulary,
                1 << 20,
                striped,
                None
            )
            .unwrap(),
            engine_config_v1(
                &manifest_with_rows(32, 1024),
                vocabulary,
                1 << 20,
                ServingOptionsV1::default(),
                None
            )
            .unwrap()
        );
        let manifest = manifest_with_rows(16, 1024);
        let plan = resolve_developer_serving_plan_v1(
            &manifest,
            1024,
            DeveloperServingEnvelopeV1::new(512, 512).unwrap(),
        )
        .unwrap();
        let refusal = engine_config_v1(&manifest, vocabulary, 1024, striped, Some(plan))
            .unwrap_err()
            .to_string();
        assert!(refusal.contains("--context-parallel-size 1"), "{refusal}");
    }

    /// The batch is the bundle's declared page, and one device per rank.
    #[test]
    fn the_fixture_batch_takes_what_the_manifest_declared() {
        let manifest = manifest(vec![topology("tp4", 4)], 8);

        let plan = FixturePlanV1::new(&manifest, &host(), None, &request(None))
            .expect("the declared page makes one batch");

        assert_eq!(plan.prefill_tokens, 8);
        assert_eq!(plan.topology_id, "tp4");
        assert_eq!(plan.rank_count, 4);
        assert_eq!(
            plan.device_ordinals().expect("four ranks fit"),
            vec![0, 1, 2, 3]
        );
        assert!(
            plan.request.starts_with(BATCH_REQUEST_MAGIC_V1),
            "the batch is encoded in the host's own wire form"
        );
    }

    /// A package that loaded has to be the package the deployment was read from.
    ///
    /// The frontend, the tokenizer, the vocabulary, and the engine sizing all
    /// came out of one manifest on one pass. A second bundle at the same path
    /// verifies just as well and means something else, so the check is over
    /// the whole validated manifest rather than over the identifier it happens
    /// to carry.
    #[test]
    fn a_package_that_changed_between_verification_and_loading_is_refused() {
        let selection = BundleSelectionV1::Directory(PathBuf::from("/opt/bundles/fixture.plan.v1"));
        let deployed = manifest(vec![topology("tp1", 1)], 8);

        same_selection_v1(&deployed, &deployed.clone(), &selection)
            .expect("the same package loads");

        // One component's bytes differ. Everything else, including the bundle
        // identifier the receipt would print, is what it was.
        let swapped = manifest_with_component_digest(
            vec![topology("tp1", 1)],
            8,
            DECLARED_ROWS_V1,
            &"1".repeat(64),
        );
        assert_eq!(swapped.id(), deployed.id());
        let refusal = same_selection_v1(&deployed, &swapped, &selection)
            .expect_err("a different package must be refused");
        assert!(
            refusal.contains("changed between verification and loading"),
            "{refusal}"
        );
        assert!(
            refusal.contains("/opt/bundles/fixture.plan.v1"),
            "{refusal}"
        );

        // So does a package whose page layout the engine was not sized for.
        let resized = manifest(vec![topology("tp1", 1)], 16);
        assert!(same_selection_v1(&deployed, &resized, &selection).is_err());
    }

    #[test]
    fn missing_alias_does_not_consume_the_authentication_flag() {
        let args = [
            "serve",
            "--bundle",
            "fixture",
            "--served-model-name",
            "--api-key",
            "secret",
        ]
        .map(String::from);
        assert!(super::parse_host_command_v1(&args).is_err());
    }

    #[test]
    fn served_aliases_and_request_id_flag_parse() {
        let args = [
            "serve",
            "--bundle",
            "fixture",
            "--served-model-name",
            "primary",
            "alias",
            "--enable-request-id-headers",
        ]
        .map(String::from);
        let super::HostCommandV1::Serve { http_options, .. } =
            super::parse_host_command_v1(&args).unwrap()
        else {
            panic!("expected serve")
        };
        assert_eq!(http_options.served_model_names, ["primary", "alias"]);
        assert!(http_options.enable_request_id_headers);
        for names in [vec!["".into()], vec!["same".into(), "same".into()]] {
            assert!(
                runtime_server::HttpOptions {
                    served_model_names: names,
                    ..Default::default()
                }
                .validate()
                .is_err()
            );
        }
    }

    #[test]
    fn api_key_precedence_and_redaction() {
        use runtime_server::ApiKey;
        let cli = Some(ApiKey::new("cli"));
        assert_eq!(
            super::resolve_api_key_v1(cli.clone(), Some("native".into()), Some("compat".into())),
            cli
        );
        assert_eq!(
            super::resolve_api_key_v1(None, Some("native".into()), Some("compat".into())),
            Some(ApiKey::new("native"))
        );
        assert_eq!(
            super::resolve_api_key_v1(None, None, Some("compat".into())),
            Some(ApiKey::new("compat"))
        );
        assert_eq!(
            super::resolve_api_key_v1(None, Some(String::new()), None),
            None
        );
        assert_eq!(format!("{:?}", ApiKey::new("secret")), "ApiKey([redacted])");
    }

    #[test]
    fn every_cors_flag_enables_cors_but_other_flags_do_not() {
        for extra in [
            vec![],
            vec!["--enable-request-id-headers"],
            vec!["--allowed-origins", "[]"],
            vec!["--allowed-methods", "[\"POST\"]"],
            vec!["--allowed-headers", "[]"],
            vec!["--allow-credentials"],
        ] {
            let expected = extra
                .first()
                .is_some_and(|flag| flag.starts_with("--allow"));
            let args: Vec<String> = ["serve", "--bundle", "fixture"]
                .into_iter()
                .chain(extra)
                .map(String::from)
                .collect();
            let super::HostCommandV1::Serve { http_options, .. } =
                super::parse_host_command_v1(&args).unwrap()
            else {
                panic!("expected serve")
            };
            assert_eq!(http_options.enable_cors, expected);
        }
    }

    #[test]
    fn cors_flags_parse_and_are_serve_only() {
        let args = [
            "serve",
            "--bundle",
            "fixture",
            "--allowed-origins",
            r#"["https://client.example"]"#,
            "--allowed-methods",
            r#"["GET","POST"]"#,
            "--allowed-headers",
            r#"["authorization"]"#,
            "--allow-credentials",
        ]
        .map(String::from);
        let parsed = super::parse_host_command_v1(&args);
        assert!(parsed.is_ok(), "{parsed:?}");
    }

    /// Transport and scheduler agree on live plus queued request capacity.
    #[test]
    fn admission_is_bounded_by_the_same_number_the_engine_is_sized_from() {
        let defaults = ServerConfig::default();

        let assumed = server_config_v1(ServingOptionsV1::default(), RequestSurfaceV1::Full, None)
            .expect("the default serves");
        assert_eq!(
            assumed.max_in_flight(),
            DEFAULT_SERVED_SEQUENCES_V1 + super::PREPARED_QUEUE_DEPTH_V1
        );
        assert_eq!(
            assumed.event_buffer_capacity(),
            defaults.event_buffer_capacity()
        );
        assert_eq!(assumed.max_body_bytes(), defaults.max_body_bytes());
        assert_eq!(assumed.max_output_bytes(), defaults.max_output_bytes());
        assert_eq!(assumed.maximum_context_tokens(), None);

        let narrowed = server_config_v1(
            ServingOptionsV1 {
                maximum_context_tokens: Some(64),
                maximum_sequences: 3,
                maximum_batch_tokens: None,
                ..ServingOptionsV1::default()
            },
            RequestSurfaceV1::Full,
            None,
        )
        .expect("a narrowed deployment serves");
        assert_eq!(narrowed.max_in_flight(), 3 + super::PREPARED_QUEUE_DEPTH_V1);
        assert_eq!(narrowed.maximum_context_tokens(), Some(64));

        // The transport's own maximum still holds, and so does its floor.
        for (sequences, expected) in [
            (MAX_IN_FLIGHT_REQUESTS + 1, "exceeds the supported maximum"),
            (0, "must admit at least one"),
        ] {
            let refusal = server_config_v1(
                ServingOptionsV1 {
                    maximum_sequences: sequences,
                    ..ServingOptionsV1::default()
                },
                RequestSurfaceV1::Full,
                None,
            )
            .expect_err("this concurrency must be refused")
            .to_string();
            assert!(
                refusal.contains("--max-sequences") || refusal.contains("serving request limits"),
                "{refusal}"
            );
            assert!(refusal.contains(expected), "{refusal}");
        }

        let refusal = server_config_v1(
            ServingOptionsV1 {
                maximum_context_tokens: Some(0),
                ..ServingOptionsV1::default()
            },
            RequestSurfaceV1::Full,
            None,
        )
        .expect_err("a deployment that serves no context is refused")
        .to_string();
        assert!(refusal.contains("--max-context-tokens"), "{refusal}");
    }

    #[test]
    fn queue_bounds_are_explicit_and_zero_disables_waiting() {
        let default =
            server_config_v1(ServingOptionsV1::default(), RequestSurfaceV1::Full, None).unwrap();
        assert!(default.has_unbounded_queue());
        for depth in [0, 4] {
            let config = server_config_v1(
                ServingOptionsV1 {
                    queue_depth: Some(depth),
                    ..Default::default()
                },
                RequestSurfaceV1::Full,
                None,
            )
            .unwrap();
            assert!(!config.has_unbounded_queue());
            assert_eq!(config.max_in_flight(), DEFAULT_SERVED_SEQUENCES_V1 + depth);
        }
    }

    /// The surface a program names is the one the transport is built with.
    ///
    /// Nothing else about the deployment moves with it: the same command, run
    /// twice, differs only in what a chat request may contain. The host holds
    /// no evidence about any bundle, so it neither chooses this nor narrows
    /// what it was given.
    #[test]
    fn a_named_request_surface_is_the_one_the_transport_serves() {
        for serving in [
            ServingOptionsV1::default(),
            ServingOptionsV1 {
                maximum_context_tokens: Some(256),
                maximum_sequences: 4,
                maximum_batch_tokens: Some(32),
                ..ServingOptionsV1::default()
            },
        ] {
            for surface in [
                RequestSurfaceV1::Full,
                RequestSurfaceV1::SingleTurnPlainChat,
            ] {
                let config =
                    server_config_v1(serving, surface, None).expect("this deployment is servable");
                assert_eq!(config.request_surface(), surface);
                assert_eq!(
                    config.max_in_flight(),
                    serving.maximum_sequences
                        + serving.queue_depth.unwrap_or(PREPARED_QUEUE_DEPTH_V1)
                );
                assert_eq!(
                    config.maximum_context_tokens(),
                    serving.maximum_context_tokens
                );
            }
        }

        // A command that names no surface serves every one the transport
        // validates, which is what every existing caller already gets.
        assert_eq!(
            ServerConfig::default().request_surface(),
            RequestSurfaceV1::Full
        );
        assert_eq!(RequestSurfaceV1::default(), RequestSurfaceV1::Full);
    }

    /// One sequence of the shape a developer envelope admits.
    fn envelope_serving() -> ServingOptionsV1 {
        ServingOptionsV1 {
            maximum_context_tokens: None,
            maximum_sequences: 1,
            maximum_batch_tokens: Some(32),
            ..ServingOptionsV1::default()
        }
    }

    fn envelope(prompt_tokens: usize, generated_tokens: usize) -> DeveloperServingEnvelopeV1 {
        DeveloperServingEnvelopeV1::new(prompt_tokens, generated_tokens)
            .expect("the test envelope is constructible")
    }

    /// A developer envelope charges one token more than the rows it writes.
    ///
    /// The last generated token is emitted from a row that already exists, so a
    /// budget of 34 tokens over a bundle that declares 33 rows of model state
    /// writes exactly the 33 rows it declared, and reserves the pages those
    /// rows already needed.
    #[test]
    fn a_developer_envelope_charges_one_token_more_than_it_writes() {
        let manifest = manifest_with_rows(32, 33);

        let plan = resolve_developer_serving_plan_v1(&manifest, 200_000, envelope(32, 2))
            .expect("a prompt of 32 answered with 2 is within the declared rows");
        assert_eq!(plan.total_tokens(), 34);
        assert_eq!(plan.consumed_rows(), 33);
        assert_eq!(plan.declared_rows(), 33);
        assert_eq!(plan.page_tokens(), 32);
        assert_eq!(plan.pages_per_sequence(), 2);
        // The pages are the ones the declared rows already needed, so the extra
        // token buys no storage.
        assert_eq!(plan.pages_per_sequence(), 33_usize.div_ceil(32));

        // A shorter answer is the same deployment, one token inside its bound.
        let shorter = resolve_developer_serving_plan_v1(&manifest, 200_000, envelope(32, 1))
            .expect("a prompt of 32 answered with 1 is within the declared rows");
        assert_eq!(shorter.total_tokens(), 33);
        assert_eq!(shorter.consumed_rows(), 32);
        assert_eq!(shorter.pages_per_sequence(), plan.pages_per_sequence());
    }

    /// Every bound a developer envelope has is the bundle's own declaration.
    #[test]
    fn a_developer_envelope_is_refused_by_what_the_bundle_declares() {
        let manifest = manifest_with_rows(32, 33);

        // One more generated token writes one more row than the bundle has.
        let rows = resolve_developer_serving_plan_v1(&manifest, 200_000, envelope(32, 3))
            .expect_err("34 rows over a declared 33 must be refused")
            .to_string();
        assert!(rows.contains("writes 34 row(s) of model state"), "{rows}");
        assert!(rows.contains("declares 33"), "{rows}");

        // Spending the same budget on the prompt instead writes them just as
        // much, so the refusal is about rows rather than about which half grew.
        let longer_prompt = resolve_developer_serving_plan_v1(&manifest, 200_000, envelope(33, 2))
            .expect_err("34 rows over a declared 33 must be refused")
            .to_string();
        assert!(
            longer_prompt.contains("writes 34 row(s) of model state"),
            "{longer_prompt}"
        );

        // The served declaration is the other bound, and it is checked rather
        // than assumed to be wider than the budget.
        let declared = resolve_developer_serving_plan_v1(&manifest, 33, envelope(32, 2))
            .expect_err("a declaration offering 33 tokens cannot be charged 34")
            .to_string();
        assert!(declared.contains("charges 34 token(s)"), "{declared}");
        assert!(declared.contains("offers 33"), "{declared}");
        assert!(
            resolve_developer_serving_plan_v1(&manifest, 34, envelope(32, 2)).is_ok(),
            "a declaration that offers exactly the budget serves it"
        );

        // A bundle whose declared rows fill its pages exactly has no room for
        // the extra token, and reserving another page would be an allocation
        // its declaration never accounted for.
        let exact = manifest_with_rows(32, 32);
        let pages = resolve_developer_serving_plan_v1(&exact, 200_000, envelope(31, 2))
            .expect_err("a page the declared rows do not need must be refused")
            .to_string();
        assert!(pages.contains("reserve 2 page(s) per sequence"), "{pages}");
        assert!(pages.contains("account for 1"), "{pages}");
        assert!(
            resolve_developer_serving_plan_v1(&exact, 200_000, envelope(30, 2)).is_ok(),
            "31 tokens over a declared 32 rows fit the page those rows needed"
        );

        // An envelope that admits nothing is refused before a bundle is read.
        for (prompt_tokens, generated_tokens) in [(0, 2), (32, 0)] {
            assert!(DeveloperServingEnvelopeV1::new(prompt_tokens, generated_tokens).is_err());
        }
        assert!(DeveloperServingEnvelopeV1::new(usize::MAX, 1).is_err());
    }

    /// A bundle that declares anchor items keeps its kept-anchor flags on a
    /// V3 wire even without media or speculation; the same prompt-rows
    /// manifest without them keeps V2.
    #[test]
    fn declared_anchor_items_select_a_v3_wire() {
        let prompt_rows = manifest_requiring(
            vec![topology("tp1", 1)],
            64,
            DECLARED_ROWS_V1,
            &"0".repeat(64),
            vec![id(crate::executor::PROMPT_ROWS_CAPABILITY_V1)],
        );
        let limits = BundleBatchLimitsV1::new(32, 8192).unwrap();
        let plain = prompt_rows.clone().with_batch_limits(limits).unwrap();
        assert_eq!(
            crate::executor::batch_wire_capability_v1(&plain),
            crate::executor::BatchWireCapabilityV1::V2PromptRows
        );
        let anchored = prompt_rows
            .with_batch_limits(limits.with_anchor_items(2).unwrap())
            .unwrap();
        assert_eq!(
            crate::executor::batch_wire_capability_v1(&anchored),
            crate::executor::BatchWireCapabilityV1::V3Speculative
        );
    }

    /// Serving options the bundle cannot execute are refused before serving.
    ///
    /// A bundle that runs 32 sequences and prefills 8192 tokens per batch
    /// would otherwise start and then refuse the first batch that exceeds
    /// either, under load.
    #[test]
    fn serving_options_beyond_the_bundle_batch_limits_are_refused() {
        let unlimited = manifest_with_rows(64, 1 << 20);
        let limited = unlimited
            .clone()
            .with_batch_limits(BundleBatchLimitsV1::new(32, 8192).unwrap())
            .unwrap();
        let vocabulary = ModelVocabulary::new(128).expect("the test vocabulary is valid");
        let serving = |sequences, batch, chunk| ServingOptionsV1 {
            maximum_sequences: sequences,
            maximum_batch_tokens: batch,
            prefill_chunk_tokens: chunk,
            ..ServingOptionsV1::default()
        };
        let config = |manifest: &BundleManifestV1, options| {
            engine_config_v1(manifest, vocabulary, 1 << 20, options, None)
        };
        for options in [
            ServingOptionsV1::default(),
            serving(16, None, None),
            serving(32, None, None),
            serving(32, Some(8192), Some(8192)),
        ] {
            config(&limited, options).expect("the bundle runs this schedule");
        }
        for (options, flag) in [
            (serving(33, None, None), "--max-sequences"),
            (serving(32, Some(8193), None), "--max-batch-tokens"),
            // Eight default chunks fill the budget past the prefill arena.
            (serving(8, Some(16_384), None), "--max-batch-tokens"),
            (serving(1, Some(16_384), Some(16_384)), "--max-batch-tokens"),
        ] {
            let refusal = config(&limited, options)
                .expect_err("the bundle cannot run this schedule")
                .to_string();
            assert!(refusal.contains(flag), "{refusal}");
        }
        // A bundle that declares no limits keeps the options it was given.
        assert!(config(&unlimited, serving(33, Some(16_384), Some(16_384))).is_ok());
        // The worker queues everything the in-flight limit admits, and waits
        // without a limit unless one is configured.
        let backend = serving_backend_config_v1(&ServingOptionsV1::default()).unwrap();
        assert_eq!(
            backend.control_capacity(),
            DEFAULT_SERVED_SEQUENCES_V1 + PREPARED_QUEUE_DEPTH_V1
        );
        assert_eq!(backend.queue_wait(), std::time::Duration::ZERO);
        let deep = ServingOptionsV1 {
            queue_depth: Some(4_000),
            queue_wait_ms: 250,
            ..ServingOptionsV1::default()
        };
        let backend = serving_backend_config_v1(&deep).unwrap();
        assert_eq!(
            backend.control_capacity(),
            DEFAULT_SERVED_SEQUENCES_V1 + 4_000
        );
        assert_eq!(backend.queue_wait(), std::time::Duration::from_millis(250));
        // The admission policy reaches the scheduler.
        let options = ServingOptionsV1 {
            admission: AdmissionPolicy::Reserved,
            ..ServingOptionsV1::default()
        };
        assert_eq!(
            config(&limited, options).unwrap().admission(),
            AdmissionPolicy::Reserved
        );
        let options = ServingOptionsV1 {
            prefill_budget: runtime_scheduler::PrefillBudget::DecodeAware(
                runtime_scheduler::PrefillTarget::DecodeSteps(3),
            ),
            ..ServingOptionsV1::default()
        };
        assert_eq!(
            config(&limited, options).unwrap().prefill_budget(),
            runtime_scheduler::PrefillBudget::DecodeAware(
                runtime_scheduler::PrefillTarget::DecodeSteps(3)
            )
        );
        assert_eq!(
            config(&limited, ServingOptionsV1::default())
                .unwrap()
                .admission(),
            AdmissionPolicy::Optimistic(OptimisticAdmission::DEFAULT)
        );
        // The scheduler bounds scored phases at the manifest's row limit, or
        // at the default for manifests that declare none.
        let scored = unlimited
            .clone()
            .with_batch_limits(
                BundleBatchLimitsV1::new(32, 8192)
                    .unwrap()
                    .with_maximum_scored_rows(256)
                    .unwrap(),
            )
            .unwrap();
        for (manifest, rows) in [
            (&scored, 256),
            (&limited, runtime_bundle::DEFAULT_MAXIMUM_SCORED_ROWS),
            (&unlimited, runtime_bundle::DEFAULT_MAXIMUM_SCORED_ROWS),
        ] {
            assert_eq!(
                config(manifest, ServingOptionsV1::default())
                    .unwrap()
                    .maximum_scored_rows(),
                rows as usize
            );
        }
    }

    /// The engine a developer envelope sizes holds the whole charged budget.
    ///
    /// Without an envelope the engine is clamped to the rows the bundle
    /// declares, which is what every existing deployment gets. With one it
    /// holds the budget that plan resolved, over the same pages either way.
    #[test]
    fn a_developer_envelope_sizes_the_engine_to_the_budget_it_resolved() {
        let manifest = manifest_with_rows(32, 33);
        let vocabulary = ModelVocabulary::new(128).expect("the test vocabulary is valid");
        let plan = resolve_developer_serving_plan_v1(&manifest, 200_000, envelope(32, 2))
            .expect("the envelope resolves");

        let sized = |sequence_tokens| {
            EngineConfig::new(
                SchedulerConfig::new(1, 2 + super::PREPARED_QUEUE_DEPTH_V1, 32, 32, 1)
                    .expect("the test schedule is valid")
                    .with_maximum_scored_rows(runtime_bundle::DEFAULT_MAXIMUM_SCORED_ROWS as usize)
                    .expect("the default scored row limit is valid")
                    .with_capacity_profile(
                        SchedulerCapacityProfile::new("served", 1 + super::PREPARED_QUEUE_DEPTH_V1)
                            .expect("the served profile is valid"),
                    )
                    .expect("the test schedule fits the profile")
                    .with_admission(AdmissionPolicy::Optimistic(OptimisticAdmission::DEFAULT)),
                KvGeometry::new(32, 2).expect("the test geometry is valid"),
                vocabulary,
                sequence_tokens,
                1,
            )
            .expect("the test engine configuration is valid")
        };

        assert_eq!(
            engine_config_v1(
                &manifest,
                vocabulary,
                200_000,
                envelope_serving(),
                Some(plan)
            )
            .expect("the resolved plan sizes an engine"),
            sized(34)
        );
        // The default path is unchanged: the declared rows are the ceiling.
        assert_eq!(
            engine_config_v1(&manifest, vocabulary, 200_000, envelope_serving(), None)
                .expect("the default path sizes an engine"),
            sized(33)
        );
    }

    #[test]
    fn bounded_envelope_configures_the_transport_with_generic_limits() {
        let envelope = DeveloperServingEnvelopeV1::bounded(255, 64, 256).unwrap();
        let config =
            server_config_v1(envelope_serving(), RequestSurfaceV1::Full, Some(envelope)).unwrap();
        assert_eq!(
            config.acceptance_profile(),
            RequestAcceptanceProfile::Bounded {
                maximum_prompt_tokens: 255,
                maximum_generated_tokens: 64,
                maximum_total_tokens: 256,
            }
        );
        assert_eq!(config.maximum_context_tokens(), Some(256));
        assert_eq!(
            config.request_surface(),
            RequestSurfaceV1::SingleTurnPlainChat
        );
        assert!(
            server_config_v1(
                ServingOptionsV1 {
                    maximum_context_tokens: Some(257),
                    ..envelope_serving()
                },
                RequestSurfaceV1::Full,
                Some(envelope)
            )
            .is_err()
        );
    }

    /// A developer envelope configures the shape, the surface, and the budget.
    #[test]
    fn a_developer_envelope_configures_one_deployment_rather_than_three_bounds() {
        let config = server_config_v1(
            envelope_serving(),
            RequestSurfaceV1::SingleTurnPlainChat,
            Some(envelope(32, 2)),
        )
        .expect("the envelope configures a transport");
        assert_eq!(
            config.acceptance_profile(),
            RequestAcceptanceProfile::ExactShape {
                prompt_tokens: 32,
                maximum_generated_tokens: 2,
            }
        );
        assert_eq!(
            config.request_surface(),
            RequestSurfaceV1::SingleTurnPlainChat
        );
        assert_eq!(config.maximum_context_tokens(), Some(34));
        assert_eq!(config.max_in_flight(), 1 + super::PREPARED_QUEUE_DEPTH_V1);

        // Naming the same budget twice says the same thing twice.
        assert_eq!(
            server_config_v1(
                ServingOptionsV1 {
                    maximum_context_tokens: Some(34),
                    ..envelope_serving()
                },
                RequestSurfaceV1::SingleTurnPlainChat,
                Some(envelope(32, 2)),
            )
            .expect("a context limit equal to the budget is the budget")
            .maximum_context_tokens(),
            Some(34)
        );

        // Naming a different one is two answers to the same question. The
        // declared rows are the tempting wrong answer, so that is the one here.
        let conflict = server_config_v1(
            ServingOptionsV1 {
                maximum_context_tokens: Some(33),
                ..envelope_serving()
            },
            RequestSurfaceV1::SingleTurnPlainChat,
            Some(envelope(32, 2)),
        )
        .expect_err("a second context limit beside an envelope must be refused")
        .to_string();
        assert!(conflict.contains("charges 34 token(s)"), "{conflict}");
        assert!(conflict.contains("also asks to serve 33"), "{conflict}");
    }

    /// A caller may size the batch, within what the bundle declared.
    #[test]
    fn a_named_token_count_is_bounded_by_the_declared_sequence_limit() {
        let manifest = manifest(vec![topology("tp1", 1)], 8);

        let named = FixturePlanV1::new(&manifest, &host(), None, &request(Some(20)))
            .expect("a named count spanning three pages is prepared");
        assert_eq!(named.prefill_tokens, 20);

        for (tokens, expected) in [
            (0, "at least one token"),
            (1025, "declares at most 1024 sequence tokens"),
        ] {
            let refusal = FixturePlanV1::new(&manifest, &host(), None, &request(Some(tokens)))
                .expect_err("this count must be refused")
                .to_string();
            assert!(refusal.contains(expected), "{refusal}");
        }
    }
}

#[cfg(test)]
mod refusal_config_tests {
    use super::batch_refusal_limits_v1;

    #[test]
    fn refusal_limits_default_override_and_reject_invalid_configuration() {
        assert_eq!(
            batch_refusal_limits_v1(|_| None).unwrap(),
            runtime_engine::BatchRefusalLimits::DEFAULT
        );
        let limits = batch_refusal_limits_v1(|name| {
            Some(if name.ends_with("SECONDS") { "15" } else { "2" }.to_owned())
        })
        .unwrap();
        assert_eq!(limits.maximum_consecutive, 2);
        assert_eq!(limits.maximum_in_window, 2);
        assert_eq!(limits.window_seconds, 15);
        for invalid in ["-1", "oops", "100001"] {
            assert!(batch_refusal_limits_v1(|_| Some(invalid.to_owned())).is_err());
        }
        assert!(batch_refusal_limits_v1(|_| Some("0".to_owned())).is_err());
        assert_eq!(
            batch_refusal_limits_v1(|name| (!name.ends_with("SECONDS")).then(|| "0".to_owned()))
                .unwrap()
                .maximum_consecutive,
            0
        );
    }
}

#[cfg(test)]
mod model_option_tests {
    use super::*;
    #[test]
    fn environment_options_validate_and_explicit_names_take_precedence() {
        for value in ["", "budget", "budget=8,budget=4", "bad name=1"] {
            assert!(model_options_environment_v1(Some(value)).is_err());
        }
        let defaults = model_options_environment_v1(Some("budget=8,mode=fast")).unwrap();
        let explicit = ["--model-option", "budget=4"].map(String::from);
        let merged = crate::profiles::apply_defaults(&defaults, &explicit);
        let parsed = ArgumentsV1::parse(&merged).unwrap();
        assert_eq!(
            parsed.model_options.get("budget").map(String::as_str),
            Some("4")
        );
        assert_eq!(
            parsed.model_options.get("mode").map(String::as_str),
            Some("fast")
        );
    }

    #[test]
    fn model_options_are_repeatable_and_serve_only() {
        let args = [
            "serve",
            "--bundle",
            "fixture",
            "--model-option",
            "budget=8",
            "--model-option",
            "mode=fast",
        ]
        .map(String::from);
        assert!(parse_host_command_v1(&args).is_ok());
        for words in [
            vec!["serve", "--bundle", "fixture", "--model-option", "budget"],
            vec![
                "serve",
                "--bundle",
                "fixture",
                "--model-option",
                "budget=8",
                "--model-option",
                "budget=4",
            ],
            vec![
                "bundle",
                "verify",
                "--bundle",
                "fixture",
                "--model-option",
                "budget=8",
            ],
        ] {
            assert!(
                parse_host_command_v1(&words.into_iter().map(String::from).collect::<Vec<_>>())
                    .is_err()
            );
        }
    }
}
