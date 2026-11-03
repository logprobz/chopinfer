//! Serving option registry and external input decoding.
use super::*;

/// The shape of a serve flag's value, as `serve --describe-options` names it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ServeValueV1 {
    /// A whole number.
    Count,
    /// A decimal number.
    Number,
    Text,
    Directory,
    File,
    Json,
    /// JSON, or the word `off`.
    JsonOrOff,
    /// One or more words, up to the next flag.
    Words,
    /// One of the listed words.
    Choice(&'static [&'static str]),
    /// A flag without a value.
    Flag,
}

/// Which part of serving a serve flag configures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ServeGroupV1 {
    Model,
    Scheduling,
    Memory,
    Speculation,
    Prefill,
    Cache,
    Server,
    Media,
    Logging,
    Debug,
}

impl ServeGroupV1 {
    const fn name(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Scheduling => "scheduling",
            Self::Memory => "memory",
            Self::Speculation => "speculation",
            Self::Prefill => "prefill",
            Self::Cache => "cache",
            Self::Server => "server",
            Self::Media => "media",
            Self::Logging => "logging",
            Self::Debug => "debug",
        }
    }
}

/// One flag `serve` accepts.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ServeOptionV1 {
    pub flag: &'static str,
    pub value: ServeValueV1,
    pub group: ServeGroupV1,
    pub help: &'static str,
    /// Whether a `serve --model` start takes it: false for the flags of an
    /// explicit package and for those that print instead of serving.
    pub model_launch: bool,
    /// Whether it may be given more than once.
    pub repeatable: bool,
}

impl ServeOptionV1 {
    const fn new(
        flag: &'static str,
        value: ServeValueV1,
        group: ServeGroupV1,
        help: &'static str,
    ) -> Self {
        Self {
            flag,
            value,
            group,
            help,
            model_launch: true,
            repeatable: false,
        }
    }

    const fn not_launch(mut self) -> Self {
        self.model_launch = false;
        self
    }

    const fn repeatable(mut self) -> Self {
        self.repeatable = true;
        self
    }
}

const ON_OFF_V1: &[&str] = &["on", "off"];

/// Every flag `serve` accepts, the list its parse refuses others against and
/// `serve --describe-options` prints: those of the shared parser, of profile
/// resolution (`--profile`) and the option catalog that ends the command.
/// The host-wide `--help` is every command's, not serve's.
pub(crate) const SERVE_OPTIONS_V1: &[ServeOptionV1] = {
    use ServeGroupV1::{
        Cache, Debug, Logging, Media, Memory, Model, Prefill, Scheduling, Server, Speculation,
    };
    use ServeValueV1::{
        Choice, Count, Directory, File, Flag, Json, JsonOrOff, Number, Text, Words,
    };
    &[
        ServeOptionV1::new(
            "--bundle",
            Directory,
            Model,
            "the package directory to serve",
        )
        .not_launch(),
        ServeOptionV1::new(
            "--model",
            Directory,
            Model,
            "the checkpoint directory to prepare a package for and serve",
        ),
        ServeOptionV1::new(
            "--tensor-parallel-size",
            Count,
            Model,
            "the rank count --model prepares and serves (default 4)",
        ),
        ServeOptionV1::new(
            "--model-cache-dir",
            Directory,
            Model,
            "where --model keeps prepared packages",
        ),
        ServeOptionV1::new(
            "--model-option",
            Text,
            Model,
            "a model option the package declares, as NAME=VALUE",
        )
        .repeatable(),
        ServeOptionV1::new(
            "--list-model-options",
            Flag,
            Model,
            "print the package's model options as JSON and exit",
        )
        .not_launch(),
        ServeOptionV1::new(
            "--activation-policy",
            Choice(&["auto", "checkpoint", "bf16"]),
            Model,
            "the activation precision --model prepares the package for",
        ),
        ServeOptionV1::new(
            "--modalities",
            Text,
            Model,
            "auto, all, or a list of text, image, video and audio",
        ),
        ServeOptionV1::new(
            "--served-model-name",
            Words,
            Model,
            "the names clients request the model by, one or more",
        ),
        ServeOptionV1::new("--topology", Text, Model, "the package topology to serve").not_launch(),
        ServeOptionV1::new(
            "--profile",
            Text,
            Model,
            "a serving profile of a --bundle package, as NAME or FILE#NAME",
        )
        .not_launch(),
        ServeOptionV1::new(
            "--devices",
            Text,
            Model,
            "visible device ordinals, one per rank, as 0,1,...",
        ),
        ServeOptionV1::new(
            "--checkpoint",
            Directory,
            Model,
            "the checkpoint a --bundle package's weight plan loads",
        )
        .not_launch(),
        ServeOptionV1::new(
            "--load-threads",
            Count,
            Model,
            "packing workers per rank of a direct load",
        ),
        ServeOptionV1::new(
            "--prewarm-mapped-views",
            Choice(&["off", "on", "auto"]),
            Model,
            "warm mapped weight views after listening; auto only when memory holds them (default off)",
        ),
        ServeOptionV1::new(
            "--weight-keeper",
            Directory,
            Model,
            "map the rank images the weight keeper in DIR lends",
        ),
        ServeOptionV1::new(
            "--weight-keeper-policy",
            Choice(&["prefer", "require"]),
            Model,
            "load without the keeper when it lends nothing (prefer), or fail (require)",
        ),
        ServeOptionV1::new(
            "--weight-keeper-identity",
            File,
            Model,
            "the content proof that lets this build borrow a keeper's rank images",
        ),
        ServeOptionV1::new(
            "--check",
            Flag,
            Model,
            "run the startup checks that need no device, print the result as JSON and exit",
        )
        .not_launch(),
        ServeOptionV1::new(
            "--describe-options",
            Flag,
            Model,
            "print serve's options and environment as JSON and exit; it ends the command",
        )
        .not_launch(),
        ServeOptionV1::new(
            "--listen",
            Text,
            Server,
            "the address and port to listen on (default 127.0.0.1:8310)",
        ),
        ServeOptionV1::new(
            "--enable-request-id-headers",
            Flag,
            Server,
            "return each request's ID in response headers",
        ),
        ServeOptionV1::new(
            "--api-key",
            Text,
            Server,
            "the key clients must send; CHOPINFER_API_KEY keeps it off the command line",
        ),
        ServeOptionV1::new(
            "--allowed-origins",
            Json,
            Server,
            "CORS origins, as a JSON array of strings",
        ),
        ServeOptionV1::new(
            "--allowed-methods",
            Json,
            Server,
            "CORS methods, as a JSON array of strings",
        ),
        ServeOptionV1::new(
            "--allowed-headers",
            Json,
            Server,
            "CORS headers, as a JSON array of strings",
        ),
        ServeOptionV1::new(
            "--allow-credentials",
            Flag,
            Server,
            "allow CORS credentials",
        ),
        ServeOptionV1::new(
            "--max-context-tokens",
            Count,
            Scheduling,
            "the most tokens one request may hold; the package's context by default",
        ),
        ServeOptionV1::new(
            "--gpu-memory-utilization",
            Number,
            Memory,
            "the share of each GPU's total memory this server may take, in (0, 1]",
        ),
        ServeOptionV1::new(
            "--graph-memory-reserve-mib",
            Count,
            Memory,
            "MiB per rank kept out of the KV pool for CUDA graphs (default 1024)",
        ),
        ServeOptionV1::new(
            "--kv-pages",
            Count,
            Memory,
            "the most KV pages to allocate; as many as fit by default",
        ),
        ServeOptionV1::new(
            "--context-parallel-size",
            Count,
            Memory,
            "ranks of one context-parallel group, which must divide the rank count (default 1)",
        ),
        ServeOptionV1::new(
            "--kv-cache-dtype",
            Choice(&["auto", "bf16", "fp8", "mixed"]),
            Memory,
            "the KV cache element; auto runs the package's preference",
        ),
        ServeOptionV1::new(
            "--speculative-tokens",
            Count,
            Speculation,
            "draft tokens per item and step; 0 serves without speculation",
        ),
        ServeOptionV1::new(
            "--speculative-rows",
            Count,
            Speculation,
            "the most verify rows of one step; the package's by default",
        ),
        ServeOptionV1::new(
            "--speculation",
            Choice(&["fixed", "balanced", "adaptive", "table"]),
            Speculation,
            "how many drafts each step verifies (default adaptive)",
        ),
        ServeOptionV1::new(
            "--speculation-costs",
            Text,
            Speculation,
            "table mode's step costs, as ITEMS:BASE:STEP:DRAFTING:ROW,...",
        ),
        ServeOptionV1::new(
            "--speculation-depths",
            Text,
            Speculation,
            "balanced mode's depth schedule, as ITEMS:DEPTH,...",
        ),
        ServeOptionV1::new(
            "--speculation-widths",
            Text,
            Speculation,
            "the verify widths balanced plans land on, as ROWS,...",
        ),
        ServeOptionV1::new(
            "--speculation-graph-rows",
            Count,
            Speculation,
            "the widest verify step captured in a graph",
        ),
        ServeOptionV1::new(
            "--speculation-confidence",
            JsonOrOff,
            Speculation,
            "a drafter confidence calibration for balanced mode, as JSON, or off",
        ),
        ServeOptionV1::new(
            "--draft-cost",
            Choice(&["per-token", "per-block"]),
            Speculation,
            "how the drafter's cost grows with the drafts of a step",
        ),
        ServeOptionV1::new(
            "--prefix-cache",
            Choice(ON_OFF_V1),
            Cache,
            "reuse cached prompt prefixes when the package declares it (default on)",
        ),
        ServeOptionV1::new(
            "--verify-warmup",
            Choice(ON_OFF_V1),
            Speculation,
            "warm up every verify graph shape at startup (default on)",
        ),
        ServeOptionV1::new(
            "--prefill-policy",
            Choice(&["fixed", "tbt", "adaptive"]),
            Prefill,
            "how prefill shares a step with decoding requests (default fixed)",
        ),
        ServeOptionV1::new(
            "--decode-tbt-target-ms",
            Count,
            Prefill,
            "tbt and adaptive: estimated decode stall target in ms, 1 to 60000 (default 100)",
        ),
        ServeOptionV1::new(
            "--kv-offload-gib",
            Count,
            Cache,
            "host memory for offloaded KV pages in GiB; 0 turns offload off",
        ),
        ServeOptionV1::new(
            "--kv-offload-min-tokens",
            Count,
            Cache,
            "the fewest tokens a sequence offloads (default 64)",
        ),
        ServeOptionV1::new(
            "--kv-offload-gbps",
            Number,
            Cache,
            "the offload rate cap per link in GB/s; uncapped by default",
        ),
        ServeOptionV1::new(
            "--kv-offload-prefix",
            Choice(ON_OFF_V1),
            Cache,
            "offload cached prefixes; only off is served",
        ),
        ServeOptionV1::new(
            "--admission",
            Choice(&["reserved", "optimistic"]),
            Scheduling,
            "how admission reserves KV pages (default optimistic)",
        ),
        ServeOptionV1::new(
            "--admission-watermark",
            Number,
            Scheduling,
            "optimistic: the share of pages kept free of reservations, in [0, 1)",
        ),
        ServeOptionV1::new(
            "--max-preemptions",
            Count,
            Scheduling,
            "optimistic: preemptions before a request keeps its worst case (default 3)",
        ),
        ServeOptionV1::new(
            "--debug-preempt-every",
            Count,
            Debug,
            "qualification only: preempt decoding requests every N tokens",
        ),
        ServeOptionV1::new(
            "--debug-swap-every",
            Count,
            Debug,
            "qualification only: swap decoding requests to host memory every N tokens",
        ),
        ServeOptionV1::new(
            "--max-sequences",
            Count,
            Scheduling,
            "the most sequences served at once (default 8)",
        ),
        ServeOptionV1::new(
            "--max-batch-tokens",
            Count,
            Scheduling,
            "the token budget of one scheduled batch",
        ),
        ServeOptionV1::new(
            "--prefill-chunk-tokens",
            Count,
            Prefill,
            "prompt tokens per prefill chunk, within the batch budget",
        ),
        ServeOptionV1::new(
            "--max-connections",
            Count,
            Server,
            "positive accepted socket limit, including buffer waiters (default 1024)",
        ),
        ServeOptionV1::new(
            "--queue-depth",
            Count,
            Scheduling,
            "requests that may wait beyond the served sequences; no queue-depth bound by default",
        ),
        ServeOptionV1::new(
            "--queue-wait-ms",
            Count,
            Scheduling,
            "how long a request may wait for admission; 0 waits without limit",
        ),
        ServeOptionV1::new(
            "--drain-timeout-ms",
            Count,
            Server,
            "how long a stop drains admitted requests (default 30000)",
        ),
        ServeOptionV1::new(
            "--log-level",
            Choice(&["error", "warn", "info", "debug"]),
            Logging,
            "the log level; CHOPINFER_LOG_LEVEL or info by default",
        ),
        ServeOptionV1::new(
            "--reply-history-observer",
            Choice(ON_OFF_V1),
            Logging,
            "count repeated generated pages and reply ends in bounded digests (default off)",
        ),
        ServeOptionV1::new(
            "--max-request-body-bytes",
            Count,
            Server,
            "the largest request body accepted",
        ),
        ServeOptionV1::new(
            "--max-request-string-bytes",
            Count,
            Server,
            "the largest string one request may carry",
        ),
        ServeOptionV1::new(
            "--responses-store-bytes",
            Count,
            Server,
            "the byte budget of stored Responses; 0 stores none",
        ),
        ServeOptionV1::new(
            "--max-tokenizer-input-bytes",
            Count,
            Server,
            "the most bytes one tokenizer input may carry, 1 to 67108864",
        ),
        ServeOptionV1::new(
            "--max-images-per-request",
            Count,
            Media,
            "the most images one request may carry; the package's count by default",
        ),
        ServeOptionV1::new(
            "--allowed-local-media-path",
            Directory,
            Media,
            "the directory file:// media may be read from",
        ),
        ServeOptionV1::new(
            "--allowed-media-domains",
            Text,
            Media,
            "the hosts remote media may come from, as HOST,...",
        )
        .repeatable(),
        ServeOptionV1::new(
            "--media-allow-private-addresses",
            Choice(ON_OFF_V1),
            Media,
            "let remote media resolve to private addresses (default off)",
        ),
        ServeOptionV1::new(
            "--media-request-timeout-ms",
            Count,
            Media,
            "the time limit of one remote media request, 1 to 300000 ms",
        ),
        ServeOptionV1::new(
            "--media-fetch-timeout-ms",
            Count,
            Media,
            "the time limit of one remote media fetch with its redirects, 1 to 300000 ms",
        ),
        ServeOptionV1::new(
            "--media-connect-timeout-ms",
            Count,
            Media,
            "the time limit of one remote media connection, 1 to 300000 ms",
        ),
        ServeOptionV1::new(
            "--media-max-redirects",
            Count,
            Media,
            "the most redirects one remote media fetch follows",
        ),
        ServeOptionV1::new(
            "--debug-reference-tokens",
            File,
            Debug,
            "qualification only: teacher-forced reference tokens",
        ),
        ServeOptionV1::new(
            "--fake-host-services",
            Flag,
            Debug,
            "serve on host-memory device services, for CPU tests",
        ),
        ServeOptionV1::new(
            "--architecture",
            Text,
            Debug,
            "the device architecture the host answers a package with (default sm_120f)",
        ),
        ServeOptionV1::new(
            "--capability",
            Text,
            Debug,
            "a host capability to declare beyond the built-in ones",
        )
        .repeatable(),
    ]
};

/// The environment variables serve reads that an operator sets.
const SERVE_ENVIRONMENT_V1: &[(&str, &str, bool, &str)] = &[
    (
        "CHOPINFER_API_KEY",
        "TEXT",
        true,
        "the key clients must send when --api-key is not given (VLLM_API_KEY when it is unset)",
    ),
    (
        "VLLM_API_KEY",
        "TEXT",
        true,
        "the key clients must send when neither --api-key nor CHOPINFER_API_KEY is given",
    ),
    (
        "CHOPINFER_LOG_LEVEL",
        "TEXT",
        false,
        "error, warn, info or debug: the log level when --log-level is not given",
    ),
    (
        "CHOPINFER_LOG_SUMMARY_SECONDS",
        "N",
        false,
        "seconds between summary lines, 1 to 86400 (default 60)",
    ),
    (
        "CHOPINFER_MODEL_OPTIONS",
        "TEXT",
        false,
        "model options as NAME=VALUE,..., under the profile's and any --model-option",
    ),
];

/// `serve --describe-options`: [`SERVE_OPTIONS_V1`] and
/// [`SERVE_ENVIRONMENT_V1`] as a `chopinfer.serve-options.v1` document.
pub(super) fn describe_serve_options_v1() -> String {
    let options: Vec<serde_json::Value> = SERVE_OPTIONS_V1
        .iter()
        .map(|option| {
            let (value, choices) = match option.value {
                ServeValueV1::Count => ("N", None),
                ServeValueV1::Number => ("F", None),
                ServeValueV1::Text => ("TEXT", None),
                ServeValueV1::Directory => ("DIR", None),
                ServeValueV1::File => ("FILE", None),
                ServeValueV1::Json => ("JSON", None),
                ServeValueV1::JsonOrOff => ("JSON|off", None),
                ServeValueV1::Words => ("TEXT...", None),
                ServeValueV1::Choice(choices) => ("TEXT", Some(choices)),
                ServeValueV1::Flag => ("none", None),
            };
            serde_json::json!({
                "flag": option.flag,
                "value": value,
                "choices": choices,
                "group": option.group.name(),
                "help": option.help,
                "model_launch": option.model_launch,
                "repeatable": option.repeatable,
            })
        })
        .collect();
    let environment: Vec<serde_json::Value> = SERVE_ENVIRONMENT_V1
        .iter()
        .map(|(name, value, secret, help)| {
            serde_json::json!({"name": name, "value": value, "secret": secret, "help": help})
        })
        .collect();
    serde_json::json!({
        "schema": "chopinfer.serve-options.v1",
        "options": options,
        "environment": environment,
    })
    .to_string()
}

/// Every flag either front end accepts, read once each.
///
/// Parsing is shared, and each command then refuses the flags it does not
/// accept, so an argument that belongs to another command is a refusal rather
/// than something quietly ignored.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ServingInputsV1 {
    pub(crate) model_options: std::collections::BTreeMap<String, String>,
    pub(crate) list_model_options: bool,
    /// `serve --check`: the startup checks that need no device.
    pub(crate) check: bool,
    pub(crate) http_options: runtime_server::HttpOptions,
    pub(crate) http_flags: Vec<String>,
    pub(crate) devices: Option<Vec<i32>>,
    pub(crate) bundle: Option<String>,
    pub(crate) model: Option<PathBuf>,
    pub(crate) tensor_parallel_size: Option<usize>,
    pub(crate) model_cache_dir: Option<PathBuf>,
    pub(crate) activation_policy: Option<String>,
    pub(crate) modalities: Option<runtime_bundle::preparation::ModalitySelection>,
    pub(crate) architecture: Option<String>,
    pub(crate) capabilities: Vec<String>,
    pub(crate) topology: Option<String>,
    pub(crate) listen: Option<String>,
    pub(crate) maximum_context_tokens: Option<usize>,
    pub(crate) gpu_memory_utilization_millionths: Option<u32>,
    pub(crate) graph_memory_reserve_mib: Option<u64>,
    pub(crate) kv_pages: Option<usize>,
    pub(crate) context_parallel_size: Option<usize>,
    pub(crate) kv_cache_dtype: Option<KvCacheDtypeV1>,
    pub(crate) speculative_tokens: Option<usize>,
    pub(crate) speculative_rows: Option<usize>,
    pub(crate) speculation_mode: Option<SpeculationMode>,
    /// `--speculation-costs`; table mode takes these costs.
    pub(crate) speculation_costs: Option<StepCosts>,
    /// `--speculation-depths`; balanced mode takes this schedule.
    pub(crate) speculation_depths: Option<DraftDepths>,
    /// `--speculation-widths`; balanced mode lands on these widths.
    pub(crate) speculation_widths: Option<VerifyWidths>,
    /// `--speculation-confidence`; balanced mode leaves draft verification
    /// to the package's calibrated confidence. `Some(None)` is `off`.
    pub(crate) speculation_confidence: Option<Option<crate::confidence::VerifyConfidence>>,
    /// `--speculation-graph-rows`: the widest verify step captured in a
    /// graph, passed to the package as its `verify_graph_rows` option.
    pub(crate) speculation_graph_rows: Option<usize>,
    pub(crate) draft_cost_model: Option<DraftCostModel>,
    pub(crate) prefix_cache: Option<bool>,
    pub(crate) reply_history_observer: Option<bool>,
    pub(crate) kv_offload_gib: Option<u64>,
    pub(crate) kv_offload_min_tokens: Option<usize>,
    pub(crate) kv_offload_rate: Option<u64>,
    pub(crate) kv_offload_prefix: Option<bool>,
    pub(crate) verify_warmup: Option<bool>,
    pub(crate) optimistic_admission: Option<bool>,
    pub(crate) prefill_policy: Option<PrefillPolicyV1>,
    pub(crate) prewarm_mapped_views: Option<MappedPrewarmV1>,
    pub(crate) decode_tbt_target_ms: Option<usize>,
    pub(crate) admission_watermark_millionths: Option<u32>,
    pub(crate) maximum_preemptions: Option<usize>,
    pub(crate) debug_preempt_every: Option<usize>,
    pub(crate) debug_swap_every: Option<usize>,
    pub(crate) maximum_sequences: Option<usize>,
    pub(crate) maximum_connections: Option<usize>,
    pub(crate) maximum_batch_tokens: Option<usize>,
    pub(crate) prefill_chunk_tokens: Option<usize>,
    pub(crate) queue_depth: Option<usize>,
    pub(crate) queue_wait_ms: Option<u64>,
    pub(crate) drain_timeout_ms: Option<u64>,
    pub(crate) log_level: Option<runtime_log::Level>,
    pub(crate) maximum_request_body_bytes: Option<usize>,
    pub(crate) maximum_request_string_bytes: Option<usize>,
    pub(crate) responses_store_bytes: Option<usize>,
    pub(crate) maximum_tokenizer_input_bytes: Option<usize>,
    pub(crate) maximum_images_per_request: Option<usize>,
    pub(crate) allowed_local_media_path: Option<PathBuf>,
    pub(crate) remote_media: runtime_media::remote::RemoteMediaPolicy,
    pub(crate) remote_media_flags: Vec<&'static str>,
    pub(crate) debug_reference_tokens: Option<PathBuf>,
    pub(crate) direct_checkpoint: Option<PathBuf>,
    pub(crate) load_threads: Option<usize>,
    pub(crate) weight_keeper: Option<PathBuf>,
    pub(crate) weight_keeper_identity: Option<PathBuf>,
    pub(crate) weight_keeper_policy: Option<crate::weight_keeper::KeeperPolicyV1>,
    pub(crate) host_memory_device: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct ArgumentsV1 {
    pub(crate) serving: ServingInputsV1,
    pub(super) fixture_request: Option<String>,
    pub(super) fixture_tokens: Option<usize>,
    pub(super) scenario: Option<FixtureScenarioV1>,
}

impl std::ops::Deref for ArgumentsV1 {
    type Target = ServingInputsV1;
    fn deref(&self) -> &Self::Target {
        &self.serving
    }
}
impl std::ops::DerefMut for ArgumentsV1 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.serving
    }
}

pub(crate) const VALUELESS_FLAGS_V1: &[&str] = &[
    "--enable-request-id-headers",
    "--allow-credentials",
    "--fake-host-services",
    "--list-model-options",
    "--check",
];

/// Removed flags, refused with what replaced them.
const REMOVED_FLAGS_V1: &[(&str, &str)] = &[
    (
        "--mixed-decode-steps",
        "--mixed-decode-steps is retired: use --prefill-policy fixed|tbt|adaptive; remove the retired flag before deploying",
    ),
    (
        "--mixed-min-prefill-tokens",
        "--mixed-min-prefill-tokens is retired: use --prefill-policy fixed|tbt|adaptive; remove the retired flag before deploying",
    ),
    (
        "--prefill-decode-steps",
        "--prefill-decode-steps is retired: use --prefill-policy fixed|tbt|adaptive; remove the retired flag before deploying",
    ),
    (
        "--prefill-itl-target-ms",
        "--prefill-itl-target-ms is retired: use --prefill-policy fixed|tbt|adaptive; remove the retired flag before deploying",
    ),
    (
        "--prefill-timing-cuts",
        "--prefill-timing-cuts is retired: use --prefill-policy fixed|tbt|adaptive; remove the retired flag before deploying",
    ),
    (
        "--prefill-budget",
        "--prefill-budget is retired: use --prefill-policy fixed|tbt|adaptive; remove the retired flag before deploying",
    ),
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
        "--kv-memory-fraction",
        "--kv-memory-fraction is gone: --gpu-memory-utilization F lets this server's own allocations take F of each GPU's total memory; the default 1.0 takes all of it but the graph memory reserve (--graph-memory-reserve-mib)",
    ),
    (
        "--refresh-model",
        "--refresh-model is gone: --model loads the checkpoint directly and reads no tensor contents to refresh; check its contents with chopinfer verify --model DIR",
    ),
];

pub(crate) fn retired_flag_message_v1(flag: &str) -> Option<&'static str> {
    let name = flag.split_once('=').map_or(flag, |(name, _)| name);
    REMOVED_FLAGS_V1
        .iter()
        .find_map(|(retired, message)| (*retired == name).then_some(*message))
}

impl ArgumentsV1 {
    pub(super) fn parse(arguments: &[String]) -> Result<Self, HostCommandErrorV1> {
        let mut parsed = Self::default();
        let mut remaining = arguments.iter().peekable();
        while let Some(flag) = remaining.next() {
            if let Some(message) = retired_flag_message_v1(flag) {
                return Err(usage_error(message));
            }
            if VALUELESS_FLAGS_V1.contains(&flag.as_str()) {
                let slot = match flag.as_str() {
                    "--enable-request-id-headers" => {
                        &mut parsed.http_options.enable_request_id_headers
                    }
                    "--allow-credentials" => &mut parsed.http_options.allow_credentials,
                    "--fake-host-services" => &mut parsed.host_memory_device,
                    "--list-model-options" => &mut parsed.list_model_options,
                    "--check" => &mut parsed.check,
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
                "--gpu-memory-utilization" => set_once_v1(
                    &mut parsed.gpu_memory_utilization_millionths,
                    parse_gpu_memory_utilization_v1(value)?,
                    flag,
                )?,
                "--graph-memory-reserve-mib" => set_once_v1(
                    &mut parsed.graph_memory_reserve_mib,
                    parse_graph_memory_reserve_mib_v1(value)?,
                    flag,
                )?,
                "--kv-pages" => set_once_v1(&mut parsed.kv_pages, count_v1(flag, value)?, flag)?,
                "--context-parallel-size" => set_once_v1(
                    &mut parsed.context_parallel_size,
                    count_v1(flag, value)?,
                    flag,
                )?,
                "--prefill-policy" => {
                    let policy = PrefillPolicyV1::parse(value).ok_or_else(|| {
                        usage_error("--prefill-policy must be fixed, tbt or adaptive")
                    })?;
                    set_once_v1(&mut parsed.prefill_policy, policy, flag)?;
                }
                "--prewarm-mapped-views" => {
                    let mode = MappedPrewarmV1::parse(value).ok_or_else(|| {
                        usage_error("--prewarm-mapped-views must be off, on or auto")
                    })?;
                    set_once_v1(&mut parsed.prewarm_mapped_views, mode, flag)?;
                }
                "--decode-tbt-target-ms" => set_once_v1(
                    &mut parsed.decode_tbt_target_ms,
                    count_v1(flag, value)?,
                    flag,
                )?,
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
                        "balanced" => SpeculationMode::Balanced(DraftDepths::default()),
                        "adaptive" => SpeculationMode::Adaptive,
                        // The costs arrive with --speculation-costs.
                        "table" => SpeculationMode::Table(StepCosts::default()),
                        _ => {
                            return Err(usage_error(
                                "--speculation must be fixed, balanced, adaptive or table",
                            ));
                        }
                    };
                    set_once_v1(&mut parsed.speculation_mode, mode, flag)?;
                }
                "--speculation-costs" => {
                    let costs = StepCosts::parse(value).map_err(|error| {
                        usage_error(format!("invalid --speculation-costs: {error:?}"))
                    })?;
                    set_once_v1(&mut parsed.speculation_costs, costs, flag)?;
                }
                "--speculation-depths" => {
                    let depths = DraftDepths::parse(value).map_err(|error| {
                        usage_error(format!("invalid --speculation-depths: {error:?}"))
                    })?;
                    set_once_v1(&mut parsed.speculation_depths, depths, flag)?;
                }
                "--speculation-widths" => {
                    let widths = VerifyWidths::parse(value).map_err(|error| {
                        usage_error(format!("invalid --speculation-widths: {error:?}"))
                    })?;
                    set_once_v1(&mut parsed.speculation_widths, widths, flag)?;
                }
                "--speculation-confidence" => {
                    let confidence = crate::confidence::parse(value).map_err(|error| {
                        usage_error(format!("invalid --speculation-confidence: {error}"))
                    })?;
                    set_once_v1(&mut parsed.speculation_confidence, confidence, flag)?;
                }
                "--speculation-graph-rows" => {
                    let rows = count_v1(flag, value)?;
                    if rows == 0 {
                        return Err(usage_error("--speculation-graph-rows must be at least 1"));
                    }
                    set_once_v1(&mut parsed.speculation_graph_rows, rows, flag)?;
                }
                "--kv-cache-dtype" => {
                    let dtype = KvCacheDtypeV1::parse(value).ok_or_else(|| {
                        usage_error("--kv-cache-dtype must be auto, bf16, fp8 or mixed")
                    })?;
                    set_once_v1(&mut parsed.kv_cache_dtype, dtype, flag)?;
                }
                "--reply-history-observer" => {
                    let enabled = match value.as_str() {
                        "on" => true,
                        "off" => false,
                        _ => return Err(usage_error("--reply-history-observer must be on or off")),
                    };
                    set_once_v1(&mut parsed.reply_history_observer, enabled, flag)?;
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
                "--max-connections" => {
                    let count = connection_count_v1(count_v1(flag, value)?)?;
                    set_once_v1(&mut parsed.maximum_connections, count, flag)?;
                }
                "--queue-depth" => {
                    set_once_v1(&mut parsed.queue_depth, count_v1(flag, value)?, flag)?;
                }
                "--log-level" => set_once_v1(
                    &mut parsed.log_level,
                    runtime_log::Level::parse(value).ok_or_else(|| {
                        usage_error("--log-level must be error, warn, info or debug")
                    })?,
                    flag,
                )?,
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
                "--weight-keeper" => {
                    set_once_v1(&mut parsed.weight_keeper, PathBuf::from(value), flag)?
                }
                "--weight-keeper-identity" => set_once_v1(
                    &mut parsed.weight_keeper_identity,
                    PathBuf::from(value),
                    flag,
                )?,
                "--weight-keeper-policy" => set_once_v1(
                    &mut parsed.weight_keeper_policy,
                    crate::weight_keeper::KeeperPolicyV1::parse(value).map_err(usage_error)?,
                    flag,
                )?,
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
    pub(super) fn present_flags(&self) -> Vec<&'static str> {
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
            ("--check", self.check),
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
                "--gpu-memory-utilization",
                self.gpu_memory_utilization_millionths.is_some(),
            ),
            (
                "--graph-memory-reserve-mib",
                self.graph_memory_reserve_mib.is_some(),
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
            ("--speculation-costs", self.speculation_costs.is_some()),
            ("--speculation-depths", self.speculation_depths.is_some()),
            ("--speculation-widths", self.speculation_widths.is_some()),
            (
                "--speculation-confidence",
                self.speculation_confidence.is_some(),
            ),
            (
                "--speculation-graph-rows",
                self.speculation_graph_rows.is_some(),
            ),
            ("--draft-cost", self.draft_cost_model.is_some()),
            ("--kv-offload-gib", self.kv_offload_gib.is_some()),
            (
                "--kv-offload-min-tokens",
                self.kv_offload_min_tokens.is_some(),
            ),
            ("--kv-offload-gbps", self.kv_offload_rate.is_some()),
            ("--kv-offload-prefix", self.kv_offload_prefix.is_some()),
            ("--prefix-cache", self.prefix_cache.is_some()),
            (
                "--reply-history-observer",
                self.reply_history_observer.is_some(),
            ),
            ("--verify-warmup", self.verify_warmup.is_some()),
            ("--prefill-policy", self.prefill_policy.is_some()),
            (
                "--prewarm-mapped-views",
                self.prewarm_mapped_views.is_some(),
            ),
            (
                "--decode-tbt-target-ms",
                self.decode_tbt_target_ms.is_some(),
            ),
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
            ("--max-connections", self.maximum_connections.is_some()),
            ("--queue-depth", self.queue_depth.is_some()),
            ("--queue-wait-ms", self.queue_wait_ms.is_some()),
            ("--drain-timeout-ms", self.drain_timeout_ms.is_some()),
            ("--log-level", self.log_level.is_some()),
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
            ("--weight-keeper", self.weight_keeper.is_some()),
            (
                "--weight-keeper-identity",
                self.weight_keeper_identity.is_some(),
            ),
            (
                "--weight-keeper-policy",
                self.weight_keeper_policy.is_some(),
            ),
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

    pub(super) fn reject_unsupported(
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
    pub(super) fn device_services(&self) -> DeviceServicesV1 {
        if self.host_memory_device {
            DeviceServicesV1::HostMemory
        } else {
            DeviceServicesV1::Compiled
        }
    }

    pub(super) fn serve_selection(&self) -> Result<ServeSelection, HostCommandErrorV1> {
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
    pub(super) fn selection(&self, command: &str) -> Result<BundleSelectionV1, HostCommandErrorV1> {
        let bundle = self.bundle.clone().ok_or_else(|| {
            usage_error(format!(
                "{command} requires --bundle, naming a package directory"
            ))
        })?;
        Ok(BundleSelectionV1::Directory(PathBuf::from(bundle)))
    }

    /// What this host tells the verifier about itself.
    pub(super) fn host_compatibility(
        &self,
    ) -> Result<BundleHostCompatibilityV1, HostCommandErrorV1> {
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
                runtime_bundle_abi::device_results::DEVICE_RESULTS_CAPABILITY_V1,
            ))
            .chain(std::iter::once(
                runtime_bundle_abi::device_results::DEVICE_RESULTS_DEFAULT_CAPABILITY_V1,
            ))
            .chain(std::iter::once(
                runtime_bundle_abi::SPECULATIVE_NARROW_ROWS_CAPABILITY_V1,
            ))
            .chain(std::iter::once(
                runtime_bundle_abi::SPECULATIVE_REQUESTED_ROWS_CAPABILITY_V1,
            ))
            .chain(std::iter::once(
                runtime_bundle_abi::SPECULATIVE_EXACT_GRAPHS_CAPABILITY_V1,
            ))
            .chain(std::iter::once(
                runtime_bundle_abi::device_results::MAPPED_DEVICE_RESULTS_CAPABILITY_V1,
            ))
            .chain(std::iter::once(crate::executor::MEDIA_INPUTS_CAPABILITY_V1))
            .chain(std::iter::once(
                crate::executor::POSITION_AXES_CAPABILITY_V1,
            ))
            .chain(std::iter::once(
                runtime_bundle_abi::DRAFT_SAMPLING_CAPABILITY_V1,
            ))
            .chain(std::iter::once(
                runtime_bundle_abi::VERIFY_ALLOCATION_CAPABILITY_V1,
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
                runtime_bundle_abi::ASYNC_MAPPED_GATHER_CAPABILITY_V1,
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

pub(super) fn count_v1(flag: &str, value: &str) -> Result<usize, HostCommandErrorV1> {
    value
        .parse::<usize>()
        .map_err(|_| usage_error(format!("{flag} takes a count, got '{value}'")))
}

pub(super) fn parse_gpu_memory_utilization_v1(value: &str) -> Result<u32, HostCommandErrorV1> {
    let fraction = value
        .parse::<f64>()
        .map_err(|_| usage_error("--gpu-memory-utilization must be a number in (0, 1]"))?;
    if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0 {
        return Err(usage_error("--gpu-memory-utilization must be in (0, 1]"));
    }
    let millionths = (fraction * 1_000_000.0).round() as u32;
    if millionths == 0 {
        return Err(usage_error("--gpu-memory-utilization is too small"));
    }
    Ok(millionths)
}

/// Whole MiB from 1 to 65,536: graphs that outgrow the reserve fail their
/// capture, and a value past 64 GiB is a unit mistake.
pub(super) fn parse_graph_memory_reserve_mib_v1(value: &str) -> Result<u64, HostCommandErrorV1> {
    match value.parse::<u64>() {
        Ok(mib @ 1..=65_536) => Ok(mib),
        _ => Err(usage_error(
            "--graph-memory-reserve-mib takes whole MiB from 1 to 65536",
        )),
    }
}

pub(super) fn parse_admission_watermark_v1(value: &str) -> Result<u32, HostCommandErrorV1> {
    let fraction = value
        .parse::<f64>()
        .map_err(|_| usage_error("--admission-watermark must be a number in [0, 1)"))?;
    if !fraction.is_finite() || !(0.0..1.0).contains(&fraction) {
        return Err(usage_error("--admission-watermark must be in [0, 1)"));
    }
    Ok(((fraction * 1_000_000.0).round() as u32).min(999_999))
}

pub(super) fn set_once_v1<T>(
    slot: &mut Option<T>,
    value: T,
    flag: &str,
) -> Result<(), HostCommandErrorV1> {
    if slot.is_some() {
        return Err(given_more_than_once_v1(flag));
    }
    *slot = Some(value);
    Ok(())
}

pub(super) fn given_more_than_once_v1(flag: &str) -> HostCommandErrorV1 {
    usage_error(format!("{flag} is given more than once"))
}

/// The options of `arguments` as `(flag, values)` words, as the serve parser
/// reads them: valueless flags take none, `--served-model-name` takes every
/// following word up to the next option, and every other flag one value even
/// when it looks like an option.
pub(crate) fn options(arguments: &[String]) -> Vec<(String, Vec<String>)> {
    let mut options = Vec::new();
    let mut words = arguments.iter().peekable();
    while let Some(word) = words.next() {
        let mut values = Vec::new();
        if !crate::command::VALUELESS_FLAGS_V1.contains(&word.as_str())
            && let Some(value) = words.next()
        {
            values.push(value.clone());
            if word == "--served-model-name" {
                while words.peek().is_some_and(|name| !name.starts_with("--")) {
                    values.push(words.next().expect("peeked name").clone());
                }
            }
        }
        options.push((word.clone(), values));
    }
    options
}
