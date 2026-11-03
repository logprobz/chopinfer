//! Presence-aware serving source resolution and semantic validation.
use super::*;

/// Snapshot source values without validating defaults an explicit option ignores.
/// Credentials are digests even in this internal record.
#[derive(Debug)]
pub(crate) struct EnvironmentV1 {
    pub(crate) model_options: std::collections::BTreeMap<String, String>,
    model_option_order: Vec<String>,
    api_key: Option<runtime_server::ApiKey>,
    log_level: Option<String>,
    summary_interval: Option<String>,
}

impl EnvironmentV1 {
    pub(crate) fn process() -> Result<Self, HostCommandErrorV1> {
        Self::read(|name| std::env::var(name).ok())
    }

    pub(crate) fn read(read: impl Fn(&str) -> Option<String>) -> Result<Self, HostCommandErrorV1> {
        let mut model_options = std::collections::BTreeMap::new();
        let mut model_option_order = Vec::new();
        if let Some(value) = read("CHOPINFER_MODEL_OPTIONS") {
            for assignment in value.split(',') {
                let (name, value) = assignment.split_once('=').ok_or_else(|| {
                    usage_error("CHOPINFER_MODEL_OPTIONS requires name=value,...")
                })?;
                if validate_model_option_v1(name, value).is_err()
                    || model_options.contains_key(name)
                {
                    return Err(usage_error(
                        "CHOPINFER_MODEL_OPTIONS contains an invalid or repeated option",
                    ));
                }
                model_options.insert(name.to_owned(), value.to_owned());
                model_option_order.push(name.to_owned());
            }
        }
        Ok(Self {
            model_options,
            model_option_order,
            api_key: resolve_api_key_v1(None, read("CHOPINFER_API_KEY"), read("VLLM_API_KEY")),
            log_level: read("CHOPINFER_LOG_LEVEL"),
            summary_interval: read("CHOPINFER_LOG_SUMMARY_SECONDS"),
        })
    }

    pub(crate) fn model_defaults(&self, mut explicit: ServingInputsV1) -> ServingInputsV1 {
        for (name, value) in &self.model_options {
            explicit
                .model_options
                .entry(name.clone())
                .or_insert_with(|| value.clone());
        }
        explicit
    }

    /// Only the public LocalModel compatibility boundary still needs words.
    pub(crate) fn compatibility_arguments(
        &self,
        explicit: &ServingInputsV1,
        words: &[String],
    ) -> Vec<String> {
        let mut arguments = Vec::new();
        for name in &self.model_option_order {
            if !explicit.model_options.contains_key(name) {
                arguments.extend([
                    "--model-option".into(),
                    format!("{name}={}", self.model_options[name]),
                ]);
            }
        }
        arguments.extend_from_slice(words);
        crate::profiles::serving_options(&arguments)
    }
}

impl ServingInputsV1 {
    pub(crate) fn without_preparation(&self) -> Self {
        let mut serving = self.clone();
        let empty = Self::default();
        for option in SERVE_OPTIONS_V1 {
            if crate::profiles::MODEL_OPTIONS.contains(&option.flag)
                && let Some(access) = option.input
            {
                (access.copy)(&mut serving, &empty);
            }
        }
        serving
            .order
            .retain(|flag| !crate::profiles::MODEL_OPTIONS.contains(flag));
        serving
    }

    /// Restore a profile-owned setting from resolved values. Text is emitted
    /// only after this typed override passes complete configuration equality.
    pub(crate) fn restore(
        &mut self,
        flag: &'static str,
        command: &HostCommandV1,
    ) -> Result<bool, HostCommandErrorV1> {
        let HostCommandV1::Serve {
            serving,
            speculative_tokens,
            speculative_rows,
            speculation_mode,
            draft_cost_model,
            verify_widths,
            verify_confidence,
            model_options,
            ..
        } = command
        else {
            return Ok(false);
        };
        match flag {
            "--max-sequences" => self.maximum_sequences = Some(serving.maximum_sequences),
            "--max-batch-tokens" => self.maximum_batch_tokens = serving.maximum_batch_tokens,
            "--prefill-chunk-tokens" => self.prefill_chunk_tokens = serving.prefill_chunk_tokens,
            "--gpu-memory-utilization" => {
                self.gpu_memory_utilization_millionths =
                    Some(serving.gpu_memory_utilization_millionths)
            }
            "--graph-memory-reserve-mib" => {
                self.graph_memory_reserve_mib = Some(serving.graph_memory_reserve_bytes >> 20)
            }
            "--kv-cache-dtype" => self.kv_cache_dtype = Some(serving.kv_cache_dtype),
            "--prefix-cache" => self.prefix_cache = Some(serving.prefix_cache),
            "--reply-history-observer" => {
                self.reply_history_observer = Some(serving.reply_history_observer)
            }
            "--speculative-tokens" => self.speculative_tokens = Some(*speculative_tokens),
            "--queue-depth" => self.queue_depth = serving.queue_depth,
            "--context-parallel-size" => {
                self.context_parallel_size = Some(serving.context_parallel_size)
            }
            "--prefill-policy" => self.prefill_policy = Some(serving.prefill_policy),
            "--decode-tbt-target-ms" => {
                self.decode_tbt_target_ms = match serving.prefill_budget {
                    runtime_scheduler::PrefillBudget::Tbt(policy) => {
                        Some(policy.target_millis() as usize)
                    }
                    _ => None,
                }
            }
            _ if *speculative_tokens == 0 => return Ok(false),
            "--draft-cost" => self.draft_cost_model = Some(*draft_cost_model),
            "--speculation" => self.speculation_mode = Some(*speculation_mode),
            "--speculation-costs" => {
                self.speculation_costs = match speculation_mode {
                    SpeculationMode::Table(costs) => Some(*costs),
                    _ => None,
                }
            }
            "--speculation-depths" => {
                self.speculation_depths = match speculation_mode {
                    SpeculationMode::Balanced(depths) if !depths.entries().is_empty() => {
                        Some(*depths)
                    }
                    _ => None,
                }
            }
            "--speculation-widths" => {
                self.speculation_widths =
                    (!verify_widths.widths().is_empty()).then_some(*verify_widths)
            }
            "--speculation-confidence"
                if matches!(speculation_mode, SpeculationMode::Balanced(_)) =>
            {
                self.speculation_confidence = Some(verify_confidence.as_deref().cloned());
            }
            "--speculative-rows" => self.speculative_rows = speculative_rows.map(usize::from),
            "--speculation-graph-rows" => {
                self.speculation_graph_rows = model_options
                    .get(runtime_bundle_abi::VERIFY_GRAPH_ROWS_OPTION_V1)
                    .map(|value| count_v1(flag, value))
                    .transpose()?
            }
            _ => return Ok(false),
        }
        let restored = self.contains(flag);
        if restored {
            self.order.push(flag);
        }
        Ok(restored)
    }

    /// Inherited settings that explicit inputs do not replace or disable.
    pub(crate) fn inherited(&self, explicit: &Self) -> Self {
        let mut inherited = self.clone();
        let disabled = explicit.speculative_tokens == Some(0);
        if disabled {
            inherited.speculation_mode = None;
            inherited.speculative_rows = None;
            inherited.speculation_graph_rows = None;
        }
        if disabled
            || explicit
                .speculation_mode
                .is_some_and(|mode| !matches!(mode, SpeculationMode::Table(_)))
        {
            inherited.speculation_costs = None;
        }
        if disabled
            || explicit
                .speculation_mode
                .is_some_and(|mode| !matches!(mode, SpeculationMode::Balanced(_)))
        {
            inherited.speculation_depths = None;
            inherited.speculation_widths = None;
            inherited.speculation_confidence = None;
        }
        if explicit.prefill_policy == Some(PrefillPolicyV1::Fixed) {
            inherited.decode_tbt_target_ms = None;
        }
        let empty = Self::default();
        for option in SERVE_OPTIONS_V1 {
            if option.flag != "--model-option"
                && let Some(access) = option.input
                && (access.present)(explicit)
            {
                (access.copy)(&mut inherited, &empty);
            }
        }
        inherited
            .model_options
            .retain(|name, _| !explicit.model_options.contains_key(name));
        inherited
            .http_flags
            .retain(|flag| !explicit.http_flags.contains(flag));
        inherited
            .remote_media_flags
            .retain(|flag| !explicit.remote_media_flags.contains(flag));
        inherited.order = self
            .order
            .iter()
            .copied()
            .filter(|flag| inherited.contains(flag))
            .collect();
        inherited
    }

    pub(crate) fn overlay(&self, explicit: &Self) -> Self {
        let mut merged = self.inherited(explicit);
        for option in SERVE_OPTIONS_V1 {
            if let Some(access) = option.input
                && (access.present)(explicit)
            {
                (access.copy)(&mut merged, explicit);
            }
        }
        merged.http_flags.extend(explicit.http_flags.clone());
        merged.http_options.enable_cors |= explicit.http_options.enable_cors;
        merged
            .remote_media_flags
            .extend(explicit.remote_media_flags.clone());
        merged.order.extend(explicit.order.iter().copied());
        merged
    }

    pub(crate) fn contains(&self, flag: &str) -> bool {
        SERVE_OPTIONS_V1
            .iter()
            .find(|option| option.flag == flag)
            .and_then(|option| option.input)
            .is_some_and(|access| (access.present)(self))
    }
}

pub(super) fn resolve_api_key_v1(
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

/// `CHOPINFER_LOG_LEVEL`, the level a serve without `--log-level` uses;
/// `info` when it is unset.
pub(super) fn environment_log_level_v1() -> Result<runtime_log::Level, HostCommandErrorV1> {
    log_level_from_v1(std::env::var("CHOPINFER_LOG_LEVEL").ok().as_deref())
}

pub(super) fn log_level_from_v1(
    value: Option<&str>,
) -> Result<runtime_log::Level, HostCommandErrorV1> {
    value.map_or(Ok(runtime_log::Level::Info), |value| {
        runtime_log::Level::parse(value)
            .ok_or_else(|| usage_error("CHOPINFER_LOG_LEVEL must be error, warn, info or debug"))
    })
}

/// `CHOPINFER_LOG_SUMMARY_SECONDS`: how often serve writes its summary line,
/// from a second to a day, or `None` for the server's default.
pub(super) fn summary_interval_from_v1(
    value: Option<&str>,
) -> Result<Option<std::time::Duration>, HostCommandErrorV1> {
    value
        .map(|value| {
            value
                .parse::<u64>()
                .ok()
                .filter(|seconds| (1..=86_400).contains(seconds))
                .map(std::time::Duration::from_secs)
                .ok_or_else(|| {
                    usage_error(
                        "CHOPINFER_LOG_SUMMARY_SECONDS must be a whole number of seconds from 1 to 86400",
                    )
                })
        })
        .transpose()
}

pub(super) fn parse_serve_command_v1(
    arguments: &[String],
) -> Result<HostCommandV1, HostCommandErrorV1> {
    let environment = EnvironmentV1::process()?;
    let explicit = ServingInputsV1::decode(arguments)?;
    let compatibility = environment.compatibility_arguments(&explicit, arguments);
    // Live serving gives environment model options explicit precedence over
    // profile defaults. Standalone profile callers apply environment below them.
    let resolved = crate::profiles::resolve_inputs(environment.model_defaults(explicit))?;
    resolve_inputs(&resolved, &environment, compatibility)
}

pub(crate) fn resolve_inputs(
    resolved: &crate::profiles::Resolution,
    environment: &EnvironmentV1,
    compatibility: Vec<String>,
) -> Result<HostCommandV1, HostCommandErrorV1> {
    let parsed = ArgumentsV1 {
        serving: environment.model_defaults(resolved.inputs.clone()),
        ..Default::default()
    };
    let defaults = ServingOptionsV1::default();
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
    // A keeper lends rank images a checkpoint's plans produce, so only a
    // direct load can borrow them.
    let weight_keeper = match (&parsed.weight_keeper, parsed.weight_keeper_policy) {
        (None, None) => None,
        (None, Some(_)) => {
            return Err(usage_error(
                "--weight-keeper-policy requires --weight-keeper",
            ));
        }
        (Some(_), _) if direct_checkpoint.is_none() && parsed.model.is_none() => {
            return Err(usage_error(
                "--weight-keeper requires --model or --checkpoint",
            ));
        }
        (Some(directory), policy) => Some((
            directory.clone(),
            policy.unwrap_or(crate::weight_keeper::KeeperPolicyV1::Prefer),
        )),
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
        .unwrap_or(defaults.maximum_sequences);
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
        minimum_tokens: parsed
            .kv_offload_min_tokens
            .unwrap_or(defaults.kv_offload.minimum_tokens),
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
    // family's serving profile only after its package is selected. Only
    // checks that depend on those defaults wait for complete typed resolution;
    // every explicit value is checked here.
    let profile_pending = parsed.model.is_some();
    let prefill_budget = prefill_budget_v1(&parsed, profile_pending)?;
    let prefill_policy = parsed.prefill_policy.unwrap_or_default();
    if !profile_pending
        && parsed.speculation_mode.is_some()
        && parsed.speculative_tokens.unwrap_or(0) == 0
    {
        return Err(usage_error(
            "--speculation requires positive --speculative-tokens",
        ));
    }
    if matches!(parsed.speculation_mode, Some(SpeculationMode::Table(_)))
        != parsed.speculation_costs.is_some()
    {
        return Err(usage_error(
            "--speculation table requires --speculation-costs, which only it uses",
        ));
    }
    if parsed.speculation_depths.is_some()
        && !matches!(parsed.speculation_mode, Some(SpeculationMode::Balanced(_)))
    {
        return Err(usage_error(
            "--speculation-depths requires --speculation balanced, which only it uses",
        ));
    }
    if parsed.speculation_widths.is_some()
        && !matches!(parsed.speculation_mode, Some(SpeculationMode::Balanced(_)))
    {
        return Err(usage_error(
            "--speculation-widths requires --speculation balanced, which only it uses",
        ));
    }
    let verify_confidence = parsed.speculation_confidence.clone().flatten();
    if let Some(confidence) = &verify_confidence {
        // With a pending profile, the profile may give the tokens and the
        // balanced mode; an explicit other mode is refused at once.
        let tokens = parsed.speculative_tokens.unwrap_or(0);
        let balanced = matches!(parsed.speculation_mode, Some(SpeculationMode::Balanced(_)));
        let pending = profile_pending && (parsed.speculation_mode.is_none() || balanced);
        if !pending && (tokens == 0 || !balanced) {
            return Err(usage_error(
                "--speculation-confidence requires --speculation balanced and speculative \
                 tokens, which only it uses",
            ));
        }
        if confidence.positions() < tokens {
            return Err(usage_error(format!(
                "--speculation-confidence calibrates {} draft positions, fewer than the {tokens} \
                 speculative tokens",
                confidence.positions()
            )));
        }
    }
    let mut http_options = parsed.http_options.clone();
    http_options.api_key = http_options.api_key.or_else(|| environment.api_key.clone());
    http_options.summary_interval =
        summary_interval_from_v1(environment.summary_interval.as_deref())?;
    http_options.validate().map_err(usage_error)?;
    if !profile_pending
        && parsed.speculative_rows.is_some()
        && parsed.speculative_tokens.unwrap_or(0) == 0
    {
        return Err(usage_error(
            "--speculative-rows requires positive --speculative-tokens",
        ));
    }
    if !profile_pending
        && parsed.speculation_graph_rows.is_some()
        && parsed.speculative_tokens.unwrap_or(0) == 0
    {
        return Err(usage_error(
            "--speculation-graph-rows requires positive --speculative-tokens",
        ));
    }
    let mut model_options = parsed.model_options.clone();
    if let Some(rows) = parsed.speculation_graph_rows
        && model_options
            .insert(
                runtime_bundle_abi::VERIFY_GRAPH_ROWS_OPTION_V1.into(),
                rows.to_string(),
            )
            .is_some()
    {
        return Err(usage_error(format!(
            "--speculation-graph-rows sets the model option '{}'; give one of them",
            runtime_bundle_abi::VERIFY_GRAPH_ROWS_OPTION_V1
        )));
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
        model_options,
        list_model_options: parsed.list_model_options,
        check: parsed.check,
        profile_name: resolved.profile.clone(),
        profile_drafter: resolved.drafter.clone(),
        qualified_max_sequences: resolved.qualified_max_sequences,
        direct_package: None,
        http_options: Box::new(http_options),
        log_level: match parsed.log_level {
            Some(level) => level,
            None => log_level_from_v1(environment.log_level.as_deref())?,
        },
        devices: parsed
            .devices
            .clone()
            .map(|ordinals| Box::new(crate::DeviceSelectionV1(ordinals))),
        host: Box::new(parsed.host_compatibility()?),
        topology: parsed.topology.clone(),
        device: parsed.device_services(),
        listen,
        serving: Box::new(ServingOptionsV1 {
            maximum_context_tokens: parsed.maximum_context_tokens,
            maximum_sequences,
            maximum_connections: parsed
                .maximum_connections
                .unwrap_or(defaults.maximum_connections),
            maximum_batch_tokens: parsed.maximum_batch_tokens,
            prefill_chunk_tokens: parsed.prefill_chunk_tokens,
            queue_depth: parsed.queue_depth,
            queue_wait_ms: parsed.queue_wait_ms.unwrap_or(defaults.queue_wait_ms),
            drain_timeout_ms: parsed.drain_timeout_ms.unwrap_or(defaults.drain_timeout_ms),
            maximum_request_body_bytes: parsed.maximum_request_body_bytes,
            maximum_request_string_bytes: parsed.maximum_request_string_bytes,
            responses_store_bytes: parsed
                .responses_store_bytes
                .unwrap_or(defaults.responses_store_bytes),
            maximum_tokenizer_input_bytes: parsed.maximum_tokenizer_input_bytes,
            maximum_images_per_request: match parsed.maximum_images_per_request {
                Some(count) if count == 0 || count > usize::from(u16::MAX) => {
                    return Err(usage_error(
                        "--max-images-per-request takes a count from 1 to 65535",
                    ));
                }
                count => count,
            },
            gpu_memory_utilization_millionths: parsed
                .gpu_memory_utilization_millionths
                .unwrap_or(defaults.gpu_memory_utilization_millionths),
            graph_memory_reserve_bytes: parsed
                .graph_memory_reserve_mib
                .map_or(defaults.graph_memory_reserve_bytes, |mib| mib << 20),
            kv_pages: parsed.kv_pages.map(|pages| pages as u64),
            context_parallel_size: match parsed.context_parallel_size {
                Some(0) => return Err(usage_error("--context-parallel-size must be at least 1")),
                size => size.unwrap_or(defaults.context_parallel_size),
            },
            kv_cache_dtype: parsed.kv_cache_dtype.unwrap_or(defaults.kv_cache_dtype),
            prefix_cache: parsed.prefix_cache.unwrap_or(defaults.prefix_cache),
            reply_history_observer: parsed
                .reply_history_observer
                .unwrap_or(defaults.reply_history_observer),
            debug_swap: parsed.debug_swap_every.is_some(),
            kv_offload,
            verify_warmup: parsed.verify_warmup.unwrap_or(defaults.verify_warmup),
            admission,
            prefill_budget,
            prefill_policy,
            prewarm_mapped_views: parsed
                .prewarm_mapped_views
                .unwrap_or(defaults.prewarm_mapped_views),
        }),
        speculative_tokens: parsed.speculative_tokens.unwrap_or(0),
        speculative_rows: parsed
            .speculative_rows
            .map(|rows| {
                u16::try_from(rows)
                    .map_err(|_| usage_error("--speculative-rows must be at most 65535"))
            })
            .transpose()?,
        speculation_mode: match (
            parsed.speculation_mode,
            parsed.speculation_costs,
            parsed.speculation_depths,
        ) {
            (Some(SpeculationMode::Table(_)), Some(costs), _) => SpeculationMode::Table(costs),
            (Some(SpeculationMode::Balanced(_)), _, Some(depths)) => {
                SpeculationMode::Balanced(depths)
            }
            (mode, _, _) => mode.unwrap_or(SpeculationMode::Adaptive),
        },
        draft_cost_model: parsed.draft_cost_model.unwrap_or_default(),
        verify_widths: parsed.speculation_widths.unwrap_or_default(),
        verify_confidence: verify_confidence.map(Box::new),
        allowed_local_media_path: parsed.allowed_local_media_path.clone(),
        remote_media: Box::new(parsed.remote_media.clone()),
        debug_reference_tokens: parsed.debug_reference_tokens.clone(),
        direct_checkpoint,
        load_threads,
        weight_keeper,
        weight_keeper_identity: parsed.weight_keeper_identity.clone(),
        selection: match parsed.serve_selection()? {
            ServeSelection::Model(model) => ServeSelection::Model(Box::new(LocalModel {
                serve_arguments: compatibility,
                ..*model
            })),
            bundle => bundle,
        },
    })
}

/// The largest decode TBT target, one minute.
const MAXIMUM_DECODE_TBT_TARGET_MS_V1: usize = 60_000;

/// Resolve the public policy into the scheduler's existing budget. A direct
/// model profile can supply an omitted policy after package selection.
fn prefill_budget_v1(
    parsed: &ArgumentsV1,
    profile_pending: bool,
) -> Result<runtime_scheduler::PrefillBudget, HostCommandErrorV1> {
    use runtime_scheduler::{MixedPolicy, PrefillBudget};
    let target = parsed
        .decode_tbt_target_ms
        .unwrap_or(runtime_scheduler::DEFAULT_DECODE_TBT_TARGET_MS as usize);
    if !(1..=MAXIMUM_DECODE_TBT_TARGET_MS_V1).contains(&target) {
        return Err(usage_error(format!(
            "--decode-tbt-target-ms must be 1 to {MAXIMUM_DECODE_TBT_TARGET_MS_V1}"
        )));
    }
    match parsed.prefill_policy.unwrap_or_default() {
        PrefillPolicyV1::Fixed => {
            if parsed.decode_tbt_target_ms.is_some()
                && !(profile_pending && parsed.prefill_policy.is_none())
            {
                return Err(usage_error(
                    "--decode-tbt-target-ms requires --prefill-policy tbt or adaptive",
                ));
            }
            Ok(PrefillBudget::Fixed)
        }
        PrefillPolicyV1::Tbt | PrefillPolicyV1::Adaptive => MixedPolicy::new(
            target as u32,
            runtime_scheduler::DEFAULT_MIXED_MIN_PREFILL_TOKENS,
            0,
        )
        .map(PrefillBudget::Tbt)
        .map_err(|error| usage_error(error.to_string())),
    }
}
