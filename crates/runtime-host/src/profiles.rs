//! Model-neutral, versioned serving presets. Artifact declarations remain authoritative.
use crate::command::HostCommandErrorV1;
use crate::command::serve_config::EnvironmentV1;
use crate::command::serve_options::options;
use crate::command::serve_options::{self, ServingInputsV1};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Profiles {
    pub schema: String,
    pub model: String,
    pub served_model_name: String,
    pub profiles: BTreeMap<String, Profile>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Profile {
    #[serde(
        default,
        skip_serializing_if = "BTreeMap::is_empty",
        deserialize_with = "deserialize_model_options"
    )]
    pub model_options: BTreeMap<String, serde_json::Value>,
    pub bundle_id: String,
    pub topology: String,
    pub tensor_parallel: usize,
    pub devices_required: usize,
    pub devices: Vec<usize>,
    pub speculation: Speculation,
    pub max_sequences: usize,
    pub qualified_max_sequences: Option<usize>,
    pub qualification: String,
    pub kv_cache_dtype: String,
    pub prefix_cache: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reply_history_observer: Option<bool>,
    pub max_batch_tokens: usize,
    pub prefill_chunk_tokens: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_policy: Option<crate::PrefillPolicyV1>,
    #[serde(
        rename = "prefill_budget",
        default,
        deserialize_with = "retired_prefill_field",
        skip_serializing
    )]
    _retired_prefill_budget: (),
    #[serde(
        rename = "prefill_timing_cuts",
        default,
        deserialize_with = "retired_prefill_field",
        skip_serializing
    )]
    _retired_prefill_timing_cuts: (),
    #[serde(
        rename = "prefill_itl_target_ms",
        default,
        deserialize_with = "retired_prefill_field",
        skip_serializing
    )]
    _retired_prefill_itl_target_ms: (),
    #[serde(
        rename = "prefill_decode_steps",
        default,
        deserialize_with = "retired_prefill_field",
        skip_serializing
    )]
    _retired_prefill_decode_steps: (),
    #[serde(
        rename = "mixed_min_prefill_tokens",
        default,
        deserialize_with = "retired_prefill_field",
        skip_serializing
    )]
    _retired_mixed_min_prefill_tokens: (),
    #[serde(
        rename = "mixed_decode_steps",
        default,
        deserialize_with = "retired_prefill_field",
        skip_serializing
    )]
    _retired_mixed_decode_steps: (),
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_tbt_target_ms: Option<u32>,
    pub queue_depth: Option<usize>,
    /// `--context-parallel-size`; absent serves size 1. It must divide the
    /// tensor-parallel size, and the package must declare it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_parallel_size: Option<usize>,
    /// `--gpu-memory-utilization`.
    pub gpu_memory_utilization: f64,
    /// `--graph-memory-reserve-mib`; absent keeps the host's 1 GiB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_memory_reserve_mib: Option<u64>,
    pub status: Status,
    pub evidence: Vec<String>,
}

/// Deserialize a retired key before any value, including null, can hide it.
fn retired_prefill_field<'de, D>(_: D) -> Result<(), D::Error>
where
    D: serde::Deserializer<'de>,
{
    Err(serde::de::Error::custom(
        "retired prefill field: use prefill_policy (fixed, tbt or adaptive); remove retired fields before deploying",
    ))
}

fn deserialize_model_options<'de, D>(
    deserializer: D,
) -> Result<BTreeMap<String, serde_json::Value>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct OptionsVisitor;

    impl<'de> serde::de::Visitor<'de> for OptionsVisitor {
        type Value = BTreeMap<String, serde_json::Value>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("model options with unique names")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            let mut options = BTreeMap::new();
            while let Some((name, value)) = map.next_entry::<String, serde_json::Value>()? {
                if options.insert(name.clone(), value).is_some() {
                    return Err(serde::de::Error::custom(format!(
                        "duplicate model option '{name}'"
                    )));
                }
            }
            Ok(options)
        }
    }

    deserializer.deserialize_map(OptionsVisitor)
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Status {
    Qualified,
    Experimental,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Speculation {
    pub mode: String,
    pub tokens: usize,
    pub draft_cost: String,
    /// Packaging requirement, interpreted by the model's preparation tools.
    pub drafter: String,
    /// The step-cost table of `table` mode, in `--speculation-costs` form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub costs: Option<String>,
    /// The depth schedule of `balanced` mode, in `--speculation-depths`
    /// form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depths: Option<String>,
    /// The verify widths `balanced` mode lands on, in
    /// `--speculation-widths` form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub widths: Option<String>,
    /// The token capacity: most verify rows of one step, as
    /// `--speculative-rows`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows: Option<usize>,
    /// The graph budget: the widest verify step captured in a graph, as
    /// `--speculation-graph-rows`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_rows: Option<usize>,
    /// The confidence calibration `balanced` mode leaves draft verification
    /// to, as `--speculation-confidence`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<crate::confidence::Calibration>,
}

fn error(message: impl Into<String>) -> HostCommandErrorV1 {
    HostCommandErrorV1::Usage(message.into())
}

impl Profiles {
    pub fn read(path: &Path) -> Result<Self, HostCommandErrorV1> {
        let bytes =
            std::fs::read(path).map_err(|e| error(format!("profile {}: {e}", path.display())))?;
        Self::parse(&bytes)
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, HostCommandErrorV1> {
        let value: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|e| error(format!("profile JSON: {e}")))?;
        if value["schema"] == "chopinfer-serve-profiles-v1" {
            return Err(error(
                "profile schema v1 is no longer supported: drop fold_ranks and set schema to chopinfer-serve-profiles-v2; use one distinct device per rank",
            ));
        }
        if value["profiles"].as_object().is_some_and(|profiles| {
            profiles
                .values()
                .any(|p| p.get("kv_memory_fraction").is_some())
        }) {
            return Err(error(
                "kv_memory_fraction is gone: set gpu_memory_utilization, the share of each GPU's total memory the server's own allocations may take (1.0 takes all of it but the graph memory reserve)",
            ));
        }
        let file: Self =
            serde_json::from_slice(bytes).map_err(|e| error(format!("profile JSON: {e}")))?;
        if file.schema != "chopinfer-serve-profiles-v2"
            || file.model.is_empty()
            || file.served_model_name.is_empty()
            || file.profiles.is_empty()
        {
            return Err(error("unsupported profile schema or empty model/profiles"));
        }
        for (name, profile) in &file.profiles {
            if name.is_empty() || name.contains('#') {
                return Err(error("invalid profile name"));
            }
            profile.validate()?;
            Resolution::plain(profile.inputs(Some(&file.served_model_name))?).command()?;
        }
        Ok(file)
    }

    pub fn get(&self, name: &str) -> Result<&Profile, HostCommandErrorV1> {
        self.profiles
            .get(name)
            .ok_or_else(|| error(format!("unknown profile '{name}'")))
    }
}

impl Profile {
    /// Decode schema values directly. The original token count gates inherited
    /// speculation companions before explicit values are merged.
    pub(crate) fn inputs(
        &self,
        served_model_name: Option<&str>,
    ) -> Result<ServingInputsV1, HostCommandErrorV1> {
        self.inputs_under(served_model_name, &ServingInputsV1::default())
    }

    fn inputs_under(
        &self,
        served_model_name: Option<&str>,
        explicit: &ServingInputsV1,
    ) -> Result<ServingInputsV1, HostCommandErrorV1> {
        let mut inputs = ServingInputsV1 {
            devices: Some(
                self.devices
                    .iter()
                    .map(|&device| {
                        i32::try_from(device).map_err(|_| error("invalid profile device ordinal"))
                    })
                    .collect::<Result<_, _>>()?,
            ),
            maximum_sequences: Some(self.max_sequences),
            maximum_batch_tokens: Some(self.max_batch_tokens),
            prefill_chunk_tokens: Some(self.prefill_chunk_tokens),
            gpu_memory_utilization_millionths: if explicit.replaces("--gpu-memory-utilization") {
                None
            } else {
                Some(serve_options::gpu_memory_utilization_v1(
                    self.gpu_memory_utilization,
                )?)
            },
            kv_cache_dtype: if explicit.replaces("--kv-cache-dtype") {
                None
            } else {
                Some(
                    crate::KvCacheDtypeV1::parse(&self.kv_cache_dtype).ok_or_else(|| {
                        crate::command::usage_error(
                            "--kv-cache-dtype must be auto, bf16, fp8 or mixed",
                        )
                    })?,
                )
            },
            prefix_cache: Some(self.prefix_cache),
            speculative_tokens: Some(self.speculation.tokens),
            draft_cost_model: Some(serve_options::draft_cost_v1(&self.speculation.draft_cost)?),
            reply_history_observer: self.reply_history_observer,
            queue_depth: self.queue_depth,
            context_parallel_size: self.context_parallel_size,
            graph_memory_reserve_mib: if explicit.replaces("--graph-memory-reserve-mib") {
                None
            } else {
                self.graph_memory_reserve_mib
                    .map(|mib| serve_options::graph_memory_reserve_mib_v1(Some(mib)))
                    .transpose()?
            },
            prefill_policy: self.prefill_policy,
            decode_tbt_target_ms: self.decode_tbt_target_ms.map(|target| target as usize),
            ..Default::default()
        };
        for (name, value) in self.model_option_texts() {
            // Preserve the assignment boundary of the profile's external
            // spelling without serializing and reparsing a serve command.
            let (name, value) = serve_options::model_option_pair_v1(name, value);
            if explicit.model_options.contains_key(&name) {
                continue;
            }
            serve_options::validate_model_option_v1(&name, &value)?;
            if inputs.model_options.insert(name.clone(), value).is_some() {
                return Err(crate::command::usage_error(format!(
                    "model option '{name}' given more than once"
                )));
            }
        }
        if let Some(name) = served_model_name {
            serve_options::validate_served_name_start_v1(name)?;
            inputs.bundle = Some(self.bundle_id.clone());
            inputs.topology = Some(self.topology.clone());
            inputs.http_options.served_model_names.push(name.to_owned());
            inputs.http_flags.push("--served-model-name".into());
        }
        if self.speculation.tokens > 0 {
            inputs.speculation_mode =
                Some(serve_options::speculation_mode_v1(&self.speculation.mode)?);
            inputs.speculation_costs = self
                .speculation
                .costs
                .as_deref()
                .map(|text| {
                    runtime_engine::StepCosts::parse(text).map_err(|e| {
                        crate::command::usage_error(format!("invalid --speculation-costs: {e:?}"))
                    })
                })
                .transpose()?;
            inputs.speculation_depths = self
                .speculation
                .depths
                .as_deref()
                .map(|text| {
                    runtime_engine::DraftDepths::parse(text).map_err(|e| {
                        crate::command::usage_error(format!("invalid --speculation-depths: {e:?}"))
                    })
                })
                .transpose()?;
            inputs.speculation_widths = self
                .speculation
                .widths
                .as_deref()
                .map(|text| {
                    runtime_engine::VerifyWidths::parse(text).map_err(|e| {
                        crate::command::usage_error(format!("invalid --speculation-widths: {e:?}"))
                    })
                })
                .transpose()?;
            inputs.speculative_rows = self.speculation.rows;
            inputs.speculation_graph_rows = self.speculation.graph_rows;
            inputs.speculation_confidence = self
                .speculation
                .confidence
                .as_ref()
                .map(|calibration| {
                    calibration.check().map(Some).map_err(|e| {
                        crate::command::usage_error(format!(
                            "invalid --speculation-confidence: {e}"
                        ))
                    })
                })
                .transpose()?;
        }
        Ok(inputs)
    }

    /// Profile declaration order, retained even when a generic value will be
    /// replaced before decoding. Declaration-specific checks still run first.
    fn option_order(&self) -> Vec<&'static str> {
        let mut order = vec![
            "--devices",
            "--max-sequences",
            "--max-batch-tokens",
            "--prefill-chunk-tokens",
            "--gpu-memory-utilization",
            "--kv-cache-dtype",
            "--prefix-cache",
            "--speculative-tokens",
            "--draft-cost",
        ];
        let active = self.speculation.tokens > 0;
        for (flag, given) in [
            (
                "--reply-history-observer",
                self.reply_history_observer.is_some(),
            ),
            ("--queue-depth", self.queue_depth.is_some()),
            (
                "--context-parallel-size",
                self.context_parallel_size.is_some(),
            ),
            (
                "--graph-memory-reserve-mib",
                self.graph_memory_reserve_mib.is_some(),
            ),
            ("--prefill-policy", self.prefill_policy.is_some()),
            (
                "--decode-tbt-target-ms",
                self.decode_tbt_target_ms.is_some(),
            ),
            ("--model-option", !self.model_options.is_empty()),
            ("--speculation", active),
            (
                "--speculation-costs",
                active && self.speculation.costs.is_some(),
            ),
            (
                "--speculation-depths",
                active && self.speculation.depths.is_some(),
            ),
            (
                "--speculation-widths",
                active && self.speculation.widths.is_some(),
            ),
            (
                "--speculative-rows",
                active && self.speculation.rows.is_some(),
            ),
            (
                "--speculation-graph-rows",
                active && self.speculation.graph_rows.is_some(),
            ),
            (
                "--speculation-confidence",
                active && self.speculation.confidence.is_some(),
            ),
        ] {
            if given {
                order.push(flag);
            }
        }
        order
    }

    fn validate(&self) -> Result<(), HostCommandErrorV1> {
        if self
            .model_options
            .values()
            .any(|value| value.is_null() || value.is_array() || value.is_object())
        {
            return Err(error("profile model options must contain scalar values"));
        }
        let unique: std::collections::BTreeSet<_> = self.devices.iter().copied().collect();
        if self.tensor_parallel == 0
            || self.devices.len() != self.tensor_parallel
            || unique.len() != self.devices_required
            || unique.iter().copied().ne(0..self.devices_required)
            || self.devices_required != self.tensor_parallel
        {
            return Err(error("profile rank/device mapping is inconsistent"));
        }
        if self.evidence.is_empty()
            || self.evidence.iter().any(String::is_empty)
            || self.qualification.is_empty()
            || self.speculation.drafter.is_empty()
            || self
                .qualified_max_sequences
                .is_some_and(|n| n == 0 || n < self.max_sequences)
            || (matches!(self.status, Status::Qualified) && self.qualified_max_sequences.is_none())
        {
            return Err(error(
                "profile requires qualification evidence and valid sequence limits",
            ));
        }
        if !matches!(
            self.speculation.mode.as_str(),
            "fixed" | "balanced" | "adaptive" | "table"
        ) || !matches!(
            self.speculation.draft_cost.as_str(),
            "per-token" | "per-block"
        ) {
            return Err(error("invalid profile speculation mode or draft cost"));
        }
        if (self.speculation.mode == "table") != self.speculation.costs.is_some()
            || self
                .speculation
                .costs
                .as_deref()
                .is_some_and(|costs| runtime_engine::StepCosts::parse(costs).is_err())
        {
            return Err(error(
                "profile table speculation needs valid costs, which only it uses",
            ));
        }
        if self.speculation.depths.as_deref().is_some_and(|depths| {
            self.speculation.mode != "balanced"
                || runtime_engine::DraftDepths::parse(depths).is_err()
        }) {
            return Err(error(
                "profile speculation depths must be valid, and only balanced mode takes them",
            ));
        }
        if self.speculation.widths.as_deref().is_some_and(|widths| {
            self.speculation.mode != "balanced"
                || runtime_engine::VerifyWidths::parse(widths).is_err()
        }) {
            return Err(error(
                "profile speculation widths must be valid, and only balanced mode takes them",
            ));
        }
        if let Some(confidence) = &self.speculation.confidence {
            let checked = confidence.check().map_err(|message| {
                error(format!("invalid profile speculation confidence: {message}"))
            })?;
            if self.speculation.mode != "balanced"
                || self.speculation.tokens == 0
                || checked.positions() < self.speculation.tokens
            {
                return Err(error(
                    "profile speculation confidence needs balanced mode, speculative tokens, \
                     and a position for each token",
                ));
            }
        }
        if (self.speculation.rows.is_some() || self.speculation.graph_rows.is_some())
            && self.speculation.tokens == 0
            || self
                .speculation
                .rows
                .is_some_and(|rows| rows <= self.speculation.tokens)
            || self.speculation.graph_rows == Some(0)
        {
            return Err(error(
                "profile speculation rows must exceed its tokens, graph rows must be positive, \
                 and both need speculative tokens",
            ));
        }
        if self
            .context_parallel_size
            .is_some_and(|size| size == 0 || self.tensor_parallel % size != 0)
        {
            return Err(error(
                "profile context-parallel size must divide its tensor-parallel size",
            ));
        }
        if (self.speculation.drafter == "none") != (self.speculation.tokens == 0) {
            return Err(error(
                "profile drafter must be none exactly when speculative tokens are zero",
            ));
        }
        Ok(())
    }

    /// The profile's model options as `--model-option` values: a string as
    /// itself, any other value as its JSON text.
    pub(crate) fn model_option_texts(&self) -> BTreeMap<String, String> {
        self.model_options
            .iter()
            .map(|(name, value)| {
                let text = value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| value.to_string());
                (name.clone(), text)
            })
            .collect()
    }

    pub fn arguments(&self, served_model_name: &str) -> Vec<String> {
        let mut args = Vec::new();
        for (flag, value) in [
            ("--bundle", self.bundle_id.clone()),
            ("--topology", self.topology.clone()),
            ("--served-model-name", served_model_name.into()),
            (
                "--devices",
                self.devices
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            ),
            ("--max-sequences", self.max_sequences.to_string()),
            ("--max-batch-tokens", self.max_batch_tokens.to_string()),
            (
                "--prefill-chunk-tokens",
                self.prefill_chunk_tokens.to_string(),
            ),
            (
                "--gpu-memory-utilization",
                self.gpu_memory_utilization.to_string(),
            ),
            ("--kv-cache-dtype", self.kv_cache_dtype.clone()),
            (
                "--prefix-cache",
                if self.prefix_cache { "on" } else { "off" }.into(),
            ),
            ("--speculative-tokens", self.speculation.tokens.to_string()),
            ("--draft-cost", self.speculation.draft_cost.clone()),
        ] {
            args.extend([flag.into(), value]);
        }
        if let Some(enabled) = self.reply_history_observer {
            args.extend([
                "--reply-history-observer".into(),
                if enabled { "on" } else { "off" }.into(),
            ]);
        }
        if let Some(depth) = self.queue_depth {
            args.extend(["--queue-depth".into(), depth.to_string()]);
        }
        if let Some(size) = self.context_parallel_size {
            args.extend(["--context-parallel-size".into(), size.to_string()]);
        }
        if let Some(mib) = self.graph_memory_reserve_mib {
            args.extend(["--graph-memory-reserve-mib".into(), mib.to_string()]);
        }
        if let Some(policy) = self.prefill_policy {
            args.extend(["--prefill-policy".into(), policy.name().into()]);
        }
        if let Some(target) = self.decode_tbt_target_ms {
            args.extend(["--decode-tbt-target-ms".into(), target.to_string()]);
        }
        for (name, value) in self.model_option_texts() {
            args.extend(["--model-option".into(), format!("{name}={value}")]);
        }
        if self.speculation.tokens > 0 {
            args.extend(["--speculation".into(), self.speculation.mode.clone()]);
            if let Some(costs) = &self.speculation.costs {
                args.extend(["--speculation-costs".into(), costs.clone()]);
            }
            if let Some(depths) = &self.speculation.depths {
                args.extend(["--speculation-depths".into(), depths.clone()]);
            }
            if let Some(widths) = &self.speculation.widths {
                args.extend(["--speculation-widths".into(), widths.clone()]);
            }
            if let Some(rows) = self.speculation.rows {
                args.extend(["--speculative-rows".into(), rows.to_string()]);
            }
            if let Some(rows) = self.speculation.graph_rows {
                args.extend(["--speculation-graph-rows".into(), rows.to_string()]);
            }
            if let Some(confidence) = &self.speculation.confidence {
                args.extend([
                    "--speculation-confidence".into(),
                    serde_json::to_string(confidence).expect("a calibration serializes"),
                ]);
            }
        }
        args
    }
}

pub(crate) struct Resolution {
    pub inputs: ServingInputsV1,
    /// The name of the serving profile applied, if any.
    pub profile: Option<String>,
    pub drafter: Option<String>,
    pub qualified_max_sequences: Option<usize>,
}

impl Resolution {
    pub(crate) fn plain(inputs: ServingInputsV1) -> Self {
        Self {
            inputs,
            profile: None,
            drafter: None,
            qualified_max_sequences: None,
        }
    }

    pub(crate) fn command(&self) -> Result<crate::HostCommandV1, HostCommandErrorV1> {
        let environment = EnvironmentV1::process()?;
        crate::command::serve_config::resolve_inputs(self, &environment, Vec::new())
    }
}

pub(crate) fn resolve_inputs(
    mut explicit: ServingInputsV1,
) -> Result<Resolution, HostCommandErrorV1> {
    let Some(reference) = explicit.profile.take() else {
        return Ok(Resolution::plain(explicit));
    };
    // A profile serves an explicit package, as `serve --model` serves the
    // package it prepared: its options are defaults under the explicit ones.
    let bundle = explicit.bundle.as_deref().ok_or_else(|| {
        error(
            "--profile applies to an explicit package: add --bundle DIR --checkpoint DIR, \
               or serve --model DIR, which applies the family's profile itself",
        )
    })?;
    let package = PackageFacts::read(Path::new(bundle))?;
    // A bare name is one of the package's own profiles; `FILE#NAME` names
    // another file explicitly.
    let (file, name) = match reference.rsplit_once('#') {
        Some((path, name)) => (Profiles::read(Path::new(path))?, name.to_owned()),
        None => (package.profiles(Path::new(bundle))?, reference.clone()),
    };
    let profile = file.get(&name)?;
    // The package declares the profile's topology at the profile's rank
    // count; no other topology stands in for it.
    if profile.bundle_id != package.id
        || !package
            .topologies
            .contains(&(profile.topology.clone(), profile.tensor_parallel))
    {
        return Err(error(format!(
            "profile {name} serves {} topology {} at {} ranks; the package is {} with \
             topologies {:?}",
            profile.bundle_id,
            profile.topology,
            profile.tensor_parallel,
            package.id,
            package.topologies
        )));
    }
    // A context-parallel size above 1 must be one the package declares for
    // the profile's topology.
    if let Some(size) = profile.context_parallel_size.filter(|&size| size > 1)
        && !package
            .context_parallel_sizes
            .get(&profile.topology)
            .is_some_and(|sizes| sizes.contains(&size))
    {
        return Err(error(format!(
            "profile {name} serves context-parallel size {size}; the package declares {:?} for \
             topology {}",
            package
                .context_parallel_sizes
                .get(&profile.topology)
                .cloned()
                .unwrap_or_default(),
            profile.topology
        )));
    }
    // The profile's own model options must be values the package accepts,
    // whatever an explicit `--model-option` later replaces them with: an
    // option the package declares with one allowed value binds every
    // profile that sets it to that package.
    if !profile.model_options.is_empty() {
        let pinned = profile.model_option_texts();
        let declared = package.model_options.as_ref().ok_or_else(|| {
            error(format!(
                "profile {name} sets model options {:?}; the package declares none",
                pinned.keys().collect::<Vec<_>>()
            ))
        })?;
        declared.resolve(&pinned).map_err(|e| {
            error(format!(
                "profile {name} sets model options the package does not accept: {e}"
            ))
        })?;
    }
    // The profile's topology is a default like its other options, so a
    // device list for another rank count is refused rather than serving
    // another topology; an explicit `--topology` replaces it.
    let mut defaults = profile.inputs(None)?;
    defaults.topology = Some(profile.topology.clone());
    Ok(Resolution {
        inputs: defaults.overlay(&explicit),
        profile: Some(name),
        drafter: Some(profile.speculation.drafter.clone()),
        qualified_max_sequences: profile.qualified_max_sequences,
    })
}

/// What a profile is checked against: the package's identifier and its
/// topologies with their rank counts. The manifest is read only for this lookup, as a
/// regular file within the manifest's bound and without following a link;
/// the load verifies it in full.
struct PackageFacts {
    id: String,
    topologies: Vec<(String, usize)>,
    /// The context-parallel sizes above 1 each topology declares.
    context_parallel_sizes: BTreeMap<String, Vec<usize>>,
    model_options: Option<runtime_bundle::model_options::ModelOptionsV1>,
    /// The serving profiles component the manifest declares, if any: its
    /// path, length and SHA-256.
    profiles: Option<(runtime_bundle::BundleRelativePathV1, u64, String)>,
}

impl PackageFacts {
    fn read(bundle: &Path) -> Result<Self, HostCommandErrorV1> {
        use std::io::Read;
        let path = bundle.join(runtime_bundle::BUNDLE_MANIFEST_FILE);
        let failed = |e: std::io::Error| error(format!("read {}: {e}", path.display()));
        let file = runtime_weights::open_nofollow(&path).map_err(failed)?;
        let bound = runtime_bundle::BUNDLE_MANIFEST_MAX_BYTES as u64;
        let too_large = || error(format!("{} exceeds its bound", path.display()));
        if file.metadata().map_err(failed)?.len() > bound {
            return Err(too_large());
        }
        let mut bytes = Vec::new();
        file.take(bound + 1)
            .read_to_end(&mut bytes)
            .map_err(failed)?;
        if bytes.len() as u64 > bound {
            return Err(too_large());
        }
        let manifest: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| error(format!("{}: {e}", path.display())))?;
        let id = manifest["id"]
            .as_str()
            .ok_or_else(|| error(format!("{} names no bundle", path.display())))?
            .to_owned();
        let listed = manifest["target"]["topologies"].as_array();
        let topologies = listed
            .into_iter()
            .flatten()
            .filter_map(|topology| {
                Some((
                    topology["id"].as_str()?.to_owned(),
                    usize::try_from(topology["rank_count"].as_u64()?).ok()?,
                ))
            })
            .collect();
        let context_parallel_sizes = listed
            .into_iter()
            .flatten()
            .filter_map(|topology| {
                let sizes = topology["context_parallel_sizes"]
                    .as_array()?
                    .iter()
                    .filter_map(|size| usize::try_from(size.as_u64()?).ok())
                    .collect();
                Some((topology["id"].as_str()?.to_owned(), sizes))
            })
            .collect();
        let model_options = match manifest.get("model_options") {
            None | Some(serde_json::Value::Null) => None,
            Some(value) => Some(
                serde_json::from_value(value.clone())
                    .map_err(|e| error(format!("{} model options: {e}", path.display())))?,
            ),
        };
        let role = runtime_bundle::deployment::SERVE_PROFILES_ROLE_V1;
        let declared: Vec<_> = manifest["components"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|component| component["role"] == role)
            .collect();
        let profiles = match declared.as_slice() {
            [] => None,
            [component] => {
                let invalid = || {
                    error(format!(
                        "{} declares invalid serving profiles",
                        path.display()
                    ))
                };
                let file = runtime_bundle::BundleRelativePathV1::new(
                    component["path"].as_str().ok_or_else(invalid)?,
                )
                .map_err(|_| invalid())?;
                Some((
                    file,
                    component["byte_length"].as_u64().ok_or_else(invalid)?,
                    component["sha256"].as_str().ok_or_else(invalid)?.to_owned(),
                ))
            }
            _ => {
                return Err(error(format!(
                    "{} declares more than one serving profiles component",
                    path.display()
                )));
            }
        };
        Ok(Self {
            id,
            topologies,
            context_parallel_sizes,
            model_options,
            profiles,
        })
    }

    /// The serving profiles the package at `bundle` carries, read whole and
    /// authenticated against the manifest's length and SHA-256.
    fn profiles(&self, bundle: &Path) -> Result<Profiles, HostCommandErrorV1> {
        let Some((path, length, sha256)) = &self.profiles else {
            return Err(error(format!(
                "the package {} carries no serving profiles; pass --profile FILE#NAME",
                self.id
            )));
        };
        let path = bundle.join(path.as_path());
        if *length > runtime_bundle::deployment::DEPLOYMENT_COMPONENT_MAX_BYTES {
            return Err(error(format!("{} exceeds its bound", path.display())));
        }
        let file = runtime_bundle::VerifiedComponentFileV1::open(&path, *length, sha256)
            .map_err(|e| error(format!("serving profiles {}: {e}", path.display())))?;
        Profiles::parse(file.bytes())
    }
}

/// Options `serve --model` hands its family helper instead of serving with,
/// or applies itself before the package serves: load threads and the weight
/// keeper are selected for the process, and the package's own parse has no
/// checkpoint flag for a keeper to lend to.
pub(crate) const MODEL_OPTIONS: &[&str] = &[
    "--model",
    "--model-cache-dir",
    "--tensor-parallel-size",
    "--activation-policy",
    "--modalities",
    "--load-threads",
    "--weight-keeper",
    "--weight-keeper-policy",
    "--weight-keeper-identity",
];

/// The serve options of a `serve --model` invocation other than the model
/// selection and preparation options.
pub(crate) fn serving_options(arguments: &[String]) -> Vec<String> {
    options(arguments)
        .into_iter()
        .filter(|(flag, _)| !MODEL_OPTIONS.contains(&flag.as_str()))
        .flat_map(|(flag, values)| std::iter::once(flag).chain(values))
        .collect()
}

/// The model selection and preparation options of `arguments`, which
/// `serve --model` applies itself: what [`serving_options`] leaves out.
pub(crate) fn preparation_options(arguments: &[String]) -> Vec<String> {
    options(arguments)
        .into_iter()
        .filter(|(flag, _)| MODEL_OPTIONS.contains(&flag.as_str()))
        .flat_map(|(flag, values)| std::iter::once(flag).chain(values))
        .collect()
}

/// The serving profile a family helper named, checked as a profile file's
/// entries are.
pub(crate) fn direct_entry(
    profile: &runtime_bundle::preparation::DirectProfile,
) -> Result<Profile, HostCommandErrorV1> {
    let entry: Profile = serde_json::from_value(profile.profile.clone())
        .map_err(|e| error(format!("serving profile {}: {e}", profile.name)))?;
    entry.validate()?;
    Ok(entry)
}

/// Emit only defaults the typed merge retained, in the profile's public order.
pub(crate) fn profile_arguments(profile: &Profile, retained: &ServingInputsV1) -> Vec<String> {
    retain_arguments(&profile.arguments(""), retained)
}

fn retain_arguments(words: &[String], retained: &ServingInputsV1) -> Vec<String> {
    options(words)
        .into_iter()
        .filter(|(flag, values)| {
            if flag == "--model-option" {
                values
                    .first()
                    .and_then(|value| value.split_once('='))
                    .is_some_and(|(name, _)| retained.model_options.contains_key(name))
            } else {
                retained.contains(flag)
            }
        })
        .flat_map(|(flag, values)| std::iter::once(flag).chain(values))
        .collect()
}

/// `serve` arguments for a direct package at `bundle`: the family's serving
/// profile as defaults under the invocation's `explicit` serve options.
pub(crate) fn direct_inputs(
    profile: &runtime_bundle::preparation::DirectProfile,
    ranks: u32,
    explicit: &ServingInputsV1,
    bundle: &Path,
) -> Result<Resolution, HostCommandErrorV1> {
    let entry = direct_entry(profile)?;
    if entry.tensor_parallel != ranks as usize {
        return Err(error(format!(
            "serving profile {} serves {} ranks, not {ranks}",
            profile.name, entry.tensor_parallel
        )));
    }
    let mut inputs = entry.inputs_under(None, explicit)?.overlay(explicit);
    inputs.bundle = Some(
        bundle
            .to_str()
            .ok_or_else(|| error("the direct package path is not UTF-8"))?
            .to_owned(),
    );
    Ok(Resolution {
        inputs,
        profile: Some(profile.name.clone()),
        drafter: Some(entry.speculation.drafter.clone()),
        qualified_max_sequences: entry.qualified_max_sequences,
    })
}

/// Resolve standalone serving settings, with environment below explicit values.
pub(crate) fn standalone(
    mut inputs: ServingInputsV1,
) -> Result<crate::HostCommandV1, HostCommandErrorV1> {
    if inputs.bundle.is_none() {
        inputs.bundle = Some("/direct-package".into());
    }
    Resolution::plain(inputs).command()
}

/// The serve command a direct package serves with `arguments`, which hold no
/// model selection or preparation options.
#[cfg(test)]
pub(crate) fn direct_command(
    arguments: &[String],
) -> Result<crate::HostCommandV1, HostCommandErrorV1> {
    standalone(ServingInputsV1::decode(arguments)?)
}

/// How `serve --model` serves another of the family's profiles than the one
/// it picks: explicit options over the pick.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Translation {
    pub inputs: ServingInputsV1,
    pub arguments: Vec<String>,
    /// Model options the pick sets and the other profile leaves to the
    /// package; the arguments keep the pick's, and a launcher resets them to
    /// the package defaults.
    pub inherited_model_options: Vec<String>,
}

/// The explicit `serve --model` options that serve `target` where the family
/// picks `pick` at the same rank count, or why none do. A target drafts as
/// the pick does or not at all, since the package carries the pick's drafter.
pub(crate) fn translate(pick: &Profile, target: &Profile) -> Result<Translation, String> {
    translatable(pick, target)?;
    let inputs = target
        .inputs(None)
        .map_err(|e| format!("it is refused: {e}"))?;
    let arguments = profile_arguments(target, &inputs);
    translate_inputs(pick, &inputs, arguments)
}

/// Why `target` cannot serve over `pick` whatever its options, if it
/// cannot: it serves another rank count, or another drafter than the
/// package carries.
pub(crate) fn translatable(pick: &Profile, target: &Profile) -> Result<(), String> {
    if pick.tensor_parallel != target.tensor_parallel {
        return Err(format!(
            "it serves {} ranks and the pick {}",
            target.tensor_parallel, pick.tensor_parallel
        ));
    }
    let drafter = target.speculation.drafter.as_str();
    if drafter != "none" && drafter != pick.speculation.drafter {
        return Err(format!(
            "it drafts with {drafter}, but the package serve --model prepares for this checkpoint {}",
            match pick.speculation.drafter.as_str() {
                "none" => "has no drafter".to_owned(),
                other => format!("drafts with {other}"),
            }
        ));
    }
    Ok(())
}

/// The explicit `serve --model` options over `pick` that serve exactly what
/// `target`, direct serve options without model selection, serves:
/// `target` without its device map, which the launcher sets, and with an
/// explicit value for each setting the pick sets and `target` leaves at its
/// default. That value is `target`'s own effective one; a setting whose
/// default no flag value names makes the translation fail. The result is
/// accepted only when it serves `target`'s exact configuration, model
/// options the pick sets and `target` omits aside.
pub(crate) fn translate_inputs(
    pick: &Profile,
    target: &ServingInputsV1,
    words: Vec<String>,
) -> Result<Translation, String> {
    let reference = standalone(target.clone()).map_err(|e| format!("it is refused: {e}"))?;
    let order = pick.option_order();
    let mut arguments: Vec<String> = options(&words)
        .into_iter()
        .filter(|(flag, _)| flag != "--devices")
        .flat_map(|(flag, values)| std::iter::once(flag).chain(values))
        .collect();
    let mut explicit = target.clone();
    explicit.devices = None;
    let mut resets = Vec::new();
    loop {
        let flag = order.iter().copied().find(|flag| {
            *flag != "--devices" && *flag != "--model-option" && !explicit.replaces(flag)
        });
        let Some(flag) = flag else {
            break;
        };
        if !explicit
            .restore(flag, &reference)
            .map_err(|e| format!("its options over the pick are refused: {e}"))?
        {
            let values = options(&pick.arguments(""))
                .into_iter()
                .find(|(option, _)| option == flag)
                .map(|(_, values)| values.join(" "))
                .unwrap_or_default();
            return Err(format!(
                "the pick sets {flag} {values}, and no flag value restores the default it leaves"
            ));
        }
        resets.push(flag);
    }
    let inherited_model_options: Vec<String> = pick
        .model_options
        .keys()
        .filter(|name| !target.model_options.contains_key(*name))
        .cloned()
        .collect();
    let defaults = pick
        .inputs_under(None, &explicit)
        .map_err(|e| format!("its options over the pick are refused: {e}"))?;
    let mut served = standalone(defaults.overlay(&explicit))
        .map_err(|e| format!("its options over the pick are refused: {e}"))?;
    let mut reference = reference;
    for command in [&mut served, &mut reference] {
        if let crate::HostCommandV1::Serve { model_options, .. } = command {
            for name in &inherited_model_options {
                model_options.remove(name);
            }
        }
    }
    if served != reference {
        let (ours, theirs) = (
            crate::command::effective_settings_v1(&served),
            crate::command::effective_settings_v1(&reference),
        );
        let setting = ours
            .keys()
            .chain(theirs.keys())
            .find(|key| ours.get(*key) != theirs.get(*key));
        return Err(match setting {
            Some(key) => format!(
                "over the pick it would serve --{key} {}, not {}",
                ours.get(key).map_or("unset", String::as_str),
                theirs.get(key).map_or("unset", String::as_str)
            ),
            None => "over the pick it would serve other options than its own".to_owned(),
        });
    }
    // Effective text is an output projection, never resolver input.
    let values = crate::command::effective_settings_v1(&reference);
    for flag in resets {
        arguments.extend([
            flag.to_owned(),
            values[flag.trim_start_matches("--")].clone(),
        ]);
    }
    Ok(Translation {
        inputs: explicit,
        arguments,
        inherited_model_options,
    })
}

pub(crate) fn inspect(arguments: &[String]) -> Result<String, HostCommandErrorV1> {
    match arguments {
        [verb, path] if verb == "list" => {
            let file = Profiles::read(Path::new(path))?;
            Ok(file
                .profiles
                .iter()
                .map(|(name, p)| {
                    format!(
                        "{name}\t{}\tTP{}\t{} devices\t{}",
                        match p.status {
                            Status::Qualified => "serving evidence",
                            Status::Experimental => "experimental",
                        },
                        p.tensor_parallel,
                        p.devices_required,
                        p.qualification
                    )
                })
                .collect::<Vec<_>>()
                .join("\n"))
        }
        [verb, path, name] if verb == "show" => {
            let file = Profiles::read(Path::new(path))?;
            let profile = file.get(name)?;
            let crate::HostCommandV1::Serve { serving, .. } =
                Resolution::plain(profile.inputs(Some(&file.served_model_name))?).command()?
            else {
                unreachable!("profile resolves to serve");
            };
            let mut shown = serde_json::to_value(profile).map_err(|e| error(e.to_string()))?;
            let target = match serving.prefill_budget {
                runtime_scheduler::PrefillBudget::Tbt(policy) => Some(policy.target_millis()),
                _ => None,
            };
            shown["reply_history_observer"] = serving.reply_history_observer.into();
            shown["prefill_policy"] = serving.prefill_policy.name().into();
            if let Some(target) = target {
                shown["decode_tbt_target_ms"] = target.into();
            }
            serde_json::to_string_pretty(&shown).map_err(|e| error(e.to_string()))
        }
        _ => Err(error("profiles list FILE | profiles show FILE NAME")),
    }
}

#[cfg(test)]
pub(crate) fn direct_defaults(profile: &Profile) -> Vec<String> {
    profile_arguments(profile, &profile.inputs(None).unwrap())
}

#[cfg(test)]
pub(crate) fn apply_defaults(defaults: &[String], explicit: &[String]) -> Vec<String> {
    let defaults_input = ServingInputsV1::decode(defaults).unwrap();
    let explicit_input = ServingInputsV1::decode(explicit).unwrap();
    let mut words = retain_arguments(defaults, &defaults_input.inherited(&explicit_input));
    words.extend_from_slice(explicit);
    words
}

#[cfg(test)]
fn translate_options(pick: &Profile, target: &[String]) -> Result<Translation, String> {
    let inputs = ServingInputsV1::decode(target).map_err(|e| format!("it is refused: {e}"))?;
    translate_inputs(pick, &inputs, target.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolve(words: &[String]) -> Result<Resolution, HostCommandErrorV1> {
        resolve_inputs(ServingInputsV1::decode(words)?)
    }

    fn direct_resolution(
        profile: &runtime_bundle::preparation::DirectProfile,
        ranks: u32,
        words: &[String],
        bundle: &Path,
    ) -> Result<Resolution, HostCommandErrorV1> {
        direct_inputs(profile, ranks, &ServingInputsV1::decode(words)?, bundle)
    }

    fn fixture() -> serde_json::Value {
        serde_json::json!({
            "schema":"chopinfer-serve-profiles-v2", "model":"fixture", "served_model_name":"fixture",
            "profiles":{"test":{
                "bundle_id":"fixture.v1", "topology":"fixture-tp2", "tensor_parallel":2,
                "devices_required":2,"devices":[0,1],
                "speculation":{"mode":"adaptive","tokens":3,"draft_cost":"per-token","drafter":"test"},
                "max_sequences":8,"qualified_max_sequences":8,"qualification":"CPU fixture only",
                "kv_cache_dtype":"bf16","prefix_cache":false,"max_batch_tokens":8192,
                "prefill_chunk_tokens":2048,"queue_depth":1024,"gpu_memory_utilization":1.0,
                "status":"experimental","evidence":["fixture"]
            }}
        })
    }
    /// A test's own directory under the temporary directory, removed when
    /// the test ends, a failed test's included.
    struct Scratch(std::path::PathBuf);

    impl Scratch {
        /// A new, empty `{label}-{process}-{serial}` under the temporary
        /// directory.
        fn new(label: &str) -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "{label}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            // A reused process ID may have left one behind.
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl std::ops::Deref for Scratch {
        type Target = std::path::Path;

        fn deref(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl AsRef<std::path::Path> for Scratch {
        fn as_ref(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A package directory whose manifest names the fixture profile's bundle
    /// at two ranks, for `--profile` to serve.
    fn package(label: &str) -> Scratch {
        let directory = Scratch::new(&format!("profile-package-{label}"));
        std::fs::write(
            directory.join("manifest.json"),
            br#"{"id":"fixture.v1","target":{"topologies":[{"id":"fixture-tp2","rank_count":2}]}}"#,
        )
        .unwrap();
        directory
    }
    #[test]
    fn serving_diagnostic_parity_checks_profiles_before_explicit_errors() {
        let package = package("diagnostic-parity");
        let profile = package.join("profiles.json");
        std::fs::write(&profile, serde_json::to_vec(&fixture()).unwrap()).unwrap();
        let missing_package = package.join("missing");
        let missing_profile = package.join("missing.json");
        let reference = format!("{}#test", profile.display());
        let absent = format!("{}#test", missing_profile.display());
        let unknown = format!("{}#absent", profile.display());
        let parse = |words: &[String]| crate::parse_host_command_v1(words).unwrap_err().to_string();
        for (bundle, selected) in [
            (None, &reference),
            (Some(missing_package.as_path()), &reference),
            (Some(&*package), &absent),
            (Some(&*package), &unknown),
        ] {
            let mut words = vec!["serve".to_owned(), "--profile".into(), selected.clone()];
            if let Some(bundle) = bundle {
                words.extend(["--bundle".into(), bundle.display().to_string()]);
            }
            let expected = parse(&words);
            for invalid in [
                vec!["--max-sequences", "invalid"],
                vec!["--scenario", "commit"],
                vec!["--unknown", "value"],
                vec!["--max-sequences"],
                vec!["--check", "--check"],
                vec!["--bundle", "another"],
            ] {
                if bundle.is_none() && invalid[0] == "--bundle" {
                    continue;
                }
                let mut invalid_words = words.clone();
                invalid_words.extend(invalid.iter().map(|word| (*word).to_owned()));
                assert_eq!(parse(&invalid_words), expected, "{invalid_words:?}");
            }
        }
    }

    /// The profile lookup reads the manifest as a bounded regular file: a
    /// link, a FIFO (without blocking) and a file past the manifest bound are
    /// refused before any of it is read into memory.
    #[test]
    fn package_facts_read_only_a_bounded_regular_manifest() {
        let regular = package("facts-regular");
        let facts = PackageFacts::read(&regular).unwrap();
        assert_eq!(
            (facts.id.as_str(), facts.topologies),
            ("fixture.v1", vec![("fixture-tp2".to_owned(), 2)])
        );
        let root = Scratch::new("profile-package-facts");
        let linked = root.join("linked");
        std::fs::create_dir_all(&linked).unwrap();
        std::os::unix::fs::symlink(regular.join("manifest.json"), linked.join("manifest.json"))
            .unwrap();
        let fifo = root.join("fifo");
        std::fs::create_dir_all(&fifo).unwrap();
        assert!(
            std::process::Command::new("mkfifo")
                .arg(fifo.join("manifest.json"))
                .status()
                .unwrap()
                .success()
        );
        let large = root.join("large");
        std::fs::create_dir_all(&large).unwrap();
        std::fs::File::create(large.join("manifest.json"))
            .unwrap()
            .set_len(runtime_bundle::BUNDLE_MANIFEST_MAX_BYTES as u64 + 1)
            .unwrap();
        for (directory, expected) in [
            (&linked, "read "),
            (&fifo, "not a regular file"),
            (&large, "exceeds its bound"),
        ] {
            let refusal = PackageFacts::read(directory).err().unwrap().to_string();
            assert!(refusal.contains(expected), "{refusal}");
        }
    }

    fn budget(command: crate::HostCommandV1) -> runtime_scheduler::PrefillBudget {
        let crate::HostCommandV1::Serve { serving, .. } = command else {
            panic!("serve");
        };
        serving.prefill_budget
    }

    #[test]
    fn profile_adapter_allows_replacing_invalid_generic_direct_defaults() {
        for (field, value, words) in [
            (
                "graph_memory_reserve_mib",
                serde_json::json!(0),
                vec!["--graph-memory-reserve-mib", "1024"],
            ),
            (
                "gpu_memory_utilization",
                serde_json::json!(0),
                vec!["--gpu-memory-utilization", "1"],
            ),
            (
                "kv_cache_dtype",
                serde_json::json!("invalid"),
                vec!["--kv-cache-dtype", "auto"],
            ),
            (
                "model_options",
                serde_json::json!({"budget": "x".repeat(1025)}),
                vec!["--model-option", "budget=4"],
            ),
        ] {
            let mut entry = fixture()["profiles"]["test"].clone();
            entry[field] = value;
            let profile = runtime_bundle::preparation::DirectProfile {
                name: "test".into(),
                profile: entry,
            };
            let explicit = words.into_iter().map(String::from).collect::<Vec<_>>();
            let resolution =
                direct_resolution(&profile, 2, &explicit, Path::new("/direct-package"))
                    .unwrap_or_else(|error| panic!("{field}: {error}"));
            assert!(resolution.command().is_ok(), "{field}");
            assert!(
                direct_resolution(&profile, 2, &[], Path::new("/direct-package"))
                    .and_then(|resolution| resolution.command())
                    .is_err(),
                "{field}"
            );
        }
    }

    #[test]
    fn profile_assignment_names_keep_the_first_equals_boundary() {
        let mut entry = fixture()["profiles"]["test"].clone();
        entry["model_options"] = serde_json::json!({"budget=small": 1});
        let mut profile = runtime_bundle::preparation::DirectProfile {
            name: "test".into(),
            profile: entry,
        };
        let resolved = direct_resolution(&profile, 2, &[], Path::new("/direct-package")).unwrap();
        let crate::HostCommandV1::Serve { model_options, .. } = resolved.command().unwrap() else {
            panic!("serve")
        };
        assert_eq!(model_options["budget"], "small=1");
        profile.profile["model_options"]["budget"] = 2.into();
        assert!(
            direct_resolution(&profile, 2, &[], Path::new("/direct-package"))
                .and_then(|resolved| resolved.command())
                .is_err()
        );
        let explicit = ["--model-option", "budget=4"].map(String::from);
        let resolved =
            direct_resolution(&profile, 2, &explicit, Path::new("/direct-package")).unwrap();
        let crate::HostCommandV1::Serve { model_options, .. } = resolved.command().unwrap() else {
            panic!("serve")
        };
        assert_eq!(model_options["budget"], "4");
    }

    #[test]
    fn profile_adapter_rejects_flag_looking_served_names() {
        let mut value = fixture();
        value["served_model_name"] = "--alias".into();
        let error = Profiles::parse(&serde_json::to_vec(&value).unwrap())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("--served-model-name requires at least one name"),
            "{error}"
        );
    }

    #[test]
    fn model_option_precedence_preserves_source_roles() {
        // Isolate the process environment from parallel Rust tests.
        if std::env::var_os("CHOPINFER_TEST_PROFILE_SOURCES").is_none() {
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "profiles::tests::model_option_precedence_preserves_source_roles",
                ])
                .env("CHOPINFER_TEST_PROFILE_SOURCES", "1")
                .env("CHOPINFER_MODEL_OPTIONS", "budget=8,mode=fast")
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stdout)
            );
            return;
        }
        let mut entry = fixture()["profiles"]["test"].clone();
        entry["model_options"] = serde_json::json!({"budget": 2});
        let profile = runtime_bundle::preparation::DirectProfile {
            name: "test".into(),
            profile: entry,
        };
        for (explicit, expected) in [(vec![], "8"), (vec!["--model-option", "budget=4"], "4")] {
            let words: Vec<String> = [
                "serve",
                "--model",
                "checkpoint",
                "--tensor-parallel-size",
                "2",
            ]
            .into_iter()
            .chain(explicit)
            .map(String::from)
            .collect();
            let crate::HostCommandV1::Serve {
                selection: crate::preparation::ServeSelection::Model(model),
                ..
            } = crate::parse_host_command_v1(&words).unwrap()
            else {
                panic!("model")
            };
            let resolved = direct_resolution(
                &profile,
                2,
                &model.serve_arguments,
                Path::new("/direct-package"),
            )
            .unwrap();
            let crate::HostCommandV1::Serve { model_options, .. } = resolved.command().unwrap()
            else {
                panic!("serve")
            };
            assert_eq!(model_options["budget"], expected);
            assert_eq!(model_options["mode"], "fast");
        }
        let entry = direct_entry(&profile).unwrap();
        let crate::HostCommandV1::Serve { model_options, .. } =
            direct_command(&direct_defaults(&entry)).unwrap()
        else {
            panic!("serve")
        };
        assert_eq!(model_options["budget"], "2");
        assert_eq!(model_options["mode"], "fast");
    }

    #[test]
    fn zero_token_profiles_gate_companions_before_explicit_tokens() {
        let package = package("ar-token-override");
        let path = package.join("profiles.json");
        for speculation in [
            serde_json::json!({"mode":"fixed", "tokens":0, "draft_cost":"per-block", "drafter":"none"}),
            serde_json::json!({"mode":"balanced", "tokens":0, "draft_cost":"per-block", "drafter":"none",
                              "depths":"1:3,8:0", "widths":"4,8"}),
            serde_json::json!({"mode":"table", "tokens":0, "draft_cost":"per-block", "drafter":"none",
                              "costs":"1:13500:4400:700:760,8:22300:4400:700:780"}),
        ] {
            let mut file = fixture();
            file["profiles"]["test"]["speculation"] = speculation;
            std::fs::write(&path, serde_json::to_vec(&file).unwrap()).unwrap();
            let direct = runtime_bundle::preparation::DirectProfile {
                name: "test".into(),
                profile: file["profiles"]["test"].clone(),
            };
            let explicit = ["--speculative-tokens", "3"].map(String::from);
            let direct = direct_resolution(&direct, 2, &explicit, &package).unwrap();
            let packaged = [
                "serve".into(),
                "--bundle".into(),
                package.to_str().unwrap().into(),
                "--profile".into(),
                format!("{}#test", path.display()),
                "--speculative-tokens".into(),
                "3".into(),
            ];
            for command in [
                direct.command().unwrap(),
                crate::parse_host_command_v1(&packaged).unwrap(),
            ] {
                let crate::HostCommandV1::Serve {
                    speculative_tokens,
                    speculation_mode,
                    draft_cost_model,
                    verify_widths,
                    ..
                } = command
                else {
                    panic!("serve")
                };
                assert_eq!(speculative_tokens, 3);
                assert_eq!(speculation_mode, runtime_engine::SpeculationMode::Adaptive);
                assert_eq!(draft_cost_model, runtime_engine::DraftCostModel::PerBlock);
                assert_eq!(verify_widths, runtime_engine::VerifyWidths::default());
            }
        }
    }

    /// Explicit speculation options over a direct profile resolve with the
    /// profile's tokens, and are refused where the profile has none.
    #[test]
    fn explicit_speculation_options_resolve_with_the_profile_tokens() {
        let entry = fixture()["profiles"]["test"].clone();
        let resolve = |entry: &serde_json::Value, explicit: &[&str]| {
            let profile = runtime_bundle::preparation::DirectProfile {
                name: "test".into(),
                profile: entry.clone(),
            };
            let explicit: Vec<String> = explicit.iter().map(|w| (*w).into()).collect();
            let resolved =
                direct_resolution(&profile, 2, &explicit, Path::new("/cache/p-x")).unwrap();
            resolved.command()
        };
        let crate::HostCommandV1::Serve {
            speculative_tokens,
            speculation_mode,
            ..
        } = resolve(&entry, &["--speculation", "fixed"]).unwrap()
        else {
            panic!("serve");
        };
        assert_eq!(speculative_tokens, 3);
        assert_eq!(speculation_mode, runtime_engine::SpeculationMode::Fixed);
        let mut autoregressive = entry.clone();
        autoregressive["speculation"]["tokens"] = 0.into();
        autoregressive["speculation"]["mode"] = "fixed".into();
        autoregressive["speculation"]["drafter"] = "none".into();
        let refusal = resolve(&autoregressive, &["--speculation", "fixed"])
            .unwrap_err()
            .to_string();
        assert!(
            refusal.contains("requires positive --speculative-tokens"),
            "{refusal}"
        );
    }

    /// A table profile passes its costs; explicit options replace or drop
    /// them with the mode.
    #[test]
    fn table_profiles_pass_their_costs_unless_the_mode_changes() {
        let costs = "1:13500:4400:700:760,8:22300:4400:700:780";
        let mut entry = fixture()["profiles"]["test"].clone();
        entry["speculation"]["mode"] = "table".into();
        entry["speculation"]["costs"] = costs.into();
        let profile = runtime_bundle::preparation::DirectProfile {
            name: "test".into(),
            profile: entry.clone(),
        };
        let mode = |explicit: &[&str]| {
            let explicit: Vec<String> = explicit.iter().map(|w| (*w).into()).collect();
            let resolved =
                direct_resolution(&profile, 2, &explicit, Path::new("/cache/p-x")).unwrap();
            match resolved.command().unwrap() {
                crate::HostCommandV1::Serve {
                    speculative_tokens,
                    speculation_mode,
                    ..
                } => (speculative_tokens, speculation_mode),
                _ => panic!("serve"),
            }
        };
        let table = runtime_engine::SpeculationMode::Table(
            runtime_engine::StepCosts::parse(costs).unwrap(),
        );
        assert_eq!(mode(&[]), (3, table));
        assert_eq!(mode(&["--speculation", "table"]), (3, table));
        let other = "1:10000:1:0:1";
        assert_eq!(
            mode(&["--speculation-costs", other]),
            (
                3,
                runtime_engine::SpeculationMode::Table(
                    runtime_engine::StepCosts::parse(other).unwrap()
                )
            )
        );
        assert_eq!(
            mode(&["--speculation", "fixed"]),
            (3, runtime_engine::SpeculationMode::Fixed)
        );
        assert_eq!(
            mode(&["--speculation", "balanced"]),
            (
                3,
                runtime_engine::SpeculationMode::Balanced(Default::default())
            )
        );
        assert_eq!(mode(&["--speculative-tokens", "0"]).0, 0);
        // Costs belong to table mode only.
        for (mode, costs) in [
            ("table", None),
            ("adaptive", Some(costs)),
            ("table", Some("1:0:0:0:0")),
        ] {
            let mut bad = entry.clone();
            bad["speculation"]["mode"] = mode.into();
            let speculation = bad["speculation"].as_object_mut().unwrap();
            match costs {
                Some(costs) => speculation.insert("costs".into(), costs.into()),
                None => speculation.remove("costs"),
            };
            let profile = runtime_bundle::preparation::DirectProfile {
                name: "test".into(),
                profile: bad,
            };
            let Err(refusal) = direct_resolution(&profile, 2, &[], Path::new("/cache/p-x")) else {
                panic!("{mode} {costs:?} accepted");
            };
            let refusal = refusal.to_string();
            assert!(refusal.contains("needs valid costs"), "{refusal}");
        }
    }

    /// A balanced profile passes its depth schedule; an explicit schedule
    /// replaces it, an explicit other mode drops it, and a schedule outside
    /// balanced mode or an invalid one is refused.
    #[test]
    fn balanced_profiles_pass_their_depths_unless_the_mode_changes() {
        let schedule = "1:5,8:3,16:0";
        let mut entry = fixture()["profiles"]["test"].clone();
        entry["speculation"]["mode"] = "balanced".into();
        entry["speculation"]["depths"] = schedule.into();
        let profile = runtime_bundle::preparation::DirectProfile {
            name: "test".into(),
            profile: entry.clone(),
        };
        let mode = |explicit: &[&str]| {
            let explicit: Vec<String> = explicit.iter().map(|w| (*w).into()).collect();
            let resolved =
                direct_resolution(&profile, 2, &explicit, Path::new("/cache/p-x")).unwrap();
            match resolved.command().unwrap() {
                crate::HostCommandV1::Serve {
                    speculation_mode, ..
                } => speculation_mode,
                _ => panic!("serve"),
            }
        };
        let balanced = |text: &str| {
            runtime_engine::SpeculationMode::Balanced(
                runtime_engine::DraftDepths::parse(text).unwrap(),
            )
        };
        assert_eq!(mode(&[]), balanced(schedule));
        assert_eq!(mode(&["--speculation", "balanced"]), balanced(schedule));
        assert_eq!(mode(&["--speculation-depths", "8:2"]), balanced("8:2"));
        assert_eq!(
            mode(&["--speculation", "adaptive"]),
            runtime_engine::SpeculationMode::Adaptive
        );
        for (mode, depths) in [("adaptive", schedule), ("balanced", "16:0,8:3")] {
            let mut bad = entry.clone();
            bad["speculation"]["mode"] = mode.into();
            bad["speculation"]["depths"] = depths.into();
            let profile = runtime_bundle::preparation::DirectProfile {
                name: "test".into(),
                profile: bad,
            };
            let Err(refusal) = direct_resolution(&profile, 2, &[], Path::new("/cache/p-x")) else {
                panic!("{mode} {depths} accepted");
            };
            let refusal = refusal.to_string();
            assert!(
                refusal.contains("only balanced mode takes them"),
                "{refusal}"
            );
        }
    }

    /// A balanced profile passes its verify widths; explicit widths replace
    /// them, an explicit other mode drops them, and widths outside balanced
    /// mode or out of order are refused.
    #[test]
    fn balanced_profiles_pass_their_widths_unless_the_mode_changes() {
        let mut entry = fixture()["profiles"]["test"].clone();
        entry["speculation"]["mode"] = "balanced".into();
        entry["speculation"]["widths"] = "6,8,16,32".into();
        let profile = runtime_bundle::preparation::DirectProfile {
            name: "test".into(),
            profile: entry.clone(),
        };
        let resolve = |explicit: &[&str]| {
            let explicit: Vec<String> = explicit.iter().map(|w| (*w).into()).collect();
            let resolved =
                direct_resolution(&profile, 2, &explicit, Path::new("/cache/p-x")).unwrap();
            match resolved.command().unwrap() {
                crate::HostCommandV1::Serve {
                    speculation_mode,
                    verify_widths,
                    ..
                } => (speculation_mode, verify_widths.widths().to_vec()),
                _ => panic!("serve"),
            }
        };
        assert_eq!(resolve(&[]).1, [6, 8, 16, 32]);
        assert_eq!(resolve(&["--speculation", "balanced"]).1, [6, 8, 16, 32]);
        assert_eq!(resolve(&["--speculation-widths", "8,32"]).1, [8, 32]);
        assert_eq!(
            resolve(&["--speculation", "adaptive"]),
            (runtime_engine::SpeculationMode::Adaptive, Vec::new())
        );
        for (mode, widths) in [("adaptive", "6,8"), ("balanced", "16,8")] {
            let mut bad = entry.clone();
            bad["speculation"]["mode"] = mode.into();
            bad["speculation"]["widths"] = widths.into();
            let profile = runtime_bundle::preparation::DirectProfile {
                name: "test".into(),
                profile: bad,
            };
            let Err(refusal) = direct_resolution(&profile, 2, &[], Path::new("/cache/p-x")) else {
                panic!("{mode} {widths} accepted");
            };
            let refusal = refusal.to_string();
            assert!(
                refusal.contains("only balanced mode takes them"),
                "{refusal}"
            );
        }
    }

    /// A balanced profile passes its confidence calibration; an explicit
    /// calibration or `off` replaces it, an explicit other mode or zero
    /// speculative tokens drop it, and a calibration outside balanced mode,
    /// short of the tokens or malformed is refused.
    #[test]
    fn balanced_profiles_pass_their_confidence_unless_the_mode_changes() {
        let calibration = serde_json::json!({
            "contract": "fixture.drafter-confidence.v1",
            "floor": 1,
            "greedy": [[1.5, -0.25], [1.0, 0.0], [1.0, 0.0]],
            "sampled": [[0.75, 0.5], [2.0, -1.0], [0.1, 0.2]],
        });
        let mut entry = fixture()["profiles"]["test"].clone();
        entry["speculation"]["mode"] = "balanced".into();
        entry["speculation"]["confidence"] = calibration.clone();
        let profile = runtime_bundle::preparation::DirectProfile {
            name: "test".into(),
            profile: entry.clone(),
        };
        let resolve = |explicit: &[&str]| {
            let explicit: Vec<String> = explicit.iter().map(|w| (*w).into()).collect();
            let resolved =
                direct_resolution(&profile, 2, &explicit, Path::new("/cache/p-x")).unwrap();
            match resolved.command().unwrap() {
                crate::HostCommandV1::Serve {
                    verify_confidence, ..
                } => verify_confidence.map(|confidence| *confidence),
                _ => panic!("serve"),
            }
        };
        let expected = crate::confidence::parse(&calibration.to_string()).unwrap();
        assert!(expected.is_some());
        assert_eq!(resolve(&[]), expected);
        assert_eq!(resolve(&["--speculation", "balanced"]), expected);
        assert_eq!(resolve(&["--speculation-confidence", "off"]), None);
        let mut other = calibration.clone();
        other["floor"] = 0.into();
        assert_eq!(
            resolve(&["--speculation-confidence", &other.to_string()]),
            crate::confidence::parse(&other.to_string()).unwrap()
        );
        assert_eq!(resolve(&["--speculation", "adaptive"]), None);
        assert_eq!(resolve(&["--speculative-tokens", "0"]), None);
        let mut short = calibration.clone();
        short["greedy"].as_array_mut().unwrap().pop();
        short["sampled"].as_array_mut().unwrap().pop();
        let mut malformed = calibration.clone();
        malformed["floor"] = 4.into();
        for (mode, confidence) in [
            ("adaptive", calibration.clone()),
            ("balanced", short),
            ("balanced", malformed),
        ] {
            let mut bad = entry.clone();
            bad["speculation"]["mode"] = mode.into();
            bad["speculation"]["confidence"] = confidence.clone();
            let profile = runtime_bundle::preparation::DirectProfile {
                name: "test".into(),
                profile: bad,
            };
            let Err(refusal) = direct_resolution(&profile, 2, &[], Path::new("/cache/p-x")) else {
                panic!("{mode} {confidence} accepted");
            };
            assert!(
                refusal
                    .to_string()
                    .contains("profile speculation confidence"),
                "{refusal}"
            );
        }
    }

    /// A profile's token capacity and graph budget reach the serve command
    /// as `--speculative-rows` and the `verify_graph_rows` model option;
    /// explicit flags replace them, zero speculative tokens drop them, and
    /// rows within the tokens or a zero graph budget are refused.
    #[test]
    fn profiles_pass_their_verify_rows_and_graph_rows() {
        let mut entry = fixture()["profiles"]["test"].clone();
        entry["speculation"]["mode"] = "balanced".into();
        entry["speculation"]["rows"] = 128.into();
        entry["speculation"]["graph_rows"] = 64.into();
        let profile = runtime_bundle::preparation::DirectProfile {
            name: "test".into(),
            profile: entry.clone(),
        };
        let resolve = |explicit: &[&str]| {
            let explicit: Vec<String> = explicit.iter().map(|w| (*w).into()).collect();
            let resolved =
                direct_resolution(&profile, 2, &explicit, Path::new("/cache/p-x")).unwrap();
            match resolved.command().unwrap() {
                crate::HostCommandV1::Serve {
                    speculative_rows,
                    model_options,
                    ..
                } => (
                    speculative_rows,
                    model_options
                        .get(runtime_bundle_abi::VERIFY_GRAPH_ROWS_OPTION_V1)
                        .cloned(),
                ),
                _ => panic!("serve"),
            }
        };
        assert_eq!(resolve(&[]), (Some(128), Some("64".into())));
        assert_eq!(
            resolve(&["--speculative-rows", "32", "--speculation-graph-rows", "32"]),
            (Some(32), Some("32".into()))
        );
        assert_eq!(resolve(&["--speculative-tokens", "0"]), (None, None));
        for (field, value) in [("rows", 3), ("graph_rows", 0)] {
            let mut bad = entry.clone();
            bad["speculation"][field] = value.into();
            let profile = runtime_bundle::preparation::DirectProfile {
                name: "test".into(),
                profile: bad,
            };
            let Err(refusal) = direct_resolution(&profile, 2, &[], Path::new("/cache/p-x")) else {
                panic!("{field} {value} accepted");
            };
            assert!(
                refusal.to_string().contains("profile speculation rows"),
                "{refusal}"
            );
        }
    }

    #[test]
    fn balanced_profiles_resolve_without_costs_and_keep_cli_overrides() {
        let mut entry = fixture()["profiles"]["test"].clone();
        entry["speculation"]["mode"] = "balanced".into();
        let profile = runtime_bundle::preparation::DirectProfile {
            name: "test".into(),
            profile: entry,
        };
        for (args, expected) in [
            (
                Vec::new(),
                runtime_engine::SpeculationMode::Balanced(Default::default()),
            ),
            (
                vec!["--speculation".into(), "fixed".into()],
                runtime_engine::SpeculationMode::Fixed,
            ),
        ] {
            let resolved = direct_resolution(&profile, 2, &args, Path::new("/cache/p-x")).unwrap();
            let crate::HostCommandV1::Serve {
                speculation_mode, ..
            } = resolved.command().unwrap()
            else {
                panic!("serve");
            };
            assert_eq!(speculation_mode, expected);
        }
    }

    #[test]
    fn a_direct_profile_supplies_serving_defaults_under_explicit_options() {
        let entry = fixture()["profiles"]["test"].clone();
        let profile = runtime_bundle::preparation::DirectProfile {
            name: "test".into(),
            profile: entry.clone(),
        };
        let inputs = |explicit: &[&str]| {
            let explicit: Vec<String> = explicit.iter().map(|word| (*word).into()).collect();
            direct_resolution(&profile, 2, &explicit, Path::new("/cache/p-x"))
                .unwrap()
                .inputs
        };
        let resolved = inputs(&["--max-sequences", "4", "--served-model-name", "a", "b"]);
        assert_eq!(resolved.bundle.as_deref(), Some("/cache/p-x"));
        assert!(resolved.topology.is_none());
        assert_eq!(resolved.devices, Some(vec![0, 1]));
        assert_eq!(resolved.maximum_sequences, Some(4));
        assert_eq!(resolved.speculative_tokens, Some(3));
        assert_eq!(
            resolved.speculation_mode,
            Some(runtime_engine::SpeculationMode::Adaptive)
        );
        assert_eq!(resolved.http_options.served_model_names, ["a", "b"]);
        assert!(standalone(resolved).is_ok());
        let off = inputs(&["--speculative-tokens", "0"]);
        assert_eq!(off.speculative_tokens, Some(0));
        assert!(off.speculation_mode.is_none());
        let resolution = direct_resolution(&profile, 2, &[], Path::new("/cache/p-x")).unwrap();
        assert_eq!(resolution.drafter.as_deref(), Some("test"));
        assert_eq!(resolution.profile.as_deref(), Some("test"));
        assert!(direct_resolution(&profile, 4, &[], Path::new("/p")).is_err());
        let mut invalid = profile.clone();
        invalid.profile["unknown"] = true.into();
        assert!(direct_resolution(&invalid, 2, &[], Path::new("/p")).is_err());
        assert_eq!(
            serving_options(&[
                "--model".into(),
                "m".into(),
                "--load-threads".into(),
                "4".into(),
                "--fake-host-services".into(),
                "--api-key".into(),
                "--model".into(),
            ]),
            ["--fake-host-services", "--api-key", "--model"]
        );
    }

    #[test]
    fn scheduling_fields_validate_through_the_serve_parser() {
        use runtime_scheduler::{MixedPolicy, PrefillBudget};
        for (policy, target, expected) in [
            (None, None, PrefillBudget::Fixed),
            (Some("fixed"), None, PrefillBudget::Fixed),
            (
                Some("adaptive"),
                None,
                PrefillBudget::Tbt(MixedPolicy::default()),
            ),
            (
                Some("tbt"),
                None,
                PrefillBudget::Tbt(MixedPolicy::default()),
            ),
            (
                Some("tbt"),
                Some(1),
                PrefillBudget::Tbt(MixedPolicy::new(1, 512, 0).unwrap()),
            ),
            (
                Some("tbt"),
                Some(150),
                PrefillBudget::Tbt(MixedPolicy::new(150, 512, 0).unwrap()),
            ),
            (
                Some("tbt"),
                Some(60_000),
                PrefillBudget::Tbt(MixedPolicy::new(60_000, 512, 0).unwrap()),
            ),
        ] {
            let mut value = fixture();
            if let Some(policy) = policy {
                value["profiles"]["test"]["prefill_policy"] = policy.into();
            }
            if let Some(target) = target {
                value["profiles"]["test"]["decode_tbt_target_ms"] = target.into();
            }
            let file = Profiles::parse(&serde_json::to_vec(&value).unwrap()).unwrap();
            let profile = file.get("test").unwrap();
            assert_eq!(
                serde_json::to_value(profile).unwrap(),
                value["profiles"]["test"]
            );
            let mut args = vec!["serve".into()];
            args.extend(profile.arguments(&file.served_model_name));
            assert_eq!(
                budget(crate::parse_host_command_v1(&args).unwrap()),
                expected
            );
        }
        for fields in [
            serde_json::json!({"prefill_policy":"unknown"}),
            serde_json::json!({"prefill_policy":false}),
            serde_json::json!({"decode_tbt_target_ms":150}),
            serde_json::json!({"prefill_policy":"fixed","decode_tbt_target_ms":150}),
            serde_json::json!({"prefill_policy":"retired","decode_tbt_target_ms":150}),
            serde_json::json!({"prefill_policy":"tbt","decode_tbt_target_ms":0}),
            serde_json::json!({"prefill_policy":"tbt","decode_tbt_target_ms":60001}),
            serde_json::json!({"prefill_policy":"tbt","decode_tbt_target_ms":-1}),
            serde_json::json!({"prefill_policy":"tbt","decode_tbt_target_ms":1.5}),
            serde_json::json!({"prefill_policy":"tbt","decode_tbt_target_ms":"150"}),
        ] {
            let mut value = fixture();
            value["profiles"]["test"]
                .as_object_mut()
                .unwrap()
                .extend(fields.as_object().unwrap().clone());
            assert!(
                Profiles::parse(&serde_json::to_vec(&value).unwrap()).is_err(),
                "{fields}"
            );
        }
    }

    #[test]
    fn scheduling_profile_is_inspectable_and_cli_overrides_in_either_order() {
        use runtime_scheduler::{MixedPolicy, PrefillBudget};
        let mut value = fixture();
        value["profiles"]["test"]["prefill_policy"] = "tbt".into();
        value["profiles"]["test"]["decode_tbt_target_ms"] = 150.into();
        let path =
            std::env::temp_dir().join(format!("scheduling-profile-{}.json", std::process::id()));
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let reference = format!("{}#test", path.display());
        let bundle = package("scheduling");
        let parse = |extra: &[&str], profile_first: bool| {
            let mut args = vec![
                "serve".to_string(),
                "--bundle".into(),
                bundle.display().to_string(),
            ];
            let profile = ["--profile".to_string(), reference.clone()];
            if profile_first {
                args.extend(profile.clone());
            }
            args.extend(extra.iter().map(|s| s.to_string()));
            if !profile_first {
                args.extend(profile);
            }
            crate::parse_host_command_v1(&args)
        };
        for profile_first in [true, false] {
            for (extra, expected) in [
                (
                    vec![],
                    PrefillBudget::Tbt(MixedPolicy::new(150, 512, 0).unwrap()),
                ),
                (
                    vec!["--prefill-policy", "tbt"],
                    PrefillBudget::Tbt(MixedPolicy::new(150, 512, 0).unwrap()),
                ),
                (
                    vec!["--decode-tbt-target-ms", "75"],
                    PrefillBudget::Tbt(MixedPolicy::new(75, 512, 0).unwrap()),
                ),
                (vec!["--prefill-policy", "fixed"], PrefillBudget::Fixed),
                (
                    vec!["--prefill-policy", "adaptive"],
                    PrefillBudget::Tbt(MixedPolicy::new(150, 512, 0).unwrap()),
                ),
            ] {
                assert_eq!(budget(parse(&extra, profile_first).unwrap()), expected);
            }
            for extra in [
                vec!["--prefill-policy", "fixed", "--decode-tbt-target-ms", "75"],
                vec![
                    "--prefill-policy",
                    "decode-aware",
                    "--decode-tbt-target-ms",
                    "75",
                ],
                vec!["--decode-tbt-target-ms", "0"],
                vec!["--decode-tbt-target-ms", "60001"],
                vec!["--prefill-policy", "fixed", "--prefill-policy", "tbt"],
                vec![
                    "--decode-tbt-target-ms",
                    "75",
                    "--decode-tbt-target-ms",
                    "80",
                ],
            ] {
                assert!(parse(&extra, profile_first).is_err(), "{extra:?}");
            }
        }
        let args = ["profiles", "show", path.to_str().unwrap(), "test"].map(String::from);
        let crate::HostCommandV1::Display { text } = crate::parse_host_command_v1(&args).unwrap()
        else {
            panic!("profile inspection");
        };
        let mut expected = value["profiles"]["test"].clone();
        expected["reply_history_observer"] = false.into();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&text).unwrap(),
            expected
        );

        // Old profiles still default to fixed and accept an explicit TBT override.
        std::fs::write(&path, serde_json::to_vec(&fixture()).unwrap()).unwrap();
        assert_eq!(budget(parse(&[], true).unwrap()), PrefillBudget::Fixed);
        assert_eq!(
            budget(
                parse(
                    &["--prefill-policy", "tbt", "--decode-tbt-target-ms", "150"],
                    false
                )
                .unwrap()
            ),
            PrefillBudget::Tbt(MixedPolicy::new(150, 512, 0).unwrap())
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn reply_history_profile_defaults_and_explicit_overrides() {
        for value in [None, Some(false), Some(true)] {
            let mut data = fixture();
            if let Some(value) = value {
                data["profiles"]["test"]["reply_history_observer"] = value.into();
            }
            let file = Profiles::parse(&serde_json::to_vec(&data).unwrap()).unwrap();
            let defaults = file.get("test").unwrap().arguments(&file.served_model_name);
            for explicit in [None, Some(false), Some(true)] {
                let flags = explicit
                    .map(|v| {
                        vec![
                            "--reply-history-observer".into(),
                            if v { "on" } else { "off" }.into(),
                        ]
                    })
                    .unwrap_or_default();
                let mut args = vec!["serve".into()];
                args.extend(apply_defaults(&defaults, &flags));
                let crate::HostCommandV1::Serve { serving, .. } =
                    crate::parse_host_command_v1(&args).unwrap()
                else {
                    panic!("serve");
                };
                assert_eq!(
                    serving.reply_history_observer,
                    explicit.or(value).unwrap_or(false)
                );
            }
        }
        let mut data = fixture();
        data["profiles"]["test"]["reply_history_observer"] = "on".into();
        assert!(Profiles::parse(&serde_json::to_vec(&data).unwrap()).is_err());
    }

    #[test]
    fn prefill_policy_profile_resolution_and_flag_override() {
        for value in ["fixed", "tbt", "adaptive"] {
            let mut data = fixture();
            data["profiles"]["test"]["prefill_policy"] = value.into();
            if value != "fixed" {
                data["profiles"]["test"]["decode_tbt_target_ms"] = 150.into();
            }
            let file = Profiles::parse(&serde_json::to_vec(&data).unwrap()).unwrap();
            let defaults = file.get("test").unwrap().arguments(&file.served_model_name);
            for override_value in [None, Some("fixed"), Some("tbt"), Some("adaptive")] {
                let explicit = override_value
                    .map(|v| vec!["--prefill-policy".into(), v.into()])
                    .unwrap_or_default();
                let mut args = vec!["serve".into()];
                args.extend(apply_defaults(&defaults, &explicit));
                let crate::HostCommandV1::Serve { serving, .. } =
                    crate::parse_host_command_v1(&args).unwrap()
                else {
                    panic!("serve command");
                };
                assert_eq!(
                    serving.prefill_policy.name(),
                    override_value.unwrap_or(value)
                );
                if let runtime_scheduler::PrefillBudget::Tbt(policy) = serving.prefill_budget {
                    assert_eq!(
                        policy.target_millis(),
                        if value == "fixed" { 100 } else { 150 }
                    );
                    assert_eq!(policy.min_prefill_tokens(), 512);
                    assert_eq!(policy.decode_steps(), 0);
                } else {
                    assert_eq!(
                        serving.prefill_budget,
                        runtime_scheduler::PrefillBudget::Fixed
                    );
                    assert_eq!(serving.prefill_policy, crate::PrefillPolicyV1::Fixed);
                }
            }
        }
    }

    #[test]
    fn prefill_policy_file_profile_override_and_inspection() {
        let path =
            std::env::temp_dir().join(format!("timing-cuts-profile-{}.json", std::process::id()));
        let bundle = package("timing-cuts");
        let mut data = fixture();
        for default in ["fixed", "tbt", "adaptive"] {
            data["profiles"]["test"]["prefill_policy"] = default.into();
            std::fs::write(&path, serde_json::to_vec(&data).unwrap()).unwrap();
            for profile_first in [false, true] {
                for explicit in ["fixed", "tbt", "adaptive"] {
                    let mut args = vec![
                        "serve".into(),
                        "--bundle".into(),
                        bundle.display().to_string(),
                    ];
                    let profile = vec!["--profile".into(), format!("{}#test", path.display())];
                    let flags = vec!["--prefill-policy".into(), explicit.into()];
                    if profile_first {
                        args.extend(profile);
                        args.extend(flags);
                    } else {
                        args.extend(flags);
                        args.extend(profile);
                    }
                    let crate::HostCommandV1::Serve { serving, .. } =
                        crate::parse_host_command_v1(&args).unwrap()
                    else {
                        panic!("serve command");
                    };
                    assert_eq!(serving.prefill_policy.name(), explicit);
                    assert!(
                        format!("{serving:?}")
                            .contains(&format!("prefill_policy: {:?}", serving.prefill_policy))
                    );
                }
            }
            let args = ["profiles", "show", path.to_str().unwrap(), "test"].map(String::from);
            let crate::HostCommandV1::Display { text } =
                crate::parse_host_command_v1(&args).unwrap()
            else {
                panic!("profile inspection");
            };
            let shown: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!(shown["prefill_policy"], default);
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn inspection_reports_effective_scheduling_defaults() {
        let path =
            std::env::temp_dir().join(format!("inspect-profile-{}.json", std::process::id()));
        for (fields, policy, target) in [
            (serde_json::json!({}), "fixed", None),
            (serde_json::json!({"prefill_policy":"fixed"}), "fixed", None),
            (
                serde_json::json!({"prefill_policy":"adaptive"}),
                "adaptive",
                Some(100),
            ),
            (
                serde_json::json!({"prefill_policy":"tbt"}),
                "tbt",
                Some(100),
            ),
            (
                serde_json::json!({"prefill_policy":"tbt","decode_tbt_target_ms":150}),
                "tbt",
                Some(150),
            ),
        ] {
            let mut value = fixture();
            value["profiles"]["test"]
                .as_object_mut()
                .unwrap()
                .extend(fields.as_object().unwrap().clone());
            std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
            let args = ["profiles", "show", path.to_str().unwrap(), "test"].map(String::from);
            let crate::HostCommandV1::Display { text } =
                crate::parse_host_command_v1(&args).unwrap()
            else {
                panic!("profile inspection");
            };
            let mut expected = value["profiles"]["test"].clone();
            expected["reply_history_observer"] = false.into();
            expected["prefill_policy"] = policy.into();
            if let Some(target) = target {
                expected["decode_tbt_target_ms"] = target.into();
            }
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&text).unwrap(),
                expected
            );
        }
        std::fs::remove_file(path).unwrap();
    }

    /// A profile serves its own topology, which the package must declare at
    /// the profile's rank count, so another device count is refused rather
    /// than served as another topology; an explicit `--topology` replaces
    /// it.
    #[test]
    fn a_profile_serves_its_topology_of_a_package() {
        let path =
            std::env::temp_dir().join(format!("topology-profile-{}.json", std::process::id()));
        std::fs::write(&path, serde_json::to_vec(&fixture()).unwrap()).unwrap();
        let reference = format!("{}#test", path.display());
        let package = package("topologies");
        std::fs::write(
            package.join("manifest.json"),
            br#"{"id":"fixture.v1","target":{"topologies":[{"id":"fixture-tp2","rank_count":2},{"id":"fixture-tp4","rank_count":4}]}}"#,
        )
        .unwrap();
        let bundle = package.to_str().unwrap();
        let topology = |explicit: &[&str]| {
            let words: Vec<String> = ["--bundle", bundle, "--profile", &reference]
                .iter()
                .chain(explicit)
                .map(|word| (*word).to_owned())
                .collect();
            resolve(&words).unwrap().inputs.topology.unwrap()
        };
        assert_eq!(topology(&[]), "fixture-tp2");
        assert_eq!(topology(&["--devices", "0,1,2,3"]), "fixture-tp2");
        assert_eq!(topology(&["--topology", "fixture-tp4"]), "fixture-tp4");
        // A package that renamed the profile's topology, or declares it at
        // another rank count, is refused, even with a device list another
        // topology would take and even when it declares only one topology.
        for manifest in [
            r#"{"id":"fixture.v1","target":{"topologies":[{"id":"renamed-tp2","rank_count":2},{"id":"fixture-tp4","rank_count":4}]}}"#,
            r#"{"id":"fixture.v1","target":{"topologies":[{"id":"renamed-tp2","rank_count":2}]}}"#,
            r#"{"id":"fixture.v1","target":{"topologies":[{"id":"fixture-tp2","rank_count":4}]}}"#,
        ] {
            std::fs::write(package.join("manifest.json"), manifest).unwrap();
            for explicit in [&[][..], &["--devices", "0,1"]] {
                let words: Vec<String> = ["--bundle", bundle, "--profile", &reference]
                    .iter()
                    .chain(explicit)
                    .map(|word| (*word).to_owned())
                    .collect();
                let refusal = resolve(&words).err().expect(manifest).to_string();
                assert!(
                    refusal.contains("topology fixture-tp2 at 2 ranks"),
                    "{refusal}"
                );
            }
        }
        std::fs::remove_file(path).unwrap();
    }

    /// Writes `profiles` into `package` as its serving profiles component,
    /// declared in `manifest` with their length and SHA-256.
    fn carry_profiles(package: &Path, mut manifest: serde_json::Value, profiles: &[u8]) {
        use runtime_bundle::deployment::{SERVE_PROFILES_PATH_V1, SERVE_PROFILES_ROLE_V1};
        std::fs::create_dir_all(package.join("components")).unwrap();
        std::fs::write(package.join(SERVE_PROFILES_PATH_V1), profiles).unwrap();
        manifest["components"] = serde_json::json!([{
            "id": "serve-profiles", "role": SERVE_PROFILES_ROLE_V1,
            "format": runtime_bundle::deployment::SERVE_PROFILES_FORMAT_V1,
            "path": SERVE_PROFILES_PATH_V1, "byte_length": profiles.len(),
            "sha256": runtime_weights::sha256_bytes(profiles),
        }]);
        std::fs::write(package.join("manifest.json"), manifest.to_string()).unwrap();
    }

    /// A bare name resolves from the profiles the package carries, read
    /// only when their bytes match the manifest; a name the package's file
    /// lacks, bytes that changed and a second profiles component are
    /// refused.
    #[test]
    fn a_bare_name_resolves_from_the_package_profiles() {
        let package = package("carried");
        let bundle = package.to_str().unwrap();
        let manifest = serde_json::json!({
            "id": "fixture.v1",
            "target": {"topologies": [{"id": "fixture-tp2", "rank_count": 2}]},
        });
        let profiles = serde_json::to_vec(&fixture()).unwrap();
        carry_profiles(&package, manifest.clone(), &profiles);
        let file =
            std::env::temp_dir().join(format!("carried-profile-{}.json", std::process::id()));
        std::fs::write(&file, &profiles).unwrap();
        let explicit = format!("{}#test", file.display());
        let words =
            |reference: &str| ["--bundle", bundle, "--profile", reference].map(String::from);
        let carried = resolve(&words("test")).unwrap();
        let named = resolve(&words(&explicit)).unwrap();
        assert_eq!(carried.inputs, named.inputs);
        assert_eq!(carried.profile.as_deref(), Some("test"));
        assert_eq!(carried.drafter.as_deref(), Some("test"));
        let refusal = resolve(&words("other")).err().unwrap().to_string();
        assert!(refusal.contains("unknown profile 'other'"), "{refusal}");
        let mut changed = profiles.clone();
        *changed.last_mut().unwrap() = b' ';
        std::fs::write(
            package.join(runtime_bundle::deployment::SERVE_PROFILES_PATH_V1),
            &changed,
        )
        .unwrap();
        let refusal = resolve(&words("test")).err().unwrap().to_string();
        assert!(refusal.contains("serving profiles"), "{refusal}");
        carry_profiles(&package, manifest, &profiles);
        let mut twice: serde_json::Value =
            serde_json::from_slice(&std::fs::read(package.join("manifest.json")).unwrap()).unwrap();
        let component = twice["components"][0].clone();
        twice["components"].as_array_mut().unwrap().push(component);
        std::fs::write(package.join("manifest.json"), twice.to_string()).unwrap();
        let refusal = resolve(&words("test")).err().unwrap().to_string();
        assert!(
            refusal.contains("more than one serving profiles"),
            "{refusal}"
        );
        std::fs::remove_file(file).unwrap();
    }

    /// Every model's profiles, carried by a package of their bundle and
    /// topology, resolve under a bare name to exactly the serve options the
    /// model's `profiles.json` gives them as `FILE#NAME`.
    #[test]
    fn carried_model_profiles_resolve_as_their_files() {
        let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
        let mut checked = 0;
        for entry in std::fs::read_dir(&models).unwrap().flatten() {
            let path = entry.path().join("profiles.json");
            if !path.is_file() {
                continue;
            }
            let bytes = std::fs::read(&path).unwrap();
            let file = Profiles::read(&path).unwrap();
            for (name, profile) in &file.profiles {
                let mut topology = serde_json::json!({
                    "id": profile.topology, "rank_count": profile.tensor_parallel,
                });
                if let Some(size) = profile.context_parallel_size.filter(|&size| size > 1) {
                    topology["context_parallel_sizes"] = serde_json::json!([size]);
                }
                let mut manifest = serde_json::json!({
                    "id": profile.bundle_id, "target": {"topologies": [topology]},
                });
                if !profile.model_options.is_empty() {
                    let options: Vec<_> = profile
                        .model_option_texts()
                        .into_iter()
                        .map(|(option, value)| {
                            serde_json::json!({"name": option, "definition": {
                                "type": "string", "default": value, "allowed_values": [value],
                            }, "default_is_qualified": true})
                        })
                        .collect();
                    manifest["model_options"] = serde_json::json!({
                        "schema": runtime_bundle::model_options::CAPABILITY, "options": options,
                    });
                }
                let package = package(&format!("model-{checked}"));
                carry_profiles(&package, manifest, &bytes);
                let bundle = package.to_str().unwrap();
                let explicit = format!("{}#{name}", path.display());
                let words = |reference: &str| {
                    ["--bundle", bundle, "--profile", reference].map(String::from)
                };
                let carried = resolve(&words(name)).unwrap();
                let named = resolve(&words(&explicit)).unwrap();
                assert_eq!(carried.inputs, named.inputs, "{}#{name}", path.display());
                assert_eq!(carried.profile, named.profile);
                assert_eq!(carried.drafter, named.drafter);
                assert_eq!(
                    carried.qualified_max_sequences,
                    named.qualified_max_sequences
                );
                checked += 1;
            }
        }
        assert!(checked > 0);
    }

    /// Every model profile that names no context-parallel size reads,
    /// shows and resolves its serve options exactly as before the field
    /// existed: no `--context-parallel-size`, and the file's own fields.
    #[test]
    fn profiles_without_a_context_parallel_size_are_unchanged() {
        let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
        let mut checked = 0;
        for entry in std::fs::read_dir(&models).unwrap().flatten() {
            let path = entry.path().join("profiles.json");
            if !path.is_file() {
                continue;
            }
            let text: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            let file = Profiles::read(&path).unwrap();
            for (name, profile) in &file.profiles {
                if text["profiles"][name]
                    .get("context_parallel_size")
                    .is_some()
                {
                    continue;
                }
                assert_eq!(
                    serde_json::to_value(profile).unwrap(),
                    text["profiles"][name],
                    "{name}"
                );
                let arguments = profile.arguments(&file.served_model_name);
                assert!(
                    !arguments
                        .iter()
                        .any(|word| word == "--context-parallel-size"),
                    "{name}"
                );
                checked += 1;
            }
        }
        assert!(checked > 0);
    }

    /// A context-parallel size reaches the serve options when it divides
    /// the tensor-parallel size and the package declares it for the
    /// profile's topology; otherwise it is refused.
    #[test]
    fn a_context_parallel_size_must_divide_and_be_declared() {
        let mut value = fixture();
        value["profiles"]["test"]["context_parallel_size"] = 2.into();
        let path = std::env::temp_dir().join(format!("cp-profile-{}.json", std::process::id()));
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let file = Profiles::read(&path).unwrap();
        let joined = file.get("test").unwrap().arguments("").join(" ");
        assert!(joined.contains("--context-parallel-size 2"), "{joined}");
        for size in [0, 3] {
            let mut odd = fixture();
            odd["profiles"]["test"]["context_parallel_size"] = size.into();
            std::fs::write(&path, serde_json::to_vec(&odd).unwrap()).unwrap();
            let refusal = Profiles::read(&path).err().unwrap().to_string();
            assert!(refusal.contains("must divide"), "{size}: {refusal}");
        }
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let reference = format!("{}#test", path.display());
        let package = package("context-parallel");
        let bundle = package.to_str().unwrap();
        let words = ["--bundle", bundle, "--profile", &reference].map(String::from);
        // The fixture package declares no context-parallel size.
        let refusal = resolve(&words).err().unwrap().to_string();
        assert!(
            refusal.contains("serves context-parallel size 2; the package declares []"),
            "{refusal}"
        );
        std::fs::write(
            package.join("manifest.json"),
            br#"{"id":"fixture.v1","target":{"topologies":[{"id":"fixture-tp2","rank_count":2,"context_parallel_sizes":[2]}]}}"#,
        )
        .unwrap();
        assert_eq!(
            resolve(&words).unwrap().inputs.context_parallel_size,
            Some(2)
        );
        std::fs::remove_file(path).unwrap();
    }

    /// A profile's own model options must resolve against the package's
    /// declaration before explicit options replace them: an explicit
    /// `--model-option` with the package's value does not admit a profile
    /// that pins another. Other explicit options still override.
    #[test]
    fn a_profile_model_option_must_be_one_the_package_accepts() {
        let mut value = fixture();
        value["profiles"]["test"]["model_options"] =
            serde_json::json!({"layout": "a", "budget": 1});
        let path = std::env::temp_dir().join(format!("option-profile-{}.json", std::process::id()));
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let reference = format!("{}#test", path.display());
        let package = package("model-options");
        let bundle = package.to_str().unwrap();
        let manifest = |layout: &str| {
            let manifest = serde_json::json!({
                "id": "fixture.v1",
                "target": {"topologies": [{"id": "fixture-tp2", "rank_count": 2}]},
                "model_options": {"schema": runtime_bundle::model_options::CAPABILITY, "options": [
                    {"name": "budget", "definition": {"type": "integer", "default": 0, "minimum": 0, "maximum": 9}, "default_is_qualified": true},
                    {"name": "layout", "definition": {"type": "string", "default": layout, "allowed_values": [layout]}, "default_is_qualified": true}
                ]}
            });
            std::fs::write(package.join("manifest.json"), manifest.to_string()).unwrap();
        };
        let words = |explicit: &[&str]| {
            let mut words = vec!["--bundle", bundle, "--profile", &reference];
            words.extend(explicit);
            words.into_iter().map(String::from).collect::<Vec<_>>()
        };
        manifest("b");
        for explicit in [&[][..], &["--model-option", "layout=b"]] {
            let refusal = resolve(&words(explicit)).err().unwrap().to_string();
            assert!(
                refusal.contains("profile test sets model options the package does not accept"),
                "{explicit:?}: {refusal}"
            );
        }
        manifest("a");
        let resolved = resolve(&words(&["--model-option", "budget=5"]))
            .unwrap()
            .inputs;
        assert_eq!(resolved.model_options["layout"], "a");
        assert_eq!(resolved.model_options["budget"], "5");
        // A package without a declaration accepts no profile option.
        std::fs::write(
            package.join("manifest.json"),
            br#"{"id":"fixture.v1","target":{"topologies":[{"id":"fixture-tp2","rank_count":2}]}}"#,
        )
        .unwrap();
        let refusal = resolve(&words(&[])).err().unwrap().to_string();
        assert!(refusal.contains("the package declares none"), "{refusal}");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn v1_profiles_require_migration() {
        let mut value = fixture();
        value["schema"] = serde_json::json!("chopinfer-serve-profiles-v1");
        value["profiles"]["test"]["fold_ranks"] = serde_json::json!(false);
        let refusal = Profiles::parse(&serde_json::to_vec(&value).unwrap())
            .unwrap_err()
            .to_string();
        assert!(refusal.contains("drop fold_ranks"), "{refusal}");
        assert!(refusal.contains("chopinfer-serve-profiles-v2"), "{refusal}");
    }

    #[test]
    fn v2_profiles_refuse_fold_ranks() {
        for folded in [false, true] {
            let mut value = fixture();
            value["schema"] = serde_json::json!("chopinfer-serve-profiles-v2");
            value["profiles"]["test"]["fold_ranks"] = serde_json::json!(folded);
            let refusal = Profiles::parse(&serde_json::to_vec(&value).unwrap())
                .unwrap_err()
                .to_string();
            assert!(refusal.contains("unknown field `fold_ranks`"), "{refusal}");
        }
    }

    #[test]
    fn every_rank_requires_a_distinct_device() {
        for (required, devices) in [(1, vec![0, 0]), (1, vec![0, 1]), (3, vec![0, 1])] {
            let mut value = fixture();
            value["profiles"]["test"]["devices_required"] = required.into();
            value["profiles"]["test"]["devices"] = serde_json::json!(devices);
            let refusal = Profiles::parse(&serde_json::to_vec(&value).unwrap())
                .unwrap_err()
                .to_string();
            assert!(
                refusal.contains("rank/device mapping is inconsistent"),
                "{refusal}"
            );
        }
    }

    #[test]
    fn duplicate_model_option_keys_are_refused() {
        let json = serde_json::to_string(&fixture()).unwrap();
        for options in [
            r#"{"budget":1,"budget":2}"#,
            r#"{"budget":1,"budget":1}"#,
            r#"{"budget":1,"bud\u0067et":2}"#,
        ] {
            let duplicate = json.replace(
                "\"bundle_id\":",
                &format!("\"model_options\":{options},\"bundle_id\":"),
            );
            let refusal = Profiles::parse(duplicate.as_bytes())
                .unwrap_err()
                .to_string();
            assert!(refusal.contains("duplicate model option"), "{refusal}");
        }
        let unique = json.replace(
            "\"bundle_id\":",
            r#""model_options":{"budget":1,"mode":"fast"},"bundle_id":"#,
        );
        let parsed = Profiles::parse(unique.as_bytes()).unwrap();
        assert_eq!(
            parsed.profiles["test"].model_options["budget"],
            serde_json::json!(1)
        );
        assert_eq!(
            parsed.profiles["test"].model_options["mode"],
            serde_json::json!("fast")
        );
    }

    #[test]
    fn duplicate_profile_fields_are_refused() {
        let json = serde_json::to_string(&fixture()).unwrap().replace(
            "\"schema\":",
            "\"schema\":\"chopinfer-serve-profiles-v2\",\"schema\":",
        );
        let refusal = Profiles::parse(json.as_bytes()).unwrap_err().to_string();
        assert!(refusal.contains("duplicate field `schema`"), "{refusal}");
    }

    #[test]
    fn strict_schema_and_nested_fields() {
        assert!(Profiles::parse(&serde_json::to_vec(&fixture()).unwrap()).is_ok());
        for pointer in ["", "/profiles/test", "/profiles/test/speculation"] {
            let mut value = fixture();
            value
                .pointer_mut(pointer)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .insert("typo".into(), true.into());
            assert!(Profiles::parse(&serde_json::to_vec(&value).unwrap()).is_err());
        }
        for (pointer, replacement) in [
            ("/schema", serde_json::json!("v2")),
            ("/profiles/test/max_sequences", serde_json::json!(0)),
            ("/profiles/test/kv_cache_dtype", serde_json::json!("typo")),
            ("/profiles/test/devices", serde_json::json!([0, 0])),
            (
                "/profiles/test/gpu_memory_utilization",
                serde_json::json!(1.1),
            ),
        ] {
            let mut value = fixture();
            *value.pointer_mut(pointer).unwrap() = replacement;
            assert!(
                Profiles::parse(&serde_json::to_vec(&value).unwrap()).is_err(),
                "{pointer}"
            );
        }
        // The replaced field names its replacement.
        let mut value = fixture();
        let profile = value
            .pointer_mut("/profiles/test")
            .unwrap()
            .as_object_mut()
            .unwrap();
        profile.remove("gpu_memory_utilization");
        profile.insert("kv_memory_fraction".into(), 0.9.into());
        let refusal = Profiles::parse(&serde_json::to_vec(&value).unwrap())
            .unwrap_err()
            .to_string();
        assert!(
            refusal.contains("kv_memory_fraction is gone: set gpu_memory_utilization"),
            "{refusal}"
        );
    }
    #[test]
    fn a_profile_names_its_graph_memory_reserve() {
        let reserve = |mib: Option<serde_json::Value>| -> Result<u64, HostCommandErrorV1> {
            let mut value = fixture();
            if let Some(mib) = mib {
                value["profiles"]["test"]["graph_memory_reserve_mib"] = mib;
            }
            let profiles = Profiles::parse(&serde_json::to_vec(&value).unwrap())?;
            let mut args = vec!["serve".to_owned()];
            args.extend(profiles.get("test").unwrap().arguments("fixture"));
            match crate::parse_host_command_v1(&args)? {
                crate::HostCommandV1::Serve { serving, .. } => {
                    Ok(serving.graph_memory_reserve_bytes)
                }
                _ => unreachable!("profile arguments name serve"),
            }
        };
        assert_eq!(reserve(None).unwrap(), 1 << 30);
        assert_eq!(reserve(Some(1280.into())).unwrap(), 1280 << 20);
        for refused in [
            serde_json::json!(0),
            serde_json::json!(1.5),
            serde_json::json!(-1),
        ] {
            assert!(reserve(Some(refused)).is_err());
        }
    }
    #[test]
    fn explicit_overrides_in_either_order_and_ar_disables_policy() {
        let path = std::env::temp_dir().join(format!("serve-profile-{}.json", std::process::id()));
        std::fs::write(&path, serde_json::to_vec(&fixture()).unwrap()).unwrap();
        let reference = format!("{}#test", path.display());
        let package = package("overrides");
        let bundle = package.to_str().unwrap();
        for words in [
            vec![
                "serve",
                "--bundle",
                bundle,
                "--profile",
                &reference,
                "--max-sequences",
                "4",
                "--speculative-tokens",
                "0",
            ],
            vec![
                "serve",
                "--bundle",
                bundle,
                "--max-sequences",
                "4",
                "--speculative-tokens",
                "00",
                "--profile",
                &reference,
            ],
        ] {
            let args = words.into_iter().map(String::from).collect::<Vec<_>>();
            let crate::HostCommandV1::Serve {
                serving,
                speculative_tokens,
                ..
            } = crate::parse_host_command_v1(&args).unwrap()
            else {
                panic!()
            };
            assert_eq!(serving.maximum_sequences, 4);
            assert_eq!(speculative_tokens, 0);
        }
        let args = [
            "serve",
            "--bundle",
            bundle,
            "--profile",
            &reference,
            "--max-sequences",
            "4",
            "--max-sequences",
            "2",
        ]
        .map(String::from);
        assert!(crate::parse_host_command_v1(&args).is_err());
        assert!(Profiles::read(&path).unwrap().get("missing").is_err());
        let args = [
            "serve",
            "--bundle",
            bundle,
            "--profile",
            &reference,
            "--api-key",
            "--max-sequences",
        ]
        .map(String::from);
        let crate::HostCommandV1::Serve { serving, .. } =
            crate::parse_host_command_v1(&args).unwrap()
        else {
            panic!()
        };
        assert_eq!(serving.maximum_sequences, 8);
        let words = ["--bundle", "fixture", "--api-key", "--profile"].map(String::from);
        assert_eq!(
            resolve(&words).unwrap().inputs,
            ServingInputsV1::decode(&words).unwrap()
        );
        let args = [
            "serve",
            "--bundle",
            bundle,
            "--profile",
            &reference,
            "--served-model-name",
            "a",
            "b",
        ]
        .map(String::from);
        let crate::HostCommandV1::Serve {
            http_options,
            profile_name,
            ..
        } = crate::parse_host_command_v1(&args).unwrap()
        else {
            panic!()
        };
        assert_eq!(http_options.served_model_names, ["a", "b"]);
        assert_eq!(profile_name.as_deref(), Some("test"));
        for flag in crate::command::serve_options::valueless_flags_v1() {
            let words = [flag, "--bundle", bundle, "--profile", &reference].map(String::from);
            let resolved = resolve(&words).unwrap();
            assert!(resolved.inputs.contains(flag));
            assert_eq!(resolved.drafter.as_deref(), Some("test"));
            assert_eq!(resolved.profile.as_deref(), Some("test"));
        }
        // A profile serves an explicit package of its bundle and rank count.
        let refusal = resolve(&["--profile", &reference].map(String::from))
            .err()
            .unwrap()
            .to_string();
        assert!(
            refusal.contains("add --bundle DIR --checkpoint DIR"),
            "{refusal}"
        );
        std::fs::write(
            package.join("manifest.json"),
            br#"{"id":"other.v1","target":{"topologies":[{"id":"t","rank_count":2}]}}"#,
        )
        .unwrap();
        let refusal = resolve(&["--bundle", bundle, "--profile", &reference].map(String::from))
            .err()
            .unwrap()
            .to_string();
        assert!(refusal.contains("the package is other.v1"), "{refusal}");
        // A bare name is one of the package's own profiles; this package
        // carries none.
        let refusal = resolve(&["--bundle", bundle, "--profile", "test"].map(String::from))
            .err()
            .unwrap()
            .to_string();
        assert!(
            refusal.contains("carries no serving profiles; pass --profile FILE#NAME"),
            "{refusal}"
        );
        std::fs::remove_file(path).unwrap();
    }
}

#[cfg(test)]
mod model_options_tests {
    use super::*;
    #[test]
    fn explicit_options_replace_only_the_named_profile_default() {
        let defaults = ["--model-option", "a=1", "--model-option", "b=2"].map(String::from);
        let explicit = ["--model-option", "a=3"].map(String::from);
        assert_eq!(
            apply_defaults(&defaults, &explicit),
            ["--model-option", "b=2", "--model-option", "a=3"]
        );
    }
}

#[cfg(test)]
mod translate_tests {
    use super::*;

    /// A profile at four ranks: autoregressive, with `changes` applied.
    fn profile(changes: serde_json::Value) -> Profile {
        let mut entry = serde_json::json!({
            "bundle_id": "fixture.v1", "topology": "fixture-tp4", "tensor_parallel": 4,
            "devices_required": 4, "devices": [0, 1, 2, 3],
            "speculation": {"mode": "fixed", "tokens": 0, "draft_cost": "per-token",
                            "drafter": "none"},
            "max_sequences": 32, "qualified_max_sequences": null,
            "qualification": "CPU fixture only", "kv_cache_dtype": "auto",
            "prefix_cache": true, "max_batch_tokens": 8192, "prefill_chunk_tokens": 2048,
            "queue_depth": null, "gpu_memory_utilization": 1.0,
            "graph_memory_reserve_mib": 320, "status": "experimental",
            "evidence": ["fixture"],
        });
        for (field, value) in changes.as_object().unwrap() {
            entry[field] = value.clone();
        }
        let profile: Profile = serde_json::from_value(entry).unwrap();
        profile.validate().unwrap();
        profile
    }

    /// A drafting profile with a TBT prefill budget.
    fn drafting(changes: serde_json::Value) -> Profile {
        let mut base = serde_json::json!({
            "speculation": {"mode": "adaptive", "tokens": 7, "draft_cost": "per-block",
                            "drafter": "block"},
            "prefill_policy": "tbt", "decode_tbt_target_ms": 150,
            "graph_memory_reserve_mib": 384,
        });
        for (field, value) in changes.as_object().unwrap() {
            base[field] = value.clone();
        }
        profile(base)
    }

    /// The serve command `pick` serves under `arguments`, as `serve --model`
    /// resolves it, beside the one `target` serves on its own.
    fn served(pick: &Profile, arguments: &[String]) -> crate::HostCommandV1 {
        direct_command(&apply_defaults(&direct_defaults(pick), arguments)).unwrap()
    }

    fn flags(arguments: &[String]) -> Vec<String> {
        options(arguments)
            .into_iter()
            .map(|(flag, _)| flag)
            .collect()
    }

    #[test]
    fn profile_adapter_translates_replaced_invalid_pick_defaults() {
        let pick = profile(serde_json::json!({
            "graph_memory_reserve_mib": 0, "gpu_memory_utilization": 0,
            "kv_cache_dtype": "invalid", "model_options": {"budget": "x".repeat(1025)},
        }));
        let mut target = profile(serde_json::json!({"model_options": {"budget": 4}}));
        target.graph_memory_reserve_mib = None;
        let translated = translate(&pick, &target).unwrap();
        assert!(
            translated
                .arguments
                .windows(2)
                .any(|pair| pair == ["--graph-memory-reserve-mib", "1024"])
        );
    }

    #[test]
    fn translation_keeps_profile_explicit_and_reset_argument_order() {
        let pick = drafting(serde_json::json!({
            "context_parallel_size": 2,
            "model_options": {"a": 1, "b": 2},
        }));
        let target = profile(serde_json::json!({"model_options": {"a": 3, "b": 4}}));
        let explicit = [
            "--max-sequences",
            "01",
            "--model-option",
            "a=5",
            "--prefix-cache",
            "off",
        ]
        .map(String::from);
        let target = apply_defaults(&direct_defaults(&target), &explicit);
        let translated = translate_options(&pick, &target).unwrap();
        assert_eq!(
            translated.arguments,
            [
                "--max-batch-tokens",
                "8192",
                "--prefill-chunk-tokens",
                "2048",
                "--gpu-memory-utilization",
                "1",
                "--kv-cache-dtype",
                "auto",
                "--speculative-tokens",
                "0",
                "--draft-cost",
                "per-token",
                "--graph-memory-reserve-mib",
                "320",
                "--model-option",
                "b=4",
                "--max-sequences",
                "01",
                "--model-option",
                "a=5",
                "--prefix-cache",
                "off",
                "--context-parallel-size",
                "1",
                "--prefill-policy",
                "fixed",
            ]
        );
        assert!(translated.inherited_model_options.is_empty());
    }

    /// An autoregressive profile over a drafting pick with a TBT budget
    /// resets the budget, which drops the pick's TBT target, and serves
    /// exactly its own configuration.
    #[test]
    fn an_autoregressive_profile_resets_the_picks_tbt_budget() {
        let pick = drafting(serde_json::json!({}));
        let target = profile(serde_json::json!({}));
        let translation = translate(&pick, &target).unwrap();
        let joined = translation.arguments.join(" ");
        assert!(joined.contains("--prefill-policy fixed"), "{joined}");
        assert!(joined.contains("--speculative-tokens 0"), "{joined}");
        assert!(!flags(&translation.arguments).contains(&"--devices".to_owned()));
        assert!(translation.inherited_model_options.is_empty());
        let reference = direct_command(&direct_defaults(&target)).unwrap();
        assert_eq!(served(&pick, &translation.arguments), reference);
        assert_eq!(
            crate::command::effective_settings_v1(&reference)["prefill-policy"],
            "fixed"
        );
    }

    /// A balanced pick with model options and adaptive cuts: the target's
    /// configuration is served without the TBT target, and the model option
    /// only the pick sets is inherited, for a launcher to reset.
    #[test]
    fn an_option_only_the_pick_sets_is_inherited() {
        let pick = drafting(serde_json::json!({
            "speculation": {"mode": "balanced", "tokens": 5, "draft_cost": "per-block",
                            "drafter": "block", "depths": "1:5,8:3,16:0", "widths": "6,8,16,32",
                            "rows": 64, "graph_rows": 64},
            "decode_tbt_target_ms": 750, "prefill_policy": "adaptive",
            "model_options": {"chunk": "8192", "skip": "true", "draft_items": "12"},
        }));
        let target = profile(serde_json::json!({
            "prefill_policy": "fixed",
            "model_options": {"chunk": "8192", "skip": "true"},
        }));
        let translation = translate(&pick, &target).unwrap();
        assert_eq!(translation.inherited_model_options, ["draft_items"]);
        let merged = apply_defaults(&direct_defaults(&pick), &translation.arguments);
        for absent in [
            "--decode-tbt-target-ms",
            "--speculation-depths",
            "--speculation-widths",
            "--speculative-rows",
            "--speculation-graph-rows",
        ] {
            assert!(!flags(&merged).contains(&absent.to_owned()), "{merged:?}");
        }
        let crate::HostCommandV1::Serve {
            serving,
            model_options,
            ..
        } = served(&pick, &translation.arguments)
        else {
            panic!("serve");
        };
        assert_eq!(
            serving.prefill_budget,
            runtime_scheduler::PrefillBudget::Fixed
        );
        assert_eq!(serving.prefill_policy.name(), "fixed");
        assert_eq!(model_options["draft_items"], "12");
        assert_eq!(model_options["chunk"], "8192");
    }

    /// The package carries the pick's drafter: a profile drafting with
    /// another cannot serve over it, and one sharing it or none can.
    #[test]
    fn a_profile_with_another_drafter_is_unavailable() {
        let pick = drafting(serde_json::json!({}));
        let other = drafting(serde_json::json!({
            "speculation": {"mode": "adaptive", "tokens": 3, "draft_cost": "per-token",
                            "drafter": "layer"},
        }));
        let reason = translate(&pick, &other).unwrap_err();
        assert!(
            reason.contains("drafts with layer") && reason.contains("drafts with block"),
            "{reason}"
        );
        let autoregressive = profile(serde_json::json!({}));
        let reason = translate(&autoregressive, &pick).unwrap_err();
        assert!(reason.contains("has no drafter"), "{reason}");
        let wider = drafting(serde_json::json!({"max_sequences": 16}));
        assert!(translate(&pick, &wider).is_ok());
    }

    /// A setting the pick sets whose default no flag value names (here an
    /// unbounded queue) makes the profile unavailable, naming it; one with a
    /// value is reset to the target's own.
    #[test]
    fn settings_reset_to_the_targets_value_or_refuse() {
        let pick = profile(serde_json::json!({"queue_depth": 8}));
        let target = profile(serde_json::json!({}));
        let reason = translate(&pick, &target).unwrap_err();
        assert!(reason.contains("--queue-depth 8"), "{reason}");
        let pick = drafting(serde_json::json!({
            "speculation": {"mode": "balanced", "tokens": 5, "draft_cost": "per-block",
                            "drafter": "block", "confidence": {
                                "contract": "fixture.drafter-confidence.v1", "floor": 1,
                                "greedy": [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
                                "sampled": [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]}},
            "context_parallel_size": 2,
        }));
        let mut target = drafting(serde_json::json!({
            "speculation": {"mode": "balanced", "tokens": 5, "draft_cost": "per-block",
                            "drafter": "block"},
        }));
        target.graph_memory_reserve_mib = None;
        let translation = translate(&pick, &target).unwrap();
        let joined = translation.arguments.join(" ");
        for reset in [
            "--speculation-confidence off",
            "--context-parallel-size 1",
            "--graph-memory-reserve-mib 1024",
        ] {
            assert!(joined.contains(reset), "{reset}: {joined}");
        }
        assert_eq!(
            served(&pick, &translation.arguments),
            direct_command(&direct_defaults(&target)).unwrap()
        );
    }

    /// The profile `serve --model` picks at `ranks` among `file`'s for a
    /// checkpoint whose first plannable drafter is `drafter`: the family
    /// helpers' rule (`tp<ranks>-<drafter>`, else the only such entry).
    fn candidate<'a>(file: &'a Profiles, ranks: usize, drafter: &str) -> Option<&'a str> {
        let matching: Vec<&String> = file
            .profiles
            .iter()
            .filter(|(_, profile)| {
                profile.tensor_parallel == ranks && profile.speculation.drafter == drafter
            })
            .map(|(name, _)| name)
            .collect();
        let preferred = format!(
            "tp{ranks}-{}",
            if drafter == "none" { "ar" } else { drafter }
        );
        matching
            .iter()
            .find(|name| ***name == preferred)
            .or(match matching.as_slice() {
                [only] => Some(only),
                _ => None,
            })
            .map(|name| name.as_str())
    }

    /// Checks `target` over `pick` strictly: a profile drafting as the pick
    /// does or not at all serves exactly its own configuration, as its own
    /// options followed only by resets of settings the pick sets and it
    /// leaves at their defaults, each to its own effective value, and
    /// inherits exactly the pick's model options it omits. Which settings
    /// need a reset is worked out here: the pick's settings that the target's
    /// own options leave in force under `serve --model`'s merge
    /// ([`apply_defaults`]), placement and model options aside, each reset
    /// in turn while one has a value to reset to. A refusal is accepted only
    /// for another drafter than the pick's, or for a setting left in force
    /// that no flag value resets, and the reason must name it.
    fn check_translation(
        pick: &Profile,
        target: &Profile,
        result: Result<Translation, String>,
        context: &str,
    ) {
        let flags_of = |arguments: &[String]| -> std::collections::BTreeSet<String> {
            options(arguments)
                .into_iter()
                .map(|(flag, _)| flag)
                .collect()
        };
        let pick_defaults = direct_defaults(pick);
        let pick_flags = flags_of(&pick_defaults);
        let base: Vec<String> = options(&direct_defaults(target))
            .into_iter()
            .filter(|(flag, _)| flag != "--devices")
            .flat_map(|(flag, values)| std::iter::once(flag).chain(values))
            .collect();
        let base_flags = flags_of(&base);
        let mut reference = direct_command(&direct_defaults(target)).unwrap();
        let effective = crate::command::effective_settings_v1(&reference);
        let drafter = target.speculation.drafter.as_str();
        if drafter != "none" && drafter != pick.speculation.drafter {
            let reason = result.expect_err(context);
            assert!(
                reason.contains(&format!("it drafts with {drafter},")),
                "{context}: {reason}"
            );
            match pick.speculation.drafter.as_str() {
                "none" => assert!(reason.ends_with("has no drafter"), "{context}: {reason}"),
                other => assert!(
                    reason.ends_with(&format!("drafts with {other}")),
                    "{context}: {reason}"
                ),
            }
            return;
        }
        // The pick's settings left in force over the target's options and the
        // resets so far; the launcher places each instance, and model options
        // are inherited rather than reset.
        let placed = |flag: &str| {
            crate::models::INSTANCE_OPTIONS.contains(&flag) || flag == "--model-option"
        };
        let mut explicit = base.clone();
        let mut resets = std::collections::BTreeMap::new();
        let unresettable = loop {
            let given = flags_of(&explicit);
            let left: Vec<(String, Vec<String>)> =
                options(&apply_defaults(&pick_defaults, &explicit))
                    .into_iter()
                    .filter(|(flag, _)| !placed(flag) && !given.contains(flag))
                    .collect();
            let Some((flag, value)) = left.iter().find_map(|(flag, _)| {
                let value = effective.get(flag.trim_start_matches("--"))?;
                Some((flag.clone(), value.clone()))
            }) else {
                break left;
            };
            explicit.extend([flag.clone(), value.clone()]);
            resets.insert(flag, vec![value]);
        };
        let translation = match result {
            Ok(translation) => translation,
            Err(reason) => {
                let named = unresettable.iter().find(|(flag, values)| {
                    reason.starts_with(&format!("the pick sets {flag} {},", values.join(" ")))
                });
                assert!(
                    named.is_some(),
                    "{context}: unjustified refusal: {reason}; the settings no flag resets are \
                     {unresettable:?}"
                );
                return;
            }
        };
        assert!(
            unresettable.is_empty(),
            "{context}: translated, though no flag resets {unresettable:?}"
        );
        assert!(
            translation.arguments.starts_with(&base),
            "{context}: {:?}",
            translation.arguments
        );
        let mut reset = std::collections::BTreeMap::new();
        for (flag, values) in options(&translation.arguments[base.len()..]) {
            assert!(
                !base_flags.contains(&flag),
                "{context}: {flag} is the target's own"
            );
            assert!(
                pick_flags.contains(&flag),
                "{context}: the pick leaves {flag}"
            );
            assert!(
                reset.insert(flag.clone(), values).is_none(),
                "{context}: {flag} reset twice"
            );
        }
        assert_eq!(reset, resets, "{context}: resets");
        let expected: Vec<String> = pick
            .model_options
            .keys()
            .filter(|name| !target.model_options.contains_key(*name))
            .cloned()
            .collect();
        assert_eq!(
            translation.inherited_model_options, expected,
            "{context}: inherited model options"
        );
        let mut served = served(pick, &translation.arguments);
        for command in [&mut served, &mut reference] {
            if let crate::HostCommandV1::Serve { model_options, .. } = command {
                for option in &expected {
                    model_options.remove(option);
                }
            }
        }
        assert_eq!(served, reference, "{context}");
    }

    /// The strict check accepts the translation and refuses each way one
    /// could go wrong: a reset the pick does not need, a reset to another
    /// value, an incomplete inherited set, a refusal without its reason or
    /// naming a setting that needs no reset, a translation of a setting no
    /// flag resets, and a profile with another drafter that translates.
    #[test]
    fn the_strict_check_refuses_every_wrong_translation() {
        let pick = drafting(serde_json::json!({"model_options": {"a": "1", "b": "2"}}));
        let target = profile(serde_json::json!({"model_options": {"a": "1"}}));
        let good = translate(&pick, &target).unwrap();
        check_translation(&pick, &target, Ok(good.clone()), "good");
        let refused = |result: Result<Translation, String>, expected: &str| {
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                check_translation(&pick, &target, result, "wrong")
            }))
            .expect_err(expected);
            let message = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|text| (*text).to_owned()))
                .unwrap_or_default();
            assert!(message.contains(expected), "{expected}: {message}");
        };
        let mut extra = good.clone();
        extra.arguments.extend(["--kv-pages".into(), "4".into()]);
        refused(Ok(extra), "the pick leaves --kv-pages");
        let mut other = good.clone();
        let at = other
            .arguments
            .iter()
            .position(|w| w == "--prefill-policy")
            .unwrap();
        other.arguments[at + 1] = "decode-aware".into();
        refused(Ok(other), "--prefill-policy");
        let mut missing = good.clone();
        missing.inherited_model_options.clear();
        refused(Ok(missing), "inherited model options");
        refused(Err("no reason".into()), "unjustified refusal");
        // Settings no reset is needed for: the device map the launcher sets,
        // and the speculation mode `--speculative-tokens 0` clears.
        let devices = pick
            .devices
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        for (flag, value) in [
            ("--devices", devices.as_str()),
            ("--speculation", "adaptive"),
        ] {
            assert!(
                options(&direct_defaults(&pick)).contains(&(flag.into(), vec![value.into()])),
                "{flag}"
            );
            refused(
                Err(format!(
                    "the pick sets {flag} {value}, and no flag value restores the default it \
                     leaves"
                )),
                "unjustified refusal",
            );
        }
        // A setting no flag value resets is a reason.
        let queued = profile(serde_json::json!({"queue_depth": 8}));
        let unbounded = profile(serde_json::json!({}));
        let reason = translate(&queued, &unbounded).unwrap_err();
        check_translation(&queued, &unbounded, Err(reason), "queue");
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_translation(&queued, &unbounded, Ok(good.clone()), "queue")
        }));
        assert!(panic.is_err());
        let mut layer = drafting(serde_json::json!({}));
        layer.speculation.drafter = "layer".into();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_translation(&pick, &layer, Ok(good.clone()), "layer")
        }));
        assert!(panic.is_err());
    }

    /// For every model's profiles, every rank count and every profile
    /// `serve --model` can pick there, each other profile at that count is
    /// checked strictly ([`check_translation`]).
    #[test]
    fn every_model_profile_translates_exactly_or_says_why_not() {
        let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
        let mut checked = 0;
        for entry in std::fs::read_dir(&models).unwrap().flatten() {
            let path = entry.path().join("profiles.json");
            if !path.is_file() {
                continue;
            }
            let file = Profiles::read(&path).unwrap();
            let sizes: std::collections::BTreeSet<usize> = file
                .profiles
                .values()
                .map(|profile| profile.tensor_parallel)
                .collect();
            for ranks in sizes {
                let drafters: std::collections::BTreeSet<&str> = file
                    .profiles
                    .values()
                    .filter(|profile| profile.tensor_parallel == ranks)
                    .map(|profile| profile.speculation.drafter.as_str())
                    .collect();
                for pick_name in drafters
                    .into_iter()
                    .filter_map(|drafter| candidate(&file, ranks, drafter))
                {
                    let pick = file.get(pick_name).unwrap();
                    for (name, target) in &file.profiles {
                        if target.tensor_parallel != ranks || name == pick_name {
                            continue;
                        }
                        let context = format!("{}: {name} over {pick_name}", path.display());
                        check_translation(pick, target, translate(pick, target), &context);
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 0);
    }
}
