//! Model-neutral, versioned serving presets. Artifact declarations remain authoritative.
use crate::command::HostCommandErrorV1;
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
    pub max_batch_tokens: usize,
    pub prefill_chunk_tokens: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_budget: Option<String>,
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

    fn parse(bytes: &[u8]) -> Result<Self, HostCommandErrorV1> {
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
            // Reuse the serve parser, including all numeric and option constraints.
            let mut args = vec!["serve".into()];
            args.extend(profile.arguments(&file.served_model_name));
            crate::parse_host_command_v1(&args)?;
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
    fn model_option_texts(&self) -> BTreeMap<String, String> {
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
        if let Some(depth) = self.queue_depth {
            args.extend(["--queue-depth".into(), depth.to_string()]);
        }
        if let Some(size) = self.context_parallel_size {
            args.extend(["--context-parallel-size".into(), size.to_string()]);
        }
        if let Some(mib) = self.graph_memory_reserve_mib {
            args.extend(["--graph-memory-reserve-mib".into(), mib.to_string()]);
        }
        if let Some(budget) = &self.prefill_budget {
            args.extend(["--prefill-budget".into(), budget.clone()]);
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
        }
        args
    }
}

pub(crate) struct Resolution {
    pub arguments: Vec<String>,
    /// The name of the serving profile applied, if any.
    pub profile: Option<String>,
    pub drafter: Option<String>,
    pub qualified_max_sequences: Option<usize>,
}

/// The options of `arguments` as `(flag, values)` words, as the serve parser
/// reads them: valueless flags take none, `--served-model-name` takes every
/// following word up to the next option, and every other flag one value even
/// when it looks like an option.
fn options(arguments: &[String]) -> Vec<(String, Vec<String>)> {
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

/// `defaults` (a profile's options) under `explicit` options: an explicit
/// flag replaces the default, `--speculative-tokens 0` also drops the
/// default speculation mode, an explicit mode other than `table` or zero
/// speculative tokens the default speculation costs, an explicit mode other
/// than `balanced` or zero speculative tokens the default depth schedule, a
/// non-TBT `--prefill-budget` the default TBT target. Duplicate explicit
/// flags are kept for the serve parser to refuse.
pub(crate) fn apply_defaults(defaults: &[String], explicit: &[String]) -> Vec<String> {
    let explicit = options(explicit);
    let flags: std::collections::BTreeSet<&str> =
        explicit.iter().map(|(flag, _)| flag.as_str()).collect();
    let disable_speculation = explicit.iter().any(|(flag, values)| {
        flag == "--speculative-tokens"
            && values.first().and_then(|v| v.parse::<usize>().ok()) == Some(0)
    });
    let replace_costs = disable_speculation
        || explicit.iter().any(|(flag, values)| {
            flag == "--speculation" && values.first().is_some_and(|v| v != "table")
        });
    let replace_depths = disable_speculation
        || explicit.iter().any(|(flag, values)| {
            flag == "--speculation" && values.first().is_some_and(|v| v != "balanced")
        });
    let disable_tbt = explicit.iter().any(|(flag, values)| {
        flag == "--prefill-budget" && values.first().is_some_and(|v| v != "tbt")
    });
    let mut resolved = Vec::new();
    for (flag, values) in options(defaults) {
        let option_replaced = flag == "--model-option"
            && values
                .first()
                .and_then(|value| value.split_once('='))
                .is_some_and(|(name, _)| {
                    explicit.iter().any(|(flag, values)| {
                        flag == "--model-option"
                            && values
                                .first()
                                .and_then(|value| value.split_once('='))
                                .is_some_and(|(explicit_name, _)| explicit_name == name)
                    })
                });
        if (flag != "--model-option" && flags.contains(flag.as_str()))
            || option_replaced
            || (disable_speculation && flag == "--speculation")
            || (replace_costs && flag == "--speculation-costs")
            || (replace_depths && flag == "--speculation-depths")
            || (disable_tbt && flag == "--decode-tbt-target-ms")
        {
            continue;
        }
        resolved.push(flag);
        resolved.extend(values);
    }
    for (flag, values) in explicit {
        resolved.push(flag);
        resolved.extend(values);
    }
    resolved
}

pub(crate) fn resolve(arguments: &[String]) -> Result<Resolution, HostCommandErrorV1> {
    let mut reference = None;
    let mut explicit = Vec::new();
    // An option's value is never read as `--profile`, even an API key or a
    // path that looks like one.
    for (flag, values) in options(arguments) {
        if flag == "--profile" {
            if reference.is_some() {
                return Err(error("--profile given more than once"));
            }
            reference = Some(
                values
                    .into_iter()
                    .next()
                    .ok_or_else(|| error("--profile requires FILE#NAME"))?,
            );
        } else {
            explicit.push(flag);
            explicit.extend(values);
        }
    }
    let Some(reference) = reference else {
        return Ok(Resolution {
            arguments: arguments.to_vec(),
            profile: None,
            drafter: None,
            qualified_max_sequences: None,
        });
    };
    // A profile serves an explicit package, as `serve --model` serves the
    // package it prepared: its options are defaults under the explicit ones.
    let bundle = options(&explicit)
        .into_iter()
        .find(|(flag, _)| flag == "--bundle")
        .and_then(|(_, values)| values.into_iter().next())
        .ok_or_else(|| {
            error(
                "--profile applies to an explicit package: add --bundle DIR --checkpoint DIR, \
                 or serve --model DIR, which applies the family's profile itself",
            )
        })?;
    let package = PackageFacts::read(Path::new(&bundle))?;
    // A bare name is one of the package's own profiles; `FILE#NAME` names
    // another file explicitly.
    let (file, name) = match reference.rsplit_once('#') {
        Some((path, name)) => (Profiles::read(Path::new(path))?, name.to_owned()),
        None => (package.profiles(Path::new(&bundle))?, reference.clone()),
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
    let mut defaults: Vec<String> = options(&profile.arguments(""))
        .into_iter()
        .filter(|(flag, _)| !SELECTION_OPTIONS.contains(&flag.as_str()))
        .flat_map(|(flag, values)| std::iter::once(flag).chain(values))
        .collect();
    defaults.extend(["--topology".to_owned(), profile.topology.clone()]);
    Ok(Resolution {
        arguments: apply_defaults(&defaults, &explicit),
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
const MODEL_OPTIONS: &[&str] = &[
    "--model",
    "--model-cache-dir",
    "--tensor-parallel-size",
    "--activation-policy",
    "--modalities",
    "--load-threads",
    "--weight-keeper",
    "--weight-keeper-policy",
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

/// Options a direct package's serving profile does not set: the package and
/// topology are the model's selection, and the served name is the
/// package's. The profile's device map stays a default.
const SELECTION_OPTIONS: &[&str] = &["--bundle", "--topology", "--served-model-name"];

/// `serve` arguments for a direct package at `bundle`: the family's serving
/// profile as defaults under the invocation's `explicit` serve options.
pub(crate) fn direct_arguments(
    profile: &runtime_bundle::preparation::DirectProfile,
    ranks: u32,
    explicit: &[String],
    bundle: &Path,
) -> Result<Resolution, HostCommandErrorV1> {
    let entry: Profile = serde_json::from_value(profile.profile.clone())
        .map_err(|e| error(format!("serving profile {}: {e}", profile.name)))?;
    entry.validate()?;
    if entry.tensor_parallel != ranks as usize {
        return Err(error(format!(
            "serving profile {} serves {} ranks, not {ranks}",
            profile.name, entry.tensor_parallel
        )));
    }
    let defaults: Vec<String> = options(&entry.arguments(""))
        .into_iter()
        .filter(|(flag, _)| !SELECTION_OPTIONS.contains(&flag.as_str()))
        .flat_map(|(flag, values)| std::iter::once(flag).chain(values))
        .collect();
    let mut arguments = vec!["serve".to_owned()];
    arguments.extend(apply_defaults(&defaults, explicit));
    arguments.extend([
        "--bundle".to_owned(),
        bundle
            .to_str()
            .ok_or_else(|| error("the direct package path is not UTF-8"))?
            .to_owned(),
    ]);
    Ok(Resolution {
        arguments,
        profile: Some(profile.name.clone()),
        drafter: Some(entry.speculation.drafter.clone()),
        qualified_max_sequences: entry.qualified_max_sequences,
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
            let mut args = vec!["serve".into()];
            args.extend(profile.arguments(&file.served_model_name));
            let crate::HostCommandV1::Serve { serving, .. } = crate::parse_host_command_v1(&args)?
            else {
                unreachable!("profile arguments name serve");
            };
            let mut shown = serde_json::to_value(profile).map_err(|e| error(e.to_string()))?;
            let (budget, target) = match serving.prefill_budget {
                runtime_scheduler::PrefillBudget::Fixed => ("fixed", None),
                runtime_scheduler::PrefillBudget::DecodeAware(_) => ("decode-aware", None),
                runtime_scheduler::PrefillBudget::Tbt(policy) => {
                    ("tbt", Some(policy.target_millis()))
                }
            };
            shown["prefill_budget"] = budget.into();
            if let Some(target) = target {
                shown["decode_tbt_target_ms"] = target.into();
            }
            serde_json::to_string_pretty(&shown).map_err(|e| error(e.to_string()))
        }
        _ => Err(error("profiles list FILE | profiles show FILE NAME")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    /// A package directory whose manifest names the fixture profile's bundle
    /// at two ranks, for `--profile` to serve.
    fn package(label: &str) -> std::path::PathBuf {
        let directory =
            std::env::temp_dir().join(format!("profile-package-{label}-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("manifest.json"),
            br#"{"id":"fixture.v1","target":{"topologies":[{"id":"fixture-tp2","rank_count":2}]}}"#,
        )
        .unwrap();
        directory
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
        let root =
            std::env::temp_dir().join(format!("profile-package-facts-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
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
        std::fs::remove_dir_all(&root).unwrap();
    }

    fn budget(command: crate::HostCommandV1) -> runtime_scheduler::PrefillBudget {
        let crate::HostCommandV1::Serve { serving, .. } = command else {
            panic!("serve");
        };
        serving.prefill_budget
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
                direct_arguments(&profile, 2, &explicit, Path::new("/cache/p-x")).unwrap();
            crate::parse_host_command_v1(&resolved.arguments)
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
                direct_arguments(&profile, 2, &explicit, Path::new("/cache/p-x")).unwrap();
            match crate::parse_host_command_v1(&resolved.arguments).unwrap() {
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
            let Err(refusal) = direct_arguments(&profile, 2, &[], Path::new("/cache/p-x")) else {
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
                direct_arguments(&profile, 2, &explicit, Path::new("/cache/p-x")).unwrap();
            match crate::parse_host_command_v1(&resolved.arguments).unwrap() {
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
            let Err(refusal) = direct_arguments(&profile, 2, &[], Path::new("/cache/p-x")) else {
                panic!("{mode} {depths} accepted");
            };
            let refusal = refusal.to_string();
            assert!(
                refusal.contains("only balanced mode takes them"),
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
            let resolved = direct_arguments(&profile, 2, &args, Path::new("/cache/p-x")).unwrap();
            let crate::HostCommandV1::Serve {
                speculation_mode, ..
            } = crate::parse_host_command_v1(&resolved.arguments).unwrap()
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
        let words = |explicit: &[&str]| {
            let explicit: Vec<String> = explicit.iter().map(|w| (*w).into()).collect();
            direct_arguments(&profile, 2, &explicit, Path::new("/cache/p-x"))
                .unwrap()
                .arguments
        };
        let resolved = words(&["--max-sequences", "4", "--served-model-name", "a", "b"]);
        // The profile's speculation, limits and device map, not its bundle,
        // topology or served name; the package is the direct one.
        for absent in ["--topology", "fixture.v1", "fixture"] {
            assert!(!resolved.iter().any(|w| w == absent), "{resolved:?}");
        }
        let joined = resolved.join(" ");
        assert!(joined.starts_with("serve "), "{joined}");
        assert!(joined.contains("--speculative-tokens 3"), "{joined}");
        assert!(joined.contains("--speculation adaptive"), "{joined}");
        assert!(joined.contains("--max-sequences 4"), "{joined}");
        assert!(joined.contains("--devices 0,1"), "{joined}");
        assert!(!joined.contains("--max-sequences 8"), "{joined}");
        assert!(joined.contains("--served-model-name a b"), "{joined}");
        assert!(joined.ends_with("--bundle /cache/p-x"), "{joined}");
        assert!(crate::parse_host_command_v1(&resolved).is_ok());
        // Zero speculative tokens also drops the profile's speculation mode.
        let off = words(&["--speculative-tokens", "0"]).join(" ");
        assert!(!off.contains("--speculation "), "{off}");
        assert!(!off.contains("--speculative-tokens 3"), "{off}");
        let resolution = direct_arguments(&profile, 2, &[], Path::new("/cache/p-x")).unwrap();
        assert_eq!(resolution.drafter.as_deref(), Some("test"));
        assert_eq!(resolution.profile.as_deref(), Some("test"));
        assert!(direct_arguments(&profile, 4, &[], Path::new("/p")).is_err());
        let mut invalid = profile.clone();
        invalid.profile["unknown"] = true.into();
        assert!(direct_arguments(&invalid, 2, &[], Path::new("/p")).is_err());
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
        use runtime_scheduler::{MixedPolicy, PrefillBudget, PrefillTarget};
        for (policy, target, expected) in [
            (None, None, PrefillBudget::Fixed),
            (Some("fixed"), None, PrefillBudget::Fixed),
            (
                Some("decode-aware"),
                None,
                PrefillBudget::DecodeAware(PrefillTarget::default()),
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
                value["profiles"]["test"]["prefill_budget"] = policy.into();
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
            serde_json::json!({"prefill_budget":"unknown"}),
            serde_json::json!({"prefill_budget":false}),
            serde_json::json!({"decode_tbt_target_ms":150}),
            serde_json::json!({"prefill_budget":"fixed","decode_tbt_target_ms":150}),
            serde_json::json!({"prefill_budget":"decode-aware","decode_tbt_target_ms":150}),
            serde_json::json!({"prefill_budget":"tbt","decode_tbt_target_ms":0}),
            serde_json::json!({"prefill_budget":"tbt","decode_tbt_target_ms":60001}),
            serde_json::json!({"prefill_budget":"tbt","decode_tbt_target_ms":-1}),
            serde_json::json!({"prefill_budget":"tbt","decode_tbt_target_ms":1.5}),
            serde_json::json!({"prefill_budget":"tbt","decode_tbt_target_ms":"150"}),
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
        use runtime_scheduler::{MixedPolicy, PrefillBudget, PrefillTarget};
        let mut value = fixture();
        value["profiles"]["test"]["prefill_budget"] = "tbt".into();
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
                    vec!["--prefill-budget", "tbt"],
                    PrefillBudget::Tbt(MixedPolicy::new(150, 512, 0).unwrap()),
                ),
                (
                    vec!["--decode-tbt-target-ms", "75"],
                    PrefillBudget::Tbt(MixedPolicy::new(75, 512, 0).unwrap()),
                ),
                (vec!["--prefill-budget", "fixed"], PrefillBudget::Fixed),
                (
                    vec!["--prefill-budget", "decode-aware"],
                    PrefillBudget::DecodeAware(PrefillTarget::default()),
                ),
            ] {
                assert_eq!(budget(parse(&extra, profile_first).unwrap()), expected);
            }
            for extra in [
                vec!["--prefill-budget", "fixed", "--decode-tbt-target-ms", "75"],
                vec![
                    "--prefill-budget",
                    "decode-aware",
                    "--decode-tbt-target-ms",
                    "75",
                ],
                vec!["--decode-tbt-target-ms", "0"],
                vec!["--decode-tbt-target-ms", "60001"],
                vec!["--prefill-budget", "fixed", "--prefill-budget", "tbt"],
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
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&text).unwrap(),
            value["profiles"]["test"]
        );

        // Old profiles still default to fixed and accept an explicit TBT override.
        std::fs::write(&path, serde_json::to_vec(&fixture()).unwrap()).unwrap();
        assert_eq!(budget(parse(&[], true).unwrap()), PrefillBudget::Fixed);
        assert_eq!(
            budget(
                parse(
                    &["--prefill-budget", "tbt", "--decode-tbt-target-ms", "150"],
                    false
                )
                .unwrap()
            ),
            PrefillBudget::Tbt(MixedPolicy::new(150, 512, 0).unwrap())
        );
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir_all(bundle).unwrap();
    }

    #[test]
    fn inspection_reports_effective_scheduling_defaults() {
        let path =
            std::env::temp_dir().join(format!("inspect-profile-{}.json", std::process::id()));
        for (fields, policy, target) in [
            (serde_json::json!({}), "fixed", None),
            (serde_json::json!({"prefill_budget":"fixed"}), "fixed", None),
            (
                serde_json::json!({"prefill_budget":"decode-aware"}),
                "decode-aware",
                None,
            ),
            (
                serde_json::json!({"prefill_budget":"tbt"}),
                "tbt",
                Some(100),
            ),
            (
                serde_json::json!({"prefill_budget":"tbt","decode_tbt_target_ms":150}),
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
            expected["prefill_budget"] = policy.into();
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
            let arguments = resolve(&words).unwrap().arguments;
            let at = arguments
                .iter()
                .position(|word| word == "--topology")
                .unwrap();
            arguments[at + 1].clone()
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
        std::fs::remove_dir_all(&package).unwrap();
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
        assert_eq!(carried.arguments, named.arguments);
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
        std::fs::remove_dir_all(&package).unwrap();
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
                assert_eq!(
                    carried.arguments,
                    named.arguments,
                    "{}#{name}",
                    path.display()
                );
                assert_eq!(carried.profile, named.profile);
                assert_eq!(carried.drafter, named.drafter);
                assert_eq!(
                    carried.qualified_max_sequences,
                    named.qualified_max_sequences
                );
                std::fs::remove_dir_all(&package).unwrap();
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
        let resolved = resolve(&words).unwrap().arguments.join(" ");
        assert!(resolved.contains("--context-parallel-size 2"), "{resolved}");
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir_all(&package).unwrap();
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
            .arguments
            .join(" ");
        assert!(resolved.contains("--model-option layout=a"), "{resolved}");
        assert!(resolved.contains("--model-option budget=5"), "{resolved}");
        assert!(!resolved.contains("budget=1"), "{resolved}");
        // A package without a declaration accepts no profile option.
        std::fs::write(
            package.join("manifest.json"),
            br#"{"id":"fixture.v1","target":{"topologies":[{"id":"fixture-tp2","rank_count":2}]}}"#,
        )
        .unwrap();
        let refusal = resolve(&words(&[])).err().unwrap().to_string();
        assert!(refusal.contains("the package declares none"), "{refusal}");
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir_all(&package).unwrap();
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
        assert_eq!(resolve(&words).unwrap().arguments, words);
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
        for flag in crate::command::VALUELESS_FLAGS_V1 {
            let words = [*flag, "--bundle", bundle, "--profile", &reference].map(String::from);
            let resolved = resolve(&words).unwrap();
            assert!(resolved.arguments.contains(&flag.to_string()));
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
        std::fs::remove_dir_all(&package).unwrap();
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
