//! Declared inputs, the checkpoint pin and pinned references, read as
//! `tools/qual/common.py` reads them, with the reference cache binding of
//! `tools/refrun/cache.py`.
//!
//! A pinned reference is read only when its bytes have the pinned sha256,
//! its schema and checkpoint provenance equal the pin's, every pinned
//! variant is present, its workload and capture digests are the pinned ones
//! and its binding (when it has one) verifies. A binding is checked against
//! itself, the checkpoint lock, the mode policies `tools/refrun/modes.py`
//! gives (`tools/refrun/mode-policies.json`, built in), the workload it
//! scored, the fixture's items and digest and, with a reference cache store,
//! the cached scores of its entries. The adapter and runner files are
//! listed as unchecked when the root is an install rather than a checkout.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use crate::declaration::{CHECKPOINT_KEYS, CheckpointPin, Input, Loaded, ReferencePin};
use crate::json::{self, canonical_sha256, sha256_hex};
use crate::repr;

pub const BINDING: &str = "chopinfer.refcache.binding.v1";
pub const REFSET: &str = "chopinfer.refcache.refset.v1";
pub const KEY: &str = "chopinfer.refcache.key.v1";
pub const ENTRY: &str = "chopinfer.refcache.entry.v1";
/// The sealed files of a cache entry (`cache.FILES`).
const FILES: [&str; 2] = ["key.json", "scores.json"];
/// Score fields whose rows a fixture may cut to a prefix (`cache.TOPK`).
const TOPK: [&str; 3] = ["topk_ids", "topk_logits", "topk_logprobs"];
pub const WORKLOAD: &str = "chopinfer.refrun.workload.v1";
/// The runner files whose content can change a score (`keys.NUMERICS`).
pub const NUMERICS: [&str; 8] = [
    "tools/refrun/adapter.py",
    "tools/refrun/modes.py",
    "tools/refrun/ops.py",
    "tools/refrun/place.py",
    "tools/refrun/quant.py",
    "tools/refrun/run.py",
    "tools/refrun/score.py",
    "tools/refrun/workload.py",
];
/// Each mode's default dequant rule and its policy digest under every rule
/// `tools/refrun/modes.py` accepts (`cache.mode_policies`), built in so an
/// install without `tools/refrun` checks the bound policies.
const MODE_POLICIES: &str = include_str!("../../../tools/refrun/mode-policies.json");
const WIDTH: [(&str, u64); 7] = [
    ("f64", 8),
    ("f32", 4),
    ("bf16", 2),
    ("f16", 2),
    ("i64", 8),
    ("i32", 4),
    ("u8", 1),
];

/// Why a reference cannot be used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReferenceError {
    /// The file is absent or unreadable.
    Missing(String),
    /// Its bytes, schema, provenance, variants or digests are not the pin's.
    Unverified(String),
    /// A bound fixture no longer matches what produced it; the first
    /// differing part and the detail.
    Stale(String, String),
}

impl std::fmt::Display for ReferenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing(message) | Self::Unverified(message) => formatter.write_str(message),
            Self::Stale(field, detail) => write!(formatter, "stale_reference: {field}: {detail}"),
        }
    }
}

impl std::error::Error for ReferenceError {}

fn unverified<T>(message: impl Into<String>) -> Result<T, ReferenceError> {
    Err(ReferenceError::Unverified(message.into()))
}

fn stale<T>(field: &str, detail: impl Into<String>) -> Result<T, ReferenceError> {
    Err(ReferenceError::Stale(field.into(), detail.into()))
}

/// The path of a declared input (`common.input_path`): a relative path
/// resolves under the root; an environment input under its variable's
/// directory, else its default. `None` for an unused input, an undeclared
/// one, or an environment input with neither.
pub fn input_path(
    loaded: &Loaded,
    name: &str,
    environment: &dyn Fn(&str) -> Option<String>,
) -> Option<PathBuf> {
    let input = if name == "receipts" {
        &loaded.declaration.receipts
    } else {
        loaded.declaration.input(name)?
    };
    declared_path(input, &loaded.root, environment)
}

/// A declared path (`common.declared_path`): a path relative to `root` (or
/// absolute), or one under an environment variable's directory, else its
/// default; `None` when unused or neither is set.
pub fn declared_path(
    input: &Input,
    root: &Path,
    environment: &dyn Fn(&str) -> Option<String>,
) -> Option<PathBuf> {
    let path = match input {
        Input::Unused => return None,
        Input::Path(path) => PathBuf::from(path),
        Input::Environment { env, default, path } => {
            PathBuf::from(environment(env).or_else(|| default.clone())?).join(path)
        }
    };
    Some(if path.is_absolute() {
        path
    } else {
        root.join(path)
    })
}

/// The reference pinned where the declared input `name` points
/// (`common.pinned_kind` of `input_path(name)`): how a greedy record finds
/// its pinned `exact` texts. `None` when no pin is stored there.
pub fn pinned_input<'a>(loaded: &'a Loaded, name: &str) -> Option<&'a ReferencePin> {
    let input = input_path(loaded, name, &process_environment)?
        .canonicalize()
        .ok()?;
    loaded.declaration.references.iter().find(|pin| {
        declared_path(&pin.path, &loaded.root, &process_environment)
            .and_then(|path| path.canonicalize().ok())
            .is_some_and(|path| path == input)
    })
}

/// An environment variable as Python's `os.environ.get` reads it.
pub fn process_environment(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

fn read(path: &Path) -> Result<(Vec<u8>, Value), ReferenceError> {
    json::read(path).map_err(ReferenceError::Missing)
}

/// The checkpoint lock the pin names, checked as `common.checkpoint_pin`
/// checks it: the registry's default entry is the pin, and the canonical
/// digest of the lock (the pin's entry of it, with a registry) is the
/// pinned `source_lock_sha256`. Returns that lock.
pub fn checkpoint_lock(loaded: &Loaded) -> Result<Value, ReferenceError> {
    let pin = &loaded.declaration.checkpoint;
    let (_, mut files) = read(&loaded.root.join(&pin.lock))?;
    if let Some((name, registry)) = &pin.registry {
        let (_, registry_value) = read(&loaded.root.join(registry))?;
        files = files.get(name).cloned().unwrap_or(Value::Null);
        let entry = &registry_value["checkpoints"][name];
        let differs = registry_value["default"].as_str() != Some(name.as_str())
            || ["repository", "revision", "served_model_name"]
                .iter()
                .any(|key| entry[*key].as_str() != pin.key(key));
        if differs {
            return unverified(format!("the gate's checkpoint pin differs from {registry}"));
        }
    }
    if canonical_sha256(&files) != pin.source_lock_sha256 {
        return unverified(format!(
            "the gate's checkpoint pin differs from {}",
            pin.lock
        ));
    }
    Ok(files)
}

/// A pinned reference's binding to the reference cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Binding {
    /// A fixture without a binding block (transitional).
    Unbound,
    Bound {
        refset_sha256: String,
        /// The fixture's scores against a local reference cache, when one
        /// was given.
        cache: Option<Cache>,
        /// The parts of the Python check not made here.
        unchecked: Vec<&'static str>,
    },
}

impl Binding {
    /// As receipts record it (`reference_binding`, as `cache.verify` gives
    /// it), with the checks not made listed as `unchecked` when there are
    /// any (an install).
    pub fn to_json(&self) -> Value {
        match self {
            Self::Unbound => json!({"status": "unbound", "refset_sha256": null}),
            Self::Bound {
                refset_sha256,
                cache,
                unchecked,
            } => {
                let mut value = json!({"status": "bound", "refset_sha256": refset_sha256,
                                       "cache": cache.as_ref().map(Cache::to_json)});
                if !unchecked.is_empty() {
                    value["unchecked"] = json!(unchecked);
                }
                value
            }
        }
    }
}

/// A bound fixture's scores compared with a reference cache (`cache.compare`):
/// every copied score equals its entry's; entries absent from the cache are
/// counted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cache {
    pub root: String,
    pub refset_present: bool,
    pub entries: usize,
    pub missing: usize,
}

impl Cache {
    pub fn to_json(&self) -> Value {
        json!({"root": self.root, "refset_present": self.refset_present,
               "entries": self.entries, "missing": self.missing})
    }
}

/// The reference cache `$CHOPINFER_DATA/refcache/v1` when that directory
/// exists (`cache.default_root`, `common.reference_binding`).
pub fn default_store(environment: &dyn Fn(&str) -> Option<String>) -> Option<PathBuf> {
    let data = environment("CHOPINFER_DATA").filter(|data| !data.is_empty())?;
    let root = PathBuf::from(data).join("refcache/v1");
    root.is_dir().then_some(root)
}

/// A pinned reference read and verified.
#[derive(Clone, Debug)]
pub struct Verified {
    pub name: String,
    pub path: PathBuf,
    pub sha256: String,
    /// The variants the pin names, when it names them.
    pub variants: Option<Vec<String>>,
    pub value: Value,
    pub binding: Binding,
}

fn reference_path(pin: &ReferencePin, root: &Path) -> Result<PathBuf, ReferenceError> {
    declared_path(&pin.path, root, &process_environment).ok_or_else(|| {
        let env = match &pin.path {
            Input::Environment { env, .. } => env.as_str(),
            _ => "",
        };
        ReferenceError::Missing(format!("the {} reference needs {env}", pin.name))
    })
}

/// Reads the pinned reference `pin` from under `references` (the root when
/// `None`) and verifies it (`common.pinned_reference`), with the scores
/// against the reference cache of [`default_store`] when there is one.
pub fn pinned(
    loaded: &Loaded,
    pin: &ReferencePin,
    references: Option<&Path>,
) -> Result<Verified, ReferenceError> {
    let path = reference_path(pin, references.unwrap_or(&loaded.root))?;
    let bytes = std::fs::read(&path)
        .map_err(|error| ReferenceError::Missing(format!("{}: {error}", path.display())))?;
    let checkpoint = &loaded.declaration.checkpoint;
    let store = default_store(&process_environment);
    verify_pinned(
        &loaded.root,
        checkpoint,
        pin,
        &path,
        bytes,
        &|| checkpoint_lock(loaded).map(|_| ()),
        store.as_deref(),
    )
}

/// Verifies the `bytes` read from `path` as the pinned reference `pin` of a
/// gate pinned to `checkpoint`, with lock files and workloads under `root`.
/// `lock` checks the checkpoint pin against its lock, in the order
/// `common.pinned_reference` does: after the digest, before the provenance.
/// A binding's scores are compared with the reference cache at `store`.
pub fn verify_pinned(
    root: &Path,
    checkpoint: &CheckpointPin,
    pin: &ReferencePin,
    path: &Path,
    bytes: Vec<u8>,
    lock: &dyn Fn() -> Result<(), ReferenceError>,
    store: Option<&Path>,
) -> Result<Verified, ReferenceError> {
    let path = path.to_path_buf();
    let sha256 = sha256_hex(&bytes);
    if sha256 != pin.sha256 {
        return unverified(format!(
            "{} is not the pinned {} reference",
            path.display(),
            pin.name
        ));
    }
    let value = json::parse(&bytes)
        .map_err(|error| ReferenceError::Unverified(format!("{}: {error}", path.display())))?;
    lock()?;
    let provenance = value.get("provenance").cloned().unwrap_or(Value::Null);
    let stated = match provenance.get("checkpoint") {
        Some(Value::String(revision)) => json!({"revision": revision}),
        Some(stated) if !stated.is_null() => stated.clone(),
        _ => {
            json!({"revision": provenance.get("checkpoint_revision").cloned().unwrap_or(Value::Null)})
        }
    };
    let keys: Vec<&str> = match &pin.checkpoint_keys {
        Some(keys) => keys.iter().map(String::as_str).collect(),
        None => CHECKPOINT_KEYS.to_vec(),
    };
    if value.get("schema").and_then(Value::as_str) != Some(pin.schema.as_str())
        || keys
            .iter()
            .any(|key| stated.get(*key).and_then(Value::as_str) != checkpoint.key(key))
    {
        return unverified(format!(
            "{}: schema or checkpoint provenance differs from the pin",
            path.display()
        ));
    }
    if let Some(variants) = &pin.variants {
        let declared: Vec<String> = match provenance.get("variants") {
            Some(Value::Object(map)) => map.keys().cloned().collect(),
            Some(Value::Array(items)) => items
                .iter()
                .map(|item| item.as_str().unwrap_or_default().to_owned())
                .collect(),
            _ => {
                let texts = texts(&value, &path)?;
                variants
                    .iter()
                    .filter(|variant| texts.iter().all(|text| text[variant.as_str()].is_object()))
                    .cloned()
                    .collect()
            }
        };
        if &declared != variants {
            return unverified(format!(
                "{}: variants differ from the pinned policy",
                path.display()
            ));
        }
    }
    if let Some(expected) = &pin.workload_sha256
        && workload_sha256(texts(&value, &path)?)? != *expected
    {
        return unverified(format!(
            "{}: workload differs from the pinned one",
            path.display()
        ));
    }
    if let Some(expected) = &pin.capture_sha256
        && value.get("capture_sha256").and_then(Value::as_str) != Some(expected.as_str())
    {
        return unverified(format!("{}: scores another capture", path.display()));
    }
    let binding = pinned_binding(root, checkpoint, pin, &value, store)?;
    Ok(Verified {
        name: pin.name.clone(),
        path,
        sha256,
        variants: pin.variants.clone(),
        value,
        binding,
    })
}

fn texts<'a>(value: &'a Value, path: &Path) -> Result<&'a Vec<Value>, ReferenceError> {
    value["texts"]
        .as_array()
        .ok_or_else(|| ReferenceError::Unverified(format!("{}: no texts", path.display())))
}

/// The digest `exact_reference` records over the scored workload: each
/// text's name, `score_from` and prompt token IDs (`common.workload_sha256`).
pub fn workload_sha256(texts: &[Value]) -> Result<String, ReferenceError> {
    let mut workload = Vec::new();
    for text in texts {
        let score_from = text
            .get("score_from")
            .cloned()
            .ok_or_else(|| ReferenceError::Unverified("a text has no score_from".into()))?;
        workload.push(json!([text["name"], score_from, text["prompt_token_ids"]]));
    }
    Ok(canonical_sha256(&Value::Array(workload)))
}

/// The reference cache binding of a pinned reference
/// (`common.pinned_binding` and `cache.verify`).
fn pinned_binding(
    root: &Path,
    checkpoint: &CheckpointPin,
    pin: &ReferencePin,
    value: &Value,
    store: Option<&Path>,
) -> Result<Binding, ReferenceError> {
    let Some(binding) = value.get("binding") else {
        return Ok(Binding::Unbound);
    };
    let lock = json!({"path": checkpoint.lock,
                      "variant": checkpoint.registry.as_ref().map(|(name, _)| name)});
    if binding.get("lock") != Some(&lock)
        || binding["digests"]["lock"].as_str() != Some(checkpoint.source_lock_sha256.as_str())
    {
        return stale(
            "lock",
            format!(
                "the {} reference is bound to another checkpoint lock than the pin",
                pin.name
            ),
        );
    }
    let workload = match &pin.workload {
        Some(declared) => match declared_path(declared, root, &process_environment) {
            Some(path) => path,
            None => {
                let env = match declared {
                    Input::Environment { env, .. } => env.as_str(),
                    _ => "",
                };
                return stale(
                    "workload",
                    format!("the bound {} reference's workload needs {env}", pin.name),
                );
            }
        },
        None => {
            let path = reference_path(pin, root)?;
            let stem = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or_default();
            path.with_file_name(format!("{stem}.workload.json"))
        }
    };
    if !workload.is_file() {
        return stale(
            "workload",
            format!(
                "the bound {} reference's workload {} is missing",
                pin.name,
                workload.display()
            ),
        );
    }
    verify_binding(value, binding, root, &workload, store)
}

fn field<'a>(value: &'a Value, key: &str) -> Result<&'a Value, ReferenceError> {
    value
        .get(key)
        .ok_or_else(|| ReferenceError::Stale("schema".into(), format!("the binding has no {key}")))
}

fn text_of<'a>(value: &'a Value, key: &str) -> Result<&'a str, ReferenceError> {
    field(value, key)?.as_str().ok_or_else(|| {
        ReferenceError::Stale(
            "schema".into(),
            format!("the binding's {key} is not a string"),
        )
    })
}

/// A bound mode: its name, dequant rule, the fixture variant holding its
/// scores, policy digest and entries (name, sequence index, key).
struct Mode<'a> {
    name: &'a str,
    dequant: &'a Value,
    variant: Option<&'a str>,
    policy_sha256: &'a str,
    entries: Vec<(&'a str, usize, &'a str)>,
}

fn modes(binding: &Value) -> Result<Vec<(&String, Mode<'_>)>, ReferenceError> {
    let shape =
        || ReferenceError::Stale("schema".into(), "the binding's modes are malformed".into());
    let mut result = Vec::new();
    for (label, mode) in field(binding, "modes")?.as_object().ok_or_else(shape)? {
        let mut entries = Vec::new();
        for entry in mode["entries"].as_array().ok_or_else(shape)? {
            let name = entry[0].as_str().ok_or_else(shape)?;
            let index = entry[1].as_u64().ok_or_else(shape)? as usize;
            let key = entry[2].as_str().ok_or_else(shape)?;
            entries.push((name, index, key));
        }
        result.push((
            label,
            Mode {
                name: mode["mode"].as_str().ok_or_else(shape)?,
                dequant: mode.get("dequant").ok_or_else(shape)?,
                variant: mode["variant"].as_str(),
                policy_sha256: mode["policy_sha256"].as_str().ok_or_else(shape)?,
                entries,
            },
        ));
    }
    Ok(result)
}

/// `cache.verify` without a checkpoint directory, with the cache `store`
/// when given.
fn verify_binding(
    value: &Value,
    binding: &Value,
    root: &Path,
    workload: &Path,
    store: Option<&Path>,
) -> Result<Binding, ReferenceError> {
    if binding.get("schema").and_then(Value::as_str) != Some(BINDING) {
        return stale("schema", format!("binding schema is not {BINDING}"));
    }
    let sections = field(binding, "key")?;
    let digests = field(binding, "digests")?;
    let model = text_of(binding, "model")?;
    let sequences = field(binding, "sequences")?
        .as_array()
        .ok_or_else(|| ReferenceError::Stale("schema".into(), "sequences is not a list".into()))?;
    let bound = modes(binding)?;

    // The binding agrees with itself.
    let mode_policy: Map<String, Value> = bound
        .iter()
        .map(|(label, mode)| ((*label).clone(), json!(mode.policy_sha256)))
        .collect();
    let stated = [
        ("lock", sections["checkpoint"]["source_lock_sha256"].clone()),
        (
            "reference_code",
            json!(canonical_sha256(&sections["reference_code"])),
        ),
        ("adapter", json!(canonical_sha256(&sections["adapter"]))),
        ("mode_policy", Value::Object(mode_policy)),
        (
            "environment",
            json!(canonical_sha256(&sections["environment"])),
        ),
    ];
    for (name, expected) in stated {
        if digests.get(name) != Some(&expected) {
            return stale(
                name,
                "the binding's sub-digest differs from its key section",
            );
        }
    }
    for (label, mode) in &bound {
        for (name, index, key) in &mode.entries {
            let sequence = sequences.get(*index).ok_or_else(|| {
                ReferenceError::Stale(
                    "key_sha256".into(),
                    format!("{label}/{name} has no sequence"),
                )
            })?;
            let document = json!({
                "schema": KEY, "model": model, "checkpoint": sections["checkpoint"],
                "sequence": sequence, "reference_code": sections["reference_code"],
                "adapter": sections["adapter"],
                "mode": {"name": mode.name, "policy_sha256": mode.policy_sha256},
                "environment": sections["environment"], "outputs": sections["outputs"],
            });
            if canonical_sha256(&document) != *key {
                return stale(
                    "key_sha256",
                    format!("{label}/{name} is not its key document's digest"),
                );
            }
        }
    }
    let pairs: Map<String, Value> = bound
        .iter()
        .map(|(label, mode)| {
            let entries = mode
                .entries
                .iter()
                .map(|(name, _, key)| json!([name, key]))
                .collect();
            ((*label).clone(), Value::Array(entries))
        })
        .collect();
    let refset = json!({"schema": REFSET, "model": model,
                        "workload_sha256": field(binding, "workload_sha256")?,
                        "modes": pairs,
                        "assembled_fixture_sha256": field(binding, "assembled_fixture_sha256")?});
    let refset_sha256 = text_of(binding, "refset_sha256")?;
    if canonical_sha256(&refset) != refset_sha256 {
        return stale(
            "refset_sha256",
            "the bound entries do not form the bound refset",
        );
    }

    // The working tree still produces the bound inputs.
    let lock_path = text_of(&binding["lock"], "path")?;
    let (_, mut lock) = read(&root.join(lock_path))?;
    if let Some(variant) = binding["lock"]["variant"].as_str() {
        lock = lock.get(variant).cloned().unwrap_or(Value::Null);
    }
    if Some(canonical_sha256(&lock).as_str()) != digests["lock"].as_str() {
        return stale(
            "lock",
            format!("{lock_path} differs from the bound checkpoint lock"),
        );
    }
    let mut unchecked = Vec::new();
    if root.join("tools/refrun").is_dir() {
        let files = sections["adapter"]["files"]
            .as_object()
            .cloned()
            .unwrap_or_default();
        let mut paths: Vec<&str> = files.keys().map(String::as_str).collect();
        paths.extend(NUMERICS);
        paths.sort_unstable();
        paths.dedup();
        for path in paths {
            let actual = std::fs::read(root.join(path))
                .ok()
                .map(|bytes| sha256_hex(&bytes));
            if files.get(path).and_then(Value::as_str) != actual.as_deref() {
                return stale(
                    "adapter",
                    format!("{path} differs from the bound adapter and numerics files"),
                );
            }
        }
    } else {
        unchecked.push("adapter");
    }
    for (label, mode) in &bound {
        if mode_policy_sha256(mode.name, mode.dequant)? != mode.policy_sha256 {
            return stale(
                "mode_policy",
                format!("modes.py no longer gives the bound {label} policy"),
            );
        }
    }
    check_workload(model, sequences, &bound, binding, workload)?;
    check_sequences(value, sequences, &bound)?;
    let cache = match store {
        Some(store) => Some(compare(value, sequences, &bound, refset_sha256, store)?),
        None => None,
    };
    let mut without = value.as_object().cloned().unwrap_or_default();
    without.remove("binding");
    let assembled = canonical_sha256(&Value::Object(without));
    if Some(assembled.as_str()) != binding["assembled_fixture_sha256"].as_str() {
        return stale(
            "assembled_fixture",
            format!(
                "the fixture's content has sha256 {assembled}, not the bound {}",
                binding["assembled_fixture_sha256"]
            ),
        );
    }
    Ok(Binding::Bound {
        refset_sha256: refset_sha256.into(),
        cache,
        unchecked,
    })
}

/// The policy digest `modes.policy(name, dequant)` gives, refused with its
/// message for a mode or rule it does not know.
pub(crate) fn mode_policy_sha256(name: &str, dequant: &Value) -> Result<String, ReferenceError> {
    let policies: Value = serde_json::from_str(MODE_POLICIES)
        .map_err(|error| ReferenceError::Unverified(format!("mode-policies.json: {error}")))?;
    let Some(mode) = policies.get(name) else {
        let mut names: Vec<&String> = policies
            .as_object()
            .map(|modes| modes.keys().collect())
            .unwrap_or_default();
        names.sort();
        return unverified(format!(
            "unknown mode {name}; expected one of {}",
            repr::repr(&json!(names))
        ));
    };
    let rule = if dequant.is_null() {
        &mode["dequant"]
    } else {
        dequant
    };
    let digest = rule
        .as_str()
        .and_then(|rule| mode["policy_sha256"][rule].as_str());
    digest.map(str::to_owned).ok_or_else(|| {
        ReferenceError::Unverified(format!(
            "mode {name} cannot dequantize by {}",
            repr::str(dequant)
        ))
    })
}

fn ids_sha256(ids: &Value) -> String {
    canonical_sha256(ids)
}

/// A workload file's model and its ordered (name, sequence section) pairs
/// (`cache.workload_sequences`, `workload.load`, `keys.sequence`).
fn workload_sequences(path: &Path) -> Result<(String, Vec<(String, Value)>), ReferenceError> {
    let invalid = |message: String| {
        ReferenceError::Stale("workload".into(), format!("{}: {message}", path.display()))
    };
    let (_, document) = read(path)?;
    let map = document
        .as_object()
        .ok_or_else(|| invalid("not an object".into()))?;
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    if map.get("schema").and_then(Value::as_str) != Some(WORKLOAD)
        || keys != ["model", "schema", "sequences"]
    {
        return Err(invalid(format!(
            "workload must be {WORKLOAD} with model and sequences"
        )));
    }
    let model = map["model"].as_str().unwrap_or_default().to_owned();
    let root = path.parent().unwrap_or(Path::new("."));
    let mut result: Vec<(String, Value)> = Vec::new();
    for spec in map["sequences"]
        .as_array()
        .ok_or_else(|| invalid("sequences is not a list".into()))?
    {
        let (name, section) = sequence(spec, root).map_err(invalid)?;
        if result.iter().any(|(seen, _)| *seen == name) {
            return Err(invalid("workload needs unique names".into()));
        }
        result.push((name, section));
    }
    if result.is_empty() {
        return Err(invalid("workload needs at least one sequence".into()));
    }
    Ok((model, result))
}

fn token_ids(value: &Value, label: &str) -> Result<Vec<u64>, String> {
    value
        .as_array()
        .and_then(|items| {
            items
                .iter()
                .map(|item| item.as_u64().filter(|_| !item.is_f64()))
                .collect()
        })
        .ok_or_else(|| format!("{label} must be a list of non-negative token IDs"))
}

fn exact_keys(map: &Map<String, Value>, keys: &[&str]) -> bool {
    map.len() == keys.len() && keys.iter().all(|key| map.contains_key(*key))
}

fn tensor(spec: &Value, root: &Path, label: &str) -> Result<Value, String> {
    let map = spec
        .as_object()
        .filter(|map| exact_keys(map, &["path", "dtype", "shape"]));
    let width = map.and_then(|map| {
        WIDTH
            .iter()
            .find(|(dtype, _)| map["dtype"].as_str() == Some(*dtype))
            .map(|(_, width)| *width)
    });
    let (Some(map), Some(width)) = (map, width) else {
        return Err(format!("{label}: tensor needs path, dtype and shape"));
    };
    let shape: Vec<u64> = map["shape"]
        .as_array()
        .and_then(|dims| dims.iter().map(|d| d.as_u64().filter(|d| *d > 0)).collect())
        .ok_or_else(|| format!("{label}: invalid shape"))?;
    let relative = Path::new(map["path"].as_str().unwrap_or_default());
    if relative.is_absolute() || relative.components().any(|part| part.as_os_str() == "..") {
        return Err(format!(
            "{label}: tensor path must be relative to the workload"
        ));
    }
    let data = std::fs::read(root.join(relative)).map_err(|error| format!("{label}: {error}"))?;
    if data.len() as u64 != shape.iter().product::<u64>() * width {
        return Err(format!(
            "{label}: {} bytes do not hold {} {shape:?}",
            data.len(),
            map["dtype"]
        ));
    }
    Ok(json!({"sha256": sha256_hex(&data), "dtype": map["dtype"], "shape": shape}))
}

fn sequence(spec: &Value, root: &Path) -> Result<(String, Value), String> {
    let allowed = [
        "name",
        "category",
        "token_ids",
        "score_from",
        "answer",
        "media",
    ];
    let map = spec
        .as_object()
        .filter(|map| {
            ["name", "category", "token_ids", "score_from"]
                .iter()
                .all(|key| map.contains_key(*key))
                && map.keys().all(|key| allowed.contains(&key.as_str()))
        })
        .ok_or_else(|| format!("sequence fields must be within {allowed:?}"))?;
    let name = match map["name"].as_str() {
        Some(name) if !name.is_empty() && map["category"].is_string() => name.to_owned(),
        _ => return Err("sequence name and category must be strings".into()),
    };
    let prompt = token_ids(&map["token_ids"], &name)?;
    let forced = json!({"policy": "forced", "ids": []});
    let answer = map.get("answer").unwrap_or(&forced);
    let answer_map = answer.as_object().cloned().unwrap_or_default();
    let (section, answer_length) = match answer_map.get("policy").and_then(Value::as_str) {
        Some("forced") if exact_keys(&answer_map, &["policy", "ids"]) => {
            let ids = token_ids(&answer_map["ids"], &format!("{name} answer"))?;
            (
                json!({"policy": "forced", "ids_sha256": ids_sha256(&json!(ids)),
                       "max_new_tokens": null, "stop_ids": null}),
                ids.len(),
            )
        }
        Some("greedy") if exact_keys(&answer_map, &["policy", "max_new_tokens", "stop_ids"]) => {
            let count = answer_map["max_new_tokens"]
                .as_u64()
                .filter(|count| *count > 0 && !answer_map["max_new_tokens"].is_f64())
                .ok_or_else(|| format!("{name}: max_new_tokens must be positive"))?;
            let mut stop = token_ids(&answer_map["stop_ids"], &format!("{name} stop_ids"))?;
            stop.sort_unstable();
            (
                json!({"policy": "greedy", "ids_sha256": null, "max_new_tokens": count,
                       "stop_ids": stop}),
                1,
            )
        }
        _ => {
            return Err(format!(
                "{name}: answer must be forced {{ids}} or greedy {{max_new_tokens, stop_ids}}"
            ));
        }
    };
    let end = prompt.len() + answer_length;
    let score_from = map["score_from"]
        .as_u64()
        .filter(|from| !map["score_from"].is_f64() && *from >= 1 && (*from as usize) < end)
        .ok_or_else(|| format!("{name}: score_from must leave at least one scored position"))?;
    let mut media = Vec::new();
    for (index, item) in map
        .get("media")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .iter()
        .enumerate()
    {
        let label = format!("{name} media {index}");
        let item = item
            .as_object()
            .filter(|item| {
                ["kind", "tensor", "processor"]
                    .iter()
                    .all(|key| item.contains_key(*key))
                    && item.keys().all(|key| {
                        ["kind", "tensor", "grid", "processor", "raw_sha256"]
                            .contains(&key.as_str())
                    })
            })
            .ok_or_else(|| format!("{label}: media needs kind, tensor and processor"))?;
        let processor = item["processor"]
            .as_object()
            .filter(|processor| exact_keys(processor, &["code_sha256", "config_sha256"]))
            .ok_or_else(|| {
                format!("{label}: processor identity needs code_sha256 and config_sha256")
            })?;
        let grid = match item.get("grid") {
            Some(grid) => tensor(grid, root, &format!("{label} grid"))?,
            None => Value::Null,
        };
        media.push(
            json!({"kind": item["kind"], "tensor": tensor(&item["tensor"], root, &label)?,
                          "grid": grid,
                          "processor": {"code_sha256": processor["code_sha256"],
                                        "config_sha256": processor["config_sha256"]}}),
        );
    }
    Ok((
        name,
        json!({"token_ids_sha256": ids_sha256(&json!(prompt)), "score_from": score_from,
               "answer": section, "media": media}),
    ))
}

/// The binding scores exactly the workload at `path` (`cache.check_workload`).
fn check_workload(
    model: &str,
    sequences: &[Value],
    bound: &[(&String, Mode<'_>)],
    binding: &Value,
    path: &Path,
) -> Result<(), ReferenceError> {
    let (workload_model, workload) = workload_sequences(path)?;
    let listed: Vec<Value> = workload
        .iter()
        .map(|(name, section)| json!([name, section]))
        .collect();
    let digest = canonical_sha256(&json!({"model": workload_model, "sequences": listed}));
    if Some(digest.as_str()) != binding["workload_sha256"].as_str() || workload_model != model {
        return stale(
            "workload",
            format!("{} is not the bound workload", path.display()),
        );
    }
    let policy = |section: &Value| section["answer"]["policy"].as_str().map(str::to_owned);
    let greedy: Vec<(&String, &Value)> = workload
        .iter()
        .filter(|(_, section)| policy(section).as_deref() == Some("greedy"))
        .map(|(name, section)| (name, section))
        .collect();
    for (label, mode) in bound {
        let listed: Vec<(&str, &Value)> = mode
            .entries
            .iter()
            .map(|(name, index, _)| (*name, &sequences[*index]))
            .collect();
        let generated: Vec<_> = listed
            .iter()
            .filter(|(_, section)| policy(section).as_deref() == Some("greedy"))
            .collect();
        let forced: Vec<_> = listed
            .iter()
            .filter(|(_, section)| policy(section).as_deref() == Some("forced"))
            .collect();
        let generated_ok =
            generated.is_empty()
                || (generated.len() == greedy.len()
                    && generated.iter().zip(&greedy).all(
                        |((name, section), (expected, other))| name == expected && section == other,
                    ));
        if !generated_ok
            || generated.len() + forced.len() != listed.len()
            || forced.len() != workload.len()
            || listed[..generated.len()]
                .iter()
                .any(|(_, s)| policy(s).as_deref() != Some("greedy"))
        {
            return stale(
                "workload",
                format!(
                    "mode {label} does not score the sequences of {} in order",
                    path.display()
                ),
            );
        }
        for ((name, bound_section), (expected, section)) in forced.iter().zip(&workload) {
            let mut bound_section = (*bound_section).clone();
            if policy(section).as_deref() == Some("greedy") {
                bound_section["answer"] = section["answer"].clone();
            }
            if name != expected || &bound_section != section {
                return stale(
                    "workload",
                    format!(
                        "mode {label}: {name} is not {expected} of {}",
                        path.display()
                    ),
                );
            }
        }
    }
    let leads_greedy = |mode: &Mode<'_>| {
        mode.entries
            .first()
            .is_some_and(|(_, index, _)| policy(&sequences[*index]).as_deref() == Some("greedy"))
    };
    if !greedy.is_empty() && !bound.iter().any(|(_, mode)| leads_greedy(mode)) {
        return stale(
            "workload",
            format!(
                "no mode generates the answers {} leaves to the model",
                path.display()
            ),
        );
    }
    Ok(())
}

/// A fixture's scored items by name: `texts` by name, `cases` by id or the
/// `probes` mapping (`cache.items`).
fn items(value: &Value) -> BTreeMap<String, &Value> {
    if let Some(texts) = value["texts"].as_array() {
        texts
            .iter()
            .map(|item| (item["name"].as_str().unwrap_or_default().to_owned(), item))
            .collect()
    } else if let Some(cases) = value["cases"].as_array() {
        cases
            .iter()
            .map(|item| (item["id"].as_str().unwrap_or_default().to_owned(), item))
            .collect()
    } else if let Some(probes) = value["probes"].as_object() {
        probes
            .iter()
            .map(|(name, item)| (name.clone(), item))
            .collect()
    } else {
        BTreeMap::new()
    }
}

/// Every bound sequence is a fixture item stating the bound token IDs,
/// score_from and forced answer, each where the item states it
/// (`cache.check_sequences`).
fn check_sequences(
    value: &Value,
    sequences: &[Value],
    bound: &[(&String, Mode<'_>)],
) -> Result<(), ReferenceError> {
    let stated = items(value);
    for (_, mode) in bound {
        for (name, index, _) in &mode.entries {
            let sequence = &sequences[*index];
            let Some(item) = stated.get(*name) else {
                return stale(
                    "sequence",
                    format!("{name} is bound but not in the fixture"),
                );
            };
            let answer = &sequence["answer"];
            let greedy = answer["policy"].as_str() == Some("greedy");
            if greedy && item.get("answer_token_ids").is_none() {
                return stale(
                    "sequence",
                    format!("{name}: a generated answer is bound but the fixture states none"),
                );
            }
            if let Some(ids) = item.get("prompt_token_ids")
                && Some(ids_sha256(ids).as_str()) != sequence["token_ids_sha256"].as_str()
            {
                return stale(
                    "sequence",
                    format!("{name}: token IDs differ from the bound sequence"),
                );
            }
            if let Some(from) = item.get("score_from")
                && from != &sequence["score_from"]
            {
                return stale(
                    "sequence",
                    format!("{name}: score_from differs from the bound sequence"),
                );
            }
            if let Some(ids) = item.get("answer_token_ids")
                && !greedy
                && Some(ids_sha256(ids).as_str()) != answer["ids_sha256"].as_str()
            {
                return stale(
                    "sequence",
                    format!("{name}: answer token IDs differ from the bound sequence"),
                );
            }
        }
    }
    Ok(())
}

/// The fixture's scores against the cache entries under `store`
/// (`cache.compare`): each scores field the fixture copies must equal the
/// entry's. Entries absent from this cache are counted, not refused.
fn compare(
    value: &Value,
    sequences: &[Value],
    bound: &[(&String, Mode<'_>)],
    refset_sha256: &str,
    store: &Path,
) -> Result<Cache, ReferenceError> {
    // As Python's `Path` spells it.
    let store: PathBuf = store.components().collect();
    let mut found: BTreeMap<&str, Value> = BTreeMap::new();
    for (_, mode) in bound {
        for (_, _, key) in &mode.entries {
            if !found.contains_key(key) {
                found.insert(*key, cached_scores(&store, key)?);
            }
        }
    }
    let by_name = items(value);
    for (label, mode) in bound {
        for (name, index, key) in &mode.entries {
            let actual = &found[key];
            if actual.is_null() {
                continue;
            }
            let item = by_name[*name];
            let differs = |field: &str| {
                stale(
                    "scores",
                    format!("{label}/{name}: {field} differs from cache entry {key}"),
                )
            };
            let fixture = if sequences[*index]["answer"]["policy"] == "greedy" {
                let answer = item.get("answer_token_ids").unwrap_or(&Value::Null);
                if !json::python_eq(answer, &actual["answer"]) {
                    return differs("answer");
                }
                item.get("greedy")
            } else {
                mode.variant.and_then(|variant| item.get(variant))
            };
            if let Some(Value::Object(fixture)) = fixture
                && let Some(field) = score_difference(fixture, actual)
            {
                return differs(field);
            }
        }
    }
    Ok(Cache {
        root: store.display().to_string(),
        refset_present: refset_present(&store, refset_sha256)?,
        entries: found.len(),
        missing: found.values().filter(|scores| scores.is_null()).count(),
    })
}

/// The first field the fixture copies from scores.json with another value
/// (`cache.score_difference`). Top-k rows may be cut to a prefix.
fn score_difference<'a>(fixture: &'a Map<String, Value>, scores: &Value) -> Option<&'a str> {
    for (field, stated) in fixture {
        let Some(actual) = scores.get(field) else {
            continue;
        };
        let cut: Value;
        let actual = match (stated, actual) {
            (Value::Array(mine), Value::Array(rows))
                if TOPK.contains(&field.as_str()) && mine.len() == rows.len() =>
            {
                cut = rows
                    .iter()
                    .zip(mine)
                    .map(|(row, mine)| match (row, mine) {
                        (Value::Array(row), Value::Array(mine)) => {
                            Value::Array(row[..row.len().min(mine.len())].to_vec())
                        }
                        _ => row.clone(),
                    })
                    .collect();
                &cut
            }
            _ => actual,
        };
        if json::canonical(stated) != json::canonical(actual) {
            return Some(field);
        }
    }
    None
}

/// The scores of the sealed cache entry `key` under `store`, `Null` when
/// absent (`cache.Store.get`). An entry whose files differ from its seal is
/// refused (`Store.read`).
fn cached_scores(store: &Path, key: &str) -> Result<Value, ReferenceError> {
    let path = store.join("entries").join(&key[..2]).join(key);
    if !path.exists() {
        return Ok(Value::Null);
    }
    let (_, entry) = read(&path.join("entry.json"))?;
    let mut data = Vec::new();
    for name in FILES {
        data.push(read_file(&path.join(name))?);
    }
    let seal: Map<String, Value> = FILES
        .iter()
        .zip(&data)
        .map(|(name, raw)| {
            let seal = json!({"bytes": raw.len(), "sha256": sha256_hex(raw)});
            ((*name).to_owned(), seal)
        })
        .collect();
    if entry.get("schema").and_then(Value::as_str) != Some(ENTRY)
        || entry.get("key_sha256").and_then(Value::as_str) != Some(key)
        || !entry
            .get("files")
            .is_some_and(|files| json::python_eq(files, &Value::Object(seal)))
        || sha256_hex(&data[0]) != key
    {
        return unverified(format!(
            "cache entry {} differs from its seal",
            path.display()
        ));
    }
    parse(&data[1], &path.join(FILES[1]))
}

/// Whether `store` holds the refset `refset_sha256`, refused when its bytes
/// are not that digest (`cache.Store.refset`).
fn refset_present(store: &Path, refset_sha256: &str) -> Result<bool, ReferenceError> {
    let path = store.join("refsets").join(format!("{refset_sha256}.json"));
    if !path.exists() {
        return Ok(false);
    }
    let bytes = read_file(&path)?;
    if sha256_hex(&bytes) != refset_sha256 {
        return unverified(format!("refset {} differs from its digest", path.display()));
    }
    parse(&bytes, &path).map(|_| true)
}

fn read_file(path: &Path) -> Result<Vec<u8>, ReferenceError> {
    std::fs::read(path)
        .map_err(|error| ReferenceError::Missing(format!("{}: {error}", path.display())))
}

fn parse(bytes: &[u8], path: &Path) -> Result<Value, ReferenceError> {
    json::parse(bytes)
        .map_err(|error| ReferenceError::Unverified(format!("{}: {error}", path.display())))
}
