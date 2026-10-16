//! CPU-side evidence policy for the opt-in fixed-input recorder.
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const FORMAT: &str = "chopinfer.diagnostic.v1";
pub const MAX_METADATA_BYTES: u64 = 256 << 20;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct Limits {
    pub logits_bytes: u64,
    pub step_bytes: u64,
    pub metadata_bytes: u64,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            logits_bytes: 4 << 30,
            step_bytes: 256 << 20,
            metadata_bytes: 16 << 20,
        }
    }
}
impl Limits {
    pub fn check(&self, step: u64, total: u64) -> Result<()> {
        if step > self.step_bytes || total > self.logits_bytes {
            return Err(
                "capture exceeds the declared byte budget; retain the original schedule".into(),
            );
        }
        Ok(())
    }
    /// Refuses a replay whose frame metadata bound exceeds the declared
    /// budget before any step runs, instead of at the step that crosses it.
    pub fn check_metadata(&self, bound: u64) -> Result<()> {
        if bound > self.metadata_bytes {
            return Err(format!(
                "frame metadata bound {bound} bytes exceeds the declared budget of {} bytes; \
                 declare --metadata-bytes of at least {bound}",
                self.metadata_bytes
            )
            .into());
        }
        Ok(())
    }
    pub fn estimate(&self, rows: impl Iterator<Item = usize>, width: u32) -> Result<u64> {
        let mut total = 0_u64;
        for count in rows {
            // F32 is the largest supported logits element; BF16 needs less.
            let step = row_bytes(1, width)?
                .checked_mul(count as u64)
                .ok_or("capture size overflow")?;
            total = total.checked_add(step).ok_or("capture size overflow")?;
            self.check(step, total)?;
        }
        Ok(total)
    }
}
pub fn row_bytes(element_type: u32, width: u32) -> Result<u64> {
    let size = match element_type {
        1 => 4,
        2 => 2,
        _ => return Err("unsupported logits dtype".into()),
    };
    if width == 0 {
        return Err("empty logits row".into());
    }
    Ok(u64::from(width) * size)
}

pub struct Budget {
    pub limits: Limits,
    total: u64,
    step: u64,
}
impl Budget {
    pub fn new(limits: Limits) -> Self {
        Self {
            limits,
            total: 0,
            step: 0,
        }
    }
    pub fn reserve(&mut self, dtype: u32, width: u32, rows: usize) -> Result<()> {
        let bytes = row_bytes(dtype, width)?
            .checked_mul(rows as u64)
            .ok_or("capture size overflow")?;
        let step = self
            .step
            .checked_add(bytes)
            .ok_or("capture size overflow")?;
        let total = self
            .total
            .checked_add(bytes)
            .ok_or("capture size overflow")?;
        self.limits.check(step, total)?;
        self.step = step;
        self.total = total;
        Ok(())
    }
    pub fn next_step(&mut self) {
        self.step = 0;
    }
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Options {
    pub environment: Value,
    pub declared_differences: Vec<Change>,
    pub reproduction: Vec<String>,
    #[serde(default)]
    pub limits: Limits,
}
impl Options {
    pub fn validate_environment(&self, vars: impl Iterator<Item = (String, String)>) -> Result<()> {
        let actual = vars
            .filter(|(key, _)| {
                key.starts_with("CHOPINFER_")
                    && !matches!(
                        key.as_str(),
                        "CHOPINFER_BATCH_REPLAY_CONFIG" | "CHOPINFER_KERNEL_GPUS"
                    )
            })
            .map(|(key, value)| (key, Value::String(value)))
            .collect::<serde_json::Map<_, _>>();
        if Value::Object(actual) != self.environment {
            return Err("effective environment differs from declared arm".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Change {
    pub field: String,
    pub expected: Value,
    pub observed: Value,
}

/// Component IDs are neutral metadata. Arrays elsewhere name whole manifest fields.
pub fn manifest_identity(manifest: &Value) -> Value {
    let mut value = manifest.clone();
    if let Some(components) = manifest["components"].as_array() {
        value["components"] = Value::Object(
            components
                .iter()
                .map(|c| (c["id"].as_str().unwrap_or_default().to_owned(), c.clone()))
                .collect(),
        );
    }
    value
}
pub fn changes(prefix: &str, a: &Value, b: &Value) -> Vec<Change> {
    if a == b {
        return vec![];
    }
    if let (Some(a), Some(b)) = (a.as_object(), b.as_object()) {
        let keys = a.keys().chain(b.keys()).collect::<BTreeSet<_>>();
        return keys
            .into_iter()
            .flat_map(|key| {
                // Presence is explicit so adding a null field is still a change.
                if !a.contains_key(key) || !b.contains_key(key) {
                    vec![Change {
                        field: format!("{prefix}/{key}"),
                        expected: json!({"present":a.contains_key(key),"value":a.get(key)}),
                        observed: json!({"present":b.contains_key(key),"value":b.get(key)}),
                    }]
                } else {
                    changes(&format!("{prefix}/{key}"), &a[key], &b[key])
                }
            })
            .collect();
    }
    vec![Change {
        field: prefix.to_owned(),
        expected: a.clone(),
        observed: b.clone(),
    }]
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Description {
    pub invariants: Value,
    pub manifest: Value,
    pub environment: Value,
    pub source_checkpoint: Option<Value>,
    pub declared_differences: Vec<Change>,
    pub reproduction: Vec<String>,
    pub expected_steps: usize,
    pub expected_rows: usize,
    pub limits: Limits,
    /// Configured graph mode, unknown observations and model-declared labels.
    pub context: Value,
}
impl Description {
    pub fn validate(&self) -> Result<()> {
        for key in [
            "host",
            "trace",
            "vocabulary",
            "topology",
            "ranks",
            "fold",
            "capacity",
            "sampling",
        ] {
            if self.invariants.get(key).is_none_or(Value::is_null) {
                return Err(format!("missing invariant {key}").into());
            }
        }
        if self.expected_steps == 0
            || self.expected_rows == 0
            || self.reproduction.is_empty()
            || !self.manifest.is_object()
            || !self.environment.is_object()
            || self.limits.metadata_bytes == 0
            || self.limits.metadata_bytes > MAX_METADATA_BYTES
            || self.limits.step_bytes == 0
            || self.limits.logits_bytes == 0
        {
            return Err("incomplete diagnostic description".into());
        }
        Ok(())
    }
}
pub fn compatible(a: &Description, b: &Description) -> Result<()> {
    a.validate()?;
    b.validate()?;
    if let Some(delta) = changes("invariants", &a.invariants, &b.invariants).first() {
        return Err(format!("incompatible {}", delta.field).into());
    }
    if a.expected_steps != b.expected_steps || a.expected_rows != b.expected_rows {
        return Err("incompatible expected coverage".into());
    }
    if a.source_checkpoint.is_some()
        && b.source_checkpoint.is_some()
        && a.source_checkpoint != b.source_checkpoint
    {
        return Err("incompatible source_checkpoint".into());
    }
    let actual = changes("manifest", &a.manifest, &b.manifest)
        .into_iter()
        .chain(changes("environment", &a.environment, &b.environment))
        .collect::<Vec<_>>();
    if actual.len() != b.declared_differences.len()
        || actual.iter().any(|d| !b.declared_differences.contains(d))
    {
        return Err(format!(
            "undeclared or stale differences: {}",
            serde_json::to_string(&actual)?
        )
        .into());
    }
    Ok(())
}

pub fn external(path: &Path) -> Result<()> {
    if !path.is_absolute() || path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err("diagnostic paths must be absolute and external".into());
    }
    for parent in path.ancestors() {
        if parent.join(".git").exists()
            || parent
                .symlink_metadata()
                .is_ok_and(|m| m.file_type().is_symlink())
        {
            return Err("diagnostic paths cannot enter a repository or traverse symlinks".into());
        }
    }
    Ok(())
}
pub fn digest(path: &Path) -> Result<String> {
    external(path)?;
    Ok(runtime_weights::sha256_reader(&mut File::open(path)?)?)
}
pub fn read_json(path: &Path, limit: u64) -> Result<Value> {
    external(path)?;
    let mut bytes = Vec::new();
    File::open(path)?
        .take(limit.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err("diagnostic metadata budget exceeded".into());
    }
    Ok(serde_json::from_slice(&bytes)?)
}
pub fn write_json(path: &Path, value: &Value, limit: u64) -> Result<()> {
    external(path)?;
    let data = serde_json::to_vec(value)?;
    if data.len() as u64 > limit {
        return Err("diagnostic metadata budget exceeded".into());
    }
    let mut file = super::record::create(path)?;
    file.write_all(&data)?;
    file.sync_all()?;
    Ok(())
}
pub fn binding(root: &Path) -> Result<Value> {
    Ok(json!({"path":root,"sha256":digest(&root.join("diagnostic.json"))?}))
}

/// Validate the exact retained files before reusing an online comparison receipt.
pub fn read_terminal(root: &Path) -> Result<Value> {
    let terminal = read_json(&root.join("diagnostic.json"), MAX_METADATA_BYTES)?;
    if terminal["format"] != FORMAT || terminal["mode"] != "fixed-schedule" {
        return Err("missing or unsupported diagnostic format".into());
    }
    let desc: Description = serde_json::from_value(terminal["description"].clone())?;
    desc.validate()?;
    if !matches!(terminal["status"].as_str(), Some("baseline" | "match"))
        || terminal["completion"] != "full"
        || terminal["completed_steps"] != desc.expected_steps
        || terminal["rows"] != desc.expected_rows
    {
        return Err("reference has no complete matching evidence".into());
    }
    let artifacts = terminal["artifacts"]
        .as_object()
        .ok_or("missing artifacts")?;
    for name in ["frames.jsonl", "receipt.json"] {
        if !artifacts.contains_key(name) {
            return Err(format!("missing artifact {name}").into());
        }
    }
    if terminal["status"] == "baseline" && !artifacts.contains_key("rows.bin") {
        return Err("missing reference logits".into());
    }
    let mut metadata_bytes = root.join("diagnostic.json").metadata()?.len();
    for (name, entry) in artifacts {
        if !matches!(
            name.as_str(),
            "frames.jsonl" | "rows.bin" | "receipt.json" | "mismatch.json" | "mismatch.bin"
        ) {
            return Err("unknown artifact path".into());
        }
        let path = root.join(name);
        external(&path)?;
        let size = path.metadata()?.len();
        let bound = if name.ends_with(".bin") {
            desc.limits.logits_bytes
        } else {
            metadata_bytes = metadata_bytes
                .checked_add(size)
                .ok_or("metadata size overflow")?;
            desc.limits.metadata_bytes
        };
        if size > bound || entry["bytes"] != size || entry["sha256"] != digest(&path)? {
            return Err(format!("stale or over-budget artifact {name}").into());
        }
    }
    if metadata_bytes > desc.limits.metadata_bytes {
        return Err("metadata budget exceeded".into());
    }
    if read_json(&root.join("receipt.json"), desc.limits.metadata_bytes)?["passed"] != true {
        return Err("reference receipt did not pass".into());
    }
    Ok(terminal)
}

pub struct Publication {
    pub description: Description,
    pub reference: Option<Value>,
    pub control: Option<Value>,
}
impl Publication {
    pub fn publish(
        &self,
        root: &Path,
        status: &str,
        counts: (usize, usize, u64),
        first: Option<&Value>,
        reason: &str,
    ) -> Result<()> {
        let (steps, rows, bytes) = counts;
        let mut artifacts = BTreeMap::new();
        let mut metadata_bytes = 0_u64;
        for name in [
            "frames.jsonl",
            "rows.bin",
            "receipt.json",
            "mismatch.json",
            "mismatch.bin",
        ] {
            let path = root.join(name);
            if path.exists() {
                let size = path.metadata()?.len();
                let bound = if name.ends_with(".bin") {
                    self.description.limits.logits_bytes
                } else {
                    metadata_bytes = metadata_bytes
                        .checked_add(size)
                        .ok_or("metadata size overflow")?;
                    self.description.limits.metadata_bytes
                };
                if size > bound {
                    return Err(format!("over-budget artifact {name}").into());
                }
                artifacts.insert(name, json!({"bytes":size,"sha256":digest(&path)?}));
            }
        }
        let completion = match status {
            "baseline" | "match" => "full",
            "diverged" => "first-divergence",
            _ => "incomplete",
        };
        write_json(
            &root.join("diagnostic.json"),
            &json!({"format":FORMAT,"mode":"fixed-schedule",
            "contract":"bitwise","description":self.description,"status":status,"completion":completion,
            "completed_steps":steps,"rows":rows,"bytes_checked":bytes,"first":first,"reason":reason,
            "reference":self.reference,"control":self.control,"artifacts":artifacts}),
            self.description
                .limits
                .metadata_bytes
                .saturating_sub(metadata_bytes),
        )
    }
}
