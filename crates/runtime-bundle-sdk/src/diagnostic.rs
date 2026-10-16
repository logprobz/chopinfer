//! Offline producer metadata. Model code supplies scope, labels and contracts.
//! A completion inventory is not a content seal; an offline reader seals it.
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

pub const SCHEMA: &str = "chopinfer.oracle-attachment.v1";
pub const MAX_ENTRIES: usize = 65536;
pub const MAX_BYTES: u64 = 8 << 30;

pub fn width(dtype: &str) -> Result<u64, String> {
    match dtype {
        "BF16" => Ok(2),
        "F32" | "U32" => Ok(4),
        "F64" | "I64" => Ok(8),
        "U8" | "BOOL" => Ok(1),
        _ => Err(format!("unsupported tensor dtype {dtype}")),
    }
}

pub fn elements(shape: &Value) -> Result<u64, String> {
    let dims = shape.as_array().ok_or("tensor shape must be an array")?;
    if dims.is_empty() || dims.len() > 8 {
        return Err("tensor shape needs 1 to 8 dimensions".into());
    }
    dims.iter().try_fold(1_u64, |n, d| {
        let d = d
            .as_u64()
            .filter(|&v| v > 0)
            .ok_or("invalid tensor dimension")?;
        n.checked_mul(d)
            .filter(|&v| v <= MAX_BYTES)
            .ok_or("tensor shape exceeds budget".into())
    })
}

pub fn validate_scope(scope: &Value) -> Result<(), String> {
    for key in ["model", "evidence", "contract"] {
        if scope[key].as_str().is_none_or(str::is_empty) {
            return Err(format!("missing scope {key}"));
        }
    }
    if !matches!(scope["evidence"].as_str(), Some("chained" | "op-isolated")) {
        return Err("unsupported evidence mode".into());
    }
    let probes = scope["probes"]
        .as_array()
        .ok_or("missing expected probes")?;
    if probes.is_empty() || probes.len() > MAX_ENTRIES {
        return Err("empty or oversized probe scope".into());
    }
    let mut names = BTreeSet::new();
    for probe in probes {
        let name = probe["name"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or("missing probe name")?;
        if !names.insert(name) {
            return Err(format!("duplicate required probe {name}"));
        }
        elements(&probe["shape"])?;
    }
    let rows = scope["inputs"]["rows"]
        .as_array()
        .ok_or("missing exact input rows")?;
    if rows.is_empty() || rows.len() > MAX_ENTRIES || !scope["inputs"]["state"].is_object() {
        return Err("missing input rows/state".into());
    }
    Ok(())
}

pub fn validate_tensor(tensor: &Value) -> Result<(), String> {
    let bytes = elements(&tensor["shape"])?
        .checked_mul(width(tensor["dtype"].as_str().ok_or("missing dtype")?)?)
        .filter(|&n| n <= MAX_BYTES)
        .ok_or("tensor bytes exceed budget")?;
    if tensor["bytes"].as_u64() != Some(bytes) {
        return Err("tensor shape/dtype/byte count differs".into());
    }
    let file = tensor["file"].as_str().ok_or("missing tensor file")?;
    if file.is_empty() || file.contains(['/', '\\']) || file == "." || file == ".." {
        return Err("tensor file must be a single relative component".into());
    }
    let offset = tensor["offset"].as_u64().ok_or("missing tensor offset")?;
    if offset.checked_add(bytes).is_none_or(|n| n > MAX_BYTES) {
        return Err("tensor span exceeds budget".into());
    }
    Ok(())
}

pub fn write_new(path: &Path, value: &Value) -> std::io::Result<()> {
    for parent in path.ancestors().skip(1) {
        if fs::symlink_metadata(parent)?.file_type().is_symlink() || parent.join(".git").exists() {
            return Err(std::io::Error::other(
                "diagnostic output must be external and no-follow",
            ));
        }
    }
    let mut file = File::create_new(path)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()
}

/// Immutable expected scope plus an ordered observed inventory.
pub struct Inventory {
    pub description: Value,
    tensors: Vec<Value>,
}

impl Inventory {
    pub fn begin(root: &Path, description: Value) -> Result<Self, String> {
        validate_scope(&description["scope"])?;
        if description["capture_id"].as_str().is_none_or(str::is_empty) {
            return Err("missing capture identity".into());
        }
        write_new(&root.join("scope.json"), &description).map_err(|e| e.to_string())?;
        Ok(Self {
            description,
            tensors: Vec::new(),
        })
    }

    pub fn record(&mut self, mut tensor: Value) -> Result<(), String> {
        validate_tensor(&tensor)?;
        if self.tensors.len() >= MAX_ENTRIES
            || self.tensors.iter().any(|t| t["name"] == tensor["name"])
        {
            return Err("duplicate or excessive observed probe".into());
        }
        tensor["ordinal"] = json!(self.tensors.len());
        self.tensors.push(tensor);
        Ok(())
    }

    pub fn finish(self, root: &Path, identity: Value) -> Result<(), String> {
        let missing: Vec<_> = self.description["scope"]["probes"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| {
                !self
                    .tensors
                    .iter()
                    .any(|t| p["name"] == t["name"] && p["shape"] == t["shape"])
            })
            .map(|p| p["name"].clone())
            .collect();
        let value = json!({"schema": SCHEMA, "description": self.description,
            "identity": identity, "tensors": self.tensors, "missing": missing,
            "completion": if missing.is_empty() {"full"} else {"incomplete"}});
        write_new(&root.join("producer.json"), &value).map_err(|e| e.to_string())?;
        if !missing.is_empty() {
            return Err(format!("missing required probes: {missing:?}"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventories_keep_order_and_refuse_missing_outputs() {
        let root = std::env::temp_dir().join(format!("sdk-diagnostic-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let scope = json!({"model":"external-example", "evidence":"chained", "contract":"example.exact.v1",
            "inputs":{"rows":[{"position":0}], "state":{"kind":"fresh"}},
            "probes":[{"name":"z", "shape":[1,2]}, {"name":"a", "shape":[1,2]}]});
        let description = json!({"capture_id":"fixture", "scope":scope});
        let mut inventory = Inventory::begin(&root, description.clone()).unwrap();
        assert!(Inventory::begin(&root, description).is_err());
        let tensor = json!({"name":"z", "shape":[1,2], "dtype":"F32", "bytes":8, "offset":0, "file":"z.bin"});
        inventory.record(tensor.clone()).unwrap();
        assert!(inventory.record(tensor).is_err());
        assert!(inventory.finish(&root, json!({})).is_err());
        let value: Value =
            serde_json::from_slice(&fs::read(root.join("producer.json")).unwrap()).unwrap();
        assert_eq!(value["missing"], json!(["a"]));
        assert_eq!(value["tensors"][0]["ordinal"], 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn descriptors_reject_unbounded_spans_and_traversal() {
        for file in ["../x", "/x", "a/b"] {
            assert!(
                validate_tensor(&json!({"shape":[1], "dtype":"F32", "bytes":4,
                "offset":0,"file":file}))
                .is_err()
            );
        }
        assert!(elements(&json!([u64::MAX, 2])).is_err());
        assert!(elements(&json!([0])).is_err());
    }
}
