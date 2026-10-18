//! The reference lock, `models/<m>/qual-references.lock.json`
//! (`tools/qual/refs.py`): every declared input a capture reads is read as
//! authenticated bytes, the digest its lock entry pins.

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::declaration::Loaded;
use crate::json::{self, sha256_hex};
use crate::references::declared_path;

pub const SCHEMA: &str = "chopinfer-qual-references/v1";
pub const FILE_NAME: &str = "qual-references.lock.json";

/// The lock's path beside the declaration: a variant file's is
/// `qual-references.VARIANT.lock.json` (`refs.lock_path`); the model's
/// declaration and every configuration expanded from it read the model's,
/// a configuration with its own entries where they differ
/// (`configurations.<name>.entries`). An external variant's is
/// `<variant root>/<model>/<variant>.lock.json`, beside its overlay.
pub fn path(loaded: &Loaded) -> PathBuf {
    if let (Some(root), Some(variant)) = (&loaded.variants, &loaded.declaration.variant) {
        return root
            .join(&loaded.declaration.model)
            .join(format!("{variant}.lock.json"));
    }
    let directory = loaded.root.join("models").join(&loaded.declaration.model);
    match &loaded.declaration.variant {
        Some(variant) if !configuration(loaded) => {
            directory.join(format!("qual-references.{variant}.lock.json"))
        }
        _ => directory.join(FILE_NAME),
    }
}

/// Whether `loaded` is a configuration expanded from the model's qual.json.
fn configuration(loaded: &Loaded) -> bool {
    loaded.declaration.variant.is_some()
        && loaded.variants.is_none()
        && loaded.path.file_name() == Some(std::ffi::OsStr::new(crate::declaration::FILE_NAME))
}

/// The lock entry of `key`: a configuration's own, else the model's.
fn entry<'a>(loaded: &Loaded, lock: &'a Value, key: &str) -> Option<&'a Value> {
    loaded
        .declaration
        .variant
        .as_ref()
        .filter(|_| configuration(loaded))
        .and_then(|variant| lock["configurations"][variant.as_str()]["entries"].get(key))
        .or_else(|| lock["entries"].get(key))
}

/// The model's reference lock (`refs.load_lock`), `None` when the model has
/// none (a declaration outside the repository, as in the tests). An
/// external variant always needs its lock.
pub fn load(loaded: &Loaded) -> Result<Option<Value>, String> {
    let path = path(loaded);
    if !path.exists() {
        if loaded.variants.is_some() {
            return Err(format!("reference lock is missing: {}", path.display()));
        }
        return Ok(None);
    }
    let (_, lock) = json::read(&path)?;
    if lock["schema"] != SCHEMA {
        return Err("reference lock: unsupported schema".into());
    }
    if lock["model"] != loaded.declaration.model.as_str() {
        return Err("reference lock: model differs from declaration".into());
    }
    if !lock["entries"].is_object() || !lock["objects"].is_object() {
        return Err("reference lock: entries and objects must be objects".into());
    }
    Ok(Some(lock))
}

/// The bytes of the declared input `key` at `path` (the input itself, or a
/// member of a declared directory), authenticated by its lock entry
/// (`refs.verify_input`, `refs.read_member`). Without a lock the bytes are
/// read as they are.
pub fn read(loaded: &Loaded, key: &str, path: &Path) -> Result<Vec<u8>, String> {
    let read =
        |path: &Path| std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()));
    let Some(lock) = load(loaded)? else {
        return read(path);
    };
    let entry =
        entry(loaded, &lock, key).ok_or_else(|| format!("lock entry {key}: not adopted"))?;
    let declared = &loaded.value["inputs"][key];
    if declared.is_null() || &entry["path"] != declared {
        return Err(format!("lock entry {key}: declaration differs"));
    }
    if !path.exists() {
        return Err(format!("lock entry {key}: missing input"));
    }
    let member = if entry["directory"] == Value::Bool(true) {
        let input = loaded
            .declaration
            .input(key)
            .ok_or_else(|| format!("lock entry {key}: declaration differs"))?;
        let base = declared_path(input, &loaded.root, &|name| loaded.environment(name))
            .ok_or_else(|| format!("lock entry {key}: missing input"))?;
        path.strip_prefix(&base)
            .map_err(|_| format!("lock entry {key}: {} is outside the input", path.display()))?
            .to_str()
            .ok_or_else(|| format!("lock entry {key}: a member name is not UTF-8"))?
            .to_owned()
    } else {
        ".".into()
    };
    let pinned = entry["members"][member.as_str()]
        .as_str()
        .ok_or_else(|| format!("lock entry {key}: member {member} is not adopted"))?;
    let bytes = read(path)?;
    if sha256_hex(&bytes) != pinned {
        return Err(format!("lock entry {key}: sha256 differs for {member}"));
    }
    Ok(bytes)
}

/// The external variant `loaded` was read from, as receipts record it: the
/// variant root and the overlay, its reference lock and the checkpoint lock
/// it pins, each by its path under that root and its SHA-256 (the overlay's
/// as read when it was expanded). `None` for a declaration in the tree.
pub fn variant_source(loaded: &Loaded) -> Result<Option<Value>, String> {
    let Some(root) = &loaded.variants else {
        return Ok(None);
    };
    let source = &loaded.value["variant_source"];
    let digest = |relative: &str| -> Result<Value, String> {
        let path = root.join(relative);
        let bytes = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        Ok(serde_json::json!({"path": relative, "sha256": sha256_hex(&bytes)}))
    };
    let lock = path(loaded);
    let lock = lock
        .strip_prefix(root)
        .map_err(|_| format!("{}: not under the variant root", lock.display()))?
        .to_str()
        .ok_or_else(|| "the reference lock's path is not UTF-8".to_owned())?
        .to_owned();
    Ok(Some(serde_json::json!({
        "root": root.display().to_string(),
        "overlay": {"path": source["overlay"], "sha256": source["sha256"]},
        "lock": digest(&lock)?,
        "checkpoint_lock": digest(&loaded.declaration.checkpoint.lock)?,
    })))
}
