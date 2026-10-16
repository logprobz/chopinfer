use crate::files::{Result, replace_file};
pub use runtime_bundle::preparation::default_cache_root;
use runtime_bundle::preparation::{CacheLease, cache_lock, managed_key, valid_digest};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

const RECORD_LIMIT: u64 = 256 * 1024 * 1024;
const VERSION: &str = "chopinfer-preparation-cache-v1\n";

fn canonical(value: &impl Serialize) -> Result<Vec<u8>> {
    let mut value = serde_json::to_value(value)?;
    value.sort_all_objects();
    Ok(serde_json::to_vec(&value)?)
}

pub fn digest(value: &impl Serialize) -> Result<String> {
    Ok(runtime_weights::sha256_bytes(&canonical(value)?))
}

pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
    pub image: bool,
}

impl Artifact {
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty()
            || self.name.contains('\\')
            || !Path::new(&self.name)
                .components()
                .all(|c| matches!(c, Component::Normal(_)))
            || !valid_digest(&self.sha256)
        {
            return Err("invalid cached artifact identity".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub kind: String,
    pub state: String,
    pub dependencies: Vec<String>,
    pub files: Vec<Artifact>,
    pub last_selected: u64,
    pub payload: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record<T> {
    version: u32,
    checksum: String,
    value: T,
}

pub fn read<T: DeserializeOwned + Serialize>(path: &Path) -> Result<T> {
    read_counted(path).map(|(value, _, _)| value)
}

pub fn read_counted<T: DeserializeOwned + Serialize>(path: &Path) -> Result<(T, u64, u64)> {
    let file = runtime_weights::open_nofollow(path)?;
    if file.metadata()?.len() > RECORD_LIMIT {
        return Err("cache record exceeds limit".into());
    }
    let mut bytes = Vec::new();
    file.take(RECORD_LIMIT + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > RECORD_LIMIT {
        return Err("cache record grew beyond limit".into());
    }
    let record: Record<T> = serde_json::from_slice(&bytes)?;
    let hashed = canonical(&record.value)?;
    if record.version != 1 || runtime_weights::sha256_bytes(&hashed) != record.checksum {
        return Err("cache record integrity mismatch".into());
    }
    Ok((record.value, bytes.len() as u64, hashed.len() as u64))
}

pub fn write<T: Serialize>(directory: &Path, name: &str, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec(&Record {
        version: 1,
        checksum: digest(value)?,
        value,
    })?;
    if bytes.len() as u64 > RECORD_LIMIT {
        return Err("cache record exceeds limit".into());
    }
    replace_file(directory, name, &bytes)
}

pub struct Cache {
    pub root: PathBuf,
}

impl Cache {
    pub fn open(root: &Path) -> Result<Self> {
        fs::create_dir_all(root)?;
        if !fs::symlink_metadata(root)?.is_dir() {
            return Err("cache root must be a real directory".into());
        }
        let root = fs::canonicalize(root)?;
        let marker = root.join("cache-version");
        let initialize = runtime_weights::open_regular_nofollow(
            &root.join("initialize.lock"),
            fs::OpenOptions::new().read(true).write(true).create(true),
        )?;
        initialize.lock()?;
        if !marker.try_exists()?
            && fs::read_dir(&root)?.any(|e| e.map_or(true, |e| e.file_name() != "initialize.lock"))
        {
            return Err("refusing a nonempty directory without a preparation-cache marker".into());
        }
        if !marker.try_exists()? {
            // Concurrent first creators may race; only one writes the same fixed marker.
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&marker)
            {
                Ok(mut file) => {
                    use std::io::Write;
                    file.write_all(VERSION.as_bytes())?;
                    file.sync_all()?;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(e.into()),
            }
        }
        let mut version = String::new();
        runtime_weights::open_nofollow(&marker)?
            .take(64)
            .read_to_string(&mut version)?;
        if version != VERSION {
            return Err("unsupported preparation cache version".into());
        }
        for name in ["locks", "records", "imports"] {
            let path = root.join(name);
            fs::create_dir_all(&path)?;
            if !fs::symlink_metadata(&path)?.is_dir() {
                return Err("cache metadata directory is a link".into());
            }
        }
        Ok(Self { root })
    }

    pub fn gate(&self, exclusive: bool) -> Result<File> {
        let lock = cache_lock(&self.root, "gate")?;
        if exclusive {
            lock.lock()?;
        } else {
            lock.lock_shared()?;
        }
        Ok(lock)
    }

    pub fn lock(&self, key: &str) -> Result<File> {
        let file = cache_lock(&self.root, key)?;
        file.lock()?;
        Ok(file)
    }

    /// Held for every read-modify-write of `key`'s record, so writers that
    /// change different fields of it (selection, publication, learned
    /// emission inputs) keep each other's changes.
    pub fn record_lock(&self, key: &str) -> Result<File> {
        self.path(key)?;
        self.lock(&format!("i-{}", digest(&("record", key))?))
    }

    pub fn path(&self, key: &str) -> Result<PathBuf> {
        if !managed_key(key) {
            return Err("invalid managed cache key".into());
        }
        Ok(self.root.join(key))
    }

    pub fn entry(&self, key: &str) -> Result<Entry> {
        self.entry_counted(key).map(|(entry, _, _)| entry)
    }

    pub fn entry_counted(&self, key: &str) -> Result<(Entry, u64, u64)> {
        self.path(key)?;
        let (entry, read, hashed): (Entry, _, _) =
            read_counted(&self.root.join("records").join(format!("{key}.json")))?;
        let kind = match &key[..2] {
            "p-" => "package",
            "w-" => "weights",
            _ => "snapshot",
        };
        if entry.kind != kind || entry.files.is_empty() {
            return Err("invalid cache entry kind or empty artifacts".into());
        }
        if entry.dependencies.iter().any(|key| !managed_key(key)) {
            return Err("invalid cached dependency".into());
        }
        for artifact in &entry.files {
            artifact.validate()?;
        }
        Ok((entry, read, hashed))
    }

    pub fn save(&self, key: &str, entry: &Entry) -> Result<()> {
        self.path(key)?;
        write(&self.root.join("records"), &format!("{key}.json"), entry)
    }

    pub fn entries(&self) -> Result<BTreeMap<String, Entry>> {
        let mut entries = BTreeMap::new();
        for file in fs::read_dir(self.root.join("records"))? {
            let name = file?.file_name().to_string_lossy().into_owned();
            if let Some(key) = name.strip_suffix(".json") {
                if !managed_key(key) {
                    continue;
                }
                let entry = self.entry(key).unwrap_or_else(|_| Entry {
                    kind: "corrupt".into(),
                    state: "invalid".into(),
                    dependencies: vec![],
                    files: vec![],
                    last_selected: 0,
                    payload: serde_json::Value::Null,
                });
                entries.insert(key.into(), entry);
            }
        }
        Ok(entries)
    }

    pub fn private_weights(&self, import_key: &str) -> Result<PathBuf> {
        if !import_key.strip_prefix("i-").is_some_and(valid_digest) {
            return Err("invalid import key".into());
        }
        // This provisional lock is held by the caller; abandoned work is never a hit.
        let path = self.root.join(format!("t-{}", &import_key[2..]));
        if path.try_exists()? {
            remove_tree(&path)?;
        }
        fs::create_dir(&path)?;
        Ok(path)
    }

    pub fn verify_files(&self, key: &str, entry: &Entry) -> Result<(u64, u64)> {
        if entry.state != "complete" {
            return Err("cache entry is not complete".into());
        }
        let directory = self.path(key)?;
        if !fs::symlink_metadata(&directory)?.is_dir() {
            return Err("cached artifact directory is a link".into());
        }
        let (mut images, mut metadata) = (0, 0);
        for artifact in &entry.files {
            artifact.validate()?;
            let path = directory.join(&artifact.name);
            let canonical = fs::canonicalize(path.parent().ok_or("invalid artifact parent")?)?;
            if canonical != path.parent().ok_or("invalid artifact parent")? {
                return Err("cached artifact escapes its directory".into());
            }
            // Every byte is read and hashed on each check: nothing records a
            // file as verified between checks.
            let file = runtime_weights::open_nofollow(&path)?;
            if file.metadata()?.len() != artifact.bytes {
                return Err(
                    format!("cached artifact {} has the wrong length", artifact.name).into(),
                );
            }
            let mut counted = (&file).take(artifact.bytes.saturating_add(1));
            let sha256 = runtime_weights::sha256_reader(&mut counted)?;
            let bytes = artifact.bytes.saturating_add(1) - counted.limit();
            if bytes != artifact.bytes || sha256 != artifact.sha256 {
                return Err(
                    format!("cached artifact {} differs from its record", artifact.name).into(),
                );
            }
            if artifact.image {
                images += bytes;
            } else {
                metadata += bytes;
            }
        }
        Ok((images, metadata))
    }

    pub fn select(&self, keys: &[String]) -> Result<CacheLease> {
        let lease = CacheLease::acquire(&self.root, keys)?;
        let selected = now();
        for key in keys {
            let _record = self.record_lock(key)?;
            let mut entry = self.entry(key)?;
            entry.last_selected = selected;
            self.save(key, &entry)?;
        }
        // One lock serializes the last-selection record, without serializing packing.
        let key = format!("i-{}", digest(&"last-selection")?);
        let _lock = self.lock(&key)?;
        write(&self.root, "last-selection.json", &keys)?;
        Ok(lease)
    }

    /// Called after dropping preparation locks. Every shared-inode consumer is
    /// invalidated together; intact weights survive package-only damage.
    pub fn quarantine(&self, key: &str) -> Result<()> {
        self.path(key)?;
        let _gate = self.gate(true)?;
        let entries = self.entries()?;
        let mut keys = vec![key.to_owned()];
        if key.starts_with("w-") {
            keys.extend(
                entries
                    .iter()
                    .filter(|(_, e)| e.kind == "package" && e.dependencies.iter().any(|d| d == key))
                    .map(|(k, _)| k.clone()),
            );
        }
        keys.sort();
        let mut locks = Vec::new();
        for key in &keys {
            let lock = cache_lock(&self.root, key)?;
            lock.try_lock().map_err(
                |_| "damaged cache entry is in active use; stop that selection before repairing it",
            )?;
            locks.push(lock);
        }
        for key in keys {
            let path = self.path(&key)?;
            if path.try_exists()? {
                let name = format!(
                    "q-{}-{}-{}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)?
                        .as_nanos(),
                    std::process::id(),
                    key
                );
                fs::rename(&path, self.root.join(name))?;
            }
            if let Some(mut entry) = entries.get(&key).cloned() {
                entry.state = "invalid".into();
                self.save(&key, &entry)?;
            }
        }
        File::open(&self.root)?.sync_all()?;
        Ok(())
    }
}

/// Never follows a cache symlink while removing an internal artifact tree.
pub fn remove_tree(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.is_dir() {
        for entry in fs::read_dir(path)? {
            remove_tree(&entry?.path())?;
        }
        fs::remove_dir(path)?;
    } else {
        fs::remove_file(path)?;
    }
    Ok(())
}
