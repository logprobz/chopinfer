//! Explicit cleanup with reference and inode accounting. No automatic eviction.
use super::cache::{self, Cache, Entry};
use crate::files::Result;
use runtime_bundle::preparation::{CacheEntry, CacheUsage, cache_lock, managed_key};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

#[derive(Default)]
struct Inode {
    allocated: u64,
    links: u64,
    owners: Vec<String>,
}

fn files(path: &Path, visit: &mut impl FnMut(&fs::Metadata)) -> Result<()> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    if metadata.is_dir() {
        let entries = match fs::read_dir(path) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        for entry in entries {
            files(&entry?.path(), visit)?;
        }
    } else if metadata.is_file() {
        visit(&metadata);
    }
    // A link is reported as damage elsewhere. Never follow it to count bytes.
    Ok(())
}

fn scan(
    cache: &Cache,
    entries: &BTreeMap<String, Entry>,
    protected: &BTreeSet<String>,
    selected: &BTreeSet<String>,
) -> Result<CacheUsage> {
    let mut result = CacheUsage::default();
    let mut inodes = BTreeMap::<(u64, u64), Inode>::new();
    for (key, entry) in entries {
        let mut row = CacheEntry {
            identity: key.clone(),
            kind: entry.kind.clone(),
            state: entry.state.clone(),
            last_selected: entry.last_selected,
            references: entry.dependencies.clone(),
            protected: protected.contains(key),
            selected_for_pruning: selected.contains(key),
            ..Default::default()
        };
        let path = cache.root.join(key);
        if path.try_exists()? {
            files(&path, &mut |m| {
                row.logical_bytes += m.len();
                row.allocated_bytes += m.blocks() * 512;
                let inode = inodes.entry((m.dev(), m.ino())).or_default();
                inode.allocated = m.blocks() * 512;
                inode.links = m.nlink();
                inode.owners.push(key.clone());
            })?;
        }
        result.logical_bytes += row.logical_bytes;
        result.entries.push(row);
    }
    for inode in inodes.values() {
        result.unique_allocated_bytes += inode.allocated;
        if inode.links > inode.owners.len() as u64 {
            result.external_alias_bytes += inode.allocated;
        }
        if inode.links == inode.owners.len() as u64
            && inode.owners.iter().all(|key| selected.contains(key))
        {
            result.reclaimable_bytes += inode.allocated;
        }
    }
    Ok(result)
}

fn selected_keys(cache: &Cache) -> Result<BTreeSet<String>> {
    let path = cache.root.join("last-selection.json");
    if !path.try_exists()? {
        return Ok(BTreeSet::new());
    }
    let keys = cache::read::<Vec<String>>(&path)?;
    if keys.iter().any(|key| !managed_key(key)) {
        return Err("invalid retained cache selection".into());
    }
    Ok(keys.into_iter().collect())
}

fn all_entries(cache: &Cache) -> Result<BTreeMap<String, Entry>> {
    let mut entries = cache.entries()?;
    for file in fs::read_dir(&cache.root)? {
        let file = file?;
        let name = file.file_name().to_string_lossy().into_owned();
        let incomplete = name
            .strip_prefix("t-")
            .is_some_and(runtime_bundle::preparation::valid_digest);
        let quarantined = name.starts_with("q-")
            && name
                .rsplit_once('-')
                .is_some_and(|(_, digest)| runtime_bundle::preparation::valid_digest(digest));
        let abandoned_package = name
            .strip_prefix(".p-")
            .and_then(|n| n.split_once(".compiling-"))
            .is_some_and(|(digest, suffix)| {
                runtime_bundle::preparation::valid_digest(digest)
                    && suffix.split('-').count() == 2
                    && suffix
                        .split('-')
                        .all(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
            });
        if incomplete
            || quarantined
            || abandoned_package
            || runtime_bundle::preparation::managed_key(&name) && !entries.contains_key(&name)
        {
            entries.entry(name).or_insert(Entry {
                kind: "abandoned".into(),
                state: if quarantined {
                    "quarantined"
                } else {
                    "incomplete"
                }
                .into(),
                dependencies: vec![],
                files: vec![],
                last_selected: 0,
                payload: serde_json::Value::Null,
            });
        }
    }
    Ok(entries)
}

pub fn list(root: &Path) -> Result<CacheUsage> {
    let cache = Cache::open(root)?;
    let _gate = cache.gate(false)?;
    let entries = all_entries(&cache)?;
    let mut protected = selected_keys(&cache)?;
    for key in entries.keys().filter(|k| managed_key(k)) {
        let lock = cache_lock(&cache.root, key)?;
        if lock.try_lock().is_err() {
            protected.insert(key.clone());
        }
    }
    scan(&cache, &entries, &protected, &BTreeSet::new())
}

pub fn prune(root: &Path, apply: bool, identities: &[String]) -> Result<CacheUsage> {
    let cache = Cache::open(root)?;
    let gate = cache_lock(&cache.root, "gate")?;
    if gate.try_lock().is_err() {
        let mut usage = list(root)?;
        for entry in &mut usage.entries {
            entry.protected = true;
        }
        return Ok(usage);
    }
    let entries = all_entries(&cache)?;
    let mut protected = selected_keys(&cache)?;
    let mut locks = Vec::new();
    for key in entries.keys().filter(|k| managed_key(k)) {
        let lock = cache_lock(&cache.root, key)?;
        if lock.try_lock().is_err() {
            protected.insert(key.clone());
        } else {
            locks.push(lock);
        }
    }
    let mut candidates = if identities.is_empty() {
        entries.keys().cloned().collect::<BTreeSet<_>>()
    } else {
        let mut selected = BTreeSet::new();
        for key in identities {
            if !entries.contains_key(key) {
                return Err(format!("unknown cache entry {key}").into());
            }
            selected.insert(key.clone());
        }
        selected
    };
    candidates.retain(|key| !protected.contains(key));
    // Retained packages and snapshots keep their weight dependencies. A corrupt
    // retained record conservatively protects all weights because its edges are unknown.
    loop {
        let mut remove = BTreeSet::new();
        for (key, entry) in &entries {
            if candidates.contains(key) {
                continue;
            }
            remove.extend(entry.dependencies.iter().cloned());
            if entry.kind == "corrupt" {
                remove.extend(entries.keys().filter(|k| k.starts_with("w-")).cloned());
            }
        }
        let before = candidates.len();
        candidates.retain(|key| !remove.contains(key));
        if candidates.len() == before {
            break;
        }
    }
    let mut usage = scan(&cache, &entries, &protected, &candidates)?;
    if !apply {
        return Ok(usage);
    }
    // Non-weight references disappear before the shared originals.
    let mut ordered = candidates.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|key| (key.starts_with("w-"), key.as_str()));
    for key in ordered {
        let path = cache.root.join(key);
        if fs::symlink_metadata(&path).is_ok() {
            cache::remove_tree(&path)?;
        }
        if managed_key(key) {
            let record = cache.root.join("records").join(format!("{key}.json"));
            if record.try_exists()? {
                fs::remove_file(record)?;
            }
        }
        usage.removed.push(key.clone());
    }
    remove_import_references(&cache, &candidates)?;
    fs::File::open(&cache.root)?.sync_all()?;
    Ok(usage)
}

/// The import record an older cache kept for a prepared-weight snapshot
/// (`s-`); pruning the snapshot removes it.
#[derive(serde::Serialize, serde::Deserialize)]
struct Import {
    snapshot: String,
}

fn remove_import_references(cache: &Cache, removed: &BTreeSet<String>) -> Result<()> {
    for file in fs::read_dir(cache.root.join("imports"))? {
        let path: PathBuf = file?.path();
        let record = cache::read::<Import>(&path);
        if record.as_ref().is_err()
            || record
                .as_ref()
                .is_ok_and(|record| removed.contains(&record.snapshot))
        {
            fs::remove_file(path)?;
        }
    }
    Ok(())
}
