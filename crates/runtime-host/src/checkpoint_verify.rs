//! `chopinfer verify --model DIR`: checks every file of a checkpoint directory
//! against the digest Hugging Face records for it, offline: SHA-256 for LFS
//! files and manifest entries, git blob SHA-1 for the hub's other files.
//!
//! A direct load authenticates only the safetensors headers and file lengths,
//! so this is the content check for release qualification. It reads no network
//! and trusts only these recorded values, first match per file:
//!
//! - a manifest (`--manifest FILE[#ENTRY]`): a JSON object from relative path
//!   to `{"sha256": HEX, "byte_length": N}` (`byte_length` optional), or an
//!   object of such objects selected by `#ENTRY`;
//! - a hub cache snapshot (`.../snapshots/<revision>/`), whose entries link to
//!   `blobs/<name>`: the hub names an LFS blob by its SHA-256 and any other file
//!   by its git blob SHA-1;
//! - `hf download --local-dir` metadata, `.cache/huggingface/download/<path>.metadata`,
//!   whose second line is the file's etag: the LFS SHA-256, or the git blob
//!   SHA-1 of a small file.
//!
//! Only regular files beneath the directory are read, opened without
//! following a link and only if still the file the walk listed. A link is
//! read only in a hub snapshot, when it names the snapshot's own
//! `blobs/<digest>` exactly; any other link fails unread. The directory is
//! listed again after hashing, and a file added, removed, replaced or
//! rewritten meanwhile fails the run.
//!
//! The directory's `model.safetensors.index.json` files name every shard, so a
//! shard they list that is absent is reported missing, as is every manifest
//! entry without a file.
use runtime_weights::Sha256State;
use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;

/// Files and directories visited before the walk refuses the directory.
const ENTRIES_MAX: usize = 1 << 20;
/// Directory depth below the checkpoint root the walk descends to.
const DEPTH_MAX: usize = 8;
/// Largest manifest, index or metadata document read.
const DOCUMENT_BYTES_MAX: u64 = 64 << 20;

/// What `verify` checks and how.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifyOptionsV1 {
    pub root: PathBuf,
    /// A manifest file and, after `#`, the entry of it that lists the files.
    pub manifest: Option<(PathBuf, Option<String>)>,
    /// Accepts files without a recorded digest, except safetensors files and
    /// their indexes, instead of failing on them.
    pub allow_unrecorded: bool,
    /// Files hashed at once.
    pub jobs: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Algorithm {
    Sha256,
    /// SHA-1 over `blob <length>\0` and the content, as git names a blob.
    GitBlobSha1,
}

#[derive(Clone, Debug)]
struct Recorded {
    algorithm: Algorithm,
    digest: String,
    byte_length: Option<u64>,
    source: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Outcome {
    Verified,
    Mismatch {
        expected: String,
        actual: String,
    },
    LengthMismatch {
        expected: u64,
        actual: u64,
    },
    Unreadable(String),
    /// Not read: a link out of the checkpoint.
    Refused(String),
    Unrecorded,
}

/// The verdict for one directory.
#[derive(Clone, Debug, Default)]
pub struct VerifyReportV1 {
    /// One line per file and per missing name, in name order.
    pub lines: Vec<String>,
    pub verified: usize,
    pub failed: usize,
    pub missing: usize,
    pub unrecorded: usize,
    pub bytes: u64,
    pub sources: BTreeMap<&'static str, usize>,
}

impl VerifyReportV1 {
    /// Whether the directory passes: every file verified, or accepted as
    /// unrecorded under `--allow-unrecorded`, and nothing missing.
    #[must_use]
    pub fn passed(&self, allow_unrecorded: bool) -> bool {
        self.failed == 0 && self.missing == 0 && (allow_unrecorded || self.unrecorded == 0)
    }

    #[must_use]
    pub fn summary(&self, seconds: f64) -> String {
        let sources = self
            .sources
            .iter()
            .map(|(source, count)| format!("{source} {count}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "verify: {} verified, {} failed, {} missing, {} unrecorded; {} bytes hashed in {seconds:.1} s (recorded by {})",
            self.verified,
            self.failed,
            self.missing,
            self.unrecorded,
            self.bytes,
            if sources.is_empty() {
                "nothing".into()
            } else {
                sources
            }
        )
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// SHA-1, only to check git blob names; it authenticates nothing a SHA-256
/// is available for.
struct Sha1 {
    state: [u32; 5],
    buffer: Vec<u8>,
    length: u64,
}

impl Sha1 {
    fn new() -> Self {
        Self {
            state: [
                0x6745_2301,
                0xefcd_ab89,
                0x98ba_dcfe,
                0x1032_5476,
                0xc3d2_e1f0,
            ],
            buffer: Vec::with_capacity(64),
            length: 0,
        }
    }

    fn block(&mut self, block: &[u8]) {
        let mut w = [0_u32; 80];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = self.state;
        for (i, word) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5a82_7999),
                20..=39 => (b ^ c ^ d, 0x6ed9_eba1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1b_bcdc),
                _ => (b ^ c ^ d, 0xca62_c1d6),
            };
            let t = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = t;
        }
        for (state, value) in self.state.iter_mut().zip([a, b, c, d, e]) {
            *state = state.wrapping_add(value);
        }
    }

    fn update(&mut self, mut bytes: &[u8]) {
        self.length += bytes.len() as u64;
        if !self.buffer.is_empty() {
            let take = (64 - self.buffer.len()).min(bytes.len());
            self.buffer.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
            if self.buffer.len() == 64 {
                let block = std::mem::take(&mut self.buffer);
                self.block(&block);
                self.buffer = block;
                self.buffer.clear();
            }
        }
        let mut blocks = bytes.chunks_exact(64);
        for block in &mut blocks {
            self.block(block);
        }
        self.buffer.extend_from_slice(blocks.remainder());
    }

    fn finish(mut self) -> String {
        let bits = self.length.wrapping_mul(8);
        let mut tail = std::mem::take(&mut self.buffer);
        tail.push(0x80);
        while tail.len() % 64 != 56 {
            tail.push(0);
        }
        tail.extend_from_slice(&bits.to_be_bytes());
        for block in tail.chunks_exact(64) {
            self.block(block);
        }
        hex(&self
            .state
            .iter()
            .flat_map(|word| word.to_be_bytes())
            .collect::<Vec<_>>())
    }
}

fn read_document(path: &Path) -> Result<Vec<u8>, String> {
    use std::os::unix::fs::OpenOptionsExt;
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if !file.metadata().map_err(|e| e.to_string())?.is_file() {
        return Err(format!("{} is not a regular file", path.display()));
    }
    let mut bytes = Vec::new();
    file.take(DOCUMENT_BYTES_MAX + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() as u64 > DOCUMENT_BYTES_MAX {
        return Err(format!(
            "{} exceeds {DOCUMENT_BYTES_MAX} bytes",
            path.display()
        ));
    }
    Ok(bytes)
}

/// What the walk found under a name, without following links.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// A file, by the device, inode, length and modification time the walk
    /// saw.
    File {
        device: u64,
        inode: u64,
        length: u64,
        modified: (i64, i64),
    },
    Link,
    /// Refused unread, for the reason given.
    Refused(&'static str),
}

/// Where a name's bytes are read from: the file itself, or the blob a hub
/// snapshot links it to; opened only if it is still that device and inode.
struct Source {
    path: PathBuf,
    device: u64,
    inode: u64,
    /// The link and its target, rechecked after the read.
    link: Option<(PathBuf, PathBuf)>,
    recorded: Option<Recorded>,
}

/// The source of `name`. A link is read only in a hub snapshot and only when
/// it names `blobs/<digest>` of the snapshot's own repository exactly
/// (`../../blobs/<digest>`, one more `../` per directory level); any other
/// link is refused without reading its target.
fn source(root: &Path, name: &str, kind: Kind) -> Result<Source, String> {
    use std::os::unix::fs::MetadataExt;
    if let Kind::Refused(reason) = kind {
        return Err(reason.into());
    }
    if let Kind::File { device, inode, .. } = kind {
        return Ok(Source {
            path: root.join(name),
            device,
            inode,
            link: None,
            recorded: None,
        });
    }
    let refuse = || {
        format!(
            "a link; only a hub snapshot's links to its own blobs are read, and {name} is not one"
        )
    };
    if root.parent().and_then(Path::file_name) != Some("snapshots".as_ref()) {
        return Err(refuse());
    }
    let link = root.join(name);
    let target = fs::read_link(&link).map_err(|_| refuse())?;
    let depth = name.matches('/').count();
    let parts: Vec<Component> = target.components().collect();
    let blob = match parts.as_slice() {
        [
            parents @ ..,
            Component::Normal(blobs),
            Component::Normal(blob),
        ] if parents.len() == depth + 2
            && parents.iter().all(|c| *c == Component::ParentDir)
            && *blobs == "blobs" =>
        {
            blob.to_str().ok_or_else(refuse)?.to_owned()
        }
        _ => return Err(refuse()),
    };
    let blobs = root
        .parent()
        .and_then(Path::parent)
        .ok_or_else(refuse)?
        .join("blobs");
    let directory = fs::symlink_metadata(&blobs).map_err(|_| refuse())?;
    let path = blobs.join(&blob);
    let metadata = fs::symlink_metadata(&path).map_err(|e| format!("{name}: {e}"))?;
    if !directory.is_dir() || !metadata.is_file() {
        return Err(refuse());
    }
    let mut recorded = named(&blob, "hub blob name").ok_or_else(refuse)?;
    if recorded.algorithm == Algorithm::GitBlobSha1 {
        recorded.source = "hub git blob name";
    }
    Ok(Source {
        path,
        device: metadata.dev(),
        inode: metadata.ino(),
        link: Some((link, target)),
        recorded: Some(recorded),
    })
}

/// Opens `path` without following a final link and only if it is still the
/// regular file `device` and `inode` name.
fn open_confined(path: &Path, device: u64, inode: u64) -> Result<fs::File, String> {
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    // Non-blocking, so a FIFO swapped in after the walk cannot block the
    // open; a regular file reads the same either way.
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|e| e.to_string())?;
    let metadata = file.metadata().map_err(|e| e.to_string())?;
    if !metadata.is_file() {
        return Err("not a regular file".into());
    }
    if metadata.dev() != device || metadata.ino() != inode {
        return Err("the file changed after the directory was listed".into());
    }
    Ok(file)
}

/// The regular files and links below `root`, by relative `/`-separated name,
/// without the hub download metadata directory.
fn walk(root: &Path) -> Result<Vec<(String, Kind)>, String> {
    let mut names = Vec::new();
    let mut pending = vec![(PathBuf::new(), 0_usize)];
    let mut visited = 0_usize;
    while let Some((relative, depth)) = pending.pop() {
        let directory = root.join(&relative);
        let entries =
            fs::read_dir(&directory).map_err(|e| format!("{}: {e}", directory.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            visited += 1;
            if visited > ENTRIES_MAX {
                return Err(format!(
                    "{} holds more than {ENTRIES_MAX} entries",
                    root.display()
                ));
            }
            let name = entry
                .file_name()
                .into_string()
                .map_err(|name| format!("{} is not UTF-8", Path::new(&name).display()))?;
            let path = relative.join(&name);
            let kind = entry.file_type().map_err(|e| e.to_string())?;
            if kind.is_dir() && !name.contains('\\') {
                if depth == 0 && name == ".cache" {
                    continue;
                }
                if depth + 1 > DEPTH_MAX {
                    return Err(format!(
                        "{} is nested deeper than {DEPTH_MAX} directories",
                        root.join(&path).display()
                    ));
                }
                pending.push((path, depth + 1));
            } else {
                let kind = if name.contains('\\') {
                    // Names are exact Unix components: a backslash would be
                    // a separator elsewhere, so no such name is trusted.
                    Kind::Refused("a backslash in its name, which a checkpoint file never has")
                } else if kind.is_symlink() {
                    Kind::Link
                } else if !kind.is_file() {
                    // A FIFO, socket or device: never opened, since opening
                    // or reading one can block.
                    Kind::Refused("not a regular file")
                } else {
                    use std::os::unix::fs::MetadataExt;
                    let metadata = entry.metadata().map_err(|e| e.to_string())?;
                    Kind::File {
                        device: metadata.dev(),
                        inode: metadata.ino(),
                        length: metadata.len(),
                        modified: (metadata.mtime(), metadata.mtime_nsec()),
                    }
                };
                let name = path
                    .iter()
                    .map(|part| part.to_str().ok_or("checkpoint name is not UTF-8"))
                    .collect::<Result<Vec<_>, _>>()?
                    .join("/");
                names.push((name, kind));
            }
        }
    }
    names.sort_by(|a, b| a.0.cmp(&b.0));
    if let Some(pair) = names.windows(2).find(|pair| pair[0].0 == pair[1].0) {
        return Err(format!("{} is listed twice", pair[0].0));
    }
    Ok(names)
}

fn manifest_records(
    path: &Path,
    entry: Option<&str>,
) -> Result<BTreeMap<String, Recorded>, String> {
    let value: serde_json::Value = serde_json::from_slice(&read_document(path)?)
        .map_err(|e| format!("manifest {}: {e}", path.display()))?;
    let files = match entry {
        Some(entry) => value
            .get(entry)
            .ok_or_else(|| format!("manifest {} has no entry {entry}", path.display()))?,
        None => &value,
    };
    let files = files.as_object().ok_or_else(|| {
        format!(
            "manifest {} does not map file names to digests",
            path.display()
        )
    })?;
    let mut records = BTreeMap::new();
    for (name, record) in files {
        let digest = record
            .get("sha256")
            .and_then(serde_json::Value::as_str)
            .filter(|digest| is_hex(digest, 64))
            .ok_or_else(|| {
                format!(
                    "manifest {} entry {name} has no lowercase hex sha256",
                    path.display()
                )
            })?;
        let byte_length = match record.get("byte_length") {
            None => None,
            Some(length) => Some(length.as_u64().ok_or_else(|| {
                format!(
                    "manifest {} entry {name} has an invalid byte_length",
                    path.display()
                )
            })?),
        };
        if !Path::new(name)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
        {
            return Err(format!(
                "manifest {} names {name}, which is not a relative path",
                path.display()
            ));
        }
        records.insert(
            name.clone(),
            Recorded {
                algorithm: Algorithm::Sha256,
                digest: digest.into(),
                byte_length,
                source: "manifest",
            },
        );
    }
    Ok(records)
}

/// A digest named by the hub: a 64-digit name is an LFS SHA-256, a 40-digit
/// one a git blob SHA-1.
fn named(value: &str, source: &'static str) -> Option<Recorded> {
    let algorithm = if is_hex(value, 64) {
        Algorithm::Sha256
    } else if is_hex(value, 40) {
        Algorithm::GitBlobSha1
    } else {
        return None;
    };
    Some(Recorded {
        algorithm,
        digest: value.into(),
        byte_length: None,
        source,
    })
}

/// The etag `hf download --local-dir` recorded for `name`, read only from a
/// regular file beneath the directory's own `.cache/huggingface/download`.
fn local_dir_metadata(root: &Path, name: &str) -> Option<Recorded> {
    use std::os::unix::fs::MetadataExt;
    let path = root
        .join(".cache/huggingface/download")
        .join(format!("{name}.metadata"));
    if fs::canonicalize(path.parent()?).ok()? != path.parent()? {
        return None;
    }
    let metadata = fs::symlink_metadata(&path).ok()?;
    let mut bytes = Vec::new();
    open_confined(&path, metadata.dev(), metadata.ino())
        .ok()?
        .take(4096)
        .read_to_end(&mut bytes)
        .ok()?;
    let text = String::from_utf8(bytes).ok()?;
    let etag = text.lines().nth(1)?.trim().trim_matches('"');
    named(etag, "hf download metadata")
}

fn hash(source: &Source, algorithm: Algorithm) -> Result<(String, u64), String> {
    let mut file = open_confined(&source.path, source.device, source.inode)?;
    let length = file.metadata().map_err(|e| e.to_string())?.len();
    let mut buffer = vec![0_u8; 8 << 20];
    let mut read = 0_u64;
    let mut sha256 = Sha256State::new();
    let mut sha1 = Sha1::new();
    if algorithm == Algorithm::GitBlobSha1 {
        sha1.update(format!("blob {length}\0").as_bytes());
    }
    loop {
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        read += count as u64;
        match algorithm {
            Algorithm::Sha256 => sha256.update(&buffer[..count]).map_err(|e| e.to_string())?,
            Algorithm::GitBlobSha1 => sha1.update(&buffer[..count]),
        }
    }
    if read != length {
        return Err(format!("read {read} bytes of a {length}-byte file"));
    }
    if let Some((link, target)) = &source.link
        && fs::read_link(link).ok().as_ref() != Some(target)
    {
        return Err("the link changed while its blob was read".into());
    }
    Ok((
        match algorithm {
            Algorithm::Sha256 => sha256.finalize(),
            Algorithm::GitBlobSha1 => sha1.finish(),
        },
        read,
    ))
}

/// Shards `index` names, relative to the checkpoint root.
fn indexed_shards(index: &str, source: &Source) -> Result<Vec<String>, String> {
    let mut bytes = Vec::new();
    open_confined(&source.path, source.device, source.inode)
        .map_err(|e| format!("{index}: {e}"))?
        .take(DOCUMENT_BYTES_MAX + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("{index}: {e}"))?;
    if bytes.len() as u64 > DOCUMENT_BYTES_MAX {
        return Err(format!("{index} exceeds {DOCUMENT_BYTES_MAX} bytes"));
    }
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("{index}: {e}"))?;
    let directory = Path::new(index).parent().unwrap_or(Path::new(""));
    let mut shards: Vec<String> = value
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| format!("{index} has no weight_map"))?
        .values()
        .map(|shard| {
            shard
                .as_str()
                .filter(|shard| {
                    Path::new(shard)
                        .components()
                        .all(|component| matches!(component, Component::Normal(_)))
                })
                .map(|shard| {
                    directory
                        .join(shard)
                        .to_str()
                        .unwrap_or_default()
                        .to_owned()
                })
                .ok_or_else(|| format!("{index} names a shard that is not a relative path"))
        })
        .collect::<Result<_, _>>()?;
    shards.sort();
    shards.dedup();
    Ok(shards)
}

fn weights(name: &str) -> bool {
    name.ends_with(".safetensors") || name.ends_with(".safetensors.index.json")
}

/// Checks every file of `options.root`.
pub fn verify_checkpoint_v1(options: &VerifyOptionsV1) -> Result<VerifyReportV1, String> {
    let root = fs::canonicalize(&options.root)
        .map_err(|e| format!("--model {}: {e}", options.root.display()))?;
    if !fs::metadata(&root).map_err(|e| e.to_string())?.is_dir() {
        return Err(format!("--model {} is not a directory", root.display()));
    }
    let manifest = match &options.manifest {
        Some((path, entry)) => manifest_records(path, entry.as_deref())?,
        None => BTreeMap::new(),
    };
    let entries = walk(&root)?;
    let names: Vec<String> = entries.iter().map(|(name, _)| name.clone()).collect();
    let mut missing: BTreeMap<String, String> = manifest
        .keys()
        .filter(|name| names.binary_search(name).is_err())
        .map(|name| (name.clone(), "listed in the manifest".to_owned()))
        .collect();
    let work: Vec<(String, Result<Source, String>, Option<Recorded>)> = entries
        .iter()
        .map(|(name, kind)| {
            let source = source(&root, name, *kind);
            let recorded = manifest
                .get(name)
                .cloned()
                .or_else(|| source.as_ref().ok().and_then(|s| s.recorded.clone()))
                .or_else(|| local_dir_metadata(&root, name));
            (name.clone(), source, recorded)
        })
        .collect();
    for (index, source, _) in work
        .iter()
        .filter(|(name, _, _)| name.ends_with(".safetensors.index.json"))
    {
        let Ok(source) = source else { continue };
        for shard in indexed_shards(index, source)? {
            if names.binary_search(&shard).is_err() {
                missing
                    .entry(shard)
                    .or_insert_with(|| format!("listed in {index}"));
            }
        }
    }
    // The largest files first, so the last worker does not start a big file late.
    let mut order: Vec<usize> = (0..work.len()).collect();
    order.sort_by_key(|&i| {
        std::cmp::Reverse(
            work[i]
                .1
                .as_ref()
                .ok()
                .and_then(|source| fs::symlink_metadata(&source.path).ok())
                .map_or(0, |m| m.len()),
        )
    });
    let next = Mutex::new(order.into_iter());
    let results: Mutex<Vec<Option<(Outcome, u64)>>> = Mutex::new(vec![None; work.len()]);
    std::thread::scope(|scope| {
        for _ in 0..options.jobs.max(1) {
            scope.spawn(|| {
                loop {
                    let Some(i) = next.lock().expect("work queue").next() else {
                        break;
                    };
                    let (_, source, recorded) = &work[i];
                    let outcome = match (source, recorded) {
                        (Err(refusal), _) => (Outcome::Refused(refusal.clone()), 0),
                        (Ok(_), None) => (Outcome::Unrecorded, 0),
                        (Ok(source), Some(recorded)) => match hash(source, recorded.algorithm) {
                            Err(error) => (Outcome::Unreadable(error), 0),
                            Ok((_, read)) if recorded.byte_length.is_some_and(|n| n != read) => (
                                Outcome::LengthMismatch {
                                    expected: recorded.byte_length.unwrap_or_default(),
                                    actual: read,
                                },
                                read,
                            ),
                            Ok((actual, read)) if actual != recorded.digest => (
                                Outcome::Mismatch {
                                    expected: recorded.digest.clone(),
                                    actual,
                                },
                                read,
                            ),
                            Ok((_, read)) => (Outcome::Verified, read),
                        },
                    };
                    results.lock().expect("results")[i] = Some(outcome);
                }
            });
        }
    });
    let results = results
        .into_inner()
        .map_err(|_| "a verify worker panicked")?;
    // A verdict covers the directory as it was: a file added, removed,
    // replaced or rewritten while it was hashed fails the run.
    let after = walk(&root)?;
    if after != entries {
        let before: BTreeMap<_, _> = entries.iter().cloned().collect();
        let now: BTreeMap<_, _> = after.iter().cloned().collect();
        let changed: Vec<&String> = before
            .keys()
            .chain(now.keys())
            .filter(|name| before.get(*name) != now.get(*name))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        return Err(format!(
            "{} changed during verification ({}); verify an unchanging directory",
            root.display(),
            changed
                .iter()
                .map(|name| name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let mut report = VerifyReportV1::default();
    let mut lines: BTreeMap<String, String> = BTreeMap::new();
    for ((name, _, recorded), result) in work.iter().zip(results) {
        let (outcome, read) = result.ok_or("a file was not checked")?;
        report.bytes += read;
        let line = match outcome {
            Outcome::Verified => {
                let recorded = recorded.as_ref().ok_or("a verified file has no record")?;
                report.verified += 1;
                *report.sources.entry(recorded.source).or_default() += 1;
                format!("ok {name} ({})", recorded.source)
            }
            Outcome::Mismatch { expected, actual } => {
                report.failed += 1;
                format!(
                    "MISMATCH {name}: recorded {expected} ({}), content {actual}",
                    recorded.as_ref().map_or("", |r| r.source)
                )
            }
            Outcome::LengthMismatch { expected, actual } => {
                report.failed += 1;
                format!("MISMATCH {name}: recorded {expected} bytes, file has {actual}")
            }
            Outcome::Unreadable(error) => {
                report.failed += 1;
                format!("UNREADABLE {name}: {error}")
            }
            Outcome::Refused(reason) => {
                report.failed += 1;
                format!("REFUSED {name}: {reason}")
            }
            Outcome::Unrecorded if weights(name) => {
                // A weight file is never accepted without its recorded digest.
                report.failed += 1;
                format!("UNRECORDED {name}: no recorded SHA-256 for a weight file")
            }
            Outcome::Unrecorded => {
                report.unrecorded += 1;
                format!("unrecorded {name}: no recorded digest")
            }
        };
        lines.insert(name.clone(), line);
    }
    for (name, reason) in missing {
        report.missing += 1;
        lines.insert(name.clone(), format!("MISSING {name}: {reason}"));
    }
    report.lines = lines.into_values().collect();
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha1_matches_the_standard_vectors_and_git_blob_names() {
        let digest = |bytes: &[u8]| {
            let mut sha1 = Sha1::new();
            sha1.update(bytes);
            sha1.finish()
        };
        assert_eq!(digest(b""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(digest(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
        let long = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        assert_eq!(digest(long), "84983e441c3bd26ebaae4aa1f95129e5e54670f1");
        // Updates split across block boundaries hash like one update.
        let mut split = Sha1::new();
        for piece in long.chunks(7) {
            split.update(piece);
        }
        assert_eq!(split.finish(), digest(long));
        // `git hash-object` of "hello\n".
        assert_eq!(
            digest(b"blob 6\0hello\n"),
            "ce013625030ba8dba906f756967f9e9ca394464a"
        );
    }
}
