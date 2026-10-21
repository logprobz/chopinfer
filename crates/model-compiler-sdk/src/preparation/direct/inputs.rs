//! Bind emission to the checkpoint observed before its cache key is hashed.
use crate::files::Result;
use runtime_weights::checkpoint::{CheckpointFile, metadata_files};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(PartialEq, Eq)]
struct DirectoryStamp {
    modified: std::time::SystemTime,
    #[cfg(unix)]
    identity: (u64, u64, i64, i64),
}

impl DirectoryStamp {
    fn of(path: &Path) -> Result<Self> {
        let metadata = fs::metadata(path)?;
        if !metadata.is_dir() {
            return Err("checkpoint input directory changed type".into());
        }
        Ok(Self {
            modified: metadata.modified()?,
            #[cfg(unix)]
            identity: {
                use std::os::unix::fs::MetadataExt;
                (
                    metadata.dev(),
                    metadata.ino(),
                    metadata.ctime(),
                    metadata.ctime_nsec(),
                )
            },
        })
    }
}

/// Retained descriptors catch timestamped writes and replacement. Directory
/// stamps also catch transient optional inputs and restoration of an inode.
/// These stamps alone do not make bytes immutable, so snapshot providers
/// consume the owned bytes below instead of reopening their metadata.
pub(super) struct CheckpointInputs {
    files: Vec<CheckpointFile>,
    directories: Vec<(PathBuf, DirectoryStamp)>,
}

impl CheckpointInputs {
    pub fn capture(root: &Path) -> Result<Self> {
        let mut directories = vec![(root.to_path_buf(), DirectoryStamp::of(root)?)];
        for entry in fs::read_dir(root)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() && !entry.file_name().to_string_lossy().starts_with('.')
            {
                directories.push((entry.path(), DirectoryStamp::of(&entry.path())?));
            }
        }
        let mut names = metadata_files(root)?;
        names.extend(super::safetensors_names(root)?);
        let files = names
            .into_iter()
            .map(|(directory, name)| {
                let parent = directory.map_or_else(|| root.to_path_buf(), |dir| root.join(dir));
                CheckpointFile::open(&parent, &name)
            })
            .collect::<std::io::Result<Vec<_>>>()?;
        let captured = Self { files, directories };
        captured.verify()?;
        Ok(captured)
    }

    pub fn verify(&self) -> Result<()> {
        for file in &self.files {
            file.verify().map_err(|error| {
                format!("checkpoint inputs changed during preparation: {error}")
            })?;
        }
        for (path, stamp) in &self.directories {
            if DirectoryStamp::of(path)? != *stamp {
                return Err("checkpoint input directory changed during preparation".into());
            }
        }
        Ok(())
    }
}

/// Owned metadata and headers used by one provider operation. Payload bytes
/// are never read. The total retained metadata is bounded at 512 MiB.
pub struct CheckpointSnapshot {
    guard: CheckpointInputs,
    pub(crate) headers: std::collections::BTreeMap<String, (u64, Vec<u8>)>,
    pub(crate) documents: std::collections::BTreeMap<String, Vec<u8>>,
    pub(crate) identity: String,
    pub(crate) metadata_identity: String,
    regular_documents: std::collections::BTreeSet<String>,
}

impl CheckpointSnapshot {
    pub fn capture(root: &Path) -> Result<Self> {
        Self::capture_selected(root, None)
    }

    /// Retain only the documents emission reads. Other metadata still enters
    /// the existing cache identity, streamed without retaining its contents.
    pub fn capture_documents(root: &Path, names: &[&str]) -> Result<Self> {
        Self::capture_selected(root, Some(names))
    }

    fn capture_selected(root: &Path, names: Option<&[&str]>) -> Result<Self> {
        use sha2::{Digest, Sha256};
        use std::io::Read;
        use std::os::unix::fs::FileExt;
        let guard = CheckpointInputs::capture(root)?;
        let mut remaining = 512_u64 << 20;
        let mut headers = std::collections::BTreeMap::new();
        for (directory, name) in super::safetensors_names(root)? {
            let parent = directory
                .as_ref()
                .map_or_else(|| root.to_path_buf(), |dir| root.join(dir));
            let full = directory.map_or_else(|| name.clone(), |dir| format!("{dir}/{name}"));
            let file = CheckpointFile::open(&parent, &name)?;
            let length = file.metadata()?.len();
            let mut prefix = [0; 8];
            file.read_exact_at(&mut prefix, 0)?;
            let bytes = u64::from_le_bytes(prefix);
            if bytes == 0
                || bytes > 256 << 20
                || bytes > remaining
                || bytes > length.saturating_sub(8)
            {
                return Err("checkpoint header exceeds snapshot bounds".into());
            }
            let mut header = vec![0; bytes as usize];
            file.read_exact_at(&mut header, 8)?;
            file.verify()?;
            remaining -= bytes;
            headers.insert(full, (length, header));
        }
        if headers.is_empty() {
            return Err("checkpoint snapshot has no safetensors headers".into());
        }
        let identity = runtime_weights::checkpoint::header_identity(
            headers
                .iter()
                .map(|(name, (length, header))| (name.as_str(), *length, header.as_slice())),
        );
        let mut documents = std::collections::BTreeMap::new();
        let mut regular_documents = std::collections::BTreeSet::new();
        let regular_root = fs::symlink_metadata(root)?.is_dir();
        let mut metadata = Sha256::new();
        metadata.update(b"chopinfer.checkpoint-metadata.v1");
        for (directory, name) in metadata_files(root)? {
            let parent = directory
                .as_ref()
                .map_or_else(|| root.to_path_buf(), |dir| root.join(dir));
            let full = directory.map_or_else(|| name.clone(), |dir| format!("{dir}/{name}"));
            let mut file = CheckpointFile::open(&parent, &name)?;
            let (length, hash) = if names.is_none_or(|names| names.contains(&full.as_str())) {
                let bytes = file.read_bounded(remaining.min(64 << 20))?;
                if regular_root && file.link_target().is_none() {
                    regular_documents.insert(full.clone());
                }
                remaining -= bytes.len() as u64;
                let measured = (bytes.len() as u64, Sha256::digest(&bytes));
                documents.insert(full.clone(), bytes);
                measured
            } else {
                let mut content = Sha256::new();
                let mut buffer = vec![0; 1 << 20];
                let mut length = 0;
                loop {
                    let count = file.read(&mut buffer)?;
                    if count == 0 {
                        break;
                    }
                    length += count as u64;
                    content.update(&buffer[..count]);
                }
                file.verify()?;
                (length, content.finalize())
            };
            metadata.update((full.len() as u64).to_le_bytes());
            metadata.update(full.as_bytes());
            metadata.update(length.to_le_bytes());
            metadata.update(hash);
        }
        for name in names.into_iter().flatten() {
            if !documents.contains_key(*name) {
                return Err(format!("missing checkpoint document {name}").into());
            }
        }
        guard.verify()?;
        let hex = |bytes: &[u8]| bytes.iter().map(|b| format!("{b:02x}")).collect();
        Ok(Self {
            guard,
            headers,
            documents,
            identity: hex(&identity),
            metadata_identity: hex(&metadata.finalize()),
            regular_documents,
        })
    }

    pub fn headers(&self) -> &std::collections::BTreeMap<String, (u64, Vec<u8>)> {
        &self.headers
    }

    pub fn document(&self, name: &str) -> Result<&[u8]> {
        self.documents
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| format!("missing checkpoint document {name}").into())
    }

    /// For inputs whose own contract refuses links, even confined snapshot
    /// links that ordinary checkpoint documents permit.
    pub fn regular_document(&self, name: &str) -> Result<&[u8]> {
        if !self.regular_documents.contains(name) {
            return Err(
                format!("checkpoint document {name} must be a regular non-symlink file").into(),
            );
        }
        self.document(name)
    }

    pub fn verify(&self) -> Result<()> {
        self.guard.verify()
    }

    pub(crate) fn verify_contents(&self, root: &Path) -> Result<()> {
        self.verify()?;
        let metadata: String = runtime_weights::checkpoint::metadata_identity(root)?
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if metadata != self.metadata_identity || super::checkpoint_identity(root)? != self.identity
        {
            return Err("checkpoint snapshot contents changed during preparation".into());
        }
        Ok(())
    }
}
