//! Confined local checkpoint inputs, including standard snapshot-to-blob links.
use crate::source_file::SourceFile;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};

fn resolve(root: &Path, name: &str) -> io::Result<(PathBuf, Option<PathBuf>)> {
    if name.contains(['/', '\\'])
        || Path::new(name).components().count() != 1
        || !matches!(
            Path::new(name).components().next(),
            Some(Component::Normal(_))
        )
    {
        return Err(io::Error::other(
            "checkpoint filename must be one normal component",
        ));
    }
    let path = root.join(name);
    if !fs::symlink_metadata(&path)?.file_type().is_symlink() {
        return Ok((path, None));
    }
    let (snapshot, parents) =
        if root.parent().and_then(Path::file_name) == Some("snapshots".as_ref()) {
            (root, 2)
        } else if root
            .parent()
            .and_then(Path::parent)
            .and_then(Path::file_name)
            == Some("snapshots".as_ref())
        {
            (root.parent().expect("snapshot parent"), 3)
        } else {
            return Err(io::Error::other(
                "links are permitted only in a confined HF snapshot",
            ));
        };
    let target = fs::read_link(&path)?;
    let parts: Vec<_> = target.components().collect();
    if parts.len() != parents + 2
        || parts[..parents]
            .iter()
            .any(|part| *part != Component::ParentDir)
    {
        return Err(io::Error::other(
            "snapshot link must name its cache's blobs file",
        ));
    }
    let [Component::Normal(blobs), Component::Normal(blob)] = &parts[parents..] else {
        return Err(io::Error::other("snapshot link must name a blobs file"));
    };
    if *blobs != "blobs" {
        return Err(io::Error::other("snapshot link escapes blobs"));
    }
    let directory = snapshot
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| io::Error::other("missing cache root"))?
        .join("blobs");
    if fs::canonicalize(&directory)? != directory {
        return Err(io::Error::other(
            "snapshot blobs directory must not be a symlink",
        ));
    }
    Ok((directory.join(blob), Some(target)))
}

/// Retains both the opened inode and the directory/link mapping used to find it.
#[derive(Clone, Debug)]
pub struct CheckpointFile {
    source: SourceFile,
    root: PathBuf,
    canonical_root: PathBuf,
    name: String,
    resolved: PathBuf,
    link: Option<PathBuf>,
}

impl CheckpointFile {
    pub fn open(root: &Path, name: &str) -> io::Result<Self> {
        let canonical_root = fs::canonicalize(root)?;
        let (resolved, link) = resolve(&canonical_root, name)?;
        let input = Self {
            source: SourceFile::open(&resolved)?,
            root: root.into(),
            canonical_root,
            name: name.into(),
            resolved,
            link,
        };
        input.verify()?;
        Ok(input)
    }

    pub fn verify(&self) -> io::Result<()> {
        self.source.verify()?;
        if fs::canonicalize(&self.root)? != self.canonical_root
            || resolve(&self.canonical_root, &self.name)?
                != (self.resolved.clone(), self.link.clone())
        {
            return Err(io::Error::other("checkpoint directory or link changed"));
        }
        Ok(())
    }

    pub fn resolved_path(&self) -> &Path {
        &self.resolved
    }
    pub fn link_target(&self) -> Option<&Path> {
        self.link.as_deref()
    }

    pub fn read_bounded(&self, limit: u64) -> io::Result<Vec<u8>> {
        self.verify()?;
        if self.metadata()?.len() > limit {
            return Err(io::Error::other("checkpoint metadata exceeds limit"));
        }
        let mut bytes = Vec::new();
        (&**self).take(limit + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > limit {
            return Err(io::Error::other("checkpoint metadata grew beyond limit"));
        }
        self.verify()?;
        Ok(bytes)
    }
}

/// The identity of a safetensors checkpoint for direct loading: SHA-256 over
/// each file's name, length and JSON header, in name order. It binds every
/// tensor's name, dtype, shape and offsets, not the tensor bytes.
#[must_use]
pub fn header_identity<'a>(files: impl IntoIterator<Item = (&'a str, u64, &'a [u8])>) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut files: Vec<_> = files.into_iter().collect();
    files.sort_by(|left, right| left.0.cmp(right.0));
    let mut digest = Sha256::new();
    digest.update(b"chopinfer.checkpoint-headers.v1");
    for (name, length, header) in files {
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update(length.to_le_bytes());
        digest.update((header.len() as u64).to_le_bytes());
        digest.update(header);
    }
    digest.finalize().into()
}

/// Every file but the safetensors shards, in `root` and its immediate
/// subdirectories other than hidden ones, named `dir/file` and sorted: the
/// configuration, index, tokenizer and other metadata a family reads.
pub fn metadata_files(root: &Path) -> io::Result<Vec<(Option<String>, String)>> {
    let entries = |directory: &Path| -> io::Result<Vec<(String, bool)>> {
        fs::read_dir(directory)?
            .map(|entry| {
                let entry = entry?;
                let name = entry
                    .file_name()
                    .into_string()
                    .map_err(|_| io::Error::other("checkpoint name is not UTF-8"))?;
                Ok((name, entry.file_type()?.is_dir()))
            })
            .collect()
    };
    let mut names = Vec::new();
    for (name, directory) in entries(root)? {
        if !directory {
            names.push((None, name));
        } else if !name.starts_with('.') {
            for (inner, nested) in entries(&root.join(&name))? {
                if !nested {
                    names.push((Some(name.clone()), inner));
                }
            }
        }
    }
    names.retain(|(_, name)| !name.ends_with(".safetensors"));
    names.sort_by_key(|(directory, name)| match directory {
        Some(directory) => format!("{directory}/{name}"),
        None => name.clone(),
    });
    Ok(names)
}

/// The identity of a checkpoint's metadata: SHA-256 over each
/// [`metadata_files`] entry's name, length and content SHA-256, in name
/// order. With [`header_identity`] it binds everything but tensor payloads.
pub fn metadata_identity(root: &Path) -> io::Result<[u8; 32]> {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    digest.update(b"chopinfer.checkpoint-metadata.v1");
    for (directory, name) in metadata_files(root)? {
        let (parent, full) = match &directory {
            Some(directory) => (root.join(directory), format!("{directory}/{name}")),
            None => (root.to_path_buf(), name.clone()),
        };
        let mut file = CheckpointFile::open(&parent, &name)?;
        let mut content = Sha256::new();
        let mut buffer = vec![0_u8; 1 << 20];
        let mut length = 0_u64;
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            content.update(&buffer[..count]);
            length += count as u64;
        }
        file.verify()?;
        digest.update((full.len() as u64).to_le_bytes());
        digest.update(full.as_bytes());
        digest.update(length.to_le_bytes());
        digest.update(content.finalize());
    }
    Ok(digest.finalize().into())
}

impl std::ops::Deref for CheckpointFile {
    type Target = File;
    fn deref(&self) -> &File {
        &self.source
    }
}

impl Read for CheckpointFile {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        (&**self).read(bytes)
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;
    #[test]
    fn arbitrary_revision_retains_confined_mapping() {
        let root = std::env::temp_dir().join(format!("checkpoint-links-{}", std::process::id()));
        fs::create_dir_all(root.join("snapshots/another-revision")).unwrap();
        fs::create_dir(root.join("blobs")).unwrap();
        fs::write(root.join("blobs/data"), b"asset").unwrap();
        let snapshot = root.join("snapshots/another-revision");
        symlink("../../blobs/data", snapshot.join("input")).unwrap();
        let input = CheckpointFile::open(&snapshot, "input").unwrap();
        assert_eq!(input.read_bounded(5).unwrap(), b"asset");
        assert!(CheckpointFile::open(&snapshot, "../input").is_err());
        fs::remove_file(snapshot.join("input")).unwrap();
        symlink("../../blobs/../data", snapshot.join("input")).unwrap();
        assert!(input.verify().is_err());
        assert!(CheckpointFile::open(&snapshot, "input").is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
