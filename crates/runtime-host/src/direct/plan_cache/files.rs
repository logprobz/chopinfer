//! Descriptor-relative cache access on Linux; no cache on other platforms.

// SAFETY-REVIEWED: the only foreign calls read the effective uid or lock a live fd.
#![allow(unsafe_code)]

use super::WeightPlanV1;
use std::fs::{self, DirBuilder, File, Metadata, OpenOptions};
use std::io::{self, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};

const MAGIC: &[u8; 8] = b"CHOPWPC1";
const HEADER_BYTES: u64 = 8 + 64 + 8 + 64;
const ENTRY_BYTES: u64 = 256 << 20;
const CACHE_BYTES: u64 = 1 << 30;
const CACHE_ENTRIES: usize = 64;

pub(in crate::direct) struct Cache {
    directory: File,
    uid: u32,
}

impl Cache {
    pub fn selected() -> Option<Self> {
        let path = match std::env::var_os("CHOPINFER_WEIGHT_PLAN_CACHE") {
            Some(value) if value == "0" || value.is_empty() => return None,
            Some(value) => PathBuf::from(value),
            None => runtime_bundle::preparation::default_cache_root()
                .ok()?
                .join("weight-plans-v1"),
        };
        Self::open(&path).ok()
    }

    pub(super) fn open(path: &Path) -> io::Result<Self> {
        if !path.is_absolute() || path == Path::new("/") {
            return Err(io::Error::other("cache needs an absolute directory"));
        }
        let mut directory = File::open("/")?;
        // Walk from open directories. Refuse symlinks at every component,
        // including an ancestor swapped while the path is being opened.
        for component in path.components() {
            let name = match component {
                Component::RootDir => continue,
                Component::Normal(name) => name,
                _ => return Err(io::Error::other("cache path contains a parent component")),
            };
            let child = fd_path(&directory).join(name);
            let mut options = OpenOptions::new();
            options
                .read(true)
                .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW);
            directory = match options.open(&child) {
                Ok(file) => file,
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    match DirBuilder::new().mode(0o700).create(&child) {
                        Ok(()) => (),
                        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => (),
                        Err(error) => return Err(error),
                    }
                    options.open(&child)?
                }
                Err(error) => return Err(error),
            };
        }
        // SAFETY-REVIEWED: geteuid takes no pointers, reads the process uid and cannot fail.
        let uid = unsafe { libc::geteuid() };
        let metadata = directory.metadata()?;
        if metadata.uid() != uid || metadata.mode() & 0o077 != 0 {
            return Err(io::Error::other("cache directory is not private"));
        }
        Ok(Self { directory, uid })
    }

    pub fn read(&self, key: &str) -> Option<WeightPlanV1> {
        self.read_checked(key).ok()
    }

    fn read_checked(&self, key: &str) -> io::Result<WeightPlanV1> {
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(self.path(key))?;
        let metadata = file.metadata()?;
        if !self.private_file(&metadata) || !(HEADER_BYTES..=ENTRY_BYTES).contains(&metadata.len())
        {
            return Err(io::Error::other("invalid cache file"));
        }
        let mut header = [0; HEADER_BYTES as usize];
        file.read_exact(&mut header)?;
        let length = u64::from_le_bytes(header[72..80].try_into().unwrap());
        if &header[..8] != MAGIC
            || &header[8..72] != key.as_bytes()
            || length != metadata.len() - HEADER_BYTES
        {
            return Err(io::Error::other("invalid cache header"));
        }
        let mut bytes = vec![0; length as usize];
        file.read_exact(&mut bytes)?;
        if runtime_weights::sha256_bytes(&bytes).as_bytes() != &header[80..] {
            return Err(io::Error::other("cache digest mismatch"));
        }
        WeightPlanV1::decode(&bytes).map_err(io::Error::other)
    }

    pub fn write_plan(&self, key: &str, plan: &WeightPlanV1) {
        if super::encoded_bound(plan) as u64 <= ENTRY_BYTES - HEADER_BYTES {
            self.write(key, &plan.encode());
        }
    }

    pub fn write(&self, key: &str, bytes: &[u8]) {
        let _ = self.publish(key, bytes, CACHE_BYTES, CACHE_ENTRIES);
    }

    pub(super) fn publish(
        &self,
        key: &str,
        bytes: &[u8],
        budget: u64,
        entries: usize,
    ) -> io::Result<()> {
        let length = HEADER_BYTES + bytes.len() as u64;
        if length > ENTRY_BYTES || length > budget || entries == 0 {
            return Err(io::Error::other("cache entry exceeds budget"));
        }
        // Every process locks its own open directory description. Reads need
        // no lock because publication replaces a complete file atomically.
        // SAFETY-REVIEWED: flock receives only integers and a descriptor kept open by self.
        if unsafe { libc::flock(self.directory.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let _lock = Lock(&self.directory);
        let pending = fd_path(&self.directory).join(".pending");
        if let Ok(metadata) = fs::symlink_metadata(&pending) {
            if !self.private_file(&metadata) {
                return Err(io::Error::other("invalid pending cache file"));
            }
            fs::remove_file(&pending)?;
        }
        let mut files = Vec::new();
        for (index, entry) in fs::read_dir(fd_path(&self.directory))?.enumerate() {
            if index >= CACHE_ENTRIES * 2 {
                return Err(io::Error::other("cache has too many files"));
            }
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str().and_then(|name| name.strip_suffix(".plan")) else {
                return Err(io::Error::other("unexpected cache file"));
            };
            let metadata = fs::symlink_metadata(entry.path())?;
            if name.len() != 64
                || !name.bytes().all(|c| c.is_ascii_hexdigit())
                || !self.private_file(&metadata)
            {
                return Err(io::Error::other("invalid cache file"));
            }
            files.push((metadata.modified()?, metadata.len(), entry.path()));
        }
        files.sort();
        let mut total: u64 = files
            .iter()
            .fold(0_u64, |total, file| total.saturating_add(file.1));
        let mut count = files.len();
        for (_, size, path) in files {
            if total.saturating_add(length) <= budget && count < entries {
                break;
            }
            fs::remove_file(path)?;
            total = total.saturating_sub(size);
            count -= 1;
        }
        let result = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW)
                .open(&pending)?;
            file.write_all(MAGIC)?;
            file.write_all(key.as_bytes())?;
            file.write_all(&(bytes.len() as u64).to_le_bytes())?;
            file.write_all(runtime_weights::sha256_bytes(bytes).as_bytes())?;
            file.write_all(bytes)?;
            file.sync_all()?;
            fs::rename(&pending, self.path(key))
        })();
        if result.is_err() {
            let _ = fs::remove_file(&pending);
        }
        result
    }

    pub(super) fn path(&self, key: &str) -> PathBuf {
        fd_path(&self.directory).join(format!("{key}.plan"))
    }

    fn private_file(&self, metadata: &Metadata) -> bool {
        metadata.is_file()
            && metadata.uid() == self.uid
            && metadata.mode() & 0o077 == 0
            && metadata.nlink() == 1
    }
}

fn fd_path(directory: &File) -> PathBuf {
    PathBuf::from(format!("/proc/self/fd/{}", directory.as_raw_fd()))
}

struct Lock<'a>(&'a File);
impl Drop for Lock<'_> {
    fn drop(&mut self) {
        // SAFETY-REVIEWED: flock receives a descriptor that the borrowed file keeps open.
        unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN) };
    }
}
