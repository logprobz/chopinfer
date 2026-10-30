//! Cubins a build script wrote, read when a test first loads them instead of
//! embedded in every binary ([`crate::Cubins::files`]).

use crate::LabResult;
use std::collections::BTreeMap;
use std::io::Read;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};
use std::sync::{Mutex, OnceLock, PoisonError};

/// Replaces the directory of every [`crate::Cubins::files`] table, for cubins
/// staged away from the build that wrote them.
const CUBIN_DIR_VAR: &str = "CHOPINFER_KERNEL_CUBIN_DIR";

/// The largest cubin a table may name; the largest lab cubin is about 4 MB.
const MAX_CUBIN_BYTES: u64 = 64 << 20;

/// Every verified read of this process, by resolved path and expected
/// SHA-256, so a descriptor that names a cached path with another digest
/// still has its own digest checked.
static LOADED: Mutex<BTreeMap<(PathBuf, &'static str), &'static [u8]>> =
    Mutex::new(BTreeMap::new());

/// A cubin a build script wrote: its spec name, its path relative to its
/// table's directory, and the SHA-256 its build recorded and checked against
/// the kernel lock.
#[derive(Debug)]
pub struct CubinFile {
    pub name: &'static str,
    pub path: &'static str,
    pub sha256: &'static str,
}

impl CubinFile {
    /// The bytes of this cubin under `dir`: a regular file, opened without
    /// following a final symlink, of at most 64 MiB, whose SHA-256 is the
    /// recorded one.
    pub fn read(&self, dir: &Path) -> LabResult<Vec<u8>> {
        let path = self.path_in(dir)?;
        let fail = |error: std::io::Error| {
            format!("cubin {}: read {}: {error}", self.name, path.display())
        };
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&path)
            .map_err(fail)?;
        if !file.metadata().map_err(fail)?.is_file() {
            return Err(format!(
                "cubin {}: {} is not a regular file",
                self.name,
                path.display()
            ));
        }
        let mut bytes = Vec::new();
        file.take(MAX_CUBIN_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(fail)?;
        if bytes.len() as u64 > MAX_CUBIN_BYTES {
            return Err(format!(
                "cubin {}: {} exceeds {MAX_CUBIN_BYTES} bytes",
                self.name,
                path.display()
            ));
        }
        let sha256 = kernel_build::digest(&bytes);
        if sha256 != self.sha256 {
            return Err(format!(
                "cubin {}: {} has SHA-256 {sha256}, but this binary was built with {}; \
                 rebuild the binary, or restage the directory {CUBIN_DIR_VAR} names when it is set",
                self.name,
                path.display(),
                self.sha256
            ));
        }
        Ok(bytes)
    }

    /// The cubin's path under `dir`, refusing a path that is empty, absolute
    /// or leaves `dir`.
    fn path_in(&self, dir: &Path) -> LabResult<PathBuf> {
        let relative = Path::new(self.path);
        if relative.as_os_str().is_empty()
            || !relative
                .components()
                .all(|part| matches!(part, Component::Normal(_)))
        {
            return Err(format!(
                "cubin {}: {:?} is not a relative path inside its directory",
                self.name, self.path
            ));
        }
        Ok(dir.join(relative))
    }
}

/// The bytes of `file`, from the table directory `dir` or the staged
/// directory `CHOPINFER_KERNEL_CUBIN_DIR` names, read and checked once per
/// process.
pub(crate) fn load(dir: &'static str, file: &'static CubinFile) -> LabResult<&'static [u8]> {
    load_from(directory(Path::new(dir), staged()), file)
}

/// `staged` when set, otherwise the table's own directory `dir`.
fn directory<'a>(dir: &'a Path, staged: Option<&'a Path>) -> &'a Path {
    staged.unwrap_or(dir)
}

/// The directory `CHOPINFER_KERNEL_CUBIN_DIR` names, read once; an empty
/// value counts as unset.
fn staged() -> Option<&'static Path> {
    static STAGED: OnceLock<Option<PathBuf>> = OnceLock::new();
    STAGED
        .get_or_init(|| {
            std::env::var_os(CUBIN_DIR_VAR)
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
        })
        .as_deref()
}

fn load_from(dir: &Path, file: &'static CubinFile) -> LabResult<&'static [u8]> {
    let key = (file.path_in(dir)?, file.sha256);
    let mut loaded = LOADED.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(bytes) = loaded.get(&key) {
        return Ok(bytes);
    }
    let bytes: &'static [u8] = Box::leak(file.read(dir)?.into_boxed_slice());
    loaded.insert(key, bytes);
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::{CubinFile, directory, load_from};
    use crate::Cubins;
    use std::path::{Path, PathBuf};

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cubin-file-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cubin(name: &str, path: &str, sha256: &str) -> &'static CubinFile {
        Box::leak(Box::new(CubinFile {
            name: Box::leak(name.into()),
            path: Box::leak(path.into()),
            sha256: Box::leak(sha256.into()),
        }))
    }

    #[test]
    fn a_file_is_read_once_and_each_descriptor_checks_its_digest() {
        let dir = scratch("cache");
        std::fs::create_dir(dir.join("families")).unwrap();
        std::fs::write(dir.join("families/a.cubin"), b"cubin bytes").unwrap();
        let digest = kernel_build::digest(b"cubin bytes");
        let file = cubin("a", "families/a.cubin", &digest);
        let first = load_from(&dir, file).unwrap();
        assert_eq!(first, b"cubin bytes");
        assert!(std::ptr::eq(first, load_from(&dir, file).unwrap()));
        // A warmed cache does not vouch for another descriptor of the path.
        let other = cubin("a", "families/a.cubin", &kernel_build::digest(b"other"));
        let error = load_from(&dir, other).unwrap_err();
        assert!(
            error.contains("cubin a:") && error.contains("SHA-256"),
            "{error}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn reads_refuse_symlinks_missing_files_and_paths_outside_the_directory() {
        let dir = scratch("refusals");
        std::fs::write(dir.join("real.cubin"), b"bytes").unwrap();
        std::os::unix::fs::symlink(dir.join("real.cubin"), dir.join("link.cubin")).unwrap();
        let digest = kernel_build::digest(b"bytes");
        assert_eq!(
            cubin("real", "real.cubin", &digest).read(&dir).unwrap(),
            b"bytes"
        );
        for (name, path) in [
            ("link", "link.cubin"),
            ("missing", "missing.cubin"),
            ("parent", "../real.cubin"),
            ("absolute", "/real.cubin"),
            ("empty", ""),
        ] {
            let error = cubin(name, path, &digest).read(&dir).unwrap_err();
            assert!(error.starts_with(&format!("cubin {name}:")), "{error}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_staged_directory_replaces_the_table_directory() {
        let (built, staged) = (Path::new("built"), Path::new("staged"));
        assert_eq!(directory(built, Some(staged)), staged);
        assert_eq!(directory(built, None), built);
    }

    #[test]
    fn unknown_names_fail_in_either_form() {
        static FILES: &[CubinFile] = &[CubinFile {
            name: "a",
            path: "a.cubin",
            sha256: "",
        }];
        let files = Cubins::files("unused", FILES, &[("a", &["chop_a_v1"])]);
        assert_eq!(files.bytes("b").unwrap_err(), "unknown cubin b");
        assert_eq!(files.holding("chop_a_v1"), Some("a"));
        let embedded = Cubins::new(&[("a", b"bytes")], &[("a", &["chop_a_v1"])]);
        assert_eq!(embedded.bytes("a").unwrap(), b"bytes");
        assert_eq!(embedded.bytes("b").unwrap_err(), "unknown cubin b");
    }
}
