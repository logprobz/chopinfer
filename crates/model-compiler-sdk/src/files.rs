//! Small file helpers the SDK's emission and preparation share.
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

use serde::{Deserialize, Serialize};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Opens a regular file without following a symlink.
pub fn open_output(path: &Path, write: bool, create: bool) -> Result<File> {
    runtime_weights::open_regular_nofollow(
        path,
        OpenOptions::new()
            .read(true)
            .write(write)
            .create_new(create),
    )
    .map_err(|error| format!("open {}: {error}", path.display()).into())
}

/// A file's device, inode, length and times, to notice a change.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceStamp {
    dev: u64,
    ino: u64,
    bytes: u64,
    mtime: i64,
    mtime_ns: i64,
    ctime: i64,
    ctime_ns: i64,
}

pub fn stamp(file: &File) -> Result<SourceStamp> {
    let m = file.metadata()?;
    Ok(SourceStamp {
        dev: m.dev(),
        ino: m.ino(),
        bytes: m.len(),
        mtime: m.mtime(),
        mtime_ns: m.mtime_nsec(),
        ctime: m.ctime(),
        ctime_ns: m.ctime_nsec(),
    })
}

/// Writes through a synced temporary file and synchronizes its directory.
/// The temporary name is this writer's own (process and call), so
/// concurrent writers of one file never remove or rename each other's.
pub fn replace_file(out: &Path, name: &str, bytes: &[u8]) -> Result<()> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let call = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // A file of this name can only be left by an exited process whose id
    // this one reuses.
    let temp = out.join(format!("{name}.{}-{call}.partial", std::process::id()));
    if temp.try_exists()? {
        fs::remove_file(&temp)?;
    }
    let mut file = open_output(&temp, true, true)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(temp, out.join(name))?;
    File::open(out)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Concurrent writers of one file each replace it whole, and leave no
    /// temporary file behind.
    #[test]
    fn concurrent_replacements_of_one_file_all_succeed() {
        let directory = std::env::temp_dir().join(format!("replace-file-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        std::thread::scope(|scope| {
            for writer in 0..8 {
                let directory = &directory;
                scope.spawn(move || {
                    for round in 0..200 {
                        replace_file(
                            directory,
                            "record.json",
                            format!("{writer} {round}").as_bytes(),
                        )
                        .unwrap();
                    }
                });
            }
        });
        let names: Vec<_> = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(names, ["record.json"]);
        assert!(
            fs::read_to_string(directory.join("record.json"))
                .unwrap()
                .ends_with(" 199")
        );
        fs::remove_dir_all(&directory).unwrap();
    }
}
