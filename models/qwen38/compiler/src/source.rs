//! Family checkpoint reads use the shared confined local source reader.
use crate::Result;
use runtime_weights::checkpoint::CheckpointFile;
use std::path::Path;

pub(crate) fn open(root: &Path, name: &str) -> Result<CheckpointFile> {
    Ok(CheckpointFile::open(root, name)?)
}

pub(crate) fn read(root: &Path, name: &str, limit: u64) -> Result<Vec<u8>> {
    Ok(open(root, name)?.read_bounded(limit)?)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    #[test]
    fn metadata_reads_refuse_unconfined_links_paths_and_oversized_files() {
        use std::os::unix::fs::symlink;
        let root = std::env::temp_dir().join(format!("qwen38-source-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("regular"), b"data").unwrap();
        symlink("regular", root.join("link")).unwrap();
        assert_eq!(read(&root, "regular", 4).unwrap(), b"data");
        assert!(read(&root, "regular", 3).is_err());
        assert!(open(&root, "link").is_err());
        assert!(open(&root, "../regular").is_err());
        assert!(open(&root, "/etc/passwd").is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
