//! Opt-in, bounded head operands for diagnostic GPU comparisons.
use std::collections::BTreeSet;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

pub(crate) struct HeadCapture {
    pub root: PathBuf,
    pub captured: BTreeSet<u32>,
    pub weights: Vec<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{DirBuilderExt, symlink};

    #[test]
    fn capture_requires_private_storage_and_an_explicit_marker() {
        let root = std::env::temp_dir().join(format!("dspark-head-capture-{}", std::process::id()));
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&root)
            .unwrap();
        let mut capture = HeadCapture::new(&root).unwrap();
        assert!(!capture.pending(40));
        let marker = root.join("enabled");
        let target = root.join("target");
        std::fs::write(&target, []).unwrap();
        symlink(&target, &marker).unwrap();
        assert!(!capture.pending(40));
        std::fs::remove_file(&marker).unwrap();
        std::fs::write(&marker, []).unwrap();
        for rows in 0..=161 {
            assert_eq!(capture.pending(rows), [40, 60, 80, 160].contains(&rows));
        }
        capture.captured.insert(40);
        assert!(!capture.pending(40));
        assert!(capture.pending(60));
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o750)).unwrap();
        assert!(HeadCapture::new(&root).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}

impl HeadCapture {
    pub fn from_env() -> Result<Option<Self>, String> {
        std::env::var_os("CHOPINFER_DSPARK_HEAD_CAPTURE")
            .map(|root| Self::new(Path::new(&root)))
            .transpose()
    }

    pub fn new(root: &Path) -> Result<Self, String> {
        let metadata = root.symlink_metadata().map_err(|e| e.to_string())?;
        if !root.is_absolute()
            || !metadata.file_type().is_dir()
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err(
                "DSpark head capture needs an existing private absolute directory, not a link"
                    .into(),
            );
        }
        let root = root.canonicalize().map_err(|e| e.to_string())?;
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .canonicalize()
            .map_err(|e| e.to_string())?;
        if root.starts_with(repository) {
            return Err("DSpark head captures must stay outside the repository".into());
        }
        Ok(Self {
            root,
            captured: BTreeSet::new(),
            weights: Vec::new(),
        })
    }

    pub fn pending(&self, rows: u32) -> bool {
        [40, 60, 80, 160].contains(&rows)
            && !self.captured.contains(&rows)
            && self
                .root
                .join("enabled")
                .symlink_metadata()
                .is_ok_and(|metadata| metadata.file_type().is_file())
    }
}
