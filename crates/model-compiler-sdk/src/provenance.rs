//! Checkpoint files authenticated against a pinned source lock, and the
//! verified reads a diagnostic's identity document records.
use crate::files::{Result, SourceStamp, stamp};
use runtime_bundle::checkpoint_binding::FileDigest;
use runtime_weights::checkpoint::CheckpointFile;
use runtime_weights::source_file::SourceFile;
use std::collections::BTreeMap;
use std::fs::File;

/// A retained source descriptor that notices when its file is replaced.
pub trait Retained: std::ops::Deref<Target = File> {
    fn verify(&self) -> std::io::Result<()>;
}

impl Retained for SourceFile {
    fn verify(&self) -> std::io::Result<()> {
        SourceFile::verify(self)
    }
}

impl Retained for CheckpointFile {
    fn verify(&self) -> std::io::Result<()> {
        CheckpointFile::verify(self)
    }
}

/// One checkpoint revision's pinned files: each one's length and SHA-256.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceLock {
    pins: BTreeMap<String, FileDigest>,
}

impl SourceLock {
    pub fn new(pins: BTreeMap<String, FileDigest>) -> Self {
        Self { pins }
    }

    /// A lock document of one revision: file names to their digests.
    pub fn parse(document: &str) -> Result<Self> {
        Ok(Self::new(serde_json::from_str(document)?))
    }

    pub fn pin(&self, name: &str) -> Option<&FileDigest> {
        self.pins.get(name)
    }

    /// The SHA-256 of the lock's JSON, which package bindings and
    /// diagnostic identities name.
    pub fn sha256(&self) -> String {
        runtime_weights::sha256_bytes(
            &serde_json::to_vec(&self.pins).expect("source pins serialize"),
        )
    }

    /// Checks that `source` holds the pinned bytes of `name`, and is the same
    /// file before and after it is hashed.
    pub fn authenticate(&self, name: &str, source: &impl Retained) -> Result<()> {
        let pin = self
            .pin(name)
            .ok_or_else(|| format!("source {name} is not pinned for selected checkpoint"))?;
        source.verify()?;
        if source.metadata()?.len() != pin.byte_length
            || runtime_weights::hash_range(source, 0, pin.byte_length)? != pin.sha256
        {
            return Err(format!(
                "source {name} differs from selected checkpoint revision; pinned expectations \
                 do not apply"
            )
            .into());
        }
        source.verify()?;
        Ok(())
    }
}

/// The pinned files a diagnostic read and the digest of every range it read.
pub struct SourceLedger {
    lock: SourceLock,
    /// Each authenticated name and the file (device, inode, length and
    /// times) whose bytes were found pinned.
    authenticated: BTreeMap<String, SourceStamp>,
    ranges: BTreeMap<String, String>,
}

impl SourceLedger {
    pub fn new(lock: SourceLock) -> Self {
        Self {
            lock,
            authenticated: BTreeMap::new(),
            ranges: BTreeMap::new(),
        }
    }

    pub fn lock(&self) -> &SourceLock {
        &self.lock
    }

    /// Authenticates `source` as `name` and records it, unless this very
    /// file was already authenticated as `name`: another descriptor under the
    /// name is hashed against the pin again.
    pub fn authenticate(&mut self, name: &str, source: &impl Retained) -> Result<()> {
        let file = stamp(source)?;
        if self.authenticated.get(name) != Some(&file) {
            self.lock.authenticate(name, source)?;
            self.authenticated.insert(name.into(), file);
        }
        Ok(())
    }

    /// Reads `length` bytes at `offset` of the pinned file `name` and
    /// records their digest as `range`.
    pub fn read(
        &mut self,
        name: &str,
        source: &impl Retained,
        range: String,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>> {
        source.verify()?;
        self.authenticate(name, source)?;
        let mut bytes = vec![0; length];
        runtime_weights::read_exact_at(source, &mut bytes, offset)?;
        source.verify()?;
        self.ranges
            .insert(range, runtime_weights::sha256_bytes(&bytes));
        Ok(bytes)
    }

    /// The identity of a diagnostic that read the pinned checkpoint
    /// `repository` at `revision` directly, preparing no weights.
    pub fn identity(&self, repository: &str, revision: &str) -> serde_json::Value {
        serde_json::json!({
            "checkpoint": {
                "repository": repository,
                "revision": revision,
                "source_lock_sha256": self.lock.sha256(),
            },
            "prepared_weights": {"kind": "none"},
            "authenticated_files": self.authenticated.keys().collect::<Vec<_>>(),
            "source_ranges": self.ranges,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only the pinned bytes authenticate, a file is hashed once, and every
    /// read range is recorded with its digest.
    #[test]
    fn reads_record_ranges_of_authenticated_pinned_files() {
        let root = std::env::temp_dir().join(format!("sdk-provenance-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("weights"), [1u8, 2, 3, 4]).unwrap();
        let source = CheckpointFile::open(&root, "weights").unwrap();
        let pin = |sha256: String| FileDigest {
            byte_length: 4,
            sha256,
        };
        let lock = SourceLock::new(BTreeMap::from([(
            "weights".into(),
            pin(runtime_weights::sha256_bytes(&[1, 2, 3, 4])),
        )]));
        let mut ledger = SourceLedger::new(lock.clone());
        assert_eq!(
            ledger
                .read("weights", &source, "t:1:2".into(), 1, 2)
                .unwrap(),
            [2, 3]
        );
        assert!(
            ledger
                .read("other", &source, "u".into(), 0, 1)
                .unwrap_err()
                .to_string()
                .contains("not pinned")
        );
        let identity = ledger.identity("org/model", "abc");
        assert_eq!(identity["checkpoint"]["source_lock_sha256"], lock.sha256());
        assert_eq!(
            identity["authenticated_files"],
            serde_json::json!(["weights"])
        );
        assert_eq!(
            identity["source_ranges"]["t:1:2"],
            runtime_weights::sha256_bytes(&[2, 3])
        );
        // Another file under the authenticated name is checked against the
        // pin before any of its bytes are read.
        let other = root.join("other");
        std::fs::create_dir(&other).unwrap();
        std::fs::write(other.join("weights"), [9u8, 9, 9, 9]).unwrap();
        let substitute = CheckpointFile::open(&other, "weights").unwrap();
        assert!(
            ledger
                .read("weights", &substitute, "t:0:4".into(), 0, 4)
                .unwrap_err()
                .to_string()
                .contains("differs from selected checkpoint")
        );
        assert!(
            ledger.identity("org/model", "abc")["source_ranges"]
                .get("t:0:4")
                .is_none()
        );
        // Another descriptor of the authenticated file reads as it.
        let reopened = CheckpointFile::open(&root, "weights").unwrap();
        assert_eq!(
            ledger
                .read("weights", &reopened, "t:0:1".into(), 0, 1)
                .unwrap(),
            [1]
        );
        let foreign = SourceLock::new(BTreeMap::from([("weights".into(), pin("0".repeat(64)))]));
        assert!(
            foreign
                .authenticate("weights", &source)
                .unwrap_err()
                .to_string()
                .contains("differs from selected checkpoint")
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
