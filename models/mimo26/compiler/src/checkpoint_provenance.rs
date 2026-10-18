//! Authenticates a checkpoint's files against the pinned source lock, for
//! package bindings and the oracle's verified tensor reads.
use crate::{
    MimoResult,
    catalog::{self, TensorRecord},
    checkpoint::Identity,
};
use model_compiler_sdk::provenance::SourceLedger;
use model_compiler_sdk::source_file::SourceFile;
use std::collections::BTreeMap;
use std::path::Path;

pub struct Verifier {
    identity: Identity,
    registry: crate::checkpoint::Registry,
    config: Option<SourceFile>,
    sources: BTreeMap<String, SourceFile>,
    tensors: BTreeMap<String, TensorRecord>,
    ledger: SourceLedger,
}

impl Verifier {
    pub fn new(root: &Path, identity: &Identity) -> MimoResult<Self> {
        Self::with_registry(root, identity, crate::checkpoint::registry())
    }

    pub fn with_registry(
        root: &Path,
        identity: &Identity,
        registry: &crate::checkpoint::Registry,
    ) -> MimoResult<Self> {
        let identity = registry.identity(registry.name(identity)?)?;
        let lock = registry.source_lock(identity)?;
        let config = catalog::open_source(&root.join("config.json"))?;
        // Authenticate index and metadata before interpreting their contents.
        for (name, _) in crate::checkpoint::METADATA {
            if name == "config.json" {
                lock.authenticate(name, &config)?;
            } else {
                lock.authenticate(name, &catalog::open_source(&root.join(name))?)?;
            }
        }
        // Refuse execution geometry before opening tensor shard headers.
        let matched = authenticated_config(&config, &lock)?;
        // A local checkpoint whose DFlash drafter does not match is served
        // without it (`catalog::Catalog::load_drafter_with_config`).
        if !identity.is_local() && root.join("dflash/config.json").try_exists()? {
            catalog::validate_dflash_config(root, matched.geometry()?)?;
        }
        let (sources, tensors) = catalog::load_indexed(root)?;
        // The index read above and the retained index must both have the pinned bytes.
        lock.authenticate(
            "model.safetensors.index.json",
            &sources["model.safetensors.index.json"],
        )?;
        Ok(Self {
            identity: identity.clone(),
            registry: registry.clone(),
            config: Some(config),
            sources,
            tensors,
            ledger: SourceLedger::new(lock),
        })
    }

    pub fn registry(&self) -> &crate::checkpoint::Registry {
        &self.registry
    }

    /// Parse only the bytes read from the retained, authenticated config handle.
    pub fn config(&self) -> MimoResult<crate::config::MiMoConfig> {
        authenticated_config(
            self.config.as_ref().ok_or("authenticated config absent")?,
            self.ledger.lock(),
        )
    }

    /// Read the exact source tensor through the retained authenticated handle.
    pub fn read_tensor(&mut self, name: &str) -> MimoResult<Vec<u8>> {
        let record = self.tensors.get(name).ok_or("source tensor absent")?;
        let source = self.sources.get(&record.file).ok_or("source file absent")?;
        let length = usize::try_from(record.bytes)?;
        if length > 2 << 30 {
            return Err("diagnostic tensor exceeds byte budget".into());
        }
        self.ledger.read(
            &record.file,
            source,
            format!("{}:{name}", record.file),
            record.file_start,
            length,
        )
    }

    pub fn diagnostic_identity(&self) -> MimoResult<serde_json::Value> {
        for source in self.sources.values() {
            source.verify()?;
        }
        Ok(self
            .ledger
            .identity(&self.identity.repository, &self.identity.revision))
    }

    pub fn identity(&self) -> &Identity {
        &self.identity
    }

    pub fn verify_tokenizer(&self, name: &str, byte_length: u64, sha256: &str) -> MimoResult<()> {
        let pin = self
            .ledger
            .lock()
            .pin(name)
            .ok_or("tokenizer is not pinned")?;
        if pin.byte_length != byte_length || pin.sha256 != sha256 {
            return Err("package tokenizer differs from selected checkpoint".into());
        }
        Ok(())
    }

    pub fn source_lock_sha256(&self) -> String {
        self.ledger.lock().sha256()
    }
}

fn authenticated_config(
    source: &SourceFile,
    lock: &model_compiler_sdk::provenance::SourceLock,
) -> MimoResult<crate::config::MiMoConfig> {
    use std::os::unix::fs::FileExt;
    source.verify()?;
    let length = source.metadata()?.len();
    if length > 1024 * 1024 {
        return Err("config.json exceeds byte budget".into());
    }
    let mut bytes = vec![0; usize::try_from(length)?];
    source.read_exact_at(&mut bytes, 0)?;
    source.verify()?;
    let pin = lock.pin("config.json").ok_or("config is not pinned")?;
    if length != pin.byte_length || runtime_weights::sha256_bytes(&bytes) != pin.sha256 {
        return Err("config.json differs from selected checkpoint".into());
    }
    let config: crate::config::MiMoConfig = serde_json::from_slice(&bytes)?;
    config.validate()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_compiler_sdk::provenance::SourceLock;
    use runtime_bundle::checkpoint_binding::FileDigest;

    struct Directory(std::path::PathBuf);
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn admits_data_only_identity_and_refuses_unknown_or_tampered_metadata() {
        let root = std::env::temp_dir().join(format!("mimo-third-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir(&root).unwrap();
        let _directory = Directory(root.clone());
        let header = br#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[3, 7]);
        std::fs::write(root.join("weight.safetensors"), bytes).unwrap();
        let index = serde_json::json!({"metadata":{"total_size":2},"weight_map":{"a":"weight.safetensors"}});
        std::fs::write(root.join("model.safetensors.index.json"), index.to_string()).unwrap();
        std::fs::write(
            root.join("config.json"),
            include_bytes!("../fixtures/config.json"),
        )
        .unwrap();
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            std::fs::write(root.join(name), b"{}").unwrap();
        }
        let mut data: serde_json::Value =
            serde_json::from_str(include_str!("../../checkpoints.json")).unwrap();
        data["checkpoints"]
            .as_object_mut()
            .unwrap()
            .retain(|name, _| matches!(name.as_str(), "mopd" | "rl"));
        let mut third = data["checkpoints"]["mopd"].clone();
        third["repository"] = "fixture/third".into();
        third["revision"] = "third-revision".into();
        third["served_model_name"] = "third".into();
        data["checkpoints"]["third"] = third;

        let mut pins = BTreeMap::new();
        for name in [
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "weight.safetensors",
        ] {
            let bytes = std::fs::read(root.join(name)).unwrap();
            pins.insert(
                name,
                FileDigest {
                    byte_length: bytes.len() as u64,
                    sha256: runtime_weights::sha256_bytes(&bytes),
                },
            );
        }
        let mut locks: serde_json::Value =
            serde_json::from_str(include_str!("../../checkpoint-files.json")).unwrap();
        locks["third"] = serde_json::to_value(pins).unwrap();
        let registry =
            crate::checkpoint::Registry::parse(&data.to_string(), &locks.to_string()).unwrap();
        let identity = registry.identity("third").unwrap();
        let admit = || registry.verifier(&root);
        let mut verifier = admit().unwrap();
        assert_eq!(verifier.config().unwrap().vocab_size, 152576);
        assert_eq!(verifier.read_tensor("a").unwrap(), vec![3, 7]);
        verifier
            .verify_tokenizer("tokenizer.json", 2, &runtime_weights::sha256_bytes(b"{}"))
            .unwrap();
        assert!(
            verifier
                .verify_tokenizer("tokenizer.json", 2, &"0".repeat(64))
                .is_err()
        );
        let error = registry.identity("unlisted").unwrap_err();
        assert!(error.contains("known identities: mopd, rl, third"));
        let mut unknown = identity.clone();
        unknown.revision = "unlisted".into();
        assert!(registry.source_lock(&unknown).is_err());
        for name in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
        ] {
            let original = std::fs::read(root.join(name)).unwrap();
            // Invalid JSON must fail authentication before it can be interpreted.
            std::fs::write(root.join(name), b"!").unwrap();
            let error = admit().err().unwrap().to_string();
            assert!(
                error.contains("differs from selected checkpoint"),
                "{name}: {error}"
            );
            std::fs::write(root.join(name), original).unwrap();
        }
        let retained = admit().unwrap();
        // A replacement with valid JSON must not become the authenticated config.
        std::fs::rename(root.join("config.json"), root.join("original-config.json")).unwrap();
        std::fs::copy(root.join("original-config.json"), root.join("config.json")).unwrap();
        assert!(
            retained
                .config()
                .unwrap_err()
                .to_string()
                .contains("source changed")
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reads_only_pinned_tensor_bytes_of_authenticated_files() {
        let root = std::env::temp_dir().join(format!("mimo-provenance-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("weight.safetensors");
        std::fs::write(&path, [11u8, 22]).unwrap();
        let source = catalog::open_source(&path).unwrap();
        let full = runtime_weights::hash_range(&source, 0, 2).unwrap();
        let raw = runtime_weights::hash_range(&source, 1, 1).unwrap();
        let pinned = |sha256: String| {
            SourceLedger::new(SourceLock::new(BTreeMap::from([(
                "weight.safetensors".into(),
                FileDigest {
                    byte_length: 2,
                    sha256,
                },
            )])))
        };
        let mut verifier = Verifier {
            identity: crate::checkpoint::identity("mopd").unwrap().clone(),
            registry: crate::checkpoint::registry().clone(),
            config: None,
            sources: BTreeMap::from([("weight.safetensors".into(), source)]),
            tensors: BTreeMap::from([(
                "a".into(),
                TensorRecord {
                    name: "a".into(),
                    dtype: runtime_weights::DType::U8,
                    shape: vec![1],
                    file: "weight.safetensors".into(),
                    file_start: 1,
                    bytes: 1,
                },
            )]),
            ledger: pinned(full),
        };
        assert_eq!(verifier.read_tensor("a").unwrap(), vec![22]);
        assert!(verifier.read_tensor("foreign").is_err());
        let identity = verifier.diagnostic_identity().unwrap();
        assert_eq!(
            identity["checkpoint"]["source_lock_sha256"],
            verifier.source_lock_sha256()
        );
        assert_eq!(identity["source_ranges"]["weight.safetensors:a"], raw);
        verifier.ledger = pinned("0".repeat(64));
        for _ in 0..2 {
            assert!(
                verifier
                    .read_tensor("a")
                    .unwrap_err()
                    .to_string()
                    .contains("differs from selected checkpoint")
            );
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn real_checkpoint_identity_rejects_the_other_pin() {
        let Ok(root) = std::env::var("CHOPINFER_MIMO_CHECKPOINT") else {
            eprintln!("SKIP: set checkpoint for provenance test");
            return;
        };
        let root = std::path::PathBuf::from(root);
        let Ok(selected) = std::env::var("CHOPINFER_MIMO_TEST_IDENTITY") else {
            eprintln!("SKIP: set rl or mopd for provenance test");
            return;
        };
        let identity = crate::checkpoint::identity(&selected).unwrap();
        Verifier::new(&root, identity).unwrap();
        let other =
            crate::checkpoint::identity(if selected == "rl" { "mopd" } else { "rl" }).unwrap();
        assert!(
            Verifier::new(&root, other)
                .err()
                .unwrap()
                .to_string()
                .contains("differs from selected checkpoint")
        );
    }
}
