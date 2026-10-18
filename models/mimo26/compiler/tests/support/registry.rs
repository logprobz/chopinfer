use model_compiler_sdk::preparation::tokenizer::TokenizerSemantics;
use model_mimo26::checkpoint::Registry;
use runtime_bundle::checkpoint_binding::FileDigest;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// The chat template of the tokenizer configuration
/// [`Fixture::state_tokenizer_semantics`] writes.
pub const TEMPLATE: &str = "{{ fixture }}";

pub struct Fixture {
    pub root: PathBuf,
    pub data: serde_json::Value,
    pub locks: serde_json::Value,
}

impl Fixture {
    pub fn new() -> Self {
        // Tests run as threads of one process and can start fixtures in the
        // same nanosecond: the counter gives each fixture its own directory,
        // and the process ID and time keep other and earlier runs apart. A
        // fixture never removes a directory it did not create.
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let index = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "mimo-registry-{}-{index}-{stamp}",
            std::process::id()
        ));
        std::fs::create_dir(&root).unwrap();
        let fixture = Self {
            root,
            data: serde_json::from_str(include_str!("../../../checkpoints.json")).unwrap(),
            locks: serde_json::from_str(include_str!("../../../checkpoint-files.json")).unwrap(),
        };
        fixture.populate()
    }

    fn populate(mut self) -> Self {
        self.data["checkpoints"]
            .as_object_mut()
            .unwrap()
            .retain(|name, _| matches!(name.as_str(), "mopd" | "rl"));
        self.locks
            .as_object_mut()
            .unwrap()
            .retain(|name, _| matches!(name.as_str(), "mopd" | "rl"));
        let header = br#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#;
        let mut weight = (header.len() as u64).to_le_bytes().to_vec();
        weight.extend_from_slice(header);
        weight.extend_from_slice(&[3, 7]);
        std::fs::write(self.root.join("weight.safetensors"), weight).unwrap();
        std::fs::write(
            self.root.join("model.safetensors.index.json"),
            br#"{"metadata":{"total_size":2},"weight_map":{"a":"weight.safetensors"}}"#,
        )
        .unwrap();
        std::fs::write(
            self.root.join("config.json"),
            include_bytes!("../../fixtures/config.json"),
        )
        .unwrap();
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            std::fs::write(self.root.join(name), b"{}").unwrap();
        }
        self.data["checkpoints"]["third"] = serde_json::json!({
            "repository": "fixture/third", "revision": "0123456789abcdef0123456789abcdef01234567", "served_model_name": "third",
            "frontend": {"tokenizer": {"model_max_length": 2097152}, "vocabulary": {"stop_token_ids": [151645]}}
        });
        let pins: BTreeMap<_, _> = [
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "weight.safetensors",
        ]
        .into_iter()
        .map(|name| {
            let bytes = std::fs::read(self.root.join(name)).unwrap();
            (
                name,
                FileDigest {
                    byte_length: bytes.len() as u64,
                    sha256: runtime_weights::sha256_bytes(&bytes),
                },
            )
        })
        .collect();
        self.locks["third"] = serde_json::to_value(pins).unwrap();
        self
    }

    /// The registry of the fixture's identities, which checks a local
    /// identity's tokenizer configuration against [`Fixture::semantics`].
    pub fn registry(&self) -> Registry {
        Registry::parse(&self.data.to_string(), &self.locks.to_string())
            .unwrap()
            .with_tokenizer_semantics(Self::semantics())
    }

    /// A tokenizer configuration with [`TEMPLATE`] and one end token.
    pub fn semantics() -> TokenizerSemantics {
        TokenizerSemantics::parse(
            &serde_json::to_vec(&serde_json::json!({
                "chat_template_sha256": runtime_weights::sha256_bytes(TEMPLATE.as_bytes()),
                "configuration": {"eos_token": "<|im_end|>"},
            }))
            .unwrap(),
        )
        .unwrap()
    }

    /// Writes a tokenizer configuration that states [`Fixture::semantics`],
    /// pinned for `third`.
    pub fn state_tokenizer_semantics(&mut self) {
        let configuration = serde_json::json!({
            "chat_template": TEMPLATE, "eos_token": "<|im_end|>", "name_or_path": "fixture",
        });
        std::fs::write(
            self.root.join("tokenizer_config.json"),
            serde_json::to_vec(&configuration).unwrap(),
        )
        .unwrap();
        self.pin("tokenizer_config.json");
    }

    /// Pins the current bytes of `name` for `third`.
    pub fn pin(&mut self, name: &str) {
        let bytes = std::fs::read(self.root.join(name)).unwrap();
        self.locks["third"][name] = serde_json::to_value(FileDigest {
            byte_length: bytes.len() as u64,
            sha256: runtime_weights::sha256_bytes(&bytes),
        })
        .unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
