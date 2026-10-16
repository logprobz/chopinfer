use crate::tokenizer::PROMPT_TOKENS;
use runtime_tokenizer::{FileIdentity, Tokenizer, TokenizerContract};

pub use crate::checkpoint_index::{Checkpoint, checkpoints};

impl Checkpoint {
    pub fn contract(&self) -> TokenizerContract {
        let p: serde_json::Value = self.fixture("tokenizer-contract.json");
        let identity = |key: &str| {
            let i = &p[key];
            FileIdentity::new(
                i["name"].as_str().unwrap(),
                i["byte_length"].as_u64().unwrap(),
                i["sha256"].as_str().unwrap(),
            )
            .unwrap()
        };
        // The capture names the dialect it was recorded under. This model's
        // declared policy reproduces it (`tests/tokenizer_goldens.rs`).
        assert_eq!(p["dialect"]["kind"], "qwen2-nfc-v1");
        let patterns: Vec<String> =
            serde_json::from_value(p["pretokenizer_patterns"].clone()).unwrap();
        let patterns: Vec<&str> = patterns.iter().map(String::as_str).collect();
        TokenizerContract::new_with_dialect(
            p["repository"].as_str().unwrap(),
            p["revision"].as_str().unwrap(),
            identity("definition"),
            identity("configuration"),
            p["base_vocabulary_size"].as_u64().unwrap() as u32,
            p["vocabulary_size"].as_u64().unwrap() as u32,
            p["merge_count"].as_u64().unwrap() as usize,
            p["added_token_count"].as_u64().unwrap() as usize,
            p["model_max_length"].as_u64().unwrap() as usize,
            p["bos_token"].as_str(),
            p["eos_token"].as_str().unwrap(),
            p["pad_token"].as_str().unwrap(),
            &patterns,
            PROMPT_TOKENS,
            crate::tokenizer::tokenizer_dialect(),
        )
        .unwrap()
    }

    pub fn tokenizer(&self) -> Option<Tokenizer> {
        self.root().map(|root| {
            crate::local_frontend::LocalFrontend::load(&root, &"0".repeat(64))
                .unwrap()
                .tokenizer()
                .unwrap()
        })
    }
}

#[test]
fn indexed_checkpoint_configs() {
    for checkpoint in checkpoints() {
        if let Some(root) = checkpoint.checkpoint_root() {
            let config = crate::config::Qwen38Config::load(&root).unwrap();
            assert_eq!(
                config.geometry().unwrap().geometry,
                checkpoint.execution_geometry()
            );
        }
    }
}
