#[allow(dead_code)]
#[path = "../src/checkpoint_index.rs"]
mod checkpoint_index;
use model_qwen38::{config, local_frontend, tokenizer};
mod support;
use model_qwen38::tokenizer::PROMPT_TOKENS;
use runtime_frontend::tokenizer_dialect_v1;
use runtime_tokenizer::{
    BaseTokenRepresentation, ByteLevelPolicy, ConfigFieldPolicy, ConfigTokenRepresentation,
    IgnoreMergesPolicy, MergeRecordPolicy, MergeResultIdPolicy, NormalizerPolicy, TokenizerDialect,
};
use serde::Deserialize;
#[derive(Deserialize)]
struct Golden {
    text: String,
    token_ids: Vec<u32>,
}
#[test]
fn hf_tokenizations() {
    for checkpoint in support::checkpoints() {
        let Some(tokenizer) = checkpoint.tokenizer() else {
            eprintln!("SKIP: {} tokenizer assets are unset", checkpoint.name);
            continue;
        };
        let goldens: Vec<Golden> = checkpoint.fixture("tokenizer-golden.json");
        assert_eq!(goldens.len(), checkpoint.tokenizer_cases);
        for g in goldens {
            let actual = tokenizer.encode(&g.text, PROMPT_TOKENS).unwrap();
            assert_eq!(actual, g.token_ids, "{}: {:?}", checkpoint.name, g.text);
            let decoded = tokenizer.decode_utf8(&actual).unwrap();
            assert_eq!(tokenizer.encode(&decoded, PROMPT_TOKENS).unwrap(), actual);
        }
    }
}

#[test]
fn checkpoint_pins_match_assets() {
    use sha2::{Digest, Sha256};
    for checkpoint in support::checkpoints() {
        let identities: Vec<serde_json::Value> = checkpoint.fixture("artifact-identities.json");
        assert!(!identities.is_empty());
        let contract = checkpoint.contract();
        if contract.revision() == model_qwen38::REVISION {
            assert_eq!(
                contract,
                model_qwen38::tokenizer::tokenizer_contract().unwrap()
            );
        }
        for identity in [contract.definition(), contract.configuration()] {
            let pin = identities
                .iter()
                .find(|p| p["name"] == identity.name())
                .unwrap();
            assert_eq!(pin["byte_length"], identity.size());
            assert_eq!(pin["sha256"], identity.sha256());
        }
        // Chat case counts are part of the index shared with chat_goldens.
        assert!(checkpoint.chat_cases > 0);
        let Some(root) = std::env::var_os(&checkpoint.checkpoint_env) else {
            continue;
        };
        for pin in identities {
            let bytes = runtime_weights::checkpoint::CheckpointFile::open(
                &std::path::PathBuf::from(&root),
                pin["name"].as_str().unwrap(),
            )
            .unwrap()
            .read_bounded(pin["byte_length"].as_u64().unwrap())
            .unwrap();
            assert_eq!(bytes.len() as u64, pin["byte_length"].as_u64().unwrap());
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                pin["sha256"].as_str().unwrap()
            );
        }
    }
}

/// The declared policy is the `qwen2-nfc-v1` dialect this model was served
/// with before tokenizer policy was declared, member for member.
#[test]
fn declared_tokenizer_policy_is_the_qwen2_nfc_dialect() {
    let byte_level = ByteLevelPolicy {
        add_prefix_space: false,
        trim_offsets: false,
        use_regex: false,
    };
    let expected = TokenizerDialect {
        normalizer: NormalizerPolicy::Nfc,
        pretokenizer_byte_level: byte_level,
        post_processor_byte_level: byte_level,
        decoder_byte_level: byte_level,
        merge_results: MergeResultIdPolicy::Vocabulary,
        merge_records: MergeRecordPolicy::StringOrPair,
        base_tokens: BaseTokenRepresentation::ByteLevel,
        config_tokens: ConfigTokenRepresentation::String,
        add_bos_token: ConfigFieldPolicy::Required(false),
        bos_token: ConfigFieldPolicy::RequiredNull,
        unk_token: ConfigFieldPolicy::RequiredNull,
        tokenizer_class: "Qwen2Tokenizer".to_owned(),
        continuing_subword_prefix: Some(String::new()),
        end_of_word_suffix: Some(String::new()),
        ignore_merges: IgnoreMergesPolicy::Disabled,
        add_eos_token: None,
        legacy: None,
        empty_sentencepiece_kwargs: false,
    };
    assert_eq!(
        tokenizer_dialect_v1(&model_qwen38::frontend::descriptor().tokenizer.policy),
        expected
    );
}
