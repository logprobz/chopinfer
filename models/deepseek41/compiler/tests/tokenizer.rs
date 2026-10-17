//! The tokenizer declaration against the official checkpoint: the reader
//! policy is the dialect COMPARISON.md section 2 established, and HF
//! `tokenizers` goldens (`models/deepseek41/tokenizer_goldens.py`) encode
//! and decode identically.
use model_deepseek41::tokenizer;
use runtime_frontend_contract::tokenizer_dialect_v1;
use runtime_tokenizer::{
    BaseTokenRepresentation, ByteLevelPolicy, ConfigFieldPolicy, ConfigTokenRepresentation,
    ConfigTokenValue, IgnoreMergesPolicy, MergeRecordPolicy, MergeResultIdPolicy, NormalizerPolicy,
    TokenizerDialect,
};
use serde::Deserialize;

mod common;

#[test]
fn the_declared_policy_is_the_checked_dialect() {
    let declared = tokenizer::declaration();
    let expected = TokenizerDialect {
        normalizer: NormalizerPolicy::IdentitySequence,
        pretokenizer_byte_level: ByteLevelPolicy {
            add_prefix_space: false,
            trim_offsets: true,
            use_regex: false,
        },
        post_processor_byte_level: ByteLevelPolicy {
            add_prefix_space: true,
            trim_offsets: false,
            use_regex: true,
        },
        decoder_byte_level: ByteLevelPolicy {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        },
        merge_results: MergeResultIdPolicy::Sequential { first_id: 259 },
        merge_records: MergeRecordPolicy::SpaceSeparatedString,
        base_tokens: BaseTokenRepresentation::LiteralPrefixThenByteLevel { byte_start_id: 3 },
        config_tokens: ConfigTokenRepresentation::AddedToken { normalized: true },
        add_bos_token: ConfigFieldPolicy::Required(false),
        bos_token: ConfigFieldPolicy::Required(ConfigTokenValue::ContractValue),
        unk_token: ConfigFieldPolicy::RequiredNull,
        tokenizer_class: "PreTrainedTokenizerFast".into(),
        continuing_subword_prefix: None,
        end_of_word_suffix: None,
        ignore_merges: IgnoreMergesPolicy::Absent,
        add_eos_token: Some(false),
        legacy: Some(true),
        empty_sentencepiece_kwargs: true,
    };
    assert_eq!(tokenizer_dialect_v1(&declared.tokenizer.policy), expected);
    assert_eq!(declared.tokenizer.base_vocabulary_size, 128_000);
    assert_eq!(declared.tokenizer.merge_count, 127_741);
    assert_eq!(declared.tokenizer.added_token_count, 1283);
    assert_eq!(declared.vocabulary.size, 129_280);
    assert_eq!(declared.tokenizer.pretokenizer_patterns.len(), 3);
    // The prompt carries only declared boundaries.
    for special in &declared.vocabulary.prompt_special_tokens {
        assert!(
            declared
                .vocabulary
                .boundary_tokens
                .iter()
                .any(|b| &b.text == special),
            "{special}"
        );
    }
    tokenizer::contract().unwrap();
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Golden {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    checkpoint_file: Option<String>,
    #[serde(default)]
    sha256: Option<String>,
    token_ids: Vec<u32>,
}

#[test]
fn hf_goldens_encode_and_decode_identically() {
    let Some(dir) = common::checkpoint() else {
        return;
    };
    let tokenizer = tokenizer::load(&dir).expect("load the pinned checkpoint tokenizer");
    let added = common::added_tokens(&dir);
    let allowed = added.iter().map(String::as_str).collect::<Vec<_>>();
    let goldens: Vec<Golden> = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek41/frontend/tokenizer-golden.json"
    ))
    .unwrap();
    assert_eq!(goldens.len(), 49);
    for golden in goldens {
        let text = match (&golden.text, &golden.checkpoint_file) {
            (Some(text), None) => text.clone(),
            (None, Some(name)) => {
                let bytes = std::fs::read(dir.join(name)).unwrap();
                let digest = sha256(&bytes);
                assert_eq!(Some(digest), golden.sha256, "{name}");
                String::from_utf8(bytes).unwrap()
            }
            _ => panic!("a golden names a text or a checkpoint file"),
        };
        let ids = tokenizer
            .encode(&text, &allowed)
            .unwrap_or_else(|e| panic!("encode {text:?}: {e}"));
        assert_eq!(ids, golden.token_ids, "encode {text:?}");
        assert_eq!(
            tokenizer.decode_bytes(&ids).unwrap(),
            text.as_bytes(),
            "decode {text:?}"
        );
    }
}

fn sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
