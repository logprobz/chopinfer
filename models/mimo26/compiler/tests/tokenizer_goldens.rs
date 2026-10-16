use std::path::Path;

use model_mimo26::tokenizer::{load_tokenizer, special_tokens};
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
fn captured_tokenizations_match_the_checkpoint() {
    let Ok(checkpoint_dir) = std::env::var("CHOPINFER_MIMO_TOKENIZER_DIR")
        .or_else(|_| std::env::var("CHOPINFER_MIMO_CHECKPOINT"))
    else {
        eprintln!(
            "CHOPINFER_MIMO_TOKENIZER_DIR and CHOPINFER_MIMO_CHECKPOINT are unset, skipping checkpoint goldens"
        );
        return;
    };
    let checkpoint = Path::new(&checkpoint_dir);
    if !checkpoint.join("tokenizer.json").exists() {
        eprintln!("checkpoint tokenizer is absent, skipping tokenizer goldens");
        return;
    }
    let tokenizer = load_tokenizer(checkpoint).expect("load pinned checkpoint tokenizer");
    let goldens: Vec<Golden> = serde_json::from_str(include_str!(
        "../../../../fixtures/mimo26/frontend/tokenizer-golden.json"
    ))
    .expect("parse tokenizer capture");
    assert_eq!(goldens.len(), 52);
    for golden in goldens {
        let actual = tokenizer
            .encode(
                &golden.text,
                &special_tokens()
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
            )
            .unwrap_or_else(|error| panic!("encode {:?}: {error}", golden.text));
        assert_eq!(actual, golden.token_ids, "encode {:?}", golden.text);
        let decoded = tokenizer
            .decode_utf8(&actual)
            .expect("decode captured tokens");
        assert_eq!(
            tokenizer
                .encode(
                    &decoded,
                    &special_tokens()
                        .iter()
                        .map(String::as_str)
                        .collect::<Vec<_>>()
                )
                .expect("re-encode decoded text"),
            golden.token_ids,
            "decode round-trip {:?}",
            golden.text
        );
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
        tokenizer_dialect_v1(&model_mimo26::frontend::descriptor().tokenizer.policy),
        expected
    );
}
