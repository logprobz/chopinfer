//! Building the tokenizer one bundle declared, from bytes it authenticated.
//!
//! The two artifacts a tokenizer is built from are components of the same
//! bundle as the frontend declaration, and the declaration pins their exact
//! identity. So a caller reads those components through the host that verified
//! them, hands the bytes here, and the reader checks the identity a second
//! time against what was declared before it parses anything.
//!
//! Nothing here reads a path, and no model's constants are compiled in: the
//! numbers a reader is built with and the component policies it reads with
//! are declared.

use runtime_tokenizer::{
    BaseTokenRepresentation, ByteLevelPolicy, ConfigFieldPolicy, ConfigTokenRepresentation,
    ConfigTokenValue, FileIdentity, IgnoreMergesPolicy, MergeRecordPolicy, MergeResultIdPolicy,
    NormalizerPolicy, Tokenizer, TokenizerContract, TokenizerDialect,
};

use crate::descriptor::{
    FrontendBaseTokensV1, FrontendByteLevelPolicyV1, FrontendConfigFieldV1,
    FrontendConfigTokenValueV1, FrontendConfigTokensV1, FrontendDescriptorV1,
    FrontendIgnoreMergesV1, FrontendMergeRecordsV1, FrontendMergeResultIdsV1,
    FrontendTokenizerNormalizerV1, FrontendTokenizerPolicyV1,
};

/// The reader policy one declared tokenizer policy selects, member for member.
#[must_use]
pub fn tokenizer_dialect_v1(policy: &FrontendTokenizerPolicyV1) -> TokenizerDialect {
    let byte_level = |declared: FrontendByteLevelPolicyV1| ByteLevelPolicy {
        add_prefix_space: declared.add_prefix_space,
        trim_offsets: declared.trim_offsets,
        use_regex: declared.use_regex,
    };
    let token_field = |declared: &FrontendConfigFieldV1<FrontendConfigTokenValueV1>| {
        config_field_v1(declared, |value| match value {
            FrontendConfigTokenValueV1::ContractValue {} => ConfigTokenValue::ContractValue,
            FrontendConfigTokenValueV1::Literal { text } => ConfigTokenValue::Literal(text.clone()),
        })
    };
    TokenizerDialect {
        normalizer: match policy.normalizer {
            FrontendTokenizerNormalizerV1::IdentitySequence => NormalizerPolicy::IdentitySequence,
            FrontendTokenizerNormalizerV1::Nfc => NormalizerPolicy::Nfc,
            FrontendTokenizerNormalizerV1::Null => NormalizerPolicy::Null,
        },
        pretokenizer_byte_level: byte_level(policy.pretokenizer_byte_level),
        post_processor_byte_level: byte_level(policy.post_processor_byte_level),
        decoder_byte_level: byte_level(policy.decoder_byte_level),
        merge_results: match policy.merge_results {
            FrontendMergeResultIdsV1::Sequential { first_id } => {
                MergeResultIdPolicy::Sequential { first_id }
            }
            FrontendMergeResultIdsV1::Vocabulary {} => MergeResultIdPolicy::Vocabulary,
        },
        merge_records: match policy.merge_records {
            FrontendMergeRecordsV1::SpaceSeparatedString => MergeRecordPolicy::SpaceSeparatedString,
            FrontendMergeRecordsV1::StringPairArray => MergeRecordPolicy::StringPairArray,
            FrontendMergeRecordsV1::StringOrPair => MergeRecordPolicy::StringOrPair,
        },
        base_tokens: match policy.base_tokens {
            FrontendBaseTokensV1::LiteralPrefixThenByteLevel { byte_start_id } => {
                BaseTokenRepresentation::LiteralPrefixThenByteLevel { byte_start_id }
            }
            FrontendBaseTokensV1::ByteLevel {} => BaseTokenRepresentation::ByteLevel,
        },
        config_tokens: match policy.config_tokens {
            FrontendConfigTokensV1::AddedToken { normalized } => {
                ConfigTokenRepresentation::AddedToken { normalized }
            }
            FrontendConfigTokensV1::String {} => ConfigTokenRepresentation::String,
        },
        add_bos_token: config_field_v1(&policy.add_bos_token, |value| *value),
        bos_token: token_field(&policy.bos_token),
        unk_token: token_field(&policy.unk_token),
        tokenizer_class: policy.tokenizer_class.clone(),
        continuing_subword_prefix: policy.continuing_subword_prefix.clone(),
        end_of_word_suffix: policy.end_of_word_suffix.clone(),
        ignore_merges: match policy.ignore_merges {
            FrontendIgnoreMergesV1::Absent => IgnoreMergesPolicy::Absent,
            FrontendIgnoreMergesV1::Disabled => IgnoreMergesPolicy::Disabled,
            FrontendIgnoreMergesV1::WholePiece => IgnoreMergesPolicy::WholePiece,
        },
        add_eos_token: policy.add_eos_token,
        legacy: policy.legacy,
        empty_sentencepiece_kwargs: policy.empty_sentencepiece_kwargs,
    }
}

fn config_field_v1<T, U>(
    declared: &FrontendConfigFieldV1<T>,
    value: impl FnOnce(&T) -> U,
) -> ConfigFieldPolicy<U> {
    match declared {
        FrontendConfigFieldV1::Required { value: declared } => {
            ConfigFieldPolicy::Required(value(declared))
        }
        FrontendConfigFieldV1::RequiredNull {} => ConfigFieldPolicy::RequiredNull,
        FrontendConfigFieldV1::Absent {} => ConfigFieldPolicy::Absent,
    }
}

/// Builds the tokenizer contract this declaration pins.
pub fn tokenizer_contract_v1(
    descriptor: &FrontendDescriptorV1,
) -> Result<TokenizerContract, String> {
    let declared = &descriptor.tokenizer;
    let identity = |file: &crate::descriptor::FrontendFileIdentityV1| {
        FileIdentity::new(&file.name, file.byte_length, &file.sha256)
            .map_err(|error| format!("declared tokenizer file identity is invalid: {error}"))
    };
    let patterns: Vec<&str> = declared
        .pretokenizer_patterns
        .iter()
        .map(String::as_str)
        .collect();
    let specials: Vec<&str> = descriptor
        .vocabulary
        .prompt_special_tokens
        .iter()
        .map(String::as_str)
        .collect();
    let dialect = tokenizer_dialect_v1(&declared.policy);
    let vocabulary_size = declared
        .vocabulary_size
        .unwrap_or(descriptor.vocabulary.size);
    if vocabulary_size > descriptor.vocabulary.size {
        return Err(format!(
            "declared tokenizer vocabulary size {vocabulary_size} exceeds model logits width {}",
            descriptor.vocabulary.size
        ));
    }
    TokenizerContract::new_with_dialect(
        &declared.repository,
        &declared.revision,
        identity(&declared.definition)?,
        identity(&declared.configuration)?,
        declared.base_vocabulary_size,
        vocabulary_size,
        declared.merge_count,
        declared.added_token_count,
        declared.model_max_length,
        declared.bos_token.as_deref(),
        &declared.eos_token,
        &declared.pad_token,
        &patterns,
        &specials,
        dialect,
    )
    .map_err(|error| format!("declared tokenizer contract is invalid: {error}"))
}

/// Builds the tokenizer from the two components its bundle ships.
///
/// The caller supplies bytes it read through the host that verified them; the
/// contract checks their length and digest again before parsing, so a caller
/// that read the wrong component gets a refusal rather than a tokenizer.
pub fn tokenizer_from_components_v1(
    descriptor: &FrontendDescriptorV1,
    definition: &[u8],
    configuration: &[u8],
) -> Result<Tokenizer, String> {
    let contract = tokenizer_contract_v1(descriptor)?;
    let tokenizer = Tokenizer::from_verified_bytes(definition, configuration, &contract)
        .map_err(|error| format!("declared tokenizer does not build: {error}"))?;
    bind_declared_boundaries_v1(descriptor, &tokenizer)?;
    Ok(tokenizer)
}

/// Checks that the built tokenizer means what the declaration says it means.
///
/// The three components were each authenticated on their own, which says
/// nothing about whether they belong together. This is what binds them: every
/// identifier the declaration gave a boundary has to decode, in this
/// tokenizer, to exactly that boundary's text. Otherwise a prompt would be
/// rendered with one boundary and generated against another, and the decoder
/// would police identifiers that mean something else.
fn bind_declared_boundaries_v1(
    descriptor: &FrontendDescriptorV1,
    tokenizer: &Tokenizer,
) -> Result<(), String> {
    let vocabulary = &descriptor.vocabulary;
    let decoded = |id: u32, what: &str| -> Result<Vec<u8>, String> {
        tokenizer
            .token_bytes(id)
            .map(<[u8]>::to_vec)
            .map_err(|error| format!("declared {what} identifier {id} does not decode: {error}"))
    };
    for boundary in &vocabulary.boundary_tokens {
        if decoded(boundary.id, "boundary token")? != boundary.text.as_bytes() {
            return Err(format!(
                "declared boundary token '{}' is not what identifier {} decodes to",
                boundary.text, boundary.id
            ));
        }
    }
    // The two identifiers the generation side stops and starts on are named in
    // the vocabulary but spelled in the tokenizer section, so they are the one
    // place the two sections have to agree.
    if decoded(vocabulary.eos_token_id, "eos token")? != descriptor.tokenizer.eos_token.as_bytes() {
        return Err(format!(
            "declared eos_token '{}' is not what identifier {} decodes to",
            descriptor.tokenizer.eos_token, vocabulary.eos_token_id
        ));
    }
    // A tokenizer that declares no beginning-of-sequence token pins nothing
    // here, so nothing is required of the identifier.
    if let Some(bos_token) = &descriptor.tokenizer.bos_token
        && decoded(vocabulary.bos_token_id, "bos token")? != bos_token.as_bytes()
    {
        return Err(format!(
            "declared bos_token '{bos_token}' is not what identifier {} decodes to",
            vocabulary.bos_token_id
        ));
    }
    Ok(())
}
