//! The tokenizer contract: the official checkpoint's `tokenizer.json` and
//! `tokenizer_config.json` pinned by identity, and the reader policy they
//! load with (`tokenizer-contract.json`). Its `tokenizer` and `vocabulary`
//! sections are the frontend declaration's, member for member.
use runtime_frontend_contract::{FrontendTokenizerV1, FrontendVocabularyV1, tokenizer_dialect_v1};
use runtime_tokenizer::{FileIdentity, Tokenizer, TokenizerContract, TokenizerError};
use serde::Deserialize;
use std::path::Path;

/// The declared tokenizer and the vocabulary facts it is bound to.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Declaration {
    pub tokenizer: FrontendTokenizerV1,
    pub vocabulary: FrontendVocabularyV1,
}

pub fn declaration() -> Declaration {
    serde_json::from_str(include_str!("tokenizer-contract.json"))
        .expect("tokenizer-contract.json is a tokenizer declaration")
}

/// The special tokens a rendered prompt may carry.
pub fn prompt_special_tokens() -> Vec<String> {
    declaration().vocabulary.prompt_special_tokens
}

/// The contract the declaration pins, built as the host builds it
/// (`runtime_frontend_contract::tokenizer_contract_v1`).
pub fn contract() -> runtime_tokenizer::Result<TokenizerContract> {
    let Declaration {
        tokenizer,
        vocabulary,
    } = declaration();
    let identity = |file: &runtime_frontend_contract::FrontendFileIdentityV1| {
        FileIdentity::new(&file.name, file.byte_length, &file.sha256)
    };
    let patterns = tokenizer
        .pretokenizer_patterns
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let specials = vocabulary
        .prompt_special_tokens
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    TokenizerContract::new_with_dialect(
        &tokenizer.repository,
        &tokenizer.revision,
        identity(&tokenizer.definition)?,
        identity(&tokenizer.configuration)?,
        tokenizer.base_vocabulary_size,
        tokenizer.vocabulary_size.unwrap_or(vocabulary.size),
        tokenizer.merge_count,
        tokenizer.added_token_count,
        tokenizer.model_max_length,
        tokenizer.bos_token.as_deref(),
        &tokenizer.eos_token,
        &tokenizer.pad_token,
        &patterns,
        &specials,
        tokenizer_dialect_v1(&tokenizer.policy),
    )
}

/// Load the tokenizer from a checkpoint directory and check that every
/// declared boundary, and the end- and beginning-of-sequence identifiers,
/// decode to their declared text.
pub fn load(checkpoint: &Path) -> runtime_tokenizer::Result<Tokenizer> {
    checked(Tokenizer::load(checkpoint, &contract()?)?)
}

/// Load the exact owned bytes that preparation hashed into its cache key.
pub fn load_bytes(definition: &[u8], configuration: &[u8]) -> runtime_tokenizer::Result<Tokenizer> {
    checked(Tokenizer::from_verified_bytes(
        definition,
        configuration,
        &contract()?,
    )?)
}

fn checked(tokenizer: Tokenizer) -> runtime_tokenizer::Result<Tokenizer> {
    let declared = declaration();
    let vocabulary = &declared.vocabulary;
    let pins = vocabulary
        .boundary_tokens
        .iter()
        .map(|b| (b.id, b.text.as_str()))
        .chain([(
            vocabulary.eos_token_id,
            declared.tokenizer.eos_token.as_str(),
        )])
        .chain(
            declared
                .tokenizer
                .bos_token
                .as_deref()
                .map(|bos| (vocabulary.bos_token_id, bos)),
        );
    for (id, text) in pins {
        if tokenizer.token_bytes(id)? != text.as_bytes() {
            return Err(TokenizerError::validation(format!(
                "declared token {text:?} is not what identifier {id} decodes to"
            )));
        }
    }
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_declaration_builds_a_contract() {
        let declared = super::declaration();
        assert_eq!(
            declared.tokenizer.repository,
            "deepseek-ai/DeepSeek-V4.1-Flash-1017"
        );
        assert_eq!(declared.vocabulary.eos_token_id, 1);
        super::contract().unwrap();
    }
}
