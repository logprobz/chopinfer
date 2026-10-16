#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt;

/// The exact number of token IDs in one model's logits domain.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ModelVocabulary {
    size: usize,
}

impl ModelVocabulary {
    pub const fn new(size: usize) -> Result<Self, ModelVocabularyError> {
        if size == 0 {
            return Err(ModelVocabularyError::ZeroVocabulary);
        }
        let maximum_size = u32::MAX as u128 + 1;
        if size as u128 > maximum_size {
            return Err(ModelVocabularyError::TokenIdOverflow { provided: size });
        }
        Ok(Self { size })
    }

    #[must_use]
    pub const fn get(self) -> usize {
        self.size
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelVocabularyError {
    ZeroVocabulary,
    TokenIdOverflow { provided: usize },
}

impl fmt::Display for ModelVocabularyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroVocabulary => formatter.write_str("model vocabulary is zero"),
            Self::TokenIdOverflow { provided } => write!(
                formatter,
                "model vocabulary {provided} exceeds the token ID domain"
            ),
        }
    }
}

impl Error for ModelVocabularyError {}
