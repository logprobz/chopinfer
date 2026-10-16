//! Token truncation runs after tokenization and before context admission.
use crate::{PreparedPrompt, error::ApiError};

#[derive(Clone, Copy, Default)]
pub(crate) struct Truncation {
    count: Option<i64>,
    right: bool,
}

impl Truncation {
    pub(crate) fn new(count: Option<i64>, side: Option<&str>) -> Result<Self, ApiError> {
        if count.is_some_and(|count| count < -1) {
            return Err(ApiError::invalid(
                "truncate_prompt_tokens must be -1 or nonnegative",
                Some("truncate_prompt_tokens"),
            ));
        }
        if !matches!(side, None | Some("left" | "right")) {
            return Err(ApiError::invalid(
                "truncation_side must be left or right",
                Some("truncation_side"),
            ));
        }
        Ok(Self {
            count,
            right: side == Some("right"),
        })
    }

    pub(crate) fn limit(self, context: usize, output: usize) -> Result<Option<usize>, ApiError> {
        let Some(count) = self.count else {
            return Ok(None);
        };
        let available = context.saturating_sub(output);
        let limit = if count == -1 {
            available
        } else {
            usize::try_from(count).unwrap_or(usize::MAX)
        };
        if limit > available {
            return Err(ApiError::invalid(
                "truncate_prompt_tokens exceeds context minus requested output tokens",
                Some("truncate_prompt_tokens"),
            ));
        }
        Ok(Some(limit))
    }

    pub(crate) fn apply(
        self,
        prompt: &mut PreparedPrompt,
        count: usize,
        limit: Option<usize>,
    ) -> Result<usize, ApiError> {
        let Some(limit) = limit else {
            return Ok(count);
        };
        if limit >= count {
            return Ok(count);
        }
        let PreparedPrompt::TokenIds(tokens) = prompt else {
            return Err(ApiError::invalid(
                "prompt truncation requires text token IDs; media spans cannot be truncated",
                Some("truncate_prompt_tokens"),
            ));
        };
        if self.right {
            tokens.truncate(limit);
        } else {
            tokens.drain(..tokens.len() - limit);
        }
        if tokens.is_empty() {
            return Err(ApiError::invalid(
                "truncation leaves an empty prompt",
                Some("truncate_prompt_tokens"),
            ));
        }
        Ok(tokens.len())
    }
}
