use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ContextLengthError {
    TokenCountOverflow {
        prompt_tokens: usize,
        max_new_tokens: usize,
        maximum: usize,
    },
    ExceedsLimit {
        prompt_tokens: usize,
        max_new_tokens: usize,
        maximum: usize,
        requested: usize,
    },
}

pub(crate) fn validate_total_tokens(
    prompt_tokens: usize,
    max_new_tokens: usize,
    maximum: usize,
) -> Result<usize, ContextLengthError> {
    let requested = prompt_tokens.checked_add(max_new_tokens).ok_or(
        ContextLengthError::TokenCountOverflow {
            prompt_tokens,
            max_new_tokens,
            maximum,
        },
    )?;
    if requested > maximum {
        return Err(ContextLengthError::ExceedsLimit {
            prompt_tokens,
            max_new_tokens,
            maximum,
            requested,
        });
    }
    Ok(requested)
}

impl fmt::Display for ContextLengthError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenCountOverflow {
                prompt_tokens,
                max_new_tokens,
                maximum,
            } => write!(
                formatter,
                "prompt token count {prompt_tokens} plus maximum generated token count {max_new_tokens} overflows the serving limit {maximum}"
            ),
            Self::ExceedsLimit {
                prompt_tokens,
                max_new_tokens,
                maximum,
                requested,
            } => write!(
                formatter,
                "prompt token count {prompt_tokens} plus maximum generated token count {max_new_tokens} is {requested}, which exceeds the serving limit {maximum}"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ContextLengthError, validate_total_tokens};

    #[test]
    fn total_token_validation_accepts_the_boundary_and_rejects_one_more() {
        assert_eq!(validate_total_tokens(524_287, 1, 524_288), Ok(524_288));
        assert_eq!(
            validate_total_tokens(524_288, 1, 524_288),
            Err(ContextLengthError::ExceedsLimit {
                prompt_tokens: 524_288,
                max_new_tokens: 1,
                maximum: 524_288,
                requested: 524_289,
            })
        );
    }

    #[test]
    fn total_token_validation_fails_closed_on_checked_overflow() {
        assert_eq!(
            validate_total_tokens(usize::MAX, 1, usize::MAX),
            Err(ContextLengthError::TokenCountOverflow {
                prompt_tokens: usize::MAX,
                max_new_tokens: 1,
                maximum: usize::MAX,
            })
        );
    }
}
