#![forbid(unsafe_code)]
//! A whole-completion parse refuses what one host token carrying the whole
//! text is refused for, in each shared tool grammar, whether or not the
//! declaration was validated.
mod common;

use frontend_format::{FormatDescriptorV1, PromptModeV1, parse_completion_v1};
use runtime_frontend_contract::FrontendBoundaryTokenV1;

/// The parser fixtures, one on each shared tool grammar.
fn fixtures() -> [(&'static str, FormatDescriptorV1); 3] {
    [
        ("key-value", common::KEY_VALUE),
        ("function-parameters", common::FUNCTION_PARAMETERS),
        ("typed-parameters", common::TYPED_PARAMETERS),
    ]
    .map(|(name, declaration)| {
        (
            name,
            FormatDescriptorV1::from_component_bytes(declaration.as_bytes()).unwrap(),
        )
    })
}

#[test]
fn a_zero_token_limit_refuses_every_completion() {
    for (name, mut descriptor) in fixtures() {
        assert_eq!(
            parse_completion_v1(&descriptor, "answer[EOS]", PromptModeV1::Chat)
                .map(|completion| completion.content),
            Ok("answer".to_owned()),
            "{name}"
        );
        descriptor.limits.maximum_generated_tokens = 0;
        assert_eq!(
            parse_completion_v1(&descriptor, "answer[EOS]", PromptModeV1::Chat),
            Err(
                "conversation resource limit: incremental completion exceeds the token limit"
                    .into()
            ),
            "{name}"
        );
    }
}

#[test]
fn the_whole_text_token_is_checked_against_its_declared_id() {
    for (name, mut descriptor) in fixtures() {
        descriptor
            .vocabulary
            .boundary_tokens
            .push(FrontendBoundaryTokenV1 {
                text: "<sentinel>".into(),
                id: u32::MAX,
            });
        assert_eq!(
            parse_completion_v1(&descriptor, "answer[EOS]", PromptModeV1::Chat),
            Err("malformed completion: special token ID has mismatched decoded bytes".into()),
            "{name}"
        );
    }
}
