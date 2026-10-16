#![forbid(unsafe_code)]
//! The whole-completion parse of a completion in either shared tool grammar
//! answers as one caller token carrying the whole text does through the
//! decoder, for declarations that do not validate too.

use std::sync::Arc;

use frontend_format::test_support::fixtures::{FUNCTION_PARAMETERS, KEY_VALUE};
use runtime_frontend::{
    AssistantCompletionV1, CompletionDecoderV1, CompletionUpdateV1, DecoderLimitsV1,
    FrontendBoundaryTokenV1, FrontendDescriptorV1, GeneratedTokenV1, PromptModeV1,
    parse_completion_v1,
};

/// The parser fixtures, with the reasoning close each declares.
fn forms() -> [(&'static str, FrontendDescriptorV1, &'static str); 2] {
    let read = |declaration: &str| {
        FrontendDescriptorV1::from_component_bytes(declaration.as_bytes()).unwrap()
    };
    [
        ("function-parameter", read(FUNCTION_PARAMETERS), "</think>"),
        ("key-value", read(KEY_VALUE), "</t>"),
    ]
}

/// What a whole-completion parse answered before the format moved out of
/// this crate: the byte cap, then one caller token through the decoder.
fn one_token(
    descriptor: &FrontendDescriptorV1,
    text: &str,
    mode: PromptModeV1,
) -> Result<AssistantCompletionV1, String> {
    if text.len() > descriptor.limits.maximum_completion_bytes {
        return Err("conversation resource limit: completion exceeds the byte limit".into());
    }
    let descriptor = Arc::new(descriptor.clone());
    let limits = DecoderLimitsV1::declared(&descriptor);
    let mut decoder = CompletionDecoderV1::new(descriptor, mode, limits);
    decoder
        .push(GeneratedTokenV1::new(0, u32::MAX, text.as_bytes().to_vec()))?
        .into_iter()
        .find_map(|update| match update {
            CompletionUpdateV1::Finished(done) => done.completion,
            _ => None,
        })
        .ok_or_else(|| "malformed completion: missing completion end token".into())
}

#[test]
fn a_whole_completion_parse_answers_as_one_decoder_token() {
    type Change = fn(&mut FrontendDescriptorV1);
    let changes: [(&str, Change); 5] = [
        ("declared", |_| {}),
        ("no generated tokens", |d| {
            d.limits.maximum_generated_tokens = 0
        }),
        ("one completion byte", |d| {
            d.limits.maximum_completion_bytes = 1
        }),
        ("whole-text token is a boundary", |d| {
            d.vocabulary.boundary_tokens.push(FrontendBoundaryTokenV1 {
                text: "<sentinel>".into(),
                id: u32::MAX,
            });
        }),
        ("whole-text token is a stop", |d| {
            d.vocabulary.stop_token_ids.push(u32::MAX);
        }),
    ];
    for (form, declared, close) in forms() {
        let texts = [
            (String::new(), PromptModeV1::Chat),
            ("answer[EOS]".to_owned(), PromptModeV1::Chat),
            ("answer".to_owned(), PromptModeV1::Chat),
            ("answer\u{fffd}".to_owned(), PromptModeV1::Chat),
            ("[EOS]trailing".to_owned(), PromptModeV1::Chat),
            (format!("plan{close}answer[EOS]"), PromptModeV1::Thinking),
            ("plan".to_owned(), PromptModeV1::Thinking),
            ("<sentinel>".to_owned(), PromptModeV1::Chat),
        ];
        for (change, apply) in changes {
            let mut descriptor = declared.clone();
            apply(&mut descriptor);
            for (text, mode) in &texts {
                assert_eq!(
                    parse_completion_v1(&descriptor, text, *mode),
                    one_token(&descriptor, text, *mode),
                    "{form}, {change}, {text:?}"
                );
            }
        }
    }
}
