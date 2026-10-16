#![forbid(unsafe_code)]
//! A host links no chat format: without test support, a decoder given no
//! bundle frontend to parse with refuses the completion rather than reading
//! it some built-in way.
#![cfg(not(feature = "test-support"))]

use std::sync::Arc;

use frontend_format::test_support::fixtures as common;
use runtime_frontend::{
    CompletionDecoderV1, DecoderLimitsV1, FrontendDescriptorV1, GeneratedTokenV1, PromptModeV1,
};

#[test]
fn a_decoder_without_a_bundle_frontend_refuses_the_completion() {
    let descriptor =
        Arc::new(FrontendDescriptorV1::from_component_bytes(common::KEY_VALUE.as_bytes()).unwrap());
    let mut decoder = CompletionDecoderV1::new(
        Arc::clone(&descriptor),
        PromptModeV1::Chat,
        DecoderLimitsV1::declared(&descriptor),
    );
    let error = decoder
        .push(GeneratedTokenV1::new(0, 200, b"hello".to_vec()))
        .unwrap_err();
    assert_eq!(
        error,
        "invalid conversation sequence: the decoder has no bundle frontend to parse the \
         completion with"
    );
}
