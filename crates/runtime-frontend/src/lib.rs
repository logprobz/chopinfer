#![forbid(unsafe_code)]

//! The model-independent frontend an installed bundle declares.
//!
//! `runtime-server` owns the OpenAI transport and names no model.
//! `runtime-host` owns one installed bundle and names no model either. This
//! crate is what joins them: it reads one verified component a bundle ships,
//! and turns that declaration into the [`ServerFrontendV1`] the transport
//! serves.
//!
//! Everything a served model decides is data in that component: the identities
//! `/v1/models` lists, the context limit, the reasoning-control vocabulary, the
//! bytes a tool name may hold, and the token identifiers its vocabulary pins.
//! Its chat format is the component's `format` object, which this crate never
//! reads: the format implementation does, and that is the package's own code.
//! Its native module exports it (`chop.frontend.v1`); this crate serves the
//! export through its documents and checks its answers. No format is linked
//! into a host: the `frontend-format` library joins only test builds
//! (`loopback.rs`, behind `test-support`). So an unchanged transport and an
//! unchanged host serve a different model by installing a different bundle,
//! with no rebuild.
//!
//! There is no discovery, no hot reload, and no migration. The caller hands
//! over bytes it already authenticated against the installed bundle's
//! manifest, and this crate reads nothing else.
//!
//! [`ServerFrontendV1`]: runtime_server::ServerFrontendV1

mod checks;
mod decode;
mod descriptor;
mod format;
#[cfg(any(test, feature = "test-support"))]
mod loopback;
mod native;
mod serve;
mod tokenizer;

pub use decode::{
    AssistantCompletionV1, CompletedCompletionV1, CompletionDecoderV1, CompletionFinishV1,
    CompletionUpdateV1, DecoderLimitsV1, GeneratedTokenV1, IncrementalTextV1,
    IncrementalToolCallsV1, ParsedToolCallV1, PromptModeV1, TokenContributionV1,
};
pub use descriptor::{
    AUDIO_MAXIMUM_SECONDS_V1, AUDIO_SAMPLE_RATE_V1, FRONTEND_COMPONENT_MAX_BYTES,
    FRONTEND_COMPONENT_ROLE_V1, FRONTEND_FORMAT_MAX_BYTES, FRONTEND_OPENAI_V2_SCHEMA,
    FRONTEND_OPENAI_V2_VERSION, FrontendBaseTokensV1, FrontendBoundaryTokenV1,
    FrontendByteLevelPolicyV1, FrontendComponentErrorV1, FrontendConfigFieldV1,
    FrontendConfigTokenValueV1, FrontendConfigTokensV1, FrontendContextPolicyV1,
    FrontendDescriptorV1, FrontendEffortUseV1, FrontendFileIdentityV1, FrontendIgnoreMergesV1,
    FrontendLimitsV1, FrontendMediaArenaV1, FrontendMediaElementV1, FrontendMediaEncoderV1,
    FrontendMediaGridRenderV1, FrontendMediaKindV1, FrontendMediaLayoutV1, FrontendMediaModalityV1,
    FrontendMediaRecipeV1, FrontendMediaRenderV1, FrontendMergeRecordsV1, FrontendMergeResultIdsV1,
    FrontendModelDeclarationV1, FrontendMultimodalV1, FrontendNameBytesV1, FrontendPositionRuleV1,
    FrontendReasoningEffortV1, FrontendReasoningFieldV1, FrontendReasoningPolicyV1,
    FrontendResponseV1, FrontendTokenizerNormalizerV1, FrontendTokenizerPolicyV1,
    FrontendTokenizerV1, FrontendToolArgumentsV1, FrontendToolPolicyV1, FrontendVocabularyV1,
    TOKENIZER_CONFIGURATION_ROLE_V1, TOKENIZER_DEFINITION_ROLE_V1,
    audio_log_mel_v1_fixed_parameters,
};
#[cfg(any(test, feature = "test-support"))]
pub use loopback::parse_completion_v1;
pub use native::BundleParserSourceV1;
pub use serve::{BundleChatStateV1, BundleFrontendV1};
pub use tokenizer::{tokenizer_contract_v1, tokenizer_dialect_v1, tokenizer_from_components_v1};

#[cfg(any(test, feature = "test-support"))]
pub mod test_support;

#[cfg(test)]
mod document_tests;
