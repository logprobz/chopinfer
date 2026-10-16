#![forbid(unsafe_code)]

//! A model's chat format, as a library a bundle exports.
//!
//! A model renders its prompt in Rust: a [`ChatFormatV1`] ported from its
//! chat template with the [`toolkit`]. Its frontend component's `format`
//! object carries only the `format.parser` section
//! ([`FrontendParserDeclarationV1`]) the shared parsers read, which
//! [`FormatDescriptorV1`] reads and checks. [`ChatFormatExportV1`] serves the
//! format and the parser section over the `chop.frontend.v1` documents: it
//! renders through the format, builds the grammar a tool choice imposes, and
//! parses completions back into reasoning, answer text and tool calls in
//! one of the shared tool-call grammars ([`FrontendToolGrammarV1`]). A
//! bundle exports it with [`export_chat_format_v1!`].
//!
//! The library knows nothing of a host. It reads only the component and the
//! documents a host sends, so it links no transport, engine or device code.
//! Model tests and benchmarks call [`tool_constraint_v1`],
//! [`parse_completion_v1`] and [`DeclaredParserV1`] directly.

mod chat;
mod chat_format;
mod chatml_decode;
mod descriptor;
mod export;
mod format;
mod format_decode;
mod request;
mod tool_grammar;
pub mod toolkit;

pub use chat::{AssistantTurnV1, ChatMessageV1, ChatRenderV1, PromptModeV1, ToolResultV1};
pub use chat_format::{ChatFormatExportV1, ChatFormatV1, chat_format_export_v1};
pub use descriptor::{
    FormatDescriptorV1, FrontendEffortUseV1, FrontendFormatV1, FrontendJsonEncodingV1,
    FrontendMediaKindV1, FrontendParserDeclarationV1, FrontendToolGrammarV1,
};
pub use export::DeclaredExportParserV1;
pub use format::{
    AssistantCompletionV1, CompletionFinishV1, FormatEventV1, FormatInputV1, FormatParserOptionsV1,
    FormatRenderRequestV1, FormatRenderedV1, FormatToolCallDeltaV1, ParsedToolCallV1,
};
pub use format_decode::{DeclaredParserV1, parse_completion_v1};
pub use request::{
    FrontendContentPartV1, FrontendErrorV1, FrontendJsonV1, FrontendMediaPartKindV1,
    FrontendMediaPartV1, FrontendMessageV1, FrontendRoleV1, FrontendToolCallV1,
    FrontendToolChoiceV1, FrontendToolV1, PromptSegmentV1, TOOL_REFERENCES_UNSUPPORTED_V1,
};
pub use tool_grammar::tool_constraint_v1;

/// What a format's render reads of a request besides the common fields:
/// the open answer [`ChatFormatV1::opened`] returns.
pub use runtime_bundle_abi::frontend_document::OpenedV1;

#[cfg(any(test, feature = "test-support"))]
pub mod test_support;

#[doc(hidden)]
pub use runtime_bundle_abi as __runtime_bundle_abi;

/// Exports a model's chat format from a bundle as `chop.frontend.v1`: the
/// `chop_bundle_frontend_api_v1` table over [`ChatFormatExportV1`] of the
/// named [`ChatFormatV1`].
///
/// Invoke it once, at the bundle's crate root:
///
/// ```
/// use frontend_format::{
///     ChatFormatV1, FormatDescriptorV1, FormatRenderRequestV1, FormatRenderedV1,
///     FrontendErrorV1, OpenedV1,
/// };
///
/// struct Format;
///
/// impl ChatFormatV1 for Format {
///     fn open(_: &FormatDescriptorV1) -> Result<Self, String> {
///         Ok(Self)
///     }
///     fn opened(&self) -> OpenedV1 {
///         OpenedV1::default()
///     }
///     fn render(
///         &self,
///         _: FormatRenderRequestV1<'_>,
///     ) -> Result<FormatRenderedV1, FrontendErrorV1> {
///         Err(FrontendErrorV1::new("not rendered", None))
///     }
/// }
///
/// frontend_format::export_chat_format_v1!(Format);
/// ```
#[macro_export]
macro_rules! export_chat_format_v1 {
    ($format:ty $(,)?) => {
        $crate::__runtime_bundle_abi::export_frontend_v1!($crate::ChatFormatExportV1<$format>);
    };
}
