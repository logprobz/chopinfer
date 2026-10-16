//! Reusable assertions and fixture readers for frontend implementations.
pub mod decode;
pub mod documents;
pub mod reasoning;
pub mod streaming;
pub mod tool_choice;

/// The test fixture format's export (`frontend_format::test_support::FixtureFormatV1`),
/// linked into this program: the loopback a test serves a `format.parser`
/// declaration with, such as the parser fixtures, where a deployment serves
/// its package's own export.
#[must_use]
pub fn fixture_export_v1() -> runtime_bundle_abi::frontend::FrontendExportV1 {
    frontend_format::chat_format_export_v1::<frontend_format::test_support::FixtureFormatV1>()
}
