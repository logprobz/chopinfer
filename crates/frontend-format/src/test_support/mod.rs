//! Fixtures and readers for tests of chat formats and their hosts.
pub mod chat;
mod fixture_format;
pub mod forced_literals;
pub mod json_reader;

pub use fixture_format::FixtureFormatV1;

/// The `format.parser` declarations that format and host tests are driven
/// by, one on each shared tool grammar.
#[path = "../../tests/common/mod.rs"]
pub mod fixtures;

/// Completions in the function-parameter tool grammar with the calls they
/// carry, shared with the qualification re-parser.
pub const FUNCTION_PARAMETER_CASES: &str =
    include_str!("../../tests/function_parameter_cases.json");

use crate::descriptor::{FormatDescriptorV1, FrontendDescriptorV1};

/// `descriptor` with its format, read and unchecked, changed by `edit` and
/// written back.
pub fn edit_format(
    descriptor: &FrontendDescriptorV1,
    edit: impl FnOnce(&mut FormatDescriptorV1),
) -> FrontendDescriptorV1 {
    let mut format = FormatDescriptorV1::read(descriptor.clone()).expect("the format reads");
    edit(&mut format);
    format.into_descriptor()
}
