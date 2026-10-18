//! The model-neutral pieces a model's own chat-format renderer builds on.
//!
//! A renderer ported from a model's chat template ([`crate::ChatFormatV1`])
//! writes its prompt with these, so every model charges the same bounds,
//! screens the same boundaries, writes JSON as Python does, and reads the
//! reasoning controls and refuses requests in the same words:
//!
//! - prompt writing under the byte bound, names and media segments
//!   ([`append_v1`], [`validate_name_v1`], [`media_placeholder_v1`],
//!   [`screen_media_text_v1`], [`media_kind_v1`], [`split_media_segments_v1`]);
//! - the conversation as a renderer reads it from a request, and the checks
//!   it passes before anything is written ([`chat_message_v1`],
//!   [`validate_chat_v1`]);
//! - bounded JSON reading and JSON writing in the declared encoding
//!   ([`parse_json_object_v1`], [`write_json_v1`], [`json_to_string_v1`]);
//! - Python's `json.dumps` under HF's `tojson` arguments, `str.strip`,
//!   `str.rstrip`, `str.split` and `str.capitalize` ([`tojson_v1`],
//!   [`python_strip_v1`], [`python_rstrip_v1`], [`python_split_v1`],
//!   [`python_capitalize_v1`]), checked against Python by
//!   `tests/python_semantics.json`;
//! - tool objects as a template writes them ([`json_truthy`],
//!   [`defers_loading_v1`], [`write_tool_object_v1`]);
//! - the reasoning controls ([`select_mode_v1`], [`preserve_thinking_v1`],
//!   [`reasoning_open_v1`]);
//! - the refusal messages ([`sequence_v1`], [`field_v1`], [`resource_v1`],
//!   [`malformed_v1`], [`json_v1`]).

mod conversation;
mod json;
mod prompt;
mod python;
mod reasoning;
mod tools;
mod unicode15;

pub use conversation::{chat_message_v1, validate_chat_v1};
#[cfg(any(test, feature = "test-support"))]
pub(crate) use json::reserved_free_v1;
pub use json::{
    field_v1, json_to_string_v1, json_v1, malformed_v1, parse_json_object_v1, parse_json_value_v1,
    reject_non_integral_numbers_v1, resource_v1, sequence_v1, write_json_v1,
};
pub use prompt::{
    append_v1, media_kind_v1, media_placeholder_v1, reserved_media_text_v1, screen_media_text_v1,
    split_media_segments_v1, validate_function_name_v1, validate_name_v1,
};
pub use python::{
    TojsonOptionsV1, python_capitalize_v1, python_rstrip_v1, python_split_v1, python_strip_v1,
    tojson_v1,
};
pub use reasoning::{preserve_thinking_v1, reasoning_open_v1, select_mode_v1};
pub use tools::{defers_loading_v1, json_truthy, write_tool_object_v1};
