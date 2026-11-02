//! The host's refusal wording and the checks it applies itself.
//!
//! A decoder refusal names one family, and a bundle's format words its own
//! refusals in the same families, so the host and the format read alike. The host keeps its own copy of the wording and of the checks it
//! makes without the format: a tool name against the declared byte set, a
//! backend tool call's arguments as bounded JSON of the declared shape, and
//! the boundary tokens a declared text spells.

use std::ops::Range;

use runtime_server::FrontendJsonV1;

use crate::descriptor::{FrontendDescriptorV1, FrontendLimitsV1, FrontendVocabularyV1};

/// The refusal families a decoder reports. A parser refusal in one of them
/// passes through as the format worded it; any other is malformed.
pub(crate) const DECODER_FAMILIES_V1: [&str; 5] = [
    "malformed completion: ",
    "invalid conversation sequence: ",
    "conversation resource limit: ",
    "invalid conversation field: ",
    "invalid JSON: ",
];

/// Fixed refusal texts permitted in request-failure lines. New literals
/// in the decoder or shared parser need an explicit addition here.
const LOGGED_REFUSALS_V1: &[&str] = &[
    "conversation resource limit: completion byte limit is invalid",
    "conversation resource limit: completion exceeds the byte limit",
    "conversation resource limit: completion marks allocation failed",
    "conversation resource limit: completion token limit is invalid",
    "conversation resource limit: generated token ordinal overflow",
    "conversation resource limit: incremental completion byte length overflow",
    "conversation resource limit: incremental completion exceeds the byte limit",
    "conversation resource limit: incremental completion exceeds the token limit",
    "conversation resource limit: incremental token count overflow",
    "conversation resource limit: too many completion tool calls",
    "conversation resource limit: too many completion tool parameters",
    "invalid conversation field: client stop strings must be nonempty and distinct",
    "invalid conversation sequence: a completion prefix must precede generation",
    "invalid conversation sequence: a push carries marks, and the declaration names no ID-only markers",
    "invalid conversation sequence: a push keeps bytes the parser never received",
    "invalid conversation sequence: a push with marks follows one without",
    "invalid conversation sequence: a push without marks follows one with marks",
    "invalid conversation sequence: bytes pushed after the format parser failed",
    "invalid conversation sequence: bytes pushed after the format parser finished",
    "invalid conversation sequence: decoding options set after decoding began",
    "invalid conversation sequence: engine and caller tokens mixed in one completion",
    "invalid conversation sequence: parser source set after decoding began",
    "invalid conversation sequence: prompt context must precede generation",
    "invalid conversation sequence: reasoning state set after decoding began",
    "invalid conversation sequence: the decoder has no bundle frontend to parse the completion with",
    "invalid conversation sequence: the marks of the kept bytes are not the marks the parser kept",
    "invalid conversation sequence: tool extraction set after decoding began",
    "invalid conversation sequence: tool schemas set after decoding began",
    "malformed completion: bytes received after terminal completion",
    "malformed completion: completion contains an unexpected reserved boundary",
    "malformed completion: decoder panicked",
    "malformed completion: generated token has no decoded bytes",
    "malformed completion: length finish received after terminal completion",
    "malformed completion: missing completion end token",
    "malformed completion: missing tool start",
    "malformed completion: special token ID has mismatched decoded bytes",
    "malformed completion: streamed tool arguments changed after confirmation",
    "malformed completion: structural bytes cannot be emitted as text",
    "malformed completion: the format delivered an empty tool-call group",
    "malformed completion: the format delivered more tool calls than the declaration allows",
    "malformed completion: the format delivered tool calls the request does not read",
    "malformed completion: the format delivered tool-call arguments that are not a JSON object",
    "malformed completion: the format did not finish on the final push",
    "malformed completion: the format finished before reading every released byte",
    "malformed completion: the format finished before the end of the released bytes",
    "malformed completion: the format finished with a streamed tool call it never delivered",
    "malformed completion: the format moved its cursor back or past the released bytes",
    "malformed completion: the format moved its cursor inside a UTF-8 character or ill-formed byte run",
    "malformed completion: the format named a tool call on other than its first delta",
    "malformed completion: the format named a tool outside the declared name charset",
    "malformed completion: the format reported a range that overlaps an earlier one or passes the released bytes",
    "malformed completion: the format reported a range that splits a UTF-8 character or ill-formed byte run",
    "malformed completion: the format reported an event after finishing",
    "malformed completion: the format reported reasoning in chat mode",
    "malformed completion: the format streamed a delta outside the open tool calls",
    "malformed completion: the format streamed tool calls the request does not stream",
    "malformed completion: the format's completion disagrees with the calls it delivered",
    "malformed completion: the format's finish reason disagrees with the calls it delivered",
    "malformed completion: the format's tool-call deltas do not add up to the delivered call",
    "malformed completion: token received after terminal completion",
    "malformed completion: trailing bytes after the completion end",
];

/// A known fixed refusal, or only its family, with no prompt or output text.
#[must_use]
pub fn loggable_refusal_v1(message: &str) -> String {
    if LOGGED_REFUSALS_V1.contains(&message) {
        return message.to_owned();
    }
    DECODER_FAMILIES_V1
        .iter()
        .find(|family| message.starts_with(**family))
        .map_or_else(
            || "unrecognized refusal".to_owned(),
            |family| format!("{family}refusal text withheld"),
        )
}

/// One refusal that names a conversation's shape.
pub(crate) fn sequence_v1(message: impl Into<String>) -> String {
    format!("invalid conversation sequence: {}", message.into())
}

/// One refusal that names a field a conversation carried.
pub(crate) fn field_v1(message: impl Into<String>) -> String {
    format!("invalid conversation field: {}", message.into())
}

/// One refusal that names a bound the declaration set.
pub(crate) fn resource_v1(message: impl Into<String>) -> String {
    format!("conversation resource limit: {}", message.into())
}

/// One refusal that names a completion this model could not have produced.
pub(crate) fn malformed_v1(message: impl Into<String>) -> String {
    format!("malformed completion: {}", message.into())
}

/// One refusal that names JSON a client sent.
fn json_v1(message: impl Into<String>) -> String {
    format!("invalid JSON: {}", message.into())
}

/// Reads one bounded JSON value a client sent as a string.
pub(crate) fn parse_json_value_v1(
    input: &str,
    limits: FrontendLimitsV1,
) -> Result<FrontendJsonV1, String> {
    if input.len() > limits.maximum_json_bytes {
        return Err(resource_v1("JSON input exceeds the byte limit"));
    }
    // The ordered value type refuses duplicate keys as it deserializes, so a
    // repeated parameter never silently collapses into one.
    serde_json::from_str(input).map_err(|error| json_v1(error.to_string()))
}

/// Whether every number `value` carries is integral, as arguments declared
/// `json-object-integral-numbers` must be: written without a fraction or an
/// exponent, or read exactly as an integer.
pub(crate) fn integral_numbers_v1(value: &FrontendJsonV1) -> bool {
    match value {
        FrontendJsonV1::Number(number) => {
            number.as_i64().is_some()
                || number.as_u64().is_some()
                || !number.to_string().contains(['.', 'e', 'E'])
        }
        FrontendJsonV1::Array(values) => values.iter().all(integral_numbers_v1),
        FrontendJsonV1::Object(entries) => {
            entries.iter().all(|(_, value)| integral_numbers_v1(value))
        }
        FrontendJsonV1::Null | FrontendJsonV1::Bool(_) | FrontendJsonV1::String(_) => true,
    }
}

/// Checks one function name: a name as [`validate_name_v1`] checks it, or,
/// where the declaration qualifies names (`tools.namespaces`), a namespace
/// and a name joined by the separator, each part such a name, so the whole
/// fits `maximum_function_name_bytes`.
pub(crate) fn validate_function_name_v1(
    descriptor: &FrontendDescriptorV1,
    name: &str,
    context: &str,
) -> Result<(), String> {
    match descriptor.tools.split_function_name(name) {
        (Some(namespace), bare) => {
            validate_name_v1(descriptor, namespace, context)?;
            validate_name_v1(descriptor, bare, context)
        }
        (None, _) => validate_name_v1(descriptor, name, context),
    }
}

/// Checks one name against the byte set and length this model declared.
pub(crate) fn validate_name_v1(
    descriptor: &FrontendDescriptorV1,
    name: &str,
    context: &str,
) -> Result<(), String> {
    let charset = &descriptor.tools.name_bytes;
    let maximum = descriptor.limits.maximum_name_bytes;
    if name.is_empty() || name.len() > maximum || !name.bytes().all(|byte| charset.permits(byte)) {
        return Err(field_v1(format!(
            "{context} name must match {} and fit {maximum} bytes",
            charset.describe()
        )));
    }
    Ok(())
}

/// `text` cut at the declared boundary tokens it spells, the longest one
/// first at each position: each piece is a byte range of `text`, and
/// whether it is one boundary token's spelling. The text between two of
/// them is one piece.
pub(crate) fn boundary_pieces_v1<'a>(
    text: &'a str,
    vocabulary: &'a FrontendVocabularyV1,
) -> impl Iterator<Item = (Range<usize>, bool)> + 'a {
    let (mut at, mut plain) = (0, 0);
    let mut token: Option<Range<usize>> = None;
    std::iter::from_fn(move || {
        if let Some(token) = token.take() {
            return Some((token, true));
        }
        while at < text.len() {
            let longest = vocabulary
                .boundary_tokens
                .iter()
                .map(|boundary| boundary.text.as_str())
                .filter(|spelling| !spelling.is_empty() && text[at..].starts_with(spelling))
                .map(str::len)
                .max();
            if let Some(length) = longest {
                let run = plain..at;
                at += length;
                plain = at;
                if run.is_empty() {
                    return Some((run.end..at, true));
                }
                token = Some(run.end..at);
                return Some((run, false));
            }
            at += text[at..].chars().next().map_or(1, char::len_utf8);
        }
        (plain < text.len()).then(|| {
            let run = plain..text.len();
            plain = text.len();
            (run, false)
        })
    })
}
