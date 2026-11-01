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
