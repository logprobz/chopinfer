//! Helpers for the tool objects a template writes.

use crate::descriptor::FrontendJsonEncodingV1;
use crate::request::FrontendJsonV1;
use crate::toolkit::json::json_to_string_v1;
use crate::toolkit::prompt::append_v1;

/// Whether a template's truth test passes for this value, as Python reads
/// the JSON: everything but null, false, zero and empty strings, arrays and
/// objects. An integer is zero only when every digit is, however long; any
/// other number is zero when it reads as a zero double, as one too small
/// for a double does. (JSON reading refuses one too large for a double.)
#[must_use]
pub fn json_truthy(value: &FrontendJsonV1) -> bool {
    match value {
        FrontendJsonV1::Null => false,
        FrontendJsonV1::Bool(value) => *value,
        FrontendJsonV1::Number(number) => {
            let text = number.to_string();
            if text.contains(['.', 'e', 'E']) {
                text.parse::<f64>().ok().is_none_or(|value| value != 0.0)
            } else {
                text.bytes()
                    .any(|byte| byte.is_ascii_digit() && byte != b'0')
            }
        }
        FrontendJsonV1::String(text) => !text.is_empty(),
        FrontendJsonV1::Array(values) => !values.is_empty(),
        FrontendJsonV1::Object(fields) => !fields.is_empty(),
    }
}

/// Whether a client's function object is left out of the tool list: its
/// `defer_loading` passes the template's truth test ([`json_truthy`]).
#[must_use]
pub fn defers_loading_v1(fields: &[(String, FrontendJsonV1)]) -> bool {
    fields
        .iter()
        .any(|(key, value)| key == "defer_loading" && json_truthy(value))
}

/// Writes one function object as the client sent it, the way a template
/// that loops over its items writes it: its keys in order, each written raw
/// between quotes and its value as JSON in `encoding`, without `strict` and
/// `defer_loading`.
pub fn write_tool_object_v1(
    output: &mut String,
    fields: &[(String, FrontendJsonV1)],
    encoding: FrontendJsonEncodingV1,
    maximum: usize,
) -> Result<(), String> {
    append_v1(output, "{", maximum)?;
    let mut first = true;
    for (key, value) in fields {
        if key == "strict" || key == "defer_loading" {
            continue;
        }
        if !first {
            append_v1(output, ", ", maximum)?;
        }
        first = false;
        // The key is written raw between quotes, as the template does.
        append_v1(output, "\"", maximum)?;
        append_v1(output, key, maximum)?;
        append_v1(output, "\": ", maximum)?;
        append_v1(
            output,
            &json_to_string_v1(value, encoding, "tool")?,
            maximum,
        )?;
    }
    append_v1(output, "}", maximum)
}

#[cfg(test)]
mod truth_tests {
    use super::*;

    #[test]
    fn numbers_follow_the_template_truth_test_exactly() {
        let huge = format!("1{}", "0".repeat(309));
        let negative_huge = format!("-{huge}");
        for (text, truthy) in [
            ("0", false),
            ("-0", false),
            ("0.0", false),
            ("-0.0", false),
            ("0e5", false),
            ("1e-400", false),
            ("1", true),
            ("-1", true),
            ("0.5", true),
            ("18446744073709551617", true),
            (huge.as_str(), true),
            (negative_huge.as_str(), true),
            ("null", false),
            ("true", true),
            ("false", false),
            (r#""""#, false),
            (r#""0""#, true),
            (r#""false""#, true),
            ("[]", false),
            ("[0]", true),
            ("{}", false),
            (r#"{"a": 0}"#, true),
        ] {
            let value: FrontendJsonV1 = serde_json::from_str(text).unwrap();
            assert_eq!(json_truthy(&value), truthy, "{text}");
        }
    }
}
