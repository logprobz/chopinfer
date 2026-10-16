//! Bounded JSON handling for the values a chat request renders into a prompt,
//! and the refusal messages a render or parse writes.
//!
//! Requests carry JSON in three places: tool parameter schemas, a response
//! format, and the arguments of a tool call the conversation already contains.
//! All three end up inside a prompt, so each one is charged against the
//! declared limits and screened for the markers the template reserves before a
//! byte of it is written.

use crate::descriptor::{FrontendJsonEncodingV1, FrontendLimitsV1};
use crate::request::FrontendJsonV1;

/// One refusal that names a conversation's shape.
pub fn sequence_v1(message: impl Into<String>) -> String {
    format!("invalid conversation sequence: {}", message.into())
}

/// One refusal that names a field a conversation carried.
pub fn field_v1(message: impl Into<String>) -> String {
    format!("invalid conversation field: {}", message.into())
}

/// One refusal that names a bound the declaration set.
pub fn resource_v1(message: impl Into<String>) -> String {
    format!("conversation resource limit: {}", message.into())
}

/// One refusal that names a completion this model could not have produced.
pub fn malformed_v1(message: impl Into<String>) -> String {
    format!("malformed completion: {}", message.into())
}

/// One refusal that names JSON a client sent.
pub fn json_v1(message: impl Into<String>) -> String {
    format!("invalid JSON: {}", message.into())
}

/// Reads one JSON object a conversation already contains.
///
/// Tool-call arguments arrive as a string a client sent, so the byte length is
/// bounded before the parse and the result must be an object.
pub fn parse_json_object_v1(
    input: &str,
    limits: FrontendLimitsV1,
    context: &str,
) -> Result<Vec<(String, FrontendJsonV1)>, String> {
    match parse_json_value_v1(input, limits)? {
        FrontendJsonV1::Object(entries) => Ok(entries),
        _ => Err(json_v1(format!("{context} must be an object"))),
    }
}

/// Reads one bounded JSON value a client sent as a string.
pub fn parse_json_value_v1(
    input: &str,
    limits: FrontendLimitsV1,
) -> Result<FrontendJsonV1, String> {
    if input.len() > limits.maximum_json_bytes {
        return Err(resource_v1("JSON input exceeds the byte limit"));
    }
    // The ordered value type refuses duplicate keys as it deserializes, so a
    // repeated parameter never silently collapses into one. The reader also
    // bounds its own nesting, so the declared depth limit charges a value that
    // was already parsed without unbounded recursion.
    serde_json::from_str(input).map_err(|error| json_v1(error.to_string()))
}

/// Charges one JSON value against every declared bound.
///
/// Numbers must be integral because the prompt encoding writes them back as
/// their exact decimal form, and no string may carry a marker the template
/// reserves for its own structure.
pub(crate) fn validate_json_v1(
    value: &FrontendJsonV1,
    context: &str,
    limits: FrontendLimitsV1,
    reserved: &[String],
) -> Result<(), String> {
    let mut count = 0;
    let mut bytes = 0;
    validate_inner_v1(value, context, 0, &mut count, &mut bytes, limits, reserved)
}

#[allow(clippy::too_many_arguments)]
fn validate_inner_v1(
    value: &FrontendJsonV1,
    context: &str,
    depth: usize,
    count: &mut usize,
    bytes: &mut usize,
    limits: FrontendLimitsV1,
    reserved: &[String],
) -> Result<(), String> {
    if depth > limits.maximum_json_depth {
        return Err(resource_v1(format!(
            "{context} exceeds the JSON depth limit"
        )));
    }
    *count = count
        .checked_add(1)
        .ok_or_else(|| resource_v1("JSON value count overflow"))?;
    if *count > limits.maximum_json_values {
        return Err(resource_v1(format!("{context} has too many JSON values")));
    }
    match value {
        FrontendJsonV1::Number(number) => integral_v1(number, context).map(|_| ()),
        FrontendJsonV1::String(text) => {
            add_bytes_v1(bytes, text.len(), context, limits)?;
            reserved_free_v1(text, context, reserved)
        }
        FrontendJsonV1::Array(values) => {
            for value in values {
                validate_inner_v1(value, context, depth + 1, count, bytes, limits, reserved)?;
            }
            Ok(())
        }
        FrontendJsonV1::Object(entries) => {
            for (name, value) in entries {
                add_bytes_v1(bytes, name.len(), context, limits)?;
                reserved_free_v1(name, context, reserved)?;
                validate_inner_v1(value, context, depth + 1, count, bytes, limits, reserved)?;
            }
            Ok(())
        }
        FrontendJsonV1::Null | FrontendJsonV1::Bool(_) => Ok(()),
    }
}

fn add_bytes_v1(
    bytes: &mut usize,
    amount: usize,
    context: &str,
    limits: FrontendLimitsV1,
) -> Result<(), String> {
    *bytes = bytes
        .checked_add(amount)
        .ok_or_else(|| resource_v1("JSON byte count overflow"))?;
    if *bytes > limits.maximum_json_bytes {
        return Err(resource_v1(format!(
            "{context} exceeds the JSON byte limit"
        )));
    }
    Ok(())
}

/// Refuses one string that carries a marker the template reserves.
pub(crate) fn reserved_free_v1(
    value: &str,
    context: &str,
    reserved: &[String],
) -> Result<(), String> {
    match reserved
        .iter()
        .find(|marker| value.contains(marker.as_str()))
    {
        Some(marker) => Err(field_v1(format!(
            "{context} contains reserved token or boundary '{marker}'"
        ))),
        None => Ok(()),
    }
}

/// Every number a rendered prompt may carry is integral.
fn integral_v1(number: &serde_json::Number, context: &str) -> Result<String, String> {
    if let Some(value) = number.as_i64() {
        return Ok(value.to_string());
    }
    if let Some(value) = number.as_u64() {
        return Ok(value.to_string());
    }
    let text = number.to_string();
    if !text.contains(['.', 'e', 'E']) {
        return Ok(text);
    }
    Err(json_v1(format!(
        "{context} carries a floating-point value, which this model's prompt encoding does not \
         represent"
    )))
}

/// Refuses every non-integral number one request value carries.
///
/// The transport hands the frontend whatever a client sent, and a value that
/// cannot be written back into the prompt has to be refused against the
/// parameter it arrived under rather than deep inside a render.
pub fn reject_non_integral_numbers_v1(value: &FrontendJsonV1, context: &str) -> Result<(), String> {
    match value {
        FrontendJsonV1::Number(number) => integral_v1(number, context).map(|_| ()),
        FrontendJsonV1::Array(values) => values
            .iter()
            .try_for_each(|value| reject_non_integral_numbers_v1(value, context)),
        FrontendJsonV1::Object(entries) => entries
            .iter()
            .try_for_each(|(_, value)| reject_non_integral_numbers_v1(value, context)),
        FrontendJsonV1::Null | FrontendJsonV1::Bool(_) | FrontendJsonV1::String(_) => Ok(()),
    }
}

/// Writes one JSON value in the encoding this model's prompt uses.
pub fn write_json_v1(
    value: &FrontendJsonV1,
    encoding: FrontendJsonEncodingV1,
    context: &str,
    output: &mut String,
) -> Result<(), String> {
    match value {
        FrontendJsonV1::Null => output.push_str("null"),
        FrontendJsonV1::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        FrontendJsonV1::Number(number) => match encoding {
            FrontendJsonEncodingV1::SpacedJsonV1 => {
                output.push_str(&integral_v1(number, context)?);
            }
            FrontendJsonEncodingV1::PythonJsonV1 => output.push_str(&python_number_v1(number)),
        },
        FrontendJsonV1::String(value) => write_string_v1(value, output),
        FrontendJsonV1::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push_str(", ");
                }
                write_json_v1(value, encoding, context, output)?;
            }
            output.push(']');
        }
        FrontendJsonV1::Object(entries) => {
            output.push('{');
            for (index, (name, value)) in entries.iter().enumerate() {
                if index != 0 {
                    output.push_str(", ");
                }
                write_string_v1(name, output);
                output.push_str(": ");
                write_json_v1(value, encoding, context, output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

/// Python uses the same shortest round-trip digits, with different notation:
/// scientific below 1e-4 or at 1e16, signed exponents with at least two digits,
/// and a decimal point for integral floats.
pub(crate) fn python_number_v1(number: &serde_json::Number) -> String {
    let text = number.to_string();
    if !text.contains(['.', 'e', 'E']) {
        return text;
    }
    // Arbitrary precision preserves the input spelling. Floats still use
    // Python's binary64 value and its shortest round-trip digits.
    let text = serde_json::to_string(&number.as_f64().expect("finite JSON float"))
        .expect("finite floats serialize");
    let (sign, magnitude) = text
        .strip_prefix('-')
        .map_or(("", text.as_str()), |s| ("-", s));
    let (mantissa, exponent) = magnitude.split_once('e').map_or((magnitude, 0), |(m, e)| {
        (m, e.parse::<i32>().expect("JSON number exponent"))
    });
    let point = mantissa.find('.').unwrap_or(mantissa.len()) as i32;
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let leading = digits.bytes().take_while(|b| *b == b'0').count();
    let digits = digits[leading..].trim_end_matches('0');
    if digits.is_empty() {
        return format!("{sign}0.0");
    }
    let exponent = exponent + point - leading as i32 - 1;
    if !(-4..16).contains(&exponent) {
        let fraction = if digits.len() > 1 {
            format!(".{}", &digits[1..])
        } else {
            String::new()
        };
        return format!("{sign}{}{fraction}e{exponent:+03}", &digits[..1]);
    }
    let point = exponent + 1;
    if point <= 0 {
        format!("{sign}0.{}{digits}", "0".repeat((-point) as usize))
    } else if point as usize >= digits.len() {
        format!(
            "{sign}{digits}{}.0",
            "0".repeat(point as usize - digits.len())
        )
    } else {
        format!(
            "{sign}{}.{}",
            &digits[..point as usize],
            &digits[point as usize..]
        )
    }
}

/// Renders one JSON value into a fresh string.
pub fn json_to_string_v1(
    value: &FrontendJsonV1,
    encoding: FrontendJsonEncodingV1,
    context: &str,
) -> Result<String, String> {
    let mut output = String::new();
    write_json_v1(value, encoding, context, &mut output)?;
    Ok(output)
}

fn write_string_v1(value: &str, output: &mut String) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{0008}' => output.push_str("\\b"),
            '\u{000c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            '\u{0000}'..='\u{001f}' => {
                use std::fmt::Write;
                write!(output, "\\u{:04x}", u32::from(character))
                    .expect("writing to String cannot fail");
            }
            value => output.push(value),
        }
    }
    output.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_float_notation_matches_json_dumps() {
        for (input, expected) in [
            ("-0", "0"),
            ("1e0", "1.0"),
            ("1.23000", "1.23"),
            ("18446744073709551617", "18446744073709551617"),
            (
                "1234567890123456789012345678901234567890",
                "1234567890123456789012345678901234567890",
            ),
            ("1e-5", "1e-05"),
            ("1e-7", "1e-07"),
            ("1e16", "1e+16"),
            ("1e20", "1e+20"),
            ("1e15", "1000000000000000.0"),
            ("0.0001", "0.0001"),
            ("-0.0", "-0.0"),
            ("1.0", "1.0"),
            ("5e-324", "5e-324"),
            ("1.7976931348623157e308", "1.7976931348623157e+308"),
        ] {
            let value = serde_json::from_str(input).unwrap();
            assert_eq!(
                json_to_string_v1(&value, FrontendJsonEncodingV1::PythonJsonV1, "test").unwrap(),
                expected
            );
        }
    }
}
