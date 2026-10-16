//! Python's `repr` and `str` of JSON values, for messages that must read as
//! the Python tooling writes them (the conformance vectors match their
//! text).

use std::fmt::Write as _;

use serde_json::Value;

use crate::json::python_float_repr;

/// Python's `repr` of a JSON value as `json.loads` reads it.
pub fn repr(value: &Value) -> String {
    let mut out = String::new();
    write_repr(value, &mut out);
    out
}

/// Python's `str`: a string as itself, anything else as its `repr`.
pub fn str(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => repr(other),
    }
}

/// Python's `str` of an optional value: `None` when absent.
pub fn str_of(value: Option<&Value>) -> String {
    value.map_or_else(|| "None".into(), str)
}

/// Python's `type(value).__name__` of a JSON value as `json.loads` reads it.
pub fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(number) if number.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

fn write_repr(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("None"),
        Value::Bool(true) => out.push_str("True"),
        Value::Bool(false) => out.push_str("False"),
        Value::Number(number) => {
            if let Some(value) = number.as_u64() {
                let _ = write!(out, "{value}");
            } else if let Some(value) = number.as_i64() {
                let _ = write!(out, "{value}");
            } else if let Some(value) = number.as_f64() {
                out.push_str(&python_float_repr(value));
            }
        }
        Value::String(text) => write_string(text, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_repr(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_string(key, out);
                out.push_str(": ");
                write_repr(item, out);
            }
            out.push('}');
        }
    }
}

/// A string literal as Python writes it: single quotes unless the text
/// holds a single quote and no double quote.
fn write_string(text: &str, out: &mut String) {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    out.push(quote);
    for character in text.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ if character == quote => {
                out.push('\\');
                out.push(character);
            }
            _ if (character as u32) < 0x20 || (0x7f..0xa0).contains(&(character as u32)) => {
                let _ = write!(out, "\\x{:02x}", character as u32);
            }
            _ => out.push(character),
        }
    }
    out.push(quote);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn values_read_as_python_writes_them() {
        assert_eq!(repr(&json!(null)), "None");
        assert_eq!(
            repr(&json!([true, false, 1, -2, 0.5, 1.0])),
            "[True, False, 1, -2, 0.5, 1.0]"
        );
        assert_eq!(repr(&json!(["chat", "code"])), "['chat', 'code']");
        assert_eq!(repr(&json!("it's")), "\"it's\"");
        assert_eq!(repr(&json!("a'b\"c\n")), "'a\\'b\"c\\n'");
        assert_eq!(repr(&json!({"a": 1, "b": [null]})), "{'a': 1, 'b': [None]}");
        assert_eq!(str(&json!("text")), "text");
        assert_eq!(str(&json!(100000)), "100000");
        assert_eq!(str(&json!(0.16666666666666666)), "0.16666666666666666");
        assert_eq!(str_of(None), "None");
    }
}
