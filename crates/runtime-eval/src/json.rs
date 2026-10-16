//! JSON helpers shared by the loaders: digests, the canonical encoding the
//! Python tooling hashes, and number reads with Python's meaning.

use std::cmp::Ordering;
use std::fmt::Write as _;
use std::path::Path;

use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};

/// The lowercase hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut text = String::with_capacity(64);
    for byte in digest {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

/// Whether `value` is a lowercase SHA-256 hex digest.
pub fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// A file's bytes and their parsed JSON ([`parse`]).
pub fn read(path: &Path) -> Result<(Vec<u8>, Value), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let value = parse(&bytes).map_err(|e| format!("{}: not JSON: {e}", path.display()))?;
    Ok((bytes, value))
}

/// Parses strict JSON with every float correctly rounded, as Python reads
/// it. serde_json's own float parsing can miss a 17-digit float by an ulp
/// unless its `float_roundtrip` feature is on, which would change parsing
/// for every crate linked with this one. Objects keep their key order; a
/// repeated key, a non-finite number, an integer outside 64 bits and
/// trailing content are refused.
pub fn parse(bytes: &[u8]) -> Result<Value, String> {
    let mut parser = Parser { bytes, at: 0 };
    let value = parser.value(0)?;
    parser.space();
    if parser.at != bytes.len() {
        return Err(parser.error("trailing content"));
    }
    Ok(value)
}

/// Nesting deeper than this is refused rather than risking the stack.
const MAX_DEPTH: usize = 512;

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn error(&self, message: &str) -> String {
        format!("{message} at byte {}", self.at)
    }

    fn space(&mut self) {
        while matches!(self.bytes.get(self.at), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
        }
    }

    fn expect(&mut self, literal: &[u8]) -> Result<(), String> {
        if self.bytes[self.at..].starts_with(literal) {
            self.at += literal.len();
            Ok(())
        } else {
            Err(self.error("unexpected character"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, String> {
        if depth > MAX_DEPTH {
            return Err(self.error("nesting too deep"));
        }
        self.space();
        match self.bytes.get(self.at) {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => self.string().map(Value::String),
            Some(b't') => self.expect(b"true").map(|()| Value::Bool(true)),
            Some(b'f') => self.expect(b"false").map(|()| Value::Bool(false)),
            Some(b'n') => self.expect(b"null").map(|()| Value::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(self.error("expected a value")),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value, String> {
        self.at += 1;
        let mut map = Map::new();
        self.space();
        if self.bytes.get(self.at) == Some(&b'}') {
            self.at += 1;
            return Ok(Value::Object(map));
        }
        loop {
            self.space();
            if self.bytes.get(self.at) != Some(&b'"') {
                return Err(self.error("expected a key"));
            }
            let key = self.string()?;
            self.space();
            self.expect(b":")?;
            let value = self.value(depth + 1)?;
            if map.contains_key(&key) {
                return Err(self.error(&format!("repeated key {key:?}")));
            }
            map.insert(key, value);
            self.space();
            match self.bytes.get(self.at) {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Value::Object(map));
                }
                _ => return Err(self.error("expected , or }")),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Value, String> {
        self.at += 1;
        let mut items = Vec::new();
        self.space();
        if self.bytes.get(self.at) == Some(&b']') {
            self.at += 1;
            return Ok(Value::Array(items));
        }
        loop {
            items.push(self.value(depth + 1)?);
            self.space();
            match self.bytes.get(self.at) {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Value::Array(items));
                }
                _ => return Err(self.error("expected , or ]")),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        // Exactly four hex digits: from_str_radix alone would take a sign.
        let digits = self
            .bytes
            .get(self.at..self.at + 4)
            .filter(|digits| digits.iter().all(u8::is_ascii_hexdigit))
            .and_then(|digits| std::str::from_utf8(digits).ok())
            .and_then(|digits| u32::from_str_radix(digits, 16).ok())
            .ok_or_else(|| self.error("invalid \\u escape"))?;
        self.at += 4;
        Ok(digits)
    }

    fn string(&mut self) -> Result<String, String> {
        self.at += 1;
        let mut text = Vec::new();
        loop {
            let byte = *self
                .bytes
                .get(self.at)
                .ok_or_else(|| self.error("unterminated string"))?;
            self.at += 1;
            match byte {
                b'"' => break,
                b'\\' => {
                    let escape = *self
                        .bytes
                        .get(self.at)
                        .ok_or_else(|| self.error("unterminated escape"))?;
                    self.at += 1;
                    let character = match escape {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let first = self.hex4()?;
                            let code = if (0xd800..0xdc00).contains(&first) {
                                self.expect(b"\\u")?;
                                let second = self.hex4()?;
                                if !(0xdc00..0xe000).contains(&second) {
                                    return Err(self.error("unpaired surrogate"));
                                }
                                0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00)
                            } else {
                                first
                            };
                            char::from_u32(code).ok_or_else(|| self.error("unpaired surrogate"))?
                        }
                        _ => return Err(self.error("invalid escape")),
                    };
                    let mut buffer = [0; 4];
                    text.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
                }
                0..0x20 => return Err(self.error("control character in a string")),
                _ => text.push(byte),
            }
        }
        String::from_utf8(text).map_err(|_| self.error("invalid UTF-8"))
    }

    fn number(&mut self) -> Result<Value, String> {
        let start = self.at;
        let digits = |parser: &mut Self| {
            let from = parser.at;
            while parser.bytes.get(parser.at).is_some_and(u8::is_ascii_digit) {
                parser.at += 1;
            }
            parser.at - from
        };
        if self.bytes[self.at] == b'-' {
            self.at += 1;
        }
        let whole_start = self.at;
        let whole = digits(self);
        if whole == 0 || (whole > 1 && self.bytes[whole_start] == b'0') {
            return Err(self.error("invalid number"));
        }
        let mut float = false;
        if self.bytes.get(self.at) == Some(&b'.') {
            self.at += 1;
            float = true;
            if digits(self) == 0 {
                return Err(self.error("invalid number"));
            }
        }
        if matches!(self.bytes.get(self.at), Some(b'e' | b'E')) {
            self.at += 1;
            float = true;
            if matches!(self.bytes.get(self.at), Some(b'+' | b'-')) {
                self.at += 1;
            }
            if digits(self) == 0 {
                return Err(self.error("invalid number"));
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.at]).expect("ASCII digits");
        let number = if float {
            text.parse::<f64>()
                .ok()
                .and_then(Number::from_f64)
                .filter(|number| number.as_f64().is_some_and(f64::is_finite))
        } else if let Ok(value) = text.parse::<u64>() {
            Some(Number::from(value))
        } else {
            text.parse::<i64>().ok().map(Number::from)
        };
        number
            .map(Value::Number)
            .ok_or_else(|| self.error("number out of range"))
    }
}

/// The canonical encoding of `value`, byte for byte what Python's
/// `json.dumps(value, sort_keys=True, separators=(",", ":"))` writes
/// (`tools/model_sdk/diagnostic.py` `canonical`): keys sorted by code
/// point, no whitespace, non-ASCII escaped, floats in Python's `repr`.
pub fn canonical(value: &Value) -> Vec<u8> {
    let mut out = String::new();
    write_canonical(value, &mut out, true, COMPACT);
    out.into_bytes()
}

/// The compact encoding of `value`, byte for byte what Python's
/// `json.dumps(value, separators=(",", ":"))` writes: [`canonical`] with
/// keys in their order in the document.
pub fn compact(value: &Value) -> Vec<u8> {
    let mut out = String::new();
    write_canonical(value, &mut out, false, COMPACT);
    out.into_bytes()
}

/// What Python's `json.dumps(value)` writes: [`compact`] with its default
/// separators `", "` and `": "`.
pub fn dumps(value: &Value) -> Vec<u8> {
    let mut out = String::new();
    write_canonical(value, &mut out, false, (", ", ": "));
    out.into_bytes()
}

/// The item and key separators of [`canonical`] and [`compact`].
const COMPACT: (&str, &str) = (",", ":");

/// The digest of `value`'s canonical encoding.
pub fn canonical_sha256(value: &Value) -> String {
    sha256_hex(&canonical(value))
}

/// A receipt as the Python tooling writes it: `json.dump(value, indent=2,
/// sort_keys=True)` and a newline.
pub fn receipt_text(value: &Value) -> Vec<u8> {
    let mut out = String::new();
    write_indented(value, &mut out, 0);
    out.push('\n');
    out.into_bytes()
}

fn write_indented(value: &Value, out: &mut String, depth: usize) {
    let indent = |out: &mut String, depth: usize| {
        out.push('\n');
        out.push_str(&"  ".repeat(depth));
    };
    match value {
        Value::Array(items) if !items.is_empty() => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                indent(out, depth + 1);
                write_indented(item, out, depth + 1);
            }
            indent(out, depth);
            out.push(']');
        }
        Value::Object(map) if !map.is_empty() => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                indent(out, depth + 1);
                write_string(key, out);
                out.push_str(": ");
                write_indented(&map[key], out, depth + 1);
            }
            indent(out, depth);
            out.push('}');
        }
        other => write_canonical(other, out, true, COMPACT),
    }
}

fn write_canonical(value: &Value, out: &mut String, sorted: bool, separators: (&str, &str)) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => write_number(number, out),
        Value::String(text) => write_string(text, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(separators.0);
                }
                write_canonical(item, out, sorted, separators);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            if sorted {
                // UTF-8 byte order is code point order.
                keys.sort();
            }
            out.push('{');
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    out.push_str(separators.0);
                }
                write_string(key, out);
                out.push_str(separators.1);
                write_canonical(&map[key], out, sorted, separators);
            }
            out.push('}');
        }
    }
}

fn write_number(number: &Number, out: &mut String) {
    if let Some(value) = number.as_u64() {
        let _ = write!(out, "{value}");
    } else if let Some(value) = number.as_i64() {
        let _ = write!(out, "{value}");
    } else if let Some(value) = number.as_f64() {
        out.push_str(&python_float_repr(value));
    }
}

/// Python's `round(value, digits)` of a float: the exact binary value
/// rounded to `digits` decimal places, an exact tie to even, read back as
/// the nearest double. A non-finite value is returned as it is. Rust's
/// fixed-precision formatting rounds the exact value the same way.
pub fn python_round(value: f64, digits: usize) -> f64 {
    if !value.is_finite() {
        return value;
    }
    format!("{value:.digits$}")
        .parse()
        .expect("a formatted float reads back")
}

/// Python's `repr` of a finite float: the shortest round-trip digits, fixed
/// notation for decimal exponents from -4 through 15 (with `.0` when
/// integral), else `d.ddde+XX` with at least two exponent digits.
pub fn python_float_repr(value: f64) -> String {
    let (digits, exponent) = shortest_digits(value);
    // The value is 0.DIGITS times ten to `point`.
    let point = exponent + 1;
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let count = digits.len() as i32;
    let body = if !(-4 < point && point <= 16) {
        let (first, rest) = digits.split_at(1);
        let fraction = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        let sign = if point - 1 < 0 { '-' } else { '+' };
        format!("{first}{fraction}e{sign}{:02}", (point - 1).abs())
    } else if point <= 0 {
        format!("0.{}{digits}", "0".repeat((-point) as usize))
    } else if point < count {
        let (whole, fraction) = digits.split_at(point as usize);
        format!("{whole}.{fraction}")
    } else {
        format!("{digits}{}.0", "0".repeat((point - count) as usize))
    };
    format!("{sign}{body}")
}

/// The digits and decimal exponent (of the first digit) of the shortest
/// decimal that reads back as `|value|`, chosen as Python's `repr` chooses:
/// of the shortest strings that read back, the one nearest the exact value,
/// ties to even. Rust's shortest formatting rounds an exact tie up
/// (0.173450469970703125 gives 0.17345046997070313 where Python writes
/// ...12), so its length is kept and the correctly rounded string of that
/// length taken when it reads back too.
pub(crate) fn shortest_digits(value: f64) -> (String, i32) {
    let split = |scientific: String| {
        let (mantissa, exponent) = scientific
            .split_once('e')
            .map(|(m, e)| (m.to_owned(), e.parse::<i32>().expect("an integer exponent")))
            .expect("Rust's LowerExp always writes an exponent");
        (mantissa.replace('.', ""), exponent)
    };
    let magnitude = value.abs();
    let shortest = split(format!("{magnitude:e}"));
    let nearest = format!("{magnitude:.*e}", shortest.0.len() - 1);
    if nearest.parse::<f64>() == Ok(magnitude) {
        split(nearest)
    } else {
        shortest
    }
}

fn write_string(text: &str, out: &mut String) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            // Python escapes everything outside space through tilde, DEL included.
            ' '..='~' => out.push(character),
            _ => {
                let mut units = [0u16; 2];
                for unit in character.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}

/// A JSON number as Python reads it: an int (or a bool, which Python
/// counts as one) or a float. `None` for any other value.
pub fn python_number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::Bool(flag) => Some(f64::from(u8::from(*flag))),
        _ => None,
    }
}

/// Whether `value` is a JSON float (written with a fraction or exponent),
/// which Python reads as `float` rather than `int`.
pub fn is_float(value: &Value) -> bool {
    matches!(value, Value::Number(number) if number.is_f64())
}

/// Python's truthiness of a JSON value.
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|value| value != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// Python's `==` on two JSON values: numbers (and bools) compare by value,
/// containers element by element, objects regardless of key order.
pub fn python_eq(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| python_eq(a, b))
        }
        (Value::Object(a), Value::Object(b)) => objects_eq(a, b, python_eq),
        (Value::Number(_) | Value::Bool(_), Value::Number(_) | Value::Bool(_)) => {
            numbers_eq(left, right)
        }
        _ => left == right,
    }
}

fn numbers_eq(left: &Value, right: &Value) -> bool {
    python_cmp(left, right) == Some(Ordering::Equal)
}

/// A JSON number or bool as Python holds it: an exact int or a float.
enum Real {
    Int(i128),
    Float(f64),
}

fn real(value: &Value) -> Option<Real> {
    match value {
        Value::Bool(flag) => Some(Real::Int(i128::from(*flag))),
        Value::Number(number) if number.is_f64() => number.as_f64().map(Real::Float),
        Value::Number(number) => number
            .as_i64()
            .map(i128::from)
            .or_else(|| number.as_u64().map(i128::from))
            .map(Real::Int),
        _ => None,
    }
}

/// An int against a float, exactly, as Python compares them: the int is
/// never rounded to a double first.
fn compare_int_float(int: i128, float: f64) -> Option<Ordering> {
    if float.is_nan() {
        return None;
    }
    // 2^127 bounds every i128 and is a double.
    const BOUND: f64 = 170_141_183_460_469_231_731_687_303_715_884_105_728.0;
    let whole = float.trunc();
    if whole >= BOUND {
        return Some(Ordering::Less);
    }
    if whole < -BOUND {
        return Some(Ordering::Greater);
    }
    let fraction = float - whole;
    Some(match int.cmp(&(whole as i128)) {
        Ordering::Equal if fraction > 0.0 => Ordering::Less,
        Ordering::Equal if fraction < 0.0 => Ordering::Greater,
        order => order,
    })
}

/// Python's ordering of two numbers (bools count as ints): exact between
/// ints, and between an int and a float. `None` for non-numbers.
pub fn python_cmp(left: &Value, right: &Value) -> Option<Ordering> {
    match (real(left)?, real(right)?) {
        (Real::Int(a), Real::Int(b)) => Some(a.cmp(&b)),
        (Real::Float(a), Real::Float(b)) => a.partial_cmp(&b),
        (Real::Int(a), Real::Float(b)) => compare_int_float(a, b),
        (Real::Float(a), Real::Int(b)) => compare_int_float(b, a).map(Ordering::reverse),
    }
}

/// Two objects with the same keys whose values agree under `equal`.
pub fn objects_eq(
    left: &Map<String, Value>,
    right: &Map<String, Value>,
    equal: fn(&Value, &Value) -> bool,
) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .all(|(key, value)| right.get(key).is_some_and(|other| equal(value, other)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn floats_round_as_python_rounds_them() {
        // Python 3.12's round(value, digits). 0.125, 0.375, 0.03125, 0.09375
        // and 0.15625 are exact ties, rounded to even; 0.00125 and 2.675 are
        // not ties in binary.
        for (value, digits, rounded) in [
            (0.125f64, 2, 0.12f64),
            (0.375, 2, 0.38),
            (-0.125, 2, -0.12),
            (0.5, 0, 0.0),
            (2.5, 0, 2.0),
            (0.03125, 4, 0.0312),
            (0.09375, 4, 0.0938),
            (0.15625, 4, 0.1562),
            (0.00125, 4, 0.0013),
            (0.0006103515625, 4, 0.0006),
            (0.99993896484375, 4, 0.9999),
            (2.675, 2, 2.67),
            (1.00005, 4, 1.0001),
            (0.99995, 4, 1.0),
            (2.0 / 3.0, 4, 0.6667),
            (-0.00001, 2, -0.0),
            (1e300, 2, 1e300),
        ] {
            let actual = python_round(value, digits);
            assert_eq!(
                actual.to_bits(),
                rounded.to_bits(),
                "round({value}, {digits})"
            );
        }
        assert!(python_round(f64::INFINITY, 2).is_infinite());
    }

    #[test]
    fn floats_read_as_python_writes_them() {
        for (value, text) in [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (123.456, "123.456"),
            (1e15, "1000000000000000.0"),
            (1234567890123456.0, "1234567890123456.0"),
            (1e16, "1e+16"),
            (1.5e300, "1.5e+300"),
            (-2.5e-300, "-2.5e-300"),
            (0.15000000000000002, "0.15000000000000002"),
            (0.17345046997070312, "0.17345046997070312"),
            (9007199254740993.0, "9007199254740992.0"),
            (1e23, "1e+23"),
            (f64::MIN_POSITIVE, "2.2250738585072014e-308"),
            (5e-324, "5e-324"),
        ] {
            assert_eq!(python_float_repr(value), text, "{value:e}");
        }
    }

    #[test]
    fn the_canonical_encoding_is_python_compatible() {
        let value: Value = serde_json::from_str(
            r#"{"b": [1, 2.0, -0.5, true, null], "a": "café 😀\n\u0001\"\\", "A": {"z": 1e-7, "y": 10}}"#,
        )
        .unwrap();
        assert_eq!(
            String::from_utf8(canonical(&value)).unwrap(),
            r#"{"A":{"y":10,"z":1e-07},"a":"caf\u00e9 \ud83d\ude00\n\u0001\"\\","b":[1,2.0,-0.5,true,null]}"#
        );
        assert_eq!(
            canonical_sha256(&json!([])),
            "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
        );
    }

    #[test]
    fn parsing_is_exact_and_strict() {
        let value =
            parse(b" {\"a\": [0.049999999999999996, 1, -2, 1e-7, \"\\u00e9\\ud83d\\ude00\\n\"]} ")
                .unwrap();
        assert_eq!(value["a"][0].as_f64(), Some(0.049999999999999996));
        assert_ne!(0.049999999999999996, 0.05);
        assert_eq!(value["a"][1], json!(1));
        assert_eq!(value["a"][2], json!(-2));
        assert_eq!(value["a"][3].as_f64(), Some(1e-7));
        assert_eq!(value["a"][4], json!("\u{e9}\u{1f600}\n"));
        for bad in [
            &b"{\"a\": 1, \"a\": 2}"[..],
            b"[1e999]",
            b"[NaN]",
            b"[01]",
            b"[1.]",
            b"[18446744073709551616]",
            b"\"\\ud800\"",
            b"[1] x",
            b"\"\t\"",
        ] {
            assert!(parse(bad).is_err(), "{}", String::from_utf8_lossy(bad));
        }
        let keys: Vec<_> = parse(b"{\"b\": 1, \"a\": 2}")
            .unwrap()
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(keys, ["b", "a"]);
    }

    #[test]
    fn every_ascii_code_point_is_written_as_python_writes_it() {
        // json.dumps over code points 0-127, then U+00E9, U+1F600 and U+2028.
        let text: String = (0..128u8)
            .map(char::from)
            .chain(['\u{e9}', '\u{1f600}', '\u{2028}'])
            .collect();
        let expected = r##""\u0000\u0001\u0002\u0003\u0004\u0005\u0006\u0007\b\t\n\u000b\f\r\u000e\u000f\u0010\u0011\u0012\u0013\u0014\u0015\u0016\u0017\u0018\u0019\u001a\u001b\u001c\u001d\u001e\u001f !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\u007f\u00e9\ud83d\ude00\u2028""##;
        assert_eq!(
            String::from_utf8(canonical(&Value::String(text.clone()))).unwrap(),
            expected
        );
        assert_eq!(parse(expected.as_bytes()).unwrap(), Value::String(text));
    }

    #[test]
    fn unicode_escapes_take_exactly_four_hex_digits() {
        for bad in [
            &b"\"\\u+041\""[..],
            b"\"\\u-041\"",
            b"\"\\u041\"",
            b"\"\\u 041\"",
            b"\"\\u04g1\"",
            b"{\"\\u+041\": 1}",
            b"\"\\ud83d\\u+e00\"",
        ] {
            assert!(parse(bad).is_err(), "{}", String::from_utf8_lossy(bad));
        }
        assert_eq!(parse(b"{\"\\u0041\": 1}").unwrap(), json!({"A": 1}));
        assert_eq!(parse(b"\"\\u00E9\"").unwrap(), json!("\u{e9}"));
    }

    #[test]
    fn numbers_compare_exactly_as_python_compares_them() {
        use std::cmp::Ordering::{Equal, Greater, Less};
        let big = json!(9007199254740993u64);
        let rounded = json!(9007199254740992.0);
        assert_eq!(python_cmp(&big, &rounded), Some(Greater));
        assert_eq!(python_cmp(&rounded, &big), Some(Less));
        assert!(!python_eq(&big, &rounded));
        assert_eq!(
            python_cmp(&json!(9007199254740992u64), &rounded),
            Some(Equal)
        );
        assert_eq!(
            python_cmp(&json!(u64::MAX), &json!(18446744073709551616.0)),
            Some(Less)
        );
        assert_eq!(python_cmp(&json!(-3), &json!(-2.5)), Some(Less));
        assert_eq!(python_cmp(&json!(-2), &json!(-2.5)), Some(Greater));
        assert_eq!(python_cmp(&json!(0), &json!(-0.0)), Some(Equal));
        assert_eq!(python_cmp(&json!(true), &json!(1.0)), Some(Equal));
        assert_eq!(python_cmp(&json!(i64::MIN), &json!(-1e300)), Some(Greater));
        assert_eq!(python_cmp(&json!(1), &json!(1e300)), Some(Less));
        assert_eq!(python_cmp(&json!("1"), &json!(1)), None);
    }

    #[test]
    fn python_equality_and_truthiness() {
        assert!(python_eq(&json!(1), &json!(1.0)));
        assert!(python_eq(&json!(true), &json!(1)));
        assert!(!python_eq(&json!(true), &json!("true")));
        assert!(python_eq(&json!({"a": [0, 1]}), &json!({"a": [0.0, true]})));
        assert!(!truthy(&json!(0.0)) && !truthy(&json!("")) && truthy(&json!([0])));
    }
}
