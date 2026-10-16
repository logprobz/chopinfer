use std::collections::BTreeMap;

use crate::{ArtifactError, Result};

// The 120,994,711-byte production representative has 6,366,005 nodes, 875,835 arrays,
// 1,867,832 array items, 4,498,172 object fields, and 88,422,623 string bytes.
// These limits leave at least 20 percent aggregate growth margin. The individual
// string limit also accommodates bounded metadata parsed by this shared parser.
const MAX_JSON_NODES: usize = 8_000_000;
const MAX_JSON_ARRAYS: usize = 1_100_000;
const MAX_JSON_ARRAY_ITEMS: usize = 2_250_000;
const MAX_JSON_OBJECT_FIELDS: usize = 5_750_000;
const MAX_JSON_STRING_BYTES: usize = 1024 * 1024;
const MAX_JSON_TOTAL_STRING_BYTES: usize = 112 * 1024 * 1024;
const MAX_JSON_NESTING: usize = 64;

#[derive(Clone, Copy)]
struct Limits {
    nodes: usize,
    arrays: usize,
    array_items: usize,
    object_fields: usize,
    string_bytes: usize,
    total_string_bytes: usize,
    nesting: usize,
}

const LIMITS: Limits = Limits {
    nodes: MAX_JSON_NODES,
    arrays: MAX_JSON_ARRAYS,
    array_items: MAX_JSON_ARRAY_ITEMS,
    object_fields: MAX_JSON_OBJECT_FIELDS,
    string_bytes: MAX_JSON_STRING_BYTES,
    total_string_bytes: MAX_JSON_TOTAL_STRING_BYTES,
    nesting: MAX_JSON_NESTING,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Value {
    Null,
    Bool,
    Number(u64),
    String(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
}

impl Value {
    pub fn object(&self, context: &str) -> Result<&BTreeMap<String, Value>> {
        match self {
            Self::Object(value) => Ok(value),
            _ => Err(ArtifactError::new(format!(
                "{context} must be a JSON object"
            ))),
        }
    }

    pub fn array(&self, context: &str) -> Result<&[Value]> {
        match self {
            Self::Array(value) => Ok(value),
            _ => Err(ArtifactError::new(format!(
                "{context} must be a JSON array"
            ))),
        }
    }

    pub fn string(&self, context: &str) -> Result<&str> {
        match self {
            Self::String(value) => Ok(value),
            _ => Err(ArtifactError::new(format!(
                "{context} must be a JSON string"
            ))),
        }
    }

    pub fn number(&self, context: &str) -> Result<u64> {
        match self {
            Self::Number(value) => Ok(*value),
            _ => Err(ArtifactError::new(format!(
                "{context} must be an unsigned JSON integer"
            ))),
        }
    }
}

pub fn parse(input: &[u8]) -> Result<Value> {
    parse_with_limits(input, LIMITS)
}

fn parse_with_limits(input: &[u8], limits: Limits) -> Result<Value> {
    let mut parser = Parser {
        input,
        offset: 0,
        limits,
        usage: Usage::default(),
    };
    let value = parser.value()?;
    parser.whitespace();
    if parser.offset != input.len() {
        return Err(parser.error("unexpected trailing JSON content"));
    }
    Ok(value)
}

#[derive(Default)]
struct Usage {
    nodes: usize,
    arrays: usize,
    array_items: usize,
    object_fields: usize,
    string_bytes: usize,
    nesting: usize,
}

struct Parser<'a> {
    input: &'a [u8],
    offset: usize,
    limits: Limits,
    usage: Usage,
}

impl Parser<'_> {
    fn value(&mut self) -> Result<Value> {
        self.whitespace();
        self.charge_node()?;
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => self.string().map(Value::String),
            Some(b't') => self.literal(b"true", Value::Bool),
            Some(b'f') => self.literal(b"false", Value::Bool),
            Some(b'n') => self.literal(b"null", Value::Null),
            Some(b'0'..=b'9') => self.number().map(Value::Number),
            Some(_) => Err(self.error("unexpected JSON token")),
            None => Err(self.error("unexpected end of JSON")),
        }
    }

    fn object(&mut self) -> Result<Value> {
        self.enter_container()?;
        self.expect(b'{')?;
        let result = self.object_contents();
        self.usage.nesting -= 1;
        result
    }

    fn object_contents(&mut self) -> Result<Value> {
        let mut values = BTreeMap::new();
        self.whitespace();
        if self.take(b'}') {
            return Ok(Value::Object(values));
        }
        loop {
            self.whitespace();
            self.charge_object_field()?;
            let key = self.string()?;
            self.whitespace();
            self.expect(b':')?;
            let value = self.value()?;
            match values.entry(key) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(value);
                }
                std::collections::btree_map::Entry::Occupied(entry) => {
                    return Err(self.error(&format!("duplicate JSON key '{}'", entry.key())));
                }
            }
            self.whitespace();
            if self.take(b'}') {
                return Ok(Value::Object(values));
            }
            self.expect(b',')?;
        }
    }

    fn array(&mut self) -> Result<Value> {
        self.usage.arrays = self.charged(self.usage.arrays, self.limits.arrays, "arrays")?;
        self.enter_container()?;
        self.expect(b'[')?;
        let result = self.array_contents();
        self.usage.nesting -= 1;
        result
    }

    fn array_contents(&mut self) -> Result<Value> {
        let mut values = Vec::new();
        self.whitespace();
        if self.take(b']') {
            return Ok(Value::Array(values));
        }
        loop {
            self.charge_array_item()?;
            values.push(self.value()?);
            self.whitespace();
            if self.take(b']') {
                return Ok(Value::Array(values));
            }
            self.expect(b',')?;
        }
    }

    fn string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let start = self.offset;
        let mut escaped = false;
        loop {
            let byte = self
                .next()
                .ok_or_else(|| self.error("unterminated JSON string"))?;
            match byte {
                b'"' => break,
                b'\\' => {
                    escaped = true;
                    self.scan_escape()?;
                }
                0x00..=0x1f => return Err(self.error("control byte in JSON string")),
                _ => {}
            }
            let encoded_bytes = self.offset - start;
            if encoded_bytes > self.limits.string_bytes {
                return Err(self.error(&format!(
                    "JSON string exceeds {} encoded bytes",
                    self.limits.string_bytes
                )));
            }
        }
        let end = self.offset - 1;
        let encoded_bytes = end - start;
        self.usage.string_bytes = self
            .usage
            .string_bytes
            .checked_add(encoded_bytes)
            .ok_or_else(|| self.error("aggregate JSON string byte count overflow"))?;
        if self.usage.string_bytes > self.limits.total_string_bytes {
            return Err(self.error(&format!(
                "aggregate JSON strings exceed {} encoded bytes",
                self.limits.total_string_bytes
            )));
        }
        let raw = &self.input[start..end];
        if escaped {
            self.decode_escaped_string(raw, encoded_bytes)
        } else {
            std::str::from_utf8(raw)
                .map(str::to_owned)
                .map_err(|error| self.error(&format!("invalid JSON UTF-8: {error}")))
        }
    }

    fn scan_escape(&mut self) -> Result<()> {
        let escaped = self
            .next()
            .ok_or_else(|| self.error("unterminated JSON escape"))?;
        match escaped {
            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => Ok(()),
            b'u' => {
                let first = self.scan_hex_quad()?;
                if (0xd800..=0xdbff).contains(&first) {
                    self.expect(b'\\')?;
                    self.expect(b'u')?;
                    let second = self.scan_hex_quad()?;
                    if !(0xdc00..=0xdfff).contains(&second) {
                        return Err(self.error("invalid low surrogate in JSON escape"));
                    }
                } else if (0xdc00..=0xdfff).contains(&first) {
                    return Err(self.error("unpaired low surrogate in JSON escape"));
                }
                Ok(())
            }
            _ => Err(self.error("invalid JSON escape")),
        }
    }

    fn scan_hex_quad(&mut self) -> Result<u16> {
        let end = self
            .offset
            .checked_add(4)
            .ok_or_else(|| self.error("JSON offset overflow"))?;
        let bytes = self
            .input
            .get(self.offset..end)
            .ok_or_else(|| self.error("short JSON Unicode escape"))?;
        let text = std::str::from_utf8(bytes)
            .map_err(|error| self.error(&format!("invalid JSON Unicode escape: {error}")))?;
        let value =
            u16::from_str_radix(text, 16).map_err(|_| self.error("invalid JSON Unicode escape"))?;
        self.offset = end;
        Ok(value)
    }

    fn decode_escaped_string(&self, raw: &[u8], capacity: usize) -> Result<String> {
        let mut output = String::with_capacity(capacity);
        let mut cursor = 0;
        while cursor < raw.len() {
            let next_escape = raw[cursor..]
                .iter()
                .position(|byte| *byte == b'\\')
                .map_or(raw.len(), |relative| cursor + relative);
            let text = std::str::from_utf8(&raw[cursor..next_escape])
                .map_err(|error| self.error(&format!("invalid JSON UTF-8: {error}")))?;
            output.push_str(text);
            if next_escape == raw.len() {
                break;
            }
            cursor = next_escape + 1;
            let escaped = raw[cursor];
            cursor += 1;
            match escaped {
                b'"' | b'\\' | b'/' => output.push(char::from(escaped)),
                b'b' => output.push('\u{0008}'),
                b'f' => output.push('\u{000c}'),
                b'n' => output.push('\n'),
                b'r' => output.push('\r'),
                b't' => output.push('\t'),
                b'u' => {
                    let first = decode_hex_quad(&raw[cursor..cursor + 4]);
                    cursor += 4;
                    let scalar = if (0xd800..=0xdbff).contains(&first) {
                        cursor += 2;
                        let second = decode_hex_quad(&raw[cursor..cursor + 4]);
                        cursor += 4;
                        0x1_0000
                            + ((u32::from(first) - 0xd800) << 10)
                            + (u32::from(second) - 0xdc00)
                    } else {
                        u32::from(first)
                    };
                    output.push(char::from_u32(scalar).expect("validated JSON Unicode scalar"));
                }
                _ => unreachable!("validated JSON escape"),
            }
        }
        Ok(output)
    }

    fn number(&mut self) -> Result<u64> {
        let start = self.offset;
        if self.take(b'0') {
            if matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("leading zero in JSON number"));
            }
        } else {
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.offset += 1;
            }
        }
        if matches!(self.peek(), Some(b'.' | b'e' | b'E')) {
            return Err(self.error("floating-point JSON number is not allowed"));
        }
        let text = std::str::from_utf8(&self.input[start..self.offset])
            .map_err(|error| self.error(&format!("invalid JSON number: {error}")))?;
        text.parse()
            .map_err(|_| self.error("JSON integer is out of u64 range"))
    }

    fn literal(&mut self, literal: &[u8], value: Value) -> Result<Value> {
        let end = self
            .offset
            .checked_add(literal.len())
            .ok_or_else(|| self.error("JSON offset overflow"))?;
        if self.input.get(self.offset..end) == Some(literal) {
            self.offset = end;
            Ok(value)
        } else {
            Err(self.error("invalid JSON literal"))
        }
    }

    fn enter_container(&mut self) -> Result<()> {
        self.usage.nesting = self
            .usage
            .nesting
            .checked_add(1)
            .ok_or_else(|| self.error("JSON nesting depth overflow"))?;
        if self.usage.nesting > self.limits.nesting {
            return Err(self.error(&format!(
                "JSON nesting exceeds {} levels",
                self.limits.nesting
            )));
        }
        Ok(())
    }

    fn charge_node(&mut self) -> Result<()> {
        self.usage.nodes = self.charged(self.usage.nodes, self.limits.nodes, "nodes")?;
        Ok(())
    }

    fn charge_array_item(&mut self) -> Result<()> {
        self.usage.array_items = self.charged(
            self.usage.array_items,
            self.limits.array_items,
            "array items",
        )?;
        Ok(())
    }

    fn charge_object_field(&mut self) -> Result<()> {
        self.usage.object_fields = self.charged(
            self.usage.object_fields,
            self.limits.object_fields,
            "object fields",
        )?;
        Ok(())
    }

    fn charged(&self, usage: usize, limit: usize, label: &str) -> Result<usize> {
        let usage = usage
            .checked_add(1)
            .ok_or_else(|| self.error(&format!("JSON {label} count overflow")))?;
        if usage > limit {
            return Err(self.error(&format!("JSON {label} exceed {limit}")));
        }
        Ok(usage)
    }

    fn whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.offset += 1;
        }
    }

    fn expect(&mut self, expected: u8) -> Result<()> {
        if self.take(expected) {
            Ok(())
        } else {
            Err(self.error(&format!("expected '{}'", char::from(expected))))
        }
    }

    fn take(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.offset += 1;
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.offset).copied()
    }

    fn next(&mut self) -> Option<u8> {
        let value = self.peek()?;
        self.offset += 1;
        Some(value)
    }

    fn error(&self, message: &str) -> ArtifactError {
        ArtifactError::new(format!("JSON error at byte {}: {message}", self.offset))
    }
}

fn decode_hex_quad(bytes: &[u8]) -> u16 {
    bytes.iter().fold(0_u16, |value, byte| {
        value * 16
            + u16::from(match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => unreachable!("validated JSON Unicode escape"),
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> Limits {
        Limits {
            nodes: 32,
            arrays: 4,
            array_items: 4,
            object_fields: 4,
            string_bytes: 16,
            total_string_bytes: 24,
            nesting: 4,
        }
    }

    #[test]
    fn parser_budgets_null_arrays_before_building_the_tail() {
        let input = format!(
            "[{}]",
            std::iter::repeat_n("null", 100)
                .collect::<Vec<_>>()
                .join(",")
        );
        let error = parse_with_limits(input.as_bytes(), limits()).unwrap_err();
        assert!(error.to_string().contains("array items exceed 4"));
        assert!(error.to_string().contains("byte 21"));
    }

    #[test]
    fn parser_enforces_total_node_and_array_budgets() {
        let mut node_limits = limits();
        node_limits.nodes = 4;
        node_limits.array_items = 8;
        let error = parse_with_limits(b"[null,null,null,null]", node_limits).unwrap_err();
        assert!(error.to_string().contains("nodes exceed 4"));

        let mut array_limits = limits();
        array_limits.arrays = 2;
        let error = parse_with_limits(b"[[],[],[]]", array_limits).unwrap_err();
        assert!(error.to_string().contains("arrays exceed 2"));
    }

    #[test]
    fn parser_budgets_object_fields_before_allocating_the_next_key() {
        let input = r#"{"a":0,"b":0,"c":0,"d":0,"this-key-is-never-allocated":0}"#;
        let error = parse_with_limits(input.as_bytes(), limits()).unwrap_err();
        assert!(error.to_string().contains("object fields exceed 4"));
        assert!(error.to_string().contains("byte 25"));
    }

    #[test]
    fn parser_budgets_individual_and_aggregate_strings_before_allocation() {
        let individual = format!("\"{}\"", "x".repeat(100));
        let error = parse_with_limits(individual.as_bytes(), limits()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("string exceeds 16 encoded bytes")
        );
        assert!(error.to_string().contains("byte 18"));

        let error =
            parse_with_limits(br#"["12345678","12345678","12345678","x"]"#, limits()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("aggregate JSON strings exceed 24")
        );
    }

    #[test]
    fn parser_bounds_non_ascii_work_at_the_string_budget() {
        let input = format!("\"{}\"", "\u{00e9}".repeat(100_000));
        let error = parse_with_limits(input.as_bytes(), limits()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("string exceeds 16 encoded bytes")
        );
        assert!(error.to_string().contains("byte 18"));
    }

    #[test]
    fn parser_decodes_valid_non_ascii_and_escaped_strings() {
        let parsed = parse(br#"["mod\u00e8le","\ud83d\ude80","raw \ud7ff \ue000"]"#).unwrap();
        assert_eq!(
            parsed,
            Value::Array(vec![
                Value::String("mod\u{00e8}le".to_owned()),
                Value::String("\u{1f680}".to_owned()),
                Value::String("raw \u{d7ff} \u{e000}".to_owned()),
            ])
        );
    }
}
