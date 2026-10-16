use std::collections::HashMap;

use crate::{Result, TokenizerError};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(u64),
    String(String),
    Array(Vec<Value>),
    Object(HashMap<String, Value>),
}

impl Value {
    pub fn object(&self, context: &str) -> Result<&HashMap<String, Value>> {
        match self {
            Self::Object(value) => Ok(value),
            _ => Err(TokenizerError::new(format!(
                "{context} must be a JSON object"
            ))),
        }
    }

    pub fn array(&self, context: &str) -> Result<&[Value]> {
        match self {
            Self::Array(value) => Ok(value),
            _ => Err(TokenizerError::new(format!(
                "{context} must be a JSON array"
            ))),
        }
    }

    pub fn string(&self, context: &str) -> Result<&str> {
        match self {
            Self::String(value) => Ok(value),
            _ => Err(TokenizerError::new(format!(
                "{context} must be a JSON string"
            ))),
        }
    }

    pub fn number(&self, context: &str) -> Result<u64> {
        match self {
            Self::Number(value) => Ok(*value),
            _ => Err(TokenizerError::new(format!(
                "{context} must be an unsigned JSON integer"
            ))),
        }
    }

    pub fn boolean(&self, context: &str) -> Result<bool> {
        match self {
            Self::Bool(value) => Ok(*value),
            _ => Err(TokenizerError::new(format!(
                "{context} must be a JSON boolean"
            ))),
        }
    }
}

pub fn parse(input: &[u8]) -> Result<Value> {
    let mut parser = Parser { input, offset: 0 };
    let value = parser.value()?;
    parser.whitespace();
    if parser.offset != input.len() {
        return Err(parser.error("unexpected trailing JSON content"));
    }
    Ok(value)
}

struct Parser<'a> {
    input: &'a [u8],
    offset: usize,
}

impl Parser<'_> {
    fn value(&mut self) -> Result<Value> {
        self.whitespace();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => self.string().map(Value::String),
            Some(b't') => self.literal(b"true", Value::Bool(true)),
            Some(b'f') => self.literal(b"false", Value::Bool(false)),
            Some(b'n') => self.literal(b"null", Value::Null),
            Some(b'0'..=b'9') => self.number().map(Value::Number),
            Some(_) => Err(self.error("unexpected JSON token")),
            None => Err(self.error("unexpected end of JSON")),
        }
    }

    fn object(&mut self) -> Result<Value> {
        self.expect(b'{')?;
        let mut values = HashMap::new();
        self.whitespace();
        if self.take(b'}') {
            return Ok(Value::Object(values));
        }
        loop {
            self.whitespace();
            let key = self.string()?;
            self.whitespace();
            self.expect(b':')?;
            let value = self.value()?;
            if values.insert(key.clone(), value).is_some() {
                return Err(self.error(&format!("duplicate JSON key '{key}'")));
            }
            self.whitespace();
            if self.take(b'}') {
                return Ok(Value::Object(values));
            }
            self.expect(b',')?;
        }
    }

    fn array(&mut self) -> Result<Value> {
        self.expect(b'[')?;
        let mut values = Vec::new();
        self.whitespace();
        if self.take(b']') {
            return Ok(Value::Array(values));
        }
        loop {
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
        let mut value = String::new();
        loop {
            let byte = self
                .next()
                .ok_or_else(|| self.error("unterminated JSON string"))?;
            match byte {
                b'"' => return Ok(value),
                b'\\' => self.escape(&mut value)?,
                0x00..=0x1f => return Err(self.error("control byte in JSON string")),
                0x20..=0x7f => value.push(char::from(byte)),
                _ => {
                    let start = self.offset - 1;
                    let width = match byte {
                        0xc2..=0xdf => 2,
                        0xe0..=0xef => 3,
                        0xf0..=0xf4 => 4,
                        _ => return Err(self.error("invalid JSON UTF-8 leading byte")),
                    };
                    let end = start
                        .checked_add(width)
                        .ok_or_else(|| self.error("JSON string offset overflow"))?;
                    let encoded = self
                        .input
                        .get(start..end)
                        .ok_or_else(|| self.error("truncated JSON UTF-8 scalar"))?;
                    let tail = std::str::from_utf8(encoded)
                        .map_err(|error| self.error(&format!("invalid JSON UTF-8: {error}")))?;
                    let character = tail
                        .chars()
                        .next()
                        .ok_or_else(|| self.error("unterminated JSON string"))?;
                    value.push(character);
                    self.offset = end;
                }
            }
        }
    }

    fn escape(&mut self, output: &mut String) -> Result<()> {
        let escaped = self
            .next()
            .ok_or_else(|| self.error("unterminated JSON escape"))?;
        match escaped {
            b'"' | b'\\' | b'/' => output.push(char::from(escaped)),
            b'b' => output.push('\u{0008}'),
            b'f' => output.push('\u{000c}'),
            b'n' => output.push('\n'),
            b'r' => output.push('\r'),
            b't' => output.push('\t'),
            b'u' => output.push(self.unicode_escape()?),
            _ => return Err(self.error("invalid JSON escape")),
        }
        Ok(())
    }

    fn unicode_escape(&mut self) -> Result<char> {
        let first = self.hex_quad()?;
        let scalar = if (0xd800..=0xdbff).contains(&first) {
            self.expect(b'\\')?;
            self.expect(b'u')?;
            let second = self.hex_quad()?;
            if !(0xdc00..=0xdfff).contains(&second) {
                return Err(self.error("invalid low surrogate in JSON escape"));
            }
            0x1_0000 + ((u32::from(first) - 0xd800) << 10) + (u32::from(second) - 0xdc00)
        } else if (0xdc00..=0xdfff).contains(&first) {
            return Err(self.error("unpaired low surrogate in JSON escape"));
        } else {
            u32::from(first)
        };
        char::from_u32(scalar).ok_or_else(|| self.error("invalid JSON Unicode scalar"))
    }

    fn hex_quad(&mut self) -> Result<u16> {
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

    fn error(&self, message: &str) -> TokenizerError {
        TokenizerError::new(format!("JSON error at byte {}: {message}", self.offset))
    }
}
