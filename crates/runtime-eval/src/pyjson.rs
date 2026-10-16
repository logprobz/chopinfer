//! Python 3.12's `json.loads` of a `str`, as the qualification modules
//! decode model output: the objects it builds and the `JSONDecodeError` it
//! raises, with Python's message and position. Unlike [`crate::json::parse`]
//! it takes what Python takes: a repeated key keeps its first position and
//! last value, `NaN`, `Infinity` and `-Infinity`, out-of-range floats as
//! infinities, and ints of any size up to Python's 4300-digit limit.

use std::cmp::Ordering;
use std::collections::HashMap;

use serde_json::Value;

use crate::json::{python_cmp, python_float_repr, python_number};
use crate::repr;

/// A Python object `json.loads` builds.
#[derive(Clone, Debug, PartialEq)]
pub enum Object {
    None,
    Bool(bool),
    Int(i128),
    /// An int outside i128, as its decimal text.
    BigInt(String),
    Float(f64),
    /// A lone surrogate escape is held as U+FFFD (Rust strings cannot hold
    /// one); every other string is exact.
    Str(String),
    List(Vec<Object>),
    Dict(Vec<(String, Object)>),
}

/// Why `json.loads` raised: a `JSONDecodeError`, Python's `ValueError` for
/// an int past the digit limit, or a `RecursionError` past the nesting
/// limit. The text is `<type>: <message>`, as `structured.run_case` writes
/// a caught error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadError {
    Decode(String),
    Value(String),
    Recursion(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decode(message) => write!(formatter, "JSONDecodeError: {message}"),
            Self::Value(message) => write!(formatter, "ValueError: {message}"),
            Self::Recursion(message) => write!(formatter, "RecursionError: {message}"),
        }
    }
}

/// The deepest nesting Python 3.12.3's C decoder reaches before its
/// recursion limit.
const MAX_DEPTH: usize = 9997;
/// `sys.int_info.default_max_str_digits`.
const MAX_DIGITS: usize = 4300;

/// `json.loads(text)`.
pub fn loads(text: &str) -> Result<Object, LoadError> {
    let chars: Vec<char> = text.chars().collect();
    let decoder = Decoder { chars: &chars };
    if chars.first() == Some(&'\u{feff}') {
        return Err(decoder.error("Unexpected UTF-8 BOM (decode using utf-8-sig)", 0));
    }
    let (value, end) = decoder.scan(decoder.space(0))?;
    let end = decoder.space(end);
    if end != chars.len() {
        return Err(decoder.error("Extra data", end));
    }
    Ok(value)
}

/// A container being decoded: a list's items, or a dict's entries (with
/// each key's index) and the key whose value comes next.
enum Frame {
    List(Vec<Object>),
    Dict(Vec<(String, Object)>, HashMap<String, usize>, String),
}

struct Decoder<'a> {
    chars: &'a [char],
}

impl Decoder<'_> {
    /// `JSONDecodeError(message, doc, position)`.
    fn error(&self, message: &str, position: usize) -> LoadError {
        let before = &self.chars[..position.min(self.chars.len())];
        let line = before
            .iter()
            .filter(|character| **character == '\n')
            .count()
            + 1;
        let column = match before.iter().rposition(|character| *character == '\n') {
            Some(newline) => position - newline,
            None => position + 1,
        };
        LoadError::Decode(format!(
            "{message}: line {line} column {column} (char {position})"
        ))
    }

    fn at(&self, index: usize) -> Option<char> {
        self.chars.get(index).copied()
    }

    fn space(&self, mut index: usize) -> usize {
        while matches!(self.at(index), Some(' ' | '\t' | '\n' | '\r')) {
            index += 1;
        }
        index
    }

    fn literal(&self, index: usize, text: &str) -> bool {
        let length = text.chars().count();
        index + length <= self.chars.len()
            && self.chars[index..index + length]
                .iter()
                .copied()
                .eq(text.chars())
    }

    /// A dict key and the start of its value: `"key"`, then `:`.
    fn key(&self, index: usize) -> Result<(String, usize), LoadError> {
        if self.at(index) != Some('"') {
            return Err(self.error("Expecting property name enclosed in double quotes", index));
        }
        let (key, end) = self.string(index + 1)?;
        let end = self.space(end);
        if self.at(end) != Some(':') {
            return Err(self.error("Expecting ':' delimiter", end));
        }
        Ok((key, self.space(end + 1)))
    }

    /// `scan_once` at `index` (`_json.c` `scan_once_unicode`), iterating
    /// where the C decoder recurses: the value and the index after it.
    fn scan(&self, mut index: usize) -> Result<(Object, usize), LoadError> {
        let mut stack: Vec<Frame> = Vec::new();
        loop {
            let mut value = match self.at(index) {
                None => return Err(self.error("Expecting value", index)),
                Some('"') => {
                    let (text, end) = self.string(index + 1)?;
                    index = end;
                    Object::Str(text)
                }
                Some(open @ ('{' | '[')) => {
                    if stack.len() >= MAX_DEPTH {
                        let what = if open == '{' { "object" } else { "array" };
                        return Err(LoadError::Recursion(format!(
                            "maximum recursion depth exceeded while decoding a JSON {what} from a unicode string"
                        )));
                    }
                    let close = if open == '{' { '}' } else { ']' };
                    let inner = self.space(index + 1);
                    if self.at(inner) == Some(close) {
                        index = inner + 1;
                        if open == '{' {
                            Object::Dict(Vec::new())
                        } else {
                            Object::List(Vec::new())
                        }
                    } else if open == '{' {
                        let (key, start) = self.key(inner)?;
                        stack.push(Frame::Dict(Vec::new(), HashMap::new(), key));
                        index = start;
                        continue;
                    } else {
                        stack.push(Frame::List(Vec::new()));
                        index = inner;
                        continue;
                    }
                }
                Some(_) => {
                    let named = [
                        ("null", Object::None),
                        ("true", Object::Bool(true)),
                        ("false", Object::Bool(false)),
                        ("NaN", Object::Float(f64::NAN)),
                        ("Infinity", Object::Float(f64::INFINITY)),
                        ("-Infinity", Object::Float(f64::NEG_INFINITY)),
                    ];
                    // A name is taken wherever it starts, whatever follows.
                    match named
                        .into_iter()
                        .find(|(text, _)| self.literal(index, text))
                    {
                        Some((text, value)) => {
                            index += text.chars().count();
                            value
                        }
                        None => {
                            let (value, end) = self.number(index)?;
                            index = end;
                            value
                        }
                    }
                }
            };
            // Attach the value to the containers it closes.
            loop {
                match stack.last_mut() {
                    None => return Ok((value, index)),
                    Some(Frame::List(items)) => {
                        items.push(value);
                        index = self.space(index);
                        match self.at(index) {
                            Some(']') => {
                                index += 1;
                                let Some(Frame::List(items)) = stack.pop() else {
                                    unreachable!("the last frame is a list")
                                };
                                value = Object::List(items);
                            }
                            Some(',') => {
                                index = self.space(index + 1);
                                break;
                            }
                            _ => return Err(self.error("Expecting ',' delimiter", index)),
                        }
                    }
                    Some(Frame::Dict(entries, positions, key)) => {
                        let key = std::mem::take(key);
                        match positions.get(&key) {
                            Some(position) => entries[*position].1 = value,
                            None => {
                                positions.insert(key.clone(), entries.len());
                                entries.push((key, value));
                            }
                        }
                        index = self.space(index);
                        match self.at(index) {
                            Some('}') => {
                                index += 1;
                                let Some(Frame::Dict(entries, _, _)) = stack.pop() else {
                                    unreachable!("the last frame is a dict")
                                };
                                value = Object::Dict(entries);
                            }
                            Some(',') => {
                                let (next, start) = self.key(self.space(index + 1))?;
                                if let Some(Frame::Dict(_, _, pending)) = stack.last_mut() {
                                    *pending = next;
                                }
                                index = start;
                                break;
                            }
                            _ => return Err(self.error("Expecting ',' delimiter", index)),
                        }
                    }
                }
            }
        }
    }

    /// A string whose first character after the quote is at `end`
    /// (`scanstring_unicode`, strict): its text and the index after it.
    fn string(&self, mut end: usize) -> Result<(String, usize), LoadError> {
        let begin = end - 1;
        let length = self.chars.len();
        let mut text = String::new();
        loop {
            let mut next = end;
            while next < length && !matches!(self.chars[next], '"' | '\\') {
                if u32::from(self.chars[next]) <= 0x1f {
                    return Err(self.error("Invalid control character at", next));
                }
                next += 1;
            }
            if next == length {
                return Err(self.error("Unterminated string starting at", begin));
            }
            text.extend(&self.chars[end..next]);
            if self.chars[next] == '"' {
                return Ok((text, next + 1));
            }
            next += 1;
            if next == length {
                return Err(self.error("Unterminated string starting at", begin));
            }
            if self.chars[next] != 'u' {
                end = next + 1;
                text.push(match self.chars[next] {
                    '"' => '"',
                    '\\' => '\\',
                    '/' => '/',
                    'b' => '\u{8}',
                    'f' => '\u{c}',
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    _ => return Err(self.error("Invalid \\escape", end - 2)),
                });
                continue;
            }
            next += 1;
            end = next + 4;
            if end >= length {
                return Err(self.error("Invalid \\uXXXX escape", next - 1));
            }
            let hex = |from: usize, to: usize| -> Result<u32, LoadError> {
                self.chars[from..to].iter().try_fold(0u32, |code, digit| {
                    digit
                        .to_digit(16)
                        .map(|digit| (code << 4) | digit)
                        .ok_or_else(|| self.error("Invalid \\uXXXX escape", to - 5))
                })
            };
            let mut code = hex(next, end)?;
            if (0xd800..0xdc00).contains(&code)
                && end + 6 < length
                && self.chars[end] == '\\'
                && self.chars[end + 1] == 'u'
            {
                let low = hex(end + 2, end + 6)?;
                if (0xdc00..0xe000).contains(&low) {
                    code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
                    end += 6;
                }
            }
            text.push(char::from_u32(code).unwrap_or('\u{fffd}'));
        }
    }

    /// A number at `start` (`_match_number_unicode`): an int unless a
    /// fraction or exponent follows; an incomplete exponent is left
    /// unread.
    fn number(&self, start: usize) -> Result<(Object, usize), LoadError> {
        let digit = |index: usize| self.at(index).is_some_and(|c| c.is_ascii_digit());
        let mut index = start;
        if self.at(index) == Some('-') {
            index += 1;
        }
        match self.at(index) {
            Some('1'..='9') => {
                while digit(index) {
                    index += 1;
                }
            }
            Some('0') => index += 1,
            _ => return Err(self.error("Expecting value", start)),
        }
        let mut float = false;
        if self.at(index) == Some('.') && digit(index + 1) {
            float = true;
            index += 2;
            while digit(index) {
                index += 1;
            }
        }
        if matches!(self.at(index), Some('e' | 'E')) {
            let mut exponent = index + 1;
            if matches!(self.at(exponent), Some('+' | '-')) {
                exponent += 1;
            }
            if digit(exponent) {
                float = true;
                index = exponent;
                while digit(index) {
                    index += 1;
                }
            }
        }
        let text: String = self.chars[start..index].iter().collect();
        if float {
            let value = text.parse::<f64>().expect("a JSON float reads");
            return Ok((Object::Float(value), index));
        }
        let digits = text.trim_start_matches('-').len();
        if digits > MAX_DIGITS {
            return Err(LoadError::Value(format!(
                "Exceeds the limit ({MAX_DIGITS} digits) for integer string conversion: value has {digits} digits; use sys.set_int_max_str_digits() to increase the limit"
            )));
        }
        let value = match text.parse::<i128>() {
            Ok(value) => Object::Int(value),
            Err(_) => Object::BigInt(text),
        };
        Ok((value, index))
    }
}

impl Object {
    /// `type(value).__name__`.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::None => "NoneType",
            Self::Bool(_) => "bool",
            Self::Int(_) | Self::BigInt(_) => "int",
            Self::Float(_) => "float",
            Self::Str(_) => "str",
            Self::List(_) => "list",
            Self::Dict(_) => "dict",
        }
    }

    /// `repr(value)`.
    pub fn repr(&self) -> String {
        match self {
            Self::None => "None".into(),
            Self::Bool(true) => "True".into(),
            Self::Bool(false) => "False".into(),
            Self::Int(value) => value.to_string(),
            Self::BigInt(text) => text.clone(),
            Self::Float(value) if value.is_nan() => "nan".into(),
            Self::Float(value) if value.is_infinite() => {
                if *value > 0.0 { "inf" } else { "-inf" }.into()
            }
            Self::Float(value) => python_float_repr(*value),
            Self::Str(text) => repr::repr(&Value::String(text.clone())),
            Self::List(items) => {
                let items: Vec<String> = items.iter().map(Self::repr).collect();
                format!("[{}]", items.join(", "))
            }
            Self::Dict(entries) => {
                let entries: Vec<String> = entries
                    .iter()
                    .map(|(key, value)| {
                        format!(
                            "{}: {}",
                            repr::repr(&Value::String(key.clone())),
                            value.repr()
                        )
                    })
                    .collect();
                format!("{{{}}}", entries.join(", "))
            }
        }
    }

    /// The number as a JSON value Python compares equally: exact for
    /// floats and for ints within 64 bits; a wider int as its nearest
    /// float. `None` for a non-number or a non-finite float.
    fn as_json_number(&self) -> Option<Value> {
        match self {
            Self::Bool(flag) => Some(Value::Bool(*flag)),
            Self::Int(value) => i64::try_from(*value)
                .map(Value::from)
                .or_else(|_| u64::try_from(*value).map(Value::from))
                .ok()
                .or_else(|| Some(Value::from(*value as f64))),
            Self::BigInt(text) => Some(Value::from(text.parse::<f64>().ok()?)),
            Self::Float(value) => serde_json::Number::from_f64(*value).map(Value::Number),
            _ => None,
        }
    }

    /// Python's ordering of this number against a JSON number (`None` for
    /// a NaN or a non-number).
    pub fn compare(&self, other: &Value) -> Option<Ordering> {
        match self {
            Self::Float(value) if value.is_infinite() => {
                python_number(other)?;
                Some(if *value > 0.0 {
                    Ordering::Greater
                } else {
                    Ordering::Less
                })
            }
            _ => python_cmp(&self.as_json_number()?, other),
        }
    }

    /// Python's `self == other` against a JSON value: numbers by value
    /// (a bool counts as an int), containers item by item, dicts
    /// regardless of key order.
    pub fn equals(&self, other: &Value) -> bool {
        match (self, other) {
            (Self::None, Value::Null) => true,
            (Self::Str(a), Value::String(b)) => a == b,
            (Self::List(a), Value::Array(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.equals(b))
            }
            (Self::Dict(a), Value::Object(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .all(|(key, value)| b.get(key).is_some_and(|other| value.equals(other)))
            }
            (Self::Bool(_) | Self::Int(_) | Self::BigInt(_) | Self::Float(_), _) => {
                self.compare(other) == Some(Ordering::Equal)
            }
            _ => false,
        }
    }

    /// Python's `type(self) is type(other)` against a JSON value.
    pub fn same_type(&self, other: &Value) -> bool {
        self.type_name() == repr::type_name(other)
    }
}
