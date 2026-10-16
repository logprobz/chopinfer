//! Python's semantics for the values a chat template writes.
//!
//! A template renders under HF's `apply_chat_template`, so `tojson` is
//! `json.dumps` with HF's arguments, `| trim` and `.strip()` are
//! `str.strip`, `.split(sep)` is `str.split`, and `| capitalize` is
//! `str.capitalize`. These functions reproduce them for the JSON values a
//! request carries, and `tests/python_semantics.json` (written by
//! `tools/refrun/python_semantics.py`) checks every one against Python.
//!
//! Every function agrees with Python 3.12 on every Unicode scalar value.
//! The character classes `capitalize` reads are pinned to Unicode 15.0, the
//! version that Python reads, in tables the same script writes
//! (`unicode15.rs`): it gives a code point Unicode 15.0 does not assign no
//! case mapping and reads a final sigma by the 15.0 Cased and
//! Case_Ignorable sets. Rust's standard library, which follows a later
//! Unicode, supplies only the mappings of assigned characters.

use std::cmp::Ordering;
use std::fmt::Write as _;

use crate::request::FrontendJsonV1;
use crate::toolkit::json::python_number_v1;
use crate::toolkit::unicode15::{CASE_IGNORABLE_V1, CASED_V1, UNASSIGNED_V1};

/// The arguments of HF's `tojson` filter, which passes them to `json.dumps`.
/// The default is HF's: `ensure_ascii=False`, no indent, the default
/// separators and keys in their order.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TojsonOptionsV1 {
    /// Writes every character outside printable ASCII as a `\u` escape.
    pub ensure_ascii: bool,
    /// The text one nesting level indents by: `indent=2` is two spaces and
    /// `indent=0` is empty, which still breaks lines. `None` writes one line.
    pub indent: Option<String>,
    /// The item and key separators; `None` is `(", ", ": ")` without an
    /// indent and `(",", ": ")` with one.
    pub separators: Option<(String, String)>,
    /// Writes object keys in code point order.
    pub sort_keys: bool,
}

/// `json.dumps(value, ...)` with `options`, as HF's `tojson` filter writes
/// it. With the default options this is [`crate::toolkit::write_json_v1`] in
/// the Python JSON encoding.
#[must_use]
pub fn tojson_v1(value: &FrontendJsonV1, options: &TojsonOptionsV1) -> String {
    let (item, key) = match (&options.separators, &options.indent) {
        (Some((item, key)), _) => (item.as_str(), key.as_str()),
        (None, None) => (", ", ": "),
        (None, Some(_)) => (",", ": "),
    };
    let mut output = String::new();
    write_tojson_v1(value, options, item, key, 0, &mut output);
    output
}

fn write_tojson_v1(
    value: &FrontendJsonV1,
    options: &TojsonOptionsV1,
    item: &str,
    key: &str,
    level: usize,
    output: &mut String,
) {
    let newline = |output: &mut String, level: usize| {
        if let Some(indent) = &options.indent {
            output.push('\n');
            for _ in 0..level {
                output.push_str(indent);
            }
        }
    };
    match value {
        FrontendJsonV1::Null => output.push_str("null"),
        FrontendJsonV1::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        FrontendJsonV1::Number(number) => output.push_str(&python_number_v1(number)),
        FrontendJsonV1::String(text) => write_json_string_v1(text, options.ensure_ascii, output),
        FrontendJsonV1::Array(values) if values.is_empty() => output.push_str("[]"),
        FrontendJsonV1::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push_str(item);
                }
                newline(output, level + 1);
                write_tojson_v1(value, options, item, key, level + 1, output);
            }
            newline(output, level);
            output.push(']');
        }
        FrontendJsonV1::Object(entries) if entries.is_empty() => output.push_str("{}"),
        FrontendJsonV1::Object(entries) => {
            let mut entries: Vec<&(String, FrontendJsonV1)> = entries.iter().collect();
            if options.sort_keys {
                entries.sort_by(|left, right| left.0.cmp(&right.0));
            }
            output.push('{');
            for (index, (name, value)) in entries.into_iter().enumerate() {
                if index != 0 {
                    output.push_str(item);
                }
                newline(output, level + 1);
                write_json_string_v1(name, options.ensure_ascii, output);
                output.push_str(key);
                write_tojson_v1(value, options, item, key, level + 1, output);
            }
            newline(output, level);
            output.push('}');
        }
    }
}

/// A JSON string literal as `json.dumps` writes it.
fn write_json_string_v1(text: &str, ensure_ascii: bool, output: &mut String) {
    output.push('"');
    for character in text.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{0008}' => output.push_str("\\b"),
            '\u{000c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            ' '..='~' => output.push(character),
            _ if !ensure_ascii && character > '\u{001f}' => output.push(character),
            _ => {
                let mut units = [0_u16; 2];
                for unit in character.encode_utf16(&mut units) {
                    write!(output, "\\u{unit:04x}").expect("writing to a String cannot fail");
                }
            }
        }
    }
    output.push('"');
}

/// Whether `character` lies in one of `table`'s ascending ranges.
fn in_table_v1(table: &[(u32, u32)], character: char) -> bool {
    let code = u32::from(character);
    table
        .binary_search_by(|(start, end)| {
            if *end < code {
                Ordering::Less
            } else if *start > code {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        })
        .is_ok()
}

/// Whether Unicode 15.0 leaves `character` unassigned.
fn unassigned_v1(character: char) -> bool {
    in_table_v1(&UNASSIGNED_V1, character)
}

/// Whether `str.isspace` holds for this character: the ASCII whitespace,
/// the four information separators `\x1c` to `\x1f`, and the Unicode space
/// and line separators.
const fn python_isspace_v1(character: char) -> bool {
    matches!(
        character,
        '\t'..='\r'
            | '\u{1c}'..='\u{20}'
            | '\u{85}'
            | '\u{a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
    )
}

/// `text.strip()`, which `| trim` also calls. Rust's `trim` keeps the
/// separators `\x1c` to `\x1f` that Python strips.
#[must_use]
pub fn python_strip_v1(text: &str) -> &str {
    text.trim_matches(python_isspace_v1)
}

/// `text.rstrip()`.
#[must_use]
pub fn python_rstrip_v1(text: &str) -> &str {
    text.trim_end_matches(python_isspace_v1)
}

/// `text.split(separator)`: every piece between non-overlapping matches from
/// the left, empty pieces included.
///
/// # Panics
///
/// When `separator` is empty, which Python refuses with a `ValueError`; a
/// template only splits on literals.
#[must_use]
pub fn python_split_v1<'a>(text: &'a str, separator: &str) -> Vec<&'a str> {
    assert!(!separator.is_empty(), "empty separator");
    text.split(separator).collect()
}

/// `text.capitalize()`: the first character in titlecase and the rest in
/// lowercase, a final sigma included.
#[must_use]
pub fn python_capitalize_v1(text: &str) -> String {
    let characters: Vec<char> = text.chars().collect();
    let Some(&first) = characters.first() else {
        return String::new();
    };
    let mut output = String::with_capacity(text.len());
    push_titlecase_v1(first, &mut output);
    for (index, &character) in characters.iter().enumerate().skip(1) {
        if character == '\u{3a3}' {
            output.push(if final_sigma_v1(&characters, index) {
                '\u{3c2}'
            } else {
                '\u{3c3}'
            });
        } else if unassigned_v1(character) {
            output.push(character);
        } else {
            output.extend(character.to_lowercase());
        }
    }
    output
}

/// Whether the capital sigma at `index` lowercases to the final form, as
/// CPython's `handle_capital_sigma` decides: past case-ignorable characters,
/// a cased character precedes it and none follows it.
fn final_sigma_v1(characters: &[char], index: usize) -> bool {
    let cased = |character: char| in_table_v1(&CASED_V1, character);
    let ignorable = |character: char| in_table_v1(&CASE_IGNORABLE_V1, character);
    let before = characters[..index]
        .iter()
        .rev()
        .copied()
        .find(|character| !ignorable(*character));
    let after = characters[index + 1..]
        .iter()
        .copied()
        .find(|character| !ignorable(*character));
    before.is_some_and(cased) && !after.is_some_and(cased)
}

/// Writes the titlecase of one character where it differs from the
/// uppercase in Python 3.12, and the uppercase otherwise.
fn push_titlecase_v1(character: char, output: &mut String) {
    let title = match character {
        // Georgian titlecases to itself. Unicode 16 gave these two Latin
        // letters uppercase forms that Unicode 15 does not have.
        '\u{10d0}'..='\u{10fa}' | '\u{10fd}'..='\u{10ff}' | '\u{19b}' | '\u{264}' => {
            output.push(character);
            return;
        }
        // Unicode 15.0 gives an unassigned code point no case mapping.
        _ if unassigned_v1(character) => {
            output.push(character);
            return;
        }
        // Greek with ypogegrammeni titlecases to the prosgegrammeni form.
        '\u{1f80}'..='\u{1faf}' => {
            output.push(char::from_u32(u32::from(character) | 0x8).expect("a Greek letter"));
            return;
        }
        '\u{df}' => "Ss",
        '\u{1c4}'..='\u{1c6}' => "\u{1c5}",
        '\u{1c7}'..='\u{1c9}' => "\u{1c8}",
        '\u{1ca}'..='\u{1cc}' => "\u{1cb}",
        '\u{1f1}'..='\u{1f3}' => "\u{1f2}",
        '\u{587}' => "\u{535}\u{582}",
        '\u{1fb2}' => "\u{1fba}\u{345}",
        '\u{1fb3}' | '\u{1fbc}' => "\u{1fbc}",
        '\u{1fb4}' => "\u{386}\u{345}",
        '\u{1fb7}' => "\u{391}\u{342}\u{345}",
        '\u{1fc2}' => "\u{1fca}\u{345}",
        '\u{1fc3}' | '\u{1fcc}' => "\u{1fcc}",
        '\u{1fc4}' => "\u{389}\u{345}",
        '\u{1fc7}' => "\u{397}\u{342}\u{345}",
        '\u{1ff2}' => "\u{1ffa}\u{345}",
        '\u{1ff3}' | '\u{1ffc}' => "\u{1ffc}",
        '\u{1ff4}' => "\u{38f}\u{345}",
        '\u{1ff7}' => "\u{3a9}\u{342}\u{345}",
        '\u{fb00}' => "Ff",
        '\u{fb01}' => "Fi",
        '\u{fb02}' => "Fl",
        '\u{fb03}' => "Ffi",
        '\u{fb04}' => "Ffl",
        '\u{fb05}' | '\u{fb06}' => "St",
        '\u{fb13}' => "\u{544}\u{576}",
        '\u{fb14}' => "\u{544}\u{565}",
        '\u{fb15}' => "\u{544}\u{56b}",
        '\u{fb16}' => "\u{54e}\u{576}",
        '\u{fb17}' => "\u{544}\u{56d}",
        _ => {
            output.extend(character.to_uppercase());
            return;
        }
    };
    output.push_str(title);
}
