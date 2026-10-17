//! The declaration as this library reads it.
//!
//! A frontend component's host-read sections come from
//! `runtime-frontend-contract`, which the host checks. Its `format` object is
//! this library's: the parser section a model's own renderer declares
//! ([`FrontendParserDeclarationV1`]), which says how the model's completions
//! read. The host passes it unread; [`FormatDescriptorV1`] reads and checks
//! it, including the rules that tie it to the vocabulary, when the library
//! opens the component.

use std::collections::BTreeSet;
use std::ops::{Deref, DerefMut};

use serde::{Deserialize, Serialize};

pub use runtime_frontend_contract::*;

const MAX_MARKERS: usize = 128;
const MAX_MARKER_BYTES: usize = 8 * 1024;

/// How this model serializes a JSON value it renders into a prompt.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FrontendJsonEncodingV1 {
    /// Compact JSON with `, ` between members and `: ` after a key.
    SpacedJsonV1,
    /// Python-style JSON spacing, including decimal numbers.
    PythonJsonV1,
}

/// A versioned grammar for tool calls inside an assistant completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FrontendToolGrammarV1 {
    /// A function element inside the call: the function open, the name, the
    /// name close, then one parameter element per argument whose name sits in
    /// its open tag, and the function close.
    XmlFunctionParametersV1,
    /// The name follows the call open directly, and each argument is a key
    /// element then a value element: `parameter_open`, the key,
    /// `parameter_name_close`, `parameter_value_open`, the value and
    /// `parameter_close`. There is no function element, so the three function
    /// markers are empty.
    XmlKeyValueArgumentsV1,
    /// The function element of [`Self::XmlFunctionParametersV1`], whose
    /// parameter open tag ends in one of two declared closes that type the
    /// value: `parameter_name_close` for raw text, read as a string, and
    /// `parameter_json_name_close` for JSON, read as the value it spells.
    /// The call open is read with the whitespace the template writes before
    /// it, so text before a call keeps none of it.
    XmlTypedParametersV1,
}

/// What the shared parsers and tool-choice grammars read of a format: the
/// `format.parser` section of a declaration, whose model renders its own
/// prompt ([`crate::ChatFormatV1`]).
///
/// The markers keep exactly the strings and whitespace the template writes;
/// the parsers read most of them without the surrounding whitespace.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendParserDeclarationV1 {
    /// How tool calls are written in a completion.
    pub tool_grammar: FrontendToolGrammarV1,
    /// How parsed tool-call arguments are encoded.
    pub json_encoding: FrontendJsonEncodingV1,
    pub reasoning_open: String,
    pub reasoning_close: String,
    /// What a template may write before the answer, which a forced tool
    /// call may begin with.
    #[serde(default)]
    pub answer_prefix: String,
    pub tool_call_open: String,
    pub tool_call_close: String,
    /// What separates two calls a forced tool choice writes.
    #[serde(default)]
    pub tool_call_separator: String,
    /// The function element's markers; empty for the key-value grammar.
    #[serde(default)]
    pub function_open: String,
    #[serde(default)]
    pub function_name_close: String,
    #[serde(default)]
    pub function_close: String,
    pub parameter_open: String,
    pub parameter_name_close: String,
    /// The parameter name close of a value written as JSON; only the typed
    /// grammar has one, where `parameter_name_close` closes a raw value.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub parameter_json_name_close: String,
    /// What opens an argument's value after its name closes; only the
    /// key-value grammar has one.
    #[serde(default)]
    pub parameter_value_open: String,
    pub parameter_close: String,
    /// Whether answer text that is only whitespace is held until real text
    /// follows, and dropped when a tool call follows it instead.
    #[serde(default)]
    pub drop_blank_tool_call_content: bool,
    /// Every marker a completion's text may not carry outside the structure
    /// that expects it.
    #[serde(default)]
    pub reserved_markers: Vec<String>,
    /// The boundary tokens that open a turn, in code point order.
    pub turn_open: Vec<String>,
    /// The assistant turn's header without its surrounding whitespace; it
    /// begins with one of `turn_open`.
    pub assistant_header: String,
}

/// The `format` object of a frontend component: the parser section, and
/// nothing else. A model writes its own prompt and its own media labels.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendFormatV1 {
    pub parser: FrontendParserDeclarationV1,
}

/// A declaration with its format read: the host-read sections, which it
/// dereferences to, and the parser section only this library reads. The
/// host sections it dereferences to hold `format` as null, so they cannot
/// pass for the declaration; [`Self::into_descriptor`] writes the format
/// back.
#[derive(Clone, Debug, PartialEq)]
pub struct FormatDescriptorV1 {
    host: FrontendDescriptorV1,
    pub parser: FrontendParserDeclarationV1,
}

impl Deref for FormatDescriptorV1 {
    type Target = FrontendDescriptorV1;

    fn deref(&self) -> &FrontendDescriptorV1 {
        &self.host
    }
}

/// The host sections may change in place, as a test's do.
impl DerefMut for FormatDescriptorV1 {
    fn deref_mut(&mut self) -> &mut FrontendDescriptorV1 {
        &mut self.host
    }
}

impl FormatDescriptorV1 {
    /// Reads one frontend component a host already authenticated, under the
    /// component file's size cap.
    pub fn from_component_bytes(bytes: &[u8]) -> Result<Self, FrontendComponentErrorV1> {
        Self::checked(FrontendDescriptorV1::from_component_bytes(bytes)?)
    }

    /// Reads a declaration as [`FrontendDescriptorV1::from_declaration_bytes`]
    /// does, then its format.
    pub fn from_declaration_bytes(bytes: &[u8]) -> Result<Self, FrontendComponentErrorV1> {
        Self::checked(FrontendDescriptorV1::from_declaration_bytes(bytes)?)
    }

    /// Reads a declaration's format and checks neither: for a caller that
    /// drives the format directly over a declaration it checked, or one a
    /// test assembled that does not need to pass. It still refuses a format
    /// that is not exactly a parser section.
    pub fn read(mut host: FrontendDescriptorV1) -> Result<Self, FrontendComponentErrorV1> {
        let format: FrontendFormatV1 = serde_json::from_value(std::mem::take(&mut host.format))
            .map_err(|error| FrontendComponentErrorV1::Syntax(format!("format: {error}")))?;
        Ok(Self {
            host,
            parser: format.parser,
        })
    }

    /// Checks a declaration assembled in code, then reads its format.
    pub fn new(descriptor: FrontendDescriptorV1) -> Result<Self, FrontendComponentErrorV1> {
        descriptor.validate()?;
        Self::checked(descriptor)
    }

    /// Reads the format of a declaration whose host sections were checked.
    fn checked(host: FrontendDescriptorV1) -> Result<Self, FrontendComponentErrorV1> {
        let descriptor = Self::read(host)?;
        descriptor.validate_format()?;
        Ok(descriptor)
    }

    /// Checks the host sections and the format, as reading a component
    /// does: for a descriptor changed in place.
    pub fn validate(&self) -> Result<(), FrontendComponentErrorV1> {
        self.clone().into_descriptor().validate()?;
        self.validate_format()
    }

    /// The host-read sections, with this format written back as `format`.
    #[must_use]
    pub fn into_descriptor(self) -> FrontendDescriptorV1 {
        let mut host = self.host;
        host.format = serde_json::to_value(FrontendFormatV1 {
            parser: self.parser,
        })
        .expect("a format serializes");
        host
    }

    /// Checks every rule the format fixes, against the host sections too.
    fn validate_format(&self) -> Result<(), FrontendComponentErrorV1> {
        let parser = &self.parser;
        validate_reserved_markers(&parser.reserved_markers)?;
        self.validate_parser(parser)?;
        self.validate_name_delimiters(parser)?;
        self.validate_reasoning_controls(parser)
    }

    /// Checks a declared `format.parser`: its markers, and its turn markers
    /// against the vocabulary.
    fn validate_parser(
        &self,
        parser: &FrontendParserDeclarationV1,
    ) -> Result<(), FrontendComponentErrorV1> {
        for (marker, name) in [
            (&parser.reasoning_open, "reasoning_open"),
            (&parser.reasoning_close, "reasoning_close"),
            (&parser.answer_prefix, "answer_prefix"),
            (&parser.tool_call_open, "tool_call_open"),
            (&parser.tool_call_close, "tool_call_close"),
            (&parser.tool_call_separator, "tool_call_separator"),
            (&parser.function_open, "function_open"),
            (&parser.function_name_close, "function_name_close"),
            (&parser.function_close, "function_close"),
            (&parser.parameter_open, "parameter_open"),
            (&parser.parameter_name_close, "parameter_name_close"),
            (
                &parser.parameter_json_name_close,
                "parameter_json_name_close",
            ),
            (&parser.parameter_value_open, "parameter_value_open"),
            (&parser.parameter_close, "parameter_close"),
            (&parser.assistant_header, "assistant_header"),
        ] {
            bounded_marker(marker, name)?;
        }
        for (marker, name) in [
            (&parser.reasoning_open, "reasoning_open"),
            (&parser.reasoning_close, "reasoning_close"),
            (&parser.tool_call_open, "tool_call_open"),
            (&parser.tool_call_close, "tool_call_close"),
            (&parser.parameter_open, "parameter_open"),
            (&parser.parameter_name_close, "parameter_name_close"),
            (&parser.parameter_close, "parameter_close"),
        ] {
            // Completions are read by the marker without its whitespace.
            if marker.trim().is_empty() {
                return Err(invalid(format!("format.parser {name} cannot be blank")));
            }
        }
        validate_tool_grammar_markers(parser)?;
        self.validate_stop_texts()?;
        let vocabulary = &self.vocabulary;
        if parser.turn_open.is_empty() || parser.turn_open.len() > MAX_MARKERS {
            return Err(invalid(format!(
                "format.parser turn_open lists 1 to {MAX_MARKERS} boundary tokens"
            )));
        }
        let mut seen = BTreeSet::new();
        for token in &parser.turn_open {
            if !vocabulary
                .boundary_tokens
                .iter()
                .any(|boundary| boundary.text == *token && !token.is_empty())
            {
                return Err(invalid(format!(
                    "format.parser turn_open '{token}' is not a boundary token"
                )));
            }
            if !seen.insert(token.as_str()) {
                return Err(invalid(format!(
                    "format.parser turn_open '{token}' is listed twice"
                )));
            }
        }
        if !parser
            .turn_open
            .iter()
            .any(|token| parser.assistant_header.starts_with(token.as_str()))
        {
            return Err(invalid(
                "format.parser assistant_header must begin with a turn_open boundary token",
            ));
        }
        Ok(())
    }

    /// A completion the shared parsers read ends at a stop token's text, so
    /// each one needs the text it is read by.
    fn validate_stop_texts(&self) -> Result<(), FrontendComponentErrorV1> {
        let vocabulary = &self.vocabulary;
        for id in &vocabulary.stop_token_ids {
            if !vocabulary
                .boundary_tokens
                .iter()
                .any(|token| token.id == *id)
            {
                return Err(invalid(format!(
                    "format.parser stop token {id} lacks a declared boundary text"
                )));
            }
        }
        Ok(())
    }

    /// Refuses a name byte set that could end a name early. Reading a tool
    /// or parameter name runs to the first of the grammar's name delimiters
    /// and trims whitespace, so a delimiter that begins with a name byte
    /// could begin inside a name, and the name would decode shorter than it
    /// rendered.
    fn validate_name_delimiters(
        &self,
        parser: &FrontendParserDeclarationV1,
    ) -> Result<(), FrontendComponentErrorV1> {
        // The shared parsers match their markers trimmed.
        let ends = match parser.tool_grammar {
            FrontendToolGrammarV1::XmlFunctionParametersV1 => vec![
                (&parser.function_name_close, "function_name_close"),
                (&parser.function_close, "function_close"),
                (&parser.parameter_name_close, "parameter_name_close"),
            ],
            FrontendToolGrammarV1::XmlKeyValueArgumentsV1 => vec![
                (&parser.parameter_open, "parameter_open"),
                (&parser.tool_call_close, "tool_call_close"),
                (&parser.parameter_name_close, "parameter_name_close"),
            ],
            FrontendToolGrammarV1::XmlTypedParametersV1 => vec![
                (&parser.function_name_close, "function_name_close"),
                (&parser.function_close, "function_close"),
                (&parser.parameter_name_close, "parameter_name_close"),
                (
                    &parser.parameter_json_name_close,
                    "parameter_json_name_close",
                ),
            ],
        };
        let delimiters: Vec<(&str, &str)> = ends
            .into_iter()
            .map(|(marker, field)| (marker.trim(), field))
            .collect();
        let names = &self.tools.name_bytes;
        for byte in (0..=127_u8).filter(|byte| names.permits(*byte)) {
            if byte.is_ascii_whitespace() {
                return Err(invalid(format!(
                    "tools.name_bytes {} admits whitespace, which reading a name trims",
                    names.describe()
                )));
            }
            if let Some((_, field)) = delimiters
                .iter()
                .find(|(marker, _)| marker.as_bytes().first() == Some(&byte))
            {
                return Err(invalid(format!(
                    "tools.name_bytes {} admits '{}', which begins the {field} that ends a name",
                    names.describe(),
                    char::from(byte)
                )));
            }
        }
        Ok(())
    }

    /// The reasoning controls as this format reads them: the keywords it
    /// knows, and the end markers its template writes.
    fn validate_reasoning_controls(
        &self,
        parser: &FrontendParserDeclarationV1,
    ) -> Result<(), FrontendComponentErrorV1> {
        let reasoning = &self.reasoning;
        for keyword in &reasoning.thinking_keywords {
            if !matches!(keyword.as_str(), "enable_thinking" | "thinking") {
                return Err(invalid(format!(
                    "thinking keyword '{keyword}' is not one this format reads"
                )));
            }
        }
        let (close, tool_open) = (&parser.reasoning_close, &parser.tool_call_open);
        // A marker ends reasoning on its own only when the vocabulary gives
        // it one identifier.
        let expected: BTreeSet<&str> = [close, tool_open]
            .into_iter()
            .filter(|marker| {
                self.vocabulary
                    .boundary_tokens
                    .iter()
                    .any(|token| token.text == **marker)
            })
            .map(String::as_str)
            .collect();
        let declared: BTreeSet<&str> = reasoning.end_markers.iter().map(String::as_str).collect();
        if declared != expected {
            return Err(invalid(format!(
                "reasoning end_markers must be {expected:?}: the template's reasoning close \
                 and tool-call open that are boundary tokens"
            )));
        }
        Ok(())
    }
}

/// The markers a tool grammar reads, which differ by grammar.
fn validate_tool_grammar_markers(
    parser: &FrontendParserDeclarationV1,
) -> Result<(), FrontendComponentErrorV1> {
    let function = [
        (&parser.function_open, "function_open"),
        (&parser.function_name_close, "function_name_close"),
        (&parser.function_close, "function_close"),
    ];
    let typed = parser.tool_grammar == FrontendToolGrammarV1::XmlTypedParametersV1;
    if typed == parser.parameter_json_name_close.is_empty() {
        return Err(invalid(
            "parameter_json_name_close is declared by the typed tool grammar, and only by it",
        ));
    }
    match parser.tool_grammar {
        FrontendToolGrammarV1::XmlFunctionParametersV1
        | FrontendToolGrammarV1::XmlTypedParametersV1 => {
            for (marker, name) in function {
                if marker.trim().is_empty() {
                    return Err(invalid(format!("format.parser {name} cannot be blank")));
                }
            }
            if !parser.parameter_value_open.is_empty() {
                return Err(invalid(
                    "parameter_value_open is declared only by the key-value tool grammar",
                ));
            }
            // The close a parameter's open tag ends in types its value, so
            // neither close may begin the other.
            let (raw, json) = (
                parser.parameter_name_close.trim(),
                parser.parameter_json_name_close.trim(),
            );
            if typed && (json.is_empty() || raw.starts_with(json) || json.starts_with(raw)) {
                return Err(invalid(
                    "format.parser parameter_json_name_close must be a nonblank close that \
                     neither begins nor is begun by parameter_name_close",
                ));
            }
        }
        FrontendToolGrammarV1::XmlKeyValueArgumentsV1 => {
            // The name follows the call open and ends where the first argument
            // or the call close begins, so there is no function element.
            for (marker, name) in function {
                if !marker.is_empty() {
                    return Err(invalid(format!(
                        "format.parser {name} must be empty for the key-value tool grammar"
                    )));
                }
            }
            if parser.parameter_value_open.trim().is_empty() {
                return Err(invalid(
                    "format.parser parameter_value_open cannot be blank",
                ));
            }
        }
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> FrontendComponentErrorV1 {
    FrontendComponentErrorV1::Invalid(message.into())
}

/// The parser section's reserved marker list.
fn validate_reserved_markers(reserved: &[String]) -> Result<(), FrontendComponentErrorV1> {
    if reserved.len() > MAX_MARKERS {
        return Err(invalid(format!(
            "{} reserved markers exceed the limit of {MAX_MARKERS}",
            reserved.len()
        )));
    }
    let mut seen = BTreeSet::new();
    for marker in reserved {
        if marker.is_empty() {
            return Err(invalid("a reserved marker cannot be empty"));
        }
        bounded_marker(marker, "reserved marker")?;
        if !seen.insert(marker.as_str()) {
            return Err(invalid(format!(
                "reserved marker '{marker}' is declared twice"
            )));
        }
    }
    Ok(())
}

fn bounded_marker(value: &str, name: &str) -> Result<(), FrontendComponentErrorV1> {
    if value.len() > MAX_MARKER_BYTES {
        return Err(invalid(format!(
            "{name} is {} bytes and the limit is {MAX_MARKER_BYTES}",
            value.len()
        )));
    }
    Ok(())
}
