#![forbid(unsafe_code)]
//! A declared name byte set and the delimiters a tool grammar reads names
//! by: the format refuses a set that admits the first byte of a name
//! delimiter, and every name an accepted set admits decodes as it was
//! written, in each grammar.
mod common;

use frontend_format::{
    FormatDescriptorV1, FrontendParserDeclarationV1, FrontendToolGrammarV1, PromptModeV1,
    parse_completion_v1,
};
use serde_json::{Value, json};

/// Each parser fixture, with the delimiters a name runs to as the decoder
/// matches them.
fn forms() -> Vec<(&'static str, FormatDescriptorV1, Vec<String>)> {
    [
        ("function-parameter", common::FUNCTION_PARAMETERS),
        ("key-value", common::KEY_VALUE),
        ("typed-parameter", common::TYPED_PARAMETERS),
    ]
    .into_iter()
    .map(|(name, declaration)| {
        let mut descriptor =
            FormatDescriptorV1::from_component_bytes(declaration.as_bytes()).unwrap();
        descriptor.limits.maximum_name_bytes = 128;
        let parser = descriptor.parser.clone();
        let delimiters = match parser.tool_grammar {
            FrontendToolGrammarV1::XmlKeyValueArgumentsV1 => vec![
                parser.parameter_open,
                parser.tool_call_close,
                parser.parameter_name_close,
            ],
            FrontendToolGrammarV1::XmlFunctionParametersV1 => vec![
                parser.function_name_close,
                parser.function_close,
                parser.parameter_name_close,
            ],
            FrontendToolGrammarV1::XmlTypedParametersV1 => vec![
                parser.function_name_close,
                parser.function_close,
                parser.parameter_name_close,
                parser.parameter_json_name_close,
            ],
        };
        let delimiters = delimiters.iter().map(|d| d.trim().to_owned()).collect();
        (name, descriptor, delimiters)
    })
    .collect()
}

/// `descriptor` with `extra` as its name bytes beside the alphanumerics.
fn with_extra(descriptor: &FormatDescriptorV1, extra: &str) -> FormatDescriptorV1 {
    let mut changed = descriptor.clone();
    changed.tools.name_bytes.extra = extra.to_owned();
    changed
}

/// Whether the host sections alone accept `extra`.
fn host_accepts(descriptor: &FormatDescriptorV1, extra: &str) -> bool {
    with_extra(descriptor, extra)
        .into_descriptor()
        .validate()
        .is_ok()
}

#[test]
fn a_name_byte_that_begins_a_name_delimiter_is_refused() {
    for (form, descriptor, delimiters) in forms() {
        let mut refused = 0;
        for delimiter in &delimiters {
            let byte = char::from(delimiter.as_bytes()[0]);
            let extra = format!("_{byte}");
            if !host_accepts(&descriptor, &extra) {
                continue;
            }
            let error = with_extra(&descriptor, &extra)
                .validate()
                .expect_err(&format!("{form}: '{byte}' begins {delimiter:?}"));
            assert!(
                error.to_string().contains("that ends a name"),
                "{form}: {error}"
            );
            refused += 1;
        }
        assert!(refused > 0, "{form}: no delimiter byte the host accepts");
    }
}

/// One call to `name` with one parameter `parameter`, written in the
/// parser section's tool grammar as a renderer writes it, decoded as a
/// completion.
fn round_trip(descriptor: &FormatDescriptorV1, name: &str, parameter: &str) -> (String, Value) {
    let FrontendParserDeclarationV1 {
        tool_call_open,
        tool_call_close,
        function_open,
        function_name_close,
        function_close,
        parameter_open,
        parameter_name_close,
        parameter_value_open,
        parameter_close,
        ..
    } = &descriptor.parser;
    let completion = format!(
        "{tool_call_open}{function_open}{name}{function_name_close}{parameter_open}{parameter}\
         {parameter_name_close}{parameter_value_open}v{parameter_close}{function_close}\
         {tool_call_close}[EOS]"
    );
    let parsed = parse_completion_v1(descriptor, &completion, PromptModeV1::Chat)
        .unwrap_or_else(|error| panic!("{completion:?} decodes: {error}"));
    assert_eq!(parsed.tool_calls.len(), 1, "{completion:?}");
    let call = &parsed.tool_calls[0];
    (
        call.name.clone(),
        serde_json::from_str(&call.arguments).unwrap(),
    )
}

#[test]
fn every_name_an_accepted_set_admits_decodes_as_it_was_written() {
    for (form, descriptor, delimiters) in forms() {
        // Each punctuation byte the host accepts beside `_`: the format
        // either refuses the set, or names that begin and end with the byte
        // decode as they were written.
        let mut accepted = String::new();
        for byte in (b'!'..=b'~').filter(u8::is_ascii_punctuation) {
            let byte = char::from(byte);
            let extra = if byte == '-' {
                "_-".to_owned()
            } else {
                format!("{byte}_")
            };
            if byte == '_' || !host_accepts(&descriptor, &extra) {
                continue;
            }
            let changed = with_extra(&descriptor, &extra);
            if changed.validate().is_err() {
                continue;
            }
            for (name, parameter) in [
                (format!("{byte}x{byte}"), format!("{byte}y{byte}")),
                (format!("x{byte}_"), format!("y{byte}_")),
            ] {
                assert_eq!(
                    round_trip(&changed, &name, &parameter),
                    (name.clone(), json!({ parameter.clone(): "v" })),
                    "{form}"
                );
            }
            if byte != '-' {
                accepted.push(byte);
            }
        }
        // Every byte accepted alone is accepted together, and the
        // delimiters' first bytes are the only punctuation refused.
        let begins = |byte: u8| delimiters.iter().any(|d| d.as_bytes()[0] == byte);
        let expected: String = (b'!'..=b'~')
            .filter(|byte| byte.is_ascii_punctuation() && !b"[]^\\-_".contains(byte))
            .filter(|byte| !begins(*byte))
            .map(char::from)
            .collect();
        assert_eq!(accepted, expected, "{form}");
        let extra = format!("{accepted}_-");
        let descriptor = with_extra(&descriptor, &extra);
        descriptor
            .validate()
            .unwrap_or_else(|error| panic!("{form}: {extra:?}: {error}"));
        let (name, parameter) = (format!("{extra}x{extra}"), format!("{extra}y"));
        assert_eq!(
            round_trip(&descriptor, &name, &parameter),
            (name.clone(), json!({ parameter.clone(): "v" })),
            "{form}"
        );
    }
}
