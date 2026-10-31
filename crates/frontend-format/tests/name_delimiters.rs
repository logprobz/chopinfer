#![forbid(unsafe_code)]
//! A declared name byte set and the delimiters a tool grammar reads names
//! by: the format refuses a set that admits the first byte of a name
//! delimiter, and every name an accepted set admits decodes as it was
//! written, in each grammar. A parameter name need not be in the set: any
//! name that holds no marker decodes as it was written, and the tool-choice
//! grammar writes it.
mod common;

use std::sync::Arc;

use frontend_format::toolkit::validate_parameter_name_v1;
use frontend_format::{
    FormatDescriptorV1, FrontendJsonV1, FrontendParserDeclarationV1, FrontendToolChoiceV1,
    FrontendToolGrammarV1, FrontendToolV1, PromptModeV1, parse_completion_v1, tool_constraint_v1,
};
use runtime_grammar::{CompiledGrammar, GrammarMatcher, TokenVocabulary, VocabularyToken};
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

/// Parameter names other than the declared name bytes: a dotted name, a
/// non-ASCII one and one with inner whitespace, as `encoding.py` reads any
/// `name="..."`.
const OPEN_NAMES: [&str; 4] = ["user.name", "名前", "first name", "a/b:c@d"];

#[test]
fn any_parameter_name_without_a_marker_decodes_as_it_was_written() {
    for (form, descriptor, _) in forms() {
        for parameter in OPEN_NAMES {
            assert_eq!(
                round_trip(&descriptor, "lookup", parameter),
                ("lookup".to_owned(), json!({ parameter: "v" })),
                "{form}: {parameter:?}"
            );
        }
    }
}

/// The markers a parameter name may not hold: every bare tool-call marker
/// the parser reads and every reserved marker.
fn name_markers(descriptor: &FormatDescriptorV1) -> Vec<String> {
    let parser = &descriptor.parser;
    [
        &parser.tool_call_open,
        &parser.tool_call_close,
        &parser.function_open,
        &parser.function_name_close,
        &parser.function_close,
        &parser.parameter_open,
        &parser.parameter_name_close,
        &parser.parameter_json_name_close,
        &parser.parameter_close,
    ]
    .into_iter()
    .map(|marker| marker.trim().to_owned())
    .chain(parser.reserved_markers.iter().cloned())
    .filter(|marker| !marker.is_empty())
    .collect()
}

/// A parsed parameter name that holds the parameter open is refused, as a
/// name that holds any marker is: it is not a name a call could be sent back
/// with.
#[test]
fn a_parsed_parameter_name_holding_a_marker_is_refused() {
    for (form, descriptor, _) in forms() {
        let open = descriptor.parser.parameter_open.trim().to_owned();
        let parameter = format!("a{open}b");
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
            "{tool_call_open}{function_open}f{function_name_close}{parameter_open}{parameter}\
             {parameter_name_close}{parameter_value_open}v{parameter_close}{function_close}\
             {tool_call_close}[EOS]"
        );
        assert!(
            parse_completion_v1(&descriptor, &completion, PromptModeV1::Chat).is_err(),
            "{form}: {completion:?}"
        );
    }
}

/// One tool `f` whose string properties are `names`, `c` required.
fn tool(names: &[&str], required: &[&str]) -> FrontendToolV1 {
    let properties: serde_json::Map<String, Value> = names
        .iter()
        .map(|name| ((*name).to_owned(), json!({"type": "string"})))
        .collect();
    FrontendToolV1 {
        name: "f".into(),
        description: None,
        parameters: serde_json::from_value::<FrontendJsonV1>(json!({
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false
        }))
        .unwrap(),
        parameters_present: true,
        strict: None,
    }
}

/// One call to `f` with each `(name, value)` string argument, as the
/// tool-choice grammar writes it.
fn forced_call(descriptor: &FormatDescriptorV1, arguments: &[(&str, &str)]) -> String {
    let parser = &descriptor.parser;
    let mut text = format!(
        "{}{}f{}",
        parser.tool_call_open, parser.function_open, parser.function_name_close
    );
    for (name, value) in arguments {
        text += &format!(
            "{}{name}{}{}{value}{}",
            parser.parameter_open,
            parser.parameter_name_close,
            parser.parameter_value_open,
            parser.parameter_close
        );
    }
    text + &parser.function_close + &parser.tool_call_close
}

/// Whether the grammar a required choice imposes on `tools` admits `text`.
fn admits(descriptor: &FormatDescriptorV1, tools: &[FrontendToolV1], text: &str) -> bool {
    let grammar = tool_constraint_v1(descriptor, tools, &FrontendToolChoiceV1::Required)
        .unwrap()
        .expect("a required choice constrains");
    let tokens = (0..=255_u8)
        .map(|byte| VocabularyToken {
            bytes: vec![byte],
            special: false,
        })
        .collect();
    let vocabulary = Arc::new(TokenVocabulary::new(256, tokens).unwrap());
    let compiled = Arc::new(CompiledGrammar::new(&grammar, vocabulary).unwrap());
    let mut matcher = GrammarMatcher::new(compiled);
    matcher.accept_bytes(text.as_bytes()).is_ok() && matcher.is_complete()
}

/// A forced call writes an optional property with any name the parser reads
/// back, and serves a required one; a name that holds a marker is never
/// written, and a tool that requires one is refused.
#[test]
fn a_forced_call_writes_every_name_the_parser_reads_back() {
    for (form, descriptor, _) in forms() {
        for name in OPEN_NAMES {
            let optional = [tool(&[name, "c"], &["c"])];
            let text = forced_call(&descriptor, &[(name, "v"), ("c", "w")]);
            assert!(admits(&descriptor, &optional, &text), "{form}: {text:?}");
            assert_eq!(
                parse_completion_v1(&descriptor, &format!("{text}[EOS]"), PromptModeV1::Chat)
                    .map(|parsed| parsed.tool_calls[0].arguments.clone())
                    .map(|arguments| serde_json::from_str::<Value>(&arguments).unwrap()),
                Ok(json!({name: "v", "c": "w"})),
                "{form}: {text:?}"
            );
            let required = [tool(&[name], &[name])];
            let text = forced_call(&descriptor, &[(name, "v")]);
            assert!(admits(&descriptor, &required, &text), "{form}: {text:?}");
        }
        for marker in name_markers(&descriptor) {
            // Both names sort before `c`, so only the marker differs.
            let readable = "a.b";
            let marked = format!("a{marker}b");
            let control = [tool(&[readable, "c"], &["c"])];
            let text = forced_call(&descriptor, &[(readable, "v"), ("c", "w")]);
            assert!(admits(&descriptor, &control, &text), "{form}: {text:?}");
            let optional = [tool(&[&marked, "c"], &["c"])];
            let text = forced_call(&descriptor, &[(&marked, "v"), ("c", "w")]);
            assert!(!admits(&descriptor, &optional, &text), "{form}: {text:?}");
            let only = forced_call(&descriptor, &[("c", "w")]);
            assert!(admits(&descriptor, &optional, &only), "{form}: {only:?}");
            let required = [tool(&[&marked], &[&marked])];
            let error = tool_constraint_v1(&descriptor, &required, &FrontendToolChoiceV1::Required)
                .unwrap_err();
            assert_eq!(
                error.parameter.as_deref(),
                Some("tools"),
                "{form}: {marker}"
            );
        }
    }
}

/// `descriptor` with `close` as its declared parameter name close.
fn with_name_close(descriptor: &FormatDescriptorV1, close: &str) -> FormatDescriptorV1 {
    let mut changed = descriptor.clone();
    changed.parser.parameter_name_close = close.to_owned();
    changed
        .validate()
        .unwrap_or_else(|error| panic!("{close:?}: {error}"));
    changed
}

/// Whether one call to `f` with the string argument `name` = `v`, written
/// as the tool-choice grammar writes it, reads back as written.
fn reads_back(descriptor: &FormatDescriptorV1, name: &str) -> bool {
    let text = format!("{}[EOS]", forced_call(descriptor, &[(name, "v")]));
    parse_completion_v1(descriptor, &text, PromptModeV1::Chat).is_ok_and(|parsed| {
        parsed.tool_calls.len() == 1
            && serde_json::from_str::<Value>(&parsed.tool_calls[0].arguments).ok()
                == Some(json!({ name: "v" }))
    })
}

/// A parameter name inside which a marker the parser stops at while it
/// reads the name begins, completed by the name close written after it, does
/// not read back: under a declared `]]` close, `a]` reads as `a` with the
/// value `]v`, or its argument is dropped, and a call close or stop text the
/// close completes ends the call or the completion early. Such a name is
/// refused, the tool-choice grammar never writes it, and a tool that
/// requires one is refused. A name no marker overlaps reads back.
#[test]
fn a_name_that_a_marker_overlaps_across_its_close_is_refused() {
    for (form, descriptor, _) in forms() {
        let parser = &descriptor.parser;
        // The marker that ends the call around a name, without its last `>`,
        // which a `>` name close then completes.
        let call_end = match parser.tool_grammar {
            FrontendToolGrammarV1::XmlKeyValueArgumentsV1 => parser.tool_call_close.trim(),
            _ => parser.function_close.trim(),
        }
        .strip_suffix('>')
        .expect("the call ends with `>`")
        .to_owned();
        let mut cases = vec![
            (
                "]]".to_owned(),
                with_name_close(&descriptor, "]]"),
                vec!["a]".to_owned(), "a[EOS".to_owned()],
                vec!["a]b", "[a"],
            ),
            (
                ">".to_owned(),
                with_name_close(&descriptor, ">"),
                vec![format!("a{call_end}")],
                vec!["a<b", "a/b"],
            ),
        ];
        // The declared close, where it completes the call's end.
        if parser.parameter_name_close.starts_with('>') {
            cases.push((
                parser.parameter_name_close.clone(),
                descriptor.clone(),
                vec![format!("a{call_end}")],
                vec!["user.name"],
            ));
        }
        for (close, declared, refused, accepted) in cases {
            for name in &refused {
                let label = format!("{form}: close {close:?}, name {name:?}");
                assert!(!reads_back(&declared, name), "{label} reads back");
                assert!(
                    validate_parameter_name_v1(&declared, name, "parameter").is_err(),
                    "{label}"
                );
                let optional = [tool(&[name, "c"], &["c"])];
                let written = forced_call(&declared, &[(name, "v"), ("c", "w")]);
                assert!(!admits(&declared, &optional, &written), "{label}");
                let only = forced_call(&declared, &[("c", "w")]);
                assert!(admits(&declared, &optional, &only), "{label}");
                let required = [tool(&[name], &[name])];
                let error =
                    tool_constraint_v1(&declared, &required, &FrontendToolChoiceV1::Required)
                        .unwrap_err();
                assert_eq!(error.parameter.as_deref(), Some("tools"), "{label}");
            }
            for name in accepted {
                let label = format!("{form}: close {close:?}, name {name:?}");
                assert!(reads_back(&declared, name), "{label}");
                assert!(
                    validate_parameter_name_v1(&declared, name, "parameter").is_ok(),
                    "{label}"
                );
                let optional = [tool(&[name, "c"], &["c"])];
                let written = forced_call(&declared, &[(name, "v"), ("c", "w")]);
                assert!(admits(&declared, &optional, &written), "{label}");
            }
        }
    }
}
