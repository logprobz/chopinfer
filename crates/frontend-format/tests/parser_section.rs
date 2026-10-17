#![forbid(unsafe_code)]
//! The `format.parser` section: what the shared parsers and tool-choice
//! grammars read, which a model that renders its own prompt declares.
#[allow(dead_code)]
mod common;

use std::sync::Arc;

use frontend_format::{
    DeclaredParserV1, FormatDescriptorV1, FormatEventV1, FormatInputV1, FormatParserOptionsV1,
    FormatRenderRequestV1, FrontendParserDeclarationV1, FrontendToolGrammarV1, FrontendToolV1,
    PromptModeV1, parse_completion_v1,
};
use runtime_frontend_contract::FrontendComponentErrorV1;
use serde_json::{Value, json};

const CASES: &str = include_str!("function_parameter_cases.json");

fn fixtures() -> [FormatDescriptorV1; 2] {
    [common::KEY_VALUE, common::FUNCTION_PARAMETERS].map(|declaration| {
        FormatDescriptorV1::from_component_bytes(declaration.as_bytes()).unwrap()
    })
}

fn tools() -> Vec<FrontendToolV1> {
    let cases: Value = serde_json::from_str(CASES).unwrap();
    cases["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| {
            let function = &tool["function"];
            FrontendToolV1 {
                name: function["name"].as_str().unwrap().to_owned(),
                description: function["description"].as_str().map(str::to_owned),
                parameters: serde_json::from_value(function["parameters"].clone()).unwrap(),
                parameters_present: !function["parameters"].is_null(),
                strict: function["strict"].as_bool(),
            }
        })
        .collect()
}

fn completions() -> Vec<String> {
    let cases: Value = serde_json::from_str(CASES).unwrap();
    let mut texts: Vec<String> = cases["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| case["text"].as_str().unwrap().to_owned())
        .collect();
    texts.extend(
        [
            "plain answer",
            " \n<tool_call>lookup<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
            "<tool_call>lookup<arg_key>n</arg_key><arg_value>3</arg_value></tool_call>\n\
             <tool_call>read<arg_key>filter</arg_key><arg_value>{}</arg_value></tool_call>",
            "<tool_call>\n<function=read>\n<parameter=text>\nunfinished",
        ]
        .map(str::to_owned),
    );
    texts
}

/// The events one parser reports for `text` and the stop, pushed in two
/// parts.
fn events(
    descriptor: &FormatDescriptorV1,
    text: &str,
    mode: PromptModeV1,
    stream_tool_calls: bool,
) -> Result<Vec<FormatEventV1>, String> {
    let mut parser = DeclaredParserV1::new(
        Arc::new(descriptor.clone()),
        FormatParserOptionsV1 {
            mode,
            reasoning_open: mode == PromptModeV1::Thinking,
            extract_tools: true,
            stream_tool_calls,
            raw: false,
            ignore_eos: false,
            engine_tokens: false,
            tool_schemas: tools(),
        },
    );
    let completion = format!("{text}[EOS]");
    let bytes = completion.as_bytes();
    let mut half = completion.len() / 2;
    while !completion.is_char_boundary(half) {
        half -= 1;
    }
    let mut events = parser.push(FormatInputV1 {
        completion: &bytes[..half],
        held: false,
        last: false,
        stop_floor: Some(0),
    })?;
    events.extend(parser.push(FormatInputV1 {
        completion: bytes,
        held: false,
        last: true,
        stop_floor: Some(0),
    })?);
    Ok(events)
}

/// The function-parameter fixture reads the shared cases' completions with
/// the tool markers their parser section gives, which tools/qual reads too.
#[test]
fn the_function_parameter_fixture_reads_the_shared_cases_markers() {
    let cases: Value = serde_json::from_str(CASES).unwrap();
    let shared = &cases["parser"];
    let [_, fixture] = fixtures();
    let fixture = serde_json::to_value(fixture.parser).unwrap();
    for field in [
        "tool_grammar",
        "json_encoding",
        "reasoning_open",
        "reasoning_close",
        "tool_call_open",
        "tool_call_close",
        "function_open",
        "function_name_close",
        "function_close",
        "parameter_open",
        "parameter_name_close",
        "parameter_close",
    ] {
        assert_eq!(fixture[field], shared[field], "{field}");
    }
}

#[test]
fn a_parser_declaration_reads_checks_and_writes_back() {
    for descriptor in fixtures() {
        let parser = descriptor.parser.clone();
        let written = descriptor.clone().into_descriptor();
        assert_eq!(written.format, json!({ "parser": parser }));
        let bytes = serde_json::to_vec(&written).unwrap();
        assert_eq!(
            FormatDescriptorV1::from_declaration_bytes(&bytes).unwrap(),
            descriptor
        );
    }
}

/// The cases reach the tool grammar, not just text, and a completion reads
/// the same incrementally as whole.
#[test]
fn the_shared_parsers_read_the_tool_grammars() {
    for descriptor in fixtures() {
        assert!(completions().iter().any(|text| {
            events(&descriptor, text, PromptModeV1::Chat, false).is_ok_and(|events| {
                events
                    .iter()
                    .any(|event| matches!(event, FormatEventV1::ToolCalls { .. }))
            })
        }));
        for text in completions() {
            for mode in [PromptModeV1::Chat, PromptModeV1::Thinking] {
                let whole = parse_completion_v1(&descriptor, &format!("{text}[EOS]"), mode);
                for stream in [false, true] {
                    assert_eq!(
                        events(&descriptor, &text, mode, stream).is_ok(),
                        whole.is_ok(),
                        "{text:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_parser_declaration_is_checked() {
    let [valid, _] = fixtures();
    let refusal = |edit: &dyn Fn(&mut FrontendParserDeclarationV1)| {
        let mut descriptor = valid.clone();
        edit(&mut descriptor.parser);
        descriptor.validate().unwrap_err().to_string()
    };
    for (edit, message) in [
        (
            &(|parser: &mut FrontendParserDeclarationV1| parser.tool_call_open = " ".into())
                as &dyn Fn(&mut FrontendParserDeclarationV1),
            "format.parser tool_call_open cannot be blank",
        ),
        (
            &|parser| parser.reasoning_close = String::new(),
            "format.parser reasoning_close cannot be blank",
        ),
        (
            &|parser| parser.function_open = "<f>".into(),
            "format.parser function_open must be empty for the key-value tool grammar",
        ),
        (
            &|parser| parser.parameter_value_open = String::new(),
            "format.parser parameter_value_open cannot be blank",
        ),
        (
            &|parser| {
                parser.tool_grammar = FrontendToolGrammarV1::XmlFunctionParametersV1;
                parser.function_open = "<function=".into();
                parser.function_name_close = ">".into();
                parser.function_close = String::new();
                parser.parameter_value_open = String::new();
            },
            "format.parser function_close cannot be blank",
        ),
        (
            &|parser| parser.turn_open = Vec::new(),
            "format.parser turn_open lists 1 to",
        ),
        (
            &|parser| parser.turn_open.push("[X]".into()),
            "format.parser turn_open '[X]' is not a boundary token",
        ),
        (
            &|parser| parser.turn_open = vec!["[A]".into(), "[A]".into()],
            "format.parser turn_open '[A]' is listed twice",
        ),
        (
            &|parser| parser.assistant_header = "A[".into(),
            "format.parser assistant_header must begin with a turn_open boundary token",
        ),
        (
            &|parser| parser.reserved_markers.push(String::new()),
            "a reserved marker cannot be empty",
        ),
        (
            &|parser| parser.reasoning_close = "[A]".into(),
            "reasoning end_markers must be",
        ),
    ] {
        let error = refusal(edit);
        assert!(error.contains(message), "{error}");
    }

    // A format with a template beside the parser, no parser, or media
    // labels is refused by every reader and by the generic export.
    let parser = serde_json::to_value(valid.parser.clone()).unwrap();
    let mut host = valid.clone().into_descriptor();
    for (format, field) in [
        (
            json!({ "parser": parser, "template": {"json_encoding": "python-json-v1"} }),
            "template",
        ),
        (json!({}), "parser"),
        (
            json!({ "parser": parser, "media": {"image": {"prefix": [], "suffix": []}} }),
            "media",
        ),
    ] {
        host.format = format;
        for error in [
            FormatDescriptorV1::new(host.clone()).unwrap_err(),
            FormatDescriptorV1::read(host.clone()).unwrap_err(),
        ] {
            assert!(
                matches!(&error, FrontendComponentErrorV1::Syntax(message)
                    if message.contains(field)),
                "{field}: {error}"
            );
        }
        assert!(
            frontend_format::chat_format_export_v1::<Unrendered>()
                .open(&serde_json::to_vec(&host).unwrap())
                .is_err()
        );
    }
}

/// A chat format that opens over any declaration and renders nothing, for
/// tests of what the generic export opens over.
struct Unrendered;

impl frontend_format::ChatFormatV1 for Unrendered {
    fn open(_: &FormatDescriptorV1) -> Result<Self, String> {
        Ok(Self)
    }

    fn opened(&self) -> frontend_format::OpenedV1 {
        frontend_format::OpenedV1::default()
    }

    fn render(
        &self,
        _: FormatRenderRequestV1<'_>,
    ) -> Result<frontend_format::FormatRenderedV1, frontend_format::FrontendErrorV1> {
        Err(frontend_format::FrontendErrorV1::new("not rendered", None))
    }
}
