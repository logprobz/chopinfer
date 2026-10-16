//! A chat format called directly and through the frontend documents answer
//! every fixture request and completion of this crate the same way.

use std::sync::Arc;

use runtime_server::{
    DecodingOptions, FrontendChatRequestV1, FrontendJsonV1, FrontendMessageV1, FrontendRoleV1,
    FrontendToolCallV1, FrontendToolChoiceV1, FrontendToolV1, ServerFrontendV1,
};

use crate::decode::test_common::{FUNCTION_PARAMETERS, KEY_VALUE};
use crate::test_support::documents::{PairedDecoderV1, PairedFrontendV1};
use crate::{
    CompletionUpdateV1, DecoderLimitsV1, FrontendDescriptorV1, FrontendToolArgumentsV1,
    GeneratedTokenV1, PromptModeV1,
};
use frontend_format::test_support::FixtureFormatV1;
use frontend_format::toolkit::preserve_thinking_v1;
use frontend_format::{
    ChatFormatV1, FormatDescriptorV1, FormatRenderRequestV1, FormatRenderedV1, FrontendErrorV1,
    FrontendParserDeclarationV1, FrontendToolGrammarV1, OpenedV1,
};

/// The function-parameter cases tools/qual reads too.
use frontend_format::test_support::FUNCTION_PARAMETER_CASES as SHARED_CASES;

fn shared_cases() -> serde_json::Value {
    serde_json::from_str(SHARED_CASES).unwrap()
}

/// The parser fixtures, one on each shared tool grammar.
fn parser_fixtures() -> Vec<FrontendDescriptorV1> {
    [KEY_VALUE, FUNCTION_PARAMETERS]
        .into_iter()
        .map(|declaration| FrontendDescriptorV1::from_component_bytes(declaration.as_bytes()))
        .collect::<Result<_, _>>()
        .unwrap()
}

/// The function-parameter fixture, whose parser section is the shared
/// cases'.
fn function_parameters() -> FrontendDescriptorV1 {
    FrontendDescriptorV1::from_component_bytes(FUNCTION_PARAMETERS.as_bytes()).unwrap()
}

/// `descriptor` served by the fixture format, both ways.
fn paired(descriptor: FrontendDescriptorV1) -> PairedFrontendV1 {
    PairedFrontendV1::chat_format::<FixtureFormatV1>(descriptor).unwrap()
}

/// A declaration's parser section, read.
fn parser(descriptor: &FrontendDescriptorV1) -> FrontendParserDeclarationV1 {
    FormatDescriptorV1::read(descriptor.clone()).unwrap().parser
}

/// The text of a declaration's first stop token.
fn stop_text(descriptor: &FrontendDescriptorV1) -> String {
    let stop = descriptor.vocabulary.stop_token_ids[0];
    descriptor
        .vocabulary
        .boundary_tokens
        .iter()
        .find(|token| token.id == stop)
        .unwrap()
        .text
        .clone()
}

fn tool(name: &str) -> FrontendToolV1 {
    FrontendToolV1 {
        name: name.to_owned(),
        description: Some("Find a city".to_owned()),
        parameters: serde_json::from_str(
            r#"{"type":"object","properties":{"city":{"type":"string"},"n":{"type":"integer"}}}"#,
        )
        .unwrap(),
        parameters_present: true,
        strict: None,
    }
}

/// One call to `name` with one `city` argument, in the declaration's tool
/// grammar.
fn call_block(descriptor: &FrontendDescriptorV1, name: &str, value: &str) -> String {
    let parser = parser(descriptor);
    let argument = format!(
        "{}city{}{}{value}{}",
        parser.parameter_open,
        parser.parameter_name_close,
        parser.parameter_value_open,
        parser.parameter_close
    );
    match parser.tool_grammar {
        FrontendToolGrammarV1::XmlKeyValueArgumentsV1 => format!(
            "{}{name}{argument}{}",
            parser.tool_call_open, parser.tool_call_close
        ),
        FrontendToolGrammarV1::XmlFunctionParametersV1 => format!(
            "{}{}{name}{}{argument}{}{}",
            parser.tool_call_open,
            parser.function_open,
            parser.function_name_close,
            parser.function_close,
            parser.tool_call_close
        ),
    }
}

#[derive(Clone, Copy, Debug)]
enum EndV1 {
    /// The text carries its own end, or none.
    Text,
    /// Generation ran out of tokens.
    Length,
    /// The engine stopped on a stop identifier it sampled.
    StopId,
}

/// Decodes `text` split at `split` both ways, with every host setting the
/// pair varies; the pair asserts that each push and finish agrees.
#[allow(clippy::too_many_arguments)]
fn drive(
    descriptor: &Arc<FrontendDescriptorV1>,
    mode: PromptModeV1,
    text: &[u8],
    split: usize,
    engine: bool,
    stream: bool,
    extract: bool,
    stops: &[String],
    minimum: usize,
    end: EndV1,
) {
    let limits = DecoderLimitsV1::declared(descriptor);
    let mut decoder = if stops.is_empty() {
        PairedDecoderV1::new(Arc::clone(descriptor), mode, limits)
    } else {
        PairedDecoderV1::with_client_stops(Arc::clone(descriptor), mode, limits, stops).unwrap()
    };
    decoder.set_decoding_options(
        DecodingOptions {
            stream_tool_calls: stream,
            ..Default::default()
        },
        false,
    );
    decoder.set_tool_schemas(&[tool("lookup")]);
    decoder.set_tool_extraction(extract);
    decoder.set_minimum_new_tokens(minimum);
    let mut ordinal = 0;
    let mut push = |decoder: &mut PairedDecoderV1, token_id: u32, bytes: &[u8]| {
        let token = GeneratedTokenV1::new(ordinal, token_id, bytes.to_vec());
        ordinal += 1;
        let updates = if engine {
            decoder.push_engine_token(token)
        } else {
            decoder.push(token)
        };
        updates.map_or(true, |updates| {
            updates
                .iter()
                .any(|update| matches!(update, CompletionUpdateV1::Finished(_)))
        })
    };
    for chunk in [&text[..split], &text[split..]] {
        if !chunk.is_empty() && push(&mut decoder, 200, chunk) {
            return;
        }
    }
    match end {
        EndV1::Text => {}
        EndV1::Length => {
            let _ = decoder.finish_length();
        }
        EndV1::StopId => {
            let stop = descriptor.vocabulary.stop_token_ids[0];
            let boundary = descriptor
                .vocabulary
                .boundary_tokens
                .iter()
                .find(|boundary| boundary.id == stop)
                .map_or(b"".as_slice(), |boundary| boundary.text.as_bytes());
            if !boundary.is_empty() {
                push(&mut decoder, stop, boundary);
            } else {
                let _ = decoder.finish_stop();
            }
        }
    }
}

/// Every completion of a fixture: at every split with engine tokens and
/// streamed calls, and at a few splits with every other combination of push
/// kind, streaming, extraction, client stops, minimum and ending.
fn sweep(descriptor: &FrontendDescriptorV1, texts: &[(PromptModeV1, String)]) {
    let descriptor = Arc::new(descriptor.clone());
    let stop_sets = [Vec::new(), vec!["or".to_owned()]];
    for (mode, text) in texts {
        let bytes = text.as_bytes();
        for split in 0..=bytes.len() {
            drive(
                &descriptor,
                *mode,
                bytes,
                split,
                true,
                true,
                true,
                &[],
                0,
                EndV1::Text,
            );
        }
        let mut splits = vec![0, 1, bytes.len() / 2, bytes.len().saturating_sub(1)];
        splits.dedup();
        for split in splits {
            for engine in [false, true] {
                for (stream, extract) in [(false, true), (true, true), (false, false)] {
                    for stops in &stop_sets {
                        for minimum in [0, 2] {
                            for end in [EndV1::Text, EndV1::Length, EndV1::StopId] {
                                drive(
                                    &descriptor,
                                    *mode,
                                    bytes,
                                    split,
                                    engine,
                                    stream,
                                    extract,
                                    stops,
                                    minimum,
                                    end,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn parser_fixture_completions_decode_the_same_both_ways() {
    for descriptor in parser_fixtures() {
        let parser = parser(&descriptor);
        let end = stop_text(&descriptor);
        let close = &parser.reasoning_close;
        let block = call_block(&descriptor, "lookup", "Paris");
        let texts = [
            (PromptModeV1::Chat, format!("hello world{end}")),
            (PromptModeV1::Thinking, format!("plan{close}answer{end}")),
            (PromptModeV1::Chat, format!("lead{block}{end}")),
            (PromptModeV1::Thinking, format!("plan{close}{block}{end}")),
            (PromptModeV1::Chat, "no end, café".to_owned()),
            (
                PromptModeV1::Chat,
                format!("x{}y{end}", parser.reserved_markers[0]),
            ),
            (PromptModeV1::Chat, format!("bad{block}{block}tail{end}")),
        ];
        sweep(&descriptor, &texts);
    }
}

#[test]
fn shared_function_parameter_completions_decode_the_same_both_ways() {
    let descriptor = function_parameters();
    let end = stop_text(&descriptor);
    let mut texts = Vec::new();
    for case in shared_cases()["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        texts.push((PromptModeV1::Thinking, format!("plan</think>{text}{end}")));
        texts.push((PromptModeV1::Chat, format!("{text}{end}")));
    }
    sweep(&descriptor, &texts);
}

fn message(role: FrontendRoleV1, content: &str) -> FrontendMessageV1 {
    FrontendMessageV1 {
        role,
        content: Some(content.to_owned()),
        parts: None,
        reasoning_content: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
        tools: Vec::new(),
    }
}

fn request(messages: Vec<FrontendMessageV1>) -> FrontendChatRequestV1 {
    FrontendChatRequestV1 {
        messages,
        continue_final_message: false,
        add_generation_prompt: true,
        tools: Vec::new(),
        tool_choice: FrontendToolChoiceV1::Auto,
        response_format: None,
        reasoning_effort: None,
        enable_thinking: None,
        thinking: None,
        nested_reasoning_effort: None,
        preserve_thinking: None,
        clear_thinking: None,
        add_vision_id: None,
        tool_objects: Vec::new(),
        chat_template_kwargs: None,
    }
}

fn json(text: &str) -> FrontendJsonV1 {
    serde_json::from_str(text).unwrap()
}

/// Requests over every field a render reads.
fn requests(descriptor: &FrontendDescriptorV1) -> Vec<FrontendChatRequestV1> {
    let user = || message(FrontendRoleV1::User, "What is the weather?");
    let mut assistant = message(FrontendRoleV1::Assistant, "Checking.");
    assistant.reasoning_content = Some("I should look it up.".to_owned());
    assistant.tool_calls = vec![FrontendToolCallV1 {
        id: "call-1".to_owned(),
        name: "lookup".to_owned(),
        arguments: r#"{"city": "Paris", "n": 2}"#.to_owned(),
    }];
    let mut result = message(FrontendRoleV1::Tool, "sunny");
    result.tool_call_id = Some("call-1".to_owned());
    let history = vec![
        message(FrontendRoleV1::System, "Be helpful."),
        user(),
        assistant.clone(),
        result,
        user(),
    ];
    let mut requests = vec![
        request(vec![user()]),
        request(vec![message(FrontendRoleV1::Developer, "Rules."), user()]),
        request(history.clone()),
    ];
    let mut with_tools = request(history.clone());
    with_tools.tools = vec![
        tool("lookup"),
        FrontendToolV1 {
            description: None,
            parameters_present: false,
            parameters: FrontendJsonV1::Object(Vec::new()),
            strict: Some(true),
            ..tool("other")
        },
    ];
    with_tools.tool_objects = with_tools
        .tools
        .iter()
        .map(|tool| {
            FrontendJsonV1::Object(vec![(
                "name".into(),
                FrontendJsonV1::String(tool.name.clone()),
            )])
        })
        .collect();
    requests.push(with_tools);
    for format in [
        r#"{"type":"json_object"}"#,
        r#"{"type":"json_schema","x":1.5}"#,
    ] {
        let mut formatted = request(vec![user()]);
        formatted.response_format = Some(json(format));
        requests.push(formatted);
    }
    let mut efforts = vec![
        None,
        Some(descriptor.reasoning.disable_value.clone()),
        Some("bogus".to_owned()),
        Some("high".to_owned()),
    ];
    efforts.extend(
        descriptor
            .reasoning
            .effort_names()
            .into_iter()
            .map(|name| Some(name.to_owned())),
    );
    for effort in &efforts {
        for enable in [None, Some(true), Some(false)] {
            let mut controlled = request(history.clone());
            controlled.reasoning_effort.clone_from(effort);
            controlled.enable_thinking = enable;
            requests.push(controlled.clone());
            controlled.reasoning_effort = None;
            controlled.nested_reasoning_effort.clone_from(effort);
            controlled.thinking = enable;
            requests.push(controlled);
        }
    }
    for (preserve, clear) in [
        (Some(false), None),
        (None, Some(json("true"))),
        (None, Some(json("\"yes\""))),
    ] {
        let mut kept = request(history.clone());
        kept.preserve_thinking = preserve;
        kept.clear_thinking = clear;
        requests.push(kept);
    }
    let mut vision = request(vec![user()]);
    vision.add_vision_id = Some(json("true"));
    requests.push(vision);
    let mut continued = request(vec![user(), message(FrontendRoleV1::Assistant, "It is")]);
    continued.continue_final_message = true;
    continued.add_generation_prompt = false;
    requests.push(continued);
    let mut closed = request(vec![user()]);
    closed.add_generation_prompt = false;
    requests.push(closed);
    let mut missing = request(vec![message(FrontendRoleV1::Tool, "orphan")]);
    missing.messages[0].tool_call_id = None;
    requests.push(missing);
    requests
}

#[test]
fn fixture_requests_render_the_same_both_ways() {
    for descriptor in parser_fixtures() {
        let frontend = paired(descriptor.clone());
        for request in requests(&descriptor) {
            let _ = frontend.render_chat(request.clone());
            for choice in [
                FrontendToolChoiceV1::Auto,
                FrontendToolChoiceV1::None,
                FrontendToolChoiceV1::Required,
                FrontendToolChoiceV1::Named("lookup".to_owned()),
                FrontendToolChoiceV1::Named("missing".to_owned()),
            ] {
                let _ = frontend.tool_constraint(&request.tools, &choice);
            }
        }
        for (name, arguments) in [
            ("lookup", r#"{"city":"Paris"}"#),
            ("lookup", r#"{"n":1.5}"#),
            ("lookup", "[1]"),
            ("lookup", "not json"),
            ("look up", "{}"),
        ] {
            let _ = frontend.validate_tool_call(name, arguments);
        }
    }
}

/// A backend tool call's arguments take the declared shape: integral
/// numbers where the declaration names them, for a format that writes them
/// back, and otherwise any JSON object.
#[test]
fn tool_call_arguments_take_the_declared_shape() {
    let invalid = Err(runtime_server::FrontendErrorV1::new(
        "backend tool call arguments are invalid",
        None,
    ));
    let integral = [
        r#"{"n":10}"#,
        r#"{"n":-3,"m":[18446744073709551616,{"k":0}]}"#,
        r#"{"s":"1.5","b":true,"z":null}"#,
    ];
    let fractional = [r#"{"n":1.5}"#, r#"{"n":[1e3]}"#, r#"{"o":{"n":2E-1}}"#];
    for mut descriptor in parser_fixtures() {
        descriptor.tools.arguments = FrontendToolArgumentsV1::JsonObjectIntegralNumbers;
        let frontend = paired(descriptor);
        for arguments in integral {
            assert_eq!(frontend.validate_tool_call("lookup", arguments), Ok(()));
        }
        for arguments in fractional {
            assert_eq!(
                frontend.validate_tool_call("lookup", arguments),
                invalid,
                "{arguments}"
            );
        }
    }
    let frontend = paired(function_parameters());
    for arguments in integral.into_iter().chain(fractional) {
        assert_eq!(frontend.validate_tool_call("lookup", arguments), Ok(()));
    }
    assert_eq!(
        frontend.validate_tool_call("lookup", "[1.5]"),
        Err(runtime_server::FrontendErrorV1::new(
            "backend tool call arguments must be a JSON object",
            None,
        ))
    );
}

/// Tool-call arguments past the declared JSON byte limit decode on both
/// paths and are refused where they always were: by the transport's check
/// of the backend call.
#[test]
fn over_limit_tool_arguments_are_refused_by_the_tool_call_check() {
    let mut descriptor = function_parameters();
    descriptor.limits.maximum_json_bytes = 64;
    descriptor.validate().unwrap();
    let declaration = Arc::new(descriptor.clone());
    let value = "x".repeat(100);
    let text = format!(
        "<tool_call>\n<function=lookup>\n<parameter=city>\n{value}\n</parameter>\n</function>\n</tool_call>[EOS]"
    );
    let mut decoder = PairedDecoderV1::new(
        Arc::clone(&declaration),
        PromptModeV1::Chat,
        DecoderLimitsV1::declared(&declaration),
    );
    decoder.set_tool_schemas(&[tool("lookup")]);
    let updates = decoder
        .push(GeneratedTokenV1::new(0, 200, text.into_bytes()))
        .expect("the completion decodes");
    let arguments = updates
        .iter()
        .find_map(|update| match update {
            CompletionUpdateV1::ToolCalls(group) => Some(group.calls[0].arguments.clone()),
            _ => None,
        })
        .expect("the call is delivered");
    assert!(arguments.len() > 64);
    let frontend = paired(descriptor);
    assert_eq!(
        frontend.validate_tool_call("lookup", &arguments),
        Err(runtime_server::FrontendErrorV1::new(
            "backend tool call arguments are invalid",
            None
        ))
    );
}

/// A forced call over 128 large tools builds a grammar past the
/// construction budget's expression bytes. Both paths answer the same
/// grammar, and it fails where it always has: at compilation, with
/// compilation's message, which admission reports as
/// `constraint: the constraint is too large`.
#[test]
fn an_oversized_tool_constraint_fails_at_compilation_on_both_paths() {
    use runtime_grammar::{CompiledGrammar, TokenVocabulary, VocabularyToken};
    let property = r#"{"type":"object","properties":{"a":{"type":"array","items":{"type":"object","properties":{"x":{"type":"integer"},"y":{"enum":["u","v"]},"z":{"type":"string","pattern":"^[a-z]+$"}}}},"b":{"type":["number","null"]}}}"#;
    let properties = (0..20)
        .map(|index| format!(r#""p{index}":{property}"#))
        .collect::<Vec<_>>()
        .join(",");
    let parameters: FrontendJsonV1 = json(&format!(
        r#"{{"type":"object","properties":{{{properties}}}}}"#
    ));
    let tools: Vec<FrontendToolV1> = (0..128)
        .map(|index| FrontendToolV1 {
            parameters: parameters.clone(),
            ..tool(&format!("t{index}"))
        })
        .collect();
    let frontend = paired(function_parameters());
    let grammar = frontend
        .tool_constraint(&tools, &FrontendToolChoiceV1::Required)
        .expect("the transport accepts the request")
        .expect("a forced call has a grammar");
    let mut tokens: Vec<VocabularyToken> = (0..=255_u8)
        .map(|byte| VocabularyToken {
            bytes: vec![byte],
            special: false,
        })
        .collect();
    tokens.push(VocabularyToken {
        bytes: b"[EOS]".to_vec(),
        special: true,
    });
    let vocabulary = Arc::new(TokenVocabulary::new(tokens.len(), tokens).unwrap());
    let compiled = CompiledGrammar::new(&grammar, vocabulary);
    assert_eq!(
        compiled.err().map(|error| error.to_string()).as_deref(),
        Some("the constraint is too large")
    );
}

/// The fixture format, reading `clear_thinking` as a template that names
/// it does, through the toolkit.
struct ClearThinkingFormatV1(FixtureFormatV1);

impl ChatFormatV1 for ClearThinkingFormatV1 {
    fn open(descriptor: &FormatDescriptorV1) -> Result<Self, String> {
        FixtureFormatV1::open(descriptor).map(Self)
    }

    fn opened(&self) -> OpenedV1 {
        self.0.opened()
    }

    fn render(
        &self,
        request: FormatRenderRequestV1<'_>,
    ) -> Result<FormatRenderedV1, FrontendErrorV1> {
        preserve_thinking_v1(true, &request)?;
        self.0.render(request)
    }
}

/// A typed request may carry an opaque keyword set to null. The keyword
/// object keeps that null across the documents, so a format that reads
/// `clear_thinking` refuses it as before, and one that reads
/// `add_vision_id` treats it as before.
#[test]
fn opaque_null_keywords_keep_their_meaning() {
    let frontend =
        PairedFrontendV1::chat_format::<ClearThinkingFormatV1>(function_parameters()).unwrap();
    let mut cleared = request(vec![message(FrontendRoleV1::User, "hello")]);
    cleared.clear_thinking = Some(FrontendJsonV1::Null);
    assert_eq!(
        frontend
            .render_chat(cleared)
            .map(|rendered| rendered.prompt),
        Err(runtime_server::FrontendErrorV1::new(
            "chat_template_kwargs.clear_thinking must be a boolean",
            Some("chat_template_kwargs")
        ))
    );
    let mut vision = request(vec![message(FrontendRoleV1::User, "hello")]);
    vision.add_vision_id = Some(FrontendJsonV1::Null);
    assert!(frontend.render_chat(vision).is_ok());
}

/// A component at the 256 KiB file cap that omits defaulted fields is valid,
/// and grows past the cap when its declaration is written back as JSON. The
/// frontend opens over the package's bytes, and over a declaration written
/// back the export applies no file cap, so both accept it, as before.
#[test]
fn a_component_at_the_file_cap_opens_either_way() {
    use crate::FRONTEND_COMPONENT_MAX_BYTES;
    let mut declaration: serde_json::Value = serde_json::from_str(KEY_VALUE).unwrap();
    declaration["multimodal"] = serde_json::json!({
        "arena": {"rows": 64, "page_rows": 64, "hidden": 8, "element": "bf16"},
        "encoders": [{"name": "image", "payload_element": "bf16", "payload_columns": 4,
            "maximum_payload_bytes": 64, "cost_per_row_milli": 1}],
        "modalities": [{"kind": "image", "part_types": ["image_url"], "maximum_items": 2,
            "maximum_bytes": 64, "maximum_tokens": 8, "encoder": 0,
            "recipe": {"tag": "image.patch-grid.v1", "parameters": {"padding": ""}},
            "render": {"pad": "[A]"}}]
    });
    let short = serde_json::to_vec(&declaration).unwrap().len();
    declaration["multimodal"]["modalities"][0]["recipe"]["parameters"]["padding"] =
        serde_json::Value::String("x".repeat(FRONTEND_COMPONENT_MAX_BYTES - short));
    let bytes = serde_json::to_vec(&declaration).unwrap();
    assert_eq!(bytes.len(), FRONTEND_COMPONENT_MAX_BYTES);
    let descriptor = FrontendDescriptorV1::from_component_bytes(&bytes).unwrap();
    assert!(serde_json::to_vec(&descriptor).unwrap().len() > FRONTEND_COMPONENT_MAX_BYTES);
    let export = crate::test_support::fixture_export_v1();
    assert!(crate::BundleFrontendV1::from_export(&export, &bytes).is_ok());
    assert!(crate::BundleParserSourceV1::from_export(&export, &bytes).is_ok());
    assert!(crate::BundleParserSourceV1::from_declaration(&descriptor).is_ok());
    let frontend = paired(descriptor);
    assert!(
        frontend
            .render_chat(request(vec![message(FrontendRoleV1::User, "hi")]))
            .is_ok()
    );
}
