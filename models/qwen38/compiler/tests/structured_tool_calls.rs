//! Forced tool choices in this model's declared call layout, whose markers
//! carry newlines and whose calls are separated: every admitted output parses
//! back through the declared decoder with the declared argument types.

use std::sync::Arc;

use frontend_format::{PromptModeV1, parse_completion_v1};
use model_qwen38::frontend::descriptor;
use runtime_grammar::{CompiledGrammar, ConstraintState, TokenVocabulary, VocabularyToken};
use runtime_server::{FrontendJsonV1, FrontendToolChoiceV1, FrontendToolV1, ServerFrontendV1};
use serde_json::{Value, json};

const STOP_TEXT: &str = "<|im_end|>";
const CALL_OPEN: u32 = 256;

fn tools() -> Vec<FrontendToolV1> {
    let tool = |name: &str, parameters: Value| FrontendToolV1 {
        name: name.to_owned(),
        description: None,
        parameters: serde_json::from_value::<FrontendJsonV1>(parameters).unwrap(),
        parameters_present: true,
        strict: None,
    };
    vec![
        tool(
            "get_weather",
            json!({"type": "object", "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer", "minimum": 1, "maximum": 14},
                "metric": {"type": "boolean"}
            }, "required": ["city", "days"]}),
        ),
        tool(
            "get_time",
            json!({"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}),
        ),
    ]
}

/// Single bytes, the call open marker as a special token without its
/// newline, multi-byte spellings of the other markers, and the stop token.
fn vocabulary() -> (Arc<TokenVocabulary>, u32) {
    let mut tokens: Vec<VocabularyToken> = (0..=255_u8)
        .map(|byte| VocabularyToken {
            bytes: vec![byte],
            special: false,
        })
        .collect();
    for (text, special) in [
        ("<tool_call>", true),
        ("\n</parameter>\n", false),
        ("</function>\n", false),
        ("</tool_call>", false),
        ("<parameter=", false),
        (STOP_TEXT, true),
    ] {
        tokens.push(VocabularyToken {
            bytes: text.as_bytes().to_vec(),
            special,
        });
    }
    let stop = tokens.len() as u32 - 1;
    (
        Arc::new(TokenVocabulary::new(tokens.len(), tokens).unwrap()),
        stop,
    )
}

fn walk(grammar: &Arc<CompiledGrammar>, stop: u32, seed: &mut u64) -> Option<String> {
    let vocabulary = Arc::clone(grammar.vocabulary());
    let mut state = ConstraintState::new(Arc::clone(grammar), None, vec![stop]);
    let mut mask = vec![0_u32; vocabulary.mask_words()];
    let mut text = Vec::new();
    let mut below = |bound: usize| {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed % bound as u64) as usize
    };
    for _ in 0..20_000 {
        assert!(state.fill_mask(&mut mask).unwrap());
        let allowed: Vec<u32> = (0..vocabulary.size() as u32)
            .filter(|&token| mask[token as usize / 32] & (1 << (token % 32)) != 0)
            .collect();
        let long: Vec<u32> = allowed
            .iter()
            .copied()
            .filter(|&token| token >= 256)
            .collect();
        let token = if !long.is_empty() && below(2) == 0 {
            long[below(long.len())]
        } else {
            allowed[below(allowed.len())]
        };
        state.accept(token).unwrap();
        if token == stop {
            return Some(String::from_utf8(text).expect("the grammar emits UTF-8"));
        }
        match vocabulary.ordinary_bytes(token) {
            Some(bytes) => text.extend_from_slice(bytes),
            None => {
                assert_eq!(token, CALL_OPEN, "only the admitted special token");
                text.extend_from_slice(b"<tool_call>");
            }
        }
    }
    None
}

#[test]
fn forced_calls_follow_the_declared_layout_and_parse_back() {
    let frontend = runtime_frontend::test_support::documents::PairedFrontendV1::chat_format::<
        model_qwen38_contract::frontend::Qwen38Format,
    >(descriptor())
    .unwrap();
    let declaration = frontend_format::FormatDescriptorV1::new(descriptor()).unwrap();
    let (vocabulary, stop) = vocabulary();
    let tools = tools();
    let mut seed = 0x9e37_79b9_u64;
    for choice in [
        FrontendToolChoiceV1::Required,
        FrontendToolChoiceV1::Named("get_weather".to_owned()),
    ] {
        let grammar = frontend.tool_constraint(&tools, &choice).unwrap().unwrap();
        let compiled = Arc::new(CompiledGrammar::new(&grammar, Arc::clone(&vocabulary)).unwrap());
        let mut parsed = 0;
        let mut several = false;
        for _ in 0..40 {
            let Some(text) = walk(&compiled, stop, &mut seed) else {
                continue;
            };
            let completion = parse_completion_v1(
                &declaration,
                &format!("{text}{STOP_TEXT}"),
                PromptModeV1::Chat,
            )
            .unwrap_or_else(|error| panic!("{error}: {text:?}"));
            assert!(completion.content.trim().is_empty(), "{text:?}");
            several |= completion.tool_calls.len() > 1;
            for call in &completion.tool_calls {
                let arguments: Value = serde_json::from_str(&call.arguments).unwrap();
                match call.name.as_str() {
                    "get_weather" => {
                        assert!(arguments.get("city").is_some(), "{text:?}");
                        let days: i64 = arguments["days"].as_str().unwrap().parse().unwrap();
                        assert!((1..=14).contains(&days), "{text:?}");
                        if let Some(metric) = arguments.get("metric") {
                            assert!(metric == "true" || metric == "false", "{text:?}");
                        }
                    }
                    "get_time" => assert!(arguments.get("timezone").is_some()),
                    other => panic!("unexpected function {other}"),
                }
                if let FrontendToolChoiceV1::Named(name) = &choice {
                    assert_eq!(&call.name, name);
                }
            }
            parsed += 1;
        }
        assert!(parsed > 10, "{choice:?}: only {parsed} walks finished");
        if choice == FrontendToolChoiceV1::Required {
            assert!(several, "no walk produced separated calls");
        }
    }
}

/// The export refuses a parser section that differs from the template's
/// (`parser_declaration`) in any field, the JSON encoding included: the
/// export parses and constrains completions with it, and the renderer writes
/// the template's markers and Python's JSON whatever it declares. The
/// shipped components open (`reasoning_controls.rs`).
#[test]
fn the_export_opens_only_over_the_templates_parser_section() {
    use model_qwen38_contract::frontend::{Qwen38Format, parser_declaration};
    use runtime_bundle_abi::frontend::FrontendCallErrorV1;
    use runtime_bundle_abi::frontend_document::RefusalV1;
    use runtime_frontend_contract::FrontendDescriptorV1;

    let export = frontend_format::chat_format_export_v1::<Qwen38Format>;
    for component in [
        serde_json::to_vec(&descriptor()).unwrap(),
        model_qwen38::frontend::multimodal_component(),
    ] {
        let declared = frontend_format::FormatDescriptorV1::from_component_bytes(&component)
            .unwrap()
            .parser;
        assert_eq!(declared, parser_declaration());
    }
    let edits = [
        ("json_encoding", Value::from("spaced-json-v1")),
        ("answer_prefix", Value::from("")),
        ("tool_call_separator", Value::from("")),
        ("parameter_close", Value::from("</parameter>")),
        ("drop_blank_tool_call_content", Value::from(false)),
    ];
    for (field, value) in edits {
        let mut declared = descriptor();
        declared.format["parser"][field] = value;
        // The host reads the changed declaration; the format refuses it.
        let component = serde_json::to_vec(&declared).unwrap();
        FrontendDescriptorV1::from_component_bytes(&component).unwrap();
        let error = export().open(&component).map(|_| ()).unwrap_err();
        let FrontendCallErrorV1::Refused(refusal) = error else {
            panic!("{field}: not a refusal: {error}");
        };
        let message = RefusalV1::decode(&refusal).unwrap().message;
        assert!(
            message.contains("its own format.parser section"),
            "{field}: {message}"
        );
    }
}
