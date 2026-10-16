//! Forced tool choices in this model's own call format: every output the
//! grammar admits parses back through the declared decoder, names an offered
//! function, and carries arguments of the declared types.

use std::path::Path;
use std::sync::Arc;

use model_mimo26::frontend::descriptor;
#[path = "support/frontend.rs"]
mod frontend_support;
use frontend_format::{PromptModeV1, parse_completion_v1};
use frontend_support::frontend;
use model_mimo26::tokenizer::load_tokenizer;
use runtime_grammar::{
    CompiledGrammar, ConstraintState, GrammarMatcher, TokenVocabulary, VocabularyToken,
};
use runtime_server::{FrontendJsonV1, FrontendToolChoiceV1, FrontendToolV1, ServerFrontendV1};
use serde_json::{Value, json};

const STOP_TEXT: &str = "<|im_end|>";

fn tool(name: &str, parameters: Value, strict: Option<bool>) -> FrontendToolV1 {
    let parameters: FrontendJsonV1 = serde_json::from_value(parameters).unwrap();
    FrontendToolV1 {
        name: name.to_owned(),
        description: None,
        parameters,
        parameters_present: true,
        strict,
    }
}

fn tools() -> Vec<FrontendToolV1> {
    vec![
        tool(
            "get_weather",
            json!({"type": "object", "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                "days": {"type": "integer", "minimum": 1, "maximum": 14}
            }, "required": ["city"]}),
            Some(true),
        ),
        tool(
            "get_time",
            json!({"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}),
            None,
        ),
        tool("anything", json!({"type": "object"}), Some(false)),
    ]
}

/// Single bytes, the call markers as special tokens, a few marker spellings
/// as ordinary tokens, and the stop token last.
fn byte_vocabulary() -> (Arc<TokenVocabulary>, u32) {
    let mut tokens: Vec<VocabularyToken> = (0..=255_u8)
        .map(|byte| VocabularyToken {
            bytes: vec![byte],
            special: false,
        })
        .collect();
    for (text, special) in [
        ("<tool_call>", true),
        ("</tool_call>", true),
        ("</parameter>", false),
        ("</function>", false),
        ("<parameter=", false),
        ("celsius", false),
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

struct Rng(u64);

impl Rng {
    fn below(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % bound as u64) as usize
    }
}

fn walk(grammar: &Arc<CompiledGrammar>, stop: u32, rng: &mut Rng) -> Option<String> {
    let vocabulary = Arc::clone(grammar.vocabulary());
    let mut state = ConstraintState::new(Arc::clone(grammar), None, vec![stop]);
    let mut mask = vec![0_u32; vocabulary.mask_words()];
    let mut text = Vec::new();
    for _ in 0..20_000 {
        assert!(state.fill_mask(&mut mask).unwrap());
        let allowed: Vec<u32> = (0..vocabulary.size() as u32)
            .filter(|&token| mask[token as usize / 32] & (1 << (token % 32)) != 0)
            .collect();
        // Multi-byte tokens are preferred half the time so markers close.
        let long: Vec<u32> = allowed
            .iter()
            .copied()
            .filter(|&token| token >= 256)
            .collect();
        let token = if !long.is_empty() && rng.below(2) == 0 {
            long[rng.below(long.len())]
        } else {
            allowed[rng.below(allowed.len())]
        };
        state.accept(token).unwrap();
        if token == stop {
            return Some(String::from_utf8(text).expect("the grammar emits UTF-8"));
        }
        let bytes = vocabulary.ordinary_bytes(token).map_or_else(
            || match token {
                256 => b"<tool_call>".to_vec(),
                257 => b"</tool_call>".to_vec(),
                _ => panic!("unexpected special token {token}"),
            },
            <[u8]>::to_vec,
        );
        text.extend(bytes);
    }
    None
}

fn check_call(name: &str, arguments: &str) {
    let arguments: Value = serde_json::from_str(arguments).unwrap();
    let arguments = arguments.as_object().unwrap();
    match name {
        "get_weather" => {
            assert!(arguments.contains_key("city"));
            for key in arguments.keys() {
                assert!(["city", "unit", "days"].contains(&key.as_str()), "{key}");
            }
            if let Some(unit) = arguments.get("unit") {
                assert!(unit == "celsius" || unit == "fahrenheit", "{unit}");
            }
            if let Some(days) = arguments.get("days") {
                // Arguments are parsed without schemas here, so typed values
                // arrive as their text.
                let days: i64 = days.as_str().unwrap().parse().unwrap();
                assert!((1..=14).contains(&days), "{days}");
            }
        }
        "get_time" => {
            assert_eq!(arguments.keys().collect::<Vec<_>>(), ["timezone"]);
        }
        "anything" => {}
        other => panic!("unexpected function {other}"),
    }
}

#[test]
fn forced_tool_calls_parse_back_with_their_declared_arguments() {
    let frontend = frontend();
    let declaration = frontend_format::FormatDescriptorV1::new(descriptor()).unwrap();
    let (vocabulary, stop) = byte_vocabulary();
    let tools = tools();
    let mut rng = Rng(0x5eed);
    for choice in [
        FrontendToolChoiceV1::Required,
        FrontendToolChoiceV1::Named("get_weather".to_owned()),
        FrontendToolChoiceV1::Named("anything".to_owned()),
    ] {
        let grammar = frontend.tool_constraint(&tools, &choice).unwrap().unwrap();
        let compiled = Arc::new(CompiledGrammar::new(&grammar, Arc::clone(&vocabulary)).unwrap());
        let mut parsed = 0;
        for _ in 0..40 {
            let Some(text) = walk(&compiled, stop, &mut rng) else {
                continue;
            };
            let completion = parse_completion_v1(
                &declaration,
                &format!("{text}{STOP_TEXT}"),
                PromptModeV1::Chat,
            )
            .unwrap_or_else(|error| panic!("{error}: {text:?}"));
            assert!(completion.content.is_empty(), "{text:?}");
            assert!(!completion.tool_calls.is_empty());
            if let FrontendToolChoiceV1::Named(name) = &choice {
                assert_eq!(completion.tool_calls.len(), 1);
                assert_eq!(&completion.tool_calls[0].name, name);
            }
            for call in &completion.tool_calls {
                check_call(&call.name, &call.arguments);
            }
            parsed += 1;
        }
        assert!(parsed > 10, "{choice:?}: only {parsed} walks finished");
    }
    // Auto with a strict tool admits text first and calls after it.
    let grammar = frontend
        .tool_constraint(&tools, &FrontendToolChoiceV1::Auto)
        .unwrap()
        .unwrap();
    let compiled = Arc::new(CompiledGrammar::new(&grammar, vocabulary).unwrap());
    let mut matcher = GrammarMatcher::new(compiled);
    matcher
        .accept_bytes(b"Let me check.<tool_call><function=get_time><parameter=timezone>UTC</parameter></function></tool_call>")
        .unwrap();
    assert!(matcher.is_complete());
    // Without a strict tool, auto imposes nothing.
    let loose: Vec<FrontendToolV1> = tools
        .into_iter()
        .filter(|tool| tool.strict != Some(true))
        .collect();
    assert!(
        frontend
            .tool_constraint(&loose, &FrontendToolChoiceV1::Auto)
            .unwrap()
            .is_none()
    );
}

#[test]
fn tool_masks_admit_the_special_call_markers_of_the_real_vocabulary() {
    let Ok(directory) = std::env::var("CHOPINFER_MIMO_TOKENIZER_DIR")
        .or_else(|_| std::env::var("CHOPINFER_MIMO_CHECKPOINT"))
    else {
        eprintln!("checkpoint tokenizer is absent, skipping");
        return;
    };
    let directory = Path::new(&directory);
    if !directory.join("tokenizer.json").exists() {
        eprintln!("checkpoint tokenizer is absent, skipping");
        return;
    }
    let tokenizer = load_tokenizer(directory).unwrap();
    let declaration = descriptor();
    let mut tokens = Vec::new();
    while let Ok(bytes) = tokenizer.token_bytes(tokens.len() as u32) {
        tokens.push(VocabularyToken {
            bytes: bytes.to_vec(),
            special: tokenizer.token_is_special(tokens.len() as u32).unwrap(),
        });
    }
    let width = declaration.vocabulary.size as usize;
    let vocabulary = Arc::new(TokenVocabulary::new(width, tokens).unwrap());
    let grammar = frontend()
        .tool_constraint(&tools(), &FrontendToolChoiceV1::Required)
        .unwrap()
        .unwrap();
    let compiled = Arc::new(CompiledGrammar::new(&grammar, vocabulary).unwrap());
    let call_open = declaration
        .vocabulary
        .boundary_tokens
        .iter()
        .find(|token| token.text == "<tool_call>")
        .unwrap()
        .id;
    let mut matcher = GrammarMatcher::new(compiled);
    let mut mask = vec![0_u32; width.div_ceil(32)];
    matcher.fill_token_bitmask(&mut mask).unwrap();
    assert!(mask[call_open as usize / 32] & (1 << (call_open % 32)) != 0);
    matcher.accept_token(call_open).unwrap();
    for piece in [
        "<function=get_time>",
        "<parameter=timezone>",
        "UTC",
        "</parameter>",
    ] {
        matcher.fill_token_bitmask(&mut mask).unwrap();
        let allowed: Vec<u32> = (0..width as u32)
            .filter(|&token| mask[token as usize / 32] & (1 << (token % 32)) != 0)
            .collect();
        let expected: Vec<u32> = (0..width as u32)
            .filter(|&token| matcher.accepts_token(token).unwrap())
            .collect();
        assert_eq!(allowed, expected, "before {piece}");
        matcher.accept_bytes(piece.as_bytes()).unwrap();
    }
}

/// The export opens over the component of every registered checkpoint and
/// media set a package may serve, whose parser section is the template's
/// (`parser_declaration`), and refuses a parser section that differs in any
/// field, the JSON encoding included: the export parses and constrains
/// completions with it, and the renderer writes the template's markers and
/// Python's JSON whatever it declares.
#[test]
fn the_export_opens_only_over_the_templates_parser_section() {
    use model_mimo26::frontend::component_for_flash;
    use model_mimo26_contract::encoders::Encoders;
    use model_mimo26_contract::frontend::{Mimo26Format, parser_declaration};
    use runtime_bundle_abi::frontend::FrontendCallErrorV1;
    use runtime_bundle_abi::frontend_document::RefusalV1;
    use runtime_frontend_contract::FrontendDescriptorV1;

    let export = frontend_format::chat_format_export_v1::<Mimo26Format>;
    for (name, identity) in model_mimo26::checkpoint::identities() {
        for encoders in Encoders::SETS {
            let component = component_for_flash(identity, encoders).unwrap();
            let declared = frontend_format::FormatDescriptorV1::from_component_bytes(&component)
                .unwrap()
                .parser;
            assert_eq!(declared, parser_declaration(), "{name} {encoders:?}");
            export()
                .open(&component)
                .map(|_| ())
                .unwrap_or_else(|error| panic!("{name} {encoders:?} opens: {error}"));
        }
    }
    let edits = [
        ("json_encoding", Value::from("spaced-json-v1")),
        ("answer_prefix", Value::from("\n")),
        ("tool_call_separator", Value::from("\n")),
        ("parameter_close", Value::from("\n</parameter>")),
        ("drop_blank_tool_call_content", Value::from(true)),
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
