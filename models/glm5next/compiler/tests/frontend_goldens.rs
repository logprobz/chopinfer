//! GLM 5.3 Flash's chat format against the pinned chat template.
//!
//! `fixtures/checkpoints/glm-5.3-flash/chat-goldens.json` holds prompts that
//! transformers rendered from revision eb9eb208's `chat_template.jinja` (see
//! `goldens.py` beside them); the format the bundle exports
//! (`model_glm5next_contract::frontend::Glm53Format`), called directly and
//! through its in-process export, has to reproduce every one byte for byte. With `CHOPINFER_GLM5NEXT_CHECKPOINT` set, the declared tokenizer
//! also has to reproduce the token ids transformers produced.
use std::path::PathBuf;

use frontend_format::parse_completion_v1;
use model_glm5next::frontend::{COMPONENT, descriptor};
use model_glm5next_contract::frontend::Glm53Format;
use runtime_frontend::test_support::documents::PairedFrontendV1;
use runtime_frontend::{
    BundleFrontendV1, FrontendDescriptorV1, FrontendMediaKindV1, PromptModeV1,
    tokenizer_dialect_v1, tokenizer_from_components_v1,
};
use runtime_server::{
    FrontendChatRequestV1, FrontendContentPartV1, FrontendJsonV1, FrontendMediaPartKindV1,
    FrontendMediaPartV1, FrontendMessageV1, FrontendRoleV1, FrontendToolCallV1,
    FrontendToolChoiceV1, FrontendToolV1, ServerFrontendV1,
};
use runtime_tokenizer::{
    BaseTokenRepresentation, ByteLevelPolicy, ConfigFieldPolicy, ConfigTokenRepresentation,
    IgnoreMergesPolicy, MergeRecordPolicy, MergeResultIdPolicy, NormalizerPolicy, TokenizerDialect,
};
use serde::Deserialize;
use serde_json::{Value, json};

const CHAT_GOLDENS: &[u8] =
    include_bytes!("../fixtures/checkpoints/glm-5.3-flash/chat-goldens.json");
const TOKENIZER_GOLDENS: &[u8] =
    include_bytes!("../fixtures/checkpoints/glm-5.3-flash/tokenizer-golden.json");

/// The bundle's chat format over `descriptor`, called directly and through
/// its in-process export, which must agree on every answer.
fn frontend(descriptor: FrontendDescriptorV1) -> PairedFrontendV1 {
    PairedFrontendV1::chat_format::<Glm53Format>(descriptor).unwrap()
}

#[derive(Deserialize)]
struct ChatGolden {
    id: String,
    messages: Vec<Value>,
    tools: Vec<Value>,
    chat_template_kwargs: Value,
    rendered_text: String,
    token_ids: Vec<u32>,
}

#[derive(Deserialize)]
struct TokenizerGolden {
    text: String,
    token_ids: Vec<u32>,
    decoded: String,
}

/// The two pinned tokenizer files, when the checkpoint is configured.
fn tokenizer_files(descriptor: &FrontendDescriptorV1) -> Option<(Vec<u8>, Vec<u8>)> {
    let Some(root) = std::env::var_os("CHOPINFER_GLM5NEXT_CHECKPOINT") else {
        eprintln!("SKIP: CHOPINFER_GLM5NEXT_CHECKPOINT is unset");
        return None;
    };
    let read = |name: &str| std::fs::read(PathBuf::from(&root).join(name)).unwrap();
    Some((
        read(&descriptor.tokenizer.definition.name),
        read(&descriptor.tokenizer.configuration.name),
    ))
}

fn prompt_tokens(descriptor: &FrontendDescriptorV1) -> Vec<&str> {
    descriptor
        .vocabulary
        .prompt_special_tokens
        .iter()
        .map(String::as_str)
        .collect()
}

fn text_of(value: &Value) -> String {
    value.as_str().unwrap_or_default().to_owned()
}

/// One captured message as the transport hands it to the frontend: tool-call
/// arguments as their JSON text, and content lists as ordered parts. A local
/// image path becomes a data URL, which the frontend renders without reading.
fn message(value: &Value) -> FrontendMessageV1 {
    let role = match value["role"].as_str().unwrap() {
        "system" => FrontendRoleV1::System,
        "user" => FrontendRoleV1::User,
        "assistant" => FrontendRoleV1::Assistant,
        "tool" => FrontendRoleV1::Tool,
        other => panic!("unexpected role {other}"),
    };
    let content = &value["content"];
    let parts = content.as_array().map(|parts| {
        parts
            .iter()
            .map(|part| match part["type"].as_str().unwrap() {
                "text" => FrontendContentPartV1::Text(text_of(&part["text"])),
                "image_url" => FrontendContentPartV1::Media(FrontendMediaPartV1 {
                    kind: FrontendMediaPartKindV1::Image,
                    part_type: "image_url".into(),
                    source: "data:image/png;base64,AA==".into(),
                    format: None,
                }),
                other => panic!("unexpected part {other}"),
            })
            .collect()
    });
    FrontendMessageV1 {
        role,
        content: content.as_str().map(str::to_owned),
        parts,
        reasoning_content: value["reasoning_content"].as_str().map(str::to_owned),
        tool_calls: value["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|call| FrontendToolCallV1 {
                id: text_of(&call["id"]),
                name: text_of(&call["function"]["name"]),
                arguments: call["function"]["arguments"].to_string(),
            })
            .collect(),
        tool_call_id: value["tool_call_id"].as_str().map(str::to_owned),
        tools: Vec::new(),
    }
}

fn tool(value: &Value) -> FrontendToolV1 {
    let function = &value["function"];
    FrontendToolV1 {
        name: text_of(&function["name"]),
        description: function["description"].as_str().map(str::to_owned),
        parameters: serde_json::from_value(function["parameters"].clone()).unwrap(),
        parameters_present: !function["parameters"].is_null(),
        strict: function["strict"].as_bool(),
    }
}

/// One captured chat as the transport hands it to the frontend: the offered
/// tools with their function objects as sent, and the template keywords.
fn chat_request(golden: &ChatGolden) -> FrontendChatRequestV1 {
    let kwargs = &golden.chat_template_kwargs;
    FrontendChatRequestV1 {
        messages: golden.messages.iter().map(message).collect(),
        tools: golden.tools.iter().map(tool).collect(),
        tool_objects: golden
            .tools
            .iter()
            .map(|tool| serde_json::from_value(tool["function"].clone()).unwrap())
            .collect(),
        nested_reasoning_effort: kwargs["reasoning_effort"].as_str().map(str::to_owned),
        clear_thinking: kwargs
            .get("clear_thinking")
            .map(|value| serde_json::from_value(value.clone()).unwrap()),
        // The whole object, as the transport forwards it.
        chat_template_kwargs: runtime_server::forwarded_template_kwargs_v1(
            serde_json::from_value(kwargs.clone()).unwrap(),
        ),
        ..request(Vec::new())
    }
}

#[test]
fn rendered_chats_match_the_pinned_template_byte_for_byte() {
    let text = descriptor();
    let frontend = frontend(descriptor());
    let tokenizer = tokenizer_files(&text).map(|(definition, configuration)| {
        tokenizer_from_components_v1(&text, &definition, &configuration).unwrap()
    });
    let goldens: Vec<ChatGolden> = serde_json::from_slice(CHAT_GOLDENS).unwrap();
    assert_eq!(goldens.len(), 16);
    let mut images = 0;
    for golden in goldens {
        let rendered = frontend
            .render_chat(chat_request(&golden))
            .unwrap_or_else(|error| panic!("{}: {}", golden.id, error.message));
        images += rendered.media.len().min(1);
        assert_eq!(rendered.prompt, golden.rendered_text, "{}", golden.id);
        if let Some(tokenizer) = &tokenizer {
            assert_eq!(
                tokenizer
                    .encode(&rendered.prompt, &prompt_tokens(&text))
                    .unwrap(),
                golden.token_ids,
                "{}",
                golden.id
            );
        }
    }
    assert_eq!(images, 2);
}

#[test]
fn tokenizer_goldens_encode_and_decode() {
    let descriptor = descriptor();
    let Some((definition, configuration)) = tokenizer_files(&descriptor) else {
        return;
    };
    let tokenizer = tokenizer_from_components_v1(&descriptor, &definition, &configuration).unwrap();
    let goldens: Vec<TokenizerGolden> = serde_json::from_slice(TOKENIZER_GOLDENS).unwrap();
    assert_eq!(goldens.len(), 27);
    for golden in goldens {
        let ids = tokenizer
            .encode(&golden.text, &prompt_tokens(&descriptor))
            .unwrap();
        assert_eq!(ids, golden.token_ids, "{:?}", golden.text);
        assert_eq!(
            tokenizer.decode_utf8(&ids).unwrap(),
            golden.decoded,
            "{:?}",
            golden.text
        );
    }
}

#[test]
fn completions_decode_reasoning_and_key_value_calls() {
    let d = frontend_format::FormatDescriptorV1::new(descriptor()).unwrap();
    let parsed = parse_completion_v1(
        &d,
        concat!(
            "Check both.</think>",
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>",
            "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>",
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value></tool_call>",
            "<|observation|>"
        ),
        PromptModeV1::Thinking,
    )
    .unwrap();
    assert_eq!(parsed.reasoning_content.as_deref(), Some("Check both."));
    assert_eq!(parsed.content, "");
    let calls: Vec<_> = parsed
        .tool_calls
        .iter()
        .map(|call| (call.name.as_str(), call.arguments.as_str()))
        .collect();
    assert_eq!(
        calls,
        [
            ("get_weather", r#"{"city": "Paris", "days": "3"}"#),
            ("get_weather", r#"{"city": "北京"}"#),
        ]
    );
    for stop in ["<|endoftext|>", "<|user|>"] {
        let parsed =
            parse_completion_v1(&d, &format!("r</think>Done.{stop}"), PromptModeV1::Thinking)
                .unwrap();
        assert_eq!(parsed.content, "Done.");
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

fn plain(role: FrontendRoleV1, text: &str) -> FrontendMessageV1 {
    FrontendMessageV1 {
        role,
        content: Some(text.into()),
        parts: None,
        reasoning_content: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
        tools: Vec::new(),
    }
}

/// What the template renders that the declaration refuses rather than
/// renders differently: efforts it maps to Max, thinking switched off, which
/// the template ignores, and developer messages, which it drops.
#[test]
fn controls_the_template_ignores_are_refused() {
    let frontend = frontend(descriptor());
    let hello = || vec![plain(FrontendRoleV1::User, "Hello")];
    let rendered = frontend.render_chat(request(hello())).unwrap();
    assert_eq!(
        rendered.prompt,
        "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hello<|assistant|><think>"
    );
    for effort in ["medium", "xhigh", "minimal", "none"] {
        let mut asked = request(hello());
        asked.reasoning_effort = Some(effort.into());
        assert!(frontend.render_chat(asked).is_err(), "{effort}");
    }
    for (enable, thinking) in [(Some(false), None), (None, Some(false))] {
        let mut asked = request(hello());
        asked.enable_thinking = enable;
        asked.thinking = thinking;
        assert!(frontend.render_chat(asked).is_err());
    }
    let developer = request(vec![
        plain(FrontendRoleV1::Developer, "Be brief."),
        plain(FrontendRoleV1::User, "Hello"),
    ]);
    assert!(frontend.render_chat(developer).is_err());
}

/// History reasoning arrives as the template reads it: `reasoning_content` on
/// every assistant turn when `reasoning` is absent (the transport's fallback),
/// inline reasoning in an answer, and `clear_thinking`.
#[test]
fn history_reasoning_controls_reach_the_template() {
    let frontend = frontend(descriptor());
    assert!(frontend.reasoning_content_fallback());
    let assistant = |content: &str, reasoning: Option<&str>| FrontendMessageV1 {
        reasoning_content: reasoning.map(str::to_owned),
        ..plain(FrontendRoleV1::Assistant, content)
    };
    let history = || {
        vec![
            plain(FrontendRoleV1::User, "q1"),
            assistant("a1", Some("r1")),
            assistant("<think>inline</think> a2 ", None),
            plain(FrontendRoleV1::User, "q2"),
        ]
    };
    let prefix = "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>q1";
    let rendered = frontend.render_chat(request(history())).unwrap();
    assert_eq!(
        rendered.prompt,
        format!(
            "{prefix}<|assistant|><think>r1</think>a1<|assistant|><think>inline</think>a2\
             <|user|>q2<|assistant|><think>"
        )
    );
    let mut cleared = request(history());
    cleared.clear_thinking = Some(FrontendJsonV1::Bool(true));
    assert_eq!(
        frontend.render_chat(cleared).unwrap().prompt,
        format!(
            "{prefix}<|assistant|><think></think>a1<|assistant|><think></think>a2\
             <|user|>q2<|assistant|><think>"
        )
    );
    // The template reads the keyword as a boolean, so anything else is refused.
    let mut odd = request(history());
    odd.clear_thinking = Some(FrontendJsonV1::Number(1.into()));
    let error = frontend.render_chat(odd).unwrap_err();
    assert_eq!(
        error.message,
        "chat_template_kwargs.clear_thinking must be a boolean"
    );
}

/// A tool result of `tool_reference` parts renders each referenced tool,
/// a deferred one too, as the template's tool list writes it (lines 18-31).
/// References are refused mixed with other content, which the template
/// drops, and without offered tools, where the template has no
/// `tool_references_to_response`.
#[test]
fn tool_reference_results_render_the_referenced_tools() {
    let frontend = frontend(descriptor());
    assert!(frontend.tool_references());
    let lookup =
        r#"{"name": "lookup", "description": "Look up.", "parameters": {"type": "object"}}"#;
    let chat = |parts: Vec<FrontendContentPartV1>, offered: bool| {
        let call = FrontendMessageV1 {
            tool_calls: vec![FrontendToolCallV1 {
                id: "call_s".into(),
                name: "tool_search".into(),
                arguments: r#"{"query": "lookup"}"#.into(),
            }],
            ..plain(FrontendRoleV1::Assistant, "")
        };
        let result = FrontendMessageV1 {
            parts: Some(parts),
            tool_call_id: Some("call_s".into()),
            ..plain(FrontendRoleV1::Tool, "")
        };
        let mut asked = request(vec![plain(FrontendRoleV1::User, "Find it."), call, result]);
        if offered {
            let mut deferred: serde_json::Map<String, Value> =
                serde_json::from_str(lookup).unwrap();
            deferred.insert("defer_loading".into(), Value::Bool(true));
            asked.tools = vec![FrontendToolV1 {
                name: "lookup".into(),
                description: Some("Look up.".into()),
                parameters: serde_json::from_str(r#"{"type": "object"}"#).unwrap(),
                parameters_present: true,
                strict: None,
            }];
            asked.tool_objects = vec![serde_json::from_value(Value::Object(deferred)).unwrap()];
        }
        asked
    };
    let reference = |name: &str| FrontendContentPartV1::ToolReference { name: name.into() };
    let rendered = frontend
        .render_chat(chat(vec![reference("lookup"), reference("missing")], true))
        .unwrap();
    assert!(
        rendered.prompt.ends_with(&format!(
            "<|observation|><tool_response><tools>\n{lookup}\n</tools></tool_response>\
             <|assistant|><think>"
        )),
        "{}",
        rendered.prompt
    );
    for (parts, offered, message) in [
        (
            vec![
                reference("lookup"),
                FrontendContentPartV1::Text(" loaded".into()),
            ],
            true,
            "invalid conversation field: a tool message's tool_reference parts cannot share it \
             with other content",
        ),
        (
            vec![reference("lookup")],
            false,
            "invalid conversation field: tool_reference parts require offered tools",
        ),
    ] {
        let error = frontend.render_chat(chat(parts, offered)).unwrap_err();
        assert_eq!(error.message, message);
    }
    assert!(
        frontend
            .render_chat(chat(vec![reference("<|image|>")], true))
            .is_err()
    );
}

#[test]
fn declaration_pins_the_checkpoint_vocabulary_and_sampling() {
    let d = descriptor();
    assert_eq!(d.reasoning.effort_names(), ["low", "high", "max"]);
    assert_eq!(d.reasoning.default_effort, "max");
    assert_eq!(d.vocabulary.stop_token_ids, [154820, 154827, 154829]);
    assert_eq!(d.tokenizer.vocabulary_size, Some(154856));
    assert_eq!(d.generation_config.temperature, Some(1.0));
    assert_eq!(d.generation_config.top_p, Some(0.95));
    assert_eq!(d.generation_config.top_k, None);
    let frontend = frontend(d);
    assert!(frontend.supports_media());
    assert_eq!(frontend.served_media(), [FrontendMediaPartKindV1::Image]);
}

/// The deploy defaults copy the explicit fields of the checkpoints'
/// `generation_config.json`: each declared sampling field equals the file's
/// (temperature 1.0, top-p 0.95, the others omitted in both) and the stop
/// ids are its EOS ids, for the FP8 checkpoint and both NVFP4 exports
/// alike, when they are configured. NVIDIA's also states `do_sample: true`,
/// which the served API follows anyway. An omitted field keeps Chopinfer's default,
/// which is not HF's effective one (HF fills top-k 50).
#[test]
fn sampling_defaults_are_the_checkpoint_generation_config() {
    let d = descriptor();
    let declared = serde_json::to_value(d.generation_config).unwrap();
    let mut checked = 0;
    for variable in [
        "CHOPINFER_GLM5NEXT_CHECKPOINT",
        "CHOPINFER_GLM5NEXT_NVFP4_CHECKPOINT",
        "CHOPINFER_GLM5NEXT_NVIDIA_NVFP4_CHECKPOINT",
    ] {
        let Some(root) = std::env::var_os(variable) else {
            eprintln!("SKIP: {variable} is unset");
            continue;
        };
        let path = PathBuf::from(root).join("generation_config.json");
        let config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        for field in declared.as_object().unwrap().keys() {
            assert_eq!(declared[field], config[field], "{variable}: {field}");
        }
        assert_eq!(
            config["eos_token_id"],
            serde_json::to_value(&d.vocabulary.stop_token_ids).unwrap(),
            "{variable}"
        );
        checked += 1;
    }
    eprintln!("checked {checked} generation configs");
}

/// Images use the zero-padded token-budget recipe, which the host loads;
/// that recipe has no video form, so a video modality beside it is refused
/// at load, and a video part is refused with a message naming it.
#[test]
fn images_load_their_recipe_and_video_is_refused() {
    let d = descriptor();
    let media = d.multimodal.as_ref().unwrap();
    let [image] = media.modalities.as_slice() else {
        panic!("one image modality");
    };
    assert_eq!(image.kind, FrontendMediaKindV1::Image);
    assert_eq!(image.recipe.tag, "image.zero-padded-token-budget.v1");
    assert!(image.maximum_tokens >= 8000);
    runtime_host::validate_media_recipes_v1(&d).unwrap();
    let mut with_video = d.clone();
    let mut video = image.clone();
    video.kind = FrontendMediaKindV1::Video;
    video.part_types = vec!["video_url".into()];
    with_video
        .multimodal
        .as_mut()
        .unwrap()
        .modalities
        .push(video);
    assert!(
        runtime_host::validate_media_recipes_v1(&with_video)
            .unwrap_err()
            .contains("has no video form")
    );
    let frontend = frontend(d);
    let mut asked = request(vec![FrontendMessageV1 {
        parts: Some(vec![
            FrontendContentPartV1::Text("What happens?".into()),
            FrontendContentPartV1::Media(FrontendMediaPartV1 {
                kind: FrontendMediaPartKindV1::Video,
                part_type: "video_url".into(),
                source: "data:video/mp4;base64,AA==".into(),
                format: None,
            }),
        ]),
        ..plain(FrontendRoleV1::User, "")
    }]);
    asked.messages[0].content = None;
    let error = frontend.render_chat(asked).unwrap_err();
    assert_eq!(
        error.message,
        "media part type is not declared by this bundle"
    );
}

/// The declared policy is the `byte-level-paired-merges-v1` dialect this model
/// was served with before tokenizer policy was declared, member for member.
#[test]
fn declared_tokenizer_policy_is_the_byte_level_paired_merges_dialect() {
    let expected = TokenizerDialect {
        normalizer: NormalizerPolicy::Null,
        pretokenizer_byte_level: ByteLevelPolicy {
            add_prefix_space: false,
            trim_offsets: true,
            use_regex: false,
        },
        post_processor_byte_level: ByteLevelPolicy {
            add_prefix_space: true,
            trim_offsets: false,
            use_regex: true,
        },
        decoder_byte_level: ByteLevelPolicy {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        },
        merge_results: MergeResultIdPolicy::Vocabulary,
        merge_records: MergeRecordPolicy::StringPairArray,
        base_tokens: BaseTokenRepresentation::ByteLevel,
        config_tokens: ConfigTokenRepresentation::String,
        add_bos_token: ConfigFieldPolicy::Absent,
        bos_token: ConfigFieldPolicy::Absent,
        unk_token: ConfigFieldPolicy::Absent,
        tokenizer_class: "TokenizersBackend".to_owned(),
        continuing_subword_prefix: None,
        end_of_word_suffix: None,
        ignore_merges: IgnoreMergesPolicy::WholePiece,
        add_eos_token: None,
        legacy: None,
        empty_sentencepiece_kwargs: false,
    };
    assert_eq!(
        tokenizer_dialect_v1(&descriptor().tokenizer.policy),
        expected
    );
}

/// The chat format's export opens over the compiler's component, and
/// refuses a declaration whose reasoning controls or parser section differ
/// from what the renderer writes its prompt for: it always opens `<think>`,
/// writes the effort the template reads, and its prompt is read back
/// through the parser section.
#[test]
fn the_export_opens_only_the_declaration_it_renders_for() {
    let export = frontend_format::chat_format_export_v1::<Glm53Format>();
    let frontend = BundleFrontendV1::from_export(&export, COMPONENT).unwrap();
    assert!(frontend.reads_tool_objects());
    assert!(frontend.reasoning_content_fallback());
    assert!(frontend.tool_references());
    let shipped: Value = serde_json::from_slice(COMPONENT).unwrap();
    let ours = "GLM 5.3 Flash's chat format: the declaration";
    // The library holds the end markers to the parser section itself.
    let library = "reasoning end_markers must be";
    let changes: [(&str, Value, &str); 17] = [
        ("/reasoning/default_enabled", json!(false), ours),
        ("/reasoning/allow_disabled", json!(true), ours),
        ("/reasoning/default_effort", json!("low"), ours),
        (
            "/reasoning/efforts/0/preamble",
            json!("Reasoning Effort: Minimal"),
            ours,
        ),
        (
            "/reasoning/efforts",
            json!([{"name": "low", "preamble": "Reasoning Effort: Low"},
                   {"name": "max", "preamble": "Reasoning Effort: Max"}]),
            ours,
        ),
        (
            "/reasoning/thinking_keywords",
            json!(["enable_thinking"]),
            ours,
        ),
        ("/reasoning/disable_value", json!("off"), ours),
        ("/reasoning/effort_use", json!("ignored"), ours),
        ("/reasoning/end_markers", json!(["</think>"]), library),
        ("/format/parser/reasoning_open", json!("<reason>"), ours),
        ("/format/parser/tool_call_close", json!("</tool>"), ours),
        (
            "/format/parser/drop_blank_tool_call_content",
            json!(true),
            ours,
        ),
        ("/format/parser/assistant_header", json!("<|user|>"), ours),
        // The renderer writes Python's JSON and no answer prefix, call
        // separator or value framing, whatever the section declares.
        (
            "/format/parser/json_encoding",
            json!("spaced-json-v1"),
            ours,
        ),
        ("/format/parser/answer_prefix", json!("\n"), ours),
        ("/format/parser/tool_call_separator", json!("\n"), ours),
        (
            "/format/parser/parameter_value_open",
            json!("<arg_value>\n"),
            ours,
        ),
    ];
    for (pointer, value, refusal) in changes {
        let mut changed = shipped.clone();
        let (parent, field) = pointer.rsplit_once('/').unwrap();
        changed
            .pointer_mut(parent)
            .and_then(Value::as_object_mut)
            .unwrap()
            .insert(field.to_owned(), value);
        let bytes = serde_json::to_vec(&changed).unwrap();
        let error = BundleFrontendV1::from_export(&export, &bytes)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(error.contains(refusal), "{pointer}: {error}");
    }
}

/// A forced call's enumeration and constant values, which the grammar writes
/// as they are, read back through this model's own parser declaration.
#[test]
fn forced_literal_values_read_back() {
    let descriptor = frontend_format::FormatDescriptorV1::from_component_bytes(COMPONENT).unwrap();
    frontend_format::test_support::forced_literals::assert_forced_literals_read_back(&descriptor);
}

/// A forced call's additional-property names exclude its property names
/// without outgrowing the grammar budgets, even for two or three tools of
/// 100 properties named as long as this model's declaration permits.
#[test]
fn long_property_names_compile() {
    let descriptor = frontend_format::FormatDescriptorV1::from_component_bytes(COMPONENT).unwrap();
    frontend_format::test_support::forced_literals::assert_long_property_names_compile(&descriptor);
}

/// A forced call's grammar crosses the bundle boundary even when its
/// schema's references nest it as deep as a grammar document may compile.
#[test]
fn deep_reference_schemas_cross_the_wire() {
    let format = frontend_format::FormatDescriptorV1::from_component_bytes(COMPONENT).unwrap();
    let call = frontend_format::test_support::forced_literals::forced_call(&format, "[0]");
    runtime_frontend::test_support::tool_choice::deep_reference_schemas_cross_the_wire(
        &frontend(descriptor()),
        &call,
    );
}
