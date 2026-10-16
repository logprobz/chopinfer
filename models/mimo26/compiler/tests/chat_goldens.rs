use frontend_format::test_support::chat::Golden;
use frontend_format::{
    ChatFormatV1, FormatDescriptorV1, FormatRenderRequestV1, FrontendJsonV1, FrontendMessageV1,
    FrontendRoleV1, FrontendToolV1, PromptModeV1,
};
use model_mimo26_contract::frontend::Mimo26Format;
use runtime_frontend::test_support::documents::golden_request;
use runtime_server::ServerFrontendV1;
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

#[path = "support/frontend.rs"]
mod frontend_support;
use frontend_support::frontend;
use model_mimo26::frontend::descriptor;
use model_mimo26::tokenizer::{load_tokenizer, special_tokens};

/// The pinned declaration with its format read.
fn declared_format() -> FormatDescriptorV1 {
    FormatDescriptorV1::new(descriptor()).unwrap()
}

/// MiMo's chat format over the pinned declaration, called directly.
fn format() -> Mimo26Format {
    Mimo26Format::open(&declared_format()).unwrap()
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

/// A request for `messages` with no tools, the generation prompt and no
/// reasoning control.
fn request(messages: Vec<FrontendMessageV1>) -> FormatRenderRequestV1<'static> {
    FormatRenderRequestV1 {
        messages,
        continue_final_message: false,
        add_generation_prompt: true,
        tools: &[],
        tool_objects: Vec::new(),
        response_format: None,
        reasoning_effort: None,
        enable_thinking: None,
        thinking: None,
        nested_reasoning_effort: None,
        preserve_thinking: None,
        clear_thinking: None,
        add_vision_id: None,
    }
}

fn render(request: FormatRenderRequestV1<'_>) -> String {
    format().render(request).unwrap().prompt
}

/// Every capture, rendered by the chat format and through its export, which
/// must agree.
fn rendered_captures() -> Vec<(Golden, String)> {
    let frontend = frontend();
    let mut goldens: Vec<Golden> = serde_json::from_str(include_str!(
        "../../../../fixtures/mimo26/frontend/chat-renders.json"
    ))
    .expect("parse chat captures");
    assert_eq!(goldens.len(), 61);
    // Float text and IDs were captured with HF 5.16.1 and the checkpoint's own Jinja template.
    goldens.extend(
        serde_json::from_str::<Vec<Golden>>(include_str!("../fixtures/chat-integers.json"))
            .unwrap(),
    );
    goldens.extend(
        serde_json::from_str::<Vec<Golden>>(include_str!("../fixtures/chat-floats.json")).unwrap(),
    );
    goldens
        .into_iter()
        .map(|golden| {
            let arguments: fn(&serde_json::Value) -> String = |args| {
                args.as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| serde_json::to_string(args).unwrap())
            };
            let mut request = golden_request(&frontend, &golden, "reasoning", arguments);
            // A capture's tool result may name no call, which the template
            // does not read; the transport requires the field.
            for message in &mut request.messages {
                if message.role == runtime_server::FrontendRoleV1::Tool {
                    message.tool_call_id.get_or_insert_with(String::new);
                }
            }
            let text = frontend
                .render_chat(request)
                .unwrap_or_else(|error| panic!("render {}: {error}", golden.id))
                .prompt;
            (golden, text)
        })
        .collect()
}

#[test]
fn captured_chat_prompts_match_text() {
    for (golden, text) in rendered_captures() {
        // Captures contain tokenizer-decoded text, including its pinned NFC normalization.
        assert_eq!(
            text.nfc().collect::<String>(),
            golden.rendered_text,
            "render {}",
            golden.id
        );
    }
}

#[test]
fn captured_chat_prompt_tokens_match_checkpoint() {
    let Ok(checkpoint_dir) = std::env::var("CHOPINFER_MIMO_TOKENIZER_DIR")
        .or_else(|_| std::env::var("CHOPINFER_MIMO_CHECKPOINT"))
    else {
        eprintln!(
            "CHOPINFER_MIMO_TOKENIZER_DIR and CHOPINFER_MIMO_CHECKPOINT are unset, skipping checkpoint goldens"
        );
        return;
    };
    let checkpoint = Path::new(&checkpoint_dir);
    if !checkpoint.join("tokenizer.json").exists() {
        eprintln!("checkpoint tokenizer is absent, skipping chat goldens");
        return;
    }
    let tokenizer = load_tokenizer(checkpoint).expect("load pinned checkpoint tokenizer");
    for (golden, text) in rendered_captures() {
        let ids = tokenizer
            .encode(
                &text,
                &special_tokens()
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
            )
            .expect("tokenize rendered chat");
        assert_eq!(ids, golden.token_ids, "tokenize {}", golden.id);
        assert_eq!(
            tokenizer.decode_utf8(&ids).expect("decode rendered chat"),
            golden.rendered_text,
            "render {}",
            golden.id
        );
    }
}

#[test]
fn final_assistant_continuation_extends_the_generation_prefix() {
    let user = message(FrontendRoleV1::User, "Describe the image.");
    let mut base = request(vec![user.clone()]);
    base.enable_thinking = Some(false);
    let mut forced = request(vec![user, message(FrontendRoleV1::Assistant, "blue")]);
    forced.enable_thinking = Some(false);
    forced.continue_final_message = true;
    forced.add_generation_prompt = false;
    assert_eq!(render(forced), format!("{}blue", render(base)));
    // A final message without content has nothing to continue.
    let mut empty = message(FrontendRoleV1::Assistant, "");
    empty.content = None;
    let mut nothing = request(vec![
        message(FrontendRoleV1::User, "Describe the image."),
        empty,
    ]);
    nothing.continue_final_message = true;
    nothing.add_generation_prompt = false;
    let error = format().render(nothing).unwrap_err();
    assert_eq!(
        error.message,
        "invalid conversation field: continue_final_message requires a final assistant message \
         with content"
    );
    assert_eq!(error.parameter.as_deref(), Some("messages"));
}

#[test]
fn tool_defaults_and_message_tool_blocks_follow_the_template() {
    let tool = FrontendToolV1 {
        name: "lookup".to_owned(),
        description: None,
        parameters: FrontendJsonV1::Object(Vec::new()),
        parameters_present: false,
        strict: None,
    };
    let tools = [tool];
    let mut with_tools = request(vec![message(FrontendRoleV1::User, "find it")]);
    with_tools.tools = &tools;
    // A tool without a description or parameters writes neither, as the
    // template writes the tool the transport hands it.
    assert!(render(with_tools).starts_with(concat!(
        "<|im_start|>system\nYou are provided with the following tools:\n\n<tools>\n",
        "{\"type\": \"function\", \"function\": {\"name\": \"lookup\"}}",
        "\n</tools><|im_end|>"
    )));

    let own_tool: FrontendJsonV1 =
        serde_json::from_str(r#"{"type":"function","function":{"name":"lookup"}}"#).unwrap();
    let mut system = message(FrontendRoleV1::System, "Rules");
    system.tools = vec![own_tool];
    let mut chat = request(vec![system]);
    chat.enable_thinking = Some(false);
    assert_eq!(
        render(chat),
        concat!(
            "<|im_start|>system\nRules\n\nYou are provided with the following tools:\n\n<tools>\n",
            "{\"type\": \"function\", \"function\": {\"name\": \"lookup\"}}",
            "\n</tools><|im_end|><|im_start|>assistant\n<think></think>"
        )
    );
}

#[test]
fn generation_prefix_is_optional_and_text_format_does_not_change_the_prompt() {
    let format: FrontendJsonV1 = serde_json::from_str(r#"{"type":"text"}"#).unwrap();
    for thinking in [false, true] {
        let ask = |add_generation_prompt, response_format: Option<&FrontendJsonV1>| {
            let mut ask = request(vec![message(FrontendRoleV1::User, "hello")]);
            ask.enable_thinking = Some(thinking);
            ask.add_generation_prompt = add_generation_prompt;
            ask.response_format = response_format.cloned();
            render(ask)
        };
        let closed = ask(false, Some(&format));
        assert_eq!(closed, "<|im_start|>user\nhello<|im_end|>");
        let open = ask(true, Some(&format));
        assert!(open.starts_with(&closed));
        assert!(open.len() > closed.len());
        assert_eq!(ask(true, None), open);
    }
}

#[test]
fn earlier_reasoning_always_renders_and_the_parser_keeps_blank_content() {
    let declaration = declared_format();
    let parser = &declaration.parser;
    assert!(!parser.drop_blank_tool_call_content);
    let mut answer = message(FrontendRoleV1::Assistant, "2");
    answer.reasoning_content = Some("one plus one".into());
    let messages = vec![
        message(FrontendRoleV1::User, "What is 1 + 1?"),
        answer,
        message(FrontendRoleV1::User, "And 2 + 2?"),
    ];
    let ask = |preserve_thinking| {
        let mut ask = request(messages.clone());
        ask.preserve_thinking = preserve_thinking;
        render(ask)
    };
    // MiMo's template does not read preserve_thinking, so the keyword is inert.
    assert!(ask(Some(true)).contains("<think>one plus one</think>2"));
    assert_eq!(ask(Some(false)), ask(Some(true)));
    assert_eq!(ask(None), ask(Some(true)));

    let parsed = frontend_format::parse_completion_v1(
        &declaration,
        "plan</think>\n\n<tool_call><function=lookup></function></tool_call><|im_end|>",
        PromptModeV1::Thinking,
    )
    .unwrap();
    assert_eq!(parsed.content, "\n\n");
    assert_eq!(parsed.tool_calls.len(), 1);
}

#[test]
fn a_user_message_that_carries_tools_keeps_its_media() {
    let tool: FrontendJsonV1 =
        serde_json::from_str(r#"{"type":"function","function":{"name":"lookup"}}"#).unwrap();
    let mut user = message(FrontendRoleV1::User, "Which tool fits?");
    user.parts = Some(vec![
        frontend_format::FrontendContentPartV1::Media(frontend_format::FrontendMediaPartV1 {
            kind: frontend_format::FrontendMediaPartKindV1::Image,
            part_type: "image_url".into(),
            source: "0".into(),
            format: None,
        }),
        frontend_format::FrontendContentPartV1::Text("Which tool fits?".into()),
    ]);
    user.tools = vec![tool];
    let rendered = format().render(request(vec![user])).unwrap();
    assert_eq!(rendered.media.len(), 1);
    assert!(
        rendered.prompt.starts_with(concat!(
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Which tool fits?\n\n",
            "You are provided with the following tools:\n\n<tools>\n",
            "{\"type\": \"function\", \"function\": {\"name\": \"lookup\"}}\n</tools><|im_end|>"
        )),
        "{}",
        rendered.prompt
    );
}
