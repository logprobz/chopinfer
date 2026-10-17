//! DeepSeek V4.1 Flash's chat format against the encoder's own test pairs.
//!
//! `fixtures/deepseek41/frontend/encoder-tests/` holds the checkpoint's
//! `encoding/tests/` pairs. The vendor encodes each input with
//! `encode_case(case, thinking_mode="chat")`; the port
//! (`model_deepseek41_contract::frontend::encoding`) must give every output
//! byte for byte, and the served format, called directly and through its
//! in-process export, must give the outputs of the pairs it serves.
use model_deepseek41::frontend::descriptor;
use model_deepseek41_contract::frontend::DeepSeekV41Format;
use model_deepseek41_contract::frontend::encoding::{
    Block, EOS, IMAGE_PLACEHOLDER, Message, ReasoningEffort, Role, ThinkingMode, ToolCall,
    encode_messages,
};
use runtime_frontend::test_support::documents::PairedFrontendV1;
use runtime_server::{
    FrontendChatRequestV1, FrontendContentPartV1, FrontendJsonV1, FrontendMediaPartKindV1,
    FrontendMediaPartV1, FrontendMessageV1, FrontendRoleV1, FrontendToolCallV1,
    FrontendToolChoiceV1, FrontendToolV1, ServerFrontendV1,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

const PAIRS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../fixtures/deepseek41/frontend/encoder-tests"
);

/// The SHA-256 of each pair's files in the checkpoint's `encoding/tests/`.
const DIGESTS: [(&str, &str); 10] = [
    (
        "test_input_1.json",
        "6a5a86694e47169c8f0c9243a7e24cc5db05447be0d3d3a9891034e913c518de",
    ),
    (
        "test_input_2.json",
        "fd985cda88e8640b8e2404d8e659f7006cad204eb41f73c9e6662b1f218c01aa",
    ),
    (
        "test_input_3.json",
        "4207613f15c79b95ad92074bdb8b2b81c5ce9cfdf15b7a67f58ded6bb3ec4fa5",
    ),
    (
        "test_input_4.json",
        "4c3890a7bc042ec6396d98d024a39a73cdcffbe1d0253e852da0c35e65296f3e",
    ),
    (
        "test_input_5.json",
        "88289b0aac8d650aa55c4ffeb392818fff3a0ac83c2293ddfd90546804d5e5c4",
    ),
    (
        "test_output_1.txt",
        "91f79aa955717b414a82aabfb38bcd83ab5a998bd47db89d2a26338e8d3de550",
    ),
    (
        "test_output_2.txt",
        "9fa162587b73ab9b315be5c88d433f9e0e681b4b36926952f289099ed9bc1dab",
    ),
    (
        "test_output_3.txt",
        "cf2b801d0738c88abb691bf4a096000c0e0737faab8a54a4ba5243c5513a3711",
    ),
    (
        "test_output_4.txt",
        "94433b68370dcc69fa61efc5011b5d0e28685e4716fee1ee77fbbfc0c9935b54",
    ),
    (
        "test_output_5.txt",
        "4d673091746553d1cb2e40a3fc01dc09a7c53c40627096a3e49c97a269a81f64",
    ),
];

fn read(name: &str) -> Vec<u8> {
    let bytes = std::fs::read(format!("{PAIRS}/{name}")).unwrap();
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let pinned = DIGESTS.iter().find(|(file, _)| *file == name).unwrap().1;
    assert_eq!(digest, pinned, "{name} is not the checkpoint's");
    bytes
}

/// One case as `load_cases` and `encode_case` read it (encoding.py lines
/// 847-883): the messages with the case's tools on the first, the mode
/// (chat unless the case names one) and the effort.
struct Case {
    messages: Vec<Value>,
    mode: ThinkingMode,
    effort: Option<ReasoningEffort>,
}

fn case(number: usize) -> Case {
    let data: Value = serde_json::from_slice(&read(&format!("test_input_{number}.json"))).unwrap();
    let case = match data {
        Value::Array(messages) => serde_json::json!({ "messages": messages }),
        case => case,
    };
    let mut messages = case["messages"].as_array().unwrap().clone();
    if let Some(tools) = case.get("tools") {
        messages[0]["tools"] = tools.clone();
    }
    Case {
        messages,
        mode: match case["thinking_mode"].as_str() {
            Some("thinking") => ThinkingMode::Thinking,
            _ => ThinkingMode::Chat,
        },
        effort: match &case["reasoning_effort"] {
            Value::String(name) => Some(ReasoningEffort::Named(name.clone())),
            Value::Number(budget) => Some(ReasoningEffort::Budget(budget.as_i64().unwrap())),
            _ => None,
        },
    }
}

fn json(value: &Value) -> FrontendJsonV1 {
    serde_json::from_value(value.clone()).unwrap()
}

/// A JSON value as the port reads it.
fn encoder_json(value: &Value) -> frontend_format::FrontendJsonV1 {
    serde_json::from_value(value.clone()).unwrap()
}

fn text(value: &Value) -> Option<String> {
    value.as_str().map(str::to_owned)
}

/// One vendor message dictionary as the port's [`Message`]; an image part
/// is the image at its index among the case's images.
fn encoder_message(value: &Value, images: &mut usize) -> Message {
    let role = match value["role"].as_str().unwrap() {
        "system" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        "latest_reminder" => Role::LatestReminder,
        other => panic!("unexpected role {other}"),
    };
    let mut message = Message::new(role, text(&value["content"]));
    if let Some(parts) = value["content"].as_array() {
        message.blocks = Some(
            parts
                .iter()
                .map(|part| match part["type"].as_str().unwrap() {
                    "text" => Block::Text(text(&part["text"]).unwrap_or_default()),
                    "image_url" => {
                        *images += 1;
                        Block::Image(*images - 1)
                    }
                    other => panic!("unexpected part {other}"),
                })
                .collect(),
        );
    }
    message.reasoning_content = text(&value["reasoning_content"]);
    message.tool_call_id = text(&value["tool_call_id"]);
    message.tool_calls = value["tool_calls"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|call| ToolCall {
            id: text(&call["id"]).unwrap_or_default(),
            name: text(&call["function"]["name"]).unwrap(),
            namespace: text(&call["namespace"]),
            arguments: match &call["function"]["arguments"] {
                Value::String(arguments) => arguments.clone(),
                arguments => arguments.to_string(),
            },
        })
        .collect();
    message.tools = value["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .map(encoder_json)
        .collect();
    message.response_format = value
        .get("response_format")
        .filter(|format| !format.is_null())
        .map(encoder_json);
    message.task = text(&value["task"]);
    message.wo_eos = value["wo_eos"].as_bool().unwrap_or(false);
    message
}

#[test]
fn the_port_reproduces_every_vendor_pair_byte_for_byte() {
    for number in 1..=5 {
        let case = case(number);
        let mut images = 0;
        let messages: Vec<Message> = case
            .messages
            .iter()
            .map(|message| encoder_message(message, &mut images))
            .collect();
        let (prompt, order) =
            encode_messages(&messages, case.mode, true, case.effort.as_ref(), &|_| {
                IMAGE_PLACEHOLDER.to_owned()
            })
            .unwrap_or_else(|error| panic!("pair {number}: {error}"));
        let expected = String::from_utf8(read(&format!("test_output_{number}.txt"))).unwrap();
        assert_eq!(prompt, expected, "pair {number}");
        assert_eq!(order, (0..images).collect::<Vec<_>>(), "pair {number}");
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

/// One vendor message as the transport hands it to the frontend.
fn served_message(value: &Value) -> FrontendMessageV1 {
    let role = match value["role"].as_str().unwrap() {
        "system" => FrontendRoleV1::System,
        "user" => FrontendRoleV1::User,
        "assistant" => FrontendRoleV1::Assistant,
        "tool" => FrontendRoleV1::Tool,
        other => panic!("{other} is not served"),
    };
    let parts = value["content"].as_array().map(|parts| {
        parts
            .iter()
            .map(|part| match part["type"].as_str().unwrap() {
                "text" => FrontendContentPartV1::Text(text(&part["text"]).unwrap()),
                _ => FrontendContentPartV1::Media(FrontendMediaPartV1 {
                    kind: FrontendMediaPartKindV1::Image,
                    part_type: "image_url".into(),
                    source: "data:image/png;base64,AA==".into(),
                    format: None,
                }),
            })
            .collect()
    });
    FrontendMessageV1 {
        role,
        content: text(&value["content"]),
        parts,
        reasoning_content: text(&value["reasoning_content"]),
        tool_calls: value["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|call| FrontendToolCallV1 {
                id: text(&call["id"]).unwrap_or_default(),
                name: text(&call["function"]["name"]).unwrap(),
                arguments: text(&call["function"]["arguments"]).unwrap(),
            })
            .collect(),
        tool_call_id: text(&value["tool_call_id"]),
        tools: Vec::new(),
    }
}

/// A pair as a served request: its messages, its tools offered by the
/// request, and the mode as the reasoning controls select it.
fn served_request(number: usize) -> FrontendChatRequestV1 {
    let data: Value = serde_json::from_slice(&read(&format!("test_input_{number}.json"))).unwrap();
    let case = case(number);
    let tools: Vec<Value> = data
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut messages: Vec<FrontendMessageV1> = case.messages.iter().map(served_message).collect();
    messages[0].tools.clear();
    FrontendChatRequestV1 {
        tools: tools
            .iter()
            .map(|tool| {
                let function = &tool["function"];
                FrontendToolV1 {
                    name: text(&function["name"]).unwrap(),
                    description: text(&function["description"]),
                    parameters: json(&function["parameters"]),
                    parameters_present: !function["parameters"].is_null(),
                    strict: function["strict"].as_bool(),
                }
            })
            .collect(),
        tool_objects: tools.iter().map(|tool| json(&tool["function"])).collect(),
        enable_thinking: Some(case.mode == ThinkingMode::Thinking),
        reasoning_effort: match case.effort {
            Some(ReasoningEffort::Named(name)) => Some(name),
            _ => None,
        },
        ..request(messages)
    }
}

#[test]
fn the_served_format_renders_the_pairs_it_serves() {
    let frontend = PairedFrontendV1::chat_format::<DeepSeekV41Format>(descriptor()).unwrap();
    // Pairs 3 and 4 hold the encoder's latest_reminder role (and pair 4 a
    // task), which the transport does not carry. Pairs 1 and 2 end on an
    // assistant turn, which renders closed with nothing to answer after it:
    // served, it is refused, and continued it renders open.
    for number in [1, 2] {
        let error = frontend.render_chat(served_request(number)).unwrap_err();
        assert!(
            error
                .message
                .contains("a final assistant message is served only with continue_final_message"),
            "pair {number}: {}",
            error.message
        );
        let rendered = frontend
            .render_chat(FrontendChatRequestV1 {
                continue_final_message: true,
                add_generation_prompt: false,
                ..served_request(number)
            })
            .unwrap_or_else(|error| panic!("pair {number}: {}", error.message));
        let expected = String::from_utf8(read(&format!("test_output_{number}.txt"))).unwrap();
        let expected = expected.strip_suffix(EOS).unwrap();
        assert_eq!(rendered.prompt, expected, "pair {number}");
    }
    // Pair 5's images need a declaration that serves images.
    let error = frontend.render_chat(served_request(5)).unwrap_err();
    assert!(
        error.message.contains("does not support media inputs"),
        "{}",
        error.message
    );
}

#[test]
fn the_default_request_thinks_at_high_effort() {
    let frontend = PairedFrontendV1::chat_format::<DeepSeekV41Format>(descriptor()).unwrap();
    let user = |text: &str| FrontendMessageV1 {
        role: FrontendRoleV1::User,
        content: Some(text.into()),
        parts: None,
        reasoning_content: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
        tools: Vec::new(),
    };
    let rendered = frontend
        .render_chat(request(vec![user("question")]))
        .unwrap();
    assert_eq!(
        rendered.prompt,
        "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the \
         value, the more thorough the reasoning)\n\n<｜User｜>question<｜Assistant｜><think>"
    );
    // The protocol aliases read as deepseek-recipe maps them, top-level
    // and nested.
    let efforts = [
        ("low", 50),
        ("max", 100),
        ("minimal", 50),
        ("medium", 75),
        ("xhigh", 75),
    ];
    for (effort, budget) in efforts {
        for nested in [false, true] {
            let rendered = frontend
                .render_chat(FrontendChatRequestV1 {
                    reasoning_effort: (!nested).then(|| effort.into()),
                    nested_reasoning_effort: nested.then(|| effort.into()),
                    ..request(vec![user("question")])
                })
                .unwrap();
            assert!(
                rendered
                    .prompt
                    .contains(&format!("Reasoning Effort: {budget} (range")),
                "{effort}"
            );
        }
    }
    let rendered = frontend
        .render_chat(FrontendChatRequestV1 {
            reasoning_effort: Some("none".into()),
            ..request(vec![user("question")])
        })
        .unwrap();
    assert_eq!(
        rendered.prompt,
        "<｜begin▁of▁sentence｜><｜User｜>question<｜Assistant｜></think>"
    );
}

fn message(role: FrontendRoleV1, content: &str) -> FrontendMessageV1 {
    FrontendMessageV1 {
        role,
        content: Some(content.into()),
        parts: None,
        reasoning_content: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
        tools: Vec::new(),
    }
}

/// A request offering `function` (a function object as sent) after a user
/// turn and one call of it with `arguments` as sent.
fn called(function: Value, arguments: &str) -> FrontendChatRequestV1 {
    let mut assistant = message(FrontendRoleV1::Assistant, "");
    assistant.tool_calls.push(FrontendToolCallV1 {
        id: "call_0".into(),
        name: "f".into(),
        arguments: arguments.into(),
    });
    let mut result = message(FrontendRoleV1::Tool, "ok");
    result.tool_call_id = Some("call_0".into());
    FrontendChatRequestV1 {
        tools: vec![FrontendToolV1 {
            name: "f".into(),
            description: None,
            parameters: FrontendJsonV1::Null,
            parameters_present: false,
            strict: None,
        }],
        tool_objects: vec![json(&function)],
        ..request(vec![message(FrontendRoleV1::User, "go"), assistant, result])
    }
}

#[test]
fn the_served_format_reads_tool_objects_and_arguments_as_the_encoder_does() {
    let frontend = PairedFrontendV1::chat_format::<DeepSeekV41Format>(descriptor()).unwrap();
    let render = |function: Value, arguments: &str| {
        frontend
            .render_chat(called(function, arguments))
            .map(|rendered| rendered.prompt)
            .map_err(|error| error.message)
    };
    let plain = serde_json::json!({"name": "f"});
    // A tool is the recipe's function object: name, description or "" and
    // parameters or {}, every other key of the client's object dropped.
    let prompt = render(
        serde_json::json!({"name": "f", "function": {"name": "g"}, "strict": true}),
        "{}",
    )
    .unwrap();
    assert!(
        prompt.contains(r#"{"name": "f", "description": "", "parameters": {}}"#),
        "{prompt}"
    );
    // Argument text the encoder reads itself: spacing kept in a value it
    // reads once, and a repeated key at its first place with its last value.
    let prompt = render(plain.clone(), "[1,  2]").unwrap();
    assert!(prompt.contains("string=\"true\">[1,  2]<"), "{prompt}");
    let prompt = render(plain.clone(), r#"{"x":1,"y":3,"x":2}"#).unwrap();
    assert!(
        prompt.contains(
            "name=\"x\" string=\"false\">2</｜DSML｜ parameter>\n\
             <｜DSML｜ parameter name=\"y\" string=\"false\">3<"
        ),
        "{prompt}"
    );
    // A namespace on a function object, and a message's own tools, are
    // refused by name.
    let refused = render(
        serde_json::json!({"name": "f", "namespace": {"name": "ns", "description": "N"}}),
        "{}",
    )
    .unwrap_err();
    assert!(
        refused.contains("tool namespaces are not served"),
        "{refused}"
    );
    let mut system = message(FrontendRoleV1::System, "s");
    system.tools = vec![json(
        &serde_json::json!({"type": "function", "namespace": "ns", "function": plain}),
    )];
    let refused = frontend
        .render_chat(request(vec![system, message(FrontendRoleV1::User, "q")]))
        .unwrap_err();
    assert!(
        refused.message.contains("message tools are not served"),
        "{}",
        refused.message
    );
    // System messages alone leave no turn to answer, once an empty one is
    // dropped as the recipe drops it.
    for only in [
        vec![message(FrontendRoleV1::System, "s")],
        vec![message(FrontendRoleV1::System, "")],
    ] {
        let refused = frontend.render_chat(request(only)).unwrap_err();
        assert!(
            refused.message.contains("has no turn to answer"),
            "{}",
            refused.message
        );
    }
}
