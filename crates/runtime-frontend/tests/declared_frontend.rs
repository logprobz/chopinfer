#![forbid(unsafe_code)]

//! Two bundles, one transport, no rebuild.
//!
//! Both frontends below are declarations, not code. They name different
//! models, serve different context limits, pin different vocabularies, render
//! materially different prompts, and grade tool calls by different rules. The
//! transport, the renderer, and this test binary are identical for both, so
//! whatever differs between the two responses came out of the installed data.

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use frontend_format::FormatDescriptorV1;
use frontend_format::test_support::edit_format;

use frontend_format::test_support::fixtures::{FUNCTION_PARAMETERS, KEY_VALUE};
use runtime_frontend::{
    BundleFrontendV1, FrontendComponentErrorV1, FrontendDescriptorV1, FrontendTokenizerPolicyV1,
    PromptModeV1, tokenizer_contract_v1,
};
use runtime_server::{
    AdmissionError, BackendReadiness, BackendToolCall, BoxFuture, EventSender, FinishReason,
    GeneratedTokenLogprob, GenerationBackend, GenerationEvent, PreparedPrompt, PreparedRequest,
    ReasoningContent, ServerConfig, ServerFrontendV1, TokenLogprob, TokenUsage, VisibleContent,
    serve,
};
use serde_json::Value;
use tokio::net::{TcpListener, TcpStream};
use tokio::time::timeout;

/// A frontend that shares nothing with the key-value fixture: other
/// identities, a tenth of the context, a distinct vocabulary and normalizer,
/// prose role markers, its own reasoning vocabulary, and a far stricter
/// tool-call rule.
const NARROW: &str = r##"{
  "schema": "chop-frontend-openai-v2",
  "version": 2,
  "models": [{"id": "fixture/beta-9", "owned_by": "fixture-beta"}],
  "context": {"maximum_total_tokens": 48},
  "limits": {
    "maximum_prompt_bytes": 4096,
    "maximum_completion_bytes": 4096,
    "maximum_generated_tokens": 256,
    "maximum_client_stops": 2,
    "maximum_messages": 16,
    "maximum_tools": 2,
    "maximum_tool_calls": 1,
    "maximum_tool_parameters": 4,
    "maximum_name_bytes": 8,
    "maximum_json_bytes": 1024,
    "maximum_json_depth": 4,
    "maximum_json_values": 64
  },
  "reasoning": {
    "disable_value": "off",
    "default_effort": "brief",
    "efforts": [
      {"name": "brief", "preamble": "Be brief.\n"},
      {"name": "deep", "preamble": "Be exhaustive.\n"}
    ],
    "end_markers": ["<</reason>>", "<<calls>>"]
  },
  "tokenizer": {
    "repository": "fixture/beta",
    "revision": "89abcdef0123456789abcdef0123456789abcdef",
    "definition": {
      "name": "vocabulary.json",
      "byte_length": 2048,
      "sha256": "3333333333333333333333333333333333333333333333333333333333333333"
    },
    "configuration": {
      "name": "vocabulary_config.json",
      "byte_length": 256,
      "sha256": "4444444444444444444444444444444444444444444444444444444444444444"
    },
    "policy": {
      "normalizer": "nfc",
      "pretokenizer_byte_level": {"add_prefix_space": false, "trim_offsets": false, "use_regex": false},
      "post_processor_byte_level": {"add_prefix_space": false, "trim_offsets": false, "use_regex": false},
      "decoder_byte_level": {"add_prefix_space": false, "trim_offsets": false, "use_regex": false},
      "merge_results": {"kind": "vocabulary"},
      "merge_records": "string-or-pair",
      "base_tokens": {"kind": "byte-level"},
      "config_tokens": {"kind": "string"},
      "add_bos_token": {"kind": "required", "value": false},
      "bos_token": {"kind": "required-null"},
      "unk_token": {"kind": "required-null"},
      "tokenizer_class": "NfcTestTokenizer",
      "continuing_subword_prefix": "",
      "end_of_word_suffix": "",
      "ignore_merges": "disabled",
      "add_eos_token": null,
      "legacy": null,
      "empty_sentencepiece_kwargs": false
    },
    "base_vocabulary_size": 300,
    "merge_count": 8,
    "added_token_count": 7,
    "model_max_length": 48,
    "bos_token": "<<begin>>",
    "eos_token": "<<end>>",
    "pad_token": "<<end>>",
    "pretokenizer_patterns": ["\\S+"]
  },
  "tools": {
    "name_bytes": {"ascii_alphanumeric": true, "extra": "_-"},
    "arguments": "json-object"
  },
  "vocabulary": {
    "size": 777,
    "bos_token_id": 5,
    "eos_token_id": 6,
    "stop_token_ids": [6, 7],
    "boundary_tokens": [
      {"text": "<<begin>>", "id": 5},
      {"text": "<<end>>", "id": 6},
      {"text": "<<stop>>", "id": 7},
      {"text": "<<user>>", "id": 8},
      {"text": "<<bot>>", "id": 9},
      {"text": "<<reason>>", "id": 10},
      {"text": "<</reason>>", "id": 11},
      {"text": "<<calls>>", "id": 12}
    ],
    "prompt_special_tokens": ["<<begin>>", "<<end>>", "<<reason>>", "<</reason>>"]
  },
  "response": {"reasoning_field": "reasoning_content"},
  "format": {
    "parser": {
      "tool_grammar": "xml-key-value-arguments-v1",
      "json_encoding": "python-json-v1",
      "reasoning_open": "<<reason>>",
      "reasoning_close": "<</reason>>",
      "tool_call_open": "<<calls>>",
      "tool_call_close": ">>calls<<",
      "parameter_open": "<<key>>",
      "parameter_name_close": "<</key>>",
      "parameter_value_open": "<<value>>",
      "parameter_close": "<</value>>",
      "reserved_markers": ["<<begin>>", "<<end>>", "<<stop>>", "<<sys>>", "<<dev>>", "<<user>>", "<<bot>>", "<<reason>>", "<</reason>>", "<<result>>", "[[", "]]", "<<calls>>"],
      "turn_open": ["<<bot>>", "<<user>>"],
      "assistant_header": "<<bot>>"
    }
  }
}"##;

/// The frontend a deployment of `declaration` serves: the test fixture
/// format over its `format.parser` section.
fn frontend(declaration: &str) -> BundleFrontendV1 {
    BundleFrontendV1::from_export(
        &runtime_frontend::test_support::fixture_export_v1(),
        declaration.as_bytes(),
    )
    .expect("the fixture declaration is readable")
}

/// Media parts render as segments through the host, and a format's refusal
/// of a pad or placeholder a client wrote reaches the caller, however the
/// text is split into parts.
#[test]
fn declared_media_renders_segments_and_refuses_client_pad_text() {
    use runtime_frontend::{
        FrontendMediaArenaV1, FrontendMediaElementV1, FrontendMediaEncoderV1, FrontendMediaKindV1,
        FrontendMediaModalityV1, FrontendMediaRecipeV1, FrontendMediaRenderV1,
        FrontendMultimodalV1, FrontendPositionRuleV1,
    };
    use runtime_server::{
        FrontendChatRequestV1, FrontendContentPartV1, FrontendMediaPartKindV1, FrontendMediaPartV1,
        FrontendMessageV1, FrontendRoleV1, FrontendToolChoiceV1, PromptSegmentV1,
    };
    let mut descriptor = FrontendDescriptorV1::from_component_bytes(KEY_VALUE.as_bytes()).unwrap();
    descriptor.multimodal = Some(FrontendMultimodalV1 {
        arena: FrontendMediaArenaV1 {
            rows: 64,
            page_rows: 64,
            hidden: 8,
            element: FrontendMediaElementV1::Bf16,
        },
        encoders: vec![FrontendMediaEncoderV1 {
            name: "image".into(),
            payload_element: FrontendMediaElementV1::Bf16,
            payload_columns: 4,
            maximum_payload_bytes: 64,
            cost_per_row_milli: 1,
        }],
        modalities: vec![FrontendMediaModalityV1 {
            kind: FrontendMediaKindV1::Image,
            part_types: vec!["image_url".into()],
            maximum_items: 1,
            maximum_bytes: 64,
            maximum_tokens: 8,
            recipe: FrontendMediaRecipeV1 {
                tag: "image.patch-grid.v1".into(),
                parameters: serde_json::json!({}),
            },
            encoder: 0,
            render: FrontendMediaRenderV1 {
                pad: "[A]".into(),
                grid: None,
            },
            positions: None,
        }],
    });
    // A grid position rule needs an image modality whose recipe merges.
    let with_positions = |merge: serde_json::Value, kind| {
        let mut declared = descriptor.clone();
        let modality = &mut declared.multimodal.as_mut().unwrap().modalities[0];
        modality.positions = Some(FrontendPositionRuleV1::Grid3AxisV1);
        modality.recipe.parameters = serde_json::json!({ "merge": merge });
        if kind == FrontendMediaKindV1::Audio {
            modality.kind = kind;
            modality.recipe.tag = "audio.log-mel.v1".into();
            modality.part_types = vec!["input_audio".into()];
            let mut unpositioned = declared.clone();
            unpositioned.multimodal.as_mut().unwrap().modalities[0].positions = None;
            unpositioned
                .validate()
                .expect("the audio modality alone validates");
        }
        declared
    };
    let positioned = with_positions(serde_json::json!(2), FrontendMediaKindV1::Image);
    positioned.validate().expect("an image grid rule validates");
    let json = serde_json::to_value(&positioned).unwrap();
    assert_eq!(
        json["multimodal"]["modalities"][0]["positions"],
        "grid-3axis.v1"
    );
    assert_eq!(
        serde_json::from_value::<runtime_frontend::FrontendDescriptorV1>(json).unwrap(),
        positioned
    );
    // An image modality may declare either image recipe, and no other.
    for (tag, valid) in [
        ("image.zero-padded-token-budget.v1", true),
        ("audio.log-mel.v1", false),
    ] {
        let mut declared = positioned.clone();
        declared.multimodal.as_mut().unwrap().modalities[0]
            .recipe
            .tag = tag.into();
        assert_eq!(declared.validate().is_ok(), valid, "{tag}");
    }
    for (merge, kind) in [
        (serde_json::json!(0), FrontendMediaKindV1::Image),
        (serde_json::json!("2"), FrontendMediaKindV1::Image),
        (serde_json::json!(2), FrontendMediaKindV1::Audio),
    ] {
        assert!(with_positions(merge, kind).validate().is_err());
    }
    // An encoder payload may exceed the prompt cap, up to 512 MiB (a 16 MP
    // image as F32 patch rows is 384 MiB).
    for (bytes, valid) in [(384 << 20, true), ((512 << 20) + 1, false)] {
        let mut declared = descriptor.clone();
        declared.multimodal.as_mut().unwrap().encoders[0].maximum_payload_bytes = bytes;
        assert_eq!(declared.validate().is_ok(), valid);
    }
    let frontend = BundleFrontendV1::with_export(
        descriptor.clone(),
        &runtime_frontend::test_support::fixture_export_v1(),
    )
    .expect("media descriptor validates");
    // It serves the kinds it declares; a deployment narrows them, never
    // widens them.
    assert_eq!(frontend.served_media(), [FrontendMediaPartKindV1::Image]);
    assert!(!frontend.has_selected_modalities());
    let narrowed = frontend.clone().with_served_media(&[]);
    assert!(!narrowed.has_selected_modalities());
    assert!(
        narrowed
            .clone()
            .with_selected_modalities(true)
            .has_selected_modalities()
    );
    assert!(narrowed.served_media().is_empty() && narrowed.supports_media());
    assert_eq!(
        frontend
            .clone()
            .with_served_media(&FrontendMediaPartKindV1::ALL)
            .served_media(),
        [FrontendMediaPartKindV1::Image]
    );
    let media = FrontendMediaPartV1 {
        kind: FrontendMediaPartKindV1::Image,
        part_type: "image_url".into(),
        source: "data:image/png;base64,AAAA".into(),
        format: None,
    };
    let request = |text: &str| FrontendChatRequestV1 {
        continue_final_message: false,
        add_generation_prompt: true,
        messages: vec![FrontendMessageV1 {
            role: FrontendRoleV1::User,
            content: Some(text.into()),
            parts: Some(vec![
                FrontendContentPartV1::Text(text.into()),
                FrontendContentPartV1::Media(media.clone()),
            ]),
            reasoning_content: None,
            tool_calls: vec![],
            tool_call_id: None,
            tools: vec![],
        }],
        tools: vec![],
        tool_choice: FrontendToolChoiceV1::None,
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
    };
    let rendered = frontend
        .render_chat(request("look"))
        .expect("media renders");
    // The fixture format writes the item bare, and the host writes its pad.
    assert_eq!(rendered.prompt, "[U]user\nlook[A]\n[A]\n");
    assert_eq!(
        rendered.segments,
        vec![
            PromptSegmentV1::Text("[U]user\nlook".into()),
            PromptSegmentV1::Media(0),
            PromptSegmentV1::Text("\n[A]\n".into())
        ]
    );
    assert_eq!(rendered.media, vec![media.clone()]);
    for text in ["\u{e000}chopinfer-media:0:\u{e001}", "[A]"] {
        for field in 0..7 {
            let mut injected = request("look");
            let mut tool = runtime_server::FrontendToolV1 {
                name: "lookup".into(),
                description: None,
                parameters: serde_json::from_str(r#"{"type":"object"}"#).unwrap(),
                parameters_present: true,
                strict: None,
            };
            match field {
                0 => tool.description = Some(text.into()),
                1 => tool.name = text.into(),
                2 => {
                    tool.parameters = runtime_server::FrontendJsonV1::Object(vec![(
                        text.into(),
                        runtime_server::FrontendJsonV1::Null,
                    )])
                }
                3 => tool.parameters = runtime_server::FrontendJsonV1::String(text.into()),
                4 => injected.messages[0]
                    .tools
                    .push(runtime_server::FrontendJsonV1::String(text.into())),
                5 | 6 => injected.messages.push(FrontendMessageV1 {
                    role: FrontendRoleV1::Assistant,
                    content: None,
                    parts: None,
                    reasoning_content: None,
                    tool_calls: vec![runtime_server::FrontendToolCallV1 {
                        id: "call".into(),
                        name: if field == 5 {
                            text.into()
                        } else {
                            "lookup".into()
                        },
                        arguments: if field == 6 {
                            format!("{{\"value\":\"{text}\"}}")
                        } else {
                            "{}".into()
                        },
                    }],
                    tool_call_id: None,
                    tools: vec![],
                }),
                _ => unreachable!(),
            }
            injected.tools.push(tool);
            assert!(
                frontend.render_chat(injected).is_err(),
                "accepted reserved text in field {field}: {text}"
            );
        }
    }
    for text in ["\u{e000}chopinfer-media:0:\u{e001}", "[A]"] {
        for media_first in [false, true] {
            let mut injected = request("look");
            let mut parts: Vec<_> = text
                .chars()
                .map(|ch| FrontendContentPartV1::Text(ch.to_string()))
                .collect();
            let item = FrontendContentPartV1::Media(media.clone());
            if media_first {
                parts.insert(0, item);
            } else {
                parts.push(item);
            }
            injected.messages[0].parts = Some(parts);
            assert!(
                frontend.render_chat(injected).is_err(),
                "accepted a reserved string assembled from text parts"
            );
        }
    }
    let mut forced = request("look");
    forced.continue_final_message = true;
    forced.messages.push(FrontendMessageV1 {
        role: FrontendRoleV1::Assistant,
        content: Some("answer".into()),
        parts: None,
        reasoning_content: None,
        tool_calls: vec![],
        tool_call_id: None,
        tools: vec![],
    });
    let forced = frontend
        .render_chat(forced)
        .expect("open assistant renders");
    assert_eq!(forced.prompt, format!("{}answer", rendered.prompt));
    assert_eq!(forced.media, vec![media.clone()]);
    let refusal = frontend
        .render_chat(request("[A]"))
        .expect_err("client pad text is refused");
    assert_eq!(refusal.parameter.as_deref(), Some("messages"));

    // The image limit is one count over every message, reported in the
    // wording clients that recover from image limits look for.
    assert_eq!(frontend.maximum_images_per_request(), Some(1));
    assert_eq!(narrowed.maximum_images_per_request(), None);
    let mut two = request("look");
    two.messages.push(two.messages[0].clone());
    let refusal = frontend
        .render_chat(two.clone())
        .expect_err("two images across two messages exceed one");
    assert_eq!(
        refusal.message,
        "At most 1 image(s) may be provided in one request; this request carried 2."
    );
    assert_eq!(refusal.parameter.as_deref(), Some("image"));
    let widened = frontend.clone().with_maximum_images(2);
    assert_eq!(widened.maximum_images_per_request(), Some(2));
    assert_eq!(
        widened
            .render_chat(two.clone())
            .expect("two images fit a limit of two")
            .media,
        vec![media.clone(), media.clone()]
    );
    // An oversized image is not a count problem, so it keeps its message
    // and parameter; over the count as well, the count is reported first.
    let mut oversized = request("look");
    let Some(FrontendContentPartV1::Media(item)) =
        oversized.messages[0].parts.as_mut().unwrap().last_mut()
    else {
        unreachable!("the request ends with its image");
    };
    item.source = format!("data:image/png;base64,{}", "A".repeat(400));
    let refusal = frontend
        .render_chat(oversized.clone())
        .expect_err("an oversized image is refused");
    assert_eq!(refusal.message, "image_url exceeds maximum_bytes 64");
    assert_eq!(refusal.parameter.as_deref(), Some("messages"));
    oversized.messages.push(oversized.messages[0].clone());
    let refusal = frontend
        .render_chat(oversized)
        .expect_err("two oversized images are refused");
    assert_eq!(refusal.parameter.as_deref(), Some("image"));

    // Other media kinds keep their per-kind count and its wording.
    let mut audio = descriptor.clone();
    let modality = &mut audio.multimodal.as_mut().unwrap().modalities[0];
    modality.kind = FrontendMediaKindV1::Audio;
    modality.recipe.tag = "audio.log-mel.v1".into();
    modality.part_types = vec!["input_audio".into()];
    let audio =
        BundleFrontendV1::with_export(audio, &runtime_frontend::test_support::fixture_export_v1())
            .expect("an audio declaration validates");
    assert_eq!(audio.maximum_images_per_request(), None);
    let clip = FrontendMediaPartV1 {
        kind: FrontendMediaPartKindV1::Audio,
        part_type: "input_audio".into(),
        source: "AAAA".into(),
        format: Some("wav".into()),
    };
    let mut clips = request("listen");
    clips.messages[0].parts = Some(vec![
        FrontendContentPartV1::Media(clip.clone()),
        FrontendContentPartV1::Media(clip),
    ]);
    let refusal = audio
        .render_chat(clips)
        .expect_err("two audio items exceed one");
    assert_eq!(refusal.message, "input_audio exceeds maximum_items 1");
    assert_eq!(refusal.parameter.as_deref(), Some("messages"));
}

/// One declaration with a single field swapped, for a refusal case.
fn swapped(declaration: &str, from: &str, to: &str) -> String {
    let replaced = declaration.replacen(from, to, 1);
    assert_ne!(replaced, declaration, "the fixture never contained {from}");
    replaced
}

struct ScriptedBackendV1 {
    events: Vec<GenerationEvent>,
    prompts: Mutex<Vec<PreparedRequest>>,
    prompt_tokens: usize,
    submissions: AtomicUsize,
}

impl ScriptedBackendV1 {
    fn new(events: Vec<GenerationEvent>, prompt_tokens: usize) -> Arc<Self> {
        Arc::new(Self {
            events,
            prompts: Mutex::new(Vec::new()),
            prompt_tokens,
            submissions: AtomicUsize::new(0),
        })
    }

    fn rendered_prompt(&self) -> String {
        let prompts = self.prompts.lock().expect("the prompt log is usable");
        let request = prompts.first().expect("one request reached the backend");
        match &request.prompt {
            PreparedPrompt::RenderedChat(prompt) => prompt.clone(),
            PreparedPrompt::Raw(prompt) => prompt.clone(),
            PreparedPrompt::TokenIds(_) => panic!("the fixture renders text prompts"),
            PreparedPrompt::MediaChat { .. } => panic!("the fixture renders text prompts"),
            PreparedPrompt::MediaTokens { .. } => panic!("the fixture renders text prompts"),
        }
    }

    fn submissions(&self) -> usize {
        self.submissions.load(Ordering::Relaxed)
    }
}

impl GenerationBackend for ScriptedBackendV1 {
    fn readiness(&self) -> BackendReadiness {
        BackendReadiness::Ready
    }

    fn maximum_sequence_tokens(&self) -> usize {
        4096
    }

    fn prompt_token_count(&self, _prompt: &PreparedPrompt) -> Result<usize, AdmissionError> {
        Ok(self.prompt_tokens)
    }

    fn submit(
        &self,
        request: PreparedRequest,
        events: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        self.submissions.fetch_add(1, Ordering::Relaxed);
        self.prompts
            .lock()
            .expect("the prompt log is usable")
            .push(request);
        let scripted = self.events.clone();
        Box::pin(async move {
            tokio::spawn(async move {
                for event in scripted {
                    if events.emit(event).await.is_err() {
                        return;
                    }
                }
            });
            Ok(())
        })
    }
}

fn finished() -> GenerationEvent {
    GenerationEvent::Finished {
        reason: FinishReason::Stop,
        usage: TokenUsage {
            prompt_tokens: 1,
            completion_tokens: 1,
            cached_tokens: None,
        },
    }
}

fn content(delta: &str) -> GenerationEvent {
    GenerationEvent::Content(VisibleContent {
        delta: delta.to_owned(),
        generated_token_count: 1,
    })
}

fn tool_call(name: &str) -> GenerationEvent {
    GenerationEvent::ToolCalls {
        calls: vec![BackendToolCall {
            name: name.to_owned(),
            arguments: "{}".to_owned(),
        }],
        generated_token_count: 1,
    }
}

/// One generated token's logprob record; `top` lists the position's entries
/// in vLLM's order, the generated token first.
fn logprob(token: &str, value: f64, top: &[(&str, f64)]) -> GenerationEvent {
    let entry = |token: &str, logprob: f64| TokenLogprob {
        token: token.to_owned(),
        bytes: token.as_bytes().to_vec(),
        logprob,
    };
    GenerationEvent::GeneratedTokenLogprobs(vec![GeneratedTokenLogprob {
        token_id: 0,
        token: token.to_owned(),
        bytes: token.as_bytes().to_vec(),
        logprob: value,
        top_logprobs: std::iter::once(entry(token, value))
            .chain(top.iter().map(|(token, logprob)| entry(token, *logprob)))
            .collect(),
    }])
}

fn finished_after(completion_tokens: u64) -> GenerationEvent {
    GenerationEvent::Finished {
        reason: FinishReason::Stop,
        usage: TokenUsage {
            prompt_tokens: 1,
            completion_tokens,
            cached_tokens: None,
        },
    }
}

/// The data payloads of a server-sent event stream.
fn stream_chunks(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("each event is JSON"))
        .collect()
}

struct TestServerV1 {
    address: SocketAddr,
}

impl TestServerV1 {
    async fn start(backend: Arc<dyn GenerationBackend>, declaration: &str) -> Self {
        Self::with_config(backend, declaration, ServerConfig::default()).await
    }

    async fn with_config(
        backend: Arc<dyn GenerationBackend>,
        declaration: &str,
        config: ServerConfig,
    ) -> Self {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("a loopback port is available");
        let address = listener.local_addr().expect("the listener has an address");
        let frontend: Arc<dyn ServerFrontendV1> = Arc::new(frontend(declaration));
        tokio::spawn(async move {
            let _ = serve(listener, backend, frontend, config).await;
        });
        Self { address }
    }

    async fn request(&self, request: &str) -> ResponseV1 {
        let stream = TcpStream::connect(self.address)
            .await
            .expect("the server accepts connections");
        let raw = timeout(
            Duration::from_secs(5),
            exchange(&stream, request.as_bytes()),
        )
        .await
        .expect("the response arrives");
        let text = String::from_utf8_lossy(&raw).into_owned();
        let (head, body) = text.split_once("\r\n\r\n").unwrap_or((text.as_str(), ""));
        let status = head
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|code| code.parse().ok())
            .unwrap_or(0);
        ResponseV1 {
            status,
            body: body.to_owned(),
        }
    }
}

async fn exchange(stream: &TcpStream, mut request: &[u8]) -> Vec<u8> {
    while !request.is_empty() {
        stream.writable().await.expect("the socket is writable");
        match stream.try_write(request) {
            Ok(0) => panic!("the socket closed while writing"),
            Ok(written) => request = &request[written..],
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(error) => panic!("the socket write failed: {error}"),
        }
    }
    let mut response = Vec::new();
    loop {
        stream.readable().await.expect("the socket is readable");
        let mut chunk = [0_u8; 4096];
        match stream.try_read(&mut chunk) {
            Ok(0) => return response,
            Ok(read) => response.extend_from_slice(&chunk[..read]),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(error) => panic!("the socket read failed: {error}"),
        }
    }
}

struct ResponseV1 {
    status: u16,
    body: String,
}

impl ResponseV1 {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body).expect("the response body is JSON")
    }
}

fn get(path: &str) -> String {
    format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
}

fn post(path: &str, body: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

/// The identities a deployment answers to come out of the installed data.
#[tokio::test]
async fn each_declaration_lists_and_accepts_only_its_own_models() {
    let alpha = TestServerV1::start(ScriptedBackendV1::new(Vec::new(), 1), KEY_VALUE).await;
    let beta = TestServerV1::start(ScriptedBackendV1::new(Vec::new(), 1), NARROW).await;

    let listed = |body: &Value| -> Vec<String> {
        body["data"]
            .as_array()
            .expect("the listing is an array")
            .iter()
            .map(|model| model["id"].as_str().expect("an id").to_owned())
            .collect()
    };

    assert_eq!(
        listed(&alpha.request(&get("/v1/models")).await.json()),
        vec!["fixture/alpha-1", "alpha"]
    );
    assert_eq!(
        listed(&beta.request(&get("/v1/models")).await.json()),
        vec!["fixture/beta-9"]
    );

    // Each transport refuses the other deployment's model, and neither one had
    // to be rebuilt to know that.
    let ask = |model: &str| {
        post(
            "/v1/chat/completions",
            &format!(
                r#"{{"model":"{model}","messages":[{{"role":"user","content":"x"}}],"max_completion_tokens":4}}"#
            ),
        )
    };
    assert_eq!(alpha.request(&ask("fixture/beta-9")).await.status, 404);
    assert_eq!(beta.request(&ask("alpha")).await.status, 404);
}

/// The prompt a conversation renders into is the declared one.
#[tokio::test]
async fn each_declaration_renders_its_own_prompt() {
    let conversation = r#"[
        {"role":"system","content":"be kind"},
        {"role":"user","content":"first"},
        {"role":"assistant","content":"answer","reasoning_content":"why"},
        {"role":"user","content":"second"}
    ]"#;
    let tools =
        r#"[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]"#;

    let alpha_backend = ScriptedBackendV1::new(vec![content("ok"), finished()], 1);
    let alpha = TestServerV1::start(Arc::clone(&alpha_backend) as Arc<_>, KEY_VALUE).await;
    let response = alpha
        .request(&post(
            "/v1/chat/completions",
            &format!(
                r#"{{"model":"alpha","messages":{conversation},"tools":{tools},"max_completion_tokens":4,"reasoning_effort":"high"}}"#
            ),
        ))
        .await;
    assert_eq!(response.status, 200);
    assert_eq!(
        alpha_backend.rendered_prompt(),
        concat!(
            "[U]system\nTHINK HARDER.\n\n",
            "[U]tools\n{\"name\": \"lookup\", \"parameters\": {\"type\": \"object\"}}\n",
            "[U]system\nbe kind\n[U]user\nfirst\n[A]\n<t>why</t>answer\n",
            "[U]user\nsecond\n[A]\n<t>"
        )
    );

    let beta_backend = ScriptedBackendV1::new(vec![content("ok"), finished()], 1);
    let beta = TestServerV1::start(Arc::clone(&beta_backend) as Arc<_>, NARROW).await;
    let response = beta
        .request(&post(
            "/v1/chat/completions",
            &format!(
                r#"{{"model":"fixture/beta-9","messages":{conversation},"tools":{tools},"max_completion_tokens":4,"reasoning_effort":"deep"}}"#
            ),
        ))
        .await;
    assert_eq!(response.status, 200);
    // No marker of this prompt is shared with the first, and its preamble is
    // the effort this declaration names.
    assert_eq!(
        beta_backend.rendered_prompt(),
        concat!(
            "<<user>>system\nBe exhaustive.\n\n",
            "<<user>>tools\n{\"name\": \"lookup\", \"parameters\": {\"type\": \"object\"}}\n",
            "<<user>>system\nbe kind\n<<user>>user\nfirst\n",
            "<<bot>>\n<<reason>>why<</reason>>answer\n",
            "<<user>>user\nsecond\n<<bot>>\n<<reason>>"
        )
    );
}

/// The typed read takes `chat_template_kwargs` given as an array by position,
/// so a declared format renders it as the object of the keywords it set.
#[tokio::test]
async fn template_keywords_given_as_an_array_render_as_their_object() {
    let mut prompts = Vec::new();
    for kwargs in [
        r#"{"enable_thinking":false}"#,
        "[null,false,null,null,null,null]",
    ] {
        let backend = ScriptedBackendV1::new(vec![content("ok"), finished()], 1);
        let alpha = TestServerV1::start(Arc::clone(&backend) as Arc<_>, KEY_VALUE).await;
        let response = alpha
            .request(&post(
                "/v1/chat/completions",
                &format!(
                    r#"{{"model":"alpha","messages":[{{"role":"user","content":"x"}}],"max_completion_tokens":4,"chat_template_kwargs":{kwargs}}}"#
                ),
            ))
            .await;
        assert_eq!(response.status, 200, "{kwargs}: {}", response.body);
        prompts.push(backend.rendered_prompt());
    }
    assert_eq!(prompts[0], "[U]user\nx\n[A]\n");
    assert_eq!(prompts[1], prompts[0]);
}

/// A declared format renders only the common roles: its frontend declares no
/// extra role, so the transport refuses an unknown role exactly as it did
/// before frontends could declare them.
#[tokio::test]
async fn a_declared_format_admits_only_the_common_roles() {
    for declaration in [KEY_VALUE, NARROW, FUNCTION_PARAMETERS] {
        assert!(frontend(declaration).extra_roles().is_empty());
    }
    let backend = ScriptedBackendV1::new(vec![content("ok"), finished()], 1);
    let alpha = TestServerV1::start(Arc::clone(&backend) as Arc<_>, KEY_VALUE).await;
    let refused = alpha
        .request(&post(
            "/v1/chat/completions",
            r#"{"model":"alpha","messages":[{"role":"observation","content":"x"}],"max_completion_tokens":4}"#,
        ))
        .await;
    assert_eq!(refused.status, 400);
    assert_eq!(
        refused.json()["error"]["message"],
        "message role must be system, developer, user, assistant, or tool"
    );
    assert_eq!(refused.json()["error"]["param"], "messages");
    assert_eq!(backend.submissions(), 0);
}

/// The reasoning vocabulary is declared, so each deployment accepts its own.
#[tokio::test]
async fn each_declaration_owns_its_reasoning_vocabulary() {
    let alpha = TestServerV1::start(
        ScriptedBackendV1::new(vec![content("ok"), finished()], 1),
        KEY_VALUE,
    )
    .await;
    let beta = TestServerV1::start(
        ScriptedBackendV1::new(vec![content("ok"), finished()], 1),
        NARROW,
    )
    .await;
    let ask = |model: &str, effort: &str| {
        post(
            "/v1/chat/completions",
            &format!(
                r#"{{"model":"{model}","messages":[{{"role":"user","content":"x"}}],"max_completion_tokens":4,"reasoning_effort":"{effort}"}}"#
            ),
        )
    };

    assert_eq!(alpha.request(&ask("alpha", "high")).await.status, 200);
    let refused = alpha.request(&ask("alpha", "deep")).await;
    assert_eq!(refused.status, 400);
    assert_eq!(refused.json()["error"]["param"], "reasoning_effort");

    assert_eq!(
        beta.request(&ask("fixture/beta-9", "deep")).await.status,
        200
    );
    let refused = beta.request(&ask("fixture/beta-9", "high")).await;
    assert_eq!(refused.status, 400);
    assert_eq!(refused.json()["error"]["param"], "reasoning_effort");
    // Even the value that turns reasoning off is declared, not assumed.
    assert_eq!(
        beta.request(&ask("fixture/beta-9", "off")).await.status,
        200
    );
    assert_eq!(alpha.request(&ask("alpha", "none")).await.status, 200);
}

/// The context a request is charged against is the declared one.
#[tokio::test]
async fn each_declaration_enforces_its_own_context_limit() {
    let alpha_backend = ScriptedBackendV1::new(vec![content("ok"), finished()], 8);
    let alpha = TestServerV1::start(Arc::clone(&alpha_backend) as Arc<_>, KEY_VALUE).await;
    let beta_backend = ScriptedBackendV1::new(vec![content("ok"), finished()], 8);
    let beta = TestServerV1::start(Arc::clone(&beta_backend) as Arc<_>, NARROW).await;
    let ask = |model: &str, tokens: usize| {
        post(
            "/v1/completions",
            &format!(r#"{{"model":"{model}","prompt":"x","max_tokens":{tokens}}}"#),
        )
    };

    // The backend reports eight prompt tokens, so forty-one of completion fit
    // the first declaration's context and overrun the second's. That is the
    // whole difference between the two deployments.
    assert_eq!(alpha.request(&ask("alpha", 41)).await.status, 200);
    assert_eq!(alpha_backend.submissions(), 1);
    let refused = beta.request(&ask("fixture/beta-9", 41)).await;
    assert_eq!(refused.status, 400);
    assert_eq!(refused.json()["error"]["code"], 400);
    assert_eq!(beta_backend.submissions(), 0);
    // One token less is inside the second declaration's context after all.
    assert_eq!(beta.request(&ask("fixture/beta-9", 40)).await.status, 200);
}

/// A tool call is graded by the rules the installed data declared.
#[tokio::test]
async fn each_declaration_grades_tool_calls_by_its_own_rules() {
    let offered = |model: &str, name: &str| {
        post(
            "/v1/chat/completions",
            &format!(
                r#"{{"model":"{model}","messages":[{{"role":"user","content":"x"}}],"max_completion_tokens":4,"tools":[{{"type":"function","function":{{"name":"{name}","parameters":{{"type":"object"}}}}}}]}}"#
            ),
        )
    };

    // A group of two calls is inside the first declaration's bound and past the
    // second's, so the same backend output serves under one and is refused
    // under the other.
    let group = GenerationEvent::ToolCalls {
        calls: (0..2)
            .map(|_| BackendToolCall {
                name: "look".to_owned(),
                arguments: "{}".to_owned(),
            })
            .collect(),
        generated_token_count: 1,
    };
    let alpha = TestServerV1::start(
        ScriptedBackendV1::new(vec![group.clone(), finished()], 1),
        KEY_VALUE,
    )
    .await;
    let response = alpha.request(&offered("alpha", "look")).await;
    assert_eq!(response.status, 200);
    assert_eq!(
        response.json()["choices"][0]["message"]["tool_calls"]
            .as_array()
            .expect("a tool call group")
            .len(),
        2
    );

    let beta =
        TestServerV1::start(ScriptedBackendV1::new(vec![group, finished()], 1), NARROW).await;
    assert_eq!(
        beta.request(&offered("fixture/beta-9", "look"))
            .await
            .status,
        500
    );

    // The name a request may offer at all is declared too: nine bytes is inside
    // the first declaration's bound and past the second's.
    let alpha = TestServerV1::start(
        ScriptedBackendV1::new(vec![tool_call("lookup_it"), finished()], 1),
        KEY_VALUE,
    )
    .await;
    let response = alpha.request(&offered("alpha", "lookup_it")).await;
    assert_eq!(response.status, 200);
    assert_eq!(
        response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "lookup_it"
    );

    let beta = TestServerV1::start(ScriptedBackendV1::new(Vec::new(), 1), NARROW).await;
    let refused = beta.request(&offered("fixture/beta-9", "lookup_it")).await;
    assert_eq!(refused.status, 400);
    assert_eq!(refused.json()["error"]["param"], "messages");
}

/// The vocabulary a deployment pins is declared beside everything else.
#[test]
fn each_declaration_pins_its_own_vocabulary() {
    let alpha = frontend(KEY_VALUE);
    let beta = frontend(NARROW);
    assert_eq!(alpha.descriptor().vocabulary.size, 4096);
    assert_eq!(beta.descriptor().vocabulary.size, 777);
    assert_eq!(alpha.descriptor().vocabulary.stop_token_ids, vec![1]);
    assert_eq!(beta.descriptor().vocabulary.stop_token_ids, vec![6, 7]);
    assert_eq!(alpha.maximum_tool_calls(), 4);
    assert_eq!(beta.maximum_tool_calls(), 1);
    assert_eq!(
        beta.descriptor().vocabulary.prompt_special_tokens,
        vec!["<<begin>>", "<<end>>", "<<reason>>", "<</reason>>"]
    );
}

/// A deployment may serve less than the declaration allows, never more.
#[test]
fn a_deployment_may_only_narrow_the_declared_context() {
    assert_eq!(frontend(KEY_VALUE).maximum_total_tokens(), 512);
    assert_eq!(
        frontend(KEY_VALUE)
            .with_maximum_total_tokens(16)
            .maximum_total_tokens(),
        16
    );
    // The bundle is what knows this model's context, so a deployment asking
    // for more than it declared gets the declaration, not its own number.
    assert_eq!(
        frontend(KEY_VALUE)
            .with_maximum_total_tokens(4_096)
            .maximum_total_tokens(),
        512
    );
    assert_eq!(
        frontend(NARROW)
            .with_maximum_total_tokens(usize::MAX)
            .maximum_total_tokens(),
        48
    );
}

/// A descriptor assembled in code is held to the rules a component is.
#[test]
fn a_frontend_cannot_be_built_from_an_unchecked_declaration() {
    let export = runtime_frontend::test_support::fixture_export_v1();
    let mut descriptor = FrontendDescriptorV1::from_component_bytes(KEY_VALUE.as_bytes())
        .expect("the fixture declaration is readable");
    assert!(BundleFrontendV1::with_export(descriptor.clone(), &export).is_ok());

    // A blank parser marker leaves the parser nothing to find, which is
    // exactly the check a caller would otherwise skip by assembling the
    // descriptor directly.
    let blank = edit_format(&descriptor, |format| {
        format.parser.tool_call_close = String::new();
    });
    let error = BundleFrontendV1::with_export(blank, &export)
        .expect_err("a blank parser marker must not build a frontend");
    assert!(
        matches!(error, FrontendComponentErrorV1::Invalid(_)),
        "{error}"
    );

    // A declaration from another schema cannot be laundered through code
    // either, because the same rules run on the way in.
    descriptor.schema = "chop-frontend-openai-v1".to_owned();
    assert!(matches!(
        BundleFrontendV1::with_export(descriptor, &export),
        Err(FrontendComponentErrorV1::Schema { .. })
    ));
}

/// Duplicates make the reserved set and the vocabulary look larger than
/// they are.
#[test]
fn a_declaration_names_each_marker_and_boundary_once() {
    for (from, to) in [
        (
            r#""reserved_markers": ["[BOS]", "#,
            r#""reserved_markers": ["[BOS]", "[BOS]", "#,
        ),
        (
            r#"{"text": "[BOS]", "id": 0}"#,
            r#"{"text": "[BOS]", "id": 0}, {"text": "[BOS]", "id": 7}"#,
        ),
        (
            r#"{"text": "[EOS]", "id": 1}"#,
            r#"{"text": "[EOS]", "id": 1}, {"text": "[U]", "id": 1}"#,
        ),
    ] {
        let error =
            FormatDescriptorV1::from_component_bytes(swapped(KEY_VALUE, from, to).as_bytes())
                .expect_err("a duplicate declaration must be refused");
        assert!(
            matches!(error, FrontendComponentErrorV1::Invalid(_)),
            "{error}"
        );
    }
}

/// A prompt may only carry boundaries the vocabulary names, so a trusted
/// special token that is not one of them is refused, however the declaration
/// arrives.
///
/// A decoder checks the identifier a boundary arrives under, which it can do
/// only for a boundary the vocabulary gave one. A package that trusts its
/// parser's call open in the prompt without declaring that boundary is the
/// case this guards.
#[test]
fn a_trusted_prompt_token_must_be_a_declared_boundary() {
    let expected = "prompt special token '<tool_call>' is not a declared boundary token";
    let export = runtime_frontend::test_support::fixture_export_v1();
    for declaration in [KEY_VALUE, FUNCTION_PARAMETERS] {
        let error = FrontendDescriptorV1::from_component_bytes(
            swapped(
                declaration,
                r#""prompt_special_tokens": ["#,
                r#""prompt_special_tokens": ["<tool_call>", "#,
            )
            .as_bytes(),
        )
        .expect_err("a trusted token outside the boundary set must be refused");
        assert!(
            matches!(&error, FrontendComponentErrorV1::Invalid(message) if message == expected),
            "{error}"
        );

        // A descriptor assembled in code meets the same check.
        let mut descriptor = FrontendDescriptorV1::from_component_bytes(declaration.as_bytes())
            .expect("the fixture declaration is readable");
        descriptor
            .vocabulary
            .prompt_special_tokens
            .insert(0, "<tool_call>".to_owned());
        let error = BundleFrontendV1::with_export(descriptor, &export)
            .expect_err("a trusted token outside the boundary set must not build a frontend");
        assert!(
            matches!(&error, FrontendComponentErrorV1::Invalid(message) if message == expected),
            "{error}"
        );
    }
}

/// A pinned digest is refused here if the tokenizer reader would refuse it.
///
/// The reader accepts one spelling of a SHA-256 and no other. Reading an
/// uppercase one here would put the refusal at deployment instead, after the
/// bundle was already selected, so the two agree on the spelling.
#[test]
fn a_declared_tokenizer_digest_must_be_lowercase() {
    let lowercase = "abcdef1111111111111111111111111111111111111111111111111111111111";
    let declared = frontend(KEY_VALUE);
    assert_eq!(declared.descriptor().tokenizer.definition.sha256, lowercase);
    // The reader agrees, so what parses here is what deploys later.
    assert!(tokenizer_contract_v1(declared.descriptor()).is_ok());

    for (from, to) in [
        (lowercase, lowercase.to_ascii_uppercase()),
        (lowercase, format!("ABCDEF{}", &lowercase[6..])),
        (
            "fedcba2222222222222222222222222222222222222222222222222222222222",
            "FEDCBA2222222222222222222222222222222222222222222222222222222222".to_owned(),
        ),
    ] {
        let error =
            FrontendDescriptorV1::from_component_bytes(swapped(KEY_VALUE, from, &to).as_bytes())
                .expect_err("an uppercase digest must be refused");
        assert!(
            matches!(error, FrontendComponentErrorV1::Invalid(_)),
            "{to}: {error}"
        );
    }

    // A digest that is not hexadecimal at all is refused the same way.
    let error = FrontendDescriptorV1::from_component_bytes(
        swapped(KEY_VALUE, lowercase, &"g".repeat(64)).as_bytes(),
    )
    .expect_err("a non-hexadecimal digest must be refused");
    assert!(
        matches!(error, FrontendComponentErrorV1::Invalid(_)),
        "{error}"
    );
}

/// A rendered chat carries the mode and effort the declaration selected.
#[test]
fn the_chat_state_carries_the_declared_effort() {
    use runtime_server::{
        FrontendChatRequestV1, FrontendMessageV1, FrontendRoleV1, FrontendToolChoiceV1,
    };

    let beta = frontend(NARROW);
    let ask = |effort: Option<&str>| FrontendChatRequestV1 {
        continue_final_message: false,
        add_generation_prompt: true,
        messages: vec![FrontendMessageV1 {
            role: FrontendRoleV1::User,
            content: Some("x".to_owned()),
            parts: None,
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tools: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: FrontendToolChoiceV1::None,
        response_format: None,
        reasoning_effort: effort.map(str::to_owned),
        enable_thinking: None,
        thinking: None,
        nested_reasoning_effort: None,
        preserve_thinking: None,
        clear_thinking: None,
        add_vision_id: None,
        tool_objects: Vec::new(),
        chat_template_kwargs: None,
    };

    let rendered = beta.render_chat(ask(Some("deep"))).expect("deep renders");
    let state = runtime_frontend::BundleChatStateV1::from_state(&rendered.state)
        .expect("this frontend produced the state");
    assert_eq!(state.mode(), PromptModeV1::Thinking);
    assert_eq!(state.reasoning_effort(), "deep");
    assert!(state.reasoning_open());

    let rendered = beta.render_chat(ask(Some("off"))).expect("off renders");
    let state = runtime_frontend::BundleChatStateV1::from_state(&rendered.state)
        .expect("this frontend produced the state");
    assert_eq!(state.mode(), PromptModeV1::Chat);
    assert!(!state.reasoning_open());
    // Reasoning is off, so the effort is the declared default rather than a
    // value the request chose.
    assert_eq!(state.reasoning_effort(), "brief");

    let required = swapped(
        NARROW,
        r#""default_effort": "brief","#,
        r#""default_effort": "brief",
    "default_enabled": true,
    "allow_disabled": false,"#,
    );
    let required = frontend(&required);
    let rendered = required
        .render_chat(ask(None))
        .expect("the omitted control selects the enabled default");
    let state = runtime_frontend::BundleChatStateV1::from_state(&rendered.state)
        .expect("this frontend produced the state");
    assert_eq!(state.mode(), PromptModeV1::Thinking);
    assert_eq!(state.reasoning_effort(), "brief");

    let error = required
        .render_chat(ask(Some("off")))
        .expect_err("this declaration refuses disabled reasoning");
    assert!(
        error
            .to_string()
            .contains("does not allow reasoning to be disabled")
    );
}

/// A declaration this host cannot read is refused rather than coerced.
#[test]
fn a_declaration_outside_the_schema_is_refused() {
    let swap = |from: &str, to: &str| KEY_VALUE.replacen(from, to, 1);
    type RefusalV1 = (String, fn(&FrontendComponentErrorV1) -> bool);
    let cases: Vec<RefusalV1> = vec![
        // There is no v1 reader.
        (
            swap("chop-frontend-openai-v2", "chop-frontend-openai-v1"),
            |error| matches!(error, FrontendComponentErrorV1::Schema { .. }),
        ),
        (swap("\"version\": 2", "\"version\": 4"), |error| {
            matches!(error, FrontendComponentErrorV1::Version { .. })
        }),
        // An unknown field is a declaration this host does not understand.
        (swap("\"tools\": {", "\"tool\": {"), |error| {
            matches!(error, FrontendComponentErrorV1::Syntax(_))
        }),
        // The tokenizer policy is closed the same way: a named dialect, an
        // unknown member, an omitted member and an unknown component choice
        // are each a declaration this host does not understand.
        (swap("\"policy\":", "\"dialect\":"), |error| {
            matches!(error, FrontendComponentErrorV1::Syntax(_))
        }),
        (
            swap("\"legacy\": true,", "\"legacy\": true, \"strict\": true,"),
            |error| matches!(error, FrontendComponentErrorV1::Syntax(_)),
        ),
        (swap("\n      \"legacy\": true,", ""), |error| {
            matches!(error, FrontendComponentErrorV1::Syntax(_))
        }),
        (
            swap(
                "\"normalizer\": \"identity-sequence\"",
                "\"normalizer\": \"nfkc\"",
            ),
            |error| matches!(error, FrontendComponentErrorV1::Syntax(_)),
        ),
        (
            swap(
                "\"tokenizer_class\": \"PreTrainedTokenizerFast\"",
                "\"tokenizer_class\": \"\"",
            ),
            |error| matches!(error, FrontendComponentErrorV1::Invalid(_)),
        ),
        (
            swap(
                "\"value\": {\"kind\": \"contract-value\"}",
                "\"value\": {\"kind\": \"literal\", \"text\": \"\"}",
            ),
            |error| matches!(error, FrontendComponentErrorV1::Invalid(_)),
        ),
        // Every declared bound is checked against a ceiling this host fixes.
        (
            swap("\"maximum_json_depth\": 8", "\"maximum_json_depth\": 0"),
            |error| matches!(error, FrontendComponentErrorV1::Invalid(_)),
        ),
        // A default effort the declaration never named would leave the
        // reasoning controls undefined.
        (
            swap("\"default_effort\": \"low\"", "\"default_effort\": \"mid\""),
            |error| matches!(error, FrontendComponentErrorV1::Invalid(_)),
        ),
        // A token identifier outside the declared vocabulary would stop nothing.
        (
            swap("\"stop_token_ids\": [1]", "\"stop_token_ids\": [9999]"),
            |error| matches!(error, FrontendComponentErrorV1::Invalid(_)),
        ),
        (
            swap("\"id\": \"alpha\"", "\"id\": \"fixture/alpha-1\""),
            |error| matches!(error, FrontendComponentErrorV1::Invalid(_)),
        ),
    ];
    for (declaration, expected) in cases {
        let error = FrontendDescriptorV1::from_component_bytes(declaration.as_bytes())
            .expect_err("this declaration must be refused");
        assert!(expected(&error), "unexpected refusal: {error}");
    }

    assert!(matches!(
        FrontendDescriptorV1::from_component_bytes(&vec![b'x'; 512 * 1024]),
        Err(FrontendComponentErrorV1::TooLarge { .. })
    ));
    assert!(matches!(
        FrontendDescriptorV1::from_component_bytes(&[0xff, 0xfe]),
        Err(FrontendComponentErrorV1::NotUtf8)
    ));
}

/// Reading a declaration proves the tokenizer it pins can actually be built.
///
/// Every case below is a rule only the tokenizer reader has. Restating them
/// here is how the two would drift, so validation builds the reader's own
/// contract instead. What that buys is when the refusal lands: reading, where
/// the bundle has not been selected, rather than deployment, where it has.
#[test]
fn a_declaration_must_build_the_tokenizer_contract_it_pins() {
    for (from, to, why) in [
        (
            r#""name": "tokenizer.json""#,
            r#""name": "sub/tokenizer.json""#,
            "an artifact name that is a path",
        ),
        (
            r#""name": "tokenizer_config.json""#,
            r#""name": "../tokenizer_config.json""#,
            "a configuration name that escapes its directory",
        ),
        (
            r#""byte_length": 4096"#,
            r#""byte_length": 33554433"#,
            "a definition past the reader's file ceiling",
        ),
        (
            r#""byte_length": 512"#,
            r#""byte_length": 65537"#,
            "a configuration past the reader's smaller ceiling",
        ),
        (
            r#""size": 4096"#,
            r#""size": 2000000"#,
            "a vocabulary past the reader's ceiling",
        ),
        (
            r#""base_vocabulary_size": 512"#,
            r#""base_vocabulary_size": 8192"#,
            "a base vocabulary larger than the vocabulary",
        ),
        (
            r#""model_max_length": 512"#,
            r#""model_max_length": 4194305"#,
            "a context past what the reader will read",
        ),
        // Policy values the reader would check and then not apply.
        (
            r#""pretokenizer_byte_level": {"add_prefix_space": false"#,
            r#""pretokenizer_byte_level": {"add_prefix_space": true"#,
            "a pretokenizer prefix space the encoder never adds",
        ),
        (
            r#""trim_offsets": true, "use_regex": false}"#,
            r#""trim_offsets": true, "use_regex": true}"#,
            "a byte-level regex split the encoder never applies",
        ),
        (
            r#""continuing_subword_prefix": null"#,
            r###""continuing_subword_prefix": "##""###,
            "a continuing-subword prefix the encoder never applies",
        ),
        (
            r#""end_of_word_suffix": null"#,
            r#""end_of_word_suffix": "</w>""#,
            "an end-of-word suffix the encoder never applies",
        ),
        (
            r#""ignore_merges": "absent""#,
            r#""ignore_merges": "whole-piece""#,
            "whole-piece lookup over literal-prefix base tokens",
        ),
        (
            r#""byte_start_id": 44"#,
            r#""byte_start_id": 300"#,
            "byte tokens that run past the base vocabulary",
        ),
        (
            r#""first_id": 300"#,
            r#""first_id": 301"#,
            "sequential merge results that run past the base vocabulary",
        ),
        (
            r#""merge_count": 212"#,
            r#""merge_count": 211"#,
            "merges that do not end at the base vocabulary",
        ),
        (
            r#""prompt_special_tokens": ["[BOS]", "[EOS]", "[U]", "[A]", "<t>", "</t>"]"#,
            r#""prompt_special_tokens": []"#,
            "no trusted prompt token at all",
        ),
    ] {
        let declaration = swapped(KEY_VALUE, from, to);
        let error =
            FrontendDescriptorV1::from_component_bytes(declaration.as_bytes()).expect_err(why);
        assert!(
            matches!(error, FrontendComponentErrorV1::Invalid(_)),
            "{why}: {error}"
        );
        // The refusal is the reader's own, reached before a bundle is served.
        assert!(
            serde_json::from_str::<FrontendDescriptorV1>(&declaration).is_ok(),
            "{why}: the case must differ only in a rule the reader owns"
        );
    }

    // Each policy combination the fixtures declare builds. The paired-merge
    // policy shares no member value with the literal-prefix policy it replaces
    // except the byte-level settings.
    let mut paired: Value = serde_json::from_str(FUNCTION_PARAMETERS).unwrap();
    paired["tokenizer"]["policy"] = serde_json::json!({
        "normalizer": "null",
        "pretokenizer_byte_level": {"add_prefix_space": false, "trim_offsets": true, "use_regex": false},
        "post_processor_byte_level": {"add_prefix_space": true, "trim_offsets": false, "use_regex": true},
        "decoder_byte_level": {"add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "merge_results": {"kind": "vocabulary"},
        "merge_records": "string-pair-array",
        "base_tokens": {"kind": "byte-level"},
        "config_tokens": {"kind": "string"},
        "add_bos_token": {"kind": "absent"},
        "bos_token": {"kind": "absent"},
        "unk_token": {"kind": "absent"},
        "tokenizer_class": "TokenizersBackend",
        "continuing_subword_prefix": null,
        "end_of_word_suffix": null,
        "ignore_merges": "whole-piece",
        "add_eos_token": null,
        "legacy": null,
        "empty_sentencepiece_kwargs": false
    });
    let paired = paired.to_string();
    for (name, declaration) in [
        ("byte-merge", KEY_VALUE),
        ("nfc", NARROW),
        ("function-parameters", FUNCTION_PARAMETERS),
        ("byte-level-paired-merges", paired.as_str()),
    ] {
        let descriptor = FrontendDescriptorV1::from_component_bytes(declaration.as_bytes())
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let contract =
            tokenizer_contract_v1(&descriptor).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(
            contract.vocabulary_size(),
            descriptor
                .tokenizer
                .vocabulary_size
                .unwrap_or(descriptor.vocabulary.size)
        );
        assert_eq!(
            contract.prompt_special_tokens(),
            descriptor.vocabulary.prompt_special_tokens
        );
    }

    let split = swapped(
        KEY_VALUE,
        r#""base_vocabulary_size": 512,"#,
        r#""base_vocabulary_size": 512,
    "vocabulary_size": 1024,"#,
    );
    let descriptor = FrontendDescriptorV1::from_component_bytes(split.as_bytes())
        .expect("a tokenizer width below the model logits width is valid");
    assert_eq!(descriptor.vocabulary.size, 4_096);
    assert_eq!(
        tokenizer_contract_v1(&descriptor)
            .expect("the split tokenizer contract builds")
            .vocabulary_size(),
        1_024
    );

    let too_wide = swapped(
        &split,
        r#""vocabulary_size": 1024"#,
        r#""vocabulary_size": 4097"#,
    );
    let error = FrontendDescriptorV1::from_component_bytes(too_wide.as_bytes())
        .expect_err("a tokenizer width above the model logits width is refused");
    assert!(
        error.to_string().contains("exceeds model logits width"),
        "{error}"
    );
}

/// A deployment may serve less context than its tokenizer reads, never more.
#[test]
fn every_tokenizer_policy_choice_refuses_unknown_fields() {
    // One policy that names every choice, including each one with no data.
    let policy = serde_json::json!({
        "normalizer": "null",
        "pretokenizer_byte_level": {"add_prefix_space": false, "trim_offsets": true, "use_regex": false},
        "post_processor_byte_level": {"add_prefix_space": true, "trim_offsets": false, "use_regex": true},
        "decoder_byte_level": {"add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "merge_results": {"kind": "vocabulary"},
        "merge_records": "string-pair-array",
        "base_tokens": {"kind": "byte-level"},
        "config_tokens": {"kind": "string"},
        "add_bos_token": {"kind": "absent"},
        "bos_token": {"kind": "required", "value": {"kind": "contract-value"}},
        "unk_token": {"kind": "required-null"},
        "tokenizer_class": "AnyTokenizer",
        "continuing_subword_prefix": null,
        "end_of_word_suffix": null,
        "ignore_merges": "whole-piece",
        "add_eos_token": null,
        "legacy": null,
        "empty_sentencepiece_kwargs": false
    });
    let parsed = serde_json::from_value::<FrontendTokenizerPolicyV1>(policy.clone())
        .expect("the policy without extras parses");
    // The choices with no data keep their persisted shape: only the tag.
    assert_eq!(serde_json::to_value(&parsed).unwrap(), policy);
    let mut accepted = Vec::new();
    for (path, extra) in [
        (&["merge_results"][..], "first_id"),
        (&["merge_results"][..], "unknown"),
        (&["base_tokens"][..], "byte_start_id"),
        (&["config_tokens"][..], "normalized"),
        (&["add_bos_token"][..], "value"),
        (&["unk_token"][..], "value"),
        (&["bos_token", "value"][..], "text"),
        (&["pretokenizer_byte_level"][..], "unknown"),
    ] {
        let mut changed = policy.clone();
        let mut target = &mut changed;
        for key in path {
            target = &mut target[*key];
        }
        target[extra] = serde_json::json!(0);
        if serde_json::from_value::<FrontendTokenizerPolicyV1>(changed).is_ok() {
            accepted.push(format!("{} with {extra}", path.join(".")));
        }
    }
    assert!(accepted.is_empty(), "unknown fields accepted: {accepted:?}");
}

#[test]
fn a_declared_context_must_fit_the_tokenizer_that_reads_it() {
    let declared = frontend(KEY_VALUE);
    assert_eq!(declared.maximum_total_tokens(), 512);
    assert_eq!(declared.descriptor().tokenizer.model_max_length, 512);

    // Below the tokenizer's own limit is what a deployment holding back looks
    // like, and it reads.
    let lower = FrontendDescriptorV1::from_component_bytes(
        swapped(
            KEY_VALUE,
            r#""maximum_total_tokens": 512"#,
            r#""maximum_total_tokens": 256"#,
        )
        .as_bytes(),
    )
    .expect("a context under the tokenizer's limit reads");
    assert_eq!(lower.context.maximum_total_tokens, 256);

    // Above it is a context this deployment would advertise and its own
    // tokenizer would refuse the moment a client filled it.
    let error = FrontendDescriptorV1::from_component_bytes(
        swapped(
            KEY_VALUE,
            r#""maximum_total_tokens": 512"#,
            r#""maximum_total_tokens": 513"#,
        )
        .as_bytes(),
    )
    .expect_err("a context past the tokenizer's limit must be refused");
    assert!(
        matches!(error, FrontendComponentErrorV1::Invalid(_)),
        "{error}"
    );
}

/// Chat logprobs cover every generated token like vLLM's with a reasoning
/// parser: the reasoning markers, the reasoning, the content and the hidden
/// stop token. A stream carries each token's row on the chunk its step
/// produced and loses the rows of steps that produced none.
#[tokio::test]
async fn chat_logprobs_cover_every_generated_token() {
    let events = vec![
        logprob("<t>", -0.01, &[]),
        logprob("plan", -0.2, &[("idea", -1.8)]),
        GenerationEvent::Reasoning(ReasoningContent {
            delta: "plan".to_owned(),
            generated_token_count: 2,
        }),
        logprob("</t>", -0.03, &[]),
        // Sampled below the requested two: the position has three entries.
        logprob("ok", -2.0, &[("yes", -0.1), ("sure", -0.5)]),
        GenerationEvent::Content(VisibleContent {
            delta: "ok".to_owned(),
            generated_token_count: 2,
        }),
        logprob("[EOS]", -12_000.0, &[]),
        GenerationEvent::TokenUsageDelta {
            generated_token_count: 1,
        },
        finished_after(5),
    ];
    let server = TestServerV1::start(ScriptedBackendV1::new(events, 1), KEY_VALUE).await;
    let ask = |extra: &str| {
        post(
            "/v1/chat/completions",
            &format!(
                r#"{{"model":"alpha","messages":[{{"role":"user","content":"x"}}],"max_tokens":8,"reasoning_effort":"high","logprobs":true,"top_logprobs":2{extra}}}"#
            ),
        )
    };
    let labels = |rows: &Value| {
        rows.as_array()
            .expect("logprob rows are an array")
            .iter()
            .map(|row| row["token"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>()
    };

    let response = server.request(&ask("")).await;
    assert_eq!(response.status, 200, "{}", response.body);
    let body = response.json();
    let choice = &body["choices"][0];
    // The declaration names the reasoning field.
    assert!(
        choice["message"]
            .as_object()
            .unwrap()
            .values()
            .any(|value| value == "plan")
    );
    assert_eq!(choice["message"]["content"], "ok");
    let rows = &choice["logprobs"]["content"];
    assert_eq!(labels(rows), ["<t>", "plan", "</t>", "ok", "[EOS]"]);
    assert_eq!(rows[1]["top_logprobs"].as_array().unwrap().len(), 2);
    assert_eq!(labels(&rows[3]["top_logprobs"]), ["ok", "yes"]);
    assert_eq!(rows[3]["bytes"], serde_json::json!(b"ok"));
    assert_eq!(rows[4]["logprob"], -9999.0);
    assert_eq!(body["usage"]["completion_tokens"], 5);

    let hidden = server.request(&ask(r#","include_reasoning":false"#)).await;
    assert_eq!(hidden.status, 200, "{}", hidden.body);
    assert!(hidden.json()["choices"][0]["logprobs"].is_null());

    let streamed = server.request(&ask(r#","stream":true"#)).await;
    assert_eq!(streamed.status, 200, "{}", streamed.body);
    let chunks = stream_chunks(&streamed.body);
    let carried = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["logprobs"]["content"].as_array())
        .flat_map(|rows| labels(&Value::from(rows.clone())))
        .collect::<Vec<_>>();
    assert_eq!(carried, ["plan", "ok", "[EOS]"]);
    let terminal = chunks
        .iter()
        .find(|chunk| chunk["choices"][0]["finish_reason"] == "stop")
        .expect("the stream finishes");
    assert_eq!(
        labels(&terminal["choices"][0]["logprobs"]["content"]),
        ["[EOS]"]
    );

    let hidden = server
        .request(&ask(r#","stream":true,"include_reasoning":false"#))
        .await;
    assert_eq!(hidden.status, 200, "{}", hidden.body);
    assert!(
        stream_chunks(&hidden.body)
            .iter()
            .all(|chunk| chunk["choices"][0]["logprobs"].is_null())
    );
}

/// Completion logprobs include the stop token, list every entry of the
/// position and place offsets by label length. A stream sends every step,
/// with empty text when the step's text is still held back.
#[tokio::test]
async fn completion_logprobs_cover_every_generated_token() {
    let events = vec![
        logprob("Hi", -0.1, &[("Hello", -2.5)]),
        GenerationEvent::Content(VisibleContent {
            delta: "Hi".to_owned(),
            generated_token_count: 1,
        }),
        logprob(" there", -0.2, &[("!", -1.7)]),
        logprob("!", -3.0, &[(".", -0.05)]),
        GenerationEvent::Content(VisibleContent {
            delta: " there!".to_owned(),
            generated_token_count: 2,
        }),
        logprob("[EOS]", -0.4, &[("?", -1.1)]),
        GenerationEvent::TokenUsageDelta {
            generated_token_count: 1,
        },
        finished_after(4),
    ];
    let server = TestServerV1::start(ScriptedBackendV1::new(events, 1), KEY_VALUE).await;
    let ask = |extra: &str| {
        post(
            "/v1/completions",
            &format!(r#"{{"model":"alpha","prompt":"x","max_tokens":8,"logprobs":1{extra}}}"#),
        )
    };

    let response = server.request(&ask("")).await;
    assert_eq!(response.status, 200, "{}", response.body);
    let body = response.json();
    let logprobs = &body["choices"][0]["logprobs"];
    assert_eq!(body["choices"][0]["text"], "Hi there!");
    assert_eq!(
        logprobs["tokens"],
        serde_json::json!(["Hi", " there", "!", "[EOS]"])
    );
    assert_eq!(logprobs["text_offset"], serde_json::json!([0, 2, 8, 9]));
    assert_eq!(
        logprobs["top_logprobs"][2],
        serde_json::json!({"!": -3.0, ".": -0.05})
    );

    let streamed = server.request(&ask(r#","stream":true"#)).await;
    assert_eq!(streamed.status, 200, "{}", streamed.body);
    let steps = stream_chunks(&streamed.body)
        .iter()
        .filter(|chunk| !chunk["choices"][0]["logprobs"].is_null())
        .map(|chunk| {
            let choice = &chunk["choices"][0];
            (
                choice["text"].as_str().unwrap().to_owned(),
                choice["logprobs"]["tokens"].clone(),
                choice["logprobs"]["text_offset"].clone(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        steps,
        [
            (
                "Hi".to_owned(),
                serde_json::json!(["Hi"]),
                serde_json::json!([0])
            ),
            (
                String::new(),
                serde_json::json!([" there"]),
                serde_json::json!([2])
            ),
            (
                " there!".to_owned(),
                serde_json::json!(["!"]),
                serde_json::json!([2])
            ),
            (
                String::new(),
                serde_json::json!(["[EOS]"]),
                serde_json::json!([9])
            ),
        ]
    );
}

#[tokio::test]
async fn sampling_defaults_follow_request_server_model_precedence() {
    for (declaration, top_k) in [(KEY_VALUE, 0), (NARROW, 20)] {
        let mut declaration: Value = serde_json::from_str(declaration).unwrap();
        declaration["generation_config"] = serde_json::json!({
            "temperature":1.0, "top_p":0.95, "top_k":top_k, "min_p":0.05, "repetition_penalty":1.1
        });
        for override_defaults in [false, true] {
            let backend = ScriptedBackendV1::new(vec![content("ok"), finished()], 1);
            let config = if override_defaults {
                ServerConfig::default()
                    .with_sampling_overrides(runtime_server::SamplingDefaults {
                        temperature: Some(0.7),
                        top_p: Some(0.8),
                        top_k: Some(9),
                        ..Default::default()
                    })
                    .unwrap()
            } else {
                ServerConfig::default()
            };
            let server =
                TestServerV1::with_config(backend.clone(), &declaration.to_string(), config).await;
            for (endpoint, input) in [
                ("/v1/completions", r#""prompt":"hello""#),
                (
                    "/v1/chat/completions",
                    r#""messages":[{"role":"user","content":"hello"}]"#,
                ),
            ] {
                for fields in [
                    "",
                    r#", "temperature":null,"top_p":null,"top_k":null,"min_p":null,"repetition_penalty":null"#,
                    r#", "temperature":0,"top_p":1,"top_k":0,"min_p":0,"repetition_penalty":1"#,
                    r#", "temperature":0,"top_p":1,"top_k":-1,"min_p":0,"repetition_penalty":1"#,
                ] {
                    let response = server
                        .request(&post(
                            endpoint,
                            &format!(r#"{{{input},"max_tokens":1{fields}}}"#),
                        ))
                        .await;
                    assert_eq!(response.status, 200, "{}", response.body);
                    let requests = backend.prompts.lock().unwrap();
                    let p = &requests.last().unwrap().sampling;
                    let explicit = fields.contains("\"temperature\":0");
                    assert_eq!(
                        p.temperature(),
                        if explicit {
                            0.0
                        } else if override_defaults {
                            0.7
                        } else {
                            1.0
                        }
                    );
                    assert_eq!(
                        p.top_p(),
                        if explicit {
                            1.0
                        } else if override_defaults {
                            0.8
                        } else {
                            0.95
                        }
                    );
                    assert_eq!(
                        p.top_k(),
                        if explicit {
                            0
                        } else if override_defaults {
                            9
                        } else {
                            top_k
                        }
                    );
                    assert_eq!(p.min_p(), if explicit { 0.0 } else { 0.05 });
                    assert_eq!(p.repetition_penalty(), if explicit { 1.0 } else { 1.1 });
                }
            }
        }
    }
}

#[test]
fn invalid_generation_defaults_are_refused_before_serving() {
    for defaults in [
        serde_json::json!({"top_k":-2}),
        serde_json::json!({"temperature":-0.1}),
        serde_json::json!({"top_p":0}),
        serde_json::json!({"min_p":2}),
        serde_json::json!({"repetition_penalty":0}),
        serde_json::json!({"unknown":1}),
    ] {
        let mut declaration: Value = serde_json::from_str(KEY_VALUE).unwrap();
        declaration["generation_config"] = defaults;
        let export = runtime_frontend::test_support::fixture_export_v1();
        assert!(
            BundleFrontendV1::from_export(&export, declaration.to_string().as_bytes()).is_err()
        );
    }
    assert!(
        ServerConfig::default()
            .with_sampling_overrides(runtime_server::SamplingDefaults {
                temperature: Some(f64::NAN),
                ..Default::default()
            })
            .is_err()
    );
}

#[tokio::test]
async fn masked_logprob_tails_serialize_as_the_reference_sentinel() {
    for (endpoint, input) in [
        ("/v1/completions", r#""prompt":"hello","logprobs":3"#),
        (
            "/v1/chat/completions",
            r#""messages":[{"role":"user","content":"hello"}],"logprobs":true,"top_logprobs":3"#,
        ),
    ] {
        let backend = ScriptedBackendV1::new(
            vec![
                content("a"),
                logprob(
                    "a",
                    0.0,
                    &[("b", f64::NEG_INFINITY), ("c", f64::NEG_INFINITY)],
                ),
                finished(),
            ],
            1,
        );
        let server = TestServerV1::start(backend, KEY_VALUE).await;
        let response = server
            .request(&post(endpoint, &format!(r#"{{{input},"max_tokens":1}}"#)))
            .await;
        assert_eq!(response.status, 200, "{}", response.body);
        let response = response.json();
        if endpoint.contains("chat") {
            let entries = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                .as_array()
                .unwrap();
            assert_eq!(entries.len(), 3);
            assert_eq!(entries[0]["logprob"], 0.0);
            assert_eq!(entries[1]["logprob"], -9999.0);
            assert_eq!(entries[2]["logprob"], -9999.0);
        } else {
            let entries = &response["choices"][0]["logprobs"]["top_logprobs"][0];
            assert_eq!(entries.as_object().unwrap().len(), 3);
            assert_eq!(entries["a"], 0.0);
            assert_eq!(entries["b"], -9999.0);
            assert_eq!(entries["c"], -9999.0);
        }
    }
}

/// `clear_thinking` reaches a frontend as the client sent it: the transport
/// does not read the keyword, so a format whose template does not read it
/// renders any value as it renders none. A format that reads it refuses
/// anything but a boolean, which the owning model's tests pin.
#[tokio::test]
async fn a_format_that_does_not_read_clear_thinking_ignores_any_value() {
    let conversation = r#"[{"role":"user","content":"q1"},
        {"role":"assistant","content":"a1","reasoning":"r1"},
        {"role":"user","content":"q2"}]"#;
    let mut prompts = Vec::new();
    for keywords in [
        "{}",
        r#"{"clear_thinking":1}"#,
        r#"{"clear_thinking":"yes"}"#,
        r#"{"clear_thinking":{"x":[1]}}"#,
        r#"{"clear_thinking":true}"#,
        r#"{"clear_thinking":false}"#,
        r#"{"clear_thinking":null}"#,
    ] {
        let backend = ScriptedBackendV1::new(vec![content("ok"), finished()], 1);
        let server = TestServerV1::start(Arc::clone(&backend) as Arc<_>, KEY_VALUE).await;
        let response = server
            .request(&post(
                "/v1/chat/completions",
                &format!(
                    r#"{{"model":"alpha","messages":{conversation},"max_completion_tokens":4,"chat_template_kwargs":{keywords}}}"#
                ),
            ))
            .await;
        assert_eq!(response.status, 200, "{keywords}: {}", response.body);
        prompts.push(backend.rendered_prompt());
    }
    assert!(
        prompts.iter().all(|prompt| *prompt == prompts[0]),
        "{prompts:?}"
    );
}
