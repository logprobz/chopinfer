//! Replays HF frontend goldens against the served Qwen frontend on CPU.
//!
//! The renders and tokenizations go through the real transport, request
//! parser and declared frontend with the checkpoint tokenizer as the backend,
//! using HF templates and tokenizers as the reference. Fixed API samples
//! exercise the declared decoder as serving-surface compatibility inputs.
//! Set `QWEN38_OFFLINE_CAPTURE_DIR` to also write the replayed renders and
//! tokenizations for `tools/qual/frontend_goldens.py`.

use std::net::Ipv4Addr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use model_qwen38::frontend::descriptor;
use model_qwen38::tokenizer::{PROMPT_TOKENS, load_tokenizer};
use runtime_frontend::test_support::documents::PairedDecoderV1;
use runtime_frontend::{CompletionUpdateV1, DecoderLimitsV1, GeneratedTokenV1, PromptModeV1};
use runtime_server::{
    AdmissionError, BackendReadiness, BoxFuture, EventSender, GenerationBackend, PreparedPrompt,
    PreparedRequest, ServerConfig, ServerFrontendV1, serve,
};
use runtime_tokenizer::Tokenizer;
use serde_json::{Value, json};
use tokio::net::{TcpListener, TcpStream};

/// This replay has no media capability and no explicit modality selection,
/// so its refusal describes the bundle without suggesting an unused flag.
const REFUSED_MEDIA: &str = "this bundle does not support media inputs";

fn fixture(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures/qwen38/frontend");
    serde_json::from_slice(&std::fs::read(path.join(name)).unwrap()).unwrap()
}

fn tokenizer() -> Option<Tokenizer> {
    let Ok(root) = std::env::var("CHOPINFER_QWEN38_TOKENIZER_DIR")
        .or_else(|_| std::env::var("CHOPINFER_QWEN38_CHECKPOINT"))
    else {
        eprintln!(
            "SKIP: set CHOPINFER_QWEN38_TOKENIZER_DIR or CHOPINFER_QWEN38_CHECKPOINT to replay the HF goldens"
        );
        return None;
    };
    Some(load_tokenizer(Path::new(&root)).unwrap())
}

/// A backend that only tokenizes, the way the served host prepares prompts.
struct TokenizerBackend(Tokenizer);

impl TokenizerBackend {
    fn encode(&self, text: &str) -> Result<Vec<u32>, AdmissionError> {
        self.0
            .encode(text, PROMPT_TOKENS)
            .map_err(|_| AdmissionError::InvalidPrompt)
    }
}

impl GenerationBackend for TokenizerBackend {
    fn readiness(&self) -> BackendReadiness {
        BackendReadiness::Ready
    }

    fn maximum_sequence_tokens(&self) -> usize {
        262_144
    }

    fn prompt_token_count(&self, prompt: &PreparedPrompt) -> Result<usize, AdmissionError> {
        Ok(self.prepare_prompt(prompt.clone())?.1)
    }

    fn prepare_prompt(
        &self,
        prompt: PreparedPrompt,
    ) -> Result<(PreparedPrompt, usize), AdmissionError> {
        let ids = match prompt {
            PreparedPrompt::Raw(text) | PreparedPrompt::RenderedChat(text) => self.encode(&text)?,
            PreparedPrompt::TokenIds(ids) => ids,
            _ => return Err(AdmissionError::InvalidPrompt),
        };
        let count = ids.len();
        Ok((PreparedPrompt::TokenIds(ids), count))
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, AdmissionError> {
        self.0
            .decode_utf8(tokens)
            .map_err(|_| AdmissionError::InvalidPrompt)
    }

    fn submit(
        &self,
        _request: PreparedRequest,
        _events: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        Box::pin(async { Err(AdmissionError::Rejected("tokenizer only".to_owned())) })
    }
}

struct Server(std::net::SocketAddr);

impl Server {
    async fn start(tokenizer: Tokenizer) -> Self {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = serve(
                listener,
                Arc::new(TokenizerBackend(tokenizer)),
                Arc::new(
                    runtime_frontend::BundleFrontendV1::from_export(
                        &model_qwen38_contract::frontend::export(),
                        model_qwen38::frontend::COMPONENT,
                    )
                    .unwrap(),
                ),
                ServerConfig::default(),
            )
            .await;
        });
        Self(address)
    }

    async fn post(&self, path: &str, body: &Value) -> (u16, Value) {
        let body = body.to_string();
        let stream = TcpStream::connect(self.0).await.unwrap();
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let mut request = request.as_bytes();
        while !request.is_empty() {
            stream.writable().await.unwrap();
            match stream.try_write(request) {
                Ok(written) => request = &request[written..],
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(error) => panic!("the socket write failed: {error}"),
            }
        }
        let mut response = Vec::new();
        loop {
            stream.readable().await.unwrap();
            let mut chunk = [0_u8; 65536];
            match stream.try_read(&mut chunk) {
                Ok(0) => break,
                Ok(read) => response.extend_from_slice(&chunk[..read]),
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(error) => panic!("the socket read failed: {error}"),
            }
        }
        let text = String::from_utf8(response).unwrap();
        let (head, body) = text.split_once("\r\n\r\n").unwrap();
        let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        (status, serde_json::from_str(body).unwrap())
    }
}

fn model() -> String {
    runtime_frontend::BundleFrontendV1::from_export(
        &model_qwen38_contract::frontend::export(),
        model_qwen38::frontend::COMPONENT,
    )
    .unwrap()
    .models()[0]
        .id
        .clone()
}

#[tokio::test]
async fn every_captured_render_and_tokenization_matches_fixture() {
    let Some(tokenizer) = tokenizer() else {
        return;
    };
    let server = Server::start(tokenizer).await;
    let mut renders = Vec::new();
    let cases = fixture("chat-renders.json");
    let cases = cases.as_array().unwrap();
    assert_eq!(cases.len(), 75);
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let mut body =
            json!({"model": model(), "messages": case["messages"], "add_generation_prompt": true});
        for key in ["tools", "chat_template_kwargs"] {
            if let Some(value) = case.get(key) {
                body[key] = value.clone();
            }
        }
        let (status, tokenized) = server.post("/tokenize", &body).await;
        let mut row = json!({"id": id, "status": status});
        if status == 200 {
            let (_, text) = server
                .post(
                    "/detokenize",
                    &json!({"model": model(), "tokens": tokenized["tokens"]}),
                )
                .await;
            row["token_ids"] = tokenized["tokens"].clone();
            row["rendered_text"] = text["prompt"].clone();
        } else {
            row["error"] = tokenized;
        }
        if id.starts_with("image-") {
            assert_eq!(status, 400, "{id}");
            assert_eq!(row["error"]["error"]["message"], REFUSED_MEDIA, "{id}");
            assert_eq!(row["error"]["error"]["param"], "messages", "{id}");
            assert!(
                !row["error"]["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("--modalities")
            );
        } else {
            assert_eq!(status, case["status"], "{id}: {row}");
            assert_eq!(row.get("token_ids"), case.get("token_ids"), "{id}");
            assert_eq!(row.get("rendered_text"), case.get("rendered_text"), "{id}");
        }
        renders.push(row);
    }

    let mut tokenizations = Vec::new();
    for golden in fixture("tokenizer-golden.json").as_array().unwrap() {
        let (status, tokenized) = server
            .post(
                "/tokenize",
                &json!({"model": model(), "prompt": golden["text"], "add_special_tokens": false}),
            )
            .await;
        assert_eq!(status, 200);
        assert_eq!(
            tokenized["tokens"], golden["token_ids"],
            "{}",
            golden["text"]
        );
        tokenizations.push(json!({"text": golden["text"], "token_ids": tokenized["tokens"]}));
    }
    if let Some(directory) = std::env::var_os("QWEN38_OFFLINE_CAPTURE_DIR") {
        let directory = PathBuf::from(directory);
        std::fs::create_dir_all(&directory).unwrap();
        for (name, rows) in [
            ("chat-renders.json", renders),
            ("tokenizer-golden.json", tokenizations),
        ] {
            std::fs::write(
                directory.join(name),
                serde_json::to_vec_pretty(&rows).unwrap(),
            )
            .unwrap();
        }
    }
}

/// The prompt of every captured API request counts the tokens HF counted.
#[tokio::test]
async fn every_captured_api_prompt_counts_the_hf_prompt_tokens() {
    let Some(tokenizer) = tokenizer() else {
        return;
    };
    let server = Server::start(tokenizer).await;
    let samples = fixture("api-compatibility.json");
    let prompts = fixture("api-prompts.json");
    let renders = fixture("chat-renders.json");
    let tools = renders
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["id"] == "toolcall-00")
        .unwrap()["tools"]
        .clone();
    let user = |text: &str| json!([{"role": "user", "content": text}]);
    let prime = user("Is 91 prime? Answer in one sentence.");
    let weather = user("What's the weather in Paris? Use the tool.");
    // The fixed API compatibility case sends the tool-call answer back with its reasoning.
    let answer = &samples["tool-call"]["body"]["choices"][0]["message"];
    let call = &answer["tool_calls"][0];
    let mut history = weather.as_array().unwrap().clone();
    history.push(
        json!({"role": "assistant", "content": "", "reasoning_content": answer["reasoning"],
                        "tool_calls": answer["tool_calls"]}),
    );
    history.push(json!({"role": "tool", "tool_call_id": call["id"],
                        "content": "{\"temperature\": 18, \"condition\": \"cloudy\"}"}));
    let requests = [
        ("chat-thinking", json!({"messages": prime})),
        (
            "chat-effort-low",
            json!({"messages": prime, "reasoning_effort": "low"}),
        ),
        (
            "chat-effort-medium",
            json!({"messages": prime, "reasoning_effort": "medium"}),
        ),
        (
            "chat-effort-xhigh",
            json!({"messages": prime, "reasoning_effort": "xhigh"}),
        ),
        (
            "chat-nothink",
            json!({"messages": prime, "chat_template_kwargs": {"enable_thinking": false}}),
        ),
        (
            "tool-call",
            json!({"messages": weather, "tools": tools, "tool_choice": "auto"}),
        ),
        ("tool-result", json!({"messages": history, "tools": tools})),
        (
            "completion-logprobs",
            json!({"prompt": "The capital of France is"}),
        ),
        (
            "completion-n2-seeded",
            json!({"prompt": "Once upon a time"}),
        ),
    ];
    for (name, mut body) in requests {
        body["model"] = json!(model());
        let (status, tokenized) = server.post("/tokenize", &body).await;
        assert_eq!(status, 200, "{name}: {tokenized}");
        assert_eq!(
            tokenized["count"],
            prompts
                .as_array()
                .unwrap()
                .iter()
                .find(|row| row["name"] == name)
                .unwrap()["prompt_tokens"],
            "{name}"
        );
    }
    let (status, refused) = server
        .post(
            "/tokenize",
            &json!({"model": model(), "messages": prime, "reasoning_effort": "high"}),
        )
        .await;
    assert_eq!(status, 400, "{refused}");
}

/// Content, reasoning, calls and streamed arguments one completion yields.
#[derive(Debug, Default, PartialEq)]
struct Answer {
    reasoning: String,
    content: String,
    calls: Vec<(String, String)>,
    streamed_arguments: Vec<String>,
}

fn decode(pieces: &[Vec<u8>], mode: PromptModeV1, stream_tool_calls: bool) -> Answer {
    let d = Arc::new(descriptor());
    let mut decoder = PairedDecoderV1::new(Arc::clone(&d), mode, DecoderLimitsV1::declared(&d));
    decoder.set_decoding_options(
        runtime_server::DecodingOptions {
            stream_tool_calls,
            ..Default::default()
        },
        false,
    );
    let mut answer = Answer::default();
    for (ordinal, bytes) in pieces.iter().enumerate() {
        for update in decoder
            .push(GeneratedTokenV1::new(ordinal as u64, 42, bytes.clone()))
            .unwrap()
        {
            match update {
                CompletionUpdateV1::Reasoning(part) => answer.reasoning.push_str(&part.text),
                CompletionUpdateV1::Content(part) => answer.content.push_str(&part.text),
                CompletionUpdateV1::ToolCalls(tools) => answer.calls.extend(
                    tools
                        .calls
                        .into_iter()
                        .map(|call| (call.name, call.arguments)),
                ),
                CompletionUpdateV1::ToolCallDeltas(deltas) => {
                    for delta in deltas {
                        if answer.streamed_arguments.len() <= delta.index {
                            answer
                                .streamed_arguments
                                .resize(delta.index + 1, String::new());
                        }
                        answer.streamed_arguments[delta.index].push_str(&delta.arguments);
                    }
                }
                CompletionUpdateV1::Finished(done) => {
                    if !stream_tool_calls && let Some(whole) = done.completion {
                        assert_eq!(whole.content, answer.content);
                    }
                }
            }
        }
    }
    answer
}

/// One completion cut into tokens when the tokenizer is present, else bytes.
fn pieces(tokenizer: Option<&Tokenizer>, text: &str) -> Vec<Vec<u8>> {
    match tokenizer {
        Some(tokenizer) => tokenizer
            .encode(text, PROMPT_TOKENS)
            .unwrap()
            .iter()
            .map(|id| tokenizer.decode_bytes(&[*id]).unwrap())
            .collect(),
        None => text.bytes().map(|byte| vec![byte]).collect(),
    }
}

#[test]
fn captured_tool_call_and_answer_completions_decode_to_its_message_shapes() {
    let tokenizer = tokenizer();
    let samples = fixture("api-compatibility.json");
    let message = &samples["tool-call"]["body"]["choices"][0]["message"];
    let reasoning = message["reasoning"].as_str().unwrap();
    let call = &message["tool_calls"][0]["function"];
    assert!(message["content"].is_null());
    let completion = format!(
        "{reasoning}</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call><|im_end|>"
    );
    let pieces_of_call = pieces(tokenizer.as_ref(), &completion);
    for stream_tool_calls in [false, true] {
        let answer = decode(&pieces_of_call, PromptModeV1::Thinking, stream_tool_calls);
        assert_eq!(answer.reasoning, reasoning);
        assert_eq!(answer.content, "", "a tool call carries no content");
        let expected = (
            call["name"].as_str().unwrap().to_owned(),
            call["arguments"].as_str().unwrap().to_owned(),
        );
        assert_eq!(answer.calls, std::slice::from_ref(&expected));
        if stream_tool_calls {
            assert_eq!(answer.streamed_arguments, [expected.1]);
        }
    }

    let message = &samples["chat-thinking"]["body"]["choices"][0]["message"];
    let (reasoning, content) = (
        message["reasoning"].as_str().unwrap(),
        message["content"].as_str().unwrap(),
    );
    let answer = decode(
        &pieces(
            tokenizer.as_ref(),
            &format!("{reasoning}</think>{content}<|im_end|>"),
        ),
        PromptModeV1::Thinking,
        false,
    );
    assert_eq!(
        (answer.reasoning.as_str(), answer.content.as_str()),
        (reasoning, content)
    );

    let content = samples["chat-nothink"]["body"]["choices"][0]["message"]["content"]
        .as_str()
        .unwrap();
    let answer = decode(
        &pieces(tokenizer.as_ref(), &format!("{content}<|im_end|>")),
        PromptModeV1::Chat,
        false,
    );
    assert_eq!(
        (answer.reasoning.as_str(), answer.content.as_str()),
        ("", content)
    );
}
