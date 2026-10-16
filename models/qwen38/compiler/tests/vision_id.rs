//! Real image preparation through the same HTTP parser for tokenization and generation.
//! Frozen image bytes are supplied outside the tree and authenticated by the capture.
use std::path::Path;
use std::sync::{Arc, Mutex};

use runtime_engine::{
    Engine, EngineConfig, ExecutorError, ModelBatchItem, ModelExecutor, ModelTransaction,
    NoopModelTransactionState,
};
use runtime_frontend::BundleFrontendV1;
use runtime_host::{EngineBackendConfig, EngineGenerationBackend, EngineModelPolicyV1};
use runtime_kv::{KvGeometry, PreparedAppendBatch};
use runtime_model_api::ModelVocabulary;
use runtime_scheduler::SchedulerConfig;
use runtime_server::{
    AdmissionError, BackendReadiness, BoxFuture, EventSender, GenerationBackend, PreparedPrompt,
    PreparedRequest, ServerConfig, ServerFrontendV1, serve,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::net::{TcpListener, TcpStream};

struct NoExecution;
impl ModelExecutor for NoExecution {
    type TransactionState<'a> = NoopModelTransactionState;
    fn execute<'a>(
        &'a mut self,
        _: &[ModelBatchItem],
        _: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        panic!("frontend regression must not execute model math");
    }
}

struct Backend {
    production: EngineGenerationBackend,
    prepared: Mutex<Vec<PreparedPrompt>>,
    submitted: Mutex<Vec<PreparedPrompt>>,
}
impl GenerationBackend for Backend {
    fn readiness(&self) -> BackendReadiness {
        self.production.readiness()
    }
    fn maximum_sequence_tokens(&self) -> usize {
        self.production.maximum_sequence_tokens()
    }
    fn prompt_token_count(&self, prompt: &PreparedPrompt) -> Result<usize, AdmissionError> {
        self.production.prompt_token_count(prompt)
    }
    fn prepare_prompt(
        &self,
        prompt: PreparedPrompt,
    ) -> Result<(PreparedPrompt, usize), AdmissionError> {
        let result = self.production.prepare_prompt(prompt)?;
        self.prepared.lock().unwrap().push(result.0.clone());
        Ok(result)
    }
    fn submit(
        &self,
        request: PreparedRequest,
        _: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        self.submitted.lock().unwrap().push(request.prompt);
        Box::pin(async { Err(AdmissionError::Rejected("captured prepared prompt".into())) })
    }
}

struct Server {
    address: std::net::SocketAddr,
    task: tokio::task::JoinHandle<()>,
    backend: Arc<Backend>,
    model: String,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Server {
    async fn start() -> Option<Self> {
        let Ok(root) = std::env::var("CHOPINFER_QWEN38_TOKENIZER_DIR")
            .or_else(|_| std::env::var("CHOPINFER_QWEN38_CHECKPOINT"))
        else {
            eprintln!("SKIP: checkpoint tokenizer is required for real media replay");
            return None;
        };
        let declaration = model_qwen38::frontend::multimodal_descriptor();
        let frontend = Arc::new(
            BundleFrontendV1::from_export(
                &model_qwen38_contract::frontend::export(),
                &model_qwen38::frontend::multimodal_component(),
            )
            .unwrap(),
        );
        let model = frontend.models()[0].id.clone();
        let engine = Engine::new(
            EngineConfig::new(
                SchedulerConfig::new(1, 1, 65536, 65536, 1).unwrap(),
                KvGeometry::new(4096, 64).unwrap(),
                ModelVocabulary::new(declaration.vocabulary.size as usize).unwrap(),
                262144,
                1,
            )
            .unwrap(),
            NoExecution,
        )
        .unwrap();
        let backend = Arc::new(Backend {
            production: EngineGenerationBackend::new(
                model_qwen38::tokenizer::load_tokenizer(Path::new(&root)).unwrap(),
                engine,
                EngineBackendConfig::default(),
                EngineModelPolicyV1::from_declaration(&declaration).with_media_inputs(true),
            )
            .unwrap(),
            prepared: Mutex::new(vec![]),
            submitted: Mutex::new(vec![]),
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let serving = backend.clone();
        let task = tokio::spawn(async move {
            serve(listener, serving, frontend, ServerConfig::default())
                .await
                .unwrap();
        });
        Some(Self {
            address,
            task,
            backend,
            model,
        })
    }
    async fn post(&self, path: &str, body: &Value) -> (u16, Value) {
        let body = body.to_string();
        let stream = TcpStream::connect(self.address).await.unwrap();
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let mut request = request.as_bytes();
        while !request.is_empty() {
            stream.writable().await.unwrap();
            match stream.try_write(request) {
                Ok(n) => request = &request[n..],
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(e) => panic!("{e}"),
            }
        }
        let mut bytes = vec![];
        loop {
            stream.readable().await.unwrap();
            let mut buffer = [0; 65536];
            match stream.try_read(&mut buffer) {
                Ok(0) => break,
                Ok(n) => bytes.extend_from_slice(&buffer[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(e) => panic!("{e}"),
            }
        }
        let text = String::from_utf8(bytes).unwrap();
        let (head, body) = text.split_once("\r\n\r\n").unwrap();
        (
            head.split_whitespace().nth(1).unwrap().parse().unwrap(),
            serde_json::from_str(body).unwrap(),
        )
    }
    async fn prepared(&self, body: &Value) -> (Value, PreparedPrompt) {
        self.backend.prepared.lock().unwrap().clear();
        self.backend.submitted.lock().unwrap().clear();
        let (status, tokens) = self.post("/tokenize", body).await;
        assert_eq!(status, 200, "{tokens}");
        let tokenized = self.backend.prepared.lock().unwrap().pop().unwrap();
        let mut generate = body.clone();
        generate["max_tokens"] = json!(1);
        let (_, response) = self.post("/v1/chat/completions", &generate).await;
        assert!(
            response["error"]["message"]
                .as_str()
                .unwrap()
                .contains("captured prepared prompt"),
            "{response}"
        );
        let submitted = self.backend.submitted.lock().unwrap().pop().unwrap();
        assert_eq!(tokenized, submitted, "tokenize and generation preparation");
        let PreparedPrompt::MediaTokens { tokens: ids, .. } = &submitted else {
            panic!("media tokens");
        };
        assert_eq!(tokens["tokens"], json!(ids));
        (tokens["tokens"].clone(), submitted)
    }
}

fn cases() -> Value {
    serde_json::from_str(include_str!(
        "../../../../fixtures/qwen38/frontend/chat-renders.json"
    ))
    .unwrap()
}
fn images() -> Option<Value> {
    let Ok(path) = std::env::var("CHOPINFER_QWEN38_VISION_IMAGES") else {
        eprintln!("SKIP: set CHOPINFER_QWEN38_VISION_IMAGES to the captured PNG data URL map");
        return None;
    };
    let images: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    for (hash, source) in images.as_object().unwrap() {
        let data = source
            .as_str()
            .unwrap()
            .strip_prefix("data:image/png;base64,")
            .unwrap();
        let bytes = runtime_media::decode_base64(data, 67108864).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(bytes)), *hash);
    }
    Some(images)
}
fn request(case: &Value, images: &Value, model: &str) -> Value {
    let mut body = json!({"model":model,"messages":case["messages"],"add_generation_prompt":true});
    for key in ["tools", "chat_template_kwargs"] {
        if let Some(value) = case.get(key) {
            body[key] = value.clone();
        }
    }
    for message in body["messages"].as_array_mut().unwrap() {
        if let Some(parts) = message["content"].as_array_mut() {
            for part in parts {
                if let Some(hash) = part["image_url"]["url"]
                    .as_str()
                    .and_then(|s| s.strip_prefix("sha256:"))
                {
                    part["image_url"]["url"] = images[hash].clone();
                }
            }
        }
    }
    body
}

#[tokio::test]
async fn real_vision_id_row70_matches_frozen_tokens_and_generation() {
    let Some(images) = images() else {
        return;
    };
    let Some(server) = Server::start().await else {
        return;
    };
    let cases = cases();
    let case = &cases[70];
    assert_eq!(case["id"], "image-vision-id");
    let body = request(case, &images, &server.model);
    let (tokens, numbered) = server.prepared(&body).await;
    assert_eq!(tokens.as_array().unwrap().len(), 292);
    assert_eq!(tokens, case["token_ids"]);
    let mut default_body = body.clone();
    default_body
        .as_object_mut()
        .unwrap()
        .remove("chat_template_kwargs");
    let (plain_tokens, plain) = server.prepared(&default_body).await;
    let expected_text = case["rendered_text"]
        .as_str()
        .unwrap()
        .replace("Picture 1: ", "")
        .replace("Picture 2: ", "");
    let (PreparedPrompt::TokenIds(expected), _) = server
        .backend
        .production
        .prepare_prompt(PreparedPrompt::RenderedChat(expected_text))
        .unwrap()
    else {
        panic!("reference text tokens");
    };
    assert_eq!(plain_tokens, json!(expected));
    assert_eq!(expected.len(), 282);
    let PreparedPrompt::MediaTokens {
        media,
        spans,
        positions,
        ..
    } = &plain
    else {
        panic!("plain media");
    };
    let PreparedPrompt::MediaTokens {
        media: numbered_media,
        spans: numbered_spans,
        positions: numbered_positions,
        ..
    } = &numbered
    else {
        panic!("numbered media");
    };
    assert_eq!(
        media, numbered_media,
        "payload bytes, hashes, recipe geometry and output rows"
    );
    assert_eq!(media.len(), 2);
    assert_eq!(spans.len(), 2);
    assert_eq!(positions.len(), 2);
    for i in 0..2 {
        let shift = 5 * (i as u32 + 1);
        let mut expected_span = spans[i];
        expected_span.prompt_row += shift;
        assert_eq!(numbered_spans[i], expected_span);
        let mut expected_position = positions[i];
        expected_position.prompt_row += shift;
        assert_eq!(numbered_positions[i], expected_position);
    }
    for option in [
        json!(false),
        Value::Null,
        json!(0),
        json!(""),
        json!([]),
        json!({}),
    ] {
        default_body["chat_template_kwargs"] = json!({"add_vision_id":option});
        let (ids, prepared) = server.prepared(&default_body).await;
        assert_eq!(ids, plain_tokens);
        assert_eq!(prepared, plain);
    }
    for option in [
        json!(true),
        json!(1),
        json!("false"),
        json!([false]),
        json!({"x":null}),
    ] {
        default_body["chat_template_kwargs"] = json!({"add_vision_id":option});
        let (ids, prepared) = server.prepared(&default_body).await;
        assert_eq!(ids, tokens);
        assert_eq!(prepared, numbered, "no request-local counter leakage");
    }
}

#[tokio::test]
async fn frozen_default_image_rows_and_media_refusals_stay_unchanged() {
    let Some(images) = images() else {
        return;
    };
    let Some(server) = Server::start().await else {
        return;
    };
    let cases = cases();
    let mut checked = 0;
    for case in cases.as_array().unwrap().iter().filter(|case| {
        case["id"].as_str().unwrap().starts_with("image-") && case["id"] != "image-vision-id"
    }) {
        let mut body = request(case, &images, &server.model);
        for option in [None, Some(json!(false)), Some(Value::Null)] {
            if let Some(value) = option {
                body["chat_template_kwargs"] = json!({"add_vision_id":value});
            }
            let (status, actual) = server.post("/tokenize", &body).await;
            assert_eq!(status, 200, "{}: {actual}", case["id"]);
            assert_eq!(actual["tokens"], case["token_ids"], "{}", case["id"]);
            server.backend.prepared.lock().unwrap().clear();
        }
        checked += 1;
    }
    assert_eq!(checked, 5);
    for kwargs in [
        json!({}),
        json!({"add_vision_id":false}),
        json!({"add_vision_id":true}),
    ] {
        let mut body = json!({"model":server.model,"chat_template_kwargs":kwargs,"messages":[{"role":"user","content":[{"type":"video_url","video_url":{"url":"data:video/mp4;base64,AAAA"}}]}]});
        for path in ["/tokenize", "/v1/chat/completions"] {
            let (status, error) = server.post(path, &body).await;
            assert_eq!(status, 400);
            assert_eq!(error["error"]["param"], "messages");
        }
        body["messages"] = request(&cases[70], &images, &server.model)["messages"].clone();
        body["messages"][0]["role"] = json!("system");
        assert_eq!(server.post("/tokenize", &body).await.0, 400);
    }
}
