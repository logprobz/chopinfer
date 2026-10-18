//! The transport's behaviour against a frontend that is not a real model.
//!
//! Every model-shaped decision here comes from a fake: two invented model
//! identities, a rendering that just joins the messages, a context limit, a
//! tool-call bound, and a tool-call grammar. If the transport still serves
//! OpenAI correctly against that, it is not carrying a model of its own.

use std::any::Any;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use runtime_server::{
    AdmissionError, BackendReadiness, BackendToolCall, BoxFuture, CancellationSignal, ChatStateV1,
    EventSender, FinishReason, FrontendChatRequestV1, FrontendContentPartV1, FrontendErrorV1,
    FrontendModelV1, FrontendRenderedChatV1, FrontendRoleV1, FrontendToolChoiceV1,
    GenerationBackend, GenerationEvent, PreparedPrompt, PreparedRequest, ReasoningContent,
    ServerConfig, ServerFrontendV1, TokenUsage, VisibleContent, serve,
};
use serde_json::Value;
use tokio::net::{TcpListener, TcpStream};
use tokio::time::timeout;

const MODEL: &str = "fixture.model.v1";
const ALIAS: &str = "fixture-alias";
const MAXIMUM_TOTAL_TOKENS: usize = 64;
const MAXIMUM_TOOL_CALLS: usize = 2;

/// A chat state whose only fact is whether reasoning was asked for.
#[derive(Clone, Copy, Debug)]
struct FakeChatStateV1 {
    reasoning: bool,
}

impl ChatStateV1 for FakeChatStateV1 {
    fn reasoning_enabled(&self) -> bool {
        self.reasoning
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A frontend for a model that does not exist.
#[derive(Debug)]
struct FakeFrontendV1 {
    models: Vec<FrontendModelV1>,
    reasoning_field: &'static str,
    tool_call_reasoning_fallback: bool,
    reasoning_content_fallback: bool,
    reads_tool_objects: bool,
    /// Whether the prompt ends with the template keywords object received.
    echoes_template_kwargs: bool,
    extra_roles: Vec<String>,
    tool_references: bool,
    /// The separator of qualified function names; with one, the prompt
    /// ends with each offered tool's name and description.
    namespace_separator: Option<&'static str>,
}

impl FakeFrontendV1 {
    fn new() -> Self {
        Self::with_reasoning_field("reasoning_content")
    }

    fn with_reasoning_field(reasoning_field: &'static str) -> Self {
        Self {
            models: vec![
                FrontendModelV1::new(MODEL, "fixture-owner"),
                FrontendModelV1::new(ALIAS, "fixture-alias-owner"),
            ],
            reasoning_field,
            tool_call_reasoning_fallback: false,
            reasoning_content_fallback: false,
            reads_tool_objects: false,
            echoes_template_kwargs: false,
            extra_roles: Vec::new(),
            tool_references: false,
            namespace_separator: None,
        }
    }
}

impl ServerFrontendV1 for FakeFrontendV1 {
    fn reasoning_field(&self) -> &'static str {
        self.reasoning_field
    }

    fn tool_call_reasoning_fallback(&self) -> bool {
        self.tool_call_reasoning_fallback
    }

    fn reasoning_content_fallback(&self) -> bool {
        self.reasoning_content_fallback
    }

    fn reads_tool_objects(&self) -> bool {
        self.reads_tool_objects
    }

    fn extra_roles(&self) -> &[String] {
        &self.extra_roles
    }

    fn tool_references(&self) -> bool {
        self.tool_references
    }

    fn tool_namespace_separator(&self) -> Option<&str> {
        self.namespace_separator
    }

    fn models(&self) -> &[FrontendModelV1] {
        &self.models
    }

    fn maximum_total_tokens(&self) -> usize {
        MAXIMUM_TOTAL_TOKENS
    }

    fn maximum_tool_calls(&self) -> usize {
        MAXIMUM_TOOL_CALLS
    }

    fn render_chat(
        &self,
        request: FrontendChatRequestV1,
    ) -> Result<FrontendRenderedChatV1, FrontendErrorV1> {
        if request.reasoning_effort.as_deref() == Some("impossible") {
            return Err(FrontendErrorV1::new(
                "this model has no impossible effort",
                Some("reasoning_effort"),
            ));
        }
        let mut prompt = request
            .messages
            .iter()
            .map(|message| {
                let references: String = message
                    .parts
                    .iter()
                    .flatten()
                    .filter_map(|part| match part {
                        FrontendContentPartV1::ToolReference { name } => Some(format!("@{name}")),
                        _ => None,
                    })
                    .collect();
                format!(
                    "{}:{}{references}{}",
                    match &message.role {
                        FrontendRoleV1::System => "system",
                        FrontendRoleV1::Developer => "developer",
                        FrontendRoleV1::User => "user",
                        FrontendRoleV1::Assistant => "assistant",
                        FrontendRoleV1::Tool => "tool",
                        FrontendRoleV1::Declared(name) => name.as_str(),
                    },
                    message.content.as_deref().unwrap_or_default(),
                    message
                        .reasoning_content
                        .as_deref()
                        .map_or_else(String::new, |reasoning| format!("~{reasoning}"))
                )
            })
            .collect::<Vec<_>>()
            .join("|");
        if let Some(preserve) = request.preserve_thinking {
            prompt.push_str(&format!("|preserve_thinking:{preserve}"));
        }
        for object in &request.tool_objects {
            prompt.push_str(&format!("|tool:{}", object.to_json_value()));
        }
        if self.namespace_separator.is_some() {
            for tool in &request.tools {
                prompt.push_str(&format!(
                    "|offered:{}:{}",
                    tool.name,
                    tool.description.as_deref().unwrap_or_default()
                ));
            }
            for message in &request.messages {
                for call in &message.tool_calls {
                    prompt.push_str(&format!("|called:{}", call.name));
                }
            }
        }
        if let Some(clear) = request.clear_thinking {
            prompt.push_str(&format!("|clear_thinking:{}", clear.to_json_value()));
        }
        if self.echoes_template_kwargs {
            let kwargs = request.chat_template_kwargs.map_or_else(
                || "none".to_owned(),
                |kwargs| kwargs.to_json_value().to_string(),
            );
            prompt.push_str(&format!("|kwargs:{kwargs}"));
        }
        let allowed_function_names = match request.tool_choice {
            FrontendToolChoiceV1::None => Vec::new(),
            _ => request.tools.iter().map(|tool| tool.name.clone()).collect(),
        };
        Ok(FrontendRenderedChatV1 {
            prompt,
            segments: Vec::new(),
            media: Vec::new(),
            allowed_function_names,
            state: Arc::new(FakeChatStateV1 {
                reasoning: request.enable_thinking.or(request.thinking) == Some(true)
                    || request
                        .reasoning_effort
                        .is_some_and(|effort| effort != "none"),
            }),
        })
    }

    fn validate_tool_call(&self, name: &str, arguments: &str) -> Result<(), FrontendErrorV1> {
        if name.starts_with("fixture.") {
            Ok(())
        } else if arguments.is_empty() {
            Err(FrontendErrorV1::new("empty tool call arguments", None))
        } else {
            Err(FrontendErrorV1::new("unknown tool call name", None))
        }
    }
}

/// A backend that emits exactly the events a test hands it.
struct ScriptedBackendV1 {
    events: Vec<GenerationEvent>,
    prompt_tokens: usize,
    maximum_sequence_tokens: usize,
    prompts: std::sync::Mutex<Vec<PreparedRequest>>,
    submissions: AtomicUsize,
}

impl ScriptedBackendV1 {
    fn new(events: Vec<GenerationEvent>) -> Arc<Self> {
        Arc::new(Self {
            events,
            prompt_tokens: 4,
            maximum_sequence_tokens: MAXIMUM_TOTAL_TOKENS,
            prompts: std::sync::Mutex::new(Vec::new()),
            submissions: AtomicUsize::new(0),
        })
    }

    fn prompts(&self) -> Vec<PreparedRequest> {
        self.prompts.lock().expect("prompt log is usable").clone()
    }
}

impl GenerationBackend for ScriptedBackendV1 {
    fn readiness(&self) -> BackendReadiness {
        BackendReadiness::Ready
    }

    fn maximum_sequence_tokens(&self) -> usize {
        self.maximum_sequence_tokens
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
            .expect("prompt log is usable")
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
            prompt_tokens: 4,
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

struct TestServerV1 {
    address: SocketAddr,
}

impl TestServerV1 {
    async fn start(backend: Arc<dyn GenerationBackend>) -> Self {
        Self::start_with_frontend(backend, FakeFrontendV1::new()).await
    }

    async fn start_with_frontend(
        backend: Arc<dyn GenerationBackend>,
        frontend: FakeFrontendV1,
    ) -> Self {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("a loopback port is available");
        let address = listener.local_addr().expect("the listener has an address");
        tokio::spawn(async move {
            let _ = serve(
                listener,
                backend,
                Arc::new(frontend),
                ServerConfig::default(),
            )
            .await;
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

/// Writes one request and reads the whole response off a closing connection.
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

fn post(path: &str, body: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn get(path: &str) -> String {
    format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
}

/// A served deployment reports the identity its host named from `/health`,
/// with the model names the frontend declares.
#[tokio::test]
async fn a_served_deployment_reports_its_identity_from_health() {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .expect("a loopback port is available");
    let address = listener.local_addr().expect("the listener has an address");
    let options = runtime_server::HttpOptions {
        serving_identity: Some(runtime_server::ServingIdentity {
            bundle_id: "fixture.bundle.v1".into(),
            checkpoint_identity: Some("b".repeat(64)),
            startup_nonce: "e".repeat(32),
            version: "1.2.3".into(),
            source_commit: Some("d".repeat(40)),
            ..Default::default()
        }),
        ..Default::default()
    };
    tokio::spawn(runtime_server::serve_with_shutdown_and_http_options(
        listener,
        ScriptedBackendV1::new(Vec::new()),
        Arc::new(FakeFrontendV1::new()),
        ServerConfig::default(),
        Arc::new(runtime_server::SystemMonotonicClock::new()),
        runtime_server::metrics_snapshot_provider(|_| String::new()),
        std::future::pending(),
        Duration::from_secs(1),
        options,
    ));
    let server = TestServerV1 { address };

    let response = server.request(&get("/health")).await;

    assert_eq!(response.status, 200);
    let body = response.json();
    assert_eq!(body["status"], "ready");
    let serving = &body["serving"];
    assert_eq!(serving["bundle_id"], "fixture.bundle.v1");
    assert_eq!(serving["package_identity"], Value::Null);
    assert_eq!(serving["checkpoint_identity"], "b".repeat(64));
    assert_eq!(serving["startup_nonce"], "e".repeat(32));
    assert_eq!(
        serving["served_model_names"],
        serde_json::json!([MODEL, ALIAS])
    );
    assert_eq!(serving["version"], "1.2.3");
    assert_eq!(serving["source_commit"], "d".repeat(40));
}

#[tokio::test]
async fn the_model_list_is_exactly_what_the_frontend_declares() {
    let server = TestServerV1::start(ScriptedBackendV1::new(Vec::new())).await;

    let response = server.request(&get("/v1/models")).await;

    assert_eq!(response.status, 200);
    let body = response.json();
    assert_eq!(body["data"][0]["id"], MODEL);
    assert_eq!(body["data"][0]["owned_by"], "fixture-owner");
    assert_eq!(body["data"][1]["id"], ALIAS);
    assert_eq!(body["data"][2], Value::Null);
}

#[tokio::test]
async fn a_model_the_frontend_does_not_declare_is_not_served() {
    let server = TestServerV1::start(ScriptedBackendV1::new(Vec::new())).await;

    let response = server
        .request(&post(
            "/v1/completions",
            r#"{"model":"some.other.model","prompt":"x"}"#,
        ))
        .await;

    assert_eq!(response.status, 404);
    assert_eq!(response.json()["error"]["code"], 404);
}

#[tokio::test]
async fn the_frontend_renders_the_prompt_the_backend_receives() {
    let backend = ScriptedBackendV1::new(vec![content("hi"), finished()]);
    let server = TestServerV1::start(backend.clone()).await;

    let response = server
        .request(&post(
            "/v1/chat/completions",
            &format!(
                r#"{{"model":"{MODEL}","messages":[{{"role":"system","content":"rules"}},{{"role":"user","content":"hello"}}],"max_completion_tokens":8}}"#
            ),
        ))
        .await;

    assert_eq!(response.status, 200);
    let prompts = backend.prompts();
    assert_eq!(
        prompts[0].prompt,
        PreparedPrompt::RenderedChat("system:rules|user:hello".to_owned())
    );
}

#[tokio::test]
async fn the_preserve_thinking_keyword_reaches_the_frontend() {
    for (options, expected) in [
        ("", "user:hello"),
        (
            r#","chat_template_kwargs":{"preserve_thinking":false}"#,
            "user:hello|preserve_thinking:false",
        ),
        (
            r#","chat_template_kwargs":{"preserve_thinking":true}"#,
            "user:hello|preserve_thinking:true",
        ),
    ] {
        let backend = ScriptedBackendV1::new(vec![content("hi"), finished()]);
        let server = TestServerV1::start(backend.clone()).await;
        let response = server
            .request(&post(
                "/v1/chat/completions",
                &format!(
                    r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hello"}}],"max_completion_tokens":8{options}}}"#
                ),
            ))
            .await;
        assert_eq!(response.status, 200, "{}", response.body);
        assert_eq!(
            backend.prompts()[0].prompt,
            PreparedPrompt::RenderedChat(expected.to_owned())
        );
    }
}

#[tokio::test]
async fn only_a_frontend_that_asks_reads_reasoning_content_on_tool_call_turns() {
    let history = r#"[{"role":"user","content":"x"},
        {"role":"assistant","content":"a","reasoning_content":"plain"},
        {"role":"assistant","content":"","reasoning_content":"called","tool_calls":[{"id":"c","type":"function","function":{"name":"fixture.lookup","arguments":"{}"}}]},
        {"role":"assistant","content":"","reasoning":"named","reasoning_content":"shadowed","tool_calls":[{"id":"d","type":"function","function":{"name":"fixture.lookup","arguments":"{}"}}]},
        {"role":"tool","tool_call_id":"c","content":"r"}]"#;
    for (fallback, expected) in [
        (
            false,
            "user:x|assistant:a|assistant:|assistant:~named|tool:r",
        ),
        (
            true,
            "user:x|assistant:a|assistant:~called|assistant:~named|tool:r",
        ),
    ] {
        let backend = ScriptedBackendV1::new(vec![content("hi"), finished()]);
        let mut frontend = FakeFrontendV1::with_reasoning_field("reasoning");
        frontend.tool_call_reasoning_fallback = fallback;
        let server = TestServerV1::start_with_frontend(backend.clone(), frontend).await;
        let response = server
            .request(&post(
                "/v1/chat/completions",
                &format!(
                    r#"{{"model":"{MODEL}","messages":{history},"tools":[{{"type":"function","function":{{"name":"fixture.lookup","parameters":{{}}}}}}],"max_completion_tokens":8}}"#
                ),
            ))
            .await;
        assert_eq!(response.status, 200, "{}", response.body);
        assert_eq!(
            backend.prompts()[0].prompt,
            PreparedPrompt::RenderedChat(expected.to_owned())
        );
    }
}

#[tokio::test]
async fn the_clear_thinking_keyword_reaches_the_frontend() {
    for (options, expected) in [
        ("", "user:hello"),
        (
            r#","chat_template_kwargs":{"clear_thinking":true}"#,
            "user:hello|clear_thinking:true",
        ),
        (
            r#","chat_template_kwargs":{"clear_thinking":false,"preserve_thinking":true}"#,
            "user:hello|preserve_thinking:true|clear_thinking:false",
        ),
        // The transport admits any value; only a frontend that reads the
        // keyword interprets it.
        (
            r#","chat_template_kwargs":{"clear_thinking":1}"#,
            "user:hello|clear_thinking:1",
        ),
        (
            r#","chat_template_kwargs":{"clear_thinking":{"x":["y"]}}"#,
            r#"user:hello|clear_thinking:{"x":["y"]}"#,
        ),
        (
            r#","chat_template_kwargs":{"clear_thinking":null}"#,
            "user:hello",
        ),
    ] {
        let backend = ScriptedBackendV1::new(vec![content("hi"), finished()]);
        let server = TestServerV1::start(backend.clone()).await;
        let response = server
            .request(&post(
                "/v1/chat/completions",
                &format!(
                    r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hello"}}],"max_completion_tokens":8{options}}}"#
                ),
            ))
            .await;
        assert_eq!(response.status, 200, "{}", response.body);
        assert_eq!(
            backend.prompts()[0].prompt,
            PreparedPrompt::RenderedChat(expected.to_owned())
        );
    }
}

/// The whole `chat_template_kwargs` object reaches the frontend: a key this
/// transport does not type, in the order sent and with its value unchanged,
/// beside the typed keywords it still reads. A typed keyword sent as null is
/// absent, as the typed read has it; any other null stays.
#[tokio::test]
async fn every_template_keyword_reaches_the_frontend_as_sent() {
    for (options, expected) in [
        ("", "user:hello|kwargs:none"),
        (r#","chat_template_kwargs":null"#, "user:hello|kwargs:none"),
        (r#","chat_template_kwargs":{}"#, "user:hello|kwargs:{}"),
        // The typed read takes an array positionally; the frontend then
        // reads the keywords it set, as before the object was forwarded.
        (
            r#","chat_template_kwargs":[null,null,null,true,null,null]"#,
            "user:hello|preserve_thinking:true|kwargs:none",
        ),
        (
            r#","chat_template_kwargs":{"model_effort":17.50,"enable_thinking":true,"z":[1,{"b":2,"a":null}]}"#,
            r#"user:hello|kwargs:{"model_effort":17.50,"enable_thinking":true,"z":[1,{"b":2,"a":null}]}"#,
        ),
        (
            r#","chat_template_kwargs":{"preserve_thinking":null,"custom":null,"clear_thinking":null}"#,
            r#"user:hello|kwargs:{"custom":null}"#,
        ),
        (
            r#","chat_template_kwargs":{"custom":"x","preserve_thinking":true}"#,
            r#"user:hello|preserve_thinking:true|kwargs:{"custom":"x","preserve_thinking":true}"#,
        ),
    ] {
        let backend = ScriptedBackendV1::new(vec![content("hi"), finished()]);
        let mut frontend = FakeFrontendV1::new();
        frontend.echoes_template_kwargs = true;
        let server = TestServerV1::start_with_frontend(backend.clone(), frontend).await;
        let response = server
            .request(&post(
                "/v1/chat/completions",
                &format!(
                    r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hello"}}],"max_completion_tokens":8{options}}}"#
                ),
            ))
            .await;
        assert_eq!(response.status, 200, "{options}: {}", response.body);
        assert_eq!(
            backend.prompts()[0].prompt,
            PreparedPrompt::RenderedChat(expected.to_owned()),
            "{options}"
        );
    }
}

/// A frontend that declares extra roles receives a text message of each by
/// name. Any other role is refused, naming the declared ones; a frontend
/// that declares none refuses an unknown role exactly as before.
#[tokio::test]
async fn declared_roles_reach_the_frontend_and_others_are_refused() {
    let roles = ["observation".to_owned(), "ipython".to_owned()];
    for (extra, messages, expected) in [
        (
            &roles[..],
            r#"[{"role":"user","content":"x"},{"role":"observation","content":"seen"},{"role":"ipython","content":"ran"}]"#,
            Ok("user:x|observation:seen|ipython:ran"),
        ),
        (
            &roles[..],
            r#"[{"role":"other","content":"x"}]"#,
            Err(
                "message role must be system, developer, user, assistant, tool, or one this model declares: observation, ipython",
            ),
        ),
        (
            &[][..],
            r#"[{"role":"observation","content":"x"}]"#,
            Err("message role must be system, developer, user, assistant, or tool"),
        ),
        (
            &roles[..],
            r#"[{"role":"observation","content":[{"type":"text","text":"x"}]}]"#,
            Err("observation messages contain only string content"),
        ),
        (
            &roles[..],
            r#"[{"role":"observation","content":"x","reasoning_content":"r"}]"#,
            Err("message contains fields that are not valid for its role"),
        ),
        (
            &roles[..],
            r#"[{"role":"observation","content":null}]"#,
            Err("message content must be a string"),
        ),
    ] {
        let backend = ScriptedBackendV1::new(vec![content("hi"), finished()]);
        let mut frontend = FakeFrontendV1::new();
        frontend.extra_roles = extra.to_vec();
        let server = TestServerV1::start_with_frontend(backend.clone(), frontend).await;
        let response = server
            .request(&post(
                "/v1/chat/completions",
                &format!(
                    r#"{{"model":"{MODEL}","messages":{messages},"max_completion_tokens":8}}"#
                ),
            ))
            .await;
        match expected {
            Ok(prompt) => {
                assert_eq!(response.status, 200, "{messages}: {}", response.body);
                assert_eq!(
                    backend.prompts()[0].prompt,
                    PreparedPrompt::RenderedChat(prompt.to_owned())
                );
            }
            Err(message) => {
                assert_eq!(response.status, 400, "{messages}: {}", response.body);
                let body: Value = serde_json::from_str(&response.body).unwrap();
                assert_eq!(body["error"]["message"], message, "{messages}");
                assert_eq!(body["error"]["param"], "messages", "{messages}");
            }
        }
    }
}

/// A frontend that renders `tool_reference` parts receives each one in a tool
/// message by its name, sent as `name`, as `tool_name` or as both when they
/// agree. A frontend that does not is never sent one, whatever the message,
/// and no other role carries one; a part of another shape is refused by the
/// field it lacks, adds or contradicts.
#[tokio::test]
async fn tool_references_reach_only_a_frontend_that_reads_them() {
    let assistant = r#"{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"find","arguments":"{}"}}]}"#;
    let tool = |content: &str| {
        format!(
            r#"[{{"role":"user","content":"q"}},{assistant},{{"role":"tool","tool_call_id":"c1","content":{content}}}]"#
        )
    };
    let user = |content: &str| format!(r#"[{{"role":"user","content":{content}}}]"#);
    let unsupported = Err("tool_reference content parts are not supported by this model");
    let unnamed = Err("a tool_reference part requires a string name");
    for (reads, messages, expected) in [
        (
            true,
            tool(r#"[{"type":"tool_reference","name":"a"},{"type":"tool_reference","name":"b"}]"#),
            Ok("user:q|assistant:|tool:@a@b"),
        ),
        (
            true,
            tool(r#"[{"type":"tool_reference","tool_name":"a"}]"#),
            Ok("user:q|assistant:|tool:@a"),
        ),
        (
            true,
            tool(r#"[{"type":"tool_reference","tool_name":"a","name":"a"}]"#),
            Ok("user:q|assistant:|tool:@a"),
        ),
        (
            true,
            tool(r#"[{"type":"text","text":"x"},{"type":"tool_reference","name":""}]"#),
            Ok("user:q|assistant:|tool:x@"),
        ),
        (
            false,
            tool(r#"[{"type":"tool_reference","name":"a"}]"#),
            unsupported,
        ),
        (
            false,
            tool(r#"[{"type":"tool_reference","tool_name":"a"}]"#),
            unsupported,
        ),
        (
            false,
            user(r#"[{"type":"tool_reference","name":"a"}]"#),
            unsupported,
        ),
        (
            true,
            user(r#"[{"type":"tool_reference","name":"a"}]"#),
            Err("only tool messages may contain tool_reference parts"),
        ),
        (
            true,
            r#"[{"role":"user","content":"q"},{"role":"assistant","content":[{"type":"tool_reference","name":"a"}]}]"#
                .to_owned(),
            Err("only tool messages may contain tool_reference parts"),
        ),
        (
            true,
            tool(r#"[{"type":"tool_reference","name":"a","tool_name":"b"}]"#),
            Err("a tool_reference part's name and tool_name differ"),
        ),
        (true, tool(r#"[{"type":"tool_reference"}]"#), unnamed),
        (
            true,
            tool(r#"[{"type":"tool_reference","name":null}]"#),
            unnamed,
        ),
        (
            true,
            tool(r#"[{"type":"tool_reference","tool_name":7}]"#),
            unnamed,
        ),
        (
            true,
            tool(r#"[{"type":"tool_reference","name":"a","tool_name":null}]"#),
            unnamed,
        ),
        (
            true,
            tool(r#"[{"type":"tool_reference","name":"a","description":"d"}]"#),
            Err("unknown content part field 'description'"),
        ),
        (
            true,
            tool(r#"[{"tool_call_id":"c1","output":[{"type":"tool_reference","name":"a"}]}]"#),
            Err("type is required"),
        ),
        (
            true,
            tool(r#"[{"type":"tool_search_result","name":"a"}]"#),
            Err("unsupported content part type 'tool_search_result'"),
        ),
    ] {
        let backend = ScriptedBackendV1::new(vec![content("hi"), finished()]);
        let mut frontend = FakeFrontendV1::new();
        frontend.tool_references = reads;
        let server = TestServerV1::start_with_frontend(backend.clone(), frontend).await;
        let response = server
            .request(&post(
                "/v1/chat/completions",
                &format!(
                    r#"{{"model":"{MODEL}","messages":{messages},"max_completion_tokens":8}}"#
                ),
            ))
            .await;
        match expected {
            Ok(prompt) => {
                assert_eq!(response.status, 200, "{messages}: {}", response.body);
                assert_eq!(
                    backend.prompts()[0].prompt,
                    PreparedPrompt::RenderedChat(prompt.to_owned())
                );
            }
            Err(message) => {
                assert_eq!(response.status, 400, "{messages}: {}", response.body);
                let body: Value = serde_json::from_str(&response.body).unwrap();
                assert_eq!(body["error"]["message"], message, "{messages}");
                assert_eq!(body["error"]["param"], "messages", "{messages}");
                assert!(backend.prompts().is_empty(), "{messages}");
            }
        }
    }
}

/// A declared role carries no reasoning: either reasoning field is refused,
/// whichever one the frontend's response selects. A standard role keeps its
/// behavior, reading only the selected field.
#[tokio::test]
async fn a_declared_role_refuses_either_reasoning_field() {
    for field in ["reasoning_content", "reasoning"] {
        let other = if field == "reasoning" {
            "reasoning_content"
        } else {
            "reasoning"
        };
        for (messages, expected) in [
            (
                format!(r#"[{{"role":"observation","content":"x","{field}":"r"}}]"#),
                Err(()),
            ),
            (
                format!(r#"[{{"role":"observation","content":"x","{other}":"r"}}]"#),
                Err(()),
            ),
            (
                format!(r#"[{{"role":"observation","content":"x","{other}":null}}]"#),
                Ok("observation:x"),
            ),
            (
                format!(r#"[{{"role":"user","content":"x","{other}":"r"}}]"#),
                Ok("user:x"),
            ),
        ] {
            let backend = ScriptedBackendV1::new(vec![content("hi"), finished()]);
            let mut frontend = FakeFrontendV1::with_reasoning_field(field);
            frontend.extra_roles = vec!["observation".to_owned()];
            let server = TestServerV1::start_with_frontend(backend.clone(), frontend).await;
            let response = server
                .request(&post(
                    "/v1/chat/completions",
                    &format!(
                        r#"{{"model":"{MODEL}","messages":{messages},"max_completion_tokens":8}}"#
                    ),
                ))
                .await;
            match expected {
                Ok(prompt) => {
                    assert_eq!(
                        response.status, 200,
                        "{field} {messages}: {}",
                        response.body
                    );
                    assert_eq!(
                        backend.prompts()[0].prompt,
                        PreparedPrompt::RenderedChat(prompt.to_owned())
                    );
                }
                Err(()) => {
                    assert_eq!(
                        response.status, 400,
                        "{field} {messages}: {}",
                        response.body
                    );
                    let body: Value = serde_json::from_str(&response.body).unwrap();
                    assert_eq!(
                        body["error"]["message"],
                        "message contains fields that are not valid for its role",
                        "{field} {messages}"
                    );
                    assert!(backend.prompts().is_empty());
                }
            }
        }
    }
}

/// A key that spells serde_json's private number marker is an ordinary key,
/// at the root of the object or nested in a value, whatever it holds: the
/// object reaches the frontend as sent.
#[tokio::test]
async fn a_key_spelling_the_number_marker_is_forwarded_as_sent() {
    for kwargs in [
        r#"{"$serde_json::private::Number":"17"}"#,
        r#"{"$serde_json::private::Number":"not-a-number"}"#,
        r#"{"custom":{"$serde_json::private::Number":"17"}}"#,
        r#"{"custom":{"$serde_json::private::Number":"not-a-number"}}"#,
        r#"{"custom":[{"$serde_json::private::Number":"1.50","b":2}]}"#,
    ] {
        let backend = ScriptedBackendV1::new(vec![content("hi"), finished()]);
        let mut frontend = FakeFrontendV1::new();
        frontend.echoes_template_kwargs = true;
        let server = TestServerV1::start_with_frontend(backend.clone(), frontend).await;
        let response = server
            .request(&post(
                "/v1/chat/completions",
                &format!(
                    r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hello"}}],"max_completion_tokens":8,"chat_template_kwargs":{kwargs}}}"#
                ),
            ))
            .await;
        assert_eq!(response.status, 200, "{kwargs}: {}", response.body);
        assert_eq!(
            backend.prompts()[0].prompt,
            PreparedPrompt::RenderedChat(format!("user:hello|kwargs:{kwargs}")),
            "{kwargs}"
        );
    }
}

/// A key the typed read does not know may be named twice, at any depth, as
/// before forwarding: the request is accepted and the forwarded object holds
/// the key once, where it first appeared, with its last value. A typed
/// keyword named twice is still refused by the typed read.
#[tokio::test]
async fn a_key_named_twice_is_forwarded_once_with_its_last_value() {
    for (kwargs, expected) in [
        (
            r#"{"custom":1,"enable_thinking":true,"custom":2}"#,
            Ok(r#"user:hello|kwargs:{"custom":2,"enable_thinking":true}"#),
        ),
        (
            r#"{"x":{"a":1,"b":2,"a":[3.50]},"y":0}"#,
            Ok(r#"user:hello|kwargs:{"x":{"a":[3.50],"b":2},"y":0}"#),
        ),
        (
            r#"{"x":[{"a":1,"a":2}]}"#,
            Ok(r#"user:hello|kwargs:{"x":[{"a":2}]}"#),
        ),
        (
            r#"{"enable_thinking":true,"enable_thinking":false}"#,
            Err("invalid JSON request: duplicate field `enable_thinking`"),
        ),
    ] {
        let backend = ScriptedBackendV1::new(vec![content("hi"), finished()]);
        let mut frontend = FakeFrontendV1::new();
        frontend.echoes_template_kwargs = true;
        let server = TestServerV1::start_with_frontend(backend.clone(), frontend).await;
        let response = server
            .request(&post(
                "/v1/chat/completions",
                &format!(
                    r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hello"}}],"max_completion_tokens":8,"chat_template_kwargs":{kwargs}}}"#
                ),
            ))
            .await;
        match expected {
            Ok(prompt) => {
                assert_eq!(response.status, 200, "{kwargs}: {}", response.body);
                assert_eq!(
                    backend.prompts()[0].prompt,
                    PreparedPrompt::RenderedChat(prompt.to_owned()),
                    "{kwargs}"
                );
            }
            Err(message) => {
                assert_eq!(response.status, 400, "{kwargs}: {}", response.body);
                let body: Value = serde_json::from_str(&response.body).unwrap();
                assert!(
                    body["error"]["message"]
                        .as_str()
                        .unwrap()
                        .starts_with(message),
                    "{kwargs}: {}",
                    response.body
                );
                assert!(backend.prompts().is_empty());
            }
        }
    }
}

#[tokio::test]
async fn a_frontend_that_asks_reads_reasoning_content_on_every_assistant_turn() {
    let history = r#"[{"role":"user","content":"x"},
        {"role":"assistant","content":"a","reasoning_content":"plain"},
        {"role":"assistant","content":"b","reasoning":"named","reasoning_content":"shadowed"},
        {"role":"assistant","content":"","reasoning_content":"called","tool_calls":[{"id":"c","type":"function","function":{"name":"fixture.lookup","arguments":"{}"}}]},
        {"role":"tool","tool_call_id":"c","content":"r"}]"#;
    let backend = ScriptedBackendV1::new(vec![content("hi"), finished()]);
    let mut frontend = FakeFrontendV1::with_reasoning_field("reasoning");
    frontend.reasoning_content_fallback = true;
    let server = TestServerV1::start_with_frontend(backend.clone(), frontend).await;
    let response = server
        .request(&post(
            "/v1/chat/completions",
            &format!(
                r#"{{"model":"{MODEL}","messages":{history},"tools":[{{"type":"function","function":{{"name":"fixture.lookup","parameters":{{}}}}}}],"max_completion_tokens":8}}"#
            ),
        ))
        .await;
    assert_eq!(response.status, 200, "{}", response.body);
    assert_eq!(
        backend.prompts()[0].prompt,
        PreparedPrompt::RenderedChat(
            "user:x|assistant:a~plain|assistant:b~named|assistant:~called|tool:r".to_owned()
        )
    );
}

#[tokio::test]
async fn only_a_frontend_that_asks_receives_the_tool_objects_as_sent() {
    let tools = r#"[{"type":"function","function":{"description":null,"name":"fixture.lookup","x":[1,{"y":true}],"defer_loading":true}}]"#;
    for (reads, expected) in [
        (false, "user:hello"),
        (
            true,
            r#"user:hello|tool:{"description":null,"name":"fixture.lookup","x":[1,{"y":true}],"defer_loading":true}"#,
        ),
    ] {
        let backend = ScriptedBackendV1::new(vec![content("hi"), finished()]);
        let mut frontend = FakeFrontendV1::new();
        frontend.reads_tool_objects = reads;
        let server = TestServerV1::start_with_frontend(backend.clone(), frontend).await;
        let response = server
            .request(&post(
                "/v1/chat/completions",
                &format!(
                    r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hello"}}],"tools":{tools},"max_completion_tokens":8}}"#
                ),
            ))
            .await;
        assert_eq!(response.status, 200, "{}", response.body);
        assert_eq!(
            backend.prompts()[0].prompt,
            PreparedPrompt::RenderedChat(expected.to_owned())
        );
    }
}

#[tokio::test]
async fn a_frontend_refusal_names_the_parameter_it_blames() {
    let server = TestServerV1::start(ScriptedBackendV1::new(Vec::new())).await;

    let response = server
        .request(&post(
            "/v1/chat/completions",
            &format!(
                r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"x"}}],"max_completion_tokens":8,"reasoning_effort":"impossible"}}"#
            ),
        ))
        .await;

    assert_eq!(response.status, 400);
    let body = response.json();
    assert_eq!(body["error"]["param"], "reasoning_effort");
    assert_eq!(
        body["error"]["message"],
        "this model has no impossible effort"
    );
}

#[tokio::test]
async fn a_reasoning_delta_needs_a_frontend_that_enabled_reasoning() {
    let reasoning = GenerationEvent::Reasoning(ReasoningContent {
        delta: "thinking".to_owned(),
        generated_token_count: 1,
    });
    let refused =
        TestServerV1::start(ScriptedBackendV1::new(vec![reasoning.clone(), finished()])).await;

    let response = refused
        .request(&post(
            "/v1/chat/completions",
            &format!(r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"x"}}],"max_completion_tokens":8}}"#),
        ))
        .await;
    assert_eq!(response.status, 500);

    let accepted = TestServerV1::start(ScriptedBackendV1::new(vec![reasoning, finished()])).await;
    let response = accepted
        .request(&post(
            "/v1/chat/completions",
            &format!(
                r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"x"}}],"max_completion_tokens":8,"chat_template_kwargs":{{"thinking":true}}}}"#
            ),
        ))
        .await;
    assert_eq!(response.status, 200);
    assert_eq!(
        response.json()["choices"][0]["message"]["reasoning_content"],
        "thinking"
    );
}

#[tokio::test]
async fn a_frontend_selects_the_reasoning_field_in_responses_and_deltas() {
    let reasoning = GenerationEvent::Reasoning(ReasoningContent {
        delta: "thinking".to_owned(),
        generated_token_count: 1,
    });
    let ask = |stream: bool| {
        post(
            "/v1/chat/completions",
            &format!(
                r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"x"}}],"max_tokens":8,"chat_template_kwargs":{{"enable_thinking":true}},"stream":{stream}}}"#
            ),
        )
    };
    let server = TestServerV1::start_with_frontend(
        ScriptedBackendV1::new(vec![reasoning.clone(), finished()]),
        FakeFrontendV1::with_reasoning_field("reasoning"),
    )
    .await;
    let response = server.request(&ask(false)).await;
    assert_eq!(response.status, 200, "{}", response.body);
    let message = &response.json()["choices"][0]["message"];
    assert_eq!(message["reasoning"], "thinking");
    assert!(message.get("reasoning_content").is_none());

    let server = TestServerV1::start_with_frontend(
        ScriptedBackendV1::new(vec![reasoning, finished()]),
        FakeFrontendV1::with_reasoning_field("reasoning"),
    )
    .await;
    let response = server.request(&ask(true)).await;
    assert_eq!(response.status, 200, "{}", response.body);
    assert!(response.body.contains("\"reasoning\":\"thinking\""));
    assert!(!response.body.contains("reasoning_content"));
}

#[tokio::test]
async fn the_frontend_bounds_and_grades_every_tool_call() {
    let allowed = format!(
        r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"x"}}],"max_completion_tokens":8,"tools":[{{"type":"function","function":{{"name":"fixture.lookup","parameters":{{}}}}}}]}}"#
    );
    let call = |name: &str| GenerationEvent::ToolCalls {
        calls: vec![BackendToolCall {
            name: name.to_owned(),
            arguments: "{}".to_owned(),
        }],
        generated_token_count: 1,
    };

    // A well-formed name the request did not offer is the model's output, and
    // is served as the reference serves it, streamed or not.
    for stream in [false, true] {
        let server = TestServerV1::start(ScriptedBackendV1::new(vec![
            GenerationEvent::ToolCallDeltas(vec![runtime_server::ToolCallDelta {
                index: 0,
                name: Some("fixture.other".into()),
                arguments: "{}".into(),
            }]),
            call("fixture.other"),
            finished(),
        ]))
        .await;
        let body = allowed.replacen('{', &format!(r#"{{"stream":{stream},"#), 1);
        let response = server.request(&post("/v1/chat/completions", &body)).await;
        assert_eq!(response.status, 200, "{}", response.body);
        assert!(!response.body.contains("error"), "{}", response.body);
        assert!(response.body.contains(r#""name":"fixture.other""#));
    }

    // A call when the request offered no tools is not a completion the
    // decoder produces, so the backend is refused.
    for request in [
        format!(
            r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"x"}}],"max_completion_tokens":8}}"#
        ),
        allowed.replacen('{', r#"{"tool_choice":"none","#, 1),
    ] {
        let server = TestServerV1::start(ScriptedBackendV1::new(vec![
            call("fixture.lookup"),
            finished(),
        ]))
        .await;
        assert_eq!(
            server
                .request(&post("/v1/chat/completions", &request))
                .await
                .status,
            500,
            "{request}"
        );
    }

    // A group larger than the frontend's bound is refused.
    let server = TestServerV1::start(ScriptedBackendV1::new(vec![
        GenerationEvent::ToolCalls {
            calls: (0..MAXIMUM_TOOL_CALLS + 1)
                .map(|_| BackendToolCall {
                    name: "fixture.lookup".to_owned(),
                    arguments: "{}".to_owned(),
                })
                .collect(),
            generated_token_count: 1,
        },
        finished(),
    ]))
    .await;
    assert_eq!(
        server
            .request(&post("/v1/chat/completions", &allowed))
            .await
            .status,
        500
    );

    // A call the frontend grades as well formed is served.
    let server = TestServerV1::start(ScriptedBackendV1::new(vec![
        call("fixture.lookup"),
        finished(),
    ]))
    .await;
    let response = server
        .request(&post("/v1/chat/completions", &allowed))
        .await;
    assert_eq!(response.status, 200);
    assert_eq!(
        response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "fixture.lookup"
    );
}

#[tokio::test]
async fn the_frontend_context_limit_is_the_one_the_transport_enforces() {
    let backend = ScriptedBackendV1::new(vec![content("hi"), finished()]);
    let server = TestServerV1::start(backend.clone()).await;

    // The backend reports four prompt tokens, so the boundary is at the limit.
    let inside = server
        .request(&post(
            "/v1/completions",
            &format!(
                r#"{{"model":"{MODEL}","prompt":"x","max_tokens":{}}}"#,
                MAXIMUM_TOTAL_TOKENS - 4
            ),
        ))
        .await;
    assert_eq!(inside.status, 200);

    let outside = server
        .request(&post(
            "/v1/completions",
            &format!(
                r#"{{"model":"{MODEL}","prompt":"x","max_tokens":{}}}"#,
                MAXIMUM_TOTAL_TOKENS - 3
            ),
        ))
        .await;
    assert_eq!(outside.status, 400);
    assert_eq!(outside.json()["error"]["code"], 400);
}

#[tokio::test]
async fn a_sampling_parameter_the_transport_owns_is_still_validated() {
    let server = TestServerV1::start(ScriptedBackendV1::new(Vec::new())).await;

    let response = server
        .request(&post(
            "/v1/completions",
            &format!(r#"{{"model":"{MODEL}","prompt":"x","temperature":2.5}}"#),
        ))
        .await;

    assert_eq!(response.status, 400);
    // The transport owns sampling bounds, so no frontend was consulted at all.
    assert_eq!(response.json()["error"]["param"], "temperature");
}

#[tokio::test]
async fn reasoning_can_be_hidden_without_changing_token_usage() {
    for stream in [false, true] {
        let server = TestServerV1::start(ScriptedBackendV1::new(vec![
            GenerationEvent::Reasoning(ReasoningContent {
                delta: "private calculation".into(),
                generated_token_count: 1,
            }),
            finished(),
        ]))
        .await;
        let options = if stream {
            r#", "stream_options":{"include_usage":true}"#
        } else {
            ""
        };
        let body = format!(
            r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"x"}}],"max_tokens":8,"chat_template_kwargs":{{"enable_thinking":true}},"include_reasoning":false,"stream":{stream}{options}}}"#
        );
        let response = server.request(&post("/v1/chat/completions", &body)).await;
        assert_eq!(response.status, 200, "{}", response.body);
        assert!(!response.body.contains("private calculation"));
        assert!(response.body.contains(r#""completion_tokens":1"#));
    }
}

#[tokio::test]
async fn disabling_parallel_tools_keeps_only_the_first_call() {
    for stream in [false, true] {
        let calls = (0..2)
            .map(|_| BackendToolCall {
                name: "fixture.lookup".into(),
                arguments: "{}".into(),
            })
            .collect();
        let server = TestServerV1::start(ScriptedBackendV1::new(vec![
            GenerationEvent::ToolCalls {
                calls,
                generated_token_count: 1,
            },
            finished(),
        ]))
        .await;
        let body = format!(
            r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"x"}}],"max_tokens":8,"parallel_tool_calls":false,"stream":{stream},"tools":[{{"type":"function","function":{{"name":"fixture.lookup","parameters":{{}}}}}}]}}"#
        );
        let response = server.request(&post("/v1/chat/completions", &body)).await;
        assert_eq!(response.status, 200, "{}", response.body);
        assert_eq!(
            response.body.matches(r#""name":"fixture.lookup""#).count(),
            1,
            "{}",
            response.body
        );
        assert!(response.body.contains(r#""finish_reason":"tool_calls""#));
    }
}

#[tokio::test]
async fn incremental_tools_preserve_ids_arguments_and_nonstream_output() {
    for stream in [false, true] {
        let server = TestServerV1::start(ScriptedBackendV1::new(vec![
            GenerationEvent::ToolCallDeltas(vec![runtime_server::ToolCallDelta {
                index: 0,
                name: Some("fixture.lookup".into()),
                arguments: "{".into(),
            }]),
            GenerationEvent::ToolCallDeltas(vec![runtime_server::ToolCallDelta {
                index: 0,
                name: None,
                arguments: "}".into(),
            }]),
            GenerationEvent::ToolCalls {
                calls: vec![BackendToolCall {
                    name: "fixture.lookup".into(),
                    arguments: "{}".into(),
                }],
                generated_token_count: 1,
            },
            finished(),
        ]))
        .await;
        let body = format!(
            r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"x"}}],"max_tokens":8,"stream":{stream},"tools":[{{"type":"function","function":{{"name":"fixture.lookup","parameters":{{}}}}}}]}}"#
        );
        let response = server.request(&post("/v1/chat/completions", &body)).await;
        assert_eq!(response.status, 200, "{}", response.body);
        assert_eq!(
            response.body.matches(r#""name":"fixture.lookup""#).count(),
            1
        );
        assert!(response.body.contains(r#""finish_reason":"tool_calls""#));
        if stream {
            assert!(response.body.contains(r#""arguments":"{""#));
            assert!(response.body.contains(r#""arguments":"}""#));
            assert!(!response.body.contains(r#""arguments":"{}""#));
        } else {
            assert_eq!(
                response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
                "{}"
            );
        }
    }
}

/// A final call must repeat the call that streamed, name included; a
/// response that streams nothing ignores the deltas and serves the call.
#[tokio::test]
async fn a_final_call_that_renames_its_streamed_call_is_refused_when_streamed() {
    for stream in [false, true] {
        let server = TestServerV1::start(ScriptedBackendV1::new(vec![
            GenerationEvent::ToolCallDeltas(vec![runtime_server::ToolCallDelta {
                index: 0,
                name: Some("fixture.streamed".into()),
                arguments: "{}".into(),
            }]),
            GenerationEvent::ToolCalls {
                calls: vec![BackendToolCall {
                    name: "fixture.lookup".into(),
                    arguments: "{}".into(),
                }],
                generated_token_count: 1,
            },
            finished(),
        ]))
        .await;
        let body = format!(
            r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"x"}}],"max_tokens":8,"stream":{stream},"tools":[{{"type":"function","function":{{"name":"fixture.lookup","parameters":{{}}}}}}]}}"#
        );
        let response = server.request(&post("/v1/chat/completions", &body)).await;
        assert_eq!(
            response
                .body
                .contains("streamed tool arguments differ from final calls"),
            stream,
            "{}",
            response.body
        );
    }
}

/// Calls separated by text arrive as one group each, after their deltas.
#[tokio::test]
async fn each_streamed_tool_group_matches_its_own_deltas() {
    let delta = |index: usize| {
        GenerationEvent::ToolCallDeltas(vec![runtime_server::ToolCallDelta {
            index,
            name: Some("fixture.lookup".into()),
            arguments: format!("{{\"n\": {index}}}"),
        }])
    };
    let group = |index: usize| GenerationEvent::ToolCalls {
        calls: vec![BackendToolCall {
            name: "fixture.lookup".into(),
            arguments: format!("{{\"n\": {index}}}"),
        }],
        generated_token_count: 1,
    };
    for stream in [false, true] {
        let server = TestServerV1::start(ScriptedBackendV1::new(vec![
            delta(0),
            group(0),
            content("between"),
            delta(1),
            group(1),
            GenerationEvent::Finished {
                reason: FinishReason::Stop,
                usage: TokenUsage {
                    prompt_tokens: 4,
                    completion_tokens: 3,
                    cached_tokens: None,
                },
            },
        ]))
        .await;
        let body = format!(
            r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"x"}}],"max_tokens":8,"stream":{stream},"tools":[{{"type":"function","function":{{"name":"fixture.lookup","parameters":{{}}}}}}]}}"#
        );
        let response = server.request(&post("/v1/chat/completions", &body)).await;
        assert_eq!(response.status, 200, "{}", response.body);
        assert!(!response.body.contains("error"), "{}", response.body);
        assert_eq!(
            response.body.matches(r#""name":"fixture.lookup""#).count(),
            2,
            "{}",
            response.body
        );
        assert!(response.body.contains("between"), "{}", response.body);
        assert!(response.body.contains(r#""finish_reason":"tool_calls""#));
    }
}

#[tokio::test]
async fn generated_ids_cover_reasoning_tools_and_hidden_control_tokens() {
    for stream in [false, true] {
        for hidden in [false, true] {
            let server = TestServerV1::start(ScriptedBackendV1::new(vec![
                GenerationEvent::Prompt {
                    token_ids: vec![1, 2, 3, 4],
                    text: "abcd".into(),
                    tokens: vec!["a".into(), "b".into(), "c".into(), "d".into()],
                },
                GenerationEvent::GeneratedTokenIds(vec![10]),
                GenerationEvent::Reasoning(ReasoningContent {
                    delta: "plan".into(),
                    generated_token_count: 1,
                }),
                GenerationEvent::GeneratedTokenIds(vec![11]),
                GenerationEvent::ToolCalls {
                    calls: vec![BackendToolCall {
                        name: "fixture.lookup".into(),
                        arguments: "{}".into(),
                    }],
                    generated_token_count: 1,
                },
                GenerationEvent::GeneratedTokenIds(vec![12]),
                GenerationEvent::TokenUsageDelta {
                    generated_token_count: 1,
                },
                GenerationEvent::Finished {
                    reason: FinishReason::ToolCalls,
                    usage: TokenUsage {
                        prompt_tokens: 4,
                        completion_tokens: 3,
                        cached_tokens: None,
                    },
                },
            ]))
            .await;
            let body = serde_json::json!({"model": MODEL, "messages": [{"role":"user","content":"x"}], "max_tokens":8,
                "stream":stream, "return_token_ids":true, "include_reasoning":!hidden,
                "chat_template_kwargs":{"enable_thinking":true}, "tools":[{"type":"function","function":{"name":"fixture.lookup","parameters":{}}}]});
            let response = server
                .request(&post("/v1/chat/completions", &body.to_string()))
                .await;
            assert_eq!(response.status, 200, "{}", response.body);
            if stream {
                let values: Vec<Value> = response
                    .body
                    .lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .filter_map(|data| serde_json::from_str(data).ok())
                    .collect();
                let ids: Vec<u64> = values
                    .iter()
                    .flat_map(|value| value["choices"].as_array().unwrap())
                    .flat_map(|choice| choice["token_ids"].as_array().into_iter().flatten())
                    .map(|id| id.as_u64().unwrap())
                    .collect();
                assert_eq!(ids, if hidden { vec![] } else { vec![10, 11, 12] });
                assert!(!response.body.contains("invalid_backend_event"));
            } else if hidden {
                assert!(response.json()["choices"][0].get("token_ids").is_none());
            } else {
                assert_eq!(
                    response.json()["choices"][0]["token_ids"],
                    serde_json::json!([10, 11, 12])
                );
            }
        }
    }
}

struct SerialBackend {
    slot: Arc<tokio::sync::Semaphore>,
    later_choice_first: bool,
}

struct RefusingReplica {
    error: AdmissionError,
    cancel: bool,
    calls: AtomicUsize,
}

impl GenerationBackend for RefusingReplica {
    fn readiness(&self) -> BackendReadiness {
        BackendReadiness::Ready
    }
    fn maximum_sequence_tokens(&self) -> usize {
        MAXIMUM_TOTAL_TOKENS
    }
    fn prompt_token_count(&self, _: &PreparedPrompt) -> Result<usize, AdmissionError> {
        Ok(4)
    }
    fn submit(
        &self,
        request: PreparedRequest,
        _: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let error = self.error.clone();
        Box::pin(async move {
            if self.cancel {
                request.cancellation.cancel();
            }
            Err(error)
        })
    }
}

#[tokio::test]
async fn grouped_pool_fallback_preserves_choices_and_usage() {
    for error in [
        AdmissionError::AtCapacity,
        AdmissionError::WorkerUnavailable,
    ] {
        let refusing = Arc::new(RefusingReplica {
            error,
            cancel: false,
            calls: AtomicUsize::new(0),
        });
        let healthy = ScriptedBackendV1::new(vec![content("ok"), finished()]);
        let replicas: Vec<Arc<dyn GenerationBackend>> = vec![refusing.clone(), healthy.clone()];
        let server = TestServerV1::start(Arc::new(
            runtime_server::LocalReplicaPool::new(replicas).unwrap(),
        ))
        .await;
        let body = serde_json::json!({"model":MODEL,"prompt":["x","y"],"max_tokens":1});
        let response = server
            .request(&post("/v1/completions", &body.to_string()))
            .await;
        assert_eq!(response.status, 200, "{}", response.body);
        assert_eq!(refusing.calls.load(Ordering::Relaxed), 1);
        assert_eq!(healthy.submissions.load(Ordering::Relaxed), 2);
        let value = response.json();
        assert_eq!(value["choices"][0]["index"], 0);
        assert_eq!(value["choices"][1]["index"], 1);
        assert_eq!(value["usage"]["prompt_tokens"], 8);
        assert_eq!(value["usage"]["completion_tokens"], 2);
    }
}

#[tokio::test]
async fn a_cancelled_pool_choice_is_not_retried() {
    let capture = ScriptedBackendV1::new(vec![content("ok"), finished()]);
    let server = TestServerV1::start(capture.clone()).await;
    let body = serde_json::json!({"model":MODEL,"prompt":"x","max_tokens":1});
    assert_eq!(
        server
            .request(&post("/v1/completions", &body.to_string()))
            .await
            .status,
        200
    );
    let mut request = capture.prompts().pop().unwrap();
    request.cancellation = CancellationSignal::new();
    let refusing = Arc::new(RefusingReplica {
        error: AdmissionError::WorkerUnavailable,
        cancel: true,
        calls: AtomicUsize::new(0),
    });
    let healthy = ScriptedBackendV1::new(vec![finished()]);
    let replicas: Vec<Arc<dyn GenerationBackend>> = vec![refusing.clone(), healthy.clone()];
    let pool = runtime_server::LocalReplicaPool::new(replicas).unwrap();
    let (sender, _events) = tokio::sync::mpsc::channel(8);
    let admitted = pool
        .submit_batch(vec![(request, EventSender::new(sender))])
        .pop()
        .unwrap()
        .await;
    assert_eq!(admitted, Err(AdmissionError::WorkerUnavailable));
    assert_eq!(healthy.submissions.load(Ordering::Relaxed), 0);
}

struct BatchFailureBackend {
    failing: usize,
    cancellations: std::sync::Mutex<Vec<CancellationSignal>>,
}

impl GenerationBackend for BatchFailureBackend {
    fn readiness(&self) -> BackendReadiness {
        BackendReadiness::Ready
    }
    fn maximum_sequence_tokens(&self) -> usize {
        MAXIMUM_TOTAL_TOKENS
    }
    fn prompt_token_count(&self, _: &PreparedPrompt) -> Result<usize, AdmissionError> {
        Ok(4)
    }
    fn submit(
        &self,
        request: PreparedRequest,
        events: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        let mut cancellations = self.cancellations.lock().unwrap();
        let index = cancellations.len();
        cancellations.push(request.cancellation.clone());
        if index == self.failing {
            return Box::pin(async { Err(AdmissionError::InvalidRequest) });
        }
        tokio::spawn(async move {
            request.cancellation.cancelled().await;
            drop(events);
        });
        Box::pin(async {
            // The engine reports validation errors before its first successful step.
            tokio::task::yield_now().await;
            Ok(())
        })
    }
}

#[tokio::test]
async fn a_failed_batch_choice_cancels_all_siblings() {
    for failing in [0, 1, 2] {
        for stream in [false, true] {
            let backend = Arc::new(BatchFailureBackend {
                failing,
                cancellations: std::sync::Mutex::new(Vec::new()),
            });
            let server = TestServerV1::start(backend.clone()).await;
            let body =
                serde_json::json!({"model": MODEL, "prompt":["x","y","z"], "stream": stream});
            let response = server
                .request(&post("/v1/completions", &body.to_string()))
                .await;
            assert_eq!(response.status, 400, "{}", response.body);
            assert!(
                response.body.contains("generation request is invalid"),
                "{}",
                response.body
            );
            let cancellations = backend.cancellations.lock().unwrap();
            assert_eq!(
                cancellations.len(),
                3,
                "every choice is submitted before admission"
            );
            assert!(cancellations.iter().all(CancellationSignal::is_cancelled));
        }
    }
}

impl GenerationBackend for SerialBackend {
    fn readiness(&self) -> BackendReadiness {
        BackendReadiness::Ready
    }
    fn maximum_sequence_tokens(&self) -> usize {
        MAXIMUM_TOTAL_TOKENS
    }
    fn prompt_token_count(&self, _: &PreparedPrompt) -> Result<usize, AdmissionError> {
        Ok(4)
    }
    fn submit(
        &self,
        request: PreparedRequest,
        events: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        Box::pin(async move {
            let later = self.later_choice_first && !request.request_id.ends_with("_0");
            let permit = if later {
                None
            } else {
                Some(self.slot.clone().acquire_owned().await.unwrap())
            };
            let slot = self.slot.clone();
            tokio::spawn(async move {
                for _ in 0..request.max_new_tokens {
                    if events.emit(content("x")).await.is_err() {
                        return;
                    }
                }
                let _ = events
                    .emit(GenerationEvent::Finished {
                        reason: FinishReason::Length,
                        usage: TokenUsage {
                            prompt_tokens: 4,
                            completion_tokens: request.max_new_tokens as u64,
                            cached_tokens: None,
                        },
                    })
                    .await;
                drop(permit);
                if later {
                    slot.add_permits(1);
                }
            });
            Ok(())
        })
    }
}

#[tokio::test]
async fn queued_choices_drain_earlier_output_during_admission() {
    for stream in [false, true] {
        let server = TestServerV1::start(Arc::new(SerialBackend {
            slot: Arc::new(tokio::sync::Semaphore::new(1)),
            later_choice_first: false,
        }))
        .await;
        let mut body = serde_json::json!({"model": MODEL, "prompt":"hello", "n":2, "max_tokens":60, "stream":stream});
        if stream {
            body["stream_options"] = serde_json::json!({"include_usage": true});
        }
        let response = server
            .request(&post("/v1/completions", &body.to_string()))
            .await;
        assert_eq!(response.status, 200, "{}", response.body);
        assert!(
            response.body.contains(r#""completion_tokens":120"#),
            "{}",
            response.body
        );
    }
}

#[tokio::test]
async fn a_later_choice_drains_while_the_first_waits_for_admission() {
    for stream in [false, true] {
        let server = TestServerV1::start(Arc::new(SerialBackend {
            slot: Arc::new(tokio::sync::Semaphore::new(0)),
            later_choice_first: true,
        }))
        .await;
        let mut body = serde_json::json!({"model": MODEL, "prompt":["hello", "world"], "max_tokens":60, "stream":stream});
        if stream {
            body["stream_options"] = serde_json::json!({"include_usage":true});
        }
        let response = server
            .request(&post("/v1/completions", &body.to_string()))
            .await;
        assert_eq!(response.status, 200, "{}", response.body);
        assert!(
            response.body.contains(r#""completion_tokens":120"#),
            "{}",
            response.body
        );
    }
}

/// Payload bytes whose live owners a lease counts.
struct Leased(#[allow(dead_code)] Arc<()>);

impl AsRef<[u8]> for Leased {
    fn as_ref(&self) -> &[u8] {
        &[0; 4]
    }
}

/// Prepares every prompt as one media item whose payload a fresh lease
/// counts, and drops its submitted copy at once, as an engine drops a payload
/// once prefill passes the item. It holds the response until released.
struct PayloadHoldingBackend {
    lease: Arc<()>,
    submitted: tokio::sync::mpsc::UnboundedSender<()>,
    release: Arc<tokio::sync::Semaphore>,
}

impl GenerationBackend for PayloadHoldingBackend {
    fn readiness(&self) -> BackendReadiness {
        BackendReadiness::Ready
    }
    fn maximum_sequence_tokens(&self) -> usize {
        MAXIMUM_TOTAL_TOKENS
    }
    fn prompt_token_count(&self, _: &PreparedPrompt) -> Result<usize, AdmissionError> {
        Ok(4)
    }
    fn prepare_prompt(&self, _: PreparedPrompt) -> Result<(PreparedPrompt, usize), AdmissionError> {
        let payload = runtime_server::MediaPayloadV1::new(Leased(Arc::clone(&self.lease)));
        Ok((
            PreparedPrompt::MediaTokens {
                tokens: vec![1, 2, 3, 4],
                media: vec![runtime_server::MediaInputV1 {
                    key: [7; 32],
                    encoder: 0,
                    geometry: [1, 1, 1, 0],
                    output_rows: 1,
                    payload,
                }],
                spans: vec![runtime_server::MediaSpanV1 {
                    media: 0,
                    prompt_row: 1,
                    rows: 1,
                    media_row: 0,
                }],
                positions: Vec::new(),
            },
            4,
        ))
    }
    fn submit(
        &self,
        request: PreparedRequest,
        events: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        drop(request);
        let _ = self.submitted.send(());
        let release = Arc::clone(&self.release);
        Box::pin(async move {
            tokio::spawn(async move {
                let _permit = release.acquire_owned().await;
                let _ = events
                    .emit(GenerationEvent::Finished {
                        reason: FinishReason::Length,
                        usage: TokenUsage {
                            prompt_tokens: 4,
                            completion_tokens: 0,
                            cached_tokens: None,
                        },
                    })
                    .await;
            });
            Ok(())
        })
    }
}

/// Once a nonstreaming request is handed to the backend, the handler holds
/// no copy of its prepared media while it waits for the response, so the
/// backend's release of a payload returns its host memory.
#[tokio::test]
async fn the_handler_keeps_no_prepared_media_while_a_response_is_open() {
    let lease = Arc::new(());
    let (submitted, mut submissions) = tokio::sync::mpsc::unbounded_channel();
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let server = TestServerV1::start(Arc::new(PayloadHoldingBackend {
        lease: Arc::clone(&lease),
        submitted,
        release: Arc::clone(&release),
    }))
    .await;
    // This test's lease and the backend's; each live payload adds one.
    let baseline = Arc::strong_count(&lease);
    let body = serde_json::json!({"model": MODEL, "prompt": "hello", "max_tokens": 4});
    let response = tokio::spawn(async move {
        server
            .request(&post("/v1/completions", &body.to_string()))
            .await
    });
    timeout(Duration::from_secs(5), submissions.recv())
        .await
        .expect("the request is submitted")
        .expect("the backend reports it");
    timeout(Duration::from_secs(5), async {
        while Arc::strong_count(&lease) > baseline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("no copy of the prepared payload outlives the submission");
    release.add_permits(1);
    let response = timeout(Duration::from_secs(5), response)
        .await
        .expect("the response completes")
        .unwrap();
    assert_eq!(response.status, 200, "{}", response.body);
}

/// Prepares by waiting for the request's cancellation, as a backend waiting
/// on a slow media decode would, and reports whether it came.
struct PreparingBackend {
    entered: tokio::sync::mpsc::UnboundedSender<()>,
    cancelled: tokio::sync::mpsc::UnboundedSender<bool>,
    release: Option<Arc<std::sync::atomic::AtomicBool>>,
}

impl GenerationBackend for PreparingBackend {
    fn readiness(&self) -> BackendReadiness {
        BackendReadiness::Ready
    }
    fn maximum_sequence_tokens(&self) -> usize {
        MAXIMUM_TOTAL_TOKENS
    }
    fn prompt_token_count(&self, _: &PreparedPrompt) -> Result<usize, AdmissionError> {
        Ok(4)
    }
    fn prepare_prompt_cancellable(
        &self,
        prompt: PreparedPrompt,
        cancellation: &CancellationSignal,
    ) -> Result<(PreparedPrompt, usize), AdmissionError> {
        let _ = self.entered.send(());
        let until = std::time::Instant::now() + Duration::from_secs(10);
        while !cancellation.is_cancelled()
            && !self
                .release
                .as_ref()
                .is_some_and(|release| release.load(Ordering::SeqCst))
            && std::time::Instant::now() < until
        {
            std::thread::sleep(Duration::from_millis(5));
        }
        let _ = self.cancelled.send(cancellation.is_cancelled());
        while self
            .release
            .as_ref()
            .is_some_and(|release| !release.load(Ordering::SeqCst))
            && std::time::Instant::now() < until
        {
            std::thread::sleep(Duration::from_millis(5));
        }
        Ok((prompt, 4))
    }
    fn submit(
        &self,
        _request: PreparedRequest,
        _events: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        Box::pin(async { Err(AdmissionError::AtCapacity) })
    }
}

#[tokio::test]
async fn a_client_that_leaves_during_preparation_cancels_it() {
    let (entered, mut entering) = tokio::sync::mpsc::unbounded_channel();
    let (cancelled, mut cancellation) = tokio::sync::mpsc::unbounded_channel();
    let server = TestServerV1::start(Arc::new(PreparingBackend {
        entered,
        cancelled,
        release: None,
    }))
    .await;
    let stream = TcpStream::connect(server.address)
        .await
        .expect("the server accepts connections");
    let body = serde_json::json!({"model": MODEL, "prompt": "hello", "max_tokens": 4});
    let mut request = post("/v1/completions", &body.to_string()).into_bytes();
    while !request.is_empty() {
        stream.writable().await.expect("the socket is writable");
        match stream.try_write(&request) {
            Ok(written) => {
                request.drain(..written);
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(error) => panic!("the socket write failed: {error}"),
        }
    }
    timeout(Duration::from_secs(5), entering.recv())
        .await
        .expect("preparation starts")
        .expect("the backend reports it");
    let health: Value = serde_json::from_str(&server.request(&get("/health")).await.body).unwrap();
    assert_eq!(health["waiting"]["preparation"], 1);
    assert_eq!(health["queue_depth"], 1);
    assert!(
        health["transport"]["buffer_reserved_bytes"]
            .as_u64()
            .unwrap()
            >= 2 * 1024 * 1024
    );
    drop(stream);
    let seen = timeout(Duration::from_secs(5), cancellation.recv())
        .await
        .expect("preparation stops once the client leaves")
        .expect("the backend reports it");
    assert!(seen);
    timeout(Duration::from_secs(5), async {
        loop {
            let health: Value =
                serde_json::from_str(&server.request(&get("/health")).await.body).unwrap();
            if health["queue_depth"] == 0 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let metrics = server.request(&get("/metrics")).await;
    assert!(metrics.body.contains("chop_requests_waiting 0\n"));
}

#[tokio::test]
async fn a_tokenize_client_that_leaves_during_preparation_cancels_it() {
    let (entered, mut entering) = tokio::sync::mpsc::unbounded_channel();
    let (cancelled, mut cancellation) = tokio::sync::mpsc::unbounded_channel();
    let server = TestServerV1::start(Arc::new(PreparingBackend {
        entered,
        cancelled,
        release: None,
    }))
    .await;
    let stream = TcpStream::connect(server.address)
        .await
        .expect("the server accepts connections");
    let body =
        serde_json::json!({"model": MODEL, "messages": [{"role":"user", "content":"hello"}]});
    let mut request = post("/tokenize", &body.to_string()).into_bytes();
    while !request.is_empty() {
        stream.writable().await.expect("the socket is writable");
        match stream.try_write(&request) {
            Ok(written) => {
                request.drain(..written);
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(error) => panic!("the socket write failed: {error}"),
        }
    }
    timeout(Duration::from_secs(5), entering.recv())
        .await
        .expect("preparation starts")
        .expect("the backend reports it");
    let health: Value = serde_json::from_str(&server.request(&get("/health")).await.body).unwrap();
    assert_eq!(health["waiting"]["preparation"], 1);
    drop(stream);
    let seen = timeout(Duration::from_secs(5), cancellation.recv())
        .await
        .expect("preparation stops once the client leaves")
        .expect("the backend reports it");
    assert!(seen);
    let health: Value = serde_json::from_str(&server.request(&get("/health")).await.body).unwrap();
    assert_eq!(health["queue_depth"], 0);
}

/// Synchronous preparation can retain its input after cancellation.
#[tokio::test]
async fn detached_preparation_keeps_intake_charged() {
    let (entered, mut entering) = tokio::sync::mpsc::unbounded_channel();
    let (cancelled, mut cancellation) = tokio::sync::mpsc::unbounded_channel();
    let release = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let address = listener.local_addr().unwrap();
    let backend = Arc::new(PreparingBackend {
        entered,
        cancelled,
        release: Some(release.clone()),
    });
    tokio::spawn(serve(
        listener,
        backend,
        Arc::new(FakeFrontendV1::new()),
        ServerConfig::new(1, 2, 16384, 1024)
            .unwrap()
            .with_intake_budget_bytes(16384),
    ));
    let server = TestServerV1 { address };
    let stream = TcpStream::connect(address).await.unwrap();
    let body = post("/v1/completions", r#"{"prompt":"hello","max_tokens":1}"#);
    stream.writable().await.unwrap();
    assert_eq!(stream.try_write(body.as_bytes()).unwrap(), body.len());
    timeout(Duration::from_secs(5), entering.recv())
        .await
        .unwrap()
        .unwrap();
    drop(stream);
    timeout(Duration::from_secs(5), cancellation.recv())
        .await
        .unwrap()
        .unwrap();
    let metrics = server.request(&get("/metrics")).await;
    // Release even on a failed assertion so the blocking worker can exit.
    let refused = timeout(Duration::from_millis(200), server.request(&body)).await;
    release.store(true, Ordering::SeqCst);
    assert!(
        metrics.body.contains("chop_intake_used_bytes 16384\n"),
        "{}",
        metrics.body
    );
    assert_eq!(refused.unwrap().status, 429);
    timeout(Duration::from_secs(5), async {
        loop {
            if server
                .request(&get("/metrics"))
                .await
                .body
                .contains("chop_intake_used_bytes 0\n")
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    server.request(&body).await;
    assert!(entering.try_recv().is_ok());
}

#[tokio::test]
async fn incomplete_headers_expire_while_healthy_requests_succeed() {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(serve(
        listener,
        ScriptedBackendV1::new(vec![]),
        Arc::new(FakeFrontendV1::new()),
        ServerConfig::default()
            .with_header_read_timeout(Duration::from_millis(100))
            .with_max_connections(4),
    ));
    let server = TestServerV1 { address };
    let mut stalled = Vec::new();
    for _ in 0..3 {
        let socket = TcpStream::connect(address).await.unwrap();
        socket.writable().await.unwrap();
        socket
            .try_write(b"GET /health HTTP/1.1\r\nHost: localhost\r\nX-Stall: ")
            .unwrap();
        stalled.push(socket);
    }
    assert_eq!(server.request(&get("/health")).await.status, 200);
    for socket in stalled {
        let response = timeout(Duration::from_secs(1), exchange(&socket, b""))
            .await
            .expect("incomplete headers must expire");
        assert!(!String::from_utf8_lossy(&response).starts_with("HTTP/1.1 200"));
    }
    assert_eq!(server.request(&get("/health")).await.status, 200);
}

#[tokio::test]
async fn accepted_connections_have_a_separate_capacity_bound() {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(serve(
        listener,
        ScriptedBackendV1::new(vec![]),
        Arc::new(FakeFrontendV1::new()),
        ServerConfig::default()
            .with_header_read_timeout(Duration::from_millis(500))
            .with_max_connections(1),
    ));
    let held = TcpStream::connect(address).await.unwrap();
    held.writable().await.unwrap();
    held.try_write(b"GET /health HTTP/1.1\r\n").unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;
    let excess = TcpStream::connect(address).await.unwrap();
    timeout(Duration::from_millis(200), excess.readable())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(excess.try_read(&mut [0; 1]).unwrap(), 0);
    let _ = timeout(Duration::from_secs(2), exchange(&held, b""))
        .await
        .unwrap();
    assert_eq!(
        TestServerV1 { address }
            .request(&get("/health"))
            .await
            .status,
        200
    );
}

#[path = "frontend_contract/responses.rs"]
mod responses;

#[path = "frontend_contract/lifecycle.rs"]
mod lifecycle;

#[tokio::test]
async fn disconnect_after_nonstreaming_admission_cancels_generation() {
    let backend = Arc::new(BatchFailureBackend {
        failing: usize::MAX,
        cancellations: std::sync::Mutex::new(Vec::new()),
    });
    let server = TestServerV1::start(backend.clone()).await;
    let stream = TcpStream::connect(server.address).await.unwrap();
    let wire = post(
        "/v1/completions",
        r#"{"prompt":"hello","max_tokens":1,"stream":false}"#,
    );
    stream.writable().await.unwrap();
    assert_eq!(stream.try_write(wire.as_bytes()).unwrap(), wire.len());
    timeout(Duration::from_secs(5), async {
        loop {
            if !backend.cancellations.lock().unwrap().is_empty() {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    // Let submit's successful admission resolve before disconnecting.
    tokio::time::sleep(Duration::from_millis(25)).await;
    let cancellation = backend.cancellations.lock().unwrap()[0].clone();
    assert!(!cancellation.is_cancelled());
    drop(stream);
    timeout(Duration::from_secs(5), cancellation.cancelled())
        .await
        .unwrap();
}

struct WaitingAdmissionBackend {
    slots: tokio::sync::Semaphore,
}

impl GenerationBackend for WaitingAdmissionBackend {
    fn readiness(&self) -> BackendReadiness {
        BackendReadiness::Ready
    }
    fn maximum_sequence_tokens(&self) -> usize {
        MAXIMUM_TOTAL_TOKENS
    }
    fn prompt_token_count(&self, _: &PreparedPrompt) -> Result<usize, AdmissionError> {
        Ok(4)
    }
    fn submit(
        &self,
        _: PreparedRequest,
        events: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        Box::pin(async move {
            self.slots.acquire().await.unwrap().forget();
            events
                .emit(GenerationEvent::Finished {
                    reason: FinishReason::Stop,
                    usage: TokenUsage {
                        prompt_tokens: 4,
                        completion_tokens: 0,
                        cached_tokens: Some(0),
                    },
                })
                .await
                .unwrap();
            Ok(())
        })
    }
}

#[tokio::test]
async fn backend_waiting_is_counted_once_and_disconnect_removes_it() {
    let backend = Arc::new(WaitingAdmissionBackend {
        slots: tokio::sync::Semaphore::new(0),
    });
    let server = TestServerV1::start(backend.clone()).await;
    let start = || {
        let server = TestServerV1 {
            address: server.address,
        };
        tokio::spawn(async move {
            let body = serde_json::json!({"model": MODEL, "prompt":"hello", "max_tokens":1});
            server
                .request(&post("/v1/completions", &body.to_string()))
                .await
        })
    };
    let first = start();
    let second = start();
    let wait_for = |count| {
        let server = &server;
        async move {
            timeout(Duration::from_secs(5), async {
                loop {
                    let health: Value =
                        serde_json::from_str(&server.request(&get("/health")).await.body).unwrap();
                    if health["waiting"]["backend"] == count {
                        assert_eq!(health["queue_depth"], count);
                        assert_eq!(health["waiting"]["pre_backend"], 0);
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
        }
    };
    wait_for(2).await;
    second.abort();
    assert!(matches!(second.await, Err(error) if error.is_cancelled()));
    wait_for(1).await;
    backend.slots.add_permits(1);
    assert_eq!(first.await.unwrap().status, 200);
    wait_for(0).await;
}

#[tokio::test]
async fn grammar_budget_failure_preserves_the_terminal_reason() {
    let reason = "grammar scanner work budget exceeded (limit: 4096 units per closure)";
    for streaming in [false, true] {
        for (path, mut body) in [
            (
                "/v1/completions",
                serde_json::json!({"model":MODEL,"prompt":"x"}),
            ),
            (
                "/v1/chat/completions",
                serde_json::json!({"model":MODEL,"messages":[{"role":"user","content":"x"}]}),
            ),
        ] {
            let server =
                TestServerV1::start(ScriptedBackendV1::new(vec![GenerationEvent::Failed {
                    message: reason.to_owned(),
                    code: Some("generation_failed".to_owned()),
                }]))
                .await;
            body["stream"] = streaming.into();
            let response = server.request(&post(path, &body.to_string())).await;
            if streaming {
                assert_eq!(response.status, 200);
                let values: Vec<Value> = response
                    .body
                    .lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .filter(|line| *line != "[DONE]")
                    .map(|line| serde_json::from_str(line).unwrap())
                    .collect();
                assert_eq!(values.last().unwrap()["error"]["message"], reason);
                assert_eq!(values.last().unwrap()["error"]["code"], "generation_failed");
                for choice in values
                    .iter()
                    .filter_map(|value| value["choices"].as_array())
                    .flatten()
                {
                    assert!(choice["finish_reason"].is_null());
                    assert!(choice["text"].as_str().is_none_or(str::is_empty));
                    assert!(
                        choice["delta"]["content"]
                            .as_str()
                            .is_none_or(str::is_empty)
                    );
                }
                assert!(response.body.contains("data: [DONE]"));
            } else {
                assert_eq!(response.status, 500);
                assert_eq!(response.json()["error"]["message"], reason);
                assert_eq!(response.json()["error"]["code"], "generation_failed");
            }
        }
    }
}
