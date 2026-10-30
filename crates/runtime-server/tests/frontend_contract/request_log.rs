//! Request-failure lines, read through a capture sink from a deployment
//! served with the full lifecycle on a loopback port.
use super::*;
use serde_json::json;
use std::sync::Mutex;

/// Text a client sends that no line may repeat.
const CANARY: &str = "canary-7f3e9b";
const FAILURE: &str = "chopinfer.request-failure.v1";
const CANCEL: &str = "chopinfer.request-cancel.v1";
const SUMMARY: &str = "chopinfer.summary.v1";

/// A backend that holds every admitted request until the test sends it
/// events. It closes a request's events once its cancellation fires, as the
/// worker does.
struct HoldingBackend {
    readiness: Mutex<BackendReadiness>,
    held: Arc<Mutex<Vec<(EventSender, CancellationSignal)>>>,
    admitted: AtomicUsize,
}

impl HoldingBackend {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            readiness: Mutex::new(BackendReadiness::Ready),
            held: Arc::new(Mutex::new(Vec::new())),
            admitted: AtomicUsize::new(0),
        })
    }

    async fn wait_admitted(&self, count: usize) {
        timeout(Duration::from_secs(5), async {
            while self.admitted.load(Ordering::Relaxed) < count {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the requests are admitted");
    }

    /// Sends `event` to every request still held.
    async fn emit(&self, event: GenerationEvent) {
        let held: Vec<EventSender> = self
            .held
            .lock()
            .unwrap()
            .iter()
            .map(|(events, _)| events.clone())
            .collect();
        for events in held {
            let _ = events.emit(event.clone()).await;
        }
    }

    /// Sends `event` to the held request admitted `index`th of those held.
    async fn emit_to(&self, index: usize, event: GenerationEvent) {
        let events = self.held.lock().unwrap()[index].0.clone();
        let _ = events.emit(event).await;
    }
}

impl GenerationBackend for HoldingBackend {
    fn readiness(&self) -> BackendReadiness {
        self.readiness.lock().unwrap().clone()
    }

    fn maximum_sequence_tokens(&self) -> usize {
        MAXIMUM_TOTAL_TOKENS
    }

    fn prompt_token_count(&self, _prompt: &PreparedPrompt) -> Result<usize, AdmissionError> {
        Ok(4)
    }

    fn submit(
        &self,
        request: PreparedRequest,
        events: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        let cancellation = request.cancellation.clone();
        self.held
            .lock()
            .unwrap()
            .push((events, cancellation.clone()));
        let held = Arc::clone(&self.held);
        tokio::spawn(async move {
            cancellation.cancelled().await;
            held.lock()
                .unwrap()
                .retain(|(_, signal)| !signal.is_cancelled());
        });
        self.admitted.fetch_add(1, Ordering::Relaxed);
        Box::pin(async { Ok(()) })
    }
}

/// A backend that refuses every submission the way a worker refuses an
/// out-of-vocabulary `logit_bias`, naming the token ID.
struct RefusingBackend;

impl GenerationBackend for RefusingBackend {
    fn readiness(&self) -> BackendReadiness {
        BackendReadiness::Ready
    }

    fn maximum_sequence_tokens(&self) -> usize {
        MAXIMUM_TOTAL_TOKENS
    }

    fn prompt_token_count(&self, _prompt: &PreparedPrompt) -> Result<usize, AdmissionError> {
        Ok(4)
    }

    fn submit(
        &self,
        _request: PreparedRequest,
        _events: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        Box::pin(async {
            Err(AdmissionError::InvalidParameter {
                message: "logit_bias token 987654321 is outside the vocabulary".to_owned(),
                parameter: "logit_bias",
            })
        })
    }
}

/// Engine counters that grow by a fixed step each time a summary reads them.
#[derive(Default)]
struct GrowingMetrics {
    reads: std::sync::atomic::AtomicU64,
}

impl runtime_server::MetricsSnapshotProvider for GrowingMetrics {
    fn snapshot(&self, _monotonic_micros: u64) -> String {
        String::new()
    }

    fn summary(&self) -> Option<runtime_server::EngineSummaryV1> {
        let reads = self.reads.fetch_add(1, Ordering::Relaxed) + 1;
        Some(runtime_server::EngineSummaryV1 {
            requests_submitted: 10 * reads,
            generation_tokens: 100 * reads,
            draft_tokens: 40 * reads,
            accepted_tokens: 30 * reads,
            // One observation at or below 1 ms and one between 1 and 10 ms
            // in every interval.
            ttft: runtime_server::LatencyHistogramV1 {
                bounds_micros: vec![1_000, 10_000],
                cumulative: vec![reads, 2 * reads],
                count: 2 * reads,
            },
            running: 3,
            waiting: 2,
            kv_tokens_used: 256,
            kv_tokens_capacity: 1024,
            worker_state: "ready",
            ..runtime_server::EngineSummaryV1::default()
        })
    }
}

struct LoggedServer {
    server: TestServerV1,
    lines: Arc<Mutex<Vec<String>>>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<std::io::Result<()>>,
}

impl LoggedServer {
    async fn start(backend: Arc<dyn GenerationBackend>, config: ServerConfig) -> Self {
        Self::start_with(
            backend,
            config,
            runtime_server::HttpOptions::default(),
            Duration::from_secs(1),
            runtime_server::metrics_snapshot_provider(|_| String::new()),
        )
        .await
    }

    async fn start_with(
        backend: Arc<dyn GenerationBackend>,
        config: ServerConfig,
        options: runtime_server::HttpOptions,
        drain_timeout: Duration,
        metrics: Arc<dyn runtime_server::MetricsSnapshotProvider>,
    ) -> Self {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let lines = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&lines);
        let options = runtime_server::HttpOptions {
            request_log_sink: Some(runtime_server::RequestLogSinkV1::new(move |line| {
                captured.lock().unwrap().push(line.to_owned());
            })),
            ..options
        };
        let (shutdown, stopped) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(runtime_server::serve_with_shutdown_and_http_options(
            listener,
            backend,
            Arc::new(FakeFrontendV1::new()),
            config,
            Arc::new(runtime_server::SystemMonotonicClock::new()),
            metrics,
            async {
                let _ = stopped.await;
            },
            drain_timeout,
            options,
        ));
        Self {
            server: TestServerV1 { address },
            lines,
            shutdown: Some(shutdown),
            task,
        }
    }

    async fn request(&self, request: &str) -> ResponseV1 {
        self.server.request(request).await
    }

    /// Writes `request` on a new connection and reads its answer in the
    /// background, so the request can stay open.
    async fn open(&self, request: String) -> tokio::task::JoinHandle<Vec<u8>> {
        let stream = TcpStream::connect(self.server.address).await.unwrap();
        tokio::spawn(async move { exchange(&stream, request.as_bytes()).await })
    }

    fn written(&self) -> Vec<Value> {
        self.lines
            .lock()
            .unwrap()
            .iter()
            .map(|line| serde_json::from_str(line).expect("each line is one JSON object"))
            .collect()
    }

    /// The request lines, once there are `count`, after a pause long enough
    /// for one more to arrive.
    async fn lines(&self, count: usize) -> Vec<Value> {
        let requests = || {
            self.written()
                .into_iter()
                .filter(|line| line["schema"] != SUMMARY)
                .collect::<Vec<_>>()
        };
        timeout(Duration::from_secs(5), async {
            while requests().len() < count {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("expected {count} lines: {:#?}", requests()));
        tokio::time::sleep(Duration::from_millis(200)).await;
        let lines = requests();
        assert_eq!(lines.len(), count, "{lines:#?}");
        lines
    }

    fn raw(&self) -> String {
        self.lines.lock().unwrap().join("\n")
    }
}

/// Writes `request` and reads its whole answer, leaving the connection open.
async fn answer_kept_alive(stream: &TcpStream, request: &[u8]) {
    stream.writable().await.unwrap();
    assert_eq!(stream.try_write(request).unwrap(), request.len());
    let mut response = Vec::new();
    loop {
        stream.readable().await.unwrap();
        let mut chunk = [0_u8; 4096];
        match stream.try_read(&mut chunk) {
            Ok(0) => panic!("closed: {}", String::from_utf8_lossy(&response)),
            Ok(read) => response.extend_from_slice(&chunk[..read]),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => continue,
            Err(error) => panic!("the socket read failed: {error}"),
        }
        let text = String::from_utf8_lossy(&response);
        if let Some((head, body)) = text.split_once("\r\n\r\n") {
            let length = head.lines().find_map(|line| {
                let line = line.to_ascii_lowercase();
                let length = line.strip_prefix("content-length:")?;
                Some(length.trim().parse::<usize>().unwrap())
            });
            if length.is_some_and(|length| body.len() >= length) {
                return;
            }
        }
    }
}

fn completion(body: Value) -> String {
    post("/v1/completions", &body.to_string())
}

fn assert_failure(line: &Value, status: u16, level: &str) {
    assert_eq!(line["schema"], FAILURE, "{line:#}");
    assert_eq!(line["status"], status, "{line:#}");
    assert_eq!(line["level"], level, "{line:#}");
}

fn assert_cancel(line: &Value, reason: &str) {
    assert_eq!(line["schema"], CANCEL, "{line:#}");
    assert_eq!(line["level"], "info", "{line:#}");
    assert_eq!(line["reason"], reason, "{line:#}");
}

/// Each refused request writes exactly one warn line saying what it was and
/// where it stopped, a request that succeeds writes none, and no line quotes
/// the request's text.
#[tokio::test]
async fn each_refused_request_writes_one_line_without_request_text() {
    let server = LoggedServer::start(
        ScriptedBackendV1::new(vec![content("hi"), finished()]),
        ServerConfig::default(),
    )
    .await;
    let served = server
        .request(&post(
            "/v1/chat/completions",
            &json!({"model": MODEL, "messages": [{"role": "user", "content": CANARY}], "max_tokens": 4})
                .to_string(),
        ))
        .await;
    assert_eq!(served.status, 200, "{}", served.body);
    let refusals = [
        (
            post(
                "/v1/completions",
                &format!(r#"{{"model":"{MODEL}","prompt":"{CANARY}"#),
            ),
            400,
        ),
        (
            post(
                "/v1/completions",
                &format!(r#"{{"model":"{MODEL}","prompt":"x","max_tokens":"{CANARY}"}}"#),
            ),
            400,
        ),
        (
            completion(json!({"model": "missing.model", "prompt": CANARY})),
            404,
        ),
        (
            "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: text/plain\r\n\
             Content-Length: 2\r\nConnection: close\r\n\r\n{}"
                .to_owned(),
            400,
        ),
        (get("/v1/nothing"), 404),
        (
            format!(
                "{} /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
                "X".repeat(100 * 1024)
            ),
            405,
        ),
    ];
    for (request, status) in &refusals {
        assert_eq!(server.request(request).await.status, *status);
    }

    let lines = server.lines(refusals.len()).await;

    for (line, (_, status)) in lines.iter().zip(&refusals) {
        assert_failure(line, *status, "warn");
        assert!(
            line["peer"].as_str().unwrap().starts_with("127.0.0.1:"),
            "{line:#}"
        );
        assert!(line["elapsed_seconds"].as_f64().unwrap() >= 0.0);
    }
    assert_eq!(lines[0]["message"], "invalid JSON request");
    assert_eq!(lines[0]["json_error"]["category"], "eof");
    assert_eq!(lines[0]["stage"], "preparation");
    assert_eq!(lines[0]["path"], "/v1/completions");
    assert_eq!(lines[0]["method"], "POST");
    assert!(lines[0]["request_id"].as_str().unwrap().starts_with("req_"));
    assert_eq!(lines[1]["json_error"]["category"], "data");
    assert_eq!(lines[2]["model"], "missing.model");
    assert_eq!(lines[2]["served"], json!([MODEL, ALIAS]));
    assert_eq!(lines[2]["param"], "model");
    assert_eq!(lines[2]["message"], "model is not available");
    assert_eq!(lines[3]["message"], "Content-Type must be application/json");
    assert_eq!(lines[3]["stage"], "intake");
    assert_eq!(lines[4]["path"], "/v1/nothing");
    assert_eq!(lines[4]["error_type"], "HttpError");
    assert!(lines[5]["method"].as_str().unwrap().len() <= 16);
    assert!(!server.raw().contains(CANARY), "{}", server.raw());
}

/// A model name, a request ID and a backend refusal that each carry what the
/// client sent are bounded or replaced.
#[tokio::test]
async fn client_controlled_fields_are_bounded_or_replaced() {
    let server = LoggedServer::start(Arc::new(RefusingBackend), ServerConfig::default()).await;
    let model = "m".repeat(1 << 20);
    let request = format!(
        "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
         X-Request-Id: {}\r\nContent-Length: {{}}\r\nConnection: close\r\n\r\n",
        "r".repeat(32 * 1024)
    );
    let body = json!({"model": model, "prompt": "x"}).to_string();
    let request = request.replace("{}", &body.len().to_string()) + &body;
    assert_eq!(server.request(&request).await.status, 404);
    let refused = server
        .request(&completion(
            json!({"model": MODEL, "prompt": "x", "logit_bias": {"987654321": 1}}),
        ))
        .await;
    assert_eq!(refused.status, 400, "{}", refused.body);

    let lines = server.lines(2).await;

    assert_failure(&lines[0], 404, "warn");
    assert!(lines[0]["model"].as_str().unwrap().len() <= 128);
    assert!(lines[0]["client_request_id"].as_str().unwrap().len() <= 128);
    assert_failure(&lines[1], 400, "warn");
    assert_eq!(lines[1]["param"], "logit_bias");
    assert_eq!(lines[1]["message"], "invalid request");
    assert_eq!(lines[1]["stage"], "admission");
    assert!(!server.raw().contains("987654321"), "{}", server.raw());
    for line in server.raw().lines() {
        assert!(line.len() < 4096, "{line}");
    }
}

/// A body that does not arrive in time, a request over capacity and one
/// without its key are each refused with one warn line.
#[tokio::test]
async fn deadline_capacity_and_authentication_refusals_are_warnings() {
    let backend = HoldingBackend::new();
    let server = LoggedServer::start(
        backend.clone(),
        ServerConfig::new(1, 8, 65536, 4096)
            .unwrap()
            .with_body_read_timeout(Duration::from_millis(200)),
    )
    .await;
    let stalled = server
        .open(
            "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
             Content-Length: 100\r\nConnection: close\r\n\r\n{\"model\""
                .to_owned(),
        )
        .await;
    let stalled = timeout(Duration::from_secs(5), stalled)
        .await
        .unwrap()
        .unwrap();
    assert!(String::from_utf8_lossy(&stalled).starts_with("HTTP/1.1 408"));
    let held = server
        .open(completion(
            json!({"model": MODEL, "prompt": "x", "max_tokens": 4}),
        ))
        .await;
    backend.wait_admitted(1).await;
    let refused = server
        .request(&completion(
            json!({"model": MODEL, "prompt": "x", "max_tokens": 4}),
        ))
        .await;
    assert_eq!(refused.status, 429, "{}", refused.body);
    backend.emit(content("hi")).await;
    backend.emit(finished()).await;
    assert!(String::from_utf8_lossy(&held.await.unwrap()).starts_with("HTTP/1.1 200"));

    let lines = server.lines(2).await;

    assert_failure(&lines[0], 408, "warn");
    assert_eq!(lines[0]["stage"], "intake");
    assert_eq!(lines[0]["message"], "request body did not arrive in time");
    assert_failure(&lines[1], 429, "warn");
    assert_eq!(lines[1]["message"], "server request capacity is exhausted");

    let keyed = LoggedServer::start_with(
        ScriptedBackendV1::new(vec![content("hi"), finished()]),
        ServerConfig::default(),
        runtime_server::HttpOptions {
            api_key: Some(runtime_server::ApiKey::new("secret")),
            ..Default::default()
        },
        Duration::from_secs(1),
        runtime_server::metrics_snapshot_provider(|_| String::new()),
    )
    .await;
    let unauthorized = keyed
        .request(&completion(json!({"model": MODEL, "prompt": "x"})))
        .await;
    assert_eq!(unauthorized.status, 401);
    let lines = keyed.lines(1).await;
    assert_failure(&lines[0], 401, "warn");
    assert_eq!(lines[0]["message"], "Unauthorized");
}

/// A failure after admission is one error line for the request, naming the
/// failing choice, whether or not it streams.
#[tokio::test]
async fn a_failure_after_admission_is_one_error_line() {
    let server = LoggedServer::start(
        ScriptedBackendV1::new(vec![
            content("hi"),
            GenerationEvent::Failed {
                message: format!("device fault near {CANARY}"),
                code: Some("device_fault".to_owned()),
            },
        ]),
        ServerConfig::default(),
    )
    .await;
    let plain = server
        .request(&completion(
            json!({"model": MODEL, "prompt": CANARY, "max_tokens": 4}),
        ))
        .await;
    assert_eq!(plain.status, 500, "{}", plain.body);
    let streamed = server
        .request(&completion(
            json!({"model": MODEL, "prompt": CANARY, "max_tokens": 4, "stream": true}),
        ))
        .await;
    assert_eq!(streamed.status, 200);
    assert!(streamed.body.contains("device_fault"), "{}", streamed.body);
    let choices = server
        .request(&completion(
            json!({"model": MODEL, "prompt": CANARY, "max_tokens": 4, "stream": true, "n": 2}),
        ))
        .await;
    assert_eq!(choices.status, 200);
    let whole_choices = server
        .request(&completion(
            json!({"model": MODEL, "prompt": CANARY, "max_tokens": 4, "n": 2}),
        ))
        .await;
    assert_eq!(whole_choices.status, 500, "{}", whole_choices.body);

    let lines = server.lines(4).await;

    for line in &lines {
        assert_failure(line, 500, "error");
        assert_eq!(line["code"], "device_fault");
        assert_eq!(line["message"], "backend failure");
        assert_eq!(line["stage"], "generation");
        assert_eq!(line["prompt_tokens"], 4);
    }
    assert!(lines[0]["request_id"].as_str().unwrap().ends_with("_0"));
    assert_eq!(lines[0]["generated_tokens"], 1);
    assert!(lines[1]["request_id"].as_str().unwrap().ends_with("_0"));
    assert_eq!(lines[1]["generated_tokens"], 1);
    // With several choices the line names the one that failed, and counts
    // what every choice generated.
    for line in &lines[2..] {
        let id = line["request_id"].as_str().unwrap();
        assert!(id.ends_with("_0") || id.ends_with("_1"), "{line:#}");
        assert!(line["generated_tokens"].as_u64().unwrap() >= 1, "{line:#}");
    }
    assert!(!server.raw().contains(CANARY), "{}", server.raw());
}

/// When one of several streamed choices fails, the request's line is that
/// choice's failure, not the cancellation of the choices still running. The
/// choices end on several threads, so the race is run many times.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failing_choice_of_a_stream_is_the_line_not_its_siblings() {
    const REQUESTS: usize = 40;
    let backend = HoldingBackend::new();
    let server = LoggedServer::start(backend.clone(), ServerConfig::default()).await;
    for request in 0..REQUESTS {
        timeout(Duration::from_secs(5), async {
            while !backend.held.lock().unwrap().is_empty() {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("the previous request's choices are released");
        let reading = server
            .open(completion(json!({
                "model": MODEL, "prompt": "x", "max_tokens": 4, "stream": true, "n": 2,
            })))
            .await;
        backend.wait_admitted(2 * (request + 1)).await;
        // Choice 1 stays open. Choice 0 counts its token before it fails, as
        // a token another choice generated may not be counted yet.
        backend.emit_to(0, content("hi")).await;
        backend
            .emit_to(
                0,
                GenerationEvent::Failed {
                    message: "device fault".to_owned(),
                    code: Some("device_fault".to_owned()),
                },
            )
            .await;
        let body = String::from_utf8_lossy(&reading.await.unwrap()).into_owned();
        assert!(body.contains("device_fault"), "{body}");
    }

    let lines = server.lines(REQUESTS).await;

    for line in &lines {
        assert_failure(line, 500, "error");
        assert_eq!(line["code"], "device_fault");
        assert!(
            line["request_id"].as_str().unwrap().ends_with("_0"),
            "{line:#}"
        );
        assert_eq!(line["generated_tokens"], 1);
    }
}

/// A client that leaves, whether waiting for a whole answer, reading a
/// stream or still sending its body, ends its request with a cancel line,
/// not a failure. A body whose framing is wrong is a failure.
#[tokio::test]
async fn a_client_that_leaves_is_a_cancel_and_bad_framing_a_failure() {
    let backend = HoldingBackend::new();
    let server = LoggedServer::start(backend.clone(), ServerConfig::default()).await;
    let waiting = TcpStream::connect(server.server.address).await.unwrap();
    let request = completion(json!({"model": MODEL, "prompt": "x", "max_tokens": 4}));
    waiting.writable().await.unwrap();
    assert_eq!(
        waiting.try_write(request.as_bytes()).unwrap(),
        request.len()
    );
    backend.wait_admitted(1).await;
    tokio::time::sleep(Duration::from_millis(25)).await;
    drop(waiting);
    let lines = server.lines(1).await;
    assert_cancel(&lines[0], "client_disconnected");

    let reading = TcpStream::connect(server.server.address).await.unwrap();
    let request =
        completion(json!({"model": MODEL, "prompt": "x", "max_tokens": 4, "stream": true}));
    reading.writable().await.unwrap();
    assert_eq!(
        reading.try_write(request.as_bytes()).unwrap(),
        request.len()
    );
    backend.wait_admitted(2).await;
    reading.readable().await.unwrap();
    drop(reading);
    for _ in 0..3 {
        backend.emit(content("hi")).await;
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let lines = server.lines(2).await;
    assert_cancel(&lines[1], "client_disconnected");
    assert!(lines[1]["request_id"].as_str().unwrap().ends_with("_0"));

    let uploading = TcpStream::connect(server.server.address).await.unwrap();
    uploading.writable().await.unwrap();
    uploading
        .try_write(
            b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
              Content-Length: 1000\r\nConnection: close\r\n\r\n{\"model\"",
        )
        .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    drop(uploading);
    let lines = server.lines(3).await;
    assert_cancel(&lines[2], "client_disconnected");

    let framing = server
        .request(
            "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
             Transfer-Encoding: chunked\r\nConnection: close\r\n\r\nzz\r\n{}\r\n0\r\n\r\n",
        )
        .await;
    assert_eq!(framing.status, 400, "{}", framing.body);
    let lines = server.lines(4).await;
    assert_failure(&lines[3], 400, "warn");
    assert_eq!(lines[3]["message"], "request body could not be read");
}

/// Requests the drain deadline cuts off, and those refused while draining,
/// are warnings: a stop is not a fault.
#[tokio::test]
async fn requests_the_drain_ends_are_warnings() {
    let backend = HoldingBackend::new();
    let mut server = LoggedServer::start_with(
        backend.clone(),
        ServerConfig::new(8, 8, 65536, 4096)
            .unwrap()
            .with_responses_store(8, 1 << 20),
        runtime_server::HttpOptions::default(),
        Duration::from_millis(300),
        runtime_server::metrics_snapshot_provider(|_| String::new()),
    )
    .await;
    let streamed = server
        .open(completion(
            json!({"model": MODEL, "prompt": "x", "max_tokens": 4, "stream": true}),
        ))
        .await;
    let whole = server
        .open(completion(
            json!({"model": MODEL, "prompt": "x", "max_tokens": 4}),
        ))
        .await;
    let response = server
        .open(post(
            "/v1/responses",
            &json!({"input": "x", "stream": true}).to_string(),
        ))
        .await;
    let background = server
        .request(&post(
            "/v1/responses",
            &json!({"input": "x", "background": true}).to_string(),
        ))
        .await;
    assert_eq!(background.status, 200, "{}", background.body);
    backend.wait_admitted(4).await;
    let uploading = TcpStream::connect(server.server.address).await.unwrap();
    uploading.writable().await.unwrap();
    uploading
        .try_write(
            b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
              Content-Length: 1000\r\nConnection: close\r\n\r\n{\"model\"",
        )
        .unwrap();
    timeout(Duration::from_secs(5), async {
        while server.request(&get("/health")).await.json()["waiting"]["intake"] != 1 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the upload reaches intake");
    let _ = server.shutdown.take().unwrap().send(());
    tokio::time::sleep(Duration::from_millis(50)).await;
    let refused = server
        .request(&completion(json!({"model": MODEL, "prompt": "x"})))
        .await;
    assert_eq!(refused.status, 503);
    // A draining server's readiness answer is the answer a probe asked for.
    let probed = server.request(&get("/health")).await;
    assert_eq!(probed.status, 503, "{}", probed.body);
    // An upload whose client leaves after the drain deadline was cut off by
    // the stop, not cancelled by its client.
    timeout(Duration::from_secs(5), async {
        while !server.raw().contains("shutdown_timeout") {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the drain deadline passes");
    drop(uploading);
    for open in [streamed, whole, response] {
        let _ = timeout(Duration::from_secs(5), open).await.unwrap();
    }

    let lines = server.lines(6).await;

    assert_failure(&lines[0], 503, "warn");
    assert_eq!(lines[0]["message"], "server is draining");
    for line in &lines[1..] {
        assert_failure(line, 503, "warn");
        assert_eq!(line["code"], "shutdown_timeout", "{line:#}");
    }
    let paths: Vec<_> = lines[1..].iter().map(|line| line["path"].clone()).collect();
    assert_eq!(
        paths
            .iter()
            .filter(|path| **path == "/v1/responses")
            .count(),
        2,
        "{lines:#?}"
    );
    assert_eq!(
        lines[1..]
            .iter()
            .filter(|line| line["stage"] == "intake")
            .count(),
        1,
        "{lines:#?}"
    );
    timeout(Duration::from_secs(5), &mut server.task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

/// A refusal during preparation still names the model and the prompt's
/// length once preparation knows them.
#[tokio::test]
async fn a_context_length_refusal_names_the_model_and_prompt_length() {
    let server = LoggedServer::start(HoldingBackend::new(), ServerConfig::default()).await;
    let refused = server
        .request(&completion(json!({
            "model": MODEL, "prompt": CANARY, "max_tokens": MAXIMUM_TOTAL_TOKENS - 3,
        })))
        .await;
    assert_eq!(refused.status, 400, "{}", refused.body);

    let lines = server.lines(1).await;

    assert_failure(&lines[0], 400, "warn");
    assert_eq!(lines[0]["stage"], "preparation");
    assert_eq!(lines[0]["message"], "context length exceeded");
    assert_eq!(lines[0]["model"], MODEL);
    assert_eq!(lines[0]["prompt_tokens"], 4);
    assert!(!server.raw().contains(CANARY), "{}", server.raw());
}

/// A readiness probe that hears "not ready" writes no line, but a refused
/// method or a refused CORS preflight on the same paths does.
#[tokio::test]
async fn only_a_readiness_answer_on_its_paths_is_not_a_failure() {
    let backend = HoldingBackend::new();
    *backend.readiness.lock().unwrap() = BackendReadiness::NotReady("loading".to_owned());
    let server = LoggedServer::start_with(
        backend.clone(),
        ServerConfig::default(),
        runtime_server::HttpOptions {
            enable_cors: true,
            allowed_origins: vec!["https://allowed.example".to_owned()],
            ..Default::default()
        },
        Duration::from_secs(1),
        runtime_server::metrics_snapshot_provider(|_| String::new()),
    )
    .await;
    for probe in [get("/health"), get("/ping"), post("/ping", "")] {
        let answered = server.request(&probe).await;
        assert_eq!(answered.status, 503, "{probe}: {}", answered.body);
    }
    let wrong_method = server.request(&post("/health", "")).await;
    assert_eq!(wrong_method.status, 405);
    let preflight = server
        .request(
            "OPTIONS /ping HTTP/1.1\r\nHost: localhost\r\nOrigin: https://other.example\r\n\
             Access-Control-Request-Method: GET\r\nConnection: close\r\n\r\n",
        )
        .await;
    assert_eq!(preflight.status, 400, "{}", preflight.body);

    let lines = server.lines(2).await;

    assert_failure(&lines[0], 405, "warn");
    assert_eq!(lines[0]["path"], "/health");
    assert_failure(&lines[1], 400, "warn");
    assert_eq!(lines[1]["path"], "/ping");
}

/// A worker that stops ends every open request with an error line, though no
/// handler answers them.
#[tokio::test]
async fn a_worker_failure_writes_an_error_line_for_each_open_request() {
    let backend = HoldingBackend::new();
    let mut server = LoggedServer::start(backend.clone(), ServerConfig::default()).await;
    let _streamed = server
        .open(completion(
            json!({"model": MODEL, "prompt": "x", "max_tokens": 4, "stream": true}),
        ))
        .await;
    let _whole = server
        .open(completion(
            json!({"model": MODEL, "prompt": "x", "max_tokens": 4}),
        ))
        .await;
    backend.wait_admitted(2).await;
    backend.emit(content("hi")).await;
    // The handlers count the token before the worker fails.
    tokio::time::sleep(Duration::from_millis(200)).await;
    *backend.readiness.lock().unwrap() = BackendReadiness::Terminated("device lost".to_owned());

    let error = timeout(Duration::from_secs(5), &mut server.task)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();

    assert_eq!(error.to_string(), "device lost");
    let lines = server.lines(2).await;
    for line in &lines {
        assert_failure(line, 503, "error");
        assert_eq!(line["message"], "generation worker terminated");
        assert_eq!(line["code"], "worker_unavailable");
        assert_eq!(line["generated_tokens"], 1, "{line:#}");
        assert!(
            line["request_id"].as_str().unwrap().ends_with("_0"),
            "{line:#}"
        );
    }
}

/// Each way a Responses request ends writes one line of its kind, from the
/// adapter, never a second one from the chat stream inside it.
#[tokio::test]
async fn each_responses_outcome_writes_one_line_of_its_kind() {
    let server = LoggedServer::start(
        ScriptedBackendV1::new(vec![
            content("hi"),
            GenerationEvent::Failed {
                message: format!("device fault near {CANARY}"),
                code: Some("device_fault".to_owned()),
            },
        ]),
        ServerConfig::new(8, 8, 65536, 4096)
            .unwrap()
            .with_responses_store(8, 1 << 20),
    )
    .await;
    let failed = server
        .request(&post(
            "/v1/responses",
            &json!({"input": CANARY, "background": true}).to_string(),
        ))
        .await;
    assert_eq!(failed.status, 200, "{}", failed.body);
    let lines = server.lines(1).await;
    assert_failure(&lines[0], 500, "error");
    assert_eq!(lines[0]["code"], "device_fault");
    assert_eq!(lines[0]["path"], "/v1/responses");
    assert!(lines[0]["request_id"].as_str().unwrap().ends_with("_0"));
    assert!(!server.raw().contains(CANARY));

    let backend = HoldingBackend::new();
    let server = LoggedServer::start(
        backend.clone(),
        ServerConfig::new(8, 8, 65536, 4096)
            .unwrap()
            .with_responses_store(8, 1 << 20),
    )
    .await;
    let queued = server
        .request(&post(
            "/v1/responses",
            &json!({"input": "x", "background": true}).to_string(),
        ))
        .await
        .json();
    backend.wait_admitted(1).await;
    let id = queued["id"].as_str().unwrap();
    let cancel = server
        .request(&post(&format!("/v1/responses/{id}/cancel"), "{}"))
        .await;
    assert_eq!(cancel.status, 200, "{}", cancel.body);
    let lines = server.lines(1).await;
    assert_cancel(&lines[0], "cancelled");

    let reading = TcpStream::connect(server.server.address).await.unwrap();
    let request = post(
        "/v1/responses",
        &json!({"input": "x", "stream": true}).to_string(),
    );
    reading.writable().await.unwrap();
    assert_eq!(
        reading.try_write(request.as_bytes()).unwrap(),
        request.len()
    );
    backend.wait_admitted(2).await;
    reading.readable().await.unwrap();
    drop(reading);
    for _ in 0..3 {
        backend.emit(content("hi")).await;
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let lines = server.lines(2).await;
    assert_cancel(&lines[1], "client_disconnected");

    let flood = (0..55)
        .map(|_| content(&"x".repeat(128 * 1024)))
        .chain([finished()])
        .collect();
    let server = LoggedServer::start(
        ScriptedBackendV1::new(flood),
        ServerConfig::new(8, 8, 65536, 16 << 20)
            .unwrap()
            .with_stream_write_timeout(Duration::from_millis(200)),
    )
    .await;
    let stalled = TcpStream::connect(server.server.address).await.unwrap();
    let request = post(
        "/v1/responses",
        &json!({"input": "x", "stream": true, "max_output_tokens": 60}).to_string(),
    );
    stalled.writable().await.unwrap();
    assert_eq!(
        stalled.try_write(request.as_bytes()).unwrap(),
        request.len()
    );
    let lines = server.lines(1).await;
    assert_failure(&lines[0], 408, "warn");
    assert_eq!(lines[0]["error_type"], "client_write_timeout");
    drop(stalled);
}

/// Requests the router never saw are logged at the transport stage, without
/// a method or path; a connection that never sent a byte is not a request.
#[tokio::test]
async fn requests_the_router_never_saw_are_logged_at_the_transport() {
    let server = LoggedServer::start(
        ScriptedBackendV1::new(vec![]),
        ServerConfig::default().with_header_read_timeout(Duration::from_millis(150)),
    )
    .await;
    let malformed = server.request("NOT A REQUEST\r\n\r\n").await;
    assert_eq!(malformed.status, 400);
    let idle = TcpStream::connect(server.server.address).await.unwrap();
    let partial = TcpStream::connect(server.server.address).await.unwrap();
    partial.writable().await.unwrap();
    partial
        .try_write(b"GET /health HTTP/1.1\r\nHost: localhost\r\nX-Stall: ")
        .unwrap();
    let _ = timeout(Duration::from_secs(2), exchange(&partial, b""))
        .await
        .unwrap();
    let _ = timeout(Duration::from_secs(2), exchange(&idle, b""))
        .await
        .unwrap();

    let lines = server.lines(2).await;

    for line in &lines {
        assert_eq!(line["stage"], "transport", "{line:#}");
        assert_eq!(line["method"], Value::Null);
        assert_eq!(line["path"], Value::Null);
        assert_eq!(line["level"], "warn");
    }
    assert_eq!(lines[0]["status"], 400);
    assert_eq!(lines[0]["message"], "request head did not parse");
    assert_eq!(lines[1]["status"], 408);
    assert_eq!(lines[1]["message"], "request head did not arrive in time");

    // The deadline runs again after each answer on a kept-alive connection:
    // one that stays idle writes no line, one that starts another head and
    // stalls writes one.
    let reused_idle = TcpStream::connect(server.server.address).await.unwrap();
    let reused_partial = TcpStream::connect(server.server.address).await.unwrap();
    for socket in [&reused_idle, &reused_partial] {
        answer_kept_alive(socket, b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n").await;
    }
    reused_partial
        .try_write(b"GET /health HTTP/1.1\r\nHost: localhost\r\nX-Stall: ")
        .unwrap();
    for socket in [&reused_partial, &reused_idle] {
        let _ = timeout(Duration::from_secs(2), exchange(socket, b""))
            .await
            .unwrap();
    }

    let lines = server.lines(3).await;

    assert_eq!(lines[2]["stage"], "transport", "{:#}", lines[2]);
    assert_eq!(lines[2]["status"], 408);
    assert_eq!(lines[2]["message"], "request head did not arrive in time");

    // A refusal that never read its body leaves hyper to drain the rest of
    // it, past its first read, after the answer. The drained bytes are not a
    // head, so the idle connection the deadline then closes writes no line.
    let unread = TcpStream::connect(server.server.address).await.unwrap();
    let body = "x".repeat(9_000);
    let request = format!(
        "POST /v1/nothing HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let answer = timeout(
        Duration::from_secs(2),
        exchange(&unread, request.as_bytes()),
    )
    .await
    .unwrap();
    assert!(String::from_utf8_lossy(&answer).starts_with("HTTP/1.1 404"));

    let lines = server.lines(4).await;

    assert_eq!(lines[3]["status"], 404, "{:#}", lines[3]);

    let limited = LoggedServer::start(
        ScriptedBackendV1::new(vec![]),
        ServerConfig::default().with_max_connections(1),
    )
    .await;
    let _held = TcpStream::connect(limited.server.address).await.unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;
    let _excess = TcpStream::connect(limited.server.address).await.unwrap();
    let lines = limited.lines(1).await;
    assert_failure(&lines[0], 503, "warn");
    assert_eq!(
        lines[0]["message"],
        "connection limit reached; the connection was closed"
    );
}

/// Summary lines report counters as deltas since the last line and gauges
/// as current values.
#[tokio::test]
async fn summaries_report_deltas_and_current_gauges() {
    let server = LoggedServer::start_with(
        ScriptedBackendV1::new(vec![content("hi"), finished()]),
        ServerConfig::default(),
        runtime_server::HttpOptions {
            summary_interval: Some(Duration::from_millis(100)),
            ..Default::default()
        },
        Duration::from_secs(1),
        Arc::new(GrowingMetrics::default()),
    )
    .await;
    for model in [MODEL, MODEL, "missing.model"] {
        server
            .request(&completion(
                json!({"model": model, "prompt": "x", "max_tokens": 4}),
            ))
            .await;
    }
    let summaries = || {
        server
            .written()
            .into_iter()
            .filter(|line| line["schema"] == SUMMARY)
            .collect::<Vec<_>>()
    };
    let total = |lines: &[Value], field: &dyn Fn(&Value) -> &Value| {
        lines
            .iter()
            .map(|line| field(line).as_u64().unwrap())
            .sum::<u64>()
    };
    timeout(Duration::from_secs(5), async {
        loop {
            let lines = summaries();
            if lines.len() >= 2 && total(&lines, &|line| &line["http_requests"]) >= 3 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("summaries count the requests");

    let lines = summaries();

    assert_eq!(total(&lines, &|line| &line["http_requests"]), 3);
    assert_eq!(total(&lines, &|line| &line["failures"]["warn"]), 1);
    assert_eq!(total(&lines, &|line| &line["refusals"]["unknown_model"]), 1);
    for line in &lines {
        assert_eq!(line["requests"]["submitted"], 10, "{line:#}");
        assert_eq!(line["tokens"]["generated"], 100);
        assert_eq!(line["acceptance"], 0.75);
        assert_eq!(line["ttft_seconds"]["count"], 2);
        assert_eq!(line["ttft_seconds"]["p50"], 0.001);
        assert_eq!(line["itl_seconds"]["count"], 0);
        assert_eq!(line["itl_seconds"]["p50"], Value::Null);
        assert_eq!(line["running"], 3);
        assert_eq!(line["waiting"], 2);
        assert_eq!(line["kv"]["usage"], 0.25);
        assert_eq!(line["worker_state"], "ready");
    }
}
