use super::*;
use std::sync::Mutex;

struct AdmittedStream {
    events: EventSender,
    cancellation: CancellationSignal,
}

struct LifecycleBackend {
    state: Mutex<BackendReadiness>,
    scripted: Arc<ScriptedBackendV1>,
    control: Mutex<Option<tokio::sync::oneshot::Sender<AdmittedStream>>>,
}

impl LifecycleBackend {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(BackendReadiness::Ready),
            scripted: ScriptedBackendV1::new(vec![content("hi"), finished()]),
            control: Mutex::new(None),
        })
    }

    fn set(&self, state: BackendReadiness) {
        *self.state.lock().unwrap() = state;
    }
}

impl GenerationBackend for LifecycleBackend {
    fn readiness(&self) -> BackendReadiness {
        self.state.lock().unwrap().clone()
    }

    fn maximum_sequence_tokens(&self) -> usize {
        self.scripted.maximum_sequence_tokens()
    }

    fn prompt_token_count(&self, prompt: &PreparedPrompt) -> Result<usize, AdmissionError> {
        self.scripted.prompt_token_count(prompt)
    }

    fn submit(
        &self,
        request: PreparedRequest,
        events: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        if self.readiness() != BackendReadiness::Ready {
            return Box::pin(async { Err(AdmissionError::WorkerUnavailable) });
        }
        if let Some(control) = self.control.lock().unwrap().take() {
            assert!(
                control
                    .send(AdmittedStream {
                        events,
                        cancellation: request.cancellation,
                    })
                    .is_ok()
            );
            return Box::pin(async { Ok(()) });
        }
        self.scripted.submit(request, events)
    }
}

async fn supervised(
    backend: Arc<dyn GenerationBackend>,
) -> (
    TestServerV1,
    tokio::sync::oneshot::Sender<()>,
    tokio::task::JoinHandle<std::io::Result<()>>,
) {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let address = listener.local_addr().unwrap();
    let (shutdown, stopped) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(runtime_server::serve_with_shutdown(
        listener,
        backend,
        Arc::new(FakeFrontendV1::new()),
        ServerConfig::default(),
        Arc::new(runtime_server::SystemMonotonicClock::new()),
        runtime_server::metrics_snapshot_provider(|_| String::new()),
        async {
            let _ = stopped.await;
        },
        Duration::from_secs(1),
    ));
    (TestServerV1 { address }, shutdown, task)
}

#[tokio::test]
async fn temporary_unreadiness_keeps_the_listener_alive_until_recovery() {
    let backend = LifecycleBackend::new();
    let (server, shutdown, mut task) = supervised(backend.clone()).await;
    assert_eq!(server.request(&get("/health")).await.status, 200);
    // This accepted connection must survive the stall too.
    let held = TcpStream::connect(server.address).await.unwrap();
    backend.set(BackendReadiness::NotReady("transfer pending".to_owned()));
    assert!(
        timeout(Duration::from_millis(80), &mut task).await.is_err(),
        "temporary unreadiness terminated serving"
    );
    assert_eq!(server.request(&get("/health")).await.status, 503);
    let request = post(
        "/v1/completions",
        r#"{"model":"fixture.model.v1","prompt":"x","max_tokens":1}"#,
    );
    assert_eq!(server.request(&request).await.status, 503);
    assert_eq!(backend.scripted.submissions.load(Ordering::Relaxed), 0);
    backend.set(BackendReadiness::Ready);
    let response = timeout(
        Duration::from_secs(2),
        exchange(&held, get("/health").as_bytes()),
    )
    .await
    .unwrap();
    assert!(response.starts_with(b"HTTP/1.1 200"));
    assert_eq!(server.request(&get("/health")).await.status, 200);
    assert_eq!(server.request(&request).await.status, 200);
    shutdown.send(()).unwrap();
    timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn terminal_failure_ends_serving_without_waiting_for_shutdown() {
    let backend = LifecycleBackend::new();
    let (server, _shutdown, task) = supervised(backend.clone()).await;
    assert_eq!(server.request(&get("/health")).await.status, 200);
    // The same reason as temporary unreadiness must not change the typed policy.
    backend.set(BackendReadiness::Terminated("transfer pending".to_owned()));
    let error = timeout(Duration::from_secs(1), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert_eq!(error.to_string(), "transfer pending");
    assert!(TcpStream::connect(server.address).await.is_err());
}

#[test]
fn replica_pool_is_terminal_only_when_every_replica_is_terminal() {
    let first = LifecycleBackend::new();
    let second = LifecycleBackend::new();
    let pool = runtime_server::LocalReplicaPool::new(vec![first.clone(), second.clone()]).unwrap();
    first.set(BackendReadiness::Terminated("failed".to_owned()));
    assert_eq!(pool.readiness(), BackendReadiness::Ready);
    second.set(BackendReadiness::NotReady("transfer pending".to_owned()));
    assert!(matches!(pool.readiness(), BackendReadiness::NotReady(_)));
    second.set(BackendReadiness::Ready);
    assert_eq!(pool.readiness(), BackendReadiness::Ready);
    second.set(BackendReadiness::Terminated("stopped".to_owned()));
    assert!(matches!(pool.readiness(), BackendReadiness::Terminated(_)));
}

/// Reads incremental response bytes, refusing EOF before the expected event.
async fn read_through(stream: &TcpStream, response: &mut Vec<u8>, expected: &[u8]) {
    timeout(Duration::from_secs(2), async {
        while !response
            .windows(expected.len())
            .any(|bytes| bytes == expected)
        {
            stream.readable().await.unwrap();
            let mut chunk = [0; 4096];
            match stream.try_read(&mut chunk) {
                Ok(0) => panic!("stream closed before the expected event"),
                Ok(read) => response.extend_from_slice(&chunk[..read]),
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(error) => panic!("stream read failed: {error}"),
            }
        }
    })
    .await
    .expect("the next streaming event arrives");
}

#[tokio::test]
async fn admitted_stream_continues_across_temporary_unreadiness() {
    let backend = LifecycleBackend::new();
    let (control, admitted) = tokio::sync::oneshot::channel();
    *backend.control.lock().unwrap() = Some(control);
    let (server, shutdown, mut task) = supervised(backend.clone()).await;
    let stream = TcpStream::connect(server.address).await.unwrap();
    let request = post(
        "/v1/completions",
        r#"{"model":"fixture.model.v1","prompt":"x","max_tokens":8,"stream":true}"#,
    );
    let mut remaining = request.as_bytes();
    while !remaining.is_empty() {
        stream.writable().await.unwrap();
        match stream.try_write(remaining) {
            Ok(0) => panic!("stream closed before admission"),
            Ok(written) => remaining = &remaining[written..],
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(error) => panic!("stream write failed: {error}"),
        }
    }
    let admitted = timeout(Duration::from_secs(2), admitted)
        .await
        .unwrap()
        .unwrap();
    let mut response = Vec::new();
    admitted.events.emit(content("before")).await.unwrap();
    read_through(&stream, &mut response, br#""text":"before""#).await;
    assert!(response.starts_with(b"HTTP/1.1 200"));
    assert!(!admitted.cancellation.is_cancelled());

    backend.set(BackendReadiness::NotReady("transfer pending".to_owned()));
    assert!(
        timeout(Duration::from_millis(80), &mut task).await.is_err(),
        "temporary unreadiness terminated the active stream"
    );
    assert_eq!(server.request(&get("/health")).await.status, 503);
    assert_eq!(server.request(&request).await.status, 503);
    assert!(!admitted.cancellation.is_cancelled());
    admitted.events.emit(content("during")).await.unwrap();
    read_through(&stream, &mut response, br#""text":"during""#).await;
    assert!(matches!(backend.readiness(), BackendReadiness::NotReady(_)));
    assert!(!admitted.cancellation.is_cancelled());

    backend.set(BackendReadiness::Ready);
    assert_eq!(server.request(&get("/health")).await.status, 200);
    assert_eq!(server.request(&request).await.status, 200);
    admitted.events.emit(content("after")).await.unwrap();
    read_through(&stream, &mut response, br#""text":"after""#).await;
    admitted
        .events
        .emit(GenerationEvent::Finished {
            reason: FinishReason::Stop,
            usage: TokenUsage {
                prompt_tokens: 4,
                completion_tokens: 3,
                cached_tokens: None,
            },
        })
        .await
        .unwrap();
    response.extend(
        timeout(Duration::from_secs(2), exchange(&stream, b""))
            .await
            .unwrap(),
    );
    let response = String::from_utf8(response).unwrap();
    let events: Vec<Value> = response
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    let text: String = events
        .iter()
        .filter_map(|event| event["choices"][0]["text"].as_str())
        .collect();
    assert_eq!(text, "beforeduringafter");
    assert_eq!(
        events.last().unwrap()["choices"][0]["finish_reason"],
        "stop"
    );
    assert_eq!(response.matches("data: [DONE]").count(), 1);
    assert!(!admitted.cancellation.is_cancelled());
    shutdown.send(()).unwrap();
    timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}
