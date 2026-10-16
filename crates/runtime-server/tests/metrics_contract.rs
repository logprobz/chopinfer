use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use runtime_engine::{
    Engine, EngineConfig, EngineMetrics, EngineRequest, ExecutionObservations, ExecutorError,
    LogitsRow, ModelBatchItem, ModelExecutor, ModelTransaction, NoopModelTransactionState,
};
use runtime_kv::{KvCachePrecision, KvGeometry, KvMetrics, PreparedAppendBatch};
use runtime_model_api::ModelVocabulary;
use runtime_sampling::SamplingParameters;
use runtime_scheduler::{RequestId, SchedulerConfig};
use runtime_server::{
    AdmissionError, BackendReadiness, BoxFuture, EventSender, FrontendChatRequestV1,
    FrontendErrorV1, FrontendModelV1, FrontendRenderedChatV1, GenerationBackend,
    MetricsSnapshotProvider, MonotonicClock, PreparedPrompt, PreparedRequest,
    RequestAcceptanceProfile, ServerCapacityProfile, ServerConfig, ServerFrontendV1,
    metrics_snapshot_provider, serve_with_metrics,
};
use tokio::net::{TcpListener, TcpStream};

#[derive(Debug)]
struct FixedClock(u64);

#[test]
fn c1_c4_c16_c32_profiles_bound_admission_and_keep_streaming_enabled() {
    for (name, concurrency) in [("C1", 1), ("C4", 4), ("C16", 16), ("C32", 32)] {
        let profile = ServerCapacityProfile::new(name, concurrency, concurrency).unwrap();
        assert!(profile.streaming_enabled());
        let config = ServerConfig::default().with_capacity_profile(profile);
        assert_eq!(config.max_in_flight(), concurrency);
        assert_eq!(config.event_buffer_capacity(), concurrency);
        assert_eq!(
            config.acceptance_profile(),
            RequestAcceptanceProfile::ProfiledStreaming { name }
        );
        assert_eq!(config.acceptance_profile().profile_name(), Some(name));
        assert!(config.acceptance_profile().streaming_allowed());
    }
}

impl MonotonicClock for FixedClock {
    fn now_micros(&self) -> u64 {
        self.0
    }
}

impl runtime_engine::MonotonicClock for FixedClock {
    fn now_micros(&self) -> u64 {
        self.0
    }
}

#[derive(Debug, Default)]
struct ObservedExecutor;

impl ModelExecutor for ObservedExecutor {
    type TransactionState<'a> = NoopModelTransactionState;

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        _kv_plan: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        let rows = items
            .iter()
            .filter(|item| item.expects_logits())
            .map(|item| {
                let mut logits = vec![0.0; 8];
                logits[2] = 1.0;
                LogitsRow::new(item.request_id(), logits)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(
            ModelTransaction::no_op(rows).with_observations(ExecutionObservations {
                kernel_launches: 3,
                kernel_micros: 20,
                collective_operations: 2,
                collective_micros: 7,
            }),
        )
    }
}

#[derive(Debug)]
struct RecordingProvider {
    observed_micros: AtomicU64,
}

impl MetricsSnapshotProvider for RecordingProvider {
    fn snapshot(&self, monotonic_micros: u64) -> String {
        self.observed_micros
            .store(monotonic_micros, Ordering::Relaxed);
        "# TYPE chop_requests_active gauge\nchop_requests_active 3\n".to_owned()
    }
}

#[derive(Debug)]
struct FakeBackend;

impl GenerationBackend for FakeBackend {
    fn readiness(&self) -> BackendReadiness {
        BackendReadiness::Ready
    }

    fn maximum_sequence_tokens(&self) -> usize {
        16
    }

    fn prompt_token_count(&self, _prompt: &PreparedPrompt) -> Result<usize, AdmissionError> {
        Ok(1)
    }

    fn submit(
        &self,
        _request: PreparedRequest,
        _events: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        Box::pin(async { Ok(()) })
    }
}

#[derive(Debug)]
struct FakeFrontend {
    models: Vec<FrontendModelV1>,
}

impl FakeFrontend {
    fn new() -> Self {
        Self {
            models: vec![FrontendModelV1::new("fixture", "tests")],
        }
    }
}

impl ServerFrontendV1 for FakeFrontend {
    fn models(&self) -> &[FrontendModelV1] {
        &self.models
    }

    fn maximum_total_tokens(&self) -> usize {
        16
    }

    fn maximum_tool_calls(&self) -> usize {
        1
    }

    fn render_chat(
        &self,
        _request: FrontendChatRequestV1,
    ) -> Result<FrontendRenderedChatV1, FrontendErrorV1> {
        unreachable!("the metrics route does not render chat")
    }

    fn validate_tool_call(&self, _name: &str, _arguments: &str) -> Result<(), FrontendErrorV1> {
        Ok(())
    }
}

async fn fetch_metrics(address: std::net::SocketAddr) -> String {
    exchange(
        address,
        b"GET /metrics HTTP/1.1\r\nHost: fixture\r\nConnection: close\r\n\r\n",
    )
    .await
}

async fn exchange(address: std::net::SocketAddr, request: &[u8]) -> String {
    let stream = TcpStream::connect(address).await.unwrap();
    stream.writable().await.unwrap();
    stream.try_write(request).unwrap();
    let mut response = Vec::new();
    loop {
        stream.readable().await.unwrap();
        let mut bytes = [0_u8; 4096];
        match stream.try_read(&mut bytes) {
            Ok(0) => break,
            Ok(count) => response.extend_from_slice(&bytes[..count]),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(error) => panic!("metrics response read failed: {error}"),
        }
    }
    String::from_utf8(response).unwrap()
}

#[tokio::test]
async fn metrics_endpoint_uses_a_snapshot_provider_and_monotonic_clock() {
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .await
        .unwrap();
    let address = listener.local_addr().unwrap();
    let provider = Arc::new(RecordingProvider {
        observed_micros: AtomicU64::new(0),
    });
    let provider_for_server: Arc<dyn MetricsSnapshotProvider> = provider.clone();
    tokio::spawn(async move {
        serve_with_metrics(
            listener,
            Arc::new(FakeBackend),
            Arc::new(FakeFrontend::new()),
            ServerConfig::default(),
            Arc::new(FixedClock(12_345_678)),
            provider_for_server,
        )
        .await
        .unwrap();
    });

    let response = fetch_metrics(address).await;
    assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
    assert!(response.contains("content-type: text/plain; version=0.0.4; charset=utf-8"));
    assert!(response.contains("cache-control: no-store"));
    assert!(response.ends_with("chop_requests_active 3\n"));
    assert_eq!(provider.observed_micros.load(Ordering::Relaxed), 12_345_678);
}

#[tokio::test]
async fn composite_endpoint_exposes_engine_and_scheduler_owned_fp8_kv_metrics() {
    let geometry = KvGeometry::new(4, 4).unwrap();
    let config = EngineConfig::new(
        SchedulerConfig::new(1, 2, 4, 4, 1).unwrap(),
        geometry,
        ModelVocabulary::new(8).unwrap(),
        geometry.token_capacity(),
        1,
    )
    .unwrap();
    let engine_metrics = EngineMetrics::default();
    let kv_metrics = KvMetrics::new(KvCachePrecision::Fp8);
    let mut engine = Engine::new_with_runtime_metrics(
        config,
        ObservedExecutor,
        Arc::new(FixedClock(100)),
        engine_metrics.clone(),
        kv_metrics.clone(),
    )
    .unwrap();
    engine
        .submit(
            EngineRequest::new(
                RequestId::new(1),
                vec![1],
                2,
                SamplingParameters::new(0.0).unwrap(),
                1,
                Vec::new(),
                ModelVocabulary::new(8).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    engine.step().unwrap();
    let served_kv_metrics = engine.kv_metrics();
    assert_eq!(served_kv_metrics.snapshot(), kv_metrics.snapshot());
    assert_eq!(served_kv_metrics.snapshot().fp8_tokens, 1);

    let provider = metrics_snapshot_provider(move |_monotonic_micros| {
        let mut output = engine_metrics.snapshot().prometheus();
        output.push_str(&served_kv_metrics.snapshot().prometheus());
        output
    });
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .await
        .unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve_with_metrics(
            listener,
            Arc::new(FakeBackend),
            Arc::new(FakeFrontend::new()),
            ServerConfig::default(),
            Arc::new(FixedClock(200)),
            provider,
        )
        .await
        .unwrap();
    });
    let response = fetch_metrics(address).await;
    for name in [
        "chop_requests_submitted_total",
        "chop_prompt_tokens_total",
        "chop_generation_tokens_total",
        "chop_queue_duration_count",
        "chop_sequences_active",
        "chop_batches_total",
        "chop_ttft_count",
        "chop_itl_count",
        "chop_kernel_launches_total",
        "chop_collective_operations_total",
        "chop_errors_total",
        "chop_cancellations_total",
        "chop_fp8_kv_tokens",
        "chop_fp8_kv_capacity_tokens",
        "chop_explicit_shared_prefix_forks_total",
        "chop_cow_copies_total",
    ] {
        assert!(response.contains(name), "missing metric {name}");
    }
    assert!(response.contains("chop_requests_active 1"));
    assert!(response.contains("chop_fp8_kv_tokens 1"));
    assert!(response.contains("chop_kernel_launches_total 3"));
    assert!(response.contains("chop_collective_operations_total 2"));
}

#[tokio::test]
async fn refused_http_traffic_updates_bounded_admission_counters_and_releases_intake() {
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .await
        .unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        runtime_server::serve(
            listener,
            Arc::new(FakeBackend),
            Arc::new(FakeFrontend::new()),
            ServerConfig::default(),
        )
        .await
        .unwrap();
    });
    for _ in 0..2 {
        let response = exchange(address, b"POST /v1/completions HTTP/1.1\r\nHost: fixture\r\nContent-Type: application/json\r\nContent-Length: 1\r\nConnection: close\r\n\r\n{").await;
        assert!(response.starts_with("HTTP/1.1 400"), "{response}");
    }
    let body = r#"{"model":"missing","prompt":"hello"}"#;
    let request = format!(
        "POST /v1/completions HTTP/1.1\r\nHost: fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let unknown = exchange(address, request.as_bytes()).await;
    assert!(unknown.starts_with("HTTP/1.1 404"), "{unknown}");
    for request in [
        b"GET /v1/responses/missing HTTP/1.1\r\nHost: fixture\r\nConnection: close\r\n\r\n"
            .as_slice(),
        b"POST /v1/responses/missing/cancel HTTP/1.1\r\nHost: fixture\r\nConnection: close\r\n\r\n"
            .as_slice(),
    ] {
        let missing = exchange(address, request).await;
        assert!(missing.starts_with("HTTP/1.1 404"), "{missing}");
        assert!(missing.contains("Response with id 'missing' not found."));
    }
    let metrics = fetch_metrics(address).await;
    assert!(metrics.contains("chop_admission_refusals_total{reason=\"invalid_request\"} 2"));
    assert!(metrics.contains("chop_admission_refusals_total{reason=\"unknown_model\"} 1"));
    assert!(metrics.contains("chop_admission_refusals_total{reason=\"not_found\"} 2"));
    assert!(metrics.contains("chop_intake_used_bytes 0\n"));
    assert!(metrics.contains("chop_intake_budget_bytes 1073741824\n"));
    let disabled = exchange(
        address,
        b"GET /debug/step-profile HTTP/1.1\r\nHost: fixture\r\nConnection: close\r\n\r\n",
    )
    .await;
    assert!(disabled.starts_with("HTTP/1.1 404"));
    server.abort();
    assert!(server.await.unwrap_err().is_cancelled());
}

struct ProfileProvider;
impl MetricsSnapshotProvider for ProfileProvider {
    fn snapshot(&self, _: u64) -> String {
        String::new()
    }
    fn step_profile(&self) -> Option<String> {
        Some(
            r#"{"schema":"chopinfer.step_profile.v1","every":2,"capacity":2,"samples":[]}"#
                .to_owned(),
        )
    }
}

#[tokio::test]
async fn enabled_profile_endpoint_serves_provider_json_without_caching() {
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .await
        .unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        serve_with_metrics(
            listener,
            Arc::new(FakeBackend),
            Arc::new(FakeFrontend::new()),
            ServerConfig::default(),
            Arc::new(FixedClock(0)),
            Arc::new(ProfileProvider),
        )
        .await
        .unwrap();
    });
    let response = exchange(
        address,
        b"GET /debug/step-profile HTTP/1.1\r\nHost: fixture\r\nConnection: close\r\n\r\n",
    )
    .await;
    assert!(response.starts_with("HTTP/1.1 200"));
    assert!(response.contains("content-type: application/json"));
    assert!(response.contains("cache-control: no-store"));
    let json: serde_json::Value =
        serde_json::from_str(response.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(json["schema"], "chopinfer.step_profile.v1");
    assert_eq!(json["every"], 2);
    server.abort();
    assert!(server.await.unwrap_err().is_cancelled());
}
