//! The real health route over a frontend that admits no generation.
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use runtime_server::{
    AdmissionError, BackendReadiness, BoxFuture, EventSender, FrontendChatRequestV1,
    FrontendErrorV1, FrontendModelV1, FrontendRenderedChatV1, GenerationBackend, PreparedPrompt,
    PreparedRequest, ServerFrontendV1,
};
use tokio::net::{TcpListener, TcpStream};

struct Frontend(Vec<FrontendModelV1>);

impl ServerFrontendV1 for Frontend {
    fn models(&self) -> &[FrontendModelV1] {
        &self.0
    }
    fn maximum_total_tokens(&self) -> usize {
        65536
    }
    fn maximum_tool_calls(&self) -> usize {
        0
    }
    fn render_chat(
        &self,
        _: FrontendChatRequestV1,
    ) -> Result<FrontendRenderedChatV1, FrontendErrorV1> {
        Err(FrontendErrorV1::new("health fixture", None))
    }
    fn validate_tool_call(&self, _: &str, _: &str) -> Result<(), FrontendErrorV1> {
        Err(FrontendErrorV1::new("health fixture", None))
    }
}

struct Backend;

impl GenerationBackend for Backend {
    fn readiness(&self) -> BackendReadiness {
        BackendReadiness::Ready
    }
    fn maximum_sequence_tokens(&self) -> usize {
        65536
    }
    fn prompt_token_count(&self, _: &PreparedPrompt) -> Result<usize, AdmissionError> {
        Err(AdmissionError::InvalidPrompt)
    }
    fn submit(
        &self,
        _: PreparedRequest,
        _: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        Box::pin(async { Err(AdmissionError::WorkerUnavailable) })
    }
}

pub(super) async fn read(options: BTreeMap<String, serde_json::Value>) -> serde_json::Value {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let address = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(runtime_server::serve_with_shutdown_and_http_options(
        listener,
        Arc::new(Backend),
        Arc::new(Frontend(vec![FrontendModelV1::new(
            "fixture.prefill.v1",
            "fixture",
        )])),
        runtime_server::ServerConfig::default(),
        Arc::new(runtime_server::SystemMonotonicClock::new()),
        runtime_server::metrics_snapshot_provider(|_| String::new()),
        async move {
            let _ = stopped.await;
        },
        Duration::from_secs(1),
        runtime_server::HttpOptions {
            serving_identity: Some(runtime_server::ServingIdentity {
                bundle_id: "fixture.prefill.v1".into(),
                model_options: options,
                ..Default::default()
            }),
            ..Default::default()
        },
    ));
    let bytes = tokio::time::timeout(Duration::from_secs(5), async {
        let stream = TcpStream::connect(address).await.unwrap();
        let mut request =
            b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".as_slice();
        while !request.is_empty() {
            stream.writable().await.unwrap();
            match stream.try_write(request) {
                Ok(0) => panic!("health connection closed"),
                Ok(written) => request = &request[written..],
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(error) => panic!("{error}"),
            }
        }
        let mut bytes = Vec::new();
        loop {
            stream.readable().await.unwrap();
            let mut chunk = [0_u8; 4096];
            match stream.try_read(&mut chunk) {
                Ok(0) => break bytes,
                Ok(read) => bytes.extend_from_slice(&chunk[..read]),
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(error) => panic!("{error}"),
            }
        }
    })
    .await;
    stop.send(()).unwrap();
    task.await.unwrap().unwrap();
    let response = String::from_utf8(bytes.unwrap()).unwrap();
    let (headers, body) = response.split_once("\r\n\r\n").unwrap();
    assert!(headers.starts_with("HTTP/1.1 200"), "{headers}");
    serde_json::from_str(body).unwrap()
}
