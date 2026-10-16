//! The served descriptor supplies omitted sampling controls without weights.
use std::sync::{Arc, Mutex};

use runtime_server::{
    AdmissionError, BackendReadiness, BoxFuture, EventSender, GenerationBackend, PreparedPrompt,
    PreparedRequest, ServerConfig, serve,
};

#[derive(Default)]
struct Capture(Mutex<Vec<PreparedRequest>>);
impl GenerationBackend for Capture {
    fn readiness(&self) -> BackendReadiness {
        BackendReadiness::Ready
    }
    fn maximum_sequence_tokens(&self) -> usize {
        262_144
    }
    fn prompt_token_count(&self, _: &PreparedPrompt) -> Result<usize, AdmissionError> {
        Ok(1)
    }
    fn submit(
        &self,
        request: PreparedRequest,
        _: EventSender,
    ) -> BoxFuture<'_, Result<(), AdmissionError>> {
        self.0.lock().unwrap().push(request);
        Box::pin(async { Err(AdmissionError::Rejected("captured".into())) })
    }
}

#[tokio::test]
async fn omitted_top_k_comes_from_the_checkpoint_declaration() {
    let backend = Arc::new(Capture::default());
    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .await
        .unwrap();
    let address = listener.local_addr().unwrap();
    let capture = backend.clone();
    let server = tokio::spawn(async move {
        serve(
            listener,
            capture,
            Arc::new(
                runtime_frontend::BundleFrontendV1::from_export(
                    &model_qwen38_contract::frontend::export(),
                    model_qwen38::frontend::COMPONENT,
                )
                .unwrap(),
            ),
            ServerConfig::default(),
        )
        .await
        .unwrap();
    });
    for (endpoint, input) in [
        ("/v1/completions", r#""prompt":"hello""#),
        (
            "/v1/chat/completions",
            r#""messages":[{"role":"user","content":"hello"}]"#,
        ),
    ] {
        for (field, expected) in [
            ("", 20),
            (",\"top_k\":null", 20),
            (",\"top_k\":0", 0),
            (",\"top_k\":-1", 0),
            (",\"top_k\":7", 7),
        ] {
            let body = format!(r#"{{{input},"max_tokens":1{field}}}"#);
            let stream = tokio::net::TcpStream::connect(address).await.unwrap();
            let request = format!(
                "POST {endpoint} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            exchange(&stream, request.as_bytes()).await;
            let requests = backend.0.lock().unwrap();
            let request = requests.last().expect("request reached capture");
            assert_eq!(request.sampling.top_k(), expected);
            assert_eq!(request.sampling.temperature(), 1.0);
            assert_eq!(request.sampling.top_p(), 0.95);
        }
    }
    server.abort();
}

async fn exchange(stream: &tokio::net::TcpStream, mut request: &[u8]) -> Vec<u8> {
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
