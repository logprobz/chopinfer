//! Exercise parsed serve options, the production option wiring, HTTP validation,
//! and the real preparation backend with a CPU-only executor.
use super::*;
use runtime_frontend::BundleFrontendV1;
use runtime_host::{
    HostCommandV1, configure_media_body_limit_v1, configure_remote_media_v1,
    configure_served_images_v1, parse_host_command_v1,
};
use runtime_server::{ServerConfig, ServerFrontendV1};
use tokio::net::{TcpListener, TcpStream};

const PNG: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";

fn media_declaration() -> FrontendDescriptorV1 {
    let mut declaration = declaration();
    declaration.multimodal = Some(
        serde_json::from_value(serde_json::json!({
            "arena": {"rows":64,"page_rows":64,"hidden":8,"element":"bf16"},
            "encoders": [{"name":"image","payload_element":"bf16","payload_columns":3,
                "maximum_payload_bytes":1024,"cost_per_row_milli":1}],
            "modalities": [{"kind":"image","part_types":["image_url"],"maximum_items":4,
                "maximum_bytes":1024,"maximum_tokens":4,"encoder":0,
                "recipe":{"tag":"image.patch-grid.v1","parameters":{
                    "factor":1,"minimum_pixels":1,"maximum_pixels":4,"patch":1,
                    "maximum_aspect_ratio":4.0,
                    "temporal_patch":1,"merge":1,"value_scale":1.0,
                    "mean":[0.0,0.0,0.0],"std":[1.0,1.0,1.0]}},
                "render":{"pad":"<<bot>>"}}]
        }))
        .unwrap(),
    );
    declaration
}

/// The fixture engine with a media arena of `MEDIA_PAGES` pages, so chat
/// requests with images generate end to end.
const MEDIA_PAGES: usize = 8;

fn media_engine<E: ModelExecutor>(executor: E) -> Engine<E> {
    let media = runtime_engine::MediaConfigV1::new(MEDIA_PAGES, 64)
        .and_then(|media| media.with_encoder(1024, 1))
        .expect("the fixture arena is valid");
    let geometry = KvGeometry::new(16, 16).expect("the fixture page pool is valid");
    Engine::new(
        EngineConfig::new(
            SchedulerConfig::new(2, 4, 64, 64, 1).expect("the fixture schedule is valid"),
            geometry,
            ModelVocabulary::new(VOCABULARY_SIZE).expect("the fixture vocabulary is valid"),
            CONTEXT_TOKENS,
            2,
        )
        .expect("the fixture engine configuration is valid")
        .with_media(media),
        executor,
    )
    .expect("the fixture engine builds")
}

struct Fixture {
    address: std::net::SocketAddr,
    media_url: String,
    model: String,
    hits: Arc<AtomicUsize>,
    open: Arc<AtomicUsize>,
    server: tokio::task::JoinHandle<()>,
    media: tokio::task::JoinHandle<()>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
        self.media.abort();
    }
}
struct ConnectionGuard(Arc<AtomicUsize>);
impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}
impl Fixture {
    async fn start(enabled: bool, request_ms: u64) -> Self {
        Self::start_with_alias(enabled, request_ms, None).await
    }

    async fn start_with_alias(enabled: bool, request_ms: u64, alias: Option<&str>) -> Self {
        Self::start_with_flags(enabled, request_ms, alias, &[]).await
    }

    /// Serves with `flags` added to the parsed serve command.
    async fn start_with_flags(
        enabled: bool,
        request_ms: u64,
        alias: Option<&str>,
        flags: &[&str],
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let media_url = format!("http://{}", listener.local_addr().unwrap());
        let hits = Arc::new(AtomicUsize::new(0));
        let open = Arc::new(AtomicUsize::new(0));
        let counted = hits.clone();
        let opened = open.clone();
        let media = tokio::spawn(async move {
            loop {
                let (socket, _) = listener.accept().await.unwrap();
                let counted = counted.clone();
                let opened = opened.clone();
                tokio::spawn(async move {
                    opened.fetch_add(1, Ordering::SeqCst);
                    let _guard = ConnectionGuard(opened);
                    let mut request = vec![0; 4096];
                    let n = read_some(&socket, &mut request).await.unwrap();
                    let request = String::from_utf8_lossy(&request[..n]);
                    counted.fetch_add(1, Ordering::SeqCst);
                    if request.starts_with("GET /slow ") {
                        write_all(&socket, b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n")
                            .await
                            .unwrap();
                        let mut byte = [0];
                        let _ = read_some(&socket, &mut byte).await;
                    } else {
                        if request.starts_with("GET /delayed ") {
                            tokio::time::sleep(Duration::from_millis(120)).await;
                        }
                        let bytes = runtime_media::decode_base64(PNG, 1024).unwrap();
                        let header = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            bytes.len()
                        );
                        if write_all(&socket, header.as_bytes()).await.is_ok() {
                            let _ = write_all(&socket, &bytes).await;
                        }
                    }
                });
            }
        });
        let mut args = vec!["serve", "--bundle", "fixture"];
        args.extend_from_slice(flags);
        if let Some(alias) = alias {
            args.extend(["--served-model-name", alias]);
        }
        let budget = request_ms.to_string();
        if enabled {
            args.extend([
                "--allowed-media-domains",
                "127.0.0.1",
                "--media-allow-private-addresses",
                "on",
                "--media-request-timeout-ms",
                &budget,
            ]);
        }
        let HostCommandV1::Serve {
            remote_media,
            http_options,
            serving,
            ..
        } = parse_host_command_v1(&args.into_iter().map(str::to_owned).collect::<Vec<_>>())
            .unwrap()
        else {
            panic!("serve command");
        };
        let declaration = media_declaration();
        let backend = EngineGenerationBackend::new(
            tokenizer(),
            media_engine(FixtureExecutorV1::emitting()),
            EngineBackendConfig::default(),
            EngineModelPolicyV1::from_declaration(&declaration).with_media_inputs(true),
        )
        .unwrap();
        let (frontend, backend) = configure_remote_media_v1(
            BundleFrontendV1::with_export(
                declaration,
                &runtime_frontend::test_support::fixture_export_v1(),
            )
            .unwrap(),
            backend,
            &remote_media,
        )
        .unwrap();
        let frontend =
            configure_served_images_v1(frontend, serving.maximum_images_per_request).unwrap();
        assert_eq!(frontend.allows_remote_media(), enabled);
        let config = configure_media_body_limit_v1(
            ServerConfig::default(),
            &frontend,
            serving.maximum_request_body_bytes.is_some(),
        )
        .unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let model = http_options
            .served_model_names
            .first()
            .cloned()
            .unwrap_or_else(|| "fixture/metrics-1".into());
        let metrics = Arc::new(backend.runtime_metrics(&http_options.served_model_names));
        let server = tokio::spawn(async move {
            runtime_server::serve_with_shutdown_and_http_options(
                listener,
                Arc::new(backend),
                Arc::new(frontend),
                config,
                Arc::new(runtime_server::SystemMonotonicClock::new()),
                metrics,
                std::future::pending::<()>(),
                Duration::from_secs(1),
                *http_options,
            )
            .await
            .unwrap();
        });
        Self {
            address,
            media_url,
            model,
            hits,
            open,
            server,
            media,
        }
    }
    async fn request(&self, paths: &[&str]) -> TcpStream {
        let content = paths
            .iter()
            .map(|path| {
                serde_json::json!({
                    "type":"image_url", "image_url":{"url":format!("{}{path}", self.media_url)}
                })
            })
            .collect::<Vec<_>>();
        let body = serde_json::json!({"model":self.model,
            "messages":[{"role":"user","content":content}]})
        .to_string();
        let socket = TcpStream::connect(self.address).await.unwrap();
        write_all(&socket, format!("POST /tokenize HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
        socket
    }
    async fn response(&self, paths: &[&str]) -> String {
        let socket = self.request(paths).await;
        let mut bytes = Vec::new();
        tokio::time::timeout(Duration::from_secs(3), read_to_end(&socket, &mut bytes))
            .await
            .unwrap()
            .unwrap();
        String::from_utf8(bytes).unwrap()
    }
    /// Sends one request and returns its status and JSON body.
    async fn exchange(
        &self,
        method: &str,
        path: &str,
        body: Option<&serde_json::Value>,
    ) -> (u16, serde_json::Value) {
        let body = body.map(serde_json::Value::to_string).unwrap_or_default();
        let socket = TcpStream::connect(self.address).await.unwrap();
        write_all(&socket, format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
        let mut bytes = Vec::new();
        tokio::time::timeout(Duration::from_secs(3), read_to_end(&socket, &mut bytes))
            .await
            .unwrap()
            .unwrap();
        let response = String::from_utf8(bytes).unwrap();
        let status = response[9..12].parse().unwrap();
        let (_, payload) = response.split_once("\r\n\r\n").unwrap();
        (
            status,
            serde_json::from_str(payload).unwrap_or(serde_json::Value::Null),
        )
    }

    /// A chat request whose user messages carry `images` data-URL images
    /// each.
    fn image_chat(&self, images: &[usize], url: &str) -> serde_json::Value {
        let messages = images
            .iter()
            .map(|count| {
                let mut content = vec![serde_json::json!({"type":"text","text":"look"})];
                content.extend(
                    (0..*count)
                        .map(|_| serde_json::json!({"type":"image_url","image_url":{"url":url}})),
                );
                serde_json::json!({"role":"user","content":content})
            })
            .collect::<Vec<_>>();
        serde_json::json!({"model":self.model, "messages":messages, "max_tokens":1})
    }

    async fn closed(&self) {
        tokio::time::timeout(Duration::from_secs(1), async {
            while self.open.load(Ordering::SeqCst) != 0 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
    }
}

#[tokio::test]
async fn allowed_media_domains_enables_remote_preparation_end_to_end() {
    let off = Fixture::start(false, 1000).await;
    let response = off.response(&["/png"]).await;
    assert!(response.starts_with("HTTP/1.1 400"), "{response}");
    assert_eq!(off.hits.load(Ordering::SeqCst), 0);
    let on = Fixture::start(true, 1000).await;
    let response = on.response(&["/png"]).await;
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert!(response.contains("tokens"), "{response}");
    assert_eq!(on.hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn served_model_name_preserves_remote_media_opt_in_end_to_end() {
    let off = Fixture::start_with_alias(false, 1000, Some("media-alias")).await;
    assert_eq!(off.model, "media-alias");
    let response = off.response(&["/png"]).await;
    assert!(response.starts_with("HTTP/1.1 400"), "{response}");
    assert!(response.contains("enabled HTTP(S) URL"), "{response}");
    assert_eq!(off.hits.load(Ordering::SeqCst), 0);

    let on = Fixture::start_with_alias(true, 1000, Some("media-alias")).await;
    assert_eq!(on.model, "media-alias");
    let response = on.response(&["/png"]).await;
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert!(response.contains("tokens"), "{response}");
    assert_eq!(on.hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn host_shares_one_fetch_budget_and_cancels_on_disconnect() {
    let fixture = Fixture::start(true, 250).await;
    let started = std::time::Instant::now();
    let response = fixture.response(&["/delayed", "/slow", "/png"]).await;
    assert!(response.starts_with("HTTP/1.1 400"), "{response}");
    assert!(
        response.contains("timeout") || response.contains("timed out"),
        "{response}"
    );
    assert!(started.elapsed() < Duration::from_millis(800));
    assert_eq!(fixture.hits.load(Ordering::SeqCst), 2);
    fixture.closed().await;

    let fixture = Fixture::start(true, 3000).await;
    let socket = fixture.request(&["/slow", "/png"]).await;
    tokio::time::timeout(Duration::from_secs(1), async {
        while fixture.hits.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    drop(socket);
    fixture.closed().await;
    assert_eq!(fixture.hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn image_count_limit_is_refused_in_the_shape_clients_recover_from() {
    let fixture = Fixture::start(false, 1000).await;
    let png = format!("data:image/png;base64,{PNG}");
    // At the declared limit of four, across two messages, the request runs.
    let (status, body) = fixture
        .exchange(
            "POST",
            "/v1/chat/completions",
            Some(&fixture.image_chat(&[2, 2], &png)),
        )
        .await;
    assert_eq!(status, 200, "{body}");
    // One more image, anywhere in the conversation, is the count refusal.
    let (status, body) = fixture
        .exchange(
            "POST",
            "/v1/chat/completions",
            Some(&fixture.image_chat(&[3, 2], &png)),
        )
        .await;
    assert_eq!(status, 400);
    assert_eq!(
        body,
        serde_json::json!({"error":{
            "message":"At most 4 image(s) may be provided in one request; this request carried 5.",
            "type":"BadRequestError","param":"image","code":400}})
    );
    // The count is refused before any image is fetched.
    let remote = Fixture::start(true, 1000).await;
    let url = format!("{}/png", remote.media_url);
    let (status, body) = remote
        .exchange(
            "POST",
            "/v1/chat/completions",
            Some(&remote.image_chat(&[5], &url)),
        )
        .await;
    assert_eq!(status, 400);
    assert_eq!(body["error"]["param"], "image", "{body}");
    assert_eq!(remote.hits.load(Ordering::SeqCst), 0);
    // An image the decoder refuses is not a count problem.
    let (status, body) = fixture
        .exchange(
            "POST",
            "/v1/chat/completions",
            Some(&fixture.image_chat(&[1], "data:image/png;base64,AAAAAAAA")),
        )
        .await;
    assert_eq!(status, 400, "{body}");
    assert_ne!(body["error"]["param"], "image", "{body}");
    // The model list reports the limit and the input modalities.
    let (status, body) = fixture.exchange("GET", "/v1/models", None).await;
    assert_eq!(status, 200);
    let entry = &body["data"][0];
    assert_eq!(
        entry["architecture"],
        serde_json::json!({"input_modalities":["text","image"]})
    );
    assert_eq!(entry["max_images_per_request"], 4);
}

/// `--max-images-per-request` sets the one limit the frontend enforces, its
/// refusal names and the model list reports, below or above the declared
/// four.
#[tokio::test]
async fn the_image_limit_flag_sets_enforcement_refusal_and_model_list_alike() {
    let png = format!("data:image/png;base64,{PNG}");
    let lowered =
        Fixture::start_with_flags(false, 1000, None, &["--max-images-per-request", "2"]).await;
    let (status, body) = lowered
        .exchange(
            "POST",
            "/v1/chat/completions",
            Some(&lowered.image_chat(&[1, 2], &png)),
        )
        .await;
    assert_eq!(status, 400);
    assert_eq!(
        body["error"]["message"],
        "At most 2 image(s) may be provided in one request; this request carried 3."
    );
    assert_eq!(body["error"]["param"], "image");
    let (_, models) = lowered.exchange("GET", "/v1/models", None).await;
    assert_eq!(models["data"][0]["max_images_per_request"], 2);

    let raised =
        Fixture::start_with_flags(false, 1000, None, &["--max-images-per-request", "8"]).await;
    let (status, body) = raised
        .exchange(
            "POST",
            "/v1/chat/completions",
            Some(&raised.image_chat(&[4, 4], &png)),
        )
        .await;
    assert_eq!(status, 200, "{body}");
    let (_, models) = raised.exchange("GET", "/v1/models", None).await;
    assert_eq!(models["data"][0]["max_images_per_request"], 8);
}

/// The four images this fixture accepts widen the chat body limit by 4 MiB
/// each, so a chat body over the 64 MiB text limit is read and served.
#[tokio::test]
async fn chat_bodies_may_grow_with_the_images_a_request_may_carry() {
    let fixture = Fixture::start(false, 1000).await;
    let png = format!("data:image/png;base64,{PNG}");
    let mut chat = fixture.image_chat(&[4], &png);
    // Unknown fields are read and ignored; each string stays under the
    // string limit.
    chat["padding"] = serde_json::json!(vec!["x".repeat(22 << 20); 3]);
    let (status, body) = fixture
        .exchange("POST", "/v1/chat/completions", Some(&chat))
        .await;
    assert_eq!(status, 200, "{body}");
}

async fn read_some(socket: &TcpStream, buffer: &mut [u8]) -> std::io::Result<usize> {
    loop {
        socket.readable().await?;
        match socket.try_read(buffer) {
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            result => return result,
        }
    }
}
async fn write_all(socket: &TcpStream, mut bytes: &[u8]) -> std::io::Result<()> {
    while !bytes.is_empty() {
        socket.writable().await?;
        match socket.try_write(bytes) {
            Ok(0) => return Err(std::io::ErrorKind::WriteZero.into()),
            Ok(n) => bytes = &bytes[n..],
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}
async fn read_to_end(socket: &TcpStream, bytes: &mut Vec<u8>) -> std::io::Result<()> {
    let mut buffer = [0; 4096];
    loop {
        let n = read_some(socket, &mut buffer).await?;
        if n == 0 {
            return Ok(());
        }
        bytes.extend_from_slice(&buffer[..n]);
    }
}
