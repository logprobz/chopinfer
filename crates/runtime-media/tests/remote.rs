use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use axum::body::{Body, Bytes};
use axum::http::{Request, Response, StatusCode};
use axum::routing::any;
use runtime_media::remote::{RemoteMediaPolicy, fetch};
use runtime_media::{MediaLimits, MediaResult, MediaSource};

fn wav() -> Vec<u8> {
    let mut out = b"RIFF".to_vec();
    out.extend(38_u32.to_le_bytes());
    out.extend(b"WAVEfmt ");
    out.extend(16_u32.to_le_bytes());
    out.extend(1_u16.to_le_bytes());
    out.extend(1_u16.to_le_bytes());
    out.extend(8000_u32.to_le_bytes());
    out.extend(16000_u32.to_le_bytes());
    out.extend(2_u16.to_le_bytes());
    out.extend(16_u16.to_le_bytes());
    out.extend(b"data");
    out.extend(2_u32.to_le_bytes());
    out.extend(0_i16.to_le_bytes());
    out
}

fn limits(maximum_bytes: usize) -> MediaLimits {
    MediaLimits {
        maximum_bytes,
        maximum_pixels: 1024,
        maximum_audio_frames: 8000,
        maximum_frames: 1,
        maximum_tokens: 1024,
        maximum_aspect_ratio: 200.0,
    }
}

struct Server {
    base: String,
    hits: Arc<AtomicUsize>,
    open_streams: Arc<AtomicUsize>,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
struct HeldStream(Arc<AtomicUsize>);
impl Drop for HeldStream {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Server {
    async fn start() -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let ipv6 = tokio::net::TcpListener::bind((std::net::Ipv6Addr::LOCALHOST, port))
            .await
            .unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let hits = Arc::new(AtomicUsize::new(0));
        let open_streams = Arc::new(AtomicUsize::new(0));
        let counted = hits.clone();
        let streams = open_streams.clone();
        // Axum's server drives Hyper HTTP/1 connections on a local test listener.
        let app = axum::Router::new().fallback(any(move |request: Request<Body>| {
            let counted = counted.clone();
            let streams = streams.clone();
            async move {
                counted.fetch_add(1, Ordering::SeqCst);
                let path = request.uri().path();
                let location = match path {
                    "/redirect" => Some("/wav"),
                    "/relative" => Some("wav"),
                    "/redirect-body" => Some("/wav"),
                    "/loop" => Some("/loop"),
                    "/file" => Some("file:///etc/passwd"),
                    "/ftp" => Some("ftp://127.0.0.1/a"),
                    "/other-host" => Some("http://localhost/wav"),
                    "/credentials" => Some("http://user:pass@127.0.0.1/wav"),
                    "/private" => Some("http://169.254.169.254/latest"),
                    _ => None,
                };
                if let Some(location) = location {
                    return Response::builder()
                        .status(StatusCode::FOUND)
                        .header("location", location)
                        .body(if path == "/redirect-body" { Body::from(vec![0; 20]) } else { Body::empty() })
                        .unwrap();
                }
                if path == "/host" {
                    assert_eq!(request.headers()["host"], format!("LOCALHOST:{port}"));
                }
                if path == "/png" {
                    return Response::new(Body::from(runtime_media::decode_base64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC", 1024).unwrap()));
                }
                if path == "/video" {
                    return Response::new(Body::from(include_bytes!("fixtures/red-then-blue.mp4").to_vec()));
                }
                if path == "/slow" || path == "/oversize" {
                    let (send, receive) =
                        tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(1);
                    streams.fetch_add(1, Ordering::SeqCst);
                    let guard = HeldStream(streams);
                    let oversize = path == "/oversize";
                    tokio::spawn(async move {
                        let _guard = guard;
                        if oversize {
                            for _ in 0..20 {
                                if send.send(Ok(Bytes::from(vec![0; 8192]))).await.is_err() {
                                    return;
                                }
                            }
                        } else {
                            let _ = send.send(Ok(Bytes::from_static(b"RIFF"))).await;
                        }
                        send.closed().await;
                    });
                    return Response::new(Body::from_stream(
                        tokio_stream::wrappers::ReceiverStream::new(receive),
                    ));
                }
                if path == "/big-header" {
                    return Response::builder()
                        .header("x-padding", "x".repeat(20_000))
                        .body(Body::from(wav()))
                        .unwrap();
                }
                if path == "/missing" {
                    return Response::builder()
                        .status(404)
                        .body(Body::from(wav()))
                        .unwrap();
                }
                Response::new(Body::from(wav()))
            }
        }));
        let task = tokio::spawn(async move {
            let (v4, v6) = tokio::join!(axum::serve(listener, app.clone()), axum::serve(ipv6, app));
            v4.unwrap();
            v6.unwrap();
        });
        Self {
            base,
            hits,
            open_streams,
            task,
        }
    }
    fn policy(&self) -> RemoteMediaPolicy {
        RemoteMediaPolicy {
            allowed_domains: vec!["127.0.0.1".into()],
            allow_private_addresses: true,
            total_timeout: Duration::from_secs(2),
            ..Default::default()
        }
    }
    async fn get(
        &self,
        path: &str,
        policy: RemoteMediaPolicy,
        maximum: usize,
    ) -> MediaResult<MediaSource> {
        let url = format!("{}{path}", self.base);
        tokio::task::spawn_blocking(move || fetch(&url, &policy, limits(maximum), || false))
            .await
            .unwrap()
    }
    async fn closed(&self) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while self.open_streams.load(Ordering::SeqCst) != 0 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("curl connection and stream must be closed");
    }
}

#[tokio::test]
async fn fetches_bytes_and_identifies_media_without_trusting_content_type() {
    let server = Server::start().await;
    for path in ["/wav", "/redirect", "/relative"] {
        let source = server.get(path, server.policy(), 46).await.unwrap();
        assert_eq!(source.mime, "audio/wav");
        assert_eq!(source.bytes, wav());
    }
    let png = server.get("/png", server.policy(), 1024).await.unwrap();
    assert_eq!(png.mime, "image/png");
    assert_eq!(
        runtime_media::png::decode(&png.bytes, limits(1024))
            .unwrap()
            .data,
        [255, 0, 0]
    );
    let video = server
        .get("/video", server.policy(), 1024 * 1024)
        .await
        .unwrap();
    assert_eq!(video.mime, "video/mp4");
    assert_eq!(video.bytes, include_bytes!("fixtures/red-then-blue.mp4"));
    // Also exercise DNS, whose answer must be checked and pinned.
    let url = server.base.replace("127.0.0.1", "localhost") + "/wav";
    let mut policy = server.policy();
    policy.allowed_domains = vec!["localhost".into()];
    // localhost may resolve to IPv6 first; use a separate IPv4-only DNS name test
    // through policy refusal below instead of assuming the machine's resolver order.
    let result = tokio::task::spawn_blocking(move || {
        policy.allow_private_addresses = false;
        fetch(&url, &policy, limits(46), || false)
    })
    .await
    .unwrap();
    assert!(result.unwrap_err().0.contains("private"));
}

#[tokio::test]
async fn rejects_default_policy_unlisted_hosts_and_private_addresses_before_connecting() {
    let server = Server::start().await;
    assert!(
        server
            .get("/wav", RemoteMediaPolicy::default(), 100)
            .await
            .unwrap_err()
            .0
            .contains("allowed-media-domains")
    );
    let mut policy = server.policy();
    policy.allow_private_addresses = false;
    assert!(
        server
            .get("/wav", policy, 100)
            .await
            .unwrap_err()
            .0
            .contains("private")
    );
    assert_eq!(server.hits.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn enforces_streaming_body_and_header_limits_and_http_status() {
    let server = Server::start().await;
    assert!(
        server
            .get("/wav", server.policy(), 45)
            .await
            .unwrap_err()
            .0
            .contains("byte limit")
    );
    assert!(
        server
            .get("/oversize", server.policy(), 9000)
            .await
            .unwrap_err()
            .0
            .contains("byte limit")
    );
    server.closed().await;
    assert!(
        server
            .get("/big-header", server.policy(), 100)
            .await
            .unwrap_err()
            .0
            .contains("header limit")
    );
    assert!(
        server
            .get("/missing", server.policy(), 100)
            .await
            .unwrap_err()
            .0
            .contains("status")
    );
}

#[tokio::test]
async fn revalidates_redirects_and_bounds_loops() {
    let server = Server::start().await;
    for path in ["/file", "/ftp", "/other-host", "/credentials", "/private"] {
        let before = server.hits.load(Ordering::SeqCst);
        assert!(
            server.get(path, server.policy(), 100).await.is_err(),
            "{path}"
        );
        assert_eq!(server.hits.load(Ordering::SeqCst), before + 1);
    }
    let mut policy = server.policy();
    policy.maximum_redirects = 0;
    assert!(
        server
            .get("/redirect", policy, 100)
            .await
            .unwrap_err()
            .0
            .contains("redirect limit")
    );
    assert!(
        server
            .get("/redirect-body", server.policy(), 46)
            .await
            .unwrap_err()
            .0
            .contains("byte limit")
    );
    let before = server.hits.load(Ordering::SeqCst);
    assert!(
        server
            .get("/loop", server.policy(), 100)
            .await
            .unwrap_err()
            .0
            .contains("redirect limit")
    );
    assert_eq!(server.hits.load(Ordering::SeqCst), before + 4);
}

#[tokio::test]
async fn deadline_stops_a_stalled_body_and_closes_the_connection() {
    let server = Server::start().await;
    let mut policy = server.policy();
    policy.total_timeout = Duration::from_millis(100);
    let started = Instant::now();
    let error = server.get("/slow", policy, 100).await.unwrap_err();
    assert!(
        error.0.contains("timeout") || error.0.contains("timed out"),
        "{error}"
    );
    assert!(started.elapsed() < Duration::from_secs(1));
    server.closed().await;
}

#[tokio::test]
async fn cancellation_stops_a_stalled_body_and_releases_the_reader() {
    let server = Server::start().await;
    let cancelled = Arc::new(AtomicBool::new(false));
    let signal = cancelled.clone();
    let policy = server.policy();
    let url = format!("{}/slow", server.base);
    let task = tokio::task::spawn_blocking(move || {
        fetch(&url, &policy, limits(100), || signal.load(Ordering::SeqCst))
    });
    tokio::time::timeout(Duration::from_secs(1), async {
        while server.open_streams.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    cancelled.store(true, Ordering::SeqCst);
    let error = tokio::time::timeout(Duration::from_secs(1), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(error.0.contains("cancelled"));
    server.closed().await;
    assert_eq!(
        server
            .get("/wav", server.policy(), 100)
            .await
            .unwrap()
            .bytes,
        wav()
    );
}

#[test]
fn urls_reject_credentials_ambiguous_hosts_and_non_http_schemes() {
    let policy = RemoteMediaPolicy {
        allowed_domains: vec!["example.com".into()],
        ..Default::default()
    };
    for url in [
        "file:///tmp/a",
        "ftp://example.com/a",
        "http://x@example.com/a",
        "http://example.com:99999/a",
        "http://example.com\\@evil/a",
        "http://example.com/a#fragment",
        "http://example.com./a",
        "http://%65xample.com/a",
        "http://example.com/a\n",
        "http://sub.example.com/a",
    ] {
        assert!(policy.validate_url(url).is_err(), "{url}");
    }
    assert!(
        policy
            .validate_url("https://EXAMPLE.com:443/a?signature=abc")
            .is_ok()
    );
}

#[tokio::test]
async fn request_deadline_spans_items_and_stops_the_in_flight_body() {
    let server = Server::start().await;
    let policy = server.policy();
    let base = server.base.clone();
    let started = Instant::now();
    let result = tokio::task::spawn_blocking(move || {
        let deadline = Instant::now() + Duration::from_millis(200);
        let first = runtime_media::remote::fetch_with_deadline(
            &format!("{base}/wav"),
            &policy,
            limits(100),
            deadline,
            || false,
        )
        .unwrap();
        assert_eq!(first.bytes, wav());
        // Time spent between items (including preprocessing) consumes the budget.
        std::thread::sleep(Duration::from_millis(120));
        runtime_media::remote::fetch_with_deadline(
            &format!("{base}/slow"),
            &policy,
            limits(100),
            deadline,
            || false,
        )
    })
    .await
    .unwrap();
    assert!(result.is_err());
    assert!(
        started.elapsed() < Duration::from_millis(800),
        "request budget reset between items"
    );
    assert_eq!(server.hits.load(Ordering::SeqCst), 2);
    server.closed().await;
}

#[tokio::test]
async fn hostname_fetch_pins_dns_and_preserves_the_host_header() {
    let server = Server::start().await;
    let url = server.base.replace("127.0.0.1", "LOCALHOST") + "/host";
    let mut policy = server.policy();
    policy.allowed_domains = vec!["localhost".into()];
    let source = tokio::task::spawn_blocking(move || fetch(&url, &policy, limits(46), || false))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(source.bytes, wav());
    assert_eq!(server.hits.load(Ordering::SeqCst), 1);
}
