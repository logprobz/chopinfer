//! HTTP/1 connections with explicit header deadlines and bounded task ownership.
use std::{
    future::Future,
    io,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use axum::Router;
use axum::http::StatusCode;
use hyper::{server::conn::http1, service::Service as _};
use hyper_util::{
    rt::{TokioIo, TokioTimer},
    service::TowerToHyperService,
};
use tokio::{
    net::TcpListener,
    sync::{Semaphore, watch},
    task::JoinSet,
};

use crate::{ServerConfig, request_log::RequestLog};

pub(crate) async fn serve(
    listener: TcpListener,
    router: Router,
    config: ServerConfig,
    budget: Arc<Budget>,
    log: Arc<RequestLog>,
    shutdown: impl Future<Output = ()>,
) -> io::Result<()> {
    serve_with_accept(|| listener.accept(), router, config, budget, log, shutdown).await
}

async fn serve_with_accept<A, F>(
    mut accept: A,
    router: Router,
    config: ServerConfig,
    budget: Arc<Budget>,
    log: Arc<RequestLog>,
    shutdown: impl Future<Output = ()>,
) -> io::Result<()>
where
    A: FnMut() -> F,
    F: Future<Output = io::Result<(tokio::net::TcpStream, std::net::SocketAddr)>>,
{
    let limit = config
        .max_connections
        .unwrap_or(crate::DEFAULT_MAX_CONNECTIONS)
        .clamp(1, Semaphore::MAX_PERMITS);
    let capacity = Arc::new(Semaphore::new(limit));
    let (stop, stopped) = watch::channel(false);
    let mut connections = JoinSet::new();
    tokio::pin!(shutdown);
    let mut retry_at = None;
    let result = loop {
        tokio::select! {
            biased;
            () = &mut shutdown => break Ok(()),
            Some(_) = connections.join_next(), if !connections.is_empty() => {},
            accepted = async {
                if let Some(deadline) = retry_at { tokio::time::sleep_until(deadline).await; }
                accept().await
            } => {
                let (socket, peer) = match accepted {
                    Ok(accepted) => { retry_at = None; accepted },
                    Err(error) => match accept_retry(&error) {
                        Some(delay) => {
                            if !delay.is_zero() { runtime_log::warn!("HTTP accept temporarily unavailable: {error}; retrying"); }
                            retry_at = Some(tokio::time::Instant::now() + delay);
                            continue;
                        }
                        None => break Err(error),
                    },
                };
                let first_deadline = tokio::time::Instant::now() + config.header_read_timeout;
                // Completed records count too: reap before testing the retained
                // task bound, so connection churn cannot grow the JoinSet.
                while connections.try_join_next().is_some() {}
                if connections.len() >= limit {
                    drop(socket);
                    log.transport_failure(Some(peer), crate::request_log::connection_limit_detail());
                    continue;
                }
                let permit = match Arc::clone(&capacity).try_acquire_owned() {
                    Ok(permit) => permit,
                    Err(_) => {
                        drop(socket);
                        log.transport_failure(Some(peer), crate::request_log::connection_limit_detail());
                        continue;
                    }
                };
                let first_head = Arc::new(tokio::sync::Notify::new());
                let head_received = Arc::clone(&first_head);
                let service = TowerToHyperService::new(router.clone().layer(axum::Extension(peer)));
                // Bytes this connection read, how many of them its finished
                // exchanges read, and whether the last request left its body
                // unread, to tell a head that never arrived from an idle
                // keep-alive connection.
                let read = Arc::new(AtomicU64::new(0));
                let settled = Arc::new(AtomicU64::new(0));
                let unread = Arc::new(AtomicBool::new(false));
                let exchange = (Arc::clone(&read), Arc::clone(&settled), Arc::clone(&unread));
                let service = hyper::service::service_fn(move |request: axum::http::Request<hyper::body::Incoming>| {
                    head_received.notify_one();
                    let (read, settled) = (Arc::clone(&exchange.0), Arc::clone(&exchange.1));
                    exchange.2.store(false, Ordering::Relaxed);
                    let unread = Arc::clone(&exchange.2);
                    let request = request.map(|body| RequestBody { body, ended: false, unread });
                    let response = service.call(request);
                    async move {
                        response
                            .await
                            .map(|response| response.map(|body| Exchanged { body, read, settled }))
                    }
                });
                let mut stopped = stopped.clone();
                let budget = Arc::clone(&budget);
                let log = Arc::clone(&log);
                connections.spawn(async move {
                    let _permit = permit;
                    let waiting = budget.waiting.enter(crate::waiting::Stage::Transport);
                    let _buffers = tokio::select! {
                        biased;
                        _ = stopped.changed() => return,
                        _ = tokio::time::sleep_until(first_deadline) => return,
                        permit = Arc::clone(&budget.slots).acquire_owned() => permit.expect("transport budget remains open"),
                        _ = peer_closed(&socket) => return,
                    };
                    drop(waiting);
                    let mut builder = http1::Builder::new();
                    builder.timer(TokioTimer::new()).header_read_timeout(config.header_read_timeout).max_buf_size(READ_BUFFER);
                    let socket = MeteredSocket { socket, budget, read: Arc::clone(&read) };
                    let connection = builder.serve_connection(TokioIo::new(socket), service);
                    tokio::pin!(connection);
                    let first_header_timeout = async {
                        tokio::select! {
                            biased;
                            _ = first_head.notified() => std::future::pending::<()>().await,
                            _ = tokio::time::sleep_until(first_deadline) => {},
                        }
                    };
                    let result = tokio::select! {
                        biased;
                        _ = first_header_timeout => {
                            if read.load(Ordering::Relaxed) > 0 {
                                log.transport_failure(Some(peer), crate::request_log::transport_detail(StatusCode::REQUEST_TIMEOUT, "request head did not arrive in time"));
                            }
                            return;
                        },
                        _ = stopped.changed() => {
                            connection.as_mut().graceful_shutdown();
                            connection.await
                        },
                        result = &mut connection => result,
                    };
                    let partial = read.load(Ordering::Relaxed) > settled.load(Ordering::Relaxed)
                        && !unread.load(Ordering::Relaxed);
                    if let Err(error) = result
                        && let Some(detail) = head_failure(&error, partial)
                    {
                        log.transport_failure(Some(peer), detail);
                    }
                });
                debug_assert!(connections.len() <= limit);
            }
        }
    };
    let _ = stop.send(true);
    if result.is_err() {
        connections.abort_all();
    }
    while connections.join_next().await.is_some() {}
    result
}

/// Interrupted accepts retry immediately; resource exhaustion backs off while
/// the outer select keeps shutdown and completed-task reaping responsive.
fn accept_retry(error: &io::Error) -> Option<std::time::Duration> {
    if matches!(
        error.kind(),
        io::ErrorKind::Interrupted | io::ErrorKind::ConnectionAborted
    ) {
        return Some(std::time::Duration::ZERO);
    }
    let mut exhausted = error.kind() == io::ErrorKind::OutOfMemory;
    // ENFILE and EMFILE have these values on the supported Unix targets.
    #[cfg(unix)]
    {
        exhausted |= matches!(error.raw_os_error(), Some(23 | 24));
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        exhausted |= error.raw_os_error() == Some(105);
    } // ENOBUFS
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        exhausted |= error.raw_os_error() == Some(55);
    } // ENOBUFS
    exhausted.then_some(std::time::Duration::from_millis(100))
}

/// The failure line of a request whose head hyper refused: one that did not
/// parse, which hyper answered, or a head that did not arrive within the
/// header deadline. The deadline also expires on an idle keep-alive
/// connection, so a timeout counts only when the connection read part of a
/// head: bytes after its last exchange ended.
fn head_failure(error: &hyper::Error, partial: bool) -> Option<crate::request_log::FailureDetail> {
    if error.is_parse_too_large() {
        Some(crate::request_log::transport_detail(
            StatusCode::REQUEST_HEADER_FIELDS_TOO_LARGE,
            "request head exceeds the size limit",
        ))
    } else if error.is_parse() {
        Some(crate::request_log::transport_detail(
            StatusCode::BAD_REQUEST,
            "request head did not parse",
        ))
    } else if error.is_timeout() && partial {
        Some(crate::request_log::transport_detail(
            StatusCode::REQUEST_TIMEOUT,
            "request head did not arrive in time",
        ))
    } else {
        None
    }
}

// Poll read closure without requesting ordinary readability: queued body
// bytes must remain in the socket, and must not cause a busy readiness loop.
#[cfg(any(target_os = "linux", target_os = "android"))]
async fn peer_closed(socket: &tokio::net::TcpStream) {
    match socket
        .ready(tokio::io::Interest::PRIORITY | tokio::io::Interest::ERROR)
        .await
    {
        Ok(ready) if !ready.is_read_closed() && !ready.is_error() => {
            std::future::pending::<()>().await
        }
        _ => {}
    }
}

// Other pollers do not expose closure independently of readable payload.
// Their disconnected waiters retire when granted a buffer slot or on shutdown.
#[cfg(not(any(target_os = "linux", target_os = "android")))]
async fn peer_closed(_: &tokio::net::TcpStream) {
    std::future::pending::<()>().await;
}

// Hyper 1.11 may grow a BytesMut to twice max_buf_size and retain split
// backings in the parsed headers, body channel and the consumer's current
// frame. Reserve eight buffers per connection, including that growth. Keep
// this allowance with the connection, even between keep-alive requests.
const READ_BUFFER: usize = 128 * 1024;
const CONNECTION_BYTES: usize = 8 * READ_BUFFER;
pub(crate) const DEFAULT_BUDGET: usize = 512 * CONNECTION_BYTES;

pub(crate) struct Budget {
    slots: Arc<Semaphore>,
    limit: usize,
    read: std::sync::atomic::AtomicU64,
    pub(crate) waiting: Arc<crate::waiting::Waiting>,
}

impl Budget {
    pub(crate) fn new(bytes: usize) -> Arc<Self> {
        let slots = (bytes / CONNECTION_BYTES).clamp(1, Semaphore::MAX_PERMITS);
        Arc::new(Self {
            slots: Arc::new(Semaphore::new(slots)),
            limit: slots * CONNECTION_BYTES,
            read: std::sync::atomic::AtomicU64::new(0),
            waiting: Arc::new(crate::waiting::Waiting::default()),
        })
    }

    fn reserved(&self) -> usize {
        self.limit - self.slots.available_permits() * CONNECTION_BYTES
    }

    pub(crate) fn value(&self) -> serde_json::Value {
        serde_json::json!({"buffer_budget_bytes": self.limit, "buffer_reserved_bytes": self.reserved()})
    }

    pub(crate) fn prometheus(&self) -> String {
        format!(
            "# TYPE chop_transport_buffer_budget_bytes gauge\nchop_transport_buffer_budget_bytes {}\n# TYPE chop_transport_buffer_reserved_bytes gauge\nchop_transport_buffer_reserved_bytes {}\n# TYPE chop_transport_read_bytes_total counter\nchop_transport_read_bytes_total {}\n",
            self.limit,
            self.reserved(),
            self.read.load(std::sync::atomic::Ordering::Relaxed)
        )
    }
}

struct MeteredSocket {
    socket: tokio::net::TcpStream,
    budget: Arc<Budget>,
    /// This connection's bytes read.
    read: Arc<AtomicU64>,
}

impl tokio::io::AsyncRead for MeteredSocket {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<io::Result<()>> {
        let before = buf.filled().len();
        let result = std::pin::Pin::new(&mut self.socket).poll_read(cx, buf);
        let read = (buf.filled().len() - before) as u64;
        self.budget.read.fetch_add(read, Ordering::Relaxed);
        self.read.fetch_add(read, Ordering::Relaxed);
        result
    }
}

/// A request body that notes, when the handler drops it, whether the
/// handler left part of it unread. Hyper then drains what it can of the rest
/// after the answer, and those bytes are not a head, so a header timeout
/// that follows is not counted. A partial head after such a request
/// therefore writes no line either.
struct RequestBody {
    body: hyper::body::Incoming,
    ended: bool,
    unread: Arc<AtomicBool>,
}

impl hyper::body::Body for RequestBody {
    type Data = axum::body::Bytes;
    type Error = hyper::Error;

    fn poll_frame(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<hyper::body::Frame<Self::Data>, Self::Error>>> {
        let frame = std::pin::Pin::new(&mut self.body).poll_frame(cx);
        if matches!(frame, std::task::Poll::Ready(None)) {
            self.ended = true;
        }
        frame
    }

    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }

    fn size_hint(&self) -> hyper::body::SizeHint {
        self.body.size_hint()
    }
}

impl Drop for RequestBody {
    fn drop(&mut self) {
        if !self.ended && !hyper::body::Body::is_end_stream(&self.body) {
            self.unread.store(true, Ordering::Relaxed);
        }
    }
}

/// A response body that, when hyper drops it, marks every byte its
/// connection has read as part of a finished exchange. Hyper drops it after
/// taking the response's last frame, and reads the next head only after
/// that, having drained or closed the request's unread body. Bytes of a pipelined head
/// read before then count as settled, so a partial pipelined head that times
/// out writes no line.
struct Exchanged {
    body: axum::body::Body,
    read: Arc<AtomicU64>,
    settled: Arc<AtomicU64>,
}

impl hyper::body::Body for Exchanged {
    type Data = axum::body::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<hyper::body::Frame<Self::Data>, Self::Error>>> {
        std::pin::Pin::new(&mut self.body).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }

    fn size_hint(&self) -> hyper::body::SizeHint {
        self.body.size_hint()
    }
}

impl Drop for Exchanged {
    fn drop(&mut self) {
        self.settled
            .store(self.read.load(Ordering::Relaxed), Ordering::Relaxed);
    }
}

impl tokio::io::AsyncWrite for MeteredSocket {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<io::Result<usize>> {
        std::pin::Pin::new(&mut self.socket).poll_write(cx, buf)
    }
    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<io::Result<()>> {
        std::pin::Pin::new(&mut self.socket).poll_flush(cx)
    }
    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<io::Result<()>> {
        std::pin::Pin::new(&mut self.socket).poll_shutdown(cx)
    }
    fn is_write_vectored(&self) -> bool {
        self.socket.is_write_vectored()
    }
    fn poll_write_vectored(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> std::task::Poll<io::Result<usize>> {
        std::pin::Pin::new(&mut self.socket).poll_write_vectored(cx, bufs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::net::TcpStream;
    use tokio::time::{Duration, timeout};

    async fn write_all(socket: &TcpStream, mut bytes: &[u8]) {
        while !bytes.is_empty() {
            socket.writable().await.unwrap();
            match socket.try_write(bytes) {
                Ok(n) => bytes = &bytes[n..],
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                Err(error) => panic!("{error}"),
            }
        }
    }

    async fn closed(socket: &TcpStream) {
        loop {
            socket.readable().await.unwrap();
            match socket.try_read(&mut [0; 1]) {
                Ok(0) => return,
                Err(error) if error.kind() == io::ErrorKind::ConnectionReset => return,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                result => panic!("expected a closed socket, got {result:?}"),
            }
        }
    }

    async fn response(socket: &TcpStream) -> Vec<u8> {
        let mut bytes = Vec::new();
        loop {
            socket.readable().await.unwrap();
            let mut chunk = [0; 4096];
            match socket.try_read(&mut chunk) {
                Ok(0) => return bytes,
                Ok(n) => bytes.extend_from_slice(&chunk[..n]),
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                Err(error) => panic!("{error}"),
            }
        }
    }

    #[test]
    fn production_connections_leave_bounded_room_for_buffer_waiters() {
        let cap = ServerConfig::default()
            .max_connections
            .expect("a default socket bound");
        assert_eq!(cap, 1024);
        assert!(cap > DEFAULT_BUDGET / CONNECTION_BYTES);
        assert_eq!(
            ServerConfig::new(8, 8, 65536, 4096)
                .unwrap()
                .max_connections,
            Some(cap)
        );
        assert_eq!(
            ServerConfig::default()
                .with_max_connections(0)
                .max_connections,
            Some(1)
        );
        assert_eq!(
            ServerConfig::default()
                .with_max_connections(usize::MAX)
                .max_connections,
            Some(Semaphore::MAX_PERMITS)
        );
    }

    #[tokio::test]
    async fn buffer_wait_counts_toward_first_header_deadline_and_socket_cap() {
        timeout(Duration::from_secs(10), async {
            let entered = Arc::new(AtomicUsize::new(0));
            let budget = Budget::new(2 * CONNECTION_BYTES);
            let router = Router::new().route("/", axum::routing::post({
                let entered = Arc::clone(&entered);
                move |body: axum::body::Body| {
                    entered.fetch_add(1, Ordering::Relaxed);
                    async move {
                        assert_eq!(axum::body::to_bytes(body, 1).await.unwrap().as_ref(), b"x");
                        "ok"
                    }
                }
            }));
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let (stop, stopped) = tokio::sync::oneshot::channel();
            let server = tokio::spawn(serve(listener, router,
                ServerConfig::default().with_max_connections(4).with_header_read_timeout(Duration::from_millis(500)),
                Arc::clone(&budget), RequestLog::new(None, Vec::new()), async { let _ = stopped.await; }));
            let head = b"POST / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: 1\r\n\r\n";
            let mut active = Vec::new();
            for _ in 0..2 {
                let socket = TcpStream::connect(address).await.unwrap();
                write_all(&socket, head).await;
                active.push(socket);
            }
            while entered.load(Ordering::Relaxed) != 2 { tokio::task::yield_now().await; }
            let read_before = budget.read.load(Ordering::Relaxed);
            let mut waiting = Vec::new();
            for _ in 0..2 {
                let socket = TcpStream::connect(address).await.unwrap();
                write_all(&socket, head).await;
                waiting.push(socket);
            }
            while budget.waiting.value()["transport"] != 2 { tokio::task::yield_now().await; }
            let excess = TcpStream::connect(address).await.unwrap();
            timeout(Duration::from_millis(250), closed(&excess)).await.expect("overflow closes without a task or HTTP response");
            for socket in &waiting {
                timeout(Duration::from_secs(2), closed(socket)).await.expect("buffer wait consumes the first-header deadline");
            }
            assert_eq!(budget.read.load(Ordering::Relaxed), read_before, "waiters stay unread");
            assert_eq!(budget.waiting.value()["transport"], 0);
            assert_eq!(budget.reserved(), 2 * CONNECTION_BYTES);
            // The first head disarmed the accept deadline, even for a slow body.
            for socket in &active {
                write_all(socket, b"x").await;
                assert!(response(socket).await.starts_with(b"HTTP/1.1 200"));
            }
            stop.send(()).unwrap();
            server.await.unwrap().unwrap();
            assert_eq!(budget.reserved(), 0);
            assert_eq!(budget.waiting.value()["total"], 0);
        }).await.unwrap();
    }

    #[tokio::test]
    async fn connection_churn_and_task_failure_release_capacity() {
        timeout(Duration::from_secs(10), async {
            let budget = Budget::new(CONNECTION_BYTES);
            let router = Router::new()
                .route("/", axum::routing::get(|| async { "ok" }))
                .route(
                    "/panic",
                    axum::routing::get(|| async {
                        panic!("injected task failure");
                        #[allow(unreachable_code)]
                        ""
                    }),
                );
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let (stop, stopped) = tokio::sync::oneshot::channel();
            let server = tokio::spawn(serve(
                listener,
                router,
                ServerConfig::default().with_max_connections(1),
                budget.clone(),
                RequestLog::new(None, Vec::new()),
                async {
                    let _ = stopped.await;
                },
            ));
            for index in 0..64 {
                let socket = TcpStream::connect(address).await.unwrap();
                let path = if index == 16 { "/panic" } else { "/" };
                write_all(
                    &socket,
                    format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
                        .as_bytes(),
                )
                .await;
                if index == 16 {
                    closed(&socket).await;
                } else {
                    assert!(response(&socket).await.starts_with(b"HTTP/1.1 200"));
                }
                while budget.reserved() != 0 {
                    tokio::task::yield_now().await;
                }
            }
            stop.send(()).unwrap();
            server.await.unwrap().unwrap();
            assert_eq!(budget.reserved(), 0);
            assert_eq!(budget.waiting.value()["total"], 0);
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn keep_alive_head_timeout_returns_the_connection_permit() {
        timeout(Duration::from_secs(5), async {
            let budget = Budget::new(CONNECTION_BYTES);
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let (stop, stopped) = tokio::sync::oneshot::channel();
            let server = tokio::spawn(serve(
                listener,
                Router::new().route("/", axum::routing::get(|| async { "ok" })),
                ServerConfig::default()
                    .with_max_connections(1)
                    .with_header_read_timeout(Duration::from_millis(100)),
                budget.clone(),
                RequestLog::new(None, Vec::new()),
                async {
                    let _ = stopped.await;
                },
            ));
            let first = TcpStream::connect(address).await.unwrap();
            write_all(&first, b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n").await;
            // Reading to EOF waits for the next keep-alive head to expire.
            assert!(response(&first).await.starts_with(b"HTTP/1.1 200"));
            while budget.reserved() != 0 {
                tokio::task::yield_now().await;
            }
            let next = TcpStream::connect(address).await.unwrap();
            write_all(
                &next,
                b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
            )
            .await;
            assert!(response(&next).await.starts_with(b"HTTP/1.1 200"));
            stop.send(()).unwrap();
            server.await.unwrap().unwrap();
            assert_eq!(budget.reserved(), 0);
            assert_eq!(budget.waiting.value()["total"], 0);
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn shutdown_releases_active_and_waiting_connections() {
        timeout(Duration::from_secs(10), async {
            for abort in [false, true] {
                let entered = Arc::new(AtomicUsize::new(0));
                let budget = Budget::new(CONNECTION_BYTES);
                let router = Router::new().route("/", axum::routing::post({
                    let entered = entered.clone();
                    move |body: axum::body::Body| {
                        entered.fetch_add(1, Ordering::Relaxed);
                        async move { let _ = axum::body::to_bytes(body, 1).await; "ok" }
                    }
                }));
                let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
                let address = listener.local_addr().unwrap();
                let (stop, stopped) = tokio::sync::oneshot::channel();
                let server = tokio::spawn(serve(listener, router, ServerConfig::default().with_max_connections(2), budget.clone(), RequestLog::new(None, Vec::new()), async { let _ = stopped.await; }));
                let head = b"POST / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: 1\r\n\r\n";
                let active = TcpStream::connect(address).await.unwrap();
                write_all(&active, head).await;
                while entered.load(Ordering::Relaxed) == 0 { tokio::task::yield_now().await; }
                let waiting = TcpStream::connect(address).await.unwrap();
                write_all(&waiting, head).await;
                while budget.waiting.value()["transport"] != 1 { tokio::task::yield_now().await; }
                if abort {
                    server.abort();
                    assert!(server.await.unwrap_err().is_cancelled());
                    closed(&active).await;
                } else {
                    stop.send(()).unwrap();
                    closed(&waiting).await;
                    write_all(&active, b"x").await;
                    assert!(response(&active).await.starts_with(b"HTTP/1.1 200"));
                    server.await.unwrap().unwrap();
                }
                closed(&waiting).await;
                while budget.reserved() != 0 { tokio::task::yield_now().await; }
                assert_eq!(budget.waiting.value()["total"], 0);
                assert_eq!(entered.load(Ordering::Relaxed), 1, "shutdown never serves a waiter");
            }
        }).await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn accept_errors_retry_without_discarding_established_requests() {
        timeout(Duration::from_secs(10), async {
            let entered = Arc::new(AtomicUsize::new(0));
            let budget = Budget::new(2 * CONNECTION_BYTES);
            let router = Router::new().route("/", axum::routing::post({
                let entered = entered.clone();
                move |body: axum::body::Body| {
                    entered.fetch_add(1, Ordering::Relaxed);
                    async move { axum::body::to_bytes(body, 1).await.unwrap(); "ok" }
                }
            }).get(|| async { "ok" }));
            let listener = Arc::new(TcpListener::bind("127.0.0.1:0").await.unwrap());
            let address = listener.local_addr().unwrap();
            let errors = [None, Some(io::Error::from(io::ErrorKind::ConnectionAborted)), Some(io::Error::from(io::ErrorKind::Interrupted)), Some(io::Error::from_raw_os_error(24)), Some(io::Error::from_raw_os_error(23)), Some(io::Error::from_raw_os_error(24))];
            let mut errors = std::collections::VecDeque::from(errors);
            let injected = Arc::new(AtomicUsize::new(0));
            let (stop, stopped) = tokio::sync::oneshot::channel();
            let server = tokio::spawn(serve_with_accept({
                let injected = injected.clone();
                move || {
                    let listener = listener.clone();
                    let error = errors.pop_front().flatten();
                    let injected = injected.clone();
                    async move {
                        if let Some(error) = error { injected.fetch_add(1, Ordering::Relaxed); Err(error) }
                        else { listener.accept().await }
                    }
                }
            }, router, ServerConfig::default().with_max_connections(4), budget.clone(), RequestLog::new(None, Vec::new()), async { let _ = stopped.await; }));
            let active = TcpStream::connect(address).await.unwrap();
            write_all(&active, b"POST / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: 1\r\n\r\n").await;
            while entered.load(Ordering::Relaxed) == 0 { tokio::task::yield_now().await; }
            let next = TcpStream::connect(address).await.unwrap();
            write_all(&next, b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n").await;
            assert!(response(&next).await.starts_with(b"HTTP/1.1 200"));
            assert_eq!(injected.load(Ordering::Relaxed), 5);
            write_all(&active, b"x").await;
            assert!(response(&active).await.starts_with(b"HTTP/1.1 200"));
            stop.send(()).unwrap();
            server.await.unwrap().unwrap();
            assert_eq!(budget.reserved(), 0);
            assert_eq!(budget.waiting.value()["total"], 0);
        }).await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn accept_exhaustion_backs_off_and_shutdown_interrupts_the_wait() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let budget = Budget::new(CONNECTION_BYTES);
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(serve_with_accept(
            {
                let attempts = attempts.clone();
                move || {
                    attempts.fetch_add(1, Ordering::Relaxed);
                    std::future::ready(Err(io::Error::from_raw_os_error(24)))
                }
            },
            Router::new(),
            ServerConfig::default(),
            budget.clone(),
            RequestLog::new(None, Vec::new()),
            async {
                let _ = stopped.await;
            },
        ));
        tokio::time::sleep(Duration::from_millis(250)).await;
        assert!(
            (1..=3).contains(&attempts.load(Ordering::Relaxed)),
            "retry must back off"
        );
        stop.send(()).unwrap();
        timeout(Duration::from_secs(1), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(budget.reserved(), 0);
        assert_eq!(budget.waiting.value()["total"], 0);
    }

    #[tokio::test]
    async fn nonretryable_accept_error_is_returned() {
        let error = serve_with_accept(
            || std::future::ready(Err(io::Error::from(io::ErrorKind::InvalidInput))),
            Router::new(),
            ServerConfig::default(),
            Budget::new(CONNECTION_BYTES),
            RequestLog::new(None, Vec::new()),
            std::future::pending(),
        )
        .await
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[tokio::test]
    async fn queued_connections_do_not_read_payloads_until_buffer_space_is_available() {
        timeout(Duration::from_secs(10), async {
            let entered = Arc::new(AtomicUsize::new(0));
            let intake = Arc::new(crate::intake::Intake::new(65536, Duration::from_secs(5)));
            let held_intake = intake.read(axum::body::Body::empty(), 32768, true, 3 * 32768).await.unwrap();
            let budget = Budget::new(2 * CONNECTION_BYTES);
            let router = Router::new().route("/", axum::routing::post({
                let entered = Arc::clone(&entered);
                let intake = Arc::clone(&intake);
                let waiting = Arc::clone(&budget.waiting);
                move |body: axum::body::Body| {
                    let entered = Arc::clone(&entered);
                    let intake = Arc::clone(&intake);
                    let waiting = Arc::clone(&waiting);
                    async move {
                        entered.fetch_add(1, Ordering::Relaxed);
                        let _waiting = waiting.enter(crate::waiting::Stage::Intake);
                        let _body = intake.read(body, 32768, true, 3 * 32768).await.unwrap();
                        "ok"
                    }
                }
            }));
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let (stop, stopped) = tokio::sync::oneshot::channel();
            let server = tokio::spawn(serve(listener, router, ServerConfig::default(), Arc::clone(&budget), RequestLog::new(None, Vec::new()), async { let _ = stopped.await; }));
            let wire = format!("POST / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: 32768\r\n\r\n{}", "x".repeat(32768));
            let mut clients = Vec::new();
            for _ in 0..2 {
                let socket = TcpStream::connect(address).await.unwrap();
                write_all(&socket, wire.as_bytes()).await;
                clients.push(socket);
            }
            while entered.load(Ordering::Relaxed) != 2 { tokio::task::yield_now().await; }
            tokio::time::sleep(Duration::from_millis(50)).await;
            let read_before = budget.read.load(Ordering::Relaxed);
            for _ in 0..16 {
                let socket = TcpStream::connect(address).await.unwrap();
                write_all(&socket, wire.as_bytes()).await;
                clients.push(socket);
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
            let read_after = budget.read.load(Ordering::Relaxed);
            assert_eq!(read_after, read_before, "queued body prefixes were read without a buffer reservation");
            assert_eq!(budget.reserved(), 2 * CONNECTION_BYTES);
            assert_eq!(budget.waiting.value()["transport"], 16);
            drop(clients.pop());
            while budget.waiting.value()["transport"] != 15 { tokio::task::yield_now().await; }
            assert_eq!(budget.waiting.value()["intake"], 2);
            assert!(intake.prometheus().contains("chop_intake_used_bytes 65536\n"));
            assert!(budget.prometheus().contains("chop_transport_buffer_reserved_bytes 2097152\n"));
            drop(held_intake);
            for socket in clients {
                let mut response = Vec::new();
                loop {
                    socket.readable().await.unwrap();
                    let mut buf = [0; 4096];
                    match socket.try_read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => response.extend_from_slice(&buf[..n]),
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {},
                        Err(error) => panic!("{error}"),
                    }
                }
                assert!(response.starts_with(b"HTTP/1.1 200"), "{}", String::from_utf8_lossy(&response));
            }
            stop.send(()).unwrap();
            server.await.unwrap().unwrap();
            assert_eq!(budget.reserved(), 0);
            assert_eq!(budget.waiting.value()["total"], 0);
        }).await.unwrap();
    }
}
