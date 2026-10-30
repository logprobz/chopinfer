//! HTTP/1 connections with explicit header deadlines and bounded task ownership.
use std::{
    future::Future,
    io,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
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
    let capacity = config
        .max_connections
        .map(|limit| Arc::new(Semaphore::new(limit)));
    let (stop, stopped) = watch::channel(false);
    let mut connections = JoinSet::new();
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            () = &mut shutdown => break,
            Some(_) = connections.join_next(), if !connections.is_empty() => {},
            accepted = listener.accept() => {
                let (socket, peer) = accepted?;
                let permit = match &capacity {
                    Some(capacity) => match Arc::clone(capacity).try_acquire_owned() {
                        Ok(permit) => Some(permit),
                        Err(_) => {
                            drop(socket);
                            log.transport_failure(Some(peer), crate::request_log::connection_limit_detail());
                            continue;
                        }
                    },
                    None => None,
                };
                let service = TowerToHyperService::new(router.clone().layer(axum::Extension(peer)));
                // Bytes this connection read, and how many of them its
                // finished exchanges read, to tell a head that never arrived
                // from an idle keep-alive connection.
                let read = Arc::new(AtomicU64::new(0));
                let settled = Arc::new(AtomicU64::new(0));
                let exchange = (Arc::clone(&read), Arc::clone(&settled));
                let service = hyper::service::service_fn(move |request| {
                    let (read, settled) = (Arc::clone(&exchange.0), Arc::clone(&exchange.1));
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
                        permit = Arc::clone(&budget.slots).acquire_owned() => permit.expect("transport budget remains open"),
                        _ = stopped.changed() => return,
                        _ = peer_closed(&socket) => return,
                    };
                    drop(waiting);
                    let mut builder = http1::Builder::new();
                    builder.timer(TokioTimer::new()).header_read_timeout(config.header_read_timeout).max_buf_size(READ_BUFFER);
                    let socket = MeteredSocket { socket, budget, read: Arc::clone(&read) };
                    let connection = builder.serve_connection(TokioIo::new(socket), service);
                    tokio::pin!(connection);
                    let result = tokio::select! {
                        result = &mut connection => result,
                        _ = stopped.changed() => {
                            connection.as_mut().graceful_shutdown();
                            connection.await
                        }
                    };
                    let partial = read.load(Ordering::Relaxed) > settled.load(Ordering::Relaxed);
                    if let Err(error) = result
                        && let Some(detail) = head_failure(&error, partial)
                    {
                        log.transport_failure(Some(peer), detail);
                    }
                });
            }
        }
    }
    let _ = stop.send(true);
    while connections.join_next().await.is_some() {}
    Ok(())
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
