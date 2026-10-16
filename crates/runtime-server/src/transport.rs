//! HTTP/1 connections with explicit header deadlines and bounded task ownership.
use std::{future::Future, io, sync::Arc};

use axum::Router;
use hyper::server::conn::http1;
use hyper_util::{
    rt::{TokioIo, TokioTimer},
    service::TowerToHyperService,
};
use tokio::{
    net::TcpListener,
    sync::{Semaphore, watch},
    task::JoinSet,
};

use crate::ServerConfig;

pub(crate) async fn serve(
    listener: TcpListener,
    router: Router,
    config: ServerConfig,
    budget: Arc<Budget>,
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
                        Err(_) => { drop(socket); continue; }
                    },
                    None => None,
                };
                let service = TowerToHyperService::new(router.clone().layer(axum::Extension(peer)));
                let mut stopped = stopped.clone();
                let budget = Arc::clone(&budget);
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
                    let connection = builder.serve_connection(TokioIo::new(MeteredSocket { socket, budget }), service);
                    tokio::pin!(connection);
                    tokio::select! {
                        _ = &mut connection => {},
                        _ = stopped.changed() => {
                            connection.as_mut().graceful_shutdown();
                            let _ = connection.await;
                        }
                    }
                });
            }
        }
    }
    let _ = stop.send(true);
    while connections.join_next().await.is_some() {}
    Ok(())
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
}

impl tokio::io::AsyncRead for MeteredSocket {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<io::Result<()>> {
        let before = buf.filled().len();
        let result = std::pin::Pin::new(&mut self.socket).poll_read(cx, buf);
        self.budget.read.fetch_add(
            (buf.filled().len() - before) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        result
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
            let server = tokio::spawn(serve(listener, router, ServerConfig::default(), Arc::clone(&budget), async { let _ = stopped.await; }));
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
