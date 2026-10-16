//! A blocking HTTP/1.1 GET over TCP or a Unix socket.
//!
//! chopinfer-top reads `/health` and `/metrics` from servers and a few
//! read-only routes from the Docker daemon. Each request opens one
//! connection (`Connection: close`). One deadline covers name resolution,
//! connecting, sending and reading, and headers and bodies beyond fixed
//! sizes are refused, so a slow or misbehaving peer cannot stall a poller or
//! grow its memory. Header parsing
//! follows `runtime-eval`'s client; this is a separate copy so the crate
//! links no workspace crate.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpStream, ToSocketAddrs};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// Largest response header block read.
const MAX_HEAD: usize = 64 * 1024;
/// Largest `/metrics` body.
pub const METRICS_LIMIT: usize = 16 * 1024 * 1024;
/// Largest `/health` body.
pub const HEALTH_LIMIT: usize = 1024 * 1024;
/// Largest Docker API body.
pub const DOCKER_LIMIT: usize = 4 * 1024 * 1024;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(1);
const DEADLINE: Duration = Duration::from_secs(3);
/// Most informational (1xx) heads skipped before the final response.
const MAX_INFORMATIONAL: usize = 8;
/// Most helper threads of one kind left blocked past their deadline. Each
/// keeps its thread, and a connect its socket, until the call returns; past
/// this many, requests fail at once instead of adding more. Requests already
/// waiting when the count is reached can still add theirs, so the bound is
/// this plus the number of concurrent callers (pollers and discovery).
const MAX_STALLED: usize = 8;
/// Host lookups still blocked past their request's deadline.
static RESOLVING: AtomicUsize = AtomicUsize::new(0);
/// Unix socket connects still blocked past their request's deadline.
static CONNECTING: AtomicUsize = AtomicUsize::new(0);

/// Where a request goes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Endpoint {
    /// A host name or address and a port.
    Tcp { host: String, port: u16 },
    /// A resolved socket address, as discovery finds them.
    Address(SocketAddr),
    /// A Unix socket path, such as the Docker daemon's.
    Unix(PathBuf),
}

impl Endpoint {
    /// `host:port`, with an IPv6 host in brackets, for display and the
    /// `Host` header.
    pub fn authority(&self) -> String {
        match self {
            Self::Tcp { host, port } if host.contains(':') => format!("[{host}]:{port}"),
            Self::Tcp { host, port } => format!("{host}:{port}"),
            Self::Address(address) => address.to_string(),
            Self::Unix(_) => "localhost".to_owned(),
        }
    }
}

/// Why a request failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HttpError {
    /// Unreachable, refused, timed out or broken off.
    Transport(String),
    /// The peer answered something that is not a bounded HTTP response.
    Protocol(String),
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transport(message) | Self::Protocol(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for HttpError {}

/// One complete response.
#[derive(Clone, Debug)]
pub struct Response {
    pub status: u16,
    /// Header names in lowercase, in the order received.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Response {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }
}

/// Resolves a host name to its addresses.
type Resolver = fn(&str, u16) -> std::io::Result<Vec<SocketAddr>>;

/// The system resolver.
fn resolve(host: &str, port: u16) -> std::io::Result<Vec<SocketAddr>> {
    Ok((host, port).to_socket_addrs()?.collect())
}

/// GETs `path` from `endpoint`, reading at most `limit` body bytes.
pub fn get(
    endpoint: &Endpoint,
    path: &str,
    bearer: Option<&str>,
    limit: usize,
) -> Result<Response, HttpError> {
    request(endpoint, path, bearer, limit, resolve)
}

/// [`get`], resolving host names with `resolver`, which tests replace.
fn request(
    endpoint: &Endpoint,
    path: &str,
    bearer: Option<&str>,
    limit: usize,
    resolver: Resolver,
) -> Result<Response, HttpError> {
    let started = Instant::now();
    let transport = |error: std::io::Error| HttpError::Transport(error.to_string());
    let stream: Box<dyn Stream> = match endpoint {
        Endpoint::Tcp { host, port } => {
            let addresses = match host.parse::<IpAddr>() {
                Ok(ip) => vec![SocketAddr::new(ip, *port)],
                Err(_) => {
                    let (name, port) = (host.clone(), *port);
                    within(&RESOLVING, started, "resolving the host", move || {
                        resolver(&name, port)
                    })
                    .map_err(|error| {
                        HttpError::Transport(format!("cannot resolve {host}: {error}"))
                    })?
                }
            };
            Box::new(connect_any(&addresses, started)?)
        }
        Endpoint::Address(address) => Box::new(connect_any(&[*address], started)?),
        Endpoint::Unix(path) => {
            let path = path.clone();
            // A Unix connect blocks while the listener's backlog is full.
            Box::new(within(&CONNECTING, started, "connecting", move || {
                UnixStream::connect(path)
            })?)
        }
    };
    let mut stream = Deadline {
        inner: stream,
        started,
    };
    let mut head = format!(
        "GET {path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nAccept: */*\r\n",
        endpoint.authority()
    );
    if let Some(key) = bearer {
        head.push_str(&format!("Authorization: Bearer {key}\r\n"));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes()).map_err(transport)?;
    let mut reader = BufReader::new(stream);
    let (mut status, mut headers) = read_head(&mut reader)?;
    // Informational heads (103 Early Hints and the like) precede the final
    // response; 101 would switch protocols and is final here.
    let mut skipped = 0;
    while (100..200).contains(&status) && status != 101 {
        skipped += 1;
        if skipped > MAX_INFORMATIONAL {
            return Err(HttpError::Protocol(format!(
                "more than {MAX_INFORMATIONAL} informational responses"
            )));
        }
        (status, headers) = read_head(&mut reader)?;
    }
    let header = |name: &str| {
        headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.to_ascii_lowercase())
    };
    let mut body = Vec::new();
    let too_large = || HttpError::Protocol(format!("{path}: the body exceeds {limit} bytes"));
    let empty = status == 204 || status == 304;
    if empty {
        // These statuses have no body whatever their headers say.
    } else if header("transfer-encoding").is_some_and(|value| value.contains("chunked")) {
        read_chunked(&mut reader, &mut body, limit).map_err(|error| match error {
            BodyError::TooLarge => too_large(),
            BodyError::Io(error) => transport(error),
        })?;
    } else if let Some(length) = header("content-length") {
        let length = length
            .trim()
            .parse::<usize>()
            .map_err(|_| HttpError::Protocol(format!("{path}: invalid content-length")))?;
        if length > limit {
            return Err(too_large());
        }
        body.resize(length, 0);
        reader.read_exact(&mut body).map_err(transport)?;
    } else {
        reader
            .take(limit as u64 + 1)
            .read_to_end(&mut body)
            .map_err(transport)?;
        if body.len() > limit {
            return Err(too_large());
        }
    }
    Ok(Response {
        status,
        headers,
        body,
    })
}

trait Stream: Read + Write {
    fn set_timeouts(&self, timeout: Duration) -> std::io::Result<()>;
}

impl Stream for TcpStream {
    fn set_timeouts(&self, timeout: Duration) -> std::io::Result<()> {
        self.set_read_timeout(Some(timeout))?;
        self.set_write_timeout(Some(timeout))
    }
}

impl Stream for UnixStream {
    fn set_timeouts(&self, timeout: Duration) -> std::io::Result<()> {
        self.set_read_timeout(Some(timeout))?;
        self.set_write_timeout(Some(timeout))
    }
}

/// The time left before the request deadline, or a timeout error.
fn remaining(started: Instant) -> Result<Duration, HttpError> {
    let left = DEADLINE.saturating_sub(started.elapsed());
    if left.is_zero() {
        return Err(HttpError::Transport(
            "the request deadline passed".to_owned(),
        ));
    }
    Ok(left)
}

/// Runs a blocking step that has no timeout of its own on a helper thread,
/// waiting for it only until the request deadline. A step still blocked then
/// is left to finish on its own, counted in `stalled` until it does; its
/// result is dropped. While `MAX_STALLED` are left, the step is refused.
fn within<T: Send + 'static>(
    stalled: &'static AtomicUsize,
    started: Instant,
    step: &str,
    work: impl FnOnce() -> std::io::Result<T> + Send + 'static,
) -> Result<T, HttpError> {
    const RUNNING: u8 = 0;
    const DONE: u8 = 1;
    const ABANDONED: u8 = 2;
    let left = remaining(started)?;
    if stalled.load(Ordering::SeqCst) >= MAX_STALLED {
        return Err(HttpError::Transport(format!(
            "{MAX_STALLED} earlier attempts at {step} are still blocked"
        )));
    }
    let state = Arc::new(AtomicU8::new(RUNNING));
    let shared = Arc::clone(&state);
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let result = work();
        if shared.swap(DONE, Ordering::SeqCst) == ABANDONED {
            stalled.fetch_sub(1, Ordering::SeqCst);
        }
        let _ = sender.send(result);
    });
    match receiver.recv_timeout(left) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => Err(HttpError::Transport(error.to_string())),
        Err(_) => {
            // Counted before it is marked, so the helper's decrement can
            // never come first.
            stalled.fetch_add(1, Ordering::SeqCst);
            if state
                .compare_exchange(RUNNING, ABANDONED, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
            {
                // It finished just now.
                stalled.fetch_sub(1, Ordering::SeqCst);
            }
            Err(HttpError::Transport(format!(
                "{step} did not finish before the request deadline"
            )))
        }
    }
}

/// Connects to the first address that accepts, each attempt bounded by the
/// connect timeout and by what is left of the request deadline.
fn connect_any(addresses: &[SocketAddr], started: Instant) -> Result<TcpStream, HttpError> {
    let mut last = None;
    for address in addresses {
        let left = remaining(started)?;
        match TcpStream::connect_timeout(address, left.min(CONNECT_TIMEOUT)) {
            Ok(stream) => return Ok(stream),
            Err(error) => last = Some(error),
        }
    }
    Err(HttpError::Transport(match last {
        Some(error) => error.to_string(),
        None => "no address".to_owned(),
    }))
}

/// A stream whose every read and write ends by the request deadline.
struct Deadline {
    inner: Box<dyn Stream>,
    started: Instant,
}

impl Deadline {
    fn arm(&self) -> std::io::Result<()> {
        let remaining = DEADLINE.saturating_sub(self.started.elapsed());
        if remaining.is_zero() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "the request deadline passed",
            ));
        }
        self.inner.set_timeouts(remaining)
    }
}

impl Read for Deadline {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.arm()?;
        self.inner.read(buffer)
    }
}

impl Write for Deadline {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.arm()?;
        self.inner.write(buffer)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn read_head(reader: &mut impl BufRead) -> Result<(u16, Vec<(String, String)>), HttpError> {
    let mut read = 0;
    let mut line = String::new();
    let mut next = |line: &mut String| -> Result<(), HttpError> {
        line.clear();
        let count = reader
            .take((MAX_HEAD - read) as u64)
            .read_line(line)
            .map_err(|error| HttpError::Transport(error.to_string()))?;
        read += count;
        if count == 0 {
            return Err(HttpError::Transport(
                "the connection closed before a response".to_owned(),
            ));
        }
        if !line.ends_with('\n') {
            return Err(HttpError::Protocol(format!(
                "response headers exceed {MAX_HEAD} bytes"
            )));
        }
        Ok(())
    };
    next(&mut line)?;
    let status = line
        .split_whitespace()
        .nth(1)
        .filter(|_| line.starts_with("HTTP/1."))
        .and_then(|code| code.parse::<u16>().ok())
        .ok_or_else(|| HttpError::Protocol("not an HTTP/1 response".to_owned()))?;
    let mut headers = Vec::new();
    loop {
        next(&mut line)?;
        let text = line.trim_end_matches(['\r', '\n']);
        if text.is_empty() {
            return Ok((status, headers));
        }
        let (name, value) = text
            .split_once(':')
            .ok_or_else(|| HttpError::Protocol(format!("malformed header {text:?}")))?;
        headers.push((name.trim().to_ascii_lowercase(), value.trim().to_owned()));
    }
}

enum BodyError {
    TooLarge,
    Io(std::io::Error),
}

impl From<std::io::Error> for BodyError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

fn read_chunked(
    reader: &mut impl BufRead,
    body: &mut Vec<u8>,
    limit: usize,
) -> Result<(), BodyError> {
    let invalid = |message: &str| {
        BodyError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            message.to_owned(),
        ))
    };
    let mut line = String::new();
    loop {
        line.clear();
        if reader.take(1024).read_line(&mut line)? == 0 {
            return Err(invalid("the connection closed inside a chunked body"));
        }
        let size = line.trim_end().split(';').next().unwrap_or_default();
        let size =
            usize::from_str_radix(size.trim(), 16).map_err(|_| invalid("invalid chunk size"))?;
        if size == 0 {
            // Trailers end at an empty line.
            loop {
                line.clear();
                if reader.take(1024).read_line(&mut line)? == 0 || line.trim_end().is_empty() {
                    return Ok(());
                }
            }
        }
        if size > limit.saturating_sub(body.len()) {
            return Err(BodyError::TooLarge);
        }
        let start = body.len();
        body.resize(start + size, 0);
        reader.read_exact(&mut body[start..])?;
        let mut end = [0u8; 2];
        reader.read_exact(&mut end)?;
        if &end != b"\r\n" {
            return Err(invalid("a chunk does not end with CRLF"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::os::unix::net::UnixListener;
    use std::thread;

    /// Serves `reply` once to the first connection after reading its
    /// request head, and returns the head it read.
    fn serve_once(reply: Vec<u8>) -> (Endpoint, thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let head = read_request(&mut socket);
            socket.write_all(&reply).unwrap();
            head
        });
        (Endpoint::Address(address), handle)
    }

    fn read_request(socket: &mut impl Read) -> String {
        let mut head = Vec::new();
        let mut byte = [0u8; 1];
        while !head.ends_with(b"\r\n\r\n") {
            if socket.read(&mut byte).unwrap() == 0 {
                break;
            }
            head.push(byte[0]);
        }
        String::from_utf8(head).unwrap()
    }

    #[test]
    fn reads_a_content_length_body_and_sends_the_request_head() {
        let (endpoint, server) = serve_once(
            b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nX-Chopinfer-Startup-Nonce: ab\r\n\r\nhello"
                .to_vec(),
        );
        let response = get(&endpoint, "/health", Some("key"), 100).unwrap();
        assert_eq!(response.status, 200);
        assert_eq!(response.body, b"hello");
        assert_eq!(response.header("x-chopinfer-startup-nonce"), Some("ab"));
        let head = server.join().unwrap();
        assert!(head.starts_with("GET /health HTTP/1.1\r\n"), "{head}");
        assert!(head.contains("Connection: close\r\n"), "{head}");
        assert!(head.contains("Authorization: Bearer key\r\n"), "{head}");
    }

    #[test]
    fn reads_chunked_and_close_delimited_bodies() {
        let (endpoint, server) = serve_once(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n2;x=y\r\nde\r\n0\r\n\r\n"
                .to_vec(),
        );
        assert_eq!(get(&endpoint, "/", None, 100).unwrap().body, b"abcde");
        server.join().unwrap();
        let (endpoint, server) = serve_once(b"HTTP/1.0 503 Busy\r\n\r\nnot ready".to_vec());
        let response = get(&endpoint, "/", None, 100).unwrap();
        assert_eq!(
            (response.status, response.body.as_slice()),
            (503, &b"not ready"[..])
        );
        server.join().unwrap();
    }

    #[test]
    fn refuses_bodies_and_headers_beyond_their_limits() {
        for reply in [
            b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\nhello world".to_vec(),
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nb\r\nhello world\r\n0\r\n\r\n"
                .to_vec(),
            b"HTTP/1.1 200 OK\r\n\r\nhello world".to_vec(),
        ] {
            let (endpoint, server) = serve_once(reply);
            let error = get(&endpoint, "/metrics", None, 10).unwrap_err();
            assert!(
                matches!(error, HttpError::Protocol(ref message) if message.contains("exceeds 10 bytes")),
                "{error:?}"
            );
            server.join().unwrap();
        }
        let mut huge = b"HTTP/1.1 200 OK\r\nX: ".to_vec();
        huge.resize(MAX_HEAD + 10, b'a');
        huge.extend_from_slice(b"\r\n\r\n");
        let (endpoint, server) = serve_once(huge);
        assert!(matches!(
            get(&endpoint, "/", None, 10),
            Err(HttpError::Protocol(_))
        ));
        server.join().unwrap();
    }

    #[test]
    fn a_huge_chunk_size_is_refused_not_overflowed() {
        let (endpoint, server) = serve_once(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n1\r\nx\r\nffffffffffffffff\r\n"
                .to_vec(),
        );
        let error = get(&endpoint, "/metrics", None, 100).unwrap_err();
        assert!(
            matches!(error, HttpError::Protocol(ref message) if message.contains("exceeds")),
            "{error:?}"
        );
        server.join().unwrap();
    }

    #[test]
    fn informational_heads_are_skipped() {
        let (endpoint, server) = serve_once(
            b"HTTP/1.1 103 Early Hints\r\nLink: </x>\r\n\r\nHTTP/1.1 200 OK\r\nX-Chopinfer-Startup-Nonce: n\r\nContent-Length: 2\r\n\r\nok"
                .to_vec(),
        );
        let response = get(&endpoint, "/health", None, 100).unwrap();
        assert_eq!(response.status, 200);
        assert_eq!(response.header("x-chopinfer-startup-nonce"), Some("n"));
        assert_eq!(response.body, b"ok");
        server.join().unwrap();
    }

    #[test]
    fn steps_without_timeouts_end_at_the_deadline_and_stalls_are_capped() {
        static STALLED: AtomicUsize = AtomicUsize::new(0);
        let started = Instant::now();
        let finished = within(&STALLED, started, "resolving the host", || {
            thread::sleep(Duration::from_millis(300));
            Ok(())
        });
        assert!(finished.is_ok());
        assert_eq!(STALLED.load(Ordering::SeqCst), 0);
        if let Some(late) = Instant::now().checked_sub(DEADLINE) {
            assert!(within(&STALLED, late, "connecting", || Ok(())).is_err());
        }
        // Steps blocked until `release` closes, each given 50 ms.
        let (release, blocked) = mpsc::channel::<()>();
        let blocked = Arc::new(std::sync::Mutex::new(blocked));
        let almost = || Instant::now() - DEADLINE + Duration::from_millis(50);
        for _ in 0..MAX_STALLED {
            let blocked = Arc::clone(&blocked);
            let error = within(&STALLED, almost(), "connecting", move || {
                let _ = blocked.lock().unwrap().recv();
                Ok(())
            })
            .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("did not finish before the request deadline")
            );
        }
        assert_eq!(STALLED.load(Ordering::SeqCst), MAX_STALLED);
        // The next attempt is refused without a thread.
        let refused = Instant::now();
        let error = within(&STALLED, Instant::now(), "connecting", || Ok(())).unwrap_err();
        assert!(refused.elapsed() < Duration::from_millis(50));
        assert!(error.to_string().contains("still blocked"), "{error}");
        drop(release);
        let until = Instant::now() + Duration::from_secs(5);
        while STALLED.load(Ordering::SeqCst) > 0 && Instant::now() < until {
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(STALLED.load(Ordering::SeqCst), 0);
    }

    /// Whether `blocked_resolver` may return.
    static RESOLVER_OPEN: std::sync::Mutex<bool> = std::sync::Mutex::new(false);
    static RESOLVER_OPENED: std::sync::Condvar = std::sync::Condvar::new();

    /// A lookup that blocks until the test opens it, or for ten seconds,
    /// so a lookup made on the caller's thread fails the test, not hangs it.
    fn blocked_resolver(_: &str, _: u16) -> std::io::Result<Vec<SocketAddr>> {
        let open = RESOLVER_OPEN.lock().unwrap();
        drop(
            RESOLVER_OPENED
                .wait_timeout_while(open, Duration::from_secs(10), |open| !*open)
                .unwrap(),
        );
        Ok(Vec::new())
    }

    #[test]
    fn a_host_lookup_that_blocks_ends_at_the_deadline() {
        let endpoint = Endpoint::Tcp {
            host: "blocked.invalid".to_owned(),
            port: 8310,
        };
        let started = Instant::now();
        let error = request(&endpoint, "/health", None, 100, blocked_resolver).unwrap_err();
        assert!(started.elapsed() < DEADLINE + Duration::from_millis(500));
        assert!(
            error
                .to_string()
                .contains("resolving the host did not finish"),
            "{error}"
        );
        assert_eq!(RESOLVING.load(Ordering::SeqCst), 1);
        *RESOLVER_OPEN.lock().unwrap() = true;
        RESOLVER_OPENED.notify_all();
        let until = Instant::now() + Duration::from_secs(5);
        while RESOLVING.load(Ordering::SeqCst) > 0 && Instant::now() < until {
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(RESOLVING.load(Ordering::SeqCst), 0);
    }

    /// A Unix listener with a backlog of one, which nothing accepts from,
    /// so connects block once two are queued. std listens with a backlog
    /// of 4096, so Python makes it.
    #[test]
    fn a_unix_connect_blocked_on_a_full_backlog_ends_at_the_deadline() {
        use std::io::BufRead as _;
        use std::process::{Command, Stdio};
        let directory =
            std::env::temp_dir().join(format!("chopinfer-top-backlog-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("full.sock");
        let mut listener = Command::new("python3")
            .args([
                "-I",
                "-c",
                "import socket, sys\n\
                 s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n\
                 s.bind(sys.argv[1]); s.listen(1)\n\
                 print('ready', flush=True)\n\
                 sys.stdin.read()\n",
            ])
            .arg(&path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut ready = String::new();
        BufReader::new(listener.stdout.take().unwrap())
            .read_line(&mut ready)
            .unwrap();
        assert_eq!(ready.trim(), "ready");
        // Fill the queue; this thread blocks on the first connect that
        // does not fit, and returns when the listener exits.
        let filler = path.clone();
        thread::spawn(move || {
            let held = (0..3)
                .map_while(|_| UnixStream::connect(&filler).ok())
                .collect::<Vec<_>>();
            drop(held);
        });
        thread::sleep(Duration::from_millis(300));
        let started = Instant::now();
        let error = get(&Endpoint::Unix(path), "/_ping", None, 100).unwrap_err();
        assert!(started.elapsed() < DEADLINE + Duration::from_millis(500));
        assert!(
            error.to_string().contains("connecting did not finish"),
            "{error}"
        );
        assert_eq!(CONNECTING.load(Ordering::SeqCst), 1);
        // The listener goes away: the blocked connect returns, and its
        // helper is no longer counted.
        drop(listener.stdin.take());
        listener.wait().unwrap();
        let until = Instant::now() + Duration::from_secs(5);
        while CONNECTING.load(Ordering::SeqCst) > 0 && Instant::now() < until {
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(CONNECTING.load(Ordering::SeqCst), 0);
        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn a_silent_peer_times_out_and_a_closed_port_is_a_transport_error() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = Endpoint::Address(listener.local_addr().unwrap());
        let holder = thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            thread::sleep(Duration::from_secs(4));
            drop(socket);
        });
        let started = Instant::now();
        assert!(matches!(
            get(&endpoint, "/", None, 10),
            Err(HttpError::Transport(_))
        ));
        assert!(started.elapsed() < Duration::from_millis(3500));
        holder.join().unwrap();

        let closed = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        assert!(matches!(
            get(&Endpoint::Address(closed), "/", None, 10),
            Err(HttpError::Transport(_))
        ));
    }

    #[test]
    fn talks_over_a_unix_socket() {
        let directory =
            std::env::temp_dir().join(format!("chopinfer-top-http-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("docker.sock");
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();
        let server = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let head = read_request(&mut socket);
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nApi-Version: 1.47\r\nContent-Length: 2\r\n\r\nOK")
                .unwrap();
            head
        });
        let response = get(&Endpoint::Unix(path.clone()), "/_ping", None, DOCKER_LIMIT).unwrap();
        assert_eq!(response.header("api-version"), Some("1.47"));
        assert!(server.join().unwrap().contains("Host: localhost\r\n"));
        std::fs::remove_dir_all(&directory).unwrap();
    }
}
