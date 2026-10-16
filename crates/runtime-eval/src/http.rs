//! A blocking HTTP/1.1 client for the server's OpenAI-compatible endpoints
//! and its serving identity, as `tools/qual/common.py` `Client` uses them.
//!
//! Plain `http://` only: the server under test listens locally. Each request
//! opens one connection (`Connection: close`). Every response's
//! `x-chopinfer-checkpoint-identity` and `x-chopinfer-startup-nonce`
//! headers are recorded, so results bound to a server's `/health` identity
//! can be refused when another process wrote any response
//! ([`Client::confirm_live`]).

use std::collections::BTreeSet;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpStream};
use std::path::Path;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::declaration::CheckpointPin;
use crate::json;

pub const NONCE_HEADER: &str = "x-chopinfer-startup-nonce";
pub const CHECKPOINT_HEADER: &str = "x-chopinfer-checkpoint-identity";
/// Largest response header block read.
const MAX_HEAD: usize = 64 * 1024;

/// What went wrong talking to the server.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HttpError {
    /// Unreachable, timed out, or the connection broke: an environment
    /// error.
    Transport(String),
    /// The server answered with an error status.
    Status { status: u16, body: String },
    /// The server answered something that is not HTTP or not JSON.
    Protocol(String),
    /// The server does not serve the requested model.
    Unavailable(String),
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transport(message) | Self::Protocol(message) | Self::Unavailable(message) => {
                formatter.write_str(message)
            }
            Self::Status { status, body } => write!(formatter, "HTTP {status}: {body}"),
        }
    }
}

impl std::error::Error for HttpError {}

/// One response.
#[derive(Clone, Debug)]
pub struct Response {
    pub status: u16,
    /// Header names in lowercase, in the order received.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    pub seconds: f64,
}

impl Response {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    pub fn json(&self) -> Result<Value, HttpError> {
        json::parse(&self.body).map_err(|error| HttpError::Protocol(format!("response: {error}")))
    }
}

/// A response's status, headers and body reader.
type Head = (u16, Vec<(String, String)>, Box<dyn Read>);

/// One server-sent event's data with the seconds since the request.
pub type Event = (f64, Value);

/// The address of a server: host, port and a path prefix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Endpoint {
    pub host: String,
    pub port: u16,
    pub prefix: String,
}

impl Endpoint {
    /// Parses `http://HOST[:PORT][/PREFIX]`.
    pub fn parse(url: &str) -> Result<Self, String> {
        let rest = url
            .strip_prefix("http://")
            .ok_or_else(|| format!("{url}: only http:// URLs are supported"))?;
        let (authority, prefix) = match rest.find('/') {
            Some(index) => (&rest[..index], rest[index..].trim_end_matches('/')),
            None => (rest, ""),
        };
        let invalid_port = |port: &str| format!("{url}: invalid port {port}");
        let (host, port) = if let Some(bracketed) = authority.strip_prefix('[') {
            let (host, rest) = bracketed
                .split_once(']')
                .ok_or_else(|| format!("{url}: unterminated IPv6 address"))?;
            match rest.strip_prefix(':') {
                Some(port) => (host, port.parse().map_err(|_| invalid_port(port))?),
                None if rest.is_empty() => (host, 80),
                None => return Err(format!("{url}: invalid authority")),
            }
        } else {
            match authority.rsplit_once(':') {
                Some((host, port)) => (host, port.parse().map_err(|_| invalid_port(port))?),
                None => (authority, 80),
            }
        };
        if host.is_empty() {
            return Err(format!("{url}: no host"));
        }
        Ok(Self {
            host: host.into(),
            port,
            prefix: prefix.into(),
        })
    }

    fn authority(&self) -> String {
        if self.host.contains(':') {
            format!("[{}]:{}", self.host, self.port)
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }
}

/// What a server reported about itself when it was read.
#[derive(Clone, Debug, PartialEq)]
pub struct ServerIdentity {
    pub url: String,
    pub model: String,
    pub models: Vec<String>,
    /// The `/health` body, `None` when it could not be read.
    pub health: Option<Value>,
}

impl ServerIdentity {
    /// The serving identity `/health` reports (`common.live_identity`): it
    /// must name a checkpoint and a startup nonce.
    pub fn serving(&self) -> Result<&Value, String> {
        let serving = self
            .health
            .as_ref()
            .and_then(|health| health.get("serving"));
        match serving {
            Some(serving)
                if serving["checkpoint_identity"]
                    .as_str()
                    .is_some_and(|identity| !identity.is_empty())
                    && startup_nonce(&serving["startup_nonce"]) =>
            {
                Ok(serving)
            }
            _ => Err(format!(
                "the server at {} reports no checkpoint identity and startup nonce in /health",
                self.url
            )),
        }
    }

    pub fn to_json(&self) -> Value {
        serde_json::json!({"url": self.url, "model": self.model, "models": self.models,
                           "health": self.health})
    }
}

/// Legacy identities omit both fields. Options qualify only when the host
/// reports a complete value map and marks the declared defaults qualified.
pub fn require_qualified_model_options(serving: &Value) -> Result<(), String> {
    if serving.get("model_options").is_none() && serving.get("model_options_qualified").is_none() {
        return Ok(());
    }
    if serving
        .get("model_options")
        .and_then(Value::as_object)
        .is_some_and(|options| !options.is_empty())
        && serving.get("model_options_qualified") == Some(&Value::Bool(true))
    {
        Ok(())
    } else {
        Err("server model options are non-default, unqualified, or incompletely reported".into())
    }
}

/// A startup nonce as the host draws it: 128 random bits in lowercase hex.
pub fn startup_nonce(value: &Value) -> bool {
    value.as_str().is_some_and(|nonce| {
        nonce.len() == 32
            && nonce
                .bytes()
                .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    })
}

/// The live checkpoint a server was bound to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundServer {
    pub checkpoint_identity: String,
    pub startup_nonce: String,
    pub package_identity: Option<String>,
    pub metadata_identity: Option<String>,
    /// The live process's startup log, when the run was bound to one.
    pub server_log: Option<ServerLog>,
}

impl BoundServer {
    pub fn to_json(&self) -> Value {
        let mut value = serde_json::json!({"checkpoint_identity": self.checkpoint_identity,
                                           "package_identity": self.package_identity,
                                           "metadata_identity": self.metadata_identity,
                                           "startup_nonce": self.startup_nonce});
        if let Some(log) = &self.server_log {
            value["server_log"] = Value::from(log.path.as_str());
            value["server_log_sha256"] = Value::from(log.sha256.as_str());
        }
        value
    }

    /// Binds the server to its startup log `log` as well
    /// (`common.bind_checkpoint` with a log): the log must be that of the
    /// server listening at `url` and name the checkpoint and startup nonce
    /// `/health` reported, so a log another process left at the same
    /// address vouches for nothing.
    pub fn with_log(mut self, log: &Path, url: &str) -> Result<Self, BindError> {
        let logged = served_checkpoint(log, url)?;
        if (
            logged.checkpoint_identity.as_str(),
            logged.startup_nonce.as_str(),
        ) != (
            self.checkpoint_identity.as_str(),
            self.startup_nonce.as_str(),
        ) {
            return Err(BindError::Mismatch(format!(
                "{} is not the log of the live server at {url}: its checkpoint or startup nonce \
                 differs from /health",
                logged.log.path
            )));
        }
        self.server_log = Some(logged.log);
        Ok(self)
    }
}

/// A startup log a run was bound to: its path as given and its SHA-256.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerLog {
    pub path: String,
    pub sha256: String,
}

/// What a startup log says the server listening at an address loaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoggedCheckpoint {
    pub checkpoint_identity: String,
    pub startup_nonce: String,
    pub log: ServerLog,
}

/// The startup records that name the loaded checkpoint
/// (`common.CHECKPOINT_RECORDS`).
const CHECKPOINT_RECORDS: [&str; 4] = [
    "direct_package",
    "direct_load",
    "weight_plan",
    "serving_identity",
];

/// The checkpoint identity and startup nonce the server listening at `url`
/// reported in its startup log `log` (`common.served_checkpoint`): the
/// last `listening` record names the URL's address, every direct package,
/// direct load, weight plan and serving identity record names the same
/// checkpoint, and exactly one serving identity record names the nonce.
pub fn served_checkpoint(log: &Path, url: &str) -> Result<LoggedCheckpoint, BindError> {
    let shown = python_path(log);
    let unidentified = |message: String| BindError::Unidentified(message);
    let bytes = std::fs::read(log).map_err(|error| unidentified(format!("{shown}: {error}")))?;
    // Read as Python's text mode reads it: universal newlines.
    let text = String::from_utf8_lossy(&bytes)
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    // `urlsplit(url).netloc`.
    let after = url.split_once("://").map_or("", |(_, rest)| rest);
    let address = after
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default()
        .to_owned();
    let mut listening = None;
    for line in python_lines(&text) {
        if line.starts_with('{') && line.contains("\"status\":\"listening\"") {
            let record = json::parse(line.as_bytes())
                .map_err(|error| unidentified(format!("{shown}: {error}")))?;
            listening = Some(record);
        }
    }
    if listening.as_ref().and_then(|record| record.get("address"))
        != Some(&Value::from(address.as_str()))
    {
        return Err(BindError::Mismatch(format!(
            "{shown} is not the log of the server at {address}"
        )));
    }
    let mut identities: Vec<Value> = Vec::new();
    let mut nonces = Vec::new();
    // `^startup (KIND): (\{.*\})$` per line, as re.M matches it.
    for line in text.split('\n') {
        let Some(rest) = line.strip_prefix("startup ") else {
            continue;
        };
        let Some((kind, record)) = CHECKPOINT_RECORDS.iter().find_map(|kind| {
            rest.strip_prefix(kind)
                .and_then(|rest| rest.strip_prefix(": "))
                .map(|record| (*kind, record))
        }) else {
            continue;
        };
        if !(record.starts_with('{') && record.ends_with('}')) {
            continue;
        }
        let record = json::parse(record.as_bytes())
            .map_err(|error| unidentified(format!("{shown}: {error}")))?;
        let identity = record
            .get("checkpoint_identity")
            .cloned()
            .unwrap_or(Value::Null);
        if !identities.contains(&identity) {
            identities.push(identity);
        }
        if kind == "serving_identity" {
            nonces.push(record.get("startup_nonce").cloned().unwrap_or(Value::Null));
        }
    }
    let checkpoint_identity = match identities.as_slice() {
        [Value::String(identity)] => identity.clone(),
        _ => {
            let mut names: Vec<String> = identities.iter().map(crate::repr::str).collect();
            names.sort();
            return Err(unidentified(format!(
                "{shown} names no single checkpoint identity: {}",
                crate::repr::repr(&Value::from(names))
            )));
        }
    };
    let startup_nonce = match nonces.as_slice() {
        [nonce] if startup_nonce(nonce) => nonce.as_str().unwrap_or_default().to_owned(),
        _ => {
            return Err(unidentified(format!(
                "{shown} names no single startup nonce"
            )));
        }
    };
    Ok(LoggedCheckpoint {
        checkpoint_identity,
        startup_nonce,
        log: ServerLog {
            path: shown,
            sha256: json::sha256_hex(&bytes),
        },
    })
}

/// `str(pathlib.PurePosixPath(path))`, the name a Python tool given `path`
/// as an argument records: repeated separators and `.` components dropped,
/// the leading one included, `..` kept, an empty path `.`, and exactly two
/// leading separators kept as POSIX allows. Nothing is resolved.
fn python_path(path: &Path) -> String {
    let text = path.to_string_lossy();
    let root = if text.starts_with("//") && !text.starts_with("///") {
        "//"
    } else if text.starts_with('/') {
        "/"
    } else {
        ""
    };
    let parts: Vec<&str> = text
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect();
    if root.is_empty() && parts.is_empty() {
        return ".".to_owned();
    }
    format!("{root}{}", parts.join("/"))
}

/// The lines of `text` as Python's `str.splitlines` cuts them.
fn python_lines(text: &str) -> impl Iterator<Item = &str> {
    let mut rest = text;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        let breaks = |c: char| {
            matches!(
                c,
                '\n' | '\r'
                    | '\u{b}'
                    | '\u{c}'
                    | '\u{1c}'
                    | '\u{1d}'
                    | '\u{1e}'
                    | '\u{85}'
                    | '\u{2028}'
                    | '\u{2029}'
            )
        };
        match rest.find(breaks) {
            Some(index) => {
                let line = &rest[..index];
                let after = &rest[index..];
                let width = if after.starts_with("\r\n") {
                    2
                } else {
                    after.chars().next().map_or(1, char::len_utf8)
                };
                rest = &after[width..];
                Some(line)
            }
            None => {
                let line = rest;
                rest = "";
                Some(line)
            }
        }
    })
}

/// Why a server's identity cannot be bound to the pin.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BindError {
    /// The server reports no identity: results cannot be bound to it.
    Unidentified(String),
    /// It serves another checkpoint or model, or changed while scored.
    Mismatch(String),
}

impl std::fmt::Display for BindError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unidentified(message) | Self::Mismatch(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for BindError {}

/// A client of one server.
#[derive(Debug)]
pub struct Client {
    pub url: String,
    endpoint: Endpoint,
    resolved: Option<SocketAddr>,
    timeout: Duration,
    deadline: Option<Instant>,
    retries: u32,
    api_key: Option<String>,
    /// The (checkpoint identity, startup nonce) every response named.
    responders: BTreeSet<(Option<String>, Option<String>)>,
    /// Responses that named an identity header more than once.
    ambiguous: Vec<String>,
}

impl Client {
    /// A client of `url` with a per-request `timeout`. The bearer token is
    /// `OPENAI_API_KEY` when set.
    pub fn new(url: &str, timeout: Duration, retries: u32) -> Result<Self, String> {
        Ok(Self {
            url: url.trim_end_matches('/').into(),
            endpoint: Endpoint::parse(url)?,
            resolved: None,
            timeout,
            deadline: None,
            retries,
            api_key: std::env::var("OPENAI_API_KEY")
                .ok()
                .filter(|key| !key.is_empty()),
            responders: BTreeSet::new(),
            ambiguous: Vec::new(),
        })
    }

    /// Bounds all attempts and backoff by the current check's absolute deadline.
    pub fn set_deadline(&mut self, deadline: Option<Instant>) {
        self.deadline = deadline;
    }

    fn budget(&self) -> Result<Duration, HttpError> {
        match self.deadline {
            Some(deadline) => deadline
                .checked_duration_since(Instant::now())
                .filter(|left| !left.is_zero())
                .ok_or_else(|| HttpError::Transport("the check's time budget is exhausted".into())),
            None => Ok(Duration::MAX),
        }
    }

    pub fn with_api_key(mut self, key: Option<String>) -> Self {
        self.api_key = key;
        self
    }

    /// Records the identity a response named. A response naming either
    /// identity header more than once is ambiguous, whatever the values, and
    /// refuses the results it is part of.
    fn responded(&mut self, headers: &[(String, String)]) {
        let values = |name: &str| -> Vec<String> {
            headers
                .iter()
                .filter(|(key, _)| key == name)
                .map(|(_, value)| value.clone())
                .collect()
        };
        let (checkpoint, nonce) = (values(CHECKPOINT_HEADER), values(NONCE_HEADER));
        if checkpoint.len() > 1 || nonce.len() > 1 {
            self.ambiguous.push(format!(
                "{CHECKPOINT_HEADER} {checkpoint:?}, {NONCE_HEADER} {nonce:?}"
            ));
        }
        self.responders
            .insert((checkpoint.into_iter().next(), nonce.into_iter().next()));
    }

    /// The identities every response so far named.
    pub fn responders(&self) -> &BTreeSet<(Option<String>, Option<String>)> {
        &self.responders
    }

    /// Sends one request, retrying a refused or broken connection and HTTP
    /// 429 and 503 up to `retries` times with backoff.
    pub fn request(
        &mut self,
        method: &str,
        path: &str,
        body: Option<&Value>,
    ) -> Result<Response, HttpError> {
        let mut attempt = 0;
        loop {
            self.budget()?;
            let started = Instant::now();
            let error = match self.send(method, path, body, started) {
                Ok((status, headers, mut reader)) => {
                    self.responded(&headers);
                    let mut bytes = Vec::new();
                    reader
                        .read_to_end(&mut bytes)
                        .map_err(|error| transport(&self.url, path, error))?;
                    self.budget()?;
                    if (200..300).contains(&status) {
                        return Ok(Response {
                            status,
                            headers,
                            body: bytes,
                            seconds: started.elapsed().as_secs_f64(),
                        });
                    }
                    let text =
                        String::from_utf8_lossy(&bytes[..bytes.len().min(1000)]).into_owned();
                    let error = HttpError::Status { status, body: text };
                    if status != 429 && status != 503 {
                        return Err(error);
                    }
                    error
                }
                Err(error) => error,
            };
            if attempt >= self.retries || matches!(error, HttpError::Protocol(_)) {
                return Err(error);
            }
            std::thread::sleep(
                Duration::from_secs_f64((0.5 * 2f64.powi(attempt as i32)).min(4.0))
                    .min(self.budget()?),
            );
            attempt += 1;
        }
    }

    /// GETs or POSTs `path` and reads its JSON.
    pub fn json(&mut self, path: &str, body: Option<&Value>) -> Result<(Value, f64), HttpError> {
        let method = if body.is_some() { "POST" } else { "GET" };
        let response = self.request(method, path, body)?;
        Ok((response.json()?, response.seconds))
    }

    /// POSTs a streaming request and reads its server-sent events up to
    /// `[DONE]`, each with its arrival time. Not retried.
    pub fn stream(&mut self, path: &str, body: &Value) -> Result<(Vec<Event>, f64), HttpError> {
        let started = Instant::now();
        let (status, headers, reader) = self.send("POST", path, Some(body), started)?;
        self.responded(&headers);
        let mut reader = BufReader::new(reader);
        if !(200..300).contains(&status) {
            let mut bytes = Vec::new();
            let _ = reader.read_to_end(&mut bytes);
            return Err(HttpError::Status {
                status,
                body: String::from_utf8_lossy(&bytes[..bytes.len().min(1000)]).into_owned(),
            });
        }
        let mut events = Vec::new();
        let mut line = Vec::new();
        loop {
            line.clear();
            let read = reader
                .read_until(b'\n', &mut line)
                .map_err(|error| transport(&self.url, path, error))?;
            if read == 0 {
                break;
            }
            let Some(data) = line.strip_prefix(b"data: ") else {
                continue;
            };
            let data = data.trim_ascii();
            if data == b"[DONE]" {
                break;
            }
            let value = json::parse(data)
                .map_err(|error| HttpError::Protocol(format!("event: {error}")))?;
            events.push((started.elapsed().as_secs_f64(), value));
        }
        self.budget()?;
        Ok((events, started.elapsed().as_secs_f64()))
    }

    fn send(
        &mut self,
        method: &str,
        path: &str,
        body: Option<&Value>,
        started: Instant,
    ) -> Result<Head, HttpError> {
        let limit = self.timeout.min(self.budget()?);
        let address = match self.resolved {
            Some(address) => address,
            None => {
                let address = match self.endpoint.host.parse::<IpAddr>() {
                    Ok(ip) => SocketAddr::new(ip, self.endpoint.port),
                    Err(_) => crate::resolver::Resolver::shared()
                        .and_then(|resolver| {
                            resolver.resolve(
                                &self.endpoint.host,
                                self.endpoint.port,
                                remaining(limit, started)?,
                            )
                        })
                        .map_err(|error| transport(&self.url, path, error))?,
                };
                self.resolved = Some(address);
                address
            }
        };
        let url = &self.url;
        let stream = TcpStream::connect_timeout(
            &address,
            remaining(limit, started).map_err(|error| transport(url, path, error))?,
        )
        .map_err(|error| transport(url, path, error))?;
        let mut stream = Deadline {
            inner: stream,
            started,
            limit,
        };
        let payload = body.map(|value| serde_json::to_vec(value).expect("JSON values serialize"));
        let mut head = format!(
            "{method} {}{path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nAccept: */*\r\n",
            self.endpoint.prefix,
            self.endpoint.authority()
        );
        if let Some(key) = &self.api_key {
            head.push_str(&format!("Authorization: Bearer {key}\r\n"));
        }
        if let Some(payload) = &payload {
            head.push_str(&format!(
                "Content-Type: application/json\r\nContent-Length: {}\r\n",
                payload.len()
            ));
        }
        head.push_str("\r\n");
        stream
            .write_all(head.as_bytes())
            .and_then(|()| stream.write_all(payload.as_deref().unwrap_or_default()))
            .map_err(|error| transport(url, path, error))?;
        let mut reader = BufReader::new(stream);
        let (status, headers) = read_head(&mut reader).map_err(|error| match error {
            HeadError::Io(error) => transport(url, path, error),
            HeadError::Malformed(message) => HttpError::Protocol(format!("{url}{path}: {message}")),
        })?;
        let header = |name: &str| {
            headers
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.to_ascii_lowercase())
        };
        let body: Box<dyn Read> = if method == "HEAD" || status == 204 || status == 304 {
            Box::new(std::io::empty())
        } else if header("transfer-encoding").is_some_and(|value| value.contains("chunked")) {
            Box::new(Chunked {
                inner: reader,
                remaining: 0,
                done: false,
            })
        } else if let Some(length) = header("content-length") {
            let length = length
                .trim()
                .parse::<u64>()
                .map_err(|_| HttpError::Protocol(format!("{url}{path}: invalid content-length")))?;
            Box::new(reader.take(length))
        } else {
            Box::new(reader)
        };
        Ok((status, headers, body))
    }

    /// The server's models and `/health` (`common.Client.identity`).
    /// `model`, when given, must be served; otherwise the first is taken.
    pub fn identity(&mut self, model: Option<&str>) -> Result<ServerIdentity, HttpError> {
        let (models, _) = self.json("/v1/models", None)?;
        let available: Vec<String> = models["data"]
            .as_array()
            .map(|rows| {
                rows.iter()
                    .filter_map(|row| row["id"].as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let chosen = match model {
            Some(model) if available.iter().any(|name| name == model) => model.to_owned(),
            None if !available.is_empty() => available[0].clone(),
            _ => {
                return Err(HttpError::Unavailable(format!(
                    "model {model:?} is unavailable: {available:?}"
                )));
            }
        };
        // A health body that is missing or not JSON leaves the server
        // unidentified; a broken connection is the environment's failure.
        let health = match self.json("/health", None) {
            Ok((health, _)) => Some(health),
            Err(error @ HttpError::Transport(_)) => return Err(error),
            Err(_) => None,
        };
        Ok(ServerIdentity {
            url: self.url.clone(),
            model: chosen,
            models: available,
            health,
        })
    }

    /// Binds results to the live server (`common.bind_checkpoint`): its
    /// `/health` must name the pinned checkpoint and a startup nonce, it
    /// must serve the pinned model name, and every response so far must
    /// carry exactly that identity in its headers.
    pub fn bind(
        &self,
        server: &ServerIdentity,
        pin: &CheckpointPin,
    ) -> Result<BoundServer, BindError> {
        let serving = server.serving().map_err(BindError::Unidentified)?;
        require_qualified_model_options(serving).map_err(BindError::Mismatch)?;
        let checkpoint = serving["checkpoint_identity"].as_str().unwrap_or_default();
        if checkpoint != pin.checkpoint_identity {
            return Err(BindError::Mismatch(format!(
                "the server loaded checkpoint {checkpoint}, not the pinned {}",
                pin.checkpoint_identity
            )));
        }
        if server.model != pin.served_model_name {
            return Err(BindError::Mismatch(format!(
                "the server serves {}, not {}",
                server.model, pin.served_model_name
            )));
        }
        let text = |key: &str| serving[key].as_str().map(String::from);
        let bound = BoundServer {
            checkpoint_identity: checkpoint.into(),
            startup_nonce: text("startup_nonce").unwrap_or_default(),
            package_identity: text("package_identity"),
            metadata_identity: text("metadata_identity"),
            server_log: None,
        };
        self.check_responders(&bound)?;
        Ok(bound)
    }

    /// Refuses results unless the bound process wrote every response
    /// (`common.confirm_live`): `/health` read again names the bound
    /// checkpoint and nonce, and every response carried both headers once,
    /// naming them.
    pub fn confirm_live(&mut self, model: &str, bound: &BoundServer) -> Result<(), BindError> {
        let live = match self.identity(Some(model)) {
            Ok(live) => live,
            Err(HttpError::Unavailable(error)) => {
                return Err(BindError::Mismatch(format!(
                    "the server at {} changed while scoring: {error}",
                    self.url
                )));
            }
            Err(error) => {
                return Err(BindError::Unidentified(format!(
                    "the server at {} no longer reports its identity: {error}",
                    self.url
                )));
            }
        };
        let serving = live.serving().map_err(BindError::Unidentified)?;
        require_qualified_model_options(serving).map_err(BindError::Mismatch)?;
        if serving["checkpoint_identity"].as_str() != Some(bound.checkpoint_identity.as_str())
            || serving["startup_nonce"].as_str() != Some(bound.startup_nonce.as_str())
        {
            return Err(BindError::Mismatch(format!(
                "the server at {} changed while scoring: it now reports checkpoint {} and startup nonce {}",
                self.url, serving["checkpoint_identity"], serving["startup_nonce"]
            )));
        }
        self.check_responders(bound)
    }

    /// Every response so far named the bound process, once per header:
    /// another or a repeated identity is a mismatch, a missing one leaves
    /// the results unidentified.
    fn check_responders(&self, bound: &BoundServer) -> Result<(), BindError> {
        if !self.ambiguous.is_empty() {
            return Err(BindError::Mismatch(format!(
                "responses from the server at {} named an identity header more than once: {:?}",
                self.url, self.ambiguous
            )));
        }
        let expected = (
            Some(bound.checkpoint_identity.clone()),
            Some(bound.startup_nonce.clone()),
        );
        let others: Vec<_> = self
            .responders
            .iter()
            .filter(|(checkpoint, nonce)| {
                // A header naming anything but the bound process.
                checkpoint.is_some() && *checkpoint != expected.0
                    || nonce.is_some() && *nonce != expected.1
            })
            .collect();
        if !others.is_empty() {
            return Err(BindError::Mismatch(format!(
                "responses from the server at {} named another process (checkpoint, startup nonce): {others:?}",
                self.url
            )));
        }
        if self.responders.iter().any(|pair| *pair != expected) {
            return Err(BindError::Unidentified(format!(
                "responses from the server at {} lack the {NONCE_HEADER} and {CHECKPOINT_HEADER} headers: \
                 the server predates per-response identity headers; rebuild it",
                self.url
            )));
        }
        Ok(())
    }
}

fn transport(url: &str, path: &str, error: std::io::Error) -> HttpError {
    HttpError::Transport(format!("{url}{path}: {error}"))
}

/// A stream that fails once the request's whole time is spent, so a server
/// trickling bytes cannot hold a request past its timeout.
struct Deadline {
    inner: TcpStream,
    started: Instant,
    limit: Duration,
}

fn remaining(limit: Duration, started: Instant) -> std::io::Result<Duration> {
    limit
        .checked_sub(started.elapsed())
        .filter(|left| !left.is_zero())
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::TimedOut, "request timed out"))
}

impl Read for Deadline {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.inner
            .set_read_timeout(Some(remaining(self.limit, self.started)?))?;
        self.inner.read(buffer)
    }
}

impl Write for Deadline {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.inner
            .set_write_timeout(Some(remaining(self.limit, self.started)?))?;
        self.inner.write(buffer)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

enum HeadError {
    Io(std::io::Error),
    Malformed(String),
}

fn read_head(reader: &mut impl BufRead) -> Result<(u16, Vec<(String, String)>), HeadError> {
    let mut total = 0;
    let mut line = String::new();
    let mut next_line = |line: &mut String| -> Result<(), HeadError> {
        line.clear();
        let read = reader.read_line(line).map_err(HeadError::Io)?;
        total += read;
        if read == 0 {
            // A dropped connection is the transport's failure, not the server's answer.
            return Err(HeadError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "the connection closed before the response head",
            )));
        }
        if total > MAX_HEAD {
            return Err(HeadError::Malformed("response head too large".into()));
        }
        Ok(())
    };
    next_line(&mut line)?;
    let status = line
        .strip_prefix("HTTP/1.")
        .and_then(|rest| rest.get(2..5))
        .and_then(|code| code.parse::<u16>().ok())
        .ok_or_else(|| {
            HeadError::Malformed(format!("not an HTTP response: {:?}", line.trim_end()))
        })?;
    let mut headers = Vec::new();
    loop {
        next_line(&mut line)?;
        let text = line.trim_end_matches(['\r', '\n']);
        if text.is_empty() {
            return Ok((status, headers));
        }
        let (name, value) = text
            .split_once(':')
            .ok_or_else(|| HeadError::Malformed(format!("malformed header {text:?}")))?;
        headers.push((name.trim().to_ascii_lowercase(), value.trim().to_owned()));
    }
}

/// A chunked transfer-encoded body.
struct Chunked<R> {
    inner: R,
    remaining: u64,
    done: bool,
}

impl<R: BufRead> Read for Chunked<R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let invalid = |message: &str| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, message.to_owned())
        };
        if self.done || buffer.is_empty() {
            return Ok(0);
        }
        if self.remaining == 0 {
            let mut line = String::new();
            self.inner.read_line(&mut line)?;
            let size = line.trim_end().split(';').next().unwrap_or_default();
            self.remaining =
                u64::from_str_radix(size.trim(), 16).map_err(|_| invalid("invalid chunk size"))?;
            if self.remaining == 0 {
                // Trailers end at an empty line.
                loop {
                    line.clear();
                    if self.inner.read_line(&mut line)? == 0 || line.trim_end().is_empty() {
                        break;
                    }
                }
                self.done = true;
                return Ok(0);
            }
        }
        let wanted = buffer
            .len()
            .min(self.remaining.min(usize::MAX as u64) as usize);
        let read = self.inner.read(&mut buffer[..wanted])?;
        if read == 0 {
            return Err(invalid("the connection closed inside a chunk"));
        }
        self.remaining -= read as u64;
        if self.remaining == 0 {
            let mut end = [0u8; 2];
            self.inner.read_exact(&mut end)?;
            if &end != b"\r\n" {
                return Err(invalid("a chunk does not end with CRLF"));
            }
        }
        Ok(read)
    }
}

#[cfg(test)]
mod resolution_tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn successful_hostname_resolution_is_cached_for_later_requests() {
        let listener = TcpListener::bind(("localhost", 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://localhost:{}", listener.local_addr().unwrap().port());
        let server = std::thread::spawn(move || {
            let until = Instant::now() + Duration::from_secs(3);
            let mut requests = 0;
            while requests < 2 && Instant::now() < until {
                match listener.accept() {
                    Ok((stream, _)) => {
                        stream
                            .set_read_timeout(Some(Duration::from_secs(1)))
                            .unwrap();
                        let mut reader = BufReader::new(stream);
                        let mut line = String::new();
                        loop {
                            line.clear();
                            reader.read_line(&mut line).unwrap();
                            if line == "\r\n" {
                                break;
                            }
                        }
                        reader
                            .get_mut()
                            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
                            .unwrap();
                        requests += 1;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("{error}"),
                }
            }
            requests
        });
        let mut client = Client::new(&url, Duration::from_secs(1), 0).unwrap();
        let first = client.request("GET", "/first", None);
        // A second resolution would fail, while the cached endpoint remains live.
        client.endpoint.host = "invalid.invalid".into();
        let second = client.request("GET", "/second", None);
        assert_eq!(server.join().unwrap(), 2);
        assert!(first.is_ok() && second.is_ok(), "{first:?}, {second:?}");
    }
}
