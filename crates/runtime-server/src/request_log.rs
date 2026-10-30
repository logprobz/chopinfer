//! One JSON line per failed or cancelled request, and the periodic summary.
//!
//! Every request gets a [`RequestRecord`] when it reaches the router. Whoever
//! learns its terminal outcome writes its line: the outer middleware for an
//! HTTP error response, a stream task for a failure after the headers, the
//! Responses adapter for its requests, and a sweep for requests cut off by a
//! worker failure or a forced stop. Each record writes at most one line.
//!
//! Lines never carry prompt text, generated text or token IDs, and every
//! field a client controls is cut to a byte limit when it is stored. A line
//! is formatted on the caller's thread and queued; one writer thread drains
//! the queue, so a handler never waits on stderr. Each level has its own
//! rate limit so a flood of client refusals cannot crowd out server faults,
//! and whatever a limit or a full queue drops is counted and reported.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;
use runtime_log::Level;
use serde_json::{Map, Value, json};

use crate::error::bounded_log_text;

pub(crate) const FAILURE_SCHEMA: &str = "chopinfer.request-failure.v1";
pub(crate) const CANCEL_SCHEMA: &str = "chopinfer.request-cancel.v1";
pub(crate) const DROPPED_SCHEMA: &str = "chopinfer.request-failure-dropped.v1";
pub(crate) const SUMMARY_SCHEMA: &str = "chopinfer.summary.v1";
const QUEUE_LINES: usize = 1024;
const FLUSH_TIMEOUT: Duration = Duration::from_millis(500);
/// Byte limits of the fields a client controls. With JSON escaping at most
/// doubling them, a line stays under 4 KiB.
const METHOD_BYTES: usize = 16;
const PATH_BYTES: usize = 256;
const NAME_BYTES: usize = 128;
const SERVED_NAME_BYTES: usize = 64;
const PARAM_BYTES: usize = 64;
const CODE_BYTES: usize = 64;
const SERVED_NAMES: usize = 8;

/// Where request-failure, cancellation and summary lines go: stderr unless
/// a test or embedder installs another sink.
#[derive(Clone)]
pub struct RequestLogSinkV1(Arc<dyn Fn(&str) + Send + Sync>);

impl RequestLogSinkV1 {
    pub fn new(write: impl Fn(&str) + Send + Sync + 'static) -> Self {
        Self(Arc::new(write))
    }
}

impl std::fmt::Debug for RequestLogSinkV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("RequestLogSinkV1")
    }
}

impl PartialEq for RequestLogSinkV1 {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for RequestLogSinkV1 {}

/// How far a request got before it ended.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum Stage {
    #[default]
    Transport,
    Intake,
    Preparation,
    Admission,
    Generation,
}

impl Stage {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Transport => "transport",
            Self::Intake => "intake",
            Self::Preparation => "preparation",
            Self::Admission => "admission",
            Self::Generation => "generation",
        }
    }
}

/// Where a JSON body stopped parsing, without the text around it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct JsonErrorPosition {
    pub(crate) category: &'static str,
    pub(crate) line: usize,
    pub(crate) column: usize,
}

impl JsonErrorPosition {
    pub(crate) fn of(error: &serde_json::Error) -> Self {
        Self {
            category: match error.classify() {
                serde_json::error::Category::Io => "io",
                serde_json::error::Category::Syntax => "syntax",
                serde_json::error::Category::Data => "data",
                serde_json::error::Category::Eof => "eof",
            },
            line: error.line(),
            column: error.column(),
        }
    }
}

/// Marks a readiness probe's answer from `/health` or `/ping`. Its 503
/// while the server starts or drains is the answer the probe asked for, not
/// a failure; any other refusal on those paths is one.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReadinessAnswer;

/// What a failed response says about itself, without request content. An
/// [`crate::error::ApiError`] attaches one to its response.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FailureDetail {
    pub(crate) status: u16,
    pub(crate) error_type: &'static str,
    pub(crate) code: Option<String>,
    pub(crate) param: Option<String>,
    /// A fixed message, never text a request supplied.
    pub(crate) message: &'static str,
    pub(crate) json_error: Option<JsonErrorPosition>,
    pub(crate) requested_model: Option<String>,
    /// The refusal ended the request because the client left.
    pub(crate) disconnected: bool,
    /// A stop or a capacity limit, not a fault: written at warn whatever
    /// the status.
    pub(crate) routine: bool,
}

impl FailureDetail {
    /// A response that carries no detail: a route the router refused, or a
    /// policy layer's answer.
    pub(crate) fn from_status(status: StatusCode) -> Self {
        Self {
            status: status.as_u16(),
            error_type: "HttpError",
            code: None,
            param: None,
            message: status.canonical_reason().unwrap_or("HTTP error"),
            json_error: None,
            requested_model: None,
            disconnected: false,
            routine: false,
        }
    }

    /// A failure after admission that the backend reported, which the
    /// client received in its terminal event.
    pub(crate) fn backend(code: Option<&str>) -> Self {
        let shutdown = code == Some("shutdown_timeout");
        Self {
            status: if shutdown || code == Some("worker_unavailable") {
                503
            } else {
                500
            },
            error_type: "InternalServerError",
            code: code.map(|code| bounded_log_text(code, CODE_BYTES)),
            param: None,
            message: if shutdown {
                "server drain deadline expired"
            } else {
                "generation failed"
            },
            json_error: None,
            requested_model: None,
            disconnected: false,
            routine: shutdown,
        }
    }

    fn level(&self) -> Level {
        if self.routine || self.status < 500 {
            Level::Warn
        } else {
            Level::Error
        }
    }
}

/// Counters a summary line reports as deltas.
#[derive(Default)]
struct Counts {
    requests: AtomicU64,
    failures: [AtomicU64; 3],
}

/// The per-server request log: its writer, its open records and its counts.
pub(crate) struct RequestLog {
    sender: SyncSender<Message>,
    queue_dropped: Arc<[AtomicU64; 3]>,
    open: Mutex<BTreeMap<u64, Weak<RequestRecord>>>,
    next: AtomicU64,
    counts: Counts,
    served: Vec<String>,
    /// Set when the drain deadline expires: a request the server drops from
    /// then on ended by shutdown, not by its client.
    shutting_down: AtomicBool,
}

enum Message {
    Line {
        level: Level,
        text: String,
    },
    /// A summary or dropped-count line, written whatever the rate limits.
    Plain(String),
    Flush(SyncSender<()>),
}

impl RequestLog {
    /// A log whose lines go to `sink`, or stderr, naming up to eight of
    /// `served` in unknown-model lines.
    pub(crate) fn new(sink: Option<RequestLogSinkV1>, served: Vec<String>) -> Arc<Self> {
        let sink = sink.unwrap_or_else(|| {
            RequestLogSinkV1::new(|line| {
                use std::io::Write;
                let _ = writeln!(std::io::stderr().lock(), "{line}");
            })
        });
        let (sender, receiver) = sync_channel(QUEUE_LINES);
        let queue_dropped = Arc::new([AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)]);
        let writer_dropped = Arc::clone(&queue_dropped);
        // A server without a writer still serves; its lines count as dropped.
        let _ = std::thread::Builder::new()
            .name("request-log".to_owned())
            .spawn(move || write_loop(&receiver, &sink, &writer_dropped));
        Arc::new(Self {
            sender,
            queue_dropped,
            open: Mutex::new(BTreeMap::new()),
            next: AtomicU64::new(1),
            counts: Counts::default(),
            served: served
                .iter()
                .take(SERVED_NAMES)
                .map(|name| bounded_log_text(name, SERVED_NAME_BYTES))
                .collect(),
            shutting_down: AtomicBool::new(false),
        })
    }

    /// Opens the record of one request as it reaches the router.
    pub(crate) fn open(
        self: &Arc<Self>,
        method: &str,
        path: &str,
        peer: Option<SocketAddr>,
        client_request_id: Option<&str>,
    ) -> Arc<RequestRecord> {
        self.open_record(Some((method, path)), peer, client_request_id)
    }

    /// Writes the failure line of a request that never reached the router:
    /// a head that did not parse or arrive, or a connection refused at the
    /// connection limit.
    pub(crate) fn transport_failure(
        self: &Arc<Self>,
        peer: Option<SocketAddr>,
        detail: FailureDetail,
    ) {
        self.open_record(None, peer, None).fail(detail);
    }

    fn open_record(
        self: &Arc<Self>,
        target: Option<(&str, &str)>,
        peer: Option<SocketAddr>,
        client_request_id: Option<&str>,
    ) -> Arc<RequestRecord> {
        self.counts.requests.fetch_add(1, Ordering::Relaxed);
        let key = self.next.fetch_add(1, Ordering::Relaxed);
        let record = Arc::new(RequestRecord {
            key,
            log: Arc::clone(self),
            started: Instant::now(),
            method: target.map(|(method, _)| bounded_log_text(method, METHOD_BYTES)),
            path: target.map(|(_, path)| bounded_log_text(path, PATH_BYTES)),
            peer,
            emitted: AtomicBool::new(false),
            deferred: AtomicBool::new(false),
            adapter: AtomicBool::new(false),
            progress: Mutex::new(Progress {
                client_request_id: client_request_id.map(|id| bounded_log_text(id, NAME_BYTES)),
                ..Progress::default()
            }),
        });
        self.open
            .lock()
            .expect("open request records")
            .insert(key, Arc::downgrade(&record));
        record
    }

    /// Marks the drain deadline expired.
    pub(crate) fn begin_shutdown(&self) {
        self.shutting_down.store(true, Ordering::Release);
    }

    fn shutting_down(&self) -> bool {
        self.shutting_down.load(Ordering::Acquire)
    }

    /// Writes a line for every open request that has none yet: a failure
    /// line with `detail`. A worker failure and a forced stop end requests
    /// that no handler will report.
    pub(crate) fn sweep(&self, detail: &FailureDetail) {
        let open = self
            .open
            .lock()
            .expect("open request records")
            .values()
            .filter_map(Weak::upgrade)
            .collect::<Vec<_>>();
        for record in open {
            record.fail(detail.clone());
        }
    }

    /// Waits up to half a second for queued lines to be written.
    pub(crate) fn flush(&self) -> bool {
        let started = Instant::now();
        let (done, acknowledged) = sync_channel(1);
        let mut message = Message::Flush(done);
        loop {
            match self.sender.try_send(message) {
                Ok(()) => break,
                Err(TrySendError::Full(returned)) => message = returned,
                Err(TrySendError::Disconnected(_)) => return false,
            }
            let remaining = FLUSH_TIMEOUT.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return false;
            }
            std::thread::sleep(remaining.min(Duration::from_millis(1)));
        }
        acknowledged
            .recv_timeout(FLUSH_TIMEOUT.saturating_sub(started.elapsed()))
            .is_ok()
    }

    fn write(&self, level: Level, line: Value) {
        self.counts.failures[level_index(level)].fetch_add(1, Ordering::Relaxed);
        if !runtime_log::enabled(level) {
            return;
        }
        let message = Message::Line {
            level,
            text: line.to_string(),
        };
        if self.sender.try_send(message).is_err() {
            self.queue_dropped[level_index(level)].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Writes one summary line at `info`, without rate limits.
    pub(crate) fn write_summary(&self, line: Value) {
        if runtime_log::enabled(Level::Info) {
            let _ = self.sender.try_send(Message::Plain(line.to_string()));
        }
    }

    /// HTTP requests and failure lines by level since the server started.
    pub(crate) fn totals(&self) -> RequestTotals {
        RequestTotals {
            requests: self.counts.requests.load(Ordering::Relaxed),
            failures: std::array::from_fn(|index| {
                self.counts.failures[index].load(Ordering::Relaxed)
            }),
        }
    }
}

/// See [`RequestLog::totals`]: requests, then error, warn and info lines.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RequestTotals {
    pub(crate) requests: u64,
    pub(crate) failures: [u64; 3],
}

const fn level_index(level: Level) -> usize {
    match level {
        Level::Error => 0,
        Level::Warn => 1,
        Level::Info | Level::Debug => 2,
    }
}

/// One request's progress, from the router to its terminal outcome.
pub(crate) struct RequestRecord {
    key: u64,
    log: Arc<RequestLog>,
    started: Instant,
    /// The request line, unless the request failed before it parsed.
    method: Option<String>,
    path: Option<String>,
    peer: Option<SocketAddr>,
    emitted: AtomicBool,
    /// The outcome is reported after the HTTP response: by a stream task,
    /// the Responses adapter or a background response.
    deferred: AtomicBool,
    /// The Responses adapter reports the outcome; the chat stream inside it
    /// only stores a failure.
    adapter: AtomicBool,
    progress: Mutex<Progress>,
}

#[derive(Default)]
struct Progress {
    request_id: Option<String>,
    client_request_id: Option<String>,
    stage: Stage,
    model: Option<String>,
    prompt_tokens: Option<u64>,
    generated_tokens: Option<u64>,
    /// A failure an inner stream stored for the outer adapter to report.
    failure: Option<FailureDetail>,
}

impl Drop for RequestRecord {
    fn drop(&mut self) {
        if let Ok(mut open) = self.log.open.lock() {
            open.remove(&self.key);
        }
    }
}

impl RequestRecord {
    fn progress(&self) -> std::sync::MutexGuard<'_, Progress> {
        self.progress
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(crate) fn set_stage(&self, stage: Stage) {
        self.progress().stage = stage;
    }

    pub(crate) fn set_request_id(&self, id: &str) {
        self.progress().request_id = Some(bounded_log_text(id, NAME_BYTES));
    }

    pub(crate) fn set_model(&self, model: &str) {
        self.progress().model = Some(bounded_log_text(model, NAME_BYTES));
    }

    pub(crate) fn set_prompt_tokens(&self, tokens: u64) {
        self.progress().prompt_tokens = Some(tokens);
    }

    /// Counts `tokens` more generated for the request, across its choices.
    pub(crate) fn add_generated_tokens(&self, tokens: u64) {
        let mut progress = self.progress();
        progress.generated_tokens = Some(
            progress
                .generated_tokens
                .unwrap_or(0)
                .saturating_add(tokens),
        );
    }

    /// Whether the drain deadline had expired: a request that ends from
    /// then on ended by shutdown, not by its client.
    pub(crate) fn ended_by_shutdown(&self) -> bool {
        self.log.shutting_down()
    }

    /// Stores a stream's failure now when the Responses adapter owns the
    /// outcome, so the adapter finds it when it reads the error frame.
    pub(crate) fn store_for_adapter(&self, detail: &FailureDetail) {
        if self.adapter.load(Ordering::Acquire) {
            self.store_failure(detail.clone());
        }
    }

    /// Stores a failure for the outer adapter instead of writing it.
    pub(crate) fn store_failure(&self, detail: FailureDetail) {
        self.progress().failure.get_or_insert(detail);
    }

    pub(crate) fn stored_failure(&self) -> Option<FailureDetail> {
        self.progress().failure.clone()
    }

    /// The outcome is reported after the HTTP response.
    pub(crate) fn defer(&self) {
        self.deferred.store(true, Ordering::Release);
    }

    pub(crate) fn is_deferred(&self) -> bool {
        self.deferred.load(Ordering::Acquire)
    }

    /// Hands the outcome to the Responses adapter.
    pub(crate) fn hand_to_adapter(&self) {
        self.adapter.store(true, Ordering::Release);
        self.defer();
    }

    /// Reports a stream's failure: written, or stored for the adapter.
    pub(crate) fn report(&self, detail: FailureDetail) {
        if self.adapter.load(Ordering::Acquire) {
            self.store_failure(detail);
        } else {
            self.fail(detail);
        }
    }

    /// Reports a stream whose reader left: the adapter, when it owns the
    /// outcome, decides what that means.
    pub(crate) fn stream_abandoned(&self) {
        if !self.adapter.load(Ordering::Acquire) {
            self.abandon();
        }
    }

    /// The request succeeded: no line, and no sweep writes one later.
    pub(crate) fn succeed(&self) {
        self.emitted.store(true, Ordering::Release);
    }

    fn claim(&self) -> bool {
        !self.emitted.swap(true, Ordering::AcqRel)
    }

    /// Writes the request's failure line, unless it already has a line.
    pub(crate) fn fail(&self, detail: FailureDetail) {
        if !self.claim() {
            return;
        }
        let level = detail.level();
        let mut line = self.common(FAILURE_SCHEMA, level);
        line.insert("status".to_owned(), json!(detail.status));
        line.insert("error_type".to_owned(), json!(detail.error_type));
        line.insert("code".to_owned(), json!(detail.code));
        line.insert(
            "param".to_owned(),
            json!(
                detail
                    .param
                    .as_deref()
                    .map(|param| bounded_log_text(param, PARAM_BYTES))
            ),
        );
        line.insert("message".to_owned(), json!(detail.message));
        if let Some(position) = detail.json_error {
            line.insert(
                "json_error".to_owned(),
                json!({"category": position.category, "line": position.line, "column": position.column}),
            );
        }
        if let Some(model) = detail.requested_model {
            line.insert("model".to_owned(), json!(model));
            line.insert("served".to_owned(), json!(self.log.served));
        }
        self.finish(line, level);
    }

    /// Writes the request's cancellation line at `info`, unless it already
    /// has a line. A cancellation is not a failure.
    pub(crate) fn cancel(&self, reason: &'static str) {
        if !self.claim() {
            return;
        }
        let mut line = self.common(CANCEL_SCHEMA, Level::Info);
        line.insert("reason".to_owned(), json!(reason));
        self.finish(line, Level::Info);
    }

    /// Writes the line of a request that ended without a terminal answer:
    /// a shutdown failure when the server was stopping, otherwise the
    /// client's cancellation.
    pub(crate) fn abandon(&self) {
        if self.ended_by_shutdown() {
            self.fail(shutdown_detail());
        } else {
            self.cancel("client_disconnected");
        }
    }

    /// The fields before the outcome: what the request was.
    fn common(&self, schema: &str, level: Level) -> Map<String, Value> {
        let progress = self.progress();
        let mut line = Map::new();
        line.insert("schema".to_owned(), json!(schema));
        line.insert("time".to_owned(), json!(runtime_log::utc_timestamp()));
        line.insert("level".to_owned(), json!(level.as_str()));
        line.insert("request_id".to_owned(), json!(progress.request_id));
        line.insert(
            "client_request_id".to_owned(),
            json!(progress.client_request_id),
        );
        line.insert("method".to_owned(), json!(self.method));
        line.insert("path".to_owned(), json!(self.path));
        line.insert("model".to_owned(), json!(progress.model));
        line
    }

    /// Adds the fields after the outcome, how far the request got, and
    /// queues the line.
    fn finish(&self, mut line: Map<String, Value>, level: Level) {
        {
            let progress = self.progress();
            line.insert("stage".to_owned(), json!(progress.stage.as_str()));
            line.insert("prompt_tokens".to_owned(), json!(progress.prompt_tokens));
            line.insert(
                "generated_tokens".to_owned(),
                json!(progress.generated_tokens),
            );
        }
        line.insert(
            "elapsed_seconds".to_owned(),
            json!((self.started.elapsed().as_secs_f64() * 1e6).round() / 1e6),
        );
        line.insert(
            "peer".to_owned(),
            json!(self.peer.map(|peer| peer.to_string())),
        );
        self.log.write(level, Value::Object(line));
    }
}

/// The failure of a request the transport refused before the router saw it.
pub(crate) fn transport_detail(status: StatusCode, message: &'static str) -> FailureDetail {
    FailureDetail {
        error_type: "transport_error",
        message,
        ..FailureDetail::from_status(status)
    }
}

/// The connection the transport closed at its connection limit, a capacity
/// refusal like a 429.
pub(crate) fn connection_limit_detail() -> FailureDetail {
    FailureDetail {
        routine: true,
        ..transport_detail(
            StatusCode::SERVICE_UNAVAILABLE,
            "connection limit reached; the connection was closed",
        )
    }
}

/// The failure of a request the drain deadline ended.
pub(crate) fn shutdown_detail() -> FailureDetail {
    FailureDetail::backend(Some("shutdown_timeout"))
}

/// The failure of a stream whose reader stopped taking frames for longer
/// than the write timeout.
pub(crate) fn client_write_timeout_detail() -> FailureDetail {
    FailureDetail {
        status: 408,
        error_type: "client_write_timeout",
        message: "the client stopped reading the stream",
        ..FailureDetail::from_status(StatusCode::REQUEST_TIMEOUT)
    }
}

/// The failure of every open request when the generation worker stops.
pub(crate) fn worker_terminated_detail() -> FailureDetail {
    FailureDetail {
        message: "generation worker terminated",
        ..FailureDetail::backend(Some("worker_unavailable"))
    }
}

/// A request's record, when the router opened one. Every method does
/// nothing without a record, so a handler need not check.
#[derive(Clone, Default)]
pub(crate) struct RecordHandle(Option<Arc<RequestRecord>>);

impl RecordHandle {
    pub(crate) fn of<B>(request: &axum::http::Request<B>) -> Self {
        Self(request.extensions().get::<Arc<RequestRecord>>().cloned())
    }

    pub(crate) fn record(&self) -> Option<&Arc<RequestRecord>> {
        self.0.as_ref()
    }

    pub(crate) fn set_stage(&self, stage: Stage) {
        if let Some(record) = &self.0 {
            record.set_stage(stage);
        }
    }

    pub(crate) fn set_request_id(&self, id: &str) {
        if let Some(record) = &self.0 {
            record.set_request_id(id);
        }
    }

    pub(crate) fn set_model(&self, model: &str) {
        if let Some(record) = &self.0 {
            record.set_model(model);
        }
    }

    pub(crate) fn set_prompt_tokens(&self, tokens: u64) {
        if let Some(record) = &self.0 {
            record.set_prompt_tokens(tokens);
        }
    }

    pub(crate) fn add_generated_tokens(&self, tokens: u64) {
        if tokens > 0
            && let Some(record) = &self.0
        {
            record.add_generated_tokens(tokens);
        }
    }

    pub(crate) fn defer(&self) {
        if let Some(record) = &self.0 {
            record.defer();
        }
    }

    pub(crate) fn hand_to_adapter(&self) {
        if let Some(record) = &self.0 {
            record.hand_to_adapter();
        }
    }
}

/// Writes one line for a request that ended before its handler returned,
/// unless it is disarmed first.
struct AbandonGuard(Option<Arc<RequestRecord>>);

impl Drop for AbandonGuard {
    fn drop(&mut self) {
        if let Some(record) = self.0.take() {
            record.abandon();
        }
    }
}

/// The outermost layer: opens each request's record and writes the line of
/// an HTTP error response, unless it is a [`ReadinessAnswer`].
pub(crate) async fn middleware(
    State(log): State<Arc<RequestLog>>,
    mut request: Request,
    next: Next,
) -> Response {
    // The ID the client named its request by, in either header the server
    // reads one from.
    let client_request_id = ["x-client-request-id", "x-request-id"]
        .iter()
        .find_map(|name| {
            request
                .headers()
                .get(*name)
                .and_then(|value| value.to_str().ok())
                .filter(|value| !value.is_empty())
        })
        .map(str::to_owned);
    let record = log.open(
        request.method().as_str(),
        request.uri().path(),
        request.extensions().get::<SocketAddr>().copied(),
        client_request_id.as_deref(),
    );
    request.extensions_mut().insert(Arc::clone(&record));
    let mut abandoned = AbandonGuard(Some(Arc::clone(&record)));
    let response = next.run(request).await;
    abandoned.0 = None;
    let status = response.status();
    let readiness = response.extensions().get::<ReadinessAnswer>().is_some();
    if (status.is_client_error() || status.is_server_error()) && !readiness {
        let detail = response
            .extensions()
            .get::<FailureDetail>()
            .cloned()
            .unwrap_or_else(|| FailureDetail::from_status(status));
        if record.progress().request_id.is_none()
            && let Some(id) = response
                .headers()
                .get(crate::error::X_REQUEST_ID)
                .and_then(|value| value.to_str().ok())
        {
            record.set_request_id(id);
        }
        if detail.disconnected {
            // After the drain deadline the stop, not the client, ended it.
            record.abandon();
        } else {
            record.fail(detail);
        }
    } else if !record.is_deferred() {
        record.succeed();
    }
    response
}

/// The writer: applies each level's rate limit and reports drops once a
/// second and at every flush.
fn write_loop(
    receiver: &Receiver<Message>,
    sink: &RequestLogSinkV1,
    queue_dropped: &[AtomicU64; 3],
) {
    let mut buckets = [
        Bucket::new(200.0, 50.0),
        Bucket::new(100.0, 20.0),
        Bucket::new(50.0, 10.0),
    ];
    let mut limited = [0_u64; 3];
    let mut last_report = Instant::now();
    let report = |limited: &mut [u64; 3]| {
        let dropped: [u64; 3] = std::array::from_fn(|index| {
            limited[index] + queue_dropped[index].swap(0, Ordering::Relaxed)
        });
        *limited = [0; 3];
        if dropped.iter().any(|count| *count > 0) {
            (sink.0)(
                &json!({
                    "schema": DROPPED_SCHEMA,
                    "time": runtime_log::utc_timestamp(),
                    "dropped": {"error": dropped[0], "warn": dropped[1], "info": dropped[2]},
                })
                .to_string(),
            );
        }
    };
    loop {
        match receiver.recv_timeout(Duration::from_secs(1)) {
            Ok(Message::Line { level, text }) => {
                let index = level_index(level);
                if buckets[index].take() {
                    (sink.0)(&text);
                } else {
                    limited[index] += 1;
                }
            }
            Ok(Message::Plain(text)) => (sink.0)(&text),
            Ok(Message::Flush(done)) => {
                report(&mut limited);
                last_report = Instant::now();
                let _ = done.try_send(());
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                report(&mut limited);
                return;
            }
        }
        if last_report.elapsed() >= Duration::from_secs(1) {
            report(&mut limited);
            last_report = Instant::now();
        }
    }
}

/// A token bucket: `burst` lines at once, refilled at `rate` per second.
struct Bucket {
    tokens: f64,
    burst: f64,
    rate: f64,
    refilled: Instant,
}

impl Bucket {
    fn new(burst: f64, rate: f64) -> Self {
        Self {
            tokens: burst,
            burst,
            rate,
            refilled: Instant::now(),
        }
    }

    fn take(&mut self) -> bool {
        let now = Instant::now();
        self.tokens = (self.tokens + now.duration_since(self.refilled).as_secs_f64() * self.rate)
            .min(self.burst);
        self.refilled = now;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// One latency histogram's cumulative counts, in microseconds, as a
/// metrics provider reports it for a summary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LatencyHistogramV1 {
    /// Upper bounds of the finite buckets.
    pub bounds_micros: Vec<u64>,
    /// Observations at or below each bound.
    pub cumulative: Vec<u64>,
    /// Every observation, including those above the last bound.
    pub count: u64,
}

/// What a periodic summary line reports from the engine: totals since start
/// for the counters, which the line reports as deltas, and current values
/// for the gauges.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EngineSummaryV1 {
    pub requests_submitted: u64,
    pub requests_completed: u64,
    pub requests_failed: u64,
    pub requests_cancelled: u64,
    pub prompt_tokens: u64,
    pub generation_tokens: u64,
    pub draft_tokens: u64,
    pub accepted_tokens: u64,
    pub preemptions: u64,
    pub ttft: LatencyHistogramV1,
    pub itl: LatencyHistogramV1,
    /// Gauges.
    pub running: u64,
    pub waiting: u64,
    pub kv_tokens_used: u64,
    pub kv_tokens_capacity: u64,
    /// `starting`, `ready`, `failed` or `stopped`.
    pub worker_state: &'static str,
}

/// The totals one summary line was computed from, so the next can report
/// what changed.
#[derive(Clone, Debug, Default)]
pub(crate) struct SummaryBase {
    requests: RequestTotals,
    refusals: Vec<u64>,
    engine: Option<EngineSummaryV1>,
}

/// One summary line: deltas since `base` for counters, current values for
/// gauges. Returns the line and the base for the next one.
pub(crate) fn summary_line(
    interval: Duration,
    requests: RequestTotals,
    refusals: &[(&'static str, u64)],
    engine: Option<EngineSummaryV1>,
    base: &SummaryBase,
) -> (Value, SummaryBase) {
    let refusal_deltas = refusals
        .iter()
        .enumerate()
        .map(|(index, (reason, total))| {
            let before = base.refusals.get(index).copied().unwrap_or(0);
            ((*reason).to_owned(), json!(total.saturating_sub(before)))
        })
        .collect::<Map<_, _>>();
    let failures =
        |index: usize| requests.failures[index].saturating_sub(base.requests.failures[index]);
    let mut line = Map::new();
    line.insert("schema".to_owned(), json!(SUMMARY_SCHEMA));
    line.insert("time".to_owned(), json!(runtime_log::utc_timestamp()));
    line.insert(
        "interval_seconds".to_owned(),
        json!((interval.as_secs_f64() * 1e3).round() / 1e3),
    );
    line.insert(
        "http_requests".to_owned(),
        json!(requests.requests.saturating_sub(base.requests.requests)),
    );
    line.insert(
        "failures".to_owned(),
        json!({"error": failures(0), "warn": failures(1), "cancelled": failures(2)}),
    );
    line.insert("refusals".to_owned(), Value::Object(refusal_deltas));
    if let Some(engine) = &engine {
        let before = base.engine.clone().unwrap_or_default();
        let delta = |now: u64, then: u64| now.saturating_sub(then);
        let drafted = delta(engine.draft_tokens, before.draft_tokens);
        let accepted = delta(engine.accepted_tokens, before.accepted_tokens);
        line.insert(
            "requests".to_owned(),
            json!({
                "submitted": delta(engine.requests_submitted, before.requests_submitted),
                "completed": delta(engine.requests_completed, before.requests_completed),
                "failed": delta(engine.requests_failed, before.requests_failed),
                "cancelled": delta(engine.requests_cancelled, before.requests_cancelled),
            }),
        );
        line.insert(
            "tokens".to_owned(),
            json!({
                "prompt": delta(engine.prompt_tokens, before.prompt_tokens),
                "generated": delta(engine.generation_tokens, before.generation_tokens),
                "drafted": drafted,
                "accepted": accepted,
            }),
        );
        line.insert(
            "acceptance".to_owned(),
            if drafted == 0 {
                Value::Null
            } else {
                json!(((accepted as f64 / drafted as f64) * 1e4).round() / 1e4)
            },
        );
        line.insert(
            "preemptions".to_owned(),
            json!(delta(engine.preemptions, before.preemptions)),
        );
        line.insert(
            "ttft_seconds".to_owned(),
            interval_latency(&engine.ttft, &before.ttft),
        );
        line.insert(
            "itl_seconds".to_owned(),
            interval_latency(&engine.itl, &before.itl),
        );
        line.insert("running".to_owned(), json!(engine.running));
        line.insert("waiting".to_owned(), json!(engine.waiting));
        line.insert(
            "kv".to_owned(),
            json!({
                "used_tokens": engine.kv_tokens_used,
                "capacity_tokens": engine.kv_tokens_capacity,
                "usage": if engine.kv_tokens_capacity == 0 {
                    Value::Null
                } else {
                    json!(((engine.kv_tokens_used as f64 / engine.kv_tokens_capacity as f64) * 1e4).round() / 1e4)
                },
            }),
        );
        line.insert("worker_state".to_owned(), json!(engine.worker_state));
    }
    (
        Value::Object(line),
        SummaryBase {
            requests,
            refusals: refusals.iter().map(|(_, total)| *total).collect(),
            engine,
        },
    )
}

/// The interval's observation count and its p50 and p95 in seconds, from
/// the change in cumulative bucket counts. A quantile falls in the first
/// bucket whose count reaches its rank and is interpolated linearly within
/// it, as Prometheus `histogram_quantile` does; one past the last bound
/// reports that bound. Both are null when the interval observed nothing.
fn interval_latency(now: &LatencyHistogramV1, before: &LatencyHistogramV1) -> Value {
    let count = now.count.saturating_sub(before.count);
    let comparable = before.bounds_micros.is_empty() || before.bounds_micros == now.bounds_micros;
    let cumulative = now
        .cumulative
        .iter()
        .enumerate()
        .map(|(index, total)| {
            total.saturating_sub(if comparable {
                before.cumulative.get(index).copied().unwrap_or(0)
            } else {
                0
            })
        })
        .collect::<Vec<_>>();
    let quantile = |q: f64| -> Value {
        if count == 0 {
            return Value::Null;
        }
        let rank = q * count as f64;
        let mut lower_bound = 0.0;
        let mut lower_count = 0.0;
        for (bound, total) in now.bounds_micros.iter().zip(&cumulative) {
            let (bound, total) = (*bound as f64, *total as f64);
            if total >= rank {
                let within = if total > lower_count {
                    (rank - lower_count) / (total - lower_count)
                } else {
                    1.0
                };
                let micros = lower_bound + (bound - lower_bound) * within;
                return json!(micros.round() / 1e6);
            }
            lower_bound = bound;
            lower_count = total;
        }
        json!(lower_bound / 1e6)
    };
    json!({"count": count, "p50": quantile(0.5), "p95": quantile(0.95)})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn histogram(cumulative: &[u64], count: u64) -> LatencyHistogramV1 {
        LatencyHistogramV1 {
            bounds_micros: vec![10_000, 20_000, 40_000, 80_000],
            cumulative: cumulative.to_vec(),
            count,
        }
    }

    #[test]
    fn interval_quantiles_interpolate_the_bucket_deltas() {
        let before = histogram(&[5, 5, 5, 5], 5);
        // Ten new observations: four up to 10 ms, four in (10, 20], two in (20, 40].
        let now = histogram(&[9, 13, 15, 15], 15);
        let latency = interval_latency(&now, &before);
        assert_eq!(latency["count"], 10);
        // Rank 5 is the first of four in (10, 20] ms: 10 + 10 * 1/4.
        assert_eq!(latency["p50"], 0.0125);
        // Rank 9.5 is three quarters into (20, 40] ms: 20 + 20 * 1.5/2.
        assert_eq!(latency["p95"], 0.035);
        let idle = interval_latency(&now, &now);
        assert_eq!(idle, json!({"count": 0, "p50": null, "p95": null}));
        // Observations past the last bound report that bound.
        let beyond = interval_latency(&histogram(&[0, 0, 0, 0], 2), &LatencyHistogramV1::default());
        assert_eq!(beyond["p95"], 0.08);
    }

    #[test]
    fn summary_reports_counter_deltas_and_current_gauges() {
        let engine = |submitted: u64, used: u64| EngineSummaryV1 {
            requests_submitted: submitted,
            draft_tokens: submitted * 4,
            accepted_tokens: submitted * 3,
            kv_tokens_used: used,
            kv_tokens_capacity: 1_000,
            running: 2,
            worker_state: "ready",
            ..EngineSummaryV1::default()
        };
        let refusals = [("invalid_request", 3), ("capacity", 1)];
        let requests = RequestTotals {
            requests: 10,
            failures: [1, 4, 0],
        };
        let (first, base) = summary_line(
            Duration::from_secs(60),
            requests,
            &refusals,
            Some(engine(5, 400)),
            &SummaryBase::default(),
        );
        assert_eq!(first["requests"]["submitted"], 5);
        assert_eq!(first["refusals"]["invalid_request"], 3);
        assert_eq!(first["kv"]["usage"], 0.4);
        let later = RequestTotals {
            requests: 12,
            failures: [1, 6, 1],
        };
        let (second, _) = summary_line(
            Duration::from_secs(60),
            later,
            &[("invalid_request", 5), ("capacity", 1)],
            Some(engine(7, 400)),
            &base,
        );
        assert_eq!(second["http_requests"], 2);
        assert_eq!(
            second["failures"],
            json!({"error": 0, "warn": 2, "cancelled": 1})
        );
        assert_eq!(
            second["refusals"],
            json!({"invalid_request": 2, "capacity": 0})
        );
        assert_eq!(second["requests"]["submitted"], 2);
        assert_eq!(second["tokens"]["drafted"], 8);
        assert_eq!(second["acceptance"], 0.75);
        // Gauges stay current values: an unchanged pool is not zero.
        assert_eq!(second["kv"]["used_tokens"], 400);
        assert_eq!(second["kv"]["capacity_tokens"], 1_000);
        assert_eq!(second["running"], 2);
        assert_eq!(second["worker_state"], "ready");
        let (idle, _) = summary_line(
            Duration::from_secs(60),
            later,
            &[],
            Some(engine(7, 0)),
            &SummaryBase {
                engine: Some(engine(7, 0)),
                ..SummaryBase::default()
            },
        );
        assert_eq!(idle["acceptance"], Value::Null);
    }

    #[test]
    fn a_full_line_stays_under_four_kibibytes() {
        let captured = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink = {
            let captured = Arc::clone(&captured);
            RequestLogSinkV1::new(move |line| captured.lock().unwrap().push(line.to_owned()))
        };
        // Quotes and backslashes each escape to two bytes.
        let hostile = "\"\\".repeat(100_000);
        let log = RequestLog::new(Some(sink), vec![hostile.clone(); 20]);
        let record = log.open(&hostile, &hostile, None, Some(&hostile));
        record.set_request_id(&hostile);
        record.set_model(&hostile);
        record.fail(FailureDetail {
            code: Some(bounded_log_text(&hostile, CODE_BYTES)),
            param: Some(hostile.clone()),
            requested_model: Some(bounded_log_text(&hostile, NAME_BYTES)),
            ..FailureDetail::from_status(StatusCode::NOT_FOUND)
        });
        assert!(log.flush());
        let lines = captured.lock().unwrap().clone();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].len() < 4096, "{}", lines[0].len());
        let line: Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(line["schema"], FAILURE_SCHEMA);
        assert_eq!(line["level"], "warn");
        assert!(line["method"].as_str().unwrap().len() <= METHOD_BYTES);
        assert_eq!(line["served"].as_array().unwrap().len(), SERVED_NAMES);
    }

    #[test]
    fn each_record_writes_at_most_one_line_and_bursts_report_drops() {
        let captured = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink = {
            let captured = Arc::clone(&captured);
            RequestLogSinkV1::new(move |line| captured.lock().unwrap().push(line.to_owned()))
        };
        let log = RequestLog::new(Some(sink), Vec::new());
        let record = log.open("POST", "/v1/completions", None, None);
        record.fail(FailureDetail::backend(None));
        record.cancel("client_disconnected");
        record.fail(FailureDetail::backend(None));
        let succeeded = log.open("POST", "/v1/completions", None, None);
        succeeded.succeed();
        log.sweep(&worker_terminated_detail());
        assert!(log.flush());
        assert_eq!(captured.lock().unwrap().len(), 1);
        // Far more warn lines than the burst allows: the excess is counted.
        for _ in 0..500 {
            log.open("POST", "/v1/completions", None, None)
                .fail(FailureDetail::from_status(StatusCode::BAD_REQUEST));
        }
        assert!(log.flush());
        let lines = captured.lock().unwrap().clone();
        let dropped = lines
            .iter()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .filter(|line| line["schema"] == DROPPED_SCHEMA)
            .map(|line| line["dropped"]["warn"].as_u64().unwrap())
            .sum::<u64>();
        let written = lines.len() as u64
            - 1
            - lines
                .iter()
                .filter(|line| line.contains(DROPPED_SCHEMA))
                .count() as u64;
        assert!(dropped > 0);
        assert_eq!(written + dropped, 500);
        assert_eq!(log.totals().failures, [1, 500, 0]);
    }
}
