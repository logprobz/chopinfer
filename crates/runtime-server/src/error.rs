use axum::Json;
use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::json;
use std::net::SocketAddr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::time::{Duration, Instant};

use crate::frontend::FrontendModelV1;

pub(crate) const X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");
const REFUSAL_LOG_QUEUE: usize = 32;
const REFUSAL_LOGS_PER_SECOND: u32 = 4;
static REFUSAL_LOGGER: OnceLock<Option<SyncSender<LogMessage>>> = OnceLock::new();
static REFUSAL_LOG_DROPPED: AtomicU64 = AtomicU64::new(0);

enum LogMessage {
    Line(String),
    Flush(SyncSender<()>),
}

struct RefusalRateLimit {
    window: Instant,
    emitted: u32,
    suppressed: u64,
}

impl RefusalRateLimit {
    fn new(now: Instant) -> Self {
        Self {
            window: now,
            emitted: 0,
            suppressed: 0,
        }
    }

    fn rollover(&mut self, now: Instant, dropped: u64) -> Option<String> {
        self.suppressed = self.suppressed.saturating_add(dropped);
        if now.duration_since(self.window) < Duration::from_secs(1) {
            return None;
        }
        let summary = (self.suppressed > 0)
            .then(|| format!("INFO unknown_model refusals suppressed={}", self.suppressed));
        self.window = now;
        self.emitted = 0;
        self.suppressed = 0;
        summary
    }

    fn allow_line(&mut self) -> bool {
        if self.emitted < REFUSAL_LOGS_PER_SECOND {
            self.emitted += 1;
            true
        } else {
            self.suppressed = self.suppressed.saturating_add(1);
            false
        }
    }

    fn take_summary(&mut self, dropped: u64) -> Option<String> {
        self.suppressed = self.suppressed.saturating_add(dropped);
        let suppressed = std::mem::take(&mut self.suppressed);
        (suppressed > 0).then(|| format!("INFO unknown_model refusals suppressed={suppressed}"))
    }
}

pub(crate) fn initialize_refusal_logger() {
    REFUSAL_LOGGER
        .get_or_init(|| {
            let (sender, receiver) = sync_channel::<LogMessage>(REFUSAL_LOG_QUEUE);
            std::thread::Builder::new()
                .name("unknown-model-log".to_owned())
                .spawn(move || run_refusal_logger(receiver))
                .ok()
                .map(|_| sender)
        })
        .as_ref();
}

fn run_refusal_logger(receiver: Receiver<LogMessage>) {
    let mut limit = RefusalRateLimit::new(Instant::now());
    loop {
        let next = receiver.recv_timeout(Duration::from_millis(250));
        if let Some(summary) = limit.rollover(
            Instant::now(),
            REFUSAL_LOG_DROPPED.swap(0, Ordering::Relaxed),
        ) {
            eprintln!("{summary}");
        }
        match next {
            Ok(LogMessage::Line(line)) if limit.allow_line() => eprintln!("{line}"),
            Ok(LogMessage::Line(_)) | Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Ok(LogMessage::Flush(done)) => {
                if let Some(summary) =
                    limit.take_summary(REFUSAL_LOG_DROPPED.swap(0, Ordering::Relaxed))
                {
                    eprintln!("{summary}");
                }
                let _ = done.try_send(());
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
}

fn enqueue_refusal(sender: &SyncSender<LogMessage>, dropped: &AtomicU64, line: String) {
    if sender.try_send(LogMessage::Line(line)).is_err() {
        dropped.fetch_add(1, Ordering::Relaxed);
    }
}

fn flush_sender(sender: &SyncSender<LogMessage>, timeout: Duration) -> bool {
    let started = Instant::now();
    let (done, ack) = sync_channel(1);
    let mut message = LogMessage::Flush(done);
    loop {
        match sender.try_send(message) {
            Ok(()) => break,
            Err(TrySendError::Full(returned)) => message = returned,
            Err(TrySendError::Disconnected(_)) => return false,
        }
        let remaining = timeout.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            return false;
        }
        std::thread::sleep(remaining.min(Duration::from_millis(1)));
    }
    ack.recv_timeout(timeout.saturating_sub(started.elapsed()))
        .is_ok()
}

pub(crate) struct RefusalLogFlushGuard;

impl Drop for RefusalLogFlushGuard {
    fn drop(&mut self) {
        if let Some(Some(sender)) = REFUSAL_LOGGER.get() {
            let _ = flush_sender(sender, Duration::from_millis(500));
        }
    }
}

#[derive(Debug)]
pub(crate) struct ApiError {
    pub(crate) status: StatusCode,
    pub(crate) message: String,
    pub(crate) kind: &'static str,
    pub(crate) parameter: Option<String>,
    pub(crate) code: Option<String>,
}

impl ApiError {
    pub(crate) fn invalid(message: impl Into<String>, parameter: Option<&str>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            kind: "BadRequestError",
            code: None,
            parameter: parameter.map(str::to_owned),
        }
    }

    /// The same refusal, when the parameter name is already owned.
    pub(crate) fn invalid_owned(message: String, parameter: Option<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message,
            kind: "BadRequestError",
            code: None,
            parameter,
        }
    }

    pub(crate) fn capacity(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: message.into(),
            kind: "TooManyRequestsError",
            code: None,
            parameter: None,
        }
    }

    pub(crate) fn context_length(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            kind: "BadRequestError",
            code: None,
            parameter: None,
        }
    }

    pub(crate) fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
            kind: "InternalServerError",
            code: None,
            parameter: None,
        }
    }

    pub(crate) fn model_not_found(model: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: format!("model '{model}' is not available"),
            kind: "NotFoundError",
            code: None,
            parameter: Some("model".to_owned()),
        }
    }

    pub(crate) fn log_model_refusal(&self, peer: Option<SocketAddr>, served: &[FrontendModelV1]) {
        if let Some(line) = self.model_refusal_line(peer, served)
            && let Some(Some(sender)) = REFUSAL_LOGGER.get()
        {
            enqueue_refusal(sender, &REFUSAL_LOG_DROPPED, line);
        }
    }

    fn model_refusal_line(
        &self,
        peer: Option<SocketAddr>,
        served: &[FrontendModelV1],
    ) -> Option<String> {
        let model = self
            .message
            .strip_prefix("model '")
            .and_then(|value| value.strip_suffix("' is not available"))
            .filter(|_| {
                self.status == StatusCode::NOT_FOUND && self.parameter.as_deref() == Some("model")
            })?;
        let names = served
            .iter()
            .take(8)
            .map(|entry| safe_log_name(&entry.id))
            .collect::<Vec<_>>();
        let overflow = if served.len() > names.len() {
            " ..."
        } else {
            ""
        };
        Some(format!(
            "INFO unknown_model refusal requested={:?} peer={} served={names:?}{overflow}",
            safe_log_name(model),
            peer.map_or_else(|| "unknown".to_owned(), |address| address.to_string())
        ))
    }

    pub(crate) fn backend(message: impl Into<String>, code: Option<String>) -> Self {
        let status = match code.as_deref() {
            Some("shutdown_timeout" | "worker_unavailable") => StatusCode::SERVICE_UNAVAILABLE,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self {
            status,
            message: message.into(),
            kind: "InternalServerError",
            code,
            parameter: None,
        }
    }

    pub(crate) fn value(&self) -> serde_json::Value {
        json!({
            "error": {
                "message": self.message,
                "type": self.kind,
                "param": self.parameter,
                "code": self.code.as_ref().map_or_else(|| json!(self.status.as_u16()), |code| json!(code)),
            }
        })
    }

    pub(crate) fn into_response(self, request_id: &str) -> Response {
        let status = self.status;
        let mut response = (status, Json(self.value())).into_response();
        if status == StatusCode::SERVICE_UNAVAILABLE {
            response.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                HeaderValue::from_static("1"),
            );
        }
        insert_request_id(&mut response, request_id);
        response
    }
}

fn safe_log_name(value: &str) -> String {
    const LIMIT: usize = 128;
    let mut result = String::new();
    let mut chars = value.chars();
    for ch in chars.by_ref().take(LIMIT) {
        result.push(
            if ch.is_control() || matches!(ch, '\u{2028}' | '\u{2029}') {
                '?'
            } else {
                ch
            },
        );
    }
    if chars.next().is_some() {
        result.push_str("...");
    }
    result
}

pub(crate) fn insert_request_id(response: &mut Response, request_id: &str) {
    if let Ok(value) = HeaderValue::from_str(request_id) {
        response.headers_mut().insert(X_REQUEST_ID, value);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ApiError, LogMessage, RefusalRateLimit, enqueue_refusal, flush_sender, safe_log_name,
    };
    use crate::frontend::FrontendModelV1;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::mpsc::sync_channel;
    use std::time::{Duration, Instant};

    #[test]
    fn log_name_is_bounded_and_single_line() {
        assert_eq!(
            safe_log_name("a\n\r\t\0\u{1b}\u{7f}\u{2028}\u{2029}b"),
            "a????????b"
        );
        assert_eq!(safe_log_name(&"x".repeat(200)).len(), 131);
    }

    #[test]
    fn model_refusal_line_identifies_request_peer_and_served_names() {
        let served = vec![FrontendModelV1::new("served", "owner")];
        let peer = "127.0.0.1:12345".parse().unwrap();
        let line = ApiError::model_not_found("wrong\nname")
            .model_refusal_line(Some(peer), &served)
            .unwrap();
        assert_eq!(
            line,
            "INFO unknown_model refusal requested=\"wrong?name\" peer=127.0.0.1:12345 served=[\"served\"]"
        );
        assert!(
            ApiError::invalid("other", None)
                .model_refusal_line(Some(peer), &served)
                .is_none()
        );
    }

    #[test]
    fn refusal_log_rate_stays_bounded_under_sustained_volume() {
        let now = Instant::now();
        let mut limit = RefusalRateLimit::new(now);
        for second in 0..3 {
            let accepted = (0..1_000).filter(|_| limit.allow_line()).count();
            assert_eq!(accepted, 4);
            assert_eq!(
                limit.rollover(now + Duration::from_secs(second + 1), 17),
                Some("INFO unknown_model refusals suppressed=1013".to_owned())
            );
        }
        assert_eq!(limit.rollover(now + Duration::from_secs(4), 0), None);
    }

    #[test]
    fn served_names_are_bounded_and_unrelated_404_is_silent() {
        let served = (0..9)
            .map(|index| FrontendModelV1::new(format!("{index}{}", "x".repeat(200)), "owner"))
            .collect::<Vec<_>>();
        let line = ApiError::model_not_found("wrong")
            .model_refusal_line(None, &served)
            .unwrap();
        assert!(line.len() < 1_200);
        assert!(line.contains("0xxxx"));
        assert!(line.contains("7xxxx"));
        assert!(!line.contains("8xxxx"));
        assert!(line.ends_with(" ..."));
        let unrelated = ApiError {
            status: axum::http::StatusCode::NOT_FOUND,
            message: "Response not found".to_owned(),
            kind: "NotFoundError",
            code: None,
            parameter: Some("response_id".to_owned()),
        };
        assert!(unrelated.model_refusal_line(None, &served).is_none());
    }

    #[test]
    fn full_queue_does_not_block_producers_and_counts_suppressed_lines() {
        let (sender, receiver) = sync_channel(1);
        let (writer_ready, ready) = sync_channel(1);
        let (release_writer, release) = sync_channel(1);
        let writer = std::thread::spawn(move || {
            let _ = receiver.recv_timeout(Duration::from_secs(1));
            let _ = writer_ready.try_send(());
            let _ = release.recv_timeout(Duration::from_secs(2));
        });
        assert!(sender.try_send(LogMessage::Line("first".into())).is_ok());
        assert_eq!(ready.recv_timeout(Duration::from_secs(1)), Ok(()));
        assert!(sender.try_send(LogMessage::Line("queued".into())).is_ok());
        let dropped = Arc::new(AtomicU64::new(0));
        let producer_dropped = Arc::clone(&dropped);
        let (done, completed) = sync_channel(1);
        std::thread::spawn(move || {
            for _ in 0..1_000 {
                enqueue_refusal(&sender, &producer_dropped, "refused".into());
            }
            let _ = done.try_send(());
        });
        assert_eq!(completed.recv_timeout(Duration::from_millis(500)), Ok(()));
        assert_eq!(dropped.load(Ordering::Relaxed), 1_000);
        let _ = release_writer.try_send(());
        writer.join().unwrap();
    }

    #[test]
    fn shutdown_flush_has_a_deadline_even_when_writer_is_stalled() {
        let (sender, _stalled_writer) = sync_channel(1);
        assert!(sender.try_send(LogMessage::Line("first".into())).is_ok());
        let started = Instant::now();
        assert!(!flush_sender(&sender, Duration::from_millis(25)));
        assert!(started.elapsed() < Duration::from_millis(250));

        let (sender, receiver) = sync_channel(1);
        let writer = std::thread::spawn(move || {
            if let Ok(LogMessage::Flush(done)) = receiver.recv_timeout(Duration::from_secs(1)) {
                let _ = done.try_send(());
            }
        });
        assert!(flush_sender(&sender, Duration::from_millis(500)));
        writer.join().unwrap();
    }
}
