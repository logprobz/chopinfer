//! Bounded, best-effort routine diagnostics. Errors use synchronous logging.
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, mpsc};
use std::time::{Duration, Instant};

const CAPACITY: usize = 4096;
const REPORT_INTERVAL: Duration = Duration::from_secs(1);
const FLUSH_TIMEOUT: Duration = Duration::from_secs(2);
const NO_TIMEOUT: u64 = u64::MAX;

struct Counts {
    dropped: AtomicU64,
    pending: AtomicU64,
    timeout_pending: AtomicU64,
}

struct Logger {
    sender: mpsc::SyncSender<Message>,
    counts: Arc<Counts>,
}

enum Message {
    Line(String),
    Flush(mpsc::SyncSender<bool>),
}

/// Snapshot when a flush finishes or reaches its deadline. Pending lines may
/// be abandoned if the process exits; they include a write still in progress.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FlushResult {
    pub flushed: bool,
    pub pending_lines: u64,
    pub dropped_lines: u64,
}

impl Logger {
    fn new(writer: impl Write + Send + 'static, capacity: usize) -> Self {
        let (sender, receiver) = mpsc::sync_channel(capacity);
        let counts = Arc::new(Counts {
            dropped: AtomicU64::new(0),
            pending: AtomicU64::new(0),
            timeout_pending: AtomicU64::new(NO_TIMEOUT),
        });
        let counter = Arc::clone(&counts);
        std::thread::Builder::new()
            .name("diagnostic-writer".into())
            .spawn(move || write_loop(writer, receiver, counter))
            .expect("start diagnostic writer");
        Self { sender, counts }
    }

    fn line(&self, line: String) {
        self.counts.pending.fetch_add(1, Ordering::Relaxed);
        if self.sender.try_send(Message::Line(line)).is_err() {
            self.counts.pending.fetch_sub(1, Ordering::Relaxed);
            self.counts.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn flush(&self, timeout: Duration) -> FlushResult {
        let started = Instant::now();
        let (sender, receiver) = mpsc::sync_channel(1);
        let mut message = Message::Flush(sender);
        let mut flushed = false;
        loop {
            match self.sender.try_send(message) {
                Ok(()) => {
                    flushed = receiver.recv_timeout(timeout.saturating_sub(started.elapsed()))
                        == Ok(true);
                    break;
                }
                Err(mpsc::TrySendError::Disconnected(_)) => break,
                Err(mpsc::TrySendError::Full(returned)) => message = returned,
            }
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                break;
            }
            std::thread::sleep(remaining.min(Duration::from_millis(1)));
        }
        let result = FlushResult {
            flushed,
            pending_lines: self.counts.pending.load(Ordering::Relaxed),
            dropped_lines: self.counts.dropped.load(Ordering::Relaxed),
        };
        if !flushed {
            // Do not touch stderr here: even its lock can be held by the
            // stalled writer. Report from that thread if output resumes.
            self.counts
                .timeout_pending
                .store(result.pending_lines, Ordering::Relaxed);
        }
        result
    }
}

fn report(writer: &mut impl Write, counts: &Counts, reported: &mut u64, shutdown: bool) {
    let pending = counts.timeout_pending.swap(NO_TIMEOUT, Ordering::Relaxed);
    if pending != NO_TIMEOUT {
        let _ = writeln!(
            writer,
            "diagnostic writer: flush timed out; {pending} pending lines may be abandoned on exit"
        );
    }
    let total = counts.dropped.load(Ordering::Relaxed);
    if (total != *reported || (shutdown && total > 0))
        && writer
            .write_all(
                format!(
                    "diagnostic writer: dropped {} lines (total {total})\n",
                    total - *reported,
                )
                .as_bytes(),
            )
            .is_ok()
    {
        *reported = total;
    }
}

fn write_loop(mut writer: impl Write, receiver: mpsc::Receiver<Message>, counts: Arc<Counts>) {
    let mut reported = 0;
    let mut last_report = Instant::now();
    loop {
        match receiver.recv_timeout(REPORT_INTERVAL) {
            Ok(Message::Line(mut line)) => {
                line.push('\n');
                if writer.write_all(line.as_bytes()).is_err() {
                    counts.dropped.fetch_add(1, Ordering::Relaxed);
                }
                counts.pending.fetch_sub(1, Ordering::Relaxed);
            }
            Ok(Message::Flush(done)) => {
                report(&mut writer, &counts, &mut reported, true);
                let _ = done.send(writer.flush().is_ok());
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
        if last_report.elapsed() >= REPORT_INTERVAL {
            report(&mut writer, &counts, &mut reported, false);
            last_report = Instant::now();
        }
    }
    report(&mut writer, &counts, &mut reported, true);
    let _ = writer.flush();
}

static LOGGER: OnceLock<Logger> = OnceLock::new();

/// The untyped bundle log callback defaults to reliable delivery. Recognize
/// only routine trace schema kinds, without depending on a model or bundle.
pub fn is_routine_bundle_trace(message: &[u8]) -> bool {
    if message.first() != Some(&b'{') {
        return false;
    }
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(message) else {
        return false;
    };
    value
        .get("schema")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|schema| {
            schema.starts_with("chopinfer.")
                && (schema.ends_with(".step-shape.v1")
                    || schema.ends_with(".commit-shape.v1")
                    || schema == "chopinfer.graph-cache.v1"
                    || schema.ends_with(".drafter-catch-up.v1")
                    || schema.ends_with(".drafter-confidence.v1"))
        })
}

/// Enqueue an already formatted routine line without waiting for writer
/// progress. Retained lines from each producer keep their submission order.
/// Do not use for startup failures, errors or fatal diagnostics.
pub fn line(line: String) {
    LOGGER
        .get_or_init(|| Logger::new(std::io::stderr(), CAPACITY))
        .line(line);
}

/// Try to drain routine diagnostics within two seconds, after producers stop.
/// A stalled output device cannot delay the caller beyond this wait budget.
/// A timeout report is best effort from the writer if output becomes available.
pub fn flush() -> FlushResult {
    LOGGER.get().map_or(
        FlushResult {
            flushed: true,
            pending_lines: 0,
            dropped_lines: 0,
        },
        |logger| logger.flush(FLUSH_TIMEOUT),
    )
}

/// Flush on command return, including error returns and unwinding. The shared
/// writer remains available for subsequent commands in an embedded host.
pub struct FlushOnDrop {
    _private: (),
}

impl Default for FlushOnDrop {
    fn default() -> Self {
        LOGGER.get_or_init(|| Logger::new(std::io::stderr(), CAPACITY));
        Self { _private: () }
    }
}

impl Drop for FlushOnDrop {
    fn drop(&mut self) {
        let _ = flush();
    }
}

/// Format a routine line as `eprintln!`, then enqueue it for the writer.
#[macro_export]
macro_rules! diagnostic {
    ($($arg:tt)*) => { $crate::logging::line(format!($($arg)*)) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct Writer {
        bytes: Arc<Mutex<Vec<u8>>>,
        entered: Option<mpsc::SyncSender<()>>,
        release: mpsc::Receiver<()>,
        flushed: mpsc::Sender<()>,
    }

    impl Write for Writer {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if let Some(entered) = self.entered.take() {
                entered.send(()).unwrap();
                self.release.recv().unwrap();
            }
            self.bytes.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            let _ = self.flushed.send(());
            Ok(())
        }
    }

    #[test]
    fn stalled_writer_drops_whole_confidence_records_and_counts_them() {
        const FIRST: &str =
            r#"bundle: {"schema":"chopinfer.example.drafter-confidence.v1","step_id":"1"}"#;
        const SECOND: &str =
            r#"bundle: {"schema":"chopinfer.example.drafter-confidence.v1","step_id":"2"}"#;
        const DROPPED: &str =
            r#"bundle: {"schema":"chopinfer.example.drafter-confidence.v1","step_id":"3"}"#;
        for line in [FIRST, SECOND, DROPPED] {
            assert!(is_routine_bundle_trace(
                line.strip_prefix("bundle: ").unwrap().as_bytes()
            ));
        }
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let (entered, waiting) = mpsc::sync_channel(0);
        let (release, released) = mpsc::channel();
        let (flushed, flushes) = mpsc::channel();
        let logger = Arc::new(Logger::new(
            Writer {
                bytes: Arc::clone(&bytes),
                entered: Some(entered),
                release: released,
                flushed,
            },
            1,
        ));
        logger.line(FIRST.into());
        waiting.recv_timeout(Duration::from_secs(30)).unwrap();
        let producer = Arc::clone(&logger);
        let (done, completed) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            producer.line(SECOND.into());
            for _ in 0..100 {
                producer.line(DROPPED.into());
            }
            done.send(()).unwrap();
        });
        let result = completed.recv_timeout(Duration::from_secs(30));
        // Release even on failure so a regression cannot leave a stuck test.
        release.send(()).unwrap();
        worker.join().unwrap();
        result.expect("producer waited for stalled output");
        assert_eq!(logger.counts.dropped.load(Ordering::Relaxed), 100);
        assert!(logger.flush(Duration::from_secs(30)).flushed);
        flushes.recv_timeout(Duration::from_secs(30)).unwrap();
        let output = String::from_utf8(bytes.lock().unwrap().clone()).unwrap();
        assert!(output.starts_with(&format!("{FIRST}\n{SECOND}\n")));
        assert!(output.contains("dropped 100 lines (total 100)"));
        assert!(!output.contains(DROPPED));
        drop(logger);
        flushes
            .recv_timeout(Duration::from_secs(30))
            .expect("shutdown flush");
    }

    #[test]
    fn idle_writer_reports_drops_periodically_and_again_at_shutdown() {
        struct Reports(mpsc::Sender<u8>);
        impl Write for Reports {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                for byte in bytes {
                    self.0.send(*byte).unwrap();
                }
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let (sender, receiver) = mpsc::channel();
        let logger = Logger::new(Reports(sender), 1);
        logger.counts.dropped.store(7, Ordering::Relaxed);
        let read_line = || {
            let mut bytes = Vec::new();
            loop {
                let byte = receiver.recv_timeout(Duration::from_secs(30)).unwrap();
                bytes.push(byte);
                if byte == b'\n' {
                    break;
                }
            }
            String::from_utf8(bytes).unwrap()
        };
        assert_eq!(
            read_line(),
            "diagnostic writer: dropped 7 lines (total 7)\n"
        );
        drop(logger);
        assert_eq!(
            read_line(),
            "diagnostic writer: dropped 0 lines (total 7)\n"
        );
    }

    #[test]
    fn only_recognized_routine_bundle_traces_allow_lossy_delivery() {
        assert!(is_routine_bundle_trace(
            br#"{"schema":"chopinfer.graph-cache.v1"}"#
        ));
        for kind in [
            "step-shape",
            "commit-shape",
            "drafter-catch-up",
            "drafter-confidence",
        ] {
            assert!(is_routine_bundle_trace(
                format!(r#"{{"schema":"chopinfer.example.{kind}.v1"}}"#).as_bytes()
            ));
        }
        for message in [
            "host component read failed: authenticated digest differs",
            "generation engine step failed: executor failure",
            "bundle create failed: invalid rank image",
            r#"{"schema":"chopinfer.example.error.v1","error":"digest"}"#,
            r#"{"schema":"chopinfer.example.unknown.v1"}"#,
            r#"{"schema":"other.step-shape.v1"}"#,
            r#"{"schema":"other.graph-cache.v1"}"#,
            r#"{"schema":"chopinfer.graph-cache.v2"}"#,
            r#"{"schema":"other.drafter-confidence.v1"}"#,
            r#"{"schema":"chopinfer.example.drafter-confidence.v2"}"#,
            r#"{"schema":"chopinfer.example.step-shape.v1""#,
        ] {
            assert!(!is_routine_bundle_trace(message.as_bytes()), "{message}");
        }
    }

    #[test]
    fn flush_deadline_covers_full_queue_and_stalled_writer_acknowledgement() {
        for full in [true, false] {
            let bytes = Arc::new(Mutex::new(Vec::new()));
            let (entered, waiting) = mpsc::sync_channel(0);
            let (release, released) = mpsc::channel();
            let (flushed, _flushes) = mpsc::channel();
            let logger = Arc::new(Logger::new(
                Writer {
                    bytes: Arc::clone(&bytes),
                    entered: Some(entered),
                    release: released,
                    flushed,
                },
                1,
            ));
            logger.line("first".into());
            waiting.recv_timeout(Duration::from_secs(30)).unwrap();
            if full {
                logger.line("second".into());
                logger.line("dropped".into());
            }
            let flusher = Arc::clone(&logger);
            let (done, completed) = mpsc::channel();
            let worker = std::thread::spawn(move || {
                done.send(flusher.flush(Duration::from_millis(20))).unwrap();
            });
            // A handshake holds the writer until flush returns. The generous
            // watchdog catches deadlocks, not scheduler timing under load.
            let result = completed.recv_timeout(Duration::from_secs(30));
            release.send(()).unwrap();
            worker.join().unwrap();
            let result = result.expect("flush waited for stalled output");
            assert_eq!(
                result,
                FlushResult {
                    flushed: false,
                    pending_lines: if full { 2 } else { 1 },
                    dropped_lines: u64::from(full),
                }
            );
            assert!(logger.flush(Duration::from_secs(30)).flushed);
            let output = String::from_utf8(bytes.lock().unwrap().clone()).unwrap();
            assert!(output.contains("pending lines may be abandoned on exit"));
            assert_eq!(logger.counts.pending.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn disconnect_drains_and_flushes_retained_lines() {
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let (_, released) = mpsc::channel();
        let (flushed, flushes) = mpsc::channel();
        let logger = Logger::new(
            Writer {
                bytes: Arc::clone(&bytes),
                entered: None,
                release: released,
                flushed,
            },
            4,
        );
        logger.line("one".into());
        logger.line("two".into());
        drop(logger);
        flushes.recv_timeout(Duration::from_secs(30)).unwrap();
        assert_eq!(*bytes.lock().unwrap(), b"one\ntwo\n");
    }
}
