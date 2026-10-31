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

/// Model-neutral routine trace kinds: `chopinfer.<model>.<kind>`.
const ROUTINE_KINDS: [&str; 4] = [
    ".step-shape.v1",
    ".commit-shape.v1",
    ".drafter-catch-up.v1",
    ".drafter-confidence.v1",
];
const GRAPH_CACHE: &str = "chopinfer.graph-cache.v1";
const ROUTE_SCHEMA: &str = "chopinfer.route.v1";
/// Graph-cache events an operator reads at `info`: the cache's capacity, a
/// failed capture, the start of eviction and a full release. Captures and
/// eager fallbacks, and any event this host does not know, are routine.
const GRAPH_CACHE_EVENTS: [&str; 4] = [
    "capacity",
    "capture-failed",
    "eviction-begin",
    "release-all",
];

/// The level of one bundle log message, without depending on a model or
/// bundle, or `None` for a message this host cannot classify. The log ABI has
/// no severity and an unclassified message can be the sole failure detail, so
/// it reaches stderr synchronously at every level. Classified messages are
/// routine: they go through the bounded queue, and only when their level is
/// enabled.
pub fn bundle_trace_level(message: &[u8]) -> Option<runtime_log::Level> {
    if message.first() != Some(&b'{') || !names_known_schema(message) {
        return None;
    }
    // Borrow only the two fields; the rest of the record is skipped, never
    // built into a tree.
    #[derive(serde::Deserialize)]
    struct Head<'a> {
        #[serde(borrow, default)]
        schema: Option<std::borrow::Cow<'a, str>>,
        #[serde(borrow, default)]
        event: Option<std::borrow::Cow<'a, str>>,
    }
    let head = serde_json::from_slice::<Head<'_>>(message).ok()?;
    let schema = head.schema?;
    if schema == GRAPH_CACHE {
        return Some(
            if head
                .event
                .is_some_and(|event| GRAPH_CACHE_EVENTS.contains(&event.as_ref()))
            {
                runtime_log::Level::Info
            } else {
                runtime_log::Level::Debug
            },
        );
    }
    is_routine_schema(&schema).then_some(runtime_log::Level::Debug)
}

fn is_routine_schema(schema: &str) -> bool {
    schema == ROUTE_SCHEMA
        || (schema.starts_with("chopinfer.")
            && ROUTINE_KINDS.iter().any(|kind| schema.ends_with(kind)))
}

/// Whether `message` holds a JSON string naming a classified schema, a byte
/// scan that keeps unrelated diagnostics away from the JSON parser.
fn names_known_schema(message: &[u8]) -> bool {
    const OPEN: &[u8] = b"\"chopinfer.";
    let mut rest = message;
    while let Some(start) = rest.windows(OPEN.len()).position(|bytes| bytes == OPEN) {
        let name = &rest[start + 1..];
        let Some(end) = name.iter().position(|&byte| byte == b'"') else {
            return false;
        };
        if std::str::from_utf8(&name[..end])
            .is_ok_and(|name| name == GRAPH_CACHE || is_routine_schema(name))
        {
            return true;
        }
        rest = &name[end..];
    }
    false
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
            assert_eq!(
                bundle_trace_level(line.strip_prefix("bundle: ").unwrap().as_bytes()),
                Some(runtime_log::Level::Debug)
            );
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
    fn only_recognized_routine_bundle_traces_are_leveled() {
        use runtime_log::Level::{Debug, Info};
        for kind in [
            "step-shape",
            "commit-shape",
            "drafter-catch-up",
            "drafter-confidence",
        ] {
            assert_eq!(
                bundle_trace_level(
                    format!(r#"{{"schema":"chopinfer.example.{kind}.v1"}}"#).as_bytes()
                ),
                Some(Debug),
                "{kind}"
            );
        }
        for (message, level) in [
            (
                r#"{"schema":"chopinfer.route.v1","fallback_launches":0}"#,
                Debug,
            ),
            (r#"{"schema":"chopinfer.graph-cache.v1"}"#, Debug),
            // Field order does not matter, and a routine record names its kind once.
            (
                r#"{"family":"verify","event":"capture","schema":"chopinfer.graph-cache.v1"}"#,
                Debug,
            ),
            (
                r#"{"schema":"chopinfer.graph-cache.v1","event":"eager"}"#,
                Debug,
            ),
            (
                r#"{"schema":"chopinfer.graph-cache.v1","event":"target-eager"}"#,
                Debug,
            ),
            (
                r#"{"schema":"chopinfer.graph-cache.v1","event":"some-new-event"}"#,
                Debug,
            ),
            (
                r#"{"schema":"chopinfer.graph-cache.v1","event":"capacity","capacity":64}"#,
                Info,
            ),
            (
                r#"{"schema":"chopinfer.graph-cache.v1","event":"capture-failed"}"#,
                Info,
            ),
            (
                r#"{"event":"eviction-begin","schema":"chopinfer.graph-cache.v1"}"#,
                Info,
            ),
            (
                r#"{"schema":"chopinfer.graph-cache.v1","event":"release-all"}"#,
                Info,
            ),
        ] {
            assert_eq!(
                bundle_trace_level(message.as_bytes()),
                Some(level),
                "{message}"
            );
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
            // A routine schema named outside the schema field is not routine.
            r#"{"schema":"chopinfer.example.error.v1","after":"chopinfer.example.step-shape.v1"}"#,
            r#"{"message":"chopinfer.route.v1"}"#,
            r#"{"schema":"chopinfer.route.v1","schema":"chopinfer.example.error.v1"}"#,
        ] {
            assert_eq!(bundle_trace_level(message.as_bytes()), None, "{message}");
        }
    }

    #[test]
    fn unrelated_bundle_messages_never_reach_the_json_parser() {
        for message in [
            b"plain diagnostic".as_slice(),
            br#"{"schema":"chopinfer.example.error.v1","detail":"x"}"#,
            br#"{"schema":"chopinfer.route.v10"}"#,
            br#"{"schema":"chopinfer.graph-cache.v1x"}"#,
            br#"{"message":"chopinfer.example.step-shape.v1 without quotes around it}"#,
            br#"{"schema":"chopinfer.unterminated"#,
        ] {
            assert!(
                !names_known_schema(message),
                "{}",
                String::from_utf8_lossy(message)
            );
        }
        for message in [
            br#"{"schema":"chopinfer.graph-cache.v1"}"#.as_slice(),
            br#"{"a":"chopinfer.x","schema":"chopinfer.route.v1"}"#,
            br#"{"schema":"chopinfer.example.drafter-confidence.v1"}"#,
        ] {
            assert!(
                names_known_schema(message),
                "{}",
                String::from_utf8_lossy(message)
            );
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

thread_local! {
    static ROUTE: std::cell::RefCell<Option<serde_json::Value>> = const { std::cell::RefCell::new(None) };
}

/// Called synchronously by the bundle log service on the stepping thread.
/// Routine log delivery may drop text, but the step retains this structured copy.
pub fn record_route(message: &[u8]) {
    let Some(mut value) = parse_route(message, |bytes| serde_json::from_slice(bytes).ok()) else {
        return;
    };
    if value["schema"] == "chopinfer.route.v1" {
        value
            .as_object_mut()
            .expect("route object")
            .remove("schema");
        ROUTE.with(|r| {
            let mut slot = r.borrow_mut();
            let total = slot.get_or_insert_with(|| {
                serde_json::json!({
                    "fallback_launches": 0, "families": {}, "entries": {}
                })
            });
            if accumulate_route(total, &value).is_none() {
                // Keep the whole step invalid if a callback supplies inconsistent
                // metadata. A later callback cannot turn partial counts into proof.
                *total = serde_json::json!({"invalid": true});
            }
        });
    }
}

fn parse_route(
    message: &[u8],
    parse: impl FnOnce(&[u8]) -> Option<serde_json::Value>,
) -> Option<serde_json::Value> {
    // Bundle records use this literal schema. Check bytes before allocating
    // a JSON tree for unrelated diagnostics, regardless of field order.
    const SCHEMA: &[u8] = b"\"chopinfer.route.v1\"";
    if !message.windows(SCHEMA.len()).any(|bytes| bytes == SCHEMA) {
        return None;
    }
    parse(message)
}

fn accumulate_route(total: &mut serde_json::Value, next: &serde_json::Value) -> Option<()> {
    total["fallback_launches"] = total["fallback_launches"]
        .as_u64()?
        .checked_add(next["fallback_launches"].as_u64()?)?
        .into();
    for (family, count) in next["families"].as_object()? {
        let count = count.as_u64()?;
        let previous = total["families"]
            .as_object_mut()?
            .entry(family.clone())
            .or_insert(0.into());
        *previous = previous.as_u64()?.checked_add(count)?.into();
    }
    for (entry, count) in next["entries"].as_object()? {
        let launches = count["launches"].as_u64()?;
        for field in ["symbol", "class", "family"] {
            count[field].as_str()?;
        }
        let wrappers = count["wrappers"].as_array()?;
        if wrappers.iter().any(|value| !value.is_string()) {
            return None;
        }
        let entries = total["entries"].as_object_mut()?;
        if let Some(previous) = entries.get_mut(entry) {
            if ["symbol", "class", "family"]
                .iter()
                .any(|field| previous[*field] != count[*field])
            {
                return None;
            }
            previous["launches"] = previous["launches"].as_u64()?.checked_add(launches)?.into();
            let previous_wrappers = previous["wrappers"].as_array_mut()?;
            for wrapper in wrappers {
                if !previous_wrappers.contains(wrapper) {
                    previous_wrappers.push(wrapper.clone());
                }
            }
        } else {
            entries.insert(entry.clone(), count.clone());
        }
    }
    if let Some(ranks) = next.get("ranks") {
        let target = total
            .as_object_mut()?
            .entry("ranks")
            .or_insert_with(|| serde_json::json!({}));
        for (rank, counts) in ranks.as_object()? {
            if rank != "unknown" && rank.parse::<u32>().is_err() {
                return None;
            }
            let target = target
                .as_object_mut()?
                .entry(rank.clone())
                .or_insert_with(|| serde_json::json!({"tagged": {}, "untagged": {}}));
            for kind in ["tagged", "untagged"] {
                for (symbol, launches) in counts[kind].as_object()? {
                    let previous = target[kind]
                        .as_object_mut()?
                        .entry(symbol.clone())
                        .or_insert(0.into());
                    *previous = previous.as_u64()?.checked_add(launches.as_u64()?)?.into();
                }
            }
        }
    }
    Some(())
}

pub(crate) fn take_route() -> Option<serde_json::Value> {
    ROUTE.with(|r| r.take())
}

#[cfg(test)]
mod route_tests {
    use super::*;

    #[test]
    fn rank_census_accumulates_callbacks_and_refuses_overflow() {
        let mut total = serde_json::json!({"fallback_launches": 0, "families": {}, "entries": {}});
        let next = serde_json::json!({"fallback_launches": 0, "families": {}, "entries": {},
            "ranks": {"1": {"tagged": {"norm/fast": 2}, "untagged": {"plain": 3}}}});
        assert_eq!(accumulate_route(&mut total, &next), Some(()));
        assert_eq!(accumulate_route(&mut total, &next), Some(()));
        assert_eq!(total["ranks"]["1"]["tagged"]["norm/fast"], 4);
        assert_eq!(total["ranks"]["1"]["untagged"]["plain"], 6);
        total["ranks"]["1"]["untagged"]["plain"] = u64::MAX.into();
        assert_eq!(accumulate_route(&mut total, &next), None);
    }

    #[test]
    fn unrelated_route_messages_never_reach_the_json_parser() {
        for message in [
            b"plain diagnostic".as_slice(),
            br#"{"schema":"chopinfer.graph-cache.v1","graphs":4}"#,
            br#"{"schema":"chopinfer.route.v10"}"#,
            br#"{"message":"mentions chopinfer.route.v1 without a schema"}"#,
        ] {
            assert!(parse_route(message, |_| panic!("unrelated JSON parsed")).is_none());
        }
        record_route(br#"{"schema":"other.v1","message":"chopinfer.route.v1"}"#);
        assert!(take_route().is_none());
        record_route(br#"{"schema" : "chopinfer.route.v1", "fallback_launches":0,"families":{},"entries":{}}"#);
        assert_eq!(take_route().unwrap()["fallback_launches"], 0);
    }

    #[test]
    fn route_records_add_execute_and_commit_launches_until_step_end() {
        assert!(take_route().is_none());
        for count in [2, 3] {
            record_route(
                serde_json::json!({"schema": "chopinfer.route.v1",
                "fallback_launches": count, "families": {"norm": count},
                "entries": {"norm/fallback": {"symbol": "norm", "class": "Baseline",
                    "family": "norm", "wrappers": ["fallback"], "launches": count}}})
                .to_string()
                .as_bytes(),
            );
        }
        let route = take_route().unwrap();
        assert_eq!(route["fallback_launches"], 5);
        assert_eq!(route["families"]["norm"], 5);
        assert_eq!(route["entries"]["norm/fallback"]["launches"], 5);
        assert_eq!(route["entries"]["norm/fallback"]["class"], "Baseline");
        assert!(take_route().is_none());
    }

    #[test]
    fn route_record_is_retained_on_the_stepping_thread_until_taken() {
        assert!(take_route().is_none());
        record_route(br#"{"schema":"chopinfer.route.v1","fallback_launches":2,"families":{"norm":2},"entries":{}}"#);
        assert!(std::thread::spawn(take_route).join().unwrap().is_none());
        assert_eq!(take_route().unwrap()["fallback_launches"], 2);
        assert!(take_route().is_none());
        record_route(br#"{"schema":"other.v1"}"#);
        assert!(take_route().is_none());
    }
}
