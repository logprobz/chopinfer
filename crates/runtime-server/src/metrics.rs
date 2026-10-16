use std::sync::Arc;
use std::time::Instant;

/// Supplies monotonic time to metrics collection.
pub trait MonotonicClock: Send + Sync + 'static {
    fn now_micros(&self) -> u64;
}

/// The production monotonic clock, measured from its construction.
#[derive(Debug)]
pub struct SystemMonotonicClock {
    origin: Instant,
}

impl SystemMonotonicClock {
    #[must_use]
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Default for SystemMonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl MonotonicClock for SystemMonotonicClock {
    fn now_micros(&self) -> u64 {
        u64::try_from(self.origin.elapsed().as_micros()).unwrap_or(u64::MAX)
    }
}

/// Produces one complete Prometheus text snapshot.
///
/// The provider is separate from the HTTP server so runtime components can be
/// composed without introducing a dependency on the transport crate.
pub trait MetricsSnapshotProvider: Send + Sync + 'static {
    fn snapshot(&self, monotonic_micros: u64) -> String;

    /// Optional bounded step profile JSON. Disabled providers return no data.
    fn step_profile(&self) -> Option<String> {
        None
    }
}

impl<F> MetricsSnapshotProvider for F
where
    F: Fn(u64) -> String + Send + Sync + 'static,
{
    fn snapshot(&self, monotonic_micros: u64) -> String {
        self(monotonic_micros)
    }
}

/// Adapts a deterministic snapshot closure into the server provider contract.
///
/// A higher layer can compose engine and KV snapshots here without making the
/// transport depend on either runtime crate.
pub fn metrics_snapshot_provider<F>(snapshot: F) -> Arc<dyn MetricsSnapshotProvider>
where
    F: Fn(u64) -> String + Send + Sync + 'static,
{
    Arc::new(snapshot)
}

#[derive(Debug, Default)]
pub(crate) struct EmptyMetricsSnapshotProvider;

impl MetricsSnapshotProvider for EmptyMetricsSnapshotProvider {
    fn snapshot(&self, monotonic_micros: u64) -> String {
        format!(
            "# TYPE chop_process_uptime_seconds gauge\nchop_process_uptime_seconds {}.{:06}\n",
            monotonic_micros / 1_000_000,
            monotonic_micros % 1_000_000
        )
    }
}

#[derive(Clone)]
pub(crate) struct MetricsEndpoint {
    provider: Arc<dyn MetricsSnapshotProvider>,
    clock: Arc<dyn MonotonicClock>,
}

impl MetricsEndpoint {
    pub(crate) fn new(
        provider: Arc<dyn MetricsSnapshotProvider>,
        clock: Arc<dyn MonotonicClock>,
    ) -> Self {
        Self { provider, clock }
    }

    pub(crate) fn step_profile(&self) -> Option<String> {
        self.provider.step_profile()
    }

    pub(crate) fn snapshot(&self) -> String {
        self.provider.snapshot(self.clock.now_micros())
    }
}

#[derive(Default)]
pub(crate) struct AdmissionMetrics {
    counts: [std::sync::atomic::AtomicU64; 8],
}

const REFUSAL_REASONS: [&str; 8] = [
    "invalid_request",
    "unknown_model",
    "not_found",
    "intake_timeout",
    "capacity",
    "unavailable",
    "internal",
    "other",
];

impl AdmissionMetrics {
    pub(crate) fn response(&self, status: axum::http::StatusCode) {
        if !status.is_client_error() && !status.is_server_error() {
            return;
        }
        let index = match status.as_u16() {
            400 | 413 | 415 | 422 => 0,
            404 => 1,
            408 => 3,
            429 => 4,
            503 => 5,
            500 => 6,
            _ => 7,
        };
        self.counts[index].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn stored_response(&self, status: axum::http::StatusCode) {
        if status == axum::http::StatusCode::NOT_FOUND {
            self.counts[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            self.response(status);
        }
    }

    pub(crate) fn prometheus(&self) -> String {
        use std::fmt::Write;
        let mut text = "# TYPE chop_admission_refusals_total counter\n".to_owned();
        for (reason, count) in REFUSAL_REASONS.iter().zip(&self.counts) {
            writeln!(
                text,
                "chop_admission_refusals_total{{reason=\"{reason}\"}} {}",
                count.load(std::sync::atomic::Ordering::Relaxed)
            )
            .unwrap();
        }
        text
    }
}
