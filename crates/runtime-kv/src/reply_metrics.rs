//! Fixed-label, counter-only contract for the optional reply history observer.
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};

/// Exclusive upper context positions. Intervals are split at every boundary.
pub const REPLY_CONTEXT_LIMITS: [usize; 5] = [32_768, 65_536, 131_072, 196_608, usize::MAX];
pub const REPLY_CONTEXT_LABELS: [&str; 5] = ["0_32k", "32_64k", "64_128k", "128_192k", "192k_plus"];

macro_rules! counters {
    ($($variant:ident => $label:literal),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        #[repr(usize)]
        pub enum ReplyHistoryCounter { $($variant),+ }
        pub const REPLY_COUNTER_LABELS: &[&str] = &[$($label),+];
    };
}
counters! {
    Submitted => "submitted",
    Compared => "compared",
    NoHistory => "no_history",
    ScopeOnly => "scope_without_ancestry",
    BeforeReply => "before_reply",
    InsideReply => "inside_reply",
    FinalToken => "final_token",
    CompleteReply => "complete_reply",
    Shorter => "shorter_consumer",
    Ambiguous => "ambiguous",
    AnyPage => "any_page",
    EligibleTurn => "eligible_turn",
    Natural => "finish_stop",
    ToolFinish => "finish_tool_calls",
    Length => "finish_length",
    Cancelled => "finish_cancelled",
    DeliveryFailed => "finish_delivery_failed",
    FrontendStop => "finish_frontend_stop",
    Evicted => "evicted",
    CapacityDrop => "drop_capacity",
    AllocationDrop => "drop_allocation",
    Unsupported => "drop_unsupported",
    LateAnnotation => "late_annotation",
    Annotated => "annotated_tokens",
    Unknown => "unknown_tokens",
    Mixed => "mixed_tokens",
    EndToken => "end_tokens",
    IgnoredEndToken => "ignored_end_tokens",
    RegionUnknown => "mismatch_unknown",
    RegionReasoning => "mismatch_reasoning",
    RegionContent => "mismatch_content",
    RegionTool => "mismatch_tool_calls",
    RegionEnd => "mismatch_end_token",
    RegionMixed => "mismatch_mixed",
    PositionFirst => "mismatch_first_quarter",
    PositionMiddle => "mismatch_middle_half",
    PositionLast => "mismatch_last_quarter",
    RetentionLoss => "retention_loss_possible",
    Fingerprints => "fingerprints_hashed",
    Probes => "digest_probes",
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum ReplyHistoryRows {
    Candidate,
    Eligible,
    Repeated,
    Rounding,
    Unobserved,
    Ambiguous,
    Prefilled,
}
pub const REPLY_ROW_LABELS: [&str; 7] = [
    "candidate",
    "eligible",
    "repeated",
    "rounding",
    "unobserved",
    "ambiguous",
    "prefilled",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplyHistorySnapshot {
    pub enabled: bool,
    /// All observer-owned capacities, including active and temporary storage.
    pub allocated_bytes: u64,
    pub entries: u64,
    pub counters: [u64; REPLY_COUNTER_LABELS.len()],
    pub rows: [[u64; 5]; 7],
}

#[derive(Debug)]
pub(crate) struct ReplyHistoryMetrics {
    enabled: AtomicU64,
    bytes: AtomicU64,
    entries: AtomicU64,
    counters: [AtomicU64; REPLY_COUNTER_LABELS.len()],
    rows: [[AtomicU64; 5]; 7],
}
impl Default for ReplyHistoryMetrics {
    fn default() -> Self {
        Self {
            enabled: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            entries: AtomicU64::new(0),
            counters: std::array::from_fn(|_| AtomicU64::new(0)),
            rows: std::array::from_fn(|_| std::array::from_fn(|_| AtomicU64::new(0))),
        }
    }
}
impl ReplyHistoryMetrics {
    pub(crate) fn count(&self, counter: ReplyHistoryCounter, count: usize) {
        self.counters[counter as usize].fetch_add(count as u64, Ordering::Relaxed);
    }
    pub(crate) fn rows(&self, kind: ReplyHistoryRows, start: usize, end: usize) {
        let mut lower = 0;
        for (index, &upper) in REPLY_CONTEXT_LIMITS.iter().enumerate() {
            let rows = end.min(upper).saturating_sub(start.max(lower));
            self.rows[kind as usize][index].fetch_add(rows as u64, Ordering::Relaxed);
            lower = upper;
        }
    }
    pub(crate) fn gauges(&self, enabled: bool, bytes: usize, entries: usize) {
        self.enabled.store(u64::from(enabled), Ordering::Relaxed);
        self.bytes.store(bytes as u64, Ordering::Relaxed);
        self.entries.store(entries as u64, Ordering::Relaxed);
    }
    pub(crate) fn snapshot(&self) -> ReplyHistorySnapshot {
        ReplyHistorySnapshot {
            enabled: self.enabled.load(Ordering::Relaxed) != 0,
            allocated_bytes: self.bytes.load(Ordering::Relaxed),
            entries: self.entries.load(Ordering::Relaxed),
            counters: std::array::from_fn(|i| self.counters[i].load(Ordering::Relaxed)),
            rows: std::array::from_fn(|i| {
                std::array::from_fn(|j| self.rows[i][j].load(Ordering::Relaxed))
            }),
        }
    }
    pub(crate) fn restore(&self, snapshot: ReplyHistorySnapshot) {
        self.gauges(
            snapshot.enabled,
            snapshot.allocated_bytes as usize,
            snapshot.entries as usize,
        );
        for (cell, value) in self.counters.iter().zip(snapshot.counters) {
            cell.store(value, Ordering::Relaxed);
        }
        for (row, values) in self.rows.iter().zip(snapshot.rows) {
            for (cell, value) in row.iter().zip(values) {
                cell.store(value, Ordering::Relaxed);
            }
        }
    }
}
impl ReplyHistorySnapshot {
    pub(crate) fn prometheus(&self, output: &mut String) {
        for (name, value) in [
            ("enabled", u64::from(self.enabled)),
            ("allocated_bytes", self.allocated_bytes),
            ("entries", self.entries),
        ] {
            writeln!(
                output,
                "# TYPE chop_reply_history_{name} gauge\nchop_reply_history_{name} {value}"
            )
            .unwrap();
        }
        writeln!(
            output,
            "# TYPE chop_reply_history_observations_total counter"
        )
        .unwrap();
        for (label, value) in REPLY_COUNTER_LABELS.iter().zip(self.counters) {
            writeln!(
                output,
                "chop_reply_history_observations_total{{outcome=\"{label}\"}} {value}"
            )
            .unwrap();
        }
        writeln!(output, "# TYPE chop_reply_history_rows_total counter").unwrap();
        for (kind, values) in REPLY_ROW_LABELS.iter().zip(self.rows) {
            for (bucket, value) in REPLY_CONTEXT_LABELS.iter().zip(values) {
                writeln!(
                    output,
                    "chop_reply_history_rows_total{{kind=\"{kind}\",context=\"{bucket}\"}} {value}"
                )
                .unwrap();
            }
        }
        let repeated: u64 = self.rows[ReplyHistoryRows::Repeated as usize].iter().sum();
        writeln!(output, "# TYPE chop_prefix_cache_recomputed_generated_tokens_total counter\nchop_prefix_cache_recomputed_generated_tokens_total {repeated}").unwrap();
    }
}
