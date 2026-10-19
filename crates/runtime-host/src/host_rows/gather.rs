//! Adaptive read-ahead for checked host-mapped row copies.

/// Rows a gather copies between clock reads, and how long that many rows
/// may take before the rest of the gather is read ahead. A row the page cache
/// does not hold faults from the file in roughly 60 to 110 us on the
/// serving host, one at a time; resident rows take well under 1 us each.
const COLD_CHECK_ROWS: usize = 16;
const COLD_GROUP: std::time::Duration = std::time::Duration::from_micros(40);

/// How a gather reads the clock and asks for its remaining rows to be read
/// ahead. Read-ahead changes when pages arrive, never the bytes copied.
pub(crate) trait GatherProbe: Sync {
    fn now(&self) -> std::time::Duration;
    fn read_ahead(&self, rows: &[u32]);
}

/// The served probe: the monotonic clock and the mapping's read-ahead.
pub(crate) struct MappedProbe<'a> {
    mapping: &'a crate::host_rows::MappedRowsV1,
    row_bytes: usize,
    origin: std::time::Instant,
}

impl<'a> MappedProbe<'a> {
    pub(crate) fn new(mapping: &'a super::MappedRowsV1, row_bytes: usize) -> Self {
        Self {
            mapping,
            row_bytes,
            origin: std::time::Instant::now(),
        }
    }
}

impl GatherProbe for MappedProbe<'_> {
    fn now(&self) -> std::time::Duration {
        self.origin.elapsed()
    }

    fn read_ahead(&self, rows: &[u32]) {
        self.mapping.will_need_rows(self.row_bytes, rows);
    }
}

/// One gather's read-ahead: asked for at most once, after the first group
/// of rows that took longer than a resident group can.
pub(crate) struct ReadAhead<'a> {
    probe: &'a dyn GatherProbe,
    started: std::sync::atomic::AtomicBool,
}

impl<'a> ReadAhead<'a> {
    pub(crate) fn configured(probe: &'a dyn GatherProbe) -> Self {
        if mapped_read_ahead_v1() { Self::new(probe) } else { Self::disabled(probe) }
    }

    pub(crate) fn disabled(probe: &'a dyn GatherProbe) -> Self {
        Self { probe, started: std::sync::atomic::AtomicBool::new(true) }
    }

    pub(crate) fn new(probe: &'a dyn GatherProbe) -> Self {
        Self {
            probe,
            started: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

/// Copies one span, reading the clock after every `COLD_CHECK_ROWS`th row
/// of the gather. The first rows found slower on average than
/// `COLD_GROUP / COLD_CHECK_ROWS` each ask for every later row of the gather
/// to be read ahead, once per gather. Rows count across the gather, not the
/// span, so chunks of any size are checked.
pub(crate) fn gather_span_v1(
    table: &[u8],
    row_bytes: usize,
    rows: &[u32],
    at: usize,
    out: &mut [u8],
    ahead: &ReadAhead<'_>,
) {
    let mut written = 0;
    let mut group = ahead.probe.now();
    let mut finished: u32 = 0;
    while written < out.len() {
        let position = at + written;
        let (index, within) = (position / row_bytes, position % row_bytes);
        let take = (row_bytes - within).min(out.len() - written);
        let source = rows[index] as usize * row_bytes + within;
        out[written..written + take].copy_from_slice(&table[source..source + take]);
        written += take;
        if within + take < row_bytes {
            continue;
        }
        finished += 1;
        if (index + 1) % COLD_CHECK_ROWS != 0
            || index + 1 == rows.len()
            || ahead.started.load(std::sync::atomic::Ordering::Relaxed)
        {
            continue;
        }
        let now = ahead.probe.now();
        let slow = now.saturating_sub(group) * COLD_CHECK_ROWS as u32 > COLD_GROUP * finished;
        if slow
            && !ahead
                .started
                .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            ahead.probe.read_ahead(&rows[index + 1..]);
        }
        (group, finished) = (now, 0);
    }
}

/// Enable adaptive mapped-row read-ahead only when explicitly requested.
pub const MAPPED_READ_AHEAD_ENV_V1: &str = "CHOPINFER_MAPPED_READ_AHEAD";

pub(crate) fn mapped_read_ahead_v1() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os(MAPPED_READ_AHEAD_ENV_V1).is_some_and(|value| value == "1")
    })
}
