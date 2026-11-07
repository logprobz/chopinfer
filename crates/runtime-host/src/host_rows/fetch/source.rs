//! The three reads a descriptor read of a row makes, behind one seam so tests
//! can script cold, short, unsupported and failing reads.
use crate::page_cache::{CacheAdviceV1, advise_v1, read_cached_v1};
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, Ordering};

/// How a read that does not wait for the device (`preadv2` with
/// `RWF_NOWAIT`) of one row ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Probe {
    /// The whole row, which is final.
    Full,
    /// Fewer bytes than the row (`0 <= n < width`): an uncached page.
    Short,
    /// The first page is not cached (`EAGAIN`).
    WouldBlock,
    /// The file system does not read without waiting (`EOPNOTSUPP`,
    /// `EINVAL`, or a platform without `preadv2`).
    Unsupported,
    /// A signal interrupted the read (`EINTR`).
    Interrupted,
    /// Any other error.
    Other,
}

impl Probe {
    /// The class of a read of `length` bytes that answered `result`.
    pub(super) fn of(result: &io::Result<usize>, length: usize) -> Self {
        match result {
            Ok(read) if *read == length => Self::Full,
            Ok(_) => Self::Short,
            Err(error) if error.kind() == io::ErrorKind::Unsupported => Self::Unsupported,
            Err(error) => match error.raw_os_error() {
                Some(libc::EAGAIN) => Self::WouldBlock,
                Some(libc::EOPNOTSUPP | libc::EINVAL) => Self::Unsupported,
                Some(libc::EINTR) => Self::Interrupted,
                _ => Self::Other,
            },
        }
    }
}

/// Reads rows of a descriptor. Every row ends as one exact read of its file
/// range: a probe that is not [`Probe::Full`] makes the row cold, and a cold
/// row is read again whole, whatever the probe left in its place.
pub(super) trait RowSource: Send + Sync {
    /// Copies the row at `offset` into `row` if the page cache holds it,
    /// without waiting for the device.
    fn probe(&self, file: &File, row: &mut [u8], offset: u64) -> Probe;
    /// Starts the device reads of a cold row's pages
    /// (`POSIX_FADV_WILLNEED`). A hint only: errors are ignored.
    fn advise(&self, file: &File, offset: u64, length: usize);
    /// Reads the whole row, waiting for the device.
    fn read_whole(&self, file: &File, row: &mut [u8], offset: u64) -> Result<(), String>;
}

/// The served source: the descriptor's reads.
pub(super) struct Descriptor;

impl RowSource for Descriptor {
    fn probe(&self, file: &File, row: &mut [u8], offset: u64) -> Probe {
        let length = row.len();
        Probe::of(&read_cached_v1(file, row, offset), length)
    }

    fn advise(&self, file: &File, offset: u64, length: usize) {
        // A zero length would advise the whole file.
        if length > 0 {
            let _ = advise_v1(file, offset, length as u64, CacheAdviceV1::WillNeed);
        }
    }

    fn read_whole(&self, file: &File, row: &mut [u8], offset: u64) -> Result<(), String> {
        file.read_exact_at(row, offset)
            .map_err(|error| format!("read mapped row source: {error}"))
    }
}

/// Probes by class, process-wide: full, short, would block, unsupported,
/// and every other end (interruptions included).
static PROBES: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// The probe classes one reader saw, added to the process counts at once.
#[derive(Default)]
pub(super) struct Classes([u64; 5]);

impl Classes {
    pub(super) fn count(&mut self, probe: Probe) {
        self.0[match probe {
            Probe::Full => 0,
            Probe::Short => 1,
            Probe::WouldBlock => 2,
            Probe::Unsupported => 3,
            Probe::Interrupted | Probe::Other => 4,
        }] += 1;
    }

    /// Adds these counts to the process's, and the rows found and missed in
    /// the page cache to the started-read counts.
    pub(super) fn publish(&self) {
        for (total, &count) in PROBES.iter().zip(&self.0) {
            if count > 0 {
                total.fetch_add(count, Ordering::Relaxed);
            }
        }
        super::count_cold_rows(self.0[0] as usize, self.0[1..].iter().sum::<u64>() as usize);
    }
}

pub(super) fn snapshot() -> serde_json::Value {
    let count = |index: usize| PROBES[index].load(Ordering::Relaxed);
    serde_json::json!({
        "full":count(0), "short":count(1), "would_block":count(2),
        "unsupported":count(3), "other":count(4),
    })
}
