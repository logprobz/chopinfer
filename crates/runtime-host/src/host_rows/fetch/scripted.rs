//! A scripted row source for tests: chosen probe classes, garbage left by
//! failed probes, failing whole reads, panics, and a record of every call.
use super::source::{Probe, RowSource};
use std::collections::BTreeMap;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Kind {
    Probe,
    Advise,
    Read,
}

/// One call, in the order calls began.
#[derive(Clone, Debug)]
pub(super) struct Call {
    pub(super) kind: Kind,
    pub(super) thread: std::thread::ThreadId,
    pub(super) fd: i32,
    pub(super) offset: u64,
    pub(super) length: usize,
    /// Where the row's bytes live, for a probe or read.
    pub(super) at: usize,
}

/// The byte a probe that is not [`Probe::Full`] leaves in its row.
pub(super) const GARBAGE: u8 = 0xee;

#[derive(Default)]
pub(super) struct Scripted {
    /// Each offset's probe class; every other probe is full.
    pub(super) classes: BTreeMap<u64, Probe>,
    /// The class of every probe not in `classes`.
    pub(super) otherwise: Option<Probe>,
    /// Whole reads after this many successful ones fail.
    pub(super) fail_after: Option<usize>,
    /// A probe of this offset panics.
    pub(super) panic_at: Option<u64>,
    calls: Mutex<Vec<Call>>,
    reads: AtomicUsize,
    in_flight: AtomicUsize,
}

/// Counts a call in flight until dropped, unwinding included.
struct Flight<'a>(&'a AtomicUsize);
impl Drop for Flight<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Scripted {
    pub(super) fn calls(&self) -> Vec<Call> {
        self.calls.lock().unwrap().clone()
    }

    /// Calls begun and not yet returned.
    pub(super) fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::SeqCst)
    }

    fn begin(&self, kind: Kind, file: &File, offset: u64, row: (usize, usize)) -> Flight<'_> {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        self.calls.lock().unwrap().push(Call {
            kind,
            thread: std::thread::current().id(),
            fd: file.as_raw_fd(),
            offset,
            length: row.1,
            at: row.0,
        });
        Flight(&self.in_flight)
    }
}

impl RowSource for Scripted {
    fn probe(&self, file: &File, row: &mut [u8], offset: u64) -> Probe {
        let _flight = self.begin(
            Kind::Probe,
            file,
            offset,
            (row.as_ptr() as usize, row.len()),
        );
        assert!(self.panic_at != Some(offset), "scripted probe panic");
        let class = self
            .classes
            .get(&offset)
            .copied()
            .or(self.otherwise)
            .unwrap_or(Probe::Full);
        if class == Probe::Full {
            file.read_exact_at(row, offset).unwrap();
        } else {
            row.fill(GARBAGE);
        }
        class
    }

    fn advise(&self, file: &File, offset: u64, length: usize) {
        let _flight = self.begin(Kind::Advise, file, offset, (0, length));
    }

    fn read_whole(&self, file: &File, row: &mut [u8], offset: u64) -> Result<(), String> {
        let _flight = self.begin(Kind::Read, file, offset, (row.as_ptr() as usize, row.len()));
        if self
            .fail_after
            .is_some_and(|after| self.reads.load(Ordering::SeqCst) >= after)
        {
            return Err("scripted read failure".into());
        }
        file.read_exact_at(row, offset)
            .map_err(|error| error.to_string())?;
        self.reads.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
