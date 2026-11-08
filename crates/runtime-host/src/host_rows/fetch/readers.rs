//! Open file descriptions of a mapping's retained descriptor for the readers
//! of large gathers.
//!
//! A mapping keeps one descriptor of its file, a duplicate of the loader's,
//! so every reader of every mapping of one file reads through one open file
//! description. A reader description is a new one: the retained
//! descriptor's procfs magic link (`/proc/self/fd/<fd>`) opened read-only
//! with close-on-exec. The magic link names the already open file, never a
//! path name, so renaming or replacing the checkpoint cannot redirect it;
//! `O_NOFOLLOW` cannot apply to a magic link, and an identity check takes
//! its place: a description is kept only if it is a regular file with the
//! retained descriptor's device and inode. The loader's no-follow open and
//! checkpoint authentication stay with the retained descriptor.
//!
//! Descriptions are bounded: at most [`MAX_PER_MAPPING`] per mapping (the
//! caller asks for no more than its gather's readers), and the process holds
//! at most `min(256, RLIMIT_NOFILE soft / 16)` in all, which leaves 15/16 of
//! its limit to everything else. The first failure to open one ends
//! acquisition for that mapping for good and is logged once per process; a
//! gather always proceeds, its other readers on the retained descriptor.
//! The mapping owns its descriptions and closes them when it drops.

use std::fs::File;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

/// The most reader descriptions one mapping holds.
pub(crate) const MAX_PER_MAPPING: usize = 24;
/// The most reader descriptions the process holds.
const MAX_PROCESS: u64 = 256;
/// The process may hold one reader description per this many descriptors
/// of its open-file limit.
const LIMIT_SHARE: u64 = 16;

static FAILURES: AtomicU64 = AtomicU64::new(0);

/// Reader descriptions held process-wide, against a fixed limit. A slot
/// is taken before its description opens and given back only after it
/// closes, so the descriptions open never exceed the limit.
pub(super) struct Budget {
    limit: usize,
    held: AtomicUsize,
    /// Called with each refund's count as it is given back.
    #[cfg(test)]
    watch: Option<Box<dyn Fn(usize) + Send + Sync>>,
}

impl Budget {
    pub(super) fn new(limit: usize) -> Self {
        Self {
            limit,
            held: AtomicUsize::new(0),
            #[cfg(test)]
            watch: None,
        }
    }

    /// A budget that calls `watch` at each refund.
    #[cfg(test)]
    pub(super) fn watched(limit: usize, watch: impl Fn(usize) + Send + Sync + 'static) -> Self {
        Self {
            watch: Some(Box::new(watch)),
            ..Self::new(limit)
        }
    }

    /// The limit for an open-file soft limit of `soft` descriptors; none
    /// where the limit is unknown.
    pub(super) fn for_soft_limit(soft: Option<u64>) -> Self {
        let limit = soft.map_or(0, |soft| (soft / LIMIT_SHARE).min(MAX_PROCESS));
        Self::new(usize::try_from(limit).unwrap_or(0))
    }

    fn take(&self) -> bool {
        self.held
            .fetch_update(Ordering::Acquire, Ordering::Acquire, |held| {
                (held < self.limit).then_some(held + 1)
            })
            .is_ok()
    }

    fn give(&self, count: usize) {
        #[cfg(test)]
        if let Some(watch) = &self.watch {
            watch(count);
        }
        self.held.fetch_sub(count, Ordering::Release);
    }

    pub(super) fn held(&self) -> usize {
        self.held.load(Ordering::Relaxed)
    }

    pub(super) fn limit(&self) -> usize {
        self.limit
    }
}

/// The process's budget, sized from its open-file limit at first use.
pub(super) fn budget() -> &'static Arc<Budget> {
    static BUDGET: OnceLock<Arc<Budget>> = OnceLock::new();
    BUDGET.get_or_init(|| {
        Arc::new(Budget::for_soft_limit(
            crate::host_rows::open_files_soft_limit_v1(),
        ))
    })
}

pub(super) fn failures() -> u64 {
    FAILURES.load(Ordering::Relaxed)
}

/// Opens a description of a retained descriptor.
pub(super) type Opener<'a> = dyn Fn(&File) -> io::Result<File> + 'a;

/// One mapping's reader descriptions.
#[derive(Default)]
pub(crate) struct ReaderFiles {
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    files: Vec<Arc<File>>,
    /// Set by the first failure: no description is opened again.
    stopped: bool,
    /// The budget the descriptions were taken from.
    budget: Option<Arc<Budget>>,
}

impl ReaderFiles {
    /// Up to `wanted` reader descriptions of `retained`, the mapping's
    /// descriptor, opened through procfs within the process budget.
    pub(super) fn get(&self, retained: &File, wanted: usize) -> Vec<Arc<File>> {
        self.get_with(retained, wanted, &reopen, budget())
    }

    /// [`Self::get`] with another opener and budget. Opens descriptions
    /// until `wanted` are held or the first failure, which is cached, and
    /// answers the first `wanted` held, fewer after a failure, none when the
    /// first open failed.
    pub(super) fn get_with(
        &self,
        retained: &File,
        wanted: usize,
        open: &Opener<'_>,
        budget: &Arc<Budget>,
    ) -> Vec<Arc<File>> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        while state.files.len() < wanted && !state.stopped {
            let budget = Arc::clone(state.budget.get_or_insert_with(|| Arc::clone(budget)));
            let opened = if budget.take() {
                open(retained)
                    .and_then(|file| same_file(retained, &file).map(|()| file))
                    .inspect_err(|_| budget.give(1))
                    .map_err(|error| error.to_string())
            } else {
                Err(format!(
                    "the process holds its budget of {} reader descriptions",
                    budget.limit()
                ))
            };
            match opened {
                Ok(file) => state.files.push(Arc::new(file)),
                Err(reason) => {
                    state.stopped = true;
                    FAILURES.fetch_add(1, Ordering::Relaxed);
                    report_failure(state.files.len(), &reason);
                }
            }
        }
        let files = state.files[..wanted.min(state.files.len())].to_vec();
        if !files.is_empty() {
            report_engaged(files.len(), state.budget.as_ref().map_or(0, |b| b.limit()));
        }
        files
    }
}

impl Drop for ReaderFiles {
    /// Closes the descriptions, then gives their slots back, so a mapping
    /// acquiring descriptions meanwhile never takes a slot whose
    /// description is still open. A reader thread drops its description
    /// before it replies, so these references are the last ones.
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        let files = std::mem::take(&mut state.files);
        let closed = files.len();
        drop(files);
        if let Some(budget) = &state.budget {
            budget.give(closed);
        }
    }
}

/// `retained` opened again through its procfs magic link, read-only with
/// close-on-exec (the standard library's default).
fn reopen(retained: &File) -> io::Result<File> {
    use std::os::fd::AsRawFd;
    File::open(format!("/proc/self/fd/{}", retained.as_raw_fd()))
}

/// Refuses a description that is not a regular file with the retained
/// descriptor's device and inode.
fn same_file(retained: &File, reopened: &File) -> io::Result<()> {
    use std::os::unix::fs::MetadataExt;
    let (retained, reopened) = (retained.metadata()?, reopened.metadata()?);
    if reopened.is_file() && reopened.dev() == retained.dev() && reopened.ino() == retained.ino() {
        Ok(())
    } else {
        Err(io::Error::other(
            "the reopened description names another file",
        ))
    }
}

/// Logs the first gather that reads through reader descriptions, so a
/// served run shows the selector engaged.
fn report_engaged(descriptions: usize, budget: usize) {
    static REPORTED: AtomicBool = AtomicBool::new(false);
    if !REPORTED.swap(true, Ordering::Relaxed) {
        runtime_engine::logging::line(
            serde_json::json!({
                "schema":"chopinfer.mapped-row-fetch.v1","phase":"reader-files",
                "descriptions":descriptions,"budget":budget,
            })
            .to_string(),
        );
    }
}

/// Logs the process's first failure to open a reader description.
fn report_failure(descriptions: usize, reason: &str) {
    static REPORTED: AtomicBool = AtomicBool::new(false);
    if !REPORTED.swap(true, Ordering::Relaxed) {
        runtime_engine::logging::line(
            serde_json::json!({
                "schema":"chopinfer.mapped-row-fetch.v1","phase":"reader-files",
                "descriptions":descriptions,"reason":reason,
            })
            .to_string(),
        );
    }
}

#[cfg(test)]
#[path = "reader_tests.rs"]
pub(super) mod tests;
