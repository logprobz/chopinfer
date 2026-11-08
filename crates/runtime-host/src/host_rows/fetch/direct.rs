//! Direct large gathers: scoped workers read a large gather's rows straight
//! into its outputs, and write the twins' copies of each row as it lands,
//! with no reader-pool shares that reply with owned buffers the caller
//! copies into place.
//!
//! The calling thread is worker 0 and starts the others. Work items are
//! [`CHUNK_ROWS`] positions of one plane, each with disjoint slices of the
//! plane's output and of every twin's, handed out from one iterator, so no
//! two workers write one byte. Cold rows follow one schedule:
//!
//! 1. Probe: every worker reads its chunks' rows without waiting for the
//!    device, each into its place. A full row is final, and its twin copies
//!    are written at once. Any other row is cold: the worker advises its
//!    range at once, so its device read starts as it is found, and keeps it.
//!    No worker blocks in this phase.
//! 2. Join: every worker finishes probing, so every cold row of the gather
//!    is advised before any worker blocks.
//! 3. Read: the cold rows are dealt across the workers, each read again
//!    whole from its first byte, which overwrites whatever a probe left;
//!    only then are its twin copies written, so a twin never copies a
//!    partial row.
//!
//! Every worker has finished before [`copy`] returns. A worker's panic, or
//! a whole read's error, fails the gather; a worker that cannot start
//! leaves its share to the others.

use super::PlaneCopy;
use super::source::{Classes, Descriptor, Probe, RowSource};
use std::fs::File;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};

/// Positions one work item covers.
pub(super) const CHUNK_ROWS: usize = 256;
/// The most workers one gather runs on.
const MAX_WORKERS: usize = 24;
/// Threads of the process the workers leave to the engine and driver.
const SPARE_THREADS: usize = 2;

static GATHERS: AtomicU64 = AtomicU64::new(0);
static WORKERS: AtomicU64 = AtomicU64::new(0);

/// The workers of a direct gather in a process that may run `threads`
/// threads: all but two, at least one and at most 24.
pub(super) fn policy(threads: usize) -> usize {
    threads.saturating_sub(SPARE_THREADS).clamp(1, MAX_WORKERS)
}

/// [`policy`] for the threads this process may run on (its affinity and
/// quota), read once.
pub(super) fn workers() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    policy(*THREADS.get_or_init(|| std::thread::available_parallelism().map_or(1, usize::from)))
}

pub(super) fn snapshot() -> serde_json::Value {
    serde_json::json!({
        "gathers":GATHERS.load(Ordering::Relaxed), "workers":WORKERS.load(Ordering::Relaxed),
    })
}

/// Reads every plane's rows into its output and its twins' (`twins[p]`,
/// each as long as plane `p`'s output; none when `twins` is empty) on up
/// to `workers` threads, worker `w` reading plane `p` through
/// `files[p][w % files[p].len()]`. The caller validated the rows.
pub(super) fn copy(
    planes: &mut [PlaneCopy<'_>],
    twins: &mut [Vec<&mut [u8]>],
    files: &[Vec<Arc<File>>],
    workers: usize,
) -> Result<(), String> {
    copy_with(planes, twins, files, workers, &Descriptor, &|_| true)
}

/// One work item.
struct Chunk<'a> {
    plane: usize,
    rows: &'a [u32],
    out: &'a mut [u8],
    twins: Vec<&'a mut [u8]>,
}

/// A row whose probe did not read it whole, and its places.
struct Cold<'a> {
    plane: usize,
    offset: u64,
    out: &'a mut [u8],
    twins: Vec<&'a mut [u8]>,
}

/// What one plane's rows are read from.
struct Plane<'a> {
    base: u64,
    width: usize,
    files: &'a [Arc<File>],
}

/// The join between the probe and read phases, and the cold rows the read
/// phase deals.
struct Join<'a> {
    state: Mutex<JoinState<'a>>,
    all: Condvar,
}

struct JoinState<'a> {
    arrived: usize,
    /// The workers that started, once the last has.
    expected: Option<usize>,
    cold: Vec<Cold<'a>>,
}

impl<'a> Join<'a> {
    fn lock(&self) -> std::sync::MutexGuard<'_, JoinState<'a>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Sets how many workers started.
    fn expect(&self, workers: usize) {
        self.lock().expected = Some(workers);
        self.all.notify_all();
    }

    /// Adds a worker's cold rows and waits for every started worker.
    fn arrive(&self, cold: Vec<Cold<'a>>) {
        let mut state = self.lock();
        state.cold.extend(cold);
        state.arrived += 1;
        self.all.notify_all();
        while state
            .expected
            .is_none_or(|expected| state.arrived < expected)
        {
            state = self.all.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn next_cold(&self) -> Option<Cold<'a>> {
        self.lock().cold.pop()
    }
}

/// [`copy`] from `source`, where `spawn(w)` says whether worker `w` beyond
/// the first may start.
pub(super) fn copy_with(
    planes: &mut [PlaneCopy<'_>],
    twins: &mut [Vec<&mut [u8]>],
    files: &[Vec<Arc<File>>],
    workers: usize,
    source: &dyn RowSource,
    spawn: &dyn Fn(usize) -> bool,
) -> Result<(), String> {
    let meta = planes
        .iter()
        .zip(files)
        .map(|(plane, files)| Plane {
            base: plane.mapping.file_offset(),
            width: plane.width,
            files,
        })
        .collect::<Vec<_>>();
    let mut chunks = Vec::new();
    let mut twins = twins.iter_mut();
    for (index, plane) in planes.iter_mut().enumerate() {
        let width = plane.width;
        let mut twin_chunks = twins
            .next()
            .map(|twins| {
                twins
                    .iter_mut()
                    .map(|twin| twin.chunks_mut(CHUNK_ROWS * width))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        for (rows, out) in plane
            .rows
            .chunks(CHUNK_ROWS)
            .zip(plane.out.chunks_mut(CHUNK_ROWS * width))
        {
            chunks.push(Chunk {
                plane: index,
                rows,
                out,
                twins: twin_chunks
                    .iter_mut()
                    .map(|twin| twin.next().expect("a twin as long as its leader"))
                    .collect(),
            });
        }
    }
    let chunks = Mutex::new(chunks.into_iter());
    let join = Join {
        state: Mutex::new(JoinState {
            arrived: 0,
            expected: None,
            cold: Vec::new(),
        }),
        all: Condvar::new(),
    };
    let failed = AtomicBool::new(false);
    let failure = Mutex::new(None);
    let fail = |error: String| {
        failed.store(true, Ordering::Relaxed);
        failure
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get_or_insert(error);
    };
    let run = |worker: usize| {
        let probed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            probe(worker, &meta, &chunks, &failed, source)
        }));
        // A worker that failed still arrives, so the join never waits for
        // it; the gather fails, and its outputs are never published.
        let cold = probed.unwrap_or_else(|_| {
            fail("a direct mapped gather worker panicked".into());
            Vec::new()
        });
        join.arrive(cold);
        let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            read(worker, &meta, &join, &failed, source)
        }));
        match read {
            Ok(Ok(())) => {}
            Ok(Err(error)) => fail(error),
            Err(_) => fail("a direct mapped gather worker panicked".into()),
        }
    };
    let started = std::thread::scope(|scope| {
        let run = &run;
        let others = (1..workers)
            .filter(|&worker| {
                spawn(worker)
                    && std::thread::Builder::new()
                        .name("mapped-row-direct".into())
                        .spawn_scoped(scope, move || run(worker))
                        .is_ok()
            })
            .count();
        join.expect(others + 1);
        run(0);
        others + 1
    });
    GATHERS.fetch_add(1, Ordering::Relaxed);
    WORKERS.fetch_add(started as u64, Ordering::Relaxed);
    report_engaged(workers, started);
    match failure.into_inner().unwrap_or_else(PoisonError::into_inner) {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Probes every row of the chunks this worker takes, writing full rows'
/// twin copies, advising the others and keeping them.
fn probe<'a>(
    worker: usize,
    planes: &[Plane<'_>],
    chunks: &Mutex<std::vec::IntoIter<Chunk<'a>>>,
    failed: &AtomicBool,
    source: &dyn RowSource,
) -> Vec<Cold<'a>> {
    let mut cold = Vec::new();
    let mut classes = Classes::default();
    while !failed.load(Ordering::Relaxed) {
        let next = chunks.lock().unwrap_or_else(PoisonError::into_inner).next();
        let Some(Chunk {
            plane: index,
            rows,
            out,
            twins,
        }) = next
        else {
            break;
        };
        let plane = &planes[index];
        let file = &*plane.files[worker % plane.files.len()];
        let mut twins = twins
            .into_iter()
            .map(|twin| twin.chunks_mut(plane.width))
            .collect::<Vec<_>>();
        for (&row, out) in rows.iter().zip(out.chunks_mut(plane.width)) {
            // The caller established row * width < the mapping's length.
            let offset = plane.base + u64::from(row) * plane.width as u64;
            let class = source.probe(file, out, offset);
            classes.count(class);
            let mut places = twins
                .iter_mut()
                .map(|twin| twin.next().expect("a twin row for every row"));
            if class == Probe::Full {
                for place in places {
                    place.copy_from_slice(out);
                }
            } else {
                source.advise(file, offset, plane.width);
                cold.push(Cold {
                    plane: index,
                    offset,
                    out,
                    twins: places.by_ref().collect(),
                });
            }
        }
    }
    classes.publish();
    cold
}

/// Reads whole the cold rows dealt to this worker, then their twin copies.
fn read(
    worker: usize,
    planes: &[Plane<'_>],
    join: &Join<'_>,
    failed: &AtomicBool,
    source: &dyn RowSource,
) -> Result<(), String> {
    while !failed.load(Ordering::Relaxed) {
        let Some(Cold {
            plane: index,
            offset,
            out,
            twins,
        }) = join.next_cold()
        else {
            break;
        };
        let plane = &planes[index];
        source.read_whole(&plane.files[worker % plane.files.len()], out, offset)?;
        for twin in twins {
            twin.copy_from_slice(out);
        }
    }
    Ok(())
}

/// Logs the process's first direct gather, so a served run shows the
/// selector engaged.
fn report_engaged(workers: usize, started: usize) {
    static REPORTED: AtomicBool = AtomicBool::new(false);
    if !REPORTED.swap(true, Ordering::Relaxed) {
        runtime_engine::logging::line(
            serde_json::json!({
                "schema":"chopinfer.mapped-row-fetch.v1","phase":"direct-gather",
                "workers":workers,"started":started,
            })
            .to_string(),
        );
    }
}

#[cfg(test)]
#[path = "direct_tests.rs"]
mod tests;
