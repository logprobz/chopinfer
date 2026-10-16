//! Rank placement and bounded execution for lab collectives.

use crate::{LabResult, Session};
use kernel_core::contract::MAX_COMM_RANKS;

// Lab launches pack one address per rank into the driver's pointer record.
const _: () = assert!(runtime_cuda::DEVICE_POINTER_ARRAY_SLOTS == MAX_COMM_RANKS);
use runtime_cuda::{GraphExec, Stream};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Write;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

pub(crate) fn parse_ordinals(input: &str) -> LabResult<Vec<i32>> {
    let ordinals = input
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<i32>()
                .map_err(|_| "invalid GPU ordinal".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if ordinals.len() > MAX_COMM_RANKS || ordinals.iter().any(|&ordinal| ordinal < 0) {
        return Err(format!(
            "select one to {MAX_COMM_RANKS} distinct nonnegative GPU ordinals"
        ));
    }
    if ordinals
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != ordinals.len()
    {
        return Err("GPU ordinals must be distinct".into());
    }
    Ok(ordinals)
}

#[derive(Default)]
struct Execution {
    /// Inside [`preflight`]: launches are prepared and checked, not submitted.
    checking: bool,
    streams: Vec<Stream>,
    submissions: Option<Arc<Mutex<BTreeMap<usize, String>>>>,
}

thread_local! {
    static EXECUTION: RefCell<Execution> = RefCell::default();
}

/// Runs `work` with every lab launch (`crate::launch`) resolved, checked and
/// prepared but none submitted, so a sequence that would be refused partway
/// is refused before its first launch: a launched collective whose peers
/// never launch waits forever. Work other than launches runs as usual.
pub fn preflight<T>(work: impl FnOnce() -> LabResult<T>) -> LabResult<T> {
    let outer = EXECUTION.with(|state| std::mem::replace(&mut state.borrow_mut().checking, true));
    let result = work();
    EXECUTION.with(|state| state.borrow_mut().checking = outer);
    result
}

/// Runs `work` for ranks `0..ranks` in order after a [`preflight`] of all of
/// them, so a rank-by-rank sequence of collective calls is refused before
/// its first launch if any of its launches would be. `work` runs twice and
/// must only launch.
pub fn each_rank(ranks: usize, work: impl Fn(usize) -> LabResult<()>) -> LabResult<()> {
    preflight(|| (0..ranks).try_for_each(&work))?;
    (0..ranks).try_for_each(work)
}

/// Whether launches are only being checked ([`preflight`]).
pub(crate) fn checking() -> bool {
    EXECUTION.with(|state| state.borrow().checking)
}

/// One captured graph per rank. The caller must prepare modules and buffers
/// before capture, and the work must only launch.
pub struct RankReplay<F> {
    graphs: Vec<GraphExec>,
    /// Keep captured resources and their borrows alive through replay.
    _work: F,
}

impl<F: Fn(usize, &Session) -> LabResult<()>> RankReplay<F> {
    pub fn new(sessions: &[Session], work: F) -> LabResult<Self> {
        let graphs = sessions
            .iter()
            .enumerate()
            .map(|(rank, session)| session.capture(|_| work(rank, session)))
            .collect::<LabResult<Vec<_>>>()?;
        Ok(Self {
            graphs,
            _work: work,
        })
    }

    pub fn launch(&self, sessions: &[Session]) -> LabResult<()> {
        for (rank, session) in sessions.iter().enumerate() {
            self.graphs[rank]
                .launch(&session.stream)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

/// A deadline covers setup, launches, synchronization and CUDA destruction.
/// Unwinding is insufficient: driver destructors can wait on stalled kernels.
pub(crate) struct Watchdog {
    cancel: Option<mpsc::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
    submissions: Arc<Mutex<BTreeMap<usize, String>>>,
}

impl Watchdog {
    pub(crate) fn start(ordinals: &[i32], timeout: Duration) -> LabResult<Self> {
        let (cancel, receiver) = mpsc::channel();
        let submissions = Arc::new(Mutex::new(BTreeMap::new()));
        let watched = Arc::clone(&submissions);
        let test = std::thread::current()
            .name()
            .unwrap_or("unnamed lab test")
            .to_owned();
        let ordinals = ordinals.to_vec();
        let thread = std::thread::Builder::new().name("kernel-lab-watchdog".into()).spawn(move || {
            if receiver.recv_timeout(timeout) != Err(mpsc::RecvTimeoutError::Timeout) {
                return;
            }
            let mut diagnostic = format!("kernel-lab watchdog: test {test} exceeded {timeout:?}; rank devices {ordinals:?}. Possible collective flag wait. Exiting test process with code 124 without CUDA cleanup.\n");
            // This thread never calls CUDA, including when a driver call is stuck.
            if let Ok(submissions) = watched.try_lock() {
                for (rank, submission) in submissions.iter() {
                    diagnostic.push_str(&format!("rank {rank}: last submitted {submission}\n"));
                }
            }
            diagnostic.push_str("These are host submissions, not sampled device flag values or confirmed wait sites.\n");
            // Bypass libtest's output capture, which process::exit cannot flush.
            let _ = std::io::stderr().write_all(diagnostic.as_bytes());
            std::process::exit(124);
        }).map_err(|e| format!("cannot start kernel-lab watchdog: {e}"))?;
        Ok(Self {
            cancel: Some(cancel),
            thread: Some(thread),
            submissions,
        })
    }

    pub(crate) fn attach(&self, sessions: &[Session]) {
        EXECUTION.with(|state| {
            *state.borrow_mut() = Execution {
                checking: false,
                streams: sessions.iter().map(|s| s.stream.clone()).collect(),
                submissions: Some(Arc::clone(&self.submissions)),
            }
        });
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        EXECUTION.with(|state| *state.borrow_mut() = Execution::default());
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

pub(crate) fn timeout() -> LabResult<Duration> {
    let seconds = match std::env::var("CHOPINFER_KERNEL_TIMEOUT_SECS") {
        Ok(value) => value
            .parse::<u64>()
            .map_err(|_| "invalid CHOPINFER_KERNEL_TIMEOUT_SECS")?,
        Err(std::env::VarError::NotPresent) => 300,
        Err(error) => return Err(error.to_string()),
    };
    if seconds == 0 {
        return Err("CHOPINFER_KERNEL_TIMEOUT_SECS must be positive".into());
    }
    Ok(Duration::from_secs(seconds))
}

pub(crate) fn record_launch(stream: &Stream, details: impl FnOnce() -> String) -> LabResult<()> {
    EXECUTION.with(|state| {
        let state = state.borrow();
        let Some(submissions) = &state.submissions else {
            return Ok(());
        };
        for (rank, known) in state.streams.iter().enumerate() {
            let same = known
                .with_raw_handle(|a| stream.with_raw_handle(|b| a == b))
                .map_err(|e| e.to_string())?
                .map_err(|e| e.to_string())?;
            if same {
                submissions
                    .lock()
                    .map_err(|e| e.to_string())?
                    .insert(rank, details());
                break;
            }
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_ordinals_are_refused() {
        for input in ["0,0", "3,1,3,1", "0,0,1,1,2,2,3,3"] {
            assert!(parse_ordinals(input).is_err(), "{input}");
        }
    }

    #[test]
    fn distinct_rank_order_is_preserved() {
        for input in ["0,1", "2,3", "0,1,2,3", "0,1,2,3,4,5,6,7", "3,1"] {
            let ordinals = parse_ordinals(input).unwrap();
            assert_eq!(
                ordinals
                    .iter()
                    .map(i32::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
                input
            );
        }
        for input in ["", "0,", "-1", "a", "0,1,2,3,4,5,6,7,8"] {
            assert!(parse_ordinals(input).is_err(), "{input}");
        }
    }

    #[test]
    fn watchdog_child() {
        if std::env::var("CHOPINFER_LAB_WATCHDOG_CHILD").as_deref() != Ok("1") {
            return;
        }
        let watchdog = Watchdog::start(&[0, 1], Duration::from_millis(50)).unwrap();
        watchdog
            .submissions
            .lock()
            .unwrap()
            .insert(1, "CPU test flag dependency".into());
        std::thread::sleep(Duration::from_secs(10));
        panic!("watchdog did not terminate the stalled process");
    }

    #[test]
    fn watchdog_terminates_stalled_process_without_cuda() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "topology::tests::watchdog_child", "--nocapture"])
            .env("CHOPINFER_LAB_WATCHDOG_CHILD", "1")
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(124));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("rank devices [0, 1]"), "{stderr}");
        assert!(
            stderr.contains("rank 1: last submitted CPU test flag dependency"),
            "{stderr}"
        );
    }

    #[test]
    fn watchdog_cancels_on_normal_completion() {
        drop(Watchdog::start(&[0, 1], Duration::from_secs(10)).unwrap());
    }
}
