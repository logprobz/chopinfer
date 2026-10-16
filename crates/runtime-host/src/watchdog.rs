//! Observes the generation worker's progress from outside its thread.
//!
//! The worker runs each engine step synchronously, and a step that never
//! returns (a device that stopped answering, a lost collective peer) leaves
//! the worker unable to report anything. The worker marks when it starts
//! each piece of work and what kind it is, and a separate thread compares
//! the time since that mark with the allowance for that kind. Past it, and
//! only if that same work is still running, the worker counts as failed:
//! readiness reports it terminated, admission refuses, and the process exits
//! so a supervisor can restart it. The exit deadline is set before anything
//! is logged, logging never waits on stderr, and a panic in the watchdog
//! still exits. Nothing here tries to recover the device, join the stuck
//! thread, or free what it may still be using.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use crate::signals::{exit_now, try_write_stderr};

/// How long one piece of worker work may run before the worker is treated
/// as stalled, by what the work is.
///
/// The defaults are several times the longest steps traced on the served
/// models (2026-10 serving runs with step traces):
/// - Decode and verify steps took at most 0.39 s; a shape the startup warmup
///   did not capture is captured at first use, and the slowest capture seen
///   took 11.8 s inside one decode step. A step that may need prompt work
///   no request has started yet is not a decode step.
/// - Prompt steps (prefill chunks, media encoding) took at most 10.1 s: the
///   final chunk of a 32K prompt on a model whose last chunk waits for the
///   whole prompt. A 128K prompt there is estimated at 25 to 40 s. One large
///   image encode took 9.5 s (33 s in debug variants).
/// - Startup warmup steps capture graphs; the slowest single step took
///   11.8 s and the slowest instrumented warmup wave 58.7 s.
/// - Building the engine (bundle create, module and weight loading) and
///   tearing it down share the lifecycle allowance. Whole startups, which
///   also include preparation and warmup, took at most 409 s over 4,016
///   logged starts (p99 238 s).
///
/// A step is classified when the engine has formed its batch: prompt work
/// when the batch prefills, a replay after preemption included.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkerWatchdogV1 {
    decode: Duration,
    prompt: Duration,
    warmup: Duration,
    lifecycle: Duration,
    /// After declaring the worker stalled, how long the process has to exit
    /// on its own (the transport sees the terminated readiness and returns)
    /// before the watchdog exits it.
    exit_grace: Option<Duration>,
}

impl WorkerWatchdogV1 {
    pub const DEFAULT_DECODE: Duration = Duration::from_secs(120);
    pub const DEFAULT_PROMPT: Duration = Duration::from_secs(600);
    pub const DEFAULT_WARMUP: Duration = Duration::from_secs(600);
    pub const DEFAULT_LIFECYCLE: Duration = Duration::from_secs(1800);
    pub const DEFAULT_EXIT_GRACE: Duration = Duration::from_secs(10);

    /// The served defaults, exiting the process after a stall.
    #[must_use]
    pub const fn serving() -> Self {
        Self {
            decode: Self::DEFAULT_DECODE,
            prompt: Self::DEFAULT_PROMPT,
            warmup: Self::DEFAULT_WARMUP,
            lifecycle: Self::DEFAULT_LIFECYCLE,
            exit_grace: Some(Self::DEFAULT_EXIT_GRACE),
        }
    }

    /// Allowances for decode, prompt and warmup work. A zero allowance is
    /// raised to one millisecond.
    #[must_use]
    pub fn with_allowances(mut self, decode: Duration, prompt: Duration, warmup: Duration) -> Self {
        let floor = Duration::from_millis(1);
        self.decode = decode.max(floor);
        self.prompt = prompt.max(floor);
        self.warmup = warmup.max(floor);
        self
    }

    /// The allowance for building the engine and for tearing it down. A zero
    /// allowance is raised to one millisecond.
    #[must_use]
    pub fn with_lifecycle_allowance(mut self, lifecycle: Duration) -> Self {
        self.lifecycle = lifecycle.max(Duration::from_millis(1));
        self
    }

    /// Whether, and how long after a stall, the watchdog exits the process.
    /// Without it, a stall only fails readiness and admission.
    #[must_use]
    pub const fn with_exit_grace(mut self, exit_grace: Option<Duration>) -> Self {
        self.exit_grace = exit_grace;
        self
    }

    const fn allowance(self, phase: WorkerPhase) -> Duration {
        match phase {
            WorkerPhase::Decode => self.decode,
            WorkerPhase::Prompt => self.prompt,
            WorkerPhase::Warmup => self.warmup,
            WorkerPhase::Startup | WorkerPhase::Shutdown => self.lifecycle,
        }
    }

    /// How often the watchdog looks: often enough to notice a stall within
    /// a tenth of the shortest allowance, and at least once a second.
    fn poll(self) -> Duration {
        (self
            .decode
            .min(self.prompt)
            .min(self.warmup)
            .min(self.lifecycle)
            / 10)
            .clamp(Duration::from_millis(1), Duration::from_secs(1))
    }
}

/// What the worker is doing since its last mark.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WorkerPhase {
    /// Building the engine: bundle create, module and weight loading.
    Startup,
    /// Host bookkeeping, and steps that only decode or verify.
    Decode,
    /// Steps that prefill a prompt or encode its media.
    Prompt,
    /// Startup warmup, which captures graphs.
    Warmup,
    /// Cancelling, draining and releasing the engine.
    Shutdown,
}

impl WorkerPhase {
    const fn name(self) -> &'static str {
        match self {
            Self::Startup => "startup",
            Self::Decode => "decode",
            Self::Prompt => "prompt",
            Self::Warmup => "warmup",
            Self::Shutdown => "shutdown",
        }
    }
}

/// One piece of worker work: a number that changes with every mark, idle
/// and end, what kind it is, and when it began. `phase` is `None` while the
/// worker waits for commands or after it ended.
#[derive(Clone, Copy, Debug)]
struct Work {
    id: u64,
    phase: Option<WorkerPhase>,
    started: Instant,
    /// Between the two startup warmup commands, every step is warmup work.
    warming: bool,
}

/// Work that has run past its allowance, as one coherent observation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Overrun {
    id: u64,
    phase: WorkerPhase,
    elapsed: Duration,
    allowance: Duration,
}

/// The worker's current work, shared with the watchdog.
pub(crate) struct WorkerProgress {
    work: Mutex<Work>,
    stalled: AtomicBool,
    ended: AtomicBool,
}

impl WorkerProgress {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            work: Mutex::new(Work {
                id: 0,
                phase: None,
                started: Instant::now(),
                warming: false,
            }),
            stalled: AtomicBool::new(false),
            ended: AtomicBool::new(false),
        })
    }

    fn work(&self) -> MutexGuard<'_, Work> {
        self.work
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn replace(&self, phase: Option<WorkerPhase>) {
        let mut work = self.work();
        work.id = work.id.wrapping_add(1);
        work.phase = phase;
        work.started = Instant::now();
    }

    /// The worker starts work of this kind now. Steps between the startup
    /// warmup commands are warmup work whatever they are asked as.
    pub(crate) fn mark(&self, phase: WorkerPhase) {
        let mut work = self.work();
        let phase = match phase {
            WorkerPhase::Decode | WorkerPhase::Prompt if work.warming => WorkerPhase::Warmup,
            phase => phase,
        };
        work.id = work.id.wrapping_add(1);
        work.phase = Some(phase);
        work.started = Instant::now();
    }

    /// The worker has formed a step's batch and now executes it.
    pub(crate) fn batch(&self, prefill: bool) {
        self.mark(if prefill {
            WorkerPhase::Prompt
        } else {
            WorkerPhase::Decode
        });
    }

    /// Startup warmup begins (true) or ends (false).
    pub(crate) fn warming(&self, warming: bool) {
        self.work().warming = warming;
    }

    /// The worker waits for a command, which may take any time.
    pub(crate) fn idle(&self) {
        self.replace(None);
    }

    /// The worker thread has returned, whether or not it stalled before.
    pub(crate) fn end(&self) {
        self.replace(None);
        self.ended.store(true, Ordering::Release);
    }

    pub(crate) fn has_ended(&self) -> bool {
        self.ended.load(Ordering::Acquire)
    }

    /// Whether the watchdog declared the worker stalled. A stalled worker is
    /// never joined, and its failure is final.
    pub(crate) fn is_stalled(&self) -> bool {
        self.stalled.load(Ordering::Acquire)
    }

    /// The current work if it has run past its allowance at `now`.
    pub(crate) fn overrun(&self, watchdog: WorkerWatchdogV1, now: Instant) -> Option<Overrun> {
        let work = *self.work();
        let phase = work.phase?;
        let elapsed = now.saturating_duration_since(work.started);
        let allowance = watchdog.allowance(phase);
        (elapsed > allowance).then_some(Overrun {
            id: work.id,
            phase,
            elapsed,
            allowance,
        })
    }

    /// Declares the worker stalled if the overrun work is still the current
    /// work; false when it finished, went idle or ended in the meantime.
    pub(crate) fn claim(&self, overrun: &Overrun) -> bool {
        let work = self.work();
        if work.id != overrun.id || work.phase.is_none() || self.has_ended() {
            return false;
        }
        self.stalled.store(true, Ordering::Release);
        true
    }
}

/// Watches `progress` until the worker ends or a piece of work overruns its
/// allowance. On an overrun, the exit deadline is set first, then `fail`
/// closes admission and fails readiness, then a message is written if stderr
/// can take it without blocking. A panic here also exits, when exiting is
/// configured.
pub(crate) fn spawn(
    watchdog: WorkerWatchdogV1,
    progress: Arc<WorkerProgress>,
    fail: impl FnOnce() + Send + 'static,
) -> std::io::Result<()> {
    thread::Builder::new()
        .name("bundle-engine-watchdog".to_owned())
        .spawn(move || {
            let watched = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                watch(watchdog, &progress, fail);
            }));
            if watched.is_err() {
                progress.stalled.store(true, Ordering::Release);
                if watchdog.exit_grace.is_some() {
                    try_write_stderr("fatal: the generation watchdog failed; exiting\n");
                    exit_now(1);
                }
            }
        })
        .map(drop)
}

fn watch(watchdog: WorkerWatchdogV1, progress: &WorkerProgress, fail: impl FnOnce()) {
    let poll = watchdog.poll();
    let overrun = loop {
        thread::sleep(poll);
        if progress.has_ended() {
            return;
        }
        if let Some(overrun) = progress.overrun(watchdog, Instant::now())
            && progress.claim(&overrun)
        {
            break overrun;
        }
    };
    // The deadline does not depend on anything below finishing.
    if let Some(grace) = watchdog.exit_grace {
        let deadline = thread::Builder::new()
            .name("bundle-engine-exit".to_owned())
            .spawn(move || {
                thread::sleep(grace);
                exit_now(1);
            });
        if deadline.is_err() {
            exit_now(1);
        }
    }
    fail();
    try_write_stderr(&format!(
        "fatal: the generation worker made no progress for {:.1}s in {} work (allowance \
         {:.1}s); new requests are refused and the process exits for supervisor restart\n",
        overrun.elapsed.as_secs_f64(),
        overrun.phase.name(),
        overrun.allowance.as_secs_f64()
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn watchdog() -> WorkerWatchdogV1 {
        WorkerWatchdogV1::serving()
            .with_allowances(
                Duration::from_secs(1),
                Duration::from_secs(10),
                Duration::from_secs(20),
            )
            .with_lifecycle_allowance(Duration::from_secs(30))
            .with_exit_grace(None)
    }

    fn started(progress: &WorkerProgress) -> Instant {
        progress.work().started
    }

    /// An observation of overrun work claims a stall only while that same
    /// work runs: not after it finished, went idle, was replaced or ended.
    #[test]
    fn an_overrun_is_claimed_only_while_the_same_work_runs() {
        let progress = WorkerProgress::new();
        progress.mark(WorkerPhase::Decode);
        let late = started(&progress) + Duration::from_secs(2);
        let overrun = progress.overrun(watchdog(), late).unwrap();
        // The step returned and the worker went idle before the claim.
        progress.idle();
        assert!(!progress.claim(&overrun));
        assert!(progress.overrun(watchdog(), late).is_none());
        // The worker began another step before the claim.
        progress.mark(WorkerPhase::Decode);
        let overrun = progress.overrun(watchdog(), late).unwrap();
        progress.mark(WorkerPhase::Decode);
        assert!(!progress.claim(&overrun));
        // The worker ended before the claim.
        let overrun = progress
            .overrun(watchdog(), started(&progress) + Duration::from_secs(2))
            .unwrap();
        progress.end();
        assert!(!progress.claim(&overrun));
        assert!(!progress.is_stalled());
        // Still running: the claim holds.
        let progress = WorkerProgress::new();
        progress.mark(WorkerPhase::Decode);
        let overrun = progress
            .overrun(watchdog(), started(&progress) + Duration::from_secs(2))
            .unwrap();
        assert!(progress.claim(&overrun));
        assert!(progress.is_stalled());
    }

    /// A phase is observed with its own start: a decode step that followed
    /// a long prompt step is judged by its own start and allowance.
    #[test]
    fn phase_and_start_are_observed_together() {
        let progress = WorkerProgress::new();
        progress.mark(WorkerPhase::Prompt);
        let prompt = started(&progress);
        // Past the decode allowance from the prompt's start.
        let now = prompt + Duration::from_millis(1500);
        assert!(progress.overrun(watchdog(), now).is_none());
        progress.mark(WorkerPhase::Decode);
        let decode = started(&progress);
        match progress.overrun(watchdog(), now) {
            None => assert!(now.saturating_duration_since(decode) <= Duration::from_secs(1)),
            Some(overrun) => {
                assert_eq!(overrun.phase, WorkerPhase::Decode);
                assert_eq!(overrun.allowance, Duration::from_secs(1));
                assert_eq!(overrun.elapsed, now.saturating_duration_since(decode));
            }
        }
        assert!(
            progress
                .overrun(watchdog(), decode + Duration::from_millis(500))
                .is_none()
        );
    }

    /// Work that ends exactly at its allowance is in budget; a microsecond
    /// later it overran.
    #[test]
    fn the_allowance_itself_is_in_budget() {
        let progress = WorkerProgress::new();
        for phase in [
            WorkerPhase::Startup,
            WorkerPhase::Decode,
            WorkerPhase::Prompt,
            WorkerPhase::Warmup,
            WorkerPhase::Shutdown,
        ] {
            progress.mark(phase);
            let limit = started(&progress) + watchdog().allowance(phase);
            assert!(progress.overrun(watchdog(), limit).is_none(), "{phase:?}");
            let overrun = progress
                .overrun(watchdog(), limit + Duration::from_micros(1))
                .unwrap();
            assert_eq!(overrun.phase, phase);
        }
    }

    /// Between the startup warmup commands every step is warmup work, and a
    /// formed batch is prompt or decode work by whether it prefills.
    #[test]
    fn steps_are_classified_by_warmup_and_by_their_batch() {
        let progress = WorkerProgress::new();
        let phase = |progress: &WorkerProgress| progress.work().phase;
        progress.batch(true);
        assert_eq!(phase(&progress), Some(WorkerPhase::Prompt));
        progress.batch(false);
        assert_eq!(phase(&progress), Some(WorkerPhase::Decode));
        progress.warming(true);
        progress.batch(true);
        assert_eq!(phase(&progress), Some(WorkerPhase::Warmup));
        progress.mark(WorkerPhase::Decode);
        assert_eq!(phase(&progress), Some(WorkerPhase::Warmup));
        progress.mark(WorkerPhase::Shutdown);
        assert_eq!(phase(&progress), Some(WorkerPhase::Shutdown));
        progress.warming(false);
        progress.batch(false);
        assert_eq!(phase(&progress), Some(WorkerPhase::Decode));
    }
}
