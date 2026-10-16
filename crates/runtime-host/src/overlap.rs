//! Host work that runs while the device executes the next step.
//!
//! With `CHOPINFER_STEP_OVERLAP=1` the serving loop hands work that decides
//! nothing about the next step (delivering a step's events to their streams)
//! to this queue instead of doing it between steps. The host services run the
//! queue when the engine thread next blocks on the device after launching
//! work, so it fills time the thread would otherwise spend waiting. Whatever
//! no device wait ran is run by the serving loop itself after the next step
//! returns, so a job waits at most one step.
//!
//! The queue belongs to the engine thread: jobs run on the thread that queued
//! them, in the order they were queued.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;

/// Whether the serving loop defers event delivery; off unless set to `1`.
pub const STEP_OVERLAP: &str = "CHOPINFER_STEP_OVERLAP";

pub(crate) fn step_overlap_from_environment() -> bool {
    std::env::var_os(STEP_OVERLAP).is_some_and(|value| value == "1")
}

type Job = Box<dyn FnOnce()>;

thread_local! {
    static QUEUE: RefCell<VecDeque<Job>> = const { RefCell::new(VecDeque::new()) };
    /// Device work was launched since the queue last ran.
    static LAUNCHED: Cell<bool> = const { Cell::new(false) };
    static RUNNING: Cell<bool> = const { Cell::new(false) };
}

/// Queues `job` to run during a later device wait on this thread.
pub(crate) fn defer(job: impl FnOnce() + 'static) {
    QUEUE.with_borrow_mut(|queue| queue.push_back(Box::new(job)));
}

/// Records that this thread launched device work outside a graph capture.
pub(crate) fn note_launch() {
    LAUNCHED.set(true);
}

/// Called before this thread blocks on the device. Runs the queue when work
/// was launched since it last ran, so the device is busy while it runs.
pub(crate) fn device_wait() {
    if LAUNCHED.replace(false) && QUEUE.with_borrow(|queue| !queue.is_empty()) {
        // The device is busy from here, whatever the host does first.
        runtime_engine::profile::device_wait_started(std::time::Instant::now());
        run_pending();
    }
}

/// Runs every queued job in order and returns how many ran.
pub(crate) fn run_pending() -> usize {
    if RUNNING.replace(true) {
        return 0;
    }
    let _profile =
        runtime_engine::profile::PhaseTimer::new(runtime_engine::profile::StepPhase::Deferred);
    let mut ran = 0;
    while let Some(job) = QUEUE.with_borrow_mut(VecDeque::pop_front) {
        job();
        ran += 1;
    }
    RUNNING.set(false);
    ran
}

#[cfg(test)]
pub(crate) fn pending() -> usize {
    QUEUE.with_borrow(VecDeque::len)
}

/// Runs the queue when dropped, so no path out of the serving loop leaves
/// events undelivered.
pub(crate) struct FlushOnDrop;

impl Drop for FlushOnDrop {
    fn drop(&mut self) {
        run_pending();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    #[test]
    fn jobs_run_in_order_at_the_first_wait_after_a_launch() {
        let log = Rc::new(RefCell::new(Vec::new()));
        for index in 0..3 {
            let log = Rc::clone(&log);
            defer(move || log.borrow_mut().push(index));
        }
        device_wait();
        assert!(log.borrow().is_empty(), "no launch, so the device is idle");
        note_launch();
        device_wait();
        assert_eq!(*log.borrow(), [0, 1, 2]);
        let later = Rc::clone(&log);
        defer(move || later.borrow_mut().push(3));
        device_wait();
        assert_eq!(pending(), 1, "the launch was consumed by the last wait");
        drop(FlushOnDrop);
        assert_eq!(*log.borrow(), [0, 1, 2, 3]);
        assert_eq!(pending(), 0);
    }

    #[test]
    fn a_job_that_waits_on_the_device_does_not_reenter_the_queue() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let inner = Rc::clone(&log);
        defer(move || {
            note_launch();
            device_wait();
            inner.borrow_mut().push("outer");
        });
        let second = Rc::clone(&log);
        defer(move || second.borrow_mut().push("second"));
        assert_eq!(run_pending(), 2);
        assert_eq!(*log.borrow(), ["outer", "second"]);
        LAUNCHED.set(false);
    }
}
