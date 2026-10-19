//! Sampled host wall timings. No device synchronization is added for profiling.
use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Host work categories, inclusive of nested calls in other categories.
#[derive(Clone, Copy, Debug)]
pub enum StepPhase {
    Scheduling,
    Upload,
    GraphLaunch,
    DeviceWait,
    Sampling,
    /// Host work deferred to a device wait (`CHOPINFER_STEP_OVERLAP=1`).
    Deferred,
    BundleExecute,
    ResultCollect,
    ResultConvert,
    PrefixCapture,
    BundleLog,
}

const PHASES: usize = 11;

thread_local! {
    static TRACING: Cell<bool> = const { Cell::new(false) };
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static TOTALS: Cell<[u64; PHASES]> = const { Cell::new([0; PHASES]) };
    /// When the step's first device wait started and its last one ended.
    static WAITS: Cell<(Option<Instant>, Option<Instant>)> = const { Cell::new((None, None)) };
}

/// Whether the current engine step requested routine trace diagnostics.
#[must_use]
pub fn is_tracing() -> bool {
    TRACING.get()
}

/// Starts accumulating phase totals for one step unless a sampled profile
/// already does; returns whether this call started them.
fn begin_totals() -> bool {
    if ACTIVE.get() {
        return false;
    }
    ACTIVE.set(true);
    TOTALS.set([0; PHASES]);
    WAITS.set((None, None));
    true
}

/// Marks the start of a device wait whose host time goes to other work
/// first, such as deferred jobs run while the device executes.
pub fn device_wait_started(at: Instant) {
    if ACTIVE.get() {
        let (first, last) = WAITS.get();
        WAITS.set((first.or(Some(at)), last));
    }
}

/// Phase totals of the current step so far, and its device-wait window.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PhaseSnapshot {
    pub(crate) totals: [u64; PHASES],
    pub(crate) first_wait_started: Option<Instant>,
    pub(crate) last_wait_ended: Option<Instant>,
}

pub(crate) fn snapshot() -> PhaseSnapshot {
    let (first_wait_started, last_wait_ended) = WAITS.get();
    PhaseSnapshot {
        totals: TOTALS.get(),
        first_wait_started,
        last_wait_ended,
    }
}

/// Accumulates phase totals for the step trace, alongside or without a
/// sampled profile. The guard must stay on the thread where it was created.
pub(crate) struct TraceTotals {
    owns: bool,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl TraceTotals {
    pub(crate) fn begin() -> Self {
        TRACING.set(true);
        Self {
            owns: begin_totals(),
            _thread: std::marker::PhantomData,
        }
    }
}

impl Drop for TraceTotals {
    fn drop(&mut self) {
        TRACING.set(false);
        if self.owns {
            ACTIVE.set(false);
        }
    }
}

/// Times a host scope only while its engine thread is sampling a step.
/// The guard must stay on the thread where it was created.
pub struct PhaseTimer {
    phase: StepPhase,
    started: Option<Instant>,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl PhaseTimer {
    /// No clock reads or allocations unless the step trace is enabled.
    #[must_use]
    pub fn traced(phase: StepPhase) -> Option<Self> {
        TRACING.get().then(|| Self::new(phase))
    }

    #[must_use]
    pub fn new(phase: StepPhase) -> Self {
        Self {
            phase,
            started: ACTIVE.get().then(Instant::now),
            _thread: std::marker::PhantomData,
        }
    }
}

impl Drop for PhaseTimer {
    fn drop(&mut self) {
        if let Some(started) = self.started {
            let ended = Instant::now();
            let micros = (ended - started).as_micros().min(u64::MAX as u128) as u64;
            if matches!(self.phase, StepPhase::DeviceWait) {
                let (first, _) = WAITS.get();
                WAITS.set((first.or(Some(started)), Some(ended)));
            }
            let mut totals = TOTALS.get();
            totals[self.phase as usize] = totals[self.phase as usize].saturating_add(micros);
            TOTALS.set(totals);
        }
    }
}

#[derive(Debug)]
struct Sample {
    step: u64,
    total_micros: u64,
    phases: [u64; PHASES],
}

#[derive(Debug)]
pub(crate) struct StepProfiles {
    every: u64,
    capacity: usize,
    sequence: AtomicU64,
    samples: Mutex<VecDeque<Sample>>,
}

impl Default for StepProfiles {
    fn default() -> Self {
        let every = std::env::var("CHOPINFER_STEP_PROFILE_EVERY")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        let capacity = std::env::var("CHOPINFER_STEP_PROFILE_CAPACITY")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(256);
        Self::new(every, capacity)
    }
}

impl StepProfiles {
    fn new(every: u64, capacity: usize) -> Self {
        Self {
            every,
            capacity: capacity.clamp(1, 4096),
            sequence: AtomicU64::new(0),
            samples: Mutex::new(VecDeque::new()),
        }
    }

    pub(crate) fn begin(self: &Arc<Self>) -> Option<StepTimer> {
        if self.every == 0 {
            return None;
        }
        let step = self.sequence.fetch_add(1, Ordering::Relaxed);
        if !step.is_multiple_of(self.every) || !begin_totals() {
            return None;
        }
        Some(StepTimer {
            profiles: Arc::clone(self),
            step,
            started: Instant::now(),
            _thread: std::marker::PhantomData,
        })
    }

    pub(crate) fn json(&self) -> Option<String> {
        if self.every == 0 {
            return None;
        }
        let samples = self
            .samples
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let rows: Vec<_> = samples
            .iter()
            .map(|sample| {
                serde_json::json!({
                    "step": sample.step,
                    "total_us": sample.total_micros,
                    "host_scheduling_us": sample.phases[0],
                    "upload_us": sample.phases[1],
                    "graph_launch_us": sample.phases[2],
                    "device_wait_us": sample.phases[3],
                    "sampling_us": sample.phases[4],
                    "deferred_us": sample.phases[5],
                    "bundle_execute_us": sample.phases[6],
                    "result_collect_us": sample.phases[7],
                    "result_convert_us": sample.phases[8],
                    "prefix_capture_us": sample.phases[9],
                    "bundle_log_us": sample.phases[10],
                })
            })
            .collect();
        Some(serde_json::json!({"schema": "chopinfer.step_profile.v1", "every": self.every, "capacity": self.capacity, "samples": rows}).to_string())
    }
}

pub(crate) struct StepTimer {
    profiles: Arc<StepProfiles>,
    step: u64,
    started: Instant,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl Drop for StepTimer {
    fn drop(&mut self) {
        ACTIVE.set(false);
        let sample = Sample {
            step: self.step,
            total_micros: self.started.elapsed().as_micros().min(u64::MAX as u128) as u64,
            phases: TOTALS.get(),
        };
        let mut samples = self
            .profiles
            .samples
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if samples.len() == self.profiles.capacity {
            samples.pop_front();
        }
        samples.push_back(sample);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_only_scopes_leave_disabled_and_sampled_steps_alone() {
        assert!(PhaseTimer::traced(StepPhase::BundleExecute).is_none());
        let profiles = Arc::new(StepProfiles::new(1, 1));
        let sample = profiles.begin();
        assert!(PhaseTimer::traced(StepPhase::BundleExecute).is_none());
        assert!(!is_tracing());
        let totals = TraceTotals::begin();
        assert!(is_tracing());
        assert!(PhaseTimer::traced(StepPhase::BundleExecute).is_some());
        drop(totals);
        assert!(!is_tracing());
        assert!(PhaseTimer::traced(StepPhase::BundleExecute).is_none());
        assert!(ACTIVE.get());
        drop(sample);
        assert!(!ACTIVE.get());
    }

    #[test]
    fn trace_totals_mark_the_device_wait_window() {
        device_wait_started(Instant::now());
        assert!(
            WAITS.get().0.is_none(),
            "nothing is recorded outside a step"
        );
        assert!(!is_tracing());
        let totals = TraceTotals::begin();
        assert!(is_tracing());
        let deferred = Instant::now();
        device_wait_started(deferred);
        drop(PhaseTimer::new(StepPhase::DeviceWait));
        let window = snapshot();
        assert_eq!(window.first_wait_started, Some(deferred));
        assert!(
            window
                .last_wait_ended
                .is_some_and(|ended| ended >= deferred)
        );
        drop(totals);
        assert!(!is_tracing());
        assert!(!ACTIVE.get());
    }

    #[test]
    fn disabled_sampling_and_bounded_ring_keep_their_contract() {
        let disabled = Arc::new(StepProfiles::new(0, 2));
        assert!(disabled.begin().is_none());
        assert!(disabled.json().is_none());
        assert!(!ACTIVE.get());
        let profiles = Arc::new(StepProfiles::new(2, 2));
        for step in 0..7 {
            let timer = profiles.begin();
            assert_eq!(timer.is_some(), step % 2 == 0);
            if timer.is_some() {
                TOTALS.set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
            }
            drop(timer);
        }
        let json: serde_json::Value = serde_json::from_str(&profiles.json().unwrap()).unwrap();
        assert_eq!(json["schema"], "chopinfer.step_profile.v1");
        assert_eq!(json["samples"].as_array().unwrap().len(), 2);
        assert_eq!(json["samples"][0]["step"], 4);
        assert_eq!(json["samples"][1]["step"], 6);
        assert_eq!(json["samples"][1]["device_wait_us"], 4);
        assert_eq!(json["samples"][1]["deferred_us"], 6);
        for (field, expected) in [
            ("bundle_execute_us", 7),
            ("result_collect_us", 8),
            ("result_convert_us", 9),
            ("prefix_capture_us", 10),
            ("bundle_log_us", 11),
        ] {
            assert_eq!(json["samples"][1][field], expected);
        }
        assert!(!ACTIVE.get());
    }
}
