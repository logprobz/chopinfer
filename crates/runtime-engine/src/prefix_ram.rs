//! The engine's side of the RAM prefix tier: whether it runs, the
//! demotions it starts after each step's captures, and the promotions it
//! submits for admission.
//!
//! A demotion copies a public GPU anchor near the head of the eviction order
//! into the host tier ahead of its eviction, so the eviction loses nothing.
//! Demotions start toward a lead of L freeable pages:
//!
//! - L is the larger of twice the most pages eviction freed in one of the
//!   last 16 steps, and the largest deferred promotion plus one prefill
//!   chunk and one snapshot;
//! - at most one demotion is outstanding, from its plan to its confirmed
//!   completion, so its whole-anchor lease and its host pages stay bounded
//!   by one anchor;
//! - a demotion of A starts only when free pages plus the pages evicting
//!   every other evictable anchor frees cover F plus A's own pages
//!   ([`Scheduler::prefix_demotion_headroom`]). F is the measured per-step
//!   bound of design amendment A8: twice the most pages allocated gross in
//!   one of the last 16 steps, plus the largest deferred promotion and the
//!   pages swapped requests may allocate next step, which the scheduler caps
//!   by its worst case ([`Scheduler::step_allocation_bound`]).
//!
//! Promotions start as soon as admission plans them, at the step's start
//! for any left over and after each admission pass.
//!
//! An allocation larger than F can still meet a pool whose only remaining
//! victim is the anchor under demotion. Then the engine cancels the
//! demotion and waits for the cancel to settle, which drains at most the
//! copy's issued chunks, instead of waiting for the whole copy (A8): when
//! batch formation, an admission allocation or a capture's snapshot finds
//! the pool exhausted, and when a promotion larger than the step's budget
//! waits for no demotion to be outstanding. No demotion starts in a step
//! that cancelled one, nor while such a promotion waits.
//!
//! Forced mode, for exactness runs only, demotes every public capture at
//! the end of its step, in capture order and without the bounds above; the
//! scheduler removes each GPU anchor as its demotion begins, so every hit
//! is a RAM hit. A capture the host tier has no room for stays on the GPU.
use super::*;
use offload::TransferOwner;
use runtime_kv::{KvError, KvTransferPoll, PrefixCacheKeyV1};
use runtime_scheduler::{AdmissionReport, PrefixCaptureV1, PrefixRamConfig};

/// Steps the measured bounds look back over.
const WINDOW_STEPS: usize = 16;

/// Whether the engine runs the RAM prefix tier, and why not when it was
/// configured and left off.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefixRamStatusV1 {
    /// Not configured.
    Off,
    /// Demotions and promotions run. `forced` is the debug mode in which
    /// every hit is a RAM hit.
    Active { forced: bool },
    /// Configured, and left off for the reason given.
    Unavailable(PrefixRamUnavailableV1),
}

/// Why a configured RAM prefix tier is off.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefixRamUnavailableV1 {
    /// Prefix caching is off, so there is no anchor to demote.
    NoPrefixCache,
    /// The executor offers no KV offload.
    NoOffload,
    /// The executor cannot cancel a demotion, so an allocation that needs
    /// the pages a demotion leases could wait for its whole copy.
    NoCancel,
}

impl PrefixRamUnavailableV1 {
    /// A sentence for the startup log.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::NoPrefixCache => "prefix caching is off",
            Self::NoOffload => "the executor offers no KV offload",
            Self::NoCancel => "the executor cannot cancel a demotion",
        }
    }
}

/// The scheduler configuration the engine builds with, and the tier's
/// status: a configured tier the engine cannot run is cleared, so the
/// scheduler runs as it does without one.
pub(super) fn resolve<E: ModelExecutor>(
    config: &EngineConfig,
    executor: &E,
) -> (SchedulerConfig, PrefixRamStatusV1) {
    let Some(tier) = config.scheduler.prefix_ram() else {
        return (config.scheduler, PrefixRamStatusV1::Off);
    };
    let unavailable = if config.scheduler.prefix_state_pages().is_none() {
        Some(PrefixRamUnavailableV1::NoPrefixCache)
    } else if executor.kv_offload_info_v1().is_none() {
        Some(PrefixRamUnavailableV1::NoOffload)
    } else if !executor.kv_transfer_cancel_v1_supported() {
        Some(PrefixRamUnavailableV1::NoCancel)
    } else {
        None
    };
    match unavailable {
        Some(reason) => (
            config.scheduler.with_prefix_ram(None),
            PrefixRamStatusV1::Unavailable(reason),
        ),
        None => (
            config.scheduler,
            PrefixRamStatusV1::Active {
                forced: tier.forced,
            },
        ),
    }
}

/// The largest of the last [`WINDOW_STEPS`] per-step values.
#[derive(Clone, Copy, Debug, Default)]
struct StepWindow {
    values: [usize; WINDOW_STEPS],
    next: usize,
}

impl StepWindow {
    fn push(&mut self, value: usize) {
        self.values[self.next] = value;
        self.next = (self.next + 1) % WINDOW_STEPS;
    }

    fn max(&self) -> usize {
        self.values.iter().copied().max().unwrap_or(0)
    }
}

/// What the driver keeps between steps.
#[derive(Clone, Debug)]
pub(super) struct PrefixRamDriver {
    config: PrefixRamConfig,
    /// Pages of one prefill chunk and of one snapshot: the lead's floor.
    chunk_pages: usize,
    snapshot_pages: usize,
    allocated: StepWindow,
    evicted: StepWindow,
    /// The pool's gross allocation and eviction counters at the last
    /// measurement.
    allocated_mark: u64,
    evicted_mark: u64,
    /// A demotion was cancelled this step.
    cancelled: bool,
    /// Forced mode: the public anchors captured since the last demotions.
    captured: Vec<PrefixCacheKeyV1>,
}

impl PrefixRamDriver {
    pub(super) fn new(config: &EngineConfig, scheduler: &Scheduler) -> Option<Self> {
        Some(Self {
            config: config.scheduler.prefix_ram()?,
            chunk_pages: config
                .scheduler
                .prefill_chunk_limit()
                .div_ceil(config.kv_geometry.block_tokens()),
            snapshot_pages: config.scheduler.prefix_state_pages().unwrap_or(0),
            allocated: StepWindow::default(),
            evicted: StepWindow::default(),
            allocated_mark: scheduler.kv_allocated_pages(),
            evicted_mark: scheduler.kv_evicted_pages(),
            cancelled: false,
            captured: Vec::new(),
        })
    }

    /// A fresh driver for a startup scheduler, measured from its pool.
    pub(super) fn isolated(&self, scheduler: &Scheduler) -> Self {
        Self {
            allocated: StepWindow::default(),
            evicted: StepWindow::default(),
            allocated_mark: scheduler.kv_allocated_pages(),
            evicted_mark: scheduler.kv_evicted_pages(),
            cancelled: false,
            captured: Vec::new(),
            ..self.clone()
        }
    }

    /// Whether a deferred promotion larger than the step's budget waits for
    /// no demotion to be outstanding.
    fn promotion_starved(&self, scheduler: &Scheduler) -> bool {
        scheduler.deferred_promotion_pages() > self.config.promotion_budget_pages
    }

    /// Records the pages the step since the last call allocated and evicted.
    fn measure(&mut self, scheduler: &Scheduler) {
        let allocated = scheduler.kv_allocated_pages();
        let evicted = scheduler.kv_evicted_pages();
        let pages = |now: u64, mark: u64| usize::try_from(now - mark).unwrap_or(usize::MAX);
        self.allocated.push(pages(allocated, self.allocated_mark));
        self.evicted.push(pages(evicted, self.evicted_mark));
        self.allocated_mark = allocated;
        self.evicted_mark = evicted;
    }

    /// The lead L, given the largest deferred promotion.
    fn lead_pages(&self, deferred: usize) -> usize {
        self.evicted.max().saturating_mul(2).max(
            deferred
                .saturating_add(self.chunk_pages)
                .saturating_add(self.snapshot_pages),
        )
    }

    /// The measured F, given the largest deferred promotion and the pages
    /// swapped requests may allocate next step, before the scheduler's cap.
    fn allocation_bound(&self, deferred: usize, swap_in: usize) -> usize {
        self.allocated
            .max()
            .saturating_mul(2)
            .saturating_add(deferred)
            .saturating_add(swap_in)
    }
}

impl<E: ModelExecutor> Engine<E> {
    /// Whether the engine runs the RAM prefix tier, for the host's startup
    /// log.
    #[must_use]
    pub const fn prefix_ram_status(&self) -> PrefixRamStatusV1 {
        self.prefix_ram
    }

    fn prefix_ram_driver(&mut self) -> Option<&mut PrefixRamDriver> {
        self.offload.as_mut().and_then(|state| state.ram.as_mut())
    }

    /// Starts the tier's step, after accepting completed transfers and
    /// before admission: settles promotions whose copies ended, drops the
    /// host entries their publishes made unusable, and submits promotions
    /// admission planned and left unsubmitted.
    pub(super) fn begin_prefix_ram_step(&mut self) -> Result<(), EngineError> {
        let Some(ram) = self.prefix_ram_driver() else {
            return Ok(());
        };
        ram.cancelled = false;
        self.scheduler
            .begin_prefix_ram_step()
            .map_err(EngineError::Scheduler)?;
        self.drop_invalidated_host_entries()?;
        self.start_promotions()
    }

    /// After an admission pass: submits the promotions it planned and drops
    /// the host entries its resumes made unusable. When an allocation found
    /// the pool exhausted, or a promotion larger than the budget waits for
    /// no demotion to be outstanding, cancels the outstanding demotion and
    /// returns that admission should run again.
    pub(super) fn after_prefix_ram_admission(
        &mut self,
        report: &AdmissionReport,
    ) -> Result<bool, EngineError> {
        let Some(ram) = self.offload.as_ref().and_then(|state| state.ram.as_ref()) else {
            return Ok(false);
        };
        let starved = ram.promotion_starved(&self.scheduler);
        self.start_promotions()?;
        self.drop_invalidated_host_entries()?;
        if report.pool_exhausted().is_empty() && !starved {
            return Ok(false);
        }
        self.cancel_demotion_for_headroom()
    }

    /// Cancels the outstanding demotion for an allocation that needs the
    /// pages it leases, and waits for the cancel to settle: the copy issues
    /// no further chunk and reports once its issued chunks drain. Returns
    /// whether it cancelled one. Forced demotions, whose GPU anchors are
    /// already gone, swaps and promotions are never cancelled.
    pub(super) fn cancel_demotion_for_headroom(&mut self) -> Result<bool, EngineError> {
        let Some(state) = self.offload.as_mut() else {
            return Ok(false);
        };
        if state.ram.as_ref().is_none_or(|ram| ram.config.forced) {
            return Ok(false);
        }
        let Some((&id, transfer)) = state.transfers.iter_mut().find(|(_, transfer)| {
            matches!(
                transfer.owner,
                TransferOwner::Demotion {
                    cancelled: false,
                    ..
                }
            )
        }) else {
            return Ok(false);
        };
        match self.executor.cancel_kv_transfer_v1(id) {
            // A copy that had already finished reports through its poll.
            Ok(_) => {}
            Err(error) if error.is_fatal() => {
                self.prefix_state_unconfirmed = true;
                self.metrics.poison();
                return Err(EngineError::Executor(error));
            }
            // A refused cancel leaves the copy to finish.
            Err(_) => return Ok(false),
        }
        if let TransferOwner::Demotion { cancelled, .. } = &mut transfer.owner {
            *cancelled = true;
        }
        if let Some(ram) = state.ram.as_mut() {
            ram.cancelled = true;
        }
        let deadline = std::time::Instant::now() + self.executor.kv_transfer_drain_timeout_v1();
        loop {
            match self.executor.poll_kv_transfer_v1(id) {
                Ok(KvTransferPoll::Finished(completion)) => {
                    self.finish_transfer(id, completion)?;
                    return Ok(true);
                }
                // A late cancel is not lost DMA: the copy keeps its lease and
                // the next step polls it again.
                Ok(KvTransferPoll::Pending) if std::time::Instant::now() >= deadline => {
                    return Ok(true);
                }
                Ok(KvTransferPoll::Pending) => {
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
                Err(error) => {
                    self.prefix_state_unconfirmed = true;
                    self.metrics.poison();
                    return Err(EngineError::Executor(error));
                }
            }
        }
    }

    fn start_promotions(&mut self) -> Result<(), EngineError> {
        for promotion in self.scheduler.pending_promotions() {
            let state = self
                .offload
                .as_mut()
                .ok_or(EngineError::InvariantViolation)?;
            let plan = self
                .scheduler
                .begin_promotion(&mut state.tier, promotion.id)
                .map_err(EngineError::Scheduler)?;
            self.submit_anchor_transfer(TransferOwner::Promotion(promotion.id), plan)?;
        }
        Ok(())
    }

    fn drop_invalidated_host_entries(&mut self) -> Result<(), EngineError> {
        let state = self
            .offload
            .as_mut()
            .ok_or(EngineError::InvariantViolation)?;
        self.scheduler
            .drop_invalidated_host_entries(&mut state.tier)
            .map_err(EngineError::Scheduler)?;
        Ok(())
    }

    /// Notes a capture the bundle wrote, which forced mode demotes after the
    /// step's captures.
    pub(super) fn note_prefix_capture(&mut self, capture: &PrefixCaptureV1) {
        if let (Some(ram), Some(key)) = (self.prefix_ram_driver(), capture.public_key())
            && ram.config.forced
        {
            ram.captured.push(key);
        }
    }

    /// After the step's captures: measures the step's allocation and
    /// eviction, drops the host entries the captures made unusable, and
    /// starts a demotion when none is outstanding and one fits, or in forced
    /// mode demotes every capture.
    pub(super) fn drive_prefix_ram(&mut self) -> Result<(), EngineError> {
        let Some(state) = self.offload.as_mut() else {
            return Ok(());
        };
        let Some(ram) = state.ram.as_mut() else {
            return Ok(());
        };
        ram.measure(&self.scheduler);
        let forced = ram.config.forced.then(|| std::mem::take(&mut ram.captured));
        self.scheduler
            .drop_invalidated_host_entries(&mut state.tier)
            .map_err(EngineError::Scheduler)?;
        let keys = forced.unwrap_or_else(|| self.plan_demotion().into_iter().collect());
        for key in keys {
            self.start_demotion(key)?;
        }
        Ok(())
    }

    fn start_demotion(&mut self, key: PrefixCacheKeyV1) -> Result<(), EngineError> {
        let state = self
            .offload
            .as_mut()
            .ok_or(EngineError::InvariantViolation)?;
        // A forced capture may have gone since: evicted, recaptured or
        // already copied.
        if !self.scheduler.prefix_demotable(&state.tier, key) {
            return Ok(());
        }
        match self.scheduler.begin_demotion(&mut state.tier, key) {
            Ok(plan) => self.submit_anchor_transfer(
                TransferOwner::Demotion {
                    key,
                    cancelled: false,
                },
                plan,
            ),
            // The host tier had no room older than the anchor.
            Err(ScheduleError::Kv(KvError::PoolExhausted { .. })) => Ok(()),
            Err(error) => Err(EngineError::Scheduler(error)),
        }
    }

    /// The first candidate within the lead that meets the start rule, when
    /// no demotion is outstanding, none was cancelled this step and no
    /// promotion larger than the budget waits.
    fn plan_demotion(&self) -> Option<PrefixCacheKeyV1> {
        let ram = self.offload.as_ref()?.ram.as_ref()?;
        if self.scheduler.outstanding_demotions() > 0
            || ram.cancelled
            || ram.promotion_starved(&self.scheduler)
        {
            return None;
        }
        let deferred = self.scheduler.deferred_promotion_pages();
        let bound = ram.allocation_bound(deferred, self.scheduler.swap_in_pages());
        self.scheduler
            .prefix_demotion_candidates(ram.lead_pages(deferred))
            .into_iter()
            .map(|candidate| candidate.key)
            .find(|key| self.scheduler.prefix_demotion_headroom(*key, bound))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_windows_keep_the_busiest_of_the_last_sixteen_steps() {
        let mut window = StepWindow::default();
        assert_eq!(window.max(), 0);
        window.push(9);
        for step in 0..WINDOW_STEPS - 1 {
            window.push(step % 3);
        }
        assert_eq!(window.max(), 9);
        window.push(1);
        assert_eq!(window.max(), 2);
    }

    #[test]
    fn the_lead_and_the_bound_follow_the_measured_steps() {
        let mut driver = PrefixRamDriver {
            config: PrefixRamConfig {
                forced: false,
                promotion_budget_pages: 8,
            },
            chunk_pages: 4,
            snapshot_pages: 2,
            allocated: StepWindow::default(),
            evicted: StepWindow::default(),
            allocated_mark: 0,
            evicted_mark: 0,
            cancelled: false,
            captured: Vec::new(),
        };
        // With nothing measured, the lead is one chunk and one snapshot past
        // the deferred promotion, and F is the promotion and the swap-ins.
        assert_eq!(driver.lead_pages(0), 6);
        assert_eq!(driver.lead_pages(5), 11);
        assert_eq!(driver.allocation_bound(5, 3), 8);
        driver.allocated.push(7);
        driver.evicted.push(10);
        driver.allocated.push(3);
        driver.evicted.push(1);
        assert_eq!(driver.lead_pages(5), 20);
        assert_eq!(driver.lead_pages(15), 21);
        assert_eq!(driver.allocation_bound(5, 3), 22);
    }
}
