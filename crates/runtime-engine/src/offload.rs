//! Copy completion is the only authority that releases transfer leases.
use super::*;
use runtime_kv::{HostTier, KvTransferDirection, KvTransferId, KvTransferPlan};
use runtime_scheduler::PrefixCaptureV1;

#[derive(Debug)]
pub(super) struct OffloadState {
    pub tier: HostTier,
    info: KvOffloadInfo,
    force_host: bool,
    transfers: BTreeMap<KvTransferId, Transfer>,
}
#[derive(Debug)]
struct Transfer {
    request: RequestId,
    direction: KvTransferDirection,
    started: u64,
    stall_after_micros: u64,
    warned: bool,
}
impl OffloadState {
    /// A startup scheduler has a separate pool identity and transfer ledger.
    pub(super) fn isolated(&self, scheduler: &Scheduler) -> Result<Self, EngineError> {
        let tier = scheduler
            .new_host_tier(self.tier.capacity())
            .map_err(EngineError::Scheduler)?;
        Ok(Self::new(tier, self.info, self.force_host))
    }

    pub(super) fn audit_transfers(&self) -> Result<(), EngineError> {
        if !self.transfers.keys().copied().eq(self.tier.transfer_ids()) {
            return Err(EngineError::InvariantViolation);
        }
        Ok(())
    }
    pub fn new(tier: HostTier, info: KvOffloadInfo, force_host: bool) -> Self {
        Self {
            tier,
            info,
            force_host,
            transfers: BTreeMap::new(),
        }
    }
}
impl<E: ModelExecutor> Engine<E> {
    pub(super) fn has_transfers(&self) -> bool {
        self.offload
            .as_ref()
            .is_some_and(|state| !state.transfers.is_empty())
    }
    pub(super) fn update_offload_support(&mut self) {
        if let Some(state) = &self.offload {
            self.scheduler
                .configure_offload(runtime_scheduler::OffloadConfig {
                    minimum_tokens: state.info.minimum_tokens,
                    snapshot_pages: state.info.snapshot_pages,
                    enabled: self.executor.kv_offload_info_v1().is_some(),
                    force_host: state.force_host,
                });
        }
    }
    pub(super) fn release_unused_swaps(&mut self) -> Result<(), EngineError> {
        if let Some(state) = &mut self.offload {
            self.scheduler
                .release_unused_swaps(&mut state.tier)
                .map_err(EngineError::Scheduler)?;
            self.metrics.offload_occupancy(
                state.tier.used_pages(),
                state.tier.capacity(),
                self.scheduler.swapped_requests(),
            );
        }
        Ok(())
    }
    pub(super) fn start_offload(
        &mut self,
        capture: &PrefixCaptureV1,
        inserted: bool,
    ) -> Result<(), EngineError> {
        let state = self
            .offload
            .as_mut()
            .ok_or(EngineError::InvariantViolation)?;
        let plan = match self
            .scheduler
            .offload_capture(&mut state.tier, capture, inserted)
        {
            Ok(plan) => plan,
            Err(ScheduleError::Kv(runtime_kv::KvError::PoolExhausted { .. })) => {
                self.metrics.offload_fallback("tier_full");
                return Ok(());
            }
            Err(error) => return Err(EngineError::Scheduler(error)),
        };
        self.submit_transfer(capture.request_id(), plan)
    }
    fn submit_transfer(
        &mut self,
        request: RequestId,
        plan: KvTransferPlan,
    ) -> Result<(), EngineError> {
        let id = plan.id();
        let direction = plan.direction();
        let submitted = match direction {
            KvTransferDirection::Out => self.executor.offload_pages_v1(plan),
            KvTransferDirection::In => self.executor.reload_pages_v1(plan),
        };
        match submitted {
            Ok(()) => {
                self.offload
                    .as_mut()
                    .expect("offload configured")
                    .transfers
                    .insert(
                        id,
                        Transfer {
                            request,
                            direction,
                            started: self.clock.now_micros(),
                            stall_after_micros: self
                                .executor
                                .kv_transfer_drain_timeout_v1()
                                .saturating_mul(2)
                                .as_micros()
                                .min(u128::from(u64::MAX))
                                as u64,
                            warned: false,
                        },
                    );
            }
            Err((error, plan)) => {
                let state = self.offload.as_mut().expect("offload configured");
                self.scheduler
                    .accept_kv_transfer(&mut state.tier, plan.cancel())
                    .map_err(EngineError::Scheduler)?;
                self.metrics
                    .offload_fallback(if direction == KvTransferDirection::Out {
                        "out_refused"
                    } else {
                        "in_refused"
                    });
                if direction == KvTransferDirection::In {
                    self.scheduler
                        .abandon_prefix_restore(request)
                        .map_err(EngineError::Scheduler)?;
                }
                self.scheduler
                    .discard_swap(&mut state.tier, request)
                    .map_err(EngineError::Scheduler)?;
                // A refused submission promises no DMA. A fatal executor error
                // still stops serving, even though these leases are safe.
                if error.is_fatal() {
                    self.model_state_failed = true;
                    self.metrics.poison();
                    return Err(EngineError::Executor(error));
                }
            }
        }
        self.release_unused_swaps()
    }
    pub(super) fn start_reloads(&mut self) -> Result<(), EngineError> {
        for id in self.scheduler.pending_reloads() {
            let state = self
                .offload
                .as_mut()
                .ok_or(EngineError::InvariantViolation)?;
            let plan = self
                .scheduler
                .begin_reload(&mut state.tier, id)
                .map_err(EngineError::Scheduler)?;
            self.submit_transfer(id, plan)?;
        }
        Ok(())
    }
    /// Poll every transfer at step start. If allocation cannot progress, wait
    /// for the oldest transfer, then retry scheduling at the next boundary.
    pub(super) fn poll_transfers(&mut self, wait: bool) -> Result<(), EngineError> {
        let ids: Vec<_> = self
            .offload
            .as_ref()
            .map(|s| s.transfers.keys().copied().collect())
            .unwrap_or_default();
        for (index, id) in ids.into_iter().enumerate() {
            // Never enter an executor's potentially unbounded event wait.
            // Brief polling preserves a retry boundary for allocation stalls.
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(5);
            let polled = loop {
                let result = self.executor.poll_kv_transfer_v1(id);
                if !wait
                    || index != 0
                    || !matches!(result, Ok(KvTransferPoll::Pending))
                    || std::time::Instant::now() >= deadline
                {
                    break result;
                }
                std::thread::sleep(std::time::Duration::from_micros(100));
            };
            let polled = match polled {
                Ok(poll) => poll,
                Err(error) => {
                    // No terminal evidence means no release, even if the
                    // executor labelled the unexpected polling error nonfatal.
                    self.prefix_state_unconfirmed = true;
                    self.metrics.poison();
                    return Err(EngineError::Executor(error));
                }
            };
            let KvTransferPoll::Finished(completion) = polled else {
                continue;
            };
            if completion.id() != id {
                return Err(EngineError::InvariantViolation);
            }
            let status = completion.status();
            let state = self.offload.as_mut().expect("offload configured");
            self.scheduler
                .accept_kv_transfer(&mut state.tier, completion)
                .map_err(EngineError::Scheduler)?;
            if status == KvTransferStatus::Unconfirmed {
                self.prefix_state_unconfirmed = true;
                self.metrics.poison();
                self.metrics.offload_fallback("unconfirmed");
                return Err(EngineError::PrefixStateUnconfirmed);
            }
            let transfer = state
                .transfers
                .remove(&id)
                .ok_or(EngineError::InvariantViolation)?;
            if status == KvTransferStatus::Complete {
                self.metrics.offload_copy(
                    transfer.direction == KvTransferDirection::In,
                    self.clock.now_micros().saturating_sub(transfer.started),
                );
            } else {
                self.metrics
                    .offload_fallback(if transfer.direction == KvTransferDirection::In {
                        "in_failed"
                    } else {
                        "out_failed"
                    });
            }
            if transfer.direction == KvTransferDirection::In
                && let Some(restore) = self.scheduler.reload_restore(transfer.request, id)
            {
                if status == KvTransferStatus::Complete && !self.model_state_failed {
                    self.restore_reloaded(restore)?;
                } else {
                    self.scheduler
                        .abandon_prefix_restore(transfer.request)
                        .map_err(EngineError::Scheduler)?;
                }
                let state = self.offload.as_mut().expect("offload configured");
                self.scheduler
                    .discard_swap(&mut state.tier, transfer.request)
                    .map_err(EngineError::Scheduler)?;
            }
        }
        self.update_transfer_health();
        self.release_unused_swaps()
    }

    fn update_transfer_health(&mut self) {
        let Some(state) = &mut self.offload else {
            return;
        };
        let now = self.clock.now_micros();
        let mut oldest = 0;
        let mut overdue = 0;
        let mut new_stalls = 0;
        for (id, transfer) in &mut state.transfers {
            let age = now.saturating_sub(transfer.started);
            oldest = oldest.max(age);
            if age >= transfer.stall_after_micros {
                overdue += 1;
                if !transfer.warned {
                    eprintln!(
                        "KV transfer {id:?} ({:?}, request {:?}) is overdue after {} ms; retaining leases and continuing polls",
                        transfer.direction,
                        transfer.request,
                        age / 1000
                    );
                    transfer.warned = true;
                    new_stalls += 1;
                }
            }
        }
        self.metrics
            .offload_transfer_health(oldest, overdue, new_stalls);
    }

    /// Readiness is recoverable: a late copy is not evidence of lost DMA.
    pub fn has_stalled_kv_transfers(&self) -> bool {
        self.offload
            .as_ref()
            .is_some_and(|state| state.transfers.values().any(|t| t.warned))
    }

    fn restore_reloaded(
        &mut self,
        restore: runtime_scheduler::PrefixRestoreV1,
    ) -> Result<(), EngineError> {
        let id = restore.request_id();
        let record = self
            .requests
            .get_mut(&id)
            .ok_or(EngineError::InvariantViolation)?;
        let prefix = record
            .history
            .get(..restore.tokens())
            .ok_or(EngineError::InvariantViolation)?;
        let started = self.clock.now_micros();
        let result = self
            .executor
            .restore_prefix_state_v1(id, prefix, restore.pages());
        self.metrics
            .prefix_state_call(false, self.clock.now_micros().saturating_sub(started));
        match result {
            Ok(()) => {
                self.scheduler
                    .complete_reload_restore(id)
                    .map_err(EngineError::Scheduler)?;
                record.prompt_offset = if restore.resumes_decoding() {
                    record.prompt_tokens.len()
                } else {
                    restore.tokens()
                };
            }
            Err(error) if error.is_fatal() => {
                self.prefix_state_unconfirmed = true;
                self.metrics.poison();
                return Err(EngineError::Executor(error));
            }
            Err(_) => {
                self.metrics.offload_fallback("restore");
                self.scheduler
                    .abandon_prefix_restore(id)
                    .map_err(EngineError::Scheduler)?;
            }
        }
        Ok(())
    }
    /// One nonblocking idle poll. Returns whether copies still need polling.
    pub fn poll_idle_kv_transfers(&mut self) -> Result<bool, EngineError> {
        if self.prefix_state_unconfirmed {
            return Err(EngineError::PrefixStateUnconfirmed);
        }
        self.poll_transfers(false)?;
        Ok(self.has_transfers())
    }

    /// Wait only during shutdown. Elapsed time is not evidence of lost DMA:
    /// timeout retains leases and leaves a healthy engine able to poll again.
    pub fn drain_kv_transfers(&mut self) -> Result<(), EngineError> {
        self.drain_until(std::time::Instant::now() + self.executor.kv_transfer_drain_timeout_v1())
    }

    fn drain_until(&mut self, deadline: std::time::Instant) -> Result<(), EngineError> {
        while self.has_transfers() {
            if self.prefix_state_unconfirmed {
                return Err(EngineError::PrefixStateUnconfirmed);
            }
            self.poll_transfers(false)?;
            if self.has_transfers() && std::time::Instant::now() >= deadline {
                return Err(EngineError::KvTransferDrainTimeout);
            }
            if self.has_transfers() {
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
        }
        Ok(())
    }
}
impl<E: ModelExecutor> Drop for Engine<E> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        // Destruction must not restore live model state or call blocking
        // executor fences. The physical arena independently retains DMA owners
        // if it cannot establish completion. A panicking executor cannot turn
        // this best-effort cleanup into a second unwind.
        self.model_state_failed = true;
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.drain_until(std::time::Instant::now() + std::time::Duration::from_millis(10))
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_kv::{BlockId, KvTransferSubmission, PreparedAppendBatch};
    use runtime_scheduler::OptimisticAdmission;

    #[derive(Default)]
    struct Model {
        refuse: bool,
        hold: bool,
        pending: Option<KvTransferSubmission>,
        snapshots: BTreeMap<BlockId, usize>,
        restored: bool,
    }

    impl ModelExecutor for Model {
        type TransactionState<'a> = NoopModelTransactionState;
        fn kv_offload_info_v1(&self) -> Option<KvOffloadInfo> {
            Some(KvOffloadInfo {
                host_page_capacity: 32,
                minimum_tokens: 1,
                snapshot_pages: 1,
            })
        }
        fn offload_pages_v1(&mut self, plan: KvTransferPlan) -> KvTransferSubmitResult {
            if self.refuse {
                return Err((ExecutorError::new("refused").unwrap(), plan));
            }
            self.pending = Some(plan.into_submission());
            Ok(())
        }
        fn poll_kv_transfer_v1(
            &mut self,
            _: KvTransferId,
        ) -> Result<KvTransferPoll, ExecutorError> {
            if self.hold {
                return Ok(KvTransferPoll::Pending);
            }
            Ok(KvTransferPoll::Finished(
                self.pending
                    .as_ref()
                    .unwrap()
                    .completion(KvTransferStatus::FailedIdle),
            ))
        }
        fn capture_prefix_state_v1(
            &mut self,
            _: RequestId,
            tokens: usize,
            pages: &[BlockId],
        ) -> Result<(), ExecutorError> {
            self.snapshots.insert(pages[0], tokens);
            Ok(())
        }
        fn restore_prefix_state_v1(
            &mut self,
            _: RequestId,
            prefix: &[u32],
            pages: &[BlockId],
        ) -> Result<(), ExecutorError> {
            assert_eq!(self.snapshots[&pages[0]], prefix.len());
            self.restored = true;
            Ok(())
        }
        fn execute<'a>(
            &'a mut self,
            items: &[ModelBatchItem],
            _: &PreparedAppendBatch,
        ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
            let rows = items
                .iter()
                .filter(|item| item.expects_logits())
                .map(|item| {
                    if self.restored {
                        assert!(!matches!(item.phase(), BatchPhase::Prefill { .. }));
                    }
                    LogitsRow::new(item.request_id(), vec![0.0, 1.0]).unwrap()
                })
                .collect();
            Ok(ModelTransaction::no_op(rows))
        }
    }

    fn queued_copy(failure: &str) {
        let id = RequestId::new(1);
        let config = EngineConfig::new(
            SchedulerConfig::new(1, 2, 8, 8, 1)
                .unwrap()
                .with_last_page_rewrites(failure == "discard_pending"),
            KvGeometry::new(4, 5).unwrap(),
            ModelVocabulary::new(2).unwrap(),
            16,
            4,
        )
        .unwrap()
        .with_admission(AdmissionPolicy::Optimistic(
            OptimisticAdmission::new(0, 2)
                .unwrap()
                .with_forced_preemption(2)
                .unwrap(),
        ));
        let mut e = Engine::new(
            config,
            Model {
                refuse: failure == "refusal",
                ..Model::default()
            },
        )
        .unwrap();
        e.submit(
            EngineRequest::new(
                id,
                vec![0; 8],
                8,
                SamplingParameters::new(0.0).unwrap(),
                0,
                vec![],
                ModelVocabulary::new(2).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        e.step().unwrap();
        e.step().unwrap();
        // Stop at the actual engine preemption boundary, before readmission.
        // The five-page pool holds the three-page prefix and its snapshot.
        e.preempt().unwrap();
        assert_eq!(e.request_state(id).unwrap(), RequestState::Queued);
        assert_eq!(e.kv_metrics().snapshot().prefix_restore_resumed_decoding, 0);
        match failure {
            "failure" => e.poll_idle_kv_transfers().unwrap(),
            "discard" | "discard_pending" => {
                let state = e.offload.as_mut().unwrap();
                e.scheduler.discard_swap(&mut state.tier, id).unwrap();
                if failure == "discard_pending" {
                    e.executor_mut().hold = true;
                    assert!(e.poll_idle_kv_transfers().unwrap());
                    e.admit_and_restore().unwrap();
                    assert_eq!(e.request_state(id).unwrap(), RequestState::Queued);
                    assert!(!e.executor().restored);
                    e.executor_mut().hold = false;
                }
                e.poll_idle_kv_transfers().unwrap()
            }
            "refusal" => false,
            _ => unreachable!(),
        };
        assert_eq!(e.request_state(id).unwrap(), RequestState::Queued);
        e.audit().unwrap();
        e.step().unwrap();
        assert_eq!(e.kv_metrics().snapshot().prefix_restore_resumed_decoding, 1);
        assert!(e.executor().restored);
        e.cancel(id).unwrap();
        e.audit().unwrap();
        assert_eq!(e.kv_stats().used_blocks(), 0);
        assert_eq!(e.metrics().snapshot().offload.used_pages, 0);
    }

    #[test]
    fn queued_outgoing_failure_preserves_gpu_resume() {
        queued_copy("failure");
    }
    #[test]
    fn queued_outgoing_refusal_preserves_gpu_resume() {
        queued_copy("refusal");
    }
    #[test]
    fn discarded_copy_still_blocks_a_rewriting_resume_until_idle() {
        queued_copy("discard_pending");
    }

    #[test]
    fn queued_swap_discard_preserves_gpu_resume() {
        queued_copy("discard");
    }
}
