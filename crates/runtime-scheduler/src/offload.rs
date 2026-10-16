//! Scheduler ownership and reservation transitions for request swaps.
use super::*;
use runtime_kv::{
    HostAnchorId, HostTier, KvTransferCompletion, KvTransferId, KvTransferPlan, PrefixHitV1,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OffloadConfig {
    pub minimum_tokens: usize,
    pub snapshot_pages: usize,
    pub enabled: bool,
    pub force_host: bool,
}

/// A preempted request's own boundary. Copies keep their leases independently
/// of this plan, including after cancellation or a GPU-anchor resume.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResumePlan {
    pub(super) key: PrefixCacheKeyV1,
    pub(super) tokens: usize,
    pub(super) host: Option<HostAnchorId>,
    pub(super) ready: bool,
    pub(super) destination: Option<PrefixHitV1>,
    pub(super) submitted: Option<KvTransferId>,
    pub(super) discard: bool,
}

impl Scheduler {
    pub fn swap_needs_discard(&self) -> bool {
        self.resumes.values().any(|plan| plan.discard)
    }

    pub fn swapped_requests(&self) -> usize {
        self.resumes.len()
    }

    pub fn kv_geometry(&self) -> KvGeometry {
        self.kv.geometry()
    }
    pub fn configure_offload(&mut self, config: OffloadConfig) {
        self.offload = Some(config);
    }

    pub fn new_host_tier(&self, capacity: usize) -> Result<HostTier, ScheduleError> {
        HostTier::new(&self.kv, capacity).map_err(ScheduleError::Kv)
    }

    pub fn audit_host_tier(&self, tier: &HostTier) -> Result<(), ScheduleError> {
        tier.audit(&self.kv).map_err(ScheduleError::Kv)?;
        for (&id, plan) in &self.resumes {
            let record = self.record(id)?;
            let anchor = plan
                .host
                .and_then(|id| tier.anchor(id))
                .ok_or(ScheduleError::InvariantViolation)?;
            if anchor.owner != Some(id.get())
                || anchor.tokens != plan.tokens
                || anchor.ready != plan.ready
                || record.state.is_terminal()
                || (record.state == RequestState::Queued) != plan.destination.is_none()
                || plan.destination.as_ref().is_some_and(|hit| {
                    record.handle != Some(hit.handle()) || record.restore.is_none()
                })
            {
                return Err(ScheduleError::InvariantViolation);
            }
        }
        Ok(())
    }

    pub fn has_capture_anchor(&self, capture: &PrefixCaptureV1) -> bool {
        self.kv
            .anchor_pages(capture.key)
            .is_some_and(|a| a.0 == capture.tokens)
    }

    pub fn capture_swaps(&self, capture: &PrefixCaptureV1) -> bool {
        capture.swap
    }

    /// The tier makes the final capacity decision; every capture copies all
    /// of its pages. A full tier leaves the existing recompute capture intact.
    pub fn offload_capture(
        &mut self,
        tier: &mut HostTier,
        capture: &PrefixCaptureV1,
        inserted: bool,
    ) -> Result<KvTransferPlan, ScheduleError> {
        let (anchor, plan) = tier
            .begin_offload(&mut self.kv, capture.key, Some(capture.request_id.get()))
            .map_err(ScheduleError::Kv)?;
        if let Ok(index) = self.record_index(capture.request_id)
            && self.requests[index].own_anchor == Some(capture.key)
        {
            self.requests[index].own_anchor_transfer = Some(plan.id());
        }
        self.resumes.insert(
            capture.request_id,
            ResumePlan {
                key: capture.key,
                tokens: capture.tokens,
                host: Some(anchor),
                ready: false,
                destination: None,
                submitted: None,
                discard: false,
            },
        );
        if inserted && self.offload.is_some_and(|config| config.force_host) {
            self.kv
                .remove_anchor(capture.key)
                .map_err(ScheduleError::Kv)?;
        }
        Ok(plan)
    }

    pub fn accept_kv_transfer(
        &mut self,
        tier: &mut HostTier,
        completion: KvTransferCompletion,
    ) -> Result<(), ScheduleError> {
        let id = completion.id();
        let idle = completion.status() != runtime_kv::KvTransferStatus::Unconfirmed;
        tier.complete_transfer(&mut self.kv, completion)
            .map_err(|(error, _)| ScheduleError::Kv(error))?;
        if idle {
            for record in &mut self.requests {
                if record.own_anchor_transfer == Some(id) {
                    record.own_anchor_transfer = None;
                }
            }
        }
        for plan in self.resumes.values_mut() {
            plan.ready = plan
                .host
                .and_then(|id| tier.anchor(id))
                .is_some_and(|a| a.ready);
        }
        Ok(())
    }

    /// Discard owners no longer needed, without touching in-flight leases.
    pub fn release_unused_swaps(&mut self, tier: &mut HostTier) -> Result<(), ScheduleError> {
        let remove: Vec<_> = self
            .resumes
            .iter()
            .filter_map(|(&id, plan)| {
                let needed = self.record(id).is_ok_and(|record| {
                    !record.state.is_terminal()
                        && (record.state == RequestState::Queued || plan.destination.is_some())
                });
                (!needed
                    || plan.discard
                    || plan.host.is_none_or(|anchor| tier.anchor(anchor).is_none()))
                .then_some(id)
            })
            .collect();
        for id in remove {
            self.discard_swap(tier, id)?;
        }
        Ok(())
    }

    pub fn discard_swap(
        &mut self,
        tier: &mut HostTier,
        id: RequestId,
    ) -> Result<(), ScheduleError> {
        if let Some(plan) = self.resumes.remove(&id) {
            if let Some(anchor) = plan.host.filter(|anchor| tier.anchor(*anchor).is_some()) {
                tier.drop_anchor(anchor).map_err(ScheduleError::Kv)?;
            }
            // A queued request still owns its valid GPU boundary even when
            // the host copy failed or was discarded. Normal restore or
            // termination releases that private anchor.
            let queued_owner = self.record(id).is_ok_and(|record| {
                record.state == RequestState::Queued && record.own_anchor == Some(plan.key)
            });
            if !queued_owner {
                self.kv.remove_anchor(plan.key).map_err(ScheduleError::Kv)?;
                if let Ok(index) = self.record_index(id) {
                    self.requests[index].own_anchor = None;
                    self.requests[index].own_anchor_transfer = None;
                }
            }
        }
        Ok(())
    }

    pub fn pending_reloads(&self) -> Vec<RequestId> {
        self.resumes
            .iter()
            .filter_map(|(&id, plan)| {
                (plan.destination.is_some() && plan.submitted.is_none()).then_some(id)
            })
            .collect()
    }

    pub fn begin_reload(
        &mut self,
        tier: &mut HostTier,
        id: RequestId,
    ) -> Result<KvTransferPlan, ScheduleError> {
        let plan = self
            .resumes
            .get_mut(&id)
            .ok_or(ScheduleError::InvariantViolation)?;
        let transfer = tier
            .begin_reload(
                &mut self.kv,
                plan.host.ok_or(ScheduleError::InvariantViolation)?,
                plan.destination
                    .as_ref()
                    .ok_or(ScheduleError::InvariantViolation)?,
            )
            .map_err(ScheduleError::Kv)?;
        plan.submitted = Some(transfer.id());
        Ok(transfer)
    }

    pub fn reload_restore(&self, id: RequestId, transfer: KvTransferId) -> Option<PrefixRestoreV1> {
        let record = self.record(id).ok()?;
        if record.state.is_terminal() {
            return None;
        }
        let plan = self.resumes.get(&id)?;
        if plan.submitted != Some(transfer) {
            return None;
        }
        let hit = plan.destination.as_ref()?;
        Some(PrefixRestoreV1 {
            request_id: id,
            tokens: hit.tokens(),
            pages: hit.state().to_vec(),
            resumes_decoding: Self::restores_decode_boundary(record, hit.tokens()),
        })
    }

    pub fn complete_reload_restore(&mut self, id: RequestId) -> Result<(), ScheduleError> {
        self.ensure_no_pending()?;
        let index = self.record_index(id)?;
        let record = &mut self.requests[index];
        self.kv
            .finish_initialized_restore(record.handle.ok_or(ScheduleError::InvariantViolation)?)
            .map_err(ScheduleError::Kv)?;
        let released = record
            .restore
            .take()
            .ok_or(ScheduleError::InvariantViolation)?
            .len();
        record.reserved_blocks -= released;
        self.reserved_blocks -= released;
        if Self::restores_decode_boundary(record, record.prefilled_tokens) {
            record.prefilled_tokens = record.spec.prompt_tokens;
            record.replayed_tokens = 0;
            record.state = RequestState::Decoding;
            self.kv.metrics().prefix_restore_resumed_decoding();
        }
        Ok(())
    }
}
