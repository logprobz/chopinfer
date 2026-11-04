//! Scheduler side of the RAM prefix tier: its configuration and the
//! demotions the engine begins. Public host entries never go through the
//! swap bookkeeping ([`Scheduler::discard_swap`],
//! [`Scheduler::audit_host_tier`]'s owner checks).
use super::*;
use runtime_kv::{
    HostTier, KvTransferId, KvTransferPlan, PrefixDemotionCandidateV1, PrefixEvictionV1,
    PrefixHostInvalidationV1,
};

/// The RAM prefix tier's scheduler configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefixRamConfig {
    /// Debug forced mode, for exactness runs only: host entries survive
    /// descendants and resumes, a demotion removes its GPU anchor, and a
    /// promoted anchor goes after the restore that waited for it, so every
    /// hit is a RAM hit.
    pub forced: bool,
    /// Pages the promotions one step starts may take together. A single
    /// larger promotion starts only as the step's first, with no demotion
    /// outstanding.
    pub promotion_budget_pages: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct PrefixRamState {
    /// Demotions begun through [`Scheduler::begin_demotion`], until their
    /// confirmed completion.
    demotions: BTreeMap<KvTransferId, PrefixCacheKeyV1>,
}

impl Scheduler {
    /// The tier's configuration while it can promote: configured, with
    /// offload enabled.
    fn prefix_ram_active(&self) -> Option<PrefixRamConfig> {
        self.config
            .prefix_ram
            .filter(|_| self.offload.is_some_and(|config| config.enabled))
    }

    /// Begins a demotion: a copy of the public frontier `key` into a host
    /// entry that keeps the anchor's last GPU use. It stays outstanding until
    /// its confirmed completion.
    pub fn begin_demotion(
        &mut self,
        tier: &mut HostTier,
        key: PrefixCacheKeyV1,
    ) -> Result<KvTransferPlan, ScheduleError> {
        self.ensure_no_pending()?;
        self.prefix_ram_active()
            .ok_or(ScheduleError::InvariantViolation)?;
        let (_, plan) = tier
            .begin_offload(&mut self.kv, key, None)
            .map_err(ScheduleError::Kv)?;
        self.ram.demotions.insert(plan.id(), key);
        Ok(plan)
    }

    /// Demotions begun and not yet confirmed complete.
    #[must_use]
    pub fn outstanding_demotions(&self) -> usize {
        self.ram.demotions.len()
    }

    /// Demotion candidates covering the next `lead_pages` pages of the
    /// eviction order, with at least the offload minimum tokens; none while
    /// the tier cannot promote.
    #[must_use]
    pub fn prefix_demotion_candidates(&self, lead_pages: usize) -> Vec<PrefixDemotionCandidateV1> {
        match (self.prefix_ram_active(), self.offload) {
            (Some(_), Some(offload)) => self
                .kv
                .demotion_candidates(lead_pages, offload.minimum_tokens),
            _ => Vec::new(),
        }
    }

    /// Drops the host entries a resume, a recapture or a new resident
    /// descendant made unusable, returning each key with its cause. Call
    /// after operations that may invalidate: captures, admissions and
    /// promotion publishes.
    pub fn drop_invalidated_host_entries(
        &mut self,
        tier: &mut HostTier,
    ) -> Result<Vec<(PrefixCacheKeyV1, PrefixHostInvalidationV1)>, ScheduleError> {
        self.ensure_no_pending()?;
        tier.drop_invalidated_entries(&mut self.kv)
            .map_err(ScheduleError::Kv)
    }

    /// Takes the public GPU evictions logged since the last call, for miss
    /// accounting, with how many the log's bound discarded.
    pub fn drain_prefix_evictions(&mut self) -> (Vec<PrefixEvictionV1>, u64) {
        self.kv.drain_prefix_evictions()
    }

    /// Ends a demotion the tier accepted as idle: confirmed, complete or
    /// failed.
    pub(super) fn finish_demotion(&mut self, id: KvTransferId) {
        self.ram.demotions.remove(&id);
    }

    /// Every outstanding demotion is a transfer of the tier.
    pub(super) fn audit_prefix_ram_transfers(&self, tier: &HostTier) -> Result<(), ScheduleError> {
        let transfers: std::collections::BTreeSet<_> = tier.transfer_ids().collect();
        if self
            .ram
            .demotions
            .keys()
            .any(|transfer| !transfers.contains(transfer))
        {
            return Err(ScheduleError::InvariantViolation);
        }
        Ok(())
    }
}
