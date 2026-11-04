//! Scheduler side of the RAM prefix tier: the choice between a GPU match and
//! a ready host entry, the promotions admission starts and the requests that
//! wait for them, the demotions the engine begins, and the per-step
//! allocation bound a demotion's start rule uses.
//!
//! A promotion copies a public host entry into a pending GPU anchor. Its
//! pages count in the scheduler's reservation from its start until it is
//! published or abandoned, and its waiters run admission again after that:
//! a published anchor is a normal public GPU anchor, so every waiter, the
//! first included, admits through the GPU-hit path. Public host entries
//! never go through the swap bookkeeping ([`Scheduler::discard_swap`],
//! [`Scheduler::audit_host_tier`]'s owner checks).
use super::*;
use runtime_kv::{
    HostTier, KvTransferId, KvTransferPlan, PendingAnchorId, PrefixDemotionCandidateV1,
    PrefixEvictionV1, PrefixHostInvalidationV1, PrefixPendingCopyV1, PrefixPublishV1,
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

/// Why a request that preferred a ready host entry did not resume from it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefixRamFallbackV1 {
    /// The pending anchor found too few free and evictable pages.
    PoolExhausted,
    /// The promotion, or the GPU hit it leads to, did not fit the pages
    /// reservations leave.
    Capacity,
    /// The promotion's copy failed or never started.
    Failed,
    /// A logical event on the key invalidated the promotion.
    Invalidated(PrefixHostInvalidationV1),
    /// The key had a GPU anchor when the promotion would publish.
    GpuResident,
}

impl PrefixRamFallbackV1 {
    /// The metric label of the reason.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PoolExhausted => "pool_exhausted",
            Self::Capacity => "capacity",
            Self::Failed => "failed",
            Self::Invalidated(_) => "invalidated",
            Self::GpuResident => "gpu_resident",
        }
    }

    /// The fallback of a promotion that ended without publishing.
    const fn of(outcome: PrefixPublishV1) -> Option<Self> {
        match outcome {
            PrefixPublishV1::Published => None,
            PrefixPublishV1::NotCopied => Some(Self::Failed),
            PrefixPublishV1::Invalidated(reason) => Some(Self::Invalidated(reason)),
            PrefixPublishV1::GpuResident => Some(Self::GpuResident),
        }
    }
}

/// A promotion admission started, before the engine submits its copy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefixRamPromotionV1 {
    pub id: PendingAnchorId,
    pub tokens: usize,
    /// KV and snapshot pages the pending anchor holds.
    pub pages: usize,
}

/// A promotion that ended: published, or abandoned with the reason.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefixRamSettledV1 {
    pub id: PendingAnchorId,
    pub tokens: usize,
    pub pages: usize,
    pub outcome: PrefixPublishV1,
}

/// A restore of a promoted anchor by a request that waited for it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefixRamHitV1 {
    pub request_id: RequestId,
    pub key: PrefixCacheKeyV1,
    pub tokens: usize,
}

/// A request's progress through the RAM tier.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct RequestRam {
    /// The promotion the queued request waits for.
    pub(super) promotion: Option<PendingAnchorId>,
    /// The key of the promotion it waited for, published and not yet used
    /// by its next admission.
    pub(super) promoted: Option<PrefixCacheKeyV1>,
    /// The promoted anchor its pending restore resumes, with its tokens:
    /// the restore counts a RAM hit.
    pub(super) restore: Option<(PrefixCacheKeyV1, usize)>,
    /// Keys whose promotion failed for it; it never chooses them again.
    pub(super) refused: Vec<PrefixCacheKeyV1>,
    /// Why it last gave up a preferred host entry, reported when it admits.
    pub(super) fallback: Option<PrefixRamFallbackV1>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Promotion {
    /// The identity the anchor publishes under, which stays valid after
    /// every waiter is cancelled.
    chain: Arc<PrefixChainV1>,
    tokens: usize,
    pages: usize,
    submitted: Option<KvTransferId>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct PrefixRamState {
    promotions: BTreeMap<PendingAnchorId, Promotion>,
    /// Demotions begun through [`Scheduler::begin_demotion`], until their
    /// confirmed completion.
    demotions: BTreeMap<KvTransferId, PrefixCacheKeyV1>,
    hits: Vec<PrefixRamHitV1>,
    /// Promotion pages started since the step began.
    budget_used: usize,
    /// The largest promotion this step deferred for the budget.
    deferred_pages: usize,
}

impl PrefixRamState {
    /// Pages the pending promotions reserve.
    pub(super) fn reserved_pages(&self) -> usize {
        self.promotions
            .values()
            .map(|promotion| promotion.pages)
            .sum()
    }
}

/// What admission does with a request after comparing its matches.
pub(super) enum RamChoice {
    /// Resume from the GPU match, or run cold.
    Gpu,
    /// Wait for a promotion: one it started or joined, reported with its
    /// id, or one the step's budget or a forced demotion in flight defers.
    Wait(Option<PendingAnchorId>),
}

impl Scheduler {
    /// The tier's configuration while it can promote: configured, with
    /// offload enabled.
    fn prefix_ram_active(&self) -> Option<PrefixRamConfig> {
        self.config
            .prefix_ram
            .filter(|_| self.offload.is_some_and(|config| config.enabled))
    }

    /// One step's allocation bound F, in pages: no engine step takes more
    /// pages from the pool than this, which the start rule of a demotion
    /// ([`Self::prefix_demotion_headroom`]) keeps free or evictable. It is
    /// a worst-case bound from the configuration and the resume plans:
    ///
    /// - the batch: a step appends at most the token budget's tokens, since
    ///   every item costs at least as many budget tokens as it appends; an
    ///   append may start a partly used page and copy a shared tail page, so
    ///   each schedulable sequence adds two pages;
    /// - captures: each batch item captures at most twice (an interior anchor
    ///   and its chunk end) with the public snapshot pages, and under
    ///   optimistic admission each sequence may be preempted once before the
    ///   batch, capturing a private boundary with the swap snapshot pages;
    /// - the promotion page budget;
    /// - each swapped request that may resume next step, which allocates its
    ///   boundary's pages and snapshot pages before the batch.
    ///
    /// A single promotion larger than the budget starts only with no
    /// demotion outstanding, so it never meets a demotion's start rule.
    #[must_use]
    pub fn step_allocation_bound(&self) -> usize {
        let geometry = self.kv.geometry();
        let pages = |tokens: usize| tokens.div_ceil(geometry.block_tokens());
        let sequences = self.config.maximum_sequences;
        let public = self.config.prefix_state_pages.unwrap_or(0);
        let swap_state = self
            .offload
            .map(|config| config.snapshot_pages)
            .or(self.config.prefix_state_pages)
            .unwrap_or(0);
        let private = match self.config.admission {
            AdmissionPolicy::Reserved => 0,
            AdmissionPolicy::Optimistic(_) => swap_state,
        };
        let promotions = self
            .config
            .prefix_ram
            .map_or(0, |config| config.promotion_budget_pages);
        let resumes = self
            .resumes
            .values()
            .filter(|plan| plan.host.is_some() && plan.destination.is_none())
            .map(|plan| pages(plan.tokens).saturating_add(swap_state))
            .fold(0usize, usize::saturating_add);
        pages(self.config.scheduled_token_budget)
            .saturating_add(sequences.saturating_mul(2))
            .saturating_add(
                sequences.saturating_mul(public.saturating_mul(2).saturating_add(private)),
            )
            .saturating_add(promotions)
            .saturating_add(resumes)
    }

    /// Starts a step of the tier: settles every promotion whose copy ended
    /// (published, or abandoned with its pages released) or that can no
    /// longer start, releases their reservations, and resets the step's
    /// promotion budget. Each settled promotion's waiters run admission
    /// again; one whose promotion did not publish never chooses that key
    /// again. Call once per engine step, after accepting completed transfers
    /// and before admission. A promotion whose copy is unconfirmed stays
    /// held.
    pub fn begin_prefix_ram_step(&mut self) -> Result<Vec<PrefixRamSettledV1>, ScheduleError> {
        self.ensure_no_pending()?;
        self.ram.budget_used = 0;
        self.ram.deferred_pages = 0;
        let mut settled = Vec::new();
        let ids: Vec<_> = self.ram.promotions.keys().copied().collect();
        for id in ids {
            let pending = self
                .kv
                .pending_anchor(id)
                .ok_or(ScheduleError::InvariantViolation)?;
            let outcome = match pending.copy {
                PrefixPendingCopyV1::InFlight(_) => continue,
                PrefixPendingCopyV1::Waiting => {
                    if self.promotion_startable(id) {
                        continue;
                    }
                    self.kv
                        .abandon_pending_anchor(id)
                        .map_err(ScheduleError::Kv)?;
                    pending
                        .invalidation
                        .map_or(PrefixPublishV1::NotCopied, PrefixPublishV1::Invalidated)
                }
                PrefixPendingCopyV1::Copied | PrefixPendingCopyV1::Failed => {
                    let chain = Arc::clone(&self.ram.promotions[&id].chain);
                    self.kv
                        .publish_pending_anchor(id, &chain)
                        .map_err(ScheduleError::Kv)?
                }
            };
            let promotion = self.ram.promotions.remove(&id).expect("promotion listed");
            self.reserved_blocks -= promotion.pages;
            let fallback = PrefixRamFallbackV1::of(outcome);
            for record in &mut self.requests {
                if record.ram.promotion != Some(id) {
                    continue;
                }
                record.ram.promotion = None;
                match fallback {
                    None => record.ram.promoted = Some(id.key()),
                    Some(reason) => {
                        record.ram.refused.push(id.key());
                        record.ram.fallback = Some(reason);
                    }
                }
            }
            settled.push(PrefixRamSettledV1 {
                id,
                tokens: promotion.tokens,
                pages: promotion.pages,
                outcome,
            });
        }
        Ok(settled)
    }

    /// Whether a pending promotion may still start its copy: the tier can
    /// promote, nothing invalidated it, and its source entry is still the
    /// ready entry it was created from.
    fn promotion_startable(&self, id: PendingAnchorId) -> bool {
        self.prefix_ram_active().is_some()
            && self
                .kv
                .pending_anchor(id)
                .is_some_and(|pending| pending.invalidation.is_none())
            && self
                .kv
                .host_entry(id.key())
                .is_some_and(|entry| entry.serial == id.serial() && entry.ready)
    }

    /// Promotions admission started whose copy the engine has not submitted,
    /// for [`Self::begin_promotion`]. One that can no longer start is left
    /// for the next [`Self::begin_prefix_ram_step`] to abandon.
    #[must_use]
    pub fn pending_promotions(&self) -> Vec<PrefixRamPromotionV1> {
        self.ram
            .promotions
            .iter()
            .filter(|(id, promotion)| {
                promotion.submitted.is_none() && self.promotion_startable(**id)
            })
            .map(|(id, promotion)| PrefixRamPromotionV1 {
                id: *id,
                tokens: promotion.tokens,
                pages: promotion.pages,
            })
            .collect()
    }

    /// Plans a pending promotion's copy from its host entry. The engine
    /// submits the plan, then accepts its completion through
    /// [`Self::accept_kv_transfer`]; the next step's
    /// [`Self::begin_prefix_ram_step`] publishes or abandons it.
    pub fn begin_promotion(
        &mut self,
        tier: &mut HostTier,
        id: PendingAnchorId,
    ) -> Result<KvTransferPlan, ScheduleError> {
        self.ensure_no_pending()?;
        let promotion = self
            .ram
            .promotions
            .get_mut(&id)
            .filter(|promotion| promotion.submitted.is_none())
            .ok_or(ScheduleError::InvariantViolation)?;
        let plan = tier
            .begin_promotion(&mut self.kv, id)
            .map_err(ScheduleError::Kv)?;
        promotion.submitted = Some(plan.id());
        Ok(plan)
    }

    /// Begins a demotion: a copy of the public frontier `key` into a host
    /// entry that keeps the anchor's last GPU use. It stays outstanding until
    /// its confirmed completion. In forced mode the GPU anchor goes at once,
    /// as a forced swap's does, so the key's next hit is a RAM hit.
    pub fn begin_demotion(
        &mut self,
        tier: &mut HostTier,
        key: PrefixCacheKeyV1,
    ) -> Result<KvTransferPlan, ScheduleError> {
        self.ensure_no_pending()?;
        let config = self
            .prefix_ram_active()
            .ok_or(ScheduleError::InvariantViolation)?;
        let (_, plan) = tier
            .begin_offload(&mut self.kv, key, None)
            .map_err(ScheduleError::Kv)?;
        self.ram.demotions.insert(plan.id(), key);
        if config.forced {
            self.kv.remove_anchor(key).map_err(ScheduleError::Kv)?;
        }
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

    /// The start rule of a demotion of `key` under
    /// [`Self::step_allocation_bound`].
    #[must_use]
    pub fn prefix_demotion_headroom(&self, key: PrefixCacheKeyV1) -> bool {
        self.kv.demotion_headroom(key, self.step_allocation_bound())
    }

    /// The largest promotion this step deferred because the step's budget
    /// was spent or a demotion was outstanding. While it is nonzero, a new
    /// demotion would defer it again.
    #[must_use]
    pub const fn deferred_promotion_pages(&self) -> usize {
        self.ram.deferred_pages
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

    /// Takes the RAM hits counted since the last call: completed restores of
    /// promoted anchors by requests that waited for them.
    pub fn take_prefix_ram_hits(&mut self) -> Vec<PrefixRamHitV1> {
        std::mem::take(&mut self.ram.hits)
    }

    /// Ends a demotion the tier accepted as idle: confirmed, complete or
    /// failed.
    pub(super) fn finish_demotion(&mut self, id: KvTransferId) {
        self.ram.demotions.remove(&id);
    }

    /// Admission's step between a request's own boundaries and a cold
    /// start. With the GPU match `g` and the ready host match `h` of its
    /// prompt, a request prefers RAM when `h - g >= max(minimum_tokens,
    /// h / 16)`: a promotion copies `h` tokens over PCIe, while the GPU
    /// match leaves `h - g` tokens to prefill, so the gain must be large
    /// enough for the copy, and recomputation stays last. Revisit the ratio
    /// with the measured promotion copy rate and prefill rate. Forced mode
    /// prefers any longer host match, so every hit is a RAM hit.
    ///
    /// A preferred host entry whose promotion is pending is joined. One with
    /// none starts a promotion when the hit it leads to fits the pages
    /// reservations leave and the step's promotion budget has room, and the
    /// request waits. Otherwise it falls back to `g` or cold.
    pub(super) fn choose_prefix_ram(
        &mut self,
        index: usize,
        chain: &Arc<PrefixChainV1>,
        target: usize,
        g: usize,
    ) -> Result<RamChoice, ScheduleError> {
        let Some(config) = self.prefix_ram_active() else {
            return Ok(RamChoice::Gpu);
        };
        // Admission's own fallbacks hold for one pass; a settled
        // promotion's reason stays until the request admits.
        let fallback = &mut self.requests[index].ram.fallback;
        if matches!(
            fallback,
            Some(PrefixRamFallbackV1::Capacity | PrefixRamFallbackV1::PoolExhausted)
        ) {
            *fallback = None;
        }
        let limit = target - 1;
        let refused = |key: &PrefixCacheKeyV1| self.requests[index].ram.refused.contains(key);
        let (key, h) = self
            .kv
            .lookup_host_anchor(chain, limit)
            .filter(|(key, _)| !refused(key))
            .map_or((None, 0), |(key, tokens)| (Some(key), tokens));
        // A forced demotion removed its GPU anchor when it started; a longer
        // entry whose copy is in flight is waited for.
        if config.forced
            && self
                .kv
                .lookup_host_demotion(chain, limit)
                .is_some_and(|(key, tokens)| tokens > g.max(h) && !refused(&key))
        {
            return Ok(RamChoice::Wait(None));
        }
        let Some(key) = key.filter(|_| h > g) else {
            return Ok(RamChoice::Gpu);
        };
        let minimum = self.offload.map_or(0, |offload| offload.minimum_tokens);
        if !config.forced && h - g < minimum.max(h / 16) {
            return Ok(RamChoice::Gpu);
        }
        let serial = self.kv.host_entry(key).map(|entry| entry.serial);
        let joined = self.ram.promotions.keys().copied().find(|id| {
            id.key() == key
                && Some(id.serial()) == serial
                && self
                    .kv
                    .pending_anchor(*id)
                    .is_some_and(|pending| pending.invalidation.is_none())
        });
        if let Some(id) = joined {
            let ram = &mut self.requests[index].ram;
            ram.promotion = Some(id);
            ram.fallback = None;
            return Ok(RamChoice::Wait(Some(id)));
        }
        let geometry = self.kv.geometry();
        let record = &self.requests[index];
        let pinned = self.pinned(record);
        let watermark = if pinned || self.active_sequences == 0 {
            0
        } else {
            self.watermark_pages
        };
        let available =
            (geometry.block_capacity() - self.reserved_blocks).saturating_sub(watermark);
        let hit_pins = self
            .offload
            .map_or(self.config.prefix_state_pages.unwrap_or(0), |offload| {
                offload.snapshot_pages
            });
        let reserved = if pinned {
            record.spec.maximum_kv_tokens
        } else {
            target
        };
        let hit = geometry.blocks_for(reserved).map_err(ScheduleError::Kv)?
            + hit_pins
            + usize::from(Self::copies_tail(
                geometry.block_tokens(),
                self.config.rewrites_last_page,
                h,
            ));
        if hit > available {
            self.requests[index].ram.fallback = Some(PrefixRamFallbackV1::Capacity);
            return Ok(RamChoice::Gpu);
        }
        let state_pages = self.config.prefix_state_pages.unwrap_or(0);
        let pages = geometry.blocks_for(h).map_err(ScheduleError::Kv)? + state_pages;
        if self.ram.budget_used + pages > config.promotion_budget_pages
            && (self.ram.budget_used > 0 || !self.ram.demotions.is_empty())
        {
            self.ram.deferred_pages = self.ram.deferred_pages.max(pages);
            return Ok(RamChoice::Wait(None));
        }
        let id = match self.kv.create_pending_anchor(key, state_pages) {
            Ok(id) => id,
            Err(KvError::PoolExhausted { .. }) => {
                self.requests[index].ram.fallback = Some(PrefixRamFallbackV1::PoolExhausted);
                return Ok(RamChoice::Gpu);
            }
            Err(error) => return Err(ScheduleError::Kv(error)),
        };
        self.ram.promotions.insert(
            id,
            Promotion {
                chain: Arc::clone(chain),
                tokens: h,
                pages,
                submitted: None,
            },
        );
        self.ram.budget_used += pages;
        self.reserved_blocks += pages;
        let ram = &mut self.requests[index].ram;
        ram.promotion = Some(id);
        ram.fallback = None;
        Ok(RamChoice::Wait(Some(id)))
    }

    /// Ends a restore: one that resumed the promoted anchor its request
    /// waited for counts a RAM hit, and in forced mode the anchor goes, so
    /// the key's next hit is a RAM hit again.
    pub(super) fn finish_prefix_ram_restore(&mut self, index: usize) -> Result<(), ScheduleError> {
        let Some((key, tokens)) = self.requests[index].ram.restore.take() else {
            return Ok(());
        };
        self.ram.hits.push(PrefixRamHitV1 {
            request_id: self.requests[index].spec.id,
            key,
            tokens,
        });
        if self.config.prefix_ram.is_some_and(|config| config.forced) {
            self.kv.remove_anchor(key).map_err(ScheduleError::Kv)?;
        }
        Ok(())
    }

    /// Each promotion holds its pending anchor's pages in the reservation,
    /// and its copy state matches its submission. Only queued requests wait
    /// for a promotion that exists, and only a pending restore counts a RAM
    /// hit.
    pub(super) fn audit_prefix_ram(&self) -> Result<(), ScheduleError> {
        for (id, promotion) in &self.ram.promotions {
            let pending = self
                .kv
                .pending_anchor(*id)
                .ok_or(ScheduleError::InvariantViolation)?;
            let copy = match (pending.copy, promotion.submitted) {
                (PrefixPendingCopyV1::Waiting, None) => true,
                (PrefixPendingCopyV1::InFlight(transfer), Some(submitted)) => transfer == submitted,
                (PrefixPendingCopyV1::Copied | PrefixPendingCopyV1::Failed, Some(_)) => true,
                _ => false,
            };
            if !copy
                || pending.pages != promotion.pages
                || pending.tokens != promotion.tokens
                || promotion.chain.key_at(promotion.tokens) != Some(id.key())
            {
                return Err(ScheduleError::InvariantViolation);
            }
        }
        for record in &self.requests {
            if record.ram.promotion.is_some_and(|id| {
                record.state != RequestState::Queued || !self.ram.promotions.contains_key(&id)
            }) || (record.ram.restore.is_some() && record.restore.is_none())
            {
                return Err(ScheduleError::InvariantViolation);
            }
        }
        Ok(())
    }

    /// Every outstanding demotion and submitted promotion copy is a
    /// transfer of the tier.
    pub(super) fn audit_prefix_ram_transfers(&self, tier: &HostTier) -> Result<(), ScheduleError> {
        let transfers: std::collections::BTreeSet<_> = tier.transfer_ids().collect();
        let in_flight = self.ram.promotions.keys().filter_map(|id| {
            match self.kv.pending_anchor(*id).map(|pending| pending.copy) {
                Some(PrefixPendingCopyV1::InFlight(transfer)) => Some(transfer),
                _ => None,
            }
        });
        if self
            .ram
            .demotions
            .keys()
            .copied()
            .chain(in_flight)
            .any(|transfer| !transfers.contains(&transfer))
        {
            return Err(ScheduleError::InvariantViolation);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_kv::KvTransferStatus;

    /// A scheduler whose request 3 waits for a promotion of a 17-token host
    /// entry, with its host tier.
    fn waiting() -> (Scheduler, HostTier) {
        let config = SchedulerConfig::new(2, 16, 64, 64, 1)
            .unwrap()
            .with_admission(AdmissionPolicy::Optimistic(
                OptimisticAdmission::new(0, 3).unwrap(),
            ))
            .with_prefix_cache(1)
            .with_prefix_ram(Some(PrefixRamConfig {
                forced: false,
                promotion_budget_pages: 64,
            }));
        let mut s = Scheduler::new(config, KvGeometry::new(4, 16).unwrap()).unwrap();
        s.configure_offload(OffloadConfig {
            minimum_tokens: 1,
            snapshot_pages: 1,
            enabled: true,
            force_host: false,
        });
        let mut tier = s.new_host_tier(16).unwrap();
        let run = |s: &mut Scheduler, id: u64, tokens: Vec<u32>, capture: bool| {
            let chain = Arc::new(PrefixChainV1::new(4, tokens.clone().into(), Vec::new()).unwrap());
            s.submit_with_prefix(
                RequestSpec::new(RequestId::new(id), tokens.len(), 1).unwrap(),
                Some(PrefixRequestV1::new(Arc::clone(&chain), true)),
            )
            .unwrap();
            while let Some(batch) = s.next_batch().unwrap() {
                let outcomes: Vec<_> = batch
                    .items()
                    .iter()
                    .map(|item| BatchOutcome::success(item.request_id(), item.phase(), 1, true))
                    .collect();
                s.commit_batch(batch.id(), &outcomes).unwrap();
                for due in s.take_prefix_captures() {
                    if capture {
                        s.begin_prefix_capture(&due).unwrap();
                    }
                    s.end_prefix_capture(&due).unwrap();
                }
            }
            chain
        };
        let chain = run(&mut s, 1, (0..17).collect(), true);
        let plan = s
            .begin_demotion(&mut tier, chain.key_at(17).unwrap())
            .unwrap();
        s.accept_kv_transfer(
            &mut tier,
            plan.into_submission()
                .completion(KvTransferStatus::Complete),
        )
        .unwrap();
        // A cold request evicts the anchor from the GPU.
        run(&mut s, 2, (100..144).collect(), false);
        let mut tokens: Vec<u32> = (0..17).collect();
        tokens.extend(200..207);
        let chain = PrefixChainV1::new(4, tokens.into(), Vec::new()).unwrap();
        s.submit_with_prefix(
            RequestSpec::new(RequestId::new(3), 24, 1).unwrap(),
            Some(PrefixRequestV1::new(Arc::new(chain), true)),
        )
        .unwrap();
        assert_eq!(s.admit().unwrap().ram_waits().len(), 1);
        s.audit().unwrap();
        s.audit_host_tier(&tier).unwrap();
        (s, tier)
    }

    #[test]
    fn audits_reject_promotions_out_of_step_with_reservations_or_waiters() {
        type Tamper = fn(&mut Scheduler, &mut HostTier);
        let tampers: [(&str, Tamper); 6] = [
            ("reservation", |s, _| s.reserved_blocks -= 1),
            ("pages", |s, _| {
                s.ram.promotions.values_mut().next().unwrap().pages += 1;
            }),
            ("missing promotion", |s, _| {
                let pages = s.ram.reserved_pages();
                s.ram.promotions.clear();
                s.reserved_blocks -= pages;
            }),
            ("live waiter", |s, _| {
                let index = s.record_index(RequestId::new(3)).unwrap();
                s.requests[index].state = RequestState::Cancelled;
            }),
            ("ram hit without a restore", |s, _| {
                let index = s.record_index(RequestId::new(3)).unwrap();
                s.requests[index].ram.restore = Some((PrefixCacheKeyV1::new([1; 32]), 17));
            }),
            ("unknown demotion", |s, tier| {
                // A finished copy's transfer is no longer the tier's.
                let id = s.pending_promotions()[0].id;
                let plan = s.begin_promotion(tier, id).unwrap();
                let transfer = plan.id();
                s.accept_kv_transfer(
                    tier,
                    plan.into_submission()
                        .completion(KvTransferStatus::FailedIdle),
                )
                .unwrap();
                s.ram.demotions.insert(transfer, id.key());
            }),
        ];
        for (name, tamper) in tampers {
            let (mut s, mut tier) = waiting();
            tamper(&mut s, &mut tier);
            let audits = s.audit().and_then(|()| s.audit_host_tier(&tier));
            assert_eq!(audits, Err(ScheduleError::InvariantViolation), "{name}");
        }
    }
}
