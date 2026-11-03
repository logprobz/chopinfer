//! Ownership retained while a compound acquisition is incomplete.

use super::*;

#[derive(Clone, Copy)]
pub(super) enum RankOwnerV1 {
    Kv(SlotKeyV1),
    State(SlotKeyV1),
}

#[derive(Clone, Copy)]
pub(super) enum CompoundCleanupV1 {
    Ranks(RankOwnerV1),
    Arenas,
    ServingKv,
    LogitsRead,
}

pub(super) struct RankAllocationV1 {
    pub buffers: Vec<SlotKeyV1>,
    pub host_rank_user: bool,
    pub complete: bool,
}

impl RankAllocationV1 {
    pub fn new(ranks: usize) -> HostServiceResultV1<Self> {
        let mut buffers = Vec::new();
        buffers
            .try_reserve_exact(ranks)
            .map_err(|_| BundleStatusV1::ResourceExhausted)?;
        Ok(Self {
            buffers,
            host_rank_user: false,
            complete: false,
        })
    }
}

#[derive(Default)]
pub(super) struct ConstructionRankV1 {
    pub buffer: Option<SlotKeyV1>,
    pub stream: Option<DeviceObjectV1>,
}

impl<D: HostDeviceBackendV1> HostStateV1<D> {
    pub(super) fn rank_allocation_mut(
        &mut self,
        owner: RankOwnerV1,
    ) -> HostServiceResultV1<&mut RankAllocationV1> {
        Ok(match owner {
            RankOwnerV1::Kv(key) => &mut self.kv.get_mut(key)?.allocation,
            RankOwnerV1::State(key) => &mut self.states.get_mut(key)?.allocation,
        })
    }

    pub(super) fn release_rank_allocation(
        &mut self,
        owner: RankOwnerV1,
    ) -> HostServiceResultV1<()> {
        let order = match owner {
            RankOwnerV1::Kv(key) => self.kv.get(key)?.order,
            RankOwnerV1::State(key) => self.states.get(key)?.order,
        };
        self.order.preflight(order)?;
        self.rank_allocation_mut(owner)?.complete = false;
        while let Some(buffer) = self.rank_allocation_mut(owner)?.buffers.last().copied() {
            self.release_registered_host_buffer(buffer)?;
            self.rank_allocation_mut(owner)?.buffers.pop();
        }
        if self.rank_allocation_mut(owner)?.host_rank_user {
            self.drop_host_rank_user()?;
            self.rank_allocation_mut(owner)?.host_rank_user = false;
        } else if self.host_rank_users == 0 {
            // Construction may have stopped before acquiring its first user.
            self.release_host_ranks()?;
        }
        self.order.release(order)?;
        match owner {
            RankOwnerV1::Kv(key) => {
                self.kv.remove(key)?;
            }
            RankOwnerV1::State(key) => {
                self.states.remove(key)?;
            }
        }
        Ok(())
    }

    pub(super) fn release_arena(&mut self, key: SlotKeyV1) -> HostServiceResultV1<()> {
        let record = self.arenas.get(key)?;
        let (order, buffer, host_buffer) = (record.order, record.buffer, record.host_buffer);
        self.order.preflight(order)?;
        self.arenas.get_mut(key)?.complete = false;
        if let Some(buffer) = host_buffer {
            self.release_registered_host_buffer(buffer)?;
            self.arenas.get_mut(key)?.host_buffer = None;
        }
        if let Some(buffer) = buffer {
            self.release_registered_host_buffer(buffer)?;
            self.arenas.get_mut(key)?.buffer = None;
        }
        self.order.release(order)?;
        self.arenas.remove(key)?;
        Ok(())
    }

    /// A failed call has no public handle. Keep its private owner until every
    /// cleanup phase succeeds, and report a cleanup refusal ahead of its cause.
    pub(super) fn fail_acquisition(
        &mut self,
        owner: CompoundCleanupV1,
        cause: BundleStatusV1,
    ) -> BundleStatusV1 {
        debug_assert!(self.pending_compound_cleanup.is_none());
        self.pending_compound_cleanup = Some(owner);
        self.retry_acquisition_cleanup().err().unwrap_or(cause)
    }

    pub(super) fn retry_acquisition_cleanup(&mut self) -> HostServiceResultV1<()> {
        // The rejected buffer was created last and may depend on the compound
        // owner's streams. Drain it before those dependencies.
        self.retry_buffer_cleanup()?;
        let Some(owner) = self.pending_compound_cleanup else {
            return Ok(());
        };
        match owner {
            CompoundCleanupV1::Ranks(owner) => self.release_rank_allocation(owner)?,
            CompoundCleanupV1::Arenas => {
                while let Some(key) = self.unpublished_arenas.last().copied() {
                    self.release_arena(key)?;
                    self.unpublished_arenas.pop();
                }
            }
            CompoundCleanupV1::ServingKv => {
                while let Some(rank) = self.serving_kv_construction.last() {
                    if let Some(buffer) = rank.buffer {
                        self.release_registered_host_buffer(buffer)?;
                        self.serving_kv_construction.last_mut().unwrap().buffer = None;
                    }
                    if let Some(stream) = self.serving_kv_construction.last().unwrap().stream {
                        self.device.release_stream(stream)?;
                        self.serving_kv_construction.last_mut().unwrap().stream = None;
                    }
                    self.serving_kv_construction.pop();
                }
            }
            CompoundCleanupV1::LogitsRead => {
                if let Some(stream) = self.logits_read_stream {
                    self.device.release_stream(stream)?;
                    self.logits_read_stream = None;
                }
            }
        }
        self.pending_compound_cleanup = None;
        Ok(())
    }
}
