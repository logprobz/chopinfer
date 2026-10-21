//! Host-private arenas and a bounded, polled copy queue.
use crate::device::{DeviceObjectV1, HostDeviceBackendV1, HostDeviceResultV1};
use runtime_bundle_abi::BundleStatusV1 as Status;
pub use runtime_engine::KvTransferStatus;
use runtime_kv::{
    KvTransferDirection, KvTransferId, KvTransferPlan, KvTransferPoll, KvTransferSubmission,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::{Duration, Instant};

pub const KV_OFFLOAD_CAPABILITY_V1: &str = "chop.kv-offload.v1";
const GIB: u64 = 1024 * 1024 * 1024;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvOffloadOptionsV1 {
    pub budget_bytes: u64,
    pub minimum_tokens: usize,
    /// Decimal bytes per second per link; None means uncapped.
    pub bytes_per_second: Option<u64>,
}
impl Default for KvOffloadOptionsV1 {
    fn default() -> Self {
        Self {
            budget_bytes: 0,
            minimum_tokens: 64,
            bytes_per_second: None,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostArenaPlanV1 {
    pub pages_per_rank: usize,
    pub page_bytes: usize,
    pub rank_count: usize,
    pub allocated_bytes: u64,
}
impl HostArenaPlanV1 {
    pub fn new(budget: u64, available: u64, page_bytes: u64, ranks: usize) -> Result<Self, String> {
        if budget == 0 || page_bytes == 0 || page_bytes > GIB || ranks == 0 {
            return Err("invalid host KV arena geometry".into());
        }
        preflight_host_memory(budget, available)?;
        let pages = budget / ranks as u64 / page_bytes;
        if pages == 0 {
            return Err(format!(
                "host KV budget {budget} bytes is smaller than one {page_bytes}-byte page per rank ({ranks} ranks)"
            ));
        }
        if pages > runtime_kv::MAX_BLOCK_CAPACITY as u64 {
            return Err("host KV page capacity exceeds the accounting bound".into());
        }
        Ok(Self {
            pages_per_rank: pages as usize,
            page_bytes: page_bytes as usize,
            rank_count: ranks,
            allocated_bytes: pages * page_bytes * ranks as u64,
        })
    }
    pub fn buffer_pages(&self) -> Vec<usize> {
        let per = GIB as usize / self.page_bytes;
        let mut remaining = self.pages_per_rank;
        let mut result = Vec::new();
        while remaining > 0 {
            let n = remaining.min(per);
            result.push(n);
            remaining -= n;
        }
        result
    }
}
/// Admission preflight runs before model load and again immediately before pinning.
pub fn preflight_host_memory(budget: u64, available: u64) -> Result<(), String> {
    if budget != 0 && budget.checked_add(16 * GIB).is_none_or(|n| n > available) {
        return Err(format!(
            "host KV budget {budget} bytes plus reserve {} bytes exceeds MemAvailable {available} bytes; reduce --kv-offload-gib and check page-locked memory limits (docs/developer-preview.md)",
            16 * GIB
        ));
    }
    Ok(())
}
pub fn allocation_enabled(budget: u64, capability: bool, startup_probe: bool) -> bool {
    budget != 0 && (capability || startup_probe)
}

pub fn mem_available_v1(input: &str) -> Result<u64, String> {
    let line = input
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))
        .ok_or("MemAvailable is missing")?;
    let fields: Vec<_> = line.split_whitespace().collect();
    if fields.len() != 3 || fields[2] != "kB" {
        return Err("invalid MemAvailable units".into());
    }
    fields[1]
        .parse::<u64>()
        .ok()
        .and_then(|n| n.checked_mul(1024))
        .ok_or_else(|| "invalid MemAvailable value".into())
}

/// The executor delegates copies here; logical page ownership stays in HostTier.
pub trait KvOffloadExecutorV1 {
    fn available(&self) -> bool;
    fn submit(&self, plan: KvTransferPlan) -> Result<(), (Status, KvTransferPlan)>;
    fn poll(&self, id: KvTransferId) -> HostDeviceResultV1<KvTransferPoll>;
    fn wait(&self, id: KvTransferId) -> HostDeviceResultV1<KvTransferPoll>;
    fn drain_timeout(&self) -> Duration {
        Duration::from_secs(5)
    }
    fn drain(&self) -> HostDeviceResultV1<()>;
}
struct Queued {
    plan: KvTransferSubmission,
    cursor: usize,
    status: KvTransferStatus,
}
struct Chunk {
    bytes: usize,
    transfer: KvTransferId,
    ranks: Vec<Option<u64>>,
}
pub(crate) struct OffloadQueue {
    pub plan: HostArenaPlanV1,
    arenas: Vec<DeviceObjectV1>,
    device_pages: usize,
    transfers: BTreeMap<KvTransferId, Queued>,
    in_flight: VecDeque<Chunk>,
    rate: Option<u64>,
    next_issue: Instant,
    poisoned: bool,
}
impl OffloadQueue {
    pub fn new<D: HostDeviceBackendV1>(
        device: &mut D,
        buffers: &[DeviceObjectV1],
        device_pages: usize,
        plan: HostArenaPlanV1,
        rate: Option<u64>,
    ) -> HostDeviceResultV1<Self> {
        if buffers.len() != plan.rank_count || rate == Some(0) {
            return Err(Status::InvalidArgument);
        }
        let mut arenas = Vec::new();
        for buffer in buffers {
            match device.create_offload_arena(*buffer, plan.page_bytes, plan.pages_per_rank) {
                Ok(arena) => arenas.push(arena),
                Err(e) => {
                    for arena in arenas {
                        device.release_offload_arena(arena)?;
                    }
                    return Err(e);
                }
            }
        }
        Ok(Self {
            plan,
            arenas,
            device_pages,
            transfers: BTreeMap::new(),
            in_flight: VecDeque::new(),
            rate,
            next_issue: Instant::now(),
            poisoned: false,
        })
    }
    pub fn drain_timeout(&self) -> Duration {
        let Some(rate) = self.rate else {
            return Duration::from_secs(5);
        };
        // Per-link pacing issues ranks together. Include unissued pages and
        // already-issued pacing debt, then allow another full paced interval
        // plus five seconds for event observation and queue scheduling.
        let bytes = self
            .transfers
            .values()
            .filter(|t| !t.plan.accepted())
            .fold(0u64, |sum, t| {
                sum.saturating_add(
                    (t.plan.pages().len().saturating_sub(t.cursor) as u64)
                        .saturating_mul(self.plan.page_bytes as u64),
                )
            });
        let paced = Duration::from_secs(bytes.div_ceil(rate))
            .saturating_add(self.next_issue.saturating_duration_since(Instant::now()));
        Duration::from_secs(5).saturating_add(paced.saturating_mul(2))
    }

    pub fn available(&self) -> bool {
        !self.poisoned
    }
    pub fn submit(&mut self, plan: KvTransferPlan) -> Result<(), (Status, KvTransferPlan)> {
        if self.poisoned || self.transfers.contains_key(&plan.id()) {
            return Err((Status::InvalidState, plan));
        }
        let mut hosts = BTreeSet::new();
        let mut devices = BTreeSet::new();
        for p in plan.pages() {
            if p.host.index() >= self.plan.pages_per_rank
                || p.device.index() >= self.device_pages
                || !hosts.insert(p.host)
                || !devices.insert(p.device)
            {
                return Err((Status::InvalidArgument, plan));
            }
        }
        self.transfers.insert(
            plan.id(),
            Queued {
                plan: plan.into_submission(),
                cursor: 0,
                status: KvTransferStatus::Pending,
            },
        );
        Ok(())
    }
    fn progress<D: HostDeviceBackendV1>(&mut self, device: &mut D) -> HostDeviceResultV1<()> {
        self.transfers.retain(|_, t| !t.plan.accepted());
        // Chunks share ordered copy streams. Stop querying at the oldest
        // pending chunk: later events cannot complete first on every rank.
        // A stalled head costs at most one query per rank, not the full window
        // again for each logical transfer polled by the engine.
        while let Some(chunk) = self.in_flight.front_mut() {
            for (rank, event) in chunk.ranks.iter_mut().enumerate() {
                if let Some(id) = *event
                    && device.query_offload_chunk(self.arenas[rank], id)?
                {
                    *event = None;
                }
            }
            if chunk.ranks.iter().any(Option::is_some) {
                break;
            }
            self.in_flight.pop_front();
        }
        let active_transfer = self.in_flight.front().map(|chunk| chunk.transfer);
        for (id, t) in &mut self.transfers {
            if t.status == KvTransferStatus::Pending
                && t.cursor == t.plan.pages().len()
                && active_transfer != Some(*id)
            {
                t.status = KvTransferStatus::Complete;
            }
        }
        let now = Instant::now();
        let horizon = now + Duration::from_millis(20);
        while self.in_flight.len() < 128
            && self.in_flight.iter().map(|c| c.bytes).sum::<usize>() < GIB as usize
            && (self.rate.is_none() || self.next_issue <= horizon)
        {
            // Do not interleave two transfers: they can name the same shared
            // pages. Once a chunk drains, waiting reloads take priority.
            let chosen = if let Some(chunk) = self.in_flight.front() {
                if self.transfers[&chunk.transfer].plan.direction() == KvTransferDirection::Out
                    && self.transfers.values().any(|t| {
                        t.status == KvTransferStatus::Pending
                            && t.plan.direction() == KvTransferDirection::In
                    })
                {
                    break;
                }
                Some(chunk.transfer)
            } else {
                self.transfers
                    .iter()
                    .filter(|(_, t)| t.status == KvTransferStatus::Pending)
                    .min_by_key(|(id, t)| (t.plan.direction() != KvTransferDirection::In, **id))
                    .map(|(id, _)| *id)
            };
            let Some(id) = chosen else { break };
            let t = self.transfers.get_mut(&id).unwrap();
            if t.cursor == t.plan.pages().len() {
                break;
            }
            let room = (GIB as usize - self.in_flight.iter().map(|c| c.bytes).sum::<usize>())
                / self.plan.page_bytes;
            if room == 0 {
                break;
            }
            let end = (t.cursor + (8 * 1024 * 1024 / self.plan.page_bytes).max(1).min(room))
                .min(t.plan.pages().len());
            let pages = &t.plan.pages()[t.cursor..end];
            let mut ranks = Vec::new();
            for arena in &self.arenas {
                ranks.push(Some(device.submit_offload_chunk(
                    *arena,
                    t.plan.direction(),
                    pages,
                )?));
            }
            self.in_flight.push_back(Chunk {
                bytes: pages.len() * self.plan.page_bytes,
                transfer: id,
                ranks,
            });
            if let Some(rate) = self.rate {
                self.next_issue = self.next_issue.max(now)
                    + Duration::from_secs_f64(
                        (pages.len() * self.plan.page_bytes) as f64 / rate as f64,
                    );
            }
            t.cursor = end;
        }
        Ok(())
    }
    pub fn poll<D: HostDeviceBackendV1>(
        &mut self,
        device: &mut D,
        id: KvTransferId,
    ) -> HostDeviceResultV1<KvTransferPoll> {
        if self.transfers.get(&id).is_none_or(|t| t.plan.accepted()) {
            return Err(Status::StaleHandle);
        }
        if !self.poisoned && self.progress(device).is_err() {
            self.poisoned = true;
            eprintln!("KV offload disabled after copy failure; later victims recompute");
            // Fence every rank even when an earlier rank fails. No references
            // can be returned unless every stream confirms completion.
            let mut idle = true;
            for arena in &self.arenas {
                idle &= device.fence_offload_arena(*arena).is_ok();
            }
            for t in self.transfers.values_mut() {
                if t.status == KvTransferStatus::Pending {
                    t.status = if idle {
                        KvTransferStatus::FailedIdle
                    } else {
                        KvTransferStatus::Unconfirmed
                    };
                }
            }
            if idle {
                self.in_flight.clear();
            }
        }
        let t = self.transfers.get(&id).ok_or(Status::StaleHandle)?;
        Ok(if t.status == KvTransferStatus::Pending {
            KvTransferPoll::Pending
        } else {
            KvTransferPoll::Finished(t.plan.completion(t.status))
        })
    }
    pub fn wait<D: HostDeviceBackendV1>(
        &mut self,
        device: &mut D,
        id: KvTransferId,
    ) -> HostDeviceResultV1<KvTransferPoll> {
        loop {
            let status = self.poll(device, id)?;
            if status.status() != KvTransferStatus::Pending {
                return Ok(status);
            }
            std::thread::sleep(Duration::from_micros(100));
        }
    }
    pub fn drain<D: HostDeviceBackendV1>(&mut self, device: &mut D) -> HostDeviceResultV1<()> {
        if self
            .transfers
            .values()
            .any(|t| t.status == KvTransferStatus::Unconfirmed)
        {
            return Err(Status::InternalError);
        }
        let ids: Vec<_> = self
            .transfers
            .iter()
            .filter(|(_, t)| !t.plan.accepted())
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            if self.wait(device, id)?.status() == KvTransferStatus::Unconfirmed {
                return Err(Status::InternalError);
            }
        }
        for arena in &self.arenas {
            device.fence_offload_arena(*arena)?;
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "fake-host-services"))]
mod tests {
    use super::*;
    use crate::device::DeviceRangeV1;
    use crate::fake::{FakeHostDeviceV1, FakeRefusalV1};
    use runtime_kv::*;
    fn setup(
        page_bytes: usize,
    ) -> (
        FakeHostDeviceV1,
        Vec<DeviceObjectV1>,
        OffloadQueue,
        KvPool,
        HostTier,
    ) {
        let mut device = FakeHostDeviceV1::new();
        let mut buffers = Vec::new();
        for rank in 0..2 {
            let context = device.create_context(rank, rank as i32).unwrap();
            let pool = device.create_memory_pool(context).unwrap();
            let stream = device.create_stream(context, 0).unwrap();
            let buffer = device
                .allocate_device_buffer(pool, Some(stream), (32 * page_bytes) as u64, 16)
                .unwrap();
            let bytes: Vec<_> = (0..32 * page_bytes)
                .map(|i| (i + rank as usize * 7) as u8)
                .collect();
            device
                .copy_host_to_device(
                    stream,
                    DeviceRangeV1 {
                        object: buffer,
                        offset: 0,
                        length: bytes.len() as u64,
                    },
                    &bytes,
                )
                .unwrap();
            buffers.push(buffer);
        }
        let plan =
            HostArenaPlanV1::new((64 * page_bytes) as u64, 32 * GIB, page_bytes as u64, 2).unwrap();
        let queue = OffloadQueue::new(&mut device, &buffers, 32, plan, None).unwrap();
        let pool = KvPool::new(KvGeometry::new(4, 32).unwrap(), 8).unwrap();
        let host = HostTier::new(&pool, plan.pages_per_rank).unwrap();
        (device, buffers, queue, pool, host)
    }
    fn capture(
        p: &mut KvPool,
        h: &mut HostTier,
        n: u8,
        tokens: usize,
    ) -> (HostAnchorId, KvTransferPlan) {
        let seq = p.create_sequence(128).unwrap();
        p.append(seq, tokens).unwrap();
        let key = PrefixCacheKeyV1::new([n; 32]);
        p.insert_private_anchor(key, seq, 2).unwrap();
        let pair = h.begin_offload(p, key, Some(u64::from(n))).unwrap();
        p.remove_anchor(key).unwrap();
        p.release(seq).unwrap();
        pair
    }
    #[test]
    fn a_pending_head_does_not_query_the_whole_chunk_window() {
        let (mut device, _, mut queue, mut pool, mut host) = setup(1024 * 1024);
        let (_, plan) = capture(&mut pool, &mut host, 1, 100);
        let id = plan.id();
        device.set_offload_query_delay(queue.arenas[1], 1000);
        queue.submit(plan).unwrap();
        assert_eq!(
            queue.poll(&mut device, id).unwrap().status(),
            KvTransferStatus::Pending
        );
        assert!(queue.in_flight.len() > 1);
        // A third event query would inject failure. Only the oldest chunk's
        // two rank events should be queried while its second rank is pending.
        device.refuse_after(FakeRefusalV1::QueryOffload, 2);
        assert_eq!(
            queue.poll(&mut device, id).unwrap().status(),
            KvTransferStatus::Pending
        );
        assert!(queue.available());
        device.allow(FakeRefusalV1::QueryOffload);
    }

    #[test]
    fn shutdown_budget_includes_rate_limited_backlog() {
        let (_, _, mut queue, mut pool, mut host) = setup(1024 * 1024);
        assert_eq!(queue.drain_timeout(), Duration::from_secs(5));
        queue.rate = Some(1024 * 1024);
        let (_, plan) = capture(&mut pool, &mut host, 1, 40);
        let bytes = plan.pages().len() * queue.plan.page_bytes;
        queue.submit(plan).unwrap();
        assert!(
            queue.drain_timeout()
                >= Duration::from_secs(5 + 2 * bytes as u64 / queue.rate.unwrap())
        );
        assert!(queue.drain_timeout() < Duration::from_secs(60));
    }

    #[test]
    fn reload_waits_for_the_outgoing_chunk_then_preempts_its_remaining_pages() {
        let (mut device, _, mut q, mut p, mut h) = setup(1024 * 1024);
        let (anchor, first) = capture(&mut p, &mut h, 1, 4);
        let id = first.id();
        q.submit(first).unwrap();
        h.complete_transfer(
            &mut p,
            q.wait(&mut device, id).unwrap().completion().unwrap(),
        )
        .unwrap();
        let (out_anchor, out) = capture(&mut p, &mut h, 2, 100);
        let out_id = out.id();
        q.rate = Some(1);
        device.set_offload_query_delay(q.arenas[1], 2);
        q.submit(out).unwrap();
        q.poll(&mut device, out_id).unwrap();
        assert_eq!(q.transfers[&out_id].cursor, 8);
        let dest = p.create_initialized_sequence(4, 20, 2).unwrap();
        let reload = h.begin_reload(&mut p, anchor, &dest).unwrap();
        let reload_id = reload.id();
        q.submit(reload).unwrap();
        q.rate = None;
        let mut saw_reload = false;
        for _ in 0..12 {
            let poll = q.poll(&mut device, reload_id).unwrap();
            let ids: BTreeSet<_> = q.in_flight.iter().map(|c| c.transfer).collect();
            assert!(ids.len() <= 1, "two transfers overlapped");
            if ids.contains(&reload_id) {
                saw_reload = true;
                assert_eq!(q.transfers[&out_id].cursor, 8);
            }
            if let Some(done) = poll.completion() {
                h.complete_transfer(&mut p, done).unwrap();
                break;
            }
        }
        assert!(saw_reload);
        h.complete_transfer(
            &mut p,
            q.wait(&mut device, out_id).unwrap().completion().unwrap(),
        )
        .unwrap();
        p.finish_initialized_restore(dest.handle()).unwrap();
        p.release(dest.handle()).unwrap();
        h.drop_anchor(anchor).unwrap();
        h.drop_anchor(out_anchor).unwrap();
        h.audit(&p).unwrap();
        assert_eq!(p.stats().free_blocks(), 32);
        assert_eq!(h.used_pages(), 0);
    }
    #[test]
    fn completion_can_be_repolled_until_the_tier_accepts_it() {
        let (mut device, _, mut q, mut p, mut h) = setup(16);
        let (anchor, plan) = capture(&mut p, &mut h, 1, 4);
        let id = plan.id();
        q.submit(plan).unwrap();
        let token = q.wait(&mut device, id).unwrap().completion().unwrap();
        let seq = p.create_sequence(8).unwrap();
        let prepared = p.prepare_append_batch(&[(seq, 1)]).unwrap();
        assert!(h.complete_transfer(&mut p, token).is_err());
        assert!(q.transfers.contains_key(&id));
        p.rollback_prepared(prepared.id()).unwrap();
        h.complete_transfer(
            &mut p,
            q.poll(&mut device, id).unwrap().completion().unwrap(),
        )
        .unwrap();
        assert!(q.poll(&mut device, id).is_err());
        h.drop_anchor(anchor).unwrap();
        p.release(seq).unwrap();
        h.audit(&p).unwrap();
        assert_eq!(p.stats().free_blocks(), 32);
    }
    #[test]
    fn startup_preflight_reports_bytes_and_skips_unsupported_bundles() {
        let message = preflight_host_memory(4 * GIB, 17 * GIB).unwrap_err();
        for part in ["4294967296", "17179869184", "18253611008", "page-locked"] {
            assert!(message.contains(part), "{message}");
        }
        assert!(preflight_host_memory(0, 0).is_ok());
        assert!(!allocation_enabled(GIB, false, false));
        assert!(allocation_enabled(GIB, true, false));
        assert!(allocation_enabled(GIB, false, true));
        assert!(!allocation_enabled(0, true, true));
    }
    #[test]
    fn arena_geometry_and_memory_reserve_fail_closed() {
        let plan = HostArenaPlanV1::new(5 * GIB, 21 * GIB, 835_584, 4).unwrap();
        assert!(plan.allocated_bytes <= 5 * GIB);
        assert_eq!(
            plan.buffer_pages().iter().sum::<usize>(),
            plan.pages_per_rank
        );
        assert!(
            plan.buffer_pages()
                .iter()
                .all(|n| *n * plan.page_bytes <= GIB as usize)
        );
        for (budget, free, page, ranks) in [
            (GIB, 16 * GIB, 16, 4),
            (u64::MAX, u64::MAX, 16, 4),
            (GIB, 32 * GIB, 0, 4),
            (GIB, 32 * GIB, GIB + 1, 4),
            (GIB, 32 * GIB, 16, 0),
            (1, 32 * GIB, 16, 4),
        ] {
            assert!(HostArenaPlanV1::new(budget, free, page, ranks).is_err());
        }
        assert_eq!(
            mem_available_v1("MemTotal: 999 kB\nMemAvailable: 123 kB\n").unwrap(),
            123 * 1024
        );
        for input in [
            "",
            "MemAvailable: 2 MB",
            "MemAvailable: -1 kB",
            "MemAvailable: 18446744073709551615 kB",
        ] {
            assert!(mem_available_v1(input).is_err());
        }
    }
    #[test]
    fn fake_copy_waits_for_every_rank_and_restores_exact_bytes_under_swaps() {
        let (mut device, buffers, mut q, mut p, mut h) = setup(16);
        device.set_offload_query_delay(q.arenas[1], 5);
        for n in 0..200u8 {
            let (anchor, out) = capture(&mut p, &mut h, n, 9);
            let expected: Vec<Vec<_>> = buffers
                .iter()
                .map(|b| {
                    out.pages()
                        .iter()
                        .map(|pair| {
                            device.buffer_bytes(*b).unwrap()
                                [pair.device.index() * 16..pair.device.index() * 16 + 16]
                                .to_vec()
                        })
                        .collect()
                })
                .collect();
            let out_id = out.id();
            q.submit(out).unwrap();

            for _ in 0..4 {
                assert_eq!(
                    q.poll(&mut device, out_id).unwrap().status(),
                    KvTransferStatus::Pending
                );
                assert!(!h.anchor(anchor).unwrap().ready);
                h.audit(&p).unwrap();
            }
            assert_eq!(
                q.wait(&mut device, out_id).unwrap().status(),
                KvTransferStatus::Complete
            );
            h.complete_transfer(
                &mut p,
                q.wait(&mut device, out_id).unwrap().completion().unwrap(),
            )
            .unwrap();
            let dest = p.create_initialized_sequence(9, 20, 2).unwrap();
            let reload = h.begin_reload(&mut p, anchor, &dest).unwrap();
            let reload_id = reload.id();
            let reload_pages = reload.pages().to_vec();
            q.submit(reload).unwrap();
            assert_eq!(
                q.wait(&mut device, reload_id).unwrap().status(),
                KvTransferStatus::Complete
            );
            for (rank, b) in buffers.iter().enumerate() {
                for (i, pair) in reload_pages.iter().enumerate() {
                    assert_eq!(
                        &device.buffer_bytes(*b).unwrap()
                            [pair.device.index() * 16..pair.device.index() * 16 + 16],
                        &expected[rank][i]
                    );
                }
            }
            h.complete_transfer(
                &mut p,
                q.wait(&mut device, reload_id)
                    .unwrap()
                    .completion()
                    .unwrap(),
            )
            .unwrap();
            h.drop_anchor(anchor).unwrap();
            p.finish_initialized_restore(dest.handle()).unwrap();
            p.release(dest.handle()).unwrap();
            h.audit(&p).unwrap();
            assert_eq!(h.used_pages(), 0);
            assert_eq!(p.stats().free_blocks(), 32);
        }
        q.drain(&mut device).unwrap();
    }
    #[test]
    fn chunk_window_rate_cap_and_partial_submission_failure() {
        let (mut device, _, mut q, mut p, mut h) = setup(1024 * 1024);
        let (a, out) = capture(&mut p, &mut h, 1, 100);
        let out_id = out.id();
        q.submit(out).unwrap();
        q.poll(&mut device, out_id).unwrap();
        assert_eq!(q.in_flight.len(), 4);
        assert_eq!(q.transfers[&out_id].cursor, 27);
        assert_eq!(
            q.wait(&mut device, out_id).unwrap().status(),
            KvTransferStatus::Complete
        );
        h.complete_transfer(
            &mut p,
            q.wait(&mut device, out_id).unwrap().completion().unwrap(),
        )
        .unwrap();
        h.drop_anchor(a).unwrap();
        let (a, out) = capture(&mut p, &mut h, 2, 4);
        q.rate = Some(1);
        let out_id = out.id();
        q.submit(out).unwrap();
        q.poll(&mut device, out_id).unwrap();
        assert_eq!(q.in_flight.len(), 1);
        assert!(q.next_issue > Instant::now());
        q.rate = None;
        q.next_issue = Instant::now();
        assert_eq!(
            q.wait(&mut device, out_id).unwrap().status(),
            KvTransferStatus::Complete
        );
        h.complete_transfer(
            &mut p,
            q.wait(&mut device, out_id).unwrap().completion().unwrap(),
        )
        .unwrap();
        h.drop_anchor(a).unwrap();
        let (_, out) = capture(&mut p, &mut h, 3, 4);
        let out_id = out.id();
        q.submit(out).unwrap();
        device.refuse_after(FakeRefusalV1::SubmitOffload, 1);
        assert_eq!(
            q.poll(&mut device, out_id).unwrap().status(),
            KvTransferStatus::FailedIdle
        );
        h.complete_transfer(
            &mut p,
            q.wait(&mut device, out_id).unwrap().completion().unwrap(),
        )
        .unwrap();
        assert!(q.poll(&mut device, out_id).is_err());
        assert!(!q.available());
        h.audit(&p).unwrap();
        assert_eq!(p.stats().free_blocks(), 32);
    }
    #[test]
    fn unconfirmed_copy_is_never_reported_complete() {
        let (mut device, _, mut q, mut p, mut h) = setup(16);
        let (a, out) = capture(&mut p, &mut h, 1, 9);
        let out_id = out.id();
        q.submit(out).unwrap();
        q.poll(&mut device, out_id).unwrap();
        device.refuse(FakeRefusalV1::QueryOffload);
        device.refuse(FakeRefusalV1::SynchronizeStream);
        assert_eq!(
            q.poll(&mut device, out_id).unwrap().status(),
            KvTransferStatus::Unconfirmed
        );
        h.complete_transfer(
            &mut p,
            q.poll(&mut device, out_id).unwrap().completion().unwrap(),
        )
        .unwrap();
        h.drop_anchor(a).unwrap();
        assert!(q.drain(&mut device).is_err());
        h.audit(&p).unwrap();
        assert_eq!(h.used_pages(), 5);
        assert_eq!(p.stats().free_blocks(), 27);
    }
}

#[cfg(all(test, feature = "fake-host-services"))]
mod storm {
    use super::*;
    use crate::device::DeviceRangeV1;
    use crate::fake::FakeHostDeviceV1;
    use runtime_kv::*;
    const PB: usize = 16;
    const PAGES: usize = 32;
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }
    struct Env {
        device: FakeHostDeviceV1,
        buffers: Vec<DeviceObjectV1>,
        streams: Vec<DeviceObjectV1>,
    }
    impl Env {
        fn page(&self, rank: usize, page: BlockId) -> Vec<u8> {
            self.device.buffer_bytes(self.buffers[rank]).unwrap()
                [page.index() * PB..page.index() * PB + PB]
                .to_vec()
        }
        fn scribble(&mut self, page: BlockId, rng: &mut Rng) {
            for rank in 0..self.buffers.len() {
                let bytes: Vec<u8> = (0..PB).map(|_| rng.next() as u8).collect();
                self.device
                    .copy_host_to_device(
                        self.streams[rank],
                        DeviceRangeV1 {
                            object: self.buffers[rank],
                            offset: (page.index() * PB) as u64,
                            length: PB as u64,
                        },
                        &bytes,
                    )
                    .unwrap();
            }
        }
    }
    struct Swapped {
        anchor: HostAnchorId,
        tokens: usize,
        bytes: Vec<Vec<Vec<u8>>>, // rank -> page -> bytes
        out: Option<KvTransferId>,
        reload: Option<(KvTransferId, PrefixHitV1)>,
    }
    fn append(p: &mut KvPool, env: &mut Env, seq: SequenceHandle, n: usize, rng: &mut Rng) -> bool {
        let before = p.snapshot(seq).unwrap();
        if p.append(seq, n).is_err() {
            return false;
        }
        let after = p.snapshot(seq).unwrap();
        for (i, page) in after.blocks().iter().enumerate() {
            let changed = before.blocks().get(i) != Some(page);
            let tail_in_place = i + 1 == before.blocks().len() && before.tokens() % 4 != 0;
            if changed || tail_in_place || i >= before.blocks().len() {
                // A kernel writes these rows. A page any other holder can see
                // must never be written in place.
                if !changed {
                    assert_eq!(
                        p.block_ref_count(*page).unwrap(),
                        1,
                        "in-place write to a shared page"
                    );
                }
                env.scribble(*page, rng);
            }
        }
        true
    }
    fn run(seed: u64, host_pages: usize, delays: bool) {
        let mut rng = Rng(seed);
        let mut device = FakeHostDeviceV1::new();
        let (mut buffers, mut streams) = (Vec::new(), Vec::new());
        for rank in 0..2 {
            let context = device.create_context(rank, rank as i32).unwrap();
            let pool = device.create_memory_pool(context).unwrap();
            let stream = device.create_stream(context, 0).unwrap();
            buffers.push(
                device
                    .allocate_device_buffer(pool, Some(stream), (PAGES * PB) as u64, 16)
                    .unwrap(),
            );
            streams.push(stream);
        }
        let plan =
            HostArenaPlanV1::new((host_pages * PB * 2) as u64, 64 * GIB, PB as u64, 2).unwrap();
        let mut q = OffloadQueue::new(&mut device, &buffers, PAGES, plan, None).unwrap();
        let mut env = Env {
            device,
            buffers,
            streams,
        };
        let mut p = KvPool::new(KvGeometry::new(4, PAGES).unwrap(), 16).unwrap();
        let mut h = HostTier::new(&p, plan.pages_per_rank).unwrap();
        let mut live: Vec<SequenceHandle> = Vec::new();
        let mut swapped: Vec<Swapped> = Vec::new();
        let mut key = 0u64;
        let mut stats = [0usize; 8];
        for step in 0..4000 {
            if delays && step % 50 == 0 {
                for a in 0..2 {
                    let d = rng.below(4) as usize;
                    env.device.set_offload_query_delay(q.arenas[a], d);
                }
            }
            match rng.below(10) {
                0 | 1 => {
                    // New request, sometimes sharing a live prefix.
                    let seq = if !live.is_empty() && rng.below(2) == 0 {
                        let src = live[rng.below(live.len() as u64) as usize];
                        p.fork(src, 40).ok()
                    } else {
                        p.create_sequence(40).ok()
                    };
                    if let Some(seq) = seq {
                        let n = 1 + rng.below(9) as usize;
                        if append(&mut p, &mut env, seq, n, &mut rng)
                            || p.snapshot(seq).unwrap().tokens() > 0
                        {
                            live.push(seq);
                        } else {
                            p.release(seq).unwrap();
                        }
                    }
                }
                2 | 3 => {
                    if let Some(i) =
                        (!live.is_empty()).then(|| rng.below(live.len() as u64) as usize)
                    {
                        let seq = live[i];
                        let n = 1 + rng.below(5) as usize;
                        if !append(&mut p, &mut env, seq, n, &mut rng) {
                            stats[0] += 1;
                        }
                    }
                }
                4 | 5 => {
                    // Preempt a live request: private anchor, capture, offload.
                    if live.is_empty() {
                        continue;
                    }
                    let seq = live.swap_remove(rng.below(live.len() as u64) as usize);
                    key += 1;
                    let mut kb = [0u8; 32];
                    kb[..8].copy_from_slice(&key.to_le_bytes());
                    let k = PrefixCacheKeyV1::new(kb);
                    let tokens = p.snapshot(seq).unwrap().tokens();
                    if tokens == 0 {
                        p.release(seq).unwrap();
                        continue;
                    }
                    match p.insert_private_anchor(k, seq, 1).unwrap() {
                        Some(state) => {
                            for s in &state {
                                env.scribble(*s, &mut rng); // the capture writes state
                            }
                        }
                        None => {
                            p.release(seq).unwrap();
                            stats[1] += 1;
                            continue;
                        }
                    }
                    let (_, blocks, state) = p.anchor_pages(k).unwrap();
                    let bytes: Vec<Vec<Vec<u8>>> = (0..2)
                        .map(|r| {
                            blocks
                                .iter()
                                .chain(&state)
                                .map(|pg| env.page(r, *pg))
                                .collect()
                        })
                        .collect();
                    match h.begin_offload(&mut p, k, Some(key)) {
                        Ok((anchor, out)) => {
                            let out_id = out.id();
                            q.submit(out).unwrap();
                            swapped.push(Swapped {
                                anchor,
                                tokens,
                                bytes,
                                out: Some(out_id),
                                reload: None,
                            });
                            stats[2] += 1;
                        }
                        Err(_) => stats[3] += 1, // host full: recompute
                    }
                    p.remove_anchor(k).unwrap();
                    p.release(seq).unwrap();
                }
                6 => {
                    // Cancel a swapped request at any point.
                    if swapped.is_empty() {
                        continue;
                    }
                    let i = rng.below(swapped.len() as u64) as usize;
                    let s = &mut swapped[i];
                    let _ = h.drop_anchor(s.anchor);
                    if let Some((_, dest)) = &s.reload {
                        let _ = p.release(dest.handle());
                    }
                    s.tokens = 0; // cancelled marker
                    stats[4] += 1;
                }
                7 => {
                    // Resume a ready swapped request.
                    let ready: Vec<usize> = swapped
                        .iter()
                        .enumerate()
                        .filter(|(_, s)| s.tokens > 0 && s.out.is_none() && s.reload.is_none())
                        .map(|(i, _)| i)
                        .collect();
                    if ready.is_empty() {
                        continue;
                    }
                    let i = ready[rng.below(ready.len() as u64) as usize];
                    let tokens = swapped[i].tokens;
                    let Ok(dest) = p.create_initialized_sequence(tokens, 40, 1) else {
                        stats[5] += 1;
                        continue;
                    };
                    let reload = h.begin_reload(&mut p, swapped[i].anchor, &dest).unwrap();
                    let reload_id = reload.id();
                    q.submit(reload).unwrap();
                    swapped[i].reload = Some((reload_id, dest));
                }
                _ => {
                    // Step start: poll everything, sometimes block on one.
                    let block = rng.below(4) == 0;
                    let mut k = 0;
                    while k < swapped.len() {
                        let id = swapped[k].out.or(swapped[k].reload.as_ref().map(|r| r.0));
                        let Some(id) = id else {
                            if swapped[k].tokens == 0 {
                                swapped.swap_remove(k);
                            } else {
                                k += 1;
                            }
                            continue;
                        };
                        let status = if block {
                            q.wait(&mut env.device, id).unwrap()
                        } else {
                            q.poll(&mut env.device, id).unwrap()
                        };
                        match status.status() {
                            KvTransferStatus::Pending => {}
                            KvTransferStatus::Complete => {
                                h.complete_transfer(
                                    &mut p,
                                    q.wait(&mut env.device, id).unwrap().completion().unwrap(),
                                )
                                .unwrap();
                                if swapped[k].out == Some(id) {
                                    swapped[k].out = None;
                                } else {
                                    let (_, dest) = swapped[k].reload.take().unwrap();
                                    if swapped[k].tokens > 0 {
                                        let snap = p.snapshot(dest.handle()).unwrap();
                                        for r in 0..2 {
                                            let got: Vec<_> = snap
                                                .blocks()
                                                .iter()
                                                .chain(dest.state())
                                                .map(|pg| env.page(r, *pg))
                                                .collect();
                                            assert_eq!(
                                                got, swapped[k].bytes[r],
                                                "seed {seed} step {step}: reload bytes differ"
                                            );
                                        }
                                        p.finish_initialized_restore(dest.handle()).unwrap();
                                        h.drop_anchor(swapped[k].anchor).unwrap();
                                        live.push(dest.handle());
                                        swapped[k].tokens = 0;
                                        stats[6] += 1;
                                    }
                                }
                            }
                            other => panic!("unexpected {other:?}"),
                        }
                        k += 1;
                    }
                }
            }
            h.audit(&p).unwrap();
            // Pool pages = live + in-flight; never below zero or leaking.
        }
        // Idle: drain, cancel all, release all.
        for s in &swapped {
            for id in s.out.into_iter().chain(s.reload.as_ref().map(|r| r.0)) {
                assert_eq!(
                    q.wait(&mut env.device, id).unwrap().status(),
                    KvTransferStatus::Complete
                );
                h.complete_transfer(
                    &mut p,
                    q.wait(&mut env.device, id).unwrap().completion().unwrap(),
                )
                .unwrap();
            }
            let _ = h.drop_anchor(s.anchor);
            if let Some((_, dest)) = &s.reload {
                let _ = p.release(dest.handle());
            }
        }
        for seq in live {
            p.release(seq).unwrap();
        }
        q.drain(&mut env.device).unwrap();
        h.audit(&p).unwrap();
        assert_eq!(h.used_pages(), 0, "host pages leaked");
        assert_eq!(h.pending_transfers(), 0);
        assert_eq!(p.stats().free_blocks(), PAGES, "device pages leaked");
        eprintln!("seed {seed} host {host_pages}: {stats:?}");
    }
    #[test]
    fn randomized_offload_reload_cancel_preserves_bytes_and_accounting() {
        for seed in 1..40 {
            run(seed, 24, seed % 2 == 0);
            run(seed * 7919, 8, true);
        }
    }
}
#[cfg(all(test, feature = "fake-host-services"))]
mod copy_window {
    use super::*;
    use crate::fake::FakeHostDeviceV1;
    use runtime_kv::*;
    #[test]
    fn a_full_transfer_is_queued_in_one_poll() {
        let pb = 1024 * 1024;
        let mut device = FakeHostDeviceV1::new();
        let mut buffers = Vec::new();
        for rank in 0..2 {
            let context = device.create_context(rank, rank as i32).unwrap();
            let pool = device.create_memory_pool(context).unwrap();
            let stream = device.create_stream(context, 0).unwrap();
            buffers.push(
                device
                    .allocate_device_buffer(pool, Some(stream), (128 * pb) as u64, 16)
                    .unwrap(),
            );
        }
        let plan = HostArenaPlanV1::new((256 * pb) as u64, 64 * GIB, pb as u64, 2).unwrap();
        let mut q = OffloadQueue::new(&mut device, &buffers, 128, plan, None).unwrap();
        for a in 0..2 {
            device.set_offload_query_delay(q.arenas[a], 0);
        }
        let mut p = KvPool::new(KvGeometry::new(4, 128).unwrap(), 8).unwrap();
        let mut h = HostTier::new(&p, plan.pages_per_rank).unwrap();
        let seq = p.create_sequence(512).unwrap();
        p.append(seq, 4 * 96).unwrap();
        let key = PrefixCacheKeyV1::new([9; 32]);
        p.insert_private_anchor(key, seq, 0).unwrap();
        let (_, out) = h.begin_offload(&mut p, key, Some(1)).unwrap();
        let out_id = out.id();
        q.submit(out).unwrap();
        let mut polls = 0;
        while q.poll(&mut device, out_id).unwrap().status() == KvTransferStatus::Pending {
            polls += 1;
        }
        assert_eq!(polls + 1, 2);
        h.complete_transfer(
            &mut p,
            q.poll(&mut device, out_id).unwrap().completion().unwrap(),
        )
        .unwrap();
    }
}
