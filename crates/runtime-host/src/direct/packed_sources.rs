//! Bounded input reuse between packed records in concurrent rank reads.
use super::*;
use std::collections::{BTreeSet, VecDeque};

/// Idle allocations have their own small cap in addition to the shared-byte
/// budget. Active temporaries are held only by the bounded worker tasks.
pub(super) struct PackedBuffers {
    limit: usize,
    state: Mutex<Buffers>,
    timing: Option<BufferCounters>,
}

#[derive(Default)]
struct BufferCounters {
    allocations: AtomicU64,
    allocated_bytes: AtomicU64,
    reuse_hits: AtomicU64,
    discarded_bytes: AtomicU64,
    take_nanos: AtomicU64,
}

/// Optional allocation observations of the packed-buffer pool. Other
/// allocations and active transaction outputs are not counted here.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct BufferTiming {
    pub allocations: u64,
    pub allocated_bytes: u64,
    pub reuse_hits: u64,
    pub discarded_bytes: u64,
    pub take_seconds: f64,
}

impl BufferTiming {
    pub(super) fn since(self, before: Self) -> Self {
        Self {
            allocations: self.allocations.saturating_sub(before.allocations),
            allocated_bytes: self.allocated_bytes.saturating_sub(before.allocated_bytes),
            reuse_hits: self.reuse_hits.saturating_sub(before.reuse_hits),
            discarded_bytes: self.discarded_bytes.saturating_sub(before.discarded_bytes),
            take_seconds: (self.take_seconds - before.take_seconds).max(0.0),
        }
    }
}

#[derive(Default)]
struct Buffers {
    buffers: Vec<Vec<u8>>,
    bytes: usize,
    peak: usize,
}

impl PackedBuffers {
    pub fn new(shared_bytes: u64) -> Self {
        Self {
            limit: shared_bytes.min(64 << 20) as usize,
            state: Mutex::default(),
            timing: detailed_timing_v1().then(BufferCounters::default),
        }
    }

    pub fn take(&self, length: usize) -> Vec<u8> {
        let start = self.timing.as_ref().map(|_| Instant::now());
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let found = state.buffers.iter().position(|b| b.capacity() == length);
        let mut bytes = if let Some(i) = found {
            let bytes = state.buffers.swap_remove(i);
            state.bytes -= bytes.capacity();
            bytes
        } else {
            Vec::new()
        };
        drop(state);
        bytes.resize(length, 0);
        if let (Some(timing), Some(start)) = (&self.timing, start) {
            elapsed_nanos(&timing.take_nanos, start);
            if found.is_some() {
                timing.reuse_hits.fetch_add(1, Ordering::Relaxed);
            } else if bytes.capacity() > 0 {
                timing.allocations.fetch_add(1, Ordering::Relaxed);
                timing
                    .allocated_bytes
                    .fetch_add(bytes.capacity() as u64, Ordering::Relaxed);
            }
        }
        bytes
    }

    pub fn give(&self, bytes: Vec<u8>) {
        if bytes.capacity() == 0 || bytes.capacity() > self.limit {
            self.discarded(bytes.capacity());
            return;
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.bytes + bytes.capacity() <= self.limit {
            state.bytes += bytes.capacity();
            state.peak = state.peak.max(state.bytes);
            state.buffers.push(bytes);
        } else {
            self.discarded(bytes.capacity());
        }
    }

    fn discarded(&self, capacity: usize) {
        if let Some(timing) = &self.timing {
            timing
                .discarded_bytes
                .fetch_add(capacity as u64, Ordering::Relaxed);
        }
    }

    pub fn timing(&self) -> Option<BufferTiming> {
        self.timing.as_ref().map(|timing| BufferTiming {
            allocations: timing.allocations.load(Ordering::Relaxed),
            allocated_bytes: timing.allocated_bytes.load(Ordering::Relaxed),
            reuse_hits: timing.reuse_hits.load(Ordering::Relaxed),
            discarded_bytes: timing.discarded_bytes.load(Ordering::Relaxed),
            take_seconds: seconds(&timing.take_nanos),
        })
    }

    pub fn clear(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.buffers.clear();
        state.bytes = 0;
    }

    pub fn peak(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .peak as u64
    }
}

struct Slot {
    consumers: usize,
    state: Mutex<SourceState>,
}

#[derive(Default)]
struct SourceState {
    bytes: Option<Arc<RetainedBytesV1>>,
    seen: BTreeSet<usize>,
}

pub(super) struct PackedSources {
    slots: Vec<Slot>,
    /// Slots with retained inputs, oldest first. Slots can appear again after
    /// a reload; eviction tolerates entries whose input has already drained.
    retained: Mutex<VecDeque<usize>>,
}

impl PackedSources {
    pub fn new(ranks: &mut [RankV1]) -> Self {
        let mut keys = BTreeMap::new();
        let mut users: Vec<BTreeSet<usize>> = Vec::new();
        for (consumer, record) in ranks
            .iter_mut()
            .flat_map(|rank| &mut rank.records)
            .enumerate()
        {
            if !matches!(
                record.prepared.record().transform,
                WeightTransform::PackedOperand(_)
            ) {
                continue;
            }
            record.consumer = consumer;
            for source in &record.sources {
                if source.length > SHARED_SOURCE_BYTES_MAX {
                    record.packed_sources.push(None);
                    continue;
                }
                // Sections are separate cohorts: scale-first views must not
                // retain codes until every later code view has been consumed.
                let key = (
                    source.shard,
                    source.begin,
                    source.length,
                    record.prepared.section(),
                );
                let slot = *keys.entry(key).or_insert_with(|| {
                    users.push(BTreeSet::new());
                    users.len() - 1
                });
                users[slot].insert(consumer);
                record.packed_sources.push(Some(slot));
            }
        }
        Self {
            slots: users
                .into_iter()
                .map(|users| Slot {
                    consumers: users.len(),
                    state: Mutex::default(),
                })
                .collect(),
            retained: Mutex::default(),
        }
    }

    /// Eviction never waits for a slot another worker is reading. A worker
    /// unable to reserve falls back to its own input, so no budget wait can
    /// hold the other ranks' current upload wave indefinitely.
    fn reserve(&self, inner: &InnerV1, bytes: u64) -> bool {
        if reserve_v1(inner, bytes) {
            return true;
        }
        let attempts = self
            .retained
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len();
        for _ in 0..attempts {
            let index = self
                .retained
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .pop_front();
            let Some(index) = index else { break };
            if let Ok(mut state) = self.slots[index].state.try_lock() {
                if let Some(bytes) = state.bytes.take() {
                    release_retained_v1(inner, bytes);
                }
                state.seen.clear();
            } else {
                self.retained
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push_back(index);
            }
            if reserve_v1(inner, bytes) {
                return true;
            }
        }
        false
    }

    pub fn read<'a>(
        &self,
        inner: &'a InnerV1,
        record: &RecordV1,
        index: usize,
    ) -> Result<Input<'a>, WeightTransformError> {
        let source = record.sources[index];
        if let Some(slot_index) = record.packed_sources.get(index).copied().flatten() {
            let slot = &self.slots[slot_index];
            let waiting = inner.detailed_timing.then(Instant::now);
            let mut state = slot
                .state
                .lock()
                .map_err(|_| WeightTransformError("shared packed source poisoned".into()))?;
            if let Some(waiting) = waiting {
                elapsed_nanos(&inner.input_wait_nanos, waiting);
            }
            let bytes = if let Some(bytes) = &state.bytes {
                inner.source_cache_hits.fetch_add(1, Ordering::Relaxed);
                inner
                    .source_cache_bytes
                    .fetch_add(source.length, Ordering::Relaxed);
                Some(Arc::clone(bytes))
            } else if slot.consumers > 1 && self.reserve(inner, source.length) {
                let mut data = inner.packed_buffers.take(source.length as usize);
                if data.capacity() as u64 != source.length {
                    inner.retained.fetch_sub(source.length, Ordering::AcqRel);
                    read_source_v1(inner, source, 0, &mut data)?;
                    return Ok(Input {
                        inner,
                        bytes: Some(InputBytes::Own(data)),
                    });
                }
                if let Err(error) = read_source_v1(inner, source, 0, &mut data) {
                    inner.packed_buffers.give(data);
                    inner.retained.fetch_sub(source.length, Ordering::AcqRel);
                    return Err(error);
                }
                let bytes = Arc::new(RetainedBytesV1::pooled(inner, data));
                state.seen.clear();
                state.bytes = Some(Arc::clone(&bytes));
                self.retained
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push_back(slot_index);
                Some(bytes)
            } else {
                None
            };
            if let Some(bytes) = bytes {
                state.seen.insert(record.consumer);
                if state.seen.len() == slot.consumers {
                    if let Some(done) = state.bytes.take() {
                        release_retained_v1(inner, done);
                    }
                    state.seen.clear();
                    self.retained
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .retain(|index| *index != slot_index);
                }
                return Ok(Input {
                    inner,
                    bytes: Some(InputBytes::Shared(bytes)),
                });
            }
        }
        let mut bytes = inner.packed_buffers.take(source.length as usize);
        read_source_v1(inner, source, 0, &mut bytes)?;
        Ok(Input {
            inner,
            bytes: Some(InputBytes::Own(bytes)),
        })
    }

    pub fn clear(&self, inner: &InnerV1) {
        for slot in &self.slots {
            let mut state = slot
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(bytes) = state.bytes.take() {
                release_retained_v1(inner, bytes);
            }
            state.seen.clear();
        }
        self.retained
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }
}

enum InputBytes {
    Shared(Arc<RetainedBytesV1>),
    Own(Vec<u8>),
}

pub(super) struct Input<'a> {
    inner: &'a InnerV1,
    bytes: Option<InputBytes>,
}
impl Input<'_> {
    pub fn bytes(&self) -> &[u8] {
        match self.bytes.as_ref().expect("input lives until dropped") {
            InputBytes::Shared(bytes) => &bytes.bytes,
            InputBytes::Own(bytes) => bytes,
        }
    }
}
impl Drop for Input<'_> {
    fn drop(&mut self) {
        match self.bytes.take().expect("input drops once") {
            InputBytes::Shared(bytes) => release_retained_v1(self.inner, bytes),
            InputBytes::Own(bytes) => self.inner.packed_buffers.give(bytes),
        }
    }
}
