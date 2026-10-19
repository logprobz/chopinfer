//! Admission reservations for bundle-owned per-request state.

use std::collections::BTreeMap;

use runtime_bundle::BundleManifestV1;
use runtime_engine::RequestId;

/// Slots are reserved before a request executes and reused only after retirement.
/// The bundle owns their bytes; this pool owns IDs and the memory reservation.
#[derive(Debug)]
pub struct StateSlotPoolV1 {
    capacity: usize,
    bytes_per_slot: u64,
    /// The bundle allocates every slot before it acquires serving KV.
    preallocated: bool,
    owners: BTreeMap<RequestId, u64>,
    /// Monotonic generations survive retirement and are bounded by slot capacity.
    generations: BTreeMap<u64, u64>,
}

impl StateSlotPoolV1 {
    #[cfg(test)]
    pub(crate) fn for_test(capacity: usize) -> Self {
        Self {
            capacity,
            bytes_per_slot: 128,
            preallocated: false,
            owners: BTreeMap::new(),
            generations: BTreeMap::new(),
        }
    }

    /// `capacity` is the most sequences the engine runs at once, and
    /// `maximum_verify_rows` the most rows one item executes before a commit.
    pub fn new(
        manifest: &BundleManifestV1,
        capacity: usize,
        maximum_verify_rows: usize,
    ) -> Result<Self, String> {
        let bytes = manifest
            .recurrent_state_bytes_per_slot()
            .map_err(|e| e.to_string())?;
        // One committed state plus a checkpoint after every tentative row.
        let copies = u64::try_from(maximum_verify_rows)
            .ok()
            .and_then(|n| n.checked_add(1))
            .ok_or("state checkpoint count overflow")?;
        let bytes_per_slot = bytes
            .checked_mul(copies)
            .ok_or("state slot size overflow")?;
        if capacity == 0 {
            return Err("request state needs at least one slot".to_owned());
        }
        // The serving KV budget subtracts this total, so it must be representable.
        bytes_per_slot
            .checked_mul(capacity as u64)
            .ok_or("state pool size overflow")?;
        let preallocated = manifest
            .resource_requirements()
            .required_capabilities()
            .iter()
            .any(|capability| {
                capability.as_str()
                    == runtime_bundle_abi::RECURRENT_STATE_PREALLOCATED_CAPABILITY_V1
            });
        Ok(Self {
            capacity,
            bytes_per_slot,
            preallocated,
            owners: BTreeMap::new(),
            generations: BTreeMap::new(),
        })
    }

    #[cfg(test)]
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            bytes_per_slot: 0,
            preallocated: false,
            owners: BTreeMap::new(),
            generations: BTreeMap::new(),
        }
    }

    /// Per-rank bytes the serving KV pool must leave free for state the
    /// bundle has not allocated yet when it acquires the pool. State a bundle
    /// declares as preallocated is already outside free memory then.
    #[must_use]
    pub fn reserved_bytes_per_rank(&self) -> u64 {
        if self.preallocated {
            0
        } else {
            self.bytes_per_slot * self.capacity as u64
        }
    }

    pub fn allocate(&mut self, request: RequestId) -> Result<u64, String> {
        if let Some(slot) = self.slot(request) {
            return Ok(slot);
        }
        let slot = (0..self.capacity as u64)
            .find(|slot| !self.owners.values().any(|owned| owned == slot))
            .ok_or("request state slots exhausted")?;
        let generation = self
            .generations
            .get(&slot)
            .copied()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or("request state generation exhausted")?;
        self.generations.insert(slot, generation);
        self.owners.insert(request, slot);
        Ok(slot)
    }

    #[must_use]
    pub fn slot(&self, request: RequestId) -> Option<u64> {
        self.owners.get(&request).copied()
    }

    /// Identity of this live lease, distinct after a slot is reused.
    #[must_use]
    pub fn lease(&self, request: RequestId) -> Option<(u64, u64)> {
        let slot = self.slot(request)?;
        Some((slot, *self.generations.get(&slot)?))
    }

    pub fn retire(&mut self, request: RequestId) {
        self.owners.remove(&request);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_late_completion_cannot_match_a_reused_slot_lease() {
        let mut pool = StateSlotPoolV1::with_capacity(1);
        let first = RequestId::new(1);
        let second = RequestId::new(2);
        pool.allocate(first).unwrap();
        let old = pool.lease(first).unwrap();
        pool.allocate(first).unwrap();
        assert_eq!(pool.lease(first), Some(old));
        pool.retire(first);
        pool.retire(first);
        assert_eq!(pool.lease(first), None);
        pool.allocate(second).unwrap();
        let next = pool.lease(second).unwrap();
        assert_eq!(old.0, next.0);
        assert_ne!(old.1, next.1);
        pool.retire(second);
        pool.allocate(first).unwrap();
        assert_ne!(
            pool.lease(first),
            Some(old),
            "request ID reuse also changes the lease"
        );
        assert_eq!(pool.generations.len(), 1);
    }

    #[test]
    fn an_exhausted_generation_does_not_take_the_slot() {
        let mut pool = StateSlotPoolV1::with_capacity(1);
        pool.generations.insert(0, u64::MAX);
        assert!(pool.allocate(RequestId::new(1)).is_err());
        assert!(pool.owners.is_empty());
        assert_eq!(pool.generations[&0], u64::MAX);
    }

    #[test]
    fn slots_are_isolated_bounded_and_reusable() {
        let mut pool = StateSlotPoolV1 {
            capacity: 2,
            bytes_per_slot: 128,
            preallocated: false,
            owners: BTreeMap::new(),
            generations: BTreeMap::new(),
        };
        let a = RequestId::new(1);
        let b = RequestId::new(2);
        let c = RequestId::new(3);
        assert_eq!(pool.allocate(a), Ok(0));
        assert_eq!(pool.allocate(a), Ok(0));
        assert_eq!(pool.allocate(b), Ok(1));
        assert!(pool.allocate(c).is_err());
        assert_eq!(pool.reserved_bytes_per_rank(), 256);
        pool.retire(a);
        pool.retire(a);
        assert_eq!(pool.allocate(c), Ok(0));
        assert_eq!(pool.slot(b), Some(1));
        pool.retire(b);
        pool.retire(c);
        assert!(pool.owners.is_empty());
    }
}
