//! Instance-scoped generation-checked handle storage.
//!
//! Every resource a host service hands to a bundle is a slot index paired with
//! the generation that issued it. A reused slot never answers to the handle its
//! previous occupant received, and a released handle is rejected for the life of
//! the instance.
//!
//! Generations come from one process-wide non-wrapping domain, so no two live
//! handles anywhere in this process carry the same raw value. Two host
//! instances that hand out the same slot index still differ in the generation
//! half of the handle, which is what makes a foreign handle rejectable while
//! the slot it collides with is still live.

use std::sync::atomic::{AtomicU32, Ordering};

use runtime_bundle_abi::BundleStatusV1;

/// The result every table operation returns to a service method.
pub(crate) type SlotResultV1<T> = Result<T, BundleStatusV1>;

/// The maximum number of live slots one table keeps for one instance. A
/// bundle may resolve every entry of its cubins on every rank, about 4,000
/// symbols at eight ranks, so 4,096 left almost no room. The cap only
/// bounds leaks; a slot is a generation and a small record.
pub(crate) const MAX_SLOTS_PER_TABLE_V1: usize = 65_536;

/// The process-wide generation domain every issued handle draws from.
///
/// Zero is never issued because the counter starts at one and never wraps: the
/// all-zero handle stays the ABI's invalid handle, and an exhausted domain is
/// reported instead of reissuing a generation a live handle already carries.
static NEXT_GENERATION_V1: AtomicU32 = AtomicU32::new(1);

/// Takes the next process-wide generation, or reports the domain exhausted.
fn issue_generation_v1() -> SlotResultV1<u32> {
    let mut generation = NEXT_GENERATION_V1.load(Ordering::Relaxed);
    loop {
        // The last value is kept as the exhausted marker rather than issued, so
        // the counter has no successor to wrap onto.
        if generation == u32::MAX {
            return Err(BundleStatusV1::ResourceExhausted);
        }
        match NEXT_GENERATION_V1.compare_exchange_weak(
            generation,
            generation + 1,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return Ok(generation),
            Err(observed) => generation = observed,
        }
    }
}

/// One issued handle, split into the parts an opaque ABI handle carries.
///
/// The order is total so a key can index a map, and it means nothing beyond
/// that: no caller reads it as an age or a rank.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct SlotKeyV1 {
    pub(crate) generation: u32,
    pub(crate) slot: u32,
}

struct SlotV1<T> {
    generation: u32,
    value: Option<T>,
}

/// A generation-checked table of one resource kind for one bundle instance.
///
/// Slot zero is never issued, because an all-zero handle is the ABI's invalid
/// handle. Generation zero is never issued for the same reason.
pub(crate) struct SlotTableV1<T> {
    slots: Vec<SlotV1<T>>,
    free: Vec<u32>,
}

impl<T> SlotTableV1<T> {
    pub(crate) const fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
        }
    }

    pub(crate) fn insert(&mut self, value: T) -> SlotResultV1<SlotKeyV1> {
        self.try_insert_with(|| Ok(value))
    }

    /// Reserves handle metadata before constructing an externally owned resource.
    /// A failed constructor leaves the slots unchanged; its generation is retired.
    pub(crate) fn try_insert_with(
        &mut self,
        make: impl FnOnce() -> SlotResultV1<T>,
    ) -> SlotResultV1<SlotKeyV1> {
        if self.free.is_empty() && self.slots.len() >= MAX_SLOTS_PER_TABLE_V1 {
            // Refuse before taking a generation, so a full table does not burn
            // the process-wide domain down.
            return Err(BundleStatusV1::ResourceExhausted);
        }
        let generation = issue_generation_v1()?;
        let slot = match self.free.last() {
            Some(slot) => *slot,
            None => u32::try_from(self.slots.len().max(1))
                .map_err(|_| BundleStatusV1::ResourceExhausted)?,
        };
        let value = make()?;
        if self.free.pop().is_some() {
            let entry = &mut self.slots[slot as usize];
            entry.generation = generation;
            entry.value = Some(value);
            return Ok(SlotKeyV1 { generation, slot });
        }
        if self.slots.is_empty() {
            // Reserve the never-issued zero slot.
            self.slots.push(SlotV1 {
                generation: 0,
                value: None,
            });
        }
        self.slots.push(SlotV1 {
            generation,
            value: Some(value),
        });
        Ok(SlotKeyV1 { generation, slot })
    }

    fn entry(&self, key: SlotKeyV1) -> SlotResultV1<&SlotV1<T>> {
        if key.slot == 0 || key.generation == 0 {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let entry = self
            .slots
            .get(key.slot as usize)
            .ok_or(BundleStatusV1::StaleHandle)?;
        if entry.generation != key.generation || entry.value.is_none() {
            return Err(BundleStatusV1::StaleHandle);
        }
        Ok(entry)
    }

    pub(crate) fn get(&self, key: SlotKeyV1) -> SlotResultV1<&T> {
        self.entry(key)?
            .value
            .as_ref()
            .ok_or(BundleStatusV1::StaleHandle)
    }

    pub(crate) fn get_mut(&mut self, key: SlotKeyV1) -> SlotResultV1<&mut T> {
        self.entry(key)?;
        self.slots[key.slot as usize]
            .value
            .as_mut()
            .ok_or(BundleStatusV1::StaleHandle)
    }

    pub(crate) fn remove(&mut self, key: SlotKeyV1) -> SlotResultV1<T> {
        self.entry(key)?;
        let value = self.slots[key.slot as usize]
            .value
            .take()
            .ok_or(BundleStatusV1::StaleHandle)?;
        self.free.push(key.slot);
        Ok(value)
    }

    /// Every live value, in slot order.
    pub(crate) fn values(&self) -> impl Iterator<Item = &T> {
        self.slots.iter().filter_map(|slot| slot.value.as_ref())
    }

    #[cfg(test)]
    pub(crate) fn live_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.value.is_some())
            .count()
    }
}

/// Reads the slot key one ABI handle carries without trusting its contents.
pub(crate) fn slot_key_v1(generation: u32, slot: u32) -> SlotResultV1<SlotKeyV1> {
    if generation == 0 || slot == 0 {
        return Err(BundleStatusV1::InvalidArgument);
    }
    Ok(SlotKeyV1 { generation, slot })
}

#[cfg(test)]
mod tests {
    use super::{MAX_SLOTS_PER_TABLE_V1, SlotTableV1, slot_key_v1};
    use runtime_bundle_abi::BundleStatusV1;

    #[test]
    fn a_reused_slot_rejects_the_handle_its_previous_occupant_received() {
        let mut table = SlotTableV1::new();
        let first = table.insert(7_u32).unwrap();
        assert_eq!(table.remove(first), Ok(7));

        let second = table.insert(9_u32).unwrap();

        assert_eq!(first.slot, second.slot);
        assert_ne!(first.generation, second.generation);
        assert_eq!(table.get(first), Err(BundleStatusV1::StaleHandle));
        assert_eq!(table.get(second), Ok(&9));
    }

    #[test]
    fn a_released_handle_stays_rejected() {
        let mut table = SlotTableV1::new();
        let key = table.insert(1_u8).unwrap();
        assert!(table.remove(key).is_ok());

        assert_eq!(table.remove(key), Err(BundleStatusV1::StaleHandle));
        assert_eq!(table.get(key), Err(BundleStatusV1::StaleHandle));
        assert_eq!(table.live_count(), 0);
    }

    #[test]
    fn zero_generation_or_slot_is_never_a_key() {
        assert_eq!(slot_key_v1(0, 1), Err(BundleStatusV1::InvalidArgument));
        assert_eq!(slot_key_v1(1, 0), Err(BundleStatusV1::InvalidArgument));
        assert!(slot_key_v1(1, 1).is_ok());
    }

    #[test]
    fn an_unknown_slot_is_stale_rather_than_a_panic() {
        let table = SlotTableV1::<u8>::new();
        let key = slot_key_v1(1, 4096).unwrap();

        assert_eq!(table.get(key), Err(BundleStatusV1::StaleHandle));
    }

    /// An eight-rank bundle's symbols (about 500 per rank, twice that with
    /// headroom) fit one table with room to spare.
    #[test]
    fn an_eight_rank_bundle_fits_one_table() {
        let mut table = SlotTableV1::new();
        for value in 0..8 * 1024 {
            table.insert(value).unwrap();
        }
        // Four times that still fits.
        const _: () = assert!(8 * 1024 * 4 <= MAX_SLOTS_PER_TABLE_V1);
    }

    #[test]
    fn the_slot_cap_reports_exhaustion_instead_of_growing() {
        let mut table = SlotTableV1::new();
        for value in 0..MAX_SLOTS_PER_TABLE_V1 - 1 {
            table.insert(value).unwrap();
        }

        assert_eq!(table.insert(0), Err(BundleStatusV1::ResourceExhausted));
    }

    #[test]
    fn a_failed_constructor_preserves_empty_and_reusable_slots() {
        let mut table = SlotTableV1::<u8>::new();
        assert_eq!(
            table.try_insert_with(|| Err(BundleStatusV1::HostError)),
            Err(BundleStatusV1::HostError)
        );
        assert!(table.slots.is_empty());
        let first = table.insert(7).unwrap();
        table.remove(first).unwrap();
        assert_eq!(
            table.try_insert_with(|| Err(BundleStatusV1::HostError)),
            Err(BundleStatusV1::HostError)
        );
        assert_eq!(table.free, vec![first.slot]);
        assert_eq!(table.get(first), Err(BundleStatusV1::StaleHandle));
        let second = table.try_insert_with(|| Ok(9)).unwrap();
        assert_eq!(second.slot, first.slot);
        assert_ne!(second.generation, first.generation);
        assert_eq!(table.get(second), Ok(&9));
    }

    #[test]
    fn a_full_table_refuses_before_calling_the_constructor() {
        let mut table = SlotTableV1::new();
        for value in 0..MAX_SLOTS_PER_TABLE_V1 - 1 {
            table.insert(value).unwrap();
        }
        assert_eq!(
            table.try_insert_with(|| panic!("a full table must not create a resource")),
            Err(BundleStatusV1::ResourceExhausted)
        );
    }

    #[test]
    fn two_tables_never_issue_the_same_generation_for_the_same_slot() {
        let mut first = SlotTableV1::new();
        let mut second = SlotTableV1::new();

        let here = first.insert(1_u8).unwrap();
        let there = second.insert(2_u8).unwrap();

        // Both tables hand out their first slot, so only the process-wide
        // generation separates the two handles.
        assert_eq!(here.slot, there.slot);
        assert_ne!(here.generation, there.generation);
        assert_eq!(first.get(there), Err(BundleStatusV1::StaleHandle));
        assert_eq!(second.get(here), Err(BundleStatusV1::StaleHandle));
        // Neither table answered a foreign handle, and both slots stayed live.
        assert_eq!(first.get(here), Ok(&1));
        assert_eq!(second.get(there), Ok(&2));
    }
}
