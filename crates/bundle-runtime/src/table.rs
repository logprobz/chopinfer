//! Generation-checked instance handles behind a bundle's ABI entry points.

use std::cell::RefCell;

use runtime_bundle_abi::{BundleHandleV1, BundleStatusV1};

use crate::pipeline::Collectives;

struct Slot<T> {
    generation: u32,
    instance: Option<Box<T>>,
}

/// Live instances by handle. A handle names a slot and the generation it was
/// issued at, so a handle to a destroyed instance never reaches its
/// successor.
pub struct InstanceTable<T> {
    slots: Vec<Slot<T>>,
    generation: u32,
    /// Instances out of the table whose resources failed to release
    /// (`entry`), kept for a retry.
    pub(crate) unreleased: Vec<T>,
}

impl<T> Default for InstanceTable<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> InstanceTable<T> {
    pub const fn new() -> Self {
        Self {
            slots: Vec::new(),
            generation: 1,
            unreleased: Vec::new(),
        }
    }

    /// Stores `instance` in the first free slot, or gives it back when the
    /// slot or generation space is exhausted.
    pub fn insert(&mut self, instance: Box<T>) -> Result<BundleHandleV1, Box<T>> {
        let index = self
            .slots
            .iter()
            .position(|slot| slot.instance.is_none())
            .unwrap_or(self.slots.len());
        let Some(slot) = index.checked_add(1).and_then(|n| u32::try_from(n).ok()) else {
            return Err(instance);
        };
        let Some(next) = self.generation.checked_add(1) else {
            return Err(instance);
        };
        let generation = self.generation;
        self.generation = next;
        let occupied = Slot {
            generation,
            instance: Some(instance),
        };
        if index == self.slots.len() {
            self.slots.push(occupied);
        } else {
            self.slots[index] = occupied;
        }
        Ok(BundleHandleV1::from_parts(generation, slot))
    }

    pub fn get_mut(&mut self, handle: BundleHandleV1) -> Option<&mut T> {
        let index = usize::try_from(handle.slot()).ok()?.checked_sub(1)?;
        let slot = self.slots.get_mut(index)?;
        (handle.has_generation() && slot.generation == handle.generation()).then_some(())?;
        slot.instance.as_deref_mut()
    }

    /// Instances whose release failed, kept for a retry.
    pub fn unreleased_count(&self) -> usize {
        self.unreleased.len()
    }

    /// Drops a live instance; `handle` must have come from `get_mut`.
    pub fn remove(&mut self, handle: BundleHandleV1) {
        let index = handle.slot() as usize - 1;
        self.slots[index].instance = None;
    }
}

/// An instance whose collectives a failed submission can poison.
pub trait Live {
    fn collectives(&self) -> &Collectives;
}

/// The live instance of `handle`, unless it is poisoned (see
/// [`Collectives`]): then every call is refused.
pub fn usable<T: Live>(
    table: &mut InstanceTable<T>,
    handle: BundleHandleV1,
) -> Result<&mut T, BundleStatusV1> {
    let instance = table.get_mut(handle).ok_or(BundleStatusV1::StaleHandle)?;
    if instance.collectives().is_poisoned() {
        return Err(BundleStatusV1::InvalidState);
    }
    Ok(instance)
}

/// Runs `run` on the [`usable`] instance of `handle`. A table already in use
/// (a call re-entered from a host callback) is invalid state.
pub fn with_instance<T: Live>(
    instances: &RefCell<InstanceTable<T>>,
    handle: BundleHandleV1,
    run: impl FnOnce(&mut T) -> BundleStatusV1,
) -> BundleStatusV1 {
    let Ok(mut table) = instances.try_borrow_mut() else {
        return BundleStatusV1::InvalidState;
    };
    usable(&mut table, handle).map_or_else(|status| status, run)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_of_removed_instances_go_stale_and_slots_are_reused() {
        let mut table = InstanceTable::new();
        let first = table.insert(Box::new(1)).unwrap();
        let second = table.insert(Box::new(2)).unwrap();
        assert_eq!(table.get_mut(first).copied(), Some(1));
        table.remove(first);
        assert!(table.get_mut(first).is_none());
        let third = table.insert(Box::new(3)).unwrap();
        assert_eq!(third.slot(), first.slot());
        assert_ne!(third.generation(), first.generation());
        assert!(table.get_mut(first).is_none());
        assert_eq!(table.get_mut(third).copied(), Some(3));
        assert_eq!(table.get_mut(second).copied(), Some(2));
    }

    impl Live for Collectives {
        fn collectives(&self) -> &Collectives {
            self
        }
    }

    #[test]
    fn poisoned_stale_and_busy_instances_are_refused() {
        let instances = RefCell::new(InstanceTable::new());
        let live = instances
            .borrow_mut()
            .insert(Box::new(Collectives::default()))
            .unwrap();
        let ran = |handle| with_instance(&instances, handle, |_| BundleStatusV1::Success);
        assert_eq!(ran(live), BundleStatusV1::Success);
        assert_eq!(
            ran(BundleHandleV1::from_parts(
                live.generation() + 1,
                live.slot()
            )),
            BundleStatusV1::StaleHandle
        );
        {
            let _busy = instances.borrow();
            assert_eq!(ran(live), BundleStatusV1::InvalidState);
        }
        instances.borrow_mut().get_mut(live).unwrap().poison();
        assert_eq!(ran(live), BundleStatusV1::InvalidState);
        assert_eq!(
            usable(&mut instances.borrow_mut(), live).err(),
            Some(BundleStatusV1::InvalidState)
        );
    }
}
