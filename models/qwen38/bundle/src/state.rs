//! Mapped committed entries and bounded checkpoint workspaces.
//! AR and prefill update slots in place. Verify items write the step pool;
//! commit exchanges the slot entry with the kept checkpoint entry. The shared ledger
//! (`bundle_runtime::ledger`) holds which request each host-named slot
//! holds and the open step; the linear pools declare `Cancel::Invalidate`,
//! so a cancel frees every touched slot and the host must retire the failed
//! batch.

use bundle_runtime::ledger::{Item, Step};

use crate::contract::{DRAFT_ITEMS, MAX_VERIFY_ROWS};
use crate::inputs::PhaseInputs;
use crate::side::ItemEntries;

pub use bundle_runtime::entries::EntryMap;

/// What an open step saved and staged beyond the ledger's step.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Open {
    /// Phases (prefill, decode) whose PLE staging commit copies back.
    pub ple: [bool; 2],
    /// The QSA ring was saved before a prefill phase wrote it.
    pub ring_saved: bool,
    /// Failed execution retained only so the host can retry cancellation.
    pub cancel_only: bool,
    /// Only the drafter layer's ring was saved, before the drafter ran.
    pub drafter_ring_saved: bool,
    /// What the drafter commits, with the drafts it wrote.
    pub drafter: Option<crate::drafter::Pending>,
}

/// The ledger items of a step's phases, prefill first, each in the slot the
/// host named. A verify item names at commit the rows it keeps.
pub fn items(phases: [&PhaseInputs; 2]) -> Vec<Item> {
    phases
        .into_iter()
        .flat_map(|phase| {
            (0..phase.request_ids.len()).map(|item| Item {
                request: phase.request_ids[item],
                slot: u32::try_from(phase.slot_ids[item]).ok(),
                start: phase.prefix_len[item] as u64,
                rows: phase.q_len.get(item).copied().unwrap_or(1) as u32,
                verify: phase.verify_rows.get(item).copied().unwrap_or(false),
            })
        })
        .collect()
}

/// Whether a phase's verify items fit the step checkpoints: at most
/// `DRAFT_ITEMS` items of more than one row, each of at most `verify_rows`.
pub fn fits_checkpoints(phase: &PhaseInputs, verify_rows: u32) -> bool {
    let rows = |item: usize| phase.q_len.get(item).copied().unwrap_or(1) as u32;
    let drafting = (0..phase.request_ids.len())
        .filter(|&item| phase.verify && rows(item) > 1)
        .collect::<Vec<_>>();
    drafting.len() <= DRAFT_ITEMS as usize && drafting.iter().all(|&item| rows(item) <= verify_rows)
}

/// The pool entries each item of `phase` reads and writes, the phase's
/// items being the open step's from `first` on: the slot's committed entry
/// (none for a new request) and working entry, and a verify item's
/// checkpoints from the mapped workspace, `verify_rows` per item
/// in the step's verify order (`Step::verify_index`), independent of slot
/// IDs. Single-row items update their slot even inside a verify phase.
pub fn entries(
    step: &Step,
    first: usize,
    phase: &PhaseInputs,
    map: &EntryMap,
    verify_rows: u32,
    ple: &EntryMap,
) -> Vec<ItemEntries> {
    (0..phase.request_ids.len())
        .map(|item| {
            let placed = step.items[first + item];
            let slot = map.committed(placed.slot);
            let mut checkpoints = [slot; MAX_VERIFY_ROWS as usize];
            if let Some(index) = step.verify_index(first + item)
                && phase.verify
                && placed.rows > 1
            {
                let base = index * verify_rows;
                checkpoints = std::array::from_fn(|row| {
                    map.checkpoint((base + (row as u32).min(placed.rows - 1)) as usize)
                });
            }
            if phase.verify && placed.rows > 1 {
                debug_assert!(
                    checkpoints
                        .iter()
                        .all(|&id| map.workspace_position(id).is_some())
                );
            }
            let ple_checkpoints = std::array::from_fn(|row| {
                if phase.verify {
                    ple.checkpoint(
                        item * MAX_VERIFY_ROWS as usize + row.min(placed.rows as usize - 1),
                    )
                } else {
                    ple.committed(placed.slot)
                }
            });
            if phase.verify {
                debug_assert!(
                    ple_checkpoints
                        .iter()
                        .all(|&id| ple.workspace_position(id).is_some())
                );
            }
            ItemEntries {
                ple_working: ple.committed(placed.slot),
                ple_checkpoints,
                committed: if placed.start == 0 { -1 } else { slot },
                working: slot,
                checkpoints,
                anchor: None,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use bundle_runtime::ledger::{Ledger, LedgerError, Placement, Rules};

    use super::*;

    #[derive(Clone, Debug)]
    struct EntryMap {
        inner: bundle_runtime::entries::EntryMap,
        slots: u32,
        checkpoints: u32,
    }

    impl EntryMap {
        fn new(slots: u32, checkpoints: u32) -> Self {
            Self {
                inner: super::EntryMap::new(slots, checkpoints),
                slots,
                checkpoints,
            }
        }
        fn slots(&self) -> Vec<i32> {
            (0..self.slots)
                .map(|slot| self.inner.committed(slot))
                .collect()
        }
        fn checkpoints(&self) -> Vec<i32> {
            (0..self.checkpoints)
                .map(|index| self.inner.checkpoint(index as usize))
                .collect()
        }
    }
    impl std::ops::Deref for EntryMap {
        type Target = bundle_runtime::entries::EntryMap;
        fn deref(&self) -> &Self::Target {
            &self.inner
        }
    }
    impl std::ops::DerefMut for EntryMap {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.inner
        }
    }

    /// A ledger of `slots` host-named slots whose state invalidates on
    /// cancel, as the linear pools declare.
    fn ledger(slots: u32) -> Ledger {
        let rules = Rules {
            invalidate_on_cancel: true,
            tail_window: None,
        };
        Ledger::new(slots, Placement::Host, rules, 1 << 20)
    }

    fn restored(slots: u32) -> Ledger {
        let mut ledger = ledger(slots);
        for slot in 0..slots {
            ledger.restore(u64::from(slot) + 1, 10, Some(slot)).unwrap();
        }
        ledger
    }

    #[test]
    fn entry_exchange_partitions_the_pool_across_accept_cancel_and_reuse() {
        for rows in 2..=MAX_VERIFY_ROWS {
            for kept in 1..=rows {
                let mut map = EntryMap::new(3, 2 * rows);
                let mut pool = (0..3 + 2 * rows)
                    .map(|entry| vec![entry])
                    .collect::<Vec<_>>();
                let source = pool[map.committed(2) as usize].clone();
                for row in 0..rows {
                    let mut value = source.clone();
                    value.extend(0..=row);
                    pool[map.checkpoint(row as usize) as usize] = value;
                }
                let selected = map.checkpoint(kept as usize - 1);
                let expected = pool[selected as usize].clone();
                map.prepare(&[(2, selected)])
                    .unwrap()
                    .commit(&mut map)
                    .unwrap();
                assert_eq!(pool[map.committed(2) as usize], expected);
                assert_eq!(map.checkpoint(kept as usize - 1), 2);
                // A repeated verify, including a cancelled one, can overwrite
                // every workspace entry without changing committed bytes.
                for generation in 0..3 {
                    for &entry in &map.checkpoints() {
                        pool[entry as usize] = vec![u32::MAX, generation];
                    }
                    assert_eq!(pool[map.committed(2) as usize], expected);
                }
                // Reinitializing a reused slot cannot touch any other slot.
                pool[map.committed(2) as usize] = vec![99];
                assert_eq!(pool[map.committed(0) as usize], [0]);
                assert_eq!(pool[map.committed(1) as usize], [1]);
                let mut owners = map
                    .slots()
                    .iter()
                    .chain(&map.checkpoints())
                    .copied()
                    .collect::<Vec<_>>();
                for (position, &entry) in map.checkpoints().iter().enumerate() {
                    assert_eq!(map.workspace_position(entry), Some(position));
                }
                assert!(
                    map.slots()
                        .iter()
                        .all(|&entry| map.workspace_position(entry).is_none())
                );
                owners.sort_unstable();
                assert_eq!(owners, (0..(3 + 2 * rows) as i32).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn invalid_exchange_batch_leaves_every_owner_unchanged() {
        let mut map = EntryMap::new(3, 8);
        let slots = map.slots().clone();
        let checkpoints = map.checkpoints().clone();
        for choices in [
            vec![(0, 3), (1, -1)],
            vec![(0, 3), (3, 4)],
            vec![(0, 3), (0, 4)],
            vec![(0, 3), (1, 3)],
        ] {
            assert!(map.prepare(&choices).is_none());
            assert_eq!(map.slots(), slots);
            assert_eq!(map.checkpoints(), checkpoints);
        }
        // Preparing one pool cannot change it if a different pool is invalid.
        let other = EntryMap::new(3, 8);
        let prepared = map.prepare(&[(0, 3)]).unwrap();
        assert!(other.prepare(&[(1, -1)]).is_none());
        drop(prepared);
        assert_eq!(map.slots(), slots);
        assert_eq!(map.checkpoints(), checkpoints);
        let stale = map.prepare(&[(0, 3)]).unwrap();
        map.prepare(&[(0, 3), (1, 4)])
            .unwrap()
            .commit(&mut map)
            .unwrap();
        assert_eq!(map.workspace_position(0), Some(0));
        assert_eq!(map.workspace_position(1), Some(1));
        assert_eq!(map.workspace_position(3), None);
        assert_eq!(map.workspace_position(4), None);
        let slots = map.slots().clone();
        let checkpoints = map.checkpoints().clone();
        assert!(stale.commit(&mut map).is_none());
        assert_eq!(map.slots(), slots);
        assert_eq!(map.checkpoints(), checkpoints);
    }

    #[test]
    fn multi_slot_steps_recycle_workspace_and_restore_a_preempted_request() {
        fn check_writes(
            phase: &PhaseInputs,
            planned: &[ItemEntries],
            linear: &EntryMap,
            ple: &EntryMap,
        ) {
            for (map, ple_pool) in [(linear, false), (ple, true)] {
                let mut writes = std::collections::BTreeSet::new();
                for (item, entry) in planned.iter().enumerate() {
                    assert_eq!(
                        entry.committed,
                        linear.committed(phase.slot_ids[item] as u32)
                    );
                    assert_eq!(
                        entry.ple_working,
                        ple.committed(phase.slot_ids[item] as u32)
                    );
                    let checkpoints = if ple_pool {
                        entry.ple_checkpoints
                    } else {
                        entry.checkpoints
                    };
                    for &checkpoint in &checkpoints[..phase.q_len[item] as usize] {
                        assert!(
                            !map.slots().contains(&checkpoint),
                            "writes another slot's committed entry"
                        );
                        assert!(writes.insert(checkpoint), "two verify rows alias");
                    }
                }
            }
        }

        let mut ledger = restored(4);
        let mut linear = EntryMap::new(4, 8);
        let mut ple = EntryMap::new(4, 4 * MAX_VERIFY_ROWS);
        let mut pools = [
            vec![Vec::new(); 12],
            vec![Vec::new(); 4 + 4 * MAX_VERIFY_ROWS as usize],
        ];
        let mut lengths = [10_u64; 4];
        let mut requests = [1_u64, 2, 3, 4];
        let mut captured = [Vec::new(), Vec::new()];
        for (round, slots) in [[0, 2], [2, 1], [3, 2]].into_iter().enumerate() {
            if round == 2 {
                // Preempt slot 0 and restore its snapshot into a different slot.
                ledger.retire(requests[0]).unwrap();
                ledger.retire(requests[3]).unwrap();
                requests[3] = 99;
                lengths[3] = 12;
                ledger.restore(99, lengths[3], Some(3)).unwrap();
                pools[0][linear.committed(3) as usize] = captured[0].clone();
                pools[1][ple.committed(3) as usize] = captured[1].clone();
                assert_eq!(pools[0][linear.committed(3) as usize], captured[0]);
                assert_eq!(pools[1][ple.committed(3) as usize], captured[1]);
            }
            let phase = PhaseInputs {
                slot_ids: slots.map(|slot| slot as i32).to_vec(),
                request_ids: slots.map(|slot| requests[slot]).to_vec(),
                prefix_len: slots.map(|slot| lengths[slot] as i32).to_vec(),
                q_len: vec![4, 3],
                verify_rows: vec![true; 2],
                verify: true,
                ..PhaseInputs::default()
            };
            let empty = PhaseInputs::default();
            ledger.begin(&items([&empty, &phase]), Vec::new()).unwrap();
            let planned = entries(ledger.step().unwrap(), 0, &phase, &linear, 4, &ple);
            check_writes(&phase, &planned, &linear, &ple);
            for (pool, ple_pool) in pools.iter_mut().zip([false, true]) {
                for (item, entry) in planned.iter().enumerate() {
                    let checkpoints = if ple_pool {
                        entry.ple_checkpoints
                    } else {
                        entry.checkpoints
                    };
                    for (row, &checkpoint) in
                        checkpoints[..phase.q_len[item] as usize].iter().enumerate()
                    {
                        pool[checkpoint as usize] = vec![round as u32, item as u32, row as u32];
                    }
                }
            }
            let choices = |ple_pool| {
                planned
                    .iter()
                    .enumerate()
                    .map(|(item, entry)| {
                        let checkpoints = if ple_pool {
                            entry.ple_checkpoints
                        } else {
                            entry.checkpoints
                        };
                        (slots[item] as u32, checkpoints[1 + item])
                    })
                    .collect::<Vec<_>>()
            };
            let expected: Vec<_> = choices(false)
                .iter()
                .map(|&(_, entry)| pools[0][entry as usize].clone())
                .collect();
            let prepared = linear.prepare(&choices(false)).unwrap();
            let prepared_ple = ple.prepare(&choices(true)).unwrap();
            prepared.commit(&mut linear).unwrap();
            prepared_ple.commit(&mut ple).unwrap();
            ledger
                .commit(&[(requests[slots[0]], 2), (requests[slots[1]], 3)])
                .unwrap();
            for (item, &slot) in slots.iter().enumerate() {
                lengths[slot] += 2 + item as u64;
                assert_eq!(
                    pools[0][linear.committed(slot as u32) as usize],
                    expected[item]
                );
                assert_eq!(
                    pools[1][ple.committed(slot as u32) as usize],
                    expected[item]
                );
            }
            if round == 0 {
                captured = [
                    pools[0][linear.committed(0) as usize].clone(),
                    pools[1][ple.committed(0) as usize].clone(),
                ];
            }
            // A second verify uses the reshuffled workspace, then cancels.
            let committed = [linear.slots().clone(), ple.slots().clone()];
            let phase = PhaseInputs {
                prefix_len: slots.map(|slot| lengths[slot] as i32).to_vec(),
                ..phase
            };
            ledger.begin(&items([&empty, &phase]), Vec::new()).unwrap();
            let repeated = entries(ledger.step().unwrap(), 0, &phase, &linear, 4, &ple);
            check_writes(&phase, &repeated, &linear, &ple);
            let before: Vec<_> = pools
                .iter()
                .zip([&linear, &ple])
                .map(|(pool, map)| {
                    map.slots()
                        .iter()
                        .map(|&entry| pool[entry as usize].clone())
                        .collect::<Vec<_>>()
                })
                .collect();
            for (pool, ple_pool) in pools.iter_mut().zip([false, true]) {
                for (item, entry) in repeated.iter().enumerate() {
                    let checkpoints = if ple_pool {
                        entry.ple_checkpoints
                    } else {
                        entry.checkpoints
                    };
                    for &checkpoint in &checkpoints[..phase.q_len[item] as usize] {
                        pool[checkpoint as usize] = vec![u32::MAX];
                    }
                }
            }
            ledger.cancel().unwrap();
            for (pool, map, expected) in [
                (&pools[0], &linear, &before[0]),
                (&pools[1], &ple, &before[1]),
            ] {
                assert_eq!(
                    &map.slots()
                        .iter()
                        .map(|&entry| pool[entry as usize].clone())
                        .collect::<Vec<_>>(),
                    expected
                );
            }
            assert_eq!(linear.slots(), committed[0]);
            assert_eq!(ple.slots(), committed[1]);
            for &slot in &slots {
                ledger.retire(requests[slot]).unwrap();
                ledger
                    .restore(requests[slot], lengths[slot], Some(slot as u32))
                    .unwrap();
            }
        }
    }

    #[test]
    fn wide_steps_share_checkpoints_and_keep_committed_slot_indices() {
        for count in [32, 33, 64, 128, 256] {
            let mut ledger = restored(count);
            let phase = PhaseInputs {
                slot_ids: (0..count as i32).rev().collect(),
                request_ids: (1..=count as u64).rev().collect(),
                prefix_len: vec![10; count as usize],
                q_len: (0..count).map(|i| if i < 32 { 4 } else { 1 }).collect(),
                verify_rows: (0..count).map(|i| i < 32).collect(),
                verify: true,
                ..PhaseInputs::default()
            };
            assert!(fits_checkpoints(&phase, 4));
            let items = items([&PhaseInputs::default(), &phase]);
            ledger.begin(&items, Vec::new()).unwrap();
            let planned = entries(
                ledger.step().unwrap(),
                0,
                &phase,
                &EntryMap::new(count, count.min(DRAFT_ITEMS) * 4),
                4,
                &EntryMap::new(count, count.next_power_of_two() * MAX_VERIFY_ROWS),
            );
            for (i, entry) in planned.iter().enumerate() {
                assert_eq!(entry.committed, count as i32 - i as i32 - 1);
                assert_eq!(entry.working, entry.committed);
                if i < 32 {
                    assert_eq!(
                        &entry.checkpoints[..4],
                        &(0..4)
                            .map(|r| count as i32 + i as i32 * 4 + r)
                            .collect::<Vec<_>>()
                    );
                } else {
                    assert_eq!(entry.checkpoints, [entry.committed; 8]);
                }
            }
            let mut kept = (1..=count as u64)
                .rev()
                .take(32)
                .map(|request| (request, 2))
                .collect::<Vec<_>>();
            kept[0].1 = 5;
            assert_eq!(ledger.commit(&kept), Err(LedgerError::Kept));
            kept[0].1 = 2;
            ledger.commit(&kept).unwrap();
            assert_eq!(ledger.committed(count as u64, 12), Ok(count - 1));
        }
    }

    #[test]
    fn cancellation_invalidates_every_touched_slot_and_restore_reuses_it() {
        let mut ledger = ledger(64);
        for slot in [0, 32, 63] {
            ledger.restore(u64::from(slot) + 1, 10, Some(slot)).unwrap();
        }
        let phase = PhaseInputs {
            slot_ids: vec![0, 63],
            request_ids: vec![1, 64],
            prefix_len: vec![10, 10],
            q_len: vec![1, 1],
            ..PhaseInputs::default()
        };
        ledger
            .begin(&items([&PhaseInputs::default(), &phase]), Vec::new())
            .unwrap();
        ledger.cancel().unwrap();
        assert_eq!(ledger.holder(0), None);
        assert_eq!(ledger.holder(63), None);
        assert_eq!(ledger.holder(32), Some((33, 10)));
        ledger.restore(65, 10, Some(63)).unwrap();
        assert_eq!(ledger.committed(65, 10), Ok(63));
        assert_eq!(
            ledger.restore(66, 10, Some(64)),
            Err(LedgerError::ForeignSlot)
        );
    }

    #[test]
    fn refuses_more_than_32_checkpoint_owners() {
        let phase = PhaseInputs {
            slot_ids: (0..33).collect(),
            request_ids: (1..34).collect(),
            prefix_len: vec![0; 33],
            q_len: vec![4; 33],
            verify_rows: vec![true; 33],
            verify: true,
            ..PhaseInputs::default()
        };
        assert!(!fits_checkpoints(&phase, 4));
        let mut fewer = phase;
        fewer.q_len[32] = 1;
        assert!(fits_checkpoints(&fewer, 4));
        assert!(!fits_checkpoints(&fewer, 3));
    }
}
