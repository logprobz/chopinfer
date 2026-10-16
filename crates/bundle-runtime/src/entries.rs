//! Checked ownership of slot and checkpoint workspace entries.

/// Physical entries owned by logical slots and by the shared checkpoint workspace.
/// These lists partition the pool. Retirement and
/// cancellation leave ownership unchanged; a new owner initializes or restores
/// the entry still assigned to its logical slot.
#[derive(Clone, Debug)]
pub struct EntryMap {
    slots: Vec<i32>,
    checkpoints: Vec<i32>,
    workspace_positions: Vec<Option<usize>>,
}

impl EntryMap {
    pub fn new(slots: u32, checkpoint_entries: u32) -> Self {
        Self {
            slots: (0..slots as i32).collect(),
            checkpoints: (slots as i32..(slots + checkpoint_entries) as i32).collect(),
            workspace_positions: std::iter::repeat_n(None, slots as usize)
                .chain((0..checkpoint_entries as usize).map(Some))
                .collect(),
        }
    }

    pub fn committed(&self, slot: u32) -> i32 {
        self.slots[slot as usize]
    }

    pub fn checkpoint(&self, index: usize) -> i32 {
        self.checkpoints[index]
    }

    /// The workspace position of an entry, or none when a slot owns it.
    pub fn workspace_position(&self, entry: i32) -> Option<usize> {
        self.workspace_positions
            .get(usize::try_from(entry).ok()?)
            .copied()
            .flatten()
    }

    /// Validate a whole step without changing ownership. Workspace lookup is
    /// constant time, including after previous commits recycle slot entries.
    pub fn prepare(&self, choices: &[(u32, i32)]) -> Option<Exchanges> {
        let mut positions = Vec::with_capacity(choices.len());
        for (item, &(slot, checkpoint)) in choices.iter().enumerate() {
            if choices[..item]
                .iter()
                .any(|&(previous, entry)| previous == slot || entry == checkpoint)
            {
                return None;
            }
            let committed = *self.slots.get(slot as usize)?;
            let index = self.workspace_position(checkpoint)?;
            if self.checkpoints.get(index).copied() != Some(checkpoint) {
                return None;
            }
            positions.push((slot as usize, index, committed, checkpoint));
        }
        Some(Exchanges { positions })
    }
}

/// Checked workspace positions, retained while fallible device copies run.
pub struct Exchanges {
    positions: Vec<(usize, usize, i32, i32)>,
}

impl Exchanges {
    pub fn matches(&self, map: &EntryMap) -> bool {
        self.positions
            .iter()
            .all(|&(slot, index, committed, checkpoint)| {
                map.slots.get(slot).copied() == Some(committed)
                    && map.checkpoints.get(index).copied() == Some(checkpoint)
                    && map.workspace_position(checkpoint) == Some(index)
            })
    }

    /// A stale plan returns an error before changing any owner.
    pub fn commit(self, map: &mut EntryMap) -> Option<()> {
        if !self.matches(map) {
            return None;
        }
        for (slot, index, committed, checkpoint) in self.positions {
            std::mem::swap(&mut map.slots[slot], &mut map.checkpoints[index]);
            map.workspace_positions[committed as usize] = Some(index);
            map.workspace_positions[checkpoint as usize] = None;
        }
        Some(())
    }
}

impl Exchanges {
    /// Checks both maps before applying either exchange. No device rollback is implied.
    pub fn commit_pair(
        self,
        first: &mut EntryMap,
        other: Self,
        second: &mut EntryMap,
    ) -> Option<()> {
        if !self.matches(first) || !other.matches(second) {
            return None;
        }
        self.commit(first)?;
        other.commit(second)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_apply_refuses_either_stale_map_before_changing_any_owner() {
        for stale_first in [false, true] {
            let mut first = EntryMap::new(2, 4);
            let mut second = EntryMap::new(2, 4);
            let a = first.prepare(&[(0, 2)]).unwrap();
            let b = second.prepare(&[(1, 3)]).unwrap();
            let stale = if stale_first { &mut first } else { &mut second };
            stale
                .prepare(&[(0, 2), (1, 3)])
                .unwrap()
                .commit(stale)
                .unwrap();
            let before = (first.clone(), second.clone());
            assert!(a.commit_pair(&mut first, b, &mut second).is_none());
            assert_eq!(first.slots, before.0.slots);
            assert_eq!(first.checkpoints, before.0.checkpoints);
            assert_eq!(first.workspace_positions, before.0.workspace_positions);
            assert_eq!(second.slots, before.1.slots);
            assert_eq!(second.checkpoints, before.1.checkpoints);
            assert_eq!(second.workspace_positions, before.1.workspace_positions);
        }
        let mut first = EntryMap::new(2, 4);
        let mut second = EntryMap::new(2, 4);
        first
            .prepare(&[(0, 2)])
            .unwrap()
            .commit_pair(&mut first, second.prepare(&[(1, 3)]).unwrap(), &mut second)
            .unwrap();
        assert_eq!(first.slots, [2, 1]);
        assert_eq!(first.checkpoints, [0, 3, 4, 5]);
        assert_eq!(second.slots, [0, 3]);
        assert_eq!(second.checkpoints, [2, 1, 4, 5]);
    }

    #[test]
    fn entry_exchange_partitions_the_pool_across_accept_cancel_and_reuse() {
        for rows in 2..=8 {
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
                    for &entry in &map.checkpoints {
                        pool[entry as usize] = vec![u32::MAX, generation];
                    }
                    assert_eq!(pool[map.committed(2) as usize], expected);
                }
                // Reinitializing a reused slot cannot touch any other slot.
                pool[map.committed(2) as usize] = vec![99];
                assert_eq!(pool[map.committed(0) as usize], [0]);
                assert_eq!(pool[map.committed(1) as usize], [1]);
                let mut owners = map
                    .slots
                    .iter()
                    .chain(&map.checkpoints)
                    .copied()
                    .collect::<Vec<_>>();
                for (position, &entry) in map.checkpoints.iter().enumerate() {
                    assert_eq!(map.workspace_position(entry), Some(position));
                }
                assert!(
                    map.slots
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
        let slots = map.slots.clone();
        let checkpoints = map.checkpoints.clone();
        for choices in [
            vec![(0, 3), (1, -1)],
            vec![(0, 3), (3, 4)],
            vec![(0, 3), (0, 4)],
            vec![(0, 3), (1, 3)],
        ] {
            assert!(map.prepare(&choices).is_none());
            assert_eq!(map.slots, slots);
            assert_eq!(map.checkpoints, checkpoints);
        }
        // Preparing one pool cannot change it if a different pool is invalid.
        let other = EntryMap::new(3, 8);
        let prepared = map.prepare(&[(0, 3)]).unwrap();
        assert!(other.prepare(&[(1, -1)]).is_none());
        drop(prepared);
        assert_eq!(map.slots, slots);
        assert_eq!(map.checkpoints, checkpoints);
        let stale = map.prepare(&[(0, 3)]).unwrap();
        map.prepare(&[(0, 3), (1, 4)])
            .unwrap()
            .commit(&mut map)
            .unwrap();
        assert_eq!(map.workspace_position(0), Some(0));
        assert_eq!(map.workspace_position(1), Some(1));
        assert_eq!(map.workspace_position(3), None);
        assert_eq!(map.workspace_position(4), None);
        let slots = map.slots.clone();
        let checkpoints = map.checkpoints.clone();
        assert!(stale.commit(&mut map).is_none());
        assert_eq!(map.slots, slots);
        assert_eq!(map.checkpoints, checkpoints);
    }
}
