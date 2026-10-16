//! The window ring: a declared buffer of [`Scope::WindowRing`] scope, whose
//! entry for a slot holds the slot's last `tokens` positions, position `p`
//! at `p % tokens` of every plane.
//!
//! A step that may be cancelled or rolled back first saves the cells it
//! touches from the working planes to the shadow (a [`Region::Touched`]
//! cancel save): the last `tokens` positions each item writes, in every
//! layer and plane. Kernels write the working planes and nothing reads the
//! shadow, so commit copies nothing but a rejected verify tail's cells
//! (a [`Rollback::RestoreTail`] buffer), and cancel copies every saved cell
//! back.
//!
//! [`Region::Touched`]: crate::state::Region::Touched
//! [`Rollback::RestoreTail`]: crate::state::Rollback::RestoreTail

use std::collections::BTreeSet;

use crate::ledger::Committed;
use crate::state::{Extent, Invalid, RequestState, Scope, Shadow};

/// One position of one slot in one layer and plane: where it lies in the
/// working planes and in the shadow.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cell {
    pub slot: u32,
    pub layer: u32,
    pub position: u32,
    pub working_offset: u64,
    pub shadow_offset: u64,
    /// Whether the shadow is a separate buffer.
    pub separate: bool,
    pub length: u64,
}

/// A rejected tail of one verify item: positions `start..start + count` of
/// `slot` go back to their values from before the step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tail {
    pub slot: u32,
    pub start: u64,
    pub count: u32,
}

/// The byte layout of one window ring in an instance of one extent.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowRing {
    pub layers: u64,
    /// Bytes of one slot's entry of each plane in one layer.
    pub planes: Vec<u64>,
    pub tokens: u64,
    /// Entries of each plane: the slots, then any spares.
    pub entries: u64,
    /// Slots the ring holds.
    pub slots: u64,
    /// Bytes of the working planes.
    pub working: u64,
    pub shadow: Shadow,
}

impl WindowRing {
    /// The ring of buffer `buffer` of `state` in an instance of `extent`.
    pub fn new(state: &RequestState, buffer: usize, extent: Extent) -> Result<Self, Invalid> {
        state.validate()?;
        state.validate_extent(extent)?;
        let declared = state.buffers.get(buffer).ok_or(Invalid {
            name: "",
            rule: "no such buffer",
        })?;
        let Scope::WindowRing { tokens } = declared.scope else {
            return Err(Invalid {
                name: declared.name,
                rule: "not a window ring",
            });
        };
        Ok(Self {
            layers: declared.layers,
            planes: declared.planes.clone(),
            tokens,
            entries: declared.entries(extent),
            slots: declared.own_entries(extent),
            working: declared.working_bytes(extent),
            shadow: declared.shadow,
        })
    }

    /// Where position `position` of `slot` lies in plane `plane` of layer
    /// `layer` of the working planes, and its length.
    pub fn cell(&self, slot: u32, layer: u64, plane: usize, position: u64) -> (u64, u64) {
        let before = self.planes[..plane].iter().sum::<u64>();
        let stride = self.planes.iter().sum::<u64>() * self.entries;
        let length = self.planes[plane] / self.tokens;
        let offset = layer * stride
            + before * self.entries
            + u64::from(slot) * self.planes[plane]
            + position % self.tokens * length;
        (offset, length)
    }

    fn cells(&self, positions: impl IntoIterator<Item = (u32, u64)>) -> Vec<Cell> {
        let (separate, shadow) = match self.shadow {
            Shadow::None => (false, None),
            Shadow::Adjacent => (false, Some(self.working)),
            Shadow::Separate => (true, Some(0)),
        };
        let mut cells = Vec::new();
        for (slot, position) in positions {
            for layer in 0..self.layers {
                for plane in 0..self.planes.len() {
                    let (offset, length) = self.cell(slot, layer, plane, position);
                    cells.push(Cell {
                        slot,
                        layer: layer as u32,
                        position: (position % self.tokens) as u32,
                        working_offset: offset,
                        shadow_offset: shadow.map_or(offset, |base| base + offset),
                        separate,
                        length,
                    });
                }
            }
        }
        cells
    }

    /// The cells a step's appends touch, each `(slot, start, rows)`: the
    /// last `tokens` positions of each append, by slot and ring position,
    /// each in every layer and plane.
    pub fn touched(&self, appends: &[(u32, u64, u32)]) -> Result<Vec<Cell>, Invalid> {
        let mut touched = BTreeSet::new();
        for &(slot, start, rows) in appends {
            let end = start
                .checked_add(u64::from(rows))
                .filter(|_| u64::from(slot) < self.slots && self.shadow != Shadow::None)
                .ok_or(Invalid {
                    name: "",
                    rule: "an append outside a shadowed ring",
                })?;
            for position in end - u64::from(rows).min(self.tokens)..end {
                touched.insert((slot, position % self.tokens));
            }
        }
        Ok(self.cells(touched))
    }

    /// The cells of rejected tails, by tail, position, layer and plane.
    pub fn tail_cells(&self, tails: &[Tail]) -> Vec<Cell> {
        self.cells(tails.iter().flat_map(|tail| {
            (tail.start..tail.start + u64::from(tail.count)).map(|position| (tail.slot, position))
        }))
    }
}

/// The rejected tails of committed verify items, in item order.
pub fn tails(committed: &[Committed]) -> Vec<Tail> {
    committed
        .iter()
        .filter(|item| item.kept < item.rows)
        .map(|item| Tail {
            slot: item.slot,
            start: item.start + u64::from(item.kept),
            count: item.rows - item.kept,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::{Item, Ledger, LedgerError, Placement, Rules};
    use crate::state::synthetic;

    const EXTENT: Extent = Extent {
        served: 6,
        bucket: 8,
        verify_rows: 3,
    };

    fn ring() -> WindowRing {
        WindowRing::new(&synthetic(), 0, EXTENT).unwrap()
    }

    #[test]
    fn cells_address_slot_major_planes() {
        let ring = ring();
        // Each of the three layers holds K (256 bytes) then V (128) of
        // every one of the bucket's eight slots.
        assert_eq!(
            (ring.entries, ring.slots, ring.working),
            (8, 8, 3 * 8 * 384)
        );
        assert_eq!(ring.cell(5, 0, 0, 3), (5 * 256 + 3 * 32, 32));
        assert_eq!(
            ring.cell(5, 2, 1, 11),
            (2 * 8 * 384 + 8 * 256 + 5 * 128 + 3 * 16, 16)
        );
        // The snapshot's piece of a slot is its whole entry.
        let state = synthetic();
        let pieces = state.pieces(EXTENT, crate::state::Holder::Slot(5)).unwrap();
        assert_eq!(
            (pieces[3].offset, pieces[3].length),
            (ring.cell(5, 1, 1, 0).0, 128)
        );
        assert!(WindowRing::new(&state, 1, EXTENT).is_err());
        assert!(WindowRing::new(&state, 4, EXTENT).is_err());
    }

    #[test]
    fn touched_cells_wrap_and_save_at_most_the_window() {
        let ring = ring();
        let touched = ring.touched(&[(2, 6, 4), (0, 0, 12)]).unwrap();
        let positions = touched
            .iter()
            .filter(|cell| cell.working_offset < 8 * 256)
            .map(|cell| (cell.slot, cell.position))
            .collect::<Vec<_>>();
        // Slot 0 wrote twelve positions: its whole window, once each. Slot
        // 2's four wrap from 6 to 1.
        let expected = (0..8)
            .map(|position| (0, position))
            .chain([(2, 0), (2, 1), (2, 6), (2, 7)])
            .collect::<Vec<_>>();
        assert_eq!(positions, expected);
        assert_eq!(touched.len(), 12 * 3 * 2);
        // Cells go by slot and position, then layer, then plane.
        let first = &touched[..6];
        assert!(
            first
                .iter()
                .all(|cell| (cell.slot, cell.position) == (0, 0))
        );
        assert_eq!(
            first.iter().map(|cell| cell.layer).collect::<Vec<_>>(),
            [0, 0, 1, 1, 2, 2]
        );
        // The adjacent shadow mirrors the working planes after them.
        assert!(touched.iter().all(|cell| {
            !cell.separate && cell.shadow_offset == ring.working + cell.working_offset
        }));
        assert!(ring.touched(&[(8, 0, 1)]).is_err());
        let separate = WindowRing {
            shadow: Shadow::Separate,
            ..ring.clone()
        };
        let cell = separate.touched(&[(1, 0, 1)]).unwrap()[0];
        assert!(cell.separate && cell.shadow_offset == cell.working_offset);
        let unshadowed = WindowRing {
            shadow: Shadow::None,
            ..ring
        };
        assert!(unshadowed.touched(&[(1, 0, 1)]).is_err());
    }

    /// A ring over 8 positions run by the ledger: a verify item that keeps
    /// one of three rows across the wrap restores the other two from the
    /// shadow, and the window then equals a serial append of the kept rows.
    #[test]
    fn a_rejected_tail_restores_across_the_wrap() {
        let ring = ring();
        let mut ledger = Ledger::new(6, Placement::Ledger, Rules::of(&synthetic()), 1 << 20);
        let value = |position: u64| position as u32 * 10 + 1;
        // Slot contents by ring position, working and shadow.
        let mut working = [0_u32; 8];
        let write = |working: &mut [u32; 8], start: u64, rows: u32| {
            for position in start..start + u64::from(rows) {
                working[(position % 8) as usize] = value(position);
            }
        };
        let item = |start, rows, verify| Item {
            request: 4,
            slot: None,
            start,
            rows,
            verify,
        };
        ledger.begin(&[item(0, 6, false)], Vec::new()).unwrap();
        write(&mut working, 0, 6);
        ledger.commit(&[]).unwrap();
        let slot = ledger.slot_of(4).unwrap();
        let step = ledger.begin(&[item(6, 3, true)], Vec::new()).unwrap();
        let touched = ring.touched(&[(slot, step.items[0].start, 3)]).unwrap();
        let saved = working;
        write(&mut working, 6, 3);
        let committed = ledger.commit(&[(4, 1)]).unwrap();
        let tails = tails(&committed);
        assert_eq!(
            tails,
            [Tail {
                slot,
                start: 7,
                count: 2
            }]
        );
        let cells = ring.tail_cells(&tails);
        assert_eq!(cells.len(), 2 * 3 * 2);
        // Every restored cell was saved before the step.
        assert!(cells.iter().all(|cell| touched.contains(cell)));
        for cell in cells.iter().filter(|cell| cell.layer == 0) {
            let position = cell.position as usize;
            working[position] = saved[position];
        }
        let mut serial = [0_u32; 8];
        write(&mut serial, 0, 7);
        assert_eq!(working, serial);
        assert_eq!(
            cells
                .iter()
                .map(|cell| cell.position)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([7, 0])
        );
    }

    /// A verify item of nine rows in an eight-cell ring that keeps one row:
    /// its ninth row overwrote the accepted first one, whose cell the
    /// rejected tail would put back to its value from before the step. The
    /// ledger refuses that commit and keeps the step open, so a cancel can
    /// still restore every touched cell.
    #[test]
    fn a_tail_longer_than_the_window_cannot_commit() {
        let ring = ring();
        let mut ledger = Ledger::new(6, Placement::Ledger, Rules::of(&synthetic()), 1 << 20);
        let item = Item {
            request: 4,
            slot: None,
            start: 0,
            rows: 9,
            verify: true,
        };
        let slot = ledger.begin(&[item], Vec::new()).unwrap().items[0].slot;
        let tail = Tail {
            slot,
            start: 1,
            count: 8,
        };
        // The tail covers the accepted position's cell.
        assert!(
            ring.tail_cells(&[tail])
                .iter()
                .any(|cell| cell.position == 0)
        );
        assert_eq!(ledger.commit(&[(4, 1)]), Err(LedgerError::Kept));
        assert!(ledger.step().is_some());
        assert_eq!(ledger.cancel().unwrap().items[0].slot, slot);
        assert_eq!(ledger.slot_of(4), None);
    }
}
