//! Qwen3.8's per-phase inputs beside the shared input arena: recurrent pool
//! entries, row offsets, QSA piece counts and M-RoPE positions. Like the
//! shared arena, one packed upload per rank and phase.

use runtime_bundle_abi::DeviceBufferSpanV1;

use crate::contract::{MAX_VERIFY_ROWS, SCORE_ROWS};
use crate::inputs::PhaseInputs;

pub(crate) fn valid_rope_position(shape: crate::contract::Shape, position: i32) -> bool {
    (0..shape.context_tokens() as i32).contains(&position)
}

const ALIGN: u64 = 128;

const fn align(value: u64) -> u64 {
    value.div_ceil(ALIGN) * ALIGN
}

/// QSA pieces of one prefill chunk.
pub const PIECES: u32 = crate::contract::PREFILL_TOKENS / SCORE_ROWS;

/// Byte offsets of one phase's side inputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SideLayout {
    shape: crate::contract::Shape,
    /// Absolute row offsets of the items, then the total: `[items + 1]`.
    pub cu_rows: u64,
    /// The committed pool entry of each item's slot; -1 before any state.
    pub committed: u64,
    /// The entry a prefill item reads and writes.
    pub working: u64,
    /// The entries a verify item writes after each row: `[items][8]`.
    pub checkpoints: u64,
    /// Nonzero where the item continues earlier state.
    pub initial: u64,
    pub ple_working: u64,
    pub ple_checkpoints: u64,
    /// 0, 1, 2, ...: items in phase order.
    pub item_ids: u64,
    /// Per item, the linear-state pool entry that keeps its state at its
    /// anchor, or -1 (`ItemEntries::anchor`).
    pub anchor_entry: u64,
    /// Per item, its anchor item (the PLE, ring and retained spares), or -1.
    pub anchor_slot: u64,
    /// Per item, the position its anchor's ring ends before.
    pub anchor_end: u64,
    /// Live rows of each QSA scoring piece.
    pub piece_rows: u64,
    /// (T, H, W) RoPE positions per token.
    pub rope_positions: u64,
    pub tokens: u64,
    /// Item array capacity in items.
    pub items: u64,
    pub end: u64,
}

impl SideLayout {
    /// Lays out `tokens` token rows and item arrays of `items` items: the
    /// largest graph bucket of the served items (`scratch::Served::items`).
    #[cfg(test)]
    pub const fn new(tokens: u32, items: u32) -> Self {
        Self::for_shape(crate::contract::Shape::TP4, tokens, items)
    }

    pub const fn for_shape(shape: crate::contract::Shape, tokens: u32, items: u32) -> Self {
        let tokens = tokens as u64;
        let items = items as u64;
        let cu_rows = 0;
        let committed = align(cu_rows + 4 * (items + 1));
        let working = align(committed + 4 * items);
        let checkpoints = align(working + 4 * items);
        let initial = align(checkpoints + 4 * items * MAX_VERIFY_ROWS as u64);
        let ple_working = align(initial + 4 * items);
        let ple_checkpoints = align(ple_working + 4 * items);
        let item_ids = align(ple_checkpoints + 4 * items * MAX_VERIFY_ROWS as u64);
        let anchor_entry = align(item_ids + 4 * items);
        let anchor_slot = align(anchor_entry + 4 * items);
        let anchor_end = align(anchor_slot + 4 * items);
        let piece_rows = align(anchor_end + 4 * items);
        let rope_positions = align(piece_rows + 4 * PIECES as u64);
        Self {
            shape,
            cu_rows,
            committed,
            working,
            checkpoints,
            initial,
            ple_working,
            ple_checkpoints,
            item_ids,
            anchor_entry,
            anchor_slot,
            anchor_end,
            piece_rows,
            rope_positions,
            tokens,
            items,
            end: rope_positions + 12 * tokens,
        }
    }

    /// Fills this phase's side bytes from its packed inputs and its items'
    /// pool entries.
    pub fn pack(&self, phase: &PhaseInputs, entries: &[ItemEntries]) -> Option<Vec<u8>> {
        let items = phase.item_order.len();
        if phase.token_ids.len() as u64 > self.tokens
            || items as u64 > self.items
            || entries.len() != items
            || !(phase.position_axes.is_empty()
                || phase.position_axes.len() == phase.token_ids.len())
        {
            return None;
        }
        let valid = if phase.position_axes.is_empty() {
            phase
                .positions
                .iter()
                .copied()
                .all(|position| valid_rope_position(self.shape, position))
        } else {
            phase
                .position_axes
                .iter()
                .flatten()
                .copied()
                .all(|position| valid_rope_position(self.shape, position))
        };
        if !valid {
            return None;
        }
        let mut bytes = vec![0_u8; self.end as usize];
        let mut put = |offset: u64, index: usize, value: i32| {
            let at = offset as usize + 4 * index;
            bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
        };
        for (index, &start) in phase.q_start.iter().enumerate() {
            put(self.cu_rows, index, start);
        }
        put(self.cu_rows, items, phase.token_ids.len() as i32);
        for (index, entry) in entries.iter().enumerate() {
            put(self.committed, index, entry.committed);
            put(self.working, index, entry.working);
            for (row, &checkpoint) in entry.checkpoints.iter().enumerate() {
                put(
                    self.checkpoints,
                    index * MAX_VERIFY_ROWS as usize + row,
                    checkpoint,
                );
            }
            put(self.initial, index, i32::from(entry.committed >= 0));
            put(self.ple_working, index, entry.ple_working);
            for (row, &checkpoint) in entry.ple_checkpoints.iter().enumerate() {
                put(
                    self.ple_checkpoints,
                    index * MAX_VERIFY_ROWS as usize + row,
                    checkpoint,
                );
            }
            put(self.item_ids, index, index as i32);
            let anchor = entry.anchor.unwrap_or(Anchor {
                entry: -1,
                slot: -1,
                end: 0,
            });
            put(self.anchor_entry, index, anchor.entry);
            put(self.anchor_slot, index, anchor.slot);
            put(self.anchor_end, index, anchor.end);
        }
        let tokens = phase.token_ids.len() as u32;
        for piece in 0..PIECES {
            let live = tokens.saturating_sub(piece * SCORE_ROWS).min(SCORE_ROWS);
            put(self.piece_rows, piece as usize, live as i32);
        }
        // Explicit positions come from the host's position rule; without
        // them, text rows give every M-RoPE axis the token position.
        if phase.position_axes.is_empty() {
            for (token, &position) in phase.positions.iter().enumerate() {
                for axis in 0..3 {
                    put(self.rope_positions, 3 * token + axis, position);
                }
            }
        } else {
            for (token, axes) in phase.position_axes.iter().enumerate() {
                for (axis, &position) in axes.iter().enumerate() {
                    put(self.rope_positions, 3 * token + axis, position);
                }
            }
        }
        Some(bytes)
    }
}

/// Pool entries of one item: the committed state (or -1 for a fresh
/// sequence), the entry a prefill writes, and the entry each verify row
/// writes (all the working entry without speculation).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ItemEntries {
    pub committed: i32,
    pub working: i32,
    pub ple_working: i32,
    pub ple_checkpoints: [i32; MAX_VERIFY_ROWS as usize],
    pub checkpoints: [i32; MAX_VERIFY_ROWS as usize],
    /// Where a prompt-ending prefill item keeps its state at its anchor.
    pub anchor: Option<Anchor>,
}

/// Where one item keeps its state at an anchor inside its rows: its pool
/// entry (-1 for a phase without linear layers), its anchor item, and the
/// position the kept ring ends before (the anchor, or for the drafter the
/// anchor's frontier).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Anchor {
    pub entry: i32,
    pub slot: i32,
    pub end: i32,
}

/// The device spans of one phase's side inputs.
#[derive(Clone, Copy, Debug)]
pub struct SideInputs {
    pub cu_rows: DeviceBufferSpanV1,
    pub committed: DeviceBufferSpanV1,
    pub working: DeviceBufferSpanV1,
    pub checkpoints: DeviceBufferSpanV1,
    pub initial: DeviceBufferSpanV1,
    pub ple_working: DeviceBufferSpanV1,
    pub ple_checkpoints: DeviceBufferSpanV1,
    pub item_ids: DeviceBufferSpanV1,
    pub anchor_entry: DeviceBufferSpanV1,
    pub anchor_slot: DeviceBufferSpanV1,
    pub anchor_end: DeviceBufferSpanV1,
    pub piece_rows: DeviceBufferSpanV1,
    pub rope_positions: DeviceBufferSpanV1,
}

impl SideInputs {
    pub fn within(arena: DeviceBufferSpanV1, layout: &SideLayout) -> Self {
        let span = |offset: u64, bytes: u64| {
            DeviceBufferSpanV1::new(arena.buffer, arena.offset + offset, bytes)
        };
        let items = layout.items;
        Self {
            cu_rows: span(layout.cu_rows, 4 * (items + 1)),
            committed: span(layout.committed, 4 * items),
            working: span(layout.working, 4 * items),
            checkpoints: span(layout.checkpoints, 4 * items * u64::from(MAX_VERIFY_ROWS)),
            initial: span(layout.initial, 4 * items),
            ple_working: span(layout.ple_working, 4 * items),
            ple_checkpoints: span(
                layout.ple_checkpoints,
                4 * items * u64::from(MAX_VERIFY_ROWS),
            ),
            item_ids: span(layout.item_ids, 4 * items),
            anchor_entry: span(layout.anchor_entry, 4 * items),
            anchor_slot: span(layout.anchor_slot, 4 * items),
            anchor_end: span(layout.anchor_end, 4 * items),
            piece_rows: span(layout.piece_rows, 4 * u64::from(PIECES)),
            rope_positions: span(layout.rope_positions, 12 * layout.tokens),
        }
    }

    /// The live-row scalar of QSA piece `piece`.
    pub fn piece_live(&self, piece: u32) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(
            self.piece_rows.buffer,
            self.piece_rows.offset + 4 * u64::from(piece),
            4,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn side_inputs_carry_entries_offsets_and_positions() {
        let phase = PhaseInputs {
            item_order: vec![0, 1],
            token_ids: vec![5; 2050],
            positions: (0..2050).collect(),
            q_start: vec![0, 2000],
            q_len: vec![2000, 50],
            ..PhaseInputs::default()
        };
        let layout = SideLayout::new(2304, 32);
        let bytes = layout
            .pack(
                &phase,
                &[
                    ItemEntries {
                        ple_working: 0,
                        ple_checkpoints: [0; 8],
                        committed: -1,
                        working: 1,
                        checkpoints: [1; 8],
                        anchor: None,
                    },
                    ItemEntries {
                        ple_working: 0,
                        ple_checkpoints: [0; 8],
                        committed: 4,
                        working: 5,
                        checkpoints: [5, 6, 7, 3, 3, 3, 3, 3],
                        anchor: Some(Anchor {
                            entry: 40,
                            slot: 2,
                            end: 2048,
                        }),
                    },
                ],
            )
            .unwrap();
        let word = |offset: u64, index: usize| {
            i32::from_le_bytes(
                bytes[offset as usize + 4 * index..][..4]
                    .try_into()
                    .unwrap(),
            )
        };
        assert_eq!(bytes.len() as u64, layout.end);
        assert_eq!(
            [word(layout.cu_rows, 1), word(layout.cu_rows, 2)],
            [2000, 2050]
        );
        assert_eq!([word(layout.initial, 0), word(layout.initial, 1)], [0, 1]);
        assert_eq!(word(layout.checkpoints, 8), 5);
        assert_eq!(word(layout.checkpoints, 11), 3);
        assert_eq!(
            [word(layout.anchor_entry, 0), word(layout.anchor_entry, 1)],
            [-1, 40]
        );
        assert_eq!(
            [word(layout.anchor_slot, 1), word(layout.anchor_end, 1)],
            [2, 2048]
        );
        assert_eq!(
            [word(layout.piece_rows, 0), word(layout.piece_rows, 1)],
            [2048, 2]
        );
        assert_eq!(word(layout.rope_positions, 3 * 7 + 2), 7);
        assert!(layout.pack(&phase, &[]).is_none());
        assert!(SideLayout::new(32, 32).end < 4096);
        // Wider item arrays move every later array; 32 items keep the
        // layout plus the PLE entry tables.
        let wide = SideLayout::new(256, 256);
        assert_eq!(wide.items, 256);
        assert!(wide.checkpoints - wide.working >= 4 * 256);
        assert!(wide.rope_positions - wide.piece_rows >= 4 * u64::from(PIECES));
        assert_eq!(
            (
                SideLayout::new(32, 32).committed,
                SideLayout::new(32, 32).end
            ),
            (256, 3_840)
        );
    }

    #[test]
    fn explicit_position_axes_replace_the_replicated_position() {
        let entries = [ItemEntries {
            ple_working: 0,
            ple_checkpoints: [0; 8],
            committed: -1,
            working: 1,
            checkpoints: [1; MAX_VERIFY_ROWS as usize],
            anchor: None,
        }];
        let mut phase = PhaseInputs {
            item_order: vec![0],
            token_ids: vec![5; 4],
            positions: (0..4).collect(),
            position_axes: vec![[0, 0, 0], [1, 1, 1], [2, 2, 3], [2, 3, 2]],
            q_start: vec![0],
            q_len: vec![4],
            ..PhaseInputs::default()
        };
        let layout = SideLayout::new(32, 32);
        let bytes = layout.pack(&phase, &entries).unwrap();
        let rows = (0..12)
            .map(|index| {
                let at = layout.rope_positions as usize + 4 * index;
                i32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
            })
            .collect::<Vec<_>>();
        assert_eq!(rows, [0, 0, 0, 1, 1, 1, 2, 2, 3, 2, 3, 2]);
        phase.position_axes.pop();
        assert!(layout.pack(&phase, &entries).is_none());
    }
}
