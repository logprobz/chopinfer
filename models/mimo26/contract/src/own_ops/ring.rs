//! The sliding-window ring's shadow and prefix snapshots
//! (`cuda/mimo_ring.cuh` in the bundle).
//!
//! A rank's ring state is a working half followed by a shadow half of the
//! same layout. Working is `[layer][K plane | V plane]`, each plane
//! `[slot][ring position][entry]`, and sequence position `p` of a slot lives
//! at ring position `p % tokens`. Before a phase writes working, the backup
//! copies every entry it will overwrite into the shadow; a verify step's
//! restore copies its rejected positions back. Nothing else reads the shadow.

use kernel_core::contract::{Refusal, Validate};

use super::{copy, entry};
use crate::Tp;

/// The ring layout, all of it math: the model's window, the instance's
/// slots, its sliding layers and a rank's K and V entry bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RingLayout {
    /// Ring positions per slot: the sliding window.
    pub tokens: u32,
    /// Slots of every plane: one per item of the instance's largest bucket.
    pub slots: u32,
    /// Sliding-window layers.
    pub layers: u32,
    /// Bytes of one position's K entry (the rank's KV heads, BF16).
    pub k_entry: u64,
    /// Bytes of one position's V entry.
    pub v_entry: u64,
}

impl RingLayout {
    /// The ring of `slots` slots at `tp`.
    pub fn of(tp: Tp, slots: u32) -> RingLayout {
        let heads = tp.kv_heads(false);
        RingLayout {
            tokens: tp.geometry.window(),
            slots,
            layers: tp.geometry.sliding_layers() as u32,
            k_entry: heads * tp.geometry.swa_head_dim() as u64 * 2,
            v_entry: heads * tp.geometry.swa_value_head_dim() as u64 * 2,
        }
    }

    /// Bytes of one layer's K plane.
    pub fn k_plane(self) -> u64 {
        u64::from(self.slots) * u64::from(self.tokens) * self.k_entry
    }

    /// Bytes of one layer: its K plane, then its V plane.
    pub fn layer(self) -> u64 {
        u64::from(self.slots) * u64::from(self.tokens) * (self.k_entry + self.v_entry)
    }

    /// Bytes of the working half; the state holds two.
    pub fn working(self) -> u64 {
        u64::from(self.layers) * self.layer()
    }

    /// Bytes of one slot's snapshot of every layer.
    pub fn slot_bytes(self) -> u64 {
        u64::from(self.layers) * u64::from(self.tokens) * (self.k_entry + self.v_entry)
    }

    /// The working offsets of cell `cell` (`slot * tokens + ring position`)
    /// in layer `layer`: its K entry, then its V entry.
    fn entries(self, layer: u32, cell: u64) -> [(u64, u64); 2] {
        let base = u64::from(layer) * self.layer();
        [
            (base + cell * self.k_entry, self.k_entry),
            (base + self.k_plane() + cell * self.v_entry, self.v_entry),
        ]
    }

    /// The cell of `slot`'s position `position`, refusing a slot outside the
    /// ring (the kernels do not check it).
    fn cell(self, slot: i32, position: u32) -> Result<u64, Refusal> {
        if slot < 0 || slot as u32 >= self.slots {
            return Err(Refusal("a ring slot id is outside the ring"));
        }
        Ok(slot as u64 * u64::from(self.tokens) + u64::from(position % self.tokens))
    }

    /// Copies cell `cell` of every layer between the halves.
    fn copy_cell(self, state: &mut [u8], cell: u64, to_shadow: bool) -> Result<(), Refusal> {
        let working = self.working();
        for layer in 0..self.layers {
            for (at, bytes) in self.entries(layer, cell) {
                let (from, to) = if to_shadow {
                    (at, working + at)
                } else {
                    (working + at, at)
                };
                state.copy_within(
                    from as usize..(from + bytes) as usize,
                    usize::try_from(to).map_err(|_| Refusal("ring offset"))?,
                );
            }
        }
        Ok(())
    }

    fn check(self, state: &[u8]) -> Result<(), Refusal> {
        if (state.len() as u64) < 2 * self.working() {
            return Err(Refusal("the ring state holds less than two halves"));
        }
        Ok(())
    }
}

impl Validate for RingLayout {
    fn validate(&self) -> Result<(), Refusal> {
        if self.tokens == 0
            || self.slots == 0
            || self.layers == 0
            || self.k_entry == 0
            || self.v_entry == 0
        {
            return Err(Refusal(
                "a ring layout needs positions, slots, layers and entries",
            ));
        }
        Ok(())
    }
}

/// Saves into the shadow the last `min(q_len, tokens, span)` positions of
/// each live item, every layer, K and V. Item `i` with `q_len[i] > 0` ends
/// at position `prefix_len[i] + q_len[i] - 1`; items with no rows and items
/// past the live count are skipped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RingBackup {
    pub layout: RingLayout,
    /// The most recent positions saved per item (math): one for decode,
    /// the verify rows for verify, the window for prefill.
    pub span: u32,
}

impl Validate for RingBackup {
    fn validate(&self) -> Result<(), Refusal> {
        self.layout.validate()?;
        if self.span == 0 {
            return Err(Refusal("a ring backup saves at least one position"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// A ring backup's buffers: the whole ring state (both halves), each
    /// item's slot, row count and prefix length, and the device count of
    /// live items. The state is copied in 16-byte vectors, so its view
    /// starts at 16-byte alignment; the `i32` tables and the `u32` count at
    /// 4 bytes.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct RingBackupBuffers {
        pub state: value,
        pub slot_ids: value,
        pub q_len: value,
        pub prefix_len: value,
        pub live_items: value,
    }
}

/// What the backup kernel computes over its first `items` items.
pub fn ring_backup(
    op: &RingBackup,
    state: &mut [u8],
    slot_ids: &[i32],
    q_len: &[i32],
    prefix_len: &[i32],
    live_items: u32,
    items: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let layout = op.layout;
    layout.check(state)?;
    for item in 0..items.min(live_items) as usize {
        let count = entry(q_len, item, "q_len")?;
        if count <= 0 {
            continue;
        }
        let touched = (count as u32).min(layout.tokens).min(op.span);
        let last = entry(prefix_len, item, "prefix_len")?.wrapping_add(count - 1) as u32;
        let slot = entry(slot_ids, item, "slot_ids")?;
        for recent in 0..touched {
            let cell = layout.cell(slot, last.wrapping_sub(recent))?;
            layout.copy_cell(state, cell, true)?;
        }
    }
    Ok(())
}

/// Restores from the shadow positions `starts[i] .. starts[i] +
/// min(counts[i], span)` of each live item, every layer, K and V. Items
/// with no count and items past the live count are skipped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RingRestore {
    pub layout: RingLayout,
    /// The most positions restored per item (math): the verify rows.
    pub span: u32,
}

impl Validate for RingRestore {
    fn validate(&self) -> Result<(), Refusal> {
        self.layout.validate()?;
        if self.span == 0 {
            return Err(Refusal("a ring restore restores at least one position"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// A ring restore's buffers: the whole ring state, each item's slot,
    /// first rejected position and rejected count, and the device count of
    /// live items. The state starts at 16-byte alignment, the tables and the
    /// count at 4 bytes.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct RingRestoreBuffers {
        pub state: value,
        pub slot_ids: value,
        pub starts: value,
        pub counts: value,
        pub live_items: value,
    }
}

/// What the restore kernel computes over its first `items` items.
pub fn ring_restore(
    op: &RingRestore,
    state: &mut [u8],
    slot_ids: &[i32],
    starts: &[i32],
    counts: &[i32],
    live_items: u32,
    items: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let layout = op.layout;
    layout.check(state)?;
    for item in 0..items.min(live_items) as usize {
        let count = entry(counts, item, "counts")?;
        if count <= 0 {
            continue;
        }
        let start = entry(starts, item, "starts")? as u32;
        let slot = entry(slot_ids, item, "slot_ids")?;
        for recent in 0..(count as u32).min(op.span) {
            let cell = layout.cell(slot, start.wrapping_add(recent))?;
            layout.copy_cell(state, cell, false)?;
        }
    }
    Ok(())
}

/// Which way a prefix copy moves a snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Direction {
    /// From the slot into the pages.
    Capture,
    /// From the pages into the slot.
    Restore,
}

/// Copies one slot's prefix snapshot between its state and pool pages.
///
/// The snapshot is, end to end: every layer's K then V entries of the slot
/// in the working ring (its whole window, position for position), then
/// `drafter_bytes` of the slot's drafter history. Byte `c` of it lives at
/// `pages[c / page_bytes] * page_bytes + c % page_bytes` of the pool. The
/// slot's whole snapshot holds `pages` pages, at least the
/// `ceil(snapshot bytes / page_bytes)` these pieces fill; its other pieces
/// are moved elsewhere.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RingPrefixCopy {
    pub layout: RingLayout,
    /// Bytes of the slot's drafter history, zero for none.
    pub drafter_bytes: u64,
    /// Bytes of one pool page.
    pub page_bytes: u64,
    /// Pages of the slot's whole snapshot: the page ids its table holds.
    pub pages: u64,
    pub direction: Direction,
}

impl RingPrefixCopy {
    /// Bytes of the snapshot.
    pub fn bytes(&self) -> u64 {
        self.layout.slot_bytes() + self.drafter_bytes
    }

    /// The pool pages these pieces fill.
    pub fn filled_pages(&self) -> u64 {
        self.bytes().div_ceil(self.page_bytes)
    }
}

impl Validate for RingPrefixCopy {
    fn validate(&self) -> Result<(), Refusal> {
        self.layout.validate()?;
        if self.page_bytes == 0 || self.pages < self.filled_pages() {
            return Err(Refusal(
                "a prefix snapshot's pages must hold the pieces it copies",
            ));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// A prefix copy's buffers: the ring state (its working half is read or
    /// written), the slot's drafter history when the snapshot has one, the
    /// pool, its page ids as little-endian `u64`, and the slot. The state,
    /// the drafter piece and the pool are copied in 16-byte vectors and
    /// start at 16-byte alignment; the page ids at 8 bytes.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct RingPrefixCopyBuffers {
        pub state: value,
        pub drafter: option,
        pub pool: value,
        pub pages: value,
        pub slot: scalar(u32),
    }
}

/// What the prefix copy kernel computes for `slot`. `drafter` is the slot's
/// history piece (empty for none).
pub fn ring_prefix_copy(
    op: &RingPrefixCopy,
    state: &mut [u8],
    drafter: &mut [u8],
    pool: &mut [u8],
    pages: &[u64],
    slot: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let layout = op.layout;
    if slot >= layout.slots {
        return Err(Refusal("a prefix copy slot is outside the ring"));
    }
    if (drafter.len() as u64) < op.drafter_bytes {
        return Err(Refusal("the drafter piece is smaller than its snapshot"));
    }
    let pages = pages
        .get(..op.filled_pages() as usize)
        .ok_or(Refusal("a prefix copy has fewer page ids than pages"))?;
    let mut seen = std::collections::BTreeSet::new();
    if pages.iter().any(|&page| !seen.insert(page)) {
        return Err(Refusal("a prefix copy names one page twice"));
    }
    // The snapshot's pieces in order: (in the state, offset, bytes).
    let slot_k = u64::from(layout.tokens) * layout.k_entry;
    let slot_v = u64::from(layout.tokens) * layout.v_entry;
    let mut pieces = (0..u64::from(layout.layers))
        .flat_map(|layer| {
            let base = layer * layout.layer();
            [
                (true, base + u64::from(slot) * slot_k, slot_k),
                (
                    true,
                    base + layout.k_plane() + u64::from(slot) * slot_v,
                    slot_v,
                ),
            ]
        })
        .collect::<Vec<_>>();
    if op.drafter_bytes > 0 {
        pieces.push((false, 0, op.drafter_bytes));
    }
    let mut at = 0_u64;
    for (in_state, offset, bytes) in pieces {
        let held: &mut [u8] = if in_state { state } else { drafter };
        // A piece may cross pages: move it a page run at a time.
        let mut done = 0;
        while done < bytes {
            let byte = at + done;
            let page = byte / op.page_bytes;
            let within = byte % op.page_bytes;
            let run = (op.page_bytes - within).min(bytes - done);
            let pooled = pages[page as usize]
                .checked_mul(op.page_bytes)
                .and_then(|base| base.checked_add(within))
                .ok_or(Refusal("a pool page offset overflows"))?;
            match op.direction {
                Direction::Capture => copy(pool, pooled, held, offset + done, run)?,
                Direction::Restore => copy(held, offset + done, pool, pooled, run)?,
            }
            done += run;
        }
        at += bytes;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> RingLayout {
        RingLayout {
            tokens: 4,
            slots: 2,
            layers: 2,
            k_entry: 2,
            v_entry: 1,
        }
    }

    #[test]
    fn the_served_layout_matches_the_ring_state() {
        for tp in Tp::SUPPORTED {
            for slots in [8, 32, 256] {
                let layout = RingLayout::of(tp, slots);
                assert_eq!(
                    2 * layout.working(),
                    crate::topology::ring_state_bytes(tp, slots)
                );
                assert_eq!(layout.tokens, 128);
                assert_eq!(layout.layers, 39);
            }
        }
    }

    #[test]
    fn the_backup_saves_the_last_span_positions_of_live_items() {
        let layout = layout();
        let working = layout.working() as usize;
        let mut state = (0..2 * working as u8).collect::<Vec<_>>();
        let before = state.clone();
        // Item 0: slot 1, rows 3 ending at position 5 (ring 1, 0, 3 with
        // span 3, but span 2 keeps 1 and 0). Item 1 is not live.
        ring_backup(
            &RingBackup { layout, span: 2 },
            &mut state,
            &[1, 0],
            &[3, 1],
            &[3, 0],
            1,
            2,
        )
        .unwrap();
        assert_eq!(state[..working], before[..working]);
        let mut expected = before.clone();
        for cell in [4 + 1, 4] {
            for layer in 0..2 {
                for (at, bytes) in layout.entries(layer, cell) {
                    let (at, bytes) = (at as usize, bytes as usize);
                    expected[working + at..][..bytes].copy_from_slice(&before[at..][..bytes]);
                }
            }
        }
        assert_eq!(state, expected);
        // A live item outside the ring is an input error.
        assert!(
            ring_backup(
                &RingBackup { layout, span: 1 },
                &mut state,
                &[2],
                &[1],
                &[0],
                1,
                1
            )
            .is_err()
        );
    }

    #[test]
    fn restore_undoes_a_backup_of_the_same_cells() {
        let layout = layout();
        let working = layout.working() as usize;
        let mut state = (0..2 * working as u8).collect::<Vec<_>>();
        let original = state.clone();
        ring_backup(
            &RingBackup { layout, span: 4 },
            &mut state,
            &[0],
            &[4],
            &[6],
            1,
            1,
        )
        .unwrap();
        let saved = state.clone();
        state[..working].fill(0xee);
        // Restore all four positions 6..10, ring 2, 3, 0, 1.
        ring_restore(
            &RingRestore { layout, span: 4 },
            &mut state,
            &[0],
            &[6],
            &[4],
            1,
            1,
        )
        .unwrap();
        let mut expected = saved.clone();
        for cell in 0..4 {
            for layer in 0..2 {
                for (at, bytes) in layout.entries(layer, cell) {
                    let (at, bytes) = (at as usize, bytes as usize);
                    expected[at..][..bytes].copy_from_slice(&original[at..][..bytes]);
                }
            }
        }
        // Slot 1's cells were not restored.
        for cell in 4..8 {
            for layer in 0..2 {
                for (at, bytes) in layout.entries(layer, cell) {
                    expected[at as usize..][..bytes as usize].fill(0xee);
                }
            }
        }
        assert_eq!(state, expected);
    }

    #[test]
    fn a_prefix_snapshot_round_trips_through_scattered_pages() {
        let layout = layout();
        let op = RingPrefixCopy {
            layout,
            drafter_bytes: 4,
            page_bytes: 5,
            pages: 7,
            direction: Direction::Capture,
        };
        assert_eq!(op.bytes(), 2 * 4 * 3 + 4);
        assert_eq!(op.filled_pages(), 6);
        assert!(RingPrefixCopy { pages: 5, ..op }.validate().is_err());
        let mut state = (0..2 * layout.working() as u8).collect::<Vec<_>>();
        let mut drafter = vec![200, 201, 202, 203];
        let mut pool = vec![0_u8; 8 * 5];
        let pages = [7, 1, 3, 0, 6, 2];
        ring_prefix_copy(&op, &mut state, &mut drafter, &mut pool, &pages, 1).unwrap();
        // Slot 1's layer 0 K entries are state bytes 8..16, V 20..24; layer 1
        // K 32..40, V 44..48.
        let snapshot = [8..16, 20..24, 32..40, 44..48]
            .into_iter()
            .flat_map(|range| range.map(|byte| byte as u8))
            .chain(200..204)
            .collect::<Vec<_>>();
        for (index, chunk) in snapshot.chunks(5).enumerate() {
            let at = pages[index] as usize * 5;
            assert_eq!(&pool[at..at + chunk.len()], chunk);
        }
        assert_eq!(pool[20..30], [0; 10]);
        let mut restored = vec![0_u8; state.len()];
        let mut history = vec![0_u8; 4];
        let restore = RingPrefixCopy {
            direction: Direction::Restore,
            ..op
        };
        ring_prefix_copy(&restore, &mut restored, &mut history, &mut pool, &pages, 0).unwrap();
        assert_eq!(history, [200, 201, 202, 203]);
        assert_eq!(restored[0..8], snapshot[0..8]);
        assert_eq!(restored[16..20], snapshot[8..12]);
        assert!(ring_prefix_copy(&op, &mut state, &mut drafter, &mut pool, &pages, 2).is_err());
        assert!(
            ring_prefix_copy(
                &op,
                &mut state,
                &mut drafter,
                &mut pool,
                &[1, 1, 2, 3, 4, 5],
                0
            )
            .is_err()
        );
    }
}
