//! One packed host upload per rank and phase, and block-table deltas.
//!
//! A phase's small input arrays live in one device arena at fixed offsets, so
//! the host fills one byte buffer and uploads it to each rank with one call.
//! Block tables stay resident on the device: the host remembers what each row
//! holds and uploads only the entries that changed.

use std::marker::PhantomData;

use runtime_bundle_abi::{
    BundleAbiError, BundleStreamHandleV1, DeviceBufferSpanV1, HostServicesV1,
};

use crate::geometry::{BundleGeometry, Items, ModelLimits};
use crate::inputs::PhaseInputs;
use crate::resources::upload_bytes;

const ALIGN: u64 = 128;

const fn align(value: u64) -> u64 {
    value.div_ceil(ALIGN) * ALIGN
}

/// Byte offsets of one phase's inputs inside its arena.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InputLayout {
    pub live_batch: u64,
    pub live_tokens: u64,
    /// How many logits rows the phase gathers.
    pub logit_rows: u64,
    pub slot_ids: u64,
    pub q_start: u64,
    pub q_len: u64,
    pub prefix_len: u64,
    pub seq_lens: u64,
    pub last_positions: u64,
    pub token_ids: u64,
    pub positions: u64,
    pub token_batch: u64,
    pub row_indices: u64,
    /// Token array capacity in tokens.
    pub tokens: u64,
    /// Logits row index capacity in rows.
    pub rows: u64,
    /// Item array capacity in items.
    pub items: u64,
    /// End of the last array; the bytes uploaded each step.
    pub end: u64,
}

impl InputLayout {
    /// Lays out `tokens` token slots and `rows` logits rows. Item arrays always
    /// hold `G`'s largest batch.
    pub const fn new<G: BundleGeometry>(tokens: u32, rows: u32) -> Self {
        Self::within(tokens, rows, Items::package::<G>())
    }

    /// Lays out `tokens` token slots and `rows` logits rows. Item arrays hold
    /// the largest graph bucket of `items`.
    pub const fn within(tokens: u32, rows: u32, items: Items) -> Self {
        let tokens = tokens as u64;
        let rows = rows as u64;
        let items = items.capacity() as u64;
        let live_batch = 0;
        let live_tokens = align(live_batch + 4);
        let logit_rows = align(live_tokens + 4);
        let slot_ids = align(logit_rows + 4);
        let q_start = align(slot_ids + 4 * items);
        let q_len = align(q_start + 4 * items);
        let prefix_len = align(q_len + 4 * items);
        let seq_lens = align(prefix_len + 4 * items);
        let last_positions = align(seq_lens + 4 * items);
        let token_ids = align(last_positions + 8 * items);
        let positions = align(token_ids + 4 * tokens);
        let token_batch = align(positions + 4 * tokens);
        let row_indices = align(token_batch + 4 * tokens);
        Self {
            live_batch,
            live_tokens,
            logit_rows,
            slot_ids,
            q_start,
            q_len,
            prefix_len,
            seq_lens,
            last_positions,
            token_ids,
            positions,
            token_batch,
            row_indices,
            tokens,
            rows,
            items,
            end: row_indices + 4 * rows,
        }
    }

    /// Decode graphs bind these offsets once, for the largest bucket. The one
    /// row index slot is never read; it keeps every span inside the arena.
    pub const fn decode<G: BundleGeometry>() -> Self {
        Self::decode_within(Items::package::<G>())
    }

    /// [`Self::decode`] for an instance serving `items`.
    pub const fn decode_within(items: Items) -> Self {
        Self::within(items.capacity(), 1, items)
    }

    /// Verify graphs bind these offsets once, for the largest verify bucket.
    /// A geometry whose verify steps are no wider than its decode batch lays
    /// them out as decode.
    pub fn verify<G: BundleGeometry>() -> Self {
        Self::verify_within::<G>(Items::package::<G>())
    }

    /// [`Self::verify`] for an instance serving `items`.
    pub fn verify_within<G: BundleGeometry>(items: Items) -> Self {
        Self::within(crate::graphs::verify_tokens::<G>(items), 1, items)
    }

    /// Bytes that hold any prefill step's inputs.
    pub const fn prefill_capacity<G: BundleGeometry>() -> u64 {
        Self::prefill_largest::<G>().end
    }

    /// [`Self::prefill_capacity`] for an instance serving `items`.
    pub const fn prefill_capacity_within<G: BundleGeometry>(items: Items) -> u64 {
        Self::prefill_largest_within::<G>(items).end
    }

    /// Prefill offsets are chosen per step, packed around its live counts.
    pub fn prefill<G: BundleGeometry>(phase: &PhaseInputs, rows: usize) -> Self {
        Self::prefill_within::<G>(phase, rows, Items::package::<G>())
    }

    /// [`Self::prefill`] for an instance serving `items`.
    pub fn prefill_within<G: BundleGeometry>(
        phase: &PhaseInputs,
        rows: usize,
        items: Items,
    ) -> Self {
        Self::within(
            phase.live_tokens().div_ceil(G::TOKEN_GRANULE) * G::TOKEN_GRANULE,
            rows as u32,
            items,
        )
    }

    /// The prefill layout for the largest chunk.
    pub const fn prefill_largest<G: BundleGeometry>() -> Self {
        Self::prefill_largest_within::<G>(Items::package::<G>())
    }

    /// [`Self::prefill_largest`] for an instance serving `items`.
    pub const fn prefill_largest_within<G: BundleGeometry>(items: Items) -> Self {
        Self::within(G::PREFILL_TOKENS, G::PREFILL_TOKENS, items)
    }

    /// Fills this phase's arena bytes. `rows` are the logits rows to gather,
    /// which only prefill uploads.
    pub fn pack(&self, phase: &PhaseInputs, rows: &[u32]) -> Option<Vec<u8>> {
        if phase.live_tokens() as u64 > self.tokens
            || rows.len() as u64 > self.rows
            || u64::from(phase.live_batch()) > self.items
        {
            return None;
        }
        let mut bytes = vec![0_u8; self.end as usize];
        let mut put = |offset: u64, values: &mut dyn Iterator<Item = [u8; 4]>| {
            for (index, value) in values.enumerate() {
                let at = offset as usize + 4 * index;
                bytes[at..at + 4].copy_from_slice(&value);
            }
        };
        put(
            self.live_batch,
            &mut std::iter::once(phase.live_batch().to_le_bytes()),
        );
        put(
            self.live_tokens,
            &mut std::iter::once(phase.live_tokens().to_le_bytes()),
        );
        put(
            self.logit_rows,
            &mut std::iter::once((rows.len() as u32).to_le_bytes()),
        );
        for (offset, values) in [
            (self.slot_ids, &phase.slot_ids),
            (self.q_start, &phase.q_start),
            (self.q_len, &phase.q_len),
            (self.prefix_len, &phase.prefix_len),
            (self.seq_lens, &phase.seq_lens),
            (self.positions, &phase.positions),
            (self.token_batch, &phase.token_batch),
        ] {
            put(offset, &mut values.iter().map(|value| value.to_le_bytes()));
        }
        put(
            self.token_ids,
            &mut phase.token_ids.iter().map(|value| value.to_le_bytes()),
        );
        put(
            self.row_indices,
            &mut rows.iter().map(|value| value.to_le_bytes()),
        );
        for (index, value) in phase.last_positions.iter().enumerate() {
            let at = self.last_positions as usize + 8 * index;
            bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
        }
        Some(bytes)
    }
}

/// One rank's device inputs for one phase: the packed arena and the
/// resident block table.
#[derive(Clone, Copy, Debug)]
pub struct PhaseArena {
    pub arena: DeviceBufferSpanV1,
    pub block_tables: DeviceBufferSpanV1,
}

/// The device spans of one phase's inputs.
#[derive(Clone, Copy, Debug)]
pub struct DeviceInputs {
    pub token_ids: DeviceBufferSpanV1,
    pub positions: DeviceBufferSpanV1,
    pub token_batch: DeviceBufferSpanV1,
    pub block_tables: DeviceBufferSpanV1,
    pub slot_ids: DeviceBufferSpanV1,
    pub q_start: DeviceBufferSpanV1,
    pub q_len: DeviceBufferSpanV1,
    pub prefix_len: DeviceBufferSpanV1,
    pub seq_lens: DeviceBufferSpanV1,
    pub live_batch: DeviceBufferSpanV1,
    pub last_positions: DeviceBufferSpanV1,
    pub live_tokens: DeviceBufferSpanV1,
    /// Prefill only: how many rows need logits, and which.
    pub logit_rows: DeviceBufferSpanV1,
    pub row_indices: DeviceBufferSpanV1,
}

impl DeviceInputs {
    /// The spans `layout` places inside one phase arena.
    pub fn within(arena: PhaseArena, layout: &InputLayout) -> Self {
        let span = |offset: u64, bytes: u64| {
            DeviceBufferSpanV1::new(arena.arena.buffer, arena.arena.offset + offset, bytes)
        };
        let items = layout.items;
        Self {
            token_ids: span(layout.token_ids, 4 * layout.tokens),
            positions: span(layout.positions, 4 * layout.tokens),
            block_tables: arena.block_tables,
            token_batch: span(layout.token_batch, 4 * layout.tokens),
            slot_ids: span(layout.slot_ids, 4 * items),
            q_start: span(layout.q_start, 4 * items),
            q_len: span(layout.q_len, 4 * items),
            prefix_len: span(layout.prefix_len, 4 * items),
            seq_lens: span(layout.seq_lens, 4 * items),
            last_positions: span(layout.last_positions, 8 * items),
            live_batch: span(layout.live_batch, 4),
            live_tokens: span(layout.live_tokens, 4),
            logit_rows: span(layout.logit_rows, 4),
            row_indices: span(layout.row_indices, 4 * layout.rows.max(1)),
        }
    }
}

/// Entries to write into one block-table row, starting at entry `first`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RowWrite {
    pub row: usize,
    pub first: usize,
    pub values: Vec<i32>,
}

impl RowWrite {
    /// Byte offset of the write inside the block-table buffer.
    fn offset(&self, max_pages: u32) -> u64 {
        4 * (self.row * max_pages as usize + self.first) as u64
    }

    pub fn bytes(&self) -> Vec<u8> {
        self.values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }
}

/// What each device block-table row of one phase holds: the page list, with
/// -1 in every later entry. `None` means unknown, so the next use rewrites the
/// whole row.
#[derive(Clone, Debug)]
pub struct BlockTableMirror<G> {
    rows: Vec<Option<Vec<i32>>>,
    max_pages: u32,
    geometry: PhantomData<G>,
}

impl<G: BundleGeometry> BlockTableMirror<G> {
    /// A mirror of one row per slot of `items`, every row unknown.
    pub fn new(limits: ModelLimits, items: Items) -> Self {
        assert_eq!(limits.page_tokens(), G::PAGE_TOKENS);
        Self {
            rows: vec![None; items.count()],
            max_pages: limits.max_pages(),
            geometry: PhantomData,
        }
    }

    /// Bytes of the device block table of an instance serving `items`: one
    /// row of `limits.max_pages()` entries per slot.
    pub const fn device_bytes(limits: ModelLimits, items: Items) -> u64 {
        assert!(limits.page_tokens() == G::PAGE_TOKENS);
        4 * limits.max_pages() as u64 * items.get() as u64
    }

    /// Byte offset of a write in this mirror's device block table.
    pub fn offset(&self, write: &RowWrite) -> u64 {
        assert!(write.row < self.rows.len());
        assert!(write.first <= self.max_pages as usize);
        assert!(write.values.len() <= self.max_pages as usize - write.first);
        write.offset(self.max_pages)
    }

    /// The rows this mirror tracks.
    pub fn rows(&self) -> usize {
        self.rows.len()
    }

    /// Records `tables` as the rows' new contents and returns the writes that
    /// bring the device there. Rows past `tables` keep their contents.
    /// `tables` holds at most one table per tracked row: packing refuses a
    /// step with more items first, and a longer list here is a caller bug
    /// that would lose rows.
    pub fn update(&mut self, tables: &[Vec<i32>]) -> Vec<RowWrite> {
        assert!(
            tables.len() <= self.rows.len(),
            "{} block tables for {} rows",
            tables.len(),
            self.rows.len()
        );
        assert!(
            tables
                .iter()
                .all(|pages| pages.len() <= self.max_pages as usize),
            "block table exceeds row stride"
        );
        let mut writes = Vec::new();
        for (row, (pages, known)) in tables.iter().zip(&mut self.rows).enumerate() {
            let (first, end) = match known {
                None => (0, self.max_pages as usize),
                Some(old) => {
                    let same = old.iter().zip(pages).take_while(|(a, b)| a == b).count();
                    (same, old.len().max(pages.len()))
                }
            };
            if first < end {
                let mut values = pages.get(first..).unwrap_or_default().to_vec();
                values.resize(end - first, -1);
                writes.push(RowWrite { row, first, values });
            }
            *known = Some(pages.clone());
        }
        writes
    }

    /// Uploads `writes` (this mirror's [`Self::update`]) into one rank's
    /// device block table `table` on `stream`, in order. A write past
    /// `table` is refused before it is queued.
    pub fn upload(
        &self,
        services: &HostServicesV1,
        stream: BundleStreamHandleV1,
        table: DeviceBufferSpanV1,
        writes: &[RowWrite],
    ) -> Result<(), BundleAbiError> {
        for write in writes {
            let offset = self.offset(write);
            let length = 4 * write.values.len() as u64;
            if offset + length > table.length {
                return Err(BundleAbiError::InvalidRange(
                    "a block-table write leaves its table",
                ));
            }
            let span = DeviceBufferSpanV1::new(table.buffer, table.offset + offset, length);
            upload_bytes(services, stream, span, &write.bytes())?;
        }
        Ok(())
    }

    /// Forgets every row after a failed upload left the device unknown.
    pub fn forget(&mut self) {
        self.rows.iter_mut().for_each(|row| *row = None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::tests::{LIMITS, Tp4};

    const MAX_PAGES: usize = LIMITS.max_pages() as usize;
    const PREFILL_TOKENS: u32 = Tp4::PREFILL_TOKENS;

    #[test]
    #[should_panic(expected = "block table exceeds row stride")]
    fn block_table_refuses_cross_row_overflow() {
        let limits = ModelLimits::new(43, 64, 64).unwrap();
        let mut mirror = BlockTableMirror::<Tp4>::new(limits, Items::new(2).unwrap());
        mirror.update(&[vec![5], vec![9]]);
        mirror.update(&[vec![5, 6], vec![9]]);
    }

    #[test]
    #[should_panic(expected = "block table exceeds row stride")]
    fn block_table_refuses_truncated_initial_row() {
        let limits = ModelLimits::new(43, 64, 64).unwrap();
        BlockTableMirror::<Tp4>::new(limits, Items::new(2).unwrap()).update(&[vec![5, 6]]);
    }

    #[test]
    fn runtime_model_limits_size_each_block_table() {
        let limits = |context_tokens| ModelLimits::new(43, context_tokens, 64).unwrap();
        let items = Items::new(2).unwrap();
        let mut small = BlockTableMirror::<Tp4>::new(limits(128), items);
        let mut large = BlockTableMirror::<Tp4>::new(limits(256), items);
        let tables = [vec![5], vec![6]];
        let small_writes = small.update(&tables);
        let large_writes = large.update(&tables);
        assert_eq!(small_writes[1].values, [6, -1]);
        assert_eq!(large_writes[1].values, [6, -1, -1, -1]);
        assert_eq!(small.offset(&small_writes[1]), 8);
        assert_eq!(large.offset(&large_writes[1]), 16);
        assert_eq!(
            BlockTableMirror::<Tp4>::device_bytes(limits(128), items),
            16
        );
        assert_eq!(
            BlockTableMirror::<Tp4>::device_bytes(limits(256), items),
            32
        );
        assert!(small.update(&tables).is_empty());
        large.forget();
        assert_eq!(large.update(&tables), large_writes);
        assert!(small.update(&tables).is_empty());
    }

    #[test]
    fn layout_aligns_every_array_and_fits_the_arena() {
        let decode = InputLayout::decode::<Tp4>();
        for offset in [
            decode.slot_ids,
            decode.q_start,
            decode.last_positions,
            decode.token_ids,
            decode.positions,
            decode.token_batch,
            decode.row_indices,
        ] {
            assert_eq!(offset % ALIGN, 0);
        }
        assert!(decode.end < 4096);
        assert_eq!(InputLayout::verify::<Tp4>(), decode);
        // Wider verify steps widen the token arrays, not the item arrays,
        // and leave decode's layout alone.
        let wide = InputLayout::verify::<crate::geometry::tests::Wide>();
        assert_eq!((wide.tokens, wide.items), (256, 32));
        assert_eq!(wide.token_ids, decode.token_ids);
        assert!(wide.end < 8192);
        assert_eq!(
            InputLayout::decode::<crate::geometry::tests::Wide>(),
            decode
        );
        assert_eq!(InputLayout::new::<Tp4>(1, 0).token_ids, decode.token_ids);
        let prefill = InputLayout::new::<Tp4>(PREFILL_TOKENS, PREFILL_TOKENS);
        assert_eq!(prefill.end, InputLayout::prefill_capacity::<Tp4>());
        let phase = PhaseInputs {
            token_ids: vec![0; 300],
            ..PhaseInputs::default()
        };
        assert_eq!(InputLayout::prefill::<Tp4>(&phase, 2).tokens, 512);
        assert!(InputLayout::prefill::<Tp4>(&phase, 2).end < prefill.end);
    }

    #[test]
    fn layouts_and_block_tables_follow_the_served_item_count() {
        use crate::geometry::tests::{Max, Wide, every_item_count};
        // The package maximum lays out as before the count was a runtime value.
        let package = Items::package::<Tp4>();
        assert_eq!(
            InputLayout::decode_within(package),
            InputLayout::decode::<Tp4>()
        );
        assert_eq!(
            InputLayout::verify_within::<Wide>(package),
            InputLayout::verify::<Wide>()
        );
        assert_eq!(
            InputLayout::prefill_capacity_within::<Tp4>(package),
            InputLayout::prefill_capacity::<Tp4>()
        );
        for items in every_item_count() {
            let capacity = u64::from(items.capacity());
            let decode = InputLayout::decode_within(items);
            assert_eq!(
                (decode.items, decode.tokens, decode.rows),
                (capacity, capacity, 1)
            );
            let verify = InputLayout::verify_within::<Max>(items);
            assert_eq!((verify.items, verify.tokens), (capacity, 256));
            let prefill = InputLayout::prefill_largest_within::<Max>(items);
            assert_eq!(prefill.items, capacity);
            assert_eq!(
                prefill.end,
                InputLayout::prefill_capacity_within::<Max>(items)
            );
            for layout in [decode, verify, prefill] {
                // Each item array ends before the next array starts.
                for (start, next, width) in [
                    (layout.slot_ids, layout.q_start, 4),
                    (layout.q_start, layout.q_len, 4),
                    (layout.q_len, layout.prefix_len, 4),
                    (layout.prefix_len, layout.seq_lens, 4),
                    (layout.seq_lens, layout.last_positions, 4),
                    (layout.last_positions, layout.token_ids, 8),
                ] {
                    assert_eq!(start % ALIGN, 0);
                    assert!(start + width * capacity <= next, "{items:?}");
                }
                assert!(layout.row_indices + 4 * layout.rows <= layout.end);
            }
            // A full decode step packs; one item more is refused.
            let count = items.count();
            let phase = |live: usize| PhaseInputs {
                decode: true,
                item_order: (0..live).collect(),
                token_ids: vec![1; live],
                slot_ids: (0..live as i32).collect(),
                last_positions: vec![0; live],
                ..PhaseInputs::default()
            };
            let bytes = decode.pack(&phase(count), &[]).unwrap();
            assert_eq!(bytes.len() as u64, decode.end);
            let last = decode.slot_ids as usize + 4 * (count - 1);
            assert_eq!(bytes[last..last + 4], ((count - 1) as u32).to_le_bytes());
            assert!(decode.pack(&phase(capacity as usize + 1), &[]).is_none());
            let mut mirror = BlockTableMirror::<Max>::new(LIMITS, items);
            assert_eq!(mirror.rows(), count);
            assert_eq!(
                BlockTableMirror::<Max>::device_bytes(LIMITS, items),
                4 * u64::from(LIMITS.max_pages()) * count as u64
            );
            let tables = vec![vec![3]; count];
            let writes = mirror.update(&tables);
            assert_eq!(writes.len(), count);
            let end = writes.last().unwrap().offset(LIMITS.max_pages())
                + 4 * u64::from(LIMITS.max_pages());
            assert_eq!(end, BlockTableMirror::<Max>::device_bytes(LIMITS, items));
        }
        assert_eq!(
            BlockTableMirror::<Tp4>::new(LIMITS, Items::package::<Tp4>()).rows(),
            32
        );
    }

    #[test]
    #[should_panic(expected = "33 block tables for 32 rows")]
    fn more_tables_than_rows_are_a_bug_not_a_dropped_write() {
        BlockTableMirror::<Tp4>::new(LIMITS, Items::package::<Tp4>()).update(&vec![vec![3]; 33]);
    }

    #[test]
    fn pack_places_each_array_at_its_offset() {
        let phase = PhaseInputs {
            decode: true,
            item_order: vec![0, 1],
            token_ids: vec![42, 43],
            positions: vec![64, 2],
            token_batch: vec![0, 1],
            slot_ids: vec![3, 4],
            q_start: vec![0, 1],
            q_len: vec![1, 1],
            prefix_len: vec![64, 2],
            seq_lens: vec![65, 3],
            last_positions: vec![64, 2],
            ..PhaseInputs::default()
        };
        let layout = InputLayout::decode::<Tp4>();
        let bytes = layout.pack(&phase, &[]).unwrap();
        let word = |offset: u64| {
            u32::from_le_bytes(
                bytes[offset as usize..offset as usize + 4]
                    .try_into()
                    .unwrap(),
            )
        };
        assert_eq!(bytes.len() as u64, layout.end);
        assert_eq!(word(layout.live_batch), 2);
        assert_eq!(word(layout.live_tokens), 2);
        assert_eq!(word(layout.token_ids + 4), 43);
        assert_eq!(word(layout.slot_ids + 4), 4);
        assert_eq!(word(layout.seq_lens), 65);
        assert_eq!(word(layout.last_positions + 8), 2);
        assert!(layout.pack(&phase, &[0, 1]).is_none());
        let prefill = InputLayout::new::<Tp4>(2, 1);
        let bytes = prefill.pack(&phase, &[1]).unwrap();
        assert_eq!(
            u32::from_le_bytes(
                bytes[prefill.row_indices as usize..][..4]
                    .try_into()
                    .unwrap()
            ),
            1
        );
    }

    #[test]
    fn block_tables_upload_only_changed_entries() {
        let mut mirror = BlockTableMirror::<Tp4>::new(LIMITS, Items::package::<Tp4>());
        let first = mirror.update(&[vec![5, 6], vec![8]]);
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].values.len(), MAX_PAGES);
        assert_eq!(first[0].values[..3], [5, 6, -1]);
        assert_eq!(first[1].offset(LIMITS.max_pages()), 4 * MAX_PAGES as u64);
        assert!(mirror.update(&[vec![5, 6], vec![8]]).is_empty());
        assert_eq!(
            mirror.update(&[vec![5, 6, 9], vec![8]]),
            [RowWrite {
                row: 0,
                first: 2,
                values: vec![9]
            }]
        );
        // A different request in row 0 with a shorter table clears the tail.
        assert_eq!(
            mirror.update(&[vec![7], vec![8]]),
            [RowWrite {
                row: 0,
                first: 0,
                values: vec![7, -1, -1]
            }]
        );
        mirror.forget();
        assert_eq!(mirror.update(&[vec![7]])[0].values.len(), MAX_PAGES);
    }
}
