//! Copies of per-item state (`cuda/qwen38_bundle.cuh` in the bundle): state
//! entries between planes of entries, and one slot's prefix-cache snapshot
//! between its state and scattered pool pages. Both move bytes without
//! reading them.

use std::collections::BTreeSet;

use kernel_core::contract::{Refusal, Validate};

use super::{copy, entry};
use crate::contract::Shape;

/// Copies entry `src_ids[i]` of every plane of the source to entry
/// `dst_ids[i]` of the destination for each live item with a source
/// (`src_ids[i] >= 0`). Planes are `plane_stride` bytes apart in both. All
/// math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CopyEntries {
    pub entry_bytes: u64,
    pub planes: u32,
    pub plane_stride: u64,
}

impl Validate for CopyEntries {
    fn validate(&self) -> Result<(), Refusal> {
        if self.entry_bytes == 0
            || self.planes == 0
            || (self.planes > 1 && self.plane_stride < self.entry_bytes)
        {
            return Err(Refusal(
                "an entry copy needs entries and planes that do not overlap",
            ));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// An entry copy's buffers: the source and destination planes (one
    /// buffer for a copy in place), the per-item source and destination
    /// ids and the device `u32` count of live items. The planes are copied
    /// in 16-byte vectors and start at 16-byte alignment; the ids and the
    /// count at 4.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct CopyEntriesBuffers {
        pub src: value,
        pub dst: value,
        pub src_ids: value,
        pub dst_ids: value,
        pub live_items: value,
    }
}

/// What the copy kernel writes into `dst` over its first `items` items,
/// reading `src`. A copy in place passes the buffer's bytes before the copy
/// as `src`; it is the kernel's only when no item's destination is another
/// item's source. Two items writing one entry are refused: the kernel's
/// result would depend on block order.
#[allow(clippy::too_many_arguments)]
pub fn copy_entries(
    op: &CopyEntries,
    src: &[u8],
    dst: &mut [u8],
    src_ids: &[i32],
    dst_ids: &[i32],
    live_items: u32,
    items: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let mut written = BTreeSet::new();
    for item in 0..items.min(live_items) {
        let from = entry(src_ids, item as usize, "src_ids")?;
        if from < 0 {
            continue;
        }
        let to = u64::try_from(entry(dst_ids, item as usize, "dst_ids")?)
            .map_err(|_| Refusal("an entry copy's destination is negative"))?;
        if !written.insert(to) {
            return Err(Refusal("two items copy into one entry"));
        }
        for plane in 0..u64::from(op.planes) {
            let base = plane * op.plane_stride;
            copy(
                dst,
                base + to * op.entry_bytes,
                src,
                base + from as u64 * op.entry_bytes,
                op.entry_bytes,
            )?;
        }
    }
    Ok(())
}

/// The state buffers a snapshot's pieces live in.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Buffer {
    /// Every linear layer's convolution entries, one plane per layer.
    Conv,
    /// Every linear layer's delta-rule entries, one plane per layer.
    Delta,
    /// The PLE state entries.
    Ple,
    /// The QSA raw-key ring: per layer every slot's keys, then their
    /// positions.
    Ring,
    /// Every slot's retained drafter rows.
    Retained,
}

/// One contiguous piece of a snapshot: `length` bytes of `buffer` from
/// `offset`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Piece {
    pub buffer: Buffer,
    pub offset: u64,
    pub length: u64,
}

/// Where one slot's snapshot lives: its pool entry in every linear plane,
/// its PLE entry, its ring slot in every ring layer and, with a drafter,
/// its retained rows. All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Snapshot {
    pub linear_layers: u32,
    pub conv_entry_bytes: u64,
    pub delta_entry_bytes: u64,
    /// Entries of each linear-state plane.
    pub pool_entries: u32,
    pub ple_entry_bytes: u64,
    pub ple_entries: u32,
    pub ring_layers: u32,
    pub ring_slots: u32,
    /// One slot's keys and positions in one ring layer.
    pub ring_key_bytes: u64,
    pub ring_position_bytes: u64,
    /// One slot's retained drafter rows, zero without a drafter.
    pub retained_bytes: u64,
}

impl Snapshot {
    /// `shape`'s snapshot over `pool_entries` linear entries, `ple_entries`
    /// PLE entries and `ring_layers` layers of a `ring_slots`-slot ring, with
    /// the drafter's retained rows when `retained`.
    pub fn of(
        shape: Shape,
        pool_entries: u32,
        ple_entries: u32,
        ring_layers: u32,
        ring_slots: u32,
        retained: bool,
    ) -> Snapshot {
        Snapshot {
            linear_layers: shape.linear_layers() as u32,
            conv_entry_bytes: shape.conv_entry_bytes,
            delta_entry_bytes: shape.delta_entry_bytes,
            pool_entries,
            ple_entry_bytes: shape.ple_entry_bytes(),
            ple_entries,
            ring_layers,
            ring_slots,
            ring_key_bytes: shape.ring_slot_key_bytes(),
            ring_position_bytes: shape.ring_slot_position_bytes(),
            retained_bytes: if retained {
                u64::from(crate::contract::MAX_VERIFY_ROWS) * u64::from(shape.wide()) * 2
            } else {
                0
            },
        }
    }

    /// The pieces of the snapshot of ring and retained slot `slot`, linear
    /// pool entry `index` and PLE entry `ple_entry`, in snapshot order:
    /// every convolution plane's entry, every delta-rule plane's, the PLE
    /// entry, each ring layer's keys and positions, then the retained rows.
    pub fn pieces(self, held: Held) -> Vec<Piece> {
        let piece = |buffer, offset, length| Piece {
            buffer,
            offset,
            length,
        };
        let index = u64::from(held.index);
        let slot = u64::from(held.slot);
        let plane = |buffer, entry_bytes: u64| {
            (0..u64::from(self.linear_layers)).map(move |layer| {
                piece(
                    buffer,
                    (layer * u64::from(self.pool_entries) + index) * entry_bytes,
                    entry_bytes,
                )
            })
        };
        let ring_layer =
            u64::from(self.ring_slots) * (self.ring_key_bytes + self.ring_position_bytes);
        let mut pieces = plane(Buffer::Conv, self.conv_entry_bytes)
            .chain(plane(Buffer::Delta, self.delta_entry_bytes))
            .collect::<Vec<_>>();
        pieces.push(piece(
            Buffer::Ple,
            u64::from(held.ple_entry) * self.ple_entry_bytes,
            self.ple_entry_bytes,
        ));
        for layer in 0..u64::from(self.ring_layers) {
            pieces.push(piece(
                Buffer::Ring,
                layer * ring_layer + slot * self.ring_key_bytes,
                self.ring_key_bytes,
            ));
            pieces.push(piece(
                Buffer::Ring,
                layer * ring_layer
                    + u64::from(self.ring_slots) * self.ring_key_bytes
                    + slot * self.ring_position_bytes,
                self.ring_position_bytes,
            ));
        }
        if self.retained_bytes > 0 {
            pieces.push(piece(
                Buffer::Retained,
                slot * self.retained_bytes,
                self.retained_bytes,
            ));
        }
        pieces
    }

    /// Bytes of one snapshot.
    pub fn bytes(self) -> u64 {
        u64::from(self.linear_layers) * (self.conv_entry_bytes + self.delta_entry_bytes)
            + self.ple_entry_bytes
            + u64::from(self.ring_layers) * (self.ring_key_bytes + self.ring_position_bytes)
            + self.retained_bytes
    }

    /// Bytes of each state buffer the snapshots of every slot address.
    pub fn buffer_bytes(self, buffer: Buffer, slots: u32) -> u64 {
        match buffer {
            Buffer::Conv => {
                u64::from(self.linear_layers) * u64::from(self.pool_entries) * self.conv_entry_bytes
            }
            Buffer::Delta => {
                u64::from(self.linear_layers)
                    * u64::from(self.pool_entries)
                    * self.delta_entry_bytes
            }
            Buffer::Ple => u64::from(self.ple_entries) * self.ple_entry_bytes,
            Buffer::Ring => {
                u64::from(self.ring_layers)
                    * u64::from(self.ring_slots)
                    * (self.ring_key_bytes + self.ring_position_bytes)
            }
            Buffer::Retained => u64::from(slots) * self.retained_bytes,
        }
    }
}

/// Which slot a copy moves: its ring and retained slot, its linear pool
/// entry and its PLE entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Held {
    pub slot: u32,
    pub index: u32,
    pub ple_entry: u32,
}

impl Held {
    /// Whether the kernels copy this slot: entries and slot within the
    /// snapshot's planes and ring. They copy nothing otherwise.
    pub fn within(self, snapshot: Snapshot) -> bool {
        self.slot < snapshot.ring_slots
            && self.index < snapshot.pool_entries
            && self.ple_entry < snapshot.ple_entries
    }
}

/// Which way a prefix copy moves a snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Direction {
    /// From the slot's state into the pages.
    Capture,
    /// From the pages back into the slot's state.
    Restore,
}

/// Copies one slot's snapshot between where it lives and `pages` pages of
/// `page_bytes` bytes: byte `c` of the snapshot ([`Snapshot::pieces`] end
/// to end) is byte `c % page_bytes` of page `pages[c / page_bytes]`. `pages`
/// is the snapshot table's page count, which holds the pieces. All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefixCopy {
    pub snapshot: Snapshot,
    pub page_bytes: u64,
    pub pages: u64,
    pub direction: Direction,
}

impl Validate for PrefixCopy {
    fn validate(&self) -> Result<(), Refusal> {
        if self.page_bytes == 0
            || self
                .pages
                .checked_mul(self.page_bytes)
                .is_none_or(|bytes| bytes < self.snapshot.bytes())
        {
            return Err(Refusal("a prefix copy's pages must hold its snapshot"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// A prefix copy's buffers: the convolution, delta-rule and PLE pools,
    /// the ring, every slot's retained rows (with a drafter), the KV pool
    /// the pages are in, the snapshot's `u64` page ids, and the slot, pool
    /// entry and PLE entry the copy moves. The state buffers and the pool
    /// are copied in 16-byte vectors and start at 16-byte alignment; the
    /// page ids at 8.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct PrefixCopyBuffers {
        pub conv_pool: value,
        pub delta_pool: value,
        pub ple_pool: value,
        pub ring: value,
        pub retained: option,
        pub pool: value,
        pub pages: value,
        pub slot: scalar(u32),
        pub index: scalar(u32),
        pub ple_entry: scalar(u32),
    }
}

/// The runs of snapshot bytes `[at, at + length)` within one page each:
/// (snapshot offset, pool offset, bytes).
fn page_runs(
    op: &PrefixCopy,
    pages: &[u64],
    at: u64,
    length: u64,
) -> Result<Vec<(u64, u64, u64)>, Refusal> {
    let mut runs = Vec::new();
    let mut done = 0;
    while done < length {
        let byte = at + done;
        let page = entry(
            pages,
            (byte / op.page_bytes) as usize,
            "a snapshot byte has no page",
        )?;
        let within = byte % op.page_bytes;
        let bytes = (op.page_bytes - within).min(length - done);
        runs.push((done, page * op.page_bytes + within, bytes));
        done += bytes;
    }
    Ok(runs)
}

fn distinct(pages: &[u64], count: u64) -> Result<(), Refusal> {
    let used = pages
        .get(..count as usize)
        .ok_or(Refusal("a prefix copy has fewer pages than its snapshot"))?;
    if used.iter().collect::<BTreeSet<_>>().len() != used.len() {
        return Err(Refusal("a prefix copy names a page twice"));
    }
    Ok(())
}

/// What a capture writes into `pool`, reading each piece through `read`;
/// a restore's piece bytes from `pool`, handed to `write` in snapshot
/// order. Either copies nothing when `held` is not [`Held::within`] the
/// snapshot. A page named twice is refused.
pub fn prefix_capture(
    op: &PrefixCopy,
    held: Held,
    mut read: impl FnMut(Piece) -> Result<Vec<u8>, Refusal>,
    pages: &[u64],
    pool: &mut [u8],
) -> Result<(), Refusal> {
    op.validate()?;
    if !held.within(op.snapshot) {
        return Ok(());
    }
    distinct(pages, op.snapshot.bytes().div_ceil(op.page_bytes))?;
    let mut at = 0;
    for piece in op.snapshot.pieces(held) {
        let bytes = read(piece)?;
        if bytes.len() as u64 != piece.length {
            return Err(Refusal("a snapshot piece read has the wrong length"));
        }
        for (from, to, length) in page_runs(op, pages, at, piece.length)? {
            copy(pool, to, &bytes, from, length)?;
        }
        at += piece.length;
    }
    Ok(())
}

/// The restore of [`prefix_capture`]: each piece's bytes from the pages,
/// in snapshot order.
pub fn prefix_restore(
    op: &PrefixCopy,
    held: Held,
    pages: &[u64],
    pool: &[u8],
    mut write: impl FnMut(Piece, &[u8]) -> Result<(), Refusal>,
) -> Result<(), Refusal> {
    op.validate()?;
    if !held.within(op.snapshot) {
        return Ok(());
    }
    distinct(pages, op.snapshot.bytes().div_ceil(op.page_bytes))?;
    let mut at = 0;
    for piece in op.snapshot.pieces(held) {
        let mut bytes = vec![0; piece.length as usize];
        for (to, from, length) in page_runs(op, pages, at, piece.length)? {
            copy(&mut bytes, to, pool, from, length)?;
        }
        write(piece, &bytes)?;
        at += piece.length;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(29).wrapping_add(seed))
            .collect()
    }

    #[test]
    fn entries_copy_across_planes_and_skip_items_without_a_source() {
        let op = CopyEntries {
            entry_bytes: 4,
            planes: 2,
            plane_stride: 12,
        };
        let src = noise(24, 1);
        let mut dst = vec![0xaa; 24];
        copy_entries(&op, &src, &mut dst, &[2, -1, 0], &[0, 1, 1], 3, 3).unwrap();
        let mut expected = vec![0xaa; 24];
        for base in [0, 12] {
            expected[base..base + 4].copy_from_slice(&src[base + 8..base + 12]);
            expected[base + 4..base + 8].copy_from_slice(&src[base..base + 4]);
        }
        assert_eq!(dst, expected);
        // Only live items copy; two items into one entry are refused.
        let mut untouched = vec![0xaa; 24];
        copy_entries(&op, &src, &mut untouched, &[2], &[0], 0, 1).unwrap();
        assert_eq!(untouched, vec![0xaa; 24]);
        assert!(copy_entries(&op, &src, &mut dst, &[0, 1], &[2, 2], 2, 2).is_err());
        assert!(
            CopyEntries {
                plane_stride: 3,
                ..op
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn serving_snapshots_are_the_bundles_pieces() {
        let shape = Shape::TP4;
        // The design's table (docs/bundle-qwen38.md, "Prefix caching").
        let snapshot = Snapshot::of(shape, 64, 32, 12, 32, false);
        assert_eq!(snapshot.ple_entry_bytes, 184_448);
        assert_eq!(
            u64::from(snapshot.linear_layers) * snapshot.conv_entry_bytes,
            552_960
        );
        assert_eq!(
            12 * (snapshot.ring_key_bytes + snapshot.ring_position_bytes),
            51_456
        );
        let drafted = Snapshot::of(shape, 128, 32, 13, 32, true);
        assert_eq!(drafted.retained_bytes, 8 * 4 * 2560 * 2);
        let pieces = drafted.pieces(Held {
            slot: 3,
            index: 7,
            ple_entry: 9,
        });
        assert_eq!(pieces.len(), 2 * 36 + 1 + 2 * 13 + 1);
        assert_eq!(
            pieces.iter().map(|p| p.length).sum::<u64>(),
            drafted.bytes()
        );
        assert_eq!(pieces[1].offset, (128 + 7) * shape.conv_entry_bytes);
        assert_eq!(pieces[72].offset, 9 * 184_448);
        let ring_layer = 32 * (drafted.ring_key_bytes + drafted.ring_position_bytes);
        assert_eq!(pieces[75].offset, ring_layer + 3 * drafted.ring_key_bytes);
        assert_eq!(
            pieces[76].offset,
            ring_layer + 32 * drafted.ring_key_bytes + 3 * drafted.ring_position_bytes
        );
        assert_eq!(pieces[99].offset, 3 * drafted.retained_bytes);
    }

    /// A small snapshot round-trips through scattered pages.
    #[test]
    fn snapshots_round_trip_through_scattered_pages() {
        let snapshot = Snapshot {
            linear_layers: 2,
            conv_entry_bytes: 16,
            delta_entry_bytes: 32,
            pool_entries: 3,
            ple_entry_bytes: 16,
            ple_entries: 2,
            ring_layers: 1,
            ring_slots: 2,
            ring_key_bytes: 16,
            ring_position_bytes: 16,
            retained_bytes: 0,
        };
        let op = PrefixCopy {
            snapshot,
            page_bytes: 64,
            pages: 3,
            direction: Direction::Capture,
        };
        assert_eq!(snapshot.bytes(), 144);
        let buffers = [Buffer::Conv, Buffer::Delta, Buffer::Ple, Buffer::Ring]
            .map(|buffer| noise(snapshot.buffer_bytes(buffer, 2) as usize, buffer as u8));
        let held = Held {
            slot: 1,
            index: 2,
            ple_entry: 0,
        };
        let pages = [3, 0, 5];
        let mut pool = vec![0xee; 6 * 64];
        let read = |piece: Piece| {
            Ok(buffers[piece.buffer as usize]
                [piece.offset as usize..(piece.offset + piece.length) as usize]
                .to_vec())
        };
        prefix_capture(&op, held, read, &pages, &mut pool).unwrap();
        let snapshot_bytes = snapshot
            .pieces(held)
            .into_iter()
            .flat_map(|piece| read(piece).unwrap())
            .collect::<Vec<_>>();
        for (index, chunk) in snapshot_bytes.chunks(64).enumerate() {
            let at = pages[index] as usize * 64;
            assert_eq!(&pool[at..at + chunk.len()], chunk);
        }
        // Unnamed pages and the last page's slack keep their bytes.
        for page in [1, 2, 4] {
            assert!(pool[page * 64..(page + 1) * 64].iter().all(|&b| b == 0xee));
        }
        assert!(pool[5 * 64 + 16..].iter().all(|&b| b == 0xee));
        let mut restored = Vec::new();
        prefix_restore(&op, held, &pages, &pool, |piece, bytes| {
            assert_eq!(bytes.len() as u64, piece.length);
            restored.extend_from_slice(bytes);
            Ok(())
        })
        .unwrap();
        assert_eq!(restored, snapshot_bytes);
        // A page twice is refused; a slot outside the ring copies nothing.
        assert!(prefix_capture(&op, held, read, &[3, 0, 3], &mut pool).is_err());
        let mut untouched = vec![0xee; 6 * 64];
        let outside = Held { slot: 2, ..held };
        prefix_capture(&op, outside, read, &pages, &mut untouched).unwrap();
        assert!(untouched.iter().all(|&b| b == 0xee));
        assert!(PrefixCopy { pages: 2, ..op }.validate().is_err());
    }
}
