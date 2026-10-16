//! The layout of one prefix-cache snapshot in serving-pool pages.
//!
//! An anchor shares the K, V and QSA block-key pages of its prefix (the MTP
//! drafter's layer included), so a snapshot holds only the per-slot state a
//! request needs beyond them to continue after its committed boundary: one
//! committed entry of the linear convolution and delta-rule pools in every
//! linear layer, the PLE entry, the slot's live QSA raw-key ring in every
//! full layer of the ring (the drafter's too) and, with the drafter, the
//! slot's retained target X rows. The contract declares them
//! (`model_qwen38_contract::state`), and the shared descriptor table
//! (`bundle_runtime::snapshot::Table`) lays them end to end across the
//! snapshot's pages, so a piece may continue on the next page. One kernel
//! per rank copies all of them (`prefix_copy` in `cuda/qwen38_bundle.cuh`,
//! whose addressing `kernel_pieces` mirrors). The last page's slack after
//! the pieces holds the header the host writes (`HEADER`).

use bundle_runtime::prefix::{HeaderLayout, Word};
use bundle_runtime::snapshot::Table;
use bundle_runtime::state::{Extent, Holder};
use kernel_core::contract::Refusal;
use runtime_bundle_abi::DeviceBufferSpanV1;

use model_qwen38_contract::own_ops::copies::{Direction, PrefixCopy, Snapshot};

use crate::contract::*;
use crate::scratch::{RETAINED_ROWS, Speculation};

/// What the host writes after the pieces, on rank 0 only: a layout tag (so
/// pages no capture wrote, or wrote under another layout, are refused), the
/// committed token count, with the drafter its frontier (drafter rows that
/// ran), which the retained rows continue from, and the request that
/// captured it.
pub const HEADER: HeaderLayout = HeaderLayout {
    magic: *b"QW38PFX2",
    words: &[Word::Tokens, Word::Frontier, Word::Owner],
};
/// Page-table entries the `prefix_pages` buffer holds at least.
pub const MAX_SNAPSHOT_PAGES: u64 = 64;

/// Page-table entries a pool of `page_bytes` pages needs: the largest
/// snapshot (the drafter's ring layer and retained rows) in those pages, and
/// at least `MAX_SNAPSHOT_PAGES`. The FP8 layout without the drafter has
/// smaller pages than the BF16 ones, so its snapshots take more of them.
pub fn page_table_entries(page_bytes: u64, shape: Shape) -> u64 {
    let drafted = &shape.kv_formats()[1];
    debug_assert!(model_qwen38_contract::state::has_mtp_layer(drafted));
    model_qwen38_contract::state::request_state(shape, drafted, true)
        .snapshot_bytes()
        .div_ceil(page_bytes)
        .max(MAX_SNAPSHOT_PAGES)
}

/// The declared buffers, by index: the convolution and delta-rule pools,
/// the PLE entries, the QSA ring and, with the drafter, the retained rows.
pub(super) const CONV: usize = 0;
pub(super) const DELTA: usize = 1;
pub(super) const PLE: usize = 2;
const RING: usize = 3;
const RETAINED: usize = 4;

/// Whose state one copy-kernel launch moves: pool entry `index` of every
/// linear plane of `pool_entries` entries, and entry `slot` of `slots` PLE
/// and retained entries and of a ring of `ring_layers` layers and
/// `ring_slots` slots. A slot's committed state, or (`anchor`) an anchor
/// item's spares.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Held {
    pub index: u64,
    pub ple_index: u64,
    pub ple_entries: u64,
    pub pool_entries: u64,
    pub slot: u32,
    pub slots: u32,
    pub ring_slots: u32,
    pub ring_layers: u64,
    pub retained: bool,
    pub anchor: bool,
}

impl Held {
    /// What `holder`'s snapshot copies in an instance of `extent` whose
    /// pools and ring follow `speculation` and hold `ring_layers` layers:
    /// the slot's committed entry, or the anchor item's entry after the
    /// step checkpoints and its spares.
    pub const fn of(
        holder: Holder,
        extent: Extent,
        speculation: Speculation,
        ring_layers: u64,
        linear_entry: u64,
        ple_entry: u64,
    ) -> Self {
        let (index, slot, slots, ring_slots, anchor) = match holder {
            Holder::Slot(slot) => (linear_entry, slot, extent.served, extent.bucket, false),
            Holder::Anchor(anchor) => (
                speculation.anchor_entry(extent.served, anchor),
                anchor,
                ANCHOR_ITEMS,
                ANCHOR_ITEMS,
                true,
            ),
        };
        Self {
            index,
            ple_index: if anchor { slot as u64 } else { ple_entry },
            ple_entries: if anchor {
                ANCHOR_ITEMS as u64
            } else {
                extent.served as u64
                    + if extent.verify_rows > 0 {
                        extent.bucket as u64 * MAX_VERIFY_ROWS as u64
                    } else {
                        0
                    }
            },
            pool_entries: speculation.pool_entries(extent.served),
            slot,
            slots,
            ring_slots,
            ring_layers,
            retained: speculation.drafter,
            anchor,
        }
    }
}

/// What the copy kernel moves for `held`, in snapshot order, as (declared
/// buffer, offset, length): the convolution entries by linear layer, the
/// delta-rule entries, the PLE entry, each ring layer's keys and positions,
/// then the retained rows.
fn kernel_pieces(shape: Shape, held: Held) -> impl Iterator<Item = (usize, u64, u64)> {
    let Held {
        index,
        pool_entries,
        slot,
        ple_index,
        ring_slots,
        ring_layers,
        retained,
        ..
    } = held;
    let slot = u64::from(slot);
    let ring = Ring {
        shape,
        layers: ring_layers,
        slots: u64::from(ring_slots),
    };
    let pool = move |buffer, entry_bytes: u64| {
        (0..shape.linear_layers()).map(move |layer| {
            (
                buffer,
                (layer * pool_entries + index) * entry_bytes,
                entry_bytes,
            )
        })
    };
    pool(CONV, shape.conv_entry_bytes)
        .chain(pool(DELTA, shape.delta_entry_bytes))
        .chain([(
            PLE,
            ple_index * shape.ple_entry_bytes(),
            shape.ple_entry_bytes(),
        )])
        .chain((0..ring_layers).flat_map(move |layer| {
            [
                (
                    RING,
                    layer * ring.layer_bytes() + slot * shape.ring_slot_key_bytes(),
                    shape.ring_slot_key_bytes(),
                ),
                (
                    RING,
                    layer * ring.layer_bytes()
                        + ring.key_bytes()
                        + slot * shape.ring_slot_position_bytes(),
                    shape.ring_slot_position_bytes(),
                ),
            ]
        }))
        .chain(retained.then_some((
            RETAINED,
            slot * (u64::from(RETAINED_ROWS) * crate::scratch::wide_row_bytes(shape)),
            (u64::from(RETAINED_ROWS) * crate::scratch::wide_row_bytes(shape)),
        )))
}

/// Where one rank's snapshot copy reads and writes.
pub struct PrefixCopyViews {
    pub conv_pool: DeviceBufferSpanV1,
    pub delta_pool: DeviceBufferSpanV1,
    /// The PLE entries, or the anchor items' spares.
    pub ple_pool: DeviceBufferSpanV1,
    /// The live half of the QSA raw-key ring, or the anchor items' ring.
    pub ring: DeviceBufferSpanV1,
    /// Every slot's retained drafter rows, with the drafter.
    pub retained: Option<DeviceBufferSpanV1>,
    /// The serving KV pool, addressed by page ID.
    pub pool: DeviceBufferSpanV1,
    /// The snapshot's page IDs as little-endian `u64`.
    pub pages: DeviceBufferSpanV1,
}

/// The copy of `table`'s snapshot between where `held` holds it and the
/// pages, in `direction` (`OwnOps::prefix_copy`).
pub fn prefix_copy_op(shape: Shape, table: &Table, held: Held, direction: Direction) -> PrefixCopy {
    PrefixCopy {
        snapshot: Snapshot::of(
            shape,
            held.pool_entries as u32,
            held.ple_entries as u32,
            held.ring_layers as u32,
            held.ring_slots,
            held.retained,
        ),
        page_bytes: table.page_bytes,
        pages: table.pages(),
        direction,
    }
}

/// Refuses a copy of `table` between where `held` holds it and `views`
/// unless the views are the snapshot's geometry and the table's pieces are
/// the ones the kernel addresses itself.
pub fn check_copy(
    shape: Shape,
    table: &Table,
    held: Held,
    views: &PrefixCopyViews,
) -> Result<(), Refusal> {
    let pool_entries = held.pool_entries;
    let ring = Ring {
        shape,
        layers: held.ring_layers,
        slots: u64::from(held.ring_slots),
    };
    let slots = u64::from(held.slots);
    let page_bytes = table.page_bytes;
    if held.index >= pool_entries
        || held.slot >= held.slots
        || held.slot >= held.ring_slots
        || table.pages() > page_table_entries(page_bytes, shape)
        || views.pages.length < table.pages() * 8
        || !views.pool.length.is_multiple_of(page_bytes)
        || views.conv_pool.length != shape.linear_layers() * pool_entries * shape.conv_entry_bytes
        || views.delta_pool.length != shape.linear_layers() * pool_entries * shape.delta_entry_bytes
        || held.ple_index >= held.ple_entries
        || views.ple_pool.length != held.ple_entries * shape.ple_entry_bytes()
        || views.ring.length != ring.bytes()
        || views.retained.map(|span| span.length)
            != held.retained.then_some(
                slots * (u64::from(RETAINED_ROWS) * crate::scratch::wide_row_bytes(shape)),
            )
    {
        return Err(Refusal("Qwen3.8 prefix snapshot geometry"));
    }
    if !table
        .descriptors
        .iter()
        .map(|piece| (piece.buffer, piece.offset, piece.length))
        .eq(kernel_pieces(shape, held))
        || table
            .descriptors
            .iter()
            .any(|piece| piece.spare != (held.anchor && piece.buffer > DELTA))
    {
        return Err(Refusal(
            "Qwen3.8 prefix snapshot state does not match its pieces",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use bundle_runtime::kv::KvFormat;
    use bundle_runtime::prefix::{Header, Stamp};
    use bundle_runtime::state::RequestState;
    use model_qwen38_contract::state::request_state;

    use super::*;

    /// An instance serving `served` sequences in a ring of `bucket` slots,
    /// with up to `verify_rows` verify rows.
    fn extent(served: u32, bucket: u32, verify_rows: u32) -> Extent {
        Extent {
            served,
            bucket,
            verify_rows,
        }
    }

    fn table(state: &RequestState, extent: Extent, holder: Holder) -> Table {
        Table::new(state, extent, holder).unwrap()
    }

    /// The format of a package with or without the drafter's layer, with
    /// BF16 or FP8 main-layer K and V.
    fn format(drafter: bool, fp8: bool) -> &'static KvFormat {
        &KV_FORMATS[usize::from(drafter) + 2 * usize::from(fp8)]
    }

    #[test]
    fn a_snapshot_is_linear_ple_ring_and_drafter_state() {
        // The design's table (docs/bundle-qwen38.md, "Prefix caching").
        assert_eq!(LINEAR_LAYERS * Shape::TP4.conv_entry_bytes, 552_960);
        assert_eq!(LINEAR_LAYERS * Shape::TP4.delta_entry_bytes, 28_311_552);
        assert_eq!(PLE_ENTRY_BYTES, 184_448);
        assert_eq!(
            FULL_LAYERS * (RING_SLOT_KEY_BYTES + RING_SLOT_POSITION_BYTES),
            51_456
        );
        let target = request_state(Shape::TP4, format(false, false), false);
        let target = table(&target, extent(32, 32, 0), Holder::Slot(5));
        assert_eq!(target.header_at(), 29_100_416);
        assert_eq!(target.pages(), 35);
        // With the drafter: its ring layer and eight retained X rows, in
        // 33 of the larger pages.
        let drafted = request_state(Shape::TP4, format(true, false), true);
        let drafted = table(&drafted, extent(32, 32, 4), Holder::Slot(5));
        assert_eq!(drafted.header_at(), 29_100_416 + 4_288 + 163_840);
        assert_eq!(drafted.pages(), 33);
        for table in [target, drafted] {
            // Every piece and page boundary is a whole 16-byte vector.
            assert!(
                table
                    .descriptors
                    .iter()
                    .all(|piece| piece.length % 16 == 0 && piece.offset % 16 == 0)
            );
            assert_eq!(table.page_bytes % 16, 0);
            assert!(table.pages() <= page_table_entries(table.page_bytes, Shape::TP4));
        }
    }

    /// Snapshots follow the pool's pages: the FP8 layouts' smaller pages
    /// take more of them, within each layout's page table.
    #[test]
    fn snapshot_pages_follow_each_kv_layout() {
        let pages = |page_bytes, drafter: bool| {
            let state = request_state(Shape::TP4, format(drafter, false), drafter);
            state.snapshot_bytes().div_ceil(page_bytes)
        };
        let layouts = [
            (format(false, false), false, 35, 64),
            (format(true, false), true, 33, 64),
            (format(false, true), false, 66, 67),
            (format(true, true), true, 58, 64),
        ];
        for (format, drafter, expected, entries) in layouts {
            let page_bytes = format.page_bytes;
            let state = request_state(Shape::TP4, format, drafter);
            assert_eq!(state.snapshot_pages(), expected, "{page_bytes}");
            assert_eq!(
                page_table_entries(page_bytes, Shape::TP4),
                entries,
                "{page_bytes}"
            );
            // Either snapshot fits the table of either pool size.
            assert!(pages(page_bytes, true) <= entries && pages(page_bytes, false) <= entries);
        }
        assert!(KV_FORMATS.iter().all(|kv| kv.page_bytes % 16 == 0));
    }

    /// The contract's declared table is, piece for piece, what the copy
    /// kernel moves, for a slot's committed entry and for an anchor item's
    /// spares, and rank 0's header fits in the last page's slack. The
    /// expected geometry comes from the package layout the bundle accepts,
    /// each buffer's size from the allocation (`scratch::sizes`) or the
    /// declared state, and the checkpoint entries from the step's plan and
    /// the commit's index.
    #[test]
    fn the_declared_state_is_what_a_snapshot_copies() {
        use crate::scratch::Speculation;

        for shape in Shape::ALL {
            for format in &KV_FORMATS {
                let kv_layers = crate::instance::kv_layers(format.layout_id).unwrap();
                for drafter in [false, kv_layers == FULL_LAYERS_MTP] {
                    let state = request_state(shape, format, drafter);
                    assert_eq!(state.header_bytes, HEADER.bytes());
                    assert_eq!(HEADER.check(&state), Ok(()));
                    assert_eq!(state.page_bytes(), format.page_bytes);
                    // Each full layer's K, V and block keys where the
                    // attention views them.
                    let planes = state.page_planes();
                    let layers = (0..LAYERS)
                        .filter(|&layer| is_full(layer))
                        .chain((kv_layers == FULL_LAYERS_MTP).then_some(MTP_LAYER));
                    for (ordinal, layer) in layers.enumerate() {
                        let (offset, sizes, _) = page_layer(format, layer);
                        let declared = &planes[3 * ordinal..3 * ordinal + 3];
                        assert_eq!(declared[0].offset, offset, "{layer}");
                        assert_eq!(
                            declared.iter().map(|plane| plane.bytes).collect::<Vec<_>>(),
                            sizes
                        );
                    }
                    assert_eq!(planes.len() as u64, 3 * kv_layers);
                    for (served, bucket) in [(1, 1), (32, 32), (100, 128), (256, 256)] {
                        for verify_rows in [0, 2, 4, MAX_VERIFY_ROWS] {
                            let speculation = Speculation {
                                verify_rows,
                                drafter,
                            };
                            check_extent(
                                &state,
                                shape,
                                format,
                                kv_layers,
                                speculation,
                                extent(served, bucket, verify_rows),
                            );
                        }
                    }
                }
            }
        }

        fn check_extent(
            state: &RequestState,
            shape: Shape,
            format: &KvFormat,
            kv_layers: u64,
            speculation: Speculation,
            extent: Extent,
        ) {
            let Extent {
                served,
                bucket,
                verify_rows,
            } = extent;
            assert_eq!(state.validate_extent(extent), Ok(()));
            let mut holders = [0, served / 2, served - 1].map(Holder::Slot).to_vec();
            holders.extend((0..ANCHOR_ITEMS).map(Holder::Anchor));
            for holder in holders {
                let table = table(state, extent, holder);
                assert_eq!(table.pages(), state.snapshot_pages());
                // The header does not cross into a page after the last.
                let at = table.header_at() % table.page_bytes;
                assert!(
                    at + HEADER.bytes() <= table.page_bytes,
                    "{shape:?} {holder:?}"
                );
                let held = Held::of(
                    holder,
                    extent,
                    speculation,
                    kv_layers,
                    match holder {
                        Holder::Slot(slot) => slot as u64,
                        Holder::Anchor(_) => 0,
                    },
                    match holder {
                        Holder::Slot(slot) => slot as u64,
                        Holder::Anchor(_) => 0,
                    },
                );
                let declared = table
                    .descriptors
                    .iter()
                    .map(|piece| {
                        // Anchor spares beside the pools.
                        assert_eq!(
                            piece.spare,
                            matches!(holder, Holder::Anchor(_)) && piece.buffer > 1
                        );
                        (piece.buffer, piece.offset, piece.length)
                    })
                    .collect::<Vec<_>>();
                let copied = kernel_pieces(shape, held).collect::<Vec<_>>();
                assert_eq!(declared, copied, "{shape:?} {holder:?}");
            }
            // The buffers the instance allocates: each declared buffer, its
            // separate checkpoints and spares.
            let served_items = crate::scratch::Served {
                slots: served,
                items: bucket,
                rows: bucket,
            };
            let sizes = crate::scratch::sizes(
                &crate::instance::target(shape, false).comm_bytes(served_items),
                0,
                speculation,
                format.page_bytes,
                shape,
                served_items,
            )
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();
            let [conv, delta, ple, ring] = [0, 1, 2, 3].map(|index| &state.buffers[index]);
            assert_eq!(conv.bytes(extent), sizes["conv_pool"]);
            assert_eq!(delta.bytes(extent), sizes["delta_pool"]);
            assert_eq!(ple.bytes(extent), sizes["ple_pool"]);
            assert_eq!(ple.spare_bytes(), sizes["anchor_ple"]);
            assert_eq!(ple.checkpoint_bytes(extent), 0);
            // The live and saved rings the host allocates, padded per slot;
            // the anchor ring always has the drafter's layer.
            let live = Ring {
                shape: model_qwen38_contract::contract::Shape::flash(
                    model_qwen38_contract::contract::Ranks::TP4,
                ),
                layers: kv_layers,
                slots: u64::from(bucket),
            };
            assert_eq!(ring.bytes(extent), 2 * live.bytes());
            assert!(ring.bytes(extent) <= live.state_bytes());
            assert!(ring.spare_bytes() <= sizes["anchor_ring"]);
            assert_eq!(
                ring.spare_bytes() == sizes["anchor_ring"],
                kv_layers == FULL_LAYERS_MTP
            );
            if speculation.drafter {
                let retained = &state.buffers[4];
                assert_eq!(retained.bytes(extent), sizes["retained"]);
                assert_eq!(retained.spare_bytes(), sizes["anchor_retained"]);
            } else {
                assert_eq!(state.buffers.len(), 4);
            }
            // The checkpoints a verify step's plan names and its commit
            // copies from.
            if verify_rows > 1 {
                use bundle_runtime::ledger::{Ledger, Placement, Rules};
                let mut ledger =
                    Ledger::new(served, Placement::Host, Rules::of(state), CONTEXT_TOKENS);
                for slot in 0..served {
                    ledger.restore(u64::from(slot) + 1, 10, Some(slot)).unwrap();
                }
                // Only `DRAFT_ITEMS` items draft in one step.
                let drafting = served.min(DRAFT_ITEMS);
                let phase = crate::inputs::PhaseInputs {
                    slot_ids: (0..served as i32).collect(),
                    request_ids: (1..=u64::from(served)).collect(),
                    prefix_len: vec![10; served as usize],
                    q_len: (0..served)
                        .map(|item| if item < drafting { verify_rows } else { 1 } as i32)
                        .collect(),
                    verify_rows: (0..served).map(|item| item < drafting).collect(),
                    verify: true,
                    ..Default::default()
                };
                let empty = crate::inputs::PhaseInputs::default();
                ledger
                    .begin(&crate::state::items([&empty, &phase]), Vec::new())
                    .unwrap();
                let planned = crate::state::entries(
                    ledger.step().unwrap(),
                    0,
                    &phase,
                    &crate::state::EntryMap::new(served, served.min(DRAFT_ITEMS) * verify_rows),
                    verify_rows,
                    &crate::state::EntryMap::new(served, bucket * MAX_VERIFY_ROWS),
                );
                for (item, entries) in planned.iter().enumerate().take(drafting as usize) {
                    for row in 0..verify_rows {
                        assert_eq!(
                            conv.checkpoint(extent, item as u32, row),
                            Some(entries.checkpoints[row as usize] as u64)
                        );
                    }
                }
                for item in 0..bucket {
                    for row in 0..MAX_VERIFY_ROWS {
                        assert_eq!(
                            ple.checkpoint(extent, item, row),
                            Some(u64::from(served + item * MAX_VERIFY_ROWS + row))
                        );
                    }
                }
            } else {
                assert_eq!(conv.checkpoint(extent, 0, 0), None);
                assert_eq!(ple.checkpoint(extent, 0, 0), None);
            }
        }
    }

    #[test]
    fn the_header_keeps_its_legacy_bytes() {
        let stamp = Stamp {
            ranks: 4,
            digest: 0,
        };
        assert_eq!(HEADER.bytes(), 32);
        for (frontier, owner) in [(None, 3), (Some(66), u64::MAX - 1)] {
            let header = Header {
                tokens: 70,
                frontier,
                owner: Some(owner),
                cursor: None,
            };
            let bytes = header.encode(&HEADER, stamp).unwrap();
            let mut legacy = b"QW38PFX2".to_vec();
            legacy.extend(70_u64.to_le_bytes());
            legacy.extend(frontier.unwrap_or(u64::MAX).to_le_bytes());
            legacy.extend(owner.to_le_bytes());
            assert_eq!(bytes, legacy);
            assert_eq!(Header::decode(&bytes, &HEADER, stamp, 70), Ok(header));
        }
        assert!(Header::decode(&[0; 32], &HEADER, stamp, 70).is_err());
    }
}
