//! The layout of one prefix-cache snapshot in serving-pool pages.
//!
//! A snapshot holds what a sequence needs beyond its global-attention pages
//! to continue after a committed step boundary: its sliding-window ring
//! slot in every layer, the loaded drafter's retained committed rows and,
//! with DFlash, the projected context and a header. The contract declares
//! them (`model_mimo26_contract::state`), and the shared descriptor table
//! (`bundle_runtime::snapshot::Table`) lays them end to end across the
//! snapshot's pages, so a piece may continue on the next page. One kernel
//! per rank copies the ring and drafter-history pieces
//! (`cuda/mimo_ring.cuh`, whose addressing the table's first pieces follow);
//! device copies move the rest.

#[cfg(test)]
use model_mimo26_contract::geometry::V26_FLASH;

use std::ops::Range;

use bundle_runtime::snapshot::Table;
#[cfg(test)]
use kernel_core::contract::Argument;
use kernel_core::contract::Refusal;
use model_mimo26_contract::own_ops::ring::{Direction, RingLayout, RingPrefixCopy};
use runtime_bundle_abi::DeviceBufferSpanV1;

#[cfg(test)]
use crate::contract::RING_TOKENS;
#[cfg(test)]
use crate::contract::SWA_LAYERS;
#[cfg(test)]
use crate::dispatch::BufferBindings;
#[cfg(test)]
use crate::pipeline::PreparedLaunch;
use crate::ring::RingGeometry;
#[cfg(test)]
use crate::topology::Tp;
/// Bytes of one retained MTP target hidden row.
#[cfg(test)]
pub const MTP_HIDDEN_BYTES: u64 = V26_FLASH.hidden() as u64 * 2;
/// Retained MTP hidden rows per slot, including an open step's rows.
#[cfg(test)]
pub const MTP_SLOT_ROWS: u64 = crate::retain::capacity(RING_TOKENS);

/// Drafter history included in an instance's snapshots.
pub use model_mimo26_contract::state::Drafter;

/// The declared buffers the copy kernel moves: the target ring, then the
/// drafter's history.
pub const KERNEL_BUFFERS: Range<usize> = 0..2;

/// Where the copy kernel moves `slot`'s ring entries, as (offset in the
/// working planes, length) in snapshot order: every layer's K slot entry,
/// then its V slot entry (`prefix_copy` in `cuda/mimo_ring.cuh`).
fn kernel_ring(slot: u32, ring: RingGeometry) -> impl Iterator<Item = (u64, u64)> {
    let slot = u64::from(slot);
    let (k_slot, v_slot) = (
        ring.tp.geometry.window() as u64 * ring.k_entry,
        ring.tp.geometry.window() as u64 * ring.v_entry,
    );
    (0..ring.tp.geometry.sliding_layers() as u64).flat_map(move |layer| {
        [
            (layer * ring.layer + slot * k_slot, k_slot),
            (layer * ring.layer + ring.k_layer + slot * v_slot, v_slot),
        ]
    })
}

/// Threads per block of the snapshot copy; each moves one 16-byte vector.
#[cfg(test)]
const PREFIX_COPY_THREADS: u32 = 256;

/// Where one rank's snapshot copy reads and writes.
pub struct PrefixCopyViews {
    /// The working half of the target ring state.
    pub ring_state: DeviceBufferSpanV1,
    /// The slot's drafter piece, when the snapshot has one.
    pub drafter: Option<DeviceBufferSpanV1>,
    /// The serving KV pool, addressed by page ID.
    pub pool: DeviceBufferSpanV1,
    /// The snapshot's page IDs as little-endian `u64`.
    pub pages: DeviceBufferSpanV1,
}

/// The copy of the pieces of `slot`'s `table` in [`KERNEL_BUFFERS`] into
/// its snapshot pages or back from them (`direction`). The kernel addresses
/// the pieces itself, so they must be the table's first pieces: the slot's
/// ring entries where the kernel reads them, then at most one drafter
/// piece. The op checks the views at launch.
pub fn copy_op(
    table: &Table,
    slot: u32,
    ring: RingGeometry,
    direction: Direction,
) -> Result<RingPrefixCopy, Refusal> {
    let covered = table
        .descriptors
        .iter()
        .take_while(|descriptor| KERNEL_BUFFERS.contains(&descriptor.buffer))
        .collect::<Vec<_>>();
    let (target, drafter) = covered.split_at(
        covered
            .iter()
            .take_while(|descriptor| descriptor.buffer == KERNEL_BUFFERS.start)
            .count(),
    );
    let drafter_bytes = drafter
        .iter()
        .map(|descriptor| descriptor.length)
        .sum::<u64>();
    let page_bytes = table.page_bytes;
    if slot >= ring.slots
        || page_bytes == 0
        || !page_bytes.is_multiple_of(16)
        || !drafter_bytes.is_multiple_of(16)
    {
        return Err(Refusal("MiMo prefix snapshot geometry"));
    }
    if table.descriptors[covered.len()..]
        .iter()
        .any(|descriptor| KERNEL_BUFFERS.contains(&descriptor.buffer))
        || drafter.len() > 1
        || !target
            .iter()
            .map(|descriptor| (descriptor.offset, descriptor.length))
            .eq(kernel_ring(slot, ring))
    {
        return Err(Refusal(
            "MiMo prefix snapshot state does not match its pieces",
        ));
    }
    Ok(RingPrefixCopy {
        layout: RingLayout::of(ring.tp, ring.slots),
        drafter_bytes,
        page_bytes,
        pages: table.pages(),
        direction,
    })
}

/// One launch that copies the pieces of `slot`'s `table` in
/// [`KERNEL_BUFFERS`] into its `pages` snapshot pages (`to_pages`) or back
/// from them. The kernel addresses the pieces itself, so they must be the
/// table's first pieces: the slot's ring entries where the kernel reads
/// them, then at most one drafter piece, which `views.drafter` binds.
#[cfg(test)]
pub fn prefix_copy(
    table: &Table,
    slot: u32,
    pages: u64,
    to_pages: bool,
    views: PrefixCopyViews,
    ring: RingGeometry,
) -> Result<PreparedLaunch, Refusal> {
    let tp = ring.tp;
    let covered = table
        .descriptors
        .iter()
        .take_while(|descriptor| KERNEL_BUFFERS.contains(&descriptor.buffer))
        .collect::<Vec<_>>();
    let (target, drafter) = covered.split_at(
        covered
            .iter()
            .take_while(|descriptor| descriptor.buffer == KERNEL_BUFFERS.start)
            .count(),
    );
    let drafter_bytes = drafter
        .iter()
        .map(|descriptor| descriptor.length)
        .sum::<u64>();
    let page_bytes = table.page_bytes;
    if slot >= ring.slots
        || page_bytes == 0
        || !page_bytes.is_multiple_of(16)
        || !drafter_bytes.is_multiple_of(16)
        || table.pages() != pages
        || views.pages.length < pages * 8
        || views.pool.length < page_bytes
    {
        return Err(Refusal("MiMo prefix snapshot geometry"));
    }
    if table.descriptors[covered.len()..]
        .iter()
        .any(|descriptor| KERNEL_BUFFERS.contains(&descriptor.buffer))
        || drafter.len() > 1
        || !target
            .iter()
            .map(|descriptor| (descriptor.offset, descriptor.length))
            .eq(kernel_ring(slot, ring))
        || views.ring_state.length < ring.tp.geometry.sliding_layers() as u64 * ring.layer
        || views.drafter.map_or(0, |span| span.length) != drafter_bytes
    {
        return Err(Refusal(
            "MiMo prefix snapshot state does not match its pieces",
        ));
    }
    let vectors = covered
        .iter()
        .map(|descriptor| descriptor.length)
        .sum::<u64>()
        / 16;
    let mut bindings = BufferBindings::default();
    let arguments = vec![
        Argument::DeviceBuffer(bindings.bind(views.ring_state)),
        views.drafter.map_or(Argument::NullDevicePointer, |span| {
            Argument::DeviceBuffer(bindings.bind(span))
        }),
        Argument::U64(drafter_bytes),
        Argument::DeviceBuffer(bindings.bind(views.pool)),
        Argument::DeviceBuffer(bindings.bind(views.pages)),
        Argument::U32(pages as u32),
        Argument::U64(page_bytes),
        Argument::U32(slot),
        Argument::U32(u32::from(to_pages)),
    ];
    Ok(PreparedLaunch {
        symbol: crate::sm120::ring_entries::prefix_copy_symbol(tp, ring.slots),
        // A snapshot copy moves one slot's state and has no token rows.
        rows: 1,
        plan: kernel_core::contract::LaunchPlan {
            grid: [
                vectors.div_ceil(u64::from(PREFIX_COPY_THREADS)) as u32,
                1,
                1,
            ],
            block: [PREFIX_COPY_THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        },
        arguments: kernel_core::contract::ArgumentList(arguments),
        bindings,
    })
}

#[cfg(test)]
mod tests {
    use bundle_runtime::state::{Extent, Holder};
    use model_mimo26_contract::state::request_state;

    use super::*;
    use crate::contract::{
        GA_PAGE_BYTES_BF16, GA_PAGE_BYTES_FP8, KV_FORMATS, MAX_VERIFY_ROWS, RING_SLOT_BYTES,
    };
    use crate::prefix_projected::HEADER;

    const K_SLOT_BYTES: u64 = RING_TOKENS * 768;
    const V_SLOT_BYTES: u64 = RING_TOKENS * 512;
    const K_LAYER_BYTES: u64 = 32 * K_SLOT_BYTES;
    const LAYER_BYTES: u64 = K_LAYER_BYTES + 32 * V_SLOT_BYTES;

    /// The ring of 32 served sequences, which the journals record.
    fn ring(tp: Tp) -> RingGeometry {
        RingGeometry::new(tp, 32)
    }

    fn extent(served: u32, bucket: u32) -> Extent {
        Extent {
            served,
            bucket,
            verify_rows: MAX_VERIFY_ROWS,
        }
    }

    /// `slot`'s table at `tp` with BF16 pages (FP8 when `fp8`), serving
    /// `served` sequences in a ring of `bucket` slots.
    fn table(tp: Tp, fp8: bool, drafter: Drafter, served: u32, bucket: u32, slot: u32) -> Table {
        let state = request_state(tp, &KV_FORMATS[usize::from(fp8)], drafter);
        Table::new(&state, extent(served, bucket), Holder::Slot(slot)).unwrap()
    }

    /// The contract's declared table is, piece for piece, what a snapshot
    /// copies: the ring entries where the copy kernel moves them, the
    /// drafter's slot of history, the DFlash projected planes, then the
    /// header. Each buffer's size is the one the instance holds.
    #[test]
    fn the_declared_state_is_what_a_snapshot_copies() {
        use crate::contract::DRAFT_ITEMS;
        use crate::dflash::state::HISTORY_CAPACITY;

        for tp in Tp::SUPPORTED {
            for drafter in [Drafter::None, Drafter::Mtp, Drafter::Dflash] {
                for format in &KV_FORMATS {
                    let state = request_state(tp, format, drafter);
                    assert_eq!(state.validate(), Ok(()));
                    assert_eq!(
                        state.page_bytes(),
                        crate::topology::ga_page_bytes(tp, format.fp8())
                    );
                    // Each global layer's K and V where the attention views
                    // them.
                    let pool = DeviceBufferSpanV1::new(
                        runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(1, 1),
                        4096,
                        8 * state.page_bytes(),
                    );
                    let planes = state.page_planes();
                    for (ordinal, &layer) in crate::program::GLOBAL_LAYERS.iter().enumerate() {
                        let views =
                            crate::ga_pool::layer_pages(pool, layer, format.fp8(), tp).unwrap();
                        let [k, v] = [&planes[2 * ordinal], &planes[2 * ordinal + 1]];
                        assert_eq!((k.layer, k.stream, v.stream), (ordinal as u64, "k", "v"));
                        assert_eq!(views.k.offset - pool.offset, k.offset);
                        assert_eq!(views.v.offset - pool.offset, v.offset);
                    }
                }
                let state = request_state(tp, &KV_FORMATS[0], drafter);
                for (served, bucket) in [(1, 1), (32, 32), (100, 128), (256, 256)] {
                    let extent = extent(served, bucket);
                    let ring = RingGeometry::new(tp, bucket);
                    // An MTP instance serves at most the MTP layer's slots.
                    if drafter == Drafter::Mtp && served > DRAFT_ITEMS {
                        assert!(state.validate_extent(extent).is_err());
                        assert!(Table::new(&state, extent, Holder::Slot(served - 1)).is_err());
                        continue;
                    }
                    assert_eq!(state.validate_extent(extent), Ok(()));
                    for slot in [0, served / 2, served - 1] {
                        let table = Table::new(&state, extent, Holder::Slot(slot)).unwrap();
                        let mut copied = kernel_ring(slot, ring)
                            .map(|(offset, length)| (0, offset, length))
                            .collect::<Vec<_>>();
                        let slot = u64::from(slot);
                        match drafter {
                            Drafter::None => {}
                            Drafter::Mtp => {
                                let bytes = MTP_SLOT_ROWS * MTP_HIDDEN_BYTES;
                                copied.push((1, slot * bytes, bytes));
                            }
                            Drafter::Dflash => {
                                let bytes = HISTORY_CAPACITY * tp.dflash_fc_k() * 2;
                                copied.push((1, slot * bytes, bytes));
                                // Ten planes, each [slot][position][column].
                                let plane = crate::dflash::program::cache_bytes(tp, 1) / 10;
                                copied.extend((0..10).map(|index| {
                                    let start = (index * u64::from(bucket) + slot) * plane;
                                    (2, start, plane)
                                }));
                            }
                        }
                        let mut at = 0;
                        let declared = table
                            .descriptors
                            .iter()
                            .map(|descriptor| {
                                assert!(!descriptor.spare);
                                assert_eq!(descriptor.at, at);
                                at += descriptor.length;
                                (descriptor.buffer, descriptor.offset, descriptor.length)
                            })
                            .collect::<Vec<_>>();
                        assert_eq!(declared, copied, "{tp:?} {drafter:?} {slot}/{bucket}");
                    }
                    // The working half the copy reads, then the shadow.
                    assert_eq!(state.buffers[0].bytes(extent), 2 * ring.working);
                    let held = match drafter {
                        Drafter::None => vec![],
                        // The hidden rows, then the MTP layer's ring of
                        // `DRAFT_ITEMS` slots (`mtp/runtime.rs` `ring_bytes`).
                        Drafter::Mtp => vec![
                            u64::from(DRAFT_ITEMS) * MTP_SLOT_ROWS * MTP_HIDDEN_BYTES,
                            crate::topology::ring_state_bytes(tp, DRAFT_ITEMS) / 2 / SWA_LAYERS,
                        ],
                        Drafter::Dflash => vec![
                            u64::from(served) * HISTORY_CAPACITY * tp.dflash_fc_k() * 2,
                            crate::dflash::program::cache_bytes(tp, bucket),
                        ],
                    };
                    let declared = state.buffers[1..]
                        .iter()
                        .map(|buffer| buffer.bytes(extent))
                        .collect::<Vec<_>>();
                    assert_eq!(declared, held);
                    // The MTP ring's cancel copy is the whole ring, in a
                    // buffer of its own.
                    if let Some(mtp) = state.buffers.get(2).filter(|_| drafter == Drafter::Mtp) {
                        assert_eq!(mtp.shadow_bytes(extent), held[1]);
                        assert_eq!(
                            mtp.region(extent, bundle_runtime::state::Region::All),
                            Some((0, held[1]))
                        );
                    }
                }
                let header = if drafter == Drafter::Dflash {
                    assert_eq!(HEADER.check(&state), Ok(()));
                    64
                } else {
                    0
                };
                assert_eq!(state.header_bytes, header);
            }
        }
    }

    #[test]
    fn other_topologies_scale_the_ring_and_dflash_pieces() {
        {
            let tp = Tp::TP2;
            let slot_bytes = table(tp, false, Drafter::None, 32, 32, 0).header_at();
            assert_eq!(slot_bytes, crate::topology::ring_slot_bytes(tp));
            let table = table(tp, false, Drafter::Dflash, 32, 32, 1);
            let history = table.descriptors[2 * SWA_LAYERS as usize];
            assert_eq!(history.length, 10_557_440 * tp.dflash_fc_k() / 5120);
        }
    }

    #[test]
    fn a_snapshot_is_one_ring_slot_plus_the_drafter_state() {
        let kernel_bytes = |drafter| {
            table(Tp::TP4, false, drafter, 32, 32, 3)
                .descriptors
                .iter()
                .filter(|descriptor| KERNEL_BUFFERS.contains(&descriptor.buffer))
                .map(|descriptor| descriptor.length)
                .sum::<u64>()
        };
        assert_eq!(kernel_bytes(Drafter::None), RING_SLOT_BYTES);
        assert_eq!(kernel_bytes(Drafter::Mtp), 7_503_872);
        let pages = |fp8, drafter| table(Tp::TP4, fp8, drafter, 32, 32, 0).pages();
        assert_eq!(pages(false, Drafter::Mtp), 21);
        assert_eq!(pages(true, Drafter::Mtp), 41);
        assert_eq!(pages(false, Drafter::None), 18);
        assert_eq!(pages(true, Drafter::None), 35);
        assert_eq!(pages(false, Drafter::Dflash), 75);
        assert_eq!(pages(true, Drafter::Dflash), 149);
        let history =
            table(Tp::TP4, false, Drafter::Dflash, 32, 32, 2).descriptors[2 * SWA_LAYERS as usize];
        assert_eq!(history.buffer, 1);
        assert_eq!(history.length, 10_557_440);
        assert_eq!(history.offset, 2 * history.length);
    }

    #[test]
    fn pieces_address_the_slot_major_ring_planes() {
        let pieces = table(Tp::TP4, false, Drafter::Mtp, 32, 32, 2).descriptors;
        assert_eq!(pieces[0].offset, 2 * K_SLOT_BYTES);
        assert_eq!(pieces[1].offset, K_LAYER_BYTES + 2 * V_SLOT_BYTES);
        assert_eq!(pieces[2].offset, LAYER_BYTES + 2 * K_SLOT_BYTES);
        let last_ring = &pieces[2 * SWA_LAYERS as usize - 1];
        assert_eq!(
            last_ring.offset + last_ring.length,
            (SWA_LAYERS - 1) * LAYER_BYTES + K_LAYER_BYTES + 3 * V_SLOT_BYTES
        );
        assert!(
            last_ring.offset + last_ring.length <= crate::contract::RING_STATE_BYTES_PER_RANK / 2
        );
        let hidden = pieces.last().unwrap();
        assert_eq!(hidden.buffer, 1);
        assert_eq!(hidden.offset, 2 * MTP_SLOT_ROWS * MTP_HIDDEN_BYTES);
    }

    fn span(buffer: u32, length: u64) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(
            runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(1, buffer),
            0,
            length,
        )
    }

    /// A wider ring places slot 200's planes past the 32-slot state and runs
    /// its own copy entry; the snapshot's size does not change.
    #[test]
    fn wide_rings_address_their_slots_and_entries() {
        let wide = RingGeometry::new(Tp::TP4, 256);
        let pieces = table(Tp::TP4, false, Drafter::None, 256, 256, 200).descriptors;
        assert_eq!(pieces[0].offset, 200 * K_SLOT_BYTES);
        assert_eq!(pieces[1].offset, wide.k_layer + 200 * V_SLOT_BYTES);
        assert_eq!(pieces[2].offset, wide.layer + 200 * K_SLOT_BYTES);
        let last = pieces.last().unwrap();
        assert!(last.offset + last.length <= wide.working);
        let views = |ring_bytes| PrefixCopyViews {
            ring_state: span(1, ring_bytes),
            drafter: None,
            pool: span(3, 128 * GA_PAGE_BYTES_BF16),
            pages: span(4, 1024),
        };
        let copy = |slot: u32, ring_bytes| {
            let table = table(Tp::TP4, false, Drafter::None, 256, 256, slot.min(255));
            prefix_copy(&table, slot, 18, false, views(ring_bytes), wide)
        };
        let launch = copy(200, wide.working).unwrap();
        assert_eq!(launch.symbol, "chop_mimo_ring_prefix_copy_s256_v1");
        assert!(copy(256, wide.working).is_err());
        assert!(copy(200, crate::contract::RING_STATE_BYTES_PER_RANK / 2).is_err());
    }

    #[test]
    fn one_launch_covers_the_snapshot_in_whole_vectors() {
        for (fp8, drafter) in [
            (false, Drafter::Mtp),
            (true, Drafter::None),
            (false, Drafter::Dflash),
            (true, Drafter::Dflash),
        ] {
            let page_bytes = if fp8 {
                GA_PAGE_BYTES_FP8
            } else {
                GA_PAGE_BYTES_BF16
            };
            let table = table(Tp::TP4, fp8, drafter, 32, 32, 5);
            let drafter_bytes = table
                .descriptors
                .iter()
                .filter(|descriptor| descriptor.buffer == 1)
                .map(|descriptor| descriptor.length)
                .sum::<u64>();
            let views = |drafter_bytes: u64| PrefixCopyViews {
                ring_state: span(1, crate::contract::RING_STATE_BYTES_PER_RANK / 2),
                drafter: (drafter_bytes > 0).then(|| span(2, drafter_bytes)),
                pool: span(3, 256 * page_bytes),
                pages: span(4, 2048),
            };
            let pages = table.pages();
            let copy = |slot, pages, drafter_bytes| {
                prefix_copy(
                    &table,
                    slot,
                    pages,
                    true,
                    views(drafter_bytes),
                    ring(Tp::TP4),
                )
            };
            let launch = copy(5, pages, drafter_bytes).unwrap();
            let bytes = RING_SLOT_BYTES + drafter_bytes;
            assert!(bytes.is_multiple_of(16) && page_bytes.is_multiple_of(16));
            let threads = u64::from(launch.plan.grid[0] * launch.plan.block[0]);
            assert!(threads * 16 >= bytes && threads * 16 < bytes + 16 * 256);
            assert_eq!(launch.arguments.0[2], Argument::U64(drafter_bytes));
            assert_eq!(launch.arguments.0[6], Argument::U64(page_bytes));
            assert_eq!(
                launch.arguments.0[1] == Argument::NullDevicePointer,
                drafter == Drafter::None
            );
            assert!(copy(5, pages + 1, drafter_bytes).is_err());
            assert!(copy(32, pages, drafter_bytes).is_err());
            // The kernel addresses the slot it is given: another slot's
            // table does not match it.
            assert!(copy(4, pages, drafter_bytes).is_err());
            assert!(
                copy(5, pages, drafter_bytes + 16).is_err(),
                "the drafter span must be exactly the slot's piece"
            );
        }
    }
}
