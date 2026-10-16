//! The per-sequence SWA ring: its byte layout, and its transactions over
//! the shared ledger (`bundle_runtime::ledger`) and window ring
//! (`bundle_runtime::ring`).
//!
//! Each rank's state has a working half and a shadow half. Before execution,
//! the bundle copies only touched ring cells from working to shadow, so the
//! shadow holds their values from before the step. Kernels write working and
//! nothing reads the shadow except cancel, which copies the touched cells
//! back. Commit therefore only advances the ledger. A verify step may commit
//! only a prefix of an item's rows: the rejected tail's positions are copied
//! back from the shadow, because a rejected write replaced an entry that
//! later queries still read. Retirement frees the sequence slot after its
//! transaction has ended.

use bundle_runtime::geometry::Items;
use bundle_runtime::ledger::Item;
use runtime_bundle_sdk::BatchRequestV1;

#[cfg(test)]
use crate::contract::{RING_TOKENS, SWA_LAYERS};
use crate::topology::Tp;

/// Byte layout of one rank's ring state at a rank count and ring slot count:
/// every layer's K then V array over all slots and positions, with the
/// rank's SWA KV heads per entry. An instance holds one ring slot per item of
/// its largest graph bucket (`Items::capacity`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RingGeometry {
    pub k_entry: u64,
    pub v_entry: u64,
    pub k_layer: u64,
    pub v_layer: u64,
    pub layer: u64,
    pub working: u64,
    /// Ring slots of every plane.
    pub slots: u32,
    pub tp: Tp,
}

impl RingGeometry {
    #[must_use]
    pub fn new(tp: Tp, slots: u32) -> Self {
        let heads = tp.kv_heads(false);
        let (k_entry, v_entry) = (
            heads * tp.geometry.swa_head_dim() as u64 * 2,
            heads * tp.geometry.swa_value_head_dim() as u64 * 2,
        );
        let k_layer = u64::from(slots) * tp.geometry.window() as u64 * k_entry;
        let v_layer = u64::from(slots) * tp.geometry.window() as u64 * v_entry;
        Self {
            k_entry,
            v_entry,
            k_layer,
            v_layer,
            layer: k_layer + v_layer,
            working: crate::topology::ring_state_bytes(tp, slots) / 2,
            slots,
            tp,
        }
    }

    /// The ring of an instance serving `items`.
    #[must_use]
    pub fn served(tp: Tp, items: Items) -> Self {
        Self::new(tp, items.capacity())
    }
}

/// The CUDA ring kernels index each layer's K and V arrays across all slots.
pub fn cache_views(
    state: runtime_bundle_abi::DeviceBufferSpanV1,
    layer: u32,
    ring: RingGeometry,
) -> Option<(
    runtime_bundle_abi::DeviceBufferSpanV1,
    runtime_bundle_abi::DeviceBufferSpanV1,
)> {
    if layer >= ring.tp.geometry.sliding_layers() as u32
        || state.length < (u64::from(layer) + 1) * ring.layer
    {
        return None;
    }
    let k_offset = state.offset.checked_add(u64::from(layer) * ring.layer)?;
    let v_offset = k_offset.checked_add(ring.k_layer)?;
    Some((
        runtime_bundle_abi::DeviceBufferSpanV1::new(state.buffer, k_offset, ring.k_layer),
        runtime_bundle_abi::DeviceBufferSpanV1::new(state.buffer, v_offset, ring.v_layer),
    ))
}

/// The ledger items of a batch: each item's rows from its committed tokens,
/// with verify items marked, in item order.
pub fn items(request: &BatchRequestV1) -> Vec<Item> {
    request
        .items()
        .iter()
        .zip(request.appends())
        .map(|(item, append)| Item {
            request: item.request_id(),
            slot: None,
            start: append.previous_tokens(),
            rows: item.phase_tokens(),
            verify: item.is_verify(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use bundle_runtime::ledger::{Ledger, LedgerError, Placement, Rules};
    use bundle_runtime::ring::{Cell, WindowRing, tails};
    use bundle_runtime::state::Extent;
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};

    use super::*;
    use crate::contract::{CONTEXT_TOKENS, KV_FORMATS, MAX_VERIFY_ROWS, RING_STATE_BYTES_PER_RANK};

    const TP4: RingGeometry = RingGeometry {
        k_entry: 768,
        v_entry: 512,
        k_layer: 32 * 128 * 768,
        v_layer: 32 * 128 * 512,
        layer: 32 * 128 * 1280,
        working: RING_STATE_BYTES_PER_RANK / 2,
        slots: 32,
        tp: Tp::TP4,
    };
    /// 32 served sequences: the ring of TP4's journals.
    const ITEMS: Items = match Items::new(32) {
        Some(items) => items,
        None => panic!("32 items"),
    };
    const K_ENTRY_BYTES: u64 = TP4.k_entry;
    const V_ENTRY_BYTES: u64 = TP4.v_entry;
    const K_LAYER_BYTES: u64 = TP4.k_layer;
    const V_LAYER_BYTES: u64 = TP4.v_layer;
    const LAYER_BYTES: u64 = TP4.layer;
    const WORKING_BYTES: u64 = TP4.working;

    /// The ledger and window ring of a TP4 instance serving `items`.
    fn declared(items: Items) -> (Ledger, WindowRing) {
        let state = model_mimo26_contract::state::request_state(
            Tp::TP4,
            &KV_FORMATS[0],
            crate::prefix::Drafter::None,
        );
        let extent = Extent {
            served: items.get(),
            bucket: items.capacity(),
            verify_rows: MAX_VERIFY_ROWS,
        };
        (
            Ledger::new(
                items.get(),
                Placement::Ledger,
                Rules::of(&state),
                CONTEXT_TOKENS,
            ),
            WindowRing::new(&state, 0, extent).unwrap(),
        )
    }

    fn item(request: u64, start: u64, rows: u32, verify: bool) -> Item {
        Item {
            request,
            slot: None,
            start,
            rows,
            verify,
        }
    }

    /// Opens a step of plain `(request, start, rows)` appends and returns
    /// the cells it saves.
    fn begin(
        ledger: &mut Ledger,
        window: &WindowRing,
        appends: &[(u64, u64, u32)],
    ) -> Result<Vec<Cell>, LedgerError> {
        let items = appends
            .iter()
            .map(|&(request, start, rows)| item(request, start, rows, false))
            .collect::<Vec<_>>();
        let step = ledger.begin(&items, Vec::new())?;
        let appends = step
            .items
            .iter()
            .map(|placed| (placed.slot, placed.start, placed.rows))
            .collect::<Vec<_>>();
        Ok(window.touched(&appends).unwrap())
    }

    fn committed(ledger: &Ledger, request: u64) -> Option<u64> {
        ledger
            .slot_of(request)
            .and_then(|slot| ledger.holder(slot))
            .map(|(_, tokens)| tokens)
    }

    #[test]
    fn geometry_scales_the_entries_with_the_rank_heads() {
        assert_eq!(RingGeometry::new(Tp::TP4, 32), TP4);
        assert_eq!(RingGeometry::served(Tp::TP4, ITEMS), TP4);
        {
            let (tp, heads) = (Tp::TP2, 4);
            let ring = RingGeometry::new(tp, 32);
            assert_eq!((ring.k_entry, ring.v_entry), (heads * 384, heads * 256));
            // `mimo_ring.cuh`'s working half: 39 layers of 32 slots x 128.
            assert_eq!(ring.working, SWA_LAYERS * ring.layer);
            assert_eq!(ring.working, 39 * 32 * 128 * heads * 320 * 2);
        }
    }

    /// Served counts above 32 widen every plane to the largest bucket's
    /// slots, and keep one ledger slot per served sequence.
    #[test]
    fn served_counts_size_the_ring_by_their_largest_bucket() {
        for (served, slots) in [(1, 1), (33, 64), (64, 64), (100, 128), (256, 256)] {
            let items = Items::new(served).unwrap();
            let ring = RingGeometry::served(Tp::TP4, items);
            assert_eq!(ring.slots, slots);
            assert_eq!(ring.k_layer, u64::from(slots) * 128 * 768);
            assert_eq!(ring.working, SWA_LAYERS * ring.layer);
            assert_eq!(
                2 * ring.working,
                crate::topology::ring_state_bytes(Tp::TP4, slots)
            );
            let (mut ledger, window) = declared(items);
            assert_eq!(window.working, ring.working);
            let appends = (0..u64::from(served))
                .map(|request| (request, 0, 1))
                .collect::<Vec<_>>();
            let cells = begin(&mut ledger, &window, &appends).unwrap();
            assert_eq!(ledger.slot_of(u64::from(served) - 1), Some(served - 1));
            // The last slot's V entry of the last layer ends the working half.
            let last = cells.iter().map(|cell| cell.working_offset).max().unwrap();
            assert!(last < ring.working);
            ledger.commit(&[]).unwrap();
            assert_eq!(
                begin(&mut ledger, &window, &[(1000, 0, 1)]),
                Err(LedgerError::NoSlot)
            );
        }
    }

    /// Every cell the window ring saves is where the ring kernels address
    /// it: slot-major K, then V, in each layer, the shadow a working half
    /// later.
    #[test]
    fn cells_follow_the_cuda_slot_major_addressing() {
        for tp in Tp::SUPPORTED {
            for served in [1, 32, 100, 256] {
                let items = Items::new(served).unwrap();
                let geometry = RingGeometry::served(tp, items);
                let state = model_mimo26_contract::state::request_state(
                    tp,
                    &KV_FORMATS[0],
                    crate::prefix::Drafter::None,
                );
                let extent = Extent {
                    served,
                    bucket: items.capacity(),
                    verify_rows: MAX_VERIFY_ROWS,
                };
                let window = WindowRing::new(&state, 0, extent).unwrap();
                let slot = served - 1;
                let cells = window.touched(&[(slot, 300, 2)]).unwrap();
                let mut expected = Vec::new();
                for position in [300 % RING_TOKENS, 301 % RING_TOKENS] {
                    for layer in 0..SWA_LAYERS {
                        let token = u64::from(slot) * RING_TOKENS + position;
                        for (plane, entry) in
                            [(0, geometry.k_entry), (geometry.k_layer, geometry.v_entry)]
                        {
                            let offset = layer * geometry.layer + plane + token * entry;
                            expected.push((offset, geometry.working + offset, entry));
                        }
                    }
                }
                let cells = cells
                    .iter()
                    .map(|cell| (cell.working_offset, cell.shadow_offset, cell.length))
                    .collect::<Vec<_>>();
                assert_eq!(cells, expected, "{tp:?} {served}");
            }
        }
    }

    #[test]
    fn cache_planes_match_cuda_slot_major_addressing() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let state = DeviceBufferSpanV1::new(buffer, 4096, RING_STATE_BYTES_PER_RANK);
        let (k, v) = cache_views(state, 1, TP4).expect("second sliding layer planes");
        assert_eq!(k.offset, 4096 + LAYER_BYTES);
        assert_eq!(v.offset, k.offset + K_LAYER_BYTES);
        assert_eq!(k.length, K_LAYER_BYTES);
        assert_eq!(v.length, V_LAYER_BYTES);
        assert!(cache_views(state, SWA_LAYERS as u32, TP4).is_none());

        let (mut ledger, window) = declared(ITEMS);
        let cells = begin(&mut ledger, &window, &[(7, 0, 1)]).expect("single token");
        assert_eq!(cells[2].working_offset, LAYER_BYTES);
        assert_eq!(cells[3].working_offset, LAYER_BYTES + K_LAYER_BYTES);
        assert_eq!(cells[2].length, K_ENTRY_BYTES);
        assert_eq!(cells[3].length, V_ENTRY_BYTES);
    }

    #[test]
    fn cancel_restores_the_ledger_and_retire_reuses_slot() {
        let (mut ledger, window) = declared(ITEMS);
        begin(&mut ledger, &window, &[(7, 0, 1)]).unwrap();
        ledger.commit(&[]).unwrap();
        let cells = begin(&mut ledger, &window, &[(7, 1, 130)]).expect("prefill touch set");
        assert_eq!(cells.len(), 2 * 128 * SWA_LAYERS as usize);
        assert!(cells.iter().all(|cell| {
            cell.working_offset + cell.length <= WORKING_BYTES
                && cell.shadow_offset + cell.length <= RING_STATE_BYTES_PER_RANK
        }));
        assert_eq!(ledger.retire(7), Err(LedgerError::Open));
        ledger.cancel().expect("cancel staged append");
        assert_eq!(committed(&ledger, 7), Some(1));
        assert_eq!(
            begin(&mut ledger, &window, &[(7, 1, 1)])
                .expect("replay")
                .len(),
            2 * SWA_LAYERS as usize
        );
        ledger.commit(&[]).expect("commit replay");
        assert_eq!(committed(&ledger, 7), Some(2));
        assert_eq!(ledger.retire(7), Ok(Some(0)));
        begin(&mut ledger, &window, &[(8, 0, 1)]).unwrap();
        assert_eq!(ledger.slot_of(8), Some(0));
    }

    /// Models both halves as one value per cell: begin saves working into
    /// shadow, the step overwrites working, cancel copies shadow back.
    #[test]
    fn cancel_restores_committed_cells_without_a_commit_copy() {
        use std::collections::BTreeMap;
        let mut memory = BTreeMap::<u64, u32>::new();
        let (mut ledger, window) = declared(ITEMS);
        let mut step = |appends: &[(u64, u64, u32)], value: u32, commit: bool| {
            let cells = begin(&mut ledger, &window, appends).expect("step");
            for cell in &cells {
                let saved = memory.get(&cell.working_offset).copied().unwrap_or(0);
                memory.insert(cell.shadow_offset, saved);
            }
            for cell in &cells {
                memory.insert(cell.working_offset, value);
            }
            if commit {
                ledger.commit(&[]).expect("commit");
            } else {
                for cell in &cells {
                    let saved = memory[&cell.shadow_offset];
                    memory.insert(cell.working_offset, saved);
                }
                ledger.cancel().expect("cancel");
            }
            (cells, memory.clone())
        };
        let (prefill, _) = step(&[(7, 0, 130)], 1, true);
        let (decode, memory) = step(&[(7, 130, 1)], 2, false);
        assert!(decode.iter().all(|cell| memory[&cell.working_offset] == 1));
        step(&[(7, 130, 1)], 3, true);
        let (next, memory) = step(&[(7, 131, 1)], 4, false);
        assert!(next.iter().all(|cell| memory[&cell.working_offset] == 1));
        assert!(decode.iter().all(|cell| memory[&cell.working_offset] == 3));
        assert_eq!(prefill.len(), 2 * 128 * SWA_LAYERS as usize);
        assert_eq!(committed(&ledger, 7), Some(131));
    }

    #[test]
    fn a_kept_prefix_advances_to_the_kept_end_and_restores_the_tail() {
        let (mut ledger, _) = declared(ITEMS);
        ledger
            .begin(&[item(1, 0, 200, false), item(2, 0, 3, false)], Vec::new())
            .expect("prefill");
        ledger.commit(&[]).expect("commit prefill");
        ledger
            .begin(
                &[
                    item(1, 200, 5, true),
                    item(2, 3, 1, false),
                    item(3, 0, 4, false),
                ],
                Vec::new(),
            )
            .expect("verify, decode and prefill");
        for kept in [&[(1, 0)][..], &[(1, 6)], &[(9, 1)], &[(1, 1), (1, 2)], &[]] {
            assert_eq!(ledger.commit(kept), Err(LedgerError::Kept));
        }
        let committed_items = ledger.commit(&[(1, 2)]).expect("kept prefix");
        assert_eq!(
            tails(&committed_items),
            [bundle_runtime::ring::Tail {
                slot: 0,
                start: 202,
                count: 3
            }]
        );
        assert_eq!(committed(&ledger, 1), Some(202));
        assert_eq!(committed(&ledger, 2), Some(4));
        assert_eq!(committed(&ledger, 3), Some(4));
        // The next step starts at the kept end.
        assert_eq!(
            ledger
                .begin(&[item(1, 205, 1, false)], Vec::new())
                .map(|_| ()),
            Err(LedgerError::Tokens)
        );
        ledger
            .begin(&[item(1, 202, 1, false)], Vec::new())
            .expect("resume at kept end");
    }

    /// Models the ring as one value per cell: a rejected tail's restore
    /// leaves every cell as if only the kept prefix had been written.
    #[test]
    fn restoring_a_rejected_tail_matches_writing_only_the_kept_prefix() {
        use std::collections::BTreeMap;
        for kept in 1..=8u32 {
            let (mut ledger, window) = declared(ITEMS);
            let mut working = BTreeMap::<u64, u64>::new();
            let mut shadow = BTreeMap::<u64, u64>::new();
            let cell = |position: u64| position % RING_TOKENS;
            // A committed history of 300 positions, each cell holding the
            // last position written to it.
            for (start, rows) in [(0, 128), (128, 128), (256, 44)] {
                begin(&mut ledger, &window, &[(7, start, rows)]).expect("history");
                ledger.commit(&[]).expect("history");
            }
            for position in 0..300 {
                working.insert(cell(position), position);
            }
            let step = ledger
                .begin(&[item(7, 300, 8, true)], Vec::new())
                .expect("verify");
            let placed = step.items[0];
            let cells = window
                .touched(&[(placed.slot, placed.start, placed.rows)])
                .unwrap();
            let touched: Vec<u64> = cells
                .iter()
                .filter(|cell| cell.layer == 0 && cell.working_offset < K_LAYER_BYTES)
                .map(|cell| u64::from(cell.position))
                .collect();
            assert_eq!(touched.len(), 8);
            for position in &touched {
                shadow.insert(*position, working[position]);
            }
            for position in 300..308 {
                working.insert(cell(position), position);
            }
            let committed_items = ledger.commit(&[(7, kept)]).expect("kept prefix");
            for tail in tails(&committed_items) {
                for position in tail.start..tail.start + u64::from(tail.count) {
                    working.insert(cell(position), shadow[&cell(position)]);
                }
            }
            let mut expected = BTreeMap::new();
            for position in 0..300 + u64::from(kept) {
                expected.insert(cell(position), position);
            }
            assert_eq!(working, expected, "kept {kept}");
            assert_eq!(committed(&ledger, 7), Some(300 + u64::from(kept)));
        }
    }

    #[test]
    fn mixed_batch_touches_each_request_once() {
        let (mut ledger, window) = declared(ITEMS);
        let cells = begin(&mut ledger, &window, &[(1, 0, 8192), (2, 0, 1)])
            .expect("mixed prefill and decode");
        assert_eq!(cells.len(), 2 * 129 * SWA_LAYERS as usize);
        assert_eq!(
            begin(&mut ledger, &window, &[(1, 0, 1)]),
            Err(LedgerError::Open)
        );
        ledger.commit(&[]).expect("commit mixed batch");
        assert_eq!(committed(&ledger, 1), Some(8192));
        assert_eq!(committed(&ledger, 2), Some(1));
    }

    #[test]
    fn a_restored_slot_resumes_at_its_committed_length() {
        let (mut ledger, window) = declared(ITEMS);
        begin(&mut ledger, &window, &[(1, 0, 1)]).unwrap();
        ledger.commit(&[]).unwrap();
        assert_eq!(ledger.restore(1, 64, None), Err(LedgerError::ForeignSlot));
        assert_eq!(ledger.restore(2, 0, None), Err(LedgerError::Tokens));
        assert_eq!(ledger.restore(2, 70, None), Ok(1));
        assert_eq!(committed(&ledger, 2), Some(70));
        assert_eq!(
            begin(&mut ledger, &window, &[(2, 0, 1)]),
            Err(LedgerError::Tokens),
            "a restored request continues at its boundary"
        );
        begin(&mut ledger, &window, &[(2, 70, 5)]).expect("continue the prefill");
        assert_eq!(ledger.restore(3, 8, None), Err(LedgerError::Open));
        ledger.commit(&[]).expect("commit the continuation");
        assert_eq!(committed(&ledger, 2), Some(75));
    }

    #[test]
    fn new_slots_follow_commit_cancel_and_failed_begin() {
        let (mut ledger, window) = declared(ITEMS);
        begin(&mut ledger, &window, &[(10, 0, 1)]).expect("new request");
        assert_eq!(ledger.slot_of(10), Some(0));
        ledger.cancel().expect("cancel new request");
        assert_eq!(ledger.slot_of(10), None);

        begin(&mut ledger, &window, &[(11, 0, 2)]).expect("new prefill");
        ledger.commit(&[]).expect("commit new request");
        assert_eq!(ledger.slot_of(11), Some(0));
        assert_eq!(committed(&ledger, 11), Some(2));

        assert_eq!(
            begin(&mut ledger, &window, &[(12, 0, 1), (13, 5, 1)]),
            Err(LedgerError::UnknownRequest)
        );
        assert_eq!(ledger.slot_of(12), None);
        assert_eq!(
            begin(&mut ledger, &window, &[(12, 0, 1), (12, 0, 1)]),
            Err(LedgerError::InvalidStep)
        );
        assert_eq!(ledger.slot_of(12), None);

        let items = [item(11, 2, 1, false), item(12, 0, 1, false)];
        assert_eq!(ledger.preview(&items), Ok(vec![0, 1]));
        assert_eq!(ledger.slot_of(12), None);
        begin(&mut ledger, &window, &[(11, 2, 1), (12, 0, 1)]).expect("mixed existing and new");
        ledger.cancel().expect("cancel mixed step");
        assert_eq!(committed(&ledger, 11), Some(2));
        assert_eq!(ledger.slot_of(12), None);
    }
}
