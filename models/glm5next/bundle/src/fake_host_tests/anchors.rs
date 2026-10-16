//! A prompt-ending prefill item that keeps its state at its anchor
//! (`batch_limits.anchor_items`): the step writes the anchor item's spares,
//! a capture at the anchor copies them, and a restore of the anchor's
//! tokens continues from them.

use model_glm5next_contract::contract::{ANCHOR_ITEMS, has_mtp_layer};

use super::*;
use crate::buffers::Buffer;

/// A new request's whole prompt of `count` tokens in slot 0, keeping its
/// state at the anchor inside it, as the host sends a prompt that does not
/// end on the granule when it captures the prefix there.
fn anchored_prompt(request: u64, count: u64) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let inputs = (0..count).map(|token| token as u32).collect();
    let item = BatchItemV1::new(request, false, count as u32, true, inputs).with_anchor(true);
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![item],
        vec![BatchAppendV1::new(
            0,
            count,
            count,
            (0..count.div_ceil(64)).collect(),
            None,
        )],
    )
    .with_state_slots(vec![0])
    .encode()
    .unwrap()
}

/// The span of snapshot `piece` in a rank's request-state buffers, or in
/// its separate spare buffer.
fn held_span(
    (buffers, spares): &crate::instance::HeldState,
    piece: &bundle_runtime::snapshot::Descriptor,
) -> DeviceBufferSpanV1 {
    let view = if piece.spare {
        spares[piece.buffer].expect("a separate spare buffer")
    } else {
        buffers[piece.buffer]
    };
    assert!(piece.offset + piece.length <= view.length);
    DeviceBufferSpanV1::new(view.buffer, view.offset + piece.offset, piece.length)
}

/// Writes distinct bytes into every piece of anchor item `anchor`'s
/// snapshot on every rank, and returns them per rank in snapshot order.
fn fill_anchor(host: &FakeHost, handle: BundleHandleV1, anchor: u32) -> Vec<Vec<u8>> {
    let (layouts, table) = crate::instance::anchor_layout(handle, anchor);
    layouts
        .iter()
        .enumerate()
        .map(|(rank, layout)| {
            let mut state = 0x2545_f491_4f6c_dd1d_u64 ^ (rank as u64) << 56;
            let mut all = Vec::new();
            for piece in &table.descriptors {
                let bytes = (0..piece.length.div_ceil(8))
                    .flat_map(|_| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        state.to_le_bytes()
                    })
                    .take(piece.length as usize)
                    .collect::<Vec<_>>();
                host.write(held_span(layout, piece), &bytes);
                all.extend(bytes);
            }
            all
        })
        .collect()
}

/// A 130-token prompt runs as one chunk that keeps its state at its anchor,
/// 128. Its KDA launches read anchor item 0's spare entry, after the slots
/// and checkpoints of both KDA buffers; with the MTP drafter the step also
/// copies the target's normalized rows from the anchor's drafter frontier,
/// 124, to the anchor into the retained rows' spare, each at its position
/// in the ring. After commit, a capture at 128 copies the spares (the slot
/// holds 130 tokens) with that frontier in the header, and a restore of
/// the 128 tokens fills another slot with them; the slot's own 130 tokens
/// still capture from the slot. The next step ends the anchor.
#[test]
fn a_prompt_chunk_keeps_its_anchor_in_the_spares() {
    let row = u64::from(Shape::TP4.profile.hidden) * 2;
    for format in [0, 1].map(|index| Shape::TP4.kv_formats()[index]) {
        let drafting = has_mtp_layer(&format);
        let host = FakeHost::with_format((3, 32), format);
        let mut instance = create_over(&host, format);
        let handle = instance.handle();
        crate::instance::run_lowered_stages_alone(handle);
        host.backing.set(true);
        let normed = crate::instance::scratch_span(handle, 0, Buffer::Normed);
        let at = |span: DeviceBufferSpanV1, index: u64| {
            DeviceBufferSpanV1::new(span.buffer, span.offset + index * row, row)
        };
        for index in 0..130 {
            host.write(at(normed, index), &vec![index as u8 ^ 0x5a; row as usize]);
        }
        instance
            .execute(&anchored_prompt(7, 130), &mut DraftSink::default())
            .unwrap();
        // The KDA engines take the first spare, after the slots and
        // checkpoints: the last `ANCHOR_ITEMS` entries of each layer.
        let entry = crate::instance::anchor_entry(handle, "kda.state", 0);
        assert_eq!(entry, crate::instance::anchor_entry(handle, "kda.conv", 0));
        let declared = request_state(Shape::TP4, &format, drafting);
        let kda = &declared.buffers[1];
        assert_eq!(kda.name, "kda.state");
        let pool = crate::instance::snapshot_layout(handle, 0).0[0][1];
        let entries = pool.length / kda.layers / kda.bytes_per_layer_rank();
        assert!(entry >= u64::from(DECODE_BATCH));
        assert_eq!(entry, entries - u64::from(ANCHOR_ITEMS));
        for rank in 0..4 {
            let ids = crate::instance::scratch_span(handle, rank, Buffer::AnchorIds);
            let first = DeviceBufferSpanV1::new(ids.buffer, ids.offset, 4);
            assert_eq!(
                host.read(first),
                (entry as i32).to_le_bytes(),
                "rank {rank}"
            );
        }
        if drafting {
            // Rows 124..128 at their positions modulo the ring of 6, in
            // the spare after the slots' rows.
            let spare = crate::instance::anchor_entry(handle, "mtp.retained", 0);
            assert_eq!(spare, u64::from(DECODE_BATCH));
            let state = crate::instance::snapshot_layout(handle, 0).0;
            let index = request_state(Shape::TP4, &format, true)
                .buffers
                .iter()
                .position(|buffer| buffer.name == "mtp.retained")
                .unwrap();
            let retained = state[0][index];
            let ring = u64::from(crate::drafter::RETAINED_ROWS);
            for position in 124..128 {
                assert!(
                    host.read(at(retained, spare * ring + position % ring))
                        == host.read(at(normed, position)),
                    "row {position}"
                );
            }
        }
        instance.commit().unwrap();
        // A capture at the anchor copies the spares, every rank its own.
        let spares = fill_anchor(&host, handle, 0);
        let (_, table) = crate::instance::anchor_layout(handle, 0);
        assert_eq!(table.pages(), instance.describe_prefix_state().unwrap());
        let pages = (100..100 + table.pages()).collect::<Vec<_>>();
        instance.capture_prefix_state(7, 128, &pages).unwrap();
        for rank in 0..4 {
            let bytes = super::snapshot_bytes(&host, &table, rank, &pages);
            assert!(bytes[..table.header_at() as usize] == spares[rank as usize][..]);
        }
        // A restore of the anchor's tokens takes them into slot 6.
        let tokens = (0..128).collect::<Vec<u32>>();
        instance
            .restore_prefix_state_in_slot(9, &tokens, &pages, 6)
            .unwrap();
        assert!(super::slot_bytes(&host, handle, 6) == spares);
        assert_eq!(
            crate::instance::drafter_frontier(handle, 6),
            drafting.then_some(124)
        );
        assert_eq!(
            crate::instance::ledger_state(handle, 6),
            (Some((9, 128)), false)
        );
        // The slot's 130 tokens still capture from the slot (into the same
        // pages, which the restore has read).
        let start = host.device.journal.borrow().len();
        instance.capture_prefix_state(7, 130, &pages).unwrap();
        let (_, slot_table) = crate::instance::snapshot_layout(handle, 0);
        assert_eq!(
            super::count(&host, start, "copy "),
            4 * slot_table.descriptors.len()
        );
        // The next step rewrites the spares, so the anchor ends.
        instance
            .execute(&step(&[(7, 0, 130, &[5])]), &mut DraftSink::default())
            .unwrap();
        instance.commit().unwrap();
        assert!(instance.capture_prefix_state(7, 128, &pages).is_err());
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// A prompt whose anchor, 2048, falls on the furthest indexer piece cut
/// runs as one chunk too: the cut moves a granule earlier
/// (`crate::step::pieces`), so the KDA inputs of whichever call holds the
/// anchor name its spare, and no other.
#[test]
fn a_long_prompt_keeps_its_anchor_past_a_piece_cut() {
    let host = FakeHost::new((3, 32));
    let mut instance = create(&host);
    let handle = instance.handle();
    crate::instance::run_lowered_stages_alone(handle);
    host.backing.set(true);
    instance
        .execute(&anchored_prompt(7, 2049), &mut DraftSink::default())
        .unwrap();
    let entry = crate::instance::anchor_entry(handle, "kda.state", 0) as i32;
    let ids = crate::instance::scratch_span(handle, 0, Buffer::AnchorIds);
    let words = host
        .read(DeviceBufferSpanV1::new(ids.buffer, ids.offset, 8))
        .chunks(4)
        .map(|word| i32::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    // The prefill's one item.
    assert!(
        words[0] == entry || words == [-1, entry],
        "{words:?} for spare {entry}"
    );
    instance.commit().unwrap();
    let (_, table) = crate::instance::anchor_layout(handle, 0);
    let pages = (100..100 + table.pages()).collect::<Vec<_>>();
    instance.capture_prefix_state(7, 2048, &pages).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// A request whose flagged item ends on the granule (128 rows from 0) is
/// refused before the step opens: the KDA engines keep no anchor in such a
/// call, so its spares would stay unwritten. Nothing is launched, the slot
/// stays free, no anchor at 64 can be captured, and the instance serves the
/// next request.
#[test]
fn an_anchored_item_that_ends_on_the_granule_is_refused() {
    let host = FakeHost::new((3, 32));
    let mut instance = create(&host);
    let handle = instance.handle();
    crate::instance::run_lowered_stages_alone(handle);
    let start = host.device.journal.borrow().len();
    assert_eq!(
        instance
            .execute(&anchored_prompt(7, 128), &mut DraftSink::default())
            .unwrap_err(),
        BundleAbiError::Status(BundleStatusV1::InvalidArgument)
    );
    assert_eq!(super::count(&host, start, "launch "), 0);
    assert_eq!(crate::instance::ledger_state(handle, 0), (None, false));
    let (_, table) = crate::instance::anchor_layout(handle, 0);
    let pages = (100..100 + table.pages()).collect::<Vec<_>>();
    assert!(instance.capture_prefix_state(7, 64, &pages).is_err());
    assert!(instance.capture_prefix_state(7, 128, &pages).is_err());
    // An unaligned prompt then runs and keeps its anchor.
    instance
        .execute(&anchored_prompt(7, 129), &mut DraftSink::default())
        .unwrap();
    instance.commit().unwrap();
    instance.capture_prefix_state(7, 128, &pages).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
