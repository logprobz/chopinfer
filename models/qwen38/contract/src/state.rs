//! Qwen3.8's per-request state, declared for the shared bundle runtime
//! (`bundle_runtime::state`): the linear layers' convolution and
//! delta-rule state with their verify checkpoints, the PLE entry, the QSA
//! raw-key ring and, with the drafter, its retained target rows.

use bundle_runtime::kv::KvFormat;
use bundle_runtime::state::{
    Anchors, Cancel, Capture, CheckpointItems, CheckpointRows, Checkpoints, Entries, KvStream,
    Owner, PagedLayers, Region, RequestState, Rollback, Save, Scope, Shadow, StateBuffer, Writer,
};

use crate::contract::*;

/// Target X rows each slot retains for the drafter, by position modulo
/// this many: the rows it has not consumed yet.
const RETAINED_ROWS: u64 = MAX_VERIFY_ROWS as u64;
/// The snapshot header: a layout tag, the tokens, the drafter's frontier
/// and the capturing request.
const HEADER_BYTES: u64 = 32;

/// Whether `format`'s pages and ring hold the MTP drafter's layer.
pub fn has_mtp_layer(format: &KvFormat) -> bool {
    format.layout_id == KV_LAYOUT_BF16_MTP || format.layout_id == KV_LAYOUT_FP8_MTP
}

/// What one instance of `shape`, serving KV pages of `format`, keeps per
/// request, with the MTP drafter loaded or not. Only a format with the
/// drafter's layer serves the drafter.
pub fn request_state(shape: Shape, format: &KvFormat, drafter: bool) -> RequestState {
    let attn_head = shape.attn_head();
    let full_layers = shape.full_layers();
    let full_layers_mtp = shape.full_layers_mtp();
    let index_dim = shape.index_dim();
    let kv_heads = shape.kv_heads();
    let linear_layers = shape.linear_layers();
    let mtp_layers = shape.mtp_layers();
    let ple_entry_bytes = shape.ple_entry_bytes();
    let qsa_block = shape.qsa_block();
    let ring_slot_key_bytes = shape.ring_slot_key_bytes();
    let ring_slot_position_bytes = shape.ring_slot_position_bytes();
    let wide = shape.wide();
    let mtp = has_mtp_layer(format) && shape.mtp_layers() > 0;
    let head = u64::from(kv_heads * attn_head);
    // K, V and the QSA block keys, one per `qsa_block` tokens, of each full
    // layer.
    let full = |name, layers, element| PagedLayers {
        name,
        layers,
        streams: vec![
            KvStream {
                name: "k",
                tokens_per_entry: 1,
                entry_bytes: head * element,
            },
            KvStream {
                name: "v",
                tokens_per_entry: 1,
                entry_bytes: head * element,
            },
            KvStream {
                name: "block-keys",
                tokens_per_entry: u64::from(qsa_block),
                entry_bytes: u64::from(index_dim) * BF16_BYTES,
            },
        ],
    };
    let mut pages = vec![full("full", full_layers, format.element_bytes())];
    if mtp {
        // The drafter's KV is not quantized.
        pages.push(full("mtp", mtp_layers as u64, BF16_BYTES));
    }
    // Up to `ANCHOR_ITEMS` prompt-ending items per step keep their state at
    // their anchor: the linear pools hold the spares after their
    // checkpoints, the rest are separate buffers.
    let anchors = |pooled| Anchors::Spare {
        per_step: ANCHOR_ITEMS,
        pooled,
    };
    // Linear state: each verify row of up to `DRAFT_ITEMS` drafting items
    // writes a checkpoint, and commit exchanges its entry with the slot
    // mapping. AR and prefill items update their slot in place, so a
    // cancelled step leaves its slots invalid.
    let linear = |name, entry_bytes| StateBuffer {
        name,
        layers: linear_layers,
        planes: vec![entry_bytes],
        entries: Entries::Served,
        scope: Scope::Slot,
        shadow: Shadow::None,
        capture: Capture::Full,
        rollback: Rollback::Checkpoint(Checkpoints {
            items: CheckpointItems::Drafting(DRAFT_ITEMS),
            rows: CheckpointRows::Verify,
            pooled: true,
        }),
        cancel: Cancel::Invalidate,
        owner: Owner::Shared,
        drafter: false,
        anchors: anchors(true),
    };
    let mut buffers = vec![
        linear("linear.conv", shape.conv_entry_bytes),
        // F32.
        linear("linear.delta", shape.delta_entry_bytes),
        // The convolution history and the two n-gram ids; the n-gram ids
        // could be rebuilt from the tokens, but they share the entry. Every
        // decode item of a verify step takes `MAX_VERIFY_ROWS` checkpoints
        // in the committed pool. Verify commit exchanges the kept entry;
        // prefill and decode still stage and copy their final entry.
        // Cancel leaves the PLE entry as it was.
        StateBuffer {
            layers: 1,
            rollback: Rollback::Checkpoint(Checkpoints {
                items: CheckpointItems::Bucket,
                rows: CheckpointRows::Fixed(MAX_VERIFY_ROWS),
                pooled: true,
            }),
            cancel: Cancel::None,
            anchors: anchors(false),
            ..linear("ple", ple_entry_bytes)
        },
        // Every full layer's raw keys, then their RoPE positions; the
        // drafter's layer last. A restored request reads the committed open
        // block, at most `qsa_block - 1` keys. A verify step writes at most
        // `MAX_VERIFY_ROWS` positions past it, and a rejected row's entry is
        // rewritten before any row reads it. A prefill chunk can overwrite
        // the open block, so a step with a prefill saves the live ring to
        // the shadow half, a step that only drafts saves the drafter's
        // layer, and cancel copies the save back.
        StateBuffer {
            name: "qsa.ring",
            layers: if mtp { full_layers_mtp } else { full_layers },
            planes: vec![ring_slot_key_bytes, ring_slot_position_bytes],
            entries: Entries::Bucket,
            scope: Scope::WindowRing {
                tokens: u64::from(shape.qsa_ring()),
            },
            shadow: Shadow::Adjacent,
            capture: Capture::RingWindow {
                tokens: u64::from(qsa_block) - 1,
            },
            rollback: Rollback::Truncate,
            cancel: Cancel::Restore(
                [Save {
                    writer: Writer::Prefill,
                    region: Region::All,
                }]
                .into_iter()
                .chain(mtp.then_some(Save {
                    writer: Writer::Drafter,
                    region: Region::Layers {
                        first: full_layers,
                        count: 1,
                    },
                }))
                .collect(),
            ),
            owner: Owner::Shared,
            drafter: false,
            anchors: anchors(false),
        },
    ];
    if drafter {
        // The drafter's frontier may lag the committed tokens by up to
        // these rows; rows past it are rewritten when it reaches them.
        buffers.push(StateBuffer {
            name: "mtp.retained",
            layers: 1,
            planes: vec![RETAINED_ROWS * u64::from(wide) * BF16_BYTES],
            entries: Entries::Served,
            scope: Scope::Rows {
                rows: RETAINED_ROWS,
            },
            shadow: Shadow::None,
            capture: Capture::LastRows {
                rows: RETAINED_ROWS,
            },
            // Commit retains a step's staged rows.
            rollback: Rollback::Truncate,
            cancel: Cancel::None,
            owner: Owner::FrontierOwner,
            drafter: true,
            anchors: anchors(false),
        });
    }
    RequestState {
        page_tokens: PAGE_TOKENS,
        pages,
        buffers,
        header_bytes: HEADER_BYTES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_configuration_validates_and_fills_its_pages() {
        for shape in Shape::ALL {
            for format in &KV_FORMATS {
                for drafter in [false, has_mtp_layer(format)] {
                    let state = request_state(shape, format, drafter);
                    assert_eq!(state.validate(), Ok(()));
                    assert_eq!(state.page_bytes(), format.page_bytes);
                    // The request state layouts the manifest declares.
                    let declared = state.buffers[..3]
                        .iter()
                        .map(|buffer| buffer.entry_bytes())
                        .collect::<Vec<_>>();
                    let layouts = shape.recurrent_state_layouts().map(|(_, bytes)| bytes);
                    assert_eq!(declared, layouts);
                    // The ring state each slot adds: live and saved rings,
                    // padded at the end.
                    let ring = &state.buffers[3];
                    let slot = bundle_runtime::state::Extent {
                        served: 1,
                        bucket: 1,
                        verify_rows: 0,
                    };
                    let padded = if has_mtp_layer(format) {
                        RING_STATE_SLOT_BYTES_MTP
                    } else {
                        RING_STATE_SLOT_BYTES
                    };
                    assert_eq!(ring.bytes(slot).next_multiple_of(256), padded);
                }
            }
        }
    }
}
