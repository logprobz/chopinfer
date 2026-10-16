//! `glm5next.v1`'s per-request state, declared for the shared bundle runtime
//! (`bundle_runtime::state`): the sparse layers' paged latent rows and
//! pooled keys, the KDA layers' convolution history and state, the indexer's
//! incomplete-pool tail and, with the drafter, its retained target rows.

use bundle_runtime::kv::KvFormat;
use bundle_runtime::state::{
    Anchors, Cancel, Capture, CheckpointItems, CheckpointRows, Checkpoints, Entries, KvStream,
    Owner, PagedLayers, RequestState, Rollback, Scope, Shadow, StateBuffer,
};
use kernel_core::latent_attention::{LATENT_FP8_SCALE_BLOCK, LatentKvFormat};

use crate::contract::*;

/// Target rows each slot retains for the drafter, by position modulo this
/// many: the rows it has not consumed yet.
const RETAINED_ROWS: u64 = MAX_VERIFY_ROWS as u64;
/// The snapshot header: the shared layout
/// (`bundle_runtime::prefix::HeaderLayout::V1`), whose tokens,
/// capturing request, rank count and layout digest a restore checks.
const HEADER_BYTES: u64 = bundle_runtime::prefix::HeaderLayout::V1.bytes();

/// The streams of one sparse layer's page: for FP8 every token's E4M3
/// codes, then their F32 scales (`LatentKvFormat::page_bytes`); for BF16
/// the latent rows. Then one pooled key per indexer pool.
fn sparse_streams(shape: Shape, kv: LatentKvFormat) -> Vec<KvStream> {
    let latent = u64::from(shape.profile.kv_rank);
    let mut streams = match kv {
        LatentKvFormat::Fp8E4M3Block128 => vec![
            KvStream {
                name: "latent-codes",
                tokens_per_entry: 1,
                entry_bytes: latent,
            },
            KvStream {
                name: "latent-scales",
                tokens_per_entry: 1,
                entry_bytes: 4 * latent.div_ceil(u64::from(LATENT_FP8_SCALE_BLOCK)),
            },
        ],
        LatentKvFormat::Bf16 => vec![KvStream {
            name: "latent",
            tokens_per_entry: 1,
            entry_bytes: latent * BF16_BYTES,
        }],
    };
    streams.push(KvStream {
        name: "pooled-keys",
        tokens_per_entry: u64::from(shape.profile.index_pool),
        entry_bytes: u64::from(shape.profile.indexer_dim) * BF16_BYTES,
    });
    streams
}

/// What one instance of `shape`, serving KV pages of `format`, keeps per
/// request, with the MTP drafter loaded or not. Only a format with the
/// drafter's layer serves the drafter.
pub fn request_state(shape: Shape, format: &KvFormat, drafter: bool) -> RequestState {
    let kv = latent_format(format);
    let mtp = has_mtp_layer(format) && shape.mtp_layers() > 0;
    let mut pages = vec![PagedLayers {
        name: "sparse",
        layers: shape.sparse_layers(),
        streams: sparse_streams(shape, kv),
    }];
    if mtp {
        pages.push(PagedLayers {
            name: "mtp",
            layers: shape.mtp_layers(),
            streams: sparse_streams(shape, kv),
        });
    }
    // Up to `ANCHOR_ITEMS` prompt-ending items per step keep their state at
    // their anchor: the KDA engines write the KDA spares after the
    // checkpoints, the step copies the drafter's retained rows into theirs,
    // and the indexer tail's are a separate buffer no step writes, because
    // an anchor ends a pool and leaves no tail to read.
    let anchors = |pooled| Anchors::Spare {
        per_step: ANCHOR_ITEMS,
        pooled,
    };
    // Every verify row of a drafting item writes a checkpoint, and commit
    // keeps the last accepted row's. Prefill and decode advance the slot in
    // place, so a cancelled step leaves its slots invalid.
    let slot = |name, layers, planes| StateBuffer {
        name,
        layers,
        planes,
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
    let tail = shape.indexer_tail_bytes();
    let mut buffers = vec![
        // The previous `conv_taps - 1` q, k and v inputs, BF16.
        slot(
            "kda.conv",
            shape.linear_layers(),
            vec![shape.kda_conv_bytes()],
        ),
        // F32 `[heads, head_dim, head_dim]`, value-major.
        slot(
            "kda.state",
            shape.linear_layers(),
            vec![shape.kda_state_bytes()],
        ),
        // The incomplete pool's normalized keys, then its gate logits, as
        // one opaque plane: the kernel stores each entry's key and gate
        // banks back to back. The drafter's layer last. The host allocates
        // it per state slot (`chop.state-per-item.v1`): every item of the
        // largest bucket owns an entry and `MAX_VERIFY_ROWS` checkpoints,
        // at its index in the step.
        StateBuffer {
            entries: Entries::Bucket,
            rollback: Rollback::Checkpoint(Checkpoints {
                items: CheckpointItems::Bucket,
                rows: CheckpointRows::Fixed(MAX_VERIFY_ROWS),
                pooled: true,
            }),
            anchors: anchors(false),
            ..slot("indexer.tail", shape.paged_layers(mtp), vec![tail])
        },
    ];
    if drafter {
        // The drafter reads each row's normalized mean of the streams
        // (BF16, hidden wide). Its frontier may lag the committed tokens by
        // up to these rows; rows past it are rewritten when it reaches them.
        buffers.push(StateBuffer {
            name: "mtp.retained",
            layers: 1,
            planes: vec![RETAINED_ROWS * u64::from(shape.profile.hidden) * BF16_BYTES],
            entries: Entries::Served,
            scope: Scope::Rows {
                rows: RETAINED_ROWS,
            },
            shadow: Shadow::None,
            capture: Capture::LastRows {
                rows: RETAINED_ROWS,
            },
            rollback: Rollback::Truncate,
            cancel: Cancel::None,
            owner: Owner::FrontierOwner,
            drafter: true,
            anchors: anchors(true),
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
        let shape = Shape::TP4;
        for format in &shape.kv_formats() {
            for drafter in [false, has_mtp_layer(format)] {
                let state = request_state(shape, format, drafter);
                assert_eq!(state.validate(), Ok(()));
                assert_eq!(state.page_bytes(), format.page_bytes);
                // The request state layouts the manifest declares.
                let declared = state.buffers[..2]
                    .iter()
                    .map(|buffer| buffer.entry_bytes())
                    .collect::<Vec<_>>();
                let layouts = shape.recurrent_state_layouts().map(|(_, bytes)| bytes);
                assert_eq!(declared, layouts);
                // The per-item tail: one state slot is one bucket item.
                let slot = bundle_runtime::state::Extent {
                    served: 1,
                    bucket: 1,
                    verify_rows: MAX_VERIFY_ROWS,
                };
                assert_eq!(
                    state.buffers[2].bytes(slot).next_multiple_of(256),
                    shape.indexer_tail_slot_bytes(has_mtp_layer(format))
                );
                assert_eq!(state.buffers.len(), 3 + usize::from(drafter));
            }
        }
    }

    #[test]
    fn verify_rows_take_a_checkpoint_of_every_slot_buffer() {
        use bundle_runtime::state::{Commit, Extent};
        let shape = Shape::TP4;
        let format = &shape.kv_formats()[1];
        let state = request_state(shape, format, true);
        let extent = Extent {
            served: DECODE_BATCH,
            bucket: DECODE_BATCH,
            verify_rows: 4,
        };
        for buffer in &state.buffers[..3] {
            assert_eq!(buffer.rollback.commit(4, 2), Commit::Checkpoint { row: 1 });
        }
        for buffer in &state.buffers[..2] {
            assert_eq!(
                buffer.checkpoint_entries(extent),
                u64::from(DECODE_BATCH) * 4
            );
        }
        assert_eq!(
            state.buffers[2].checkpoint_entries(extent),
            u64::from(DECODE_BATCH * MAX_VERIFY_ROWS)
        );
        // The host's KDA reservation (the committed entry and four
        // checkpoints for each served item) and its per-item tail, then the
        // KDA anchor spares, which come from the memory the KV pool leaves.
        let reserved: u64 = state.buffers[..3]
            .iter()
            .map(|buffer| buffer.bytes(extent))
            .sum();
        let copy: u64 = shape
            .recurrent_state_layouts()
            .iter()
            .map(|(_, bytes)| bytes)
            .sum();
        assert_eq!(
            reserved,
            u64::from(DECODE_BATCH) * shape.state_slot_bytes(true, 4)
                + u64::from(ANCHOR_ITEMS) * copy
        );
    }

    /// An anchor capture copies the KDA spares after the checkpoints, at
    /// one entry both KDA buffers share, the drafter's retained spares after
    /// the slots, and the separate indexer-tail spares.
    #[test]
    fn anchor_spares_follow_the_checkpoints_or_take_their_own_buffer() {
        use bundle_runtime::state::{Extent, Holder};
        let shape = Shape::TP4;
        let format = &shape.kv_formats()[1];
        let state = request_state(shape, format, true);
        assert_eq!(state.validate(), Ok(()));
        let extent = Extent {
            served: DECODE_BATCH,
            bucket: DECODE_BATCH,
            verify_rows: 4,
        };
        let served = u64::from(DECODE_BATCH);
        for anchor in 0..ANCHOR_ITEMS {
            let pieces = state.pieces(extent, Holder::Anchor(anchor)).unwrap();
            for (index, buffer) in state.buffers.iter().enumerate() {
                let entry = served
                    + match buffer.name {
                        "kda.conv" | "kda.state" => served * 4,
                        _ => 0,
                    }
                    + u64::from(anchor);
                let separate = buffer.name == "indexer.tail";
                let entries = if separate {
                    u64::from(ANCHOR_ITEMS)
                } else {
                    buffer.entries(extent)
                };
                let expected = (0..buffer.layers)
                    .map(|layer| {
                        let at = if separate { u64::from(anchor) } else { entry };
                        (
                            separate,
                            layer * buffer.layer_stride(entries)
                                + at * buffer.bytes_per_layer_rank(),
                        )
                    })
                    .collect::<Vec<_>>();
                let observed = pieces
                    .iter()
                    .filter(|piece| piece.buffer == index)
                    .map(|piece| (piece.spare, piece.offset))
                    .collect::<Vec<_>>();
                assert_eq!(observed, expected, "{}", buffer.name);
            }
        }
        assert!(state.pieces(extent, Holder::Anchor(ANCHOR_ITEMS)).is_err());
        assert_eq!(
            state.buffers[2].spare_bytes(),
            u64::from(ANCHOR_ITEMS) * state.buffers[2].entry_bytes()
        );
    }
}
