//! MiMo's per-request state, declared for the shared bundle runtime
//! (`bundle_runtime::state`): the sliding-window ring of every SWA
//! layer, and the loaded drafter's rows and ring.

use bundle_runtime::kv::KvFormat;
use bundle_runtime::state::{
    Anchors, Cancel, Capture, Entries, KvStream, Owner, PagedLayers, Region, RequestState,
    Rollback, Save, Scope, Shadow, StateBuffer, Writer,
};

use crate::Tp;
use crate::serving::{BF16_BYTES, DRAFT_ITEMS, PAGE_TOKENS};

/// The drafter an instance loads, whose history its snapshots hold.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Drafter {
    None,
    Mtp,
    Dflash,
}

/// Positions of each slot's DFlash projected context ring.
/// Policy: the projected context ring keeps room for a window and an open draft block.
pub(crate) const DFLASH_CACHE_TOKENS: u64 = 2048;
/// The DFlash snapshot header: the boundary and the projection cursor.
const DFLASH_HEADER_BYTES: u64 = 64;

/// What one instance at `tp`, serving KV pages of `format` with `drafter`
/// loaded, keeps per request.
pub fn request_state(tp: Tp, format: &KvFormat, drafter: Drafter) -> RequestState {
    let element = format.element_bytes();
    let heads = tp.kv_heads(true);
    let stream = |name, width| KvStream {
        name,
        tokens_per_entry: 1,
        entry_bytes: heads * width * element,
    };
    let swa = tp.kv_heads(false);
    // One SWA layer's ring: K, then V.
    let ring_planes = vec![
        (tp.geometry.window() as u64) * swa * (tp.geometry.swa_head_dim() as u64) * BF16_BYTES,
        (tp.geometry.window() as u64)
            * swa
            * (tp.geometry.swa_value_head_dim() as u64)
            * BF16_BYTES,
    ];
    let mut buffers = vec![StateBuffer {
        name: "swa.ring",
        layers: (tp.geometry.sliding_layers() as u64),
        planes: ring_planes.clone(),
        entries: Entries::Bucket,
        scope: Scope::WindowRing {
            tokens: (tp.geometry.window() as u64),
        },
        // Every step saves the ring entries it touches to the shadow half;
        // commit restores a rejected tail's from it, cancel all of them.
        shadow: Shadow::Adjacent,
        capture: Capture::RingWindow {
            tokens: (tp.geometry.window() as u64),
        },
        rollback: Rollback::RestoreTail,
        cancel: Cancel::Restore(vec![Save {
            writer: Writer::Any,
            region: Region::Touched,
        }]),
        owner: Owner::Shared,
        drafter: false,
        anchors: Anchors::None,
    }];
    // A drafter's history continues at the committed end: rejected rows
    // lie past it and the next step rewrites them.
    let history = |name, entries, window: u64, row_bytes: u64| StateBuffer {
        name,
        layers: 1,
        planes: vec![crate::resources::retained_capacity(window) * row_bytes],
        entries,
        scope: Scope::Rows {
            rows: crate::resources::retained_capacity(window),
        },
        shadow: Shadow::None,
        capture: Capture::LastRows { rows: window },
        rollback: Rollback::Truncate,
        cancel: Cancel::None,
        owner: Owner::Shared,
        drafter: true,
        anchors: Anchors::None,
    };
    match drafter {
        Drafter::None => {}
        // The target hidden rows the MTP layer rebuilds its ring from, in
        // the `DRAFT_ITEMS` slots an MTP instance serves at most.
        Drafter::Mtp => buffers.extend([
            history(
                "mtp.hidden",
                Entries::Fixed(DRAFT_ITEMS),
                tp.geometry.window() as u64,
                u64::from(tp.geometry.hidden()) * BF16_BYTES,
            ),
            // The MTP layer's own ring, rebuilt from the hidden rows after a
            // restore, so not captured. A step whose drafter runs copies it
            // whole to a separate buffer first, and cancel copies it back.
            // Its catch-up writes committed rows only; the drafter program
            // undoes its chained rows' writes within the step.
            StateBuffer {
                name: "mtp.ring",
                layers: 1,
                planes: ring_planes,
                entries: Entries::Fixed(DRAFT_ITEMS),
                scope: Scope::WindowRing {
                    tokens: (tp.geometry.window() as u64),
                },
                shadow: Shadow::Separate,
                capture: Capture::None,
                rollback: Rollback::None,
                cancel: Cancel::Restore(vec![Save {
                    writer: Writer::Drafter,
                    region: Region::All,
                }]),
                owner: Owner::Shared,
                drafter: true,
                anchors: Anchors::None,
            },
        ]),
        Drafter::Dflash => {
            // The rank's K shard of the raw context features.
            buffers.push(history(
                "dflash.history",
                Entries::Served,
                (tp.geometry.dflash().sliding_window - 1) as u64,
                tp.dflash_fc_k() * BF16_BYTES,
            ));
            // The context K and V projected from them, up to the cursor the
            // header records. Projection reads committed rows only.
            let plane = DFLASH_CACHE_TOKENS
                * tp.dflash_kv_heads()
                * (tp.geometry.dflash().head_dim as u64)
                * BF16_BYTES;
            buffers.push(StateBuffer {
                name: "dflash.context",
                layers: u64::from(tp.geometry.dflash().num_hidden_layers),
                planes: vec![plane, plane],
                entries: Entries::Bucket,
                scope: Scope::WindowRing {
                    tokens: DFLASH_CACHE_TOKENS,
                },
                shadow: Shadow::None,
                capture: Capture::RingWindow {
                    tokens: DFLASH_CACHE_TOKENS,
                },
                rollback: Rollback::None,
                cancel: Cancel::None,
                owner: Owner::Shared,
                drafter: true,
                anchors: Anchors::None,
            });
        }
    }
    RequestState {
        page_tokens: PAGE_TOKENS,
        pages: vec![PagedLayers {
            name: "ga",
            layers: (tp.geometry.global_layers() as u64),
            streams: vec![
                stream("k", tp.geometry.head_dim() as u64),
                stream("v", tp.geometry.value_head_dim() as u64),
            ],
        }],
        buffers,
        header_bytes: if drafter == Drafter::Dflash {
            DFLASH_HEADER_BYTES
        } else {
            0
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving::KV_FORMATS;

    #[test]
    fn every_configuration_validates_and_fills_its_pages() {
        for tp in Tp::SUPPORTED {
            for format in &KV_FORMATS {
                for drafter in [Drafter::None, Drafter::Mtp, Drafter::Dflash] {
                    let state = request_state(tp, format, drafter);
                    assert_eq!(state.validate(), Ok(()), "{tp:?} {drafter:?}");
                    assert_eq!(
                        state.page_bytes(),
                        crate::topology::ga_page_bytes(tp, format.fp8())
                    );
                    assert_eq!(
                        state.buffers[0].entry_bytes(),
                        crate::topology::ring_slot_bytes(tp)
                    );
                }
            }
            for slots in [1, 32, 256] {
                let extent = bundle_runtime::state::Extent {
                    served: slots,
                    bucket: slots,
                    verify_rows: 0,
                };
                assert_eq!(
                    request_state(tp, &KV_FORMATS[0], Drafter::None).buffers[0].bytes(extent),
                    crate::topology::ring_state_bytes(tp, slots)
                );
            }
        }
    }
}
