//! `deepseek41.v1`'s per-request state, declared for the shared bundle
//! runtime (`bundle_runtime::state`).
//!
//! The pages hold the KV sources' compressed latents and index keys: one
//! entry per `ratio` tokens of each source layer (CSA2,
//! `Geometry::layer_role`).
//! Per slot the state keeps every attention layer's window ring of raw keys,
//! the ratio-2 compressors' incomplete group, and with DSpark loaded its
//! stages' rings and the target rows they are projected from.
//!
//! Rollback is by position everywhere: a ring has `window +
//! RING_HEADROOM` cells and a tail `TAIL_ROWS` rows, so the rows of one
//! verify item past the committed end write cells no later query reads
//! before the next step rewrites them (`Rollback::Truncate`). A prefill
//! writes the cells of its last rows only after its attention read the
//! cells before it, and a cancelled step leaves its slots invalid, as the
//! host retires and restarts them.

use bundle_runtime::state::{
    Anchors, Cancel, Capture, Entries, KvStream, Owner, PagedLayers, RequestState, Rollback, Scope,
    Shadow, StateBuffer,
};

use crate::geometry::Geometry;
use crate::serving::PAGE_TOKENS;

/// Cells a window ring keeps beyond the window: at least the verify rows of
/// one item, so a rejected row never overwrites a cell inside the window of
/// the committed end.
pub const RING_HEADROOM: u64 = 8;
/// Rows of a compressor's incomplete-group history, by position modulo this
/// many: the group's first `ratio - 1` rows and a verify item's rows.
pub const TAIL_ROWS: u64 = 8;
/// The snapshot header: the shared layout
/// (`bundle_runtime::prefix::HeaderLayout::V1`). The Engram n-gram history
/// needs no room: a restore names every prefix token.
pub const HEADER_BYTES: u64 = bundle_runtime::prefix::HeaderLayout::V1.bytes();

/// How the KV pages and rings store their values. Both hold the same
/// values: every stored value is the reference's fake-quantized one, which
/// BF16 holds exactly (D4).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvStorage {
    /// The trained formats: compressed latents E2M1 with an E4M3 scale per
    /// 16, index keys E2M1 with a UE8M0 scale per 32, window keys E4M3 with
    /// a UE8M0 scale per 32.
    Native,
    /// Every value as BF16, the comparison profile.
    Bf16,
}

/// The widths the state is sized by, from the geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Widths {
    /// The attention latent (`head_dim`, key = value).
    pub latent: u64,
    /// One index key (`index_head_dim`).
    pub index: u64,
    /// `sliding_window`.
    pub window: u64,
    /// `hidden_size`.
    pub hidden: u64,
}

impl Widths {
    pub const fn of(g: &Geometry) -> Widths {
        Widths {
            latent: g.shape.attention.head_dim as u64,
            index: g.shape.indexer.dim as u64,
            window: g.shape.attention.window as u64,
            hidden: g.shape.hidden as u64,
        }
    }
}

const BF16: u64 = 2;
const F32: u64 = 4;

/// E2M1 codes of `width` values, two per byte, then one scale byte per
/// `block` values.
const fn fp4(width: u64, block: u64) -> u64 {
    width / 2 + width / block
}

impl KvStorage {
    /// Bytes of one compressed latent.
    pub const fn latent_bytes(self, w: Widths) -> u64 {
        match self {
            Self::Native => fp4(w.latent, 16),
            Self::Bf16 => w.latent * BF16,
        }
    }

    /// Bytes of one index key.
    pub const fn index_bytes(self, w: Widths) -> u64 {
        match self {
            Self::Native => fp4(w.index, 32),
            Self::Bf16 => w.index * BF16,
        }
    }

    /// One window cell's planes, codes then scales: E4M3 codes and one
    /// UE8M0 scale per 32, or BF16 values.
    pub const fn ring_cell(self, w: Widths) -> [u64; 2] {
        match self {
            Self::Native => [w.latent, w.latent / 32],
            Self::Bf16 => [w.latent * BF16, 0],
        }
    }
}

/// Cells of each window ring.
pub const fn ring_cells(w: Widths) -> u64 {
    w.window + RING_HEADROOM
}

/// The paged streams of the KV sources of `ratio`: per `ratio` tokens one
/// compressed latent and one index key.
fn source_streams(storage: KvStorage, w: Widths, ratio: u64) -> Vec<KvStream> {
    vec![
        KvStream {
            name: "latent",
            tokens_per_entry: ratio,
            entry_bytes: storage.latent_bytes(w),
        },
        KvStream {
            name: "index-key",
            tokens_per_entry: ratio,
            entry_bytes: storage.index_bytes(w),
        },
    ]
}

/// A window ring per layer: the cells of the last `window +
/// RING_HEADROOM` positions, a snapshot capturing the window.
fn ring(
    name: &'static str,
    layers: u64,
    storage: KvStorage,
    w: Widths,
    drafter: bool,
) -> StateBuffer {
    let cells = ring_cells(w);
    StateBuffer {
        name,
        layers,
        planes: storage
            .ring_cell(w)
            .into_iter()
            .filter(|&bytes| bytes > 0)
            .map(|bytes| cells * bytes)
            .collect(),
        entries: Entries::Bucket,
        scope: Scope::WindowRing { tokens: cells },
        shadow: Shadow::None,
        capture: if drafter {
            // Rebuilt from the retained rows after a restore.
            Capture::None
        } else {
            Capture::RingWindow { tokens: w.window }
        },
        rollback: Rollback::Truncate,
        cancel: Cancel::Invalidate,
        owner: Owner::Shared,
        drafter,
        anchors: Anchors::None,
    }
}

/// The KV sources of compress ratio `ratio`, in layer order.
pub fn sources_of(g: &Geometry, ratio: u32) -> impl Iterator<Item = u32> + '_ {
    g.shape
        .csa
        .kv_sources
        .iter()
        .copied()
        .filter(move |&layer| g.compress_ratio(layer) == ratio)
}

/// What one instance of `g` keeps per request: the KV sources in the
/// pages, the backbone's rings and the ratio-2 compressors' tails, and with
/// `dspark` the drafter's rings and retained rows.
pub fn request_state(g: &Geometry, storage: KvStorage, dspark: bool) -> RequestState {
    let w = Widths::of(g);
    let mut pages = Vec::new();
    for (name, ratio) in [("csa2.ratio2", 2), ("csa2.ratio1", 1)] {
        let layers = sources_of(g, ratio).count() as u64;
        if layers > 0 {
            pages.push(PagedLayers {
                name,
                layers,
                streams: source_streams(storage, w, u64::from(ratio)),
            });
        }
    }
    let mut buffers = vec![ring(
        "window.ring",
        u64::from(g.layers()),
        storage,
        w,
        false,
    )];
    let tails = sources_of(g, 2).count() as u64;
    if tails > 0 {
        // Per row the compressor's F32 `wkv` and `wgate` outputs, side by
        // side. A decode or verify row that opens a group writes its row; a
        // prefill writes its trailing open group's rows.
        buffers.push(StateBuffer {
            name: "compressor.tail",
            layers: tails,
            planes: vec![TAIL_ROWS * 2 * w.latent * F32],
            entries: Entries::Bucket,
            scope: Scope::Rows { rows: TAIL_ROWS },
            shadow: Shadow::None,
            capture: Capture::LastRows { rows: 1 },
            rollback: Rollback::Truncate,
            cancel: Cancel::Invalidate,
            owner: Owner::Shared,
            drafter: false,
            anchors: Anchors::None,
        });
    }
    if dspark && g.dspark_stages() > 0 {
        buffers.push(ring(
            "dspark.ring",
            u64::from(g.dspark_stages()),
            storage,
            w,
            true,
        ));
        // The DSpark target layers' attention inputs (the mean of the
        // streams, BF16) of the last window of rows, which the drafter
        // projects into its rings.
        let targets = g.shape.dspark.targets.len() as u64;
        let rows = ring_cells(w);
        buffers.push(StateBuffer {
            name: "dspark.features",
            layers: 1,
            planes: vec![rows * targets * w.hidden * BF16],
            entries: Entries::Bucket,
            scope: Scope::Rows { rows },
            shadow: Shadow::None,
            capture: Capture::LastRows { rows: w.window },
            rollback: Rollback::Truncate,
            cancel: Cancel::Invalidate,
            owner: Owner::Shared,
            drafter: true,
            anchors: Anchors::None,
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
    use crate::geometry::FLASH;
    use bundle_runtime::state::Extent;

    /// The model card's 890 bytes of global KV per token natively, 3,200
    /// in BF16 (plan §2.9, T6).
    #[test]
    fn pages_hold_the_cards_bytes_per_token() {
        for (storage, per_token) in [(KvStorage::Native, 890), (KvStorage::Bf16, 3200)] {
            for dspark in [false, true] {
                let state = request_state(&FLASH, storage, dspark);
                assert_eq!(state.validate(), Ok(()), "{storage:?}");
                assert_eq!(state.page_bytes(), per_token * PAGE_TOKENS, "{storage:?}");
                let extent = Extent {
                    served: 4,
                    bucket: 4,
                    verify_rows: if dspark { 6 } else { 0 },
                };
                assert_eq!(state.validate_extent(extent), Ok(()));
            }
        }
    }

    /// 43 rings of 128 positions are 2.9 MB per sequence natively (40 of
    /// them without DSpark); the headroom adds 8 cells to each.
    #[test]
    fn rings_hold_the_window_and_the_headroom() {
        let state = request_state(&FLASH, KvStorage::Native, true);
        let rings: u64 = state
            .buffers
            .iter()
            .filter(|b| b.name.ends_with(".ring"))
            .map(|b| b.entry_bytes())
            .sum();
        assert_eq!(rings, 43 * 136 * (512 + 16));
        assert_eq!(43 * 128 * (512 + 16), 2_906_112);
        let tail = &state.buffers[1];
        assert_eq!((tail.name, tail.layers), ("compressor.tail", 3));
        assert_eq!(tail.entry_bytes(), 3 * 8 * 2 * 512 * 4);
        assert_eq!(sources_of(&FLASH, 1).collect::<Vec<_>>(), [20]);
    }
}
