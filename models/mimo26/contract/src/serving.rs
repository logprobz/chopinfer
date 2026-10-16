//! Fixed geometry shared by the package declaration and native module.

use crate::geometry::V26_FLASH as GEOMETRY;

use bundle_runtime::kv::KvFormat;
use kernel_core::contract::DType;

/// The package ID. One package serves every supported rank count.
pub const BUNDLE_ID: &str = "mimo.v26.flash.v2";
pub const PROFILE: &str = "mimo-v26-flash";
pub const TOPOLOGY_ID: &str = "mimo-v26-flash-tp4";
/// TP4, the topology whose constants this module names. The per-rank
/// shapes of every served topology are in `crate::topology`.
pub const TOPOLOGY: bundle_runtime::config::Topology<'static> = bundle_runtime::config::Topology {
    id: TOPOLOGY_ID,
    ranks: bundle_runtime::geometry::Ranks::TP4,
};
/// The topologies this module serves: TP2 and TP4.
pub const TOPOLOGIES: [bundle_runtime::config::Topology<'static>; 2] = [
    bundle_runtime::config::Topology {
        id: "mimo-v26-flash-tp2",
        ranks: bundle_runtime::geometry::Ranks::TP2,
    },
    TOPOLOGY,
];
/// TP4's rank count.
pub const RANKS: u32 = TOPOLOGY.ranks.get();
/// Hidden width, the row width of the fused all-reduce and RMSNorm.
pub const HIDDEN: u32 = GEOMETRY.hidden();
pub const CONTEXT_TOKENS: u64 = GEOMETRY.context_tokens() as u64;
/// Policy: bound prefill workspace and prepared row capacity.
pub const PREFILL_TOKENS: u32 = 8192;
/// Maximum logits rows in a scored prefill phase.
/// Policy: bound logits workspace.
pub const SCORED_ROWS: u32 = 512;
/// The package maximum: most sequences an instance serves and most items in
/// one step (docs/serving-profiles.md). An instance serves a
/// runtime `Items` count up to it, from the host's `--max-sequences`.
/// The shared runtime defines the maximum item count.
pub const DECODE_BATCH: u32 = bundle_runtime::geometry::MAX_ITEMS;
/// The largest graph bucket of the decode kernels: the cluster decode
/// GEMMs, the decode route and experts, the decode heads and the push
/// all-reduce with the folded MoE combine. Larger decode and verify buckets
/// run the prefill-shaped entries at the bucket's rows.
/// Policy: match prepared decode buckets.
pub const DECODE_ENTRY_ROWS: u32 = 32;
/// An instance serving 32 sequences: what the package's memory minimum and
/// the launch journals cover.
pub const ITEMS_32: bundle_runtime::geometry::Items =
    match bundle_runtime::geometry::Items::new(DECODE_ENTRY_ROWS) {
        Some(items) => items,
        None => panic!("32 served sequences"),
    };
/// Most sequences an instance with the MTP drafter serves: the drafter
/// drafts and catches up at the decode kernels' capacities.
/// Policy: bound drafter workspace.
pub const DRAFT_ITEMS: u32 = 32;
/// Most rows one verify item may carry: its bonus input and up to 7 drafts.
/// Policy: bound speculative verification.
pub const MAX_VERIFY_ROWS: u32 = 8;
/// Verify buckets of at most this many rows run exact rows: every verify
/// row's arithmetic is that of the AR row at its position (a one-row decode
/// step), so greedy speculation serves AR's tokens and logprobs whatever
/// draft lengths the adaptive policy picks. Every C1 verify step is exact.
/// Policy: select exact arithmetic for C1 verification.
pub const EXACT_VERIFY_ROWS: u32 = MAX_VERIFY_ROWS;
/// Most decode-phase rows of one verify step (`MAX_VERIFY_TOKENS` of the
/// attention kernels). Verify buckets above `DECODE_ENTRY_ROWS` run the
/// prefill-shaped entries at the bucket's rows.
pub const VERIFY_TOKENS: u32 = DECODE_ENTRY_ROWS * MAX_VERIFY_ROWS;
/// Set to `0` to run verify buckets of up to [`EXACT_VERIFY_ROWS`] rows on
/// the verify kernels instead of exact rows (an A/B baseline whose rows
/// differ from AR's in the last bits).
pub const VERIFY_EXACT: &str = "CHOPINFER_MIMO_VERIFY_EXACT";

/// Whether verify buckets of up to [`EXACT_VERIFY_ROWS`] rows run exact rows
/// ([`VERIFY_EXACT`], on unless `0`), read once per process.
pub fn verify_exact_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var(VERIFY_EXACT).as_deref() != Ok("0"))
}

/// Diagnostic only: set to `rows` to run the attention of verify buckets
/// wider than [`EXACT_VERIFY_ROWS`] on the decode variant's per-row exact
/// rows entries at the bucket's token rows, so each row's attention bits are
/// the one-row decode step's. Every other stage keeps the bucket's plans,
/// which isolates verify attention arithmetic on real activations
/// (DFLASH-VERIFY-EXCESS, 2026-10-06). MTP draft batches of 16 and 32 rows
/// switch too. The attention workspace then holds every verify token row,
/// about 1 GiB more per TP4 rank, which the KV pool gives up.
pub const VERIFY_ATTENTION: &str = "CHOPINFER_MIMO_VERIFY_ATTENTION";

/// Whether wide verify buckets run attention as per-row exact rows
/// ([`VERIFY_ATTENTION`] is `rows`), read once per process.
pub fn verify_attention_rows() -> bool {
    static ROWS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| std::env::var(VERIFY_ATTENTION).as_deref() == Ok("rows"))
}

/// The largest decode and verify bucket whose experts run the decode route
/// and the ring decode entries. Wider buckets route their experts as
/// prefill does and run the prefill expert entries. Both reduce each routed
/// row on its own, so a row's expert sums ignore the other rows. At 32 rows
/// the ring's per-warp slots leave a one-deep ring and the prefill entries
/// are faster (MIMO-B64-NUMERICS, 2026-10-05).
/// Policy: select row-local expert entries.
pub const DECODE_EXPERT_ROWS: u32 = 16;

/// Whether a `phase` step of bucket `capacity` routes its experts as
/// prefill does (router GEMM and top-8, then the tiled permutation) and
/// runs the prefill expert entries: prefill, and decode and verify
/// buckets above [`DECODE_EXPERT_ROWS`].
pub fn prefill_route(phase: kernel_core::precision::ExecutionPhase, capacity: u32) -> bool {
    phase == kernel_core::precision::ExecutionPhase::Prefill || capacity > DECODE_EXPERT_ROWS
}

/// Verify rows per item the module is prepared for, for the manifest.
/// Policy: select prepared verification capacities.
pub const VERIFY_ROW_BUCKETS: [u32; 3] = [2, 4, 8];
/// The module can draft with packaged MTP layer 0 when speculation is enabled.
pub const BUNDLE_DRAFTS: bool = true;
/// Policy: match the paged KV engine.
pub const PAGE_TOKENS: u64 = 64;
pub const GA_LAYERS: u64 = GEOMETRY.global_layers() as u64;
pub const SWA_LAYERS: u64 = GEOMETRY.layers() as u64 - GA_LAYERS;
pub const RING_TOKENS: u64 = GEOMETRY.window() as u64;
pub const GA_HEADS_PER_RANK: u64 = crate::tp::GA_KV_HEADS / RANKS as u64;
pub const SWA_HEADS_PER_RANK: u64 = crate::tp::SWA_KV_HEADS / RANKS as u64;
pub const QK_DIM: u64 = crate::tp::QK_DIM;
pub const V_DIM: u64 = crate::tp::V_DIM;
pub const BF16_BYTES: u64 = 2;
pub const VOCAB_PER_RANK: u32 = GEOMETRY.vocab() / RANKS;
const _: () = assert!(GEOMETRY.vocab() % RANKS == 0);
pub const PREFILL_LOGIT_BYTES: u64 = 4;
pub const DECODE_LOGIT_BYTES: u64 = 2;

pub const GA_PAGE_BYTES_BF16: u64 =
    GA_LAYERS * GA_HEADS_PER_RANK * PAGE_TOKENS * (QK_DIM + V_DIM) * BF16_BYTES;
pub const GA_PAGE_BYTES_FP8: u64 = GA_LAYERS * GA_HEADS_PER_RANK * PAGE_TOKENS * (QK_DIM + V_DIM);
/// Global-attention layouts. Sliding-window rings stay BF16.
pub const KV_LAYOUT_BF16: &[u8] = b"mimo.ga-paged-bf16.v1";
pub const KV_LAYOUT_FP8: &[u8] = crate::kv_calibration::LEGACY_LAYOUT.as_bytes();
/// The shipped MOPD/RL table, retained for historical layout consumers.
pub use crate::kv_calibration::LEGACY_SCALES as GA_FP8_KV_SCALES;
pub struct Fp8KvCalibration {
    pub geometry: &'static str,
    pub layout_id: &'static [u8],
    pub scales: &'static [[f32; 2]],
}
include!(concat!(env!("OUT_DIR"), "/kv-calibration.rs"));
pub const KV_FORMATS: [KvFormat; 2] = [
    KvFormat {
        layout_id: KV_LAYOUT_BF16,
        page_bytes: GA_PAGE_BYTES_BF16,
        element: DType::Bf16,
        scales: &[],
    },
    KvFormat {
        layout_id: KV_LAYOUT_FP8,
        page_bytes: GA_PAGE_BYTES_FP8,
        element: DType::Fp8E4M3,
        scales: &GA_FP8_KV_SCALES,
    },
];
/// Compiled layouts for this geometry. Preparation selects the authenticated
/// checkpoint's table from these formats. Create accepts any compiled table
/// of this geometry; the package manifest and host selection bind the checkpoint.
pub fn kv_formats(tp: crate::Tp) -> Vec<KvFormat> {
    let mut formats = vec![KvFormat {
        page_bytes: crate::topology::ga_page_bytes(tp, false),
        ..KV_FORMATS[0]
    }];
    for table in COMPILED_FP8_KV {
        if table.geometry == tp.geometry.id {
            formats.push(KvFormat {
                layout_id: table.layout_id,
                page_bytes: crate::topology::ga_page_bytes(tp, true),
                element: DType::Fp8E4M3,
                scales: table.scales,
            });
        }
    }
    formats
}

pub const RING_SLOT_BYTES: u64 =
    SWA_LAYERS * SWA_HEADS_PER_RANK * RING_TOKENS * (QK_DIM + V_DIM) * BF16_BYTES;
/// TP4's ring state at 32 ring slots (an instance serving 32 sequences).
pub const RING_STATE_BYTES_PER_RANK: u64 = 2 * DECODE_ENTRY_ROWS as u64 * RING_SLOT_BYTES;

/// MiMo's geometry for the shared bundle runtime.
pub struct Mimo;

/// Model dimensions passed to shared runtime mechanics.
pub const MODEL_LIMITS: bundle_runtime::ModelLimits =
    match bundle_runtime::ModelLimits::new(VOCAB_PER_RANK * RANKS, CONTEXT_TOKENS, PAGE_TOKENS) {
        Ok(limits) => limits,
        Err(_) => panic!("invalid model limits"),
    };

/// Entries of one item's device block-table row.
pub const MAX_PAGES: u32 = MODEL_LIMITS.max_pages();

impl bundle_runtime::BundleGeometry for Mimo {
    const PAGE_TOKENS: u64 = PAGE_TOKENS;
    const PREFILL_TOKENS: u32 = PREFILL_TOKENS;
    const DECODE_BATCH: u32 = DECODE_BATCH;
    const MAX_VERIFY_ROWS: u32 = MAX_VERIFY_ROWS;
    const VERIFY_TOKENS: u32 = VERIFY_TOKENS;
    const TOKEN_GRANULE: u32 = 256;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_formats_use_the_rank_geometry_and_keep_calibration_scoped() {
        for geometry in crate::geometry::GEOMETRIES {
            for ranks in crate::Tp::SUPPORTED {
                let tp = crate::Tp::of_geometry(geometry, ranks.ranks()).unwrap();
                for format in kv_formats(tp) {
                    assert!(format.valid(geometry.global_layers() as u64));
                    assert_eq!(
                        format.page_bytes,
                        crate::topology::ga_page_bytes(tp, format.fp8())
                    );
                }
            }
        }
        let mut geometry = GEOMETRY;
        geometry.id = "kv-test";
        let mut tp = crate::Tp::TP2;
        tp.geometry = Box::leak(Box::new(geometry));
        assert_eq!(kv_formats(tp).len(), 1);
        assert!(!kv_formats(tp)[0].fp8());
    }

    #[test]
    fn resource_geometry_matches_mimo() {
        assert_eq!(GA_PAGE_BYTES_BF16, 368_640);
        assert_eq!(GA_PAGE_BYTES_FP8, 184_320);
        assert!(KV_FORMATS.iter().all(|format| format.valid(GA_LAYERS)));
        assert_eq!(RING_SLOT_BYTES, 6_389_760);
        assert_eq!(RING_STATE_BYTES_PER_RANK, 408_944_640);
        assert_eq!(VOCAB_PER_RANK * RANKS, 152_576);
    }
}
