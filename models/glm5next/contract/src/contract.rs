//! The `glm5next.v1` serving contract: the package and topology identities,
//! the per-rank shape at TP4, the latent KV page formats, the
//! context-parallel sizes and the step limits the bundle and the package
//! share.
//!
//! Every sparse (latent attention) layer keeps one plane of latent rows and
//! one plane of pooled indexer keys per 64-token page; the KDA layers keep no
//! pages, only per-slot state (`crate::state`).

use crate::geometry::{FLASH, Geometry, LayerKind};
use crate::kernels::KernelProfile;
use bundle_runtime::config::Topology;
use bundle_runtime::geometry::Ranks;
use bundle_runtime::kv::KvFormat;
use kernel_core::contract::{DType, RecurrentPhase};
use kernel_core::gated_pooling::GatedGroupPooling;
use kernel_core::latent_attention::LatentKvFormat;
use kernel_core::per_channel_decay_delta_rule::{
    DELTA_RULE_CHUNK, DecayGate, PerChannelDecayDeltaRuleQuery, QkvConvs,
};
use kernel_core::rmsnorm::{RmsNormNumerics, RmsNormRequirement};

/// The package ID. One package serves the TP4 topology at every declared
/// context-parallel size.
pub const BUNDLE_ID: &str = "glm5next.v1";
pub const TOPOLOGY_ID: &str = "glm5next-tp4";
pub const TOPOLOGY: Topology<'static> = Topology {
    id: TOPOLOGY_ID,
    ranks: Ranks::TP4,
};
/// The capabilities every glm5next package requires. Each one whose host
/// path calls into the bundle is listed only once the native module exports
/// that callback.
pub const CAPABILITIES: [&str; 16] = [
    runtime_bundle_abi::SERVING_KV_CAPABILITY_V1,
    runtime_bundle_abi::RECURRENT_STATE_CAPABILITY_V1,
    // Create allocates the KDA pools, their verify checkpoints and the
    // anchor spares before the serving KV pool, so the host reserves no
    // state copies of its own.
    runtime_bundle_abi::RECURRENT_STATE_PREALLOCATED_CAPABILITY_V1,
    runtime_bundle_abi::REQUEST_RETIREMENT_CAPABILITY_V1,
    // Verify steps of host drafts (`commit_accepted`); the MTP drafter
    // joins with its own step.
    runtime_bundle_abi::SPECULATIVE_CAPABILITY_V1,
    runtime_bundle_abi::PROMPT_ROWS_CAPABILITY_V2,
    // Prefix snapshots of the KDA state and indexer tail, restored into the
    // host's state slot.
    runtime_bundle_abi::PREFIX_CACHE_CAPABILITY_V1,
    runtime_bundle_abi::PREFIX_CACHE_STATE_SLOT_CAPABILITY_V1,
    // Preemption swaps through the host tier: the host copies the pages,
    // and the prefix snapshot and state-slot restore above carry the
    // request state. No callback of its own.
    runtime_bundle_abi::KV_OFFLOAD_CAPABILITY_V1,
    runtime_bundle_abi::SERVED_ITEMS_CAPABILITY_V1,
    runtime_bundle_abi::SERVED_CONTEXT_CAPABILITY_V1,
    runtime_bundle_abi::STATE_PER_ITEM_CAPABILITY_V1,
    runtime_bundle_abi::RECURRENT_STATE_IN_PLACE_CAPABILITY_V1,
    runtime_bundle_abi::weight_plan::WEIGHT_PLAN_CAPABILITY_V1,
    runtime_bundle::model_options::CAPABILITY,
    // The module exports the declared chat frontend.
    runtime_bundle_abi::frontend::FRONTEND_CAPABILITY_V1,
];
/// Most rows the head scores in a decode or verify phase: the largest
/// verify bucket, which holds every decode bucket. The package declares it
/// as its speculation's verify tokens.
pub fn head_rows() -> u32 {
    bundle_runtime::graphs::verify_tokens::<Glm>(bundle_runtime::geometry::Items::package::<Glm>())
}

/// Most logits rows a prefill phase scores, the package's
/// `maximum_scored_rows`: the host ends a scored chunk on the prefill
/// granule, so the limit is a multiple of it; one granule holds the last
/// rows of every served item.
pub const SCORED_ROWS: u32 = 64;
// Scored prefill chunks end on the granule; a prefill phase's logits rows
// include every served item's last row.
const _: () = assert!(SCORED_ROWS >= PREFILL_GRANULE && SCORED_ROWS % PREFILL_GRANULE == 0);
const _: () = assert!(SCORED_ROWS >= DECODE_BATCH);

/// The rows the head's logits buffers hold: a prefill's scored rows or a
/// decode or verify phase's.
pub fn logits_rows() -> u32 {
    SCORED_ROWS.max(head_rows())
}
/// Context-parallel group sizes the topology serves. Latent KV is shared by
/// every attention head, so it splits over any size dividing the ranks; the
/// manifest lists the sizes above 1 (`context_parallel_sizes`).
pub const CONTEXT_PARALLEL_SIZES: [u32; 3] = [1, 2, 4];

/// The manifest topology with its context-parallel sizes above 1.
pub fn bundle_topology() -> Result<runtime_bundle::BundleTopologyV1, runtime_bundle::BundleError> {
    runtime_bundle::BundleTopologyV1::new(
        runtime_bundle::BundleOpaqueId::new(TOPOLOGY_ID)?,
        TOPOLOGY.ranks.get(),
    )?
    .with_context_parallel_sizes(
        CONTEXT_PARALLEL_SIZES
            .into_iter()
            .filter(|&size| size > 1)
            .collect(),
    )
}

/// Tokens per serving KV page. A page holds whole indexer pools (a pool is
/// at most 16 tokens), and under context parallelism one stripe covers
/// `size * PAGE_TOKENS` positions.
pub const PAGE_TOKENS: u64 = 64;
/// Positions a prefill chunk may end at before the prompt end, and prefix
/// anchors sit at: the KDA chunk, whose chunked prefill is bit-for-bit chunk
/// invariant only at 64-aligned absolute splits (docs/numerics.md, "Prefill
/// chunk invariance"). The manifest's `batch_limits.prefill_granule`.
pub const PREFILL_GRANULE: u32 = DELTA_RULE_CHUNK;
/// Prompt-ending prefill items per step that keep their state at the last
/// `PREFILL_GRANULE` multiple inside their rows, for a prefix anchor there
/// (the manifest's `batch_limits.anchor_items`), so a prompt that does not
/// end on the granule runs as one chunk instead of ending a chunk at its
/// anchor and running its last rows as a step of their own. Each takes a
/// spare KDA entry per layer (37 MB per rank) and small indexer-tail and
/// retained-row spares. The KDA and tail spares come from the memory the
/// KV pool leaves; the drafter's retained rows are allocated before the
/// pool is sized, so with the drafter their spares (48 KB each) shrink the
/// KV budget, by one page at most. Sixteen cover a burst of sixteen new
/// prompts in one step (at eight, C16 bursts ran 6-7 items' last rows as a
/// tail step, +87-94 ms to their first token); the scheduler ends the chunk
/// of any further such item at its anchor, and that item's last rows run in
/// the next step. Prompt chunks are invariant at 64-aligned splits, so the
/// count moves no bit.
/// Policy: spare state capacity bounds simultaneous prefix anchors.
pub const ANCHOR_ITEMS: u32 = 16;
/// Policy: prefill launches cap scheduler chunks (the profiles'
/// `max_batch_tokens`).
pub const PREFILL_TOKENS: u32 = 8192;
/// The package maximum of served sequences and items per step: the
/// profiles' 32 `max_sequences`. KDA committed state and the shared
/// verify-row checkpoint pool are allocated before the KV pool.
pub const DECODE_BATCH: u32 = 32;
/// Verify rows one item may carry: the bonus token and up to five drafts
/// (the MTP drafter is benchmarked at three and five).
pub const MAX_VERIFY_ROWS: u32 = 6;
pub const MAX_DRAFT_TOKENS: u32 = MAX_VERIFY_ROWS - 1;
/// Drafts the MTP drafter makes per verify item by default (the tp4-mtp
/// profile's `--speculative-tokens`). The package declares
/// `MAX_DRAFT_TOKENS`, so the experimental tp4-mtp-k5 profile serves five
/// from the same package; state and graphs follow the served count.
pub const MTP_DRAFTS: u32 = 3;
/// Most items that draft in one step: every served item.
pub const DRAFT_ITEMS: u32 = DECODE_BATCH;
/// Rows per verify item the host may draft for: every count up to
/// `MAX_VERIFY_ROWS`, which the verify lowering serves alike.
pub const VERIFY_ROW_BUCKETS: [u32; 5] = [2, 3, 4, 5, 6];
/// Policy: token buffers align to the runtime allocation granule.
pub const TOKEN_GRANULE: u32 = 256;
pub const BF16_BYTES: u64 = 2;

/// Package-facing identifiers of the serving KV layouts: latent rows in FP8
/// E4M3 with per-128 F32 scales (the default) or BF16, each without and with
/// the MTP drafter's sparse layer after the main ones. The drafter's layer
/// uses the main layers' format, so one latent attention instance serves
/// both.
pub const KV_LAYOUT_FP8: &[u8] = b"glm5next.latent-paged-fp8.v1";
pub const KV_LAYOUT_FP8_MTP: &[u8] = b"glm5next.latent-paged-fp8-mtp.v1";
pub const KV_LAYOUT_BF16: &[u8] = b"glm5next.latent-paged-bf16.v1";
pub const KV_LAYOUT_BF16_MTP: &[u8] = b"glm5next.latent-paged-bf16-mtp.v1";

/// The request state layouts (`recurrent_state_layouts`, under
/// `chop.recurrent-state.v1`), one committed copy each per slot and rank.
/// The bundle allocates every slot's copy, the verify checkpoints and the
/// anchor spares before it acquires the KV pool
/// (`chop.recurrent-state.preallocated.v1`), so the host reserves none.
pub const KDA_CONV_STATE_ID: &str = "glm5next.kda-conv.v1";
pub const KDA_STATE_ID: &str = "glm5next.kda-state.v1";
/// The per-item state layout (`state_layout` under
/// `chop.state-per-item.v1`): every sparse layer's indexer tail with one
/// checkpoint per verify row, which the host allocates per state slot.
pub const INDEXER_TAIL_STATE_ID: &str = "glm5next.indexer-tail.v1";

/// The package geometry for the shared bundle runtime.
pub struct Glm;

impl bundle_runtime::BundleGeometry for Glm {
    const PAGE_TOKENS: u64 = PAGE_TOKENS;
    const PREFILL_TOKENS: u32 = PREFILL_TOKENS;
    const DECODE_BATCH: u32 = DECODE_BATCH;
    const MAX_VERIFY_ROWS: u32 = MAX_VERIFY_ROWS;
    const TOKEN_GRANULE: u32 = TOKEN_GRANULE;
}

/// The per-rank shape of a geometry at a served rank count: the kernel
/// profile's per-rank sizes plus the page and state sizes derived from
/// them. Rank views: KDA and latent attention at 16 heads per rank, the
/// latent, query-down, indexer and router replicated, a quarter of every
/// routed expert's intermediate per rank, the dense and shared MLPs split by
/// column and the embedding and head by vocabulary row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    pub geometry: &'static Geometry,
    pub ranks: Ranks,
    pub profile: KernelProfile,
}

impl Shape {
    pub const TP4: Self = Self::new(&FLASH, Ranks::TP4);

    /// The shape of a served rank count; other counts panic at compile time
    /// in a const and are refused by [`crate::fit::check`] at run time.
    pub const fn new(geometry: &'static Geometry, ranks: Ranks) -> Self {
        assert!(ranks.get() == 4, "glm5next serves TP4 only");
        Self {
            geometry,
            ranks,
            profile: KernelProfile::new(geometry, ranks.get()),
        }
    }

    pub const fn ranks(self) -> u32 {
        self.ranks.get()
    }
    pub const fn layers(self) -> u32 {
        self.geometry.layers()
    }
    pub const fn linear_layers(self) -> u64 {
        self.geometry.count_layers(LayerKind::Linear) as u64
    }
    pub const fn sparse_layers(self) -> u64 {
        self.geometry.count_layers(LayerKind::Sparse) as u64
    }
    pub const fn mtp_layers(self) -> u64 {
        self.geometry.mtp_layers() as u64
    }
    /// The sparse layers a page holds: the main ones, then the drafter's.
    pub const fn paged_layers(self, mtp: bool) -> u64 {
        self.sparse_layers() + if mtp { self.mtp_layers() } else { 0 }
    }
    /// A sparse layer's index among the sparse layers, the drafter's layers
    /// (numbered from `layers()`) after the main ones.
    pub const fn sparse_ordinal(self, layer: u32) -> Option<u64> {
        if layer >= self.layers() {
            return Some(self.sparse_layers() + (layer - self.layers()) as u64);
        }
        if !matches!(self.geometry.layer_kind(layer), LayerKind::Sparse) {
            return None;
        }
        let mut ordinal = 0;
        let mut i = 0;
        while i < layer {
            if matches!(self.geometry.layer_kind(i), LayerKind::Sparse) {
                ordinal += 1;
            }
            i += 1;
        }
        Some(ordinal)
    }

    /// The indexer's gated pooling of its key width.
    pub const fn pooling(self) -> GatedGroupPooling {
        GatedGroupPooling {
            group: self.profile.index_pool,
            width: self.profile.indexer_dim,
            position_dtype: DType::Bf16,
        }
    }

    /// The keys one latent attention row reads at most: the indexer's
    /// compiled top-k plus the incomplete pool tail.
    pub const fn latent_keys(self) -> u32 {
        self.profile.index_topk_max + self.profile.index_pool - 1
    }

    /// The widths the program normalizes: the hidden row, the query-down
    /// output and the latent.
    pub const fn rmsnorm_widths(self) -> [u32; 3] {
        [
            self.profile.hidden,
            self.profile.q_rank,
            self.profile.kv_rank,
        ]
    }

    /// The RMSNorm of a row of `width`: HF's BF16 norm of an F32
    /// intermediate (`RmsNormKind::HfBf16Intermediate`) over the geometry's
    /// eps. Hidden-width rows (every input norm, the final and the
    /// drafter's norms) sum their squares in the fixed 256-lane tree
    /// ([`RmsNormNumerics::HF_TREE_256`]), the same order in every phase and
    /// kernel that normalizes them (the row engine, the fused mixer norm and
    /// the comm epilogue); the query-down and latent rows keep the ordered
    /// sum.
    pub fn rmsnorm_requirement(self, width: u32) -> RmsNormRequirement {
        let numerics = if width == self.profile.hidden {
            RmsNormNumerics::HF_TREE_256
        } else {
            RmsNormNumerics::HF_ORDERED
        };
        RmsNormRequirement::bf16(
            width,
            self.geometry.eps(),
            kernel_core::contract::RmsNormKind::HfBf16Intermediate,
            false,
            numerics,
        )
    }

    /// One sparse layer's latent rows in one page.
    pub fn latent_page_bytes(self, kv: LatentKvFormat) -> u64 {
        kv.page_bytes(self.profile.kv_rank, PAGE_TOKENS as u32)
    }
    /// One sparse layer's pooled keys in one page, BF16.
    pub const fn pooled_page_bytes(self) -> u64 {
        PAGE_TOKENS / self.profile.index_pool as u64 * self.profile.indexer_dim as u64 * BF16_BYTES
    }
    pub fn page_layer_bytes(self, kv: LatentKvFormat) -> u64 {
        self.latent_page_bytes(kv) + self.pooled_page_bytes()
    }

    /// The serving KV layouts, FP8 first (the package preference).
    pub fn kv_formats(self) -> [KvFormat; 4] {
        let format = |layout_id, kv: LatentKvFormat, mtp| KvFormat {
            layout_id,
            page_bytes: self.paged_layers(mtp) * self.page_layer_bytes(kv),
            element: match kv {
                LatentKvFormat::Bf16 => DType::Bf16,
                LatentKvFormat::Fp8E4M3Block128 => DType::Fp8E4M3,
                LatentKvFormat::Fp8E4M3Ue8M0Block32 | LatentKvFormat::Fp4E2M1E4M3Block16 => {
                    unreachable!("native latent formats are not declared by this model")
                }
            },
            // The FP8 latent carries its own block scales in the page.
            scales: &[],
        };
        [
            format(KV_LAYOUT_FP8, LatentKvFormat::Fp8E4M3Block128, false),
            format(KV_LAYOUT_FP8_MTP, LatentKvFormat::Fp8E4M3Block128, true),
            format(KV_LAYOUT_BF16, LatentKvFormat::Bf16, false),
            format(KV_LAYOUT_BF16_MTP, LatentKvFormat::Bf16, true),
        ]
    }

    /// The KDA convolution of one layer at `phase`: separate q, k and v
    /// weights over this rank's heads.
    pub const fn kda_conv(
        self,
        phase: RecurrentPhase,
        capacity: u32,
    ) -> PerChannelDecayDeltaRuleQuery {
        PerChannelDecayDeltaRuleQuery::ShortConv {
            heads: self.profile.linear_heads,
            head_dim: self.profile.linear_head_dim,
            kernel: self.profile.conv_taps,
            convs: QkvConvs::Separate,
            phase,
            capacity,
        }
    }

    /// The KDA recurrence of one layer at `phase`, with the bounded sigmoid
    /// gate at the geometry's lower bound.
    pub const fn kda_delta_rule(
        self,
        phase: RecurrentPhase,
        capacity: u32,
    ) -> PerChannelDecayDeltaRuleQuery {
        PerChannelDecayDeltaRuleQuery::DeltaRule {
            heads: self.profile.linear_heads,
            head_dim: self.profile.linear_head_dim,
            gate: DecayGate::BoundedSigmoid {
                lower_bound_bits: (self.geometry.shape.linear.gate_lower_bound as f32).to_bits(),
            },
            phase,
            capacity,
        }
    }

    /// One layer's KDA convolution history for one slot, BF16.
    pub fn kda_conv_bytes(self) -> u64 {
        self.kda_conv(RecurrentPhase::Decode, 1).state_bytes()
    }
    /// One layer's KDA state for one slot, F32.
    pub fn kda_state_bytes(self) -> u64 {
        self.kda_delta_rule(RecurrentPhase::Decode, 1).state_bytes()
    }
    /// One sparse layer's indexer tail for one slot: the normalized keys,
    /// then the gate logits, of the incomplete pool, BF16. The indexer is
    /// replicated, so every rank holds it.
    pub fn indexer_tail_bytes(self) -> u64 {
        self.pooling().tail_state_elements() * BF16_BYTES
    }

    /// The request state layouts: one committed copy per slot on one rank,
    /// over every KDA layer.
    pub fn recurrent_state_layouts(self) -> [(&'static str, u64); 2] {
        [
            (
                KDA_CONV_STATE_ID,
                self.linear_layers() * self.kda_conv_bytes(),
            ),
            (KDA_STATE_ID, self.linear_layers() * self.kda_state_bytes()),
        ]
    }

    /// Bytes of one state slot of the per-item indexer tail on one rank:
    /// the committed tail and `MAX_VERIFY_ROWS` checkpoints of every sparse
    /// layer, padded to 256 bytes.
    pub fn indexer_tail_slot_bytes(self, mtp: bool) -> u64 {
        (self.paged_layers(mtp) * self.indexer_tail_bytes() * (u64::from(MAX_VERIFY_ROWS) + 1))
            .next_multiple_of(256)
    }

    /// The unpooled upper bound for one sequence at `rows` verify rows:
    /// its committed KDA copy, a checkpoint per row and the indexer tail.
    /// `request_state` bounds the instance's shared checkpoint pool by its
    /// total verify-row budget, so this bound is not multiplied by slots.
    pub fn state_slot_bytes(self, mtp: bool, rows: u32) -> u64 {
        let copy: u64 = self
            .recurrent_state_layouts()
            .iter()
            .map(|(_, bytes)| bytes)
            .sum();
        copy * (u64::from(rows) + 1) + self.indexer_tail_slot_bytes(mtp)
    }

    pub const fn model_limits(self) -> bundle_runtime::ModelLimits {
        match bundle_runtime::ModelLimits::new(
            self.geometry.vocab(),
            self.geometry.context_tokens(),
            PAGE_TOKENS,
        ) {
            Ok(limits) => limits,
            Err(_) => panic!("invalid model limits"),
        }
    }
}

/// Whether `format`'s pages hold the MTP drafter's layer.
pub fn has_mtp_layer(format: &KvFormat) -> bool {
    format.layout_id == KV_LAYOUT_FP8_MTP || format.layout_id == KV_LAYOUT_BF16_MTP
}

/// The latent format of a serving layout.
pub fn latent_format(format: &KvFormat) -> LatentKvFormat {
    if format.fp8() {
        LatentKvFormat::Fp8E4M3Block128
    } else {
        LatentKvFormat::Bf16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tp4_rank_views() {
        let shape = Shape::TP4;
        let p = shape.profile;
        assert_eq!(
            (p.latent_heads, p.linear_heads, p.local_experts),
            (16, 16, 288)
        );
        assert_eq!(p.expert_intermediate, 512);
        assert_eq!((p.dense_intermediate, p.shared_intermediate), (3072, 512));
        assert_eq!(p.vocab_rows * 4, FLASH.vocab());
        assert_eq!((shape.linear_layers(), shape.sparse_layers()), (34, 11));
        assert_eq!(shape.sparse_ordinal(3), Some(0));
        assert_eq!(shape.sparse_ordinal(43), Some(10));
        assert_eq!(shape.sparse_ordinal(45), Some(11));
        assert_eq!(shape.sparse_ordinal(4), None);
    }

    #[test]
    fn pages_hold_latent_rows_and_pooled_keys() {
        let shape = Shape::TP4;
        // 64 tokens of 512 E4M3 codes and four F32 scales, then 16 pooled
        // keys of 128 BF16.
        assert_eq!(
            shape.page_layer_bytes(LatentKvFormat::Fp8E4M3Block128),
            64 * 528 + 4096
        );
        assert_eq!(
            shape.page_layer_bytes(LatentKvFormat::Bf16),
            64 * 1024 + 4096
        );
        let formats = shape.kv_formats();
        assert_eq!(
            formats.map(|f| f.page_bytes),
            [11 * 37_888, 12 * 37_888, 11 * 69_632, 12 * 69_632]
        );
        assert!(formats[0].fp8() && !formats[2].fp8());
        assert_eq!(
            formats.map(|f| has_mtp_layer(&f)),
            [false, true, false, true]
        );
    }

    #[test]
    fn state_slots_count_every_checkpoint() {
        let shape = Shape::TP4;
        // 16 heads of 128 x 128 F32 per KDA layer: 35.65 MB over 34 layers.
        assert_eq!(shape.kda_state_bytes(), 16 * 128 * 128 * 4);
        assert_eq!(shape.kda_conv_bytes(), 3 * 3 * 16 * 128 * 2);
        assert_eq!(shape.indexer_tail_bytes(), 2 * 3 * 128 * 2);
        assert_eq!(
            shape.recurrent_state_layouts(),
            [
                (KDA_CONV_STATE_ID, 34 * 36_864),
                (KDA_STATE_ID, 34 * 1_048_576)
            ]
        );
        // Twelve sparse layers' tails and six checkpoints each.
        assert_eq!(shape.indexer_tail_slot_bytes(true), 7 * 12 * 1_536);
        // The committed copy plus four rows at three drafts: about 184 MB.
        assert_eq!(
            shape.state_slot_bytes(true, 4),
            5 * (1_253_376 + 35_651_584) + 7 * 12 * 1_536
        );
    }

    #[test]
    fn topology_declares_its_context_parallel_sizes() {
        let topology = bundle_topology().unwrap();
        assert_eq!(topology.rank_count(), 4);
        assert_eq!(topology.context_parallel_sizes(), [2, 4]);
        assert!(
            CONTEXT_PARALLEL_SIZES
                .iter()
                .all(|&n| topology.serves_context_parallel(n))
        );
        assert!(!topology.serves_context_parallel(8));
        assert_eq!(PREFILL_GRANULE, 64);
        assert_eq!((head_rows(), logits_rows()), (32, 64));
        assert_eq!(PAGE_TOKENS % u64::from(Shape::TP4.profile.index_pool), 0);
    }
}
