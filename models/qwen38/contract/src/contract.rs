//! Qwen geometry shared by package declarations and the native module.
//! The selected Geometry owns model dimensions; Shape derives per-rank
//! layouts for the topology (docs/serving-profiles.md#qwen-evidence).

use crate::geometry::{FLASH, GEOMETRIES, Geometry};
use bundle_runtime::config::Topology;
pub use bundle_runtime::geometry::{Items, Ranks};
use bundle_runtime::kv::KvFormat;
use kernel_core::contract::DType;

/// Qwen3.8's per-rank shape at `ranks` ranks, from its configuration values
/// (`crate::rank::CONFIG`), refusing a rank count over which a
/// configured size or the gated attention heads do not split exactly.
/// Placement shards weights with it; `Shape` caches the sizes the bundle
/// reads.
pub fn rank_shape(
    ranks: u32,
) -> Result<kernel_core::rank_shape::RankShape, kernel_core::contract::Refusal> {
    let shape = kernel_core::rank_shape::RankShape::new(crate::rank::CONFIG, ranks)?;
    shape.validate_attention(crate::rank::ATTENTION)?;
    Ok(shape)
}

/// Qwen3.8's one MoE block class: every decoder layer's and the MTP
/// layer's experts.
pub const MOE: kernel_core::rank_shape::MoeConfig = crate::rank::MOE;

/// The package ID. One package serves every topology.
pub const BUNDLE_ID: &str = "qwen38.flash.v2";
pub const PROFILE: &str = "qwen38-flash";
pub const TOPOLOGY_ID: &str = "qwen38-flash-tp4";
pub const TOPOLOGY: Topology<'static> = Topology {
    id: TOPOLOGY_ID,
    ranks: Ranks::TP4,
};
/// The rank counts this module serves, TP4 first: its tag also names the
/// rank-invariant kernels.
pub const SERVED_RANKS: [u32; 2] = [4, 2];
/// The topologies this module serves, with one package and one native
/// module for all of them.
pub const TOPOLOGIES: [Topology<'static>; 2] = [
    Topology {
        id: "qwen38-flash-tp2",
        ranks: Ranks::TP2,
    },
    TOPOLOGY,
];
/// Per-rank sizes of one topology: 2 or 4 ranks. Heads, experts,
/// n-gram heads and the vocabulary split evenly; each rank holds one KV
/// head; the shared expert pads to 128 columns.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub geometry: &'static Geometry,
    pub ranks: Ranks,
    pub vocab_per_rank: u32,
    /// Linear attention key and value heads.
    pub key_heads: u32,
    pub value_heads: u32,
    /// q, k and v columns of `in_proj_qkv`, and the convolution channels.
    pub linear_qkv: u32,
    pub linear_value: u32,
    /// `in_proj_a` then `in_proj_b` in one row, padded to a multiple of eight.
    pub linear_ab_stride: u32,
    /// Full attention query heads; the rank's KV head is `kv_head`.
    pub q_heads: u32,
    /// One packed projection row: the heads' query then gate, K, then V.
    pub attn_row: u32,
    pub attn_value: u32,
    pub local_experts: u32,
    /// The rank's shared expert columns, zero-padded to 128.
    pub shared_intermediate: u32,
    pub ngram_local_heads: u32,
    pub ngram_local: u32,
    /// One linear layer's state for one pool entry.
    pub conv_entry_bytes: u64,
    pub delta_entry_bytes: u64,
}

impl PartialEq for Shape {
    fn eq(&self, other: &Self) -> bool {
        self.geometry.id == other.geometry.id
            && self.ranks == other.ranks
            && self.vocab_per_rank == other.vocab_per_rank
            && self.key_heads == other.key_heads
            && self.value_heads == other.value_heads
            && self.linear_qkv == other.linear_qkv
            && self.linear_value == other.linear_value
            && self.linear_ab_stride == other.linear_ab_stride
            && self.q_heads == other.q_heads
            && self.attn_row == other.attn_row
            && self.attn_value == other.attn_value
            && self.local_experts == other.local_experts
            && self.shared_intermediate == other.shared_intermediate
            && self.ngram_local_heads == other.ngram_local_heads
            && self.ngram_local == other.ngram_local
            && self.conv_entry_bytes == other.conv_entry_bytes
            && self.delta_entry_bytes == other.delta_entry_bytes
    }
}

impl Eq for Shape {}

impl Shape {
    pub const fn attn_head(self) -> u32 {
        self.geometry.head_dim()
    }
    pub const fn context_tokens(self) -> u64 {
        self.geometry.context_tokens()
    }
    pub const fn conv_taps(self) -> u32 {
        self.geometry.conv_taps()
    }
    pub const fn eps(self) -> f32 {
        self.geometry.eps()
    }
    pub const fn experts(self) -> u32 {
        self.geometry.experts()
    }
    pub const fn expert_intermediate(self) -> u32 {
        self.geometry.expert_intermediate()
    }
    pub const fn full_layers(self) -> u64 {
        self.geometry.full_layers()
    }
    pub const fn full_layers_mtp(self) -> u64 {
        self.full_layers() + self.mtp_layers() as u64
    }
    pub const fn hc_rank(self) -> u32 {
        self.geometry.hc_rank()
    }
    pub const fn hidden(self) -> u32 {
        self.geometry.hidden()
    }
    pub const fn index_dim(self) -> u32 {
        self.geometry.index_dim()
    }
    pub const fn index_heads(self) -> u32 {
        self.geometry.index_heads()
    }
    pub const fn index_row(self) -> u32 {
        (self.index_heads() + self.geometry.index_kv_heads()) * self.index_dim()
    }
    pub const fn kv_heads(self) -> u32 {
        self.geometry.attention_kv_heads().div_ceil(self.ranks())
    }
    pub const fn layers(self) -> u32 {
        self.geometry.layers()
    }
    pub const fn linear_head(self) -> u32 {
        self.geometry.linear_head()
    }
    pub const fn linear_key_heads(self) -> u32 {
        self.geometry.linear_key_heads()
    }
    pub const fn linear_layers(self) -> u64 {
        self.layers() as u64 - self.full_layers()
    }
    pub const fn linear_value_heads(self) -> u32 {
        self.geometry.linear_value_heads()
    }
    pub const fn mtp_layer(self) -> u32 {
        self.layers()
    }
    pub const fn mtp_layers(self) -> u32 {
        self.geometry.mtp_layers()
    }
    pub const fn ngram_eos(self) -> u32 {
        self.geometry.eos_token_id()
    }
    pub const fn ngram_heads(self) -> u32 {
        self.geometry.ngram_heads()
    }
    pub const fn ngram_heads_per_order(self) -> u32 {
        self.geometry.ngram_heads_per_order()
    }
    pub const fn ngram_head_width(self) -> u32 {
        self.geometry.ngram_head_width()
    }
    pub const fn ngram_size(self) -> u32 {
        self.geometry.ngram_orders()
    }
    pub const fn page_bytes_bf16(self) -> u64 {
        self.full_layers() * self.page_layer_bytes_bf16()
    }
    pub const fn page_bytes_bf16_mtp(self) -> u64 {
        self.full_layers_mtp() * self.page_layer_bytes_bf16()
    }
    pub const fn page_bytes_fp8(self) -> u64 {
        self.full_layers() * self.page_layer_bytes_fp8()
    }
    pub const fn page_bytes_fp8_mtp(self) -> u64 {
        self.page_bytes_fp8() + self.mtp_layers() as u64 * self.page_layer_bytes_bf16()
    }
    pub const fn page_planes_bf16(self) -> [u64; 3] {
        let kv = PAGE_TOKENS * self.kv_heads() as u64 * self.attn_head() as u64 * BF16_BYTES;
        [
            kv,
            kv,
            PAGE_TOKENS / self.qsa_block() as u64 * self.index_dim() as u64 * BF16_BYTES,
        ]
    }
    pub const fn page_planes_fp8(self) -> [u64; 3] {
        let planes = self.page_planes_bf16();
        [planes[0] / BF16_BYTES, planes[1] / BF16_BYTES, planes[2]]
    }
    pub const fn page_layer_bytes_bf16(self) -> u64 {
        self.page_planes_bf16()[0] + self.page_planes_bf16()[1] + self.page_planes_bf16()[2]
    }
    pub const fn page_layer_bytes_fp8(self) -> u64 {
        self.page_planes_fp8()[0] + self.page_planes_fp8()[1] + self.page_planes_fp8()[2]
    }
    pub const fn ple_conv_bytes(self) -> u64 {
        self.ple_conv_rows() as u64 * self.wide() as u64 * BF16_BYTES
    }
    pub const fn ple_conv_dilation(self) -> u32 {
        self.geometry.ple_conv_dilation()
    }
    pub const fn ple_conv_rows(self) -> u32 {
        (self.ple_conv_taps() - 1) * self.ple_conv_dilation()
    }
    pub const fn ple_conv_taps(self) -> u32 {
        self.geometry.ple_conv_taps()
    }
    pub const fn ple_embed_dim(self) -> u32 {
        self.geometry.ple_embed_dim()
    }
    pub const fn ple_entry_bytes(self) -> u64 {
        self.ple_conv_bytes() + 128
    }
    pub const fn ple_layer(self) -> u32 {
        self.geometry.ple_layer()
    }
    pub const fn qsa_block(self) -> u32 {
        self.geometry.qsa_block()
    }
    pub const fn ring_slot_key_bytes(self) -> u64 {
        self.qsa_ring() as u64 * self.index_dim() as u64 * BF16_BYTES
    }
    pub const fn ring_slot_layer_bytes(self) -> u64 {
        self.ring_slot_key_bytes() + self.ring_slot_position_bytes()
    }
    pub const fn ring_slot_position_bytes(self) -> u64 {
        self.qsa_ring() as u64 * 3 * 4
    }
    pub const fn rope_theta(self) -> f32 {
        self.geometry.rope_theta()
    }
    pub const fn rotary(self) -> u32 {
        self.geometry.rotary()
    }
    pub const fn shared_expert_intermediate(self) -> u32 {
        self.geometry.shared_expert_intermediate()
    }
    pub const fn softmax_scale(self) -> f32 {
        self.geometry.softmax_scale()
    }
    pub const fn streams(self) -> u32 {
        self.geometry.streams()
    }
    pub const fn top_blocks(self) -> u32 {
        self.geometry.top_blocks()
    }
    pub const fn top_k(self) -> u32 {
        self.geometry.top_k()
    }
    pub const fn vocab(self) -> u32 {
        self.geometry.vocab()
    }
    pub const fn wide(self) -> u32 {
        self.hidden() * self.streams()
    }
    pub const fn is_full(self, layer: u32) -> bool {
        self.geometry.is_full(layer)
    }
    pub const fn ordinal(self, layer: u32) -> u64 {
        self.geometry.ordinal(layer)
    }
    pub const fn max_pages(self) -> u32 {
        self.model_limits().max_pages()
    }

    pub const fn kv_formats(self) -> [KvFormat; 4] {
        [
            KvFormat {
                layout_id: KV_LAYOUT_BF16,
                page_bytes: self.page_bytes_bf16(),
                element: DType::Bf16,
                scales: &[],
            },
            KvFormat {
                layout_id: KV_LAYOUT_BF16_MTP,
                page_bytes: self.page_bytes_bf16_mtp(),
                element: DType::Bf16,
                scales: &[],
            },
            KvFormat {
                layout_id: KV_LAYOUT_FP8,
                page_bytes: self.page_bytes_fp8(),
                element: DType::Fp8E4M3,
                scales: &[],
            },
            KvFormat {
                layout_id: KV_LAYOUT_FP8_MTP,
                page_bytes: self.page_bytes_fp8_mtp(),
                element: DType::Fp8E4M3,
                scales: &[],
            },
        ]
    }
    pub const fn ring_state_slot_bytes(self, mtp: bool) -> u64 {
        let layers = if mtp {
            self.full_layers_mtp()
        } else {
            self.full_layers()
        };
        (2 * layers * self.ring_slot_layer_bytes()).next_multiple_of(256)
    }

    pub const fn qsa_ring(self) -> u32 {
        QSA_RING
    }

    pub const fn ring(self, layers: u64, slots: u64) -> Ring {
        Ring {
            shape: self,
            layers,
            slots,
        }
    }

    pub fn page_layer(self, format: &KvFormat, layer: u32) -> (u64, [u64; 3], bool) {
        let main = if format.fp8() {
            self.page_layer_bytes_fp8()
        } else {
            self.page_layer_bytes_bf16()
        };
        let fp8 = format.fp8() && layer < self.layers();
        let planes = if fp8 {
            self.page_planes_fp8()
        } else {
            self.page_planes_bf16()
        };
        (self.ordinal(layer) * main, planes, fp8)
    }

    pub const TP2: Self = Self::flash(Ranks::TP2);
    pub const TP4: Self = Self::flash(Ranks::TP4);
    /// Every topology, in rank order; one package serves them all.
    pub const ALL: [Self; 2] = [Self::TP2, Self::TP4];

    /// The shape of 2 or 4 ranks. Other counts do not compile into a
    /// const and panic at run time; see [`Shape::of`].
    pub const fn flash(ranks: Ranks) -> Self {
        Self::new(&FLASH, ranks)
    }

    pub const fn new(geometry: &'static Geometry, ranks: Ranks) -> Self {
        let n = ranks.get();
        assert!(n == 2 || n == 4, "Qwen serves 2 or 4 ranks");
        let key_heads = geometry.linear_key_heads() / n;
        let value_heads = geometry.linear_value_heads() / n;
        let q_heads = geometry.attention_q_heads() / n;
        let kv_heads = geometry.attention_kv_heads().div_ceil(n);
        let linear_qkv =
            2 * key_heads * geometry.linear_head() + value_heads * geometry.linear_value_head();
        Self {
            geometry,
            ranks,
            vocab_per_rank: geometry.vocab() / n,
            key_heads,
            value_heads,
            linear_qkv,
            linear_value: value_heads * geometry.linear_value_head(),
            linear_ab_stride: (2 * value_heads).next_multiple_of(8),
            q_heads,
            attn_row: (2 * q_heads + 2 * kv_heads) * geometry.head_dim(),
            attn_value: q_heads * geometry.head_dim(),
            local_experts: geometry.experts() / n,
            shared_intermediate: (geometry.shared_expert_intermediate() / n).next_multiple_of(128),
            ngram_local_heads: geometry.ngram_heads() / n,
            ngram_local: geometry.ngram_heads() / n * geometry.ngram_head_width(),
            conv_entry_bytes: (geometry.conv_taps() as u64 - 1) * linear_qkv as u64 * 2,
            delta_entry_bytes: value_heads as u64
                * geometry.linear_head() as u64
                * geometry.linear_value_head() as u64
                * 4,
        }
    }

    /// The shape of a topology this module serves: 2 or 4 ranks, over
    /// which every configured size and the attention heads split exactly.
    pub fn of(ranks: Ranks) -> Option<Self> {
        Self::of_geometry(&FLASH, ranks)
    }

    pub fn of_geometry(geometry: &'static Geometry, ranks: Ranks) -> Option<Self> {
        if !matches!(ranks.get(), 2 | 4) {
            return None;
        }
        let shape = Self::new(geometry, ranks);
        shape.rank_shape_checked().ok().map(|_| shape)
    }

    fn rank_shape_checked(
        self,
    ) -> Result<kernel_core::rank_shape::RankShape, kernel_core::contract::Refusal> {
        let shape = kernel_core::rank_shape::RankShape::new(self.model_config(), self.ranks())?;
        shape.validate_attention(self.attention_config())?;
        Ok(shape)
    }

    /// The model-neutral derivation of this topology's sizes.
    pub fn rank_shape(self) -> kernel_core::rank_shape::RankShape {
        self.rank_shape_checked()
            .expect("a served topology splits exactly")
    }

    pub const fn ranks(self) -> u32 {
        self.ranks.get()
    }

    /// The symbol tag of the per-rank entries: the geometry's kernel tag
    /// (`qwen38` for Flash) at TP4, `{tag}_tp2` at TP2.
    pub fn tag(self) -> &'static str {
        if self.ranks() != 2 {
            return self.geometry.kernel_tag;
        }
        static TAGS: std::sync::OnceLock<Vec<(&'static str, String)>> = std::sync::OnceLock::new();
        let tags = TAGS.get_or_init(|| {
            GEOMETRIES
                .iter()
                .map(|g| (g.id, format!("{}_tp2", g.kernel_tag)))
                .collect()
        });
        &tags
            .iter()
            .find(|(id, _)| *id == self.geometry.id)
            .expect("registered geometry tag")
            .1
    }

    /// Row-gather flag sets: one per n-gram all-gather target, then the
    /// logits gather.
    pub const fn gather_sets(self) -> usize {
        self.ranks.count() + 1
    }

    /// The package topology: `qwen38-flash-tp{N}`.
    pub fn topology(self) -> Topology<'static> {
        static IDS: std::sync::OnceLock<Vec<(&'static str, u32, String)>> =
            std::sync::OnceLock::new();
        let ids = IDS.get_or_init(|| {
            GEOMETRIES
                .iter()
                .flat_map(|g| [2, 4].map(|n| (g.id, n, format!("qwen38-{}-tp{n}", g.id))))
                .collect()
        });
        let id = ids
            .iter()
            .find(|(id, n, _)| *id == self.geometry.id && *n == self.ranks())
            .expect("registered geometry topology");
        Topology {
            id: &id.2,
            ranks: self.ranks,
        }
    }

    pub fn from_topology(id: &str) -> Option<Self> {
        let (geometry, ranks) = id.strip_prefix("qwen38-")?.rsplit_once("-tp")?;
        let geometry = GEOMETRIES.iter().find(|g| g.id == geometry)?;
        let shape = Self::of_geometry(geometry, Ranks::new(ranks.parse().ok()?)?)?;
        (shape.topology().id == id).then_some(shape)
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

    pub const fn attention_config(self) -> kernel_core::rank_shape::AttentionConfig {
        kernel_core::rank_shape::AttentionConfig {
            q_heads: self.geometry.attention_q_heads(),
            kv_heads: self.geometry.attention_kv_heads(),
            qk_head: self.geometry.head_dim(),
            v_head: self.geometry.head_dim(),
        }
    }

    pub const fn moe(self) -> kernel_core::rank_shape::MoeConfig {
        kernel_core::rank_shape::MoeConfig {
            experts: self.geometry.experts(),
            top_k: self.geometry.top_k(),
            intermediate: self.geometry.expert_intermediate(),
            split: kernel_core::rank_shape::ExpertSplit::Experts,
            shared_intermediate: self.geometry.shared_expert_intermediate(),
            shared_tile: 128,
        }
    }

    pub fn model_config(self) -> kernel_core::rank_shape::ModelConfig {
        crate::geometry::model_config(self.geometry)
    }

    /// Full attention KV head of `rank`: ranks share a head in contiguous
    /// groups (`rank / 2` at TP4).
    pub const fn kv_head(self, rank: u32) -> u32 {
        rank * self.geometry.attention_kv_heads() / self.ranks.get()
    }

    /// The request state layouts the manifest declares: bytes of one
    /// sequence slot on one rank, over every layer.
    pub const fn recurrent_state_layouts(self) -> [(&'static str, u64); 3] {
        [
            (
                "qwen38.linear-conv.v1",
                self.linear_layers() * self.conv_entry_bytes,
            ),
            (
                "qwen38.linear-delta.v1",
                self.linear_layers() * self.delta_entry_bytes,
            ),
            ("qwen38.ple.v1", self.ple_entry_bytes()),
        ]
    }
}

// The current attention layout serves one KV head per rank, and out is shared
// only while linear and full attention have equal value widths.
const _: () = {
    let mut i = 0;
    while i < Shape::ALL.len() {
        let shape = Shape::ALL[i];
        let rank = crate::rank::RankShape {
            ranks: shape.ranks.get(),
        };
        assert!(rank.shape().attention(crate::rank::ATTENTION).kv_heads == 1);
        assert!(shape.linear_value == shape.attn_value);
        i += 1;
    }
};

/// Hash table rows and offsets for the served PLE geometry. Seed zero is enough
/// here because callers use rows and offsets; multipliers come from the checkpoint.
#[cfg(test)]
pub fn ngram_tables()
-> Result<kernel_core::ngram_oracle::NgramTables, kernel_core::contract::Refusal> {
    ngram_tables_for(&FLASH)
}

pub fn ngram_tables_for(
    geometry: &Geometry,
) -> Result<kernel_core::ngram_oracle::NgramTables, kernel_core::contract::Refusal> {
    kernel_core::ngram_oracle::ngram_tables(
        u64::from(geometry.vocab()),
        geometry.ngram_orders() as usize,
        geometry.ngram_heads_per_order() as usize,
        u64::from(geometry.ngram_vocab_base()),
        0,
        0,
        u64::from(geometry.ngram_alignment()),
    )
}

/// Native context; YaRN to 1M is a later option.
pub const CONTEXT_TOKENS: u64 = crate::geometry::FLASH.context_tokens();
/// Policy: prefill launches cap scheduler chunks.
pub const PREFILL_TOKENS: u32 = 8192;
/// The package maximum: most sequences an instance serves and most items in
/// one step (docs/serving-profiles.md). An instance serves a
/// runtime `Items` count up to it, from the host's `--max-sequences`.
/// Policy: the runtime bounds served items.
pub const DECODE_BATCH: u32 = bundle_runtime::geometry::MAX_ITEMS;
/// The largest graph bucket of the decode kernels: the cluster decode
/// GEMMs, the fused hyper-connection mix, the decode route and experts, the
/// decode heads and the decode all-reduce. Larger buckets run the prefill
/// entries at the bucket's rows.
/// Policy: the largest compiled decode bucket bounds graph rows.
pub const DECODE_ENTRY_ROWS: u32 = 32;
/// The decode kernels' graph buckets.
/// Policy: compiled graph buckets round scheduled rows.
pub const DECODE_ENTRY_BUCKETS: [u32; 6] = [1, 2, 4, 8, 16, 32];
/// Most items that draft in one step. The checkpoint pool is bounded by
/// this count, independently of the served slots and total verify rows.
/// Policy: checkpoint capacity bounds drafting items.
pub const DRAFT_ITEMS: u32 = 32;
/// Maximum logits rows in a scored prefill phase.
/// The qualified scored-prefill path allocates buffers for this row cap.
pub const SCORED_ROWS: u32 = 512;
/// Positions a prefill chunk may end at before the prompt end, and prefix
/// anchors sit at: the linear attention's delta-rule chunk, so its chunks
/// start at the same positions whatever the prompt's chunking or anchor.
/// Policy: the linear engine chunk fixes prefix anchor alignment.
pub const PREFILL_GRANULE: u32 =
    <crate::backend::Selected as crate::backend::Kernels>::LINEAR_CHUNK;
/// Prompt-ending prefill items per step that keep their model state at the
/// last `PREFILL_GRANULE` multiple inside their rows, for a prefix anchor
/// there (the manifest's `batch_limits.anchor_items`). Each takes a spare
/// linear-state entry per layer (115 MB over the ranks, 58 MB per TP2 rank)
/// and small PLE, ring and retained-row spares; two keep TP2 at its 64
/// sequences. The scheduler ends the chunk of any further such item at its
/// anchor instead.
/// Policy: spare state capacity bounds simultaneous prefix anchors.
pub const ANCHOR_ITEMS: u32 = 2;
/// The scored rows' F32 buffers also hold a decode bucket's rows above the
/// decode heads' (`lower.rs` head), and a prefill phase's last rows of every
/// served item.
const _: () = assert!(SCORED_ROWS >= DECODE_BATCH);
/// Verify rows one item may carry: the bonus token and up to seven drafts.
/// The qualified speculation path supports seven drafts plus the bonus.
pub const MAX_VERIFY_ROWS: u32 = 8;
/// The speculation a package with the MTP drafter declares: drafts per
/// item, verify rows per step (one decode bucket) and the verify-row
/// buckets the host rounds drafts to.
pub const MAX_DRAFT_TOKENS: u32 = MAX_VERIFY_ROWS - 1;
pub const MAX_VERIFY_TOKENS: u32 = DECODE_BATCH;
/// The qualified speculation graphs cover two, four, and eight rows.
pub const VERIFY_ROW_BUCKETS: [u32; 3] = [2, 4, 8];
/// The verify steps whose rows are exact (docs/numerics.md#state-and-verification-contract):
/// the MTP drafter's K3 steps of one request served at a time, four rows
/// (the bonus and three drafts) whose arithmetic is AR's. Larger buckets and
/// concurrent requests are not covered.
pub const EXACT_VERIFY: kernel_core::properties::ExactVerify =
    kernel_core::properties::ExactVerify {
        rows: crate::rank::EXACT_VERIFY_ROWS,
        items: 1,
    };
/// Policy: page allocation and attention launches use a fixed tile.
pub const PAGE_TOKENS: u64 = 64;
pub const VOCAB: u32 = crate::geometry::FLASH.vocab();
/// Policy: token buffers align to the runtime allocation granule.
pub const TOKEN_GRANULE: u32 = 256;
pub const BF16_BYTES: u64 = 2;

pub const LAYERS: u32 = crate::geometry::FLASH.layers();
/// The layer with the per-layer n-gram embedding (0-indexed; the config's
/// `ple_layer_ids` is 1-based).
pub const PLE_LAYER: u32 = crate::geometry::FLASH.ple_layer();
pub const HIDDEN: u32 = crate::geometry::FLASH.hidden();
pub const STREAMS: u32 = crate::geometry::FLASH.streams();
/// The four hyper-connection streams side by side.
pub const WIDE: u32 = HIDDEN * STREAMS;
pub const HC_RANK: u32 = crate::geometry::FLASH.hc_rank();
/// Every weight is replicated or sharded 1:1, so the norm epsilon is shared.
pub const EPS: f32 = crate::geometry::FLASH.eps();

/// Linear attention over all ranks: 16 key heads and 48 value heads.
pub const LINEAR_KEY_HEADS: u32 = crate::geometry::FLASH.linear_key_heads();
pub const LINEAR_VALUE_HEADS: u32 = crate::geometry::FLASH.linear_value_heads();
pub const LINEAR_HEAD: u32 = crate::geometry::FLASH.linear_head();
pub const CONV_TAPS: u32 = crate::geometry::FLASH.conv_taps();

/// Full attention over all ranks: 24 query heads and 2 KV heads.
pub const ATTN_Q_HEADS: u32 = crate::geometry::FLASH.attention_q_heads();
pub const ATTN_KV_HEADS: u32 = crate::geometry::FLASH.attention_kv_heads();
/// Every topology holds one full attention KV head per rank.
pub const KV_HEADS: u32 = crate::rank::RankShape { ranks: 4 }
    .shape()
    .attention(crate::rank::ATTENTION)
    .kv_heads;
pub const ATTN_HEAD: u32 = crate::geometry::FLASH.head_dim();
pub const ROTARY: u32 = crate::geometry::FLASH.rotary();
pub const ROPE_THETA: f32 = crate::geometry::FLASH.rope_theta();
pub const SOFTMAX_SCALE: f32 = crate::geometry::FLASH.softmax_scale();
/// The replicated QSA indexer: four query heads and one key of 128.
pub const INDEX_HEADS: u32 = crate::geometry::FLASH.index_heads();
pub const INDEX_DIM: u32 = crate::geometry::FLASH.index_dim();
pub const INDEX_ROW: u32 = (INDEX_HEADS + crate::geometry::FLASH.index_kv_heads()) * INDEX_DIM;
pub const QSA_BLOCK: u32 = crate::geometry::FLASH.qsa_block();
/// The ring covers the open QSA block and the decode verification window.
pub const QSA_RING: u32 = 16;
pub const TOP_BLOCKS: u32 = crate::geometry::FLASH.top_blocks();
/// QSA scores and selects prefill rows in pieces of at most this many.
/// Policy: the indexer scratch allocation bounds score chunks.
pub const SCORE_ROWS: u32 = 2048;

pub const EXPERTS: u32 = crate::geometry::FLASH.experts();
pub const TOP_K: u32 = crate::geometry::FLASH.top_k();
pub const EXPERT_INTERMEDIATE: u32 = crate::geometry::FLASH.expert_intermediate();
pub const SHARED_EXPERT_INTERMEDIATE: u32 = crate::geometry::FLASH.shared_expert_intermediate();

pub const NGRAM_HEADS: u32 = crate::geometry::FLASH.ngram_heads();
pub const NGRAM_HEAD_WIDTH: u32 = crate::geometry::FLASH.ngram_head_width();
/// Maximum n-gram order; orders start at two.
pub const NGRAM_SIZE: u32 = crate::geometry::FLASH.ngram_orders();
pub const NGRAM_HEADS_PER_ORDER: u32 = crate::geometry::FLASH.ngram_heads_per_order();
/// End-of-document token used when the n-gram history crosses a boundary.
pub const NGRAM_EOS: u32 = crate::geometry::FLASH.eos_token_id();
/// Taps of the PLE convolution.
pub const PLE_CONV_TAPS: u32 = crate::geometry::FLASH.ple_conv_taps();
/// The checkpoint defines PLE dilation as its maximum n-gram order.
pub const PLE_CONV_DILATION: u32 = crate::geometry::FLASH.ple_conv_dilation();
/// Width of the PLE embedding before its key and value projections.
pub const PLE_EMBED_DIM: u32 = crate::geometry::FLASH.ple_embed_dim();
/// Previous inputs retained by the dilated PLE convolution.
pub const PLE_CONV_ROWS: u32 = (PLE_CONV_TAPS - 1) * PLE_CONV_DILATION;

pub const FULL_LAYERS: u64 = crate::geometry::FLASH.full_layers();
pub const LINEAR_LAYERS: u64 = LAYERS as u64 - FULL_LAYERS;
/// The MTP drafter's decoder layer, a full-attention layer numbered after
/// the 48. With the drafter, pages and the ring hold it after the main
/// full layers.
pub const MTP_LAYER: u32 = LAYERS;
pub const MTP_LAYERS: u32 = crate::geometry::FLASH.mtp_layers();
pub const FULL_LAYERS_MTP: u64 = FULL_LAYERS + MTP_LAYERS as u64;

/// One page of one full-attention layer: K, V and QSA block keys.
pub const PAGE_PLANES_BF16: [u64; 3] = [
    PAGE_TOKENS * KV_HEADS as u64 * ATTN_HEAD as u64 * BF16_BYTES,
    PAGE_TOKENS * KV_HEADS as u64 * ATTN_HEAD as u64 * BF16_BYTES,
    PAGE_TOKENS / QSA_BLOCK as u64 * INDEX_DIM as u64 * BF16_BYTES,
];
pub const PAGE_LAYER_BYTES_BF16: u64 =
    PAGE_PLANES_BF16[0] + PAGE_PLANES_BF16[1] + PAGE_PLANES_BF16[2];
pub const PAGE_BYTES_BF16: u64 = FULL_LAYERS * PAGE_LAYER_BYTES_BF16;
pub const PAGE_BYTES_BF16_MTP: u64 = FULL_LAYERS_MTP * PAGE_LAYER_BYTES_BF16;
/// The FP8 E4M3 page: the main full layers' K and V at one
/// byte per element; their QSA block keys stay BF16, so block selection
/// does not change. With the drafter, the MTP layer follows in BF16: the
/// drafter's KV is not quantized.
pub const PAGE_PLANES_FP8: [u64; 3] = [
    PAGE_PLANES_BF16[0] / BF16_BYTES,
    PAGE_PLANES_BF16[1] / BF16_BYTES,
    PAGE_PLANES_BF16[2],
];
pub const PAGE_LAYER_BYTES_FP8: u64 = PAGE_PLANES_FP8[0] + PAGE_PLANES_FP8[1] + PAGE_PLANES_FP8[2];
pub const PAGE_BYTES_FP8: u64 = FULL_LAYERS * PAGE_LAYER_BYTES_FP8;
pub const PAGE_BYTES_FP8_MTP: u64 = PAGE_BYTES_FP8 + PAGE_LAYER_BYTES_BF16;

/// The PLE convolution history, then the two n-gram history ids.
pub const PLE_CONV_BYTES: u64 = PLE_CONV_ROWS as u64 * WIDE as u64 * BF16_BYTES;
pub const PLE_ENTRY_BYTES: u64 = PLE_CONV_BYTES + 128;

/// Bytes of one slot's raw-key ring in one full layer: its keys, and its
/// RoPE positions.
pub const RING_SLOT_KEY_BYTES: u64 = QSA_RING as u64 * INDEX_DIM as u64 * BF16_BYTES;
pub const RING_SLOT_POSITION_BYTES: u64 = QSA_RING as u64 * 3 * 4;

/// The QSA raw-key ring of `slots` slots over `layers` full layers. Each
/// layer holds every slot's keys, then every slot's positions; the live
/// rings of all layers come first, then the copy a prefill saves. The
/// instance state (`RING_STATE_ID`) is both, for the largest graph bucket's
/// slots (`Items::capacity`, `chop.state-per-item.v1`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Ring {
    pub shape: Shape,
    pub layers: u64,
    pub slots: u64,
}

impl Ring {
    pub const fn key_bytes(self) -> u64 {
        self.slots * self.shape.ring_slot_key_bytes()
    }

    pub const fn position_bytes(self) -> u64 {
        self.slots * self.shape.ring_slot_position_bytes()
    }

    pub const fn layer_bytes(self) -> u64 {
        self.slots * self.shape.ring_slot_layer_bytes()
    }

    /// The live ring of every layer.
    pub const fn bytes(self) -> u64 {
        self.layers * self.layer_bytes()
    }

    /// The live ring and its saved copy: the instance state's bytes.
    pub const fn state_bytes(self) -> u64 {
        (2 * self.layers * self.shape.ring_slot_layer_bytes()).next_multiple_of(256) * self.slots
    }
}

/// The ring state one slot adds, which a package without the drafter
/// declares per state slot (`chop.state-per-item.v1`).
pub const RING_STATE_SLOT_BYTES: u64 = Ring {
    shape: Shape::flash(crate::contract::Ranks::TP4),
    layers: FULL_LAYERS,
    slots: 1,
}
.state_bytes();
/// Aligned bytes per state slot with the MTP layer. Padding is at the end
/// of the allocation; the live and saved ring layouts remain unchanged.
pub const RING_STATE_SLOT_BYTES_MTP: u64 = Ring {
    shape: Shape::flash(crate::contract::Ranks::TP4),
    layers: FULL_LAYERS_MTP,
    slots: 1,
}
.state_bytes();
/// Compatibility constant for fixtures with 32 slots.
pub const RING_STATE_BYTES_MTP: u64 = RING_STATE_SLOT_BYTES_MTP * DRAFT_ITEMS as u64;

/// Package-facing identifiers: the rank-image component, its per-rank
/// index roles, the serving KV layout and the persistent ring state.
pub const RANK_IMAGE_ROLE: &[u8] = b"qwen38.rank-images.v1";
pub const KV_LAYOUT_BF16: &[u8] = b"qwen38.qsa-paged-bf16.v1";
/// The same pages and ring with the MTP drafter's layer after the main
/// full layers (a package built with the drafter).
pub const KV_LAYOUT_BF16_MTP: &[u8] = b"qwen38.qsa-paged-bf16-mtp.v1";
pub const KV_LAYOUT_FP8: &[u8] = b"qwen38.qsa-paged-fp8.v1";
/// FP8 main layers, then the MTP drafter's layer in BF16.
pub const KV_LAYOUT_FP8_MTP: &[u8] = b"qwen38.qsa-paged-fp8-mtp.v1";
pub const RING_STATE_ID: &[u8] = b"qwen38.qsa-ring.v1";
/// The compiled-image view a package built with `--ngram-placement host`
/// declares over each rank's n-gram table, placed in host memory. Without
/// it the table stays in the rank arena on the device.
pub const NGRAM_HOST_VIEW: &[u8] = b"qwen38.ngram-table.v1";

/// The serving KV layouts: BF16 by default, and FP8 E4M3 K and V at scale
/// 1.0 on every main full layer, each without and with the
/// MTP drafter's layer. `element` is the main layers' element.
pub const KV_FORMATS: [KvFormat; 4] = [
    KvFormat {
        layout_id: KV_LAYOUT_BF16,
        page_bytes: PAGE_BYTES_BF16,
        element: DType::Bf16,
        scales: &[],
    },
    KvFormat {
        layout_id: KV_LAYOUT_BF16_MTP,
        page_bytes: PAGE_BYTES_BF16_MTP,
        element: DType::Bf16,
        scales: &[],
    },
    KvFormat {
        layout_id: KV_LAYOUT_FP8,
        page_bytes: PAGE_BYTES_FP8,
        element: DType::Fp8E4M3,
        scales: &[],
    },
    KvFormat {
        layout_id: KV_LAYOUT_FP8_MTP,
        page_bytes: PAGE_BYTES_FP8_MTP,
        element: DType::Fp8E4M3,
        scales: &[],
    },
];

/// Where full layer `layer` sits in a page of `format`: its byte offset and
/// planes, and whether its K and V are FP8. The MTP layer is always BF16.
pub fn page_layer(format: &KvFormat, layer: u32) -> (u64, &'static [u64; 3], bool) {
    let main = if format.fp8() {
        PAGE_LAYER_BYTES_FP8
    } else {
        PAGE_LAYER_BYTES_BF16
    };
    let fp8 = format.fp8() && layer != MTP_LAYER;
    let planes = if fp8 {
        &PAGE_PLANES_FP8
    } else {
        &PAGE_PLANES_BF16
    };
    (ordinal(layer) * main, planes, fp8)
}

pub fn index_role(rank: u32) -> String {
    bundle_runtime::index::index_role("qwen38", bundle_runtime::index::ImageKind::Text, rank)
}

/// Qwen3.8's geometry for the shared bundle runtime.
pub struct Qwen;

/// Model dimensions passed to shared runtime mechanics.
pub const MODEL_LIMITS: bundle_runtime::ModelLimits =
    match bundle_runtime::ModelLimits::new(VOCAB, CONTEXT_TOKENS, PAGE_TOKENS) {
        Ok(limits) => limits,
        Err(_) => panic!("invalid model limits"),
    };

/// Entries of one item's device block-table row.
pub const MAX_PAGES: u32 = MODEL_LIMITS.max_pages();

impl bundle_runtime::BundleGeometry for Qwen {
    const PAGE_TOKENS: u64 = PAGE_TOKENS;
    const PREFILL_TOKENS: u32 = PREFILL_TOKENS;
    const DECODE_BATCH: u32 = DECODE_BATCH;
    const MAX_VERIFY_ROWS: u32 = MAX_VERIFY_ROWS;
    const VERIFY_TOKENS: u32 = MAX_VERIFY_TOKENS;
    const NARROW_VERIFY_TOKENS: Option<u32> = Some(DECODE_ENTRY_ROWS);
    fn narrow_verify_tokens() -> Option<u32> {
        // Read once: allocation, upload offsets and cached graphs must use
        // the same geometry for the entire lifetime of this process.
        static WIDE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let wide = *WIDE
            .get_or_init(|| std::env::var("CHOPINFER_SPECULATIVE_WIDE_ROWS").as_deref() == Ok("1"));
        if wide {
            None
        } else {
            Self::NARROW_VERIFY_TOKENS
        }
    }
    const TOKEN_GRANULE: u32 = TOKEN_GRANULE;
}

/// Full attention sits at layers 3, 7, ..., 47 and in the MTP layer; the
/// rest are linear.
pub const fn is_full(layer: u32) -> bool {
    crate::geometry::FLASH.is_full(layer)
}

/// A layer's index among the layers of its kind.
pub const fn ordinal(layer: u32) -> u64 {
    crate::geometry::FLASH.ordinal(layer)
}

#[cfg(test)]
mod rank_shape_tests {
    use super::*;

    static SMALL: Geometry = {
        let mut g = FLASH;
        g.shape.hidden = 512;
        g.shape.vocab = 1024;
        g.shape.context = 512;
        g.shape.layers = &[
            crate::geometry::LayerKind::Full,
            crate::geometry::LayerKind::Linear,
        ];
        g.shape.hc.streams = 2;
        g.shape.linear.key_heads = 4;
        g.shape.linear.value_heads = 8;
        g.shape.linear.key_dim = 64;
        g.shape.linear.value_dim = 64;
        g.shape.attention.heads = 8;
        g.shape.attention.head_dim = 64;
        g.shape.indexer.dim = 32;
        g.shape.moe.experts = 16;
        g.shape.moe.top_k = 2;
        g.shape.moe.intermediate = 256;
        g
    };

    #[test]
    fn shape_equality_uses_geometry_identity() {
        let mut geometry = FLASH;
        geometry.document = "different serialization";
        geometry.shape.attention.eps = f64::NAN;
        let geometry = Box::leak(Box::new(geometry));
        for shape in Shape::ALL {
            assert_eq!(shape, Shape::new(geometry, shape.ranks));
        }
    }

    #[test]
    fn runtime_geometry_controls_views_pages_state_and_precision() {
        use crate::precision::DenseRole;
        for ranks in [Ranks::TP2, Ranks::TP4] {
            let shape = Shape::new(&SMALL, ranks);
            assert_eq!(shape.hidden(), 512);
            assert_eq!(shape.local_experts, 16 / ranks.get());
            assert_eq!(shape.linear_value, 512 / ranks.get());
            assert!(shape.is_full(0));
            assert!(!shape.is_full(1));
            assert_eq!(shape.model_limits().max_pages(), 8);
            let formats = shape.kv_formats();
            for format in &formats {
                let state = crate::state::request_state(shape, format, false);
                state.validate().unwrap();
                assert_eq!(state.page_bytes(), format.page_bytes);
                assert_eq!(
                    shape.page_layer(format, 0).1[0],
                    PAGE_TOKENS * 64 * format.element_bytes()
                );
            }
            let views = crate::views::required(0, shape);
            assert!(
                views
                    .iter()
                    .any(|v| v.name.ends_with("layers.0.self_attn.q_proj.weight"))
            );
            assert!(
                views
                    .iter()
                    .any(|v| v.name.ends_with("layers.1.linear_attn.in_proj_qkv.weight"))
            );
            assert!(!views.iter().any(|v| v.name.contains("layers.2.")));
            assert_eq!(
                DenseRole::AttentionQ.logical_geometry(shape).unwrap().k,
                512
            );
            assert_eq!(
                shape.ring(shape.full_layers(), 3).key_bytes(),
                3 * shape.ring_slot_key_bytes()
            );
            assert!(
                crate::package::working_bytes(shape, 256)
                    < crate::package::working_bytes(Shape::flash(ranks), 256)
            );
        }
        static TAGGED: Geometry = {
            let mut g = SMALL;
            g.id = "small";
            g.kernel_tag = "qwen38_small";
            g
        };
        let shape = Shape::new(&TAGGED, Ranks::TP4);
        let lookup = crate::precision::legacy_lookup_capability(
            shape,
            false,
            crate::precision::uses().unwrap()[0],
        )
        .unwrap();
        assert_eq!(lookup.stages, ["chop_row_embedding_qwen38_small_v1"]);
    }

    #[test]
    fn geometry_without_mtp_has_no_drafter_views_or_state() {
        static NO_MTP: Geometry = {
            let mut g = SMALL;
            g.shape.mtp.layers = &[];
            g
        };
        let shape = Shape::new(&NO_MTP, Ranks::TP4);
        assert!(crate::views::drafter_required(shape).is_empty());
        let formats = shape.kv_formats();
        assert_eq!(formats[0].page_bytes, formats[1].page_bytes);
        assert_eq!(formats[2].page_bytes, formats[3].page_bytes);
        let state = crate::state::request_state(shape, &formats[1], false);
        assert!(state.pages.iter().all(|p| p.layers > 0));
        state.validate().unwrap();
    }

    #[test]
    fn topology_ids_round_trip_and_refuse_aliases() {
        for geometry in GEOMETRIES {
            for ranks in [Ranks::TP2, Ranks::TP4] {
                let shape = Shape::new(geometry, ranks);
                assert_eq!(Shape::from_topology(shape.topology().id), Some(shape));
                assert_eq!(
                    shape.profile().as_str(),
                    if ranks == Ranks::TP4 {
                        format!("qwen38-{}", geometry.id)
                    } else {
                        format!("qwen38-{}-tp2", geometry.id)
                    }
                );
            }
        }
        for id in ["qwen38-flash-tp02", "qwen38-flash-tp8", "qwen38-absent-tp2"] {
            assert_eq!(Shape::from_topology(id), None);
        }
    }

    #[test]
    fn tp8_is_not_a_served_topology() {
        assert_eq!(Shape::of(Ranks::TP8), None);
        assert_eq!(crate::sm120::config::profile(8), None);
    }

    /// `Shape` caches sizes `kernel_core`'s rank shape derives from Qwen3.8's
    /// configuration values; they must stay equal at every served count.
    #[test]
    fn shape_sizes_are_the_rank_shapes() {
        for shape in Shape::ALL {
            let rank = shape.rank_shape();
            rank.validate().unwrap();
            let attention = rank.validate_attention(crate::rank::ATTENTION).unwrap();
            let heads = ATTN_HEAD;
            assert_eq!(shape.vocab_per_rank, rank.vocab_shard());
            assert_eq!(shape.key_heads, rank.linear_key_heads());
            assert_eq!(shape.value_heads, rank.linear_value_heads());
            assert_eq!(shape.q_heads, attention.q_heads);
            assert_eq!(
                shape.attn_row,
                attention.q_heads * 2 * heads + attention.kv_heads * 2 * heads
            );
            assert_eq!(shape.attn_value, attention.q_heads * heads);
            assert_eq!(shape.local_experts, rank.experts(MOE).experts);
            assert_eq!(shape.shared_intermediate, rank.experts(MOE).shared);
            assert_eq!(shape.ngram_local_heads, rank.ngram_heads());
            assert_eq!(Shape::of(shape.ranks), Some(shape));
        }
        assert!(rank_shape(3).is_err());
    }
}
