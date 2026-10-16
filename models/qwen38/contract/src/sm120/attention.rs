//! Qwen3.8 Flash's attention on the shared engines (`kernels_sm120::attention`):
//! its gated attention geometry and rows (`attention::gated`), its dense
//! paged attention variants, which fold the gated norm and M-RoPE into
//! decode, and its QSA sparse attention and indexer (`attention::sparse`).

use crate::sm120::config::EXACT_VERIFY_ROWS;
use kernel_core::contract::{AttentionPhase, DType};
use kernels_sm120::Profile;
use kernels_sm120::attention::sparse::{self, Geometry};
use kernels_sm120::attention::{CacheLayout, Shape, Variant, gated, variant};

/// Q heads per TP4 rank, derived from the configuration: 24 / 4
/// (`qwen38::RankShape::q_heads` at other rank counts).
pub const Q_HEADS: u32 = crate::sm120::config::RankShape { ranks: 4 }.q_heads();
/// KV heads per rank: KV head r / 2 of the checkpoint's two.
pub const KV_HEADS: u32 = crate::contract::KV_HEADS;
pub const HEAD_DIM: u32 = crate::contract::ATTN_HEAD;
pub const ROTARY: u32 = crate::contract::ROTARY;
pub const THETA: f32 = crate::contract::ROPE_THETA;
pub const SECTIONS: [u32; 3] = crate::geometry::FLASH.mrope();
pub const EPS: f32 = crate::contract::EPS;
pub const PAGE: u32 = crate::contract::PAGE_TOKENS as u32;
/// The native context: block-table rows hold 4096 pages.
pub const MAX_CONTEXT: u32 = crate::contract::CONTEXT_TOKENS as u32;
/// Rows of one gated launch: a prefill chunk.
pub const MAX_ROWS: u32 = crate::contract::PREFILL_TOKENS;

/// The gated attention geometry per TP4 rank.
pub const GATED: gated::Config = gated::Config {
    q_heads: Q_HEADS,
    kv_heads: KV_HEADS,
    head_dim: HEAD_DIM,
    rotary: ROTARY,
    theta: THETA,
    sections: SECTIONS,
    eps: EPS,
    page: PAGE,
    max_rows: MAX_ROWS,
};

/// Elements of one packed projection row at TP4.
pub const ROW_WIDTH: u32 = GATED.row_width();

/// The QSA geometry per TP4 rank: 6 of 24 query heads, one KV head, and the
/// replicated indexer.
pub const SPARSE: Geometry = Geometry {
    index_heads: crate::contract::INDEX_HEADS,
    index_dim: crate::contract::INDEX_DIM,
    rotary_dim: ROTARY,
    theta: THETA,
    mrope: SECTIONS,
    eps: EPS,
    block: crate::contract::QSA_BLOCK,
    ring: crate::contract::QSA_RING,
    top_blocks: crate::contract::TOP_BLOCKS,
    max_context: MAX_CONTEXT,
    q_heads: Q_HEADS,
    kv_heads: KV_HEADS,
    head_dim: HEAD_DIM,
    page_size: PAGE,
    softmax_scale: crate::contract::SOFTMAX_SCALE,
};

/// The QSA geometry of a Qwen3.8 profile: its Q heads over one KV head, and
/// the replicated indexer.
pub fn sparse_geometry(profile: Profile) -> Option<Geometry> {
    let shape = super::geometry::shape(profile)?;
    let g = shape.geometry;
    Some(Geometry {
        index_heads: g.index_heads(),
        index_dim: g.index_dim(),
        rotary_dim: g.rotary(),
        theta: g.rope_theta(),
        mrope: g.mrope(),
        eps: g.eps(),
        block: g.qsa_block(),
        ring: crate::contract::QSA_RING,
        top_blocks: g.top_blocks(),
        max_context: g.context_tokens() as u32,
        q_heads: shape.q_heads(),
        kv_heads: shape.kv_heads(),
        head_dim: g.head_dim(),
        page_size: PAGE,
        softmax_scale: g.softmax_scale(),
    })
}

/// The gated geometry of a Qwen3.8 profile: its Q heads over one KV head.
pub fn gated_config(profile: Profile) -> Option<gated::Config> {
    let shape = super::geometry::shape(profile)?;
    let g = shape.geometry;
    Some(gated::Config {
        q_heads: shape.q_heads(),
        kv_heads: shape.kv_heads(),
        head_dim: g.head_dim(),
        rotary: g.rotary(),
        theta: g.rope_theta(),
        sections: g.mrope(),
        eps: g.eps(),
        page: PAGE,
        max_rows: MAX_ROWS,
    })
}

/// The gated rows of one Qwen3.8 profile, symbols `chop_attn_{tag}_...`.
/// Empty for other profiles.
pub fn gated_catalog(profile: Profile) -> Vec<gated::Entry> {
    gated_config(profile).map_or_else(Vec::new, |config| gated::entries(profile, config))
}

/// The gated rows at TP4, then at TP2.
pub fn gated_entries() -> Vec<gated::Entry> {
    crate::sm120::PROFILES
        .into_iter()
        .flat_map(gated_catalog)
        .collect()
}

/// The dense paged attention variants at TP4 (the bundle runs QSA, so the
/// other rank counts have none): rows that see more than 2051 tokens attend
/// through the sparse kernels instead. Each takes the native context, and a
/// decode variant folds the gated norm, M-RoPE and KV write.
pub fn catalog() -> Vec<Variant> {
    super::geometry::MODELS
        .iter()
        .flat_map(|m| dense_catalog(m.model.at(4)))
        .collect()
}

fn dense_catalog(profile: Profile) -> Vec<Variant> {
    let g = super::geometry::owner(profile).unwrap().geometry;
    let gated = gated_config(profile).unwrap();
    let names = super::geometry::owner(profile).unwrap().attention_names;
    use AttentionPhase::{Decode, Prefill, Verify};
    let bf16 = Shape {
        q_heads: gated.q_heads,
        kv_heads: gated.kv_heads,
        qk_dim: g.head_dim(),
        v_dim: g.head_dim(),
        page_size: PAGE,
        kv_dtype: DType::Bf16,
        layout: CacheLayout::Paged,
        window_left: None,
        sink: false,
        softcap: None,
        causal: true,
    };
    let fp8 = Shape {
        kv_dtype: DType::Fp8E4M3,
        ..bf16
    };
    let native = |mut variant: Variant| {
        variant.query.max_context_capacity = g.context_tokens() as u32;
        if variant.query.phase == Decode {
            variant.gated_fold = Some(gated.fold());
        }
        variant
    };
    [
        variant(names[0], profile, Prefill, bf16),
        variant(names[1], profile, Decode, bf16),
        variant(names[2], profile, Verify, bf16),
        variant(names[3], profile, Prefill, fp8),
        variant(names[4], profile, Decode, fp8),
        variant(names[5], profile, Verify, fp8),
    ]
    .map(native)
    .into()
}

/// Every QSA entry: the replicated indexer's (TP4's profile, which serves
/// every topology), then each profile's attention entries. The rows of the
/// MTP drafter's exact verify steps (`EXACT_VERIFY_ROWS`) keep one-row bits.
pub fn sparse_catalog() -> Vec<sparse::Entry> {
    let mut entries = Vec::new();
    for model in super::geometry::MODELS {
        let profile = model.model.at(4);
        entries.extend(sparse::indexer_entries(
            profile,
            sparse_geometry(profile).unwrap(),
            EXACT_VERIFY_ROWS,
        ));
        for profile in model.model.profiles() {
            entries.extend(sparse::attention_entries(
                profile,
                sparse_geometry(profile).unwrap(),
                EXACT_VERIFY_ROWS,
            ));
        }
    }
    entries
}
