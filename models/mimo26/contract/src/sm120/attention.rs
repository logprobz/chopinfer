//! MiMo V2.6 Flash's paged attention variants on the shared engines
//! (`kernels_sm120::attention`): global attention (GA, BF16 and FP8 caches),
//! sliding-window attention (SWA) and the DFlash drafter's block attention,
//! with the exact verify rows entries of its decode variants.

use crate::geometry::V26_FLASH as GEOMETRY;

use super::geometry::{self, GeometryModel};
use kernel_core::contract::{AttentionPhase, DType};
use kernel_core::rank_shape::AttentionConfig;
use kernels_sm120::attention::{CacheLayout, Shape, Variant, variant};
use kernels_sm120::invariance::Invariance;
use kernels_sm120::names::intern;

use crate::sm120::config as mimo26;

/// Rotary width of the NeoX RoPE the decode variants fold in with the KV
/// write.
pub const ROPE_FOLD: u32 = GEOMETRY.rotary();

/// Global attention per TP4 rank.
const GA: Shape = Shape {
    q_heads: mimo26::GA.q_heads / crate::serving::RANKS,
    kv_heads: mimo26::GA.kv_heads / crate::serving::RANKS,
    qk_dim: mimo26::GA.qk_head,
    v_dim: mimo26::GA.v_head,
    page_size: crate::serving::PAGE_TOKENS as u32,
    kv_dtype: DType::Bf16,
    layout: CacheLayout::Paged,
    window_left: None,
    sink: GEOMETRY.semantics.attention.global_sink,
    softcap: None,
    causal: true,
};

/// Sliding-window attention per TP4 rank.
const SWA: Shape = Shape {
    q_heads: mimo26::SWA.q_heads / crate::serving::RANKS,
    kv_heads: mimo26::SWA.kv_heads / crate::serving::RANKS,
    qk_dim: mimo26::SWA.qk_head,
    v_dim: mimo26::SWA.v_head,
    page_size: GEOMETRY.window(),
    kv_dtype: DType::Bf16,
    layout: CacheLayout::Ring,
    window_left: Some(GEOMETRY.window() - 1),
    sink: GEOMETRY.semantics.attention.sliding_sink,
    softcap: None,
    causal: true,
};

/// The DFlash drafter's block attention per TP4 rank: a circular per-slot
/// context cache, a 1023-token window per query and a bidirectional block.
const DFLASH: Shape = Shape {
    q_heads: mimo26::DFLASH.q_heads / crate::serving::RANKS,
    kv_heads: mimo26::DFLASH.kv_heads / crate::serving::RANKS,
    qk_dim: mimo26::DFLASH.qk_head,
    v_dim: mimo26::DFLASH.v_head,
    page_size: crate::state::DFLASH_CACHE_TOKENS as u32,
    kv_dtype: DType::Bf16,
    layout: CacheLayout::Ring,
    window_left: Some(GEOMETRY.dflash().sliding_window - 1),
    sink: GEOMETRY.dflash().dflash_config.attention_sink_bias,
    softcap: None,
    causal: GEOMETRY.dflash().is_causal,
};

/// Every variant: TP4's, then TP2's.
pub fn catalog() -> Vec<Variant> {
    geometry::MODELS
        .iter()
        .flat_map(|owner| catalog_for(*owner))
        .collect()
}

/// A decode variant that folds its NeoX RoPE and KV write and has exact
/// verify rows entries.
fn folded(variant: Variant) -> Variant {
    Variant {
        rope_fold: Some(ROPE_FOLD),
        exact_rows: true,
        ..variant
    }
}

/// The variants at `tp`: TP4's shapes with the rank's Q heads and each
/// cache's KV heads (`RankShape::attention` over `mimo26::GA`, `SWA` and
/// `DFLASH`: one copied head when there are fewer heads than ranks). Every
/// group stays 16 or 8. Prefill and decode, then verify.
pub fn catalog_for(owner: GeometryModel) -> Vec<Variant> {
    owner
        .profiles()
        .into_iter()
        .flat_map(|p| variants(owner, p))
        .collect()
}

fn variants(owner: GeometryModel, profile: kernels_sm120::Profile) -> Vec<Variant> {
    use AttentionPhase::{Decode, Prefill, Verify};
    let g = owner.geometry;
    let tag = profile.tag();
    let rank = profile.rank_shape().expect("a MiMo profile");
    let heads = |shape: Shape, config: AttentionConfig| {
        let heads = rank.attention(config);
        Shape {
            q_heads: heads.q_heads,
            kv_heads: heads.kv_heads,
            ..shape
        }
    };
    let ga = heads(
        Shape {
            qk_dim: g.head_dim(),
            v_dim: g.value_head_dim(),
            sink: g.semantics.attention.global_sink,
            ..GA
        },
        geometry::ga(g),
    );
    let ga_fp8 = Shape {
        kv_dtype: DType::Fp8E4M3,
        ..ga
    };
    let swa = heads(
        Shape {
            qk_dim: g.swa_head_dim(),
            v_dim: g.swa_value_head_dim(),
            page_size: g.window(),
            window_left: Some(g.window() - 1),
            sink: g.semantics.attention.sliding_sink,
            ..SWA
        },
        geometry::swa(g),
    );
    let dflash = heads(
        Shape {
            qk_dim: g.dflash().head_dim,
            v_dim: g.dflash().v_head_dim,
            window_left: Some(g.dflash().sliding_window - 1),
            sink: g.dflash().dflash_config.attention_sink_bias,
            causal: g.dflash().is_causal,
            ..DFLASH
        },
        geometry::dflash(g),
    );
    // A paged entry reads an item's block-table row at
    // `max_context_capacity / page_size` entries per item, so it plans for
    // the geometry's context: the rows the bundle packs hold the geometry's
    // pages (`ModelLimits::max_pages`). A prefill row splits its keys over
    // fixed absolute tiles (GA every 8 tiles, SWA every 32 keys), so its
    // bits ignore the chunk split and the other items
    // (`attention_prefill_rows_ignore_chunking`).
    let declared = |mut variant: Variant| {
        if variant.layout == CacheLayout::Paged {
            variant.query.max_context_capacity = g.context_tokens();
        }
        match variant.query.phase {
            Prefill => Variant {
                invariance: Invariance::PREFILL,
                ..variant
            },
            _ => variant,
        }
    };
    let variants = vec![
        variant(
            intern(format!("{tag}_ga_bf16_prefill")),
            profile,
            Prefill,
            ga,
        ),
        folded(variant(
            intern(format!("{tag}_ga_bf16_decode")),
            profile,
            Decode,
            ga,
        )),
        variant(
            intern(format!("{tag}_ga_fp8_prefill")),
            profile,
            Prefill,
            ga_fp8,
        ),
        folded(variant(
            intern(format!("{tag}_ga_fp8_decode")),
            profile,
            Decode,
            ga_fp8,
        )),
        variant(intern(format!("{tag}_swa_prefill")), profile, Prefill, swa),
        folded(variant(
            intern(format!("{tag}_swa_decode")),
            profile,
            Decode,
            swa,
        )),
        variant(intern(format!("{tag}_ga_bf16_verify")), profile, Verify, ga),
        variant(
            intern(format!("{tag}_ga_fp8_verify")),
            profile,
            Verify,
            ga_fp8,
        ),
        variant(intern(format!("{tag}_swa_verify")), profile, Verify, swa),
        variant(
            intern(format!("{tag}_dflash_block")),
            profile,
            Verify,
            dflash,
        ),
    ];
    variants.into_iter().map(declared).collect()
}
