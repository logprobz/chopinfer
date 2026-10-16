//! MiMo V2.6 Flash's attention geometry and bundle shape adapters.

use super::geometry;
use crate::geometry::Geometry;
use crate::geometry::V26_FLASH as GEOMETRY;

use kernel_core::rank_shape::AttentionConfig;
use kernels_sm120::RankShape;

use kernel_core::rank_shape::{
    Block, BlockClass, DenseConfig, ExpertSplit, ModelConfig, MoeConfig,
};
use kernels_sm120::{Model, Profile};

pub const HIDDEN: u32 = GEOMETRY.hidden();
pub const VOCAB: u32 = GEOMETRY.vocab();

/// 256 experts (top 8) of intermediate 2048, every expert on every rank
/// with its intermediate columns split.
pub const MOE: MoeConfig = MoeConfig {
    experts: GEOMETRY.experts(),
    top_k: GEOMETRY.top_k(),
    intermediate: GEOMETRY.expert_intermediate(),
    split: ExpertSplit::Intermediate,
    shared_intermediate: 0,
    shared_tile: 0,
};

/// Layer 0's dense MLP, which the MTP drafter layers share.
pub const DENSE: DenseConfig = DenseConfig {
    intermediate: GEOMETRY.dense_intermediate(),
};

/// Decoder layers.
pub const LAYERS: u32 = GEOMETRY.layers();

/// Layer 0 dense, the other decoder layers MoE.
pub const CONFIG: ModelConfig = ModelConfig {
    hidden: HIDDEN,
    vocab: VOCAB,
    blocks: &[
        BlockClass {
            name: "dense",
            layers: (0, GEOMETRY.dense_layer_end()),
            block: Block::Dense(DENSE),
        },
        BlockClass {
            name: "moe",
            layers: (GEOMETRY.dense_layer_end(), LAYERS),
            block: Block::Moe(MOE),
        },
    ],
    linear: None,
    ngram_heads: 0,
};

/// MiMo V2.6 Flash's kernel profile names at TP4, its base, and TP2.
const NAMES: &[(u32, kernels_sm120::ProfileNames)] = &[
    (
        4,
        kernels_sm120::ProfileNames {
            name: kernels_sm120::const_name!("mimo-", GEOMETRY.id),
            tag: GEOMETRY.kernel_tag,
        },
    ),
    (
        2,
        kernels_sm120::ProfileNames {
            name: kernels_sm120::const_name!("mimo-", GEOMETRY.id, "-tp2"),
            tag: kernels_sm120::const_name!(GEOMETRY.kernel_tag, "_tp2"),
        },
    ),
];

/// MiMo V2.6 Flash's kernel profiles at TP4 and TP2.
pub const MODEL: Model = Model::new(NAMES, CONFIG);

/// The rank-invariant and TP4 profile.
pub const PROFILE: Profile = MODEL.base();

/// The profile for a supported tensor-parallel rank count.
pub const fn profile(ranks: u32) -> Option<Profile> {
    MODEL.try_at(ranks)
}

/// Global attention: 64 Q heads (qk 192, v 128) over 4 KV heads.
pub const GA: AttentionConfig = AttentionConfig {
    q_heads: GEOMETRY.attention_q_heads(),
    kv_heads: GEOMETRY.attention_kv_heads(),
    qk_head: GEOMETRY.head_dim(),
    v_head: GEOMETRY.value_head_dim(),
};

/// Sliding-window attention, and the MTP layer's: GA's heads over 8 KV heads.
pub const SWA: AttentionConfig = AttentionConfig {
    q_heads: GEOMETRY.swa_q_heads(),
    kv_heads: GEOMETRY.swa_kv_heads(),
    qk_head: GEOMETRY.swa_head_dim(),
    v_head: GEOMETRY.swa_value_head_dim(),
};

/// The DFlash drafter's attention: 64 Q and 8 KV heads of 128.
pub const DFLASH: AttentionConfig = AttentionConfig {
    q_heads: GEOMETRY.dflash().num_attention_heads,
    kv_heads: GEOMETRY.dflash().num_key_value_heads,
    qk_head: GEOMETRY.dflash().head_dim,
    v_head: GEOMETRY.dflash().v_head_dim,
};

/// The DFlash drafter's `fc` input: five target layers' hidden states.
pub const DFLASH_CONTEXT: u32 = crate::tp::DFLASH_FC_K as u32;

/// SWA layers, and ring slots and positions of the bundle's SWA ring state
/// (`models/mimo26/bundle/cuda/mimo_ring.cuh`) before other slot counts.
/// Policy: size the default instance ring.
const RING_SLOTS: u64 = crate::serving::DECODE_ENTRY_ROWS as u64;

/// Policy: row copies, mask fills and retained-row transfers use eight warps.
pub const ROW_THREADS: u32 = 256;
/// Policy: prefix copies use eight warps over 16-byte vectors.
pub const PREFIX_THREADS: u32 = 256;
/// Policy: one warp selects draft tokens from gathered rank scores.
pub const DFLASH_DRAFT_THREADS: u32 = 32;

/// The drafter's geometry and cache-capacity policy for one topology.
pub fn dflash_defines_for(g: Geometry, shape: RankShape) -> Vec<(String, String)> {
    vec![
        ("CHOP_MIMO_RANKS".into(), shape.ranks().to_string()),
        (
            "CHOP_MIMO_DFLASH_KV_HEADS".into(),
            shape.attention(geometry::dflash(g)).kv_heads.to_string(),
        ),
        (
            "CHOP_MIMO_DFLASH_HIDDEN".into(),
            g.dflash().hidden_size.to_string(),
        ),
        (
            "CHOP_MIMO_DFLASH_HEAD_DIM".into(),
            geometry::dflash(g).qk_head.to_string(),
        ),
        (
            "CHOP_MIMO_DFLASH_CACHE_TOKENS".into(),
            crate::state::DFLASH_CACHE_TOKENS.to_string(),
        ),
        (
            "CHOP_MIMO_DFLASH_SLOTS".into(),
            crate::serving::DECODE_BATCH.to_string(),
        ),
    ]
}

/// Threads of one ring backup or restore block at `shape`'s rank count: one
/// per 16-byte vector of a K and V entry, at least 96.
pub fn ring_threads(shape: RankShape) -> u32 {
    ring_threads_for(GEOMETRY, shape)
}

pub fn ring_threads_for(g: Geometry, shape: RankShape) -> u32 {
    (shape.attention(geometry::swa(g)).kv_heads
        * (geometry::swa(g).qk_head + geometry::swa(g).v_head)
        * 2
        / 16)
        .max(96)
}

/// The ring cubins' SWA dimensions, layers, tokens, slots, working bytes
/// and ring and prefix thread counts at `shape`'s rank count.
pub fn ring_defines(shape: RankShape) -> Vec<(String, String)> {
    ring_defines_for(GEOMETRY, shape)
}

pub fn ring_defines_for(g: Geometry, shape: RankShape) -> Vec<(String, String)> {
    let heads = shape.attention(geometry::swa(g)).kv_heads;
    let entry =
        u64::from(heads) * u64::from(geometry::swa(g).qk_head + geometry::swa(g).v_head) * 2;
    let working = (g.sliding_layers() as u64) * RING_SLOTS * u64::from(g.window()) * entry;
    vec![
        ("CHOP_MIMO_SWA_KV_HEADS".into(), heads.to_string()),
        (
            "CHOP_MIMO_RING_TOKENS".into(),
            u64::from(g.window()).to_string(),
        ),
        ("CHOP_MIMO_RING_SLOTS".into(), RING_SLOTS.to_string()),
        (
            "CHOP_MIMO_SWA_LAYERS".into(),
            (g.sliding_layers() as u64).to_string(),
        ),
        (
            "CHOP_MIMO_SWA_QK_DIM".into(),
            geometry::swa(g).qk_head.to_string(),
        ),
        (
            "CHOP_MIMO_SWA_V_DIM".into(),
            geometry::swa(g).v_head.to_string(),
        ),
        (
            "CHOP_MIMO_PREFIX_THREADS".into(),
            PREFIX_THREADS.to_string(),
        ),
        (
            "CHOP_MIMO_RING_WORKING_BYTES".into(),
            format!("{working}ull"),
        ),
        (
            "CHOP_MIMO_RING_THREADS".into(),
            ring_threads_for(g, shape).to_string(),
        ),
    ]
}

// The target row and TP engines share Q-head and head-width layouts for GA and SWA.
const _: () =
    assert!(GA.q_heads == SWA.q_heads && GA.qk_head == SWA.qk_head && GA.v_head == SWA.v_head);
// The DFlash adapters share target hidden rows and equal-width K/V cache heads.
const _: () = assert!(GEOMETRY.dflash().hidden_size == HIDDEN && DFLASH.qk_head == DFLASH.v_head);
// The MTP EH projection shares the target output kernel only at this equal K width.
const _: () = assert!(crate::tp::MTP_EH_K == crate::tp::Q_HEADS * crate::tp::V_DIM);

#[cfg(test)]
mod tests {
    use super::*;

    /// docs/topology-mimo.md's per-rank table.
    #[test]
    fn rank_shapes_match_the_topology_table() {
        for (tp, q, ga, swa, dense, expert, vocab, fc, dflash_kv) in [
            (2, 32, 2, 4, 8192, 1024, 76_288, 10_240, 4),
            (4, 16, 1, 2, 4096, 512, 38_144, 5_120, 2),
        ] {
            let shape = crate::sm120::config::MODEL.at(tp).rank_shape().unwrap();
            shape.validate().unwrap();
            let rank_ga = shape.validate_attention(GA).unwrap();
            let rank_swa = shape.validate_attention(SWA).unwrap();
            let rank_dflash = shape.validate_attention(DFLASH).unwrap();
            assert_eq!((rank_ga.q_heads, rank_ga.kv_heads), (q, ga));
            assert_eq!((rank_swa.q_heads, rank_swa.kv_heads), (q, swa));
            assert_eq!((rank_dflash.q_heads, rank_dflash.kv_heads), (q, dflash_kv));
            assert_eq!(shape.dense(DENSE).intermediate, dense);
            assert_eq!(shape.experts(MOE).experts, 256);
            assert_eq!(shape.experts(MOE).intermediate, expert);
            assert_eq!(shape.vocab_shard(), vocab);
            assert_eq!(shape.split(DFLASH_CONTEXT), fc);
        }
    }
}
