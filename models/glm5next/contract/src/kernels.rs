//! Per-rank kernel instance shapes of each compiled geometry.
//!
//! The geometry's compile-time fields fix these shapes; engines and the
//! bundle instantiate kernels from them. Each served rank count is checked
//! at compile time: a geometry that does not divide evenly fails the build.
use crate::geometry::Geometry;

/// FP8 weight block edge (`quantization_config.weight_block_size`).
pub const FP8_BLOCK: u32 = 128;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KernelProfile {
    pub geometry: &'static str,
    pub kernel_tag: &'static str,
    pub ranks: u32,
    pub hidden: u32,
    /// Latent attention query heads per rank (one key and value head each).
    pub latent_heads: u32,
    pub q_rank: u32,
    pub kv_rank: u32,
    pub qk_nope: u32,
    pub v_head: u32,
    /// KDA heads per rank.
    pub linear_heads: u32,
    pub linear_head_dim: u32,
    pub conv_taps: u32,
    /// The indexer is replicated: every rank scores all heads.
    pub indexer_heads: u32,
    pub indexer_dim: u32,
    pub index_pool: u32,
    pub index_topk_max: u32,
    pub hc_streams: u32,
    pub hc_width: u32,
    pub hc_mix: u32,
    /// Routed experts every rank runs: all of them, as each rank holds a
    /// slice of every expert.
    pub local_experts: u32,
    pub top_k: u32,
    /// Routed expert intermediate columns per rank.
    pub expert_intermediate: u32,
    /// Shared expert intermediate columns per rank.
    pub shared_intermediate: u32,
    /// Dense MLP intermediate columns per rank.
    pub dense_intermediate: u32,
    /// Vocabulary rows of the embedding and head per rank.
    pub vocab_rows: u32,
}

const fn split(total: u32, ranks: u32, granule: u32) -> u32 {
    assert!(
        total % (ranks * granule) == 0,
        "a geometry dimension does not split evenly across ranks"
    );
    total / ranks
}

impl KernelProfile {
    pub const fn new(g: &Geometry, ranks: u32) -> Self {
        let s = &g.shape;
        assert!(s.hidden % FP8_BLOCK == 0 && s.latent.q_rank % FP8_BLOCK == 0);
        assert!(s.latent.kv_rank % FP8_BLOCK == 0 && s.moe.intermediate % FP8_BLOCK == 0);
        assert!(s.latent.qk_rope == 0, "rotary latent keys are not compiled");
        assert!(
            s.moe.groups == 1 && s.moe.top_groups == 1,
            "grouped routing is not compiled"
        );
        let shared = s.moe.shared * s.moe.intermediate;
        Self {
            geometry: g.id,
            kernel_tag: g.kernel_tag,
            ranks,
            hidden: s.hidden,
            latent_heads: split(s.latent.heads, ranks, 1),
            q_rank: s.latent.q_rank,
            kv_rank: s.latent.kv_rank,
            qk_nope: s.latent.qk_nope,
            v_head: s.latent.v_head,
            linear_heads: split(s.linear.heads, ranks, 1),
            linear_head_dim: s.linear.head_dim,
            conv_taps: s.linear.conv_taps,
            indexer_heads: s.indexer.heads,
            indexer_dim: s.indexer.dim,
            index_pool: s.indexer.pool,
            index_topk_max: g.limits.index_topk,
            hc_streams: s.hc.streams,
            hc_width: g.hc_width(),
            hc_mix: g.hc_mix(),
            local_experts: s.moe.experts,
            top_k: s.moe.top_k,
            expert_intermediate: split(s.moe.intermediate, ranks, FP8_BLOCK),
            shared_intermediate: split(shared, ranks, FP8_BLOCK),
            dense_intermediate: split(s.dense.intermediate, ranks, FP8_BLOCK),
            vocab_rows: split(s.vocab, ranks, 1),
        }
    }
}

include!(concat!(env!("OUT_DIR"), "/kernel_profiles.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_tp4_instances() {
        let profile = PROFILES[0];
        assert_eq!((profile.geometry, profile.ranks), ("flash", 4));
        assert_eq!(profile.latent_heads, 16);
        assert_eq!(profile.linear_heads, 16);
        assert_eq!(profile.local_experts, 288);
        assert_eq!(profile.expert_intermediate, 512);
        assert_eq!(profile.indexer_heads, 32);
        assert_eq!(profile.shared_intermediate, 512);
        assert_eq!(profile.dense_intermediate, 3072);
        assert_eq!(profile.vocab_rows, 38720);
        assert_eq!(profile.hc_width, 16384);
        assert_eq!(profile.index_topk_max, 2048);
        assert_eq!(
            PROFILES.len(),
            crate::geometry::GEOMETRIES.len() * SERVED_RANKS.len()
        );
    }
}
