//! Qwen3.8 Flash's configuration values and per-rank sizes, in
//! `kernel_core::rank_shape`'s model-neutral terms.

use kernel_core::rank_shape::{
    AttentionConfig, Block, BlockClass, ExpertSplit, LinearAttentionConfig, ModelConfig, MoeConfig,
};

use crate::contract::{
    ATTN_HEAD, ATTN_Q_HEADS, CONV_TAPS, EXPERT_INTERMEDIATE, EXPERTS, HIDDEN, LAYERS, LINEAR_HEAD,
    LINEAR_KEY_HEADS, LINEAR_VALUE_HEADS, NGRAM_HEADS, SHARED_EXPERT_INTERMEDIATE, VOCAB,
};

// The delta engine has one dimension parameter for both key and value heads.
const _: () =
    assert!(crate::geometry::FLASH.linear_head() == crate::geometry::FLASH.linear_value_head());

/// 512 routed experts (top 10), expert parallel, plus a shared expert whose
/// columns split and pad to 128.
pub const MOE: MoeConfig = MoeConfig {
    experts: EXPERTS,
    top_k: crate::geometry::FLASH.top_k(),
    intermediate: EXPERT_INTERMEDIATE,
    split: ExpertSplit::Experts,
    shared_intermediate: SHARED_EXPERT_INTERMEDIATE,
    shared_tile: 128,
};

/// Every decoder layer, and the MTP drafter's, is MoE.
pub const CONFIG: ModelConfig = ModelConfig {
    hidden: HIDDEN,
    vocab: VOCAB,
    blocks: &[BlockClass {
        name: "moe",
        layers: (0, LAYERS),
        block: Block::Moe(MOE),
    }],
    linear: Some(LinearAttentionConfig {
        key_heads: LINEAR_KEY_HEADS,
        value_heads: LINEAR_VALUE_HEADS,
        head: LINEAR_HEAD,
        conv_taps: CONV_TAPS,
    }),
    ngram_heads: NGRAM_HEADS,
};

/// Rows of the verify steps whose rows Qwen3.8 serves exactly: the MTP
/// drafter's K3 steps (the bonus and three drafts) of one request
/// (docs/numerics.md#state-and-verification-contract). The entries those steps launch declare
/// one-row rows up to this many.
/// The qualified K3 verify path fixes the exact row count.
pub const EXACT_VERIFY_ROWS: u32 = 4;
/// Gated attention: 24 Q heads with an output gate over 2 KV heads of 256.
pub const ATTENTION: AttentionConfig = AttentionConfig {
    q_heads: ATTN_Q_HEADS,
    kv_heads: crate::geometry::FLASH.attention_kv_heads(),
    qk_head: ATTN_HEAD,
    v_head: ATTN_HEAD,
};

/// Qwen3.8 Flash's per-rank sizes at 2 or 4 ranks: heads, rows,
/// experts and vocabulary split evenly, one
/// KV head per rank, and the shared expert's columns padded to 128. A view
/// of [`RankShape::shape`] with Qwen's derived sizes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RankShape {
    pub ranks: u32,
}

impl RankShape {
    /// The model-neutral derivation from [`CONFIG`].
    pub const fn shape(self) -> kernel_core::rank_shape::RankShape {
        kernel_core::rank_shape::RankShape::unchecked(CONFIG, self.ranks)
    }

    /// Gated attention Q heads, the GQA group over the rank's one KV head.
    pub const fn q_heads(self) -> u32 {
        self.shape().attention(ATTENTION).q_heads
    }

    /// `q_proj` rows: each Q head's query then its gate.
    pub const fn q_proj_rows(self) -> u32 {
        self.q_heads() * 2 * ATTN_HEAD
    }

    /// The packed `q|gate, k, v` row of the attention input projection.
    pub const fn attn_row(self) -> u32 {
        let attention = self.shape().attention(ATTENTION);
        self.q_proj_rows() + attention.kv_heads * (attention.qk_head + attention.v_head)
    }

    /// `o_proj` K, the attention output width.
    pub const fn attn_value(self) -> u32 {
        self.q_heads() * ATTN_HEAD
    }

    pub const fn linear_key_heads(self) -> u32 {
        self.shape().linear_key_heads()
    }

    /// Linear attention value heads, the rows of in_proj_a and in_proj_b.
    pub const fn linear_value_heads(self) -> u32 {
        self.shape().linear_value_heads()
    }

    /// in_proj_qkv rows and convolution channels.
    pub const fn linear_qkv(self) -> u32 {
        (2 * self.linear_key_heads() + self.linear_value_heads()) * LINEAR_HEAD
    }

    /// in_proj_z rows, out_proj K and the gated norm width.
    pub const fn linear_value(self) -> u32 {
        self.linear_value_heads() * LINEAR_HEAD
    }

    /// Elements of a `lin_ab` row: a then b, padded to a multiple of 8.
    pub const fn lin_ab_stride(self) -> u32 {
        (2 * self.linear_value_heads()).next_multiple_of(8)
    }

    /// Routed experts of one rank, `[rank * local, rank * local + local)`.
    pub const fn local_experts(self) -> u32 {
        self.shape().experts(MOE).experts
    }

    /// The shared expert's columns of one rank, zero-padded to 128.
    pub const fn shared_padded(self) -> u32 {
        self.shape().experts(MOE).shared
    }

    /// `lm_head` rows of one rank.
    pub const fn vocab_shard(self) -> u32 {
        self.shape().vocab_shard()
    }

    /// N-gram hash heads of one rank: head `rank * local + y`.
    pub const fn ngram_heads(self) -> u32 {
        self.shape().ngram_heads()
    }

    /// Bytes of one linear layer's convolution state entry.
    pub const fn conv_entry_bytes(self) -> u64 {
        (CONV_TAPS as u64 - 1) * self.linear_qkv() as u64 * 2
    }

    /// Bytes of one linear layer's delta-rule state entry.
    pub const fn delta_entry_bytes(self) -> u64 {
        self.linear_value_heads() as u64 * (LINEAR_HEAD as u64 * LINEAR_HEAD as u64) * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The independent expected per-rank sizes at TP2 and TP4.
    #[test]
    fn rank_shapes_match_the_topology_table() {
        for (ranks, q, attn_row, qkv, experts, shared, vocab, conv, delta) in [
            (2, 12, 6656, 5120, 256, 384, 124_160, 30_720, 1_572_864),
            (4, 6, 3584, 2560, 128, 256, 62_080, 15_360, 786_432),
        ] {
            let shape = RankShape { ranks };
            shape.shape().validate().unwrap();
            let attention = shape.shape().validate_attention(ATTENTION).unwrap();
            assert_eq!(attention.kv_heads, 1);
            assert_eq!(shape.q_heads(), q);
            assert_eq!(shape.q_proj_rows(), q * 512);
            assert_eq!(shape.attn_row(), attn_row);
            assert_eq!(shape.attn_value(), q * 256);
            assert_eq!(shape.linear_qkv(), qkv);
            assert_eq!(shape.linear_value(), 48 / ranks * 128);
            assert_eq!(shape.lin_ab_stride(), (96 / ranks).next_multiple_of(8));
            assert_eq!(shape.local_experts(), experts);
            assert_eq!(shape.shared_padded(), shared);
            assert_eq!(shape.vocab_shard(), vocab);
            assert_eq!(shape.ngram_heads(), 16 / ranks);
            assert_eq!(shape.conv_entry_bytes(), conv);
            assert_eq!(shape.delta_entry_bytes(), delta);
        }
    }
}
