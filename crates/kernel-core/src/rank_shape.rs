//! Per-rank shapes derived from a model's configuration values and its
//! tensor-parallel rank count.
//!
//! A model states its configuration once as a [`ModelConfig`] (hidden size,
//! vocabulary, and its feed-forward block classes: which layers run a dense
//! MLP or which mixture of experts). [`RankShape`] derives every per-rank
//! size from it: kernel catalogs compile these sizes ahead of time, and host
//! placement code shards weights with the same arithmetic. The rank count is
//! the tensor-parallel degree, not the GPU count.
//!
//! The derivation is model-neutral and has no CUDA dependency. Which rank
//! counts have compiled kernels is the kernel base's decision.

use crate::contract::Refusal;
use crate::topology::TensorParallel;

/// One attention kind's heads and head sizes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionConfig {
    pub q_heads: u32,
    pub kv_heads: u32,
    pub qk_head: u32,
    pub v_head: u32,
}

impl AttentionConfig {
    /// Q heads per KV head.
    pub const fn group(self) -> u32 {
        self.q_heads / self.kv_heads
    }
}

/// How a mixture of experts places its routed experts on the ranks.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ExpertSplit {
    /// Expert parallel: each rank holds whole experts, `experts / ranks`
    /// of them.
    Experts,
    /// Tensor parallel: every rank holds every expert, with the
    /// intermediate columns split over the ranks.
    Intermediate,
}

/// A dense MLP's configuration values: its intermediate columns split over
/// the ranks.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct DenseConfig {
    pub intermediate: u32,
}

/// A mixture of experts' configuration values and placement.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct MoeConfig {
    pub experts: u32,
    pub top_k: u32,
    /// One routed expert's intermediate size.
    pub intermediate: u32,
    pub split: ExpertSplit,
    /// The shared expert's intermediate size; zero without one. Its columns
    /// split over the ranks, padded to [`MoeConfig::shared_tile`].
    pub shared_intermediate: u32,
    pub shared_tile: u32,
}

/// One rank's part of a mixture of experts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RankExperts {
    /// Routed experts on the rank.
    pub experts: u32,
    pub top_k: u32,
    /// One local expert's intermediate columns.
    pub intermediate: u32,
    /// The shared expert's columns on the rank, padded to its tile; zero
    /// without one.
    pub shared: u32,
}

/// A feed-forward block's kind and configuration values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Block {
    Dense(DenseConfig),
    Moe(MoeConfig),
}

/// A class of blocks with one configuration: the decoder layers
/// `layers.0..layers.1` that run it, by name. A drafter or another stack
/// outside the decoder layers refers to its class by name; such a class may
/// cover no decoder layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BlockClass {
    pub name: &'static str,
    pub layers: (u32, u32),
    pub block: Block,
}

/// Linear (gated delta rule) attention's heads.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LinearAttentionConfig {
    pub key_heads: u32,
    pub value_heads: u32,
    pub head: u32,
    pub conv_taps: u32,
}

/// The configuration values the per-rank shapes split. Absent parts are
/// zero, `None` or not listed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ModelConfig {
    pub hidden: u32,
    pub vocab: u32,
    /// The feed-forward block classes. Their names are distinct and their
    /// decoder layer ranges disjoint.
    pub blocks: &'static [BlockClass],
    pub linear: Option<LinearAttentionConfig>,
    /// N-gram hash heads, split over the ranks; zero without them.
    pub ngram_heads: u32,
}

impl ModelConfig {
    /// The block class named `name`.
    pub fn block(self, name: &str) -> Option<BlockClass> {
        self.blocks.iter().copied().find(|class| class.name == name)
    }
}

/// One rank's sizes under `ranks`-way tensor parallelism.
///
/// Heads, dense and routed columns, experts and the vocabulary split evenly.
/// KV heads split while there are at least as many as ranks; with fewer,
/// each rank holds one copied head. Construct through [`RankShape::new`],
/// which refuses a split with a remainder.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RankShape {
    config: ModelConfig,
    ranks: u32,
}

impl RankShape {
    /// Refuse rank counts [`TensorParallel`] refuses and any configured
    /// size the rank count does not divide.
    pub fn new(config: ModelConfig, ranks: u32) -> Result<Self, Refusal> {
        TensorParallel::new(ranks as usize)?;
        let shape = Self { config, ranks };
        shape.validate()?;
        Ok(shape)
    }

    /// The shape without validation, for compile-time constants. Validate
    /// such a constant in a test with [`RankShape::validate`].
    pub const fn unchecked(config: ModelConfig, ranks: u32) -> Self {
        Self { config, ranks }
    }

    /// Check that every configured split is exact and that the block
    /// classes have distinct names and disjoint decoder layers.
    pub fn validate(self) -> Result<(), Refusal> {
        let c = self.config;
        let even = |total: u32| total.is_multiple_of(self.ranks);
        if self.ranks == 0 || !even(c.vocab) || !even(c.ngram_heads) {
            return Err(Refusal("rank count does not divide the model's sizes"));
        }
        for (i, class) in c.blocks.iter().enumerate() {
            let (start, end) = class.layers;
            if start > end {
                return Err(Refusal("a block class's layer range is reversed"));
            }
            for other in &c.blocks[..i] {
                if other.name == class.name {
                    return Err(Refusal("two block classes share a name"));
                }
                // An empty range holds no layer, so it overlaps nothing in
                // either declaration order.
                let nonempty = start < end && other.layers.0 < other.layers.1;
                if nonempty && other.layers.0 < end && start < other.layers.1 {
                    return Err(Refusal("two block classes share a layer"));
                }
            }
            match class.block {
                Block::Dense(dense) if !even(dense.intermediate) => {
                    return Err(Refusal("rank count does not divide the dense MLP"));
                }
                Block::Moe(moe) => {
                    let split = match moe.split {
                        ExpertSplit::Experts => moe.experts,
                        ExpertSplit::Intermediate => moe.intermediate,
                    };
                    if !even(split) || (moe.shared_intermediate != 0 && moe.shared_tile == 0) {
                        return Err(Refusal("rank count does not divide the experts"));
                    }
                }
                Block::Dense(_) => {}
            }
        }
        if let Some(linear) = c.linear
            && (!even(linear.key_heads) || !even(linear.value_heads))
        {
            return Err(Refusal("rank count does not divide the linear heads"));
        }
        Ok(())
    }

    pub const fn ranks(self) -> u32 {
        self.ranks
    }

    pub const fn config(self) -> ModelConfig {
        self.config
    }

    /// `total` columns, rows or heads split evenly. Panics (a compile error
    /// in const context) when the rank count does not divide `total`; use
    /// [`RankShape::padded_split`] for an intentionally padded split.
    pub const fn split(self, total: u32) -> u32 {
        assert!(
            total.is_multiple_of(self.ranks),
            "rank count does not divide the split size"
        );
        total / self.ranks
    }

    /// [`RankShape::split`], refusing a remainder.
    pub fn checked_split(self, total: u32) -> Result<u32, Refusal> {
        if !total.is_multiple_of(self.ranks) {
            return Err(Refusal("rank count does not divide the split size"));
        }
        Ok(total / self.ranks)
    }

    /// `total` columns split into `ceil(total / ranks)`, then padded to a
    /// multiple of `tile`: the padding rule is explicit at the call.
    pub const fn padded_split(self, total: u32, tile: u32) -> u32 {
        assert!(tile != 0, "padding tile must be nonzero");
        total.div_ceil(self.ranks).next_multiple_of(tile)
    }

    /// Why an attention kind does not split over the ranks, if it does not.
    /// Heads are nonzero, the Q heads form whole GQA groups, and each rank's
    /// contiguous Q shard reads exactly one run of KV heads: with at least
    /// as many KV heads as ranks both split evenly; with fewer, each rank's Q
    /// shard lies inside one group and the rank holds a copy of its head
    /// (head `rank * kv_heads / ranks`, as `TensorParallel::kv_heads`).
    const fn attention_error(self, config: AttentionConfig) -> Option<&'static str> {
        let (q, kv, ranks) = (config.q_heads, config.kv_heads, self.ranks);
        if q == 0 || kv == 0 {
            return Some("attention heads must be nonzero");
        }
        if !q.is_multiple_of(kv) {
            return Some("Q heads do not form whole GQA groups");
        }
        if !q.is_multiple_of(ranks) {
            return Some("rank count does not divide the Q heads");
        }
        if kv >= ranks {
            if !kv.is_multiple_of(ranks) {
                return Some("rank count does not divide the KV heads");
            }
        } else if !(q / kv).is_multiple_of(q / ranks) {
            return Some("a rank's Q heads straddle GQA groups");
        }
        None
    }

    /// One rank's heads of an attention kind: Q heads split, KV heads split
    /// down to one copied head. Panics (a compile error in const context)
    /// where [`RankShape::validate_attention`] refuses.
    pub const fn attention(self, config: AttentionConfig) -> AttentionConfig {
        if let Some(error) = self.attention_error(config) {
            panic!("{}", error);
        }
        AttentionConfig {
            q_heads: config.q_heads / self.ranks,
            kv_heads: if config.kv_heads >= self.ranks {
                config.kv_heads / self.ranks
            } else {
                1
            },
            ..config
        }
    }

    /// Check that an attention kind splits into whole groups, and return
    /// one rank's heads.
    pub fn validate_attention(self, config: AttentionConfig) -> Result<AttentionConfig, Refusal> {
        match self.attention_error(config) {
            Some(error) => Err(Refusal(error)),
            None => Ok(self.attention(config)),
        }
    }

    pub const fn vocab_shard(self) -> u32 {
        self.split(self.config.vocab)
    }

    /// One rank's dense MLP: its intermediate columns split.
    pub const fn dense(self, config: DenseConfig) -> DenseConfig {
        DenseConfig {
            intermediate: self.split(config.intermediate),
        }
    }

    /// One rank's part of a mixture of experts: whole experts under
    /// [`ExpertSplit::Experts`], every expert's split intermediate columns
    /// under [`ExpertSplit::Intermediate`], and the shared expert's columns
    /// padded to its tile.
    pub const fn experts(self, config: MoeConfig) -> RankExperts {
        let (experts, intermediate) = match config.split {
            ExpertSplit::Experts => (self.split(config.experts), config.intermediate),
            ExpertSplit::Intermediate => (config.experts, self.split(config.intermediate)),
        };
        RankExperts {
            experts,
            top_k: config.top_k,
            intermediate,
            shared: if config.shared_intermediate != 0 {
                self.padded_split(config.shared_intermediate, config.shared_tile)
            } else {
                0
            },
        }
    }

    pub const fn linear_key_heads(self) -> u32 {
        match self.config.linear {
            Some(linear) => self.split(linear.key_heads),
            None => 0,
        }
    }

    pub const fn linear_value_heads(self) -> u32 {
        match self.config.linear {
            Some(linear) => self.split(linear.value_heads),
            None => 0,
        }
    }

    pub const fn ngram_heads(self) -> u32 {
        self.split(self.config.ngram_heads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MOE: MoeConfig = MoeConfig {
        experts: 512,
        top_k: 10,
        intermediate: 640,
        split: ExpertSplit::Experts,
        shared_intermediate: 640,
        shared_tile: 128,
    };
    const DENSE: DenseConfig = DenseConfig {
        intermediate: 16384,
    };
    const CONFIG: ModelConfig = ModelConfig {
        hidden: 2560,
        vocab: 248_320,
        blocks: &[
            BlockClass {
                name: "dense",
                layers: (0, 1),
                block: Block::Dense(DENSE),
            },
            BlockClass {
                name: "moe",
                layers: (1, 48),
                block: Block::Moe(MOE),
            },
        ],
        linear: Some(LinearAttentionConfig {
            key_heads: 16,
            value_heads: 48,
            head: 128,
            conv_taps: 4,
        }),
        ngram_heads: 16,
    };
    const GQA: AttentionConfig = AttentionConfig {
        q_heads: 64,
        kv_heads: 4,
        qk_head: 192,
        v_head: 128,
    };

    #[test]
    fn splits_follow_the_rank_count() {
        for (ranks, local, shared, kv, vocab) in [
            (1, 512, 640, 4, 248_320),
            (2, 256, 384, 2, 124_160),
            (4, 128, 256, 1, 62_080),
            (8, 64, 128, 1, 31_040),
        ] {
            let shape = RankShape::new(CONFIG, ranks).unwrap();
            assert_eq!(shape.ranks(), ranks);
            let experts = shape.experts(MOE);
            assert_eq!((experts.experts, experts.top_k), (local, 10));
            assert_eq!(experts.intermediate, 640);
            assert_eq!(experts.shared, shared);
            assert_eq!(shape.vocab_shard(), vocab);
            assert_eq!(shape.dense(DENSE).intermediate, 16384 / ranks);
            assert_eq!(shape.linear_key_heads(), 16 / ranks);
            assert_eq!(shape.linear_value_heads(), 48 / ranks);
            assert_eq!(shape.ngram_heads(), 16 / ranks);
            let heads = shape.validate_attention(GQA).unwrap();
            assert_eq!((heads.q_heads, heads.kv_heads), (64 / ranks, kv));
            assert_eq!((heads.qk_head, heads.v_head), (192, 128));
        }
    }

    #[test]
    fn intermediate_split_keeps_every_expert() {
        let moe = MoeConfig {
            experts: 256,
            top_k: 8,
            intermediate: 2048,
            split: ExpertSplit::Intermediate,
            shared_intermediate: 0,
            shared_tile: 0,
        };
        for ranks in [2, 4, 8] {
            let experts = RankShape::new(CONFIG, ranks).unwrap().experts(moe);
            assert_eq!(experts.experts, 256);
            assert_eq!(experts.intermediate, 2048 / ranks);
            assert_eq!(experts.shared, 0);
        }
    }

    /// Two expert classes in one model (a drafter's layers with fewer
    /// experts and a smaller top-k than the target's) split independently.
    #[test]
    fn expert_classes_split_independently() {
        const TARGET: MoeConfig = MoeConfig {
            experts: 384,
            top_k: 6,
            intermediate: 3072,
            split: ExpertSplit::Experts,
            shared_intermediate: 3072,
            shared_tile: 128,
        };
        const DRAFTER: MoeConfig = MoeConfig {
            experts: 128,
            top_k: 3,
            ..TARGET
        };
        let (target, drafter) = (TARGET, DRAFTER);
        let config = ModelConfig {
            blocks: &[
                BlockClass {
                    name: "target",
                    layers: (0, 40),
                    block: Block::Moe(TARGET),
                },
                BlockClass {
                    name: "drafter",
                    layers: (40, 40),
                    block: Block::Moe(DRAFTER),
                },
            ],
            ..CONFIG
        };
        assert_eq!(config.block("drafter").unwrap().block, Block::Moe(drafter));
        assert_eq!(config.block("dense"), None);
        let shape = RankShape::new(config, 4).unwrap();
        assert_eq!(
            (shape.experts(target).experts, shape.experts(target).top_k),
            (96, 6)
        );
        assert_eq!(
            (shape.experts(drafter).experts, shape.experts(drafter).top_k),
            (32, 3)
        );
        assert_eq!(shape.experts(drafter).shared, 768);
        // 384 experts split over 8 ranks; 128 do not split over 256.
        assert!(RankShape::new(config, 8).is_ok());
        assert!(RankShape::unchecked(config, 256).validate().is_err());
    }

    #[test]
    fn refuses_inconsistent_block_classes() {
        let class = |name, layers, intermediate| BlockClass {
            name,
            layers,
            block: Block::Dense(DenseConfig { intermediate }),
        };
        for blocks in [
            &[class("a", (0, 2), 64), class("a", (2, 4), 64)][..],
            &[class("a", (0, 2), 64), class("b", (1, 4), 64)][..],
            &[class("a", (2, 1), 64)][..],
            &[class("a", (0, 2), 66)][..],
        ] {
            let blocks: &'static [BlockClass] = Box::leak(blocks.to_vec().into_boxed_slice());
            let config = ModelConfig { blocks, ..CONFIG };
            assert!(RankShape::new(config, 4).is_err(), "{blocks:?}");
        }
        // Empty ranges (drafter classes outside the decoder layers) share no
        // layer, whichever class is declared first and wherever they lie.
        for blocks in [
            &[
                class("a", (0, 2), 64),
                class("b", (2, 4), 64),
                class("c", (4, 4), 64),
            ][..],
            &[class("d", (1, 1), 64), class("a", (0, 4), 64)][..],
            &[class("a", (0, 4), 64), class("d", (1, 1), 64)][..],
            &[class("d", (2, 2), 64), class("e", (2, 2), 64)][..],
        ] {
            let blocks: &'static [BlockClass] = Box::leak(blocks.to_vec().into_boxed_slice());
            assert!(
                RankShape::new(ModelConfig { blocks, ..CONFIG }, 4).is_ok(),
                "{blocks:?}"
            );
        }
    }

    #[test]
    fn refuses_inexact_splits() {
        assert!(RankShape::new(CONFIG, 3).is_err());
        assert!(RankShape::new(CONFIG, 32).is_err());
        let odd_vocab = ModelConfig {
            vocab: 248_321,
            ..CONFIG
        };
        assert!(RankShape::new(odd_vocab, 2).is_err());
        let odd_linear = ModelConfig {
            linear: Some(LinearAttentionConfig {
                key_heads: 6,
                value_heads: 48,
                head: 128,
                conv_taps: 4,
            }),
            ..CONFIG
        };
        assert!(RankShape::new(odd_linear, 4).is_err());
        assert!(RankShape::unchecked(odd_linear, 4).validate().is_err());
        let shape = RankShape::new(CONFIG, 16).unwrap();
        // 64 / 16 = 4 Q heads over one copied KV head is whole; 6 KV heads
        // over 4 ranks is not.
        assert!(shape.validate_attention(GQA).is_ok());
        let uneven = AttentionConfig {
            kv_heads: 6,
            q_heads: 48,
            ..GQA
        };
        assert!(
            RankShape::new(CONFIG, 4)
                .unwrap()
                .validate_attention(uneven)
                .is_err()
        );
    }

    /// REVIEW-A4 finding 3: an even split refuses a remainder instead of
    /// truncating.
    #[test]
    #[should_panic(expected = "does not divide")]
    fn split_refuses_a_remainder() {
        RankShape::new(CONFIG, 8).unwrap().split(25);
    }

    #[test]
    fn checked_split_reports_a_remainder() {
        let shape = RankShape::new(CONFIG, 8).unwrap();
        assert_eq!(shape.checked_split(24), Ok(3));
        assert!(shape.checked_split(25).is_err());
        assert_eq!(shape.padded_split(25, 1), 4);
    }

    /// REVIEW-A4 finding 2: a rank's contiguous query shard must lie in one
    /// GQA group, and zero heads are refused.
    #[test]
    fn attention_refuses_shards_that_straddle_groups() {
        let shape = RankShape::new(CONFIG, 8).unwrap();
        // 8-query groups; a 3-query shard of rank 2 (queries 6..9) spans
        // KV heads 0 and 1.
        let straddle = AttentionConfig {
            q_heads: 24,
            kv_heads: 3,
            qk_head: 128,
            v_head: 128,
        };
        assert!(shape.validate_attention(straddle).is_err());
        for (q_heads, kv_heads) in [(64, 0), (0, 4), (24, 5), (12, 8)] {
            let config = AttentionConfig {
                q_heads,
                kv_heads,
                ..straddle
            };
            assert!(
                shape.validate_attention(config).is_err(),
                "{q_heads} Q over {kv_heads} KV"
            );
        }
        // 24 Q over 2 KV, and 64 over 4, 8 or 16, split into whole groups.
        for (q_heads, kv_heads, rank_kv) in [(24, 2, 1), (64, 4, 1), (64, 8, 1), (64, 16, 2)] {
            let config = AttentionConfig {
                q_heads,
                kv_heads,
                ..straddle
            };
            let rank = shape.validate_attention(config).unwrap();
            assert_eq!((rank.q_heads, rank.kv_heads), (q_heads / 8, rank_kv));
        }
    }
}
