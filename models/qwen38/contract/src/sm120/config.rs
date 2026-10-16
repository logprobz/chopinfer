//! Qwen3.8 Flash's kernel profiles and the CUDA defines of its bundle rows;
//! the configuration values and per-rank sizes are `crate::rank`'s.

use kernels_sm120::{Model, Profile};

pub use crate::rank::{ATTENTION, CONFIG, EXACT_VERIFY_ROWS, MOE, RankShape};

/// Rank-invariant hyper-connection sizes used by the owned CUDA headers.
pub fn hc_defines() -> Vec<(String, String)> {
    super::geometry::hc_defines(crate::geometry::FLASH)
}

/// The PLE convolution's taps, dilation and derived history capacity.
pub fn ple_defines() -> Vec<(String, String)> {
    super::geometry::ple_defines(crate::geometry::FLASH)
}

/// The TP4 local-head default shared by CUDA and entry-body generation.
pub const NGRAM_HEADER_LOCAL: u32 = TP4.ngram_heads();

/// Hash geometry and the TP4 default that keeps the entry bodies' spelling.
pub fn ngram_defines() -> Vec<(String, String)> {
    super::geometry::ngram_defines(crate::geometry::FLASH)
}

/// The replicated model sizes the per-rank shapes split.
pub const LINEAR_KEY_HEADS: u32 = crate::contract::LINEAR_KEY_HEADS;
pub const LINEAR_VALUE_HEADS: u32 = crate::contract::LINEAR_VALUE_HEADS;
pub const LINEAR_HEAD: u32 = crate::contract::LINEAR_HEAD;
pub const EXPERTS: u32 = crate::contract::EXPERTS;
pub const SHARED_EXPERT_INTERMEDIATE: u32 = crate::contract::SHARED_EXPERT_INTERMEDIATE;
pub const VOCAB: u32 = crate::contract::VOCAB;
pub const NGRAM_HEADS: u32 = crate::contract::NGRAM_HEADS;
pub const CONV_TAPS: u32 = crate::contract::CONV_TAPS;
pub const HIDDEN: u32 = crate::contract::HIDDEN;
pub const EXPERT_INTERMEDIATE: u32 = crate::contract::EXPERT_INTERMEDIATE;

/// Decoder layers.
pub const LAYERS: u32 = crate::contract::LAYERS;

/// Qwen3.8 Flash's kernel profiles at TP2 and TP4.
pub const MODEL: Model = Model::new(
    kernels_sm120::profile_names!("qwen38-flash", "qwen38", 4, 2),
    CONFIG,
);

/// The rank-invariant and TP4 profile.
pub const PROFILE: Profile = MODEL.base();

/// The default TP4 profile's per-rank shape.
pub const TP4: RankShape = match profile_shape(PROFILE) {
    Some(shape) => shape,
    None => panic!("TP4 profile must have a supported rank shape"),
};

/// The profile for a supported tensor-parallel rank count.
pub const fn profile(ranks: u32) -> Option<Profile> {
    MODEL.try_at(ranks)
}

pub const ATTN_Q_HEADS: u32 = crate::contract::ATTN_Q_HEADS;
pub const ATTN_HEAD: u32 = crate::contract::ATTN_HEAD;

/// The per-rank shapes of a Qwen3.8 profile.
pub const fn profile_shape(profile: Profile) -> Option<RankShape> {
    match profile.ranks_of(MODEL) {
        Some(ranks @ (2 | 4)) => Some(RankShape { ranks }),
        _ => None,
    }
}

/// The CUDA defines of a rank count's bundle rows.
impl RankShape {
    /// The defines of the bundle's rows (`models/qwen38/bundle/cuda/
    /// qwen38_bundle.cuh`): geometry, state bytes and verify capacity.
    pub fn bundle_defines(self) -> Vec<(String, String)> {
        super::geometry::GeometryShape {
            geometry: crate::geometry::FLASH,
            ranks: self.ranks,
        }
        .bundle_defines()
    }

    /// The defines of the bundle's speculation rows (`qwen38_speculation.cuh`,
    /// which includes the bundle's rows): the rank count, one rank's n-gram
    /// hash heads and vocabulary shard, then [`RankShape::bundle_defines`].
    pub fn speculation_defines(self) -> Vec<(String, String)> {
        super::geometry::GeometryShape {
            geometry: crate::geometry::FLASH,
            ranks: self.ranks,
        }
        .speculation_defines()
    }
}
