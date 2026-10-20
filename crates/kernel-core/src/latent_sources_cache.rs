//! Temporary decoded selected keys for latent source attention.
use crate::contract::Refusal;
use crate::latent_attention::LatentSourcesBuffers;
use std::ops::Range;

pub struct Buffers<B> {
    pub native: LatentSourcesBuffers<B>,
    pub decoded: B,
    pub valid: B,
}

/// Disjoint regions within the model's separately owned cache allocation.
pub struct Layout {
    pub decoded: Range<u64>,
    pub valid: Range<u64>,
    pub bytes: u64,
}

impl Layout {
    pub fn new(decoded: u64, valid: u64) -> Result<Self, Refusal> {
        if decoded == 0 || valid == 0 {
            return Err(Refusal("selected cache regions must be nonempty"));
        }
        let first = decoded
            .checked_add(255)
            .ok_or(Refusal("selected cache layout overflows"))?
            / 256
            * 256;
        let end = first
            .checked_add(valid)
            .ok_or(Refusal("selected cache layout overflows"))?;
        Ok(Self {
            decoded: 0..decoded,
            valid: first..end,
            bytes: end,
        })
    }
}
