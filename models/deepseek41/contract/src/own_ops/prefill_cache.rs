//! Temporary decoded operands for native prefill attention.
use kernel_core::backend::View;
use kernel_core::latent_attention::LatentSourcesBuffers;
use std::ops::Range;

pub struct Buffers {
    pub native: LatentSourcesBuffers<View>,
    pub window: View,
    pub compressed: View,
    pub marks: View,
}

/// Guarded regions within one separately owned allocation.
pub struct Layout {
    pub window: Range<u64>,
    pub compressed: Range<u64>,
    pub marks: Range<u64>,
    pub bytes: u64,
}
