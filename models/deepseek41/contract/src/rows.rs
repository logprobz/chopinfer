//! The row ops' contracts, which the bundle prepares and its SM120 module
//! registers entries for: the RMSNorms (F32 arithmetic, the weight applied
//! before the one BF16 rounding, eps 1e-20, `model.py:281-293`), the
//! vocabulary-sharded embedding gather and the embedding copied into every
//! residual stream.

use kernel_core::contract::{DType, RmsNormKind, RowQuery};
use kernel_core::rmsnorm::RmsNormRequirement;
use kernel_core::row_copy::{GatherByIndex, StreamBroadcast};

use crate::geometry::Geometry;

/// The alignment of every row view the bundle binds.
pub const ALIGNMENT_BYTES: u32 = 16;

/// The widths the program normalizes: the sublayer inputs and the final row
/// (hidden), the query-down output (`q_rank`) and the window key (the
/// latent).
pub const fn norm_widths(g: &Geometry) -> [u32; 3] {
    [
        g.shape.hidden,
        g.shape.attention.q_rank,
        g.shape.attention.head_dim,
    ]
}

/// The RMSNorm of rows `width` wide at `g`'s eps, its reduction order the
/// backend's.
pub fn norm(g: &Geometry, width: u32) -> RmsNormRequirement {
    RmsNormRequirement::bf16_backend_chosen(width, g.shape.eps as f32, RmsNormKind::Plain, false)
}

/// `g`'s final norm (F32-TQ): the hidden width's plain RMSNorm over the
/// unrounded F32 collapse (`hc::head`) into F32 rows, normalized without a
/// BF16 round, which the exact F32 LM head reads.
pub fn final_norm(g: &Geometry) -> RmsNormRequirement {
    RmsNormRequirement {
        input: DType::F32,
        output: DType::F32,
        ..norm(g, g.shape.hidden)
    }
}

/// The embedding gather of one rank's vocabulary shard at `ranks` ranks.
pub const fn embedding(g: &Geometry, ranks: u32) -> RowQuery {
    RowQuery::EmbeddingGather {
        vocab: g.shape.vocab / ranks,
        width: g.shape.hidden,
    }
}

/// The embedding copied into every residual stream.
pub const fn expand_streams(g: &Geometry) -> StreamBroadcast {
    StreamBroadcast {
        dtype: DType::Bf16,
        width: g.shape.hidden,
        streams: g.shape.hc.streams,
        alignment_bytes: ALIGNMENT_BYTES,
    }
}

/// A prefill's F32 final-norm rows that need logits, gathered by index into
/// the head's input.
pub const fn head_rows(g: &Geometry) -> GatherByIndex {
    GatherByIndex {
        dtype: DType::F32,
        width: g.shape.hidden,
        alignment_bytes: ALIGNMENT_BYTES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::FLASH;

    #[test]
    fn norms_round_once_after_the_weight_at_eps_1e_20() {
        let n = norm(&FLASH, 5120);
        assert_eq!(n.kind, RmsNormKind::Plain);
        assert_eq!(f32::from_bits(n.epsilon_bits), 1e-20);
        assert_eq!(
            (n.input, n.weight, n.output),
            (DType::Bf16, DType::Bf16, DType::Bf16)
        );
        assert!(!n.fused_residual);
        assert_eq!(norm_widths(&FLASH), [5120, 1280, 512]);
    }
}
