//! The media encoder stages' buffers by role, for any backend that prepares
//! them (`crate::backend::Backend`'s multimodal encoder methods).
//!
//! Each stage is a [`MultimodalQuery`]. Its fields are math the model
//! states (widths, heads, masks, epsilons, limits, epilogues, the F32-pair
//! splits the reference accuracy needs), except `capacity` (a GEMM's
//! `m_capacity`): the rows a backend sizes its launches for, which a model
//! states as `Need::capacity` and a backend may serve from an entry
//! prepared for more rows. Tiles, threads, grids and reduction orders are a
//! backend's choice and no field of the contract. Every launch counts its
//! live rows on the device (`live_rows`), so one launch serves any row
//! count up to the capacity.
//!
//! # Variants by record
//!
//! [`EncoderStage::of`] names the record each variant's operands fill, and
//! so the `Backend` method that prepares it; a method refuses a variant of
//! another stage. `T` is the step's rows (`Counts::rows`), `r` a role the
//! op reads, `w` one it writes and `rw` one it rewrites in place. Values
//! are dense and row-major. A role not listed for a variant stays absent;
//! a backend refuses a passed role its variant does not read, except a
//! GEMM's bias (below).
//!
//! Rows (`EncoderRowBuffers`, `Backend::encoder_rows`):
//!
//! | Variant | Roles |
//! |---|---|
//! | `LayerNormF32 { width }` | `input` r F32 `[T, width]`; `weight`, `bias` r BF16 `[width]`; `output` w F32 `[T, width]` |
//! | `RmsNormF32 { width }` | `input` r F32 `[T, width]`; `weight` r BF16 `[width]`; `output` w F32 `[T, width]` |
//! | `InterpolatedPositionAddF32 { width, table_rows }` | `input` r F32 `[T, width]`; `table` r BF16 `[table_rows, width]`; `indices` r i32 `[T, 4]` (table rows); `blend` r F32 `[T, 4]`; `output` w F32 `[T, width]` |
//! | `PermuteF32 { width }` | `input` r F32 `[T, width]`; `indices` r i32 `[T]`, a permutation; `output` w F32 `[T, width]`, row `r` input row `indices[r]` |
//! | `GatherRowsF32 { width, group }` | `input` r F32 `[S, width]`; `indices` r i32 `[T, group]`; `output` w F32 `[T, group * width]`, block `j` of row `r` input row `indices[r * group + j]`, or zeros where it is negative; every other index is below `S` |
//! | `ResidualAddF32 { width }` | `input`, `residual` r F32 `[T, width]`; `output` w F32 `[T, width]` |
//! | `SwigluF32 { hidden }`, `SwigluClampedF32 { hidden }` | `input` r F32 `[T, 2 hidden]`, gate half first; `output` w F32 `[T, hidden]` |
//! | `GeluRowsF32 { width }` | `input` r F32 `[T, width]`; `output` w F32 `[T, width]`, which may alias `input` |
//! | `PadColumnsF32 { input, output }` | `input` r F32 `[T, input]`; `output` w F32 `[T, output]` |
//! | `Im2col1dF32 { channels, kernel, stride, pad_left, pad_right }` | `input` r F32 frames `[T, channels]` (`live_rows` counts frames); `output` w F32 `[U, kernel * channels]`, `U = (T + pad_left + pad_right - kernel) / stride + 1` |
//!
//! Heads (`EncoderHeadBuffers`, `Backend::encoder_heads`), each over
//! packed `rows` `[T, (q_heads + 2 kv_heads) head_dim]` (q, k, v):
//!
//! | Variant | Roles |
//! |---|---|
//! | `RopeQkvPairs { rotary_dim }` | `rows` rw: F32 in, BF16 pair planes out (same bytes); `cos`, `sin` r F32 `[T, rotary_dim / 2]` |
//! | `QkRmsNormF32 { head_dim }` | `rows` rw F32; `q_weight`, `k_weight` r BF16 `[head_dim]` |
//!
//! Attention (`EncoderAttentionBuffers`, `Backend::encoder_attention`):
//!
//! | Variant | Roles |
//! |---|---|
//! | `ContiguousAttentionPairs { q_heads, kv_heads, head_dim, key0_sink }` | `qkv` r BF16 pairs `[T, (q_heads + 2 kv_heads) head_dim]` (as `RopeQkvPairs` writes them); `cu_seqlens` r i32 `[live_sequences + 1]`; `sinks` r F32 `[q_heads]` with `key0_sink` only; `output` w F32 `[T, q_heads head_dim]` |
//!
//! GEMM (`EncoderGemmBuffers`, `Backend::encoder_gemm`):
//!
//! | Variant | Roles |
//! |---|---|
//! | `GemmPairs(GemmQuery { n, k, output, epilogues })` | `activation` r F32 `[T, k]`; `weight` r BF16 `[n, k]`; `bias` r BF16 `[n]` with the bias epilogue (a passed bias stays unread without it); `output` w (rw with the residual epilogue) `[T, n]` in the query's output dtype, F32 or BF16 |
//!
//! Override (`EmbeddingOverrideBuffers`, `Backend::embedding_override`):
//!
//! | Variant | Roles |
//! |---|---|
//! | `EmbeddingOverride { width }` | `activation` rw BF16 `[T, width]`; `arena` r BF16 `[arena_rows, width]`; `source` r i64 `[T]` |
//!
//! The BF16 variants (`AudioRopeBf16`, `EmbeddingSumBf16`, `ResidualAdd`,
//! `Bf16ToF32`, `GeluRows`, `SplitQkv`, `ContiguousAttention`, `LayerNorm`,
//! `Gemm`, `GemmBf16`, `ScratchRope`, `Permute`, `Im2col1d`,
//! `EmbeddingSum`) fill no record: their operands differ (BF16 rows, several
//! embedding tables, split outputs), and the encoders run their F32
//! counterparts above. [`EncoderStage::of`] refuses them, so every backend
//! refuses them alike.

use crate::contract::{MultimodalQuery, Refusal};

/// The record a multimodal stage's operands fill, which names the
/// `Backend` method that prepares it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EncoderStage {
    /// `EncoderRowBuffers`, `Backend::encoder_rows`.
    Rows,
    /// `EncoderHeadBuffers`, `Backend::encoder_heads`.
    Heads,
    /// `EncoderAttentionBuffers`, `Backend::encoder_attention`.
    Attention,
    /// `EncoderGemmBuffers`, `Backend::encoder_gemm`.
    Gemm,
    /// `EmbeddingOverrideBuffers`, `Backend::embedding_override`.
    Override,
}

impl EncoderStage {
    /// The stage `query` runs as, or a refusal for the BF16 variants, whose
    /// operands no record describes.
    pub fn of(query: &MultimodalQuery) -> Result<EncoderStage, Refusal> {
        use MultimodalQuery as Q;
        match query {
            Q::LayerNormF32 { .. }
            | Q::RmsNormF32 { .. }
            | Q::InterpolatedPositionAddF32 { .. }
            | Q::PermuteF32 { .. }
            | Q::GatherRowsF32 { .. }
            | Q::ResidualAddF32 { .. }
            | Q::SwigluF32 { .. }
            | Q::SwigluClampedF32 { .. }
            | Q::GeluRowsF32 { .. }
            | Q::PadColumnsF32 { .. }
            | Q::Im2col1dF32 { .. } => Ok(EncoderStage::Rows),
            Q::RopeQkvPairs { .. } | Q::QkRmsNormF32 { .. } => Ok(EncoderStage::Heads),
            Q::ContiguousAttentionPairs { .. } => Ok(EncoderStage::Attention),
            Q::GemmPairs(_) => Ok(EncoderStage::Gemm),
            Q::EmbeddingOverride { .. } => Ok(EncoderStage::Override),
            Q::AudioRopeBf16 { .. }
            | Q::EmbeddingSumBf16 { .. }
            | Q::ResidualAdd { .. }
            | Q::Bf16ToF32 { .. }
            | Q::GeluRows { .. }
            | Q::SplitQkv { .. }
            | Q::ContiguousAttention { .. }
            | Q::LayerNorm { .. }
            | Q::Gemm(_)
            | Q::GemmBf16(_)
            | Q::ScratchRope { .. }
            | Q::Permute { .. }
            | Q::Im2col1d { .. }
            | Q::EmbeddingSum { .. } => Err(Refusal(
                "a BF16 encoder stage fills no buffer record; its F32 counterpart does",
            )),
        }
    }
}

crate::scratch_record! {
    /// An encoder row pass's buffers: F32 `input` rows (an im2col's
    /// frames), the F32 `output` rows it writes and the device count of
    /// live input rows. A pass reads the side roles its query names and
    /// refuses any other:
    /// - `weight`: a norm's BF16 weight `[width]`.
    /// - `bias`: a LayerNorm's BF16 bias `[width]`.
    /// - `residual`: a residual add's second F32 operand rows.
    /// - `indices`: a permute's i32 source row per row, or the position
    ///   add's four i32 table rows per row.
    /// - `table`: the position add's BF16 table `[table_rows, width]`.
    /// - `blend`: the position add's four F32 tap weights per row.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct EncoderRowBuffers {
        pub input: value,
        pub output: value,
        pub live_rows: value,
        pub weight: option,
        pub bias: option,
        pub residual: option,
        pub indices: option,
        pub table: option,
        pub blend: option,
    }
}

crate::scratch_record! {
    /// An in-place pass's buffers: packed F32 `rows` `[T, (q_heads + 2
    /// kv_heads) head_dim]` and the device count of live rows. The rotary
    /// pass reads each row's F32 `cos` and `sin` `[T, rotary_dim / 2]`; the
    /// q/k RMSNorm reads the BF16 `q_weight` and `k_weight` `[head_dim]`.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct EncoderHeadBuffers {
        pub rows: value,
        pub live_rows: value,
        pub cos: option,
        pub sin: option,
        pub q_weight: option,
        pub k_weight: option,
    }
}

crate::scratch_record! {
    /// Contiguous attention's buffers: the BF16 pair rows `qkv` the rotary
    /// pass writes, the i32 cumulative starts of the live sequences
    /// (`live_sequences + 1`), a key-0 sink's F32 logit per q head (`sinks`,
    /// for a query with `key0_sink` only), the F32 `output` rows `[T,
    /// q_heads head_dim]`, the device count of live tokens and the host's
    /// count of live sequences.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct EncoderAttentionBuffers {
        pub qkv: value,
        pub cu_seqlens: value,
        pub sinks: option,
        pub output: value,
        pub live_tokens: value,
        pub live_sequences: scalar(u32),
    }
}

crate::scratch_record! {
    /// An encoder GEMM's buffers: F32 `activation` rows `[T, k]`, the BF16
    /// `weight` `[n, k]` (row-major), a BF16 `bias` `[n]`, the `output` rows
    /// `[T, n]` (a residual epilogue adds to them in place) and the device
    /// count of live rows. A query with a bias epilogue reads the bias; one
    /// without leaves a passed bias unread (an image may hold one for every
    /// GEMM) and passes none as a null pointer.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct EncoderGemmBuffers {
        pub activation: value,
        pub weight: value,
        pub bias: option,
        pub output: value,
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// The embedding override's buffers: the language model's BF16
    /// `activation` rows `[T, width]`, overridden in place, the encoders'
    /// BF16 `arena` `[arena_rows, width]`, each row's i64 arena row
    /// (`source`; one outside the arena keeps the row) and the device count
    /// of live rows.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct EmbeddingOverrideBuffers {
        pub activation: value,
        pub arena: value,
        pub source: value,
        pub live_rows: value,
        pub arena_rows: scalar(u64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_f32_variant_fills_its_record_and_bf16_ones_are_refused() {
        use MultimodalQuery as Q;
        let rows = Q::GeluRowsF32 {
            width: 64,
            capacity: 4,
        };
        assert_eq!(EncoderStage::of(&rows), Ok(EncoderStage::Rows));
        let pairs = Q::RopeQkvPairs {
            q_heads: 4,
            kv_heads: 4,
            head_dim: 64,
            rotary_dim: 64,
            style: crate::contract::RotaryStyle::Neox,
            capacity: 4,
        };
        assert_eq!(EncoderStage::of(&pairs), Ok(EncoderStage::Heads));
        let attention = |pairs: bool| {
            let (q_heads, kv_heads, head_dim, causal) = (4, 4, 64, false);
            let (window_left, window_right, key0_sink, capacity) = (None, None, false, 4);
            if pairs {
                Q::ContiguousAttentionPairs {
                    q_heads,
                    kv_heads,
                    head_dim,
                    causal,
                    window_left,
                    window_right,
                    key0_sink,
                    capacity,
                }
            } else {
                Q::ContiguousAttention {
                    q_heads,
                    kv_heads,
                    head_dim,
                    causal,
                    window_left,
                    window_right,
                    key0_sink,
                    capacity,
                }
            }
        };
        assert_eq!(
            EncoderStage::of(&attention(true)),
            Ok(EncoderStage::Attention)
        );
        let refused = Err(Refusal(
            "a BF16 encoder stage fills no buffer record; its F32 counterpart does",
        ));
        assert_eq!(EncoderStage::of(&attention(false)), refused);
        let summed = Q::EmbeddingSum {
            tables: 2,
            vocab: 64,
            width: 64,
            capacity: 4,
        };
        assert_eq!(EncoderStage::of(&summed), refused);
        let overridden = Q::EmbeddingOverride {
            width: 64,
            capacity: 4,
        };
        assert_eq!(EncoderStage::of(&overridden), Ok(EncoderStage::Override));
    }
}
