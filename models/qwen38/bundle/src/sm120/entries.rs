//! The row catalog entries the bundle launches by name, at TP4 and the
//! Flash geometry's tag (`launch::tagged`, `launch::per_rank` retag them).

/// The Qwen3.8 embedding row of the row catalog, also the prefill head's row
/// gather.
pub const EMBEDDING: &str = "chop_row_embedding_qwen38_v1";

pub const HC_NORM: &str = "chop_row_qwen38_hc_norm_v1";

pub const NGRAM: &str = "chop_row_qwen38_ngram_v1";
/// The n-gram and PLE convolution over every decode bucket item.
pub const NGRAM_DECODE: &str = "chop_row_qwen38_ngram_decode_v1";
pub const PLE_CONV_PREFILL: &str = "chop_row_qwen38_ple_conv_prefill_v1";
pub const PLE_CONV_ITEMS: &str = "chop_row_qwen38_ple_conv_items_decode_v1";
