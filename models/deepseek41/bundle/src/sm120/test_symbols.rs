//! Hardware entry names used by the model's structural audits.

pub(crate) fn packed_fold_symbol(row_major: &str) -> String {
    kernels_sm120::gemm::mxfp8_fold_packed::symbol(row_major)
}

pub(crate) fn row_major_fold_symbol(packed: &str) -> Option<&str> {
    packed.strip_suffix(kernels_sm120::gemm::mxfp8_fold_packed::SUFFIX)
}

pub(crate) fn scale_expand_pair_symbol(tiled: bool) -> &'static str {
    if tiled {
        kernels_sm120::moe::scale_expand::PAIR_TILES_SYMBOL
    } else {
        kernels_sm120::moe::scale_expand::PAIR_SYMBOL
    }
}

/// Paired fallback expansion for tile-encoded scale banks.
pub(crate) fn scale_tiles_expand_pair_symbol(tiled: bool) -> &'static str {
    if tiled {
        kernels_sm120::moe::scale_tiles_expand::PAIR_TILES_SYMBOL
    } else {
        kernels_sm120::moe::scale_tiles_expand::PAIR_SYMBOL
    }
}

pub(crate) fn scale_tiles_expand_gate_symbol(width: u32) -> String {
    super::scale_inplace::gate_expansion_cubin()
        .unwrap()
        .1
        .entries
        .into_iter()
        .find(|entry| entry.symbol.ends_with(&format!("_w{width}_v1")))
        .unwrap()
        .symbol
}
