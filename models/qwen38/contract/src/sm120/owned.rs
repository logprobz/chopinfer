//! The entries of the bundle's own kernels (`cuda/` in the bundle) at TP4,
//! which [`crate::owned`] declares through the backend
//! ([`crate::backend::Kernels::owned_entry`]) and the bundle's own ops
//! launch.

use crate::backend::OwnedKernel;

pub const PLE_RESIDUAL: &str = "chop_qwen38_ple_residual_v1";
pub const PLE_STAGE: &str = "chop_qwen38_ple_stage_v1";
pub const COPY_ENTRIES: &str = "chop_qwen38_copy_entries_v1";
pub const NGRAM_VERIFY: &str = "chop_qwen38_ngram_verify_v1";
pub const PLE_CONV_VERIFY: &str = "chop_qwen38_ple_conv_verify_v1";
pub const PLE_CHECKPOINTS: &str = "chop_qwen38_ple_checkpoints_v1";
pub const PLE_ANCHOR: &str = "chop_qwen38_ple_anchor_v1";
pub const MTP_GATHER: &str = "chop_qwen38_mtp_gather_v1";
pub const MTP_NORMS: &str = "chop_qwen38_mtp_norms_v1";
pub const MTP_COMBINE: &str = "chop_qwen38_mtp_combine_v1";
pub const DRAFT_ARGMAX_PARTIAL: &str = "chop_qwen38_draft_argmax_partial_v1";
pub const DRAFT_ARGMAX_FINAL: &str = "chop_qwen38_draft_argmax_final_v1";
pub const PREFIX_COPY_INDEXED: &str = "chop_qwen38_prefix_copy_indexed_v1";
pub const PLE_CHECKPOINTS_INDEXED: &str = "chop_qwen38_ple_checkpoints_indexed_v1";

/// The TP4 entry of `kernel`.
pub fn entry(kernel: OwnedKernel) -> &'static str {
    match kernel {
        OwnedKernel::PleResidual => PLE_RESIDUAL,
        OwnedKernel::PleStage => PLE_STAGE,
        OwnedKernel::PleCheckpoints => PLE_CHECKPOINTS,
        OwnedKernel::PleCheckpointsIndexed => PLE_CHECKPOINTS_INDEXED,
        OwnedKernel::PleAnchor => PLE_ANCHOR,
        OwnedKernel::PleConvVerify => PLE_CONV_VERIFY,
        OwnedKernel::NgramVerify => NGRAM_VERIFY,
    }
}

/// The symbol of a topology's variant of the TP4 entry `symbol`, as the
/// bundle's `launch::per_rank` names it: `qwen38` becomes the topology's
/// symbol tag.
pub fn per_rank(symbol: &str, tag: &str) -> String {
    symbol.replacen("qwen38", tag, 1)
}
