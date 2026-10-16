//! The split's lab-only grouped downs at K = 4096
//! (`chained::split::accumulation_projections`): FP8 128x128 in columns and
//! in block warps, NVFP4, MXFP8 and BF16, whose raw sums the 128-K
//! accumulation gate of `tests/moe_chained.rs` reads.

use kernel_build::CubinSpec;
use kernels_sm120::moe::chained::{self, split};

pub const FAMILY: &str = "moe_chained_split_accumulation";

/// The cubin name prefix: `{PREFIX}_{format}` per format, the block-warp
/// FP8 down in `{PREFIX}_fp8block_bw4`.
pub const PREFIX: &str = "chop_moe_chain_lab_split_k4096";

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(
        &[
            "chop_moe_chain_lab_split_k4096_fp8block",
            "chop_moe_chain_lab_split_k4096_fp8block_bw4",
            "chop_moe_chain_lab_split_k4096_nvfp4",
            "chop_moe_chain_lab_split_k4096_mxfp8",
            "chop_moe_chain_lab_split_k4096_bf16",
        ],
        chained::projection_cubin_specs(PREFIX, &split::accumulation_projections()),
    )
}
