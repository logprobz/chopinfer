//! The split's routed decode projections in the column schedule where the
//! split serves only another schedule (`chained::split::forms`): the FP8
//! 128x128 gate/up (served in warp streams) and the NVFP4 gate/up and down
//! (served in block warps), at the decode row tile and tokens, from
//! `chained::split::catalog`. Nothing serves them; the bitwise and mutant
//! checks of `tests/moe_chained.rs` compare the served schedules with them.

use kernel_build::CubinSpec;
use kernels_sm120::moe::chained::{self, Projection, Weights, split};

pub const FAMILY: &str = "moe_chained_split_references";

/// The cubin name prefix: `{PREFIX}_{format}`.
pub const PREFIX: &str = "chop_moe_chain_lab_split";

/// The references: the FP8 gate/up, then the NVFP4 gate/up and down.
pub fn projections() -> [Projection; 3] {
    let pick = |weights: Weights, gate_up: bool| {
        split::catalog()
            .into_iter()
            .find(|p| p.weights() == weights && p.gate_up() == gate_up && p.row_tile == 16)
            .expect("the split catalog holds every decode projection in columns")
    };
    [
        pick(Weights::Fp8Block128, true),
        pick(Weights::Nvfp4, true),
        pick(Weights::Nvfp4, false),
    ]
}

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(
        &[
            "chop_moe_chain_lab_split_fp8block",
            "chop_moe_chain_lab_split_nvfp4",
        ],
        chained::projection_cubin_specs(PREFIX, &projections()),
    )
}
