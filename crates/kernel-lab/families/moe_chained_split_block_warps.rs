//! Lab instances of the split's NVFP4 routed decode gate/up and down in
//! block warps (`chained::split::forms`, `chained_block_warps_fp4.cuh`),
//! each built unmodified and with every deliberate defect of
//! `chop::moe::block_warps::Mutation` (no dependency wait, the wait after
//! the K loop, a round's chains added in descending K order). Shared with
//! `tests/moe_chained.rs`, whose producer, back-to-back and bitwise checks
//! must pass on the unmodified instances and fail on each mutant.

use kernel_build::CubinSpec;
use kernels_sm120::moe::chained::{self, Projection, Schedule, Weights, split};

pub const FAMILY: &str = "moe_chained_split_block_warps";

/// The NVFP4 instances' cubin.
pub const FP4_CUBIN: &str = "chop_moe_chain_lab_split_bw4_fp4_mutants";

/// `chop::moe::block_warps::Mutation` values; the first is the unmodified
/// kernel.
pub const MUTATIONS: [&str; 4] = ["none", "drop_wait", "late_wait", "reversed_fold"];

/// The split's served NVFP4 decode gate/up and down in block warps.
pub fn fp4_projections() -> [Projection; 2] {
    let forms = split::forms();
    [true, false].map(|gate_up| {
        forms
            .iter()
            .find(|p| {
                p.weights() == Weights::Nvfp4
                    && p.schedule == Schedule::BlockWarps
                    && p.gate_up() == gate_up
            })
            .expect("the split serves the NVFP4 decode projections in block warps")
            .clone()
    })
}

/// The symbol of an NVFP4 projection's instance with `mutation`.
pub fn fp4_symbol(mutation: &str, gate_up: bool) -> String {
    let projection = if gate_up { "gateup" } else { "down" };
    format!("chop_lab_split_bw4_fp4_{mutation}_{projection}_v1")
}

/// The NVFP4 instances' cubin: the projections' entries once per mutation,
/// each body given its mutation as the kernel's last template argument.
pub fn fp4_cubin() -> CubinSpec {
    let mut cubin = chained::projection_cubin_specs(FP4_CUBIN, &fp4_projections())
        .pop()
        .expect("one block-warp cubin");
    cubin.name = FP4_CUBIN.into();
    let base = std::mem::take(&mut cubin.entries);
    for mutation in MUTATIONS {
        for (entry, gate_up) in base.iter().zip([true, false]) {
            let mut entry = entry.clone();
            assert_eq!(
                entry.body.matches(">>(input").count(),
                1,
                "{}",
                entry.symbol
            );
            entry.symbol = fp4_symbol(mutation, gate_up);
            entry.body = entry.body.replace(
                ">>(input",
                &format!(">, chop::moe::block_warps::Mutation::{mutation}>(input"),
            );
            cubin.entries.push(entry);
        }
    }
    cubin
}

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(&[FP4_CUBIN], vec![fp4_cubin()])
}
