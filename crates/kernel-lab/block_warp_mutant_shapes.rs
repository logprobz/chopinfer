//! Lab-only block-warp chained projections (`chained::Schedule::BlockWarps`):
//! the shared expert's FP8 decode gate/up and down (intermediate 512), each
//! built unmodified and with every deliberate defect of
//! `chop::moe::block_warps::Mutation` (no dependency wait, the wait after
//! the K loop, a round's chains added in descending K order). Shared by
//! build.rs, which builds their cubin in the `moe_chained` lock family, and
//! `tests/moe_chained.rs`, whose producer, back-to-back and bitwise checks
//! must pass on the unmodified instances and fail on each mutant. The
//! split's NVFP4 routed mutants are `moe_chained_split_block_warps`'.

use kernel_build::CubinSpec;
use kernels_sm120::moe::chained::{self, Projection, Weights};

/// The instances' cubin.
pub const CUBIN: &str = "chop_moe_chain_lab_bw4_mutants";

/// `chop::moe::block_warps::Mutation` values; the first is the unmodified
/// kernel.
pub const MUTATIONS: [&str; 4] = ["none", "drop_wait", "late_wait", "reversed_fold"];

/// The shared expert's FP8 decode gate/up and down in block warps.
pub fn projections() -> [Projection; 2] {
    let shared = chained::TP4_288_TOP8.dense_intermediates[1];
    let pick = |gate_up: bool| {
        chained::block_warp_catalog()
            .into_iter()
            .find(|p| {
                p.weights() == Weights::Fp8Block128
                    && p.query.gemm.grouped.is_none()
                    && p.gate_up() == gate_up
                    && (if gate_up { p.columns() } else { p.query.gemm.k }) == shared
            })
            .expect("the shared expert's block-warp projections are cataloged")
    };
    [pick(true), pick(false)]
}

/// The symbol of a projection's instance with `mutation`.
pub fn symbol(mutation: &str, gate_up: bool) -> String {
    let projection = if gate_up { "gateup" } else { "down" };
    format!("chop_lab_bw4_{mutation}_{projection}_v1")
}

/// The cubin: the projections' entries once per mutation, each body given
/// its mutation as the kernel's last template argument.
#[allow(dead_code)] // Read by the build script only.
pub fn cubin() -> CubinSpec {
    let mut cubin = chained::projection_cubin_specs(CUBIN, &projections())
        .pop()
        .expect("one block-warp cubin");
    cubin.name = CUBIN.into();
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
            entry.symbol = symbol(mutation, gate_up);
            entry.body = entry.body.replace(
                ">>(input",
                &format!(">, chop::moe::block_warps::Mutation::{mutation}>(input"),
            );
            cubin.entries.push(entry);
        }
    }
    cubin
}
