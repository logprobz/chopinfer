//! Lab instances of the split's routed FP8 decode gate/up in warp streams
//! (`chained::split::forms`, `chained_warp_streams.cuh`) with the
//! deliberate defects of `chop::moe::warp_streams::Mutation` its layout can
//! run (no dependency wait, the wait before the first epilogue, a round's
//! chains added in descending K order, the second 8-row fragment never
//! multiplied). The split's decode down (K 512, one 512-K round) does not
//! take warp streams, so it has no instance here, and the gate/up's
//! one-item-per-CTA layout never runs a CTA's later items, which
//! `stale_slot` mutates. Shared with `tests/moe_chained.rs`, whose bitwise,
//! oracle and producer checks must pass on the unmodified instance and fail
//! on each mutant.

use kernel_build::CubinSpec;
use kernels_sm120::moe::chained::{self, Projection, Schedule, split};

pub const FAMILY: &str = "moe_chained_split_warp_streams";

/// The mutants' cubin.
pub const MUTANTS_CUBIN: &str = "chop_moe_chain_lab_split_ws_mutants";

/// `chop::moe::warp_streams::Mutation` values but `stale_slot`, which only
/// the persistent layout reaches; the first is the unmodified kernel.
pub const MUTATIONS: [&str; 5] = [
    "none",
    "drop_wait",
    "late_wait",
    "reversed_fold",
    "dead_fragment",
];

/// The split's served routed FP8 decode gate/up in warp streams.
pub fn projection() -> Projection {
    split::forms()
        .into_iter()
        .find(|p| p.schedule == Schedule::WarpStreams)
        .expect("the split serves its FP8 decode gate/up in warp streams")
}

/// The symbol of the gate/up's instance with `mutation`.
pub fn symbol(mutation: &str) -> String {
    format!("chop_lab_split_ws_{mutation}_gateup_v1")
}

/// The mutants: the gate/up's entry once per mutation, each body given its
/// mutation as the kernel's last template argument.
pub fn mutants_cubin() -> CubinSpec {
    let mut cubin = chained::projection_cubin_specs(MUTANTS_CUBIN, &[projection()])
        .pop()
        .expect("one warp-stream cubin");
    cubin.name = MUTANTS_CUBIN.into();
    let base = std::mem::take(&mut cubin.entries);
    let [entry] = &base[..] else {
        panic!("one warp-stream entry");
    };
    for mutation in MUTATIONS {
        let mut entry = entry.clone();
        assert_eq!(
            entry.body.matches(">(input").count(),
            1,
            "{}",
            entry.symbol
        );
        entry.symbol = symbol(mutation);
        entry.body = entry.body.replace(
            ">(input",
            &format!(",chop::moe::warp_streams::Mutation::{mutation}>(input"),
        );
        cubin.entries.push(entry);
    }
    cubin
}

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(&[MUTANTS_CUBIN], vec![mutants_cubin()])
}
