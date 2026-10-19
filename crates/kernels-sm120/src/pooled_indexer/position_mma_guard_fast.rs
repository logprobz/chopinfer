//! Conservative row/key admission before the exact per-head guard.
use super::position_mma_guard;
use kernel_build::{CubinSpec, EntrySpec};

pub fn score_symbol(guarded: &str) -> String {
    assert!(guarded.ends_with("_mma_fp4_guard_v1"));
    guarded.replace("_mma_fp4_guard_v1", "_mma_fp4_guard_fast_v1")
}

/// Adapt a guarded score entry. Packing, scratch and selectors are unchanged.
pub fn score(mut entry: EntrySpec) -> EntrySpec {
    entry.symbol = score_symbol(&entry.symbol);
    entry.body = entry.body.replace(
        "chop::position_mma_guard::",
        "chop::position_mma_guard_fast::",
    );
    entry
}

pub fn diagnostic(score: EntrySpec) -> EntrySpec {
    let mut guarded = score;
    guarded.symbol = guarded
        .symbol
        .replace("_mma_fp4_guard_fast_v1", "_mma_fp4_guard_v1");
    let mut entry = position_mma_guard::diagnostic(guarded);
    entry.symbol = entry
        .symbol
        .replace("_mma_fp4_guard_v1", "_mma_fp4_guard_fast_v1");
    entry
}

pub fn cubin_spec(name: impl Into<String>, entries: Vec<EntrySpec>) -> CubinSpec {
    let mut spec = position_mma_guard::cubin_spec(name, entries);
    spec.headers = vec!["#include \"pooled_indexer/position_mma_guard_fast.cuh\"".into()];
    spec
}
