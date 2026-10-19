//! Native scoring inside an exact accumulation domain, with SIMT repair.
use super::{position, position_blocks, position_mma};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::Refusal;

pub fn score_symbol(original: &str) -> String {
    format!("{original}_mma_fp4_guard_v1")
}
pub fn pack_symbol(original: &str) -> String {
    format!("{original}_pack_fp4_guard_v1")
}
fn adapt(entries: Vec<EntrySpec>) -> Vec<EntrySpec> {
    entries
        .into_iter()
        .map(|mut e| {
            e.symbol = e
                .symbol
                .replace("_mma_fp4_v1", "_mma_fp4_guard_v1")
                .replace("_pack_fp4_v1", "_pack_fp4_guard_v1");
            e.body = e
                .body
                .replace(
                    "chop::position_mma_native::scores",
                    "chop::position_mma_guard::scores",
                )
                .replace(
                    "chop::position_mma_compact::scores",
                    "chop::position_mma_guard::compact_scores",
                );
            e
        })
        .collect()
}
pub fn dense_specs(entry: &position::Entry) -> Result<Vec<EntrySpec>, Refusal> {
    Ok(adapt(position_mma::dense_specs(entry)?))
}
pub fn compact_specs(entry: &position_blocks::Entry) -> Result<Vec<EntrySpec>, Refusal> {
    Ok(adapt(position_mma::compact_specs(entry)?))
}
pub fn cubin_spec(name: impl Into<String>, entries: Vec<EntrySpec>) -> CubinSpec {
    let mut spec = position_mma::cubin_spec(name, entries);
    spec.headers = vec!["#include \"pooled_indexer/position_mma_guard.cuh\"".into()];
    spec
}

/// Lab-only counters use a separate symbol and lock from serving.
pub fn diagnostic(mut score: EntrySpec) -> EntrySpec {
    assert!(score.symbol.ends_with("_mma_fp4_guard_v1"));
    score.symbol.push_str("_counts_v1");
    score.params.push(kernel_build::EntryParam::DevicePointer {
        name: "guard_counts".into(),
        cxx_type: "uint32_t".into(),
    });
    score.body = score.body.replace(
        "reinterpret_cast<const uint32_t*>(scratch+qstatus_offset));",
        "reinterpret_cast<const uint32_t*>(scratch+qstatus_offset),guard_counts);",
    );
    score
}
