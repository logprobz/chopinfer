//! Retained-prefix compact work and visible-only candidate maxima.
use super::{position_blocks, position_live};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::Refusal;

pub const CANDIDATE_CTAS: u32 = 32;

pub fn symbol(original: &str) -> String {
    format!("{original}_valid_v1")
}
pub fn merge_symbol(original: &str) -> String {
    format!("{original}_valid_merge_v1")
}
fn rename(specs: &mut [EntrySpec]) {
    for spec in specs {
        spec.symbol = spec
            .symbol
            .replace("_live_v1", "_valid_v1")
            .replace("_live_merge_v1", "_valid_merge_v1");
    }
}
pub fn compact_specs(
    entry: &position_blocks::Entry,
    bf16: bool,
) -> Result<Vec<EntrySpec>, Refusal> {
    let mut specs = position_live::compact_specs(entry, bf16)?;
    rename(&mut specs);
    for spec in &mut specs {
        spec.body = spec.body.replace(
            "chop::position_live::compact_",
            "chop::position_valid::compact_",
        );
    }
    Ok(specs)
}
pub fn candidate_specs(
    entry: &position_blocks::CandidateEntry,
    runtime: bool,
) -> Result<Vec<EntrySpec>, Refusal> {
    let mut specs = position_live::candidate_specs(entry, runtime)?;
    rename(&mut specs);
    specs[0].body = specs[0].body.replace(
        "chop::position_live::block_scores",
        "chop::position_valid::block_scores",
    );
    Ok(specs)
}
pub fn cubin_spec(name: impl Into<String>, entries: Vec<EntrySpec>) -> CubinSpec {
    let mut spec = position_live::cubin_spec(name, entries);
    spec.headers = vec!["#include \"pooled_indexer/position_valid.cuh\"".into()];
    spec
}
