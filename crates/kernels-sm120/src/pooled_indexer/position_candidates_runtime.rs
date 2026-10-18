//! Runtime candidate counts with the compiled block-ID output stride.
use super::position_blocks::CandidateEntry;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::Refusal;

/// A separate symbol set; never replaces the compiled-count producer.
pub fn entry(compiled: &CandidateEntry) -> CandidateEntry {
    CandidateEntry {
        name: format!("{}_runtime", compiled.name),
        query: compiled.query,
    }
}

pub fn specs(compiled: &CandidateEntry) -> Result<Vec<EntrySpec>, Refusal> {
    let mut specs = entry(compiled).specs()?;
    specs[1].params.push(EntryParam::U32 {
        name: "top_k_blocks".into(),
    });
    specs[1].body = format!(
        "chop::position_candidates_runtime::gather<{},{}>(blocks,live_rows,tags,visible,source_status,ids,out_tags,out_visible,status,width,top_k_blocks);",
        compiled.query.block_size, compiled.query.top_k_blocks
    );
    Ok(specs)
}

pub fn cubin_spec(name: impl Into<String>, entries: Vec<EntrySpec>) -> CubinSpec {
    let mut spec = super::position_blocks::cubin_spec(name, entries);
    spec.headers = vec!["#include \"pooled_indexer/position_candidates_runtime.cuh\"".into()];
    spec
}
