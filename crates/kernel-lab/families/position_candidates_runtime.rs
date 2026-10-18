//! Runtime-count candidate producers and their compiled-count controls.
use kernel_build::CubinSpec;
use kernel_core::pooled_indexer::CandidateBlocksQuery;
use kernels_sm120::pooled_indexer::{position_blocks::CandidateEntry, position_candidates_runtime};

pub const FAMILY: &str = "position_candidates_runtime";
pub const CUBIN: &str = "position_candidates_runtime_sm120";

pub fn compiled() -> CandidateEntry {
    CandidateEntry {
        name: "b8_k2048".into(),
        query: CandidateBlocksQuery { block_size: 8, top_k_blocks: 2048, capacity: 8 },
    }
}

pub fn cubins() -> Vec<CubinSpec> {
    let entry = compiled();
    let mut specs = position_candidates_runtime::specs(&entry).unwrap();
    specs.extend(entry.specs().unwrap());
    crate::declared(&[CUBIN], vec![position_candidates_runtime::cubin_spec(CUBIN, specs)])
}
