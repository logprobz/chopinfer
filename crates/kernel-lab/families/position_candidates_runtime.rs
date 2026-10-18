//! Runtime-count candidate producers and their compiled-count controls.
use kernel_build::CubinSpec;
use kernel_core::pooled_indexer::CandidateBlocksQuery;
use kernels_sm120::pooled_indexer::{position_blocks::CandidateEntry, position_candidates_runtime};

pub const FAMILY: &str = "position_candidates_runtime";
pub const CUBIN: &str = "position_candidates_runtime_sm120";

pub fn compiled() -> CandidateEntry {
    CandidateEntry {
        // This comparison copy belongs to this lab family. The ordinary
        // position_blocks family retains its b8_k2048 symbols and cubin.
        name: "runtime_candidates_reference_b8_k2048".into(),
        query: CandidateBlocksQuery {
            block_size: 8,
            top_k_blocks: 2048,
            capacity: 8,
        },
    }
}

pub fn cubins() -> Vec<CubinSpec> {
    let entry = compiled();
    let mut specs = position_candidates_runtime::specs(&entry).unwrap();
    specs.extend(entry.specs().unwrap());
    crate::declared(
        &[CUBIN],
        vec![position_candidates_runtime::cubin_spec(CUBIN, specs)],
    )
}
