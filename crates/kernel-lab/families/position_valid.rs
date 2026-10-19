//! Full prefill row-block comparisons, including retained-prefix readers.
use kernel_build::CubinSpec;
use kernel_core::pooled_indexer::{CandidateBlocksQuery, IndexerCandidates, PositionIndexerQuery};
use kernel_core::position_blocks::PositionBlockIndexerQuery;
use kernels_sm120::pooled_indexer::{
    position, position_bf16, position_blocks, position_candidates_runtime, position_live,
    position_valid,
};

pub const FAMILY: &str = "position_valid";
pub const CUBIN: &str = "position_valid_sm120";
pub fn dense(ratio: u32) -> position::Entry {
    position::Entry {
        name: format!("valid_reference_r{ratio}"),
        query: PositionIndexerQuery {
            heads: 32,
            head_dim: 128,
            compression_ratio: ratio,
            tail_rope_dims: 64,
            key_norm_eps_bits: 1e-20_f32.to_bits(),
            max_top_k: 512,
            page_size: 64 / ratio,
            candidates: IndexerCandidates::All,
            capacity: 64,
        },
    }
}
pub fn compact() -> position_blocks::Entry {
    let mut indexer = dense(1).query;
    indexer.capacity = 4096;
    position_blocks::Entry {
        name: "valid_reference".into(),
        query: PositionBlockIndexerQuery {
            indexer,
            block_size: 8,
            top_k_blocks: 2048,
        },
    }
}
pub fn candidates() -> position_blocks::CandidateEntry {
    position_blocks::CandidateEntry {
        name: "valid_reference".into(),
        query: CandidateBlocksQuery {
            block_size: 8,
            top_k_blocks: 2048,
            capacity: 64,
        },
    }
}
pub fn cubins() -> Vec<CubinSpec> {
    let mut specs = Vec::new();
    for ratio in [1, 2] {
        let entry = dense(ratio);
        specs.extend(entry.specs().unwrap());
        specs.extend(position_bf16::dense_specs(&entry).unwrap());
        for bf16 in [false, true] {
            specs.extend(position_live::dense_specs(&entry, bf16).unwrap());
        }
    }
    let entry = compact();
    specs.extend(entry.specs().unwrap());
    specs.extend(position_bf16::compact_specs(&entry).unwrap());
    for bf16 in [false, true] {
        specs.extend(position_valid::compact_specs(&entry, bf16).unwrap());
    }
    let entry = candidates();
    specs.extend(entry.specs().unwrap());
    specs.extend(position_candidates_runtime::specs(&entry).unwrap());
    for runtime in [false, true] {
        specs.extend(position_valid::candidate_specs(&entry, runtime).unwrap());
    }
    let mut spec = position_valid::cubin_spec(CUBIN, specs);
    spec.headers.extend([
        "#include \"pooled_indexer/position_bf16.cuh\"".into(),
        "#include \"pooled_indexer/position_candidates_runtime.cuh\"".into(),
    ]);
    crate::declared(&[CUBIN], vec![spec])
}
