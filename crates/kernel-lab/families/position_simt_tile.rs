//! Old and live-context position engines with identical shape and arithmetic.
use kernel_build::CubinSpec;
use kernel_core::pooled_indexer::{CandidateBlocksQuery, IndexerCandidates, PositionIndexerQuery};
use kernel_core::position_blocks::PositionBlockIndexerQuery;
use kernels_sm120::pooled_indexer::{position, position_bf16, position_blocks, position_candidates_runtime, position_live};

pub const FAMILY: &str = "position_simt_tile";
pub const CUBIN: &str = "position_simt_tile_sm120";
pub fn dense(ratio: u32) -> position::Entry {
    position::Entry {
        name: format!("simt_reference_r{ratio}"),
        query: PositionIndexerQuery {
            heads: 32, head_dim: 128, compression_ratio: ratio,
            tail_rope_dims: 64, key_norm_eps_bits: 1e-20_f32.to_bits(),
            max_top_k: 512, page_size: 64 / ratio,
            candidates: IndexerCandidates::All, capacity: 64,
        },
    }
}
pub fn compact() -> position_blocks::Entry {
    position_blocks::Entry {
        name: "simt_reference".into(),
        query: PositionBlockIndexerQuery { indexer: dense(1).query, block_size: 8, top_k_blocks: 2048 },
    }
}
pub fn candidates() -> position_blocks::CandidateEntry {
    position_blocks::CandidateEntry {
        name: "simt_reference".into(),
        query: CandidateBlocksQuery { block_size: 8, top_k_blocks: 2048, capacity: 64 },
    }
}
pub fn cubins() -> Vec<CubinSpec> {
    let mut specs = Vec::new();
    for ratio in [1, 2] {
        let e = dense(ratio);
        specs.extend(e.specs().unwrap());
        specs.extend(position_bf16::dense_specs(&e).unwrap());
        for bf16 in [false, true] { specs.extend(position_live::dense_specs(&e, bf16).unwrap()); }
    }
    for ratio in [1,2] {
        for bf16 in [false,true] {
            for control in [false,true] {
                specs.push(kernels_sm120::pooled_indexer::position_simt_tile::score_spec(&dense(ratio),bf16,control).unwrap());
            }
        }
    }
    let e = compact();
    specs.extend(e.specs().unwrap());
    specs.extend(position_bf16::compact_specs(&e).unwrap());
    for bf16 in [false, true] { specs.extend(position_live::compact_specs(&e, bf16).unwrap()); }
    let e = candidates();
    specs.extend(e.specs().unwrap());
    specs.extend(position_candidates_runtime::specs(&e).unwrap());
    for runtime in [false, true] { specs.extend(position_live::candidate_specs(&e, runtime).unwrap()); }
    let mut spec = position_live::cubin_spec(CUBIN, specs);
    spec.headers.extend([
        "#include \"pooled_indexer/position_bf16.cuh\"".into(),
        "#include \"pooled_indexer/position_candidates_runtime.cuh\"".into(),
    ]);
    spec.headers.push("#include \"pooled_indexer/position_simt_tile.cuh\"".into());
    crate::declared(&[CUBIN], vec![spec])
}
