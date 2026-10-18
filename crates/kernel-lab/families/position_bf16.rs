//! Exact BF16 comparison storage, native references and rounding controls.
use kernel_build::CubinSpec;
use kernel_core::pooled_indexer::{CandidateBlocksQuery, IndexerCandidates, PositionIndexerQuery};
use kernel_core::position_blocks::PositionBlockIndexerQuery;
use kernels_sm120::pooled_indexer::{position, position_bf16, position_blocks};
pub const FAMILY: &str = "position_bf16";
pub const CUBIN: &str = "position_bf16_sm120";
pub fn compact() -> Vec<position_blocks::Entry> {
    [
        (3, 160, 2, 32, 7, 2),
        (32, 128, 1, 64, 512, 2048),
        (32, 128, 2, 32, 512, 2048),
        (32, 128, 1, 64, 64, 16),
        (32, 128, 2, 32, 64, 16),
    ]
    .into_iter()
    .map(
        |(heads, head_dim, compression_ratio, page_size, max_top_k, top_k_blocks)| {
            position_blocks::Entry {
                name: format!("h{heads}_d{head_dim}_r{compression_ratio}_p{page_size}_k{max_top_k}_b{top_k_blocks}"),
                query: PositionBlockIndexerQuery {
                    indexer: PositionIndexerQuery {
                        heads,
                        head_dim,
                        compression_ratio,
                        page_size,
                        max_top_k,
                        tail_rope_dims: 64,
                        key_norm_eps_bits: 1e-20_f32.to_bits(),
                        capacity: 8,
                        candidates: IndexerCandidates::All,
                    },
                    block_size: 8,
                    top_k_blocks,
                },
            }
        },
    )
    .collect()
}
pub fn dense() -> Vec<position::Entry> {
    compact()
        .into_iter()
        .flat_map(|e| {
            [IndexerCandidates::All, IndexerCandidates::Mask].map(|candidates| position::Entry {
                name: format!(
                    "{}_{}",
                    e.name,
                    if candidates == IndexerCandidates::All {
                        "all"
                    } else {
                        "mask"
                    }
                ),
                query: PositionIndexerQuery {
                    candidates,
                    ..e.query.indexer
                },
            })
        })
        .collect()
}
pub fn filters() -> Vec<position_blocks::CandidateEntry> {
    [2, 16, 2048]
        .into_iter()
        .map(|top_k_blocks| position_blocks::CandidateEntry {
            name: format!("comparison_b8_k{top_k_blocks}"),
            query: CandidateBlocksQuery {
                block_size: 8,
                top_k_blocks,
                capacity: 8,
            },
        })
        .collect()
}
pub fn dense_filters() -> Vec<position::CandidateEntry> {
    filters()
        .into_iter()
        .map(|e| position::CandidateEntry {
            name: format!("reference_{}", e.name),
            query: e.query,
        })
        .collect()
}
pub fn cubins() -> Vec<CubinSpec> {
    let mut specs = Vec::new();
    for e in dense() {
        specs.extend(e.specs().unwrap());
        specs.extend(position_bf16::dense_specs(&e).unwrap());
        specs.extend(position_bf16::dense_control_specs(&e, true).unwrap());
    }
    for e in compact() {
        specs.extend(e.specs().unwrap());
        specs.extend(position_bf16::compact_specs(&e).unwrap());
        specs.extend(position_bf16::compact_control_specs(&e, true).unwrap());
    }
    for e in filters() {
        specs.extend(e.specs().unwrap());
    }
    for e in dense_filters() {
        let mut reference = e.specs().unwrap();
        let mut safe = position_blocks::CandidateEntry {
            name: e.name.clone(),
            query: e.query,
        }
        .specs()
        .unwrap()
        .remove(0);
        safe.symbol = reference[0].symbol.clone();
        reference[0] = safe;
        specs.extend(reference);
    }
    crate::declared(&[CUBIN], vec![position_bf16::cubin_spec(CUBIN, specs)])
}
