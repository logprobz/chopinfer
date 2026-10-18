//! Compact block-ID score/select, dense references and malformed-list controls.
use kernel_build::CubinSpec;
use kernel_core::pooled_indexer::{CandidateBlocksQuery, IndexerCandidates, PositionIndexerQuery};
use kernel_core::position_blocks::PositionBlockIndexerQuery;
use kernels_sm120::pooled_indexer::{
    position,
    position_blocks::{self, CandidateEntry, Entry},
};

pub const FAMILY: &str = "position_blocks";
pub const CUBIN: &str = "position_blocks_sm120";

pub fn entries() -> Vec<Entry> {
    [
        (3, 160, 1, 2, 7),
        (3, 160, 2, 2, 7),
        (32, 128, 1, 2048, 512),
        (32, 128, 2, 2048, 512),
    ]
    .into_iter()
    .map(
        |(heads, head_dim, compression_ratio, top_k_blocks, max_top_k)| Entry {
            name: format!("h{heads}_d{head_dim}_r{compression_ratio}_k{top_k_blocks}"),
            query: PositionBlockIndexerQuery {
                indexer: PositionIndexerQuery {
                    heads,
                    head_dim,
                    compression_ratio,
                    max_top_k,
                    tail_rope_dims: 64,
                    key_norm_eps_bits: 1e-20_f32.to_bits(),
                    page_size: 64,
                    candidates: IndexerCandidates::All,
                    capacity: 4096,
                },
                block_size: 8,
                top_k_blocks,
            },
        },
    )
    .collect()
}
pub fn candidates() -> Vec<CandidateEntry> {
    [2, 2048]
        .into_iter()
        .map(|top_k_blocks| CandidateEntry {
            name: format!("b8_k{top_k_blocks}"),
            query: CandidateBlocksQuery {
                block_size: 8,
                top_k_blocks,
                capacity: 64,
            },
        })
        .collect()
}
pub fn dense() -> Vec<position::Entry> {
    entries()
        .into_iter()
        .flat_map(|entry| {
            [IndexerCandidates::All, IndexerCandidates::Mask].map(|candidates| position::Entry {
                name: format!(
                    "reference_{}_{}",
                    entry.name,
                    if candidates == IndexerCandidates::All {
                        "all"
                    } else {
                        "mask"
                    }
                ),
                query: PositionIndexerQuery {
                    candidates,
                    ..entry.query.indexer
                },
            })
        })
        .collect()
}
pub fn dense_candidates() -> Vec<position::CandidateEntry> {
    candidates()
        .into_iter()
        .map(|entry| position::CandidateEntry {
            name: format!("reference_{}", entry.name),
            query: entry.query,
        })
        .collect()
}
pub fn cubins() -> Vec<CubinSpec> {
    let mut specs = Vec::new();
    for entry in entries() {
        specs.extend(entry.specs().unwrap());
    }
    for entry in candidates() {
        specs.extend(entry.specs().unwrap());
    }
    for entry in dense() {
        specs.extend(entry.specs().unwrap());
    }
    for entry in dense_candidates() {
        let mut reference = entry.specs().unwrap();
        // Keep the dense mask selection, but use the status-safe maximum
        // stage so the reference contains no data-dependent trap either.
        // The independent CPU block oracle checks both paths' membership.
        let mut safe_score = CandidateEntry {
            name: entry.name.clone(),
            query: entry.query,
        }
        .specs()
        .unwrap()
        .remove(0);
        safe_score.symbol = reference[0].symbol.clone();
        reference[0] = safe_score;
        specs.extend(reference);
    }
    for control in 1..=3 {
        specs.extend(entries()[0].control_specs(control).unwrap());
    }
    crate::declared(&[CUBIN], vec![position_blocks::cubin_spec(CUBIN, specs)])
}
