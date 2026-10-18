//! Position scoring and candidate filtering at generic shapes, including
//! a 160-channel score that crosses a 128-K boundary and explicit mutants.

use kernel_build::CubinSpec;
use kernel_core::pooled_indexer::{CandidateBlocksQuery, IndexerCandidates, PositionIndexerQuery};
use kernels_sm120::pooled_indexer::position::{self, CandidateEntry, Entry};

pub const FAMILY: &str = "position_indexer";
pub const CUBIN: &str = "position_indexer_sm120";

pub fn entries() -> Vec<Entry> {
    [(3, 160, 2, 2048), (32, 128, 1, 512), (32, 128, 2, 512)]
        .into_iter()
        .flat_map(|(heads, dim, ratio, max_top_k)| {
            [IndexerCandidates::All, IndexerCandidates::Mask]
                .into_iter()
                .map(move |candidates| {
                    let mask = if candidates == IndexerCandidates::Mask {
                        "mask"
                    } else {
                        "all"
                    };
                    Entry {
                        name: format!("h{heads}_d{dim}_r{ratio}_{mask}"),
                        query: PositionIndexerQuery {
                            heads,
                            head_dim: dim,
                            compression_ratio: ratio,
                            tail_rope_dims: 64,
                            key_norm_eps_bits: 1e-20_f32.to_bits(),
                            max_top_k,
                            page_size: 64,
                            candidates,
                            capacity: 8,
                        },
                    }
                })
        })
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
                capacity: 8,
            },
        })
        .collect()
}

pub fn cubins() -> Vec<CubinSpec> {
    let mut specs = Vec::new();
    for entry in entries() {
        specs.extend(entry.specs().expect("position selection entry"));
    }
    for entry in candidates() {
        specs.extend(entry.specs().expect("candidate filter entry"));
    }
    specs.extend(
        entries()[1]
            .ignored_mask_specs()
            .expect("ignored-mask control"),
    );
    for control in 2..=5 {
        specs.extend(
            entries()[0]
                .control_specs(control)
                .expect("selection control"),
        );
    }
    specs.extend(
        candidates()[0]
            .wrong_pin_specs()
            .expect("wrong-pin control"),
    );
    crate::declared(&[CUBIN], vec![position::cubin_spec(CUBIN, specs)])
}
