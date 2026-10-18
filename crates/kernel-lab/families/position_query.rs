//! F32 tail rotation followed by shared MXFP4 fake quantization.

use kernel_build::CubinSpec;
use kernel_core::pooled_indexer::{IndexerCandidates, PositionIndexerQuery};
use kernels_sm120::pooled_indexer::position_query::{self, Entry};
use kernels_sm120::row::block_quantize;

pub const FAMILY: &str = "position_query";
pub const CUBIN: &str = "position_query_sm120";

pub fn queries() -> Vec<PositionIndexerQuery> {
    [(3, 160), (32, 128)]
        .into_iter()
        .map(|(heads, head_dim)| PositionIndexerQuery {
            heads,
            head_dim,
            compression_ratio: 2,
            tail_rope_dims: 64,
            key_norm_eps_bits: 1e-20_f32.to_bits(),
            max_top_k: 512,
            page_size: 64,
            candidates: IndexerCandidates::All,
            capacity: 8,
        })
        .collect()
}

pub fn entries() -> Vec<Entry> {
    queries()
        .into_iter()
        .map(|q| Entry::new(format!("h{}_d{}", q.heads, q.head_dim), q).unwrap())
        .collect()
}

pub fn quantizers() -> Vec<block_quantize::Entry> {
    queries()
        .into_iter()
        .map(|q| {
            block_quantize::Entry::new(
                q.query_quantization().unwrap(),
                q.capacity,
                format!("chop_position_query_h{}_d{}_mxfp4_v2", q.heads, q.head_dim),
            )
            .unwrap()
        })
        .collect()
}

pub fn cubins() -> Vec<CubinSpec> {
    let mut spec = position_query::cubin_spec(CUBIN, &entries(), &quantizers()).unwrap();
    spec.entries.extend(
        entries()
            .iter()
            .map(|entry| entry.ignored_bounds_spec().unwrap()),
    );
    crate::declared(&[CUBIN], vec![spec])
}
