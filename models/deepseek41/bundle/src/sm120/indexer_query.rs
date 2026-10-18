//! F32 index queries: tail rotation followed by the shared trained quantizer.

use kernel_build::CubinSpec;
use kernel_core::pooled_indexer::PositionIndexerQuery;
use kernels_sm120::backend::Sm120;
use kernels_sm120::pooled_indexer::position_query::{self, Entry};
use kernels_sm120::row::block_quantize;
use model_deepseek41_contract::{geometry::Geometry, indexer, serving::PREFILL_TOKENS};

pub const FAMILY: &str = "position_query";

pub fn query(g: &Geometry) -> PositionIndexerQuery {
    let layer = g.shape.csa.index_sources[0];
    indexer::selection(g, layer, PREFILL_TOKENS).expect("the geometry's first index source")
}

pub fn entry(g: &Geometry) -> Entry {
    Entry::new(g.kernel_tag, query(g)).expect("position query entry")
}

pub fn quantizer(g: &Geometry) -> block_quantize::Entry {
    block_quantize::Entry::new(
        query(g)
            .query_quantization()
            .expect("F32 query quantization"),
        PREFILL_TOKENS,
        format!("chop_position_query_{}_mxfp4_v2", g.kernel_tag),
    )
    .expect("query quantizer entry")
}

pub fn cubin(g: &Geometry) -> Result<CubinSpec, String> {
    position_query::cubin_spec(
        format!("position_query_{}_sm120", g.kernel_tag),
        &[entry(g)],
        &[quantizer(g)],
    )
    .map_err(|e| e.0.to_owned())
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries.iter().map(|g| Ok((FAMILY, cubin(g)?))).collect()
}

pub fn register(sm120: Sm120, g: &Geometry) -> Sm120 {
    sm120
        .with_position_queries([entry(g)])
        .with_block_quantize(quantizer(g))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::{Backend, Need};
    use kernel_core::properties::Invariance;
    use model_deepseek41_contract::geometry::GEOMETRIES;

    #[test]
    fn every_index_query_prepares_the_registered_f32_rotation_and_shared_quantizer() {
        for g in GEOMETRIES {
            let backend = register(Sm120::new(), g);
            let cubin = cubin(g).unwrap();
            assert_eq!(cubin.entries.len(), 2);
            for layer in 0..g.layers() {
                let Some(query) = indexer::selection(g, layer, 4) else {
                    continue;
                };
                let need = Need::rows(4).with_exact(Invariance::any_rows(4));
                let op = backend.position_indexer_query(&query, need).unwrap();
                assert_eq!(
                    op.info().workspace,
                    4 * u64::from(query.heads * query.head_dim) * 4
                );
                for (symbol, _) in &op.info().symbols {
                    assert!(cubin.entries.iter().any(|entry| entry.symbol == *symbol));
                }
            }
        }
    }
}
