//! Replicated embedding rows through direct immutable shard reads.
use kernel_build::CubinSpec;
use kernel_core::replicated_embedding::ReplicatedEmbedding;
use kernels_sm120::{
    backend::Sm120,
    row::replicated_embedding::{self, Entry},
};
use model_deepseek41_contract::{
    admission::{SUPPORTED_RANKS, check_ranks},
    geometry::Geometry,
    serving::DECODE_BATCH,
};
pub const FAMILY: &str = "embedding_replicated";
pub fn cubin_name(g: &Geometry) -> String {
    format!("embedding_{}_replicated_sm120", g.kernel_tag)
}
pub fn entry(g: &Geometry, ranks: u32) -> Entry {
    Entry {
        query: ReplicatedEmbedding {
            vocab: g.shape.vocab,
            width: g.shape.hidden,
            ranks,
            streams: g.shape.hc.streams,
        },
        capacity: DECODE_BATCH,
        symbol: format!("chop_{}_embedding_replicated_tp{ranks}_v1", g.kernel_tag),
    }
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            let entries: Vec<_> = SUPPORTED_RANKS
                .iter()
                .copied()
                .filter(|&r| check_ranks(g, r).is_ok())
                .map(|r| entry(g, r))
                .collect();
            Ok((
                FAMILY,
                replicated_embedding::cubin_spec(&cubin_name(g), &entries)
                    .map_err(|e| e.to_string())?,
            ))
        })
        .collect()
}
pub fn register(backend: Sm120, g: &Geometry, ranks: u32) -> Sm120 {
    backend.with_replicated_embedding(entry(g, ranks))
}
