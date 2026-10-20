//! The drafter's Markov embedding through immutable peer table reads.
//!
//! Each drafting position used to gather its rank's vocabulary shard row,
//! widen it and reduce the partials across ranks before the round. Exactly
//! one rank owns each ID and every other rank adds +0, so the shared
//! replicated lookup's closed form of that sum gives the same BF16 bits
//! from one launch, with no cross-rank wait. The old shard and widened
//! rows are still written. One stream: the broadcast copy is scratch.
use kernel_build::CubinSpec;
use kernel_core::replicated_embedding::ReplicatedEmbedding;
use kernels_sm120::{backend::Sm120, row::replicated_embedding::Entry};
use model_deepseek41_contract::{
    admission::{SUPPORTED_RANKS, check_ranks},
    geometry::Geometry,
    serving::DECODE_BATCH,
};

pub const FAMILY: &str = "dspark_peer_embedding";

pub fn cubin_name(g: &Geometry) -> String {
    format!("dspark_peer_embedding_{}_sm120", g.kernel_tag)
}

pub fn query(g: &Geometry, ranks: u32) -> ReplicatedEmbedding {
    ReplicatedEmbedding {
        vocab: g.shape.vocab,
        width: g.shape.dspark.markov_rank,
        ranks,
        streams: 1,
    }
}

pub fn entry(g: &Geometry, ranks: u32) -> Entry {
    Entry {
        query: query(g, ranks),
        capacity: DECODE_BATCH,
        symbol: format!(
            "chop_{}_dspark_markov_replicated_tp{ranks}_v1",
            g.kernel_tag
        ),
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
            kernels_sm120::row::replicated_embedding::cubin_spec(&cubin_name(g), &entries)
                .map(|spec| (FAMILY, spec))
                .map_err(|e| e.to_string())
        })
        .collect()
}

pub fn register(backend: Sm120, g: &Geometry, ranks: u32) -> Sm120 {
    backend.with_replicated_embedding(entry(g, ranks))
}
