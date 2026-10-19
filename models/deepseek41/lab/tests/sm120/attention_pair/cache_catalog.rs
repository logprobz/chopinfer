//! Flash TP4 selected-KV cache experiment, without serving registration.
use kernel_build::CubinSpec;
use kernels_sm120::latent::sources_cached::{self, Entry};
use model_deepseek41_contract::geometry::Geometry;

pub const FAMILY: &str = "latent_decode_cache";
pub const CUBIN: &str = "latent_decode_cache_deepseek41_sm120";

pub fn entries(g: &Geometry) -> Result<Vec<Entry>, String> {
    Ok(super::candidate::entries(g, 4)?
        .into_iter()
        .filter(|entry| !entry.name.ends_with("_draft_block"))
        .flat_map(|source| {
            [false, true].map(move |profile| Entry {
                source: source.clone(),
                profile,
            })
        })
        .collect())
}

pub fn cubin(g: &Geometry) -> Result<CubinSpec, String> {
    sources_cached::cubin_spec(CUBIN, &entries(g)?).map_err(|e| e.to_string())
}
