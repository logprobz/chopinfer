//! Locked head groups for measured native TP4 verify4 attention.
use kernel_build::CubinSpec;
use kernels_sm120::latent::sources_cached::{self, GroupedEntry};
use model_deepseek41_contract::geometry::Geometry;

pub const FAMILY: &str = "latent_cache_groups";
pub const CUBIN: &str = "latent_cache_groups_deepseek41_sm120";

pub fn entries(g: &Geometry) -> Result<Vec<GroupedEntry>, String> {
    Ok(super::latent_cache::entries(g)?
        .into_iter()
        .filter(|e| {
            matches!(
                e.source.query,
                kernel_core::latent_attention::LatentAttentionQuery::Attention {
                    phase: kernel_core::contract::AttentionPhase::Verify,
                    ..
                }
            )
        })
        .flat_map(|cache| {
            [1, 2, 4, 8].map(move |heads| GroupedEntry {
                cache: cache.clone(),
                heads,
            })
        })
        .collect())
}

pub fn cubin(g: &Geometry) -> Result<CubinSpec, String> {
    let entries = entries(g)?;
    let first = entries.first().ok_or("empty grouped cache catalog")?;
    let mut spec = sources_cached::cubin_spec(CUBIN, std::slice::from_ref(&first.cache))
        .map_err(|e| e.to_string())?;
    spec.entries = entries
        .iter()
        .map(|e| e.spec().map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    Ok(spec)
}
