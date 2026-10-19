//! Additive attention candidates for paired lab checks before serving adoption.

use kernel_build::CubinSpec;
use kernel_core::contract::AttentionPhase;
use kernel_core::latent_attention::LatentAttentionQuery;
use kernels_sm120::backend::Sm120;
use kernels_sm120::latent::sources::{self, Entry, Schedule};
use model_deepseek41_contract::admission::{SUPPORTED_RANKS, check_ranks};
use model_deepseek41_contract::geometry::Geometry;

pub const FAMILY: &str = "latent_sources_scheduled";
pub const QUERY_FAMILY: &str = "latent_query_parallel";

pub fn query() -> kernels_sm120::latent::query_parallel::Entry {
    kernels_sm120::latent::query_parallel::Entry {
        capacity: model_deepseek41_contract::serving::PREFILL_TOKENS,
        symbol: "chop_deepseek41_latent_query_rope_heads_v1".into(),
    }
}

pub fn query_cubin() -> CubinSpec {
    kernels_sm120::latent::query_parallel::cubin_spec(
        "latent_query_parallel_deepseek41_sm120",
        &[query()],
    )
}

pub fn entries(g: &Geometry, ranks: u32) -> Result<Vec<Entry>, String> {
    Ok(super::latent::entries(g, ranks)?
        .into_iter()
        .filter(|e| {
            matches!(
                e.query,
                LatentAttentionQuery::Attention {
                    phase: AttentionPhase::Decode | AttentionPhase::Verify,
                    ..
                }
            )
        })
        .map(|mut e| {
            e.schedule = Schedule::IndependentHeads;
            e
        })
        .collect())
}

pub fn register(backend: Sm120, g: &Geometry, ranks: u32) -> Result<Sm120, String> {
    Ok(entries(g, ranks)?.into_iter().fold(
        backend.with_latent_glue(query().binding()),
        Sm120::with_latent_sources,
    ))
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<CubinSpec>, String> {
    geometries
        .iter()
        .map(|g| {
            let mut all = Vec::new();
            for &ranks in SUPPORTED_RANKS {
                if check_ranks(g, ranks).is_ok() {
                    all.extend(entries(g, ranks)?);
                }
            }
            let mut spec =
                sources::cubin_spec(super::latent::profile(g), &all).map_err(|e| e.to_string())?;
            spec.name = format!("latent_sources_scheduled_{}_sm120", g.kernel_tag);
            // A wrong sink multiplicity must change the compared output.
            for e in &all {
                spec.entries
                    .push(e.sink_per_split_entry_spec().map_err(|e| e.to_string())?);
            }
            Ok(spec)
        })
        .collect()
}
