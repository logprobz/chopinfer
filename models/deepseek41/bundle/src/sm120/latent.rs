//! Native latent attention registered from the admitted geometry. Each
//! geometry's cubin holds the entries for every supported rank count.

use kernel_build::CubinSpec;
use kernel_core::latent_attention::LatentAttentionQuery;
use kernels_sm120::backend::Sm120;
use kernels_sm120::latent::sources::{self, Entry};
use kernels_sm120::{Profile, ProfileNames};
use model_deepseek41_contract::admission::{SUPPORTED_RANKS, check_ranks};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::latent::{self, Keys};

pub const FAMILY: &str = "latent_sources";

pub const fn profile(g: &Geometry) -> Profile {
    Profile::synthetic(ProfileNames {
        name: "deepseek41",
        tag: g.kernel_tag,
    })
}

pub fn entries(g: &Geometry, ranks: u32) -> Result<Vec<Entry>, String> {
    latent::queries(g, ranks)?.into_iter().map(|(keys, query)| {
        let LatentAttentionQuery::Attention { phase, page_size, .. } = query else { unreachable!() };
        let built = if page_size == 32 { sources::paged_entries(profile(g), query) }
            else { sources::entries(profile(g), query) };
        let mut entry = built.map_err(|e| e.to_string())?
            .into_iter()
            .find(|entry| matches!(entry.query, LatentAttentionQuery::Attention { phase: p, .. } if p == phase))
            .expect("shared builder supplies every phase");
        if keys == Keys::DraftBlock {
            entry.name.push_str("_draft_block");
        }
        Ok(entry)
    }).collect()
}

pub fn register(backend: Sm120, g: &Geometry, ranks: u32) -> Result<Sm120, String> {
    Ok(entries(g, ranks)?
        .into_iter()
        .chain(bf16_entries(g, ranks)?)
        .fold(backend, Sm120::with_latent_sources))
}

pub fn cubin(g: &Geometry) -> Result<CubinSpec, String> {
    let mut all = Vec::new();
    for &ranks in SUPPORTED_RANKS {
        if check_ranks(g, ranks).is_ok() {
            all.extend(entries(g, ranks)?.into_iter().filter(|e| {
                !matches!(
                    e.query,
                    LatentAttentionQuery::Attention { page_size: 32, .. }
                )
            }));
        }
    }
    sources::cubin_spec(profile(g), &all).map_err(|e| e.to_string())
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let mut out = Vec::new();
    for g in geometries {
        out.push((FAMILY, cubin(g)?));
        out.push(("latent_sources_bf16", bf16_cubin(g)?));
        if g.shape.compress_ratios.contains(&2) {
            out.push(("latent_sources_pages", ratio2_cubin(g)?));
        }
    }
    Ok(out)
}

/// Ratio-2 entries have a separate cubin; legacy geometry bytes stay fixed.
pub fn ratio2_cubin(g: &Geometry) -> Result<CubinSpec, String> {
    let mut all = Vec::new();
    for &ranks in SUPPORTED_RANKS {
        if check_ranks(g, ranks).is_ok() {
            all.extend(entries(g, ranks)?.into_iter().filter(|e| {
                matches!(
                    e.query,
                    LatentAttentionQuery::Attention { page_size: 32, .. }
                )
            }));
        }
    }
    sources::paged_cubin_spec(profile(g), 32, &all).map_err(|e| e.to_string())
}

/// The comparison layout stores each native quantized operand as BF16.
pub fn bf16_entries(g: &Geometry, ranks: u32) -> Result<Vec<Entry>, String> {
    entries(g, ranks)?
        .into_iter()
        .map(|mut entry| {
            entry.query = latent::with_storage(
                entry.query,
                model_deepseek41_contract::state::KvStorage::Bf16,
            )?;
            entry.name.push_str("_bf16");
            Ok(entry)
        })
        .collect()
}

pub fn bf16_cubin(g: &Geometry) -> Result<CubinSpec, String> {
    let mut all = Vec::new();
    for &ranks in SUPPORTED_RANKS {
        if check_ranks(g, ranks).is_ok() {
            all.extend(bf16_entries(g, ranks)?);
        }
    }
    let mut spec = sources::cubin_spec(profile(g), &all).map_err(|e| e.to_string())?;
    spec.name = format!("latent_sources_{}_bf16_sm120", g.kernel_tag);
    Ok(spec)
}
