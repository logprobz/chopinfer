//! Three consecutive tiles in one reference reduction expose signed-zero updates.
use kernel_build::CubinSpec;
use kernel_core::contract::AttentionPhase;
use kernel_core::latent_attention::LatentAttentionQuery;
use kernels_sm120::latent::sources::{self, Entry, Schedule};
use model_deepseek41_contract::geometry::GEOMETRIES;

pub const FAMILY: &str = "latent_sources_state_probe";

pub fn pair() -> Result<(Entry, Entry), String> {
    let geometry = GEOMETRIES
        .iter()
        .find(|g| g.kernel_tag == "deepseek41")
        .ok_or("Flash geometry")?;
    let old = super::latent::entries(geometry, 4)?
        .into_iter()
        .find(|entry| {
            matches!(
                entry.query,
                LatentAttentionQuery::Attention {
                    phase: AttentionPhase::Prefill,
                    page_size: 64,
                    second_fp4_keys: true,
                    ..
                }
            )
        })
        .ok_or("one-split reference attention")?;
    let mut new = old.clone();
    new.schedule = Schedule::IndependentHeads;
    new.name.push_str("_signed_zero");
    Ok((old, new))
}

pub fn cubin() -> Result<CubinSpec, String> {
    let (_, entry) = pair()?;
    let mut spec = sources::cubin_spec(entry.profile, &[entry]).map_err(|e| e.to_string())?;
    spec.name = "latent_sources_state_probe_deepseek41_sm120".into();
    Ok(spec)
}
