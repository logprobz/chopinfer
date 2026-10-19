//! Lab-only decode staging variants over the qualified TP4 source entries.
use kernel_build::CubinSpec;
use kernels_sm120::latent::sources_decode::{self, Entry};
use model_deepseek41_contract::geometry::Geometry;

pub const FAMILY: &str = "latent_decode_staging";
pub const VARIANTS: &[(u32, u32, bool)] = &[
    (64, 0, false),
    (64, 0, true),
    (64, 2, true),
    (64, 4, true),
    (64, 8, true),
    (32, 0, true),
    (32, 2, true),
    (32, 4, true),
    (32, 8, true),
];
pub fn entries(g: &Geometry) -> Result<Vec<Entry>, String> {
    if !super::candidate::qualified(g, 4) {
        return Err("decode staging requires the qualified Flash TP4 geometry".into());
    }
    Ok(super::candidate::entries(g, 4)?
        .into_iter()
        .filter(|s| !s.name.ends_with("_draft_block"))
        .flat_map(|source| {
            VARIANTS
                .iter()
                .flat_map(move |&(stage_keys, head_group, packed)| {
                    let source = source.clone();
                    [false, true].map(move |profile| Entry {
                        source: source.clone(),
                        stage_keys,
                        head_group,
                        packed,
                        profile,
                    })
                })
        })
        .collect())
}
pub fn cubin(g: &Geometry) -> Result<CubinSpec, String> {
    let mut spec =
        sources_decode::cubin_spec("latent_decode_staging_deepseek41_sm120", &entries(g)?)
            .map_err(|e| e.to_string())?;
    spec.entries.extend([
        sources_decode::widen_spec(false),
        sources_decode::widen_spec(true),
    ]);
    Ok(spec)
}
