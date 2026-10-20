//! Diagnostic metadata for the exact served SM120 attention entries.
use kernel_build::EntrySpec;
use kernel_core::contract::{AttentionPhase, LaunchPlan};
use kernel_core::latent_attention::LatentAttentionQuery;
use kernels_sm120::latent::sources_cached;
use model_deepseek41_contract::geometry::Geometry;

type Metadata = (String, Vec<EntrySpec>, LaunchPlan, u64);

impl crate::Kernels {
    pub(crate) fn attention_capture_sources(
        geometry: &Geometry,
        rows: u32,
    ) -> Result<Vec<Metadata>, String> {
        if !super::latent_scheduled::qualified(geometry, 4) {
            return Err("attention capture requires native Flash TP4".into());
        }
        super::latent_scheduled::serving_entries(geometry, 4)?
            .into_iter()
            .filter(|source| {
                !source.name.ends_with("_draft_block")
                    && matches!(
                        source.query,
                        LatentAttentionQuery::Attention {
                            phase: AttentionPhase::Verify,
                            ..
                        }
                    )
            })
            .map(|source| {
                let cached = sources_cached::Entry {
                    source: source.clone(),
                    profile: false,
                };
                let mut specs = source.entry_specs().map_err(|e| e.to_string())?;
                specs.push(cached.spec(None).map_err(|e| e.to_string())?);
                specs.push(cached.spec(Some(false)).map_err(|e| e.to_string())?);
                let plan = source.plan(rows).map_err(|e| e.to_string())?;
                let bytes = source.workspace_bytes(rows);
                Ok((source.name, specs, plan, bytes))
            })
            .collect()
    }
}
