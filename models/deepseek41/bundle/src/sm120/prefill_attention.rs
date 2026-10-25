//! Paired-row attention for native Flash TP4 prefill rows.
use kernel_build::CubinSpec;
use kernel_core::backend::{
    Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View,
};
use kernel_core::contract::{AttentionPhase, Refusal};
use kernel_core::latent_attention::{LatentAttentionQuery, LatentSourcesBuffers};
use kernels_sm120::latent::{sources, sources_prefill};
use model_deepseek41_contract::geometry::{FLASH, Geometry};
use model_deepseek41_contract::state::KvStorage;

pub const FAMILY: &str = "latent_prefill_reuse";
pub const CUBIN: &str = "latent_sources_prefill_reuse2_deepseek41_sm120";
pub const ROWS: u32 = 8192;

pub fn entries() -> Result<Vec<sources::Entry>, String> {
    let entries: Vec<_> = super::latent::entries(&FLASH, 4)?
        .into_iter()
        .filter(|entry| {
            matches!(
                entry.query,
                LatentAttentionQuery::Attention {
                    phase: AttentionPhase::Prefill,
                    ..
                }
            )
        })
        .collect();
    if entries.len() != 3 {
        return Err("expected window and two mixed prefill entries".into());
    }
    Ok(entries)
}

pub fn cubin() -> Result<CubinSpec, String> {
    sources_prefill::reuse_cubin_spec(super::latent::profile(&FLASH), &entries()?, 2)
        .map_err(|e| e.to_string())
}

/// Other layouts and topologies do not load an unused attention module.
pub fn loads(cubin: &str, geometry: &Geometry, ranks: u32, storage: KvStorage) -> bool {
    cubin != CUBIN || (geometry.id == FLASH.id && ranks == 4 && storage == KvStorage::Native)
}

pub fn prepare(
    backend: &dyn Backend,
    geometry: &Geometry,
    ranks: u32,
    query: &LatentAttentionQuery,
    need: Need,
) -> Result<Prepared<LatentSourcesBuffers<View>>, Refused> {
    let reference = backend.latent_attention_sources(query, need)?;
    let source = reference.info().symbols[0].0.clone();
    let reference = kernel_core::route::tagged(
        reference,
        vec![kernel_core::route::branch(
            &source,
            kernel_core::route::RouteClass::Reference,
            "latent_prefill",
            "reference",
            [1..=need.capacity],
        )],
    );
    if geometry.id != FLASH.id || ranks != 4 || need.capacity < ROWS {
        return Ok(reference);
    }
    let Some(entry) = entries()
        .map_err(Refused::Rule)?
        .into_iter()
        .find(|e| e.query == *query)
    else {
        return Ok(reference);
    };
    let mut info = reference.info().clone();
    // Retain the reference symbols: the reference validates every launch and
    // keeps its merge. The numerical contract is unchanged; only input
    // staging is different.
    info.symbols.push((
        sources_prefill::reuse_symbol(&entry, 2),
        sources_prefill::reuse_plan(&entry, ROWS, 2)
            .map_err(|e| Refused::Rule(e.to_string()))?
            .dynamic_shared_bytes,
    ));
    need.check(&info)?;
    Ok(Box::new(PairedPrefill {
        reference,
        info,
        entry,
    }))
}

struct PairedPrefill {
    reference: Prepared<LatentSourcesBuffers<View>>,
    info: OpInfo,
    entry: sources::Entry,
}

impl PreparedOp<LatentSourcesBuffers<View>> for PairedPrefill {
    fn route(&self, counts: Counts, launch: &Launch) -> Option<kernel_core::route::RouteTag> {
        if launch.symbol == sources_prefill::reuse_symbol(&self.entry, 2) {
            Some(kernel_core::route::branch(
                &launch.symbol,
                kernel_core::route::RouteClass::Fast,
                "latent_prefill",
                "paired",
                [1..=ROWS],
            ))
        } else {
            self.reference.route(counts, launch)
        }
    }
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &LatentSourcesBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        // Validate through the reference before changing the attention plan.
        // Device live counts still mask graph replays, including empty tails.
        // Every row count and packing of prompts pairs rows: each row reads
        // its own item's keys and keeps the reference arithmetic, and a CTA
        // stages one key tile for both rows only when their keys are the
        // same or shifted by one, otherwise one tile per row.
        let mut launches = self.reference.launches(buffers, workspace, counts)?;
        for launch in &mut launches {
            if launch.symbol == self.entry.symbol() {
                launch.symbol = sources_prefill::reuse_symbol(&self.entry, 2);
                launch.plan = sources_prefill::reuse_plan(&self.entry, counts.rows, 2)?;
            }
        }
        Ok(launches)
    }
}
