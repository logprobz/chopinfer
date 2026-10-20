//! Measured head sharing for full mixed prefill rows and decoded keys.
use kernel_build::CubinSpec;
use kernel_core::backend::{
    Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View,
};
use kernel_core::contract::{AttentionPhase, Refusal};
use kernel_core::latent_attention::{LatentAttentionQuery, LatentSourcesBuffers};
use kernels_sm120::latent::{sources, sources_prefill::heads, sources_prefill_cache};
use model_deepseek41_contract::geometry::FLASH;

pub const NATIVE_FAMILY: &str = "latent_prefill_head_shared";
pub const CACHED_FAMILY: &str = "latent_prefill_head_async";
pub const NATIVE_CUBIN: &str = "latent_sources_prefill_head_shared_deepseek41_sm120";
pub const CACHED_CUBIN: &str = "latent_sources_prefill_head_async_deepseek41_sm120";

fn entries(ranks: u32) -> Result<Vec<sources::Entry>, String> {
    Ok(super::latent::entries(&FLASH, ranks)?
        .into_iter()
        .filter(|e| {
            matches!(
                e.query,
                LatentAttentionQuery::Attention {
                    phase: AttentionPhase::Prefill,
                    second_fp4_keys: true,
                    ..
                }
            )
        })
        .collect())
}

/// Preserve the complete qualified module's entry order and binary identity.
pub fn cubins() -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let native = [1, 2, 4]
        .into_iter()
        .map(entries)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let decoded = native
        .iter()
        .map(sources_prefill_cache::decoded_entry)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    let mut all = native;
    all.extend(decoded.iter().cloned());
    let profile = super::latent::profile(&FLASH);
    Ok(vec![
        (
            NATIVE_FAMILY,
            heads::cubin_spec(profile, &all).map_err(|e| e.to_string())?,
        ),
        (
            CACHED_FAMILY,
            heads::asynchronous::cubin_spec(profile, &decoded, false).map_err(|e| e.to_string())?,
        ),
    ])
}

pub fn prepare(
    backend: &dyn Backend,
    query: &LatentAttentionQuery,
    need: Need,
) -> Result<Prepared<LatentSourcesBuffers<View>>, Refused> {
    let entry = entries(4)
        .map_err(Refused::Rule)?
        .into_iter()
        .find(|e| e.query == *query)
        .ok_or_else(|| Refused::Rule("unqualified shared-head prefill query".into()))?;
    let reference = backend.latent_attention_sources(query, need)?;
    let mut info = reference.info().clone();
    info.symbols
        .push((heads::symbol(&entry), heads::SHARED_BYTES));
    need.check(&info)?;
    Ok(Box::new(Shared {
        reference,
        info,
        entry,
    }))
}

struct Shared {
    reference: Prepared<LatentSourcesBuffers<View>>,
    info: OpInfo,
    entry: sources::Entry,
}

impl PreparedOp<LatentSourcesBuffers<View>> for Shared {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &LatentSourcesBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let mut launches = self.reference.launches(buffers, workspace, counts)?;
        if counts.rows != super::prefill_attention::ROWS {
            return Err(Refusal(
                "shared-head prefill requires a full qualified chunk",
            ));
        }
        for launch in &mut launches {
            if launch.symbol == self.entry.symbol() {
                launch.symbol = heads::symbol(&self.entry);
                launch.plan = heads::plan(&self.entry, counts.rows)?;
            }
        }
        Ok(launches)
    }
}
