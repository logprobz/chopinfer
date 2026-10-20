//! Measured head sharing and exact output fusion for mixed prefill rows.
use kernel_build::{CubinSpec, EntryParam};
use kernel_core::backend::{
    Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View,
};
use kernel_core::contract::{Argument, AttentionPhase, Refusal};
use kernel_core::latent_attention::{LatentAttentionQuery, LatentSourcesBuffers};
use kernels_sm120::latent::{sources, sources_prefill::heads, sources_prefill_cache};
use model_deepseek41_contract::geometry::FLASH;

pub const NATIVE_FAMILY: &str = "latent_prefill_head_shared";
pub const CACHED_FAMILY: &str = "latent_prefill_head_async";
pub const NATIVE_CUBIN: &str = "latent_sources_prefill_head_shared_deepseek41_sm120";
pub const CACHED_CUBIN: &str = "latent_sources_prefill_head_async_deepseek41_sm120";
pub const FINISH_FAMILY: &str = "latent_prefill_head_finish";
pub const FINISH_CUBIN: &str = "latent_sources_prefill_head_finish_deepseek41_sm120";

pub fn finish_symbol(entry: &sources::Entry) -> String {
    heads::next::symbol(
        entry,
        entry.name.ends_with("_decoded"),
        heads::next::Mode::Finish,
    )
}

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
        (
            FINISH_FAMILY,
            heads::next::cubin_spec(profile, &all, heads::next::Mode::Finish)
                .map_err(|e| e.to_string())?,
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
    info.symbols
        .push((finish_symbol(&entry), heads::SHARED_BYTES));
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
        let launches = self.reference.launches(buffers, workspace, counts)?;
        let full_or_part = matches!(counts.rows, 8192 | 4096);
        let retained = (320..=2496).contains(&counts.rows)
            && counts.rows % 128 == 64
            && matches!(
                self.entry.query,
                LatentAttentionQuery::Attention { page_size: 64, .. }
            );
        if !(full_or_part || retained) {
            return Err(Refusal("unqualified fused prefill row count"));
        }
        finish(&self.entry, &self.entry, buffers, counts.rows, launches)
    }
}

/// The reference has already validated every native input and output span.
/// Keep its argument order, disable only the optional partial audit store,
/// and pass the same final views directly to the qualified one-split body.
pub fn finish(
    reference: &sources::Entry,
    entry: &sources::Entry,
    buffers: &LatentSourcesBuffers<View>,
    rows: u32,
    mut launches: Vec<Launch>,
) -> Result<Vec<Launch>, Refusal> {
    if launches.len() != 2
        || launches[0].symbol != reference.symbol()
        || launches[1].symbol != reference.merge_symbol()
        || reference.splits != 1
    {
        return Err(Refusal("unexpected fused prefill reference launches"));
    }
    let params = reference.params()?;
    if launches[0].arguments.0.len() != params.len() {
        return Err(Refusal("unexpected fused prefill reference arguments"));
    }
    let first = &mut launches[0];
    let partial = params
        .iter()
        .position(|p| matches!(p, EntryParam::DevicePointer { name, .. } if name == "workspace"))
        .ok_or(Refusal("missing fused prefill partial argument"))?;
    first.arguments.0[partial] = Argument::NullDevicePointer;
    for view in [buffers.sinks, buffers.cos_sin, buffers.output] {
        first.arguments.0.push(Argument::DeviceBuffer(
            view.ok_or(Refusal("missing fused prefill output view"))?
                .handle,
        ));
    }
    first.symbol = finish_symbol(entry);
    first.plan = heads::plan(entry, rows)?;
    launches.pop();
    Ok(launches)
}
