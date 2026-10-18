//! A scale-only narrowing control with finite low-address aliases.
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::AttentionPhase;
use kernel_core::latent_attention::{LatentAttentionOutput, LatentAttentionQuery};
use kernels_sm120::latent::sources::{self, Entry};
use crate::latent_sources_pages;

pub const FAMILY: &str = "latent_sources_scale_narrow";
pub fn entry_spec(entry: &Entry) -> EntrySpec {
    let mut spec = entry.entry_specs().unwrap().remove(0);
    spec.symbol = format!("chop_latent_sources_{}_scale_u32_v1", entry.name);
    spec.body = spec.body.replace(">(q,", ",chop::latent::scale_narrow_control::Address>(q,");
    spec
}
pub fn cubins() -> Vec<CubinSpec> {
    let entries = latent_sources_pages::catalog().into_iter().filter(|e| matches!(e.query,
        LatentAttentionQuery::Attention { phase: AttentionPhase::Decode, output: LatentAttentionOutput::Normalized, .. })).collect::<Vec<_>>();
    let mut spec = sources::cubin_spec(latent_sources_pages::PROFILE, &entries).unwrap();
    spec.name = "latent_sources_latent_native_scale_narrow_sm120".into();
    spec.headers.push("#include \"latent/scale_narrow_control.cuh\"".into());
    spec.entries = entries.iter().map(entry_spec).collect();
    crate::declared(&["latent_sources_latent_native_scale_narrow_sm120"], vec![spec])
}
