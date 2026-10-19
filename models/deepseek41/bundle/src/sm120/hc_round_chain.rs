//! Exact rounded HC producers with the served partial workspace layout.
use kernel_build::CubinSpec;
use kernels_sm120::backend::Sm120;
use kernels_sm120::sinkhorn_hc::round_chain_partials::Entry;
use model_deepseek41_contract::geometry::Geometry;
pub const FAMILY: &str = "sinkhorn_hc_round_chain";
pub fn cubin_name(g: &Geometry) -> String {
    format!("sinkhorn_hc_{}_round_chain_sm120", g.kernel_tag)
}
pub fn entry(g: &Geometry) -> Entry {
    Entry {
        reference: super::hc_round_expand_mix::entry(g),
        prefix: format!("chop_{}_hc_round_chain", g.kernel_tag),
    }
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            Ok((
                FAMILY,
                entry(g)
                    .cubin_spec(&cubin_name(g))
                    .map_err(|e| e.to_string())?,
            ))
        })
        .collect()
}

/// Select only the measured Flash decode buckets.
pub fn register(backend: Sm120, g: &Geometry) -> Sm120 {
    if g.id == model_deepseek41_contract::geometry::FLASH.id {
        backend.with_sinkhorn_round_chain(entry(g), vec![(1, 8), (8, 4)])
    } else {
        backend
    }
}
