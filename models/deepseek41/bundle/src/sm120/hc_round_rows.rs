//! Exact rounded projection weights shared across independent rows.
use kernel_build::CubinSpec;
use kernels_sm120::sinkhorn_hc::round_rows::Entry;
use model_deepseek41_contract::geometry::Geometry;

pub const FAMILY: &str = "sinkhorn_hc_round_rows";
pub fn cubin_name(g: &Geometry) -> String {
    format!("sinkhorn_hc_{}_round_rows_sm120", g.kernel_tag)
}
pub fn entry(g: &Geometry) -> Entry {
    Entry {
        reference: super::hc_round_chain::entry(g),
        prefix: format!("chop_{}_hc_round_rows", g.kernel_tag),
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
