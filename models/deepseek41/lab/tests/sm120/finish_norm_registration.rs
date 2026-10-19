//! Lab-only specialization. No backend or serving catalog registration.
use kernel_build::CubinSpec;
use kernels_sm120::sinkhorn_hc::finish_norm::Entry;
use model_deepseek41_contract::geometry::Geometry;
pub const FAMILY: &str = "sinkhorn_hc_finish_norm";
pub fn cubin_name(g: &Geometry) -> String {
    format!("sinkhorn_hc_{}_finish_norm_sm120", g.kernel_tag)
}
pub fn entry(g: &Geometry) -> Entry {
    Entry {
        mixer: bundle_deepseek41::sm120::hc_round_expand_mix::entry(g),
        norm: bundle_deepseek41::sm120::norm_quantize_cached::entry(g),
        prefix: format!("chop_{}_hc_finish_norm", g.kernel_tag),
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
