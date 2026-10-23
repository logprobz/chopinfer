//! Exact cached ceil groups with the original RMS tree.
use kernel_build::CubinSpec;
use kernels_sm120::backend::Sm120;
use kernels_sm120::row::rmsnorm_quantize_cached::Entry;
use model_deepseek41_contract::geometry::Geometry;
pub const FAMILY: &str = "norm_quantize_cached";
pub fn cubin_name(g: &Geometry) -> String {
    format!("norm_quantize_cached_{}_sm120", g.kernel_tag)
}
pub fn entry(g: &Geometry) -> Entry {
    Entry {
        reference: super::norm_quantize::entries(g)
            .expect("input norm entry")
            .remove(0),
        prefix: format!("chop_norm_quantize_{}_input", g.kernel_tag),
        early_release: false,
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
        backend.with_rmsnorm_quantize_cached(entry(g), vec![(1, 32), (8, 32)])
    } else {
        backend
    }
}
