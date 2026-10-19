//! Prefill candidates, kept out of serving until their row ranges win
//! paired measurements. Heads gather smaller row sets and are not here.
use kernel_build::CubinSpec;
use kernels_sm120::f32_projection::{Entry, cubin_spec};
use model_deepseek41_contract::geometry::Geometry;

pub const FAMILY: &str = "f32_prefill";
pub const TILE_FEATURES: [u32; 2] = [32, 64];

pub fn baseline_entries(g: &Geometry) -> Result<Vec<Entry>, String> {
    let mut entries = super::f32_dense::entries(g)?;
    entries.extend(super::dspark_router::entries(g)?);
    Ok(entries)
}

pub fn cubin_name(g: &Geometry, features: u32) -> String {
    format!("f32_prefill_{}_n{features}_sm120", g.kernel_tag)
}

pub fn entries(g: &Geometry, features: u32) -> Result<Vec<Entry>, String> {
    baseline_entries(g)?
        .into_iter()
        .map(|entry| {
            entry
                .with_prefill_tiles(features)
                .map_err(|e| e.to_string())
        })
        .collect()
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<CubinSpec>, String> {
    geometries
        .iter()
        .flat_map(|g| TILE_FEATURES.map(move |features| (g, features)))
        .map(|(g, features)| {
            cubin_spec(cubin_name(g, features), &entries(g, features)?).map_err(|e| e.to_string())
        })
        .collect()
}
