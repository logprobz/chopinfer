//! Keep the qualified cubin intact, including never-launched negative controls.
use super::indexer;
use kernel_build::CubinSpec;
use kernels_sm120::pooled_indexer::position_hierarchical as hierarchy;
use model_deepseek41_contract::geometry::Geometry;

pub const FAMILY: &str = "position_hierarchical";
pub fn cubin_name(g: &Geometry) -> String {
    format!("position_hierarchical_{}_sm120", g.kernel_tag)
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .filter(|g| g.native_indexer_supported())
        .map(|g| {
            let mut entries = Vec::new();
            for entry in indexer::entries(g) {
                let pair = hierarchy::dense_specs(&entry).map_err(|e| e.to_string())?;
                entries.push(hierarchy::negative(pair[0].clone()));
                entries.extend(pair);
            }
            for entry in indexer::block_entries(g) {
                let pair = hierarchy::compact_specs(&entry).map_err(|e| e.to_string())?;
                entries.push(hierarchy::negative(pair[0].clone()));
                entries.extend(pair);
            }
            Ok((FAMILY, hierarchy::cubin_spec(cubin_name(g), entries)))
        })
        .collect()
}
