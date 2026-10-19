//! Guarded native FP4 scoring; all selection entries remain separately locked.
use super::indexer;
use kernel_build::CubinSpec;
use kernels_sm120::pooled_indexer::position_mma_guard;
use model_deepseek41_contract::geometry::Geometry;

pub const FAMILY: &str = "position_mma_guard";
pub fn cubin_name(g: &Geometry) -> String {
    format!("position_mma_guard_{}_sm120", g.kernel_tag)
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .filter(|g| g.native_indexer_supported())
        .map(|g| {
            let mut entries = Vec::new();
            for entry in indexer::entries(g) {
                entries.extend(position_mma_guard::dense_specs(&entry).map_err(|e| e.0)?);
            }
            for entry in indexer::block_entries(g) {
                entries.extend(position_mma_guard::compact_specs(&entry).map_err(|e| e.0)?);
            }
            Ok((
                FAMILY,
                position_mma_guard::cubin_spec(cubin_name(g), entries),
            ))
        })
        .collect()
}
