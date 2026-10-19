//! The qualified coarse guard cubin, including its unused diagnostic entries.
//! Keeping those entries preserves the exact GPU-tested cubin bytes.
use super::indexer_mma_guard;
use kernel_build::CubinSpec;
use kernels_sm120::pooled_indexer::position_mma_guard_fast as fast;
use model_deepseek41_contract::geometry::Geometry;

pub const FAMILY: &str = "position_mma_guard_fast";
pub fn cubin_name(g: &Geometry) -> String {
    format!("position_mma_guard_fast_{}_sm120", g.kernel_tag)
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    Ok(indexer_mma_guard::cubins(geometries)?
        .into_iter()
        .map(|(_, spec)| {
            let entries = spec
                .entries
                .into_iter()
                .filter(|entry| entry.symbol.ends_with("_mma_fp4_guard_v1"))
                .flat_map(|entry| {
                    let score = fast::score(entry);
                    [score.clone(), fast::diagnostic(score)]
                })
                .collect();
            (
                FAMILY,
                fast::cubin_spec(
                    spec.name
                        .replace("position_mma_guard_", "position_mma_guard_fast_"),
                    entries,
                ),
            )
        })
        .collect())
}
