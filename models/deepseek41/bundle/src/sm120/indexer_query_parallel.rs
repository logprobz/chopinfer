//! Qualified small-row query registration; larger buckets keep the original path.
use super::indexer_query as baseline;
use kernel_build::CubinSpec;
use kernels_sm120::{backend::Sm120, pooled_indexer::position_query_parallel};
use model_deepseek41_contract::geometry::Geometry;
pub const PARALLEL_FAMILY: &str = "position_query_parallel";
pub fn parallel_cubin_name(g: &Geometry) -> String {
    format!("position_query_parallel_{}_sm120", g.kernel_tag)
}
pub fn parallel_cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            position_query_parallel::cubin(&parallel_cubin_name(g), &[baseline::entry(g)])
                .map(|c| (PARALLEL_FAMILY, c))
                .map_err(|e| e.0.into())
        })
        .collect()
}
pub fn register_parallel(sm120: Sm120, g: &Geometry) -> Sm120 {
    baseline::register(sm120, g).with_parallel_position_queries()
}
