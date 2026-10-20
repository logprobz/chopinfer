//! Preserve the qualified selector cubin, including its two source ratios.
use super::indexer;
use kernel_build::CubinSpec;
use kernels_sm120::pooled_indexer::position_select_cached as cached;
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const FAMILY: &str = "position_select_cached";
pub const CUBIN: &str = "position_select_cached_deepseek41_sm120";

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.contains(&FLASH) {
        return Ok(Vec::new());
    }
    let entries = indexer::entries(&FLASH)
        .iter()
        .map(cached::select)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    Ok(vec![(FAMILY, cached::cubin_spec(CUBIN, entries))])
}
