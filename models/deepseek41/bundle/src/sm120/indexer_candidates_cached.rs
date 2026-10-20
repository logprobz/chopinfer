//! The qualified fused candidate publisher, built as the lab built it.
use super::indexer;
use kernel_build::CubinSpec;
use kernels_sm120::pooled_indexer::position_candidates_cached as cached;
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const FAMILY: &str = "position_candidates_cached";
pub const CUBIN: &str = "position_candidates_cached_sm120";

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.contains(&FLASH) {
        return Ok(Vec::new());
    }
    let spec = cached::spec(&indexer::candidates(&FLASH)).map_err(|e| e.to_string())?;
    Ok(vec![(FAMILY, cached::cubin_spec(CUBIN, vec![spec]))])
}
