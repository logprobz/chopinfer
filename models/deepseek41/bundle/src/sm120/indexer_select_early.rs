//! The qualified early-stop prefill selectors, built as the lab built them.
use super::indexer;
use kernel_build::CubinSpec;
use kernels_sm120::pooled_indexer::position_select_early as early;
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const FAMILY: &str = "position_select_early";
pub const CUBIN: &str = "position_select_early_deepseek41_sm120";

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.contains(&FLASH) {
        return Ok(Vec::new());
    }
    let mut entries = Vec::new();
    for entry in indexer::entries(&FLASH) {
        entries.extend(early::dense(&entry).map_err(|e| e.to_string())?);
    }
    for entry in indexer::block_entries(&FLASH) {
        entries.extend(early::compact(&entry).map_err(|e| e.to_string())?);
    }
    Ok(vec![(FAMILY, early::cubin_spec(CUBIN, entries))])
}
