//! Preserve the qualified module byte for byte; serving uses only tile4 mode0.
use super::indexer;
use kernel_build::CubinSpec;
use kernels_sm120::pooled_indexer::position_mma_tile as tile;
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const FAMILY: &str = "position_mma_tile";
pub const CUBIN: &str = "position_mma_tile_deepseek41_sm120";

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.contains(&FLASH) {
        return Ok(Vec::new());
    }
    let entries = indexer::entries(&FLASH)
        .iter()
        .flat_map(|entry| tile::VARIANTS.map(|variant| variant.spec(entry)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    let mut spec = tile::cubin_spec(entries);
    // Keep the qualified exports and bytes; the module belongs to Flash alone.
    spec.name = CUBIN.into();
    Ok(vec![(FAMILY, spec)])
}
