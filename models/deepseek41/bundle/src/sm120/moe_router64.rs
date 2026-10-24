//! Separate domain and partial spans for the wide backbone router finish.
use kernel_build::CubinSpec;
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const FAMILY: &str = "routed_moe_router_finish64";
pub const CUBIN: &str = "router_finish_moe_deepseek41_m64_sm120";

pub fn enabled(geometry: &Geometry, ranks: u32, verify_rows: u32) -> bool {
    geometry.id == FLASH.id && ranks == 4 && verify_rows > 32
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(Vec::new());
    }
    let entries = super::moe::floor_entries(&FLASH)?
        .into_iter()
        .filter(|e| e.experts() == 384 && e.top_k() == 6)
        .collect::<Vec<_>>();
    Ok(vec![(
        FAMILY,
        kernels_sm120::moe::chained::router_finish64::cubin_spec(CUBIN, &entries),
    )])
}
