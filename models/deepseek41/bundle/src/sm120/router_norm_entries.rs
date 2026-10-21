//! The router-in-norm serving cubin (`router_norm`), shared with the build
//! script and the qualification lab: the served cached MoE-input prefill
//! norm with its router certificate, and the served router entry's
//! certified projection at 32-row, 64-expert tiles.
use kernel_build::CubinSpec;
use kernels_sm120::f32_projection::{Entry, certified};
use kernels_sm120::row::rmsnorm_prefill;
use model_deepseek41_contract::geometry::{FLASH, Geometry};

use super::{f32_zero, norm_prefill};

pub const FAMILY: &str = "router_norm_serving";
pub const CUBIN: &str = "router_norm_deepseek41_sm120";
/// The served router's tile rows (`f32_zero`).
pub const TILE_ROWS: u32 = 32;
/// Experts per tile (`certified::wide_entry_spec`): each activation row
/// block is read 6 times instead of 12 (models/deepseek41/lab/router_wide.md).
pub const FEATURES: u32 = 64;

/// The served cached prefill norm, also writing the certificate.
pub fn norm() -> rmsnorm_prefill::Entry {
    let mut e = norm_prefill::entries()
        .expect("served prefill norm entries")
        .into_iter()
        .find(|e| e.cache_groups)
        .expect("the cached prefill norm");
    e.reference.symbol = format!(
        "{}_certified_v1",
        e.reference.symbol.trim_end_matches("_v1")
    );
    e.certify = true;
    e.validate().expect("certifying prefill norm");
    e
}

pub fn router() -> Entry {
    f32_zero::entry()
}

pub fn router_symbol() -> String {
    certified::wide_symbol(&router(), TILE_ROWS, FEATURES)
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<CubinSpec>, String> {
    if !geometries.contains(&FLASH) {
        return Ok(Vec::new());
    }
    let mut spec = certified::cubin_spec(CUBIN, &[], &[]).map_err(|e| e.to_string())?;
    spec.headers
        .push("#include \"gemm/exact_f32_certified_wide.cuh\"".into());
    spec.entries.push(
        certified::wide_entry_spec(&router(), TILE_ROWS, FEATURES).map_err(|e| e.to_string())?,
    );
    let norms =
        rmsnorm_prefill::cubin_spec("unused".into(), &[norm()]).map_err(|e| e.to_string())?;
    spec.headers.extend(norms.headers);
    spec.entries.extend(norms.entries);
    Ok(vec![spec])
}
