//! Decode rounded expansion and the next lagged mixer in a separate lock family.
use kernel_build::CubinSpec;
use kernel_core::sinkhorn_expand_mix::SinkhornExpandMixQuery;
use kernels_sm120::{
    backend::Sm120,
    sinkhorn_hc::round_expand_mix::{self as expand_mix, Entry},
};
use model_deepseek41_contract::{geometry::Geometry, hc, serving::DECODE_BATCH};
pub const FAMILY: &str = "sinkhorn_hc_round_expand_mix";
pub fn cubin_name(g: &Geometry) -> String {
    format!("sinkhorn_hc_{}_round_expand_mix_sm120", g.kernel_tag)
}
pub fn entry(g: &Geometry) -> Entry {
    Entry {
        query: SinkhornExpandMixQuery { mix: hc::mix(g) },
        capacity: if g.id == model_deepseek41_contract::geometry::FLASH.id {
            model_deepseek41_contract::serving::VERIFY_DECODE_ROWS
        } else {
            DECODE_BATCH
        },
        symbol: format!("chop_{}_hc_round_expand_mix", g.kernel_tag),
        probe: false,
    }
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            Ok((
                FAMILY,
                expand_mix::cubin_spec(&cubin_name(g), &[entry(g)]).map_err(|e| e.to_string())?,
            ))
        })
        .collect()
}
pub fn register(backend: Sm120, g: &Geometry) -> Sm120 {
    backend.with_sinkhorn_round_expand_mix(entry(g))
}
