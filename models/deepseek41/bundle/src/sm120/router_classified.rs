//! The `router_prepare_fusion` candidate's cubin: the router's cached exact
//! F32 decode producer that classifies its own rows and resets the router
//! finish's ticket (`kernels_sm120::f32_projection::classified_decode`),
//! which replaces the finish's prepare launch and the producer at 1 to 32
//! rows. Its chains, partials and arithmetic are the producer's. Flash
//! TP4 only; the serving side is `router_classified_serving`.
use kernel_build::CubinSpec;
use kernels_sm120::f32_projection::{Entry, classified_decode};
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const FAMILY: &str = "f32_router_classified";
pub const CUBIN: &str = "f32_router_classified_deepseek41_sm120";

/// The router's cached exact F32 entry: the experts from the hidden width.
pub fn entry() -> Entry {
    let s = &FLASH.shape;
    super::f32_dense::entries(&FLASH)
        .expect("dense F32 entries")
        .into_iter()
        .find(|e| e.query().n == s.moe.experts && e.query().k == s.hidden)
        .expect("the router's dense F32 entry")
        .with_decode()
}

/// Whether `geometry` at `ranks` has the classifying router producer.
pub fn serves(geometry: &Geometry, ranks: u32) -> bool {
    geometry.id == FLASH.id && ranks == 4
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| serves(g, 4)) {
        return Ok(Vec::new());
    }
    Ok(vec![(
        FAMILY,
        classified_decode::cubin_spec(CUBIN, &[entry()]).map_err(|e| e.to_string())?,
    )])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_classifying_producer_is_the_served_router_decode_stage() {
        let entry = entry();
        assert_eq!(
            entry.stage_symbol("_decode"),
            "chop_deepseek41_dense_f32_m8192_n384_k5120_v1_cached_decode_v1"
        );
        let spec = &cubins(&[FLASH]).unwrap()[0].1;
        assert_eq!(spec.name, CUBIN);
        assert_eq!(spec.entries.len(), 1);
        assert_eq!(
            spec.entries[0].symbol,
            "chop_deepseek41_dense_f32_m8192_n384_k5120_v1_cached_decode_classified_v1"
        );
        assert!(cubins(&[]).unwrap().is_empty());
        assert!(serves(&FLASH, 4) && !serves(&FLASH, 2));
    }
}
