//! The opt-in prefill module retains the exact lab entry inventory.
use super::indexer;
use kernel_build::CubinSpec;
use kernels_sm120::pooled_indexer::{
    position_mma_guard as guard, position_mma_guard_fast as fast, position_mma_prefill_tile as tile,
};
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const FAMILY: &str = "position_mma_prefill_tile";
pub const CUBIN: &str = tile::CUBIN;

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.contains(&FLASH) {
        return Ok(Vec::new());
    }
    let mut scores = Vec::new();
    for entry in indexer::entries(&FLASH) {
        let score = guard::dense_specs(&entry)
            .map_err(|e| e.to_string())?
            .remove(1);
        let mut probe = guard::diagnostic(score.clone());
        probe.symbol.push_str("_prefill_probe_v1");
        scores.push(probe);
        let mut probe = fast::diagnostic(fast::score(score));
        probe.symbol.push_str("_prefill_probe_v1");
        scores.push(probe);
    }
    for entry in indexer::block_entries(&FLASH) {
        let score = guard::compact_specs(&entry)
            .map_err(|e| e.to_string())?
            .remove(1);
        let mut probe = guard::diagnostic(score.clone());
        probe.symbol.push_str("_prefill_probe_v1");
        scores.push(probe);
        let mut probe = fast::diagnostic(fast::score(score));
        probe.symbol.push_str("_prefill_probe_v1");
        scores.push(probe);
        let score = tile::compact_spec(&entry).map_err(|e| e.to_string())?;
        scores.push(score.clone());
        scores.push(tile::diagnostic(score));
    }
    let spec = tile::cubin_spec(scores);
    assert_eq!(spec.name, CUBIN);
    Ok(vec![(FAMILY, spec)])
}
