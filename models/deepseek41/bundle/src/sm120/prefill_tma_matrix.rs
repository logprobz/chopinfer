//! One captured Flash TP4 prefill bucket over the qualified TMA module. It
//! reads row-major weight codes and serves no instance: the staged-scale
//! TMA prefill (`prefill_tma_scales`) serves the bucket and reads the packed
//! codes. The lab measures against it.
use kernel_build::CubinSpec;
use kernels_sm120::gemm::mxfp8_prefill_tma_matrix::{self, Entry};
use model_deepseek41_contract::geometry::{FLASH, Geometry};
pub const FAMILY: &str = "mxfp8_prefill_tma_matrix_reference";
pub const CUBIN: &str = "fold_tma_matrix_deepseek41_sm120";
pub fn entries() -> Vec<Entry> {
    super::mxfp8_fold::prefill_entries(&FLASH, 4)
        .unwrap()
        .into_iter()
        .filter(|e| {
            [
                (1280, 5120),
                (8192, 1280),
                (1024, 4096),
                (5120, 2048),
                (25600, 6144),
            ]
            .contains(&(e.query.n, e.query.k))
        })
        .flat_map(|base| {
            [2, 3].map(move |stages| Entry {
                base: base.clone(),
                stages,
            })
        })
        .collect()
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(Vec::new());
    }
    Ok(vec![(
        FAMILY,
        mxfp8_prefill_tma_matrix::cubin_spec(CUBIN, &entries()).map_err(|e| e.to_string())?,
    )])
}
