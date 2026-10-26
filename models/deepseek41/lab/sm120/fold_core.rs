//! Unselected native fold core variants with their exact arithmetic controls.
use kernel_build::CubinSpec;
use kernels_sm120::gemm::{mxfp8_fold, mxfp8_prefill_matrix as prefill};
use model_deepseek41_contract::geometry::FLASH;
pub const FAMILY: &str = "fold_core";
pub const PREFILL: &str = "fold_core_prefill_sm120";
pub const TILES: &[(u32, u32)] = &[(32, 3), (64, 2), (64, 3), (128, 2)];
pub fn bases() -> Vec<mxfp8_fold::Entry> {
    bundle_deepseek41::sm120::mxfp8_fold::prefill_entries(&FLASH, 4)
        .unwrap()
        .into_iter()
        .filter(|e| e.query.k != 15360)
        .collect()
}
pub fn prefill_entries() -> Vec<prefill::Entry> {
    bases()
        .into_iter()
        .flat_map(|base| {
            TILES
                .iter()
                .map(move |&(m, s)| prefill::Entry::new(base.clone(), s, m).unwrap())
        })
        .collect()
}
pub fn cubins() -> Vec<CubinSpec> {
    vec![prefill::cubin_spec(PREFILL, &prefill_entries()).unwrap()]
}
