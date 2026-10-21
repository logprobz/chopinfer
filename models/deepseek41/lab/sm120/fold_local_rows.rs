//! C8/C32 WQ_b probes that retain folds locally and read each weight once.
use kernel_build::CubinSpec;
use kernels_sm120::gemm::{mxfp8_fold, mxfp8_fold_local_rows::Entry};
use model_deepseek41_contract::geometry::FLASH;
pub const FAMILY: &str = "fold_local_rows";
pub const CUBIN: &str = "fold_local_rows_sm120";
pub fn entries() -> Vec<Entry> {
    let base = bundle_deepseek41::sm120::mxfp8_fold::prefill_entries(&FLASH, 4)
        .unwrap()
        .into_iter()
        .find(|e| (e.query.n, e.query.k, e.query.groups) == (8192, 1280, 1))
        .unwrap();
    [16, 32]
        .into_iter()
        .map(|tile_n| {
            let mut q = base.query;
            q.m_capacity = 32;
            Entry {
                base: mxfp8_fold::Entry::new(q, "chop_fold_local_n8192_k1280_g1_f32").unwrap(),
                tile_n,
            }
        })
        .collect()
}
pub fn cubins() -> Vec<CubinSpec> {
    vec![kernels_sm120::gemm::mxfp8_fold_local_rows::cubin_spec(CUBIN, &entries()).unwrap()]
}
