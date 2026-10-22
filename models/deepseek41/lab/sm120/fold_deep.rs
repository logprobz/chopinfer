//! C16/C32 row-local folds with a refilling ring of several folds in flight
//! (`mxfp8_fold_local_rows::run_deep`), for the WQ_b and WO_b decode shapes.
use kernel_build::CubinSpec;
use kernels_sm120::gemm::{mxfp8_fold, mxfp8_fold_local_rows::Entry};
use model_deepseek41_contract::geometry::FLASH;
pub const FAMILY: &str = "fold_deep";
pub const CUBIN: &str = "fold_deep_sm120";
/// (N, K, groups, column tile, folds in flight).
pub const SHAPES: &[(u32, u32, u32, u32, u32)] = &[
    (8192, 1280, 1, 32, 5),
    (8192, 1280, 1, 16, 6),
    (5120, 2048, 1, 32, 5),
    (5120, 2048, 1, 16, 6),
];
pub fn entries() -> Vec<Entry> {
    let bases = bundle_deepseek41::sm120::mxfp8_fold::prefill_entries(&FLASH, 4).unwrap();
    SHAPES
        .iter()
        .map(|&(n, k, groups, tile_n, stages)| {
            let base = bases
                .iter()
                .find(|e| (e.query.n, e.query.k, e.query.groups) == (n, k, groups))
                .unwrap();
            let mut q = base.query;
            q.m_capacity = 32;
            Entry {
                base: mxfp8_fold::Entry::new(q, format!("chop_fold_deep_n{n}_k{k}_g{groups}_f32"))
                    .unwrap(),
                tile_n,
                stages,
            }
        })
        .collect()
}
pub fn cubins() -> Vec<CubinSpec> {
    vec![kernels_sm120::gemm::mxfp8_fold_local_rows::cubin_spec(CUBIN, &entries()).unwrap()]
}
