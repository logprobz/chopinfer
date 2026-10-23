//! C16/C32 split-K folds whose K splits form one cluster
//! (`mxfp8_fold_cluster::run`), for the five attention projection shapes:
//! every fold's loads in flight at once and a rank-parallel ordered merge,
//! through the workspace or distributed shared memory.
use kernel_build::CubinSpec;
use kernels_sm120::gemm::{mxfp8_fold, mxfp8_fold_cluster::Entry};
use model_deepseek41_contract::geometry::FLASH;
pub const FAMILY: &str = "fold_cluster";
pub const CUBIN: &str = "fold_cluster_sm120";
/// (N, K, groups, rows, column tile, splits, folds in flight, distributed).
pub type Variant = (u32, u32, u32, u32, u32, u32, u32, bool);
#[rustfmt::skip]
pub const SHAPES: &[Variant] = &[
    (1280, 5120, 1, 16, 32, 8, 5, false),
    (1280, 5120, 1, 16, 32, 8, 5, true),
    (1280, 5120, 1, 32, 32, 8, 5, false),
    (1280, 5120, 1, 32, 32, 8, 5, true),
    (1280, 5120, 1, 32, 32, 8, 3, false),
    (1280, 5120, 1, 32, 64, 8, 5, false),
    (1280, 5120, 1, 32, 64, 8, 3, false),
    (1024, 4096, 2, 16, 32, 8, 4, false),
    (1024, 4096, 2, 16, 32, 8, 4, true),
    (1024, 4096, 2, 32, 32, 8, 4, false),
    (1024, 4096, 2, 32, 32, 8, 3, false),
    (1024, 4096, 2, 32, 32, 8, 3, true),
    (1024, 4096, 2, 32, 64, 8, 2, false),
    (1024, 4096, 2, 32, 64, 8, 3, false),
    (512, 5120, 1, 16, 16, 8, 5, false),
    (512, 5120, 1, 32, 16, 8, 5, false),
    (512, 5120, 1, 32, 32, 8, 5, false),
    (5120, 2048, 1, 16, 32, 2, 4, false),
    (5120, 2048, 1, 32, 32, 2, 4, false),
    (5120, 2048, 1, 32, 32, 2, 3, false),
    (8192, 1280, 1, 16, 32, 2, 3, false),
    (8192, 1280, 1, 32, 32, 2, 3, false),
];
pub fn entries() -> Vec<Entry> {
    let bases = bundle_deepseek41::sm120::mxfp8_fold::prefill_entries(&FLASH, 4).unwrap();
    SHAPES
        .iter()
        .map(
            |&(n, k, groups, rows, tile_n, splits, stages, distributed)| {
                let base = bases
                    .iter()
                    .find(|e| (e.query.n, e.query.k, e.query.groups) == (n, k, groups))
                    .unwrap();
                let mut q = base.query;
                q.m_capacity = rows;
                Entry {
                    base: mxfp8_fold::Entry::new(
                        q,
                        format!("chop_fold_cluster_n{n}_k{k}_g{groups}_f32"),
                    )
                    .unwrap(),
                    tile_n,
                    tile_rows: rows,
                    splits,
                    stages,
                    distributed,
                }
            },
        )
        .collect()
}
pub fn label(e: &Entry) -> String {
    format!(
        "cluster{}n{}c{}s{}{}",
        e.tile_rows,
        e.tile_n,
        e.splits,
        e.stages,
        if e.distributed { "d" } else { "" }
    )
}
pub fn cubins() -> Vec<CubinSpec> {
    vec![kernels_sm120::gemm::mxfp8_fold_cluster::cubin_spec(CUBIN, &entries(), true).unwrap()]
}
