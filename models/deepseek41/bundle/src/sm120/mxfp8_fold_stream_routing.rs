//! Flash buckets that beat round 3 on both devices.
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
use model_deepseek41_contract::geometry::Geometry;

pub fn qualified(g: &Geometry, q: Mxfp8FoldGemmQuery, rows: u32, paired: bool) -> bool {
    super::previous::tuning_row(
        g,
        model_deepseek41_contract::tuning::Family::DenseFold,
        "dense_fold.stream.decode",
        q,
        paired,
        rows,
    )
    .is_some()
}
