//! Measured decode selection; all entries retain the same fold arithmetic.
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
use model_deepseek41_contract::geometry::Geometry;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodePath {
    Split,
    Ticket,
    Prefetch,
}

pub fn decode_path(g: &Geometry, q: Mxfp8FoldGemmQuery, rows: u32, paired: bool) -> DecodePath {
    match super::tuning_row(
        g,
        model_deepseek41_contract::tuning::Family::DenseFold,
        "dense_fold.fold.decode",
        q,
        paired,
        rows,
    )
    .map(|r| r.schedule.as_str())
    {
        Some("split") => DecodePath::Split,
        Some("prefetch") => DecodePath::Prefetch,
        None => DecodePath::Ticket,
        Some(schedule) => panic!("unknown fold schedule {schedule}"),
    }
}
