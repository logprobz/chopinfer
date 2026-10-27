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
    use kernels_sm120::gemm::dense_decode::Finish;
    let knobs = super::dense_knobs(g, q, rows, paired).expect("valid dense decode row");
    if knobs.finish == Finish::Merge {
        DecodePath::Split
    } else if knobs.weight_prefetch {
        DecodePath::Prefetch
    } else {
        DecodePath::Ticket
    }
}
