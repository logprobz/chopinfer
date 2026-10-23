//! Native MXFP8 folds reading `mxfp8-fold-packed-8x128-v1` weight codes,
//! beside the row-major entries they must equal bit for bit.

use kernel_build::CubinSpec;
use kernels_sm120::gemm::mxfp8_fold;

pub const FAMILY: &str = "mxfp8_fold_packed";
pub const CUBIN: &str = "chop_mxfp8_fold_packed_lab";

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(
        &[CUBIN],
        vec![mxfp8_fold::packed_lab_cubin_spec(CUBIN).unwrap()],
    )
}
