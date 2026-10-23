//! Matrix-load prefill tiles reading `mxfp8-fold-packed-8x128-v1` weight
//! codes, beside the row-major entries they must equal bit for bit.

use kernel_build::CubinSpec;
use kernels_sm120::gemm::mxfp8_prefill_matrix;

pub const FAMILY: &str = "mxfp8_prefill_matrix_packed";
pub const CUBIN: &str = "chop_mxfp8_prefill_matrix_packed_lab";

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(
        &[CUBIN],
        vec![mxfp8_prefill_matrix::packed_lab_cubin_spec(CUBIN).unwrap()],
    )
}
