//! BF16-rounded ceil MXFP8 and FP4 row quantizers.

use kernel_build::CubinSpec;

pub const FAMILY: &str = "block_quantize";

pub fn cubins() -> Vec<CubinSpec> {
    use kernels_sm120::row::block_quantize::{LAB_CUBIN, cubin_spec, lab_entries};
    vec![cubin_spec(LAB_CUBIN, &lab_entries()).expect("block quantization lab entries")]
}
