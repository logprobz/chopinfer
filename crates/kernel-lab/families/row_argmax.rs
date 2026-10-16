//! The shared sharded argmax (`kernels_sm120::row::argmax`): the F32 and
//! BF16 partials and the final under their kernel-base symbols, which the
//! conformance kit runs against the contract's oracle.

use kernel_build::CubinSpec;

pub const FAMILY: &str = "row_argmax";

pub fn cubins() -> Vec<CubinSpec> {
    kernels_sm120::row::argmax::lab_cubin_specs()
}
