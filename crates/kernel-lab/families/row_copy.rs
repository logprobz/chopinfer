//! The shared row copies (`kernels_sm120::row::copy`): the stream broadcast
//! and the launch-shape column slice under their kernel-base symbols, which
//! the lab compares with the model kernels they would replace.

use kernel_build::CubinSpec;

pub const FAMILY: &str = "row_copy";

pub fn cubins() -> Vec<CubinSpec> {
    kernels_sm120::row::copy::lab_cubin_specs()
}
