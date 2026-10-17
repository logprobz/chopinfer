//! The shared gather by index (`kernels_sm120::row::gather`) under its
//! kernel-base symbol, which the lab compares with the model kernel it
//! replaces.

use kernel_build::CubinSpec;

pub const FAMILY: &str = "row_gather";

pub fn cubins() -> Vec<CubinSpec> {
    vec![kernels_sm120::row::gather::cubin_spec()]
}
