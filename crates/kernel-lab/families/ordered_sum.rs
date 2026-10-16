//! The exact parallel ordered F32 sum's test entries at the lab layouts and
//! their mutants (`row::ordered_sum::lab_cubin_specs`).

use kernel_build::CubinSpec;

pub const FAMILY: &str = "ordered_sum";

pub fn cubins() -> Vec<CubinSpec> {
    kernels_sm120::row::ordered_sum::lab_cubin_specs()
}
