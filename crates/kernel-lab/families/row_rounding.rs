//! The partial rounding rows (`kernels_sm120::row::rounding`): the shared
//! round and widen entries at the lab's widths, which the kit composes
//! into the BF16-partials all-reduce where no phased entry serves.

use kernel_build::CubinSpec;

pub const FAMILY: &str = "row_rounding";

pub fn cubins() -> Vec<CubinSpec> {
    kernels_sm120::row::rounding::lab_cubin_specs()
}
