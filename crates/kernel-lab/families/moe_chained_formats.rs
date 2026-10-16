//! The chained engine's projections in formats the served catalog does not
//! hold, and their lab-only companions (`chained::formats`).

use kernel_build::CubinSpec;
use kernels_sm120::moe::chained::formats;

pub const FAMILY: &str = "moe_chained_formats";

pub fn cubins() -> Vec<CubinSpec> {
    let mut specs = formats::cubin_specs();
    specs.extend(formats::lab_cubin_specs());
    specs
}
