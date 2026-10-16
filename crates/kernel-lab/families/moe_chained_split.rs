//! The chained engine's routed experts split over the ranks
//! (`chained::split`): the served forms, routes and route sums.

use kernel_build::CubinSpec;
use kernels_sm120::moe::chained::split;

pub const FAMILY: &str = "moe_chained_split";

pub fn cubins() -> Vec<CubinSpec> {
    split::cubin_specs().expect("the split cubins are supported")
}
