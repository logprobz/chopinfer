//! Shared PDL live-count and input-copy probes, compiled once for the lab.
use kernel_build::CubinSpec;

pub const FAMILY: &str = "dependency_probes";
pub fn cubins() -> Vec<CubinSpec> {
    vec![kernels_sm120::probes::cubin_spec()]
}
