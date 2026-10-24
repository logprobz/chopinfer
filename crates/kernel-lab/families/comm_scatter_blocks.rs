//! Last-destination blocks with the ordinary F32 gather protocol.
use kernel_build::CubinSpec;
pub const FAMILY: &str = "comm_scatter_blocks";
pub const CUBIN: &str = "comm_scatter_blocks_sm120";
pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(&[CUBIN], vec![kernels_sm120::comm::scatter_blocks::cubin_spec(CUBIN)])
}
