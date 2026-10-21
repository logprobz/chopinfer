//! The existing rank-ordered F32 owner sum, bound for prefill transport.

pub const FAMILY: &str = "comm_dma_sum";
pub const CUBIN: &str = "comm_dma_sum_glm5next_sm120";

pub fn cubin_spec() -> kernel_build::CubinSpec {
    kernels_sm120::comm::owned_sum_f32::cubin_spec(CUBIN.into())
}
