//! The routed expert scale banks' expansion: a bank the weight plan stores
//! encoded (`scale-row-offset-v1`) is decoded into its raw layout in scratch
//! before its consumers read it (`kernels_sm120::moe::scale_expand`). The
//! cubin is independent of geometry and topology, in its own lock family.
use kernel_build::CubinSpec;

pub const FAMILY: &str = "routed_moe_scale_expand";
pub const CUBIN: &str = "moe_scale_expand_deepseek41_sm120";

pub fn cubin() -> CubinSpec {
    kernels_sm120::moe::scale_expand::cubin_spec(CUBIN)
}
