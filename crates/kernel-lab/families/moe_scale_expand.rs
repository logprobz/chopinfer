//! Bit-exact scale bank expansion, independent of model geometry.
use kernel_build::CubinSpec;

pub const FAMILY: &str = "moe_scale_expand";
pub const CUBIN: &str = "chop_moe_scale_expand_lab";

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(
        &[CUBIN],
        vec![kernels_sm120::moe::scale_expand::cubin_spec(CUBIN)],
    )
}
