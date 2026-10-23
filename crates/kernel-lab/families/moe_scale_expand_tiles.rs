//! The tiled paired scale expansion: the paired entry's bytes written as
//! the row-tiled gate/up scale section, independent of model geometry.
use kernel_build::CubinSpec;

pub const FAMILY: &str = "moe_scale_expand_tiles";
pub const CUBIN: &str = "chop_moe_scale_expand_tiles_lab";

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(
        &[CUBIN],
        vec![kernels_sm120::moe::scale_expand::tiles_cubin_spec(CUBIN)],
    )
}
