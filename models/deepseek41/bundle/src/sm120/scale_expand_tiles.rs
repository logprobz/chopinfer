//! The routed gate/up scale bank's tiled expansion: where the gate/up
//! consumers read the bank in row tiles (`RoutedLayout::Tiles`), an encoded
//! bank is decoded straight into its row tiles in scratch
//! (`kernels_sm120::moe::scale_expand::PAIR_TILES_SYMBOL`), the down bank
//! into rows beside it. The cubin is independent of geometry and topology,
//! in its own lock family beside the row-major expansion's.
use kernel_build::CubinSpec;

pub const FAMILY: &str = "routed_moe_scale_expand_tiles";
pub const CUBIN: &str = "moe_scale_expand_tiles_deepseek41_sm120";

pub fn cubin() -> CubinSpec {
    kernels_sm120::moe::scale_expand::tiles_cubin_spec(CUBIN)
}
