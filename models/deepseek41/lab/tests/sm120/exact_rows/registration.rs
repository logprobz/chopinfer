//! The same row-tiled producer declaration the bundle embeds.
pub const FAMILY: &str = "sinkhorn_hc_round_rows";
pub const CUBIN: &str = "sinkhorn_hc_deepseek41_round_rows_sm120";
pub fn entry() -> kernels_sm120::sinkhorn_hc::round_rows::Entry {
    bundle_deepseek41::sm120::hc_round_rows::entry(&model_deepseek41_contract::geometry::FLASH)
}
