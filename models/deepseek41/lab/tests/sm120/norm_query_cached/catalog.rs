//! The cached F32 query norm entries paired with the served query
//! norm/quantize entry they would replace.
use bundle_deepseek41::sm120::norm_quantize as registration;
use kernel_build::CubinSpec;
use kernels_sm120::row::{rmsnorm_quantize, rmsnorm_quantize_cached};
use model_deepseek41_contract::geometry::FLASH;

pub const FAMILY: &str = "norm_query_cached";
pub const MODULE: &str = "norm_query_cached_sm120";
/// Warps per cached CTA, in `rmsnorm_quantize_cached::Entry::symbols` order.
pub const WARPS: [u32; 3] = [8, 16, 32];

pub fn served() -> rmsnorm_quantize::Entry {
    registration::entries(&FLASH)
        .expect("Flash norm entries")
        .remove(1)
}
pub fn cached() -> rmsnorm_quantize_cached::Entry {
    rmsnorm_quantize_cached::Entry {
        reference: served(),
        prefix: format!("chop_norm_quantize_{}_query", FLASH.kernel_tag),
    }
}
pub fn specs() -> Vec<CubinSpec> {
    vec![cached().cubin_spec(MODULE).unwrap()]
}
