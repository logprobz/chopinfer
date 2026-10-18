//! Plain F32-input RMSNorm with an F32 output boundary.
use kernel_build::CubinSpec;
use kernel_core::contract::{DType, RmsNormKind};
use kernel_core::rmsnorm::RmsNormRequirement;
use kernels_sm120::row::rmsnorm::{RmsNorm, RmsNormTarget};

pub const FAMILY: &str = "rmsnorm_f32";
pub const WIDTHS: [u32; 3] = [512, 1280, 1536];
pub fn requirement(width: u32) -> RmsNormRequirement {
    RmsNormRequirement {
        input: DType::F32,
        output: DType::F32,
        ..RmsNormRequirement::bf16_backend_chosen(width, 1e-20, RmsNormKind::Plain, false)
    }
}
pub fn entries() -> Vec<(RmsNormRequirement, String)> {
    WIDTHS.into_iter().map(|width| (requirement(width), format!("chop_rmsnorm_f32_w{width}_v1"))).collect()
}
pub fn cubins() -> Vec<CubinSpec> {
    let entries = entries();
    let norm = |req| RmsNorm::new(req, 32, RmsNormTarget::SM120F).unwrap();
    let mut spec = norm(entries[0].0).cubin_spec("rmsnorm_f32_sm120", &entries[0].1);
    spec.entries = entries.into_iter().map(|(req, symbol)| norm(req).entry(symbol)).collect();
    vec![spec]
}
