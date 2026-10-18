//! BF16 comparison storage of the native window quantizer's exact products.
use kernel_build::CubinSpec;
use kernel_core::latent_attention::LatentKvFormat;
use kernels_sm120::backend::Sm120;
use kernels_sm120::latent::{glue, window_stage};
use model_deepseek41_contract::{geometry::Geometry, serving::PREFILL_TOKENS};
pub const FAMILY: &str = "latent_window_bf16";
pub fn store() -> glue::Entry {
    glue::Entry {
        kernel: glue::Kernel::WindowStoreBf16,
        capacity: PREFILL_TOKENS,
        symbol: "chop_deepseek41_window_store_bf16_v1".into(),
    }
}
pub fn stage() -> window_stage::Entry {
    window_stage::Entry {
        kv: LatentKvFormat::Bf16,
        capacity: PREFILL_TOKENS,
        symbol: "chop_deepseek41_window_stage_bf16_v1".into(),
        mutation: None,
    }
}
pub fn register(backend: Sm120) -> Sm120 {
    backend
        .with_latent_glue(store())
        .with_latent_window_stage(stage())
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    for g in geometries {
        if g.shape.attention.head_dim != 512 {
            return Err("BF16 window storage requires 512 channels".into());
        }
    }
    Ok(vec![
        (
            FAMILY,
            glue::cubin_spec("latent_window_bf16_store_deepseek41_sm120", &[store()]),
        ),
        (
            FAMILY,
            window_stage::cubin_spec("latent_window_bf16_stage_deepseek41_sm120", &[stage()]),
        ),
    ])
}
