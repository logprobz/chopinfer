use kernel_build::CubinSpec;
use kernel_core::latent_attention::LatentKvFormat;
use kernels_sm120::{
    backend::Sm120,
    latent::window_fused::{self, Entry},
};
use model_deepseek41_contract::serving::DECODE_BATCH;

pub const FAMILY: &str = "window_fused";
pub const CUBIN: &str = "window_fused_deepseek41_sm120";
pub fn entries() -> Vec<Entry> {
    [
        (LatentKvFormat::Fp8E4M3Ue8M0Block32, "native"),
        (LatentKvFormat::Bf16, "bf16"),
    ]
    .into_iter()
    .map(|(kv, tag)| Entry {
        kv,
        capacity: DECODE_BATCH,
        symbol: format!("chop_deepseek41_window_fused_{tag}_v1"),
    })
    .collect()
}
pub fn register(backend: Sm120) -> Sm120 {
    entries()
        .into_iter()
        .fold(backend, Sm120::with_latent_window_fused)
}
pub fn cubin() -> Result<CubinSpec, String> {
    window_fused::cubin_spec(CUBIN, &entries()).map_err(|e| e.to_string())
}
