//! Prefill window prefix registration, separate from the trained writer.
use kernel_build::CubinSpec;
use kernels_sm120::backend::Sm120;
use kernels_sm120::latent::window_stage::{self, Entry};
use model_deepseek41_contract::{geometry::Geometry, serving::PREFILL_TOKENS};
pub const FAMILY: &str = "latent_window_stage";
pub fn entry() -> Entry {
    Entry {
        kv: kernel_core::latent_attention::LatentKvFormat::Fp8E4M3Ue8M0Block32,
        capacity: PREFILL_TOKENS,
        symbol: "chop_deepseek41_window_stage_v1".into(),
        mutation: None,
    }
}
pub fn register(backend: Sm120, g: &Geometry, ranks: u32) -> Result<Sm120, String> {
    model_deepseek41_contract::admission::check_ranks(g, ranks)?;
    model_deepseek41_contract::latent_glue::window_stage(
        g,
        PREFILL_TOKENS + g.shape.attention.window - 1,
    )?;
    Ok(backend
        .with_latent_window_stage(entry())
        .with_latent_staged_window_keys(keys_entry()))
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    for g in geometries {
        model_deepseek41_contract::latent_glue::window_stage(
            g,
            PREFILL_TOKENS + g.shape.attention.window - 1,
        )?;
    }
    Ok(vec![(
        FAMILY,
        window_stage::cubin_spec("latent_window_stage_deepseek41_sm120", &[entry()]),
    )])
}

pub const KEY_FAMILY: &str = "latent_staged_window_keys";
pub fn keys_entry() -> kernels_sm120::latent::staged_window_keys::Entry {
    kernels_sm120::latent::staged_window_keys::Entry {
        capacity: PREFILL_TOKENS,
        symbol: "chop_deepseek41_staged_window_keys_v1".into(),
    }
}
pub fn key_cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    for g in geometries {
        model_deepseek41_contract::latent_glue::staged_window_keys(
            g,
            PREFILL_TOKENS + g.shape.attention.window - 1,
        )?;
    }
    Ok(vec![(
        KEY_FAMILY,
        kernels_sm120::latent::staged_window_keys::cubin_spec(
            "latent_staged_window_keys_deepseek41_sm120",
            &[keys_entry()],
        ),
    )])
}
