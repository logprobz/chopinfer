//! Model-owned registration for the shared native latent glue operations.
use kernel_build::CubinSpec;
use kernel_core::contract::DType;
use kernel_core::precision::quantizer::{BlockQuantization, BlockQuantizeQuery, QuantizeOutput};
use kernels_sm120::backend::Sm120;
use kernels_sm120::latent::glue::{self, Entry, Kernel};
use kernels_sm120::row::block_quantize;
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::serving::PREFILL_TOKENS;

pub const FAMILY: &str = "latent_glue";
pub fn entries() -> Vec<Entry> {
    [
        (Kernel::QueryRope, "query_rope"),
        (Kernel::WindowRotate, "window_rope"),
        (Kernel::WindowStore, "window_store"),
        (Kernel::WindowKeys, "window_keys"),
    ]
    .into_iter()
    .map(|(kernel, tag)| Entry {
        kernel,
        capacity: PREFILL_TOKENS,
        symbol: format!("chop_deepseek41_latent_{tag}_v1"),
    })
    .collect()
}
pub fn quantizers() -> Vec<block_quantize::Entry> {
    vec![
        block_quantize::Entry::new(
            BlockQuantizeQuery {
                width: 512,
                input: DType::F32,
                format: BlockQuantization::E4M3_CEIL32,
                output: QuantizeOutput::Codes,
            },
            PREFILL_TOKENS,
            "chop_deepseek41_latent_ceil32_v1",
        )
        .unwrap(),
    ]
}
fn check(g: &Geometry) -> Result<(), String> {
    if g.shape.attention.head_dim != 512 || g.shape.attention.rope_dim != 64 {
        return Err("native latent glue requires a 64-channel tail in 512 channels".into());
    }
    Ok(())
}
/// `g`'s query RMSNorm and its symbol, which `register` registers.
pub fn query_norm(g: &Geometry) -> (kernel_core::rmsnorm::RmsNormRequirement, String) {
    (
        model_deepseek41_contract::latent_glue::query_norm(g),
        norm_symbol(g),
    )
}
pub fn register(backend: Sm120, g: &Geometry, ranks: u32) -> Result<Sm120, String> {
    model_deepseek41_contract::admission::check_ranks(g, ranks)?;
    check(g)?;
    let (requirement, symbol) = query_norm(g);
    let backend = backend.with_rmsnorm(requirement, symbol);
    let backend = entries().into_iter().fold(backend, Sm120::with_latent_glue);
    Ok(quantizers()
        .into_iter()
        .fold(backend, Sm120::with_block_quantize))
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    for g in geometries {
        check(g)?;
    }
    let mut specs = vec![
        (
            FAMILY,
            glue::cubin_spec("latent_glue_deepseek41_sm120", &entries()),
        ),
        (
            FAMILY,
            block_quantize::cubin_spec("latent_glue_quantize_deepseek41_sm120", &quantizers())
                .map_err(|e| e.to_string())?,
        ),
    ];
    for g in geometries {
        let norm = kernels_sm120::row::rmsnorm::RmsNorm::new(
            model_deepseek41_contract::latent_glue::query_norm(g),
            PREFILL_TOKENS,
            kernels_sm120::row::rmsnorm::RmsNormTarget::SM120F,
        )
        .map_err(|e| e.to_string())?;
        specs.push((
            FAMILY,
            norm.cubin_spec(
                format!("latent_q_norm_{}_sm120", g.kernel_tag),
                norm_symbol(g),
            ),
        ));
    }
    Ok(specs)
}

fn norm_symbol(g: &Geometry) -> String {
    format!("chop_{}_q_norm_f32_v1", g.kernel_tag)
}
