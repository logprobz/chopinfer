//! Qualified prefill norm and ceil fusion, with materialized BF16/F32 rows.
use kernel_build::CubinSpec;
use kernels_sm120::{
    backend::Sm120,
    row::{
        block_quantize,
        rmsnorm_prefill::{self, Entry},
    },
};
use model_deepseek41_contract::{
    geometry::{FLASH, Geometry},
    rows,
    serving::PREFILL_TOKENS,
};

pub const FAMILY: &str = "norm_prefill_fused";
pub const CUBIN: &str = "norm_prefill_deepseek41_sm120";

// Retain the complete qualified module, including its unselected ablation.
pub fn entries() -> Result<Vec<Entry>, String> {
    [false, true]
        .into_iter()
        .map(|cached| {
            Entry::new(
                rows::norm_quantizers(&FLASH)[0],
                PREFILL_TOKENS,
                format!(
                    "chop_prefill_norm_ceil_{}_{}_v1",
                    FLASH.kernel_tag,
                    if cached { "cached" } else { "recompute" }
                ),
                cached,
            )
            .map_err(|e| e.to_string())
        })
        .collect()
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.contains(&FLASH) {
        return Ok(Vec::new());
    }
    Ok(vec![(
        FAMILY,
        rmsnorm_prefill::cubin_spec(CUBIN.into(), &entries()?).map_err(|e| e.to_string())?,
    )])
}

/// The chain a fused launch replaces on `g` at `ranks`, for the lab's
/// comparison: the row RMSNorm's symbol, the widening of its rows and the
/// serial sublayer input quantizer that the unfused prefill launches.
pub fn chain(g: &Geometry, ranks: u32) -> Result<(String, String, block_quantize::Entry), String> {
    let width = g.shape.hidden;
    let quantizer = super::moe::quantizers(g, ranks)?
        .into_iter()
        .find(|entry| entry.query.width == width)
        .ok_or_else(|| format!("{} has no serial input quantizer", g.kernel_tag))?;
    Ok((
        super::rows::norm_symbol(g, width),
        super::reduce::rounding_symbol(g, kernels_sm120::row::rounding::Rounding::Widen),
        quantizer,
    ))
}

pub fn register(backend: Sm120, g: &Geometry) -> Result<Sm120, String> {
    if g != &FLASH {
        return Ok(backend);
    }
    let entry = selected()?;
    Ok(backend.with_rmsnorm_quantize(entry.reference))
}

fn selected() -> Result<Entry, String> {
    let cached = super::norm_quantize_cached::choice(
        &FLASH,
        "norm.input_strategy.prefill",
        FLASH.shape.hidden,
        PREFILL_TOKENS,
        1,
    )
    .is_some_and(|row| row.schedule == "cached");
    Ok(entries()?
        .into_iter()
        .find(|entry| entry.cache_groups == cached)
        .expect("prefill norm strategy"))
}
