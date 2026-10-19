//! Qualified prefill norm and ceil fusion, with materialized BF16/F32 rows.
use kernel_build::CubinSpec;
use kernels_sm120::{
    backend::Sm120,
    row::rmsnorm_prefill::{self, Entry},
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

pub fn register(backend: Sm120, g: &Geometry) -> Result<Sm120, String> {
    if g != &FLASH {
        return Ok(backend);
    }
    let entry = entries()?.pop().expect("cached prefill norm");
    Ok(backend.with_rmsnorm_quantize(entry.reference))
}
