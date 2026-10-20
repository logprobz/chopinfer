//! Prefill F32 query norm/ceil composition and its lab ablations.
use kernel_build::CubinSpec;
use kernels_sm120::{
    backend::Sm120,
    row::rmsnorm_f32_prefill::{self, Entry, Mutation},
};
use model_deepseek41_contract::{
    geometry::{FLASH, Geometry},
    rows,
    serving::PREFILL_TOKENS,
};

pub const FAMILY: &str = "query_prefill_fused";
pub const CUBIN: &str = "query_prefill_fused_deepseek41_sm120";

pub fn entries() -> Result<Vec<Entry>, String> {
    [false, true]
        .into_iter()
        .map(|cached| {
            Entry::new(
                rows::norm_quantizers(&FLASH)[1],
                PREFILL_TOKENS,
                format!(
                    "chop_prefill_query_norm_ceil_{}_v1",
                    if cached { "cached" } else { "reread" }
                ),
                cached,
            )
            .map_err(|e| e.to_string())
        })
        .collect()
}

pub fn controls() -> Result<Vec<Entry>, String> {
    [
        ("round_before_weight", Mutation::RoundBeforeWeight),
        ("clear_status", Mutation::ClearStatus),
    ]
    .into_iter()
    .map(|(name, mutation)| {
        let mut entry = entries()?.pop().expect("cached query norm");
        entry.reference.symbol = format!("chop_prefill_query_control_{name}_v1");
        entry.mutation = mutation;
        Ok(entry)
    })
    .collect()
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.contains(&FLASH) {
        return Ok(Vec::new());
    }
    Ok(vec![(
        FAMILY,
        rmsnorm_f32_prefill::cubin_spec(CUBIN, &entries()?).map_err(|e| e.to_string())?,
    )])
}

pub fn register(backend: Sm120, g: &Geometry) -> Result<Sm120, String> {
    if g != &FLASH {
        return Ok(backend);
    }
    Ok(backend.with_rmsnorm_quantize(selected()?.reference))
}

/// Both strategies passed bitwise; reread ties cold and wins more hot cases.
pub fn selected() -> Result<Entry, String> {
    Ok(entries()?.remove(0))
}
