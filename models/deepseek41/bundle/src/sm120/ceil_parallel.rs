//! Decode registrations for the shared warp-parallel ceil quantizers.
use kernel_build::CubinSpec;
use kernels_sm120::{
    backend::Sm120,
    row::{block_quantize, silu_mul_quantize},
};
use model_deepseek41_contract::{
    admission::SUPPORTED_RANKS,
    dspark_options::SMALL_DRAFT_ROWS,
    geometry::{FLASH, Geometry},
    serving::DECODE_BATCH,
};

pub const FAMILY: &str = "ceil_parallel";

pub fn cubin_name(g: &Geometry) -> String {
    format!("ceil_parallel_{}_sm120", g.kernel_tag)
}

pub fn baseline_quantizers(g: &Geometry, ranks: u32) -> Result<Vec<block_quantize::Entry>, String> {
    Ok(super::moe::quantizers(g, ranks)?
        .into_iter()
        .chain(super::dense::quantizers(g, ranks)?)
        .chain(super::latent_glue::quantizers())
        .collect())
}

/// Flash TP4's WO_a and WO_b inputs: the attention output (8192 BF16
/// columns) and WO_a's output (2048 F32 columns). Their quantizers signal
/// dependents as soon as their own producer wait returns, so the following
/// projection streams its immutable weights while the row quantizes.
fn early_release(g: &Geometry, symbol: &str) -> bool {
    g.id == FLASH.id
        && [
            format!("chop_{}_dense_ceil_w8192_v1", g.kernel_tag),
            format!("chop_{}_dense_ceil_f32_w2048_v1", g.kernel_tag),
        ]
        .iter()
        .any(|s| s == symbol)
}

/// Flash TP4's bounded verify rows. Capacity is not compiled; each row
/// is quantized alone, including every drafter forward through 60 rows.
pub const CAPACITY: u32 = model_deepseek41_contract::serving::VERIFY_DECODE_ROWS;

/// Flash TP4 includes the target intermediate in its wide verify forms.
/// Other topologies retain their target decode and drafter row limits.
fn capacity(g: &Geometry, ranks: u32, entry: &block_quantize::Entry) -> u32 {
    if g.id == FLASH.id && ranks == 4 {
        return CAPACITY;
    }
    let main = model_deepseek41_contract::moe::expert_sets(g)[0];
    let intermediate = main.top_k * (g.shape.moe.intermediate / ranks);
    let draft = g.shape.dspark.top_k * (g.shape.moe.intermediate / ranks);
    if entry.symbol == format!("chop_{}_moe_ceil_w{intermediate}_v1", g.kernel_tag)
        && intermediate != draft
    {
        DECODE_BATCH
    } else {
        SMALL_DRAFT_ROWS
    }
}

pub fn quantizers(g: &Geometry, ranks: u32) -> Result<Vec<block_quantize::Entry>, String> {
    baseline_quantizers(g, ranks)?
        .into_iter()
        .map(|mut entry| {
            entry.capacity = capacity(g, ranks, &entry);
            if entry.symbol == "chop_deepseek41_latent_ceil32_v1" {
                entry.symbol = format!("chop_{}_latent_ceil32_v1", g.kernel_tag);
            }
            let early = early_release(g, &entry.symbol);
            entry.symbol = format!("{}_parallel_v1", entry.symbol.strip_suffix("_v1").unwrap());
            let entry = entry.parallel().map_err(|e| e.to_string())?;
            if early {
                entry.early_release().map_err(|e| e.to_string())
            } else {
                Ok(entry)
            }
        })
        .collect()
}

pub fn activation(g: &Geometry, ranks: u32) -> Result<silu_mul_quantize::Entry, String> {
    let mut entry = super::shared::activation(g, ranks)?;
    entry.capacity = if g.id == FLASH.id && ranks == 4 {
        CAPACITY
    } else {
        SMALL_DRAFT_ROWS
    };
    entry.symbol = format!("{}_parallel_v1", entry.symbol.strip_suffix("_v1").unwrap());
    entry.parallel().map_err(|e| e.to_string())
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let mut specs = Vec::new();
    for g in geometries {
        let mut quantizers = Vec::<block_quantize::Entry>::new();
        let mut activations = Vec::<silu_mul_quantize::Entry>::new();
        for &ranks in SUPPORTED_RANKS {
            for entry in self::quantizers(g, ranks)? {
                if !quantizers.iter().any(|e| e.symbol == entry.symbol) {
                    quantizers.push(entry);
                }
            }
            let entry = activation(g, ranks)?;
            if !activations.iter().any(|e| e.symbol == entry.symbol) {
                activations.push(entry);
            }
        }
        let mut spec =
            block_quantize::cubin_spec(&cubin_name(g), &quantizers).map_err(|e| e.to_string())?;
        spec.entries
            .extend(activations.iter().map(silu_mul_quantize::Entry::spec));
        specs.push((FAMILY, spec));
    }
    Ok(specs)
}

/// Preserve generic expert-chain selection while exposing wide entries
/// to explicit callers. Direct wide gates quantize in their own finisher.
pub(super) fn registered_quantizers(
    g: &Geometry,
    ranks: u32,
) -> Result<Vec<block_quantize::Entry>, String> {
    let mut entries = quantizers(g, ranks)?;
    let intermediate = g.shape.moe.top_k * (g.shape.moe.intermediate / ranks);
    let draft = g.shape.dspark.top_k * (g.shape.moe.intermediate / ranks);
    for entry in &mut entries {
        if entry.query == model_deepseek41_contract::moe::quantizer(intermediate)
            && intermediate != draft
        {
            // A generic capacity-64 expert chain can still serve small
            // live plans. Keep its previous quantizer at those plans.
            entry.capacity = DECODE_BATCH;
        }
    }
    Ok(entries)
}

pub fn register(mut backend: Sm120, g: &Geometry, ranks: u32) -> Result<Sm120, String> {
    for entry in registered_quantizers(g, ranks)? {
        backend = backend.with_block_quantize(entry);
    }
    Ok(backend.with_silu_mul_quantize(activation(g, ranks)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_target_quantization_keeps_its_prepared_capacity_selection() {
        use kernel_core::backend::{Backend, Need};

        let query = model_deepseek41_contract::moe::quantizer(3456);
        for register in [register, super::super::ceil_cached::register] {
            let baseline = baseline_quantizers(&FLASH, 4)
                .unwrap()
                .into_iter()
                .fold(Sm120::new(), Sm120::with_block_quantize);
            let backend = register(baseline, &FLASH, 4).unwrap();
            // A 12-row step uses a chain prepared for 64 rows. Widening
            // registration changes its quantizer even at that small step.
            for capacity in 1..=CAPACITY {
                let op = backend
                    .block_quantize(&query, Need::rows(capacity))
                    .unwrap();
                let symbol = if capacity <= DECODE_BATCH {
                    "chop_deepseek41_moe_ceil_w3456_parallel_v1"
                } else {
                    "chop_deepseek41_moe_ceil_w3456_v1"
                };
                assert_eq!(op.info().symbols, [(symbol.into(), 0)]);
            }
        }
    }

    #[test]
    fn wide_verify_includes_the_target_intermediate_quantizer() {
        for entry in quantizers(&FLASH, 4).unwrap() {
            assert_eq!(entry.capacity, CAPACITY);
            for rows in 33..=CAPACITY {
                assert!(entry.plan(rows).is_ok());
            }
            assert!(entry.plan(CAPACITY + 1).is_err());
        }
        let shared = activation(&FLASH, 4).unwrap();
        assert_eq!(shared.capacity, CAPACITY);
        assert!(shared.plan(CAPACITY).is_ok());
        assert!(shared.plan(CAPACITY + 1).is_err());
    }

    /// Only Flash's WO_a and WO_b input quantizers release dependents early,
    /// and their cached twins inherit it.
    #[test]
    fn only_the_wo_inputs_release_dependents_early() {
        for &ranks in SUPPORTED_RANKS {
            let early: Vec<_> = quantizers(&FLASH, ranks)
                .unwrap()
                .into_iter()
                .filter(|e| e.early_release)
                .map(|e| e.symbol)
                .collect();
            if ranks == 4 {
                assert_eq!(
                    early,
                    [
                        "chop_deepseek41_dense_ceil_w8192_parallel_v1",
                        "chop_deepseek41_dense_ceil_f32_w2048_parallel_v1"
                    ]
                );
            }
            for entry in quantizers(&FLASH, ranks).unwrap() {
                if let Some(cached) = super::super::ceil_cached::cached(&entry) {
                    assert_eq!(cached.early_release, entry.early_release);
                }
            }
        }
    }
}
