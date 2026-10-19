//! Decode registrations for the shared warp-parallel ceil quantizers.
use kernel_build::CubinSpec;
use kernels_sm120::{
    backend::Sm120,
    row::{block_quantize, silu_mul_quantize},
};
use model_deepseek41_contract::{
    admission::SUPPORTED_RANKS, geometry::Geometry, serving::DECODE_BATCH,
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

pub fn quantizers(g: &Geometry, ranks: u32) -> Result<Vec<block_quantize::Entry>, String> {
    baseline_quantizers(g, ranks)?
        .into_iter()
        .map(|mut entry| {
            entry.capacity = DECODE_BATCH;
            if entry.symbol == "chop_deepseek41_latent_ceil32_v1" {
                entry.symbol = format!("chop_{}_latent_ceil32_v1", g.kernel_tag);
            }
            entry.symbol = format!("{}_parallel_v1", entry.symbol.strip_suffix("_v1").unwrap());
            entry.parallel().map_err(|e| e.to_string())
        })
        .collect()
}

pub fn activation(g: &Geometry, ranks: u32) -> Result<silu_mul_quantize::Entry, String> {
    let mut entry = super::shared::activation(g, ranks)?;
    entry.capacity = DECODE_BATCH;
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

pub fn register(mut backend: Sm120, g: &Geometry, ranks: u32) -> Result<Sm120, String> {
    for entry in quantizers(g, ranks)? {
        backend = backend.with_block_quantize(entry);
    }
    Ok(backend.with_silu_mul_quantize(activation(g, ranks)?))
}
