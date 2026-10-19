//! Large-bucket dense quantizers. Decode and verify retain the separately
//! registered parallel entries at DECODE_BATCH capacity. MoE and latent
//! quantizers remain with their owning registrations.
use kernel_build::CubinSpec;
use kernels_sm120::{
    backend::Sm120,
    row::block_quantize::{self, Entry},
};
use model_deepseek41_contract::{admission::SUPPORTED_RANKS, geometry::Geometry};
pub const FAMILY: &str = "ceil_prefill";
pub const WARPS: [u32; 3] = [4, 8, 16];
pub const SERVED_WARPS: u32 = 8;
pub fn cubin_name(g: &Geometry) -> String {
    format!("ceil_prefill_{}_sm120", g.kernel_tag)
}
pub fn baseline_quantizers(g: &Geometry, ranks: u32) -> Result<Vec<Entry>, String> {
    super::dense::quantizers(g, ranks)
}
pub fn quantizers(g: &Geometry, ranks: u32, warps: u32) -> Result<Vec<Entry>, String> {
    baseline_quantizers(g, ranks)?
        .into_iter()
        .map(|mut e| {
            e.symbol = format!(
                "{}_prefill_w{warps}_v1",
                e.symbol.strip_suffix("_v1").unwrap()
            );
            e.prefill(warps).map_err(|e| e.to_string())
        })
        .collect()
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            let mut entries = Vec::<Entry>::new();
            for &ranks in SUPPORTED_RANKS {
                for warps in WARPS {
                    for entry in quantizers(g, ranks, warps)? {
                        if !entries.iter().any(|e| e.symbol == entry.symbol) {
                            entries.push(entry);
                        }
                    }
                }
            }
            Ok((
                FAMILY,
                block_quantize::cubin_spec(&cubin_name(g), &entries).map_err(|e| e.to_string())?,
            ))
        })
        .collect()
}
pub fn register(backend: Sm120, g: &Geometry, ranks: u32) -> Result<Sm120, String> {
    Ok(quantizers(g, ranks, SERVED_WARPS)?
        .into_iter()
        .fold(backend, Sm120::with_block_quantize))
}
