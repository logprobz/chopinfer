//! Large-bucket dense quantizers. Decode and verify retain the separately
//! registered parallel entries up to their capacity. MoE and latent
//! quantizers remain with their owning registrations.
use kernel_build::CubinSpec;
use kernels_sm120::{
    backend::Sm120,
    row::block_quantize::{self, Entry},
};
use model_deepseek41_contract::{admission::SUPPORTED_RANKS, geometry::Geometry};
pub const FAMILY: &str = "ceil_prefill";
pub const WARPS: [u32; 3] = [4, 8, 16];
pub fn cubin_name(g: &Geometry) -> String {
    format!("ceil_prefill_{}_sm120", g.kernel_tag)
}
pub fn baseline_quantizers(g: &Geometry, ranks: u32) -> Result<Vec<Entry>, String> {
    super::dense::quantizers(g, ranks)
}
fn prefill(mut entry: Entry, warps: u32) -> Result<Entry, String> {
    entry.symbol = format!(
        "{}_prefill_w{warps}_v1",
        entry.symbol.strip_suffix("_v1").unwrap()
    );
    entry.prefill(warps).map_err(|e| e.to_string())
}

pub fn quantizers(g: &Geometry, ranks: u32, warps: u32) -> Result<Vec<Entry>, String> {
    baseline_quantizers(g, ranks)?
        .into_iter()
        .map(|entry| prefill(entry, warps))
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
    // Pair measurements at 1024, 2048 and 8192 rows favor sixteen warps
    // for wide rows. Eight retains the small-width wins across row counts.
    baseline_quantizers(g, ranks)?
        .into_iter()
        .try_fold(backend, |backend, entry| {
            let warps = if entry.query.width >= 8192 { 16 } else { 8 };
            Ok(backend.with_block_quantize(prefill(entry, warps)?))
        })
}
