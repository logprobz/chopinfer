//! Decode norm epilogues in their own lock family.
use kernel_build::CubinSpec;
use kernels_sm120::{
    backend::Sm120,
    row::rmsnorm_quantize::{self, Entry},
};
use model_deepseek41_contract::{geometry::Geometry, rows, serving::DECODE_BATCH};

pub const FAMILY: &str = "norm_quantize";
pub fn cubin_name(g: &Geometry) -> String {
    format!("norm_quantize_{}_sm120", g.kernel_tag)
}
pub fn entries(g: &Geometry) -> Result<Vec<Entry>, String> {
    rows::norm_quantizers(g)
        .into_iter()
        .zip(["input", "query"])
        .map(|(query, kind)| {
            Entry::new(
                query,
                DECODE_BATCH,
                format!("chop_norm_quantize_{}_{kind}_v1", g.kernel_tag),
            )
            .map_err(|e| e.to_string())
        })
        .collect()
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            Ok((
                FAMILY,
                rmsnorm_quantize::cubin_spec(&cubin_name(g), &entries(g)?)
                    .map_err(|e| e.to_string())?,
            ))
        })
        .collect()
}
pub fn register(mut backend: Sm120, g: &Geometry) -> Result<Sm120, String> {
    for entry in entries(g)? {
        backend = backend.with_rmsnorm_quantize(entry);
    }
    Ok(backend)
}
