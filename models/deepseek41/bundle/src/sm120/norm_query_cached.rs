//! Measured cached form of the F32 query norm and ceil quantizer. The served
//! query entry keeps its cubin; the cached entries replace it only at the
//! measured row buckets.
use kernel_build::CubinSpec;
use kernels_sm120::backend::Sm120;
use kernels_sm120::row::rmsnorm_quantize_cached::Entry;
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const FAMILY: &str = "norm_query_cached";

/// Rows and warps per CTA where the paired lab timing gained on both devices:
/// the warp count with the larger minimum chained saving at each row count.
/// Other rows keep the served entry.
pub const BUCKETS: [(u32, u32); 4] = [(1, 16), (6, 16), (8, 32), (32, 32)];

pub fn cubin_name(g: &Geometry) -> String {
    format!("norm_query_cached_{}_sm120", g.kernel_tag)
}
pub fn entry(g: &Geometry) -> Entry {
    Entry {
        reference: super::norm_quantize::entries(g)
            .expect("query norm entry")
            .remove(1),
        prefix: format!("chop_norm_quantize_{}_query", g.kernel_tag),
    }
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            Ok((
                FAMILY,
                entry(g)
                    .cubin_spec(&cubin_name(g))
                    .map_err(|e| e.to_string())?,
            ))
        })
        .collect()
}

/// Select only the measured Flash decode and verify buckets.
pub fn register(backend: Sm120, g: &Geometry) -> Sm120 {
    if g.id == FLASH.id {
        backend.with_rmsnorm_quantize_cached(entry(g), BUCKETS.to_vec())
    } else {
        backend
    }
}
