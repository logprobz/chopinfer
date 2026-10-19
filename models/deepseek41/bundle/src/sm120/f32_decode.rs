//! Cached exact-F32 projections. The original families remain locked as
//! baselines; this family adds create-time flags and small-row dispatch.
use kernel_build::CubinSpec;
use kernels_sm120::f32_projection::{Entry, cubin_spec};
use model_deepseek41_contract::admission::SUPPORTED_RANKS;
use model_deepseek41_contract::geometry::Geometry;

pub const FAMILY: &str = "f32_decode";

pub fn cubin_name(g: &Geometry) -> String {
    format!("f32_decode_{}_sm120", g.kernel_tag)
}

pub fn entries(g: &Geometry, ranks: u32) -> Result<Vec<Entry>, String> {
    let mut entries = super::f32_dense::entries(g)?;
    entries.extend(super::dspark_router::entries(g)?);
    let mut entries: Vec<_> = entries.into_iter().map(Entry::with_decode).collect();
    entries.push(super::f32_head::decode_entry(g, ranks)?);
    Ok(entries)
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            let mut all: Vec<Entry> = Vec::new();
            for &ranks in SUPPORTED_RANKS {
                if let Ok(entries) = entries(g, ranks) {
                    for entry in entries {
                        if !all.iter().any(|e| e.symbol() == entry.symbol()) {
                            all.push(entry);
                        }
                    }
                }
            }
            Ok((
                FAMILY,
                cubin_spec(cubin_name(g), &all).map_err(|e| e.to_string())?,
            ))
        })
        .collect()
}
