//! Measured one-pass cached forms of the parallel decode ceil quantizers.
//! The parallel cubin keeps every entry so its identity stays fixed; the
//! cached entries replace their parallel twins only at the measured widths.
use kernel_build::CubinSpec;
use kernels_sm120::row::block_quantize;
use model_deepseek41_contract::{admission::SUPPORTED_RANKS, geometry::Geometry};

pub const FAMILY: &str = "ceil_cached";

/// Widths where the paired lab timing gained on both devices at every
/// decode bucket (1, 8 and 32 rows). Narrower rows tied and keep the
/// parallel entry; the cached form admits at most 8192 columns.
pub const WIDTHS: std::ops::RangeInclusive<u32> = 4096..=8192;

pub fn cubin_name(g: &Geometry) -> String {
    format!("ceil_cached_{}_sm120", g.kernel_tag)
}

/// The cached twin of a served parallel entry at a measured width.
pub fn cached(parallel: &block_quantize::Entry) -> Option<block_quantize::Entry> {
    if !WIDTHS.contains(&parallel.query.width) {
        return None;
    }
    let mut entry = parallel.clone();
    entry.symbol = format!(
        "{}_cached_v1",
        parallel.symbol.strip_suffix("_parallel_v1")?
    );
    entry.cached().ok()
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let mut specs = Vec::new();
    for g in geometries {
        let mut entries = Vec::<block_quantize::Entry>::new();
        for &ranks in SUPPORTED_RANKS {
            for entry in super::ceil_parallel::quantizers(g, ranks)?
                .iter()
                .filter_map(cached)
            {
                if !entries.iter().any(|e| e.symbol == entry.symbol) {
                    entries.push(entry);
                }
            }
        }
        if !entries.is_empty() {
            specs.push((
                FAMILY,
                block_quantize::cubin_spec(&cubin_name(g), &entries).map_err(|e| e.to_string())?,
            ));
        }
    }
    Ok(specs)
}
