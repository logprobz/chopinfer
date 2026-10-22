//! Cluster-split forms of Flash's one-row decode ceil quantizers
//! (`kernels_sm120::row::block_quantize::Entry::split`): the WO_a input
//! (the attention output, 8192 BF16 columns, served by its cached entry)
//! and the WO_b input (WO_a's output, 2048 F32 columns, served by its
//! parallel entry). A one-CTA row runs each warp's groups back to back on
//! one SM; split over a cluster of 2, 4 or 8 CTAs, each warp runs one or
//! two. Codes, scales and the row status are the served entry's, and each
//! twin keeps its served entry's early dependent release. Every candidate
//! part count is compiled; [`SERVED`] names the ones registered.
use kernel_build::CubinSpec;
use kernel_core::contract::DType;
use kernels_sm120::{backend::Sm120, row::block_quantize};
use model_deepseek41_contract::{
    admission::SUPPORTED_RANKS,
    geometry::{FLASH, Geometry},
};

pub const FAMILY: &str = "ceil_split";

/// The rows a split twin holds: the one-row decode bucket.
pub const ROWS: u32 = 1;

/// The widths and input types the twins quantize.
pub const WIDTHS: [(u32, DType); 2] = [(8192, DType::Bf16), (2048, DType::F32)];

/// The part counts compiled for every width.
pub const PARTS: [u32; 3] = [2, 4, 8];

/// The part count registered for each width; none until measured.
pub const SERVED: [(u32, DType, u32); 0] = [];

/// Set to zero to keep the served one-CTA quantizers at one row.
pub const SWITCH: &str = "CHOPINFER_DEEPSEEK41_CEIL_SPLIT";

pub fn cubin_name(g: &Geometry) -> String {
    format!("ceil_split_{}_sm120", g.kernel_tag)
}

/// Whether `cubin` is a split ceil module, which an instance loads only
/// where its twins serve ([`serves`]).
pub fn is_cubin(cubin: &str) -> bool {
    cubin.starts_with("ceil_split_")
}

/// Whether `g` at `ranks` registers [`SERVED`] twins: Flash TP4 once a
/// width is measured, unless [`SWITCH`] is zero.
pub fn serves(g: &Geometry, ranks: u32) -> bool {
    g.id == FLASH.id
        && ranks == 4
        && SERVED
            .iter()
            .any(|&(width, input, _)| WIDTHS.contains(&(width, input)))
        && std::env::var(SWITCH).as_deref() != Ok("0")
}

/// The served one-row entry of `parallel`, a served parallel quantizer:
/// its cached twin where one serves.
fn served(parallel: block_quantize::Entry) -> block_quantize::Entry {
    super::ceil_cached::cached(&parallel).unwrap_or(parallel)
}

/// The split twin over `parts` CTAs of a served Flash dense quantizer at
/// one of [`WIDTHS`].
pub fn split(served: &block_quantize::Entry, parts: u32) -> Option<block_quantize::Entry> {
    if !WIDTHS.contains(&(served.query.width, served.query.input)) {
        return None;
    }
    let stem = served
        .symbol
        .strip_suffix("_cached_v1")
        .or_else(|| served.symbol.strip_suffix("_parallel_v1"))?;
    if !stem.starts_with(&format!("chop_{}_dense_ceil_", FLASH.kernel_tag)) {
        return None;
    }
    let mut entry = served.clone();
    entry.symbol = format!("{stem}_split{parts}_v1");
    entry.capacity = ROWS;
    entry.split(parts).ok()
}

/// Every split twin of Flash's served quantizers at `ranks`, each part
/// count of [`PARTS`].
pub fn candidates(g: &Geometry, ranks: u32) -> Result<Vec<block_quantize::Entry>, String> {
    if g.id != FLASH.id {
        return Ok(Vec::new());
    }
    let mut entries = Vec::new();
    for parallel in super::ceil_parallel::quantizers(g, ranks)? {
        let served = served(parallel);
        for parts in PARTS {
            entries.extend(split(&served, parts));
        }
    }
    Ok(entries)
}

/// The served entry and its split twins, by width: what the lab pairs.
pub fn pairs(g: &Geometry, ranks: u32) -> Result<Vec<[block_quantize::Entry; 2]>, String> {
    let mut pairs = Vec::new();
    for parallel in super::ceil_parallel::quantizers(g, ranks)? {
        let served = served(parallel);
        for parts in PARTS {
            if let Some(split) = split(&served, parts) {
                pairs.push([served.clone(), split]);
            }
        }
    }
    Ok(pairs)
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let mut specs = Vec::new();
    for g in geometries {
        let mut entries = Vec::<block_quantize::Entry>::new();
        for &ranks in SUPPORTED_RANKS {
            for entry in candidates(g, ranks)? {
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

/// The [`SERVED`] twins of `g` at `ranks`, which the one-row decode
/// bucket's quantizers select (they hold the fewest rows).
pub fn register(mut backend: Sm120, g: &Geometry, ranks: u32) -> Result<Sm120, String> {
    if !serves(g, ranks) {
        return Ok(backend);
    }
    for entry in candidates(g, ranks)? {
        if SERVED.contains(&(entry.query.width, entry.query.input, entry.split_parts)) {
            backend = backend.with_block_quantize(entry);
        }
    }
    Ok(backend)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flash TP4 compiles a twin per width and part count, each with its
    /// served entry's query, ABI and early release; other geometries and
    /// widths compile none.
    #[test]
    fn twins_cover_the_measured_widths_at_flash_tp4() {
        let pairs = pairs(&FLASH, 4).unwrap();
        assert_eq!(pairs.len(), WIDTHS.len() * PARTS.len());
        let mut symbols = Vec::new();
        for [served, split] in &pairs {
            assert_eq!(served.query, split.query);
            assert_eq!(served.spec().params, split.spec().params);
            assert_eq!(served.early_release, split.early_release);
            assert!(served.early_release);
            assert_eq!(split.capacity, ROWS);
            assert!(PARTS.contains(&split.split_parts));
            assert_eq!(
                split.plan(1).unwrap().grid,
                [split.split_parts, 1, 1],
                "{}",
                split.symbol
            );
            symbols.push(split.symbol.clone());
        }
        assert!(symbols.contains(&"chop_deepseek41_dense_ceil_w8192_split8_v1".to_owned()));
        assert!(symbols.contains(&"chop_deepseek41_dense_ceil_f32_w2048_split4_v1".to_owned()));
        let served: Vec<_> = pairs.iter().map(|[s, _]| s.symbol.as_str()).collect();
        assert!(served.contains(&"chop_deepseek41_dense_ceil_w8192_cached_v1"));
        assert!(served.contains(&"chop_deepseek41_dense_ceil_f32_w2048_parallel_v1"));
        let specs = cubins(model_deepseek41_contract::geometry::GEOMETRIES).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].1.entries.len(), WIDTHS.len() * PARTS.len());
    }
}
