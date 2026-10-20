//! Flash TP4 row buckets qualified with captured AR and verify operands.
use kernel_build::CubinSpec;
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
use kernels_sm120::{
    backend::Sm120,
    f32_projection,
    gemm::mxfp8_fold_reuse::{self, Entry},
};
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const FOLD_FAMILY: &str = "mxfp8_fold_reuse";
pub const HEAD_FAMILY: &str = "f32_decode_reuse";
pub const FOLD_CUBIN: &str = "mxfp8_fold_reuse_deepseek41_sm120";
pub const HEAD_CUBIN: &str = "f32_decode_reuse_deepseek41_sm120";
const SHAPES: [(u32, u32, u32); 4] = [
    (8192, 1280, 1),
    (5120, 2048, 1),
    (1024, 4096, 2),
    (1280, 5120, 1),
];

// SM120, 250 W, 2026-10-20. Every selected tuple is bitwise in eager and
// graph execution, and candidate/current is below 0.97 in all twelve
// paired samples: three real input sets, two devices, cold and warm L2.
// No crossover is inferred for omitted rows. C1, C8 and verify-6 stay put.
pub fn choice(q: Mxfp8FoldGemmQuery, rows: u32) -> Option<(u32, u32)> {
    match (q.n, q.k, q.groups) {
        (1280, 5120, 1) if (17..=21).contains(&rows) || (23..=24).contains(&rows) => Some((32, 4)),
        (8192, 1280, 1) if (9..=12).contains(&rows) => Some((64, 4)),
        (1024, 4096, 2) if (17..=32).contains(&rows) => Some((32, 4)),
        (5120, 2048, 1) if (17..=23).contains(&rows) || (25..=28).contains(&rows) => Some((32, 8)),
        (5120, 2048, 1) if (29..=32).contains(&rows) => Some((32, 2)),
        _ => None,
    }
}

pub fn enabled(g: &Geometry, ranks: u32) -> bool {
    g.id == FLASH.id && ranks == 4
}

pub fn entries() -> Result<Vec<Entry>, String> {
    let mut entries = Vec::new();
    for stream in super::mxfp8_fold_stream::catalog(&FLASH)? {
        let q = stream.ticket.query;
        if !stream.ticket.paired && SHAPES.contains(&(q.n, q.k, q.groups)) {
            for tile_n in [32, 64] {
                entries.push(Entry {
                    ticket: stream.ticket.clone(),
                    tile_n,
                });
            }
        }
    }
    if entries.len() != 8 {
        return Err("missing qualified projection reuse shapes".into());
    }
    Ok(entries)
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(Vec::new());
    }
    Ok(vec![
        (
            FOLD_FAMILY,
            mxfp8_fold_reuse::cubin_spec(FOLD_CUBIN, &entries()?).map_err(|e| e.to_string())?,
        ),
        (
            HEAD_FAMILY,
            f32_projection::reuse::cubin_spec(
                HEAD_CUBIN,
                &[super::f32_head::decode_entry(&FLASH, 4)?],
            )
            .map_err(|e| e.to_string())?,
        ),
    ])
}

pub fn register(mut backend: Sm120, g: &Geometry, ranks: u32) -> Result<Sm120, String> {
    if !selected(g, ranks) {
        return Ok(backend);
    }
    for entry in entries()? {
        for rows in 9..=32 {
            if let Some((tile, folds)) = choice(entry.ticket.query, rows)
                && tile == entry.tile_n
            {
                let mut entry = entry.clone();
                entry.ticket.query = super::mxfp8_fold::routed_query(entry.ticket.query, rows);
                backend = backend.with_mxfp8_fold_reuse(entry, folds);
            }
        }
    }
    Ok(backend)
}

pub fn head(
    g: &Geometry,
    ranks: u32,
    entry: f32_projection::Entry,
) -> Result<f32_projection::Entry, String> {
    if selected(g, ranks) && entry.query() == super::f32_head::query(g, ranks) {
        entry
            .with_decode_reuse((9..=32).collect())
            .map_err(|e| e.to_string())
    } else {
        Ok(entry)
    }
}

#[cfg(test)]
#[path = "projection_reuse_tests.rs"]
mod tests;

fn selected(g: &Geometry, ranks: u32) -> bool {
    #[cfg(test)]
    if REFERENCE.with(std::cell::Cell::get) {
        return false;
    }
    enabled(g, ranks)
}
#[cfg(test)]
std::thread_local! { static REFERENCE: std::cell::Cell<bool> = const {std::cell::Cell::new(false)}; }
#[cfg(test)]
pub(crate) fn with_reference<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            REFERENCE.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(REFERENCE.with(|flag| flag.replace(true)));
    run()
}
