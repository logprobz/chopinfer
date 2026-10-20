//! Measured Flash TP4 head buckets with the original ordered global fold.
use kernel_build::CubinSpec;
use kernels_sm120::f32_projection::{
    Entry,
    last::{self, Candidate},
};
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const FAMILY: &str = "f32_decode_last";
pub const CUBIN: &str = "f32_decode_last_deepseek41_sm120";
pub fn enabled(g: &Geometry, ranks: u32) -> bool {
    g.id == FLASH.id && ranks == 4
}

// Preserve the complete GPU-qualified module, including unused controls.
// Only the split-fast scalar-load entries are advertised by the serving op.
pub fn entries() -> Result<Vec<Candidate>, String> {
    let entry = super::f32_head::decode_entry(&FLASH, 4)?;
    let mut entries = Vec::new();
    for tile_rows in [8, 32] {
        for split_fast in [false, true] {
            for prefetch in [1, 16] {
                entries.push(Candidate {
                    entry: entry.clone(),
                    tile_rows,
                    split_fast,
                    prefetch,
                });
            }
        }
    }
    Ok(entries)
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(Vec::new());
    }
    Ok(vec![(
        FAMILY,
        last::cubin_spec(CUBIN, &entries()?).map_err(|e| e.to_string())?,
    )])
}
pub fn head(g: &Geometry, ranks: u32, entry: Entry) -> Result<Entry, String> {
    #[cfg(test)]
    if REFERENCE.with(std::cell::Cell::get) {
        return Ok(entry);
    }
    if enabled(g, ranks) && entry.query() == super::f32_head::query(g, ranks) {
        // SM120, 250 W, 2026-10-20: all real-input/device/cache pairs win
        // by more than 3%. Do not infer a crossover for unmeasured rows.
        entry
            .with_decode_last(vec![1, 8, 32])
            .map_err(|e| e.to_string())
    } else {
        Ok(entry)
    }
}
#[cfg(test)]
std::thread_local! {static REFERENCE:std::cell::Cell<bool>=const {std::cell::Cell::new(false)};}
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
