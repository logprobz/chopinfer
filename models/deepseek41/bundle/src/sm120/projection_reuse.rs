//! Flash TP4 row buckets qualified with captured AR and verify operands.
use kernel_build::CubinSpec;
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
use kernels_sm120::{backend::Sm120, f32_projection};
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const HEAD_FAMILY: &str = "f32_decode_reuse";
pub const HEAD_CUBIN: &str = "f32_decode_reuse_deepseek41_sm120";
// SM120, 250 W, 2026-10-20. Every selected tuple is bitwise in eager and
// graph execution, and candidate/current is below 0.97 in all twelve
// paired samples: three real input sets, two devices, cold and warm L2.
// No crossover is inferred for omitted rows. C1, C8 and verify-6 stay put.
pub fn choice(q: Mxfp8FoldGemmQuery, rows: u32) -> Option<(u32, u32)> {
    let row = super::mxfp8_fold::tuning_row(
        &FLASH,
        model_deepseek41_contract::tuning::Family::Reuse,
        "reuse.fold.decode",
        q,
        false,
        rows,
    )?;
    let (knobs, folds) =
        kernels_sm120::gemm::dense_decode::Knobs::from_row(row).expect("valid reuse row");
    Some((knobs.tile_n, folds.expect("reuse row has runtime folds")))
}

pub fn enabled(g: &Geometry, ranks: u32) -> bool {
    g.id == FLASH.id && ranks == 4
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(Vec::new());
    }
    Ok(vec![(
        HEAD_FAMILY,
        f32_projection::reuse::cubin_spec(HEAD_CUBIN, &[super::f32_head::decode_entry(&FLASH, 4)?])
            .map_err(|e| e.to_string())?,
    )])
}

pub fn dense_entries() -> Result<Vec<(kernels_sm120::gemm::dense_decode::Entry, u32)>, String> {
    super::mxfp8_fold::dense_reuse_entries(&FLASH, 4)
}

pub fn register(mut backend: Sm120, g: &Geometry, ranks: u32) -> Result<Sm120, String> {
    if selected(g, ranks) {
        for (entry, folds) in dense_entries()? {
            backend = backend.with_dense_decode(entry, Some(folds));
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
        let q = entry.query();
        let rows = (1..=q.m_capacity)
            .filter(|&rows| {
                model_deepseek41_contract::tuning::select(
                    g,
                    model_deepseek41_contract::tuning::Family::Reuse,
                    "reuse.head.decode",
                    "bf16",
                    &[("n", q.n), ("k", q.k), ("ranks", ranks)],
                    q.m_capacity,
                    rows,
                )
                .is_some()
            })
            .collect();
        entry.with_decode_reuse(rows).map_err(|e| e.to_string())
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
