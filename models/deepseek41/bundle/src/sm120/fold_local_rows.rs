//! Captured native folds retained for the measured Flash TP4 C32 query bucket.
use kernel_build::CubinSpec;
use kernels_sm120::gemm::{mxfp8_fold, mxfp8_fold_local_rows::Entry};
use model_deepseek41_contract::geometry::{FLASH, Geometry};
pub const FAMILY: &str = "fold_local_rows";
pub const CUBIN: &str = "fold_local_rows_deepseek41_sm120";
pub fn entries() -> Vec<Entry> {
    let base = super::mxfp8_fold::prefill_entries(&FLASH, 4)
        .unwrap()
        .into_iter()
        .find(|e| (e.query.n, e.query.k, e.query.groups) == (8192, 1280, 1))
        .unwrap();
    [16, 32]
        .into_iter()
        .map(|tile_n| {
            let mut q = base.query;
            q.m_capacity = 32;
            Entry {
                base: mxfp8_fold::Entry::new(q, "chop_fold_local_n8192_k1280_g1_f32").unwrap(),
                tile_n,
                stages: 0,
                swizzle: false,
            }
        })
        .collect()
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(Vec::new());
    }
    Ok(vec![(
        FAMILY,
        kernels_sm120::gemm::mxfp8_fold_local_rows::cubin_spec(CUBIN, &entries())
            .map_err(|e| e.to_string())?,
    )])
}
pub fn enabled(g: &Geometry, ranks: u32) -> bool {
    g.id == FLASH.id && ranks == 4
}
pub fn selected(g: &Geometry, ranks: u32) -> bool {
    #[cfg(test)]
    if REFERENCE.with(|f| f.get()) {
        return false;
    }
    enabled(g, ranks)
}
#[cfg(test)]
std::thread_local! { static REFERENCE:std::cell::Cell<bool>=const{std::cell::Cell::new(false)}; }
#[cfg(test)]
pub(crate) fn with_reference<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            REFERENCE.with(|f| f.set(self.0));
        }
    }
    let _restore = Restore(REFERENCE.with(|f| f.replace(true)));
    run()
}
