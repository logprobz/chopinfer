//! Captured one-row query and KV folds, retaining their independent chains.
use kernel_build::CubinSpec;
use kernels_sm120::gemm::mxfp8_fold_dual::{self, Entry};
use model_deepseek41_contract::geometry::{FLASH, Geometry};
pub const FAMILY: &str = "mxfp8_fold_dual";
pub const CUBIN: &str = "fold_dual_deepseek41_sm120";
pub fn entry() -> Entry {
    let catalog = super::mxfp8_fold_stream::catalog(&FLASH).unwrap();
    Entry {
        projections: [1280, 512].map(|n| {
            catalog
                .iter()
                .find(|e| {
                    (e.ticket.query.n, e.ticket.query.k, e.ticket.query.groups) == (n, 5120, 1)
                        && !e.ticket.paired
                })
                .unwrap()
                .clone()
        }),
        symbol: "chop_mxfp8_dual_n1280_n512_k5120_g1_f32_v1".into(),
    }
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(Vec::new());
    }
    Ok(vec![(
        FAMILY,
        mxfp8_fold_dual::cubin_spec(CUBIN, &[entry()]).map_err(|e| e.to_string())?,
    )])
}
pub fn enabled(g: &Geometry, ranks: u32) -> bool {
    #[cfg(test)]
    if REFERENCE.with(|f| f.get()) {
        return false;
    }
    g.id == FLASH.id && ranks == 4
}
#[cfg(test)]
std::thread_local! {static REFERENCE:std::cell::Cell<bool>=const{std::cell::Cell::new(false)};}
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
