//! One captured Flash TP4 prefill bucket over the qualified TMA module.
use kernel_build::CubinSpec;
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
use kernels_sm120::backend::Sm120;
use kernels_sm120::gemm::mxfp8_prefill_tma_matrix::{self, Entry};
use model_deepseek41_contract::geometry::{FLASH, Geometry};
pub const FAMILY: &str = "mxfp8_prefill_tma_matrix";
pub const CUBIN: &str = "fold_tma_matrix_sm120";
pub fn entries() -> Vec<Entry> {
    super::mxfp8_fold::prefill_entries(&FLASH, 4)
        .unwrap()
        .into_iter()
        .filter(|e| {
            [
                (1280, 5120),
                (8192, 1280),
                (1024, 4096),
                (5120, 2048),
                (25600, 6144),
            ]
            .contains(&(e.query.n, e.query.k))
        })
        .flat_map(|base| {
            [2, 3].map(move |stages| Entry {
                base: base.clone(),
                stages,
            })
        })
        .collect()
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(Vec::new());
    }
    Ok(vec![(
        FAMILY,
        mxfp8_prefill_tma_matrix::cubin_spec(CUBIN, &entries()).map_err(|e| e.to_string())?,
    )])
}
// SM120, 250 W: at least 3.8% faster on both devices, cold and warm,
// including captured verify operands repeated to the complete chunk.
pub fn choice(g: &Geometry, ranks: u32, q: Mxfp8FoldGemmQuery, rows: u32) -> bool {
    g.id == FLASH.id && ranks == 4 && (q.n, q.k, q.groups, rows) == (8192, 1280, 1, 8192)
}
pub fn register(mut backend: Sm120, g: &Geometry, ranks: u32) -> Result<Sm120, String> {
    #[cfg(test)]
    if REFERENCE.with(|f| f.get()) {
        return Ok(backend);
    }
    for e in entries() {
        if e.stages == 2 && choice(g, ranks, e.base.query, 8192) {
            backend = backend.with_mxfp8_prefill_tma_matrix(e, 8192);
        }
    }
    Ok(backend)
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

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::{
        backend::{Backend, Counts, Need, View},
        mxfp8_gemm::Mxfp8FoldGemmBuffers,
    };

    #[test]
    fn select_only_measured_rows_and_keep_fallback_contracts() {
        let mut changed = std::collections::BTreeSet::new();
        for ranks in [1, 2, 4] {
            let old = super::super::mxfp8_fold::register(Sm120::new(), &FLASH, ranks).unwrap();
            let new = register(
                super::super::mxfp8_fold::register(Sm120::new(), &FLASH, ranks).unwrap(),
                &FLASH,
                ranks,
            )
            .unwrap();
            for base in super::super::mxfp8_fold::prefill_entries(&FLASH, ranks).unwrap() {
                let q = base.query;
                let before = old.mxfp8_prefill_gemm(&q, Need::rows(8192)).unwrap();
                let after = new.mxfp8_prefill_gemm(&q, Need::rows(8192)).unwrap();
                assert_eq!(before.info().bytes, after.info().bytes);
                assert_eq!(before.info().workspace, after.info().workspace);
                assert_eq!(before.info().weight_layouts, after.info().weight_layouts);
                assert_eq!(before.info().guarantees, after.info().guarantees);
                let b = base.bytes(8192);
                let v = |handle, bytes| View { handle, bytes };
                let buffers = Mxfp8FoldGemmBuffers {
                    activations: v(1, b.activations),
                    activation_scales: v(2, b.activation_scales),
                    weights: v(3, b.weights),
                    weight_scales: v(4, b.weight_scales),
                    output: v(5, b.output),
                    live_rows: v(6, 4),
                    status: v(7, b.status + 12),
                    status_row_offset: 3,
                };
                for rows in [0, 1, 8, 32, 64, 127, 128, 8191, 8193]
                    .into_iter()
                    .chain([8191, 8192, 8193])
                {
                    let a = after.launches(&buffers, View::EMPTY, Counts::rows(rows));
                    if choice(&FLASH, ranks, q, rows) {
                        let expected = Entry {
                            base: base.clone(),
                            stages: 2,
                        }
                        .prepare(rows)
                        .unwrap();
                        let calls = a.unwrap();
                        assert_eq!(
                            calls,
                            expected
                                .launches(&buffers, View::EMPTY, Counts::rows(rows))
                                .unwrap()
                        );
                        assert!(calls.iter().all(|call| after.info().symbols.iter().any(
                            |(s, b)| *s == call.symbol && *b >= call.plan.dynamic_shared_bytes
                        )));
                        changed.insert((q.n, q.k, q.groups, rows));
                    } else {
                        assert_eq!(
                            a,
                            before.launches(&buffers, View::EMPTY, Counts::rows(rows))
                        );
                    }
                }
            }
        }
        assert_eq!(changed.len(), 1);
    }
}
