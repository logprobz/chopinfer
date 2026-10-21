//! Exact Flash TP4 prefill rows qualified against the served fold entries.
use kernel_build::CubinSpec;
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
use kernels_sm120::{
    backend::Sm120,
    gemm::mxfp8_prefill_matrix::{self, Entry},
};
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const FAMILY: &str = "mxfp8_prefill_matrix";
pub const CUBIN: &str = "fold_core_prefill_deepseek41_sm120";
pub const ROWS: [u32; 7] = [192, 448, 960, 1472, 2496, 4096, 8192];
// SM120, 250 W, 2026-10-20. All selected rows beat 0.97 on both devices
// with cold and warm caches. Four Q/O families additionally pass captured
// verify operands repeated to 64, 2496 and 8192 rows. Other measured rows
// use distinct synthetic operands. No 64-row timing is used: serving
// keeps the general fold below 128 rows. No other family is selected.
// Each cell gives (tile rows, pipeline stages), ordered as ROWS.
type RowChoices = (u32, u32, u32, [(u32, u32); 7]);
#[rustfmt::skip]
const CHOICES: &[RowChoices] = &[
    (1280, 5120, 1, [(32, 3), (32, 3), (64, 2), (128, 2), (64, 2), (128, 2), (128, 2)]),
    (8192, 1280, 1, [(128, 2), (128, 2), (128, 2), (128, 2), (128, 2), (128, 2), (128, 2)]),
    (1024, 4096, 2, [(32, 3), (64, 2), (128, 2), (64, 2), (128, 2), (128, 2), (128, 2)]),
    (5120, 2048, 1, [(64, 3), (128, 2), (128, 2), (128, 2), (128, 2), (128, 2), (128, 2)]),
];

pub fn choice(q: Mxfp8FoldGemmQuery, rows: u32) -> Option<(u32, u32)> {
    let index = ROWS.iter().position(|&r| r == rows)?;
    CHOICES
        .iter()
        .find(|&&(n, k, g, _)| (n, k, g) == (q.n, q.k, q.groups))
        .map(|row| row.3[index])
}
pub fn enabled(g: &Geometry, ranks: u32) -> bool {
    g.id == FLASH.id && ranks == 4
}
fn selected(g: &Geometry, ranks: u32) -> bool {
    #[cfg(test)]
    if REFERENCE.with(|flag| flag.get()) {
        return false;
    }
    enabled(g, ranks)
}

/// Preserve every entry and control from the qualified translation unit.
pub fn catalog() -> Result<Vec<Entry>, String> {
    super::mxfp8_fold::prefill_entries(&FLASH, 4)?
        .into_iter()
        .filter(|base| base.query.k != 15360)
        .flat_map(|base| {
            [(32, 3), (64, 2), (64, 3), (128, 2)]
                .map(move |(m, s)| Entry::new(base.clone(), s, m).map_err(|e| e.to_string()))
        })
        .collect()
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(Vec::new());
    }
    Ok(vec![(
        FAMILY,
        mxfp8_prefill_matrix::cubin_spec(CUBIN, &catalog()?).map_err(|e| e.to_string())?,
    )])
}
pub fn register(mut backend: Sm120, g: &Geometry, ranks: u32) -> Result<Sm120, String> {
    if !selected(g, ranks) {
        return Ok(backend);
    }
    for entry in catalog()? {
        for rows in ROWS {
            if choice(entry.base.query, rows) == Some((entry.tile_rows, entry.stages)) {
                backend = backend.with_mxfp8_prefill_matrix(entry.clone(), rows);
            }
        }
    }
    Ok(backend)
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
                    .chain(ROWS.into_iter().flat_map(|r| [r - 1, r, r + 1]))
                {
                    let a = after.launches(&buffers, View::EMPTY, Counts::rows(rows));
                    if ranks == 4
                        && let Some((tile, stages)) = choice(q, rows)
                    {
                        let expected = Entry::new(base.clone(), stages, tile)
                            .unwrap()
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
        assert_eq!(changed.len(), 28);
    }
}
