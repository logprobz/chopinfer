//! Both-device compact tile wins from the paired 250 W bitwise matrix.
//! Exact measured rows only; other geometries and counts keep their prior route.
use kernel_build::CubinSpec;
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
use kernels_sm120::gemm::mxfp8_prefill_compact::Entry;
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const FAMILY: &str = "mxfp8_prefill_compact";
pub const CUBIN: &str = "compact_dense_prefill_sm120";
pub const ROWS: [u32; 5] = [128, 512, 2048, 4096, 8192];
// d2-0381: event and exclusive median ratios must both be below 0.97 on
// both devices. Select the winning stage count with the best worst-device
// speedup. Zero retains the baseline. See mxfp8_compact.md for the timings.
const CHOICES: &[(u32, u32, u32, [u32; 5])] = &[
    (512, 5120, 1, [2, 2, 2, 2, 2]),
    (576, 5120, 1, [2, 2, 2, 2, 2]),
    (1024, 4096, 2, [3, 2, 2, 2, 2]),
    (1024, 4096, 4, [3, 2, 2, 2, 2]),
    (1024, 4096, 8, [2, 2, 2, 2, 2]),
    (1152, 5120, 1, [2, 2, 2, 2, 2]),
    (1280, 5120, 1, [2, 2, 2, 2, 2]),
    (2304, 5120, 1, [3, 2, 2, 2, 2]),
    (4096, 1280, 1, [3, 0, 0, 0, 0]),
    (5120, 576, 1, [3, 0, 0, 0, 0]),
    (5120, 1152, 1, [3, 0, 0, 0, 0]),
    (5120, 2048, 1, [3, 0, 0, 0, 0]),
    (5120, 2304, 1, [3, 0, 0, 0, 0]),
    (5120, 4096, 1, [3, 2, 2, 2, 2]),
    (5120, 8192, 1, [3, 2, 2, 2, 2]),
    (5120, 15360, 1, [3, 2, 2, 2, 2]),
    (8192, 1280, 1, [2, 2, 0, 0, 0]),
    (25600, 6144, 1, [2, 2, 2, 0, 0]),
    (32768, 1280, 1, [2, 0, 0, 0, 0]),
];

pub fn stages(g: &Geometry, q: Mxfp8FoldGemmQuery, rows: u32) -> Option<u32> {
    if g.id != FLASH.id {
        return None;
    }
    let index = ROWS.iter().position(|r| *r == rows)?;
    CHOICES
        .iter()
        .find(|(n, k, groups, _)| (*n, *k, *groups) == (q.n, q.k, q.groups))
        .map(|(_, _, _, choices)| choices[index])
        .filter(|s| *s != 0)
}

/// Retain the complete qualified translation unit, including controls and
/// unused stage variants, so its cubin bytes match the paired evidence.
pub fn catalog() -> Result<Vec<Entry>, String> {
    super::prefill_catalog(&FLASH)?
        .into_iter()
        .flat_map(|base| {
            [2, 3].map(|stages| Entry::new(base.clone(), stages).map_err(|e| e.to_string()))
        })
        .collect()
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(Vec::new());
    }
    Ok(vec![(
        FAMILY,
        kernels_sm120::gemm::mxfp8_prefill_compact::cubin_spec(CUBIN, &catalog()?)
            .map_err(|e| e.to_string())?,
    )])
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::{Backend, Counts, Need, View};
    use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers;
    use kernels_sm120::backend::Sm120;

    #[test]
    fn route_only_measured_flash_rows_across_all_topologies() {
        let mut changed = std::collections::BTreeSet::new();
        for ranks in [1, 2, 4] {
            let backend = super::super::register(Sm120::new(), &FLASH, ranks).unwrap();
            for entry in super::super::prefill_entries(&FLASH, ranks).unwrap() {
                let q = entry.query;
                let op = backend
                    .mxfp8_prefill_gemm(&q, Need::rows(q.m_capacity))
                    .unwrap();
                let old = Sm120::new()
                    .with_mxfp8_prefill_split_policy(
                        entry.clone(),
                        super::super::prefill_tail_policy(q),
                    )
                    .mxfp8_prefill_gemm(&q, Need::rows(q.m_capacity))
                    .unwrap();
                assert_eq!(op.info().bytes, old.info().bytes);
                assert_eq!(op.info().workspace, old.info().workspace);
                assert_eq!(op.info().weight_layouts, old.info().weight_layouts);
                assert_eq!(op.info().guarantees, old.info().guarantees);
                for rows in [
                    1, 16, 64, 127, 128, 129, 257, 511, 512, 513, 2048, 2049, 4096, 4097, 8191,
                    8192,
                ] {
                    let b = entry.bytes(rows);
                    let v = |handle, bytes| View { handle, bytes };
                    let views = Mxfp8FoldGemmBuffers {
                        activations: v(1, b.activations),
                        activation_scales: v(2, b.activation_scales),
                        weights: v(3, b.weights),
                        weight_scales: v(4, b.weight_scales),
                        output: v(5, b.output),
                        live_rows: v(6, 4),
                        status: v(7, b.status),
                        status_row_offset: 0,
                    };
                    let calls = op
                        .launches(&views, View::EMPTY, Counts::rows(rows))
                        .unwrap();
                    if let Some(stages) = stages(&FLASH, q, rows) {
                        changed.insert((q.n, q.k, q.groups, rows));
                        let candidate = Entry::new(entry.clone(), stages)
                            .unwrap()
                            .prepare(rows)
                            .unwrap();
                        assert_eq!(
                            calls,
                            candidate
                                .launches(&views, View::EMPTY, Counts::rows(rows))
                                .unwrap()
                        );
                        assert!(calls.iter().all(|call| {
                            op.info().symbols.iter().any(|(s, size)| {
                                s == &call.symbol && *size >= call.plan.dynamic_shared_bytes
                            })
                        }));
                    } else {
                        assert_eq!(
                            calls,
                            old.launches(&views, View::EMPTY, Counts::rows(rows))
                                .unwrap()
                        );
                    }
                    let custom = Geometry {
                        id: "custom-shape",
                        ..FLASH
                    };
                    assert_eq!(stages(&custom, q, rows), None);
                }
            }
        }
        assert_eq!(changed.len(), 66);
    }
}
