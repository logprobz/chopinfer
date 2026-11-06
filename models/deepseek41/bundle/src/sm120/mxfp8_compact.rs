//! Both-device compact tile wins from the paired 250 W bitwise matrix.
//! Exact measured rows only; other geometries and counts keep their prior route.
use kernel_build::CubinSpec;
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
use kernels_sm120::backend::Sm120;
use kernels_sm120::gemm::mxfp8_prefill_compact::Entry;
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const FAMILY: &str = "mxfp8_prefill_compact";
pub const CUBIN: &str = "compact_dense_prefill_deepseek41_sm120";
pub const ROWS: [u32; 5] = [128, 512, 2048, 4096, 8192];
pub fn stages(g: &Geometry, q: Mxfp8FoldGemmQuery, rows: u32) -> Option<u32> {
    use model_deepseek41_contract::tuning::{Family, select};
    let row = select(
        g,
        Family::DensePrefill,
        "dense_prefill.compact.prefill",
        "mxfp8",
        &[("n", q.n), ("k", q.k), ("groups", q.groups)],
        q.m_capacity,
        rows,
    )?;
    assert_eq!(row.schedule, "prefill_tiles");
    assert_eq!(row.knobs["matrix_fetch"].flag(), Some(false));
    assert_eq!(row.knobs["tile_rows"].count(), Some(Entry::tile(rows)));
    row.knobs["stages"].count()
}

/// The compact range site: the row ranges of each TP4 prefill projection
/// that the compact entry serves.
pub const RANGE_SITE: &str = "dense_prefill.compact_range.prefill";

/// The row ranges, each with its compact stages, that the range site
/// admits for `q` at `ranks`, in row order. The table refuses overlapping
/// rows of one projection.
pub fn ranges(
    g: &Geometry,
    ranks: u32,
    q: Mxfp8FoldGemmQuery,
) -> Result<Vec<(std::ops::RangeInclusive<u32>, u32)>, String> {
    use model_deepseek41_contract::tuning::{Family, table};
    let Some(table) = table(g, Family::DensePrefill) else {
        return Ok(Vec::new());
    };
    let key: std::collections::BTreeMap<String, u32> = [
        ("n", q.n),
        ("k", q.k),
        ("groups", q.groups),
        ("ranks", ranks),
    ]
    .map(|(name, value)| (name.to_string(), value))
    .into();
    table
        .rows()
        .iter()
        .filter(|row| {
            row.key.op == RANGE_SITE
                && row.key.weights == "mxfp8"
                && row.key.shape == key
                && row.key.bucket == q.m_capacity
        })
        .map(|row| {
            let [first, last] = row
                .key
                .rows
                .ok_or_else(|| format!("{}: the compact range has no rows", row.key))?;
            if row.schedule != "prefill_tiles"
                || row.knobs.get("matrix_fetch").and_then(|v| v.flag()) != Some(false)
                || row.knobs.get("tile_rows").and_then(|v| v.count()) != Some(Entry::tile(first))
                || Entry::tile(first) != Entry::tile(last)
            {
                return Err(format!("{}: unsupported compact range recipe", row.key));
            }
            let stages = row
                .knobs
                .get("stages")
                .and_then(|v| v.count())
                .ok_or_else(|| format!("{}: the compact range has no stages", row.key))?;
            Ok((first..=last, stages))
        })
        .collect()
}

/// Registers each prefill projection's ranges on the shared
/// compact wrapper, which plans each count as it launches. The measured
/// exact rows keep precedence inside a range.
pub fn register_range(mut backend: Sm120, g: &Geometry, ranks: u32) -> Result<Sm120, String> {
    for base in super::prefill_entries(g, ranks)? {
        for (rows, stages) in ranges(g, ranks, base.query)? {
            let entry = Entry::new(base.clone(), stages).map_err(|e| e.to_string())?;
            backend = backend.with_mxfp8_prefill_compact_range(entry, rows);
        }
    }
    Ok(backend)
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
        super::packed_all(
            kernels_sm120::gemm::mxfp8_prefill_compact::cubin_spec(CUBIN, &catalog()?),
            catalog()?.iter().map(|e| e.packed_specs()),
        )?,
    )])
}

#[cfg(test)]
pub fn tile(rows: u32) -> u32 {
    Entry::tile(rows)
}

#[cfg(test)]
pub fn launch_shape(
    base: &kernels_sm120::gemm::mxfp8_fold::Entry,
    rows: u32,
    stages: u32,
) -> (String, kernel_core::contract::LaunchPlan) {
    let entry = Entry::new(base.clone(), stages).unwrap();
    (
        entry.symbol(Entry::tile(rows)),
        entry.plan(rows, Entry::tile(rows)).unwrap(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::{Backend, Counts, Need, View};
    use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers;

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

    /// With the ranges registered, every count of a range runs
    /// the compact entry the lab compares, and every other count, shape and
    /// topology launches as before, with the same layout and guarantees.
    #[test]
    fn ranges_change_only_their_counts() {
        let mut ranged = std::collections::BTreeMap::new();
        for ranks in [1, 2, 4] {
            let old = super::super::register(Sm120::new(), &FLASH, ranks).unwrap();
            let new = super::super::register(Sm120::new(), &FLASH, ranks).unwrap();
            let new = register_range(new, &FLASH, ranks).unwrap();
            for base in super::super::prefill_entries(&FLASH, ranks).unwrap() {
                let q = base.query;
                let ranges = ranges(&FLASH, ranks, q).unwrap();
                let a = old
                    .mxfp8_prefill_gemm(&q, Need::rows(q.m_capacity))
                    .unwrap();
                let b = new
                    .mxfp8_prefill_gemm(&q, Need::rows(q.m_capacity))
                    .unwrap();
                assert_eq!(a.info().bytes, b.info().bytes);
                assert_eq!(a.info().workspace, b.info().workspace);
                assert_eq!(a.info().weight_layouts, b.info().weight_layouts);
                assert_eq!(a.info().guarantees, b.info().guarantees);
                let size = base.bytes(q.m_capacity);
                let v = |handle, bytes| View { handle, bytes };
                let views = Mxfp8FoldGemmBuffers {
                    activations: v(1, size.activations),
                    activation_scales: v(2, size.activation_scales),
                    weights: v(3, size.weights),
                    weight_scales: v(4, size.weight_scales),
                    output: v(5, size.output),
                    live_rows: v(6, 4),
                    status: v(7, size.status),
                    status_row_offset: 0,
                };
                let counts: Vec<u32> = if ranges.is_empty() {
                    vec![
                        1, 64, 1152, 1153, 1280, 1500, 2048, 2304, 2369, 2944, 3072, 3200, 3201,
                        8192,
                    ]
                } else {
                    (1..=q.m_capacity).collect()
                };
                for count in counts {
                    let rows = Counts::rows(count);
                    let before = a.launches(&views, View::EMPTY, rows).unwrap();
                    let after = b.launches(&views, View::EMPTY, rows).unwrap();
                    match ranges.iter().find(|(band, _)| band.contains(&count)) {
                        Some((band, stages)) => {
                            assert_eq!(*stages, 2);
                            let lab = Entry::new(base.clone(), *stages)
                                .unwrap()
                                .prepare(count)
                                .unwrap()
                                .launches(&views, View::EMPTY, rows)
                                .unwrap();
                            assert_eq!(after, lab, "n{} rows={count}", q.n);
                            for call in &after {
                                assert!(b.info().symbols.iter().any(|(symbol, shared)| {
                                    symbol == &call.symbol
                                        && *shared >= call.plan.dynamic_shared_bytes
                                }));
                                let tag = b.route(rows, call).unwrap();
                                assert_eq!(tag.class, kernel_core::route::RouteClass::Fast);
                                assert!(tag.admitted.iter().any(|r| r.contains(&count)));
                            }
                            ranged
                                .entry((q.n, q.k, q.groups))
                                .or_insert_with(std::collections::BTreeSet::new)
                                .insert((*band.start(), *band.end()));
                        }
                        None => assert_eq!(before, after, "ranks={ranks} n{} rows={count}", q.n),
                    }
                }
            }
        }
        assert_eq!(
            ranged,
            [
                ((512, 5120, 1), [(1153, 3071), (3073, 3199)].into()),
                ((576, 5120, 1), [(1153, 2943), (2945, 3200)].into()),
                ((1024, 4096, 2), [(1153, 3200)].into()),
                (
                    (1280, 5120, 1),
                    [(1153, 1279), (1281, 2368), (2401, 3200)].into()
                ),
            ]
            .into()
        );
    }
}
