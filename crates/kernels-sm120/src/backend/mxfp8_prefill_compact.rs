//! Exact-row and row-range compact opt-ins layered over the caller's
//! existing prefill fallback.
use crate::gemm::mxfp8_prefill_tiles::Entry;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers as Buffers, Mxfp8FoldGemmQuery};
use std::collections::BTreeMap;

pub(super) fn prepare<const MATRIX_FETCH: bool>(
    fallback: Prepared<Buffers<View>>,
    entries: &[(Entry<MATRIX_FETCH>, u32)],
    query: &Mxfp8FoldGemmQuery,
    need: Need,
) -> Result<Prepared<Buffers<View>>, Refused> {
    let mut selected = BTreeMap::<u32, &Entry<MATRIX_FETCH>>::new();
    for (entry, rows) in entries.iter().filter(|(e, _)| e.base.query == *query) {
        if *rows == 0 || *rows > query.m_capacity {
            return Err(Refusal(if MATRIX_FETCH {
                "matrix prefill registration exceeds query capacity"
            } else {
                "compact prefill registration exceeds query capacity"
            })
            .into());
        }
        if let Some(old) = selected.insert(*rows, entry)
            && (old.base.symbol != entry.base.symbol
                || old.stages != entry.stages
                || old.tile_rows != entry.tile_rows)
        {
            return Err(Refusal(if MATRIX_FETCH {
                "ambiguous matrix prefill row registration"
            } else {
                "ambiguous compact prefill row registration"
            })
            .into());
        }
    }
    selected.retain(|rows, _| *rows <= need.capacity);
    if selected.is_empty() {
        return Ok(fallback);
    }
    let mut info = fallback.info().clone();
    let mut plans = BTreeMap::new();
    let mut choices = Vec::new();
    for (rows, entry) in selected {
        let op = entry.prepare(rows)?;
        info.symbols.extend(op.info().symbols.clone());
        choices.push(if MATRIX_FETCH {
            format!("({rows}, {}, {})", entry.tile_rows, entry.stages)
        } else {
            format!("({rows}, {})", entry.stages)
        });
        plans.insert(rows, op);
    }
    info.symbols.sort();
    info.symbols.dedup();
    info.implementation.push_str(&format!(
        ";{}-v1:[{}]",
        Entry::<MATRIX_FETCH>::fetch_name(),
        choices.join(", ")
    ));
    need.check(&info)?;
    Ok(Box::new(Op {
        fallback,
        plans,
        info,
        wrapper: Entry::<MATRIX_FETCH>::fetch_name(),
    }))
}

/// The route wrapper a compact range reports.
pub(super) const RANGE_WRAPPER: &str = "compact_range";

/// The registration metadata of a compact range's `symbol`: its own entry,
/// apart from exact compact rows that launch the same symbol.
pub(super) fn range_tag(
    symbol: String,
    rows: std::ops::RangeInclusive<u32>,
) -> kernel_core::route::RouteTag {
    use kernel_core::route::{RouteClass, RouteTag};
    let entry = format!("mxfp8_prefill/{RANGE_WRAPPER}/{symbol}");
    RouteTag::new(symbol, RouteClass::Fast, "mxfp8_prefill", [rows])
        .with_entry(entry)
        .with_wrapper(RANGE_WRAPPER)
}

/// Compact ranges over `fallback`: each count inside a registered range
/// launches the compact entry planned for that count, prepared when it
/// launches, so a range keeps no plan per count. Every count of one range
/// uses one tile, so the range's symbols are its first count's.
pub(super) fn prepare_range(
    fallback: Prepared<Buffers<View>>,
    ranges: &[(Entry<false>, std::ops::RangeInclusive<u32>)],
    query: &Mxfp8FoldGemmQuery,
    need: Need,
) -> Result<Prepared<Buffers<View>>, Refused> {
    let ranges: Vec<_> = ranges
        .iter()
        .filter(|(e, _)| e.base.query == *query)
        .collect();
    if ranges.is_empty() {
        return Ok(fallback);
    }
    let mut info = fallback.info().clone();
    let mut kept = Vec::new();
    let mut choices = Vec::new();
    for (entry, rows) in ranges {
        let (first, last) = (*rows.start(), *rows.end());
        if first == 0 || first > last || last > query.m_capacity {
            return Err(Refusal("compact prefill range is empty or exceeds query capacity").into());
        }
        if Entry::<false>::tile(first) != Entry::<false>::tile(last) {
            return Err(Refusal("compact prefill range spans tiles").into());
        }
        if kept.iter().any(
            |(_, other): &(Entry<false>, std::ops::RangeInclusive<u32>)| {
                *other.start() <= last && first <= *other.end()
            },
        ) {
            return Err(Refusal("overlapping compact prefill ranges").into());
        }
        if first > need.capacity {
            continue;
        }
        let rows = first..=last.min(need.capacity);
        let op = entry.prepare(*rows.start())?;
        info.symbols.extend(op.info().symbols.clone());
        choices.push(format!("({first}..={last}, {})", entry.stages));
        kept.push((entry.clone(), rows));
    }
    if kept.is_empty() {
        return Ok(fallback);
    }
    info.symbols.sort();
    info.symbols.dedup();
    info.implementation
        .push_str(&format!(";{RANGE_WRAPPER}-v1:[{}]", choices.join(", ")));
    need.check(&info)?;
    Ok(Box::new(RangeOp {
        fallback,
        ranges: kept,
        info,
    }))
}

struct RangeOp {
    fallback: Prepared<Buffers<View>>,
    ranges: Vec<(Entry<false>, std::ops::RangeInclusive<u32>)>,
    info: OpInfo,
}
impl RangeOp {
    fn range(&self, rows: u32) -> Option<&(Entry<false>, std::ops::RangeInclusive<u32>)> {
        self.ranges.iter().find(|(_, range)| range.contains(&rows))
    }
}
impl PreparedOp<Buffers<View>> for RangeOp {
    fn route(&self, counts: Counts, launch: &Launch) -> Option<kernel_core::route::RouteTag> {
        use kernel_core::route::{RouteClass, RouteTag};
        if let Some((_, rows)) = self.range(counts.rows) {
            return Some(range_tag(launch.symbol.clone(), rows.clone()));
        }
        let mut tag = self.fallback.route(counts, launch).unwrap_or_else(|| {
            RouteTag::new(
                launch.symbol.clone(),
                RouteClass::Baseline,
                "mxfp8_prefill",
                [],
            )
        });
        tag.admitted
            .extend(self.ranges.iter().map(|(_, rows)| rows.clone()));
        tag.wrappers.push(RANGE_WRAPPER);
        Some(tag)
    }
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        buffers: &Buffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        match self.range(counts.rows) {
            Some((entry, _)) => entry
                .prepare(counts.rows)?
                .launches(buffers, workspace, counts),
            None => self.fallback.launches(buffers, workspace, counts),
        }
    }
}

struct Op {
    fallback: Prepared<Buffers<View>>,
    plans: BTreeMap<u32, Prepared<Buffers<View>>>,
    info: OpInfo,
    wrapper: &'static str,
}
impl PreparedOp<Buffers<View>> for Op {
    fn route(&self, counts: Counts, launch: &Launch) -> Option<kernel_core::route::RouteTag> {
        use kernel_core::route::{RouteClass, RouteTag};
        let rows: Vec<_> = self.plans.keys().map(|&r| r..=r).collect();
        if self.plans.contains_key(&counts.rows) {
            return Some(
                RouteTag::new(
                    launch.symbol.clone(),
                    RouteClass::Fast,
                    "mxfp8_prefill",
                    rows,
                )
                .with_wrapper(self.wrapper),
            );
        }
        let mut tag = self.fallback.route(counts, launch).unwrap_or_else(|| {
            RouteTag::new(
                launch.symbol.clone(),
                RouteClass::Baseline,
                "mxfp8_prefill",
                [],
            )
        });
        tag.admitted.extend(rows);
        tag.wrappers.push(self.wrapper);
        Some(tag)
    }
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        buffers: &Buffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        self.plans
            .get(&counts.rows)
            .unwrap_or(&self.fallback)
            .launches(buffers, workspace, counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm::mxfp8_prefill_tiles::compact::Entry;
    use crate::{backend::Sm120, gemm::mxfp8_fold};
    use kernel_core::{
        backend::Backend, contract::DType, precision::execution::BlockScaledExecution,
    };

    #[test]
    fn keep_fallback_capacity_and_reject_conflicting_row_routes() {
        let q = Mxfp8FoldGemmQuery {
            m_capacity: 512,
            n: 128,
            k: 128,
            groups: 1,
            activation_row_stride: 128,
            activation_group_stride: 128,
            output: DType::F32,
            execution: BlockScaledExecution::NativeBlockScaledMma,
        };
        let base = mxfp8_fold::Entry::new(q, "compact_route_test").unwrap();
        let old = || Sm120::new().with_mxfp8_prefill(base.clone());
        let e = Entry::new(base.clone(), 2).unwrap();
        let backend = old().with_mxfp8_prefill_compact(e.clone(), 128);
        let op = backend.mxfp8_prefill_gemm(&q, Need::rows(512)).unwrap();
        let fallback = old().mxfp8_prefill_gemm(&q, Need::rows(512)).unwrap();
        let v = |handle, bytes| View { handle, bytes };
        let sizes = base.bytes(512);
        let b = Buffers {
            activations: v(1, sizes.activations),
            activation_scales: v(2, sizes.activation_scales),
            weights: v(3, sizes.weights),
            weight_scales: v(4, sizes.weight_scales),
            output: v(5, sizes.output),
            live_rows: v(6, 4),
            status: v(7, sizes.status),
            status_row_offset: 0,
        };
        for rows in [0, 1, 127, 129, 256, 512] {
            assert_eq!(
                op.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap(),
                fallback
                    .launches(&b, View::EMPTY, Counts::rows(rows))
                    .unwrap()
            );
        }
        assert_eq!(
            op.launches(&b, View::EMPTY, Counts::rows(128)).unwrap(),
            e.prepare(128)
                .unwrap()
                .launches(&b, View::EMPTY, Counts::rows(128))
                .unwrap()
        );
        assert!(op.launches(&b, View::EMPTY, Counts::rows(513)).is_err());
        let shorter = backend.mxfp8_prefill_gemm(&q, Need::rows(64)).unwrap();
        assert_eq!(
            shorter.info(),
            old().mxfp8_prefill_gemm(&q, Need::rows(64)).unwrap().info()
        );
        assert!(
            backend
                .with_mxfp8_prefill_compact(Entry::new(base.clone(), 3).unwrap(), 128)
                .mxfp8_prefill_gemm(&q, Need::rows(512))
                .is_err()
        );
        for rows in [0, 513] {
            assert!(
                old()
                    .with_mxfp8_prefill_compact(e.clone(), rows)
                    .mxfp8_prefill_gemm(&q, Need::rows(512))
                    .is_err()
            );
        }
    }
}

#[cfg(test)]
mod range_tests {
    use super::*;
    use crate::gemm::mxfp8_prefill_tiles::compact::Entry;
    use crate::{backend::Sm120, gemm::mxfp8_fold};
    use kernel_core::{
        backend::Backend, contract::DType, precision::execution::BlockScaledExecution,
    };

    #[test]
    fn ranges_plan_each_count_and_keep_exact_rows_and_the_fallback() {
        let q = Mxfp8FoldGemmQuery {
            m_capacity: 512,
            n: 128,
            k: 128,
            groups: 1,
            activation_row_stride: 128,
            activation_group_stride: 128,
            output: DType::F32,
            execution: BlockScaledExecution::NativeBlockScaledMma,
        };
        let base = mxfp8_fold::Entry::new(q, "compact_range_test").unwrap();
        let old = || Sm120::new().with_mxfp8_prefill(base.clone());
        let two = Entry::new(base.clone(), 2).unwrap();
        let three = Entry::new(base.clone(), 3).unwrap();
        let backend = old()
            .with_mxfp8_prefill_compact_range(two.clone(), 129..=256)
            .with_mxfp8_prefill_compact(three.clone(), 200);
        let op = backend.mxfp8_prefill_gemm(&q, Need::rows(512)).unwrap();
        let fallback = old().mxfp8_prefill_gemm(&q, Need::rows(512)).unwrap();
        let v = |handle, bytes| View { handle, bytes };
        let sizes = base.bytes(512);
        let b = Buffers {
            activations: v(1, sizes.activations),
            activation_scales: v(2, sizes.activation_scales),
            weights: v(3, sizes.weights),
            weight_scales: v(4, sizes.weight_scales),
            output: v(5, sizes.output),
            live_rows: v(6, 4),
            status: v(7, sizes.status),
            status_row_offset: 0,
        };
        let launches = |op: &Prepared<Buffers<View>>, rows| {
            op.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap()
        };
        for rows in [1, 128, 257, 512] {
            assert_eq!(
                launches(&op, rows),
                launches(&fallback, rows),
                "rows={rows}"
            );
        }
        for rows in [129, 199, 201, 256] {
            let planned = two.prepare(rows).unwrap();
            assert_eq!(launches(&op, rows), launches(&planned, rows), "rows={rows}");
            for call in launches(&op, rows) {
                let tag = op.route(Counts::rows(rows), &call).unwrap();
                assert_eq!(
                    tag.entry,
                    format!("mxfp8_prefill/compact_range/{}", call.symbol)
                );
                assert_eq!(tag.class, kernel_core::route::RouteClass::Fast);
                // Outer wrappers combine their own rows into the admitted set.
                assert!(tag.admitted.contains(&(129..=256)));
                assert!(op.info().symbols.iter().any(|(symbol, shared)| {
                    *symbol == call.symbol && *shared >= call.plan.dynamic_shared_bytes
                }));
            }
        }
        // The exact registration keeps its own entry inside the range.
        let exact = three.prepare(200).unwrap();
        assert_eq!(launches(&op, 200), launches(&exact, 200));
        let registered = backend.route_tags();
        for call in launches(&op, 256) {
            assert!(registered.iter().any(|tag| tag.entry
                == format!("mxfp8_prefill/compact_range/{}", call.symbol)
                && tag.admitted == [129..=256]));
        }
        // A capacity below the range keeps the fallback's description.
        assert_eq!(
            backend
                .mxfp8_prefill_gemm(&q, Need::rows(128))
                .unwrap()
                .info(),
            old()
                .mxfp8_prefill_gemm(&q, Need::rows(128))
                .unwrap()
                .info()
        );
        for (first, last) in [(0, 64), (100, 200), (300, 600), (64, 32)] {
            assert!(
                old()
                    .with_mxfp8_prefill_compact_range(two.clone(), first..=last)
                    .mxfp8_prefill_gemm(&q, Need::rows(512))
                    .is_err(),
                "{first}..={last}"
            );
        }
        assert!(
            old()
                .with_mxfp8_prefill_compact_range(two.clone(), 129..=256)
                .with_mxfp8_prefill_compact_range(two.clone(), 256..=300)
                .mxfp8_prefill_gemm(&q, Need::rows(512))
                .is_err()
        );
    }
}

#[cfg(test)]
mod matrix_tests {
    use super::*;
    use crate::gemm::mxfp8_prefill_tiles::matrix::Entry;
    use crate::{backend::Sm120, gemm::mxfp8_fold};
    use kernel_core::{
        backend::Backend, contract::DType, precision::execution::BlockScaledExecution,
    };

    #[test]
    fn keep_fallback_capacity_and_reject_conflicting_row_routes() {
        let q = Mxfp8FoldGemmQuery {
            m_capacity: 512,
            n: 128,
            k: 128,
            groups: 1,
            activation_row_stride: 128,
            activation_group_stride: 128,
            output: DType::F32,
            execution: BlockScaledExecution::NativeBlockScaledMma,
        };
        let base = mxfp8_fold::Entry::new(q, "matrix_route_test").unwrap();
        let old = || Sm120::new().with_mxfp8_prefill(base.clone());
        let e = Entry::new(base.clone(), 2, 64).unwrap();
        let backend = old().with_mxfp8_prefill_matrix(e.clone(), 128);
        let op = backend.mxfp8_prefill_gemm(&q, Need::rows(512)).unwrap();
        let fallback = old().mxfp8_prefill_gemm(&q, Need::rows(512)).unwrap();
        let v = |handle, bytes| View { handle, bytes };
        let sizes = base.bytes(512);
        let b = Buffers {
            activations: v(1, sizes.activations),
            activation_scales: v(2, sizes.activation_scales),
            weights: v(3, sizes.weights),
            weight_scales: v(4, sizes.weight_scales),
            output: v(5, sizes.output),
            live_rows: v(6, 4),
            status: v(7, sizes.status),
            status_row_offset: 0,
        };
        for rows in [0, 1, 127, 129, 256, 512] {
            assert_eq!(
                op.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap(),
                fallback
                    .launches(&b, View::EMPTY, Counts::rows(rows))
                    .unwrap()
            );
        }
        assert_eq!(
            op.launches(&b, View::EMPTY, Counts::rows(128)).unwrap(),
            e.prepare(128)
                .unwrap()
                .launches(&b, View::EMPTY, Counts::rows(128))
                .unwrap()
        );
        assert!(op.launches(&b, View::EMPTY, Counts::rows(513)).is_err());
        let shorter = backend.mxfp8_prefill_gemm(&q, Need::rows(64)).unwrap();
        assert_eq!(
            shorter.info(),
            old().mxfp8_prefill_gemm(&q, Need::rows(64)).unwrap().info()
        );
        assert!(
            backend
                .with_mxfp8_prefill_matrix(Entry::new(base.clone(), 2, 32).unwrap(), 128)
                .mxfp8_prefill_gemm(&q, Need::rows(512))
                .is_err()
        );
        for rows in [0, 513] {
            assert!(
                old()
                    .with_mxfp8_prefill_matrix(e.clone(), rows)
                    .mxfp8_prefill_gemm(&q, Need::rows(512))
                    .is_err()
            );
        }
    }
}
