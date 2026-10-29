//! The candidate main-projection range, using the existing compact template.
use kernel_core::backend::{
    Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View,
};
use kernel_core::contract::Refusal;
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers as Buffers, Mxfp8FoldGemmQuery};
use kernel_core::route::{RouteClass, RouteTag};
use kernels_sm120::backend::Sm120;
use kernels_sm120::gemm::mxfp8_prefill_compact::Entry;
use model_deepseek41_contract::tuning::{Family, select};
use std::ops::RangeInclusive;

const SITE: &str = "dense_prefill.dspark_main.prefill";
const FAMILY: &str = "dspark_main";

fn recipe(
    shape: &crate::shape::Shape,
    query: Mxfp8FoldGemmQuery,
) -> Option<(RangeInclusive<u32>, u32)> {
    let row = select(
        shape.geometry,
        Family::DensePrefill,
        SITE,
        "mxfp8",
        &[
            ("n", query.n),
            ("k", query.k),
            ("groups", query.groups),
            ("ranks", shape.ranks()),
        ],
        query.m_capacity,
        129,
    )?;
    let [first, last] = row.key.rows.expect("main projection range");
    assert_eq!(row.schedule, "prefill_tiles");
    assert_eq!(row.knobs["matrix_fetch"].flag(), Some(false));
    assert_eq!(row.knobs["tile_rows"].count(), Some(Entry::tile(first)));
    assert_eq!(Entry::tile(first), Entry::tile(last));
    Some((
        first..=last,
        row.knobs["stages"].count().expect("compact stages"),
    ))
}

fn candidate(
    shape: &crate::shape::Shape,
    query: Mxfp8FoldGemmQuery,
    rows: RangeInclusive<u32>,
    stages: u32,
) -> Result<Sm120, Refused> {
    let base = super::mxfp8_fold::prefill_entries(shape.geometry, shape.ranks())
        .map_err(Refused::Rule)?
        .into_iter()
        .find(|entry| entry.query == query)
        .ok_or(Refusal("main compact query has no compiled prefill entry"))?;
    let entry = Entry::new(base.clone(), stages)?;
    let mut backend = Sm120::new().with_mxfp8_prefill(base);
    // The shared wrapper prepares each count once and normalizes adjacent
    // registration rows into one range in its route metadata.
    for count in rows {
        backend = backend.with_mxfp8_prefill_compact(entry.clone(), count);
    }
    Ok(backend.with_packed_fold_weights())
}

pub(crate) fn prepare(
    backend: &dyn Backend,
    shape: &crate::shape::Shape,
    query: Mxfp8FoldGemmQuery,
    need: Need,
    enabled: bool,
) -> Result<Prepared<Buffers<View>>, Refused> {
    let fallback = backend.mxfp8_fold_gemm(&query, need)?;
    let Some((rows, stages)) = recipe(shape, query) else {
        return Ok(fallback);
    };
    let mut info = fallback.info().clone();
    let compact = if enabled {
        let backend = candidate(shape, query, rows.clone(), stages)?;
        let op = backend.mxfp8_prefill_gemm(&query, need)?;
        if op.info().bytes != info.bytes
            || op.info().weight_layouts != info.weight_layouts
            || op.info().workspace != info.workspace
        {
            return Err(Refusal("main compact changes operand layout or workspace").into());
        }
        let symbols: std::collections::BTreeSet<_> = backend
            .route_tags()
            .into_iter()
            .filter(|tag| tag.family == "mxfp8_prefill" && tag.class == RouteClass::Fast)
            .map(|tag| tag.symbol)
            .collect();
        info.symbols.extend(
            op.info()
                .symbols
                .iter()
                .filter(|(symbol, _)| symbols.contains(symbol))
                .cloned(),
        );
        info.symbols.sort();
        info.symbols.dedup();
        info.implementation
            .push_str(&format!(";dspark-main-compact:{rows:?}:stages={stages}"));
        Some(op)
    } else {
        None
    };
    need.check(&info)?;
    Ok(Box::new(Main {
        fallback,
        compact,
        rows,
        info,
    }))
}

struct Main {
    fallback: Prepared<Buffers<View>>,
    compact: Option<Prepared<Buffers<View>>>,
    rows: RangeInclusive<u32>,
    info: OpInfo,
}

impl PreparedOp<Buffers<View>> for Main {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &Buffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let op = self
            .compact
            .as_ref()
            .filter(|_| self.rows.contains(&counts.rows))
            .unwrap_or(&self.fallback);
        op.launches(buffers, workspace, counts)
    }

    fn route(&self, counts: Counts, launch: &Launch) -> Option<RouteTag> {
        let compact = self.compact.is_some() && self.rows.contains(&counts.rows);
        Some(kernel_core::route::branch(
            &launch.symbol,
            if compact {
                RouteClass::Fast
            } else {
                RouteClass::Baseline
            },
            FAMILY,
            if compact {
                "compact"
            } else if counts.rows < *self.rows.start() {
                "generic-below-range"
            } else if counts.rows > *self.rows.end() {
                "generic-above-range"
            } else {
                "candidate-off"
            },
            self.compact.iter().map(|_| self.rows.clone()),
        ))
    }
}

#[cfg(test)]
pub(crate) fn route_tags(shape: &crate::shape::Shape, enabled: bool) -> Vec<RouteTag> {
    let mut tags = Vec::new();
    for base in super::mxfp8_fold::prefill_entries(shape.geometry, shape.ranks()).unwrap() {
        let Some((rows, stages)) = recipe(shape, base.query) else {
            continue;
        };
        for wrapper in ["generic-below-range", "generic-above-range"]
            .into_iter()
            .chain((!enabled).then_some("candidate-off"))
        {
            tags.push(kernel_core::route::branch(
                &kernels_sm120::gemm::mxfp8_fold_packed::symbol(&base.symbol),
                RouteClass::Baseline,
                FAMILY,
                wrapper,
                enabled.then_some(rows.clone()),
            ));
        }
        if enabled {
            let backend = candidate(shape, base.query, rows.clone(), stages).unwrap();
            tags.extend(
                backend
                    .route_tags()
                    .into_iter()
                    .filter(|tag| tag.family == "mxfp8_prefill" && tag.class == RouteClass::Fast)
                    .map(|tag| {
                        kernel_core::route::branch(
                            &tag.symbol,
                            RouteClass::Fast,
                            FAMILY,
                            "compact",
                            [rows.clone()],
                        )
                    }),
            );
        }
    }
    kernel_core::route::merge(tags).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;
    use model_deepseek41_contract::geometry::FLASH;

    #[test]
    fn contiguous_main_range_keeps_every_other_launch_and_layout() {
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let shape = crate::shape::Shape::new(&FLASH, ranks).unwrap();
            let backend = crate::Kernels::new(&shape).unwrap();
            let base = super::super::mxfp8_fold::prefill_entries(&FLASH, shape.ranks())
                .unwrap()
                .into_iter()
                .find(|e| e.query.n == 5120 && e.query.k == 15360)
                .unwrap();
            let q = base.query;
            let old = prepare(
                backend.backend.as_ref(),
                &shape,
                q,
                Need::rows(q.m_capacity),
                false,
            )
            .unwrap();
            let new = prepare(
                backend.backend.as_ref(),
                &shape,
                q,
                Need::rows(q.m_capacity),
                true,
            )
            .unwrap();
            assert_eq!(new.info().bytes, old.info().bytes);
            assert_eq!(new.info().workspace, old.info().workspace);
            assert_eq!(new.info().weight_layouts, old.info().weight_layouts);
            assert_eq!(new.info().guarantees, old.info().guarantees);
            let b = base.bytes(q.m_capacity);
            let v = |handle, bytes| View { handle, bytes };
            let buffers = Buffers {
                activations: v(1, b.activations),
                activation_scales: v(2, b.activation_scales),
                weights: v(3, b.weights),
                weight_scales: v(4, b.weight_scales),
                output: v(5, b.output),
                live_rows: v(6, 4),
                status: v(7, b.status),
                status_row_offset: 0,
            };
            for count in 1..=q.m_capacity {
                let counts = Counts::rows(count);
                let a = old.launches(&buffers, View::EMPTY, counts).unwrap();
                let b = new.launches(&buffers, View::EMPTY, counts).unwrap();
                if ranks == Ranks::TP4 && (129..=511).contains(&count) {
                    assert_eq!(a.len(), 1);
                    assert_eq!(b.len(), 2);
                    assert!(
                        b.iter()
                            .all(|call| call.symbol.contains("_compact_m64_n128_s2_v1"))
                    );
                    for call in b {
                        let tag = new.route(counts, &call).unwrap();
                        assert_eq!(tag.class, RouteClass::Fast);
                        assert_eq!(tag.admitted, [129..=511]);
                        assert!(
                            new.info()
                                .symbols
                                .iter()
                                .any(|(symbol, _)| *symbol == call.symbol)
                        );
                    }
                } else {
                    assert_eq!(a, b, "rank={ranks:?} rows={count}");
                    if ranks == Ranks::TP4 {
                        let tag = new.route(counts, &b[0]).unwrap();
                        assert_eq!(tag.class, RouteClass::Baseline);
                        let wrapper = if count < 129 {
                            "generic-below-range"
                        } else {
                            "generic-above-range"
                        };
                        assert_eq!(tag.wrappers, [wrapper]);
                    }
                }
                if ranks == Ranks::TP4 && (129..=511).contains(&count) {
                    assert_eq!(
                        old.route(counts, &a[0]).unwrap().wrappers,
                        ["candidate-off"]
                    );
                }
            }
        }
    }
}
