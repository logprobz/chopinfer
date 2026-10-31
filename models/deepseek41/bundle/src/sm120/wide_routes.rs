//! Selected entries qualified for target verification above the default row budget.
use kernel_build::tuning::Table;
use kernel_core::backend::{Prepared, PreparedOp};
use kernel_core::route::{RouteClass, RouteTag};
use model_deepseek41_contract::{
    geometry::FLASH,
    tuning::{self, Family},
    verify_bounds,
};

/// The verify bound of a wide route family.
fn bound(family: &str) -> u32 {
    match family {
        "f32_projection" => verify_bounds::F32_CACHED,
        "indexer_query" => verify_bounds::INDEX_QUERY,
        "latent_query" => verify_bounds::LATENT_QUERY,
        "mxfp8_prefill" => verify_bounds::ENGRAM_FOLD,
        _ => unreachable!("unregistered wide route family"),
    }
}

/// `first` applies only to Engram's structural admission.
pub(super) fn tag(symbol: &str, family: &'static str, first: u32) -> RouteTag {
    tag_with_tables(symbol, family, first, |source| {
        tuning::table(&FLASH, source).expect("Flash wide route table")
    })
}

fn tag_with_tables<'a>(
    symbol: &str,
    family: &'static str,
    first: u32,
    tables: impl Fn(Family) -> &'a Table,
) -> RouteTag {
    tag_at(symbol, family, first, bound(family), tables)
}

/// The tag of `symbol` admitted by `family`'s table rows at `capacity`.
fn tag_at<'a>(
    symbol: &str,
    family: &'static str,
    first: u32,
    capacity: u32,
    tables: impl Fn(Family) -> &'a Table,
) -> RouteTag {
    let key = match family {
        "f32_projection" => {
            let entry = super::f32_verify::entries()
                .expect("wide F32 entries")
                .into_iter()
                .find(|entry| entry.symbols().iter().any(|name| name == symbol))
                .expect("registered wide F32 symbol");
            let query = entry.entry().query();
            Some((
                Family::F32,
                "f32.cached.verify",
                "bf16",
                std::collections::BTreeMap::from([
                    ("n".to_owned(), query.n),
                    ("k".to_owned(), query.k),
                    ("ranks".to_owned(), 4),
                ]),
            ))
        }
        "indexer_query" => Some((
            Family::Indexer,
            "indexer.query.verify",
            "f32",
            [("ranks".to_owned(), 4)].into(),
        )),
        "latent_query" => Some((
            Family::Latent,
            "latent.query.verify",
            "f32",
            [("ranks".to_owned(), 4)].into(),
        )),
        "mxfp8_prefill" => None,
        _ => unreachable!("unregistered wide route family"),
    };
    // Export the full verification envelope, including for smaller instances.
    let admitted = if let Some((source, op, weights, shape)) = key {
        tables(source)
            .rows()
            .iter()
            .filter(|row| {
                row.key.op == op
                    && row.key.weights == weights
                    && row.key.shape == shape
                    && row.key.bucket == capacity
            })
            .map(|row| {
                let [first, last] = row.key.rows.unwrap_or([1, row.key.bucket]);
                first..=last
            })
            .collect::<Vec<_>>()
    } else {
        vec![first..=capacity]
    };
    kernel_core::route::branch(symbol, RouteClass::Fast, family, "wide-verify", admitted)
}

pub(super) fn tags<B>(op: &dyn PreparedOp<B>, family: &'static str, first: u32) -> Vec<RouteTag> {
    op.info()
        .symbols
        .iter()
        .map(|(symbol, _)| tag(symbol, family, first))
        .collect()
}

pub(super) fn tagged<B: 'static>(op: Prepared<B>, family: &'static str, first: u32) -> Prepared<B> {
    let tags = tags(op.as_ref(), family, first);
    kernel_core::route::tagged(op, tags)
}

/// `op` of a wide verify bucket (`Shape::wide_verify_rows`, the
/// `wide_verify` candidate) tagged with the rows `family`'s table admits
/// at the widest bucket (`verify_bounds::WIDE`).
pub(super) fn tagged_wide<B: 'static>(op: Prepared<B>, family: &'static str) -> Prepared<B> {
    let capacity = verify_bounds::WIDE[verify_bounds::WIDE.len() - 1];
    let tags = op
        .info()
        .symbols
        .iter()
        .map(|(symbol, _)| {
            tag_at(symbol, family, capacity, capacity, |source| {
                tuning::table(&FLASH, source).expect("Flash wide route table")
            })
        })
        .collect();
    kernel_core::route::tagged(op, tags)
}

#[cfg(test)]
pub(super) fn catalogue(shape: &crate::shape::Shape) -> Result<Vec<RouteTag>, String> {
    let mut tags = Vec::new();
    let rows = |family| shape.verify_rows_within(bound(family));
    if !super::f32_verify::enabled(shape.geometry, shape.ranks(), rows("f32_projection")) {
        return Ok(tags);
    }
    for entry in super::f32_verify::entries().map_err(|e| e.to_string())? {
        tags.extend(
            entry
                .symbols()
                .iter()
                .map(|symbol| tag(symbol, "f32_projection", 33)),
        );
    }
    if let Some(op) = crate::Kernels::wide_engram_fold(shape, rows("mxfp8_prefill"))
        .map_err(|e| format!("{e:?}"))?
    {
        tags.extend(self::tags(op.as_ref(), "mxfp8_prefill", 33));
    }
    let need = |rows| crate::ops::exact::need(rows, crate::ops::exact::verify(rows));
    let index_rows = rows("indexer_query");
    let query = model_deepseek41_contract::indexer::selection(
        shape.geometry,
        shape.geometry.shape.csa.index_sources[0],
        index_rows,
    )
    .ok_or("missing wide index query")?;
    let op = crate::Kernels::wide_index_query(shape, &query, need(index_rows))
        .map_err(|e| format!("{e:?}"))?;
    tags.extend(self::tags(op.as_ref(), "indexer_query", 33));
    let latent_rows = rows("latent_query");
    if latent_rows > super::latent_scheduled::QUERY_ROWS {
        let query =
            model_deepseek41_contract::latent_glue::query_rope(shape.geometry, shape.ranks())?;
        let op = crate::Kernels::wide_query_rope(shape, &query, need(latent_rows))
            .map_err(|e| format!("{e:?}"))?;
        tags.extend(self::tags(op.as_ref(), "latent_query", 49));
    }
    Ok(tags)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ops::{
            Asks,
            indexer::{IndexerOps, QueryViews},
            latent::{GlueOps, QueryRopeViews},
        },
        program::Phase,
        shape::Shape,
    };
    use bundle_runtime::geometry::Ranks;
    use kernel_core::{
        backend::{Counts, View},
        mxfp8_gemm::Mxfp8FoldGemmBuffers,
        route::Metadata,
    };
    use model_deepseek41_contract::{geometry::FLASH, state::KvStorage};
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};

    #[test]
    fn wide_route_ranges_follow_mutated_table_bounds() {
        let mut cases = 0;
        for (source, family, op, weights, first) in [
            (
                Family::F32,
                "f32_projection",
                "f32.cached.verify",
                "bf16",
                33,
            ),
            (
                Family::Indexer,
                "indexer_query",
                "indexer.query.verify",
                "f32",
                33,
            ),
            (
                Family::Latent,
                "latent_query",
                "latent.query.verify",
                "f32",
                49,
            ),
        ] {
            let table = tuning::table(&FLASH, source).unwrap();
            for selected in table.rows().iter().filter(|row| {
                row.key.op == op && row.key.weights == weights && row.key.bucket == 64
            }) {
                let mut changed = Table::new(table.target);
                for mut row in table.rows().iter().cloned() {
                    if row.key == selected.key {
                        row.key.rows = Some([first + 1, 64]);
                    }
                    changed.upsert(row).unwrap();
                }
                assert!(
                    changed
                        .select(op, weights, &selected.key.shape, 64, first)
                        .is_none()
                );
                assert!(
                    changed
                        .select(op, weights, &selected.key.shape, 64, first + 1)
                        .is_some()
                );
                let symbols = if source == Family::F32 {
                    super::super::f32_verify::entries()
                        .unwrap()
                        .into_iter()
                        .find(|entry| {
                            let query = entry.entry().query();
                            query.n == selected.key.shape["n"] && query.k == selected.key.shape["k"]
                        })
                        .unwrap()
                        .symbols()
                        .to_vec()
                } else {
                    vec![family.to_owned()]
                };
                for symbol in symbols {
                    let mut expected = tag(&symbol, family, first);
                    assert_eq!(expected.admitted, [first..=64]);
                    expected.admitted = vec![first + 1..=64];
                    let observed = tag_with_tables(&symbol, family, first, |requested| {
                        assert_eq!(requested, source);
                        &changed
                    });
                    assert_eq!(observed, expected, "{op} {:?}", selected.key.shape);
                }
                cases += 1;
            }
        }
        assert_eq!(cases, 7);
    }

    #[test]
    fn wide_route_ranges_keep_table_segments_and_engram_coverage() {
        let table = tuning::table(&FLASH, Family::Indexer).unwrap();
        let mut changed = Table::new(table.target);
        for mut row in table.rows().iter().cloned() {
            if row.key.op == "indexer.query.verify" && row.key.bucket == 64 {
                let mut first = row.clone();
                first.key.rows = Some([33, 40]);
                changed.upsert(first).unwrap();
                row.key.rows = Some([42, 64]);
            }
            changed.upsert(row).unwrap();
        }
        let observed = tag_with_tables("query", "indexer_query", 33, |_| &changed);
        assert_eq!(observed.admitted, [33..=40, 42..=64]);
        let engram = tag_with_tables("engram", "mxfp8_prefill", 33, |_| {
            panic!("Engram has structural admission")
        });
        assert_eq!(engram.admitted, [33..=64]);
        assert_eq!(engram.wrappers, ["wide-verify"]);
    }

    fn span(id: u64) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_raw(id), 0, 1 << 32)
    }

    fn index_views() -> QueryViews {
        QueryViews {
            q: span(1),
            cos_sin: span(2),
            positions: span(3),
            live_rows: span(4),
            output: span(5),
            workspace: span(6),
            status: span(7),
        }
    }

    fn latent_views() -> QueryRopeViews {
        QueryRopeViews {
            query: span(1),
            native_q: span(2),
            cos_sin: span(3),
            live_rows: span(4),
            status: span(5),
        }
    }

    #[test]
    fn every_wide_query_and_engram_row_reports_the_selected_target_entry() {
        let _metadata = Metadata::enable();
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap().with_verify_rows(64);
        let kernels = crate::Kernels::new(&shape).unwrap();
        let (index_views, latent_views) = (index_views(), latent_views());
        for target in [true, false] {
            let asks = Asks::default();
            let index =
                IndexerOps::prepare(kernels.backend.as_ref(), &shape, &asks, target).unwrap();
            let latent = GlueOps::prepare(
                kernels.backend.as_ref(),
                kernels.own.as_ref(),
                &shape,
                &asks,
                KvStorage::Native,
                target,
            )
            .unwrap();
            for rows in 33..=64 {
                for phase in [
                    Phase::Verify {
                        rows,
                        items: rows.div_ceil(6),
                    },
                    Phase::Prefill { rows },
                ] {
                    for (calls, first, family) in [
                        (
                            index.prepare_query(phase, 65536, &index_views).unwrap(),
                            33,
                            "indexer_query",
                        ),
                        (
                            latent.query_rope(phase, &latent_views).unwrap(),
                            49,
                            "latent_query",
                        ),
                    ] {
                        assert!(!calls.is_empty());
                        for call in calls {
                            let selected = call
                                .launch
                                .bindings
                                .route()
                                .filter(|tag| tag.wrappers.contains(&"wide-verify"));
                            assert_eq!(
                                selected.is_some(),
                                target && !phase.prefill() && rows >= first
                            );
                            if let Some(tag) = selected {
                                assert_eq!((tag.class, tag.family), (RouteClass::Fast, family));
                                assert!(tag.admitted.iter().any(|range| range.contains(&rows)));
                            }
                        }
                    }
                }
            }
        }
        let view = |handle| View {
            handle,
            bytes: 1 << 32,
        };
        let buffers = Mxfp8FoldGemmBuffers {
            activations: view(1),
            activation_scales: view(2),
            weights: view(3),
            weight_scales: view(4),
            output: view(5),
            live_rows: view(6),
            status: view(7),
            status_row_offset: 0,
        };
        for rows in 33..=64 {
            let op = crate::Kernels::wide_engram_fold(&shape, rows)
                .unwrap()
                .unwrap();
            let counts = Counts::rows(rows);
            let calls = op.launches(&buffers, view(8), counts).unwrap();
            assert!(!calls.is_empty());
            for call in calls {
                let tag = kernel_core::route::selected(op.as_ref(), counts, &call).unwrap();
                assert_eq!((tag.class, tag.family), (RouteClass::Fast, "mxfp8_prefill"));
                assert!(tag.admitted.iter().any(|range| range.contains(&rows)));
            }
        }
    }

    /// With the `wide_verify` option, every target verify step above the
    /// decode route takes the decode queries' entries at its wide verify
    /// bucket, tagged with the widest bucket's rows; without it, in
    /// prefill and in the drafter, none does.
    #[test]
    fn wide_verify_query_rows_report_the_widest_bucket_entry() {
        let _metadata = Metadata::enable();
        let (index_views, latent_views) = (index_views(), latent_views());
        for enabled in [true, false] {
            let shape = Shape::new(&FLASH, Ranks::TP4)
                .unwrap()
                .with_verify_rows(192)
                .with_wide_verify(enabled);
            let kernels = crate::Kernels::new(&shape).unwrap();
            for target in [true, false] {
                let asks = Asks::default();
                let index =
                    IndexerOps::prepare(kernels.backend.as_ref(), &shape, &asks, target).unwrap();
                let latent = GlueOps::prepare(
                    kernels.backend.as_ref(),
                    kernels.own.as_ref(),
                    &shape,
                    &asks,
                    KvStorage::Native,
                    target,
                )
                .unwrap();
                for rows in 65..=192 {
                    for phase in [
                        Phase::Verify {
                            rows,
                            items: rows.div_ceil(6),
                        },
                        Phase::Prefill { rows },
                    ] {
                        for (calls, symbol, family) in [
                            (
                                index.prepare_query(phase, 65536, &index_views).unwrap(),
                                "chop_position_query_deepseek41_parallel_v1",
                                "indexer_query",
                            ),
                            (
                                latent.query_rope(phase, &latent_views).unwrap(),
                                "chop_deepseek41_latent_query_rope_heads_v1",
                                "latent_query",
                            ),
                        ] {
                            let wide = enabled && target && !phase.prefill();
                            assert_eq!(
                                calls.len(),
                                if wide || family == "latent_query" {
                                    1
                                } else {
                                    2
                                }
                            );
                            for call in calls {
                                let selected = call
                                    .launch
                                    .bindings
                                    .route()
                                    .filter(|tag| tag.wrappers.contains(&"wide-verify"));
                                assert_eq!(selected.is_some(), wide, "{family} {rows} {phase:?}");
                                assert_eq!(call.launch.symbol == symbol, wide);
                                if let Some(tag) = selected {
                                    assert_eq!((tag.class, tag.family), (RouteClass::Fast, family));
                                    assert_eq!(tag.admitted, [65..=192]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
