//! Selected entries qualified for target verification above the default row budget.
use kernel_core::backend::{Prepared, PreparedOp};
use kernel_core::route::{RouteClass, RouteTag};

pub(super) fn tag(symbol: &str, family: &'static str, first: u32) -> RouteTag {
    kernel_core::route::branch(
        symbol,
        RouteClass::Fast,
        family,
        "wide-verify",
        [first..=64],
    )
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

#[cfg(test)]
pub(super) fn catalogue(shape: &crate::shape::Shape) -> Result<Vec<RouteTag>, String> {
    let mut tags = Vec::new();
    let rows = shape.verify_decode_rows();
    if !super::f32_verify::enabled(shape.geometry, shape.ranks(), rows) {
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
    if let Some(op) = crate::Kernels::wide_engram_fold(shape, rows).map_err(|e| format!("{e:?}"))? {
        tags.extend(self::tags(op.as_ref(), "mxfp8_prefill", 33));
    }
    let need = crate::ops::exact::need(rows, crate::ops::exact::verify(rows));
    let query = model_deepseek41_contract::indexer::selection(
        shape.geometry,
        shape.geometry.shape.csa.index_sources[0],
        rows,
    )
    .ok_or("missing wide index query")?;
    let op = crate::Kernels::wide_index_query(shape, &query, need).map_err(|e| format!("{e:?}"))?;
    tags.extend(self::tags(op.as_ref(), "indexer_query", 33));
    if rows > super::latent_scheduled::QUERY_ROWS {
        let query =
            model_deepseek41_contract::latent_glue::query_rope(shape.geometry, shape.ranks())?;
        let op =
            crate::Kernels::wide_query_rope(shape, &query, need).map_err(|e| format!("{e:?}"))?;
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
    fn every_wide_query_and_engram_row_reports_the_selected_target_entry() {
        let _metadata = Metadata::enable();
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap().with_verify_rows(64);
        let kernels = crate::Kernels::new(&shape).unwrap();
        let span =
            |id| DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_raw(id), 0, 1 << 32);
        let index_views = QueryViews {
            q: span(1),
            cos_sin: span(2),
            positions: span(3),
            live_rows: span(4),
            output: span(5),
            workspace: span(6),
            status: span(7),
        };
        let latent_views = QueryRopeViews {
            query: span(1),
            native_q: span(2),
            cos_sin: span(3),
            live_rows: span(4),
            status: span(5),
        };
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
}
