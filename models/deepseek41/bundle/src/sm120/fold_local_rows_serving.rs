//! Prepare the measured status entries through the shared typed interface.
//!
//! Every fold reads packed weight codes (`mxfp8-fold-packed-8x128-v1`), so
//! each op launches its entry's packed twin
//! (`kernels_sm120::backend::packed_op`). The row-local and cluster folds
//! have no packed entries; the status folds replace them at every bucket
//! they served, and without the status folds (other topologies, or their
//! reference in tests) the backend's decode folds serve.
use crate::ops::dense::{FusedFold, SlicedQuantize};
use crate::ops::moe::QuantizeViews;
use crate::ops::{Bind, GemmViews, Lowered};
use kernel_core::backend::{Counts, Need, OpInfo, Prepared, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldGemmQuery};
use kernels_sm120::gemm::mxfp8_fold_cluster_fused::{FusedBuffers, Quantized, Quantizer};
use runtime_bundle_abi::DeviceBufferSpanV1;

/// WO_a's or WO_b's status fold over its fused buffers (`fold_fused`).
struct StatusFold(Prepared<FusedBuffers>);

impl FusedFold for StatusFold {
    fn info(&self) -> &OpInfo {
        self.0.info()
    }

    fn lower(
        &self,
        views: &GemmViews,
        next: Option<[DeviceBufferSpanV1; 3]>,
        pending: Option<DeviceBufferSpanV1>,
        workspace: DeviceBufferSpanV1,
        rows: u32,
    ) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let buffers = FusedBuffers {
            fold: Mxfp8FoldGemmBuffers {
                activations: bind.view(views.activations),
                activation_scales: bind.view(views.activation_scales),
                weights: bind.view(views.weights),
                weight_scales: bind.view(views.weight_scales),
                output: bind.view(views.output),
                live_rows: bind.view(views.live_rows),
                status: bind.view(views.status),
                status_row_offset: 0,
            },
            quantized: next.map(|[codes, scales, flags]| Quantized {
                codes: bind.view(codes),
                scales: bind.view(scales),
                flags: bind.view(flags),
            }),
            pending: pending.map(|flags| bind.view(flags)),
        };
        bind.launch_within(&self.0, &buffers, workspace, Counts::rows(rows))
    }
}

/// WO_a's row-sliced input quantizer (`fold_fused::SLICED`), whose rejected
/// rows only a status fold may follow.
struct SlicedQuantizer(Quantizer);

impl SlicedQuantize for SlicedQuantizer {
    fn lower(&self, views: &QuantizeViews, rows: u32) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let launch = self.0.launch(
            bind.view(views.input),
            bind.view(views.codes),
            bind.view(views.scales),
            bind.view(views.live_rows),
            bind.view(views.status),
            0,
            rows,
        )?;
        Ok(bind.lowered_launches(vec![launch]))
    }
}

/// `entry` serves `query` at `need.capacity` rows.
fn matches(entry: Mxfp8FoldGemmQuery, query: Mxfp8FoldGemmQuery, need: Need) -> bool {
    query.m_capacity == need.capacity
        && Mxfp8FoldGemmQuery {
            m_capacity: query.m_capacity,
            ..entry
        } == query
}
impl crate::Kernels {
    /// WO_a's or WO_b's status fold over its fused buffers at
    /// `need.capacity` rows, if one serves (`fold_fused`).
    pub(crate) fn prepare_fused_fold(
        shape: &crate::shape::Shape,
        id: &str,
        query: Mxfp8FoldGemmQuery,
        need: Need,
    ) -> Result<Option<Box<dyn FusedFold>>, Refused> {
        if !super::fold_fused::selected(shape.geometry, shape.ranks())
            || !matches!(id, "attention.o-a" | "attention.o-b")
        {
            return Ok(None);
        }
        let Some(entry) = super::fold_fused::fused(id, need.capacity) else {
            return Ok(None);
        };
        if !matches(entry.cluster.base.query, query, need) {
            return Err(Refusal("a status fold differs from its projection").into());
        }
        let op = kernels_sm120::backend::packed_op(entry.prepare(need)?, [entry.symbol()])?;
        Ok(Some(Box::new(StatusFold(op)) as Box<dyn FusedFold>))
    }

    /// Whether the status folds serve `shape` (and its scratch holds WO_a's
    /// block flags).
    pub(crate) fn status_folds(shape: &crate::shape::Shape) -> bool {
        super::fold_fused::selected(shape.geometry, shape.ranks())
    }

    /// WO_a's sliced input quantizer by bucket, where its status fold
    /// serves.
    pub(crate) fn sliced_quantizers(
        shape: &crate::shape::Shape,
    ) -> Vec<(u32, Box<dyn SlicedQuantize>)> {
        if !super::fold_fused::selected(shape.geometry, shape.ranks()) {
            return Vec::new();
        }
        super::fold_fused::ROWS
            .iter()
            .filter_map(|&rows| {
                super::fold_fused::sliced(rows).map(|q| {
                    (
                        rows,
                        Box::new(SlicedQuantizer(q)) as Box<dyn SlicedQuantize>,
                    )
                })
            })
            .collect()
    }

    /// The status fold without fused buffers measured for projection `id`
    /// at `need.capacity` rows, if any; `query` is the projection's fold at
    /// that bucket.
    pub(crate) fn prepare_local_fold(
        shape: &crate::shape::Shape,
        id: &str,
        query: Mxfp8FoldGemmQuery,
        need: Need,
    ) -> Result<Option<Prepared<Mxfp8FoldGemmBuffers<View>>>, Refused> {
        // The status folds without fused buffers: WQ_b's wide fold, and
        // WQ_a's and WKV's cluster folds without their admission launch.
        if super::fold_fused::selected(shape.geometry, shape.ranks()) {
            if id == "attention.q-b"
                && let Some(entry) = super::fold_fused::wide(need.capacity)
            {
                if !matches(entry.base.query, query, need) {
                    return Err(Refusal("a status fold differs from its projection").into());
                }
                return Ok(Some(kernels_sm120::backend::packed_op(
                    entry.prepare(need)?,
                    [entry.symbol()],
                )?));
            }
            if matches!(id, "attention.q-a" | "attention.kv")
                && let Some(entry) = super::fold_fused::fused(id, need.capacity)
            {
                if !matches(entry.cluster.base.query, query, need) {
                    return Err(Refusal("a status fold differs from its projection").into());
                }
                return Ok(Some(kernels_sm120::backend::packed_op(
                    entry.prepare_plain(need)?,
                    [entry.symbol()],
                )?));
            }
            if matches!(id, "attention.o-a" | "attention.o-b")
                && super::fold_fused::fused(id, need.capacity).is_some()
            {
                // `prepare_fused_fold` serves these buckets.
                return Ok(None);
            }
        }
        Ok(None)
    }
}
