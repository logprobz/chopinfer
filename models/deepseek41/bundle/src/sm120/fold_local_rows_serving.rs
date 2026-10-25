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
fn cluster(
    shape: &crate::shape::Shape,
    id: &str,
    rows: u32,
) -> Option<kernels_sm120::gemm::mxfp8_fold_cluster_fused::Entry> {
    if super::fold_verify::enabled(shape.geometry, shape.ranks(), shape.verify_rows)
        && rows <= shape.verify_decode_rows()
        && let Some(entry) = super::fold_verify::served(id, rows)
    {
        return Some(entry);
    }
    super::fold_tails::cluster(id, rows).or_else(|| super::fold_fused::fused(id, rows))
}

impl crate::Kernels {
    /// Drafter blocks reuse the input/query norm quantizers and private
    /// scratch, including WO_a's block flags. Only their measured complete
    /// five-position blocks may use the additional status folds.
    pub(crate) fn drafter_status_fold(shape: &crate::shape::Shape, id: &str, rows: u32) -> bool {
        super::fold_fused::selected(shape.geometry, shape.ranks())
            && matches!(
                id,
                "attention.q-a" | "attention.q-b" | "attention.o-a" | "attention.o-b"
            )
            && (5..=60).contains(&rows)
            && rows.is_multiple_of(5)
    }

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
        let Some(entry) = cluster(shape, id, need.capacity) else {
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
        (1..=Self::dense_decode_capacity())
            .filter_map(|rows| {
                cluster(shape, "attention.o-a", rows)?;
                let q = if rows > 32 {
                    super::fold_fused::sliced(32).map(|mut q| {
                        q.capacity = 64;
                        q
                    })
                } else {
                    super::fold_fused::sliced(super::fold_tails::tile(rows).unwrap_or(rows))
                };
                q.map(|q| {
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
                && (super::fold_tails::wide_rows(need.capacity)
                    || (super::fold_verify::enabled(
                        shape.geometry,
                        shape.ranks(),
                        shape.verify_rows,
                    ) && (33..=shape.verify_decode_rows()).contains(&need.capacity)))
            {
                let entry = super::fold_verify_local::entry()?;
                if !matches(entry.query(), query, need) {
                    return Err(Refusal("a status fold differs from its projection").into());
                }
                return Ok(Some(kernels_sm120::backend::packed_op(
                    entry.prepare(need)?,
                    [entry.symbol()],
                )?));
            }
            if id == "attention.q-b"
                && let Some(entry) = super::fold_fused::wide(
                    super::fold_tails::tile(need.capacity).unwrap_or(need.capacity),
                )
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
                && let Some(entry) = cluster(shape, id, need.capacity)
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
                && cluster(shape, id, need.capacity).is_some()
            {
                // `prepare_fused_fold` serves these buckets.
                return Ok(None);
            }
        }
        Ok(None)
    }
}
