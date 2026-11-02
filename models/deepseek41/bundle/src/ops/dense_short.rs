//! Reuse qualified exact-row folds for short Flash TP4 prefills.
use crate::shape::Shape;
use model_deepseek41_contract::geometry::FLASH;

pub const MAX_ROWS: u32 = 64;

pub fn tail_folds_enabled(shape: &Shape) -> bool {
    shape.prefill_tail_folds && enabled(shape)
}

pub fn enabled(shape: &Shape) -> bool {
    shape.geometry == &FLASH && shape.ranks() == 4
}

/// Two exact-row folds retain separate scratch, aligned like the rank arena.
/// Full TMA tiles take no workspace. This bound includes every tail bucket.
pub(super) fn workspace(ops: &super::DenseOps, shared: bool) -> u64 {
    ops.tail_queries
        .iter()
        .filter(|(id, _)| !shared || id.starts_with("moe.shared-") || *id == "attention.kv")
        .map(|(id, _)| {
            let largest = ops
                .decode
                .iter()
                .filter(|(of, _, _)| of == id)
                .map(|(_, _, op)| op.info().workspace.div_ceil(256) * 256)
                .max()
                .unwrap_or(0);
            2 * largest
        })
        .max()
        .unwrap_or(0)
}

/// Only eager prefill uses this partition: its live count is the complete
/// step width, and each piece bounds that count by its own static rows.
/// Codes, scales and output are sliced; status keeps the step's base and
/// carries the piece's row offset, so poison stays attached to its row.
pub(super) fn project(
    ops: &super::DenseOps,
    id: &str,
    q: kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery,
    rows: u32,
    views: &super::GemmViews,
    workspace: runtime_bundle_abi::DeviceBufferSpanV1,
) -> Result<Vec<super::Lowered>, kernel_core::contract::Refusal> {
    use kernel_core::backend::Counts;
    use kernel_core::contract::Refusal;
    use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers;
    use kernels_sm120::gemm::mxfp8_prefill as tiled;
    use runtime_bundle_abi::DeviceBufferSpanV1;
    let span = |source: DeviceBufferSpanV1, offset, bytes| {
        bundle_runtime::dma::subspan(source, offset, bytes).map_err(Refusal)
    };
    let full = rows / tiled::TILE * tiled::TILE;
    let mut offset = 0;
    let mut scratch_offset = 0;
    let mut calls = Vec::new();
    while offset < rows {
        let count = if offset == 0 && full != 0 {
            full
        } else {
            (rows - offset).min(MAX_ROWS)
        };
        let op = if count >= tiled::TILE {
            &ops.prefill_gemms
                .iter()
                .find(|(of, _)| *of == id)
                .ok_or(Refusal("missing tiled prefill projection"))?
                .1
        } else {
            &ops.decode
                .iter()
                .find(|(of, capacity, _)| *of == id && *capacity == count)
                .ok_or(Refusal("missing exact-row prefill tail fold"))?
                .2
        };
        let slice = |source, stride: u64| {
            span(
                source,
                u64::from(offset) * stride,
                u64::from(count) * stride,
            )
        };
        let mut bind = super::Bind::default();
        let buffers = Mxfp8FoldGemmBuffers {
            activations: bind.view(slice(
                views.activations,
                u64::from(q.activation_row_stride),
            )?),
            activation_scales: bind.view(slice(
                views.activation_scales,
                u64::from(q.activation_row_stride) / 32,
            )?),
            weights: bind.view(views.weights),
            weight_scales: bind.view(views.weight_scales),
            output: bind.view(slice(
                views.output,
                u64::from(q.n) * u64::from(q.groups) * 4,
            )?),
            live_rows: bind.view(views.live_rows),
            status: bind.view(views.status),
            status_row_offset: u64::from(offset),
        };
        let scratch = span(workspace, scratch_offset, op.info().workspace)?;
        calls.extend(bind.launch_within(op, &buffers, scratch, Counts::rows(count))?);
        scratch_offset += op.info().workspace.div_ceil(256) * 256;
        offset += count;
    }
    Ok(calls)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{
        Asks,
        dense::{DenseOps, GemmViews, projections},
    };
    use crate::program::Phase;
    use bundle_runtime::geometry::Ranks;
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};

    #[test]
    fn threshold_keeps_unqualified_topologies_on_full_folds() {
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let shape = Shape::new(&FLASH, ranks).unwrap();
            let kernels = crate::Kernels::new(&shape).unwrap();
            let ops =
                DenseOps::prepare(kernels.backend.as_ref(), &shape, &Asks::default()).unwrap();
            assert_eq!(ops.short_prefill, ranks == Ranks::TP4);
            for d in projections(&shape) {
                for rows in 1..=65 {
                    let q = d.query;
                    let span = |id, bytes| {
                        DeviceBufferSpanV1::new(
                            BundleDeviceBufferHandleV1::from_parts(id, 1),
                            128,
                            bytes,
                        )
                    };
                    let views = GemmViews {
                        stream: None,
                        activations: span(1, u64::from(rows) * u64::from(q.activation_row_stride)),
                        activation_scales: span(
                            2,
                            u64::from(rows) * u64::from(q.activation_row_stride) / 32,
                        ),
                        weights: span(3, u64::from(q.n) * u64::from(q.k) * u64::from(q.groups)),
                        weight_scales: span(
                            4,
                            u64::from(q.n) * u64::from(q.k) * u64::from(q.groups) / 32,
                        ),
                        output: span(
                            5,
                            u64::from(rows) * u64::from(q.n) * u64::from(q.groups) * 4,
                        ),
                        live_rows: span(6, 4),
                        status: span(7, u64::from(rows) * 4),
                    };
                    let new = ops
                        .project(d.id, Phase::Prefill { rows }, &views, span(99, 1 << 30))
                        .unwrap();
                    if ranks == Ranks::TP4 && rows <= MAX_ROWS {
                        assert!(new.len() >= 2);
                        assert!(
                            new.iter()
                                .all(|c| !c.launch.symbol.contains("mxfp8_fold_m8192"))
                        );
                    } else {
                        assert_eq!(new.len(), 1);
                        assert!(new[0].launch.symbol.contains("mxfp8_fold_m8192"));
                    }
                }
            }
        }
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        assert!(enabled(&shape));
    }
    #[test]
    fn tail_partition_pins_routes_offsets_and_disjoint_workspaces() {
        use kernel_core::contract::Argument;
        use kernels_sm120::gemm::mxfp8_prefill as tiled;
        let shape = Shape::new(&FLASH, Ranks::TP4)
            .unwrap()
            .with_prefill_tail_folds(true);
        let kernels = crate::Kernels::new(&shape).unwrap();
        let ops = DenseOps::prepare(kernels.backend.as_ref(), &shape, &Asks::default()).unwrap();
        let span = |id, bytes| {
            DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(id, 1), 256, bytes)
        };
        for d in projections(&shape) {
            let q = d.query;
            for rows in [65, 85, 127, 128, 129, 191, 789, 831, 853, 1557, 1599] {
                let stride = u64::from(q.activation_row_stride);
                let output_stride = u64::from(q.n) * u64::from(q.groups) * 4;
                let views = GemmViews {
                    stream: None,
                    activations: span(1, u64::from(rows) * stride),
                    activation_scales: span(2, u64::from(rows) * stride / 32),
                    weights: span(3, u64::from(q.n) * u64::from(q.k) * u64::from(q.groups)),
                    weight_scales: span(
                        4,
                        u64::from(q.n) * u64::from(q.k) * u64::from(q.groups) / 32,
                    ),
                    output: span(5, u64::from(rows) * output_stride),
                    live_rows: span(6, 4),
                    status: span(7, u64::from(rows) * 4),
                };
                let calls = ops
                    .project(d.id, Phase::Prefill { rows }, &views, span(99, 1 << 30))
                    .unwrap();
                let full = rows / tiled::TILE * tiled::TILE;
                let mut expected = Vec::new();
                if full != 0 {
                    expected.push((0, full));
                }
                let mut offset = full;
                while offset < rows {
                    let count = (rows - offset).min(MAX_ROWS);
                    expected.push((offset, count));
                    offset += count;
                }
                let mut at = 0;
                let mut workspaces = Vec::new();
                for (offset, count) in expected {
                    let start = at;
                    while at < calls.len() && calls[at].launch.rows == count {
                        at += 1;
                    }
                    let piece = &calls[start..at];
                    assert!(!piece.is_empty(), "{} rows={rows} offset={offset}", d.id);
                    assert_eq!(piece[0].launch.rows, count);
                    assert!(!piece.iter().any(|c| c.launch.symbol.contains("split_tail")
                        || c.launch.symbol.contains("split_full")));
                    if count >= tiled::TILE {
                        assert_eq!(piece.len(), 2);
                        assert!(piece[1].launch.symbol.contains("_prefill_tma_v1"));
                    } else {
                        assert!(piece.len() >= 2);
                        assert!(
                            piece
                                .iter()
                                .all(|c| !c.launch.symbol.contains("mxfp8_fold_m8192"))
                        );
                    }
                    let mut seen = [false; 3];
                    let mut scratch = None;
                    for call in piece {
                        let mut status_offset = false;
                        for arg in &call.launch.arguments.0 {
                            if let Argument::U64(value) = arg {
                                status_offset |= *value == u64::from(offset);
                            }
                            if let Argument::DeviceBuffer(id) = arg {
                                let bound = call.launch.bindings.get(*id).unwrap();
                                for (i, (original, row_stride)) in [
                                    (views.activations, stride),
                                    (views.activation_scales, stride / 32),
                                    (views.output, output_stride),
                                ]
                                .into_iter()
                                .enumerate()
                                {
                                    if bound.buffer == original.buffer {
                                        assert_eq!(
                                            bound.offset,
                                            original.offset + u64::from(offset) * row_stride
                                        );
                                        assert_eq!(bound.length, u64::from(count) * row_stride);
                                        seen[i] = true;
                                    }
                                }
                                if bound.buffer == views.status.buffer {
                                    assert_eq!(bound, views.status);
                                    assert!(
                                        status_offset
                                            || call
                                                .launch
                                                .arguments
                                                .0
                                                .iter()
                                                .any(|a| *a == Argument::U64(u64::from(offset)))
                                    );
                                }
                                if bound.buffer == span(99, 0).buffer {
                                    scratch = Some(bound);
                                }
                            }
                        }
                    }
                    assert_eq!(seen, [true; 3]);
                    if let Some(scratch) = scratch {
                        workspaces.push(scratch);
                    }
                }
                assert_eq!(at, calls.len());
                for pair in workspaces.windows(2) {
                    assert!(pair[0].offset + pair[0].length <= pair[1].offset);
                }
            }
        }
    }
}
