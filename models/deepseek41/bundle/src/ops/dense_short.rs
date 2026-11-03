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
/// Prefill relocates only shared-expert projections. Window KV uses the main
/// workspace here; its shared relocation applies only to decode and verify.
pub(super) fn workspace(ops: &super::DenseOps, shared: bool) -> u64 {
    ops.tail_queries
        .iter()
        .filter(|(id, _)| !shared || id.starts_with("moe.shared-"))
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
#[allow(clippy::too_many_arguments)]
pub(super) fn project(
    ops: &super::DenseOps,
    id: &str,
    q: kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery,
    rows: u32,
    views: &super::GemmViews,
    workspace: runtime_bundle_abi::DeviceBufferSpanV1,
    mut calls: Vec<super::Lowered>,
    (full_rows, full_launches, padded): (u32, usize, bool),
) -> Result<Vec<super::Lowered>, kernel_core::contract::Refusal> {
    use kernel_core::backend::Counts;
    use kernel_core::contract::Refusal;
    use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers;
    use runtime_bundle_abi::DeviceBufferSpanV1;
    let span = |source: DeviceBufferSpanV1, offset, bytes| {
        bundle_runtime::dma::subspan(source, offset, bytes).map_err(Refusal)
    };
    // Keep the baseline admission and full launch, including its bindings,
    // runtime rows and split_full symbol. For padded TMA, remove only the last
    // grid tile; the other CTAs keep their exact launch arguments and tile.
    calls.truncate(full_launches);
    if padded {
        calls[1].launch.plan.grid[1] = full_rows / crate::Kernels::dense_prefill_tile();
    }
    let mut offset = full_rows;
    let mut scratch_offset = 0;
    while offset < rows {
        let count = (rows - offset).min(MAX_ROWS);
        let op = &ops
            .decode
            .iter()
            .find(|(of, capacity, _)| *of == id && *capacity == count)
            .ok_or(Refusal("missing exact-row prefill tail fold"))?
            .2;
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
                u64::from(q.n)
                    * u64::from(q.groups)
                    * if q.output == kernel_core::contract::DType::F32 {
                        4
                    } else {
                        2
                    },
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
                    if ranks == Ranks::TP4 {
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
    fn tail_folds_preserve_aligned_and_tabled_prefill_routes() {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let baseline = crate::Kernels::new(&shape.with_prefill_tail_folds(false)).unwrap();
        let candidate = crate::Kernels::new(&shape.with_prefill_tail_folds(true)).unwrap();
        use kernel_core::backend::{Counts, Need, View};
        use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers;
        for dense in projections(&shape) {
            let q = dense.query;
            let old = baseline
                .backend
                .mxfp8_prefill_gemm(&q, Need::rows(q.m_capacity))
                .unwrap();
            let new = candidate
                .backend
                .mxfp8_prefill_gemm(&q, Need::rows(q.m_capacity))
                .unwrap();
            let v = |handle| View {
                handle,
                bytes: 1 << 34,
            };
            let b = Mxfp8FoldGemmBuffers {
                activations: v(1),
                activation_scales: v(2),
                weights: v(3),
                weight_scales: v(4),
                output: v(5),
                live_rows: v(6),
                status: v(7),
                status_row_offset: 0,
            };
            for rows in [128, 192, 448, 512, 960, 1472, 2048, 2496, 4096, 8192] {
                assert_eq!(
                    new.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap(),
                    old.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap(),
                    "{} rows={rows}",
                    dense.id,
                );
            }
        }
    }
    #[test]
    fn tail_folds_keep_other_topologies_and_tail_policy_bounds() {
        use kernel_core::backend::{Counts, Need, View};
        use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers;
        for ranks in [Ranks::TP1, Ranks::TP2] {
            let shape = Shape::new(&FLASH, ranks).unwrap();
            let old = crate::Kernels::new(&shape.with_prefill_tail_folds(false)).unwrap();
            let new = crate::Kernels::new(&shape.with_prefill_tail_folds(true)).unwrap();
            assert!(!tail_folds_enabled(&shape.with_prefill_tail_folds(true)));
            for d in projections(&shape) {
                let q = d.query;
                let old = old
                    .backend
                    .mxfp8_prefill_gemm(&q, Need::rows(q.m_capacity))
                    .unwrap();
                let new = new
                    .backend
                    .mxfp8_prefill_gemm(&q, Need::rows(q.m_capacity))
                    .unwrap();
                assert_eq!(old.info(), new.info());
                let v = |handle| View {
                    handle,
                    bytes: 1 << 34,
                };
                let b = Mxfp8FoldGemmBuffers {
                    activations: v(1),
                    activation_scales: v(2),
                    weights: v(3),
                    weight_scales: v(4),
                    output: v(5),
                    live_rows: v(6),
                    status: v(7),
                    status_row_offset: 3,
                };
                for rows in [129, 144, 145, 255, 1023, 1025, 1151] {
                    let baseline = old.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap();
                    assert_eq!(
                        baseline,
                        new.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap()
                    );
                    let policy = crate::sm120::mxfp8_fold::prefill_tail_policy(q);
                    let split = policy.splits(rows);
                    assert_eq!(baseline.len(), if split { 3 } else { 2 });
                    let partition = crate::sm120::mxfp8_fold::prefill_tail_partition(
                        rows,
                        &baseline.last().unwrap().symbol,
                    )
                    .expect("both split and padded TMA have remainders");
                    assert_eq!(partition.padded, !split);
                    if policy.max_tail_rows == 16 {
                        assert_eq!(split, rows % 128 <= 16);
                    }
                }
            }
        }
    }

    #[test]
    fn tail_fold_direct_allowances_cover_every_selected_remainder() {
        use kernel_core::route::{Metadata, RouteClass};
        let _routes = Metadata::enable();
        let allow: Vec<serde_json::Value> =
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/route-allowlist.toml",))
                .split("[[allow]]")
                .skip(1)
                .map(|block| {
                    serde_json::Value::Object(
                        block
                            .lines()
                            .filter_map(|line| {
                                if line.trim().starts_with('#') {
                                    return None;
                                }
                                let (key, value) = line.split_once(" = ")?;
                                Some((key.into(), serde_json::from_str(value).unwrap()))
                            })
                            .collect(),
                    )
                })
                .filter(|entry| entry["task"] == "PREFILL-tail-folds-direct")
                .collect();
        assert!(!allow.is_empty());
        let shape = Shape::new(&FLASH, Ranks::TP4)
            .unwrap()
            .with_prefill_tail_folds(true);
        let kernels = crate::Kernels::new(&shape).unwrap();
        let ops = DenseOps::prepare(kernels.backend.as_ref(), &shape, &Asks::default()).unwrap();
        let span = |id, bytes| {
            DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(id, 1), 256, bytes)
        };
        let views = GemmViews {
            stream: None,
            activations: span(1, 1 << 34),
            activation_scales: span(2, 1 << 34),
            weights: span(3, 1 << 34),
            weight_scales: span(4, 1 << 34),
            output: span(5, 1 << 34),
            live_rows: span(6, 4),
            status: span(7, 8192 * 4),
        };
        let mut covered = std::collections::BTreeSet::new();
        for d in projections(&shape) {
            for rows in 65..=8192 {
                let calls = ops
                    .project(
                        d.id,
                        Phase::Prefill { rows },
                        &views,
                        span(99, ops.tail_workspace()),
                    )
                    .unwrap();
                for call in calls.iter().filter(|call| call.launch.rows <= MAX_ROWS) {
                    let tag = call.launch.bindings.route().expect("selected tail route");
                    if tag.class == RouteClass::Fast {
                        continue;
                    }
                    assert!(
                        allow.iter().any(|a| {
                            a["entry"] == tag.entry
                                && a["symbol"] == call.launch.symbol
                                && a["phase"] == "prefill"
                                && a["family"] == tag.family
                                && ["tail-folds-tp4-ar", "tail-folds-tp4-dspark"].iter().all(
                                    |profile| {
                                        a["profiles"]
                                            .as_array()
                                            .unwrap()
                                            .contains(&(*profile).into())
                                    },
                                )
                                && a["rows"].as_array().unwrap().iter().any(|range| {
                                    range[0].as_u64().unwrap() <= u64::from(rows)
                                        && u64::from(rows) <= range[1].as_u64().unwrap()
                                })
                        }),
                        "missing independent tail allowance: {} rows={rows} {}",
                        d.id,
                        tag.entry
                    );
                    covered.insert(tag.entry.clone());
                }
            }
        }
        assert_eq!(
            covered.len(),
            allow.len(),
            "every direct allowance must serve a tail"
        );
    }

    fn same_launch(old: &super::super::Lowered, new: &super::super::Lowered, padded: bool) {
        use kernel_core::contract::Argument;
        let (a, b) = (&old.launch, &new.launch);
        assert_eq!(a.symbol, b.symbol);
        assert_eq!(a.rows, b.rows);
        assert_eq!(old.collective, new.collective);
        let mut plan = a.plan;
        if padded {
            plan.grid[1] -= 1;
        }
        assert_eq!(plan, b.plan);
        assert_eq!(a.arguments, b.arguments);
        assert_eq!(a.bindings.route(), b.bindings.route());
        for arg in &a.arguments.0 {
            let id = match arg {
                Argument::DeviceBuffer(id) => Some(*id),
                Argument::TensorMap(map) => Some(map.base_buffer),
                _ => None,
            };
            if let Some(id) = id {
                assert_eq!(a.bindings.get(id).unwrap(), b.bindings.get(id).unwrap());
            }
        }
    }

    #[test]
    fn tail_partition_pins_routes_offsets_and_disjoint_workspaces() {
        use kernel_core::contract::Argument;
        const TILE: u32 = 128;
        let _routes = kernel_core::route::Metadata::enable();
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let prepare = |shape: &Shape| {
            let kernels = crate::Kernels::new(shape).unwrap();
            let ops = crate::ops::Ops::prepare_set(
                &kernels,
                shape,
                model_deepseek41_contract::state::KvStorage::Native,
                model_deepseek41_contract::moe::Experts {
                    count: FLASH.shape.moe.experts,
                    top_k: FLASH.shape.moe.top_k,
                },
            );
            assert!(ops.missing.is_empty(), "{:?}", ops.missing);
            let workspace = ops.workspace();
            let shared_workspace = ops.shared_workspace();
            (ops.dense.unwrap(), ops.asked, workspace, shared_workspace)
        };
        let (old, old_asks, old_workspace, _) = prepare(&shape.with_prefill_tail_folds(false));
        let (new, new_asks, workspace, shared_workspace) =
            prepare(&shape.with_prefill_tail_folds(true));
        assert_eq!(
            old_asks, new_asks,
            "the knob must preserve registrations and asks"
        );
        assert_eq!(
            old.tail_workspace(),
            0,
            "option-off workspace stays unchanged"
        );
        let span = |id, bytes| {
            DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(id, 1), 256, bytes)
        };
        let widths: std::collections::BTreeSet<_> = (1..=300)
            .chain((128..=8192).step_by(128))
            .chain((64..=8192).step_by(128))
            .chain(
                model_deepseek41_contract::tuning::table(
                    &FLASH,
                    model_deepseek41_contract::tuning::Family::DensePrefill,
                )
                .unwrap()
                .rows()
                .iter()
                .flat_map(|row| {
                    let [first, last] = row.key.rows.unwrap_or([1, row.key.bucket]);
                    first..=last
                }),
            )
            .chain([789, 831, 853, 1557, 1599, 2069, 8191])
            .collect();
        for d in projections(&shape) {
            let q = d.query;
            let workspace = if d.id.starts_with("moe.shared-") {
                shared_workspace
            } else {
                workspace
            };
            for &rows in &widths {
                let stride = u64::from(q.activation_row_stride);
                let output_stride = u64::from(q.n)
                    * u64::from(q.groups)
                    * if q.output == kernel_core::contract::DType::F32 {
                        4
                    } else {
                        2
                    };
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
                let baseline = old
                    .project(
                        d.id,
                        Phase::Prefill { rows },
                        &views,
                        span(99, old_workspace),
                    )
                    .unwrap();
                let calls = new
                    .project(d.id, Phase::Prefill { rows }, &views, span(99, workspace))
                    .unwrap();
                let measured = crate::sm120::prefill_matrix::choice(q, rows).is_some()
                    || crate::sm120::mxfp8_fold::compact::stages(&FLASH, q, rows).is_some()
                    || crate::sm120::prefill_tma_scales::choice(&FLASH, 4, q, rows);
                let changed = rows > MAX_ROWS && rows % TILE != 0 && !measured;
                if !changed {
                    assert_eq!(baseline.len(), calls.len(), "{} rows={rows}", d.id);
                    for (a, b) in baseline.iter().zip(&calls) {
                        same_launch(a, b, false);
                    }
                    continue;
                }
                let full = rows / TILE * TILE;
                let mut at = if full == 0 { 0 } else { 2 };
                if full > 0 {
                    let split = crate::sm120::mxfp8_fold::prefill_tail_policy(q).splits(rows);
                    assert_eq!(baseline.len(), if split { 3 } else { 2 });
                    same_launch(&baseline[0], &calls[0], false);
                    same_launch(&baseline[1], &calls[1], !split);
                    assert_eq!(calls[1].launch.plan.grid[1], full / TILE);
                }
                let mut offset = full;
                let mut scratch_offset = 0;
                let mut workspaces = Vec::new();
                while offset < rows {
                    let count = (rows - offset).min(MAX_ROWS);
                    let start = at;
                    while at < calls.len() && calls[at].launch.rows == count {
                        at += 1;
                    }
                    let piece = &calls[start..at];
                    assert!(!piece.is_empty(), "{} rows={rows} offset={offset}", d.id);
                    let slice = |source, stride| {
                        bundle_runtime::dma::subspan(
                            source,
                            u64::from(offset) * stride,
                            u64::from(count) * stride,
                        )
                        .unwrap()
                    };
                    let tail_views = GemmViews {
                        stream: None,
                        activations: slice(views.activations, stride),
                        activation_scales: slice(views.activation_scales, stride / 32),
                        weights: views.weights,
                        weight_scales: views.weight_scales,
                        output: slice(views.output, output_stride),
                        live_rows: views.live_rows,
                        status: views.status,
                    };
                    let scratch_view = bundle_runtime::dma::subspan(
                        span(99, workspace),
                        scratch_offset,
                        workspace - scratch_offset,
                    )
                    .unwrap();
                    let mut expected = old
                        .project(
                            d.id,
                            Phase::Prefill { rows: count },
                            &tail_views,
                            scratch_view,
                        )
                        .unwrap();
                    // Option-off project uses local status row zero. Translate
                    // only that coordinate; every symbol, plan, other argument,
                    // route and buffer binding must equal the exact-row route.
                    for call in &mut expected {
                        let args = &mut call.launch.arguments.0;
                        for i in 0..args.len() {
                            if let Argument::DeviceBuffer(handle) = args[i]
                                && call.launch.bindings.get(handle).unwrap() == views.status
                            {
                                assert_eq!(args[i + 1], Argument::U64(0));
                                args[i + 1] = Argument::U64(u64::from(offset));
                            }
                        }
                    }
                    assert_eq!(
                        piece.len(),
                        expected.len(),
                        "{} rows={rows} offset={offset}",
                        d.id
                    );
                    for (a, b) in expected.iter().zip(piece) {
                        same_launch(a, b, false);
                    }
                    let mut seen = [false; 3];
                    let mut status_seen = false;
                    let mut scratch = None;
                    for call in piece {
                        assert!(!call.launch.symbol.contains("mxfp8_fold_m8192"));
                        assert!(call.launch.bindings.route().is_some());
                        for (index, arg) in call.launch.arguments.0.iter().enumerate() {
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
                                    // Check the status offset argument itself, after the status
                                    // pointer, including the second fold and nonzero span origin.
                                    assert_eq!(
                                        call.launch.arguments.0[index + 1],
                                        Argument::U64(u64::from(offset))
                                    );
                                    status_seen = true;
                                }
                                if bound.buffer == span(99, 0).buffer {
                                    assert!(
                                        bound.offset >= 256
                                            && bound.offset + bound.length <= 256 + workspace
                                    );
                                    assert_eq!(bound.offset % 256, 0);
                                    scratch = Some(bound);
                                }
                            }
                        }
                    }
                    assert_eq!(seen, [true; 3]);
                    assert!(status_seen);
                    let scratch = scratch.expect("fold workspace");
                    scratch_offset += scratch.length.div_ceil(256) * 256;
                    workspaces.push(scratch);
                    offset += count;
                }
                assert_eq!(at, calls.len());
                for pair in workspaces.windows(2) {
                    assert!(pair[0].offset + pair[0].length <= pair[1].offset);
                }
            }
        }
    }
}
