//! Bind qualified direct MoE buckets at the existing fork/join boundaries.
use super::{Lowering, layer_view};
use crate::ops::{Lowered, moe::direct::Point};
use crate::program::{Phase, Stage};

impl Lowering<'_> {
    pub(super) fn moe_direct_stage(
        &self,
        phase: Phase,
        stage: Stage,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let Some(op) = self.ops.moe.as_ref().and_then(|m| m.direct(phase)) else {
            return Ok(None);
        };
        let (layer, point, partial) = match stage {
            Stage::Project(layer, "moe.router") => {
                let projected = self.exact_f32(phase, layer, "moe.router")?;
                if !op.fuses_router() {
                    return Ok(projected);
                }
                let mut projected =
                    projected.ok_or("fused router requires exact F32 projection")?;
                let suffixes = ["_cached_rows_v1", "_cached_decode_v1", "_cached_reduce_v1"];
                if projected.len() != suffixes.len()
                    || !projected
                        .iter()
                        .zip(suffixes)
                        .all(|(p, suffix)| p.launch.symbol.ends_with(suffix))
                {
                    return Err(
                        "fused router requires the qualified cached partial producer".into(),
                    );
                }
                // Keep the original producer and all of its bindings. The
                // route stage now performs its ordered reduction.
                (layer, Point::RouterPrepare, Some(projected.remove(1)))
            }
            Stage::Route(l) => (l, Point::Route, None),
            Stage::ExpertsGateUp(l) => (l, Point::Gate, None),
            Stage::ExpertsDown(l) => (l, Point::Down, None),
            Stage::RoutedSum(l) => (l, Point::Sum, None),
            _ => return Ok(None),
        };
        let s = &self.buffers.scratch;
        let weight = |name| self.buffers.view(&layer_view(layer, name));
        let mut calls = op.launch(point, self.buffers.workspace, |role| {
            Ok(match role {
                "wide" => s.normed_wide,
                "logits" => s.router_logits,
                "bias" => weight("ffn.gate.bias")?,
                "alternate_bias" => weight("ffn.gate.bias_vl")?,
                "bias_select" => s.bias_select,
                "scores" => s.router_scores,
                "ids" => s.expert_ids,
                "weights" => s.expert_weights,
                "counts" => s.route_counts,
                "offsets" => s.route_offsets,
                "sorted" => self.buffers.sorted_rows,
                "padded" => s.padded_rows,
                "active" => s.active_experts,
                "live" => self.inputs.live_tokens,
                "input_codes" => s.normed_codes,
                "input_scales" => s.normed_scales,
                "gate_weight" => weight("ffn.experts.gate_up.weight")?,
                "gate_scales" => weight("ffn.experts.gate_up.scale")?,
                "down_weight" => weight("ffn.experts.down.weight")?,
                "down_scales" => weight("ffn.experts.down.scale")?,
                "intermediate" => s.expert_intermediate,
                "codes" => s.intermediate_codes,
                "scales" => s.intermediate_scales,
                "status" => s.row_status,
                "routed" => s.routed,
                "shared" => s.shared_partial,
                "output" => s.partial,
                _ => return Err(format!("unbound MoE pipeline role {role}")),
            })
        })?;
        calls.extend(partial);
        Ok(Some(calls))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::{KvExtent, tests::buffers};
    use crate::{ops::Ops, shape::Shape};
    use bundle_runtime::state::Extent;
    use bundle_runtime::{
        geometry::{Items, Ranks},
        upload::{DeviceInputs, InputLayout, PhaseArena},
    };
    use kernel_core::contract::Argument;
    use model_deepseek41_contract::{geometry::FLASH, moe, state::KvStorage};
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};

    fn same(
        a: &bundle_runtime::pipeline::PreparedLaunch,
        b: &bundle_runtime::pipeline::PreparedLaunch,
    ) {
        assert_eq!(a.symbol, b.symbol);
        assert_eq!(a.plan, b.plan);
        assert_eq!(a.rows, b.rows);
        let arguments = |op: &bundle_runtime::pipeline::PreparedLaunch| {
            op.arguments
                .0
                .iter()
                .map(|arg| {
                    if let Argument::DeviceBuffer(h) = arg {
                        format!("{:?}", op.bindings.get(*h).unwrap())
                    } else {
                        format!("{arg:?}")
                    }
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(arguments(a), arguments(b));
    }

    fn router_audit(
        actual: &[(Stage, Vec<Lowered>)],
        plain: &[(Stage, Vec<Lowered>)],
        workspace: DeviceBufferSpanV1,
        layout: Option<crate::sm120::moe_router_finish::Layout>,
    ) {
        let args = |op: &bundle_runtime::pipeline::PreparedLaunch, shift: &[usize]| {
            op.arguments
                .0
                .iter()
                .enumerate()
                .map(|(i, arg)| {
                    if let Argument::DeviceBuffer(h) = arg {
                        let mut span = op.bindings.get(*h).unwrap();
                        if shift.contains(&i) {
                            assert_eq!(span.buffer, workspace.buffer);
                            span.offset = span.offset.checked_sub(layout.unwrap().routed).unwrap();
                        }
                        format!("{span:?}")
                    } else {
                        format!("{arg:?}")
                    }
                })
                .collect::<Vec<_>>()
        };
        for ((stage, new), (old_stage, old)) in actual.iter().zip(plain) {
            assert_eq!(stage, old_stage);
            let Some(layout) = layout else {
                assert_eq!(new.len(), old.len());
                for (a, b) in new.iter().zip(old) {
                    same(&a.launch, &b.launch);
                }
                continue;
            };
            match stage {
                Stage::Project(_, "moe.router") => {
                    assert_eq!((new.len(), old.len()), (2, 3));
                    same(&new[1].launch, &old[1].launch);
                    assert!(new[0].launch.symbol.ends_with("_prepare_v1"));
                    let a = args(&new[0].launch, &[]);
                    let b = args(&old[0].launch, &[]);
                    assert_eq!(&a[..3], &b[..3]);
                    let span = |i| {
                        let Argument::DeviceBuffer(h) = new[0].launch.arguments.0[i] else {
                            panic!("expected span")
                        };
                        new[0].launch.bindings.get(h).unwrap()
                    };
                    assert_eq!(
                        span(3),
                        DeviceBufferSpanV1::new(
                            workspace.buffer,
                            workspace.offset,
                            layout.partial_bytes
                        )
                    );
                    assert_eq!(
                        span(4),
                        DeviceBufferSpanV1::new(
                            workspace.buffer,
                            workspace.offset + layout.arrival,
                            4
                        )
                    );
                    assert!(
                        layout.partial_bytes <= layout.arrival
                            && layout.arrival + 4 <= layout.routed
                    );
                }
                Stage::Route(_) => {
                    assert_eq!((new.len(), old.len()), (1, 2));
                    let a = args(&new[0].launch, &[16, 17]);
                    let select = args(&old[0].launch, &[]);
                    let meta = args(&old[1].launch, &[]);
                    let projected = &plain
                        .iter()
                        .find(|(s, _)| matches!(s, Stage::Project(_, "moe.router")))
                        .unwrap()
                        .1;
                    let reduce = args(&projected[2].launch, &[]);
                    // A drafter fixture may reserve a backbone-sized
                    // logits plane. The fused writer binds the same exact
                    // prefix as its route reader, within the old output.
                    assert_eq!(a[1], select[0]);
                    assert_eq!(&a[2..4], &reduce[2..4]);
                    let bound = |op: &bundle_runtime::pipeline::PreparedLaunch, index| {
                        let Argument::DeviceBuffer(h) = op.arguments.0[index] else {
                            panic!("expected span")
                        };
                        op.bindings.get(h).unwrap()
                    };
                    let written = bound(&new[0].launch, 1);
                    let prior = bound(&projected[2].launch, 1);
                    assert_eq!(
                        (written.buffer, written.offset),
                        (prior.buffer, prior.offset)
                    );
                    assert!(written.length <= prior.length);
                    assert_eq!(&a[4..10], &select[1..7]);
                    assert_eq!(a[10], select[8]);
                    assert_eq!(&a[11..15], &meta[1..5]);
                    assert_eq!(&a[15..18], &meta[6..9]);
                    let prepared = &actual
                        .iter()
                        .find(|(s, _)| matches!(s, Stage::Project(_, "moe.router")))
                        .unwrap()
                        .1;
                    let prepare = args(&prepared[0].launch, &[]);
                    assert_eq!(a[0], prepare[3]);
                    assert_eq!(a[18], prepare[4]);
                    assert_eq!(new[0].launch.plan.grid, [new[0].launch.rows, 1, 1]);
                }
                Stage::ExpertsGateUp(_) | Stage::ExpertsDown(_) | Stage::RoutedSum(_) => {
                    assert_eq!(new.len(), old.len());
                    for (a, b) in new.iter().zip(old) {
                        assert_eq!(a.launch.symbol, b.launch.symbol);
                        assert_eq!(a.launch.plan, b.launch.plan);
                        let moved = a
                            .launch
                            .arguments
                            .0
                            .iter()
                            .enumerate()
                            .filter_map(|(i, arg)| {
                                let Argument::DeviceBuffer(h) = arg else {
                                    return None;
                                };
                                (a.launch.bindings.get(*h).unwrap().buffer == workspace.buffer)
                                    .then_some(i)
                            })
                            .collect::<Vec<_>>();
                        assert_eq!(args(&a.launch, &moved), args(&b.launch, &[]));
                    }
                }
                _ => {
                    assert_eq!(new.len(), old.len());
                    for (a, b) in new.iter().zip(old) {
                        same(&a.launch, &b.launch);
                    }
                }
            }
        }
    }

    #[test]
    fn direct_stage_audit_preserves_router_shared_and_join_order() {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let kernels = crate::Kernels::new(&shape).unwrap();
        for set in moe::expert_sets(&FLASH) {
            let ops = Ops::prepare_set(&kernels, &shape, KvStorage::Native, set);
            let unfused = crate::sm120::moe_router_finish::without_fusion(|| {
                Ops::prepare_set(&kernels, &shape, KvStorage::Native, set)
            });
            assert_eq!(ops.workspace(), unfused.workspace());
            let mut reference = Ops::prepare_set(&kernels, &shape, KvStorage::Native, set);
            reference.moe.as_mut().unwrap().retain_reference_in_test();
            for rank in 0..4 {
                let mut b = buffers(&shape, rank);
                b.workspace.length = ops.workspace();
                b.shared_workspace.length = ops.shared_workspace();
                b.sorted_rows.length = ops.moe.as_ref().unwrap().sorted_rows_bytes();
                for (n, name) in [
                    "ffn.gate.weight",
                    "ffn.gate.weight.f32_domain",
                    "ffn.gate.bias",
                    "ffn.gate.bias_vl",
                    "ffn.experts.gate_up.weight",
                    "ffn.experts.gate_up.scale",
                    "ffn.experts.down.weight",
                    "ffn.experts.down.scale",
                    "ffn.shared_experts.w1.weight",
                    "ffn.shared_experts.w1.scale",
                    "ffn.shared_experts.w3.weight",
                    "ffn.shared_experts.w3.scale",
                    "ffn.shared_experts.w2.weight",
                    "ffn.shared_experts.w2.scale",
                ]
                .iter()
                .enumerate()
                {
                    b.views.insert(
                        layer_view(0, name),
                        DeviceBufferSpanV1::new(
                            BundleDeviceBufferHandleV1::from_parts(10000 + n as u32, rank + 1),
                            0,
                            1 << 32,
                        ),
                    );
                }
                // Optional packed folds bind these private views in the
                // integrated selector; unselected folds leave them unused.
                let mut storage_id = 11000;
                for tensor in ["w1", "w2", "w3"] {
                    for kind in ["single", "pair"] {
                        for role in ["packed0", "packed1", "state", "workspace"] {
                            storage_id += 1;
                            b.views.insert(
                                format!(
                                    "{}.fold_{kind}.{role}",
                                    layer_view(0, &format!("ffn.shared_experts.{tensor}.weight"))
                                ),
                                DeviceBufferSpanV1::new(
                                    BundleDeviceBufferHandleV1::from_parts(storage_id, rank + 1),
                                    0,
                                    1 << 32,
                                ),
                            );
                        }
                    }
                }
                let live = DeviceBufferSpanV1::new(
                    BundleDeviceBufferHandleV1::from_parts(999, rank + 1),
                    0,
                    4,
                );
                let inputs = DeviceInputs {
                    live_tokens: live,
                    ..DeviceInputs::within(
                        PhaseArena {
                            arena: live,
                            block_tables: live,
                        },
                        &InputLayout::decode_within(Items::new(1).unwrap()),
                    )
                };
                let phases = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32]
                    .map(Phase::Decode)
                    .into_iter()
                    .chain([1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 48, 64, 192].map(|rows| {
                        Phase::Verify {
                            rows,
                            items: rows.div_ceil(6),
                        }
                    }));
                for phase in phases {
                    let rows = match phase {
                        Phase::Decode(rows) | Phase::Verify { rows, .. } => rows,
                        _ => unreachable!(),
                    };
                    let selected = ops.moe.as_ref().unwrap().direct(phase).is_some();
                    let fused = ops
                        .moe
                        .as_ref()
                        .unwrap()
                        .direct(phase)
                        .is_some_and(|p| p.fuses_router());
                    let lower = |op: &Ops, buffers: &crate::buffers::RankBuffers| {
                        let peers = [buffers.clone()];
                        let lowering = Lowering {
                            prefill_row_origin: None,
                            shape: &shape,
                            ops: op,
                            buffers,
                            peers: &peers,
                            inputs: &inputs,
                            rank,
                            kv: KvExtent {
                                storage: KvStorage::Native,
                                pages: 1,
                                extent: Extent {
                                    served: 1,
                                    bucket: rows.min(32),
                                    verify_rows: rows,
                                },
                                sequences: 1,
                                max_pages: 1,
                            },
                            head_rows: 1,
                            prefill_hc_fusion: true,
                        };
                        crate::program::moe(0)
                            .into_iter()
                            .map(|stage| {
                                let calls = if std::ptr::eq(op, &reference)
                                    && matches!(stage, Stage::Project(_, "moe.router"))
                                {
                                    lowering.exact_f32(phase, 0, "moe.router")
                                } else {
                                    lowering.stage(phase, stage)
                                };
                                (stage, calls.unwrap().unwrap())
                            })
                            .collect::<Vec<_>>()
                    };
                    let actual = lower(&ops, &b);
                    let plain = lower(&unfused, &b);
                    let layout = fused.then(|| {
                        let entry = crate::sm120::moe_direct_kernels::entries(&FLASH)
                            .unwrap()
                            .into_iter()
                            .find(|e| e.experts() == set.count)
                            .unwrap();
                        crate::sm120::moe_router_finish::layout(&entry, rows)
                    });
                    router_audit(&actual, &plain, b.workspace, layout);
                    let old = lower(&reference, &b);
                    for ((stage, new), (_, old)) in actual.iter().zip(&old) {
                        let replaced = if matches!(stage, Stage::Project(_, "moe.router")) {
                            fused
                        } else {
                            selected
                                && matches!(
                                    stage,
                                    Stage::Route(_)
                                        | Stage::ExpertsGateUp(_)
                                        | Stage::ExpertsDown(_)
                                        | Stage::RoutedSum(_)
                                )
                        };
                        if !replaced {
                            assert_eq!(new.len(), old.len());
                            for (a, b) in new.iter().zip(old) {
                                same(&a.launch, &b.launch);
                            }
                        }
                    }
                    let stages = actual
                        .iter()
                        .flat_map(|(s, ops)| std::iter::repeat_n(*s, ops.len()))
                        .collect::<Vec<_>>();
                    let launches = actual
                        .into_iter()
                        .flat_map(|(_, ops)| ops.into_iter().map(|op| op.launch))
                        .collect::<Vec<_>>();
                    let readonly = [b.scratch.normed_codes, b.scratch.normed_scales, live];
                    crate::shared_overlap::checked_units(&stages, &launches, &readonly, false)
                        .unwrap();
                    if selected {
                        assert!(launches.last().unwrap().symbol.ends_with("_down_finish_v1"));
                        assert!(
                            launches
                                .iter()
                                .any(|p| p.symbol.contains("bandwidth_direct"))
                        );
                        assert!(
                            !launches
                                .iter()
                                .any(|p| p.symbol.contains("bandwidth_prefetch"))
                        );
                        let mut bad = b.clone();
                        bad.shared_workspace = bad.workspace;
                        for (name, span) in &mut bad.views {
                            if name.contains("ffn.shared_experts.")
                                && (name.ends_with(".state") || name.ends_with(".workspace"))
                            {
                                *span = bad.workspace;
                            }
                        }
                        let bad = lower(&ops, &bad)
                            .into_iter()
                            .flat_map(|(_, ops)| ops.into_iter().map(|op| op.launch))
                            .collect::<Vec<_>>();
                        assert!(
                            crate::shared_overlap::checked_units(&stages, &bad, &readonly, false)
                                .unwrap_err()
                                .contains("unordered scratch alias")
                        );
                    }
                }
            }
        }
    }
}
