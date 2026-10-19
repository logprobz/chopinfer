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
        let (layer, point) = match stage {
            Stage::Route(l) => (l, Point::Route),
            Stage::ExpertsGateUp(l) => (l, Point::Gate),
            Stage::ExpertsDown(l) => (l, Point::Down),
            Stage::RoutedSum(l) => (l, Point::Sum),
            _ => return Ok(None),
        };
        let s = &self.buffers.scratch;
        let weight = |name| self.buffers.view(&layer_view(layer, name));
        op.launch(point, self.buffers.workspace, |role| {
            Ok(match role {
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
        })
        .map(Some)
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

    #[test]
    fn direct_stage_audit_preserves_router_shared_and_join_order() {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let kernels = crate::Kernels::new(&shape).unwrap();
        for set in moe::expert_sets(&FLASH) {
            let ops = Ops::prepare_set(&kernels, &shape, KvStorage::Native, set);
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
                for rows in [1, 2, 4, 6, 8, 16, 32, 48, 64, 192] {
                    let phase = if rows == 6 || rows > 32 {
                        Phase::Verify {
                            rows,
                            items: rows.div_ceil(6),
                        }
                    } else {
                        Phase::Decode(rows)
                    };
                    let selected = ops.moe.as_ref().unwrap().direct(phase).is_some();
                    let lower = |op: &Ops, buffers: &crate::buffers::RankBuffers| {
                        let peers = [buffers.clone()];
                        let lowering = Lowering {
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
                        };
                        crate::program::moe(0)
                            .into_iter()
                            .map(|stage| (stage, lowering.stage(phase, stage).unwrap().unwrap()))
                            .collect::<Vec<_>>()
                    };
                    let actual = lower(&ops, &b);
                    let old = lower(&reference, &b);
                    for ((stage, new), (_, old)) in actual.iter().zip(&old) {
                        if !selected
                            || !matches!(
                                stage,
                                Stage::Route(_)
                                    | Stage::ExpertsGateUp(_)
                                    | Stage::ExpertsDown(_)
                                    | Stage::RoutedSum(_)
                            )
                        {
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
                    crate::shared_overlap::checked_units(&stages, &launches, &readonly).unwrap();
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
                        let bad = lower(&ops, &bad)
                            .into_iter()
                            .flat_map(|(_, ops)| ops.into_iter().map(|op| op.launch))
                            .collect::<Vec<_>>();
                        assert!(
                            crate::shared_overlap::checked_units(&stages, &bad, &readonly)
                                .unwrap_err()
                                .contains("unordered scratch alias")
                        );
                    }
                }
            }
        }
    }
}
