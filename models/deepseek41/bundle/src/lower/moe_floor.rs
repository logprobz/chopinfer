//! Bind the measured one-row pipeline without changing prefill row origins.
use super::{Lowering, layer_view};
use crate::ops::{Lowered, moe::floor::Point};
use crate::program::{Phase, Stage};

impl Lowering<'_> {
    pub(super) fn moe_floor_stage(
        &self,
        phase: Phase,
        stage: Stage,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let Some(op) = self.ops.moe.as_ref().and_then(|m| m.floor(phase)) else {
            return Ok(None);
        };
        let (layer, point) = match stage {
            Stage::Project(l, "moe.router") => (l, Point::Router),
            Stage::Route(l) => (l, Point::Metadata),
            Stage::ExpertsGateUp(l) => (l, Point::Gate),
            Stage::ExpertsDown(l) => (l, Point::Down),
            Stage::RoutedSum(l) => (l, Point::Sum),
            _ => return Ok(None),
        };
        let s = &self.buffers.scratch;
        let weight = |name| self.buffers.view(&layer_view(layer, name));
        op.launch(point, self.buffers.workspace, |role| {
            Ok(match role {
                "wide" => s.normed_wide,
                "router_weight" => weight("ffn.gate.weight")?,
                "domain" => self
                    .buffers
                    .view(&crate::ops::exact_f32::domain_name(&layer_view(
                        layer,
                        "ffn.gate.weight",
                    )))?,
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
    use model_deepseek41_contract::{geometry::FLASH, moe::Experts, state::KvStorage};
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};

    #[test]
    fn measured_floor_lowering_keeps_the_shared_workspace_disjoint() {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let kernels = crate::Kernels::new(&shape).unwrap();
        let ops = Ops::prepare_set(
            &kernels,
            &shape,
            KvStorage::Native,
            Experts {
                count: 128,
                top_k: 3,
            },
        );
        assert!(ops.moe.as_ref().unwrap().floor(Phase::Decode(1)).is_some());
        for rank in 0..4 {
            let mut b = buffers(&shape, rank);
            b.workspace.length = ops.workspace();
            b.shared_workspace.length = ops.shared_workspace();
            b.sorted_rows.length = ops.moe.as_ref().unwrap().sorted_rows_bytes();
            let mut next = 10000;
            for name in [
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
            ] {
                next += 1;
                b.views.insert(
                    layer_view(0, name),
                    DeviceBufferSpanV1::new(
                        BundleDeviceBufferHandleV1::from_parts(next, rank + 1),
                        0,
                        1 << 30,
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
            let lower = |b: &crate::buffers::RankBuffers| {
                let peers = [b.clone()];
                let lowering = Lowering {
                    shape: &shape,
                    ops: &ops,
                    buffers: b,
                    peers: &peers,
                    inputs: &inputs,
                    rank,
                    kv: KvExtent {
                        storage: KvStorage::Native,
                        pages: 1,
                        extent: Extent {
                            served: 1,
                            bucket: 1,
                            verify_rows: 0,
                        },
                        sequences: 1,
                        max_pages: 1,
                    },
                    head_rows: 1,
                };
                let mut stages = Vec::new();
                let mut program = Vec::new();
                for stage in crate::program::moe(0) {
                    for op in lowering.stage(Phase::Decode(1), stage).unwrap().unwrap() {
                        stages.push(stage);
                        program.push(op.launch);
                    }
                }
                (stages, program)
            };
            let (stages, program) = lower(&b);
            assert_eq!(
                program
                    .iter()
                    .filter(|p| p.symbol.contains("_moe_floor_"))
                    .count(),
                7
            );
            assert!(program.last().unwrap().symbol.ends_with("_down_finish_v1"));
            let readonly = [b.scratch.normed_codes, b.scratch.normed_scales, live];
            crate::shared_overlap::checked_units(&stages, &program, &readonly).unwrap();
            for op in &program {
                for arg in &op.arguments.0 {
                    if let Argument::DeviceBuffer(h) = arg {
                        let span = op.bindings.get(*h).unwrap();
                        if span.buffer == b.workspace.buffer {
                            assert!(span.offset + span.length <= b.workspace.length);
                        }
                    }
                }
            }
            b.shared_workspace = b.workspace;
            let (stages, bad) = lower(&b);
            assert!(
                crate::shared_overlap::checked_units(&stages, &bad, &readonly)
                    .unwrap_err()
                    .contains("unordered scratch alias")
            );
        }
    }
}
