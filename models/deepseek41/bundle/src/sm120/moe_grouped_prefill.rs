//! Bind the exact 8192-row TP4 down bucket qualified on captured routed inputs.
use super::moe_grouped_entries as family;
use kernel_core::backend::{Counts, Launch, OpInfo, PreparedOp, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::gated_mlp::GatedMlpProjection;
use kernel_core::moe_experts::{GatedMlpBuffers, PreparedExperts};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use model_deepseek41_contract::{
    geometry::{FLASH, Geometry},
    moe::Experts,
};
/// TP4, 250 W, captured 8192-row inputs: 0.840-0.869 candidate/reference
/// time over every tested layer/rank, with bitwise output and clean memcheck.
pub fn selected(
    g: &Geometry,
    ranks: u32,
    set: Experts,
    kind: GatedMlpProjection,
    usage: OperationUse,
) -> bool {
    g.id == FLASH.id
        && g.shape.hidden == FLASH.shape.hidden
        && g.shape.moe.intermediate == FLASH.shape.moe.intermediate
        && ranks == 4
        && set
            == Experts {
                count: 384,
                top_k: 6,
            }
        && kind == GatedMlpProjection::Down
        && usage.phase == ExecutionPhase::Prefill
        && usage.capacity == 8192
        && !usage.graph
}
pub fn prepare(base: Box<dyn PreparedExperts>) -> Result<Box<dyn PreparedExperts>, Refused> {
    let entry = family::entry();
    if base.route_tile() != 64
        || base.info().symbols.len() != 1
        || !base.info().symbols[0].0.ends_with(&entry.projection.symbol)
    {
        return Err(Refusal("grouped down requires its qualified column baseline").into());
    }
    let mut info = base.info().clone();
    let original = info.symbols[0].0.clone();
    info.implementation
        .push_str("+grouped_down_m128n128_prefill_v1");
    info.workspace = info.workspace.max(entry.workspace_bytes());
    for (name, bytes) in [
        ("routing.routes.counts", 384 * 4),
        ("routing.routes.active_experts", 385 * 4),
    ] {
        if let Some(v) = info.bytes.iter_mut().find(|(n, _)| n == name) {
            v.1 = v.1.max(bytes);
        } else {
            info.bytes.push((name.into(), bytes));
        }
    }
    info.symbols = entry
        .launches(8192, [1; 10], 1, 1, 1, 376, false)
        .map_err(Refused::from)?
        .into_iter()
        .map(|l| (l.symbol, l.plan.dynamic_shared_bytes))
        .collect();
    Ok(Box::new(Prepared {
        base,
        entry,
        info,
        original,
    }))
}
struct Prepared {
    base: Box<dyn PreparedExperts>,
    entry: kernels_sm120::moe::grouped_bf16::Entry,
    info: OpInfo,
    original: String,
}
impl PreparedExperts for Prepared {
    fn route_tile(&self) -> u32 {
        self.base.route_tile()
    }
    fn padded_rows(&self) -> u32 {
        self.base.padded_rows()
    }
}
impl PreparedOp<GatedMlpBuffers<View>> for Prepared {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &GatedMlpBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if counts.rows != 8192 {
            return Err(Refusal(
                "grouped down requires its measured 8192-row bucket",
            ));
        }
        workspace.covers(self.info.workspace)?;
        let old = self.base.launches(b, workspace, counts)?;
        if old.len() != 1 || old[0].symbol != self.original {
            return Err(Refusal("grouped down baseline changed"));
        }
        let routing = b.routing.ok_or(Refusal("grouped down requires routes"))?;
        let route_counts = routing
            .routes
            .counts
            .ok_or(Refusal("grouped down requires route counts"))?;
        let active = routing
            .routes
            .active_experts
            .ok_or(Refusal("grouped down requires active experts"))?;
        route_counts.covers(384 * 4)?;
        active.covers(385 * 4)?;
        let h = [
            b.activation.handle,
            b.weights.codes.handle,
            b.weights
                .local_scales
                .ok_or(Refusal("grouped down needs weight scales"))?
                .handle,
            b.activation_scales
                .ok_or(Refusal("grouped down needs input scales"))?
                .handle,
            routing.routes.offsets.handle,
            routing.routes.sorted_rows.handle,
            routing.routes.padded_rows.handle,
            b.live_tokens.handle,
            0,
            b.output.handle,
        ];
        if workspace.handle == 0
            || h.into_iter()
                .chain([route_counts.handle, active.handle])
                .any(|h| h == workspace.handle)
        {
            return Err(Refusal("grouped down queue aliases an operand"));
        }
        let mut ops = self.entry.launches(
            8192,
            h,
            route_counts.handle,
            active.handle,
            workspace.handle,
            376,
            false,
        )?;
        for op in &mut ops {
            op.plan.workspace_bytes = self.info.workspace;
        }
        Ok(ops)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::{Backend, Need};
    use kernel_core::moe::RouteMetadata;
    use kernel_core::moe_experts::GatedMlpScratch;
    use kernel_core::precision::buffers::{ProjectionRouting, QuantizedWeights};
    use kernels_sm120::backend::Sm120;
    #[test]
    fn selects_only_the_qualified_down_bucket() {
        for ranks in [1, 2, 4] {
            for set in model_deepseek41_contract::moe::expert_sets(&FLASH) {
                for rows in [1, 128, 512, 2048, 4096, 8191, 8192] {
                    for phase in [
                        ExecutionPhase::Prefill,
                        ExecutionPhase::Decode,
                        ExecutionPhase::Verify,
                    ] {
                        for graph in [false, true] {
                            for kind in [GatedMlpProjection::GateUp, GatedMlpProjection::Down] {
                                let usage = OperationUse {
                                    phase,
                                    capacity: rows,
                                    graph,
                                };
                                assert_eq!(
                                    selected(&FLASH, ranks, set, kind, usage),
                                    ranks == 4
                                        && set.count == 384
                                        && rows == 8192
                                        && phase == ExecutionPhase::Prefill
                                        && !graph
                                        && kind == GatedMlpProjection::Down
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn prepared_down_preserves_operands_and_refuses_queue_aliases() {
        let set = Experts {
            count: 384,
            top_k: 6,
        };
        let kind = GatedMlpProjection::Down;
        let usage = OperationUse {
            phase: ExecutionPhase::Prefill,
            capacity: 8192,
            graph: false,
        };
        let backend = super::super::moe::register(Sm120::new(), &FLASH, 4).unwrap();
        let query = model_deepseek41_contract::moe::projection(&FLASH, 4, set, kind, 8192).unwrap();
        let make = || {
            backend
                .chained_mlp_projection(&query, usage, Need::rows(8192))
                .unwrap()
        };
        let old = make();
        let op = super::super::moe_pipeline::prepare(&FLASH, 4, set, kind, usage, make()).unwrap();
        let mut handle = 0;
        let mut sized = |name: &str| {
            handle += 1;
            View {
                handle,
                bytes: op.info().bytes_of(name).unwrap_or(0),
            }
        };
        let b = GatedMlpBuffers {
            activation: sized("activation"),
            activation_scales: Some(sized("activation_scales")),
            activation_global: None,
            live_tokens: sized("live_tokens"),
            output: sized("output"),
            weights: QuantizedWeights {
                codes: sized("weights.codes"),
                local_scales: Some(sized("weights.local_scales")),
                global_scales: None,
            },
            routing: Some(ProjectionRouting {
                weights: sized("routing.weights"),
                routes: RouteMetadata {
                    offsets: sized("routing.routes.offsets"),
                    sorted_rows: sized("routing.routes.sorted_rows"),
                    padded_rows: sized("routing.routes.padded_rows"),
                    counts: Some(sized("routing.routes.counts")),
                    active_experts: Some(sized("routing.routes.active_experts")),
                },
            }),
            scratch: GatedMlpScratch {
                partial: None,
                output_codes: None,
                output_scales: None,
                quantizer_status: None,
                quantizer_status_row_offset: 0,
                unused: View::EMPTY,
            },
        };
        let workspace = View {
            handle: 100,
            bytes: op.info().workspace,
        };
        let original = old.launches(&b, workspace, Counts::rows(8192)).unwrap();
        let actual = op.launches(&b, workspace, Counts::rows(8192)).unwrap();
        let h = original[0]
            .arguments
            .0
            .iter()
            .map(|v| {
                if let kernel_core::contract::Argument::DeviceBuffer(h) = v {
                    *h
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        let routes = b.routing.unwrap().routes;
        let expected = family::entry()
            .launches(
                8192,
                h.try_into().unwrap(),
                routes.counts.unwrap().handle,
                routes.active_experts.unwrap().handle,
                workspace.handle,
                376,
                false,
            )
            .unwrap();
        assert_eq!(actual.len(), 2);
        assert_eq!(actual[1].plan.dynamic_shared_bytes, 96784);
        for (a, e) in actual.iter().zip(expected) {
            assert_eq!(a.symbol, e.symbol);
            assert_eq!(a.arguments, e.arguments);
            assert!(crate::sm120::fold_cubin_for_symbol(&a.symbol).is_some());
        }
        assert!(op.launches(&b, workspace, Counts::rows(8191)).is_err());
        assert!(
            op.launches(
                &b,
                View {
                    handle: b.output.handle,
                    ..workspace
                },
                Counts::rows(8192)
            )
            .is_err()
        );
        assert!(
            op.launches(
                &b,
                View {
                    bytes: workspace.bytes - 1,
                    ..workspace
                },
                Counts::rows(8192)
            )
            .is_err()
        );
    }
}
