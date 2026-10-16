//! Lowering of the MoE at context-parallel size 1: each rank's routes,
//! the routed experts', shared expert's and dense MLP's projections and
//! the routed sum, as the ops the instance prepared (`crate::ops::moe`)
//! over the step's spans. The backend chooses every entry, schedule, row
//! tile and admission; the lowering binds spans only.

use super::Lowering;
use super::formats::Class;
use crate::buffers::Buffer;
use crate::ops::Lowered;
use crate::ops::moe::{
    Key, ProjectionViews, RouteSpans, RouteViews, RoutingSpans, SumViews, WeightSpans,
};
use crate::program::Phase;

impl Lowering<'_> {
    /// The route metadata's spans.
    fn route_spans(&self) -> RouteSpans {
        RouteSpans {
            counts: self.scratch(Buffer::RouteCounts),
            offsets: self.scratch(Buffer::RouteOffsets),
            sorted_rows: self.scratch(Buffer::SortedRows),
            padded_rows: self.scratch(Buffer::PaddedRows),
            active_experts: self.scratch(Buffer::ActiveExperts),
        }
    }

    /// The launches of a step's routes: the router's selection at `Route`
    /// and the routes of the rank's experts at `LocalRoutes` (`local`),
    /// the op's first launch and the rest (none where one launch does
    /// both).
    pub(super) fn route(
        &self,
        phase: Phase,
        layer: u32,
        local: bool,
    ) -> Result<Vec<Lowered>, String> {
        let views = RouteViews {
            logits: self.scratch(Buffer::RouterLogits),
            bias: self.buffers.view(&super::layer_view(
                layer,
                "mlp.gate.e_score_correction_bias",
            ))?,
            ids: self.scratch(Buffer::RouteIds),
            weights: self.scratch(Buffer::RouteWeights),
            routes: self.route_spans(),
            live_rows: self.inputs.live_tokens,
        };
        let mut launches = self
            .ops
            .moe
            .routes(phase, self.rank, &views)
            .map_err(|refusal| format!("routes of layer {layer}: {refusal}"))?;
        let rest = launches.split_off(launches.len().min(1));
        Ok(if local { rest } else { launches })
    }

    /// The packed weights `name` of `layer` that the projection of `key`
    /// reads, in its format.
    fn weight_spans(&self, layer: u32, name: &str, class: Class) -> Result<WeightSpans, String> {
        let [codes, block, local, globals] =
            self.chained_views(layer, name, self.chained_weights(layer, class))?;
        Ok(WeightSpans {
            codes: codes.ok_or_else(|| format!("{name} of layer {layer} has no codes"))?,
            // The block scales (FP8) and the local scales (NVFP4, MXFP8)
            // are one role.
            local_scales: block.or(local),
            global_scales: globals,
        })
    }

    /// The projection of `class` at `layer` as the op of its key.
    fn mlp_projection(
        &self,
        phase: Phase,
        layer: u32,
        class: Class,
        gate_up: bool,
        name: &str,
        views: impl FnOnce(WeightSpans) -> ProjectionViews,
    ) -> Result<Vec<Lowered>, String> {
        let key = Key {
            class,
            weight: super::formats::weight_format(self.ops.plan.layout, self.shape, layer, class),
            gate_up,
        };
        let weights = self.weight_spans(layer, name, class)?;
        self.ops
            .moe
            .projection(phase, key, &views(weights))
            .map_err(|refusal| format!("{name} of layer {layer}: {refusal}"))
    }

    /// The routed experts' gate/up (into the SwiGLU rows) or down (into the
    /// weighted route rows), over this rank's packed expert weights.
    pub(super) fn experts(
        &self,
        phase: Phase,
        layer: u32,
        gate_up: bool,
    ) -> Result<Vec<Lowered>, String> {
        let (name, input, output) = if gate_up {
            (
                "mlp.experts.gate_up",
                Buffer::Normed,
                Buffer::ExpertIntermediate,
            )
        } else {
            (
                "mlp.experts.down",
                Buffer::ExpertIntermediate,
                Buffer::Routed,
            )
        };
        self.mlp_projection(phase, layer, Class::Experts, gate_up, name, |weights| {
            ProjectionViews {
                activation: self.scratch(input),
                weights,
                output: self.scratch(output),
                live_tokens: self.inputs.live_tokens,
                routing: Some(RoutingSpans {
                    weights: self.scratch(Buffer::RouteWeights),
                    routes: self.route_spans(),
                }),
            }
        })
    }

    /// The routed sum: the rank's weighted routes summed per token in F32.
    /// A prefill's goes into the MLP output, which the shared expert's down
    /// projection adds to; a decode or verify graph's adds the shared
    /// expert's F32 output, which its down projection left in the MLP
    /// output, into the partial.
    pub(super) fn routed_sum(&self, phase: Phase) -> Result<Vec<Lowered>, String> {
        let prefill = matches!(phase, Phase::Prefill { .. });
        let mlp_out = self.scratch(Buffer::MlpOut);
        let views = SumViews {
            routed: self.scratch(Buffer::Routed),
            ids: self.scratch(Buffer::RouteIds),
            addend: (!prefill).then_some(mlp_out),
            output: if prefill {
                mlp_out
            } else {
                self.scratch(Buffer::Partial)
            },
            live_rows: self.inputs.live_tokens,
        };
        self.ops
            .moe
            .routed_sum(phase, self.rank, &views)
            .map_err(|refusal| format!("routed sum: {refusal}"))
    }

    /// The dense MLP's or shared expert's clamped gate/up into the SwiGLU
    /// rows.
    pub(super) fn dense_gate_up(
        &self,
        phase: Phase,
        layer: u32,
        shared: bool,
    ) -> Result<Vec<Lowered>, String> {
        let (class, name) = if shared {
            (Class::Shared, "mlp.shared_experts.gate_up")
        } else {
            (Class::Dense, "mlp.gate_up")
        };
        self.mlp_projection(phase, layer, class, true, name, |weights| ProjectionViews {
            activation: self.scratch(Buffer::Normed),
            weights,
            output: self.scratch(Buffer::DenseIntermediate),
            live_tokens: self.inputs.live_tokens,
            routing: None,
        })
    }
}
