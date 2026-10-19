//! Lowering of a sublayer's input quantization and the routed MoE through
//! the instance's prepared ops (`crate::ops::moe`): the routes, the
//! experts' gate/up and down over the rank's expert bank, the shared
//! expert's activation between its dense projections, and the routed sum
//! with the shared expert's partial into the sublayer's F32 partial. The
//! engines check their views
//! against their prepared capacity, so each binds its whole buffers.

use super::{Lowering, layer_view, rows};
use crate::ops::Lowered;
use crate::ops::moe::{ExpertViews, QuantizeViews, RouteSpans, RouteViews, SumViews};
use crate::ops::shared::ActivationViews;
use crate::program::Phase;

impl Lowering<'_> {
    /// The normed rows quantized to E4M3 per 32 with ceil UE8M0 scales.
    pub(super) fn quantize_input(&self, phase: Phase) -> Result<Option<Vec<Lowered>>, String> {
        let Some(ops) = &self.ops.moe else {
            return Ok(None);
        };
        let scratch = &self.buffers.scratch;
        let views = QuantizeViews {
            input: rows(scratch.normed, phase.rows()),
            codes: rows(scratch.normed_codes, phase.rows()),
            scales: rows(scratch.normed_scales, phase.rows()),
            live_rows: self.inputs.live_tokens,
            status: self.carried(phase, scratch.row_status)?,
        };
        ops.quantize(phase.rows(), &views)
            .map(Some)
            .map_err(|refusal| format!("input quantization: {refusal}"))
    }

    fn route_spans(&self) -> RouteSpans {
        let scratch = &self.buffers.scratch;
        RouteSpans {
            counts: scratch.route_counts,
            offsets: scratch.route_offsets,
            sorted_rows: self.buffers.sorted_rows,
            padded_rows: scratch.padded_rows,
            active_experts: scratch.active_experts,
        }
    }

    /// Layer `layer`'s routes: the router's selection over its F32 logits
    /// with the text bias (the image bias where a row selects it), and the
    /// rank's routes.
    pub(super) fn route(&self, phase: Phase, layer: u32) -> Result<Option<Vec<Lowered>>, String> {
        let Some(ops) = &self.ops.moe else {
            return Ok(None);
        };
        let scratch = &self.buffers.scratch;
        let view = |name: &str| self.buffers.view(&layer_view(layer, name));
        let views = RouteViews {
            logits: scratch.router_logits,
            bias: view("ffn.gate.bias")?,
            image_bias: view("ffn.gate.bias_vl")?,
            bias_select: scratch.bias_select,
            scores: scratch.router_scores,
            ids: scratch.expert_ids,
            weights: scratch.expert_weights,
            routes: self.route_spans(),
            live_rows: self.inputs.live_tokens,
        };
        ops.routes(phase, self.rank, &views)
            .map(Some)
            .map_err(|refusal| format!("routes of layer {layer}: {refusal}"))
    }

    /// Layer `layer`'s experts' gate/up, from the quantized input into the
    /// weighted SwiGLU rows and their codes, or down, from those codes into
    /// the F32 routed rows. The rank's expert bank is the weight plan's
    /// `experts.gate_up` and `experts.down` views, codes and scales.
    pub(super) fn experts(
        &self,
        phase: Phase,
        layer: u32,
        gate_up: bool,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let Some(ops) = &self.ops.moe else {
            return Ok(None);
        };
        let scratch = &self.buffers.scratch;
        let bank = if gate_up { "gate_up" } else { "down" };
        let view = |part: &str| {
            self.buffers
                .view(&layer_view(layer, &format!("ffn.experts.{bank}.{part}")))
        };
        let (activation, activation_scales, output, output_quantized) = if gate_up {
            (
                scratch.normed_codes,
                scratch.normed_scales,
                scratch.expert_intermediate,
                Some((scratch.intermediate_codes, scratch.intermediate_scales)),
            )
        } else {
            (
                scratch.intermediate_codes,
                scratch.intermediate_scales,
                scratch.routed,
                None,
            )
        };
        let views = ExpertViews {
            activation,
            activation_scales,
            codes: view("weight")?,
            scales: view("scale")?,
            output,
            output_quantized,
            route_weights: scratch.expert_weights,
            routes: self.route_spans(),
            live_tokens: self.inputs.live_tokens,
            workspace: self.buffers.workspace,
            status: self.carried(phase, scratch.row_status)?,
        };
        ops.experts(phase, gate_up, &views)
            .map(Some)
            .map_err(|refusal| format!("experts {bank} of layer {layer}: {refusal}"))
    }

    /// The rank's routed rows summed per token into the sublayer's F32
    /// partial, plus the shared expert's F32 partial.
    pub(super) fn routed_sum(&self, phase: Phase) -> Result<Option<Vec<Lowered>>, String> {
        let Some(ops) = &self.ops.moe else {
            return Ok(None);
        };
        let scratch = &self.buffers.scratch;
        let views = SumViews {
            routed: scratch.routed,
            ids: scratch.expert_ids,
            addend: Some(scratch.shared_partial),
            output: if phase.prefill() && self.prefill_row_origin.is_some() {
                self.carried(phase, scratch.partial)?
            } else {
                scratch.partial
            },
            live_rows: self.inputs.live_tokens,
        };
        ops.routed_sum(phase, self.rank, &views)
            .map(Some)
            .map_err(|refusal| format!("routed sum: {refusal}"))
    }

    /// The shared expert's gate and up projections of layer `layer`.
    pub(super) fn shared_gate_up(
        &self,
        phase: Phase,
        layer: u32,
    ) -> Result<Option<Vec<Lowered>>, String> {
        if !phase.prefill() && phase.rows() <= crate::Kernels::dense_decode_capacity() {
            let count = phase.rows();
            let Some(ops) = &self.ops.dense else {
                return Ok(None);
            };
            let scratch = &self.buffers.scratch;
            let views = |tensor: &str, output| -> Result<crate::ops::GemmViews, String> {
                Ok(crate::ops::GemmViews {
                    stream: crate::ops::fold_stream::storage(
                        &self.buffers.views,
                        &layer_view(layer, "ffn.shared_experts.w1.weight"),
                        true,
                    ),
                    activations: rows(scratch.normed_codes, count),
                    activation_scales: rows(scratch.normed_scales, count),
                    weights: self.buffers.view(&layer_view(
                        layer,
                        &format!("ffn.shared_experts.{tensor}.weight"),
                    ))?,
                    weight_scales: self.buffers.view(&layer_view(
                        layer,
                        &format!("ffn.shared_experts.{tensor}.scale"),
                    ))?,
                    output: rows(output, count),
                    live_rows: self.inputs.live_tokens,
                    status: rows(scratch.row_status, count),
                })
            };
            return ops
                .project_pair(
                    count,
                    &[
                        views("w1", scratch.shared_gate)?,
                        views("w3", scratch.shared_up)?,
                    ],
                    self.buffers.workspace,
                )
                .map(Some)
                .map_err(|e| format!("shared gate/up of layer {layer}: {e}"));
        }
        let mut launches = Vec::new();
        for id in ["moe.shared-gate", "moe.shared-up"] {
            let Some(lowered) = self.dense(phase, layer, id)? else {
                return Ok(None);
            };
            launches.extend(lowered);
        }
        Ok(Some(launches))
    }

    /// The shared expert's activation of layer `layer`'s gate and up rows.
    pub(super) fn shared_activation(
        &self,
        phase: Phase,
        layer: u32,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let Some(ops) = &self.ops.shared else {
            return Ok(None);
        };
        let scratch = &self.buffers.scratch;
        let rows_of = |span| rows(span, phase.rows());
        let views = ActivationViews {
            gate: rows_of(scratch.shared_gate),
            up: rows_of(scratch.shared_up),
            codes: rows_of(scratch.shared_codes),
            scales: rows_of(scratch.shared_scales),
            live_rows: self.inputs.live_tokens,
            status: self.carried(phase, scratch.row_status)?,
        };
        ops.activate(phase.rows(), &views)
            .map(Some)
            .map_err(|refusal| format!("shared expert activation of layer {layer}: {refusal}"))
    }
}
