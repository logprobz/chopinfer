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
use runtime_bundle_abi::DeviceBufferSpanV1;

/// Layer `layer`'s routed gate/up (`gate_up`) or down scale bank view.
fn scale_bank_name(layer: u32, gate_up: bool) -> String {
    let bank = if gate_up { "gate_up" } else { "down" };
    layer_view(layer, &format!("ffn.experts.{bank}.scale"))
}

impl Lowering<'_> {
    /// Select a placed tile-scale bank only through the schedule's closed
    /// consumer catalog. A row-offset bank always follows its raw expansion.
    fn inplace_scale_bank(
        &self,
        phase: Phase,
        layer: u32,
        gate_up: bool,
    ) -> Option<(DeviceBufferSpanV1, u32)> {
        let (span, encoding) = self
            .buffers
            .encoded_scale(&scale_bank_name(layer, gate_up))?;
        (encoding.format == kernel_core::weight_transform::ScaleFormat::RowTilesOffsetV1
            && self
                .ops
                .moe
                .as_ref()?
                .inplace_scales(phase, gate_up, encoding.width))
        .then_some((span, encoding.width))
    }

    /// Replace only the declared scale pointer of each encoded consumer.
    /// Other arguments and launch geometry retain the prepared schedule.
    pub(super) fn bind_inplace_scales(
        &self,
        phase: Phase,
        layer: u32,
        gate_up: bool,
        calls: &mut [Lowered],
    ) -> Result<(), String> {
        let Some((encoded, width)) = self.inplace_scale_bank(phase, layer, gate_up) else {
            return Ok(());
        };
        let raw = self.expert_scales(layer, gate_up)?;
        let bank = self.ops.moe.as_ref().unwrap().scale_bank(gate_up);
        let minimum = crate::Kernels::encoded_scale_minimum(
            bank,
            width,
            kernel_core::weight_transform::ScaleFormat::RowTilesOffsetV1,
        )
        .map_err(|e| e.to_string())?;
        crate::ops::moe::scale_inplace::bind(calls, raw, encoded, width, minimum)
    }

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
        ops.quantize(phase, &views)
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
        let mut launches = ops
            .routes(phase, self.rank, &views)
            .map_err(|refusal| format!("routes of layer {layer}: {refusal}"))?;
        let listed = ops.lists_active(phase, self.rank);
        launches.extend(self.expand_scales(phase, layer, listed)?);
        Ok(Some(launches))
    }

    /// The launch expanding layer `layer`'s routed scale banks the rank
    /// image stores encoded into their raw banks in scratch
    /// (`Scratch::expert_scales`), both in one launch after the routes: the
    /// experts the routes list where `listed` (every expert otherwise). None
    /// while both banks are stored raw. The expansion waits for its
    /// predecessor before it reads or writes, so it follows the routes and,
    /// on one stream, every read of the previous layer's banks.
    pub(super) fn expand_scales(
        &self,
        phase: Phase,
        layer: u32,
        listed: bool,
    ) -> Result<Option<Lowered>, String> {
        let encoded = [true, false]
            .map(|gate_up| self.buffers.encoded_scale(&scale_bank_name(layer, gate_up)));
        let Some(first) = encoded.iter().position(Option::is_some) else {
            return Ok(None);
        };
        let name = scale_bank_name(layer, first == 0);
        let ops = self
            .ops
            .moe
            .as_ref()
            .ok_or_else(|| format!("{name} is stored encoded without the routed MoE ops"))?;
        let mut expanded = [None, None];
        let mut format = None;
        for (at, bank) in encoded.into_iter().enumerate() {
            if let Some((encoded, encoding)) = bank {
                if self.inplace_scale_bank(phase, layer, at == 0).is_some() {
                    continue;
                }
                if format.is_some_and(|f| f != encoding.format) {
                    return Err("paired scale banks have different encoding formats".into());
                }
                format = Some(encoding.format);
                expanded[at] = Some((encoding.width, encoded, self.expert_scales(layer, at == 0)?));
            }
        }
        let Some(format) = format else {
            return Ok(None);
        };
        let active = listed.then_some(self.buffers.scratch.active_experts);
        ops.expand_scales(phase, expanded, format, active)
            .map(Some)
            .map_err(|refusal| format!("{name}: {refusal}"))
    }

    /// Layer `layer`'s gate/up (`gate_up`) or down scale bank as its
    /// experts bind it: the placed view, or where the rank image stores the
    /// bank encoded, its raw bank in scratch, which the routes stage expands
    /// ([`Self::expand_scales`]).
    pub(super) fn expert_scales(
        &self,
        layer: u32,
        gate_up: bool,
    ) -> Result<DeviceBufferSpanV1, String> {
        let name = scale_bank_name(layer, gate_up);
        if self.buffers.encoded_scale(&name).is_none() {
            return self.buffers.view(&name);
        }
        let ops = self
            .ops
            .moe
            .as_ref()
            .ok_or_else(|| format!("{name} is stored encoded without the routed MoE ops"))?;
        let expanded = self
            .buffers
            .scratch
            .expert_scales
            .ok_or_else(|| format!("{name} is stored encoded, but no raw bank is allocated"))?;
        let banks = [true, false].map(|gate_up| ops.scale_bank(gate_up).bytes());
        crate::buffers::expanded_scale_bank(expanded, banks, gate_up)
            .map_err(|error| format!("{name}: {error}"))
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
            scales: self.expert_scales(layer, gate_up)?,
            output,
            output_quantized,
            route_weights: scratch.expert_weights,
            routes: self.route_spans(),
            live_tokens: self.inputs.live_tokens,
            workspace: self.buffers.workspace,
            status: self.carried(phase, scratch.row_status)?,
        };
        let mut calls = ops
            .experts(phase, gate_up, &views)
            .map_err(|refusal| format!("experts {bank} of layer {layer}: {refusal}"))?;
        self.bind_inplace_scales(phase, layer, gate_up, &mut calls)?;
        Ok(Some(calls))
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
        ops.activate(phase, &views)
            .map(Some)
            .map_err(|refusal| format!("shared expert activation of layer {layer}: {refusal}"))
    }
}

#[cfg(test)]
#[path = "moe_scales_tests.rs"]
mod scales_tests;
