//! Lowering of the exact F32 dense projections (`crate::ops::exact_f32`):
//! the sublayer input norm's rows widened once to F32, and each projection
//! over them (or over the KV source's normalized F32 latent) into the F32
//! rows its consumer reads.

use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Lowering, layer_view, rows};
use crate::ops::Lowered;
use crate::ops::exact_f32::{Input, WidenViews, project, widens};
use crate::program::{Phase, Sublayer};

impl Lowering<'_> {
    /// Sublayer `(layer, sublayer)`'s normed rows widened to F32, when one
    /// of its projections reads them; `None` without the ops.
    pub(super) fn widen_normed(
        &self,
        phase: Phase,
        layer: u32,
        sublayer: Sublayer,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let Some(ops) = &self.ops.exact_f32 else {
            return Ok(None);
        };
        if !widens(self.shape.geometry, layer, sublayer) {
            return Ok(Some(Vec::new()));
        }
        let scratch = &self.buffers.scratch;
        let views = WidenViews {
            input: rows(scratch.normed, phase.rows()),
            output: rows(scratch.normed_wide, phase.rows()),
            live_rows: self.inputs.live_tokens,
        };
        ops.widen(phase.rows(), &views)
            .map(Some)
            .map_err(|refusal| format!("F32 widening of layer {layer}: {refusal}"))
    }

    /// Projection `id` of layer `layer` if it is an exact F32 projection;
    /// `None` for any other projection, or while the ops are absent.
    pub(super) fn exact_f32(
        &self,
        phase: Phase,
        layer: u32,
        id: &str,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let Some((dense, op)) = self
            .ops
            .exact_f32
            .as_ref()
            .and_then(|ops| ops.projection(id))
        else {
            return Ok(None);
        };
        let scratch = &self.buffers.scratch;
        let input = match dense.input {
            Input::Normed => scratch.normed_wide,
            Input::Latent => scratch.compressed,
        };
        let output: DeviceBufferSpanV1 = match id {
            "moe.router" => scratch.router_logits,
            "compressor.kv" if self.shape.geometry.compress_ratio(layer) == 1 => {
                scratch.compressor_rows
            }
            "compressor.kv" => scratch.compressor_key,
            "compressor.gate" => scratch.compressor_gate,
            "indexer.k" => scratch.index_key,
            "indexer.weights" => scratch.index_weights,
            other => return Err(format!("F32 projection {other} has no output buffer")),
        };
        let weight = self
            .buffers
            .view(&layer_view(layer, &format!("{}.weight", dense.tensor)))?;
        if id == "moe.router" && self.router_in_norm(phase) {
            return self.certified_router(phase, layer, weight).map(Some);
        }
        project(
            op,
            phase.rows(),
            rows(input, phase.rows()),
            weight,
            rows(output, phase.rows()),
            self.inputs.live_tokens,
            crate::ops::exact_f32::state(
                self.buffers,
                &layer_view(layer, &format!("{}.weight", dense.tensor)),
            ),
        )
        .map(Some)
        .map_err(|refusal| format!("{id} of layer {layer}: {refusal}"))
    }

    /// The router's certified projection over the MoE-input norm's BF16
    /// rows and certificate (`sm120::router_norm`), which the norm stage
    /// wrote in place of the zero-component prepare.
    fn certified_router(
        &self,
        phase: Phase,
        layer: u32,
        weight: DeviceBufferSpanV1,
    ) -> Result<Vec<Lowered>, String> {
        let scratch = &self.buffers.scratch;
        let domain =
            crate::ops::exact_f32::state(self.buffers, &layer_view(layer, "ffn.gate.weight"))
                .weight_domain
                .ok_or_else(|| format!("moe.router of layer {layer} has no weight domain"))?;
        let mut bind = crate::ops::Bind::default();
        let mut view = |span| bind.view(span);
        let views = crate::ops::exact_f32::CertifiedRouterViews {
            widened: view(rows(scratch.normed_wide, phase.rows())),
            normed: view(rows(scratch.normed, phase.rows())),
            certified: view(rows(scratch.router_certified, phase.rows())),
            weight: view(weight),
            output: view(rows(scratch.router_logits, phase.rows())),
            live_rows: view(self.inputs.live_tokens),
            weight_domain: view(domain),
        };
        let launch = crate::Kernels::router_in_norm_router(&views)
            .map_err(|refusal| format!("moe.router of layer {layer}: {refusal}"))?;
        Ok(bind.lowered_launches(vec![launch]))
    }
}
