//! Prefill fusion preserves local norm planes and the carried status origin.
use super::{Lowering, layer_view, rows};
use crate::{
    ops::{Lowered, exact_f32::widens, norm_prefill::qualified},
    program::{Phase, Stage, Sublayer},
};
use kernel_core::rmsnorm_quantize::RmsNormQuantizeBuffers;

impl Lowering<'_> {
    pub(super) fn prefill_norm_stage(
        &self,
        phase: Phase,
        stage: Stage,
    ) -> Result<Option<Vec<Lowered>>, String> {
        if !qualified(self.shape.geometry, phase) {
            return Ok(None);
        }
        let Some(ops) = &self.ops.norm_prefill else {
            return Ok(None);
        };
        let (layer, sublayer) = match stage {
            Stage::InputNorm(layer, sublayer) => (layer, sublayer),
            Stage::QuantizeInput(..) => return Ok(Some(Vec::new())),
            _ => return Ok(None),
        };
        let s = &self.buffers.scratch;
        let row = |span| rows(span, phase.rows());
        let weight = match sublayer {
            Sublayer::Attention => "attn_norm.weight",
            Sublayer::Mlp => "ffn_norm.weight",
        };
        // The router-in-norm norm writes no widened rows: the router's SIMT
        // path widens its BF16 rows itself, and nothing else reads them.
        let in_norm = sublayer == Sublayer::Mlp && self.router_in_norm(phase);
        let views = RmsNormQuantizeBuffers {
            input: row(s.collapsed),
            weight: self.buffers.view(&layer_view(layer, weight))?,
            normed: row(s.normed),
            widened: (widens(self.shape.geometry, layer, sublayer) && !in_norm)
                .then(|| row(s.normed_wide)),
            codes: row(s.normed_codes),
            scales: row(s.normed_scales),
            live_rows: self.inputs.live_tokens,
            status: self.carried(phase, s.row_status)?,
            status_row_offset: 0,
        };
        let launched = if in_norm {
            ops.launch_certified(phase.rows(), views, row(s.router_certified))
        } else {
            ops.launch(phase.rows(), views)
        };
        launched
            .map(Some)
            .map_err(|e| format!("prefill input norm quantize of layer {layer}: {e}"))
    }

    /// Whether `phase` runs the router in the MoE-input norm
    /// (`sm120::router_norm`): the fused prefill norm and the exact F32
    /// router are both lowered at its row count.
    pub(super) fn router_in_norm(&self, phase: Phase) -> bool {
        phase.prefill()
            && qualified(self.shape.geometry, phase)
            && self.ops.norm_prefill.is_some()
            && self
                .ops
                .exact_f32
                .as_ref()
                .is_some_and(|ops| ops.projection("moe.router").is_some())
            && crate::Kernels::router_in_norm(self.shape.geometry, self.shape.ranks(), phase.rows())
    }
}
