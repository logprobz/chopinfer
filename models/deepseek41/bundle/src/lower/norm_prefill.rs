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
        ops.launch(
            phase.rows(),
            RmsNormQuantizeBuffers {
                input: row(s.collapsed),
                weight: self.buffers.view(&layer_view(layer, weight))?,
                normed: row(s.normed),
                widened: widens(self.shape.geometry, layer, sublayer).then(|| row(s.normed_wide)),
                codes: row(s.normed_codes),
                scales: row(s.normed_scales),
                live_rows: self.inputs.live_tokens,
                status: self.carried(phase, s.row_status)?,
                status_row_offset: 0,
            },
        )
        .map(Some)
        .map_err(|e| format!("prefill input norm quantize of layer {layer}: {e}"))
    }
}
