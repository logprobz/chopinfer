//! Produce query codes in the full-chunk norm without changing latent kernels.
use super::{Lowering, layer_view, rows};
use crate::{
    ops::{Lowered, query_prefill::eligible},
    program::{Phase, Stage},
};
use kernel_core::rmsnorm_quantize::RmsNormQuantizeBuffers;

impl Lowering<'_> {
    pub(super) fn prefill_query_stage(
        &self,
        phase: Phase,
        stage: Stage,
    ) -> Result<Option<Vec<Lowered>>, String> {
        if !eligible(self.shape.geometry, phase) {
            return Ok(None);
        }
        let (Stage::QueryNorm(layer), Some(ops)) = (stage, &self.ops.query_prefill) else {
            return Ok(None);
        };
        let s = &self.buffers.scratch;
        ops.launch(
            phase.rows(),
            RmsNormQuantizeBuffers {
                input: rows(s.query_a, phase.rows()),
                weight: self
                    .buffers
                    .view(&layer_view(layer, "attn.q_norm.weight"))?,
                normed: rows(s.query_normed, phase.rows()),
                widened: None,
                codes: rows(s.query_codes, phase.rows()),
                scales: rows(s.query_scales, phase.rows()),
                live_rows: self.inputs.live_tokens,
                status: self.carried(phase, s.row_status)?,
                status_row_offset: 0,
            },
        )
        .map(Some)
        .map_err(|e| format!("prefill query norm of layer {layer}: {e}"))
    }
}
