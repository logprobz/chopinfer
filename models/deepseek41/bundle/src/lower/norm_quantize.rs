use super::{Lowering, layer_view, rows};
use crate::{
    ops::{Lowered, exact_f32::widens},
    program::{Phase, Sublayer},
};
use kernel_core::rmsnorm_quantize::RmsNormQuantizeBuffers;
use runtime_bundle_abi::DeviceBufferSpanV1;

impl Lowering<'_> {
    pub(super) fn fused_input_norm(
        &self,
        phase: Phase,
        layer: u32,
        sublayer: Sublayer,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let Some(ops) = &self.ops.norm_quantize else {
            return Ok(None);
        };
        if self.finish_norm_was_fused(phase, layer, sublayer)? {
            return Ok(Some(Vec::new()));
        }
        ops.launch(
            false,
            phase.rows(),
            self.input_norm_views(phase, layer, sublayer)?,
        )
        .map(Some)
        .map_err(|e| format!("input norm quantize of layer {layer}: {e}"))
    }

    pub(super) fn input_norm_views(
        &self,
        phase: Phase,
        layer: u32,
        sublayer: Sublayer,
    ) -> Result<RmsNormQuantizeBuffers<DeviceBufferSpanV1>, String> {
        let s = &self.buffers.scratch;
        let row = |span| rows(span, phase.rows());
        let weight = match sublayer {
            Sublayer::Attention => "attn_norm.weight",
            Sublayer::Mlp => "ffn_norm.weight",
        };
        Ok(RmsNormQuantizeBuffers {
            input: row(s.collapsed),
            weight: self.buffers.view(&layer_view(layer, weight))?,
            normed: row(s.normed),
            widened: widens(self.shape.geometry, layer, sublayer).then(|| row(s.normed_wide)),
            codes: row(s.normed_codes),
            scales: row(s.normed_scales),
            live_rows: self.inputs.live_tokens,
            status: row(s.row_status),
            status_row_offset: 0,
        })
    }

    pub(super) fn fused_query_norm(
        &self,
        phase: Phase,
        layer: u32,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let Some(ops) = &self.ops.norm_quantize else {
            return Ok(None);
        };
        let s = &self.buffers.scratch;
        let row = |span| rows(span, phase.rows());
        ops.launch(
            true,
            phase.rows(),
            RmsNormQuantizeBuffers {
                input: row(s.query_a),
                weight: self
                    .buffers
                    .view(&layer_view(layer, "attn.q_norm.weight"))?,
                normed: row(s.query_normed),
                widened: None,
                codes: row(s.query_codes),
                scales: row(s.query_scales),
                live_rows: self.inputs.live_tokens,
                status: row(s.row_status),
                status_row_offset: 0,
            },
        )
        .map(Some)
        .map_err(|e| format!("query norm quantize of layer {layer}: {e}"))
    }
}
