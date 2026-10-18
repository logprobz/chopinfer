//! Lowering of the dense FP8 projections (`crate::ops::dense`).

use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Lowering, layer_view, rows};
use crate::ops::dense::{Input, projections};
use crate::ops::moe::QuantizeViews;
use crate::ops::{GemmViews, Lowered};
use crate::program::Phase;

impl Lowering<'_> {
    /// Projection `id` of layer `layer` if it is a dense FP8 projection:
    /// its input quantized unless the sublayer did, then its fold GEMM.
    /// `None` for any other projection, or while the dense ops are absent.
    pub(super) fn dense(
        &self,
        phase: Phase,
        layer: u32,
        id: &str,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let Some(ops) = &self.ops.dense else {
            return Ok(None);
        };
        let Some(dense) = projections(self.shape).into_iter().find(|d| d.id == id) else {
            return Ok(None);
        };
        let scratch = &self.buffers.scratch;
        let rows_of = |span| rows(span, phase.rows());
        let error = |refusal| format!("{id} of layer {layer}: {refusal}");
        let mut launches = Vec::new();
        let quantized = |input: DeviceBufferSpanV1,
                         codes: DeviceBufferSpanV1,
                         scales: DeviceBufferSpanV1,
                         launches: &mut Vec<Lowered>|
         -> Result<(DeviceBufferSpanV1, DeviceBufferSpanV1), String> {
            let views = QuantizeViews {
                input: rows_of(input),
                codes: rows_of(codes),
                scales: rows_of(scales),
                live_rows: self.inputs.live_tokens,
                status: rows_of(scratch.row_status),
            };
            let width = dense.query.activation_row_stride;
            launches.extend(
                ops.quantize(width, dense.input.dtype(), phase.rows(), &views)
                    .map_err(error)?,
            );
            Ok((views.codes, views.scales))
        };
        let (activations, activation_scales) = match dense.input {
            Input::Layer => (
                rows_of(scratch.normed_codes),
                rows_of(scratch.normed_scales),
            ),
            Input::SharedActivation => (
                rows_of(scratch.shared_codes),
                rows_of(scratch.shared_scales),
            ),
            Input::Query => quantized(
                scratch.query_normed,
                scratch.query_codes,
                scratch.query_scales,
                &mut launches,
            )?,
            Input::Attention => quantized(
                scratch.attention,
                scratch.attention_codes,
                scratch.attention_scales,
                &mut launches,
            )?,
            Input::OutputA => quantized(
                scratch.output_a,
                scratch.output_a_codes,
                scratch.output_a_scales,
                &mut launches,
            )?,
            Input::Engram => quantized(
                scratch.engram_rows,
                scratch.engram_quantized,
                scratch.engram_quantized_scales,
                &mut launches,
            )?,
        };
        let output = match id {
            "attention.q-a" => scratch.query_a,
            "attention.q-b" => scratch.query,
            "indexer.q" => scratch.index_query,
            "attention.kv" => scratch.kv,
            "attention.o-a" => scratch.output_a,
            "attention.o-b" => scratch.partial,
            "moe.shared-gate" => scratch.shared_gate,
            "moe.shared-up" => scratch.shared_up,
            "moe.shared-down" => scratch.shared_partial,
            "engram.kv" => scratch.engram_kv,
            other => return Err(format!("dense projection {other} has no output buffer")),
        };
        let weight = |part: &str| {
            self.buffers
                .view(&layer_view(layer, &format!("{}.{part}", dense.tensor)))
        };
        let views = GemmViews {
            activations,
            activation_scales,
            weights: weight("weight")?,
            weight_scales: weight("scale")?,
            output: rows_of(output),
            live_rows: self.inputs.live_tokens,
            status: rows_of(scratch.row_status),
        };
        launches.extend(ops.project(id, phase.rows(), &views).map_err(error)?);
        Ok(Some(launches))
    }
}
