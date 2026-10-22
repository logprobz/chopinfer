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
        let dual = ops
            .dual
            .as_ref()
            .filter(|op| op.eligible(self.shape, phase, layer));
        if dual.is_some() && id == "attention.kv" {
            // The preceding query stage has already written this private KV span.
            return Ok(Some(Vec::new()));
        }
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
                status: self.carried(phase, scratch.row_status)?,
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
            Input::Query
                if id == "attention.q-b"
                    && (matches!(phase, Phase::Decode(_))
                        || (self.ops.query_prefill.is_some()
                            && crate::ops::query_prefill::eligible(
                                self.shape.geometry,
                                phase,
                            ))) =>
            {
                (rows_of(scratch.query_codes), rows_of(scratch.query_scales))
            }
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
            stream: crate::ops::fold_stream::storage(
                &self.buffers.views,
                &layer_view(layer, &format!("{}.weight", dense.tensor)),
                false,
            ),
            activations,
            activation_scales,
            weights: weight("weight")?,
            weight_scales: weight("scale")?,
            output: if id == "attention.o-b" {
                self.carried(phase, output)?
            } else {
                rows_of(output)
            },
            live_rows: self.inputs.live_tokens,
            status: self.carried(phase, scratch.row_status)?,
        };
        if id == "attention.q-a"
            && let Some(dual) = dual
        {
            let kv = GemmViews {
                stream: crate::ops::fold_stream::storage(
                    &self.buffers.views,
                    &layer_view(layer, "attn.wkv.weight"),
                    false,
                ),
                activations: views.activations,
                activation_scales: views.activation_scales,
                weights: self.buffers.view(&layer_view(layer, "attn.wkv.weight"))?,
                weight_scales: self.buffers.view(&layer_view(layer, "attn.wkv.scale"))?,
                output: rows_of(scratch.kv),
                live_rows: views.live_rows,
                status: views.status,
            };
            return dual.launch(&[views, kv]).map(Some).map_err(error);
        }
        let local = if layer < self.shape.geometry.layers() {
            ops.local(id, phase, &views, self.buffers.workspace)
                .map_err(error)?
        } else {
            None
        };
        launches.extend(match local {
            Some(calls) => calls,
            None => ops
                .project(id, phase, &views, self.buffers.workspace)
                .map_err(error)?,
        });
        Ok(Some(launches))
    }
}
