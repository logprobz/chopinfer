//! Lowering of an Engram layer (`crate::ops::engram`): the table rows the
//! host gathered into the staging buffers before the step's launches
//! (`crate::engram_rows`) decoded to the BF16 rows `wkv` quantizes, and the
//! layer's residual streams, before its attention sublayer's mixer reads
//! them, gated in place by the F32 keys and value of its `wkv`.

use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Lowering, layer_view, rows, streams, sublayer_index};
use crate::buffers::engram_staged;
use crate::ops::Lowered;
use crate::ops::engram::{GateViews, RowsViews};
use crate::program::{Phase, Sublayer};

impl Lowering<'_> {
    /// Engram layer `layer`'s gathered table rows decoded to BF16.
    pub(super) fn engram_rows(
        &self,
        phase: Phase,
        layer: u32,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let Some(ops) = &self.ops.engram else {
            return Ok(None);
        };
        let g = self.shape.geometry;
        let index = g
            .shape
            .engram
            .layers
            .iter()
            .position(|&l| l == layer)
            .ok_or_else(|| format!("layer {layer} has no Engram rows"))?;
        let scratch = &self.buffers.scratch;
        let views = RowsViews {
            codes: engram_staged(scratch.engram_codes, self.shape, index, phase.rows()),
            scales: engram_staged(scratch.engram_scales, self.shape, index, phase.rows()),
            output: rows(scratch.engram_rows, phase.rows()),
            live_rows: self.inputs.live_tokens,
        };
        ops.decode(phase.rows(), &views)
            .map(Some)
            .map_err(|refusal| format!("Engram rows of layer {layer}: {refusal}"))
    }

    /// Engram layer `layer`'s gate of the streams its attention sublayer
    /// reads.
    pub(super) fn engram_gate(
        &self,
        phase: Phase,
        layer: u32,
    ) -> Result<Option<Vec<Lowered>>, String> {
        self.engram_gate_of(phase, layer, None)
    }

    /// Layer `layer`'s Engram gate closed on the prefill rows whose `text`
    /// flag (one U8 a row) is zero: a prefill that injects encoder rows.
    pub(crate) fn engram_gate_masked(
        &self,
        phase: Phase,
        layer: u32,
        text: DeviceBufferSpanV1,
    ) -> Result<Option<Vec<Lowered>>, String> {
        if !phase.prefill() {
            return Err("only a prefill's rows are image rows".into());
        }
        self.engram_gate_of(phase, layer, Some(rows(text, phase.rows())))
    }

    fn engram_gate_of(
        &self,
        phase: Phase,
        layer: u32,
        text: Option<DeviceBufferSpanV1>,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let Some(ops) = &self.ops.engram else {
            return Ok(None);
        };
        let scratch = &self.buffers.scratch;
        let index = sublayer_index(layer, Sublayer::Attention);
        let weight = |name: &str| {
            self.buffers
                .view(&layer_view(layer, &format!("engram.{name}")))
        };
        let views = GateViews {
            streams: rows(streams(scratch, index), phase.rows()),
            kv: rows(scratch.engram_kv, phase.rows()),
            q_weight: weight("q_weight")?,
            k_weight: weight("k_weight")?,
            live_rows: self.inputs.live_tokens,
        };
        ops.gate(phase.rows(), &views, text)
            .map(Some)
            .map_err(|refusal| format!("Engram gate of layer {layer}: {refusal}"))
    }
}
