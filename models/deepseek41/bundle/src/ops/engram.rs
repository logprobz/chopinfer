//! The Engram ops of an instance: each layer's gathered table rows decoded
//! from MXFP8 to the BF16 rows `wkv` quantizes (`kernel_core::mxfp8_rows`,
//! the vendor's `BF16_RNE(F32(code) * 2^(scale - 127))` per 32-block), and
//! the gate (`own_ops::engram::EngramGate`), open on every row of a step
//! without image rows and masked by each row's text flag in a prefill that
//! injects encoder rows (the vendor's `engram_mask`). The gate reads the F32 keys
//! and value of the layer's `wkv` (`ops::dense`), each row
//! `[(streams + 1) * width]`, and writes the residual streams in place.

use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::mxfp8_rows::{Mxfp8Rows, Mxfp8RowsBuffers};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::own_ops::OwnOps;
use model_deepseek41_contract::own_ops::engram::{EngramGate, EngramGateBuffers};
use model_deepseek41_contract::serving::PREFILL_TOKENS;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Asks, Bind, Lowered, exact};
use crate::shape::Shape;

/// The decode of `g`'s gathered Engram rows: a row's hash columns' table
/// rows end to end, `ops::dense::engram_k` columns.
pub(crate) fn rows_op(g: &Geometry) -> Mxfp8Rows {
    Mxfp8Rows {
        width: super::dense::engram_k(g),
    }
}

/// The Engram ops, prepared when the geometry has Engram layers.
pub(crate) struct EngramOps {
    layers: Option<Layers>,
}

struct Layers {
    decode: Prepared<Mxfp8RowsBuffers<View>>,
    gate: Prepared<EngramGateBuffers<View>>,
    /// The gate closed on rows whose text flag is zero, which only a
    /// prefill's rows take.
    masked: Prepared<EngramGateBuffers<View>>,
}

/// A step's spans of one Engram layer's row decode: the codes and scales
/// the host gathered, the BF16 rows and the live row count.
pub(crate) struct RowsViews {
    pub codes: DeviceBufferSpanV1,
    pub scales: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
}

/// A step's spans of one Engram layer's gate: the streams it updates in
/// place, the `wkv` output and the layer's weights.
pub(crate) struct GateViews {
    pub streams: DeviceBufferSpanV1,
    pub kv: DeviceBufferSpanV1,
    pub q_weight: DeviceBufferSpanV1,
    pub k_weight: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
}

impl EngramOps {
    pub(crate) fn prepare(
        backend: &dyn Backend,
        own: &dyn OwnOps,
        shape: &Shape,
        asks: &Asks,
    ) -> Result<EngramOps, Refused> {
        let g = shape.geometry;
        if g.engram_layers() == 0 {
            return Ok(EngramOps { layers: None });
        }
        let every = exact::need(PREFILL_TOKENS, exact::every_phase(shape.verify_rows));
        let decode = asks.ask("engram.rows", every, |need| {
            backend.decode_mxfp8_rows(&rows_op(g), need)
        })?;
        let gate = asks.ask("engram.gate", every, |need| {
            own.engram_gate(&EngramGate::of(g, false), need)
        })?;
        let prefill = exact::need(PREFILL_TOKENS, exact::PREFILL);
        let masked = asks.ask("engram.gate.masked", prefill, |need| {
            own.engram_gate(&EngramGate::of(g, true), need)
        })?;
        Ok(EngramOps {
            layers: Some(Layers {
                decode,
                gate,
                masked,
            }),
        })
    }

    fn layers(&self) -> Result<&Layers, Refusal> {
        self.layers.as_ref().ok_or(Refusal(
            "DeepSeek V4.1 has no Engram layer at this geometry",
        ))
    }

    /// The decode of `rows` gathered rows to BF16.
    pub(crate) fn decode(&self, rows: u32, views: &RowsViews) -> Result<Vec<Lowered>, Refusal> {
        let op = &self.layers()?.decode;
        let mut bind = Bind::default();
        let buffers = Mxfp8RowsBuffers {
            codes: bind.view(views.codes),
            scales: bind.view(views.scales),
            live_rows: bind.view(views.live_rows),
            output: bind.view(views.output),
        };
        bind.launch(op, &buffers, Counts::rows(rows))
    }

    /// The gate of `rows` rows: open, or with `text` (one U8 flag a row)
    /// closed on the rows whose flag is zero.
    pub(crate) fn gate(
        &self,
        rows: u32,
        views: &GateViews,
        text: Option<DeviceBufferSpanV1>,
    ) -> Result<Vec<Lowered>, Refusal> {
        let layers = self.layers()?;
        let op = match text {
            Some(_) => &layers.masked,
            None => &layers.gate,
        };
        let mut bind = Bind::default();
        let streams = bind.view(views.streams);
        let buffers = EngramGateBuffers {
            x: streams,
            kv: bind.view(views.kv),
            q_weight: bind.view(views.q_weight),
            k_weight: bind.view(views.k_weight),
            text: text.map(|text| bind.view(text)),
            live_rows: bind.view(views.live_rows),
            output: streams,
        };
        bind.launch(op, &buffers, Counts::rows(rows))
    }
}
