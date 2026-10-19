use super::{Asks, Bind, Lowered, exact};
use crate::shape::Shape;
use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::rmsnorm_quantize::RmsNormQuantizeBuffers;
use model_deepseek41_contract::{rows, serving::DECODE_BATCH};
use runtime_bundle_abi::DeviceBufferSpanV1;

pub(crate) struct NormQuantizeOps {
    input: Prepared<RmsNormQuantizeBuffers<View>>,
    query: Prepared<RmsNormQuantizeBuffers<View>>,
}
impl NormQuantizeOps {
    pub(crate) fn prepare(
        backend: &dyn Backend,
        shape: &Shape,
        asks: &Asks,
    ) -> Result<Self, Refused> {
        let [input, query] = rows::norm_quantizers(shape.geometry);
        let need = exact::need(DECODE_BATCH, exact::every_phase(DECODE_BATCH));
        Ok(Self {
            input: asks.ask("rows.input_norm_quantize", need, |need| {
                backend.rmsnorm_quantize(&input, need)
            })?,
            query: asks.ask("rows.query_norm_quantize", need, |need| {
                backend.rmsnorm_quantize(&query, need)
            })?,
        })
    }
    pub(crate) fn launch(
        &self,
        query: bool,
        rows: u32,
        views: RmsNormQuantizeBuffers<DeviceBufferSpanV1>,
    ) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let buffers = RmsNormQuantizeBuffers {
            input: bind.view(views.input),
            weight: bind.view(views.weight),
            normed: bind.view(views.normed),
            widened: views.widened.map(|v| bind.view(v)),
            codes: bind.view(views.codes),
            scales: bind.view(views.scales),
            live_rows: bind.view(views.live_rows),
            status: bind.view(views.status),
            status_row_offset: views.status_row_offset,
        };
        bind.launch(
            if query { &self.query } else { &self.input },
            &buffers,
            Counts::rows(rows),
        )
    }
}
