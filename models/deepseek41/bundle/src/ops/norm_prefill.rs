//! Exact measured prefill buckets for the materialized norm/ceil chain.
use super::{Asks, Bind, Lowered, exact};
use crate::program::Phase;
use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::{contract::Refusal, rmsnorm_quantize::RmsNormQuantizeBuffers};
use model_deepseek41_contract::{
    geometry::{FLASH, Geometry},
    rows,
    serving::PREFILL_TOKENS,
};
use runtime_bundle_abi::DeviceBufferSpanV1;

pub(crate) fn qualified(g: &Geometry, phase: Phase) -> bool {
    let rows = phase.rows();
    g == &FLASH
        && phase.prefill()
        && (rows == 8192 || ((64..=2624).contains(&rows) && (rows - 64).is_multiple_of(128)))
}

pub(crate) struct NormPrefillOps {
    input: Prepared<RmsNormQuantizeBuffers<View>>,
}
impl NormPrefillOps {
    pub(crate) fn prepare(backend: &dyn Backend, asks: &Asks) -> Result<Self, Refused> {
        let query = rows::norm_quantizers(&FLASH)[0];
        Ok(Self {
            input: asks.ask(
                "rows.prefill_norm_quantize",
                exact::need(PREFILL_TOKENS, exact::PREFILL),
                |need| backend.rmsnorm_quantize(&query, need),
            )?,
        })
    }

    pub(crate) fn launch(
        &self,
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
        bind.launch(&self.input, &buffers, Counts::rows(rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admit_only_measured_prefill_rows() {
        for rows in 1..=8192 {
            assert_eq!(
                qualified(&FLASH, Phase::Prefill { rows }),
                rows == 8192 || (0..21).any(|i| rows == 64 + 128 * i)
            );
            assert!(!qualified(&FLASH, Phase::Decode(rows)));
        }
    }
}
