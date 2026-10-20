//! Full-chunk query norm/ceil composition, separate from decode selection.
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

pub(crate) fn eligible(g: &Geometry, phase: Phase) -> bool {
    g == &FLASH
        && matches!(
            phase,
            Phase::Prefill {
                rows: PREFILL_TOKENS
            }
        )
}

pub(crate) struct QueryPrefillOps {
    query: Prepared<RmsNormQuantizeBuffers<View>>,
}
impl QueryPrefillOps {
    pub(crate) fn prepare(backend: &dyn Backend, asks: &Asks) -> Result<Self, Refused> {
        Ok(Self {
            query: asks.ask(
                "rows.prefill_query_norm_quantize",
                exact::need(PREFILL_TOKENS, exact::PREFILL),
                |need| backend.rmsnorm_quantize(&rows::norm_quantizers(&FLASH)[1], need),
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
            widened: None,
            codes: bind.view(views.codes),
            scales: bind.view(views.scales),
            live_rows: bind.view(views.live_rows),
            status: bind.view(views.status),
            status_row_offset: views.status_row_offset,
        };
        bind.launch(&self.query, &buffers, Counts::rows(rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn limit_query_fusion_to_full_prefill_chunks() {
        for rows in 1..=PREFILL_TOKENS {
            assert_eq!(
                eligible(&FLASH, Phase::Prefill { rows }),
                rows == PREFILL_TOKENS
            );
            assert!(!eligible(&FLASH, Phase::Decode(rows)));
        }
    }
}
