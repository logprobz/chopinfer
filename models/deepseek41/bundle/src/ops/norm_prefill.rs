//! Exact measured prefill buckets for the materialized norm/ceil chain.
use super::{Asks, Bind, Lowered, exact};
use crate::program::Phase;
use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::{contract::Refusal, rmsnorm_quantize::RmsNormQuantizeBuffers};
use model_deepseek41_contract::{geometry::FLASH, rows, serving::PREFILL_TOKENS};
use runtime_bundle_abi::DeviceBufferSpanV1;

/// Whether `shape`'s prefill of `phase`'s rows runs the fused norm-ceil:
/// at the rows of any of its [`sites`].
pub(crate) fn qualified(shape: &crate::shape::Shape, phase: Phase) -> bool {
    use model_deepseek41_contract::tuning::{self, Family};
    let g = shape.geometry;
    g == &FLASH
        && phase.prefill()
        && sites(shape).any(|op| {
            tuning::select(
                g,
                Family::Norm,
                op,
                "bf16",
                &[("width", g.shape.hidden)],
                PREFILL_TOKENS,
                phase.rows(),
            )
            .is_some()
        })
}

/// The table sites whose rows admit `shape`'s fused prefill norm-ceil: the
/// measured rows, and with the `prefill_norm_ceil_rows` candidate also its
/// range (`norm.input_range.prefill`).
pub(crate) fn sites(shape: &crate::shape::Shape) -> impl Iterator<Item = &'static str> {
    std::iter::once("norm.input_admit.prefill").chain(
        shape
            .prefill_norm_ceil_rows
            .then_some("norm.input_range.prefill"),
    )
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
        let buffers = Self::bind(&mut bind, views);
        bind.launch(&self.input, &buffers, Counts::rows(rows))
    }

    /// An MoE-input norm whose router runs in it (`sm120::router_norm`):
    /// the served launch, also certifying each row into `certified`.
    pub(crate) fn launch_certified(
        &self,
        rows: u32,
        views: RmsNormQuantizeBuffers<DeviceBufferSpanV1>,
        certified: DeviceBufferSpanV1,
    ) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let buffers = Self::bind(&mut bind, views);
        let certified = bind.view(certified);
        let served = self
            .input
            .launches(&buffers, View::EMPTY, Counts::rows(rows))?;
        let launch = crate::Kernels::router_in_norm_norm(served, certified)?;
        Ok(bind.lowered_launches(vec![launch]))
    }

    fn bind(
        bind: &mut Bind,
        views: RmsNormQuantizeBuffers<DeviceBufferSpanV1>,
    ) -> RmsNormQuantizeBuffers<View> {
        RmsNormQuantizeBuffers {
            input: bind.view(views.input),
            weight: bind.view(views.weight),
            normed: bind.view(views.normed),
            widened: views.widened.map(|v| bind.view(v)),
            codes: bind.view(views.codes),
            scales: bind.view(views.scales),
            live_rows: bind.view(views.live_rows),
            status: bind.view(views.status),
            status_row_offset: views.status_row_offset,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admit_prefill_rows_from_33() {
        use bundle_runtime::geometry::Ranks;
        let shape = crate::shape::Shape::new(&FLASH, Ranks::TP4).unwrap();
        let candidate = shape.with_prefill_norm_ceil_rows(true);
        for rows in 1..=8192 {
            assert_eq!(qualified(&shape, Phase::Prefill { rows }), rows > 32);
            assert_eq!(qualified(&candidate, Phase::Prefill { rows }), rows > 32);
            assert!(!qualified(&shape, Phase::Decode(rows)));
            assert!(!qualified(&candidate, Phase::Decode(rows)));
        }
    }
}
