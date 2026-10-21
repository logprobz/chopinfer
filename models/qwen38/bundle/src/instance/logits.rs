//! Target logits ownership. Draft logits keep the drafter's shard argmax.

use super::*;

/// Row-owned decode and verify logits, on by default at TP2 and TP4
/// after fixed-batch device runs matched the gather bit for bit
/// in the same configuration; `0` keeps the rank-0 gather. The host defers
/// views by default; `CHOPINFER_SAMPLING_DEFER_VIEWS=0` serializes their
/// sampling.
pub const ROW_OWNED_LOGITS: &str = "CHOPINFER_QWEN38_ROW_OWNED_LOGITS";

#[cfg(test)]
thread_local! {
    pub(crate) static FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

pub(super) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = FOR_TEST.with(std::cell::Cell::get) {
        return enabled;
    }
    std::env::var_os(ROW_OWNED_LOGITS).is_none_or(|value| value != "0")
}

impl Instance {
    pub(super) fn logit_views(
        &self,
        step: &PackedStep,
        decode: Option<&PhasePlan>,
    ) -> Result<Vec<runtime_bundle_abi::BundleLogitsViewV1>, BundleStatusV1> {
        let ranks = if decode.is_some() && !self.exchange.is_empty() {
            self.contexts
                .iter()
                .zip(&self.scratch)
                .map(|(&context, scratch)| (context, scratch.gathered_decode))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let owners = decode
            .filter(|_| !ranks.is_empty())
            .map(|plan| crate::logits::RowOwners {
                block: plan.rows.div_ceil(self.shape.ranks()),
                ranks: &ranks,
            });
        crate::logits::ordered_views_owned(
            self.shape.model_limits(),
            self.contexts[0],
            crate::logits::PhaseOutput {
                inputs: &step.prefill,
                buffer: self.scratch[0].gathered_scored,
                element_type: runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_F32,
                compact: true,
            },
            crate::logits::PhaseOutput {
                inputs: &step.decode,
                buffer: self.scratch[0].gathered_decode,
                element_type: runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_F32,
                compact: false,
            },
            owners,
        )
        .map_err(status)
    }
}

#[cfg(test)]
pub(crate) fn with_row_owners<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            FOR_TEST.with(|slot| slot.set(self.0));
        }
    }
    let _restore = Restore(FOR_TEST.with(|slot| slot.replace(Some(enabled))));
    run()
}
