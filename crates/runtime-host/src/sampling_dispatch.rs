//! Preserve the gathered view's numerical dispatch across owner partitions.
use std::ops::Range;

use runtime_bundle_abi::BundleStatusV1;
use runtime_engine::ModelBatchItem;
use runtime_scheduler::BatchPhase;

use crate::sampling_workspace::SAMPLER_VIEW_ROWS;

/// A gathered view's kernel choices, retained by every owned subview.
/// This is host-side Rust metadata, never part of the bundle C ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SamplingDispatch {
    pub(crate) rows: u32,
    pub(crate) simple: bool,
    pub(crate) top: bool,
    pub(crate) greedy: bool,
}

impl SamplingDispatch {
    pub(crate) fn for_items(items: &[&ModelBatchItem]) -> Result<Self, BundleStatusV1> {
        let mut result = Self {
            rows: items.len() as u32,
            simple: true,
            top: false,
            greedy: true,
        };
        for item in items {
            let work = item.sampling().ok_or(BundleStatusV1::InvalidArgument)?;
            let settings = &work.parameters;
            result.top |= settings.top_logprobs() != 0;
            result.greedy &= settings.temperature() == 0.0 && settings.top_logprobs() <= 1;
            let masked = work.constraint.as_ref().is_some_and(|c| c.mask().is_some());
            let masks_stop =
                work.emitted_tokens < settings.minimum_new_tokens() && !work.stop_ids.is_empty();
            let unfiltered = settings.temperature() == 0.0
                || (settings.top_k() == 0 && settings.top_p() == 1.0 && settings.min_p() == 0.0);
            result.simple &= !masked
                && !masks_stop
                && unfiltered
                && settings.logit_bias().is_empty()
                && settings.repetition_penalty() == 1.0
                && settings.frequency_penalty() == 0.0
                && settings.presence_penalty() == 0.0;
        }
        Ok(result)
    }

    #[cfg(any(test, all(target_os = "linux", feature = "linux-cuda-host")))]
    pub(crate) const fn threads(self) -> u32 {
        if self.top || self.rows <= 8 { 256 } else { 64 }
    }
}

/// Reconstruct the shared bundle view builder's unsplit ranges: compact
/// prefill rows and token-position decode/verify rows, in item order. Each
/// item is chunked before contiguous chunks merge, with a 64-row bound.
/// Scored steps retain item boundaries. Neither ranks nor owner blocks enter
/// this calculation, so gathered and owned callbacks use the same policy.
pub(crate) fn gathered_ranges(items: &[ModelBatchItem]) -> Vec<Range<usize>> {
    let scored = items.iter().any(|item| !item.prompt_targets().is_empty());
    let mut result: Vec<(bool, usize, Range<usize>)> = Vec::new();
    let mut phase_rows = [0_usize; 2];
    let mut first = 0;
    for item in items {
        let prefill = matches!(item.phase(), BatchPhase::Prefill { .. });
        let phase = usize::from(!prefill);
        let count = if !item.prompt_targets().is_empty() {
            item.prompt_targets().len()
        } else if item.expects_logits() {
            1 + item.draft().map_or(0, |d| d.tokens())
        } else {
            0
        };
        let offset = phase_rows[phase];
        phase_rows[phase] += if prefill {
            count
        } else {
            item.phase().appended_tokens()
        };
        for segment in (0..count).step_by(SAMPLER_VIEW_ROWS) {
            let rows = (count - segment).min(SAMPLER_VIEW_ROWS);
            let at = offset + segment;
            let range = first + segment..first + segment + rows;
            if !scored
                && let Some((last_phase, last_offset, last)) = result.last_mut()
                && *last_phase == prefill
                && *last_offset + last.len() == at
                && last.end == range.start
                && last.len() + rows <= SAMPLER_VIEW_ROWS
            {
                last.end = range.end;
            } else {
                result.push((prefill, at, range));
            }
        }
        first += count;
    }
    result.into_iter().map(|(_, _, range)| range).collect()
}
