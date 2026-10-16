//! Startup bounds shared by KV sizing and the reusable device sampler.
use runtime_bundle_abi::BundleStatusV1;

pub(crate) const SAMPLER_VIEW_ROWS: usize = 64;
pub(crate) const META_BYTES: usize = 112;
pub(crate) const OUTPUT_BYTES: usize = 256;
pub(crate) const COUNT_BYTES: usize = 16;
pub(crate) const BIAS_BYTES: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SamplingWorkspace {
    pub input: usize,
    pub output: usize,
    pub adjusted: usize,
    pub order: usize,
    pub partials: usize,
    pub filter: usize,
}

impl SamplingWorkspace {
    pub fn new(vocabulary: usize) -> Result<Self, BundleStatusV1> {
        if vocabulary == 0 || vocabulary > runtime_sampling::MAX_VOCABULARY {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let round = |bytes: usize| {
            bytes
                .checked_next_power_of_two()
                .ok_or(BundleStatusV1::ResourceExhausted)
        };
        // Counts are unique token IDs, biases are validated and unique, stop
        // IDs are bounded, and masks carry one bit per vocabulary token.
        let input = SAMPLER_VIEW_ROWS
            * (META_BYTES
                + vocabulary.min(runtime_sampling::MAX_HISTORY) * COUNT_BYTES
                + vocabulary.min(runtime_sampling::MAX_LOGIT_BIASES) * BIAS_BYTES
                + runtime_sampling::MAX_STOP_IDS * 4
                + vocabulary.div_ceil(32) * 4)
            + 48;
        Ok(Self {
            input: round(input)?,
            output: round(SAMPLER_VIEW_ROWS * OUTPUT_BYTES)?,
            adjusted: round(SAMPLER_VIEW_ROWS * vocabulary * 8)?,
            order: round(SAMPLER_VIEW_ROWS * vocabulary * 4)?,
            partials: round(
                SAMPLER_VIEW_ROWS
                    * vocabulary.div_ceil(runtime_kernel_contract::SAMPLING_SIMPLE_TILE)
                    * runtime_kernel_contract::SAMPLING_SIMPLE_PARTIAL_BYTES,
            )?,
            filter: round(SAMPLER_VIEW_ROWS * runtime_kernel_contract::SAMPLING_FILTER_ROW_BYTES)?,
        })
    }

    pub fn device_bytes(self) -> u64 {
        (self.input + self.output + self.adjusted + self.order + self.partials + self.filter) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn workspace_accounts_for_rounding_and_every_buffer() {
        let plan = SamplingWorkspace::new(152576).unwrap();
        assert_eq!(plan.adjusted, 128 * 1024 * 1024);
        assert_eq!(plan.order, 64 * 1024 * 1024);
        assert_eq!(plan.input, 256 * 1024 * 1024);
        assert_eq!(plan.filter, 4 * 1024 * 1024);
        assert_eq!(plan.partials, 2 * 1024 * 1024);
        assert_eq!(plan.output, 16 * 1024);
        assert_eq!(plan.device_bytes(), 454 * 1024 * 1024 + 16 * 1024);
        assert!(SamplingWorkspace::new(0).is_err());
        assert!(SamplingWorkspace::new(usize::MAX).is_err());
    }
}
