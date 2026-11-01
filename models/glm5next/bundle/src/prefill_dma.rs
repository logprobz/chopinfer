//! Ordered DMA for the qualified target prefill configuration.

use crate::config::Configuration;
use model_glm5next_contract::contract::KV_LAYOUT_FP8_MTP;
use model_glm5next_contract::quantization::FP8_BLOCK128;

pub const MIN_ROWS: u32 = 512;

fn default_for(kv: &[u8], layout: &str, cp: u32, comm_epilogue: bool) -> bool {
    kv == KV_LAYOUT_FP8_MTP && layout == FP8_BLOCK128.id && cp == 1 && !comm_epilogue
}

/// Whether an instance of `config` runs its eligible prefills over ordered
/// DMA: the qualified configuration's, block-FP8 with FP8 MTP KV at CP1
/// without the comm epilogue. Every other configuration keeps the
/// collective.
pub(crate) fn requested(config: &Configuration) -> bool {
    let qualified = default_for(
        config.format.layout_id,
        config.layout.id,
        config.context_parallel.size(),
        config.comm_epilogue,
    );
    #[cfg(test)]
    return IN_TESTS.get().unwrap_or(qualified);
    #[cfg(not(test))]
    qualified
}

#[cfg(test)]
thread_local! {
    // Tests and route sweeps follow the served policy. The transport audits
    // select the legacy collective or the DMA explicitly to compare them.
    static IN_TESTS: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_selection<T>(selection: Option<bool>, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            IN_TESTS.set(self.0);
        }
    }
    let _restore = Restore(IN_TESTS.replace(selection));
    run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_glm5next_contract::contract::{KV_LAYOUT_BF16_MTP, KV_LAYOUT_FP8};
    use model_glm5next_contract::quantization::FP8_NVFP4_ROUTED;

    #[test]
    fn default_is_scoped_to_the_served_configuration() {
        assert!(default_for(KV_LAYOUT_FP8_MTP, FP8_BLOCK128.id, 1, false));
        for (kv, layout, cp, epilogue) in [
            (KV_LAYOUT_FP8, FP8_BLOCK128.id, 1, false),
            (KV_LAYOUT_BF16_MTP, FP8_BLOCK128.id, 1, false),
            (KV_LAYOUT_FP8_MTP, FP8_NVFP4_ROUTED.id, 1, false),
            (KV_LAYOUT_FP8_MTP, FP8_BLOCK128.id, 2, false),
            (KV_LAYOUT_FP8_MTP, FP8_BLOCK128.id, 4, false),
            (KV_LAYOUT_FP8_MTP, FP8_BLOCK128.id, 1, true),
        ] {
            assert!(!default_for(kv, layout, cp, epilogue));
        }
    }
}
