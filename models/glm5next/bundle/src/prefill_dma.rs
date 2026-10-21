//! Ordered DMA for the qualified target prefill configuration.

use crate::config::Configuration;
use model_glm5next_contract::contract::KV_LAYOUT_FP8_MTP;
use model_glm5next_contract::quantization::FP8_BLOCK128;

/// Unset selects the qualified default; `0` disables it and `1` opts in.
pub const SWITCH: &str = "CHOPINFER_GLM5NEXT_PREFILL_DMA";
pub const MIN_ROWS: u32 = 512;

fn default_for(kv: &[u8], layout: &str, cp: u32, comm_epilogue: bool) -> bool {
    kv == KV_LAYOUT_FP8_MTP && layout == FP8_BLOCK128.id && cp == 1 && !comm_epilogue
}

fn select(value: Option<&str>, qualified: bool) -> bool {
    match value {
        None => qualified,
        Some("1") => true,
        Some(_) => false,
    }
}

pub(crate) fn requested(config: &Configuration) -> bool {
    let qualified = default_for(
        config.format.layout_id,
        config.layout.id,
        config.context_parallel.size(),
        config.comm_epilogue,
    );
    #[cfg(test)]
    return IN_TESTS.get().unwrap_or_else(|| select(None, qualified));
    #[cfg(not(test))]
    match std::env::var(SWITCH) {
        Ok(value) => select(Some(&value), qualified),
        Err(std::env::VarError::NotPresent) => select(None, qualified),
        Err(std::env::VarError::NotUnicode(_)) => false,
    }
}

#[cfg(test)]
thread_local! {
    // Existing journals exercise the legacy transport. The automatic policy
    // has a separate full-program comparison against the qualified opt-in.
    static IN_TESTS: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(Some(false)) };
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

    #[test]
    fn explicit_switches_preserve_opt_out_and_opt_in() {
        for qualified in [false, true] {
            assert_eq!(select(None, qualified), qualified);
            assert!(select(Some("1"), qualified));
            for value in ["0", "", "false", "invalid"] {
                assert!(!select(Some(value), qualified));
            }
        }
    }
}
