//! Opt-in transport for target prefill's F32 MoE reductions.

/// Exactly `1` selects the candidate. Unset keeps the qualified transport.
pub const SWITCH: &str = "CHOPINFER_GLM5NEXT_PREFILL_DMA";
pub const MIN_ROWS: u32 = 512;

pub(crate) fn requested() -> bool {
    #[cfg(test)]
    return IN_TESTS.get();
    #[cfg(not(test))]
    matches!(std::env::var(SWITCH).as_deref(), Ok("1"))
}

#[cfg(test)]
thread_local! {
    static IN_TESTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn with_enabled<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            IN_TESTS.set(self.0);
        }
    }
    let _restore = Restore(IN_TESTS.replace(enabled));
    run()
}
