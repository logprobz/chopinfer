//! Explicit selection of the AR readiness bridge. Tests use thread-local
//! settings so journal tests do not depend on the process environment.
#[cfg(test)]
thread_local! {
    static ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    return ENABLED.get();
    #[cfg(not(test))]
    std::env::var_os("CHOPINFER_DEVICE_RESULTS").is_some_and(|value| value == "1")
}

#[cfg(test)]
pub(crate) fn with_enabled<T>(create: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            ENABLED.set(self.0);
        }
    }
    let _restore = Restore(ENABLED.replace(true));
    create()
}
