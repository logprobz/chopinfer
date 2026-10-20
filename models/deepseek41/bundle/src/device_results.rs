//! Qualified defaults for the AR readiness bridge. Tests use thread-local
//! settings so journal tests do not depend on the process environment.
#[cfg(test)]
thread_local! {
    static OVERRIDE: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(Some("0")) };
}

pub(crate) fn enabled(
    shape: crate::shape::Shape,
    kv: model_deepseek41_contract::state::KvStorage,
) -> bool {
    let default =
        model_deepseek41_contract::serving::default_device_results(shape.geometry, shape.ranks())
            && kv == model_deepseek41_contract::state::KvStorage::Native;
    #[cfg(test)]
    let value = OVERRIDE.get().map(std::ffi::OsString::from);
    #[cfg(not(test))]
    let value = std::env::var_os("CHOPINFER_DEVICE_RESULTS");
    runtime_bundle_abi::device_results::device_results_selected_v1(default, value.as_deref())
}

#[cfg(test)]
pub(crate) fn with_override<T>(value: Option<&'static str>, create: impl FnOnce() -> T) -> T {
    struct Restore(Option<&'static str>);
    impl Drop for Restore {
        fn drop(&mut self) {
            OVERRIDE.set(self.0);
        }
    }
    let _restore = Restore(OVERRIDE.replace(value));
    create()
}
