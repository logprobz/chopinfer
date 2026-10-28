//! Explicitly enabled NVTX annotations through a profiling-only preload helper.
//! The helper lives outside the install. Ordinary serving resolves no symbols.
use runtime_engine::profile::ranges::Subscription;

#[cfg(target_os = "linux")]
mod linux {
    use std::ffi::CStr;
    use std::sync::OnceLock;

    type Notify = unsafe extern "C" fn(*const libc::c_char, libc::c_int);
    static NOTIFY: OnceLock<Result<Notify, String>> = OnceLock::new();

    #[allow(unsafe_code)]
    fn resolve() -> Result<Notify, String> {
        // SAFETY-REVIEWED: RTLD_DEFAULT looks up a static symbol in already loaded
        // code. The explicitly supplied preload helper has this exact C signature
        // and remains loaded for the process lifetime.
        let symbol =
            unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"chopinfer_profile_range_v1".as_ptr()) };
        if symbol.is_null() {
            return Err("CHOPINFER_NVTX_RANGES=1 requires the profiling preload helper".into());
        }
        // SAFETY-REVIEWED: the helper exports the documented two-argument function.
        Ok(unsafe { std::mem::transmute::<*mut libc::c_void, Notify>(symbol) })
    }

    #[allow(unsafe_code)]
    fn notify(name: &'static CStr, begin: bool) {
        let function = NOTIFY
            .get()
            .expect("initialized observer")
            .as_ref()
            .expect("loaded observer");
        // SAFETY-REVIEWED: the function remains loaded, name is static and terminated,
        // and the thread-bound guard balances push and pop on this same thread.
        unsafe { function(name.as_ptr(), i32::from(begin)) };
    }

    pub(super) fn subscribe() -> Result<super::Subscription, String> {
        NOTIFY.get_or_init(resolve).as_ref().map_err(Clone::clone)?;
        Ok(runtime_engine::profile::ranges::subscribe(notify))
    }

    #[cfg(test)]
    #[test]
    fn absent_helper_is_refused() {
        assert!(resolve().is_err());
    }
}

pub(crate) fn initialize() -> Result<Option<Subscription>, String> {
    if std::env::var("CHOPINFER_NVTX_RANGES").as_deref() != Ok("1") {
        return Ok(None);
    }
    #[cfg(target_os = "linux")]
    return linux::subscribe().map(Some);
    #[cfg(not(target_os = "linux"))]
    Err("NVTX host ranges require Linux".into())
}

#[cfg(test)]
#[test]
fn ordinary_serving_installs_no_observer() {
    assert_ne!(std::env::var("CHOPINFER_NVTX_RANGES").as_deref(), Ok("1"));
    assert!(initialize().unwrap().is_none());
}
