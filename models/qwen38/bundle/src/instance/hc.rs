//! Exact HC producer bottleneck, default-on after fixed-batch qualification.
pub const SWITCH: &str = "CHOPINFER_QWEN38_HC_PRECOMPUTE";
#[cfg(test)]
thread_local! {
    pub(crate) static PADDED_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    pub(crate) static FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = FOR_TEST.with(std::cell::Cell::get) {
        return value;
    }
    enabled_by_override(std::env::var_os(SWITCH).as_deref())
}
#[cfg(test)]
pub(crate) fn with_hc_precompute<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            FOR_TEST.with(|slot| slot.set(self.0));
        }
    }
    let _restore = Restore(FOR_TEST.with(|slot| slot.replace(Some(enabled))));
    run()
}

fn enabled_by_override(value: Option<&std::ffi::OsStr>) -> bool {
    value.and_then(std::ffi::OsStr::to_str).is_none_or(|value| {
        !["0", "false", "off"]
            .iter()
            .any(|off| value.eq_ignore_ascii_case(off))
    })
}

/// Unqualified layout candidate, enabled explicitly for queued GPU comparison.
pub(crate) fn padded_enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = PADDED_FOR_TEST.with(std::cell::Cell::get) {
        return value;
    }
    std::env::var("CHOPINFER_QWEN38_HC_PADDED_MIX").as_deref() == Ok("1")
}

#[cfg(test)]
pub(crate) fn with_hc_padded_mix<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PADDED_FOR_TEST.with(|slot| slot.set(self.0));
        }
    }
    let _restore = Restore(PADDED_FOR_TEST.with(|slot| slot.replace(Some(enabled))));
    run()
}

#[cfg(test)]
mod tests {
    use super::enabled_by_override;
    use std::ffi::OsStr;

    #[test]
    fn defaults_on_and_preserves_explicit_opt_out() {
        assert!(enabled_by_override(None));
        for value in ["0", "false", "FALSE", "FaLsE", "off", "OFF", "Off"] {
            assert!(!enabled_by_override(Some(OsStr::new(value))));
        }
        assert!(enabled_by_override(Some(OsStr::new("1"))));
        assert!(enabled_by_override(Some(OsStr::new(""))));
    }
}
