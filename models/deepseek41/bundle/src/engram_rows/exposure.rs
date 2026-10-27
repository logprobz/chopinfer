//! Passive elapsed host time around one synchronous mapped gather call.
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub(super) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("CHOPINFER_SYNC_GATHER_TRACE").as_deref() == Ok("1"))
}

/// The disabled path reads no clocks. Neither path waits on a device or
/// changes the gather's result; logging follows the measured interval.
pub(super) fn measure<T, E>(
    enabled: bool,
    gather: impl FnOnce() -> Result<T, E>,
    emit: impl FnOnce(u64, u64, bool),
) -> Result<T, E> {
    if !enabled {
        return gather();
    }
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let start = Instant::now();
    let result = gather();
    let elapsed = start.elapsed();
    emit(
        u64::try_from(unix.as_nanos()).unwrap_or(u64::MAX),
        u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
        result.is_ok(),
    );
    result
}

#[cfg(test)]
mod tests {
    use super::measure;
    use std::cell::Cell;

    #[test]
    fn timer_preserves_one_call_and_result_with_and_without_logging() {
        for enabled in [false, true] {
            for result in [Ok(17), Err("gather refused")] {
                let calls = Cell::new(0);
                let logs = Cell::new(0);
                let observed = measure(
                    enabled,
                    || {
                        calls.set(calls.get() + 1);
                        result
                    },
                    |unix, _, succeeded| {
                        assert!(unix > 0);
                        assert_eq!(calls.get(), 1);
                        assert_eq!(succeeded, result.is_ok());
                        logs.set(logs.get() + 1);
                    },
                );
                assert_eq!(observed, result);
                assert_eq!(calls.get(), 1);
                assert_eq!(logs.get(), u32::from(enabled));
            }
        }
    }
}
