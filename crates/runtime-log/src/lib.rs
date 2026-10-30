//! The serving process's log level and leveled output lines.
//!
//! A level decides only whether a line is written; the caller owns its text,
//! which tools parse, so these macros never add a prefix. Lines go to stderr
//! (`info_stdout!` to stdout) and a failed write is ignored rather than
//! panicking as `eprintln!` would. The level is process-wide and model-neutral:
//! the host sets it once from `--log-level` or `CHOPINFER_LOG_LEVEL`.
use std::fmt;
use std::io::Write;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// How much a process writes, least first. Each level includes the ones
/// before it: `Error` is always written.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Level {
    /// Server faults: 5xx responses, failed generations, fatal and startup failures.
    Error = 0,
    /// Client and capacity refusals, and requests ended by drain or shutdown.
    Warn = 1,
    /// Startup, readiness, lifecycle events and the periodic summary.
    Info = 2,
    /// Routine per-step and per-item traces.
    Debug = 3,
}

impl Level {
    pub const ALL: [Self; 4] = [Self::Error, Self::Warn, Self::Info, Self::Debug];

    /// The level a lowercase name selects, or `None` for any other text.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|level| level.as_str() == text)
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Info => "info",
            Self::Debug => "debug",
        }
    }
}

impl fmt::Display for Level {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

static LEVEL: AtomicU8 = AtomicU8::new(Level::Info as u8);

/// Sets the process-wide level. The serving process does this once, before
/// it starts threads that log.
pub fn set_level(level: Level) {
    LEVEL.store(level as u8, Ordering::Relaxed);
}

/// The process-wide level; `Info` until [`set_level`] changes it.
#[must_use]
pub fn level() -> Level {
    Level::ALL[usize::from(LEVEL.load(Ordering::Relaxed))]
}

/// Whether lines at `level` are written.
#[must_use]
pub fn enabled(level: Level) -> bool {
    level as u8 <= LEVEL.load(Ordering::Relaxed)
}

#[doc(hidden)]
pub fn write_stderr(arguments: fmt::Arguments<'_>) {
    let _ = writeln!(std::io::stderr().lock(), "{arguments}");
}

#[doc(hidden)]
pub fn write_stdout(arguments: fmt::Arguments<'_>) {
    let _ = writeln!(std::io::stdout().lock(), "{arguments}");
}

/// Writes one line to stderr at every level.
#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {
        if $crate::enabled($crate::Level::Error) {
            $crate::write_stderr(format_args!($($arg)*))
        }
    };
}

/// Writes one line to stderr at `warn` and above.
#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {
        if $crate::enabled($crate::Level::Warn) {
            $crate::write_stderr(format_args!($($arg)*))
        }
    };
}

/// Writes one line to stderr at `info` and above.
#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {
        if $crate::enabled($crate::Level::Info) {
            $crate::write_stderr(format_args!($($arg)*))
        }
    };
}

/// Writes one line to stderr at `debug`.
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {
        if $crate::enabled($crate::Level::Debug) {
            $crate::write_stderr(format_args!($($arg)*))
        }
    };
}

/// Writes one informational line to stdout at `info` and above.
#[macro_export]
macro_rules! info_stdout {
    ($($arg:tt)*) => {
        if $crate::enabled($crate::Level::Info) {
            $crate::write_stdout(format_args!($($arg)*))
        }
    };
}

/// The current time as RFC 3339 UTC with milliseconds, such as
/// `2026-10-25T14:03:11.123Z`.
#[must_use]
pub fn utc_timestamp() -> String {
    utc_timestamp_at(SystemTime::now())
}

/// `time` as RFC 3339 UTC with milliseconds. A time before the epoch is
/// written as the epoch.
#[must_use]
pub fn utc_timestamp_at(time: SystemTime) -> String {
    let since = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let seconds = since.as_secs();
    let (year, month, day) = civil_from_days(seconds / 86_400);
    let within = seconds % 86_400;
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        within / 3_600,
        within / 60 % 60,
        within % 60,
        since.subsec_millis()
    )
}

/// The proleptic Gregorian date `days` after 1970-01-01 (Howard Hinnant's
/// `civil_from_days`, for non-negative days).
fn civil_from_days(days: u64) -> (u64, u64, u64) {
    let days = days + 719_468;
    let era = days / 146_097;
    let day_of_era = days % 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + u64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    #[test]
    fn names_round_trip_and_nothing_else_parses() {
        for level in Level::ALL {
            assert_eq!(Level::parse(level.as_str()), Some(level));
            assert_eq!(level.to_string(), level.as_str());
        }
        for text in ["", "INFO", "Info", "warning", "trace", " info", "info\n"] {
            assert_eq!(Level::parse(text), None, "{text:?}");
        }
    }

    // One test owns the process-wide level, so parallel tests cannot race on it.
    #[test]
    fn each_level_includes_the_ones_before_it_and_skips_formatting_beyond() {
        assert_eq!(level(), Level::Info);
        static FORMATTED: AtomicUsize = AtomicUsize::new(0);
        struct Counted;
        impl fmt::Display for Counted {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                FORMATTED.fetch_add(1, Ordering::Relaxed);
                formatter.write_str("counted")
            }
        }
        for selected in Level::ALL {
            set_level(selected);
            assert_eq!(level(), selected);
            for candidate in Level::ALL {
                assert_eq!(enabled(candidate), candidate <= selected);
            }
            FORMATTED.store(0, Ordering::Relaxed);
            debug!("{}", Counted);
            info!("{}", Counted);
            warn!("{}", Counted);
            error!("{}", Counted);
            info_stdout!("{}", Counted);
            let expected = match selected {
                Level::Error => 1,
                Level::Warn => 2,
                Level::Info => 4,
                Level::Debug => 5,
            };
            assert_eq!(FORMATTED.load(Ordering::Relaxed), expected, "{selected}");
        }
        set_level(Level::Info);
    }

    #[test]
    fn timestamps_match_known_instants() {
        let at = |seconds: u64, millis: u64| {
            utc_timestamp_at(UNIX_EPOCH + Duration::from_millis(seconds * 1_000 + millis))
        };
        assert_eq!(at(0, 0), "1970-01-01T00:00:00.000Z");
        // 2000-02-29, a leap day in a year divisible by 400.
        assert_eq!(at(951_782_400, 5), "2000-02-29T00:00:00.005Z");
        assert_eq!(at(951_868_799, 999), "2000-02-29T23:59:59.999Z");
        assert_eq!(at(1_709_208_000, 0), "2024-02-29T12:00:00.000Z");
        assert_eq!(at(4_107_542_399, 0), "2100-02-28T23:59:59.000Z");
        assert_eq!(at(4_107_542_400, 0), "2100-03-01T00:00:00.000Z");
        assert_eq!(
            utc_timestamp_at(UNIX_EPOCH - Duration::from_secs(1)),
            "1970-01-01T00:00:00.000Z"
        );
        assert_eq!(utc_timestamp().len(), "1970-01-01T00:00:00.000Z".len());
    }
}
