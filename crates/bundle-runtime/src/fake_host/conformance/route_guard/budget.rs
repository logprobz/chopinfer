//! Fail closed before a full sweep unless its process is in a limited cgroup.

use std::path::Path;
use std::time::{Duration, Instant};

const GIB: u64 = 1024 * 1024 * 1024;
const MAX_CGROUP_BYTES: u64 = 24 * GIB;
const MAX_RSS_BYTES: u64 = 16 * GIB;
/// Set in a child process that runs one test alone.
const CHILD: &str = "CHOPINFER_TEST_ROUTE_MEMORY_CHILD";

#[cfg(test)]
thread_local! {
    /// bundle-runtime's own probe sweeps run under a smaller bound.
    static LIMIT: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

pub(super) struct Budget {
    started: Instant,
    peak: u64,
    limit: u64,
    label: String,
    cgroup: Option<u64>,
}

fn rss_limit() -> u64 {
    #[cfg(test)]
    if let Some(limit) = LIMIT.get() {
        return limit;
    }
    MAX_RSS_BYTES
}

/// Run `run` with sweeps on this thread bounded by `limit` bytes.
#[cfg(test)]
pub(in crate::fake_host::conformance) fn with_rss_limit<R>(
    limit: u64,
    run: impl FnOnce() -> R,
) -> R {
    struct Restore(Option<u64>);
    impl Drop for Restore {
        fn drop(&mut self) {
            LIMIT.set(self.0);
        }
    }
    let _restore = Restore(LIMIT.replace(Some(limit)));
    run()
}

/// Whether this process is the child that runs one test alone.
#[cfg(test)]
pub(in crate::fake_host::conformance) fn in_own_process() -> bool {
    std::env::var_os(CHILD).is_some()
}

/// Run `run` in a child process of this test binary that runs only the
/// calling test, and return whether `run` ran in this process. VmHWM is a
/// whole-process high-water mark, and a library whose tests share one
/// process runs many of them at once, so a sweep measured there would be
/// charged for every other test's memory.
pub(super) fn own_process(run: impl FnOnce()) -> bool {
    if std::env::var_os(CHILD).is_some() {
        run();
        return true;
    }
    // libtest names each test's thread after the test.
    let test = std::thread::current()
        .name()
        .expect("route sweeps run on a libtest test thread")
        .to_owned();
    // Quiet output keeps the summary line but not a second `test ... ok`
    // line, which the CPU gate's inventory would count as another run.
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([test.as_str(), "--exact", "--nocapture", "--quiet"])
        .env(CHILD, "1")
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap_or_else(|e| panic!("{test}: start its own process: {e}"));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    print!("{stdout}");
    eprint!("{stderr}");
    assert!(
        output.status.success() && stdout.contains("test result: ok. 1 passed;"),
        "{test}: route sweep failed in its own process ({}):\n{stdout}{stderr}",
        output.status
    );
    false
}

fn rss(status: &str) -> Result<u64, String> {
    status
        .lines()
        .find_map(|line| {
            let value = line.strip_prefix("VmHWM:")?;
            let mut words = value.split_whitespace();
            Some((|| {
                let kib = words
                    .next()
                    .ok_or("missing VmHWM")?
                    .parse::<u64>()
                    .map_err(|_| "invalid VmHWM")?;
                if words.next() != Some("kB") {
                    return Err("invalid VmHWM unit".into());
                }
                kib.checked_mul(1024).ok_or_else(|| "VmHWM overflow".into())
            })())
        })
        .unwrap_or_else(|| Err("missing peak RSS in process status".into()))
}

fn limit(path: &Path, name: &str) -> Option<u64> {
    std::fs::read_to_string(path.join(name))
        .ok()?
        .trim()
        .parse()
        .ok()
}

fn cgroup_limit() -> u64 {
    let membership =
        std::fs::read_to_string("/proc/self/cgroup").expect("full sweep needs cgroup v2");
    let relative = membership
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .expect("full sweep needs cgroup v2");
    assert!(relative.starts_with('/') && !relative.split('/').any(|part| part == ".."));
    let root = Path::new("/sys/fs/cgroup");
    let path = root.join(relative.trim_start_matches('/'));
    let mut memory = u64::MAX;
    let mut swap = u64::MAX;
    for ancestor in path.ancestors().take_while(|p| p.starts_with(root)) {
        if let Some(value) = limit(ancestor, "memory.max") {
            memory = memory.min(value);
        }
        if let Some(value) = limit(ancestor, "memory.swap.max") {
            swap = swap.min(value);
        }
    }
    validate_limits(memory, swap).expect(
        "full sweep requires systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0; use tools/route_guard/sweep.py",
    );
    memory
}

fn validate_limits(memory: u64, swap: u64) -> Result<(), &'static str> {
    if memory > MAX_CGROUP_BYTES || swap != 0 {
        Err("unbounded route scope")
    } else {
        Ok(())
    }
}

impl Budget {
    pub fn begin(label: &str, full: bool) -> Self {
        let mut budget = Self {
            started: Instant::now(),
            peak: 0,
            limit: rss_limit(),
            label: label.into(),
            cgroup: full.then(cgroup_limit),
        };
        budget.check();
        budget
    }

    pub fn check(&mut self) {
        // VmHWM includes transient allocations between shape boundaries.
        let status = std::fs::read_to_string("/proc/self/status")
            .expect("route sweep needs peak RSS accounting");
        self.peak = self.peak.max(rss(&status).expect("process peak RSS"));
        assert!(
            self.peak <= self.limit,
            "{}: route sweep peak RSS {} exceeds budget {} bytes",
            self.label,
            self.peak,
            self.limit
        );
        if self.cgroup.is_some() {
            assert!(
                self.started.elapsed() < Duration::from_secs(780),
                "{}: sweep piece exceeded 13 minutes; increase --parts",
                self.label
            );
        }
    }
}

impl Drop for Budget {
    fn drop(&mut self) {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status")
            && let Ok(peak) = rss(&status)
        {
            self.peak = self.peak.max(peak);
        }
        eprintln!(
            "{}: peak_rss_bytes={} rss_budget_bytes={} cgroup_memory_max_bytes={:?} elapsed_seconds={:.3}",
            self.label,
            self.peak,
            self.limit,
            self.cgroup,
            self.started.elapsed().as_secs_f64()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[should_panic(expected = "exceeds budget")]
    fn retained_peak_over_budget_fails_without_allocating_large_buffers() {
        Budget {
            started: Instant::now(),
            peak: MAX_RSS_BYTES + 1,
            limit: MAX_RSS_BYTES,
            label: "fixture".into(),
            cgroup: None,
        }
        .check();
    }

    #[test]
    #[should_panic(expected = "exceeded 13 minutes")]
    fn full_piece_fails_at_its_internal_deadline() {
        Budget {
            started: Instant::now() - Duration::from_secs(781),
            peak: 0,
            limit: MAX_RSS_BYTES,
            label: "fixture".into(),
            cgroup: Some(MAX_CGROUP_BYTES),
        }
        .check();
    }

    #[test]
    fn refuses_unbounded_memory_or_enabled_swap() {
        assert!(validate_limits(24 * GIB, 0).is_ok());
        assert!(validate_limits(16 * GIB, 0).is_ok());
        assert!(validate_limits(u64::MAX, 0).is_err());
        assert!(validate_limits(24 * GIB + 1, 0).is_err());
        assert!(validate_limits(24 * GIB, 1).is_err());
        assert!(validate_limits(24 * GIB, u64::MAX).is_err());
    }

    #[test]
    fn reads_high_water_mark_and_refuses_missing_or_invalid_units() {
        assert_eq!(
            rss("Name:\tprobe\nVmRSS:\t4 kB\nVmHWM:\t8192 kB\n").unwrap(),
            8 * 1024 * 1024
        );
        assert!(rss("VmRSS: 12 kB").is_err());
        assert!(rss("VmHWM: 12 MB").is_err());
        assert!(rss("VmHWM: 18446744073709551615 kB").is_err());
    }
}
