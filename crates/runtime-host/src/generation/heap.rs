//! Return unused allocator pages at drained worker boundaries. Live buffers,
//! prefix state and device leases keep their owners and addresses.

use std::time::{Duration, Instant};

const INTERVAL: Duration = Duration::from_secs(60);
#[cfg(all(target_os = "linux", target_env = "gnu"))]
const MIN_FREE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Default)]
pub(super) struct IdleHeap {
    dirty: bool,
    last: Option<Instant>,
}

impl IdleHeap {
    pub(super) fn busy(&mut self) {
        self.dirty = true;
    }

    /// Called only after request and transfer drain, with no command waiting.
    /// A timed wait ensures a final burst is reclaimed even if no new request
    /// arrives before the cooldown ends.
    pub(super) fn idle(&mut self, now: Instant) -> Option<Duration> {
        self.idle_with(now, trim)
    }

    fn idle_with(&mut self, now: Instant, reclaim: impl FnOnce()) -> Option<Duration> {
        if !self.dirty {
            return None;
        }
        if let Some(last) = self.last
            && let Some(remaining) = INTERVAL.checked_sub(now.duration_since(last))
            && !remaining.is_zero()
        {
            return Some(remaining);
        }
        self.dirty = false;
        self.last = Some(now);
        reclaim();
        None
    }
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn trim() {
    let started = Instant::now();
    // SAFETY-REVIEWED: mallinfo2 returns allocator-owned counters by value
    // and synchronizes its reads with concurrent allocator activity.
    #[allow(unsafe_code)]
    let free = unsafe { libc::mallinfo2() }.fordblks;
    if free < MIN_FREE_BYTES {
        return;
    }
    // SAFETY-REVIEWED: malloc_trim releases only unused allocator pages and
    // synchronizes with concurrent allocations. It does not move live data.
    #[allow(unsafe_code)]
    let released = unsafe { libc::malloc_trim(0) };
    eprintln!(
        "generation worker heap trim: released={released} free_bytes={free} elapsed_us={}",
        started.elapsed().as_micros()
    );
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn trim() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unused_or_already_reclaimed_worker_needs_no_timer() {
        let now = Instant::now();
        let mut heap = IdleHeap::default();
        assert_eq!(heap.idle_with(now, || panic!("unused")), None);
        heap.busy();
        let mut calls = 0;
        assert_eq!(heap.idle_with(now, || calls += 1), None);
        assert_eq!(calls, 1);
        assert_eq!(heap.idle_with(now + INTERVAL, || panic!("clean")), None);
    }

    #[test]
    fn a_final_burst_waits_out_the_cooldown_without_another_request() {
        let now = Instant::now();
        let mut heap = IdleHeap::default();
        heap.busy();
        heap.idle_with(now, || {});
        heap.busy();
        for seconds in [1, 10, 59] {
            assert_eq!(
                heap.idle_with(now + Duration::from_secs(seconds), || panic!("too soon")),
                Some(INTERVAL - Duration::from_secs(seconds))
            );
        }
        let mut calls = 0;
        assert_eq!(heap.idle_with(now + INTERVAL, || calls += 1), None);
        assert_eq!(calls, 1);
        assert_eq!(heap.idle_with(now + INTERVAL * 2, || panic!("clean")), None);
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[test]
    fn drained_heap_returns_resident_free_pages_and_preserves_live_bytes() {
        const CHILD: &str = "CHOPINFER_TEST_IDLE_HEAP_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "generation::heap::tests::drained_heap_returns_resident_free_pages_and_preserves_live_bytes",
                    "--exact",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("MALLOC_ARENA_MAX", "1")
                .env("MALLOC_MMAP_THRESHOLD_", "1048576")
                .env("MALLOC_TRIM_THRESHOLD_", "1073741824")
                .env_remove("GLIBC_TUNABLES")
                .env_remove("LD_PRELOAD")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
            return;
        }
        // Live guards separate freed chunks so ordinary top-chunk trimming
        // cannot return the interior free pages. All allocations are touched.
        let mut live = Vec::new();
        let mut temporary = Vec::new();
        for index in 0..4096 {
            temporary.push(vec![0x5a_u8; 16 * 1024]);
            live.push(vec![(index % 251) as u8; 4096]);
        }
        std::hint::black_box(&temporary);
        drop(temporary);
        let before = anonymous_kib();
        let mut heap = IdleHeap::default();
        heap.busy();
        heap.idle(Instant::now());
        let after = anonymous_kib();
        eprintln!("resident anonymous KiB before {before}, after {after}");
        assert!(before >= after + 24 * 1024, "free pages remained resident");
        for (index, bytes) in live.iter().enumerate() {
            assert!(bytes.iter().all(|&byte| byte == (index % 251) as u8));
        }
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    fn anonymous_kib() -> u64 {
        std::fs::read_to_string("/proc/self/smaps_rollup")
            .unwrap()
            .lines()
            .find_map(|line| line.strip_prefix("Anonymous:"))
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap()
    }
}
