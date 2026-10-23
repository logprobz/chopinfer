//! What each rank's device holds once the serving KV pool is sized.
//!
//! The host services record it when they size the pool and after every graph
//! capture, on the generation worker. The metrics endpoint and the startup
//! warmup report read it from other threads through a shared handle.
use std::sync::{Arc, Mutex};

const MIB: u64 = 1 << 20;

/// Free memory a rank should keep after warmup: graphs captured later (a
/// bucket warmup missed, a prefill program) are allocated from it.
pub const WARMUP_HEADROOM_BYTES_V1: u64 = 256 * MIB;

/// One rank's device memory after the serving KV pool was sized.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RankDeviceMemoryV1 {
    pub rank: u32,
    /// The device's total memory, when the device backend reports it.
    pub total_bytes: Option<u64>,
    /// Memory this process's own allocations held when the pool was sized,
    /// before the pool itself, when the device backend accounts for it.
    pub own_bytes: Option<u64>,
    /// The rank's serving KV pool.
    pub kv_bytes: u64,
    /// Memory kept out of the pool for what is allocated after it.
    pub reserve_bytes: u64,
    /// Free memory right after the pool was allocated.
    pub free_after_sizing_bytes: u64,
    /// Free memory after the latest graph capture, or the sizing.
    pub free_bytes: u64,
    /// The lowest free memory seen since the sizing.
    pub free_low_bytes: u64,
}

impl RankDeviceMemoryV1 {
    /// The line the host logs when it sizes the pool.
    #[must_use]
    pub fn sizing_line(&self) -> String {
        let mib = |bytes: Option<u64>| {
            bytes.map_or("unknown".to_owned(), |bytes| format!("{} MiB", bytes / MIB))
        };
        format!(
            "serving KV memory rank {}: {} total, {} held by this process before the pool, \
             KV {} MiB, reserve {} MiB, {} MiB free",
            self.rank,
            mib(self.total_bytes),
            mib(self.own_bytes),
            self.kv_bytes / MIB,
            self.reserve_bytes / MIB,
            self.free_bytes / MIB
        )
    }

    /// Records the free memory measured after one graph capture.
    pub(crate) fn observe_free(&mut self, free: u64) {
        self.free_bytes = free;
        self.free_low_bytes = self.free_low_bytes.min(free);
    }
}

/// The shared record of every rank, empty until a serving KV pool is sized.
#[derive(Clone, Debug, Default)]
pub struct DeviceMemoryReportV1(Arc<Mutex<Vec<RankDeviceMemoryV1>>>);

impl DeviceMemoryReportV1 {
    pub(crate) fn publish(&self, ranks: &[RankDeviceMemoryV1]) {
        if let Ok(mut held) = self.0.lock() {
            held.clear();
            held.extend_from_slice(ranks);
        }
    }

    #[must_use]
    pub fn ranks(&self) -> Vec<RankDeviceMemoryV1> {
        self.0.lock().map(|ranks| ranks.clone()).unwrap_or_default()
    }

    /// One line per rank of what warmup left free, and a warning for each
    /// rank left with less than [`WARMUP_HEADROOM_BYTES_V1`].
    #[must_use]
    pub fn warmup_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for rank in self.ranks() {
            let taken = rank.free_after_sizing_bytes.saturating_sub(rank.free_bytes);
            lines.push(format!(
                "device memory after warmup rank {}: KV {} MiB, {} MiB taken since sizing \
                 (graphs), {} MiB free",
                rank.rank,
                rank.kv_bytes / MIB,
                taken / MIB,
                rank.free_bytes / MIB
            ));
            if rank.free_bytes < WARMUP_HEADROOM_BYTES_V1 {
                lines.push(format!(
                    "WARNING: rank {} has {} MiB free after warmup, less than the {} MiB a \
                     graph captured later may need; lower --gpu-memory-utilization or \
                     --max-sequences to leave more",
                    rank.rank,
                    rank.free_bytes / MIB,
                    WARMUP_HEADROOM_BYTES_V1 / MIB
                ));
            }
        }
        lines
    }

    /// Prometheus gauges of every rank; nothing before the pool is sized.
    #[must_use]
    pub fn prometheus(&self) -> String {
        let ranks = self.ranks();
        if ranks.is_empty() {
            return String::new();
        }
        let mut output = String::new();
        let gauges: [(&str, fn(&RankDeviceMemoryV1) -> Option<u64>); 6] = [
            ("chop_device_memory_total_bytes", |rank| rank.total_bytes),
            ("chop_device_memory_own_bytes", |rank| rank.own_bytes),
            ("chop_device_memory_kv_bytes", |rank| Some(rank.kv_bytes)),
            ("chop_device_memory_reserve_bytes", |rank| {
                Some(rank.reserve_bytes)
            }),
            ("chop_device_memory_free_bytes", |rank| {
                Some(rank.free_bytes)
            }),
            ("chop_device_memory_free_low_bytes", |rank| {
                Some(rank.free_low_bytes)
            }),
        ];
        for (name, value) in gauges {
            let values = ranks
                .iter()
                .filter_map(|rank| value(rank).map(|bytes| (rank.rank, bytes)))
                .collect::<Vec<_>>();
            if values.is_empty() {
                continue;
            }
            output.push_str(&format!("# TYPE {name} gauge\n"));
            for (rank, bytes) in values {
                output.push_str(&format!("{name}{{rank=\"{rank}\"}} {bytes}\n"));
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rank(rank: u32, free: u64) -> RankDeviceMemoryV1 {
        RankDeviceMemoryV1 {
            rank,
            total_bytes: Some(96 << 30),
            own_bytes: None,
            kv_bytes: 40 << 30,
            reserve_bytes: 1 << 30,
            free_after_sizing_bytes: 1 << 30,
            free_bytes: free,
            free_low_bytes: free,
        }
    }

    #[test]
    fn the_sizing_line_states_what_the_backend_reports() {
        assert_eq!(
            rank(0, 900 * MIB).sizing_line(),
            "serving KV memory rank 0: 98304 MiB total, unknown held by this process before the \
             pool, KV 40960 MiB, reserve 1024 MiB, 900 MiB free"
        );
    }

    #[test]
    fn warmup_lines_warn_only_below_the_headroom() {
        let report = DeviceMemoryReportV1::default();
        assert!(report.warmup_lines().is_empty());
        report.publish(&[rank(0, 900 * MIB), rank(1, 200 * MIB)]);
        let lines = report.warmup_lines();
        assert_eq!(lines.len(), 3);
        assert_eq!(
            lines[0],
            "device memory after warmup rank 0: KV 40960 MiB, 124 MiB taken since sizing \
             (graphs), 900 MiB free"
        );
        assert!(lines[2].starts_with("WARNING: rank 1 has 200 MiB free after warmup"));
    }

    #[test]
    fn prometheus_omits_what_the_backend_does_not_report() {
        let report = DeviceMemoryReportV1::default();
        assert_eq!(report.prometheus(), "");
        let mut low = rank(0, 900 * MIB);
        low.observe_free(800 * MIB);
        low.observe_free(850 * MIB);
        report.publish(&[low]);
        let text = report.prometheus();
        assert!(text.contains("chop_device_memory_total_bytes{rank=\"0\"} 103079215104\n"));
        assert!(!text.contains("chop_device_memory_own_bytes"));
        assert!(text.contains(&format!(
            "chop_device_memory_free_bytes{{rank=\"0\"}} {}\n",
            850 * MIB
        )));
        assert!(text.contains(&format!(
            "chop_device_memory_free_low_bytes{{rank=\"0\"}} {}\n",
            800 * MIB
        )));
    }
}
