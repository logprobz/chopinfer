//! Opt-in aggregate I/O diagnostics. Timings are summed worker time, not
//! elapsed startup time or isolated storage-device service time.
use std::collections::BTreeMap;
use std::io;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

macro_rules! counters {
    ($($field:ident),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, Default, serde::Serialize)]
        pub struct IoTiming {
            $(pub $field: u64,)+
            /// Requests through 4 KiB, 16 KiB, 64 KiB, 256 KiB, 1 MiB,
            /// 4 MiB, 16 MiB, and larger. These are loader requests.
            pub request_sizes: [u64; 8],
            /// Backward, contiguous, gaps through 4 KiB, 64 KiB, 1 MiB,
            /// 16 MiB, and larger, in request issue order within each shard.
            pub request_gaps: [u64; 7],
            /// Lifetime high-water mark, omitted from phase deltas.
            pub peak_reads: u64,
        }

        #[derive(Default)]
        pub(super) struct IoCounters {
            $(pub $field: AtomicU64,)+
            sizes: [AtomicU64; 8],
            gaps: [AtomicU64; 7],
            ends: Mutex<BTreeMap<usize, u64>>,
            active: AtomicU64,
            peak: AtomicU64,
        }

        impl IoCounters {
            pub fn snapshot(&self) -> IoTiming {
                IoTiming {
                    $($field: self.$field.load(Ordering::Relaxed),)+
                    request_sizes: std::array::from_fn(|i| self.sizes[i].load(Ordering::Relaxed)),
                    request_gaps: std::array::from_fn(|i| self.gaps[i].load(Ordering::Relaxed)),
                    peak_reads: self.peak.load(Ordering::Relaxed),
                }
            }
        }

        impl IoTiming {
            pub(super) fn since(self, before: Self) -> Self {
                Self {
                    $($field: self.$field.saturating_sub(before.$field),)+
                    request_sizes: std::array::from_fn(|i| self.request_sizes[i].saturating_sub(before.request_sizes[i])),
                    request_gaps: std::array::from_fn(|i| self.request_gaps[i].saturating_sub(before.request_gaps[i])),
                    peak_reads: 0,
                }
            }
        }
    };
}

counters!(
    buffered_calls,
    direct_calls,
    read_errors,
    read_bytes,
    syscall_nanos,
    residency_nanos,
    bounce_bytes,
    bounce_nanos,
    source_copy_bytes,
    output_copy_bytes,
    packing_nanos,
    unused_section_bytes,
);

impl IoCounters {
    pub fn request(&self, shard: usize, at: u64, bytes: u64) {
        let bucket = [4096, 16384, 65536, 262144, 1048576, 4194304, 16777216]
            .partition_point(|&limit| limit < bytes);
        self.sizes[bucket].fetch_add(1, Ordering::Relaxed);
        let mut ends = self
            .ends
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(end) = ends.insert(shard, at + bytes) {
            let bucket = if at < end {
                0
            } else if at == end {
                1
            } else {
                2 + [4096, 65536, 1048576, 16777216].partition_point(|&limit| limit < at - end)
            };
            self.gaps[bucket].fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn read(
        &self,
        direct: bool,
        read: impl FnOnce() -> io::Result<usize>,
    ) -> io::Result<usize> {
        let counter = if direct {
            &self.direct_calls
        } else {
            &self.buffered_calls
        };
        counter.fetch_add(1, Ordering::Relaxed);
        let now = self.active.fetch_add(1, Ordering::Relaxed) + 1;
        self.peak.fetch_max(now, Ordering::Relaxed);
        let start = Instant::now();
        let result = read();
        self.syscall_nanos
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.active.fetch_sub(1, Ordering::Relaxed);
        match &result {
            Ok(bytes) => {
                self.read_bytes.fetch_add(*bytes as u64, Ordering::Relaxed);
            }
            Err(_) => {
                self.read_errors.fetch_add(1, Ordering::Relaxed);
            }
        }
        result
    }
}

/// The same short-read and interruption handling as `FileExt::read_exact_at`,
/// with each attempted syscall visible to optional diagnostics.
pub(super) fn read_exact(
    mut read: impl FnMut(&mut [u8], u64) -> io::Result<usize>,
    mut buffer: &mut [u8],
    mut at: u64,
) -> io::Result<()> {
    while !buffer.is_empty() {
        match read(buffer, at) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(count) => {
                buffer = &mut buffer[count..];
                at += count as u64;
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_partial_reads_interruptions_and_errors_instead_of_requests() {
        let counters = IoCounters::default();
        let mut calls = 0;
        let mut out = [0; 4];
        read_exact(
            |out, at| {
                counters.read(false, || {
                    calls += 1;
                    if calls == 1 {
                        return Err(io::ErrorKind::Interrupted.into());
                    }
                    let n = out.len().min(2);
                    out[..n].fill(at as u8 + 1);
                    Ok(n)
                })
            },
            &mut out,
            0,
        )
        .unwrap();
        assert_eq!(out, [1, 1, 3, 3]);
        let before = counters.snapshot();
        assert_eq!(
            (before.buffered_calls, before.read_bytes, before.read_errors),
            (3, 4, 1)
        );
        assert!(read_exact(|_, _| counters.read(true, || Ok(0)), &mut out, 0).is_err());
        let delta = counters.snapshot().since(before);
        assert_eq!(
            (delta.direct_calls, delta.read_bytes, delta.peak_reads),
            (1, 0, 0)
        );
    }

    #[test]
    fn histograms_are_bounded_and_keep_shards_separate() {
        let counters = IoCounters::default();
        for (shard, at, bytes) in [
            (0, 0, 4096),
            (1, 0, 1),
            (0, 4096, 1),
            (0, 0, 1),
            (1, 4097, 16777217),
        ] {
            counters.request(shard, at, bytes);
        }
        let stats = counters.snapshot();
        assert_eq!(stats.request_sizes, [4, 0, 0, 0, 0, 0, 0, 1]);
        assert_eq!(stats.request_gaps, [1, 1, 1, 0, 0, 0, 0]);
    }
}
