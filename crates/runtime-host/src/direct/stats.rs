use std::{collections::BTreeMap, ops::Range};

/// Optional diagnostics split shared materialization from lock contention.
/// Materialization includes its source reads and input-cache waits.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct SharedTiming {
    pub output_wait_seconds: f64,
    pub materialize_seconds: f64,
    pub input_wait_seconds: f64,
}

impl SharedTiming {
    fn since(self, before: Self) -> Self {
        Self {
            output_wait_seconds: (self.output_wait_seconds - before.output_wait_seconds).max(0.0),
            materialize_seconds: (self.materialize_seconds - before.materialize_seconds).max(0.0),
            input_wait_seconds: (self.input_wait_seconds - before.input_wait_seconds).max(0.0),
        }
    }
}

/// Counters of one planned image's production so far.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct DirectLoadStatsV1 {
    pub produced_bytes: u64,
    pub source_bytes: u64,
    /// Positional reads of the checkpoint that produced `source_bytes`.
    pub source_reads: u64,
    /// Source bytes requested, including reads served from retained inputs.
    pub requested_source_bytes: u64,
    /// Bytes returned by successful file reads, including direct-I/O alignment.
    /// Buffered reads can come from the page cache, not the storage device.
    pub file_read_bytes: u64,
    /// Union of requested file ranges fetched from files, excluding alignment.
    pub unique_source_bytes: u64,
    /// Exact range unions are an opt-in diagnostic. Without tracking, the
    /// unique-byte counter is zero and reads take no range-accounting lock.
    pub source_ranges_tracked: bool,
    pub source_cache_hits: u64,
    pub source_cache_bytes: u64,
    pub shared_output_hits: u64,
    pub shared_output_productions: u64,
    pub shared_output_reproductions: u64,
    pub advised_bytes: u64,
    /// The part of `source_bytes` read with `O_DIRECT`.
    pub direct_bytes: u64,
    /// Thread-seconds in source reads, in production, in shared outputs and
    /// in rank reads (wall time of each rank's reads, summed over ranks).
    pub read_seconds: f64,
    pub produce_seconds: f64,
    pub shared_seconds: f64,
    pub rank_seconds: f64,
    /// Present only with `CHOPINFER_DIRECT_LOAD_TIMING=1`. These are summed
    /// worker seconds and overlap the existing read and shared counters.
    pub shared_timing: Option<SharedTiming>,
    /// Lifetime high-water mark, not an additive phase counter.
    #[serde(skip)]
    pub retained_peak_bytes: u64,
    /// Idle packed-buffer capacity, separately bounded from shared inputs.
    #[serde(skip)]
    pub pooled_peak_bytes: u64,
}

impl DirectLoadStatsV1 {
    /// Additive work since an earlier snapshot of the same image.
    /// Snapshots are taken between phases after their workers have joined.
    pub fn since(self, before: Self) -> Self {
        Self {
            produced_bytes: self.produced_bytes.saturating_sub(before.produced_bytes),
            source_bytes: self.source_bytes.saturating_sub(before.source_bytes),
            source_reads: self.source_reads.saturating_sub(before.source_reads),
            requested_source_bytes: self
                .requested_source_bytes
                .saturating_sub(before.requested_source_bytes),
            file_read_bytes: self.file_read_bytes.saturating_sub(before.file_read_bytes),
            unique_source_bytes: self
                .unique_source_bytes
                .saturating_sub(before.unique_source_bytes),
            source_ranges_tracked: self.source_ranges_tracked,
            source_cache_hits: self
                .source_cache_hits
                .saturating_sub(before.source_cache_hits),
            source_cache_bytes: self
                .source_cache_bytes
                .saturating_sub(before.source_cache_bytes),
            shared_output_hits: self
                .shared_output_hits
                .saturating_sub(before.shared_output_hits),
            shared_output_productions: self
                .shared_output_productions
                .saturating_sub(before.shared_output_productions),
            shared_output_reproductions: self
                .shared_output_reproductions
                .saturating_sub(before.shared_output_reproductions),
            advised_bytes: self.advised_bytes.saturating_sub(before.advised_bytes),
            direct_bytes: self.direct_bytes.saturating_sub(before.direct_bytes),
            read_seconds: (self.read_seconds - before.read_seconds).max(0.0),
            produce_seconds: (self.produce_seconds - before.produce_seconds).max(0.0),
            shared_seconds: (self.shared_seconds - before.shared_seconds).max(0.0),
            rank_seconds: (self.rank_seconds - before.rank_seconds).max(0.0),
            shared_timing: self
                .shared_timing
                .map(|timing| timing.since(before.shared_timing.unwrap_or_default())),
            retained_peak_bytes: 0,
            pooled_peak_bytes: 0,
        }
    }
}

/// Coalesced requested file ranges. Adjacent reads occupy one entry.
#[derive(Default)]
pub(super) struct SourceRanges(BTreeMap<(usize, u64), u64>);

impl SourceRanges {
    pub fn insert(&mut self, file: usize, range: Range<u64>) -> u64 {
        let (mut start, mut end) = (range.start, range.end);
        let mut covered = 0;
        if let Some((&(prior_file, prior_start), &prior_end)) =
            self.0.range(..=(file, start)).next_back()
            && prior_file == file
            && prior_end >= start
        {
            start = prior_start;
            end = end.max(prior_end);
        }
        while let Some((&key, &prior_end)) = self.0.range((file, start)..=(file, end)).next() {
            covered += prior_end - key.1;
            end = end.max(prior_end);
            self.0.remove(&key);
        }
        self.0.insert((file, start), end);
        end - start - covered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_timing_deltas_preserve_disabled_diagnostics() {
        let before = DirectLoadStatsV1 {
            shared_timing: Some(SharedTiming {
                output_wait_seconds: 4.0,
                materialize_seconds: 12.0,
                input_wait_seconds: 3.0,
            }),
            ..DirectLoadStatsV1::default()
        };
        let after = DirectLoadStatsV1 {
            shared_timing: Some(SharedTiming {
                output_wait_seconds: 6.0,
                materialize_seconds: 19.0,
                input_wait_seconds: 4.0,
            }),
            ..DirectLoadStatsV1::default()
        };
        let delta = after.since(before).shared_timing.unwrap();
        assert_eq!(delta.output_wait_seconds, 2.0);
        assert_eq!(delta.materialize_seconds, 7.0);
        assert_eq!(delta.input_wait_seconds, 1.0);
        assert!(
            DirectLoadStatsV1::default()
                .since(before)
                .shared_timing
                .is_none()
        );
    }

    #[test]
    fn counts_only_new_bytes_across_overlapping_and_adjacent_file_ranges() {
        let mut ranges = SourceRanges::default();
        for (file, range, expected) in [
            (0, 20..30, 10),
            (0, 0..10, 10),
            (0, 5..25, 10),
            (0, 0..30, 0),
            (1, 0..30, 30),
            (0, 30..40, 10),
            (0, 15..20, 0),
        ] {
            assert_eq!(ranges.insert(file, range), expected);
        }
        assert_eq!(ranges.0.len(), 2);
    }
}
