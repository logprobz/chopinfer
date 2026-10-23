//! Recent block timings reject isolated latency outliers before averaging.

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct BlockStep {
    pub(super) micros: f64,
    pub(super) sequences: f64,
    samples: [(f64, f64); 3],
    count: usize,
    next: usize,
}

impl BlockStep {
    pub(super) fn new(micros: f64, sequences: f64) -> Self {
        let mut samples = [(0.0, 0.0); 3];
        samples[0] = (micros, sequences);
        Self {
            micros,
            sequences,
            samples,
            count: 1,
            next: 1,
        }
    }

    /// Three samples back the average, so its median rejects an outlier.
    pub(super) const fn filtered(&self) -> bool {
        self.count == self.samples.len()
    }

    pub(super) fn record(&mut self, micros: f64, sequences: f64, weight: f64) {
        self.samples[self.next] = (micros, sequences);
        self.next = (self.next + 1) % self.samples.len();
        self.count = (self.count + 1).min(self.samples.len());
        let mut samples = self.samples;
        samples[..self.count].sort_by(|a, b| a.0.total_cmp(&b.0));
        // Before three samples exist, the lower of two avoids extending a
        // startup stall. Keep the sequence count paired with its timing.
        let (micros, sequences) = samples[(self.count - 1) / 2];
        self.micros += weight * (micros - self.micros);
        self.sequences += weight * (sequences - self.sequences);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolated_high_and_low_outliers_do_not_replace_warmed_costs() {
        for outlier in [1.0, 600_000.0] {
            let mut step = BlockStep::new(60_000.0, 8.0);
            for _ in 0..2 {
                step.record(60_000.0, 8.0, 1.0);
            }
            step.record(outlier, 1.0, 1.0);
            assert_eq!((step.micros, step.sequences), (60_000.0, 8.0));
            step.record(60_000.0, 8.0, 1.0);
            assert_eq!((step.micros, step.sequences), (60_000.0, 8.0));
        }
    }

    #[test]
    fn two_realized_samples_refresh_a_stale_cost_in_either_direction() {
        for (old, new) in [(200_000.0, 60_000.0), (60_000.0, 200_000.0)] {
            let mut step = BlockStep::new(old, 4.0);
            for _ in 0..2 {
                step.record(old, 4.0, 1.0);
            }
            step.record(new, 16.0, 1.0);
            assert_eq!((step.micros, step.sequences), (old, 4.0));
            step.record(new, 16.0, 1.0);
            assert_eq!((step.micros, step.sequences), (new, 16.0));
        }
    }
}
