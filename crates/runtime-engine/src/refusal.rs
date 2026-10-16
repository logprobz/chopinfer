use std::collections::VecDeque;

/// Bounded recovery from batches refused before device work. Counts are per
/// process. The rolling window uses the engine's monotonic clock.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchRefusalLimits {
    pub maximum_consecutive: u32,
    pub maximum_in_window: u32,
    pub window_seconds: u32,
}

impl BatchRefusalLimits {
    pub const DEFAULT: Self = Self {
        maximum_consecutive: 3,
        maximum_in_window: 10,
        window_seconds: 60,
    };
}

impl Default for BatchRefusalLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RefusalGuard {
    limits: BatchRefusalLimits,
    consecutive: u64,
    recent: VecDeque<u64>,
}

impl RefusalGuard {
    pub(crate) fn new(limits: BatchRefusalLimits) -> Self {
        Self {
            limits,
            consecutive: 0,
            recent: VecDeque::new(),
        }
    }

    pub(crate) fn committed(&mut self) {
        self.consecutive = 0;
    }

    pub(crate) fn refused(&mut self, now: u64) -> Option<&'static str> {
        self.consecutive = self.consecutive.saturating_add(1);
        if self.consecutive > u64::from(self.limits.maximum_consecutive) {
            return Some("batch refusal loop guard: consecutive refusal limit exceeded");
        }
        let window = u64::from(self.limits.window_seconds) * 1_000_000;
        while self
            .recent
            .front()
            .is_some_and(|time| now.saturating_sub(*time) >= window)
        {
            self.recent.pop_front();
        }
        if self.recent.len() as u64 >= u64::from(self.limits.maximum_in_window) {
            return Some("batch refusal loop guard: rolling-window refusal limit exceeded");
        }
        self.recent.push_back(now);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consecutive_refusals_stop_but_commits_reset_the_streak() {
        let mut guard = RefusalGuard::new(BatchRefusalLimits::DEFAULT);
        for _ in 0..3 {
            assert_eq!(guard.refused(0), None);
        }
        guard.committed();
        for _ in 0..3 {
            assert_eq!(guard.refused(0), None);
        }
        assert!(guard.refused(0).unwrap().contains("consecutive"));
    }

    #[test]
    fn intermittent_refusals_stop_within_the_window_and_old_ones_expire() {
        let mut guard = RefusalGuard::new(BatchRefusalLimits {
            maximum_consecutive: 3,
            maximum_in_window: 2,
            window_seconds: 60,
        });
        assert_eq!(guard.refused(0), None);
        guard.committed();
        assert_eq!(guard.refused(59_000_000), None);
        guard.committed();
        assert_eq!(guard.refused(60_000_000), None);
        guard.committed();
        assert!(
            guard
                .refused(60_000_001)
                .unwrap()
                .contains("rolling-window")
        );
        assert_eq!(guard.recent.len(), 2);
    }
}
