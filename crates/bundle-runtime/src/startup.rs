//! Startup timing lines for a bundle's create, in the host's own form:
//! `startup <stage>: <seconds>s` on standard error.

use std::time::{Duration, Instant};

/// Prints one stage's time.
pub fn report(stage: &str, elapsed: Duration) {
    eprintln!("startup {stage}: {:.3}s", elapsed.as_secs_f64());
}

/// Times consecutive stages: each [`Stages::end`] reports the time since the
/// previous one (or since [`Stages::start`]).
pub struct Stages {
    last: Instant,
}

impl Stages {
    #[must_use]
    pub fn start() -> Self {
        Self {
            last: Instant::now(),
        }
    }

    /// Reports the stage that ends now and starts the next.
    pub fn end(&mut self, stage: &str) {
        self.end_excluding(stage, Duration::ZERO);
    }

    /// Reports the stage that ends now, less `excluded`: time within it that
    /// is reported as a stage of its own.
    pub fn end_excluding(&mut self, stage: &str, excluded: Duration) {
        let now = Instant::now();
        report(stage, (now - self.last).saturating_sub(excluded));
        self.last = now;
    }
}
