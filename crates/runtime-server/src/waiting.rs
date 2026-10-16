//! Exclusive admission stages. A guard owns each queued unit until admission
//! or cancellation; moving it changes stages under one snapshot lock.
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy)]
pub(crate) enum Stage {
    Transport,
    Intake,
    Preparation,
    Staging,
    Storage,
    Backend,
}

const STAGES: [(Stage, &str); 6] = [
    (Stage::Transport, "transport"),
    (Stage::Intake, "intake"),
    (Stage::Preparation, "preparation"),
    (Stage::Staging, "staging"),
    (Stage::Storage, "storage"),
    (Stage::Backend, "backend"),
];

#[derive(Default)]
pub(crate) struct Waiting(Mutex<[usize; 6]>);

pub(crate) struct Guard {
    waiting: Arc<Waiting>,
    stage: Stage,
    units: usize,
}

impl Waiting {
    pub(crate) fn enter(self: &Arc<Self>, stage: Stage) -> Guard {
        self.0.lock().expect("waiting counters")[stage as usize] += 1;
        Guard {
            waiting: Arc::clone(self),
            stage,
            units: 1,
        }
    }

    pub(crate) fn value(&self) -> serde_json::Value {
        let counts = *self.0.lock().expect("waiting counters");
        let total: usize = counts.iter().sum();
        let mut value = serde_json::json!({
            "total": total,
            "pre_backend": total - counts[Stage::Backend as usize],
        });
        for (stage, name) in STAGES {
            value[name] = counts[stage as usize].into();
        }
        value
    }

    pub(crate) fn prometheus(&self) -> String {
        use std::fmt::Write;
        let counts = *self.0.lock().expect("waiting counters");
        let total: usize = counts.iter().sum();
        let mut out = format!(
            "# TYPE chop_requests_waiting gauge\nchop_requests_waiting {total}\n# TYPE chop_requests_waiting_by_stage gauge\n"
        );
        for (stage, name) in STAGES {
            writeln!(
                out,
                "chop_requests_waiting_by_stage{{stage=\"{name}\"}} {}",
                counts[stage as usize]
            )
            .unwrap();
        }
        out
    }
}

impl Guard {
    pub(crate) fn move_to(&mut self, stage: Stage) {
        let mut counts = self.waiting.0.lock().expect("waiting counters");
        counts[self.stage as usize] -= self.units;
        counts[stage as usize] += self.units;
        self.stage = stage;
    }

    pub(crate) fn expand(&mut self, units: usize) {
        let mut counts = self.waiting.0.lock().expect("waiting counters");
        counts[self.stage as usize] -= self.units;
        counts[self.stage as usize] += units;
        self.units = units;
    }

    pub(crate) fn into_backend(mut self) -> Vec<Self> {
        self.move_to(Stage::Backend);
        let units = std::mem::take(&mut self.units);
        (0..units)
            .map(|_| Self {
                waiting: Arc::clone(&self.waiting),
                stage: Stage::Backend,
                units: 1,
            })
            .collect()
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        self.waiting.0.lock().expect("waiting counters")[self.stage as usize] -= self.units;
    }
}
