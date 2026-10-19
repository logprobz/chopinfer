//! Run the shared expert beside the routed experts inside decode and verify
//! graphs. The input quantization precedes the fork, and the routed sum
//! follows the join. Each stream retains its original arithmetic and buffers.
//! Capture pieces contain whole fork/join regions. Eager runs remain serial.

use std::collections::BTreeMap;
use std::ops::Range;

use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::pipeline::PreparedLaunch;
use runtime_bundle_abi::{
    BundleCudaSymbolHandleV1, BundleEventHandleV1, BundleStreamHandleV1, HostServicesV1,
};

use crate::program::Stage;
use bundle_runtime::pipeline::launch_all;

/// Set to zero to retain serial graph captures. Positive values limit rows.
#[cfg_attr(test, allow(dead_code))]
pub const SWITCH: &str = "CHOPINFER_DEEPSEEK41_SHARED_OVERLAP";

/// Which graph buckets overlap, read at create (which creates each rank's
/// side stream and events when any does).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Settings {
    /// The largest bucket's rows that overlap; 0 for none.
    pub max_rows: u32,
}

impl Settings {
    /// Every bucket on one stream.
    pub const OFF: Self = Self { max_rows: 0 };
    /// Every bucket overlapped, the default.
    pub const ALL: Self = Self { max_rows: u32::MAX };

    /// The settings [`SWITCH`]'s `value` names.
    pub fn read(value: Option<&str>) -> Self {
        match value.map(str::parse::<u32>) {
            Some(Ok(0)) => Self::OFF,
            Some(Ok(max_rows)) => Self { max_rows },
            None | Some(Err(_)) => Self::ALL,
        }
    }

    /// The process environment's settings; a test's instances take the
    /// default (not the process environment) unless it sets theirs within
    /// [`with_settings_in_tests`].
    pub fn from_env() -> Self {
        #[cfg(test)]
        return IN_TESTS.get().unwrap_or_else(|| Self::read(None));
        #[cfg(not(test))]
        Self::read(std::env::var(SWITCH).ok().as_deref())
    }

    /// Whether any bucket overlaps.
    pub fn on(self) -> bool {
        self.max_rows > 0
    }

    /// Whether a graph bucket of `rows` rows overlaps.
    pub fn serves(self, rows: u32) -> bool {
        self.on() && rows <= self.max_rows
    }
}

#[cfg(test)]
thread_local! {
    static IN_TESTS: std::cell::Cell<Option<Settings>> = const { std::cell::Cell::new(None) };
}

/// Runs `create` with `settings`, as [`SWITCH`] would set them.
#[cfg(test)]
pub(crate) fn with_settings_in_tests<T>(settings: Settings, create: impl FnOnce() -> T) -> T {
    IN_TESTS.set(Some(settings));
    let created = create();
    IN_TESTS.set(None);
    created
}

/// Events each rank creates beside its side stream: the fork and the join.
pub const EVENTS: usize = 2;

/// One rank's streams and events of the overlap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Lanes {
    /// The rank's stream, which captures and replays its graphs.
    pub compute: BundleStreamHandleV1,
    /// The stream the shared expert records on inside a capture.
    pub side: BundleStreamHandleV1,
    pub fork: BundleEventHandleV1,
    pub join: BundleEventHandleV1,
}

/// One submission of a capture, over launch indices of the rank's program.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Op {
    /// These launches on the compute stream.
    Compute(Range<usize>),
    /// The side stream waits for the compute stream's work so far.
    Fork,
    /// These launches on the side stream.
    Side(Range<usize>),
    /// The compute stream waits for the side stream's work so far.
    Join,
}

/// A program's submissions in units the capture's graph pieces never split:
/// each launch outside an overlapped MoE sublayer alone, and each such
/// sublayer's routed work and shared expert between their fork and join as
/// one. `stages` names each launch's stage.
pub fn units(stages: &[Stage]) -> Result<Vec<Vec<Op>>, String> {
    let mut units = Vec::new();
    let mut at = 0;
    while at < stages.len() {
        let Stage::Project(layer, "moe.router") = stages[at] else {
            units.push(vec![Op::Compute(at..at + 1)]);
            at += 1;
            continue;
        };
        let start = at;
        let mut shared = None;
        for stage in crate::program::moe(layer).into_iter().take(7) {
            let count = stages[at..].iter().take_while(|&&s| s == stage).count();
            if count == 0 {
                return Err(format!(
                    "layer {layer} is missing {stage:?} before its routed sum"
                ));
            }
            if stage == Stage::SharedGateUp(layer) {
                shared = Some(at);
            }
            at += count;
        }
        if stages.get(at) != Some(&Stage::RoutedSum(layer)) {
            return Err(format!(
                "layer {layer} has no routed sum after its shared down"
            ));
        }
        let shared = shared.expect("shared gate/up was visited");
        units.push(vec![
            Op::Fork,
            Op::Side(shared..at),
            Op::Compute(start..shared),
            Op::Join,
        ]);
    }
    Ok(units)
}

impl Lanes {
    /// Issues `unit`'s submissions of `program`. Once the fork reached the
    /// side stream, the join is issued last even after a failed submission,
    /// so the side stream never stays in an open capture; the first error
    /// is returned.
    pub fn issue(
        &self,
        services: &HostServicesV1,
        symbols: &BTreeMap<String, BundleCudaSymbolHandleV1>,
        program: &[PreparedLaunch],
        unit: &[Op],
    ) -> Result<(), DispatchError> {
        let mut forked = false;
        let mut issued = Ok(());
        for op in unit {
            issued = match op {
                Op::Compute(range) => {
                    launch_all(services, self.compute, symbols, &program[range.clone()])
                }
                Op::Side(range) => {
                    launch_all(services, self.side, symbols, &program[range.clone()])
                }
                Op::Fork => {
                    let recorded = services.record_event(self.fork, self.compute);
                    forked = recorded.is_ok();
                    recorded
                        .and_then(|()| services.wait_event(self.side, self.fork))
                        .map_err(DispatchError::from)
                }
                Op::Join => {
                    forked = false;
                    self.join(services)
                }
            };
            if issued.is_err() {
                break;
            }
        }
        if forked {
            return issued.and(self.join(services));
        }
        issued
    }

    fn join(&self, services: &HostServicesV1) -> Result<(), DispatchError> {
        services
            .record_event(self.join, self.side)
            .and_then(|()| services.wait_event(self.compute, self.join))
            .map_err(DispatchError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::{Phase, Sublayer, stages};
    use model_deepseek41_contract::geometry::FLASH;

    #[test]
    fn shared_overlap_keeps_each_launch_and_the_input_and_sum_dependencies() {
        for phase in [
            Phase::Decode(1),
            Phase::Decode(32),
            Phase::Verify {
                rows: 192,
                items: 32,
            },
        ] {
            // Include repeated tags: lowering may emit several kernels per stage.
            let stages = stages(&FLASH, phase)
                .into_iter()
                .flat_map(|s| [s, s])
                .collect::<Vec<_>>();
            let units = units(&stages).unwrap();
            let mut seen = vec![0; stages.len()];
            let mut regions = 0;
            for unit in units {
                if unit.len() == 4 {
                    regions += 1;
                    let [Op::Fork, Op::Side(side), Op::Compute(compute), Op::Join] = &unit[..]
                    else {
                        panic!("whole fork/join region")
                    };
                    let Stage::Project(layer, "moe.router") = stages[compute.start] else {
                        panic!("router starts compute")
                    };
                    assert_eq!(
                        stages[compute.start - 1],
                        Stage::QuantizeInput(layer, Sublayer::Mlp)
                    );
                    assert_eq!(stages[side.end], Stage::RoutedSum(layer));
                    assert!(stages[side.clone()].iter().all(|&s| matches!(
                        s,
                        Stage::SharedGateUp(_) | Stage::SharedActivation(_) | Stage::SharedDown(_)
                    )));
                }
                for op in unit {
                    if let Op::Compute(range) | Op::Side(range) = op {
                        for i in range {
                            seen[i] += 1;
                        }
                    }
                }
            }
            assert!(regions > 0);
            assert!(seen.iter().all(|&count| count == 1));
        }
    }

    #[test]
    fn shared_overlap_refuses_incomplete_shared_sequences() {
        let stages = crate::program::moe(0);
        for omitted in 1..stages.len() {
            let mut broken = stages.clone();
            broken.remove(omitted);
            assert!(units(&broken).is_err(), "missing {:?}", stages[omitted]);
        }
        assert!(Settings::read(None).serves(192));
        assert!(!Settings::read(Some("0")).serves(1));
        assert!(!Settings::read(Some("8")).serves(32));
    }
}
