//! The shared expert beside the routed experts in decode and verify graphs
//! (PERF-B12: its gate/up; PERF-B24: its down projection too).
//!
//! A MoE sublayer of a decode or verify graph (`crate::program`'s MoE
//! sublayer) reads the normalized row (`Normed`) into the router, the
//! routing and the routed experts' gate/up and down (`Routed`), and into
//! the shared expert's gate/up (`DenseIntermediate`). The shared expert's
//! down projection reads `DenseIntermediate` and adds `ZeroRows` into
//! `MlpOut`; the routed sum then adds `MlpOut` into the sublayer's partial.
//! The shared expert needs nothing the routed work writes, and the routed
//! work nothing it writes, so a captured graph may run it on a second
//! stream of the rank:
//!
//! ```text
//! compute: norm | router route local gate/up down |      | routed sum
//! side:         | shared gate/up, shared down     |      |
//!            fork                                 join
//! ```
//!
//! The fork records an event on the compute stream after the norm and the
//! side stream waits for it; the join records an event on the side stream
//! after the shared down projection and the compute stream waits for it
//! before the routed sum. Every launch is the one-stream program's, with the same
//! bindings; the compute stream issues the others in the same order. A
//! graph capture records both streams into the compute stream's graph,
//! whose replays are unchanged, so the side stream does work only while a
//! capture runs. Eager runs keep one stream.
//!
//! On for every bucket by default; [`SWITCH`] `0` turns it off
//! ([`Settings::read`]).

use std::collections::BTreeMap;
use std::ops::Range;

use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::pipeline::PreparedLaunch;
use runtime_bundle_abi::{
    BundleCudaSymbolHandleV1, BundleEventHandleV1, BundleStreamHandleV1, HostServicesV1,
};

use crate::program::Stage;
use crate::submit::launch_all;

/// The graph buckets whose MoE sublayers overlap the shared expert.
/// Exactly `0` turns the overlap off (the off-ramp); a positive number
/// names the largest bucket's rows that overlap (a decode bucket's items,
/// a verify bucket's tokens), so `8` overlaps buckets of 1 to 8 rows; unset
/// or any other value overlaps every bucket, the default: PERF-B12 job 1
/// (g2-075) served it bitwise equal to one stream, AR ms/token C1 16.21 ->
/// 15.47, C4 22.0 -> 21.2, C8 25.95 -> 24.95, C16 33.6 -> 32.8, MTP C1 7.72
/// -> 7.75, C4 11.0 -> 10.6, C8 15.1 -> 14.4, C16 23.5 -> 23.2.
pub const SWITCH: &str = "CHOPINFER_GLM5NEXT_SHARED_OVERLAP";

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
        let Some(layer) = routed_layer(stages[at]) else {
            units.push(vec![Op::Compute(at..at + 1)]);
            at += 1;
            continue;
        };
        let end = at
            + stages[at..]
                .iter()
                .take_while(|&&stage| routed(stage, layer))
                .count();
        if stages.get(end) != Some(&Stage::RoutedSum(layer)) {
            return Err(format!(
                "layer {layer}'s routed experts and shared expert are not followed by its \
                 routed sum: {:?}",
                stages.get(end)
            ));
        }
        let shared = (at..end)
            .filter(|&index| {
                stages[index] == Stage::SharedGateUp(layer) || shared_down(stages[index], layer)
            })
            .collect::<Vec<_>>();
        let (Some(&first), Some(&last)) = (shared.first(), shared.last()) else {
            return Err(format!("layer {layer} runs no shared expert"));
        };
        if last + 1 - first != shared.len() {
            return Err(format!("layer {layer}'s shared expert is not contiguous"));
        }
        // Its gate/up, then its down projection.
        let gate_up = shared
            .iter()
            .take_while(|&&index| stages[index] == Stage::SharedGateUp(layer))
            .count();
        if gate_up == 0 || gate_up == shared.len() {
            return Err(format!(
                "layer {layer}'s shared expert is not its gate/up, then its down projection"
            ));
        }
        let mut unit = vec![Op::Fork, Op::Side(first..last + 1)];
        unit.extend(
            [at..first, last + 1..end]
                .into_iter()
                .filter(|range| !range.is_empty())
                .map(Op::Compute),
        );
        unit.push(Op::Join);
        units.push(unit);
        at = end;
    }
    Ok(units)
}

/// The layer of a stage that starts a MoE sublayer's routed work: its
/// router projection.
fn routed_layer(stage: Stage) -> Option<u32> {
    match stage {
        Stage::Project(layer, "moe.router" | "mtp.moe.router") => Some(layer),
        _ => None,
    }
}

/// Whether `stage` is layer `layer`'s work between its norm and its
/// routed sum: the router, the routing, the routed experts and the shared
/// expert's gate/up and down projection.
fn routed(stage: Stage, layer: u32) -> bool {
    match stage {
        Stage::Project(l, "moe.router" | "mtp.moe.router") => l == layer,
        Stage::Route(l)
        | Stage::LocalRoutes(l)
        | Stage::ExpertsGateUp(l)
        | Stage::ExpertsDown(l)
        | Stage::SharedGateUp(l) => l == layer,
        stage => shared_down(stage, layer),
    }
}

fn shared_down(stage: Stage, layer: u32) -> bool {
    matches!(
        stage,
        Stage::Project(l, "moe.shared.down" | "mtp.moe.shared.down") if l == layer
    )
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
    use crate::program::{DraftPass, Phase, drafter_stages, stages};
    use model_glm5next_contract::contract::Shape;

    #[test]
    fn the_switch_names_the_largest_overlapped_bucket() {
        assert_eq!(Settings::read(None), Settings::ALL);
        for all in ["all", "", "on", "-1", "1.5"] {
            assert_eq!(Settings::read(Some(all)), Settings::ALL, "{all}");
        }
        assert_eq!(Settings::read(Some("0")), Settings::OFF);
        assert!(!Settings::OFF.on() && !Settings::OFF.serves(1));
        assert!(Settings::ALL.serves(1) && Settings::ALL.serves(256));
        let eight = Settings::read(Some("8"));
        assert!(eight.serves(1) && eight.serves(8) && !eight.serves(16));
    }

    /// Units over each stage's index (one launch per stage).
    fn units_of(stages: &[Stage]) -> Vec<Vec<Op>> {
        units(stages).unwrap()
    }

    #[test]
    fn each_moe_sublayer_forks_after_its_norm_and_joins_before_its_routed_sum() {
        let shape = Shape::TP4;
        for phase in [Phase::Decode(8), Phase::Verify { rows: 32, items: 8 }] {
            let program = stages(shape, phase);
            let units = units_of(&program);
            // Every launch once, in program order on its stream.
            let mut compute = Vec::new();
            let mut side = Vec::new();
            let mut regions = 0;
            for unit in &units {
                if unit.len() > 1 {
                    regions += 1;
                    assert_eq!(unit.first(), Some(&Op::Fork));
                    assert_eq!(unit.last(), Some(&Op::Join));
                }
                for op in unit {
                    match op {
                        Op::Compute(range) => compute.extend(range.clone()),
                        Op::Side(range) => side.extend(range.clone()),
                        Op::Fork | Op::Join => {}
                    }
                }
            }
            assert_eq!(regions, 42);
            assert!(compute.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(side.windows(2).all(|pair| pair[0] < pair[1]));
            let mut all = compute.clone();
            all.extend(&side);
            all.sort_unstable();
            assert_eq!(all, (0..program.len()).collect::<Vec<_>>());
            // The side stream runs the shared gate/ups and down projections
            // alone; the stage before each fork is the sublayer's norm, the
            // one after each join its routed sum.
            assert!(side.iter().all(|&index| matches!(
                program[index],
                Stage::SharedGateUp(_) | Stage::Project(_, "moe.shared.down")
            )));
            assert_eq!(side.len(), 2 * 42);
            let mut at = 0;
            for unit in &units {
                let launches = unit
                    .iter()
                    .map(|op| match op {
                        Op::Compute(range) | Op::Side(range) => range.len(),
                        Op::Fork | Op::Join => 0,
                    })
                    .sum::<usize>();
                if unit.len() > 1 {
                    let layer = program[at].layer().unwrap();
                    assert_eq!(
                        program[at - 1],
                        Stage::InputNorm(layer, crate::program::Sublayer::Mlp)
                    );
                    assert_eq!(program[at], Stage::Project(layer, "moe.router"));
                    assert_eq!(program[at + launches], Stage::RoutedSum(layer));
                    assert_eq!(
                        unit[1..unit.len() - 1],
                        [Op::Side(at + 5..at + 7), Op::Compute(at..at + 5)]
                    );
                }
                at += launches;
            }
        }
    }

    #[test]
    fn the_drafters_moe_sublayer_forks_too() {
        let shape = Shape::TP4;
        let layer = shape.geometry.layers();
        for (pass, phase) in [
            (DraftPass::Extend, Phase::Verify { rows: 4, items: 1 }),
            (DraftPass::Step, Phase::Decode(1)),
        ] {
            let program = drafter_stages(shape, pass, phase);
            let regions = units_of(&program)
                .into_iter()
                .filter(|unit| unit.len() > 1)
                .collect::<Vec<_>>();
            let at = program
                .iter()
                .position(|&stage| stage == Stage::Project(layer, "mtp.moe.router"))
                .unwrap();
            assert_eq!(program[at + 7], Stage::RoutedSum(layer));
            assert_eq!(
                regions,
                [vec![
                    Op::Fork,
                    Op::Side(at + 5..at + 7),
                    Op::Compute(at..at + 5),
                    Op::Join
                ]]
            );
        }
    }

    #[test]
    fn a_program_whose_moe_sublayer_changed_shape_is_refused() {
        use Stage::*;
        let routed = [
            Project(5, "moe.router"),
            Route(5),
            LocalRoutes(5),
            ExpertsGateUp(5),
            ExpertsDown(5),
        ];
        let (gate_up, down, sum) = (SharedGateUp(5), Project(5, "moe.shared.down"), RoutedSum(5));
        let with = |tail: &[Stage]| [&routed[..], tail].concat();
        assert!(units(&with(&[gate_up, down, sum])).is_ok());
        // No routed sum after the shared expert, no shared expert, no shared
        // down projection, no shared gate/up, the down projection first.
        assert!(units(&with(&[gate_up, down])).is_err());
        assert!(units(&with(&[sum])).is_err());
        assert!(units(&with(&[gate_up, sum])).is_err());
        assert!(units(&with(&[down, sum])).is_err());
        assert!(units(&with(&[down, gate_up, sum])).is_err());
        // A prefill's order: the routed sum before the shared expert.
        assert!(units(&with(&[sum, gate_up, down])).is_err());
        // The shared expert split by routed work.
        let split = [
            Project(5, "moe.router"),
            SharedGateUp(5),
            Route(5),
            Project(5, "moe.shared.down"),
            RoutedSum(5),
        ];
        assert!(units(&split).is_err());
    }
}
