//! A two-lane list scheduler: a step's stages split into parts (token
//! ranges of one chunk) and run as blocks on a compute and a communication
//! stream, ordered by events.
//!
//! A bundle groups its stages into a [`Program`]: stages that run once for
//! the whole chunk, a per-part prologue, a list of [`Block`]s each on one
//! lane, and an epilogue. Every part runs the blocks in list order. A block
//! on another lane than the one before waits for that block's mark of the
//! same part, so one part's blocks keep their unpipelined order while
//! another part's work fills the other lane:
//!
//! ```text
//! compute: B0.0 B0.1          B2.0 B2.1 ...
//! comm:              B1.0 B1.1          B3.0 B3.1 ...
//! ```
//!
//! The list is any sequence of blocks: nothing assumes a fixed number of
//! blocks per layer, so a layer with an all-reduce inside attention adds a
//! block. Which stage belongs to which block, and which grids the lanes
//! may shrink to share SMs, is classified by the bundle. [`Settings`] and
//! [`adjust_grid`] apply bundle-supplied policy. [`execute`] walks the ops
//! through a bundle adapter; [`drain`] handles event-independent recovery.

use kernel_core::backend::Launch;
use runtime_bundle_abi::{
    BundleAbiError, BundleContextHandleV1, BundleEventHandleV1, BundleStreamHandleV1,
    HostServicesV1,
};

use crate::resources::Held;

/// The stream a launch runs on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Lane {
    Compute,
    Comm,
}

/// The event that marks a block of one part done. Blocks that share a
/// `mark` class reuse its events: the class's next block of the same part
/// runs only after every wait on the previous one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Mark {
    pub class: usize,
    pub part: usize,
}

impl Mark {
    /// The event of this mark among `parts` parts: classes major.
    pub const fn index(self, parts: usize) -> usize {
        self.class * parts + self.part
    }
}

/// Consecutive stages of one part on one lane, marked with events of
/// `mark` class.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Block<S> {
    pub lane: Lane,
    pub mark: usize,
    pub stages: Vec<S>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Op<S> {
    /// Launch `stage` for a part, or for the whole chunk when `part` is
    /// None.
    Launch {
        stage: S,
        part: Option<usize>,
        lane: Lane,
    },
    Record(Lane, Mark),
    Wait(Lane, Mark),
}

/// A step's stages as the scheduler orders them.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Program<S> {
    /// Run once for the whole chunk, first, on the compute lane.
    pub whole: Vec<S>,
    /// Run for each part in turn, on the compute lane.
    pub prologue: Vec<S>,
    pub blocks: Vec<Block<S>>,
    /// Run once for the whole chunk, last, on the compute lane, after every
    /// part's last block.
    pub epilogue: Vec<S>,
}

impl<S: Copy> Program<S> {
    /// Events per lane-crossing mark class, for `parts` parts: the events a
    /// rank holds for [`Mark::index`].
    pub fn events(&self, parts: usize) -> usize {
        self.blocks
            .iter()
            .map(|block| block.mark + 1)
            .max()
            .unwrap_or(0)
            * parts
    }

    /// The ops of this program over `parts` parts.
    pub fn schedule(&self, parts: usize) -> Vec<Op<S>> {
        let mut ops = Vec::new();
        let launch = |ops: &mut Vec<Op<S>>, stage, part, lane| {
            ops.push(Op::Launch { stage, part, lane });
        };
        for &stage in &self.whole {
            launch(&mut ops, stage, None, Lane::Compute);
        }
        for part in 0..parts {
            for &stage in &self.prologue {
                launch(&mut ops, stage, Some(part), Lane::Compute);
            }
        }
        // The lane each block hands over to, and the epilogue's.
        let next_lanes = self
            .blocks
            .iter()
            .skip(1)
            .map(|block| block.lane)
            .chain([Lane::Compute]);
        let mut previous: Option<&Block<S>> = None;
        for (block, next) in self.blocks.iter().zip(next_lanes) {
            for part in 0..parts {
                if let Some(before) = previous
                    && before.lane != block.lane
                {
                    let mark = Mark {
                        class: before.mark,
                        part,
                    };
                    ops.push(Op::Wait(block.lane, mark));
                }
                for &stage in &block.stages {
                    launch(&mut ops, stage, Some(part), block.lane);
                }
                if next != block.lane {
                    let mark = Mark {
                        class: block.mark,
                        part,
                    };
                    ops.push(Op::Record(block.lane, mark));
                }
            }
            previous = Some(block);
        }
        if let Some(last) = previous
            && last.lane != Lane::Compute
        {
            for part in 0..parts {
                let mark = Mark {
                    class: last.mark,
                    part,
                };
                ops.push(Op::Wait(Lane::Compute, mark));
            }
        }
        for &stage in &self.epilogue {
            launch(&mut ops, stage, None, Lane::Compute);
        }
        ops
    }
}

/// Bundle-owned names and defaults for overlap configuration.
#[derive(Clone, Copy, Debug)]
pub struct Settings<'a> {
    pub enable: &'a str,
    pub reserved_sms: &'a str,
    pub comm_ctas: &'a str,
    pub min_tokens: &'a str,
    pub default_reserved_sms: u32,
    pub default_comm_ctas: u32,
    pub default_min_tokens: u32,
    pub sms: u32,
    pub comm_cluster: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub sms: u32,
    pub reserved_sms: u32,
    pub comm_ctas: Option<u32>,
    pub min_tokens: u32,
}

impl Settings<'_> {
    pub fn from_env(self) -> Option<Config> {
        self.read(|name| std::env::var(name).ok())
    }

    /// [`Self::read`] behind the bundle's switch: `enable` exactly `0` turns
    /// the lanes off and exactly `1` on; unset or any other value (empty,
    /// `true`, `2`, padded with whitespace) keeps `default_on`.
    pub fn read_switched(
        self,
        default_on: bool,
        mut value: impl FnMut(&str) -> Option<String>,
    ) -> Option<Config> {
        let on = match value(self.enable).as_deref() {
            Some("0") => false,
            Some("1") => true,
            _ => default_on,
        };
        if !on {
            return None;
        }
        let enable = self.enable;
        self.read(|name| if name == enable { None } else { value(name) })
    }

    /// Reads through a supplied lookup so callers can test without changing
    /// process-wide environment variables. Zero SM or cluster counts reject
    /// the settings.
    pub fn read(self, mut value: impl FnMut(&str) -> Option<String>) -> Option<Config> {
        if self.sms == 0 || self.comm_cluster == 0 || value(self.enable).as_deref() == Some("0") {
            return None;
        }
        let mut number = |name: &str| value(name)?.parse::<u32>().ok();
        Some(Config {
            sms: self.sms,
            reserved_sms: number(self.reserved_sms)
                .unwrap_or(self.default_reserved_sms)
                .min(self.sms - 1),
            comm_ctas: Some(number(self.comm_ctas).unwrap_or(self.default_comm_ctas))
                .filter(|&ctas| ctas > self.comm_cluster && ctas % self.comm_cluster == 0),
            min_tokens: number(self.min_tokens).unwrap_or(self.default_min_tokens),
        })
    }
}

/// Symbol classification belongs to the bundle that owns the launch.
#[derive(Clone, Copy, Debug, Default)]
pub struct LaunchFacts {
    pub persistent: bool,
    pub phased_comm: bool,
    /// The launch is a collective that waits for its peers (the op's
    /// `Launch::collective`): eager runs queue the ranks' programs
    /// collective by collective (`pipeline::run_ranks`).
    pub collective: bool,
}

impl LaunchFacts {
    /// The facts of an op's `launches`, in order: each launch's persistence
    /// and whether it is a collective as the op states them, and the phased
    /// pair, an op of exactly two collective launches (scatter, then
    /// gather).
    pub fn of_op(launches: &[Launch]) -> Vec<Self> {
        let phased = launches.len() == 2 && launches.iter().all(|launch| launch.collective);
        launches
            .iter()
            .map(|launch| Self {
                persistent: launch.persistent,
                phased_comm: phased,
                collective: launch.collective,
            })
            .collect()
    }
}

/// One rank's second stream and the events that order it against the
/// rank's compute stream.
pub struct SideLane {
    pub stream: BundleStreamHandleV1,
    pub events: Vec<BundleEventHandleV1>,
}

impl SideLane {
    /// Creates the stream, then `events` events, in `context`, each pushed
    /// to `held` once it exists.
    pub fn create(
        services: &HostServicesV1,
        context: BundleContextHandleV1,
        events: usize,
        held: &mut Vec<Held>,
    ) -> Result<Self, BundleAbiError> {
        let stream = services.create_stream(context, 1)?;
        held.push(Held::Stream(stream));
        let events = (0..events)
            .map(|_| {
                let event = services.create_event(context)?;
                held.push(Held::Event(event));
                Ok(event)
            })
            .collect::<Result<_, BundleAbiError>>()?;
        Ok(Self { stream, events })
    }
}

pub fn adjust_grid(grid: &mut [u32; 3], lane: Lane, config: &Config, facts: LaunchFacts) {
    match lane {
        Lane::Compute if facts.persistent && config.sms > 0 && *grid == [config.sms, 1, 1] => {
            if config.reserved_sms > 0 {
                grid[0] = config.sms.saturating_sub(config.reserved_sms).max(1);
            }
        }
        Lane::Comm if facts.phased_comm => {
            if let Some(ctas) = config.comm_ctas {
                grid[0] = ctas;
            }
        }
        _ => {}
    }
}

/// Bundle adapter: lower every rank before submitting any rank, and keep
/// collective bookkeeping and taps with the stage that owns them.
pub trait Executor<S> {
    type Error;

    fn launch(&mut self, stage: S, part: Option<usize>, lane: Lane) -> Result<(), Self::Error>;
    fn record(&mut self, lane: Lane, mark: Mark) -> Result<(), Self::Error>;
    fn wait(&mut self, lane: Lane, mark: Mark) -> Result<(), Self::Error>;
}

/// Stop at the first failed operation. The caller drains both lanes before
/// rollback, including when input preparation fails before this walk.
pub fn execute<S, E: Executor<S>>(
    ops: impl IntoIterator<Item = Op<S>>,
    executor: &mut E,
) -> Result<(), E::Error> {
    for op in ops {
        match op {
            Op::Launch { stage, part, lane } => executor.launch(stage, part, lane)?,
            Op::Record(lane, mark) => executor.record(lane, mark)?,
            Op::Wait(lane, mark) => executor.wait(lane, mark)?,
        }
    }
    Ok(())
}

/// Drain every communication stream before any compute stream, without
/// relying on events from a failed schedule. Stop and poison at the first
/// synchronization failure; leave an already poisoned instance alone.
pub fn drain<T, E>(
    poisoned: bool,
    comm_streams: impl IntoIterator<Item = T>,
    compute_streams: impl IntoIterator<Item = T>,
    mut synchronize: impl FnMut(T) -> Result<(), E>,
    mut poison: impl FnMut(E),
) {
    if poisoned {
        return;
    }
    for stream in comm_streams.into_iter().chain(compute_streams) {
        if let Err(error) = synchronize(stream) {
            poison(error);
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use kernel_core::contract::{ArgumentList, LaunchPlan};

    use super::*;

    fn op_launch(collective: bool, persistent: bool) -> Launch {
        Launch {
            symbol: "op".into(),
            rows: 1,
            plan: LaunchPlan {
                grid: [1, 1, 1],
                block: [384, 1, 1],
                dynamic_shared_bytes: 0,
                workspace_bytes: 0,
                cluster: [1, 1, 1],
                pdl: true,
            },
            arguments: ArgumentList(Vec::new()),
            collective,
            persistent,
        }
    }

    /// An op's launches are persistent and collective as it says, and
    /// phased only as the two collective launches of a phased all-reduce.
    #[test]
    fn op_facts_follow_their_launches() {
        let facts = LaunchFacts::of_op(&[op_launch(false, true), op_launch(false, false)]);
        assert_eq!(
            facts
                .iter()
                .map(|f| (f.persistent, f.phased_comm, f.collective))
                .collect::<Vec<_>>(),
            [(true, false, false), (false, false, false)]
        );
        let pair = LaunchFacts::of_op(&[op_launch(true, false), op_launch(true, false)]);
        assert!(
            pair.iter()
                .all(|f| f.phased_comm && f.collective && !f.persistent)
        );
        let one = LaunchFacts::of_op(&[op_launch(true, false)])[0];
        assert!(one.collective && !one.phased_comm);
    }

    /// The switch turns the lanes on or off only at exactly `1` or `0`.
    #[test]
    fn the_switch_overrides_the_default_only_when_exact() {
        let read = |default_on, enable: Option<&str>| {
            settings()
                .read_switched(default_on, |name| {
                    (name == "ENABLE")
                        .then(|| enable.map(String::from))
                        .flatten()
                })
                .is_some()
        };
        for default_on in [false, true] {
            assert_eq!(read(default_on, None), default_on);
            assert!(read(default_on, Some("1")));
            assert!(!read(default_on, Some("0")));
            for other in ["", "true", "2", " 1"] {
                assert_eq!(read(default_on, Some(other)), default_on, "{other:?}");
            }
        }
    }

    fn settings() -> Settings<'static> {
        Settings {
            enable: "ENABLE",
            reserved_sms: "RESERVED",
            comm_ctas: "COMM",
            min_tokens: "MIN",
            default_reserved_sms: 16,
            default_comm_ctas: 24,
            default_min_tokens: 1024,
            sms: 188,
            comm_cluster: 8,
        }
    }

    #[test]
    fn configuration_preserves_defaults_and_invalid_override_policy() {
        let read = |values: &[(&str, &str)]| {
            settings().read(|name| {
                values
                    .iter()
                    .find(|(key, _)| *key == name)
                    .map(|(_, value)| (*value).into())
            })
        };
        let defaults = Config {
            sms: 188,
            reserved_sms: 16,
            comm_ctas: Some(24),
            min_tokens: 1024,
        };
        assert_eq!(read(&[]), Some(defaults));
        assert_eq!(read(&[("ENABLE", "0")]), None);
        assert_eq!(
            read(&[("ENABLE", "false"), ("RESERVED", "bad"), ("COMM", "bad")]),
            Some(defaults)
        );
        assert_eq!(
            read(&[("RESERVED", "999"), ("MIN", "0")]),
            Some(Config {
                reserved_sms: 187,
                min_tokens: 0,
                ..defaults
            })
        );
        for ctas in ["0", "8", "9", "23"] {
            assert_eq!(read(&[("COMM", ctas)]).unwrap().comm_ctas, None);
        }
    }

    #[test]
    fn grid_policy_requires_bundle_facts_and_full_persistent_grid() {
        let config = settings().read(|_| None).unwrap();
        for (lane, facts, original, expected) in [
            (
                Lane::Compute,
                LaunchFacts {
                    persistent: true,
                    phased_comm: false,
                    collective: false,
                },
                [188, 1, 1],
                [172, 1, 1],
            ),
            (
                Lane::Compute,
                LaunchFacts {
                    persistent: true,
                    phased_comm: false,
                    collective: false,
                },
                [376, 1, 1],
                [376, 1, 1],
            ),
            (
                Lane::Compute,
                LaunchFacts::default(),
                [188, 1, 1],
                [188, 1, 1],
            ),
            (
                Lane::Comm,
                LaunchFacts {
                    persistent: false,
                    phased_comm: true,
                    collective: true,
                },
                [64, 1, 1],
                [24, 1, 1],
            ),
            (Lane::Comm, LaunchFacts::default(), [64, 1, 1], [64, 1, 1]),
        ] {
            let mut grid = original;
            adjust_grid(&mut grid, lane, &config, facts);
            assert_eq!(grid, expected);
        }
    }

    #[derive(Default)]
    struct Fake {
        ops: Vec<Op<u32>>,
        fail_at: Option<usize>,
    }

    impl Fake {
        fn push(&mut self, op: Op<u32>) -> Result<(), usize> {
            let index = self.ops.len();
            self.ops.push(op);
            if self.fail_at == Some(index) {
                Err(index)
            } else {
                Ok(())
            }
        }
    }

    impl Executor<u32> for Fake {
        type Error = usize;
        fn launch(&mut self, stage: u32, part: Option<usize>, lane: Lane) -> Result<(), usize> {
            self.push(Op::Launch { stage, part, lane })
        }
        fn record(&mut self, lane: Lane, mark: Mark) -> Result<(), usize> {
            self.push(Op::Record(lane, mark))
        }
        fn wait(&mut self, lane: Lane, mark: Mark) -> Result<(), usize> {
            self.push(Op::Wait(lane, mark))
        }
    }

    #[test]
    fn executor_stops_at_each_launch_record_or_wait_failure() {
        let ops = Program {
            whole: vec![1],
            prologue: vec![],
            blocks: vec![block(Lane::Compute, 0, &[2]), block(Lane::Comm, 1, &[3])],
            epilogue: vec![4],
        }
        .schedule(2);
        let mut fake = Fake::default();
        assert_eq!(execute(ops.clone(), &mut fake), Ok(()));
        assert_eq!(fake.ops, ops);
        for failed in 0..ops.len() {
            let mut fake = Fake {
                fail_at: Some(failed),
                ..Fake::default()
            };
            assert_eq!(execute(ops.clone(), &mut fake), Err(failed));
            assert_eq!(fake.ops, ops[..=failed]);
        }
    }

    #[test]
    fn drain_orders_lanes_stops_on_failure_and_skips_poisoned_instances() {
        let expected = [2, 3, 0, 1];
        for failed in [None, Some(2), Some(3), Some(0), Some(1)] {
            let mut drained = Vec::new();
            let mut poisoned = None;
            drain(
                false,
                [2, 3],
                [0, 1],
                |stream| {
                    drained.push(stream);
                    if Some(stream) == failed {
                        Err(stream)
                    } else {
                        Ok(())
                    }
                },
                |error| poisoned = Some(error),
            );
            let length = failed.map_or(expected.len(), |stream| {
                expected
                    .iter()
                    .position(|&candidate| candidate == stream)
                    .unwrap()
                    + 1
            });
            assert_eq!(drained, expected[..length]);
            assert_eq!(poisoned, failed);
        }
        drain(
            true,
            [2, 3],
            [0, 1],
            |_| -> Result<(), ()> { panic!("poisoned instance synchronized") },
            |_| panic!("poisoned twice"),
        );
    }

    #[test]
    fn zero_sms_is_rejected_without_panicking() {
        assert_eq!(
            Settings {
                sms: 0,
                ..settings()
            }
            .read(|_| None),
            None
        );
    }

    #[test]
    fn zero_comm_cluster_is_rejected_without_panicking() {
        assert_eq!(
            Settings {
                comm_cluster: 0,
                ..settings()
            }
            .read(|_| None),
            None
        );
    }

    #[test]
    fn grid_adjustment_uses_config_sms_and_clamps_excessive_reservations() {
        let facts = LaunchFacts {
            persistent: true,
            phased_comm: false,
            collective: false,
        };
        let defaults = settings().read(|_| None).unwrap();
        for (sms, reserved_sms, original, expected) in [
            (8, 16, [8, 1, 1], [1, 1, 1]),
            (8, 8, [8, 1, 1], [1, 1, 1]),
            (8, u32::MAX, [8, 1, 1], [1, 1, 1]),
            (8, 0, [8, 1, 1], [8, 1, 1]),
            (8, 4, [188, 1, 1], [188, 1, 1]),
            (0, 16, [0, 1, 1], [0, 1, 1]),
        ] {
            let config = Config {
                sms,
                reserved_sms,
                ..defaults
            };
            let mut grid = original;
            adjust_grid(&mut grid, Lane::Compute, &config, facts);
            assert_eq!(grid, expected);
        }
        let config = Settings {
            sms: 1,
            ..settings()
        }
        .read(|_| None)
        .unwrap();
        assert_eq!(config.reserved_sms, 0);
    }

    fn block(lane: Lane, mark: usize, stages: &[u32]) -> Block<u32> {
        Block {
            lane,
            mark,
            stages: stages.to_vec(),
        }
    }

    fn mark(class: usize, part: usize) -> Mark {
        Mark { class, part }
    }

    #[test]
    fn lane_changes_wait_on_the_same_parts_previous_block() {
        let program = Program {
            whole: vec![1],
            prologue: vec![2],
            blocks: vec![
                block(Lane::Compute, 0, &[10]),
                block(Lane::Compute, 0, &[11]),
                block(Lane::Comm, 1, &[12]),
            ],
            epilogue: vec![3],
        };
        let launch = |stage, part, lane| Op::Launch { stage, part, lane };
        assert_eq!(
            program.schedule(2),
            [
                launch(1, None, Lane::Compute),
                launch(2, Some(0), Lane::Compute),
                launch(2, Some(1), Lane::Compute),
                launch(10, Some(0), Lane::Compute),
                launch(10, Some(1), Lane::Compute),
                launch(11, Some(0), Lane::Compute),
                Op::Record(Lane::Compute, mark(0, 0)),
                launch(11, Some(1), Lane::Compute),
                Op::Record(Lane::Compute, mark(0, 1)),
                Op::Wait(Lane::Comm, mark(0, 0)),
                launch(12, Some(0), Lane::Comm),
                Op::Record(Lane::Comm, mark(1, 0)),
                Op::Wait(Lane::Comm, mark(0, 1)),
                launch(12, Some(1), Lane::Comm),
                Op::Record(Lane::Comm, mark(1, 1)),
                Op::Wait(Lane::Compute, mark(1, 0)),
                Op::Wait(Lane::Compute, mark(1, 1)),
                launch(3, None, Lane::Compute),
            ]
        );
        assert_eq!(program.events(2), 4);
        assert_eq!(mark(1, 1).index(2), 3);
    }

    #[test]
    fn a_compute_only_program_runs_in_order_without_events() {
        let program = Program {
            whole: Vec::new(),
            prologue: Vec::new(),
            blocks: vec![block(Lane::Compute, 0, &[1, 2])],
            epilogue: vec![3],
        };
        assert!(
            program
                .schedule(1)
                .iter()
                .all(|op| matches!(op, Op::Launch { .. }))
        );
    }
}
