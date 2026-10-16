#![forbid(unsafe_code)]

//! The scenarios one conformance run drives, over one instance.
//!
//! A conformance host proves a bundle boundary by running the bundle, so every
//! scenario here is something the host program itself does through the V1 ABI:
//! commit one batch, cancel one and replay it, or let an instance go with an
//! execution still open. A scenario is named on the command line and selects
//! nothing else, and each one ends with the host's own resource table empty.
//!
//! The instance below is a fake, so these tests bound what the driver does
//! rather than what any bundle computes: which callbacks it makes, in what
//! order, what it replays, and what it refuses.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use runtime_bundle_abi::{BundleAbiError, OwnedLogitsRowV1, ResultSinkLimitsV1};
use runtime_host::{
    BundleExecutionV1, FixtureScenarioV1, HostCommandV1, drive_fixture_scenario_v1,
    parse_conformance_host_command_v1, parse_host_command_v1,
};

/// The one encoded batch every scenario drives. Its bytes select nothing.
const REQUEST: &[u8] = b"chop-batch-v1-fixture";

/// The request the fixture batch schedules.
const REQUEST_ID: u64 = 1;

/// How many resources one created instance holds before it executes anything.
const ACQUIRED: usize = 3;

/// What one fake instance did, in call order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CallV1 {
    Execute,
    Commit,
    Cancel,
    Destroy,
}

/// One instance's shared record, still readable after the instance is gone.
#[derive(Default)]
struct RecordV1 {
    calls: RefCell<Vec<CallV1>>,
    /// What the host still owns on this instance's behalf.
    live: Cell<usize>,
}

impl RecordV1 {
    fn new() -> Rc<Self> {
        let record = Rc::new(Self::default());
        record.live.set(ACQUIRED);
        record
    }

    fn calls(&self) -> Vec<CallV1> {
        self.calls.borrow().clone()
    }

    fn note(&self, call: CallV1) {
        self.calls.borrow_mut().push(call);
    }
}

/// An instance that answers the way the ABI wrapper does.
///
/// An execution is pending until it is finalized, dropping a pending one
/// cancels it, and an instance that was never destroyed destroys itself on the
/// way out. That is the whole of what the driver is allowed to rely on.
struct FakeInstanceV1 {
    record: Rc<RecordV1>,
    /// Whether a replayed batch emits a different row from the first one.
    drifts_on_replay: bool,
    /// Whether a successful release still leaves one resource behind.
    leaks_on_release: bool,
    executions: u64,
    pending: bool,
    destroyed: bool,
}

impl FakeInstanceV1 {
    fn new(record: &Rc<RecordV1>) -> Self {
        Self {
            record: Rc::clone(record),
            drifts_on_replay: false,
            leaks_on_release: false,
            executions: 0,
            pending: false,
            destroyed: false,
        }
    }

    fn drifting(record: &Rc<RecordV1>) -> Self {
        let mut instance = Self::new(record);
        instance.drifts_on_replay = true;
        instance
    }

    fn leaking(record: &Rc<RecordV1>) -> Self {
        let mut instance = Self::new(record);
        instance.leaks_on_release = true;
        instance
    }

    /// The row this execution emits, which a drifting replay changes.
    fn row(&self) -> OwnedLogitsRowV1 {
        let lane: f32 = if self.drifts_on_replay && self.executions > 1 {
            2.0
        } else {
            1.0
        };
        OwnedLogitsRowV1::new(REQUEST_ID, lane.to_le_bytes().to_vec())
    }

    fn finalize(&mut self, call: CallV1) -> Result<(), BundleAbiError> {
        self.record.note(call);
        self.pending = false;
        self.record.live.set(self.record.live.get() - 1);
        Ok(())
    }
}

impl BundleExecutionV1 for FakeInstanceV1 {
    fn execute_batch(
        &mut self,
        request: &[u8],
        limits: ResultSinkLimitsV1,
    ) -> Result<Vec<OwnedLogitsRowV1>, BundleAbiError> {
        assert_eq!(request, REQUEST, "every execution drives the same batch");
        assert_eq!(limits.maximum_rows, 1, "one row is scheduled per execution");
        self.executions += 1;
        self.pending = true;
        self.record.note(CallV1::Execute);
        // An open execution is one more resource the host owns.
        self.record.live.set(self.record.live.get() + 1);
        Ok(vec![self.row()])
    }

    fn commit(&mut self) -> Result<(), BundleAbiError> {
        self.finalize(CallV1::Commit)
    }

    fn cancel(&mut self) -> Result<(), BundleAbiError> {
        self.finalize(CallV1::Cancel)
    }

    fn destroy(&mut self) -> Result<(), BundleAbiError> {
        self.record.note(CallV1::Destroy);
        self.destroyed = true;
        self.record.live.set(usize::from(self.leaks_on_release));
        Ok(())
    }

    fn is_poisoned(&self) -> bool {
        false
    }
}

impl Drop for FakeInstanceV1 {
    fn drop(&mut self) {
        if self.pending {
            let _ = self.cancel();
        }
        if !self.destroyed {
            let _ = self.destroy();
        }
    }
}

/// Drives one scenario and reads the record the instance left behind.
fn driven(
    instance: FakeInstanceV1,
    record: &Rc<RecordV1>,
    scenario: FixtureScenarioV1,
) -> Result<(Vec<CallV1>, usize, usize), String> {
    let counted = Rc::clone(record);
    let report = drive_fixture_scenario_v1(instance, REQUEST, scenario, move || counted.live.get())
        .map_err(|error| error.to_string())?;
    assert_eq!(report.live_resources, 0, "a finished scenario owns nothing");
    Ok((record.calls(), report.executions, report.rows))
}

fn arguments(words: &[&str]) -> Vec<String> {
    words.iter().map(|word| (*word).to_owned()).collect()
}

fn scenario_of(words: &[&str]) -> FixtureScenarioV1 {
    match parse_conformance_host_command_v1(&arguments(words)) {
        Ok(HostCommandV1::ConformanceRun { request, .. }) => request.scenario,
        Ok(command) => panic!("this invocation names a conformance run, it parsed as {command:?}"),
        Err(error) => panic!("this invocation must parse: {error}"),
    }
}

/// Only the argument that names a scenario selects one.
#[test]
fn a_scenario_is_named_explicitly_and_nothing_else_selects_one() {
    let bare = arguments(&["--bundle", "b", "--fixture-request", "run"]);
    let named = |scenario: &str| {
        let mut words = vec!["--bundle", "b", "--fixture-request", "run", "--scenario"];
        words.push(scenario);
        scenario_of(&words)
    };

    // A run that names no scenario commits, which is what a positive run is.
    match parse_conformance_host_command_v1(&bare) {
        Ok(HostCommandV1::ConformanceRun { request, .. }) => {
            assert_eq!(request.scenario, FixtureScenarioV1::Commit);
        }
        other => panic!("a bare flag list is a conformance run, and parsed as {other:?}"),
    }

    assert_eq!(named("commit"), FixtureScenarioV1::Commit);
    assert_eq!(named("cancel"), FixtureScenarioV1::Cancel);
    assert_eq!(named("drop"), FixtureScenarioV1::Drop);

    for (words, expected) in [
        (
            vec![
                "conformance-run",
                "--bundle",
                "b",
                "--fixture-request",
                "run",
                "--scenario",
                "rollback",
            ],
            "--scenario takes one of commit, cancel, drop",
        ),
        (
            vec![
                "conformance-run",
                "--bundle",
                "b",
                "--fixture-request",
                "run",
                "--scenario",
                "commit",
                "--scenario",
                "cancel",
            ],
            "--scenario is given more than once",
        ),
        (
            vec!["serve", "--bundle", "b", "--scenario", "cancel"],
            "serve does not accept --scenario",
        ),
    ] {
        let refusal = parse_host_command_v1(&arguments(&words))
            .err()
            .unwrap_or_else(|| panic!("{words:?} must be refused"))
            .to_string();
        assert!(refusal.contains(expected), "{refusal}");
    }
}

/// A committed run executes once, commits, and gives everything back.
#[test]
fn the_commit_scenario_commits_one_batch_and_releases_the_instance() {
    let record = RecordV1::new();

    let (calls, executions, rows) = driven(
        FakeInstanceV1::new(&record),
        &record,
        FixtureScenarioV1::Commit,
    )
    .expect("a committed batch finishes");

    assert_eq!(
        calls,
        vec![CallV1::Execute, CallV1::Commit, CallV1::Destroy],
        "a committed run commits the batch it executed, then destroys"
    );
    assert_eq!(executions, 1);
    assert_eq!(rows, 1);
}

/// A cancelled run replays the same batch and requires the same row back.
///
/// The replay is the whole assertion. A bundle that rolled the cancelled
/// execution back is one whose next run of the same bytes produces what the
/// cancelled one produced; a bundle that kept part of it produces something
/// else. Neither claim needs the host to know what the numbers mean.
#[test]
fn the_cancel_scenario_replays_the_cancelled_batch_and_requires_the_same_row() {
    let record = RecordV1::new();

    let (calls, executions, rows) = driven(
        FakeInstanceV1::new(&record),
        &record,
        FixtureScenarioV1::Cancel,
    )
    .expect("a cancelled batch replays");

    assert_eq!(
        calls,
        vec![
            CallV1::Execute,
            CallV1::Cancel,
            CallV1::Execute,
            CallV1::Cancel,
            CallV1::Destroy,
        ],
        "a cancelled run leaves nothing pending before it destroys"
    );
    assert_eq!(executions, 2);
    assert_eq!(rows, 2);
}

/// A replay that differs says the cancel kept something, and is refused.
#[test]
fn a_replay_that_differs_from_the_cancelled_batch_is_refused() {
    let record = RecordV1::new();

    let refusal = driven(
        FakeInstanceV1::drifting(&record),
        &record,
        FixtureScenarioV1::Cancel,
    )
    .expect_err("a drifted replay must be refused");

    assert!(
        refusal.contains("did not roll that execution back"),
        "{refusal}"
    );
    // The refusal closed the execution it had open rather than leaving it.
    assert_eq!(
        record.calls(),
        vec![
            CallV1::Execute,
            CallV1::Cancel,
            CallV1::Execute,
            CallV1::Cancel,
            CallV1::Destroy,
        ]
    );
    assert_eq!(record.live.get(), 0);
}

/// An instance dropped mid-execution is cancelled and released by the wrapper.
#[test]
fn the_drop_scenario_leaves_an_execution_open_and_the_wrapper_releases_it() {
    let record = RecordV1::new();

    let (calls, executions, rows) = driven(
        FakeInstanceV1::new(&record),
        &record,
        FixtureScenarioV1::Drop,
    )
    .expect("a dropped instance releases what it held");

    assert_eq!(
        calls,
        vec![CallV1::Execute, CallV1::Cancel, CallV1::Destroy],
        "the host finalizes nothing here: dropping the instance does it"
    );
    assert_eq!(executions, 1);
    assert_eq!(rows, 1);
}

/// A release that succeeded and still kept a resource is not a clean run.
#[test]
fn a_scenario_that_leaves_a_host_resource_behind_is_refused() {
    for scenario in [
        FixtureScenarioV1::Commit,
        FixtureScenarioV1::Cancel,
        FixtureScenarioV1::Drop,
    ] {
        let record = RecordV1::new();

        let refusal = driven(FakeInstanceV1::leaking(&record), &record, scenario)
            .expect_err("a leaked resource must be refused");

        assert!(
            refusal.contains("1 host resource") && refusal.contains(scenario.as_str()),
            "{refusal}"
        );
    }
}

/// Each scenario names the status its receipt reports, and no two share one.
#[test]
fn every_scenario_names_its_own_status() {
    let named = [
        (FixtureScenarioV1::Commit, "commit", "committed"),
        (FixtureScenarioV1::Cancel, "cancel", "cancelled"),
        (FixtureScenarioV1::Drop, "drop", "dropped"),
    ];

    for (scenario, name, status) in named {
        assert_eq!(scenario.as_str(), name);
        assert_eq!(scenario.status(), status);
    }
    let statuses = named.map(|(_, _, status)| status);
    assert_eq!(
        statuses
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        named.len(),
        "a receipt's status must say which scenario produced it"
    );
}
