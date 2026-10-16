//! The end of a step: the order in which `commit`, `commit_accepted`,
//! `cancel` and `retire_request` run over a bundle's [`Ledger`].
//!
//! A bundle implements [`Steps`] for its instance and calls [`commit`],
//! [`commit_accepted`], [`cancel`] and [`retire`] from its
//! [`crate::entry::Bundle`] methods. The transaction owns the order; the
//! bundle's hooks own every side effect and the status of every failure.
//!
//! A commit (or a commit of accepted rows):
//!
//! 1. refuses a plain commit of a step with verify items
//!    ([`Failure::VerifyItems`]): those commit only through their kept rows;
//! 2. runs [`Steps::begin`];
//! 3. refuses when no step is open ([`Failure::NotOpen`]);
//! 4. plans the commit of the kept rows, `(request_id, kept_rows)` per
//!    record in record order ([`Ledger::plan_commit`], [`Stage::Plan`]);
//! 5. runs [`Steps::prepare`] with the plan, before the ledger changes;
//! 6. applies the plan ([`Ledger::apply_commit`], [`Stage::Apply`]);
//! 7. runs [`Steps::ended`] with how each item ended (the rejected tails a
//!    ring restores are [`crate::ring::tails`] of them);
//! 8. returns [`Steps::finish_taps`] with `commit` true.
//!
//! A cancel runs [`Steps::begin`], refuses when no step is open, runs
//! [`Steps::restore`] with the open step, cancels it ([`Ledger::cancel`],
//! [`Stage::Cancel`]), runs [`Steps::ended`] and returns
//! [`Steps::finish_taps`] with `commit` false.
//!
//! A retire refuses under [`Retire::DuringNoStep`] while any step is open
//! ([`Failure::StepOpen`]), runs [`Steps::before_retire`], retires the
//! request ([`Ledger::retire`], [`Stage::Retire`]) and runs
//! [`Steps::retired`].
//!
//! The first hook or ledger call that fails ends the call with its status,
//! and nothing after it runs. A ledger refusal changes nothing; a hook that
//! fails after changing device or host state leaves it for the ABI client,
//! which poisons the instance.

use runtime_bundle_abi::{BundleKeptRowsV1, BundleStatusV1};

use crate::ledger::{CommitPlan, Committed, Ledger, LedgerError, Step};

/// Which call ends the step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum End {
    /// `commit`: every item keeps all of its rows.
    Commit,
    /// `commit_accepted`: verify items keep the rows their records name.
    CommitAccepted,
    Cancel,
}

impl End {
    /// Whether the step's rows are kept.
    pub fn commits(self) -> bool {
        self != Self::Cancel
    }
}

/// The ledger call that refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stage {
    Plan,
    Apply,
    Cancel,
    Retire,
}

/// A refusal the bundle maps to its status ([`Steps::status`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Failure {
    /// A plain commit of a step with verify items.
    VerifyItems,
    /// A commit or cancel with no step open.
    NotOpen,
    /// A retire while a step is open, under [`Retire::DuringNoStep`].
    StepOpen,
    /// The ledger refused at `Stage`.
    Ledger(Stage, LedgerError),
}

/// When a retire may run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Retire {
    /// Unless the open step holds the request (the ledger's own rule).
    UnlessHeld,
    /// Only while no step is open.
    DuringNoStep,
}

/// How the step ended, for [`Steps::ended`].
#[derive(Clone, Copy, Debug)]
pub enum Ended<'a> {
    /// Each item's kept rows, in item order.
    Committed(&'a [Committed]),
    /// The cancelled step.
    Cancelled(&'a Step),
}

/// A bundle's side of the step transaction. Every hook but
/// [`Steps::ledger`] and [`Steps::status`] does nothing by default.
pub trait Steps {
    /// When a retire may run.
    const RETIRE: Retire;

    fn ledger(&mut self) -> &mut Ledger;

    /// The status a refusal returns.
    fn status(&self, failure: Failure) -> BundleStatusV1;

    /// Runs first, before the ledger is read for an open step: a bundle's
    /// own precondition, or work that runs whether or not a step is open.
    fn begin(&mut self, end: End) -> Result<(), BundleStatusV1> {
        let _ = end;
        Ok(())
    }

    /// A commit's effects before the ledger applies `plan`: `kept` is the
    /// requested rows per verify item (empty for a plain commit).
    fn prepare(&mut self, kept: &[(u64, u32)], plan: &CommitPlan) -> Result<(), BundleStatusV1> {
        let _ = (kept, plan);
        Ok(())
    }

    /// A cancel's effects before the ledger cancels `step`.
    fn restore(&mut self, step: &Step) -> Result<(), BundleStatusV1> {
        let _ = step;
        Ok(())
    }

    /// Runs once the ledger has ended the step.
    fn ended(&mut self, ended: Ended<'_>) -> Result<(), BundleStatusV1> {
        let _ = ended;
        Ok(())
    }

    /// Publishes (`commit`) or removes the step's captures; the call's
    /// status.
    fn finish_taps(&mut self, commit: bool) -> BundleStatusV1 {
        let _ = commit;
        BundleStatusV1::Success
    }

    /// Retires model-owned per-request state before the ledger retires
    /// `request`.
    fn before_retire(&mut self, request: u64) -> Result<(), BundleStatusV1> {
        let _ = request;
        Ok(())
    }

    /// Runs once the ledger has retired `request`.
    fn retired(&mut self, request: u64) {
        let _ = request;
    }
}

/// Commits the open step, every item keeping all of its rows.
pub fn commit<M: Steps>(model: &mut M) -> BundleStatusV1 {
    if model
        .ledger()
        .step()
        .is_some_and(|step| step.items.iter().any(|item| item.verify))
    {
        return model.status(Failure::VerifyItems);
    }
    finish_commit(model, End::Commit, &[])
}

/// Commits the open step, each verify item keeping the rows its record
/// names.
pub fn commit_accepted<M: Steps>(model: &mut M, records: &[BundleKeptRowsV1]) -> BundleStatusV1 {
    let kept = records
        .iter()
        .map(|record| (record.request_id, record.kept_rows))
        .collect::<Vec<_>>();
    finish_commit(model, End::CommitAccepted, &kept)
}

fn finish_commit<M: Steps>(model: &mut M, end: End, kept: &[(u64, u32)]) -> BundleStatusV1 {
    if let Err(status) = model.begin(end) {
        return status;
    }
    if model.ledger().step().is_none() {
        return model.status(Failure::NotOpen);
    }
    let plan = match model.ledger().plan_commit(kept) {
        Ok(plan) => plan,
        Err(error) => return model.status(Failure::Ledger(Stage::Plan, error)),
    };
    if let Err(status) = model.prepare(kept, &plan) {
        return status;
    }
    let committed = match model.ledger().apply_commit(plan) {
        Ok(committed) => committed,
        Err(error) => return model.status(Failure::Ledger(Stage::Apply, error)),
    };
    if let Err(status) = model.ended(Ended::Committed(&committed)) {
        return status;
    }
    model.finish_taps(true)
}

/// Cancels the open step.
pub fn cancel<M: Steps>(model: &mut M) -> BundleStatusV1 {
    if let Err(status) = model.begin(End::Cancel) {
        return status;
    }
    let Some(step) = model.ledger().step().cloned() else {
        return model.status(Failure::NotOpen);
    };
    if let Err(status) = model.restore(&step) {
        return status;
    }
    let step = match model.ledger().cancel() {
        Ok(step) => step,
        Err(error) => return model.status(Failure::Ledger(Stage::Cancel, error)),
    };
    if let Err(status) = model.ended(Ended::Cancelled(&step)) {
        return status;
    }
    model.finish_taps(false)
}

/// Retires `request`.
pub fn retire<M: Steps>(model: &mut M, request: u64) -> BundleStatusV1 {
    if M::RETIRE == Retire::DuringNoStep && model.ledger().step().is_some() {
        return model.status(Failure::StepOpen);
    }
    if let Err(status) = model.before_retire(request) {
        return status;
    }
    if let Err(error) = model.ledger().retire(request) {
        return model.status(Failure::Ledger(Stage::Retire, error));
    }
    model.retired(request);
    BundleStatusV1::Success
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::{Item, Placement, Rules};

    /// Records every hook call and fails the one it is told to.
    struct Recorder {
        ledger: Ledger,
        calls: Vec<String>,
        fail: Option<&'static str>,
    }

    impl Recorder {
        fn new() -> Self {
            Self {
                ledger: Ledger::new(
                    4,
                    Placement::Ledger,
                    Rules {
                        invalidate_on_cancel: false,
                        tail_window: None,
                    },
                    64,
                ),
                calls: Vec::new(),
                fail: None,
            }
        }

        /// Opens a step of a three-row item of request 7, verifying when
        /// `verify`.
        fn open(&mut self, verify: bool) {
            let item = Item {
                request: 7,
                slot: None,
                start: 0,
                rows: 3,
                verify,
            };
            self.ledger.begin(&[item], Vec::new()).unwrap();
        }

        fn call(&mut self, call: String, hook: &'static str) -> Result<(), BundleStatusV1> {
            self.calls.push(call);
            if self.fail == Some(hook) {
                return Err(BundleStatusV1::HostError);
            }
            Ok(())
        }
    }

    impl Steps for Recorder {
        const RETIRE: Retire = Retire::UnlessHeld;

        fn ledger(&mut self) -> &mut Ledger {
            &mut self.ledger
        }

        // Passes each refusal through as a distinct status the test reads
        // back: the transaction never picks a status itself.
        fn status(&self, failure: Failure) -> BundleStatusV1 {
            match failure {
                Failure::VerifyItems => BundleStatusV1::Unsupported,
                Failure::NotOpen => BundleStatusV1::InvalidState,
                Failure::StepOpen => BundleStatusV1::BundleError,
                Failure::Ledger(Stage::Plan, _) => BundleStatusV1::InvalidArgument,
                Failure::Ledger(Stage::Apply | Stage::Cancel, _) => BundleStatusV1::InternalError,
                Failure::Ledger(Stage::Retire, _) => BundleStatusV1::NotFound,
            }
        }

        fn begin(&mut self, end: End) -> Result<(), BundleStatusV1> {
            let open = self.ledger.step().is_some();
            self.call(format!("begin {end:?} open={open}"), "begin")
        }

        fn prepare(
            &mut self,
            kept: &[(u64, u32)],
            plan: &CommitPlan,
        ) -> Result<(), BundleStatusV1> {
            let open = self.ledger.step().is_some();
            let planned = plan
                .items()
                .iter()
                .map(|item| item.kept)
                .collect::<Vec<_>>();
            self.call(
                format!("prepare {kept:?} {planned:?} open={open}"),
                "prepare",
            )
        }

        fn restore(&mut self, step: &Step) -> Result<(), BundleStatusV1> {
            let open = self.ledger.step().is_some();
            let call = format!("restore {} open={open}", step.items.len());
            self.call(call, "restore")
        }

        fn ended(&mut self, ended: Ended<'_>) -> Result<(), BundleStatusV1> {
            let open = self.ledger.step().is_some();
            let call = match ended {
                Ended::Committed(items) => {
                    let tails = crate::ring::tails(items)
                        .iter()
                        .map(|tail| (tail.start, tail.count))
                        .collect::<Vec<_>>();
                    format!("committed tails {tails:?} open={open}")
                }
                Ended::Cancelled(step) => format!("cancelled {} open={open}", step.items.len()),
            };
            self.call(call, "ended")
        }

        fn finish_taps(&mut self, commit: bool) -> BundleStatusV1 {
            match self.call(format!("taps {commit}"), "taps") {
                Ok(()) => BundleStatusV1::Success,
                Err(status) => status,
            }
        }

        fn before_retire(&mut self, request: u64) -> Result<(), BundleStatusV1> {
            self.call(format!("before_retire {request}"), "before_retire")
        }

        fn retired(&mut self, request: u64) {
            self.calls.push(format!("retired {request}"));
        }
    }

    /// The same hooks, refusing retires during any open step.
    struct Strict(Recorder);

    impl Steps for Strict {
        const RETIRE: Retire = Retire::DuringNoStep;

        fn ledger(&mut self) -> &mut Ledger {
            self.0.ledger()
        }

        fn status(&self, failure: Failure) -> BundleStatusV1 {
            self.0.status(failure)
        }

        fn before_retire(&mut self, request: u64) -> Result<(), BundleStatusV1> {
            self.0.before_retire(request)
        }

        fn retired(&mut self, request: u64) {
            self.0.retired(request);
        }
    }

    fn kept(request_id: u64, kept_rows: u32) -> BundleKeptRowsV1 {
        BundleKeptRowsV1 {
            struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
            reserved: 0,
            request_id,
            kept_rows,
            reserved_tail: 0,
        }
    }

    #[test]
    fn a_commit_runs_its_hooks_around_the_ledger_in_order() {
        let mut model = Recorder::new();
        model.open(false);
        assert_eq!(commit(&mut model), BundleStatusV1::Success);
        assert_eq!(
            model.calls,
            [
                "begin Commit open=true",
                "prepare [] [3] open=true",
                "committed tails [] open=false",
                "taps true",
            ]
        );
        assert_eq!(model.ledger.holder(0), Some((7, 3)));
    }

    #[test]
    fn an_accepted_commit_maps_its_records_and_ends_with_the_rejected_tails() {
        let mut model = Recorder::new();
        model.open(true);
        assert_eq!(
            commit_accepted(&mut model, &[kept(7, 1)]),
            BundleStatusV1::Success
        );
        assert_eq!(
            model.calls,
            [
                "begin CommitAccepted open=true",
                "prepare [(7, 1)] [1] open=true",
                "committed tails [(1, 2)] open=false",
                "taps true",
            ]
        );
        assert_eq!(model.ledger.holder(0), Some((7, 1)));
    }

    #[test]
    fn a_cancel_restores_before_the_ledger_cancels() {
        let mut model = Recorder::new();
        model.open(true);
        assert_eq!(cancel(&mut model), BundleStatusV1::Success);
        assert_eq!(
            model.calls,
            [
                "begin Cancel open=true",
                "restore 1 open=true",
                "cancelled 1 open=false",
                "taps false",
            ]
        );
        assert!(model.ledger.step().is_none());
        assert_eq!(model.ledger.holder(0), None);
    }

    #[test]
    fn a_commit_of_verify_items_is_refused_before_any_hook() {
        let mut model = Recorder::new();
        model.open(true);
        assert_eq!(commit(&mut model), BundleStatusV1::Unsupported);
        assert!(model.calls.is_empty());
        assert!(model.ledger.step().is_some());
        // The same step commits through its kept rows.
        assert_eq!(
            commit_accepted(&mut model, &[kept(7, 3)]),
            BundleStatusV1::Success
        );
    }

    #[test]
    fn no_open_step_is_refused_after_begin() {
        let mut model = Recorder::new();
        assert_eq!(commit(&mut model), BundleStatusV1::InvalidState);
        assert_eq!(
            commit_accepted(&mut model, &[]),
            BundleStatusV1::InvalidState
        );
        assert_eq!(cancel(&mut model), BundleStatusV1::InvalidState);
        assert_eq!(
            model.calls,
            [
                "begin Commit open=false",
                "begin CommitAccepted open=false",
                "begin Cancel open=false",
            ]
        );
    }

    #[test]
    fn a_planning_refusal_passes_through_and_changes_nothing() {
        let mut model = Recorder::new();
        model.open(true);
        // Request 8 is not the step's verify item.
        assert_eq!(
            commit_accepted(&mut model, &[kept(8, 1)]),
            BundleStatusV1::InvalidArgument
        );
        assert_eq!(model.calls, ["begin CommitAccepted open=true"]);
        assert!(model.ledger.step().is_some());
    }

    #[test]
    fn a_failed_hook_returns_its_status_and_stops_the_call() {
        for (hook, expected) in [
            ("begin", &["begin Commit open=true"][..]),
            (
                "prepare",
                &["begin Commit open=true", "prepare [] [3] open=true"][..],
            ),
            (
                "ended",
                &[
                    "begin Commit open=true",
                    "prepare [] [3] open=true",
                    "committed tails [] open=false",
                ][..],
            ),
        ] {
            let mut model = Recorder::new();
            model.open(false);
            model.fail = Some(hook);
            assert_eq!(commit(&mut model), BundleStatusV1::HostError, "{hook}");
            assert_eq!(model.calls, expected, "{hook}");
            // A failure before the apply leaves the step open.
            assert_eq!(model.ledger.step().is_some(), hook != "ended", "{hook}");
        }
        let mut model = Recorder::new();
        model.open(false);
        model.fail = Some("restore");
        assert_eq!(cancel(&mut model), BundleStatusV1::HostError);
        assert_eq!(
            model.calls,
            ["begin Cancel open=true", "restore 1 open=true"]
        );
        assert!(model.ledger.step().is_some());
        let mut model = Recorder::new();
        model.open(false);
        model.fail = Some("taps");
        assert_eq!(cancel(&mut model), BundleStatusV1::HostError);
        assert_eq!(model.calls.last().unwrap(), "taps false");
    }

    #[test]
    fn a_retire_unless_held_waits_only_for_the_step_that_holds_it() {
        let mut model = Recorder::new();
        model.open(false);
        assert_eq!(retire(&mut model, 7), BundleStatusV1::NotFound);
        assert_eq!(retire(&mut model, 9), BundleStatusV1::Success);
        assert_eq!(
            model.calls,
            ["before_retire 7", "before_retire 9", "retired 9"]
        );
        assert_eq!(commit(&mut model), BundleStatusV1::Success);
        model.calls.clear();
        assert_eq!(retire(&mut model, 7), BundleStatusV1::Success);
        assert_eq!(model.calls, ["before_retire 7", "retired 7"]);
        assert_eq!(model.ledger.holder(0), None);
    }

    #[test]
    fn a_retire_during_no_step_refuses_any_open_step_before_its_hooks() {
        let mut model = Strict(Recorder::new());
        model.0.open(false);
        assert_eq!(retire(&mut model, 9), BundleStatusV1::BundleError);
        assert_eq!(retire(&mut model, 7), BundleStatusV1::BundleError);
        assert!(model.0.calls.is_empty());
        model.0.ledger.cancel().unwrap();
        assert_eq!(retire(&mut model, 9), BundleStatusV1::Success);
        assert_eq!(model.0.calls, ["before_retire 9", "retired 9"]);
    }

    #[test]
    fn a_failed_retire_hook_keeps_the_request() {
        let mut model = Recorder::new();
        model.open(false);
        assert_eq!(commit(&mut model), BundleStatusV1::Success);
        model.calls.clear();
        model.fail = Some("before_retire");
        assert_eq!(retire(&mut model, 7), BundleStatusV1::HostError);
        assert_eq!(model.calls, ["before_retire 7"]);
        assert_eq!(model.ledger.holder(0), Some((7, 3)));
    }
}
