//! Opt-in helpers for a model's own SM120 kernels (its `OwnOps`
//! implementation): the prepared op that launches what a model's planner
//! builds from the step's buffers, and the checks and argument plumbing
//! every model's own ops repeat.
//!
//! A model states its own kernels once ([`OwnKernels`]: the prefix of its
//! implementation names, which symbols its cubins hold, what its
//! declarations claim for a symbol, and how a launch is checked) and keeps
//! each op's validation against its compiled constants, its grid, threads
//! and argument order as its own code, in the planner it hands
//! [`OwnKernels::prepare`] or [`OwnKernels::prepare_one`].

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal, Validate};
use kernel_core::properties::Invariance;

use crate::invariance::Declared;

/// A model's own SM120 kernels as its bundle compiles them.
#[derive(Clone, Copy)]
pub struct OwnKernels {
    /// The prefix of the ops' implementation names: an op `name` is
    /// `{prefix}.{name}.v1`.
    pub prefix: &'static str,
    /// Whether the bundle's cubins hold a symbol.
    pub embedded: fn(&str) -> bool,
    /// The invariance a declaration claims for a symbol, `Invariance::NONE`
    /// for one without a declaration.
    pub declared: fn(&str) -> Invariance,
    /// Checks each launch of an op ([`plan`], [`geometry`] or [`listed`]).
    pub check: fn(&OpInfo, &Launch) -> Result<(), Refusal>,
}

impl OwnKernels {
    /// The info of the op `name` launching `symbols` (each with the most
    /// dynamic shared memory its launches plan): every symbol embedded, a
    /// capacity, and the guarantees the symbols' declarations all claim
    /// (none when they differ). The caller adds what else the op reports
    /// (workspace, weight layouts) before preparing it.
    pub fn info(
        &self,
        name: &str,
        symbols: Vec<(String, u32)>,
        need: Need,
    ) -> Result<OpInfo, Refused> {
        for (symbol, _) in &symbols {
            if !(self.embedded)(symbol) {
                return Err(rule("the bundle embeds no entry for this own op"));
            }
        }
        if need.capacity == 0 {
            return Err(rule("an own op needs a capacity"));
        }
        let guarantees = symbols
            .iter()
            .map(|(symbol, _)| (self.declared)(symbol))
            .reduce(|all, one| if all == one { all } else { Invariance::NONE })
            .unwrap_or(Invariance::NONE);
        Ok(OpInfo::new(
            format!("{}.{name}.v1", self.prefix),
            guarantees,
            symbols,
        ))
    }

    /// Prepares the op of `info` after checking `need`: per step it checks
    /// that the workspace covers the info's, builds its launches with
    /// `planner` and checks each.
    pub fn prepare<B: 'static>(
        &self,
        info: OpInfo,
        need: Need,
        planner: impl Fn(&B, View, Counts) -> Result<Vec<Launch>, Refusal> + Send + Sync + 'static,
    ) -> Result<Prepared<B>, Refused> {
        need.check(&info)?;
        Ok(Box::new(Own {
            info,
            check: self.check,
            planner: Box::new(planner),
        }))
    }

    /// [`Self::prepare`] for an op of one launch per step.
    pub fn prepare_one<B: 'static>(
        &self,
        info: OpInfo,
        need: Need,
        planner: impl Fn(&B, Counts) -> Result<Launch, Refusal> + Send + Sync + 'static,
    ) -> Result<Prepared<B>, Refused> {
        self.prepare(info, need, move |buffers: &B, _, counts| {
            Ok(vec![planner(buffers, counts)?])
        })
    }
}

type Planner<B> = dyn Fn(&B, View, Counts) -> Result<Vec<Launch>, Refusal> + Send + Sync;

/// A prepared own op: the launches a planner builds from the step's
/// buffers.
struct Own<B> {
    info: OpInfo,
    check: fn(&OpInfo, &Launch) -> Result<(), Refusal>,
    planner: Box<Planner<B>>,
}

impl<B> PreparedOp<B> for Own<B> {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &B,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        workspace.covers(self.info.workspace)?;
        let launches = (self.planner)(buffers, workspace, counts)?;
        for launch in &launches {
            (self.check)(&self.info, launch)?;
        }
        Ok(launches)
    }
}

/// A launch check: the plan within CUDA's geometry and SM120's dynamic
/// shared memory (`Validate`).
pub fn plan(_: &OpInfo, launch: &Launch) -> Result<(), Refusal> {
    launch.plan.validate()
}

/// A launch check: the plan's geometry only (`LaunchPlan::validate_geometry`).
pub fn geometry(_: &OpInfo, launch: &Launch) -> Result<(), Refusal> {
    launch.plan.validate_geometry()
}

/// A launch check: [`plan`], and a symbol the op lists, planning at most
/// the dynamic shared memory the op lists for it.
pub fn listed(info: &OpInfo, launch: &Launch) -> Result<(), Refusal> {
    launch.plan.validate()?;
    if !info.symbols.iter().any(|(symbol, bytes)| {
        *symbol == launch.symbol && launch.plan.dynamic_shared_bytes <= *bytes
    }) {
        return Err(Refusal(
            "an own launch plans more shared memory than its op lists",
        ));
    }
    Ok(())
}

/// The invariance `declared` claims for `symbol`, none when it has no
/// declaration: a declaration covers its own symbol only.
pub fn declared_in(declared: &[Declared], symbol: &str) -> Invariance {
    declared
        .iter()
        .find(|declared| declared.symbol == symbol)
        .map_or(Invariance::NONE, |declared| declared.invariance)
}

/// A refusal of a contract rule or a compiled limit.
pub fn rule(text: &str) -> Refused {
    Refused::Rule(text.into())
}

/// A device buffer argument of `view`.
pub fn buffer(view: View) -> Argument {
    Argument::DeviceBuffer(view.handle)
}

/// The rows of one launch: from one to the prepared capacity.
pub fn rows(counts: Counts, capacity: u32) -> Result<u32, Refusal> {
    if counts.rows == 0 || counts.rows > capacity {
        return Err(Refusal(
            "an own op launches from one row to its prepared capacity",
        ));
    }
    Ok(counts.rows)
}

/// `count` rows of `bytes` bytes each.
pub fn bytes(count: u32, bytes: u64) -> u64 {
    u64::from(count) * bytes
}

/// A launch of a bundle-owned entry with PDL and no shared memory,
/// workspace or cluster.
pub fn pdl_plan(grid: [u32; 3], threads: u32) -> LaunchPlan {
    LaunchPlan {
        grid,
        block: [threads, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    }
}

/// A launch of `symbol` that waits for no peer and leaves the SMs free.
pub fn launch(symbol: &str, rows: u32, plan: LaunchPlan, arguments: Vec<Argument>) -> Launch {
    Launch {
        symbol: symbol.into(),
        rows,
        plan,
        arguments: ArgumentList(arguments),
        collective: false,
        persistent: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedded(symbol: &str) -> bool {
        symbol.starts_with("own_")
    }

    fn declared(symbol: &str) -> Invariance {
        match symbol {
            "own_a" | "own_b" => Invariance::one_row(8),
            _ => Invariance::NONE,
        }
    }

    const OWN: OwnKernels = OwnKernels {
        prefix: "sm120.test",
        embedded,
        declared,
        check: listed,
    };

    #[test]
    fn info_refuses_unembedded_symbols_and_no_capacity_and_meets_declarations() {
        let symbols = |names: &[&str]| names.iter().map(|name| (name.to_string(), 0)).collect();
        assert!(matches!(
            OWN.info("x", symbols(&["other"]), Need::rows(4)),
            Err(Refused::Rule(_))
        ));
        assert!(matches!(
            OWN.info("x", symbols(&["own_a"]), Need::rows(0)),
            Err(Refused::Rule(_))
        ));
        let info = OWN
            .info("x", symbols(&["own_a", "own_b"]), Need::rows(4))
            .unwrap();
        assert_eq!(info.implementation, "sm120.test.x.v1");
        assert_eq!(info.guarantees, Invariance::one_row(8));
        let info = OWN
            .info("x", symbols(&["own_a", "own_c"]), Need::rows(4))
            .unwrap();
        assert_eq!(info.guarantees, Invariance::NONE);
    }

    #[test]
    fn launches_are_checked_against_the_listed_symbols() {
        let info = OWN
            .info("x", vec![("own_a".into(), 64)], Need::rows(4))
            .unwrap();
        let op = OWN
            .prepare_one(info, Need::rows(4), |&shared: &u32, counts| {
                let mut plan = pdl_plan([rows(counts, 4)?, 1, 1], 256);
                plan.dynamic_shared_bytes = shared;
                Ok(launch("own_a", counts.rows, plan, vec![]))
            })
            .unwrap();
        assert_eq!(
            op.launches(&64, View::EMPTY, Counts::rows(3))
                .unwrap()
                .len(),
            1
        );
        assert!(op.launches(&65, View::EMPTY, Counts::rows(3)).is_err());
        assert!(op.launches(&0, View::EMPTY, Counts::rows(5)).is_err());
        assert!(op.launches(&0, View::EMPTY, Counts::rows(0)).is_err());
    }
}
