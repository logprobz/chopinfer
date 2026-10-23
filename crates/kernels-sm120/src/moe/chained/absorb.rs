//! Small queues that also run the wide queue's experts
//! (`chained_absorb.cuh`), so no wide launch follows them. A wide item
//! keeps its weight strip in registers and runs the 8-row fragment over
//! each 8 rows of the expert: every chain and partial is the streamed
//! queues', so the merge and the finishers are unchanged.
use super::{
    bandwidth::{self, Loads},
    floor::{Entry, Stage},
};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal};

/// The small queues this family replaces; the wide queues it removes.
pub const STAGES: [Stage; 2] = [Stage::GateSmall, Stage::DownSmall];
pub const REMOVED: [Stage; 2] = [Stage::GateWide, Stage::DownWide];

/// Eight CTAs per SM, the served 32-row gate's (resident8) and down's.
pub const RESIDENT: u32 = 8;

fn small(stage: Stage) -> Result<(), Refusal> {
    if STAGES.contains(&stage) {
        Ok(())
    } else {
        Err(Refusal(
            "the absorbing queues replace only the small queues",
        ))
    }
}

pub fn symbol(e: &Entry, stage: Stage) -> String {
    bandwidth::symbol(e, stage, Loads::Direct).replace("_bandwidth_direct_", "_absorb_")
}

/// The served small queue's launch: the item bound counts every routed
/// expert, wide ones included.
pub fn plan(e: &Entry, stage: Stage, rows: u32) -> Result<LaunchPlan, Refusal> {
    small(stage)?;
    bandwidth::plan(e, stage, rows)
}

/// The direct small queue's ABI and operands at eight CTAs per SM.
pub fn entry(e: &Entry, stage: Stage) -> EntrySpec {
    assert!(plan(e, stage, 32).is_ok());
    let mut s = bandwidth::entry(e, stage, Loads::Direct);
    assert_eq!(s.body.matches("chop::moe::streamed::compute<").count(), 1);
    assert_eq!(s.body.matches(",8>(").count(), 1);
    s.symbol = symbol(e, stage);
    s.body = s
        .body
        .replace(
            "chop::moe::streamed::compute<",
            "chop::moe::absorb::compute<",
        )
        .replace(",8>(", ">(");
    s.min_blocks = RESIDENT;
    s
}

/// Both absorbing queues, once per expert bank.
pub fn cubin_spec(name: &str, entries: &[Entry]) -> CubinSpec {
    super::spec(
        name.into(),
        "chained_absorb.cuh",
        entries
            .iter()
            .flat_map(|e| STAGES.into_iter().map(move |s| entry(e, s)))
            .collect(),
    )
}
