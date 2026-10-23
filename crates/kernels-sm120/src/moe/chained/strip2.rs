//! The absorbing down small queue with each item covering two adjacent
//! 64-column strips (`chained_strip2.cuh`): one row and activation staging
//! per two strips, as the gate queue's halves share theirs. Every strip's
//! chain and partial is the absorbing item's, so the merge and the
//! finishers are unchanged.
use super::{
    absorb,
    floor::{Entry, Stage},
};
use crate::moe::SM_COUNT;
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal};

/// The queue this family replaces.
pub const STAGE: Stage = Stage::DownSmall;

pub fn symbol(e: &Entry) -> String {
    absorb::symbol(e, STAGE).replace("_moe_absorb_", "_moe_strip2_")
}

/// Half the absorbing queue's items (one per routed expert bound, strip
/// pair and 128-K block), at most eight CTAs per SM.
pub fn plan(e: &Entry, rows: u32) -> Result<LaunchPlan, Refusal> {
    let mut plan = absorb::plan(e, STAGE, rows)?;
    let g = e.down.query.projection.gemm;
    let set = g.grouped.expect("routed contract");
    if e.down.columns() % 128 != 0 {
        return Err(Refusal("two-strip items need whole strip pairs"));
    }
    let items = set.experts.min(rows * set.top_k) * (e.down.columns() / 128) * g.k.div_ceil(128);
    plan.grid[0] = items.min(SM_COUNT * absorb::RESIDENT);
    Ok(plan)
}

/// The absorbing down queue's ABI and operands.
pub fn entry(e: &Entry) -> EntrySpec {
    assert!(plan(e, 32).is_ok());
    let mut s = absorb::entry(e, STAGE);
    assert_eq!(s.body.matches("chop::moe::absorb::compute<").count(), 1);
    s.symbol = symbol(e);
    s.body = s
        .body
        .replace("chop::moe::absorb::compute<", "chop::moe::strip2::compute<");
    s
}

/// The two-strip down queue, once per expert bank.
pub fn cubin_spec(name: &str, entries: &[Entry]) -> CubinSpec {
    super::spec(
        name.into(),
        "chained_strip2.cuh",
        entries.iter().map(entry).collect(),
    )
}
