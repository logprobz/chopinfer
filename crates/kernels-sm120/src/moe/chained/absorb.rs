//! Small queues that also run the wide queue's experts
//! (`chained_absorb.cuh`), so no wide launch follows them. A wide item
//! keeps its weight strip in registers and runs the 8-row fragment over
//! each 8 rows of the expert: every chain and partial is the streamed
//! queues', so the merge and the finishers are unchanged.
//!
//! The gate queue also has a build reading its bank in 64-row by 128-K
//! tiles ([`ROW_TILES_64X128`]); each lane loads the same bytes, so its
//! chains and partials are the row-major queue's.
//!
//! [`ROW_TILES_64X128`]: kernel_core::precision::layout::ROW_TILES_64X128
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

/// The gate queue reading its bank in row tiles (`row-tiles-64x128-groups`).
pub fn tiled_symbol(e: &Entry) -> String {
    symbol(e, Stage::GateSmall).replace("_absorb_", "_absorb_tiles_")
}

/// The gate queue's ABI and launch with its bank in row tiles:
/// the same operands, read at the tiled addresses.
pub fn tiled_entry(e: &Entry) -> Result<EntrySpec, Refusal> {
    let gate = &e.gate.query.projection.gemm;
    if gate.k % 128 != 0 || e.gate.columns() % 64 != 0 {
        return Err(Refusal("row tiles need whole 64-row by 128-K tiles"));
    }
    let mut s = entry(e, Stage::GateSmall);
    assert_eq!(s.body.matches("chop::moe::absorb::compute<").count(), 1);
    assert_eq!(s.body.matches(">(input,").count(), 1);
    s.symbol = tiled_symbol(e);
    s.body = s.body.replace(">(input,", ",true>(input,");
    Ok(s)
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

/// [`cubin_spec`] and, after its entries, the tiled gate queue of each
/// bank in `tiled`.
pub fn tiled_cubin_spec(
    name: &str,
    entries: &[Entry],
    tiled: &[Entry],
) -> Result<CubinSpec, Refusal> {
    let mut spec = cubin_spec(name, entries);
    for e in tiled {
        spec.entries.push(tiled_entry(e)?);
    }
    Ok(spec)
}
